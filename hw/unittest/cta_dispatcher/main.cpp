// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vl_simulator.h"
#include "VVX_cta_dispatch_top.h"
#include <iostream>
#include "VX_config.h"
#include "VX_types.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

static uint64_t timestamp = 0;
static bool trace_enabled = false;
static uint64_t trace_start_time = TRACE_START_TIME;
static uint64_t trace_stop_time  = TRACE_STOP_TIME;

double sc_time_stamp() {
  return timestamp;
}

bool sim_trace_enabled() {
  if (timestamp >= trace_start_time
   && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable(bool enable) {
  trace_enabled = enable;
}

using Sim = vl_simulator<VVX_cta_dispatch_top>;

constexpr uint32_t NUM_WARPS   = VX_CFG_NUM_WARPS;
constexpr uint32_t NUM_THREADS = VX_CFG_NUM_THREADS;
constexpr uint32_t LMEM_SIZE   = 1u << VX_CFG_LMEM_LOG_SIZE;
constexpr uint32_t BLOCK_ALIGN = VX_CFG_MEM_BLOCK_SIZE;

#if VX_CFG_EXT_DLMEM_ENABLED
static constexpr uint32_t c_clog2(uint64_t x) {
  uint32_t n = 0;
  while ((uint64_t(1) << n) < x) {
    ++n;
  }
  return n;
}
static constexpr uint32_t c_up(uint32_t x) {
  return x ? x : 1;
}
constexpr uint32_t NCC     = c_up(VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE) * VX_CFG_SOCKET_SIZE;
constexpr uint32_t MAXC    = VX_CFG_DLMEM_MAX_CLUSTER;
constexpr uint32_t CORE_W  = c_up(c_clog2(NCC));
constexpr uint32_t NW_W    = c_up(c_clog2(NUM_WARPS));
constexpr uint32_t K_W     = c_clog2(MAXC + 1);
constexpr uint32_t PEER_W  = CORE_W + NW_W;
constexpr uint32_t ROW_W   = K_W + MAXC * PEER_W;
constexpr uint32_t GANGS   = std::max<uint32_t>(2, NCC * NUM_WARPS / 2);
constexpr uint32_t GANG_W  = c_clog2(GANGS);
constexpr uint32_t MY_CORE = TB_CORE_ID % NCC;

// Port access: Verilator picks the C type from the width.
template <typename T>
static typename std::enable_if<std::is_integral<T>::value, std::vector<uint32_t>>::type words_of(const T& v) {
  return {uint32_t(uint64_t(v)), uint32_t(uint64_t(v) >> 32)};
}
template <std::size_t N>
static std::vector<uint32_t> words_of(const VlWide<N>& v) {
  std::vector<uint32_t> w(N);
  for (std::size_t i = 0; i < N; ++i) {
    w[i] = v[i];
  }
  return w;
}
template <typename T>
static typename std::enable_if<std::is_integral<T>::value>::type set_words(T& v, const std::vector<uint32_t>& w) {
  uint64_t x = w.empty() ? 0 : w[0];
  if (w.size() > 1) {
    x |= uint64_t(w[1]) << 32;
  }
  v = T(x);
}
template <std::size_t N>
static void set_words(VlWide<N>& v, const std::vector<uint32_t>& w) {
  for (std::size_t i = 0; i < N; ++i) {
    v[i] = (i < w.size()) ? w[i] : 0;
  }
}

// A peer row as the RTL packs it: peers[r] = {core, slot} at r*PEER_W, k on top.
static std::vector<uint32_t> row_words(uint32_t k, const std::vector<std::pair<uint32_t, uint32_t>>& peers) {
  std::vector<uint32_t> w((ROW_W + 31) / 32 + 2, 0);
  auto put = [&](uint32_t lo, uint32_t width, uint64_t val) {
    for (uint32_t b = 0; b < width; ++b) {
      if ((val >> b) & 1) {
        w[(lo + b) / 32] |= 1u << ((lo + b) % 32);
      }
    }
  };
  for (uint32_t r = 0; r < peers.size() && r < MAXC; ++r) {
    put(r * PEER_W, NW_W, peers[r].second);
    put(r * PEER_W + NW_W, CORE_W, peers[r].first);
  }
  put(MAXC * PEER_W, K_W, k);
  return w;
}

// The row ports are ROW_W bits wide; compare only those.
static bool same_bits(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, uint32_t width) {
  for (uint32_t i = 0; i < width; ++i) {
    bool x = (i / 32 < a.size()) && ((a[i / 32] >> (i % 32)) & 1);
    bool y = (i / 32 < b.size()) && ((b[i / 32] >> (i % 32)) & 1);
    if (x != y) {
      return false;
    }
  }
  return true;
}

// dlmem_ref::legacy_row: a same-core cluster's members hold consecutive slots.
static std::vector<uint32_t> legacy_row_words(uint32_t base, uint32_t k) {
  uint32_t kk = std::min(k ? k : 1, MAXC);
  std::vector<std::pair<uint32_t, uint32_t>> peers;
  for (uint32_t r = 0; r < kk; ++r) {
    peers.push_back({MY_CORE, base + r});
  }
  return row_words(kk, peers);
}
#endif

// Single launch: one KMU request, warps retire on a timer.
static uint64_t smoke(Sim& sim, uint64_t tick) {
  tick = sim.reset(tick);

  // Drive a single KMU request: grid 2x2x1, block 4x1x1
  sim->task_in_valid  = 1;
  sim->in_PC          = 0x12345678;
  sim->in_cta_id      = 0;
  sim->in_block_idx_x = 0;
  sim->in_block_idx_y = 0;
  sim->in_block_idx_z = 0;
  sim->in_block_dim_x = 4;
  sim->in_block_dim_y = 1;
  sim->in_block_dim_z = 1;
  sim->in_grid_dim_x  = 2;
  sim->in_grid_dim_y  = 2;
  sim->in_grid_dim_z  = 1;
  sim->in_param       = 0x87654321;
  sim->in_lmem_size   = 0;
  sim->in_block_size  = 4;
  sim->in_warp_step_x = VX_CFG_NUM_THREADS;
  sim->in_warp_step_y = 0;
  sim->in_warp_step_z = 0;
  sim->in_ctx_id      = 0;
  sim->in_cluster_size = 1;
#if VX_CFG_EXT_DLMEM_ENABLED
  sim->in_cluster_rank = 0;
#endif
  sim->in_is_first    = 1;
  sim->csr_rd_wid     = 0;
  sim->csr_rd_cta_id  = 0;
  sim->active_warps   = 0;
  sim->warp_done      = 0;
  sim->warp_done_wid  = 0;
  sim->schedule_wid   = 0;

  // Per-warp countdown timers: expire after 4 cycles → warp slot freed
  std::vector<int> warp_timer(VX_CFG_NUM_WARPS, 0);

  for (int i = 0; i < 30; ++i) {
    tick = sim.step(tick, 2);

    // Retire after accepting one request
    if (sim->task_in_ready)
      sim->task_in_valid = 0;

    // A new warp was dispatched this cycle
    if (sim->cta_fire) {
      int w = sim->cta_wid;
      if (warp_timer[w] == 0) warp_timer[w] = 4;
    }

    // Advance timers; signal warp_done when a timer expires
    sim->warp_done = 0;
    uint32_t next_active_warps = 0;
    for (int w = 0; w < VX_CFG_NUM_WARPS; ++w) {
      if (warp_timer[w] > 0) {
        warp_timer[w]--;
        if (warp_timer[w] > 0) {
          next_active_warps |= (1u << w);
        } else {
          sim->warp_done     = 1;
          sim->warp_done_wid = w;
        }
      }
    }
    sim->active_warps = next_active_warps;
  }
  return tick;
}

///////////////////////////////////////////////////////////////////////////////
// Multi-launch slot placement, checked against a reference placement model.

struct Cta {
  uint32_t launch     = 0;
  uint8_t  ctx_id     = 0;
  uint32_t k          = 1; // cluster size
  uint32_t rank       = 0; // index within its cluster
  uint32_t lmem_size  = 0;
  uint32_t block_size = NUM_THREADS;
  uint32_t block_idx  = 0;
  // DLMEM: a SPREAD block, pinned to the slot its gang unit reserved. The bench
  // plays the gang: it forces a slot that is free and has no done event pending.
  bool     pinned     = false;
  bool     eager      = false; // presented without waiting for the last launch's done events
  int      want_slot  = -1;    // pinned: force this slot instead of a random free one
  uint32_t forced     = 0;
  uint32_t gang       = 0;
  uint32_t epoch      = 0;
  std::vector<uint32_t> row;
};

static uint32_t stride_of(uint32_t lmem_size) {
  return (lmem_size + BLOCK_ALIGN - 1) & ~(BLOCK_ALIGN - 1);
}

static uint32_t usable_slots(uint32_t stride) {
  if (stride == 0) {
    return NUM_WARPS;
  }
  uint32_t fit = LMEM_SIZE / stride;
  if (fit > NUM_WARPS) {
    fit = NUM_WARPS;
  }
  return fit ? fit : 1;
}

// Reference slot choice: the round-robin pointer restarts at 0 on every
// launch, as in the C++ model. A pinned block takes its forced slot and leaves
// the pointer alone.
class PlacementModel {
public:
  uint32_t place(const Cta& c) {
    bool is_new = (c.launch != launch_);
    launch_ = c.launch;
    if (c.pinned) {
      return c.forced;
    }
    if (is_new) {
      tail_ = 0;
    }
    uint32_t max_slots = usable_slots(stride_of(c.lmem_size));
    if (tail_ >= max_slots) {
      tail_ = 0;
    }
    uint32_t base = tail_;
    if (c.rank == 0 && base + c.k > max_slots) {
      base = 0;
    }
    tail_ = (base + 1) % max_slots;
    if (c.rank == 0) {
      window_ = base;
    }
    return base;
  }

  // the slot of the last cluster's first member
  uint32_t window() const {
    return window_;
  }

private:
  uint32_t launch_ = ~0u;
  uint32_t tail_ = 0;
  uint32_t window_ = 0;
};

class PlacementBench {
public:
  PlacementBench(Sim& sim, uint64_t tick, uint32_t seed)
    : sim_(sim), tick_(tick), rng_(seed), timer_(NUM_WARPS, 0) {}

  uint64_t tick() const {
    return tick_;
  }

  // CTA_DONE ready, per cycle (DLMEM): the barrier unit's back-pressure.
  uint32_t done_ready_pct = 100;
  uint64_t done_hold_from = 0, done_hold_to = 0;   // ready forced low in [from, to)

  // Results of the last run (DLMEM).
  uint32_t events = 0;
  uint64_t pend_blocked = 0;   // cycles a presented block waited on a pending done event
  uint32_t cbar_records = 0;   // cluster-barrier records checked
  uint32_t cbar_gangs = 0;     // of them, gangs that complete through the gang unit
  std::vector<uint64_t> accept_cycle;
  std::map<uint32_t, uint64_t> last_event_cycle;   // slot -> cycle its last event left

  // Runs the CTAs in order, the host waiting for the device to drain between
  // launches; returns each CTA's slot.
  bool run(std::vector<Cta>& ctas, std::vector<int>& slots) {
    tick_ = sim_.reset(tick_);
    slots.assign(ctas.size(), -1);
    timer_.assign(NUM_WARPS, 0);
    active_ = 0;
    retire_q_.clear();
    lookup_.reset();
    csr_read_.reset();
    csr_checks_ = 0;
    owed_ = 0;
    drained_ = 0;
    events = 0;
    pend_blocked = 0;
    cbar_records = 0;
    cbar_gangs = 0;
    accept_cycle.assign(ctas.size(), 0);
    last_event_cycle.clear();
    warp_cta_.assign(NUM_WARPS, -1);
    left_.assign(ctas.size(), 0);
    slot_of_.assign(ctas.size(), -1);
    pinned_.assign(ctas.size(), false);
    for (size_t i = 0; i < ctas.size(); ++i) {
      pinned_[i] = ctas[i].pinned;
    }
    taken_.clear();
    pending_.clear();
    release_q_.clear();
    events_of_.clear();
    model_ = PlacementModel();
    pinned_ready_ = nullptr;
    cycle_ = 0;
    prev_accept_ = -1;
    size_t next = 0;
    int cur = -1;
    uint64_t idle = 0;
    while (next < ctas.size() || cur < 0 || this->busy(false)) {
      bool boundary = (next < ctas.size()) && (next == 0 || ctas[next].launch != ctas[next - 1].launch);
      bool present = next < ctas.size();
      if (boundary && this->busy(next < ctas.size() && ctas[next].eager)) {
        present = false;   // the next launch starts on an idle device
        drained_ = 0;
      } else if (boundary && drained_ < 4) {
        present = false;   // descriptor settles before valid, as DCRs do before start
        ++drained_;
      }
      while (!release_q_.empty() && release_q_.front().first <= cycle_) {
        taken_.erase(release_q_.front().second);
        release_q_.pop_front();
      }
      if (present && !this->pin(ctas[next])) {
        present = false;   // no slot to force yet
      }
      this->drive(next < ctas.size() ? &ctas[next] : nullptr, present);
      sim_->eval();

      bool accept = sim_->task_in_valid && sim_->task_in_ready;
      bool fire = sim_->cta_fire;
      uint32_t wid = sim_->cta_wid;
      if (csr_read_ && !this->check_csrs(ctas[*csr_read_])) {
        return false;
      }
#if VX_CFG_EXT_DLMEM_ENABLED
      if (!this->check_dlmem(ctas, present ? int(next) : -1, accept)) {
        return false;
      }
#endif
      csr_read_.reset();
      if (lookup_) {
        int slot = sim_->schedule_cta_id;
        int& s = slots.at(lookup_->second);
        if (s < 0) {
          s = slot;
        } else if (s != slot) {
          std::printf("FAIL: CTA %d has warps in slots %d and %d\n", lookup_->second, s, slot);
          return false;
        }
        // The slot's context was written as the warp launched; read it back.
        sim_->csr_rd_wid    = lookup_->first;
        sim_->csr_rd_cta_id = slot;
        sim_->eval();
        csr_read_ = lookup_->second;
        lookup_.reset();
      }
      if (fire) {
        if (cur < 0 || ((active_ >> wid) & 1)) {
          std::printf("FAIL: warp %u launched while %s\n", wid, cur < 0 ? "no CTA was accepted" : "still active");
          return false;
        }
        lookup_ = std::make_pair(wid, cur);
        warp_cta_[wid] = cur;
      }

      tick_ = sim_.step(tick_, 2);
      ++cycle_;

      prev_accept_ = -1;
      if (accept) {
        cur = static_cast<int>(next++);
        owed_ += (ctas[cur].block_size + NUM_THREADS - 1) / NUM_THREADS;
        left_[cur] = (ctas[cur].block_size + NUM_THREADS - 1) / NUM_THREADS;
        accept_cycle[cur] = cycle_;
        prev_accept_ = cur;
        // the model holds every block admitted before this one
        PlacementModel tmp = model_;
        slot_of_[cur] = int(tmp.place(ctas[cur]));
        taken_[slot_of_[cur]] = cur;
        pinned_ready_ = nullptr;
        drained_ = 0;
      }
      if (fire) {
        --owed_;
        timer_[wid] = 2 + rng_() % 24;
        active_ |= 1u << wid;
      }
      for (uint32_t w = 0; w < NUM_WARPS; ++w) {
        if (timer_[w] > 0 && --timer_[w] == 0) {
          retire_q_.push_back(w);
        }
      }
      idle = (accept || fire || event_fire_) ? 0 : idle + 1;
      if (idle > 2000) {
        std::printf("FAIL: no CTA accepted or warp launched for 2000 cycles (next CTA %zu of %zu)\n", next, ctas.size());
        return false;
      }
      if (next == ctas.size() && cur >= 0 && !this->busy(false) && !lookup_ && !csr_read_) {
        break;
      }
    }
    if (csr_checks_ == 0) {
      std::printf("FAIL: no CTA context was read back\n");
      return false;
    }
    for (size_t i = 0; i < ctas.size(); ++i) {
      if (ctas[i].pinned && events_of_[int(i)] != 1) {
        std::printf("FAIL: pinned CTA %zu reported %u done events\n", i, events_of_[i]);
        return false;
      }
    }
    return true;
  }

private:
  // `eager`: ignore done events still owed by the last launch.
  bool busy(bool eager) const {
    return owed_ != 0 || active_ != 0 || !retire_q_.empty() || lookup_.has_value() || csr_read_.has_value()
        || (!eager && !taken_.empty());
  }

  // Plays the gang unit for a pinned CTA: force a slot that holds no block and
  // owes no done event. Returns false when there is none yet.
  bool pin(Cta& c) {
    if (!c.pinned) {
      return true;
    }
    if (pinned_ready_ == &c) {
      return true;
    }
    uint32_t usable = usable_slots(stride_of(c.lmem_size));
    std::vector<uint32_t> free;
    for (uint32_t s = 0; s < usable; ++s) {
      if (taken_.count(s) == 0 && (c.want_slot < 0 || uint32_t(c.want_slot) == s)) {
        free.push_back(s);
      }
    }
    if (free.empty()) {
      return false;
    }
    c.forced = free[rng_() % free.size()];
#if VX_CFG_EXT_DLMEM_ENABLED
    c.gang  = rng_() % GANGS;
    c.epoch = rng_() & 1;
    std::vector<std::pair<uint32_t, uint32_t>> peers;
    uint32_t k = std::min(c.k, MAXC);
    for (uint32_t r = 0; r < k; ++r) {
      peers.push_back({uint32_t(rng_() % NCC), uint32_t(rng_() % NUM_WARPS)});
    }
    c.row = row_words(k, peers);
#endif
    pinned_ready_ = &c;
    return true;
  }

#if VX_CFG_EXT_DLMEM_ENABLED
  // Peer-row writes, CTA_DONE events and the CTA_DONE back-pressure.
  bool check_dlmem(const std::vector<Cta>& ctas, int presented, bool accept) {
    // a peer row for every admitted block, one cycle after it was taken
    bool want_wr = (prev_accept_ >= 0);
    if (bool(sim_->peer_wr_valid) != want_wr) {
      std::printf("FAIL: peer-row write %d, %s\n", int(sim_->peer_wr_valid), want_wr ? "a block was admitted" : "no block was admitted");
      return false;
    }
    uint32_t wr_slot = 0;
    if (want_wr) {
      const Cta& c = ctas[prev_accept_];
      uint32_t slot = model_.place(c);
      wr_slot = slot;
      std::vector<uint32_t> row = c.pinned ? c.row
          : legacy_row_words(model_.window(), std::min(c.k, usable_slots(stride_of(c.lmem_size))));
      std::vector<uint32_t> got = words_of(sim_->peer_wr_row);
      if (sim_->peer_wr_slot != slot || sim_->peer_wr_stride != stride_of(c.lmem_size) || !same_bits(got, row, ROW_W)) {
        std::printf("FAIL: launch %u CTA %u (%s, K=%u): peer row slot %u stride %u, want slot %u stride %u%s\n",
                    c.launch + 1, c.block_idx, c.pinned ? "pinned" : "legacy", c.k, (uint32_t)sim_->peer_wr_slot,
                    (uint32_t)sim_->peer_wr_stride, slot, stride_of(c.lmem_size),
                    same_bits(got, row, ROW_W) ? "" : ", row differs");
        return false;
      }
    }
    // and its cluster-barrier record: a gang of K > 1 completes through the gang
    // unit, any other block in this core at its cluster's base slot
    if (bool(sim_->cbar_wr_valid) != want_wr) {
      std::printf("FAIL: cluster-barrier record write %d, %s\n", int(sim_->cbar_wr_valid),
                  want_wr ? "a block was admitted" : "no block was admitted");
      return false;
    }
    if (want_wr) {
      const Cta& c = ctas[prev_accept_];
      uint32_t slot = wr_slot;
      uint32_t k = c.pinned ? std::min(c.k, MAXC) : std::min(c.k, usable_slots(stride_of(c.lmem_size)));
      bool gang = c.pinned && k > 1;
      uint32_t base = c.pinned ? slot : model_.window();
      bool ok = (sim_->cbar_wr_slot == slot) && (bool(sim_->cbar_wr_gang) == gang) && (sim_->cbar_wr_k == k)
             && (sim_->cbar_wr_base == base);
      if (c.pinned) {
        ok = ok && (sim_->cbar_wr_gang_id == c.gang) && (sim_->cbar_wr_epoch == c.epoch);
      }
      if (!ok) {
        std::printf("FAIL: launch %u CTA %u (%s, K=%u): cluster-barrier record slot %u gang %d id %u epoch %u base %u"
                    " k %u, want slot %u gang %d id %u epoch %u base %u k %u\n", c.launch + 1, c.block_idx,
                    c.pinned ? "pinned" : "legacy", c.k, (uint32_t)sim_->cbar_wr_slot, int(sim_->cbar_wr_gang),
                    (uint32_t)sim_->cbar_wr_gang_id, (uint32_t)sim_->cbar_wr_epoch, (uint32_t)sim_->cbar_wr_base,
                    (uint32_t)sim_->cbar_wr_k, slot, int(gang), c.gang, c.epoch, base, k);
        return false;
      }
      ++cbar_records;
      cbar_gangs += gang;
    }
    // a presented block whose slot still owes a done event must wait
    if (presented >= 0 && !ctas[presented].pinned) {
      const Cta& c = ctas[presented];
      PlacementModel tmp = model_;
      if (pending_.count(tmp.place(c)) != 0) {
        ++pend_blocked;
        if (accept) {
          std::printf("FAIL: launch %u CTA %u admitted while its slot's done event is pending\n", c.launch + 1, c.block_idx);
          return false;
        }
      }
    }
    // the events themselves
    event_fire_ = false;
    if (sim_->cta_done_valid) {
      if (!sim_->busy_out) {
        std::printf("FAIL: dispatcher idle with a done event pending\n");
        return false;
      }
      if (sim_->cta_done_ready) {
        uint32_t slot = sim_->cta_done_slot;
        auto it = pending_.find(slot);
        if (it == pending_.end()) {
          std::printf("FAIL: done event for slot %u, which has no finished pinned CTA\n", slot);
          return false;
        }
        const Cta& c = ctas[it->second];
        if (sim_->cta_done_core != MY_CORE || sim_->cta_done_gang != c.gang || sim_->cta_done_epoch != c.epoch) {
          std::printf("FAIL: done event for slot %u: core %u gang %u epoch %u, want core %u gang %u epoch %u\n", slot,
                      (uint32_t)sim_->cta_done_core, (uint32_t)sim_->cta_done_gang, (uint32_t)sim_->cta_done_epoch,
                      MY_CORE, c.gang, c.epoch);
          return false;
        }
        ++events_of_[it->second];
        ++events;
        last_event_cycle[slot] = cycle_;
        pending_.erase(it);
        taken_.erase(slot);
        event_fire_ = true;
      }
    }
    return true;
  }
#endif

  // The CTA CSRs of the slot read last cycle must be the launching CTA's own.
  bool check_csrs(const Cta& c) {
    ++csr_checks_;
    bool ok = (sim_->rd_block_idx_x == c.block_idx) && (sim_->rd_cluster_size == c.k);
#if VX_CFG_EXT_DLMEM_ENABLED
    ok = ok && (sim_->rd_cluster_rank == c.rank);
    uint32_t rank = sim_->rd_cluster_rank;
    // a block's local memory is its slot's: LMEM_BASE + slot * stride
    uint64_t lmem = (uint64_t(VX_MEM_LMEM_BASE_ADDR) | (uint64_t(sim_->rd_cta_id) * stride_of(c.lmem_size)))
                  & ((VX_CFG_MEM_ADDR_WIDTH >= 64) ? ~0ull : ((1ull << VX_CFG_MEM_ADDR_WIDTH) - 1));
    if (uint64_t(sim_->rd_lmem_addr) != lmem || (c.pinned && sim_->rd_cta_id != c.forced)) {
      std::printf("FAIL: launch %u CTA %u%s in slot %u has lmem_addr 0x%lx, want 0x%lx\n", c.launch + 1, c.block_idx,
                  c.pinned ? " (pinned)" : "", (uint32_t)sim_->rd_cta_id, (unsigned long)sim_->rd_lmem_addr,
                  (unsigned long)lmem);
      ok = false;
    }
#else
    uint32_t rank = c.rank;
#endif
    if (!ok) {
      std::printf("FAIL: launch %u CTA %u (K=%u, rank %u) reads back block %u, cluster size %u, cluster rank %u from slot %u\n",
                  c.launch + 1, c.block_idx, c.k, c.rank, (uint32_t)sim_->rd_block_idx_x,
                  (uint32_t)sim_->rd_cluster_size, rank, (uint32_t)sim_->rd_cta_id);
    }
    return ok;
  }

  void drive(const Cta* c, bool valid) {
    Cta d = c ? *c : Cta{};
    sim_->task_in_valid  = valid;
    sim_->in_PC          = 0x1000;
    sim_->in_cta_id      = d.block_idx;
    sim_->in_block_idx_x = d.block_idx;
    sim_->in_block_idx_y = 0;
    sim_->in_block_idx_z = 0;
    sim_->in_block_dim_x = d.block_size;
    sim_->in_block_dim_y = 1;
    sim_->in_block_dim_z = 1;
    sim_->in_grid_dim_x  = 64;
    sim_->in_grid_dim_y  = 1;
    sim_->in_grid_dim_z  = 1;
    sim_->in_param       = 0x2000;
    sim_->in_lmem_size   = d.lmem_size;
    sim_->in_block_size  = d.block_size;
    sim_->in_warp_step_x = NUM_THREADS;
    sim_->in_warp_step_y = 0;
    sim_->in_warp_step_z = 0;
    sim_->in_ctx_id      = d.ctx_id;
    sim_->in_cluster_size = d.k;
#if VX_CFG_EXT_DLMEM_ENABLED
    sim_->in_cluster_rank = d.rank;
#endif
    sim_->in_is_first    = (d.rank == 0);
#if VX_CFG_EXT_DLMEM_ENABLED
    sim_->in_policy      = d.pinned;
    sim_->in_pin         = d.pinned;
    sim_->in_forced_slot = d.forced;
    sim_->in_gang_id     = d.gang;
    sim_->in_epoch       = d.epoch;
    set_words(sim_->in_row, d.row);
    bool hold = (cycle_ >= done_hold_from && cycle_ < done_hold_to);
    sim_->cta_done_ready = !hold && (done_ready_pct >= 100 || (rng_() % 100) < done_ready_pct);
#endif
    sim_->schedule_wid   = lookup_ ? lookup_->first : 0;
    // One retirement per cycle; the warp leaves active_warps as it retires.
    sim_->warp_done = 0;
    if (!retire_q_.empty()) {
      uint32_t w = retire_q_.front();
      retire_q_.pop_front();
      active_ &= ~(1u << w);
      sim_->warp_done     = 1;
      sim_->warp_done_wid = w;
      // The CTA's last warp: a pinned block now owes its done event. A legacy
      // block's slot frees in the dispatcher two cycles later, so the bench's
      // gang reuses it only after that.
      int c = warp_cta_[w];
      if (c >= 0 && left_[c] > 0 && --left_[c] == 0) {
        if (pinned_cta(c)) {
          pending_[uint32_t(slot_of_[c])] = c;
        } else {
          release_q_.push_back({cycle_ + 4, uint32_t(slot_of_[c])});
        }
      }
    }
    sim_->active_warps = active_;
  }

  Sim& sim_;
  uint64_t tick_;
  std::mt19937 rng_;
  std::vector<uint32_t> timer_;
  uint32_t active_ = 0;
  std::deque<uint32_t> retire_q_;
  std::optional<std::pair<uint32_t, int>> lookup_;
  std::optional<int> csr_read_;
  uint32_t csr_checks_ = 0;
  uint32_t owed_ = 0; // warps of accepted CTAs not launched yet
  uint32_t drained_ = 0;
  uint64_t cycle_ = 0;
  int      prev_accept_ = -1;
  bool     event_fire_ = false;
  const Cta* pinned_ready_ = nullptr;
  std::vector<int> warp_cta_;           // wid -> CTA
  std::vector<uint32_t> left_;          // warps still running, per CTA
  std::vector<int> slot_of_;            // CTA -> slot
  std::vector<bool> pinned_;            // CTA -> pinned
  std::map<uint32_t, int> taken_;       // slot -> resident CTA (a pinned one until its done event leaves)
  std::map<uint32_t, int> pending_;     // slot -> finished pinned CTA owing its event
  std::deque<std::pair<uint64_t, uint32_t>> release_q_;  // (cycle, slot) of finished legacy CTAs
  std::map<int, uint32_t> events_of_;   // pinned CTA -> events seen
  PlacementModel model_;                // every admitted block, advanced at its peer-row write

  bool pinned_cta(int c) const {
    return c >= 0 && size_t(c) < pinned_.size() && pinned_[c];
  }
};

static void add_launch(std::vector<Cta>& ctas, uint32_t launch, uint8_t ctx_id, uint32_t k,
                       uint32_t num_ctas, uint32_t lmem_size, uint32_t block_size, bool pinned = false) {
  for (uint32_t i = 0; i < num_ctas; ++i) {
    Cta c;
    c.launch     = launch;
    c.ctx_id     = ctx_id;
    c.k          = k;
    c.rank       = i % k;
    c.lmem_size  = lmem_size;
    c.block_size = block_size;
    c.block_idx  = i;
    c.pinned     = pinned;
    ctas.push_back(c);
  }
}

// Every CTA must land where the reference model puts it; with the pointer
// restarting per launch, a cluster's window starts at a multiple of K and member
// r sits at window + r, which is what the kernel's rank (CTA_ID % K) assumes.
static bool check_slots(const char* name, const std::vector<Cta>& ctas, const std::vector<int>& slots) {
  PlacementModel model;
  uint32_t errors = 0;
  int window = 0;
  for (size_t i = 0; i < ctas.size(); ++i) {
    const Cta& c = ctas[i];
    int want = model.place(c);
    uint32_t usable = usable_slots(stride_of(c.lmem_size));
    bool ok = (slots[i] == want);
    if (!c.pinned) {
      if (c.rank == 0) {
        window = slots[i];
      }
      if (c.rank == 0 && (slots[i] % c.k) != 0) {
        ok = false;
      }
      if (c.k <= usable && slots[i] != window + static_cast<int>(c.rank)) {
        ok = false;
      }
    }
    if (!ok && ++errors <= 10) {
      std::printf("FAIL %s: launch %u (ctx %u, K=%u, usable %u) CTA %u rank %u landed in slot %d, the reference places it in slot %d\n",
                  name, c.launch + 1, c.ctx_id, c.k, usable, c.block_idx, c.rank, slots[i], want);
    }
  }
  return errors == 0;
}

// Launch 1 (K=1, one CTA) leaves the round-robin pointer at slot 1; launch 2
// (K=2) must still start its first cluster in slot 0.
static bool two_launch(Sim& sim, uint64_t& tick) {
  if (NUM_WARPS < 2) {
    std::printf("two-launch: needs at least 2 CTA slots, skipped\n");
    return true;
  }
  std::vector<Cta> ctas;
  add_launch(ctas, 0, 1, 1, 1, 0, NUM_THREADS);
  add_launch(ctas, 1, 2, 2, 2 * (NUM_WARPS / 2), 0, NUM_THREADS);
  PlacementBench bench(sim, tick, 1);
  std::vector<int> slots;
  bool ok = bench.run(ctas, slots);
  tick = bench.tick();
  std::printf("two-launch slots:");
  for (size_t i = 0; i < slots.size(); ++i) {
    std::printf(" %s%d", (i > 0 && ctas[i].launch != ctas[i - 1].launch) ? "| " : "", slots[i]);
  }
  std::printf("\n");
  return ok && check_slots("two-launch", ctas, slots);
}

// Random launches: cluster size, CTA count, LMEM footprint and block size
// vary per launch; ctx_id skips values for launches that sent this core none.
// With DLMEM, half the launches are SPREAD (every block pinned to a free slot)
// and the done events see back-pressure.
static bool multi_launch(Sim& sim, uint64_t& tick, uint32_t seed, uint32_t num_launches) {
  std::mt19937 rng(seed);
  const uint32_t lmem_choices[] = {0, 1, LMEM_SIZE / 8, LMEM_SIZE / 4, LMEM_SIZE / 3, LMEM_SIZE / 2, LMEM_SIZE};
  std::vector<Cta> ctas;
  uint8_t ctx_id = 0;
  for (uint32_t l = 0; l < num_launches; ++l) {
    ctx_id = static_cast<uint8_t>(ctx_id + 1 + rng() % 3);
    bool pinned = false;
#if VX_CFG_EXT_DLMEM_ENABLED
    pinned = (rng() % 2) != 0;
#endif
    uint32_t k;
    do {
      k = 1 + rng() % NUM_WARPS;
    } while (k > 4 && (k & (k - 1)) != 0);
#if VX_CFG_EXT_DLMEM_ENABLED
    if (pinned) {
      k = 1 + rng() % MAXC;
    }
#endif
    uint32_t num_ctas = k * (1 + rng() % 5);
    uint32_t lmem_size = lmem_choices[rng() % 7];
    uint32_t block_size = 1 + rng() % std::min(2 * NUM_THREADS, NUM_WARPS * NUM_THREADS);
    add_launch(ctas, l, ctx_id, k, num_ctas, lmem_size, block_size, pinned);
  }
  PlacementBench bench(sim, tick, seed);
#if VX_CFG_EXT_DLMEM_ENABLED
  bench.done_ready_pct = 40 + rng() % 61;
#endif
  std::vector<int> slots;
  bool ok = bench.run(ctas, slots);
  tick = bench.tick();
  ok = ok && check_slots("multi-launch", ctas, slots);
#if VX_CFG_EXT_DLMEM_ENABLED
  std::printf("multi-launch: %u launches, %zu CTAs, %u done events, %lu cycles a block waited on one,"
              " %u cluster-barrier records (%u gangs): %s\n", num_launches, ctas.size(), bench.events,
              (unsigned long)bench.pend_blocked, bench.cbar_records, bench.cbar_gangs, ok ? "ok" : "FAILED");
#else
  std::printf("multi-launch: %u launches, %zu CTAs: %s\n", num_launches, ctas.size(), ok ? "ok" : "FAILED");
#endif
  return ok;
}

#if VX_CFG_EXT_DLMEM_ENABLED
// Pinned blocks of one launch land in their forced slots, every slot of the
// core in turn, under done-event back-pressure; each reports exactly one done
// event after its last warp, carrying its gang id and epoch.
static bool pinned_launch(Sim& sim, uint64_t& tick) {
  std::vector<Cta> ctas;
  uint32_t stride = stride_of(LMEM_SIZE / NUM_WARPS);
  add_launch(ctas, 0, 1, std::min<uint32_t>(2, MAXC), 3 * NUM_WARPS, stride, NUM_THREADS + 1, true);
  PlacementBench bench(sim, tick, 7);
  bench.done_ready_pct = 30;
  std::vector<int> slots;
  bool ok = bench.run(ctas, slots);
  tick = bench.tick();
  ok = ok && check_slots("pinned", ctas, slots) && (bench.events == ctas.size());
  std::vector<bool> seen(NUM_WARPS, false);
  for (int s : slots) {
    if (s >= 0) {
      seen[s] = true;
    }
  }
  uint32_t used = std::count(seen.begin(), seen.end(), true);
  std::printf("pinned launch: %zu blocks in %u distinct slots, %u done events: %s\n", ctas.size(), used, bench.events,
              ok ? "ok" : "FAILED");
  return ok;
}

// A slot whose done event is held back stays taken: the next launch's first
// block, which the round robin sends to slot 0, waits until slot 0's event leaves.
static bool done_pend_blocks(Sim& sim, uint64_t& tick) {
  std::vector<Cta> ctas;
  add_launch(ctas, 0, 1, 1, 1, 0, NUM_THREADS, true);
  ctas[0].want_slot = 0;
  add_launch(ctas, 1, 2, 1, 1, 0, NUM_THREADS);
  ctas[1].eager = true;
  PlacementBench bench(sim, tick, 9);
  bench.done_hold_from = 0;
  bench.done_hold_to   = 300;
  std::vector<int> slots;
  bool ok = bench.run(ctas, slots);
  tick = bench.tick();
  ok = ok && check_slots("done-pend", ctas, slots);
  uint64_t ev = bench.last_event_cycle.count(0) ? bench.last_event_cycle.at(0) : ~0ull;
  if (ok && (slots[1] != 0 || bench.accept_cycle[1] <= ev || bench.pend_blocked < 100)) {
    std::printf("FAIL done-pend: next launch's block in slot %d taken in cycle %lu, slot 0's event left in %lu,"
                " the block waited %lu cycles on it\n", slots[1], (unsigned long)bench.accept_cycle[1],
                (unsigned long)ev, (unsigned long)bench.pend_blocked);
    ok = false;
  }
  std::printf("done-pend: event held until cycle %lu, next block taken in cycle %lu after waiting %lu cycles: %s\n",
              (unsigned long)ev, (unsigned long)bench.accept_cycle[1], (unsigned long)bench.pend_blocked,
              ok ? "ok" : "FAILED");
  return ok;
}

// Inside one launch a pinned block leaves the round-robin tail alone: legacy,
// pinned to the last slot, legacy lands in slots 0, NW-1, 1.
static bool pinned_keeps_tail(Sim& sim, uint64_t& tick) {
  if (NUM_WARPS < 3) {
    std::printf("pinned-tail: needs 3 CTA slots, skipped\n");
    return true;
  }
  std::vector<Cta> ctas;
  add_launch(ctas, 0, 1, 1, 3, 0, NUM_THREADS);
  ctas[1].pinned = true;
  ctas[1].want_slot = NUM_WARPS - 1;
  PlacementBench bench(sim, tick, 11);
  std::vector<int> slots;
  bool ok = bench.run(ctas, slots);
  tick = bench.tick();
  ok = ok && check_slots("pinned-tail", ctas, slots)
          && (slots[0] == 0) && (slots[1] == int(NUM_WARPS - 1)) && (slots[2] == 1);
  std::printf("pinned-tail: slots %d %d %d: %s\n", slots[0], slots[1], slots[2], ok ? "ok" : "FAILED");
  return ok;
}
#endif

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  uint32_t seed = 1;
  uint32_t launches = 40;
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '+') {
      continue;
    }
    if (std::strncmp(argv[i], "--seed=", 7) == 0) {
      seed = static_cast<uint32_t>(std::strtoul(argv[i] + 7, nullptr, 0));
    } else if (std::strncmp(argv[i], "--launches=", 11) == 0) {
      launches = static_cast<uint32_t>(std::strtoul(argv[i] + 11, nullptr, 0));
    } else {
      std::printf("usage: %s [--seed=N] [--launches=N]\n", argv[0]);
      return 2;
    }
  }

  Sim sim;
  uint64_t tick = 0;
  std::printf("cta_dispatcher: warps=%u threads=%u lmem=%u dlmem=%d seed=%u\n", NUM_WARPS, NUM_THREADS, LMEM_SIZE,
              VX_CFG_EXT_DLMEM_ENABLED, seed);
#if VX_CFG_EXT_DLMEM_ENABLED
  std::printf("cta_dispatcher: core %u of %u, max cluster %u\n", MY_CORE, NCC, MAXC);
#endif

  tick = smoke(sim, tick);

  bool ok = two_launch(sim, tick);
#if VX_CFG_EXT_DLMEM_ENABLED
  ok = pinned_launch(sim, tick) && ok;
  ok = done_pend_blocks(sim, tick) && ok;
  ok = pinned_keeps_tail(sim, tick) && ok;
#endif
  ok = multi_launch(sim, tick, seed, launches) && ok;

  if (!ok) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
