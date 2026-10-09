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
#include <optional>
#include <random>
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
// launch, as in the C++ model.
class PlacementModel {
public:
  uint32_t place(const Cta& c) {
    if (c.launch != launch_) {
      launch_ = c.launch;
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
    return base;
  }

private:
  uint32_t launch_ = ~0u;
  uint32_t tail_ = 0;
};

class PlacementBench {
public:
  PlacementBench(Sim& sim, uint64_t tick, uint32_t seed)
    : sim_(sim), tick_(tick), rng_(seed), timer_(NUM_WARPS, 0) {}

  uint64_t tick() const {
    return tick_;
  }

  // Runs the CTAs in order, the host waiting for the device to drain between
  // launches; returns each CTA's slot.
  bool run(const std::vector<Cta>& ctas, std::vector<int>& slots) {
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
    size_t next = 0;
    int cur = -1;
    uint64_t idle = 0;
    while (next < ctas.size() || cur < 0 || this->busy()) {
      bool boundary = (next < ctas.size()) && (next == 0 || ctas[next].launch != ctas[next - 1].launch);
      bool present = next < ctas.size();
      if (boundary && this->busy()) {
        present = false;   // the next launch starts on an idle device
        drained_ = 0;
      } else if (boundary && drained_ < 4) {
        present = false;   // descriptor settles before valid, as DCRs do before start
        ++drained_;
      }
      this->drive(next < ctas.size() ? &ctas[next] : nullptr, present);
      sim_->eval();

      bool accept = sim_->task_in_valid && sim_->task_in_ready;
      bool fire = sim_->cta_fire;
      uint32_t wid = sim_->cta_wid;
      if (csr_read_ && !this->check_csrs(ctas[*csr_read_])) {
        return false;
      }
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
      }

      tick_ = sim_.step(tick_, 2);

      if (accept) {
        cur = static_cast<int>(next++);
        owed_ += (ctas[cur].block_size + NUM_THREADS - 1) / NUM_THREADS;
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
      idle = (accept || fire) ? 0 : idle + 1;
      if (idle > 2000) {
        std::printf("FAIL: no CTA accepted or warp launched for 2000 cycles (next CTA %zu of %zu)\n", next, ctas.size());
        return false;
      }
      if (next == ctas.size() && cur >= 0 && !this->busy() && !lookup_ && !csr_read_) {
        break;
      }
    }
    if (csr_checks_ == 0) {
      std::printf("FAIL: no CTA context was read back\n");
      return false;
    }
    return true;
  }

private:
  bool busy() const {
    return owed_ != 0 || active_ != 0 || !retire_q_.empty() || lookup_.has_value() || csr_read_.has_value();
  }

  // The CTA CSRs of the slot read last cycle must be the launching CTA's own.
  bool check_csrs(const Cta& c) {
    ++csr_checks_;
    bool ok = (sim_->rd_block_idx_x == c.block_idx) && (sim_->rd_cluster_size == c.k);
#if VX_CFG_EXT_DLMEM_ENABLED
    ok = ok && (sim_->rd_cluster_rank == c.rank);
    uint32_t rank = sim_->rd_cluster_rank;
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
    sim_->schedule_wid   = lookup_ ? lookup_->first : 0;
    // One retirement per cycle; the warp leaves active_warps as it retires.
    sim_->warp_done = 0;
    if (!retire_q_.empty()) {
      uint32_t w = retire_q_.front();
      retire_q_.pop_front();
      active_ &= ~(1u << w);
      sim_->warp_done     = 1;
      sim_->warp_done_wid = w;
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
};

static void add_launch(std::vector<Cta>& ctas, uint32_t launch, uint8_t ctx_id, uint32_t k,
                       uint32_t num_ctas, uint32_t lmem_size, uint32_t block_size) {
  for (uint32_t i = 0; i < num_ctas; ++i) {
    Cta c;
    c.launch     = launch;
    c.ctx_id     = ctx_id;
    c.k          = k;
    c.rank       = i % k;
    c.lmem_size  = lmem_size;
    c.block_size = block_size;
    c.block_idx  = i;
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
    if (c.rank == 0) {
      window = slots[i];
    }
    bool ok = (slots[i] == want);
    if (c.rank == 0 && (slots[i] % c.k) != 0) {
      ok = false;
    }
    if (c.k <= usable && slots[i] != window + static_cast<int>(c.rank)) {
      ok = false;
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
static bool multi_launch(Sim& sim, uint64_t& tick, uint32_t seed, uint32_t num_launches) {
  std::mt19937 rng(seed);
  const uint32_t lmem_choices[] = {0, 1, LMEM_SIZE / 8, LMEM_SIZE / 4, LMEM_SIZE / 3, LMEM_SIZE / 2, LMEM_SIZE};
  std::vector<Cta> ctas;
  uint8_t ctx_id = 0;
  for (uint32_t l = 0; l < num_launches; ++l) {
    ctx_id = static_cast<uint8_t>(ctx_id + 1 + rng() % 3);
    uint32_t k;
    do {
      k = 1 + rng() % NUM_WARPS;
    } while (k > 4 && (k & (k - 1)) != 0);
    uint32_t num_ctas = k * (1 + rng() % 5);
    uint32_t lmem_size = lmem_choices[rng() % 7];
    uint32_t block_size = 1 + rng() % std::min(2 * NUM_THREADS, NUM_WARPS * NUM_THREADS);
    add_launch(ctas, l, ctx_id, k, num_ctas, lmem_size, block_size);
  }
  PlacementBench bench(sim, tick, seed);
  std::vector<int> slots;
  bool ok = bench.run(ctas, slots);
  tick = bench.tick();
  ok = ok && check_slots("multi-launch", ctas, slots);
  std::printf("multi-launch: %u launches, %zu CTAs: %s\n", num_launches, ctas.size(), ok ? "ok" : "FAILED");
  return ok;
}

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

  tick = smoke(sim, tick);

  bool ok = two_launch(sim, tick);
  ok = multi_launch(sim, tick, seed, launches) && ok;

  if (!ok) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
