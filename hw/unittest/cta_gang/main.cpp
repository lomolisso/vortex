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

// Gang admission unit, checked cycle by cycle against a model built on the
// dlmem_ref placement functions: every cycle the unit's input ready, output
// valid and busy must match the model, and every output beat must match it
// field for field and payload bit for bit. Independently of the model, the
// bench tracks the blocks it holds resident (no slot used twice, no core over
// its warps, no gang id reused while live), and the directed scenarios check
// hand-derived placements.
//
// Deterministic schedule (--csv), the one the SimX model must reproduce:
//   launches, in order, ctx ids 1, 2, 3, 4:
//     1. SPREAD  K=MAX       W=1         stride 0              2*NCC*NW/MAX+2 gangs
//     2. DEFAULT K=min(2,NW) W=1         stride 0              2 messages
//     3. SPREAD  K=max(1,MAX/2) W=max(1,NW/2) stride LMEM/2   2*NCC+1 gangs
//     4. SPREAD  K=1         W=NW        stride LMEM           NCC+2 gangs
//   block_size = W*NT; the output is always ready; beats are presented back to
//   back; a launch's first beat is presented in the cycle after the first cycle
//   in which the previous launch has fully drained and busy is low.
//   A block leaves the unit's output in cycle t and its CTA_DONE is queued in
//   cycle t + 6 + ((gang_seq*7 + rank*5) % 23); queued events are presented one
//   per cycle in queue order. Cycle 0 is the first cycle after reset.
//   placement.csv: gang_seq,rank,core,slot,gang_id,epoch,commit_cycle, where
//   gang_seq counts SPREAD gangs from 0 across the run and commit_cycle is the
//   placement's last cycle (the gang's first beat leaves at commit_cycle + 2).

#include "vl_simulator.h"
#include "VVX_cta_gang_top.h"
#include "VX_config.h"
#include "dlmem_ref.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
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
  if (timestamp >= trace_start_time && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable(bool enable) {
  trace_enabled = enable;
}

using namespace vortex::dlmem_ref;
using Sim = vl_simulator<VVX_cta_gang_top>;

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

constexpr uint32_t SOCKET_SIZE = VX_CFG_SOCKET_SIZE;
constexpr uint32_t NUM_SOCKETS = c_up(VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE);
constexpr uint32_t NCC         = NUM_SOCKETS * SOCKET_SIZE;
constexpr uint32_t NW          = VX_CFG_NUM_WARPS;
constexpr uint32_t NT          = VX_CFG_NUM_THREADS;
constexpr uint32_t MAXC        = VX_CFG_DLMEM_MAX_CLUSTER;
constexpr uint32_t LMEM_LOG    = VX_CFG_LMEM_LOG_SIZE;
constexpr uint32_t LMEM_SIZE   = 1u << LMEM_LOG;
constexpr uint32_t BLOCK_ALIGN = VX_CFG_MEM_BLOCK_SIZE;
constexpr uint32_t NUM_GANGS   = std::max<uint32_t>(2, NCC * NW / 2);
constexpr uint32_t CORE_W      = c_up(c_clog2(NCC));
constexpr uint32_t NW_W        = c_up(c_clog2(NW));
constexpr uint32_t K_W         = c_clog2(MAXC + 1);
constexpr uint32_t PEER_W      = CORE_W + NW_W;
constexpr uint32_t DEST_W      = c_up(c_clog2(VX_CFG_NUM_CLUSTERS * VX_CFG_NUM_CORES));
constexpr uint32_t CLUSTER_ID  = TB_CLUSTER_ID;
constexpr uint32_t MAX_LEGACY  = std::max<uint32_t>(NW, MAXC);

static const Cfg CFG = {NCC, NW, NT, MAXC, LMEM_LOG, NUM_GANGS, SOCKET_SIZE, NUM_SOCKETS, 0, 0};

///////////////////////////////////////////////////////////////////////////////
// Port access: Verilator picks the C type from the width.

template <typename T>
static typename std::enable_if<std::is_integral<T>::value, uint64_t>::type
bits_of(const T& v, uint32_t lo, uint32_t w) {
  uint64_t x = (lo >= 64) ? 0 : (uint64_t(v) >> lo);
  return (w >= 64) ? x : (x & ((uint64_t(1) << w) - 1));
}

template <std::size_t N>
static uint64_t bits_of(const VlWide<N>& v, uint32_t lo, uint32_t w) {
  uint64_t r = 0;
  for (uint32_t i = 0; i < w; ++i) {
    uint32_t b = lo + i;
    if (b / 32 < N && ((v[b / 32] >> (b % 32)) & 1)) {
      r |= uint64_t(1) << i;
    }
  }
  return r;
}

template <std::size_t N>
static std::vector<uint32_t> words_of(const VlWide<N>& v) {
  std::vector<uint32_t> w(N);
  for (std::size_t i = 0; i < N; ++i) {
    w[i] = v[i];
  }
  return w;
}

template <std::size_t N>
static void set_words(VlWide<N>& v, const std::vector<uint32_t>& w, uint32_t width) {
  for (std::size_t i = 0; i < N; ++i) {
    uint32_t x = (i < w.size()) ? w[i] : 0;
    uint32_t lo = uint32_t(i) * 32;
    if (lo >= width) {
      x = 0;
    } else if (width - lo < 32) {
      x &= (1u << (width - lo)) - 1;
    }
    v[i] = x;
  }
}

///////////////////////////////////////////////////////////////////////////////

struct Launch {
  bool     spread;
  uint32_t k;
  uint32_t w;
  uint32_t stride;
  uint32_t gangs;   // messages (clusters)
};

struct Beat {
  uint32_t launch;
  uint8_t  ctx;
  bool     spread;
  uint32_t k;
  uint32_t rank;
  uint32_t block_size;
  uint32_t stride;
  int      gang_seq;   // SPREAD gangs only, else -1
  std::vector<uint32_t> fill;
};

// One beat inside the unit's output buffer, as the model predicts it.
struct Expect {
  bool     pin;
  bool     eop;
  bool     policy;
  uint32_t dest;
  uint32_t epoch;
  uint32_t gang;
  uint32_t forced;
  Row      row;
  std::vector<uint32_t> word;
  int      gang_seq;
  uint32_t rank;
  uint32_t w;
  uint64_t in_cycle;
  uint64_t commit_cycle;
};

struct DoneEvt {
  uint32_t core;
  uint32_t slot;
  uint32_t gang;
  uint32_t epoch;
};

static std::string row_str(const Row& r) {
  std::string s = "{k=" + std::to_string(r.k) + ":";
  for (const auto& p : r.peers) {
    s += " " + std::to_string(p.core) + "/" + std::to_string(p.slot);
  }
  return s + "}";
}

static bool same_row(const Row& a, const Row& b) {
  if (a.k != b.k || a.peers.size() != b.peers.size())
    return false;
  for (size_t i = 0; i < a.peers.size(); ++i) {
    if (a.peers[i].core != b.peers[i].core || a.peers[i].slot != b.peers[i].slot)
      return false;
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////
// Cycle model of VX_cta_gang (and of its 2-entry registered output buffer).
// Within a cycle the CTA_DONE event applies first, then the unit decides.

class GangModel {
public:
  enum Phase { IDLE, PLACE, EMIT };

  GangModel() {
    reset(CFG, m_);
    reset(CFG, g_);
  }

  struct Out {
    bool in_ready;
    bool out_valid;
    bool busy;
    const Expect* front;
  };

  // Registered outputs, valid before the cycle's inputs are applied.
  Out peek(bool in_valid) const {
    Out o;
    o.out_valid = vout_;
    o.front = buf_.empty() ? nullptr : &buf_.front();
    bool used = false;
    for (uint32_t c = 0; c < NCC; ++c) {
      used = used || (m_.free[c] != low_mask(NW));
    }
    bool live = false;
    for (bool l : g_.live) {
      live = live || l;
    }
    o.busy = in_valid || vout_ || (phase_ != IDLE) || used || live;
    o.in_ready = false;
    return o;
  }

  // Advances one cycle. Returns the input ready the unit must show.
  bool step(uint64_t cycle, const Beat* head, const std::vector<uint32_t>& head_word,
            const DoneEvt* done, bool out_ready, std::string& err) {
    if (done) {
      if (policy_) {
        release(m_, done->core, done->slot, w_);
        if (k_ > 1) {
          vortex::dlmem_ref::done(g_, done->gang, done->epoch);
        }
      }
    }

    bool buf_valid_in = false;
    bool in_ready = false;
    bool start = false;
    Expect push{};

    switch (phase_) {
    case IDLE:
      if (head) {
        bool is_new = (head->ctx != ctx_);
        if (is_new) {
          new_launch(CFG, m_, g_);
          failed_ = false;
        }
        if (!head->spread) {
          buf_valid_in = true;
          in_ready = vin_;
          if (in_ready) {
            push = this->legacy_beat(*head, head_word, cycle);
          }
        } else {
          uint32_t K = head->k;
          bool id_free = (K == 1);
          for (bool l : g_.live) {
            id_free = id_free || !l;
          }
          if (id_free && (!failed_ || dirty_)) {
            start = true;
            uint32_t W = warps_of(head->block_size, NT);
            uint32_t U = usable_slots(CFG, head->stride);
            Mirror s = m_;
            ok_ = place(CFG, s, K, W, U, members_);
            fail_step_ = K;
            if (!ok_) {
              for (uint32_t kk = 1; kk <= K; ++kk) {
                Mirror t = m_;
                std::vector<Peer> tmp;
                if (!place(CFG, t, kk, W, U, tmp)) {
                  fail_step_ = kk - 1;
                  break;
                }
              }
            }
            idx_ = 0;
          }
        }
        policy_ = head->spread;
        k_ = head->k;
        w_ = warps_of(head->block_size, NT);
        ctx_ = head->ctx;
      }
      break;

    case PLACE:
      if (!ok_ && idx_ == fail_step_) {
        failed_ = true;
        phase_n_ = IDLE;
      } else if (idx_ == k_ - 1) {
        for (const auto& p : members_) {
          m_.free[p.core] &= ~(uint64_t(1) << p.slot);
          m_.used[p.core] += w_;
        }
        m_.rr = (members_.back().core + 1) % NCC;
        if (k_ > 1) {
          int id = alloc(g_, k_);
          if (id < 0) {
            err = "model: no free gang id at commit";
            return false;
          }
          cur_gang_ = uint32_t(id);
          cur_epoch_ = g_.epoch[id];
        } else {
          cur_gang_ = 0;
          cur_epoch_ = 0;
        }
        failed_ = false;
        commit_cycle_ = cycle;
        idx_n_ = 0;
        phase_n_ = EMIT;
      } else {
        idx_n_ = idx_ + 1;
      }
      break;

    case EMIT:
      if (head) {
        buf_valid_in = true;
        in_ready = vin_;
        if (in_ready) {
          if (!head->spread || head->rank != idx_ || head->ctx != ctx_) {
            err = "model: beat out of step with the gang being emitted";
            return false;
          }
          push = this->spread_beat(*head, head_word, cycle);
          idx_n_ = idx_ + 1;
          if (idx_ + 1 == k_) {
            phase_n_ = IDLE;
            idx_n_ = 0;
          }
        }
      }
      break;
    }

    if (phase_ == IDLE) {
      phase_n_ = start ? PLACE : IDLE;
      idx_n_ = 0;
    }
    dirty_ = start ? false : (dirty_ || (done != nullptr));

    // output buffer (VX_stream_buffer with a registered output)
    bool flow_out = out_ready || !vout_;
    if (vout_ && out_ready) {
      buf_.pop_front();
    }
    if (buf_valid_in && vin_) {
      buf_.push_back(push);
    }
    bool vin_n = (buf_valid_in || flow_out) ? flow_out : vin_;
    bool vout_n = flow_out ? (buf_valid_in || !vin_) : vout_;
    vin_ = vin_n;
    vout_ = vout_n;
    if (vout_ != !buf_.empty()) {
      err = "model: output buffer state inconsistent";
      return false;
    }

    phase_ = phase_n_;
    idx_ = idx_n_;
    return in_ready;
  }

private:
  Expect legacy_beat(const Beat& b, const std::vector<uint32_t>& word, uint64_t cycle) const {
    Expect e{};
    e.pin = false;
    e.eop = (b.rank + 1 == b.k);
    e.policy = false;
    e.dest = 0;
    e.row.k = 0;
    e.word = word;
    e.gang_seq = -1;
    e.rank = b.rank;
    e.w = warps_of(b.block_size, NT);
    e.in_cycle = cycle;
    return e;
  }

  Expect spread_beat(const Beat& b, const std::vector<uint32_t>& word, uint64_t cycle) const {
    Expect e{};
    const Peer& p = members_.at(idx_);
    e.pin = true;
    e.eop = true;
    e.policy = true;
    e.dest = kmu_dest(CFG, CLUSTER_ID, p.core) & uint32_t(low_mask(DEST_W));
    e.epoch = cur_epoch_;
    e.gang = cur_gang_;
    e.forced = p.slot;
    e.row.k = k_;
    e.row.peers = members_;
    e.word = word;
    e.gang_seq = b.gang_seq;
    e.rank = b.rank;
    e.w = w_;
    e.in_cycle = cycle;
    e.commit_cycle = commit_cycle_;
    return e;
  }

  Mirror   m_;
  GangIds  g_;
  Phase    phase_ = IDLE, phase_n_ = IDLE;
  uint32_t idx_ = 0, idx_n_ = 0;
  uint32_t ctx_ = 0;
  bool     policy_ = false;
  uint32_t k_ = 1, w_ = 1;
  bool     failed_ = false, dirty_ = false;
  bool     ok_ = false;
  uint32_t fail_step_ = 0;
  std::vector<Peer> members_;
  uint32_t cur_gang_ = 0, cur_epoch_ = 0;
  uint64_t commit_cycle_ = 0;
  bool     vin_ = true, vout_ = false;
  std::deque<Expect> buf_;
};

///////////////////////////////////////////////////////////////////////////////

struct GangRec {
  int      gang_seq;
  uint32_t k;
  uint32_t gang_id;
  uint32_t epoch;
  std::vector<Peer> members;      // by rank, as delivered
  std::vector<uint64_t> out_cycle;
  std::vector<uint64_t> done_cycle;
  uint64_t commit_cycle;
};

struct Resident {
  int      gang_seq;
  uint32_t rank;
  uint32_t core, slot, gang, epoch, w;
  uint64_t done_at;
};

struct RunOpts {
  uint32_t ready_pct = 100;   // output ready probability
  uint32_t gap_pct   = 0;     // probability of holding a beat back for a cycle
  // a block's lifetime in cycles, from the cycle it leaves the unit's output
  std::function<uint64_t(int gang_seq, uint32_t rank, uint32_t core)> life;
};

class Bench {
public:
  Bench(Sim& sim, uint64_t& tick, uint32_t seed) : sim_(sim), tick_(tick), rng_(seed) {
    kmu_dataw_ = sim_->cfg_kmu_dataw;
  }

  std::vector<GangRec> gangs;
  std::vector<std::pair<uint64_t, uint64_t>> legacy_lat;   // (in fire, out fire)
  uint64_t cycles = 0;
  uint32_t errors = 0;

  bool run(const std::vector<Launch>& launches, const RunOpts& o) {
    sim_->in_valid = 0;
    sim_->out_ready = 0;
    sim_->done_valid = 0;
    tick_ = sim_.reset(tick_);
    GangModel model;
    std::vector<std::deque<Beat>> beats(launches.size());
    int gang_seq = 0;
    for (uint32_t l = 0; l < launches.size(); ++l) {
      const Launch& L = launches[l];
      uint32_t bs = L.w * NT;
      for (uint32_t gi = 0; gi < L.gangs; ++gi) {
        int seq = L.spread ? gang_seq++ : -1;
        for (uint32_t r = 0; r < L.k; ++r) {
          Beat b;
          b.launch = l;
          b.ctx = uint8_t(l + 1);
          b.spread = L.spread;
          b.k = L.k;
          b.rank = r;
          b.block_size = bs;
          b.stride = L.stride;
          b.gang_seq = seq;
          for (uint32_t i = 0; i < (kmu_dataw_ + 31) / 32; ++i) {
            b.fill.push_back(uint32_t(rng_()));
          }
          beats[l].push_back(b);
        }
      }
    }

    std::map<int, size_t> gang_index;
    std::vector<Resident> res;
    std::deque<DoneEvt> done_q;
    std::vector<uint64_t> occ(NCC, 0);
    std::vector<uint32_t> warps(NCC, 0);
    std::map<uint32_t, std::pair<int, uint32_t>> live_ids;   // id -> (gang_seq, members left)
    uint32_t cur_launch = 0;
    bool head_valid = false;
    Beat head;
    bool busy_prev = true;
    uint64_t idle = 0;
    cycles = 0;

    auto fail = [&](const char* fmt, auto... args) {
      if (++errors <= 10) {
        std::printf("FAIL cycle %lu: ", (unsigned long)cycles);
        std::printf(fmt, args...);
        std::printf("\n");
      }
    };

    while (true) {
      // the launch in progress has drained: present the next one
      bool launch_open = (cur_launch < launches.size()) && !beats[cur_launch].empty();
      if (!launch_open && !head_valid && cur_launch < launches.size()) {
        bool drained = res.empty() && done_q.empty() && !busy_prev;
        if (drained) {
          ++cur_launch;
          if (cur_launch >= launches.size())
            break;
          launch_open = !beats[cur_launch].empty();
        }
      }
      // CTA completions due this cycle, one presented per cycle
      for (auto it = res.begin(); it != res.end();) {
        if (it->done_at <= cycles) {
          done_q.push_back({it->core, it->slot, it->gang, it->epoch});
          uint32_t c = it->core;
          occ[c] &= ~(uint64_t(1) << it->slot);
          warps[c] -= it->w;
          if (k_of(it->gang_seq) > 1) {
            auto lit = live_ids.find(it->gang);
            if (lit != live_ids.end() && --lit->second.second == 0) {
              live_ids.erase(lit);
            }
          }
          gangs[gang_index[it->gang_seq]].done_cycle[it->rank] = cycles;
          it = res.erase(it);
        } else {
          ++it;
        }
      }
      DoneEvt dev{};
      bool dvalid = !done_q.empty();
      if (dvalid) {
        dev = done_q.front();
        done_q.pop_front();
      }
      sim_->done_valid   = dvalid;
      sim_->done_core    = dev.core;
      sim_->done_slot    = dev.slot;
      sim_->done_gang_id = dev.gang;
      sim_->done_epoch   = dev.epoch;

      if (!head_valid && cur_launch < launches.size() && !beats[cur_launch].empty()
       && (o.gap_pct == 0 || (rng_() % 100) >= o.gap_pct)) {
        head = beats[cur_launch].front();
        beats[cur_launch].pop_front();
        head_valid = true;
      }
      this->drive(head_valid ? &head : nullptr);
      bool out_ready = (o.ready_pct >= 100) || ((rng_() % 100) < o.ready_pct);
      sim_->out_ready = out_ready;
      sim_->eval();

      GangModel::Out exp = model.peek(head_valid);
      std::vector<uint32_t> in_word = words_of(sim_->in_word);

      if (bool(sim_->busy) != exp.busy) {
        fail("busy=%d, the model says %d", int(sim_->busy), int(exp.busy));
      }
      if (bool(sim_->out_valid) != exp.out_valid) {
        fail("out_valid=%d, the model says %d", int(sim_->out_valid), int(exp.out_valid));
      }
      bool out_fire = sim_->out_valid && out_ready;
      if (sim_->out_valid && exp.front) {
        this->check_out(*exp.front, fail);
      }
      if (out_fire && exp.front) {
        const Expect& e = *exp.front;
        if (e.pin) {
          // the bench's own view: the block is now resident
          uint32_t c = this->core_of_dest(sim_->out_dest);
          uint32_t s = sim_->out_forced_slot;
          if ((occ[c] >> s) & 1) {
            fail("slot %u of core %u handed out twice", s, c);
          }
          occ[c] |= uint64_t(1) << s;
          warps[c] += e.w;
          if (warps[c] > NW) {
            fail("core %u holds %u warps (of %u)", c, warps[c], NW);
          }
          if (e.row.k > 1) {
            auto lit = live_ids.find(e.gang);
            if (lit == live_ids.end()) {
              live_ids[e.gang] = {e.gang_seq, e.row.k};
            } else if (lit->second.first != e.gang_seq) {
              fail("gang id %u handed to gang %d while gang %d still holds it", e.gang, e.gang_seq,
                   lit->second.first);
            }
          }
          auto git = gang_index.find(e.gang_seq);
          if (git == gang_index.end()) {
            GangRec g;
            g.gang_seq = e.gang_seq;
            g.k = e.row.k;
            g.gang_id = e.gang;
            g.epoch = e.epoch;
            g.members.assign(e.row.k, Peer{0, 0});
            g.out_cycle.assign(e.row.k, 0);
            g.done_cycle.assign(e.row.k, 0);
            g.commit_cycle = e.commit_cycle;
            gang_index[e.gang_seq] = gangs.size();
            gangs.push_back(g);
            git = gang_index.find(e.gang_seq);
          }
          GangRec& g = gangs[git->second];
          g.members[e.rank] = {c, s};
          g.out_cycle[e.rank] = cycles;
          ks_[e.gang_seq] = e.row.k;
          res.push_back({e.gang_seq, e.rank, c, s, e.gang, e.epoch, e.w, cycles + o.life(e.gang_seq, e.rank, c)});
        } else {
          legacy_lat.push_back({e.in_cycle, cycles});
        }
      }

      std::string err;
      bool want_ready = model.step(cycles, head_valid ? &head : nullptr, in_word, dvalid ? &dev : nullptr,
                                   out_ready, err);
      if (!err.empty()) {
        fail("%s", err.c_str());
        return false;
      }
      if (head_valid && bool(sim_->in_ready) != want_ready) {
        fail("in_ready=%d, the model says %d (launch %u, gang %d, rank %u)", int(sim_->in_ready), int(want_ready),
             head.launch + 1, head.gang_seq, head.rank);
      }
      bool in_fire = head_valid && sim_->in_ready;
      if (in_fire) {
        head_valid = false;
      }
      busy_prev = sim_->busy;

      tick_ = sim_.step(tick_, 2);
      ++cycles;

      idle = (in_fire || out_fire || dvalid) ? 0 : idle + 1;
      if (idle > 5000) {
        fail("no progress for 5000 cycles (launch %u)", cur_launch + 1);
        return false;
      }
      if (errors > 10) {
        return false;
      }
    }
    if (!live_ids.empty()) {
      fail("%zu gang ids still live at the end", live_ids.size());
    }
    return errors == 0;
  }

private:
  uint32_t k_of(int gang_seq) const {
    auto it = ks_.find(gang_seq);
    return (it == ks_.end()) ? 1 : it->second;
  }

  // Inverse of kmu_dest for this cluster.
  uint32_t core_of_dest(uint32_t dest) const {
    uint32_t ss_w = c_clog2(SOCKET_SIZE);
    uint32_t ns_w = c_clog2(NUM_SOCKETS);
    uint32_t cis = dest & uint32_t(low_mask(ss_w));
    uint32_t soc = (dest >> ss_w) & uint32_t(low_mask(ns_w));
    return soc * SOCKET_SIZE + cis;
  }

  void drive(const Beat* b) {
    static const std::vector<uint32_t> zero;
    sim_->in_valid = (b != nullptr);
    Beat d = b ? *b : Beat{0, 0, false, 1, 0, 1, 0, -1, {}};
    set_words(sim_->in_fill, d.fill, kmu_dataw_);
    sim_->in_ctx_id       = d.ctx;
    sim_->in_policy       = d.spread;
    sim_->in_cluster_size = d.k;
    sim_->in_cluster_rank = d.rank;
    sim_->in_is_first     = (d.rank == 0);
    sim_->in_eop          = (d.rank + 1 == d.k);
    sim_->in_block_size   = d.block_size;
    sim_->in_lmem_stride  = d.stride;
  }

  template <typename F>
  void check_out(const Expect& e, F& fail) {
    std::vector<uint32_t> w = words_of(sim_->out_word);
    if (w != e.word) {
      fail("output payload differs from the input beat (gang %d rank %u)", e.gang_seq, e.rank);
    }
    Row row;
    row.k = uint32_t(bits_of(sim_->out_row, MAXC * PEER_W, K_W));
    for (uint32_t r = 0; r < MAXC; ++r) {
      uint64_t p = bits_of(sim_->out_row, r * PEER_W, PEER_W);
      if (r < row.k) {
        row.peers.push_back({uint32_t(p >> NW_W), uint32_t(p & low_mask(NW_W))});
      } else if (p != 0) {
        fail("row entry %u past k=%u is not zero", r, row.k);
      }
    }
    bool ok = (bool(sim_->out_pin) == e.pin) && (bool(sim_->out_eop) == e.eop)
           && (bool(sim_->out_kind) == false) && (uint32_t(sim_->out_dest) == e.dest)
           && (bool(sim_->out_policy) == e.policy);
    if (e.pin) {
      ok = ok && (uint32_t(sim_->out_epoch) == e.epoch) && (uint32_t(sim_->out_gang_id) == e.gang)
              && (uint32_t(sim_->out_forced_slot) == e.forced) && same_row(row, e.row);
    } else {
      ok = ok && (sim_->out_epoch == 0) && (sim_->out_gang_id == 0) && (sim_->out_forced_slot == 0) && (row.k == 0);
    }
    if (!ok) {
      Row want = e.row;
      fail("beat (gang %d rank %u): pin %d eop %d dest %u policy %d epoch %u gang %u slot %u row %s;"
           " model: pin %d eop %d dest %u policy %d epoch %u gang %u slot %u row %s",
           e.gang_seq, e.rank, int(sim_->out_pin), int(sim_->out_eop), uint32_t(sim_->out_dest),
           int(sim_->out_policy), uint32_t(sim_->out_epoch), uint32_t(sim_->out_gang_id),
           uint32_t(sim_->out_forced_slot), row_str(row).c_str(), int(e.pin), int(e.eop), e.dest, int(e.policy),
           e.epoch, e.gang, e.forced, row_str(want).c_str());
    }
  }

  Sim& sim_;
  uint64_t& tick_;
  std::mt19937 rng_;
  uint32_t kmu_dataw_ = 0;
  std::map<int, uint32_t> ks_;
};

///////////////////////////////////////////////////////////////////////////////

static uint32_t align_up(uint32_t x) {
  return (x + BLOCK_ALIGN - 1) & ~(BLOCK_ALIGN - 1);
}

static RunOpts fixed_life(uint64_t cycles) {
  RunOpts o;
  o.life = [cycles](int, uint32_t, uint32_t) { return cycles; };
  return o;
}

static bool report(const char* name, bool ok) {
  std::printf("%s: %s\n", name, ok ? "ok" : "FAILED");
  return ok;
}

// Legacy beats pass unchanged, one cycle through the unit.
static bool t_legacy(Sim& sim, uint64_t& tick) {
  Bench b(sim, tick, 11);
  std::vector<Launch> ls = {{false, 1, 1, 0, 6}, {false, MAX_LEGACY, 1, 0, 3}};
  RunOpts o = fixed_life(1);
  bool ok = b.run(ls, o);
  for (auto& l : b.legacy_lat) {
    if (l.second != l.first + 1) {
      std::printf("FAIL legacy: a beat taken in cycle %lu left in cycle %lu\n", (unsigned long)l.first,
                  (unsigned long)l.second);
      ok = false;
      break;
    }
  }
  ok = ok && (b.legacy_lat.size() == 6 + 3 * MAX_LEGACY) && b.gangs.empty();
  // and under output back-pressure
  Bench b2(sim, tick, 12);
  RunOpts o2 = fixed_life(1);
  o2.ready_pct = 40;
  o2.gap_pct = 30;
  ok = b2.run(ls, o2) && ok;
  return report("legacy pass-through", ok);
}

// An empty cluster: gang j member r lands on core (j*K + r) % NCC, slot (j*K + r) / NCC.
static bool t_spread_empty(Sim& sim, uint64_t& tick) {
  uint32_t K = MAXC;
  uint32_t G = std::min<uint32_t>(NW, 3) * NCC / K;
  Bench b(sim, tick, 21);
  bool ok = b.run({{true, K, 1, 0, G}}, fixed_life(600));
  ok = ok && (b.gangs.size() == G);
  for (uint32_t j = 0; ok && j < G; ++j) {
    const GangRec& g = b.gangs[j];
    for (uint32_t r = 0; r < K; ++r) {
      uint32_t n = j * K + r;
      if (g.members[r].core != n % NCC || g.members[r].slot != n / NCC) {
        std::printf("FAIL spread-empty: gang %u rank %u on core %u slot %u, want core %u slot %u\n", j, r,
                    g.members[r].core, g.members[r].slot, n % NCC, n / NCC);
        ok = false;
        break;
      }
    }
    if (K > 1 && (g.gang_id != j || g.epoch != 1)) {
      std::printf("FAIL spread-empty: gang %u got id %u epoch %u, want id %u epoch 1\n", j, g.gang_id, g.epoch, j);
      ok = false;
    }
    // the first beat leaves two cycles after the placement's last cycle
    if (g.out_cycle[0] != g.commit_cycle + 2) {
      std::printf("FAIL spread-empty: gang %u committed in cycle %lu, first beat out in %lu\n", j,
                  (unsigned long)g.commit_cycle, (unsigned long)g.out_cycle[0]);
      ok = false;
    }
  }
  return report("spread on an empty cluster", ok);
}

// A gang that does not fit waits, without output, until enough blocks finish.
static bool t_no_fit(Sim& sim, uint64_t& tick) {
  // one usable slot per core: NCC/K gangs fill the cluster
  uint32_t K = MAXC;
  uint32_t fill = NCC / K;
  Bench b(sim, tick, 31);
  RunOpts o;
  o.life = [fill](int seq, uint32_t, uint32_t) { return (uint32_t(seq) < fill) ? uint64_t(300) : uint64_t(20); };
  bool ok = b.run({{true, K, 1, LMEM_SIZE, fill + 2}}, o);
  ok = ok && (b.gangs.size() == fill + 2);
  if (ok) {
    uint64_t first_done = ~0ull;
    for (uint32_t j = 0; j < fill; ++j) {
      for (auto d : b.gangs[j].done_cycle) {
        first_done = std::min(first_done, d);
      }
    }
    const GangRec& w = b.gangs[fill];
    if (w.out_cycle[0] <= first_done) {
      std::printf("FAIL no-fit: the waiting gang left in cycle %lu, before the first block finished (%lu)\n",
                  (unsigned long)w.out_cycle[0], (unsigned long)first_done);
      ok = false;
    }
    for (uint32_t r = 0; r < K; ++r) {
      if (w.members[r].slot != 0) {
        std::printf("FAIL no-fit: rank %u in slot %u, only slot 0 is usable\n", r, w.members[r].slot);
        ok = false;
      }
    }
  }
  return report("gang that does not fit waits", ok);
}

// Every core but core 0 is busy for long: a gang of two must share core 0.
static bool t_share(Sim& sim, uint64_t& tick) {
  if (MAXC < 2 || NW < 2) {
    std::printf("forced core sharing: needs MAX_CLUSTER >= 2 and 2 warps, skipped\n");
    return true;
  }
  uint32_t K = 2;
  uint32_t W = NW / 2;
  // gangs 0..NCC-1 load every core with two blocks; core 0's leave soon
  Bench b(sim, tick, 41);
  RunOpts o;
  o.life = [](int seq, uint32_t, uint32_t core) {
    return (seq >= 0 && core != 0) ? uint64_t(800) : uint64_t(30);
  };
  uint32_t G = NCC + 1;
  bool ok = b.run({{true, K, W, 0, G}}, o);
  ok = ok && (b.gangs.size() == G);
  if (ok) {
    const GangRec& g = b.gangs[NCC];
    if (g.members[0].core != 0 || g.members[1].core != 0 || g.members[0].slot == g.members[1].slot) {
      std::printf("FAIL share: gang %u on %u/%u and %u/%u, want two slots of core 0\n", NCC, g.members[0].core,
                  g.members[0].slot, g.members[1].core, g.members[1].slot);
      ok = false;
    }
  }
  return report("forced core sharing", ok);
}

// A new launch restarts the pointer and the epochs.
static bool t_new_launch(Sim& sim, uint64_t& tick) {
  // a gang smaller than the cluster leaves the pointer off core 0
  uint32_t K = (MAXC == NCC && MAXC > 1) ? MAXC - 1 : MAXC;
  Bench b(sim, tick, 51);
  bool ok = b.run({{true, K, 1, 0, 1}, {true, K, 1, 0, 1}}, fixed_life(10));
  ok = ok && (b.gangs.size() == 2);
  if (ok) {
    const GangRec& g = b.gangs[1];
    for (uint32_t r = 0; r < K; ++r) {
      if (g.members[r].core != r || g.members[r].slot != 0) {
        std::printf("FAIL new-launch: rank %u on core %u slot %u, want core %u slot 0\n", r, g.members[r].core,
                    g.members[r].slot, r);
        ok = false;
      }
    }
    if (K > 1 && (g.gang_id != 0 || g.epoch != 1)) {
      std::printf("FAIL new-launch: id %u epoch %u, want id 0 epoch 1\n", g.gang_id, g.epoch);
      ok = false;
    }
  }
  return report("new launch resets pointer and epochs", ok);
}

// An id handed out again flips its epoch.
static bool t_epoch(Sim& sim, uint64_t& tick) {
  if (MAXC < 2) {
    std::printf("epoch flip: needs MAX_CLUSTER >= 2, skipped\n");
    return true;
  }
  Bench b(sim, tick, 61);
  RunOpts o = fixed_life(5);
  bool ok = b.run({{true, 2, 1, LMEM_SIZE, 4 * NCC}}, o);
  std::map<uint32_t, uint32_t> last;
  uint32_t reused = 0;
  for (const auto& g : b.gangs) {
    auto it = last.find(g.gang_id);
    uint32_t want = (it == last.end()) ? 1 : (it->second ^ 1);
    if (g.epoch != want) {
      std::printf("FAIL epoch: gang %d got id %u epoch %u, want %u\n", g.gang_seq, g.gang_id, g.epoch, want);
      ok = false;
      break;
    }
    reused += (it != last.end());
    last[g.gang_id] = g.epoch;
  }
  if (reused == 0) {
    std::printf("FAIL epoch: no gang id was reused\n");
    ok = false;
  }
  return report("epoch flip on id reuse", ok);
}

// Every id is live although slots are free: the next gang waits for an id.
static bool t_ids(Sim& sim, uint64_t& tick) {
  if (MAXC < 2) {
    std::printf("gang-id exhaustion: needs MAX_CLUSTER >= 2, skipped\n");
    return true;
  }
  uint32_t G = NUM_GANGS;   // NUM_GANGS gangs of two fill every slot
  Bench b(sim, tick, 71);
  RunOpts o;
  o.life = [G](int seq, uint32_t rank, uint32_t) {
    if (uint32_t(seq) >= G)
      return uint64_t(10);
    return (rank == 0) ? uint64_t(40) : uint64_t(400 + 10 * seq);
  };
  bool ok = b.run({{true, 2, 1, 0, G + 1}}, o);
  ok = ok && (b.gangs.size() == G + 1);
  if (ok) {
    uint64_t first_gang_done = ~0ull;
    for (uint32_t j = 0; j < G; ++j) {
      first_gang_done = std::min(first_gang_done, std::max(b.gangs[j].done_cycle[0], b.gangs[j].done_cycle[1]));
    }
    uint64_t rank0_done = 0;
    for (uint32_t j = 0; j < G; ++j) {
      rank0_done = std::max(rank0_done, b.gangs[j].done_cycle[0]);
    }
    const GangRec& w = b.gangs[G];
    if (w.out_cycle[0] <= first_gang_done) {
      std::printf("FAIL ids: gang %u left in cycle %lu, before any gang id was free (%lu)\n", G,
                  (unsigned long)w.out_cycle[0], (unsigned long)first_gang_done);
      ok = false;
    }
    if (rank0_done >= first_gang_done) {
      std::printf("FAIL ids: the schedule did not free slots before ids\n");
      ok = false;
    }
  }
  return report("gang-id exhaustion stalls", ok);
}

static bool t_random(Sim& sim, uint64_t& tick, uint32_t seed, uint32_t num_launches) {
  std::mt19937 rng(seed);
  const uint32_t strides[] = {0, BLOCK_ALIGN, align_up(LMEM_SIZE / 4), align_up(LMEM_SIZE / 3), LMEM_SIZE / 2,
                              LMEM_SIZE};
  std::vector<Launch> ls;
  for (uint32_t l = 0; l < num_launches; ++l) {
    Launch L;
    L.spread = (rng() % 4) != 0;
    L.k = 1 + rng() % (L.spread ? MAXC : MAX_LEGACY);
    L.w = 1 + rng() % NW;
    L.stride = strides[rng() % 6];
    L.gangs = 1 + rng() % (2 * NCC * NW / L.k + 2);
    ls.push_back(L);
  }
  Bench b(sim, tick, seed);
  RunOpts o;
  o.ready_pct = 50 + rng() % 51;
  o.gap_pct = rng() % 30;
  std::mt19937 lrng(seed * 7 + 1);
  o.life = [&lrng](int, uint32_t, uint32_t) { return uint64_t(1 + lrng() % 60); };
  bool ok = b.run(ls, o);
  std::printf("random: %u launches, %zu gangs, %lu cycles, ready %u%%, gaps %u%%\n", num_launches, b.gangs.size(),
              (unsigned long)b.cycles, o.ready_pct, o.gap_pct);
  return report("random", ok);
}

static bool t_schedule(Sim& sim, uint64_t& tick, const std::string& csv) {
  std::vector<Launch> ls = {
    {true, MAXC, 1, 0, 2 * NCC * NW / MAXC + 2},
    {false, std::min<uint32_t>(2, NW), 1, 0, 2},
    {true, std::max<uint32_t>(1, MAXC / 2), std::max<uint32_t>(1, NW / 2), LMEM_SIZE / 2, 2 * NCC + 1},
    {true, 1, NW, LMEM_SIZE, NCC + 2},
  };
  Bench b(sim, tick, 81);
  RunOpts o;
  o.life = [](int seq, uint32_t rank, uint32_t) { return uint64_t(6 + ((seq * 7 + rank * 5) % 23)); };
  bool ok = b.run(ls, o);
  for (const auto& g : b.gangs) {
    if (g.out_cycle[0] != g.commit_cycle + 2) {
      std::printf("FAIL schedule: gang %d committed in cycle %lu, first beat out in %lu\n", g.gang_seq,
                  (unsigned long)g.commit_cycle, (unsigned long)g.out_cycle[0]);
      ok = false;
      break;
    }
  }
  if (ok && !csv.empty()) {
    FILE* f = std::fopen(csv.c_str(), "w");
    if (!f) {
      std::printf("FAIL schedule: cannot write %s\n", csv.c_str());
      return false;
    }
    std::fprintf(f, "gang_seq,rank,core,slot,gang_id,epoch,commit_cycle\n");
    for (const auto& g : b.gangs) {
      for (uint32_t r = 0; r < g.k; ++r) {
        std::fprintf(f, "%d,%u,%u,%u,%u,%u,%lu\n", g.gang_seq, r, g.members[r].core, g.members[r].slot,
                     (g.k > 1) ? g.gang_id : 0, (g.k > 1) ? g.epoch : 0, (unsigned long)g.commit_cycle);
      }
    }
    std::fclose(f);
    std::printf("schedule: %zu gangs, %lu cycles, wrote %s\n", b.gangs.size(), (unsigned long)b.cycles, csv.c_str());
  }
  return report("deterministic schedule", ok);
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  uint32_t seed = 1;
  uint32_t launches = 30;
  std::string csv;
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '+') {
      continue;
    }
    if (std::strncmp(argv[i], "--seed=", 7) == 0) {
      seed = uint32_t(std::strtoul(argv[i] + 7, nullptr, 0));
    } else if (std::strncmp(argv[i], "--launches=", 11) == 0) {
      launches = uint32_t(std::strtoul(argv[i] + 11, nullptr, 0));
    } else if (std::strncmp(argv[i], "--csv=", 6) == 0) {
      csv = argv[i] + 6;
    } else {
      std::printf("usage: %s [--seed=N] [--launches=N] [--csv=FILE]\n", argv[0]);
      return 2;
    }
  }

  std::printf("cta_gang: cores=%u (%u sockets of %u) warps=%u max cluster=%u gang ids=%u lmem=%u cluster=%u seed=%u\n",
              NCC, NUM_SOCKETS, SOCKET_SIZE, NW, MAXC, NUM_GANGS, LMEM_SIZE, CLUSTER_ID, seed);

  Sim sim;
  uint64_t tick = 0;
  bool ok = true;
  ok = t_legacy(sim, tick) && ok;
  ok = t_spread_empty(sim, tick) && ok;
  ok = t_no_fit(sim, tick) && ok;
  ok = t_share(sim, tick) && ok;
  ok = t_new_launch(sim, tick) && ok;
  ok = t_epoch(sim, tick) && ok;
  ok = t_ids(sim, tick) && ok;
  ok = t_random(sim, tick, seed, launches) && ok;
  ok = t_schedule(sim, tick, csv) && ok;

  if (!ok) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
