// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// SimX gang unit (CtaGang) driven from host context the way hw/unittest/cta_gang
// drives VX_cta_gang, and checked cycle by cycle against a model built on the
// dlmem_ref placement functions: every cycle the unit's input take, its busy and
// the beat on each core lane must match the model, field for field. The model
// is the RTL bench's, with SimX's per-core launch lanes (capacity plus launch
// credits) in place of the RTL's output buffer; with always-ready lanes the two
// coincide, so the deterministic schedule's placement.csv must equal the RTL
// bench's byte for byte. Independently of the model, the bench tracks the
// blocks it holds resident (no slot used twice, no core over its warps, no gang
// id reused while live), the directed scenarios check hand-derived placements,
// and a set of hand-computed vectors checks dlmem_ref itself.
//
// Deterministic schedule (--csv): the RTL bench's, see hw/unittest/cta_gang.
//   A bench cycle n is SimX cycle n: inputs offered before the tick (a beat as
//   a held valid on bus_in, a done event through cta_done), a lane beat visible
//   before the tick is taken before it, and the input fired when the tick
//   consumed it.

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
#include <vector>

#include <dlmem_ref.h>
#include "kmu.h"
#include "cta_gang.h"

using namespace vortex;
namespace ref = vortex::dlmem_ref;

namespace {

constexpr uint32_t BLOCK_ALIGN = VX_CFG_MEM_BLOCK_SIZE;

uint32_t align_up(uint32_t x) {
  return (x + BLOCK_ALIGN - 1) & ~(BLOCK_ALIGN - 1);
}

uint32_t lmem_size_of(const ref::Cfg& cfg) {
  return 1u << cfg.lmem_log;
}

uint32_t max_legacy(const ref::Cfg& cfg) {
  return std::max(cfg.nw, cfg.max_cluster);
}

ref::Cfg make_cfg(uint32_t ncc, uint32_t socket_size, uint32_t nw, uint32_t max_cluster) {
  ref::Cfg cfg = dlmem_cfg();
  cfg.ncc         = ncc;
  cfg.socket_size = socket_size;
  cfg.num_sockets = ncc / socket_size;
  cfg.nw          = nw;
  cfg.max_cluster = max_cluster;
  cfg.num_gangs   = std::max<uint32_t>(2, ncc * nw / 2);
  return cfg;
}

std::string row_str(const ref::Row& r) {
  std::string s = "{k=" + std::to_string(r.k) + ":";
  for (const auto& p : r.peers) {
    s += " " + std::to_string(p.core) + "/" + std::to_string(p.slot);
  }
  return s + "}";
}

bool same_row(const ref::Row& a, const ref::Row& b) {
  if (a.k != b.k || a.peers.size() != b.peers.size())
    return false;
  for (size_t i = 0; i < a.peers.size(); ++i) {
    if (a.peers[i].core != b.peers[i].core || a.peers[i].slot != b.peers[i].slot)
      return false;
  }
  return true;
}

// Every field of a launch beat.
bool same_req(const kmu_req_t& a, const kmu_req_t& b) {
  bool ok = a.PC == b.PC && a.entry == b.entry && a.param == b.param && a.cta_id == b.cta_id
         && a.lmem_size == b.lmem_size && a.block_size == b.block_size
         && a.is_first_of_cluster == b.is_first_of_cluster && a.ctx_id == b.ctx_id
         && a.cluster_rank == b.cluster_rank && a.policy == b.policy && a.pinned == b.pinned
         && a.dest_core == b.dest_core && a.forced_slot == b.forced_slot && a.gang_id == b.gang_id
         && a.epoch == b.epoch && same_row(a.row, b.row);
  for (int i = 0; i < 3; ++i) {
    ok = ok && a.block_idx[i] == b.block_idx[i] && a.block_dim[i] == b.block_dim[i]
            && a.grid_dim[i] == b.grid_dim[i] && a.warp_step[i] == b.warp_step[i]
            && a.cluster_dim[i] == b.cluster_dim[i];
  }
  return ok;
}

uint32_t cluster_size(const kmu_req_t& req) {
  uint32_t k = req.cluster_dim[0] * req.cluster_dim[1] * req.cluster_dim[2];
  return (k == 0) ? 1 : k;
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
  uint32_t  launch;
  int       gang_seq;   // SPREAD gangs only, else -1
  uint32_t  rank;
  kmu_req_t req;
};

// One beat on its way down a core lane, as the model predicts it.
struct Expect {
  kmu_req_t req;
  int       gang_seq;
  uint32_t  rank;
  uint32_t  w;
  uint64_t  in_cycle;
  uint64_t  commit_cycle;
};

struct DoneEvt {
  uint32_t core;
  uint32_t slot;
  uint32_t gang;
  uint32_t epoch;
};

///////////////////////////////////////////////////////////////////////////////
// Cycle model of the gang unit and its core lanes. Within a cycle the CTA_DONE
// event applies first, then the unit decides. A lane takes a beat while it
// holds fewer than its capacity and owes fewer than LAUNCH_CREDITS admissions.

class GangModel {
public:
  enum Phase { IDLE, PLACE, EMIT };

  GangModel(const ref::Cfg& cfg, uint32_t lane_cap)
    : cfg_(cfg), lanes_(cfg.ncc), inflight_(cfg.ncc, 0), lane_cap_(lane_cap) {
    ref::reset(cfg_, m_);
    ref::reset(cfg_, g_);
  }

  // The unit's busy, before this cycle's inputs apply.
  bool busy(bool in_valid) const {
    bool used = false;
    for (uint32_t c = 0; c < cfg_.ncc; ++c) {
      used = used || (m_.free[c] != ref::low_mask(cfg_.nw));
    }
    bool live = false;
    for (bool l : g_.live) {
      live = live || l;
    }
    return in_valid || (phase_ != IDLE) || used || live;
  }

  bool lane_empty(uint32_t c) const {
    return lanes_.at(c).empty();
  }

  // The bench took lane c's head (and returned its admission credit).
  bool take(uint32_t c, Expect* e) {
    auto& q = lanes_.at(c);
    if (q.empty())
      return false;
    *e = q.front();
    q.pop_front();
    --inflight_[c];
    return true;
  }

  // Advances one cycle; returns the lane the head beat leaves on, or -1.
  int step(uint64_t cycle, const Beat* head, const DoneEvt* done, std::string& err) {
    if (done && policy_) {
      ref::release(m_, done->core, done->slot, w_);
      if (k_ > 1) {
        ref::done(g_, done->gang, done->epoch);
      }
    }

    Phase    phase_n = phase_;
    uint32_t idx_n   = idx_;
    bool     start   = false;
    int      fire    = -1;

    switch (phase_) {
    case IDLE:
      if (head) {
        const kmu_req_t& h = head->req;
        uint32_t K = cluster_size(h);
        uint32_t W = ref::warps_of(h.block_size, cfg_.nt);
        if (h.ctx_id != ctx_) {
          ref::new_launch(cfg_, m_, g_);
          failed_ = false;
        }
        if (h.policy != KMU_POLICY_SPREAD) {
          uint32_t target = locked_ ? lock_core_ : rr_core_;
          if (this->lane_ready(target)) {
            fire = int(target);
            this->push(target, this->legacy_beat(*head, W, cycle));
            locked_ = (h.cluster_rank + 1 != K);
            lock_core_ = target;
            if (!locked_) {
              rr_core_ = (target + 1) % cfg_.ncc;
            }
          }
        } else {
          bool id_free = (K == 1);
          for (bool l : g_.live) {
            id_free = id_free || !l;
          }
          if (id_free && (!failed_ || dirty_)) {
            start = true;
            uint32_t U = ref::usable_slots(cfg_, align_up(h.lmem_size));
            ref::Mirror s = m_;
            ok_ = ref::place(cfg_, s, K, W, U, members_);
            fail_step_ = K;
            if (!ok_) {
              for (uint32_t kk = 1; kk <= K; ++kk) {
                ref::Mirror t = m_;
                std::vector<ref::Peer> tmp;
                if (!ref::place(cfg_, t, kk, W, U, tmp)) {
                  fail_step_ = kk - 1;
                  break;
                }
              }
            }
          }
        }
        policy_ = (h.policy == KMU_POLICY_SPREAD);
        k_ = K;
        w_ = W;
        ctx_ = h.ctx_id;
      }
      phase_n = start ? PLACE : IDLE;
      idx_n = 0;
      break;

    case PLACE:
      if (!ok_ && idx_ == fail_step_) {
        failed_ = true;
        phase_n = IDLE;
      } else if (idx_ == k_ - 1) {
        for (const auto& p : members_) {
          m_.free[p.core] &= ~(uint64_t(1) << p.slot);
          m_.used[p.core] += w_;
        }
        m_.rr = (members_.back().core + 1) % cfg_.ncc;
        if (k_ > 1) {
          int id = ref::alloc(g_, k_);
          if (id < 0) {
            err = "model: no free gang id at commit";
            return -1;
          }
          cur_gang_ = uint32_t(id);
          cur_epoch_ = g_.epoch[id];
        } else {
          cur_gang_ = 0;
          cur_epoch_ = 0;
        }
        failed_ = false;
        commit_cycle_ = cycle;
        idx_n = 0;
        phase_n = EMIT;
      } else {
        idx_n = idx_ + 1;
      }
      break;

    case EMIT:
      if (head) {
        const ref::Peer& p = members_.at(idx_);
        if (this->lane_ready(p.core)) {
          const kmu_req_t& h = head->req;
          if (h.policy != KMU_POLICY_SPREAD || h.cluster_rank != idx_ || h.ctx_id != ctx_) {
            err = "model: beat out of step with the gang being emitted";
            return -1;
          }
          fire = int(p.core);
          this->push(p.core, this->spread_beat(*head, cycle));
          idx_n = idx_ + 1;
          if (idx_ + 1 == k_) {
            phase_n = IDLE;
            idx_n = 0;
          }
        }
      }
      break;
    }

    dirty_ = start ? false : (dirty_ || (done != nullptr));
    phase_ = phase_n;
    idx_ = idx_n;
    return fire;
  }

private:
  bool lane_ready(uint32_t c) const {
    return lanes_.at(c).size() < lane_cap_ && inflight_.at(c) < Kmu::LAUNCH_CREDITS;
  }

  void push(uint32_t c, const Expect& e) {
    lanes_.at(c).push_back(e);
    ++inflight_[c];
  }

  Expect legacy_beat(const Beat& b, uint32_t W, uint64_t cycle) const {
    Expect e{};
    e.req = b.req;
    e.gang_seq = -1;
    e.rank = b.rank;
    e.w = W;
    e.in_cycle = cycle;
    return e;
  }

  Expect spread_beat(const Beat& b, uint64_t cycle) const {
    Expect e{};
    const ref::Peer& p = members_.at(idx_);
    e.req = b.req;
    e.req.pinned = true;
    e.req.dest_core = p.core;
    e.req.forced_slot = p.slot;
    e.req.gang_id = cur_gang_;
    e.req.epoch = cur_epoch_;
    e.req.row.k = k_;
    e.req.row.peers = members_;
    e.gang_seq = b.gang_seq;
    e.rank = b.rank;
    e.w = w_;
    e.in_cycle = cycle;
    e.commit_cycle = commit_cycle_;
    return e;
  }

  ref::Cfg    cfg_;
  ref::Mirror m_;
  ref::GangIds g_;
  Phase    phase_ = IDLE;
  uint32_t idx_ = 0;
  uint32_t ctx_ = 0;
  bool     policy_ = false;
  uint32_t k_ = 1, w_ = 1;
  bool     failed_ = false, dirty_ = false;
  bool     ok_ = false;
  uint32_t fail_step_ = 0;
  std::vector<ref::Peer> members_;
  uint32_t cur_gang_ = 0, cur_epoch_ = 0;
  uint64_t commit_cycle_ = 0;
  // legacy steering: round robin, a message locked to the core of its first beat
  uint32_t rr_core_ = 0;
  bool     locked_ = false;
  uint32_t lock_core_ = 0;
  std::vector<std::deque<Expect>> lanes_;
  std::vector<uint32_t> inflight_;
  uint32_t lane_cap_;
};

///////////////////////////////////////////////////////////////////////////////

struct GangRec {
  int      gang_seq;
  uint32_t k;
  uint32_t gang_id;
  uint32_t epoch;
  std::vector<ref::Peer> members;   // by rank, as delivered
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
  uint32_t ready_pct = 100;   // lane ready probability
  uint32_t gap_pct   = 0;     // probability of holding a beat back for a cycle
  // a block's lifetime in cycles, from the cycle it leaves the unit's lane
  std::function<uint64_t(int gang_seq, uint32_t rank, uint32_t core)> life;
};

class Bench {
public:
  Bench(const ref::Cfg& cfg, uint32_t seed) : cfg_(cfg), rng_(seed) {}

  std::vector<GangRec> gangs;
  std::vector<std::pair<uint64_t, uint64_t>> legacy_lat;   // (in fire, lane take)
  uint64_t cycles = 0;
  uint32_t errors = 0;

  bool run(const std::vector<Launch>& launches, const RunOpts& o) {
    auto& platform = SimPlatform::instance();
    platform.set_num_workers(1);
    bool ok;
    {
      auto gang = CtaGang::Create("cta-gang", cfg_);
      platform.reset();
      ok = this->loop(*gang, launches, o);
    }
    platform.finalize();
    return ok;
  }

private:
  template <typename... Args>
  void fail(const char* fmt, Args... args) {
    if (++errors <= 10) {
      std::printf("FAIL cycle %lu: ", (unsigned long)cycles);
      std::printf(fmt, args...);
      std::printf("\n");
    }
  }

  kmu_req_t make_req(uint32_t launch, const Launch& L, uint32_t rank) {
    kmu_req_t r{};
    r.PC     = (uint64_t(rng_()) << 32) | rng_();
    r.entry  = (uint64_t(rng_()) << 32) | rng_();
    r.param  = (uint64_t(rng_()) << 32) | rng_();
    r.cta_id = rng_();
    for (int i = 0; i < 3; ++i) {
      r.block_idx[i] = rng_();
      r.block_dim[i] = rng_();
      r.grid_dim[i]  = rng_();
      r.warp_step[i] = rng_();
    }
    r.cluster_dim[0] = L.k;
    r.cluster_dim[1] = 1;
    r.cluster_dim[2] = 1;
    r.lmem_size = L.stride;
    r.block_size = L.w * cfg_.nt;
    r.is_first_of_cluster = (rank == 0);
    r.ctx_id = uint8_t(launch + 1);
    r.cluster_rank = rank;
    r.policy = L.spread ? KMU_POLICY_SPREAD : KMU_POLICY_DEFAULT;
    r.pinned = false;
    r.dest_core = 0;
    r.forced_slot = 0;
    r.gang_id = 0;
    r.epoch = 0;
    r.row = ref::Row{0, {}};
    return r;
  }

  uint32_t k_of(int gang_seq) const {
    auto it = ks_.find(gang_seq);
    return (it == ks_.end()) ? 1 : it->second;
  }

  void check_out(uint32_t lane, const kmu_req_t& got, const Expect& e) {
    if (same_req(got, e.req))
      return;
    if (got.pinned != e.req.pinned || got.dest_core != e.req.dest_core || got.forced_slot != e.req.forced_slot
     || got.gang_id != e.req.gang_id || got.epoch != e.req.epoch || !same_row(got.row, e.req.row)) {
      this->fail("lane %u beat (gang %d rank %u): pin %d core %u slot %u gang %u epoch %u row %s;"
                 " model: pin %d core %u slot %u gang %u epoch %u row %s",
                 lane, e.gang_seq, e.rank, int(got.pinned), got.dest_core, got.forced_slot, got.gang_id, got.epoch,
                 row_str(got.row).c_str(), int(e.req.pinned), e.req.dest_core, e.req.forced_slot, e.req.gang_id,
                 e.req.epoch, row_str(e.req.row).c_str());
    } else {
      this->fail("lane %u beat (gang %d rank %u): payload differs from the input beat", lane, e.gang_seq, e.rank);
    }
  }

  bool loop(CtaGang& gang, const std::vector<Launch>& launches, const RunOpts& o) {
    auto& platform = SimPlatform::instance();
    const uint32_t NCC = cfg_.ncc;
    GangModel model(cfg_, gang.bus_out.at(0).capacity());
    std::vector<std::deque<Beat>> beats(launches.size());
    int gang_seq = 0;
    for (uint32_t l = 0; l < launches.size(); ++l) {
      const Launch& L = launches[l];
      for (uint32_t gi = 0; gi < L.gangs; ++gi) {
        int seq = L.spread ? gang_seq++ : -1;
        for (uint32_t r = 0; r < L.k; ++r) {
          beats[l].push_back({l, seq, r, this->make_req(l, L, r)});
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
    Beat head{};
    bool busy_prev = true;
    uint64_t idle = 0;
    cycles = 0;

    while (true) {
      // the launch in progress has drained: present the next one
      bool launch_open = (cur_launch < launches.size()) && !beats[cur_launch].empty();
      if (!launch_open && !head_valid && cur_launch < launches.size()) {
        bool drained = res.empty() && done_q.empty() && !busy_prev;
        if (drained) {
          ++cur_launch;
          if (cur_launch >= launches.size())
            break;
        }
      }
      // CTA completions due this cycle, one presented per cycle
      for (auto it = res.begin(); it != res.end();) {
        if (it->done_at <= cycles) {
          done_q.push_back({it->core, it->slot, it->gang, it->epoch});
          uint32_t c = it->core;
          occ[c] &= ~(uint64_t(1) << it->slot);
          warps[c] -= it->w;
          if (this->k_of(it->gang_seq) > 1) {
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
        gang.cta_done(dev.core, dev.slot, dev.gang, dev.epoch);
      }

      if (!head_valid && cur_launch < launches.size() && !beats[cur_launch].empty()
       && (o.gap_pct == 0 || (rng_() % 100) >= o.gap_pct)) {
        head = beats[cur_launch].front();
        beats[cur_launch].pop_front();
        head_valid = true;
        gang.bus_in.send(head.req, 0);
      }

      // core lanes: a beat visible before the tick is taken before it
      bool any_had = false;
      bool out_fire = false;
      for (uint32_t c = 0; c < NCC; ++c) {
        bool had = !gang.bus_out.at(c).empty();
        any_had = any_had || had;
        if (had == model.lane_empty(c)) {
          this->fail("lane %u holds %s beat, the model says %s", c, had ? "a" : "no", had ? "none" : "one");
          return false;
        }
        bool rdy = (o.ready_pct >= 100) || ((rng_() % 100) < o.ready_pct);
        if (!had || !rdy)
          continue;
        kmu_req_t beat = gang.bus_out.at(c).peek();
        gang.bus_out.at(c).pop();
        gang.credit_in.at(c).send(1, 0);
        out_fire = true;
        Expect e{};
        model.take(c, &e);
        this->check_out(c, beat, e);
        if (e.req.pinned) {
          // the bench's own view: the block is now resident
          uint32_t s = beat.forced_slot;
          if (beat.dest_core != c) {
            this->fail("a beat for core %u left on lane %u", beat.dest_core, c);
          }
          if ((occ[c] >> s) & 1) {
            this->fail("slot %u of core %u handed out twice", s, c);
          }
          occ[c] |= uint64_t(1) << s;
          warps[c] += e.w;
          if (warps[c] > cfg_.nw) {
            this->fail("core %u holds %u warps (of %u)", c, warps[c], cfg_.nw);
          }
          if (e.req.row.k > 1) {
            auto lit = live_ids.find(e.req.gang_id);
            if (lit == live_ids.end()) {
              live_ids[e.req.gang_id] = {e.gang_seq, e.req.row.k};
            } else if (lit->second.first != e.gang_seq) {
              this->fail("gang id %u handed to gang %d while gang %d still holds it", e.req.gang_id, e.gang_seq,
                         lit->second.first);
            }
          }
          auto git = gang_index.find(e.gang_seq);
          if (git == gang_index.end()) {
            GangRec g;
            g.gang_seq = e.gang_seq;
            g.k = e.req.row.k;
            g.gang_id = e.req.gang_id;
            g.epoch = e.req.epoch;
            g.members.assign(e.req.row.k, ref::Peer{0, 0});
            g.out_cycle.assign(e.req.row.k, 0);
            g.done_cycle.assign(e.req.row.k, 0);
            g.commit_cycle = e.commit_cycle;
            gang_index[e.gang_seq] = gangs.size();
            gangs.push_back(g);
            git = gang_index.find(e.gang_seq);
          }
          GangRec& g = gangs[git->second];
          g.members[e.rank] = {c, s};
          g.out_cycle[e.rank] = cycles;
          ks_[e.gang_seq] = e.req.row.k;
          res.push_back({e.gang_seq, e.rank, c, s, e.req.gang_id, e.req.epoch, e.w,
                         cycles + o.life(e.gang_seq, e.rank, c)});
        } else {
          legacy_lat.push_back({e.in_cycle, cycles});
        }
      }

      bool busy = gang.busy();
      if (busy != model.busy(head_valid)) {
        this->fail("busy=%d, the model says %d", int(busy), int(!busy));
      }
      busy_prev = busy || any_had;

      std::string err;
      int want = model.step(cycles, head_valid ? &head : nullptr, dvalid ? &dev : nullptr, err);
      if (!err.empty()) {
        this->fail("%s", err.c_str());
        return false;
      }

      platform.tick();

      bool in_fire = head_valid && (gang.bus_in.size() == 0);
      if (head_valid && in_fire != (want >= 0)) {
        this->fail("input %s, the model says %s (launch %u, gang %d, rank %u)", in_fire ? "taken" : "held",
                   (want >= 0) ? "taken" : "held", head.launch + 1, head.gang_seq, head.rank);
      }
      uint32_t credits = 0;
      while (!gang.credit_out.empty()) {
        gang.credit_out.pop();
        ++credits;
      }
      if (credits != (in_fire ? 1u : 0u)) {
        this->fail("%u launch credits returned for %u beats taken", credits, in_fire ? 1u : 0u);
      }
      if (in_fire) {
        head_valid = false;
      }
      ++cycles;

      idle = (in_fire || out_fire || dvalid) ? 0 : idle + 1;
      if (idle > 5000) {
        this->fail("no progress for 5000 cycles (launch %u)", cur_launch + 1);
        return false;
      }
      if (errors > 10) {
        return false;
      }
    }
    if (!live_ids.empty()) {
      this->fail("%zu gang ids still live at the end", live_ids.size());
    }
    if (gang.busy()) {
      this->fail("%s", "the unit is still busy at the end");
    }
    return errors == 0;
  }

  ref::Cfg cfg_;
  std::mt19937 rng_;
  std::map<int, uint32_t> ks_;
};

///////////////////////////////////////////////////////////////////////////////

RunOpts fixed_life(uint64_t cycles) {
  RunOpts o;
  o.life = [cycles](int, uint32_t, uint32_t) { return cycles; };
  return o;
}

bool report(const char* name, bool ok) {
  std::printf("%s: %s\n", name, ok ? "ok" : "FAILED");
  return ok;
}

// dlmem_ref against values derived by hand, so the model is not only checked against itself.
bool t_ref_vectors() {
  bool ok = true;
  auto expect = [&](bool cond, const char* what) {
    if (!cond) {
      std::printf("FAIL ref vectors: %s\n", what);
      ok = false;
    }
  };
  auto placed = [](const std::vector<ref::Peer>& got, std::vector<ref::Peer> want) {
    if (got.size() != want.size())
      return false;
    for (size_t i = 0; i < got.size(); ++i) {
      if (got[i].core != want[i].core || got[i].slot != want[i].slot)
        return false;
    }
    return true;
  };

  ref::Cfg c4 = {4, 4, 4, 4, 14, 8, 2, 2, 0, 0};
  expect(ref::usable_slots(c4, 0) == 4, "usable_slots(stride 0)");
  expect(ref::usable_slots(c4, 16384) == 1, "usable_slots(16384)");
  expect(ref::usable_slots(c4, 8192) == 2, "usable_slots(8192)");
  expect(ref::usable_slots(c4, 5504) == 2, "usable_slots(5504)");
  expect(ref::usable_slots(c4, 4096) == 4, "usable_slots(4096)");
  expect(ref::usable_slots(c4, 64) == 4, "usable_slots(64)");
  expect(ref::warps_of(17, 4) == 5 && ref::warps_of(16, 4) == 4, "warps_of");

  ref::Cfg d8 = {8, 4, 4, 8, 14, 16, 2, 4, 0, 0};
  ref::Cfg d4 = {4, 4, 4, 4, 14, 8, 4, 1, 0, 0};
  ref::Cfg d6 = {6, 4, 4, 6, 14, 12, 3, 2, 0, 0};
  expect(ref::kmu_dest(d8, 1, 5) == 13, "kmu_dest(4 sockets of 2, cluster 1, core 5)");
  expect(ref::kmu_dest(d8, 0, 7) == 7, "kmu_dest(4 sockets of 2, cluster 0, core 7)");
  expect(ref::kmu_dest(d4, 2, 3) == 11, "kmu_dest(1 socket of 4, cluster 2, core 3)");
  expect(ref::kmu_dest(d6, 1, 4) == 13, "kmu_dest(2 sockets of 3, cluster 1, core 4)");

  ref::Mirror m;
  ref::reset(c4, m);
  std::vector<ref::Peer> out;
  expect(ref::place(c4, m, 4, 1, 4, out) && placed(out, {{0, 0}, {1, 0}, {2, 0}, {3, 0}}) && m.rr == 0,
         "K=4 on an empty mirror");
  expect(ref::place(c4, m, 2, 1, 4, out) && placed(out, {{0, 1}, {1, 1}}) && m.rr == 2, "K=2 from rr 0");
  expect(ref::place(c4, m, 3, 1, 4, out) && placed(out, {{2, 1}, {3, 1}, {0, 2}}) && m.rr == 1,
         "K=3 from rr 2 wraps to core 0");
  // core 0 has a free slot but only one free warp
  expect(ref::place(c4, m, 2, 2, 4, out) && placed(out, {{1, 2}, {2, 2}}) && m.rr == 3,
         "K=2 W=2 skips the core short of warps");
  // only cores 0 and 3 are capable: the third member starts a second round on core 3
  expect(ref::place(c4, m, 3, 1, 4, out) && placed(out, {{3, 2}, {0, 3}, {3, 3}}) && m.rr == 0,
         "K=3 shares a core in a second round");
  ref::Mirror before = m;
  expect(!ref::place(c4, m, 1, 1, 4, out) && out.empty(), "a full cluster places nothing");
  expect(m.free == before.free && m.used == before.used && m.rr == before.rr, "a failed placement leaves the mirror");
  ref::release(m, 1, 2, 2);
  expect(m.free[1] == 0xC && m.used[1] == 2, "release");
  expect(!ref::place(c4, m, 2, 1, 1, out), "U=1 with every slot 0 taken");
  expect(ref::place(c4, m, 2, 1, 4, out) && placed(out, {{1, 2}, {1, 3}}) && m.rr == 2,
         "K=2 with one capable core takes two of its slots");

  ref::Cfg p2 = c4;
  p2.num_gangs = 2;
  ref::GangIds g;
  ref::reset(p2, g);
  int a = ref::alloc(g, 2);
  int b = ref::alloc(g, 3);
  expect(a == 0 && b == 1 && g.epoch[0] == 1 && g.epoch[1] == 1, "alloc hands out the lowest ids at epoch 1");
  expect(ref::alloc(g, 2) == -1, "an exhausted pool");
  ref::done(g, 0, 1);
  expect(g.live[0] && ref::alloc(g, 2) == -1, "an id stays live until its last member is done");
  ref::done(g, 0, 1);
  expect(!g.live[0] && ref::alloc(g, 2) == 0 && g.epoch[0] == 0, "a reused id flips its epoch");

  ref::Row row = ref::legacy_row(3, 2, 5, 4);
  expect(row.k == 4 && placed(row.peers, {{3, 2}, {3, 3}, {3, 4}, {3, 5}}), "legacy_row clamps to max_cluster");

  return report("dlmem_ref hand vectors", ok);
}

// Legacy beats pass unchanged, one cycle through the unit.
bool t_legacy(const ref::Cfg& cfg) {
  std::vector<Launch> ls = {{false, 1, 1, 0, 6}, {false, max_legacy(cfg), 1, 0, 3}};
  Bench b(cfg, 11);
  bool ok = b.run(ls, fixed_life(1));
  for (auto& l : b.legacy_lat) {
    if (l.second != l.first + 1) {
      std::printf("FAIL legacy: a beat taken in cycle %lu left in cycle %lu\n", (unsigned long)l.first,
                  (unsigned long)l.second);
      ok = false;
      break;
    }
  }
  ok = ok && (b.legacy_lat.size() == 6 + 3 * max_legacy(cfg)) && b.gangs.empty();
  // and under lane back-pressure
  Bench b2(cfg, 12);
  RunOpts o2 = fixed_life(1);
  o2.ready_pct = 40;
  o2.gap_pct = 30;
  ok = b2.run(ls, o2) && ok;
  return report("legacy pass-through", ok);
}

// An empty cluster: gang j member r lands on core (j*K + r) % NCC, slot (j*K + r) / NCC.
bool t_spread_empty(const ref::Cfg& cfg) {
  uint32_t K = cfg.max_cluster;
  uint32_t G = std::min<uint32_t>(cfg.nw, 3) * cfg.ncc / K;
  Bench b(cfg, 21);
  bool ok = b.run({{true, K, 1, 0, G}}, fixed_life(600));
  ok = ok && (b.gangs.size() == G);
  for (uint32_t j = 0; ok && j < G; ++j) {
    const GangRec& g = b.gangs[j];
    for (uint32_t r = 0; r < K; ++r) {
      uint32_t n = j * K + r;
      if (g.members[r].core != n % cfg.ncc || g.members[r].slot != n / cfg.ncc) {
        std::printf("FAIL spread-empty: gang %u rank %u on core %u slot %u, want core %u slot %u\n", j, r,
                    g.members[r].core, g.members[r].slot, n % cfg.ncc, n / cfg.ncc);
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
bool t_no_fit(const ref::Cfg& cfg) {
  // one usable slot per core: NCC/K gangs fill the cluster
  uint32_t K = cfg.max_cluster;
  uint32_t fill = cfg.ncc / K;
  Bench b(cfg, 31);
  RunOpts o;
  o.life = [fill](int seq, uint32_t, uint32_t) { return (uint32_t(seq) < fill) ? uint64_t(300) : uint64_t(20); };
  bool ok = b.run({{true, K, 1, lmem_size_of(cfg), fill + 2}}, o);
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
bool t_share(const ref::Cfg& cfg) {
  if (cfg.max_cluster < 2 || cfg.nw < 2) {
    std::printf("forced core sharing: needs MAX_CLUSTER >= 2 and 2 warps, skipped\n");
    return true;
  }
  uint32_t K = 2;
  uint32_t W = cfg.nw / 2;
  // gangs 0..NCC-1 load every core with two blocks; core 0's leave soon
  Bench b(cfg, 41);
  RunOpts o;
  o.life = [](int seq, uint32_t, uint32_t core) {
    return (seq >= 0 && core != 0) ? uint64_t(800) : uint64_t(30);
  };
  uint32_t G = cfg.ncc + 1;
  bool ok = b.run({{true, K, W, 0, G}}, o);
  ok = ok && (b.gangs.size() == G);
  if (ok) {
    const GangRec& g = b.gangs[cfg.ncc];
    if (g.members[0].core != 0 || g.members[1].core != 0 || g.members[0].slot == g.members[1].slot) {
      std::printf("FAIL share: gang %u on %u/%u and %u/%u, want two slots of core 0\n", cfg.ncc, g.members[0].core,
                  g.members[0].slot, g.members[1].core, g.members[1].slot);
      ok = false;
    }
  }
  return report("forced core sharing", ok);
}

// A new launch restarts the pointer and the epochs.
bool t_new_launch(const ref::Cfg& cfg) {
  // a gang smaller than the cluster leaves the pointer off core 0
  uint32_t K = (cfg.max_cluster == cfg.ncc && cfg.max_cluster > 1) ? cfg.max_cluster - 1 : cfg.max_cluster;
  Bench b(cfg, 51);
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
bool t_epoch(const ref::Cfg& cfg) {
  if (cfg.max_cluster < 2) {
    std::printf("epoch flip: needs MAX_CLUSTER >= 2, skipped\n");
    return true;
  }
  Bench b(cfg, 61);
  bool ok = b.run({{true, 2, 1, lmem_size_of(cfg), 4 * cfg.ncc}}, fixed_life(5));
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
bool t_ids(const ref::Cfg& cfg) {
  if (cfg.max_cluster < 2) {
    std::printf("gang-id exhaustion: needs MAX_CLUSTER >= 2, skipped\n");
    return true;
  }
  uint32_t G = cfg.num_gangs;   // num_gangs gangs of two fill every slot
  Bench b(cfg, 71);
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

bool t_random(const ref::Cfg& cfg, uint32_t seed, uint32_t num_launches) {
  std::mt19937 rng(seed);
  uint32_t lmem = lmem_size_of(cfg);
  const uint32_t strides[] = {0, BLOCK_ALIGN, align_up(lmem / 4), align_up(lmem / 3), lmem / 2, lmem};
  std::vector<Launch> ls;
  for (uint32_t l = 0; l < num_launches; ++l) {
    Launch L;
    L.spread = (rng() % 4) != 0;
    L.k = 1 + rng() % (L.spread ? cfg.max_cluster : max_legacy(cfg));
    L.w = 1 + rng() % cfg.nw;
    L.stride = strides[rng() % 6];
    L.gangs = 1 + rng() % (2 * cfg.ncc * cfg.nw / L.k + 2);
    ls.push_back(L);
  }
  Bench b(cfg, seed);
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

bool t_schedule(const ref::Cfg& cfg, const std::string& csv) {
  const uint32_t NCC = cfg.ncc, NW = cfg.nw, MAXC = cfg.max_cluster, LMEM = lmem_size_of(cfg);
  std::vector<Launch> ls = {
    {true, MAXC, 1, 0, 2 * NCC * NW / MAXC + 2},
    {false, std::min<uint32_t>(2, NW), 1, 0, 2},
    {true, std::max<uint32_t>(1, MAXC / 2), std::max<uint32_t>(1, NW / 2), LMEM / 2, 2 * NCC + 1},
    {true, 1, NW, LMEM, NCC + 2},
  };
  Bench b(cfg, 81);
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

bool parse_shape(const char* s, ref::Cfg* cfg) {
  unsigned ncc = 0, ss = 0, nw = 0, maxc = 0, lmem_log = 0;
  int n = std::sscanf(s, "%u,%u,%u,%u,%u", &ncc, &ss, &nw, &maxc, &lmem_log);
  if (n != 4 && n != 5)
    return false;
  if (ncc == 0 || ss == 0 || (ncc % ss) != 0 || nw == 0 || nw > 64 || maxc == 0 || maxc > ncc)
    return false;
  *cfg = make_cfg(ncc, ss, nw, maxc);
  if (n == 5) {
    if (lmem_log < 6 || lmem_log > 18)
      return false;
    cfg->lmem_log = lmem_log;
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  uint32_t seed = 1;
  uint32_t launches = 30;
  std::string csv;
  ref::Cfg cfg = dlmem_cfg();
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--seed=", 7) == 0) {
      seed = uint32_t(std::strtoul(argv[i] + 7, nullptr, 0));
    } else if (std::strncmp(argv[i], "--launches=", 11) == 0) {
      launches = uint32_t(std::strtoul(argv[i] + 11, nullptr, 0));
    } else if (std::strncmp(argv[i], "--csv=", 6) == 0) {
      csv = argv[i] + 6;
    } else if (std::strncmp(argv[i], "--shape=", 8) == 0 && parse_shape(argv[i] + 8, &cfg)) {
      continue;
    } else {
      std::printf("usage: %s [--seed=N] [--launches=N] [--csv=FILE] [--shape=CORES,SOCKET_SIZE,WARPS,MAX_CLUSTER[,LMEM_LOG]]\n",
                  argv[0]);
      return 2;
    }
  }

  std::printf("cta_gang: cores=%u (%u sockets of %u) warps=%u max cluster=%u gang ids=%u lmem=%u seed=%u\n", cfg.ncc,
              cfg.num_sockets, cfg.socket_size, cfg.nw, cfg.max_cluster, cfg.num_gangs, lmem_size_of(cfg), seed);

  bool ok = true;
  ok = t_ref_vectors() && ok;
  ok = t_legacy(cfg) && ok;
  ok = t_spread_empty(cfg) && ok;
  ok = t_no_fit(cfg) && ok;
  ok = t_share(cfg) && ok;
  ok = t_new_launch(cfg) && ok;
  ok = t_epoch(cfg) && ok;
  ok = t_ids(cfg) && ok;
  ok = t_random(cfg, seed, launches) && ok;
  ok = t_schedule(cfg, csv) && ok;

  if (!ok) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
