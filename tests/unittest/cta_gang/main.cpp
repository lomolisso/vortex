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
// dlmem_ref placement functions: every cycle the unit's input take, its busy,
// the beat on its output lane and the cluster-barrier release it sends must
// match the model, field for field. The model
// is the RTL bench's, with the lane standing for the RTL's two-entry output
// buffer. The bench returns a lane credit in the cycle it takes the beat, so
// under random back-pressure a full lane drains a cycle earlier than the RTL
// buffer's registered ready allows; with an always-ready lane the two coincide,
// so the deterministic schedule's placement.csv must equal the RTL bench's byte
// for byte. Independently of the model, the bench tracks the
// blocks it holds resident (no slot used twice, no core over its warps, no gang
// id reused while live), the directed scenarios check hand-derived placements,
// and a set of hand-computed vectors checks dlmem_ref itself.
//
// Deterministic schedule (--csv): the RTL bench's, see hw/unittest/cta_gang.
//   A bench cycle n is SimX cycle n: inputs offered before the tick (a beat as
//   a held valid on bus_in, an event through cta_event), a lane beat visible
//   before the tick is taken before it, and the input fired when the tick
//   consumed it. A release sent in cycle n reaches the bench's sinks at the
//   start of cycle n + release_delay().
//
// Cluster-barrier schedule (--cbar-csv): the RTL bench's, see hw/unittest/cta_gang.
//   One SPREAD launch, K=MAX, W=1, stride 0, 2*NCC*NW/MAX+2 gangs (none when
//   MAX=1). Gang s runs 1 + s%3 phases. Block (s, r) leaves the unit in cycle o;
//   its phase-0 arrival is queued in cycle o + 3 + (5s + 3r) % 11, its phase-i
//   arrival in cycle R + 2 + (3s + 5r + i) % 7, where R is the cycle the gang's
//   phase-(i-1) release leaves the unit, and its CTA_DONE in cycle
//   A + 1 + (7s + 5r) % 9, A the cycle of its last arrival. Events are queued in
//   the order the blocks left the unit and presented one per cycle.
//   cbar.csv: gang,epoch,arrive_cycle,release_cycle, one row per phase in
//   release order; arrive_cycle presents the phase's K-th arrival.

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
#include <tuple>
#include <vector>

#include <dlmem_ref.h>
#include "kmu.h"
#include "cta_gang.h"
#include "barrier_unit.h"

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

// One beat on its way down the output lane, as the model predicts it.
struct Expect {
  kmu_req_t req;
  int       gang_seq;
  uint32_t  rank;
  uint32_t  w;
  uint64_t  in_cycle;
  uint64_t  commit_cycle;
};

// A CTA_DONE event, or with cbar set a cluster-barrier arrival.
struct DoneEvt {
  uint32_t core;
  uint32_t slot;
  uint32_t gang;
  uint32_t epoch;
  bool     cbar = false;
  uint32_t count = 0;
};

///////////////////////////////////////////////////////////////////////////////
// Cycle model of the gang unit and its output lane. Within a cycle a CTA_DONE
// event applies first, then the unit decides, then the lowest pending release
// leaves, and last a CBAR event counts (its release is a register). The lane
// takes a beat while it holds fewer than its capacity.

class GangModel {
public:
  enum Phase { IDLE, PLACE, EMIT };

  GangModel(const ref::Cfg& cfg, uint32_t lane_cap)
    : cfg_(cfg), lane_cap_(lane_cap) {
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
    for (size_t i = 0; i < g_.live.size(); ++i) {
      live = live || g_.live[i] || g_.rel_pend[i];
    }
    return in_valid || (phase_ != IDLE) || used || live;
  }

  bool lane_empty() const {
    return lane_.empty();
  }

  // The bench took the lane's head (and returned its credit).
  bool take(Expect* e) {
    if (lane_.empty())
      return false;
    *e = lane_.front();
    lane_.pop_front();
    return true;
  }

  // Advances one cycle; returns the core the head beat is for (0 for a
  // pass-through beat), or -1 when it stays. *rel
  // is the gang whose release leaves this cycle, or -1, and *rel_ep its epoch.
  int step(uint64_t cycle, const Beat* head, const DoneEvt* ev, std::string& err, int* rel, uint32_t* rel_ep) {
    const DoneEvt* done = (ev && !ev->cbar) ? ev : nullptr;
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
          if (this->lane_ready()) {
            fire = 0;
            this->push(this->legacy_beat(*head, W, cycle));
          }
        } else {
          bool id_free = (K == 1);
          for (size_t i = 0; i < g_.live.size(); ++i) {
            id_free = id_free || (!g_.live[i] && !g_.rel_pend[i]);
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
        if (this->lane_ready()) {
          const kmu_req_t& h = head->req;
          if (h.policy != KMU_POLICY_SPREAD || h.cluster_rank != idx_ || h.ctx_id != ctx_) {
            err = "model: beat out of step with the gang being emitted";
            return -1;
          }
          fire = int(p.core);
          this->push(this->spread_beat(*head, cycle));
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

    *rel = ref::cbar_pop(g_);
    *rel_ep = (*rel >= 0) ? g_.epoch[*rel] : 0;
    if (ev && ev->cbar) {
      if (!policy_ || k_ < 2 || ev->count != k_) {
        err = "model: cluster-barrier arrival outside a SPREAD gang of its K";
        return -1;
      }
      ref::cbar_arrive(g_, ev->gang, ev->epoch, k_);
    }
    return fire;
  }

private:
  bool lane_ready() const {
    return lane_.size() < lane_cap_;
  }

  void push(const Expect& e) {
    lane_.push_back(e);
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
  std::deque<Expect> lane_;
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
  std::vector<uint64_t> done_pres;  // the cycle each CTA_DONE was presented
  uint64_t commit_cycle;
};

// One cluster-barrier phase of a gang, from its last arrival to its release.
struct CbarRec {
  int      gang_seq;
  uint32_t phase;
  uint32_t gang_id;
  uint32_t epoch;
  uint64_t arrive_cycle;
  uint64_t release_cycle;
};

struct Resident {
  int      gang_seq;
  uint32_t rank;
  uint32_t core, slot, gang, epoch, w;
  uint64_t done_at;
  uint32_t phases;        // cluster-barrier arrivals the block makes
  uint32_t phase;         // arrivals made
  uint64_t arrive_at;     // its next arrival, or NEVER while it waits for a release
};

constexpr uint64_t NEVER = ~0ull;

struct RunOpts {
  uint32_t ready_pct = 100;   // lane ready probability
  uint32_t gap_pct   = 0;     // probability of holding a beat back for a cycle
  // a block's lifetime in cycles, from the cycle it leaves the unit's lane
  // (blocks without cluster-barrier phases)
  std::function<uint64_t(int gang_seq, uint32_t rank, uint32_t core)> life;
  // cluster-barrier phases of a SPREAD gang of K > 1 (none by default)
  std::function<uint32_t(int gang_seq)> phases;
  // a block's arrival at a phase, in cycles from the cycle it left the unit's
  // lane (phase 0) or from the cycle the previous release left the unit
  std::function<uint64_t(int gang_seq, uint32_t rank, uint32_t phase)> gap;
  // a block's CTA_DONE, in cycles from its last arrival
  std::function<uint64_t(int gang_seq, uint32_t rank)> done_gap;
};

// A core's end of the gang unit's release link.
class ReleaseSink : public SimObject<ReleaseSink> {
public:
  SimEventLink<GbarResume> in;
  std::vector<std::pair<uint64_t, GbarResume>> log;

  ReleaseSink(const SimContext& ctx, const char* name)
    : SimObject<ReleaseSink>(ctx, name)
    , in(this) {
    in.bind(this, &ReleaseSink::on_release);
  }

  void on_release(const GbarResume& msg) {
    log.push_back({SimPlatform::instance().cycles(), msg});
  }

  friend class SimObject<ReleaseSink>;
};

class Bench {
public:
  Bench(const ref::Cfg& cfg, uint32_t seed) : cfg_(cfg), rng_(seed) {}

  std::vector<GangRec> gangs;
  std::vector<CbarRec> cbars;   // in release order
  std::vector<std::pair<uint64_t, uint64_t>> legacy_lat;   // (in fire, lane take)
  uint64_t cycles = 0;
  uint32_t errors = 0;

  bool run(const std::vector<Launch>& launches, const RunOpts& o) {
    auto& platform = SimPlatform::instance();
    platform.set_num_workers(1);
    bool ok;
    {
      auto gang = CtaGang::Create("cta-gang", cfg_);
      std::vector<ReleaseSink::Ptr> sinks;
      for (uint32_t c = 0; c < cfg_.ncc; ++c) {
        std::string name = "release-sink" + std::to_string(c);
        sinks.push_back(ReleaseSink::Create(name.c_str()));
        gang->release_out.at(c).bind(&sinks.back()->in);
      }
      platform.reset();
      ok = this->loop(*gang, sinks, launches, o);
    }
    platform.finalize();
    return ok;
  }

private:
  // The cluster-barrier progress of a gang.
  struct Phase {
    uint32_t k;
    uint32_t gang_id;
    uint32_t epoch;
    uint32_t phases;
    uint32_t phase = 0;       // releases received
    uint32_t left = 0;        // releases that left the unit
    uint32_t arrived = 0;     // arrivals presented at the current phase
    uint64_t arrive_cycle = 0;
    bool     pending = false; // all arrived, the release has not left
    // releases that left and have not reached the cores: {arrive, leave} cycles
    std::deque<std::pair<uint64_t, uint64_t>> inflight;
  };

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

  void check_out(const kmu_req_t& got, const Expect& e) {
    if (same_req(got, e.req))
      return;
    if (got.pinned != e.req.pinned || got.dest_core != e.req.dest_core || got.forced_slot != e.req.forced_slot
     || got.gang_id != e.req.gang_id || got.epoch != e.req.epoch || !same_row(got.row, e.req.row)) {
      this->fail("output beat (gang %d rank %u): pin %d core %u slot %u gang %u epoch %u row %s;"
                 " model: pin %d core %u slot %u gang %u epoch %u row %s",
                 e.gang_seq, e.rank, int(got.pinned), got.dest_core, got.forced_slot, got.gang_id, got.epoch,
                 row_str(got.row).c_str(), int(e.req.pinned), e.req.dest_core, e.req.forced_slot, e.req.gang_id,
                 e.req.epoch, row_str(e.req.row).c_str());
    } else {
      this->fail("output beat (gang %d rank %u): payload differs from the input beat", e.gang_seq, e.rank);
    }
  }

  // Every core receives the release once, in the cycle the model predicts.
  void check_releases(std::vector<ReleaseSink::Ptr>& sinks, std::map<uint64_t, std::pair<uint32_t, uint32_t>>& expect,
                      std::map<int, Phase>& phases, uint64_t delay, bool* progress) {
    auto it = expect.find(cycles);
    for (uint32_t c = 0; c < cfg_.ncc; ++c) {
      auto& log = sinks[c]->log;
      uint32_t n = 0;
      for (const auto& e : log) {
        if (e.first != cycles) {
          this->fail("core %u got a release in cycle %lu, read in cycle %lu", c, (unsigned long)e.first,
                     (unsigned long)cycles);
          continue;
        }
        ++n;
        if (it == expect.end()) {
          this->fail("core %u got release {gang %u, epoch %u}, the model sent none", c, e.second.gang_id,
                     e.second.epoch);
        } else if (e.second.kind != GBAR_KIND_CBAR || e.second.gang_id != it->second.first
                || e.second.epoch != it->second.second) {
          this->fail("core %u got release {kind %u, gang %u, epoch %u}, the model sent {gang %u, epoch %u}", c,
                     e.second.kind, e.second.gang_id, e.second.epoch, it->second.first, it->second.second);
        }
      }
      if (it != expect.end() && n != 1) {
        this->fail("core %u got %u copies of release {gang %u, epoch %u}", c, n, it->second.first, it->second.second);
      }
      log.clear();
    }
    if (it == expect.end())
      return;
    *progress = true;
    uint32_t id = it->second.first, ep = it->second.second;
    expect.erase(it);
    int seq = -1;
    for (auto& p : phases) {
      if (!p.second.inflight.empty() && p.second.gang_id == id && p.second.epoch == ep) {
        seq = p.first;
      }
    }
    if (seq < 0) {
      this->fail("release {gang %u, epoch %u} for no gang with a released phase", id, ep);
      return;
    }
    Phase& ph = phases[seq];
    auto rel = ph.inflight.front();
    ph.inflight.pop_front();
    if (rel.second + delay != cycles) {
      this->fail("release {gang %u, epoch %u} left in cycle %lu, reached the cores in cycle %lu", id, ep,
                 (unsigned long)rel.second, (unsigned long)cycles);
    }
    cbars.push_back({seq, ph.phase, id, ep, rel.first, rel.second});
    cbars_left_[seq] = rel.second;
    ++ph.phase;
  }

  bool loop(CtaGang& gang, std::vector<ReleaseSink::Ptr>& sinks, const std::vector<Launch>& launches,
            const RunOpts& o) {
    auto& platform = SimPlatform::instance();
    const uint32_t NCC = cfg_.ncc;
    // the gbar unit's response register, then one fork stage per fan-out level
    const uint64_t delay = 1 + (cfg_.num_sockets > 1 ? 1 : 0) + (cfg_.socket_size > 1 ? 1 : 0);
    if (gang.release_delay() != delay) {
      this->fail("release delay %lu cycles, the RTL path takes %lu", (unsigned long)gang.release_delay(),
                 (unsigned long)delay);
      return false;
    }
    GangModel model(cfg_, gang.bus_out.capacity());
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

    struct Evt {
      DoneEvt  ev;
      int      seq;
      uint32_t rank;
    };
    std::map<int, size_t> gang_index;
    std::map<int, Phase> phases;
    std::vector<Resident> res;
    std::deque<Evt> done_q;
    std::vector<uint64_t> occ(NCC, 0);
    std::vector<uint32_t> warps(NCC, 0);
    std::map<uint32_t, std::pair<int, uint32_t>> live_ids;   // id -> (gang_seq, members left)
    std::map<uint32_t, int> id_holder;                       // id -> its last gang
    std::map<uint64_t, std::pair<uint32_t, uint32_t>> rel_expect;   // delivery cycle -> {id, epoch}
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
        bool drained = res.empty() && done_q.empty() && rel_expect.empty() && !busy_prev;
        if (drained) {
          ++cur_launch;
          if (cur_launch >= launches.size())
            break;
        }
      }
      // events due this cycle, in the order the blocks left the unit; one
      // presented per cycle
      for (auto it = res.begin(); it != res.end();) {
        if (it->phase < it->phases && it->arrive_at <= cycles) {
          DoneEvt a{it->core, it->slot, it->gang, it->epoch, true, this->k_of(it->gang_seq)};
          done_q.push_back({a, it->gang_seq, it->rank});
          ++it->phase;
          it->arrive_at = NEVER;
          if (it->phase == it->phases) {
            it->done_at = cycles + o.done_gap(it->gang_seq, it->rank);
          }
          ++it;
          continue;
        }
        if (it->phase == it->phases && it->done_at <= cycles) {
          done_q.push_back({DoneEvt{it->core, it->slot, it->gang, it->epoch}, it->gang_seq, it->rank});
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
        Evt e = done_q.front();
        done_q.pop_front();
        dev = e.ev;
        GangEvent ge;
        ge.kind    = dev.cbar ? GangEvent::CBAR : GangEvent::CTA_DONE;
        ge.core    = dev.core;
        ge.slot    = dev.slot;
        ge.gang_id = dev.gang;
        ge.epoch   = dev.epoch;
        ge.count   = dev.count;
        gang.cta_event(ge);
        if (dev.cbar) {
          Phase& ph = phases.at(e.seq);
          if (ph.pending) {
            this->fail("gang %d rank %u arrived before its phase %u was released", e.seq, e.rank, ph.left);
          }
          if (++ph.arrived == ph.k) {
            ph.arrived = 0;
            ph.arrive_cycle = cycles;
            ph.pending = true;
          }
        } else {
          gangs[gang_index[e.seq]].done_pres[e.rank] = cycles;
        }
      }

      if (!head_valid && cur_launch < launches.size() && !beats[cur_launch].empty()
       && (o.gap_pct == 0 || (rng_() % 100) >= o.gap_pct)) {
        head = beats[cur_launch].front();
        beats[cur_launch].pop_front();
        head_valid = true;
        gang.bus_in.send(head.req, 0);
      }

      // the output lane: a beat visible before the tick is taken before it
      bool out_fire = false;
      bool had = !gang.bus_out.empty();
      if (had == model.lane_empty()) {
        this->fail("the lane holds %s beat, the model says %s", had ? "a" : "no", had ? "none" : "one");
        return false;
      }
      bool rdy = (o.ready_pct >= 100) || ((rng_() % 100) < o.ready_pct);
      kmu_req_t beat{};
      Expect e{};
      if (had && rdy) {
        beat = gang.bus_out.peek();
        gang.bus_out.pop();
        gang.credit_in.send(1, 0);
        out_fire = true;
        model.take(&e);
        this->check_out(beat, e);
      }
      uint32_t c = beat.dest_core;
      if (out_fire && e.req.pinned) {
        // the bench's own view: the block is now resident on the core it is routed to
        uint32_t s = beat.forced_slot;
        if (c >= NCC || c != e.req.dest_core) {
          this->fail("a beat routed to core %u, the model placed it on core %u (of %u)", c, e.req.dest_core, NCC);
          return false;
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
          g.done_pres.assign(e.req.row.k, 0);
          g.commit_cycle = e.commit_cycle;
          gang_index[e.gang_seq] = gangs.size();
          gangs.push_back(g);
          git = gang_index.find(e.gang_seq);
          if (g.k > 1) {
            // an id is handed out again only once its last release left
            auto hit = id_holder.find(g.gang_id);
            if (hit != id_holder.end()) {
              const Phase& old = phases.at(hit->second);
              uint64_t left = old.inflight.empty() ? (old.phase ? cbars_left_[hit->second] : 0) : old.inflight.back().second;
              if (old.pending || ((old.phase || !old.inflight.empty()) && g.commit_cycle <= left)) {
                this->fail("gang id %u handed to gang %d (commit %lu) before the release of gang %d left", g.gang_id,
                           e.gang_seq, (unsigned long)g.commit_cycle, hit->second);
              }
            }
            id_holder[g.gang_id] = e.gang_seq;
            Phase ph;
            ph.k = g.k;
            ph.gang_id = g.gang_id;
            ph.epoch = g.epoch;
            ph.phases = o.phases ? o.phases(e.gang_seq) : 0;
            phases[e.gang_seq] = ph;
          }
        }
        GangRec& g = gangs[git->second];
        g.members[e.rank] = {c, s};
        g.out_cycle[e.rank] = cycles;
        ks_[e.gang_seq] = e.req.row.k;
        uint32_t P = (g.k > 1) ? phases.at(e.gang_seq).phases : 0;
        res.push_back({e.gang_seq, e.rank, c, s, e.req.gang_id, e.req.epoch, e.w,
                       (P > 0) ? NEVER : cycles + o.life(e.gang_seq, e.rank, c), P, 0,
                       (P > 0) ? cycles + o.gap(e.gang_seq, e.rank, 0) : NEVER});
      } else if (out_fire) {
        legacy_lat.push_back({e.in_cycle, cycles});
      }


      bool busy = gang.busy();
      if (busy != model.busy(head_valid)) {
        this->fail("busy=%d, the model says %d", int(busy), int(!busy));
      }
      busy_prev = busy || had;

      std::string err;
      int rel = -1;
      uint32_t rel_ep = 0;
      int want = model.step(cycles, head_valid ? &head : nullptr, dvalid ? &dev : nullptr, err, &rel, &rel_ep);
      if (!err.empty()) {
        this->fail("%s", err.c_str());
        return false;
      }
      if (rel >= 0) {
        rel_expect[cycles + delay] = {uint32_t(rel), rel_ep};
        // the blocks' next arrivals count from the cycle the release leaves;
        // the cycle it reaches the cores is checked against this one
        bool found = false;
        for (auto& p : phases) {
          if (p.second.pending && p.second.gang_id == uint32_t(rel) && p.second.epoch == rel_ep) {
            p.second.pending = false;
            p.second.inflight.push_back({p.second.arrive_cycle, cycles});
            found = true;
            uint32_t next = ++p.second.left;
            for (auto& r : res) {
              if (r.gang_seq == p.first && r.phase == next && r.phase < r.phases) {
                r.arrive_at = cycles + o.gap(p.first, r.rank, r.phase);
              }
            }
          }
        }
        if (!found) {
          this->fail("the model releases {gang %d, epoch %u}, which no gang completed", rel, rel_ep);
        }
      }

      platform.tick();
      // the releases the sinks took at the start of this cycle's tick
      bool rel_progress = false;
      this->check_releases(sinks, rel_expect, phases, delay, &rel_progress);

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

      idle = (in_fire || out_fire || dvalid || rel_progress) ? 0 : idle + 1;
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
    for (const auto& p : phases) {
      if (p.second.pending || !p.second.inflight.empty() || p.second.phase != p.second.phases) {
        this->fail("gang %d passed %u of its %u cluster-barrier phases", p.first, p.second.phase, p.second.phases);
      }
    }
    if (!rel_expect.empty()) {
      this->fail("%zu releases never reached the cores", rel_expect.size());
    }
    if (gang.busy()) {
      this->fail("%s", "the unit is still busy at the end");
    }
    return errors == 0;
  }

  ref::Cfg cfg_;
  std::mt19937 rng_;
  std::map<int, uint32_t> ks_;
  std::map<int, uint64_t> cbars_left_;   // gang -> the cycle its last delivered release left
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
  // cluster-barrier phases, so arrivals, releases and done events interleave
  o.phases = [seed](int seq) { return uint32_t((uint32_t(seq) * 2654435761u + seed) >> 7) % 4; };
  std::mt19937 grng(seed * 13 + 5);
  o.gap = [&grng](int, uint32_t, uint32_t) { return uint64_t(1 + grng() % 8); };
  o.done_gap = [&grng](int, uint32_t) { return uint64_t(1 + grng() % 30); };
  bool ok = b.run(ls, o);
  std::printf("random: %u launches, %zu gangs, %zu cluster-barrier phases, %lu cycles, ready %u%%, gaps %u%%\n",
              num_launches, b.gangs.size(), b.cbars.size(), (unsigned long)b.cycles, o.ready_pct, o.gap_pct);
  return report("random", ok);
}

// Gangs pass three cluster-barrier phases each; with the release port always
// ready, every release leaves in the cycle after the phase's last arrival.
bool t_cbar_phases(const ref::Cfg& cfg) {
  if (cfg.max_cluster < 2) {
    std::printf("cluster-barrier phases: needs MAX_CLUSTER >= 2, skipped\n");
    return true;
  }
  uint32_t K = cfg.max_cluster;
  uint32_t G = 2 * cfg.ncc * cfg.nw / K + 2;   // more gangs than fit: ids are reused
  Bench b(cfg, 91);
  RunOpts o = fixed_life(4);
  o.phases = [](int) { return 3u; };
  o.gap = [](int seq, uint32_t rank, uint32_t phase) { return uint64_t(1 + (seq + 3 * rank + 5 * phase) % 7); };
  o.done_gap = [](int seq, uint32_t rank) { return uint64_t(1 + (seq + rank) % 4); };
  bool ok = b.run({{true, K, 1, 0, G}}, o);
  ok = ok && (b.cbars.size() == 3 * G);
  std::map<int, uint32_t> next;
  for (const auto& c : b.cbars) {
    const GangRec* g = nullptr;
    for (const auto& r : b.gangs) {
      if (r.gang_seq == c.gang_seq)
        g = &r;
    }
    if (!g || c.phase != next[c.gang_seq]++ || c.gang_id != g->gang_id || c.epoch != g->epoch
     || c.release_cycle != c.arrive_cycle + 1) {
      std::printf("FAIL cbar-phases: gang %d phase %u {id %u epoch %u} arrived %lu, released %lu\n", c.gang_seq,
                  c.phase, c.gang_id, c.epoch, (unsigned long)c.arrive_cycle, (unsigned long)c.release_cycle);
      ok = false;
      break;
    }
  }
  return report("cluster-barrier phases", ok);
}

// A gang's last done event lands in the cycle its release leaves: its id is
// not free in that cycle, so the gang waiting for an id starts one cycle later.
bool t_cbar_id_hold(const ref::Cfg& cfg) {
  if (cfg.max_cluster < 2 || 2 * cfg.num_gangs > cfg.ncc * cfg.nw) {
    std::printf("gang id held by a pending release: needs MAX_CLUSTER >= 2 and 2 slots per gang id, skipped\n");
    return true;
  }
  const uint32_t K = 2;
  const int G = int(cfg.num_gangs);   // gangs 0..G-1 hold every id; gang G waits for one
  Bench b(cfg, 95);
  RunOpts o;
  // gang 0's two blocks arrive late, after every gang is placed, and the last
  // done follows the last arrival; the others hold their ids
  o.life = [G](int seq, uint32_t, uint32_t) { return uint64_t(seq == G ? 10 : 2000); };
  o.phases = [](int seq) { return seq == 0 ? 1u : 0u; };
  o.gap = [](int, uint32_t rank, uint32_t) { return uint64_t(rank == 0 ? 204 : 208); };
  o.done_gap = [](int, uint32_t rank) { return uint64_t(rank == 0 ? 3 : 1); };
  bool ok = b.run({{true, K, 1, 0, uint32_t(G) + 1}}, o);
  ok = ok && (b.gangs.size() == uint32_t(G) + 1) && (b.cbars.size() == 1);
  if (ok) {
    const CbarRec& c = b.cbars[0];
    const GangRec& g0 = b.gangs[0];
    const GangRec& w = b.gangs[G];
    uint64_t last_done = std::max(g0.done_pres[0], g0.done_pres[1]);
    if (last_done != c.release_cycle) {
      std::printf("FAIL id-hold: gang 0's last done presented in cycle %lu, its release left in cycle %lu\n",
                  (unsigned long)last_done, (unsigned long)c.release_cycle);
      ok = false;
    }
    if (w.gang_id != g0.gang_id || w.out_cycle[0] != c.release_cycle + 1 + K + 2) {
      std::printf("FAIL id-hold: gang %d got id %u, first beat in cycle %lu; want id %u in cycle %lu\n", G,
                  w.gang_id, (unsigned long)w.out_cycle[0], g0.gang_id,
                  (unsigned long)(c.release_cycle + 1 + K + 2));
      ok = false;
    }
  }
  return report("gang id held by a pending release", ok);
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

// The RTL bench's cluster-barrier schedule.
bool t_schedule_cbar(const ref::Cfg& cfg, const std::string& csv) {
  const uint32_t MAXC = cfg.max_cluster;
  uint32_t G = (MAXC > 1) ? 2 * cfg.ncc * cfg.nw / MAXC + 2 : 0;
  Bench b(cfg, 94);
  RunOpts o;
  o.life = [](int, uint32_t, uint32_t) { return uint64_t(1); };
  o.phases = [](int s) { return uint32_t(1 + s % 3); };
  o.gap = [](int s, uint32_t r, uint32_t i) {
    return (i == 0) ? uint64_t(3 + (5 * s + 3 * r) % 11) : uint64_t(2 + (3 * s + 5 * r + i) % 7);
  };
  o.done_gap = [](int s, uint32_t r) { return uint64_t(1 + (7 * s + 5 * r) % 9); };
  bool ok = (G == 0) || b.run({{true, MAXC, 1, 0, G}}, o);
  std::vector<CbarRec> rows = b.cbars;
  std::sort(rows.begin(), rows.end(), [](const CbarRec& x, const CbarRec& y) {
    return std::make_tuple(x.release_cycle, x.gang_id, x.epoch, x.arrive_cycle)
         < std::make_tuple(y.release_cycle, y.gang_id, y.epoch, y.arrive_cycle);
  });
  if (ok && !csv.empty()) {
    FILE* f = std::fopen(csv.c_str(), "w");
    if (!f) {
      std::printf("FAIL cbar schedule: cannot write %s\n", csv.c_str());
      return false;
    }
    std::fprintf(f, "gang,epoch,arrive_cycle,release_cycle\n");
    for (const auto& c : rows) {
      std::fprintf(f, "%u,%u,%lu,%lu\n", c.gang_id, c.epoch, (unsigned long)c.arrive_cycle,
                   (unsigned long)c.release_cycle);
    }
    std::fclose(f);
    std::printf("cbar schedule: %u gangs, %zu releases, %lu cycles, wrote %s\n", G, rows.size(),
                (unsigned long)b.cycles, csv.c_str());
  }
  return report("deterministic cluster-barrier schedule", ok);
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
  std::string cbar_csv;
  ref::Cfg cfg = dlmem_cfg();
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--seed=", 7) == 0) {
      seed = uint32_t(std::strtoul(argv[i] + 7, nullptr, 0));
    } else if (std::strncmp(argv[i], "--launches=", 11) == 0) {
      launches = uint32_t(std::strtoul(argv[i] + 11, nullptr, 0));
    } else if (std::strncmp(argv[i], "--csv=", 6) == 0) {
      csv = argv[i] + 6;
    } else if (std::strncmp(argv[i], "--cbar-csv=", 11) == 0) {
      cbar_csv = argv[i] + 11;
    } else if (std::strncmp(argv[i], "--shape=", 8) == 0 && parse_shape(argv[i] + 8, &cfg)) {
      continue;
    } else {
      std::printf("usage: %s [--seed=N] [--launches=N] [--csv=FILE] [--cbar-csv=FILE] [--shape=CORES,SOCKET_SIZE,WARPS,MAX_CLUSTER[,LMEM_LOG]]\n",
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
  ok = t_cbar_phases(cfg) && ok;
  ok = t_cbar_id_hold(cfg) && ok;
  ok = t_random(cfg, seed, launches) && ok;
  ok = t_schedule(cfg, csv) && ok;
  ok = t_schedule_cbar(cfg, cbar_csv) && ok;

  if (!ok) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
