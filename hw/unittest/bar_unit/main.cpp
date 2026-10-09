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

// CTA_DONE events share each core's global-barrier request queue. The bench
// drives one hardware cluster's barrier units with done events, global barrier
// rounds and local barrier traffic that stalls the barrier responses, and checks
// at the gang unit's done port that every event arrives exactly once, in its
// core's order, with its core index; that an event never enters a queue in the
// cycle a barrier arrival does; and that every barrier round releases each core
// once, only after all cores arrived. It first measures the done latency on an
// idle cluster: from the cycle an event enters its core's queue to the cycle it
// reaches the gang unit.
//
// Then cluster barriers (the per-slot state beside the barrier store): gangs
// whose blocks send one CBAR per phase to the gang unit and resume on its
// release, clusters completed inside one core, split arrive/wait with the
// returned phase checked, __syncthreads between arrive and wait, releases that
// must match no slot (a wrong epoch, an idle gang, a slot admitted again), and
// all of it at once with done events and global and local barriers. A warp may
// leave a cluster barrier only once every warp of every block of its cluster has
// arrived (and, for a gang, after the gang's release); both latency legs are
// measured on an idle cluster.

#include "vl_simulator.h"
#include "VVX_bar_unit_top.h"
#include "VX_config.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
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
  if (timestamp >= trace_start_time && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable(bool enable) {
  trace_enabled = enable;
}

using Sim = vl_simulator<VVX_bar_unit_top>;

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

constexpr uint32_t SS        = VX_CFG_SOCKET_SIZE;
constexpr uint32_t NS        = c_up(VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE);
constexpr uint32_t NCC       = NS * SS;
constexpr uint32_t NW        = VX_CFG_NUM_WARPS;
constexpr uint32_t NB        = VX_CFG_NUM_BARRIERS;
constexpr uint32_t NW_W      = c_up(c_clog2(NW));
constexpr uint32_t NB_W      = c_up(c_clog2(NB));
constexpr uint32_t NC_W      = c_up(c_clog2(VX_CFG_NUM_CORES));
constexpr uint32_t CORE_W    = c_up(c_clog2(NCC));
constexpr uint32_t GANGS     = std::max<uint32_t>(2, NCC * NW / 2);
constexpr uint32_t GANG_W    = c_clog2(GANGS);
constexpr uint32_t BAR_SIZE_W = std::max(std::max(NW_W, NC_W), c_clog2(VX_CFG_MAX_BAR_EVENTS));
constexpr uint32_t BAR_ADDR_W = c_up(c_clog2(NW) + c_clog2(NB));
constexpr uint32_t EVT_BITS  = NW_W + GANG_W + 1;
constexpr uint32_t NB_BITS   = c_clog2(NB);
constexpr uint32_t MAXC      = VX_CFG_DLMEM_MAX_CLUSTER;
constexpr uint64_t NEVER     = ~0ull;
constexpr uint32_t KIND_DONE = 1;   // GBAR_KIND_CTA_DONE
constexpr uint32_t KIND_CBAR = 2;   // GBAR_KIND_CBAR

static_assert(NB >= 2, "the bench needs a local barrier id besides the global ones");

///////////////////////////////////////////////////////////////////////////////
// Packed-array port access: Verilator picks the C type from the width.

template <typename T>
static typename std::enable_if<std::is_integral<T>::value>::type
put(T& v, uint32_t i, uint32_t w, uint64_t x) {
  uint64_t m = ((w >= 64) ? ~0ull : ((1ull << w) - 1)) << (i * w);
  v = T((uint64_t(v) & ~m) | ((x << (i * w)) & m));
}

template <std::size_t N>
static void put(VlWide<N>& v, uint32_t i, uint32_t w, uint64_t x) {
  for (uint32_t b = 0; b < w; ++b) {
    uint32_t p = i * w + b;
    if ((x >> b) & 1) {
      v[p / 32] |= 1u << (p % 32);
    } else {
      v[p / 32] &= ~(1u << (p % 32));
    }
  }
}

template <typename T>
static typename std::enable_if<std::is_integral<T>::value, uint64_t>::type
get(const T& v, uint32_t i, uint32_t w) {
  return (uint64_t(v) >> (i * w)) & ((w >= 64) ? ~0ull : ((1ull << w) - 1));
}

template <std::size_t N>
static uint64_t get(const VlWide<N>& v, uint32_t i, uint32_t w) {
  uint64_t r = 0;
  for (uint32_t b = 0; b < w; ++b) {
    uint32_t p = i * w + b;
    r |= uint64_t((v[p / 32] >> (p % 32)) & 1) << b;
  }
  return r;
}

///////////////////////////////////////////////////////////////////////////////

struct Event {
  uint32_t code;    // per-core sequence number, spread over {epoch, gang, slot}
  uint64_t fired;   // cycle it entered its core's queue
};

enum OpKind { NONE, GLOBAL, LOCAL };

struct Op {
  OpKind   kind  = NONE;
  uint32_t round = 0;
};

struct Opts {
  uint32_t events_per_core = 0;
  uint32_t rounds          = 0;
  uint32_t event_pct       = 0;   // new event per core per cycle
  uint32_t arrive_pct      = 0;   // a waiting-free warp 0 arrives
  uint32_t local_pct       = 0;   // local barrier traffic, which stalls responses
  bool     isolated        = false;   // one event in flight in the whole cluster
};

struct Stats {
  uint64_t cycles      = 0;
  uint32_t delivered   = 0;
  uint32_t collisions  = 0;   // an event presented in the cycle of a barrier arrival
  uint32_t full_waits  = 0;   // an event held back with no arrival: the queue was full
  uint32_t releases    = 0;
  uint64_t rsp_stalls  = 0;
  uint64_t lat_min     = ~0ull;
  uint64_t lat_max     = 0;
};

static uint32_t global_id(uint32_t round) {
  return round % (NB - 1);
}

class Bench {
public:
  Bench(Sim& sim, uint64_t& tick, uint32_t seed) : sim_(sim), tick_(tick), rng_(seed) {}

  bool run(const Opts& o, Stats& st) {
    std::vector<std::deque<Event>> todo(NCC), sent(NCC);
    std::vector<uint32_t> made(NCC, 0);
    std::vector<Op> cur(NCC), next(NCC);
    std::vector<uint32_t> round(NCC, 0);          // warp 0's next round
    std::vector<bool> waiting(NCC, false);        // warp 0 arrived, not released
    std::vector<uint32_t> arrived(o.rounds + 1, 0);
    std::vector<uint32_t> released(o.rounds + 1, 0);
    uint32_t in_flight = 0;
    uint64_t idle = 0;
    uint32_t errors = 0;
    auto fail = [&](const char* fmt, auto... args) {
      if (++errors <= 10) {
        std::printf("FAIL cycle %lu: ", (unsigned long)st.cycles);
        std::printf(fmt, args...);
        std::printf("\n");
      }
    };

    this->clear();
    tick_ = sim_.reset(tick_);
    st = Stats();
    bool global = (NCC > 1) && (o.rounds > 0);

    while (true) {
      // new events
      for (uint32_t c = 0; c < NCC; ++c) {
        bool want = (made[c] < o.events_per_core)
                 && (o.isolated ? (in_flight == 0 && (rng_() % NCC) == c) : (rng_() % 100) < o.event_pct);
        if (want) {
          todo[c].push_back({made[c]++, 0});
          if (o.isolated) {
            ++in_flight;
          }
        }
      }
      // the barrier requests of this cycle were planned last cycle; plan the next
      bool work = false;
      for (uint32_t c = 0; c < NCC; ++c) {
        work = work || made[c] < o.events_per_core || !todo[c].empty() || !sent[c].empty()
            || (global && round[c] < o.rounds);
      }
      for (uint32_t c = 0; c < NCC; ++c) {
        cur[c] = next[c];
        next[c] = Op();
        bool can_arrive = global && !waiting[c] && cur[c].kind != GLOBAL && round[c] < o.rounds;
        if (can_arrive && (rng_() % 100) < o.arrive_pct) {
          next[c].kind = GLOBAL;
          next[c].round = round[c];
          waiting[c] = true;
          // half the arrivals meet a done event in the same cycle
          if ((rng_() & 1) && made[c] < o.events_per_core && todo[c].empty()) {
            todo[c].push_back({made[c]++, 0});
          }
        } else if (work && (rng_() % 100) < o.local_pct) {
          next[c].kind = LOCAL;
        }
      }
      this->drive(cur, next, todo);
      sim_->eval();

      // CTA_DONE intake
      for (uint32_t c = 0; c < NCC; ++c) {
        if (todo[c].empty()) {
          continue;
        }
        bool ready = get(sim_->done_in_ready, c, 1);
        if (cur[c].kind == GLOBAL) {
          ++st.collisions;
          if (ready) {
            fail("core %u took a done event in the cycle of a barrier arrival", c);
          }
        } else if (!ready) {
          ++st.full_waits;
        }
        if (ready) {
          Event e = todo[c].front();
          todo[c].pop_front();
          e.fired = st.cycles;
          sent[c].push_back(e);
        }
      }
      // the gang unit's event port
      bool progress = false;
      if (sim_->evt_valid) {
        uint32_t c = sim_->evt_core;
        uint32_t code = uint32_t(sim_->evt_slot) | (uint32_t(sim_->evt_gang) << NW_W)
                      | (uint32_t(sim_->evt_epoch) << (NW_W + GANG_W));
        if (sim_->evt_kind != KIND_DONE) {
          fail("event of kind %u from core %u, only done events were sent", uint32_t(sim_->evt_kind), c);
        } else if (c >= NCC || sent[c].empty()) {
          fail("done event from core %u, which sent none", c);
        } else {
          const Event& e = sent[c].front();
          uint32_t want = e.code & ((1u << EVT_BITS) - 1);
          if (code != want) {
            fail("core %u: done event %u out of order, expected %u", c, code, want);
          }
          uint64_t lat = st.cycles - e.fired;
          st.lat_min = std::min(st.lat_min, lat);
          st.lat_max = std::max(st.lat_max, lat);
          sent[c].pop_front();
          ++st.delivered;
          if (o.isolated) {
            --in_flight;
          }
        }
        progress = true;
      }
      // barrier releases
      for (uint32_t c = 0; c < NCC; ++c) {
        if (!get(sim_->unlock_valid, c, 1) || !(get(sim_->unlock_mask, c, NW) & 1)) {
          continue;
        }
        if (!waiting[c] || cur[c].kind == GLOBAL) {
          fail("core %u released while not waiting on a global barrier", c);
          continue;
        }
        uint32_t r = round[c];
        if (arrived[r] != NCC) {
          fail("core %u released from round %u after %u of %u arrivals", c, r, arrived[r], NCC);
        }
        ++released[r];
        ++st.releases;
        waiting[c] = false;
        ++round[c];
        progress = true;
      }
      for (uint32_t c = 0; c < NCC; ++c) {
        if (cur[c].kind == GLOBAL) {
          ++arrived[cur[c].round];
        }
      }
      for (uint32_t c = 0; c < NCC; ++c) {
        st.rsp_stalls += get(sim_->gbar_rsp_stall, c, 1);
      }

      tick_ = sim_.step(tick_, 2);
      ++st.cycles;

      bool more = work;
      for (uint32_t c = 0; c < NCC; ++c) {
        more = more || next[c].kind != NONE;
      }
      if (!more) {
        break;
      }
      idle = progress ? 0 : idle + 1;
      if (idle > 2000) {
        fail("no done event or release for 2000 cycles");
        break;
      }
      if (errors > 10) {
        break;
      }
    }
    for (uint32_t r = 0; global && r < o.rounds; ++r) {
      if (released[r] != NCC) {
        fail("round %u released %u of %u cores", r, released[r], NCC);
      }
    }
    if (st.delivered != NCC * o.events_per_core) {
      fail("%u of %u done events delivered", st.delivered, NCC * o.events_per_core);
    }
    return errors == 0;
  }

private:
  void clear() {
    sim_->req_valid = 0;
    sim_->req_is_global = 0;
    sim_->req_is_arrive = 0;
    sim_->req_is_sync = 0;
    sim_->req_is_cluster = 0;
    sim_->req_phase = 0;
    sim_->read_cluster = 0;
    sim_->cbar_wr_valid = 0;
    sim_->rel_valid = 0;
    sim_->done_in_valid = 0;
    sim_->done_in_epoch = 0;
    for (uint32_t c = 0; c < NCC; ++c) {
      put(sim_->req_wid, c, NW_W, 0);
      put(sim_->req_id, c, NB_W, 0);
      put(sim_->req_size_m1, c, BAR_SIZE_W, 0);
      put(sim_->read_addr, c, BAR_ADDR_W, 0);
      put(sim_->active_warps, c, NW, 1);   // warp 0 alone takes part in global barriers
      put(sim_->done_in_slot, c, NW_W, 0);
      put(sim_->done_in_gang, c, std::max<uint32_t>(GANG_W, 1), 0);
    }
  }

  void drive(const std::vector<Op>& cur, const std::vector<Op>& next, const std::vector<std::deque<Event>>& todo) {
    for (uint32_t c = 0; c < NCC; ++c) {
      const Op& op = cur[c];
      bool g = (op.kind == GLOBAL);
      put(sim_->req_valid, c, 1, op.kind != NONE);
      put(sim_->req_is_global, c, 1, g);
      put(sim_->req_is_arrive, c, 1, op.kind != NONE);
      put(sim_->req_is_sync, c, 1, g);
      // a global arrival names the cluster's core count; a local one pairs warp 1 with itself
      put(sim_->req_size_m1, c, BAR_SIZE_W, g ? (NCC - 1) : 1);
      put(sim_->req_wid, c, NW_W, g ? 0 : (NW > 1 ? 1 : 0));
      put(sim_->req_id, c, NB_W, g ? global_id(op.round) : (NB - 1));
      // the barrier store is read a cycle ahead of the request
      const Op& nx = next[c];
      put(sim_->read_addr, c, BAR_ADDR_W, (nx.kind == GLOBAL) ? global_id(nx.round) : (NB - 1));

      bool v = !todo[c].empty();
      uint32_t code = v ? todo[c].front().code : 0;
      put(sim_->done_in_valid, c, 1, v);
      put(sim_->done_in_slot, c, NW_W, code & ((1u << NW_W) - 1));
      if (GANG_W > 0) {
        put(sim_->done_in_gang, c, GANG_W, (code >> NW_W) & ((1u << GANG_W) - 1));
      }
      put(sim_->done_in_epoch, c, 1, (code >> (NW_W + GANG_W)) & 1);
    }
  }

  Sim& sim_;
  uint64_t& tick_;
  std::mt19937 rng_;
};

///////////////////////////////////////////////////////////////////////////////
// Cluster barriers. The bench plays every core's warps (one barrier request
// per core and cycle, its read a cycle ahead), the dispatcher's slot records
// and the gang unit: it counts a gang's block arrivals on the event port and
// answers the K-th with a release on the release port.

struct SlotInfo {
  bool     gang    = false;
  uint32_t gang_id = 0;
  uint32_t epoch   = 0;
  uint32_t base    = 0;
  uint32_t k       = 1;
};

enum COpT { C_SYNC, C_ARRIVE, C_WAIT, C_LBAR, C_GLOBAL, C_DELAY, C_ADMIT, C_HOLD_REL };

struct COp {
  COpT     t;
  uint32_t slot = 0;
  uint32_t n    = 0;    // DELAY, HOLD_REL: cycles
  SlotInfo info;        // ADMIT: the slot's new record
  int      cl   = -1;   // ADMIT: the cluster the slot joins; HOLD_REL: the cluster released
  uint32_t ph   = 0;    // HOLD_REL: the phase released
};

static COp cop(COpT t, uint32_t slot = 0, uint32_t n = 0) {
  COp o;
  o.t = t;
  o.slot = slot;
  o.n = n;
  return o;
}

struct Cluster {
  bool     gang    = false;
  uint32_t gang_id = 0;
  uint32_t epoch   = 0;
  uint32_t K       = 1;
  uint32_t W       = 1;
  bool     bogus   = false;   // two releases that match no pending slot go first
  std::map<uint32_t, uint32_t> arr;                    // phase -> arrivals issued
  std::map<uint32_t, uint64_t> last_req;               // phase -> request cycle of its last arrival
  std::map<uint32_t, uint64_t> rel_fire;               // phase -> cycle its release left the gang
  std::map<uint32_t, std::vector<uint64_t>> unlocks;   // phase -> unlock cycles of its waiters
  uint32_t gang_arr   = 0;    // the gang's count of block arrivals, current phase
  uint32_t gang_phase = 0;
};

struct CScenario {
  std::vector<Cluster> clusters;
  // per core: (slot, cluster, record) written before the warps start
  std::vector<std::vector<std::pair<uint32_t, int>>> admit;
  std::vector<std::vector<std::vector<COp>>> prog;   // per core, per warp
  uint32_t done_events = 0;   // CTA_DONE events per core
  uint32_t done_pct    = 0;
  bool     random_pick = false;
  uint32_t jitter      = 0;   // random delay before an op, up to this many cycles
  // hold each release until the next global round completes at the global
  // barrier unit, and send it in that very cycle
  bool     collide     = false;

  CScenario() : admit(NCC), prog(NCC, std::vector<std::vector<COp>>(NW)) {}

  SlotInfo info_of(int cl, uint32_t slot) const {
    const Cluster& x = clusters[cl];
    SlotInfo i;
    i.gang = x.gang;
    i.gang_id = x.gang_id;
    i.epoch = x.epoch;
    i.k = x.K;
    i.base = slot;
    return i;
  }

  // A gang over (core, slot) members, or a cluster completed inside one core
  // (legacy or K=1) over consecutive slots from members[0].
  int add(bool gang, uint32_t gang_id, uint32_t epoch, uint32_t W,
          const std::vector<std::pair<uint32_t, uint32_t>>& members, bool bogus = false) {
    Cluster x;
    x.gang = gang;
    x.gang_id = gang_id;
    x.epoch = epoch;
    x.K = uint32_t(members.size());
    x.W = W;
    x.bogus = bogus;
    clusters.push_back(x);
    int cl = int(clusters.size()) - 1;
    for (const auto& m : members) {
      admit[m.first].push_back({m.second, cl});
    }
    return cl;
  }
};

struct CStats {
  uint64_t cycles = 0;
  uint32_t cbars = 0, dones = 0, releases = 0, unlocks = 0, globals = 0;
  uint32_t collisions = 0;     // a done event offered in the cycle a block's arrival pushes its CBAR
  uint32_t rel_under_req = 0;  // a core took a release in a cycle it had a barrier request
  uint64_t rsp_stalls = 0;
  uint32_t collide_tries = 0;   // releases timed onto a global barrier's completion
  uint64_t a_min = ~0ull, a_max = 0;   // CBAR push -> gang event port
  uint64_t b_min = ~0ull, b_max = 0;   // release leaves the gang -> unlock
};

class CBench {
public:
  CBench(Sim& sim, uint64_t& tick, uint32_t seed) : sim_(sim), tick_(tick), rng_(seed) {}

  bool run(CScenario sc, CStats& st) {
    struct Wait {
      int      kind = 0;   // 0 none, 1 cluster, 2 local barrier, 3 global
      int      cl = -1;
      uint32_t ph = 0, slot = 0, inst = 0, round = 0;
    };
    struct Warp {
      size_t   pc = 0;
      uint64_t ready_at = 0;
      Wait     wait;
      int      last_cl = -1;   // the cluster of its last arrival
      uint32_t last_ph = 0;
      int      cnt_slot = -1;  // arrivals counted on (slot, admission)
      uint32_t cnt_adm = 0;
      uint32_t cnt = 0;
    };
    struct Slot {
      int      cl = -1;
      uint32_t adm = 0;
      uint32_t sarr = 0;       // arrivals of the block in its current phase
      uint32_t lbar = 0;       // local barrier arrivals
      std::map<uint32_t, uint64_t> lbar_done;   // instance -> request cycle of its last arrival
    };
    struct Req {
      bool     valid = false;
      COpT     t = C_SYNC;
      uint32_t wid = 0, slot = 0, size_m1 = 0, id = 0, token = 0;
      bool     cbar_push = false;
      int      cl = -1;
      uint32_t round = 0;
    };
    struct Exp {
      uint32_t kind, slot, gang, epoch, size_m1;
      uint64_t push;
    };
    struct Rel {
      uint32_t gang, epoch;
      int      cl;
      uint32_t ph;
      uint64_t not_before;
    };

    std::vector<std::vector<Warp>> warps(NCC, std::vector<Warp>(NW));
    std::vector<std::vector<Slot>> slots(NCC, std::vector<Slot>(NW));
    std::vector<Req> pend(NCC);
    std::vector<std::deque<Exp>> expect(NCC);
    std::vector<std::deque<uint32_t>> todo(NCC);
    std::vector<uint32_t> made(NCC, 0), done_seq(NCC, 0);
    std::deque<Rel> rels;
    std::vector<uint32_t> g_round(NCC, 0);
    std::map<uint32_t, uint32_t> g_arrived, g_released;
    uint32_t errors = 0;
    uint64_t idle = 0, tail = 0;
    auto fail = [&](const char* fmt, auto... args) {
      if (++errors <= 10) {
        std::printf("FAIL cycle %lu: ", (unsigned long)st.cycles);
        std::printf(fmt, args...);
        std::printf("\n");
      }
    };

    this->clear();
    tick_ = sim_.reset(tick_);
    st = CStats();

    // a gang id no cluster of the scenario uses, for releases that must match nothing
    uint32_t idle_gang = 0;
    for (bool used = true; used && idle_gang < GANGS;) {
      used = false;
      for (const auto& x : sc.clusters) {
        used = used || (x.gang && x.gang_id == idle_gang);
      }
      if (used) {
        ++idle_gang;
      }
    }

    // the slot records go in first, one per core and cycle
    size_t start = 0;
    for (uint32_t c = 0; c < NCC; ++c) {
      start = std::max(start, sc.admit[c].size());
    }
    start += 2;

    auto addr_of = [](uint32_t slot, uint32_t id) { return (slot << NB_BITS) | id; };
    auto admit = [&](uint32_t c, uint32_t slot, int cl, const SlotInfo& info) {
      put(sim_->cbar_wr_valid, c, 1, 1);
      put(sim_->cbar_wr_slot, c, NW_W, slot);
      put(sim_->cbar_wr_gang, c, 1, info.gang);
      put(sim_->cbar_wr_gang_id, c, GANG_W, info.gang_id);
      put(sim_->cbar_wr_epoch, c, 1, info.epoch);
      put(sim_->cbar_wr_base, c, NW_W, info.base);
      put(sim_->cbar_wr_k, c, 8, info.k);
      Slot& s = slots[c][slot];
      s.cl = cl;
      ++s.adm;
      s.sarr = 0;
      s.lbar = 0;
      s.lbar_done.clear();
    };

    while (true) {
      bool progress = false;
      for (uint32_t c = 0; c < NCC; ++c) {
        put(sim_->cbar_wr_valid, c, 1, 0);
        put(sim_->read_cluster, c, 1, 0);
        put(sim_->read_addr, c, BAR_ADDR_W, 0);
      }
      // slot records at the start
      if (st.cycles < start) {
        for (uint32_t c = 0; c < NCC; ++c) {
          if (st.cycles < sc.admit[c].size()) {
            uint32_t slot = sc.admit[c][st.cycles].first;
            int cl = sc.admit[c][st.cycles].second;
            SlotInfo info = sc.info_of(cl, slot);
            if (!sc.clusters[cl].gang) {
              // a cluster completed in the core: every member names the first slot
              for (const auto& a : sc.admit[c]) {
                if (a.second == cl) {
                  info.base = a.first;
                  break;
                }
              }
            }
            admit(c, slot, cl, info);
          }
        }
      }
      // new CTA_DONE events
      for (uint32_t c = 0; c < NCC; ++c) {
        if (made[c] < sc.done_events && (rng_() % 100) < sc.done_pct) {
          todo[c].push_back(made[c]++);
        }
      }
      // the requests of this cycle were issued last cycle
      std::vector<Req> cur = pend;
      std::vector<int> issued(NCC, -1);
      std::vector<Req> next(NCC);
      for (uint32_t c = 0; c < NCC && st.cycles >= start; ++c) {
        std::vector<uint32_t> order(NW);
        for (uint32_t w = 0; w < NW; ++w) {
          order[w] = w;
        }
        if (sc.random_pick) {
          std::shuffle(order.begin(), order.end(), rng_);
        }
        for (uint32_t w : order) {
          Warp& wp = warps[c][w];
          const auto& pg = sc.prog[c][w];
          if (wp.wait.kind != 0 || wp.pc >= pg.size() || wp.ready_at > st.cycles) {
            continue;
          }
          const COp& o = pg[wp.pc];
          if (o.t == C_DELAY) {
            wp.ready_at = st.cycles + o.n;
            ++wp.pc;
            continue;
          }
          if (o.t == C_HOLD_REL) {
            auto it = sc.clusters[o.cl].rel_fire.find(o.ph);
            if (it != sc.clusters[o.cl].rel_fire.end() && st.cycles >= it->second + o.n) {
              ++wp.pc;
            }
            continue;
          }
          if (issued[c] >= 0) {
            continue;
          }
          if (sc.jitter && (rng_() % 4) == 0) {
            wp.ready_at = st.cycles + 1 + rng_() % sc.jitter;
            continue;
          }
          issued[c] = int(w);
          ++wp.pc;
          wp.ready_at = st.cycles + 1;
          progress = true;
          if (o.t == C_ADMIT) {
            admit(c, o.slot, o.cl, o.info);
            continue;
          }
          Req& r = next[c];
          r.valid = true;
          r.t = o.t;
          r.wid = w;
          r.slot = o.slot;
          if (o.t == C_GLOBAL) {
            r.round = g_round[c]++;
            r.id = r.round % (NB - 1);
            r.size_m1 = NCC - 1;
            wp.wait = Wait();
            wp.wait.kind = 3;
            wp.wait.round = r.round;
            put(sim_->read_addr, c, BAR_ADDR_W, r.id);
            continue;
          }
          Slot& s = slots[c][o.slot];
          if (s.cl < 0) {
            fail("core %u warp %u: barrier on slot %u, which holds no block", c, w, o.slot);
            continue;
          }
          Cluster& x = sc.clusters[s.cl];
          r.cl = s.cl;
          r.size_m1 = x.W - 1;
          put(sim_->read_addr, c, BAR_ADDR_W, addr_of(o.slot, 0));
          if (o.t == C_LBAR) {
            uint32_t inst = s.lbar / x.W;
            if (++s.lbar % x.W == 0) {
              s.lbar_done[inst] = st.cycles + 1;
            }
            wp.wait = Wait();
            wp.wait.kind = 2;
            wp.wait.slot = o.slot;
            wp.wait.inst = inst;
            continue;
          }
          put(sim_->read_cluster, c, 1, 1);
          if (o.t == C_WAIT) {
            if (wp.last_cl != s.cl) {
              fail("core %u warp %u waits without an arrival", c, w);
            }
            r.token = wp.last_ph & 1;
            wp.wait = Wait();
            wp.wait.kind = 1;
            wp.wait.cl = s.cl;
            wp.wait.ph = wp.last_ph;
            continue;
          }
          // an arrival: its phase is the warp's count on this admission of the slot
          if (wp.cnt_slot != int(o.slot) || wp.cnt_adm != s.adm) {
            wp.cnt_slot = int(o.slot);
            wp.cnt_adm = s.adm;
            wp.cnt = 0;
          }
          uint32_t ph = wp.cnt++;
          r.token = ph & 1;
          wp.last_cl = s.cl;
          wp.last_ph = ph;
          if (++x.arr[ph] == x.K * x.W) {
            x.last_req[ph] = st.cycles + 1;
          }
          if (++s.sarr == x.W) {
            s.sarr = 0;
            r.cbar_push = x.gang;
          }
          if (o.t == C_SYNC) {
            wp.wait = Wait();
            wp.wait.kind = 1;
            wp.wait.cl = s.cl;
            wp.wait.ph = ph;
          }
        }
      }
      pend = next;

      // drive the request port and the done and release inputs
      for (uint32_t c = 0; c < NCC; ++c) {
        const Req& r = cur[c];
        bool cl = r.valid && (r.t == C_SYNC || r.t == C_ARRIVE || r.t == C_WAIT);
        put(sim_->req_valid, c, 1, r.valid);
        put(sim_->req_wid, c, NW_W, r.wid);
        put(sim_->req_id, c, NB_W, r.id);
        put(sim_->req_is_global, c, 1, r.valid && r.t == C_GLOBAL);
        put(sim_->req_is_cluster, c, 1, cl);
        put(sim_->req_is_arrive, c, 1, r.valid && (r.t == C_SYNC || r.t == C_ARRIVE || r.t == C_LBAR || r.t == C_GLOBAL));
        put(sim_->req_is_sync, c, 1, r.valid && (r.t == C_SYNC || r.t == C_LBAR || r.t == C_GLOBAL));
        put(sim_->req_phase, c, 1, r.valid && r.t == C_WAIT && r.token);
        put(sim_->req_size_m1, c, BAR_SIZE_W, r.size_m1);
        bool v = !todo[c].empty();
        uint32_t code = v ? todo[c].front() : 0;
        put(sim_->done_in_valid, c, 1, v);
        put(sim_->done_in_slot, c, NW_W, code & ((1u << NW_W) - 1));
        put(sim_->done_in_gang, c, GANG_W, (code >> NW_W) & ((1u << GANG_W) - 1));
        put(sim_->done_in_epoch, c, 1, (code >> (NW_W + GANG_W)) & 1);
      }
      if (sc.collide) {
        // no global round left to meet: let the releases go
        bool more = false;
        for (uint32_t c = 0; c < NCC; ++c) {
          more = more || warps[c][0].pc < sc.prog[c][0].size() || warps[c][0].wait.kind != 0;
        }
        if (!more) {
          for (auto& rl : rels) {
            rl.not_before = std::min<uint64_t>(rl.not_before, st.cycles);
          }
        }
      }
      bool rel_driven = !rels.empty() && rels.front().not_before <= st.cycles;
      sim_->rel_valid = rel_driven;
      sim_->rel_gang = rel_driven ? rels.front().gang : 0;
      sim_->rel_epoch = rel_driven ? rels.front().epoch : 0;
      sim_->eval();

      // an arrival returns the slot's phase before its own
      for (uint32_t c = 0; c < NCC; ++c) {
        const Req& r = next[c];
        if (r.valid && r.t == C_ARRIVE) {
          uint32_t got = get(sim_->read_phase, c, 1);
          if (got != r.token) {
            fail("core %u warp %u: arrival on slot %u returned phase %u, want %u", c, r.wid, r.slot, got, r.token);
          }
        }
      }
      // requests of this cycle: CBAR pushes, global arrivals
      for (uint32_t c = 0; c < NCC; ++c) {
        const Req& r = cur[c];
        if (r.valid && r.cbar_push) {
          const Cluster& x = sc.clusters[r.cl];
          expect[c].push_back({KIND_CBAR, r.slot, x.gang_id, x.epoch, x.K - 1, st.cycles});
        }
        if (r.valid && r.t == C_GLOBAL) {
          ++st.globals;
          if (++g_arrived[r.round] == NCC && sc.collide) {
            // the round's last request reaches the global barrier unit this many cycles later
            uint64_t at = st.cycles + 1 + (SS > 1) + (NS > 1);
            for (auto& rl : rels) {
              rl.not_before = std::min(rl.not_before, at);
            }
            ++st.collide_tries;
          }
        }
      }
      // CTA_DONE intake
      for (uint32_t c = 0; c < NCC; ++c) {
        if (todo[c].empty()) {
          continue;
        }
        bool ready = get(sim_->done_in_ready, c, 1);
        const Req& r = cur[c];
        if (r.valid && (r.cbar_push || r.t == C_GLOBAL)) {
          ++st.collisions;
          if (ready) {
            fail("core %u took a done event in the cycle its barrier request entered the queue", c);
          }
        }
        if (ready) {
          uint32_t code = todo[c].front();
          todo[c].pop_front();
          expect[c].push_back({KIND_DONE, code & ((1u << NW_W) - 1), (code >> NW_W) & ((1u << GANG_W) - 1),
                               (code >> (NW_W + GANG_W)) & 1, 0, st.cycles});
          progress = true;
        }
      }
      // the gang unit's event port
      if (sim_->evt_valid) {
        progress = true;
        uint32_t c = sim_->evt_core;
        uint32_t kind = sim_->evt_kind;
        if (c >= NCC || expect[c].empty()) {
          fail("event of kind %u from core %u, which sent none", kind, c);
        } else {
          Exp e = expect[c].front();
          expect[c].pop_front();
          bool same = (kind == e.kind) && (uint32_t(sim_->evt_slot) == e.slot) && (uint32_t(sim_->evt_gang) == e.gang)
                   && (uint32_t(sim_->evt_epoch) == e.epoch)
                   && (kind != KIND_CBAR || uint32_t(sim_->evt_size_m1) == e.size_m1);
          if (!same) {
            fail("core %u: event kind %u slot %u gang %u epoch %u size_m1 %u, expected kind %u slot %u gang %u epoch %u"
                 " size_m1 %u", c, kind, uint32_t(sim_->evt_slot), uint32_t(sim_->evt_gang), uint32_t(sim_->evt_epoch),
                 uint32_t(sim_->evt_size_m1), e.kind, e.slot, e.gang, e.epoch, e.size_m1);
          } else if (kind == KIND_DONE) {
            ++st.dones;
          } else {
            ++st.cbars;
            uint64_t lat = st.cycles - e.push;
            st.a_min = std::min(st.a_min, lat);
            st.a_max = std::max(st.a_max, lat);
            int cl = -1;
            for (size_t i = 0; i < sc.clusters.size(); ++i) {
              const Cluster& x = sc.clusters[i];
              if (x.gang && x.gang_id == e.gang && x.epoch == e.epoch) {
                cl = int(i);
              }
            }
            if (cl < 0) {
              fail("CBAR for gang %u epoch %u, which the bench never made", e.gang, e.epoch);
            } else {
              Cluster& x = sc.clusters[cl];
              if (++x.gang_arr == x.K) {
                x.gang_arr = 0;
                uint32_t ph = x.gang_phase++;
                // releases that must match nothing go first, well ahead of the real one
                uint64_t at = 0;
                if (x.bogus) {
                  rels.push_back({x.gang_id, x.epoch ^ 1u, -1, 0, 0});
                  rels.push_back({idle_gang % GANGS, x.epoch, -1, 0, 0});
                  at = st.cycles + 24;
                }
                rels.push_back({x.gang_id, x.epoch, cl, ph, sc.collide ? NEVER : at});
              }
            }
          }
        }
      }
      // the release port: the gang unit's release leaves a cycle after its last arrival at the earliest
      if (rel_driven && sim_->rel_ready) {
        Rel r = rels.front();
        rels.pop_front();
        if (r.cl >= 0) {
          sc.clusters[r.cl].rel_fire[r.ph] = st.cycles;
          ++st.releases;
        }
        progress = true;
      }
      // unlocks
      for (uint32_t c = 0; c < NCC; ++c) {
        if (get(sim_->cbar_rsp_stall, c, 1)) {
          fail("core %u held back a cluster-barrier release", c);
        }
        if (get(sim_->cbar_rsp_fire, c, 1) && cur[c].valid) {
          ++st.rel_under_req;
        }
        st.rsp_stalls += get(sim_->gbar_rsp_stall, c, 1);
        if (!get(sim_->unlock_valid, c, 1)) {
          continue;
        }
        uint64_t mask = get(sim_->unlock_mask, c, NW);
        for (uint32_t w = 0; w < NW; ++w) {
          if (!((mask >> w) & 1)) {
            continue;
          }
          Warp& wp = warps[c][w];
          progress = true;
          ++st.unlocks;
          if (wp.wait.kind == 0) {
            fail("core %u warp %u unlocked while not waiting", c, w);
            continue;
          }
          if (wp.wait.kind == 1) {
            Cluster& x = sc.clusters[wp.wait.cl];
            uint32_t ph = wp.wait.ph;
            auto lr = x.last_req.find(ph);
            if (lr == x.last_req.end() || lr->second >= st.cycles) {
              fail("core %u warp %u released from phase %u of cluster %d after %u of %u arrivals", c, w, ph,
                   wp.wait.cl, x.arr[ph], x.K * x.W);
            } else if (x.gang) {
              auto rf = x.rel_fire.find(ph);
              if (rf == x.rel_fire.end() || rf->second >= st.cycles) {
                fail("core %u warp %u released from phase %u of gang %u before the gang's release", c, w, ph,
                     x.gang_id);
              } else {
                uint64_t lat = st.cycles - rf->second;
                st.b_min = std::min(st.b_min, lat);
                st.b_max = std::max(st.b_max, lat);
              }
            }
            x.unlocks[ph].push_back(st.cycles);
          } else if (wp.wait.kind == 2) {
            const Slot& s = slots[c][wp.wait.slot];
            auto it = s.lbar_done.find(wp.wait.inst);
            if (it == s.lbar_done.end() || it->second >= st.cycles) {
              fail("core %u warp %u released from a local barrier of slot %u before its block arrived", c, w,
                   wp.wait.slot);
            }
          } else {
            if (g_arrived[wp.wait.round] != NCC) {
              fail("core %u warp %u released from global round %u after %u of %u arrivals", c, w, wp.wait.round,
                   g_arrived[wp.wait.round], NCC);
            }
            ++g_released[wp.wait.round];
          }
          wp.wait = Wait();
          wp.ready_at = st.cycles + 1;
        }
      }

      tick_ = sim_.step(tick_, 2);
      ++st.cycles;

      bool busy = !rels.empty();
      for (uint32_t c = 0; c < NCC; ++c) {
        busy = busy || pend[c].valid || !expect[c].empty() || !todo[c].empty() || made[c] < sc.done_events;
        for (uint32_t w = 0; w < NW; ++w) {
          busy = busy || warps[c][w].wait.kind != 0 || warps[c][w].pc < sc.prog[c][w].size();
        }
      }
      if (!busy && st.cycles >= start && ++tail > 20) {
        break;
      }
      idle = progress ? 0 : idle + 1;
      if (idle > 3000) {
        for (uint32_t c = 0; c < NCC; ++c) {
          for (uint32_t w = 0; w < NW; ++w) {
            const Warp& wp = warps[c][w];
            if (wp.wait.kind != 0 || wp.pc < sc.prog[c][w].size()) {
              std::printf("  core %u warp %u: op %zu of %zu, waiting kind %d cluster %d phase %u\n", c, w, wp.pc,
                          sc.prog[c][w].size(), wp.wait.kind, wp.wait.cl, wp.wait.ph);
            }
          }
        }
        fail("no progress for 3000 cycles");
        break;
      }
      if (errors > 10) {
        break;
      }
    }
    for (const auto& r : g_released) {
      if (r.second != NCC) {
        fail("global round %u released %u of %u cores", r.first, r.second, NCC);
      }
    }
    clusters = sc.clusters;
    return errors == 0;
  }

  std::vector<Cluster> clusters;   // the last run's clusters, with their records

private:
  void clear() {
    sim_->req_valid = 0;
    sim_->req_is_global = 0;
    sim_->req_is_arrive = 0;
    sim_->req_is_sync = 0;
    sim_->req_is_cluster = 0;
    sim_->req_phase = 0;
    sim_->read_cluster = 0;
    sim_->cbar_wr_valid = 0;
    sim_->rel_valid = 0;
    sim_->done_in_valid = 0;
    sim_->done_in_epoch = 0;
    for (uint32_t c = 0; c < NCC; ++c) {
      put(sim_->req_wid, c, NW_W, 0);
      put(sim_->req_id, c, NB_W, 0);
      put(sim_->req_size_m1, c, BAR_SIZE_W, 0);
      put(sim_->read_addr, c, BAR_ADDR_W, 0);
      put(sim_->active_warps, c, NW, 1);   // warp 0 alone takes part in global barriers
      put(sim_->done_in_slot, c, NW_W, 0);
      put(sim_->done_in_gang, c, GANG_W, 0);
      put(sim_->cbar_wr_slot, c, NW_W, 0);
      put(sim_->cbar_wr_gang_id, c, GANG_W, 0);
      put(sim_->cbar_wr_base, c, NW_W, 0);
      put(sim_->cbar_wr_k, c, 8, 0);
    }
    sim_->cbar_wr_gang = 0;
    sim_->cbar_wr_epoch = 0;
  }

  Sim& sim_;
  uint64_t& tick_;
  std::mt19937 rng_;
};

///////////////////////////////////////////////////////////////////////////////
// Cluster-barrier scenarios

static bool creport(const char* name, bool ok, const CStats& st) {
  std::printf("%s: %u CBARs, %u releases, %u unlocks, %u done events, %u global arrivals, %lu cycles: %s\n", name,
              st.cbars, st.releases, st.unlocks, st.dones, st.globals, (unsigned long)st.cycles, ok ? "ok" : "FAILED");
  return ok;
}

// A gang of one block per core, W warps each, all sync arrivals: one CBAR per
// core and phase, one release, every warp unlocked once. On an idle cluster the
// two latency legs are fixed.
static bool c_spread(Sim& sim, uint64_t& tick, uint32_t W, uint32_t phases, uint64_t* leg_a, uint64_t* leg_b) {
  CScenario sc;
  uint32_t slot = (NW > W) ? 1 : 0;
  std::vector<std::pair<uint32_t, uint32_t>> m;
  for (uint32_t c = 0; c < NCC; ++c) {
    m.push_back({c, slot});
  }
  sc.add(true, 1 % GANGS, 1, W, m);
  for (uint32_t c = 0; c < NCC; ++c) {
    for (uint32_t i = 0; i < W; ++i) {
      for (uint32_t p = 0; p < phases; ++p) {
        // the measured run keeps the cores' arrivals apart: no arbitration
        uint32_t d = leg_a ? 1 + 6 * c : 1 + (c * 7 + i * 3 + p * 5) % 13;
        sc.prog[c][slot + i].push_back(cop(C_DELAY, 0, d));
        sc.prog[c][slot + i].push_back(cop(C_SYNC, slot));
      }
    }
  }
  CBench b(sim, tick, 101 + W);
  CStats st;
  bool ok = b.run(sc, st);
  ok = ok && (st.cbars == NCC * phases) && (st.releases == phases) && (st.unlocks == NCC * W * phases);
  if (ok && leg_a && (st.a_min != st.a_max || st.b_min != st.b_max)) {
    std::printf("FAIL spread: latency varies on an idle cluster: CBAR %lu..%lu, release %lu..%lu\n",
                (unsigned long)st.a_min, (unsigned long)st.a_max, (unsigned long)st.b_min, (unsigned long)st.b_max);
    ok = false;
  }
  if (leg_a) {
    *leg_a = st.a_min;
    *leg_b = st.b_min;
  }
  char name[96];
  std::snprintf(name, sizeof(name), "spread gang K=%u W=%u, %u phases", NCC, W, phases);
  return creport(name, ok, st);
}

// Split arrive/wait: an arrival returns the phase before its own; waits come
// before and after the release; a __syncthreads of the block between arrive and
// wait completes on its own; the last phase is arrive-only.
static bool c_split(Sim& sim, uint64_t& tick) {
  CScenario sc;
  uint32_t W = std::min<uint32_t>(2, NW - 1);
  uint32_t base = 1;
  std::vector<std::pair<uint32_t, uint32_t>> m;
  bool gang = (NCC > 1);
  if (gang) {
    for (uint32_t c = 0; c < NCC; ++c) {
      m.push_back({c, base});
    }
  } else {
    m.push_back({0, base});   // one core: a cluster of one block
  }
  int cl = sc.add(gang, 2 % GANGS, 0, W, m);
  (void)cl;
  for (const auto& mm : m) {
    for (uint32_t i = 0; i < W; ++i) {
      auto& pg = sc.prog[mm.first][base + i];
      for (uint32_t p = 0; p < 4; ++p) {
        pg.push_back(cop(C_ARRIVE, base));
        // core 0's warps wait at once, the others after a long delay
        pg.push_back(cop(C_DELAY, 0, (mm.first == 0) ? 1 : 40 + 3 * i));
        pg.push_back(cop(C_LBAR, base));
        pg.push_back(cop(C_WAIT, base));
      }
      pg.push_back(cop(C_ARRIVE, base));
    }
  }
  CBench b(sim, tick, 111);
  CStats st;
  bool ok = b.run(sc, st);
  return creport(gang ? "split arrive/wait across the gang, __syncthreads between" :
                        "split arrive/wait, one block, __syncthreads between", ok, st);
}

// Two slots of one gang on one core, then two blocks on every other core: the
// release unlocks both slots of a core at once.
static bool c_two_slots(Sim& sim, uint64_t& tick) {
  if (NW < 3) {
    std::printf("two slots of a gang on one core: needs 3 warps, skipped\n");
    return true;
  }
  CScenario sc;
  // two slots on each of the first MAX/2 cores; at one core the bar unit's widths
  // still hold a gang of two, which the gang unit never forms there
  uint32_t cores = std::max<uint32_t>(1, std::min<uint32_t>(NCC, MAXC / 2));
  std::vector<std::pair<uint32_t, uint32_t>> m;
  for (uint32_t c = 0; c < cores; ++c) {
    m.push_back({c, 1});
    m.push_back({c, 2});
  }
  int cl = sc.add(true, 0, 1, 1, m, true);
  for (const auto& mm : m) {
    auto& pg = sc.prog[mm.first][mm.second];
    for (uint32_t p = 0; p < 3; ++p) {
      pg.push_back(cop(C_DELAY, 0, 1 + (mm.first * 5 + mm.second * 11 + p) % 9));
      pg.push_back(cop(C_SYNC, mm.second));
    }
  }
  CBench b(sim, tick, 121);
  CStats st;
  bool ok = b.run(sc, st);
  // both slots of a core unlock in the same cycle
  for (const auto& u : b.clusters[cl].unlocks) {
    std::map<uint64_t, uint32_t> per;
    for (auto t : u.second) {
      ++per[t];
    }
    if (u.second.size() != 2 * cores || per.size() > cores) {
      std::printf("FAIL two-slots: phase %u unlocked %zu warps over %zu cycles\n", u.first, u.second.size(), per.size());
      ok = false;
    }
  }
  return creport("two slots of one gang per core, with releases that match no slot", ok, st);
}

// Releases that match no pending slot (wrong epoch, idle gang) change nothing,
// and a slot admitted again starts at phase 0 and ignores its old gang's
// release: the old gang completes one phase, then arrives again, and slot 1
// takes a new block before the second release.
static bool c_readmit(Sim& sim, uint64_t& tick) {
  if (NW < 4) {
    std::printf("re-admission: needs 4 warps, skipped\n");
    return true;
  }
  CScenario sc;
  // gang g over slots 1 and 2 of core 0, one warp each
  int g = sc.add(true, 0, 1, 1, {{0, 1}, {0, 2}}, true);
  // slot 1's next block: a cluster of one with warps 1 and 3
  Cluster nx;
  nx.K = 1;
  nx.W = 2;
  sc.clusters.push_back(nx);
  int n = int(sc.clusters.size()) - 1;
  COp ad = cop(C_ADMIT, 1);
  ad.cl = n;
  ad.info.gang = false;
  ad.info.base = 1;
  ad.info.k = 1;
  COp hold = cop(C_HOLD_REL, 0, 12);
  hold.cl = g;
  hold.ph = 1;
  COp hold3 = hold;
  hold3.n = 16;
  sc.prog[0][1] = {cop(C_ARRIVE, 1), cop(C_WAIT, 1), cop(C_ARRIVE, 1), cop(C_DELAY, 0, 2), ad, hold,
                   cop(C_ARRIVE, 1), cop(C_DELAY, 0, 4), cop(C_WAIT, 1)};
  sc.prog[0][2] = {cop(C_DELAY, 0, 6), cop(C_ARRIVE, 2), cop(C_WAIT, 2), cop(C_DELAY, 0, 20), cop(C_ARRIVE, 2)};
  sc.prog[0][3] = {hold3, cop(C_ARRIVE, 1)};
  CBench b(sim, tick, 131);
  CStats st;
  bool ok = b.run(sc, st);
  ok = ok && (st.releases == 2) && (st.cbars == 4) && (st.unlocks == 3);
  return creport("stale releases and re-admission", ok, st);
}

// A cluster completed inside the core (legacy, or K=1): no CBAR on the bus, all
// of its warps unlocked in the cycle after its last arrival.
static bool c_local(Sim& sim, uint64_t& tick) {
  uint32_t K = std::min<uint32_t>(3, NW / 2);
  uint32_t W = 2;
  if (K < 2) {
    std::printf("legacy cluster: needs 4 warps, skipped\n");
    return true;
  }
  uint32_t base = (NW >= 2 * K + 2) ? 2 : 0;
  CScenario sc;
  std::vector<std::pair<uint32_t, uint32_t>> m;
  for (uint32_t r = 0; r < K; ++r) {
    m.push_back({0, base + r});
  }
  int cl = sc.add(false, 0, 0, W, m);
  for (uint32_t r = 0; r < K; ++r) {
    for (uint32_t i = 0; i < W; ++i) {
      auto& pg = sc.prog[0][r * W + i];
      for (uint32_t p = 0; p < 3; ++p) {
        pg.push_back(cop(C_DELAY, 0, 1 + (r * 5 + i * 3 + p * 7) % 11));
        pg.push_back(cop(C_SYNC, base + r));
      }
    }
  }
  // K=1 for both policies: on core 1, or on two free slots and warps of core 0
  bool k1 = (NCC > 1) || (K * W + 2 <= NW && base + K <= NW - 2);
  uint32_t c1 = (NCC > 1) ? 1 : 0;
  uint32_t s1 = (NCC > 1) ? 0 : NW - 2;
  if (k1) {
    int pinned = sc.add(true, 3 % GANGS, 1, 1, {{c1, s1}});
    sc.clusters[pinned].gang = false;   // a pinned block of K=1 completes in its core
    sc.add(false, 0, 0, 1, {{c1, s1 + 1}});
    for (uint32_t p = 0; p < 3; ++p) {
      sc.prog[c1][s1].push_back(cop(C_SYNC, s1));
      sc.prog[c1][s1 + 1].push_back(cop(C_ARRIVE, s1 + 1));
      sc.prog[c1][s1 + 1].push_back(cop(C_WAIT, s1 + 1));
    }
  }
  CBench b(sim, tick, 141);
  CStats st;
  bool ok = b.run(sc, st);
  ok = ok && (st.cbars == 0) && (st.releases == 0);
  for (const auto& u : b.clusters[cl].unlocks) {
    uint64_t want = b.clusters[cl].last_req.at(u.first) + 1;
    for (auto t : u.second) {
      if (t != want) {
        std::printf("FAIL legacy: phase %u: a warp unlocked in cycle %lu, the last arrival was in %lu\n", u.first,
                    (unsigned long)t, (unsigned long)(want - 1));
        ok = false;
        break;
      }
    }
    if (u.second.size() != K * W) {
      std::printf("FAIL legacy: phase %u unlocked %zu of %u warps\n", u.first, u.second.size(), K * W);
      ok = false;
    }
  }
  char name[96];
  std::snprintf(name, sizeof(name), "legacy cluster K=%u W=%u at slot %u, and K=1 of both policies", K, W, base);
  return creport(name, ok, st);
}

// A release presented in the cycle a global barrier completes into the global
// barrier unit's response register waits for the next cycle: neither is lost.
static bool c_collide(Sim& sim, uint64_t& tick) {
  CScenario sc;
  std::vector<std::pair<uint32_t, uint32_t>> m;
  for (uint32_t c = 0; c < NCC; ++c) {
    m.push_back({c, 1});
  }
  sc.add(true, 1 % GANGS, 0, 1, m);
  for (uint32_t c = 0; c < NCC; ++c) {
    for (uint32_t r = 0; r < 16; ++r) {
      sc.prog[c][0].push_back(cop(C_DELAY, 0, 1 + (c * 3 + r) % 5));
      sc.prog[c][0].push_back(cop(C_GLOBAL));
      sc.prog[c][1].push_back(cop(C_DELAY, 0, 1 + (c + r) % 3));
      sc.prog[c][1].push_back(cop(C_SYNC, 1));
    }
  }
  sc.collide = true;
  CBench b(sim, tick, 151);
  CStats st;
  bool ok = b.run(sc, st);
  std::printf("collide: %u releases timed onto a global barrier's completion\n", st.collide_tries);
  ok = ok && (st.collide_tries > 0) && (st.releases == 16);
  return creport("releases against global barrier completions", ok, st);
}

// Everything at once: gang arrivals and their CBARs, done events, global rounds
// on warp 0, local barriers and releases, under response stalls.
static bool c_mix(Sim& sim, uint64_t& tick, uint32_t seed) {
  CScenario sc;
  std::mt19937 rng(seed * 31 + 7);
  // slot s >= 1 of every core joins gang s (one warp per block); at one core,
  // slots pair up into gangs of two
  std::vector<int> cls(NW, -1);
  for (uint32_t s = 1; s < NW; ++s) {
    if (NCC > 1) {
      std::vector<std::pair<uint32_t, uint32_t>> m;
      for (uint32_t c = 0; c < NCC; ++c) {
        m.push_back({c, s});
      }
      cls[s] = sc.add(true, s % GANGS, s & 1, 1, m, (s % 3) == 0);
    } else if (s + 1 < NW && (s % 2) == 1) {
      cls[s] = cls[s + 1] = sc.add(true, (s / 2) % GANGS, 1, 1, {{0, s}, {0, s + 1}}, true);
    } else if (cls[s] < 0) {
      cls[s] = sc.add(false, 0, 0, 1, {{0, s}});
    }
  }
  for (uint32_t c = 0; c < NCC; ++c) {
    for (uint32_t s = 1; s < NW; ++s) {
      auto& pg = sc.prog[c][s];
      for (uint32_t p = 0; p < 6; ++p) {
        uint32_t f = rng() % 3;
        if (f == 0) {
          pg.push_back(cop(C_SYNC, s));
        } else {
          pg.push_back(cop(C_ARRIVE, s));
          pg.push_back(cop(C_DELAY, 0, rng() % 12));
          if (f == 2) {
            pg.push_back(cop(C_LBAR, s));
          }
          pg.push_back(cop(C_WAIT, s));
        }
        pg.push_back(cop(C_LBAR, s));
      }
    }
    if (NCC > 1) {
      for (uint32_t r = 0; r < 12; ++r) {
        sc.prog[c][0].push_back(cop(C_DELAY, 0, rng() % 20));
        sc.prog[c][0].push_back(cop(C_GLOBAL));
      }
    }
  }
  sc.done_events = 60;
  sc.done_pct = 30;
  sc.random_pick = true;
  sc.jitter = 6;
  CBench b(sim, tick, seed);
  CStats st;
  bool ok = b.run(sc, st);
  if (ok && (st.collisions == 0 || (NCC > 1 && st.rsp_stalls == 0) || st.rel_under_req == 0)) {
    std::printf("FAIL mix: %u collisions, %lu response stalls, %u releases taken under a request; the schedule must"
                " produce each\n", st.collisions, (unsigned long)st.rsp_stalls, st.rel_under_req);
    ok = false;
  }
  std::printf("mix: %u collisions, %lu response stalls, %u releases taken under a request\n", st.collisions,
              (unsigned long)st.rsp_stalls, st.rel_under_req);
  return creport("cluster barriers with done events, global and local barriers", ok, st);
}


int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  uint32_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '+') {
      continue;
    }
    if (std::strncmp(argv[i], "--seed=", 7) == 0) {
      seed = uint32_t(std::strtoul(argv[i] + 7, nullptr, 0));
    } else {
      std::printf("usage: %s [--seed=N]\n", argv[0]);
      return 2;
    }
  }

  std::printf("bar_unit: cores=%u (%u sockets of %u) warps=%u barriers=%u seed=%u\n", NCC, NS, SS, NW, NB, seed);

  Sim sim;
  uint64_t tick = 0;
  Bench bench(sim, tick, seed);
  bool ok = true;

  // Done latency on an idle cluster: one event at a time.
  Opts iso;
  iso.events_per_core = 8;
  iso.isolated = true;
  Stats st;
  bool r = bench.run(iso, st);
  if (r && st.lat_min != st.lat_max) {
    std::printf("FAIL latency: an idle cluster delivered done events in %lu to %lu cycles\n",
                (unsigned long)st.lat_min, (unsigned long)st.lat_max);
    r = false;
  }
  uint64_t expect = 2 + (SS > 1) + (NS > 1);
  std::printf("done latency: %lu cycles from the core's queue to the gang unit (%lu from the dispatcher's last"
              " warp_done); expected 2 + [SOCKET_SIZE>1] + [NUM_SOCKETS>1] = %lu: %s\n",
              (unsigned long)st.lat_min, (unsigned long)st.lat_min + 3, (unsigned long)expect,
              (r && st.lat_min == expect) ? "ok" : "FAILED");
  ok = r && (st.lat_min == expect) && ok;

  // Done events alone, at full rate: the queues fill and push back.
  Opts flood;
  flood.events_per_core = 200;
  flood.event_pct = 90;
  flood.local_pct = 40;
  r = bench.run(flood, st);
  std::printf("done flood: %u events in %lu cycles, %u held back on a full queue, %lu response stalls,"
              " latency %lu..%lu: %s\n", st.delivered, (unsigned long)st.cycles, st.full_waits,
              (unsigned long)st.rsp_stalls, (unsigned long)st.lat_min, (unsigned long)st.lat_max, r ? "ok" : "FAILED");
  ok = r && ok;

  // Done events colliding with global barrier rounds, under response stalls.
  if (NCC > 1) {
    Opts mix;
    mix.events_per_core = 300;
    mix.rounds = 60;
    mix.event_pct = 20;
    mix.arrive_pct = 25;
    mix.local_pct = 30;
    r = bench.run(mix, st);
    if (r && (st.collisions == 0 || st.rsp_stalls == 0)) {
      std::printf("FAIL mix: %u collisions, %lu response stalls; the schedule must produce both\n", st.collisions,
                  (unsigned long)st.rsp_stalls);
      r = false;
    }
    std::printf("barriers and done events: %u events, %u releases of %u rounds, %u collisions, %lu response stalls,"
                " %lu cycles: %s\n", st.delivered, st.releases, mix.rounds, st.collisions,
                (unsigned long)st.rsp_stalls, (unsigned long)st.cycles, r ? "ok" : "FAILED");
    ok = r && ok;
  } else {
    std::printf("barriers and done events: one core has no global barriers, skipped\n");
  }

  // Cluster barriers.
  if (NCC > 1) {
    uint64_t leg_a = 0, leg_b = 0;
    r = c_spread(sim, tick, 1, 3, &leg_a, &leg_b);
    uint64_t want = 2 + (SS > 1) + (NS > 1);
    std::printf("cluster-barrier latency: %lu cycles from the CBAR's push to the gang unit, %lu from the gang's release"
                " to the unlock; expected 2 + [SOCKET_SIZE>1] + [NUM_SOCKETS>1] = %lu for both: %s\n",
                (unsigned long)leg_a, (unsigned long)leg_b, (unsigned long)want,
                (r && leg_a == want && leg_b == want) ? "ok" : "FAILED");
    ok = r && (leg_a == want) && (leg_b == want) && ok;
    ok = c_spread(sim, tick, std::min<uint32_t>(3, NW - 1), 4, nullptr, nullptr) && ok;
  } else {
    std::printf("spread gang: one core has no gang of K > 1 across cores, skipped\n");
  }
  ok = c_split(sim, tick) && ok;
  ok = c_two_slots(sim, tick) && ok;
  ok = c_readmit(sim, tick) && ok;
  ok = c_local(sim, tick) && ok;
  if (NCC > 1) {
    ok = c_collide(sim, tick) && ok;
  }
  ok = c_mix(sim, tick, seed) && ok;

  if (!ok) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
