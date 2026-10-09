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
      // the gang unit's done port
      bool progress = false;
      if (sim_->done_valid) {
        uint32_t c = sim_->done_core;
        uint32_t code = uint32_t(sim_->done_slot) | (uint32_t(sim_->done_gang) << NW_W)
                      | (uint32_t(sim_->done_epoch) << (NW_W + GANG_W));
        if (c >= NCC || sent[c].empty()) {
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

  if (!ok) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
