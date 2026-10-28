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

// VX_mbar_waiters testbench. Directed scenarios pin the cycle of every
// unlock (a release decided in cycle T is on the unlock port in T+1): wake on
// a flip, the time limit, an insert and a flip back to back on one bank, flips
// on other banks in the same cycle, every warp waiting on one object and on
// many, a retire racing a flip, a retire with no entry, two lanes of one warp
// inserting in one cycle (the higher bank wins) and one after the other (the
// last wins, a wake-up is kept). A random run then drives try_wait lifecycles
// against a reference model of the table, cycle by cycle. --neg=insert sends
// an insert for a retired warp, and the table's assert must abort the run.

#include "vl_simulator.h"
#include "VVX_mbar_waiters_top.h"
#include "VX_config.h"

#include <array>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

constexpr uint32_t NUM_WARPS   = VX_CFG_NUM_WARPS;
constexpr uint32_t NUM_BANKS   = VX_CFG_LMEM_NUM_BANKS;
constexpr uint32_t NUM_RETIRES = VX_CFG_NUM_LSU_BLOCKS;
constexpr uint32_t SUSPEND     = VX_CFG_MBAR_SUSPEND_CYCLES;
constexpr uint32_t ROW_BITS    = 8; // VX_mbar_waiters_top ROW_WIDTH
constexpr uint32_t WID_BITS    = 8;

static_assert(NUM_WARPS <= 32, "the unlock mask is read as 32 bits");
static_assert(NUM_BANKS <= 8, "event lanes are packed in 64 bits");
static_assert(NUM_RETIRES <= 8, "retire lanes are packed in 64 bits");

using Mask = uint32_t;

static std::string strf(const char* fmt, ...) {
  char buf[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

struct Evt {
  bool     valid = false;
  bool     flip  = false;
  uint32_t wid   = 0;
  uint32_t row   = 0;
};

struct Ret {
  bool     valid = false;
  uint32_t wid   = 0;
};

struct Inputs {
  std::array<Evt, NUM_BANKS>   evt{};
  std::array<Ret, NUM_RETIRES> ret{};

  Inputs& insert(uint32_t bank, uint32_t wid, uint32_t row) {
    evt.at(bank) = {true, false, wid, row};
    return *this;
  }
  Inputs& flip(uint32_t bank, uint32_t row) {
    evt.at(bank) = {true, true, 0, row};
    return *this;
  }
  Inputs& retire(uint32_t wid) {
    ret.at(wid % NUM_RETIRES) = {true, wid};
    return *this;
  }
};

static uint64_t pack_bits(uint64_t v, uint32_t lsb, uint32_t width, uint64_t into) {
  uint64_t m = ((width == 64) ? ~0ull : ((1ull << width) - 1)) << lsb;
  return (into & ~m) | ((v << lsb) & m);
}

///////////////////////////////////////////////////////////////////////////////
// Reference model of the table: inserts in bank order, flips, retires, timers.

class Model {
public:
  struct Entry {
    bool     valid   = false;
    bool     woke    = false;
    bool     retired = false;
    uint32_t bank    = 0;
    uint32_t row     = 0;
    uint32_t timer   = 0;
  };

  // The warps released in this cycle; timeouts are the ones the time limit released.
  Mask step(const Inputs& in, Mask* timeouts, std::vector<std::string>& errors) {
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      const Evt& e = in.evt[b];
      if (e.valid && !e.flip) {
        Entry& x = entries_.at(e.wid);
        if (x.retired) {
          errors.push_back(strf("model: insert after retire, wid=%u", e.wid));
        }
        x.valid = true;
        x.bank = b;
        x.row = e.row;
      }
    }
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      const Evt& e = in.evt[b];
      if (e.valid && e.flip) {
        for (auto& x : entries_) {
          if (x.valid && x.bank == b && x.row == e.row) {
            x.woke = true;
          }
        }
      }
    }
    Mask rel = 0;
    *timeouts = 0;
    for (const Ret& r : in.ret) {
      if (!r.valid) {
        continue;
      }
      Entry& x = entries_.at(r.wid);
      if (!x.valid) {
        rel |= 1u << r.wid;
      } else if (x.woke) {
        rel |= 1u << r.wid;
        x = Entry{};
      } else {
        x.retired = true;
        x.timer = SUSPEND;
      }
    }
    for (uint32_t w = 0; w < NUM_WARPS; ++w) {
      Entry& x = entries_[w];
      if (!x.valid || !x.retired) {
        continue;
      }
      if (x.woke) {
        rel |= 1u << w;
        x = Entry{};
      } else if (x.timer == 0) {
        rel |= 1u << w;
        *timeouts |= 1u << w;
        x = Entry{};
      } else {
        --x.timer;
      }
    }
    return rel;
  }

  bool empty() const {
    for (const auto& x : entries_) {
      if (x.valid) {
        return false;
      }
    }
    return true;
  }

  const Entry& entry(uint32_t w) const {
    return entries_.at(w);
  }

private:
  std::array<Entry, NUM_WARPS> entries_{};
};

///////////////////////////////////////////////////////////////////////////////
// DUT and model in lockstep: the unlock port in cycle c carries the release of c-1.

class Bench {
public:
  Bench() {
    auto* t = sim_.operator->();
    t->reset = 1;
    this->drive(Inputs{});
    for (int i = 0; i < 4; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
  }

  // One cycle; returns the unlock mask seen on the port in this cycle.
  Mask cycle(const Inputs& in) {
    auto* t = sim_.operator->();
    this->drive(in);
    t->eval();
    bool valid = t->unlock_valid;
    Mask got = static_cast<Mask>(t->unlock_mask);
    if (valid != (got != 0)) {
      this->fail(strf("cycle %" PRIu64 ": unlock_valid=%d with mask 0x%x", cycle_, valid, got));
    }
    if (got != prev_rel_) {
      this->fail(strf("cycle %" PRIu64 ": unlock mask 0x%x, the model released 0x%x in the previous cycle",
                      cycle_, got, prev_rel_));
    }
    std::vector<std::string> errors;
    Mask timeouts = 0;
    prev_rel_ = model_.step(in, &timeouts, errors);
    for (auto& e : errors) {
      this->fail(strf("cycle %" PRIu64 ": %s", cycle_, e.c_str()));
    }
    total_unlocks_ += __builtin_popcount(got);
    total_timeouts_ += __builtin_popcount(timeouts);
    tick_ = sim_.step(tick_, 2);
    ++cycle_;
    return valid ? got : 0;
  }

  void fail(const std::string& msg) {
    ++failures_;
    if (failures_ <= 50) {
      std::printf("FAIL %s\n", msg.c_str());
    }
  }

  uint64_t now() const { return cycle_; }
  uint32_t failures() const { return failures_; }
  uint64_t unlocks() const { return total_unlocks_; }
  uint64_t timeouts() const { return total_timeouts_; }
  const Model& model() const { return model_; }

private:
  void drive(const Inputs& in) {
    auto* t = sim_.operator->();
    uint64_t ev = 0, ef = 0, ew = 0, er = 0, rv = 0, rw = 0;
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      ev = pack_bits(in.evt[b].valid, b, 1, ev);
      ef = pack_bits(in.evt[b].flip, b, 1, ef);
      ew = pack_bits(in.evt[b].wid, b * WID_BITS, WID_BITS, ew);
      er = pack_bits(in.evt[b].row, b * ROW_BITS, ROW_BITS, er);
    }
    for (uint32_t i = 0; i < NUM_RETIRES; ++i) {
      rv = pack_bits(in.ret[i].valid, i, 1, rv);
      rw = pack_bits(in.ret[i].wid, i * WID_BITS, WID_BITS, rw);
    }
    t->evt_valid = ev;
    t->evt_flip = ef;
    t->evt_wid = ew;
    t->evt_row = er;
    t->retire_valid = rv;
    t->retire_wid = rw;
  }

  vl_simulator<VVX_mbar_waiters_top> sim_;
  Model    model_;
  Mask     prev_rel_ = 0;
  uint64_t tick_ = 0;
  uint64_t cycle_ = 0;
  uint64_t total_unlocks_ = 0;
  uint64_t total_timeouts_ = 0;
  uint32_t failures_ = 0;
};

///////////////////////////////////////////////////////////////////////////////
// Directed scenarios: a script of inputs by cycle and the unlock expected on
// the port by cycle, every other cycle expecting none. Each ends idle.

class Directed {
public:
  explicit Directed(Bench& tb) : tb_(tb) {}

  void run(const char* name, const std::map<uint32_t, Inputs>& script,
           const std::map<uint32_t, Mask>& expect, uint32_t length) {
    uint32_t before = tb_.failures();
    for (uint32_t c = 0; c < length; ++c) {
      auto s = script.find(c);
      Mask got = tb_.cycle(s != script.end() ? s->second : Inputs{});
      auto e = expect.find(c);
      Mask want = (e != expect.end()) ? e->second : 0;
      if (got != want) {
        tb_.fail(strf("%s: cycle +%u: unlock mask 0x%x, expected 0x%x", name, c, got, want));
      }
    }
    if (!tb_.model().empty()) {
      tb_.fail(strf("%s: entries left at the end", name));
    }
    std::printf("  %-44s %s\n", name, (tb_.failures() == before) ? "ok" : "FAILED");
    ++count_;
  }

  uint32_t count() const { return count_; }

private:
  Bench&   tb_;
  uint32_t count_ = 0;
};

static Mask bit(uint32_t w) {
  return 1u << w;
}

static Mask all_warps() {
  return (NUM_WARPS == 32) ? ~0u : ((1u << NUM_WARPS) - 1);
}

static void directed(Bench& tb, Directed& d) {
  const uint32_t B1 = 1 % NUM_BANKS;
  const uint32_t B2 = 2 % NUM_BANKS;
  const uint32_t W1 = 1 % NUM_WARPS;
  const uint32_t W2 = 2 % NUM_WARPS;
  const uint32_t S  = SUSPEND;

  // Retired, then a flip of its object at 10: released at 10, on the port at 11.
  d.run("wake on flip",
        {{0, Inputs().insert(0, 0, 5)}, {3, Inputs().retire(0)}, {10, Inputs().flip(0, 5)}},
        {{11, bit(0)}}, 14);

  // No flip: the time limit releases it SUSPEND cycles after the retire.
  d.run("time limit",
        {{0, Inputs().insert(B1, W1, 7)}, {2, Inputs().retire(W1)}},
        {{2 + S + 1, bit(W1)}}, S + 6);

  // A flip of another row of the bank, or of the row on another bank, does not wake.
  d.run("flip of another object",
        {{0, Inputs().insert(0, 0, 3)}, {1, Inputs().retire(0)},
         {4, Inputs().flip(0, 4).flip(B1, 3)}},
        {{1 + S + 1, bit(0)}}, S + 5);

  // Insert at T+1 of a try_wait, a completing arrive's flip at T+5 (the next
  // op the bank can take), the retire after: released at the retire.
  d.run("insert then flip, same bank",
        {{1, Inputs().insert(0, 0, 9)}, {5, Inputs().flip(0, 9)}, {8, Inputs().retire(0)}},
        {{9, bit(0)}}, 12);

  // Back to back in consecutive cycles: still woken.
  d.run("insert then flip, next cycle",
        {{1, Inputs().insert(B2, W2, 1)}, {2, Inputs().flip(B2, 1)}, {3, Inputs().retire(W2)}},
        {{4, bit(W2)}}, 6);

  // The flip came first: the lane read before it would have seen the phase
  // complete, so this insert is for the next phase and waits for the limit.
  d.run("flip then insert, same bank",
        {{1, Inputs().flip(0, 9)}, {2, Inputs().insert(0, 0, 9)}, {4, Inputs().retire(0)}},
        {{4 + S + 1, bit(0)}}, S + 8);

  // Entries on banks 0..3 at row 1 (and bank 2 at row 2); one cycle flips
  // row 1 on bank 1 and row 2 on bank 3: only the bank 1 entry wakes.
  if (NUM_BANKS >= 4 && NUM_WARPS >= 4) {
    // The retires end before the flips (one per LSU block per cycle).
    std::map<uint32_t, Inputs> s = {{0, Inputs().insert(0, 0, 1).insert(1, 1, 1).insert(2, 2, 2).insert(3, 3, 1)},
                                    {6, Inputs().flip(1, 1).flip(3, 2)}};
    std::map<uint32_t, Mask> e = {{7, bit(1)}};
    for (uint32_t w = 0; w < 4; ++w) {
      uint32_t r = 1 + w / NUM_RETIRES;
      s[r].retire(w);
      if (w != 1) {
        e[r + S + 1] |= bit(w);
      }
    }
    d.run("flips on other banks, same cycle", s, e, S + 10);
  }

  // Every warp on one object, retiring over a few cycles; one flip wakes all.
  {
    std::map<uint32_t, Inputs> s;
    for (uint32_t w = 0; w < NUM_WARPS; ++w) {
      s[w].insert(B1, w, 0x2a);
    }
    uint32_t c = NUM_WARPS;
    for (uint32_t w = 0; w < NUM_WARPS; ++w) {
      s[c + w / NUM_RETIRES].retire(w);
    }
    uint32_t f = c + NUM_WARPS + 3;
    s[f].flip(B1, 0x2a);
    d.run("all warps on one object", s, {{f + 1, all_warps()}}, f + 4);
  }

  // Every warp on its own object; the flips come one per bank per cycle and
  // each wakes exactly its warp.
  {
    std::map<uint32_t, Inputs> s;
    std::map<uint32_t, Mask> e;
    for (uint32_t w = 0; w < NUM_WARPS; ++w) {
      s[w].insert(w % NUM_BANKS, w, w);
    }
    uint32_t c = NUM_WARPS;
    for (uint32_t w = 0; w < NUM_WARPS; ++w) {
      s[c + w / NUM_RETIRES].retire(w);
    }
    uint32_t f = c + NUM_WARPS + 2;
    for (uint32_t w = 0; w < NUM_WARPS; ++w) {
      uint32_t at = f + 2 * (w / NUM_BANKS);
      s[at].flip(w % NUM_BANKS, w);
      e[at + 1] |= bit(w);
    }
    d.run("each warp on its own object", s, e, f + 2 * NUM_WARPS + 4);
  }

  // A retire and the flip of its object in the same cycle: woken at the retire.
  d.run("retire racing a flip",
        {{0, Inputs().insert(0, 0, 6)}, {5, Inputs().retire(0).flip(0, 6)}},
        {{6, bit(0)}}, 8);

  // The same race against another object: suspended.
  d.run("retire beside another flip",
        {{0, Inputs().insert(0, 0, 6)}, {5, Inputs().retire(0).flip(0, 7)}},
        {{5 + S + 1, bit(0)}}, S + 8);

  // Every lane read the phase complete: no entry, released at the retire.
  d.run("retire with no entry", {{3, Inputs().retire(W1)}}, {{4, bit(W1)}}, 6);

  // Two lanes of warp 0 insert in one cycle on two banks: the higher bank's
  // object is the one waited on.
  if (NUM_BANKS >= 2) {
    d.run("two inserts in one cycle, lower bank flips",
          {{0, Inputs().insert(0, 0, 4).insert(1, 0, 5)}, {1, Inputs().retire(0)},
           {3, Inputs().flip(0, 4)}},
          {{1 + S + 1, bit(0)}}, S + 5);
    d.run("two inserts in one cycle, higher bank flips",
          {{0, Inputs().insert(0, 0, 4).insert(1, 0, 5)}, {1, Inputs().retire(0)},
           {3, Inputs().flip(1, 5)}},
          {{4, bit(0)}}, 6);
  }

  // Lanes one after the other: the last insert is the object waited on...
  d.run("last insert wins",
        {{0, Inputs().insert(0, 0, 1)}, {1, Inputs().insert(B1, 0, 2)}, {2, Inputs().retire(0)},
         {4, Inputs().flip(0, 1)}},
        {{2 + S + 1, bit(0)}}, S + 6);
  // ...but a wake-up already received is kept.
  d.run("wake kept across inserts",
        {{0, Inputs().insert(0, 0, 1)}, {1, Inputs().flip(0, 1)}, {2, Inputs().insert(B1, 0, 2)},
         {3, Inputs().retire(0)}},
        {{4, bit(0)}}, 6);

  // An entry not yet retired is not released by a flip: the flip only marks it.
  d.run("flip before the retire",
        {{0, Inputs().insert(0, 0, 3)}, {1, Inputs().flip(0, 3)}, {9, Inputs().retire(0)}},
        {{10, bit(0)}}, 12);

  // Released warps start over: a second try_wait of the same warp.
  d.run("second try_wait of a warp",
        {{0, Inputs().insert(0, 0, 3)}, {1, Inputs().retire(0)}, {2, Inputs().flip(0, 3)},
         {6, Inputs().insert(0, 0, 3)}, {7, Inputs().retire(0)}, {9, Inputs().flip(0, 3)}},
        {{3, bit(0)}, {10, bit(0)}}, 12);
}

///////////////////////////////////////////////////////////////////////////////
// Random try_wait lifecycles: each warp issues a try_wait whose lanes read
// objects over a few cycles (inserting when not complete), then retires; the
// warp is suspended until its unlock comes back, then idles and starts over.
// Flips land on a small set of objects so many of them wake waiters.

static void random_run(Bench& tb, uint32_t seed, uint32_t cycles) {
  std::mt19937 rng(seed);
  auto below = [&](uint32_t n) { return static_cast<uint32_t>(rng() % n); };

  enum class St { IDLE, READING, SUSPENDED };
  struct Warp {
    St       st = St::IDLE;
    uint32_t lanes = 0;
    uint64_t retired_at = 0;
    uint32_t wait = 0;
  };
  std::array<Warp, NUM_WARPS> warps{};
  constexpr uint32_t ROWS = 4;
  uint64_t retires = 0, inserts = 0, flips = 0, wakes = 0;
  uint64_t start = tb.now();
  uint32_t before = tb.failures();

  for (uint32_t c = 0; c < cycles + SUSPEND + 64; ++c) {
    bool draining = c >= cycles;
    Inputs in;
    std::array<bool, NUM_BANKS> busy{};
    std::array<bool, NUM_RETIRES> rbusy{};
    for (uint32_t w = 0; w < NUM_WARPS; ++w) {
      Warp& x = warps[w];
      if (x.st == St::IDLE) {
        if (x.wait != 0) {
          --x.wait;
        } else if (!draining && below(4) == 0) {
          x.st = St::READING;
          x.lanes = 1 + below(4);
        }
      } else if (x.st == St::READING) {
        if (x.lanes != 0) {
          uint32_t b = below(NUM_BANKS);
          if (!busy[b] && below(2) == 0) {
            busy[b] = true;
            --x.lanes;
            if (below(3) != 0) {
              in.insert(b, w, below(ROWS));
              ++inserts;
            }
          }
        } else if (!rbusy[w % NUM_RETIRES] && below(2) == 0) {
          rbusy[w % NUM_RETIRES] = true;
          in.retire(w);
          x.st = St::SUSPENDED;
          x.retired_at = tb.now();
          ++retires;
        }
      }
    }
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      if (!busy[b] && below(draining ? 64 : 6) == 0) {
        in.flip(b, below(ROWS));
        ++flips;
      }
    }
    Mask got = tb.cycle(in);
    for (uint32_t w = 0; w < NUM_WARPS; ++w) {
      if (!((got >> w) & 1)) {
        continue;
      }
      Warp& x = warps[w];
      if (x.st != St::SUSPENDED) {
        tb.fail(strf("random: cycle %" PRIu64 ": unlock of warp %u, which is not suspended", tb.now() - 1, w));
        continue;
      }
      uint64_t waited = tb.now() - 1 - x.retired_at;
      if (waited > SUSPEND + 1) {
        tb.fail(strf("random: warp %u unlocked %" PRIu64 " cycles after its retire", w, waited));
      }
      wakes += (waited <= SUSPEND);
      x.st = St::IDLE;
      x.wait = 1 + below(3);
    }
  }
  for (uint32_t w = 0; w < NUM_WARPS; ++w) {
    if (warps[w].st != St::IDLE) {
      tb.fail(strf("random: warp %u never released", w));
    }
  }
  std::printf("  random seed=%u: %" PRIu64 " cycles, %" PRIu64 " retires, %" PRIu64 " inserts, %" PRIu64
              " flips, %" PRIu64 " released before the limit %s\n",
              seed, tb.now() - start, retires, inserts, flips, wakes, (tb.failures() == before) ? "ok" : "FAILED");
}

static int negative(Bench& tb, const std::string& kind) {
  if (kind != "insert") {
    std::printf("unknown --neg=%s\n", kind.c_str());
    return 2;
  }
  tb.cycle(Inputs().insert(0, 0, 1));
  tb.cycle(Inputs().retire(0));
  tb.cycle(Inputs().insert(0, 0, 1));
  for (int i = 0; i < 4; ++i) {
    tb.cycle(Inputs{});
  }
  std::printf("the table took an insert for a retired warp without asserting\n");
  return 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  uint32_t seed = 1;
  uint32_t cycles = 20000;
  std::string neg;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strncmp(a, "--seed=", 7) == 0) {
      seed = static_cast<uint32_t>(std::strtoul(a + 7, nullptr, 0));
    } else if (std::strncmp(a, "--cycles=", 9) == 0) {
      cycles = static_cast<uint32_t>(std::strtoul(a + 9, nullptr, 0));
    } else if (std::strncmp(a, "--neg=", 6) == 0) {
      neg = a + 6;
    } else if (a[0] == '+') {
      continue;
    } else {
      std::printf("usage: %s [--seed=N] [--cycles=N] [--neg=insert]\n", argv[0]);
      return 2;
    }
  }

  std::printf("mbar_waiters: warps=%u banks=%u lsu_blocks=%u suspend=%u seed=%u\n",
              NUM_WARPS, NUM_BANKS, NUM_RETIRES, SUSPEND, seed);
  Bench tb;
  if (!neg.empty()) {
    return negative(tb, neg);
  }
  Directed d(tb);
  // The scenarios place their flips before the time limit of the warps they wake.
  if (SUSPEND >= NUM_WARPS + 2 * NUM_WARPS / NUM_BANKS + 8) {
    directed(tb, d);
  } else {
    std::printf("  directed scenarios skipped: the time limit is below %u cycles\n",
                NUM_WARPS + 2 * NUM_WARPS / NUM_BANKS + 8);
  }
  for (uint32_t s = 0; s < 3; ++s) {
    random_run(tb, seed * 7919u + s, cycles);
  }
  std::printf("%u scenarios, %" PRIu64 " unlocks (%" PRIu64 " by the time limit), %" PRIu64 " cycles\n",
              d.count(), tb.unlocks(), tb.timeouts(), tb.now());
  if (tb.failures() != 0) {
    std::printf("FAILED: %u check(s) failed\n", tb.failures());
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
