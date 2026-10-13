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

// VX_lmem_switch testbench: LSU requests with single-side and mixed lane
// masks (loads, stores, AMOs) under independent random back-pressure on the
// memory ports: global and local, plus remote in DLMEM builds. Each port must
// see every request that has lanes for it exactly once, in order, with its
// lane subset and an unchanged payload, and every port response must reach the
// LSU exactly once. In DLMEM builds the round-robin response merge is also
// checked: a waiting port response is passed over at most NUM_SIDES-1 times.

#include "vl_simulator.h"
#include "VVX_lmem_switch_top.h"
#include "VX_config.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

constexpr uint32_t XLEN        = VX_CFG_XLEN;
constexpr uint32_t WORD_SIZE   = XLEN / 8;
constexpr uint32_t NUM_LANES   = VX_CFG_NUM_LSU_LANES;
// Must match the VX_lmem_switch_top localparams (checked there).
constexpr uint32_t ADDR_BITS   = VX_CFG_MEM_ADDR_WIDTH - __CLOG2(WORD_SIZE);
constexpr uint32_t TAG_BITS    = __CLOG2(VX_CFG_LSU_PENDING_SIZE);
constexpr uint32_t AMO_OP_BITS = 4;
constexpr uint32_t NUM_AMO_OPS = 10;

constexpr uint64_t WORD_MASK = (XLEN == 64) ? ~0ull : 0xffffffffull;
constexpr uint64_t ADDR_MASK = (ADDR_BITS >= 64) ? ~0ull : ((1ull << ADDR_BITS) - 1);
constexpr uint32_t LANE_MASK = (1u << NUM_LANES) - 1;
constexpr uint32_t BE_MASK   = (1u << WORD_SIZE) - 1;
constexpr uint32_t TAG_MASK  = (1u << TAG_BITS) - 1;

constexpr int GLOBAL = 0;
constexpr int LOCAL  = 1;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
constexpr int REMOTE    = 2;
constexpr int NUM_SIDES = 3;
// The top's DLMEM default: round-robin merge, so fairness is checkable.
constexpr bool RR_MERGE = true;
#else
constexpr int REMOTE    = -1;
constexpr int NUM_SIDES = 2;
constexpr bool RR_MERGE = false;
#endif
static const char* const side_name[3] = {"global", "local", "remote"};

static_assert(NUM_LANES >= 1 && NUM_LANES <= 32, "unsupported lane count");
static_assert(WORD_SIZE == 4 || WORD_SIZE == 8, "unsupported XLEN");

///////////////////////////////////////////////////////////////////////////////

// Bit-field access that works on every Verilator port type (CData..QData
// and VlWide), so the testbench is independent of the configured widths.
template <typename T>
static void set_bits(T& sig, uint32_t lsb, uint32_t width, uint64_t value) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    uint64_t v = static_cast<uint64_t>(sig);
    v = (v & ~(mask << lsb)) | ((value & mask) << lsb);
    sig = static_cast<T>(v);
  } else {
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      uint32_t chunk = static_cast<uint32_t>(value >> i) & m;
      sig[w] = (sig[w] & ~(m << o)) | (chunk << o);
      i += n;
    }
  }
}

template <typename T>
static uint64_t get_bits(const T& sig, uint32_t lsb, uint32_t width) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    return (static_cast<uint64_t>(sig) >> lsb) & mask;
  } else {
    uint64_t value = 0;
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      value |= static_cast<uint64_t>((sig[w] >> o) & m) << i;
      i += n;
    }
    return value;
  }
}

static std::string strf(const char* fmt, ...) {
  char buf[1024];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

///////////////////////////////////////////////////////////////////////////////

struct Req {
  uint64_t seq    = 0;
  bool     rw     = false;
  uint32_t mask   = 0;
  uint32_t local  = 0; // per-lane is_addr_local, also on inactive lanes
  uint32_t remote = 0; // per-lane is_addr_remote, also on inactive lanes
  uint32_t amo    = 0; // per-lane amo_valid
  uint32_t tag    = 0;
  std::array<uint64_t, NUM_LANES> addr{};
  std::array<uint64_t, NUM_LANES> data{};
  std::array<uint32_t, NUM_LANES> byteen{};
  std::array<uint32_t, NUM_LANES> amo_op{};

  // A remote lane goes remote whatever its local bit says.
  uint32_t side_mask(int side) const {
    if (side == REMOTE) {
      return mask & remote;
    }
    return (side == LOCAL) ? (mask & local & ~remote) : (mask & ~local & ~remote & LANE_MASK);
  }
  int num_sides() const {
    int n = 0;
    for (int s = 0; s < NUM_SIDES; ++s) {
      n += (side_mask(s) != 0);
    }
    return n;
  }
  bool expects_rsp() const {
    return !rw;
  }
};

struct Rsp {
  uint64_t seq  = 0;
  int      side = GLOBAL;
  uint32_t mask = 0;
  uint32_t tag  = 0;
  std::array<uint64_t, NUM_LANES> data{};
};

static bool same_payload(const Req& a, const Req& b) {
  return a.rw == b.rw && a.mask == b.mask && a.local == b.local && a.remote == b.remote && a.amo == b.amo
      && a.tag == b.tag && a.addr == b.addr && a.data == b.data
      && a.byteen == b.byteen && a.amo_op == b.amo_op;
}

static bool same_payload(const Rsp& a, const Rsp& b) {
  return a.mask == b.mask && a.tag == b.tag && a.data == b.data;
}

static std::string to_string(const Req& r) {
  std::string s = strf("{%s mask=0x%x local=0x%x remote=0x%x amo=0x%x tag=0x%x", r.rw ? "st" : (r.amo ? "amo" : "ld"),
                       r.mask, r.local, r.remote, r.amo, r.tag);
  for (uint32_t i = 0; i < NUM_LANES; ++i) {
    s += strf(" [%u]a=0x%" PRIx64 ",d=0x%" PRIx64 ",be=0x%x,op=%u", i, r.addr[i], r.data[i], r.byteen[i], r.amo_op[i]);
  }
  return s + "}";
}

static std::string to_string(const Rsp& r) {
  std::string s = strf("{mask=0x%x tag=0x%x", r.mask, r.tag);
  for (uint32_t i = 0; i < NUM_LANES; ++i) {
    s += strf(" [%u]0x%" PRIx64, i, r.data[i]);
  }
  return s + "}";
}

///////////////////////////////////////////////////////////////////////////////

struct Options {
  uint32_t seed     = 1;
  uint32_t requests = 20000;
  uint32_t watchdog = 4000;
  bool     verbose  = false;
};

// Per-side readiness: a probability in percent, or a forced level.
struct Pressure {
  uint32_t req_ready_pct = 100;
  uint32_t rsp_valid_pct = 100;
  std::optional<bool> req_ready_force;
};

class Testbench {
public:
  explicit Testbench(const Options& opts)
    : opts_(opts)
    , rng_(opts.seed) {
    auto* t = sim_.operator->();
    t->reset = 1;
    this->drive();
    for (int i = 0; i < 8; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
  }

  int run() {
    std::printf("lmem_switch: XLEN=%u lanes=%u sides=%d addr_bits=%u tag_bits=%u seed=%u requests=%u\n",
                XLEN, NUM_LANES, NUM_SIDES, ADDR_BITS, TAG_BITS, opts_.seed, opts_.requests);
    this->directed();
    if (RR_MERGE && failures_ == 0) {
      this->merge_fairness();
    }
    if (failures_ == 0) {
      this->random();
    }
    this->drain("final");
    this->final_check();
    std::string ports;
    for (int s = 0; s < NUM_SIDES; ++s) {
      ports += strf(" %s %" PRIu64, side_name[s], n_out_[s]);
    }
    std::printf("total cycles: %" PRIu64 ", requests: %" PRIu64 " (mixed %" PRIu64 ", all sides %" PRIu64 "), "
                "port requests:%s, responses: %" PRIu64 "\n",
                cycle_, n_in_, n_mixed_, n_all_sides_, ports.c_str(), n_rsp_);
    if (RR_MERGE) {
      std::printf("response merge: most times a waiting response was passed over: global %u local %u remote %u "
                  "(bound %d), cycles with all port responses waiting: %" PRIu64 "\n",
                  max_bypass_[GLOBAL], max_bypass_[LOCAL], max_bypass_[NUM_SIDES - 1], NUM_SIDES - 1, n_all_rsp_);
    }
    if (failures_ != 0) {
      std::printf("FAILED: %u check(s) failed\n", failures_);
      return 1;
    }
    std::printf("PASSED\n");
    return 0;
  }

private:
  ///////////////////////////////////////////////////////////////////////////
  // stimulus

  uint32_t rand_below(uint32_t n) {
    return static_cast<uint32_t>(rng_() % n);
  }

  bool chance(uint32_t pct) {
    return rand_below(100) < pct;
  }

  // A request whose active lanes cover exactly the sides in `sides`.
  Req make_req_sides(uint32_t op, const std::vector<int>& sides) {
    Req r;
    r.rw = (op == 1);
    const uint32_t need = static_cast<uint32_t>(sides.size());
    do {
      r.mask = static_cast<uint32_t>(rng_()) & LANE_MASK;
    } while (static_cast<uint32_t>(__builtin_popcount(r.mask)) < need);
    std::vector<uint32_t> lanes;
    for (uint32_t i = 0; i < NUM_LANES; ++i) {
      if ((r.mask >> i) & 1) {
        lanes.push_back(i);
      }
    }
    std::shuffle(lanes.begin(), lanes.end(), rng_);
    for (size_t k = 0; k < lanes.size(); ++k) {
      int side = (k < need) ? sides[k] : sides[this->rand_below(need)];
      uint32_t bit = 1u << lanes[k];
      if (side == LOCAL) {
        r.local |= bit;
      } else if (side == REMOTE) {
        r.remote |= bit;
        if (this->chance(50)) {
          r.local |= bit;
        }
      }
    }
    uint32_t inactive = ~r.mask & LANE_MASK;
    r.local |= static_cast<uint32_t>(rng_()) & inactive;
    if (NUM_SIDES > 2) {
      r.remote |= static_cast<uint32_t>(rng_()) & inactive;
    }
    r.amo = (op == 2) ? LANE_MASK : 0;
    r.tag = static_cast<uint32_t>(rng_()) & TAG_MASK;
    uint32_t amo_op = rand_below(NUM_AMO_OPS);
    for (uint32_t i = 0; i < NUM_LANES; ++i) {
      r.addr[i]   = rng_() & ADDR_MASK;
      r.data[i]   = rng_() & WORD_MASK;
      r.byteen[i] = static_cast<uint32_t>(rng_()) & BE_MASK;
      r.amo_op[i] = (op == 2) ? amo_op : rand_below(NUM_AMO_OPS);
    }
    return r;
  }

  // kind: a side index for a single-side request, or MIXED for two or more sides
  static constexpr int MIXED = NUM_SIDES;

  Req make_req(int kind) {
    uint32_t op = rand_below(3); // load, store, AMO
    if (kind != MIXED) {
      return this->make_req_sides(op, {kind});
    }
    std::vector<int> sides;
    do {
      sides.clear();
      for (int s = 0; s < NUM_SIDES; ++s) {
        if (this->chance(60)) {
          sides.push_back(s);
        }
      }
    } while (sides.size() < 2 || sides.size() > NUM_LANES);
    return this->make_req_sides(op, sides);
  }

  int rand_kind() {
    if (NUM_LANES < 2) {
      return rand_below(NUM_SIDES);
    }
    // 40% single-side requests, split evenly over the sides
    uint32_t p = rand_below(100 * NUM_SIDES);
    return (p < 40u * NUM_SIDES) ? static_cast<int>(p / 40u) : MIXED;
  }

  // Response words carry the request sequence and side, so a misrouted or
  // repeated response cannot alias a legitimate one.
  Rsp make_rsp(const Req& r, int side) {
    Rsp p;
    p.seq  = r.seq;
    p.side = side;
    p.mask = r.side_mask(side);
    p.tag  = r.tag;
    for (uint32_t i = 0; i < NUM_LANES; ++i) {
      uint64_t id = (r.seq << 7) | (static_cast<uint64_t>(side) << 5) | i;
      p.data[i] = ((p.mask >> i) & 1) ? ((id * 0x9e3779b97f4a7c15ull) & WORD_MASK) : (rng_() & WORD_MASK);
    }
    return p;
  }

  ///////////////////////////////////////////////////////////////////////////
  // DUT access

  void drive() {
    auto* t = sim_.operator->();
    Req r = in_req_.value_or(Req{});
    t->in_req_valid = in_req_.has_value();
    t->in_req_rw    = r.rw;
    set_bits(t->in_req_mask, 0, NUM_LANES, r.mask);
    set_bits(t->in_req_local, 0, NUM_LANES, r.local);
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    set_bits(t->in_req_remote, 0, NUM_LANES, r.remote);
#endif
    set_bits(t->in_req_amo, 0, NUM_LANES, r.amo);
    set_bits(t->in_req_tag, 0, TAG_BITS, r.tag);
    for (uint32_t i = 0; i < NUM_LANES; ++i) {
      set_bits(t->in_req_addr, i * ADDR_BITS, ADDR_BITS, r.addr[i]);
      set_bits(t->in_req_data, i * XLEN, XLEN, r.data[i]);
      set_bits(t->in_req_byteen, i * WORD_SIZE, WORD_SIZE, r.byteen[i]);
      set_bits(t->in_req_amo_op, i * AMO_OP_BITS, AMO_OP_BITS, r.amo_op[i]);
    }
    t->in_rsp_ready = in_rsp_ready_;
    for (int s = 0; s < NUM_SIDES; ++s) {
      Rsp p = out_rsp_[s].value_or(Rsp{});
      set_bits(t->out_req_ready, s, 1, out_req_ready_[s]);
      set_bits(t->out_rsp_valid, s, 1, out_rsp_[s].has_value());
      set_bits(t->out_rsp_mask, s * NUM_LANES, NUM_LANES, p.mask);
      set_bits(t->out_rsp_tag, s * TAG_BITS, TAG_BITS, p.tag);
      for (uint32_t i = 0; i < NUM_LANES; ++i) {
        set_bits(t->out_rsp_data, (s * NUM_LANES + i) * XLEN, XLEN, p.data[i]);
      }
    }
  }

  Req sample_out_req(int s) {
    auto* t = sim_.operator->();
    Req r;
    r.rw    = get_bits(t->out_req_rw, s, 1);
    r.mask  = static_cast<uint32_t>(get_bits(t->out_req_mask, s * NUM_LANES, NUM_LANES));
    r.local = static_cast<uint32_t>(get_bits(t->out_req_local, s * NUM_LANES, NUM_LANES));
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    r.remote = static_cast<uint32_t>(get_bits(t->out_req_remote, s * NUM_LANES, NUM_LANES));
#endif
    r.amo   = static_cast<uint32_t>(get_bits(t->out_req_amo, s * NUM_LANES, NUM_LANES));
    r.tag   = static_cast<uint32_t>(get_bits(t->out_req_tag, s * TAG_BITS, TAG_BITS));
    for (uint32_t i = 0; i < NUM_LANES; ++i) {
      uint32_t l = s * NUM_LANES + i;
      r.addr[i]   = get_bits(t->out_req_addr, l * ADDR_BITS, ADDR_BITS);
      r.data[i]   = get_bits(t->out_req_data, l * XLEN, XLEN);
      r.byteen[i] = static_cast<uint32_t>(get_bits(t->out_req_byteen, l * WORD_SIZE, WORD_SIZE));
      r.amo_op[i] = static_cast<uint32_t>(get_bits(t->out_req_amo_op, l * AMO_OP_BITS, AMO_OP_BITS));
    }
    return r;
  }

  Rsp sample_in_rsp() {
    auto* t = sim_.operator->();
    Rsp p;
    p.mask = static_cast<uint32_t>(get_bits(t->in_rsp_mask, 0, NUM_LANES));
    p.tag  = static_cast<uint32_t>(get_bits(t->in_rsp_tag, 0, TAG_BITS));
    for (uint32_t i = 0; i < NUM_LANES; ++i) {
      p.data[i] = get_bits(t->in_rsp_data, i * XLEN, XLEN);
    }
    return p;
  }

  void fail(const std::string& msg) {
    ++failures_;
    if (failures_ <= 20) {
      std::printf("FAIL [cycle %" PRIu64 "] %s\n", cycle_, msg.c_str());
    } else if (failures_ == 21) {
      std::printf("... further failure messages suppressed\n");
    }
  }

  // One cycle: drive, sample every handshake before the edge, check, clock.
  void clock() {
    auto* t = sim_.operator->();
    for (int s = 0; s < NUM_SIDES; ++s) {
      if (pressure_[s].req_ready_force) {
        out_req_ready_[s] = *pressure_[s].req_ready_force;
      } else {
        out_req_ready_[s] = this->chance(pressure_[s].req_ready_pct);
      }
      if (!out_rsp_[s] && !mem_q_[s].empty() && mem_q_[s].front().first <= cycle_
       && this->chance(pressure_[s].rsp_valid_pct)) {
        out_rsp_[s] = mem_q_[s].front().second;
        mem_q_[s].pop_front();
      }
    }
    in_rsp_ready_ = this->chance(lsu_rsp_ready_pct_);
    this->drive();
    t->eval();

    bool progress = false;

    if (in_req_ && t->in_req_ready) {
      const Req& r = *in_req_;
      for (int s = 0; s < NUM_SIDES; ++s) {
        if (r.side_mask(s) != 0) {
          Req e = r;
          e.mask = r.side_mask(s);
          exp_req_[s].push_back(e);
        }
      }
      if (r.num_sides() >= 2) {
        ++n_mixed_;
      }
      if (r.num_sides() == NUM_SIDES) {
        ++n_all_sides_;
      }
      issued_.push_back(r);
      ++n_in_;
      in_req_.reset();
      progress = true;
    }

    int n_rsp_taken = 0;
    bool all_rsp_waiting = true;
    std::array<bool, NUM_SIDES> rsp_waiting{};
    std::array<bool, NUM_SIDES> rsp_taken{};
    for (int s = 0; s < NUM_SIDES; ++s) {
      rsp_waiting[s] = out_rsp_[s].has_value();
      all_rsp_waiting = all_rsp_waiting && rsp_waiting[s];
    }
    for (int s = 0; s < NUM_SIDES; ++s) {
      if (get_bits(t->out_req_valid, s, 1) && out_req_ready_[s]) {
        Req got = this->sample_out_req(s);
        ++n_out_[s];
        progress = true;
        if (exp_req_[s].empty()) {
          this->fail(strf("%s port: request with no matching LSU request (duplicate?): %s",
                          side_name[s], to_string(got).c_str()));
          continue;
        }
        Req exp = exp_req_[s].front();
        exp_req_[s].pop_front();
        if (!same_payload(got, exp)) {
          this->fail(strf("%s port: request mismatch for seq %" PRIu64 "\n  expected %s\n  got      %s",
                          side_name[s], exp.seq, to_string(exp).c_str(), to_string(got).c_str()));
          continue;
        }
        if (exp.expects_rsp()) {
          uint32_t lat = 1 + this->rand_below(max_lat_);
          mem_q_[s].emplace_back(cycle_ + lat, this->make_rsp(exp, s));
        }
      }
      if (out_rsp_[s] && get_bits(t->out_rsp_ready, s, 1)) {
        exp_rsp_[s].push_back(*out_rsp_[s]);
        out_rsp_[s].reset();
        rsp_taken[s] = true;
        ++n_rsp_taken;
        progress = true;
      }
    }
    if (n_rsp_taken > 1) {
      this->fail(strf("%d port responses accepted in one cycle, the merge takes one", n_rsp_taken));
    }
    if (RR_MERGE) {
      n_all_rsp_ += all_rsp_waiting;
      for (int s = 0; s < NUM_SIDES; ++s) {
        if (rsp_taken[s]) {
          bypass_[s] = 0;
        } else if (rsp_waiting[s] && n_rsp_taken != 0) {
          max_bypass_[s] = std::max(max_bypass_[s], ++bypass_[s]);
          if (bypass_[s] == NUM_SIDES) {
            this->fail(strf("%s port response passed over %d times in a row by the round-robin merge",
                            side_name[s], NUM_SIDES));
          }
        }
      }
    }

    if (t->in_rsp_valid && in_rsp_ready_) {
      Rsp got = this->sample_in_rsp();
      ++n_rsp_;
      progress = true;
      bool matched = false;
      for (int s = 0; s < NUM_SIDES && !matched; ++s) {
        if (!exp_rsp_[s].empty() && same_payload(got, exp_rsp_[s].front())) {
          ++rsp_count_[s].at(exp_rsp_[s].front().seq);
          exp_rsp_[s].pop_front();
          matched = true;
        }
      }
      if (!matched) {
        std::string heads;
        for (int s = 0; s < NUM_SIDES; ++s) {
          heads += strf("\n  %s head: %s", side_name[s],
                        exp_rsp_[s].empty() ? "(none)" : to_string(exp_rsp_[s].front()).c_str());
        }
        this->fail(strf("LSU response matches neither port's next response: %s%s",
                        to_string(got).c_str(), heads.c_str()));
      }
    }

    tick_ = sim_.step(tick_, 2);
    ++cycle_;
    if (progress) {
      last_progress_ = cycle_;
    }
  }

  bool outstanding() const {
    bool any = in_req_.has_value();
    for (int s = 0; s < NUM_SIDES; ++s) {
      any = any || !exp_req_[s].empty() || !mem_q_[s].empty() || out_rsp_[s].has_value() || !exp_rsp_[s].empty();
    }
    return any;
  }

  bool watchdog() {
    if (cycle_ - last_progress_ <= opts_.watchdog) {
      return true;
    }
    std::string owed;
    for (int s = 0; s < NUM_SIDES; ++s) {
      owed += strf(", %s: %zu port request(s), %zu response(s) owed to the LSU", side_name[s], exp_req_[s].size(),
                   exp_rsp_[s].size() + mem_q_[s].size() + out_rsp_[s].has_value());
    }
    this->fail(strf("no progress for %u cycles: input %s%s",
                    opts_.watchdog, in_req_ ? "pending" : "idle", owed.c_str()));
    return false;
  }

  void submit(Req r) {
    r.seq = rsp_count_[0].size();
    in_req_ = r;
    for (int s = 0; s < NUM_SIDES; ++s) {
      rsp_count_[s].push_back(0);
    }
  }

  void drain(const char* ctx) {
    for (int s = 0; s < NUM_SIDES; ++s) {
      pressure_[s] = Pressure{};
    }
    lsu_rsp_ready_pct_ = 100;
    last_progress_ = cycle_;
    while (this->outstanding()) {
      this->clock();
      if (!this->watchdog()) {
        std::printf("drain (%s) aborted\n", ctx);
        return;
      }
    }
    for (int i = 0; i < 4; ++i) {
      this->clock();
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // directed: a single-port request fills the stalled port's output buffer,
  // then a mixed request waits on it for several cycles while the other ports
  // are ready the whole time; each ready port must take the mixed request once.

  void wait_accept() {
    last_progress_ = cycle_;
    while (in_req_) {
      this->clock();
      if (!this->watchdog()) {
        in_req_.reset();
        return;
      }
    }
  }

  void directed() {
    if (NUM_LANES < 2) {
      std::printf("directed: one lane cannot mix sides, skipped\n");
      return;
    }
    for (int stalled = 0; stalled < NUM_SIDES; ++stalled) {
      // Every other side when the lanes allow it, else one other side in turn.
      std::vector<std::vector<int>> mixes;
      if (NUM_LANES >= static_cast<uint32_t>(NUM_SIDES)) {
        std::vector<int> all;
        for (int s = 0; s < NUM_SIDES; ++s) {
          all.push_back(s);
        }
        mixes.push_back(all);
      } else {
        for (int s = 0; s < NUM_SIDES; ++s) {
          if (s != stalled) {
            mixes.push_back({stalled, s});
          }
        }
      }
      for (const auto& sides : mixes) {
        for (uint32_t op = 0; op < 3; ++op) {
          uint32_t before = failures_;
          std::array<uint64_t, NUM_SIDES> n_before = n_out_;
          for (int s = 0; s < NUM_SIDES; ++s) {
            pressure_[s].req_ready_force = (s != stalled);
          }
          this->submit(this->make_req_sides(rand_below(3), {stalled}));
          this->wait_accept();
          this->submit(this->make_req_sides(op, sides));
          for (int i = 0; i < 8; ++i) {
            this->clock();
          }
          pressure_[stalled].req_ready_force = true;
          this->wait_accept();
          this->drain("directed");
          const char* kind = (op == 0) ? "load" : (op == 1) ? "store" : "AMO";
          std::string got_s, want_s, mix_s;
          bool bad = false;
          for (int s = 0; s < NUM_SIDES; ++s) {
            bool in_mix = std::find(sides.begin(), sides.end(), s) != sides.end();
            uint64_t want = (s == stalled) ? 2 : (in_mix ? 1 : 0);
            uint64_t got = n_out_[s] - n_before[s];
            bad = bad || (got != want);
            got_s += strf(" %s %" PRIu64, side_name[s], got);
            want_s += strf(" %s %" PRIu64, side_name[s], want);
            if (in_mix) {
              mix_s += strf("%s%s", mix_s.empty() ? "" : "+", side_name[s]);
            }
          }
          if (bad) {
            this->fail(strf("directed %s over %s, %s port stalled: ports took%s request(s), expected%s",
                            kind, mix_s.c_str(), side_name[stalled], got_s.c_str(), want_s.c_str()));
          }
          std::printf("directed: %s over %s, %s port stalled 8 cycles: %s\n", kind, mix_s.c_str(),
                      side_name[stalled], (failures_ == before) ? "ok" : "FAILED");
        }
      }
    }
  }

  // Every port holds responses while the LSU takes one per cycle: the merge
  // must rotate over the ports (the per-cycle bound is checked in clock()).
  void merge_fairness() {
    const uint32_t per_side = 24;
    uint32_t before = failures_;
    uint64_t all_before = n_all_rsp_;
    for (int s = 0; s < NUM_SIDES; ++s) {
      pressure_[s].req_ready_force = true;
      pressure_[s].rsp_valid_pct = 0;
    }
    lsu_rsp_ready_pct_ = 100;
    max_lat_ = 1;
    for (uint32_t n = 0; n < per_side; ++n) {
      for (int s = 0; s < NUM_SIDES; ++s) {
        this->submit(this->make_req_sides(0, {s}));
        this->wait_accept();
      }
    }
    for (int i = 0; i < 4; ++i) {
      this->clock();
    }
    for (int s = 0; s < NUM_SIDES; ++s) {
      pressure_[s].rsp_valid_pct = 100;
    }
    this->drain("merge fairness");
    uint64_t all_waiting = n_all_rsp_ - all_before;
    if (all_waiting < per_side) {
      this->fail(strf("merge fairness: every port had a response waiting on only %" PRIu64 " cycle(s)", all_waiting));
    }
    std::printf("merge fairness: %u loads per port, every port waiting on %" PRIu64 " cycle(s): %s\n",
                per_side, all_waiting, (failures_ == before) ? "ok" : "FAILED");
  }

  ///////////////////////////////////////////////////////////////////////////
  // random: phases of independent readiness on each port and on the LSU
  // response, random request gaps and memory latency.

  void new_phase() {
    static const uint32_t levels[] = {0, 10, 30, 50, 80, 100};
    for (int s = 0; s < NUM_SIDES; ++s) {
      pressure_[s].req_ready_force.reset();
      pressure_[s].req_ready_pct = levels[this->rand_below(6)];
      pressure_[s].rsp_valid_pct = levels[1 + this->rand_below(5)];
    }
    lsu_rsp_ready_pct_ = levels[1 + this->rand_below(5)];
    gap_pct_ = levels[this->rand_below(4)];
    max_lat_ = 1 + this->rand_below(16);
    phase_end_ = cycle_ + 16 + this->rand_below(128);
  }

  void random() {
    this->new_phase();
    last_progress_ = cycle_;
    uint32_t sent = 0;
    while (sent < opts_.requests) {
      if (cycle_ >= phase_end_) {
        this->new_phase();
      }
      if (!in_req_ && !this->chance(gap_pct_)) {
        this->submit(this->make_req(this->rand_kind()));
        ++sent;
      }
      this->clock();
      if (!this->watchdog() || failures_ > 20) {
        in_req_.reset();
        return;
      }
    }
    if (opts_.verbose) {
      std::printf("random: %u requests issued by cycle %" PRIu64 "\n", sent, cycle_);
    }
  }

  ///////////////////////////////////////////////////////////////////////////

  void final_check() {
    for (int s = 0; s < NUM_SIDES; ++s) {
      if (!exp_req_[s].empty()) {
        this->fail(strf("%s port: %zu request(s) never arrived (first seq %" PRIu64 ")",
                        side_name[s], exp_req_[s].size(), exp_req_[s].front().seq));
      }
    }
    uint32_t bad = 0;
    for (const Req& r : issued_) {
      for (int s = 0; s < NUM_SIDES; ++s) {
        uint32_t want = (r.expects_rsp() && r.side_mask(s) != 0) ? 1 : 0;
        uint32_t got = rsp_count_[s].at(r.seq);
        if (got != want && ++bad <= 10) {
          this->fail(strf("seq %" PRIu64 ": %u %s response(s) reached the LSU, expected %u",
                          r.seq, got, side_name[s], want));
        }
      }
    }
  }

  Options opts_;
  std::mt19937_64 rng_;
  vl_simulator<VVX_lmem_switch_top> sim_;
  uint64_t tick_  = 0;
  uint64_t cycle_ = 0;
  uint64_t last_progress_ = 0;

  std::optional<Req> in_req_;
  bool in_rsp_ready_ = true;
  std::array<bool, NUM_SIDES> out_req_ready_{};
  std::array<std::optional<Rsp>, NUM_SIDES> out_rsp_;
  std::array<Pressure, NUM_SIDES> pressure_;
  uint32_t lsu_rsp_ready_pct_ = 100;
  uint32_t gap_pct_ = 0;
  uint32_t max_lat_ = 4;
  uint64_t phase_end_ = 0;

  std::array<std::deque<Req>, NUM_SIDES> exp_req_;
  std::array<std::deque<std::pair<uint64_t, Rsp>>, NUM_SIDES> mem_q_;
  std::array<std::deque<Rsp>, NUM_SIDES> exp_rsp_;
  std::array<std::vector<uint32_t>, NUM_SIDES> rsp_count_;
  std::vector<Req> issued_;

  std::array<uint32_t, NUM_SIDES> bypass_{};
  std::array<uint32_t, NUM_SIDES> max_bypass_{};
  uint64_t n_all_rsp_ = 0;

  uint64_t n_in_ = 0;
  uint64_t n_mixed_ = 0;
  uint64_t n_all_sides_ = 0;
  std::array<uint64_t, NUM_SIDES> n_out_{};
  uint64_t n_rsp_ = 0;
  uint32_t failures_ = 0;
};

static void usage(const char* prog) {
  std::printf("usage: %s [--seed=N] [--requests=N] [--watchdog=N] [--verbose]\n", prog);
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  Options opts;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (a[0] == '+') {
      continue;
    }
    if (std::strncmp(a, "--seed=", 7) == 0) {
      opts.seed = static_cast<uint32_t>(std::strtoul(a + 7, nullptr, 0));
    } else if (std::strncmp(a, "--requests=", 11) == 0) {
      opts.requests = static_cast<uint32_t>(std::strtoul(a + 11, nullptr, 0));
    } else if (std::strncmp(a, "--watchdog=", 11) == 0) {
      opts.watchdog = static_cast<uint32_t>(std::strtoul(a + 11, nullptr, 0));
    } else if (std::strcmp(a, "--verbose") == 0) {
      opts.verbose = true;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  Testbench tb(opts);
  return tb.run();
}
