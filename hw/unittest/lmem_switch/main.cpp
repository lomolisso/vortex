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

// VX_lmem_switch testbench: LSU requests with all-global, all-local and mixed
// lane masks (loads, stores, AMOs) under independent random back-pressure on
// the two memory ports. Each port must see every request that has lanes for it
// exactly once, in order, with its lane subset and an unchanged payload, and
// every port response must reach the LSU exactly once.

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
static const char* const side_name[2] = {"global", "local"};

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
  uint32_t amo    = 0; // per-lane amo_valid
  uint32_t tag    = 0;
  std::array<uint64_t, NUM_LANES> addr{};
  std::array<uint64_t, NUM_LANES> data{};
  std::array<uint32_t, NUM_LANES> byteen{};
  std::array<uint32_t, NUM_LANES> amo_op{};

  uint32_t side_mask(int side) const {
    return (side == LOCAL) ? (mask & local) : (mask & ~local & LANE_MASK);
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
  return a.rw == b.rw && a.mask == b.mask && a.local == b.local && a.amo == b.amo
      && a.tag == b.tag && a.addr == b.addr && a.data == b.data
      && a.byteen == b.byteen && a.amo_op == b.amo_op;
}

static bool same_payload(const Rsp& a, const Rsp& b) {
  return a.mask == b.mask && a.tag == b.tag && a.data == b.data;
}

static std::string to_string(const Req& r) {
  std::string s = strf("{%s mask=0x%x local=0x%x amo=0x%x tag=0x%x", r.rw ? "st" : (r.amo ? "amo" : "ld"),
                       r.mask, r.local, r.amo, r.tag);
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
    std::printf("lmem_switch: XLEN=%u lanes=%u addr_bits=%u tag_bits=%u seed=%u requests=%u\n",
                XLEN, NUM_LANES, ADDR_BITS, TAG_BITS, opts_.seed, opts_.requests);
    this->directed();
    if (failures_ == 0) {
      this->random();
    }
    this->drain("final");
    this->final_check();
    std::printf("total cycles: %" PRIu64 ", requests: %" PRIu64 " (mixed %" PRIu64 "), "
                "port requests: global %" PRIu64 " local %" PRIu64 ", responses: %" PRIu64 "\n",
                cycle_, n_in_, n_mixed_, n_out_[GLOBAL], n_out_[LOCAL], n_rsp_);
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

  // kind: 0 = all lanes global, 1 = all lanes local, 2 = both
  Req make_req(int kind) {
    Req r;
    uint32_t op = rand_below(3); // load, store, AMO
    r.rw = (op == 1);
    do {
      r.mask = static_cast<uint32_t>(rng_()) & LANE_MASK;
    } while (r.mask == 0 || (kind == 2 && __builtin_popcount(r.mask) < 2));
    uint32_t junk = static_cast<uint32_t>(rng_()) & ~r.mask & LANE_MASK;
    if (kind == 0) {
      r.local = junk;
    } else if (kind == 1) {
      r.local = r.mask | junk;
    } else {
      uint32_t lsel;
      do {
        lsel = static_cast<uint32_t>(rng_()) & r.mask;
      } while (lsel == 0 || lsel == r.mask);
      r.local = lsel | junk;
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

  int rand_kind() {
    if (NUM_LANES < 2) {
      return rand_below(2);
    }
    uint32_t p = rand_below(100);
    return (p < 20) ? 0 : (p < 40) ? 1 : 2;
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
      uint64_t id = (r.seq << 6) | (static_cast<uint64_t>(side) << 5) | i;
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
    set_bits(t->in_req_amo, 0, NUM_LANES, r.amo);
    set_bits(t->in_req_tag, 0, TAG_BITS, r.tag);
    for (uint32_t i = 0; i < NUM_LANES; ++i) {
      set_bits(t->in_req_addr, i * ADDR_BITS, ADDR_BITS, r.addr[i]);
      set_bits(t->in_req_data, i * XLEN, XLEN, r.data[i]);
      set_bits(t->in_req_byteen, i * WORD_SIZE, WORD_SIZE, r.byteen[i]);
      set_bits(t->in_req_amo_op, i * AMO_OP_BITS, AMO_OP_BITS, r.amo_op[i]);
    }
    t->in_rsp_ready = in_rsp_ready_;
    for (int s = 0; s < 2; ++s) {
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
    for (int s = 0; s < 2; ++s) {
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
      for (int s = 0; s < 2; ++s) {
        if (r.side_mask(s) != 0) {
          Req e = r;
          e.mask = r.side_mask(s);
          exp_req_[s].push_back(e);
        }
      }
      if (r.side_mask(GLOBAL) != 0 && r.side_mask(LOCAL) != 0) {
        ++n_mixed_;
      }
      issued_.push_back(r);
      ++n_in_;
      in_req_.reset();
      progress = true;
    }

    for (int s = 0; s < 2; ++s) {
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
        progress = true;
      }
    }

    if (t->in_rsp_valid && in_rsp_ready_) {
      Rsp got = this->sample_in_rsp();
      ++n_rsp_;
      progress = true;
      bool matched = false;
      for (int s = 0; s < 2 && !matched; ++s) {
        if (!exp_rsp_[s].empty() && same_payload(got, exp_rsp_[s].front())) {
          ++rsp_count_[s].at(exp_rsp_[s].front().seq);
          exp_rsp_[s].pop_front();
          matched = true;
        }
      }
      if (!matched) {
        std::string heads;
        for (int s = 0; s < 2; ++s) {
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
    return in_req_.has_value()
        || !exp_req_[0].empty() || !exp_req_[1].empty()
        || !mem_q_[0].empty() || !mem_q_[1].empty()
        || out_rsp_[0].has_value() || out_rsp_[1].has_value()
        || !exp_rsp_[0].empty() || !exp_rsp_[1].empty();
  }

  bool watchdog() {
    if (cycle_ - last_progress_ <= opts_.watchdog) {
      return true;
    }
    this->fail(strf("no progress for %u cycles: input %s, port requests owed global %zu local %zu, "
                    "responses owed to the LSU global %zu local %zu",
                    opts_.watchdog, in_req_ ? "pending" : "idle",
                    exp_req_[0].size(), exp_req_[1].size(),
                    exp_rsp_[0].size() + mem_q_[0].size() + out_rsp_[0].has_value(),
                    exp_rsp_[1].size() + mem_q_[1].size() + out_rsp_[1].has_value()));
    return false;
  }

  void submit(Req r) {
    r.seq = rsp_count_[0].size();
    in_req_ = r;
    rsp_count_[0].push_back(0);
    rsp_count_[1].push_back(0);
  }

  void drain(const char* ctx) {
    for (int s = 0; s < 2; ++s) {
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
  // then a mixed request waits on it for several cycles while the other port
  // is ready the whole time; the ready port must take the mixed request once.

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
      std::printf("directed: one lane cannot mix global and local, skipped\n");
      return;
    }
    for (int stalled = 0; stalled < 2; ++stalled) {
      for (int op = 0; op < 3; ++op) {
        uint32_t before = failures_;
        uint64_t n_before[2] = {n_out_[0], n_out_[1]};
        int other = 1 - stalled;
        pressure_[stalled].req_ready_force = false;
        pressure_[other].req_ready_force = true;
        this->submit(this->make_req(stalled == GLOBAL ? 0 : 1));
        this->wait_accept();
        Req r;
        do {
          r = this->make_req(2);
        } while ((op == 1) != r.rw || (op == 2) != (r.amo != 0));
        this->submit(r);
        for (int i = 0; i < 8; ++i) {
          this->clock();
        }
        pressure_[stalled].req_ready_force = true;
        this->wait_accept();
        this->drain("directed");
        const char* kind = (op == 0) ? "load" : (op == 1) ? "store" : "AMO";
        uint64_t got[2] = {n_out_[0] - n_before[0], n_out_[1] - n_before[1]};
        uint64_t want[2];
        want[stalled] = 2;
        want[other] = 1;
        if (got[0] != want[0] || got[1] != want[1]) {
          this->fail(strf("directed %s, %s port stalled: global port took %" PRIu64 " request(s), local %" PRIu64
                          ", expected %" PRIu64 " and %" PRIu64,
                          kind, side_name[stalled], got[0], got[1], want[0], want[1]));
        }
        std::printf("directed: mixed %s, %s port stalled 8 cycles: %s\n", kind, side_name[stalled],
                    (failures_ == before) ? "ok" : "FAILED");
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // random: phases of independent readiness on each port and on the LSU
  // response, random request gaps and memory latency.

  void new_phase() {
    static const uint32_t levels[] = {0, 10, 30, 50, 80, 100};
    for (int s = 0; s < 2; ++s) {
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
    for (int s = 0; s < 2; ++s) {
      if (!exp_req_[s].empty()) {
        this->fail(strf("%s port: %zu request(s) never arrived (first seq %" PRIu64 ")",
                        side_name[s], exp_req_[s].size(), exp_req_[s].front().seq));
      }
    }
    uint32_t bad = 0;
    for (const Req& r : issued_) {
      for (int s = 0; s < 2; ++s) {
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
  std::array<bool, 2> out_req_ready_{};
  std::array<std::optional<Rsp>, 2> out_rsp_;
  std::array<Pressure, 2> pressure_;
  uint32_t lsu_rsp_ready_pct_ = 100;
  uint32_t gap_pct_ = 0;
  uint32_t max_lat_ = 4;
  uint64_t phase_end_ = 0;

  std::array<std::deque<Req>, 2> exp_req_;
  std::array<std::deque<std::pair<uint64_t, Rsp>>, 2> mem_q_;
  std::array<std::deque<Rsp>, 2> exp_rsp_;
  std::array<std::vector<uint32_t>, 2> rsp_count_;
  std::vector<Req> issued_;

  uint64_t n_in_ = 0;
  uint64_t n_mixed_ = 0;
  std::array<uint64_t, 2> n_out_{};
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
