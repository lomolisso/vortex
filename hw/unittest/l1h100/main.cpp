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

// VX_sm_sram_top testbench for the SM L1 on the array (l1h100_cases.h): one LSU
// block's global leg, a flat memory on the miss bus, the host flush; --csv=<file>
// writes the outcome lines the SimX twin (tests/unittest/l1_sector) must reproduce.

#include "vl_simulator.h"
#include "VVX_sm_sram_top.h"
#include "VX_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <type_traits>
#include <vector>

#include "l1h100_cases.h"

using namespace l1h100_cases;

#ifndef VX_CFG_H100_SM_ENABLE
#error "l1h100 tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t clog2(uint32_t n) { return n <= 1 ? 0 : 1 + clog2((n + 1) / 2); }
constexpr uint32_t log2up(uint32_t n) { return n > 1 ? clog2(n) : 1; }

constexpr uint32_t NL            = VX_CFG_NUM_LSU_LANES;
constexpr uint32_t GL_ADDR_BITS  = VX_CFG_MEM_ADDR_WIDTH - clog2(WORD);
constexpr uint32_t MEM_ADDR_BITS = VX_CFG_MEM_ADDR_WIDTH - clog2(LINE);
// The L1's miss-bus tag: {uuid, special (atomic or fence), index} (VX_gpu_pkg SM_L1_MEM_TAG_WIDTH).
constexpr uint32_t IDX_BITS      = std::max(clog2(VX_CFG_SM_MAX_READ_LINES),
    clog2(2 * VX_CFG_LSU_PENDING_SIZE) + clog2(VX_CFG_NUM_LSU_BLOCKS) + clog2(LINE / WORD) + log2up(NL));
#ifdef VX_CFG_EXT_FPAMO_ENABLE
constexpr uint32_t OP_BITS       = 5;
#else
constexpr uint32_t OP_BITS       = 4;
#endif
constexpr uint32_t OP_ADD        = 2;
constexpr uint32_t ATTR_IO       = 1;
constexpr uint32_t ATTR_AMO      = 4;
constexpr uint32_t RD_TCU_PORT   = 0;

static_assert(NL <= MAX_LANES, "the bench drives at most 32 lanes");
static_assert(VX_CFG_LSU_PENDING_SIZE >= NUM_TAGS, "the bench keeps up to NUM_TAGS requests in flight");
static_assert(WORD == 8, "the H100 SM path is RV64: one lane word per uint64");

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

class RtlDriver {
public:
  RtlDriver() {
    auto* t = sim_.operator->();
    const L1SlotCfg c = knob_l1_cfg();
    s_rows_ = c.s_bytes / c.row_bytes;
    t->cfg_s_rows = s_rows_;
    t->cfg_l1_lines = l1_lines(c);
    t->cfg_l1_protect = l1_protect(c);
    t->reset = 1;
    for (int i = 0; i < 8; ++i)
      tick_ = sim_.step(tick_, 2);
    t->reset = 0;
  }

  void push(const Req& r, uint32_t tag) { q_.push_back({r, tag}); }
  void flush() { flush_on_ = true; }
  void set_mem_delay(uint32_t d) { mem_.delay = d; }
  void set_row_stream(bool on) {
    rows_on_ = on;
    row_phase_ = 0;
    row_pend_ = false;
  }
  FlatMem& mem() { return mem_; }
  bool busy() const { return busy_ || !q_.empty() || !mrsp_.empty() || flush_on_; }

  std::vector<Event> cycle() {
    auto* t = sim_.operator->();
    std::vector<Event> ev;
    const bool v = !q_.empty();
    const Req r = v ? q_.front().first : Req{};
    const uint32_t tag = v ? q_.front().second : 0;
    set_bits(t->gl_in_valid, 0, 1, v);
    set_bits(t->gl_in_rw, 0, 1, r.kind == K_ST);
    set_bits(t->gl_in_mask, 0, NL, r.mask);
    bool io = false;
    for (uint32_t i = 0; i < NL; ++i) {
      const bool on = (r.mask >> i) & 1;
      io |= on && nc_line(r.addr[i] / LINE);
      set_bits(t->gl_in_addr, i * GL_ADDR_BITS, GL_ADDR_BITS, r.addr[i] / WORD);
      set_bits(t->gl_in_byteen, i * WORD, WORD, on ? lane_be(r, i) : 0);
      set_bits(t->gl_in_data, i * 64, 64, on ? lane_wdata(r, i) : 0);
    }
    set_bits(t->gl_in_attr, 0, ATTR_AMO + 1 + OP_BITS, 0);
    set_bits(t->gl_in_attr, ATTR_IO, 1, io);
    set_bits(t->gl_in_attr, ATTR_AMO, 1, r.kind == K_ADD);
    set_bits(t->gl_in_attr, ATTR_AMO + 1, OP_BITS, r.kind == K_ADD ? OP_ADD : 0);
    set_bits(t->gl_in_tag, 0, 16, tag);
    set_bits(t->gl_in_rsp_ready, 0, 1, 1);
    t->mem_req_ready = 1;
    const bool mv = !mrsp_.empty() && mrsp_.front().when <= now_;
    t->mem_rsp_valid = mv;
    if (mv) {
      const auto& m = mrsp_.front();
      for (uint32_t i = 0; i < LINE; ++i)
        set_bits(t->mem_rsp_data, i * 8, 8, m.data[i]);
      set_bits(t->mem_rsp_tag, 0, 64, m.tag);
    }
    t->flush_req = flush_on_;
    // every other cycle, so the global path keeps read-port cycles under the
    // TCU's higher priority; a presented row is held until taken
    const bool rv = rows_on_ && (row_pend_ || (row_phase_ & 1) == 0);
    set_bits(t->rd_valid, RD_TCU_PORT, 1, rv);
    set_bits(t->rd_row, RD_TCU_PORT * 32, 32, row_ % s_rows_);
    t->eval();

    if (v && get_bits(t->gl_in_ready, 0, 1)) {
      ev.push_back(Event{EV_ACCEPT, tag});
      q_.pop_front();
    }
    if (get_bits(t->gl_in_rsp_valid, 0, 1)) {
      Event e{EV_RSP};
      e.tag = uint32_t(get_bits(t->gl_in_rsp_tag, 0, 16)) & (NUM_TAGS - 1);
      e.mask = uint32_t(get_bits(t->gl_in_rsp_mask, 0, NL));
      for (uint32_t i = 0; i < NL; ++i)
        e.data[i] = get_bits(t->gl_in_rsp_data, i * 64, 64);
      ev.push_back(e);
    }
    if (t->mem_req_valid) {
      const uint64_t line = get_bits(t->mem_req_addr, 0, MEM_ADDR_BITS);
      const uint64_t mtag = get_bits(t->mem_req_tag, 0, 64);
      uint8_t data[LINE], be[LINE];
      for (uint32_t i = 0; i < LINE; ++i) {
        data[i] = uint8_t(get_bits(t->mem_req_data, i * 8, 8));
        be[i] = uint8_t(get_bits(t->mem_req_byteen, i, 1));
      }
      if (t->mem_req_rw) {
        mem_.write(line * LINE, data, be, LINE);
      } else if ((mtag >> IDX_BITS) & 1) {
        std::vector<uint8_t> old(LINE, 0);
        if (get_bits(t->mem_req_attr, ATTR_AMO, 1))
          old = mem_.amo_add(line * LINE, data, be, LINE);
        this->reply(mtag, old);
      } else {
        Event e{EV_MEMRD};
        e.line = line;
        for (uint32_t i = 0; i < LINE; ++i)
          e.mask |= uint32_t(be[i]) << (i / SECTOR);
        ev.push_back(e);
        this->reply(mtag, mem_.read(line * LINE, LINE));
      }
    }
    if (mv && t->mem_rsp_ready)
      mrsp_.pop_front();
    if (flush_on_ && t->flush_done) {
      ev.push_back(Event{EV_FLUSHED});
      flush_on_ = false;
    }
    if (rv && get_bits(t->rd_ready, RD_TCU_PORT, 1)) {
      ++row_;
      row_pend_ = false;
    } else {
      row_pend_ = rv;
    }
    if (rows_on_)
      ++row_phase_;
    busy_ = t->l1_busy;
    tick_ = sim_.step(tick_, 2);
    ++now_;
    return ev;
  }

private:
  struct MemRsp {
    uint64_t when, tag;
    std::vector<uint8_t> data;
  };

  void reply(uint64_t tag, std::vector<uint8_t> data) {
    const uint64_t when = std::max(now_ + mem_.delay, mrsp_.empty() ? 0 : mrsp_.back().when);
    mrsp_.push_back({when, tag, std::move(data)});
  }

  vl_simulator<VVX_sm_sram_top> sim_;
  uint64_t tick_ = 0, now_ = 0;
  uint32_t s_rows_ = 0, row_ = 0, row_phase_ = 0;
  bool flush_on_ = false, rows_on_ = false, row_pend_ = false, busy_ = false;
  std::deque<std::pair<Req, uint32_t>> q_;
  std::deque<MemRsp> mrsp_;
  FlatMem mem_;
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  uint32_t seed = 1, nrand = 2000;
  const char* csv_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--seed=", 7)) seed = (uint32_t)atoi(argv[i] + 7);
    else if (!strncmp(argv[i], "--random=", 9)) nrand = (uint32_t)atoi(argv[i] + 9);
    else if (!strncmp(argv[i], "--csv=", 6)) csv_path = argv[i] + 6;
    else { std::printf("usage: %s [--seed=N] [--random=N] [--csv=file]\n", argv[0]); return 2; }
  }
  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  std::printf("l1h100 (RTL): %u lanes, line %u B, sector %u B, seed %u\n", NL, LINE, SECTOR, seed);
  RtlDriver d;
  Runner<RtlDriver> run(d, NL, csv);
  int rc = run.run(seed, nrand);
  if (csv)
    std::fclose(csv);
  std::printf(rc ? "FAILED!\n" : "PASSED!\n");
  return rc;
}
