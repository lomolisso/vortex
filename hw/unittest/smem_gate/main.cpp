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

// VX_smem_gate testbench: the H100 SM path's shared-memory port in front of an
// ideal bank sink (smem_gate_cases.h). --csv=<file> writes the per-request
// timing the SimX twin (tests/unittest/smem_bank) must reproduce.

#include "vl_simulator.h"
#include "VVX_smem_gate_top.h"
#include "VX_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "smem_gate_cases.h"

using namespace smem_gate_cases;

#ifndef VX_CFG_H100_SM_ENABLE
#error "smem_gate tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t NB         = VX_CFG_NUM_LSU_BLOCKS;
constexpr uint32_t NL         = VX_CFG_NUM_LSU_LANES;
constexpr uint32_t WORD_SIZE  = VX_CFG_XLEN / 8;
constexpr uint32_t ADDR_BITS  = 16;
constexpr uint32_t TAG_BITS   = 6;
constexpr uint32_t DATA_BITS  = WORD_SIZE * 8;
constexpr uint32_t clog2(uint32_t n) { return n <= 1 ? 0 : 1 + clog2((n + 1) / 2); }
constexpr uint32_t FOLD_BITS  = clog2(NL + 1);
#ifdef VX_CFG_EXT_FPAMO_ENABLE
constexpr uint32_t OP_BITS    = 5;
#else
constexpr uint32_t OP_BITS    = 4;
#endif
constexpr uint32_t OP_ADD     = 2;
constexpr uint32_t OP_CAS     = 9;

static_assert(NL <= MAX_LANES, "the bench drives at most 32 lanes");
static_assert(DATA_BITS == 64, "the H100 SM path is RV64: one lane word per uint64");

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
    t->reset = 1;
    for (int i = 0; i < 8; ++i)
      tick_ = sim_.step(tick_, 2);
    t->reset = 0;
    now_ = 0;
  }

  uint64_t now() const { return now_; }

  void push(uint32_t b, const Req& r, uint32_t tag) {
    q_[b].push_back({r, tag});
    reqs_[b][tag] = r;
  }

  void set_ext(bool v) { ext_ = v; }

  std::vector<Event> cycle() {
    auto* t = sim_.operator->();
    std::vector<Event> ev;
    t->ext_valid = ext_;
    for (uint32_t b = 0; b < NB; ++b) {
      bool v = !q_[b].empty();
      const Req r = v ? q_[b].front().first : Req{};
      const uint32_t tag = v ? q_[b].front().second : 0;
      set_bits(t->in_valid, b, 1, v);
      set_bits(t->in_rw, b, 1, r.kind == K_ST);
      set_bits(t->in_mask, b * NL, NL, r.mask);
      for (uint32_t i = 0; i < NL; ++i) {
        uint64_t off = r.addr[i] % WORD_SIZE;
        uint64_t be = (r.mask >> i & 1) ? (((1ull << r.bytes) - 1) << off) : 0;
        set_bits(t->in_addr, (b * NL + i) * ADDR_BITS, ADDR_BITS, r.addr[i] / WORD_SIZE);
        set_bits(t->in_byteen, (b * NL + i) * WORD_SIZE, WORD_SIZE, be);
      }
      set_bits(t->in_amo, b, 1, r.kind == K_ADD || r.kind == K_CAS);
      set_bits(t->in_amo_op, b * OP_BITS, OP_BITS, r.kind == K_CAS ? OP_CAS : OP_ADD);
      set_bits(t->in_lacc, b, 1, r.lacc);
      set_bits(t->in_tag, b * TAG_BITS, TAG_BITS, tag);
      set_bits(t->in_rsp_ready, b, 1, 1);
      set_bits(t->out_ready, b, 1, 1);
      // The sink's oldest answer once it is due.
      bool sv = !sink_[b].empty() && sink_[b].front().due <= now_;
      const Sink s = sv ? sink_[b].front() : Sink{};
      set_bits(t->out_rsp_valid, b, 1, sv);
      set_bits(t->out_rsp_mask, b * NL, NL, s.mask);
      for (uint32_t i = 0; i < NL; ++i)
        set_bits(t->out_rsp_data, (b * NL + i) * DATA_BITS, DATA_BITS, s.data[i]);
      set_bits(t->out_rsp_tag, b * TAG_BITS, TAG_BITS, s.tag);
    }
    t->eval();
    for (uint32_t b = 0; b < NB; ++b) {
      if (!q_[b].empty() && get_bits(t->in_ready, b, 1)) {
        Event e{EV_ACCEPT, b, q_[b].front().second, (uint32_t)get_bits(t->out_mask, b * NL, NL), true};
        const Req& r = q_[b].front().first;
        if (r.kind != K_ST) {
          Sink s;
          s.due = now_ + SINK_DELAY;
          s.tag = (uint32_t)get_bits(t->out_tag, b * TAG_BITS, TAG_BITS);
          s.mask = e.out_mask;
          for (uint32_t i = 0; i < NL; ++i)
            s.data[i] = word_value(get_bits(t->out_addr, (b * NL + i) * ADDR_BITS, ADDR_BITS));
          sink_[b].push_back(s);
        }
        ev.push_back(e);
        q_[b].pop_front();
      }
#ifdef VX_CFG_EXT_DLMEM_ENABLE
      if (get_bits(t->fold_valid, b, 1))
        ev.push_back({EV_FOLD, b, 0, 0, true, (uint32_t)get_bits(t->fold_cnt, b * FOLD_BITS, FOLD_BITS)});
#endif
      if (!sink_[b].empty() && sink_[b].front().due <= now_ && get_bits(t->out_rsp_ready, b, 1))
        sink_[b].pop_front();
      if (get_bits(t->in_rsp_valid, b, 1)) {
        uint32_t tag = (uint32_t)get_bits(t->in_rsp_tag, b * TAG_BITS, TAG_BITS);
        uint32_t mask = (uint32_t)get_bits(t->in_rsp_mask, b * NL, NL);
        Event e{EV_RSP, b, tag, 0, true};
        auto it = reqs_[b].find(tag);
        if (it == reqs_[b].end() || mask != it->second.mask) {
          e.data_ok = false;
        } else {
          for (uint32_t i = 0; i < NL; ++i) {
            if (mask >> i & 1)
              e.data_ok &= get_bits(t->in_rsp_data, (b * NL + i) * DATA_BITS, DATA_BITS)
                           == word_value(it->second.addr[i] / WORD_SIZE);
          }
        }
        ev.push_back(e);
      }
    }
    tick_ = sim_.step(tick_, 2);
    ++now_;
    return ev;
  }

private:
  struct Sink {
    uint64_t due = 0;
    uint32_t tag = 0;
    uint32_t mask = 0;
    uint64_t data[MAX_LANES] = {};
  };

  vl_simulator<VVX_smem_gate_top> sim_;
  uint64_t tick_ = 0;
  uint64_t now_ = 0;
  bool ext_ = false;
  std::deque<std::pair<Req, uint32_t>> q_[NB];
  std::map<uint32_t, Req> reqs_[NB];
  std::deque<Sink> sink_[NB];
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  uint32_t seed = 1, nrand = 2000;
  const char* csv_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--seed=", 7)) seed = (uint32_t)atoi(argv[i] + 7);
    else if (!strncmp(argv[i], "--random=", 9)) nrand = (uint32_t)atoi(argv[i] + 9);
    else if (!strncmp(argv[i], "--csv=", 6)) csv_path = argv[i] + 6;
    else if (!strncmp(argv[i], "+", 1)) continue;
    else { std::printf("usage: %s [--seed=N] [--random=N] [--csv=file]\n", argv[0]); return 2; }
  }
  Verilated::commandArgs(argc, argv);
  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  std::printf("smem_gate (RTL): %u blocks x %u lanes, banks %u x %u B, seed %u\n", NB, NL, VX_CFG_LMEM_NUM_BANKS,
              VX_CFG_LMEM_BANK_BYTES, seed);
  RtlDriver d;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  const bool lacc_ok = true;
#else
  const bool lacc_ok = false;
#endif
  Runner<RtlDriver, WORD_SIZE> run(d, NL, NB, csv, lacc_ok);
  int rc = run.run(seed, nrand);
  if (csv)
    std::fclose(csv);
  std::printf(rc ? "FAILED!\n" : "PASSED!\n");
  return rc;
}
