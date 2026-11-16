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

// VX_cache with H100_L1 as the SM's L1 (l1h100_cases.h); --csv=<file> writes
// the per-access outcome and timing the SimX twin (tests/unittest/l1_sector)
// must reproduce.

#include "vl_simulator.h"
#include "VVX_cache_l1h100_top.h"
#include "VX_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <type_traits>
#include <vector>

#include "l1h100_cases.h"

using namespace l1h100_cases;

#ifndef VX_CFG_H100_SM_ENABLE
#error "l1h100 tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t ADDR_BITS = VX_CFG_MEM_ADDR_WIDTH - __builtin_ctz(LINE);

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

template <typename T>
static void set_bytes(T& sig, const Bytes& b) {
  for (uint32_t i = 0; i < LINE; ++i)
    set_bits(sig, i * 8, 8, b[i]);
}

template <typename T>
static Bytes get_bytes(const T& sig) {
  Bytes b;
  for (uint32_t i = 0; i < LINE; ++i)
    b[i] = uint8_t(get_bits(sig, i * 8, 8));
  return b;
}

template <typename T>
static void set_flags(T& sig, const Bytes& b) {
  for (uint32_t i = 0; i < LINE; ++i)
    set_bits(sig, i, 1, b[i]);
}

template <typename T>
static Bytes get_flags(const T& sig) {
  Bytes b;
  for (uint32_t i = 0; i < LINE; ++i)
    b[i] = uint8_t(get_bits(sig, i, 1));
  return b;
}

class RtlDriver {
public:
  RtlDriver() {
    auto* t = sim_.operator->();
    t->reset = 1;
    for (int i = 0; i < 8; ++i)
      tick_ = sim_.step(tick_, 2);
    t->reset = 0;
    // The tag arrays clear after reset.
    for (uint32_t i = 0; i < CAP_LINES + 64; ++i)
      tick_ = sim_.step(tick_, 2);
    now_ = 0;
  }

  uint64_t now() const { return now_; }

  void push(const Op& op, uint32_t tag) {
    q_.push_back({op, tag});
    ops_[tag] = op;
  }

  std::vector<Event> cycle() {
    auto* t = sim_.operator->();
    std::vector<Event> ev;
    const bool v = !q_.empty();
    const Op op = v ? q_.front().first : Op{};
    t->core_req_valid = v;
    t->core_req_rw = (op.kind == K_ST);
    t->core_req_flush = (op.kind == K_FENCE);
    set_flags(t->core_req_byteen, op.byteen);
    set_bits(t->core_req_addr, 0, ADDR_BITS, op.line);
    set_bytes(t->core_req_data, op.data);
    t->core_req_tag = v ? q_.front().second : 0;
    t->core_rsp_ready = 1;
    t->mem_req_ready = 1;
    const bool mv = !rsp_.empty() && rsp_.front().due <= now_;
    t->mem_rsp_valid = mv;
    if (mv) {
      set_bytes(t->mem_rsp_data, mem_.at(rsp_.front().line));
      set_bits(t->mem_rsp_tag, 0, 64, rsp_.front().tag);
    }
    t->eval();
    if (v && t->core_req_ready) {
      ev.push_back({EV_ACCEPT, q_.front().second, true, false, false, 0, {}});
      q_.pop_front();
    }
    if (mv && t->mem_rsp_ready)
      rsp_.pop_front();
    if (t->mem_req_valid) {
      Event e{EV_MEM, 0, true, (bool)t->mem_req_rw, (bool)t->mem_req_fence, get_bits(t->mem_req_addr, 0, ADDR_BITS),
              get_flags(t->mem_req_byteen)};
      if (e.fence) {
        rsp_.push_back({now_ + MEM_DELAY, e.line, get_bits(t->mem_req_tag, 0, 64)});
      } else if (e.rw) {
        Bytes d = get_bytes(t->mem_req_data);
        Bytes& m = mem_.at(e.line);
        for (uint32_t i = 0; i < LINE; ++i)
          if (e.byteen[i]) m[i] = d[i];
      } else {
        rsp_.push_back({now_ + MEM_DELAY, e.line, get_bits(t->mem_req_tag, 0, 64)});
      }
      ev.push_back(e);
    }
    if (t->core_rsp_valid) {
      uint32_t tag = (uint32_t)t->core_rsp_tag;
      Event e{EV_RSP, tag, true, false, false, 0, {}};
      auto it = ops_.find(tag);
      if (it != ops_.end() && it->second.kind == K_LD) {
        Bytes d = get_bytes(t->core_rsp_data);
        Bytes& m = mem_.at(it->second.line);
        for (uint32_t i = 0; i < LINE; ++i)
          if (it->second.byteen[i]) e.data_ok &= (d[i] == m[i]);
      }
      ev.push_back(e);
    }
    tick_ = sim_.step(tick_, 2);
    ++now_;
    return ev;
  }

private:
  struct Pending {
    uint64_t due;
    uint64_t line;
    uint64_t tag;
  };

  vl_simulator<VVX_cache_l1h100_top> sim_;
  uint64_t tick_ = 0;
  uint64_t now_ = 0;
  std::deque<std::pair<Op, uint32_t>> q_;
  std::map<uint32_t, Op> ops_;
  std::deque<Pending> rsp_;
  MemImage mem_;
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  uint32_t seed = 1, nrand = 3000;
  bool quick = false;
  const char* csv_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--seed=", 7)) seed = (uint32_t)atoi(argv[i] + 7);
    else if (!strncmp(argv[i], "--random=", 9)) nrand = (uint32_t)atoi(argv[i] + 9);
    else if (!strncmp(argv[i], "--csv=", 6)) csv_path = argv[i] + 6;
    else if (!strcmp(argv[i], "--quick")) quick = true;
    else if (!strncmp(argv[i], "+", 1)) continue;
    else { std::printf("usage: %s [--seed=N] [--random=N] [--quick] [--csv=file]\n", argv[0]); return 2; }
  }
  Verilated::commandArgs(argc, argv);
  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  std::printf("l1h100 (RTL): %u B, %u ways, line %u, sector %u, protected %u, seed %u\n", VX_CFG_DCACHE_SIZE,
              VX_CFG_DCACHE_NUM_WAYS, LINE, SECTOR, VX_CFG_L1_REPL_PROTECTED, seed);
  RtlDriver d;
  Runner<RtlDriver> run(d, csv);
  int rc = run.run(seed, nrand, quick);
  if (csv)
    std::fclose(csv);
  std::printf(rc ? "FAILED!\n" : "PASSED!\n");
  return rc;
}
