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

// VX_alu_warp against the golden: the seeded streams of the SimX twin
// (tests/unittest/warp_ops/stream.h), with the result port always ready (set 0)
// and stalling a seeded quarter of the cycles (set 1): the accept and done cycle
// and every active lane's result of each op equal warp_ops_ref's Pipe; --csv
// writes the rows the twin's must equal line for line.

#include "vl_simulator.h"
#include "VVX_alu_warp_top.h"
#include "VX_config.h"
#include "stream.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#ifndef VX_CFG_H100_FE_ENABLE
#error "the warp-op unit needs the H100 front end"
#endif

namespace wop = vortex::warp_ops_ref;
namespace ws = warp_ops_stream;

static constexpr uint32_t NT = VX_CFG_NUM_THREADS;
static constexpr uint32_t NW = VX_CFG_NUM_WARPS;
static constexpr uint32_t XLEN = VX_CFG_XLEN;
#ifndef WOP_BENCH_PORTS
#define WOP_BENCH_PORTS 1
#endif
static constexpr uint32_t NP = WOP_BENCH_PORTS;
static constexpr uint32_t WOP_BITS = 5;
static constexpr uint32_t NW_W = (NW <= 2) ? 1 : 32 - __builtin_clz(NW - 1);

template <typename W>
static void set_bits(W& w, uint32_t off, uint32_t width, uint64_t v) {
  for (uint32_t b = 0; b < width; ++b) {
    const uint32_t bit = off + b, word = bit / 32, sh = bit % 32;
    w[word] = (w[word] & ~(1u << sh)) | (uint32_t((v >> b) & 1) << sh);
  }
}

template <typename W>
static void put(W& w, uint32_t off, uint32_t width, uint64_t v) {
  if constexpr (std::is_integral_v<W>) {
    const uint64_t m = ((width >= 64) ? ~0ull : ((1ull << width) - 1)) << off;
    w = W((uint64_t(w) & ~m) | ((v << off) & m));
  } else {
    set_bits(w, off, width, v);
  }
}

template <typename W>
static uint64_t get_bits(const W& w, uint32_t off, uint32_t width) {
  uint64_t v = 0;
  for (uint32_t b = 0; b < width; ++b) {
    const uint32_t bit = off + b;
    v |= uint64_t((w[bit / 32] >> (bit % 32)) & 1) << b;
  }
  return v;
}

struct Row {
  uint64_t accept = 0, done = 0, result = 0;
  bool seen = false;
};

int main(int argc, char** argv) {
  const char* csv_path = nullptr;
  uint32_t seed = 1, n_ops = 4000;
  for (int i = 1; i < argc; ++i) {
    if (!std::strncmp(argv[i], "--csv=", 6))
      csv_path = argv[i] + 6;
    else if (!std::strncmp(argv[i], "--seed=", 7))
      seed = (uint32_t)std::stoul(argv[i] + 7);
    else if (!std::strncmp(argv[i], "--ops=", 6))
      n_ops = (uint32_t)std::stoul(argv[i] + 6);
  }
  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  if (csv)
    ws::csv_header(csv);

  vl_simulator<VVX_alu_warp_top> sim;
  auto* t = sim.operator->();
  uint64_t tick = 0;
  int errors = 0;
  auto fail = [&](uint32_t set, const char* what, size_t i, uint64_t got, uint64_t want) {
    if (++errors <= 20)
      std::printf("FAIL set %u op %zu %s: got %llu want %llu\n", set, i, what, (unsigned long long)got,
                  (unsigned long long)want);
  };

  for (uint32_t set = 0; set < 2; ++set) {
    const uint32_t sseed = seed + 1000 * set;
    std::vector<std::vector<ws::Op>> ops(NP);
    std::vector<std::vector<wop::Req>> reqs(NP);
    size_t total = 0;
    for (uint32_t p = 0; p < NP; ++p) {
      ops[p] = ws::make_stream(ws::port_seed(sseed, p), n_ops, NT, NW, XLEN);
      for (auto& op : ops[p]) {
        const uint32_t k = wop::is_match_any(op.code)
            ? wop::distinct_values(op.code, XLEN, NT, op.tmask, op.rs1.data(), op.rs2.data()) : 1;
        reqs[p].push_back({op.arrive, op.code, k});
      }
      total += ops[p].size();
    }
    auto ready = [&](uint32_t p, uint64_t c) { return ws::out_ready(set, ws::port_seed(sseed, p), c); };
    const auto ev = wop::schedule_ports(NT, reqs, ready);

    t->in_valid = 0;
    t->out_ready = 0;
    t->reset = 1;
    for (int i = 0; i < 4; ++i)
      tick = sim.step(tick, 2);
    t->reset = 0;

    // UUID_WIDTH is 1 in NDEBUG builds, so a result is matched by the cycle the golden retires it
    // (one result per cycle leaves the unit)
    std::map<uint64_t, std::pair<uint32_t, size_t>> by_done;
    uint64_t limit = 1000;
    for (uint32_t p = 0; p < NP; ++p) {
      for (size_t i = 0; i < ev[p].size(); ++i) {
        by_done[ev[p][i].done] = {p, i};
        limit = std::max<uint64_t>(limit, ev[p][i].done + 1000);
      }
    }
    std::vector<std::vector<Row>> rows(NP);
    std::vector<size_t> head(NP, 0);
    for (uint32_t p = 0; p < NP; ++p)
      rows[p].resize(ops[p].size());
    size_t n_done = 0;
    for (uint64_t cyc = 0; n_done < total && cyc < limit; ++cyc) {
      uint32_t valid = 0, rdy = 0;
      for (uint32_t p = 0; p < NP; ++p) {
        const bool offer = head[p] < ops[p].size() && ops[p][head[p]].arrive <= cyc;
        valid |= uint32_t(offer) << p;
        rdy |= uint32_t(ready(p, cyc)) << p;
        if (offer) {
          const auto& op = ops[p][head[p]];
          put(t->in_code, p * WOP_BITS, WOP_BITS, op.code);
          put(t->in_wid, p * NW_W, NW_W, op.wid);
          put(t->in_tmask, p * NT, NT, op.tmask);
          for (uint32_t l = 0; l < NT; ++l) {
            put(t->in_rs1, (p * NT + l) * XLEN, XLEN, op.rs1[l]);
            put(t->in_rs2, (p * NT + l) * XLEN, XLEN, op.rs2[l]);
          }
        }
      }
      t->in_valid = valid;
      t->out_ready = rdy;
      t->eval();
      const uint32_t acc = valid & uint32_t(t->in_ready);
      for (uint32_t p = 0; p < NP; ++p) {
        if ((acc >> p) & 1)
          rows[p][head[p]++].accept = cyc;
      }
      const uint32_t fire = uint32_t(t->out_valid) & rdy;
      if (fire) {
        const auto it = by_done.find(cyc);
        const bool known = it != by_done.end();
        const uint32_t p = known ? it->second.first : 0;
        const size_t i = known ? it->second.second : 0;
        if (!known || rows[p][i].seen || fire != (1u << p)) {
          fail(set, "result at a cycle or port the golden retires nothing", 0, cyc, fire);
        } else {
          const auto& op = ops[p][i];
          std::vector<uint64_t> want(NT, 0), got(NT, 0);
          wop::execute(op.code, XLEN, NT, op.tmask, op.rs1.data(), op.rs2.data(), want.data());
          for (uint32_t l = 0; l < NT; ++l)
            got[l] = get_bits(t->out_data, l * XLEN, XLEN);
          if ((uint64_t)t->out_tmask != op.tmask)
            fail(set, "tmask", i, t->out_tmask, op.tmask);
          if ((uint32_t)t->out_wid != op.wid)
            fail(set, "wid", i, t->out_wid, op.wid);
          for (uint32_t l = 0; l < NT; ++l) {
            if (((op.tmask >> l) & 1) && got[l] != want[l])
              fail(set, "lane result", i, got[l], want[l]);
          }
          rows[p][i] = Row{rows[p][i].accept, cyc, ws::fold(got, op.tmask), true};
          ++n_done;
        }
      }
      tick = sim.step(tick, 2);
    }
    if (n_done != total) {
      std::printf("FAIL set %u: %zu of %zu ops done by cycle %llu\n", set, n_done, total, (unsigned long long)limit);
      ++errors;
    }
    for (uint32_t p = 0; p < NP; ++p) {
      for (size_t i = 0; i < ops[p].size(); ++i) {
        if (!rows[p][i].seen)
          continue;
        if (rows[p][i].accept != ev[p][i].accept)
          fail(set, "accept", i, rows[p][i].accept, ev[p][i].accept);
        if (csv)
          ws::csv_row(csv, set + 2 * p, (uint32_t)i, ops[p][i], reqs[p][i].k, rows[p][i].accept, rows[p][i].done,
                      rows[p][i].result);
      }
    }
  }
  if (csv)
    std::fclose(csv);
  if (errors) {
    std::printf("hw-warp-ops: %d errors\nFAILED!\n", errors);
    return 1;
  }
  std::printf("hw-warp-ops: %u ports x %u ops x 2 ready patterns, results, accept and done cycles equal the golden (tree depth %u)\nPASSED!\n",
              NP, n_ops, wop::tree_depth(NT));
  return 0;
}
