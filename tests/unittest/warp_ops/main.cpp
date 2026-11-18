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

// SimX's warp-op unit (sim/simx/warp_ops.h lane results, warp_ops_ref.h Pipe
// timing as AluUnit steps it) against the golden: lane results and the
// accept/done cycle of every op of seeded streams with the result port always
// ready (set 0) and stalling a seeded quarter of the cycles (set 1); directed
// structural properties of the tree and the comparator row. --csv writes the
// rows the RTL bench must reproduce.

#include "stream.h"
#include "../../../sim/simx/warp_ops.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace vortex;
namespace ws = warp_ops_stream;

namespace {

struct Reg { uint64_t u64; };

struct Mask {
  uint64_t bits;
  bool test(uint32_t i) const { return (bits >> i) & 1; }
};

int errors = 0;

void expect(bool ok, const char* what, uint32_t set, uint32_t idx, uint64_t got, uint64_t want) {
  if (ok)
    return;
  if (++errors <= 20)
    std::printf("FAIL set %u op %u %s: got %llu want %llu\n", set, idx, what,
                (unsigned long long)got, (unsigned long long)want);
}

#ifndef WOP_BENCH_PORTS
#define WOP_BENCH_PORTS 1
#endif

// Timing of per-port streams through one unit, stepped as AluUnit::warp_op_tick does: the
// result goes to its op's port, the input picks round robin after the last port accepted.
std::vector<std::vector<wop::Event>> run_ports(uint32_t nt, const std::vector<std::vector<wop::Req>>& reqs,
                                               uint32_t set, uint32_t seed) {
  const uint32_t np = (uint32_t)reqs.size();
  wop::Pipe<std::pair<uint32_t, uint32_t>> pipe(nt);
  std::vector<std::vector<wop::Event>> ev(np);
  std::vector<size_t> head(np, 0);
  size_t total = 0, done = 0;
  for (uint32_t p = 0; p < np; ++p) {
    ev[p].assign(reqs[p].size(), wop::Event{0, 0});
    total += reqs[p].size();
  }
  uint32_t rr = np - 1;
  for (uint64_t cyc = 0; done < total; ++cyc) {
    const auto out = pipe.offered();
    const bool out_ready = out.valid && ws::out_ready(set, ws::port_seed(seed, out.tag.first), cyc);
    if (out_ready) {
      ev[out.tag.first][out.tag.second].done = cyc;
      ++done;
    }
    std::pair<uint32_t, uint32_t> in;
    uint32_t sel = 0;
    bool offer = false;
    for (uint32_t i = 1; i <= np && !offer; ++i) {
      sel = (rr + i) % np;
      if (head[sel] < reqs[sel].size() && reqs[sel][head[sel]].arrive <= cyc) {
        in = {sel, (uint32_t)head[sel]};
        offer = true;
      }
    }
    const wop::Req* r = offer ? &reqs[sel][head[sel]] : nullptr;
    if (pipe.step(out_ready, offer ? &in : nullptr, r ? r->code : 0, r ? r->k : 1)) {
      ev[sel][head[sel]++].accept = cyc;
      rr = sel;
    }
  }
  return ev;
}

std::vector<wop::Event> run_pipe(uint32_t nt, const std::vector<wop::Req>& reqs, uint32_t set, uint32_t seed) {
  return run_ports(nt, {reqs}, set, seed)[0];
}

// Back-to-back streams on an idle unit: tree ops one per cycle and 1 + depth cycles each,
// MATCH.ANY k cycles each and one every k, a tree result ahead of a MATCH.ANY one.
void directed() {
  expect(wop::code_of(0x19, 7) == wop::REDUX_XOR && wop::code_of(0x1A, 1) == wop::ELECT_L &&
         wop::code_of(0x1B, 3) == wop::MATCH_ALL_B64, "code_of", 9, 0, 0, 0);
  for (uint32_t nt : {2u, 4u, 8u, 16u, 32u, 64u}) {
    const uint32_t d = wop::tree_depth(nt);
    expect((1u << d) == nt, "tree depth", 9, nt, d, 0);
    for (uint32_t code : {(uint32_t)wop::REDUX_ADD, (uint32_t)wop::ELECT_P, (uint32_t)wop::ELECT_L,
                          (uint32_t)wop::MATCH_ALL_B32, (uint32_t)wop::MATCH_ALL_B64}) {
      std::vector<wop::Req> reqs(8, wop::Req{0, code, 1});
      const auto ev = run_pipe(nt, reqs, 0, 0);
      for (uint32_t i = 0; i < 8; ++i) {
        expect(ev[i].accept == i, "tree op one per cycle", 9, i, ev[i].accept, i);
        expect(ev[i].done - ev[i].accept == 1 + d, "tree op latency 1 + depth", 9, i,
               ev[i].done - ev[i].accept, 1 + d);
      }
    }
    for (uint32_t code : {(uint32_t)wop::MATCH_ANY_B32, (uint32_t)wop::MATCH_ANY_B64}) {
      for (uint32_t k = 1; k <= nt; k *= 2) {
        std::vector<wop::Req> reqs(4, wop::Req{0, code, k});
        const auto ev = run_pipe(nt, reqs, 0, 0);
        for (uint32_t i = 0; i < 4; ++i) {
          expect(ev[i].accept == i * k, "MATCH.ANY one every k", 9, k, ev[i].accept, i * k);
          expect(ev[i].done - ev[i].accept == k, "MATCH.ANY latency k", 9, k, ev[i].done - ev[i].accept, k);
          expect(wop::latency(nt, code, k) == k, "latency(MATCH.ANY)", 9, k, wop::latency(nt, code, k), k);
        }
      }
    }
    // A MATCH.ANY with k = depth accepted right after a REDUX finishes when the REDUX does: the
    // tree's result goes first, MATCH.ANY's one cycle later.
    if (d >= 1) {
      const auto ev = run_pipe(nt, {{0, wop::REDUX_ADD, 1}, {0, wop::MATCH_ANY_B32, d}}, 0, 0);
      expect(ev[0].done == 1 + d, "tree first", 9, nt, ev[0].done, 1 + d);
      expect(ev[1].done == 2 + d, "MATCH.ANY after the tree", 9, nt, ev[1].done, 2 + d);
    }
    // Ports offering together alternate, port 0 first after reset; each result goes to its port.
    const std::vector<wop::Req> two(2, wop::Req{0, wop::REDUX_ADD, 1});
    const auto rr = run_ports(nt, {two, two, two}, 0, 0);
    for (uint32_t p = 0; p < 3; ++p) {
      for (uint32_t i = 0; i < 2; ++i)
        expect(rr[p][i].accept == 3 * i + p, "round robin", 9, p, rr[p][i].accept, 3 * i + p);
    }
    expect(wop::schedule_ports(nt, {two, two, two}, [](uint32_t, uint64_t) { return true; })[2][1].accept == 5,
           "golden round robin", 9, nt, 0, 5);
  }
}

void run_set(uint32_t set, uint32_t seed, uint32_t n_ops, FILE* csv) {
  const uint32_t nt = VX_CFG_NUM_THREADS, nw = VX_CFG_NUM_WARPS, xlen = VX_CFG_XLEN;
  const uint32_t np = WOP_BENCH_PORTS;
  std::vector<std::vector<ws::Op>> ops(np);
  std::vector<std::vector<wop::Req>> reqs(np), twin_reqs(np);
  std::vector<std::vector<std::vector<uint64_t>>> want(np);
  for (uint32_t p = 0; p < np; ++p) {
    ops[p] = ws::make_stream(ws::port_seed(seed, p), n_ops, nt, nw, xlen);
    want[p].resize(ops[p].size());
    for (size_t i = 0; i < ops[p].size(); ++i) {
      const auto& op = ops[p][i];
      want[p][i] = op.rs1;
      wop::execute(op.code, xlen, nt, op.tmask, op.rs1.data(), op.rs2.data(), want[p][i].data());
      const uint32_t k = wop::is_match_any(op.code)
          ? wop::distinct_values(op.code, xlen, nt, op.tmask, op.rs1.data(), op.rs2.data()) : 1;
      reqs[p].push_back({op.arrive, op.code, k});
      const Mask m{op.tmask};
      std::vector<Reg> rs1(nt), rs2(nt);
      for (uint32_t t = 0; t < nt; ++t) {
        rs1[t].u64 = op.rs1[t];
        rs2[t].u64 = op.rs2[t];
      }
      twin_reqs[p].push_back(reqs[p].back());
      twin_reqs[p].back().k = wop::is_match_any(op.code) ? warp_op_distinct(op.code, xlen, nt, m, rs1, rs2) : 1;
    }
  }
  const auto gold = wop::schedule_ports(
      nt, reqs, [&](uint32_t p, uint64_t c) { return ws::out_ready(set, ws::port_seed(seed, p), c); });
  const auto ev = run_ports(nt, twin_reqs, set, seed);

  for (uint32_t p = 0; p < np; ++p) {
    for (size_t i = 0; i < ops[p].size(); ++i) {
      const auto& op = ops[p][i];
      const Mask m{op.tmask};
      std::vector<Reg> rs1(nt), rs2(nt), rd(nt);
      for (uint32_t t = 0; t < nt; ++t) {
        rs1[t].u64 = op.rs1[t];
        rs2[t].u64 = op.rs2[t];
        rd[t].u64 = op.rs1[t];
      }
      warp_op_execute(op.code, xlen, nt, m, rs1, rs2, rd);
      std::vector<uint64_t> got(nt);
      for (uint32_t t = 0; t < nt; ++t)
        got[t] = rd[t].u64;
      const auto& e = ev[p][i];
      const auto& g = gold[p][i];
      expect(twin_reqs[p][i].k == reqs[p][i].k, "k", set, i, twin_reqs[p][i].k, reqs[p][i].k);
      expect(e.accept == g.accept, "accept", set, i, e.accept, g.accept);
      expect(e.done == g.done, "done", set, i, e.done, g.done);
      expect(e.done >= e.accept + wop::latency(nt, op.code, reqs[p][i].k), "done before latency", set, i,
             e.done - e.accept, wop::latency(nt, op.code, reqs[p][i].k));
      for (uint32_t t = 0; t < nt; ++t)
        expect(!((op.tmask >> t) & 1) || got[t] == want[p][i][t], "lane result", set, i, got[t], want[p][i][t]);
      if (csv)
        ws::csv_row(csv, set + 2 * p, (uint32_t)i, op, reqs[p][i].k, e.accept, e.done, ws::fold(got, op.tmask));
    }
  }
}

} // namespace

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

  directed();
  for (uint32_t s = 0; s < 2; ++s)
    run_set(s, seed + 1000 * s, n_ops, csv);
  if (csv)
    std::fclose(csv);

  if (errors) {
    std::printf("warp_ops: %d errors\nFAILED!\n", errors);
    return 1;
  }
  std::printf("warp_ops: %u ops x 2 port patterns, results and timing equal the golden, tree and comparator-row properties checked\nPASSED!\n", n_ops);
  return 0;
}
