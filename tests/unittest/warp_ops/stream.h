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

#pragma once

// Seeded warp-op streams and the CSV row format, shared by the SimX twin
// (tests/unittest/warp_ops) and the RTL bench (hw/unittest/warp_ops), so both
// replay the same ops and their CSVs compare line for line.

#include <warp_ops_ref.h>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace warp_ops_stream {

namespace wop = vortex::warp_ops_ref;

struct Op {
  uint64_t arrive;
  uint32_t code;
  uint32_t wid;
  uint64_t tmask;
  std::vector<uint64_t> rs1, rs2;
};

// The result port's ready in cycle `cyc`: set 0 always, set 1 a seeded quarter of the
// cycles not ready (the tree stalls and MATCH.ANY waits behind the tree's result).
inline bool out_ready(uint32_t set, uint32_t seed, uint64_t cyc) {
  if (set == 0)
    return true;
  uint64_t h = (cyc + 1) * 0x9e3779b97f4a7c15ull ^ (uint64_t(seed) << 32);
  h ^= h >> 29;
  h *= 0xbf58476d1ce4e5b9ull;
  h ^= h >> 32;
  return (h & 3) != 0;
}

// The seed of port `p`'s stream and ready pattern (port 0 keeps the set's seed).
inline uint32_t port_seed(uint32_t seed, uint32_t p) {
  return seed + 7919 * p;
}

inline std::vector<Op> make_stream(uint32_t seed, uint32_t n_ops, uint32_t num_threads,
                                   uint32_t num_warps, uint32_t xlen) {
  static const uint32_t codes[] = {
      wop::REDUX_ADD, wop::REDUX_MIN_S, wop::REDUX_MAX_S, wop::REDUX_MIN_U, wop::REDUX_MAX_U,
      wop::REDUX_AND, wop::REDUX_OR, wop::REDUX_XOR, wop::ELECT_P, wop::ELECT_L,
      wop::MATCH_ANY_B32, wop::MATCH_ANY_B64, wop::MATCH_ALL_B32, wop::MATCH_ALL_B64};
  const uint64_t full = (num_threads == 64) ? ~0ull : ((1ull << num_threads) - 1);
  const uint64_t xmask = (xlen == 64) ? ~0ull : 0xffffffffull;
  std::mt19937_64 rng(seed);
  std::vector<Op> ops;
  uint64_t t = 0;
  for (uint32_t n = 0; n < n_ops; ++n) {
    Op op;
    const uint32_t gap = rng() % 8;
    t += (gap < 5) ? 0 : (gap < 7) ? rng() % 4 : rng() % 64;
    op.arrive = t;
    op.code = codes[rng() % (sizeof(codes) / sizeof(codes[0]))];
    op.wid = (uint32_t)(rng() % num_warps);
    switch (rng() % 4) {
    case 0:  op.tmask = full; break;
    case 1:  op.tmask = 1ull << (rng() % num_threads); break;
    default: op.tmask = rng() & full; break;
    }
    if (op.tmask == 0)
      op.tmask = full;
    // A small pool gives MATCH.ANY every k from 1 to num_threads.
    const uint32_t pool = 1 + (uint32_t)(rng() % num_threads);
    std::vector<uint64_t> vals(pool);
    for (auto& v : vals) {
      v = rng() & xmask;
      if (rng() % 4 == 0)
        v = (v & 0xffffffffull) | ((rng() % 2) ? (xmask & ~0xffffffffull) : 0);
    }
    const uint64_t member = (rng() % 8 == 0) ? (rng() & full & ~op.tmask) : (rng() % 2 ? full : rng() & full);
    op.rs1.resize(num_threads);
    op.rs2.resize(num_threads);
    for (uint32_t i = 0; i < num_threads; ++i) {
      const uint64_t v = vals[rng() % pool];
      op.rs1[i] = wop::is_elect(op.code) ? (member & xmask) : v;
      op.rs2[i] = (xlen == 32 && wop::is_b64(op.code)) ? (vals[(v >> 3) % pool] & xmask) : 0;
    }
    ops.push_back(std::move(op));
  }
  return ops;
}

inline uint64_t fold(const std::vector<uint64_t>& rd, uint64_t tmask) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < rd.size(); ++i) {
    if ((tmask >> i) & 1)
      h = (h ^ rd[i]) * 1099511628211ull;
  }
  return h;
}

inline void csv_header(FILE* f) {
  std::fprintf(f, "set,idx,wid,code,tmask,k,accept,done,result\n");
}

inline void csv_row(FILE* f, uint32_t set, uint32_t idx, const Op& op, uint32_t k, uint64_t accept,
                    uint64_t done, uint64_t result) {
  std::fprintf(f, "%u,%u,%u,0x%02x,0x%llx,%u,%llu,%llu,0x%016llx\n", set, idx, op.wid, op.code,
               (unsigned long long)op.tmask, k, (unsigned long long)accept,
               (unsigned long long)done, (unsigned long long)result);
}

} // namespace warp_ops_stream
