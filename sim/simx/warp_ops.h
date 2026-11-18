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

#include <warp_ops_ref.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace vortex {

namespace wop = warp_ops_ref;

// Lane results of a warp op. Mask: test(i); Src/Dst: indexable, elements with
// a .u64 member (reg_data_t). Values are XLEN bits wide.
template <typename Mask, typename Src, typename Dst>
void warp_op_execute(uint32_t code, uint32_t xlen, uint32_t num_threads, const Mask& tmask,
                     const Src& rs1, const Src& rs2, Dst& rd) {
  const uint64_t xmask = (xlen == 64) ? ~0ull : 0xffffffffull;
  uint64_t active = 0;
  for (uint32_t i = 0; i < num_threads; ++i)
    active |= uint64_t(tmask.test(i)) << i;
  auto key = [&](uint32_t i) -> uint64_t {
    if (!wop::is_b64(code))
      return uint32_t(rs1[i].u64);
    return (xlen == 64) ? rs1[i].u64 : ((rs2[i].u64 << 32) | uint32_t(rs1[i].u64));
  };

  uint64_t value = 0;
  bool per_lane = false;
  if (wop::is_redux(code)) {
    uint32_t acc = 0;
    for (uint64_t m = active; m != 0; m &= m - 1) {
      const uint32_t i = __builtin_ctzll(m), v = uint32_t(rs1[i].u64);
      acc = (m == active) ? v : wop::redux_step(code, acc, v);
    }
    value = uint64_t(int64_t(int32_t(acc))) & xmask;
  } else if (wop::is_elect(code)) {
    uint64_t members = 0;
    for (uint64_t m = active; m != 0; m &= m - 1) {
      const uint32_t i = __builtin_ctzll(m);
      members |= ((rs1[i].u64 >> i) & 1) << i;
    }
    if (code == wop::ELECT_L) {
      value = members ? uint64_t(__builtin_ctzll(members)) : (wop::NO_LEADER & xmask);
    } else {
      for (uint64_t m = active; m != 0; m &= m - 1) {
        const uint32_t i = __builtin_ctzll(m);
        rd[i].u64 = (members & -members) == (uint64_t(1) << i);
      }
      return;
    }
  } else if (wop::is_match_all(code)) {
    const uint64_t k0 = key(__builtin_ctzll(active));
    bool all = true;
    for (uint64_t m = active; m != 0; m &= m - 1)
      all &= (key(__builtin_ctzll(m)) == k0);
    value = all ? active : 0;
  } else {
    per_lane = true;
    uint64_t left = active;
    while (left != 0) {
      const uint64_t k0 = key(__builtin_ctzll(left));
      uint64_t same = 0;
      for (uint64_t m = left; m != 0; m &= m - 1) {
        const uint32_t i = __builtin_ctzll(m);
        if (key(i) == k0)
          same |= uint64_t(1) << i;
      }
      for (uint64_t m = same; m != 0; m &= m - 1)
        rd[__builtin_ctzll(m)].u64 = same;
      left &= ~same;
    }
  }
  if (per_lane)
    return;
  for (uint64_t m = active; m != 0; m &= m - 1)
    rd[__builtin_ctzll(m)].u64 = value;
}

// Distinct values among the active lanes, as MATCH.ANY's latency counts them.
template <typename Mask, typename Src>
uint32_t warp_op_distinct(uint32_t code, uint32_t xlen, uint32_t num_threads, const Mask& tmask,
                          const Src& rs1, const Src& rs2) {
  std::vector<uint64_t> keys;
  for (uint32_t i = 0; i < num_threads; ++i) {
    if (!tmask.test(i))
      continue;
    uint64_t v = uint32_t(rs1[i].u64);
    if (wop::is_b64(code))
      v = (xlen == 64) ? rs1[i].u64 : ((rs2[i].u64 << 32) | uint32_t(rs1[i].u64));
    bool found = false;
    for (auto k : keys)
      found |= (k == v);
    if (!found)
      keys.push_back(v);
  }
  return keys.empty() ? 1 : uint32_t(keys.size());
}

} // namespace vortex
