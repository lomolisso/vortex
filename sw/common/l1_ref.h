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

// The SM L1's XOR-folded set index and its replacement LFSR, shared by the
// slot golden (l1_slot_ref.h). Shapes come from arguments, never VX_config.h.

#include <cstdint>

namespace vortex {
namespace l1_ref {

// Every line-address bit above the set field folded (XOR) into it.
inline uint32_t l1_set_index(uint64_t line_addr, uint32_t sets_log2) {
  if (sets_log2 == 0)
    return 0;
  uint64_t mask = (uint64_t(1) << sets_log2) - 1;
  uint64_t idx = 0;
  while (line_addr != 0) {
    idx ^= line_addr & mask;
    line_addr >>= sets_log2;
  }
  return uint32_t(idx);
}

// 16-bit Galois LFSR, x^16 + x^14 + x^13 + x^11 + 1, one per bank.
constexpr uint32_t L1_LFSR_SEED = 0xACE1;
constexpr uint32_t L1_LFSR_TAPS = 0xB400;

struct L1Lfsr {
  uint32_t s = L1_LFSR_SEED;
  uint32_t next() {
    uint32_t lsb = s & 1;
    s >>= 1;
    if (lsb)
      s ^= L1_LFSR_TAPS;
    return s;
  }
};

} // namespace l1_ref
} // namespace vortex
