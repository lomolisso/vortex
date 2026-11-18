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

#include <uniform_ref.h>
#include <cstdint>

namespace vortex {

// Lane results of a uniform op. UR sources arrive broadcast; every active lane gets the
// value, and the UR writeback keeps the first active lane's. Mask: test(i); Src/Dst:
// indexable, elements with a .u64 member (reg_data_t).
template <typename Mask, typename Src, typename Dst>
void uniform_execute(uint32_t op, uint32_t xlen, uint32_t num_threads, const Mask& tmask,
                     const Src& rs1, const Src& rs2, Dst& rd) {
  uint32_t first = num_threads;
  for (uint32_t i = 0; i < num_threads; ++i) {
    if (tmask.test(i)) {
      first = i;
      break;
    }
  }
  if (first == num_threads)
    return;
  for (uint32_t i = 0; i < num_threads; ++i) {
    if (!tmask.test(i))
      continue;
    const uint32_t src = (op == uniform_ref::UOP_R2UR) ? first : i;
    rd[i].u64 = uniform_ref::execute(op, xlen, rs1[src].u64, rs2[src].u64);
  }
}

} // namespace vortex
