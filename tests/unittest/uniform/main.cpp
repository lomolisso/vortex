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

// SimX twin of hw/unittest/uniform: the operand stage's UR rules (broadcast, quad lanes 0..3,
// URZ, first-active-lane write, in-order UR hazards) and SimX's collect-cycle rule.

#include "script.h"
#include "../../../sim/simx/uniform_ops.h"

#include <bitset>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef VX_CFG_H100_FE_ENABLE
#error "the uniform datapath needs the H100 front end"
#endif

namespace us = uniform_script;

static constexpr uint32_t NT = VX_CFG_NUM_THREADS;
static constexpr uint32_t NW = VX_CFG_NUM_WARPS;
static constexpr uint32_t XLEN = VX_CFG_XLEN;
static constexpr uint32_t UREGS = VX_CFG_UREG_COUNT;
static constexpr uint32_t URZ = UREGS - 1;

struct Lane {
  uint64_t u64;
};

int main(int argc, char** argv) {
  const char* csv_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!std::strncmp(argv[i], "--csv=", 6))
      csv_path = argv[i] + 6;
  }
  const auto script = us::make_script(NW, NT, URZ);
  std::vector<uint64_t> gpr(NW * 32 * NT, 0), ur(NW * UREGS, 0);
  auto G = [&](uint32_t w, uint32_t r, uint32_t l) -> uint64_t& { return gpr[(w * 32 + r) * NT + l]; };
  auto U = [&](uint32_t w, uint32_t r) -> uint64_t { return r >= URZ ? 0 : ur[w * UREGS + r]; };

  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  if (csv)
    us::csv_header(csv);
  int errors = 0;
  for (const auto& in : script) {
    uint64_t rs[3][NT], out[NT];
    for (uint32_t i = 0; i < 3; ++i) {
      for (uint32_t l = 0; l < NT; ++l) {
        uint64_t v = 0;
        if (i < 2 && ((in.ur_quad >> i) & 1))
          v = (l < 4) ? U(in.wid, in.rs[i] + l) : 0;
        else if ((in.ur_rs >> i) & 1)
          v = U(in.wid, in.rs[i]);
        else if ((in.used_rs >> i) & 1)
          v = in.rs[i] ? G(in.wid, in.rs[i], l) : 0;
        rs[i][l] = v;
      }
    }
    us::execute(in, NT, XLEN, rs[0], rs[1], rs[2], out);

    const uint32_t first = us::first_lane(in.tmask, NT);
    if (in.kind == us::OP && !((us::ur_slots(in) | in.used_rs) & 4)) {
      std::bitset<NT> mask(in.tmask);
      Lane a[NT], b[NT], r[NT];
      for (uint32_t l = 0; l < NT; ++l) {
        a[l].u64 = rs[0][l];
        b[l].u64 = rs[1][l];
        r[l].u64 = 0;
      }
      vortex::uniform_execute(in.op, XLEN, NT, mask, a, b, r);
      if (r[first].u64 != out[first]) {
        std::printf("FAIL tag %u: SimX uniform_execute %llx, script %llx\n", in.tag,
                    (unsigned long long)r[first].u64, (unsigned long long)out[first]);
        ++errors;
      }
    }

    int extra = -1;
    if (in.isolated) {
      uint32_t gpr_stalls = 0;
      for (uint32_t i = 0; i < 3; ++i)
        for (uint32_t j = i + 1; j < 3; ++j)
          if (((in.used_rs >> i) & 1) && ((in.used_rs >> j) & 1) && in.rs[i] && in.rs[j]
              && in.rs[i] % VX_CFG_NUM_GPR_BANKS == in.rs[j] % VX_CFG_NUM_GPR_BANKS)
            ++gpr_stalls;
      uint32_t slots = 0;
      for (uint32_t i = 0; i < 3; ++i)
        slots += (us::ur_slots(in) >> i) & 1;
      extra = (int)uniform_ref::collect_stalls(gpr_stalls, slots);
    }

    uint64_t ur_value = 0;
    if (in.dst_ur) {
      if (in.rd < URZ)
        ur[in.wid * UREGS + in.rd] = out[first];
      ur_value = U(in.wid, in.rd);
    } else {
      for (uint32_t l = 0; l < NT; ++l)
        if ((in.tmask >> l) & 1)
          G(in.wid, in.rd, l) = out[l];
    }
    const uint64_t* rsp[3] = {rs[0], rs[1], rs[2]};
    if (csv)
      us::csv_row(csv, in, NT, extra, rsp, out, ur_value);
  }
  if (csv)
    std::fclose(csv);
  if (errors) {
    std::printf("sw-uniform: %d errors\nFAILED!\n", errors);
    return 1;
  }
  std::printf("sw-uniform: %zu instructions (%u warps, %u threads, %u URs)\nPASSED!\n",
              script.size(), NW, NT, UREGS);
  return 0;
}
