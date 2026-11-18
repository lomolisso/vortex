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

#include "vl_simulator.h"
#include "VVX_uniform_top.h"
#include "VX_config.h"
#include "script.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef VX_CFG_H100_FE_ENABLE
#error "the uniform datapath needs the H100 front end"
#endif

namespace us = uniform_script;

static constexpr uint32_t NT = VX_CFG_NUM_THREADS;
static constexpr uint32_t NW = VX_CFG_NUM_WARPS;
static constexpr uint32_t XLEN = VX_CFG_XLEN;
static constexpr uint32_t URZ = VX_CFG_UREG_COUNT - 1;
static constexpr uint32_t QUIET = 8;

static_assert(VX_CFG_SIMD_WIDTH == VX_CFG_NUM_THREADS, "the bench drives one packet per warp");

template <typename W>
static void set_bits(W& w, uint32_t off, uint32_t width, uint64_t v) {
  for (uint32_t b = 0; b < width; ++b) {
    const uint32_t bit = off + b, word = bit / 32, sh = bit % 32;
    w[word] = (w[word] & ~(1u << sh)) | (uint32_t((v >> b) & 1) << sh);
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

struct Result {
  bool dispatched = false;
  uint64_t fire = 0, dispatch = 0;
  uint64_t rs[3][NT] = {};
  uint64_t out[NT] = {};
};

struct Wb {
  uint64_t due;
  uint32_t idx;
};

int main(int argc, char** argv) {
  const char* csv_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!std::strncmp(argv[i], "--csv=", 6))
      csv_path = argv[i] + 6;
  }
  const auto script = us::make_script(NW, NT, URZ);
  std::vector<Result> res(script.size());

  vl_simulator<VVX_uniform_top> sim;
  auto* t = sim.operator->();
  uint64_t tick = 0;
  t->decode_valid = 0;
  t->writeback_valid = 0;
  t->dispatch_ready = 1;
  t->reset = 1;
  for (int i = 0; i < 4; ++i)
    tick = sim.step(tick, 2);
  t->reset = 0;

  int errors = 0;
  size_t next = 0, written = 0;
  uint32_t outstanding = 0, quiet = 0;
  bool hold = false; // an isolated instruction is in flight
  std::vector<Wb> wbq;
  const uint64_t limit = script.size() * 200 + 10000;
  uint64_t cyc = 0;
  for (; written < script.size() && cyc < limit; ++cyc) {
    const bool empty = outstanding == 0 && wbq.empty();
    quiet = empty ? quiet + 1 : 0;
    if (hold && empty)
      hold = false;
    bool offer = next < script.size() && !hold;
    if (offer && script[next].isolated)
      offer = quiet > QUIET;
    t->decode_valid = offer;
    if (offer) {
      const auto& in = script[next];
      t->decode_wid = in.wid;
      t->decode_tmask = in.tmask;
      t->decode_PC = in.tag;
      t->decode_wb = !in.dst_ur;
      t->decode_wr_ur = in.dst_ur;
      t->decode_ur_rs = in.ur_rs & ~in.ur_quad;
      t->decode_ur_quad = in.ur_quad;
      t->decode_used_rs = in.used_rs;
      t->decode_rd = in.rd;
      t->decode_rs1 = in.rs[0];
      t->decode_rs2 = in.rs[1];
      t->decode_rs3 = in.rs[2];
    }

    size_t wb_pick = wbq.size();
    for (size_t i = 0; i < wbq.size(); ++i) {
      if (wbq[i].due <= cyc && (wb_pick == wbq.size() || wbq[i].due < wbq[wb_pick].due))
        wb_pick = i;
    }
    t->writeback_valid = wb_pick < wbq.size();
    if (wb_pick < wbq.size()) {
      const auto& in = script[wbq[wb_pick].idx];
      const auto& r = res[wbq[wb_pick].idx];
      t->writeback_wid = in.wid;
      t->writeback_tmask = in.tmask;
      t->writeback_PC = in.tag;
      t->writeback_rd = in.rd;
      t->writeback_wb = !in.dst_ur;
      t->writeback_wr_ur = in.dst_ur;
      for (uint32_t l = 0; l < NT; ++l)
        set_bits(t->writeback_data, l * XLEN, XLEN, r.out[l]);
    }

    t->eval();
    if (offer && t->decode_ready) {
      res[next].fire = cyc;
      ++outstanding;
      hold = script[next].isolated;
      ++next;
    }
    if (wb_pick < wbq.size()) {
      wbq.erase(wbq.begin() + wb_pick);
      --outstanding;
      ++written;
    }
    if (t->dispatch_valid) {
      const uint64_t tag = t->dispatch_PC;
      if (tag == 0 || tag > script.size() || res[tag - 1].dispatched || !t->dispatch_sop) {
        std::printf("FAIL cycle %llu: unexpected dispatch tag %llu\n", (unsigned long long)cyc,
                    (unsigned long long)tag);
        ++errors;
        break;
      }
      const uint32_t idx = uint32_t(tag - 1);
      const auto& in = script[idx];
      auto& r = res[idx];
      r.dispatched = true;
      r.dispatch = cyc;
      if ((uint32_t)t->dispatch_tmask != in.tmask) {
        std::printf("FAIL tag %u: tmask %x\n", in.tag, (uint32_t)t->dispatch_tmask);
        ++errors;
      }
      for (uint32_t l = 0; l < NT; ++l) {
        r.rs[0][l] = get_bits(t->dispatch_rs1_data, l * XLEN, XLEN);
        r.rs[1][l] = get_bits(t->dispatch_rs2_data, l * XLEN, XLEN);
        r.rs[2][l] = get_bits(t->dispatch_rs3_data, l * XLEN, XLEN);
      }
      us::execute(in, NT, XLEN, r.rs[0], r.rs[1], r.rs[2], r.out);
      wbq.push_back({cyc + in.lat, idx});
    }
    tick = sim.step(tick, 2);
  }
  if (written != script.size()) {
    std::printf("FAIL %zu of %zu instructions written back by cycle %llu\n", written, script.size(),
                (unsigned long long)cyc);
    ++errors;
  }

  FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
  if (csv)
    us::csv_header(csv);
  int64_t base = -1;
  for (size_t i = 0; i < script.size() && csv; ++i) {
    const auto& in = script[i];
    const auto& r = res[i];
    int extra = -1;
    if (in.isolated) {
      const int64_t lat = int64_t(r.dispatch - r.fire);
      if (base < 0)
        base = lat;
      extra = int(lat - base);
    }
    const uint32_t first = us::first_lane(in.tmask, NT);
    const uint64_t ur_value = (in.dst_ur && in.rd < URZ) ? r.out[first] : 0;
    const uint64_t* rsp[3] = {r.rs[0], r.rs[1], r.rs[2]};
    us::csv_row(csv, in, NT, extra, rsp, r.out, ur_value);
  }
  if (csv)
    std::fclose(csv);
  if (errors) {
    std::printf("hw-uniform: %d errors\nFAILED!\n", errors);
    return 1;
  }
  std::printf("hw-uniform: %zu instructions in %llu cycles (decode-to-dispatch base %lld)\nPASSED!\n",
              script.size(), (unsigned long long)cyc, (long long)base);
  return 0;
}
