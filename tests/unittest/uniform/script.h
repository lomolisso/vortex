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

// The uniform-datapath script shared by the RTL bench (hw/unittest/uniform) and its SimX twin:
// decoded instructions, the stand-in execute unit and the CSV row both sides print.
#pragma once

#include <uniform_ref.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace uniform_script {

enum Kind : uint32_t { INIT_GPR, INIT_UR, OP };

struct Instr {
  uint32_t tag;        // carried in PC
  uint32_t wid;
  uint32_t tmask;
  Kind kind;
  uint32_t op;         // uniform_ref::UOP_*
  bool dst_ur;
  uint32_t rd;
  uint32_t rs[3];
  uint32_t used_rs;    // GPR source slots
  uint32_t ur_rs;      // UR source slots
  uint32_t ur_quad;    // quad source slots (0, 1)
  uint32_t lat;        // stand-in execute latency
  bool isolated;       // issued into an empty pipeline; its collect cycles are reported
};

inline uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

inline uint64_t xmask(uint32_t xlen) { return xlen == 64 ? ~0ull : 0xffffffffull; }

inline uint32_t first_lane(uint32_t tmask, uint32_t nt) {
  for (uint32_t l = 0; l < nt; ++l) {
    if ((tmask >> l) & 1)
      return l;
  }
  return nt;
}

inline uint32_t ur_slots(const Instr& in) { return in.ur_rs | in.ur_quad; }

// Lane results of the stand-in execute unit; a UR destination gets distinct lanes so that only
// the first active lane's value may reach the UR file.
inline void execute(const Instr& in, uint32_t nt, uint32_t xlen, const uint64_t* a,
                    const uint64_t* b, const uint64_t* c, uint64_t* out) {
  const uint32_t first = first_lane(in.tmask, nt);
  for (uint32_t l = 0; l < nt; ++l) {
    uint64_t r;
    switch (in.kind) {
    case INIT_GPR: r = mix(in.wid * 1000 + in.rd * 37 + l); break;
    case INIT_UR:  r = mix(0x55550000ull + in.wid * 1000 + in.rd * 37 + first); break;
    default: {
      const uint32_t s = (in.op == uniform_ref::UOP_R2UR) ? first : l;
      r = uniform_ref::execute(in.op, xlen, a[s], b[s]);
      if ((ur_slots(in) | in.used_rs) & 4)
        r ^= c[s];
    } break;
    }
    if (in.dst_ur && l != first)
      r += uint64_t(l) << 40 | uint64_t(l) << 8;
    out[l] = r & xmask(xlen);
  }
}

inline std::vector<Instr> make_script(uint32_t nw, uint32_t nt, uint32_t urz) {
  std::vector<Instr> s;
  const uint32_t full = (nt >= 32) ? ~0u : ((1u << nt) - 1);
  auto add = [&](uint32_t wid, uint32_t tmask, Kind kind, uint32_t op, bool dst_ur, uint32_t rd,
                 uint32_t r1, uint32_t r2, uint32_t r3, uint32_t used, uint32_t ur, uint32_t quad,
                 uint32_t lat, bool iso) {
    s.push_back({uint32_t(s.size() + 1), wid, tmask, kind, op, dst_ur, rd, {r1, r2, r3}, used, ur,
                 quad, lat, iso});
  };
  using namespace uniform_ref;
  // Register init, warps interleaved; UR3 and UR7 written by divergent masks.
  for (uint32_t r = 1; r <= 8; ++r)
    for (uint32_t w = 0; w < nw; ++w)
      add(w, full, INIT_GPR, 0, false, r, 0, 0, 0, 0, 0, 0, 1 + (r + w) % 3, false);
  for (uint32_t r = 1; r <= 12; ++r)
    for (uint32_t w = 0; w < nw; ++w) {
      uint32_t m = full;
      if (r == 3) m = full & ~((1u << (nt / 2)) - 1);
      if (r == 7) m = full & ~(1u << w % nt) & ~1u;
      add(w, m, INIT_UR, 0, true, r, 0, 0, 0, 0, 0, 0, 1 + (r * 3 + w) % 4, false);
    }
  // Collect cycles, warp 0, one instruction in an empty pipeline each (row 1 is the zero).
  add(0, full, OP, UOP_ADD, false, 20, 0, 0, 0, 0, 0, 0, 1, true);
  add(0, full, OP, UOP_ADD, false, 20, 1, 2, 0, 3, 0, 0, 1, true);
  add(0, full, OP, UOP_ADD, false, 20, 5, 1, 0, 2, 1, 0, 1, true);   // UR5 / x1 share a bank
  add(0, full, OP, UOP_ADD, true, 20, 1, 2, 0, 0, 3, 0, 1, true);
  add(0, full, OP, UOP_ADD, false, 21, 4, 2, 0, 2, 0, 1, 1, true);
  add(0, full, OP, UOP_ADD, false, 22, 4, 8, 0, 0, 0, 3, 1, true);
  add(0, full, OP, UOP_ADD, false, 23, 1, 2, 3, 0, 7, 0, 1, true);
  add(0, full, OP, UOP_ADD, false, 24, urz, 3, 0, 2, 1, 0, 1, true);
  // Back-to-back hazards on URs, warps interleaved per instruction.
  for (uint32_t w = 0; w < nw; ++w) {
    const uint32_t div = full & ~3u;
    add(w, full, OP, UOP_ADD, true, 21, 1, 2, 0, 0, 3, 0, 3, false);
    add(w, full, OP, UOP_ADD, false, 25, 21, 2, 0, 2, 1, 0, 2, false);           // RAW UR21
    add(w, div, OP, UOP_R2UR, true, 22, 3, 0, 0, 1, 0, 0, 20, false);            // slow write
    add(w, full, OP, UOP_ADD, true, 22, 1, 4, 0, 0, 3, 0, 3, false);             // WAW UR22
    add(w, full, OP, UOP_XOR, false, 26, 22, 1, 0, 2, 1, 0, 1, false);           // RAW UR22
    add(w, full, OP, UOP_SUB, true, 23, 22, 21, 0, 0, 3, 0, 3, false);
    add(w, full, OP, UOP_ADD, false, 27, 20, 4, 0, 2, 0, 1, 2, false);           // quad UR20-23
    add(w, div, OP, UOP_UR2R, false, 28, 23, 0, 0, 0, 1, 0, 1, false);
    add(w, full, OP, UOP_SRA, true, 24, 6, 7, 0, 0, 3, 0, 3, false);
    add(w, full, OP, UOP_SLTU, true, 25, 8, 9, 0, 0, 3, 0, 3, false);
    add(w, full, OP, UOP_SLL, false, 29, 5, 10, 0, 1, 2, 0, 1, false);
    add(w, full, OP, UOP_SEQ, false, 30, 24, 25, 0, 0, 3, 0, 1, false);
    add(w, full, OP, UOP_OR, false, 31, 7, urz, 0, 0, 3, 0, 1, false);           // URZ source
    add(w, full, OP, UOP_ADD, false, 9, 3, 24, 7, 0, 7, 0, 2, false);            // three URs
  }
  return s;
}

inline std::string lanes(const uint64_t* v, uint32_t tmask, uint32_t nt) {
  std::string out;
  char buf[24];
  for (uint32_t l = 0; l < nt; ++l) {
    if (l)
      out += ':';
    if ((tmask >> l) & 1) {
      std::snprintf(buf, sizeof(buf), "%llx", (unsigned long long)v[l]);
      out += buf;
    } else {
      out += '-';
    }
  }
  return out;
}

inline void csv_header(FILE* f) { std::fprintf(f, "tag,wid,tmask,extra,rs1,rs2,rs3,result\n"); }

// extra < 0: not reported. Operand slots the instruction does not read print as '-'.
inline void csv_row(FILE* f, const Instr& in, uint32_t nt, int extra, const uint64_t* const rs[3],
                    const uint64_t* out, uint64_t ur_value) {
  std::fprintf(f, "%u,%u,%x,", in.tag, in.wid, in.tmask);
  if (extra >= 0)
    std::fprintf(f, "%d", extra);
  else
    std::fprintf(f, "-");
  const uint32_t read = in.used_rs | ur_slots(in);
  for (uint32_t i = 0; i < 3; ++i)
    std::fprintf(f, ",%s", ((read >> i) & 1) ? lanes(rs[i], in.tmask, nt).c_str() : "-");
  if (in.dst_ur)
    std::fprintf(f, ",U%u=%llx\n", in.rd, (unsigned long long)ur_value);
  else
    std::fprintf(f, ",x%u=%s\n", in.rd, lanes(out, in.tmask, nt).c_str());
}

} // namespace uniform_script
