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

// Golden model of the H100 wgmma shared-memory matrix descriptor (PTX ISA
// "Matrix Descriptor Format" and the canonical K-/MN-major layouts of the
// swizzle modes) and of the wgmma accumulator fragment, for the tensor unit
// built with VX_CFG_TCU_H100_ENABLE. Host tests, SimX and the RTL unit bench
// share it.
//
// Addresses are CTA-relative bytes (the block's local-memory slot base is
// added last). The swizzle XORs bits [4,7) of the CTA-relative address with
// bits [7,10), the same function and the same origin the DXA's tensor loads
// use, so what a swizzled TMA box writes is what a wgmma with the matching
// descriptor reads.

#include <cstdint>
#include "dxa_ref.h"

namespace vortex {
namespace tcu_desc_ref {

// Layout field [63:62].
enum : uint32_t { LAYOUT_NONE = 0, LAYOUT_128B = 1, LAYOUT_64B = 2, LAYOUT_32B = 3 };

constexpr uint32_t FIELD_MASK   = 0x3FFFFu;  // byte addresses and offsets reach 256 KB
constexpr uint32_t START_LSB    = 0;
constexpr uint32_t LBO_LSB      = 16;
constexpr uint32_t SBO_LSB      = 32;
constexpr uint32_t BASE_LSB     = 49;
constexpr uint32_t MN_MAJOR_BIT = 52;        // Vortex: the transpose flag (reserved on sm_90)
constexpr uint32_t LAYOUT_LSB   = 62;

constexpr uint64_t FIELD14_MASK = 0x3FFFull;
constexpr uint64_t MN_MAJOR     = 1ull << MN_MAJOR_BIT;
constexpr uint64_t DEFINED_MASK = (FIELD14_MASK << START_LSB) | (FIELD14_MASK << LBO_LSB)
                                | (FIELD14_MASK << SBO_LSB) | (7ull << BASE_LSB) | MN_MAJOR
                                | (3ull << LAYOUT_LSB);
constexpr uint64_t RESERVED_MASK = ~DEFINED_MASK;  // bits 14-15, 30-31, 46-48, 53-61

// Assert texts, identical in the RTL, SimX and the kernel library.
constexpr const char* MSG_BASE_OFFSET = "wgmma descriptor base offset must be 0";
constexpr const char* MSG_RESERVED    = "wgmma descriptor reserved bits must be 0";
constexpr const char* MSG_MN_TYPE     = "wgmma MN-major needs a 16-bit type";

struct Desc {
  uint32_t start = 0;        // bytes, 16-B granular
  uint32_t lbo = 0;          // bytes
  uint32_t sbo = 0;          // bytes
  uint32_t base_offset = 0;
  uint32_t layout = LAYOUT_NONE;
  bool     mn_major = false;
  uint64_t reserved = 0;     // the reserved bits as found
};

inline uint64_t field(uint32_t bytes) {
  return (uint64_t)((bytes & FIELD_MASK) >> 4);
}

inline uint64_t encode(uint32_t start, uint32_t lbo, uint32_t sbo, uint32_t layout,
                       bool mn_major = false, uint32_t base_offset = 0) {
  return (field(start) << START_LSB) | (field(lbo) << LBO_LSB) | (field(sbo) << SBO_LSB)
       | ((uint64_t)(base_offset & 7u) << BASE_LSB) | (mn_major ? MN_MAJOR : 0)
       | ((uint64_t)(layout & 3u) << LAYOUT_LSB);
}

inline uint64_t encode(const Desc& d) {
  return encode(d.start, d.lbo, d.sbo, d.layout, d.mn_major, d.base_offset) | (d.reserved & RESERVED_MASK);
}

inline Desc decode(uint64_t v) {
  Desc d;
  d.start       = (uint32_t)((v >> START_LSB) & FIELD14_MASK) << 4;
  d.lbo         = (uint32_t)((v >> LBO_LSB) & FIELD14_MASK) << 4;
  d.sbo         = (uint32_t)((v >> SBO_LSB) & FIELD14_MASK) << 4;
  d.base_offset = (uint32_t)((v >> BASE_LSB) & 7u);
  d.mn_major    = ((v >> MN_MAJOR_BIT) & 1u) != 0;
  d.layout      = (uint32_t)((v >> LAYOUT_LSB) & 3u);
  d.reserved    = v & RESERVED_MASK;
  return d;
}

inline bool is_mn_major(uint64_t v) {
  return ((v >> MN_MAJOR_BIT) & 1u) != 0;
}

// 16-B chunks per swizzle row: 8 / 4 / 2, and 1 without swizzle.
inline uint32_t swizzle_chunks(uint32_t layout) {
  return layout == LAYOUT_128B ? 8 : layout == LAYOUT_64B ? 4 : layout == LAYOUT_32B ? 2 : 1;
}

// The DXA swizzle mode of a descriptor layout (the two encodings differ).
inline uint32_t dxa_swizzle(uint32_t layout) {
  return layout == LAYOUT_128B ? dxa_ref::SWZ_128B
       : layout == LAYOUT_64B  ? dxa_ref::SWZ_64B
       : layout == LAYOUT_32B  ? dxa_ref::SWZ_32B
       : dxa_ref::SWZ_NONE;
}

// The descriptor layout of a DXA swizzle mode.
inline uint32_t layout_of_dxa(uint32_t swz) {
  return swz == dxa_ref::SWZ_128B ? LAYOUT_128B
       : swz == dxa_ref::SWZ_64B  ? LAYOUT_64B
       : swz == dxa_ref::SWZ_32B  ? LAYOUT_32B
       : LAYOUT_NONE;
}

// The first assert a descriptor trips for an operand of esize-byte elements
// (the major is the descriptor's bit 52), or nullptr.
inline const char* check(uint64_t v, uint32_t esize) {
  if (v & RESERVED_MASK)
    return MSG_RESERVED;
  if ((v >> BASE_LSB) & 7u)
    return MSG_BASE_OFFSET;
  if (is_mn_major(v) && esize != 2)
    return MSG_MN_TYPE;
  return nullptr;
}

// Element (mn, k) of the operand before the swizzle, CTA-relative bytes.
// K-major: mn is the row (m of A, n of B) and k runs along the 16-B chunks.
// MN-major: mn runs along the 16-B chunks and k is the row. In swizzled
// K-major the LBO is not used (the k chunks of a row are adjacent).
inline uint32_t linear_addr(const Desc& d, bool mn_major, uint32_t esize, uint32_t mn, uint32_t k) {
  const uint32_t W = swizzle_chunks(d.layout);
  const uint32_t T = 16 / esize;
  if (!mn_major) {
    const uint32_t kc = (k * esize) / 16, kb = (k * esize) % 16;
    if (d.layout == LAYOUT_NONE)
      return d.start + (mn % 8) * 16 + (mn / 8) * d.sbo + kc * d.lbo + kb;
    return d.start + (mn % 8) * (W * 16) + (mn / 8) * d.sbo + k * esize;
  }
  const uint32_t mb = (mn % T) * esize, mc = mn / T;
  if (d.layout == LAYOUT_NONE)
    return d.start + mb + mc * d.sbo + (k % 8) * 16 + (k / 8) * d.lbo;
  return d.start + mb + (mc % W) * 16 + (mc / W) * d.lbo + (k % 8) * (W * 16) + (k / 8) * d.sbo;
}

// The swizzle on a CTA-relative address (the identity without swizzle).
inline uint32_t swizzle(uint32_t a, uint32_t layout) {
  return dxa_ref::swizzle(a, dxa_swizzle(layout));
}

// The CTA-relative byte address of element (mn, k) of the operand a
// descriptor names. mn_major is the instruction's major (the hardware takes
// it from bit 52); check() lists the asserts.
inline uint32_t operand_addr(uint64_t desc, bool mn_major, uint32_t esize, uint32_t mn, uint32_t k) {
  const Desc d = decode(desc);
  return swizzle(linear_addr(d, mn_major, esize, mn, k), d.layout);
}

// The address of a 16-B granule, the unit the tensor unit fetches.
// K-major: line = mn, chunk = the 16-B chunk along k. MN-major: line = k,
// chunk = the 16-B chunk along mn. The granule's bytes are contiguous.
inline uint32_t chunk_addr(uint64_t desc, bool mn_major, uint32_t esize, uint32_t line, uint32_t chunk) {
  const uint32_t T = 16 / esize;
  return mn_major ? operand_addr(desc, true, esize, chunk * T, line)
                  : operand_addr(desc, false, esize, line, chunk * T);
}

// The local-memory byte the tensor unit reads: the block's slot base plus
// the CTA-relative address.
inline uint64_t lmem_addr(uint64_t slot_base, uint32_t cta_rel) {
  return slot_base + cta_rel;
}

///////////////////////////////////////////////////////////////////////////////
// Instruction shape and the accumulator fragment of a warpgroup (4 warps of
// 32 threads; thread t = 32 w + L, warp w computes rows 16w..16w+15 of m64).

// K of one instruction: 16 for 16-bit inputs, 32 for 8-bit inputs.
inline uint32_t inst_k(uint32_t esize) {
  return 32 / esize;
}

// Accumulator registers per thread: N/2 for an fp32 D, N/4 for a packed f16 D.
inline uint32_t acc_regs(uint32_t n, bool packed_f16) {
  return packed_f16 ? n / 4 : n / 2;
}

// Row of register r of thread t (fp32: r = 4j + 2h + e; packed: r = 2j + h).
inline uint32_t acc_row(uint32_t t, uint32_t r, bool packed_f16) {
  const uint32_t w = t / 32, L = t % 32;
  const uint32_t h = packed_f16 ? (r & 1u) : ((r >> 1) & 1u);
  return 16 * w + L / 4 + 8 * h;
}

// Column of register r of thread t; half picks bits 15:0 (0) or 31:16 (1)
// of a packed f16 register and is ignored for fp32.
inline uint32_t acc_col(uint32_t t, uint32_t r, uint32_t half, bool packed_f16) {
  const uint32_t L = t % 32;
  if (packed_f16)
    return 8 * (r / 2) + 2 * (L % 4) + (half & 1u);
  return 8 * (r / 4) + 2 * (L % 4) + (r & 1u);
}

// The inverse: the thread, register and half holding D(m, n), m < 64.
inline void acc_slot(uint32_t m, uint32_t n, bool packed_f16, uint32_t* t, uint32_t* r, uint32_t* half) {
  const uint32_t w = m / 16, h = (m % 16) / 8, j = n / 8, e = n % 2;
  *t = 32 * w + 4 * (m % 8) + (n % 8) / 2;
  *r = packed_f16 ? 2 * j + h : 4 * j + 2 * h + e;
  *half = packed_f16 ? e : 0;
}

} // namespace tcu_desc_ref
} // namespace vortex
