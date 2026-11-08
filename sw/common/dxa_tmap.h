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

// The Vortex tensor map (the CUtensorMap model, PTX §5.5.8): a 128-byte,
// 64-byte aligned object in global memory that the DXA reads into its
// descriptor cache. Words 0-21 are the descriptor-slot words in
// VX_DCR_DXA_DESC_*_OFF order, so one decoder serves slots and maps; words
// 22-23 exist only in maps. Shared by the host encoder (dxa.h), the kernel
// library (vx_dxa.h), SimX and the tests; the RTL mirror lives in VX_dxa_pkg.
// The packers and the replace helpers are plain C++ for host and device.

#pragma once

#include <stdint.h>
#include "dxa_meta.h"

struct alignas(64) vx_tensor_map_t {
  uint32_t w[32];
};
static_assert(sizeof(vx_tensor_map_t) == 128, "a tensor map is 128 bytes");
static_assert(alignof(vx_tensor_map_t) == 64, "a tensor map is 64-byte aligned");

#define TMAP_WORDS         32
#define TMAP_FETCH_WORDS   24  // words 0-23: what a decoder reads; 24-31 are reserved, 0

#define TMAP_W_BASE_LO     0
#define TMAP_W_BASE_HI     1
#define TMAP_W_SIZE0       2   // SIZE0..4 = words 2..6, elements
#define TMAP_W_STRIDE0     7   // STRIDE0..3 = words 7..10: bytes 31:0 of the stride of dims 1..4
#define TMAP_W_META        11
#define TMAP_W_ESTRIDE0    12  // ESTRIDE0..4 = words 12..16; 0 reads as 1
#define TMAP_W_TILESIZE01  17  // tile: {box1, box0}; im2col: {pixelsPerColumn, channelsPerPixel}
#define TMAP_W_TILESIZE23  18  // tile: {box3, box2}; im2col: the pixel-box corners
#define TMAP_W_TILESIZE4   19  // tile: box4
#define TMAP_W_CFILL       20
#define TMAP_W_SMEM_STRIDE 21  // 0 in maps
#define TMAP_W_STRIDE_HI   22  // byte i = bits 39:32 of the stride of dim i + 1
#define TMAP_W_TYPE        23

// Word 23.
#define TMAP_TYPE_ELEM_LSB  0
#define TMAP_TYPE_ELEM_BITS 4
#define TMAP_TYPE_FILL_LSB  4
#define TMAP_TYPE_FILL_BITS 1

// Element types, PTX Table 33 (tensormap.replace .elemtype).
#define TMAP_ET_U8       0
#define TMAP_ET_U16      1
#define TMAP_ET_U32      2
#define TMAP_ET_S32      3
#define TMAP_ET_U64      4
#define TMAP_ET_S64      5
#define TMAP_ET_F16      6
#define TMAP_ET_F32      7
#define TMAP_ET_F32_FTZ  8
#define TMAP_ET_F64      9
#define TMAP_ET_BF16     10
#define TMAP_ET_TF32     11
#define TMAP_ET_TF32_FTZ 12
#define TMAP_ET_COUNT    13

// Fill modes, PTX Table 33 (.fill_mode).
#define TMAP_FILL_ZERO 0
#define TMAP_FILL_NAN  1

// tmap_red_type() of an element type a tensor reduction cannot take.
#define TMAP_RT_NONE 0xFFu

namespace vortex {
namespace dxa_tmap {

inline uint32_t field(uint32_t w, uint32_t lsb, uint32_t bits) {
  return (w >> lsb) & ((1u << bits) - 1u);
}

inline uint32_t set_field(uint32_t w, uint32_t lsb, uint32_t bits, uint32_t v) {
  const uint32_t m = ((1u << bits) - 1u) << lsb;
  return (w & ~m) | ((v << lsb) & m);
}

// Bytes of an element type (1 for codes past the table: tensormap.replace has no range check).
inline uint32_t elem_bytes(uint32_t et) {
  switch (et) {
  case TMAP_ET_U16: case TMAP_ET_F16: case TMAP_ET_BF16:
    return 2;
  case TMAP_ET_U32: case TMAP_ET_S32: case TMAP_ET_F32: case TMAP_ET_F32_FTZ: case TMAP_ET_TF32:
  case TMAP_ET_TF32_FTZ:
    return 4;
  case TMAP_ET_U64: case TMAP_ET_S64: case TMAP_ET_F64:
    return 8;
  default:
    return 1;
  }
}

inline bool elem_float(uint32_t et) {
  return et >= TMAP_ET_F16 && et <= TMAP_ET_TF32_FTZ;
}

// META's element-size field: log2 of the bytes (1, 2, 4, 8).
inline uint32_t elem_log2(uint32_t bytes) {
  return bytes >= 8 ? 3 : bytes >= 4 ? 2 : bytes >= 2 ? 1 : 0;
}

// CFILL: zero, or the OOB-NaN word (0x7FFF7FFF for 2-byte elements, 0x7FFFFFFF otherwise).
inline uint32_t cfill(uint32_t fill_mode, uint32_t bytes) {
  return !fill_mode ? 0u : (bytes == 2 ? 0x7FFF7FFFu : 0x7FFFFFFFu);
}

// DXA_RT_* of a map-form tensor reduction, or TMAP_RT_NONE.
inline uint32_t red_type(uint32_t et) {
  switch (et) {
  case TMAP_ET_U32:     return DXA_RT_U32;
  case TMAP_ET_S32:     return DXA_RT_S32;
  case TMAP_ET_U64:     return DXA_RT_U64;
  case TMAP_ET_S64:     return DXA_RT_S64;
  case TMAP_ET_F16:     return DXA_RT_F16;
  case TMAP_ET_BF16:    return DXA_RT_BF16;
  case TMAP_ET_F32:
  case TMAP_ET_F32_FTZ: return DXA_RT_F32;
  case TMAP_ET_F64:     return DXA_RT_F64;
  default:              return TMAP_RT_NONE;
  }
}

inline uint32_t pack_meta(uint32_t rank, uint32_t bytes, uint32_t layout, uint32_t swizzle,
                          uint32_t interleave, uint32_t l2promo, bool im2col) {
  uint32_t m = 0;
  m = set_field(m, DXA_DESC_META_DIM_LSB, DXA_DESC_META_DIM_BITS, rank);
  m = set_field(m, DXA_DESC_META_ELEMSZ_LSB, DXA_DESC_META_ELEMSZ_BITS, elem_log2(bytes));
  m = set_field(m, DXA_DESC_META_LAYOUT_LSB, DXA_DESC_META_LAYOUT_BITS, layout);
  m = set_field(m, DXA_DESC_META_SWIZZLE_LSB, DXA_DESC_META_SWIZZLE_BITS, swizzle);
  m = set_field(m, DXA_DESC_META_INTERLEAVE_LSB, DXA_DESC_META_INTERLEAVE_BITS, interleave);
  m = set_field(m, DXA_DESC_META_L2PROMO_LSB, DXA_DESC_META_L2PROMO_BITS, l2promo);
  m = set_field(m, DXA_DESC_META_IM2COL_LSB, DXA_DESC_META_IM2COL_BITS, im2col ? 1u : 0u);
  return m;
}

inline uint32_t pack_2x16(uint32_t lo, uint32_t hi) {
  return ((hi & 0xFFFFu) << 16) | (lo & 0xFFFFu);
}

// TILESIZE23 in im2col mode: S = rank - 2 signed corners of 16, 8 or 5 bits, lower corners first.
inline uint32_t pack_corners(uint32_t rank, const int32_t* lower, const int32_t* upper) {
  const uint32_t S = rank - 2;
  const uint32_t bits = S == 1 ? 16 : S == 2 ? 8 : 5;
  const uint32_t m = (1u << bits) - 1u;
  uint32_t w = 0;
  for (uint32_t s = 0; s < S; ++s) {
    w |= ((uint32_t)lower[s] & m) << (s * bits);
    w |= ((uint32_t)upper[s] & m) << ((S + s) * bits);
  }
  return w;
}

inline uint32_t pack_type(uint32_t et, uint32_t fill_mode) {
  return set_field(set_field(0u, TMAP_TYPE_ELEM_LSB, TMAP_TYPE_ELEM_BITS, et), TMAP_TYPE_FILL_LSB,
                   TMAP_TYPE_FILL_BITS, fill_mode);
}

inline uint32_t elem_type(const uint32_t* w) {
  return field(w[TMAP_W_TYPE], TMAP_TYPE_ELEM_LSB, TMAP_TYPE_ELEM_BITS);
}

inline uint32_t fill_mode(const uint32_t* w) {
  return field(w[TMAP_W_TYPE], TMAP_TYPE_FILL_LSB, TMAP_TYPE_FILL_BITS);
}

inline uint32_t rank(const uint32_t* w) {
  return field(w[TMAP_W_META], DXA_DESC_META_DIM_LSB, DXA_DESC_META_DIM_BITS);
}

inline uint64_t global_stride(const uint32_t* w, uint32_t ord) {
  return ((uint64_t)((w[TMAP_W_STRIDE_HI] >> (8 * ord)) & 0xFFu) << 32) | w[TMAP_W_STRIDE0 + ord];
}

///////////////////////////////////////////////////////////////////////////////
// tensormap.replace.tile (PTX §9.7.9.26): one field, as a read-modify-write
// of the words it touches with ordinary loads and stores. Operand meanings
// are PTX's; no range check (an invalid value is undefined).

inline void replace_global_address(uint32_t* w, uint64_t addr) {
  w[TMAP_W_BASE_LO] = (uint32_t)addr;
  w[TMAP_W_BASE_HI] = (uint32_t)(addr >> 32);
}

inline void replace_rank(uint32_t* w, uint32_t rank_minus_1) {
  w[TMAP_W_META] = set_field(w[TMAP_W_META], DXA_DESC_META_DIM_LSB, DXA_DESC_META_DIM_BITS, rank_minus_1 + 1);
}

inline void replace_box_dim(uint32_t* w, uint32_t ord, uint32_t v) {
  if (ord == 4) {
    w[TMAP_W_TILESIZE4] = v;
    return;
  }
  const uint32_t i = TMAP_W_TILESIZE01 + ord / 2;
  w[i] = set_field(w[i], 16 * (ord & 1), 16, v);
}

inline void replace_global_dim(uint32_t* w, uint32_t ord, uint32_t v) {
  w[TMAP_W_SIZE0 + ord] = v;
}

// ord i = the stride of dim i + 1.
inline void replace_global_stride(uint32_t* w, uint32_t ord, uint64_t v) {
  w[TMAP_W_STRIDE0 + ord] = (uint32_t)v;
  w[TMAP_W_STRIDE_HI] = set_field(w[TMAP_W_STRIDE_HI], 8 * ord, 8, (uint32_t)(v >> 32));
}

inline void replace_element_stride(uint32_t* w, uint32_t ord, uint32_t v) {
  w[TMAP_W_ESTRIDE0 + ord] = v;
}

// Also sets META's element size and recomputes CFILL.
inline void replace_elemtype(uint32_t* w, uint32_t et) {
  const uint32_t t = set_field(w[TMAP_W_TYPE], TMAP_TYPE_ELEM_LSB, TMAP_TYPE_ELEM_BITS, et);
  w[TMAP_W_TYPE] = t;
  w[TMAP_W_META] = set_field(w[TMAP_W_META], DXA_DESC_META_ELEMSZ_LSB, DXA_DESC_META_ELEMSZ_BITS,
                             elem_log2(elem_bytes(et)));
  w[TMAP_W_CFILL] = cfill(field(t, TMAP_TYPE_FILL_LSB, TMAP_TYPE_FILL_BITS), elem_bytes(et));
}

inline void replace_interleave_layout(uint32_t* w, uint32_t v) {
  w[TMAP_W_META] = set_field(w[TMAP_W_META], DXA_DESC_META_INTERLEAVE_LSB, DXA_DESC_META_INTERLEAVE_BITS, v);
}

inline void replace_swizzle_mode(uint32_t* w, uint32_t v) {
  w[TMAP_W_META] = set_field(w[TMAP_W_META], DXA_DESC_META_SWIZZLE_LSB, DXA_DESC_META_SWIZZLE_BITS, v);
}

// Recomputes CFILL.
inline void replace_fill_mode(uint32_t* w, uint32_t v) {
  const uint32_t t = set_field(w[TMAP_W_TYPE], TMAP_TYPE_FILL_LSB, TMAP_TYPE_FILL_BITS, v);
  w[TMAP_W_TYPE] = t;
  w[TMAP_W_CFILL] = cfill(field(t, TMAP_TYPE_FILL_LSB, TMAP_TYPE_FILL_BITS),
                          elem_bytes(field(t, TMAP_TYPE_ELEM_LSB, TMAP_TYPE_ELEM_BITS)));
}

} // namespace dxa_tmap
} // namespace vortex
