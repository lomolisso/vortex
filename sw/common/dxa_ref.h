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

// Golden model of the DXA tensor and bulk loads with mbarriers (PTX
// cp.async.bulk.tensor / cp.async.bulk .global sources, PTX ISA §5.5): the box
// and element mapping with signed coordinates, element strides and
// per-element out-of-bounds fill in every dimension, interleave, the swizzle
// address function, the im2col pixel walk, the complete_tx byte count, the
// fill byte rule, the descriptor and operand encodings and the asserts; and
// the reverse direction (tensor and bulk stores, global and cluster
// reductions, bulk groups, proxy fences). Host code, SimX and the RTL unit
// bench share it; plain functions over Desc + Req.

#include <cstdint>
#include <cstddef>
#include <VX_types.h>
#include "dxa_meta.h"
#include "dxa_tmap.h"

namespace vortex {
namespace dxa_ref {

enum : uint32_t { LAYOUT_ROW = 0, LAYOUT_KMAJOR = 1, LAYOUT_FLAT = 2, LAYOUT_BLOCKMAJOR = 3 };
enum : uint32_t { SWZ_NONE = 0, SWZ_32B = 1, SWZ_64B = 2, SWZ_128B = 3 };
enum : uint32_t { IL_NONE = 0, IL_16B = 1, IL_32B = 2, IL_INVALID = 3 };

constexpr uint32_t MAX_RANK   = 5;
constexpr uint32_t DESC_WORDS = VX_DCR_DXA_DESC_STRIDE;

// funct3 of the DXA instruction (custom-0, funct7 = 3).
enum : uint32_t { F3_TENSOR = 0, F3_BULK = 1, F3_TLOAD = 2, F3_BLOAD = 3, F3_BPREFETCH = 4 };

// TLOAD ctl word (rs2 lane 0, bits [63:32]).
constexpr uint32_t CTL_SLOT_MASK = 0xFu;
constexpr uint32_t CTL_PREFETCH  = 1u << 4;

// Assert texts, identical in the RTL RUNTIME_ASSERT and SimX.
constexpr const char* MSG_MBAR_ALIGN   = "DXA mbarrier must be 8-byte aligned";
constexpr const char* MSG_BULK_SELF    = "DXA bulk copy into the issuing block";
constexpr const char* MSG_SWZ_ALIGN    = "DXA swizzle needs a 128-byte aligned destination";
constexpr const char* MSG_SWZ_SPAN     = "DXA swizzle span smaller than the inner box";
constexpr const char* MSG_ES0          = "DXA element stride on dimension 0 needs interleave";
constexpr const char* MSG_IL32         = "DXA 32B interleave needs 32B swizzle";
constexpr const char* MSG_IL_RANK      = "DXA interleave needs rank 3 to 5";
constexpr const char* MSG_I2C_RANK     = "DXA im2col needs rank 3 to 5";
constexpr const char* MSG_LAYOUT       = "DXA swizzle and element strides need the row-major layout";
constexpr const char* MSG_IL_INVALID   = "DXA interleave mode 3 is invalid";
constexpr const char* MSG_BULK_ALIGN   = "DXA bulk copy needs 16-byte size and alignment";

// One descriptor, decoded. Sizes are elements, strides bytes.
struct Desc {
  uint64_t base = 0;
  uint32_t rank = 1;                 // 1..5
  uint32_t elem = 4;                 // bytes: 1, 2, 4, 8
  uint32_t layout = LAYOUT_ROW;
  uint32_t swizzle = SWZ_NONE;
  uint32_t interleave = IL_NONE;
  bool     im2col = false;
  uint32_t size[MAX_RANK] = {0, 0, 0, 0, 0};
  uint32_t stride[MAX_RANK - 1] = {0, 0, 0, 0};  // stride[d-1] steps dimension d
  uint32_t estride[MAX_RANK] = {1, 1, 1, 1, 1};  // 1..8; 0 is read as 1
  uint32_t tile[MAX_RANK] = {0, 0, 0, 0, 0};     // tile mode box
  uint32_t channels = 0;             // im2col channelsPerPixel, 1..256
  uint32_t pixels = 0;               // im2col pixelsPerColumn, 1..1024
  int32_t  lower[3] = {0, 0, 0};     // im2col pixel-box corners, W H D
  int32_t  upper[3] = {0, 0, 0};
  uint32_t cfill = 0;
  uint32_t tc_geom = 0;              // Flat/BlockMajor: raw ESTRIDE2 (tcN[15:0], blk_k[31:16])
};

// One request: signed coordinates in dimension order, im2col offsets W H D.
struct Req {
  int32_t  coord[MAX_RANK] = {0, 0, 0, 0, 0};
  uint32_t offs[3] = {0, 0, 0};
};

inline uint32_t elem_log2(uint32_t elem) {
  uint32_t n = 0;
  while ((1u << n) < elem)
    ++n;
  return n;
}

inline uint32_t sext(uint32_t v, uint32_t bits) {
  const uint32_t m = 1u << (bits - 1);
  v &= (bits >= 32) ? ~0u : ((1u << bits) - 1u);
  return (v ^ m) - m;
}

inline uint32_t field(uint32_t w, uint32_t lsb, uint32_t bits) {
  return (w >> lsb) & ((1u << bits) - 1u);
}

// The OOB-NaN fill word: the element type's NaN replicated to 32 bits.
inline uint32_t nan_fill(uint32_t elem) {
  return elem == 2 ? 0x7FFF7FFFu : 0x7FFFFFFFu;
}

inline uint32_t es(const Desc& d, uint32_t dim) {
  if (d.layout == LAYOUT_FLAT || d.layout == LAYOUT_BLOCKMAJOR)
    return 1;
  return d.estride[dim] ? d.estride[dim] : 1;
}

// Elements along dim in tile mode: ceil(tile / es).
inline uint32_t n_elems(const Desc& d, uint32_t dim) {
  if (dim >= d.rank)
    return 1;
  const uint32_t s = es(d, dim);
  return (d.tile[dim] + s - 1) / s;
}

inline uint32_t spatial_dims(const Desc& d) {
  return d.rank - 2;
}

// complete_tx bytes, fill included, unchanged by the swizzle.
inline uint32_t tx_bytes(const Desc& d) {
  if (d.im2col)
    return d.pixels * d.channels * d.elem;
  uint32_t n = d.elem;
  for (uint32_t i = 0; i < d.rank; ++i)
    n *= n_elems(d, i);
  return n;
}

inline uint32_t swizzle_span(uint32_t swz) {
  return swz == SWZ_32B ? 32 : swz == SWZ_64B ? 64 : swz == SWZ_128B ? 128 : 0;
}

inline uint32_t swizzle_mask(uint32_t swz) {
  return swz == SWZ_32B ? 1 : swz == SWZ_64B ? 3 : swz == SWZ_128B ? 7 : 0;
}

// The PTX §5.5.7 layout: the 16-byte chunk at block-relative offset r moves
// to chunk ^ ((r / 128) & M) within its 128-byte row. Bytes keep their offset
// in the chunk, so the function applies per byte too.
inline uint32_t swizzle(uint32_t r, uint32_t swz) {
  return r ^ (((r >> 7) & swizzle_mask(swz)) << 4);
}

// The inner box bytes the swizzle span must cover.
inline uint32_t inner_bytes(const Desc& d) {
  if (d.im2col)
    return d.channels * d.elem;
  if (d.interleave == IL_16B)
    return 16;
  if (d.interleave == IL_32B)
    return 32;
  return n_elems(d, 0) * d.elem;
}

// The OOB byte at destination offset a.
inline uint8_t fill_byte(uint32_t cfill, uint32_t a) {
  return (uint8_t)(cfill >> (8 * (a & 3)));
}

// The first assert a tensor request trips, or nullptr; dst_rel is the
// block-relative destination offset (ignored by a prefetch: pass 0).
inline const char* check_tensor(const Desc& d, uint32_t dst_rel, bool prefetch = false) {
  if (d.im2col && (d.rank < 3 || d.rank > 5))
    return MSG_I2C_RANK;
  if (d.layout != LAYOUT_ROW) {
    bool strided = false;
    for (uint32_t i = 0; i < d.rank; ++i)
      strided |= (es(d, i) != 1);
    if (d.swizzle != SWZ_NONE || strided)
      return MSG_LAYOUT;
  }
  if (d.interleave == IL_INVALID)
    return MSG_IL_INVALID;
  if (d.interleave != IL_NONE && (d.rank < 3 || d.rank > 5))
    return MSG_IL_RANK;
  if (d.interleave == IL_32B && d.swizzle != SWZ_32B)
    return MSG_IL32;
  if (es(d, 0) > 1 && (d.interleave == IL_NONE || d.im2col))
    return MSG_ES0;
  if (d.swizzle != SWZ_NONE) {
    if (inner_bytes(d) > swizzle_span(d.swizzle))
      return MSG_SWZ_SPAN;
    if (!prefetch && (dst_rel & 127u) != 0)
      return MSG_SWZ_ALIGN;
  }
  return nullptr;
}

inline bool mbar_aligned(uint64_t mbar) {
  return (mbar & 7u) == 0;
}

// BLOAD / BPREFETCH: size a nonzero multiple of 16, both addresses 16-B aligned.
inline const char* check_bulk(uint64_t dst, uint64_t gsrc, uint32_t size) {
  if (size == 0 || (size & 15u) || (gsrc & 15u) || (dst & 15u))
    return MSG_BULK_ALIGN;
  return nullptr;
}

// The im2col pixel walk: f(k, x, pix_in) for pixel k = 0 .. pixels-1, x = its
// tensor coordinates with x[0] = the first channel, pix_in = its spatial and
// image coordinates are inside the tensor.
// Rules read from the PTX §5.5.4 figures, where the text leaves them open:
// the examples' "tensor coordinates (7, 7, 4, 0)" are (n, h, w, c);
// the offsets shift the whole pixel box: pixel = p + o, p in [lower, size - 1 + upper];
// a carry out of the last spatial dim moves to image n + 1, n >= size_N is filled.
template <typename F>
void im2col_pixels(const Desc& d, const Req& q, F f) {
  const uint32_t S = spatial_dims(d), R = d.rank;
  int64_t p[3] = {0, 0, 0}, lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
  for (uint32_t s = 0; s < S; ++s) {
    p[s]  = q.coord[s + 1];
    lo[s] = d.lower[s];
    hi[s] = (int64_t)d.size[s + 1] - 1 + d.upper[s];
  }
  int64_t n = q.coord[R - 1];
  for (uint32_t k = 0; k < d.pixels; ++k) {
    int64_t x[MAX_RANK] = {q.coord[0], 0, 0, 0, 0};
    bool pix_in = (n >= 0 && n < (int64_t)d.size[R - 1]);
    for (uint32_t s = 0; s < S; ++s) {
      x[s + 1] = p[s] + (int64_t)q.offs[s];
      pix_in &= (x[s + 1] >= 0 && x[s + 1] < (int64_t)d.size[s + 1]);
    }
    x[R - 1] = n;
    f(k, x, pix_in);
    // W fastest; a wrap restarts the dim at its lower corner and carries.
    uint32_t s = 0;
    for (; s < S; ++s) {
      p[s] += es(d, s + 1);
      if (p[s] <= hi[s])
        break;
      p[s] = lo[s];
    }
    if (s == S)
      ++n;
  }
}

// Visits every element of the box in destination order:
// f(dense_off, in_bounds, gaddr), dense_off = the element's byte offset in
// the box before the swizzle, gaddr = its global byte address (in-bounds only).
// Row-major (tile and im2col) and K-major (rank <= 2) layouts.
// The Vortex layouts (K-major, Flat, BlockMajor) keep the tensor issue's
// bounds: dim 0 is never filled, and outer coordinates compare unsigned.
template <typename F>
void for_each_element(const Desc& d, const Req& q, F f) {
  const uint32_t E = d.elem;
  auto gaddr = [&](const int64_t* x) {
    int64_t a = (int64_t)d.base + x[0] * (int64_t)E;
    for (uint32_t i = 1; i < d.rank; ++i)
      a += x[i] * (int64_t)d.stride[i - 1];
    return (uint64_t)a;
  };
  if (d.im2col) {
    im2col_pixels(d, q, [&](uint32_t k, const int64_t* x, bool pix_in) {
      int64_t y[MAX_RANK] = {x[0], x[1], x[2], x[3], x[4]};
      for (uint32_t j = 0; j < d.channels; ++j) {
        y[0] = x[0] + j;
        const bool in = pix_in && y[0] >= 0 && y[0] < (int64_t)d.size[0];
        f((k * d.channels + j) * E, in, in ? gaddr(y) : 0);
      }
    });
    return;
  }
  uint32_t nn[MAX_RANK];
  for (uint32_t i = 0; i < MAX_RANK; ++i)
    nn[i] = n_elems(d, i);
  const bool kmajor = (d.layout == LAYOUT_KMAJOR);
  for (uint32_t i4 = 0; i4 < nn[4]; ++i4)
  for (uint32_t i3 = 0; i3 < nn[3]; ++i3)
  for (uint32_t i2 = 0; i2 < nn[2]; ++i2)
  for (uint32_t i1 = 0; i1 < nn[1]; ++i1)
  for (uint32_t i0 = 0; i0 < nn[0]; ++i0) {
    const uint32_t idx[MAX_RANK] = {i0, i1, i2, i3, i4};
    bool in = true;
    int64_t y[MAX_RANK] = {0, 0, 0, 0, 0};
    for (uint32_t i = 0; i < d.rank; ++i) {
      y[i] = (int64_t)q.coord[i] + (int64_t)idx[i] * es(d, i);
      if (d.layout == LAYOUT_ROW)
        in &= (y[i] >= 0 && y[i] < (int64_t)d.size[i]);
      else if (i > 0)
        in &= ((uint64_t)(uint32_t)q.coord[i] + idx[i] < d.size[i]);
    }
    const uint32_t lin = kmajor
        ? i0 * nn[1] + i1
        : (((i4 * nn[3] + i3) * nn[2] + i2) * nn[1] + i1) * nn[0] + i0;
    f(lin * E, in, in ? gaddr(y) : 0);
  }
}

// Writes the box into img, the destination block's bytes, at block-relative
// offset dst_rel (swizzled per byte), reading global bytes through
// read(addr); fill bytes take CFILL. Returns the complete_tx bytes. Bytes
// past img_size are dropped.
template <typename Read>
uint32_t render(const Desc& d, const Req& q, uint32_t dst_rel, Read read, uint8_t* img, size_t img_size) {
  const uint32_t E = d.elem;
  for_each_element(d, q, [&](uint32_t off, bool in, uint64_t ga) {
    for (uint32_t b = 0; b < E; ++b) {
      const uint32_t a = swizzle(dst_rel + off + b, d.swizzle);
      if (a < img_size)
        img[a] = in ? read(ga + b) : fill_byte(d.cfill, a);
    }
  });
  return tx_bytes(d);
}

// The block-relative byte range [lo, hi) the box writes.
inline void footprint(const Desc& d, uint32_t dst_rel, uint32_t* lo, uint32_t* hi) {
  uint32_t mn = ~0u, mx = 0;
  const Req q;
  for_each_element(d, q, [&](uint32_t off, bool, uint64_t) {
    const uint32_t a = swizzle(dst_rel + off, d.swizzle);
    mn = a < mn ? a : mn;
    mx = a + d.elem > mx ? a + d.elem : mx;
  });
  *lo = (mn == ~0u) ? dst_rel : mn;
  *hi = mx;
}

///////////////////////////////////////////////////////////////////////////////
// Descriptor words (VX_DCR_DXA_DESC_*_OFF) <-> Desc

inline uint32_t pack_corners(const Desc& d) {
  const uint32_t S = spatial_dims(d);
  const uint32_t bits = S == 1 ? 16 : S == 2 ? 8 : 5;
  const uint32_t m = (1u << bits) - 1u;
  uint32_t w = 0;
  for (uint32_t s = 0; s < S; ++s) {
    w |= ((uint32_t)d.lower[s] & m) << (s * bits);
    w |= ((uint32_t)d.upper[s] & m) << ((S + s) * bits);
  }
  return w;
}

inline void unpack_corners(Desc& d, uint32_t w) {
  const uint32_t S = spatial_dims(d);
  const uint32_t bits = S == 1 ? 16 : S == 2 ? 8 : 5;
  for (uint32_t s = 0; s < 3; ++s) {
    d.lower[s] = s < S ? (int32_t)sext(w >> (s * bits), bits) : 0;
    d.upper[s] = s < S ? (int32_t)sext(w >> ((S + s) * bits), bits) : 0;
  }
}

inline uint32_t pack_meta(const Desc& d) {
  return (d.rank << DXA_DESC_META_DIM_LSB)
       | (elem_log2(d.elem) << DXA_DESC_META_ELEMSZ_LSB)
       | (d.layout << DXA_DESC_META_LAYOUT_LSB)
       | (d.swizzle << DXA_DESC_META_SWIZZLE_LSB)
       | (d.interleave << DXA_DESC_META_INTERLEAVE_LSB)
       | ((d.im2col ? 1u : 0u) << DXA_DESC_META_IM2COL_LSB);
}

// w[0 .. DESC_WORDS) in VX_DCR_DXA_DESC_*_OFF order; the multicast stride word is left as is.
inline void encode(const Desc& d, uint32_t* w) {
  w[VX_DCR_DXA_DESC_BASE_LO_OFF] = (uint32_t)d.base;
  w[VX_DCR_DXA_DESC_BASE_HI_OFF] = (uint32_t)(d.base >> 32);
  w[VX_DCR_DXA_DESC_SIZE0_OFF] = d.size[0];
  w[VX_DCR_DXA_DESC_SIZE1_OFF] = d.size[1];
  w[VX_DCR_DXA_DESC_SIZE2_OFF] = d.size[2];
  w[VX_DCR_DXA_DESC_SIZE3_OFF] = d.size[3];
  w[VX_DCR_DXA_DESC_SIZE4_OFF] = d.size[4];
  w[VX_DCR_DXA_DESC_STRIDE0_OFF] = d.stride[0];
  w[VX_DCR_DXA_DESC_STRIDE1_OFF] = d.stride[1];
  w[VX_DCR_DXA_DESC_STRIDE2_OFF] = d.stride[2];
  w[VX_DCR_DXA_DESC_STRIDE3_OFF] = d.stride[3];
  w[VX_DCR_DXA_DESC_META_OFF] = pack_meta(d);
  w[VX_DCR_DXA_DESC_ESTRIDE0_OFF] = d.estride[0];
  w[VX_DCR_DXA_DESC_ESTRIDE1_OFF] = d.estride[1];
  w[VX_DCR_DXA_DESC_ESTRIDE2_OFF] = (d.layout == LAYOUT_FLAT || d.layout == LAYOUT_BLOCKMAJOR)
                                    ? d.tc_geom : d.estride[2];
  w[VX_DCR_DXA_DESC_ESTRIDE3_OFF] = d.estride[3];
  w[VX_DCR_DXA_DESC_ESTRIDE4_OFF] = d.estride[4];
  if (d.im2col) {
    w[VX_DCR_DXA_DESC_TILESIZE01_OFF] = ((d.pixels & 0xFFFFu) << 16) | (d.channels & 0xFFFFu);
    w[VX_DCR_DXA_DESC_TILESIZE23_OFF] = pack_corners(d);
    w[VX_DCR_DXA_DESC_TILESIZE4_OFF]  = 0;
  } else {
    w[VX_DCR_DXA_DESC_TILESIZE01_OFF] = ((d.tile[1] & 0xFFFFu) << 16) | (d.tile[0] & 0xFFFFu);
    w[VX_DCR_DXA_DESC_TILESIZE23_OFF] = ((d.tile[3] & 0xFFFFu) << 16) | (d.tile[2] & 0xFFFFu);
    w[VX_DCR_DXA_DESC_TILESIZE4_OFF]  = d.tile[4];
  }
  w[VX_DCR_DXA_DESC_CFILL_OFF] = d.cfill;
}

inline Desc decode(const uint32_t* w) {
  Desc d;
  const uint32_t meta = w[VX_DCR_DXA_DESC_META_OFF];
  d.base = ((uint64_t)w[VX_DCR_DXA_DESC_BASE_HI_OFF] << 32) | w[VX_DCR_DXA_DESC_BASE_LO_OFF];
  d.rank = field(meta, DXA_DESC_META_DIM_LSB, DXA_DESC_META_DIM_BITS);
  d.elem = 1u << field(meta, DXA_DESC_META_ELEMSZ_LSB, DXA_DESC_META_ELEMSZ_BITS);
  d.layout = field(meta, DXA_DESC_META_LAYOUT_LSB, DXA_DESC_META_LAYOUT_BITS);
  d.swizzle = field(meta, DXA_DESC_META_SWIZZLE_LSB, DXA_DESC_META_SWIZZLE_BITS);
  d.interleave = field(meta, DXA_DESC_META_INTERLEAVE_LSB, DXA_DESC_META_INTERLEAVE_BITS);
  d.im2col = field(meta, DXA_DESC_META_IM2COL_LSB, DXA_DESC_META_IM2COL_BITS) != 0;
  const uint32_t sz[MAX_RANK] = {VX_DCR_DXA_DESC_SIZE0_OFF, VX_DCR_DXA_DESC_SIZE1_OFF, VX_DCR_DXA_DESC_SIZE2_OFF,
                                 VX_DCR_DXA_DESC_SIZE3_OFF, VX_DCR_DXA_DESC_SIZE4_OFF};
  const uint32_t st[MAX_RANK - 1] = {VX_DCR_DXA_DESC_STRIDE0_OFF, VX_DCR_DXA_DESC_STRIDE1_OFF,
                                     VX_DCR_DXA_DESC_STRIDE2_OFF, VX_DCR_DXA_DESC_STRIDE3_OFF};
  const uint32_t ew[MAX_RANK] = {VX_DCR_DXA_DESC_ESTRIDE0_OFF, VX_DCR_DXA_DESC_ESTRIDE1_OFF,
                                 VX_DCR_DXA_DESC_ESTRIDE2_OFF, VX_DCR_DXA_DESC_ESTRIDE3_OFF,
                                 VX_DCR_DXA_DESC_ESTRIDE4_OFF};
  for (uint32_t i = 0; i < MAX_RANK; ++i) {
    d.size[i] = w[sz[i]];
    d.estride[i] = w[ew[i]] ? w[ew[i]] : 1;
  }
  for (uint32_t i = 0; i + 1 < MAX_RANK; ++i)
    d.stride[i] = w[st[i]];
  if (d.layout == LAYOUT_FLAT || d.layout == LAYOUT_BLOCKMAJOR) {
    d.tc_geom = w[VX_DCR_DXA_DESC_ESTRIDE2_OFF];
    for (uint32_t i = 0; i < MAX_RANK; ++i)
      d.estride[i] = 1;
  }
  const uint32_t t01 = w[VX_DCR_DXA_DESC_TILESIZE01_OFF];
  const uint32_t t23 = w[VX_DCR_DXA_DESC_TILESIZE23_OFF];
  if (d.im2col) {
    d.channels = t01 & 0xFFFFu;
    d.pixels   = t01 >> 16;
    if (d.rank >= 3 && d.rank <= 5)
      unpack_corners(d, t23);
  } else {
    d.tile[0] = t01 & 0xFFFFu;
    d.tile[1] = t01 >> 16;
    d.tile[2] = t23 & 0xFFFFu;
    d.tile[3] = t23 >> 16;
    d.tile[4] = w[VX_DCR_DXA_DESC_TILESIZE4_OFF];
  }
  d.cfill = w[VX_DCR_DXA_DESC_CFILL_OFF];
  return d;
}

///////////////////////////////////////////////////////////////////////////////
// Instruction operands (XLEN 64, lanes 0-3 of rs1 and rs2)

struct Ops {
  uint64_t rs1[4] = {0, 0, 0, 0};
  uint64_t rs2[4] = {0, 0, 0, 0};
};

inline uint64_t pair(int32_t lo, int32_t hi) {
  return ((uint64_t)(uint32_t)hi << 32) | (uint32_t)lo;
}

// funct3 = 2: rs1 = {dst, mbar, {c1, c0}, {c3, c2}},
// rs2 = {{ctl, c4}, {o3, o2, o1, ctaMask}, 0, 0}.
inline Ops pack_tload(uint32_t slot, bool prefetch, uint64_t dst, uint64_t mbar, const Req& q, uint16_t cta_mask) {
  Ops o;
  const uint32_t ctl = (slot & CTL_SLOT_MASK) | (prefetch ? CTL_PREFETCH : 0u);
  o.rs1[0] = dst;
  o.rs1[1] = mbar;
  o.rs1[2] = pair(q.coord[0], q.coord[1]);
  o.rs1[3] = pair(q.coord[2], q.coord[3]);
  o.rs2[0] = ((uint64_t)ctl << 32) | (uint32_t)q.coord[4];
  o.rs2[1] = ((uint64_t)(q.offs[2] & 0xFFFFu) << 48) | ((uint64_t)(q.offs[1] & 0xFFFFu) << 32)
           | ((uint64_t)(q.offs[0] & 0xFFFFu) << 16) | cta_mask;
  return o;
}

inline void unpack_tload(const Ops& o, uint32_t* slot, bool* prefetch, uint64_t* dst, uint64_t* mbar,
                         Req* q, uint16_t* cta_mask) {
  const uint32_t ctl = (uint32_t)(o.rs2[0] >> 32);
  *slot = ctl & CTL_SLOT_MASK;
  *prefetch = (ctl & CTL_PREFETCH) != 0;
  *dst = o.rs1[0];
  *mbar = o.rs1[1];
  q->coord[0] = (int32_t)(uint32_t)o.rs1[2];
  q->coord[1] = (int32_t)(uint32_t)(o.rs1[2] >> 32);
  q->coord[2] = (int32_t)(uint32_t)o.rs1[3];
  q->coord[3] = (int32_t)(uint32_t)(o.rs1[3] >> 32);
  q->coord[4] = (int32_t)(uint32_t)o.rs2[0];
  q->offs[0] = (uint32_t)(o.rs2[1] >> 16) & 0xFFFFu;
  q->offs[1] = (uint32_t)(o.rs2[1] >> 32) & 0xFFFFu;
  q->offs[2] = (uint32_t)(o.rs2[1] >> 48) & 0xFFFFu;
  *cta_mask = (uint16_t)o.rs2[1];
}

// funct3 = 3: rs1 = {dst, mbar, gsrc, size}, rs2 lane 1 = ctaMask (rs2 = x0 without multicast).
// funct3 = 4: rs1 = {0, 0, gsrc, size}, rs2 = x0.
inline Ops pack_bload(uint64_t dst, uint64_t mbar, uint64_t gsrc, uint32_t size, uint16_t cta_mask) {
  Ops o;
  o.rs1[0] = dst;
  o.rs1[1] = mbar;
  o.rs1[2] = gsrc;
  o.rs1[3] = size;
  o.rs2[1] = cta_mask;
  return o;
}


///////////////////////////////////////////////////////////////////////////////
// Stores, reductions, bulk groups and proxy fences (funct3 5 and 6; PTX
// cp.async.bulk{,.tensor}.global.shared::cta, cp.reduce.async.bulk{,.tensor},
// cp.async.bulk.commit_group / wait_group, fence.proxy.async): the walk above
// in the LMEM -> global direction. The reduction semantics need amo_ref.h,
// hence softfloat (see amo_ref.h); the rest of this part does not.

enum : uint32_t { F3_STORE = DXA_F3_STORE, F3_BGOP = DXA_F3_BGOP };

enum : uint32_t {
  ST_BULK = DXA_ST_BULK, ST_TENSOR = DXA_ST_TENSOR, ST_RED_GLOBAL = DXA_ST_RED_GLOBAL,
  ST_RED_CLUSTER = DXA_ST_RED_CLUSTER, ST_RED_TENSOR = DXA_ST_RED_TENSOR, ST_KINDS = 5
};
enum : uint32_t {
  RED_ADD = DXA_RED_ADD, RED_MIN = DXA_RED_MIN, RED_MAX = DXA_RED_MAX, RED_INC = DXA_RED_INC,
  RED_DEC = DXA_RED_DEC, RED_AND = DXA_RED_AND, RED_OR = DXA_RED_OR, RED_XOR = DXA_RED_XOR, RED_OPS = 8
};
enum : uint32_t {
  RT_U32 = DXA_RT_U32, RT_S32 = DXA_RT_S32, RT_U64 = DXA_RT_U64, RT_S64 = DXA_RT_S64,
  RT_F32 = DXA_RT_F32, RT_F64 = DXA_RT_F64, RT_F16 = DXA_RT_F16, RT_BF16 = DXA_RT_BF16, RT_TYPES = 8
};
enum : uint32_t { BG_COMMIT = DXA_BG_COMMIT, BG_WAIT = DXA_BG_WAIT, BG_WAIT_READ = DXA_BG_WAIT_READ };
enum : uint32_t { PFENCE_SHARED = DXA_PFENCE_SHARED, PFENCE_GLOBAL = DXA_PFENCE_GLOBAL, PFENCE_ALL = DXA_PFENCE_ALL };
constexpr uint32_t BG_GROUPS  = DXA_BG_GROUPS;
constexpr uint32_t BG_MAX_OPS = DXA_BG_MAX_OPS;

constexpr const char* MSG_ST_NEG       = "DXA tensor store starts below the tensor";
constexpr const char* MSG_RED_LLC      = "DXA global reduction needs an L2 or L3";
constexpr const char* MSG_RED_FPAMO    = "DXA reduction needs FPAMO";
constexpr const char* MSG_RED_INVALID  = "DXA invalid reduction";
constexpr const char* MSG_RED_SIZE     = "DXA reduction type does not match the element size";
constexpr const char* MSG_ENCODING     = "invalid DXA encoding";
constexpr const char* MSG_PFENCE       = "invalid proxy fence";
constexpr const char* MSG_BULK_SRC     = "DXA bulk copy source outside the block";
constexpr const char* MSG_ST_LAYOUT    = "DXA store needs a tile or im2col layout";

inline const char* red_op_name(uint32_t op) {
  static const char* const n[] = {"add", "min", "max", "inc", "dec", "and", "or", "xor"};
  return op < RED_OPS ? n[op] : "?";
}

// PTX names; and/or/xor print as b32/b64.
inline const char* red_type_name(uint32_t type, uint32_t op = RED_ADD) {
  static const char* const n[] = {"u32", "s32", "u64", "s64", "f32", "f64", "f16", "bf16"};
  const bool bits = (op == RED_AND || op == RED_OR || op == RED_XOR);
  if (bits && type == RT_U32) return "b32";
  if (bits && type == RT_U64) return "b64";
  return type < RT_TYPES ? n[type] : "?";
}

inline uint32_t type_bytes(uint32_t type) {
  switch (type) {
  case RT_U64: case RT_S64: case RT_F64: return 8;
  case RT_F16: case RT_BF16:             return 2;
  default:                               return 4;
  }
}

inline bool type_float(uint32_t type) {
  return type == RT_F32 || type == RT_F64 || type == RT_F16 || type == RT_BF16;
}

inline bool is_reduce(uint32_t kind) {
  return kind == ST_RED_GLOBAL || kind == ST_RED_CLUSTER || kind == ST_RED_TENSOR;
}

inline bool is_tensor(uint32_t kind) {
  return kind == ST_TENSOR || kind == ST_RED_TENSOR;
}

// The bulk and tensor stores and the global and tensor reductions join the
// warp's open group; the cluster reduction does not.
inline bool is_bulk_group(uint32_t kind) {
  return kind < ST_KINDS && kind != ST_RED_CLUSTER;
}

// The legal op x type pairs of the global and tensor reductions and of the
// cluster reductions. and/or/xor take type 0 (b32) or 2 (b64).
inline bool red_legal(uint32_t kind, uint32_t op, uint32_t type) {
  if (kind == ST_RED_CLUSTER) {
    switch (op) {
    case RED_AND: case RED_OR: case RED_XOR: return type == RT_U32;
    case RED_ADD:                            return type == RT_U32 || type == RT_S32 || type == RT_U64;
    case RED_MIN: case RED_MAX:              return type == RT_U32 || type == RT_S32;
    case RED_INC: case RED_DEC:              return type == RT_U32;
    default:                                 return false;
    }
  }
  switch (op) {
  case RED_AND: case RED_OR: case RED_XOR:
    return type == RT_U32 || type == RT_U64;
  case RED_MIN: case RED_MAX:
    return type == RT_U32 || type == RT_S32 || type == RT_U64 || type == RT_S64 || type == RT_F16
        || type == RT_BF16;
  case RED_ADD:
    return type == RT_U32 || type == RT_S32 || type == RT_U64 || type == RT_F32 || type == RT_F64
        || type == RT_F16 || type == RT_BF16;
  case RED_INC: case RED_DEC:
    return type == RT_U32;
  default:
    return false;
  }
}

inline bool red_needs_fpamo(uint32_t op, uint32_t type) {
  return type_float(type) || op == RED_INC || op == RED_DEC;
}

// The atomic each request of a reduction runs (VX AMO_OP_* = amo_ref::Op codes),
// on 32-bit words (4- and 2-byte types; 16-bit types as packed pairs) or
// 64-bit words (8-byte types). is_unsigned selects the unsigned min/max.
struct RedAmo {
  uint32_t amo_op;
  uint32_t width;       // amo_ref::Width: 2 = W, 3 = D
  bool     is_unsigned;
};

enum : uint32_t {
  AMO_ADD = 0x02, AMO_XOR = 0x04, AMO_OR = 0x05, AMO_AND = 0x06, AMO_MIN = 0x07, AMO_MAX = 0x08,
  AMO_INC = 0x10, AMO_DEC = 0x11, AMO_FADD = 0x12, AMO_FADD2 = 0x14, AMO_FADD2_BF = 0x15,
  AMO_FMIN2 = 0x16, AMO_FMAX2 = 0x17, AMO_FMIN2_BF = 0x18, AMO_FMAX2_BF = 0x19
};

inline RedAmo red_amo(uint32_t op, uint32_t type) {
  RedAmo a;
  a.width = type_bytes(type) == 8 ? 3 : 2;
  a.is_unsigned = (type == RT_U32 || type == RT_U64);
  switch (op) {
  case RED_ADD:
    a.amo_op = type == RT_F32 || type == RT_F64 ? AMO_FADD : type == RT_F16 ? AMO_FADD2
             : type == RT_BF16 ? AMO_FADD2_BF : AMO_ADD;
    break;
  case RED_MIN:
    a.amo_op = type == RT_F16 ? AMO_FMIN2 : type == RT_BF16 ? AMO_FMIN2_BF : AMO_MIN;
    break;
  case RED_MAX:
    a.amo_op = type == RT_F16 ? AMO_FMAX2 : type == RT_BF16 ? AMO_FMAX2_BF : AMO_MAX;
    break;
  case RED_INC: a.amo_op = AMO_INC; break;
  case RED_DEC: a.amo_op = AMO_DEC; break;
  case RED_AND: a.amo_op = AMO_AND; break;
  case RED_OR:  a.amo_op = AMO_OR;  break;
  default:      a.amo_op = AMO_XOR; break;
  }
  return a;
}

// The bytes one request of a reduction covers: 4 or 8 (a word of the global
// reduction, an element of the cluster reduction).
inline uint32_t red_word_bytes(uint32_t type) {
  return type_bytes(type) == 8 ? 8 : 4;
}

// The ctl-level assert of a STORE-family instruction, or nullptr, checked
// first and in this order: encoding, the op x type table, FPAMO, the
// last-level cache (global and tensor reductions), the type size against the
// descriptor's element size (tensor reductions; d may be null otherwise).
inline const char* check_store_ctl(uint32_t kind, uint32_t op, uint32_t type, bool has_llc, bool has_fpamo,
                                   const Desc* d = nullptr) {
  if (kind >= ST_KINDS)
    return MSG_ENCODING;
  if (!is_reduce(kind))
    return nullptr;
  if (!red_legal(kind, op, type))
    return MSG_RED_INVALID;
  if (red_needs_fpamo(op, type) && !has_fpamo)
    return MSG_RED_FPAMO;
  if (kind != ST_RED_CLUSTER && !has_llc)
    return MSG_RED_LLC;
  if (kind == ST_RED_TENSOR && d && type_bytes(type) != d->elem)
    return MSG_RED_SIZE;
  return nullptr;
}

// The tensor store's operand asserts (after check_store_ctl): the layout
// (the K-major, Flat and BlockMajor layouts have no store form), the load's
// descriptor rules on src_rel (the block-relative source offset), then, in
// tile mode, a coordinate below 0. The box must also fit the issuer's block
// (footprint(); text "DXA box exceeds the destination block").
inline const char* check_tensor_store(const Desc& d, const Req& q, uint32_t src_rel) {
  if (d.layout != LAYOUT_ROW)
    return MSG_ST_LAYOUT;
  if (const char* m = check_tensor(d, src_rel))
    return m;
  if (!d.im2col) {
    for (uint32_t i = 0; i < d.rank; ++i) {
      if (q.coord[i] < 0)
        return MSG_ST_NEG;
    }
  }
  return nullptr;
}

inline const char* check_bgop(uint32_t op) {
  return op > BG_WAIT_READ ? MSG_ENCODING : nullptr;
}

inline const char* check_pfence(uint32_t sel) {
  return sel > PFENCE_ALL ? MSG_PFENCE : nullptr;
}

// The tensor store's byte map: f(lmem_rel, gaddr) for every byte the box
// writes, in walk order. lmem_rel is the block-relative byte (swizzled) that
// a load of the same box at src_rel writes; out-of-bounds elements give
// nothing. Returns the bytes written.
template <typename F>
uint32_t for_each_store_byte(const Desc& d, const Req& q, uint32_t src_rel, F f) {
  uint32_t n = 0;
  for_each_element(d, q, [&](uint32_t off, bool in, uint64_t ga) {
    if (!in)
      return;
    for (uint32_t b = 0; b < d.elem; ++b)
      f(swizzle(src_rel + off + b, d.swizzle), ga + b);
    n += d.elem;
  });
  return n;
}

// Applies a tensor store to global memory: gptr(gaddr) returns the byte at
// gaddr; lmem holds the issuer's block (lmem_size bytes; bytes past it read 0).
template <typename GPtr>
uint32_t store_box(const Desc& d, const Req& q, uint32_t src_rel, const uint8_t* lmem, size_t lmem_size,
                   GPtr gptr) {
  return for_each_store_byte(d, q, src_rel, [&](uint32_t a, uint64_t ga) {
    *gptr(ga) = a < lmem_size ? lmem[a] : 0;
  });
}

// funct3 = 5 operands: rs1 = {src, a1, a2, a3}, rs2 = {{ctl, c4}, 0, 0, 0};
// a1..a3 = {gdst, size, 0} (bulk store, global reduction),
// {dst, size, mbar} (cluster reduction), {{c1, c0}, {c3, c2}, 0} (tensor forms).
struct Store {
  uint32_t kind = ST_BULK;
  uint32_t op = 0;
  uint32_t type = 0;
  uint32_t slot = 0;
  uint64_t src = 0;
  uint64_t dst = 0;    // gdst, or the cluster reduction's raw destination
  uint64_t mbar = 0;
  uint32_t size = 0;
  Req q;
  bool     map_form = false;  // the tensor kinds only: ctl bit 15, the map address in rs2 lane 1
  uint64_t map = 0;
};

inline uint32_t st_ctl(uint32_t kind, uint32_t op, uint32_t type, uint32_t slot) {
  auto m = [](uint32_t v, uint32_t bits) { return v & ((1u << bits) - 1u); };
  return (m(kind, DXA_ST_CTL_KIND_BITS) << DXA_ST_CTL_KIND_LSB) | (m(op, DXA_ST_CTL_OP_BITS) << DXA_ST_CTL_OP_LSB)
       | (m(type, DXA_ST_CTL_TYPE_BITS) << DXA_ST_CTL_TYPE_LSB)
       | (m(slot, DXA_ST_CTL_SLOT_BITS) << DXA_ST_CTL_SLOT_LSB);
}

inline Ops pack_store(const Store& s) {
  Ops o;
  o.rs1[0] = s.src;
  if (is_tensor(s.kind)) {
    o.rs1[1] = pair(s.q.coord[0], s.q.coord[1]);
    o.rs1[2] = pair(s.q.coord[2], s.q.coord[3]);
  } else {
    o.rs1[1] = s.dst;
    o.rs1[2] = s.size;
    o.rs1[3] = s.kind == ST_RED_CLUSTER ? s.mbar : 0;
  }
  const uint32_t c4 = is_tensor(s.kind) ? (uint32_t)s.q.coord[4] : 0u;
  const uint32_t ctl = st_ctl(s.kind, s.op, s.type, s.map_form ? 0u : s.slot)
                     | (s.map_form ? (1u << DXA_ST_CTL_MAP_LSB) : 0u);
  o.rs2[0] = ((uint64_t)ctl << 32) | c4;
  if (s.map_form)
    o.rs2[1] = s.map;
  return o;
}

inline Store unpack_store(const Ops& o) {
  Store s;
  const uint32_t ctl = (uint32_t)(o.rs2[0] >> 32);
  s.kind = field(ctl, DXA_ST_CTL_KIND_LSB, DXA_ST_CTL_KIND_BITS);
  s.op   = field(ctl, DXA_ST_CTL_OP_LSB, DXA_ST_CTL_OP_BITS);
  s.type = field(ctl, DXA_ST_CTL_TYPE_LSB, DXA_ST_CTL_TYPE_BITS);
  s.slot = field(ctl, DXA_ST_CTL_SLOT_LSB, DXA_ST_CTL_SLOT_BITS);
  s.map_form = field(ctl, DXA_ST_CTL_MAP_LSB, 1) != 0;
  s.map  = s.map_form ? o.rs2[1] : 0;
  s.src  = o.rs1[0];
  if (is_tensor(s.kind)) {
    s.q.coord[0] = (int32_t)(uint32_t)o.rs1[1];
    s.q.coord[1] = (int32_t)(uint32_t)(o.rs1[1] >> 32);
    s.q.coord[2] = (int32_t)(uint32_t)o.rs1[2];
    s.q.coord[3] = (int32_t)(uint32_t)(o.rs1[2] >> 32);
    s.q.coord[4] = (int32_t)(uint32_t)o.rs2[0];
  } else {
    s.dst  = o.rs1[1];
    s.size = (uint32_t)o.rs1[2];
    s.mbar = s.kind == ST_RED_CLUSTER ? o.rs1[3] : 0;
  }
  return s;
}

// Instruction words: DXA (custom-0, funct7 3) funct3 6 with the op in the
// rs1 field and N in the rs2 field; the proxy fence (custom-2, funct3 7,
// funct7 1) with its selector in the rs1 field.
inline uint32_t insn_bgop(uint32_t op, uint32_t n) {
  return (3u << 25) | ((n & 31u) << 20) | ((op & 31u) << 15) | (F3_BGOP << 12) | 0x0Bu;
}

inline uint32_t insn_pfence(uint32_t sel) {
  return (1u << 25) | ((sel & 31u) << 15) | (7u << 12) | 0x5Bu;
}

///////////////////////////////////////////////////////////////////////////////
// Tensor maps in memory (dxa_tmap.h): the map forms of funct3 2 and 5, and
// the tensormap ops of funct3 7.

enum : uint32_t { F3_TMAP = DXA_F3_TMAP, TMAP_PREFETCH = DXA_TMAP_PREFETCH, TMAP_INVAL = DXA_TMAP_INVAL };
constexpr uint32_t TMAP_ENTRIES = DXA_TMAP_ENTRIES;
constexpr uint32_t CTL_MAP      = 1u << DXA_TL_CTL_MAP_LSB;
constexpr uint32_t ST_CTL_MAP   = 1u << DXA_ST_CTL_MAP_LSB;
constexpr uint32_t RT_FROM_MAP  = DXA_RT_FROM_MAP;

constexpr const char* MSG_TMAP_ALIGN    = "DXA tensor map is not 64-byte aligned";
constexpr const char* MSG_TMAP_GLOBAL   = "DXA tensor map must be in global memory";
constexpr const char* MSG_TMAP_STRIDE   = "DXA tensor stride needs 32 bits";
constexpr const char* MSG_TMAP_RANK     = "DXA tensor map rank must be 1-5";
constexpr const char* MSG_TMAP_REDTYPE  = "DXA tensor reduction type not supported by the map";
constexpr const char* MSG_TMAP_SLOT_RT  = "DXA reduction type 15 needs a tensor map";
constexpr const char* MSG_TMAP_OP       = "DXA tensormap op not supported";

// The map-form TLOAD: the slot bits 0, the map bit, the map address in rs2 lane 2.
inline Ops pack_tload_map(uint64_t map, bool prefetch, uint64_t dst, uint64_t mbar, const Req& q,
                          uint16_t cta_mask) {
  Ops o = pack_tload(0, prefetch, dst, mbar, q, cta_mask);
  o.rs2[0] |= (uint64_t)CTL_MAP << 32;
  o.rs2[2] = map;
  return o;
}

// The decode-time asserts of a map's words, in order (the descriptor asserts follow).
inline const char* check_map_words(const uint32_t* w) {
  if (w[TMAP_W_STRIDE_HI] != 0)
    return MSG_TMAP_STRIDE;
  const uint32_t r = dxa_tmap::rank(w);
  if (r == 0 || r > MAX_RANK)
    return MSG_TMAP_RANK;
  return nullptr;
}

// A map-form tensor reduction's type (DXA_RT_FROM_MAP): the map's element
// type, which must give a legal op x type pair; *type is the resolved DXA_RT_*.
inline const char* map_red_type(uint32_t op, const uint32_t* w, uint32_t* type) {
  const uint32_t rt = dxa_tmap::red_type(dxa_tmap::elem_type(w));
  if (rt == TMAP_RT_NONE || !red_legal(ST_RED_TENSOR, op, rt))
    return MSG_TMAP_REDTYPE;
  *type = rt;
  return nullptr;
}

// funct3 7: custom-0, funct7 3, rs1 = the register holding the map address, the op in the rs2 field.
inline uint32_t insn_tmap(uint32_t op, uint32_t rs1) {
  return (3u << 25) | ((op & 31u) << 20) | ((rs1 & 31u) << 15) | (F3_TMAP << 12) | 0x0Bu;
}

} // namespace dxa_ref
} // namespace vortex

#if __has_include(<softfloat.h>)
#include "amo_ref.h"

namespace vortex {
namespace dxa_ref {

static_assert((uint32_t)AMO_INC == (uint32_t)amo_ref::OP_INC && (uint32_t)AMO_DEC == (uint32_t)amo_ref::OP_DEC
              && (uint32_t)AMO_FADD == (uint32_t)amo_ref::OP_FADD
              && (uint32_t)AMO_FADD2 == (uint32_t)amo_ref::OP_FADD2
              && (uint32_t)AMO_FADD2_BF == (uint32_t)amo_ref::OP_FADD2_BF
              && (uint32_t)AMO_FMIN2 == (uint32_t)amo_ref::OP_FMIN2
              && (uint32_t)AMO_FMAX2 == (uint32_t)amo_ref::OP_FMAX2
              && (uint32_t)AMO_FMIN2_BF == (uint32_t)amo_ref::OP_FMIN2_BF
              && (uint32_t)AMO_FMAX2_BF == (uint32_t)amo_ref::OP_FMAX2_BF,
              "DXA reduction codes must be the amo_ref op codes");

// One element of a reduction: old and src right-aligned at the type's width,
// the result likewise. Float rules are the last-level cache's (amo_ref
// SITE_LLC: f32 flushed, f16 and bf16 .noftz); a 16-bit element is one lane
// of the packed op. An illegal pair returns old.
inline uint64_t reduce_elem(uint32_t op, uint32_t type, uint64_t old, uint64_t src) {
  const uint32_t E = type_bytes(type);
  const uint64_t m = E == 8 ? ~0ull : (1ull << (8 * E)) - 1;
  old &= m;
  src &= m;
  if (!red_legal(ST_RED_GLOBAL, op, type) && !red_legal(ST_RED_CLUSTER, op, type))
    return old;
  if (type_float(type) || op == RED_INC || op == RED_DEC) {
    const RedAmo a = red_amo(op, type);
    return amo_ref::apply(a.amo_op, a.width, old, src, amo_ref::SITE_LLC) & m;
  }
  const bool sgn = (type == RT_S32 || type == RT_S64);
  const int64_t so = E == 8 ? (int64_t)old : (int64_t)(int32_t)(uint32_t)old;
  const int64_t ss = E == 8 ? (int64_t)src : (int64_t)(int32_t)(uint32_t)src;
  uint64_t r = old;
  switch (op) {
  case RED_ADD: r = old + src; break;
  case RED_AND: r = old & src; break;
  case RED_OR:  r = old | src; break;
  case RED_XOR: r = old ^ src; break;
  case RED_MIN: r = sgn ? (so < ss ? old : src) : (old < src ? old : src); break;
  case RED_MAX: r = sgn ? (so > ss ? old : src) : (old > src ? old : src); break;
  default: break;
  }
  return r & m;
}

inline uint64_t load_le(const uint8_t* p, uint32_t n) {
  uint64_t v = 0;
  for (uint32_t i = 0; i < n; ++i)
    v |= (uint64_t)p[i] << (8 * i);
  return v;
}

inline void store_le(uint8_t* p, uint32_t n, uint64_t v) {
  for (uint32_t i = 0; i < n; ++i)
    p[i] = (uint8_t)(v >> (8 * i));
}

// The bulk reductions: dst[i] = op(dst[i], src[i]) over n bytes of whole elements.
inline void reduce_bytes(uint32_t op, uint32_t type, uint8_t* dst, const uint8_t* src, uint32_t n) {
  const uint32_t E = type_bytes(type);
  for (uint32_t o = 0; o + E <= n; o += E)
    store_le(dst + o, E, reduce_elem(op, type, load_le(dst + o, E), load_le(src + o, E)));
}

// The tensor reduction: the store's walk with reduce_elem on every in-bounds
// element; gptr(gaddr) returns the element's first byte (an element is
// contiguous in global memory). Returns the bytes reduced.
template <typename GPtr>
uint32_t reduce_box(const Desc& d, const Req& q, uint32_t src_rel, uint32_t op, uint32_t type,
                    const uint8_t* lmem, size_t lmem_size, GPtr gptr) {
  const uint32_t E = d.elem;
  uint32_t n = 0;
  for_each_element(d, q, [&](uint32_t off, bool in, uint64_t ga) {
    if (!in)
      return;
    uint8_t s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t b = 0; b < E && b < 8; ++b) {
      const uint32_t a = swizzle(src_rel + off + b, d.swizzle);
      s[b] = a < lmem_size ? lmem[a] : 0;
    }
    uint8_t* g = gptr(ga);
    store_le(g, E, reduce_elem(op, type, load_le(g, E), load_le(s, E)));
    n += E;
  });
  return n;
}

} // namespace dxa_ref
} // namespace vortex
#endif
