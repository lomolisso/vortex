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
// fill byte rule, the descriptor and operand encodings and the asserts. Host
// code, SimX and the RTL unit bench share it; plain functions over Desc + Req.

#include <cstdint>
#include <cstddef>
#include <VX_types.h>
#include "dxa_meta.h"

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

} // namespace dxa_ref
} // namespace vortex
