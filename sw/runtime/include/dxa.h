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

#ifndef __VX_DXA_HOST_H__
#define __VX_DXA_HOST_H__

#include <cstdint>
#include <initializer_list>

#include <vortex.h>
#include <VX_types.h>
#include <dxa_meta.h>
#include <dxa_tmap.h>

namespace vortex {
namespace dxa {

namespace detail {

struct reg_write {
  uint32_t off;
  uint32_t val;
};

inline int write_regs(vx_device_h dev, uint32_t dcr,
                      std::initializer_list<reg_write> writes) {
  for (const auto& w : writes) {
    if (int ret = vx_dcr_write(dev, dcr + w.off, w.val)) {
      return ret;
    }
  }
  return 0;
}

constexpr uint32_t pack_meta(uint32_t rank, uint32_t elem_size_enc) {
  return ((rank & ((1u << DXA_DESC_META_DIM_BITS) - 1u)) << DXA_DESC_META_DIM_LSB)
       | ((elem_size_enc & ((1u << DXA_DESC_META_ELEMSZ_BITS) - 1u)) << DXA_DESC_META_ELEMSZ_LSB);
}

constexpr uint32_t pack_2x16(uint32_t lo, uint32_t hi) {
  return ((hi & 0xffffu) << 16) | (lo & 0xffffu);
}

// elem_bytes must be a power of 2 (1, 2, 4, 8). Returns log2.
constexpr uint32_t elem_size_enc(uint32_t elem_bytes) {
  uint32_t enc = 0;
  for (uint32_t v = elem_bytes; v > 1; v >>= 1) {
    ++enc;
  }
  return enc;
}

} // namespace detail

// ── Destination SMEM layout ─────────────────────────────────────────
//
// ROW_MAJOR (default): a 2D tile is laid out as smem[i1 * tile0 + i0].
// K_MAJOR (TMA transposing mode): smem[i0 * tile1 + i1]; the
//   writer scatters one element per beat at +tile1*elem_bytes per
//   element. Rank must be ≤ 2; the writer drains 1 element/cycle in
//   this mode (~8× slower for B-style tiles, amortized across the many
//   WGMMA uops that consume each tile from LMEM).
// FLAT (sparse candidate-pair) / BLOCK_MAJOR (dense) are the WGMMA
//   B-buffer's native layouts. The writer reads B[K][N] row-major
//   (contiguous, like ROW_MAJOR) and scatters each element to the
//   flat/block-major destination — the cheap K_MAJOR direction
//   (contiguous read → permuted write). The tile geometry the
//   destination formula needs (tcN and the dense block K extent) is
//   conveyed via set_tile_geometry().
enum class Layout : uint32_t {
  RowMajor   = 0,
  KMajor     = 1,
  Flat       = 2,  // sparse B candidate-pair (vx_tensor.h::b_sp_flat_idx)
  BlockMajor = 3,  // dense B block-major (vx_tensor.h::b_blockmajor_idx)
};

// ── Public API ─────────────────────────────────────────────────────
//
// Each function writes only the DCR registers needed for its rank.
//   1D:  7 DCR writes
//   2D: 10 DCR writes
//   3D: 14 DCR writes
//   4D: 17 DCR writes
//   5D: 21 DCR writes
//
// `base_addr` is written to VX_DCR_DXA_DESC_BASE_{LO,HI} and is
// consumed by the DXA worker's AXI master, which bypasses the
// per-core MMU. The backing buffer must be allocated with
// VX_MEM_PHYS (see vortex2.h).

// Program a 1D DXA descriptor (7 DCR writes).
//   size0: dimension size (elements)
//   tile0: tile size (elements per transfer)
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
inline int program_1d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0,
    uint32_t tile0,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(1, detail::elem_size_enc(elem_bytes));
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_BASE_LO_OFF,   (uint32_t)(base_addr & 0xffffffffu)},
    {VX_DCR_DXA_DESC_BASE_HI_OFF,   (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF,     size0},
    {VX_DCR_DXA_DESC_META_OFF,      meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF,  1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, detail::pack_2x16(tile0, 0)},
    {VX_DCR_DXA_DESC_CFILL_OFF,     0},
  });
}

// Program a 2D DXA descriptor (10 DCR writes).
//   size0, size1: dimension sizes (elements)
//   stride0_bytes: byte stride between dim-1 rows
//   tile0, tile1: tile sizes (elements per transfer dimension)
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
inline int program_2d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1,
    uint32_t stride0_bytes,
    uint32_t tile0, uint32_t tile1,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(2, detail::elem_size_enc(elem_bytes));
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_BASE_LO_OFF,   (uint32_t)(base_addr & 0xffffffffu)},
    {VX_DCR_DXA_DESC_BASE_HI_OFF,   (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF,     size0},
    {VX_DCR_DXA_DESC_SIZE1_OFF,     size1},
    {VX_DCR_DXA_DESC_STRIDE0_OFF,   stride0_bytes},
    {VX_DCR_DXA_DESC_META_OFF,      meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE1_OFF,  1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, detail::pack_2x16(tile0, tile1)},
    {VX_DCR_DXA_DESC_CFILL_OFF,     0},
  });
}

// Program a 3D DXA descriptor (14 DCR writes).
//   size0, size1, size2: dimension sizes (elements)
//   stride0_bytes, stride1_bytes: byte strides for dim-1 and dim-2
//   tile0, tile1, tile2: tile sizes
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
inline int program_3d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1, uint32_t size2,
    uint32_t stride0_bytes, uint32_t stride1_bytes,
    uint32_t tile0, uint32_t tile1, uint32_t tile2,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(3, detail::elem_size_enc(elem_bytes));
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_BASE_LO_OFF,   (uint32_t)(base_addr & 0xffffffffu)},
    {VX_DCR_DXA_DESC_BASE_HI_OFF,   (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF,     size0},
    {VX_DCR_DXA_DESC_SIZE1_OFF,     size1},
    {VX_DCR_DXA_DESC_SIZE2_OFF,     size2},
    {VX_DCR_DXA_DESC_STRIDE0_OFF,   stride0_bytes},
    {VX_DCR_DXA_DESC_STRIDE1_OFF,   stride1_bytes},
    {VX_DCR_DXA_DESC_META_OFF,      meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE1_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE2_OFF,  1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, detail::pack_2x16(tile0, tile1)},
    {VX_DCR_DXA_DESC_TILESIZE23_OFF, detail::pack_2x16(tile2, 0)},
    {VX_DCR_DXA_DESC_CFILL_OFF,     0},
  });
}

// Program a 4D DXA descriptor (17 DCR writes).
//   size0..size3: dimension sizes (elements)
//   stride0..stride2_bytes: byte strides for dim-1 through dim-3
//   tile0..tile3: tile sizes
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
inline int program_4d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1, uint32_t size2, uint32_t size3,
    uint32_t stride0_bytes, uint32_t stride1_bytes, uint32_t stride2_bytes,
    uint32_t tile0, uint32_t tile1, uint32_t tile2, uint32_t tile3,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(4, detail::elem_size_enc(elem_bytes));
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_BASE_LO_OFF,   (uint32_t)(base_addr & 0xffffffffu)},
    {VX_DCR_DXA_DESC_BASE_HI_OFF,   (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF,     size0},
    {VX_DCR_DXA_DESC_SIZE1_OFF,     size1},
    {VX_DCR_DXA_DESC_SIZE2_OFF,     size2},
    {VX_DCR_DXA_DESC_SIZE3_OFF,     size3},
    {VX_DCR_DXA_DESC_STRIDE0_OFF,   stride0_bytes},
    {VX_DCR_DXA_DESC_STRIDE1_OFF,   stride1_bytes},
    {VX_DCR_DXA_DESC_STRIDE2_OFF,   stride2_bytes},
    {VX_DCR_DXA_DESC_META_OFF,      meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE1_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE2_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE3_OFF,  1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, detail::pack_2x16(tile0, tile1)},
    {VX_DCR_DXA_DESC_TILESIZE23_OFF, detail::pack_2x16(tile2, tile3)},
    {VX_DCR_DXA_DESC_CFILL_OFF,     0},
  });
}

// Program a 5D DXA descriptor (21 DCR writes).
//   size0..size4: dimension sizes (elements)
//   stride0..stride3_bytes: byte strides for dim-1 through dim-4
//   tile0..tile4: tile sizes
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
inline int program_5d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1, uint32_t size2, uint32_t size3, uint32_t size4,
    uint32_t stride0_bytes, uint32_t stride1_bytes, uint32_t stride2_bytes, uint32_t stride3_bytes,
    uint32_t tile0, uint32_t tile1, uint32_t tile2, uint32_t tile3, uint32_t tile4,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(5, detail::elem_size_enc(elem_bytes));
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_BASE_LO_OFF,   (uint32_t)(base_addr & 0xffffffffu)},
    {VX_DCR_DXA_DESC_BASE_HI_OFF,   (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF,     size0},
    {VX_DCR_DXA_DESC_SIZE1_OFF,     size1},
    {VX_DCR_DXA_DESC_SIZE2_OFF,     size2},
    {VX_DCR_DXA_DESC_SIZE3_OFF,     size3},
    {VX_DCR_DXA_DESC_SIZE4_OFF,     size4},
    {VX_DCR_DXA_DESC_STRIDE0_OFF,   stride0_bytes},
    {VX_DCR_DXA_DESC_STRIDE1_OFF,   stride1_bytes},
    {VX_DCR_DXA_DESC_STRIDE2_OFF,   stride2_bytes},
    {VX_DCR_DXA_DESC_STRIDE3_OFF,   stride3_bytes},
    {VX_DCR_DXA_DESC_META_OFF,      meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE1_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE2_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE3_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE4_OFF,  1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, detail::pack_2x16(tile0, tile1)},
    {VX_DCR_DXA_DESC_TILESIZE23_OFF, detail::pack_2x16(tile2, tile3)},
    {VX_DCR_DXA_DESC_TILESIZE4_OFF, tile4},
    {VX_DCR_DXA_DESC_CFILL_OFF,     0},
  });
}

// Program the multicast SMEM stride for an existing DXA descriptor.
//   smem_stride_bytes: byte offset between consecutive CTAs' SMEM bases
// Bar stride is always 1 (hardcoded in hardware).
// Multicast is active when cta_mask has more than one bit set.
inline int set_multicast(
    vx_device_h dev, uint32_t slot,
    uint32_t smem_stride_bytes) {
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  return vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_SMEM_STRIDE_OFF, smem_stride_bytes);
}

// Override the destination SMEM layout for an already-programmed descriptor.
// META is rewritten from scratch (rank + elem_bytes + layout); call AFTER
// program_Nd, passing the same rank and elem_bytes you used. (DCR is
// write-only on most paths — we don't read-modify-write META; we re-pack
// it locally.)
inline int set_layout(
    vx_device_h dev, uint32_t slot,
    Layout layout,
    uint32_t rank, uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(rank, detail::elem_size_enc(elem_bytes))
                | ((static_cast<uint32_t>(layout) & ((1u << DXA_DESC_META_LAYOUT_BITS) - 1u))
                   << DXA_DESC_META_LAYOUT_LSB);
  return vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_META_OFF, meta);
}

// Convey the WGMMA tile geometry the Flat/BlockMajor destination formula
// needs. `tcN` is the micro-tile N dimension (= tcK for canonical WGMMA
// configs); `blk_k` is the K extent of one dense B block in elements
// (fedpK * i_ratio — set by the consumer's FEDP depth, not derivable from
// tcN). Both are powers of two. The engine derives n_steps = tile0/tcN and
// ratio = 4/elem_bytes. Rides the vestigial ESTRIDE2 register (unused for
// rank ≤ 2): tcN in [15:0], blk_k in [31:16]. Call after program_Nd +
// set_layout(Flat|BlockMajor). No-op for RowMajor/KMajor.
inline int set_tile_geometry(
    vx_device_h dev, uint32_t slot,
    uint32_t tcN, uint32_t blk_k) {
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  return vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_ESTRIDE2_OFF,
                      (blk_k << 16) | (tcN & 0xffffu));
}

// ── PTX tensor-map options (DXA with mbarriers, vx_dxa_load_*) ─────────
//
// Swizzle: the PTX §5.5.7 layouts with 16-byte chunks (CU_TENSOR_MAP_SWIZZLE_*).
// Interleave: CU_TENSOR_MAP_INTERLEAVE_{NONE,16B,32B}; rank 3-5.
enum class Swizzle : uint32_t {
  None = 0,
  B32  = 1,
  B64  = 2,
  B128 = 3,
};

enum class Interleave : uint32_t {
  None = 0,
  B16  = 1,
  B32  = 2,
};

// Rewrites META (rank, elem_bytes, row-major layout, swizzle, interleave and
// the im2col bit, which must repeat what program_im2col_Nd set), the element
// strides ESTRIDE0..4 (1-8; nullptr = all 1) and CFILL: zero, or with
// oob_nan the element type's NaN replicated to 32 bits (0x7FFF7FFF for
// 2-byte elements, 0x7FFFFFFF otherwise). Call after program_Nd or
// program_im2col_Nd.
inline int set_tensor_options(
    vx_device_h dev, uint32_t slot,
    uint32_t rank, uint32_t elem_bytes,
    Swizzle swizzle, Interleave interleave,
    bool oob_nan,
    const uint32_t elem_strides[5],
    bool im2col = false) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(rank, detail::elem_size_enc(elem_bytes))
                | ((static_cast<uint32_t>(swizzle) & ((1u << DXA_DESC_META_SWIZZLE_BITS) - 1u))
                   << DXA_DESC_META_SWIZZLE_LSB)
                | ((static_cast<uint32_t>(interleave) & ((1u << DXA_DESC_META_INTERLEAVE_BITS) - 1u))
                   << DXA_DESC_META_INTERLEAVE_LSB)
                | ((im2col ? 1u : 0u) << DXA_DESC_META_IM2COL_LSB);
  const uint32_t ones[5] = {1, 1, 1, 1, 1};
  const uint32_t* es = elem_strides ? elem_strides : ones;
  const uint32_t cfill = !oob_nan ? 0u : (elem_bytes == 2 ? 0x7FFF7FFFu : 0x7FFFFFFFu);
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_META_OFF,     meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF, es[0]},
    {VX_DCR_DXA_DESC_ESTRIDE1_OFF, es[1]},
    {VX_DCR_DXA_DESC_ESTRIDE2_OFF, es[2]},
    {VX_DCR_DXA_DESC_ESTRIDE3_OFF, es[3]},
    {VX_DCR_DXA_DESC_ESTRIDE4_OFF, es[4]},
    {VX_DCR_DXA_DESC_CFILL_OFF,    cfill},
  });
}

namespace detail {

// TILESIZE23 in im2col mode: the pixel-box corners, signed, S = rank - 2
// spatial dims (W, H, D) of 16, 8 or 5 bits, lower corners first.
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

inline int program_im2col(
    vx_device_h dev, uint32_t slot, uint32_t rank,
    uint64_t base_addr,
    const uint32_t* sizes, const uint32_t* strides_bytes,
    uint32_t channels_per_pixel, uint32_t pixels_per_column,
    const int32_t* lower, const int32_t* upper,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = pack_meta(rank, elem_size_enc(elem_bytes)) | (1u << DXA_DESC_META_IM2COL_LSB);
  if (int ret = write_regs(dev, dcr, {
        {VX_DCR_DXA_DESC_BASE_LO_OFF,    (uint32_t)(base_addr & 0xffffffffu)},
        {VX_DCR_DXA_DESC_BASE_HI_OFF,    (uint32_t)(base_addr >> 32)},
        {VX_DCR_DXA_DESC_META_OFF,       meta},
        {VX_DCR_DXA_DESC_ESTRIDE0_OFF,   1},
        {VX_DCR_DXA_DESC_ESTRIDE1_OFF,   1},
        {VX_DCR_DXA_DESC_ESTRIDE2_OFF,   1},
        {VX_DCR_DXA_DESC_ESTRIDE3_OFF,   1},
        {VX_DCR_DXA_DESC_ESTRIDE4_OFF,   1},
        {VX_DCR_DXA_DESC_TILESIZE01_OFF, pack_2x16(channels_per_pixel, pixels_per_column)},
        {VX_DCR_DXA_DESC_TILESIZE23_OFF, pack_corners(rank, lower, upper)},
        {VX_DCR_DXA_DESC_TILESIZE4_OFF,  0},
        {VX_DCR_DXA_DESC_CFILL_OFF,      0},
      })) {
    return ret;
  }
  for (uint32_t d = 0; d < rank; ++d) {
    if (int ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_SIZE0_OFF + d, sizes[d]))
      return ret;
  }
  for (uint32_t d = 0; d + 1 < rank; ++d) {
    if (int ret = vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_STRIDE0_OFF + d, strides_bytes[d]))
      return ret;
  }
  return 0;
}

} // namespace detail

// Program an im2col descriptor (PTX §5.5.4, cuTensorMapEncodeIm2col):
// rank 3 (NWC), 4 (NHWC) or 5 (NDHWC), dims in order C, W[, H[, D]], N.
//   sizes[R]: elements per dim; strides_bytes[R-1]: byte stride of dims 1..R-1
//   channels_per_pixel 1-256, pixels_per_column 1-1024
//   lower[S], upper[S]: the pixel-box corners for W, H, D (S = R - 2)
// Set swizzle/NaN fill/traversal strides afterwards with
// set_tensor_options(..., im2col = true).
inline int program_im2col_3d(
    vx_device_h dev, uint32_t slot, uint64_t base_addr,
    const uint32_t sizes[3], const uint32_t strides_bytes[2],
    uint32_t channels_per_pixel, uint32_t pixels_per_column,
    const int32_t lower[1], const int32_t upper[1], uint32_t elem_bytes) {
  return detail::program_im2col(dev, slot, 3, base_addr, sizes, strides_bytes, channels_per_pixel,
                                pixels_per_column, lower, upper, elem_bytes);
}

inline int program_im2col_4d(
    vx_device_h dev, uint32_t slot, uint64_t base_addr,
    const uint32_t sizes[4], const uint32_t strides_bytes[3],
    uint32_t channels_per_pixel, uint32_t pixels_per_column,
    const int32_t lower[2], const int32_t upper[2], uint32_t elem_bytes) {
  return detail::program_im2col(dev, slot, 4, base_addr, sizes, strides_bytes, channels_per_pixel,
                                pixels_per_column, lower, upper, elem_bytes);
}

inline int program_im2col_5d(
    vx_device_h dev, uint32_t slot, uint64_t base_addr,
    const uint32_t sizes[5], const uint32_t strides_bytes[4],
    uint32_t channels_per_pixel, uint32_t pixels_per_column,
    const int32_t lower[3], const int32_t upper[3], uint32_t elem_bytes) {
  return detail::program_im2col(dev, slot, 5, base_addr, sizes, strides_bytes, channels_per_pixel,
                                pixels_per_column, lower, upper, elem_bytes);
}

// ── Tensor maps in memory (DXA with mbarriers, XLEN 64) ─────────────
//
// The cuTensorMapEncode* model: a vx_tensor_map_t (dxa_tmap.h) is filled on
// the host, copied into device memory (a kernel-argument member or a global
// buffer, physical as every DXA address) and named by the map forms of
// vx_dxa_load_* / vx_dxa_store_* / vx_dxa_reduce_*. The enums keep the CUDA
// driver API order (CUtensorMap*). Every function checks the driver API's
// rules and returns VX_SUCCESS, VX_ERR_INVALID_VALUE for a broken rule, or
// VX_ERR_NOT_SUPPORTED for a value the H100 (CC 9.0) or Vortex does not take;
// on an error the map is left unchanged.

enum class TensorMapDataType : uint32_t {
  UInt8 = 0, UInt16, UInt32, Int32, UInt64, Int64, Float16, Float32, Float64, BFloat16,
  Float32Ftz, TFloat32, TFloat32Ftz, U4x16Align8B, U4x16Align16B, U6x16Align16B,
};

enum class TensorMapInterleave : uint32_t { None = 0, B16, B32 };

enum class TensorMapSwizzle : uint32_t {
  None = 0, B32, B64, B128, B128Atom32B, B128Atom32BFlip8B, B128Atom64B,
};

enum class TensorMapL2Promotion : uint32_t { None = 0, L2_64B, L2_128B, L2_256B };

enum class TensorMapFloatOOBfill : uint32_t { None = 0, NanRequestZeroFma };

namespace detail {

// CUtensorMapDataType -> PTX Table 33 element type (the order differs from FLOAT64 on).
inline uint32_t tmap_elem_type(TensorMapDataType t) {
  static const uint8_t k[13] = {TMAP_ET_U8, TMAP_ET_U16, TMAP_ET_U32, TMAP_ET_S32, TMAP_ET_U64,
                                TMAP_ET_S64, TMAP_ET_F16, TMAP_ET_F32, TMAP_ET_F64, TMAP_ET_BF16,
                                TMAP_ET_F32_FTZ, TMAP_ET_TF32, TMAP_ET_TF32_FTZ};
  return k[static_cast<uint32_t>(t)];
}

inline bool tmap_aligned(const vx_tensor_map_t* map) {
  return map && ((uintptr_t)map & 63u) == 0;
}

// The rules cuTensorMapEncodeTiled and cuTensorMapEncodeIm2col share, and the
// word fields they give. box0_bytes is the swizzle's inner box (boxDim[0] or
// channelsPerPixel times the element size).
struct TmapCommon {
  uint32_t et = 0;
  uint32_t bytes = 0;
  uint32_t estride[5] = {0, 0, 0, 0, 0};  // dims past the rank stay 0 (read as 1), as a zero buffer has them
};

inline int tmap_check_common(const vx_tensor_map_t* map, TensorMapDataType dtype, uint32_t rank,
                             uint64_t gaddr, const uint64_t* gdim, const uint64_t* gstrides,
                             const uint32_t* estrides, TensorMapInterleave il, TensorMapSwizzle swz,
                             TensorMapL2Promotion l2, TensorMapFloatOOBfill oob, bool im2col,
                             uint32_t box0_elems, TmapCommon* c) {
  const uint32_t t = static_cast<uint32_t>(dtype), i = static_cast<uint32_t>(il),
                 z = static_cast<uint32_t>(swz), p = static_cast<uint32_t>(l2),
                 f = static_cast<uint32_t>(oob);
  if (!tmap_aligned(map))
    return VX_ERR_INVALID_VALUE;
  if (t > static_cast<uint32_t>(TensorMapDataType::U6x16Align16B) || i > 2 || z > 6 || p > 3 || f > 1)
    return VX_ERR_INVALID_VALUE;                                                  // enum ranges
  if (t >= static_cast<uint32_t>(TensorMapDataType::U4x16Align8B) || z >= 4)
    return VX_ERR_NOT_SUPPORTED;                                                  // CC 10.0 values
  c->et = tmap_elem_type(dtype);
  c->bytes = dxa_tmap::elem_bytes(c->et);
  if (f && !dxa_tmap::elem_float(c->et))                                          // NaN fill: float types only
    return VX_ERR_INVALID_VALUE;
  if (im2col ? (rank < 3 || rank > 5) : (rank < 1 || rank > 5))
    return VX_ERR_INVALID_VALUE;
  if (il != TensorMapInterleave::None && rank < 3)                                // interleave: rank >= 3
    return VX_ERR_INVALID_VALUE;
  const uint64_t align = il == TensorMapInterleave::B32 ? 32 : 16;
  if (gaddr % align)
    return VX_ERR_INVALID_VALUE;
  if (!gdim || (rank > 1 && !gstrides) || !estrides)
    return VX_ERR_INVALID_VALUE;
  bool wide = false;
  for (uint32_t d = 0; d < rank; ++d) {
    if (gdim[d] == 0 || gdim[d] > (1ull << 32))
      return VX_ERR_INVALID_VALUE;
    wide |= gdim[d] == (1ull << 32);
  }
  for (uint32_t d = 0; d + 1 < rank; ++d) {
    if (gstrides[d] % align || gstrides[d] >= (1ull << 40))
      return VX_ERR_INVALID_VALUE;
    wide |= gstrides[d] >= (1ull << 32);
  }
  for (uint32_t d = 0; d < rank; ++d) {
    if (estrides[d] == 0 || estrides[d] > 8)
      return VX_ERR_INVALID_VALUE;
  }
  if (il == TensorMapInterleave::None && swz != TensorMapSwizzle::None) {         // inner box <= swizzle span
    const uint32_t span = swz == TensorMapSwizzle::B32 ? 32 : swz == TensorMapSwizzle::B64 ? 64 : 128;
    if ((uint64_t)box0_elems * c->bytes > span)
      return VX_ERR_INVALID_VALUE;
  }
  if (il == TensorMapInterleave::B32 && swz != TensorMapSwizzle::B32)
    return VX_ERR_INVALID_VALUE;
  if (wide)                                                     // Vortex: 32-bit dims and strides
    return VX_ERR_NOT_SUPPORTED;
  if (im2col && il != TensorMapInterleave::None && estrides[0] != 1)
    return VX_ERR_NOT_SUPPORTED;                                // Vortex: no dim-0 stride in im2col
  for (uint32_t d = 0; d < rank; ++d)
    c->estride[d] = estrides[d];
  if (il == TensorMapInterleave::None)
    c->estride[0] = 1;                                          // ignored without interleave
  return VX_SUCCESS;
}

inline void tmap_fill_common(uint32_t* w, const TmapCommon& c, uint32_t rank, uint64_t gaddr,
                             const uint64_t* gdim, const uint64_t* gstrides, TensorMapInterleave il,
                             TensorMapSwizzle swz, TensorMapL2Promotion l2, TensorMapFloatOOBfill oob,
                             bool im2col) {
  for (uint32_t k = 0; k < TMAP_WORDS; ++k)
    w[k] = 0;
  w[TMAP_W_BASE_LO] = (uint32_t)gaddr;
  w[TMAP_W_BASE_HI] = (uint32_t)(gaddr >> 32);
  for (uint32_t d = 0; d < rank; ++d)
    w[TMAP_W_SIZE0 + d] = (uint32_t)gdim[d];
  for (uint32_t d = 0; d + 1 < rank; ++d)
    w[TMAP_W_STRIDE0 + d] = (uint32_t)gstrides[d];
  w[TMAP_W_META] = dxa_tmap::pack_meta(rank, c.bytes, static_cast<uint32_t>(Layout::RowMajor),
                                       static_cast<uint32_t>(swz), static_cast<uint32_t>(il),
                                       static_cast<uint32_t>(l2), im2col);
  for (uint32_t d = 0; d < 5; ++d)
    w[TMAP_W_ESTRIDE0 + d] = c.estride[d];
  const uint32_t fill = static_cast<uint32_t>(oob);
  w[TMAP_W_CFILL] = dxa_tmap::cfill(fill, c.bytes);
  w[TMAP_W_TYPE] = dxa_tmap::pack_type(c.et, fill);
}

inline void tmap_store(vx_tensor_map_t* map, const uint32_t* w) {
  for (uint32_t k = 0; k < TMAP_WORDS; ++k)
    map->w[k] = w[k];
}

} // namespace detail

// cuTensorMapEncodeTiled: global_dim[rank], global_strides[rank - 1] (bytes,
// of dims 1..rank-1), box_dim[rank], element_strides[rank] (with
// TensorMapInterleave::None the first is ignored and stored as 1).
inline int encode_tiled(vx_tensor_map_t* map, TensorMapDataType dtype, uint32_t rank, uint64_t global_address,
                        const uint64_t* global_dim, const uint64_t* global_strides, const uint32_t* box_dim,
                        const uint32_t* element_strides, TensorMapInterleave interleave,
                        TensorMapSwizzle swizzle, TensorMapL2Promotion l2_promotion,
                        TensorMapFloatOOBfill oob_fill) {
  detail::TmapCommon c;
  const uint32_t box0 = box_dim && rank >= 1 && rank <= 5 ? box_dim[0] : 0;
  if (int r = detail::tmap_check_common(map, dtype, rank, global_address, global_dim, global_strides,
                                        element_strides, interleave, swizzle, l2_promotion, oob_fill,
                                        false, box0, &c))
    return r;
  if (!box_dim)
    return VX_ERR_INVALID_VALUE;
  for (uint32_t d = 0; d < rank; ++d) {
    if (box_dim[d] == 0 || box_dim[d] > 256)
      return VX_ERR_INVALID_VALUE;
  }
  if (interleave == TensorMapInterleave::None && (box_dim[0] * c.bytes) % 16)    // inner box: 16-byte multiple
    return VX_ERR_INVALID_VALUE;
  uint32_t w[TMAP_WORDS];
  detail::tmap_fill_common(w, c, rank, global_address, global_dim, global_strides, interleave, swizzle,
                           l2_promotion, oob_fill, false);
  uint32_t b[5] = {0, 0, 0, 0, 0};
  for (uint32_t d = 0; d < rank; ++d)
    b[d] = box_dim[d];
  w[TMAP_W_TILESIZE01] = dxa_tmap::pack_2x16(b[0], b[1]);
  w[TMAP_W_TILESIZE23] = dxa_tmap::pack_2x16(b[2], b[3]);
  w[TMAP_W_TILESIZE4]  = b[4];
  detail::tmap_store(map, w);
  return VX_SUCCESS;
}

// cuTensorMapEncodeIm2col: rank 3 (NWC), 4 (NHWC) or 5 (NDHWC), dims in
// order C, W[, H[, D]], N; pixel_box_lower/upper[rank - 2] in order W, H, D
// (index 0 = W, as CUTLASS passes them).
inline int encode_im2col(vx_tensor_map_t* map, TensorMapDataType dtype, uint32_t rank, uint64_t global_address,
                         const uint64_t* global_dim, const uint64_t* global_strides,
                         const int32_t* pixel_box_lower, const int32_t* pixel_box_upper,
                         uint32_t channels_per_pixel, uint32_t pixels_per_column,
                         const uint32_t* element_strides, TensorMapInterleave interleave,
                         TensorMapSwizzle swizzle, TensorMapL2Promotion l2_promotion,
                         TensorMapFloatOOBfill oob_fill) {
  detail::TmapCommon c;
  if (int r = detail::tmap_check_common(map, dtype, rank, global_address, global_dim, global_strides,
                                        element_strides, interleave, swizzle, l2_promotion, oob_fill,
                                        true, channels_per_pixel, &c))
    return r;
  if (!pixel_box_lower || !pixel_box_upper)
    return VX_ERR_INVALID_VALUE;
  const uint32_t S = rank - 2;
  const int32_t lim = S == 1 ? 32768 : S == 2 ? 128 : 16;
  for (uint32_t s = 0; s < S; ++s) {
    const int32_t lo = pixel_box_lower[s], hi = pixel_box_upper[s];
    if (lo < -lim || lo >= lim || hi < -lim || hi >= lim)                          // corner range by rank
      return VX_ERR_INVALID_VALUE;
    if ((int64_t)global_dim[s + 1] + hi - lo <= 0)                                // non-zero bounding box
      return VX_ERR_INVALID_VALUE;
  }
  if (channels_per_pixel == 0 || channels_per_pixel > 256 || pixels_per_column == 0
      || pixels_per_column > 1024)
    return VX_ERR_INVALID_VALUE;
  uint32_t w[TMAP_WORDS];
  detail::tmap_fill_common(w, c, rank, global_address, global_dim, global_strides, interleave, swizzle,
                           l2_promotion, oob_fill, true);
  w[TMAP_W_TILESIZE01] = dxa_tmap::pack_2x16(channels_per_pixel, pixels_per_column);
  w[TMAP_W_TILESIZE23] = dxa_tmap::pack_corners(rank, pixel_box_lower, pixel_box_upper);
  detail::tmap_store(map, w);
  return VX_SUCCESS;
}

// cuTensorMapReplaceAddress: the new address keeps the encode-time alignment
// rule (16 B, 32 B with TensorMapInterleave::B32).
inline int replace_address(vx_tensor_map_t* map, uint64_t global_address) {
  if (!detail::tmap_aligned(map))
    return VX_ERR_INVALID_VALUE;
  const uint32_t il = dxa_tmap::field(map->w[TMAP_W_META], DXA_DESC_META_INTERLEAVE_LSB,
                                      DXA_DESC_META_INTERLEAVE_BITS);
  if (global_address % (il == static_cast<uint32_t>(TensorMapInterleave::B32) ? 32 : 16))
    return VX_ERR_INVALID_VALUE;
  dxa_tmap::replace_global_address(map->w, global_address);
  return VX_SUCCESS;
}

// Vortex extension, the map form of set_layout + set_tile_geometry: the
// destination smem layout of a tile map. The K-major, Flat and BlockMajor
// layouts take rank <= 2, no swizzle and element strides 1; Flat and
// BlockMajor also take the WGMMA tile geometry (tcN, blk_k) in ESTRIDE2.
inline int set_layout(vx_tensor_map_t* map, Layout layout, uint32_t tcN, uint32_t blk_k) {
  if (!detail::tmap_aligned(map) || static_cast<uint32_t>(layout) > 3)
    return VX_ERR_INVALID_VALUE;
  uint32_t* w = map->w;
  const uint32_t meta = w[TMAP_W_META];
  if (layout != Layout::RowMajor) {
    if (dxa_tmap::field(meta, DXA_DESC_META_IM2COL_LSB, DXA_DESC_META_IM2COL_BITS)
        || dxa_tmap::rank(w) > 2
        || dxa_tmap::field(meta, DXA_DESC_META_SWIZZLE_LSB, DXA_DESC_META_SWIZZLE_BITS))
      return VX_ERR_INVALID_VALUE;
    for (uint32_t d = 0; d < 2; ++d) {
      if (w[TMAP_W_ESTRIDE0 + d] > 1)
        return VX_ERR_INVALID_VALUE;
    }
  }
  w[TMAP_W_META] = dxa_tmap::set_field(meta, DXA_DESC_META_LAYOUT_LSB, DXA_DESC_META_LAYOUT_BITS,
                                       static_cast<uint32_t>(layout));
  if (layout == Layout::Flat || layout == Layout::BlockMajor)
    w[TMAP_W_ESTRIDE0 + 2] = (blk_k << 16) | (tcN & 0xffffu);
  else if (dxa_tmap::rank(w) <= 2)
    w[TMAP_W_ESTRIDE0 + 2] = 0;
  return VX_SUCCESS;
}

} // namespace dxa
} // namespace vortex

#endif // __VX_DXA_HOST_H__
