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

#ifndef __VX_DXA_H__
#define __VX_DXA_H__

#include <vx_intrinsics.h>
#include <stdint.h>

#ifdef VX_CFG_EXT_MBAR_ENABLE
#include <vx_mbar.h>
#include <vx_dlmem.h>
#include <dxa_meta.h>
#ifdef __cplusplus
#include <dxa_tmap.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VX_DXA_EXT_OPCODE RISCV_CUSTOM0
#define VX_DXA_FUNCT7     0x3

// Packed launch metadata:
// meta[3:0]   = descriptor slot (up to 16 descriptors)
// meta[30]    = completion mode: 0 = barrier event, 1 = mbarrier complete_tx
// meta[29:4]  = mode 0: raw barrier id payload;
//               mode 1: the mbarrier's doubleword offset in the core's local memory;
//               with meta[29] set (_cluster forms), meta[28:24] is its cluster
//               rank and the offset is in that block
// meta[31]    = packed marker

// R type: .insn r opcode7, funct3, funct7, rd, rs1, rs2
// +--------+-----+-----+--------+----+---------+
// | funct7 | rs2 | rs1 | funct3 | rd | opcode7 |
// +--------+-----+-----+--------+----+---------+
// 31       25    20    15       12   7         0

// Wgather-based DXA issue.  vx_wgather distributes args into per-lane register
// slots before the DXA instruction reads them across all 4 lanes:
//   Lane 0: rs1 = smem_addr,  rs2 = coord2
//   Lane 1: rs1 = meta,       rs2 = coord3
//   Lane 2: rs1 = coord0,     rs2 = coord4
//   Lane 3: rs1 = coord1,     rs2 = 0
//
// 1D and 2D: all rs2 lanes are zero, so rs2 = x0 (no second vx_wgather).
// 3D–5D: rs2 carries coord2..coord4, requiring a second vx_wgather.
//
// The issue may come from the whole warp or from one elected thread, as long
// as thread 0 of the warp is active: it is the wgather source lane, and the
// gather fills lanes 1-3 even when they are inactive.

inline uint32_t vx_dxa_pack_meta(uint32_t desc_slot, uint32_t barrier_id) {
  return (barrier_id << 4) | desc_slot;
}

// 1D: rs1 = wgather(smem_addr, meta, coord0, 0), rs2 = x0
inline void vx_dxa_issue_1d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                const void* smem_addr,
                                uint32_t coord0) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)0u);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, x0\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0)
      : "memory");
}

// 2D: rs1 = wgather(smem_addr, meta, coord0, coord1), rs2 = x0
inline void vx_dxa_issue_2d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                const void* smem_addr,
                                uint32_t coord0,
                                uint32_t coord1) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, x0\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0)
      : "memory");
}

// 3D–5D: rs2 = wgather(coord2, coord3, coord4, 0)
inline void vx_dxa_issue_3d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                const void* smem_addr,
                                uint32_t coord0,
                                uint32_t coord1,
                                uint32_t coord2) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)0u,
                                            (size_t)0u,
                                            (size_t)0u);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

inline void vx_dxa_issue_4d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                const void* smem_addr,
                                uint32_t coord0,
                                uint32_t coord1,
                                uint32_t coord2,
                                uint32_t coord3) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)coord3,
                                            (size_t)0u,
                                            (size_t)0u);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

inline void vx_dxa_issue_5d_wg(uint32_t desc_slot,
                                uint32_t barrier_id,
                                const void* smem_addr,
                                uint32_t coord0,
                                uint32_t coord1,
                                uint32_t coord2,
                                uint32_t coord3,
                                uint32_t coord4) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)coord3,
                                            (size_t)coord4,
                                            (size_t)0u);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

#ifdef VX_CFG_EXT_MBAR_ENABLE
// mbarrier completion, like a TMA load with .mbarrier::complete_tx::bytes:
// once the box (tile0 x .. x tile4 x elem_bytes, out-of-bounds fill
// included) has landed, the engine applies complete_tx(box bytes) to `mbar`,
// an object in the issuing block's own local memory. The kernel declares the
// bytes with vx_mbar_arrive_tx() or vx_mbar_expect_tx().

// meta[29:4] for the object, with meta[30] set (bit 26 of the payload).
inline uint32_t __vx_dxa_mbar_payload(const uint64_t* mbar) {
  const uint32_t off = (uint32_t)(((uintptr_t)mbar - (uintptr_t)VX_MEM_LMEM_BASE_ADDR) >> 3);
  return (1u << 26) | (off & ((1u << (VX_CFG_LMEM_LOG_SIZE - 3)) - 1u));
}

inline uint32_t vx_dxa_pack_meta_mbar(uint32_t desc_slot, const uint64_t* mbar) {
  return vx_dxa_pack_meta(desc_slot, __vx_dxa_mbar_payload(mbar));
}

inline void vx_dxa_issue_1d_mbar(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                 uint32_t coord0) {
  vx_dxa_issue_1d_wg(desc_slot, __vx_dxa_mbar_payload(mbar), smem_addr, coord0);
}

inline void vx_dxa_issue_2d_mbar(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                 uint32_t coord0, uint32_t coord1) {
  vx_dxa_issue_2d_wg(desc_slot, __vx_dxa_mbar_payload(mbar), smem_addr, coord0, coord1);
}

inline void vx_dxa_issue_3d_mbar(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                 uint32_t coord0, uint32_t coord1, uint32_t coord2) {
  vx_dxa_issue_3d_wg(desc_slot, __vx_dxa_mbar_payload(mbar), smem_addr, coord0, coord1, coord2);
}

inline void vx_dxa_issue_4d_mbar(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                 uint32_t coord0, uint32_t coord1, uint32_t coord2,
                                 uint32_t coord3) {
  vx_dxa_issue_4d_wg(desc_slot, __vx_dxa_mbar_payload(mbar), smem_addr,
                     coord0, coord1, coord2, coord3);
}

inline void vx_dxa_issue_5d_mbar(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                 uint32_t coord0, uint32_t coord1, uint32_t coord2,
                                 uint32_t coord3, uint32_t coord4) {
  vx_dxa_issue_5d_wg(desc_slot, __vx_dxa_mbar_payload(mbar), smem_addr,
                     coord0, coord1, coord2, coord3, coord4);
}

// The _cluster forms, like cp.async.bulk.tensor.shared::cluster: `smem_addr`
// and `mbar` are window addresses (vx_cluster_map_shared_rank) of one block of
// the issuer's cluster, which may be the issuer itself. A local pointer is
// undefined.

// meta[29:4] for a window object: the window flag, its rank and its
// doubleword offset in that block.
inline uint32_t __vx_dxa_mbar_payload_cluster(const uint64_t* mbar) {
  const uintptr_t a = (uintptr_t)mbar;
  return (1u << 26) | (1u << 25) | (((uint32_t)(a >> VX_CFG_LMEM_LOG_SIZE) & 0x1Fu) << 20)
       | ((uint32_t)(a >> 3) & ((1u << (VX_CFG_LMEM_LOG_SIZE - 3)) - 1u));
}

inline void vx_dxa_issue_1d_mbar_cluster(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                         uint32_t coord0) {
  vx_dxa_issue_1d_wg(desc_slot, __vx_dxa_mbar_payload_cluster(mbar), smem_addr, coord0);
}

inline void vx_dxa_issue_2d_mbar_cluster(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                         uint32_t coord0, uint32_t coord1) {
  vx_dxa_issue_2d_wg(desc_slot, __vx_dxa_mbar_payload_cluster(mbar), smem_addr, coord0, coord1);
}

inline void vx_dxa_issue_3d_mbar_cluster(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                         uint32_t coord0, uint32_t coord1, uint32_t coord2) {
  vx_dxa_issue_3d_wg(desc_slot, __vx_dxa_mbar_payload_cluster(mbar), smem_addr,
                     coord0, coord1, coord2);
}

inline void vx_dxa_issue_4d_mbar_cluster(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                         uint32_t coord0, uint32_t coord1, uint32_t coord2,
                                         uint32_t coord3) {
  vx_dxa_issue_4d_wg(desc_slot, __vx_dxa_mbar_payload_cluster(mbar), smem_addr,
                     coord0, coord1, coord2, coord3);
}

inline void vx_dxa_issue_5d_mbar_cluster(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                         uint32_t coord0, uint32_t coord1, uint32_t coord2,
                                         uint32_t coord3, uint32_t coord4) {
  vx_dxa_issue_5d_wg(desc_slot, __vx_dxa_mbar_payload_cluster(mbar), smem_addr,
                     coord0, coord1, coord2, coord3, coord4);
}

// Bulk copy, like cp.async.bulk.shared::cluster.shared::cta
// .mbarrier::complete_tx::bytes: the engine copies `size` bytes from `src` in
// the issuer's own block to `dst`, then applies complete_tx(size) to `mbar`.
// `dst` and `mbar` are each a local pointer into the issuer's own block or a
// window address (vx_cluster_map_shared_rank) of one block of its cluster,
// the issuer's own included, and `mbar` must be in the block of `dst`.
// `size` is a nonzero multiple of 16, `src` and `dst` are 16-byte aligned, and
// overlapping ranges are undefined. The DXA reads `src` as the async proxy:
// order earlier stores to it with vx_fence_proxy_async(). Same issue rules as
// the tensor issue: rs1 = wgather(dst, mbar, src, size), funct3 = 1.
inline void vx_dxa_bulk_copy(void* dst, const void* src, uint32_t size, uint64_t* mbar) {
  const size_t a0 = vx_wgather((size_t)(uintptr_t)dst,
                               (size_t)(uintptr_t)mbar,
                               (size_t)(uintptr_t)src,
                               (size_t)size);
  __asm__ volatile (
      ".insn r %0, 1, %1, x0, %2, x0\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0)
      : "memory");
}

// fence.proxy.async: holds the warp until its earlier stores to local memory
// have been performed and its earlier window stores acknowledged, so that a
// later DXA access of the same bytes observes them. Nothing reaches the caches.
inline void vx_fence_proxy_async(void) {
  __asm__ volatile (".insn r %0, 7, 1, x0, x0, x0" :: "i"(RISCV_CUSTOM2) : "memory");
}

// PTX tensor loads, cp.async.bulk.tensor.{1..5}d.shared::cluster.global
// .{tile,im2col}.mbarrier::complete_tx::bytes: `dst` and `mbar` are each a
// local pointer into the issuer's own block or a window address
// (vx_cluster_map_shared_rank) of one block of its cluster, decoded by the
// hardware, and `mbar` (8-byte aligned) must be in the block of `dst`. The
// descriptor slot selects tile or im2col mode; coordinates are signed and a box
// element outside the tensor is filled (zero or OOB-NaN). The engine applies
// complete_tx(box bytes) to `mbar` once the box has landed. The _multicast
// forms land in every block of the cluster whose rank is set in `cta_mask`, at
// the block-relative offsets of the issuer's local `dst` and `mbar`. The _hint
// forms take the .L2::cache_hint operand and drop it.
// funct3 = 2: rs1 = wgather(dst, mbar, {c1, c0}, {c3, c2}),
//             rs2 = wgather({ctl, c4}, {o3, o2, o1, cta_mask}, 0, 0), ctl = slot | prefetch << 4.

#define VX_DXA_CTL_PREFETCH (1u << 4)

inline size_t __vx_dxa_pair(int32_t lo, int32_t hi) {
  return ((size_t)(uint32_t)hi << 32) | (size_t)(uint32_t)lo;
}

inline void __vx_dxa_tload(uint32_t ctl, void* dst, uint64_t* mbar,
                           int32_t c0, int32_t c1, int32_t c2, int32_t c3, int32_t c4,
                           uint32_t o1, uint32_t o2, uint32_t o3, uint32_t cta_mask) {
  const size_t a0 = vx_wgather((size_t)(uintptr_t)dst,
                               (size_t)(uintptr_t)mbar,
                               __vx_dxa_pair(c0, c1),
                               __vx_dxa_pair(c2, c3));
  const size_t a1 = vx_wgather(((size_t)ctl << 32) | (size_t)(uint32_t)c4,
                               ((size_t)(o3 & 0xFFFFu) << 48) | ((size_t)(o2 & 0xFFFFu) << 32)
                                 | ((size_t)(o1 & 0xFFFFu) << 16) | (size_t)(cta_mask & 0xFFFFu),
                               (size_t)0,
                               (size_t)0);
  __asm__ volatile (
      ".insn r %0, 2, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

inline void vx_dxa_load_1d(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0) {
  __vx_dxa_tload(slot, dst, mbar, c0, 0, 0, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_load_2d(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, 0, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_load_3d(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1, int32_t c2) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_load_4d(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1, int32_t c2,
                           int32_t c3) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, c3, 0, 0, 0, 0, 0);
}
inline void vx_dxa_load_5d(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1, int32_t c2,
                           int32_t c3, int32_t c4) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, c3, c4, 0, 0, 0, 0);
}

inline void vx_dxa_load_1d_multicast(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0,
                                     uint16_t cta_mask) {
  __vx_dxa_tload(slot, dst, mbar, c0, 0, 0, 0, 0, 0, 0, 0, cta_mask);
}
inline void vx_dxa_load_2d_multicast(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                     uint16_t cta_mask) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, 0, 0, 0, 0, 0, 0, cta_mask);
}
inline void vx_dxa_load_3d_multicast(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                     int32_t c2, uint16_t cta_mask) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, 0, 0, 0, 0, 0, cta_mask);
}
inline void vx_dxa_load_4d_multicast(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                     int32_t c2, int32_t c3, uint16_t cta_mask) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, c3, 0, 0, 0, 0, cta_mask);
}
inline void vx_dxa_load_5d_multicast(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                     int32_t c2, int32_t c3, int32_t c4, uint16_t cta_mask) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, c3, c4, 0, 0, 0, cta_mask);
}

// im2col: c0 = first channel, c1..c(R-2) = the filter base (w, h, d), c(R-1) = n;
// o1..o3 = the im2col offsets for W, H, D.
inline void vx_dxa_load_im2col_3d(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                  int32_t c2, uint16_t o1) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, 0, 0, o1, 0, 0, 0);
}
inline void vx_dxa_load_im2col_4d(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                  int32_t c2, int32_t c3, uint16_t o1, uint16_t o2) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, c3, 0, o1, o2, 0, 0);
}
inline void vx_dxa_load_im2col_5d(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                  int32_t c2, int32_t c3, int32_t c4, uint16_t o1, uint16_t o2, uint16_t o3) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, c3, c4, o1, o2, o3, 0);
}
inline void vx_dxa_load_im2col_3d_multicast(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0,
                                            int32_t c1, int32_t c2, uint16_t o1, uint16_t cta_mask) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, 0, 0, o1, 0, 0, cta_mask);
}
inline void vx_dxa_load_im2col_4d_multicast(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0,
                                            int32_t c1, int32_t c2, int32_t c3, uint16_t o1, uint16_t o2,
                                            uint16_t cta_mask) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, c3, 0, o1, o2, 0, cta_mask);
}
inline void vx_dxa_load_im2col_5d_multicast(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0,
                                            int32_t c1, int32_t c2, int32_t c3, int32_t c4, uint16_t o1,
                                            uint16_t o2, uint16_t o3, uint16_t cta_mask) {
  __vx_dxa_tload(slot, dst, mbar, c0, c1, c2, c3, c4, o1, o2, o3, cta_mask);
}

inline void vx_dxa_load_1d_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_1d(slot, dst, mbar, c0);
}
inline void vx_dxa_load_2d_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_2d(slot, dst, mbar, c0, c1);
}
inline void vx_dxa_load_3d_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                int32_t c2, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_3d(slot, dst, mbar, c0, c1, c2);
}
inline void vx_dxa_load_4d_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                int32_t c2, int32_t c3, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_4d(slot, dst, mbar, c0, c1, c2, c3);
}
inline void vx_dxa_load_5d_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                int32_t c2, int32_t c3, int32_t c4, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_5d(slot, dst, mbar, c0, c1, c2, c3, c4);
}
inline void vx_dxa_load_1d_multicast_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0,
                                          uint16_t cta_mask, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_1d_multicast(slot, dst, mbar, c0, cta_mask);
}
inline void vx_dxa_load_2d_multicast_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                          uint16_t cta_mask, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_2d_multicast(slot, dst, mbar, c0, c1, cta_mask);
}
inline void vx_dxa_load_3d_multicast_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                          int32_t c2, uint16_t cta_mask, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_3d_multicast(slot, dst, mbar, c0, c1, c2, cta_mask);
}
inline void vx_dxa_load_4d_multicast_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                          int32_t c2, int32_t c3, uint16_t cta_mask, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_4d_multicast(slot, dst, mbar, c0, c1, c2, c3, cta_mask);
}
inline void vx_dxa_load_5d_multicast_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                          int32_t c2, int32_t c3, int32_t c4, uint16_t cta_mask,
                                          uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_5d_multicast(slot, dst, mbar, c0, c1, c2, c3, c4, cta_mask);
}
inline void vx_dxa_load_im2col_3d_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                       int32_t c2, uint16_t o1, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_3d(slot, dst, mbar, c0, c1, c2, o1);
}
inline void vx_dxa_load_im2col_4d_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                       int32_t c2, int32_t c3, uint16_t o1, uint16_t o2, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_4d(slot, dst, mbar, c0, c1, c2, c3, o1, o2);
}
inline void vx_dxa_load_im2col_5d_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                       int32_t c2, int32_t c3, int32_t c4, uint16_t o1, uint16_t o2,
                                       uint16_t o3, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_5d(slot, dst, mbar, c0, c1, c2, c3, c4, o1, o2, o3);
}
inline void vx_dxa_load_im2col_3d_multicast_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0,
                                                 int32_t c1, int32_t c2, uint16_t o1, uint16_t cta_mask,
                                                 uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_3d_multicast(slot, dst, mbar, c0, c1, c2, o1, cta_mask);
}
inline void vx_dxa_load_im2col_4d_multicast_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0,
                                                 int32_t c1, int32_t c2, int32_t c3, uint16_t o1, uint16_t o2,
                                                 uint16_t cta_mask, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_4d_multicast(slot, dst, mbar, c0, c1, c2, c3, o1, o2, cta_mask);
}
inline void vx_dxa_load_im2col_5d_multicast_hint(uint32_t slot, void* dst, uint64_t* mbar, int32_t c0,
                                                 int32_t c1, int32_t c2, int32_t c3, int32_t c4, uint16_t o1,
                                                 uint16_t o2, uint16_t o3, uint16_t cta_mask,
                                                 uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_5d_multicast(slot, dst, mbar, c0, c1, c2, c3, c4, o1, o2, o3, cta_mask);
}

// cp.async.bulk.prefetch.tensor.{1..5}d.L2.global.{tile,im2col}: the engine
// reads the box's in-bounds lines into the L2. No local-memory write, no
// mbarrier, and the warp does not wait.
inline void vx_dxa_prefetch_1d(uint32_t slot, int32_t c0) {
  __vx_dxa_tload(slot | VX_DXA_CTL_PREFETCH, 0, 0, c0, 0, 0, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_prefetch_2d(uint32_t slot, int32_t c0, int32_t c1) {
  __vx_dxa_tload(slot | VX_DXA_CTL_PREFETCH, 0, 0, c0, c1, 0, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_prefetch_3d(uint32_t slot, int32_t c0, int32_t c1, int32_t c2) {
  __vx_dxa_tload(slot | VX_DXA_CTL_PREFETCH, 0, 0, c0, c1, c2, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_prefetch_4d(uint32_t slot, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
  __vx_dxa_tload(slot | VX_DXA_CTL_PREFETCH, 0, 0, c0, c1, c2, c3, 0, 0, 0, 0, 0);
}
inline void vx_dxa_prefetch_5d(uint32_t slot, int32_t c0, int32_t c1, int32_t c2, int32_t c3, int32_t c4) {
  __vx_dxa_tload(slot | VX_DXA_CTL_PREFETCH, 0, 0, c0, c1, c2, c3, c4, 0, 0, 0, 0);
}
inline void vx_dxa_prefetch_im2col_3d(uint32_t slot, int32_t c0, int32_t c1, int32_t c2, uint16_t o1) {
  __vx_dxa_tload(slot | VX_DXA_CTL_PREFETCH, 0, 0, c0, c1, c2, 0, 0, o1, 0, 0, 0);
}
inline void vx_dxa_prefetch_im2col_4d(uint32_t slot, int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                                      uint16_t o1, uint16_t o2) {
  __vx_dxa_tload(slot | VX_DXA_CTL_PREFETCH, 0, 0, c0, c1, c2, c3, 0, o1, o2, 0, 0);
}
inline void vx_dxa_prefetch_im2col_5d(uint32_t slot, int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                                      int32_t c4, uint16_t o1, uint16_t o2, uint16_t o3) {
  __vx_dxa_tload(slot | VX_DXA_CTL_PREFETCH, 0, 0, c0, c1, c2, c3, c4, o1, o2, o3, 0);
}

// cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes: the
// engine copies `size` bytes from global memory at `gsrc` (a physical
// address, VX_MEM_PHYS) to `dst`, then applies complete_tx(size) to `mbar`.
// `dst` and `mbar` follow the tensor-load rules; `size` is a nonzero multiple
// of 16 and `gsrc` and `dst` are 16-byte aligned.
// funct3 = 3: rs1 = wgather(dst, mbar, gsrc, size), rs2 = x0 or wgather(0, cta_mask, 0, 0).
inline void vx_dxa_bulk_load(void* dst, const void* gsrc, uint32_t size, uint64_t* mbar) {
  const size_t a0 = vx_wgather((size_t)(uintptr_t)dst,
                               (size_t)(uintptr_t)mbar,
                               (size_t)(uintptr_t)gsrc,
                               (size_t)size);
  __asm__ volatile (
      ".insn r %0, 3, %1, x0, %2, x0\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0)
      : "memory");
}

// .multicast::cluster: lands in every rank set in `cta_mask` at the
// block-relative offsets of the issuer's local `dst` and `mbar`.
inline void vx_dxa_bulk_load_multicast(void* dst, const void* gsrc, uint32_t size, uint64_t* mbar,
                                       uint16_t cta_mask) {
  const size_t a0 = vx_wgather((size_t)(uintptr_t)dst,
                               (size_t)(uintptr_t)mbar,
                               (size_t)(uintptr_t)gsrc,
                               (size_t)size);
  const size_t a1 = vx_wgather((size_t)0, (size_t)cta_mask, (size_t)0, (size_t)0);
  __asm__ volatile (
      ".insn r %0, 3, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

// cp.async.bulk.prefetch.L2.global: reads [gsrc, gsrc + size) into the L2.
// funct3 = 4: rs1 = wgather(0, 0, gsrc, size), rs2 = x0.
inline void vx_dxa_prefetch_l2(const void* gsrc, uint32_t size) {
  const size_t a0 = vx_wgather((size_t)0, (size_t)0, (size_t)(uintptr_t)gsrc, (size_t)size);
  __asm__ volatile (
      ".insn r %0, 4, %1, x0, %2, x0\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0)
      : "memory");
}

// TMA stores and reductions, PTX cp.async.bulk{,.tensor}.global.shared::cta,
// cp.reduce.async.bulk{,.tensor}: the engine reads `src`, a local pointer
// into the issuer's own block, as the async proxy (order earlier stores to it
// with vx_fence_proxy_async()), and writes global memory (physical addresses,
// VX_MEM_PHYS) or, for the cluster reduction, another block of the cluster.
// Every form but the cluster reduction is a .bulk_group operation: it joins
// the warp's open group (vx_bulk_commit_group); `src` may be reused once
// vx_bulk_wait_group_read<N>() has released the group, and the writes are
// performed once vx_bulk_wait_group<N>() has; a later generic load of them
// also needs vx_fence_proxy_async_global(). Groups are tracked per warp.
// funct3 = 5: rs1 = wgather(src, a1, a2, a3), rs2 = wgather({ctl, c4}, 0, 0, 0),
// ctl = kind | op << 3 | type << 7 | slot << 11 (DXA_ST_CTL_* in dxa_meta.h);
// a1, a2, a3 = gdst, size, 0 (bulk store, global reduction), dst, size, mbar
// (cluster reduction), {c1, c0}, {c3, c2}, 0 (tensor forms).

inline uint32_t vx_dxa_st_ctl(uint32_t kind, uint32_t op, uint32_t type, uint32_t slot) {
  return ((kind & ((1u << DXA_ST_CTL_KIND_BITS) - 1)) << DXA_ST_CTL_KIND_LSB)
       | ((op & ((1u << DXA_ST_CTL_OP_BITS) - 1)) << DXA_ST_CTL_OP_LSB)
       | ((type & ((1u << DXA_ST_CTL_TYPE_BITS) - 1)) << DXA_ST_CTL_TYPE_LSB)
       | ((slot & ((1u << DXA_ST_CTL_SLOT_BITS) - 1)) << DXA_ST_CTL_SLOT_LSB);
}

inline void __vx_dxa_store(uint32_t ctl, const void* src, size_t a1, size_t a2, size_t a3, int32_t c4) {
  const size_t r1 = vx_wgather((size_t)(uintptr_t)src, a1, a2, a3);
  const size_t r2 = vx_wgather(((size_t)ctl << 32) | (size_t)(uint32_t)c4, (size_t)0, (size_t)0, (size_t)0);
  __asm__ volatile (
      ".insn r %0, 5, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(r1), "r"(r2)
      : "memory");
}

inline void __vx_dxa_store_tensor(uint32_t ctl, const void* src, int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                                  int32_t c4) {
  __vx_dxa_store(ctl, src, __vx_dxa_pair(c0, c1), __vx_dxa_pair(c2, c3), (size_t)0, c4);
}

// cp.async.bulk.global.shared::cta.bulk_group: `size` bytes from `src` to
// `gdst`; `size` is a nonzero multiple of 16, both addresses 16-byte aligned.
inline void vx_dxa_bulk_store(void* gdst, const void* src, uint32_t size) {
  __vx_dxa_store(vx_dxa_st_ctl(DXA_ST_BULK, 0, 0, 0), src, (size_t)(uintptr_t)gdst, (size_t)size, (size_t)0, 0);
}

// cp.async.bulk.tensor.{1..5}d.global.shared::cta.{tile,im2col_no_offs}.bulk_group:
// the box of descriptor `slot` at the coordinates, read from `src` in the
// layout a load of the same box writes; elements outside the tensor are not
// written. Tile mode: a coordinate below 0 is an error. The im2col forms take
// (c, w[, h[, d]], n) and the descriptor's im2col mode, all offsets 0.
inline void vx_dxa_store_1d(uint32_t slot, const void* src, int32_t c0) {
  __vx_dxa_store_tensor(vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, slot), src, c0, 0, 0, 0, 0);
}
inline void vx_dxa_store_2d(uint32_t slot, const void* src, int32_t c0, int32_t c1) {
  __vx_dxa_store_tensor(vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, slot), src, c0, c1, 0, 0, 0);
}
inline void vx_dxa_store_3d(uint32_t slot, const void* src, int32_t c0, int32_t c1, int32_t c2) {
  __vx_dxa_store_tensor(vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, slot), src, c0, c1, c2, 0, 0);
}
inline void vx_dxa_store_4d(uint32_t slot, const void* src, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
  __vx_dxa_store_tensor(vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, slot), src, c0, c1, c2, c3, 0);
}
inline void vx_dxa_store_5d(uint32_t slot, const void* src, int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                            int32_t c4) {
  __vx_dxa_store_tensor(vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, slot), src, c0, c1, c2, c3, c4);
}
inline void vx_dxa_store_im2col_3d(uint32_t slot, const void* src, int32_t c, int32_t w, int32_t n) {
  vx_dxa_store_3d(slot, src, c, w, n);
}
inline void vx_dxa_store_im2col_4d(uint32_t slot, const void* src, int32_t c, int32_t w, int32_t h, int32_t n) {
  vx_dxa_store_4d(slot, src, c, w, h, n);
}
inline void vx_dxa_store_im2col_5d(uint32_t slot, const void* src, int32_t c, int32_t w, int32_t h, int32_t d,
                                   int32_t n) {
  vx_dxa_store_5d(slot, src, c, w, h, d, n);
}

// cp.reduce.async.bulk.global.shared::cta.bulk_group.<op>.<type>:
// gdst[i] = op(gdst[i], src[i]) per element, each atomic, in no order, at the
// last-level cache (an L2 or L3 is required). `op` is DXA_RED_*, `type`
// DXA_RT_* (dxa_meta.h); the size and alignment rules of the bulk store apply.
//   and, or, xor   b32, b64
//   min, max       u32, s32, u64, s64, f16, bf16
//   add            u32, s32, u64, f32, f64, f16 and bf16 (.noftz)
//   inc, dec       u32
// f16, bf16, f32, f64, inc and dec need VX_CFG_EXT_FPAMO_ENABLE.
inline void vx_dxa_reduce_global(void* gdst, const void* src, uint32_t size, uint32_t op, uint32_t type) {
  __vx_dxa_store(vx_dxa_st_ctl(DXA_ST_RED_GLOBAL, op, type, 0), src, (size_t)(uintptr_t)gdst, (size_t)size,
                 (size_t)0, 0);
}

// cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.<op>.<type>:
// dst[i] = op(dst[i], src[i]) per element in another block of the cluster
// (a window address, or a local pointer into another block), each element
// applying complete_tx(element bytes) to `mbar`, which is 8-byte aligned and
// lies in the destination block. Not a bulk_group operation: `src` may be
// reused once the mbarrier's phase completes.
//   and, or, xor   b32
//   add, min, max  u32, s32
//   inc, dec       u32 (VX_CFG_EXT_FPAMO_ENABLE)
//   add            u64
inline void vx_dxa_reduce_cluster(void* dst, const void* src, uint32_t size, uint64_t* mbar, uint32_t op,
                                  uint32_t type) {
  __vx_dxa_store(vx_dxa_st_ctl(DXA_ST_RED_CLUSTER, op, type, 0), src, (size_t)(uintptr_t)dst, (size_t)size,
                 (size_t)(uintptr_t)mbar, 0);
}

// cp.reduce.async.bulk.tensor.{1..5}d.global.shared::cta.<op>.{tile,im2col_no_offs}.bulk_group:
// the tensor store's walk with the global reduction's per-element op. The
// type's size must be the descriptor's element size.
inline void vx_dxa_reduce_1d(uint32_t slot, const void* src, uint32_t op, uint32_t type, int32_t c0) {
  __vx_dxa_store_tensor(vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, type, slot), src, c0, 0, 0, 0, 0);
}
inline void vx_dxa_reduce_2d(uint32_t slot, const void* src, uint32_t op, uint32_t type, int32_t c0, int32_t c1) {
  __vx_dxa_store_tensor(vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, type, slot), src, c0, c1, 0, 0, 0);
}
inline void vx_dxa_reduce_3d(uint32_t slot, const void* src, uint32_t op, uint32_t type, int32_t c0, int32_t c1,
                             int32_t c2) {
  __vx_dxa_store_tensor(vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, type, slot), src, c0, c1, c2, 0, 0);
}
inline void vx_dxa_reduce_4d(uint32_t slot, const void* src, uint32_t op, uint32_t type, int32_t c0, int32_t c1,
                             int32_t c2, int32_t c3) {
  __vx_dxa_store_tensor(vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, type, slot), src, c0, c1, c2, c3, 0);
}
inline void vx_dxa_reduce_5d(uint32_t slot, const void* src, uint32_t op, uint32_t type, int32_t c0, int32_t c1,
                             int32_t c2, int32_t c3, int32_t c4) {
  __vx_dxa_store_tensor(vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, type, slot), src, c0, c1, c2, c3, c4);
}
inline void vx_dxa_reduce_im2col_3d(uint32_t slot, const void* src, uint32_t op, uint32_t type, int32_t c,
                                    int32_t w, int32_t n) {
  vx_dxa_reduce_3d(slot, src, op, type, c, w, n);
}
inline void vx_dxa_reduce_im2col_4d(uint32_t slot, const void* src, uint32_t op, uint32_t type, int32_t c,
                                    int32_t w, int32_t h, int32_t n) {
  vx_dxa_reduce_4d(slot, src, op, type, c, w, h, n);
}
inline void vx_dxa_reduce_im2col_5d(uint32_t slot, const void* src, uint32_t op, uint32_t type, int32_t c,
                                    int32_t w, int32_t h, int32_t d, int32_t n) {
  vx_dxa_reduce_5d(slot, src, op, type, c, w, h, d, n);
}

// Typed forms of every legal pair; f16 and bf16 elements are their bit patterns.
#define __VX_DXA_RED_G(op, ty, T, OP, RT)                                                          \
  inline void vx_dxa_reduce_global_##op##_##ty(T* gdst, const T* src, uint32_t size) {             \
    vx_dxa_reduce_global(gdst, src, size, OP, RT);                                                 \
  }
#define __VX_DXA_RED_C(op, ty, T, OP, RT)                                                          \
  inline void vx_dxa_reduce_cluster_##op##_##ty(T* dst, const T* src, uint32_t size, uint64_t* mbar) { \
    vx_dxa_reduce_cluster(dst, src, size, mbar, OP, RT);                                           \
  }
__VX_DXA_RED_G(and, b32, uint32_t, DXA_RED_AND, DXA_RT_U32)
__VX_DXA_RED_G(or,  b32, uint32_t, DXA_RED_OR,  DXA_RT_U32)
__VX_DXA_RED_G(xor, b32, uint32_t, DXA_RED_XOR, DXA_RT_U32)
__VX_DXA_RED_G(and, b64, uint64_t, DXA_RED_AND, DXA_RT_U64)
__VX_DXA_RED_G(or,  b64, uint64_t, DXA_RED_OR,  DXA_RT_U64)
__VX_DXA_RED_G(xor, b64, uint64_t, DXA_RED_XOR, DXA_RT_U64)
__VX_DXA_RED_G(min, u32, uint32_t, DXA_RED_MIN, DXA_RT_U32)
__VX_DXA_RED_G(min, s32, int32_t,  DXA_RED_MIN, DXA_RT_S32)
__VX_DXA_RED_G(min, u64, uint64_t, DXA_RED_MIN, DXA_RT_U64)
__VX_DXA_RED_G(min, s64, int64_t,  DXA_RED_MIN, DXA_RT_S64)
__VX_DXA_RED_G(min, f16, uint16_t, DXA_RED_MIN, DXA_RT_F16)
__VX_DXA_RED_G(min, bf16, uint16_t, DXA_RED_MIN, DXA_RT_BF16)
__VX_DXA_RED_G(max, u32, uint32_t, DXA_RED_MAX, DXA_RT_U32)
__VX_DXA_RED_G(max, s32, int32_t,  DXA_RED_MAX, DXA_RT_S32)
__VX_DXA_RED_G(max, u64, uint64_t, DXA_RED_MAX, DXA_RT_U64)
__VX_DXA_RED_G(max, s64, int64_t,  DXA_RED_MAX, DXA_RT_S64)
__VX_DXA_RED_G(max, f16, uint16_t, DXA_RED_MAX, DXA_RT_F16)
__VX_DXA_RED_G(max, bf16, uint16_t, DXA_RED_MAX, DXA_RT_BF16)
__VX_DXA_RED_G(add, u32, uint32_t, DXA_RED_ADD, DXA_RT_U32)
__VX_DXA_RED_G(add, s32, int32_t,  DXA_RED_ADD, DXA_RT_S32)
__VX_DXA_RED_G(add, u64, uint64_t, DXA_RED_ADD, DXA_RT_U64)
__VX_DXA_RED_G(add, f32, float,    DXA_RED_ADD, DXA_RT_F32)
__VX_DXA_RED_G(add, f64, double,   DXA_RED_ADD, DXA_RT_F64)
__VX_DXA_RED_G(add, f16, uint16_t, DXA_RED_ADD, DXA_RT_F16)
__VX_DXA_RED_G(add, bf16, uint16_t, DXA_RED_ADD, DXA_RT_BF16)
__VX_DXA_RED_G(inc, u32, uint32_t, DXA_RED_INC, DXA_RT_U32)
__VX_DXA_RED_G(dec, u32, uint32_t, DXA_RED_DEC, DXA_RT_U32)
__VX_DXA_RED_C(and, b32, uint32_t, DXA_RED_AND, DXA_RT_U32)
__VX_DXA_RED_C(or,  b32, uint32_t, DXA_RED_OR,  DXA_RT_U32)
__VX_DXA_RED_C(xor, b32, uint32_t, DXA_RED_XOR, DXA_RT_U32)
__VX_DXA_RED_C(add, u32, uint32_t, DXA_RED_ADD, DXA_RT_U32)
__VX_DXA_RED_C(add, s32, int32_t,  DXA_RED_ADD, DXA_RT_S32)
__VX_DXA_RED_C(min, u32, uint32_t, DXA_RED_MIN, DXA_RT_U32)
__VX_DXA_RED_C(min, s32, int32_t,  DXA_RED_MIN, DXA_RT_S32)
__VX_DXA_RED_C(max, u32, uint32_t, DXA_RED_MAX, DXA_RT_U32)
__VX_DXA_RED_C(max, s32, int32_t,  DXA_RED_MAX, DXA_RT_S32)
__VX_DXA_RED_C(inc, u32, uint32_t, DXA_RED_INC, DXA_RT_U32)
__VX_DXA_RED_C(dec, u32, uint32_t, DXA_RED_DEC, DXA_RT_U32)
__VX_DXA_RED_C(add, u64, uint64_t, DXA_RED_ADD, DXA_RT_U64)
#undef __VX_DXA_RED_G
#undef __VX_DXA_RED_C

// cp.async.bulk.commit_group: closes the warp's open group, even an empty one.
// funct3 = 6, rs1 field = op, rs2 field = N; no register is read.
inline void vx_bulk_commit_group(void) {
  __asm__ volatile (".insn r %0, 6, %1, x0, x0, x0" :: "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7) : "memory");
}

// commit_group, then cp.async.bulk.wait_group 0.
inline void vx_bulk_wait_all(void) {
  vx_bulk_commit_group();
  __asm__ volatile (".insn r %0, 6, %1, x0, x1, x0" :: "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7) : "memory");
}

// fence.proxy.async.global: vx_fence_proxy_async(), and the warp's earlier
// global stores reach the level the DXA reads and writes, and the core's data
// cache is invalidated, so that a later generic load sees a DXA write that a
// wait_group has released. vx_fence_proxy_async_all is fence.proxy.async with
// no state space, the same fence.
inline void vx_fence_proxy_async_global(void) {
  __asm__ volatile (".insn r %0, 7, 1, x0, x1, x0" :: "i"(RISCV_CUSTOM2) : "memory");
}
inline void vx_fence_proxy_async_all(void) {
  __asm__ volatile (".insn r %0, 7, 1, x0, x2, x0" :: "i"(RISCV_CUSTOM2) : "memory");
}
#endif // VX_CFG_EXT_MBAR_ENABLE

// Multicast DXA issues read GMEM once and replay SMEM writes to multiple
// CTAs within the same SM core.

// 1D multicast: rs2 = wgather(0, 0, 0, cta_mask)
inline void vx_dxa_issue_1d_multicast_wg(uint32_t desc_slot,
                                          uint32_t barrier_id,
                                          const void* smem_addr,
                                          uint32_t coord0,
                                          uint32_t cta_mask) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)0u);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)0,
                                            (size_t)0,
                                            (size_t)0,
                                            (size_t)cta_mask);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

// 2D multicast: rs2 = wgather(0, 0, 0, cta_mask)
inline void vx_dxa_issue_2d_multicast_wg(uint32_t desc_slot,
                                          uint32_t barrier_id,
                                          const void* smem_addr,
                                          uint32_t coord0,
                                          uint32_t coord1,
                                          uint32_t cta_mask) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)0,
                                            (size_t)0,
                                            (size_t)0,
                                            (size_t)cta_mask);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

// 3D multicast: rs2 = wgather(coord2, 0, 0, cta_mask)
inline void vx_dxa_issue_3d_multicast_wg(uint32_t desc_slot,
                                          uint32_t barrier_id,
                                          const void* smem_addr,
                                          uint32_t coord0,
                                          uint32_t coord1,
                                          uint32_t coord2,
                                          uint32_t cta_mask) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)0,
                                            (size_t)0,
                                            (size_t)cta_mask);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

// 4D multicast: rs2 = wgather(coord2, coord3, 0, cta_mask)
inline void vx_dxa_issue_4d_multicast_wg(uint32_t desc_slot,
                                          uint32_t barrier_id,
                                          const void* smem_addr,
                                          uint32_t coord0,
                                          uint32_t coord1,
                                          uint32_t coord2,
                                          uint32_t coord3,
                                          uint32_t cta_mask) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)coord3,
                                            (size_t)0,
                                            (size_t)cta_mask);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

// 5D multicast: rs2 = wgather(coord2, coord3, coord4, cta_mask)
inline void vx_dxa_issue_5d_multicast_wg(uint32_t desc_slot,
                                          uint32_t barrier_id,
                                          const void* smem_addr,
                                          uint32_t coord0,
                                          uint32_t coord1,
                                          uint32_t coord2,
                                          uint32_t coord3,
                                          uint32_t coord4,
                                          uint32_t cta_mask) {
  const uint32_t meta = vx_dxa_pack_meta(desc_slot, barrier_id);
  const uint32_t a0 = (uint32_t)vx_wgather((size_t)(uintptr_t)smem_addr,
                                            (size_t)meta,
                                            (size_t)coord0,
                                            (size_t)coord1);
  const uint32_t a1 = (uint32_t)vx_wgather((size_t)coord2,
                                            (size_t)coord3,
                                            (size_t)coord4,
                                            (size_t)cta_mask);
  __asm__ volatile (
      ".insn r %0, 0, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

#ifdef VX_CFG_EXT_MBAR_ENABLE
// Cluster multicast, like cp.async.bulk.tensor.shared::cluster with
// .multicast::cluster and .mbarrier::complete_tx::bytes: one read of the box,
// landing in every block of the issuer's cluster whose rank r has bit r set in
// `cta_mask`, at the block-relative offset of `smem_addr`; the engine then
// applies complete_tx(box bytes) to the object at the block-relative offset
// of `mbar` in that block. `smem_addr` and `mbar` are the issuer's own local
// pointers, not window addresses. The issuer may leave itself out of the
// mask, and then its own object is not credited.

inline void vx_dxa_issue_1d_mbar_multicast(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                           uint32_t cta_mask, uint32_t coord0) {
  vx_dxa_issue_1d_multicast_wg(desc_slot, __vx_dxa_mbar_payload(mbar), smem_addr, coord0, cta_mask);
}

inline void vx_dxa_issue_2d_mbar_multicast(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                           uint32_t cta_mask, uint32_t coord0, uint32_t coord1) {
  vx_dxa_issue_2d_multicast_wg(desc_slot, __vx_dxa_mbar_payload(mbar), smem_addr, coord0, coord1, cta_mask);
}

inline void vx_dxa_issue_3d_mbar_multicast(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                           uint32_t cta_mask, uint32_t coord0, uint32_t coord1,
                                           uint32_t coord2) {
  vx_dxa_issue_3d_multicast_wg(desc_slot, __vx_dxa_mbar_payload(mbar), smem_addr,
                               coord0, coord1, coord2, cta_mask);
}

inline void vx_dxa_issue_4d_mbar_multicast(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                           uint32_t cta_mask, uint32_t coord0, uint32_t coord1,
                                           uint32_t coord2, uint32_t coord3) {
  vx_dxa_issue_4d_multicast_wg(desc_slot, __vx_dxa_mbar_payload(mbar), smem_addr,
                               coord0, coord1, coord2, coord3, cta_mask);
}

inline void vx_dxa_issue_5d_mbar_multicast(uint32_t desc_slot, uint64_t* mbar, void* smem_addr,
                                           uint32_t cta_mask, uint32_t coord0, uint32_t coord1,
                                           uint32_t coord2, uint32_t coord3, uint32_t coord4) {
  vx_dxa_issue_5d_multicast_wg(desc_slot, __vx_dxa_mbar_payload(mbar), smem_addr,
                               coord0, coord1, coord2, coord3, coord4, cta_mask);
}
#endif // VX_CFG_EXT_MBAR_ENABLE

#ifdef __cplusplus
}
#endif

// ════════════════════════════════════════════════════════════════════════
//  C++ helper API — vortex::dxa_multicast_Nd
//
//  Wraps the two-barrier idiom so the kernel author cannot violate the
//  mask <-> expect_tx invariant by hand. The helper:
//    * registers exactly one tx event on the per-CTA local_bar
//      (in the constructor — call from the loader warp only)
//    * rendezvouses K cluster members on the group_bar in sync_and_issue()
//      and (for rank 0) fires the multicast.
//
//  The helper deliberately has NO wait() method. The kernel must call
//  local_bar.arrive_and_wait() from ALL warps of the CTA at top scope
//  (not just the loader warp) because arrive_and_wait waits for both
//  arrivals == warps_per_CTA AND events_r == 0.
//
//  Requires the dispatcher to guarantee that the K cluster members are
//  co-resident on one core in contiguous cta_local_id slots starting at
//  the issuer's id. Without this contract the helper is correctness-fragile
//  on grids larger than one cluster.
// ════════════════════════════════════════════════════════════════════════
#ifdef __cplusplus

#include <vx_barrier.h>
#include <vx_spawn2.h>   // get_local_group_id

namespace vortex {

// num_members (= K, the cluster size / multicast group size) must be
// supplied explicitly by the caller.

class dxa_multicast_1d {
public:
  dxa_multicast_1d(uint32_t desc_slot, uint32_t num_members,
                   vortex::barrier&       local_bar,
                   vortex::group_barrier& group_bar)
    : desc_slot_(desc_slot), mask_((1u << num_members) - 1u),
      local_bar_(local_bar), group_bar_(group_bar) {
    local_bar_.expect_tx(1);
  }
  void sync_and_issue(const void* my_smem_offset, uint32_t coord0) {
    group_bar_.arrive_and_wait();
    // Only cluster rank-0 issues. get_cluster_rank() (CTA_ID % cluster_size)
    // is 0 for every cluster's issuer; get_local_group_id() is 0 only for the
    // very first CTA globally, causing later clusters to deadlock on local_bar.
    if (::get_cluster_rank() == 0) {
      vx_dxa_issue_1d_multicast_wg(desc_slot_, local_bar_.id(),
                                    my_smem_offset, coord0, mask_);
    }
  }
private:
  uint32_t                desc_slot_;
  uint32_t                mask_;
  vortex::barrier&        local_bar_;
  vortex::group_barrier&  group_bar_;
};

class dxa_multicast_2d {
public:
  dxa_multicast_2d(uint32_t desc_slot, uint32_t num_members,
                   vortex::barrier&       local_bar,
                   vortex::group_barrier& group_bar)
    : desc_slot_(desc_slot), mask_((1u << num_members) - 1u),
      local_bar_(local_bar), group_bar_(group_bar) {
    local_bar_.expect_tx(1);
  }
  void sync_and_issue(const void* my_smem_offset,
                      uint32_t coord0, uint32_t coord1) {
    group_bar_.arrive_and_wait();
    // Only cluster rank-0 issues. get_cluster_rank() (CTA_ID % cluster_size)
    // is 0 for every cluster's issuer; get_local_group_id() is 0 only for the
    // very first CTA globally, causing later clusters to deadlock on local_bar.
    if (::get_cluster_rank() == 0) {
      vx_dxa_issue_2d_multicast_wg(desc_slot_, local_bar_.id(),
                                    my_smem_offset, coord0, coord1, mask_);
    }
  }
private:
  uint32_t                desc_slot_;
  uint32_t                mask_;
  vortex::barrier&        local_bar_;
  vortex::group_barrier&  group_bar_;
};

class dxa_multicast_3d {
public:
  dxa_multicast_3d(uint32_t desc_slot, uint32_t num_members,
                   vortex::barrier&       local_bar,
                   vortex::group_barrier& group_bar)
    : desc_slot_(desc_slot), mask_((1u << num_members) - 1u),
      local_bar_(local_bar), group_bar_(group_bar) {
    local_bar_.expect_tx(1);
  }
  void sync_and_issue(const void* my_smem_offset,
                      uint32_t coord0, uint32_t coord1, uint32_t coord2) {
    group_bar_.arrive_and_wait();
    // Only cluster rank-0 issues. get_cluster_rank() (CTA_ID % cluster_size)
    // is 0 for every cluster's issuer; get_local_group_id() is 0 only for the
    // very first CTA globally, causing later clusters to deadlock on local_bar.
    if (::get_cluster_rank() == 0) {
      vx_dxa_issue_3d_multicast_wg(desc_slot_, local_bar_.id(),
                                    my_smem_offset, coord0, coord1, coord2,
                                    mask_);
    }
  }
private:
  uint32_t                desc_slot_;
  uint32_t                mask_;
  vortex::barrier&        local_bar_;
  vortex::group_barrier&  group_bar_;
};

class dxa_multicast_4d {
public:
  dxa_multicast_4d(uint32_t desc_slot, uint32_t num_members,
                   vortex::barrier&       local_bar,
                   vortex::group_barrier& group_bar)
    : desc_slot_(desc_slot), mask_((1u << num_members) - 1u),
      local_bar_(local_bar), group_bar_(group_bar) {
    local_bar_.expect_tx(1);
  }
  void sync_and_issue(const void* my_smem_offset,
                      uint32_t coord0, uint32_t coord1,
                      uint32_t coord2, uint32_t coord3) {
    group_bar_.arrive_and_wait();
    // Only cluster rank-0 issues. get_cluster_rank() (CTA_ID % cluster_size)
    // is 0 for every cluster's issuer; get_local_group_id() is 0 only for the
    // very first CTA globally, causing later clusters to deadlock on local_bar.
    if (::get_cluster_rank() == 0) {
      vx_dxa_issue_4d_multicast_wg(desc_slot_, local_bar_.id(),
                                    my_smem_offset,
                                    coord0, coord1, coord2, coord3, mask_);
    }
  }
private:
  uint32_t                desc_slot_;
  uint32_t                mask_;
  vortex::barrier&        local_bar_;
  vortex::group_barrier&  group_bar_;
};

class dxa_multicast_5d {
public:
  dxa_multicast_5d(uint32_t desc_slot, uint32_t num_members,
                   vortex::barrier&       local_bar,
                   vortex::group_barrier& group_bar)
    : desc_slot_(desc_slot), mask_((1u << num_members) - 1u),
      local_bar_(local_bar), group_bar_(group_bar) {
    local_bar_.expect_tx(1);
  }
  void sync_and_issue(const void* my_smem_offset,
                      uint32_t coord0, uint32_t coord1,
                      uint32_t coord2, uint32_t coord3, uint32_t coord4) {
    group_bar_.arrive_and_wait();
    // Only cluster rank-0 issues. get_cluster_rank() (CTA_ID % cluster_size)
    // is 0 for every cluster's issuer; get_local_group_id() is 0 only for the
    // very first CTA globally, causing later clusters to deadlock on local_bar.
    if (::get_cluster_rank() == 0) {
      vx_dxa_issue_5d_multicast_wg(desc_slot_, local_bar_.id(),
                                    my_smem_offset, coord0, coord1, coord2,
                                    coord3, coord4, mask_);
    }
  }
private:
  uint32_t                desc_slot_;
  uint32_t                mask_;
  vortex::barrier&        local_bar_;
  vortex::group_barrier&  group_bar_;
};

#ifdef VX_CFG_EXT_MBAR_ENABLE
// ════════════════════════════════════════════════════════════════════════
//  C++ helper API — vortex::dxa_multicast_cluster_Nd
//
//  The cluster-multicast idiom on mbarriers, for a cluster of any placement.
//  Every warp of every block of the cluster calls sync_and_issue() once per
//  copy: each block whose rank is in the mask arms its own mbarrier with
//  arrive_tx(box_bytes) from one thread, the cluster barrier then orders
//  every receiver's declaration before the copy, and warp 0 of rank `issuer`
//  fires the multicast. The kernel initializes `mbar` beforehand, from
//  thread 0 or followed by __syncthreads(), and every receiver waits on its
//  own object afterwards, as with cuda::barrier.
// ════════════════════════════════════════════════════════════════════════

class dxa_multicast_cluster_base {
protected:
  dxa_multicast_cluster_base(uint32_t desc_slot, uint32_t cta_mask, uint32_t issuer,
                             uint64_t* mbar, uint32_t box_bytes)
    : desc_slot_(desc_slot), mask_(cta_mask), issuer_(issuer), mbar_(mbar),
      box_bytes_(box_bytes) {}

  // True in the warp that issues.
  bool arm_and_sync() const {
    const uint32_t rank = vx_cluster_block_rank();
    const bool warp0 = (::get_sub_group_id() == 0);
    if (warp0 && vx_thread_id() == 0 && ((mask_ >> rank) & 1u))
      vx_mbar_arrive_tx(mbar_, box_bytes_);
    vx_cluster_sync();
    return warp0 && rank == issuer_;
  }

  uint32_t  desc_slot_;
  uint32_t  mask_;
  uint32_t  issuer_;
  uint64_t* mbar_;
  uint32_t  box_bytes_;
};

class dxa_multicast_cluster_1d : dxa_multicast_cluster_base {
public:
  dxa_multicast_cluster_1d(uint32_t desc_slot, uint32_t cta_mask, uint32_t issuer,
                           uint64_t* mbar, uint32_t box_bytes)
    : dxa_multicast_cluster_base(desc_slot, cta_mask, issuer, mbar, box_bytes) {}
  void sync_and_issue(void* smem, uint32_t coord0) const {
    if (arm_and_sync())
      vx_dxa_issue_1d_mbar_multicast(desc_slot_, mbar_, smem, mask_, coord0);
  }
};

class dxa_multicast_cluster_2d : dxa_multicast_cluster_base {
public:
  dxa_multicast_cluster_2d(uint32_t desc_slot, uint32_t cta_mask, uint32_t issuer,
                           uint64_t* mbar, uint32_t box_bytes)
    : dxa_multicast_cluster_base(desc_slot, cta_mask, issuer, mbar, box_bytes) {}
  void sync_and_issue(void* smem, uint32_t coord0, uint32_t coord1) const {
    if (arm_and_sync())
      vx_dxa_issue_2d_mbar_multicast(desc_slot_, mbar_, smem, mask_, coord0, coord1);
  }
};

class dxa_multicast_cluster_3d : dxa_multicast_cluster_base {
public:
  dxa_multicast_cluster_3d(uint32_t desc_slot, uint32_t cta_mask, uint32_t issuer,
                           uint64_t* mbar, uint32_t box_bytes)
    : dxa_multicast_cluster_base(desc_slot, cta_mask, issuer, mbar, box_bytes) {}
  void sync_and_issue(void* smem, uint32_t coord0, uint32_t coord1, uint32_t coord2) const {
    if (arm_and_sync())
      vx_dxa_issue_3d_mbar_multicast(desc_slot_, mbar_, smem, mask_, coord0, coord1, coord2);
  }
};

class dxa_multicast_cluster_4d : dxa_multicast_cluster_base {
public:
  dxa_multicast_cluster_4d(uint32_t desc_slot, uint32_t cta_mask, uint32_t issuer,
                           uint64_t* mbar, uint32_t box_bytes)
    : dxa_multicast_cluster_base(desc_slot, cta_mask, issuer, mbar, box_bytes) {}
  void sync_and_issue(void* smem, uint32_t coord0, uint32_t coord1, uint32_t coord2,
                      uint32_t coord3) const {
    if (arm_and_sync())
      vx_dxa_issue_4d_mbar_multicast(desc_slot_, mbar_, smem, mask_, coord0, coord1, coord2, coord3);
  }
};

class dxa_multicast_cluster_5d : dxa_multicast_cluster_base {
public:
  dxa_multicast_cluster_5d(uint32_t desc_slot, uint32_t cta_mask, uint32_t issuer,
                           uint64_t* mbar, uint32_t box_bytes)
    : dxa_multicast_cluster_base(desc_slot, cta_mask, issuer, mbar, box_bytes) {}
  void sync_and_issue(void* smem, uint32_t coord0, uint32_t coord1, uint32_t coord2,
                      uint32_t coord3, uint32_t coord4) const {
    if (arm_and_sync())
      vx_dxa_issue_5d_mbar_multicast(desc_slot_, mbar_, smem, mask_, coord0, coord1, coord2,
                                     coord3, coord4);
  }
};
#endif // VX_CFG_EXT_MBAR_ENABLE

} // namespace vortex

#endif // __cplusplus

#if defined(__cplusplus) && defined(VX_CFG_EXT_MBAR_ENABLE)
// cp.async.bulk.wait_group N: holds the warp until every committed group of
// the warp but the N most recent has its writes performed; operations not
// yet committed are not waited for. N >= DXA_BG_GROUPS returns at once.
template <unsigned N>
inline void vx_bulk_wait_group() {
  static_assert(N < 32, "wait_group N is 0-31");
  __asm__ volatile (".insn r %0, 6, %1, x0, x1, x%2" :: "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "i"(N)
                    : "memory");
}

// cp.async.bulk.wait_group.read N: the same, until the groups' sources have been read.
template <unsigned N>
inline void vx_bulk_wait_group_read() {
  static_assert(N < 32, "wait_group.read N is 0-31");
  __asm__ volatile (".insn r %0, 6, %1, x0, x2, x%2" :: "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "i"(N)
                    : "memory");
}
#endif

#if defined(__cplusplus) && defined(VX_CFG_EXT_MBAR_ENABLE)
// ════════════════════════════════════════════════════════════════════════
//  Tensor maps in memory (PTX §5.5.8, the CUtensorMap model)
//
//  The map forms of the PTX tensor operations name a vx_tensor_map_t
//  (dxa_tmap.h) in global memory, 64-byte aligned, physical (VX_MEM_PHYS), in
//  place of a descriptor slot: a kernel-argument member (`&arg->map`, the
//  __grid_constant__ form), a .const map or a global buffer. The DXA reads it
//  into its descriptor cache at the level below the L1D; every launch empties
//  the cache. A map written by the kernel is published with a release
//  (vx_tensormap_cp_fenceproxy or vx_fence_proxy_tensormap_release) and
//  picked up by each block with vx_fence_proxy_tensormap_acquire.
//  funct3 2: rs2 lane 0 ctl bit 5 = map, rs2 lane 2 = the map; funct3 5: ctl
//  bit 15 = map, rs2 lane 1 = the map; a map-form reduction's type is the map's.
//  As every DXA op, these take their operands from thread 0, which must be
//  active (prefetch and acquire read the map address from its rs1).
// ════════════════════════════════════════════════════════════════════════

inline void __vx_dxa_tload_map(uint32_t ctl, const vx_tensor_map_t* map, void* dst, uint64_t* mbar,
                               int32_t c0, int32_t c1, int32_t c2, int32_t c3, int32_t c4,
                               uint32_t o1, uint32_t o2, uint32_t o3, uint32_t cta_mask) {
  const size_t a0 = vx_wgather((size_t)(uintptr_t)dst,
                               (size_t)(uintptr_t)mbar,
                               __vx_dxa_pair(c0, c1),
                               __vx_dxa_pair(c2, c3));
  const size_t a1 = vx_wgather(((size_t)(ctl | (1u << DXA_TL_CTL_MAP_LSB)) << 32) | (size_t)(uint32_t)c4,
                               ((size_t)(o3 & 0xFFFFu) << 48) | ((size_t)(o2 & 0xFFFFu) << 32)
                                 | ((size_t)(o1 & 0xFFFFu) << 16) | (size_t)(cta_mask & 0xFFFFu),
                               (size_t)(uintptr_t)map,
                               (size_t)0);
  __asm__ volatile (
      ".insn r %0, 2, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a0), "r"(a1)
      : "memory");
}

inline void vx_dxa_load_1d(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, 0, 0, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_load_2d(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0, int32_t c1) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, 0, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_load_3d(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                           int32_t c2) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_load_4d(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                           int32_t c2, int32_t c3) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, c3, 0, 0, 0, 0, 0);
}
inline void vx_dxa_load_5d(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                           int32_t c2, int32_t c3, int32_t c4) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, c3, c4, 0, 0, 0, 0);
}

inline void vx_dxa_load_1d_multicast(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                     uint16_t cta_mask) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, 0, 0, 0, 0, 0, 0, 0, cta_mask);
}
inline void vx_dxa_load_2d_multicast(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                     int32_t c1, uint16_t cta_mask) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, 0, 0, 0, 0, 0, 0, cta_mask);
}
inline void vx_dxa_load_3d_multicast(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                     int32_t c1, int32_t c2, uint16_t cta_mask) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, 0, 0, 0, 0, 0, cta_mask);
}
inline void vx_dxa_load_4d_multicast(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                     int32_t c1, int32_t c2, int32_t c3, uint16_t cta_mask) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, c3, 0, 0, 0, 0, cta_mask);
}
inline void vx_dxa_load_5d_multicast(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                     int32_t c1, int32_t c2, int32_t c3, int32_t c4, uint16_t cta_mask) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, c3, c4, 0, 0, 0, cta_mask);
}

inline void vx_dxa_load_im2col_3d(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                  int32_t c1, int32_t c2, uint16_t o1) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, 0, 0, o1, 0, 0, 0);
}
inline void vx_dxa_load_im2col_4d(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                  int32_t c1, int32_t c2, int32_t c3, uint16_t o1, uint16_t o2) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, c3, 0, o1, o2, 0, 0);
}
inline void vx_dxa_load_im2col_5d(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                  int32_t c1, int32_t c2, int32_t c3, int32_t c4, uint16_t o1, uint16_t o2,
                                  uint16_t o3) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, c3, c4, o1, o2, o3, 0);
}
inline void vx_dxa_load_im2col_3d_multicast(const vx_tensor_map_t* map, void* dst, uint64_t* mbar,
                                            int32_t c0, int32_t c1, int32_t c2, uint16_t o1,
                                            uint16_t cta_mask) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, 0, 0, o1, 0, 0, cta_mask);
}
inline void vx_dxa_load_im2col_4d_multicast(const vx_tensor_map_t* map, void* dst, uint64_t* mbar,
                                            int32_t c0, int32_t c1, int32_t c2, int32_t c3, uint16_t o1,
                                            uint16_t o2, uint16_t cta_mask) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, c3, 0, o1, o2, 0, cta_mask);
}
inline void vx_dxa_load_im2col_5d_multicast(const vx_tensor_map_t* map, void* dst, uint64_t* mbar,
                                            int32_t c0, int32_t c1, int32_t c2, int32_t c3, int32_t c4,
                                            uint16_t o1, uint16_t o2, uint16_t o3, uint16_t cta_mask) {
  __vx_dxa_tload_map(0, map, dst, mbar, c0, c1, c2, c3, c4, o1, o2, o3, cta_mask);
}

// The .L2::cache_hint forms drop the hint.
inline void vx_dxa_load_1d_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_1d(map, dst, mbar, c0);
}
inline void vx_dxa_load_2d_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_2d(map, dst, mbar, c0, c1);
}
inline void vx_dxa_load_3d_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                int32_t c2, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_3d(map, dst, mbar, c0, c1, c2);
}
inline void vx_dxa_load_4d_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                int32_t c2, int32_t c3, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_4d(map, dst, mbar, c0, c1, c2, c3);
}
inline void vx_dxa_load_5d_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0, int32_t c1,
                                int32_t c2, int32_t c3, int32_t c4, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_5d(map, dst, mbar, c0, c1, c2, c3, c4);
}
inline void vx_dxa_load_1d_multicast_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                          uint16_t cta_mask, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_1d_multicast(map, dst, mbar, c0, cta_mask);
}
inline void vx_dxa_load_2d_multicast_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                          int32_t c1, uint16_t cta_mask, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_2d_multicast(map, dst, mbar, c0, c1, cta_mask);
}
inline void vx_dxa_load_3d_multicast_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                          int32_t c1, int32_t c2, uint16_t cta_mask, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_3d_multicast(map, dst, mbar, c0, c1, c2, cta_mask);
}
inline void vx_dxa_load_4d_multicast_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                          int32_t c1, int32_t c2, int32_t c3, uint16_t cta_mask,
                                          uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_4d_multicast(map, dst, mbar, c0, c1, c2, c3, cta_mask);
}
inline void vx_dxa_load_5d_multicast_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                          int32_t c1, int32_t c2, int32_t c3, int32_t c4, uint16_t cta_mask,
                                          uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_5d_multicast(map, dst, mbar, c0, c1, c2, c3, c4, cta_mask);
}
inline void vx_dxa_load_im2col_3d_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                       int32_t c1, int32_t c2, uint16_t o1, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_3d(map, dst, mbar, c0, c1, c2, o1);
}
inline void vx_dxa_load_im2col_4d_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                       int32_t c1, int32_t c2, int32_t c3, uint16_t o1, uint16_t o2,
                                       uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_4d(map, dst, mbar, c0, c1, c2, c3, o1, o2);
}
inline void vx_dxa_load_im2col_5d_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar, int32_t c0,
                                       int32_t c1, int32_t c2, int32_t c3, int32_t c4, uint16_t o1, uint16_t o2,
                                       uint16_t o3, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_5d(map, dst, mbar, c0, c1, c2, c3, c4, o1, o2, o3);
}
inline void vx_dxa_load_im2col_3d_multicast_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar,
                                                 int32_t c0, int32_t c1, int32_t c2, uint16_t o1,
                                                 uint16_t cta_mask, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_3d_multicast(map, dst, mbar, c0, c1, c2, o1, cta_mask);
}
inline void vx_dxa_load_im2col_4d_multicast_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar,
                                                 int32_t c0, int32_t c1, int32_t c2, int32_t c3, uint16_t o1,
                                                 uint16_t o2, uint16_t cta_mask, uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_4d_multicast(map, dst, mbar, c0, c1, c2, c3, o1, o2, cta_mask);
}
inline void vx_dxa_load_im2col_5d_multicast_hint(const vx_tensor_map_t* map, void* dst, uint64_t* mbar,
                                                 int32_t c0, int32_t c1, int32_t c2, int32_t c3, int32_t c4,
                                                 uint16_t o1, uint16_t o2, uint16_t o3, uint16_t cta_mask,
                                                 uint64_t l2_hint) {
  (void)l2_hint;
  vx_dxa_load_im2col_5d_multicast(map, dst, mbar, c0, c1, c2, c3, c4, o1, o2, o3, cta_mask);
}

inline void vx_dxa_prefetch_1d(const vx_tensor_map_t* map, int32_t c0) {
  __vx_dxa_tload_map(VX_DXA_CTL_PREFETCH, map, 0, 0, c0, 0, 0, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_prefetch_2d(const vx_tensor_map_t* map, int32_t c0, int32_t c1) {
  __vx_dxa_tload_map(VX_DXA_CTL_PREFETCH, map, 0, 0, c0, c1, 0, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_prefetch_3d(const vx_tensor_map_t* map, int32_t c0, int32_t c1, int32_t c2) {
  __vx_dxa_tload_map(VX_DXA_CTL_PREFETCH, map, 0, 0, c0, c1, c2, 0, 0, 0, 0, 0, 0);
}
inline void vx_dxa_prefetch_4d(const vx_tensor_map_t* map, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
  __vx_dxa_tload_map(VX_DXA_CTL_PREFETCH, map, 0, 0, c0, c1, c2, c3, 0, 0, 0, 0, 0);
}
inline void vx_dxa_prefetch_5d(const vx_tensor_map_t* map, int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                               int32_t c4) {
  __vx_dxa_tload_map(VX_DXA_CTL_PREFETCH, map, 0, 0, c0, c1, c2, c3, c4, 0, 0, 0, 0);
}
inline void vx_dxa_prefetch_im2col_3d(const vx_tensor_map_t* map, int32_t c0, int32_t c1, int32_t c2,
                                      uint16_t o1) {
  __vx_dxa_tload_map(VX_DXA_CTL_PREFETCH, map, 0, 0, c0, c1, c2, 0, 0, o1, 0, 0, 0);
}
inline void vx_dxa_prefetch_im2col_4d(const vx_tensor_map_t* map, int32_t c0, int32_t c1, int32_t c2,
                                      int32_t c3, uint16_t o1, uint16_t o2) {
  __vx_dxa_tload_map(VX_DXA_CTL_PREFETCH, map, 0, 0, c0, c1, c2, c3, 0, o1, o2, 0, 0);
}
inline void vx_dxa_prefetch_im2col_5d(const vx_tensor_map_t* map, int32_t c0, int32_t c1, int32_t c2,
                                      int32_t c3, int32_t c4, uint16_t o1, uint16_t o2, uint16_t o3) {
  __vx_dxa_tload_map(VX_DXA_CTL_PREFETCH, map, 0, 0, c0, c1, c2, c3, c4, o1, o2, o3, 0);
}

inline void __vx_dxa_store_tensor_map(uint32_t ctl, const vx_tensor_map_t* map, const void* src, int32_t c0,
                                      int32_t c1, int32_t c2, int32_t c3, int32_t c4) {
  const size_t r1 = vx_wgather((size_t)(uintptr_t)src, __vx_dxa_pair(c0, c1), __vx_dxa_pair(c2, c3), (size_t)0);
  const size_t r2 = vx_wgather(((size_t)(ctl | (1u << DXA_ST_CTL_MAP_LSB)) << 32) | (size_t)(uint32_t)c4,
                               (size_t)(uintptr_t)map, (size_t)0, (size_t)0);
  __asm__ volatile (
      ".insn r %0, 5, %1, x0, %2, %3\n\t"
      :
      : "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(r1), "r"(r2)
      : "memory");
}

inline void vx_dxa_store_1d(const vx_tensor_map_t* map, const void* src, int32_t c0) {
  __vx_dxa_store_tensor_map(vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, 0), map, src, c0, 0, 0, 0, 0);
}
inline void vx_dxa_store_2d(const vx_tensor_map_t* map, const void* src, int32_t c0, int32_t c1) {
  __vx_dxa_store_tensor_map(vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, 0), map, src, c0, c1, 0, 0, 0);
}
inline void vx_dxa_store_3d(const vx_tensor_map_t* map, const void* src, int32_t c0, int32_t c1, int32_t c2) {
  __vx_dxa_store_tensor_map(vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, 0), map, src, c0, c1, c2, 0, 0);
}
inline void vx_dxa_store_4d(const vx_tensor_map_t* map, const void* src, int32_t c0, int32_t c1, int32_t c2,
                            int32_t c3) {
  __vx_dxa_store_tensor_map(vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, 0), map, src, c0, c1, c2, c3, 0);
}
inline void vx_dxa_store_5d(const vx_tensor_map_t* map, const void* src, int32_t c0, int32_t c1, int32_t c2,
                            int32_t c3, int32_t c4) {
  __vx_dxa_store_tensor_map(vx_dxa_st_ctl(DXA_ST_TENSOR, 0, 0, 0), map, src, c0, c1, c2, c3, c4);
}
inline void vx_dxa_store_im2col_3d(const vx_tensor_map_t* map, const void* src, int32_t c, int32_t w, int32_t n) {
  vx_dxa_store_3d(map, src, c, w, n);
}
inline void vx_dxa_store_im2col_4d(const vx_tensor_map_t* map, const void* src, int32_t c, int32_t w, int32_t h,
                                   int32_t n) {
  vx_dxa_store_4d(map, src, c, w, h, n);
}
inline void vx_dxa_store_im2col_5d(const vx_tensor_map_t* map, const void* src, int32_t c, int32_t w, int32_t h,
                                   int32_t d, int32_t n) {
  vx_dxa_store_5d(map, src, c, w, h, d, n);
}

// The element type comes from the map (u32, s32, u64, s64, f16, bf16, f32,
// f32.ftz, f64), with the op x type pairs of vx_dxa_reduce_global.
inline void vx_dxa_reduce_1d(const vx_tensor_map_t* map, const void* src, uint32_t op, int32_t c0) {
  __vx_dxa_store_tensor_map(vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, DXA_RT_FROM_MAP, 0), map, src, c0, 0, 0, 0, 0);
}
inline void vx_dxa_reduce_2d(const vx_tensor_map_t* map, const void* src, uint32_t op, int32_t c0, int32_t c1) {
  __vx_dxa_store_tensor_map(vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, DXA_RT_FROM_MAP, 0), map, src, c0, c1, 0, 0, 0);
}
inline void vx_dxa_reduce_3d(const vx_tensor_map_t* map, const void* src, uint32_t op, int32_t c0, int32_t c1,
                             int32_t c2) {
  __vx_dxa_store_tensor_map(vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, DXA_RT_FROM_MAP, 0), map, src, c0, c1, c2, 0, 0);
}
inline void vx_dxa_reduce_4d(const vx_tensor_map_t* map, const void* src, uint32_t op, int32_t c0, int32_t c1,
                             int32_t c2, int32_t c3) {
  __vx_dxa_store_tensor_map(vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, DXA_RT_FROM_MAP, 0), map, src, c0, c1, c2, c3, 0);
}
inline void vx_dxa_reduce_5d(const vx_tensor_map_t* map, const void* src, uint32_t op, int32_t c0, int32_t c1,
                             int32_t c2, int32_t c3, int32_t c4) {
  __vx_dxa_store_tensor_map(vx_dxa_st_ctl(DXA_ST_RED_TENSOR, op, DXA_RT_FROM_MAP, 0), map, src, c0, c1, c2, c3,
                            c4);
}
inline void vx_dxa_reduce_im2col_3d(const vx_tensor_map_t* map, const void* src, uint32_t op, int32_t c,
                                    int32_t w, int32_t n) {
  vx_dxa_reduce_3d(map, src, op, c, w, n);
}
inline void vx_dxa_reduce_im2col_4d(const vx_tensor_map_t* map, const void* src, uint32_t op, int32_t c,
                                    int32_t w, int32_t h, int32_t n) {
  vx_dxa_reduce_4d(map, src, op, c, w, h, n);
}
inline void vx_dxa_reduce_im2col_5d(const vx_tensor_map_t* map, const void* src, uint32_t op, int32_t c,
                                    int32_t w, int32_t h, int32_t d, int32_t n) {
  vx_dxa_reduce_5d(map, src, op, c, w, h, d, n);
}

// prefetch{.param,.const,.global}.tensormap [map]: brings the map into the
// descriptor cache; the warp does not wait. funct3 7, rs1 = map, rs2 field 0.
inline void vx_prefetch_tensormap(const vx_tensor_map_t* map) {
  __asm__ volatile (".insn r %0, 7, %1, x0, %2, x0"
                    :: "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(map) : "memory");
}

// fence.proxy.tensormap::generic.acquire.{cta,cluster,gpu,sys} [map], 128:
// drops the cached copy of `map`, so the issuing core's later tensor
// operations (and its block's, after a block barrier) read it again. Each
// block that uses a rewritten map acquires it itself. rs2 field 1.
inline void vx_fence_proxy_tensormap_acquire(const vx_tensor_map_t* map) {
  __asm__ volatile (".insn r %0, 7, %1, x0, %2, x1"
                    :: "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(map) : "memory");
}

// fence.proxy.tensormap::generic.release.{cta,cluster,gpu,sys}: the warp's
// earlier generic writes of a map reach the level the DXA reads maps from.
inline void vx_fence_proxy_tensormap_release(void) {
  vx_fence_proxy_async_global();
}

// tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.*.sync.aligned
// [gdst], [ssrc], 128: copies the map from shared to global memory with
// ordinary loads and stores, then releases it. Every active thread copies all
// words (the same values), so a partial warp still copies the whole map.
inline void vx_tensormap_cp_fenceproxy(vx_tensor_map_t* gdst, const vx_tensor_map_t* ssrc) {
  auto d = reinterpret_cast<volatile uint32_t*>(gdst->w);
  auto s = reinterpret_cast<const volatile uint32_t*>(ssrc->w);
  for (uint32_t k = 0; k < TMAP_WORDS; ++k)
    d[k] = s[k];
  vx_fence_proxy_tensormap_release();
}

// tensormap.replace.tile.<field>{.global,.shared::cta}.b1024 on a map in
// global or shared memory (a generic address), with PTX's operand meanings
// (rank is rank - 1; elemtype, interleave, swizzle and fill take the Table 33
// codes). A weak generic write: publish it with a release.
inline void vx_tensormap_replace_global_address(vx_tensor_map_t* map, uint64_t addr) {
  vortex::dxa_tmap::replace_global_address(map->w, addr);
}
inline void vx_tensormap_replace_rank(vx_tensor_map_t* map, uint32_t rank_minus_1) {
  vortex::dxa_tmap::replace_rank(map->w, rank_minus_1);
}
inline void vx_tensormap_replace_box_dim(vx_tensor_map_t* map, uint32_t ord, uint32_t v) {
  vortex::dxa_tmap::replace_box_dim(map->w, ord, v);
}
inline void vx_tensormap_replace_global_dim(vx_tensor_map_t* map, uint32_t ord, uint32_t v) {
  vortex::dxa_tmap::replace_global_dim(map->w, ord, v);
}
inline void vx_tensormap_replace_global_stride(vx_tensor_map_t* map, uint32_t ord, uint64_t v) {
  vortex::dxa_tmap::replace_global_stride(map->w, ord, v);
}
inline void vx_tensormap_replace_element_stride(vx_tensor_map_t* map, uint32_t ord, uint32_t v) {
  vortex::dxa_tmap::replace_element_stride(map->w, ord, v);
}
inline void vx_tensormap_replace_elemtype(vx_tensor_map_t* map, uint32_t code) {
  vortex::dxa_tmap::replace_elemtype(map->w, code);
}
inline void vx_tensormap_replace_interleave_layout(vx_tensor_map_t* map, uint32_t v) {
  vortex::dxa_tmap::replace_interleave_layout(map->w, v);
}
inline void vx_tensormap_replace_swizzle_mode(vx_tensor_map_t* map, uint32_t v) {
  vortex::dxa_tmap::replace_swizzle_mode(map->w, v);
}
inline void vx_tensormap_replace_fill_mode(vx_tensor_map_t* map, uint32_t v) {
  vortex::dxa_tmap::replace_fill_mode(map->w, v);
}
#endif // __cplusplus && VX_CFG_EXT_MBAR_ENABLE

#endif // __VX_DXA_H__
