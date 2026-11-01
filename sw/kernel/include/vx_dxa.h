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

#endif // __VX_DXA_H__
