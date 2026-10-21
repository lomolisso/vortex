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

#ifndef __VX_DLMEM_H__
#define __VX_DLMEM_H__

// Distributed local memory: the blocks of a thread-block cluster address each
// other's local memory through one window, rank r of the caller's cluster at
// VX_MEM_DLMEM_BASE_ADDR + r * (1 << VX_CFG_LMEM_LOG_SIZE).

#if !defined(__riscv_xlen) || __riscv_xlen != 64
#error "DLMEM requires XLEN 64"
#endif

// The rank CSR is illegal on a device built without DLMEM.
#if !VX_CFG_EXT_DLMEM_ENABLED
#error "vx_dlmem.h needs a device built with VX_CFG_EXT_DLMEM_ENABLE"
#endif

#include <vx_intrinsics.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Linear index of the calling block in its cluster: x fastest, then y, then z.
static inline __attribute__((always_inline)) uint32_t vx_cluster_block_rank(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_CLUSTER_RANK);
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_cluster_num_blocks(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_CLUSTER_SIZE);
  return v;
}

// Address of the object `p` (in the caller's own local memory) inside the
// local memory of block `rank` of the caller's cluster.
static inline __attribute__((always_inline)) void* vx_cluster_map_shared_rank(const void* p, uint32_t rank) {
  uintptr_t lmem_base = (uintptr_t)csr_read_nv(VX_CSR_CTA_LMEM_ADDR);
  uintptr_t offset    = (uintptr_t)p - lmem_base;
  return (void*)((uintptr_t)VX_MEM_DLMEM_BASE_ADDR
               + ((uintptr_t)rank << VX_CFG_LMEM_LOG_SIZE)
               + offset);
}

// CUDA's __threadfence_block(): the warp's earlier global accesses are
// performed for its block before its later ones. Waits for the warp's
// outstanding global loads and tensor-core metadata loads; drains the L1 only
// when the warp has written global memory since its last fence. Every
// release (an mbarrier arrive, an .rl atomic, a block barrier) does the same
// first. Stack accesses are thread-private local memory, as in CUDA, and are
// not ordered: share stack data through a pointer only after vx_fence().
static inline __attribute__((always_inline)) void vx_fence_cta(void) {
  __asm__ volatile (".insn r %0, 7, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM2) : "memory");
}

// The cluster barrier: one per block, addressed by the block's slot with bit 30
// set and no barrier id. Every warp of every block of the cluster must make
// each call exactly once per phase, as with __syncthreads().
// The barrier itself orders the cluster's local and window memory; the
// default forms add the fences that extend it to global memory, whose L1s are
// not coherent across cores (docs/proposals/dlmem_proposal.md).
static inline __attribute__((always_inline)) uint32_t vx_cluster_bar_id(void) {
  uint32_t __UNIFORM__ v = (1u << 30) | (uint32_t)csr_read_nv(VX_CSR_CTA_ID);
  return v;
}

// PTX barrier.cluster.arrive.relaxed: arrive without waiting and without a
// fence; returns the phase token for a wait.
static inline __attribute__((always_inline)) uint32_t vx_cluster_arrive_relaxed(void) {
  return (uint32_t)vx_barrier_arrive((int)vx_cluster_bar_id(), (int)csr_read_nv(VX_CSR_CTA_SIZE));
}

// Wait until the phase of `token` has completed, without a fence.
static inline __attribute__((always_inline)) void vx_cluster_wait_relaxed(uint32_t token) {
  vx_barrier_wait((int)vx_cluster_bar_id(), (int)token);
}

// PTX barrier.cluster.arrive (.release): the warp's earlier global accesses
// are visible to the cluster once the phase completes.
static inline __attribute__((always_inline)) uint32_t vx_cluster_arrive(void) {
  vx_fence();
  return vx_cluster_arrive_relaxed();
}

// PTX barrier.cluster.wait (.acquire): later global accesses see what the
// cluster released before arriving.
static inline __attribute__((always_inline)) void vx_cluster_wait(uint32_t token) {
  vx_cluster_wait_relaxed(token);
  vx_fence();
}

// CUDA's cluster.sync().
static inline __attribute__((always_inline)) void vx_cluster_sync(void) {
  vx_fence();
  vx_barrier((int)vx_cluster_bar_id(), (int)csr_read_nv(VX_CSR_CTA_SIZE));
  vx_fence();
}

#ifdef __cplusplus
}

namespace vortex {

// The calling block's thread-block cluster, after CUDA's cooperative_groups::cluster_group.
class cluster_group {
public:
  uint32_t block_rank() const { return vx_cluster_block_rank(); }

  uint32_t num_blocks() const { return vx_cluster_num_blocks(); }

  template <typename T>
  T* map_shared_rank(T* p, uint32_t rank) const {
    return static_cast<T*>(vx_cluster_map_shared_rank(p, rank));
  }

  // Arrive and wait for the whole cluster, ordering global memory too.
  void sync() const { vx_cluster_sync(); }

  // Split arrive and wait, with sync()'s ordering: arrive releases and wait
  // acquires global memory too.
  uint32_t barrier_arrive() const { return vx_cluster_arrive(); }

  void barrier_wait(uint32_t token) const { vx_cluster_wait(token); }

  // Split arrive and wait without fences: they order only local and window memory.
  uint32_t barrier_arrive_relaxed() const { return vx_cluster_arrive_relaxed(); }

  void barrier_wait_relaxed(uint32_t token) const { vx_cluster_wait_relaxed(token); }
};

inline cluster_group this_cluster() { return cluster_group(); }

} // namespace vortex

#endif // __cplusplus

#endif // __VX_DLMEM_H__
