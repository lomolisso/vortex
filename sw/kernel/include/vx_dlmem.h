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

// The cluster barrier: one per block, addressed by the block's slot with bit 30
// set and no barrier id. Every warp of every block of the cluster must make
// each call exactly once per phase, as with __syncthreads().
// Arrive releases and wait acquires the cluster's local and window memory;
// global memory only through sync()'s fences (docs/proposals/dlmem_proposal.md).
static inline __attribute__((always_inline)) uint32_t vx_cluster_bar_id(void) {
  uint32_t __UNIFORM__ v = (1u << 30) | (uint32_t)csr_read_nv(VX_CSR_CTA_ID);
  return v;
}

// Arrive without waiting; returns the phase token for vx_cluster_wait(). No fence.
static inline __attribute__((always_inline)) uint32_t vx_cluster_arrive(void) {
  return (uint32_t)vx_barrier_arrive((int)vx_cluster_bar_id(), (int)csr_read_nv(VX_CSR_CTA_SIZE));
}

// Wait until the phase of `token` has completed. No fence.
static inline __attribute__((always_inline)) void vx_cluster_wait(uint32_t token) {
  vx_barrier_wait((int)vx_cluster_bar_id(), (int)token);
}

// CUDA's cluster.sync(): the fences extend the barrier's ordering to global
// memory, whose L1s are not coherent across cores.
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

  // Split arrive and wait, without the fences of sync(): like PTX
  // barrier.cluster.arrive.relaxed, they order only local and window memory.
  uint32_t barrier_arrive() const { return vx_cluster_arrive(); }

  void barrier_wait(uint32_t token) const { vx_cluster_wait(token); }
};

inline cluster_group this_cluster() { return cluster_group(); }

} // namespace vortex

#endif // __cplusplus

#endif // __VX_DLMEM_H__
