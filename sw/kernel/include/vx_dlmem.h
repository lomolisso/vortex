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
#include <stdbool.h>
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

// PTX special registers of the cluster. Clusters tile the grid, so the ids
// derive exactly from the block id and the cluster shape. An axis without a
// cluster shape has dimension 1, also in a launch that named none
// (vx_is_explicit_cluster() is then 0).
static inline __attribute__((always_inline)) uint32_t vx_cluster_nctaid_x(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_CLUSTER_DIM_X);
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_cluster_nctaid_y(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_CLUSTER_DIM_Y);
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_cluster_nctaid_z(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_CLUSTER_DIM_Z);
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_cluster_ctaid_x(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_BLOCK_ID_X) % vx_cluster_nctaid_x();
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_cluster_ctaid_y(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_BLOCK_ID_Y) % vx_cluster_nctaid_y();
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_cluster_ctaid_z(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_BLOCK_ID_Z) % vx_cluster_nctaid_z();
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_clusterid_x(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_BLOCK_ID_X) / vx_cluster_nctaid_x();
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_clusterid_y(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_BLOCK_ID_Y) / vx_cluster_nctaid_y();
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_clusterid_z(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_BLOCK_ID_Z) / vx_cluster_nctaid_z();
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_nclusterid_x(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_GRID_DIM_X) / vx_cluster_nctaid_x();
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_nclusterid_y(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_GRID_DIM_Y) / vx_cluster_nctaid_y();
  return v;
}

static inline __attribute__((always_inline)) uint32_t vx_nclusterid_z(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_GRID_DIM_Z) / vx_cluster_nctaid_z();
  return v;
}

// PTX %is_explicit_cluster: the launch named a cluster shape.
static inline __attribute__((always_inline)) uint32_t vx_is_explicit_cluster(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_CLUSTER_FLAGS) & 1u;
  return v;
}

// PTX %dynamic_smem_size: the launch's local-memory request in bytes.
static inline __attribute__((always_inline)) uint32_t vx_dynamic_smem_size(void) {
  uint32_t __UNIFORM__ v = (uint32_t)csr_read_nv(VX_CSR_CTA_LMEM_SIZE);
  return v;
}

// PTX %total_smem_size and %aggr_smem_size: the block's slot, the request
// rounded up as the dispatcher rounds it. No local memory is reserved for the
// system, so PTX %reserved_smem_offset_* are 0.
static inline __attribute__((always_inline)) uint32_t vx_total_smem_size(void) {
#if VX_CFG_EXT_DXA_ENABLED && (VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8) > VX_CFG_MEM_BLOCK_SIZE)
  const uint32_t align = VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);
#else
  const uint32_t align = VX_CFG_MEM_BLOCK_SIZE;
#endif
  return (vx_dynamic_smem_size() + align - 1) & ~(align - 1);
}

static inline __attribute__((always_inline)) uint32_t vx_aggr_smem_size(void) {
  return vx_total_smem_size();
}

static inline __attribute__((always_inline)) uint32_t vx_reserved_smem_offset(void) {
  return 0;
}

// PTX %globaltimer in nanoseconds: the cycle counter at a nominal 1 GHz clock.
#define VX_GLOBALTIMER_NS_PER_CYCLE 1

static inline __attribute__((always_inline)) uint64_t vx_globaltimer(void) {
  return vx_rdcycle() * VX_GLOBALTIMER_NS_PER_CYCLE;
}

// The 32-bit .shared::cluster address space: rank r's local memory at
// r << VX_CFG_LMEM_LOG_SIZE. A block's own variables have their own rank's
// address, which converts back to a window pointer that takes the local path.
#define VX_DLMEM_OFFSET_MASK ((1u << VX_CFG_LMEM_LOG_SIZE) - 1u)

// The cluster window: a slice per rank of the largest cluster, rounded up to a power of two.
static inline __attribute__((always_inline)) uint64_t vx_dlmem_window_size(void) {
  uint64_t n = 1;
  while (n < VX_CFG_DLMEM_MAX_CLUSTER)
    n <<= 1;
  return n << VX_CFG_LMEM_LOG_SIZE;
}

// PTX isspacep.shared::cta: `p` is in the core's local memory.
static inline __attribute__((always_inline)) bool vx_isspacep_shared(const void* p) {
  uintptr_t a = (uintptr_t)p;
  return a >= (uintptr_t)VX_MEM_LMEM_BASE_ADDR
      && a - (uintptr_t)VX_MEM_LMEM_BASE_ADDR < ((uintptr_t)1 << VX_CFG_LMEM_LOG_SIZE);
}

// PTX isspacep.shared::cluster: `p` is in the cluster window, or is a local
// variable of the block (shared::cta lies inside shared::cluster).
static inline __attribute__((always_inline)) bool vx_isspacep_shared_cluster(const void* p) {
  uintptr_t a = (uintptr_t)p;
  return (a >= (uintptr_t)VX_MEM_DLMEM_BASE_ADDR
       && a - (uintptr_t)VX_MEM_DLMEM_BASE_ADDR < (uintptr_t)vx_dlmem_window_size())
      || vx_isspacep_shared(p);
}

// PTX cvta.to.shared::cluster on a generic pointer to a local variable of the block.
static inline __attribute__((always_inline)) uint32_t vx_cvta_shared(const void* p) {
  uintptr_t lmem_base = (uintptr_t)csr_read_nv(VX_CSR_CTA_LMEM_ADDR);
  return (vx_cluster_block_rank() << VX_CFG_LMEM_LOG_SIZE) + (uint32_t)((uintptr_t)p - lmem_base);
}

// PTX cvta.shared::cluster: the generic (window) pointer of a 32-bit cluster address.
static inline __attribute__((always_inline)) void* vx_cvta_cluster_to_generic(uint32_t a) {
  return (void*)((uintptr_t)VX_MEM_DLMEM_BASE_ADDR + a);
}

// PTX mapa.shared::cluster.u32: the same offset in rank `rank`.
static inline __attribute__((always_inline)) uint32_t vx_mapa_u32(uint32_t a, uint32_t rank) {
  return (rank << VX_CFG_LMEM_LOG_SIZE) | (a & VX_DLMEM_OFFSET_MASK);
}

// PTX mapa (generic): `p` is a local variable of the block or a window pointer.
static inline __attribute__((always_inline)) void* vx_mapa(const void* p, uint32_t rank) {
  uintptr_t a = (uintptr_t)p;
  if (vx_isspacep_shared(p))
    return vx_cluster_map_shared_rank(p, rank);
  return vx_cvta_cluster_to_generic(vx_mapa_u32((uint32_t)(a - (uintptr_t)VX_MEM_DLMEM_BASE_ADDR), rank));
}

// PTX getctarank.shared::cluster.u32.
static inline __attribute__((always_inline)) uint32_t vx_getctarank_u32(uint32_t a) {
  return a >> VX_CFG_LMEM_LOG_SIZE;
}

// PTX getctarank (generic): the rank owning a window pointer; the caller's own
// rank for a local variable of the block.
static inline __attribute__((always_inline)) uint32_t vx_getctarank(const void* p) {
  if (vx_isspacep_shared(p))
    return vx_cluster_block_rank();
  return vx_getctarank_u32((uint32_t)((uintptr_t)p - (uintptr_t)VX_MEM_DLMEM_BASE_ADDR));
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
// set and no barrier id. Every warp of the cluster that has not exited must
// make each call exactly once per phase, as with __syncthreads().
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
  vx_fence_cluster();
  return vx_cluster_arrive_relaxed();
}

// PTX barrier.cluster.wait (.acquire): later global accesses see what the
// cluster released before arriving.
static inline __attribute__((always_inline)) void vx_cluster_wait(uint32_t token) {
  vx_cluster_wait_relaxed(token);
  vx_fence_cluster();
}

// CUDA's cluster.sync().
static inline __attribute__((always_inline)) void vx_cluster_sync(void) {
  vx_fence_cluster();
  vx_barrier((int)vx_cluster_bar_id(), (int)csr_read_nv(VX_CSR_CTA_SIZE));
  vx_fence_cluster();
}

// PTX scoped memory operations at .cluster scope (sm_90). An acquire is the
// access, then vx_fence_cluster(); a release is vx_fence_cluster(), then the
// access; a relaxed access is a plain (atomic) one.
static inline __attribute__((always_inline)) void vx_fence_acquire_cluster(void) { vx_fence_cluster(); }
static inline __attribute__((always_inline)) void vx_fence_release_cluster(void) { vx_fence_cluster(); }
static inline __attribute__((always_inline)) void vx_fence_acq_rel_cluster(void) { vx_fence_cluster(); }
static inline __attribute__((always_inline)) void vx_fence_sc_cluster(void)      { vx_fence_cluster(); }

// PTX fence.acquire.sync_restrict::shared::cluster.cluster.
static inline __attribute__((always_inline)) void vx_fence_acquire_sync_restrict_shared_cluster(void) {
  vx_fence_cluster();
}

#define __VX_LDST_CLUSTER(sfx, T)                                                              \
  static inline __attribute__((always_inline)) T vx_ld_relaxed_cluster_##sfx(const T* p) {     \
    return __atomic_load_n(p, __ATOMIC_RELAXED);                                               \
  }                                                                                            \
  static inline __attribute__((always_inline)) T vx_ld_acquire_cluster_##sfx(const T* p) {     \
    T v = __atomic_load_n(p, __ATOMIC_RELAXED);                                                \
    vx_fence_cluster();                                                                        \
    return v;                                                                                  \
  }                                                                                            \
  static inline __attribute__((always_inline)) void vx_st_relaxed_cluster_##sfx(T* p, T v) {   \
    __atomic_store_n(p, v, __ATOMIC_RELAXED);                                                  \
  }                                                                                            \
  static inline __attribute__((always_inline)) void vx_st_release_cluster_##sfx(T* p, T v) {   \
    vx_fence_cluster();                                                                        \
    __atomic_store_n(p, v, __ATOMIC_RELAXED);                                                  \
  }

__VX_LDST_CLUSTER(u32, uint32_t)
__VX_LDST_CLUSTER(u64, uint64_t)

// atom.{relaxed,acquire,release,acq_rel}.cluster.<op> returns the old value;
// red.{relaxed,release}.cluster.<op> discards it.
#define __VX_ATOM_CLUSTER(op, sfx, T, expr)                                                        \
  static inline __attribute__((always_inline)) T vx_atom_##op##_relaxed_cluster_##sfx(T* p, T v) { \
    return expr;                                                                                   \
  }                                                                                                \
  static inline __attribute__((always_inline)) T vx_atom_##op##_acquire_cluster_##sfx(T* p, T v) { \
    T old = expr;                                                                                  \
    vx_fence_cluster();                                                                            \
    return old;                                                                                    \
  }                                                                                                \
  static inline __attribute__((always_inline)) T vx_atom_##op##_release_cluster_##sfx(T* p, T v) { \
    vx_fence_cluster();                                                                            \
    return expr;                                                                                   \
  }                                                                                                \
  static inline __attribute__((always_inline)) T vx_atom_##op##_acq_rel_cluster_##sfx(T* p, T v) { \
    vx_fence_cluster();                                                                            \
    T old = expr;                                                                                  \
    vx_fence_cluster();                                                                            \
    return old;                                                                                    \
  }                                                                                                \
  static inline __attribute__((always_inline)) void vx_red_##op##_relaxed_cluster_##sfx(T* p, T v) { \
    (void)(expr);                                                                                  \
  }                                                                                                \
  static inline __attribute__((always_inline)) void vx_red_##op##_release_cluster_##sfx(T* p, T v) { \
    vx_fence_cluster();                                                                            \
    (void)(expr);                                                                                  \
  }

#define __VX_ATOM_CLUSTER_INT(sfx, T)                                                \
  __VX_ATOM_CLUSTER(add,  sfx, T, __atomic_fetch_add(p, v, __ATOMIC_RELAXED))        \
  __VX_ATOM_CLUSTER(and,  sfx, T, __atomic_fetch_and(p, v, __ATOMIC_RELAXED))        \
  __VX_ATOM_CLUSTER(or,   sfx, T, __atomic_fetch_or(p, v, __ATOMIC_RELAXED))         \
  __VX_ATOM_CLUSTER(xor,  sfx, T, __atomic_fetch_xor(p, v, __ATOMIC_RELAXED))        \
  __VX_ATOM_CLUSTER(min,  sfx, T, __atomic_fetch_min(p, v, __ATOMIC_RELAXED))        \
  __VX_ATOM_CLUSTER(max,  sfx, T, __atomic_fetch_max(p, v, __ATOMIC_RELAXED))        \
  __VX_ATOM_CLUSTER(exch, sfx, T, __atomic_exchange_n(p, v, __ATOMIC_RELAXED))

__VX_ATOM_CLUSTER_INT(u32, uint32_t)
__VX_ATOM_CLUSTER_INT(s32, int32_t)
__VX_ATOM_CLUSTER_INT(u64, uint64_t)
__VX_ATOM_CLUSTER_INT(s64, int64_t)

#define __VX_CAS_CLUSTER(sfx, T)                                                                      \
  static inline __attribute__((always_inline)) T vx_atom_cas_relaxed_cluster_##sfx(T* p, T cmp, T v) { \
    __atomic_compare_exchange_n(p, &cmp, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);                \
    return cmp;                                                                                       \
  }                                                                                                   \
  static inline __attribute__((always_inline)) T vx_atom_cas_acquire_cluster_##sfx(T* p, T cmp, T v) { \
    T old = vx_atom_cas_relaxed_cluster_##sfx(p, cmp, v);                                             \
    vx_fence_cluster();                                                                               \
    return old;                                                                                       \
  }                                                                                                   \
  static inline __attribute__((always_inline)) T vx_atom_cas_release_cluster_##sfx(T* p, T cmp, T v) { \
    vx_fence_cluster();                                                                               \
    return vx_atom_cas_relaxed_cluster_##sfx(p, cmp, v);                                              \
  }                                                                                                   \
  static inline __attribute__((always_inline)) T vx_atom_cas_acq_rel_cluster_##sfx(T* p, T cmp, T v) { \
    vx_fence_cluster();                                                                               \
    T old = vx_atom_cas_relaxed_cluster_##sfx(p, cmp, v);                                             \
    vx_fence_cluster();                                                                               \
    return old;                                                                                       \
  }

__VX_CAS_CLUSTER(u32, uint32_t)
__VX_CAS_CLUSTER(u64, uint64_t)

#ifdef __cplusplus
}

namespace vortex {

struct dim3 {
  uint32_t x, y, z;
};

// The calling block's thread-block cluster, after CUDA's cooperative_groups::cluster_group.
class cluster_group {
public:
  uint32_t block_rank() const { return vx_cluster_block_rank(); }

  uint32_t num_blocks() const { return vx_cluster_num_blocks(); }

  dim3 dim_blocks() const { return {vx_cluster_nctaid_x(), vx_cluster_nctaid_y(), vx_cluster_nctaid_z()}; }

  dim3 block_index() const { return {vx_cluster_ctaid_x(), vx_cluster_ctaid_y(), vx_cluster_ctaid_z()}; }

  dim3 dim_threads() const {
    return {block_dim(0) * vx_cluster_nctaid_x(), block_dim(1) * vx_cluster_nctaid_y(),
            block_dim(2) * vx_cluster_nctaid_z()};
  }

  uint32_t num_threads() const { return block_threads() * num_blocks(); }

  // The calling thread's index in the cluster, block by block in rank order.
  uint32_t thread_rank() const {
    const uint32_t t = (uint32_t)csr_read_nv(VX_CSR_CTA_THREAD_ID_X)
        + block_dim(0) * ((uint32_t)csr_read_nv(VX_CSR_CTA_THREAD_ID_Y)
        + block_dim(1) * (uint32_t)csr_read_nv(VX_CSR_CTA_THREAD_ID_Z));
    return block_rank() * block_threads() + t;
  }

  // Every launch is a grid of clusters, 1x1x1 when it named no shape.
  bool is_valid() const { return true; }

  // The rank of the block whose local memory holds `p` (a window pointer, or the caller's own).
  uint32_t query_shared_rank(const void* p) const { return vx_getctarank(p); }

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

private:
  static uint32_t block_dim(uint32_t i) {
    return (uint32_t)((i == 0) ? csr_read_nv(VX_CSR_CTA_BLOCK_DIM_X)
                    : (i == 1) ? csr_read_nv(VX_CSR_CTA_BLOCK_DIM_Y)
                               : csr_read_nv(VX_CSR_CTA_BLOCK_DIM_Z));
  }

  static uint32_t block_threads() { return block_dim(0) * block_dim(1) * block_dim(2); }
};

inline cluster_group this_cluster() { return cluster_group(); }

} // namespace vortex

#endif // __cplusplus

#endif // __VX_DLMEM_H__
