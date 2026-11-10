// The thread-block-cluster primitives the DSMEM collectives are written in,
// with the names and meanings of h100-dsmem-collectives/collectives/
// dsmem_primitives.cuh (commit 2b9a97e), over the CUDA porting layer.
//
// A 32-bit address is a .shared::cluster address, (rank << VX_CFG_LMEM_LOG_SIZE)
// | offset. A block's own variables have their own rank's address, as PTX's
// .shared::cta window lies inside .shared::cluster.
//
// Departures from the H100 sequences:
// - ld/st_cluster_v4 are two 64-bit accesses (no 128-bit load or store);
// - mbar_arrive_remote and mbar_arrive_expect_tx_remote are the
//   .release.cluster forms, which drain a cluster fence before the arrive
//   (libcu++ names only the cluster scope for a remote arrive; the H100
//   default is .release.cta): stronger ordering, a later arrive;
// - globaltimer() counts cycles at a nominal 1 ns each.
//
// The host includes this file for CollectiveSmem and illegal() only; device
// code of a collective sits under #ifdef __riscv.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef CLUSTER_N
#define CLUSTER_N 4
#endif
#ifndef MSG_ELEMS
#define MSG_ELEMS 1024
#endif
#ifndef ELEM_T
#define ELEM_T float
#endif

#ifdef __riscv
#include <vx_cuda.h>
#else
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif
#endif

namespace dsmem {

// ts[rank * kStamps + i]: 0 entry, 1 exit, 2 + r end of round r.
constexpr int kStamps = 32;

#ifdef __riscv

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

// ---------------------------------------------------------------- identity

__device__ __forceinline__ uint32_t cluster_rank() { return vx_cluster_block_rank(); }
__device__ __forceinline__ uint32_t cluster_size() { return vx_cluster_num_blocks(); }
__device__ __forceinline__ uint32_t smid() { return __smid(); }
__device__ __forceinline__ unsigned long long globaltimer() { return vx_globaltimer(); }

// -------------------------------------------------------------- addressing

template <typename T>
__device__ __forceinline__ uint32_t smem_u32(const T* p) {
  return vx_cvta_shared(p);
}

__device__ __forceinline__ uint32_t mapa(uint32_t smem_addr, uint32_t rank) { return vx_mapa_u32(smem_addr, rank); }

template <typename T>
__device__ __forceinline__ T* map_ptr(T* local, uint32_t rank) {
  return cg::this_cluster().map_shared_rank(local, rank);
}

namespace detail {

// The block's own variable at the offset of `a` (operations on the own block).
__device__ __forceinline__ void* cta_ptr(uint32_t a) {
  return (char*)__local_mem() + (a & VX_DLMEM_OFFSET_MASK);
}

// The window pointer of `a` (operations on any rank of the cluster).
__device__ __forceinline__ void* cluster_ptr(uint32_t a) { return vx_cvta_cluster_to_generic(a); }

__device__ __forceinline__ uint64_t* cta_bar(uint32_t a) { return (uint64_t*)cta_ptr(a); }
__device__ __forceinline__ uint64_t* cluster_bar(uint32_t a) { return (uint64_t*)cluster_ptr(a); }

} // namespace detail

// ------------------------------------------------- remote loads and stores

__device__ __forceinline__ uint32_t ld_cluster_u32(uint32_t caddr) {
  return *(volatile uint32_t*)detail::cluster_ptr(caddr);
}
__device__ __forceinline__ void st_cluster_u32(uint32_t caddr, uint32_t v) {
  *(volatile uint32_t*)detail::cluster_ptr(caddr) = v;
}
__device__ __forceinline__ float ld_cluster_f32(uint32_t caddr) {
  return *(volatile float*)detail::cluster_ptr(caddr);
}
__device__ __forceinline__ void st_cluster_f32(uint32_t caddr, float v) {
  *(volatile float*)detail::cluster_ptr(caddr) = v;
}

__device__ __forceinline__ uint4 ld_cluster_v4(uint32_t caddr) {
  volatile uint64_t* p = (volatile uint64_t*)detail::cluster_ptr(caddr);
  const uint64_t lo = p[0], hi = p[1];
  uint4 v;
  v.x = (uint32_t)lo;
  v.y = (uint32_t)(lo >> 32);
  v.z = (uint32_t)hi;
  v.w = (uint32_t)(hi >> 32);
  return v;
}
__device__ __forceinline__ void st_cluster_v4(uint32_t caddr, uint4 v) {
  volatile uint64_t* p = (volatile uint64_t*)detail::cluster_ptr(caddr);
  p[0] = (uint64_t)v.x | ((uint64_t)v.y << 32);
  p[1] = (uint64_t)v.z | ((uint64_t)v.w << 32);
}

__device__ __forceinline__ void red_cluster_add_f32(uint32_t caddr, float v) {
  (void)atomicAdd((float*)detail::cluster_ptr(caddr), v);
}
__device__ __forceinline__ void red_cluster_add_u32(uint32_t caddr, uint32_t v) {
  (void)atomicAdd((unsigned*)detail::cluster_ptr(caddr), (unsigned)v);
}

// Orders the block's ordinary shared-memory accesses against its bulk copies,
// in both directions.
__device__ __forceinline__ void fence_proxy_async_smem() { ptx::fence_proxy_async(ptx::space_shared); }

// ---------------------------------------------------------------- mbarrier
// A barrier operand names an 8-byte aligned uint64_t of the own block by its
// 32-bit address.

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t arrival_count) {
  ptx::mbarrier_init(detail::cta_bar(bar), arrival_count);
}

__device__ __forceinline__ void fence_mbar_init() { ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster); }

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, detail::cta_bar(bar),
                                       bytes);
}

// On block `rank`'s barrier at the offset of `bar`.
__device__ __forceinline__ void mbar_arrive_expect_tx_remote(uint32_t bar, uint32_t rank, uint32_t bytes) {
  ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cluster, ptx::space_cluster,
                                 detail::cluster_bar(mapa(bar, rank)), bytes);
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) { (void)ptx::mbarrier_arrive(detail::cta_bar(bar)); }

__device__ __forceinline__ void mbar_arrive_remote(uint32_t bar, uint32_t rank) {
  ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cluster, ptx::space_cluster,
                       detail::cluster_bar(mapa(bar, rank)));
}

// Returns once the barrier has left the phase whose parity is `parity`.
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
  while (!ptx::mbarrier_try_wait_parity(detail::cta_bar(bar), parity)) {
  }
}

// ------------------------------------------------------------- bulk copies

// Issued by a single thread. The copy lands in block dst_rank at the offset of
// `dst` and completes its bytes on that block's barrier at the offset of
// `dst_bar`.
__device__ __forceinline__ void bulk_push(uint32_t dst_rank, uint32_t dst, uint32_t src, uint32_t bytes,
                                          uint32_t dst_bar) {
  ptx::cp_async_bulk(ptx::space_cluster, ptx::space_shared, detail::cluster_ptr(mapa(dst, dst_rank)),
                     detail::cta_ptr(src), bytes, detail::cluster_bar(mapa(dst_bar, dst_rank)));
}

__device__ __forceinline__ void st_async_v4(uint32_t dst_rank, uint32_t dst, uint4 v, uint32_t dst_bar) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  ptx::st_async((uint32_t*)detail::cluster_ptr(mapa(dst, dst_rank)), w,
                detail::cluster_bar(mapa(dst_bar, dst_rank)));
}

// --------------------------------------------------------- cluster barrier

__device__ __forceinline__ void cluster_arrive() { ptx::barrier_cluster_arrive(ptx::sem_release); }
__device__ __forceinline__ void cluster_arrive_relaxed() { ptx::barrier_cluster_arrive(ptx::sem_relaxed); }
__device__ __forceinline__ void cluster_wait() { ptx::barrier_cluster_wait(ptx::sem_acquire); }
__device__ __forceinline__ void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

// -------------------------------------------------------------- reductions

template <typename T>
__device__ __forceinline__ T lowest_value();
template <>
__device__ __forceinline__ float lowest_value<float>() { return -__int_as_float(0x7f800000); }
template <>
__device__ __forceinline__ double lowest_value<double>() { return -__longlong_as_double(0x7ff0000000000000ll); }
template <>
__device__ __forceinline__ int lowest_value<int>() { return (int)0x80000000u; }
template <>
__device__ __forceinline__ unsigned lowest_value<unsigned>() { return 0u; }
template <>
__device__ __forceinline__ __half lowest_value<__half>() { return __float2half(-__int_as_float(0x7f800000)); }

struct Sum {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
  template <typename T>
  static __device__ __forceinline__ T identity() { return T(0); }
};

struct Max {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const { return a > b ? a : b; }
  template <typename T>
  static __device__ __forceinline__ T identity() { return lowest_value<T>(); }
};

// ---------------------------------------------------------------- harness

// Every block's dynamic shared memory begins at one common offset, so mapa of
// a member's address names that member in the other block.
template <class S>
__device__ __forceinline__ S& dyn_smem() {
  return *reinterpret_cast<S*>(__local_mem());
}

struct Stamps {
  unsigned long long* base;
  __device__ __forceinline__ void mark(int i) const {
    if (threadIdx.x == 0)
      base[i] = globaltimer();
  }
  __device__ __forceinline__ void round(int r) const { mark(2 + r); }
};

template <typename T, int M>
__device__ __forceinline__ void load_block(T* smem, const T* g) {
  for (int i = threadIdx.x; i < M; i += blockDim.x)
    smem[i] = g[i];
  __syncthreads();
}

template <typename T, int M>
__device__ __forceinline__ void store_block(T* g, const T* smem) {
  __syncthreads();
  for (int i = threadIdx.x; i < M; i += blockDim.x)
    g[i] = smem[i];
}

#endif // __riscv

} // namespace dsmem
