// FlashAttention-4 two-CTA backward, the dS exchange over distributed shared
// memory, with the dQ MMA replaced by a plain shared-memory product.
//
// Re-implementation of h100-dsmem-collectives/kernels/
// flash_bwd_2cta_ds_exchange__fa4.cu (a Hopper port of the exchange in
// flash-attention 98c1cee, flash_attn/cute/flash_bwd_sm100.py and
// copy_utils.py). Same tile shape, shared-memory layout, barriers, rounds and
// peers: each CTA splits its 128 dS rows by Q-column half, arms the peer's
// full barrier with a remote expect_tx and pushes 16 KB into the peer's sdS
// with a bulk copy; a relay warp per CTA waits on its own full barrier and
// arrives on the leader's barrier, which the leader's warp 0 waits on; then
// every CTA computes its 64 x 128 slice of dQ = dS K and adds it to global
// memory with f32 atomics, between cluster barriers.
//
// Departures:
// - Each CTA reads its own KV rows of K from sK and the peer's from K_in; the
//   original read sK for rows 0..127 on both ranks, which is wrong on rank 1
//   (its sK holds rows 128..255). Rank 0's reads are the original's.
// - The unused local exchange_stage is deleted (-Werror). dS_cluster_empty_mbar
//   stays initialised and unused, as in the original.
// - The remote arrive.expect_tx and the relay's remote arrive are the
//   .release.cluster forms (the original's default is .release.cta): Vortex
//   names only the cluster scope for a remote arrive. Stronger ordering, a
//   later arrive.
// - The pointers come in one argument struct, with a buffer for each CTA's
//   core id (placement check); no __shared__: the storage is the block's
//   dynamic local memory at offset 0 in both CTAs.
// - The comments of the original speak of bf16; the code uses __half, and so
//   does this port.
// - Built with -fconvergent-functions, as clang compiles CUDA device code:
//   without it a barrier was duplicated into both arms of a merged branch.

#include <vx_cuda.h>
#include "common.h"

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

namespace {

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return vx_cvta_shared(p); }

// The same offset in block `peer` (mapa.shared::cluster).
__device__ __forceinline__ uint32_t set_block_rank(uint32_t smem_addr, uint32_t peer) {
  return vx_mapa_u32(smem_addr, peer);
}

// A barrier of the own block, named by its 32-bit address.
__device__ __forceinline__ uint64_t* own_bar(uint32_t a) {
  return (uint64_t*)((char*)__local_mem() + (a & VX_DLMEM_OFFSET_MASK));
}

__device__ __forceinline__ void* window(uint32_t a) { return vx_cvta_cluster_to_generic(a); }

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_remote(uint32_t bar, uint32_t bytes, uint32_t peer) {
  ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cluster, ptx::space_cluster,
                                 (uint64_t*)window(set_block_rank(bar, peer)), bytes);
}

__device__ __forceinline__ void mbarrier_arrive_remote(uint32_t bar, uint32_t peer) {
  ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cluster, ptx::space_cluster,
                       (uint64_t*)window(set_block_rank(bar, peer)));
}

// The own block's bytes at `src` land in block `peer` at the offset of `dst`
// and complete on that block's barrier at the offset of `bar`.
__device__ __forceinline__ void cpasync_bulk_s2cluster(uint32_t src, uint32_t dst, uint32_t bar, uint32_t bytes,
                                                       uint32_t peer) {
  ptx::cp_async_bulk(ptx::space_cluster, ptx::space_shared, window(set_block_rank(dst, peer)),
                     (char*)__local_mem() + (src & VX_DLMEM_OFFSET_MASK), bytes,
                     (uint64_t*)window(set_block_rank(bar, peer)));
}

__device__ __forceinline__ void mbarrier_init(uint32_t bar, uint32_t count) {
  ptx::mbarrier_init(own_bar(bar), count);
  ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster);
}

__device__ __forceinline__ void mbarrier_wait(uint32_t bar, uint32_t phase) {
  while (!ptx::mbarrier_try_wait_parity(own_bar(bar), phase)) {
  }
}

__device__ __forceinline__ void fence_view_async_shared() { ptx::fence_proxy_async(ptx::space_shared); }

} // namespace

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256)
flash_bwd_2cta_ds_exchange(kernel_arg_t* __UNIFORM__ arg) {
  VX_CUDA_KERNEL_ENTRY(flash_bwd_2cta_ds_exchange);
  const __half* __restrict__ dS_in = (const __half*)arg->dS_in;
  const __half* __restrict__ K_in = (const __half*)arg->K_in;
  float* __restrict__ dQ_out = (float*)arg->dQ_out;

  SharedStorage& s = *reinterpret_cast<SharedStorage*>(__local_mem());
  cg::cluster_group cluster = cg::this_cluster();
  const uint32_t rank = cluster.block_rank();
  const uint32_t peer = rank ^ 1;
  const uint32_t full_bar = smem_u32(&s.dS_cluster_full_mbar);
  const uint32_t leader_bar = smem_u32(&s.dS_cluster_leader_mbar);
  const int tid = threadIdx.x, warp = tid / 32;

  // One thread sets up the three barriers (warp 4 in FA4).
  if (tid == 0) {
    mbarrier_init(full_bar, 1);
    mbarrier_init(smem_u32(&s.dS_cluster_empty_mbar), 1);
    mbarrier_init(leader_bar, 2);
    ((uint32_t*)arg->smid)[rank] = __smid();
  }
  // K rows of this CTA, then its dS rows: the own Q-column half goes to sdS
  // at this CTA's KV-row offset, the peer's half to the staging buffer.
  for (int i = tid; i < TILE_N * HEAD_D; i += blockDim.x)
    s.sK[i] = K_in[rank * TILE_N * HEAD_D + i];
  for (int i = tid; i < TILE_N * TILE_M; i += blockDim.x) {
    const int kv = i / TILE_M, q = i % TILE_M, stage = q / HALF_M, qq = q % HALF_M;
    const __half v = dS_in[rank * TILE_N * TILE_M + i];
    if (stage == (int)rank)
      s.sdS[(rank * TILE_N + kv) * HALF_M + qq] = v;
    else
      s.sdS_xchg[kv * HALF_M + qq] = v;
  }
  __syncthreads();
  // Both CTAs' barriers exist before either sends a remote arrive.
  cluster.sync();

  // The exchange: make the generic stores to sdS_xchg visible to the bulk
  // copy, then one thread arms the peer's full barrier for 16 KB and pushes
  // the half into the peer's sdS rows of this CTA.
  fence_view_async_shared();
  __syncthreads();
  if (tid == 0) {
    const uint32_t dst = smem_u32(s.sdS) + rank * XCHG_BYTES;
    mbarrier_arrive_and_expect_tx_remote(full_bar, XCHG_BYTES, peer);
    cpasync_bulk_s2cluster(smem_u32(s.sdS_xchg), dst, full_bar, XCHG_BYTES, peer);
  }

  // Relay warp: once the peer's half has landed here, tell the leader (rank 0).
  if (warp == 1) {
    mbarrier_wait(full_bar, 0);
    if (tid % 32 == 0)
      mbarrier_arrive_remote(leader_bar, 0);
  }

  // The leader's MMA warp waits for both relays. FA4 then issues one
  // pair-wide MMA over both CTAs' operands; here each CTA computes its own
  // half after a cluster barrier.
  if (rank == 0 && warp == 0)
    mbarrier_wait(leader_bar, 0);
  cluster.sync();

  for (int i = tid; i < HALF_M * HEAD_D; i += blockDim.x) {
    const int q = i / HEAD_D, d = i % HEAD_D;
    float acc = 0.f;
    for (int kv = 0; kv < 2 * TILE_N; ++kv) {
      const bool own_rows = kv / TILE_N == (int)rank;
      const __half kval = own_rows ? s.sK[(kv % TILE_N) * HEAD_D + d] : K_in[kv * HEAD_D + d];
      acc += __half2float(s.sdS[kv * HALF_M + q]) * __half2float(kval);
    }
    s.dQ_acc[i] = acc;
  }
  __syncthreads();
  // Each CTA adds only its 64 rows of dQ to global memory.
  for (int i = tid; i < HALF_M * HEAD_D; i += blockDim.x)
    atomicAdd(&dQ_out[rank * HALF_M * HEAD_D + i], s.dQ_acc[i]);
  // Neither CTA leaves while the other may still address its shared memory.
  cluster.sync();
}
