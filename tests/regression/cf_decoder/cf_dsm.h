// The cluster collective of ClusterFusion's decoder kernels: a re-implementation
// of cluster_reduce in include/dsm.cuh of ClusterFusion 8755446
// (https://github.com/xinhao-luo/ClusterFusion) with the H100-run patch
// clusterfusion__residual_out_c4_fix applied (the "cluster_reduce hardening"),
// over the CUDA porting layer.
//
// Addresses: src_addr, dst_addr and barrier are 32-bit .shared::cluster
// addresses of the calling block's own variables (__cvta_generic_to_shared);
// mapa moves them to a peer. src and dst are the same variables as pointers.
//
// Departures:
// - the thread index is read here instead of being passed in (the porting
//   layer's divergence rule);
// - the FFN stage, which neither decoder kernel uses, and the unknown-stage
//   assert are left out;
// - the barrier and copy instructions are the porting layer's cuda::ptx calls
//   (mbarrier.init, arrive.expect_tx, try_wait.parity, the shared::cta to
//   shared::cluster bulk copy, fence.proxy.async.shared::cta,
//   fence.mbarrier_init.release.cluster) instead of inline PTX.

#pragma once

#include <vx_cuda.h>

namespace cf {

namespace cg = cooperative_groups;

enum class Stage { LINEAR, ATTN, LINEAR_DEEPSEEK, QUK_DEEPSEEK, ATTN_DEEPSEEK };

namespace dsm {

// The caller's own variable at the offset of cluster address `a`.
__device__ __forceinline__ void* own(uint32_t a) { return (char*)__local_mem() + (a & VX_DLMEM_OFFSET_MASK); }
// Any block's variable named by cluster address `a`.
__device__ __forceinline__ void* any(uint32_t a) { return vx_cvta_cluster_to_generic(a); }

__device__ __forceinline__ void mbar_init_one(uint32_t bar) { cuda::ptx::mbarrier_init((uint64_t*)own(bar), 1); }

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  (void)cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared,
                                             (uint64_t*)own(bar), bytes);
}

__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
  while (!cuda::ptx::mbarrier_try_wait_parity((uint64_t*)own(bar), parity)) {
  }
}

// cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes:
// `bytes` from the own src into the cluster address dst, completing on the
// cluster address bar.
__device__ __forceinline__ void bulk_push(uint32_t dst, uint32_t src, uint32_t bytes, uint32_t bar) {
  cuda::ptx::cp_async_bulk(cuda::ptx::space_cluster, cuda::ptx::space_shared, any(dst), own(src), bytes,
                           (uint64_t*)any(bar));
}

__device__ __forceinline__ void fence_async_smem() { cuda::ptx::fence_proxy_async(cuda::ptx::space_shared); }

} // namespace dsm

// Every block of the cluster calls this with the same arguments shape.
// QUK_DEEPSEEK is an all-gather: N - 1 rounds, each a fresh single-arrival
// mbarrier that expects one push of `size` bytes from block rank - i. Every
// other stage is a reduce: N - 2 rounds on one mbarrier initialised once, each
// pushing the running sum to block rank + i and adding the block's own
// arrival into src. That sums every block only for N = 4 (rank r holds
// r + (r-1) after round 1 and all four after round 2), the cluster both
// decoders are built for.
template <int cluster_size, Stage stage>
__device__ __forceinline__ void cluster_reduce(const uint32_t size, const uint32_t tile_size,
                                               const uint32_t cluster_block_id, const uint32_t src_addr,
                                               const uint32_t dst_addr, uint32_t barrier, uint32_t neighbor_dst_bar,
                                               half* src, half* dst) {
  cg::cluster_group cluster = cg::this_cluster();
  const uint32_t tid = threadIdx.x;
  uint32_t dst_cta, neighbor_dst_addr;
  half2 buffer;
  half __align__(16) reg_input[8];

  if constexpr (stage == Stage::QUK_DEEPSEEK) {
    for (uint32_t i = 1; i < cluster.num_blocks(); i++) {
      if (tid == 0) {
        dsm::mbar_init_one(barrier);
        dsm::mbar_expect_tx(barrier, size);
      }
      cluster.sync();
      if (tid == 0) {
        dst_cta = (cluster_block_id + i) % cluster.num_blocks();
        neighbor_dst_addr = vx_mapa_u32(dst_addr, dst_cta);
        neighbor_dst_bar = vx_mapa_u32(barrier, dst_cta);
        dsm::bulk_push(neighbor_dst_addr, src_addr, size, neighbor_dst_bar);
      }
      dsm::mbar_wait_parity(barrier, 0);
      cluster.sync();
    }
    (void)tile_size;
    (void)src;
    (void)dst;
    (void)buffer;
    (void)reg_input;
  } else {
    // The caller wrote src with ordinary stores; the bulk copies read it
    // through the async proxy.
    dsm::fence_async_smem();
    if (tid == 0) {
      dsm::mbar_init_one(barrier);
      // The peers' complete_tx must see the initialised barrier.
      cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster);
    }
    cluster.sync();
    for (uint32_t i = 1; i < cluster.num_blocks() - 1; i++) {
      if (tid == 0) {
        dsm::mbar_expect_tx(barrier, size);
        dst_cta = (cluster_block_id + i) % cluster.num_blocks();
        neighbor_dst_addr = vx_mapa_u32(dst_addr, dst_cta);
        neighbor_dst_bar = vx_mapa_u32(barrier, dst_cta);
        dsm::bulk_push(neighbor_dst_addr, src_addr, size, neighbor_dst_bar);
      }
      // Round i completes phase i - 1 of the barrier.
      dsm::mbar_wait_parity(barrier, i ^ 1);
      // The barrier covers only what arrived here; this block's own push out
      // of src is tracked by the peer. Every block must have received before
      // anyone overwrites its src below.
      cluster.sync();

      if constexpr (stage == Stage::LINEAR) {
        if (tid < tile_size / 2) {
          for (int j = 0; j < 3; j++) {
            buffer = *(half2*)(&dst[j * tile_size + tid * 2]);
            *(half2*)(&src[j * tile_size + tid * 2]) = __hadd2(*(half2*)(&src[j * tile_size + tid * 2]), buffer);
          }
        }
      } else if constexpr (stage == Stage::ATTN) {
        if (tid < tile_size / 2) {
          buffer = *(half2*)(&dst[tid * 2]);
          *(half2*)(&src[tid * 2]) = __hadd2(*(half2*)(&src[tid * 2]), buffer);
        }
      } else if constexpr (stage == Stage::LINEAR_DEEPSEEK) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&dst[tid * 8]);
        for (int di = 0; di < 8; di++)
          src[tid * 8 + di] = __hadd(src[tid * 8 + di], reg_input[di]);
        src[tile_size + tid] = __hadd(src[tile_size + tid], dst[tile_size + tid]);
      } else {
        static_assert(stage == Stage::ATTN_DEEPSEEK, "a reduce stage");
        if (tid < tile_size / 8) {
          *(uint4*)(&reg_input[0]) = *(uint4*)(&dst[tid * 8]);
          for (int di = 0; di < 8; di++)
            src[tid * 8 + di] = __hadd(src[tid * 8 + di], reg_input[di]);
        }
      }

      // The next round's copy reads the sum just stored into src.
      dsm::fence_async_smem();
      cluster.sync();
    }
  }
}

// The scalar cluster all-reduces of both kernels (sum and max of one shared
// float `slot`), written inline upstream: round i takes this block's running
// value into block rank + i, by a remote f32 atomic add or a remote load and
// store of the maximum. Thread 0 alone keeps `local` and `dst_shmem` current.
__device__ __forceinline__ void cluster_sum_rounds(float& slot, float& local, float*& dst_shmem,
                                                   uint32_t cluster_block_id) {
  cg::cluster_group cluster = cg::this_cluster();
  for (uint32_t i = 1; i < cluster.num_blocks() - 1; i++) {
    if (threadIdx.x == 0) {
      local = slot;
      int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
      dst_shmem = cluster.map_shared_rank(&slot, dst_cta);
    }
    cluster.sync();
    if (threadIdx.x == 0) {
      atomicAdd(dst_shmem, local);
    }
    cluster.sync();
  }
}

// The same rounds for the maximum, as a remote load and store.
__device__ __forceinline__ void cluster_max_rounds(float& slot, float& local, float*& dst_shmem,
                                                   uint32_t cluster_block_id) {
  cg::cluster_group cluster = cg::this_cluster();
  for (uint32_t i = 1; i < cluster.num_blocks() - 1; i++) {
    if (threadIdx.x == 0) {
      local = slot;
      int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
      dst_shmem = cluster.map_shared_rank(&slot, dst_cta);
    }
    cluster.sync();
    if (threadIdx.x == 0) {
      *dst_shmem = fmaxf(*dst_shmem, local);
    }
    cluster.sync();
  }
}

} // namespace cf
