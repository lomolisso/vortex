#ifndef _DEEPGEMM_FP8_1D2D_DG_CUTLASS_H_
#define _DEEPGEMM_FP8_1D2D_DG_CUTLASS_H_

// The CUTLASS, CuTe and DeepGEMM device names the kernel uses, each over the
// vx_cuda.h primitive with the same PTX semantics. Device only.

#include <vx_cuda.h>

namespace dg {

namespace ptx = ::cuda::ptx;

// cutlass::arch::ClusterTransactionBarrier: one 64-bit mbarrier word.
struct ClusterTransactionBarrier {
  uint64_t word;

  __forceinline__ void init(uint32_t arrive_count) { ptx::mbarrier_init(&word, arrive_count); }

  // try_wait.parity in a loop, as CUTLASS spins on it.
  __forceinline__ void wait(uint32_t phase) {
    while (!ptx::mbarrier_try_wait_parity(&word, phase)) {
    }
  }

  __forceinline__ void arrive() { (void)ptx::mbarrier_arrive(&word); }

  // mapa + mbarrier.arrive.shared::cluster: a plain (release.cta) arrive on
  // the same object in block cta_id of the cluster, through its window
  // address (the caller's own block included).
  __forceinline__ void arrive(uint32_t cta_id) {
    (void)ptx::mbarrier_arrive(cooperative_groups::this_cluster().map_shared_rank(&word, cta_id));
  }

  __forceinline__ void arrive_and_expect_tx(uint32_t bytes) {
    (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &word, bytes);
  }
};

// cutlass::arch::NamedBarrier: user barrier id + the 8 ids CUTLASS reserves.
constexpr uint32_t kReservedNamedBarrierCount = 8;
struct NamedBarrier {
  static __forceinline__ void sync(uint32_t num_threads, uint32_t barrier_id) {
    bar_sync(barrier_id + kReservedNamedBarrierCount, num_threads);
  }
};

__forceinline__ void fence_barrier_init() { ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster); }

__forceinline__ void cluster_arrive_relaxed() { ptx::barrier_cluster_arrive(ptx::sem_relaxed); }
__forceinline__ void cluster_wait() { ptx::barrier_cluster_wait(); }

// deep_gemm::comm::cluster_sync_with_relaxed_arrive.
__forceinline__ void cluster_sync_with_relaxed_arrive() {
  cluster_arrive_relaxed();
  cluster_wait();
}

__forceinline__ uint32_t block_rank_in_cluster() { return cooperative_groups::this_cluster().block_rank(); }

template <uint32_t N>
__forceinline__ void warpgroup_reg_alloc() { ptx::setmaxnreg_inc(ptx::n32_t<(int)N>{}); }
template <uint32_t N>
__forceinline__ void warpgroup_reg_dealloc() { ptx::setmaxnreg_dec(ptx::n32_t<(int)N>{}); }

__forceinline__ void prefetch_tma_descriptor(const void* map) { ptx::prefetch_tensormap(map); }

// tma::copy for a 2-D box whose inner extent is one swizzle atom (or an
// unswizzled box): one tensor load; with multicast, only rank 0 issues it,
// to every block of the mask.
__forceinline__ void tma_copy_2d(const void* map, ClusterTransactionBarrier* bar, void* smem, uint32_t inner_idx,
                                 uint32_t outer_idx, uint32_t num_tma_multicast) {
  const int32_t c[2] = {(int32_t)inner_idx, (int32_t)outer_idx};
  if (num_tma_multicast == 1) {
    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, smem, map, c, &bar->word);
  } else if (block_rank_in_cluster() == 0) {
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem, map, c, &bar->word,
                              (uint16_t)((1u << num_tma_multicast) - 1));
  }
}

// cute::SM90_TMA_STORE_2D, tma_store_arrive / wait<0> / fence.
__forceinline__ void tma_store_2d(const void* map, const void* smem, uint32_t c0, uint32_t c1) {
  const int32_t c[2] = {(int32_t)c0, (int32_t)c1};
  ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared, map, c, smem);
}
__forceinline__ void tma_store_arrive() { ptx::cp_async_bulk_commit_group(); }
__forceinline__ void tma_store_wait_0() { ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>{}); }
__forceinline__ void tma_store_fence() { ptx::fence_proxy_async(ptx::space_shared); }

// deep_gemm::ptx helpers.
__forceinline__ void warpgroup_arrive() { cute::warpgroup_arrive(); }
__forceinline__ void warpgroup_commit_batch() { cute::warpgroup_commit_batch(); }
template <int N>
__forceinline__ void warpgroup_wait() { cute::warpgroup_wait<N>(); }
template <typename T>
__forceinline__ void warpgroup_fence_operand(T& x) { cute::warpgroup_fence_operand(x); }

__forceinline__ uint32_t get_lane_idx() { return __laneid(); }
__forceinline__ float ld_shared(const float* p) { return *(const volatile float*)p; }
__forceinline__ void st_shared(float* p, float v) { *(volatile float*)p = v; }

} // namespace dg

#endif
