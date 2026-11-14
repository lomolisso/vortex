#ifndef _DAK_CLUSTER_H_
#define _DAK_CLUSTER_H_

// The CUTLASS and CuTe cluster names the kernel uses, over vx_cuda.h. Each
// keeps its original's semantics; the namespace keeps them apart from any
// future library spelling of the same names.

#include <vx_cuda.h>

namespace dak {

// cutlass::arch::ClusterBarrier: a plain mbarrier word.
struct ClusterBarrier {
  using ValueType = uint64_t;

  static __forceinline__ void init(ValueType* bar, uint32_t count) { cuda::ptx::mbarrier_init(bar, count); }

  // try_wait.parity until the phase of that parity has completed.
  static __forceinline__ void wait(ValueType* bar, uint32_t phase) {
    while (!cuda::ptx::mbarrier_try_wait_parity(bar, phase)) {}
  }

  // mbarrier.arrive.shared::cluster on block cta_id's copy of bar (CTA-scope
  // release, no tx).
  static __forceinline__ void arrive(ValueType* bar, uint32_t cta_id, uint32_t pred) {
    if (pred)
      (void)cuda::ptx::mbarrier_arrive(cooperative_groups::this_cluster().map_shared_rank(bar, cta_id));
  }
};

// cute::cluster_sync: barrier.cluster.arrive + wait (release/acquire).
__forceinline__ void cluster_sync() { cooperative_groups::this_cluster().sync(); }

__forceinline__ uint32_t block_rank_in_cluster() { return cooperative_groups::this_cluster().block_rank(); }

// DAK's __sync_barrier<Id, N>: bar.sync Id, N.
template <uint32_t BarrierID, uint32_t Count>
__forceinline__ void sync_barrier() {
  bar_sync<BarrierID, Count>();
}

} // namespace dak

#endif
