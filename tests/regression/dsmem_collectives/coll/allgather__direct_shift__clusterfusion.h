// AllGather by direct pushes with a rotating shift: own vector in segment b;
// in round i = 1 .. N-1 each block pushes segment b into segment b of block
// (b + i) mod N. Each round re-initialises the barrier, arms it, syncs the
// cluster, pushes, waits on parity 0 and syncs again, as shipped.
// Port of h100-dsmem-collectives/collectives/
// allgather__direct_shift__clusterfusion.cu (commit 2b9a97e); origin:
// ClusterFusion include/dsm.cuh, the QUK_DEEPSEEK branch.
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "../dsmem_primitives.h"

namespace allgather__direct_shift {

constexpr int kOp = 1;
constexpr bool kTypeAgnostic = true;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  __align__(16) ELEM_T D[N * M];
  __align__(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  if ((M * sizeof(T)) % 16 != 0)
    return "bulk copy needs a multiple of 16 bytes";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(illegal<N, M, T>() == nullptr, "illegal build");
  constexpr uint32_t blk = M * sizeof(T);

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* D = sm.D;
  const uint32_t me = cluster_rank();
  const Stamps st{ts + me * kStamps};

  load_block<T, M>(D + me * M, in + me * M);
  fence_proxy_async_smem();
  if (threadIdx.x == 0)
    smid_out[me] = smid();
  cluster_sync();
  st.mark(0);

  const uint32_t bar = smem_u32(&sm.bar);
  const uint32_t seg_a = smem_u32(D + me * M); // one offset in every block
  for (int i = 1; i < N; ++i) {
    if (threadIdx.x == 0) {
      mbar_init(bar, 1);
      fence_mbar_init();
      mbar_arrive_expect_tx(bar, blk); // expects block (b - i)'s segment
    }
    cluster_sync(); // no copy can reach a barrier that is not armed yet
    if (threadIdx.x == 0)
      bulk_push((me + i) % N, seg_a, seg_a, blk, bar);
    mbar_wait_parity(bar, 0);
    cluster_sync();
    st.round(i - 1);
  }

  st.mark(1);
  store_block<T, N * M>(out + me * N * M, D);
  cluster_sync();
}
#endif

} // namespace allgather__direct_shift
