// AllGather on a ring b -> (b + 1) mod N: a block starts with its vector in
// slot b; at step s one bulk copy moves slot (b - s) mod N to the identical
// slot of the next block. After N-1 steps every buffer is complete.
// Port of h100-dsmem-collectives/collectives/allgather__ring__nccl.cu
// (commit 2b9a97e); origin: NCCL ring all-gather.
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "../dsmem_primitives.h"

namespace allgather__ring {

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
  const uint32_t succ = (me + 1) % N;
  const Stamps st{ts + me * kStamps};

  load_block<T, M>(D + me * M, in + me * M);
  fence_proxy_async_smem();
  const uint32_t bar = smem_u32(&sm.bar);
  if (threadIdx.x == 0) {
    mbar_init(bar, 1);
    fence_mbar_init();
    smid_out[me] = smid();
  }
  cluster_sync();
  st.mark(0);

  const uint32_t D_a = smem_u32(D);
  uint32_t parity = 0;
  for (int s = 0; s < N - 1; ++s) {
    const uint32_t slot = (me + N - s) % N;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(bar, blk);
      bulk_push(succ, D_a + slot * blk, D_a + slot * blk, blk, bar);
    }
    mbar_wait_parity(bar, parity); // the predecessor's slot is in
    parity ^= 1;
    cluster_sync(); // re-arm only after the step ends
    st.round(s);
  }

  st.mark(1);
  store_block<T, N * M>(out + me * N * M, D);
  cluster_sync();
}
#endif

} // namespace allgather__ring
