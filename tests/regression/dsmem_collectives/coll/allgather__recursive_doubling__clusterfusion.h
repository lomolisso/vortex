// AllGather, recursive doubling: a block starts with its vector in segment
// 0. With stride s = 1, 2, 4, ... it copies segments 0 .. s-1 into segments
// s .. 2s-1 of block (b + s) mod N. In the end segment k holds the vector of
// rank (b - k) mod N, and the store restores rank order. N is a power of two.
// Port of h100-dsmem-collectives/collectives/
// allgather__recursive_doubling__clusterfusion.cu (commit 2b9a97e); origin:
// ClusterFusion (NeurIPS'25) Algorithm 2 "ClusterGather".
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "../dsmem_primitives.h"

namespace allgather__recursive_doubling {

constexpr int kOp = 1;
constexpr bool kTypeAgnostic = true;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  __align__(16) ELEM_T D[N * M];
  __align__(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  if ((N & (N - 1)) != 0)
    return "recursive doubling needs a power-of-two cluster size";
  if ((M * sizeof(T)) % 16 != 0)
    return "bulk copy needs a multiple of 16 bytes";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(illegal<N, M, T>() == nullptr, "illegal build");

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* D = sm.D;
  const uint32_t me = cluster_rank();
  const Stamps st{ts + me * kStamps};

  load_block<T, M>(D, in + me * M);
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
  int round = 0;
  for (int s = 1; s < N; s <<= 1, ++round) {
    const uint32_t bytes = (uint32_t)s * M * sizeof(T);
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(bar, bytes);
      bulk_push((me + s) % N, D_a + bytes, D_a, bytes, bar);
    }
    mbar_wait_parity(bar, parity); // segments of (b - s) landed at [s, 2s)
    parity ^= 1;
    cluster_sync(); // keeps every send range stable for the round
    st.round(round);
  }

  st.mark(1);
  T* o = out + me * N * M;
  for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
    const int k = i / M, e = i % M;
    o[((me + N - k) % N) * M + e] = D[k * M + e];
  }
  cluster_sync();
}
#endif

} // namespace allgather__recursive_doubling
