// AllGather, Bruck: a block starts with its own vector in slot 0. In round
// d = 1, 2, 4, ... it copies its leading min(d, N-d) slots into slot d onward
// of block (b - d) mod N. Afterwards rank (b + i) mod N's vector sits in slot
// i, and the final store puts the slots back in rank order. Any N.
// Port of h100-dsmem-collectives/collectives/allgather__bruck__mpich.cu
// (commit 2b9a97e); origin: Bruck et al. 1997, MPICH MPI_Allgather.
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "../dsmem_primitives.h"

namespace allgather__bruck {

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
  for (int d = 1; d < N; d <<= 1, ++round) {
    const uint32_t bytes = (uint32_t)(d < N - d ? d : N - d) * blk;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(bar, bytes);
      bulk_push((me + N - d) % N, D_a + d * blk, D_a, bytes, bar);
    }
    mbar_wait_parity(bar, parity); // block (b + d)'s slots are in from slot d on
    parity ^= 1;
    cluster_sync();
    st.round(round);
  }

  st.mark(1);
  T* o = out + me * N * M;
  for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
    const int k = i / M, e = i % M;
    o[((me + k) % N) * M + e] = D[k * M + e];
  }
  cluster_sync();
}
#endif

} // namespace allgather__bruck
