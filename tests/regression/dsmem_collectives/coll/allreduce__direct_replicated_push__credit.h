// AllReduce by replicated push: plain remote stores put block b's vector in
// slots[b] of all N blocks (its own too); after one cluster barrier a block
// sums the N slots it holds, and a second barrier keeps the slots untouched
// until every block has read them.
// Port of h100-dsmem-collectives/collectives/allreduce__direct_replicated_push__credit.cu
// (commit 2b9a97e); origin: CREDIT cluster_reduce_pair (also hpc-ops
// rmsNormLamport).
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__direct_replicated_push {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  ELEM_T mine[M];
  ELEM_T slots[N][M];
  ELEM_T result[M];
};

template <int N, int M, class T>
constexpr const char* illegal() {
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "run() must match the build's CLUSTER_N and MSG_ELEMS");

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* mine = sm.mine;
  T* result = sm.result;
  const Op op{};
  const uint32_t rank = cluster_rank();
  Stamps st{ts + rank * kStamps};

  load_block<T, M>(mine, in + rank * M);
  if (threadIdx.x == 0)
    smid_out[rank] = smid();
  cluster_sync();
  st.mark(0);

  // Element e of this block's vector goes to slots[rank][e] of block p.
  for (int idx = threadIdx.x; idx < N * M; idx += (int)blockDim.x) {
    const int p = idx / M, e = idx % M;
    T* remote = map_ptr(&sm.slots[rank][e], p);
    *remote = mine[e];
  }
  cluster_sync();
  st.round(0);
  for (int e = threadIdx.x; e < M; e += (int)blockDim.x) {
    T acc = sm.slots[0][e];
#pragma unroll
    for (int r = 1; r < N; ++r)
      acc = op(acc, sm.slots[r][e]);
    result[e] = acc;
  }
  cluster_sync();

  st.mark(1);
  store_block<T, M>(out + rank * M, result);
  cluster_sync();
}
#endif

} // namespace allreduce__direct_replicated_push
