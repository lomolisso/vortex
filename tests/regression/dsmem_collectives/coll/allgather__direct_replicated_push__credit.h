// AllGather by replicated pushes: every block stores its vector into slot b of
// every block (itself included) with plain remote stores through
// map_shared_rank pointers; one cluster barrier completes the gather.
// Port of h100-dsmem-collectives/collectives/
// allgather__direct_replicated_push__credit.cu (commit 2b9a97e); origin:
// CREDIT cluster_reduce_pair, the gather half.
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "../dsmem_primitives.h"

namespace allgather__direct_replicated_push {

constexpr int kOp = 1;
constexpr bool kTypeAgnostic = true;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  ELEM_T mine[M];
  ELEM_T slots[N][M];
};

template <int N, int M, class T>
constexpr const char* illegal() {
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* mine = sm.mine;
  T (*slots)[M] = sm.slots;
  const uint32_t me = cluster_rank();
  const Stamps st{ts + me * kStamps};

  load_block<T, M>(mine, in + me * M);
  if (threadIdx.x == 0)
    smid_out[me] = smid();
  cluster_sync();
  st.mark(0);

  for (int idx = threadIdx.x; idx < N * M; idx += blockDim.x) {
    const int p = idx / M, e = idx % M;
    *map_ptr(&slots[me][e], p) = mine[e]; // element e into slot b of block p
  }
  cluster_sync(); // every slot of every block filled
  st.round(0);

  st.mark(1);
  store_block<T, N * M>(out + me * N * M, &slots[0][0]);
  cluster_sync();
}
#endif

} // namespace allgather__direct_replicated_push
