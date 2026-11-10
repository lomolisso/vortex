// AllReduce (sum) by shift-and-accumulate with remote atomics: in round i
// block b snapshots its running value and atomically adds it into the value
// of block (b + i) mod N; one round for N = 2, N - 2 rounds for N = 4. Two
// cluster barriers per round separate the snapshot from the adds and the
// adds from the next snapshot.
// Port of h100-dsmem-collectives/collectives/allreduce__shift_accumulate_atomic__clusterfusion.cu
// (commit 2b9a97e); origin: ClusterFusion llama/norm kernels (ClusterReduce
// loops, scalar path).
// Departures: none (the float atomicAdd on the peer's shared memory is a
// relaxed float AMO on the cluster window).
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__shift_accumulate_atomic {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  ELEM_T val[M];
  ELEM_T snap[M];
};

template <int N, int M, class T>
constexpr const char* illegal() {
  if (!(N == 2 || N == 4))
    return "shipped schedule: 1 round for N=2 (norm kernel), 2 rounds for N=4 (llama kernel)";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "run() must match the build's CLUSTER_N and MSG_ELEMS");
  static_assert(N == 2 || N == 4, "shipped schedule: 1 round for N=2 (norm kernel), 2 rounds for N=4 (llama kernel)");
  constexpr int rounds = (N == 2) ? 1 : N - 2;

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* val = sm.val;
  T* snap = sm.snap;
  const uint32_t rank = cluster_rank();
  Stamps st{ts + rank * kStamps};

  load_block<T, M>(val, in + rank * M);
  if (threadIdx.x == 0)
    smid_out[rank] = smid();
  cluster_sync();
  st.mark(0);

  for (int i = 1; i <= rounds; ++i) {
    for (int e = threadIdx.x; e < M; e += (int)blockDim.x)
      snap[e] = val[e];
    T* peer = map_ptr(val, (rank + i) % N);
    cluster_sync();
    for (int e = threadIdx.x; e < M; e += (int)blockDim.x)
      atomicAdd(&peer[e], snap[e]);
    cluster_sync();
    st.round(i - 1);
  }

  st.mark(1);
  store_block<T, M>(out + rank * M, val);
  cluster_sync();
}
#endif

} // namespace allreduce__shift_accumulate_atomic
