// AllReduce (sum) by remote atomics: a block initialises its accumulator with
// its input, then adds that input into the accumulator of each peer
// with one remote atomic per (peer, element); two cluster barriers frame it.
// Port of h100-dsmem-collectives/collectives/allreduce__atomic_all_push__generic.cu
// (commit 2b9a97e); origin: generic one-shot schedule of ClusterFusion's
// scalar remote-atomicAdd ClusterReduce.
// Departures: red.shared::cluster.add.f32 is a relaxed float AMO on the
// cluster window with its result discarded (dsmem::red_cluster_add_f32), the
// Vortex form of a fire-and-forget remote float add.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <type_traits>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__atomic_all_push {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  ELEM_T mine[M];
  ELEM_T acc[M];
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
  T* acc = sm.acc;
  const uint32_t rank = cluster_rank();
  Stamps st{ts + rank * kStamps};

  load_block<T, M>(mine, in + rank * M);
  for (int i = threadIdx.x; i < M; i += (int)blockDim.x)
    acc[i] = mine[i];
  if (threadIdx.x == 0)
    smid_out[rank] = smid();
  cluster_sync();
  st.mark(0);

  const uint32_t acc_a = smem_u32(acc);
  // Every accumulator holds its block's input before any remote add lands.
  cluster_sync();
  for (int idx = threadIdx.x; idx < (N - 1) * M; idx += (int)blockDim.x) {
    const uint32_t peer = (rank + 1 + idx / M) % N;
    const int e = idx % M;
    if constexpr (std::is_same<T, float>::value) {
      red_cluster_add_f32(mapa(acc_a + e * sizeof(T), peer), mine[e]);
    } else {
      atomicAdd(map_ptr(&acc[e], peer), mine[e]);
    }
  }
  st.round(0);
  cluster_sync();

  st.mark(1);
  store_block<T, M>(out + rank * M, acc);
  cluster_sync();
}
#endif

} // namespace allreduce__atomic_all_push
