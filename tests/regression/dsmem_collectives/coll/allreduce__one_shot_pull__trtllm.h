// AllReduce by one-shot pull: once a cluster barrier has made every input
// visible, each block fetches every element from all N blocks (its own
// included) in 16-byte DSMEM loads and sums them in registers; a second
// barrier holds the buffers in place until the last peer read.
// Port of h100-dsmem-collectives/collectives/allreduce__one_shot_pull__trtllm.cu
// (commit 2b9a97e); origin: TensorRT-LLM customAllReduce one-shot.
// Departures: the 16-byte ld.shared::cluster.v4 is two 64-bit window loads
// (dsmem::ld_cluster_v4); the buffers are stable between the two barriers, so
// the split is not observable.
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__one_shot_pull {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T buf[M];
  alignas(16) ELEM_T result[M];
};

template <int N, int M, class T>
constexpr const char* illegal() {
  if ((M * sizeof(T)) % 16 != 0)
    return "vectorised 16-byte remote loads";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "run() must match the build's CLUSTER_N and MSG_ELEMS");
  static_assert((M * sizeof(T)) % 16 == 0, "vectorised 16-byte remote loads");
  constexpr int V = 16 / sizeof(T);

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* buf = sm.buf;
  T* result = sm.result;
  const Op op{};
  const uint32_t rank = cluster_rank();
  Stamps st{ts + rank * kStamps};

  load_block<T, M>(buf, in + rank * M);
  if (threadIdx.x == 0)
    smid_out[rank] = smid();
  cluster_sync();
  st.mark(0);

  const uint32_t buf_a = smem_u32(buf);
  cluster_sync();
  for (int v = threadIdx.x; v < M / V; v += (int)blockDim.x) {
    T acc[V];
#pragma unroll
    for (int k = 0; k < V; ++k)
      acc[k] = Op::template identity<T>();
#pragma unroll 1
    for (int r = 0; r < N; ++r) {
      const auto raw = ld_cluster_v4(mapa(buf_a + v * 16, r));
      const T* vals = reinterpret_cast<const T*>(&raw);
#pragma unroll
      for (int k = 0; k < V; ++k)
        acc[k] = op(acc[k], vals[k]);
    }
#pragma unroll
    for (int k = 0; k < V; ++k)
      result[v * V + k] = acc[k];
  }
  st.round(0);
  cluster_sync();

  st.mark(1);
  store_block<T, M>(out + rank * M, result);
  cluster_sync();
}
#endif

} // namespace allreduce__one_shot_pull
