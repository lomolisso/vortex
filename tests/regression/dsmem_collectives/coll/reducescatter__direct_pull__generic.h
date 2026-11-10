// ReduceScatter, direct pull: after one cluster barrier, block b reads chunk b of every
// block's input through the cluster window and reduces it in registers; no pushes, no mbarrier.
// Port of h100-dsmem-collectives/collectives/reducescatter__direct_pull__generic.cu (commit 2b9a97e),
// a generic algorithm (the pull phase of hpc-ops' histogram reduce).
// Departures: each 16-byte remote load is two 64-bit window loads
// (dsmem::ld_cluster_v4); the inputs do not change between the two barriers,
// so the split is not observable.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace reducescatter__direct_pull {

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T buf[N * M];
  alignas(16) ELEM_T result[M];
};

template <int N, int M, class T>
constexpr const char* illegal() {
  return ((M * sizeof(T)) % 16 != 0) ? "vectorised 16-byte remote loads" : nullptr;
}

constexpr int kOp = 2;
constexpr bool kTypeAgnostic = false;

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid) {
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "shape differs from CollectiveSmem");
  constexpr int V = 16 / sizeof(T);
  CollectiveSmem& sm = dsmem::dyn_smem<CollectiveSmem>();
  T* buf = sm.buf;
  T* result = sm.result;
  const Op op{};
  const uint32_t rank = dsmem::cluster_rank();
  const dsmem::Stamps st{ts + rank * dsmem::kStamps};

  dsmem::load_block<T, N * M>(buf, in + rank * N * M);
  if (threadIdx.x == 0)
    smid[rank] = dsmem::smid();
  dsmem::cluster_sync();
  st.mark(0);

  const uint32_t chunk_a = dsmem::smem_u32(buf + rank * M);
  dsmem::cluster_sync();
  for (int v = threadIdx.x; v < M / V; v += blockDim.x) {
    T acc[V];
#pragma unroll
    for (int k = 0; k < V; ++k)
      acc[k] = Op::template identity<T>();
#pragma unroll 1
    for (int r = 0; r < N; ++r) {
      const auto raw = dsmem::ld_cluster_v4(dsmem::mapa(chunk_a + v * 16, r));
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
  dsmem::cluster_sync();  // every peer has finished reading my input

  st.mark(1);
  dsmem::store_block<T, M>(out + rank * M, result);
  dsmem::cluster_sync();
}
#endif

}  // namespace reducescatter__direct_pull
