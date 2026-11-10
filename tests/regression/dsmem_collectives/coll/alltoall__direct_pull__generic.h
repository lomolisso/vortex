// AlltoAll, direct pull: once a cluster barrier has made every send buffer visible, block b
// reads chunk b of each peer r (one offset in all blocks) through the cluster window into recv[r];
// its own chunk is a local copy. No pushes, no mbarrier; two cluster barriers.
// Port of h100-dsmem-collectives/collectives/alltoall__direct_pull__generic.cu (commit 2b9a97e),
// a generic algorithm.
// Departures: each 16-byte unit is two 64-bit window loads when remote (dsmem::ld_cluster_v4) and a
// struct copy of narrower loads when local, and its store into recv is a struct copy of narrower
// stores: Vortex has no 128-bit access. The send buffers do not change between the two barriers
// and recv is read only after one, so no split is observable.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace alltoall__direct_pull {

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T chunks[N * M];
  alignas(16) ELEM_T recv[N * M];
};

template <int N, int M, class T>
constexpr const char* illegal() {
  return ((M * sizeof(T)) % 16 != 0) ? "vectorised 16-byte remote loads" : nullptr;
}

constexpr int kOp = 3;
constexpr bool kTypeAgnostic = true;

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid) {
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "shape differs from CollectiveSmem");
  using V4 = decltype(dsmem::ld_cluster_v4(0u));
  constexpr int V16 = M * sizeof(T) / 16;
  CollectiveSmem& sm = dsmem::dyn_smem<CollectiveSmem>();
  T* chunks = sm.chunks;
  T* recv = sm.recv;
  const uint32_t rank = dsmem::cluster_rank();
  const dsmem::Stamps st{ts + rank * dsmem::kStamps};

  dsmem::load_block<T, N * M>(chunks, in + rank * N * M);
  if (threadIdx.x == 0)
    smid[rank] = dsmem::smid();
  dsmem::cluster_sync();
  st.mark(0);

  const uint32_t my_chunk_a = dsmem::smem_u32(chunks + rank * M);
  dsmem::cluster_sync();
  for (int idx = threadIdx.x; idx < N * V16; idx += blockDim.x) {
    const int r = idx / V16, u = idx % V16;
    const V4 v = (r == (int)rank) ? reinterpret_cast<const V4*>(chunks + rank * M)[u]
                                  : dsmem::ld_cluster_v4(dsmem::mapa(my_chunk_a + u * 16, r));
    reinterpret_cast<V4*>(recv + r * M)[u] = v;
  }
  st.round(0);
  dsmem::cluster_sync();  // no peer still reads my send buffer

  st.mark(1);
  dsmem::store_block<T, N * M>(out + rank * N * M, recv);
  dsmem::cluster_sync();
}
#endif

}  // namespace alltoall__direct_pull
