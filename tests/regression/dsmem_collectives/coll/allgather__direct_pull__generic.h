// AllGather by direct pulls: once a cluster barrier has made every input
// visible, all threads of a block fetch the peers' vectors from their shared
// memory in 16-byte DSMEM loads, the own vector is copied locally, and a
// second barrier holds the inputs in place until the last read.
// Port of h100-dsmem-collectives/collectives/allgather__direct_pull__generic.cu
// (commit 2b9a97e); origin: generic.
// Departures: each 16-byte remote load is two 64-bit window loads
// (dsmem::ld_cluster_v4), and its 16-byte local store is a 16-byte copy (two
// 64-bit stores): Vortex has no 128-bit access. The inputs do not change
// between the two barriers and D is read only after one, so neither split
// is observable.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "../dsmem_primitives.h"

namespace allgather__direct_pull {

constexpr int kOp = 1;
constexpr bool kTypeAgnostic = true;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  __align__(16) ELEM_T mine[M];
  __align__(16) ELEM_T D[N * M];
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
  static_assert(illegal<N, M, T>() == nullptr, "illegal build");
  constexpr int units = M * sizeof(T) / 16;

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* mine = sm.mine;
  T* D = sm.D;
  const uint32_t me = cluster_rank();
  const Stamps st{ts + me * kStamps};

  load_block<T, M>(mine, in + me * M);
  if (threadIdx.x == 0)
    smid_out[me] = smid();
  cluster_sync();
  st.mark(0);

  const uint32_t mine_a = smem_u32(mine);
  cluster_sync(); // every peer's input is in place
  for (int idx = threadIdx.x; idx < N * units; idx += blockDim.x) {
    const int r = idx / units, u = idx % units;
    if (r == (int)me)
      continue; // taken by the local loop that follows
    const uint4 v = ld_cluster_v4(mapa(mine_a + u * 16, r));
    __builtin_memcpy((char*)(D + r * M) + u * 16, &v, 16);
  }
  for (int i = threadIdx.x; i < M; i += blockDim.x)
    D[me * M + i] = mine[i];
  st.round(0);
  cluster_sync(); // no peer still reads my input

  st.mark(1);
  store_block<T, N * M>(out + me * N * M, D);
  cluster_sync();
}
#endif

} // namespace allgather__direct_pull
