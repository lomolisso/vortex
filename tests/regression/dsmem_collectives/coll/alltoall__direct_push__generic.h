// AlltoAll, direct push: for each j != b, block b bulk-copies chunk j into recv[b] of block j;
// chunk b stays local. A block's single mbarrier is armed once for all (N-1)*M*sizeof(T)
// incoming bytes and completes with the last of them. One round, no cluster barrier inside.
// Port of h100-dsmem-collectives/collectives/alltoall__direct_push__generic.cu (commit 2b9a97e),
// a generic algorithm.
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace alltoall__direct_push {

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T chunks[N * M];
  alignas(16) ELEM_T recv[N * M];
  alignas(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  return ((M * sizeof(T)) % 16 != 0) ? "bulk copy needs a multiple of 16 bytes" : nullptr;
}

constexpr int kOp = 3;
constexpr bool kTypeAgnostic = true;

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid) {
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "shape differs from CollectiveSmem");
  constexpr uint32_t bytes = M * sizeof(T);
  CollectiveSmem& sm = dsmem::dyn_smem<CollectiveSmem>();
  T* chunks = sm.chunks;
  T* recv = sm.recv;
  const uint32_t bar = dsmem::smem_u32(&sm.bar);
  const uint32_t rank = dsmem::cluster_rank();
  const dsmem::Stamps st{ts + rank * dsmem::kStamps};

  dsmem::load_block<T, N * M>(chunks, in + rank * N * M);
  dsmem::fence_proxy_async_smem();
  if (threadIdx.x == 0) {
    dsmem::mbar_init(bar, 1);
    dsmem::fence_mbar_init();
    smid[rank] = dsmem::smid();
  }
  dsmem::cluster_sync();
  st.mark(0);

  const uint32_t src_a = dsmem::smem_u32(chunks);
  const uint32_t recv_a = dsmem::smem_u32(recv);
  if (threadIdx.x == 0) {
    dsmem::mbar_arrive_expect_tx(bar, (N - 1) * bytes);
    for (uint32_t j = 1; j < N; ++j) {
      const uint32_t dst = (rank + j) % N;
      dsmem::bulk_push(dst, recv_a + rank * bytes, src_a + dst * bytes, bytes, bar);
    }
  }
  for (int i = threadIdx.x; i < M; i += blockDim.x)
    recv[rank * M + i] = chunks[rank * M + i];  // own chunk
  dsmem::mbar_wait_parity(bar, 0);
  st.round(0);

  st.mark(1);
  dsmem::store_block<T, N * M>(out + rank * N * M, recv);
  dsmem::cluster_sync();
}
#endif

}  // namespace alltoall__direct_push
