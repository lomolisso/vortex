// AlltoAll for N = 2, pairwise half-buffer exchange: block b keeps chunk b and bulk-copies chunk
// b^1 into the peer's recv buffer. The sender arms the receiver's mbarrier itself (a remote
// arrive.expect_tx) right before its copy; the receiver then polls only its local barrier. One
// round, no cluster barrier inside the collective.
// Port of h100-dsmem-collectives/collectives/alltoall__pairwise_exchange__fa4.cu (commit 2b9a97e),
// algorithm from FlashAttention-4 (2-CTA backward dS exchange).
// Departures: the remote arrive.expect_tx is the .release.cluster form, which drains a cluster
// fence before the arm (dsmem::mbar_arrive_expect_tx_remote; libcu++ names only the cluster scope
// for a remote arrive, the H100 default is .release.cta). The ordering is stronger; the arm,
// and so the receiver's completion, can come later than on the H100.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace alltoall__pairwise_exchange {

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T chunks[N * M];  // rank 0's half, then rank 1's
  alignas(16) ELEM_T recv[M];
  alignas(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  return (N != 2) ? "pairwise exchange is the N = 2 case; see alltoall__pairwise_xor__mpich.cu for general N"
         : ((M * sizeof(T)) % 16 != 0) ? "bulk copy needs a multiple of 16 bytes"
                                       : nullptr;
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
  if (threadIdx.x == 0) {
    dsmem::mbar_init(bar, 1);
    dsmem::fence_mbar_init();
    smid[rank] = dsmem::smid();
  }
  dsmem::cluster_sync();
  st.mark(0);

  const uint32_t peer = rank ^ 1;
  const uint32_t stage = rank ^ 1;  // the chunk destined for the peer
  dsmem::fence_proxy_async_smem();
  __syncthreads();
  if (threadIdx.x == 0) {
    dsmem::mbar_arrive_expect_tx_remote(bar, peer, bytes);  // the sender arms the receiver's barrier
    dsmem::bulk_push(peer, dsmem::smem_u32(recv), dsmem::smem_u32(chunks + stage * M), bytes, bar);
  }
  dsmem::mbar_wait_parity(bar, 0);  // recv is complete
  st.round(0);

  st.mark(1);
  T* o = out + rank * N * M;
  dsmem::store_block<T, M>(o + rank * M, chunks + rank * M);
  dsmem::store_block<T, M>(o + (rank ^ 1) * M, recv);
  dsmem::cluster_sync();
}
#endif

}  // namespace alltoall__pairwise_exchange
