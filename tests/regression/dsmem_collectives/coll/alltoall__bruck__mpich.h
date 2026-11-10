// AlltoAll, Bruck (N a power of two): chunk j is loaded at position (j-b) mod N; then log2(N)
// rounds with d = 1, 2, 4, ..: one thread bulk-copies the N/2 positions with bit d set
// to block (b+d) mod N (into its staging buffer, all completing on one mbarrier armed with the
// aggregate bytes), a cluster barrier, a local copy-back from staging, a second cluster barrier;
// finally position i is stored as the chunk of source (b-i) mod N.
// Port of h100-dsmem-collectives/collectives/alltoall__bruck__mpich.cu (commit 2b9a97e),
// algorithm from Bruck et al. 1997 as in MPICH's short-message MPI_Alltoall.
// Departures: the copy-back finds the s-th position with bit d set by a full scan instead of an
// early break (same position, a trip count equal on every thread).
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace alltoall__bruck {

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T D[N * M];
  alignas(16) ELEM_T tmp[N * M / 2];
  alignas(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  return ((N & (N - 1)) != 0)            ? "this Bruck variant assumes a power-of-two cluster size"
         : ((M * sizeof(T)) % 16 != 0)   ? "bulk copy needs a multiple of 16 bytes"
                                         : nullptr;
}

constexpr int kOp = 3;
constexpr bool kTypeAgnostic = true;

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid) {
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "shape differs from CollectiveSmem");
  constexpr uint32_t blk = M * sizeof(T);
  CollectiveSmem& sm = dsmem::dyn_smem<CollectiveSmem>();
  T* D = sm.D;
  T* tmp = sm.tmp;
  const uint32_t bar = dsmem::smem_u32(&sm.bar);
  const uint32_t rank = dsmem::cluster_rank();
  const dsmem::Stamps st{ts + rank * dsmem::kStamps};

  // The load rotates: input chunk j lands at position (j - rank) mod N.
  const T* src = in + rank * N * M;
  for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
    const int j = i / M, e = i % M;
    D[((j - (int)rank + N) % N) * M + e] = src[i];
  }
  __syncthreads();
  dsmem::fence_proxy_async_smem();
  if (threadIdx.x == 0) {
    dsmem::mbar_init(bar, 1);
    dsmem::fence_mbar_init();
    smid[rank] = dsmem::smid();
  }
  dsmem::cluster_sync();
  st.mark(0);

  const uint32_t D_a = dsmem::smem_u32(D);
  const uint32_t tmp_a = dsmem::smem_u32(tmp);
  uint32_t phase = 0;
  int round = 0;
  for (int d = 1; d < N; d <<= 1, ++round) {
    const uint32_t to = (rank + d) % N;
    if (threadIdx.x == 0) {
      dsmem::mbar_arrive_expect_tx(bar, (N / 2) * blk);
      int s = 0;
      for (int pos = 0; pos < N; ++pos) {
        if (pos & d) {
          dsmem::bulk_push(to, tmp_a + s * blk, D_a + pos * blk, blk, bar);  // position pos -> tmp[s] of block (b+d)
          ++s;
        }
      }
    }
    dsmem::mbar_wait_parity(bar, phase);  // block (b-d)'s N/2 positions are in tmp
    phase ^= 1;
    dsmem::cluster_sync();                // all copies of the round have landed, so the positions I sent are free
    for (int i = threadIdx.x; i < (N / 2) * M; i += blockDim.x) {
      const int s = i / M, e = i % M;
      int pos = 0, c = 0;
      for (int p = 0; p < N; ++p) {
        if (p & d) {
          if (c == s)
            pos = p;
          ++c;
        }
      }
      D[pos * M + e] = tmp[i];
    }
    dsmem::fence_proxy_async_smem();
    dsmem::cluster_sync();                // the next round may write tmp and read D
    st.round(round);
  }

  st.mark(1);
  // Undo the rotation: source (rank - i) mod N's chunk is at position i.
  T* o = out + rank * N * M;
  __syncthreads();
  for (int i = threadIdx.x; i < N * M; i += blockDim.x) {
    const int pos = i / M, e = i % M;
    o[(((int)rank - pos + N) % N) * M + e] = D[i];
  }
  dsmem::cluster_sync();
}
#endif

}  // namespace alltoall__bruck
