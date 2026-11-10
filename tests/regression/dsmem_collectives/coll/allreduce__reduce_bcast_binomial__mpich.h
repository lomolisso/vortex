// AllReduce rooted at rank 0: a binomial-tree reduce (d = 1, 2, ..; once bit
// d of its rank is set, a block hands its partial sum to b - d and stops)
// and then the reverse tree as a broadcast (d = N/2 .. 1; a block with the
// result sends it to b + d).
// Only the receiver arms its barrier; the sender's bytes may land first.
// Port of h100-dsmem-collectives/collectives/allreduce__reduce_bcast_binomial__mpich.cu
// (commit 2b9a97e); origin: MPICH short-message MPI_Allreduce (Thakur et al.
// 2005).
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__reduce_bcast_binomial {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T buf[M];
  alignas(16) ELEM_T scratch[M];
  alignas(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  if ((N & (N - 1)) != 0)
    return "binomial tree as written needs a power-of-two cluster size";
  if ((M * sizeof(T)) % 16 != 0)
    return "bulk copy needs a multiple of 16 bytes";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "run() must match the build's CLUSTER_N and MSG_ELEMS");
  static_assert((N & (N - 1)) == 0, "binomial tree as written needs a power-of-two cluster size");
  static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
  constexpr uint32_t bytes = M * sizeof(T);

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* buf = sm.buf;
  T* scratch = sm.scratch;
  const Op op{};
  const uint32_t rank = cluster_rank();
  Stamps st{ts + rank * kStamps};

  load_block<T, M>(buf, in + rank * M);
  fence_proxy_async_smem();
  if (threadIdx.x == 0) {
    mbar_init(smem_u32(&sm.bar), 1);
    fence_mbar_init();
    smid_out[rank] = smid();
  }
  cluster_sync();
  st.mark(0);

  const uint32_t bar = smem_u32(&sm.bar);
  const uint32_t buf_a = smem_u32(buf), scr_a = smem_u32(scratch);
  uint32_t phase = 0;
  int round = 0;
  bool active = true;

  // Reduce phase.
  for (int d = 1; d < N; d <<= 1, ++round) {
    if (active) {
      if (rank & d) {
        if (threadIdx.x == 0)
          bulk_push(rank - d, scr_a, buf_a, bytes, bar);
        active = false;
      } else {
        if (threadIdx.x == 0)
          mbar_arrive_expect_tx(bar, bytes);
        mbar_wait_parity(bar, phase);
        phase ^= 1;
        for (int i = threadIdx.x; i < M; i += (int)blockDim.x)
          buf[i] = op(buf[i], scratch[i]);
        fence_proxy_async_smem();
      }
    }
    cluster_sync();
    st.round(round);
  }

  // Broadcast from rank 0 into the receivers' buf.
  for (int d = N / 2; d >= 1; d >>= 1, ++round) {
    if ((rank % (2 * d)) == 0) {
      if (threadIdx.x == 0)
        bulk_push(rank + d, buf_a, buf_a, bytes, bar);
    } else if ((rank % (2 * d)) == (uint32_t)d) {
      if (threadIdx.x == 0)
        mbar_arrive_expect_tx(bar, bytes);
      mbar_wait_parity(bar, phase);
      phase ^= 1;
      fence_proxy_async_smem();
    }
    cluster_sync();
    st.round(round);
  }

  st.mark(1);
  store_block<T, M>(out + rank * M, buf);
  cluster_sync();
}
#endif

} // namespace allreduce__reduce_bcast_binomial
