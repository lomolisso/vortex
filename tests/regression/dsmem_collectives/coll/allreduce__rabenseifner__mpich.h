// AllReduce by Rabenseifner's algorithm: a recursive-halving reduce-scatter
// (d = N/2 .. 1, partner rank ^ d; of the two, the smaller rank retains the
// bottom half of the range, the other the top half, and each adds the
// partner's copy of what it retains), then a recursive-doubling all-gather
// that grows the ranges back (d = 1 .. N/2).
// Port of h100-dsmem-collectives/collectives/allreduce__rabenseifner__mpich.cu
// (commit 2b9a97e); origin: MPICH (Rabenseifner 2004; Thakur, Rabenseifner,
// Gropp 2005).
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__rabenseifner {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T buf[M];
  alignas(16) ELEM_T scratch[M / 2];
  alignas(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  if ((N & (N - 1)) != 0)
    return "needs a power-of-two cluster size";
  if (M % N != 0 || ((M / N) * sizeof(T)) % 16 != 0)
    return "final chunk must be a multiple of 16 bytes";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "run() must match the build's CLUSTER_N and MSG_ELEMS");
  static_assert((N & (N - 1)) == 0, "needs a power-of-two cluster size");
  static_assert(M % N == 0 && ((M / N) * sizeof(T)) % 16 == 0, "final chunk must be a multiple of 16 bytes");

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
  int lo = 0, hi = M, round = 0;

  // Recursive halving: the half this block drops goes to the partner's
  // scratch; the partner's copy of the kept half is added in.
  for (int d = N / 2; d >= 1; d >>= 1, ++round) {
    const uint32_t partner = rank ^ d;
    const int mid = (lo + hi) / 2;
    const bool upper = (rank & d) != 0;
    const int send_lo = upper ? lo : mid, send_hi = upper ? mid : hi;
    const int keep_lo = upper ? mid : lo;
    const uint32_t bytes = (send_hi - send_lo) * sizeof(T);
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(bar, bytes);
      bulk_push(partner, scr_a, buf_a + send_lo * sizeof(T), bytes, bar);
    }
    mbar_wait_parity(bar, phase);
    phase ^= 1;
    const int len = hi - mid;
    for (int i = threadIdx.x; i < len; i += (int)blockDim.x)
      buf[keep_lo + i] = op(buf[keep_lo + i], scratch[i]);
    lo = keep_lo;
    hi = keep_lo + len;
    fence_proxy_async_smem();
    cluster_sync();
    st.round(round);
  }

  // Recursive doubling: the live range goes to the same offset in the
  // partner, whose range lands next to it here.
  for (int d = 1; d < N; d <<= 1, ++round) {
    const uint32_t partner = rank ^ d;
    const int len = hi - lo;
    const uint32_t bytes = len * sizeof(T);
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(bar, bytes);
      bulk_push(partner, buf_a + lo * sizeof(T), buf_a + lo * sizeof(T), bytes, bar);
    }
    mbar_wait_parity(bar, phase);
    phase ^= 1;
    if (rank & d)
      lo -= len;
    else
      hi += len;
    cluster_sync();
    st.round(round);
  }

  st.mark(1);
  store_block<T, M>(out + rank * M, buf);
  cluster_sync();
}
#endif

} // namespace allreduce__rabenseifner
