// ReduceScatter, recursive halving, for N a power of two: one round per bit, d = N/2 down to 1,
// with partner b^d. A block retains one half of its live range, sends the other half to the
// partner's scratch with a bulk copy, and adds what the partner sent for the retained half.
// After the last round the live range is chunk b. One cluster barrier per round.
// Port of h100-dsmem-collectives/collectives/reducescatter__recursive_halving__mpich.cu
// (commit 2b9a97e), algorithm from MPICH (first phase of Rabenseifner's all-reduce).
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace reducescatter__recursive_halving {

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T buf[N * M];
  alignas(16) ELEM_T scratch[N * M / 2];
  alignas(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  return ((N & (N - 1)) != 0)            ? "needs a power-of-two cluster size"
         : ((M * sizeof(T)) % 16 != 0)   ? "bulk copy needs a multiple of 16 bytes"
                                         : nullptr;
}

constexpr int kOp = 2;
constexpr bool kTypeAgnostic = false;

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid) {
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "shape differs from CollectiveSmem");
  CollectiveSmem& sm = dsmem::dyn_smem<CollectiveSmem>();
  T* buf = sm.buf;
  T* scratch = sm.scratch;
  const uint32_t bar = dsmem::smem_u32(&sm.bar);
  const Op op{};
  const uint32_t rank = dsmem::cluster_rank();
  const dsmem::Stamps st{ts + rank * dsmem::kStamps};

  dsmem::load_block<T, N * M>(buf, in + rank * N * M);
  dsmem::fence_proxy_async_smem();
  if (threadIdx.x == 0) {
    dsmem::mbar_init(bar, 1);
    dsmem::fence_mbar_init();
    smid[rank] = dsmem::smid();
  }
  dsmem::cluster_sync();
  st.mark(0);

  const uint32_t buf_a = dsmem::smem_u32(buf);
  const uint32_t scr_a = dsmem::smem_u32(scratch);
  uint32_t phase = 0;
  int lo = 0, hi = N * M, round = 0;
  for (int d = N / 2; d >= 1; d >>= 1, ++round) {
    const uint32_t partner = rank ^ d;
    const int mid = (lo + hi) / 2;
    const int len = hi - mid;
    const bool upper = (rank & d) != 0;
    const int send_lo = upper ? lo : mid;
    const int keep_lo = upper ? mid : lo;
    const uint32_t bytes = len * sizeof(T);
    if (threadIdx.x == 0) {
      dsmem::mbar_arrive_expect_tx(bar, bytes);
      dsmem::bulk_push(partner, scr_a, buf_a + send_lo * sizeof(T), bytes, bar);
    }
    dsmem::mbar_wait_parity(bar, phase);
    phase ^= 1;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
      buf[keep_lo + i] = op(buf[keep_lo + i], scratch[i]);
    lo = keep_lo;
    hi = keep_lo + len;
    dsmem::fence_proxy_async_smem();
    dsmem::cluster_sync();
    st.round(round);
  }

  st.mark(1);
  dsmem::store_block<T, M>(out + rank * M, buf + rank * M);
  dsmem::cluster_sync();
}
#endif

}  // namespace reducescatter__recursive_halving
