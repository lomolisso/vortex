// AllReduce on a one-way ring b -> (b + 1) mod N over N chunks of M/N
// elements: N-1 reduce-scatter steps (chunk (b - s) mod N goes to the next
// block's scratch; the arriving chunk (b - s - 1) mod N is added), then N-1
// all-gather steps that pass each finished chunk on to the next block.
// Port of h100-dsmem-collectives/collectives/allreduce__ring__nccl.cu
// (commit 2b9a97e); origin: NCCL ring (Patarasuk & Yuan 2009).
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__ring {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T buf[M];
  alignas(16) ELEM_T scratch[M / N];
  alignas(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  if (M % N != 0)
    return "M must split into N chunks";
  if (((M / N) * sizeof(T)) % 16 != 0)
    return "chunk must be a multiple of 16 bytes";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "run() must match the build's CLUSTER_N and MSG_ELEMS");
  static_assert(M % N == 0, "M must split into N chunks");
  constexpr int C = M / N;
  static_assert((C * sizeof(T)) % 16 == 0, "chunk must be a multiple of 16 bytes");
  constexpr uint32_t bytes = C * sizeof(T);

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
  const uint32_t next = (rank + 1) % N;
  const uint32_t buf_a = smem_u32(buf), scr_a = smem_u32(scratch);
  uint32_t phase = 0;
  int round = 0;

  // Reduce-scatter.
  for (int s = 0; s < N - 1; ++s, ++round) {
    const int send_c = (rank - s + N) % N, recv_c = (rank - s - 1 + N) % N;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(bar, bytes);
      bulk_push(next, scr_a, buf_a + send_c * bytes, bytes, bar);
    }
    mbar_wait_parity(bar, phase);
    phase ^= 1;
    for (int i = threadIdx.x; i < C; i += (int)blockDim.x)
      buf[recv_c * C + i] = op(buf[recv_c * C + i], scratch[i]);
    fence_proxy_async_smem();
    cluster_sync();
    st.round(round);
  }

  // All-gather: this block holds the finished chunk (rank + 1) mod N.
  for (int s = 0; s < N - 1; ++s, ++round) {
    const int send_c = (rank + 1 - s + N) % N;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(bar, bytes);
      bulk_push(next, buf_a + send_c * bytes, buf_a + send_c * bytes, bytes, bar);
    }
    mbar_wait_parity(bar, phase);
    phase ^= 1;
    cluster_sync();
    st.round(round);
  }

  st.mark(1);
  store_block<T, M>(out + rank * M, buf);
  cluster_sync();
}
#endif

} // namespace allreduce__ring
