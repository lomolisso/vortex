// AllReduce in two shots of N-1 concurrent bulk copies: block b owns chunk
// b of M/N. In shot 1 block j's slot[b] receives chunk j from b, and b sums
// the slots of chunk b it received; in shot 2 the summed chunk b is copied to
// offset b of every peer. Each shot completes on one barrier armed with the
// aggregate (N-1) chunks; no cluster barrier separates the shots, because a
// block's own shot-1 wait implies every peer's shot-1 read of the chunk it
// overwrites.
// Port of h100-dsmem-collectives/collectives/allreduce__two_shot_push__trtllm.cu
// (commit 2b9a97e); origin: TensorRT-LLM customAllReduce two-shot.
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__two_shot_push {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T buf[M];
  alignas(16) ELEM_T slots[N][M / N];
  alignas(8) uint64_t bars[2];
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
  const Op op{};
  const uint32_t rank = cluster_rank();
  Stamps st{ts + rank * kStamps};

  load_block<T, M>(buf, in + rank * M);
  fence_proxy_async_smem();
  if (threadIdx.x == 0) {
    mbar_init(smem_u32(&sm.bars[0]), 1);
    mbar_init(smem_u32(&sm.bars[1]), 1);
    fence_mbar_init();
    smid_out[rank] = smid();
  }
  cluster_sync();
  st.mark(0);

  const uint32_t bar_rs = smem_u32(&sm.bars[0]), bar_ag = smem_u32(&sm.bars[1]);
  const uint32_t buf_a = smem_u32(buf), slots_a = smem_u32(&sm.slots[0][0]);

  // Shot 1: chunk dst goes to slot[rank] of block dst.
  if (threadIdx.x == 0) {
    mbar_arrive_expect_tx(bar_rs, (N - 1) * bytes);
    for (uint32_t j = 1; j < (uint32_t)N; ++j) {
      const uint32_t dst = (rank + j) % N;
      bulk_push(dst, slots_a + rank * bytes, buf_a + dst * bytes, bytes, bar_rs);
    }
  }
  mbar_wait_parity(bar_rs, 0);
  for (int i = threadIdx.x; i < C; i += (int)blockDim.x) {
    T acc = buf[rank * C + i];
#pragma unroll
    for (int r = 0; r < N; ++r)
      if (r != (int)rank)
        acc = op(acc, sm.slots[r][i]);
    buf[rank * C + i] = acc;
  }
  fence_proxy_async_smem();
  __syncthreads();
  st.round(0);

  // Shot 2: the reduced chunk goes to the same position in every peer.
  if (threadIdx.x == 0) {
    mbar_arrive_expect_tx(bar_ag, (N - 1) * bytes);
    for (uint32_t j = 1; j < (uint32_t)N; ++j) {
      const uint32_t dst = (rank + j) % N;
      bulk_push(dst, buf_a + rank * bytes, buf_a + rank * bytes, bytes, bar_ag);
    }
  }
  mbar_wait_parity(bar_ag, 0);
  st.round(1);

  st.mark(1);
  store_block<T, M>(out + rank * M, buf);
  cluster_sync();
}
#endif

} // namespace allreduce__two_shot_push
