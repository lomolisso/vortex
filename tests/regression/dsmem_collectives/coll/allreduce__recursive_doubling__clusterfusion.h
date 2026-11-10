// AllReduce by recursive doubling with a ring shift: in round k (stride
// s = 2^k) block b pushes its current buffer into the scratch of block
// (b + s) mod N; what block (b - s) mod N sent it is summed into a second
// buffer, because the current one may still be the source of its own copy.
// Port of h100-dsmem-collectives/collectives/allreduce__recursive_doubling__clusterfusion.cu
// (commit 2b9a97e); origin: ClusterFusion (NeurIPS'25) Algorithm 1.
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__recursive_doubling {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T buf[M];
  alignas(16) ELEM_T alt[M];
  alignas(16) ELEM_T scratch[M];
  alignas(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  if ((N & (N - 1)) != 0)
    return "recursive doubling needs a power-of-two cluster size";
  if ((M * sizeof(T)) % 16 != 0)
    return "bulk copy needs a multiple of 16 bytes";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "run() must match the build's CLUSTER_N and MSG_ELEMS");
  static_assert((N & (N - 1)) == 0, "recursive doubling needs a power-of-two cluster size");
  static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
  constexpr uint32_t bytes = M * sizeof(T);

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* scratch = sm.scratch;
  const Op op{};
  const uint32_t rank = cluster_rank();
  Stamps st{ts + rank * kStamps};

  load_block<T, M>(sm.buf, in + rank * M);
  fence_proxy_async_smem();
  if (threadIdx.x == 0) {
    mbar_init(smem_u32(&sm.bar), 1);
    fence_mbar_init();
    smid_out[rank] = smid();
  }
  cluster_sync();
  st.mark(0);

  const uint32_t bar = smem_u32(&sm.bar);
  const uint32_t scr_a = smem_u32(scratch);
  T* cur = sm.buf;
  T* other = sm.alt;
  uint32_t phase = 0;
  int round = 0;
  for (int s = 1; s < N; s <<= 1, ++round) {
    const uint32_t send_to = (rank + s) % N;
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(bar, bytes);
      bulk_push(send_to, scr_a, smem_u32(cur), bytes, bar);
    }
    mbar_wait_parity(bar, phase);
    phase ^= 1;
    for (int i = threadIdx.x; i < M; i += (int)blockDim.x)
      other[i] = op(cur[i], scratch[i]);
    fence_proxy_async_smem();
    cluster_sync();
    T* t = cur;
    cur = other;
    other = t;
    st.round(round);
  }

  st.mark(1);
  store_block<T, M>(out + rank * M, cur);
  cluster_sync();
}
#endif

} // namespace allreduce__recursive_doubling
