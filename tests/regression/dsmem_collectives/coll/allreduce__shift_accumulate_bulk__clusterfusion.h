// AllReduce (sum) by shift-and-accumulate bulk pushes, ClusterFusion's shipped
// schedule: in round i = 1 .. N-2 block b copies its running sum into the
// scratch of block (b + i) mod N and sums the scratch it got into a second
// buffer. Round 1 brings {b-1} and round 2 {b-2, b-3}, which covers every
// rank only when N = 4. The wait parity is (i ^ 1) & 1.
// Port of h100-dsmem-collectives/collectives/allreduce__shift_accumulate_bulk__clusterfusion.cu
// (commit 2b9a97e); origin: ClusterFusion include/dsm.cuh cluster_reduce.
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__shift_accumulate_bulk {

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
  if (N != 4)
    return "shipped loop bound (i < N-1) only reduces correctly for N == 4";
  if ((M * sizeof(T)) % 16 != 0)
    return "bulk copy needs a multiple of 16 bytes";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "run() must match the build's CLUSTER_N and MSG_ELEMS");
  static_assert(N == 4, "shipped loop bound (i < N-1) only reduces correctly for N == 4");
  static_assert((M * sizeof(T)) % 16 == 0, "bulk copy needs a multiple of 16 bytes");
  constexpr uint32_t bytes = M * sizeof(T);

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* scratch = sm.scratch;
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
  for (int i = 1; i < N - 1; ++i) {
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(bar, bytes);
      const uint32_t dst_cta = (rank + i) % N;
      bulk_push(dst_cta, scr_a, smem_u32(cur), bytes, bar);
    }
    mbar_wait_parity(bar, (i ^ 1) & 1);
    for (int e = threadIdx.x; e < M; e += (int)blockDim.x)
      other[e] = cur[e] + scratch[e];
    fence_proxy_async_smem();
    cluster_sync();
    T* t = cur;
    cur = other;
    other = t;
    st.round(i - 1);
  }

  st.mark(1);
  store_block<T, M>(out + rank * M, cur);
  cluster_sync();
}
#endif

} // namespace allreduce__shift_accumulate_bulk
