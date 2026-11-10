// AllReduce by pull reduce-scatter and push all-gather: block b owns M/N
// elements. A group of N adjacent lanes handles one of them: lane r reads
// block r's copy, xor shuffles within the group form the sum, and each lane
// writes the sum to its block's copy of that element.
// Port of h100-dsmem-collectives/collectives/allreduce__pull_reduce_scatter_push_allgather__hpcops.cu
// (commit 2b9a97e); origin: hpc-ops topk_filtered cluster_allreduce_histogram.
// Departures: none (the width-N __shfl_xor_sync is the segmented butterfly
// shuffle of 32-thread warps).
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__pull_reduce_scatter_push_allgather {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  ELEM_T buf[M];
};

template <int N, int M, class T>
constexpr const char* illegal() {
  if (32 % N != 0)
    return "lane groups of N lanes must tile a warp";
  if (M % N != 0)
    return "every rank owns M/N elements";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "run() must match the build's CLUSTER_N and MSG_ELEMS");
  static_assert(32 % N == 0, "lane groups of N lanes must tile a warp");
  static_assert(M % N == 0, "every rank owns M/N elements");
  constexpr int C = M / N;

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* buf = sm.buf;
  const Op op{};
  const uint32_t rank = cluster_rank();
  Stamps st{ts + rank * kStamps};

  load_block<T, M>(buf, in + rank * M);
  if (threadIdx.x == 0)
    smid_out[rank] = smid();
  cluster_sync();
  st.mark(0);

  cluster_sync();
  // Equal trip counts on every thread keep the shuffles' lanes all present.
  for (int base = 0; base < C * N; base += (int)blockDim.x) {
    const int linear = base + (int)threadIdx.x;
    const bool active = linear < C * N;
    const int e = rank * C + (active ? linear : 0) / N;
    const int peer = linear % N;
    T* addr = nullptr;
    T v = Op::template identity<T>();
    if (active) {
      addr = map_ptr(&buf[e], peer);
      v = *addr;
    }
#pragma unroll
    for (int off = N / 2; off > 0; off >>= 1)
      v = op(v, __shfl_xor_sync(0xffffffffu, v, off, N));
    if (active)
      *addr = v;
  }
  st.round(0);
  cluster_sync();

  st.mark(1);
  store_block<T, M>(out + rank * M, buf);
  cluster_sync();
}
#endif

} // namespace allreduce__pull_reduce_scatter_push_allgather
