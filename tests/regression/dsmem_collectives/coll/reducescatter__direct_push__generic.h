// ReduceScatter, direct push: for each j != b, block b bulk-copies chunk j into slot b of block
// j. A receiver arms one mbarrier for the total of its N-1 incoming copies, waits for it, and
// sums the received partials of chunk b with its own. One round.
// Port of h100-dsmem-collectives/collectives/reducescatter__direct_push__generic.cu (commit 2b9a97e),
// a generic algorithm (first shot of the two-shot all-reduce).
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace reducescatter__direct_push {

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T buf[N * M];
  alignas(16) ELEM_T slots[N][M];
  ELEM_T result[M];
  alignas(8) uint64_t bar;
};

template <int N, int M, class T>
constexpr const char* illegal() {
  return ((M * sizeof(T)) % 16 != 0) ? "bulk copy needs a multiple of 16 bytes" : nullptr;
}

constexpr int kOp = 2;
constexpr bool kTypeAgnostic = false;

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid) {
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "shape differs from CollectiveSmem");
  constexpr uint32_t bytes = M * sizeof(T);
  CollectiveSmem& sm = dsmem::dyn_smem<CollectiveSmem>();
  T* buf = sm.buf;
  T (*slots)[M] = sm.slots;
  T* result = sm.result;
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
  const uint32_t slots_a = dsmem::smem_u32(&slots[0][0]);
  if (threadIdx.x == 0) {
    dsmem::mbar_arrive_expect_tx(bar, (N - 1) * bytes);  // all incoming copies at once
    for (uint32_t j = 1; j < N; ++j) {
      const uint32_t dst = (rank + j) % N;
      dsmem::bulk_push(dst, slots_a + rank * bytes, buf_a + dst * bytes, bytes, bar);
    }
  }
  dsmem::mbar_wait_parity(bar, 0);
  for (int i = threadIdx.x; i < M; i += blockDim.x) {
    T acc = buf[rank * M + i];
#pragma unroll
    for (int r = 0; r < N; ++r) {
      if (r != (int)rank)
        acc = op(acc, slots[r][i]);
    }
    result[i] = acc;
  }
  st.round(0);

  st.mark(1);
  dsmem::store_block<T, M>(out + rank * M, result);
  dsmem::cluster_sync();  // keeps my slots alive until every push into them has landed
}
#endif

}  // namespace reducescatter__direct_push
