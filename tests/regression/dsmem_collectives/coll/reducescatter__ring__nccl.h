// ReduceScatter on the ring b -> b+1, N-1 steps. In step s block b bulk-copies its partial sum
// of chunk (b-s-1) mod N to the next block's scratch and adds the incoming partial of chunk
// (b-s-2) mod N; after the last step block b has the full sum of chunk b. A cluster barrier
// ends each step, because the scratch is reused.
// Port of h100-dsmem-collectives/collectives/reducescatter__ring__nccl.cu (commit 2b9a97e),
// algorithm from NCCL (start rotated so that block b ends with chunk b).
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace reducescatter__ring {

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  alignas(16) ELEM_T buf[N * M];
  alignas(16) ELEM_T scratch[M];
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

  const uint32_t next = (rank + 1) % N;
  const uint32_t buf_a = dsmem::smem_u32(buf);
  const uint32_t scr_a = dsmem::smem_u32(scratch);
  uint32_t phase = 0;
  for (int s = 0; s < N - 1; ++s) {
    const int send_c = ((int)rank - s - 1 + N) % N;
    const int recv_c = ((int)rank - s - 2 + 2 * N) % N;
    if (threadIdx.x == 0) {
      dsmem::mbar_arrive_expect_tx(bar, bytes);
      dsmem::bulk_push(next, scr_a, buf_a + send_c * bytes, bytes, bar);
    }
    dsmem::mbar_wait_parity(bar, phase);
    phase ^= 1;
    for (int i = threadIdx.x; i < M; i += blockDim.x)
      buf[recv_c * M + i] = op(buf[recv_c * M + i], scratch[i]);
    dsmem::fence_proxy_async_smem();
    dsmem::cluster_sync();
    st.round(s);
  }

  st.mark(1);
  dsmem::store_block<T, M>(out + rank * M, buf + rank * M);
  dsmem::cluster_sync();
}
#endif

}  // namespace reducescatter__ring
