// AllGather in two binomial trees: rank 0 first collects every vector, then
// sends the assembled buffer back down. Gather, d = 1 .. N/2: once bit d of
// its rank is set, a block copies the d slots it has gathered into slots
// d .. 2d-1 of block b - d and leaves the gather. Broadcast, d = N/2 .. 1:
// each block that has the result copies the N slots to block b + d.
// N must be a power of two.
// Port of h100-dsmem-collectives/collectives/
// allgather__gather_bcast_binomial__mpich.cu (commit 2b9a97e); origin: MPICH
// composition of a gather and a broadcast (Thakur et al. 2005).
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "../dsmem_primitives.h"

namespace allgather__gather_bcast_binomial {

constexpr int kOp = 1;
constexpr bool kTypeAgnostic = true;

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS;
  __align__(16) ELEM_T D[N * M];
  __align__(8) uint64_t bar;
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
  static_assert(illegal<N, M, T>() == nullptr, "illegal build");
  constexpr uint32_t blk = M * sizeof(T);

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* D = sm.D;
  const uint32_t me = cluster_rank();
  const Stamps st{ts + me * kStamps};

  load_block<T, M>(D, in + me * M); // gather: slot i is rank b + i's
  fence_proxy_async_smem();
  const uint32_t bar = smem_u32(&sm.bar);
  if (threadIdx.x == 0) {
    mbar_init(bar, 1);
    fence_mbar_init();
    smid_out[me] = smid();
  }
  cluster_sync();
  st.mark(0);

  const uint32_t D_a = smem_u32(D);
  uint32_t parity = 0;
  int round = 0;
  bool active = true;
  for (uint32_t d = 1; d < (uint32_t)N; d <<= 1, ++round) { // gather
    if (active) {
      if (me & d) {
        if (threadIdx.x == 0)
          bulk_push(me - d, D_a + d * blk, D_a, d * blk, bar);
        active = false;
      } else {
        if (threadIdx.x == 0)
          mbar_arrive_expect_tx(bar, d * blk);
        mbar_wait_parity(bar, parity);
        parity ^= 1;
      }
    }
    cluster_sync();
    st.round(round);
  }
  for (uint32_t d = N / 2; d >= 1; d >>= 1, ++round) { // broadcast
    if (me % (2 * d) == 0) {
      if (threadIdx.x == 0)
        bulk_push(me + d, D_a, D_a, N * blk, bar);
    } else if (me % (2 * d) == d) {
      if (threadIdx.x == 0)
        mbar_arrive_expect_tx(bar, N * blk);
      mbar_wait_parity(bar, parity);
      parity ^= 1;
    }
    cluster_sync();
    st.round(round);
  }

  st.mark(1);
  store_block<T, N * M>(out + me * N * M, D); // in rank order on every block
  cluster_sync();
}
#endif

} // namespace allgather__gather_bcast_binomial
