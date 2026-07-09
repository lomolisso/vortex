// dsmem_pingpong — cross-core DSMEM addressing (Task 3b).
//
// A CTA cluster of `num_ranks` single-warp CTAs, one rank per core. Each rank
// publishes a pattern in its own scratchpad, rendezvouses on the cluster
// barrier, then (A) remote-LOADs its peer's pattern over the SM-to-SM network
// and (B) remote-STOREs a second pattern into the peer's scratchpad. The
// second cluster barrier drains the remote stores, so the final local read
// must observe the peer's data — a stale value there is a drain failure.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>

#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t rank = get_cluster_rank();
  const uint32_t K    = get_cluster_size();
  const uint32_t peer = (rank + 1) % K;
  const uint32_t t    = threadIdx.x;
  const uint32_t T    = arg->buf_elems;

  auto smem = reinterpret_cast<volatile uint32_t*>(__local_mem());

  // Publish this rank's phase-A pattern in its own scratchpad.
  smem[t] = (rank + 1) * 1000 + t;

  vortex::cluster_barrier cbar(0);
  cbar.arrive_and_wait(); // publish + full-cluster residency

  // Phase A: thread-issued remote load of the peer's scratchpad.
  auto peerA = reinterpret_cast<const void*>(
      vx_mapa_cluster((const void*)&smem[t], peer));
  uint32_t got = vx_ld_cluster(peerA);
  auto dstA = reinterpret_cast<uint32_t*>(arg->dstA_addr);
  dstA[rank * T + t] = got;

  // Phase B: thread-issued remote store into the peer's scratchpad...
  auto peerB = vx_mapa_cluster((const void*)&smem[T + t], peer);
  vx_st_cluster(peerB, (rank + 1) * 7777 + t);

  // ...drained by the cluster barrier: when it releases, every pre-barrier
  // remote store has committed in the peers' LMEM banks.
  cbar.arrive_and_wait();

  // Local read of what the peer wrote into us.
  auto dstB = reinterpret_cast<uint32_t*>(arg->dstB_addr);
  dstB[rank * T + t] = smem[T + t];
}
