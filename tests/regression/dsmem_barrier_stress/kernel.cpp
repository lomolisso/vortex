// dsmem_barrier_stress — F2 cluster-barrier ordering + suspend, no deadlock.
//
// Each phase: rank r remote-writes a phase-tagged value into peer (r+1)%K, a
// cluster barrier orders that write before r reads what writer (r-1)%K wrote to
// it, and a second barrier orders the read before the next phase overwrites.
// Correct output requires every barrier to (a) actually suspend all warps of
// all ranks and (b) release only after every arrival — an early release
// corrupts the phase chain; a lost cross-SM release hangs (watchdog aborts).

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>

#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t K      = get_cluster_size();
  const uint32_t cid    = get_cluster_id();
  const uint32_t rank   = get_cluster_rank();
  const uint32_t t      = threadIdx.x;
  const uint32_t T      = arg->buf_elems;
  const uint32_t N      = arg->phases;
  const uint32_t peer   = (rank + 1) % K;

  auto smem = reinterpret_cast<volatile uint32_t*>(__local_mem());
  smem[t] = 0;

  vortex::cluster_barrier cbar(0);
  cbar.arrive_and_wait();   // residency

  auto dst = reinterpret_cast<uint32_t*>(arg->dst_addr);
  for (uint32_t p = 0; p < N; ++p) {
    // produce: write this phase's value into the peer.
    auto peerp = vx_mapa_cluster((const void*)&smem[t], peer);
    vx_st_cluster(peerp, DSMEM_BS_PATTERN(p, rank, t));

    cbar.arrive_and_wait();   // write ordered before the read

    // consume: read what `writer` put into us this phase.
    uint32_t got = smem[t];
    dst[((cid * K + rank) * N + p) * T + t] = got;

    cbar.arrive_and_wait();   // read ordered before the next phase overwrites
  }
}
