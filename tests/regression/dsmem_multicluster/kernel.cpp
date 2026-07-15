// dsmem_multicluster — F1 residency-replica keying + cross-cluster isolation.
//
// Several clusters co-reside; gang placement puts one rank per core, so a core
// that hosts rank r of cluster A and rank r' of cluster B keeps BOTH in its
// residency replica. Each CTA remote-loads a peer within its own cluster; the
// replica must resolve by (cluster_id, rank) so cluster A never reads B's data.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>

#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t K    = get_cluster_size();
  const uint32_t cid  = get_cluster_id();
  const uint32_t rank = get_cluster_rank();
  const uint32_t t    = threadIdx.x;
  const uint32_t T    = arg->buf_elems;
  const uint32_t peer = (rank + 1) % K;

  auto smem = reinterpret_cast<volatile uint32_t*>(__local_mem());
  smem[t] = DSMEM_MC_PATTERN(cid, rank, t);

  vortex::cluster_barrier cbar(0);
  cbar.arrive_and_wait();   // publish + full-cluster residency

  // Remote-load this CTA's in-cluster peer through the per-SM replica.
  auto peerp = reinterpret_cast<const void*>(
      vx_mapa_cluster((const void*)&smem[t], peer));
  uint32_t got = vx_ld_cluster(peerp);

  auto dst = reinterpret_cast<uint32_t*>(arg->dst_addr);
  dst[(cid * K + rank) * T + t] = got;
}
