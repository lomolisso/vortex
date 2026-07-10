// dsmem_pingpong — cross-core DSMEM addressing (Task 3).
//
// A CTA cluster of `num_ranks` CTAs (one or more warps each), one rank per
// core. Each rank publishes a pattern in its own scratchpad, rendezvouses on
// the cluster barrier, then:
//   A: remote-LOADs its peer's pattern (explicit ld.cluster intrinsic);
//   B: remote-STOREs a second pattern into the peer — even threads via the
//      explicit st.cluster intrinsic, odd threads via a PLAIN sw through the
//      mapa'd pointer (the aperture-decode path). The second cluster barrier
//      drains both, so the final local read must observe the peer's data —
//      a stale value there is a drain failure;
//   C: plain lw through a mapa'd pointer (remote aperture decode) of the
//      peer's still-intact phase-A slot, plus the region-0 self-map check:
//      mapa to one's OWN rank must return the pointer unchanged and plain
//      accesses through it stay local.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>

#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t rank = get_cluster_rank();
  const uint32_t K    = get_cluster_size();
  const uint32_t peer = (rank + 1) % K;
  const uint32_t t    = threadIdx.x;   // spans all warps of the CTA
  const uint32_t T    = arg->buf_elems;

  auto smem = reinterpret_cast<volatile uint32_t*>(__local_mem());

  // Publish this rank's phase-A pattern in its own scratchpad.
  smem[t] = (rank + 1) * 1000 + t;

  vortex::cluster_barrier cbar(0);
  cbar.arrive_and_wait(); // publish + full-cluster residency

  // Phase A: thread-issued remote load of the peer's scratchpad (ld.cluster).
  auto peerA = reinterpret_cast<const void*>(
      vx_mapa_cluster((const void*)&smem[t], peer));
  uint32_t got = vx_ld_cluster(peerA);
  auto dstA = reinterpret_cast<uint32_t*>(arg->dstA_addr);
  dstA[rank * T + t] = got;

  // Phase B: thread-issued remote store into the peer's scratchpad. Odd
  // threads use a plain sw through the mapa'd pointer — the LSU's
  // address-based aperture decode must route (and drain-count) it exactly
  // like the explicit st.cluster of the even threads.
  auto peerB = vx_mapa_cluster((const void*)&smem[T + t], peer);
  uint32_t valB = (rank + 1) * 7777 + t;
  if (t & 1) {
    *reinterpret_cast<volatile uint32_t*>(peerB) = valB;
  } else {
    vx_st_cluster(peerB, valB);
  }

  // ...drained by the cluster barrier: when it releases, every pre-barrier
  // remote store has committed in the peers' LMEM banks.
  cbar.arrive_and_wait();

  // Local read of what the peer wrote into us.
  auto dstB = reinterpret_cast<uint32_t*>(arg->dstB_addr);
  dstB[rank * T + t] = smem[T + t];

  // Phase C: plain lw through a mapa'd pointer — the remote aperture-decode
  // path — of the peer's still-intact phase-A slot.
  auto peerC = reinterpret_cast<volatile const uint32_t*>(
      vx_mapa_cluster((const void*)&smem[t], peer));
  uint32_t gotC = *peerC;

  // Everyone is done reading phase-A slots before the self-map overwrite.
  cbar.arrive_and_wait();

  // Region-0 self-map: mapping to one's own rank returns the pointer
  // unchanged (relative-region encoding, region 0 = local window), and plain
  // accesses through it behave as local scratchpad accesses.
  auto selfp = reinterpret_cast<volatile uint32_t*>(
      vx_mapa_cluster((const void*)&smem[t], rank));
  bool self_ok = (selfp == &smem[t]);
  *selfp = (rank + 1) * 555 + t;
  self_ok = self_ok && (smem[t] == (rank + 1) * 555 + t);

  auto dstC = reinterpret_cast<uint32_t*>(arg->dstC_addr);
  dstC[rank * T + t] = gotC + (self_ok ? 0u : 0xDEAD0000u);
}
