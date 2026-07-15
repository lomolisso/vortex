// dsmem_storeflood — F3 outstanding-remote-store credit backpressure + drain.
//
// Each rank floods iters*T remote stores into its peer's scratchpad. When that
// exceeds VX_CFG_DSMEM_STORE_CREDITS the LSU stalls issue until acks free
// credit; the cluster barrier then drains every outstanding store before the
// peer reads its scratchpad back. A credit-accounting bug either drops a store
// (wrong readback) or never frees credit (deadlock -> scheduler watchdog).

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>

#include "common.h"

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t K      = get_cluster_size();
  const uint32_t rank   = get_cluster_rank();
  const uint32_t t      = threadIdx.x;
  const uint32_t T      = arg->buf_elems;
  const uint32_t L      = arg->iters;
  const uint32_t peer   = (rank + 1) % K;
  const uint32_t writer = (rank + K - 1) % K;

  auto smem = reinterpret_cast<volatile uint32_t*>(__local_mem());
  for (uint32_t i = 0; i < L; ++i)   // converged zero-init (no divergence)
    smem[i * T + t] = 0;

  vortex::cluster_barrier cbar(0);
  cbar.arrive_and_wait();

  // FLOOD: iters*T fire-and-forget remote stores into the peer's scratchpad.
  for (uint32_t i = 0; i < L; ++i) {
    auto p = vx_mapa_cluster((const void*)&smem[i * T + t], peer);
    vx_st_cluster(p, DSMEM_SF_PATTERN(rank, i, t));
  }

  cbar.arrive_and_wait();   // drains all outstanding remote stores

  // Read back what `writer` stored into us.
  auto dst = reinterpret_cast<uint32_t*>(arg->dst_addr);
  (void)writer;
  for (uint32_t i = 0; i < L; ++i)
    dst[(rank * L + i) * T + t] = smem[i * T + t];
}
