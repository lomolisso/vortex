// dxa_multicast — cross-core TMA multicast (Task 3c).
//
// A CTA cluster of `num_recv` single-warp CTAs, one rank per core. Every rank
// arms its own completion barrier (expect_tx), the cluster barrier
// rendezvouses across cores (so all receivers are resident and armed), then
// rank 0's per-SM DXA issues ONE multicast: a single GMEM read of the tile,
// fanned out into every rank's scratchpad over the SM-to-SM network. Each
// receiver's barrier fires when ITS slice lands (per-destination completion).

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

constexpr uint32_t kDescSrc = 0;

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t rank      = get_cluster_rank();
  const uint32_t tile_rows = arg->tile_rows;
  const uint32_t tile_cols = arg->tile_cols;
  const uint32_t num_recv  = arg->num_recv;

  auto shmem = reinterpret_cast<TYPE*>(__local_mem());

  vortex::barrier         local_bar(0);  // my slice's completion (expect_tx)
  vortex::cluster_barrier rdv(1);        // cross-core pre-issue rendezvous

  local_bar.expect_tx(1);
  rdv.arrive_and_wait();  // every rank resident + armed before the issue

  if (rank == 0 && get_sub_group_id() == 0) {
    uint32_t mask = (1u << num_recv) - 1u;
    vx_dxa_issue_2d_multicast_wg(kDescSrc, local_bar.id(), shmem,
                                 /*coord0=*/0, /*coord1=*/0, mask);
  }

  // Per-destination completion: released when THIS rank's slice lands.
  local_bar.arrive_and_wait();

  // Store the received tile to this rank's dst region for host verification.
  auto dst = reinterpret_cast<TYPE*>(arg->dst_addr);
  const uint32_t tile_elems = tile_rows * tile_cols;
  const uint32_t l_col = threadIdx.x;
  for (uint32_t r = 0; r < tile_rows; ++r) {
    if (l_col < tile_cols) {
      dst[rank * tile_elems + r * tile_cols + l_col] =
          shmem[r * tile_cols + l_col];
    }
  }

  (void)arg->src_addr;
  (void)arg->src_row_stride;
}
