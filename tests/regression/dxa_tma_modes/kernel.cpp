// dxa_tma_modes — TMA copy-mode features (Task 4): element strides,
// 3D K-major, destination swizzle. The kernel issues one DXA tile copy
// (descriptor pre-programmed by the host), waits on the barrier-based
// completion, then dumps the raw SMEM tile image for host-side comparison
// against a reference layout model.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

constexpr uint32_t kDescSlot = 0;

__kernel void kernel_main(kernel_arg_t* arg) {
  auto shmem = reinterpret_cast<uint32_t*>(__local_mem());

  vortex::barrier bar(0);
  if (get_sub_group_id() == 0) {
    bar.expect_tx(1);
    if (arg->ndim == 3) {
      vx_dxa_issue_3d_wg(kDescSlot, bar.id(), shmem, 0, 0, 0);
    } else {
      vx_dxa_issue_2d_wg(kDescSlot, bar.id(), shmem, 0, 0);
    }
  }
  bar.arrive_and_wait();

  // Dump the raw SMEM image — the host knows the expected layout.
  auto dst = reinterpret_cast<uint32_t*>(arg->dst_addr);
  for (uint32_t i = threadIdx.x; i < arg->tile_words; i += arg->block_size) {
    dst[i] = shmem[i];
  }
}
