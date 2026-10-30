#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

#ifdef VX_CFG_EXT_DXA_ENABLE
#include <vx_dxa.h>
#include <vx_barrier.h>

// DXA descriptor slot (programmed by host).
constexpr uint32_t kDescSrc = 0;
#endif

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t ndim = arg->ndim;
  const uint32_t num_elems = ({
    uint32_t n = 1;
    for (uint32_t d = 0; d < ndim; ++d) n *= arg->tiles[d];
    n;
  });

  // Decompose flat blockIdx.x into per-dimension block coords.
  uint32_t coords[DXA_MAX_DIMS] = {};
  {
    uint32_t rem = blockIdx.x;
    for (uint32_t d = 0; d < ndim; ++d) {
      uint32_t grid_d = arg->grids[d];
      coords[d] = (rem % grid_d) * arg->tiles[d]; // element coord = block_coord * tile_size
      rem /= grid_d;
    }
  }

  // Allocate shared memory for one tile.
  auto shmem = reinterpret_cast<TYPE*>(__local_mem());

  // This thread's tile element: per-dim local offsets, then its global
  // row-major index.
  const uint32_t tid = threadIdx.x;
  uint32_t gidx = 0;
  {
    uint32_t rem = tid;
    uint32_t stride = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
      gidx += (coords[d] + rem % arg->tiles[d]) * stride;
      rem /= arg->tiles[d];
      stride *= arg->sizes[d];
    }
  }

#ifdef VX_CFG_EXT_DXA_ENABLE
  // ── DXA path: issue N-D tile copy, barrier wait ──
  vortex::barrier bar(0);
  const bool is_dxa_warp = (get_sub_group_id() == 0);
  // -e: a single elected thread issues, like a CUDA kernel issuing TMA.
  const bool is_issuer = is_dxa_warp && (arg->elect == 0 || (tid % VX_CFG_NUM_THREADS) == 0);
  if (is_issuer) {
    bar.expect_tx(1);  // Pre-register 1 pending DXA transaction
    switch (ndim) {
    case 1:
      vx_dxa_issue_1d_wg(kDescSrc, bar.id(), shmem, coords[0]);
      break;
    case 2:
      vx_dxa_issue_2d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1]);
      break;
    case 3:
      vx_dxa_issue_3d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1], coords[2]);
      break;
    case 4:
      vx_dxa_issue_4d_wg(kDescSrc, bar.id(), shmem,
        coords[0], coords[1], coords[2], coords[3]);
      break;
    case 5:
      vx_dxa_issue_5d_wg(kDescSrc, bar.id(), shmem,
        coords[0], coords[1], coords[2], coords[3], coords[4]);
      break;
    }
  }
  bar.arrive_and_wait();

#else
  // ── LSU path: each thread loads one element, syncthreads ──
  auto src = reinterpret_cast<const TYPE*>(arg->src_addr);
  if (tid < num_elems) {
    shmem[tid] = src[gidx];
  }
  __syncthreads();
#endif

  auto dst = reinterpret_cast<TYPE*>(arg->dst_addr);
  if (tid < num_elems) {
    dst[gidx] = shmem[tid];
  }
}

#ifdef VX_CFG_EXT_DXA_ENABLE
#ifdef VX_CFG_EXT_MBAR_ENABLE
#include <vx_mbar.h>
#endif

// -m and -c: the copy completes on an mbarrier (-m) or on a barrier event,
// and canary words follow the tile (-c). Every thread reports ERR_* bits.
__kernel void copy_ext(ext_arg_t* xarg) {
  const kernel_arg_t* arg = &xarg->base;
  const uint32_t ndim = arg->ndim;
  uint32_t num_elems = 1;
  for (uint32_t d = 0; d < ndim; ++d) num_elems *= arg->tiles[d];

  uint32_t coords[DXA_MAX_DIMS] = {};
  {
    uint32_t rem = blockIdx.x;
    for (uint32_t d = 0; d < ndim; ++d) {
      uint32_t grid_d = arg->grids[d];
      coords[d] = (rem % grid_d) * arg->tiles[d];
      rem /= grid_d;
    }
  }

  const uint32_t tid = threadIdx.x;
  uint32_t gidx = 0;
  {
    uint32_t rem = tid;
    uint32_t stride = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
      gidx += (coords[d] + rem % arg->tiles[d]) * stride;
      rem /= arg->tiles[d];
      stride *= arg->sizes[d];
    }
  }

  auto lm = reinterpret_cast<uint8_t*>(__local_mem());
  auto shmem = reinterpret_cast<TYPE*>(lm + TILE_OFF);
  auto canary = reinterpret_cast<volatile uint32_t*>(shmem + num_elems);
  const uint32_t nc = xarg->canary_words;
  for (uint32_t i = tid; i < nc; i += blockDim.x) {
    canary[i] = canary_value(blockIdx.x, i);
  }
#ifdef VX_CFG_EXT_MBAR_ENABLE
  auto mbar = reinterpret_cast<uint64_t*>(lm);
  if (xarg->mbar && tid == 0) {
    vx_mbar_init(mbar, 1);
  }
#endif
  __syncthreads();

  const bool is_dxa_warp = (get_sub_group_id() == 0);
  const bool is_issuer = is_dxa_warp && (arg->elect == 0 || (tid % VX_CFG_NUM_THREADS) == 0);
  uint32_t err = 0;
#ifdef VX_CFG_EXT_MBAR_ENABLE
  if (xarg->mbar) {
    if (tid == 0) {
      vx_mbar_arrive_tx(mbar, xarg->tx_bytes);
    }
    if (is_issuer) {
      switch (ndim) {
      case 1:
        vx_dxa_issue_1d_mbar(kDescSrc, mbar, shmem, coords[0]);
        break;
      case 2:
        vx_dxa_issue_2d_mbar(kDescSrc, mbar, shmem, coords[0], coords[1]);
        break;
      case 3:
        vx_dxa_issue_3d_mbar(kDescSrc, mbar, shmem, coords[0], coords[1], coords[2]);
        break;
      case 4:
        vx_dxa_issue_4d_mbar(kDescSrc, mbar, shmem, coords[0], coords[1], coords[2], coords[3]);
        break;
      case 5:
        vx_dxa_issue_5d_mbar(kDescSrc, mbar, shmem,
          coords[0], coords[1], coords[2], coords[3], coords[4]);
        break;
      }
    }
    for (uint32_t n = 1; !vx_mbar_try_wait_parity(mbar, 0); ++n) {
      if (n >= WAIT_LIMIT) {
        err |= ERR_WAIT;
        break;
      }
    }
  } else
#endif
  {
    vortex::barrier bar(1);
    if (is_issuer) {
      bar.expect_tx(1);
      switch (ndim) {
      case 1:
        vx_dxa_issue_1d_wg(kDescSrc, bar.id(), shmem, coords[0]);
        break;
      case 2:
        vx_dxa_issue_2d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1]);
        break;
      case 3:
        vx_dxa_issue_3d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1], coords[2]);
        break;
      case 4:
        vx_dxa_issue_4d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1], coords[2], coords[3]);
        break;
      case 5:
        vx_dxa_issue_5d_wg(kDescSrc, bar.id(), shmem,
          coords[0], coords[1], coords[2], coords[3], coords[4]);
        break;
      }
    }
    bar.arrive_and_wait();
  }

  for (uint32_t i = tid; i < nc; i += blockDim.x) {
    if (canary[i] != canary_value(blockIdx.x, i)) {
      err |= ERR_CANARY;
    }
  }

  auto dst = reinterpret_cast<TYPE*>(arg->dst_addr);
  if (tid < num_elems) {
    dst[gidx] = shmem[tid];
  }
  reinterpret_cast<uint32_t*>(xarg->err_addr)[blockIdx.x * blockDim.x + tid] = err;
}
#endif
