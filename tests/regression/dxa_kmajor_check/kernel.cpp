// Each CTA does TWO DXA issues:
//   kDescA (row-major): A_smem ← A[cta_row*tileM .. , 0..tileK-1]
//   kDescB (K-major):   B_smem ← B[0..tileK-1, cta_col*tileN ..]
// Then byte-copies both SMEM blocks to dst for host verification.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid     = threadIdx.x;
  const uint32_t cta_id  = blockIdx.x;   // 1D grid — one CTA per (M-tile, N-tile) pair
  const uint32_t grid_n  = arg->N / arg->tileN;
  const uint32_t cta_row = cta_id / grid_n;
  const uint32_t cta_col = cta_id % grid_n;

  auto A_smem = reinterpret_cast<uint8_t*>(__local_mem());
  auto B_smem = A_smem + arg->a_bytes;

  vortex::barrier bar(0);
  if (get_sub_group_id() == 0) {
    bar.expect_tx(2);
    // A: 2D row-major fetch (coord0 = K-axis start, coord1 = M-axis start).
    vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem,
                       /*coord0=*/0, /*coord1=*/cta_row * arg->tileM);
    // B: 2D K-major scatter (coord0 = N-axis start, coord1 = K-axis start).
    vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem,
                       /*coord0=*/cta_col * arg->tileN, /*coord1=*/0);
  }
  bar.arrive_and_wait();

  // Copy both SMEM regions to dst[cta_id * cta_bytes ..]
  auto pDst = reinterpret_cast<uint8_t*>(arg->dst_addr) + cta_id * arg->cta_bytes;
  const uint32_t total = arg->cta_bytes;
  for (uint32_t i = tid; i < total; i += blockDim.x) {
    pDst[i] = A_smem[i];  // A_smem is followed by B_smem in memory
  }
  __syncthreads();
}

#ifdef VX_CFG_EXT_MBAR_ENABLE
#include <vx_mbar.h>

// -m: the same two copies, completed on one mbarrier that expects both boxes.
__kernel void kmajor_mbar(mbar_arg_t* __UNIFORM__ marg) {
  const kernel_arg_t* arg = &marg->base;
  const uint32_t tid     = threadIdx.x;
  const uint32_t cta_id  = blockIdx.x;
  const uint32_t grid_n  = arg->N / arg->tileN;
  const uint32_t cta_row = cta_id / grid_n;
  const uint32_t cta_col = cta_id % grid_n;

  auto A_smem = reinterpret_cast<uint8_t*>(__local_mem());
  auto B_smem = A_smem + arg->a_bytes;
  auto mbar   = reinterpret_cast<uint64_t*>(A_smem + ((arg->cta_bytes + 7) & ~7u));

  if (tid == 0) {
    vx_mbar_init(mbar, 1);
  }
  __syncthreads();
  if (get_sub_group_id() == 0) {
    if (tid == 0) {
      vx_mbar_arrive_tx(mbar, arg->a_bytes + arg->b_bytes);
    }
    vx_dxa_issue_2d_mbar(kDescA, mbar, A_smem, 0, cta_row * arg->tileM);
    vx_dxa_issue_2d_mbar(kDescB, mbar, B_smem, cta_col * arg->tileN, 0);
  }
  uint32_t err = 0;
  for (uint32_t n = 1; !vx_mbar_try_wait_parity(mbar, 0); ++n) {
    if (n >= WAIT_LIMIT) {
      err = 1;
      break;
    }
  }

  auto pDst = reinterpret_cast<uint8_t*>(arg->dst_addr) + cta_id * arg->cta_bytes;
  for (uint32_t i = tid; i < arg->cta_bytes; i += blockDim.x) {
    pDst[i] = A_smem[i];
  }
  reinterpret_cast<uint32_t*>(marg->err_addr)[cta_id * blockDim.x + tid] = err;
}
#endif
