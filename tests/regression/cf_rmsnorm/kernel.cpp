// RMSNorm of one fp16 row per cluster of two blocks: each block sums the
// squares of its half of the row (8 elements per thread, a warp shuffle
// tree, then warp 0 over the 16 warp partials), thread 0 adds its block's sum
// into the peer block's local copy with one remote atomic (the peer is rank
// + 1 mod 2, so with 2 blocks each block ends up holding the row's total),
// and every thread scales its 8 elements by rsqrt(mean + eps) and the weight.
//
// Port of ClusterFusion include/H100/norm/kernel.cuh (commit 8755446, the
// tree of the residual_out_c4_fix patch, which leaves this file unchanged),
// kernel RMSNormKernel, __cluster_dims__(2, 1, 1), at config.h's
// HIDDEN_DIM 8192, CLUSTER_SIZE 2, NUM_PER_THREAD 8 (512 threads).
// Departures:
// - reduction[] and cluster_local_sum (the kernel's __shared__ variables) are
//   the members of RMSNormSmem at offset 0 of the block's local memory; the
//   launch requests those 68 bytes as dynamic local memory;
// - the parameters are one argument block, and thread 0 of each block also
//   stores its core id (smid) for the harness's placement check;
// - uint is a local typedef (the kernel libc has none);
// - dst_shmem starts as nullptr (only thread 0 sets and uses it) so that the
//   build's -Werror accepts it;
// - the uint4 copies of 8 halves compile to narrower loads and stores
//   (Vortex has no 128-bit access);
// - built with -fconvergent-functions (Makefile): the barriers are inline asm,
//   which clang may otherwise copy into both arms of a divergent branch.
// The single shift-by-one atomic round is the original's: it gives the row
// total only for a cluster of 2, the size the H100 ran.

#include <vx_cuda.h>
#include "common.h"

namespace cg = cooperative_groups;

// The CUDA headers provide uint; the kernel toolchain's libc does not.
typedef unsigned int uint;

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) RMSNormKernel(RMSNormArgs* __UNIFORM__ args) {
  VX_CUDA_KERNEL_ENTRY(RMSNormKernel);
  half* output = (half*)args->output;
  half* input = (half*)args->input;
  half* weight = (half*)args->weight;

  cg::grid_group grid = cg::this_grid();
  cg::cluster_group cluster = cg::this_cluster();
  cg::thread_block block = cg::this_thread_block();
  const uint32_t batch_id = grid.cluster_rank();
  const uint32_t cluster_block_id = cluster.block_rank();
  const uint32_t tid = block.thread_rank();
  const uint32_t lane_id = tid % WARP_SIZE;
  const uint32_t warp_id = tid / WARP_SIZE;
  if (threadIdx.x == 0)
    ((unsigned*)args->smid)[blockIdx.x] = __smid();

  RMSNormSmem& sm = *reinterpret_cast<RMSNormSmem*>(__local_mem());
  float* reduction = sm.reduction;
  float& cluster_local_sum = sm.cluster_local_sum;

  float local_sum = 0.0, eps = 1e-6, rms_rcp = 0.0;
  half __align__(16) reg_input[NUM_PER_THREAD];
  half __align__(16) reg_weight[NUM_PER_THREAD];
  float* dst_shmem = nullptr;

  uint cluster_block_st_idx = cluster_block_id * DIM_PER_BLOCK;

  // Sum of squares of this thread's 8 elements (one pass at this shape).
  for (int d = tid * 8; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 8) {
    *(uint4*)(&reg_input[0]) = *(uint4*)(&input[batch_id * HIDDEN_DIM + cluster_block_st_idx + d]);
#pragma unroll
    for (int di = 0; di < 8; di++)
      local_sum += __half2float(reg_input[di]) * __half2float(reg_input[di]);
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
  if (lane_id == 0)
    reduction[warp_id] = local_sum;
  block.sync();
  if (tid < NUM_WARPS)
    local_sum = reduction[tid];
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1)
    local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
  if (tid == 0)
    cluster_local_sum = local_sum;
  block.sync();

  // One remote atomic add into the next block's copy of the sum.
  if (tid == 0) {
    int dst_cta = (cluster_block_id + 1) % cluster.num_blocks();
    dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
  }
  cluster.sync();
  if (tid == 0)
    atomicAdd(dst_shmem, local_sum);
  cluster.sync();
  rms_rcp = __frsqrt_rn(cluster_local_sum / HIDDEN_DIM + eps);

  // reg_input still holds this thread's elements from the first loop.
  for (int d = tid * 8; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 8) {
    *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[cluster_block_st_idx + d]);
#pragma unroll
    for (int i = 0; i < 8; i++)
      reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
    *(uint4*)(&output[batch_id * HIDDEN_DIM + cluster_block_st_idx + d]) = *(uint4*)(&reg_input[0]);
  }
}
