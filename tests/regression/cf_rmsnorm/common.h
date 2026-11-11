#ifndef _CF_RMSNORM_COMMON_H_
#define _CF_RMSNORM_COMMON_H_

#include <stdint.h>

// ClusterFusion include/H100/norm/config.h. Its BATCH_SIZE only sizes the
// host's grid (one cluster per row); here the batch is the -b option.
#define HIDDEN_DIM 8192
#define CLUSTER_SIZE 2
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)
#define NUM_PER_THREAD 8
#define BLOCK_SIZE (DIM_PER_BLOCK / NUM_PER_THREAD)
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// The kernel's static shared memory (68 bytes on the H100), at offset 0 of
// the block's local memory in every block.
struct RMSNormSmem {
  float reduction[NUM_WARPS];
  float cluster_local_sum;
};

// output, input: [batch, HIDDEN_DIM] f16; weight: [HIDDEN_DIM] f16;
// smid: one word per block.
struct RMSNormArgs {
  uint64_t output, input, weight, smid;
};

VX_CUDA_CLUSTER_DIMS(RMSNormKernel, CLUSTER_SIZE, 1, 1);

#endif
