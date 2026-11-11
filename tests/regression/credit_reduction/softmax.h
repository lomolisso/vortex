// Softmax backward from the logits with a row split over a cluster: block b
// stages slice b of the row's logits and dy; the row max, then the sum of
// exp(l - max), then the dot of p and dy are each reduced over the block and
// then over the cluster, and every block writes dx = p * (dy - dot).
//
// Port of CREDIT src/workloads/softmax_logits_backward.cu (commit c8f2ad2;
// the roofline_hooks patch the H100 build applied changes host code only),
// kernel cluster_softmax_logits_bwd_staged_kernel<256, 8> with
// subwarp_reduce_sum/max, block_reduce_sum/max and cluster_reduce_sum/max.
// Departures:
// - the helpers' __shared__ variables are the members of softmax::Smem at
//   offset 0 of the block's local memory (one set per helper, as each helper
//   declares its own), the two tile arrays follow at a 16-byte aligned
//   offset, and the launch requests both as dynamic local memory
//   (__shared__ does not exist on Vortex); the helpers take MaxClusterSize as
//   well, to name that layout;
// - the parameters are one argument block, and the template kernel is
//   exported by name;
// - thread 0 of each block also stores its core id (smid) for the harness's
//   placement check;
// - the `threadIdx.x < blocks_per_cluster` tests compare `(int)threadIdx.x`
//   (the build's -Wsign-compare is an error);
// - built with -fconvergent-functions (Makefile): the barriers are inline asm,
//   which clang may otherwise copy into both arms of a divergent branch.

#pragma once

#include "credit_dev.h"

namespace softmax {

using namespace credit;

template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
  for (int offset = Width / 2; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
  return value;
}

template <int Width>
__device__ __forceinline__ float subwarp_reduce_max(float value) {
  for (int offset = Width / 2; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset, Width));
  return value;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float block_reduce_sum(float value) {
  constexpr int Warps = BlockThreads / 32;
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  value = subwarp_reduce_sum<32>(value);
  if ((threadIdx.x & 31) == 0)
    sv.sum_warp_values[threadIdx.x >> 5] = value;
  __syncthreads();

  float total = 0.0f;
  if (threadIdx.x < Warps)
    total = sv.sum_warp_values[threadIdx.x];
  if (threadIdx.x < 32)
    total = subwarp_reduce_sum<32>(total);
  if (threadIdx.x == 0)
    sv.sum_broadcast = total;
  __syncthreads();
  return sv.sum_broadcast;
}

// The identity of the max is -inf, also for the lanes past the warp count.
template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float block_reduce_max(float value) {
  constexpr int Warps = BlockThreads / 32;
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  value = subwarp_reduce_max<32>(value);
  if ((threadIdx.x & 31) == 0)
    sv.max_warp_values[threadIdx.x >> 5] = value;
  __syncthreads();

  float total = -CUDART_INF_F;
  if (threadIdx.x < Warps)
    total = sv.max_warp_values[threadIdx.x];
  if (threadIdx.x < 32)
    total = subwarp_reduce_max<32>(total);
  if (threadIdx.x == 0)
    sv.max_broadcast = total;
  __syncthreads();
  return sv.max_broadcast;
}

// Replicated push of the block's value into slot [rank] of every block, a
// cluster barrier, a local reduction of the slots, and a barrier that
// protects the slots until every block has read them.
template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float cluster_reduce_sum(cg::cluster_group cluster, float block_value) {
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  const int block_rank = cluster.block_rank();
  const int blocks_per_cluster = cluster.dim_blocks().x;

  if ((int)threadIdx.x < blocks_per_cluster) {
    float* remote = cluster.map_shared_rank(&sv.sum_slots[block_rank], threadIdx.x);
    *remote = block_value;
  }
  cluster.sync();

  float total = 0.0f;
  if ((int)threadIdx.x < blocks_per_cluster)
    total = sv.sum_slots[threadIdx.x];
  if (threadIdx.x < 32)
    total = subwarp_reduce_sum<32>(total);
  if (threadIdx.x == 0)
    sv.cluster_sum_broadcast = total;
  __syncthreads();
  cluster.sync();
  return sv.cluster_sum_broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float cluster_reduce_max(cg::cluster_group cluster, float block_value) {
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  const int block_rank = cluster.block_rank();
  const int blocks_per_cluster = cluster.dim_blocks().x;

  if ((int)threadIdx.x < blocks_per_cluster) {
    float* remote = cluster.map_shared_rank(&sv.max_slots[block_rank], threadIdx.x);
    *remote = block_value;
  }
  cluster.sync();

  float total = -CUDART_INF_F;
  if ((int)threadIdx.x < blocks_per_cluster)
    total = sv.max_slots[threadIdx.x];
  if (threadIdx.x < 32)
    total = subwarp_reduce_max<32>(total);
  if (threadIdx.x == 0)
    sv.cluster_max_broadcast = total;
  __syncthreads();
  cluster.sync();
  return sv.cluster_max_broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ void cluster_softmax_logits_bwd_staged_kernel(Args* args) {
  using S = Smem<BlockThreads, MaxClusterSize>;
  const float* __restrict__ logits = (const float*)args->logits;
  const float* __restrict__ dy = (const float*)args->dy;
  float* __restrict__ dx = (float*)args->dx;
  const int rows = args->rows, cols = args->cols, slice_elems = args->slice_elems;
  float* tile = credit::tile<S>();
  float* tile_logits = tile;
  float* tile_dy = tile + slice_elems;
  record_smid(args->smid);

  cg::cluster_group cluster = cg::this_cluster();
  const int blocks_per_cluster = cluster.dim_blocks().x;
  const int block_rank = cluster.block_rank();
  const int row = blockIdx.x / blocks_per_cluster;
  const bool row_valid = row < rows;
  const int slice_start = block_rank * slice_elems;
  const int remaining = cols - slice_start;
  const int local_cols = (row_valid && remaining > 0) ? (remaining < slice_elems ? remaining : slice_elems) : 0;
  const size_t row_base = (size_t)row * cols;

  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const int global_col = slice_start + col;
    tile_logits[col] = logits[row_base + global_col];
    tile_dy[col] = dy[row_base + global_col];
  }
  __syncthreads();

  float local_max = -CUDART_INF_F;
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads)
    local_max = fmaxf(local_max, tile_logits[col]);
  const float block_max = block_reduce_max<BlockThreads, MaxClusterSize>(local_max);
  const float row_max = cluster_reduce_max<BlockThreads, MaxClusterSize>(cluster, block_max);

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads)
    local_sum += expf(tile_logits[col] - row_max);
  const float block_sum = block_reduce_sum<BlockThreads, MaxClusterSize>(local_sum);
  const float row_sum = cluster_reduce_sum<BlockThreads, MaxClusterSize>(cluster, block_sum);

  float local_dot = 0.0f;
  const float inv_sum = 1.0f / row_sum;
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const float p = expf(tile_logits[col] - row_max) * inv_sum;
    local_dot += p * tile_dy[col];
  }
  const float block_dot = block_reduce_sum<BlockThreads, MaxClusterSize>(local_dot);
  const float row_dot = cluster_reduce_sum<BlockThreads, MaxClusterSize>(cluster, block_dot);

  if (!row_valid)
    return;
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const int global_col = slice_start + col;
    const float p = expf(tile_logits[col] - row_max) * inv_sum;
    dx[row_base + global_col] = p * (tile_dy[col] - row_dot);
  }
}

} // namespace softmax
