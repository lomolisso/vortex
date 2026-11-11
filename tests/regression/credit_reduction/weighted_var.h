// Weighted-variance backward with a row split over a cluster: block b stages
// slice b of the row's x, w and dy; three reductions of a pair, each over the
// block and then over the cluster, give (sum w, sum w*x), then sum w*(x-mean)^2,
// then (sum dy, sum dy*(x-mean)); every block writes its slice of the gradient
// and waits at a last cluster barrier.
//
// Port of CREDIT src/workloads/weighted_var_backward.cu (commit c8f2ad2; the
// roofline_hooks patch the H100 build applied changes host code only),
// kernel cluster_bwd_staged_kernel<256, 8> with subwarp_reduce_sum,
// block_reduce_pair, cluster_reduce_pair and backward_value.
// Departures:
// - the helpers' __shared__ variables are the members of weighted_var::Smem at
//   offset 0 of the block's local memory, the three tile arrays follow at a
//   16-byte aligned offset, and the launch requests both as dynamic local
//   memory (__shared__ does not exist on Vortex); the helpers take
//   MaxClusterSize as well, to name that layout;
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

namespace weighted_var {

using namespace credit;

template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
  for (int offset = Width / 2; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
  return value;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ Pair block_reduce_pair(Pair value) {
  constexpr int Warps = BlockThreads / 32;
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  value.sum = subwarp_reduce_sum<32>(value.sum);
  value.sumsq = subwarp_reduce_sum<32>(value.sumsq);
  if ((threadIdx.x & 31) == 0) {
    const int warp = threadIdx.x >> 5;
    sv.warp_sum[warp] = value.sum;
    sv.warp_sumsq[warp] = value.sumsq;
  }
  __syncthreads();

  Pair total{0.0f, 0.0f};
  if (threadIdx.x < Warps) {
    total.sum = sv.warp_sum[threadIdx.x];
    total.sumsq = sv.warp_sumsq[threadIdx.x];
  }
  if (threadIdx.x < 32) {
    total.sum = subwarp_reduce_sum<32>(total.sum);
    total.sumsq = subwarp_reduce_sum<32>(total.sumsq);
  }
  if (threadIdx.x == 0)
    sv.block_broadcast = total;
  __syncthreads();
  return sv.block_broadcast;
}

// Replicated push of the block's pair into slot [rank] of every block, a
// cluster barrier, a local sum of the slots, and a barrier that protects the
// slots until every block has read them.
template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ Pair cluster_reduce_pair(cg::cluster_group cluster, Pair block_value) {
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  const int block_rank = cluster.block_rank();
  const int blocks_per_cluster = cluster.dim_blocks().x;

  if ((int)threadIdx.x < blocks_per_cluster) {
    float* remote_sum = cluster.map_shared_rank(&sv.cluster_sum[block_rank], threadIdx.x);
    float* remote_sumsq = cluster.map_shared_rank(&sv.cluster_sumsq[block_rank], threadIdx.x);
    *remote_sum = block_value.sum;
    *remote_sumsq = block_value.sumsq;
  }
  cluster.sync();

  Pair total{0.0f, 0.0f};
  if ((int)threadIdx.x < blocks_per_cluster) {
    total.sum = sv.cluster_sum[threadIdx.x];
    total.sumsq = sv.cluster_sumsq[threadIdx.x];
  }
  if (threadIdx.x < 32) {
    total.sum = subwarp_reduce_sum<32>(total.sum);
    total.sumsq = subwarp_reduce_sum<32>(total.sumsq);
  }
  if (threadIdx.x == 0)
    sv.cluster_broadcast = total;
  __syncthreads();
  cluster.sync();
  return sv.cluster_broadcast;
}

__device__ __forceinline__ float backward_value(float value, float weight, float grad, float mean,
                                                float inv_sum_w, float inv_std, float sum_dy,
                                                float sum_dy_centered) {
  const float centered = value - mean;
  const float inv_var = inv_std * inv_std;
  return inv_std * (grad - weight * inv_sum_w * sum_dy - weight * centered * inv_var * inv_sum_w * sum_dy_centered);
}

template <int BlockThreads, int MaxClusterSize>
__device__ void cluster_bwd_staged_kernel(Args* args) {
  using S = Smem<BlockThreads, MaxClusterSize>;
  const float* __restrict__ input = (const float*)args->input;
  const float* __restrict__ weights = (const float*)args->weights;
  const float* __restrict__ dy = (const float*)args->dy;
  float* __restrict__ output = (float*)args->output;
  const int rows = args->rows, cols = args->cols, slice_elems = args->slice_elems;
  float* tile = credit::tile<S>();
  float* tile_x = tile;
  float* tile_w = tile + slice_elems;
  float* tile_dy = tile + 2 * slice_elems;
  record_smid(args->smid);

  cg::cluster_group cluster = cg::this_cluster();
  const int blocks_per_cluster = cluster.dim_blocks().x;
  const int block_rank = cluster.block_rank();
  const int row = blockIdx.x / blocks_per_cluster;
  const bool row_valid = row < rows;
  const int slice_start = block_rank * slice_elems;
  const int local_cols = row_valid ? max(0, min(slice_elems, cols - slice_start)) : 0;
  const size_t row_base = (size_t)row * cols;

  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const size_t idx = row_base + slice_start + col;
    tile_x[col] = input[idx];
    tile_w[col] = weights[idx];
    tile_dy[col] = dy[idx];
  }
  __syncthreads();

  Pair local{0.0f, 0.0f};
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const float value = tile_x[col];
    const float weight = tile_w[col];
    local.sum += weight;
    local.sumsq += weight * value;
  }
  const Pair block_weighted = block_reduce_pair<BlockThreads, MaxClusterSize>(local);
  const Pair row_weighted = cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_weighted);
  const float inv_sum_w = 1.0f / (row_weighted.sum + kEps);
  const float mean = row_weighted.sumsq * inv_sum_w;

  Pair local_var{0.0f, 0.0f};
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const float diff = tile_x[col] - mean;
    local_var.sum += tile_w[col] * diff * diff;
  }
  const Pair block_var = block_reduce_pair<BlockThreads, MaxClusterSize>(local_var);
  const Pair row_var_pair = cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_var);
  const float var = fmaxf(row_var_pair.sum * inv_sum_w, 0.0f);
  const float inv_std = rsqrtf(var + kEps);

  Pair local_grad{0.0f, 0.0f};
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const float grad = tile_dy[col];
    local_grad.sum += grad;
    local_grad.sumsq += grad * (tile_x[col] - mean);
  }
  const Pair block_grad = block_reduce_pair<BlockThreads, MaxClusterSize>(local_grad);
  const Pair row_grad = cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_grad);

  if (!row_valid)
    return;
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads)
    output[row_base + slice_start + col] = backward_value(tile_x[col], tile_w[col], tile_dy[col], mean, inv_sum_w,
                                                          inv_std, row_grad.sum, row_grad.sumsq);
  cluster.sync();
}

} // namespace weighted_var
