// Pearson-correlation backward with a row split over a cluster: block b
// stages slice b of the row's x and y; one reduction of five sums (x, y, x^2,
// y^2, xy), over the block and then over the cluster, gives the means,
// variances and covariance, and every block writes dx and dy for its slice.
//
// Port of CREDIT src/workloads/pearson_backward.cu (commit c8f2ad2; the
// roofline_hooks patch the H100 build applied changes host code only),
// kernel cluster_bwd_staged_kernel<256, 8> with subwarp_reduce_sum,
// block_reduce_stats and cluster_reduce_stats.
// Departures:
// - the helpers' __shared__ variables are the members of pearson::Smem at
//   offset 0 of the block's local memory, the two tile arrays follow at a
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

namespace pearson {

using namespace credit;

template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
  for (int offset = Width / 2; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
  return value;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ Stats block_reduce_stats(Stats value) {
  constexpr int Warps = BlockThreads / 32;
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  value.sx = subwarp_reduce_sum<32>(value.sx);
  value.sy = subwarp_reduce_sum<32>(value.sy);
  value.sxx = subwarp_reduce_sum<32>(value.sxx);
  value.syy = subwarp_reduce_sum<32>(value.syy);
  value.sxy = subwarp_reduce_sum<32>(value.sxy);
  if ((threadIdx.x & 31) == 0) {
    const int warp = threadIdx.x >> 5;
    sv.warp_sx[warp] = value.sx;
    sv.warp_sy[warp] = value.sy;
    sv.warp_sxx[warp] = value.sxx;
    sv.warp_syy[warp] = value.syy;
    sv.warp_sxy[warp] = value.sxy;
  }
  __syncthreads();

  Stats total{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (threadIdx.x < Warps) {
    total.sx = sv.warp_sx[threadIdx.x];
    total.sy = sv.warp_sy[threadIdx.x];
    total.sxx = sv.warp_sxx[threadIdx.x];
    total.syy = sv.warp_syy[threadIdx.x];
    total.sxy = sv.warp_sxy[threadIdx.x];
  }
  if (threadIdx.x < 32) {
    total.sx = subwarp_reduce_sum<32>(total.sx);
    total.sy = subwarp_reduce_sum<32>(total.sy);
    total.sxx = subwarp_reduce_sum<32>(total.sxx);
    total.syy = subwarp_reduce_sum<32>(total.syy);
    total.sxy = subwarp_reduce_sum<32>(total.sxy);
  }
  if (threadIdx.x == 0)
    sv.block_broadcast = total;
  __syncthreads();
  return sv.block_broadcast;
}

// Replicated push of the five block sums into slot [rank] of every block,
// a cluster barrier, a local sum of the slots, and a barrier that protects the
// slots until every block has read them.
template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ Stats cluster_reduce_stats(cg::cluster_group cluster, Stats block_value) {
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  const int block_rank = cluster.block_rank();
  const int blocks_per_cluster = cluster.dim_blocks().x;

  if ((int)threadIdx.x < blocks_per_cluster) {
    float* remote_sx = cluster.map_shared_rank(&sv.cluster_sx[block_rank], threadIdx.x);
    float* remote_sy = cluster.map_shared_rank(&sv.cluster_sy[block_rank], threadIdx.x);
    float* remote_sxx = cluster.map_shared_rank(&sv.cluster_sxx[block_rank], threadIdx.x);
    float* remote_syy = cluster.map_shared_rank(&sv.cluster_syy[block_rank], threadIdx.x);
    float* remote_sxy = cluster.map_shared_rank(&sv.cluster_sxy[block_rank], threadIdx.x);
    *remote_sx = block_value.sx;
    *remote_sy = block_value.sy;
    *remote_sxx = block_value.sxx;
    *remote_syy = block_value.syy;
    *remote_sxy = block_value.sxy;
  }
  cluster.sync();

  Stats total{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if ((int)threadIdx.x < blocks_per_cluster) {
    total.sx = sv.cluster_sx[threadIdx.x];
    total.sy = sv.cluster_sy[threadIdx.x];
    total.sxx = sv.cluster_sxx[threadIdx.x];
    total.syy = sv.cluster_syy[threadIdx.x];
    total.sxy = sv.cluster_sxy[threadIdx.x];
  }
  if (threadIdx.x < 32) {
    total.sx = subwarp_reduce_sum<32>(total.sx);
    total.sy = subwarp_reduce_sum<32>(total.sy);
    total.sxx = subwarp_reduce_sum<32>(total.sxx);
    total.syy = subwarp_reduce_sum<32>(total.syy);
    total.sxy = subwarp_reduce_sum<32>(total.sxy);
  }
  if (threadIdx.x == 0)
    sv.cluster_broadcast = total;
  __syncthreads();
  cluster.sync();
  return sv.cluster_broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ void cluster_bwd_staged_kernel(Args* args) {
  using S = Smem<BlockThreads, MaxClusterSize>;
  const float* __restrict__ x = (const float*)args->x;
  const float* __restrict__ y = (const float*)args->y;
  const float* __restrict__ grad = (const float*)args->grad;
  float* __restrict__ dx = (float*)args->dx;
  float* __restrict__ dy = (float*)args->dy;
  const int rows = args->rows, cols = args->cols, slice_elems = args->slice_elems;
  const float eps = args->eps;
  float* tile = credit::tile<S>();
  float* tile_x = tile;
  float* tile_y = tile + slice_elems;
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
    tile_x[col] = x[row_base + slice_start + col];
    tile_y[col] = y[row_base + slice_start + col];
  }
  __syncthreads();

  Stats stats{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const float xv = tile_x[col];
    const float yv = tile_y[col];
    stats.sx += xv;
    stats.sy += yv;
    stats.sxx += xv * xv;
    stats.syy += yv * yv;
    stats.sxy += xv * yv;
  }
  const Stats block_stats = block_reduce_stats<BlockThreads, MaxClusterSize>(stats);
  const Stats row_stats = cluster_reduce_stats<BlockThreads, MaxClusterSize>(cluster, block_stats);
  const float inv_cols = 1.0f / (float)cols;
  const float mean_x = row_stats.sx * inv_cols;
  const float mean_y = row_stats.sy * inv_cols;
  const float var_x = fmaxf(row_stats.sxx - row_stats.sx * mean_x, 0.0f);
  const float var_y = fmaxf(row_stats.syy - row_stats.sy * mean_y, 0.0f);
  const float cov = row_stats.sxy - row_stats.sx * mean_y;
  const float inv_x = rsqrtf(var_x + eps);
  const float inv_y = rsqrtf(var_y + eps);
  const float inv_xy = inv_x * inv_y;
  const float coeff_x = cov * inv_x * inv_x * inv_x * inv_y;
  const float coeff_y = cov * inv_x * inv_y * inv_y * inv_y;
  const float grad_row = row_valid ? grad[row] : 0.0f;

  if (!row_valid)
    return;
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const int global_col = slice_start + col;
    const float xc = tile_x[col] - mean_x;
    const float yc = tile_y[col] - mean_y;
    dx[row_base + global_col] = grad_row * (yc * inv_xy - xc * coeff_x);
    dy[row_base + global_col] = grad_row * (xc * inv_xy - yc * coeff_y);
  }
}

} // namespace pearson
