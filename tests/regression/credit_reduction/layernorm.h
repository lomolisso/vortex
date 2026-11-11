// LayerNorm backward with a row split over a cluster: block b of a cluster
// stages slice b of the row's x in its local memory; the row's (sum x,
// sum x^2) and then (sum dy*g, sum dy*g*xhat) are each reduced in the block
// and then over the cluster, and every block writes dx for its slice.
//
// Port of CREDIT src/workloads/layernorm_backward.cu (commit c8f2ad2; the
// roofline_hooks patch the H100 build applied changes host code only),
// kernel cluster_bwd_staged_kernel<256, 8> with its helpers
// subwarp_reduce_sum, block_reduce_pair and cluster_reduce_pair.
// Departures:
// - the helpers' __shared__ variables are the members of layernorm::Smem at
//   offset 0 of the block's local memory, the tile follows at a 16-byte
//   aligned offset, and the launch requests both as dynamic local memory
//   (__shared__ does not exist on Vortex); the helpers take MaxClusterSize
//   as well, to name that layout;
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

namespace layernorm {

using namespace credit;

template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
  for (int offset = Width / 2; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
  return value;
}

// Warp sums, then warp 0 adds the per-warp partials; the result is broadcast
// through local memory.
template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ Pair block_reduce_pair(Pair value) {
  constexpr int Warps = BlockThreads / 32;
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  value.a = subwarp_reduce_sum<32>(value.a);
  value.b = subwarp_reduce_sum<32>(value.b);
  if ((threadIdx.x & 31) == 0) {
    const int warp = threadIdx.x >> 5;
    sv.warp_a[warp] = value.a;
    sv.warp_b[warp] = value.b;
  }
  __syncthreads();

  Pair total{0.0f, 0.0f};
  if (threadIdx.x < Warps) {
    total.a = sv.warp_a[threadIdx.x];
    total.b = sv.warp_b[threadIdx.x];
  }
  if (threadIdx.x < 32) {
    total.a = subwarp_reduce_sum<32>(total.a);
    total.b = subwarp_reduce_sum<32>(total.b);
  }
  if (threadIdx.x == 0)
    sv.block_broadcast = total;
  __syncthreads();
  return sv.block_broadcast;
}

// Replicated push: thread t writes this block's pair into slot [rank] of
// block t; after the cluster barrier each block adds its own slots. The second
// barrier keeps the slots from being overwritten by a later call before every
// block has read them.
template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ Pair cluster_reduce_pair(cg::cluster_group cluster, Pair block_value) {
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  const int block_rank = cluster.block_rank();
  const int blocks_per_cluster = cluster.dim_blocks().x;

  if ((int)threadIdx.x < blocks_per_cluster) {
    float* remote_a = cluster.map_shared_rank(&sv.cluster_a[block_rank], threadIdx.x);
    float* remote_b = cluster.map_shared_rank(&sv.cluster_b[block_rank], threadIdx.x);
    *remote_a = block_value.a;
    *remote_b = block_value.b;
  }
  cluster.sync();

  Pair total{0.0f, 0.0f};
  if ((int)threadIdx.x < blocks_per_cluster) {
    total.a = sv.cluster_a[threadIdx.x];
    total.b = sv.cluster_b[threadIdx.x];
  }
  if (threadIdx.x < 32) {
    total.a = subwarp_reduce_sum<32>(total.a);
    total.b = subwarp_reduce_sum<32>(total.b);
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
  const float* __restrict__ dy = (const float*)args->dy;
  const float* __restrict__ gamma = (const float*)args->gamma;
  float* __restrict__ dx = (float*)args->dx;
  const int rows = args->rows, cols = args->cols, slice_elems = args->slice_elems;
  const float eps = args->eps;
  float* tile = credit::tile<S>();
  record_smid(args->smid);

  cg::cluster_group cluster = cg::this_cluster();
  const int blocks_per_cluster = cluster.dim_blocks().x;
  const int block_rank = cluster.block_rank();
  const int row = blockIdx.x / blocks_per_cluster;
  const bool row_valid = row < rows;
  const int slice_start = block_rank * slice_elems;
  const int local_cols = row_valid ? max(0, min(slice_elems, cols - slice_start)) : 0;
  const size_t row_base = (size_t)row * cols;

  for (int col = threadIdx.x; col < local_cols; col += BlockThreads)
    tile[col] = x[row_base + slice_start + col];
  __syncthreads();

  Pair x_stats{0.0f, 0.0f};
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const float xv = tile[col];
    x_stats.a += xv;
    x_stats.b += xv * xv;
  }
  const Pair block_x = block_reduce_pair<BlockThreads, MaxClusterSize>(x_stats);
  const Pair row_x = cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_x);
  const float mean = row_x.a / cols;
  const float var = fmaxf(row_x.b / cols - mean * mean, 0.0f);
  const float inv_std = rsqrtf(var + eps);

  Pair grad_stats{0.0f, 0.0f};
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const int global_col = slice_start + col;
    const float xhat = (tile[col] - mean) * inv_std;
    const float dyg = dy[row_base + global_col] * gamma[global_col];
    grad_stats.a += dyg;
    grad_stats.b += dyg * xhat;
  }
  const Pair block_grad = block_reduce_pair<BlockThreads, MaxClusterSize>(grad_stats);
  const Pair row_grad = cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_grad);
  const float mean_dyg = row_grad.a / cols;
  const float mean_dyg_xhat = row_grad.b / cols;

  if (!row_valid)
    return;
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const int global_col = slice_start + col;
    const float xhat = (tile[col] - mean) * inv_std;
    const float dyg = dy[row_base + global_col] * gamma[global_col];
    dx[row_base + global_col] = (dyg - mean_dyg - xhat * mean_dyg_xhat) * inv_std;
  }
}

} // namespace layernorm
