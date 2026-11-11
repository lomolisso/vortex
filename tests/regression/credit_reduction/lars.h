// LARS momentum update with a row split over a cluster: block b stages slice
// b of the row's weight, grad and momentum; one reduction of (sum w^2,
// sum u^2), with u = 0.9 m + g + 0.01 w, over the block and then over the
// cluster gives the trust ratio, and every block writes its slice of the new
// weight and momentum.
//
// Port of CREDIT src/workloads/lars_momentum.cu (commit c8f2ad2; the
// roofline_hooks patch the H100 build applied changes host code only), kernel
// cluster_lars_staged_kernel<256, 8> with subwarp_reduce_sum,
// block_reduce_pair, cluster_reduce_pair and update_value.
// Departures:
// - the helpers' __shared__ variables are the members of lars::Smem at offset
//   0 of the block's local memory, the three tile arrays follow at a 16-byte
//   aligned offset, and the launch requests both as dynamic local memory
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

namespace lars {

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

// Replicated push of the block's pair into slot [rank] of every block, a
// cluster barrier, a local sum of the slots, and a barrier that protects the
// slots until every block has read them.
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

__device__ __forceinline__ float update_value(float weight, float grad, float momentum) {
  return kMomentumCoeff * momentum + grad + kWeightDecay * weight;
}

template <int BlockThreads, int MaxClusterSize>
__device__ void cluster_lars_staged_kernel(Args* args) {
  using S = Smem<BlockThreads, MaxClusterSize>;
  const float* __restrict__ weight = (const float*)args->weight;
  const float* __restrict__ grad = (const float*)args->grad;
  const float* __restrict__ momentum = (const float*)args->momentum;
  float* __restrict__ weight_out = (float*)args->weight_out;
  float* __restrict__ momentum_out = (float*)args->momentum_out;
  const int rows = args->rows, cols = args->cols, slice_elems = args->slice_elems;
  float* tile = credit::tile<S>();
  float* tile_w = tile;
  float* tile_g = tile + slice_elems;
  float* tile_m = tile + 2 * slice_elems;
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
    tile_w[col] = weight[row_base + slice_start + col];
    tile_g[col] = grad[row_base + slice_start + col];
    tile_m[col] = momentum[row_base + slice_start + col];
  }
  __syncthreads();

  Pair stats{0.0f, 0.0f};
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const float update = update_value(tile_w[col], tile_g[col], tile_m[col]);
    stats.a += tile_w[col] * tile_w[col];
    stats.b += update * update;
  }
  const Pair block_stats = block_reduce_pair<BlockThreads, MaxClusterSize>(stats);
  const Pair row_stats = cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_stats);
  const float trust = kTrustCoeff * sqrtf(row_stats.a) / (sqrtf(row_stats.b) + kEps);
  const float step = kLearningRate * trust;

  if (!row_valid)
    return;
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
    const int global_col = slice_start + col;
    const float update = update_value(tile_w[col], tile_g[col], tile_m[col]);
    weight_out[row_base + global_col] = tile_w[col] - step * update;
    momentum_out[row_base + global_col] = update;
  }
}

} // namespace lars
