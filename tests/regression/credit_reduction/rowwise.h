// Row-wise symmetric int8 quantization with a row split over a cluster:
// block b stages slice b of the row; the row's max |x| is reduced over the
// block and then over the cluster, rank 0 writes the row's scale, every block
// writes its slice of round(x * 127 / max) clamped to [-127, 127], and all
// wait at a last cluster barrier.
//
// Port of CREDIT src/workloads/rowwise_quant.cu (commit c8f2ad2; the
// roofline_hooks patch the H100 build applied changes host code only), kernel
// cluster_quant_staged_kernel<256, 8> with subwarp_reduce_max,
// block_reduce_max, cluster_reduce_max and quantize_to_i8.
// Departures:
// - the helpers' __shared__ variables are the members of rowwise::Smem at
//   offset 0 of the block's local memory, the tile follows at a 16-byte
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
// The max reductions start from 0 (not -inf), as the original: the values
// reduced are absolute values.

#pragma once

#include "credit_dev.h"

namespace rowwise {

using namespace credit;

template <int Width>
__device__ __forceinline__ float subwarp_reduce_max(float value) {
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, value, offset, Width);
    value = fmaxf(value, other);
  }
  return value;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float block_reduce_max(float value) {
  constexpr int Warps = BlockThreads / 32;
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  value = subwarp_reduce_max<32>(value);
  if ((threadIdx.x & 31) == 0)
    sv.warp_values[threadIdx.x >> 5] = value;
  __syncthreads();

  float total = 0.0f;
  if (threadIdx.x < Warps)
    total = sv.warp_values[threadIdx.x];
  if (threadIdx.x < 32)
    total = subwarp_reduce_max<32>(total);
  if (threadIdx.x == 0)
    sv.block_broadcast = total;
  __syncthreads();
  return sv.block_broadcast;
}

// Replicated push of the block's max into slot [rank] of every block, a
// cluster barrier, a local max of the slots, and a barrier that protects the
// slots until every block has read them.
template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float cluster_reduce_max(cg::cluster_group cluster, float block_value) {
  auto& sv = shared_vars<Smem<BlockThreads, MaxClusterSize>>();
  const int block_rank = cluster.block_rank();
  const int blocks_per_cluster = cluster.dim_blocks().x;

  if ((int)threadIdx.x < blocks_per_cluster) {
    float* remote_value = cluster.map_shared_rank(&sv.cluster_values[block_rank], threadIdx.x);
    *remote_value = block_value;
  }
  cluster.sync();

  float total = 0.0f;
  if ((int)threadIdx.x < blocks_per_cluster)
    total = sv.cluster_values[threadIdx.x];
  if (threadIdx.x < 32)
    total = subwarp_reduce_max<32>(total);
  if (threadIdx.x == 0)
    sv.cluster_broadcast = total;
  __syncthreads();
  cluster.sync();
  return sv.cluster_broadcast;
}

__device__ __forceinline__ int8_t quantize_to_i8(float value, float inv_scale) {
  float q = nearbyintf(value * inv_scale);
  q = fminf(kQMax, fmaxf(-kQMax, q));
  return (int8_t)q;
}

template <int BlockThreads, int MaxClusterSize>
__device__ void cluster_quant_staged_kernel(Args* args) {
  using S = Smem<BlockThreads, MaxClusterSize>;
  const float* __restrict__ input = (const float*)args->input;
  int8_t* __restrict__ output = (int8_t*)args->output;
  float* __restrict__ scales = (float*)args->scales;
  const int rows = args->rows, cols = args->cols, slice_elems = args->slice_elems;
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
    tile[col] = input[row_base + slice_start + col];
  __syncthreads();

  float local_max = 0.0f;
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads)
    local_max = fmaxf(local_max, fabsf(tile[col]));
  const float block_max = block_reduce_max<BlockThreads, MaxClusterSize>(local_max);
  const float max_abs = cluster_reduce_max<BlockThreads, MaxClusterSize>(cluster, block_max);
  const float inv_scale = max_abs > 0.0f ? kQMax / max_abs : 0.0f;
  if (row_valid && block_rank == 0 && threadIdx.x == 0)
    scales[row] = max_abs / kQMax;
  if (!row_valid)
    return;
  for (int col = threadIdx.x; col < local_cols; col += BlockThreads)
    output[row_base + slice_start + col] = quantize_to_i8(tile[col], inv_scale);
  cluster.sync();
}

} // namespace rowwise
