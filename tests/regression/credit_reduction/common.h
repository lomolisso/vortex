#ifndef _CREDIT_COMMON_H_
#define _CREDIT_COMMON_H_

// What the host and the kernels share: the argument block of each kernel and
// the layout of each kernel's block-local memory.
//
// Local memory of a block: the kernel's shared variables (the ones the
// original declares __shared__ inside its reduction helpers) as one struct at
// offset 0, then the staging tile (the original's dynamic shared memory) at
// tile_offset<S>(): k arrays of slice_elems floats each. Every block of a
// cluster uses the same offsets, so a peer's variable is at the same offset
// in its own window.

#include <stddef.h>
#include <stdint.h>

namespace credit {

// Compile-time in the H100 build; the cluster size P and every problem size
// are launch parameters.
constexpr int kBlockThreads = 256;
constexpr int kMaxClusterSize = 8;

template <class S>
constexpr uint32_t tile_offset() {
  return (uint32_t)((sizeof(S) + 15) & ~(size_t)15);
}

} // namespace credit

namespace layernorm {
struct Pair {
  float a;
  float b;
};
template <int BlockThreads, int MaxClusterSize>
struct Smem {
  // block_reduce_pair
  float warp_a[BlockThreads / 32];
  float warp_b[BlockThreads / 32];
  Pair block_broadcast;
  // cluster_reduce_pair
  float cluster_a[MaxClusterSize];
  float cluster_b[MaxClusterSize];
  Pair cluster_broadcast;
};
constexpr int kTileArrays = 1;
struct Args {
  uint64_t x, dy, gamma, dx, smid;
  int32_t rows, cols;
  float eps;
  int32_t slice_elems;
};
} // namespace layernorm

namespace weighted_var {
struct Pair {
  float sum;
  float sumsq;
};
template <int BlockThreads, int MaxClusterSize>
struct Smem {
  // block_reduce_pair
  float warp_sum[BlockThreads / 32];
  float warp_sumsq[BlockThreads / 32];
  Pair block_broadcast;
  // cluster_reduce_pair
  float cluster_sum[MaxClusterSize];
  float cluster_sumsq[MaxClusterSize];
  Pair cluster_broadcast;
};
constexpr int kTileArrays = 3;
constexpr float kEps = 1.0e-6f;
struct Args {
  uint64_t input, weights, dy, output, smid;
  int32_t rows, cols, slice_elems;
};
} // namespace weighted_var

namespace pearson {
struct Stats {
  float sx;
  float sy;
  float sxx;
  float syy;
  float sxy;
};
template <int BlockThreads, int MaxClusterSize>
struct Smem {
  // block_reduce_stats
  float warp_sx[BlockThreads / 32];
  float warp_sy[BlockThreads / 32];
  float warp_sxx[BlockThreads / 32];
  float warp_syy[BlockThreads / 32];
  float warp_sxy[BlockThreads / 32];
  Stats block_broadcast;
  // cluster_reduce_stats
  float cluster_sx[MaxClusterSize];
  float cluster_sy[MaxClusterSize];
  float cluster_sxx[MaxClusterSize];
  float cluster_syy[MaxClusterSize];
  float cluster_sxy[MaxClusterSize];
  Stats cluster_broadcast;
};
constexpr int kTileArrays = 2;
struct Args {
  uint64_t x, y, grad, dx, dy, smid;
  int32_t rows, cols;
  float eps;
  int32_t slice_elems;
};
} // namespace pearson

namespace softmax {
template <int BlockThreads, int MaxClusterSize>
struct Smem {
  // block_reduce_sum
  float sum_warp_values[BlockThreads / 32];
  float sum_broadcast;
  // block_reduce_max
  float max_warp_values[BlockThreads / 32];
  float max_broadcast;
  // cluster_reduce_sum
  float sum_slots[MaxClusterSize];
  float cluster_sum_broadcast;
  // cluster_reduce_max
  float max_slots[MaxClusterSize];
  float cluster_max_broadcast;
};
constexpr int kTileArrays = 2;
struct Args {
  uint64_t logits, dy, dx, smid;
  int32_t rows, cols, slice_elems;
};
} // namespace softmax

namespace lars {
struct Pair {
  float a;
  float b;
};
template <int BlockThreads, int MaxClusterSize>
struct Smem {
  // block_reduce_pair
  float warp_a[BlockThreads / 32];
  float warp_b[BlockThreads / 32];
  Pair block_broadcast;
  // cluster_reduce_pair
  float cluster_a[MaxClusterSize];
  float cluster_b[MaxClusterSize];
  Pair cluster_broadcast;
};
constexpr int kTileArrays = 3;
constexpr float kEps = 1e-6f;
constexpr float kMomentumCoeff = 0.9f;
constexpr float kWeightDecay = 0.01f;
constexpr float kLearningRate = 1.0e-3f;
constexpr float kTrustCoeff = 0.02f;
struct Args {
  uint64_t weight, grad, momentum, weight_out, momentum_out, smid;
  int32_t rows, cols, slice_elems;
};
} // namespace lars

namespace rowwise {
template <int BlockThreads, int MaxClusterSize>
struct Smem {
  // block_reduce_max
  float warp_values[BlockThreads / 32];
  float block_broadcast;
  // cluster_reduce_max
  float cluster_values[MaxClusterSize];
  float cluster_broadcast;
};
constexpr int kTileArrays = 1;
constexpr float kQMax = 127.0f;
struct Args {
  uint64_t input, output, scales, smid;
  int32_t rows, cols, slice_elems;
};
} // namespace rowwise

#endif
