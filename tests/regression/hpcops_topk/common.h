#ifndef _HPCOPS_TOPK_COMMON_H_
#define _HPCOPS_TOPK_COMMON_H_

#include <stdint.h>

namespace hpcops_topk {

// 11-bit coarse key (f16 projection) and 11/11/10-bit f32 refinement rounds.
constexpr int RADIX = 2048;
constexpr int NUM_REFINE_ROUNDS = 3;
constexpr int COARSE_CANDIDATE_CAPACITY = 2048;
constexpr int SAMPLED_CANDIDATE_CAPACITY = 4096;

// The cluster-8 kernel's instantiation <VEC, 2048, 8, 1024>.
constexpr int kTopK = 2048;
constexpr int kSplits = 8;
constexpr int kBlockThreads = 1024;

template <int kThreads>
struct PrefixStorage {
  int warp_sum[kThreads / 32];
};

// One row's shared state, in the original's member order: the coarse or fine
// histogram (prefix-summed in place), the boundary bin, the selected and tie
// cursors, the ping-pong candidate counts (resident and spilled), the result
// indices, the ping-pong candidate buffers and the per-warp prefix totals.
template <int kK, bool kSampled, int kThreads>
struct RowSmem {
  static constexpr int CANDIDATE_CAPACITY = kSampled ? SAMPLED_CANDIDATE_CAPACITY : COARSE_CANDIDATE_CAPACITY;

  int histogram[RADIX + 1];
  int threshold_bin;
  int num_selected;
  int ties_remaining;
  int num_cand[2];
  int num_spilled_cand[2];
  int32_t indices[kK];
  alignas(8) int32_t cand_idx[2][CANDIDATE_CAPACITY];
  PrefixStorage<kThreads> prefix;
};

typedef RowSmem<kTopK, false, kBlockThreads> ClusterSmem;
static_assert(sizeof(ClusterSmem) == 32928, "RowSmem<2048, false, 1024> is 32928 bytes on the H100");

// The route the primary CTA took, recorded for the harness.
enum Route : uint32_t {
  kRouteShortRow = 1,       // length <= k: the identity answer
  kRouteLargeBoundary = 2,  // VEC 4: the boundary bin exceeds one candidate per thread
  kRouteRegister = 3,       // VEC 4: one boundary candidate per primary thread
  kRouteResident = 4,       // VEC 2: every boundary candidate fits the primary's buffers
  kRouteOverflow = 5,       // VEC 2: the boundary overflows; the spilled continuation
};

// Per CTA (cluster rank r of row y at y * 8 + r): [0] its core, [1] from the
// primary the route, [2] on VEC 2 rows the candidates it spilled to spill1.
constexpr int kInfoWords = 3;

// The original kernel's parameters, in order, plus the record buffer.
typedef struct {
  uint64_t input;           // const float*  [m][row_stride]
  uint64_t output;          // int32_t*      [m][out_stride]
  uint64_t ke;              // const int32_t* [m] valid length per row
  uint64_t spill_buffer;    // int32_t*      [m][2][n_cols]
  uint64_t num_valid_rows;  // const int32_t*
  uint64_t info;            // uint32_t*     [m * 8][kInfoWords]
  uint32_t row_stride;
  uint32_t out_stride;
  uint32_t n_cols;
} kernel_arg_t;

} // namespace hpcops_topk

#endif
