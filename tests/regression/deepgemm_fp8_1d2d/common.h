#ifndef _DEEPGEMM_FP8_1D2D_COMMON_H_
#define _DEEPGEMM_FP8_1D2D_COMMON_H_

// The argument block and the fixed configuration of the DeepGEMM SM90 FP8
// 1D2D grouped GEMM port, shared by the host and the kernel.

#include <stdint.h>
#include <dxa_tmap.h>

#ifndef DG_SHAPE_N
#error "DG_SHAPE_N must be set (Makefile)"
#endif
#ifndef DG_SHAPE_K
#error "DG_SHAPE_K must be set (Makefile)"
#endif
#ifndef DG_NUM_GROUPS
#error "DG_NUM_GROUPS must be set (Makefile)"
#endif
#ifndef DG_NUM_SMS
#error "DG_NUM_SMS must be set (Makefile)"
#endif

namespace dg {

// The H100 configuration (block 128x192x128, 4 stages, 128 TMA threads,
// multicast 2 on B); only kNumMathThreads grows, because each H100 math
// warpgroup's m64n192 becomes three m64n64 warpgroups.
constexpr uint32_t kBlockM = 128;
constexpr uint32_t kBlockN = 192;
constexpr uint32_t kBlockK = 128;
constexpr uint32_t kSwizzleAMode = 128;
constexpr uint32_t kSwizzleBMode = 128;
constexpr uint32_t kSwizzleDMode = 128;
constexpr uint32_t kNumStages = 4;
constexpr uint32_t kNumTMAThreads = 128;
constexpr uint32_t kNumMathThreads = 768;
constexpr uint32_t kNumTMAMulticast = 2;
constexpr bool kIsTMAMulticastOnA = false;
constexpr uint32_t kNumThreads = kNumTMAThreads + kNumMathThreads;

constexpr uint32_t kShapeN = DG_SHAPE_N;
constexpr uint32_t kShapeK = DG_SHAPE_K;
constexpr uint32_t kNumGroups = DG_NUM_GROUPS;
constexpr uint32_t kNumSMs = DG_NUM_SMS;

constexpr uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
constexpr uint32_t align_up(uint32_t a, uint32_t b) { return ceil_div(a, b) * b; }

// The dynamic shared memory the H100 host asks for (heuristics/sm90.hpp):
// D, 16 stages' worth of barriers, the extra B-scale rows, and the stages.
constexpr uint32_t kSmemCD = align_up(kBlockM * kBlockN * 2, 1024);
constexpr uint32_t kSmemBarriers = 16 * 8 * 2;
constexpr uint32_t kSmemExtraSFB = align_up(ceil_div(kShapeK, kBlockK) * 4 * ((kBlockK % kBlockN == 0) ? 1 : 2), 8);
constexpr uint32_t kSmemPerStage = kBlockM * kBlockK + kBlockN * kBlockK + align_up(kBlockM * 4, 128);
constexpr uint32_t kSmemSize = kSmemCD + kSmemBarriers + kSmemExtraSFB + kNumStages * kSmemPerStage;

// Producer trace: one record per scheduled tile, per CTA.
constexpr uint32_t kTraceCap = 32;
struct trace_rec_t {
  uint32_t iter, m_block, n_block, mcast;
};

} // namespace dg

// sfb: fp32 [G][ceil(N/128)][K/128] (K-major); grouped_layout: int32 [M];
// smid: uint32 [2]; trace: trace_rec_t [2][kTraceCap].
typedef struct {
  vx_tensor_map_t tensor_map_a;
  vx_tensor_map_t tensor_map_b;
  vx_tensor_map_t tensor_map_d;
  vx_tensor_map_t tensor_map_sfa;
  uint64_t sfb;
  uint64_t grouped_layout;
  uint32_t shape_m, shape_n, shape_k, pad;
  uint64_t smid;
  uint64_t trace;
} __attribute__((aligned(64))) kernel_arg_t;

// The kernel has no __cluster_dims__ (the cluster comes with the launch, as
// on the H100); this constant only lets the entry check that the launch gave
// the cluster of kNumTMAMulticast blocks the multicast code assumes.
VX_CUDA_CLUSTER_DIMS(dg_launch, 2, 1, 1);

#endif
