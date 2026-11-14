// DeepGEMM's SM90 FP8 GEMM with 1-D A scales and 2-D B scales
// (DeepGEMM 5081f85, deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh,
// with scheduler/gemm.cuh, common/tma_copy.cuh and mma/sm90.cuh), as the H100
// ran it: sm90_fp8_gemm_1d2d_impl<K, 0, 4096, 4096, 8, 128, 192, 128,
// 128, 128, 128, 4, 128, 256, 2, false, 132, MGroupedContiguous, bf16>.
// A persistent grouped GEMM: a TMA warpgroup streams A, its scales and B
// (multicast to both blocks of the cluster) through 4 stages of full/empty
// mbarriers; the math warpgroups run e4m3 wgmmas per 128-wide k-block and
// promote each block's fp32 result with its A and B scales in software, then
// store bf16 D through stmatrix and three swizzled TMA stores per tile.
//
// Departures:
// - Split N: each H100 math warpgroup (m64n192k32) becomes three warpgroups,
//   each an m64n64k32 on one n-third of the tile, because the tensor unit's
//   fp32 accumulator is at most 64 columns wide. 6 math warpgroups: block
//   384 -> 896 threads, kNumMathThreads 256 -> 768, kNumWGMMAStoreThreads
//   256 -> 768, empty-barrier count 16 -> 48, named barriers 8 and 9 at 768
//   threads; 32 accumulators per thread instead of 96.
// - Index remap for 6 warpgroups: warpgroup g = warp / 4 takes m-half
//   mh = g & 1 (math_wg_idx: A descriptor, is_computation_valid) and n-third
//   nt = g >> 1 (B descriptor + 64 rows, promotion group 8 nt + i, D atom);
//   its warp stands for H100 warp hw = 4 mh + warp % 4 (rows r_0/r_1, the
//   stmatrix warp offset). Threads 0-255 keep the H100 math threads' roles.
// - The k-loop of each num_former_iters dispatch instance is a __noinline__
//   function (a lambda on the H100): accumulators pinned to f0-f31 next to
//   the TMA asm crashed clang's register allocator in an earlier port, and
//   the kernel entry stays under the size vxbin.py accepts. The empty
//   arrive's lane test reads __laneid() inside it (the H100 lambda captures
//   the kernel's lane_idx): a lane id passed in would not be seen as
//   divergent and every lane would arrive. Its parameters are warp-uniform.
// - CUTLASS, CuTe and DeepGEMM helpers are app-local code (dg_cutlass.h,
//   dg_scheduler.h); wgmmas take descriptors made by h100::make_desc from
//   the CTA-relative address (__cvta_generic_to_shared gives the cluster
//   window address here); DeepGEMM's low-word arithmetic is kept.
// - scale_d = 0 (the first k-step) clears the accumulators in software.
// - No __shared__: the dynamic local memory at the H100 offsets, origin 0
//   (1024-aligned like the H100's 0x400).
// - One argument block, tensor maps first; templated export. The entry
//   checks the launch gave a cluster of 2 (the kernel has no __cluster_dims__).
// - setmaxnreg and cudaGridDependencySynchronize are no-ops; the TMA cache
//   hint is dropped.
// - Inside the k-loop, the do_wgmma_store guards (on the A-scale reads and
//   the promotion skip) and the one-pass loop over WAVE_BLOCK_M (m_offset 0)
//   are dropped: with BLOCK_M 128 >= 64 the guards are always true and
//   BLOCK_M / WAVE_BLOCK_M is 1, as on the H100. The epilogue keeps its guard.
// - Problem shrunk (build-time N, K, G, SM count; rows per group at run
//   time): N 4096 -> DG_SHAPE_N, K 4096 -> DG_SHAPE_K, 8 -> DG_NUM_GROUPS
//   groups, 132 -> DG_NUM_SMS SMs (so 8 blocks per swizzle group, not 16, and
//   64 bytes of B scales in shared memory, not 256). Block, tile, box, stage
//   and cluster constants are unchanged.
// - Schedule shape: the m-blocks form one truncated swizzle group (4 or 6 of
//   8) instead of full groups of 16. With 384 rows per group the pair of
//   m-blocks (2, 3) spans expert groups 0 and 1, so those tiles take the
//   multicast-off branch, which the H100's 4096 rows per group never take.
// - Inherited tensor-unit gaps: the e4m3 products are summed in the tensor
//   unit's fp32 window adder, not at the H100's fp8 accumulation precision;
//   warpgroups with different B descriptors take the shared B buffer in
//   turn (slower, same results).
// - Ported as written but not run by any case: the dead-peer path
//   (is_peer_cta_alive false needs odd m- and n-block counts) and
//   is_computation_valid false (needs rows mapped to group -1).
// - Instrumentation: thread 0 records the block's core; the producer records
//   each tile it schedules (iteration, m block, n block, multicast).

#include <vx_cuda.h>
#include "common.h"
#include "dg_cutlass.h"
#include "dg_scheduler.h"

namespace h = vortex::h100;
namespace vt = vortex::tensor;

namespace dg {

using Barrier = ClusterTransactionBarrier;

// One Vortex warpgroup's wgmma: e4m3 x e4m3 -> f32, m64 n64 k32.
constexpr uint32_t kWgmmaM = 64, kWgmmaN = 64, kWgmmaK = 32;
constexpr uint32_t kNumAccum = kWgmmaM * kWgmmaN / 128;
using Accum = h::acc<kWgmmaN, vt::fp32>;
static_assert(Accum::NRC == kNumAccum, "32 fp32 accumulators per thread");

constexpr bool kMustUseUniformedScaleB = (kBlockK % kBlockN == 0);
constexpr uint32_t SMEM_D_SIZE = align_up(kBlockM * kBlockN * 2, 1024u);
constexpr uint32_t SMEM_A_SIZE_PER_STAGE = kBlockM * kBlockK;
constexpr uint32_t SMEM_B_SIZE_PER_STAGE = kBlockN * kBlockK;
constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = kBlockM * 4;
constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = align_up(SMEM_SFA_SIZE_PER_STAGE, 128u);
constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
constexpr uint32_t kShapeKScales = ceil_div(kShapeK, kBlockK);
constexpr uint32_t kNumTotalKBlocks = ceil_div(kShapeK, kBlockK);
constexpr uint32_t SMEM_SFB_SIZE = align_up(kShapeKScales * (kMustUseUniformedScaleB ? 1 : 2) * 4, (uint32_t)sizeof(Barrier));
constexpr uint32_t SMEM_BARRIER_OFFSET = SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE;
static_assert(SMEM_D_SIZE % 1024 == 0, "A and B stay 1024-aligned for the 128B swizzle");
static_assert(SMEM_BARRIER_OFFSET + 2 * kNumStages * sizeof(Barrier) <= kSmemSize, "the layout fits the request");

__forceinline__ uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

__forceinline__ uint32_t bf16x2_bits(float lo, float hi) {
  return __builtin_bit_cast(uint32_t, __float22bfloat162_rn(make_float2(lo, hi)));
}

// What one math warp's k-loop needs; every field is the same in all lanes.
struct MathArgs {
  Barrier* full_barriers;
  Barrier* empty_barriers;
  const float* smem_sfa0;
  const float* smem_sfb;
  uint32_t a_desc_hi, a_desc_lo, b_desc_hi, b_desc_lo;
  uint32_t hw, nt;
  bool is_peer_cta_alive;
};

// The k-blocks of one valid tile for a known split of the tile's columns
// between the two B-scale rows (kNumFormerIters 8-column groups take row 0).
template <uint32_t kNumFormerIters>
__noinline__ void math_k_blocks(const MathArgs& c, uint32_t& stage_io, uint32_t& phase_io, float* final_accum) {
  uint32_t stage_idx = stage_io, phase = phase_io;
  auto advance_pipeline = [&](uint32_t& k_block_idx) {
    ++k_block_idx;
    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;
  };

  // Per-lane values come from the lane id read here, not from parameters.
  const uint32_t lane_idx = get_lane_idx();
  const uint32_t r_0 = c.hw * 16 + lane_idx / 4, r_1 = r_0 + 8;

  Accum accum;
#pragma unroll 8
  for (uint32_t k_block_idx = 0; k_block_idx < kNumTotalKBlocks; advance_pipeline(k_block_idx)) {
    const uint32_t a_desc_base_lo = c.a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
    const uint32_t b_desc_base_lo = c.b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

    // Both B-scale rows, even where the tile needs one, as the original.
    const float scale_b_0 = ld_shared(c.smem_sfb + k_block_idx);
    const float scale_b_1 = ld_shared(c.smem_sfb + k_block_idx + kShapeKScales);

    c.full_barriers[stage_idx].wait(phase);

    // The stage's A scales are read before the wgmmas: once this warp
    // releases the stage, the producer may refill it.
    const float* smem_sfa = c.smem_sfa0 + stage_idx * (ALIGNED_SMEM_SFA_SIZE_PER_STAGE / 4);
    const float scale_a_0 = ld_shared(smem_sfa + r_0);
    const float scale_a_1 = ld_shared(smem_sfa + r_1);

    warpgroup_fence_operand(accum);
    warpgroup_arrive();
    vt::detail::unroll_for<kBlockK / kWgmmaK>([&](auto k) __attribute__((always_inline)) {
      const uint64_t a_desc = ((uint64_t)c.a_desc_hi << 32) | (uint32_t)(a_desc_base_lo + (uint32_t)k * kWgmmaK / 16);
      const uint64_t b_desc = ((uint64_t)c.b_desc_hi << 32) | (uint32_t)(b_desc_base_lo + (uint32_t)k * kWgmmaK / 16);
      h::wgmma<kWgmmaN, vt::fp32, vt::fp8, vt::fp8>(a_desc, b_desc, accum, (uint32_t)k != 0);
    });
    warpgroup_commit_batch();
    warpgroup_fence_operand(accum);
    warpgroup_wait<0>();

    // Lane l releases the stage of block l of the cluster (its own block
    // through the window; both lanes to its own block if the peer is idle).
    const uint32_t target_cta = c.is_peer_cta_alive ? lane_idx : block_rank_in_cluster();
    if (lane_idx < kNumTMAMulticast)
      c.empty_barriers[stage_idx].arrive(target_cta);

    // Promotion: the column group's B-scale row is a constant of the instance
    // and the warpgroup's n-third.
    const float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
    const float scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;
    vt::detail::unroll_for<kNumAccum / 4>([&](auto ii) __attribute__((always_inline)) {
      constexpr uint32_t i = decltype(ii)::value;
      const bool predicate = kMustUseUniformedScaleB || (8 * c.nt + i) < kNumFormerIters;
      final_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum.data[i * 4 + 0];
      final_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum.data[i * 4 + 1];
      final_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum.data[i * 4 + 2];
      final_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum.data[i * 4 + 3];
    });
  }
  stage_io = stage_idx;
  phase_io = phase;
}

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd>
__forceinline__ void dispatch_num_former_iters(uint32_t num_former_iters, const MathArgs& c, uint32_t& stage_idx,
                                               uint32_t& phase, float* final_accum) {
  if (num_former_iters == kNumFormerIters) {
    math_k_blocks<kNumFormerIters>(c, stage_idx, phase, final_accum);
    return;
  }
  if constexpr (kNumFormerIters + kGap <= kEnd)
    dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, c, stage_idx, phase, final_accum);
}

constexpr uint32_t constexpr_gcd(uint32_t a, uint32_t b) { return b == 0 ? a : constexpr_gcd(b, a % b); }

template <Major kMajorSFB, uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t kNumGroups_,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kSwizzleAMode_, uint32_t kSwizzleBMode_,
          uint32_t kSwizzleDMode_, uint32_t kNumStages_, uint32_t kNumTMAThreads_, uint32_t kNumMathThreads_,
          uint32_t kNumTMAMulticast_, bool kIsTMAMulticastOnA_, uint32_t kNumSMs_, GemmType kGemmType,
          typename cd_dtype_t>
__device__ void sm90_fp8_gemm_1d2d_impl(kernel_arg_t* arg) {
  VX_CUDA_KERNEL_ENTRY(dg_launch);

  // The helpers above use the configuration of common.h.
  static_assert(BLOCK_M == kBlockM && BLOCK_N == kBlockN && BLOCK_K == kBlockK && kNumStages_ == kNumStages &&
                kNumTMAThreads_ == kNumTMAThreads && kNumMathThreads_ == kNumMathThreads &&
                kNumTMAMulticast_ == kNumTMAMulticast && kIsTMAMulticastOnA_ == kIsTMAMulticastOnA &&
                kSwizzleAMode_ == kSwizzleAMode && kSwizzleBMode_ == kSwizzleBMode &&
                kSwizzleDMode_ == kSwizzleDMode && kNumGroups_ == kNumGroups && kNumSMs_ == kNumSMs &&
                SHAPE_K == kShapeK, "one configuration");
  static_assert(BLOCK_K == 128, "per-128-channel FP8 scaling");
  static_assert(ceil_div(BLOCK_N, BLOCK_K) == 1 || constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K,
                "at most two B scales per block");
  static_assert(std::is_same_v<cd_dtype_t, __nv_bfloat16>, "bf16 D without accumulation");
  static_assert(BLOCK_M % kWgmmaM == 0 || BLOCK_M < kWgmmaM, "block of whole wgmmas");
  static_assert(BLOCK_N % kWgmmaN == 0, "the n-thirds tile BLOCK_N");
  static_assert(kReservedNamedBarrierCount + 1 < VX_CFG_NUM_BARRIERS, "named barriers 8 and 9");
  static_assert(kNumTMAMulticast_ <= 32, "multicast within the cluster");

  const uint32_t shape_m = SHAPE_M != 0 ? SHAPE_M : arg->shape_m;
  const uint32_t shape_n = SHAPE_N != 0 ? SHAPE_N : arg->shape_n;
  const uint32_t shape_k = SHAPE_K != 0 ? SHAPE_K : arg->shape_k;

  const uint32_t shape_k_scales = ceil_div(shape_k, BLOCK_K);
  const uint32_t shape_n_sfb = ceil_div(shape_n, BLOCK_K);
  const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
  const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const uint32_t lane_idx = get_lane_idx();

  if (threadIdx.x == 0)
    ((uint32_t*)arg->smid)[blockIdx.x] = __smid();

  if (warp_idx == kNumMathThreads_ / 32 && cute::elect_one_sync()) {
    prefetch_tma_descriptor(&arg->tensor_map_a);
    prefetch_tma_descriptor(&arg->tensor_map_b);
    prefetch_tma_descriptor(&arg->tensor_map_sfa);
    prefetch_tma_descriptor(&arg->tensor_map_d);
  }
  __syncwarp();

  char* smem_buffer = (char*)__local_mem();
  __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
  auto smem_a = [&](uint32_t i) { return smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE; };
  auto smem_b = [&](uint32_t i) {
    return smem_buffer + SMEM_D_SIZE + kNumStages_ * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE;
  };
  auto smem_sfa = [&](uint32_t i) {
    return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
  };
  float* smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages_ * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

  Barrier* full_barriers = reinterpret_cast<Barrier*>(reinterpret_cast<char*>(smem_sfb) + SMEM_SFB_SIZE);
  Barrier* empty_barriers = full_barriers + kNumStages_;

  // One elected thread initialises the barriers; each lane l of a math warp
  // later arrives on block l's empty barrier, so 2 arrivals per math warp of
  // each block reach every block's empty barrier.
  if (warp_idx == kNumMathThreads_ / 32 + 1 && cute::elect_one_sync()) {
    for (uint32_t i = 0; i < kNumStages_; ++i) {
      full_barriers[i].init(1);
      empty_barriers[i].init(kNumTMAMulticast_ * kNumMathThreads_ / 32);
    }
    fence_barrier_init();
  }

  if (kNumTMAMulticast_ > 1)
    cluster_sync_with_relaxed_arrive();
  else
    __syncthreads();

  constexpr uint32_t kNumTMARegisters = 40;
  constexpr uint32_t kNumMathRegisters = 232;

  cudaGridDependencySynchronize();

  uint32_t m_block_idx, n_block_idx;
  auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups_, kNumTMAMulticast_, kIsTMAMulticastOnA_,
                             kNumSMs_>(shape_m, shape_n, (const int*)arg->grouped_layout);

  uint32_t stage_idx = 0, phase = 0;
  auto advance_pipeline = [&](uint32_t& k_block_idx) {
    ++k_block_idx;
    stage_idx = stage_idx == kNumStages_ - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;
  };

  if (warp_idx >= kNumMathThreads_ / 32) {
    // TMA warpgroup: one thread of its third warp produces.
    warpgroup_reg_dealloc<kNumTMARegisters>();
    if (warp_idx == kNumMathThreads_ / 32 + 2 && cute::elect_one_sync()) {
      trace_rec_t* trace = (trace_rec_t*)arg->trace + (size_t)blockIdx.x * kTraceCap;
      while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
        const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA_ && is_tma_multicast_valid) ? kNumTMAMulticast_ : 1;
        const uint32_t num_tma_multicast_b = (!kIsTMAMulticastOnA_ && is_tma_multicast_valid) ? kNumTMAMulticast_ : 1;
        static_assert(kNumTMAMulticast_ <= 2, "the scheduler pairs at most 2 blocks");

        if ((uint32_t)scheduler.current_iter < kTraceCap) {
          trace_rec_t* r = trace + scheduler.current_iter;
          r->iter = (uint32_t)scheduler.current_iter;
          r->m_block = m_block_idx;
          r->n_block = n_block_idx;
          r->mcast = is_tma_multicast_valid ? 1u : 0u;
        }

        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
          empty_barriers[stage_idx].wait(phase ^ 1);

          Barrier& full_barrier = full_barriers[stage_idx];
          const uint32_t k_idx = k_block_idx * BLOCK_K;
          tma_copy_2d(&arg->tensor_map_a, &full_barrier, smem_a(stage_idx), k_idx,
                      scheduler.template get_global_idx<false>(shape_m, BLOCK_M, m_block_idx), num_tma_multicast_a);
          tma_copy_2d(&arg->tensor_map_sfa, &full_barrier, smem_sfa(stage_idx), m_block_idx * BLOCK_M,
                      scheduler.template get_global_idx<false>(shape_k_scales, 1, k_block_idx), num_tma_multicast_a);
          tma_copy_2d(&arg->tensor_map_b, &full_barrier, smem_b(stage_idx), k_idx,
                      scheduler.template get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                      num_tma_multicast_b);
          full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE);
        }
      }

      // The peer still arrives on this block's empty barriers: wait for the
      // last phase of every stage before the block may exit.
      if constexpr (kNumTMAMulticast_ > 1) {
        for (uint32_t i = 0; i < kNumStages_; advance_pipeline(i))
          empty_barriers[stage_idx].wait(phase ^ 1);
      }
    }
  } else {
    warpgroup_reg_alloc<kNumMathRegisters>();

    // The Vortex warpgroup (0-5): m-half and n-third; this warp's H100 index.
    const uint32_t math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const uint32_t mh = math_wg_idx & 1, nt = math_wg_idx >> 1;
    const uint32_t hw = 4 * mh + warp_idx % 4;

    const uint64_t a_desc = h::make_desc(smem_a(0) + mh * kWgmmaM * BLOCK_K, h::swz::b128, 0, 1024);
    const uint64_t b_desc = h::make_desc(smem_b(0) + nt * kWgmmaN * BLOCK_K, h::swz::b128, 0, 1024);
    const uint32_t a_desc_lo = __shfl_sync(0xffffffff, (uint32_t)a_desc, 0);
    const uint32_t b_desc_lo = __shfl_sync(0xffffffff, (uint32_t)b_desc, 0);

    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      __vx_cuda_trap_if_any(shape_n % 8 != 0);
      uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
      if constexpr (!kMustUseUniformedScaleB) {
        num_former_iters = umin(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
        num_full_iters = umin(shape_n - n_block_idx * BLOCK_N, BLOCK_N) / 8;
      }
      const uint32_t num_sfb = shape_k_scales * (num_former_iters >= num_full_iters ? 1 : 2);

      // Threads past the first warp load the tile's B scales (the first warp
      // may still be waiting on the previous tile's stores).
      if (threadIdx.x >= 32) {
        const float* sfb = (const float*)arg->sfb;
        const uint32_t previous_group_offset =
            scheduler.template get_global_idx<true>(shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
        const uint32_t stride_n_sfb = kMajorSFB == Major::MN ? 1 : shape_k_scales;
        const uint32_t stride_k_sfb = kMajorSFB == Major::MN ? shape_n_sfb : 1;
        const float* local_sfb = sfb + previous_group_offset + ((n_block_idx * BLOCK_N) / BLOCK_K) * stride_n_sfb;
        for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads_ - 32)
          st_shared(smem_sfb + i, i < shape_k_scales ? local_sfb[i * stride_k_sfb]
                                                     : local_sfb[(i - shape_k_scales) * stride_k_sfb + stride_n_sfb]);
      }
      NamedBarrier::sync(kNumMathThreads_, 0);

      constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= kWgmmaM ? BLOCK_M : kWgmmaM * 2;
      static_assert(BLOCK_M % WAVE_BLOCK_M == 0 && BLOCK_M / WAVE_BLOCK_M == 1, "one wave of m-halves");
      float final_accum[kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

      // Every math warp holds part of D: the H100 count times the n-thirds.
      constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / kWgmmaM) * (BLOCK_N / kWgmmaN);
      static_assert(kNumWGMMAStoreThreads == kNumMathThreads_, "all math threads store");
      const bool do_wgmma_store = BLOCK_M >= kWgmmaM || warp_idx < kNumWGMMAStoreThreads / 32;

      if (scheduler.is_computation_valid(m_block_idx, mh * kWgmmaM)) {
        constexpr bool kShouldOptimize = BLOCK_K / constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 && !kMustUseUniformedScaleB;
        constexpr uint32_t kGap = constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
        constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;
        const MathArgs c = {full_barriers, empty_barriers, smem_sfa(0), smem_sfb,
                            (uint32_t)(a_desc >> 32), a_desc_lo, (uint32_t)(b_desc >> 32), b_desc_lo,
                            hw, nt, scheduler.is_peer_cta_alive};
        dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, c, stage_idx, phase,
                                                 final_accum);
      } else {
        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
          full_barriers[stage_idx].wait(phase);
          const uint32_t target_cta = scheduler.is_peer_cta_alive ? lane_idx : block_rank_in_cluster();
          if (lane_idx < kNumTMAMulticast_)
            empty_barriers[stage_idx].arrive(target_cta);
        }
      }

      constexpr uint32_t kNumElemBytes = 2;
      constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode_ == 0 ? BLOCK_N : (kSwizzleDMode_ / kNumElemBytes);
      constexpr uint32_t WGMMA_M_PER_WARP = kWgmmaM / 4;
      static_assert(BLOCK_M % 8 == 0, "whole swizzle atoms");
      static_assert(BLOCK_N % TMA_D_BLOCK_N == 0 && BLOCK_N / TMA_D_BLOCK_N <= 32, "whole TMA store boxes");
      static_assert(TMA_D_BLOCK_N % 8 == 0 && kSwizzleDMode_ / 16 == 8, "128B swizzle of 16-byte chunks");

      if (!do_wgmma_store)
        continue;

      // Shared D is reused only once the previous tile's stores have read it.
      if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
        tma_store_wait_0();
      NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

      // stmatrix x2 per 8-column group: this warpgroup's groups are the
      // H100's 8 nt .. 8 nt + 7, in swizzle atom nt.
      constexpr uint32_t m_offset = 0;
      for (uint32_t i = 0; i < kNumAccum / 4; ++i) {
        constexpr uint32_t kNumBankGroupBytes = 16;
        const uint32_t gi = 8 * nt + i;
        const uint32_t atom_offset = gi / (TMA_D_BLOCK_N / 8), in_atom_offset = gi % (TMA_D_BLOCK_N / 8);
        const uint32_t row = in_atom_offset / 8 + lane_idx;
        uint32_t col = in_atom_offset;
        col ^= row % (kSwizzleDMode_ / 16);
        uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(smem_d) + hw * (WGMMA_M_PER_WARP * kSwizzleDMode_) +
                            m_offset * kSwizzleDMode_ + atom_offset * BLOCK_M * kSwizzleDMode_ +
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
        stmatrix_x2(smem_ptr, bf16x2_bits(final_accum[i * 4 + 0], final_accum[i * 4 + 1]),
                    bf16x2_bits(final_accum[i * 4 + 2], final_accum[i * 4 + 3]));
      }
      tma_store_fence();
      NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

      // Threads 0-2 store the three 64-column boxes of the tile.
      if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
        const uint32_t in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
        const __nv_bfloat16* smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
        const uint32_t n_idx = n_block_idx * BLOCK_N + in_block_n_offset;
        const uint32_t m_idx = scheduler.template get_global_idx<false>(shape_m, BLOCK_M, m_block_idx);
        tma_store_2d(&arg->tensor_map_d, smem_ptr, n_idx, m_idx);
        tma_store_arrive();
      }
      __syncwarp();
    }
  }
}

} // namespace dg

VX_CUDA_KERNEL_EXPORT(sm90_fp8_gemm_1d2d,
                      dg::sm90_fp8_gemm_1d2d_impl<dg::Major::K, 0, dg::kShapeN, dg::kShapeK, dg::kNumGroups,
                                                  128, 192, 128, 128, 128, 128, 4, 128, 768, 2, false,
                                                  dg::kNumSMs, dg::GemmType::MGroupedContiguous, __nv_bfloat16>)
