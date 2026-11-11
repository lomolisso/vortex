// Exact f32 top-2048 of one long row per 8-CTA cluster: the hpc-ops
// cluster-8 route.
//
// Re-implementation of topk_filtered_cluster8_exact_kernel<VEC, 2048, 8, 1024>
// and the helpers it calls, from hpc-ops 2a2e265 src/topk/topk_filtered.cu
// (+ topk_filtered_boundary.cuh, topk.h; the hpc-ops__pin build, no patch).
// Same block (1024 threads), cluster (8, launch attribute), RowSmem layout,
// passes, barriers and peers: eight CTAs build the exact coarse histogram of
// their row segments and all-reduce it over DSMEM (each rank owns 256 bins,
// reads them from every peer, sums with xor shuffles and writes the sum back
// to every peer); every rank finds the same boundary bin, rescans its segment
// and publishes its selected indices and boundary candidates to the primary
// through remote atomics on the primary's counters; rank 0 then settles the
// boundary by itself in 11/11/10-bit f32 rounds. Both vector paths: VEC 4 (rows of a multiple of
// 4 floats) and VEC 2 (rows of 2 mod 4 floats, the H100 profile n = 244650).
//
// Departures:
// - ld.global.cg.v4.f32 is __ldcg of a float4 (a plain load: Vortex has no
//   cache-global hint).
// - The kernel takes one argument struct; it also records each CTA's core id,
//   its count of candidates spilled to its spill1 range (VEC 2) and, from the
//   primary's thread 0, the route it took (harness checks).
// - No __shared__: RowSmem sits at offset 0 of the block's dynamic local
//   memory in every CTA.
// - Left out as dead code for this kernel: run_exact_coarse_stage,
//   to_coarse_key2, fp32_boundary_accepts, prefix_sum_find_threshold's
//   secondary-remaining search (only the -1 instantiation is reachable here)
//   and the sampled RowSmem variants.
// - Fixed upstream defect (a route the H100 run did not take): the VEC 2
//   overflow continuation built its round-0 fine histogram with the
//   primary counting min(cluster total, 2048) slots of its buffers, which
//   include the peers' copied candidates that the peers count again. The
//   doubled counts move the round-0 threshold and leave the result short
//   of k entries; the primary now counts only its own candidates.
// - Launched directly, not through choose_topk2048_plan: the host shrinks n
//   below the dispatcher's cluster-8 envelope (n >= 32768) and keeps the
//   kernel's template arguments. The spill workspace has 2 n ints for each of
//   the m rows (the kernel's indexing) instead of grid_num_blocks() rows.
// - Only the VEC 4 and VEC 2 instantiations are exported; the host refuses an
//   odd n (VEC 1, not ported).
// - warp_inclusive_sum casts threadIdx.x & 31 to int before comparing it with
//   the shuffle offset (-Wsign-compare under -Werror).
// - Built with -fconvergent-functions, as clang compiles CUDA device code:
//   without it a barrier was duplicated into both arms of a merged branch.

#include <vx_cuda.h>
#include "common.h"
#include "topk_filtered_boundary.h"

namespace cg = cooperative_groups;

namespace hpcops_topk {

// Shift and width of each refinement round over the ordered f32 bits:
// [31:21], [20:10], [9:0].
__device__ constexpr int kFineShifts[NUM_REFINE_ROUNDS] = {21, 10, 0};
__device__ constexpr uint32_t kFineMasks[NUM_REFINE_ROUNDS] = {0x7FFu, 0x7FFu, 0x3FFu};

// f32 bits mapped so that unsigned order is descending score order.
__device__ __forceinline__ uint32_t to_ordered(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? bits : ((bits ^ 0xFFFFFFFFu) & 0x7FFFFFFFu);
}

// The VEC 4 route keeps its boundary candidates as 64-bit (index, f32 bits)
// pairs in the candidate buffers.
__device__ __forceinline__ uint64_t pack_cluster_candidate(int32_t index, uint32_t key) {
  return static_cast<uint64_t>(static_cast<uint32_t>(index)) | (static_cast<uint64_t>(key) << 32);
}

__device__ __forceinline__ int32_t cluster_candidate_index(uint64_t pair) {
  return static_cast<int32_t>(static_cast<uint32_t>(pair));
}

__device__ __forceinline__ uint32_t cluster_candidate_key(uint64_t pair) { return static_cast<uint32_t>(pair >> 32); }

template <int kK, int kThreads>
__device__ __forceinline__ uint64_t* cluster_candidate_pairs(RowSmem<kK, false, kThreads>* s) {
  return reinterpret_cast<uint64_t*>(&s->cand_idx[0][0]);
}

__device__ __forceinline__ int warp_inclusive_sum(int value) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int upper = __shfl_up_sync(0xFFFFFFFFu, value, offset);
    if (static_cast<int>(threadIdx.x & 31) >= offset)
      value += upper;
  }
  return value;
}

// Turns the histogram into its inclusive prefix sum (in place) and stores
// the bin b with prefix[b - 1] <= remaining < prefix[b] in threshold_bin.
// Each thread sums a run of kItems bins; warps combine through the per-warp
// totals. With kResetState, thread 0 also clears the next round's candidate
// counts and sets *reset_target.
template <int kK, bool kSampled, int kThreads, bool kResetState = true>
__device__ __forceinline__ void prefix_sum_find_threshold(RowSmem<kK, kSampled, kThreads>* s, int tidx,
                                                          int remaining, int next_idx, int* reset_target,
                                                          int reset_value) {
  static_assert(kThreads >= 32 && kThreads <= 1024, "block size");
  static_assert((kThreads & 31) == 0, "whole warps");
  static_assert(RADIX % kThreads == 0, "bins split evenly");
  constexpr int kItems = RADIX / kThreads;
  constexpr int kNumWarps = kThreads / 32;
  int values[kItems];
  int local_sum = 0;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int value = s->histogram[tidx * kItems + item];
    values[item] = value;
    local_sum += value;
  }

  if constexpr (kResetState) {
    if (tidx == 0) {
      s->num_cand[next_idx] = 0;
      s->num_spilled_cand[next_idx] = 0;
      *reset_target = reset_value;
    }
  }
  const int lane = tidx & 31;
  const int warp = tidx >> 5;
  const int warp_inclusive = warp_inclusive_sum(local_sum);
  const int warp_exclusive = warp_inclusive - local_sum;
  if (lane == 31)
    s->prefix.warp_sum[warp] = warp_inclusive;
  __syncthreads();
  const int preceding_warp_value = lane < kNumWarps && lane < warp ? s->prefix.warp_sum[lane] : 0;
  const int preceding_warps = __reduce_add_sync(0xFFFFFFFFu, preceding_warp_value);

  int prefix = preceding_warps + warp_exclusive;
  int found_bin = -1;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int ibin = tidx * kItems + item;
    const int exclusive = prefix;
    prefix += values[item];
    s->histogram[ibin] = prefix;
    if (found_bin < 0 && exclusive <= remaining && prefix > remaining)
      found_bin = ibin;
  }
  if (found_bin >= 0)
    s->threshold_bin = found_bin;
  __syncthreads();
}

// Settles the boundary in f32: on entry the boundary candidates are in
// cand_idx[0] and (past its capacity) buf0, and histogram counts their
// first-round keys. A round takes every candidate in a bin above the
// threshold, passes the threshold bin's candidates on to the next round
// through the other buffer pair, and the last round fills the remaining slots
// with ties, leaving exactly kK indices.
template <int kK, bool kSampled, int kThreads>
__device__ __noinline__ void refine_rounds(RowSmem<kK, kSampled, kThreads>* s, const float* score, int32_t* buf0,
                                           int32_t* buf1, int topk_remaining, int tidx) {
  using Smem = RowSmem<kK, kSampled, kThreads>;
  bool run = true;
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    if (!run)
      break;
    const int r = round & 1;

    prefix_sum_find_threshold(s, tidx, topk_remaining, r ^ 1, &s->ties_remaining, topk_remaining);
    __syncthreads();

    const int num_input = min(s->num_cand[r], Smem::CANDIDATE_CAPACITY);
    const int g_num = s->num_spilled_cand[r];
    const int threshold = s->threshold_bin;
    if (threshold > 0)
      topk_remaining -= s->histogram[threshold - 1];
    // The ties left for the last round, now that the bins above are known.
    if (tidx == 0)
      s->ties_remaining = topk_remaining;
    __syncthreads();

    const int cur_shift = kFineShifts[round];
    const uint32_t cur_mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;
    const int next_shift = is_last ? 0 : kFineShifts[round + 1];
    const uint32_t next_mask = is_last ? 0 : kFineMasks[round + 1];
    int32_t* cur_buf = r ? buf1 : buf0;
    int32_t* next_buf = r ? buf0 : buf1;

    if (topk_remaining == 0) {
      for (int i = tidx; i < num_input; i += kThreads) {
        const int idx = s->cand_idx[r][i];
        const int b = (to_ordered(score[idx]) >> cur_shift) & cur_mask;
        if (b < threshold) {
          const int pos = atomicAdd(&s->num_selected, 1);
          s->indices[pos] = idx;
        }
      }
      for (int i = tidx; i < g_num; i += kThreads) {
        const int idx = cur_buf[i];
        const int b = (to_ordered(score[idx]) >> cur_shift) & cur_mask;
        if (b < threshold) {
          const int pos = atomicAdd(&s->num_selected, 1);
          s->indices[pos] = idx;
        }
      }
      __syncthreads();
      run = false;
    } else {
      for (int i = tidx; i < RADIX + 1; i += kThreads)
        s->histogram[i] = 0;
      __syncthreads();

      // One candidate: taken, kept for the next round, or discarded.
      auto refine_one = [&](int idx) {
        const float raw = score[idx];
        const uint32_t bits = to_ordered(raw);
        const int b = (bits >> cur_shift) & cur_mask;
        if (b < threshold) {
          const int pos = atomicAdd(&s->num_selected, 1);
          s->indices[pos] = idx;
        } else if (b == threshold) {
          if (is_last) {
            const int cur = atomicAdd(&s->ties_remaining, -1);
            if (cur > 0)
              s->indices[kK - cur] = idx;
          } else {
            const int cur = atomicAdd(&s->num_cand[r ^ 1], 1);
            if (cur < Smem::CANDIDATE_CAPACITY) {
              s->cand_idx[r ^ 1][cur] = idx;
            } else {
              const int gpos = atomicAdd(&s->num_spilled_cand[r ^ 1], 1);
              next_buf[gpos] = idx;
            }
            const int sub = (bits >> next_shift) & next_mask;
            atomicAdd(&s->histogram[sub], 1);
          }
        }
      };

      for (int i = tidx; i < num_input; i += kThreads)
        refine_one(s->cand_idx[r][i]);
      for (int i = tidx; i < g_num; i += kThreads)
        refine_one(cur_buf[i]);
      __syncthreads();
    }
  }
}

// The rounds over a resident boundary, from the ordered keys kept in
// cand_idx[1] beside the indices in cand_idx[0]: no score reloads and no
// candidate copies; a candidate stays live while its bits match the prefix
// of the thresholds found so far.
template <int kK, int kThreads>
__device__ __noinline__ void refine_resident_keys(RowSmem<kK, false, kThreads>* s, int topk_remaining, int tidx) {
  const int num_input = s->num_cand[0];
  uint32_t active_mask = 0;
  uint32_t active_prefix = 0;

#pragma unroll
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    prefix_sum_find_threshold<kK, false, kThreads, false>(s, tidx, topk_remaining, 0, &s->ties_remaining,
                                                         topk_remaining);
    const int threshold = s->threshold_bin;
    if (threshold > 0)
      topk_remaining -= s->histogram[threshold - 1];
    if (tidx == 0)
      s->ties_remaining = topk_remaining;
    __syncthreads();

    const int cur_shift = kFineShifts[round];
    const uint32_t cur_mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;
    if (topk_remaining != 0 && !is_last) {
      for (int bin = tidx; bin < RADIX + 1; bin += kThreads)
        s->histogram[bin] = 0;
      __syncthreads();
    }

    for (int pos = tidx; pos < num_input; pos += kThreads) {
      const uint32_t ordered = static_cast<uint32_t>(s->cand_idx[1][pos]);
      const int idx = s->cand_idx[0][pos];
      if ((ordered & active_mask) != active_prefix)
        continue;
      const int bin = (ordered >> cur_shift) & cur_mask;
      if (bin < threshold) {
        const int out = atomicAdd(&s->num_selected, 1);
        s->indices[out] = idx;
      } else if (bin == threshold && topk_remaining != 0) {
        if (is_last) {
          const int cur = atomicAdd(&s->ties_remaining, -1);
          if (cur > 0)
            s->indices[kK - cur] = idx;
        } else {
          const int next_bin = (ordered >> kFineShifts[round + 1]) & kFineMasks[round + 1];
          atomicAdd(&s->histogram[next_bin], 1);
        }
      }
    }
    __syncthreads();
    if (topk_remaining == 0 || is_last)
      break;
    active_mask |= cur_mask << cur_shift;
    active_prefix |= static_cast<uint32_t>(threshold) << cur_shift;
  }
}

// VEC 4 route: primary thread t owns boundary candidate t through all three
// rounds and writes its index straight to the output row.
template <int kK, int kThreads>
__device__ __noinline__ void refine_cluster_register_candidate(RowSmem<kK, false, kThreads>* s, uint64_t raw_pair,
                                                               bool active, int topk_remaining, int tidx,
                                                               int32_t* direct_output) {
  uint32_t ordered = 0;
  int idx = -1;
  int write_pos = kK;
  if (active) {
    idx = cluster_candidate_index(raw_pair);
    ordered = to_ordered(__uint_as_float(cluster_candidate_key(raw_pair)));
  }

  for (int bin = tidx; bin < RADIX + 1; bin += kThreads)
    s->histogram[bin] = 0;
  __syncthreads();
  if (active)
    atomicAdd(&s->histogram[(ordered >> kFineShifts[0]) & kFineMasks[0]], 1);
  __syncthreads();

#pragma unroll
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    prefix_sum_find_threshold<kK, false, kThreads, false>(s, tidx, topk_remaining, 0, &s->ties_remaining,
                                                         topk_remaining);
    const int threshold = s->threshold_bin;
    if (threshold > 0)
      topk_remaining -= s->histogram[threshold - 1];
    if (tidx == 0)
      s->ties_remaining = topk_remaining;
    __syncthreads();

    const int cur_shift = kFineShifts[round];
    const uint32_t cur_mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;
    if (topk_remaining != 0 && !is_last) {
      for (int bin = tidx; bin < RADIX + 1; bin += kThreads)
        s->histogram[bin] = 0;
      __syncthreads();
    }

    if (active) {
      const int bin = (ordered >> cur_shift) & cur_mask;
      if (bin < threshold) {
        write_pos = atomicAdd(&s->num_selected, 1);
        active = false;
      } else if (bin > threshold) {
        active = false;
      } else if (topk_remaining != 0) {
        if (is_last) {
          const int cur = atomicAdd(&s->ties_remaining, -1);
          if (cur > 0)
            write_pos = kK - cur;
          active = false;
        } else {
          atomicAdd(&s->histogram[(ordered >> kFineShifts[round + 1]) & kFineMasks[round + 1]], 1);
        }
      } else {
        active = false;
      }
    }
    __syncthreads();
    if (topk_remaining == 0 || is_last)
      break;
  }
  if (write_pos < kK)
    direct_output[write_pos] = idx;
}

// DSMEM all-reduce of the 2048-bin histogram in one round: rank r owns bins
// [256r, 256r + 256); eight consecutive lanes load one owned bin from the
// eight ranks, sum it with xor shuffles and store the sum back to each rank.
template <int kK, int kS, int kThreads>
__device__ __forceinline__ void cluster_allreduce_histogram(RowSmem<kK, false, kThreads>* s,
                                                            cg::cluster_group cluster, int rank, int tidx) {
  static_assert(kS == 8, "eight ranks");
  static_assert(kThreads == 1024, "1024 threads");
  static_assert(RADIX % kS == 0 && RADIX % kThreads == 0, "bins split evenly");
  constexpr int kBinsPerRank = RADIX / kS;
  constexpr int kItems = RADIX / kThreads;

  cluster.sync();
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int linear = tidx + item * kThreads;
    const int bin = rank * kBinsPerRank + linear / kS;
    const int peer = linear & (kS - 1);
    int* addr = cluster.map_shared_rank(&s->histogram[bin], peer);
    int value = *addr;
#pragma unroll
    for (int offset = 1; offset < kS; offset <<= 1)
      value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    *addr = value;
  }
  cluster.sync();
}

// VEC 4 route for a boundary bin larger than the block: the primary alone
// rescans the whole row and refines; the peers wait at the closing cluster
// barrier.
template <int kK, int kThreads>
__device__ __noinline__ void refine_large_boundary_single_cta(RowSmem<kK, false, kThreads>* s, const float* score,
                                                              int32_t* dst, int length,
                                                              PackedCoarseBoundary select_boundary,
                                                              PackedCoarseBoundary candidate_boundary,
                                                              int topk_remaining) {
  const int tidx = threadIdx.x;
  uint32_t active_mask = 0;
  uint32_t active_prefix = 0;

  for (int bin = tidx; bin < RADIX + 1; bin += kThreads)
    s->histogram[bin] = 0;
  if (tidx == 0)
    s->num_selected = 0;
  __syncthreads();
  for (int i = tidx; i < length; i += kThreads) {
    const float raw = score[i];
    const CoarseClass cls = coarse_classify(raw, select_boundary, candidate_boundary);
    if (cls == CoarseClass::kSelected) {
      const int pos = atomicAdd(&s->num_selected, 1);
      if (pos < kK)
        dst[pos] = i;
    } else if (cls == CoarseClass::kCandidate) {
      atomicAdd(&s->histogram[(to_ordered(raw) >> kFineShifts[0]) & kFineMasks[0]], 1);
    }
  }
  __syncthreads();

#pragma unroll 1
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    prefix_sum_find_threshold<kK, false, kThreads, false>(s, tidx, topk_remaining, 0, &s->ties_remaining,
                                                         topk_remaining);
    const int threshold = s->threshold_bin;
    if (threshold > 0)
      topk_remaining -= s->histogram[threshold - 1];
    const int shift = kFineShifts[round];
    const uint32_t mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;

    if (topk_remaining != 0 && !is_last) {
      for (int bin = tidx; bin < RADIX + 1; bin += kThreads)
        s->histogram[bin] = 0;
    }
    __syncthreads();

    // Only boundary-bucket scores whose bits match the prefix so far.
    for (int i = tidx; i < length; i += kThreads) {
      const float raw = score[i];
      if (coarse_classify(raw, select_boundary, candidate_boundary) != CoarseClass::kCandidate)
        continue;
      const uint32_t ordered = to_ordered(raw);
      if ((ordered & active_mask) != active_prefix)
        continue;
      const int bin = (ordered >> shift) & mask;
      if (bin < threshold) {
        const int pos = atomicAdd(&s->num_selected, 1);
        if (pos < kK)
          dst[pos] = i;
      } else if (bin == threshold && topk_remaining != 0) {
        if (is_last) {
          const int pos = atomicAdd(&s->num_selected, 1);
          if (pos < kK)
            dst[pos] = i;
        } else {
          atomicAdd(&s->histogram[(ordered >> kFineShifts[round + 1]) & kFineMasks[round + 1]], 1);
        }
      }
    }
    __syncthreads();

    if (topk_remaining == 0 || is_last)
      break;
    active_mask |= mask << shift;
    active_prefix |= static_cast<uint32_t>(threshold) << shift;
  }
}

// VEC 2 continuation when the cluster's boundary candidates overflow the
// primary's buffers: a fresh round-0 fine histogram is all-reduced over the
// cluster, the primary gathers every peer's selected indices and candidates
// (resident ones over DSMEM, spilled ones from the global spill rows) into
// its buffers and spill0, and alone runs refine_rounds.
template <int kK, int kS, int kThreads>
__device__ __noinline__ void run_cluster_unaligned_overflow(RowSmem<kK, false, kThreads>* s, const float* score,
                                                            int32_t* dst, int32_t* spill0, int32_t* spill1,
                                                            int length, int seg_lo, int seg_len,
                                                            int topk_remaining) {
  using Smem = RowSmem<kK, false, kThreads>;
  cg::cluster_group cluster = cg::this_cluster();
  const int tidx = threadIdx.x;
  const int rank = static_cast<int>(cluster.block_rank());

  for (int bin = tidx; bin < RADIX + 1; bin += kThreads)
    s->histogram[bin] = 0;
  __syncthreads();
  // The primary's num_cand[0] is by now the cluster total and its buffers
  // also hold the peers' copies, which the peers count themselves: the
  // primary counts only its own candidates (saved in ties_remaining).
  const int own_candidates = rank == 0 ? s->ties_remaining : s->num_cand[0];
  const int local_resident = min(own_candidates, Smem::CANDIDATE_CAPACITY);
  for (int pos = tidx; pos < local_resident; pos += kThreads) {
    const uint32_t ordered = static_cast<uint32_t>(s->cand_idx[1][pos]);
    atomicAdd(&s->histogram[(ordered >> kFineShifts[0]) & kFineMasks[0]], 1);
  }
  for (int pos = tidx; pos < s->num_spilled_cand[0]; pos += kThreads) {
    const int idx = spill1[seg_lo + pos];
    atomicAdd(&s->histogram[(to_ordered(score[idx]) >> kFineShifts[0]) & kFineMasks[0]], 1);
  }
  __syncthreads();
  cluster_allreduce_histogram<kK, kS, kThreads>(s, cluster, rank, tidx);

  if (rank == 0) {
    if (tidx == 0) {
      s->num_cand[1] = s->num_spilled_cand[0];
      s->num_selected = s->threshold_bin;
      s->num_cand[0] = min(s->ties_remaining, Smem::CANDIDATE_CAPACITY);
      s->num_spilled_cand[0] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int sp = 1; sp < kS; ++sp) {
      Smem* peer = cluster.map_shared_rank(s, sp);
      const int selected = peer->num_selected;
      for (int i = tidx; i < selected; i += kThreads) {
        const int pos = atomicAdd(&s->num_selected, 1);
        s->indices[pos] = peer->indices[i];
      }
      const int candidates = min(peer->num_cand[0], Smem::CANDIDATE_CAPACITY);
      for (int i = tidx; i < candidates; i += kThreads) {
        const int pos = atomicAdd(&s->num_cand[0], 1);
        const int idx = peer->cand_idx[0][i];
        if (pos < Smem::CANDIDATE_CAPACITY) {
          s->cand_idx[0][pos] = idx;
        } else {
          const int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
          spill0[gpos] = idx;
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int sp = 0; sp < kS; ++sp) {
      Smem* peer = cluster.map_shared_rank(s, sp);
      const int overflow = (sp == 0) ? s->num_cand[1] : peer->num_spilled_cand[0];
      const int source_lo = min(sp * seg_len, length);
      for (int i = tidx; i < overflow; i += kThreads) {
        const int pos = atomicAdd(&s->num_cand[0], 1);
        const int idx = spill1[source_lo + i];
        if (pos < Smem::CANDIDATE_CAPACITY) {
          s->cand_idx[0][pos] = idx;
        } else {
          const int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
          spill0[gpos] = idx;
        }
      }
      __syncthreads();
    }
  }
  cluster.sync();

  if (rank != 0)
    return;
  if (topk_remaining != 0)
    refine_rounds<kK, false, kThreads>(s, score, spill0, spill1, topk_remaining, tidx);
  for (int i = tidx; i < kK; i += kThreads)
    dst[i] = s->indices[i];
}

// Pass 1: every rank histograms the coarse keys of its segment; one DSMEM
// all-reduce gives every rank the row's histogram and the same boundary bin.
// Pass 2: every rank rescans its segment, keeping the selected indices and
// the boundary-bin candidates, and reserves ranges of the primary's buffers
// with remote atomics. The primary refines the boundary alone.
template <uint32_t VEC, int kK, int kS, int kThreads>
__device__ void __launch_bounds__(kThreads, 2) topk_filtered_cluster8_exact_kernel(kernel_arg_t* arg) {
  using Smem = RowSmem<kK, false, kThreads>;
  const float* __restrict__ input = (const float*)arg->input;
  int32_t* __restrict__ output = (int32_t*)arg->output;
  const int32_t* __restrict__ ke = (const int32_t*)arg->ke;
  int32_t* __restrict__ spill_buffer = (int32_t*)arg->spill_buffer;
  const int32_t* __restrict__ num_valid_rows = (const int32_t*)arg->num_valid_rows;
  uint32_t* __restrict__ info = (uint32_t*)arg->info;
  const uint32_t row_stride = arg->row_stride, out_stride = arg->out_stride, n_cols = arg->n_cols;

  cg::cluster_group cluster = cg::this_cluster();
  const int tidx = threadIdx.x;
  const int rank = static_cast<int>(cluster.block_rank());
  const int row_i = static_cast<int>(blockIdx.y);
  uint32_t* const my_info = info + ((size_t)row_i * kS + rank) * kInfoWords;
  if (tidx == 0)
    my_info[0] = __smid();
  if (row_i >= *num_valid_rows)
    return;

  Smem* s = reinterpret_cast<Smem*>(__local_mem());
  Smem* primary = cluster.map_shared_rank(s, 0);
  const float* score = input + static_cast<size_t>(row_i) * row_stride;
  int32_t* dst = output + static_cast<size_t>(row_i) * out_stride;
  const int length = ke[row_i];
  int32_t* spill0 = spill_buffer + static_cast<size_t>(row_i) * 2 * n_cols;
  int32_t* spill1 = spill0 + n_cols;

  if (length <= kK) {
    if (rank == 0) {
      if (tidx == 0)
        my_info[1] = kRouteShortRow;
      for (int i = tidx; i < kK; i += kThreads)
        dst[i] = (i < length) ? i : -1;
    }
    return;
  }

  const int seg_len = (length + kS - 1) / kS;
  const int seg_lo = min(rank * seg_len, length);
  const int seg_hi = min(seg_lo + seg_len, length);

  // Pass 1: the exact coarse histogram of this segment.
  for (int i = tidx; i < RADIX + 1; i += kThreads)
    s->histogram[i] = 0;
  __syncthreads();

  auto add_coarse = [&](float raw) { atomicAdd(&s->histogram[to_coarse_key(raw)], 1); };
  if constexpr (VEC == 4) {
    const int v_lo = min((seg_lo + 7) & ~7, seg_hi);
    const int v_hi = max(seg_hi & ~7, v_lo);
    for (int i = seg_lo + tidx; i < v_lo; i += kThreads)
      add_coarse(score[i]);
    const int n8 = (v_hi - v_lo) >> 3;
    for (int g = tidx; g < n8; g += kThreads) {
      const float* p = score + v_lo + (g << 3);
      const float4 a = __ldcg(reinterpret_cast<const float4*>(p));
      const float4 b = __ldcg(reinterpret_cast<const float4*>(p + 4));
      add_coarse(a.x);
      add_coarse(a.y);
      add_coarse(a.z);
      add_coarse(a.w);
      add_coarse(b.x);
      add_coarse(b.y);
      add_coarse(b.z);
      add_coarse(b.w);
    }
    for (int i = v_hi + tidx; i < seg_hi; i += kThreads)
      add_coarse(score[i]);
  } else {
    for (int i = seg_lo + tidx; i < seg_hi; i += kThreads)
      add_coarse(score[i]);
  }
  __syncthreads();

  cluster_allreduce_histogram<kK, kS, kThreads>(s, cluster, rank, tidx);
  // histogram[RADIX] stays zero from the clear (the all-reduce skips it).
  // The prefix pass also zeroes num_selected, so no extra cluster barrier.
  prefix_sum_find_threshold<kK, false, kThreads>(s, tidx, kK, 0, &s->num_selected, 0);
  if (tidx == 0) {
    const int local_threshold = s->threshold_bin;
    s->ties_remaining = kK - (local_threshold == 0 ? 0 : s->histogram[local_threshold - 1]);
  }
  // Every rank holds the same histogram and boundary; only thread 0's
  // remainder must reach the block before it is read.
  __syncthreads();

  const int threshold = s->threshold_bin;
  const int topk_remaining = s->ties_remaining;
  const int coarse_candidate_count = s->histogram[threshold] - (threshold == 0 ? 0 : s->histogram[threshold - 1]);
  const bool large_boundary = topk_remaining != 0 && coarse_candidate_count > kThreads;
  const PackedCoarseBoundary select_boundary = pack_coarse_boundary(coarse_boundary(threshold - 1));
  const PackedCoarseBoundary candidate_boundary = pack_coarse_boundary(coarse_boundary(threshold));

  if constexpr (VEC == 4) {
    if (large_boundary) {
      if (rank == 0) {
        if (tidx == 0)
          my_info[1] = kRouteLargeBoundary;
        refine_large_boundary_single_cta<kK, kThreads>(s, score, dst, length, select_boundary, candidate_boundary,
                                                       topk_remaining);
      }
      cluster.sync();
      return;
    }
  } else {
    (void)large_boundary;
  }

  // Pass 2: keep the selected indices and the boundary-bin candidates of this
  // segment (VEC 4: pairs of index and f32 bits; VEC 2: indices plus ordered
  // keys, spilling past the capacity to the segment's spill1 range).
  auto classify = [&](float raw, int i) {
    const CoarseClass cls = coarse_classify(raw, select_boundary, candidate_boundary);
    if (cls == CoarseClass::kSelected) {
      const int pos = atomicAdd(&s->num_selected, 1);
      if constexpr (VEC == 4) {
        if (rank == 0)
          dst[pos] = i;
        else
          s->indices[pos] = i;
      } else {
        s->indices[pos] = i;
      }
    } else if (cls == CoarseClass::kCandidate && topk_remaining != 0) {
      const int pos = atomicAdd(&s->num_cand[0], 1);
      if (pos < Smem::CANDIDATE_CAPACITY) {
        if constexpr (VEC == 4) {
          cluster_candidate_pairs(s)[pos] = pack_cluster_candidate(i, __float_as_uint(raw));
        } else {
          s->cand_idx[0][pos] = i;
          s->cand_idx[1][pos] = static_cast<int32_t>(to_ordered(raw));
        }
      } else {
        if constexpr (VEC != 4) {
          const int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
          spill1[seg_lo + gpos] = i;
        }
      }
    }
  };

  if constexpr (VEC == 4) {
    const int v_lo = min((seg_lo + 3) & ~3, seg_hi);
    const int v_hi = max(seg_hi & ~3, v_lo);
    for (int i = seg_lo + tidx; i < v_lo; i += kThreads)
      classify(score[i], i);
    const int n4 = (v_hi - v_lo) >> 2;
    int g = tidx;
    float4 next;
    if (g < n4)
      next = __ldcg(reinterpret_cast<const float4*>(score + v_lo + (g << 2)));
    while (g < n4) {
      const float4 current = next;
      const int base = v_lo + (g << 2);
      g += kThreads;
      if (g < n4)
        next = __ldcg(reinterpret_cast<const float4*>(score + v_lo + (g << 2)));
      classify(current.x, base);
      classify(current.y, base + 1);
      classify(current.z, base + 2);
      classify(current.w, base + 3);
    }
    for (int i = v_hi + tidx; i < seg_hi; i += kThreads)
      classify(score[i], i);
  } else {
    for (int i = seg_lo + tidx; i < seg_hi; i += kThreads)
      classify(score[i], i);
  }

  // VEC 2 stages selected indices through the primary's buffers, so the
  // primary snapshots its own counts (in the dead threshold_bin and
  // ties_remaining) before the peers add theirs.
  if constexpr (VEC != 4) {
    __syncthreads();
    if (tidx == 0)
      my_info[2] = static_cast<uint32_t>(s->num_spilled_cand[0]);
    if (tidx == 0 && rank == 0) {
      s->threshold_bin = s->num_selected;
      s->ties_remaining = s->num_cand[0];
    }
    cluster.sync();
  }

  // Peers reserve ranges of the primary's counters. With VEC 4 the primary
  // goes straight to the cluster barrier: its own and the peers' atomics
  // get disjoint ranges anyway.
  if constexpr (VEC == 4) {
    if (rank != 0)
      __syncthreads();
  }
  if (rank != 0 && tidx == 0) {
    s->threshold_bin = atomicAdd(&primary->num_selected, s->num_selected);
    s->ties_remaining = atomicAdd(&primary->num_cand[0], s->num_cand[0]);
    if constexpr (VEC == 4) {
      // The peer's candidate range (start, count), in dead histogram words of
      // the primary, so the primary reads each candidate from its owner.
      primary->histogram[rank] = s->ties_remaining;
      primary->histogram[kS + rank] = s->num_cand[0];
    }
  }
  if constexpr (VEC == 4) {
    if (rank != 0)
      __syncthreads();
  } else {
    __syncthreads();
  }

  if constexpr (VEC == 4) {
    cluster.sync();
    const int aggregate_candidates = primary->num_cand[0];
    if (rank == 0) {
      if (tidx == 0)
        my_info[1] = kRouteRegister;
      bool active = tidx < aggregate_candidates;
      uint64_t raw_pair = 0;
      if (active) {
        int owner = 0;
        int local_pos = tidx;
#pragma unroll
        for (int sp = 1; sp < kS; ++sp) {
          const int start = primary->histogram[sp];
          const int count = primary->histogram[kS + sp];
          if (tidx >= start && tidx < start + count) {
            owner = sp;
            local_pos = tidx - start;
          }
        }
        Smem* source = cluster.map_shared_rank(s, owner);
        raw_pair = cluster_candidate_pairs(source)[local_pos];
      }
      // The range words are read before the refinement clears the histogram.
      __syncthreads();
      if (topk_remaining != 0)
        refine_cluster_register_candidate<kK, kThreads>(s, raw_pair, active, topk_remaining, tidx, dst);
    } else {
      const int selected_prefix = s->threshold_bin;
      for (int i = tidx; i < s->num_selected; i += kThreads)
        dst[selected_prefix + i] = s->indices[i];
    }
    cluster.sync();
    return;
  } else {
    if (rank != 0) {
      const int candidate_prefix = s->ties_remaining;
      const int selected_prefix = s->threshold_bin;
      for (int i = tidx; i < s->num_selected; i += kThreads)
        primary->indices[selected_prefix + i] = s->indices[i];
      if (candidate_prefix < Smem::CANDIDATE_CAPACITY) {
        const int copy_candidates = min(s->num_cand[0], Smem::CANDIDATE_CAPACITY - candidate_prefix);
        for (int i = tidx; i < copy_candidates; i += kThreads) {
          primary->cand_idx[0][candidate_prefix + i] = s->cand_idx[0][i];
          primary->cand_idx[1][candidate_prefix + i] = s->cand_idx[1][i];
        }
      }
    }
    cluster.sync();

    // The aggregate count includes every rank's spilled candidates, so a
    // count within capacity also means nothing spilled.
    if (tidx == 0)
      s->num_cand[1] = primary->num_cand[0] <= Smem::CANDIDATE_CAPACITY ? 1 : 0;
    __syncthreads();
    const bool resident = s->num_cand[1] != 0;
    if (resident) {
      if (rank == 0) {
        if (tidx == 0)
          my_info[1] = kRouteResident;
        s->num_spilled_cand[0] = 0;
        if (topk_remaining != 0) {
          for (int bin = tidx; bin < RADIX + 1; bin += kThreads)
            s->histogram[bin] = 0;
          __syncthreads();
          for (int pos = tidx; pos < s->num_cand[0]; pos += kThreads) {
            const uint32_t ordered = static_cast<uint32_t>(s->cand_idx[1][pos]);
            atomicAdd(&s->histogram[(ordered >> kFineShifts[0]) & kFineMasks[0]], 1);
          }
          __syncthreads();
          refine_resident_keys(s, topk_remaining, tidx);
        }
        for (int i = tidx; i < kK; i += kThreads)
          dst[i] = s->indices[i];
      }
      // The peers' DSMEM must outlive the primary's reads.
      cluster.sync();
      return;
    }

    // Every rank must see the overflow decision before the counters change.
    cluster.sync();
    if (rank == 0 && tidx == 0)
      my_info[1] = kRouteOverflow;
    run_cluster_unaligned_overflow<kK, kS, kThreads>(s, score, dst, spill0, spill1, length, seg_lo, seg_len,
                                                     topk_remaining);
  }
}

} // namespace hpcops_topk

VX_CUDA_KERNEL_EXPORT(topk_filtered_cluster8_exact_kernel_v4,
                      hpcops_topk::topk_filtered_cluster8_exact_kernel<4, hpcops_topk::kTopK, hpcops_topk::kSplits,
                                                                       hpcops_topk::kBlockThreads>)
VX_CUDA_KERNEL_EXPORT(topk_filtered_cluster8_exact_kernel_v2,
                      hpcops_topk::topk_filtered_cluster8_exact_kernel<2, hpcops_topk::kTopK, hpcops_topk::kSplits,
                                                                       hpcops_topk::kBlockThreads>)
