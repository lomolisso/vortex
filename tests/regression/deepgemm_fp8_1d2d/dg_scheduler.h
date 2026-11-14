#ifndef _DEEPGEMM_FP8_1D2D_DG_SCHEDULER_H_
#define _DEEPGEMM_FP8_1D2D_DG_SCHEDULER_H_

// DeepGEMM's persistent block scheduler (scheduler/gemm.cuh) for the layouts
// the SM90 1D2D kernel is instantiated with here: the m-grouped contiguous
// layout, plus the dense one it shares the code path with. Device only.

#include <vx_cuda.h>
#include "common.h"

namespace dg {

enum class GemmType : uint32_t { Normal, MGroupedContiguous };
enum class Major : uint32_t { K, MN };

// The 1-D group width (8 or 16 blocks) that minimises the blocks one wave of
// kNumSMs CTAs touches; multicast on B groups along M.
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs, bool kIsMulticastOnA>
constexpr uint32_t get_num_1d_blocks_per_group() {
  uint32_t best = 0, min_usage = 0xffffffffu;
  for (const uint32_t candidate : {8u, 16u}) {
    const uint32_t usage = kIsMulticastOnA ? candidate * BLOCK_N + ceil_div(kNumSMs, candidate) * BLOCK_M
                                           : candidate * BLOCK_M + ceil_div(kNumSMs, candidate) * BLOCK_N;
    if (usage < min_usage) {
      min_usage = usage;
      best = candidate;
    }
  }
  return best;
}

template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups, uint32_t kNumMulticast,
          bool kIsMulticastOnA, uint32_t kNumSMs,
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>()>
struct Scheduler {
  static_assert(kGemmType == GemmType::Normal || kGemmType == GemmType::MGroupedContiguous, "layouts of this port");
  static_assert(kNum1DBlocksPerGroup % kNumMulticast == 0, "a group holds whole multicast pairs");

  int current_iter = -1;
  uint32_t num_blocks;
  uint32_t num_m_blocks;
  uint32_t num_n_blocks;
  uint32_t num_blocks_in_group;
  bool is_peer_cta_alive = true;
  const int* grouped_layout;

  __forceinline__ Scheduler(uint32_t shape_m, uint32_t shape_n, const int* layout) {
    num_m_blocks = ceil_div(shape_m, BLOCK_M);
    num_n_blocks = ceil_div(shape_n, BLOCK_N);
    num_blocks = num_m_blocks * num_n_blocks;
    grouped_layout = layout;
  }

  __forceinline__ void get_swizzled_block_idx(uint32_t block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
    const uint32_t primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
    const uint32_t secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
    const uint32_t num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
    const uint32_t group_idx = block_idx / num_blocks_per_group;
    uint32_t first_block_idx = group_idx * kNum1DBlocksPerGroup;
    uint32_t in_group_idx = block_idx % num_blocks_per_group;
    const uint32_t remaining = primary_num_blocks - first_block_idx;
    num_blocks_in_group = remaining < kNum1DBlocksPerGroup ? remaining : kNum1DBlocksPerGroup;

    // An odd group cannot be split into multicast pairs: its last block row
    // becomes a group of its own (where multicast is then off).
    if (kNumMulticast > 1 && num_blocks_in_group % 2 != 0) {
      if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
        num_blocks_in_group = num_blocks_in_group ^ 1;
      } else {
        in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
        first_block_idx += num_blocks_in_group ^ 1;
        num_blocks_in_group = 1;
      }
    }

    if constexpr (kIsMulticastOnA) {
      m_block_idx = in_group_idx / num_blocks_in_group;
      n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
    } else {
      m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
      n_block_idx = in_group_idx / num_blocks_in_group;
    }
  }

  // MN and SF_K indices share one formula for these layouts.
  template <bool kWithGroupOffset>
  __forceinline__ uint32_t get_global_idx(uint32_t shape_dim, uint32_t block_size, uint32_t block_idx,
                                          uint32_t m_block_idx = 0) {
    if constexpr (kGemmType == GemmType::Normal) {
      return block_idx * block_size;
    } else {
      const int g = kWithGroupOffset ? grouped_layout[m_block_idx * BLOCK_M] : 0;
      const uint32_t offset = g > 0 ? (uint32_t)g : 0u;
      return offset * shape_dim + block_idx * block_size;
    }
  }

  __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    const uint32_t next_block_idx = (uint32_t)(++current_iter) * kNumSMs + blockIdx.x;
    if (next_block_idx >= num_blocks)
      return false;
    is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 || num_m_blocks % kNumMulticast == 0
                     || (next_block_idx ^ 1) < num_blocks;
    get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
    return true;
  }

  __forceinline__ bool is_tma_multicast_valid(uint32_t m_block_idx) const {
    if (num_blocks_in_group == 1)
      return false;
    if constexpr (kGemmType == GemmType::Normal || kIsMulticastOnA) {
      return true;
    } else {
      return grouped_layout[m_block_idx * BLOCK_M] == grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
    }
  }

  __forceinline__ bool is_computation_valid(uint32_t m_block_idx, uint32_t m_offset) const {
    if constexpr (kGemmType == GemmType::Normal)
      return true;
    else
      return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
  }
};

} // namespace dg

#endif
