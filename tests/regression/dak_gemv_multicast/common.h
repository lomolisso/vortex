#ifndef _DAK_GEMV_MULTICAST_COMMON_H_
#define _DAK_GEMV_MULTICAST_COMMON_H_

// Tile constants, the local-memory layout and the argument block of DAK's
// multicast GEMV kernel, shared by the host and the kernel.

#include <stdint.h>
#include <dxa_tmap.h>

namespace dak {

// The H100 build's template arguments (include/task/config.cuh).
constexpr int TILE_M = 64;
constexpr int TILE_N = 256;
constexpr int TILE_K = 64;
constexpr int BUFFER_SLOTS = 3;
constexpr int BUFFER_SLOTS_B = 3;
constexpr int CHUNK_SIZE_A = TILE_M * TILE_K;
constexpr int CHUNK_SIZE_B = TILE_N * TILE_K;
constexpr int CHUNK_SIZE_C = TILE_M * TILE_N;
constexpr int CLUSTER_SIZE = 2;

// One H100 consumer warpgroup of m64n256 is two Vortex warpgroups of
// m64n128 (N split in halves): a consumer group has 256 threads.
constexpr int NUM_CONSUMER_THREADS = 256;
constexpr int NUM_PRODUCER_THREADS = 32;
constexpr int BLOCK_THREADS = NUM_PRODUCER_THREADS * 2 + 2 * NUM_CONSUMER_THREADS;
constexpr int WG_THREADS = 128;
constexpr int WGS_PER_GROUP = NUM_CONSUMER_THREADS / WG_THREADS;
static_assert(NUM_CONSUMER_THREADS % WG_THREADS == 0, "a consumer group is whole warpgroups");

// Local memory, CTA-relative. The static mbarriers first, in the original's
// declaration order, padded to the H100's 256-byte static block, so that the
// dynamic buffers start at 1024-byte phase 2, as at the H100's 0x500.
constexpr uint32_t LM_FILLED = 0x000;            // filled[6]
constexpr uint32_t LM_DRAINED = 0x030;           // drained[6]
constexpr uint32_t LM_FILLED_B = 0x060;          // filled_b[6]
constexpr uint32_t LM_DRAINED_B = 0x090;         // drained_b[3]
constexpr uint32_t LM_CLUSTER_DRAINED_A = 0x0A8; // cluster_drained_a[6]
constexpr uint32_t LM_STATIC = 0x100;
constexpr uint32_t LM_DYN = LM_STATIC;
// get_smem_size(multicast=True), app/python/utils.py.
constexpr uint32_t DYN_SMEM = (2 * TILE_M * TILE_K * BUFFER_SLOTS + TILE_N * TILE_K * BUFFER_SLOTS_B
                               + 2 * TILE_M * TILE_N) * 2 + 1024;
constexpr uint32_t LMEM_BYTES = LM_STATIC + DYN_SMEM;
static_assert(LM_CLUSTER_DRAINED_A + 6 * 8 <= LM_STATIC, "the static mbarriers fit their block");
static_assert(LMEM_BYTES == 214272, "the H100's request");

// The buffers, as the kernel carves them out of the dynamic part.
constexpr uint32_t LM_A = LM_DYN;
constexpr uint32_t LM_A_2 = LM_A + BUFFER_SLOTS * CHUNK_SIZE_A * 2;
constexpr uint32_t LM_B = LM_A_2 + BUFFER_SLOTS * CHUNK_SIZE_A * 2;
constexpr uint32_t LM_C = LM_B + BUFFER_SLOTS_B * CHUNK_SIZE_B * 2;
constexpr uint32_t LM_C_2 = LM_C + CHUNK_SIZE_C * 2;
static_assert(LM_A_2 == 0x6100 && LM_B == 0xC100 && LM_C == 0x24100 && LM_C_2 == 0x2C100, "the layout");

// The debug descriptor dump: per block, warpgroup (0..3 = warps 0-15 / 4),
// slot and k-step, the A and the B descriptor as the kernel builds them
// (before the MN-major bit).
constexpr uint32_t DESC_WGS = 2 * WGS_PER_GROUP;
constexpr uint32_t DESC_PER_CTA = DESC_WGS * BUFFER_SLOTS * (TILE_K / 16) * 2;
constexpr uint32_t desc_index(uint32_t wg, uint32_t slot, uint32_t kk, uint32_t is_b) {
  return ((wg * BUFFER_SLOTS + slot) * (TILE_K / 16) + kk) * 2 + is_b;
}

} // namespace dak

// The kernel's argument block: the tensor maps first (the argument buffer is
// 64-byte aligned, a map must be), then the scalars of the original's
// parameter list, then the placement records and the debug dump.
typedef struct {
  vx_tensor_map_t a_map;   // W [K][M]
  vx_tensor_map_t b_map;   // X [T][K]
  vx_tensor_map_t c_map;   // out [T][M]
  int32_t M, N, K;
  int32_t num_blocks, sms_per_row;
  uint32_t pad;
  uint64_t smid;           // one core id per block
  uint64_t wid0;           // the hardware warp of each block's thread 0
  uint64_t desc;           // 0, or DESC_PER_CTA descriptors per block
} __attribute__((aligned(64))) kernel_arg_t;

// The kernel is __cluster_dims__(CLUSTER_SIZE, 1, 1).
VX_CUDA_CLUSTER_DIMS(gemv_multicast, dak::CLUSTER_SIZE, 1, 1);

#endif
