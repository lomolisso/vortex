#ifndef _WGMMA_H100_COMMON_H_
#define _WGMMA_H100_COMMON_H_

// The argument block of the H100 wgmma test, shared by the host and the kernel.

#include <stdint.h>
#include <dxa_tmap.h>

// Type pairs (fmt_s -> fmt_d).
enum : uint32_t { PAIR_F16F32 = 0, PAIR_F16F16, PAIR_BF16F32, PAIR_E4M3F32, PAIR_E5M2F32, PAIR_COUNT };

constexpr uint32_t kMaxChain = 8;      // wgmmas chained along k
constexpr uint32_t kMaxGroups = 6;     // warpgroups per block
constexpr uint32_t kMaxCtas = 4;       // blocks
constexpr uint32_t kMaxBoxes = 512;    // tensor-load boxes of one block's fill
constexpr uint32_t kMaxRegs = 32;      // accumulator registers per thread

// Per-block output strides, in elements.
constexpr uint32_t kSlotsPerCta = kMaxGroups * kMaxRegs * 128;
constexpr uint32_t kDescsPerCta = kMaxGroups * kMaxChain * 2;

// Tensor maps: the A of warpgroup g at g, its B at kMaxGroups + g.
constexpr uint32_t MAP_COUNT = 2 * kMaxGroups;
constexpr uint32_t map_a(uint32_t g) { return g; }
constexpr uint32_t map_b(uint32_t g) { return kMaxGroups + g; }

// One tensor-load box: map, box coordinates (block 0), CTA-relative destination.
typedef struct {
  uint32_t map, c0, c1, dst;
} box_t;

// Where a block ran: its core, its local-memory slot (byte offset of the
// block's window from the core's local memory) and the hardware warp of its warp 0.
typedef struct {
  uint32_t core, slot, warp, pad;
} place_t;

// mbarriers of ring mode: full[2] at kRingMbar, empty[2] after them.
constexpr uint32_t kRingMbar = 0x40;

// Block b, warpgroup g, chained wgmma i. Operands sit at the same CTA-relative
// offsets in every block, with the block's own data. Outputs are per block:
// c_addr/frag_addr [b][g][r][t] (32-bit), d_addr [b][g][r][t] (the 64-bit
// register), desc_addr [b][g][i][{A, B}], place_addr [b].
typedef struct {
  vx_tensor_map_t map[MAP_COUNT];      // the blocks' tensors stacked along the outer dimension
  uint64_t img_addr;                   // software fill: per block, local memory [fill_lo, fill_hi)
  uint64_t box_addr;                   // TMA fill: box_t[nboxes]
  uint64_t c_addr;
  uint64_t d_addr;
  uint64_t desc_addr;
  uint64_t frag_addr;
  uint64_t place_addr;
  uint32_t pair, n, tnsp_a, tnsp_b;
  uint32_t groups, chain, zero_first, tma;
  uint32_t layout, lbo_a, sbo_a, lbo_b, sbo_b;
  uint32_t fill_lo, fill_hi, img_stride, nboxes, tx_bytes, mbar_off, distinct_a;
  uint32_t blk_outer[MAP_COUNT];       // outer rows of one block's tensor in each map
  // Ring mode: 2 stages, each an A and a B tile of one wgmma, refilled by a
  // producer warp (the block's warp 4) through full/empty mbarriers. Stage s's
  // boxes are box_t[ring_box_lo[s] .. + ring_box_n[s]); wgmma i's tile is the
  // box coordinates moved by i * ki along k (dimension kdim[map] of the box).
  uint32_t ring, ki, skew;
  uint32_t ring_box_lo[2], ring_box_n[2], ring_tx[2];
  uint32_t kdim[MAP_COUNT];
  uint32_t a_start[kMaxGroups][kMaxChain];
  uint32_t b_start[kMaxGroups][kMaxChain];
} __attribute__((aligned(64))) kernel_arg_t;

VX_CUDA_CLUSTER_DIMS(wgmma_h100_c2, 2, 1, 1);

#endif
