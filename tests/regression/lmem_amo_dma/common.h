#ifndef _LMEM_AMO_DMA_COMMON_H_
#define _LMEM_AMO_DMA_COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t num_copies;        // back-to-back DXA copies issued by warp 0
  uint32_t tile_elems;        // 32-bit elements per tile; one tile is region A
  uint32_t num_src_tiles;     // source tiles the copies rotate through
  uint32_t min_iters;         // atomic iterations every hammer hart runs at least
  uint64_t tile_out_addr;     // [grid][tile_elems]  final region A
  uint64_t hammer_out_addr;   // [grid][hammers]     hammer_out_t
  uint64_t lane_out_addr;     // [grid][threads]     lane_out_t
  uint64_t overlap_out_addr;  // [grid]              hammer iterations done when the last copy landed
} kernel_arg_t;

typedef struct {
  uint32_t add;      // private add counter
  uint32_t swap;     // private swap word
  uint32_t iters;    // iterations this hart ran
  uint32_t bad_old;  // atomics that returned an old value other than the last one written
} hammer_out_t;

// Words every hammer warp's lane l contends on.
typedef struct {
  uint32_t add;
  uint32_t xor_acc;
} lane_out_t;

static inline uint32_t xor_pattern(uint32_t hammer, uint32_t iter) {
  return ((iter + 1) * 0x9E3779B1u) ^ (hammer << 24);
}

#endif
