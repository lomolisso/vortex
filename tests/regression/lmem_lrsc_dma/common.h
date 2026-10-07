#ifndef _LMEM_LRSC_DMA_COMMON_H_
#define _LMEM_LRSC_DMA_COMMON_H_

#include <stdint.h>

#define MAX_LANES 4

// Local-memory offsets in the plan are relative to the CTA window rounded up
// to this, so a word's place in its 64-byte block is the same on both sides.
#define LMEM_ALIGN 64

typedef struct {
  uint32_t active;
  uint32_t word_off;   // byte offset of the reserved word
  uint32_t dword;      // 1: lr.d/sc.d on an 8-byte word, 0: lr.w/sc.w
  uint32_t re_lr;      // reserve the word again after the copy lands, before the SC
} lane_plan_t;

typedef struct {
  uint32_t dst_off;    // byte offset the DXA copy lands at
  uint32_t src_coord;  // first source element of the copy
  uint32_t probe_off;  // last word the copy writes
  uint32_t probe_val;  // its value once the copy has landed
  lane_plan_t lanes[MAX_LANES];
} round_plan_t;

typedef struct {
  uint64_t old;    // first LR
  uint64_t old2;   // re-LR
  uint64_t final;  // word after the SC
  uint32_t rc;     // SC result, 0 = success
  uint32_t spins;  // probe loads that still saw stale data after the barrier
  uint32_t hart;
  uint32_t pad;
} lane_out_t;

typedef struct {
  uint32_t num_rounds;
  uint32_t pad;
  uint64_t plan_addr;  // [num_rounds]             round_plan_t
  uint64_t out_addr;   // [num_rounds][MAX_LANES] lane_out_t
} kernel_arg_t;

// Probe loads before the copy is declared lost.
#define SPIN_BOUND 1000

// Values a lane stores before its LR and conditionally after it. Tagged with
// the round and lane so a value left over from another round cannot match.
static inline uint64_t sentinel_value(uint32_t round, uint32_t lane) {
  return 0x5E5E5E5E5E5E0000ull | (round << 4) | lane;
}

static inline uint64_t marker_value(uint32_t round, uint32_t lane) {
  return 0x3C3C3C3C3C3C0000ull ^ ((round << 4) | lane);
}

#endif
