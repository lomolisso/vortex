#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

enum { MODE_CHASE = 0, MODE_PAIR = 1, MODE_STREAM = 2, MODE_SPARSE = 3, MODE_TOP = 4 };

enum { SP_LD64 = 0, SP_LD32 = 1, SP_ST32 = 2, SP_ST64 = 3, SP_ST16 = 4 };

enum { PLACE_ONE = 0, PLACE_ALL = 1 };

// Lines touched per thread and loop iteration: independent accesses in flight per warp.
#define BENCH_ILP 4

// stream/sparse result words per thread: read sum, then (first, last) cycle of each placement's phase.
#define LINES_OUT_WORDS 5

typedef struct {
  uint32_t mode;
  uint32_t count;          // chase: nodes per pass; pair: trials; stream/sparse: lines per thread and phase; top: words
  uint32_t passes;         // chase: passes over the chain (the first misses, the rest hit the L2)
  uint32_t param;          // stream: writes per 8 slots; sparse: pattern
  uint32_t phases;         // stream: placements run in order (bit 0: one channel, bit 1: every channel)
  uint32_t lg_channels;    // log2 of the DRAM channels
  uint32_t lg_sel;         // log2 of the channel-select granule (bytes)
  uint32_t gap;            // cycles of idle between pair trials
  uint64_t buf_addr;       // data buffer, aligned to the granule times the channels
  uint64_t list_addr;      // chase: start node; pair: two line addresses per trial
  uint64_t out_addr;       // per-step cycles (chase, pair) or per-thread read sums
} kernel_arg_t;

#endif
