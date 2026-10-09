#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define FENCE_MODE_VECADD 0
#define FENCE_MODE_STORM  1

// Storm control block: one word per 256-byte line, wider than any cache line,
// so no two control words share a line at any level.
#define STORM_LINE_WORDS  64
#define STORM_STARTED     (0 * STORM_LINE_WORDS)
#define STORM_DONE        (1 * STORM_LINE_WORDS)
#define STORM_COUNTER     (2 * STORM_LINE_WORDS)
#define STORM_CTRL_WORDS  (3 * STORM_LINE_WORDS)

// The timer loads one fresh line per iteration, a word per lane.
#define STORM_ITER_WORDS  STORM_LINE_WORDS

// Per-thread storm record, written once at the end.
#define STORM_REC_KIND    0  // STORM_KIND_*
#define STORM_REC_COUNT   1  // timer: iterations; fencer: fences issued
#define STORM_REC_FLAG    2  // timer: fencer lanes running at its start; fencer: saw the done flag
#define STORM_REC_CYCLES  3  // timer: timed-phase cycles; fencer: cycles it ran
#define STORM_REC_MAXLAT  4  // timer: slowest iteration in cycles
#define STORM_REC_ERRORS  5  // timer: wrong load values
#define STORM_REC_WAIT    6  // timer: cycles waiting for the fencers to start
#define STORM_REC_SEEN    7  // fencer: counter read after its acquire fence
#define STORM_REC_LAST    8  // fencer: counter read in its last pass
#define STORM_REC_WORDS   16

#define STORM_KIND_TIMER  1u
#define STORM_KIND_FENCER 2u
#define STORM_KIND_IDLE   3u
#define STORM_UNWRITTEN   0xFFFFFFFFu

static inline uint32_t storm_value(uint32_t iter, uint32_t lane) {
  return 0x51000000u ^ (iter << 8) ^ lane;
}

typedef struct {
  uint32_t mode;
  uint32_t num_tasks;
  uint32_t task_size;
  uint32_t storm_iters;   // timed load+AMO iterations of each timer lane
  uint32_t storm_fencers; // fencing warps
  uint32_t storm_budget;  // cycles the timer may take, and a fencer waits for it
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;
  uint64_t ctrl_addr;
  uint64_t rec_addr;
} kernel_arg_t;

#endif
