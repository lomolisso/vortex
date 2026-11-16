#ifndef _L1_SECTOR_COMMON_H_
#define _L1_SECTOR_COMMON_H_

#include <stdint.h>

#define LS_LANES   32  // the widest warp
#define LS_UNROLL  64

enum {
  LS_CHASE  = 0,  // per-lane ld chase over host-written pointers: warm hops, optional fence, timed hops
  LS_OPS    = 1,  // one thread runs an op list (loads, stores, timed loads, fences)
  LS_SWEEP  = 2,  // one thread: fence, then `laps` passes over n lines at a stride, timing the last passes
  LS_BW     = 3,  // every warp issues independent lw on L1-resident lines
  LS_BW_ALU = 4,  // LS_BW's loop with an add in place of each lw: the front end's issue rate
};

enum {
  OP_LOAD = 0,        // lw
  OP_STORE = 1,       // sw
  OP_TIMED = 2,       // lw, the cycles until a dependent instruction can issue
  OP_FENCE_GPU = 3,   // fence iorw, iorw
  OP_FENCE_CTA = 4,   // the block-scope fence (DLMEM builds)
  OP_FENCE_CLUSTER = 5,
};

typedef struct {
  uint32_t kind;
  uint32_t lanes;      // chase: active lanes
  uint32_t warm;       // chase: warm hops before timing
  uint32_t hops;       // chase: timed hops, a multiple of LS_UNROLL
  uint32_t fence;      // chase: OP_FENCE_* after warming, 0 = none
  uint32_t nops;       // ops: entries
  uint32_t n;          // sweep: lines per pass; bw: unrolled groups per warp
  uint32_t laps;       // sweep: passes (the last `timed_laps` are timed)
  uint32_t timed_laps;
  uint32_t stride;     // sweep: bytes between lines
  uint32_t miss_thr;   // sweep: cycles above which a timed load is a miss
  uint32_t tail_from;  // sweep: timed passes start at this line
  uint64_t start_addr; // chase: [lanes] first pointers; ops: [nops][2] (op, addr); sweep/bw: base
  uint64_t out_addr;   // chase: 2 cycles; ops: cycles per op; sweep: per timed load cycles; bw: [warps][2]
} kernel_arg_t;

#endif
