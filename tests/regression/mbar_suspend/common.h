#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Modes (-m).
#define M_LIMIT 0   // warp 0, lanes [0, L): R try_waits on a phase that never completes (hint H unless NO_HINT)
#define M_DONE  1   // warp 0, every lane: R try_waits on a completed phase (the op alone)
#define M_WAKE  2   // warp 0 loops on try_wait (hint H unless NO_HINT); warp 1 lane 0 arrives D cycles later (12/spec.md §4.2)
#define M_SPIN  3   // as M_WAKE, warp 0 spinning on test_wait

#define NO_HINT    0xffffffffu
#define MAX_ROUNDS 16

#define LM_NEVER   0x0000
#define LM_DONE    0x0008
#define LM_WAKE    0x0010         // one object per round

// Output words (u32): per round the waiter's time, the arriver's clock and the waiter's clock (low words).
#define O_T        0
#define O_TA       (O_T + MAX_ROUNDS)
#define O_TW       (O_TA + MAX_ROUNDS)
#define O_ERR      (O_TW + MAX_ROUNDS)
#define O_KNOBS    (O_ERR + 1)    // suspend, front end on
#define O_WORDS    (O_KNOBS + 2)

typedef struct {
  uint64_t out_addr;
  uint32_t mode;
  uint32_t lanes;
  uint32_t hint;
  uint32_t rounds;
  uint32_t delay;
} sus_arg_t;

#endif
