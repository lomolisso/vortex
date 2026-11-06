#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define FENCE_NONE    0
#define FENCE_CTA     1   // vx_fence_cta()
#define FENCE_CLUSTER 2   // vx_fence_cluster()
#define FENCE_INIT    3   // vx_fence_mbarrier_init_release_cluster()

typedef struct {
  uint64_t out_addr;   // OUT_WORDS uint32 per rank
  uint32_t rounds;
  uint32_t fence;
  uint32_t hammer;     // rank 0's other warps hammer its banks with atomics meanwhile
  uint32_t tries;      // try_wait calls before a round counts as lost
} fence_arg_t;

// Doubleword indexes in the block's local memory.
#define LM_BAR   0       // MAX_ROUNDS objects, a fresh one per round
#define LM_FLAG  256     // rank 1's round flag, written by rank 0 through the window
#define LM_DONE  264
#define LM_HAM   272     // the hammered words
#define LM_WORDS (LM_HAM + 64)
#define MAX_ROUNDS 256

#define OUT_RAN   0
#define OUT_RANK  1
#define OUT_LOST  2
#define OUT_FIRST 3      // 1 + the first lost round
#define OUT_HAM   4      // rank 0's hammer atomics
#define OUT_CORE  5
#define OUT_WORDS 8

#endif
