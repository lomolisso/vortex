#ifndef _LMEM_FAIRNESS_COMMON_H_
#define _LMEM_FAIRNESS_COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t num_blocks;    // LSU blocks; warp w issues through block w % num_blocks
  uint32_t poll_block;    // block whose warps poll the flag
  uint32_t producer_wid;  // warp whose lane 0 sets the flag
  uint32_t delay;         // register-only iterations the producer waits first
  uint32_t bound;         // polls a poller makes at most
  uint32_t pad;
  uint64_t polls_addr;    // [grid][warps][threads] polls a poller needed, 0 = bound exhausted
} kernel_arg_t;

#endif
