#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef ITYPE
#define ITYPE mxfp8
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t MX_A_addr;
  uint64_t MX_B_addr;
  uint64_t err_addr;   // per block and warp: phases given up after SPIN_LIMIT polls
} kernel_arg_t;

// Local memory: FULL and EMPTY, then the A and B metadata words of one k-tile.
#define LM_FULL  0
#define LM_EMPTY 1
#define LM_META  2

// Polls of an incomplete phase before a warp gives the k-tile up.
#define SPIN_LIMIT (1u << 16)

#endif
