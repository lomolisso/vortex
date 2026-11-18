#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define WOP_MAX_LANES 32
#define WOP_SENTINEL  0x5a5a5a5a5a5a5a5aull
#define WOP_ILP       8
#define WOP_CHAIN_UNROLL 8

enum { MODE_BITEXACT = 0, MODE_LATENCY = 1, MODE_RATE = 2 };

// One op over one warp: lane i runs it when tmask bit i is set and reads rs1[i]
// (and rs2[i], the high word of a b64 value at XLEN 32).
typedef struct {
  uint32_t code;
  uint32_t pad;
  uint64_t tmask;
  uint64_t rs1[WOP_MAX_LANES];
  uint64_t rs2[WOP_MAX_LANES];
} wop_case_t;

typedef struct {
  uint64_t cases_addr;  // bitexact: wop_case_t[ncases]
  uint64_t out_addr;    // bitexact: uint64_t[ncases][lanes]; latency/rate: per warp {start, end, sink}
  uint32_t mode;
  uint32_t ncases;
  uint32_t code;        // latency/rate: the op
  uint32_t k;           // MATCH.ANY: distinct values per warp
  uint32_t iters;
  uint32_t pad;
} wop_arg_t;

#endif
