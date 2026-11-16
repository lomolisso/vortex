#ifndef _SMEM_BANK_COMMON_H_
#define _SMEM_BANK_COMMON_H_

#include <stdint.h>

#define SB_LANES      32  // pattern row stride: the widest warp
#define SB_INACTIVE   0xFFFFFFFFu
#define SB_UNROLL     64

enum {
  SB_CHASE32 = 0,  // lw chase (+ add of the row base)
  SB_CHASE64 = 1,  // ld chase over full pointers; calib: then the same chase with an add
  SB_ATOM_LAT = 2, // amoadd.w / amocas.w chase (+ add)
  SB_ATOM_TPUT = 3,// independent atomics from every warp
  SB_LOAD_BW = 4,  // independent lw from every warp
};

typedef struct {
  uint32_t kind;
  uint32_t npat;
  uint32_t hops;       // per chase, a multiple of SB_UNROLL
  uint32_t rows;       // rows of a chain
  uint32_t row_bytes;  // a multiple of the bank span
  uint32_t iters;      // throughput: unrolled groups per warp
  uint32_t cas;        // atomics: amocas.w instead of amoadd.w
  uint32_t calib;      // ld chase: also time it with an add on every hop; amoadd.w chase: with amocas.w's mv
  uint64_t pat_addr;   // [npat][SB_LANES] byte offset in a row per lane, SB_INACTIVE = idle lane
  uint64_t out_addr;   // chase: [npat][3] cycles, cycles, sink; throughput: [npat][warps][2] start/end
} kernel_arg_t;

#endif
