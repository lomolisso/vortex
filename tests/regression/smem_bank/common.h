#ifndef _SMEM_BANK_COMMON_H_
#define _SMEM_BANK_COMMON_H_

#include <stdint.h>

#define SB_LANES      32  // pattern row stride: the widest warp
#define SB_INACTIVE   0xFFFFFFFFu
#define SB_UNROLL     64
#define SB_TIMED_ACCESSES 128  // accesses per iteration of a timed throughput loop
// SB_UM_LDS_STREAM lane width: 8-B lanes where XLEN allows
#ifdef __riscv
#define SB_UM_LANE_BYTES (__riscv_xlen / 8)
#else
#define SB_UM_LANE_BYTES (VX_CFG_XLEN / 8)
#endif

enum {
  SB_CHASE32 = 0,  // lw chase (+ add of the row base)
  SB_CHASE64 = 1,  // ld chase over full pointers; calib: then the same chase with an add
  SB_ATOM_LAT = 2, // amoadd.w / amocas.w chase (+ add)
  SB_ATOM_TPUT = 3,// independent atomics from every warp
  SB_LOAD_BW = 4,  // independent lw from every warp
  SB_MIX = 5,      // warp classes in order: LDS lw, L1-hit global lw, STS sw
  SB_UNDER_MISS = 6, // miss streamers first, then one chase warp or the LDS stream warps
};

enum {
  SB_UM_LDS_CHASE = 0,
  SB_UM_L1_CHASE = 1,
  SB_UM_LDS_STREAM = 2,
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
  uint32_t ncls[3];    // SB_MIX: warps per class
  uint32_t um_kind;    // SB_UNDER_MISS: SB_UM_*
  uint32_t miss_warps; // SB_UNDER_MISS: L2-hit streamers (0 = the victim alone)
  uint32_t instr;      // SB_UNDER_MISS chases: read the cycle counter after every hop
  uint64_t pat_addr;   // [npat][SB_LANES] byte offset in a row per lane, SB_INACTIVE = idle lane
  uint64_t out_addr;   // chase: [npat][3] cycles, cycles, sink; throughput: [npat][warps][2] start/end
                       // SB_UNDER_MISS: [warps][4] start, end, lines or bytes, max hop (chase warp)
  uint64_t gbl_addr;   // SB_MIX / SB_UNDER_MISS global lines; the global chase chain sits at gbl_bytes
  uint64_t gbl_bytes;  // SB_UNDER_MISS: the miss stream's footprint
} kernel_arg_t;

#endif
