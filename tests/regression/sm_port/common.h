#ifndef _SM_PORT_COMMON_H_
#define _SM_PORT_COMMON_H_

#include <stdint.h>

#define SP_LANES 32  // the widest warp

enum {
  SP_GSTORE = 0,  // every warp: sw at the lane pattern, the base advancing by `step` within `span`
  SP_RW     = 1,  // every warp: n / warps full-line stores, then warp 0 times one load of a cold line
  SP_DSTORE = 2,  // cluster: every warp of each rank but `recv` sd into rank recv's window
  SP_DLOAD  = 3,  // cluster: rank 0's warps ld from rank 1's window
  SP_DATOM  = 4,  // cluster: rank 0's warps amoadd.w into rank 1's window
  SP_MIX    = 5,  // cluster: rank 0's even warps sd into rank 1, odd warps sw to global memory
  SP_TMA    = 6,  // cluster: warp 0 bulk-loads global lines into its own block, rank 0 alone, then every rank
};

typedef struct {
  uint32_t kind;
  uint32_t iters;      // instructions per warp, a multiple of 32
  uint32_t step;       // bytes the pattern base advances per instruction
  uint32_t span;       // bytes the base wraps within (a power of two)
  uint32_t recv;       // DSTORE: receiving rank
  uint32_t n;          // RW: stores before the timed load, a multiple of 8 x warps
  uint32_t lmem_bytes;
  uint32_t pad;
  uint64_t pat_addr;   // [SP_LANES] byte offsets, 0xFFFFFFFF = idle lane
  uint64_t gmem_addr;  // global region (span bytes, plus a cold line at span for RW)
  uint64_t out_addr;   // [blocks][warps][2] start/end; RW: [4]; TMA: [2 phases][blocks][2]
} kernel_arg_t;

#endif
