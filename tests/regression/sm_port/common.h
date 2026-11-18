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
  SP_FILLWR = 7,  // cluster: rank 0's warps 1.. stream LDS / L1-hit lw / STS (recv = SP_FW_*) beside a fill
                  // source (n = SP_BG_*): rank 0's warp 0 TMA bulk loads, or every warp of rank 1 sd'ing into rank 0
};

enum { SP_FW_LDS = 0, SP_FW_L1 = 1, SP_FW_STS = 2 };
enum { SP_BG_NONE = 0, SP_BG_TMA = 1, SP_BG_DSMEM = 2 };

typedef struct {
  uint32_t kind;
  uint32_t iters;      // instructions per warp, a multiple of 32
  uint32_t step;       // bytes the pattern base advances per instruction
  uint32_t span;       // bytes the base wraps within (a power of two)
  uint32_t recv;       // DSTORE: receiving rank
  uint32_t n;          // RW: stores before the timed load, a multiple of 8 x warps
  uint32_t lmem_bytes;
  uint32_t extra;      // FILLWR: index in out of the words after the per-warp pairs
  uint64_t pat_addr;   // [SP_LANES] byte offsets, 0xFFFFFFFF = idle lane
  uint64_t gmem_addr;  // global region (span bytes, plus a cold line at span for RW)
  uint64_t out_addr;   // [blocks][warps][2] start/end; RW: [4]; TMA: [2 phases][blocks][2];
                       // FILLWR: then [3] sd bytes sent, TMA bytes and cycles between the window's completions
} kernel_arg_t;

#endif
