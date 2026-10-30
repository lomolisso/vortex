#ifndef _DXA_COPY_COMMON_H_
#define _DXA_COPY_COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

#define DXA_MAX_DIMS 5

typedef struct {
  uint32_t ndim;               // number of dimensions (1-5)
  uint32_t elect;              // nonzero: only thread 0 of the DXA warp issues
  uint32_t sizes[DXA_MAX_DIMS];  // per-dimension array sizes
  uint32_t tiles[DXA_MAX_DIMS];  // per-dimension tile sizes
  uint32_t grids[DXA_MAX_DIMS];  // per-dimension grid sizes (sizes[d]/tiles[d])
  uint64_t src_addr;
  uint64_t dst_addr;           // each block writes its tile back here for the host check
} kernel_arg_t;

// copy_ext, the kernel of -m and -c. The tile sits at TILE_OFF in the block's
// local memory, after the mbarrier, and canary_words words follow it.
typedef struct {
  kernel_arg_t base;
  uint64_t err_addr;           // one ERR_* word per thread
  uint32_t mbar;               // complete on an mbarrier instead of a barrier event
  uint32_t canary_words;
  uint32_t tx_bytes;           // the box bytes the mbarrier expects
  uint32_t pad;
} ext_arg_t;

#define TILE_OFF     64
#define CANARY_WORDS 16
#define WAIT_LIMIT   64        // try_waits before a thread gives the phase up

#define ERR_WAIT     1         // the mbarrier phase did not complete
#define ERR_CANARY   2         // a canary word after the tile changed

static inline uint32_t canary_value(uint32_t block, uint32_t i) {
  return 0xC0DE0000u ^ (block << 8) ^ i;
}

#endif
