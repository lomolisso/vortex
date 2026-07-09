#ifndef _DXA_MULTICAST_COMMON_H_
#define _DXA_MULTICAST_COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// Cross-core TMA multicast test arguments: a CTA cluster of `num_recv`
// single-warp CTAs, one per core; rank 0's DXA reads the tile from GMEM once
// and fans it out into every rank's scratchpad over the SM-to-SM network.
typedef struct {
  uint32_t tile_rows;
  uint32_t tile_cols;
  uint32_t src_row_stride;  // elements between source rows (descriptor stride)
  uint32_t num_recv;        // multicast group size (= cluster size)
  uint64_t src_addr;
  uint64_t dst_addr;        // tightly packed: num_recv × tile_rows × tile_cols
} kernel_arg_t;

#endif
