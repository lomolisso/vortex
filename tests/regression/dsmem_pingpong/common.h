#ifndef _DSMEM_PINGPONG_COMMON_H_
#define _DSMEM_PINGPONG_COMMON_H_

#include <stdint.h>

// Cross-core DSMEM ping-pong test arguments. A CTA cluster of `num_ranks`
// single-warp CTAs is placed one-rank-per-core; ranks exchange scratchpad
// data over the SM-to-SM network in both directions:
//   dstA — each rank's remote LOAD of its peer's scratchpad
//   dstB — each rank's local read of what its peer remote-STOREd into it
//          (validates the cluster barrier's drain: no stale reads)
typedef struct {
  uint32_t num_ranks;
  uint32_t buf_elems;  // elements per phase per rank (= threads per CTA)
  uint64_t dstA_addr;  // num_ranks × buf_elems
  uint64_t dstB_addr;  // num_ranks × buf_elems
} kernel_arg_t;

#endif
