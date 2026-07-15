#ifndef _DSMEM_STOREFLOOD_COMMON_H_
#define _DSMEM_STOREFLOOD_COMMON_H_

#include <stdint.h>

// F3 test: a warp FLOODS `iters * buf_elems` remote stores into a peer CTA's
// scratchpad (more than the outstanding-remote-store credit), then the cluster
// barrier drains them. Verifies (a) every stored value lands despite the
// backpressure, (b) no deadlock when the credit is exhausted. Run at several
// VX_CFG_DSMEM_STORE_CREDITS to exercise stall vs no-stall.
typedef struct {
  uint32_t cluster_size;   // K
  uint32_t buf_elems;      // threads per CTA (= 1 warp)
  uint32_t iters;          // store-loop count (iters*buf_elems stores/warp)
  uint64_t dst_addr;       // [K * iters * buf_elems] readback
} kernel_arg_t;

// Value written by (writer rank r, loop i, thread t) — distinct per (r,i,t).
#define DSMEM_SF_PATTERN(r, i, t) (((r) + 1u) * 1000000u + (i) * 1000u + (t))

#endif
