#ifndef _DSMEM_BARRIER_STRESS_COMMON_H_
#define _DSMEM_BARRIER_STRESS_COMMON_H_

#include <stdint.h>

// F2 test: a loop of `phases` cluster barriers forms a producer->consumer chain
// across a cluster (and across several clusters at once). In each phase rank r
// remote-writes a phase-tagged value into its peer, a cluster barrier orders
// the write before the peer reads it, and a second barrier orders the read
// before the next phase overwrites. An early-releasing barrier (F2 bug) makes a
// read observe a stale / next-phase value; a broken cross-SM release deadlocks.
typedef struct {
  uint32_t cluster_size;   // K
  uint32_t buf_elems;      // threads per CTA (warps * NUM_THREADS)
  uint32_t phases;         // N barriered phases
  uint64_t dst_addr;       // [total_ctas * phases * buf_elems]
} kernel_arg_t;

// Value written in phase p by (rank r, thread t) — distinct per (p, r, t).
#define DSMEM_BS_PATTERN(p, r, t) (((p) + 1u) * 100000u + ((r) + 1u) * 1000u + (t))

#endif
