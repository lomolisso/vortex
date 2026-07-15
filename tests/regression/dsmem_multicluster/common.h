#ifndef _DSMEM_MULTICLUSTER_COMMON_H_
#define _DSMEM_MULTICLUSTER_COMMON_H_

#include <stdint.h>

// F1 test: multiple thread-block clusters resident at once. A grid of
// `num_clusters` clusters, each of `cluster_size` (K) single-warp CTAs, is
// launched one-rank-per-core; gang placement overlaps different clusters on
// shared SMs, so every SM's residency REPLICA must hold several clusters and
// resolve each remote access by cluster_id. Each CTA publishes a pattern keyed
// by (cluster_id, rank, tid) and remote-loads a peer WITHIN ITS OWN cluster;
// a keying/isolation bug would read another cluster's value or a stale slot.
typedef struct {
  uint32_t cluster_size;   // K (CTAs per cluster)
  uint32_t buf_elems;      // threads per CTA (= 1 warp)
  uint64_t dst_addr;       // [total_ctas * buf_elems] remote-loaded values
} kernel_arg_t;

// Pattern published by (cluster c, rank r, thread t). Distinct across clusters
// so a cross-cluster leak is detectable.
#define DSMEM_MC_PATTERN(c, r, t) (((c) + 1u) * 100000u + ((r) + 1u) * 1000u + (t))

#endif
