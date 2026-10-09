#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Kernel-argument block shared between host and device.
// The runtime stages it into a scratch slot; the KMU passes its address in a0.
typedef struct {
  uint32_t count;
  uint64_t src_addr;
  uint64_t dst_addr;
} kernel_arg_t;

// rank_k writes one record of RANK_REC_WORDS words per CTA, at blockIdx.x.
typedef struct {
  uint32_t cluster_k;
  uint64_t dst_addr;
} rank_arg_t;

#define RANK_REC_CTA_ID       0  // VX_CSR_CTA_ID: the dispatcher slot
#define RANK_REC_CLUSTER_RANK 1  // get_cluster_rank()
#define RANK_REC_INDEX        2  // blockIdx.x % cluster_k
#define RANK_REC_CLUSTER_SIZE 3  // get_cluster_size()
#define RANK_REC_WORDS        4

#endif
