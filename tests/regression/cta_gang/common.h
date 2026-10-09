#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// gang_k: one record of REC_WORDS words per block, at its blockIdx.x. The grid is
// 1-D in clusters of k x 1 x 1, so block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t rec_addr;
  uint64_t arrive_addr;  // 0, or one arrival counter per cluster (the rendezvous)
  uint32_t k;
  uint32_t lmem_bytes;
  uint32_t bound;        // rendezvous polls before giving up
  uint32_t work;         // ALU iterations every thread runs first
} gang_arg_t;

#define REC_BLOCK    0  // blockIdx.x
#define REC_CORE     1  // VX_CSR_CORE_ID
#define REC_CTA_ID   2  // VX_CSR_CTA_ID: the dispatcher slot
#define REC_RANK     3  // get_cluster_rank()
#define REC_SIZE     4  // get_cluster_size()
#define REC_LMEM_LO  5  // VX_CSR_CTA_LMEM_ADDR
#define REC_LMEM_HI  6
#define REC_RUNS     7  // times the block ran (1 if it ran, without the A extension)
#define REC_LMEM_OK  8  // 1 when the first and last words of the slice read back
#define REC_POLLS    9  // rendezvous polls until the whole cluster arrived; 0 on a timeout
#define REC_WORDS    10

#endif
