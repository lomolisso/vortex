#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// cta_t: one record of REC_WORDS words per block, at its blockIdx.x. The grid is
// 1-D in clusters of k x 1 x 1, so block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t rec_addr;
  uint32_t k;
  uint32_t work;          // ALU iterations every thread runs
  uint32_t long_work;     // the same for the blocks of cluster long_cluster
  uint32_t long_cluster;  // ~0: none
} cta_arg_t;

#define REC_BLOCK    0  // blockIdx.x
#define REC_SMID     1  // %smid (VX_CSR_CORE_ID)
#define REC_SLOT     2  // VX_CSR_CTA_ID: the dispatcher slot
#define REC_RANK     3  // get_cluster_rank()
#define REC_SIZE     4  // get_cluster_size()
#define REC_T0_LO    5  // thread 0's first clock read
#define REC_T0_HI    6
#define REC_T1_LO    7  // thread 0's last clock read
#define REC_T1_HI    8
#define REC_RUNS     9
#define REC_WORDS    10

#endif
