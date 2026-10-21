#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define MODE_SYNC   0  // cluster.sync() every phase
#define MODE_SPLIT  1  // arrive, rank-dependent work, wait
#define MODE_SKEW   2  // rank r works r units before cluster.sync()
#define MODE_MIXED  3  // __syncthreads() and a block barrier between arrive and wait
#define MODE_EXIT   4  // the last phase only arrives, then the blocks exit
#define MODE_HANDOFF 5 // handoff_k: global data passed to the next rank through the split barrier

// sync_k: the grid is 1-D in clusters of k x 1 x 1, so block b is rank b % k
// of cluster b / k.
typedef struct {
  uint64_t rec_addr;     // REC_WORDS per block
  uint64_t err_addr;     // one error count per warp, at b * warps + warp
  uint64_t count_addr;   // 0, or one arrival counter per cluster and phase
  uint32_t k;
  uint32_t phases;
  uint32_t mode;
  uint32_t spin;         // ALU iterations per unit of skew or work
  uint32_t num_threads;  // threads per warp
  uint32_t relaxed;      // sync with the fence-free arrive and wait
  uint64_t data_addr;    // handoff: two buffers of one line per block, `stride` words
  uint32_t stride;
} sync_arg_t;

#define REC_BLOCK    0  // blockIdx.x
#define REC_CORE     1  // VX_CSR_CORE_ID
#define REC_CTA_ID   2  // VX_CSR_CTA_ID: the dispatcher slot
#define REC_RANK     3  // cluster.block_rank()
#define REC_SIZE     4  // cluster.num_blocks()
#define REC_PHASES   5  // phases the block got through
#define REC_RUNS     6  // times the block ran (1 if it ran, without counters)
#define REC_WORDS    7

#endif
