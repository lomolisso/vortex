#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define MODE_ALL    0   // every thread arrives on every rank's object, each phase
#define MODE_DATA   1   // window stores into the next rank, then a remote arrive
#define MODE_GLOBAL 2   // global stores, then a cluster-scope release and acquire
#define MODE_TX     3   // remote arrive_tx, expect_tx and complete_tx
#define MODE_SELF   4   // MODE_ALL in DEFAULT clusters: every peer on the caller's core
#define NUM_MODES   5

// The grid is 1-D in clusters of k x 1 x 1: block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t err_addr;    // ERR_WORDS uint32 per thread
  uint64_t rec_addr;    // REC_WORDS uint32 per block
  uint64_t comp_addr;   // uint32 per block, rank and phase: the block's ops that completed that rank's FULL phase
  uint64_t word_addr;   // uint64: the NUM_OBJS final object words of each block
  uint64_t glob_addr;   // global: per block two buffers of `stride` doublewords
  uint32_t mode;
  uint32_t k;
  uint32_t rounds;      // phases
  uint32_t delay;       // idle iterations of the phase's late arriver
  uint32_t elems;       // data, global: doublewords per buffer
  uint32_t burst;       // data, global: stores per element and phase, the last one the tag
  uint32_t stride;      // global: doublewords per block buffer, whole cache lines
  uint32_t relaxed;     // all, self: relaxed arrives
  uint32_t fenced;      // global: the _cluster forms (0: the plain ones, which may read stale data)
  uint32_t parity;      // data: a fixed schedule with an exact instruction count
  uint32_t iters;       // cost: measurements of each kind
  uint32_t stores;      // cost: window stores in front of each arrive
} rem_arg_t;

#define REC_BLOCK  0   // blockIdx.x
#define REC_CORE   1   // VX_CSR_CORE_ID
#define REC_SLOT   2   // the dispatcher slot
#define REC_RANK   3
#define REC_SIZE   4
#define REC_RAN    5   // times the block ran
#define REC_WORDS  6

#define ERR_COUNT    0    // failed checks of the thread
#define ERR_FIRST    1    // 1 + phase of the first failed check, 0 if none
#define ERR_WHAT     2    // CHK_* code of the first failed check
#define ERR_NEG      3    // returned words with a negative tx (tx)
#define ERR_STALE    4    // stale global reads (global)
#define ERR_WAITS    5    // waits
#define ERR_SLEPT    6    // try_waits that read the phase incomplete, so suspended the warp
#define ERR_WOKE     7    // waits whose suspending try_wait the phase's completion ended
#define ERR_TIMEOUTS 8    // try_waits that ended at the time limit
#define ERR_WOKE_LO  9    // lane 0: bit p if phase p's wait was woken by the completion
#define ERR_WOKE_HI  10
#define ERR_CYC_RLX  11   // cost: cycles of the relaxed measurements, summed
#define ERR_CYC_REL  12   // cost: cycles of the release measurements, summed
#define ERR_MAX_RLX  13   // cost: the slowest relaxed measurement
#define ERR_MIN_REL  14   // cost: the fastest release measurement
#define ERR_WORDS    15

#define CHK_TOKEN  1   // a returned word's phase or fields are outside what the schedule allows
#define CHK_DATA   2   // data released by the arrive was not visible after the wait
#define CHK_WAIT   3   // the phase never completed (poll limit)
#define CHK_CALLS  4   // parity: a wait was not one suspending try_wait woken by the completion
#define CHK_NEVER  5   // a try_wait on a phase nobody completes returned true

// Doubleword indexes in the block's local memory.
#define MAX_ROUNDS  64
#define MAX_CLUSTER 8

#define OBJ_FULL   0    // FULL[2]: all, tx: the phase objects; data, global: filled by the previous rank
#define OBJ_EMPTY  2    // EMPTY[2]: data, global: emptied by the next rank
#define OBJ_COST   4    // cost: the measured arrives
#define NUM_OBJS   8
#define LM_COST    8    // cost: the window stores' targets
#define MAX_STORES 8
#define LM_FLAG    (LM_COST + MAX_STORES)   // data, tx: 1 + the last phase in which a peer said it is about to wait
#define LM_DONE    (LM_FLAG + 1)            // data, tx: thread-phases whose ops the block has issued (uint32)
#define LM_COMP    (LM_DONE + 1)            // MAX_CLUSTER x MAX_ROUNDS uint32: the completion records
#define LM_DATA    (LM_COMP + MAX_CLUSTER * MAX_ROUNDS / 2)   // data: two buffers of `elems` doublewords

#define TX_CHUNK   8          // tx: rank q's thread i completes TX_CHUNK * (1 + q * N + i) bytes
#define TX_EXTRA   24         // tx: bytes the peer expects and the owner completes on odd phases

// Polls before a thread gives a wait up.
#define TRY_LIMIT  (1u << 12)
#define POLL_LIMIT (1u << 20)

#endif
