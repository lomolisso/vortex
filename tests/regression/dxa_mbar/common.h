#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define NEG_NONE    0
#define NEG_EXPECT  1   // the issuer expects more bytes than the box: the phase never completes
#define NEG_RETIRED 2   // the issuing block exits before its copy lands
#define NEG_BOUNDS  3   // the mbarrier lies outside the issuing block's local memory
#define NEG_MCAST   4   // an mbarrier-mode copy with a multicast mask

typedef struct {
  uint64_t out_addr;     // OUT_WORDS uint32 per thread
  uint64_t word_addr;    // uint64: the final mbarrier word
  uint32_t legacy;       // complete on a barrier event instead of the mbarrier
  uint32_t neg;
  uint32_t rounds;
  uint32_t tile_elems;   // 32-bit elements per copy
  uint32_t num_src_tiles;
  uint32_t tx_bytes;     // the bytes the issuer's arrive_tx declares
  uint32_t scr_off;      // byte offset of the contention words, a multiple of a bank row
  uint32_t probe_bank;   // the bank the contention hammers: the last tile word's, or (-q) the mbarrier's
  uint32_t lmem_stride;  // bytes of the block's local-memory slot
  uint32_t linger;       // negatives: cycles the block stays alive after its issue
  uint32_t iters;        // contention iterations per thread; 0: until the waiters finish
  uint32_t copies;       // copies per round, each into the next tile: 1 or QUEUE_COPIES
} kernel_arg_t;

// Byte offsets in the block's local memory.
#define LM_MBAR 0
#define LM_DONE 8        // the contention warps stop once the waiters set it
#define LM_TILE 64

// Warps 0 and 1 wait on the copy (warp 0 issues it); the others contend for
// the probe's bank.
#define NUM_WAITERS 2

#define OUT_DONE     0   // 1 once the thread has written its words
#define OUT_ERRS     1   // failed checks
#define OUT_FIRST    2   // (check << 16) | round of the first failed check
#define OUT_SPINS    3   // probe loads that saw stale data after the wait, all rounds
#define OUT_WAKES    4   // try_waits that slept and were woken by the phase completing
#define OUT_TIMEOUTS 5   // try_waits that returned false
#define OUT_ITERS    6   // contention: load/store/AMO iterations
#define OUT_WORDS    8

#define CHK_WAIT  1      // the phase did not complete within TRY_LIMIT try_waits
#define CHK_DATA  2      // a tile element differs from the source
#define CHK_PROBE 3      // the probe never saw the copy's last word

// -q: copies a round, three times the completion queue of a 4-warp core,
// half of them issued by each waiter warp.
#define QUEUE_COPIES 12

#define TRY_LIMIT  64
#define SPIN_BOUND 1000

static inline uint32_t src_value(uint32_t i) {
  return i * 2654435761u + 0x9E37u;
}

#endif
