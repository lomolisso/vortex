#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define JOB_TLOAD     0   // tensor load (tile or im2col by the descriptor)
#define JOB_TPREFETCH 1   // tensor prefetch
#define JOB_BLOAD     2   // global -> shared bulk load
#define JOB_BPREFETCH 3   // bulk prefetch into the L2

#define DST_LOCAL  0xFFFFFFFFu   // dst_rank: the issuer's own local pointer

#define MAX_JOBS     32
#define MAX_CLUSTER  4

// One instruction of the run, issued by warp 0 of rank `issuer`, in order.
typedef struct {
  uint32_t kind;
  uint32_t rank;       // tensor rank
  uint32_t im2col;
  uint32_t slot;
  uint32_t issuer;
  uint32_t dst_rank;   // DST_LOCAL, or the window rank of the destination and its mbarrier
  uint32_t dst_off;    // byte offset of the destination in its block
  uint32_t mbar_skew;  // bytes added to the mbarrier pointer (-x mbar-align)
  uint32_t mask;       // cluster multicast ctaMask, 0 = none
  uint32_t size;       // bulk bytes
  uint64_t gsrc;       // bulk source
  int32_t  coord[5];
  uint32_t offs[3];
} job_t;

// -p: a local-memory byte the last repeat must leave.
typedef struct {
  uint32_t off;
  uint32_t val;
} spot_t;

typedef struct {
  uint64_t jobs_addr;   // job_t[num_jobs]
  uint64_t dump_addr;   // lmem_bytes per block: its local memory at the end
  uint64_t err_addr;    // ERR_WORDS uint32 per block
  uint64_t spots_addr;  // -p: spot_t[num_spots] of block 0
  uint32_t num_jobs;
  uint32_t lmem_bytes;
  uint32_t tx_in[MAX_CLUSTER];  // bytes that land in rank r, its arrive_tx
  uint32_t neg;
  uint32_t linger;
  uint32_t repeats;     // -p: the jobs run this many times, one phase each; 0 = the checked run
  uint32_t num_spots;
} load_arg_t;

// The mbarrier in the first doubleword of every block, canaries after it.
#define LM_MBAR   0
#define LM_DATA   128

#define ERR_DONE    0
#define ERR_WAIT    1   // the phase did not complete within WAIT_LIMIT try_waits
#define ERR_TRIES   2
#define ERR_SPOTS   3   // -p, thread 0: wrong spot bytes
#define ERR_CYCLES  4   // -p, thread 0: cycles from the first issue to the last phase
#define ERR_MBAR_LO 5   // -p, thread 0: the final mbarrier word
#define ERR_MBAR_HI 6
#define ERR_WORDS   8

#define WAIT_LIMIT 256

static inline uint32_t mix32(uint32_t a, uint32_t b) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

// Word w of block b's local memory before any load lands.
static inline uint32_t canary_word(uint32_t b, uint32_t w) {
  return mix32(0xCA0000u + b, w);
}

#endif
