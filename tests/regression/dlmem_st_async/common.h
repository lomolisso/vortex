#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define OP_ST32     0   // st.async.b32
#define OP_ST64     1   // st.async.b64
#define OP_RED32    2   // red.async.add.u32
#define OP_RED64    3   // red.async.add.u64
#define NUM_OPS     4

#define MODE_RING   0   // stores: rank r into rank (r + 1) % K; adds: every rank into rank 0
#define MODE_SELF   1   // stores and adds: every rank into itself, through its own window address
#define NUM_MODES   2

#define NEG_NONE      0
#define NEG_LOCAL     1   // the destination is a local pointer, not a window address
#define NEG_MBAR      2   // the mbarrier is in another block than the destination
#define NEG_MBARLOCAL 3   // the mbarrier is a local pointer
#define NEG_RANK      4   // the destination and its mbarrier lie outside the cluster
#define NEG_OFF       5   // the destination lies past the end of the peer's block
#define NUM_NEGS      6

// The grid is 1-D in clusters of k x 1 x 1: block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t err_addr;     // ERR_WORDS uint32 per thread
  uint64_t rec_addr;     // REC_WORDS uint32 per block
  uint64_t dump_addr;    // lmem_bytes per block: its local memory at the end
  uint32_t mode;
  uint32_t neg;
  uint32_t k;
  uint32_t ops;          // bit o: op o runs
  uint32_t rounds;
  uint32_t elems;        // stores (adds) per thread, op and round
  uint32_t lmem_bytes;
  uint32_t linger;       // negatives: cycles each block stays alive around its issue
  uint64_t neg_win;      // -x rank: window offset of the bad block; -x off: offset past the peer's block
} sa_arg_t;

// Byte offsets in the block's local memory: one mbarrier per op, the two
// counters, then the 32-bit and the 64-bit store areas of elems x threads
// words each, element i of thread t at i * threads + t, so the lanes of one
// instruction write consecutive words. Everything else starts as canary
// bytes, so a stray write shows.
#define LM_MBAR     0
#define LM_CTR32    32
#define LM_CTR64    40
#define LM_ST32     64
#define LM_CANARY   12   // words from here on start as canaries (the mbarriers and counters come first)

static inline uint32_t lm_st64(uint32_t threads, uint32_t elems) {
  return (LM_ST32 + threads * elems * 4 + 7) / 8 * 8;
}

static inline uint32_t lm_bytes(uint32_t threads, uint32_t elems) {
  return lm_st64(threads, elems) + threads * elems * 8 + 64;
}

#define REC_BLOCK   0   // blockIdx.x
#define REC_CORE    1
#define REC_SLOT    2   // the dispatcher slot
#define REC_RANK    3
#define REC_RAN     4   // times the block ran
#define REC_WORDS   5

#define ERR_DONE    0   // 1 once the thread finished
#define ERR_WAIT    1   // waits that gave up after WAIT_LIMIT try_waits
#define ERR_STALE   2   // stored words that did not hold the round's value right after the wait
#define ERR_RED     3   // counters that did not hold the round's total right after the wait
#define ERR_TRIES   4   // try_waits of the thread
#define ERR_WORDS   5

#define WAIT_LIMIT  64

static inline uint32_t mix32(uint32_t a, uint32_t b) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

static inline uint32_t canary_word(uint32_t b, uint32_t w) {
  return mix32(0xCD0000u + b, w);
}

// What sender block b's thread t stores (adds) as element i of round p.
static inline uint32_t st32_value(uint32_t p, uint32_t b, uint32_t t, uint32_t i) {
  return mix32(0x5A0000u + p, (b << 16) | (t << 6) | i);
}

static inline uint64_t st64_value(uint32_t p, uint32_t b, uint32_t t, uint32_t i) {
  return ((uint64_t)mix32(0x5B0000u + p, (b << 16) | (t << 6) | i) << 32) | st32_value(p, b, t, i);
}

// Wraps the 32-bit counter within a round.
static inline uint32_t red32_value(uint32_t p, uint32_t b, uint32_t t, uint32_t i) {
  return 0x01000193u * (1 + p + ((b << 16) | (t << 6) | i));
}

// Carries across bit 32.
static inline uint64_t red64_value(uint32_t p, uint32_t b, uint32_t t, uint32_t i) {
  return 0xFFFFFF00ull + (uint64_t)(((b << 16) | (t << 6) | i) + 1) * 0x10001ull + p;
}

// The rank whose block the ops of rank r go to.
static inline uint32_t st_dst(uint32_t mode, uint32_t r, uint32_t k) {
  return mode == MODE_RING ? (r + 1) % k : r;
}

static inline uint32_t red_dst(uint32_t mode, uint32_t r) {
  return mode == MODE_RING ? 0 : r;
}

// Ranks whose ops go to rank q.
static inline uint32_t red_senders(uint32_t mode, uint32_t q, uint32_t k) {
  return mode == MODE_RING ? (q == 0 ? k : 0) : 1;
}

static inline uint32_t op_bytes(uint32_t op) {
  return (op == OP_ST32 || op == OP_RED32) ? 4 : 8;
}

#endif
