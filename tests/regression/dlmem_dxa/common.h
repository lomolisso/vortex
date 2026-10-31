#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define MODE_RING   0   // rank r copies into rank (r + 1) % K
#define MODE_MANY   1   // every rank copies into rank 0, back to back
#define MODE_SELF   2   // control: every rank copies into its own local memory through local pointers
#define NUM_MODES   3

#define NEG_NONE    0
#define NEG_RANK    1   // the destination and its mbarrier lie outside the cluster
#define NEG_MBAR    2   // the mbarrier is in another block than the destination
#define NEG_LEGACY  3   // a window destination completing on a barrier event
#define NEG_MCAST   4   // a window destination with a multicast mask
#define NEG_RETIRED 5   // the destination block exits before the copy reaches it
#define NEG_BOX     6   // the box starts inside the destination block and runs past its end
#define NUM_NEGS    7

#define MAX_DIMS    5
#define MAX_CLUSTER 8

// The grid is 1-D in clusters of k x 1 x 1: block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t err_addr;     // ERR_WORDS uint32 per thread
  uint64_t rec_addr;     // REC_WORDS uint32 per block
  uint64_t dump_addr;    // lmem_bytes per block: its local memory at the end
  uint32_t mode;
  uint32_t neg;
  uint32_t k;
  uint32_t ndim;
  uint32_t copies;       // copies each block issues
  uint32_t ntiles;       // source tiles
  uint32_t box;          // bytes per copy
  uint32_t tstride;      // bytes between the tile slots in local memory
  uint32_t lmem_bytes;
  uint32_t linger;       // negatives: cycles each block stays alive around its issue
  uint32_t tiles[MAX_DIMS];  // tile extents, elements
  uint32_t grids[MAX_DIMS];  // source tiles per dimension
  uint64_t neg_win;      // -x rank: window offset of the bad block (rank << LMEM_LOG | offset); -x box: block offset
} dxa_arg_t;

// Byte offsets in the block's local memory: the mbarrier, then the tile
// slots, each followed by at least a 64-byte gap. Everything but the
// mbarrier starts as canary bytes, so a stray write shows.
#define LM_MBAR     0
#define LM_TILE     64
#define DESC_SLOT   0

#define REC_BLOCK   0   // blockIdx.x
#define REC_CORE    1
#define REC_SLOT    2   // the dispatcher slot
#define REC_RANK    3
#define REC_SIZE    4
#define REC_RAN     5   // times the block ran
#define REC_WORDS   6

#define ERR_DONE    0   // 1 once the thread finished
#define ERR_WAIT    1   // the mbarrier phase did not complete within WAIT_LIMIT try_waits
#define ERR_TRIES   2   // try_waits of the thread
#define ERR_WORDS   4

#define WAIT_LIMIT  64

static inline uint32_t mix32(uint32_t a, uint32_t b) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

// Byte i of the source array.
static inline uint8_t src_byte(uint32_t i) {
  return (uint8_t)(mix32(0xD2A0u, i) >> 8);
}

// Byte i of block b's local memory before any copy lands.
static inline uint8_t canary_byte(uint32_t b, uint32_t i) {
  return (uint8_t)(mix32(0xCA0000u + b, i >> 2) >> (8 * (i & 3)));
}

static inline uint32_t canary_word(uint32_t b, uint32_t w) {
  return mix32(0xCA0000u + b, w);
}

// Who sends what: copy i of block b (rank r of its cluster, `k` blocks).
static inline uint32_t dst_rank(uint32_t mode, uint32_t r, uint32_t k) {
  return mode == MODE_RING ? (r + 1) % k : mode == MODE_MANY ? 0 : r;
}

static inline uint32_t dst_slot(uint32_t mode, uint32_t r, uint32_t i, uint32_t copies) {
  return mode == MODE_MANY ? r * copies + i : i;
}

static inline uint32_t num_slots(uint32_t mode, uint32_t k, uint32_t copies) {
  return mode == MODE_MANY ? k * copies : copies;
}

// Copies that land in rank q.
static inline uint32_t incoming(uint32_t mode, uint32_t q, uint32_t k, uint32_t copies) {
  return mode == MODE_MANY ? (q == 0 ? k * copies : 0) : copies;
}

static inline uint32_t src_tile(uint32_t b, uint32_t i, uint32_t copies, uint32_t ntiles) {
  return ((b * copies + i) * 5 + 1) % ntiles;
}

#endif
