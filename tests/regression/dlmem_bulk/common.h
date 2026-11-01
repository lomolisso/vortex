#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define MODE_RING   0   // rank r copies into rank (r + 1) % K
#define MODE_MANY   1   // every rank copies into rank 0, back to back
#define MODE_SELF   2   // every rank copies into its own local memory through local pointers
#define MODE_RACE   3   // rounds of plain stores to the source, then a local copy of it
#define NUM_MODES   4

#define NEG_NONE    0
#define NEG_SIZE    1   // a size that is not a multiple of 16
#define NEG_ALIGN   2   // a source 8 bytes off a 16-byte boundary
#define NEG_DALIGN  3   // a destination 8 bytes off a 16-byte boundary in its block
#define NEG_SRC     4   // a source that starts inside the issuer's block and runs past its end
#define NEG_RANK    5   // the destination and its mbarrier lie outside the cluster
#define NEG_MBAR    6   // the mbarrier is in another block than the destination
#define NEG_BOX     7   // the copy starts inside the destination block and runs past its end
#define NEG_RETIRED 8   // the destination block exits before the copy reaches it
#define NUM_NEGS    9

#define MAX_CLUSTER 8
#define MAX_COPIES  8
#define MAX_ROUNDS  16

// The grid is 1-D in clusters of k x 1 x 1: block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t err_addr;     // ERR_WORDS uint32 per thread
  uint64_t rec_addr;     // REC_WORDS uint32 per block
  uint64_t dump_addr;    // lmem_bytes per block: its local memory at the end
  uint64_t neg_win;      // -x rank: window offset of the bad block (rank << LMEM_LOG | offset)
  uint64_t sync_addr;    // race with a hog: per cluster, the round that may go and the racer's ready round
  uint32_t mode;
  uint32_t neg;
  uint32_t k;
  uint32_t copies;       // copies each block issues (race: 1 per round)
  uint32_t size;         // bytes per copy; 0 = the size mix
  uint32_t salign;       // source misalignment base, 16-byte units
  uint32_t dalign;       // destination misalignment base, 16-byte units
  uint32_t pitch;        // bytes between slots
  uint32_t lmem_bytes;   // a multiple of 256, so it is the block's slot stride too
  uint32_t rounds;       // race rounds
  uint32_t fence;        // vx_fence_proxy_async() between the source stores and the copy
  uint32_t warp_threads;
  uint32_t soft;         // copy with the warp's own loads and stores instead of the DXA
  uint32_t linger;       // negatives: cycles each block stays alive around its issue
  uint32_t neg_off;      // -x src, -x box: a block offset 32 bytes before its end
  uint32_t lm_mbar;      // the mbarrier's byte offset in the block, a doubleword of the header
  uint32_t hog;          // race: rank 0 copies this many bytes per round instead of racing; 0: none
} bulk_arg_t;

// Byte offsets in the block's local memory: the header (the mbarrier, at
// lm_mbar, and the flag word after it), the source slots, the destination
// slots, then (race) the words the other warps hammer. Slots are `pitch` apart
// and each copy sits 0-48 bytes into its slot, so at least 64 canary bytes
// follow every source and destination range.
#define LM_SRC      64
#define MAX_MBAR_DW 6   // the mbarrier's doubleword in the header, at most
#define HOG_LEAD    16  // race with a hog: cycles its copy runs before the racers go

static inline uint32_t lm_flag(const bulk_arg_t* a) {
  return a->lm_mbar + 8;
}

// Race with a hog: rank 0 of each cluster.
static inline bool is_hog(const bulk_arg_t* a, uint32_t r) {
  return a->hog != 0 && r == 0;
}

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

static inline uint8_t word_byte(uint32_t w, uint32_t i) {
  return (uint8_t)(w >> (8 * (i & 3)));
}

// Word w of block b's local memory before any copy lands.
static inline uint32_t canary_word(uint32_t b, uint32_t w) {
  return mix32(0xCA0000u + b, w);
}

// Word w of block b's source ranges.
static inline uint32_t src_word(uint32_t b, uint32_t w) {
  return mix32(0x5C0000u + b, w);
}

// Word w of block b's source in race round j.
static inline uint32_t race_word(uint32_t b, uint32_t j, uint32_t w) {
  return mix32(0x7A0000u + b * MAX_ROUNDS + j, w);
}

// The final word of hammer thread h of block b.
static inline uint32_t ham_word(uint32_t b, uint32_t h) {
  return mix32(0x4A0000u + b, h);
}

// Sizes of the mix: odd multiples of 16.
static inline uint32_t mix_size(uint32_t n) {
  return n % 4 == 0 ? 16 : n % 4 == 1 ? 48 : n % 4 == 2 ? 208 : 720;
}

#define MIX_MAX 720

static inline uint32_t copy_size(const bulk_arg_t* a, uint32_t r, uint32_t i) {
  return a->size ? a->size : mix_size(i + r);
}

static inline uint32_t dst_rank(uint32_t mode, uint32_t r, uint32_t k) {
  return mode == MODE_RING ? (r + 1) % k : mode == MODE_MANY ? 0 : r;
}

static inline uint32_t num_src_slots(const bulk_arg_t* a) {
  return a->mode == MODE_RACE ? 1 : a->copies;
}

static inline uint32_t num_dst_slots(const bulk_arg_t* a) {
  return a->mode == MODE_RACE ? a->rounds : a->mode == MODE_MANY ? a->k * a->copies : a->copies;
}

static inline uint32_t dst_slot(uint32_t mode, uint32_t r, uint32_t i, uint32_t copies) {
  return mode == MODE_MANY ? r * copies + i : i;
}

// Source and destination offsets of copy i of rank r: the misalignments step
// so that the source and destination offsets of one copy differ by an odd
// multiple of 16 under the defaults, thus modulo the 32-byte row and the
// 64-byte line.
static inline uint32_t src_off(const bulk_arg_t* a, uint32_t r, uint32_t i) {
  const uint32_t s = a->mode == MODE_RACE ? 0 : i;
  return LM_SRC + s * a->pitch + 16 * ((i + r + a->salign) % 4);
}

static inline uint32_t lm_dst(const bulk_arg_t* a) {
  return LM_SRC + num_src_slots(a) * a->pitch;
}

static inline uint32_t dst_off(const bulk_arg_t* a, uint32_t r, uint32_t i) {
  const uint32_t s = a->mode == MODE_RACE ? i : dst_slot(a->mode, r, i, a->copies);
  return lm_dst(a) + s * a->pitch + 16 * ((3 * i + r + a->dalign) % 4);
}

static inline uint32_t lm_ham(const bulk_arg_t* a) {
  return lm_dst(a) + num_dst_slots(a) * a->pitch;
}

// Bytes that land in rank q.
static inline uint32_t incoming(const bulk_arg_t* a, uint32_t q) {
  uint32_t n = 0;
  for (uint32_t s = 0; s < a->k; ++s) {
    if (dst_rank(a->mode, s, a->k) != q)
      continue;
    for (uint32_t i = 0; i < a->copies; ++i)
      n += copy_size(a, s, i);
  }
  return n;
}

#endif
