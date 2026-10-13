#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// One argument block for every kernel. The grid is 1-D in clusters of k x 1 x 1,
// so block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t err_addr;   // ERR_WORDS per thread
  uint64_t rec_addr;   // REC_WORDS per block
  uint64_t glob_addr;  // mixed_k: two words per thread; parity_k: one sum per thread
  uint32_t k;
  uint32_t lmem_bytes;
  uint32_t seed;
  uint32_t rounds;     // pingpong_k and raw_k: rounds; parity_k: loads per thread
} load_arg_t;

#define TEST_ALL      1   // own fill, sync, every peer's slice through the window
#define TEST_PP       2   // ping-pong of polled window loads between two blocks
#define TEST_MIXED    3   // global, own and window lanes in one load
#define TEST_RAW      4   // a thread's window store, then its load of the same bytes
#define TEST_PARITY   5
#define TEST_SETUP    7   // the launch itself: the cluster size
#define NUM_TESTS     4

#define ERR_COUNT     0   // mismatches; ~0 if the thread never ran
#define ERR_TEST      1   // the first mismatch: its test,
#define ERR_INDEX     2   // its index (peer or round in bits 31:24),
#define ERR_GOT       4   // the value read (two words),
#define ERR_WANT      6   // and the value expected (two words)
#define ERR_WORDS     8

#define REC_RAN       0
#define REC_RANK      1   // vx_cluster_block_rank()
#define REC_SIZE      2   // vx_cluster_num_blocks()
#define REC_CORE      3
#define REC_SLOT      4   // VX_CSR_CTA_ID
#define REC_WORDS     5

// Doublewords a test checks per slice: all of them up to this many, then a spread sample.
#define CHECK_WORDS   64

// Polls of a word before the kernel reports the peer as stuck.
#define SPIN_LIMIT    (1u << 16)

static inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u;
  h ^= (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

// Word i of test `test` in cluster c; q < 64 names the writer, the target or both.
static inline uint32_t pattern(uint32_t seed, uint32_t test, uint32_t c, uint32_t q, uint32_t i) {
  return hash3(seed ^ (test << 24), c * 64u + q, i);
}

static inline uint64_t pattern64(uint32_t seed, uint32_t test, uint32_t c, uint32_t q, uint32_t i) {
  return ((uint64_t)pattern(seed, test, c, q, i) << 32)
       | pattern(seed ^ 0xA5A5A5A5u, test, c, q, i);
}

// Word w of mixed_k's global buffer, which the host fills.
static inline uint32_t glob_word(uint32_t seed, uint32_t w) {
  return hash3(seed ^ 0x61000000u, 0xFFFFu, w);
}

// Sampled doubleword m of an n-doubleword slice, for m < check_count(n): a
// spread over the slice that also moves the low bits, and its last doubleword.
static inline uint32_t check_count(uint32_t n) {
  return (n < CHECK_WORDS) ? n : CHECK_WORDS;
}
static inline uint32_t check_index(uint32_t m, uint32_t n) {
  const uint32_t M = check_count(n), s = n / M;
  return (m + 1 == M) ? (n - 1) : (m * s + m % s);
}

// mixed_k: lane t loads from global memory (0), its own local memory (1) or the
// window on another rank (2), at doubleword mixed_cell() of the slice.
static inline uint32_t mixed_kind(uint32_t t) {
  return t % 3u;
}
static inline uint32_t mixed_peer(uint32_t rank, uint32_t t, uint32_t k) {
  return (rank + 1 + (t / 3u) % (k - 1)) % k;
}
static inline uint32_t mixed_cell(uint32_t rank, uint32_t t, uint32_t n8) {
  return (t * 5u + rank) % n8;
}

// parity_k: thread t of rank p loads word (j * nt + t) % n of rank p + 1's slice.
static inline uint32_t parity_fill(uint32_t seed, uint32_t c, uint32_t q, uint32_t e) {
  return pattern(seed, TEST_PARITY, c, q, e);
}

#endif
