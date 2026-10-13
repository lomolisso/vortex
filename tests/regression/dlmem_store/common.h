#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// One argument block for every kernel. The grid is 1-D in clusters of k x 1 x 1,
// so block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t err_addr;   // ERR_WORDS per thread
  uint64_t rec_addr;   // REC_WORDS per block
  uint64_t glob_addr;  // parity_k: each block's slice, word for word
  uint32_t k;
  uint32_t lmem_bytes;
  uint32_t seed;
  uint32_t rounds;     // pc_k: producer/consumer rounds; parity_k: stores per thread
} store_arg_t;

#define TEST_A2A      1   // every block stores into every peer's slice, sync, own check
#define TEST_PC       2   // producer/consumer ring: data, vx_fence(), flag
#define TEST_PARITY   3
#define TEST_SETUP    7   // the launch itself: the cluster size
#define NUM_TESTS     2

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

// pc_k's slice: the flag the producer raises, the ack the consumer returns, then data.
#define PC_FLAG       0
#define PC_ACK        1
#define PC_DATA       2
#define PC_MAX_WORDS  128

// Polls of a flag before the kernel reports the peer as stuck.
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

// What a slice holds before the all-to-all stores.
#define TEST_FILL     0xF

// Sampled doubleword m of an n-doubleword slice, for m < check_count(n): a
// spread over the slice that also moves the low bits, and its last doubleword.
static inline uint32_t check_count(uint32_t n) {
  return (n < CHECK_WORDS) ? n : CHECK_WORDS;
}
static inline uint32_t check_index(uint32_t m, uint32_t n) {
  const uint32_t M = check_count(n), s = n / M;
  return (m + 1 == M) ? (n - 1) : (m * s + m % s);
}
// The m whose check_index is i, or ~0 for a doubleword no one samples.
static inline uint32_t check_slot(uint32_t i, uint32_t n) {
  const uint32_t M = check_count(n), s = n / M;
  if (i + 1 == n)
    return M - 1;
  const uint32_t m = i / s;
  return (m + 1 < M && check_index(m, n) == i) ? m : ~0u;
}

// a2a_k: sampled doubleword m of every slice is written by rank m % k, as one
// doubleword, two words, four halves or eight bytes by its iteration and target.
static inline uint32_t a2a_kind(uint32_t j, uint32_t q) {
  return (j + q) % 4u;
}

// parity_k's stores: thread t of rank p owns words [t * m, t * m + m) of rank
// p + 1's slice, m = n / nt, and writes word t * m + j % m on round j, in
// steps of PARITY_STEP rounds.
#define PARITY_STEP 8
static inline uint32_t parity_value(uint32_t seed, uint32_t c, uint32_t p, uint32_t j, uint32_t t) {
  return pattern(seed, TEST_PARITY, c, p, t) + j;
}

#endif
