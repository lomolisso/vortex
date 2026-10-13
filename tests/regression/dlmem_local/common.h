#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// One argument block for every kernel. The grid is 1-D in clusters of k x 1 x 1,
// so block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t err_addr;   // ERR_WORDS per thread
  uint64_t rec_addr;   // REC_WORDS per block
  uint64_t glob_addr;  // one word per thread (local_k, exit_k), two (parity_k), BURST_WORDS (burst_k)
  uint32_t k;
  uint32_t lmem_bytes;
  uint32_t seed;
  uint32_t tests;      // local_k: bit t-1 runs test t; neg_k: the mode
  uint32_t flags;
} local_arg_t;

#define FLAG_ATOMICS  1u  // the device has the A extension
#define FLAG_CBAR     2u  // sync with cluster_group::sync() instead of a local BAR

#define NUM_TESTS     6
#define TEST_OWN      1   // own write, sync, read every peer through the window
#define TEST_STORE    2   // stores into every peer through the window
#define TEST_AMO      3   // amoadd into rank 0 through the window
#define TEST_MIXED    4   // global, own and window lanes in one instruction
#define TEST_EDGE     5   // sub-word stores at the slice and line edges of each peer
#define TEST_SELF     6   // the self window aliases the plain pointer
#define TEST_SETUP    7   // the launch itself: the cluster size
#define TEST_EXIT     8   // the exit test's own words, through the self window
#define TEST_BURST    9   // the burst test's loads

#define ERR_COUNT     0   // mismatches; ~0 if the thread never ran
#define ERR_TEST      1   // the first mismatch: its test,
#define ERR_INDEX     2   // its word or byte index (peer rank in bits 31:24)
#define ERR_GOT       3   // and the value read
#define ERR_WORDS     4

#define REC_RAN       0
#define REC_RANK      1   // vx_cluster_block_rank()
#define REC_SIZE      2   // vx_cluster_num_blocks()
#define REC_CORE      3
#define REC_SLOT      4   // VX_CSR_CTA_ID
#define REC_WORDS     5

#define NEG_OOB       1   // rank 0 stores to rank 2 of a 2-block cluster

// Words a test touches per peer: every word up to this many, then a spread sample.
#define CHECK_WORDS   64

#define PARITY_ITERS  128

// The burst test: rounds of eight independent loads, each from its own line.
#define BURST_LOADS   8
#define BURST_ROUNDS  4
#define BURST_STRIDE  16  // words: one 64-byte line
#define BURST_WORDS   (BURST_ROUNDS * BURST_LOADS * BURST_STRIDE)

#define EXIT_MAGIC    0xEu

static inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u;
  h ^= (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

// Value of word (or byte offset) i in cluster c for test `test`; q < 64 names
// the writer, the target or both.
static inline uint32_t pattern(uint32_t seed, uint32_t test, uint32_t c, uint32_t q, uint32_t i) {
  return hash3(seed ^ (test << 24), c * 64u + q, i);
}

// Sampled word m of an n-word slice, for m < check_count(n); spreads the low bits too.
static inline uint32_t check_count(uint32_t n) {
  return (n < CHECK_WORDS) ? n : CHECK_WORDS;
}
static inline uint32_t check_index(uint32_t m, uint32_t n) {
  const uint32_t s = n / check_count(n);
  return (s <= 1) ? m : (m * s + m % s);
}

// Test 4: lane t of rank r goes to the global buffer (0), its own local memory
// (1) or the window on rank (r + t) % k (2). Its local cell is word r * nt + t,
// so a lane whose cell is past the slice goes global.
static inline uint32_t mixed_kind(uint32_t r, uint32_t t, uint32_t nt, uint32_t n) {
  return (r * nt + t < n) ? (t % 3u) : 0u;
}

// The exit test's word t of block b: magic in the top nibble, then b, then t.
static inline uint32_t exit_word(uint32_t b, uint32_t t) {
  return (EXIT_MAGIC << 28) | ((b & 0xFFFFFu) << 8) | (t & 0xFFu);
}

#endif
