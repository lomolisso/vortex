#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// One argument block for every kernel. The grid is 1-D in clusters of k x 1 x 1,
// so block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t err_addr;     // ERR_WORDS per thread
  uint64_t rec_addr;     // REC_WORDS per block
  uint64_t glob_addr;    // per-thread words, then per-block words (see each kernel)
  uint32_t k;
  uint32_t lmem_bytes;
  uint32_t seed;
  uint32_t iters;        // operations per thread (per warp for the lock)
  uint32_t num_threads;  // threads per warp
} atomic_arg_t;

#define TEST_HIST     1   // amoadd.w/.d histogram into rank 0, and amoadd tickets
#define TEST_LOCK     2   // an amoswap spinlock around a window read-modify-write
#define TEST_OPS      3   // amo{add,xor,and,or,min,max,minu,maxu}.{w,d} into every slice
#define TEST_CAS      4   // compare-and-swap counters (Zacas)
#define TEST_PARITY   5
#define TEST_SETUP    7   // the launch itself: the cluster size
#define NUM_TESTS     4

#define ERR_COUNT     0   // mismatches; ~0 if the thread never ran
#define ERR_TEST      1   // the first mismatch: its test,
#define ERR_INDEX     2   // its index,
#define ERR_GOT       4   // the value read (two words),
#define ERR_WANT      6   // and the value expected (two words)
#define ERR_WORDS     8

#define REC_RAN       0
#define REC_RANK      1   // vx_cluster_block_rank()
#define REC_SIZE      2   // vx_cluster_num_blocks()
#define REC_CORE      3
#define REC_SLOT      4   // VX_CSR_CTA_ID
#define REC_WORDS     5

// Attempts at a lock or a compare-and-swap before the kernel reports it stuck.
#define SPIN_LIMIT    (1u << 16)

// hist_k: word bins in rank 0 (words 0-15), doubleword bins after them
// (doublewords 8-23), and two tickets in rank k - 1.
#define HIST_BINS     16
#define HIST_W        0
#define HIST_D        8    // doublewords
#define TICKET_W      48   // words
#define TICKET_D      25   // doublewords
#define HIST_LMEM     208
// Per thread: the word ticket as returned (sign-extended), then the doubleword one.
#define HIST_TWORDS   4
// Per block: rank 0's bins and rank k - 1's tickets, as left after the sync.
#define HIST_BWORDS   (HIST_BINS + 2 * HIST_BINS + 4)

// lock_k: the lock in rank k - 1, the counter in rank 0, the holder in rank 1 % k.
#define LOCK_WORD     0
#define LOCK_HOLDER   1
#define LOCK_COUNT    1    // doubleword
#define LOCK_LMEM     64
#define LOCK_BWORDS   2

// ops_k: per slice, eight words then eight doublewords, one per operation.
#define OPS_W         0
#define OPS_D         4    // doublewords
#define OPS_LMEM      96
#define OPS_BWORDS    (8 + 16)
#define OP_ADD        0
#define OP_XOR        1
#define OP_AND        2
#define OP_OR         3
#define OP_MIN        4
#define OP_MAX        5
#define OP_MINU       6
#define OP_MAXU       7

// cas_k: a word counter in rank 0 and a doubleword counter in rank k - 1.
#define CAS_WORD      0
#define CAS_DWORD     1
#define CAS_LMEM      64
#define CAS_BWORDS    4

// parity_k: thread t owns word t and doubleword nt + t of the next rank's slice.
#define PARITY_TWORDS 2
#define PARITY_BWORDS(nt) (3 * (nt))

static inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u;
  h ^= (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

static inline uint32_t pattern(uint32_t seed, uint32_t test, uint32_t c, uint32_t q, uint32_t i) {
  return hash3(seed ^ (test << 24), c * 64u + q, i);
}

// hist_k: thread t of rank r adds hist_w() and hist_d() to bin hist_bin() on iteration j.
static inline uint32_t hist_hash(uint32_t seed, uint32_t c, uint32_t r, uint32_t t, uint32_t j) {
  return pattern(seed, TEST_HIST, c, r, (t << 16) | j);
}
static inline uint32_t hist_bin(uint32_t h) { return h >> 28; }
static inline uint32_t hist_w(uint32_t h) { return (h & 0xFFu) + 1; }
static inline uint64_t hist_d(uint32_t h) { return ((uint64_t)h << 20) + 1; }

// The tickets start just below the sign bit of a word and the carry out of
// the low word of a doubleword, so the returned values cross both.
static inline uint32_t ticket_w0(uint32_t n) { return 0x7FFFFFFFu - n / 2; }
static inline uint64_t ticket_d0(uint32_t n) { return 0xFFFFFFFFull - n / 2; }

// ops_k: the initial value of operation o in rank q's slice, and the operand of
// thread t of rank r. AND clears one bit and OR sets one, so neither saturates.
static inline uint64_t ops_init(uint32_t seed, uint32_t c, uint32_t q, uint32_t o) {
  return ((uint64_t)pattern(seed, TEST_OPS, c, q, o) << 32) | pattern(seed, TEST_OPS, c, q, o | 0x100u);
}
static inline uint64_t ops_operand(uint32_t seed, uint32_t c, uint32_t r, uint32_t t,
                                   uint32_t q, uint32_t o, uint32_t d) {
  const uint32_t i = (t << 16) | (q << 8) | (d << 4) | o;
  const uint64_t v = ((uint64_t)pattern(seed, TEST_OPS, c, 32 + r, i) << 32)
                   | pattern(seed, TEST_OPS, c, 32 + r, i | 0x80u);
  const uint32_t bit = (uint32_t)(v % (d ? 64 : 32));
  if (o == OP_AND)
    return ~(1ull << bit);
  if (o == OP_OR)
    return 1ull << bit;
  return v;
}

static inline uint64_t ops_apply(uint32_t o, uint32_t d, uint64_t a, uint64_t b) {
  if (!d) {
    const uint32_t x = (uint32_t)a, y = (uint32_t)b;
    uint32_t z;
    switch (o) {
    case OP_ADD:  z = x + y; break;
    case OP_XOR:  z = x ^ y; break;
    case OP_AND:  z = x & y; break;
    case OP_OR:   z = x | y; break;
    case OP_MIN:  z = ((int32_t)x < (int32_t)y) ? x : y; break;
    case OP_MAX:  z = ((int32_t)x > (int32_t)y) ? x : y; break;
    case OP_MINU: z = (x < y) ? x : y; break;
    default:      z = (x > y) ? x : y; break;
    }
    return z;
  }
  switch (o) {
  case OP_ADD:  return a + b;
  case OP_XOR:  return a ^ b;
  case OP_AND:  return a & b;
  case OP_OR:   return a | b;
  case OP_MIN:  return ((int64_t)a < (int64_t)b) ? a : b;
  case OP_MAX:  return ((int64_t)a > (int64_t)b) ? a : b;
  case OP_MINU: return (a < b) ? a : b;
  default:      return (a > b) ? a : b;
  }
}

// parity_k: thread t adds j + 1 to its word and raises its doubleword to
// parity_d(j), folding every returned value into its checksum.
static inline uint64_t parity_d(uint32_t seed, uint32_t c, uint32_t r, uint32_t t, uint32_t j) {
  return ((uint64_t)pattern(seed, TEST_PARITY, c, r, (t << 16) | j) << 24) | j;
}

#endif
