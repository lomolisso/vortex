#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <dxa_meta.h>

// Job kinds, issued in order by warp 0.
#define J_BSTORE  0   // bulk store
#define J_TSTORE  1   // tensor store (tile or im2col by the descriptor)
#define J_BRED    2   // bulk global reduction
#define J_TRED    3   // tensor reduction
#define J_COMMIT  4   // commit_group
#define J_WAIT    5   // wait_group n
#define J_WAITR   6   // wait_group.read n
#define J_REWRITE 7   // local memory [src, src + size) takes the second pattern
#define J_RAW     8   // -x: n = 0 a STORE of kind 5, n = 1 a proxy fence with selector 3

// Kernels.
#define K_JOBS 0
#define K_I7A  1      // generic stores, fence.proxy.async.global, then a bulk load of them
#define K_I7B  2      // a line in the data cache, a bulk store over it, wait, fence, then a generic load
#define K_I7C  3      // as K_I7B, with the first 16 bytes stored generically and the rest by the bulk store
#define K_EXIT 4      // a bulk store, commit, wait_group.read 0, and the warp exits

typedef struct {
  uint32_t kind;
  uint32_t op;        // reductions: DXA_RED_*
  uint32_t type;      // reductions: DXA_RT_*
  uint32_t rank;
  uint32_t im2col;
  uint32_t slot;
  uint32_t src;       // local-memory byte offset
  uint32_t size;      // bulk bytes
  uint64_t gdst;
  int32_t  coord[5];
  uint32_t n;         // waits: N; J_RAW: which
  uint32_t group;     // bulk stores: the group they join
  uint32_t check;     // waits: generic-load the bulk stores of the groups this wait completes
} job_t;

// A typed source fill, applied after the pattern and before any job.
typedef struct {
  uint32_t src;
  uint32_t size;
  uint32_t type;
  uint32_t seed;
} fill_t;

typedef struct {
  uint64_t jobs_addr;   // job_t[num_jobs]
  uint64_t fills_addr;  // fill_t[num_fills]
  uint64_t err_addr;    // ERR_WORDS uint32 per thread
  uint64_t gaddr;       // i7a, i7b, i7c, exit: the global region
  uint32_t num_jobs;
  uint32_t num_fills;
  uint32_t lmem_bytes;
  uint32_t kernel;      // K_*
  uint32_t size;        // i7a, i7b, i7c, exit: bytes
  uint32_t neg;
  uint32_t linger;
  uint32_t repeats;     // -p: the jobs run this many times, each repeat one group; 0 = the checked run
} store_arg_t;

#define LM_MBAR 0
#define LM_DATA 128

#define ERR_DONE   0
#define ERR_BAD    1   // words a generic load found wrong (grp, i7a, i7b, i7c)
#define ERR_CYCLES 2   // -p, thread 0: cycles of the timed loop
#define ERR_WAIT   3   // i7a: an mbarrier phase that did not complete
#define ERR_WORDS  4

#define WAIT_LIMIT 256

static inline uint32_t mix32(uint32_t a, uint32_t b) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

// Local-memory word w before the fills, and after J_REWRITE.
static inline uint32_t lmem_word(uint32_t w)  { return mix32(0x5C0000u, w); }
static inline uint32_t lmem_word2(uint32_t w) { return mix32(0x2B0000u, w); }

// i7a: the new value of word w of the region.
static inline uint32_t new_word(uint32_t w) { return mix32(0x17A000u, w); }

static inline uint32_t type_size(uint32_t type) {
  return (type == DXA_RT_U64 || type == DXA_RT_S64 || type == DXA_RT_F64) ? 8
       : (type == DXA_RT_F16 || type == DXA_RT_BF16) ? 2 : 4;
}

// A float of eb exponent and mb mantissa bits: zeros, subnormals, infinities,
// NaNs one class in 16 each, else a normal number within 2^+-8. Kept out of
// line and branch-coded, so that the kernel and the host compute it the same way.
static __attribute__((noinline)) uint64_t fp_bits(uint32_t eb, uint32_t mb, uint32_t h) {
  const uint64_t mant = ((((uint64_t)mix32(h, 1u) << 32) | mix32(h, 2u)) & ((1ull << mb) - 1));
  const uint64_t emax = (1ull << eb) - 1;
  const uint32_t cls = h & 15u;
  uint64_t e = (emax >> 1) + ((h >> 4) & 15u) - 8, f = mant;
  if (cls == 0) {
    e = 0;
    f = 0;
  } else if (cls == 1) {
    e = 0;
    f = mant ? mant : 1;
  } else if (cls == 2) {
    e = emax;
    f = 0;
  } else if (cls == 3) {
    e = emax;
    f = mant ? mant : 1;
  }
  return ((uint64_t)(h >> 31) << (eb + mb)) | (e << mb) | f;
}

// Element i of a typed region: integers are small, extreme or random.
static inline uint64_t typed_val(uint32_t type, uint32_t seed, uint32_t i) {
  const uint32_t h = mix32(seed, i);
  uint32_t eb = 0, mb = 0;
  if (type == DXA_RT_F32) {
    eb = 8;
    mb = 23;
  } else if (type == DXA_RT_F64) {
    eb = 11;
    mb = 52;
  } else if (type == DXA_RT_F16) {
    eb = 5;
    mb = 10;
  } else if (type == DXA_RT_BF16) {
    eb = 8;
    mb = 7;
  }
  if (eb)
    return fp_bits(eb, mb, h);
  const uint32_t bits = 8 * type_size(type);
  const uint64_t mask = bits == 64 ? ~0ull : (1ull << bits) - 1;
  const uint64_t top = 1ull << (bits - 1);
  const uint32_t cls = h & 7u;
  if (cls < 2)
    return (h >> 8) & 15u;
  if (cls == 2) {
    const uint32_t x = (h >> 8) & 3u;
    return x == 0 ? 0 : x == 1 ? top - 1 : x == 2 ? top : mask;
  }
  return ((((uint64_t)mix32(h, 3u)) << 32) | h) & mask;
}

#endif
