#ifndef _ATOMIC_FP_COMMON_H_
#define _ATOMIC_FP_COMMON_H_

#include <stdint.h>

#define SPACE_GLOBAL  0
#define SPACE_LOCAL   1   // the block's own local memory
#define SPACE_REMOTE  2   // rank + 1 of the cluster, through the DLMEM window

#define MODE_EDGE     0
#define MODE_HIST     1
#define MODE_CONTEND  2
#define MODE_ALLPUSH  3
#define MODE_F16LOOP  4

// edge: how a record runs its op.
#define IMPL_HW       0   // the FPAMO instruction (op, width), encoded here
#define IMPL_RED      1   // the same with rd = x0
#define IMPL_HDR      2   // a vx_atomic.h function, by form

// vx_atomic.h forms run by IMPL_HDR records (and by hist/contend/f16loop).
#define FORM_ADD_F16        0
#define FORM_ADD_BF16       1
#define FORM_CAS_B16        2
#define FORM_ADD_F32        3
#define FORM_ADD_F64        4
#define FORM_ADD_F16X2      5
#define FORM_ADD_BF16X2     6
#define FORM_INC_U32        7
#define FORM_DEC_U32        8
#define FORM_ADD_V2_F32     9
#define FORM_ADD_V4_F32     10
#define FORM_ADD_V8_F16     11
#define FORM_ADD_V8_BF16    12
#define FORM_MIN_V8_F16     13
#define FORM_MAX_V8_BF16    14
#define FORM_ADD_V4_F16X2   15
#define FORM_MAX_V4_BF16X2  16
#define FORM_RED_ADD_V4_F32 17
#define FORM_MIN_V2_F16     18
#define FORM_MAX_V2_BF16    19

// Which vx_atomic.h space variant a form runs as.
#define SP_GENERIC    0
#define SP_GLOBAL     1   // vx_atom_global_*
#define SP_SHARED     2   // vx_atom_shared_* (.shared::cta)
#define SP_CLUSTER    3   // vx_atom_shared_cluster_*

// One edge record: the 16-byte slot image before the op (the operand at
// `offset`, guard bytes around it), the operand(s), and for cas the comparand.
// Vector forms use the whole slot from offset 0.
typedef struct {
  uint32_t impl, op, width, form;
  uint32_t offset, sp;
  uint64_t old[2];
  uint64_t rhs[2];
  uint64_t cmp;
} rec_t;

// What the device saw: the returned value(s), the slot after the op, and
// fflags before (bits 8..12) and after (bits 0..4) it.
typedef struct {
  uint64_t ret[2];
  uint64_t mem[2];
  uint64_t fflags;
} res_t;

#define SLOT_BYTES    16
#define HIST_BINS     16
#define CONTEND_WORDS 2

typedef struct {
  uint64_t rec_addr;   // edge: records
  uint64_t res_addr;   // edge: per block, one res_t per record
  uint64_t out_addr;   // hist bins, contend word and cycles, allpush vectors, per block
  uint64_t glob_addr;  // global targets
  uint32_t mode, space, num_recs, k;
  uint32_t form, iters, m, seed;
  uint32_t red, sp;
  uint64_t operand;
} arg_t;

// Per-block words of out_addr.
#define OUT_CYCLES    0    // contend: cycles of thread 0's loop (two words)
#define OUT_RAN       2
#define OUT_RANK      3
#define OUT_CORE      4
#define OUT_DATA      8    // bins (as 64-bit words) or the allpush vector
#define OUT_WORDS(m)  (OUT_DATA + 2 * ((m) > HIST_BINS ? (m) : HIST_BINS))

static inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u;
  h ^= (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

// hist: thread t of block b on iteration j adds hist_val() to bin hist_bin().
static inline uint32_t hist_hash(uint32_t seed, uint32_t b, uint32_t t, uint32_t j) {
  return hash3(seed, (b << 16) | t, j);
}
static inline uint32_t hist_bin(uint32_t h) { return h >> 28; }
// Integer-valued, so every order of the adds gives the same exact sum.
static inline uint32_t hist_val(uint32_t h, uint32_t lane, uint32_t maxv) {
  return 1 + ((h >> (8 + 4 * lane)) % maxv);
}

// allpush: element e of rank r's vector.
static inline uint32_t allpush_val(uint32_t r, uint32_t e) { return (e % 7) + 2 * r + 1; }

// f16loop: elements of the global 16-bit vector (four words).
#define F16LOOP_ELEMS 8

#endif
