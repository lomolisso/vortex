#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// The ops, by bit of the -o mask.
#define OP_ST32     0   // st.async.b32
#define OP_ST64     1   // st.async.b64
#define OP_RED32    2   // red.async.add.u32
#define OP_RED64    3   // red.async.add.u64
#define OP_V2B32    4   // st.async.v2.b32
#define OP_V4B32    5   // st.async.v4.b32
#define OP_V2B64    6   // st.async.v2.b64
#define OP_ADDS32   7   // red.async.add.s32
#define OP_MINS32   8   // red.async.min.s32
#define OP_MAXS32   9   // red.async.max.s32
#define OP_MINU32   10  // red.async.min.u32
#define OP_MAXU32   11  // red.async.max.u32
#define OP_AND      12  // red.async.and.b32
#define OP_OR       13  // red.async.or.b32
#define OP_XOR      14  // red.async.xor.b32
#define OP_INC      15  // red.async.inc.u32 (FPAMO)
#define OP_DEC      16  // red.async.dec.u32 (FPAMO)
#define NUM_OPS     17
#define OPS_ALL     ((1u << NUM_OPS) - 1)
#define OPS_FPAMO   ((1u << OP_INC) | (1u << OP_DEC))

#define NEG_NONE      0
#define NEG_LOCAL     1   // the destination is a local pointer, not a window address
#define NEG_MBAR      2   // the mbarrier is in another block than the destination
#define NEG_MBARLOCAL 3   // the mbarrier is a local pointer
#define NEG_RANK      4   // the destination and its mbarrier lie outside the cluster
#define NEG_OFF       5   // the destination lies past the end of the peer's block
#define NEG_SELF      6   // the destination is the issuer's own block, through its window address
#define NEG_ALIGN16   7   // a 16-byte st.async at an address 8 past a 16-byte boundary
#define NEG_ODDPAIR   8   // a 16-byte st.async whose data register is odd
#define NUM_NEGS      9

// The grid is 1-D in clusters of k x 1 x 1: block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t err_addr;     // ERR_WORDS uint32 per thread
  uint64_t rec_addr;     // REC_WORDS uint32 per block
  uint64_t dump_addr;    // lmem_bytes per block: its local memory at the end
  uint64_t want_addr;    // rounds x NUM_OPS uint64 per block: each counter after round p
  uint32_t neg;
  uint32_t k;
  uint32_t ops;          // bit o: op o runs
  uint32_t rounds;
  uint32_t elems;        // ops per thread, op and round
  uint32_t lmem_bytes;
  uint32_t linger;       // negatives: cycles each block stays alive around its issue
  uint32_t data_off;     // canary words from here on
  uint32_t mbar[NUM_OPS];  // byte offset of op o's mbarrier
  uint32_t area[NUM_OPS];  // byte offset of op o's store area, or of its counter
  uint64_t neg_win;      // -x rank: window offset of the bad block; -x off: offset past the peer's block
} sa_arg_t;

// The block's local memory (lm_layout()), for the enabled ops only: one
// mbarrier per op, one 8-byte counter word per reduction (a 32-bit counter in
// its low or high half, the other half a canary), then the store areas, each
// 16-byte aligned, of elems x threads elements, element i of thread t at
// i * threads + t, so the lanes of one instruction write consecutive
// elements. Everything from data_off on starts as canary words, so a stray
// write shows.
#define LM_TAIL     64

static inline uint32_t op_is_red(uint32_t o) {
  return o == OP_RED32 || o == OP_RED64 || o >= OP_ADDS32;
}

#define OPS_RED     ((1u << OP_RED32) | (1u << OP_RED64) | (OPS_ALL & ~((1u << OP_ADDS32) - 1)))
#define OPS_ST      (OPS_ALL & ~OPS_RED)

// Bytes one op of one lane carries, and so completes on its mbarrier.
static inline uint32_t op_bytes(uint32_t o) {
  switch (o) {
  case OP_ST64: case OP_RED64: case OP_V2B32: return 8;
  case OP_V4B32: case OP_V2B64: return 16;
  default: return 4;
  }
}

// A 16-byte op completes as two 8-byte halves.
static inline uint32_t op_cx_bytes(uint32_t o) {
  return op_bytes(o) > 8 ? 8 : op_bytes(o);
}

// 32-bit counters in the high half of their word, the 0xf0 byte lanes.
static inline uint32_t ctr_hi(uint32_t o) {
  return o == OP_ADDS32 || o == OP_MINS32 || o == OP_MINU32 || o == OP_AND || o == OP_XOR || o == OP_DEC;
}

static inline uint32_t align16(uint32_t x) {
  return (x + 15) / 16 * 16;
}

// Fills mbar[], area[] and *data_off; returns the local memory bytes a block needs.
static inline uint32_t lm_layout(uint32_t threads, uint32_t elems, uint32_t ops, uint32_t* mbar, uint32_t* area,
                                 uint32_t* data_off) {
  uint32_t off = 0;
  for (uint32_t o = 0; o < NUM_OPS; ++o) {
    mbar[o] = area[o] = 0;
    if ((ops >> o) & 1u) {
      mbar[o] = off;
      off += 8;
    }
  }
  for (uint32_t o = 0; o < NUM_OPS; ++o) {
    if (((ops >> o) & 1u) && op_is_red(o)) {
      area[o] = off + (ctr_hi(o) ? 4 : 0);
      off += 8;
    }
  }
  *data_off = off;
  for (uint32_t o = 0; o < NUM_OPS; ++o) {
    if (((ops >> o) & 1u) && !op_is_red(o)) {
      off = align16(off);
      area[o] = off;
      off += threads * elems * op_bytes(o);
    }
  }
  return align16(off) + LM_TAIL;
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
#define ERR_RED     3   // counters that did not hold the round's result right after the wait
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

static inline uint32_t op_key(uint32_t b, uint32_t t, uint32_t i) {
  return (b << 16) | (t << 6) | i;
}

// Word j of the element sender block b's thread t stores as element i of
// round p with store op o.
static inline uint32_t st_word(uint32_t o, uint32_t p, uint32_t b, uint32_t t, uint32_t i, uint32_t j) {
  if (o == OP_ST32 || o == OP_ST64) {
    return j ? mix32(0x5B0000u + p, op_key(b, t, i)) : mix32(0x5A0000u + p, op_key(b, t, i));
  }
  return mix32(mix32(0x5D0000u + (o << 8) + p, j), op_key(b, t, i));
}

// The operand sender block b's thread t gives a 32-bit reduction op o as
// element i of round p. min/max draw both signs, so signed and unsigned
// orders differ; inc/dec take a bound above every value their counter
// reaches, so neither wraps and the result is the count, in any order.
#define INC_BOUND   0x40000000u
#define DEC_INIT    0x40000000u

static inline uint32_t red_value(uint32_t o, uint32_t p, uint32_t b, uint32_t t, uint32_t i) {
  const uint32_t key = op_key(b, t, i);
  if (o == OP_RED32)
    return 0x01000193u * (1 + p + key);   // wraps within a round
  const uint32_t h = mix32(0x5C0000u + (o << 8) + p, key);
  switch (o) {
  case OP_AND: return ~(1u << (h & 31));
  case OP_OR:  return 1u << (h & 31);
  case OP_INC: case OP_DEC: return INC_BOUND + (h & 0xFFFFu);
  default:     return h;
  }
}

// Carries across bit 32.
static inline uint64_t red64_value(uint32_t p, uint32_t b, uint32_t t, uint32_t i) {
  return 0xFFFFFF00ull + (uint64_t)(op_key(b, t, i) + 1) * 0x10001ull + p;
}

static inline uint32_t red_init(uint32_t o) {
  switch (o) {
  case OP_ADDS32: return 0x7FFFFFF0u;
  case OP_MINS32: return 0x3FFFFFFFu;
  case OP_MAXS32: return 0xC0000000u;
  case OP_MINU32: return 0xF0000000u;
  case OP_MAXU32: return 0x00000010u;
  case OP_AND:    return 0xFFFFFFFFu;
  case OP_XOR:    return 0xA5A5A5A5u;
  case OP_DEC:    return DEC_INIT;
  default:        return 0;
  }
}

static inline uint32_t red_fold(uint32_t o, uint32_t acc, uint32_t v) {
  switch (o) {
  case OP_MINS32: return (int32_t)v < (int32_t)acc ? v : acc;
  case OP_MAXS32: return (int32_t)v > (int32_t)acc ? v : acc;
  case OP_MINU32: return v < acc ? v : acc;
  case OP_MAXU32: return v > acc ? v : acc;
  case OP_AND:    return acc & v;
  case OP_OR:     return acc | v;
  case OP_XOR:    return acc ^ v;
  case OP_INC:    return acc >= v ? 0 : acc + 1;
  case OP_DEC:    return (acc == 0 || acc > v) ? v : acc - 1;
  default:        return acc + v;
  }
}

// The initial 8-byte counter word of reduction o of block b, whose counter
// is at byte `off`: the counter's initial value, the other half a canary.
static inline uint64_t ctr_word_init(uint32_t b, uint32_t o, uint32_t off) {
  if (o == OP_RED64)
    return 0;
  const uint32_t w = (off & ~7u) / 4;
  const uint32_t lo = ctr_hi(o) ? canary_word(b, w) : red_init(o);
  const uint32_t hi = ctr_hi(o) ? red_init(o) : canary_word(b, w + 1);
  return ((uint64_t)hi << 32) | lo;
}

// Stores go from rank r to rank (r + 1) % k; reductions from rank 0 to rank
// 1 and from every other rank to rank 0. No op targets its own block.
static inline uint32_t st_dst(uint32_t r, uint32_t k) {
  return (r + 1) % k;
}

static inline uint32_t st_src(uint32_t r, uint32_t k) {
  return (r + k - 1) % k;
}

static inline uint32_t red_dst(uint32_t r) {
  return r == 0 ? 1 : 0;
}

// Ranks whose reductions go to rank q.
static inline uint32_t red_senders(uint32_t q, uint32_t k) {
  return q == 0 ? k - 1 : (q == 1 ? 1 : 0);
}

#endif
