#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// The grid is 1-D in clusters of k x 1 x 1, so block b is rank b % k of cluster b / k.
typedef struct {
  uint64_t err_addr;     // ERR_WORDS per thread
  uint64_t rec_addr;     // REC_WORDS per block
  uint64_t dump_addr;    // dump_words() per block: its cells and counters at the end
  uint64_t dxa_addr;     // the DXA copies' source, DXA_TILES tiles of DXA_WORDS words
  uint32_t k;
  uint32_t lmem_bytes;
  uint32_t seed;
  uint32_t phases;
  uint32_t steps;        // operations per thread and phase
  uint32_t num_threads;  // threads per warp
  uint32_t flags;
} stress_arg_t;

#define FLAG_DXA      1u  // a DXA copy into the block's own local memory each phase
#define FLAG_RDXA     2u  // and, after each cluster.sync() phase end, one into the next rank

#define TEST_STEP     1   // a load or atomic returned other than the thread's model
#define TEST_SIG      2   // the previous rank's signature after a phase end
#define TEST_RING     3   // the ring flag never came
#define TEST_DXA      4   // a word of the phase's DXA copy
#define TEST_RDXA     5   // a word of the previous rank's DXA copy, or (index ~0) its mbarrier phase never completed
#define TEST_SETUP    7   // the launch itself: the cluster size

#define ERR_COUNT     0   // mismatches; ~0 if the thread never ran
#define ERR_TEST      1   // the first mismatch: its test,
#define ERR_INDEX     2   // its phase (bits 31:24) and step,
#define ERR_GOT       4   // the value read (two words),
#define ERR_WANT      6   // and the value expected (two words)
#define ERR_WORDS     8

#define REC_RAN       0
#define REC_RANK      1   // vx_cluster_block_rank()
#define REC_SIZE      2   // vx_cluster_num_blocks()
#define REC_CORE      3
#define REC_SLOT      4   // VX_CSR_CTA_ID
#define REC_WORDS     5

// Doubleword cells each thread owns, spread over the slices of its cluster.
#define ST_CELLS      4
// Shared word counters per slice that every thread adds into.
#define ST_CTRS       8
// Signature slots per slice: a previous rank runs at most K - 1 phases ahead.
#define SIG_SLOTS     8
#define DXA_WORDS     64
#define DXA_TILES     64

// Polls of a ring flag before the kernel reports the previous rank as stuck.
#define SPIN_LIMIT    (1u << 16)
// try_waits on a remote copy's mbarrier phase before the kernel gives it up.
#define RDXA_TRY_LIMIT 64

// The slice, in bytes: the cells of every thread of the cluster, the signatures
// from the previous rank, the counters, a ring flag per warp, then the DXA tile.
static inline uint32_t cells_off(void) { return 0; }
static inline uint32_t sig_off(uint32_t k, uint32_t nt) { return 8 * k * nt * ST_CELLS; }
static inline uint32_t ctr_off(uint32_t k, uint32_t nt) { return sig_off(k, nt) + 4 * SIG_SLOTS * nt; }
static inline uint32_t flag_off(uint32_t k, uint32_t nt) { return ctr_off(k, nt) + 4 * ST_CTRS; }
static inline uint32_t dxa_off(uint32_t k, uint32_t nt, uint32_t warps) {
  return (flag_off(k, nt) + 4 * warps + 63) & ~63u;
}
// FLAG_RDXA: the tile the previous rank copies in, then its mbarrier.
static inline uint32_t rdxa_off(uint32_t k, uint32_t nt, uint32_t warps) {
  return dxa_off(k, nt, warps) + 4 * DXA_WORDS;
}
static inline uint32_t rmbar_off(uint32_t k, uint32_t nt, uint32_t warps) {
  return rdxa_off(k, nt, warps) + 4 * DXA_WORDS;
}
static inline uint32_t lmem_needed(uint32_t k, uint32_t nt, uint32_t warps, uint32_t flags) {
  return (flags & FLAG_RDXA) ? rmbar_off(k, nt, warps) + 8
       : dxa_off(k, nt, warps) + ((flags & FLAG_DXA) ? 4 * DXA_WORDS : 0);
}
// Cell j of thread t of rank r: slice (r + 1 + j) % k, doubleword (r * nt + t) * ST_CELLS + j.
static inline uint32_t cell_rank(uint32_t r, uint32_t j, uint32_t k) { return (r + 1 + j) % k; }
static inline uint32_t cell_index(uint32_t r, uint32_t t, uint32_t j, uint32_t nt) {
  return (r * nt + t) * ST_CELLS + j;
}
// What each block dumps: its cells, then its counters, in words.
static inline uint32_t dump_words(uint32_t k, uint32_t nt) {
  return 2 * k * nt * ST_CELLS + ST_CTRS;
}

static inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u;
  h ^= (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

// The operation of step s of phase p, the same for every lane of a warp, so
// each one is a single instruction; its lanes differ only in cell, byte and value.
static inline uint32_t st_warp_rnd(uint32_t seed, uint32_t c, uint32_t p, uint32_t s,
                                   uint32_t r, uint32_t w) {
  return hash3(seed ^ 0x57000000u, (c * 64u + r) * 64u + w, (p << 16) | s);
}
static inline uint32_t st_lane_rnd(uint32_t seed, uint32_t c, uint32_t p, uint32_t s,
                                   uint32_t r, uint32_t t) {
  return hash3(seed ^ 0x4C000000u, c * 64u + r, (t << 20) | (p << 12) | s);
}
static inline uint64_t st_value(uint32_t seed, uint32_t c, uint32_t p, uint32_t s, uint32_t r, uint32_t t) {
  return ((uint64_t)hash3(seed ^ 0x56000000u, c * 64u + r, (t << 20) | (p << 12) | s) << 32)
       | hash3(seed ^ 0x76000000u, c * 64u + r, (t << 20) | (p << 12) | s);
}
// How phase p ends, the same for the whole cluster: cluster.sync(), the
// fence-free arrive/wait, or a ring of flags (each warp waits for its
// counterpart on the previous rank only).
#define END_SYNC      0
#define END_RELAXED   1
#define END_RING      2
static inline uint32_t st_end(uint32_t seed, uint32_t c, uint32_t p) {
  return hash3(seed ^ 0x45000000u, c, p) % 3u;
}
static inline uint32_t st_sig(uint32_t seed, uint32_t c, uint32_t p, uint32_t r, uint32_t t) {
  return hash3(seed ^ 0x53000000u, c * 64u + r, (p << 16) | t);
}
static inline uint32_t dxa_word(uint32_t seed, uint32_t i) {
  return hash3(seed ^ 0x44000000u, 0xD7Au, i);
}
// The tile block b copies in phase p: into itself, and (FLAG_RDXA) into the next rank.
static inline uint32_t dxa_tile(uint32_t p, uint32_t blocks, uint32_t b) {
  return (p * blocks + b) % DXA_TILES;
}
static inline uint32_t rdxa_tile(uint32_t p, uint32_t blocks, uint32_t b) {
  return (p * blocks + b + DXA_TILES / 2 + 1) % DXA_TILES;
}
// Whether phase p has remote copies: the first, and those after a
// cluster.sync(), which orders the receiver's checks of the last one before
// the next one is issued.
static inline uint32_t rdxa_phase(uint32_t seed, uint32_t c, uint32_t p) {
  return p == 0 || st_end(seed, c, p - 1) == END_SYNC;
}

#define OPC_ST        0   // store of `size` bytes at `off`
#define OPC_LD        1   // doubleword load
#define OPC_LDN       2   // signed word, unsigned half or unsigned byte load at `off`
#define OPC_AMOADD_D  3
#define OPC_AMOXOR_W  4   // on the word at `off`
#define OPC_AMOSWAP_D 5
#define OPC_AMOMAXU_W 6   // on the word at `off`
#define OPC_CTR       7   // amoadd.w of `v` into counter ctr_idx of rank ctr_rank
#define OPC_FENCE     8
#define OPC_STLD      9   // doubleword store, then a load of it

typedef struct {
  uint32_t op, j, off, size, ctr_rank, ctr_idx;
  uint64_t v, want;
} st_step_t;

static inline uint64_t st_field(uint64_t x, uint32_t off, uint32_t size) {
  return (size == 8) ? x : ((x >> (8 * off)) & ((1ull << (8 * size)) - 1));
}
static inline uint64_t st_merge(uint64_t x, uint32_t off, uint32_t size, uint64_t v) {
  if (size == 8)
    return v;
  const uint64_t mask = ((1ull << (8 * size)) - 1) << (8 * off);
  return (x & ~mask) | ((v << (8 * off)) & mask);
}
static inline uint64_t st_sext32(uint64_t x) {
  return (uint64_t)(int64_t)(int32_t)(uint32_t)x;
}

// One step of a thread: what to do, what it must return, and the cells after it.
static inline void st_plan(st_step_t* st, uint32_t u, uint32_t l, uint64_t v, uint32_t k, uint64_t* cells) {
  const uint32_t op = u % 16u, sel = u >> 4;
  st->j = l % ST_CELLS;
  st->v = v;
  st->off = 0;
  st->size = 8;
  st->want = 0;
  st->ctr_rank = st->ctr_idx = 0;
  uint64_t* x = &cells[st->j];
  if (op < 4) {
    st->op = OPC_ST;
    st->size = 1u << (sel % 4u);
    st->off = ((l >> 8) % (8 / st->size)) * st->size;
    *x = st_merge(*x, st->off, st->size, v);
  } else if (op < 7) {
    st->op = OPC_LD;
    st->want = *x;
  } else if (op == 7) {
    st->op = OPC_LDN;
    st->size = 4u >> (sel % 3u);
    st->off = ((l >> 8) % (8 / st->size)) * st->size;
    st->want = st_field(*x, st->off, st->size);
    if (st->size == 4)
      st->want = st_sext32(st->want);
  } else if (op < 10) {
    st->op = OPC_AMOADD_D;
    st->want = *x;
    *x += v;
  } else if (op == 10 || op == 12) {
    st->op = (op == 10) ? OPC_AMOXOR_W : OPC_AMOMAXU_W;
    st->size = 4;
    st->off = ((l >> 8) & 1u) * 4;
    const uint32_t old = (uint32_t)st_field(*x, st->off, 4);
    const uint32_t rhs = (uint32_t)v;
    st->want = st_sext32(old);
    *x = st_merge(*x, st->off, 4, (op == 10) ? (old ^ rhs) : (old > rhs ? old : rhs));
  } else if (op == 11) {
    st->op = OPC_AMOSWAP_D;
    st->want = *x;
    *x = v;
  } else if (op == 13) {
    st->op = OPC_CTR;
    st->ctr_rank = (l >> 8) % k;
    st->ctr_idx = (l >> 4) % ST_CTRS;
    st->v = (l & 0xFu) + 1;
  } else if (op == 14) {
    st->op = OPC_FENCE;
  } else {
    st->op = OPC_STLD;
    st->want = v;
    *x = v;
  }
}

#endif
