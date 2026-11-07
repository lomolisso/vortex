#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <dxa_meta.h>

#define MODE_RING   0   // rank r copies into rank (r + 1) % K
#define MODE_MANY   1   // ranks 1.. copy into rank 0, back to back, and rank 0 into rank 1
#define MODE_SELF   2   // every rank copies into its own local memory: must stop the simulator (strict PTX)
#define MODE_RACE   3   // rounds of plain stores to the source, then a copy of it into the next rank
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
#define NEG_SELF    9   // a copy into the issuer's own block through local pointers
#define NEG_MALIGN  10  // an mbarrier 4 bytes off its doubleword
#define NEG_RCSELF  11  // a cluster reduction into the issuer's own block
#define NEG_RCCLU   12  // a cluster reduction by a local pointer into a block of the core outside the cluster
#define NUM_NEGS    13

// -g: copies from global memory (cp.async.bulk.shared::cluster.global)
#define GK_NONE     0
#define GK_CTA      1   // every block loads into its own local memory through local pointers
#define GK_WIN      2   // rank r loads into rank (r + 1) % K through the window
#define GK_MC       3   // rank 0 multicasts every copy into the ranks of the mask
#define GK_PF       4   // GK_CTA, each copy's source prefetched into the L2 first
#define GK_RC       5   // cluster reductions: rank r reduces into rank (r + 1) % K through the window
#define GK_RCWIN    6   // GK_RC with the cluster on one core: the window destination is on the issuer's core
#define GK_RCLOC    7   // GK_RCWIN through local pointers into the destination block and its mbarrier
#define GK_RCP      8   // timed cluster reductions: rounds of one pair, -s bytes into rank (r + 1) % K
#define NUM_GKINDS  9

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
  uint64_t gsrc_addr;    // -g: the global source buffer
  uint32_t gkind;
  uint32_t mask;         // -g mc: the ctaMask; -g rc, rcwin, rcloc, rcp: the reduction pairs (bit i = rc_pair(i))
  uint32_t pf_big[2];    // -g pf -P: bulk prefetches of the source buffer larger than a local memory
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
#define ERR_CYCLES  3   // -g rcp, thread 0 of each block: cycles of the timed rounds
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

// The destination rank of rank r's copies as the kernels issue them: never r
// itself (a copy into the issuer's own block is an error), except in -m self.
// copy_k keeps dst_rank for ring and self, so their code is unchanged.
static inline uint32_t strict_dst(uint32_t mode, uint32_t r, uint32_t k) {
  return (mode == MODE_RING || mode == MODE_RACE) ? (r + 1) % k : mode == MODE_MANY ? (r == 0 ? 1 : 0) : r;
}

// Race: the bytes rank q receives per round, from rank q - 1.
static inline uint32_t race_in(const bulk_arg_t* a, uint32_t q) {
  return is_hog(a, (q + a->k - 1) % a->k) ? a->hog : a->size;
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

// Bytes that land in rank q under strict_dst.
static inline uint32_t strict_incoming(const bulk_arg_t* a, uint32_t q) {
  uint32_t n = 0;
  for (uint32_t s = 0; s < a->k; ++s) {
    if (strict_dst(a->mode, s, a->k) != q)
      continue;
    for (uint32_t i = 0; i < a->copies; ++i)
      n += copy_size(a, s, i);
  }
  return n;
}

// -g: copy i is the same size on every rank; the global mix spans 16 B to 4 KB.
static inline uint32_t g_size(const bulk_arg_t* a, uint32_t i) {
  return a->size ? a->size : (i % 4 == 0 ? 16 : i % 4 == 1 ? 4096 : i % 4 == 2 ? 720 : 48);
}

#define G_MIX_MAX 4096

// Bytes between destination slots: each copy sits 0-48 bytes into its slot
// and at least 64 canary bytes follow it.
static inline uint32_t g_slot(const bulk_arg_t* a, uint32_t i) {
  return (g_size(a, i) + 48 + 64 + 63) / 64 * 64;
}

// Destination of copy i of sender rank r, the same in every receiver.
static inline uint32_t g_dst_off(const bulk_arg_t* a, uint32_t r, uint32_t i) {
  uint32_t o = LM_SRC;
  for (uint32_t j = 0; j < i; ++j)
    o += g_slot(a, j);
  return o + 16 * ((3 * i + r + a->dalign) % 4);
}

static inline uint32_t g_end(const bulk_arg_t* a) {
  return g_dst_off(a, 0, a->copies);
}

// Bytes between the sources of block b's copies in the global buffer.
static inline uint32_t g_pitch(const bulk_arg_t* a) {
  uint32_t m = 0;
  for (uint32_t i = 0; i < a->copies; ++i)
    m = g_size(a, i) > m ? g_size(a, i) : m;
  return (m + 48 + 63) / 64 * 64;
}

static inline uint64_t g_src_off(const bulk_arg_t* a, uint32_t b, uint32_t i) {
  return (uint64_t)(b * a->copies + i) * g_pitch(a) + 16 * ((i + b + a->salign) % 4);
}

// Byte o of the global source buffer.
static inline uint8_t g_byte(uint64_t o) {
  return word_byte(mix32(0x6B0000u, (uint32_t)(o / 4)), (uint32_t)o);
}

static inline uint32_t g_dst_rank(const bulk_arg_t* a, uint32_t r) {
  return a->gkind == GK_WIN ? (r + 1) % a->k : r;
}

// Bytes that land in rank q.
static inline uint32_t g_incoming(const bulk_arg_t* a, uint32_t q) {
  uint32_t per = 0;
  for (uint32_t i = 0; i < a->copies; ++i)
    per += g_size(a, i);
  if (a->gkind == GK_MC)
    return ((a->mask >> q) & 1u) ? per : 0;
  return per;
}

// -g rc, rcwin, rcloc, rcp: the PTX cluster reductions (op, type) in a fixed order.
#define RC_PAIRS 12

static inline uint32_t rc_op(uint32_t i) {
  const uint32_t op[RC_PAIRS] = {DXA_RED_AND, DXA_RED_OR, DXA_RED_XOR, DXA_RED_ADD, DXA_RED_ADD, DXA_RED_MIN,
                                 DXA_RED_MIN, DXA_RED_MAX, DXA_RED_MAX, DXA_RED_INC, DXA_RED_DEC, DXA_RED_ADD};
  return op[i];
}

static inline uint32_t rc_type(uint32_t i) {
  const uint32_t ty[RC_PAIRS] = {DXA_RT_U32, DXA_RT_U32, DXA_RT_U32, DXA_RT_U32, DXA_RT_S32, DXA_RT_U32,
                                 DXA_RT_S32, DXA_RT_U32, DXA_RT_S32, DXA_RT_U32, DXA_RT_U32, DXA_RT_U64};
  return ty[i];
}

static inline uint32_t rc_elem(uint32_t i) {
  return rc_type(i) == DXA_RT_U64 ? 8 : 4;
}

// The j-th selected pair of the mask, and how many there are.
static inline uint32_t rc_count(const bulk_arg_t* a) {
  uint32_t n = 0;
  for (uint32_t i = 0; i < RC_PAIRS; ++i)
    n += (a->mask >> i) & 1u;
  return n;
}

static inline uint32_t rc_pair(const bulk_arg_t* a, uint32_t j) {
  for (uint32_t i = 0; i < RC_PAIRS; ++i) {
    if (((a->mask >> i) & 1u) && j-- == 0)
      return i;
  }
  return 0;
}

static inline uint32_t rc_size(const bulk_arg_t* a, uint32_t j) {
  return a->size ? a->size : mix_size(j);
}

static inline uint32_t rc_slot(const bulk_arg_t* a, uint32_t j) {
  return (rc_size(a, j) + 48 + 64 + 63) / 64 * 64;
}

static inline uint32_t rc_src_off(const bulk_arg_t* a, uint32_t j) {
  uint32_t o = LM_SRC;
  for (uint32_t i = 0; i < j; ++i)
    o += rc_slot(a, i);
  return o + 16 * ((j + a->salign) % 4);
}

static inline uint32_t rc_dst_base(const bulk_arg_t* a) {
  uint32_t o = LM_SRC;
  for (uint32_t i = 0; i < rc_count(a); ++i)
    o += rc_slot(a, i);
  return o;
}

static inline uint32_t rc_dst_off(const bulk_arg_t* a, uint32_t j) {
  uint32_t o = rc_dst_base(a);
  for (uint32_t i = 0; i < j; ++i)
    o += rc_slot(a, i);
  return o + 16 * ((3 * j + a->dalign) % 4);
}

static inline uint32_t rc_end(const bulk_arg_t* a) {
  return rc_dst_off(a, rc_count(a));
}

// Bytes each block receives (from rank r - 1).
static inline uint32_t rc_incoming(const bulk_arg_t* a) {
  uint32_t n = 0;
  for (uint32_t j = 0; j < rc_count(a); ++j)
    n += rc_size(a, j);
  return n;
}

// Element e of selected pair j's source (role 0) or destination (role 1) in
// block b: small, extreme or random, at the element's width.
static inline uint64_t rc_val(uint32_t b, uint32_t role, uint32_t j, uint32_t e, uint32_t bytes) {
  const uint32_t h = mix32(0x3C0000u + b * 64 + role * 32 + j, e);
  const uint64_t mask = bytes == 8 ? ~0ull : 0xFFFFFFFFull;
  const uint64_t top = bytes == 8 ? (1ull << 63) : (1ull << 31);
  switch (h & 7u) {
  case 0:
  case 1:  return (h >> 8) & 15u;
  case 2: {
    const uint64_t ext[4] = {0, top - 1, top, mask};
    return ext[(h >> 8) & 3u];
  }
  default: return ((((uint64_t)mix32(h, 3u)) << 32) | h) & mask;
  }
}

// -g rcp: the source and the destination of the timed reductions.
static inline uint32_t rcp_src(const bulk_arg_t* a) {
  (void)a;
  return LM_SRC;
}

static inline uint32_t rcp_dst(const bulk_arg_t* a) {
  return LM_SRC + (a->size + 63) / 64 * 64 + 64;
}

static inline uint32_t rcp_end(const bulk_arg_t* a) {
  return rcp_dst(a) + a->size + 64;
}

#endif
