#ifndef _CUDA_API_COMMON_H_
#define _CUDA_API_COMMON_H_

#include <stdint.h>
#include <dxa_tmap.h>

// Kernel argument, one layout for every kernel. The maps come first: the
// argument buffer is 64-byte aligned, and a map must be.
typedef struct {
  vx_tensor_map_t map[2];  // ptx: a 2D tile map (box 16 x 4 of u32) and a 1D map
  uint64_t out;            // per-thread results, `stride` uint64 words each
  uint64_t aux;            // the mode's global buffer
  uint64_t in;             // the mode's inputs
  uint64_t dump;           // per-block copy of local memory, `lmem` bytes each
  uint32_t stride;
  uint32_t lmem;           // the launch's dynamic shared memory
  uint32_t sub;            // the mode's variant
  uint32_t rounds;
  uint32_t n;
  uint32_t pad[3];
} __attribute__((aligned(64))) cuda_arg_t;

#define WARP 32

static inline uint32_t mix32(uint32_t a, uint32_t b) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

static inline uint64_t mix64(uint32_t a, uint32_t b) {
  return ((uint64_t)mix32(a, b + 0x5000u) << 32) | mix32(a, b);
}

// ── cg ────────────────────────────────────────────────────────────────
#define CG_WORDS 64
#define CG_BLOCK_X 32
#define CG_BLOCK_Y 2
#define CG_BLOCK_Z 2
#define CG_LMEM 256

// ── warp ──────────────────────────────────────────────────────────────
#define W_WARPS     2
#define W_NSHFL     4      // idx, up, down, xor
#define W_NWIDTH    6      // 1, 2, 4, 8, 16, 32
#define W_NDELTA    6      // 0, 1, 3, width - 1, width, 31
#define W_NTYPE     5      // u32, u64, float, double, __half2
#define W_SHFL      0
#define W_VOTE      (W_SHFL + W_NSHFL * W_NWIDTH * W_NDELTA * W_NTYPE)
#define W_NVCASE    4      // odd lanes, lanes < 5, lane 7, every lane
#define W_VOTE_N    6      // ballot, any, all, activemask, all(true), any(false)
#define W_ELECT     (W_VOTE + W_NVCASE * W_VOTE_N)
#define W_NELECT    4      // full mask, odd lanes, lanes 4-7, elect_one_sync
#define W_RED       (W_ELECT + W_NELECT)
#define W_NROP      9      // add u, add s, min u, min s, max u, max s, and, or, xor
#define W_NRCASE    3      // full mask, lanes % 3 != 0 in a branch, signed extremes
#define W_MATCH     (W_RED + W_NROP * W_NRCASE)
#define W_NMCASE    3      // all equal, all distinct, 3 groups
#define W_MATCH_N   3      // any, all, pred
#define W_LANE      (W_MATCH + 2 * W_NMCASE * W_MATCH_N)
#define W_LANE_N    6      // eq, lt, le, gt, ge, laneid
#define W_TILE      (W_LANE + W_LANE_N)
#define W_TILE_N    9      // shfl, shfl_up, shfl_down, shfl_xor, any, all, ballot, thread_rank, default-width shfl
#define W_WORDS     (W_TILE + W_TILE_N)

static inline uint32_t w_width(uint32_t i) { return 1u << i; }

static inline uint32_t w_delta(uint32_t i, uint32_t width) {
  switch (i) {
  case 0: return 0;
  case 1: return 1;
  case 2: return 3;
  case 3: return width - 1;
  case 4: return width;
  default: return 31;
  }
}

// Lane values of each shuffle type (floats finite: bit 30 / 62 clear).
static inline uint64_t w_value(uint32_t type, uint32_t t) {
  switch (type) {
  case 0: return mix32(t, 1);
  case 1: return mix64(t, 2);
  case 2: return mix32(t, 3) & 0xBFFFFFFFu;
  case 3: return mix64(t, 4) & 0xBFFFFFFFFFFFFFFFull;
  default: return mix32(t, 5) & 0xBFFFBFFFu;
  }
}

static inline uint32_t w_vcase_mask(uint32_t c) {
  return c == 0 ? 0xAAAAAAAAu : c == 1 ? 0x1Fu : c == 2 ? 0x80u : 0xFFFFFFFFu;
}

static inline uint32_t w_pred(uint32_t c, uint32_t t) { return mix32(t, 40 + c) & 1; }

static inline uint32_t w_rmask(uint32_t c) {
  if (c != 1)
    return 0xFFFFFFFFu;
  uint32_t m = 0;
  for (uint32_t l = 0; l < WARP; ++l)
    m |= (l % 3 != 0) ? (1u << l) : 0;
  return m;
}

static inline uint32_t w_rvalue(uint32_t c, uint32_t t) {
  if (c == 2) {
    static const uint32_t ext[6] = {0x80000000u, 0x7FFFFFFFu, 0xFFFFFFFFu, 0u, 1u, 0x80000001u};
    return ext[(t % WARP) % 6];
  }
  return mix32(t, 60 + c);
}

static inline uint64_t w_key(uint32_t wide, uint32_t c, uint32_t lane) {
  const uint64_t k = c == 0 ? 7 : c == 1 ? lane + 100 : lane % 3;
  // 64-bit keys differ only in their upper half.
  return wide ? (k << 33) | 0x1234u : (uint64_t)(uint32_t)(k * 0x01010101u + 9);
}

// ── atomic ────────────────────────────────────────────────────────────
// Op x type targets, each a uint64 slot in global memory (G), the block's
// own local memory (L) and the peer block's local memory through the
// cluster window (W). Then the _block and _system forms once each, in G.
enum {
  A_ADD_S32, A_ADD_U32, A_ADD_U64, A_ADD_F32, A_ADD_F64, A_ADD_F16, A_ADD_F16X2, A_ADD_BF16, A_ADD_BF16X2,
  A_SUB_S32, A_SUB_U32,
  A_EXCH_S32, A_EXCH_U32, A_EXCH_U64, A_EXCH_F32,
  A_MIN_S32, A_MIN_U32, A_MIN_S64, A_MIN_U64,
  A_MAX_S32, A_MAX_U32, A_MAX_S64, A_MAX_U64,
  A_INC_U32, A_DEC_U32,
  A_AND_S32, A_AND_U32, A_AND_U64,
  A_OR_S32, A_OR_U32, A_OR_U64,
  A_XOR_S32, A_XOR_U32, A_XOR_U64,
  A_CAS_S32, A_CAS_U32, A_CAS_U64, A_CAS_U16, A_CAS_U16H,
  A_NUM_OPS
};
// _block then _system: add, sub, exch, min, max, inc, dec, and, or, xor, cas (u32).
#define A_NSCOPE_OPS 11
#define A_SCOPED     (3 * A_NUM_OPS)
#define A_TARGETS    (A_SCOPED + 2 * A_NSCOPE_OPS)
#define A_WARPS      2
#define A_INC_LIMIT  50u
#define A_DEC_LIMIT  50u
#define A_LMEM       (2 * A_NUM_OPS * 8)  // own targets, then the peer's

// The value thread t (rank in its block, `blk` its block) applies to target
// `tg` of op `op`. CAS: the compare is the target's initial value.
static inline uint64_t a_value(uint32_t op, uint32_t tg, uint32_t blk, uint32_t t) {
  const uint32_t g = blk * 64 + t;
  const uint32_t h = mix32(g, 1000 + tg);
  switch (op) {
  case A_ADD_S32: case A_ADD_U32: return 1 + (g * 37) % 101;
  case A_ADD_U64: return (1ull << 33) + g + 1;
  case A_ADD_F32: { float f = (float)(1 + g % 4); uint32_t b; __builtin_memcpy(&b, &f, 4); return b; }
  case A_ADD_F64: { double d = (g + 1) * 0.5; uint64_t b; __builtin_memcpy(&b, &d, 8); return b; }
  case A_ADD_F16: return (g & 1) ? 0x4000u : 0x3C00u;            // 2.0 or 1.0
  case A_ADD_F16X2: return ((g & 1) ? 0x4000u : 0x3C00u) | (0x3C00u << 16);
  case A_ADD_BF16: return (g & 1) ? 0x4000u : 0x3F80u;           // 2.0 or 1.0
  case A_ADD_BF16X2: return ((g & 1) ? 0x4000u : 0x3F80u) | (0x3F80u << 16);
  case A_SUB_S32: case A_SUB_U32: return 1 + g % 7;
  case A_EXCH_S32: case A_EXCH_U32: return 1000 + g * 3;
  case A_EXCH_U64: return ((uint64_t)(g + 1) << 36) | (g * 5);
  case A_EXCH_F32: { float f = (g + 1) * 1.25f; uint32_t b; __builtin_memcpy(&b, &f, 4); return b; }
  case A_MIN_S32: case A_MAX_S32: return (uint32_t)((int32_t)(h % 200001) - 100000);
  case A_MIN_U32: case A_MAX_U32: return h;
  case A_MIN_S64: case A_MAX_S64: return (uint64_t)((int64_t)(mix64(g, tg) % 2000000001ull) - 1000000000) * 4096;
  case A_MIN_U64: case A_MAX_U64: return mix64(g, tg);
  case A_INC_U32: return A_INC_LIMIT;
  case A_DEC_U32: return A_DEC_LIMIT;
  case A_AND_S32: case A_AND_U32: return ~(1u << (g % 32));
  case A_AND_U64: return ~(1ull << (g % 64));
  case A_OR_S32: case A_OR_U32: return 1u << (g % 32);
  case A_OR_U64: return 1ull << (g % 64);
  case A_XOR_S32: case A_XOR_U32: return h;
  case A_XOR_U64: return mix64(g, tg + 7);
  case A_CAS_U16: case A_CAS_U16H: return (g + 1) & 0xFFFFu;
  default: return g + 1;  // CAS
  }
}

// Initial value of target `tg` (op `op`).
static inline uint64_t a_init(uint32_t op, uint32_t tg) {
  switch (op) {
  case A_SUB_S32: case A_SUB_U32: return 1000000;
  case A_EXCH_S32: case A_EXCH_U32: return 7;
  case A_EXCH_U64: return 0x77777777ull;
  case A_EXCH_F32: return 0x3F800000u;  // 1.0
  case A_MIN_S32: return 0x7FFFFFF0u;
  case A_MIN_U32: return 0xFFFFFFF0u;
  case A_MIN_S64: return 0x7FFFFFFFFFFFFFF0ull;
  case A_MIN_U64: return 0xFFFFFFFFFFFFFFF0ull;
  case A_MAX_S32: return 0x80000010u;
  case A_MAX_S64: return 0x8000000000000010ull;
  case A_INC_U32: return 0;
  case A_DEC_U32: return 20;
  case A_AND_S32: case A_AND_U32: return 0xFFFFFFFFu;
  case A_AND_U64: return ~0ull;
  case A_XOR_S32: case A_XOR_U32: return mix32(tg, 77);
  case A_XOR_U64: return mix64(tg, 77);
  case A_CAS_S32: case A_CAS_U32: return 5555;
  case A_CAS_U64: return 0x123456789ull;
  case A_CAS_U16: return 0xBEEF;
  case A_CAS_U16H: return 0xCAFE;
  default: return 0;
  }
}

// The slot's first value: A_CAS_U16H works on bytes 2-3 of its slot (an
// address 2 mod 4), with a canary in bytes 0-1.
#define A_U16H_LOW 0x1234u
static inline uint64_t a_slot_init(uint32_t op, uint32_t tg) {
  return op == A_CAS_U16H ? (a_init(op, tg) << 16) | A_U16H_LOW : a_init(op, tg);
}

// The op of target `tg` (the scoped targets use the u32 form of their op).
static inline uint32_t a_op(uint32_t tg) {
  if (tg < A_SCOPED)
    return tg % A_NUM_OPS;
  static const uint8_t k[A_NSCOPE_OPS] = {A_ADD_U32, A_SUB_U32, A_EXCH_U32, A_MIN_U32, A_MAX_U32, A_INC_U32,
                                          A_DEC_U32, A_AND_U32, A_OR_U32, A_XOR_U32, A_CAS_U32};
  return k[(tg - A_SCOPED) % A_NSCOPE_OPS];
}

// ── barrier ───────────────────────────────────────────────────────────
#define B_WARPS    4
#define B_THREADS  (B_WARPS * WARP)
#define B_PHASES   8
#define B_DROP     4         // the phase where warp 3 arrives and drops
#define B_LM_BAR   0         // three barriers: phases, tx, memcpy_async
#define B_LM_DATA  64        // B_PHASES x B_THREADS words
#define B_LM_TX    (B_LM_DATA + B_PHASES * B_THREADS * 4)  // 3 tx copies, 4096 bytes apart
#define B_LM_MC    (B_LM_TX + 3 * 4096)                    // memcpy_async: aligned 1024, unaligned 100, sized 256
#define B_LM_G     (B_LM_MC + 4096)                    // group form, per-lane bulk, byte copy
#define B_LMEM     (B_LM_G + 4096)
#define B_NTX      3
#define B_WORDS    (B_PHASES + 2)
static inline uint32_t b_tx_size(uint32_t i) { return i == 0 ? 16 : i == 1 ? 528 : 2048; }
static inline uint32_t b_data(uint32_t p, uint32_t t) { return mix32(p * 1000 + t, 99); }
#define B_MC_ALIGNED   1024
#define B_MC_UNALIGNED 100
#define B_MC_SIZED     256
#define B_MC_DST_UN    (B_LM_MC + 1024 + 4)   // 4 mod 16
#define B_MC_SRC_UN    (8192 + 12)            // global source offset, 12 mod 16
#define B_MC_DST_SZ    (B_LM_MC + 2048)
#define B_MC_SRC_SZ    16384
// Part 8: cuda::memcpy_async(block, ...) aligned (bulk) and unaligned; lanes
// 0-3 of warp 0 each an aligned 64-byte thread-form copy; thread 5 an odd copy.
#define B_G_AL      0          // offsets in B_LM_G
#define B_G_AL_N    512
#define B_G_AL_SRC  20480
#define B_G_UN      1028       // 4 mod 16
#define B_G_UN_N    100
#define B_G_UN_SRC  (24576 + 12)
#define B_G_PL      2048       // + 64 * lane
#define B_G_PL_N    64
#define B_G_PL_SRC  28672
#define B_G_ODD     3073       // odd
#define B_G_ODD_N   37
#define B_G_ODD_SRC 30001

// ── ptx ───────────────────────────────────────────────────────────────
// A cluster of 2 blocks of 2 warps. The 2D u32 tensors are 64 x X_TH words
// with 256-byte rows; the box is 16 x 4 words (256 bytes).
#define X_WARPS   2
#define X_THREADS (X_WARPS * WARP)
#define X_LMEM    8192
#define X_NMB     12
#define X_TW      64
#define X_TH      16
#define X_BX      16
#define X_BY      4
#define X_TILE    (X_BX * X_BY * 4)
// local memory
#define XL_MB     0
#define XL_G2S    256      // 512: bulk loads (shared::cta then shared::cluster destination)
#define XL_TENS   1024     // tensor load
#define XL_MC     1536     // multicast bulk load
#define XL_TMC    2048     // multicast tensor load
#define XL_S2C    2560     // the peer's shared::cta -> shared::cluster copy
#define XL_SRC    3072     // 1024: the block's source words
#define XL_ST     4096     // 256: the peer's st.async
#define XL_RED    4352     // 64: the peer's red.async (11 u32, then 2 u64 at +48)
#define XL_RC     4608     // 256: the peer's cp.reduce.async.bulk.shared::cluster add
#define XL_MAP    5120     // a map built field by field from zeros
#define XL_TR     5376     // tensor load through that map
#define XL_TR2    5632     // tensor load after a global-map replace
#define XL_DUMP   6144
// global input buffer
#define XG_G2S    0x100    // + block * 512
#define XG_MC     0x2000
#define XG_PF     0x3000
#define XG_T0     0x8000   // tensor of map[0]
#define XG_T2     0xC000   // tensor of the field-built map
#define XG_T3     0x10000  // tensor after the global replace
#define XG_IN     0x14000
// global aux buffer
#define XA_STORE  0        // + block * 512: bulk store
#define XA_RED    1024     // 256: both blocks' bulk add
#define XA_TENS   2048     // tensor of map[1], 64 x 8 words
#define XA_MAP    4096     // the field-built map, copied by cp_fenceproxy
#define XA_FENCE  4352     // thread 0: a checksum of the bulk store read back
#define XA_SIZE   8192
#define X_TX_ST   (256 + 11 * WARP * 4 + 2 * WARP * 8)
#define X_WORDS   16
#define X_INC_LIMIT 20u
#define X_DEC_LIMIT 20u
static inline uint32_t x_src(uint32_t b, uint32_t w) { return mix32(b * 1024 + w, 0x33); }
static inline uint32_t x_in(uint32_t w) { return mix32(w, 0x77); }
static inline uint32_t x_aux(uint32_t w) { return mix32(w, 0x99); }
static inline uint32_t x_rc_init(uint32_t w) { return 1000 + w; }
// red.async targets: the owner's initial words, the writer lane's operand.
static inline uint64_t x_red_init(uint32_t k) {
  static const uint64_t v[13] = {100, 0xFFFFFFF0u, 5, 0, 3, 1000, (uint32_t)-1000, (uint32_t)-7, 0xFFFFFFFFu, 0,
                                 0x12345678u, 1ull << 40, (uint64_t)-(1ll << 40)};
  return v[k];
}
static inline uint64_t x_red_value(uint32_t k, uint32_t w, uint32_t l) {
  const uint32_t h = mix32(w * 64 + l, 200 + k);
  switch (k) {
  case 0:  return l + 1;
  case 3:  return X_INC_LIMIT;
  case 4:  return X_DEC_LIMIT;
  case 5: case 6: return (uint32_t)((int32_t)(h % 5000) - 2500);
  case 7:  return (uint32_t)((int32_t)l - 16);
  case 8:  return ~(1u << (l % 8));
  case 9:  return 1u << (l % 16);
  case 11: return (uint64_t)(l + 1) << 20;
  case 12: return (uint64_t)-(int64_t)(l * 3);
  default: return h;
  }
}

// ── sync ──────────────────────────────────────────────────────────────
#define S_WARPS   4
#define S_THREADS (S_WARPS * WARP)
#define S_NPRED   4          // all true, all false, one thread, half
#define S_ROUNDS  3
#define S_SWROUNDS 4
#define S_WORDS   (S_ROUNDS * S_NPRED * 3 + 2 * S_SWROUNDS)
#define S_RESERVE 16      // vx_cuda.h's __syncthreads_count counter, at the top of the request
#define S_LMEM    (1024 + S_SWROUNDS * S_THREADS * 4 + S_RESERVE)
static inline uint32_t s_pred(uint32_t k, uint32_t r, uint32_t t) {
  switch (k) {
  case 0: return 1;
  case 1: return 0;
  case 2: return t == 77u - r;
  default: return ((t ^ r) & 1) != 0;
  }
}
static inline uint32_t s_sw_value(uint32_t r, uint32_t t, uint32_t space) { return mix32(r * 512 + t, 7 + space); }

// The same with 8 bytes of the kernel's own data (a cuda::barrier at offset
// 0, used after the block reductions): the request is 8 + S_RESERVE.
#define S8_LMEM   (8 + S_RESERVE)
#define S8_WORDS  (S_ROUNDS * S_NPRED * 3 + 2)

// ── warp, exited lanes ────────────────────────────────────────────────
// Threads 48-63 of a 64-thread block return at once: warp 1 keeps lanes 0-15.
#define WT_THREADS 48
#define WT_RED     0           // W_NROP reductions with the full mask
#define WT_MATCH   (WT_RED + W_NROP)   // any 32, any 64, all-equal, pred, all-distinct, pred
#define WT_VOTE    (WT_MATCH + 6)      // ballot, activemask, shfl idx, shfl xor 1, shfl down 8
#define WT_WORDS   (WT_VOTE + 5)

// ── types ─────────────────────────────────────────────────────────────
#define T_INPUTS 512
#define T_WARPS  4

// ── launch ────────────────────────────────────────────────────────────
VX_CUDA_CLUSTER_DIMS(launch_k, 2, 1, 1);
VX_CUDA_CLUSTER_DIMS(ptx_mbar_k, 2, 1, 1);
VX_CUDA_CLUSTER_DIMS(ptx_copy_k, 2, 1, 1);
VX_CUDA_CLUSTER_DIMS(ptx_async_k, 2, 1, 1);
VX_CUDA_CLUSTER_DIMS(ptx_global_k, 2, 1, 1);
#define L_WORDS 8   // per block: nctaid x, y, z, ctarank, is_explicit, dynamic smem, threads, ran

// ── launch: a templated kernel and the smoke use of the other names ──
#define N_THREADS 64
#define N_WORDS   24
#define N_LMEM    256
static inline uint32_t n_word(uint32_t t) { return mix32(t, 0x4E) | 1u; }

// ── perf (-p, the parity cell) ────────────────────────────────────────
#define P_WARPS  4
#define P_ROUNDS 48
#define P_WORDS  4

#endif
