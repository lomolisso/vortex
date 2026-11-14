#ifndef _CUDA_MATRIX_COMMON_H_
#define _CUDA_MATRIX_COMMON_H_

// Local-memory layout and argument block of the stmatrix / ldmatrix and named
// barrier test, shared by the host and the kernel.

#include <stdint.h>

constexpr uint32_t kWarps = 8;
constexpr uint32_t kThreads = kWarps * 32;

// Six stmatrix variants then six ldmatrix variants: x1, x2, x4, x1.trans,
// x2.trans, x4.trans. Each (variant, warp) owns a 1 KB region of 64 rows of
// 16 bytes; the row-address lanes point at distinct rows 0..55, the other
// lanes at rows 56..63, which must stay untouched.
constexpr uint32_t kVariants = 12;
constexpr uint32_t kRegion = 1024;
constexpr uint32_t kRows = kRegion / 16;
constexpr uint32_t kPoisonRow = 56;
constexpr uint32_t LM_MATRIX = 0;
// The DAK epilogue: sC, 64 (m) x 256 (n) halves, 128B-swizzled, stored by
// one warpgroup with stmatrix.x4.trans, 16 calls per thread.
constexpr uint32_t LM_DAK = LM_MATRIX + kVariants * kWarps * kRegion;   // 0x18000
constexpr uint32_t kDakCalls = 16;
// The DeepGEMM epilogue: D, 128 (m) x 192 (n) bf16 in three 64-column
// 128B-swizzled boxes, stored by two warpgroups with stmatrix.x2, 24 calls.
constexpr uint32_t LM_DG = LM_DAK + 64 * 256 * 2;                       // 0x20000
constexpr uint32_t kDgCalls = 24;
// Named-barrier mailboxes: one word per round.
constexpr uint32_t LM_BAR = LM_DG + 128 * 192 * 2;                      // 0x2C000
constexpr uint32_t LM_BYTES = LM_BAR + 128;

enum : uint32_t { PHASE_MATRIX = 1, PHASE_DAK = 2, PHASE_DG = 4, PHASE_BAR = 8, PHASE_ALL = 15 };

constexpr uint32_t kBarDelay = 3000;   // writer spin before its store

typedef struct {
  uint64_t init_addr;    // LM_BYTES of initial local memory
  uint64_t image_addr;   // LM_BYTES: local memory at the end
  uint64_t mat_addr;     // [variant][warp][lane]: row offset (uint32) then 4 registers (uint32)
  uint64_t ld_addr;      // [ld variant][warp][lane][4] registers read
  uint64_t dak_regs;     // [t < 128][64] packed f16 accumulators
  uint64_t dak_addr;     // [t][call] the row offset each lane supplied
  uint64_t dg_words;     // [t < 256][call][2] bf16x2 words
  uint64_t bar_addr;     // [round] the token the reader saw
  uint32_t bar_tokens;   // token base
  uint32_t phases;       // PHASE_* parts to run
} kernel_arg_t;

// One (variant, warp, lane) entry of mat_addr: row offset and registers.
typedef struct {
  uint32_t row;
  uint32_t r[4];
} mat_entry_t;

// Named-barrier rounds: barrier id, threads, writer warp, reader warp, and
// whether the writer arrives (bar.arrive) instead of syncing.
typedef struct {
  uint32_t id, nthreads, writer, reader, arrive;
} bar_round_t;

// Ids 1-15 pair two warps each (64 threads); 8 and 9 also run with all 256
// threads (DeepGEMM's), 13 and 15 with one warpgroup each (DAK's). The last
// three rounds mix bar.arrive (the writer) with bar.sync (the others), in
// both directions between warps 2 and 5, and one warp for three.
constexpr uint32_t kBarRounds = 22;
constexpr bar_round_t kBarRound[kBarRounds] = {
  {1, 64, 1, 4, 0},  {2, 64, 2, 5, 0},  {3, 64, 3, 6, 0},  {4, 64, 4, 7, 0},  {5, 64, 5, 0, 0},
  {6, 64, 6, 1, 0},  {7, 64, 7, 2, 0},  {8, 64, 0, 3, 0},  {9, 64, 1, 5, 0},  {10, 64, 2, 6, 0},
  {11, 64, 3, 7, 0}, {12, 64, 4, 0, 0}, {13, 64, 5, 1, 0}, {14, 64, 6, 2, 0}, {15, 64, 7, 3, 0},
  {8, 256, 0, 7, 0}, {9, 256, 6, 1, 0}, {13, 128, 0, 3, 0}, {15, 128, 4, 7, 0},
  {10, 64, 2, 5, 1}, {11, 64, 5, 2, 1}, {12, 128, 4, 7, 1},
};

#endif
