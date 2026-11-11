#ifndef _FA4_DS_EXCHANGE_COMMON_H_
#define _FA4_DS_EXCHANGE_COMMON_H_

#include <stdint.h>
#include <vx_cuda_fp.h>

// Tile shape of the 2-CTA backward step: 128 Q rows per tile pair, 128 KV
// rows per CTA (256 per pair), head dimension 128. CTA r owns Q columns
// [64r, 64r + 64) of dQ and sends the other 64 columns of its dS rows to
// the peer, 128 x 64 halves = 16 KB.
constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int HEAD_D = 128;
constexpr int HALF_M = TILE_M / 2;
constexpr int XCHG_ELEMS = TILE_N * HALF_M;
constexpr uint32_t XCHG_BYTES = XCHG_ELEMS * sizeof(__half);
constexpr uint32_t kBlockThreads = 256;
constexpr uint32_t kClusterSize = 2;

// The block's dynamic shared memory, member for member as the original.
struct __attribute__((aligned(16))) SharedStorage {
  __half sdS[2 * XCHG_ELEMS];   // dS for the own 64 Q columns, KV rows 0..255 (rows 128r.. from CTA r)
  __half sdS_xchg[XCHG_ELEMS];  // the peer's 64 Q columns of the own KV rows
  __half sK[TILE_N * HEAD_D];   // the own 128 K rows
  float dQ_acc[HALF_M * HEAD_D];
  uint64_t dS_cluster_full_mbar;
  uint64_t dS_cluster_empty_mbar;
  uint64_t dS_cluster_leader_mbar;
};

// dS_in: [256 KV rows][128 Q columns] f16, CTA r reading rows 128r..;
// K_in: [256][128] f16; dQ_out: [128][128] f32, accumulated with atomics;
// smid[r]: the core of rank r.
typedef struct {
  uint64_t dS_in;
  uint64_t K_in;
  uint64_t dQ_out;
  uint64_t smid;
} kernel_arg_t;

VX_CUDA_CLUSTER_DIMS(flash_bwd_2cta_ds_exchange, 2, 1, 1);

#endif
