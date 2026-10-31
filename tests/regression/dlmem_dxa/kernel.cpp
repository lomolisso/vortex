#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dlmem.h>
#include <vx_dxa.h>
#include <vx_mbar.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

// Window pointers for both: a block of the cluster, the issuer's own included.
INL void issue_cluster(uint32_t ndim, uint64_t* mbar, void* smem, const uint32_t* c) {
  switch (ndim) {
  case 1:  vx_dxa_issue_1d_mbar_cluster(DESC_SLOT, mbar, smem, c[0]); break;
  case 2:  vx_dxa_issue_2d_mbar_cluster(DESC_SLOT, mbar, smem, c[0], c[1]); break;
  case 3:  vx_dxa_issue_3d_mbar_cluster(DESC_SLOT, mbar, smem, c[0], c[1], c[2]); break;
  case 4:  vx_dxa_issue_4d_mbar_cluster(DESC_SLOT, mbar, smem, c[0], c[1], c[2], c[3]); break;
  default: vx_dxa_issue_5d_mbar_cluster(DESC_SLOT, mbar, smem, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

// The control path: the public local-memory form.
INL void issue_local(uint32_t ndim, uint64_t* mbar, void* smem, const uint32_t* c) {
  switch (ndim) {
  case 1:  vx_dxa_issue_1d_mbar(DESC_SLOT, mbar, smem, c[0]); break;
  case 2:  vx_dxa_issue_2d_mbar(DESC_SLOT, mbar, smem, c[0], c[1]); break;
  case 3:  vx_dxa_issue_3d_mbar(DESC_SLOT, mbar, smem, c[0], c[1], c[2]); break;
  case 4:  vx_dxa_issue_4d_mbar(DESC_SLOT, mbar, smem, c[0], c[1], c[2], c[3]); break;
  default: vx_dxa_issue_5d_mbar(DESC_SLOT, mbar, smem, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

INL void tile_coords(const dxa_arg_t* arg, uint32_t t, uint32_t* c) {
  for (uint32_t d = 0; d < MAX_DIMS; ++d) {
    c[d] = (t % arg->grids[d]) * arg->tiles[d];
    t /= arg->grids[d];
  }
}

INL void idle_cycles(uint64_t n) {
  const uint64_t t0 = vx_rdcycle();
  while (vx_rdcycle() - t0 < n) {}
}

INL void record(const dxa_arg_t* arg, uint32_t b, uint32_t rank) {
  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * REC_WORDS;
  rec[REC_BLOCK] = b;
  rec[REC_CORE]  = vx_core_id();
  rec[REC_SLOT]  = get_local_group_id();
  rec[REC_RANK]  = rank;
  rec[REC_SIZE]  = vx_cluster_num_blocks();
  __atomic_fetch_add(&rec[REC_RAN], 1u, __ATOMIC_RELAXED);
}

} // namespace

// Each block fills its local memory with canaries, initializes its mbarrier
// and arrives on it declaring the bytes of the copies it will receive, all
// before the cluster barrier. Then warp 0 issues its copies into the
// destination block's tile slots, completing on that block's mbarrier;
// every thread waits for its own block's phase, and the block dumps its
// whole local memory for the host's byte-exact check.
__kernel void copy_k(dxa_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t nthr = blockDim.x;
  const uint32_t b    = blockIdx.x;
  const uint32_t k    = arg->k;
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t r    = cl.block_rank();
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto lw   = reinterpret_cast<volatile uint32_t*>(lm);
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);
  const uint32_t words = arg->lmem_bytes / sizeof(uint32_t);

  if (tid == 0)
    record(arg, b, r);
  for (uint32_t w = 2 + tid; w < words; w += nthr)
    lw[w] = canary_word(b, w);
  if (tid == 0) {
    vx_mbar_init(mbar, 1);
    vx_mbar_arrive_tx(mbar, incoming(arg->mode, r, k, arg->copies) * arg->box);
  }
  cl.sync();

  if (get_sub_group_id() == 0) {
    const uint32_t q = dst_rank(arg->mode, r, k);
    for (uint32_t i = 0; i < arg->copies; ++i) {
      uint32_t c[MAX_DIMS];
      tile_coords(arg, src_tile(b, i, arg->copies, arg->ntiles), c);
      void* tile = lm + LM_TILE + dst_slot(arg->mode, r, i, arg->copies) * arg->tstride;
      if (arg->mode == MODE_SELF) {
        issue_local(arg->ndim, mbar, tile, c);
      } else {
        issue_cluster(arg->ndim, cl.map_shared_rank(mbar, q), cl.map_shared_rank(tile, q), c);
      }
    }
  }

  uint32_t err = 0, tries = 1;
  while (!vx_mbar_try_wait_parity(mbar, 0)) {
    if (++tries > WAIT_LIMIT) {
      err = 1;
      break;
    }
  }

  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr + (uint64_t)b * arg->lmem_bytes);
  for (uint32_t w = tid; w < words; w += nthr)
    dump[w] = lw[w];
  cl.sync();

  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + (b * nthr + tid) * ERR_WORDS;
  out[ERR_WAIT]  = err;
  out[ERR_TRIES] = tries;
  out[ERR_DONE]  = 1;
}

// Runs the simulator must stop. Rank 0 issues one 1-D copy; rank 1 stays
// alive meanwhile, except in -x retired, where it exits at once and rank 0
// issues into it only after lingering.
__kernel void neg_k(dxa_arg_t* __UNIFORM__ arg) {
  const uint32_t tid = threadIdx.x;
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t r = cl.block_rank();
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);
  auto tile = lm + LM_TILE;

  if (tid == 0)
    record(arg, blockIdx.x, r);
  if (arg->neg == NEG_RETIRED && r != 0)
    return;
  if (tid == 0)
    vx_mbar_init(mbar, 1);
  __syncthreads();
  if (arg->neg == NEG_RETIRED)
    idle_cycles(arg->linger);
  if (r == 0 && get_sub_group_id() == 0) {
    const uint32_t c[MAX_DIMS] = {};
    auto peer_mbar = cl.map_shared_rank(mbar, 1);
    auto peer_tile = cl.map_shared_rank(tile, 1);
    switch (arg->neg) {
    case NEG_RANK: {
      auto bad = reinterpret_cast<uint8_t*>((uintptr_t)VX_MEM_DLMEM_BASE_ADDR + arg->neg_win);
      issue_cluster(1, reinterpret_cast<uint64_t*>(bad + LM_MBAR), bad + LM_TILE, c);
      break;
    }
    case NEG_MBAR:
      issue_cluster(1, cl.map_shared_rank(mbar, 0), peer_tile, c);
      break;
    case NEG_LEGACY: {
      vortex::barrier bar(1);
      bar.expect_tx(1);
      vx_dxa_issue_1d_wg(DESC_SLOT, bar.id(), peer_tile, 0);
      break;
    }
    case NEG_MCAST:
      vx_dxa_issue_1d_multicast_wg(DESC_SLOT, __vx_dxa_mbar_payload_cluster(peer_mbar), peer_tile, 0, 0x3);
      break;
    case NEG_BOX:
      issue_cluster(1, peer_mbar, cl.map_shared_rank(lm, 1) + arg->neg_win, c);
      break;
    default:
      issue_cluster(1, peer_mbar, peer_tile, c);
      break;
    }
  }
  idle_cycles(arg->linger);
  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + (blockIdx.x * blockDim.x + tid) * ERR_WORDS;
  out[ERR_DONE] = 1;
}
