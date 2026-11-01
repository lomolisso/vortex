// dxa_copy_mcast — Intra-core DXA multicast copy test.
//
// `num_recv` single-warp CTAs co-resident on one core all fetch the SAME tile
// from GMEM via a single DXA multicast issued by CTA 0. Each receiver then
// writes its SMEM copy to a distinct dst region for host verification.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

constexpr uint32_t kDescSrc = 0;

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t cta_id    = get_local_group_id();
  const uint32_t tile_rows = arg->tile_rows;
  const uint32_t tile_cols = arg->tile_cols;
  const uint32_t num_recv  = arg->num_recv;

  auto shmem = reinterpret_cast<TYPE*>(__local_mem());

  // Naming:
  //   local_bar — per-CTA bar (vortex::barrier); receives MY multicast event.
  //   group_bar — local-group-shared bar (vortex::group_barrier); K members
  //               rendezvous here so every receiver's expect_tx is visible
  //               before rank-0 fires the multicast.
  vortex::barrier        local_bar(0);
  vortex::group_barrier  group_bar(1, num_recv);
  const bool is_loader_warp = (get_sub_group_id() == 0);

  if (is_loader_warp) {
    // Helper bundles expect_tx + sync_and_issue so the mask <-> expect_tx
    // invariant cannot be violated by hand. Mask = (1<<num_recv) - 1.
    vortex::dxa_multicast_2d mc(kDescSrc, num_recv, local_bar, group_bar);
    mc.sync_and_issue(shmem, /*coord0=*/0, /*coord1=*/0);
  }
  // ALL warps of THIS CTA wait for the multicast release to land in LMEM.
  local_bar.arrive_and_wait();

  // Store the received tile to dst region for host verification.
  // Layout: dst[cta_id * tile_elems + r * tile_cols + c]
  auto dst = reinterpret_cast<TYPE*>(arg->dst_addr);
  const uint32_t tile_elems = tile_rows * tile_cols;
  const uint32_t l_col = threadIdx.x;
  for (uint32_t r = 0; r < tile_rows; ++r) {
    if (l_col < tile_cols) {
      dst[cta_id * tile_elems + r * tile_cols + l_col] =
          shmem[r * tile_cols + l_col];
    }
  }

  (void)arg->src_addr;
  (void)arg->src_row_stride;
}

// The checked modes exist only on an mbarrier device, so every other build
// keeps the kernel binary above unchanged.
#ifdef VX_CFG_EXT_MBAR_ENABLE
#include <vx_dlmem.h>
#include <vx_mbar.h>

namespace {

#define INL inline __attribute__((always_inline))

INL void issue_legacy(uint32_t ndim, uint32_t bar, void* smem, uint32_t mask, const uint32_t* c) {
  switch (ndim) {
  case 1:  vx_dxa_issue_1d_multicast_wg(DESC_SLOT, bar, smem, c[0], mask); break;
  case 2:  vx_dxa_issue_2d_multicast_wg(DESC_SLOT, bar, smem, c[0], c[1], mask); break;
  case 3:  vx_dxa_issue_3d_multicast_wg(DESC_SLOT, bar, smem, c[0], c[1], c[2], mask); break;
  case 4:  vx_dxa_issue_4d_multicast_wg(DESC_SLOT, bar, smem, c[0], c[1], c[2], c[3], mask); break;
  default: vx_dxa_issue_5d_multicast_wg(DESC_SLOT, bar, smem, c[0], c[1], c[2], c[3], c[4], mask); break;
  }
}

INL void issue_cluster(uint32_t ndim, uint64_t* mbar, void* smem, uint32_t mask, const uint32_t* c) {
  switch (ndim) {
  case 1:  vx_dxa_issue_1d_mbar_multicast(DESC_SLOT, mbar, smem, mask, c[0]); break;
  case 2:  vx_dxa_issue_2d_mbar_multicast(DESC_SLOT, mbar, smem, mask, c[0], c[1]); break;
  case 3:  vx_dxa_issue_3d_mbar_multicast(DESC_SLOT, mbar, smem, mask, c[0], c[1], c[2]); break;
  case 4:  vx_dxa_issue_4d_mbar_multicast(DESC_SLOT, mbar, smem, mask, c[0], c[1], c[2], c[3]); break;
  default: vx_dxa_issue_5d_mbar_multicast(DESC_SLOT, mbar, smem, mask, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

// Every warp of every block of the cluster arms, syncs and (one warp) issues.
INL void issue_helper(uint32_t ndim, uint64_t* mbar, void* smem, const mc_arg_t* a, const uint32_t* c) {
  switch (ndim) {
  case 1: vortex::dxa_multicast_cluster_1d(DESC_SLOT, a->mask, a->issuer, mbar, a->box)
              .sync_and_issue(smem, c[0]); break;
  case 2: vortex::dxa_multicast_cluster_2d(DESC_SLOT, a->mask, a->issuer, mbar, a->box)
              .sync_and_issue(smem, c[0], c[1]); break;
  case 3: vortex::dxa_multicast_cluster_3d(DESC_SLOT, a->mask, a->issuer, mbar, a->box)
              .sync_and_issue(smem, c[0], c[1], c[2]); break;
  case 4: vortex::dxa_multicast_cluster_4d(DESC_SLOT, a->mask, a->issuer, mbar, a->box)
              .sync_and_issue(smem, c[0], c[1], c[2], c[3]); break;
  default: vortex::dxa_multicast_cluster_5d(DESC_SLOT, a->mask, a->issuer, mbar, a->box)
              .sync_and_issue(smem, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

INL void tile_coords(const mc_arg_t* arg, uint32_t t, uint32_t* c) {
  for (uint32_t d = 0; d < MAX_DIMS; ++d) {
    c[d] = (t % arg->grids[d]) * arg->tiles[d];
    t /= arg->grids[d];
  }
}

INL void idle_cycles(uint64_t n) {
  const uint64_t t0 = vx_rdcycle();
  while (vx_rdcycle() - t0 < n) {}
}

INL void record(const mc_arg_t* arg, uint32_t b, uint32_t rank) {
  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * REC_WORDS;
  rec[REC_BLOCK] = b;
  rec[REC_CORE]  = vx_core_id();
  rec[REC_SLOT]  = get_local_group_id();
  rec[REC_RANK]  = rank;
  __atomic_fetch_add(&rec[REC_RAN], 1u, __ATOMIC_RELAXED);
}

// The bank's own answer to a try_wait: false when it read the phase
// incomplete, and then the warp has slept.
INL bool bank_try_wait(uint64_t* obj, uint64_t tok) {
  uint64_t r;
  __asm__ volatile (".insn r %1, 5, 0, %0, %2, %3"
                    : "=r"(r) : "i"(RISCV_CUSTOM2), "r"(obj), "r"(tok) : "memory");
  return r != 0;
}

// A wait for the phase of parity 0 that runs the same instructions whether
// or not the try_wait slept, so the instruction count does not depend on timing.
INL uint32_t wait_phase0(uint64_t* mbar, uint32_t& tries) {
  for (;;) {
    ++tries;
    const bool raw    = bank_try_wait(mbar, 0);
    const bool tested = vx_mbar_test_wait(mbar, 0);
    if (raw | tested)
      return 0;
    if (tries >= WAIT_LIMIT)
      return 1;
  }
}

INL void dump_lmem(const mc_arg_t* arg, uint32_t b, volatile uint32_t* lw) {
  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr + (uint64_t)b * arg->lmem_bytes);
  for (uint32_t w = threadIdx.x; w < arg->lmem_bytes / 4; w += blockDim.x)
    dump[w] = lw[w];
}

INL void finish(const mc_arg_t* arg, uint32_t err, uint32_t tries) {
  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + (blockIdx.x * blockDim.x + threadIdx.x) * ERR_WORDS;
  out[ERR_WAIT]  = err;
  out[ERR_TRIES] = tries;
  out[ERR_DONE]  = 1;
}

} // namespace

// Legacy multicast with any mask, in a DEFAULT cluster: rank 0 issues, and
// bit j selects the block in the issuer's slot + j, rank j. Each receiver
// registers its tx events before the rendezvous and waits on its barrier;
// a second rendezvous keeps every block from dumping before all copies landed.
__kernel void legacy_k(mc_arg_t* __UNIFORM__ arg) {
  const uint32_t b = blockIdx.x;
  const uint32_t r = get_cluster_rank();
  auto lm = reinterpret_cast<uint8_t*>(__local_mem());
  auto lw = reinterpret_cast<volatile uint32_t*>(lm);
  const bool recv = (arg->mask >> r) & 1u;

  if (threadIdx.x == 0)
    record(arg, b, r);
  for (uint32_t w = threadIdx.x; w < arg->lmem_bytes / 4; w += blockDim.x)
    lw[w] = canary_word(b, w);
  __syncthreads();

  vortex::barrier       local_bar(0);
  vortex::group_barrier group_bar(1, arg->k);
  const bool loader = (get_sub_group_id() == 0);
  if (loader) {
    if (recv)
      local_bar.expect_tx(arg->copies);
    group_bar.arrive_and_wait();
    if (r == 0) {
      for (uint32_t i = 0; i < arg->copies; ++i) {
        uint32_t c[MAX_DIMS];
        tile_coords(arg, src_tile(b / arg->k, i, arg->copies, arg->ntiles), c);
        issue_legacy(arg->ndim, local_bar.id(), lm + LM_TILE + i * arg->tstride, arg->mask, c);
      }
    }
  }
  if (recv)
    local_bar.arrive_and_wait();
  if (loader)
    group_bar.arrive_and_wait();
  __syncthreads();

  dump_lmem(arg, b, lw);
  finish(arg, 0, 0);
}

// Cluster multicast: every block fills its local memory with canaries,
// initializes its mbarrier and, if its rank is in the mask, arms it with the
// bytes of every copy before the cluster barrier. The issuer then multicasts
// its copies into the tile slots of the masked ranks; every receiver waits
// for its own phase, and after a second cluster barrier every block dumps its
// whole local memory for the host's byte-exact check.
__kernel void cluster_k(mc_arg_t* __UNIFORM__ arg) {
  const uint32_t b = blockIdx.x;
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t r = cl.block_rank();
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto lw   = reinterpret_cast<volatile uint32_t*>(lm);
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);
  const bool recv = (arg->mask >> r) & 1u;
  const uint32_t g = b / arg->k;

  if (threadIdx.x == 0)
    record(arg, b, r);
  for (uint32_t w = 2 + threadIdx.x; w < arg->lmem_bytes / 4; w += blockDim.x)
    lw[w] = canary_word(b, w);
  if (threadIdx.x == 0)
    vx_mbar_init(mbar, arg->helper ? arg->copies : 1);

  if (arg->helper) {
    __syncthreads();
    for (uint32_t i = 0; i < arg->copies; ++i) {
      uint32_t c[MAX_DIMS];
      tile_coords(arg, src_tile(g, i, arg->copies, arg->ntiles), c);
      issue_helper(arg->ndim, mbar, lm + LM_TILE + i * arg->tstride, arg, c);
    }
  } else {
    if (threadIdx.x == 0 && recv)
      vx_mbar_arrive_tx(mbar, arg->copies * arg->box);
    cl.sync();
    if (r == arg->issuer && get_sub_group_id() == 0) {
      for (uint32_t i = 0; i < arg->copies; ++i) {
        uint32_t c[MAX_DIMS];
        tile_coords(arg, src_tile(g, i, arg->copies, arg->ntiles), c);
        issue_cluster(arg->ndim, mbar, lm + LM_TILE + i * arg->tstride, arg->mask, c);
      }
    }
  }

  uint32_t err = 0, tries = 0;
  if (recv)
    err = wait_phase0(mbar, tries);
  cl.sync();
  dump_lmem(arg, b, lw);
  finish(arg, err, tries);
}

// Runs the simulator must stop, in a SPREAD cluster of two blocks: rank 0
// issues one 1-D multicast; rank 1 stays alive meanwhile, except in
// -x retired, where it exits at once and rank 0 issues only after lingering.
__kernel void neg_k(mc_arg_t* __UNIFORM__ arg) {
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t r = cl.block_rank();
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);

  if (threadIdx.x == 0)
    record(arg, blockIdx.x, r);
  if (arg->neg == NEG_RETIRED && r != 0)
    return;
  if (threadIdx.x == 0)
    vx_mbar_init(mbar, 1);
  __syncthreads();
  if (arg->neg == NEG_RETIRED)
    idle_cycles(arg->linger);
  if (r == 0 && get_sub_group_id() == 0) {
    const uint32_t c[MAX_DIMS] = {};
    switch (arg->neg) {
    case NEG_SPREAD: {
      vortex::barrier bar(1);
      bar.expect_tx(1);
      issue_legacy(1, bar.id(), lm + LM_TILE, 0x3, c);
      break;
    }
    case NEG_BOX:
      issue_cluster(1, mbar, lm + arg->neg_off, 0x3, c);
      break;
    default:
      issue_cluster(1, mbar, lm + LM_TILE, arg->mask, c);
      break;
    }
  }
  idle_cycles(arg->linger);
  finish(arg, 0, 0);
}
#endif // VX_CFG_EXT_MBAR_ENABLE
