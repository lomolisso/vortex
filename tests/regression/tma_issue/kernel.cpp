#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dlmem.h>
#include <vx_dxa.h>
#include <vx_mbar.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

INL const vx_tensor_map_t* map_at(const tma_arg_t* arg, uint32_t i) {
  return reinterpret_cast<const vx_tensor_map_t*>(arg->maps_addr + (uint64_t)i * sizeof(vx_tensor_map_t));
}

INL uint32_t wait_phase(uint64_t* mbar, uint32_t parity) {
  for (uint32_t t = 0; !vx_mbar_try_wait_parity(mbar, parity); ) {
    if (++t > WAIT_LIMIT)
      return 1;
  }
  return 0;
}

INL bool is_load(uint32_t f) {
  return (f <= F_TLOAD_5D && f != F_COPY_S2S) || f == F_TMAP_PF || f == F_TMAP_INV;
}

// Without the front-end switch the operands travel by wgather, which writes
// every lane's destination register: only a converged warp may issue, and
// only the lead lane arrives on the mbarriers.
INL bool issues(uint32_t lane, uint32_t want) {
#if VX_CFG_H100_FE_ENABLED
  return lane == want;
#else
  (void)lane;
  (void)want;
  return true;
#endif
}

// shared::cluster destinations other than the issuing block: rank 1's.
INL bool is_peer(uint32_t f) {
  return f == F_COPY_S2S || f == F_CRED;
}

INL uint32_t tload_bytes(uint32_t rank) {
  const uint32_t box[5] = {TBOX0, TBOX1, TBOX2, TBOX3, TBOX4};
  uint32_t n = 4;
  for (uint32_t i = 0; i < rank; ++i)
    n *= box[i];
  return n;
}

INL uint32_t form_tx(uint32_t f) {
  if (f >= F_TLOAD_1D && f <= F_TLOAD_5D)
    return tload_bytes(f - F_TLOAD_1D + 1);
  if (f == F_TMAP_PF || f == F_TMAP_INV)
    return tload_bytes(2);
  return BULK_BYTES;
}

void issue_forms(const tma_arg_t* arg, uint8_t* lm, bool lead) {
  auto mb   = [&](uint32_t f) { return reinterpret_cast<uint64_t*>(lm + LM_MBAR + 8 * f); };
  auto area = [&](uint32_t f) { return static_cast<void*>(lm + LM_AREA + f * AREA_BYTES); };
  const void* src  = lm + LM_SRC;
  void* gsrc       = reinterpret_cast<void*>(arg->gsrc_addr);
  uint8_t* gdst    = reinterpret_cast<uint8_t*>(arg->gdst_addr);
  const vx_tensor_map_t* m2 = map_at(arg, 1);

  for (uint32_t f = 0; f < F_COUNT; ++f) {
    if (lead && is_load(f))
      vx_mbar_arrive_tx(mb(f), form_tx(f));
  }
  vx_fence_proxy_async();

  vx_dxa_bulk_load(area(F_BULK_G2S), gsrc, BULK_BYTES, mb(F_BULK_G2S));
  vx_dxa_bulk_load_multicast(area(F_BULK_MC), gsrc, BULK_BYTES, mb(F_BULK_MC), 0x1);
  vx_dxa_bulk_copy(vx_cluster_map_shared_rank(area(F_COPY_S2S), 1), src, BULK_BYTES,
                   static_cast<uint64_t*>(vx_cluster_map_shared_rank(mb(F_COPY_S2S), 1)));
  vx_dxa_load_1d(map_at(arg, 0), area(F_TLOAD_1D + 0), mb(F_TLOAD_1D + 0), TC0);
  vx_dxa_load_2d(map_at(arg, 1), area(F_TLOAD_1D + 1), mb(F_TLOAD_1D + 1), TC0, TC1);
  vx_dxa_load_3d(map_at(arg, 2), area(F_TLOAD_1D + 2), mb(F_TLOAD_1D + 2), TC0, TC1, TC2);
  vx_dxa_load_4d(map_at(arg, 3), area(F_TLOAD_1D + 3), mb(F_TLOAD_1D + 3), TC0, TC1, TC2, TC3);
  vx_dxa_load_5d(map_at(arg, 4), area(F_TLOAD_1D + 4), mb(F_TLOAD_1D + 4), TC0, TC1, TC2, TC3, TC4);

  vx_dxa_bulk_store(gdst, src, BULK_BYTES);
  vx_dxa_reduce_global(gdst + BULK_BYTES, src, BULK_BYTES, DXA_RED_ADD, DXA_RT_U32);
  vx_dxa_store_2d(map_at(arg, MAP_STORE), src, TC0, TC1);
  vx_dxa_reduce_2d(map_at(arg, MAP_RED), src, DXA_RED_ADD, TC0, TC1);
  vx_bulk_commit_group();
  vx_bulk_wait_all();

  vx_dxa_reduce_cluster(vx_cluster_map_shared_rank(area(F_CRED), 1), src, BULK_BYTES,
                        static_cast<uint64_t*>(vx_cluster_map_shared_rank(mb(F_CRED), 1)), DXA_RED_ADD,
                        DXA_RT_U32);

  vx_dxa_prefetch_l2(gsrc, BULK_BYTES);
  vx_dxa_prefetch_2d(m2, TC0, TC1);

  vx_prefetch_tensormap(m2);
  vx_dxa_load_2d(m2, area(F_TMAP_PF), mb(F_TMAP_PF), TC0, TC1);
  vx_fence_proxy_tensormap_acquire(m2);
  vx_dxa_load_2d(m2, area(F_TMAP_INV), mb(F_TMAP_INV), TC0, TC1);
}

#define REP8(s) do { s; s; s; s; s; s; s; s; } while (0)
// Straight-line copies: a loop branch between them would stretch every eighth gap.
#define REPN(s) do { REP8(s); if (REPS > 1) REP8(s); if (REPS > 2) REP8(s); } while (0)
// An unmeasured lead copy right before t0: the first measured copy's interval
// then counts from it in every instance, not from the previous call's last
// copy (the code between calls differs by a few instructions).
#define TIMED(s) do { s; t0 = vx_rdcycle(); REPN(s); } while (0)

// One lead copy, then REPS x 8 copies of form `c`; returns the cycles from
// after the lead copy to the return of the last.
template <uint32_t REPS>
__attribute__((noinline)) uint32_t cost_run(const tma_arg_t* arg, uint8_t* lm, uint32_t c, uint64_t* mb, uint32_t w,
                                            bool lead) {
  static_assert(REPS >= 1 && REPS <= 3, "cost_run unrolls 8 to 24 copies");
  void* dst        = lm + LM_AREA + w * AREA_BYTES;
  const void* src  = lm + LM_SRC;
  void* gsrc       = reinterpret_cast<void*>(arg->gsrc_addr);
  uint8_t* gdst    = reinterpret_cast<uint8_t*>(arg->gdst_addr) + w * COST_BYTES;
  const vx_tensor_map_t* m2 = map_at(arg, 1);
  const vx_tensor_map_t* ms = map_at(arg, MAP_STORE);
  const vx_tensor_map_t* mr = map_at(arg, MAP_RED);
  const uint32_t copies = REPS * 8 + 1;
  if (lead && c == C_BULK)
    vx_mbar_arrive_tx(mb, copies * COST_BYTES);
  if (lead && c == C_TENSOR)
    vx_mbar_arrive_tx(mb, copies * tload_bytes(2));
  vx_fence_proxy_async();

  uint64_t t0;
  switch (c) {
  case C_BULK:         TIMED(vx_dxa_bulk_load(dst, gsrc, COST_BYTES, mb)); break;
  case C_TENSOR:       TIMED(vx_dxa_load_2d(m2, dst, mb, TC0, TC1)); break;
  case C_BULK_STORE:   TIMED(vx_dxa_bulk_store(gdst, src, COST_BYTES)); break;
  case C_BULK_REDUCE:  TIMED(vx_dxa_reduce_global(gdst, src, COST_BYTES, DXA_RED_ADD, DXA_RT_U32)); break;
  case C_TENSOR_STORE: TIMED(vx_dxa_store_2d(ms, src, TC0, TC1)); break;
  case C_TENSOR_RED:   TIMED(vx_dxa_reduce_2d(mr, src, DXA_RED_ADD, TC0, TC1)); break;
  case C_PF_BULK:      TIMED(vx_dxa_prefetch_l2(gsrc, COST_BYTES)); break;
  case C_PF_TENSOR:    TIMED(vx_dxa_prefetch_2d(m2, TC0, TC1)); break;
  case C_TMAP_INVAL:   TIMED(vx_fence_proxy_tensormap_acquire(m2)); break;
  default:             TIMED(vx_prefetch_tensormap(m2)); break;
  }
  const uint64_t t1 = vx_rdcycle();

  if (c == C_BULK_STORE || c == C_BULK_REDUCE || c == C_TENSOR_STORE || c == C_TENSOR_RED) {
    vx_bulk_commit_group();
    vx_bulk_wait_all();
  }
  return (uint32_t)(t1 - t0);
}

// Last bytes first: they are the last the engine writes.
__attribute__((noinline)) uint32_t visible_run(const tma_arg_t* arg, uint8_t* lm) {
  const void* src = lm + LM_AREA;
  auto gs = reinterpret_cast<volatile uint32_t*>(arg->vis_addr);
  auto gr = gs + VIS_BYTES / 4;
  vx_fence_proxy_async();
  vx_dxa_bulk_store(const_cast<uint32_t*>(gs), src, VIS_BYTES);
  vx_bulk_commit_group();
  vx_bulk_wait_all();
  uint32_t stale = 0;
  for (uint32_t i = VIS_BYTES / 4; i-- > 0; )
    stale += (gs[i] != vis_word(i));
  vx_dxa_reduce_global(const_cast<uint32_t*>(gr), src, VIS_BYTES, DXA_RED_ADD, DXA_RT_U32);
  vx_bulk_commit_group();
  vx_bulk_wait_all();
  for (uint32_t i = VIS_BYTES / 4; i-- > 0; )
    stale += (gr[i] != vis_red_init(i) + vis_word(i)) << 16;
  return stale;
}

#if VX_CFG_H100_SM_ENABLED && VX_CFG_H100_FE_ENABLED
// Cycles of one load, from a drained pipeline to its value in a register. Out of
// line so that one warm-up call leaves every timed window's code in the I-cache.
__attribute__((noinline)) uint32_t ld_timed(const volatile uint32_t* p, uint32_t* t) {
  const uint64_t t0 = vx_rdcycle_sync();
  uint32_t v = *p;
  __asm__ volatile ("mv %0, %0" : "+r"(v));
  *t = (uint32_t)(vx_rdcycle_sync() - t0);
  return v;
}

// Loads every word of [p, p + n) and returns once all of them are in registers.
INL void touch(const volatile uint32_t* p, uint32_t n) {
  uint32_t s = 0;
  for (uint32_t i = 0; i < n; ++i)
    s += p[i];
  __asm__ volatile ("mv %0, %0" : "+r"(s));
  vx_rdcycle_sync();
}

// Out of line and called twice, the second call timed with its code warm.
__attribute__((noinline)) uint32_t inval_timed() {
  const uint64_t t0 = vx_rdcycle_sync();
  vx_l1_inval();
  return (uint32_t)(vx_rdcycle_sync() - t0);
}

__attribute__((noinline)) uint32_t fence_timed() {
  const uint64_t t0 = vx_rdcycle_sync();
  vx_fence();
  return (uint32_t)(vx_rdcycle_sync() - t0);
}

__attribute__((noinline)) void visible_l1_run(const tma_arg_t* arg, uint8_t* lm, uint32_t* out) {
  constexpr uint32_t LW = VX_CFG_DCACHE_LINE_SIZE / 4;
  static_assert(LW * 4 > VL1_WBYTES && 2 * LW * 4 <= VL1_BYTES, "the written bytes lie in the first line");
  auto r  = reinterpret_cast<volatile uint32_t*>(arg->vis_addr);
  auto r2 = r + VL1_R2 / 4;
  auto u  = reinterpret_cast<const volatile uint32_t*>(arg->gsrc_addr);
  const void* src = lm + LM_AREA;
  uint32_t bad = 0, kept = 0, t;

  touch(r, VL1_BYTES / 4);
  touch(u, VL1_BYTES / 4);
  ld_timed(&u[0], &t);
  vx_fence_proxy_async();
  vx_dxa_bulk_store(const_cast<uint32_t*>(r), src, VL1_WBYTES);
  vx_bulk_commit_group();
  vx_bulk_wait_group<0>();
  bad += ld_timed(&r[VL1_WBYTES / 4 - 1], &t) != vis_word(VL1_WBYTES / 4 - 1);
  out[O_TW] = t;
  bad += ld_timed(&r[LW], &t) != vl1_old(LW);
  out[O_TN] = t;
  const uint32_t u0 = ld_timed(&u[0], &t);
  out[O_TU] = t;
  bad += ld_timed(&u[0], &t) != u0;
  out[O_TH] = t;
  for (uint32_t i = 0; i < VL1_BYTES / 4; ++i)
    bad += r[i] != (i < VL1_WBYTES / 4 ? vis_word(i) : vl1_old(i));

  touch(r2, VL1_BYTES / 4);
  vx_dxa_bulk_store(const_cast<uint32_t*>(r2), src, VL1_WBYTES);
  vx_bulk_commit_group();
  vx_bulk_wait_group_read<0>();
  kept += ld_timed(&r2[0], &t) == vl1_old(VL1_R2 / 4);
  out[O_TS] = t;
  for (uint32_t i = 1; i < VL1_WBYTES / 4; ++i)
    kept += r2[i] == vl1_old(VL1_R2 / 4 + i);
  vx_bulk_wait_group<0>();

  auto r3 = r + VL1_R3 / 4;
  touch(r3, VL1_BYTES / 4);
  vx_dxa_reduce_global(const_cast<uint32_t*>(r3), src, VL1_WBYTES, DXA_RED_ADD, DXA_RT_U32);
  vx_bulk_commit_group();
  vx_bulk_wait_group<0>();
  for (uint32_t i = 0; i < VL1_BYTES / 4; ++i)
    bad += r3[i] != vl1_old(VL1_R3 / 4 + i) + (i < VL1_WBYTES / 4 ? vis_word(i) : 0);

  inval_timed();
  out[O_TINV] = inval_timed();
  fence_timed();
  out[O_TFENCE] = fence_timed();
  out[O_STALE] = bad | kept << 16;
}
#endif

} // namespace

__kernel void tma_k(tma_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t nthr = blockDim.x;
  const uint32_t w    = get_sub_group_id();
  const uint32_t lane = threadIdx.x % (uint32_t)vx_num_threads();
  auto lm  = reinterpret_cast<uint8_t*>(__local_mem());
  auto lw  = reinterpret_cast<volatile uint32_t*>(lm);
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t rank = arg->kernel == K_FORMS ? cl.block_rank() : 0;
  auto out = reinterpret_cast<uint32_t*>(arg->out_addr) + (rank * nthr + tid) * O_WORDS;

  for (uint32_t i = tid; i < LM_BYTES / 4; i += nthr) {
    uint32_t v = 0xA5A5A5A5u;
    if (i >= LM_SRC / 4 && i < (LM_SRC + SRC_BYTES) / 4)
      v = 0xC0000000u | ((i - LM_SRC / 4) * 7919u);
    else if (i >= (LM_AREA + F_CRED * AREA_BYTES) / 4 && i < (LM_AREA + (F_CRED + 1) * AREA_BYTES) / 4)
      v = (i - (LM_AREA + F_CRED * AREA_BYTES) / 4) * 3u + 1u;
    if ((arg->kernel == K_VISIBLE || arg->kernel == K_VISIBLE_L1) && i >= LM_AREA / 4 && i < (LM_AREA + VIS_BYTES) / 4)
      v = vis_word(i - LM_AREA / 4);
    if (i >= (LM_MBAR + 8 * NUM_MBARS) / 4)
      lw[i] = v;
  }
  if (tid == 0) {
    for (uint32_t k = 0; k < NUM_MBARS; ++k)
      vx_mbar_init(reinterpret_cast<uint64_t*>(lm + LM_MBAR + 8 * k), 1);
  }
  if (arg->kernel == K_FORMS && rank == 1 && tid == 0) {
    vx_mbar_arrive_tx(reinterpret_cast<uint64_t*>(lm + LM_MBAR + 8 * F_COPY_S2S), BULK_BYTES);
    vx_mbar_arrive_tx(reinterpret_cast<uint64_t*>(lm + LM_MBAR + 8 * F_CRED), BULK_BYTES);
  }
  vx_fence_proxy_async();
  __syncthreads();
  if (arg->kernel == K_FORMS)
    cl.sync();

  uint32_t alive = tid * 0x9E3779B1u + 7u;
  __asm__ volatile ("" : "+r"(alive));
  uint32_t err = 0;

  if (arg->kernel == K_FORMS) {
    if (rank == 0 && w == 0 && issues(lane, arg->lane))
      issue_forms(arg, lm, lane == arg->lane);
    __asm__ volatile ("" : "+r"(alive));
    for (uint32_t f = 0; f < F_COUNT; ++f) {
      if (rank == 0 ? is_load(f) : is_peer(f))
        err |= wait_phase(reinterpret_cast<uint64_t*>(lm + LM_MBAR + 8 * f), 0);
    }
  } else if (arg->kernel == K_VISIBLE) {
    out[O_STALE] = (w == 0 && issues(lane, arg->lane)) ? visible_run(arg, lm) : 0;
  } else if (arg->kernel == K_VISIBLE_L1) {
#if VX_CFG_H100_SM_ENABLED && VX_CFG_H100_FE_ENABLED
    if (w == 0 && issues(lane, arg->lane))
      visible_l1_run(arg, lm, out);
#endif
  } else if (w < arg->warps && issues(lane, arg->lane)) {
    auto mb = [&](uint32_t k) { return reinterpret_cast<uint64_t*>(lm + LM_MBAR + 8 * (4 * w + k)); };
    const bool lead = (lane == arg->lane);
    // Each unrolled body runs once unmeasured: cold, every copy waits for an I-cache line fill.
    cost_run<COST_N1 / 8>(arg, lm, arg->form, mb(0), w, lead);
    cost_run<COST_N2 / 8>(arg, lm, arg->form, mb(1), w, lead);
    const uint32_t t1 = cost_run<COST_N1 / 8>(arg, lm, arg->form, mb(2), w, lead);
    const uint32_t t2 = cost_run<COST_N2 / 8>(arg, lm, arg->form, mb(3), w, lead);
    if (arg->form == C_BULK || arg->form == C_TENSOR) {
      for (uint32_t k = 0; k < 4; ++k)
        err |= wait_phase(mb(k), 0);
    }
    out[O_T1] = t1;
    out[O_T2] = t2;
  }
  __asm__ volatile ("" : "+r"(alive));

  __syncthreads();
  if (arg->kernel == K_FORMS)
    cl.sync();
  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr) + rank * (LM_BYTES / 4);
  for (uint32_t i = tid; i < LM_BYTES / 4; i += nthr)
    dump[i] = lw[i];
  out[O_ALIVE] = alive;
  out[O_WAIT]  = err;
  out[O_DONE]  = 1;
}
