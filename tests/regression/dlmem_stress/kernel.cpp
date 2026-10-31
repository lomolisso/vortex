#include <vx_spawn2.h>
#include <vx_dlmem.h>
#include "common.h"

#ifdef VX_CFG_EXT_DXA_ENABLE
#include <vx_dxa.h>
#include <vx_barrier.h>
#ifdef VX_CFG_EXT_MBAR_ENABLE
#include <vx_mbar.h>
#endif
#endif

namespace {

#define INL inline __attribute__((always_inline))

struct Ctx {
  uint32_t k, rank, c, b, tid, nt, warp, lane, seed;
  volatile uint32_t* own;
  vortex::cluster_group cl;
  uint32_t errors, first_test, first_index;
  uint64_t first_got, first_want;

  INL volatile uint8_t* window(uint32_t q) {
    return reinterpret_cast<volatile uint8_t*>(cl.map_shared_rank(const_cast<uint32_t*>(own), q));
  }

  INL void check(uint32_t test, uint32_t index, uint64_t got, uint64_t want) {
    if (got != want && errors++ == 0) {
      first_test  = test;
      first_index = index;
      first_got   = got;
      first_want  = want;
    }
  }
};

INL void ctx_init(Ctx& x, const stress_arg_t* arg) {
  x.k     = arg->k;
  x.rank  = vx_cluster_block_rank();
  x.b     = blockIdx.x;
  x.c     = x.b / x.k;
  x.tid   = threadIdx.x;
  x.nt    = blockDim.x;
  x.warp  = x.tid / arg->num_threads;
  x.lane  = x.tid % arg->num_threads;
  x.seed  = arg->seed;
  x.own   = reinterpret_cast<volatile uint32_t*>(__local_mem());
  x.errors = x.first_test = x.first_index = 0;
  x.first_got = x.first_want = 0;
  if (x.tid == 0) {
    auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + x.b * REC_WORDS;
    rec[REC_RANK] = x.rank;
    rec[REC_SIZE] = vx_cluster_num_blocks();
    rec[REC_CORE] = vx_core_id();
    rec[REC_SLOT] = get_local_group_id();
    rec[REC_RAN]  = 1;
  }
  x.check(TEST_SETUP, 0, vx_cluster_num_blocks(), x.k);
}

INL void ctx_done(const Ctx& x, const stress_arg_t* arg) {
  auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + (x.b * x.nt + x.tid) * ERR_WORDS;
  err[ERR_TEST]      = x.first_test;
  err[ERR_INDEX]     = x.first_index;
  err[ERR_GOT]       = (uint32_t)x.first_got;
  err[ERR_GOT + 1]   = (uint32_t)(x.first_got >> 32);
  err[ERR_WANT]      = (uint32_t)x.first_want;
  err[ERR_WANT + 1]  = (uint32_t)(x.first_want >> 32);
  err[ERR_COUNT]     = x.errors;
}

// One planned step on cell st.j of this thread, through the window; returns
// what the load or atomic read (0 for a store, a fence or a counter add).
// Inlined, it leaves stress_k with no split/join for its divergent branches.
__attribute__((noinline)) uint64_t run_step(Ctx& x, const st_step_t& st) {
  const uint32_t q = cell_rank(x.rank, st.j, x.k);
  volatile uint8_t* p = x.window(q) + cells_off() + 8 * cell_index(x.rank, x.tid, st.j, x.nt) + st.off;
  uint64_t got = 0;
  switch (st.op) {
  case OPC_ST:
    switch (st.size) {
    case 8:  *reinterpret_cast<volatile uint64_t*>(p) = st.v; break;
    case 4:  *reinterpret_cast<volatile uint32_t*>(p) = (uint32_t)st.v; break;
    case 2:  *reinterpret_cast<volatile uint16_t*>(p) = (uint16_t)st.v; break;
    default: *p = (uint8_t)st.v; break;
    }
    break;
  case OPC_LD:
    got = *reinterpret_cast<volatile uint64_t*>(p);
    break;
  case OPC_LDN:
    switch (st.size) {
    case 4:  got = (uint64_t)(int64_t)*reinterpret_cast<volatile int32_t*>(p); break;
    case 2:  got = *reinterpret_cast<volatile uint16_t*>(p); break;
    default: got = *p; break;
    }
    break;
  case OPC_AMOADD_D:
    asm volatile ("amoadd.d %0, %2, (%1)" : "=r"(got) : "r"(p), "r"(st.v) : "memory");
    break;
  case OPC_AMOXOR_W:
    asm volatile ("amoxor.w %0, %2, (%1)" : "=r"(got) : "r"(p), "r"(st.v) : "memory");
    break;
  case OPC_AMOSWAP_D:
    asm volatile ("amoswap.d %0, %2, (%1)" : "=r"(got) : "r"(p), "r"(st.v) : "memory");
    break;
  case OPC_AMOMAXU_W:
    asm volatile ("amomaxu.w %0, %2, (%1)" : "=r"(got) : "r"(p), "r"(st.v) : "memory");
    break;
  case OPC_CTR: {
    volatile uint8_t* ctr = x.window(st.ctr_rank) + ctr_off(x.k, x.nt) + 4 * st.ctr_idx;
    asm volatile ("amoadd.w zero, %1, (%0)" :: "r"(ctr), "r"(st.v) : "memory");
    break;
  }
  case OPC_FENCE:
    vx_fence();
    break;
  default:
    *reinterpret_cast<volatile uint64_t*>(p) = st.v;
    got = *reinterpret_cast<volatile uint64_t*>(p);
    break;
  }
  return got;
}

}  // namespace

// Phases of seeded random window loads, stores and atomics on the thread's own
// cells (spread over every slice of the cluster), checked against the thread's
// model of them, plus adds into shared counters. Each phase ends with a
// signature store into the next rank and one of three cluster-wide orderings,
// after which each thread checks the previous rank's signature for it. The
// host checks every cell and counter at the end against its own model.
__kernel void stress_k(stress_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t warps = x.nt / arg->num_threads;
  const uint32_t prev = (x.rank + x.k - 1) % x.k;
  const uint32_t clear_end = dxa_off(x.k, x.nt, warps) / sizeof(uint32_t);
  for (uint32_t e = x.tid; e < clear_end; e += x.nt)
    x.own[e] = 0;
  uint64_t cells[ST_CELLS] = {};
#ifdef VX_CFG_EXT_DXA_ENABLE
  const bool dxa = (arg->flags & FLAG_DXA) != 0;
  vortex::barrier dxa_bar(0);
  auto dxa_dst = x.own + dxa_off(x.k, x.nt, warps) / sizeof(uint32_t);
#ifdef VX_CFG_EXT_MBAR_ENABLE
  // The previous rank's copies land in rdxa; each completes a phase of rmbar.
  const bool rdxa = (arg->flags & FLAG_RDXA) != 0;
  auto rdxa_dst = x.own + rdxa_off(x.k, x.nt, warps) / sizeof(uint32_t);
  auto rmbar = reinterpret_cast<uint64_t*>(const_cast<uint32_t*>(x.own) + rmbar_off(x.k, x.nt, warps) / sizeof(uint32_t));
  const uint32_t next = (x.rank + 1) % x.k;
  uint32_t rphase = 0;
  if (rdxa && x.tid == 0)
    vx_mbar_init(rmbar, 1);
#endif
#endif
  x.cl.sync();
  for (uint32_t p = 0; p < arg->phases; ++p) {
#ifdef VX_CFG_EXT_DXA_ENABLE
    const uint32_t tile = dxa_tile(p, gridDim.x, x.b);
    if (dxa && x.warp == 0) {
      dxa_bar.expect_tx(1);
      vx_dxa_issue_1d_wg(0, dxa_bar.id(), const_cast<uint32_t*>(dxa_dst), tile * DXA_WORDS);
    }
#ifdef VX_CFG_EXT_MBAR_ENABLE
    const bool remote = rdxa && rdxa_phase(x.seed, x.c, p);
    if (remote) {
      if (x.tid == 0)
        vx_mbar_arrive_tx(rmbar, 4 * DXA_WORDS);
      if (x.warp == 0) {
        auto peer = x.window(next);
        vx_dxa_issue_1d_mbar_cluster(0, reinterpret_cast<uint64_t*>(const_cast<uint8_t*>(peer + rmbar_off(x.k, x.nt, warps))),
                                     const_cast<uint8_t*>(peer + rdxa_off(x.k, x.nt, warps)),
                                     rdxa_tile(p, gridDim.x, x.b) * DXA_WORDS);
      }
    }
#endif
#endif
    for (uint32_t s = 0; s < arg->steps; ++s) {
      st_step_t st;
      st_plan(&st, st_warp_rnd(x.seed, x.c, p, s, x.rank, x.warp),
              st_lane_rnd(x.seed, x.c, p, s, x.rank, x.tid),
              st_value(x.seed, x.c, p, s, x.rank, x.tid), x.k, cells);
      const uint64_t got = run_step(x, st);
      if (st.op != OPC_ST && st.op != OPC_CTR && st.op != OPC_FENCE)
        x.check(TEST_STEP, (p << 24) | (st.op << 16) | s, got, st.want);
    }
#if defined(VX_CFG_EXT_DXA_ENABLE) && defined(VX_CFG_EXT_MBAR_ENABLE)
    // Before the phase end: the next remote copy follows a cluster.sync() after it.
    if (remote) {
      uint32_t tries = 0;
      while (!vx_mbar_try_wait_parity(rmbar, rphase & 1) && ++tries < RDXA_TRY_LIMIT) {}
      x.check(TEST_RDXA, (p << 24) | 0xFFFFFFu, tries < RDXA_TRY_LIMIT, 1);
      const uint32_t from = x.b - x.rank + (x.rank + x.k - 1) % x.k;
      const uint32_t rtile = rdxa_tile(p, gridDim.x, from);
      for (uint32_t e = x.tid; e < DXA_WORDS; e += x.nt)
        x.check(TEST_RDXA, (p << 24) | e, rdxa_dst[e], dxa_word(x.seed, rtile * DXA_WORDS + e));
      ++rphase;
    }
#endif
    auto sig = reinterpret_cast<volatile uint32_t*>(x.window((x.rank + 1) % x.k) + sig_off(x.k, x.nt));
    sig[(p % SIG_SLOTS) * x.nt + x.tid] = st_sig(x.seed, x.c, p, x.rank, x.tid);
    switch (st_end(x.seed, x.c, p)) {
    case END_SYNC:
      x.cl.sync();
      break;
    case END_RELAXED:
      x.cl.barrier_wait_relaxed(x.cl.barrier_arrive_relaxed());
      break;
    default: {
      vx_fence();
      if (x.lane == 0) {
        auto flag_next = reinterpret_cast<volatile uint32_t*>(
            x.window((x.rank + 1) % x.k) + flag_off(x.k, x.nt));
        auto flag_own = x.own + flag_off(x.k, x.nt) / sizeof(uint32_t);
        flag_next[x.warp] = p + 1;
        uint32_t spins = 0, got;
        while ((got = flag_own[x.warp]) < p + 1 && ++spins < SPIN_LIMIT) {}
        x.check(TEST_RING, p << 24, got >= p + 1, 1);
      }
      break;
    }
    }
    const uint32_t got = x.own[sig_off(x.k, x.nt) / sizeof(uint32_t) + (p % SIG_SLOTS) * x.nt + x.tid];
    x.check(TEST_SIG, (p << 24) | x.tid, got, st_sig(x.seed, x.c, p, prev, x.tid));
#ifdef VX_CFG_EXT_DXA_ENABLE
    if (dxa) {
      dxa_bar.arrive_and_wait();
      for (uint32_t e = x.tid; e < DXA_WORDS; e += x.nt)
        x.check(TEST_DXA, (p << 24) | e, dxa_dst[e], dxa_word(x.seed, tile * DXA_WORDS + e));
      // The next phase's copy must not land under a warp still checking this one.
      __syncthreads();
    }
#endif
  }
  x.cl.sync();
  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr) + x.b * dump_words(x.k, x.nt);
  const uint32_t cw = 2 * x.k * x.nt * ST_CELLS;
  for (uint32_t e = x.tid; e < cw; e += x.nt)
    dump[e] = x.own[cells_off() / sizeof(uint32_t) + e];
  for (uint32_t e = x.tid; e < ST_CTRS; e += x.nt)
    dump[cw + e] = x.own[ctr_off(x.k, x.nt) / sizeof(uint32_t) + e];
  ctx_done(x, arg);
}
