#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dlmem.h>
#include <vx_dxa.h>
#include <vx_mbar.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

struct Ctx {
  tc_arg_t* arg;
  uint8_t*  lm;
  uint64_t* mbar;
  uint32_t* o;
  uint32_t  phase;
  bool      failed;
};

INL void spin(uint64_t n) {
  const uint64_t t0 = vx_rdcycle();
  while (vx_rdcycle() - t0 < n) {}
}

// Warp 0 waits on the mbarrier, warp 1 issues: the waiter is asleep before the
// copy can land, so the copy alone sets the hop, never the wait's own steps.
INL bool waiter() {
  return threadIdx.x < (uint32_t)vx_num_threads();
}

// Warp 1's thread 0 issues the single-thread operations: the lane id is a
// source the compiler knows to diverge (a context flag is taken as uniform).
INL bool lead() {
  return !waiter() && vx_thread_id() == 0;
}

// The argument map's box: the copy of the hop just made.
INL bool param_data_ok(const Ctx& c) {
  auto d = reinterpret_cast<volatile uint32_t*>(c.lm + LM_DATA);
  for (uint32_t k = 0; k < 4; ++k) {
    if (d[k] != 0x11u * (k + 1))
      return false;
  }
  return true;
}

INL const vx_tensor_map_t* gmap(Ctx& c, uint32_t i) {
  return reinterpret_cast<const vx_tensor_map_t*>(c.arg->maps_addr + (uint64_t)i * 128u);
}

// One copy of the timed code for every hop: an inlined copy per call site would
// time its own cold instruction-cache misses. The hop runs from the issuing
// warp's clock read before its copy to the waiting warp's wake-up.
__attribute__((noinline)) uint32_t hop(Ctx& c, const vx_tensor_map_t* m) {
  if (c.failed)
    return 0;
  uint8_t* const  dst    = c.lm + LM_DATA;
  uint64_t* const mbar   = c.mbar;
  volatile uint64_t* const t0 = reinterpret_cast<volatile uint64_t*>(c.lm + LM_T0);
  volatile uint32_t* const failed = reinterpret_cast<volatile uint32_t*>(c.lm + LM_FAIL);
  const uint32_t parity = c.phase++ & 1;
  if (lead())
    vx_mbar_arrive_tx(mbar, 16);
  __syncthreads();
  uint64_t t1 = 0;
  if (waiter()) {
    for (uint32_t t = 0; !vx_mbar_try_wait_parity(mbar, parity); ) {
      if (++t > WAIT_LIMIT) {
        c.o[O_ERR] = 1;
        *failed = 1;
        break;
      }
    }
    t1 = vx_rdcycle();
  } else {
    spin(ISSUE_SPIN);
    const uint64_t s = vx_rdcycle();
    vx_dxa_load_2d(m, dst, mbar, 0, 0);
    // Every lane stores the same value: a divergent single-lane store would keep
    // this warp issuing (split/join) while the copy lands and its issue slots
    // would contend with the waiter's wake-up, which then times the other warp.
    *t0 = s;
  }
  __syncthreads();
  c.failed = (*failed != 0);
  return (uint32_t)(t1 - *t0);
}

// The level-1 hit of this SM: the fourth hop through one map.
INL uint32_t base(Ctx& c, const vx_tensor_map_t* m) {
  uint32_t t = 0;
  for (uint32_t r = 0; r < 4; ++r)
    t = hop(c, m);
  return t;
}

INL void put(Ctx& c, uint32_t v) {
  const uint32_t n = c.o[O_N];
  if (n < MAX_VALS)
    c.o[O_VALS + n] = v;
  c.o[O_N] = n + 1;
}

INL void knobs(uint32_t* o, const tc_arg_t* a) {
#if VX_CFG_H100_FE_ENABLED
  o[0] = tmap_l2_trip(a->maps_addr, VX_CFG_L1_LINE_SIZE);
  o[1] = tmap_l2_trip(reinterpret_cast<uint64_t>(&a->map), VX_CFG_L1_LINE_SIZE);
  o[2] = VX_CFG_TMAP_L1_ENTRIES;
  o[3] = VX_CFG_TMAP_L1_REPL;
  o[4] = VX_CFG_TMAP_L2_ENTRIES;
  o[5] = VX_CFG_TMAP_L2_WAYS;
#else
  for (uint32_t i = 0; i < 6; ++i)
    o[i] = 0;
#endif
}

INL void run(Ctx& c, uint32_t block) {
  tc_arg_t* a = c.arg;
  const uint32_t M = a->nmaps;
  switch (a->test) {
  case T_RR:
    c.o[O_BASE] = base(c, gmap(c, M));
    for (uint32_t r = 0; r < a->rounds; ++r) {
      for (uint32_t i = 0; i < M; ++i) {
        const uint32_t t = hop(c, gmap(c, i));
        if (r + 1 == a->rounds)
          put(c, t);
      }
    }
    break;
  case T_LRU:
    c.o[O_BASE] = base(c, gmap(c, 9));
    for (uint32_t i = 0; i < 8; ++i)
      hop(c, gmap(c, i));
    hop(c, gmap(c, 0));
    hop(c, gmap(c, 8));
    c.o[O_T] = hop(c, gmap(c, 0));
    break;
  case T_INVAL:
    c.o[O_BASE] = base(c, gmap(c, 0));
    if (lead())
      vx_fence_proxy_tensormap_acquire(gmap(c, 0));
    c.o[O_T] = hop(c, gmap(c, 0));
    hop(c, gmap(c, 1));
    for (uint32_t i = 2; i < 18; ++i)
      hop(c, gmap(c, i));
    if (lead())
      vx_fence_proxy_tensormap_acquire(gmap(c, 1));
    c.o[O_T + 1] = hop(c, gmap(c, 1));
    hop(c, gmap(c, 18));
    for (uint32_t i = 19; i < 35; ++i)
      hop(c, gmap(c, i));
    c.o[O_T + 2] = hop(c, gmap(c, 18));
    break;
  case T_PREFETCH:
    c.o[O_BASE] = base(c, gmap(c, 0));
    if (lead())
      vx_prefetch_tensormap(gmap(c, 1));
    spin(a->spin);
    c.o[O_T] = hop(c, gmap(c, 1));
    c.o[O_T + 1] = hop(c, gmap(c, 2));
    break;
  case T_RELAUNCH:
    if (!a->launch)
      break;
    [[fallthrough]];
  case T_PARAM:
    c.o[O_BASE] = base(c, gmap(c, 0));
    if (waiter() && vx_thread_id() == 0) {
      auto d = reinterpret_cast<volatile uint32_t*>(c.lm + LM_DATA);
      for (uint32_t k = 0; k < 4; ++k)
        d[k] = 0;
    }
    c.o[O_T] = hop(c, &a->map);
    if (waiter() && vx_thread_id() == 0 && !param_data_ok(c))
      c.o[O_ERR] = 3;
    c.o[O_T + 1] = hop(c, &a->map);
    if (lead())
      vx_prefetch_tensormap(&a->map2);
    spin(a->spin);
    c.o[O_T + 2] = hop(c, &a->map2);
    break;
  case T_LAUNCH:
    c.o[O_BASE] = base(c, gmap(c, M));
    for (uint32_t r = 0; r < (a->launch ? 1u : 2u); ++r) {
      for (uint32_t i = 0; i < M; ++i) {
        const uint32_t t = hop(c, gmap(c, i));
        if (a->launch)
          put(c, t);
      }
    }
    break;
  case T_SHARE: {
    auto flag = reinterpret_cast<uint32_t*>(a->flag_addr);
    c.o[O_CORE + block] = (uint32_t)vx_core_id();
    if (block == 0) {
      for (uint32_t r = 0; r < 2; ++r)
        for (uint32_t i = 0; i < M; ++i)
          hop(c, gmap(c, i));
      if (lead())
        __atomic_store_n(flag, 1u, __ATOMIC_RELEASE);
    } else {
      c.o[O_BASE] = base(c, gmap(c, M));
      for (uint32_t t = 0; __atomic_fetch_add(flag, 0u, __ATOMIC_ACQUIRE) == 0; ) {
        if (++t > WAIT_LIMIT) {
          c.o[O_ERR] = 1;
          break;
        }
      }
      for (uint32_t i = 0; i < M; ++i)
        put(c, hop(c, gmap(c, i)));
    }
    break;
  }
  case T_PLRU: {
    const uint32_t probe = M;
    c.o[O_BASE] = base(c, gmap(c, M + 1));
    for (uint32_t i = 0; i < M; ++i) {
      if (i % a->rounds == 0) {
        const uint32_t t = hop(c, gmap(c, probe));
        if (i >= 2 * a->rounds)
          put(c, t);
      }
      hop(c, gmap(c, i));
    }
    break;
  }
  default:
    c.o[O_ERR] = 2;
    break;
  }
}

} // namespace

__kernel void tc_k(tc_arg_t* __UNIFORM__ arg) {
  const uint32_t tid   = threadIdx.x;
  const uint32_t block = blockIdx.x;
  auto lm = reinterpret_cast<uint8_t*>(__local_mem());
  Ctx c{arg, lm, reinterpret_cast<uint64_t*>(lm + LM_MBAR), reinterpret_cast<uint32_t*>(arg->out_addr) + (waiter() ? 0 : O_WORDS), 0, false};
  if (tid == 0) {
    vx_mbar_init(c.mbar, 1);
    *reinterpret_cast<volatile uint32_t*>(lm + LM_FAIL) = 0;
    if (block == 0) {
      c.o[O_ERR] = 0;
      c.o[O_N] = 0;
      knobs(c.o + O_KNOBS, arg);
    }
  }
  vx_fence_proxy_async();
  __syncthreads();
  // Warp 1 writes its copy of the outputs past warp 0's, which the host reads.
  if (block == 0 || arg->test == T_SHARE)
    run(c, block);
  __syncthreads();
}
