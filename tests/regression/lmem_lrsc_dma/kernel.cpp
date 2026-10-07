#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

// Each round, some lanes of the one warp reserve a local-memory word, the warp
// copies a tile into local memory with the DXA, and the lanes then try their
// store-conditional. A DXA write is a write by another device: an SC whose
// reserved bytes it wrote must fail and leave the copied data in place.

constexpr uint32_t kDescSrc = 0;

static inline uint32_t lr_w(uint8_t* p) {
  uint32_t v;
  asm volatile ("lr.w %0, (%1)" : "=r"(v) : "r"(p) : "memory");
  return v;
}

static inline uint32_t sc_w(uint8_t* p, uint32_t v) {
  uint32_t rc;
  asm volatile ("sc.w %0, %2, (%1)" : "=r"(rc) : "r"(p), "r"(v) : "memory");
  return rc;
}

#if __riscv_xlen == 64
static inline uint64_t lr_d(uint8_t* p) {
  uint64_t v;
  asm volatile ("lr.d %0, (%1)" : "=r"(v) : "r"(p) : "memory");
  return v;
}

static inline uint32_t sc_d(uint8_t* p, uint64_t v) {
  uint64_t rc;
  asm volatile ("sc.d %0, %2, (%1)" : "=r"(rc) : "r"(p), "r"(v) : "memory");
  return (uint32_t)rc;
}
#endif

static inline uint64_t reserve(uint8_t* p, uint32_t dword) {
#if __riscv_xlen == 64
  if (dword) {
    return lr_d(p);
  }
#endif
  (void)dword;
  return lr_w(p);
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t lane = threadIdx.x;
  auto lmem = reinterpret_cast<uint8_t*>(
    (reinterpret_cast<uintptr_t>(__local_mem()) + LMEM_ALIGN - 1) & ~uintptr_t(LMEM_ALIGN - 1));
  auto plan = reinterpret_cast<const round_plan_t*>(arg->plan_addr);
  auto out  = reinterpret_cast<lane_out_t*>(arg->out_addr);
  const uint32_t hart = vx_hart_id();

  vortex::barrier bar(1, 1);

  for (uint32_t r = 0; r < arg->num_rounds; ++r) {
    const round_plan_t* rp = plan + r;
    lane_plan_t lp = {};
    if (lane < MAX_LANES) {
      lp = rp->lanes[lane];
    }
    uint8_t* word = lmem + lp.word_off;
    lane_out_t* o = out + r * MAX_LANES + (lane % MAX_LANES);

    // The global store waits on the LR's result, so the reservation is in
    // place before the copy is issued.
    if (lp.active) {
      const uint64_t s = sentinel_value(r, lane);
      if (lp.dword) {
        *reinterpret_cast<volatile uint64_t*>(word) = s;
      } else {
        *reinterpret_cast<volatile uint32_t*>(word) = (uint32_t)s;
      }
      o->old  = reserve(word, lp.dword);
      o->hart = hart;
    }

    bar.expect_tx(1);
    vx_dxa_issue_1d_wg(kDescSrc, bar.id(), lmem + rp->dst_off, rp->src_coord);
    bar.arrive_and_wait();

    // The barrier may release before the copy's last write is visible; the
    // SC must not race the very write it is meant to observe.
    auto probe = reinterpret_cast<volatile uint32_t*>(lmem + rp->probe_off);
    const uint32_t probe_val = rp->probe_val;
    uint32_t spins = 0;
    while (*probe != probe_val && spins < SPIN_BOUND) {
      ++spins;
    }

    if (lp.active) {
      if (lp.re_lr) {
        o->old2 = reserve(word, lp.dword);
      }
      const uint64_t m = marker_value(r, lane);
      uint32_t rc;
#if __riscv_xlen == 64
      if (lp.dword) {
        rc = sc_d(word, m);
        o->final = *reinterpret_cast<volatile uint64_t*>(word);
      } else
#endif
      {
        rc = sc_w(word, (uint32_t)m);
        o->final = *reinterpret_cast<volatile uint32_t*>(word);
      }
      o->rc    = rc;
      o->spins = spins;
    }
  }
}
