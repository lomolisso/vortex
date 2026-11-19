// The opt-in cases, in a kernel of their own (kernel_cas.vxbin) so that
// kernel.vxbin, and the cycles of every default case, stay as they are.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#if VX_CFG_EXT_ZACAS_ENABLED
// Compare-and-swap whose comparand is not zero, hit and miss, at both widths
// and both halves of a doubleword.
static inline void cas_nonzero(uint32_t* line, uint32_t hid) {
  uint32_t w = CASG_W_SEED(hid);
  asm volatile ("amocas.w %0, %2, (%1)" : "+r"(w) : "r"(&line[CASG_W_HIT]), "r"(CASG_W_NEW(hid)) : "memory");
  line[CASG_W_HIT_RET] = w;
  w = CASG_W_MISS_CMP(hid);
  asm volatile ("amocas.w %0, %2, (%1)" : "+r"(w) : "r"(&line[CASG_W_MISS]), "r"(CASG_W_NEW(hid)) : "memory");
  line[CASG_W_MISS_RET] = w;
  w = CASG_W_ODD_SEED(hid);
  asm volatile ("amocas.w %0, %2, (%1)" : "+r"(w) : "r"(&line[CASG_W_ODD]), "r"(CASG_W_NEW(hid)) : "memory");
  line[CASG_W_ODD_RET] = w;
  asm volatile ("amoadd.w %0, %2, (%1)" : "=r"(w) : "r"(&line[CASG_W_ODD]), "r"(CASG_W_ODD_ADD) : "memory");
  line[CASG_W_ODD_ADD_RET] = w;
#if __riscv_xlen == 64
  auto dline = (uint64_t*)line;
  uint64_t d = CASG_D_SEED(hid);
  asm volatile ("amocas.d %0, %2, (%1)" : "+r"(d) : "r"(&dline[CASG_D_HIT / 2]), "r"(CASG_D_NEW(hid)) : "memory");
  dline[CASG_D_HIT_RET / 2] = d;
  d = CASG_D_SEED(hid) + 1;
  asm volatile ("amocas.d %0, %2, (%1)" : "+r"(d) : "r"(&dline[CASG_D_MISS / 2]), "r"(CASG_D_NEW(hid)) : "memory");
  dline[CASG_D_MISS_RET / 2] = d;
#endif
}

// cas-global: on the hart's line of the shared buffer.
static void kernel_cas_global(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t hid = (uint32_t)vx_hart_id();
  cas_nonzero((uint32_t*)arg->shared_addr + hid * CASG_LINE_WORDS, hid);
}

// cas-local: on a local-memory copy of that line, copied back after.
static void kernel_cas_local(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t hid = (uint32_t)vx_hart_id();
  auto glob  = (uint32_t*)arg->shared_addr + hid * CASG_LINE_WORDS;
  auto local = (uint32_t*)__local_mem() + threadIdx.x * CASG_LINE_WORDS;
  for (uint32_t i = 0; i < CASG_LINE_WORDS; ++i)
    local[i] = glob[i];
  cas_nonzero(local, hid);
  for (uint32_t i = 0; i < CASG_LINE_WORDS; ++i)
    glob[i] = local[i];
}
#endif

#if __riscv_xlen == 64
static const uint32_t LRSCW_MAX_RETRIES = 4096;

// lrsc_wide_lmem: lr.d/sc.d increments of a doubleword's low word race with
// writes to its high word only, so a reservation that misses a write to the
// upper half lets the sc.d put back a stale high word. Roles go by warp, since
// the lanes of one warp take a divergent branch one side at a time and never
// race. Phase 0: the odd warps amoadd.w the high word. Phase 1: one lane of
// warp 1 increments it with a plain load and store, the only writer of it, so
// any write-back of a stale value loses an increment for good.
static void kernel_lrsc_wide_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto pair = (volatile uint64_t*)__local_mem();
  auto out  = (uint32_t*)arg->shared_addr;
  const uint32_t lane = threadIdx.x % vx_num_threads();
  const uint32_t warp = threadIdx.x / vx_num_threads();

  if (threadIdx.x == 0) {
    pair[0] = 0;
    pair[1] = 0;
  }
  __syncthreads();

  for (uint32_t phase = 0; phase < 2; ++phase) {
    auto dword = &pair[phase];
    auto high  = (volatile uint32_t*)dword + 1;
    if ((warp & 1) == 0) {
      for (uint32_t i = 0; i < arg->iters; ++i) {
        uint64_t old, sc_fail;
        uint32_t guard = 0;
        do {
          asm volatile ("lr.d %0, (%1)" : "=r"(old) : "r"(dword) : "memory");
          asm volatile ("sc.d %0, %2, (%1)" : "=r"(sc_fail) : "r"(dword), "r"(old + 1) : "memory");
          ++guard;
        } while (sc_fail && guard < LRSCW_MAX_RETRIES);
      }
    } else if (phase == 0) {
      for (uint32_t i = 0; i < arg->iters; ++i) {
        asm volatile ("amoadd.w zero, %1, (%0)" : : "r"(high), "r"(1) : "memory");
      }
    } else if (warp == 1 && lane == 0) {
      for (uint32_t i = 0; i < arg->iters; ++i) {
        *high = *high + 1;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < LRSCW_OUT_WORDS; ++i)
      out[i] = ((volatile uint32_t*)pair)[i];
  }
}
#endif

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
#if __riscv_xlen == 64
  if (arg->testid == LRSCW_TEST) {
    kernel_lrsc_wide_lmem(arg);
    return;
  }
#endif
#if VX_CFG_EXT_ZACAS_ENABLED
  if (arg->testid == CASG_TEST_GLOBAL)
    kernel_cas_global(arg);
  else
    kernel_cas_local(arg);
#else
  (void)arg;
#endif
}
