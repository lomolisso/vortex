// The opt-in compare-and-swap cases, in a kernel of their own (kernel_cas.vxbin)
// so that kernel.vxbin, and the cycles of every default case, stay as they are.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#if VX_CFG_EXT_ZACAS_ENABLED
// Compare-and-swap whose comparand is not zero, hit and miss, at both widths.
static inline void cas_nonzero(uint32_t* line, uint32_t hid) {
  uint32_t w = CASG_W_SEED(hid);
  asm volatile ("amocas.w %0, %2, (%1)" : "+r"(w) : "r"(&line[CASG_W_HIT]), "r"(CASG_W_NEW(hid)) : "memory");
  line[CASG_W_HIT_RET] = w;
  w = CASG_W_SEED(hid) + 1;
  asm volatile ("amocas.w %0, %2, (%1)" : "+r"(w) : "r"(&line[CASG_W_MISS]), "r"(CASG_W_NEW(hid)) : "memory");
  line[CASG_W_MISS_RET] = w;
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

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
#if VX_CFG_EXT_ZACAS_ENABLED
  if (arg->testid == CASG_TEST_GLOBAL)
    kernel_cas_global(arg);
  else
    kernel_cas_local(arg);
#else
  (void)arg;
#endif
}
