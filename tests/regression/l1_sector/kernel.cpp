#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

#define INL static inline __attribute__((always_inline))

#define R4(x)  x x x x
#define R16(x) R4(R4(x))
#define R64(x) R4(R16(x))

INL void fence(uint32_t kind) {
  if (kind == OP_FENCE_GPU) {
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
  } else if (kind == OP_FENCE_CTA) {
#if VX_CFG_EXT_DLMEM_ENABLED
    __asm__ volatile (".insn r %0, 7, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM2) : "memory");
#endif
  } else if (kind == OP_FENCE_CLUSTER) {
    vx_fence_cluster();
  }
}

// Cycles from a load's issue until an instruction that uses it can issue.
INL uint64_t timed_lw(uint64_t a) {
  uint64_t t0 = vx_rdcycle();
  uint64_t v;
  __asm__ volatile ("lw %0, 0(%1)\n add %0, %0, zero\n" : "=&r"(v) : "r"(a) : "memory");
  uint64_t t1 = vx_rdcycle();
  __asm__ volatile ("" :: "r"(v));
  return t1 - t0;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t lane = vx_thread_id();
  const uint32_t wid  = vx_warp_id();
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);

  if (arg->kind == LS_BW || arg->kind == LS_BW_ALU) {
    uint64_t a = arg->start_addr + (wid % 8) * 128 + lane * 4;
    const uint32_t n = arg->n;
    uint64_t r0, r1, r2, r3, sink;
    __asm__ volatile ("lw %0, 0(%1)" : "=r"(r0) : "r"(a) : "memory");
    __syncthreads();
    uint64_t t0 = vx_rdcycle();
    if (arg->kind == LS_BW) {
      for (uint32_t i = 0; i < n; ++i) {
        __asm__ volatile (R4("lw %0, 0(%4)\n lw %1, 0(%4)\n lw %2, 0(%4)\n lw %3, 0(%4)\n")
                          : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3) : "r"(a) : "memory");
      }
    } else {
      for (uint32_t i = 0; i < n; ++i) {
        __asm__ volatile (R4("add %0, %4, zero\n add %1, %4, zero\n add %2, %4, zero\n add %3, %4, zero\n")
                          : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3) : "r"(a) : "memory");
      }
    }
    __asm__ volatile ("add %0, %1, %2\n add %0, %0, %3\n add %0, %0, %4\n"
                      : "=&r"(sink) : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    uint64_t t1 = vx_rdcycle();
    if (lane == 0) {
      out[wid * 2 + 0] = t0;
      out[wid * 2 + 1] = t1;
      out[64 + wid] = sink;
    }
    return;
  }

  if (wid != 0) {
    return;
  }

  if (arg->kind == LS_CHASE) {
    if (lane < arg->lanes) {
      auto starts = reinterpret_cast<const uint64_t*>(arg->start_addr);
      uint64_t p = starts[lane];
      for (uint32_t i = 0; i < arg->warm; ++i) {
        __asm__ volatile ("ld %0, 0(%0)\n" : "+r"(p) :: "memory");
      }
      // Restart the timed part on the same lines the warm part touched.
      p = starts[lane];
      fence(arg->fence);
      const uint32_t n = arg->hops / LS_UNROLL;
      uint64_t t0 = vx_rdcycle();
      for (uint32_t i = 0; i < n; ++i) {
        __asm__ volatile (R64("ld %0, 0(%0)\n") : "+r"(p) :: "memory");
      }
      uint64_t t1 = vx_rdcycle();
      if (lane == 0) {
        out[0] = t1 - t0;
        out[1] = p;
      }
    }
    return;
  }

  if (lane != 0) {
    return;
  }

  if (arg->kind == LS_OPS) {
    auto ops = reinterpret_cast<const uint64_t*>(arg->start_addr);
    for (uint32_t i = 0; i < arg->nops; ++i) {
      uint32_t op = (uint32_t)ops[2 * i];
      uint64_t a = ops[2 * i + 1];
      uint64_t c = 0;
      if (op == OP_LOAD) {
        uint64_t v;
        __asm__ volatile ("lw %0, 0(%1)\n add %0, %0, zero\n" : "=&r"(v) : "r"(a) : "memory");
      } else if (op == OP_STORE) {
        __asm__ volatile ("sw %0, 0(%1)" :: "r"(a), "r"(a) : "memory");
      } else if (op == OP_TIMED) {
        c = timed_lw(a);
      } else {
        fence(op);
      }
      out[i] = c;
    }
    return;
  }

  // LS_SWEEP: the out writes allocate L1 lines too, so they wait until the end.
  {
    const uint64_t base = arg->start_addr;
    const uint32_t n = arg->n, stride = arg->stride, laps = arg->laps;
    const uint32_t first_timed = arg->laps - arg->timed_laps;
    const uint64_t thr = arg->miss_thr;
    const uint32_t tail = arg->tail_from;
    fence(OP_FENCE_GPU);
    uint32_t misses = 0;
    for (uint32_t l = 0; l < laps; ++l) {
      const bool timed = (l >= first_timed);
      for (uint32_t i = (timed ? tail : 0); i < n; ++i) {
        uint64_t a = base + (uint64_t)i * stride;
        if (timed) {
          misses += (timed_lw(a) > thr);
        } else {
          uint64_t v;
          __asm__ volatile ("lw %0, 0(%1)\n add %0, %0, zero\n" : "=&r"(v) : "r"(a) : "memory");
        }
      }
    }
    out[0] = misses;
  }
}
