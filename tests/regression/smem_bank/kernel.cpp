#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

#define INL static inline __attribute__((always_inline))

#define R4(x)  x x x x
#define R16(x) R4(R4(x))
#define R64(x) R4(R16(x))

INL uint64_t opaque(uint64_t v) {
  __asm__ volatile ("" : "+r"(v));
  return v;
}

// One chain per lane through `rows` rows; the word at a row holds the next
// row's offset (32-bit kinds) or address (64-bit), so every hop repeats the
// lane pattern of the first.
INL void init_chain(char* base, const kernel_arg_t* arg, uint32_t off, bool wide) {
  for (uint32_t k = 0; k < arg->rows; ++k) {
    uint32_t next = ((k + 1) % arg->rows) * arg->row_bytes + off;
    char* p = base + k * arg->row_bytes + off;
    if (wide) {
      *reinterpret_cast<volatile uint64_t*>(p) = reinterpret_cast<uint64_t>(base) + next;
    } else {
      *reinterpret_cast<volatile uint32_t*>(p) = next;
    }
  }
}

INL void chase(const kernel_arg_t* arg, char* base, uint32_t off, uint64_t* res) {
  uint64_t b = reinterpret_cast<uint64_t>(base);
  uint64_t zero = opaque(0);
  uint64_t p = b + off;
  uint64_t t = 0;
  const uint32_t n = arg->hops / SB_UNROLL;
  uint64_t t0, t1, t2, t3;
  switch (arg->kind) {
  case SB_CHASE32:
    t0 = vx_rdcycle();
    for (uint32_t i = 0; i < n; ++i) {
      __asm__ volatile (R64("lw %0, 0(%0)\n add %0, %0, %1\n") : "+r"(p) : "r"(b) : "memory");
    }
    t1 = vx_rdcycle();
    t2 = t3 = 0;
    break;
  case SB_CHASE64:
    t0 = vx_rdcycle();
    for (uint32_t i = 0; i < n; ++i) {
      __asm__ volatile (R64("ld %0, 0(%0)\n") : "+r"(p) :: "memory");
    }
    t1 = vx_rdcycle();
    t2 = t3 = 0;
    if (arg->calib) {
      t2 = vx_rdcycle();
      for (uint32_t i = 0; i < n; ++i) {
        __asm__ volatile (R64("ld %0, 0(%0)\n add %0, %0, %1\n") : "+r"(p) : "r"(zero) : "memory");
      }
      t3 = vx_rdcycle();
    }
    break;
  default: // SB_ATOM_LAT
    if (arg->cas) {
#ifdef __riscv_zacas
      t0 = vx_rdcycle();
      for (uint32_t i = 0; i < n; ++i) {
        __asm__ volatile (R64("mv %1, zero\n amocas.w %1, zero, (%0)\n add %0, %1, %2\n")
                          : "+r"(p), "+r"(t) : "r"(b) : "memory");
      }
      t1 = vx_rdcycle();
      t2 = t3 = 0;
#else
      t0 = t1 = t2 = t3 = 0;
#endif
    } else {
      t0 = vx_rdcycle();
      for (uint32_t i = 0; i < n; ++i) {
        __asm__ volatile (R64("amoadd.w %0, zero, (%0)\n add %0, %0, %1\n") : "+r"(p) : "r"(b) : "memory");
      }
      t1 = vx_rdcycle();
      t2 = t3 = 0;
      if (arg->calib) {
        t2 = vx_rdcycle();
        for (uint32_t i = 0; i < n; ++i) {
          __asm__ volatile (R64("mv %1, zero\n amoadd.w %1, %1, (%0)\n add %0, %1, %2\n")
                            : "+r"(p), "+r"(t) : "r"(b) : "memory");
        }
        t3 = vx_rdcycle();
      }
    }
    break;
  }
  res[0] = t1 - t0;
  res[1] = t3 - t2;
  res[2] = p + t;
}

// An untimed pass, a CTA barrier, the timed pass and a CTA barrier in one asm
// block: the timed pass runs from the I-cache, and no warp's global accesses
// (spill reloads, result stores) fall inside the window, where the LSU would
// queue other warps' shared-memory requests behind them. The end time waits for
// the last results, not only for their issue.
#define SB_TIMED(OP)                                                                                      \
  __asm__ volatile ("mv %[cnt], %[warm]\n li %[ph], 0\n"                                                  \
                    "1:\n" R16(OP) "addi %[cnt], %[cnt], -1\n bnez %[cnt], 1b\n"                          \
                    "add %[s], %[r0], %[r1]\n add %[s], %[s], %[r2]\n add %[s], %[s], %[r3]\n"            \
                    "csrr %[t1], %[mcycle]\n bnez %[ph], 2f\n"                                            \
                    "li %[ph], 1\n mv %[cnt], %[n]\n"                                                     \
                    ".insn r %[custom0], 4, 0, x0, %[bar], %[nwarps]\n"                                   \
                    "csrr %[t0], %[mcycle]\n j 1b\n"                                                      \
                    "2:\n .insn r %[custom0], 4, 0, x0, %[bar], %[nwarps]\n"                              \
                    : [r0] "+&r"(r0), [r1] "+&r"(r1), [r2] "+&r"(r2), [r3] "+&r"(r3), [cnt] "=&r"(cnt),   \
                      [ph] "=&r"(ph), [s] "=&r"(sink), [t0] "=&r"(t0), [t1] "=&r"(t1)                     \
                    : [a] "r"(a), [one] "r"(one), [warm] "r"(warm), [n] "r"(n), [bar] "r"(bar),           \
                      [nwarps] "r"(nwarps), [mcycle] "i"(VX_CSR_MCYCLE), [custom0] "i"(RISCV_CUSTOM0)     \
                    : "memory")

INL void throughput(const kernel_arg_t* arg, char* base, uint32_t off, uint32_t wid, uint64_t* res) {
  const uint64_t n = arg->iters;
  const uint64_t warm = 1;
  const uint64_t a = reinterpret_cast<uint64_t>(base) + (wid % arg->rows) * arg->row_bytes + off;
  const uint64_t one = opaque(1);
  const uint32_t bar = get_local_group_id();
  const uint32_t nwarps = get_num_sub_groups();
  uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  uint64_t cnt = 0, ph = 0, sink = 0, t0 = 0, t1 = 0;
  if (arg->kind == SB_LOAD_BW) {
    SB_TIMED("lw %[r0], 0(%[a])\n lw %[r1], 0(%[a])\n lw %[r2], 0(%[a])\n lw %[r3], 0(%[a])\n");
  } else if (arg->cas) {
#ifdef __riscv_zacas
    SB_TIMED("amocas.w %[r0], %[one], (%[a])\n amocas.w %[r1], %[one], (%[a])\n"
             "amocas.w %[r2], %[one], (%[a])\n amocas.w %[r3], %[one], (%[a])\n");
#endif
  } else {
    SB_TIMED("amoadd.w %[r0], %[one], (%[a])\n amoadd.w %[r1], %[one], (%[a])\n"
             "amoadd.w %[r2], %[one], (%[a])\n amoadd.w %[r3], %[one], (%[a])\n");
  }
  res[2] = sink;
  res[0] = t0;
  res[1] = t1;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  char* base = reinterpret_cast<char*>(__local_mem());
  const uint32_t lane = vx_thread_id();
  const uint32_t wid  = vx_warp_id();
  const uint32_t nw   = vx_num_warps();
  auto pat = reinterpret_cast<const uint32_t*>(arg->pat_addr);
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
  const bool tput = (arg->kind == SB_ATOM_TPUT || arg->kind == SB_LOAD_BW);
  const bool wide = (arg->kind == SB_CHASE64);

  for (uint32_t p = 0; p < arg->npat; ++p) {
    const uint32_t off = pat[p * SB_LANES + lane];
    const bool active = (off != SB_INACTIVE);
    if (!tput || wid == 0) {
      if (active) {
        init_chain(base, arg, off, wide);
      }
    }
    vx_fence();
    __syncthreads();
    if (tput) {
      if (active) {
        uint64_t res[3];
        throughput(arg, base, off, wid, res);
        if (lane == 0) {
          out[(p * nw + wid) * 2 + 0] = res[0];
          out[(p * nw + wid) * 2 + 1] = res[1];
        }
      }
    } else if (wid == 0 && active) {
      uint64_t res[3];
      chase(arg, base, off, res);
      if (lane == 0) {
        out[p * 3 + 0] = res[0];
        out[p * 3 + 1] = res[1];
        out[p * 3 + 2] = res[2];
      }
    }
    __syncthreads();
  }
}
