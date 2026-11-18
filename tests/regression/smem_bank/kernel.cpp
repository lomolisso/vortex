#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

#define INL static inline __attribute__((always_inline))

#define R4(x)  x x x x
#define R16(x) R4(R4(x))
#define R32(x) R16(x) R16(x)
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

// The chase operands live in fixed registers of distinct GPR banks, so no hop pays an
// operand-bank conflict that depends on the compiler's register choice.
INL void chase(const kernel_arg_t* arg, char* base, uint32_t off, uint64_t* res) {
  register uint64_t b __asm__("t1") = reinterpret_cast<uint64_t>(base);
  register uint64_t zero __asm__("t2") = opaque(0);
  register uint64_t p __asm__("t0") = b + off;
  register uint64_t t __asm__("a5") = 0;
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
                    "1:\n" R32(OP) "addi %[cnt], %[cnt], -1\n bnez %[cnt], 1b\n"                          \
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
  // SB_MIX classes: 0 LDS, 1 an L1-hit line of its own, 2 STS.
  const uint32_t cls = (arg->kind != SB_MIX) ? 0
                     : (wid < arg->ncls[0]) ? 0 : (wid < arg->ncls[0] + arg->ncls[1]) ? 1 : 2;
  const uint64_t a = (cls == 1) ? arg->gbl_addr + wid * 128 + off
                                : reinterpret_cast<uint64_t>(base) + (wid % arg->rows) * arg->row_bytes + off;
  const uint64_t one = opaque(1);
  const uint32_t bar = get_local_group_id();
  const uint32_t nwarps = get_num_sub_groups();
  uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  uint64_t cnt = 0, ph = 0, sink = 0, t0 = 0, t1 = 0;
  if (cls == 2) {
    SB_TIMED("sw %[one], 0(%[a])\n sw %[one], 0(%[a])\n sw %[one], 0(%[a])\n sw %[one], 0(%[a])\n");
  } else if (arg->kind == SB_LOAD_BW || arg->kind == SB_MIX) {
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

#define SB_FLAG_OFF 64
#define SB_DONE_OFF 72

// Waits for `n` warps with one poll per sixteen dependent divides: a waiting warp leaves the issue slots and
// the shared-memory port to the warps still timed, and reaches the closing barrier (whose LSU drain gate is
// SM-wide) only when every timed warp is done.
INL void idle_until(volatile uint32_t* cnt, uint32_t n) {
  float x = 1.0f, y = 1.0f;
  __asm__ volatile ("" : "+f"(y));
  while (*cnt < n) {
    __asm__ volatile (R16("fdiv.s %0, %0, %1\n") : "+f"(x) : "f"(y));
  }
}

// A one-lane chase in warp 0 while the last `miss_warps` warps stream L2-hit lines through the L1 (8 lines
// in flight per warp, a footprint well above the L1), or LDS stream warps beside the same misses. The chase
// runs in warp 0 because the warp pick is fixed-priority (lowest warp first), where the H100's four schedulers
// do not starve one warp. The streamers stop when the victim is done.
INL void under_miss(const kernel_arg_t* arg, char* base, uint32_t lane, uint32_t wid, uint64_t* out) {
  volatile uint32_t* flag = reinterpret_cast<volatile uint32_t*>(base + SB_FLAG_OFF);
  volatile uint32_t* done = reinterpret_cast<volatile uint32_t*>(base + SB_DONE_OFF);
  // Fields read in the timed loops live in locals: the asm's memory clobber would reload them from the
  // argument's line on every iteration, and the miss stream evicts that line.
  const uint64_t nlines = arg->gbl_bytes / 128, gbl = arg->gbl_addr;
  const uint32_t iters = arg->iters, hops = arg->hops;
  const uint32_t nwarps = get_num_sub_groups();
  const uint32_t first_miss = nwarps - arg->miss_warps;
  const uint64_t step = 8 * arg->miss_warps;
  const bool miss = (wid >= first_miss);
  char* gchain = reinterpret_cast<char*>(arg->gbl_addr + arg->gbl_bytes);
  const bool l1 = (arg->um_kind == SB_UM_L1_CHASE);
  char* cbase = l1 ? gchain : base;
  if (wid == 0 && lane == 0) {
    if (arg->um_kind != SB_UM_LDS_STREAM) {
      init_chain(cbase, arg, 0, false);
    }
    *flag = 0;
    *done = 0;
  }
  vx_fence();
  __syncthreads();
  uint64_t r[8], sink = 0;
  if (miss) {
    for (uint64_t line = (wid - first_miss) * 8; line < nlines; line += step) {
      const uint64_t a = gbl + line * 128 + lane * 4;
      __asm__ volatile ("lw %0, 0(%8)\n lw %1, 128(%8)\n lw %2, 256(%8)\n lw %3, 384(%8)\n"
                        "lw %4, 512(%8)\n lw %5, 640(%8)\n lw %6, 768(%8)\n lw %7, 896(%8)\n"
                        : "=&r"(r[0]), "=&r"(r[1]), "=&r"(r[2]), "=&r"(r[3]), "=&r"(r[4]), "=&r"(r[5]),
                          "=&r"(r[6]), "=&r"(r[7]) : "r"(a) : "memory");
      sink += r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];
    }
  }
  __syncthreads();
  uint64_t t0 = vx_rdcycle(), t1 = t0, amount = 0, maxd = 0;
  if (miss) {
    // No per-line consume: the next pass's writes to r0..r7 already keep 8 lines in flight, and the loop
    // stays a few instructions per line so the issue rate does not bound the read-out.
    const uint64_t first = gbl + (wid - first_miss) * 8 * 128 + lane * 4, end = gbl + nlines * 128;
    uint64_t a = first;
    while (*flag == 0) {
      __asm__ volatile ("lw %0, 0(%8)\n lw %1, 128(%8)\n lw %2, 256(%8)\n lw %3, 384(%8)\n"
                        "lw %4, 512(%8)\n lw %5, 640(%8)\n lw %6, 768(%8)\n lw %7, 896(%8)\n"
                        : "=&r"(r[0]), "=&r"(r[1]), "=&r"(r[2]), "=&r"(r[3]), "=&r"(r[4]), "=&r"(r[5]),
                          "=&r"(r[6]), "=&r"(r[7]) : "r"(a) : "memory");
      amount += 8 * 128;
      a += step * 128;
      if (a >= end) a -= nlines * 128;
    }
    t1 = vx_rdcycle();
    sink += r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];
  } else if (arg->um_kind == SB_UM_LDS_STREAM) {
    // 8-B lanes: two wavefronts per instruction, so the port, not the issue rate, bounds the stream.
    const uint64_t a = reinterpret_cast<uint64_t>(base) + (wid % arg->rows) * arg->row_bytes + lane * SB_UM_LANE_BYTES;
    t0 = vx_rdcycle();
    for (uint32_t i = 0; i < iters; ++i) {
#if __riscv_xlen == 64
      __asm__ volatile (R16("ld %0, 0(%4)\n ld %1, 0(%4)\n ld %2, 0(%4)\n ld %3, 0(%4)\n")
                        : "=&r"(r[0]), "=&r"(r[1]), "=&r"(r[2]), "=&r"(r[3]) : "r"(a) : "memory");
#else
      __asm__ volatile (R16("lw %0, 0(%4)\n lw %1, 0(%4)\n lw %2, 0(%4)\n lw %3, 0(%4)\n")
                        : "=&r"(r[0]), "=&r"(r[1]), "=&r"(r[2]), "=&r"(r[3]) : "r"(a) : "memory");
#endif
      sink += r[0] + r[1] + r[2] + r[3];
    }
    t1 = vx_rdcycle();
    amount = (uint64_t)iters * 64 * vx_num_threads() * SB_UM_LANE_BYTES;
    if (lane == 0) *flag = 1;
  } else if (lane == 0) {
    register uint64_t p __asm__("t0") = reinterpret_cast<uint64_t>(cbase);
    register uint64_t b __asm__("t1") = p;
    // The first pass warms the chase's code and lines; the second is timed.
    for (uint32_t pass = 0; pass < 2; ++pass) {
      t0 = vx_rdcycle();
      if (arg->instr) {
        uint64_t prev = t0, t;
        maxd = 0;
        for (uint32_t i = 0; i < hops; ++i) {
          __asm__ volatile ("lw %0, 0(%0)\n add %0, %0, %2\n csrr %1, %3\n"
                            : "+r"(p), "=r"(t) : "r"(b), "i"(VX_CSR_MCYCLE) : "memory");
          // The first hop also waits for the loop's prologue reloads, which the miss stream evicted.
          if (i != 0) maxd = (t - prev > maxd) ? t - prev : maxd;
          prev = t;
        }
        t1 = prev;
      } else {
        const uint32_t n = hops / SB_UNROLL;
        for (uint32_t i = 0; i < n; ++i) {
          __asm__ volatile (R64("lw %0, 0(%0)\n add %0, %0, %1\n") : "+r"(p) : "r"(b) : "memory");
        }
        t1 = vx_rdcycle();
      }
    }
    amount = hops;
    sink += p;
    *flag = 1;
  }
  if (lane == 0) {
    out[wid * 4 + 0] = t0;
    out[wid * 4 + 1] = t1;
    out[wid * 4 + 2] = amount;
    out[wid * 4 + 3] = maxd;
    __atomic_fetch_add(const_cast<uint32_t*>(done), 1u, __ATOMIC_RELAXED);
  }
  __asm__ volatile ("" :: "r"(sink));
  idle_until(done, nwarps);
  __syncthreads();
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  char* base = reinterpret_cast<char*>(__local_mem());
  const uint32_t lane = vx_thread_id();
  const uint32_t wid  = vx_warp_id();
  const uint32_t nw   = vx_num_warps();
  auto pat = reinterpret_cast<const uint32_t*>(arg->pat_addr);
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
  if (arg->kind == SB_UNDER_MISS) {
    under_miss(arg, base, lane, wid, out);
    return;
  }
  const bool tput = (arg->kind == SB_ATOM_TPUT || arg->kind == SB_LOAD_BW || arg->kind == SB_MIX);
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
