#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// The sequences below are straight-line asm so the compiler can neither
// reorder nor merge the accesses whose order is under test. Each line's miss
// chain is headed by a load of another word that the fill forward answers,
// which leaves the bank input open to a younger request while the rest of the
// chain still waits to replay. Several warps in flight keep requests queued at
// the bank input, so one of them is admitted in exactly that window.

#define R1 "lw %[r1], 32(%[p])\n"
#define R2 "lw %[r1], 32(%[p])\nlw %[r2], 40(%[p])\n"

// war: L reads word 0 before X overwrites it; S, a store to another word,
// closes the forward window behind the leading load(s).
#define WAR_PROBE(RS, A, B, C)                               \
  asm volatile(                                              \
    ".option push\n.option norvc\n"                          \
    RS                                                       \
    ".rept " #A "\nnop\n.endr\n"                             \
    "sw   %[sv], 16(%[p])\n"                                 \
    ".rept " #B "\nnop\n.endr\n"                             \
    "lw   %[l], 0(%[p])\n"                                   \
    ".rept " #C "\nnop\n.endr\n"                             \
    "sw   %[mk], 0(%[p])\n"                                  \
    ".option pop\n"                                          \
    : [r1]"=&r"(r1), [r2]"=&r"(r2), [l]"=&r"(l)              \
    : [p]"r"(p), [sv]"r"(sv), [mk]"r"(mk)                    \
    : "memory")

static __attribute__((always_inline)) inline
uint32_t war_probe(uint32_t v, uintptr_t p, uint32_t sv, uint32_t mk) {
  uint32_t r1 = 0, r2 = 0, l;
  switch (v) {
  default:
  case 0:  WAR_PROBE(R1, 0, 0, 0); break;
  case 1:  WAR_PROBE(R1, 0, 0, 1); break;
  case 2:  WAR_PROBE(R2, 0, 0, 0); break;
  case 3:  WAR_PROBE(R1, 1, 0, 0); break;
  case 4:  WAR_PROBE(R2, 0, 0, 1); break;
  case 5:  WAR_PROBE(R1, 0, 1, 0); break;
  case 6:  WAR_PROBE(R1, 0, 0, 2); break;
  case 7:  WAR_PROBE(R2, 1, 0, 0); break;
  case 8:  WAR_PROBE(R1, 1, 0, 1); break;
  case 9:  WAR_PROBE(R2, 0, 1, 0); break;
  case 10: WAR_PROBE(R1, 2, 0, 0); break;
  case 11: WAR_PROBE(R2, 0, 0, 2); break;
  }
  (void)r1; (void)r2;
  return l;
}

// raw: X reads word 0 after S wrote it.
#define RAW_PROBE(RS, A, B)                                  \
  asm volatile(                                              \
    ".option push\n.option norvc\n"                          \
    RS                                                       \
    ".rept " #A "\nnop\n.endr\n"                             \
    "sw   %[sv], 0(%[p])\n"                                  \
    ".rept " #B "\nnop\n.endr\n"                             \
    "lw   %[x], 0(%[p])\n"                                   \
    ".option pop\n"                                          \
    : [r1]"=&r"(r1), [r2]"=&r"(r2), [x]"=&r"(x)              \
    : [p]"r"(p), [sv]"r"(sv)                                 \
    : "memory")

static __attribute__((always_inline)) inline
uint32_t raw_probe(uint32_t v, uintptr_t p, uint32_t sv) {
  uint32_t r1 = 0, r2 = 0, x;
  switch (v) {
  default:
  case 0:  RAW_PROBE(R1, 0, 0); break;
  case 1:  RAW_PROBE(R1, 0, 1); break;
  case 2:  RAW_PROBE(R2, 0, 0); break;
  case 3:  RAW_PROBE(R1, 1, 0); break;
  case 4:  RAW_PROBE(R2, 0, 1); break;
  case 5:  RAW_PROBE(R1, 0, 2); break;
  case 6:  RAW_PROBE(R2, 1, 0); break;
  case 7:  RAW_PROBE(R1, 1, 1); break;
  case 8:  RAW_PROBE(R2, 0, 2); break;
  case 9:  RAW_PROBE(R1, 2, 0); break;
  case 10: RAW_PROBE(R2, 1, 1); break;
  case 11: RAW_PROBE(R1, 0, 3); break;
  }
  (void)r1; (void)r2;
  return x;
}

static __attribute__((noinline)) void run_probes(kernel_arg_t* arg) {
  uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t nh = arg->num_harts;
  if (h >= nh)
    return;
  uintptr_t buf = (uintptr_t)arg->buf_addr;
  uint32_t* out = (uint32_t*)arg->out_addr;
  uint32_t n = arg->iters;
  bool war = (arg->mode == MODE_WAR);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t v = probe_variant(h, i);
    uint32_t ln = probe_line(h, i, nh);
    uintptr_t p = buf + (uintptr_t)ln * PROBE_LINE;
    out[ln] = war ? war_probe(v, p, war_side(ln), war_marker(ln)) : raw_probe(v, p, raw_value(ln));
  }
}

// store_amo/store_lr: a back-to-back store and atomic on the thread's own
// word, behind two loads of another word of the line.
#define STORE_AMO_SEQ(XOP)                                               \
  asm volatile ("lw %[r0], 0(%[pr])\n\t"                                 \
                "lw %[r1], 0(%[pr])\n\t"                                 \
                "sw %[sv], 0(%[ps])\n\t"                                 \
                XOP "\n\t"                                               \
                : [r0] "=&r"(r0), [r1] "=&r"(r1), [old] "=&r"(old)       \
                : [pr] "r"(pr), [sv] "r"(sval), [ps] "r"(ps), [one] "r"(1u) \
                : "memory")

static __attribute__((noinline)) void run_store_amo(kernel_arg_t* arg) {
  const uint32_t tid = threadIdx.x;
  const uint32_t nt  = blockDim.x;
  const uint32_t lb  = VX_CFG_DCACHE_LINE_SIZE;
  const uint32_t dw  = VX_CFG_DCACHE_WORD_SIZE;
  const uint32_t iters = arg->iters;
  const bool lr = (arg->mode == MODE_STORE_LR);
  uint8_t* buf = (uint8_t*)arg->buf_addr;
  uint32_t* out = (uint32_t*)arg->out_addr;
  const uint32_t s_off = (tid * dw) % lb;
  const uint32_t r_off = (dw >= 16) ? (s_off + 8) : ((s_off + lb / 2) % lb);
  // result lines stay resident, so result stores never take bank MSHR entries
  for (uint32_t j = 0; j < iters * nt; j += 16) {
    (void)*(volatile uint32_t*)&out[j];
  }
  uint32_t sink = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    uint8_t* L = buf + (size_t)i * lb;
    uint32_t* pr = (uint32_t*)(L + r_off);
    uint32_t* ps = (uint32_t*)(L + s_off);
    uint32_t sval = STORE_AMO_BASE + i * 4 + tid;
    uint32_t r0 = 0, r1 = 0, old = 0;
    if (lr) {
      STORE_AMO_SEQ("lr.w %[old], (%[ps])");
    } else {
      STORE_AMO_SEQ("amoadd.w %[old], %[one], (%[ps])");
    }
    out[i * nt + tid] = old;
    sink += r0 + r1;
  }
  out[iters * nt + tid] = sink;
}

// Holds the warp for n (0..15) issue slots: a jump into a sled of nops.
static __attribute__((always_inline)) inline void delay(uint32_t n) {
  asm volatile(
    ".option push\n"
    ".option norvc\n"
    "la    t0, 9f\n"
    "slli  t1, %0, 2\n"
    "sub   t0, t0, t1\n"
    "jr    t0\n"
    ".rept 16\n"
    "nop\n"
    ".endr\n"
    "9:\n"
    ".option pop\n"
    :: "r"(n) : "t0", "t1", "memory");
}

// amo_store: thread 0 of two warps. Each round both pass a barrier, then
// warp 0 adds to slot i while warp 1 stores to it, one of them held for up
// to 15 issue slots, so the two requests meet the cache at every skew.
static __attribute__((noinline)) void run_amo_store(kernel_arg_t* arg) {
  auto buf  = (volatile uint32_t*)arg->buf_addr;
  auto olds = (uint32_t*)arg->out_addr;
  uint32_t nt = blockDim.x / 2;
  uint32_t w  = threadIdx.x / nt;
  uint32_t n  = arg->iters;
  vx_tmc_one();
  if (w == 0) {
    for (uint32_t i = 0; i < n; ++i) {
      volatile uint32_t* p = buf + i * SLOT_STRIDE_WORDS;
      uint32_t inc = slot_inc(i);
      uint32_t old;
      vx_barrier(0, 2);
      delay(slot_delay(i, 1));
      asm volatile("amoadd.w %0, %2, (%1)" : "=&r"(old) : "r"(p), "r"(inc) : "memory");
      olds[i] = old;
    }
  } else {
    for (uint32_t i = 0; i < n; ++i) {
      volatile uint32_t* p = buf + i * SLOT_STRIDE_WORDS;
      uint32_t val = slot_val(i);
      vx_barrier(0, 2);
      delay(slot_delay(i, 0));
      asm volatile("sw %1, 0(%0)" :: "r"(p), "r"(val) : "memory");
    }
  }
}

// amo_load: one warp, lane = hart. In the default two-bank dcache even lines
// map to one bank (the test bank), odd lines to the other. Per round:
//   L0: lw   B+32        lane 0: fresh even line B (heads B's miss chain)
//   Z : amoadd (Z_lane)  resident lines; lanes in zmask hit the test bank, so
//                        their serialized commits hold its input
//   Y : amoadd (B)       joins B's chain behind L0 when it gets in
//   X : lw     (B)       queued right behind Y
// Lanes 1..3 use private odd lines for L0/Y/X. Each lane's B word is written
// only by its own amoadd, so X must return old + 1.
#define AMO_LOAD_SEQ(PAD)                                         \
  asm volatile(                                                   \
    ".option push\n"                                              \
    ".option norvc\n"                                             \
    "lw    %[l0], 32(%[b])\n"                                     \
    PAD                                                           \
    "amoadd.w %[z], %[one], (%[zz])\n"                            \
    "amoadd.w %[o], %[one], (%[b])\n"                             \
    "lw    %[xv], 0(%[b])\n"                                      \
    ".option pop\n"                                               \
    : [l0]"=&r"(l0), [o]"=&r"(o), [xv]"=&r"(xv), [z]"=&r"(z)      \
    : [b]"r"(B), [zz]"r"(Z), [one]"r"(1)                          \
    : "memory")

static __attribute__((noinline)) void run_amo_load(kernel_arg_t* arg) {
  uint32_t lane = threadIdx.x;
  if (blockIdx.x != 0 || lane >= AMO_LOAD_LANES)
    return;
  uintptr_t base = (uintptr_t)arg->buf_addr;
  uint32_t mask = arg->pool_lines - 1;
  uint32_t* res = (uint32_t*)arg->out_addr + lane * AMO_LOAD_RES;
  uint32_t par0 = (uint32_t)((base >> 6) & 1);
  // resident Z lines: one even (test bank) and one odd per lane, at the pool start
  uintptr_t zt = base + (uintptr_t)((2 * lane) | par0) * 64;
  uintptr_t zo = base + (uintptr_t)((2 * lane) | (par0 ^ 1)) * 64;
  (void)*(volatile uint32_t*)zt;
  (void)*(volatile uint32_t*)zo;
  uint32_t next = 16;
  uint32_t seed = arg->seed * 747796405u + 2891336453u;
  uint32_t viol = 0, first_bad = ~0u, first_old = 0, first_x = 0;
  uint32_t sink = 0;
  for (uint32_t r = 0; r < arg->iters; ++r) {
    seed = seed * 1664525u + 1013904223u;          // same sequence in every lane
    uint32_t k1    = (seed >> 28) & 3;             // NOPs between L0 and Z
    uint32_t zmask = ((seed >> 16) & 15) | 1;      // lanes whose Z blocks the test bank
    uint32_t lb = next & mask;
    next += 16;
    if (next >= arg->pool_lines) next = 16;
    uintptr_t B = (lane == 0) ? base + (uintptr_t)((lb & ~1u) | par0) * 64
                              : base + (uintptr_t)((((lb + 2 * lane) & mask) & ~1u) | (par0 ^ 1)) * 64;
    uintptr_t Z = ((zmask >> lane) & 1) ? zt : zo;
    uint32_t o, xv, l0, z;
    if (k1 == 0) {
      AMO_LOAD_SEQ("");
    } else if (k1 == 1) {
      AMO_LOAD_SEQ("nop\n");
    } else if (k1 == 2) {
      AMO_LOAD_SEQ("nop\nnop\n");
    } else {
      AMO_LOAD_SEQ("nop\nnop\nnop\n");
    }
    sink += l0 + z;
    if (xv != o + 1) {
      if (viol == 0) { first_bad = r; first_old = o; first_x = xv; }
      ++viol;
    }
  }
  res[0] = viol;
  res[1] = first_bad;
  res[2] = first_old;
  res[3] = first_x + (sink & 0);
}

// amo_storm: every hart, every round, on its own fresh line B and its own
// resident line Z: a load of B heads B's miss chain, an atomic add on Z hits,
// and an atomic add on B joins B's chain and replays when the fill lands.
// Across harts, replayed atomics on cold lines meet atomic hits on resident
// lines of the same bank. Every add must land exactly once.
#define STORM_SEQ(PAD)                                            \
  asm volatile(                                                   \
    ".option push\n"                                              \
    ".option norvc\n"                                             \
    "lw    %[l0], 32(%[b])\n"                                     \
    PAD                                                           \
    "amoadd.w %[z], %[one], (%[zz])\n"                            \
    "amoadd.w %[o], %[one], (%[b])\n"                             \
    ".option pop\n"                                               \
    : [l0]"=&r"(l0), [o]"=&r"(o), [z]"=&r"(z)                     \
    : [b]"r"(B), [zz]"r"(Z), [one]"r"(1)                          \
    : "memory")

static __attribute__((noinline)) void run_amo_storm(kernel_arg_t* arg) {
  uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t nh = arg->num_harts;
  if (h >= nh)
    return;
  uintptr_t base = (uintptr_t)arg->buf_addr;
  uint32_t* out = (uint32_t*)arg->out_addr;
  uintptr_t Z = base + (uintptr_t)h * STORM_LINE;
  (void)*(volatile uint32_t*)Z;
  uint32_t seed = (arg->seed + h) * 747796405u + 2891336453u;
  uint32_t bad = 0, sink = 0;
  for (uint32_t r = 0; r < arg->iters; ++r) {
    seed = seed * 1664525u + 1013904223u;
    uint32_t k = (seed >> 28) & 3;
    uintptr_t B = base + (uintptr_t)(nh + r * nh + h) * STORM_LINE;
    uint32_t o, l0, z;
    if (k == 0) {
      STORM_SEQ("");
    } else if (k == 1) {
      STORM_SEQ("nop\n");
    } else if (k == 2) {
      STORM_SEQ("nop\nnop\n");
    } else {
      STORM_SEQ("nop\nnop\nnop\n");
    }
    sink += l0;
    bad += (o != 0) + (z != r);
  }
  out[h] = bad + (sink & 0);
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  switch (arg->mode) {
  case MODE_RAW:
  case MODE_WAR:
    run_probes(arg);
    break;
  case MODE_STORE_AMO:
  case MODE_STORE_LR:
    run_store_amo(arg);
    break;
  case MODE_AMO_STORE:
    run_amo_store(arg);
    break;
  case MODE_AMO_LOAD:
    run_amo_load(arg);
    break;
  default:
    run_amo_storm(arg);
    break;
  }
}
