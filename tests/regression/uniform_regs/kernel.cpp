#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_uniform.h>
#include "common.h"
#include <utility>

#define US(x) VX_UASM_STR(x)

// URd = UR12 op UR13 for the uniform_ref code C, through UR2R into every active lane.
template <uint32_t C>
inline uint64_t ualu(uint64_t a, uint64_t b) {
  uintptr_t r;
  if constexpr (C < 8) {
    __asm__ volatile (VX_UASM_R2UR(12, "%1") VX_UASM_R2UR(13, "%2") ".insn r 0x0b, %3, " US(VX_FE_F7_UALU) ", x14, x12, x13\n\t"
                      VX_UASM_UR2R("%0", 14) : "=r"(r) : "r"((uintptr_t)a), "r"((uintptr_t)b), "i"(C));
  } else if constexpr (C < 11) {
    __asm__ volatile (VX_UASM_R2UR(12, "%1") VX_UASM_R2UR(13, "%2") ".insn r 0x0b, %3, " US(VX_FE_F7_UCMP) ", x14, x12, x13\n\t"
                      VX_UASM_UR2R("%0", 14) : "=r"(r) : "r"((uintptr_t)a), "r"((uintptr_t)b), "i"(C - 8));
  } else if constexpr (C == 11) {
    __asm__ volatile (VX_UASM_R2UR(12, "%1") VX_UASM_UR2R("%0", 12) : "=r"(r) : "r"((uintptr_t)a));
  } else {
    // URZ = UR63 ({rs2[0], field} = {1, 31}): a write to it is dropped, a read is 0.
    __asm__ volatile (".insn r 0x0b, " US(VX_FE_F3_R2UR) ", " US(VX_FE_F7_UMOV) ", x31, %1, x1\n\t"
                      ".insn r 0x0b, " US(VX_FE_F3_UR2R) ", " US(VX_FE_F7_UMOV) ", %0, x31, x1\n\t"
                      : "=r"(r) : "r"((uintptr_t)a));
  }
  return r;
}

#define UR_CODES(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12)

inline uint64_t run_alu(uint32_t code, uint64_t a, uint64_t b) {
  switch (code) {
#define UR_CASE(c) case c: return ualu<c>(a, b);
  UR_CODES(UR_CASE)
#undef UR_CASE
  default: return 0;
  }
}

// S2UR of selector S next to a CSR read of the same register: {s2ur, csr} at out[2S], out[2S+1].
template <uint32_t S>
inline void s2ur_pair(uint64_t* out) {
  constexpr uint32_t csr = vortex::fe::S2UR_CSR[S];
#ifndef VX_CFG_EXT_DLMEM_ENABLE
  // The cluster rank/dims and the shared-memory size CSRs exist only with DLMEM.
  if constexpr (csr >= VX_CSR_CTA_CLUSTER_RANK && csr <= VX_CSR_CTA_LMEM_SIZE) {
    out[2 * S] = 0;
    out[2 * S + 1] = 0;
    return;
  }
#endif
  uintptr_t s, c;
  __asm__ volatile (".insn r 0x0b, " US(VX_FE_F3_S2UR) ", " US(VX_FE_F7_UMOV) ", x12, x%1, x0\n\t"
                    VX_UASM_UR2R("%0", 12) : "=r"(s) : "i"(S));
  __asm__ volatile ("csrr %0, %1" : "=r"(c) : "i"(csr));
  out[2 * S] = s;
  out[2 * S + 1] = c;
}

template <uint32_t... I>
inline void s2ur_pairs(uint64_t* out, std::integer_sequence<uint32_t, I...>) {
  (s2ur_pair<I + 2>(out), ...);
}

// Selectors 0/1 are the clock: an S2UR between two cycle reads lies between them
// (out[0] <= out[1] <= out[2] low word; XLEN 32 also the high word: out[3] S2UR, out[54] CSR).
inline void s2ur_all(uint64_t* out) {
  out[0] = csr_read(VX_CSR_MCYCLE);
  out[1] = vx_us2r(0);
  out[2] = csr_read(VX_CSR_MCYCLE);
#if (VX_CFG_XLEN == 32)
  out[3] = vx_us2r(1);
  out[54] = csr_read(VX_CSR_MCYCLE_H);
#else
  out[3] = 0;
  out[54] = 0;
#endif
  s2ur_pairs(out, std::make_integer_sequence<uint32_t, vortex::fe::S2UR_NUM_SELECTORS - 2>{});
}

// Operand mode: each entry {GPR form, UR form} of one instruction; the UR form reads the
// slot named in the UPFX from UR15 (= u of the first active lane, R2UR in the same block).
#define OPD_RR(k, insn, slot, a0, a1)                                                         \
  do {                                                                                        \
    uintptr_t g, p;                                                                           \
    __asm__ volatile (insn " %0, %1, %2\n\t" : "=r"(g) : "r"(slot == 0 ? u : x),             \
                      "r"(slot == 0 ? x : u));                                                \
    if (slot == 0)                                                                            \
      __asm__ volatile (VX_UASM_R2UR(15, "%2") VX_UASM_UPFX(0, 15) insn " %0, x0, %1\n\t"     \
                        : "=r"(p) : "r"(x), "r"(u));                                          \
    else                                                                                      \
      __asm__ volatile (VX_UASM_R2UR(15, "%2") VX_UASM_UPFX(1, 15) insn " %0, %1, x0\n\t"     \
                        : "=r"(p) : "r"(x), "r"(u));                                          \
    o[2 * (k)] = g; o[2 * (k) + 1] = p; (void)(a0); (void)(a1);                               \
  } while (0)

#define OPD_RI(k, insn, imm)                                                                  \
  do {                                                                                        \
    uintptr_t g, p;                                                                           \
    __asm__ volatile (insn " %0, %1, " #imm "\n\t" : "=r"(g) : "r"(u));                       \
    __asm__ volatile (VX_UASM_R2UR(15, "%1") VX_UASM_UPFX(0, 15) insn " %0, x0, " #imm "\n\t" \
                      : "=r"(p) : "r"(u));                                                    \
    o[2 * (k)] = g; o[2 * (k) + 1] = p;                                                       \
  } while (0)

// F32 sources from a UR use its raw bits: the UR holds the NaN-boxed value.
#if (VX_CFG_XLEN == 64)
#define FBOX(v) (0xffffffff00000000ull | (v))
#else
#define FBOX(v) (v)
#endif

#define OPD_FF(k, insn, slot)                                                                 \
  do {                                                                                        \
    float g, p;                                                                               \
    if (slot == 0) {                                                                          \
      __asm__ volatile (insn " %0, %1, %2\n\t" : "=f"(g) : "f"(fu), "f"(fx));                 \
      __asm__ volatile (VX_UASM_R2UR(15, "%2") VX_UASM_UPFX(0, 15) insn " %0, f0, %1\n\t"     \
                        : "=f"(p) : "f"(fx), "r"((uintptr_t)FBOX(ub)));                       \
    } else {                                                                                  \
      __asm__ volatile (insn " %0, %1, %2\n\t" : "=f"(g) : "f"(fx), "f"(fu));                 \
      __asm__ volatile (VX_UASM_R2UR(15, "%2") VX_UASM_UPFX(1, 15) insn " %0, %1, f0\n\t"     \
                        : "=f"(p) : "f"(fx), "r"((uintptr_t)FBOX(ub)));                       \
    }                                                                                         \
    o[2 * (k)] = fbits(g); o[2 * (k) + 1] = fbits(p);                                         \
  } while (0)

inline uint64_t fbits(float f) { uint32_t b; __builtin_memcpy(&b, &f, 4); return b; }
inline uint64_t dbits(double d) { uint64_t b; __builtin_memcpy(&b, &d, 8); return b; }

inline void operand_checks(uint64_t x, uint64_t u, uint64_t* o) {
  OPD_RR(0, "add", 1, 0, 0);   OPD_RR(1, "add", 0, 0, 0);   OPD_RR(2, "sub", 1, 0, 0);
  OPD_RR(3, "sub", 0, 0, 0);   OPD_RR(4, "and", 1, 0, 0);   OPD_RR(5, "or", 0, 0, 0);
  OPD_RR(6, "xor", 1, 0, 0);   OPD_RR(7, "sll", 1, 0, 0);   OPD_RR(8, "srl", 0, 0, 0);
  OPD_RR(9, "sra", 1, 0, 0);   OPD_RR(10, "slt", 0, 0, 0);  OPD_RR(11, "sltu", 1, 0, 0);
  OPD_RR(12, "mul", 1, 0, 0);  OPD_RR(13, "mulh", 0, 0, 0); OPD_RR(14, "mulhu", 1, 0, 0);
  OPD_RR(15, "div", 1, 0, 0);  OPD_RR(16, "divu", 0, 0, 0); OPD_RR(17, "rem", 1, 0, 0);
  OPD_RR(18, "remu", 0, 0, 0);
  OPD_RI(19, "addi", 123);     OPD_RI(20, "xori", -7);      OPD_RI(21, "slli", 3);
  OPD_RI(22, "srai", 2);       OPD_RI(23, "sltiu", 99);
  const uint32_t xb = (uint32_t)(x & 0x7fffff) | 0x3f800000u, ub = (uint32_t)(u & 0x7fffff) | 0x40000000u;
  float fx, fu;
  __builtin_memcpy(&fx, &xb, 4);
  __builtin_memcpy(&fu, &ub, 4);
  OPD_FF(24, "fadd.s", 1);     OPD_FF(25, "fadd.s", 0);     OPD_FF(26, "fmul.s", 1);
  OPD_FF(27, "fmin.s", 0);     OPD_FF(28, "fsgnjn.s", 1);   OPD_FF(29, "fsub.s", 0);
  {
    float g, p;
    __asm__ volatile ("fmadd.s %0, %1, %2, %3\n\t" : "=f"(g) : "f"(fx), "f"(fx), "f"(fu));
    __asm__ volatile (VX_UASM_R2UR(15, "%2") VX_UASM_UPFX(2, 15) "fmadd.s %0, %1, %1, f0\n\t"
                      : "=f"(p) : "f"(fx), "r"((uintptr_t)FBOX(ub)));
    o[60] = fbits(g); o[61] = fbits(p);
  }
  {
    uintptr_t g, p;
    __asm__ volatile ("flt.s %0, %1, %2\n\t" : "=r"(g) : "f"(fx), "f"(fu));
    __asm__ volatile (VX_UASM_R2UR(15, "%2") VX_UASM_UPFX(1, 15) "flt.s %0, %1, f0\n\t"
                      : "=r"(p) : "f"(fx), "r"((uintptr_t)FBOX(ub)));
    o[62] = g; o[63] = p;
  }
  {
    float g, p;
    const intptr_t iu = (intptr_t)(int32_t)u;
    __asm__ volatile ("fcvt.s.w %0, %1\n\t" : "=f"(g) : "r"(iu));
    __asm__ volatile (VX_UASM_R2UR(15, "%1") VX_UASM_UPFX(0, 15) "fcvt.s.w %0, x0\n\t"
                      : "=f"(p) : "r"(iu));
    o[64] = fbits(g); o[65] = fbits(p);
  }
#if (VX_CFG_XLEN == 64)
  {
    double dx = 1.0 + (double)(x & 0xffff), du = 3.0 + (double)(u & 0xffff), g, p;
    __asm__ volatile ("fadd.d %0, %1, %2\n\t" : "=f"(g) : "f"(dx), "f"(du));
    __asm__ volatile (VX_UASM_R2UR(15, "%2") VX_UASM_UPFX(1, 15) "fadd.d %0, %1, f0\n\t"
                      : "=f"(p) : "f"(dx), "r"(dbits(du)));
    o[66] = dbits(g); o[67] = dbits(p);
    __asm__ volatile ("fmul.d %0, %1, %2\n\t" : "=f"(g) : "f"(du), "f"(dx));
    __asm__ volatile (VX_UASM_R2UR(15, "%2") VX_UASM_UPFX(0, 15) "fmul.d %0, f0, %1\n\t"
                      : "=f"(p) : "f"(dx), "r"(dbits(du)));
    o[68] = dbits(g); o[69] = dbits(p);
  }
#endif
}

// instret of 8 prefixed adds against 8 plain adds (each block also has one R2UR / mv).
inline void instret_check(uint64_t u, uint64_t* c) {
  // The counter reads sit in the same asm blocks, so the compiler adds nothing between them.
  uintptr_t r = 0, t, i0, i1, i2, i3;
  __asm__ volatile ("csrr %1, minstret\n\t"
                    VX_UASM_R2UR(15, "%3")
                    VX_UASM_UPFX(1, 15) "add %0, %0, x0\n\t" VX_UASM_UPFX(1, 15) "add %0, %0, x0\n\t"
                    VX_UASM_UPFX(1, 15) "add %0, %0, x0\n\t" VX_UASM_UPFX(1, 15) "add %0, %0, x0\n\t"
                    VX_UASM_UPFX(1, 15) "add %0, %0, x0\n\t" VX_UASM_UPFX(1, 15) "add %0, %0, x0\n\t"
                    VX_UASM_UPFX(1, 15) "add %0, %0, x0\n\t" VX_UASM_UPFX(1, 15) "add %0, %0, x0\n\t"
                    "csrr %2, minstret\n\t"
                    : "+r"(r), "=&r"(i0), "=&r"(i1) : "r"((uintptr_t)u));
  __asm__ volatile ("csrr %2, minstret\n\t"
                    "mv %1, %4\n\t"
                    "add %0, %0, %1\n\t" "add %0, %0, %1\n\t" "add %0, %0, %1\n\t" "add %0, %0, %1\n\t"
                    "add %0, %0, %1\n\t" "add %0, %0, %1\n\t" "add %0, %0, %1\n\t" "add %0, %0, %1\n\t"
                    "csrr %3, minstret\n\t"
                    : "+r"(r), "=&r"(t), "=&r"(i2), "=&r"(i3) : "r"((uintptr_t)u));
  c[0] = i1 - i0;
  c[1] = i3 - i2;
  c[2] = r;
}

#define UALU_DEP4 VX_UASM_UALU(0, 12, 12, 13) VX_UASM_UALU(0, 12, 12, 13) \
                  VX_UASM_UALU(0, 12, 12, 13) VX_UASM_UALU(0, 12, 12, 13)
// S2UR (warp id) and a uniform add that depends on it.
#define S2UR_DEP VX_UASM_S2UR(14, 4) VX_UASM_UALU(0, 12, 12, 14)
#define S2UR_DEP4 S2UR_DEP S2UR_DEP S2UR_DEP S2UR_DEP
// The vector ALU's and the CSR unit's counterparts of the two chains.
#define ADD_DEP4 "add %0, %0, %2\n\t" "add %0, %0, %2\n\t" "add %0, %0, %2\n\t" "add %0, %0, %2\n\t"
#define CSR_DEP "csrr %1, 0xCC1\n\t" "add %0, %0, %1\n\t"
#define CSR_DEP4 CSR_DEP CSR_DEP CSR_DEP CSR_DEP
// Independent S2URs of one warp.
#define S2UR_IND4 VX_UASM_S2UR(12, 4) VX_UASM_S2UR(13, 4) VX_UASM_S2UR(14, 4) VX_UASM_S2UR(15, 4)

#define UALU_DEP16 UALU_DEP4 UALU_DEP4 UALU_DEP4 UALU_DEP4
#define S2UR_DEP16 S2UR_DEP4 S2UR_DEP4 S2UR_DEP4 S2UR_DEP4
#define ADD_DEP16 ADD_DEP4 ADD_DEP4 ADD_DEP4 ADD_DEP4
#define CSR_DEP16 CSR_DEP4 CSR_DEP4 CSR_DEP4 CSR_DEP4

// Each chain runs on across loop passes with nothing else in its path, as its
// vector counterpart does. Every dependent chain is timed with a 16-op and a 32-op
// body: the difference is the chain's own step, free of the per-pass loop glue.
inline void latency_runs(uint32_t iters, uint64_t* rec) {
  uintptr_t r, r1;
  uintptr_t a = 0, c;
  __asm__ volatile (VX_UASM_R2UR(12, "%0") VX_UASM_R2UR(13, "%0") :: "r"((uintptr_t)iters));
  uint64_t t[10];
  t[0] = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < iters; ++i)
    __asm__ volatile (UALU_DEP16);
  t[1] = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < iters; ++i)
    __asm__ volatile (S2UR_DEP16);
  t[2] = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < iters; ++i)
    __asm__ volatile (S2UR_IND4 S2UR_IND4 S2UR_IND4 S2UR_IND4 VX_UASM_UR2R("%0", 15) : "=r"(r));
  t[3] = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < iters; ++i)
    __asm__ volatile (ADD_DEP16 : "+r"(a), "=&r"(c) : "r"((uintptr_t)i));
  t[4] = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < iters; ++i)
    __asm__ volatile (CSR_DEP16 : "+r"(a), "=&r"(c) : "r"((uintptr_t)i));
  t[5] = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < iters; ++i)
    __asm__ volatile (UALU_DEP16 UALU_DEP16);
  t[6] = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < iters; ++i)
    __asm__ volatile (S2UR_DEP16 S2UR_DEP16);
  t[7] = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < iters; ++i)
    __asm__ volatile (ADD_DEP16 ADD_DEP16 : "+r"(a), "=&r"(c) : "r"((uintptr_t)i));
  t[8] = csr_read(VX_CSR_MCYCLE);
  for (uint32_t i = 0; i < iters; ++i)
    __asm__ volatile (CSR_DEP16 CSR_DEP16 : "+r"(a), "=&r"(c) : "r"((uintptr_t)i));
  t[9] = csr_read(VX_CSR_MCYCLE);
  __asm__ volatile (VX_UASM_UR2R("%0", 12) : "=r"(r1));
  rec[0] = t[1] - t[0];
  rec[1] = t[2] - t[1];
  rec[2] = t[3] - t[2];
  rec[3] = r + r1 + a;
  rec[4] = t[4] - t[3];
  rec[5] = t[5] - t[4];
  rec[6] = t[6] - t[5];
  rec[7] = t[7] - t[6];
  rec[8] = t[8] - t[7];
  rec[9] = t[9] - t[8];
}

__kernel void ur_main(ur_arg_t* __UNIFORM__ arg) {
  const uint32_t lane = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t warp = threadIdx.x / VX_CFG_NUM_THREADS;
  const uint32_t nwarps = blockDim.x / VX_CFG_NUM_THREADS;
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);

  if (arg->mode == MODE_ALU) {
    auto cases = reinterpret_cast<const ur_case_t*>(arg->cases_addr);
    for (uint32_t c = warp; c < arg->ncases; c += nwarps) {
      const ur_case_t& t = cases[c];
      if ((t.tmask >> lane) & 1) {
        uint64_t* o = out + (uint64_t)c * VX_CFG_NUM_THREADS * S2UR_WORDS;
        if (t.code == CASE_S2UR)
          s2ur_all(o + lane * S2UR_WORDS);
        else
          o[lane] = run_alu(t.code, t.a[lane], t.b[lane]);
      }
    }
    return;
  }

  if (arg->mode == MODE_OPERAND) {
    if (warp != 0)
      return;
    const uint64_t x = 0x9e3779b97f4a7c15ull * (lane + 1) >> (lane & 7);
    operand_checks(x, arg->u, out + lane * OPD_MAX * 2);
    if (lane == 0)
      instret_check(arg->u, out + VX_CFG_NUM_THREADS * OPD_MAX * 2);
    return;
  }

  // Latency: warps [0, iters-warps) run the chains; the host launches 1 or 2 warps.
  latency_runs(arg->iters, out + warp * LAT_RECS);
}
