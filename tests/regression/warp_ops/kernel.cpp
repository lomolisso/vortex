#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_warp_ops.h>
#include <warp_ops_ref.h>
#include "common.h"

namespace wop = vortex::warp_ops_ref;

// One op on this lane's operands; results widened as the golden writes them.
template <uint32_t C>
inline uint64_t op(uint64_t rs1, uint64_t rs2) {
  (void)rs2;
  const uint32_t lo = (uint32_t)rs1;
#if (VX_CFG_XLEN == 64)
  const uint64_t v64 = rs1;
#else
  const uint64_t v64 = (rs2 << 32) | lo;
#endif
  if constexpr (C == wop::REDUX_ADD)     return (uint64_t)(int64_t)(int32_t)vx_redux_add(lo);
  if constexpr (C == wop::REDUX_MIN_S)   return (uint64_t)(int64_t)vx_redux_min_s((int32_t)lo);
  if constexpr (C == wop::REDUX_MAX_S)   return (uint64_t)(int64_t)vx_redux_max_s((int32_t)lo);
  if constexpr (C == wop::REDUX_MIN_U)   return (uint64_t)(int64_t)(int32_t)vx_redux_min_u(lo);
  if constexpr (C == wop::REDUX_MAX_U)   return (uint64_t)(int64_t)(int32_t)vx_redux_max_u(lo);
  if constexpr (C == wop::REDUX_AND)     return (uint64_t)(int64_t)(int32_t)vx_redux_and(lo);
  if constexpr (C == wop::REDUX_OR)      return (uint64_t)(int64_t)(int32_t)vx_redux_or(lo);
  if constexpr (C == wop::REDUX_XOR)     return (uint64_t)(int64_t)(int32_t)vx_redux_xor(lo);
  if constexpr (C == wop::ELECT_P)       return (uint64_t)vx_elect_p(lo);
  if constexpr (C == wop::ELECT_L)       return (uint64_t)vx_elect_lane(lo);
  if constexpr (C == wop::MATCH_ANY_B32) return vx_match_any_b32(lo);
  if constexpr (C == wop::MATCH_ANY_B64) return vx_match_any_b64(v64);
  if constexpr (C == wop::MATCH_ALL_B32) return vx_match_all_b32(lo);
  if constexpr (C == wop::MATCH_ALL_B64) return vx_match_all_b64(v64);
  return 0;
}

#define WOP_CODES(X) X(0x00) X(0x01) X(0x02) X(0x03) X(0x04) X(0x05) X(0x06) X(0x07) \
                     X(0x10) X(0x11) X(0x18) X(0x19) X(0x1A) X(0x1B)

inline uint64_t run_op(uint32_t code, uint64_t rs1, uint64_t rs2) {
  switch (code) {
#define WOP_CASE(c) case c: return op<c>(rs1, rs2);
  WOP_CODES(WOP_CASE)
#undef WOP_CASE
  default: return 0;
  }
}

// A dependent chain: each op's input depends on the previous result through a
// compare the compiler cannot fold, so a step is the op's latency plus that glue.
// Every op's result is taken as its low 32 bits, so the glue is the same for all.
// WOP_CHAIN_UNROLL steps per loop pass, so the loop's own branch and fetch, which
// do not depend on r, stay off the critical path even for a one-cycle unit.
// Pass 0 (one loop pass) warms the I-cache: a cold fetch would hide the first op's latency.
#define WOP_STEP r = (uint32_t)op<C>(x, 0); x = x0 + (r == 0x5eed5eedu);
template <uint32_t C>
inline void chain(uint64_t x0, uint32_t iters, uint64_t* rec) {
  static_assert(WOP_CHAIN_UNROLL == 8, "the loop body below holds 8 steps");
  uint64_t x = x0, t0 = 0, t1 = 0;
  uint32_t r = 0;
#pragma clang loop unroll(disable)
  for (uint32_t pass = 0; pass < 2; ++pass) {
    const uint32_t n = pass ? iters : WOP_CHAIN_UNROLL;
    t0 = vx_rdcycle();
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < n; i += WOP_CHAIN_UNROLL) {
      WOP_STEP WOP_STEP WOP_STEP WOP_STEP WOP_STEP WOP_STEP WOP_STEP WOP_STEP
    }
    t1 = vx_rdcycle();
  }
  rec[0] = t0;
  rec[1] = t1;
  rec[2] = r;
}

// WOP_ILP independent ops in one asm block, so no glue sits between them; a
// GPR-result op writes its own register. REDUX writes its own UR (UR16-UR23, above
// the headers' UR0-UR15) and the timed loop holds no copies out: with one fetch per
// core per cycle and one fetch in flight per warp, eight UR2Rs per burst made the
// fetch, not the warp-op unit, the limit (the H100 issues its MOVs from four SMSPs).
#define WOP_G(n) ".insn r %8, %9, %10, %" #n ", %11, x0\n\t"
#define WOP_R(u) ".insn r %8, %9, %10, x" #u ", %11, x0\n\t"
#define WOP_M(n, u) ".insn r %8, %12, %13, %" #n ", x" #u ", x0\n\t"
#define WOP_OUTS(r) "=&r"(r[0]), "=&r"(r[1]), "=&r"(r[2]), "=&r"(r[3]), \
                    "=&r"(r[4]), "=&r"(r[5]), "=&r"(r[6]), "=&r"(r[7])
static_assert(WOP_ILP == 8, "the burst below issues 8 ops");

template <uint32_t C>
inline void burst(unsigned long v, unsigned long* r) {
  constexpr uint32_t f3 = C & 7;
  if constexpr (C == wop::REDUX_ADD) {
    (void)r;
    __asm__ volatile (WOP_R(16) WOP_R(17) WOP_R(18) WOP_R(19) WOP_R(20) WOP_R(21) WOP_R(22) WOP_R(23)
                      :: "i"(0), "i"(0), "i"(0), "i"(0), "i"(0), "i"(0), "i"(0), "i"(0),
                         "i"(RISCV_CUSTOM0), "i"(f3), "i"(VX_FE_F7_REDUX), "r"(v));
  } else {
    static_assert(C == wop::MATCH_ALL_B32 || C == wop::MATCH_ANY_B32, "rate forms only");
    __asm__ volatile (WOP_G(0) WOP_G(1) WOP_G(2) WOP_G(3) WOP_G(4) WOP_G(5) WOP_G(6) WOP_G(7)
                      : WOP_OUTS(r)
                      : "i"(RISCV_CUSTOM0), "i"(f3), "i"(VX_FE_F7_MATCH), "r"(v));
  }
}

// Every warp of the block starts the timed pass together (barrier after a
// warm-up pass); the window closes on the last results.
template <uint32_t C>
inline void stream(uint64_t x0, uint32_t iters, uint64_t* rec) {
  unsigned long r[WOP_ILP] = {};
  uint64_t t0 = 0;
#pragma clang loop unroll(disable)
  for (uint32_t pass = 0; pass < 2; ++pass) {
    const uint32_t n = pass ? iters : 1;
    __syncthreads();
    t0 = vx_rdcycle();
    for (uint32_t i = 0; i < n; ++i)
      burst<C>((unsigned long)x0, r);
  }
  if constexpr (C == wop::REDUX_ADD) {
    __asm__ volatile (WOP_M(0, 16) WOP_M(1, 17) WOP_M(2, 18) WOP_M(3, 19)
                      WOP_M(4, 20) WOP_M(5, 21) WOP_M(6, 22) WOP_M(7, 23)
                      : WOP_OUTS(r)
                      : "i"(RISCV_CUSTOM0), "i"(0), "i"(0), "i"(0),
                        "i"(VX_FE_F3_UR2R), "i"(VX_FE_F7_UMOV));
  }
  unsigned long s = 0;
  for (int j = 0; j < WOP_ILP; ++j)
    s ^= r[j];
  __asm__ volatile ("" :: "r"(s));
  const uint64_t t1 = vx_rdcycle();
  rec[0] = t0;
  rec[1] = t1;
  rec[2] = s;
}

#define WOP_RATE_CODES(X) X(0x00) X(0x18) X(0x1A)

__kernel void wop_main(wop_arg_t* __UNIFORM__ arg) {
  const uint32_t lane = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t warp = threadIdx.x / VX_CFG_NUM_THREADS;
  const uint32_t nwarps = blockDim.x / VX_CFG_NUM_THREADS;
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);

  if (arg->mode == MODE_BITEXACT) {
    auto cases = reinterpret_cast<const wop_case_t*>(arg->cases_addr);
    for (uint32_t c = warp; c < arg->ncases; c += nwarps) {
      const wop_case_t& t = cases[c];
      if ((t.tmask >> lane) & 1)
        out[c * VX_CFG_NUM_THREADS + lane] = run_op(t.code, t.rs1[lane], t.rs2[lane]);
    }
    return;
  }

  // MATCH.ANY sees k distinct values (lane % k); the other ops a lane-varying value.
  const uint32_t k = arg->k ? arg->k : 1;
  const uint64_t x0 = (uint64_t)(lane % k) * 0x01000193u + 1;
  uint64_t* rec = out + warp * 4;
  if (arg->mode == MODE_LATENCY) {
    if (warp != 0)
      return;
    switch (arg->code) {
#define WOP_CASE(c) case c: chain<c>((c == wop::ELECT_P || c == wop::ELECT_L) ? ~0ull : x0, arg->iters, rec); break;
    WOP_CODES(WOP_CASE)
#undef WOP_CASE
    default: break;
    }
    return;
  }
  switch (arg->code) {
#define WOP_CASE(c) case c: stream<c>(x0, arg->iters, rec); break;
  WOP_RATE_CODES(WOP_CASE)
#undef WOP_CASE
  default: break;
  }
}
