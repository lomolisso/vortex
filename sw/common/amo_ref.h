// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// Golden model of the FPAMO atomics (inc/dec, float add, packed 16-bit
// min/max), of compare-and-swap, and of the 16-bit adds the kernel headers run
// as CAS loops. Shared by SimX, the testbenches and the test apps' host code;
// independent of VX_config.h. Header-only on softfloat:
//   CXXFLAGS += -I$(SW_COMMON_DIR) -I$(THIRD_PARTY_DIR)/softfloat/source/include
//   LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

#include <cstdint>

extern "C" {
#include "softfloat_ext.h"
#include <softfloat.h>
}

namespace vortex {
namespace amo_ref {

// amo_op codes; the custom-3 (0x7B) funct5 is the code minus OP_BASE.
enum Op : uint32_t {
  OP_CAS      = 9,
  OP_INC      = 16,  // u32: old >= rhs ? 0 : old + 1
  OP_DEC,            // u32: (old == 0 || old > rhs) ? rhs : old - 1
  OP_FADD,           // W f32, D f64
  OP_FADD_BF,        // reserved: no 16-bit atomic runs in hardware
  OP_FADD2,          // W f16x2, D f32x2
  OP_FADD2_BF,       // W bf16x2
  OP_FMIN2,          // W f16x2
  OP_FMAX2,
  OP_FMIN2_BF,       // W bf16x2
  OP_FMAX2_BF,
};
constexpr uint32_t OP_BASE = OP_INC;

// funct3 of the instruction; the access is 1 << width bytes.
enum Width : uint32_t { W_H = 1, W_W = 2, W_D = 3 };

// The NaN rule and f32 flush of the place an atomic runs.
//   SITE_LLC   the last-level cache: f32 inputs and results flushed to a
//              signed zero; an f64 NaN input returned unchanged, rhs first.
//   SITE_LOOP  the kernel headers' CAS loops: subnormals kept; an f64 NaN
//              input returned quiet, old first.
enum Site : uint32_t { SITE_LLC, SITE_LOOP };

// Canonical NaNs: any NaN result of the 16-bit and f32 forms, and the f64
// result of an invalid add (inf - inf).
constexpr uint16_t F16_QNAN  = 0x7fff;
constexpr uint16_t BF16_QNAN = 0x7fff;
constexpr uint32_t F32_QNAN  = 0x7fffffff;
constexpr uint64_t F64_QNAN  = 0xfff8000000000000ull;
constexpr uint64_t F64_QUIET = 0x0008000000000000ull;

inline bool is_fp(uint32_t op) {
  return op >= OP_FADD && op <= OP_FMAX2_BF;
}

// The decodable (op, width) pairs. CAS is amocas.w/.d (Zacas); amocas.h is
// not decoded.
inline bool legal(uint32_t op, uint32_t width) {
  switch (op) {
  case OP_CAS:
  case OP_FADD:
  case OP_FADD2:    return width == W_W || width == W_D;
  case OP_INC:
  case OP_DEC:
  case OP_FADD2_BF:
  case OP_FMIN2:
  case OP_FMAX2:
  case OP_FMIN2_BF:
  case OP_FMAX2_BF: return width == W_W;
  default:          return false;
  }
}

inline uint64_t width_mask(uint32_t width) {
  return (width >= W_D) ? ~0ull : ((1ull << (8u << width)) - 1);
}

namespace detail {

inline void round_rne() {
  softfloat_roundingMode   = softfloat_round_near_even;
  softfloat_exceptionFlags = 0;
}

inline bool f16_is_nan(uint16_t x)  { return (x & 0x7c00) == 0x7c00 && (x & 0x03ff) != 0; }
inline bool bf16_is_nan(uint16_t x) { return (x & 0x7f80) == 0x7f80 && (x & 0x007f) != 0; }
inline bool f32_is_nan(uint32_t x)  { return (x & 0x7f800000u) == 0x7f800000u && (x & 0x007fffffu) != 0; }
inline bool f64_is_nan(uint64_t x) {
  return (x & 0x7ff0000000000000ull) == 0x7ff0000000000000ull && (x & 0x000fffffffffffffull) != 0;
}

inline uint32_t f32_ftz(uint32_t x) {
  return ((x & 0x7f800000u) == 0) ? (x & 0x80000000u) : x;
}

inline uint16_t fadd_f16(uint16_t a, uint16_t b) {
  round_rne();
  uint16_t r = f16_add(float16_t{a}, float16_t{b}).v;
  return f16_is_nan(r) ? F16_QNAN : r;
}

// Exact in f32, then one rounding to bf16: p(f32) = 24 >= 2 * 8 + 2, so the
// two roundings give the correctly rounded sum.
inline uint16_t fadd_bf16(uint16_t a, uint16_t b) {
  round_rne();
  float32_t s = f32_add(bf16_to_f32(bfloat16_t{a}), bf16_to_f32(bfloat16_t{b}));
  if (f32_is_nan(s.v))
    return BF16_QNAN;
  round_rne();
  return f32_to_bf16(s).v;
}

inline uint32_t fadd_f32(uint32_t a, uint32_t b, bool ftz) {
  if (ftz) {
    a = f32_ftz(a);
    b = f32_ftz(b);
  }
  round_rne();
  uint32_t r = f32_add(float32_t{a}, float32_t{b}).v;
  if (f32_is_nan(r))
    return F32_QNAN;
  return ftz ? f32_ftz(r) : r;
}

inline uint64_t fadd_f64(uint64_t a, uint64_t b, Site site) {
  if (site == SITE_LLC) {
    if (f64_is_nan(b)) return b;
    if (f64_is_nan(a)) return a;
  } else {
    if (f64_is_nan(a)) return a | F64_QUIET;
    if (f64_is_nan(b)) return b | F64_QUIET;
  }
  round_rne();
  uint64_t r = f64_add(float64_t{a}, float64_t{b}).v;
  return f64_is_nan(r) ? F64_QNAN : r;
}

// Order key of a non-NaN 16-bit float (f16 or bf16): -0 below +0.
inline uint32_t key16(uint16_t x) {
  return (x & 0x8000) ? (uint32_t)(uint16_t)~x : (uint32_t)(x | 0x8000);
}

// IEEE 754-2019 minimumNumber/maximumNumber: a NaN operand yields the other
// one, two NaNs the canonical NaN.
inline uint16_t minmax16(uint16_t a, uint16_t b, bool is_max, bool bf, uint16_t qnan) {
  const bool na = bf ? bf16_is_nan(a) : f16_is_nan(a);
  const bool nb = bf ? bf16_is_nan(b) : f16_is_nan(b);
  if (na && nb) return qnan;
  if (na) return b;
  if (nb) return a;
  const bool a_lt_b = key16(a) < key16(b);
  return (a_lt_b != is_max) ? a : b;
}

template <typename F>
inline uint32_t lanes16(uint32_t a, uint32_t b, F f) {
  const uint16_t lo = f((uint16_t)a, (uint16_t)b);
  const uint16_t hi = f((uint16_t)(a >> 16), (uint16_t)(b >> 16));
  return ((uint32_t)hi << 16) | lo;
}

}  // namespace detail

// old/rhs right-aligned at the op width; result right-aligned and masked to it.
// OP_CAS needs a comparand: use cas() for it; apply() returns `old`.
inline uint64_t apply(uint32_t op, uint32_t width, uint64_t old, uint64_t rhs, Site site) {
  using namespace detail;
  const bool ftz_f32 = (site == SITE_LLC);
  const uint64_t m = width_mask(width);
  old &= m;
  rhs &= m;
  if (!legal(op, width))
    return old;
  const uint32_t a = (uint32_t)old, b = (uint32_t)rhs;
  uint64_t r = old;
  switch (op) {
  case OP_INC:
    r = (a >= b) ? 0 : a + 1;
    break;
  case OP_DEC:
    r = (a == 0 || a > b) ? b : a - 1;
    break;
  case OP_FADD:
    if (width == W_W)
      r = fadd_f32(a, b, ftz_f32);
    else
      r = fadd_f64(old, rhs, site);
    break;
  case OP_FADD2:
    if (width == W_W) {
      r = lanes16(a, b, [](uint16_t x, uint16_t y) { return fadd_f16(x, y); });
    } else {
      const uint64_t lo = fadd_f32((uint32_t)old, (uint32_t)rhs, ftz_f32);
      const uint64_t hi = fadd_f32((uint32_t)(old >> 32), (uint32_t)(rhs >> 32), ftz_f32);
      r = (hi << 32) | lo;
    }
    break;
  case OP_FADD2_BF:
    r = lanes16(a, b, [](uint16_t x, uint16_t y) { return fadd_bf16(x, y); });
    break;
  case OP_FMIN2:
    r = lanes16(a, b, [](uint16_t x, uint16_t y) { return minmax16(x, y, false, false, F16_QNAN); });
    break;
  case OP_FMAX2:
    r = lanes16(a, b, [](uint16_t x, uint16_t y) { return minmax16(x, y, true, false, F16_QNAN); });
    break;
  case OP_FMIN2_BF:
    r = lanes16(a, b, [](uint16_t x, uint16_t y) { return minmax16(x, y, false, true, BF16_QNAN); });
    break;
  case OP_FMAX2_BF:
    r = lanes16(a, b, [](uint16_t x, uint16_t y) { return minmax16(x, y, true, true, BF16_QNAN); });
    break;
  default:
    break;
  }
  return r & m;
}

// The scalar 16-bit adds of the kernel headers' CAS loops (no flush).
inline uint16_t add_f16(uint16_t a, uint16_t b)  { return detail::fadd_f16(a, b); }
inline uint16_t add_bf16(uint16_t a, uint16_t b) { return detail::fadd_bf16(a, b); }

// Compare-and-swap at `width` (W_H too, for the cas.b16 loop): the stored
// value; the old value is returned to rd.
inline uint64_t cas(uint32_t width, uint64_t old, uint64_t cmp, uint64_t val) {
  const uint64_t m = width_mask(width);
  return ((old & m) == (cmp & m)) ? (val & m) : (old & m);
}

// The rd value of a returning op: H and W sign-extended, D raw.
inline uint64_t ret_value(uint32_t width, uint64_t old) {
  switch (width) {
  case W_H: return (uint64_t)(int64_t)(int16_t)(uint16_t)old;
  case W_W: return (uint64_t)(int64_t)(int32_t)(uint32_t)old;
  default:  return old;
  }
}

}  // namespace amo_ref
}  // namespace vortex
