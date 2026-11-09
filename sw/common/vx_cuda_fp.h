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

// CUDA's 16-bit and 8-bit floating-point types (cuda_fp16.h, cuda_bf16.h,
// cuda_fp8.h names) as software types over integer and f32/f64 code, for host
// code and Vortex kernels alike. Results are bit-exact and identical on every
// target: each narrow result is rounded once, to nearest even, from the exact
// value, and every f16/bf16 NaN result is 0x7fff.

#pragma once

#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__riscv)
#define __VX_CUDA_FP_HD inline __attribute__((always_inline))
#else
#define __VX_CUDA_FP_HD inline
#endif

// Compilers may contract a*b+c into an fma (clang by default, GCC with
// -ffp-contract=fast). Every f64 sequence below is written so that this cannot
// change a result: each product that feeds an add is exact, and every other
// multiply-add is an explicit fma.

// CUDA's built-in vector types, with CUDA's alignments (n = 1 and 3: the
// element's; n = 2: twice it; n = 4: four times it, at most 16). A build that
// already has them defines VX_CUDA_FP_HAVE_VECTOR_TYPES.
#ifndef VX_CUDA_FP_HAVE_VECTOR_TYPES
#define __VX_CUDA_FP_VEC(N, T)                                                                                         \
  struct alignas(sizeof(T)) N##1 { T x; };                                                                             \
  struct alignas(2 * sizeof(T)) N##2 { T x, y; };                                                                      \
  struct alignas(sizeof(T)) N##3 { T x, y, z; };                                                                       \
  struct alignas(4 * sizeof(T) > 16 ? 16 : 4 * sizeof(T)) N##4 { T x, y, z, w; };                                     \
  __VX_CUDA_FP_HD N##1 make_##N##1(T x) { N##1 r; r.x = x; return r; }                                                 \
  __VX_CUDA_FP_HD N##2 make_##N##2(T x, T y) { N##2 r; r.x = x; r.y = y; return r; }                                   \
  __VX_CUDA_FP_HD N##3 make_##N##3(T x, T y, T z) { N##3 r; r.x = x; r.y = y; r.z = z; return r; }                     \
  __VX_CUDA_FP_HD N##4 make_##N##4(T x, T y, T z, T w) { N##4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }
__VX_CUDA_FP_VEC(char, signed char)
__VX_CUDA_FP_VEC(uchar, unsigned char)
__VX_CUDA_FP_VEC(short, short)
__VX_CUDA_FP_VEC(ushort, unsigned short)
__VX_CUDA_FP_VEC(int, int)
__VX_CUDA_FP_VEC(uint, unsigned int)
__VX_CUDA_FP_VEC(long, long)
__VX_CUDA_FP_VEC(ulong, unsigned long)
__VX_CUDA_FP_VEC(longlong, long long)
__VX_CUDA_FP_VEC(ulonglong, unsigned long long)
__VX_CUDA_FP_VEC(float, float)
__VX_CUDA_FP_VEC(double, double)
#undef __VX_CUDA_FP_VEC
#endif

namespace vx_cuda_fp {

__VX_CUDA_FP_HD uint32_t f2u(float f) { uint32_t u; __builtin_memcpy(&u, &f, 4); return u; }
__VX_CUDA_FP_HD float u2f(uint32_t u) { float f; __builtin_memcpy(&f, &u, 4); return f; }
__VX_CUDA_FP_HD uint64_t d2u(double d) { uint64_t u; __builtin_memcpy(&u, &d, 8); return u; }
__VX_CUDA_FP_HD double u2d(uint64_t u) { double d; __builtin_memcpy(&d, &u, 8); return d; }

// f32 NaN results: x86 and RISC-V produce different default NaNs, so every
// f32-returning intrinsic returns this one.
constexpr uint32_t F32_NAN = 0x7fffffff;

__VX_CUDA_FP_HD float canon_f32(float r) { return (r != r) ? u2f(F32_NAN) : r; }

struct fmt_f16  { static constexpr int MB = 10, BIAS = 15;  static constexpr uint32_t SIGN = 0x8000, MAXFIN = 0x7bff, INF = 0x7c00, NAN_ = 0x7fff; };
struct fmt_bf16 { static constexpr int MB = 7,  BIAS = 127; static constexpr uint32_t SIGN = 0x8000, MAXFIN = 0x7f7f, INF = 0x7f80, NAN_ = 0x7fff; };
// e4m3 has no infinity: an overflow without saturation is NaN.
struct fmt_e4m3 { static constexpr int MB = 3,  BIAS = 7;   static constexpr uint32_t SIGN = 0x80,   MAXFIN = 0x7e,   INF = 0x7f,   NAN_ = 0x7f; };

// Magnitude code of m * 2^e (m != 0) rounded to nearest even in format F.
template <class F>
__VX_CUDA_FP_HD uint32_t round_mag(uint64_t m, int e, bool sat) {
  const int emin = 1 - F::BIAS;
  const int eu = e + 63 - __builtin_clzll(m);
  if (eu - emin > (int)(F::MAXFIN >> F::MB) + 1)
    return sat ? F::MAXFIN : F::INF;
  const int qe = (eu > emin ? eu : emin) - F::MB;
  const int sh = qe - e;
  uint64_t n;
  if (sh <= 0) {
    n = m << -sh;
  } else if (sh > 64) {
    n = 0;
  } else if (sh == 64) {
    n = (m > (1ull << 63)) ? 1 : 0;
  } else {
    n = m >> sh;
    const uint64_t rem = m & ((1ull << sh) - 1), half = 1ull << (sh - 1);
    if (rem > half || (rem == half && (n & 1)))
      ++n;
  }
  const uint64_t code = ((uint64_t)(eu > emin ? eu - emin : 0) << F::MB) + n;
  if (code > F::MAXFIN)
    return sat ? F::MAXFIN : F::INF;
  return (uint32_t)code;
}

template <class F>
__VX_CUDA_FP_HD uint32_t from_f32(float f, bool sat = false) {
  const uint32_t u = f2u(f);
  const uint32_t s = (u >> 31) ? F::SIGN : 0, a = u & 0x7fffffff;
  if (a > 0x7f800000) return F::NAN_;
  if (a == 0x7f800000) return s | (sat ? F::MAXFIN : F::INF);
  if (a == 0) return s;
  const uint32_t ex = a >> 23, fr = a & 0x7fffff;
  const uint64_t m = ex ? (fr | 0x800000) : fr;
  return s | round_mag<F>(m, (ex ? (int)ex : 1) - 150, sat);
}

template <class F>
__VX_CUDA_FP_HD uint32_t from_f64(double d, bool sat = false) {
  const uint64_t u = d2u(d);
  const uint32_t s = (u >> 63) ? F::SIGN : 0;
  const uint64_t a = u & 0x7fffffffffffffffull;
  if (a > 0x7ff0000000000000ull) return F::NAN_;
  if (a == 0x7ff0000000000000ull) return s | (sat ? F::MAXFIN : F::INF);
  if (a == 0) return s;
  const int ex = (int)(a >> 52);
  const uint64_t fr = a & 0xfffffffffffffull;
  const uint64_t m = ex ? (fr | (1ull << 52)) : fr;
  return s | round_mag<F>(m, (ex ? ex : 1) - 1075, sat);
}

template <class F>
__VX_CUDA_FP_HD uint32_t from_u64(uint64_t v, bool neg, bool sat = false) {
  if (v == 0) return 0;
  return (neg ? F::SIGN : 0) | round_mag<F>(v, 0, sat);
}

template <class F>
__VX_CUDA_FP_HD uint32_t from_i64(int64_t v, bool sat = false) {
  return from_u64<F>(v < 0 ? 0 - (uint64_t)v : (uint64_t)v, v < 0, sat);
}

__VX_CUDA_FP_HD float f16_to_f32(uint16_t h) {
  const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e == 31) return u2f(s | 0x7f800000 | (m ? (0x400000 | (m << 13)) : 0));
  if (e == 0) {
    if (m == 0) return u2f(s);
    const int p = 31 - __builtin_clz(m);
    return u2f(s | ((uint32_t)(p + 103) << 23) | ((m << (23 - p)) & 0x7fffff));
  }
  return u2f(s | ((e + 112) << 23) | (m << 13));
}

__VX_CUDA_FP_HD float bf16_to_f32(uint16_t h) {
  const uint32_t u = (uint32_t)h << 16;
  return u2f(((u & 0x7fffffff) > 0x7f800000) ? (u | 0x400000) : u);
}

__VX_CUDA_FP_HD uint16_t e4m3_to_f16(uint8_t b) {
  if ((b & 0x7f) == 0x7f) return 0x7fff;
  const uint16_t s = (uint16_t)((b & 0x80) << 8);
  const uint32_t e = (b >> 3) & 0xf, m = b & 7;
  if (e == 0) {
    if (m == 0) return s;
    const int p = 31 - __builtin_clz(m);
    return (uint16_t)(s | ((uint32_t)(p + 6) << 10) | ((m << (10 - p)) & 0x3ff));
  }
  return (uint16_t)(s | ((e + 8) << 10) | (m << 7));
}

// clang emits a libm call for __builtin_fma while errno is on; with the D
// extension the instruction is used directly (dynamic rounding = nearest even).
__VX_CUDA_FP_HD double fma_d(double a, double b, double c) {
#if defined(__riscv) && defined(__riscv_flen) && __riscv_flen >= 64
  double r;
  __asm__("fmadd.d %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
#else
  return __builtin_fma(a, b, c);
#endif
}

// 2^k * 2^r for |r| <= 1/2 + 2^-20 and -1000 < k < 1000, about 1e-15
// relative error: an e^(r ln 2) series evaluated with explicit fmas.
__VX_CUDA_FP_HD double exp2_kr(int k, double r) {
  const double t = r * 0.69314718055994530942;
  double p = 1.0 / 479001600.0;
  p = fma_d(p, t, 1.0 / 39916800.0);
  p = fma_d(p, t, 1.0 / 3628800.0);
  p = fma_d(p, t, 1.0 / 362880.0);
  p = fma_d(p, t, 1.0 / 40320.0);
  p = fma_d(p, t, 1.0 / 5040.0);
  p = fma_d(p, t, 1.0 / 720.0);
  p = fma_d(p, t, 1.0 / 120.0);
  p = fma_d(p, t, 1.0 / 24.0);
  p = fma_d(p, t, 1.0 / 6.0);
  p = fma_d(p, t, 0.5);
  p = fma_d(p, t, 1.0);
  p = fma_d(p, t, 1.0);
  return p * u2d((uint64_t)(k + 1023) << 52);
}

__VX_CUDA_FP_HD int round_half_away(double z) { return (int)(z + (z >= 0 ? 0.5 : -0.5)); }

// log2(e) as two 24-bit parts, so that x * part is exact for a float x.
constexpr double LOG2E_HI = (double)(float)1.44269504088896340736;
constexpr double LOG2E_LO = (double)(float)(1.44269504088896340736 - LOG2E_HI);

// Sign of m*m*x - 1, exactly (m has at most 26 significant bits, x 24).
__VX_CUDA_FP_HD int sq_x_cmp1(double m, double x) {
  const double a = m * m;
  const double hi = a * x;
  const double lo = fma_d(a, x, -hi);
  if (hi != 1.0) return hi > 1.0 ? 1 : -1;
  return lo > 0 ? 1 : (lo < 0 ? -1 : 0);
}

// Correctly rounded 1/sqrt(x) for x in [1, 4): the f64 estimate is within one
// float ulp; the midpoints around it are then compared with 1/sqrt(x) exactly.
// No midpoint is ever exactly 1/sqrt(x) (x would then be 1/m^2, not a float).
__VX_CUDA_FP_HD float rsqrt_1_4(float x) {
  const double xd = x;
  float y = (float)(1.0 / __builtin_sqrt(xd));
  for (int i = 0; i < 2; ++i) {
    const uint32_t yu = f2u(y);
    const double yd = y;
    const double up = (yd + (double)u2f(yu + 1)) * 0.5;
    const double dn = (yd + (double)u2f(yu - 1)) * 0.5;
    if (sq_x_cmp1(up, xd) < 0)
      y = u2f(yu + 1);
    else if (sq_x_cmp1(dn, xd) > 0)
      y = u2f(yu - 1);
    else
      break;
  }
  return y;
}

} // namespace vx_cuda_fp

typedef unsigned char __nv_fp8_storage_t;

struct __half_raw { unsigned short x; };
struct __half2_raw { unsigned short x, y; };
struct __nv_bfloat16_raw { unsigned short x; };
struct __nv_bfloat162_raw { unsigned short x, y; };

#define __VX_CUDA_FP_SCALAR(T, RAW, FMT, TOF)                                                            \
  struct T {                                                                                             \
    unsigned short __x;                                                                                  \
    T() = default;                                                                                       \
    __VX_CUDA_FP_HD constexpr T(const RAW& r) : __x(r.x) {}                                              \
    __VX_CUDA_FP_HD T(float f) : __x((unsigned short)vx_cuda_fp::from_f32<FMT>(f)) {}                     \
    __VX_CUDA_FP_HD T(double f) : __x((unsigned short)vx_cuda_fp::from_f64<FMT>(f)) {}                    \
    __VX_CUDA_FP_HD T(short v) : __x((unsigned short)vx_cuda_fp::from_i64<FMT>(v)) {}                     \
    __VX_CUDA_FP_HD T(unsigned short v) : __x((unsigned short)vx_cuda_fp::from_u64<FMT>(v, false)) {}     \
    __VX_CUDA_FP_HD T(int v) : __x((unsigned short)vx_cuda_fp::from_i64<FMT>(v)) {}                       \
    __VX_CUDA_FP_HD T(unsigned int v) : __x((unsigned short)vx_cuda_fp::from_u64<FMT>(v, false)) {}       \
    __VX_CUDA_FP_HD T(long v) : __x((unsigned short)vx_cuda_fp::from_i64<FMT>(v)) {}                      \
    __VX_CUDA_FP_HD T(unsigned long v) : __x((unsigned short)vx_cuda_fp::from_u64<FMT>(v, false)) {}      \
    __VX_CUDA_FP_HD T(long long v) : __x((unsigned short)vx_cuda_fp::from_i64<FMT>(v)) {}                 \
    __VX_CUDA_FP_HD T(unsigned long long v) : __x((unsigned short)vx_cuda_fp::from_u64<FMT>(v, false)) {} \
    __VX_CUDA_FP_HD T& operator=(const RAW& r) { __x = r.x; return *this; }                              \
    __VX_CUDA_FP_HD operator RAW() const { RAW r; r.x = __x; return r; }                                 \
    __VX_CUDA_FP_HD operator float() const { return vx_cuda_fp::TOF(__x); }                              \
  }

__VX_CUDA_FP_SCALAR(__half, __half_raw, vx_cuda_fp::fmt_f16, f16_to_f32);
__VX_CUDA_FP_SCALAR(__nv_bfloat16, __nv_bfloat16_raw, vx_cuda_fp::fmt_bf16, bf16_to_f32);

#define __VX_CUDA_FP_PAIR(T2, T, RAW2)                                              \
  struct alignas(4) T2 {                                                            \
    T x, y;                                                                         \
    T2() = default;                                                                 \
    __VX_CUDA_FP_HD constexpr T2(const T& a, const T& b) : x(a), y(b) {}            \
    __VX_CUDA_FP_HD T2(const RAW2& r) { x.__x = r.x; y.__x = r.y; }                 \
    __VX_CUDA_FP_HD T2& operator=(const RAW2& r) { x.__x = r.x; y.__x = r.y; return *this; } \
    __VX_CUDA_FP_HD operator RAW2() const { RAW2 r; r.x = x.__x; r.y = y.__x; return r; }   \
  }

__VX_CUDA_FP_PAIR(__half2, __half, __half2_raw);
__VX_CUDA_FP_PAIR(__nv_bfloat162, __nv_bfloat16, __nv_bfloat162_raw);

#undef __VX_CUDA_FP_SCALAR
#undef __VX_CUDA_FP_PAIR

typedef __half half;
typedef __half2 half2;
typedef __nv_bfloat16 nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

// e4m3: bias 7, no infinity, NaN only as S.1111.111, max finite 448.
struct __nv_fp8_e4m3 {
  __nv_fp8_storage_t __x;
  __nv_fp8_e4m3() = default;
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(float f) : __x((__nv_fp8_storage_t)vx_cuda_fp::from_f32<vx_cuda_fp::fmt_e4m3>(f, true)) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(double f) : __x((__nv_fp8_storage_t)vx_cuda_fp::from_f64<vx_cuda_fp::fmt_e4m3>(f, true)) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(__half f) : __nv_fp8_e4m3((float)f) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(__nv_bfloat16 f) : __nv_fp8_e4m3((float)f) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(short v) : __x((__nv_fp8_storage_t)vx_cuda_fp::from_i64<vx_cuda_fp::fmt_e4m3>(v, true)) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(unsigned short v) : __x((__nv_fp8_storage_t)vx_cuda_fp::from_u64<vx_cuda_fp::fmt_e4m3>(v, false, true)) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(int v) : __x((__nv_fp8_storage_t)vx_cuda_fp::from_i64<vx_cuda_fp::fmt_e4m3>(v, true)) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(unsigned int v) : __x((__nv_fp8_storage_t)vx_cuda_fp::from_u64<vx_cuda_fp::fmt_e4m3>(v, false, true)) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(long v) : __x((__nv_fp8_storage_t)vx_cuda_fp::from_i64<vx_cuda_fp::fmt_e4m3>(v, true)) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(unsigned long v) : __x((__nv_fp8_storage_t)vx_cuda_fp::from_u64<vx_cuda_fp::fmt_e4m3>(v, false, true)) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(long long v) : __x((__nv_fp8_storage_t)vx_cuda_fp::from_i64<vx_cuda_fp::fmt_e4m3>(v, true)) {}
  __VX_CUDA_FP_HD explicit __nv_fp8_e4m3(unsigned long long v) : __x((__nv_fp8_storage_t)vx_cuda_fp::from_u64<vx_cuda_fp::fmt_e4m3>(v, false, true)) {}
  __VX_CUDA_FP_HD explicit operator __half() const { __half_raw r; r.x = vx_cuda_fp::e4m3_to_f16(__x); return __half(r); }
  __VX_CUDA_FP_HD explicit operator float() const { return vx_cuda_fp::f16_to_f32(vx_cuda_fp::e4m3_to_f16(__x)); }
};

namespace vx_cuda_fp {

template <class T> struct fmt_of;
template <> struct fmt_of<__half> : fmt_f16 {};
template <> struct fmt_of<__nv_bfloat16> : fmt_bf16 {};

template <class T> __VX_CUDA_FP_HD T mk(uint32_t b) { T r; r.__x = (unsigned short)b; return r; }
template <class T> __VX_CUDA_FP_HD T rnd(float f) { return mk<T>(from_f32<fmt_of<T>>(f)); }
template <class T> __VX_CUDA_FP_HD bool is_nan(T a) { return (a.__x & 0x7fffu) > fmt_of<T>::INF; }
template <class T> __VX_CUDA_FP_HD T qnan() { return mk<T>(fmt_of<T>::NAN_); }

template <class T> __VX_CUDA_FP_HD T add(T a, T b) { return rnd<T>((float)a + (float)b); }
template <class T> __VX_CUDA_FP_HD T sub(T a, T b) { return rnd<T>((float)a - (float)b); }
template <class T> __VX_CUDA_FP_HD T mul(T a, T b) { return rnd<T>((float)a * (float)b); }
template <class T> __VX_CUDA_FP_HD T div(T a, T b) { return rnd<T>((float)a / (float)b); }
template <class T> __VX_CUDA_FP_HD T sqrt(T a) { return rnd<T>(__builtin_sqrtf((float)a)); }

// a*b is exact in f64; TwoSum gives the exact error of +c, and round-to-odd
// on that error keeps the single rounding to the narrow type correct.
template <class T> __VX_CUDA_FP_HD T fma(T a, T b, T c) {
  const double p = (double)(float)a * (double)(float)b;
  const double q = (double)(float)c;
  const double s = p + q;
  uint64_t su = d2u(s);
  if ((su & 0x7ff0000000000000ull) == 0x7ff0000000000000ull)
    return (su & 0xfffffffffffffull) ? qnan<T>() : mk<T>((su >> 63) ? (fmt_of<T>::SIGN | fmt_of<T>::INF) : fmt_of<T>::INF);
  const double bq = s - p;
  const double t = (p - (s - bq)) + (q - bq);
  if (t != 0 && !(su & 1))
    su = ((t > 0) == (s > 0)) ? su + 1 : su - 1;
  return mk<T>(from_f64<fmt_of<T>>(u2d(su)));
}

template <class T> __VX_CUDA_FP_HD T neg(T a) { return is_nan(a) ? qnan<T>() : mk<T>(a.__x ^ 0x8000u); }
template <class T> __VX_CUDA_FP_HD T abs(T a) { return is_nan(a) ? qnan<T>() : mk<T>(a.__x & 0x7fffu); }

template <class T> __VX_CUDA_FP_HD T max(T a, T b) {
  const bool an = is_nan(a), bn = is_nan(b);
  if (an && bn) return qnan<T>();
  if (an) return b;
  if (bn) return a;
  const float fa = a, fb = b;
  if (fa != fb) return fa > fb ? a : b;
  return (a.__x & 0x8000u) ? b : a;
}
template <class T> __VX_CUDA_FP_HD T min(T a, T b) {
  const bool an = is_nan(a), bn = is_nan(b);
  if (an && bn) return qnan<T>();
  if (an) return b;
  if (bn) return a;
  const float fa = a, fb = b;
  if (fa != fb) return fa < fb ? a : b;
  return (a.__x & 0x8000u) ? a : b;
}
template <class T> __VX_CUDA_FP_HD T max_nan(T a, T b) { return (is_nan(a) || is_nan(b)) ? qnan<T>() : vx_cuda_fp::max(a, b); }
template <class T> __VX_CUDA_FP_HD T min_nan(T a, T b) { return (is_nan(a) || is_nan(b)) ? qnan<T>() : vx_cuda_fp::min(a, b); }

template <class T> __VX_CUDA_FP_HD int inf_sign(T a) {
  return ((a.__x & 0x7fffu) == fmt_of<T>::INF) ? ((a.__x & 0x8000u) ? -1 : 1) : 0;
}

template <class T> __VX_CUDA_FP_HD T one() { return mk<T>(fmt_of<T>::BIAS << fmt_of<T>::MB); }
template <class T> __VX_CUDA_FP_HD T flag(bool p) { return p ? one<T>() : mk<T>(0); }

} // namespace vx_cuda_fp

// Conversions and bit casts.

__VX_CUDA_FP_HD __half __float2half(const float a) { return __half(a); }
__VX_CUDA_FP_HD __half __float2half_rn(const float a) { return __half(a); }
__VX_CUDA_FP_HD __half __double2half(const double a) { return __half(a); }
__VX_CUDA_FP_HD float __half2float(const __half a) { return (float)a; }
__VX_CUDA_FP_HD __half2 __float2half2_rn(const float a) { const __half h(a); return __half2(h, h); }
__VX_CUDA_FP_HD __half2 __floats2half2_rn(const float a, const float b) { return __half2(__half(a), __half(b)); }
__VX_CUDA_FP_HD __half2 __float22half2_rn(const float2 a) { return __half2(__half(a.x), __half(a.y)); }
__VX_CUDA_FP_HD float2 __half22float2(const __half2 a) { return make_float2((float)a.x, (float)a.y); }
__VX_CUDA_FP_HD __half __low2half(const __half2 a) { return a.x; }
__VX_CUDA_FP_HD __half __high2half(const __half2 a) { return a.y; }
__VX_CUDA_FP_HD __half2 __halves2half2(const __half a, const __half b) { return __half2(a, b); }
__VX_CUDA_FP_HD __half2 __half2half2(const __half a) { return __half2(a, a); }
__VX_CUDA_FP_HD float __low2float(const __half2 a) { return (float)a.x; }
__VX_CUDA_FP_HD float __high2float(const __half2 a) { return (float)a.y; }
__VX_CUDA_FP_HD unsigned short __half_as_ushort(const __half h) { return h.__x; }
__VX_CUDA_FP_HD short __half_as_short(const __half h) { return (short)h.__x; }
__VX_CUDA_FP_HD __half __ushort_as_half(const unsigned short i) { return vx_cuda_fp::mk<__half>(i); }
__VX_CUDA_FP_HD __half __short_as_half(const short i) { return vx_cuda_fp::mk<__half>((unsigned short)i); }

__VX_CUDA_FP_HD __nv_bfloat16 __float2bfloat16(const float a) { return __nv_bfloat16(a); }
__VX_CUDA_FP_HD __nv_bfloat16 __float2bfloat16_rn(const float a) { return __nv_bfloat16(a); }
__VX_CUDA_FP_HD __nv_bfloat16 __double2bfloat16(const double a) { return __nv_bfloat16(a); }
__VX_CUDA_FP_HD float __bfloat162float(const __nv_bfloat16 a) { return (float)a; }
__VX_CUDA_FP_HD __nv_bfloat162 __float2bfloat162_rn(const float a) { const __nv_bfloat16 h(a); return __nv_bfloat162(h, h); }
__VX_CUDA_FP_HD __nv_bfloat162 __floats2bfloat162_rn(const float a, const float b) { return __nv_bfloat162(__nv_bfloat16(a), __nv_bfloat16(b)); }
__VX_CUDA_FP_HD __nv_bfloat162 __float22bfloat162_rn(const float2 a) { return __nv_bfloat162(__nv_bfloat16(a.x), __nv_bfloat16(a.y)); }
__VX_CUDA_FP_HD float2 __bfloat1622float2(const __nv_bfloat162 a) { return make_float2((float)a.x, (float)a.y); }
__VX_CUDA_FP_HD __nv_bfloat16 __low2bfloat16(const __nv_bfloat162 a) { return a.x; }
__VX_CUDA_FP_HD __nv_bfloat16 __high2bfloat16(const __nv_bfloat162 a) { return a.y; }
__VX_CUDA_FP_HD __nv_bfloat162 __halves2bfloat162(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __nv_bfloat162(a, b); }
__VX_CUDA_FP_HD __nv_bfloat162 __bfloat162bfloat162(const __nv_bfloat16 a) { return __nv_bfloat162(a, a); }
__VX_CUDA_FP_HD float __low2float(const __nv_bfloat162 a) { return (float)a.x; }
__VX_CUDA_FP_HD float __high2float(const __nv_bfloat162 a) { return (float)a.y; }
__VX_CUDA_FP_HD unsigned short __bfloat16_as_ushort(const __nv_bfloat16 h) { return h.__x; }
__VX_CUDA_FP_HD short __bfloat16_as_short(const __nv_bfloat16 h) { return (short)h.__x; }
__VX_CUDA_FP_HD __nv_bfloat16 __ushort_as_bfloat16(const unsigned short i) { return vx_cuda_fp::mk<__nv_bfloat16>(i); }
__VX_CUDA_FP_HD __nv_bfloat16 __short_as_bfloat16(const short i) { return vx_cuda_fp::mk<__nv_bfloat16>((unsigned short)i); }

__VX_CUDA_FP_HD int __float_as_int(const float x) { return (int)vx_cuda_fp::f2u(x); }
__VX_CUDA_FP_HD unsigned int __float_as_uint(const float x) { return vx_cuda_fp::f2u(x); }
__VX_CUDA_FP_HD float __int_as_float(const int x) { return vx_cuda_fp::u2f((uint32_t)x); }
__VX_CUDA_FP_HD float __uint_as_float(const unsigned int x) { return vx_cuda_fp::u2f(x); }
__VX_CUDA_FP_HD long long __double_as_longlong(const double x) { return (long long)vx_cuda_fp::d2u(x); }
__VX_CUDA_FP_HD double __longlong_as_double(const long long x) { return vx_cuda_fp::u2d((uint64_t)x); }

// Arithmetic, comparisons and operators, the same set for __half/__half2 and
// __nv_bfloat16/__nv_bfloat162 (CUDA overloads the names on both).

#define __VX_CUDA_FP_OPS(T, T2)                                                                                  \
  __VX_CUDA_FP_HD T __hadd(const T a, const T b) { return vx_cuda_fp::add(a, b); }                               \
  __VX_CUDA_FP_HD T __hsub(const T a, const T b) { return vx_cuda_fp::sub(a, b); }                               \
  __VX_CUDA_FP_HD T __hmul(const T a, const T b) { return vx_cuda_fp::mul(a, b); }                               \
  __VX_CUDA_FP_HD T __hdiv(const T a, const T b) { return vx_cuda_fp::div(a, b); }                               \
  __VX_CUDA_FP_HD T __hfma(const T a, const T b, const T c) { return vx_cuda_fp::fma(a, b, c); }                 \
  __VX_CUDA_FP_HD T __hneg(const T a) { return vx_cuda_fp::neg(a); }                                             \
  __VX_CUDA_FP_HD T __habs(const T a) { return vx_cuda_fp::abs(a); }                                             \
  __VX_CUDA_FP_HD T __hmax(const T a, const T b) { return vx_cuda_fp::max(a, b); }                               \
  __VX_CUDA_FP_HD T __hmin(const T a, const T b) { return vx_cuda_fp::min(a, b); }                               \
  __VX_CUDA_FP_HD T __hmax_nan(const T a, const T b) { return vx_cuda_fp::max_nan(a, b); }                       \
  __VX_CUDA_FP_HD T __hmin_nan(const T a, const T b) { return vx_cuda_fp::min_nan(a, b); }                       \
  __VX_CUDA_FP_HD T hsqrt(const T a) { return vx_cuda_fp::sqrt(a); }                                             \
  __VX_CUDA_FP_HD bool __heq(const T a, const T b) { return (float)a == (float)b; }                              \
  __VX_CUDA_FP_HD bool __hne(const T a, const T b) { return (float)a < (float)b || (float)a > (float)b; }        \
  __VX_CUDA_FP_HD bool __hlt(const T a, const T b) { return (float)a < (float)b; }                               \
  __VX_CUDA_FP_HD bool __hle(const T a, const T b) { return (float)a <= (float)b; }                              \
  __VX_CUDA_FP_HD bool __hgt(const T a, const T b) { return (float)a > (float)b; }                               \
  __VX_CUDA_FP_HD bool __hge(const T a, const T b) { return (float)a >= (float)b; }                              \
  __VX_CUDA_FP_HD bool __hequ(const T a, const T b) { return !__hne(a, b); }                                     \
  __VX_CUDA_FP_HD bool __hneu(const T a, const T b) { return !__heq(a, b); }                                     \
  __VX_CUDA_FP_HD bool __hltu(const T a, const T b) { return !__hge(a, b); }                                     \
  __VX_CUDA_FP_HD bool __hleu(const T a, const T b) { return !__hgt(a, b); }                                     \
  __VX_CUDA_FP_HD bool __hgtu(const T a, const T b) { return !__hle(a, b); }                                     \
  __VX_CUDA_FP_HD bool __hgeu(const T a, const T b) { return !__hlt(a, b); }                                     \
  __VX_CUDA_FP_HD bool __hisnan(const T a) { return vx_cuda_fp::is_nan(a); }                                        \
  __VX_CUDA_FP_HD int __hisinf(const T a) { return vx_cuda_fp::inf_sign(a); }                                       \
                                                                                                                 \
  __VX_CUDA_FP_HD T2 __hadd2(const T2 a, const T2 b) { return T2(__hadd(a.x, b.x), __hadd(a.y, b.y)); }          \
  __VX_CUDA_FP_HD T2 __hsub2(const T2 a, const T2 b) { return T2(__hsub(a.x, b.x), __hsub(a.y, b.y)); }          \
  __VX_CUDA_FP_HD T2 __hmul2(const T2 a, const T2 b) { return T2(__hmul(a.x, b.x), __hmul(a.y, b.y)); }          \
  __VX_CUDA_FP_HD T2 __h2div(const T2 a, const T2 b) { return T2(__hdiv(a.x, b.x), __hdiv(a.y, b.y)); }          \
  __VX_CUDA_FP_HD T2 __hfma2(const T2 a, const T2 b, const T2 c) {                                               \
    return T2(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y));                                                     \
  }                                                                                                              \
  __VX_CUDA_FP_HD T2 __hneg2(const T2 a) { return T2(__hneg(a.x), __hneg(a.y)); }                                \
  __VX_CUDA_FP_HD T2 __habs2(const T2 a) { return T2(__habs(a.x), __habs(a.y)); }                                \
  __VX_CUDA_FP_HD T2 __hmax2(const T2 a, const T2 b) { return T2(__hmax(a.x, b.x), __hmax(a.y, b.y)); }          \
  __VX_CUDA_FP_HD T2 __hmin2(const T2 a, const T2 b) { return T2(__hmin(a.x, b.x), __hmin(a.y, b.y)); }          \
  __VX_CUDA_FP_HD T2 __hmax2_nan(const T2 a, const T2 b) { return T2(__hmax_nan(a.x, b.x), __hmax_nan(a.y, b.y)); } \
  __VX_CUDA_FP_HD T2 __hmin2_nan(const T2 a, const T2 b) { return T2(__hmin_nan(a.x, b.x), __hmin_nan(a.y, b.y)); } \
  __VX_CUDA_FP_HD T2 h2sqrt(const T2 a) { return T2(hsqrt(a.x), hsqrt(a.y)); }                                   \
  __VX_CUDA_FP_HD T2 __hisnan2(const T2 a) {                                                                     \
    return T2(vx_cuda_fp::flag<T>(__hisnan(a.x)), vx_cuda_fp::flag<T>(__hisnan(a.y)));                           \
  }                                                                                                              \
  __VX_CUDA_FP_CMP2(T, T2, eq) __VX_CUDA_FP_CMP2(T, T2, ne) __VX_CUDA_FP_CMP2(T, T2, lt)                         \
  __VX_CUDA_FP_CMP2(T, T2, le) __VX_CUDA_FP_CMP2(T, T2, gt) __VX_CUDA_FP_CMP2(T, T2, ge)                         \
  __VX_CUDA_FP_CMP2(T, T2, equ) __VX_CUDA_FP_CMP2(T, T2, neu) __VX_CUDA_FP_CMP2(T, T2, ltu)                      \
  __VX_CUDA_FP_CMP2(T, T2, leu) __VX_CUDA_FP_CMP2(T, T2, gtu) __VX_CUDA_FP_CMP2(T, T2, geu)                      \
                                                                                                                 \
  __VX_CUDA_FP_HD T operator+(const T& a, const T& b) { return __hadd(a, b); }                                   \
  __VX_CUDA_FP_HD T operator-(const T& a, const T& b) { return __hsub(a, b); }                                   \
  __VX_CUDA_FP_HD T operator*(const T& a, const T& b) { return __hmul(a, b); }                                   \
  __VX_CUDA_FP_HD T operator/(const T& a, const T& b) { return __hdiv(a, b); }                                   \
  __VX_CUDA_FP_HD T& operator+=(T& a, const T& b) { a = __hadd(a, b); return a; }                                \
  __VX_CUDA_FP_HD T& operator-=(T& a, const T& b) { a = __hsub(a, b); return a; }                                \
  __VX_CUDA_FP_HD T& operator*=(T& a, const T& b) { a = __hmul(a, b); return a; }                                \
  __VX_CUDA_FP_HD T& operator/=(T& a, const T& b) { a = __hdiv(a, b); return a; }                                \
  __VX_CUDA_FP_HD T& operator++(T& a) { a = __hadd(a, vx_cuda_fp::one<T>()); return a; }                         \
  __VX_CUDA_FP_HD T& operator--(T& a) { a = __hsub(a, vx_cuda_fp::one<T>()); return a; }                         \
  __VX_CUDA_FP_HD T operator++(T& a, int) { const T r = a; ++a; return r; }                                      \
  __VX_CUDA_FP_HD T operator--(T& a, int) { const T r = a; --a; return r; }                                      \
  __VX_CUDA_FP_HD T operator+(const T& a) { return a; }                                                          \
  __VX_CUDA_FP_HD T operator-(const T& a) { return __hneg(a); }                                                  \
  __VX_CUDA_FP_HD bool operator==(const T& a, const T& b) { return __heq(a, b); }                                \
  __VX_CUDA_FP_HD bool operator!=(const T& a, const T& b) { return __hneu(a, b); }                               \
  __VX_CUDA_FP_HD bool operator<(const T& a, const T& b) { return __hlt(a, b); }                                 \
  __VX_CUDA_FP_HD bool operator<=(const T& a, const T& b) { return __hle(a, b); }                                \
  __VX_CUDA_FP_HD bool operator>(const T& a, const T& b) { return __hgt(a, b); }                                 \
  __VX_CUDA_FP_HD bool operator>=(const T& a, const T& b) { return __hge(a, b); }                                \
                                                                                                                 \
  __VX_CUDA_FP_HD T2 operator+(const T2& a, const T2& b) { return __hadd2(a, b); }                               \
  __VX_CUDA_FP_HD T2 operator-(const T2& a, const T2& b) { return __hsub2(a, b); }                               \
  __VX_CUDA_FP_HD T2 operator*(const T2& a, const T2& b) { return __hmul2(a, b); }                               \
  __VX_CUDA_FP_HD T2 operator/(const T2& a, const T2& b) { return __h2div(a, b); }                               \
  __VX_CUDA_FP_HD T2& operator+=(T2& a, const T2& b) { a = __hadd2(a, b); return a; }                            \
  __VX_CUDA_FP_HD T2& operator-=(T2& a, const T2& b) { a = __hsub2(a, b); return a; }                            \
  __VX_CUDA_FP_HD T2& operator*=(T2& a, const T2& b) { a = __hmul2(a, b); return a; }                            \
  __VX_CUDA_FP_HD T2& operator/=(T2& a, const T2& b) { a = __h2div(a, b); return a; }                            \
  __VX_CUDA_FP_HD T2 operator+(const T2& a) { return a; }                                                        \
  __VX_CUDA_FP_HD T2 operator-(const T2& a) { return __hneg2(a); }                                               \
  __VX_CUDA_FP_HD bool operator==(const T2& a, const T2& b) { return __hbeq2(a, b); }                            \
  __VX_CUDA_FP_HD bool operator!=(const T2& a, const T2& b) { return __hbneu2(a, b); }                           \
  __VX_CUDA_FP_HD bool operator<(const T2& a, const T2& b) { return __hblt2(a, b); }                             \
  __VX_CUDA_FP_HD bool operator<=(const T2& a, const T2& b) { return __hble2(a, b); }                            \
  __VX_CUDA_FP_HD bool operator>(const T2& a, const T2& b) { return __hbgt2(a, b); }                             \
  __VX_CUDA_FP_HD bool operator>=(const T2& a, const T2& b) { return __hbge2(a, b); }

// __h<op>2 sets each half to 1.0 or 0.0; __hb<op>2 is true when both halves hold.
#define __VX_CUDA_FP_CMP2(T, T2, op)                                                                             \
  __VX_CUDA_FP_HD T2 __h##op##2(const T2 a, const T2 b) {                                                        \
    return T2(vx_cuda_fp::flag<T>(__h##op(a.x, b.x)), vx_cuda_fp::flag<T>(__h##op(a.y, b.y)));                   \
  }                                                                                                              \
  __VX_CUDA_FP_HD bool __hb##op##2(const T2 a, const T2 b) { return __h##op(a.x, b.x) && __h##op(a.y, b.y); }

__VX_CUDA_FP_OPS(__half, __half2)
__VX_CUDA_FP_OPS(__nv_bfloat16, __nv_bfloat162)

#undef __VX_CUDA_FP_OPS
#undef __VX_CUDA_FP_CMP2

// Single-precision intrinsics. The _rn forms are correctly rounded; NaN
// results are F32_NAN.

__VX_CUDA_FP_HD float __fdiv_rn(const float x, const float y) { return vx_cuda_fp::canon_f32(x / y); }
__VX_CUDA_FP_HD float __frcp_rn(const float x) { return vx_cuda_fp::canon_f32(1.0f / x); }
__VX_CUDA_FP_HD float __fsqrt_rn(const float x) { return vx_cuda_fp::canon_f32(__builtin_sqrtf(x)); }

// rsqrt(4^k x) = rsqrt(x) / 2^k exactly, and every result is a normal float,
// so x is reduced to [1, 4) and the exponent is restored afterwards.
__VX_CUDA_FP_HD float __frsqrt_rn(const float x) {
  const uint32_t u = vx_cuda_fp::f2u(x);
  if ((u & 0x7fffffff) == 0) return vx_cuda_fp::u2f((u & 0x80000000) | 0x7f800000);
  if (u > 0x7f800000) return vx_cuda_fp::u2f(vx_cuda_fp::F32_NAN);
  if (u == 0x7f800000) return 0.0f;
  int ex = (int)(u >> 23);
  uint32_t m = u & 0x7fffff;
  if (ex == 0) {
    const int sh = __builtin_clz(m) - 8;
    m = (m << sh) & 0x7fffff;
    ex = 1 - sh;
  }
  const int e = ex - 127, k = (e - (e & 1)) / 2;
  const float y = vx_cuda_fp::rsqrt_1_4(vx_cuda_fp::u2f(((uint32_t)(127 + e - 2 * k) << 23) | m));
  return vx_cuda_fp::u2f((uint32_t)((int32_t)vx_cuda_fp::f2u(y) - k * (1 << 23)));
}

// CUDA's rsqrtf is within 2 ulp; the correctly rounded result is used.
__VX_CUDA_FP_HD float rsqrtf(const float x) { return __frsqrt_rn(x); }

// Within one ulp of e^x over the whole range (CUDA bounds __expf by
// 2 + floor(|1.173 x|) ulp).
__VX_CUDA_FP_HD float __expf(const float x) {
  if (x != x) return vx_cuda_fp::u2f(vx_cuda_fp::F32_NAN);
  if (x > 89.0f) return vx_cuda_fp::u2f(0x7f800000);
  if (x < -110.0f) return 0.0f;
  const double zh = (double)x * vx_cuda_fp::LOG2E_HI, zl = (double)x * vx_cuda_fp::LOG2E_LO;
  const int k = vx_cuda_fp::round_half_away(zh);
  return (float)vx_cuda_fp::exp2_kr(k, (zh - (double)k) + zl);
}

// ex2.approx.ftz.f32: subnormal inputs and results flush to zero.
__VX_CUDA_FP_HD float __vx_ex2_approx(const float x) {
  if (x != x) return vx_cuda_fp::u2f(vx_cuda_fp::F32_NAN);
  if ((vx_cuda_fp::f2u(x) & 0x7f800000) == 0) return 1.0f;
  if (x >= 128.0f) return vx_cuda_fp::u2f(0x7f800000);
  if (x < -126.0f) return 0.0f;
  const int k = vx_cuda_fp::round_half_away(x);
  return (float)vx_cuda_fp::exp2_kr(k, (double)x - (double)k);
}

// As CUDA states: for 2^126 < |y| < 2^128 the result is 0, or NaN when x is
// infinite (x times the flushed reciprocal of y).
__VX_CUDA_FP_HD float __fdividef(const float x, const float y) {
  const uint32_t ay = vx_cuda_fp::f2u(y) & 0x7fffffff;
  if (ay > 0x7e800000 && ay < 0x7f800000) {
    const uint32_t ax = vx_cuda_fp::f2u(x) & 0x7fffffff;
    if (ax >= 0x7f800000) return vx_cuda_fp::u2f(vx_cuda_fp::F32_NAN);
    return vx_cuda_fp::u2f((vx_cuda_fp::f2u(x) ^ vx_cuda_fp::f2u(y)) & 0x80000000);
  }
  return __fdiv_rn(x, y);
}

#undef __VX_CUDA_FP_HD
