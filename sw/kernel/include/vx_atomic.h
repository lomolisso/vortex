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

#ifndef __VX_ATOMIC_H__
#define __VX_ATOMIC_H__

// PTX atom/red forms beyond RISC-V A: float add, inc/dec, cas.b16 and the
// global vector forms (golden model: sw/common/amo_ref.h).
//
//   vx_atom_[<space>_]<op>_<sem>_<scope>_<type>(p, v)  returns the old value
//   vx_red_[<space>_]<op>_<sem>_<scope>_<type>(p, v)   discards it (rd = x0)
//   space  none (generic), global, shared (::cta), shared_cluster
//   sem    relaxed, acquire, release, acq_rel (red: relaxed, release)
//   scope  cta, cluster, gpu
//   type   f32 f64 f16 bf16 f16x2 bf16x2 u32 (inc/dec) b16 (cas, atom only);
//          vectors, generic and global only, as (p, const T* v, T* old) with
//          old optional: v{2,4}_f32 and add/min/max v{2,4,8}_{f16,bf16},
//          v{2,4}_{f16x2,bf16x2}
//
// 16-bit floats are passed as their bit patterns. Each form lowers as nvcc
// does for sm_90: one FPAMO instruction at global memory; a compare-and-swap
// loop where nvcc emits one (16-bit scalar adds and cas.b16 everywhere,
// .shared::cta float adds); generic and .shared::cluster float adds test the
// address and take the instruction on global memory, the loop otherwise.
// Vectors are per-element atomic, as in PTX, and run as 64- or 32-bit packed
// instructions.

#if !defined(__riscv_xlen) || __riscv_xlen != 64
#error "vx_atomic.h requires XLEN 64"
#endif

#if !VX_CFG_EXT_FPAMO_ENABLED
#error "vx_atomic.h needs a device built with VX_CFG_EXT_FPAMO_ENABLE"
#endif

#include <vx_intrinsics.h>
#include <stdint.h>
#if VX_CFG_EXT_DLMEM_ENABLED
#include <vx_dlmem.h>
#endif

// The NaN results of amo_ref.h's loop site: canonical for the 16-bit and f32
// adds and for an invalid f64 add; a NaN f64 input comes back quieted.
#define __VX_ATOM_F16_QNAN  0x7fffu
#define __VX_ATOM_BF16_QNAN 0x7fffu
#define __VX_ATOM_F32_QNAN  0x7fffffffu
#define __VX_ATOM_F64_QNAN  0xfff8000000000000ull
#define __VX_ATOM_F64_QUIET 0x0008000000000000ull

#ifdef __cplusplus
extern "C" {
#endif

#define __VX_ATOM_INL static inline __attribute__((always_inline))

// custom-3 R-type: funct7 = funct5 << 2 | aq << 1 | rl, funct3 = width.
#define __VX_FPAMO(f5, f3, p, v) ({                                       \
    uint64_t __r;                                                         \
    __asm__ volatile (".insn r %1, %2, %3, %0, %4, %5"                    \
                      : "=r"(__r)                                         \
                      : "i"(RISCV_CUSTOM3), "i"(f3), "i"((f5) << 2),      \
                        "r"(p), "r"(v)                                    \
                      : "memory");                                        \
    __r; })

#define __VX_FPAMO_RED(f5, f3, p, v)                                      \
    __asm__ volatile (".insn r %0, %1, %2, x0, %3, %4"                    \
                      :: "i"(RISCV_CUSTOM3), "i"(f3), "i"((f5) << 2),     \
                         "r"(p), "r"(v)                                   \
                      : "memory")

// funct5 = amo_op - 16
#define __VX_F5_INC      0
#define __VX_F5_DEC      1
#define __VX_F5_FADD     2
#define __VX_F5_FADD2    4
#define __VX_F5_FADD2_BF 5
#define __VX_F5_FMIN2    6
#define __VX_F5_FMAX2    7
#define __VX_F5_FMIN2_BF 8
#define __VX_F5_FMAX2_BF 9

// amocas.{w,d}: AMO opcode, funct5 0x05; the comparand rides in rd.
#define __VX_AMOCAS(f3, p, cmp, v) ({                                     \
    uint64_t __c = (cmp);                                                 \
    __asm__ volatile (".insn r 0x2F, %2, 0x14, %0, %1, %3"                \
                      : "+r"(__c) : "r"(p), "i"(f3), "r"(v) : "memory");  \
    __c; })

__VX_ATOM_INL void __vx_atom_fence_cta(void) {
#if VX_CFG_EXT_DLMEM_ENABLED
  vx_fence_cta();
#else
  vx_fence();
#endif
}
#define __VX_ATOM_FENCE_cta     __vx_atom_fence_cta
#define __VX_ATOM_FENCE_cluster vx_fence_cluster
#define __VX_ATOM_FENCE_gpu     vx_fence

// The software address-space test of the generic forms (as QSPC on sm_90).
__VX_ATOM_INL int __vx_atom_is_global(const void* p) {
  const uintptr_t a = (uintptr_t)p;
  if (a - (uintptr_t)VX_MEM_LMEM_BASE_ADDR < ((uintptr_t)1 << VX_CFG_LMEM_LOG_SIZE))
    return 0;
#if VX_CFG_EXT_DLMEM_ENABLED
  if (a - (uintptr_t)VX_MEM_DLMEM_BASE_ADDR < (uintptr_t)vx_dlmem_window_size())
    return 0;
#endif
  return 1;
}

///////////////////////////////////////////////////////////////////////////////
// Software arithmetic of the loops. 16-bit floats go to f32 exactly, add there
// with RNE, and round back once more with RNE: p(f32) = 24 >= 2p + 2, so the
// result is the correctly rounded 16-bit sum. The shared loops keep
// subnormals, as the H100's FADD/DADD/HFMA2 fallbacks do. An atomic leaves
// fflags as it found them: each loop restores the flags its arithmetic raised.

__VX_ATOM_INL uint32_t __vx_f2u(float x)    { union { float f; uint32_t u; } c; c.f = x; return c.u; }
__VX_ATOM_INL float    __vx_u2f(uint32_t x) { union { float f; uint32_t u; } c; c.u = x; return c.f; }
__VX_ATOM_INL uint64_t __vx_d2u(double x)   { union { double f; uint64_t u; } c; c.f = x; return c.u; }
__VX_ATOM_INL double   __vx_u2d(uint64_t x) { union { double f; uint64_t u; } c; c.u = x; return c.f; }

__VX_ATOM_INL uint32_t __vx_fflags_get(void) {
  uint32_t f;
  __asm__ volatile ("csrr %0, fflags" : "=r"(f));
  return f;
}

__VX_ATOM_INL void __vx_fflags_set(uint32_t f) {
  __asm__ volatile ("csrw fflags, %0" :: "r"(f));
}

// volatile: kept between the loop's fflags save and restore.
__VX_ATOM_INL uint32_t __vx_fadd_s(uint32_t a, uint32_t b) {
  float r;
  __asm__ volatile ("fadd.s %0, %1, %2, rne" : "=f"(r) : "f"(__vx_u2f(a)), "f"(__vx_u2f(b)));
  return __vx_f2u(r);
}

__VX_ATOM_INL uint64_t __vx_fadd_d(uint64_t a, uint64_t b) {
  double r;
  __asm__ volatile ("fadd.d %0, %1, %2, rne" : "=f"(r) : "f"(__vx_u2d(a)), "f"(__vx_u2d(b)));
  return __vx_d2u(r);
}

__VX_ATOM_INL int __vx_f32_isnan(uint32_t x) { return (x & 0x7fffffffu) > 0x7f800000u; }

// The conversions select rather than branch: data-dependent branches here,
// inside a divergent CAS loop, are taken apart by lanes with special values.
__VX_ATOM_INL uint32_t __vx_fcvt_wu_s_rne(float x) {
  uint32_t r;
  __asm__ volatile ("fcvt.wu.s %0, %1, rne" : "=r"(r) : "f"(x));
  return r;
}

__VX_ATOM_INL uint32_t __vx_f16_to_f32(uint32_t h) {
  const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  const uint32_t normal = s | ((e + 112) << 23) | (m << 13);
  const uint32_t infnan = s | 0x7f800000u | (m << 13);
  const uint32_t sub    = s | __vx_f2u((float)m * 0x1p-24f);  // m * 2^-24 is exact
  const uint32_t r = (e == 0x1f) ? infnan : normal;
  return (e == 0) ? sub : r;
}

__VX_ATOM_INL uint32_t __vx_f32_to_f16(uint32_t x) {
  const uint32_t s = (x >> 16) & 0x8000u, a = x & 0x7fffffffu;
  const uint32_t h = a - 0x38000000u;
  const uint32_t normal = (h + 0xfffu + ((h >> 13) & 1)) >> 13;
  // Below 2^-14 the f16 step is 2^-24: scale exactly, round to an integer.
  const uint32_t sub = __vx_fcvt_wu_s_rne(__vx_u2f(a) * 0x1p24f);
  uint32_t r = (a >= 0x38800000u) ? normal : sub;
  r = (a >= 0x477ff000u) ? 0x7c00u : r;  // >= 65520 rounds to infinity
  return (a > 0x7f800000u) ? __VX_ATOM_F16_QNAN : (s | r);
}

__VX_ATOM_INL uint32_t __vx_f32_to_bf16(uint32_t x) {
  const uint32_t r = (x + 0x7fffu + ((x >> 16) & 1)) >> 16;
  return __vx_f32_isnan(x) ? __VX_ATOM_BF16_QNAN : r;
}

__VX_ATOM_INL uint32_t __vx_add_f16(uint32_t a, uint32_t b) {
  return __vx_f32_to_f16(__vx_fadd_s(__vx_f16_to_f32(a), __vx_f16_to_f32(b)));
}

__VX_ATOM_INL uint32_t __vx_add_bf16(uint32_t a, uint32_t b) {
  return __vx_f32_to_bf16(__vx_fadd_s(a << 16, b << 16));
}

__VX_ATOM_INL uint32_t __vx_add_f32(uint32_t a, uint32_t b) {
  uint32_t r = __vx_fadd_s(a, b);
  return __vx_f32_isnan(r) ? __VX_ATOM_F32_QNAN : r;
}

// All ones when x is an f64 NaN, else zero.
__VX_ATOM_INL uint64_t __vx_f64_nanmask(uint64_t x) {
  return -(uint64_t)((x & 0x7fffffffffffffffull) > 0x7ff0000000000000ull);
}

// fadd.d returns the canonical 0x7ff8... for any NaN; the H100's DADD loop
// returns the old NaN quieted, else the rhs NaN quieted. Masks, not branches.
__VX_ATOM_INL uint64_t __vx_add_f64(uint64_t a, uint64_t b) {
  const uint64_t r  = __vx_fadd_d(a, b);
  const uint64_t an = __vx_f64_nanmask(a), in = an | __vx_f64_nanmask(b), rn = __vx_f64_nanmask(r);
  const uint64_t q  = ((a & an) | (b & ~an)) | __VX_ATOM_F64_QUIET;
  const uint64_t n  = (q & in) | (__VX_ATOM_F64_QNAN & ~in);
  return (n & rn) | (r & ~rn);
}

enum { __VX_L_ADD_F16, __VX_L_ADD_BF16, __VX_L_ADD_F32, __VX_L_ADD_F16X2, __VX_L_ADD_BF16X2, __VX_L_ADD_F64 };

__VX_ATOM_INL uint32_t __vx_op32(int k, uint32_t a, uint32_t b) {
  switch (k) {
  case __VX_L_ADD_F16:   return __vx_add_f16(a, b);
  case __VX_L_ADD_BF16:  return __vx_add_bf16(a, b);
  case __VX_L_ADD_F32:   return __vx_add_f32(a, b);
  case __VX_L_ADD_F16X2: return __vx_add_f16(a & 0xffff, b & 0xffff) | (__vx_add_f16(a >> 16, b >> 16) << 16);
  default:               return __vx_add_bf16(a & 0xffff, b & 0xffff) | (__vx_add_bf16(a >> 16, b >> 16) << 16);
  }
}

// As nvcc: read the aligned word holding *p, compute the new half, merge it,
// amocas.w the word, retry with what the CAS saw.
__VX_ATOM_INL uint32_t __vx_loop16(void* p, uint32_t v, int k) {
  volatile uint32_t* w = (volatile uint32_t*)((uintptr_t)p & ~(uintptr_t)3);
  const uint32_t sh = ((uintptr_t)p & 2) * 8;
  const uint32_t fl = __vx_fflags_get();
  uint32_t old = *w;
  for (;;) {
    const uint32_t nw = (old & ~(0xffffu << sh)) | (__vx_op32(k, (old >> sh) & 0xffff, v & 0xffff) << sh);
    const uint32_t seen = (uint32_t)__VX_AMOCAS(2, w, (uint64_t)(int64_t)(int32_t)old, nw);
    if (seen == old)
      break;
    old = seen;
  }
  __vx_fflags_set(fl);
  return (old >> sh) & 0xffff;
}

__VX_ATOM_INL uint32_t __vx_loop32(void* p, uint32_t v, int k) {
  volatile uint32_t* w = (volatile uint32_t*)p;
  const uint32_t fl = __vx_fflags_get();
  uint32_t old = *w;
  for (;;) {
    const uint32_t seen = (uint32_t)__VX_AMOCAS(2, w, (uint64_t)(int64_t)(int32_t)old, __vx_op32(k, old, v));
    if (seen == old)
      break;
    old = seen;
  }
  __vx_fflags_set(fl);
  return old;
}

__VX_ATOM_INL uint64_t __vx_loop64(void* p, uint64_t v, int k) {
  volatile uint64_t* w = (volatile uint64_t*)p;
  const uint32_t fl = __vx_fflags_get();
  uint64_t old = *w;
  (void)k;
  for (;;) {
    const uint64_t seen = __VX_AMOCAS(3, w, old, __vx_add_f64(old, v));
    if (seen == old)
      break;
    old = seen;
  }
  __vx_fflags_set(fl);
  return old;
}

///////////////////////////////////////////////////////////////////////////////
// Relaxed forms per space: __vx_atom_<space_><op>_<type>, __vx_red_<space_><op>_<type>.

// Float adds: the instruction at global memory, the loop at .shared::cta, and
// for generic and .shared::cluster the instruction on a global address and the
// loop on a local or window one (the H100 has no shared-memory float atomic).
#define __VX_FADD_FORM(ty, T, U, TOU, FROMU, f5, f3, LOOP, K)                        \
  __VX_ATOM_INL T __vx_hw_add_##ty(T* p, T v) {                                      \
    return FROMU((U)__VX_FPAMO(f5, f3, p, TOU(v)));                                  \
  }                                                                                  \
  __VX_ATOM_INL void __vx_hwred_add_##ty(T* p, T v) { __VX_FPAMO_RED(f5, f3, p, TOU(v)); } \
  __VX_ATOM_INL T __vx_sw_add_##ty(T* p, T v) { return FROMU((U)LOOP(p, TOU(v), K)); } \
  __VX_ATOM_INL T __vx_atom_global_add_##ty(T* p, T v) { return __vx_hw_add_##ty(p, v); } \
  __VX_ATOM_INL void __vx_red_global_add_##ty(T* p, T v) { __vx_hwred_add_##ty(p, v); } \
  __VX_ATOM_INL T __vx_atom_shared_add_##ty(T* p, T v) { return __vx_sw_add_##ty(p, v); } \
  __VX_ATOM_INL void __vx_red_shared_add_##ty(T* p, T v) { (void)__vx_sw_add_##ty(p, v); } \
  __VX_ATOM_INL T __vx_atom_add_##ty(T* p, T v) {                                    \
    return __vx_atom_is_global(p) ? __vx_hw_add_##ty(p, v) : __vx_sw_add_##ty(p, v); \
  }                                                                                  \
  __VX_ATOM_INL void __vx_red_add_##ty(T* p, T v) {                                  \
    if (__vx_atom_is_global(p))                                                      \
      __vx_hwred_add_##ty(p, v);                                                     \
    else                                                                             \
      (void)__vx_sw_add_##ty(p, v);                                                  \
  }                                                                                  \
  __VX_ATOM_INL T __vx_atom_shared_cluster_add_##ty(T* p, T v) { return __vx_atom_add_##ty(p, v); } \
  __VX_ATOM_INL void __vx_red_shared_cluster_add_##ty(T* p, T v) { __vx_red_add_##ty(p, v); }

#define __VX_ID(x) (x)

__VX_FADD_FORM(f32,    float,    uint32_t, __vx_f2u, __vx_u2f, __VX_F5_FADD,     2, __vx_loop32, __VX_L_ADD_F32)
__VX_FADD_FORM(f64,    double,   uint64_t, __vx_d2u, __vx_u2d, __VX_F5_FADD,     3, __vx_loop64, __VX_L_ADD_F64)
__VX_FADD_FORM(f16x2,  uint32_t, uint32_t, __VX_ID,  __VX_ID,  __VX_F5_FADD2,    2, __vx_loop32, __VX_L_ADD_F16X2)
__VX_FADD_FORM(bf16x2, uint32_t, uint32_t, __VX_ID,  __VX_ID,  __VX_F5_FADD2_BF, 2, __vx_loop32, __VX_L_ADD_BF16X2)

// One lowering in every space.
#define __VX_ALL_SPACES(op, ty, T, ATOM, RED)                                        \
  __VX_ATOM_INL T __vx_atom_##op##_##ty(T* p, T v) { return ATOM; }                  \
  __VX_ATOM_INL void __vx_red_##op##_##ty(T* p, T v) { RED; }                        \
  __VX_ATOM_INL T __vx_atom_global_##op##_##ty(T* p, T v) { return ATOM; }           \
  __VX_ATOM_INL void __vx_red_global_##op##_##ty(T* p, T v) { RED; }                 \
  __VX_ATOM_INL T __vx_atom_shared_##op##_##ty(T* p, T v) { return ATOM; }           \
  __VX_ATOM_INL void __vx_red_shared_##op##_##ty(T* p, T v) { RED; }                 \
  __VX_ATOM_INL T __vx_atom_shared_cluster_##op##_##ty(T* p, T v) { return ATOM; }   \
  __VX_ATOM_INL void __vx_red_shared_cluster_##op##_##ty(T* p, T v) { RED; }

__VX_ALL_SPACES(add, f16,  uint16_t, (uint16_t)__vx_loop16(p, v, __VX_L_ADD_F16),  (void)__vx_loop16(p, v, __VX_L_ADD_F16))
__VX_ALL_SPACES(add, bf16, uint16_t, (uint16_t)__vx_loop16(p, v, __VX_L_ADD_BF16), (void)__vx_loop16(p, v, __VX_L_ADD_BF16))
__VX_ALL_SPACES(inc, u32,  uint32_t, (uint32_t)__VX_FPAMO(__VX_F5_INC, 2, p, v),   __VX_FPAMO_RED(__VX_F5_INC, 2, p, v))
__VX_ALL_SPACES(dec, u32,  uint32_t, (uint32_t)__VX_FPAMO(__VX_F5_DEC, 2, p, v),   __VX_FPAMO_RED(__VX_F5_DEC, 2, p, v))

// cas.b16: the 32-bit loop, retrying only while the half still equals cmp.
__VX_ATOM_INL uint16_t __vx_cas_b16(uint16_t* p, uint16_t cmp, uint16_t v) {
  volatile uint32_t* w = (volatile uint32_t*)((uintptr_t)p & ~(uintptr_t)3);
  const uint32_t sh = ((uintptr_t)p & 2) * 8;
  uint32_t old = *w;
  for (;;) {
    const uint16_t o16 = (uint16_t)(old >> sh);
    if (o16 != cmp)
      return o16;
    const uint32_t nw = (old & ~(0xffffu << sh)) | ((uint32_t)v << sh);
    const uint32_t seen = (uint32_t)__VX_AMOCAS(2, w, (uint64_t)(int64_t)(int32_t)old, nw);
    if (seen == old)
      return cmp;
    old = seen;
  }
}

// Vectors: global addresses only. Elements in address order, element 0 in the
// low half of a packed pair. The packed adds reuse __vx_hw_add_{f16x2,bf16x2}.
#define __VX_HW_PACKED(op, ty, T, f5, f3)                                            \
  __VX_ATOM_INL T __vx_hw_##op##_##ty(T* p, T v) { return (T)__VX_FPAMO(f5, f3, p, v); } \
  __VX_ATOM_INL void __vx_hwred_##op##_##ty(T* p, T v) { __VX_FPAMO_RED(f5, f3, p, v); }

__VX_HW_PACKED(add, f32x2,  uint64_t, __VX_F5_FADD2,    3)
__VX_HW_PACKED(min, f16x2,  uint32_t, __VX_F5_FMIN2,    2)
__VX_HW_PACKED(max, f16x2,  uint32_t, __VX_F5_FMAX2,    2)
__VX_HW_PACKED(min, bf16x2, uint32_t, __VX_F5_FMIN2_BF, 2)
__VX_HW_PACKED(max, bf16x2, uint32_t, __VX_F5_FMAX2_BF, 2)

#define __VX_VEC_F32(n)                                                              \
  __VX_ATOM_INL void __vx_atom_add_v##n##_f32(float* p, const float* v, float* old) { \
    for (int i = 0; i < (n); i += 2) {                                               \
      const uint64_t r = __vx_hw_add_f32x2((uint64_t*)(p + i),                       \
                           (uint64_t)__vx_f2u(v[i]) | ((uint64_t)__vx_f2u(v[i + 1]) << 32)); \
      if (old) {                                                                     \
        old[i] = __vx_u2f((uint32_t)r);                                              \
        old[i + 1] = __vx_u2f((uint32_t)(r >> 32));                                  \
      }                                                                              \
    }                                                                                \
  }                                                                                  \
  __VX_ATOM_INL void __vx_red_add_v##n##_f32(float* p, const float* v) {             \
    for (int i = 0; i < (n); i += 2)                                                 \
      __vx_hwred_add_f32x2((uint64_t*)(p + i),                                       \
                           (uint64_t)__vx_f2u(v[i]) | ((uint64_t)__vx_f2u(v[i + 1]) << 32)); \
  }

__VX_VEC_F32(2)
__VX_VEC_F32(4)

// v{2,4,8} of a 16-bit type: one packed op per pair.
#define __VX_VEC16(op, ty, n, pair)                                                  \
  __VX_ATOM_INL void __vx_atom_##op##_v##n##_##ty(uint16_t* p, const uint16_t* v, uint16_t* old) { \
    for (int i = 0; i < (n); i += 2) {                                               \
      const uint32_t r = __vx_hw_##op##_##pair((uint32_t*)(p + i),                   \
                                               (uint32_t)v[i] | ((uint32_t)v[i + 1] << 16)); \
      if (old) {                                                                     \
        old[i] = (uint16_t)r;                                                        \
        old[i + 1] = (uint16_t)(r >> 16);                                            \
      }                                                                              \
    }                                                                                \
  }                                                                                  \
  __VX_ATOM_INL void __vx_red_##op##_v##n##_##ty(uint16_t* p, const uint16_t* v) {   \
    for (int i = 0; i < (n); i += 2)                                                 \
      __vx_hwred_##op##_##pair((uint32_t*)(p + i), (uint32_t)v[i] | ((uint32_t)v[i + 1] << 16)); \
  }

// v{2,4} of a packed type: one packed op per element.
#define __VX_VECX2(op, ty, n)                                                        \
  __VX_ATOM_INL void __vx_atom_##op##_v##n##_##ty(uint32_t* p, const uint32_t* v, uint32_t* old) { \
    for (int i = 0; i < (n); ++i) {                                                  \
      const uint32_t r = __vx_hw_##op##_##ty(p + i, v[i]);                           \
      if (old)                                                                       \
        old[i] = r;                                                                  \
    }                                                                                \
  }                                                                                  \
  __VX_ATOM_INL void __vx_red_##op##_v##n##_##ty(uint32_t* p, const uint32_t* v) {   \
    for (int i = 0; i < (n); ++i)                                                    \
      __vx_hwred_##op##_##ty(p + i, v[i]);                                           \
  }

#define __VX_VEC16_OPS(op, et, pair) \
  __VX_VEC16(op, et, 2, pair) __VX_VEC16(op, et, 4, pair) __VX_VEC16(op, et, 8, pair)
#define __VX_VECX2_OPS(op, ty) __VX_VECX2(op, ty, 2) __VX_VECX2(op, ty, 4)

__VX_VEC16_OPS(add, f16,  f16x2)
__VX_VEC16_OPS(add, bf16, bf16x2)
__VX_VEC16_OPS(min, f16,  f16x2)
__VX_VEC16_OPS(max, f16,  f16x2)
__VX_VEC16_OPS(min, bf16, bf16x2)
__VX_VEC16_OPS(max, bf16, bf16x2)
__VX_VECX2_OPS(add, f16x2)
__VX_VECX2_OPS(add, bf16x2)
__VX_VECX2_OPS(min, f16x2)
__VX_VECX2_OPS(max, f16x2)
__VX_VECX2_OPS(min, bf16x2)
__VX_VECX2_OPS(max, bf16x2)

///////////////////////////////////////////////////////////////////////////////
// Memory semantics: an acquire is the relaxed op, then the scope's fence; a
// release is the fence, then the op. Cluster and gpu scope fence at GPU
// strength, as on sm_90.

#define __VX_ATOM_SEM1(sp, op, sc, ty, T)                                            \
  __VX_ATOM_INL T vx_atom_##sp##op##_relaxed_##sc##_##ty(T* p, T v) {                \
    return __vx_atom_##sp##op##_##ty(p, v);                                          \
  }                                                                                  \
  __VX_ATOM_INL T vx_atom_##sp##op##_acquire_##sc##_##ty(T* p, T v) {                \
    T o = __vx_atom_##sp##op##_##ty(p, v);                                           \
    __VX_ATOM_FENCE_##sc();                                                          \
    return o;                                                                        \
  }                                                                                  \
  __VX_ATOM_INL T vx_atom_##sp##op##_release_##sc##_##ty(T* p, T v) {                \
    __VX_ATOM_FENCE_##sc();                                                          \
    return __vx_atom_##sp##op##_##ty(p, v);                                          \
  }                                                                                  \
  __VX_ATOM_INL T vx_atom_##sp##op##_acq_rel_##sc##_##ty(T* p, T v) {                \
    __VX_ATOM_FENCE_##sc();                                                          \
    T o = __vx_atom_##sp##op##_##ty(p, v);                                           \
    __VX_ATOM_FENCE_##sc();                                                          \
    return o;                                                                        \
  }                                                                                  \
  __VX_ATOM_INL void vx_red_##sp##op##_relaxed_##sc##_##ty(T* p, T v) {              \
    __vx_red_##sp##op##_##ty(p, v);                                                  \
  }                                                                                  \
  __VX_ATOM_INL void vx_red_##sp##op##_release_##sc##_##ty(T* p, T v) {              \
    __VX_ATOM_FENCE_##sc();                                                          \
    __vx_red_##sp##op##_##ty(p, v);                                                  \
  }

#define __VX_ATOM_SEM_SC(sp, op, ty, T) \
  __VX_ATOM_SEM1(sp, op, cta, ty, T) __VX_ATOM_SEM1(sp, op, cluster, ty, T) __VX_ATOM_SEM1(sp, op, gpu, ty, T)
#define __VX_ATOM_SEM(op, ty, T)                                                     \
  __VX_ATOM_SEM_SC(, op, ty, T) __VX_ATOM_SEM_SC(global_, op, ty, T)                 \
  __VX_ATOM_SEM_SC(shared_, op, ty, T) __VX_ATOM_SEM_SC(shared_cluster_, op, ty, T)

__VX_ATOM_SEM(add, f32,    float)
__VX_ATOM_SEM(add, f64,    double)
__VX_ATOM_SEM(add, f16,    uint16_t)
__VX_ATOM_SEM(add, bf16,   uint16_t)
__VX_ATOM_SEM(add, f16x2,  uint32_t)
__VX_ATOM_SEM(add, bf16x2, uint32_t)
__VX_ATOM_SEM(inc, u32,    uint32_t)
__VX_ATOM_SEM(dec, u32,    uint32_t)

#define __VX_CAS_SEM1(sp, sc)                                                        \
  __VX_ATOM_INL uint16_t vx_atom_##sp##cas_relaxed_##sc##_b16(uint16_t* p, uint16_t c, uint16_t v) { \
    return __vx_cas_b16(p, c, v);                                                    \
  }                                                                                  \
  __VX_ATOM_INL uint16_t vx_atom_##sp##cas_acquire_##sc##_b16(uint16_t* p, uint16_t c, uint16_t v) { \
    uint16_t o = __vx_cas_b16(p, c, v);                                              \
    __VX_ATOM_FENCE_##sc();                                                          \
    return o;                                                                        \
  }                                                                                  \
  __VX_ATOM_INL uint16_t vx_atom_##sp##cas_release_##sc##_b16(uint16_t* p, uint16_t c, uint16_t v) { \
    __VX_ATOM_FENCE_##sc();                                                          \
    return __vx_cas_b16(p, c, v);                                                    \
  }                                                                                  \
  __VX_ATOM_INL uint16_t vx_atom_##sp##cas_acq_rel_##sc##_b16(uint16_t* p, uint16_t c, uint16_t v) { \
    __VX_ATOM_FENCE_##sc();                                                          \
    uint16_t o = __vx_cas_b16(p, c, v);                                              \
    __VX_ATOM_FENCE_##sc();                                                          \
    return o;                                                                        \
  }
#define __VX_CAS_SEM(sp) __VX_CAS_SEM1(sp, cta) __VX_CAS_SEM1(sp, cluster) __VX_CAS_SEM1(sp, gpu)
__VX_CAS_SEM()
__VX_CAS_SEM(global_)
__VX_CAS_SEM(shared_)
__VX_CAS_SEM(shared_cluster_)

#define __VX_VEC_SEM1(sp, op, sc, ty, T)                                             \
  __VX_ATOM_INL void vx_atom_##sp##op##_relaxed_##sc##_##ty(T* p, const T* v, T* old) { \
    __vx_atom_##op##_##ty(p, v, old);                                                \
  }                                                                                  \
  __VX_ATOM_INL void vx_atom_##sp##op##_acquire_##sc##_##ty(T* p, const T* v, T* old) { \
    __vx_atom_##op##_##ty(p, v, old);                                                \
    __VX_ATOM_FENCE_##sc();                                                          \
  }                                                                                  \
  __VX_ATOM_INL void vx_atom_##sp##op##_release_##sc##_##ty(T* p, const T* v, T* old) { \
    __VX_ATOM_FENCE_##sc();                                                          \
    __vx_atom_##op##_##ty(p, v, old);                                                \
  }                                                                                  \
  __VX_ATOM_INL void vx_atom_##sp##op##_acq_rel_##sc##_##ty(T* p, const T* v, T* old) { \
    __VX_ATOM_FENCE_##sc();                                                          \
    __vx_atom_##op##_##ty(p, v, old);                                                \
    __VX_ATOM_FENCE_##sc();                                                          \
  }                                                                                  \
  __VX_ATOM_INL void vx_red_##sp##op##_relaxed_##sc##_##ty(T* p, const T* v) {       \
    __vx_red_##op##_##ty(p, v);                                                      \
  }                                                                                  \
  __VX_ATOM_INL void vx_red_##sp##op##_release_##sc##_##ty(T* p, const T* v) {       \
    __VX_ATOM_FENCE_##sc();                                                          \
    __vx_red_##op##_##ty(p, v);                                                      \
  }

#define __VX_VEC_SEM_SC(sp, op, ty, T) \
  __VX_VEC_SEM1(sp, op, cta, ty, T) __VX_VEC_SEM1(sp, op, cluster, ty, T) __VX_VEC_SEM1(sp, op, gpu, ty, T)
#define __VX_VEC_SEM(op, ty, T) __VX_VEC_SEM_SC(, op, ty, T) __VX_VEC_SEM_SC(global_, op, ty, T)

__VX_VEC_SEM(add, v2_f32, float)
__VX_VEC_SEM(add, v4_f32, float)
#define __VX_VEC16_SEM(op, et) \
  __VX_VEC_SEM(op, v2_##et, uint16_t) __VX_VEC_SEM(op, v4_##et, uint16_t) __VX_VEC_SEM(op, v8_##et, uint16_t)
#define __VX_VECX2_SEM(op, ty) __VX_VEC_SEM(op, v2_##ty, uint32_t) __VX_VEC_SEM(op, v4_##ty, uint32_t)
__VX_VEC16_SEM(add, f16)
__VX_VEC16_SEM(add, bf16)
__VX_VEC16_SEM(min, f16)
__VX_VEC16_SEM(max, f16)
__VX_VEC16_SEM(min, bf16)
__VX_VEC16_SEM(max, bf16)
__VX_VECX2_SEM(add, f16x2)
__VX_VECX2_SEM(add, bf16x2)
__VX_VECX2_SEM(min, f16x2)
__VX_VECX2_SEM(max, f16x2)
__VX_VECX2_SEM(min, bf16x2)
__VX_VECX2_SEM(max, bf16x2)

///////////////////////////////////////////////////////////////////////////////
// CUDA atomicAdd/atomicInc/atomicDec/atomicCAS (relaxed, device scope, generic).

__VX_ATOM_INL float    vx_atomic_add_f32(float* p, float v)          { return vx_atom_add_relaxed_gpu_f32(p, v); }
__VX_ATOM_INL double   vx_atomic_add_f64(double* p, double v)        { return vx_atom_add_relaxed_gpu_f64(p, v); }
__VX_ATOM_INL uint16_t vx_atomic_add_f16(uint16_t* p, uint16_t v)    { return vx_atom_add_relaxed_gpu_f16(p, v); }
__VX_ATOM_INL uint16_t vx_atomic_add_bf16(uint16_t* p, uint16_t v)   { return vx_atom_add_relaxed_gpu_bf16(p, v); }
__VX_ATOM_INL uint32_t vx_atomic_add_f16x2(uint32_t* p, uint32_t v)  { return vx_atom_add_relaxed_gpu_f16x2(p, v); }
__VX_ATOM_INL uint32_t vx_atomic_add_bf16x2(uint32_t* p, uint32_t v) { return vx_atom_add_relaxed_gpu_bf16x2(p, v); }
__VX_ATOM_INL uint32_t vx_atomic_inc_u32(uint32_t* p, uint32_t v)    { return vx_atom_inc_relaxed_gpu_u32(p, v); }
__VX_ATOM_INL uint32_t vx_atomic_dec_u32(uint32_t* p, uint32_t v)    { return vx_atom_dec_relaxed_gpu_u32(p, v); }
__VX_ATOM_INL uint16_t vx_atomic_cas_b16(uint16_t* p, uint16_t c, uint16_t v) { return vx_atom_cas_relaxed_gpu_b16(p, c, v); }
__VX_ATOM_INL void vx_atomic_add_v2_f32(float* p, const float* v, float* old) { vx_atom_add_relaxed_gpu_v2_f32(p, v, old); }
__VX_ATOM_INL void vx_atomic_add_v4_f32(float* p, const float* v, float* old) { vx_atom_add_relaxed_gpu_v4_f32(p, v, old); }

#ifdef __cplusplus
}
#endif

#endif // __VX_ATOMIC_H__
