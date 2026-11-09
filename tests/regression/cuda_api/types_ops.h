#ifndef _CUDA_API_TYPES_OPS_H_
#define _CUDA_API_TYPES_OPS_H_

// Every number-type conversion and op of vx_cuda_fp.h on input i of a fixed
// vector: the kernel and the host run this same code, and their words must
// be equal bit for bit.

#include <stdint.h>
#include <string.h>
#include <vx_cuda_fp.h>

#ifdef __VORTEX__
#define T_INL inline __attribute__((always_inline))
#else
#define T_INL inline
#endif

namespace types_ops {

T_INL uint32_t fbits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
T_INL float bitsf(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
T_INL uint32_t hb(__half h) { return __half_as_ushort(h); }
T_INL uint32_t bb(__nv_bfloat16 h) { return __bfloat16_as_ushort(h); }
T_INL __half hraw(uint32_t b) { return __ushort_as_half((unsigned short)b); }
T_INL __nv_bfloat16 braw(uint32_t b) { return __ushort_as_bfloat16((unsigned short)b); }
T_INL uint32_t h2b(__half2 v) { return hb(v.x) | (hb(v.y) << 16); }
T_INL uint32_t b2b(__nv_bfloat162 v) { return bb(v.x) | (bb(v.y) << 16); }
T_INL __half2 h2raw(uint32_t b) { __half2 v; v.x = hraw(b & 0xffff); v.y = hraw(b >> 16); return v; }
T_INL __nv_bfloat162 b2raw(uint32_t b) { __nv_bfloat162 v; v.x = braw(b & 0xffff); v.y = braw(b >> 16); return v; }

// Inputs: specials first, then patterns spread over every exponent.
T_INL uint32_t input(uint32_t i) {
  static const uint32_t sp[32] = {
    0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00001u, 0x7F800001u, 0x00000001u,
    0x807FFFFFu, 0x00800000u, 0x7F7FFFFFu, 0x3F800000u, 0xBF800000u, 0x477FE000u, 0x477FF000u, 0x47800000u,
    0x33800000u, 0x33000000u, 0x38800000u, 0x387FC000u, 0x43E00000u, 0x43E80000u, 0x43F00000u, 0x3B800000u,
    0x7C007C00u, 0xFC00FC00u, 0x7E017C01u, 0x3C003C00u, 0x00010001u, 0x83FF7BFFu, 0x7F807F80u, 0x7FC1FF81u};
  if (i < 32)
    return sp[i];
  uint32_t h = i * 0x9E3779B1u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  h *= 0x297A2D39u;
  h ^= h >> 15;
  return h;
}

#define T_OPS 140

// The words of input i (a, and its partners b, c in the same vector).
T_INL void eval(uint32_t a, uint32_t b, uint32_t c, uint32_t* o) {
  const float fa = bitsf(a), fb = bitsf(b), fc = bitsf(c);
  const __half ha = hraw(a), hb_ = hraw(b), hc = hraw(c);
  const __nv_bfloat16 ba = braw(a), bb_ = braw(b), bc = braw(c);
  const __half2 h2a = h2raw(a), h2b_ = h2raw(b), h2c = h2raw(c);
  const __nv_bfloat162 b2a = b2raw(a), b2b_ = b2raw(b), b2c = b2raw(c);
  uint32_t k = 0;

  // conversions
  o[k++] = hb(__float2half(fa));
  o[k++] = hb(__float2half_rn(fa));
  o[k++] = fbits(__half2float(ha));
  o[k++] = hb(__half(fa));
  o[k++] = hb(__half((double)fa * 1.0000001));
  o[k++] = fbits((float)ha);
  o[k++] = bb(__float2bfloat16(fa));
  o[k++] = bb(__float2bfloat16_rn(fa));
  o[k++] = fbits(__bfloat162float(ba));
  o[k++] = bb(__nv_bfloat16(fa));
  o[k++] = bb(__nv_bfloat16((double)fa * 1.0000001));
  o[k++] = fbits((float)ba);
  {
    __nv_fp8_e4m3 e(fa);
    o[k++] = e.__x;
    __nv_fp8_e4m3 r;
    r.__x = (__nv_fp8_storage_t)(a & 0xff);
    o[k++] = fbits((float)r);
  }
  {
    __half_raw r;
    r.x = (unsigned short)(b & 0xffff);
    o[k++] = hb(__half(r));
    __nv_bfloat16_raw s;
    s.x = (unsigned short)(b & 0xffff);
    o[k++] = bb(__nv_bfloat16(s));
  }
  o[k++] = h2b(__floats2half2_rn(fa, fb));
  {
    float2 f;
    f.x = fa;
    f.y = fc;
    o[k++] = h2b(__float22half2_rn(f));
    const float2 g = __half22float2(h2a);
    o[k++] = fbits(g.x);
    o[k++] = fbits(g.y);
  }
  o[k++] = hb(__low2half(h2a));
  o[k++] = hb(__high2half(h2a));
  o[k++] = h2b(__halves2half2(ha, hb_));
  o[k++] = h2b(__half2half2(hc));
  o[k++] = fbits(__low2float(h2b_));
  o[k++] = fbits(__high2float(h2b_));
  o[k++] = b2b(__floats2bfloat162_rn(fa, fb));
  {
    float2 f;
    f.x = fb;
    f.y = fa;
    o[k++] = b2b(__float22bfloat162_rn(f));
    const float2 g = __bfloat1622float2(b2a);
    o[k++] = fbits(g.x);
    o[k++] = fbits(g.y);
  }
  o[k++] = bb(__low2bfloat16(b2a));
  o[k++] = bb(__high2bfloat16(b2a));
  o[k++] = b2b(__halves2bfloat162(ba, bb_));
  o[k++] = b2b(__bfloat162bfloat162(bc));
  o[k++] = __half_as_ushort(ha) | ((uint32_t)__bfloat16_as_ushort(bb_) << 16);

  // f16 scalar
  o[k++] = hb(__hadd(ha, hb_));
  o[k++] = hb(__hsub(ha, hb_));
  o[k++] = hb(__hmul(ha, hb_));
  o[k++] = hb(__hdiv(ha, hb_));
  o[k++] = hb(__hfma(ha, hb_, hc));
  o[k++] = hb(__hneg(ha));
  o[k++] = hb(__habs(ha));
  o[k++] = hb(__hmax(ha, hb_));
  o[k++] = hb(__hmin(ha, hb_));
  o[k++] = hb(__hmax_nan(ha, hb_));
  o[k++] = hb(__hmin_nan(ha, hb_));
  o[k++] = hb(hsqrt(ha));
  // f16 packed
  o[k++] = h2b(__hadd2(h2a, h2b_));
  o[k++] = h2b(__hsub2(h2a, h2b_));
  o[k++] = h2b(__hmul2(h2a, h2b_));
  o[k++] = h2b(__h2div(h2a, h2b_));
  o[k++] = h2b(__hfma2(h2a, h2b_, h2c));
  o[k++] = h2b(__hneg2(h2a));
  o[k++] = h2b(__habs2(h2a));
  o[k++] = h2b(__hmax2(h2a, h2b_));
  o[k++] = h2b(__hmin2(h2a, h2b_));
  o[k++] = h2b(__hmax2_nan(h2a, h2b_));
  o[k++] = h2b(__hmin2_nan(h2a, h2b_));
  o[k++] = h2b(h2sqrt(h2a));
  // f16 comparisons
  o[k++] = (uint32_t)__heq(ha, hb_) | ((uint32_t)__hne(ha, hb_) << 1) | ((uint32_t)__hlt(ha, hb_) << 2)
         | ((uint32_t)__hle(ha, hb_) << 3) | ((uint32_t)__hgt(ha, hb_) << 4) | ((uint32_t)__hge(ha, hb_) << 5)
         | ((uint32_t)__hisnan(ha) << 6) | ((uint32_t)(__hisinf(ha) != 0) << 7);
  // f16 operators
  o[k++] = hb(ha + hb_);
  o[k++] = hb(ha - hb_);
  o[k++] = hb(ha * hb_);
  o[k++] = hb(ha / hb_);
  o[k++] = hb(-ha);
  {
    __half t = ha;
    t += hb_;
    t *= hc;
    o[k++] = hb(t);
    t = ha;
    t -= hb_;
    t /= hc;
    o[k++] = hb(t);
  }
  o[k++] = (uint32_t)(ha == hb_) | ((uint32_t)(ha != hb_) << 1) | ((uint32_t)(ha < hb_) << 2)
         | ((uint32_t)(ha <= hb_) << 3) | ((uint32_t)(ha > hb_) << 4) | ((uint32_t)(ha >= hb_) << 5);
  o[k++] = h2b(h2a + h2b_);
  o[k++] = h2b(h2a - h2b_);
  o[k++] = h2b(h2a * h2b_);
  o[k++] = h2b(h2a / h2b_);
  o[k++] = h2b(-h2a);
  {
    __half2 t = h2a;
    t += h2b_;
    t *= h2c;
    o[k++] = h2b(t);
  }

  // bf16 scalar
  o[k++] = bb(__hadd(ba, bb_));
  o[k++] = bb(__hsub(ba, bb_));
  o[k++] = bb(__hmul(ba, bb_));
  o[k++] = bb(__hdiv(ba, bb_));
  o[k++] = bb(__hfma(ba, bb_, bc));
  o[k++] = bb(__hneg(ba));
  o[k++] = bb(__habs(ba));
  o[k++] = bb(__hmax(ba, bb_));
  o[k++] = bb(__hmin(ba, bb_));
  o[k++] = bb(__hmax_nan(ba, bb_));
  o[k++] = bb(__hmin_nan(ba, bb_));
  o[k++] = bb(hsqrt(ba));
  // bf16 packed
  o[k++] = b2b(__hadd2(b2a, b2b_));
  o[k++] = b2b(__hsub2(b2a, b2b_));
  o[k++] = b2b(__hmul2(b2a, b2b_));
  o[k++] = b2b(__h2div(b2a, b2b_));
  o[k++] = b2b(__hfma2(b2a, b2b_, b2c));
  o[k++] = b2b(__hneg2(b2a));
  o[k++] = b2b(__habs2(b2a));
  o[k++] = b2b(__hmax2(b2a, b2b_));
  o[k++] = b2b(__hmin2(b2a, b2b_));
  o[k++] = b2b(__hmax2_nan(b2a, b2b_));
  o[k++] = b2b(__hmin2_nan(b2a, b2b_));
  o[k++] = b2b(h2sqrt(b2a));
  // bf16 comparisons
  o[k++] = (uint32_t)__heq(ba, bb_) | ((uint32_t)__hne(ba, bb_) << 1) | ((uint32_t)__hlt(ba, bb_) << 2)
         | ((uint32_t)__hle(ba, bb_) << 3) | ((uint32_t)__hgt(ba, bb_) << 4) | ((uint32_t)__hge(ba, bb_) << 5)
         | ((uint32_t)__hisnan(ba) << 6) | ((uint32_t)(__hisinf(ba) != 0) << 7);
  // bf16 operators
  o[k++] = bb(ba + bb_);
  o[k++] = bb(ba - bb_);
  o[k++] = bb(ba * bb_);
  o[k++] = bb(ba / bb_);
  o[k++] = bb(-ba);
  {
    __nv_bfloat16 t = ba;
    t += bb_;
    t *= bc;
    o[k++] = bb(t);
    t = ba;
    t -= bb_;
    t /= bc;
    o[k++] = bb(t);
  }
  o[k++] = (uint32_t)(ba == bb_) | ((uint32_t)(ba != bb_) << 1) | ((uint32_t)(ba < bb_) << 2)
         | ((uint32_t)(ba <= bb_) << 3) | ((uint32_t)(ba > bb_) << 4) | ((uint32_t)(ba >= bb_) << 5);
  o[k++] = b2b(b2a + b2b_);
  o[k++] = b2b(b2a - b2b_);
  o[k++] = b2b(b2a * b2b_);
  o[k++] = b2b(b2a / b2b_);
  o[k++] = b2b(-b2a);
  {
    __nv_bfloat162 t = b2a;
    t += b2b_;
    t *= b2c;
    o[k++] = b2b(t);
  }

  // math intrinsics
  o[k++] = fbits(__expf(fa));
  o[k++] = fbits(__vx_ex2_approx(fa));
  o[k++] = fbits(__frsqrt_rn(fa));
  o[k++] = fbits(__frcp_rn(fa));
  o[k++] = fbits(__fdividef(fa, fb));
  o[k++] = fbits(__fsqrt_rn(fa));
  o[k++] = fbits(__fdiv_rn(fa, fb));
  o[k++] = fbits(__expf(fa * 0x1p-24f));
  o[k++] = fbits(__vx_ex2_approx(fb * 0x1p-25f));
  o[k++] = fbits(__frsqrt_rn(fc));

  while (k < T_OPS)
    o[k++] = 0;
}

T_INL void eval_input(uint32_t i, uint32_t* o) {
  eval(input(i), input((i * 7 + 3) % 512), input((i * 13 + 5) % 512), o);
}

} // namespace types_ops

#endif
