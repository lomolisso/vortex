// vx_cuda_fp.h against independent references: softfloat for f16 and the f32
// intrinsics; for bf16, softfloat f128 with round-to-odd, then f32 with
// round-to-odd, then round-to-nearest-even on the bits; for e4m3, the 256
// values built from the format definition; for rsqrt, exact integer
// comparisons of the rounding midpoints with 1/sqrt(x).

#include <vx_cuda_fp.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

extern "C" {
#include "softfloat_ext.h"
#include <softfloat.h>
}

namespace {

__extension__ typedef unsigned __int128 u128;

// Checks and per-group reporting.

struct Group {
  const char* name;
  uint64_t n = 0, fails = 0;
  std::chrono::steady_clock::time_point t0;
  explicit Group(const char* nm) : name(nm), t0(std::chrono::steady_clock::now()) {}
};

int g_failed_groups = 0;
uint64_t g_checks = 0;

void check(Group& g, bool ok, const char* fmt, ...) __attribute__((format(printf, 3, 4)));
void check(Group& g, bool ok, const char* fmt, ...) {
  ++g.n;
  if (ok) return;
  if (g.fails++ < 10) {
    std::printf("  FAIL %s: ", g.name);
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::printf("\n");
  }
}

void finish(Group& g) {
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g.t0).count();
  g_checks += g.n;
  if (g.fails) {
    ++g_failed_groups;
    std::printf("FAIL %-16s %" PRIu64 " of %" PRIu64 " checks failed (%.2f s)\n", g.name, g.fails, g.n, s);
  } else {
    std::printf("PASS %-16s %" PRIu64 " checks (%.2f s)\n", g.name, g.n, s);
  }
  std::fflush(stdout);
}

uint64_t g_rng = 0x9e3779b97f4a7c15ull;
uint64_t rnd64() {
  uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
uint32_t rnd32() { return (uint32_t)(rnd64() >> 32); }
uint16_t rnd16() { return (uint16_t)(rnd64() >> 48); }

uint32_t fb(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float bf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
uint64_t db(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }
double bd(uint64_t u) { double d; std::memcpy(&d, &u, 8); return d; }

constexpr uint32_t F32_QNAN = 0x7fffffff;
bool nan32(uint32_t u) { return (u & 0x7fffffff) > 0x7f800000; }

// Type views.

enum Kind { F16, BF16 };

template <Kind K> struct KT;
template <> struct KT<F16> { using T = __half; using T2 = __half2; static constexpr const char* name = "f16"; static constexpr uint16_t INF = 0x7c00, ONE = 0x3c00; };
template <> struct KT<BF16> { using T = __nv_bfloat16; using T2 = __nv_bfloat162; static constexpr const char* name = "bf16"; static constexpr uint16_t INF = 0x7f80, ONE = 0x3f80; };

template <Kind K> typename KT<K>::T mkT(uint16_t b) { typename KT<K>::T r; r.__x = b; return r; }
template <Kind K> typename KT<K>::T2 mkT2(uint16_t x, uint16_t y) { return typename KT<K>::T2(mkT<K>(x), mkT<K>(y)); }
template <Kind K> bool isnanK(uint16_t b) { return (b & 0x7fff) > KT<K>::INF; }

const uint16_t kSpecF16[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x7bff, 0xfbff,
                             0x7c00, 0xfc00, 0x7e00, 0x3c00, 0xbc00, 0x7c01, 0xfe00, 0x3555};
const uint16_t kSpecBF16[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x7f7f, 0xff7f,
                              0x7f80, 0xff80, 0x7fc0, 0x3f80, 0xbf80, 0x7f81, 0xffc0, 0x3eab};

template <Kind K> std::vector<uint16_t> specials() {
  if (K == F16) return std::vector<uint16_t>(std::begin(kSpecF16), std::end(kSpecF16));
  return std::vector<uint16_t>(std::begin(kSpecBF16), std::end(kSpecBF16));
}

// f16 reference: softfloat, NaN results mapped to 0x7fff.

float16_t h16(uint16_t b) { float16_t r; r.v = b; return r; }
uint16_t canon16(float16_t r) { return isnanK<F16>(r.v) ? 0x7fff : r.v; }

// bf16 reference: exact value in f128, round-to-odd to f128 and then to f32,
// then round-to-nearest-even on the f32 bits.

uint16_t rne_bf16(uint32_t u) {
  if (nan32(u)) return 0x7fff;
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float128_t q_bf16(uint16_t b) { float32_t f; f.v = (uint32_t)b << 16; return f32_to_f128(f); }
uint16_t bf16_of_q(float128_t q) {
  softfloat_roundingMode = softfloat_round_odd;
  const float32_t f = f128_to_f32(q);
  softfloat_roundingMode = softfloat_round_near_even;
  return rne_bf16(f.v);
}

enum Op { ADD, SUB, MUL, DIV, FMA, SQRT, NEG, ABS, NOPS };
const char* kOpName[] = {"add", "sub", "mul", "div", "fma", "sqrt", "neg", "abs"};

template <Kind K> uint16_t ref_op(Op op, uint16_t a, uint16_t b, uint16_t c);

template <> uint16_t ref_op<F16>(Op op, uint16_t a, uint16_t b, uint16_t c) {
  softfloat_roundingMode = softfloat_round_near_even;
  switch (op) {
  case ADD:  return canon16(f16_add(h16(a), h16(b)));
  case SUB:  return canon16(f16_sub(h16(a), h16(b)));
  case MUL:  return canon16(f16_mul(h16(a), h16(b)));
  case DIV:  return canon16(f16_div(h16(a), h16(b)));
  case FMA:  return canon16(f16_mulAdd(h16(a), h16(b), h16(c)));
  case SQRT: return canon16(f16_sqrt(h16(a)));
  case NEG:  return isnanK<F16>(a) ? 0x7fff : (uint16_t)(a ^ 0x8000);
  case ABS:  return isnanK<F16>(a) ? 0x7fff : (uint16_t)(a & 0x7fff);
  default:   return 0;
  }
}

template <> uint16_t ref_op<BF16>(Op op, uint16_t a, uint16_t b, uint16_t c) {
  if (op == NEG) return isnanK<BF16>(a) ? 0x7fff : (uint16_t)(a ^ 0x8000);
  if (op == ABS) return isnanK<BF16>(a) ? 0x7fff : (uint16_t)(a & 0x7fff);
  const float128_t qa = q_bf16(a), qb = q_bf16(b), qc = q_bf16(c);
  softfloat_roundingMode = softfloat_round_odd;
  float128_t r;
  switch (op) {
  case ADD:  r = f128_add(qa, qb); break;
  case SUB:  r = f128_sub(qa, qb); break;
  case MUL:  r = f128_mul(qa, qb); break;
  case DIV:  r = f128_div(qa, qb); break;
  case FMA:  r = f128_mulAdd(qa, qb, qc); break;
  default:   r = f128_sqrt(qa); break;
  }
  return bf16_of_q(r);
}

template <Kind K> uint16_t dut_op(Op op, uint16_t a, uint16_t b, uint16_t c) {
  const auto A = mkT<K>(a), B = mkT<K>(b), C = mkT<K>(c);
  switch (op) {
  case ADD:  return __hadd(A, B).__x;
  case SUB:  return __hsub(A, B).__x;
  case MUL:  return __hmul(A, B).__x;
  case DIV:  return __hdiv(A, B).__x;
  case FMA:  return __hfma(A, B, C).__x;
  case SQRT: return hsqrt(A).__x;
  case NEG:  return __hneg(A).__x;
  default:   return __habs(A).__x;
  }
}

// Packed: lane x computes (a0, b0, c0), lane y (a1, b1, c1).
template <Kind K> void dut_op2(Op op, const uint16_t* a, const uint16_t* b, const uint16_t* c, uint16_t* r) {
  const auto A = mkT2<K>(a[0], a[1]), B = mkT2<K>(b[0], b[1]), C = mkT2<K>(c[0], c[1]);
  typename KT<K>::T2 R;
  switch (op) {
  case ADD:  R = __hadd2(A, B); break;
  case SUB:  R = __hsub2(A, B); break;
  case MUL:  R = __hmul2(A, B); break;
  case DIV:  R = __h2div(A, B); break;
  case FMA:  R = __hfma2(A, B, C); break;
  case SQRT: R = h2sqrt(A); break;
  case NEG:  R = __hneg2(A); break;
  default:   R = __habs2(A); break;
  }
  r[0] = R.x.__x;
  r[1] = R.y.__x;
}

// e4m3 reference table from the format: bias 7, 3 fraction bits, no infinity,
// S.1111.111 is NaN, everything else finite.

double e4m3_value(uint8_t b) {
  if ((b & 0x7f) == 0x7f) return NAN;
  const int e = (b >> 3) & 15, m = b & 7;
  const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
  return (b & 0x80) ? -v : v;
}

// Round to nearest even over the finite codes, then saturate to +-448.
uint8_t ref_e4m3(double x) {
  if (x != x) return 0x7f;
  const uint8_t s = std::signbit(x) ? 0x80 : 0;
  const double a = std::fabs(x);
  if (a >= 448.0) return s | 0x7e;
  int lo = 0, hi = 0x7e;
  while (lo < hi) {  // largest code with value <= a
    const int mid = (lo + hi + 1) / 2;
    if (e4m3_value((uint8_t)mid) <= a) lo = mid; else hi = mid - 1;
  }
  const double v0 = e4m3_value((uint8_t)lo);
  if (v0 == a) return s | (uint8_t)lo;
  const double mid = (v0 + e4m3_value((uint8_t)(lo + 1))) / 2;
  int c;
  if (a < mid) c = lo;
  else if (a > mid) c = lo + 1;
  else c = (lo & 1) ? lo + 1 : lo;
  return s | (uint8_t)c;
}

// Exact sign of m*m*x - 1 for m and x positive: integers scaled by powers of 2.
int cmp_m2x_1(long double m, float x) {
  const int em = std::ilogb(m) - 30;
  const uint64_t M = (uint64_t)std::ldexp(m, -em);
  if (std::ldexp((long double)M, em) != m) { std::printf("cmp_m2x_1: m not exact\n"); std::abort(); }
  const int ex = std::ilogb(x) - 23;
  const uint64_t X = (uint64_t)std::ldexp((double)x, -ex);
  const u128 P = (u128)M * M * X;  // < 2^(31+31+24)
  const int e = 2 * em + ex;
  if (e >= 0) return (P == 1 && e == 0) ? 0 : 1;
  if (-e >= 127) return -1;
  const u128 one = (u128)1 << -e;
  return P < one ? -1 : (P > one ? 1 : 0);
}

// Correctly rounded 1/sqrt(x) for positive finite x: walk from a long double
// estimate until the two midpoints around y bracket 1/sqrt(x).
uint32_t ref_rsqrt(float x) {
  float y = (float)(1.0L / std::sqrt((long double)x));
  for (int it = 0; it < 8; ++it) {
    const uint32_t yu = fb(y);
    const long double up = ((long double)y + bf(yu + 1)) / 2, dn = ((long double)y + bf(yu - 1)) / 2;
    if (cmp_m2x_1(up, x) < 0) { y = bf(yu + 1); continue; }
    if (cmp_m2x_1(dn, x) > 0) { y = bf(yu - 1); continue; }
    return yu;
  }
  std::printf("ref_rsqrt: no convergence\n");
  std::abort();
}

///////////////////////////////////////////////////////////////////////////////

// CUDA Programming Guide, "Alignment Requirements" table (LP64 for long).
#define VEC_LAYOUT(N, T, A1, A2, A3, A4)                                                                     \
  static_assert(sizeof(N##1) == sizeof(T) && alignof(N##1) == A1, #N "1");                                   \
  static_assert(sizeof(N##2) == 2 * sizeof(T) && alignof(N##2) == A2, #N "2");                               \
  static_assert(sizeof(N##3) == 3 * sizeof(T) && alignof(N##3) == A3, #N "3");                               \
  static_assert(sizeof(N##4) == 4 * sizeof(T) && alignof(N##4) == A4, #N "4");                               \
  static_assert(std::is_trivial<N##1>::value && std::is_trivial<N##4>::value, #N " trivial");                \
  static_assert(std::is_same<decltype(N##4::w), T>::value, #N " element");
VEC_LAYOUT(char, signed char, 1, 2, 1, 4)
VEC_LAYOUT(uchar, unsigned char, 1, 2, 1, 4)
VEC_LAYOUT(short, short, 2, 4, 2, 8)
VEC_LAYOUT(ushort, unsigned short, 2, 4, 2, 8)
VEC_LAYOUT(int, int, 4, 8, 4, 16)
VEC_LAYOUT(uint, unsigned int, 4, 8, 4, 16)
VEC_LAYOUT(long, long, 8, 16, 8, 16)
VEC_LAYOUT(ulong, unsigned long, 8, 16, 8, 16)
VEC_LAYOUT(longlong, long long, 8, 16, 8, 16)
VEC_LAYOUT(ulonglong, unsigned long long, 8, 16, 8, 16)
VEC_LAYOUT(float, float, 4, 8, 4, 16)
VEC_LAYOUT(double, double, 8, 16, 8, 16)
#undef VEC_LAYOUT

void test_layout() {
  Group g("layout");
  static_assert(sizeof(__half) == 2 && alignof(__half) == 2, "__half");
  static_assert(sizeof(__half2) == 4 && alignof(__half2) == 4, "__half2");
  static_assert(sizeof(__nv_bfloat16) == 2 && alignof(__nv_bfloat16) == 2, "__nv_bfloat16");
  static_assert(sizeof(__nv_bfloat162) == 4 && alignof(__nv_bfloat162) == 4, "__nv_bfloat162");
  static_assert(sizeof(__nv_fp8_e4m3) == 1 && alignof(__nv_fp8_e4m3) == 1, "__nv_fp8_e4m3");
  static_assert(sizeof(float2) == 8 && alignof(float2) == 8, "float2");
  static_assert(std::is_trivial<__half>::value && std::is_trivial<__half2>::value, "trivial");
  static_assert(std::is_trivial<__nv_bfloat16>::value && std::is_trivial<__nv_bfloat162>::value, "trivial");
  static_assert(std::is_trivial<__nv_fp8_e4m3>::value, "trivial");
  static_assert(!std::is_convertible<float, __nv_fp8_e4m3>::value, "fp8 constructors are explicit");
  static_assert(!std::is_convertible<__nv_fp8_e4m3, float>::value, "fp8 conversions are explicit");
  static_assert(std::is_same<half, __half>::value && std::is_same<nv_bfloat162, __nv_bfloat162>::value, "typedefs");

  const __half2 h2(__ushort_as_half(0x1234), __ushort_as_half(0xabcd));
  uint32_t w;
  std::memcpy(&w, &h2, 4);
  check(g, w == 0xabcd1234u, "__half2 x is the low half: %#x", w);
  const __nv_bfloat162 b2(__ushort_as_bfloat16(0x1111), __ushort_as_bfloat16(0x2222));
  std::memcpy(&w, &b2, 4);
  check(g, w == 0x22221111u, "__nv_bfloat162 x is the low half: %#x", w);
  const __half_raw hr = __half(1.0f);
  check(g, hr.x == 0x3c00, "__half_raw %#x", hr.x);
  __half h = hr;
  h = __half_raw{0xc000};
  check(g, __half_as_ushort(h) == 0xc000 && __half_as_short(h) == (short)0xc000, "raw assignment");
  check(g, __half_as_ushort(__short_as_half(-1)) == 0xffff, "__short_as_half");
  check(g, __bfloat16_as_ushort(__short_as_bfloat16(-2)) == 0xfffe, "__short_as_bfloat16");
  const __half2_raw r2 = __half2(__half(1.0f), __half(2.0f));
  check(g, r2.x == 0x3c00 && r2.y == 0x4000, "__half2_raw");
  const __nv_bfloat162_raw rb2 = __nv_bfloat162(__nv_bfloat16(1.0f), __nv_bfloat16(-2.0f));
  check(g, rb2.x == 0x3f80 && rb2.y == 0xc000, "__nv_bfloat162_raw");
  check(g, __float_as_uint(1.0f) == 0x3f800000u && __float_as_int(-2.0f) == (int)0xc0000000u, "float bits");
  check(g, __uint_as_float(0x40400000u) == 3.0f && __int_as_float(0x40800000) == 4.0f, "bits float");
  check(g, __double_as_longlong(1.0) == 0x3ff0000000000000ll && __longlong_as_double(0x4000000000000000ll) == 2.0,
        "double bits");
  check(g, -__int_as_float(0x7f800000) == -INFINITY, "-inf");
  const int4 i4 = make_int4(1, -2, 3, -4);
  int words[4];
  std::memcpy(words, &i4, 16);
  check(g, words[0] == 1 && words[1] == -2 && words[2] == 3 && words[3] == -4, "int4 member order");
  const uint4 u4 = make_uint4(5, 6, 7, 8);
  check(g, u4.x == 5 && u4.y == 6 && u4.z == 7 && u4.w == 8, "make_uint4");
  const float4 f4 = make_float4(1.5f, 2.5f, 3.5f, 4.5f);
  const float2 f2 = make_float2(f4.z, f4.w);
  check(g, f4.x == 1.5f && f4.y == 2.5f && f2.x == 3.5f && f2.y == 4.5f, "make_float4/make_float2");
  const double2 d2 = make_double2(0.25, -8.0);
  const longlong2 l2 = make_longlong2(-1, 1ll << 40);
  const ulonglong2 ul2 = make_ulonglong2(~0ull, 3);
  check(g, d2.x == 0.25 && d2.y == -8.0 && l2.x == -1 && l2.y == (1ll << 40) && ul2.x == ~0ull && ul2.y == 3, "2-wide 64-bit");
  const int2 i2 = make_int2(-7, 9);
  const uint2 u2 = make_uint2(1u << 31, 2);
  const int3 i3 = make_int3(1, 2, 3);
  const uint3 u3 = make_uint3(4, 5, 6);
  const float3 f3 = make_float3(0.5f, 1.0f, 2.0f);
  check(g, i2.x == -7 && i2.y == 9 && u2.x == (1u << 31) && u2.y == 2 && i3.z == 3 && u3.y == 5 && f3.z == 2.0f, "int2/uint2/3-wide");
  const char4 c4 = make_char4(-1, 2, -3, 4);
  const uchar4 uc4 = make_uchar4(250, 1, 2, 3);
  const short2 s2 = make_short2(-300, 300);
  const ushort2 us2 = make_ushort2(65535, 1);
  const uchar2 uc2 = make_uchar2(7, 8);
  const char2 c2 = make_char2(-8, 7);
  check(g, c4.x == -1 && c4.w == 4 && uc4.x == 250 && s2.x == -300 && us2.x == 65535 && uc2.y == 8 && c2.x == -8, "8/16-bit vectors");
  finish(g);
}

template <Kind K> void test_widen() {
  Group g(K == F16 ? "f16-widen" : "bf16-widen");
  for (uint32_t i = 0; i < 65536; ++i) {
    const uint16_t b = (uint16_t)i;
    const auto v = mkT<K>(b);
    uint32_t want;
    if (K == F16) {
      // The linked softfloat returns its default NaN here; the rule (sign and
      // payload kept, quieted) is checked directly.
      want = isnanK<F16>(b) ? (((uint32_t)(b & 0x8000) << 16) | 0x7fc00000u | ((uint32_t)(b & 0x3ff) << 13))
                            : f16_to_f32(h16(b)).v;
    } else {
      want = (uint32_t)b << 16;
      if (nan32(want)) want |= 0x400000;
    }
    const float f = K == F16 ? __half2float(mkT<F16>(b)) : __bfloat162float(mkT<BF16>(b));
    check(g, fb(f) == want, "%04x widen: got %08x want %08x", b, fb(f), want);
    check(g, fb((float)v) == want, "%04x operator float: got %08x", b, fb((float)v));
    const uint16_t back = isnanK<K>(b) ? 0x7fff : b;
    const typename KT<K>::T n1(f), n2((double)f);
    check(g, n1.__x == back && n2.__x == back, "%04x round trip: %04x %04x", b, n1.__x, n2.__x);
  }
  finish(g);
}

template <Kind K> uint16_t ref_from_f32(uint32_t u) {
  if (K == F16) {
    softfloat_roundingMode = softfloat_round_near_even;
    float32_t f; f.v = u;
    return canon16(f32_to_f16(f));
  }
  return rne_bf16(u);
}
template <Kind K> uint16_t ref_from_f64(uint64_t u) {
  float64_t d; d.v = u;
  if (K == F16) {
    softfloat_roundingMode = softfloat_round_near_even;
    return canon16(f64_to_f16(d));
  }
  softfloat_roundingMode = softfloat_round_odd;
  const float32_t f = f64_to_f32(d);
  softfloat_roundingMode = softfloat_round_near_even;
  return rne_bf16(f.v);
}

template <Kind K> uint16_t dut_from_f32(float f) {
  if (K == F16) return __float2half(f).__x;
  return __float2bfloat16(f).__x;
}

template <Kind K> void test_narrow() {
  Group g(K == F16 ? "f32-to-f16" : "f32-to-bf16");
  using T = typename KT<K>::T;
  auto one = [&](uint32_t u) {
    const float f = bf(u);
    const uint16_t want = ref_from_f32<K>(u);
    const uint16_t a = dut_from_f32<K>(f);
    const uint16_t b = K == F16 ? __float2half_rn(f).__x : __float2bfloat16_rn(f).__x;
    const uint16_t c = T(f).__x;
    check(g, a == want && b == want && c == want, "f32 %08x: got %04x/%04x/%04x want %04x", u, a, b, c, want);
  };
  // Every narrow value and every midpoint between neighbours, +-4 f32 ulps.
  for (uint32_t i = 0; i < KT<K>::INF; ++i) {
    const double v0 = (float)mkT<K>((uint16_t)i), v1 = (float)mkT<K>((uint16_t)(i + 1));
    const double v1x = (i + 1 == KT<K>::INF) ? 2 * v0 - (float)mkT<K>((uint16_t)(i - 1)) : v1;
    for (double c : {v0, (v0 + v1x) / 2}) {
      const uint32_t cu = fb((float)c);
      for (int d = -4; d <= 4; ++d) {
        const uint32_t u = cu + d;
        if ((int64_t)cu + d < 0 || u > 0x7f800000) continue;
        one(u);
        one(u | 0x80000000);
      }
    }
  }
  for (uint32_t u : {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7f800001u, 0x7fbfffffu, 0x00000001u, 0x7f7fffffu})
    one(u);
  for (int i = 0; i < 10000000; ++i) one(rnd32());

  // f64 inputs, including doubles that round to an f16/bf16 midpoint in f32.
  auto oned = [&](uint64_t u) {
    const double d = bd(u);
    const uint16_t want = ref_from_f64<K>(u);
    const uint16_t a = K == F16 ? __double2half(d).__x : __double2bfloat16(d).__x;
    const uint16_t b = T(d).__x;
    check(g, a == want && b == want, "f64 %016" PRIx64 ": got %04x/%04x want %04x", u, a, b, want);
  };
  for (uint32_t i = 0; i + 1 < KT<K>::INF; ++i) {
    const double mid = ((double)(float)mkT<K>((uint16_t)i) + (double)(float)mkT<K>((uint16_t)(i + 1))) / 2;
    const uint64_t mu = db(mid);
    for (int d = -4; d <= 4; ++d) {
      if (mu == 0 && d < 0) continue;
      oned(mu + d);
      oned((mu + d) | 0x8000000000000000ull);
    }
    oned(db(mid * (1 + 0x1p-40)));
    oned(db(mid * (1 - 0x1p-40)));
  }
  for (uint64_t u : {0x7ff0000000000000ull, 0xfff0000000000000ull, 0x7ff8000000000000ull, 0x0000000000000001ull,
                     0x7fefffffffffffffull, 0x47efffffe0000000ull, 0x47efffffefffffffull, 0x47effffff0000000ull})
    oned(u);
  for (int i = 0; i < 1000000; ++i) {
    // Exponents from far below to far above the narrow range.
    const uint64_t e = 1023 - 160 + rnd64() % 320;
    oned((rnd64() & 0x800fffffffffffffull) | (e << 52));
    oned(rnd64());
  }
  finish(g);
}

template <Kind K> uint16_t ref_from_int(int64_t v, bool is_unsigned) {
  const uint64_t uv = (uint64_t)v;
  if (K == F16) {
    softfloat_roundingMode = softfloat_round_near_even;
    return is_unsigned ? ui64_to_f16(uv).v : i64_to_f16(v).v;
  }
  softfloat_roundingMode = softfloat_round_odd;
  const float32_t f = is_unsigned ? ui64_to_f32(uv) : i64_to_f32(v);
  softfloat_roundingMode = softfloat_round_near_even;
  return rne_bf16(f.v);
}

template <Kind K> void test_int() {
  Group g(K == F16 ? "int-to-f16" : "int-to-bf16");
  using T = typename KT<K>::T;
  auto one = [&](int64_t v) {
    const uint16_t ws = ref_from_int<K>(v, false), wu = ref_from_int<K>(v, true);
    check(g, T((long long)v).__x == ws && T((long)v).__x == ws, "%" PRId64 ": %04x want %04x", v, T((long long)v).__x, ws);
    check(g, T((unsigned long long)v).__x == wu && T((unsigned long)v).__x == wu, "%" PRIu64 "u: %04x want %04x",
          (uint64_t)v, T((unsigned long long)v).__x, wu);
    const int32_t i32 = (int32_t)v;
    check(g, T(i32).__x == ref_from_int<K>(i32, false), "int %d", i32);
    check(g, T((uint32_t)v).__x == ref_from_int<K>((uint32_t)v, true), "unsigned %u", (uint32_t)v);
    check(g, T((short)v).__x == ref_from_int<K>((short)v, false), "short %d", (short)v);
    check(g, T((unsigned short)v).__x == ref_from_int<K>((unsigned short)v, true), "ushort %u", (unsigned short)v);
  };
  for (int64_t v = -70000; v <= 70000; ++v) one(v);
  for (int i = 0; i < 200000; ++i) { const uint64_t r = rnd64(); one((int64_t)(r >> (r & 63))); }
  for (int64_t v : {INT64_MIN, INT64_MAX, (int64_t)-1, (int64_t)0x7fffffff, (int64_t)0x80000000ll}) one(v);
  finish(g);
}

void test_e4m3() {
  Group g("e4m3");
  for (uint32_t b = 0; b < 256; ++b) {
    __nv_fp8_e4m3 e;
    e.__x = (__nv_fp8_storage_t)b;
    const double v = e4m3_value((uint8_t)b);
    const float f = (float)e;
    if (v != v) {
      check(g, fb(f) == 0x7fffe000u, "%02x NaN to f32: %08x", b, fb(f));
      check(g, ((__half)e).__x == 0x7fff, "%02x NaN to f16", b);
    } else {
      check(g, (double)f == v && std::signbit(f) == std::signbit(v), "%02x to f32: %a want %a", b, f, v);
      softfloat_roundingMode = softfloat_round_near_even;
      float64_t d; d.v = db(v);
      check(g, ((__half)e).__x == f64_to_f16(d).v, "%02x to f16", b);
      check(g, __nv_fp8_e4m3(f).__x == b, "%02x round trip", b);
    }
  }
  auto one = [&](uint32_t u) {
    const float f = bf(u);
    const uint8_t want = ref_e4m3((double)f);
    const uint8_t a = __nv_fp8_e4m3(f).__x, c = __nv_fp8_e4m3((double)f).__x;
    check(g, a == want && c == want, "f32 %08x (%a): got %02x/%02x want %02x", u, f, a, c, want);
  };
  for (uint32_t b = 0; b < 0x7e; ++b) {
    const float v0 = (float)e4m3_value((uint8_t)b), v1 = (float)e4m3_value((uint8_t)(b + 1));
    for (float c : {v0, (v0 + v1) / 2}) {
      for (int d = -4; d <= 4; ++d) {
        if (fb(c) == 0 && d < 0) continue;
        one(fb(c) + d);
        one((fb(c) + d) | 0x80000000u);
      }
    }
  }
  for (float c : {448.0f, 464.0f, 480.0f, 512.0f, 3.4028235e38f}) {
    for (int d = -4; d <= 4; ++d) { one(fb(c) + d); one((fb(c) + d) | 0x80000000u); }
  }
  for (uint32_t u : {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7f800001u, 0x00000001u, 0x80000000u})
    one(u);
  check(g, __nv_fp8_e4m3(INFINITY).__x == 0x7e && __nv_fp8_e4m3(-INFINITY).__x == 0xfe, "inf saturates");
  check(g, __nv_fp8_e4m3(NAN).__x == 0x7f && __nv_fp8_e4m3(-NAN).__x == 0x7f, "NaN is 0x7f");
  check(g, __nv_fp8_e4m3(464.0f).__x == 0x7e && __nv_fp8_e4m3(1e30).__x == 0x7e, "satfinite");
  for (int i = 0; i < 1000000; ++i) {
    one(rnd32());
    one(fb((float)((int32_t)rnd32() / 4194304.0)));  // about [-512, 512]
  }
  // Doubles next to e4m3 midpoints: a float cast would round them onto the midpoint.
  for (uint32_t b = 0; b < 0x7e; ++b) {
    const double mid = (e4m3_value((uint8_t)b) + e4m3_value((uint8_t)(b + 1))) / 2;
    for (int d = -2; d <= 2; ++d) {
      const double x = bd(db(mid) + d);
      const uint8_t want = ref_e4m3(x);
      check(g, __nv_fp8_e4m3(x).__x == want, "f64 %a: got %02x want %02x", x, __nv_fp8_e4m3(x).__x, want);
    }
  }
  for (uint32_t i = 0; i < 65536; ++i) {
    const double hv = (float)mkT<F16>((uint16_t)i), bv = (float)mkT<BF16>((uint16_t)i);
    const uint8_t wh = ref_e4m3(hv), wb = ref_e4m3(bv);
    check(g, __nv_fp8_e4m3(mkT<F16>((uint16_t)i)).__x == wh, "from f16 %04x", i);
    check(g, __nv_fp8_e4m3(mkT<BF16>((uint16_t)i)).__x == wb, "from bf16 %04x", i);
  }
  for (int v = -1000; v <= 1000; ++v) {
    const uint8_t want = ref_e4m3((double)v);
    check(g, __nv_fp8_e4m3(v).__x == want && __nv_fp8_e4m3((long long)v).__x == want &&
                 __nv_fp8_e4m3((short)v).__x == want && __nv_fp8_e4m3((long)v).__x == want, "int %d", v);
    if (v >= 0)
      check(g, __nv_fp8_e4m3((unsigned)v).__x == want && __nv_fp8_e4m3((unsigned long long)v).__x == want &&
                   __nv_fp8_e4m3((unsigned short)v).__x == want && __nv_fp8_e4m3((unsigned long)v).__x == want,
            "unsigned %d", v);
  }
  finish(g);
}

template <Kind K> void test_arith() {
  const std::vector<uint16_t> sp = specials<K>();
  for (int o = 0; o < NOPS; ++o) {
    const Op op = (Op)o;
    char name[32];
    std::snprintf(name, sizeof(name), "%s-%s", KT<K>::name, kOpName[o]);
    Group g(name);
    auto one = [&](uint16_t a, uint16_t b, uint16_t c) {
      const uint16_t want = ref_op<K>(op, a, b, c), got = dut_op<K>(op, a, b, c);
      check(g, got == want, "%s(%04x, %04x, %04x): got %04x want %04x", kOpName[o], a, b, c, got, want);
    };
    auto two = [&](const uint16_t* a, const uint16_t* b, const uint16_t* c) {
      uint16_t r[2];
      dut_op2<K>(op, a, b, c, r);
      for (int l = 0; l < 2; ++l) {
        const uint16_t want = ref_op<K>(op, a[l], b[l], c[l]);
        check(g, r[l] == want, "%s2 lane %d (%04x, %04x, %04x): got %04x want %04x", kOpName[o], l, a[l], b[l], c[l],
              r[l], want);
      }
    };
    const bool ternary = op == FMA;
    for (uint16_t a : sp)
      for (uint16_t b : sp)
        for (size_t k = 0; k < (ternary ? sp.size() : 1); ++k) {
          const uint16_t c = sp[k];
          one(a, b, c);
          const uint16_t A[2] = {a, b}, B[2] = {b, a}, C[2] = {c, a};
          two(A, B, C);
        }
    for (int i = 0; i < 1000000; ++i) one(rnd16(), rnd16(), rnd16());
    for (int i = 0; i < 500000; ++i) {
      const uint16_t A[2] = {rnd16(), rnd16()}, B[2] = {rnd16(), rnd16()}, C[2] = {rnd16(), rnd16()};
      two(A, B, C);
    }
    finish(g);
  }
}

// fma with the addend far from the product: exponent distance >= 12 (f16) or
// >= 9 (bf16) in either direction, up to the whole range.
template <Kind K> void test_fma_far() {
  Group g(K == F16 ? "f16-fma-far" : "bf16-fma-far");
  const int MB = K == F16 ? 10 : 7, EB = K == F16 ? 5 : 8, BIAS = K == F16 ? 15 : 127;
  auto ilog = [&](uint16_t b) {
    const int e = (b >> MB) & ((1 << EB) - 1);
    if (e) return e - BIAS;
    const uint32_t m = b & ((1u << MB) - 1);
    return (31 - __builtin_clz(m)) - MB + 1 - BIAS;
  };
  auto finite_nz = [&]() {
    for (;;) {
      const uint16_t b = rnd16();
      if ((b & 0x7fff) != 0 && (b & 0x7fff) < KT<K>::INF) return b;
    }
  };
  uint64_t naive_diff = 0;
  int done = 0;
  while (done < 100000) {
    const uint16_t a = finite_nz(), b = finite_nz(), c = finite_nz();
    const int dist = ilog(a) + ilog(b) - ilog(c);
    if (std::abs(dist) < (K == F16 ? 12 : 9)) continue;
    ++done;
    const uint16_t want = ref_op<K>(FMA, a, b, c), got = dut_op<K>(FMA, a, b, c);
    check(g, got == want, "fma(%04x, %04x, %04x): got %04x want %04x", a, b, c, got, want);
    const double p = (double)(float)mkT<K>(a) * (double)(float)mkT<K>(b), s = p + (double)(float)mkT<K>(c);
    const uint16_t naive = K == F16 ? vx_cuda_fp::from_f64<vx_cuda_fp::fmt_f16>(s) : vx_cuda_fp::from_f64<vx_cuda_fp::fmt_bf16>(s);
    naive_diff += naive != want;
  }
  if (K == BF16) {
    // 3 * 129 = 387 is the midpoint of 386 and 388; -2^-100 decides it.
    const uint16_t a = __float2bfloat16(3.0f).__x, b = __float2bfloat16(129.0f).__x, c = __float2bfloat16(-0x1p-100f).__x;
    const uint16_t got = dut_op<K>(FMA, a, b, c), want = ref_op<K>(FMA, a, b, c);
    check(g, got == want && want == __float2bfloat16(386.0f).__x, "fma(3, 129, -2^-100): got %04x want %04x", got, want);
    check(g, naive_diff > 0, "rounding the f64 sum directly never differed (the vectors do not discriminate)");
  }
  std::printf("  %s: rounding the f64 sum directly would differ in %" PRIu64 " of %d triples\n", g.name, naive_diff, done);
  finish(g);
}

template <Kind K> void test_cmp() {
  Group g(K == F16 ? "f16-cmp" : "bf16-cmp");
  using T2 = typename KT<K>::T2;
  auto val = [](uint16_t b) -> double {
    if (K == F16) return bd(f16_to_f64(h16(b)).v);
    return (double)bf((uint32_t)b << 16);
  };
  auto ref_max = [&](uint16_t a, uint16_t b, bool is_max, bool nan_mode) -> uint16_t {
    const bool an = isnanK<K>(a), bn = isnanK<K>(b);
    if (an && bn) return 0x7fff;
    if (nan_mode && (an || bn)) return 0x7fff;
    if (an) return b;
    if (bn) return a;
    const double va = val(a), vb = val(b);
    if (va == 0 && vb == 0) {  // +0 > -0
      const bool a_pos = !(a & 0x8000);
      return (is_max == a_pos) ? a : b;
    }
    if (is_max) return va >= vb ? a : b;
    return va <= vb ? a : b;
  };
  auto one = [&](uint16_t a, uint16_t b) {
    const auto A = mkT<K>(a), B = mkT<K>(b);
    const double va = val(a), vb = val(b);
    const bool un = isnanK<K>(a) || isnanK<K>(b);
    const bool r[12] = {va == vb, !un && va != vb, va < vb, va <= vb, va > vb, va >= vb,
                        un || va == vb, un || va != vb, un || va < vb, un || va <= vb, un || va > vb, un || va >= vb};
    const bool d[12] = {__heq(A, B), __hne(A, B), __hlt(A, B), __hle(A, B), __hgt(A, B), __hge(A, B),
                        __hequ(A, B), __hneu(A, B), __hltu(A, B), __hleu(A, B), __hgtu(A, B), __hgeu(A, B)};
    const bool o[6] = {A == B, A != B, A < B, A <= B, A > B, A >= B};
    const bool ro[6] = {r[0], r[7], r[2], r[3], r[4], r[5]};
    for (int i = 0; i < 12; ++i) check(g, d[i] == r[i], "cmp %d (%04x, %04x): %d", i, a, b, d[i]);
    for (int i = 0; i < 6; ++i) check(g, o[i] == ro[i], "operator %d (%04x, %04x): %d", i, a, b, o[i]);
    // Packed: lane x (a, b), lane y (b, a).
    const T2 P = mkT2<K>(a, b), Q = mkT2<K>(b, a);
    const T2 h[12] = {__heq2(P, Q), __hne2(P, Q), __hlt2(P, Q), __hle2(P, Q), __hgt2(P, Q), __hge2(P, Q),
                      __hequ2(P, Q), __hneu2(P, Q), __hltu2(P, Q), __hleu2(P, Q), __hgtu2(P, Q), __hgeu2(P, Q)};
    const bool hb[12] = {__hbeq2(P, Q), __hbne2(P, Q), __hblt2(P, Q), __hble2(P, Q), __hbgt2(P, Q), __hbge2(P, Q),
                         __hbequ2(P, Q), __hbneu2(P, Q), __hbltu2(P, Q), __hbleu2(P, Q), __hbgtu2(P, Q), __hbgeu2(P, Q)};
    const bool rs[12] = {va == vb, !un && va != vb, vb < va, vb <= va, vb > va, vb >= va,
                         un || va == vb, un || va != vb, un || vb < va, un || vb <= va, un || vb > va, un || vb >= va};
    for (int i = 0; i < 12; ++i) {
      const uint16_t wx = r[i] ? KT<K>::ONE : 0, wy = rs[i] ? KT<K>::ONE : 0;
      check(g, h[i].x.__x == wx && h[i].y.__x == wy, "packed cmp %d (%04x, %04x): %04x %04x", i, a, b, h[i].x.__x, h[i].y.__x);
      check(g, hb[i] == (r[i] && rs[i]), "packed bool cmp %d (%04x, %04x)", i, a, b);
    }
    const bool po[6] = {P == Q, P != Q, P < Q, P <= Q, P > Q, P >= Q};
    const bool rpo[6] = {r[0] && rs[0], r[7] && rs[7], r[2] && rs[2], r[3] && rs[3], r[4] && rs[4], r[5] && rs[5]};
    for (int i = 0; i < 6; ++i) check(g, po[i] == rpo[i], "packed operator %d (%04x, %04x)", i, a, b);
    check(g, __hmax(A, B).__x == ref_max(a, b, true, false), "max(%04x, %04x) = %04x", a, b, __hmax(A, B).__x);
    check(g, __hmin(A, B).__x == ref_max(a, b, false, false), "min(%04x, %04x) = %04x", a, b, __hmin(A, B).__x);
    check(g, __hmax_nan(A, B).__x == ref_max(a, b, true, true), "max_nan(%04x, %04x)", a, b);
    check(g, __hmin_nan(A, B).__x == ref_max(a, b, false, true), "min_nan(%04x, %04x)", a, b);
    const T2 m[4] = {__hmax2(P, Q), __hmin2(P, Q), __hmax2_nan(P, Q), __hmin2_nan(P, Q)};
    for (int i = 0; i < 4; ++i) {
      const bool is_max = !(i & 1), nm = i >= 2;
      check(g, m[i].x.__x == ref_max(a, b, is_max, nm) && m[i].y.__x == ref_max(b, a, is_max, nm),
            "packed minmax %d (%04x, %04x)", i, a, b);
    }
    check(g, __hisnan(A) == isnanK<K>(a), "isnan %04x", a);
    const int wi = ((a & 0x7fff) == KT<K>::INF) ? ((a & 0x8000) ? -1 : 1) : 0;
    check(g, __hisinf(A) == wi, "isinf %04x", a);
    const T2 n2 = __hisnan2(P);
    check(g, n2.x.__x == (isnanK<K>(a) ? KT<K>::ONE : 0) && n2.y.__x == (isnanK<K>(b) ? KT<K>::ONE : 0), "isnan2");
  };
  const std::vector<uint16_t> sp = specials<K>();
  for (uint16_t a : sp)
    for (uint16_t b : sp) one(a, b);
  for (int i = 0; i < 200000; ++i) one(rnd16(), rnd16());
  for (int i = 0; i < 20000; ++i) { const uint16_t a = rnd16(); one(a, a); one(a, (uint16_t)(a ^ 1)); }
  finish(g);
}

template <Kind K> void test_operators() {
  Group g(K == F16 ? "f16-operators" : "bf16-operators");
  using T = typename KT<K>::T;
  using T2 = typename KT<K>::T2;
  for (int i = 0; i < 100000; ++i) {
    const uint16_t a = rnd16(), b = rnd16(), c = rnd16(), d = rnd16();
    const T A = mkT<K>(a), B = mkT<K>(b);
    check(g, (A + B).__x == __hadd(A, B).__x && (A - B).__x == __hsub(A, B).__x && (A * B).__x == __hmul(A, B).__x &&
                 (A / B).__x == __hdiv(A, B).__x && (-A).__x == __hneg(A).__x && (+A).__x == a,
          "binary operators %04x %04x", a, b);
    T s = A;
    s += B; check(g, s.__x == __hadd(A, B).__x, "+=");
    s = A; s -= B; check(g, s.__x == __hsub(A, B).__x, "-=");
    s = A; s *= B; check(g, s.__x == __hmul(A, B).__x, "*=");
    s = A; s /= B; check(g, s.__x == __hdiv(A, B).__x, "/=");
    const T one_ = mkT<K>(KT<K>::ONE);
    s = A; const T pre = ++s; check(g, pre.__x == __hadd(A, one_).__x && s.__x == pre.__x, "++");
    s = A; const T post = s++; check(g, post.__x == a && s.__x == __hadd(A, one_).__x, "post ++");
    s = A; --s; check(g, s.__x == __hsub(A, one_).__x, "--");
    s = A; s--; check(g, s.__x == __hsub(A, one_).__x, "post --");
    const T2 P = mkT2<K>(a, b), Q = mkT2<K>(c, d);
    auto eq2 = [](T2 x, T2 y) { return x.x.__x == y.x.__x && x.y.__x == y.y.__x; };
    check(g, eq2(P + Q, __hadd2(P, Q)) && eq2(P - Q, __hsub2(P, Q)) && eq2(P * Q, __hmul2(P, Q)) &&
                 eq2(P / Q, __h2div(P, Q)) && eq2(-P, __hneg2(P)) && eq2(+P, P),
          "packed operators");
    T2 t = P; t += Q; check(g, eq2(t, __hadd2(P, Q)), "packed +=");
    t = P; t -= Q; check(g, eq2(t, __hsub2(P, Q)), "packed -=");
    t = P; t *= Q; check(g, eq2(t, __hmul2(P, Q)), "packed *=");
    t = P; t /= Q; check(g, eq2(t, __h2div(P, Q)), "packed /=");
  }
  // Pair conversions against the scalar ones.
  for (int i = 0; i < 100000; ++i) {
    const float x = bf(rnd32()), y = bf(rnd32());
    const uint16_t wx = dut_from_f32<K>(x), wy = dut_from_f32<K>(y);
    T2 p, q, r;
    float2 back;
    float lo, hi;
    T l, h;
    if (K == F16) {
      const __half2 p_ = __floats2half2_rn(x, y), q_ = __float22half2_rn(make_float2(x, y)), r_ = __float2half2_rn(x);
      std::memcpy(&p, &p_, 4); std::memcpy(&q, &q_, 4); std::memcpy(&r, &r_, 4);
      const float2 b_ = __half22float2(p_); back = b_;
      lo = __low2float(p_); hi = __high2float(p_);
      const __half l_ = __low2half(p_), h_ = __high2half(p_);
      std::memcpy(&l, &l_, 2); std::memcpy(&h, &h_, 2);
      const __half2 s_ = __halves2half2(h_, l_), d_ = __half2half2(h_);
      check(g, s_.x.__x == wy && s_.y.__x == wx && d_.x.__x == wy && d_.y.__x == wy, "halves2half2/half2half2");
    } else {
      const __nv_bfloat162 p_ = __floats2bfloat162_rn(x, y), q_ = __float22bfloat162_rn(make_float2(x, y)),
                           r_ = __float2bfloat162_rn(x);
      std::memcpy(&p, &p_, 4); std::memcpy(&q, &q_, 4); std::memcpy(&r, &r_, 4);
      const float2 b_ = __bfloat1622float2(p_); back = b_;
      lo = __low2float(p_); hi = __high2float(p_);
      const __nv_bfloat16 l_ = __low2bfloat16(p_), h_ = __high2bfloat16(p_);
      std::memcpy(&l, &l_, 2); std::memcpy(&h, &h_, 2);
      const __nv_bfloat162 s_ = __halves2bfloat162(h_, l_), d_ = __bfloat162bfloat162(h_);
      check(g, s_.x.__x == wy && s_.y.__x == wx && d_.x.__x == wy && d_.y.__x == wy, "halves2bfloat162/bfloat162bfloat162");
    }
    check(g, p.x.__x == wx && p.y.__x == wy && q.x.__x == wx && q.y.__x == wy && r.x.__x == wx && r.y.__x == wx,
          "pair from floats %08x %08x", fb(x), fb(y));
    check(g, fb(back.x) == fb((float)mkT<K>(wx)) && fb(back.y) == fb((float)mkT<K>(wy)) && fb(lo) == fb(back.x) &&
                 fb(hi) == fb(back.y) && l.__x == wx && h.__x == wy,
          "pair to floats");
  }
  finish(g);
}

void test_rsqrt() {
  Group g("frsqrt_rn");
  uint64_t fixed = 0, naive32 = 0;
  for (uint32_t u = 0x3f800000; u < 0x40800000; ++u) {  // [1, 4)
    const float x = bf(u);
    const uint32_t want = ref_rsqrt(x), got = fb(__frsqrt_rn(x));
    check(g, got == want, "rsqrt(%a): got %08x want %08x", x, got, want);
    fixed += fb((float)(1.0 / std::sqrt((double)x))) != want;
    naive32 += fb(1.0f / std::sqrt(x)) != want;
  }
  std::printf("  frsqrt_rn: of the 2^24 inputs in [1, 4), the f64 estimate alone is wrong for %" PRIu64
              ", 1.0f / sqrtf(x) for %" PRIu64 "\n", fixed, naive32);
  for (int i = 0; i < 1000000; ++i) {
    const uint32_t u = 1 + rnd32() % 0x7f7fffffu;
    const uint32_t want = ref_rsqrt(bf(u)), got = fb(__frsqrt_rn(bf(u)));
    check(g, got == want, "rsqrt(%08x): got %08x want %08x", u, got, want);
    check(g, fb(rsqrtf(bf(u))) == got, "rsqrtf(%08x)", u);
  }
  for (int i = 0; i < 100000; ++i) {  // subnormals
    const uint32_t u = 1 + rnd32() % 0x7fffff;
    const uint32_t want = ref_rsqrt(bf(u)), got = fb(__frsqrt_rn(bf(u)));
    check(g, got == want, "rsqrt(%08x) subnormal: got %08x want %08x", u, got, want);
  }
  const struct { uint32_t x, want; } sp[] = {
    {0x00000000, 0x7f800000}, {0x80000000, 0xff800000}, {0x7f800000, 0x00000000}, {0xff800000, F32_QNAN},
    {0xbf800000, F32_QNAN},   {0x80000001, F32_QNAN},   {0x7fc00000, F32_QNAN},   {0xffc00001, F32_QNAN},
    {0x7f800001, F32_QNAN},   {0x3f800000, 0x3f800000}, {0x40800000, 0x3f000000}, {0x3e800000, 0x40000000},
  };
  for (const auto& s : sp) check(g, fb(__frsqrt_rn(bf(s.x))) == s.want, "rsqrt(%08x) = %08x", s.x, fb(__frsqrt_rn(bf(s.x))));
  for (uint32_t u : {0x00000001u, 0x007fffffu, 0x00800000u, 0x7f7fffffu, 0x3f800001u, 0x407fffffu})
    check(g, fb(__frsqrt_rn(bf(u))) == ref_rsqrt(bf(u)), "rsqrt(%08x) edge", u);
  finish(g);
}

void test_rn() {
  Group g("rn-intrinsics");
  auto canon = [](float32_t r) { return nan32(r.v) ? F32_QNAN : r.v; };
  softfloat_roundingMode = softfloat_round_near_even;
  float32_t one;
  one.v = 0x3f800000;
  auto run = [&](uint32_t x, uint32_t y) {
    float32_t X, Y;
    X.v = x; Y.v = y;
    const uint32_t r1 = fb(__frcp_rn(bf(x))), w1 = canon(f32_div(one, X));
    const uint32_t r2 = fb(__fdiv_rn(bf(x), bf(y))), w2 = canon(f32_div(X, Y));
    const uint32_t r3 = fb(__fsqrt_rn(bf(x))), w3 = canon(f32_sqrt(X));
    check(g, r1 == w1, "frcp_rn(%08x): got %08x want %08x", x, r1, w1);
    check(g, r2 == w2, "fdiv_rn(%08x, %08x): got %08x want %08x", x, y, r2, w2);
    check(g, r3 == w3, "fsqrt_rn(%08x): got %08x want %08x", x, r3, w3);
  };
  const uint32_t sp[] = {0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
                         0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc00000, 0x3f800000, 0xbf800000, 0x40490fdb};
  for (uint32_t x : sp)
    for (uint32_t y : sp) run(x, y);
  for (int i = 0; i < 1000000; ++i) run(rnd32(), rnd32());
  finish(g);
}

// Error of got against the exact value e, in ulps of e (an infinite got
// counts as 2^128).
long double ulp_err(float got, long double e) {
  const long double g = std::isinf(got) ? std::ldexp(1.0L, 128) : (long double)got;
  int ex = e == 0 ? -149 : std::ilogb(e);
  if (ex < -126) ex = -126;
  return std::fabs(g - e) / std::ldexp(1.0L, ex - 23);
}

void test_approx() {
  Group g("approx");
  long double worst_exp = 0, worst_ex2 = 0;
  for (int i = 0; i < 1000000; ++i) {
    const float x = -104.0f + (float)(rnd64() % 193000000ull) * 1e-6f;  // [-104, 89]
    const float got = __expf(x);
    const long double ex = std::exp((long double)x);
    if (ex > std::ldexp(1.0L, 128)) {
      check(g, std::isinf(got), "__expf(%a) = %a, not inf", x, got);
    } else {
      const long double err = ulp_err(got, ex);
      worst_exp = std::max(worst_exp, err);
      check(g, err <= 1.0L, "__expf(%a) = %a: %.3Lf ulp", x, got, err);
      check(g, err <= 2 + std::floor(std::fabs(1.173 * x)), "__expf(%a) beyond the CUDA bound", x);
    }
    const float y = -126.0f + (float)(rnd64() % 254000000ull) * 1e-6f;  // [-126, 128)
    const float got2 = __vx_ex2_approx(y);
    const long double err2 = ulp_err(got2, std::exp2((long double)y));
    worst_ex2 = std::max(worst_ex2, err2);
    check(g, err2 <= 1.0L, "ex2(%a) = %a: %.3Lf ulp", y, got2, err2);
  }
  for (int i = 0; i < 1000000; ++i) {  // random bits over the whole input range
    const float x = bf(rnd32());
    if (x != x) continue;
    const long double top = std::ldexp(1.0L, 128);
    const float got = __expf(x);
    const long double e = std::exp((long double)x);
    if (e > top) check(g, std::isinf(got), "__expf(%a) = %a, not inf", x, got);
    else check(g, ulp_err(got, e) <= 1.0L, "__expf(%a) = %a", x, got);
    const float got2 = __vx_ex2_approx(x);
    const long double e2 = std::exp2((long double)x);
    const bool sub_in = (fb(x) & 0x7f800000) == 0;
    if (sub_in) check(g, got2 == 1.0f, "ex2 of subnormal %a = %a", x, got2);
    else if (e2 < std::ldexp(1.0L, -126)) check(g, fb(got2) == 0, "ex2(%a) flushes: %a", x, got2);
    else if (e2 > top) check(g, std::isinf(got2), "ex2(%a) = %a, not inf", x, got2);
    else check(g, ulp_err(got2, e2) <= 1.0L, "ex2(%a) = %a", x, got2);
  }
  for (int k = -126; k < 128; ++k)
    check(g, __vx_ex2_approx((float)k) == std::ldexp(1.0f, k), "ex2(%d) exact", k);
  const struct { uint32_t x, e, e2; } sp[] = {
    {0xff800000, 0x00000000, 0x00000000},  // -inf
    {0x80000000, 0x3f800000, 0x3f800000},  // -0
    {0x00000000, 0x3f800000, 0x3f800000},  // +0
    {0x7f800000, 0x7f800000, 0x7f800000},  // +inf
    {0x7fc00000, F32_QNAN, F32_QNAN},      // NaN
    {0xffc00001, F32_QNAN, F32_QNAN},
    {0x00000001, 0x3f800000, 0x3f800000},  // subnormal input
    {0x43000000, 0x7f800000, 0x7f800000},  // 128
    {0xc2fc0000, 0x00000000, 0x00800000},  // -126
    {0xc2fe0000, 0x00000000, 0x00000000},  // -127: 2^-127 flushes
    {0x42b20000, 0x7f800000, 0x6c000000},  // 89: e^89 overflows
  };
  for (const auto& s : sp) {
    check(g, fb(__expf(bf(s.x))) == s.e, "__expf(%08x) = %08x", s.x, fb(__expf(bf(s.x))));
    check(g, fb(__vx_ex2_approx(bf(s.x))) == s.e2, "ex2(%08x) = %08x", s.x, fb(__vx_ex2_approx(bf(s.x))));
  }
  // __fdividef: 2^126 < |y| < 2^128 gives 0 (NaN for infinite x), else the IEEE quotient.
  softfloat_roundingMode = softfloat_round_near_even;
  for (int i = 0; i < 1000000; ++i) {
    const uint32_t x = rnd32();
    uint32_t y = rnd32();
    if (i & 1) y = (y & 0x807fffff) | ((0xfd + (rnd32() % 2)) << 23);  // |y| in [2^126, 2^128)
    const uint32_t ay = y & 0x7fffffff;
    uint32_t want;
    if (ay > 0x7e800000 && ay < 0x7f800000) {
      want = ((x & 0x7fffffff) >= 0x7f800000) ? F32_QNAN : ((x ^ y) & 0x80000000);
    } else {
      float32_t X, Y;
      X.v = x; Y.v = y;
      const float32_t r = f32_div(X, Y);
      want = nan32(r.v) ? F32_QNAN : r.v;
    }
    const uint32_t got = fb(__fdividef(bf(x), bf(y)));
    check(g, got == want, "fdividef(%08x, %08x): got %08x want %08x", x, y, got, want);
  }
  check(g, fb(__fdividef(1.0f, 0x1p126f)) == fb(0x1p-126f), "fdividef at y = 2^126");
  check(g, fb(__fdividef(INFINITY, 0x1.8p126f)) == F32_QNAN, "fdividef(inf, 1.5*2^126)");
  check(g, fb(__fdividef(-3.0f, 0x1.8p127f)) == 0x80000000u, "fdividef(-3, 1.5*2^127)");
  std::printf("  approx: worst __expf %.4Lf ulp, worst ex2 %.4Lf ulp on [-104, 89] / [-126, 128)\n", worst_exp, worst_ex2);
  finish(g);
}

// Every f32 pattern (about 3 min): run with -x.
void test_exhaustive() {
  Group g("f32-exhaustive");
  for (uint64_t i = 0; i <= 0xffffffffull; ++i) {
    const uint32_t u = (uint32_t)i;
    const float f = bf(u);
    const uint16_t h = __float2half(f).__x, b = __float2bfloat16(f).__x;
    const uint8_t e = __nv_fp8_e4m3(f).__x;
    check(g, h == ref_from_f32<F16>(u), "f16 of %08x: %04x", u, h);
    check(g, b == ref_from_f32<BF16>(u), "bf16 of %08x: %04x", u, b);
    check(g, e == ref_e4m3((double)f), "e4m3 of %08x: %02x", u, e);
  }
  finish(g);
}

} // namespace

int main(int argc, char** argv) {
  const auto t0 = std::chrono::steady_clock::now();
  softfloat_roundingMode = softfloat_round_near_even;
  softfloat_detectTininess = softfloat_tininess_afterRounding;
  test_layout();
  test_widen<F16>();
  test_widen<BF16>();
  test_narrow<F16>();
  test_narrow<BF16>();
  test_int<F16>();
  test_int<BF16>();
  test_e4m3();
  test_arith<F16>();
  test_arith<BF16>();
  test_fma_far<F16>();
  test_fma_far<BF16>();
  test_cmp<F16>();
  test_cmp<BF16>();
  test_operators<F16>();
  test_operators<BF16>();
  test_rsqrt();
  test_rn();
  test_approx();
  if (argc > 1 && std::strcmp(argv[1], "-x") == 0) test_exhaustive();
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("PERF: cuda_fp %" PRIu64 " checks in %.1f s\n", g_checks, s);
  if (g_failed_groups) {
    std::printf("FAILED: %d groups\n", g_failed_groups);
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
