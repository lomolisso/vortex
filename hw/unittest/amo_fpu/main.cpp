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

// VX_amo_fpu testbench: every float op and width, checked bit for bit against
// amo_ref::apply under the last-level cache's rules. A new operand pair enters every
// cycle, so each result must leave exactly VX_CFG_FPAMO_LATENCY cycles later.
// Operands: per-format edge values (zeros, subnormals, extremes, infinities,
// NaNs, ones) crossed with each other, then random pairs biased toward close
// exponents, cancellation and rounding ties, then pairs at the bottom of the
// exponent range, whose sums land in and around the subnormal range.

#include "vl_simulator.h"
#include "VVX_amo_fpu_top.h"
#include "VX_config.h"
#include "amo_ref.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <string>
#include <vector>

namespace ar = vortex::amo_ref;

enum Fmt { F16, BF16, F32, F64 };

struct OpForm {
  uint32_t op;
  uint32_t width;
  Fmt      fmt;    // element format
  int      lanes;  // elements per operand
  const char* name;
};

static const OpForm FORMS[] = {
  {ar::OP_FADD,     ar::W_W, F32,  1, "fadd.w (f32)"},
  {ar::OP_FADD,     ar::W_D, F64,  1, "fadd.d (f64)"},
  {ar::OP_FADD2,    ar::W_W, F16,  2, "fadd2.w (f16x2)"},
  {ar::OP_FADD2,    ar::W_D, F32,  2, "fadd2.d (f32x2)"},
  {ar::OP_FADD2_BF, ar::W_W, BF16, 2, "fadd2.bf.w (bf16x2)"},
  {ar::OP_FMIN2,    ar::W_W, F16,  2, "fmin2 (f16x2)"},
  {ar::OP_FMAX2,    ar::W_W, F16,  2, "fmax2 (f16x2)"},
  {ar::OP_FMIN2_BF, ar::W_W, BF16, 2, "fmin2.bf (bf16x2)"},
  {ar::OP_FMAX2_BF, ar::W_W, BF16, 2, "fmax2.bf (bf16x2)"},
};

static uint32_t fmt_bits(Fmt f) {
  return (f == F64) ? 64 : (f == F32) ? 32 : 16;
}

static uint32_t mant_bits(Fmt f) {
  return (f == F64) ? 52 : (f == F32) ? 23 : (f == F16) ? 10 : 7;
}

static const std::vector<uint64_t>& edges(Fmt f) {
  static const std::vector<uint64_t> e16 = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0200, 0x0400, 0x8400, 0x0401,
    0x3bff, 0x3c00, 0xbc00, 0x3c01, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x7c01,
    0xfe00, 0x7fff, 0x1400, 0x5640, 0x4000, 0xc000};
  static const std::vector<uint64_t> ebf = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0040, 0x0080, 0x8080, 0x0081,
    0x3f7f, 0x3f80, 0xbf80, 0x3f81, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0x7f81,
    0xffc0, 0x7fff, 0x3b80, 0x4b00, 0x4000, 0xc000};
  static const std::vector<uint64_t> e32 = {
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00400000,
    0x00800000, 0x80800000, 0x00800001, 0x3f7fffff, 0x3f800000, 0xbf800000, 0x3f800001,
    0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc00000,
    0x7fffffff, 0x33800000, 0x4b800000, 0x40000000, 0xc0000000, 0x00ffffff, 0x80800001,
    0x7fa00123, 0xffc01234, 0xff800001};
  static const std::vector<uint64_t> e64 = {
    0x0000000000000000ull, 0x8000000000000000ull, 0x0000000000000001ull, 0x8000000000000001ull,
    0x000fffffffffffffull, 0x800fffffffffffffull, 0x0010000000000000ull, 0x8010000000000000ull,
    0x3fefffffffffffffull, 0x3ff0000000000000ull, 0xbff0000000000000ull, 0x3ff0000000000001ull,
    0x7fefffffffffffffull, 0xffefffffffffffffull, 0x7ff0000000000000ull, 0xfff0000000000000ull,
    0x7ff8000000000000ull, 0x7ff0000000000001ull, 0xfff8000000000000ull, 0x7fffffffffffffffull,
    0x3ca0000000000000ull, 0x4340000000000000ull, 0x4000000000000000ull, 0xc000000000000000ull,
    0x7ff4000000000123ull, 0xfff0000000000abcull, 0x7ff800000000beefull, 0xfffc000000001234ull};
  switch (f) {
  case F16:  return e16;
  case BF16: return ebf;
  case F32:  return e32;
  default:   return e64;
  }
}

struct Vec {
  uint32_t op = 0, width = 0;
  uint64_t old = 0, rhs = 0;
  int form = -1;
};

static uint64_t mask_bits(uint32_t n) {
  return (n >= 64) ? ~0ull : ((1ull << n) - 1);
}

class Bench {
public:
  Bench() {
    auto* t = sim_.operator->();
    t->reset = 1;
    for (int i = 0; i < 4; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
    t->eval();
    latency_ = t->cfg_latency;
  }

  uint32_t latency() const {
    return latency_;
  }

  uint32_t op_width() {
    return sim_->cfg_amo_op_width;
  }

  // One cycle: present v, check the result of the operand pair LATENCY cycles back.
  void cycle(const Vec& v) {
    auto* t = sim_.operator->();
    t->op = static_cast<uint8_t>(v.op);
    t->width = v.width;
    t->old_word = v.old;
    t->rhs = v.rhs;
    t->eval();
    inflight_.push_back(v);
    if (inflight_.size() > latency_) {
      Vec p = inflight_.front();
      inflight_.pop_front();
      if (p.form >= 0) {
        this->check(p, t->new_word);
      }
    }
    tick_ = sim_.step(tick_, 2);
  }

  void drain() {
    for (uint32_t i = 0; i <= latency_; ++i) {
      this->cycle(Vec{});
    }
  }

  uint64_t checks = 0, errors = 0;
  uint64_t per_form[sizeof(FORMS) / sizeof(FORMS[0])] = {};

private:
  void check(const Vec& p, uint64_t got) {
    uint64_t exp = ar::apply(p.op, p.width, p.old, p.rhs, ar::SITE_LLC);
    ++checks;
    ++per_form[p.form];
    if (got != exp) {
      if (errors < 40) {
        std::printf("FAIL %s: old=0x%016" PRIx64 " rhs=0x%016" PRIx64 " -> got 0x%016" PRIx64
                    " expected 0x%016" PRIx64 "\n",
                    FORMS[p.form].name, p.old, p.rhs, got, exp);
      }
      ++errors;
    }
  }

  vl_simulator<VVX_amo_fpu_top> sim_;
  uint64_t tick_ = 0;
  uint32_t latency_ = 0;
  std::deque<Vec> inflight_;
};

static uint64_t pack(const OpForm& f, const uint64_t* e) {
  uint32_t n = fmt_bits(f.fmt);
  uint64_t v = 0;
  for (int l = 0; l < f.lanes; ++l) {
    v |= (e[l] & mask_bits(n)) << (l * n);
  }
  return v;
}

// A value near x: same or shifted exponent, a few low bits flipped, maybe negated.
static uint64_t near_value(std::mt19937_64& rng, Fmt f, uint64_t x) {
  uint32_t n = fmt_bits(f);
  uint32_t mant = mant_bits(f);
  uint64_t sign = 1ull << (n - 1);
  uint64_t r = x;
  switch (rng() % 4) {
  case 0: r ^= sign; r ^= rng() & 0x7; break;                     // cancellation
  case 1: r += (static_cast<uint64_t>(rng() % (mant + 3)) << mant); break; // exponent apart
  case 2: r ^= (rng() & 1) ? 1 : 0; r ^= (rng() & 1) ? sign : 0; break;
  default: r = x - (static_cast<uint64_t>(rng() % (mant + 3)) << mant); break;
  }
  return r & mask_bits(n);
}

// A subnormal or one of the two lowest normal binades, either sign.
static uint64_t tiny_value(std::mt19937_64& rng, Fmt f) {
  uint32_t n = fmt_bits(f);
  uint32_t mant = mant_bits(f);
  uint64_t sign = (rng() & 1) ? (1ull << (n - 1)) : 0;
  return sign | ((rng() % 3) << mant) | (rng() & mask_bits(mant));
}

int main(int argc, char** argv) {
  uint32_t seed = 1;
  uint32_t rounds = 200000;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--seed=", 7) == 0) {
      seed = static_cast<uint32_t>(std::strtoul(argv[i] + 7, nullptr, 0));
    } else if (std::strncmp(argv[i], "--rounds=", 9) == 0) {
      rounds = static_cast<uint32_t>(std::strtoul(argv[i] + 9, nullptr, 0));
    }
  }

  Bench b;
  std::printf("amo_fpu: latency %u, amo_op %u bits, seed %u\n", b.latency(), b.op_width(), seed);
  if (b.latency() != VX_CFG_FPAMO_LATENCY || b.op_width() != 5) {
    std::printf("FAIL: the unit reports latency %u / op width %u, the configuration %u / 5\n",
                b.latency(), b.op_width(), VX_CFG_FPAMO_LATENCY);
    return 1;
  }

  const int NF = sizeof(FORMS) / sizeof(FORMS[0]);
  // Edge values crossed, every lane of an x2 form taking each pair.
  for (int fi = 0; fi < NF; ++fi) {
    const OpForm& f = FORMS[fi];
    const auto& e = edges(f.fmt);
    for (size_t i = 0; i < e.size(); ++i) {
      for (size_t j = 0; j < e.size(); ++j) {
        uint64_t a[2] = {e[i], e[(i + j + 1) % e.size()]};
        uint64_t c[2] = {e[j], e[(i * 7 + j) % e.size()]};
        Vec v;
        v.op = f.op;
        v.width = f.width;
        v.old = pack(f, a);
        v.rhs = pack(f, c);
        v.form = fi;
        b.cycle(v);
      }
    }
  }

  std::mt19937_64 rng(seed);
  for (uint32_t r = 0; r < rounds; ++r) {
    int fi = static_cast<int>(rng() % NF);
    const OpForm& f = FORMS[fi];
    uint32_t n = fmt_bits(f.fmt);
    uint64_t a[2], c[2];
    for (int l = 0; l < 2; ++l) {
      a[l] = rng() & mask_bits(n);
      switch (rng() % 3) {
      case 0:  c[l] = rng() & mask_bits(n); break;
      case 1:  c[l] = near_value(rng, f.fmt, a[l]); break;
      default: {
        const auto& e = edges(f.fmt);
        c[l] = e[rng() % e.size()];
        break;
      }
      }
    }
    Vec v;
    v.op = f.op;
    v.width = f.width;
    v.old = pack(f, a);
    v.rhs = pack(f, c);
    // Bits above the operand width must not matter.
    if (f.width != ar::W_D && (rng() & 1)) {
      uint64_t junk = rng() & ~mask_bits(8u << f.width);
      v.old |= junk;
      v.rhs |= rng() & ~mask_bits(8u << f.width);
    }
    v.form = fi;
    b.cycle(v);
  }

  // Sums in the subnormal range are exact, so the rounding must leave them alone.
  for (uint32_t r = 0; r < rounds / 8; ++r) {
    int fi = static_cast<int>(rng() % NF);
    const OpForm& f = FORMS[fi];
    uint64_t a[2], c[2];
    for (int l = 0; l < 2; ++l) {
      a[l] = tiny_value(rng, f.fmt);
      c[l] = tiny_value(rng, f.fmt);
    }
    Vec v;
    v.op = f.op;
    v.width = f.width;
    v.old = pack(f, a);
    v.rhs = pack(f, c);
    v.form = fi;
    b.cycle(v);
  }
  b.drain();

  for (int fi = 0; fi < NF; ++fi) {
    std::printf("  %-22s %" PRIu64 " checks\n", FORMS[fi].name, b.per_form[fi]);
  }
  std::printf("amo_fpu: %" PRIu64 " checks, %" PRIu64 " errors\n", b.checks, b.errors);
  if (b.errors != 0) {
    std::printf("FAILED\n");
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
