// dxa_ref against the PTX ISA: the §5.5.7 swizzle tables cell for cell, the
// three §5.5.4 im2col examples pixel for pixel (as their figures draw them),
// the complete_tx byte counts, per-element out-of-bounds fill with signed
// coordinates and element strides, the fill byte rule, the descriptor and
// operand encodings and the order of the asserts.

#include <dxa_ref.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dr = vortex::dxa_ref;

namespace {

int g_fail = 0, g_pass = 0;

void expect(const char* what, uint64_t got, uint64_t want) {
  if (got != want) {
    std::printf("FAIL %s: got %#" PRIx64 ", expected %#" PRIx64 "\n", what, got, want);
    ++g_fail;
  } else {
    ++g_pass;
  }
}

void expect_msg(const char* what, const char* got, const char* want) {
  const bool ok = (got == nullptr && want == nullptr) || (got && want && !std::strcmp(got, want));
  if (!ok) {
    std::printf("FAIL %s: got '%s', expected '%s'\n", what, got ? got : "(none)", want ? want : "(none)");
    ++g_fail;
  } else {
    ++g_pass;
  }
}

// PTX §5.5.7: each row is 128 bytes of eight 16-byte cells; a cell holds the
// number of the source chunk that lands there. The pattern repeats.
const int kSwz32[2][8] = {
  {0, 1, 2, 3, 4, 5, 6, 7},
  {1, 0, 3, 2, 5, 4, 7, 6},
};
const int kSwz64[4][8] = {
  {0, 1, 2, 3, 4, 5, 6, 7},
  {1, 0, 3, 2, 5, 4, 7, 6},
  {2, 3, 0, 1, 6, 7, 4, 5},
  {3, 2, 1, 0, 7, 6, 5, 4},
};
const int kSwz128[8][8] = {
  {0, 1, 2, 3, 4, 5, 6, 7},
  {1, 0, 3, 2, 5, 4, 7, 6},
  {2, 3, 0, 1, 6, 7, 4, 5},
  {3, 2, 1, 0, 7, 6, 5, 4},
  {4, 5, 6, 7, 0, 1, 2, 3},
  {5, 4, 7, 6, 1, 0, 3, 2},
  {6, 7, 4, 5, 2, 3, 0, 1},
  {7, 6, 5, 4, 3, 2, 1, 0},
};

void swizzle_table(uint32_t mode, const int (*table)[8], uint32_t rows) {
  char what[96];
  // Three repetitions of the pattern, from a pattern-aligned start.
  for (uint32_t row = 0; row < 3 * rows; ++row) {
    for (uint32_t cell = 0; cell < 8; ++cell) {
      int src = -1;
      for (uint32_t c = 0; c < 8; ++c) {
        if (dr::swizzle(row * 128 + c * 16, mode) == row * 128 + cell * 16)
          src = (int)c;
      }
      std::snprintf(what, sizeof what, "swizzle %uB row %u cell %u", dr::swizzle_span(mode), row, cell);
      expect(what, (uint64_t)src, (uint64_t)table[row % rows][cell]);
    }
  }
  // Bytes keep their offset in the chunk.
  for (uint32_t r = 0; r < 4096; ++r) {
    std::snprintf(what, sizeof what, "swizzle %uB byte %u in its chunk", dr::swizzle_span(mode), r);
    if ((dr::swizzle(r, mode) & 15u) != (r & 15u) || (dr::swizzle(r, mode) >> 7) != (r >> 7))
      expect(what, dr::swizzle(r, mode), r);
  }
}

void swizzle_tables() {
  for (uint32_t r = 0; r < 4096; ++r)
    expect("no swizzle is the identity", dr::swizzle(r, dr::SWZ_NONE), r);
  swizzle_table(dr::SWZ_32B, kSwz32, 2);
  swizzle_table(dr::SWZ_64B, kSwz64, 4);
  swizzle_table(dr::SWZ_128B, kSwz128, 8);
}

// One run of pixels of image n, row h (and depth 0), w from w0 to w1 by step.
struct Run {
  int n, h, w0, w1, step;
};

// PTX §5.5.4 example: NHWC, C = 64, W = Wsz, H = 14, N = 64; the
// instruction's coordinates (7, 7, w, 0) are (n, h, w, c) (figure: "Image7",
// "H=7 W=4"). Checks every pixel's coordinates and whether it is filled,
// against the green cells of the figure.
void im2col_example(const char* name, uint32_t wsz, uint32_t pixels, uint32_t channels, int32_t lw, int32_t lh,
                    int32_t uw, int32_t uh, int32_t w, uint32_t ow, uint32_t oh, uint32_t es,
                    const std::vector<Run>& runs) {
  dr::Desc d;
  d.rank = 4;
  d.elem = 2;
  d.im2col = true;
  d.size[0] = 64; d.size[1] = wsz; d.size[2] = 14; d.size[3] = 64;
  d.stride[0] = d.size[0] * d.elem;
  d.stride[1] = d.stride[0] * d.size[1];
  d.stride[2] = d.stride[1] * d.size[2];
  d.estride[1] = es;
  d.estride[2] = es;
  d.pixels = pixels;
  d.channels = channels;
  d.lower[0] = lw; d.lower[1] = lh;
  d.upper[0] = uw; d.upper[1] = uh;
  dr::Req q;
  q.coord[0] = 0;   // c
  q.coord[1] = w;   // w
  q.coord[2] = 7;   // h
  q.coord[3] = 7;   // n
  q.offs[0] = ow;
  q.offs[1] = oh;

  std::vector<int64_t> want;  // n, h, w per pixel
  for (const Run& r : runs) {
    for (int x = r.w0; x <= r.w1; x += r.step) {
      want.push_back(r.n);
      want.push_back(r.h);
      want.push_back(x);
    }
  }
  char what[128];
  std::snprintf(what, sizeof what, "%s: pixels in the figure", name);
  expect(what, want.size() / 3, pixels);
  uint32_t k_seen = 0;
  dr::im2col_pixels(d, q, [&](uint32_t k, const int64_t* x, bool in) {
    ++k_seen;
    if (3 * k + 2 >= want.size())
      return;
    std::snprintf(what, sizeof what, "%s pixel %u (n, h, w)", name, k);
    const uint64_t got = ((uint64_t)(uint16_t)x[3] << 32) | ((uint64_t)(uint16_t)x[2] << 16) | (uint16_t)x[1];
    const uint64_t exp = ((uint64_t)(uint16_t)want[3 * k] << 32) | ((uint64_t)(uint16_t)want[3 * k + 1] << 16)
                       | (uint16_t)want[3 * k + 2];
    expect(what, got, exp);
    const bool exp_in = want[3 * k + 1] >= 0 && want[3 * k + 1] < 14 && want[3 * k + 2] >= 0
                     && want[3 * k + 2] < (int64_t)wsz;
    std::snprintf(what, sizeof what, "%s pixel %u in bounds", name, k);
    expect(what, in, exp_in);
    std::snprintf(what, sizeof what, "%s pixel %u channel", name, k);
    expect(what, (uint64_t)x[0], 0);
  });
  std::snprintf(what, sizeof what, "%s: pixels walked", name);
  expect(what, k_seen, pixels);
  std::snprintf(what, sizeof what, "%s: tx bytes", name);
  expect(what, dr::tx_bytes(d), pixels * channels * 2);

  // Element order: pixel-major rows of channelsPerPixel elements.
  uint32_t e = 0;
  bool ordered = true;
  dr::for_each_element(d, q, [&](uint32_t off, bool, uint64_t) { ordered &= (off == 2 * e++); });
  std::snprintf(what, sizeof what, "%s: dense element order", name);
  expect(what, ordered && e == pixels * channels, 1);
}

void im2col_examples() {
  // Example 1 (Figure 13): box W [-1, 7], H [-1, 12], from (w 4, h 7) of image 7.
  im2col_example("im2col example 1", 9, 64, 8, -1, -1, -1, -1, 4, 0, 0, 1, {
    {7, 7, 4, 7, 1},
    {7, 8, -1, 7, 1}, {7, 9, -1, 7, 1}, {7, 10, -1, 7, 1}, {7, 11, -1, 7, 1}, {7, 12, -1, 7, 1},
    {8, -1, -1, 7, 1},
    {8, 0, -1, 4, 1},
  });
  // Example 2 (Figure 14): offsets (2, 2) shift the box to W [2, 8], H [2, 13]; start (w 6, h 9).
  im2col_example("im2col example 2", 9, 64, 8, 0, 0, -2, -2, 4, 2, 2, 1, {
    {7, 9, 6, 8, 1},
    {7, 10, 2, 8, 1}, {7, 11, 2, 8, 1}, {7, 12, 2, 8, 1}, {7, 13, 2, 8, 1},
    {8, 2, 2, 8, 1}, {8, 3, 2, 8, 1}, {8, 4, 2, 8, 1}, {8, 5, 2, 8, 1},
    {8, 6, 2, 6, 1},
  });
  // Traversal stride 2 (Figure 15): W = 8, offsets (1, 1), box W [0, 7], H [0, 13]; start (w 6, h 8).
  im2col_example("im2col traversal stride", 8, 32, 16, -1, -1, -1, -1, 5, 1, 1, 2, {
    {7, 8, 6, 6, 2},
    {7, 10, 0, 6, 2}, {7, 12, 0, 6, 2},
    {8, 0, 0, 6, 2}, {8, 2, 0, 6, 2}, {8, 4, 0, 6, 2}, {8, 6, 0, 6, 2}, {8, 8, 0, 6, 2},
    {8, 10, 0, 4, 2},
  });
}

void tx_counts() {
  dr::Desc d;
  d.rank = 3;
  d.elem = 4;
  d.tile[0] = 8; d.tile[1] = 5; d.tile[2] = 3;
  expect("tx 8x5x3 x4", dr::tx_bytes(d), 8 * 5 * 3 * 4);
  d.estride[1] = 2; d.estride[2] = 3;
  expect("tx es (1,2,3): 8x3x1", dr::tx_bytes(d), 8 * 3 * 1 * 4);
  d.estride[1] = 8;
  expect("tx es 8 on 5: one element", dr::n_elems(d, 1), 1);
  d.estride[1] = 0;
  expect("es 0 reads as 1", dr::n_elems(d, 1), 5);
  dr::Desc f;
  f.rank = 5;
  f.elem = 8;
  for (uint32_t i = 0; i < 5; ++i) {
    f.tile[i] = 2 + i;
    f.estride[i] = 1 + (i % 3);
  }
  expect("tx 5D strided", dr::tx_bytes(f), 2 * 2 * 2 * 5 * 3 * 8);
  dr::Desc g;
  g.rank = 5;
  g.elem = 1;
  g.im2col = true;
  g.pixels = 1024;
  g.channels = 256;
  expect("tx im2col max", dr::tx_bytes(g), 1024 * 256);
  g.estride[1] = 4;
  expect("tx im2col ignores the traversal stride", dr::tx_bytes(g), 1024 * 256);
}

uint8_t gbyte(uint64_t a) {
  return (uint8_t)(a * 131u + (a >> 8) * 7u + 1u);
}

// A 2D tile against an independent per-element loop: signed coordinates,
// per-element fill on both sides of both dims, element strides, NaN fill.
void tile_oob() {
  for (int32_t c0 = -5; c0 <= 9; ++c0) {
    for (int32_t c1 = -3; c1 <= 6; ++c1) {
      for (uint32_t es1 = 1; es1 <= 3; ++es1) {
        dr::Desc d;
        d.rank = 2;
        d.elem = 2;
        d.base = 0x10000;
        d.size[0] = 7; d.size[1] = 5;
        d.stride[0] = 64;
        d.tile[0] = 8; d.tile[1] = 4;
        d.estride[1] = es1;
        d.cfill = dr::nan_fill(2);
        dr::Req q;
        q.coord[0] = c0;
        q.coord[1] = c1;
        std::vector<uint8_t> img(256, 0xEE), want(256, 0xEE);
        const uint32_t tx = dr::render(d, q, 32, gbyte, img.data(), img.size());
        const uint32_t n1 = (4 + es1 - 1) / es1;
        for (uint32_t i1 = 0; i1 < n1; ++i1) {
          for (uint32_t i0 = 0; i0 < 8; ++i0) {
            const int32_t x0 = c0 + (int32_t)i0, x1 = c1 + (int32_t)(i1 * es1);
            const bool in = x0 >= 0 && x0 < 7 && x1 >= 0 && x1 < 5;
            for (uint32_t b = 0; b < 2; ++b) {
              const uint32_t a = 32 + (i1 * 8 + i0) * 2 + b;
              want[a] = in ? gbyte(0x10000 + (uint64_t)(x0 * 2 + x1 * 64 + (int32_t)b)) : (b ? 0x7F : 0xFF);
            }
          }
        }
        char what[96];
        std::snprintf(what, sizeof what, "tile OOB c (%d, %d) es1 %u image", c0, c1, es1);
        expect(what, img == want, 1);
        std::snprintf(what, sizeof what, "tile OOB c (%d, %d) es1 %u tx", c0, c1, es1);
        expect(what, tx, 8 * n1 * 2);
      }
    }
  }
}

// A swizzled 5D box with OOB against the dense render plus the XOR.
void swizzle_render() {
  for (uint32_t mode = dr::SWZ_32B; mode <= dr::SWZ_128B; ++mode) {
    for (uint32_t elem = 1; elem <= 8; elem *= 2) {
      dr::Desc d;
      d.rank = 3;
      d.elem = elem;
      d.size[0] = 13; d.size[1] = 9; d.size[2] = 3;
      d.stride[0] = 256; d.stride[1] = 4096;
      d.tile[0] = dr::swizzle_span(mode) / elem; d.tile[1] = 6; d.tile[2] = 2;
      d.cfill = 0x11223344u;
      dr::Req q;
      q.coord[0] = -2; q.coord[1] = 5; q.coord[2] = 2;
      std::vector<uint8_t> plain(8192, 0), swz(8192, 0);
      dr::Desc p = d;
      p.swizzle = dr::SWZ_NONE;
      dr::render(p, q, 1024, gbyte, plain.data(), plain.size());
      d.swizzle = mode;
      dr::render(d, q, 1024, gbyte, swz.data(), swz.size());
      bool ok = true;
      for (uint32_t a = 0; a < 8192; ++a)
        ok &= (swz[dr::swizzle(a, mode)] == plain[a]);
      char what[96];
      std::snprintf(what, sizeof what, "swizzle %uB elem %u render", dr::swizzle_span(mode), elem);
      expect(what, ok, 1);
      std::snprintf(what, sizeof what, "swizzle %uB elem %u check", dr::swizzle_span(mode), elem);
      expect_msg(what, dr::check_tensor(d, 1024), nullptr);
      d.tile[0] += 1;
      expect_msg("swizzle span", dr::check_tensor(d, 1024), dr::MSG_SWZ_SPAN);
      d.tile[0] -= 1;
      expect_msg("swizzle align", dr::check_tensor(d, 1024 + 16), dr::MSG_SWZ_ALIGN);
      expect_msg("swizzle align, prefetch", dr::check_tensor(d, 16, true), nullptr);
    }
  }
}

void fill_bytes() {
  expect("NaN f16", dr::nan_fill(2), 0x7FFF7FFFu);
  expect("NaN f32", dr::nan_fill(4), 0x7FFFFFFFu);
  expect("NaN f64", dr::nan_fill(8), 0x7FFFFFFFu);
  for (uint32_t a = 0; a < 8; ++a)
    expect("fill byte a & 3", dr::fill_byte(0x44332211u, a), 0x11u * ((a & 3) + 1));
}

void asserts() {
  dr::Desc d;
  d.rank = 2;
  d.elem = 2;
  d.tile[0] = 8; d.tile[1] = 2;
  expect_msg("plain tile", dr::check_tensor(d, 0), nullptr);
  d.estride[0] = 2;
  expect_msg("es0 without interleave", dr::check_tensor(d, 0), dr::MSG_ES0);
  d.interleave = dr::IL_16B;
  expect_msg("interleave at rank 2", dr::check_tensor(d, 0), dr::MSG_IL_RANK);
  d.rank = 3;
  d.tile[2] = 1;
  expect_msg("es0 with interleave", dr::check_tensor(d, 0), nullptr);
  d.interleave = dr::IL_32B;
  expect_msg("32B interleave without swizzle", dr::check_tensor(d, 0), dr::MSG_IL32);
  d.swizzle = dr::SWZ_32B;
  expect_msg("32B interleave + 32B swizzle", dr::check_tensor(d, 0), nullptr);
  d.interleave = dr::IL_INVALID;
  expect_msg("interleave 3", dr::check_tensor(d, 0), dr::MSG_IL_INVALID);
  d.interleave = dr::IL_NONE;
  d.estride[0] = 1;
  d.layout = dr::LAYOUT_KMAJOR;
  expect_msg("K-major with swizzle", dr::check_tensor(d, 0), dr::MSG_LAYOUT);
  d.swizzle = dr::SWZ_NONE;
  d.estride[1] = 2;
  expect_msg("K-major with an element stride", dr::check_tensor(d, 0), dr::MSG_LAYOUT);
  d.layout = dr::LAYOUT_FLAT;
  expect_msg("Flat reads element strides as 1", dr::check_tensor(d, 0), nullptr);
  dr::Desc i;
  i.rank = 2;
  i.im2col = true;
  expect_msg("im2col at rank 2", dr::check_tensor(i, 0), dr::MSG_I2C_RANK);
  i.rank = 4;
  i.elem = 2;
  i.channels = 64;
  i.pixels = 4;
  i.swizzle = dr::SWZ_128B;
  expect_msg("im2col 128B swizzle, 64 channels", dr::check_tensor(i, 0), nullptr);
  i.channels = 65;
  expect_msg("im2col swizzle span", dr::check_tensor(i, 0), dr::MSG_SWZ_SPAN);
  i.channels = 8;
  i.estride[0] = 2;
  i.interleave = dr::IL_16B;
  expect_msg("im2col es0", dr::check_tensor(i, 0), dr::MSG_ES0);
  expect("mbar aligned", dr::mbar_aligned(0x40), 1);
  expect("mbar misaligned", dr::mbar_aligned(0x44), 0);
  expect_msg("bulk ok", dr::check_bulk(0x100, 0x2000, 64), nullptr);
  expect_msg("bulk size", dr::check_bulk(0x100, 0x2000, 24), dr::MSG_BULK_ALIGN);
  expect_msg("bulk size 0", dr::check_bulk(0x100, 0x2000, 0), dr::MSG_BULK_ALIGN);
  expect_msg("bulk gsrc", dr::check_bulk(0x100, 0x2008, 64), dr::MSG_BULK_ALIGN);
  expect_msg("bulk dst", dr::check_bulk(0x108, 0x2000, 64), dr::MSG_BULK_ALIGN);
}

void encodings() {
  for (uint32_t rank = 3; rank <= 5; ++rank) {
    dr::Desc d;
    d.base = 0x123456789ull;
    d.rank = rank;
    d.elem = 8;
    d.im2col = true;
    d.swizzle = dr::SWZ_64B;
    d.channels = 256;
    d.pixels = 1024;
    const int32_t lim = rank == 3 ? 32768 : rank == 4 ? 128 : 16;
    for (uint32_t s = 0; s < rank - 2; ++s) {
      d.lower[s] = -lim + (int32_t)s;
      d.upper[s] = lim - 1 - (int32_t)s;
      d.estride[s + 1] = 8 - s;
    }
    for (uint32_t i = 0; i < 5; ++i)
      d.size[i] = 100 + i;
    for (uint32_t i = 0; i < 4; ++i)
      d.stride[i] = 1000 * (i + 1);
    d.cfill = dr::nan_fill(8);
    uint32_t w[dr::DESC_WORDS] = {};
    dr::encode(d, w);
    const dr::Desc e = dr::decode(w);
    char what[64];
    std::snprintf(what, sizeof what, "im2col %uD descriptor round trip", rank);
    bool ok = e.base == d.base && e.rank == rank && e.elem == 8 && e.im2col && e.swizzle == d.swizzle
           && e.channels == 256 && e.pixels == 1024 && e.cfill == d.cfill;
    for (uint32_t s = 0; s < 3; ++s)
      ok &= e.lower[s] == d.lower[s] && e.upper[s] == d.upper[s];
    for (uint32_t i = 0; i < 5; ++i)
      ok &= e.size[i] == d.size[i] && e.estride[i] == d.estride[i];
    expect(what, ok, 1);
    std::snprintf(what, sizeof what, "im2col %uD META bit 13", rank);
    expect(what, (w[VX_DCR_DXA_DESC_META_OFF] >> 13) & 1, 1);
  }
  dr::Req q;
  const int32_t c[5] = {-1, 2147483647, -2147483647 - 1, 7, -9};
  for (uint32_t i = 0; i < 5; ++i)
    q.coord[i] = c[i];
  q.offs[0] = 0xFFFF; q.offs[1] = 3; q.offs[2] = 31;
  const dr::Ops o = dr::pack_tload(13, true, 0xDEADBEEF0ull, 0xC0FFEE8ull, q, 0xA5A5);
  expect("TLOAD rs1 lane 2", o.rs1[2], 0x7FFFFFFFFFFFFFFFull);
  expect("TLOAD rs1 lane 3", o.rs1[3], 0x0000000780000000ull);
  expect("TLOAD rs2 lane 0", o.rs2[0], 0x0000001DFFFFFFF7ull);
  expect("TLOAD rs2 lane 1", o.rs2[1], 0x001F0003FFFFA5A5ull);
  uint32_t slot = 0;
  bool pf = false;
  uint64_t dst = 0, mbar = 0;
  dr::Req r;
  uint16_t mask = 0;
  dr::unpack_tload(o, &slot, &pf, &dst, &mbar, &r, &mask);
  bool ok = slot == 13 && pf && dst == 0xDEADBEEF0ull && mbar == 0xC0FFEE8ull && mask == 0xA5A5;
  for (uint32_t i = 0; i < 5; ++i)
    ok &= r.coord[i] == c[i];
  for (uint32_t i = 0; i < 3; ++i)
    ok &= r.offs[i] == q.offs[i];
  expect("TLOAD operand round trip", ok, 1);
}

} // namespace

int main() {
  swizzle_tables();
  im2col_examples();
  tx_counts();
  tile_oob();
  swizzle_render();
  fill_bytes();
  asserts();
  encodings();
  std::printf("dxa_ref: %d checks passed, %d failed\n", g_pass, g_fail);
  if (g_fail) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
