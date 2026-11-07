// dxa_ref against the PTX ISA: the §5.5.7 swizzle tables cell for cell, the
// three §5.5.4 im2col examples pixel for pixel (as their figures draw them),
// the complete_tx byte counts, per-element out-of-bounds fill with signed
// coordinates and element strides, the fill byte rule, the descriptor and
// operand encodings and the order of the asserts; the stores' byte map, the
// reduction tables and semantics, and the store and bulk-group encodings.

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

// Stores: a load of the box into local memory, then the store of that local
// memory, writes back exactly the in-bounds elements, each byte from its
// source, and nothing else; tile, edges past the end of every dim, element
// strides, swizzle, interleave and im2col.
void store_round_trip(const char* name, const dr::Desc& d, const dr::Req& q, uint32_t src_rel) {
  const uint64_t G = 1u << 16;
  std::vector<uint8_t> lmem(32768, 0xA5);
  auto src = [&](uint64_t a) -> uint8_t { return gbyte(a); };
  dr::render(d, q, src_rel, src, lmem.data(), lmem.size());
  std::vector<uint8_t> out(G, 0xEE);
  std::vector<uint8_t> hits(G, 0);
  bool inside = true;
  const uint32_t n = dr::store_box(d, q, src_rel, lmem.data(), lmem.size(), [&](uint64_t ga) {
    if (ga < d.base || ga - d.base >= G) {
      inside = false;
      static uint8_t sink;
      return &sink;
    }
    ++hits[ga - d.base];
    return &out[ga - d.base];
  });
  uint32_t in_elems = 0;
  std::vector<uint8_t> want(G, 0xEE);
  dr::for_each_element(d, q, [&](uint32_t, bool in, uint64_t ga) {
    if (!in)
      return;
    ++in_elems;
    for (uint32_t b = 0; b < d.elem; ++b)
      want[ga + b - d.base] = gbyte(ga + b);
  });
  char what[128];
  std::snprintf(what, sizeof what, "%s: store inside the tensor", name);
  expect(what, inside, 1);
  std::snprintf(what, sizeof what, "%s: store bytes", name);
  expect(what, n, in_elems * d.elem);
  std::snprintf(what, sizeof what, "%s: global image", name);
  expect(what, out == want, 1);
  if (!d.im2col) {
    bool once = true;
    for (uint64_t a = 0; a < G; ++a)
      once &= hits[a] <= 1;
    std::snprintf(what, sizeof what, "%s: each global byte written once", name);
    expect(what, once, 1);
  }
}

dr::Desc store_tile(uint32_t rank, uint32_t elem, std::initializer_list<uint32_t> size,
                    std::initializer_list<uint32_t> tile) {
  dr::Desc d;
  d.base = 0x40000;
  d.rank = rank;
  d.elem = elem;
  uint32_t i = 0;
  for (uint32_t s : size)
    d.size[i++] = s;
  i = 0;
  for (uint32_t t : tile)
    d.tile[i++] = t;
  d.stride[0] = (d.size[0] * elem + 16 + 15) / 16 * 16;
  for (uint32_t k = 1; k + 1 < rank; ++k)
    d.stride[k] = d.stride[k - 1] * d.size[k];
  return d;
}

dr::Req coords(std::initializer_list<int32_t> c) {
  dr::Req q;
  uint32_t i = 0;
  for (int32_t v : c)
    q.coord[i++] = v;
  return q;
}

void stores() {
  store_round_trip("store 1D edge", store_tile(1, 4, {37}, {16}), coords({30}), 64);
  store_round_trip("store 2D edge both dims", store_tile(2, 2, {13, 6}, {16, 4}), coords({8, 4}), 64);
  store_round_trip("store 3D in bounds", store_tile(3, 1, {40, 5, 3}, {32, 4, 2}), coords({3, 1, 1}), 16);
  dr::Desc e5 = store_tile(5, 8, {5, 3, 3, 2, 2}, {8, 4, 4, 3, 3});
  store_round_trip("store 5D past every end", e5, coords({2, 1, 1, 1, 1}), 0);
  dr::Desc es = store_tile(5, 4, {4, 12, 10, 9, 5}, {4, 8, 7, 5, 4});
  const uint32_t esv[5] = {1, 2, 3, 4, 5};
  for (uint32_t i = 0; i < 5; ++i)
    es.estride[i] = esv[i];
  store_round_trip("store 5D element strides", es, coords({0, 1, 0, 2, 0}), 32);
  for (uint32_t mode = dr::SWZ_32B; mode <= dr::SWZ_128B; ++mode) {
    for (uint32_t elem = 1; elem <= 8; elem *= 2) {
      dr::Desc d = store_tile(2, elem, {dr::swizzle_span(mode) / elem + 3, 20},
                              {dr::swizzle_span(mode) / elem, 16});
      d.swizzle = mode;
      char what[64];
      std::snprintf(what, sizeof what, "store swizzle %uB elem %u", dr::swizzle_span(mode), elem);
      store_round_trip(what, d, coords({2, 7}), 384);
    }
  }
  dr::Desc il = store_tile(3, 2, {20, 4, 3}, {16, 3, 2});
  il.interleave = dr::IL_16B;
  il.estride[0] = 2;
  store_round_trip("store 16B interleave, es0 2", il, coords({7, 2, 2}), 32);
  dr::Desc il32 = store_tile(4, 2, {40, 5, 3, 2}, {32, 4, 2, 2});
  il32.interleave = dr::IL_32B;
  il32.swizzle = dr::SWZ_32B;
  il32.estride[0] = 2;
  store_round_trip("store 32B interleave", il32, coords({12, 3, 2, 1}), 128);
  for (uint32_t rank = 3; rank <= 5; ++rank) {
    dr::Desc d;
    d.base = 0x40000;
    d.rank = rank;
    d.elem = 2;
    d.im2col = true;
    const uint32_t sz[5] = {16, 7, rank > 3 ? 6u : 2u, rank > 4 ? 3u : 2u, 2};
    for (uint32_t i = 0; i < rank; ++i)
      d.size[i] = sz[i];
    d.stride[0] = 32;
    for (uint32_t i = 1; i + 1 < rank; ++i)
      d.stride[i] = d.stride[i - 1] * d.size[i];
    d.channels = 8;
    d.pixels = 24;
    for (uint32_t s = 0; s + 2 < rank; ++s) {
      d.lower[s] = -1;
      d.upper[s] = 0;
    }
    dr::Req q;
    q.coord[0] = 4;
    for (uint32_t s = 1; s + 1 < rank; ++s)
      q.coord[s] = -1;
    q.coord[rank - 1] = 0;
    char what[64];
    std::snprintf(what, sizeof what, "store im2col %uD", rank);
    store_round_trip(what, d, q, 64);
  }

  // The byte map of a 2D tile against an independent formula.
  dr::Desc d = store_tile(2, 4, {10, 6}, {8, 4});
  d.swizzle = dr::SWZ_64B;
  const dr::Req q = coords({5, 3});
  std::vector<uint64_t> got, want;
  dr::for_each_store_byte(d, q, 128, [&](uint32_t l, uint64_t g) { got.push_back(((uint64_t)l << 40) | g); });
  for (uint32_t i1 = 0; i1 < 4; ++i1)
    for (uint32_t i0 = 0; i0 < 8; ++i0)
      for (uint32_t b = 0; b < 4; ++b)
        if (5 + i0 < 10 && 3 + i1 < 6)
          want.push_back(((uint64_t)dr::swizzle(128 + (i1 * 8 + i0) * 4 + b, dr::SWZ_64B) << 40)
                         | (d.base + (3 + i1) * d.stride[0] + (5 + i0) * 4 + b));
  expect("store byte map 2D, 64B swizzle", got == want, 1);

  dr::Desc t = store_tile(2, 4, {10, 6}, {8, 4});
  expect_msg("tile store at (0, 0)", dr::check_tensor_store(t, coords({0, 0}), 0), nullptr);
  expect_msg("tile store below dim 0", dr::check_tensor_store(t, coords({-1, 0}), 0), dr::MSG_ST_NEG);
  expect_msg("tile store below dim 1", dr::check_tensor_store(t, coords({0, -3}), 0), dr::MSG_ST_NEG);
  expect_msg("tile store past the end", dr::check_tensor_store(t, coords({9, 5}), 0), nullptr);
  t.swizzle = dr::SWZ_128B;
  expect_msg("tile store swizzle alignment first", dr::check_tensor_store(t, coords({-1, 0}), 16),
             dr::MSG_SWZ_ALIGN);
  dr::Desc i2 = d;
  i2.swizzle = dr::SWZ_NONE;
  i2.rank = 3;
  i2.im2col = true;
  i2.channels = 4;
  i2.pixels = 4;
  expect_msg("im2col store below the tensor", dr::check_tensor_store(i2, coords({0, -2, 0}), 0), nullptr);
  dr::Desc km = store_tile(2, 4, {10, 6}, {8, 4});
  const uint32_t non_tile[] = {dr::LAYOUT_KMAJOR, dr::LAYOUT_FLAT, dr::LAYOUT_BLOCKMAJOR};
  for (uint32_t layout : non_tile) {
    km.layout = layout;
    expect_msg("store with a non-tile layout", dr::check_tensor_store(km, coords({0, 0}), 0), dr::MSG_ST_LAYOUT);
    expect_msg("store layout before a negative coordinate", dr::check_tensor_store(km, coords({-1, 0}), 0),
               dr::MSG_ST_LAYOUT);
  }
  km.layout = dr::LAYOUT_KMAJOR;
  km.swizzle = dr::SWZ_32B;
  expect_msg("store layout before the load's layout rule", dr::check_tensor_store(km, coords({0, 0}), 0),
             dr::MSG_ST_LAYOUT);
}

// The PTX tables, written out as text.
const char* const kGlobalPairs[] = {
  "and.b32", "or.b32", "xor.b32", "and.b64", "or.b64", "xor.b64",
  "min.u32", "min.s32", "min.u64", "min.s64", "min.f16", "min.bf16",
  "max.u32", "max.s32", "max.u64", "max.s64", "max.f16", "max.bf16",
  "add.u32", "add.s32", "add.u64", "add.f32", "add.f64", "add.f16", "add.bf16",
  "inc.u32", "dec.u32",
};
const char* const kClusterPairs[] = {
  "and.b32", "or.b32", "xor.b32", "add.u32", "add.s32", "min.u32", "min.s32", "max.u32", "max.s32",
  "inc.u32", "dec.u32", "add.u64",
};

void legality() {
  for (uint32_t kind : {dr::ST_RED_GLOBAL, dr::ST_RED_TENSOR, dr::ST_RED_CLUSTER}) {
    const bool cl = kind == dr::ST_RED_CLUSTER;
    const char* const* pairs = cl ? kClusterPairs : kGlobalPairs;
    const size_t np = cl ? sizeof(kClusterPairs) / sizeof(*kClusterPairs)
                         : sizeof(kGlobalPairs) / sizeof(*kGlobalPairs);
    uint32_t legal = 0;
    for (uint32_t op = 0; op < 16; ++op) {
      for (uint32_t type = 0; type < 16; ++type) {
        char name[32];
        std::snprintf(name, sizeof name, "%s.%s", dr::red_op_name(op), dr::red_type_name(type, op));
        bool listed = false;
        for (size_t i = 0; i < np && op < dr::RED_OPS && type < dr::RT_TYPES; ++i)
          listed |= !std::strcmp(pairs[i], name);
        legal += dr::red_legal(kind, op, type);
        char what[96];
        std::snprintf(what, sizeof what, "kind %u %s (op %u type %u) legal", kind, name, op, type);
        expect(what, dr::red_legal(kind, op, type), listed);
      }
    }
    char what[64];
    std::snprintf(what, sizeof what, "kind %u legal pairs", kind);
    expect(what, legal, np);
  }
  expect("needs FPAMO add.f32", dr::red_needs_fpamo(dr::RED_ADD, dr::RT_F32), 1);
  expect("needs FPAMO min.bf16", dr::red_needs_fpamo(dr::RED_MIN, dr::RT_BF16), 1);
  expect("needs FPAMO inc.u32", dr::red_needs_fpamo(dr::RED_INC, dr::RT_U32), 1);
  expect("needs FPAMO add.u64", dr::red_needs_fpamo(dr::RED_ADD, dr::RT_U64), 0);
  expect("needs FPAMO max.s64", dr::red_needs_fpamo(dr::RED_MAX, dr::RT_S64), 0);

  dr::Desc d;
  d.elem = 4;
  expect_msg("kind 5", dr::check_store_ctl(5, 0, 0, true, true), dr::MSG_ENCODING);
  expect_msg("kind 7", dr::check_store_ctl(7, 0, 0, true, true), dr::MSG_ENCODING);
  expect_msg("store ignores op and type", dr::check_store_ctl(dr::ST_BULK, 15, 15, false, false), nullptr);
  expect_msg("tensor store ignores op and type", dr::check_store_ctl(dr::ST_TENSOR, 1, 0, false, false), nullptr);
  expect_msg("min.b32", dr::check_store_ctl(dr::ST_RED_GLOBAL, dr::RED_MIN, 8, true, true), dr::MSG_RED_INVALID);
  expect_msg("and.s32", dr::check_store_ctl(dr::ST_RED_GLOBAL, dr::RED_AND, dr::RT_S32, true, true),
             dr::MSG_RED_INVALID);
  expect_msg("cluster add.f32", dr::check_store_ctl(dr::ST_RED_CLUSTER, dr::RED_ADD, dr::RT_F32, true, true),
             dr::MSG_RED_INVALID);
  expect_msg("invalid before FPAMO", dr::check_store_ctl(dr::ST_RED_GLOBAL, dr::RED_INC, dr::RT_S32, false, false),
             dr::MSG_RED_INVALID);
  expect_msg("f32 without FPAMO", dr::check_store_ctl(dr::ST_RED_GLOBAL, dr::RED_ADD, dr::RT_F32, true, false),
             dr::MSG_RED_FPAMO);
  expect_msg("FPAMO before the LLC", dr::check_store_ctl(dr::ST_RED_GLOBAL, dr::RED_ADD, dr::RT_F32, false, false),
             dr::MSG_RED_FPAMO);
  expect_msg("cluster inc without FPAMO", dr::check_store_ctl(dr::ST_RED_CLUSTER, dr::RED_INC, dr::RT_U32, true,
             false), dr::MSG_RED_FPAMO);
  expect_msg("global without an LLC", dr::check_store_ctl(dr::ST_RED_GLOBAL, dr::RED_ADD, dr::RT_U32, false, true),
             dr::MSG_RED_LLC);
  expect_msg("tensor without an LLC", dr::check_store_ctl(dr::ST_RED_TENSOR, dr::RED_ADD, dr::RT_U32, false, true,
             &d), dr::MSG_RED_LLC);
  expect_msg("cluster without an LLC", dr::check_store_ctl(dr::ST_RED_CLUSTER, dr::RED_ADD, dr::RT_U32, false,
             false), nullptr);
  expect_msg("tensor type size", dr::check_store_ctl(dr::ST_RED_TENSOR, dr::RED_ADD, dr::RT_U64, true, true, &d),
             dr::MSG_RED_SIZE);
  d.elem = 2;
  expect_msg("tensor f16 on 2-byte elements", dr::check_store_ctl(dr::ST_RED_TENSOR, dr::RED_ADD, dr::RT_F16, true,
             true, &d), nullptr);
  expect_msg("bulk global size not checked", dr::check_store_ctl(dr::ST_RED_GLOBAL, dr::RED_ADD, dr::RT_F16, true,
             true, &d), nullptr);
  expect_msg("bgop 2", dr::check_bgop(2), nullptr);
  expect_msg("bgop 3", dr::check_bgop(3), dr::MSG_ENCODING);
  expect_msg("pfence x2", dr::check_pfence(2), nullptr);
  expect_msg("pfence x3", dr::check_pfence(3), dr::MSG_PFENCE);
  expect("bulk group kinds", (dr::is_bulk_group(0) << 0) | (dr::is_bulk_group(1) << 1) | (dr::is_bulk_group(2) << 2)
         | (dr::is_bulk_group(3) << 3) | (dr::is_bulk_group(4) << 4) | (dr::is_bulk_group(5) << 5), 0x17);
}

void amo_map() {
  struct { uint32_t op, type, amo, width, uns; } t[] = {
    {dr::RED_ADD, dr::RT_U32, 0x02, 2, 1}, {dr::RED_ADD, dr::RT_S32, 0x02, 2, 0},
    {dr::RED_ADD, dr::RT_U64, 0x02, 3, 1}, {dr::RED_ADD, dr::RT_F32, 0x12, 2, 0},
    {dr::RED_ADD, dr::RT_F64, 0x12, 3, 0}, {dr::RED_ADD, dr::RT_F16, 0x14, 2, 0},
    {dr::RED_ADD, dr::RT_BF16, 0x15, 2, 0}, {dr::RED_MIN, dr::RT_U32, 0x07, 2, 1},
    {dr::RED_MIN, dr::RT_S64, 0x07, 3, 0}, {dr::RED_MAX, dr::RT_U64, 0x08, 3, 1},
    {dr::RED_MIN, dr::RT_F16, 0x16, 2, 0}, {dr::RED_MAX, dr::RT_F16, 0x17, 2, 0},
    {dr::RED_MIN, dr::RT_BF16, 0x18, 2, 0}, {dr::RED_MAX, dr::RT_BF16, 0x19, 2, 0},
    {dr::RED_INC, dr::RT_U32, 0x10, 2, 1}, {dr::RED_DEC, dr::RT_U32, 0x11, 2, 1},
    {dr::RED_AND, dr::RT_U64, 0x06, 3, 1}, {dr::RED_OR, dr::RT_U32, 0x05, 2, 1},
    {dr::RED_XOR, dr::RT_U32, 0x04, 2, 1},
  };
  for (const auto& e : t) {
    const dr::RedAmo a = dr::red_amo(e.op, e.type);
    char what[64];
    std::snprintf(what, sizeof what, "AMO of %s.%s", dr::red_op_name(e.op), dr::red_type_name(e.type, e.op));
    expect(what, ((uint64_t)a.amo_op << 16) | (a.width << 8) | a.is_unsigned, ((uint64_t)e.amo << 16)
           | (e.width << 8) | e.uns);
  }
  expect("request bytes f16", dr::red_word_bytes(dr::RT_F16), 4);
  expect("request bytes s64", dr::red_word_bytes(dr::RT_S64), 8);
}

void reductions() {
  struct { uint32_t op, type; uint64_t old, src, want; } t[] = {
    {dr::RED_ADD, dr::RT_U32, 0xFFFFFFFF, 2, 1},
    {dr::RED_ADD, dr::RT_S32, 0x80000000, 0xFFFFFFFF, 0x7FFFFFFF},
    {dr::RED_ADD, dr::RT_U64, ~0ull, 3, 2},
    {dr::RED_MIN, dr::RT_U32, 0xFFFFFFFE, 5, 5},
    {dr::RED_MIN, dr::RT_S32, 0xFFFFFFFE, 5, 0xFFFFFFFE},
    {dr::RED_MAX, dr::RT_S32, 0x80000000, 0x7FFFFFFF, 0x7FFFFFFF},
    {dr::RED_MAX, dr::RT_U32, 0x80000000, 0x7FFFFFFF, 0x80000000},
    {dr::RED_MIN, dr::RT_U64, 0x8000000000000000ull, 1, 1},
    {dr::RED_MIN, dr::RT_S64, 0x8000000000000000ull, 1, 0x8000000000000000ull},
    {dr::RED_MAX, dr::RT_S64, 0xFFFFFFFFFFFFFFFFull, 0, 0},
    {dr::RED_MAX, dr::RT_U64, 0xFFFFFFFFFFFFFFFFull, 0, 0xFFFFFFFFFFFFFFFFull},
    {dr::RED_AND, dr::RT_U64, 0xF0F0F0F0F0F0F0F0ull, 0xFF00FF00FF00FF00ull, 0xF000F000F000F000ull},
    {dr::RED_OR,  dr::RT_U32, 0xF0F0F0F0, 0x0F, 0xF0F0F0FF},
    {dr::RED_XOR, dr::RT_U32, 0xFFFF0000, 0x0FF00FF0, 0xF00F0FF0},
    {dr::RED_INC, dr::RT_U32, 4, 5, 5},
    {dr::RED_INC, dr::RT_U32, 5, 5, 0},
    {dr::RED_DEC, dr::RT_U32, 0, 7, 7},
    {dr::RED_DEC, dr::RT_U32, 9, 7, 7},
    {dr::RED_DEC, dr::RT_U32, 3, 7, 2},
    {dr::RED_ADD, dr::RT_F32, 0x3F800000, 0x3F800000, 0x40000000},
    {dr::RED_ADD, dr::RT_F32, 0x00000001, 0x00000000, 0x00000000},   // f32 flushed at the LLC
    {dr::RED_ADD, dr::RT_F32, 0x7F800000, 0xFF800000, 0x7FFFFFFF},
    {dr::RED_ADD, dr::RT_F64, 0x3FF0000000000000ull, 0x3FF0000000000000ull, 0x4000000000000000ull},
    {dr::RED_ADD, dr::RT_F64, 0x0000000000000001ull, 0, 0x0000000000000001ull},
    {dr::RED_ADD, dr::RT_F16, 0x3C00, 0x3C00, 0x4000},
    {dr::RED_ADD, dr::RT_F16, 0x0001, 0x0001, 0x0002},                // .noftz
    {dr::RED_ADD, dr::RT_BF16, 0x3F80, 0x3F80, 0x4000},
    {dr::RED_MIN, dr::RT_F16, 0x8000, 0x0000, 0x8000},
    {dr::RED_MAX, dr::RT_F16, 0x8000, 0x0000, 0x0000},
    {dr::RED_MIN, dr::RT_F16, 0x7E01, 0x3C00, 0x3C00},
    {dr::RED_MAX, dr::RT_BF16, 0xBF80, 0x3F80, 0x3F80},
    {dr::RED_MIN, dr::RT_BF16, 0xBF80, 0x3F80, 0xBF80},
    {dr::RED_MIN, dr::RT_F32, 5, 3, 5},                                // illegal: old kept
  };
  for (const auto& e : t) {
    char what[96];
    std::snprintf(what, sizeof what, "%s.%s %#" PRIx64 " %#" PRIx64, dr::red_op_name(e.op),
                  dr::red_type_name(e.type, e.op), e.old, e.src);
    expect(what, dr::reduce_elem(e.op, e.type, e.old, e.src), e.want);
  }

  // Packed 16-bit pairs reduce lane by lane.
  const uint8_t a[8] = {0x00, 0x3C, 0x00, 0xC0, 0x01, 0x00, 0x00, 0x7C};
  const uint8_t b[8] = {0x00, 0x3C, 0x00, 0x40, 0x01, 0x00, 0x00, 0xFC};
  uint8_t r[8];
  std::memcpy(r, a, 8);
  dr::reduce_bytes(dr::RED_ADD, dr::RT_F16, r, b, 8);
  const uint8_t w[8] = {0x00, 0x40, 0x00, 0x00, 0x02, 0x00, 0xFF, 0x7F};
  expect("f16 pairs: 1+1, -2+2, tiny+tiny, inf-inf", !std::memcmp(r, w, 8), 1);

  // A tensor reduction touches exactly the store's bytes.
  dr::Desc d = store_tile(2, 4, {13, 6}, {16, 4});
  d.swizzle = dr::SWZ_64B;
  const dr::Req q = coords({8, 3});
  std::vector<uint8_t> lmem(4096);
  for (uint32_t i = 0; i < lmem.size(); ++i)
    lmem[i] = (uint8_t)(i * 7 + 3);
  std::vector<uint8_t> st(1u << 16, 0), red(1u << 16, 0), red2(1u << 16, 0);
  for (uint32_t i = 0; i < red2.size(); ++i)
    red2[i] = (uint8_t)(i * 13 + 5);
  std::vector<uint8_t> red2_in = red2;
  auto at = [&](std::vector<uint8_t>& v) { return [&v, &d](uint64_t ga) { return &v[ga - d.base]; }; };
  const uint32_t ns = dr::store_box(d, q, 256, lmem.data(), lmem.size(), at(st));
  const uint32_t nr = dr::reduce_box(d, q, 256, dr::RED_ADD, dr::RT_U32, lmem.data(), lmem.size(), at(red));
  expect("tensor reduce bytes", nr, ns);
  expect("add.u32 onto zeros is the store", red == st, 1);
  dr::reduce_box(d, q, 256, dr::RED_XOR, dr::RT_U32, lmem.data(), lmem.size(), at(red2));
  bool ok = true;
  std::vector<uint8_t> touched(1u << 16, 0);
  dr::for_each_store_byte(d, q, 256, [&](uint32_t, uint64_t ga) { touched[ga - d.base] = 1; });
  for (uint32_t i = 0; i < red2.size(); ++i)
    ok &= touched[i] ? red2[i] == (uint8_t)(red2_in[i] ^ st[i]) : red2[i] == red2_in[i];
  expect("xor.b32: the box's bytes only", ok, 1);
}

void store_encodings() {
  dr::Store s;
  s.kind = dr::ST_RED_CLUSTER;
  s.op = dr::RED_MAX;
  s.type = dr::RT_S32;
  s.src = 0x1230;
  s.dst = 0xC0000000ull + 0x2040;
  s.mbar = 0xC0000000ull + 0x2008;
  s.size = 4096;
  dr::Ops o = dr::pack_store(s);
  expect("STORE cluster rs1 lane 0", o.rs1[0], 0x1230);
  expect("STORE cluster rs1 lane 1", o.rs1[1], 0xC0002040ull);
  expect("STORE cluster rs1 lane 2", o.rs1[2], 4096);
  expect("STORE cluster rs1 lane 3", o.rs1[3], 0xC0002008ull);
  expect("STORE cluster ctl", o.rs2[0], (uint64_t)(3 | (2 << 3) | (1 << 7)) << 32);
  dr::Store u = dr::unpack_store(o);
  expect("STORE cluster round trip", u.kind == s.kind && u.op == s.op && u.type == s.type && u.src == s.src
         && u.dst == s.dst && u.mbar == s.mbar && u.size == s.size, 1);
  dr::Store t;
  t.kind = dr::ST_RED_TENSOR;
  t.op = dr::RED_ADD;
  t.type = dr::RT_BF16;
  t.slot = 15;
  t.src = 0x400;
  const int32_t c[5] = {-1, 2147483647, 3, -2147483647 - 1, -9};
  for (uint32_t i = 0; i < 5; ++i)
    t.q.coord[i] = c[i];
  o = dr::pack_store(t);
  expect("STORE tensor rs1 lane 1", o.rs1[1], 0x7FFFFFFFFFFFFFFFull);
  expect("STORE tensor rs1 lane 2", o.rs1[2], 0x8000000000000003ull);
  expect("STORE tensor rs1 lane 3", o.rs1[3], 0);
  expect("STORE tensor rs2 lane 0", o.rs2[0], ((uint64_t)(4 | (0 << 3) | (7 << 7) | (15 << 11)) << 32) | 0xFFFFFFF7u);
  u = dr::unpack_store(o);
  bool ok = u.kind == t.kind && u.op == t.op && u.type == t.type && u.slot == 15 && u.src == 0x400;
  for (uint32_t i = 0; i < 5; ++i)
    ok &= u.q.coord[i] == c[i];
  expect("STORE tensor round trip", ok, 1);
  dr::Store b;
  b.kind = dr::ST_BULK;
  b.src = 0x80;
  b.dst = 0x80001000ull;
  b.size = 65536;
  o = dr::pack_store(b);
  expect("STORE bulk lanes", o.rs1[1] == 0x80001000ull && o.rs1[2] == 65536 && o.rs1[3] == 0 && o.rs2[0] == 0, 1);
  expect("ctl field limits", dr::st_ctl(7, 15, 15, 15), 0x7FFFu);
  expect("commit_group word", dr::insn_bgop(dr::BG_COMMIT, 0), 0x0600600Bu);
  expect("wait_group 3 word", dr::insn_bgop(dr::BG_WAIT, 3), 0x0630E00Bu);
  expect("wait_group.read 31 word", dr::insn_bgop(dr::BG_WAIT_READ, 31), 0x07F1600Bu);
  expect("fence.proxy.async.shared word", dr::insn_pfence(dr::PFENCE_SHARED), 0x0200705Bu);
  expect("fence.proxy.async.global word", dr::insn_pfence(dr::PFENCE_GLOBAL), 0x0200F05Bu);
  expect("fence.proxy.async word", dr::insn_pfence(dr::PFENCE_ALL), 0x0201705Bu);
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
  stores();
  legality();
  amo_map();
  reductions();
  store_encodings();
  std::printf("dxa_ref: %d checks passed, %d failed\n", g_pass, g_fail);
  if (g_fail) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
