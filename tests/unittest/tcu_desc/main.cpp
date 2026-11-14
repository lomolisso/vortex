// tcu_desc_ref against the H100: the descriptor bit fields and the asserts;
// the descriptors CuTe and DeepGEMM built for the DAK and DeepGEMM kernels,
// with the shared-memory layout rows of their operands; the accumulator
// fragment rows CuTe printed; and, for every swizzle mode, both majors and
// 1-, 2- and 4-byte elements, tiles written box by box by the DXA tensor-load
// model (dxa_ref) and read back through operand_addr k-step by k-step, which
// must give back the tensor at bases inside and beyond the first 64 KB.

#include <tcu_desc_ref.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

namespace td = vortex::tcu_desc_ref;
namespace dr = vortex::dxa_ref;

namespace {

int g_fail = 0, g_pass = 0;

bool expect(const char* what, uint64_t got, uint64_t want) {
  if (got != want) {
    if (g_fail < 50)
      std::printf("FAIL %s: got %#" PRIx64 ", expected %#" PRIx64 "\n", what, got, want);
    ++g_fail;
    return false;
  }
  ++g_pass;
  return true;
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

uint32_t sw128(uint32_t x) {
  return x ^ (((x >> 7) & 7u) << 4);
}

uint64_t rng_state = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

// Each field set to all ones gives the bits cute::GmmaDescriptor sets.
void bitfields() {
  td::Desc d;
  d.start = 0x3FFF0;
  expect("start all ones", td::encode(d), 0x0000000000003fffull);
  d = td::Desc();
  d.lbo = 0x3FFF0;
  expect("LBO all ones", td::encode(d), 0x000000003fff0000ull);
  d = td::Desc();
  d.sbo = 0x3FFF0;
  expect("SBO all ones", td::encode(d), 0x00003fff00000000ull);
  d = td::Desc();
  d.base_offset = 7;
  expect("base offset all ones", td::encode(d), 0x000e000000000000ull);
  d = td::Desc();
  d.layout = 3;
  expect("layout all ones", td::encode(d), 0xc000000000000000ull);
  expect("transpose flag", td::encode(0, 0, 0, td::LAYOUT_NONE, true), 1ull << 52);
  expect("reserved mask", td::RESERVED_MASK, 0x3FE1C000C000C000ull);
  // Fields keep 18 address bits: bit 18 and up drop, bits 3:0 drop.
  expect("start masked to 256 KB", td::encode(0x4000F + 0x10, 0, 0, 0), 0x0001ull);
  expect("cluster rank bits dropped", td::encode((3u << 18) | 0x500, 0, 0, 0), 0x50ull);

  for (int i = 0; i < 2000; ++i) {
    const uint64_t v = rnd() & td::DEFINED_MASK;
    const td::Desc x = td::decode(v);
    expect("decode/encode round trip", td::encode(x), v);
    expect("decoded fields are 16-B granular", (x.start | x.lbo | x.sbo) & 15u, 0);
    expect("decoded fields stay below 256 KB", (x.start | x.lbo | x.sbo) >> 18, 0);
    expect("transpose flag decodes", x.mn_major, td::is_mn_major(v));
  }
  for (uint32_t b = 0; b < 64; ++b) {
    const uint64_t v = 1ull << b;
    char what[64];
    std::snprintf(what, sizeof what, "reserved bit %u", b);
    const bool reserved = (td::RESERVED_MASK >> b) & 1u;
    expect(what, td::decode(v).reserved != 0, reserved);
  }
}

void asserts() {
  const uint64_t ok = td::encode(0x500, 16, 1024, td::LAYOUT_128B);
  expect_msg("legal K-major f16", td::check(ok, 2), nullptr);
  expect_msg("legal K-major e4m3", td::check(ok, 1), nullptr);
  expect_msg("legal K-major tf32", td::check(ok, 4), nullptr);
  expect_msg("legal MN-major f16", td::check(ok | td::MN_MAJOR, 2), nullptr);
  expect_msg("MN-major e4m3", td::check(ok | td::MN_MAJOR, 1), td::MSG_MN_TYPE);
  expect_msg("MN-major tf32", td::check(ok | td::MN_MAJOR, 4), td::MSG_MN_TYPE);
  expect_msg("base offset 2", td::check(ok | (2ull << 49), 2), td::MSG_BASE_OFFSET);
  expect_msg("reserved bit 14", td::check(ok | (1ull << 14), 2), td::MSG_RESERVED);
  expect_msg("reserved bit 61", td::check(ok | (1ull << 61), 2), td::MSG_RESERVED);
  // Order: reserved bits, then the base offset, then the type.
  expect_msg("reserved before base offset", td::check(ok | (1ull << 47) | (1ull << 49), 2), td::MSG_RESERVED);
  expect_msg("base offset before type", td::check(ok | td::MN_MAJOR | (1ull << 50), 1), td::MSG_BASE_OFFSET);
  expect("text: base offset", std::strcmp(td::MSG_BASE_OFFSET, "wgmma descriptor base offset must be 0"), 0);
  expect("text: MN-major type", std::strcmp(td::MSG_MN_TYPE, "wgmma MN-major needs a 16-bit type"), 0);
  // The two swizzle encodings.
  const uint32_t layouts[4] = {td::LAYOUT_NONE, td::LAYOUT_128B, td::LAYOUT_64B, td::LAYOUT_32B};
  const uint32_t spans[4] = {0, 128, 64, 32};
  for (int i = 0; i < 4; ++i) {
    expect("layout -> DXA swizzle span", dr::swizzle_span(td::dxa_swizzle(layouts[i])), spans[i]);
    expect("DXA swizzle -> layout", td::layout_of_dxa(td::dxa_swizzle(layouts[i])), layouts[i]);
    expect("chunks per swizzle row", td::swizzle_chunks(layouts[i]) * 16, spans[i] ? spans[i] : 16);
  }
}

// The descriptors of the two H100 kernels (start relative to each buffer).
void step0_descriptors() {
  // DAK, SM90_64x256x16_F16F16F16_SS: A MN-major, B K-major, both 128B.
  const uint64_t dak_a[4] = {0x4000004000000000ull, 0x4000004000000080ull, 0x4000004000000100ull, 0x4000004000000180ull};
  const uint64_t dak_b[4] = {0x4000004000010000ull, 0x4000004000010002ull, 0x4000004000010004ull, 0x4000004000010006ull};
  for (uint32_t s = 0; s < 4; ++s) {
    expect("DAK A k-step", td::encode(0x800 * s, 0, 1024, td::LAYOUT_128B), dak_a[s]);
    expect("DAK B k-step", td::encode(0x20 * s, 16, 1024, td::LAYOUT_128B), dak_b[s]);
    expect("DAK B slice at 0x4000", td::encode(0x4000 + 0x20 * s, 16, 1024, td::LAYOUT_128B), dak_b[s] + 0x400);
    const td::Desc a = td::decode(dak_a[s]);
    expect("DAK A start", a.start, 0x800 * s);
    expect("DAK A LBO", a.lbo, 0);
    expect("DAK A SBO", a.sbo, 1024);
    expect("DAK A layout", a.layout, td::LAYOUT_128B);
    expect("DAK A base offset", a.base_offset, 0);
    expect_msg("DAK A legal (with the transpose flag)", td::check(dak_a[s] | td::MN_MAJOR, 2), nullptr);
    expect_msg("DAK B legal", td::check(dak_b[s], 2), nullptr);
  }
  // DeepGEMM make_smem_desc(ptr, 128B, LBO 0, SBO 1024), and CuTe's on the same regions (LBO 16).
  const uint64_t dg_a0[4] = {0x4000004000000c00ull, 0x4000004000000c02ull, 0x4000004000000c04ull, 0x4000004000000c06ull};
  const uint64_t dg_a1[4] = {0x4000004000000e00ull, 0x4000004000000e02ull, 0x4000004000000e04ull, 0x4000004000000e06ull};
  const uint64_t dg_b[4]  = {0x4000004000001c00ull, 0x4000004000001c02ull, 0x4000004000001c04ull, 0x4000004000001c06ull};
  for (uint32_t s = 0; s < 4; ++s) {
    expect("DeepGEMM A wg0", td::encode(0xc000 + 32 * s, 0, 1024, td::LAYOUT_128B), dg_a0[s]);
    expect("DeepGEMM A wg1", td::encode(0xe000 + 32 * s, 0, 1024, td::LAYOUT_128B), dg_a1[s]);
    expect("DeepGEMM B", td::encode(0x1c000 + 32 * s, 0, 1024, td::LAYOUT_128B), dg_b[s]);
  }
  expect("CuTe A wg0 k0", td::encode(0xc000, 16, 1024, td::LAYOUT_128B), 0x4000004000010c00ull);
  expect("CuTe A wg0 k1", td::encode(0xc020, 16, 1024, td::LAYOUT_128B), 0x4000004000010c02ull);
  expect("CuTe B k0", td::encode(0x1c000, 16, 1024, td::LAYOUT_128B), 0x4000004000011c00ull);
  expect("CuTe B k1", td::encode(0x1c020, 16, 1024, td::LAYOUT_128B), 0x4000004000011c02ull);
  // A stage advances the start field by the stage bytes / 16.
  expect("DeepGEMM A stage 1", td::encode(0xc000 + 16384, 0, 1024, td::LAYOUT_128B), dg_a0[0] + 16384 / 16);
  expect("DeepGEMM B stage 1", td::encode(0x1c000 + 24576, 0, 1024, td::LAYOUT_128B), dg_b[0] + 24576 / 16);
}

// The operand layouts the kernels' TMA boxes write, element by element.
void step0_layouts() {
  char what[96];
  // DAK sA(m, k) = sw128(k*128 + 2m) (64 x 64 f16, MN-major) and
  // sB(n, k) = sw128(n*128 + 2k) (256 x 64 f16, K-major), both at a buffer
  // start of phase 0x500 (not 1024-aligned) as on the H100.
  for (uint32_t base : {0u, 0x500u, 0x4500u}) {
    for (uint32_t s = 0; s < 4; ++s) {
      const uint64_t da = td::encode(base + 0x800 * s, 0, 1024, td::LAYOUT_128B, true);
      const uint64_t db = td::encode(base + 0x20 * s, 16, 1024, td::LAYOUT_128B);
      uint32_t bad_a = 0, bad_b = 0;
      for (uint32_t k = 0; k < 16; ++k) {
        const uint32_t kk = 16 * s + k;
        for (uint32_t m = 0; m < 64; ++m)
          bad_a += td::operand_addr(da, true, 2, m, k) != sw128(base + kk * 128 + 2 * m);
        for (uint32_t n = 0; n < 256; ++n)
          bad_b += td::operand_addr(db, false, 2, n, k) != sw128(base + n * 128 + 2 * kk);
      }
      std::snprintf(what, sizeof what, "DAK sA k-step %u at %#x", s, base);
      expect(what, bad_a, 0);
      std::snprintf(what, sizeof what, "DAK sB k-step %u at %#x", s, base);
      expect(what, bad_b, 0);
    }
  }
  // DeepGEMM: A (128 x 128 e4m3, K-major 128B) at +0xc000, one m64 half per
  // math warpgroup; B (192 x 128) at +0x1c000. LBO 0 and CuTe's 16 read alike.
  for (uint32_t s = 0; s < 4; ++s) {
    uint32_t bad = 0;
    for (uint32_t lbo : {0u, 16u}) {
      for (uint32_t wg = 0; wg < 2; ++wg) {
        const uint64_t da = td::encode(0xc000 + wg * 0x2000 + 32 * s, lbo, 1024, td::LAYOUT_128B);
        for (uint32_t m = 0; m < 64; ++m)
          for (uint32_t k = 0; k < 32; ++k)
            bad += td::operand_addr(da, false, 1, m, k) != 0xc000 + sw128((64 * wg + m) * 128 + 32 * s + k);
      }
      const uint64_t db = td::encode(0x1c000 + 32 * s, lbo, 1024, td::LAYOUT_128B);
      for (uint32_t n = 0; n < 192; ++n)
        for (uint32_t k = 0; k < 32; ++k)
          bad += td::operand_addr(db, false, 1, n, k) != 0x1c000 + sw128(n * 128 + 32 * s + k);
    }
    std::snprintf(what, sizeof what, "DeepGEMM A/B k-step %u", s);
    expect(what, bad, 0);
  }
}

// Rows of the accumulator tables CuTe printed: DAK m64n256 f16 (packed) and
// DeepGEMM m64n192 e4m3 -> f32.
void fragments() {
  struct Row { uint32_t t, i, m, n; };
  // DAK: element i = 4j + 2h + e lives in packed register i/2, half i%2.
  const Row dak[] = {
    {0, 0, 0, 0}, {0, 1, 0, 1}, {0, 2, 8, 0}, {0, 3, 8, 1}, {0, 4, 0, 8}, {0, 5, 0, 9}, {0, 126, 8, 248}, {0, 127, 8, 249},
    {1, 0, 0, 2}, {1, 1, 0, 3}, {1, 2, 8, 2}, {1, 3, 8, 3}, {1, 4, 0, 10}, {1, 5, 0, 11}, {1, 126, 8, 250}, {1, 127, 8, 251},
    {5, 0, 1, 2}, {5, 1, 1, 3}, {5, 2, 9, 2}, {5, 3, 9, 3}, {5, 4, 1, 10}, {5, 5, 1, 11}, {5, 126, 9, 250}, {5, 127, 9, 251},
    {33, 0, 16, 2}, {33, 1, 16, 3}, {33, 2, 24, 2}, {33, 3, 24, 3}, {33, 4, 16, 10}, {33, 5, 16, 11}, {33, 126, 24, 250}, {33, 127, 24, 251},
    {127, 0, 55, 6}, {127, 1, 55, 7}, {127, 2, 63, 6}, {127, 3, 63, 7}, {127, 4, 55, 14}, {127, 5, 55, 15}, {127, 126, 63, 254}, {127, 127, 63, 255},
  };
  for (const auto& r : dak) {
    expect("DAK fragment row", td::acc_row(r.t, r.i / 2, true), r.m);
    expect("DAK fragment column", td::acc_col(r.t, r.i / 2, r.i % 2, true), r.n);
  }
  const Row dg[] = {
    {0, 0, 0, 0}, {0, 1, 0, 1}, {0, 2, 8, 0}, {0, 3, 8, 1}, {0, 4, 0, 8}, {0, 94, 8, 184}, {0, 95, 8, 185},
    {1, 0, 0, 2}, {1, 1, 0, 3}, {1, 2, 8, 2}, {1, 3, 8, 3}, {1, 4, 0, 10}, {1, 94, 8, 186}, {1, 95, 8, 187},
    {5, 0, 1, 2}, {5, 1, 1, 3}, {5, 2, 9, 2}, {5, 3, 9, 3}, {5, 4, 1, 10}, {5, 94, 9, 186}, {5, 95, 9, 187},
    {33, 0, 16, 2}, {33, 1, 16, 3}, {33, 2, 24, 2}, {33, 3, 24, 3}, {33, 4, 16, 10}, {33, 94, 24, 186}, {33, 95, 24, 187},
    {127, 0, 55, 6}, {127, 1, 55, 7}, {127, 2, 63, 6}, {127, 3, 63, 7}, {127, 4, 55, 14}, {127, 94, 63, 190}, {127, 95, 63, 191},
  };
  for (const auto& r : dg) {
    expect("DeepGEMM fragment row", td::acc_row(r.t, r.i, false), r.m);
    expect("DeepGEMM fragment column", td::acc_col(r.t, r.i, 0, false), r.n);
  }
  // Each (m, n) of m64 x N has exactly one slot, and acc_slot inverts it.
  for (bool packed : {false, true}) {
    for (uint32_t n : {16u, 32u, 64u, 128u, 256u}) {
      const uint32_t nr = td::acc_regs(n, packed);
      std::vector<int> hit(64 * n, 0);
      uint32_t bad = 0;
      for (uint32_t t = 0; t < 128; ++t) {
        for (uint32_t r = 0; r < nr; ++r) {
          for (uint32_t h = 0; h < (packed ? 2u : 1u); ++h) {
            const uint32_t m = td::acc_row(t, r, packed), c = td::acc_col(t, r, h, packed);
            if (m >= 64 || c >= n) {
              ++bad;
              continue;
            }
            ++hit[m * n + c];
            uint32_t t2, r2, h2;
            td::acc_slot(m, c, packed, &t2, &r2, &h2);
            bad += (t2 != t || r2 != r || h2 != h);
          }
        }
      }
      for (int x : hit)
        bad += (x != 1);
      expect(packed ? "packed f16 fragment is a bijection" : "fp32 fragment is a bijection", bad, 0);
    }
  }
  expect("K of f16", td::inst_k(2), 16);
  expect("K of e4m3", td::inst_k(1), 32);
}

///////////////////////////////////////////////////////////////////////////////
// TMA-written tiles read back through the descriptors.

constexpr uint32_t kImg = 1u << 18;            // one block's 256 KB
constexpr uint64_t kGlobal = 0x10000000ull;

struct Case {
  uint32_t layout;
  bool mn_major;
  uint32_t esize;
  uint32_t MN, K;
};

const char* layout_name(uint32_t l) {
  return l == td::LAYOUT_128B ? "128B" : l == td::LAYOUT_64B ? "64B" : l == td::LAYOUT_32B ? "32B" : "none";
}

// Writes an MN x K operand tile at base with one tensor-load box per swizzle
// atom (8 rows of W 16-B chunks), atoms MN first then K (CuTe tile_to_shape),
// and checks every k-step's descriptor against the tensor.
void tma_tile(const Case& c, uint32_t base) {
  const uint32_t W = td::swizzle_chunks(c.layout);
  const uint32_t E = c.esize;
  const uint32_t atom_el = W * 16 / E;          // elements of an atom row
  const uint32_t atom_b = 8 * W * 16;           // bytes of an atom
  const uint32_t ksteps = c.K / (32 / E);
  char what[160];
  std::snprintf(what, sizeof what, "%s %s esize %u tile %ux%u at %#x", c.mn_major ? "MN" : "K ",
                layout_name(c.layout), E, c.MN, c.K, base);

  // The tensor in global memory: K-major [MN][K], MN-major [K][MN], k or mn innermost.
  const uint32_t inner = c.mn_major ? c.MN : c.K, outer = c.mn_major ? c.K : c.MN;
  std::vector<uint8_t> g((size_t)inner * outer * E);
  for (auto& b : g)
    b = (uint8_t)rnd();
  auto elem_at = [&](uint32_t mn, uint32_t k) -> const uint8_t* {
    const size_t i = c.mn_major ? (size_t)k * c.MN + mn : (size_t)mn * c.K + k;
    return &g[i * E];
  };
  auto read = [&](uint64_t a) -> uint8_t { return g[(size_t)(a - kGlobal)]; };

  std::vector<uint8_t> img(kImg, 0xA5);
  dr::Desc d;
  d.base = kGlobal;
  d.rank = 2;
  d.elem = E;
  d.layout = dr::LAYOUT_ROW;
  d.swizzle = td::dxa_swizzle(c.layout);
  d.size[0] = inner;
  d.size[1] = outer;
  d.stride[0] = inner * E;
  d.tile[0] = atom_el;
  d.tile[1] = 8;
  const uint32_t n_in = inner / atom_el, n_out = outer / 8;
  bool box_ok = true;
  for (uint32_t o = 0; o < n_out; ++o) {
    for (uint32_t i = 0; i < n_in; ++i) {
      dr::Req q;
      q.coord[0] = (int32_t)(i * atom_el);
      q.coord[1] = (int32_t)(o * 8);
      // K-major: atom (row group o, k atom i); MN-major: atom (mn atom i, k group o).
      const uint32_t idx = c.mn_major ? i + o * n_in : o + i * n_out;
      const uint32_t dst = base + idx * atom_b;
      box_ok &= (dr::check_tensor(d, dst) == nullptr);
      dr::render(d, q, dst, read, img.data(), img.size());
    }
  }
  if (!expect("TMA boxes legal", box_ok, 1))
    std::printf("  in %s\n", what);

  uint32_t bad = 0, dup = 0, chunk_bad = 0;
  const uint32_t KI = 32 / E;                   // K of one instruction
  for (uint32_t s = 0; s < ksteps; ++s) {
    const uint32_t k0 = s * KI;
    uint32_t start, lbo, sbo;
    if (!c.mn_major) {
      start = base + (k0 / atom_el) * c.MN / 8 * atom_b + (k0 % atom_el) * E;
      sbo = atom_b;
      lbo = (c.layout == td::LAYOUT_NONE) ? c.MN / 8 * 128 : (uint32_t)(rnd() & 0x3FFF0);
    } else {
      start = base + (k0 / 8) * (c.MN / atom_el) * atom_b;
      if (c.layout == td::LAYOUT_NONE) {
        sbo = 128;
        lbo = c.MN / 8 * 128;
      } else {
        lbo = (c.MN == atom_el && (s & 1)) ? 0 : atom_b;   // CuTe writes 0 for a single atom
        sbo = (c.MN / atom_el) * atom_b;
      }
    }
    const uint64_t desc = td::encode(start, lbo, sbo, c.layout, c.mn_major);
    if (td::check(desc, E) != nullptr) {
      ++bad;
      continue;
    }
    std::set<uint32_t> seen;
    for (uint32_t mn = 0; mn < c.MN; ++mn) {
      for (uint32_t k = 0; k < KI; ++k) {
        const uint32_t a = td::operand_addr(desc, c.mn_major, E, mn, k);
        if (a + E > kImg || std::memcmp(&img[a], elem_at(mn, k0 + k), E) != 0)
          ++bad;
        dup += !seen.insert(a).second;
      }
    }
    // Granules: 16 contiguous bytes from chunk_addr.
    const uint32_t T = 16 / E;
    const uint32_t lines = c.mn_major ? KI : c.MN, chunks = c.mn_major ? c.MN / T : KI / T;
    for (uint32_t l = 0; l < lines; ++l) {
      for (uint32_t ch = 0; ch < chunks; ++ch) {
        const uint32_t a = td::chunk_addr(desc, c.mn_major, E, l, ch);
        chunk_bad += (a & 15u) != 0;
        for (uint32_t e = 0; e < T; ++e) {
          const uint32_t mn = c.mn_major ? ch * T + e : l, k = c.mn_major ? l : ch * T + e;
          chunk_bad += td::operand_addr(desc, c.mn_major, E, mn, k) != a + e * E;
        }
      }
    }
  }
  if (!expect("tile read back through the descriptors", bad, 0))
    std::printf("  in %s\n", what);
  if (!expect("one address per element", dup, 0))
    std::printf("  in %s\n", what);
  if (!expect("granules contiguous", chunk_bad, 0))
    std::printf("  in %s\n", what);
}

void tma_tiles() {
  const uint32_t layouts[4] = {td::LAYOUT_128B, td::LAYOUT_64B, td::LAYOUT_32B, td::LAYOUT_NONE};
  const uint32_t bases[5] = {0, 0x180, 0x500, 0x10080, 0x2F000};
  uint32_t n = 0;
  for (uint32_t layout : layouts) {
    for (uint32_t base : bases) {
      // K-major: 1-, 2- and 4-byte elements, m64 (A) and n up to 256 (B).
      for (uint32_t E : {1u, 2u, 4u}) {
        for (uint32_t MN : {64u, 128u, 256u}) {
          const uint32_t K = (E == 4) ? 32 : 128 / E;
          tma_tile({layout, false, E, MN, K}, base);
          ++n;
        }
      }
      // MN-major: 16-bit only; one atom and several atoms along MN.
      for (uint32_t MN : {64u, 128u, 256u}) {
        const uint32_t atom_el = td::swizzle_chunks(layout) * 8;
        if (MN % atom_el)
          continue;
        tma_tile({layout, true, 2, MN, 64}, base);
        ++n;
      }
      if (layout != td::LAYOUT_NONE) {
        tma_tile({layout, true, 2, td::swizzle_chunks(layout) * 8, 32}, base);
        ++n;
      }
    }
  }
  std::printf("tcu_desc: %u TMA-written tiles\n", n);
}

// LBO is not used by swizzled K-major descriptors, and the start's low four
// bits are dropped.
void lbo_ignored() {
  uint32_t bad = 0;
  for (uint32_t layout : {td::LAYOUT_128B, td::LAYOUT_64B, td::LAYOUT_32B}) {
    for (int i = 0; i < 200; ++i) {
      const uint32_t start = (uint32_t)rnd() & 0x3FFF0, sbo = (uint32_t)rnd() & 0x3FF0;
      const uint64_t a = td::encode(start, 16, sbo, layout), b = td::encode(start, (uint32_t)rnd(), sbo, layout);
      for (uint32_t mn = 0; mn < 64; mn += 7)
        for (uint32_t k = 0; k < 16; ++k)
          bad += td::operand_addr(a, false, 2, mn, k) != td::operand_addr(b, false, 2, mn, k);
    }
  }
  expect("LBO ignored for swizzled K-major", bad, 0);
  expect("start low bits dropped", td::encode(0x1237, 0, 0, 0), td::encode(0x1230, 0, 0, 0));
}

} // namespace

int main() {
  bitfields();
  asserts();
  step0_descriptors();
  step0_layouts();
  fragments();
  lbo_ignored();
  tma_tiles();
  std::printf("tcu_desc: %d checks passed, %d failed\n", g_pass, g_fail);
  if (g_fail) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
