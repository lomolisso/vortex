// The tensor-map host API (sw/runtime/include/dxa.h) and layout
// (sw/common/dxa_tmap.h) against the CUDA driver API rules: every
// cuTensorMapEncodeTiled / cuTensorMapEncodeIm2col / cuTensorMapReplaceAddress
// rule rejected with its exact code and accepted at its bounds, a map left
// unchanged by an error, every encoding word for word against a packer written
// here from the layout table alone, the slot decoder (dxa_ref::decode) reading
// the same descriptor from words 0-21, and the tensormap.replace helpers: each
// field round-trips, and a zero buffer plus a full replace sequence equals the
// encoder's map.

#include <dxa.h>
#include <dxa_ref.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace dh = vortex::dxa;
namespace dt = vortex::dxa_tmap;
namespace dr = vortex::dxa_ref;

using DT = dh::TensorMapDataType;
using IL = dh::TensorMapInterleave;
using SW = dh::TensorMapSwizzle;
using L2 = dh::TensorMapL2Promotion;
using OOB = dh::TensorMapFloatOOBfill;

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

///////////////////////////////////////////////////////////////////////////////
// The golden packer, from the layout table: words 0-1 address, 2-6 dims,
// 7-10 strides 31:0, 11 META (rank [2:0], log2 element bytes [4:3], layout
// [6:5], swizzle [8:7], interleave [10:9], L2 promotion [12:11], im2col [13]),
// 12-16 element strides, 17-19 box or im2col fields, 20 fill word, 21 0, 22
// strides 39:32, 23 {fill mode [4], PTX element type [3:0]}, 24-31 0.

// CUtensorMapDataType -> PTX Table 33 code and bytes.
const uint32_t kPtxCode[13] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 8, 11, 12};
const uint32_t kBytes[13]   = {1, 2, 4, 4, 8, 8, 2, 4, 8, 2, 4, 4, 4};

struct Spec {
  DT type = DT::Float32;
  uint32_t rank = 2;
  uint64_t gaddr = 0x10000;
  uint64_t gdim[5] = {64, 32, 1, 1, 1};
  uint64_t gstride[4] = {256, 0, 0, 0};
  uint32_t box[5] = {16, 8, 1, 1, 1};
  uint32_t es[5] = {1, 1, 1, 1, 1};
  IL il = IL::None;
  SW swz = SW::None;
  L2 l2 = L2::None;
  OOB oob = OOB::None;
  bool im2col = false;
  int32_t lower[3] = {0, 0, 0};
  int32_t upper[3] = {0, 0, 0};
  uint32_t channels = 16;
  uint32_t pixels = 32;
};

uint32_t log2u(uint32_t v) {
  uint32_t n = 0;
  while ((1u << n) < v)
    ++n;
  return n;
}

void golden(const Spec& s, uint32_t* g) {
  std::memset(g, 0, 32 * 4);
  const uint32_t t = static_cast<uint32_t>(s.type);
  const uint32_t bytes = kBytes[t];
  g[0] = (uint32_t)s.gaddr;
  g[1] = (uint32_t)(s.gaddr >> 32);
  for (uint32_t i = 0; i < s.rank; ++i)
    g[2 + i] = (uint32_t)s.gdim[i];
  for (uint32_t i = 0; i + 1 < s.rank; ++i)
    g[7 + i] = (uint32_t)s.gstride[i];
  g[11] = s.rank | (log2u(bytes) << 3) | (static_cast<uint32_t>(s.swz) << 7)
        | (static_cast<uint32_t>(s.il) << 9) | (static_cast<uint32_t>(s.l2) << 11) | ((s.im2col ? 1u : 0u) << 13);
  for (uint32_t i = 0; i < 5; ++i)
    g[12 + i] = i < s.rank ? s.es[i] : 0;
  if (s.il == IL::None)
    g[12] = 1;
  if (s.im2col) {
    g[17] = (s.pixels << 16) | s.channels;
    const uint32_t S = s.rank - 2, b = S == 1 ? 16 : S == 2 ? 8 : 5, m = (1u << b) - 1;
    for (uint32_t k = 0; k < S; ++k)
      g[18] |= (((uint32_t)s.lower[k] & m) << (k * b)) | (((uint32_t)s.upper[k] & m) << ((S + k) * b));
  } else {
    g[17] = (s.rank > 1 ? s.box[1] << 16 : 0) | s.box[0];
    g[18] = (s.rank > 3 ? s.box[3] << 16 : 0) | (s.rank > 2 ? s.box[2] : 0);
    g[19] = s.rank > 4 ? s.box[4] : 0;
  }
  const bool nan = s.oob == OOB::NanRequestZeroFma;
  g[20] = !nan ? 0 : bytes == 2 ? 0x7FFF7FFFu : 0x7FFFFFFFu;
  g[23] = kPtxCode[t] | ((nan ? 1u : 0u) << 4);
}

int encode(vx_tensor_map_t* m, const Spec& s) {
  if (s.im2col)
    return dh::encode_im2col(m, s.type, s.rank, s.gaddr, s.gdim, s.gstride, s.lower, s.upper, s.channels,
                             s.pixels, s.es, s.il, s.swz, s.l2, s.oob);
  return dh::encode_tiled(m, s.type, s.rank, s.gaddr, s.gdim, s.gstride, s.box, s.es, s.il, s.swz, s.l2, s.oob);
}

// The map words the slot decoder must read back as this descriptor.
void check_decode(const char* name, const Spec& s, const uint32_t* w) {
  const dr::Desc d = dr::decode(w);
  char what[160];
  auto e = [&](const char* f, uint64_t got, uint64_t want) {
    std::snprintf(what, sizeof(what), "%s decode %s", name, f);
    expect(what, got, want);
  };
  e("base", d.base, s.gaddr);
  e("rank", d.rank, s.rank);
  e("elem", d.elem, kBytes[static_cast<uint32_t>(s.type)]);
  e("swizzle", d.swizzle, static_cast<uint32_t>(s.swz));
  e("interleave", d.interleave, static_cast<uint32_t>(s.il));
  e("im2col", d.im2col, s.im2col);
  for (uint32_t i = 0; i < s.rank; ++i)
    e("size", d.size[i], s.gdim[i]);
  if (s.im2col) {
    e("channels", d.channels, s.channels);
    e("pixels", d.pixels, s.pixels);
    for (uint32_t k = 0; k + 2 < s.rank; ++k) {
      e("lower", (uint32_t)d.lower[k], (uint32_t)s.lower[k]);
      e("upper", (uint32_t)d.upper[k], (uint32_t)s.upper[k]);
    }
  } else {
    for (uint32_t i = 0; i < s.rank; ++i)
      e("tile", d.tile[i], s.box[i]);
  }
  expect_msg(name, dr::check_map_words(w), nullptr);
}

void accept(const char* name, const Spec& s) {
  alignas(64) vx_tensor_map_t m;
  std::memset(&m, 0xA5, sizeof(m));
  uint32_t g[32];
  golden(s, g);
  char what[160];
  std::snprintf(what, sizeof(what), "%s result", name);
  expect(what, (uint64_t)encode(&m, s), VX_SUCCESS);
  for (uint32_t k = 0; k < 32; ++k) {
    std::snprintf(what, sizeof(what), "%s word %u", name, k);
    expect(what, m.w[k], g[k]);
  }
  check_decode(name, s, m.w);
}

void reject(const char* name, const Spec& s, int code, vx_tensor_map_t* at = nullptr) {
  alignas(64) static uint8_t buf[256];
  vx_tensor_map_t* m = at ? at : reinterpret_cast<vx_tensor_map_t*>(buf);
  if ((uintptr_t)m >= (uintptr_t)buf && (uintptr_t)m < (uintptr_t)buf + sizeof(buf))
    std::memset(buf, 0x5A, sizeof(buf));
  char what[160];
  std::snprintf(what, sizeof(what), "%s result", name);
  expect(what, (uint64_t)(uint32_t)encode(m, s), (uint64_t)(uint32_t)code);
  if (code != VX_SUCCESS && (uintptr_t)m >= (uintptr_t)buf && (uintptr_t)m < (uintptr_t)buf + sizeof(buf)) {
    bool same = true;
    for (uint32_t k = 0; k < sizeof(buf); ++k)
      same &= (buf[k] == 0x5A);
    std::snprintf(what, sizeof(what), "%s map unchanged", name);
    expect(what, same, true);
  }
}

Spec tile2d() {
  return Spec();
}

Spec tile(uint32_t rank) {
  Spec s;
  s.rank = rank;
  for (uint32_t i = 0; i < 5; ++i) {
    s.gdim[i] = i < rank ? 8 + i : 1;
    s.box[i] = i == 0 ? 4 : 2;
  }
  s.gstride[0] = 64;
  for (uint32_t i = 1; i < 4; ++i)
    s.gstride[i] = s.gstride[i - 1] * s.gdim[i];
  return s;
}

Spec i2c(uint32_t rank) {
  Spec s = tile(rank);
  s.im2col = true;
  s.type = DT::Float16;
  s.channels = 8;
  s.pixels = 16;
  return s;
}

const int OK = VX_SUCCESS, INV = VX_ERR_INVALID_VALUE, NS = VX_ERR_NOT_SUPPORTED;

///////////////////////////////////////////////////////////////////////////////

void test_layout() {
  expect("sizeof", sizeof(vx_tensor_map_t), 128);
  expect("alignof", alignof(vx_tensor_map_t), 64);
  expect("words 0-21 = slot words", TMAP_W_SMEM_STRIDE + 1, VX_DCR_DXA_DESC_SMEM_STRIDE_OFF + 1);
  expect("META word", TMAP_W_META, VX_DCR_DXA_DESC_META_OFF);
  expect("CFILL word", TMAP_W_CFILL, VX_DCR_DXA_DESC_CFILL_OFF);
  expect("TILESIZE4 word", TMAP_W_TILESIZE4, VX_DCR_DXA_DESC_TILESIZE4_OFF);
  expect("ESTRIDE0 word", TMAP_W_ESTRIDE0, VX_DCR_DXA_DESC_ESTRIDE0_OFF);
  expect("STRIDE0 word", TMAP_W_STRIDE0, VX_DCR_DXA_DESC_STRIDE0_OFF);
  expect("words read", TMAP_FETCH_WORDS, 24);
  expect("entries", DXA_TMAP_ENTRIES, 8);
  // CUDA enum order (driver API) and the PTX element types.
  expect("FLOAT64", static_cast<uint32_t>(DT::Float64), 8);
  expect("FLOAT32_FTZ", static_cast<uint32_t>(DT::Float32Ftz), 10);
  expect("16U6_ALIGN16B", static_cast<uint32_t>(DT::U6x16Align16B), 15);
  expect("128B_ATOM_64B", static_cast<uint32_t>(SW::B128Atom64B), 6);
  expect("L2_256B", static_cast<uint32_t>(L2::L2_256B), 3);
  for (uint32_t t = 0; t < 13; ++t) {
    expect("elem bytes", dt::elem_bytes(kPtxCode[t]), kBytes[t]);
    expect("elem float", dt::elem_float(kPtxCode[t]), t >= 6);
  }
  expect("ptx f32.ftz", TMAP_ET_F32_FTZ, 8);
  expect("ptx f64", TMAP_ET_F64, 9);
  expect("ptx bf16", TMAP_ET_BF16, 10);
}

void test_accept() {
  // Every data type, 1D, box 16 elements; NaN fill on the float types.
  for (uint32_t t = 0; t < 13; ++t) {
    Spec s;
    s.type = static_cast<DT>(t);
    s.rank = 1;
    s.box[0] = 16;
    s.gdim[0] = 1000;
    s.oob = t >= 6 ? OOB::NanRequestZeroFma : OOB::None;
    char name[64];
    std::snprintf(name, sizeof(name), "type %u", t);
    accept(name, s);
  }
  {
    Spec s = tile2d();
    s.swz = SW::B128;
    s.box[0] = 32;  // 128 B = the span
    s.l2 = L2::L2_128B;
    s.oob = OOB::NanRequestZeroFma;
    accept("2d swz128 l2 nan", s);
  }
  for (uint32_t r = 1; r <= 5; ++r) {
    char name[64];
    std::snprintf(name, sizeof(name), "tile rank %u", r);
    accept(name, tile(r));
  }
  {
    Spec s = tile(5);
    s.type = DT::UInt8;
    s.box[0] = 32;
    for (uint32_t i = 1; i < 5; ++i)
      s.es[i] = 2 * i;
    s.es[0] = 8;     // ignored without interleave: stored as 1
    s.il = IL::None;
    accept("5d element strides", s);
  }
  {
    Spec s = tile(3);
    s.type = DT::BFloat16;
    s.il = IL::B32;
    s.swz = SW::B32;
    s.gaddr = 0x20020;
    s.gstride[0] = 96;
    s.gstride[1] = 96 * 9;
    s.es[0] = 2;     // kept with interleave
    s.box[0] = 3;    // no 16-byte rule with interleave
    accept("3d il32", s);
  }
  {
    Spec s = tile(4);
    s.il = IL::B16;
    s.swz = SW::B128;
    s.box[0] = 64;   // any inner box with interleave
    s.l2 = L2::L2_256B;
    accept("4d il16 swz128", s);
  }
  // Bounds.
  {
    Spec s = tile(2);
    s.gdim[0] = (1ull << 32) - 1;
    s.gstride[0] = (1ull << 32) - 16;
    s.box[0] = 256;
    s.box[1] = 256;
    s.es[1] = 8;
    s.type = DT::UInt8;
    s.gaddr = 0xFFFFFFFFFFF0ull;
    accept("bounds", s);
  }
  {
    Spec s = i2c(3);
    s.lower[0] = -32768;
    s.upper[0] = 32767;
    s.channels = 256;
    s.pixels = 1024;
    accept("im2col 3d bounds", s);
  }
  {
    Spec s = i2c(4);
    s.lower[0] = -128;
    s.lower[1] = 127;
    s.upper[0] = 127;
    s.upper[1] = -128 + 1;
    s.gdim[2] = 300;  // the H extent 300 - 127 - 127 = 46 pixels
    s.swz = SW::B64;
    s.channels = 32;  // 64 B = the span
    s.es[1] = 3;
    s.es[2] = 2;
    accept("im2col 4d", s);
  }
  {
    Spec s = i2c(5);
    s.lower[0] = -16;
    s.lower[1] = 15;
    s.lower[2] = -1;
    s.upper[0] = 15;
    s.upper[1] = -16;
    s.upper[2] = 1;
    s.gdim[2] = 32;   // 32 - 16 - 15 = 1 pixel
    s.type = DT::Float32;
    s.oob = OOB::NanRequestZeroFma;
    accept("im2col 5d", s);
  }
}

void test_reject() {
  alignas(64) static uint8_t raw[256];
  {
    Spec s = tile2d();
    reject("map alignment: 16 B off", s, INV, reinterpret_cast<vx_tensor_map_t*>(raw + 16));
    reject("map alignment: 32 B off", s, INV, reinterpret_cast<vx_tensor_map_t*>(raw + 32));
    expect("map alignment: null", (uint32_t)encode(nullptr, s), (uint32_t)INV);
  }
  for (uint32_t r : {0u, 6u}) {
    Spec s = tile2d();
    s.rank = r;
    reject("rank: 0 and 6", s, INV);
  }
  {
    Spec s = tile(2);
    s.il = IL::B16;
    reject("rank: interleave rank 2", s, INV);
    s = tile(3);
    s.il = IL::B16;
    reject("rank: interleave rank 3 ok", s, OK);
  }
  for (uint32_t r : {2u, 6u}) {
    Spec s = i2c(3);
    s.rank = r;
    reject("im2col: im2col rank", s, INV);
  }
  {
    Spec s = tile2d();
    s.gaddr = 0x10008;
    reject("globalAddress: address 16", s, INV);
    s = tile(3);
    s.il = IL::B32;
    s.swz = SW::B32;
    s.gstride[0] = 64;
    s.gstride[1] = 640;
    s.gaddr = 0x10010;
    reject("globalAddress: address 32 with il32", s, INV);
    s.gaddr = 0x10020;
    reject("globalAddress: address 32 with il32 ok", s, OK);
  }
  {
    Spec s = tile2d();
    s.gdim[1] = 0;
    reject("globalDim: dim 0", s, INV);
    s.gdim[1] = (1ull << 32) + 1;
    reject("globalDim: dim 2^32+1", s, INV);
    s.gdim[1] = 1ull << 32;
    reject("globalDim: dim 2^32 (Vortex)", s, NS);
  }
  {
    Spec s = tile2d();
    s.gstride[0] = 72;
    reject("globalStrides: stride 16", s, INV);
    s.gstride[0] = 1ull << 40;
    reject("globalStrides: stride 2^40", s, INV);
    s.gstride[0] = (1ull << 40) - 16;
    reject("globalStrides: stride 2^40-16 (Vortex)", s, NS);
    s.gstride[0] = 1ull << 32;
    reject("globalStrides: stride 2^32 (Vortex)", s, NS);
    s = tile(3);
    s.il = IL::B32;
    s.swz = SW::B32;
    s.gaddr = 0x10020;
    s.gstride[0] = 48;
    s.gstride[1] = 640;
    reject("globalStrides: stride 32 with il32", s, INV);
    s.gstride[0] = 0;
    s.gstride[1] = 640;
    reject("globalStrides: stride 0 ok", s, OK);
  }
  {
    Spec s = tile2d();
    s.box[1] = 0;
    reject("boxDim: box 0", s, INV);
    s.box[1] = 257;
    reject("boxDim: box 257", s, INV);
    s = tile2d();
    s.type = DT::UInt8;
    s.box[0] = 8;
    reject("boxDim: inner 16 B", s, INV);
    s = tile(3);
    s.type = DT::UInt8;
    s.il = IL::B16;
    s.box[0] = 8;
    reject("boxDim: inner with interleave ok", s, OK);
  }
  {
    Spec s = tile2d();
    s.es[1] = 0;
    reject("elementStrides: estride 0", s, INV);
    s.es[1] = 9;
    reject("elementStrides: estride 9", s, INV);
    s.es[1] = 1;
    s.es[0] = 9;
    reject("elementStrides: estride[0] 9 (ignored, still checked)", s, INV);
    s.es[0] = 0;
    reject("elementStrides: estride[0] 0", s, INV);
  }
  {
    Spec s = tile2d();
    s.box[0] = 16;   // 64 B
    s.swz = SW::B32;
    reject("swizzle span: inner > 32B span", s, INV);
    s.swz = SW::B64;
    reject("swizzle span: inner = 64B span", s, OK);
    s.box[0] = 64;   // 256 B
    s.swz = SW::B128;
    reject("swizzle span: inner > 128B span", s, INV);
    s = tile(3);
    s.il = IL::B32;
    s.gaddr = 0x10020;
    s.gstride[0] = 64;
    s.gstride[1] = 640;
    s.swz = SW::None;
    reject("swizzle span: il32 needs swz32 (none)", s, INV);
    s.swz = SW::B64;
    reject("swizzle span: il32 needs swz32 (64)", s, INV);
    s = i2c(3);
    s.channels = 32;  // f16: 64 B
    s.swz = SW::B32;
    reject("swizzle span: im2col channels > span", s, INV);
    s.swz = SW::B64;
    reject("swizzle span: im2col channels = span", s, OK);
  }
  {
    Spec s = tile2d();
    s.type = static_cast<DT>(16);
    reject("enum type 16", s, INV);
    for (uint32_t t : {13u, 14u, 15u}) {
      s.type = static_cast<DT>(t);
      reject("enum packed type (CC 10.0)", s, NS);
    }
    s = tile2d();
    s.il = static_cast<IL>(3);
    reject("enum interleave 3", s, INV);
    s = tile2d();
    s.swz = static_cast<SW>(7);
    reject("enum swizzle 7", s, INV);
    for (uint32_t z : {4u, 5u, 6u}) {
      s.swz = static_cast<SW>(z);
      reject("enum 128B_ATOM (CC 10.0)", s, NS);
    }
    s = tile2d();
    s.l2 = static_cast<L2>(4);
    reject("enum l2 4", s, INV);
    s = tile2d();
    s.oob = static_cast<OOB>(2);
    reject("enum oob 2", s, INV);
    for (uint32_t t = 0; t < 6; ++t) {
      s = tile2d();
      s.type = static_cast<DT>(t);
      s.box[0] = 16;
      s.oob = OOB::NanRequestZeroFma;
      reject("oobFill: NaN on an integer type", s, INV);
    }
  }
  {
    Spec s = i2c(3);
    s.lower[0] = -32769;
    reject("im2col: rank 3 lower", s, INV);
    s.lower[0] = 0;
    s.upper[0] = 32768;
    reject("im2col: rank 3 upper", s, INV);
    s = i2c(4);
    s.lower[1] = -129;
    reject("im2col: rank 4 lower", s, INV);
    s.lower[1] = 0;
    s.upper[0] = 128;
    reject("im2col: rank 4 upper", s, INV);
    s = i2c(5);
    s.upper[2] = 16;
    reject("im2col: rank 5 upper", s, INV);
    s.upper[2] = 0;
    s.lower[2] = -17;
    reject("im2col: rank 5 lower", s, INV);
    s = i2c(4);      // W = 9
    s.lower[0] = 5;
    s.upper[0] = -4;
    reject("im2col: zero-area box", s, INV);
    s.upper[0] = -3;
    reject("im2col: one-pixel box", s, OK);
    s = i2c(3);
    s.channels = 0;
    reject("im2col: channels 0", s, INV);
    s.channels = 257;
    reject("im2col: channels 257", s, INV);
    s.channels = 8;
    s.pixels = 0;
    reject("im2col: pixels 0", s, INV);
    s.pixels = 1025;
    reject("im2col: pixels 1025", s, INV);
    s = i2c(4);
    s.il = IL::B16;
    s.es[0] = 2;
    reject("im2col dim-0 stride with interleave (Vortex)", s, NS);
    s.es[0] = 1;
    reject("im2col interleave ok", s, OK);
  }
  {
    Spec s = tile2d();
    alignas(64) vx_tensor_map_t m;
    expect("null dims", (uint32_t)dh::encode_tiled(&m, s.type, 2, s.gaddr, nullptr, s.gstride, s.box, s.es,
                                                   s.il, s.swz, s.l2, s.oob), (uint32_t)INV);
    expect("null strides", (uint32_t)dh::encode_tiled(&m, s.type, 2, s.gaddr, s.gdim, nullptr, s.box, s.es,
                                                      s.il, s.swz, s.l2, s.oob), (uint32_t)INV);
    expect("null strides rank 1", (uint32_t)dh::encode_tiled(&m, s.type, 1, s.gaddr, s.gdim, nullptr, s.box,
                                                             s.es, s.il, s.swz, s.l2, s.oob), (uint32_t)OK);
    expect("null box", (uint32_t)dh::encode_tiled(&m, s.type, 2, s.gaddr, s.gdim, s.gstride, nullptr, s.es,
                                                  s.il, s.swz, s.l2, s.oob), (uint32_t)INV);
    expect("null estrides", (uint32_t)dh::encode_tiled(&m, s.type, 2, s.gaddr, s.gdim, s.gstride, s.box,
                                                       nullptr, s.il, s.swz, s.l2, s.oob), (uint32_t)INV);
  }
}

void test_replace_address() {
  alignas(64) vx_tensor_map_t m, before;
  Spec s = tile2d();
  encode(&m, s);
  before = m;
  expect("replaceAddress: addr 8", (uint32_t)dh::replace_address(&m, 0x20008), (uint32_t)INV);
  expect("replaceAddress: unchanged", std::memcmp(&m, &before, sizeof(m)), 0);
  expect("replaceAddress: addr 16", (uint32_t)dh::replace_address(&m, 0x123456789A0ull), (uint32_t)OK);
  expect("replaceAddress: lo", m.w[0], 0x456789A0u);
  expect("replaceAddress: hi", m.w[1], 0x123u);
  for (uint32_t k = 2; k < 32; ++k)
    expect("replaceAddress: other words", m.w[k], before.w[k]);
  alignas(64) static uint8_t raw[192];
  expect("map alignment: replace unaligned map",
         (uint32_t)dh::replace_address(reinterpret_cast<vx_tensor_map_t*>(raw + 8), 0x1000), (uint32_t)INV);
  s = tile(3);
  s.il = IL::B32;
  s.swz = SW::B32;
  s.gaddr = 0x10020;
  s.gstride[0] = 64;
  s.gstride[1] = 640;
  expect("il32 map", (uint32_t)encode(&m, s), (uint32_t)OK);
  expect("replaceAddress: il32 addr 16", (uint32_t)dh::replace_address(&m, 0x30010), (uint32_t)INV);
  expect("replaceAddress: il32 addr 32", (uint32_t)dh::replace_address(&m, 0x30020), (uint32_t)OK);
}

void test_set_layout() {
  alignas(64) vx_tensor_map_t m, before;
  Spec s = tile2d();
  s.type = DT::Float16;
  encode(&m, s);
  before = m;
  expect("layout kmajor", (uint32_t)dh::set_layout(&m, dh::Layout::KMajor, 0, 0), (uint32_t)OK);
  expect("layout kmajor META", m.w[TMAP_W_META], before.w[TMAP_W_META] | (1u << 5));
  expect("layout kmajor ESTRIDE2", m.w[TMAP_W_ESTRIDE0 + 2], 0);
  expect("layout flat", (uint32_t)dh::set_layout(&m, dh::Layout::Flat, 16, 32), (uint32_t)OK);
  expect("layout flat META", m.w[TMAP_W_META], before.w[TMAP_W_META] | (2u << 5));
  expect("layout flat ESTRIDE2", m.w[TMAP_W_ESTRIDE0 + 2], (32u << 16) | 16u);
  {
    dr::Desc d = dr::decode(m.w);
    expect("layout flat decode", d.layout, dr::LAYOUT_FLAT);
    expect("layout flat decode geom", d.tc_geom, (32u << 16) | 16u);
  }
  expect("layout block-major", (uint32_t)dh::set_layout(&m, dh::Layout::BlockMajor, 8, 64), (uint32_t)OK);
  expect("layout row", (uint32_t)dh::set_layout(&m, dh::Layout::RowMajor, 0, 0), (uint32_t)OK);
  expect("layout row META", m.w[TMAP_W_META], before.w[TMAP_W_META]);
  expect("layout row ESTRIDE2", m.w[TMAP_W_ESTRIDE0 + 2], 0);
  expect("layout 4", (uint32_t)dh::set_layout(&m, static_cast<dh::Layout>(4), 0, 0), (uint32_t)INV);
  encode(&m, tile(3));
  expect("layout kmajor rank 3", (uint32_t)dh::set_layout(&m, dh::Layout::KMajor, 0, 0), (uint32_t)INV);
  s = tile2d();
  s.swz = SW::B64;
  encode(&m, s);
  expect("layout kmajor swizzle", (uint32_t)dh::set_layout(&m, dh::Layout::KMajor, 0, 0), (uint32_t)INV);
  s = tile2d();
  s.es[1] = 2;
  encode(&m, s);
  expect("layout kmajor strides", (uint32_t)dh::set_layout(&m, dh::Layout::KMajor, 0, 0), (uint32_t)INV);
  encode(&m, i2c(3));
  expect("layout im2col", (uint32_t)dh::set_layout(&m, dh::Layout::KMajor, 0, 0), (uint32_t)INV);
  alignas(64) static uint8_t raw[192];
  expect("layout unaligned map",
         (uint32_t)dh::set_layout(reinterpret_cast<vx_tensor_map_t*>(raw + 4), dh::Layout::RowMajor, 0, 0),
         (uint32_t)INV);
}

// Words that differ between two maps, as a bit mask.
uint32_t diff(const vx_tensor_map_t& a, const vx_tensor_map_t& b) {
  uint32_t m = 0;
  for (uint32_t k = 0; k < 32; ++k)
    m |= (a.w[k] != b.w[k]) ? (1u << k) : 0u;
  return m;
}

void test_replace() {
  Spec s = tile(5);
  s.type = DT::Float32;
  alignas(64) vx_tensor_map_t base, m;
  encode(&base, s);

  m = base;
  dt::replace_global_address(m.w, 0xABCDE0123440ull);
  expect("replace address words", diff(m, base), 0x3u);
  expect("replace address", ((uint64_t)m.w[1] << 32) | m.w[0], 0xABCDE0123440ull);

  for (uint32_t r = 0; r < 5; ++r) {
    m = base;
    dt::replace_rank(m.w, r);
    expect("replace rank", dt::rank(m.w), r + 1);
    expect("replace rank decode", dr::decode(m.w).rank, r + 1);
    expect("replace rank words", diff(m, base) & ~(1u << TMAP_W_META), 0);
  }
  for (uint32_t o = 0; o < 5; ++o) {
    m = base;
    dt::replace_box_dim(m.w, o, 200 + o);
    expect("replace box", dr::decode(m.w).tile[o], 200 + o);
    for (uint32_t i = 0; i < 5; ++i) {
      if (i != o)
        expect("replace box others", dr::decode(m.w).tile[i], dr::decode(base.w).tile[i]);
    }
    m = base;
    dt::replace_global_dim(m.w, o, 0xFFFFFFF0u - o);
    expect("replace dim", dr::decode(m.w).size[o], 0xFFFFFFF0u - o);
    expect("replace dim words", diff(m, base), 1u << (TMAP_W_SIZE0 + o));
    m = base;
    dt::replace_element_stride(m.w, o, 7);
    expect("replace estride", dr::decode(m.w).estride[o], 7);
    expect("replace estride words", diff(m, base), 1u << (TMAP_W_ESTRIDE0 + o));
  }
  for (uint32_t o = 0; o < 4; ++o) {
    m = base;
    const uint64_t v = (0x10ull + o) << 32 | (0x1000u * (o + 1));
    dt::replace_global_stride(m.w, o, v);
    expect("replace stride", dt::global_stride(m.w, o), v);
    expect("replace stride word", m.w[TMAP_W_STRIDE0 + o], (uint32_t)v);
    expect("replace stride hi", m.w[TMAP_W_STRIDE_HI], (0x10u + o) << (8 * o));
    expect_msg("replace stride wide", dr::check_map_words(m.w), dr::MSG_TMAP_STRIDE);
    dt::replace_global_stride(m.w, o, 0x2000);
    expect("replace stride back", m.w[TMAP_W_STRIDE_HI], 0);
    expect_msg("replace stride narrow", dr::check_map_words(m.w), nullptr);
  }
  for (uint32_t t = 0; t < 13; ++t) {
    m = base;
    dt::replace_elemtype(m.w, kPtxCode[t]);
    expect("replace elemtype", dt::elem_type(m.w), kPtxCode[t]);
    expect("replace elemtype size", dr::decode(m.w).elem, kBytes[t]);
    expect("replace elemtype zero fill", m.w[TMAP_W_CFILL], 0);
    dt::replace_fill_mode(m.w, TMAP_FILL_NAN);
    expect("replace fill", m.w[TMAP_W_CFILL], kBytes[t] == 2 ? 0x7FFF7FFFu : 0x7FFFFFFFu);
    vx_tensor_map_t n = base;
    dt::replace_fill_mode(n.w, TMAP_FILL_NAN);
    dt::replace_elemtype(n.w, kPtxCode[t]);
    expect("replace fill then type", diff(m, n), 0);
    dt::replace_fill_mode(m.w, TMAP_FILL_ZERO);
    expect("replace fill zero", m.w[TMAP_W_CFILL], 0);
  }
  for (uint32_t v = 0; v < 3; ++v) {
    m = base;
    dt::replace_interleave_layout(m.w, v);
    expect("replace interleave", dr::decode(m.w).interleave, v);
    expect("replace interleave words", diff(m, base) & ~(1u << TMAP_W_META), 0);
  }
  for (uint32_t v = 0; v < 4; ++v) {
    m = base;
    dt::replace_swizzle_mode(m.w, v);
    expect("replace swizzle", dr::decode(m.w).swizzle, v);
    expect("replace swizzle words", diff(m, base) & ~(1u << TMAP_W_META), 0);
  }

  // A zero buffer and the full replace sequence equal the encoder's map.
  struct Case {
    const char* name;
    Spec s;
  };
  Case cases[4];
  cases[0] = {"zero+replace 2d f32", tile2d()};
  cases[1] = {"zero+replace 5d", tile(5)};
  cases[1].s.type = DT::BFloat16;
  cases[1].s.box[0] = 8;
  cases[1].s.oob = OOB::NanRequestZeroFma;
  for (uint32_t i = 1; i < 5; ++i)
    cases[1].s.es[i] = i + 1;
  cases[2] = {"zero+replace il32", tile(3)};
  cases[2].s.il = IL::B32;
  cases[2].s.swz = SW::B32;
  cases[2].s.gaddr = 0x40020;
  cases[2].s.gstride[0] = 64;
  cases[2].s.gstride[1] = 640;
  cases[2].s.es[0] = 3;
  cases[2].s.type = DT::Float64;
  cases[3] = {"zero+replace swz128 u8", tile2d()};
  cases[3].s.type = DT::UInt8;
  cases[3].s.box[0] = 128;
  cases[3].s.swz = SW::B128;
  for (const Case& c : cases) {
    alignas(64) vx_tensor_map_t e, z;
    expect(c.name, (uint32_t)encode(&e, c.s), (uint32_t)OK);
    std::memset(&z, 0, sizeof(z));
    const Spec& s = c.s;
    dt::replace_global_address(z.w, s.gaddr);
    dt::replace_rank(z.w, s.rank - 1);
    for (uint32_t i = 0; i < s.rank; ++i) {
      dt::replace_box_dim(z.w, i, s.box[i]);
      dt::replace_global_dim(z.w, i, (uint32_t)s.gdim[i]);
    }
    for (uint32_t i = 0; i + 1 < s.rank; ++i)
      dt::replace_global_stride(z.w, i, s.gstride[i]);
    for (uint32_t i = 0; i < s.rank; ++i)
      dt::replace_element_stride(z.w, i, i > 0 || s.il != IL::None ? s.es[i] : 1);
    dt::replace_elemtype(z.w, kPtxCode[static_cast<uint32_t>(s.type)]);
    dt::replace_interleave_layout(z.w, static_cast<uint32_t>(s.il));
    dt::replace_swizzle_mode(z.w, static_cast<uint32_t>(s.swz));
    dt::replace_fill_mode(z.w, static_cast<uint32_t>(s.oob));
    for (uint32_t k = 0; k < 32; ++k) {
      char what[96];
      std::snprintf(what, sizeof(what), "%s word %u", c.name, k);
      expect(what, z.w[k], e.w[k]);
    }
  }

  // A zero buffer is rank 0 until the rank is replaced.
  alignas(64) vx_tensor_map_t z;
  std::memset(&z, 0, sizeof(z));
  expect_msg("zero map", dr::check_map_words(z.w), dr::MSG_TMAP_RANK);
  dt::replace_rank(z.w, 5);
  expect_msg("rank 6", dr::check_map_words(z.w), dr::MSG_TMAP_RANK);
}

void test_red_type() {
  alignas(64) vx_tensor_map_t m;
  std::memset(&m, 0, sizeof(m));
  const struct { uint32_t et, rt; } k[] = {
    {TMAP_ET_U32, DXA_RT_U32}, {TMAP_ET_S32, DXA_RT_S32}, {TMAP_ET_U64, DXA_RT_U64}, {TMAP_ET_S64, DXA_RT_S64},
    {TMAP_ET_F16, DXA_RT_F16}, {TMAP_ET_BF16, DXA_RT_BF16}, {TMAP_ET_F32, DXA_RT_F32},
    {TMAP_ET_F32_FTZ, DXA_RT_F32}, {TMAP_ET_F64, DXA_RT_F64}, {TMAP_ET_U8, TMAP_RT_NONE},
    {TMAP_ET_U16, TMAP_RT_NONE}, {TMAP_ET_TF32, TMAP_RT_NONE}, {TMAP_ET_TF32_FTZ, TMAP_RT_NONE}};
  for (const auto& e : k)
    expect("red type", dt::red_type(e.et), e.rt);
  uint32_t t = 99;
  dt::replace_elemtype(m.w, TMAP_ET_F32);
  expect_msg("f32 add", dr::map_red_type(DXA_RED_ADD, m.w, &t), nullptr);
  expect("f32 add type", t, DXA_RT_F32);
  expect_msg("f32 max", dr::map_red_type(DXA_RED_MAX, m.w, &t), dr::MSG_TMAP_REDTYPE);
  dt::replace_elemtype(m.w, TMAP_ET_U32);
  expect_msg("u32 max", dr::map_red_type(DXA_RED_MAX, m.w, &t), nullptr);
  expect("u32 max type", t, DXA_RT_U32);
  dt::replace_elemtype(m.w, TMAP_ET_U8);
  expect_msg("u8 add", dr::map_red_type(DXA_RED_ADD, m.w, &t), dr::MSG_TMAP_REDTYPE);
  dt::replace_elemtype(m.w, TMAP_ET_S64);
  expect_msg("s64 add", dr::map_red_type(DXA_RED_ADD, m.w, &t), dr::MSG_TMAP_REDTYPE);
  expect_msg("s64 min", dr::map_red_type(DXA_RED_MIN, m.w, &t), nullptr);
}

void test_operands() {
  dr::Req q;
  q.coord[0] = -3;
  q.coord[1] = 7;
  const dr::Ops o = dr::pack_tload_map(0x80040, false, 0x1000, 0x8, q, 0);
  expect("tload map ctl", (uint32_t)(o.rs2[0] >> 32), 1u << 5);
  expect("tload map lane 2", o.rs2[2], 0x80040);
  expect("tload map lane 3", o.rs2[3], 0);
  dr::Store s;
  s.kind = dr::ST_RED_TENSOR;
  s.op = DXA_RED_ADD;
  s.type = DXA_RT_FROM_MAP;
  s.map_form = true;
  s.map = 0x90080;
  s.slot = 5;
  const dr::Ops p = dr::pack_store(s);
  expect("store map ctl", (uint32_t)(p.rs2[0] >> 32),
         DXA_ST_RED_TENSOR | (DXA_RED_ADD << 3) | (15u << 7) | (1u << 15));
  expect("store map lane 1", p.rs2[1], 0x90080);
  const dr::Store u = dr::unpack_store(p);
  expect("store map unpack", u.map_form, true);
  expect("store map unpack addr", u.map, 0x90080);
  expect("store map unpack slot", u.slot, 0);
  expect("tmap insn prefetch", dr::insn_tmap(DXA_TMAP_PREFETCH, 10), 0x0605700Bu);
  expect("tmap insn inval", dr::insn_tmap(DXA_TMAP_INVAL, 10), 0x0615700Bu);
}

} // namespace

int main() {
  test_layout();
  test_accept();
  test_reject();
  test_replace_address();
  test_set_layout();
  test_replace();
  test_red_type();
  test_operands();
  std::printf("dxa_tmap: %d checks passed, %d failed\n", g_pass, g_fail);
  if (g_fail) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
