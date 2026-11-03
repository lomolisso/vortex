// atomic_fp: the FPAMO atomics (float add, packed 16-bit min/max, inc/dec) and
// the vx_atomic.h loop and vector forms, on global memory, the block's own
// local memory, or a peer block's through the DLMEM window. Checked bit for
// bit against sw/common/amo_ref.h; edge also runs H100 measurements of the
// memory's spaces and compares with the H100's results.
//
//   -s local|remote|global  the memory the atomics hit (default global)
//   -x edge     every form on edge-case operand pairs: each record writes a
//               16-byte slot (the operand at some offset, guard bytes around
//               it), runs one op, reads the slot back and checks that
//               fflags did not change (default)
//      hist     integer-valued adds into 16 bins: exact sums
//      contend  every thread on one word; prints the cycles
//      allpush  red.shared::cluster.add.f32 of every rank's vector into every
//               peer (the atomic_all_push collective); remote only
//      f16loop  the global 16-bit add loop (-o add.f16 or add.bf16)
//   -o form     edge: PTX forms, comma-separated (default all); hist/contend: the form added
//               (default add.f32): add.{f32,f64,f16,bf16,f16x2,bf16x2,v2.f32},
//               {inc,dec}.u32
//   -k K        blocks: a cluster of K for remote (default 2), else K blocks
//   -w W        warps per block (default: all of a core; 1 for remote)
//   -i I        iterations per thread (hist 8, contend 16, f16loop 8)
//   -m M        allpush: elements per vector (default 64)
//   -r          hist/contend/f16loop: the red form (no return value)
//   -g          hist/contend/f16loop: the generic form instead of the space's own
//   -c path     the H100 measurements (default tests/unittest/amo_ref/h100_atomics.csv.gz)
//   -p N        edge: at most N measured rows per form, space and variant,
//               spread over the group (default 32; 0 takes every row)
//   -n          ignore the H100 measurements

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"
#include "probe_csv.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vortex::amo_ref;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

#ifndef PROBE_CSV
#define PROBE_CSV ""
#endif

namespace {

const char* kernel_file = "kernel.vxbin";
uint32_t space = SPACE_GLOBAL, mode = MODE_EDGE;
std::string form_name;
uint32_t req_k = 0, req_w = 0, req_iters = 0, req_m = 64, probe_cap = 32;
bool red = false, generic = false, use_probe = true, verbose = false;
std::string csv_path = PROBE_CSV;

struct Caps {
  uint64_t num_cores = 0, num_warps = 0, num_threads = 0, lmem_size = 0, max_cluster = 0, isa = 0;
};

void usage() {
  std::printf("Usage: [-s local|remote|global] [-x edge|hist|contend|allpush|f16loop] [-o form]"
              " [-k K] [-w warps] [-i iters] [-m elems] [-r] [-g] [-c csv] [-p N] [-n] [-v] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "s:x:o:k:w:i:m:rgc:p:nvh")) != -1) {
    switch (c) {
    case 's':
      if (!std::strcmp(optarg, "global")) space = SPACE_GLOBAL;
      else if (!std::strcmp(optarg, "local")) space = SPACE_LOCAL;
      else if (!std::strcmp(optarg, "remote")) space = SPACE_REMOTE;
      else { usage(); std::exit(-1); }
      break;
    case 'x':
      if (!std::strcmp(optarg, "edge")) mode = MODE_EDGE;
      else if (!std::strcmp(optarg, "hist")) mode = MODE_HIST;
      else if (!std::strcmp(optarg, "contend")) mode = MODE_CONTEND;
      else if (!std::strcmp(optarg, "allpush")) mode = MODE_ALLPUSH;
      else if (!std::strcmp(optarg, "f16loop")) mode = MODE_F16LOOP;
      else { usage(); std::exit(-1); }
      break;
    case 'o': form_name = optarg; break;
    case 'k': req_k = (uint32_t)std::atoi(optarg); break;
    case 'w': req_w = (uint32_t)std::atoi(optarg); break;
    case 'i': req_iters = (uint32_t)std::atoi(optarg); break;
    case 'm': req_m = (uint32_t)std::atoi(optarg); break;
    case 'r': red = true; break;
    case 'g': generic = true; break;
    case 'c': csv_path = optarg; break;
    case 'p': probe_cap = (uint32_t)std::atoi(optarg); break;
    case 'n': use_probe = false; break;
    case 'v': verbose = true; break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || (mode == MODE_ALLPUSH && space != SPACE_REMOTE)
   || (mode == MODE_F16LOOP && space == SPACE_REMOTE) || req_m == 0) {
    usage();
    std::exit(-1);
  }
}

const char* space_name(uint32_t s) {
  return (s == SPACE_GLOBAL) ? "global" : (s == SPACE_LOCAL) ? "local" : "remote";
}

///////////////////////////////////////////////////////////////////////////////
// Forms

// amo_ref pseudo-ops of the loop-only 16-bit adds (amo_ref::add_f16/add_bf16).
constexpr uint32_t SW_ADD_F16 = 100, SW_ADD_BF16 = 101;

struct FormInfo {
  const char* name;
  uint32_t form;
  uint32_t op, width;  // the amo_ref op of one element or packed word
  uint32_t elems;      // vector elements (1 = scalar)
  bool global_only;
};

// vx_atomic.h forms, by PTX name.
const FormInfo kForms[] = {
  {"add.f16",        FORM_ADD_F16,        SW_ADD_F16,  W_H, 1, false},
  {"add.bf16",       FORM_ADD_BF16,       SW_ADD_BF16, W_H, 1, false},
  {"cas.b16",        FORM_CAS_B16,        OP_CAS,      W_H, 1, false},
  {"add.f32",        FORM_ADD_F32,        OP_FADD,     W_W, 1, false},
  {"add.f64",        FORM_ADD_F64,        OP_FADD,     W_D, 1, false},
  {"add.f16x2",      FORM_ADD_F16X2,      OP_FADD2,    W_W, 1, false},
  {"add.bf16x2",     FORM_ADD_BF16X2,     OP_FADD2_BF, W_W, 1, false},
  {"inc.u32",        FORM_INC_U32,        OP_INC,      W_W, 1, false},
  {"dec.u32",        FORM_DEC_U32,        OP_DEC,      W_W, 1, false},
  {"add.v2.f32",     FORM_ADD_V2_F32,     OP_FADD,     W_W, 2, true},
  {"add.v4.f32",     FORM_ADD_V4_F32,     OP_FADD,     W_W, 4, true},
  {"add.v8.f16",     FORM_ADD_V8_F16,     OP_FADD2,    W_W, 4, true},
  {"add.v8.bf16",    FORM_ADD_V8_BF16,    OP_FADD2_BF, W_W, 4, true},
  {"min.v8.f16",     FORM_MIN_V8_F16,     OP_FMIN2,    W_W, 4, true},
  {"max.v8.bf16",    FORM_MAX_V8_BF16,    OP_FMAX2_BF, W_W, 4, true},
  {"add.v4.f16x2",   FORM_ADD_V4_F16X2,   OP_FADD2,    W_W, 4, true},
  {"max.v4.bf16x2",  FORM_MAX_V4_BF16X2,  OP_FMAX2_BF, W_W, 4, true},
  {"red.add.v4.f32", FORM_RED_ADD_V4_F32, OP_FADD,     W_W, 4, true},
  {"min.v2.f16",     FORM_MIN_V2_F16,     OP_FMIN2,    W_W, 1, true},
  {"max.v2.bf16",    FORM_MAX_V2_BF16,    OP_FMAX2_BF, W_W, 1, true},
};

const FormInfo* find_form(const std::string& name) {
  for (auto& f : kForms)
    if (name == f.name) return &f;
  return nullptr;
}

const FormInfo& form_by_id(uint32_t id) {
  for (auto& f : kForms)
    if (f.form == id) return f;
  std::abort();
}

// The FPAMO instructions, by (op, width), with the PTX form they serve.
struct HwInfo { const char* name; uint32_t op, width; };
const HwInfo kHw[] = {
  {"inc.u32", OP_INC, W_W},         {"dec.u32", OP_DEC, W_W},
  {"add.f32", OP_FADD, W_W},        {"add.f64", OP_FADD, W_D},
  {"add.f16x2", OP_FADD2, W_W},     {"add.v2.f32", OP_FADD2, W_D},
  {"add.bf16x2", OP_FADD2_BF, W_W},
  {"min.v2.f16", OP_FMIN2, W_W},    {"max.v2.f16", OP_FMAX2, W_W},
  {"min.v2.bf16", OP_FMIN2_BF, W_W}, {"max.v2.bf16", OP_FMAX2_BF, W_W},
};

// Whether the memory of `s` runs (op, width) in hardware: the cache every op,
// the local-memory banks inc/dec only.
bool site_row(uint32_t s, uint32_t op, uint32_t width) {
  if (!legal(op, width) || op == OP_CAS)
    return false;
  return op == OP_INC || op == OP_DEC || s == SPACE_GLOBAL;
}

// The amo_ref site of an op on the memory of `s`: global memory runs the
// instruction at the last-level cache; local memory, own or a peer's, runs the
// loop.
Site site_of(uint32_t s) { return (s == SPACE_GLOBAL) ? SITE_LLC : SITE_LOOP; }

// The vx_atomic.h space variant matching the memory of `s`.
uint32_t space_sp(uint32_t s) {
  return (s == SPACE_GLOBAL) ? SP_GLOBAL : (s == SPACE_LOCAL) ? SP_SHARED : SP_CLUSTER;
}

// -o takes one form, a comma-separated list, or all.
bool selected(const char* name) {
  if (form_name.empty() || form_name == "all")
    return true;
  return ("," + form_name + ",").find("," + std::string(name) + ",") != std::string::npos;
}

///////////////////////////////////////////////////////////////////////////////
// Slot images

uint64_t img_get(const uint64_t* img, uint32_t off, uint32_t bytes) {
  uint8_t b[16];
  std::memcpy(b, img, 16);
  uint64_t v = 0;
  std::memcpy(&v, b + off, bytes);
  return v;
}

void img_put(uint64_t* img, uint32_t off, uint32_t bytes, uint64_t v) {
  uint8_t b[16];
  std::memcpy(b, img, 16);
  std::memcpy(b + off, &v, bytes);
  std::memcpy(img, b, 16);
}

uint64_t elem_apply(uint32_t op, uint32_t width, uint64_t old, uint64_t rhs, Site site) {
  if (op == SW_ADD_F16)  return add_f16((uint16_t)old, (uint16_t)rhs);
  if (op == SW_ADD_BF16) return add_bf16((uint16_t)old, (uint16_t)rhs);
  return apply(op, width, old, rhs, site);
}

struct Expect {
  std::string what;
  uint64_t mem[2];
  uint64_t ret[2];
  bool check_ret;
  bool has_h100;
  uint32_t off, bytes;
  probe_csv::V128 h100_mem, h100_ret;
  bool h100_has_ret;
};

void expect_rec(const rec_t& r, Site site, Expect& e) {
  e.mem[0] = r.old[0];
  e.mem[1] = r.old[1];
  e.ret[0] = e.ret[1] = 0;
  e.check_ret = (r.impl != IMPL_RED);
  if (r.impl != IMPL_HDR) {
    const uint32_t bytes = 1u << r.width;
    const uint64_t old = img_get(r.old, r.offset, bytes);
    img_put(e.mem, r.offset, bytes, apply(r.op, r.width, old, r.rhs[0], site));
    e.ret[0] = ret_value(r.width, old);
    e.off = r.offset;
    e.bytes = bytes;
    return;
  }
  const FormInfo& f = form_by_id(r.form);
  const uint32_t eb = 1u << f.width;
  if (f.elems == 1) {
    const uint64_t old = img_get(r.old, r.offset, eb);
    const uint64_t nv = (f.op == OP_CAS) ? cas(W_H, old, r.cmp, r.rhs[0]) : elem_apply(f.op, f.width, old, r.rhs[0], site);
    img_put(e.mem, r.offset, eb, nv);
    e.ret[0] = old;
    e.off = r.offset;
    e.bytes = eb;
    return;
  }
  for (uint32_t i = 0; i < f.elems; ++i) {
    const uint64_t old = img_get(r.old, i * eb, eb);
    img_put(e.mem, i * eb, eb, elem_apply(f.op, f.width, old, img_get(r.rhs, i * eb, eb), site));
    img_put(e.ret, i * eb, eb, old);
  }
  e.check_ret = (f.form != FORM_RED_ADD_V4_F32);
  e.off = 0;
  e.bytes = f.elems * eb;
}

///////////////////////////////////////////////////////////////////////////////
// Edge records

const uint32_t kF32[] = {0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x40000000, 0x33800000,
                         0x34400000, 0x3f800001, 0x00000001, 0x80000001, 0x007fffff, 0x00800000,
                         0x80800000, 0x00800001, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                         0x7fc00000, 0x7fc12345, 0xffc00001, 0x7f800001};
const uint64_t kF64[] = {0, 0x8000000000000000ull, 0x3ff0000000000000ull, 0xbff0000000000000ull,
                         0x4000000000000000ull, 0x3ca0000000000000ull, 0x3ff0000000000001ull, 1,
                         0x8000000000000001ull, 0x000fffffffffffffull, 0x0010000000000000ull,
                         0x8010000000000000ull, 0x7fefffffffffffffull, 0xffefffffffffffffull,
                         0x7ff0000000000000ull, 0xfff0000000000000ull, 0x7ff8000000000000ull,
                         0x7ff8000000012345ull, 0x7ff0000000000001ull};
const uint32_t kF16[] = {0x0000, 0x8000, 0x3c00, 0xbc00, 0x4000, 0x1000, 0x1600, 0x3c01, 0x0001,
                         0x8001, 0x03ff, 0x0400, 0x8400, 0x0401, 0x7bff, 0xfbff, 0x4c00, 0x4800,
                         0x7c00, 0xfc00, 0x7e00, 0x7e01, 0xfe01, 0x7c01};
const uint32_t kBF16[] = {0x0000, 0x8000, 0x3f80, 0xbf80, 0x4000, 0x3b80, 0x3c40, 0x3f81, 0x0001,
                          0x8001, 0x007f, 0x0080, 0x8080, 0x0081, 0x7f7f, 0xff7f, 0x7f80, 0xff80,
                          0x7fc0, 0x7fc1, 0xffc1, 0x7f81};
const uint32_t kU32[] = {0, 1, 2, 5, 7, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff};

template <typename T, size_t N>
std::vector<uint64_t> vals(const T (&a)[N]) { return std::vector<uint64_t>(a, a + N); }

// Operand pairs of an (op, width): every (a, b) of the element type's
// specials, and for packed types a second lane from another pair.
struct Pair { uint64_t a, b; };
std::vector<Pair> pairs_of(uint32_t op, uint32_t width, size_t cap) {
  std::vector<uint64_t> s;
  const bool bf = (op == SW_ADD_BF16 || op == OP_FADD2_BF || op == OP_FMIN2_BF || op == OP_FMAX2_BF);
  const bool x2 = (op == OP_FADD2 || op == OP_FADD2_BF || op == OP_FMIN2 || op == OP_FMAX2
                || op == OP_FMIN2_BF || op == OP_FMAX2_BF);
  if (op == OP_INC || op == OP_DEC) s = vals(kU32);
  else if (op == OP_FADD && width == W_D) s = vals(kF64);
  else if ((op == OP_FADD && width == W_W) || (op == OP_FADD2 && width == W_D)) s = vals(kF32);
  else s = bf ? vals(kBF16) : vals(kF16);
  if (s.size() > cap) s.resize(cap);
  const size_t n = s.size();
  const uint32_t lb = (width == W_D) ? 32 : 16;
  std::vector<Pair> out;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      Pair p{s[i], s[j]};
      if (x2) {
        p.a |= s[j] << lb;
        p.b |= s[(i + j + 1) % n] << lb;
      }
      out.push_back(p);
    }
  }
  return out;
}

void guard(uint64_t* img, uint32_t i) {
  img[0] = ((uint64_t)hash3(i, 1, 0xA5) << 32) | hash3(i, 2, 0xA5);
  img[1] = ((uint64_t)hash3(i, 3, 0xA5) << 32) | hash3(i, 4, 0xA5);
}

uint32_t pick_offset(uint32_t width, uint32_t i) {
  return (width == W_H) ? (i % 8) * 2 : (width == W_W) ? (i % 4) * 4 : (i % 2) * 8;
}

void add_hw(std::vector<rec_t>& recs, std::vector<Expect>& exps, uint32_t op, uint32_t width,
            uint64_t old, uint64_t rhs, uint32_t impl, const std::string& what) {
  rec_t r = {};
  const uint32_t i = (uint32_t)recs.size();
  r.impl = impl;
  r.op = op;
  r.width = width;
  r.offset = pick_offset(width, i);
  guard(r.old, i);
  img_put(r.old, r.offset, 1u << width, old);
  r.rhs[0] = rhs & width_mask(width);
  Expect e = {};
  e.what = what;
  expect_rec(r, site_of(space), e);
  recs.push_back(r);
  exps.push_back(e);
}

// offset ~0u picks one; scalars take old[0] and rhs[0], vectors the 16 bytes.
void add_hdr(std::vector<rec_t>& recs, std::vector<Expect>& exps, const FormInfo& f, uint32_t sp,
             const uint64_t* old, const uint64_t* rhs, uint64_t cmp, uint32_t offset, const std::string& what) {
  rec_t r = {};
  const uint32_t i = (uint32_t)recs.size();
  r.impl = IMPL_HDR;
  r.form = f.form;
  r.sp = sp;
  if (f.elems == 1 && !f.global_only) {
    r.offset = (offset != ~0u) ? offset : pick_offset(f.width, i);
    guard(r.old, i);
    img_put(r.old, r.offset, 1u << f.width, old[0]);
    r.rhs[0] = rhs[0];
  } else {
    guard(r.old, i);
    const uint32_t nb = f.elems << f.width;
    for (uint32_t k = 0; k < nb; k += 8)
      img_put(r.old, k, std::min<uint32_t>(8, nb - k), old[k / 8]);
    r.rhs[0] = rhs[0];
    r.rhs[1] = (nb > 8) ? rhs[1] : 0;
  }
  r.cmp = cmp;
  Expect e = {};
  e.what = what + (sp == SP_GENERIC ? " (generic)" : "");
  expect_rec(r, site_of(space), e);
  recs.push_back(r);
  exps.push_back(e);
}

std::string hex(uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof(b), "%#" PRIx64, v);
  return b;
}

// The records of one vx_atomic.h form in space variant sp.
void build_form(std::vector<rec_t>& recs, std::vector<Expect>& exps, const FormInfo& f, uint32_t sp) {
  if (f.form == FORM_CAS_B16) {
    for (uint32_t i = 0; i < 32; ++i) {
      const uint64_t o = hash3(i, 7, sp) & 0xffff, v = hash3(i, 8, sp) & 0xffff;
      const uint64_t c = (i & 1) ? o : (o ^ (1u << (i % 16)));
      add_hdr(recs, exps, f, sp, &o, &v, c, ~0u, std::string(f.name) + ((i & 1) ? " hit " : " miss ") + hex(o));
    }
    return;
  }
  if (f.elems == 1 && !f.global_only) {
    // The loops take longer than one instruction: a smaller set than the hardware rows.
    auto ps = pairs_of(f.op, f.width, (sp == SP_GENERIC) ? 12 : 24);
    for (auto& p : ps)
      add_hdr(recs, exps, f, sp, &p.a, &p.b, 0, ~0u, std::string(f.name) + " " + hex(p.a) + " " + hex(p.b));
    return;
  }
  // Vectors: elements drawn from the element type's specials.
  const uint32_t eb = 1u << f.width, nb = f.elems * eb;
  auto ps = pairs_of(f.op, f.width, 32);
  for (size_t i = 0; i < ps.size(); i += 7) {
    uint64_t o[2] = {0, 0}, v[2] = {0, 0};
    for (uint32_t k = 0; k * eb < nb; ++k) {
      const Pair& p = ps[(i + k * 5) % ps.size()];
      img_put(o, k * eb, eb, p.a);
      img_put(v, k * eb, eb, p.b);
    }
    add_hdr(recs, exps, f, sp, o, v, 0, 0, std::string(f.name) + " #" + std::to_string(i));
  }
}

// Repeats the records from `start` on until their count is a multiple of `nt`:
// thread t takes records t, t + nt, ..., so every warp then runs one form at a
// time. Warps whose lanes sit in different forms' CAS loops at once end in
// "IPDOM stack is empty" (the divergence of a multi-way branch around loops).
void pad_group(std::vector<rec_t>& recs, std::vector<Expect>& exps, size_t start, uint32_t nt) {
  const size_t n = recs.size() - start;
  for (size_t k = 0; n && (recs.size() - start) % nt; ++k) {
    recs.push_back(recs[start + k % n]);
    exps.push_back(exps[start + k % n]);
  }
}

void build_edge(std::vector<rec_t>& recs, std::vector<Expect>& exps, uint32_t nt) {
  for (auto& h : kHw) {
    if (!selected(h.name) || !site_row(space, h.op, h.width))
      continue;
    const size_t start = recs.size();
    auto ps = pairs_of(h.op, h.width, 32);
    for (size_t i = 0; i < ps.size(); ++i) {
      const std::string what = std::string("hw ") + h.name + " " + hex(ps[i].a) + " " + hex(ps[i].b);
      add_hw(recs, exps, h.op, h.width, ps[i].a, ps[i].b, IMPL_HW, what);
      if (i % 8 == 0)
        add_hw(recs, exps, h.op, h.width, ps[i].a, ps[i].b, IMPL_RED, "red " + what);
    }
    pad_group(recs, exps, start, nt);
  }
  // Each form both as the variant of this memory's space and as the generic
  // one, whose address test must send it the same way.
  for (auto& f : kForms) {
    if (!selected(f.name) || (f.global_only && space != SPACE_GLOBAL))
      continue;
    for (uint32_t sp : {space_sp(space), (uint32_t)SP_GENERIC}) {
      const size_t start = recs.size();
      build_form(recs, exps, f, sp);
      pad_group(recs, exps, start, nt);
    }
  }
}

void report(const std::string& msg);

// H100 rows of this memory's spaces, run as the same form here: per (form,
// op, space, variant) group, at most probe_cap rows spread over the group.
void build_probe(std::vector<rec_t>& recs, std::vector<Expect>& exps, uint32_t nt) {
  if (!use_probe || csv_path.empty())
    return;
  const std::string path = probe_csv::locate(csv_path);
  std::vector<probe_csv::Row> rows;
  std::string msg;
  if (path.empty() || !probe_csv::load(path, rows, msg)) {
    report("H100 measurements not loaded (" + (path.empty() ? "absent: " + csv_path : msg) + ")");
    return;
  }
  const char* own[] = {"global", "shared_cta", "shared_cluster"};
  const char* gen[] = {"generic_global", "generic_smem", "generic_dsmem"};
  std::map<std::string, std::vector<const probe_csv::Row*>> groups;
  for (auto& r : rows) {
    if ((r.space != own[space] && r.space != gen[space]) || r.f.kind == probe_csv::K_NONE
     || r.variant.compare(0, 4, "cuda") == 0 || probe_csv::excluded(r))
      continue;
    groups[r.form + " " + r.op + " " + r.space + " " + r.variant].push_back(&r);
  }
  size_t used = 0;
  for (auto& kv : groups) {
    const auto& g = kv.second;
    const size_t step = (probe_cap == 0 || g.size() <= probe_cap) ? 1 : (g.size() + probe_cap - 1) / probe_cap;
    const size_t start = recs.size();
    for (size_t k = 0; k < g.size(); k += step) {
      const probe_csv::Row& r = *g[k];
      const uint32_t sp = (r.space == own[space]) ? space_sp(space) : (uint32_t)SP_GENERIC;
      const bool red_row = (r.variant == "red");
      const std::string vname = (r.form.size() > 2 && r.form[0] == 'v' && r.form[2] == '.')
                              ? (r.op + "." + r.form) : "";
      const FormInfo* f = nullptr;
      if (r.f.kind == probe_csv::K_ADD16) f = find_form(r.f.bf ? "add.bf16" : "add.f16");
      else if (r.f.kind == probe_csv::K_CAS16) f = find_form("cas.b16");
      else if (!vname.empty()) f = find_form(vname);
      else f = find_form(r.op + "." + r.form);
      if (!f || !selected(f->name))
        continue;
      const std::string what = "H100 " + r.op + "." + r.form + "@" + r.space + "/" + r.variant;
      const size_t before = recs.size();
      if (sp != SP_GENERIC && r.f.kind == probe_csv::K_OP && vname.empty() && site_row(space, r.f.op, r.f.width)) {
        add_hw(recs, exps, r.f.op, r.f.width, r.old.lo, r.rhs.lo, red_row ? IMPL_RED : IMPL_HW, what);
      } else if (r.f.kind == probe_csv::K_CAS16) {
        add_hdr(recs, exps, *f, sp, &r.old.lo, &r.rhs2.lo, r.rhs.lo, ~0u, what);
      } else {
        const uint64_t o[2] = {r.old.lo, r.old.hi}, v[2] = {r.rhs.lo, r.rhs.hi};
        add_hdr(recs, exps, *f, sp, o, v, 0, ~0u, what);
      }
      if (recs.size() > before) {
        Expect& e = exps.back();
        e.has_h100 = true;
        e.h100_mem = r.mem;
        e.h100_ret = r.ret;
        e.h100_has_ret = r.has_ret && !red_row;
        ++used;
      }
    }
    pad_group(recs, exps, start, nt);
  }
  std::printf("atomic_fp: %zu of %zu H100 rows from %s\n", used, rows.size(), path.c_str());
}

///////////////////////////////////////////////////////////////////////////////
// Launch

struct Run {
  vx_device_h dev;
  vx_queue_h q;
  vx_module_h mod;
  Caps caps;
  uint32_t blocks, nt, lmem;
};

uint64_t buffer(Run& R, std::vector<vx_buffer_h>& bufs, const void* data, size_t bytes) {
  vx_buffer_h b = nullptr;
  CHECK(vx_buffer_create(R.dev, std::max<size_t>(bytes, 64), VX_MEM_READ_WRITE, &b));
  if (data)
    CHECK(vx_enqueue_write(R.q, b, 0, data, bytes, 0, nullptr, nullptr));
  uint64_t addr = 0;
  CHECK(vx_buffer_address(b, &addr));
  bufs.push_back(b);
  return addr;
}

void release(std::vector<vx_buffer_h>& bufs) {
  for (auto b : bufs)
    vx_buffer_release(b);
  bufs.clear();
}

void launch(Run& R, const char* kname, arg_t& arg) {
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(R.mod, kname, &kernel));
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = R.blocks;
  li.grid_dim[1]  = 1;
  li.grid_dim[2]  = 1;
  li.block_dim[0] = R.nt;
  li.block_dim[1] = 1;
  li.block_dim[2] = 1;
  li.lmem_size    = R.lmem;
  if (space == SPACE_REMOTE) {
    li.cluster_dim[0] = R.blocks;
    li.cluster_dim[1] = 1;
    li.cluster_dim[2] = 1;
    li.cluster_policy = VX_CLUSTER_POLICY_SPREAD;
    li.cluster_flags  = (R.blocks > VX_CLUSTER_MAX_PORTABLE) ? VX_CLUSTER_FLAG_NON_PORTABLE : 0;
  }
  vx_event_h ev = nullptr;
  CHECK(vx_enqueue_launch(R.q, &li, 0, nullptr, &ev));
  CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
  vx_kernel_release(kernel);
}

void readback(Run& R, vx_buffer_h b, void* data, size_t bytes) {
  vx_event_h ev = nullptr;
  CHECK(vx_enqueue_read(R.q, data, b, 0, bytes, 0, nullptr, &ev));
  CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
}

int g_errors = 0;
void report(const std::string& msg) {
  if (g_errors < 24)
    std::printf("*** %s\n", msg.c_str());
  ++g_errors;
}

// Blocks that ran, and (remote) whether the cluster spans cores.
void check_out(const std::vector<uint32_t>& out, uint32_t m, Run& R) {
  uint32_t first_core = ~0u;
  bool spans = false;
  for (uint32_t b = 0; b < R.blocks; ++b) {
    const uint32_t* o = &out[(size_t)b * OUT_WORDS(m)];
    if (o[OUT_RAN] != 1)
      report("block " + std::to_string(b) + " did not run");
    if (first_core == ~0u) first_core = o[OUT_CORE];
    spans |= (o[OUT_CORE] != first_core);
  }
  if (space == SPACE_REMOTE && R.blocks > 1 && !spans)
    report("the cluster does not span two cores: nothing crossed the network");
}

int block_of_rank0(const std::vector<uint32_t>& out, uint32_t m, uint32_t blocks) {
  for (uint32_t b = 0; b < blocks; ++b)
    if (out[(size_t)b * OUT_WORDS(m) + OUT_RANK] == 0) return (int)b;
  return 0;
}

void run_edge(Run& R) {
  std::vector<rec_t> recs;
  std::vector<Expect> exps;
  build_edge(recs, exps, R.nt);
  build_probe(recs, exps, R.nt);
  if (recs.empty()) {
    report("no records for form '" + form_name + "' on " + space_name(space));
    return;
  }
  std::printf("atomic_fp: edge, %s, %zu records, %u blocks of %u threads\n", space_name(space),
              recs.size(), R.blocks, R.nt);
  std::vector<vx_buffer_h> bufs;
  const uint32_t m = 0;
  std::vector<res_t> res((size_t)R.blocks * recs.size());
  std::vector<uint32_t> out((size_t)R.blocks * OUT_WORDS(m), 0);
  arg_t arg = {};
  arg.mode = MODE_EDGE;
  arg.space = space;
  arg.k = R.blocks;
  arg.num_recs = (uint32_t)recs.size();
  arg.rec_addr = buffer(R, bufs, recs.data(), recs.size() * sizeof(rec_t));
  arg.res_addr = buffer(R, bufs, nullptr, res.size() * sizeof(res_t));
  arg.out_addr = buffer(R, bufs, out.data(), out.size() * 4);
  arg.glob_addr = buffer(R, bufs, nullptr, (size_t)R.blocks * R.nt * SLOT_BYTES);
  launch(R, "edge_k", arg);
  readback(R, bufs[1], res.data(), res.size() * sizeof(res_t));
  readback(R, bufs[2], out.data(), out.size() * 4);
  check_out(out, m, R);
  size_t h100_checked = 0, h100_diff = 0;
  for (uint32_t b = 0; b < R.blocks; ++b) {
    for (size_t i = 0; i < recs.size(); ++i) {
      const res_t& g = res[b * recs.size() + i];
      const Expect& e = exps[i];
      if (g.mem[0] != e.mem[0] || g.mem[1] != e.mem[1])
        report("block " + std::to_string(b) + " " + e.what + ": slot " + hex(g.mem[1]) + ":" + hex(g.mem[0])
               + ", amo_ref " + hex(e.mem[1]) + ":" + hex(e.mem[0]));
      if (e.check_ret && (g.ret[0] != e.ret[0] || g.ret[1] != e.ret[1]))
        report("block " + std::to_string(b) + " " + e.what + ": returned " + hex(g.ret[1]) + ":" + hex(g.ret[0])
               + ", amo_ref " + hex(e.ret[1]) + ":" + hex(e.ret[0]));
      // Atomics write no fflags, the loops included.
      if ((g.fflags >> 8) != (g.fflags & 0xff))
        report("block " + std::to_string(b) + " " + e.what + ": fflags " + hex(g.fflags >> 8) + " became "
               + hex(g.fflags & 0xff));
      if (e.has_h100 && b == 0) {
        ++h100_checked;
        uint64_t gm[2] = {0, 0}, gr[2] = {0, 0}, hm[2] = {e.h100_mem.lo, e.h100_mem.hi};
        uint64_t hr[2] = {e.h100_ret.lo, e.h100_ret.hi};
        for (uint32_t k = 0; k < e.bytes; k += 8) {
          const uint32_t n = std::min<uint32_t>(8, e.bytes - k);
          img_put(gm, k, n, img_get(g.mem, e.off + k, n));
          img_put(gr, k, n, img_get(g.ret, k, n));
        }
        auto masked = [&](uint64_t* x) {
          uint64_t y[2] = {0, 0};
          for (uint32_t k = 0; k < e.bytes; k += 8)
            img_put(y, k, std::min<uint32_t>(8, e.bytes - k), img_get(x, k, std::min<uint32_t>(8, e.bytes - k)));
          x[0] = y[0];
          x[1] = y[1];
        };
        masked(hm);
        masked(hr);
        bool bad = gm[0] != hm[0] || gm[1] != hm[1];
        if (e.h100_has_ret && e.check_ret && (gr[0] != hr[0] || gr[1] != hr[1]))
          bad = true;
        if (bad) {
          ++h100_diff;
          report(e.what + ": device " + hex(gm[1]) + ":" + hex(gm[0]) + " ret " + hex(gr[0]) + ", H100 mem "
                 + hex(hm[1]) + ":" + hex(hm[0]) + " ret " + hex(hr[0]));
        }
      }
    }
  }
  if (h100_checked)
    std::printf("atomic_fp: %zu records against the H100, %zu differ\n", h100_checked, h100_diff);
  release(bufs);
}

// The (op, width) of a hist/contend/f16loop form.
// A hist/contend/f16loop form: a scalar add, inc/dec, or add.v2.f32.
const FormInfo& form_or(const char* dflt) {
  const std::string n = form_name.empty() ? dflt : form_name;
  const FormInfo* f = find_form(n);
  if (!f || f->form == FORM_CAS_B16 || (f->global_only && f->form != FORM_ADD_V2_F32)) {
    std::printf("Error: form '%s' is not available in this mode\n", n.c_str());
    std::exit(-1);
  }
  return *f;
}

// The operand bits the device's encode() produces for integers a, b.
uint64_t encode(const FormInfo& f, uint32_t a, uint32_t b) {
  auto f32 = [](uint32_t v) { float x = (float)v; uint32_t u; std::memcpy(&u, &x, 4); return (uint64_t)u; };
  auto h = [&](uint32_t v) {
    softfloat_roundingMode = softfloat_round_near_even;
    return (uint64_t)f32_to_f16(float32_t{(uint32_t)f32(v)}).v;
  };
  switch (f.form) {
  case FORM_ADD_F32:    return f32(a);
  case FORM_ADD_F64:    { double x = (double)a; uint64_t u; std::memcpy(&u, &x, 8); return u; }
  case FORM_ADD_F16:    return h(a);
  case FORM_ADD_BF16:   return f32(a) >> 16;
  case FORM_ADD_F16X2:  return h(a) | (h(b) << 16);
  case FORM_ADD_BF16X2: return (f32(a) >> 16) | ((f32(b) >> 16) << 16);
  case FORM_ADD_V2_F32: return f32(a) | (f32(b) << 32);
  default:              return 0xffffffffu;
  }
}

// One op of a hist/contend form on an 8-byte bin (the operand at its start).
uint64_t bin_apply(const FormInfo& f, uint64_t bin, uint64_t v) {
  if (f.form == FORM_ADD_V2_F32)
    return apply(OP_FADD2, W_D, bin, v, site_of(space));
  const uint64_t m = width_mask(f.width);
  return (bin & ~m) | elem_apply(f.op, f.width, bin & m, v, site_of(space));
}

void run_hist(Run& R, bool contend) {
  const FormInfo& f = form_or("add.f32");
  if (f.elems != 1 && f.form != FORM_ADD_V2_F32) {
    report(std::string("form ") + f.name + " is not a hist/contend form");
    return;
  }
  if (f.form == FORM_ADD_V2_F32 && space != SPACE_GLOBAL) {
    report("add.v2.f32 is a global form");
    return;
  }
  const uint32_t iters = req_iters ? req_iters : (contend ? 16 : 8);
  const bool small16 = (f.form == FORM_ADD_BF16 || f.form == FORM_ADD_BF16X2);
  const uint32_t maxv = contend ? 1 : (small16 ? 1 : 3);
  std::vector<vx_buffer_h> bufs;
  const uint32_t m = maxv;
  std::vector<uint32_t> out((size_t)R.blocks * OUT_WORDS(m), 0);
  std::vector<uint64_t> glob(HIST_BINS, 0);
  arg_t arg = {};
  arg.mode = contend ? MODE_CONTEND : MODE_HIST;
  arg.space = space;
  arg.k = R.blocks;
  arg.form = f.form;
  arg.iters = iters;
  arg.m = m;
  arg.seed = 0x5EED0000u + f.form;
  arg.red = red;
  arg.sp = generic ? SP_GENERIC : space_sp(space);
  arg.operand = encode(f, 1, 1);
  arg.out_addr = buffer(R, bufs, out.data(), out.size() * 4);
  arg.glob_addr = buffer(R, bufs, glob.data(), glob.size() * 8);
  std::printf("atomic_fp: %s %s%s%s, %s, %u blocks of %u threads, %u iterations\n",
              contend ? "contend" : "hist", red ? "red." : "", f.name, generic ? " (generic)" : "",
              space_name(space), R.blocks, R.nt, iters);
  launch(R, contend ? "contend_k" : "hist_k", arg);
  readback(R, bufs[0], out.data(), out.size() * 4);
  readback(R, bufs[1], glob.data(), glob.size() * 8);
  check_out(out, m, R);
  release(bufs);

  // Integer-valued operands: the sum is exact in any order (and +1 sequences
  // that saturate do so identically), so a sequential replay is the answer.
  const uint32_t sets = (space == SPACE_LOCAL) ? R.blocks : 1;
  for (uint32_t s = 0; s < sets; ++s) {
    std::vector<uint64_t> bins(HIST_BINS, 0);
    for (uint32_t b = 0; b < R.blocks; ++b) {
      if (space == SPACE_LOCAL && b != s) continue;
      for (uint32_t t = 0; t < R.nt; ++t) {
        for (uint32_t j = 0; j < iters; ++j) {
          if (contend) {
            bins[0] = bin_apply(f, bins[0], arg.operand);
          } else {
            const uint32_t h = hist_hash(arg.seed, b, t, j);
            bins[hist_bin(h)] = bin_apply(f, bins[hist_bin(h)], encode(f, hist_val(h, 0, maxv), hist_val(h, 1, maxv)));
          }
        }
      }
    }
    const uint64_t* got;
    if (space == SPACE_GLOBAL) {
      got = glob.data();
    } else {
      const uint32_t b = (space == SPACE_LOCAL) ? s : (uint32_t)block_of_rank0(out, m, R.blocks);
      got = (const uint64_t*)&out[(size_t)b * OUT_WORDS(m) + OUT_DATA];
    }
    for (uint32_t i = 0; i < (contend ? 1u : (uint32_t)HIST_BINS); ++i) {
      if (got[i] != bins[i])
        report(std::string(contend ? "word" : "bin ") + (contend ? "" : std::to_string(i)) + " of set "
               + std::to_string(s) + ": " + hex(got[i]) + ", expected " + hex(bins[i]));
    }
  }
  if (contend) {
    for (uint32_t b = 0; b < R.blocks; ++b) {
      const uint32_t* o = &out[(size_t)b * OUT_WORDS(m)];
      const uint64_t cyc = ((uint64_t)o[OUT_CYCLES + 1] << 32) | o[OUT_CYCLES];
      std::printf("atomic_fp: contend block %u (rank %u, core %u): %" PRIu64 " cycles for %u ops\n",
                  b, o[OUT_RANK], o[OUT_CORE], cyc, R.nt * iters);
    }
  }
}

void run_allpush(Run& R) {
  const uint32_t m = req_m, k = R.blocks;
  std::vector<vx_buffer_h> bufs;
  std::vector<uint32_t> out((size_t)R.blocks * OUT_WORDS(m), 0);
  arg_t arg = {};
  arg.mode = MODE_ALLPUSH;
  arg.space = space;
  arg.k = k;
  arg.m = m;
  arg.out_addr = buffer(R, bufs, out.data(), out.size() * 4);
  std::printf("atomic_fp: allpush red.add.f32, K=%u, %u elements, %u threads per block\n", k, m, R.nt);
  launch(R, "allpush_k", arg);
  readback(R, bufs[0], out.data(), out.size() * 4);
  check_out(out, m, R);
  release(bufs);
  for (uint32_t b = 0; b < R.blocks; ++b) {
    const uint32_t* o = &out[(size_t)b * OUT_WORDS(m)];
    for (uint32_t e = 0; e < m; ++e) {
      uint32_t sum = 0;
      for (uint32_t r = 0; r < k; ++r) sum += allpush_val(r, e);
      float x = (float)sum;
      uint32_t want;
      std::memcpy(&want, &x, 4);
      if (o[OUT_DATA + e] != want)
        report("rank " + std::to_string(o[OUT_RANK]) + " element " + std::to_string(e) + ": "
               + hex(o[OUT_DATA + e]) + ", expected " + hex(want));
    }
  }
}

void run_f16loop(Run& R) {
  const FormInfo& f = form_or("add.f16");
  if (f.form != FORM_ADD_F16 && f.form != FORM_ADD_BF16) {
    report("f16loop runs add.f16 or add.bf16");
    return;
  }
  const uint32_t iters = req_iters ? req_iters : 8;
  std::vector<vx_buffer_h> bufs;
  const uint32_t m = 0;
  std::vector<uint32_t> out((size_t)R.blocks * OUT_WORDS(m), 0);
  std::vector<uint64_t> glob(2, 0);
  arg_t arg = {};
  arg.mode = MODE_F16LOOP;
  arg.space = space;
  arg.k = R.blocks;
  arg.form = f.form;
  arg.iters = iters;
  arg.red = red;
  arg.sp = generic ? SP_GENERIC : space_sp(space);
  arg.out_addr = buffer(R, bufs, out.data(), out.size() * 4);
  arg.glob_addr = buffer(R, bufs, glob.data(), 16);
  std::printf("atomic_fp: f16loop %s, %s, %u blocks of %u threads, %u iterations\n", f.name,
              space_name(space), R.blocks, R.nt, iters);
  launch(R, "f16loop_k", arg);
  readback(R, bufs[0], out.data(), out.size() * 4);
  readback(R, bufs[1], glob.data(), 16);
  check_out(out, m, R);
  release(bufs);
  const uint64_t one = (f.form == FORM_ADD_BF16) ? 0x3f80 : 0x3c00;
  const uint32_t sets = (space == SPACE_LOCAL) ? R.blocks : 1;
  for (uint32_t s = 0; s < sets; ++s) {
    uint64_t want[2] = {0, 0};
    for (uint32_t b = 0; b < R.blocks; ++b) {
      if (space == SPACE_LOCAL && b != s) continue;
      for (uint32_t t = 0; t < R.nt; ++t)
        for (uint32_t j = 0; j < iters; ++j) {
          const uint32_t e = (t + j + b) % F16LOOP_ELEMS;
          img_put(want, 2 * e, 2, elem_apply(f.op, W_H, img_get(want, 2 * e, 2), one, SITE_LOOP));
        }
    }
    const uint64_t* got = (space == SPACE_GLOBAL) ? glob.data()
                        : (const uint64_t*)&out[(size_t)s * OUT_WORDS(m) + OUT_DATA];
    for (uint32_t e = 0; e < F16LOOP_ELEMS; ++e)
      if (img_get(got, 2 * e, 2) != img_get(want, 2 * e, 2))
        report("element " + std::to_string(e) + " of set " + std::to_string(s) + ": "
               + hex(img_get(got, 2 * e, 2)) + ", expected " + hex(img_get(want, 2 * e, 2)));
  }
}

}  // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  Run R = {};
  CHECK(vx_device_open(0, &R.dev));
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
  CHECK(vx_queue_create(R.dev, &qi, &R.q));
  CHECK(vx_device_query(R.dev, VX_CAPS_NUM_CORES,      &R.caps.num_cores));
  CHECK(vx_device_query(R.dev, VX_CAPS_NUM_WARPS,      &R.caps.num_warps));
  CHECK(vx_device_query(R.dev, VX_CAPS_NUM_THREADS,    &R.caps.num_threads));
  CHECK(vx_device_query(R.dev, VX_CAPS_LOCAL_MEM_SIZE, &R.caps.lmem_size));
  CHECK(vx_device_query(R.dev, VX_CAPS_ISA_FLAGS,      &R.caps.isa));
  if (space == SPACE_REMOTE) {
#if VX_CFG_EXT_DLMEM_ENABLED
    CHECK(vx_device_query(R.dev, VX_CAPS_DLMEM_MAX_CLUSTER, &R.caps.max_cluster));
#else
    std::printf("Error: -s remote needs a DLMEM build\n");
    return 1;
#endif
  }
  std::printf("atomic_fp: %u cores, %u warps of %u threads, %u bytes of local memory\n",
              (uint32_t)R.caps.num_cores, (uint32_t)R.caps.num_warps, (uint32_t)R.caps.num_threads,
              (uint32_t)R.caps.lmem_size);

  const uint32_t w = req_w ? req_w : (space == SPACE_REMOTE ? 1 : (uint32_t)R.caps.num_warps);
  R.blocks = req_k ? req_k : (space == SPACE_REMOTE ? 2 : 1);
  R.nt = w * (uint32_t)R.caps.num_threads;
  const uint32_t need = std::max<uint32_t>({R.nt * SLOT_BYTES, HIST_BINS * 8, req_m * 4, 64});
  R.lmem = (need + 63) / 64 * 64;
  if (space == SPACE_REMOTE && (R.blocks < 2 || R.blocks > R.caps.max_cluster)) {
    std::printf("Error: K=%u is outside [2, %u]\n", R.blocks, (uint32_t)R.caps.max_cluster);
    return 1;
  }
  if (w > R.caps.num_warps || R.lmem > R.caps.lmem_size) {
    std::printf("Error: %u warps and %u bytes of local memory per block do not fit a core\n", w, R.lmem);
    return 1;
  }
  CHECK(vx_module_load_file(R.dev, kernel_file, &R.mod));

  switch (mode) {
  case MODE_EDGE:    run_edge(R); break;
  case MODE_HIST:    run_hist(R, false); break;
  case MODE_CONTEND: run_hist(R, true); break;
  case MODE_ALLPUSH: run_allpush(R); break;
  default:           run_f16loop(R); break;
  }

  vx_module_release(R.mod);
  vx_queue_release(R.q);
  vx_device_dump_perf(R.dev, stdout);
  vx_device_release(R.dev);
  if (g_errors) {
    std::printf("Found %d errors!\nFAILED!\n", g_errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
