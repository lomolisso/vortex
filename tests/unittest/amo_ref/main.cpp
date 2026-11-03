// amo_ref against hand-computed vectors, then against the H100 measurements
// (default: h100_atomics.csv.gz next to this file), bit for bit under the
// site rule of each row's space.

#include <amo_ref.h>
#include <probe_csv.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

using namespace vortex::amo_ref;

#ifndef PROBE_CSV
#define PROBE_CSV ""
#endif

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

// The loop-only 16-bit adds, through amo_ref::add_f16/add_bf16.
constexpr uint32_t SW_ADD_F16 = 100, SW_ADD_BF16 = 101;

uint64_t run(uint32_t op, uint32_t width, uint64_t old, uint64_t rhs, Site site) {
  if (op == SW_ADD_F16)  return add_f16((uint16_t)old, (uint16_t)rhs);
  if (op == SW_ADD_BF16) return add_bf16((uint16_t)old, (uint16_t)rhs);
  return apply(op, width, old, rhs, site);
}

constexpr Site LLC = SITE_LLC, LOOP = SITE_LOOP;

struct Vec {
  const char* name;
  uint32_t op, width;
  uint64_t old, rhs;
  Site site;
  uint64_t want;
};

const Vec kVectors[] = {
  {"inc 0,0",            OP_INC, W_W, 0, 0, LOOP, 0},
  {"inc 5,10",           OP_INC, W_W, 5, 10, LOOP, 6},
  {"inc 10,10",          OP_INC, W_W, 10, 10, LOOP, 0},
  {"inc 11,10",          OP_INC, W_W, 11, 10, LOOP, 0},
  {"inc max,max",        OP_INC, W_W, 0xffffffff, 0xffffffff, LOOP, 0},
  {"inc 9,max",          OP_INC, W_W, 9, 0xffffffff, LOOP, 10},
  {"inc 3,0",            OP_INC, W_W, 3, 0, LOOP, 0},
  {"dec 0,7",            OP_DEC, W_W, 0, 7, LOOP, 7},
  {"dec 5,7",            OP_DEC, W_W, 5, 7, LOOP, 4},
  {"dec 7,7",            OP_DEC, W_W, 7, 7, LOOP, 6},
  {"dec 8,7",            OP_DEC, W_W, 8, 7, LOOP, 7},
  {"dec 0,0",            OP_DEC, W_W, 0, 0, LOOP, 0},
  {"dec 1,0",            OP_DEC, W_W, 1, 0, LOOP, 0},
  {"dec masks rhs",      OP_DEC, W_W, 0, 0xabcd00000003ull, LOOP, 3},

  {"f32 1+2",            OP_FADD, W_W, 0x3f800000, 0x40000000, LOOP, 0x40400000},
  {"f32 tie to even",    OP_FADD, W_W, 0x3f800000, 0x33800000, LOOP, 0x3f800000},
  {"f32 tie up to even", OP_FADD, W_W, 0x3f800000, 0x34400000, LOOP, 0x3f800002},
  {"f32 inf-inf",        OP_FADD, W_W, 0x7f800000, 0xff800000, LOOP, F32_QNAN},
  {"f32 nan payload",    OP_FADD, W_W, 0x7fc12345, 0x3f800000, LOOP, F32_QNAN},
  {"f32 snan",           OP_FADD, W_W, 0x7f800001, 0x3f800000, LLC, F32_QNAN},
  {"f32 snan loop",      OP_FADD, W_W, 0x3f800000, 0xff800001, LOOP, F32_QNAN},
  {"f32 inf-inf llc",    OP_FADD, W_W, 0xff800000, 0x7f800000, LLC, F32_QNAN},
  {"f32 +0 + -0",        OP_FADD, W_W, 0x00000000, 0x80000000, LOOP, 0x00000000},
  {"f32 -0 + -0",        OP_FADD, W_W, 0x80000000, 0x80000000, LOOP, 0x80000000},
  {"f32 1 + -1",         OP_FADD, W_W, 0x3f800000, 0xbf800000, LOOP, 0x00000000},
  {"f32 overflow",       OP_FADD, W_W, 0x7f7fffff, 0x7f7fffff, LOOP, 0x7f800000},
  {"f32 -overflow",      OP_FADD, W_W, 0xff7fffff, 0xff7fffff, LLC, 0xff800000},
  {"f32 inf + 1",        OP_FADD, W_W, 0x7f800000, 0x3f800000, LLC, 0x7f800000},
  {"f32 sub+sub",        OP_FADD, W_W, 0x00000001, 0x00000001, LOOP, 0x00000002},
  {"f32 sub+sub ftz",    OP_FADD, W_W, 0x00000001, 0x00000001, LLC, 0x00000000},
  {"f32 -sub-sub ftz",   OP_FADD, W_W, 0x80000001, 0x80000001, LLC, 0x80000000},
  {"f32 norm-norm=sub",  OP_FADD, W_W, 0x00800001, 0x80800000, LOOP, 0x00000001},
  {"f32 n-n=sub ftz",    OP_FADD, W_W, 0x00800001, 0x80800000, LLC, 0x00000000},
  {"f32 -(n-n)=sub ftz", OP_FADD, W_W, 0x80800001, 0x00800000, LLC, 0x80000000},
  {"f32 sub+min",        OP_FADD, W_W, 0x00000001, 0x00800000, LOOP, 0x00800001},
  {"f32 sub+min ftz",    OP_FADD, W_W, 0x00000001, 0x00800000, LLC, 0x00800000},
  {"f32 largest sub x2", OP_FADD, W_W, 0x007fffff, 0x007fffff, LOOP, 0x00fffffe},
  {"f32 largest sub ftz",OP_FADD, W_W, 0x007fffff, 0x007fffff, LLC, 0x00000000},

  {"f64 1+2",            OP_FADD, W_D, 0x3ff0000000000000ull, 0x4000000000000000ull, LOOP, 0x4008000000000000ull},
  {"f64 sub keeps",      OP_FADD, W_D, 1, 1, LLC, 2},
  {"f64 inf-inf",        OP_FADD, W_D, 0x7ff0000000000000ull, 0xfff0000000000000ull, LOOP, 0xfff8000000000000ull},
  {"f64 inf-inf llc",    OP_FADD, W_D, 0xfff0000000000000ull, 0x7ff0000000000000ull, LLC, 0xfff8000000000000ull},
  {"f64 llc old qnan",   OP_FADD, W_D, 0x7ff8000000012345ull, 0x3ff0000000000000ull, LLC, 0x7ff8000000012345ull},
  {"f64 llc rhs snan",   OP_FADD, W_D, 0x3ff0000000000000ull, 0x7ff0000000000001ull, LLC, 0x7ff0000000000001ull},
  {"f64 llc old snan",   OP_FADD, W_D, 0xfff4000000000007ull, 0x7ff0000000000000ull, LLC, 0xfff4000000000007ull},
  {"f64 llc rhs first",  OP_FADD, W_D, 0x7ff8000000000001ull, 0xfff4000000000002ull, LLC, 0xfff4000000000002ull},
  {"f64 loop old snan",  OP_FADD, W_D, 0x7ff0000000000001ull, 0x3ff0000000000000ull, LOOP, 0x7ff8000000000001ull},
  {"f64 loop rhs nan",   OP_FADD, W_D, 0x3ff0000000000000ull, 0xfff0000000000003ull, LOOP, 0xfff8000000000003ull},
  {"f64 loop old first", OP_FADD, W_D, 0x7ff4000000000002ull, 0xfff8000000000005ull, LOOP, 0x7ffc000000000002ull},
  {"f64 loop nan + inf", OP_FADD, W_D, 0xfff0000000000000ull, 0x7ff8000000abcdefull, LOOP, 0x7ff8000000abcdefull},
  {"f64 tie to even",    OP_FADD, W_D, 0x3ff0000000000000ull, 0x3ca0000000000000ull, LOOP, 0x3ff0000000000000ull},
  {"f64 overflow",       OP_FADD, W_D, 0x7fefffffffffffffull, 0x7fefffffffffffffull, LOOP, 0x7ff0000000000000ull},

  {"f16 1+1",            SW_ADD_F16, W_H, 0x3c00, 0x3c00, LOOP, 0x4000},
  {"f16 65504+16=inf",   SW_ADD_F16, W_H, 0x7bff, 0x4c00, LOOP, 0x7c00},
  {"f16 65504+8",        SW_ADD_F16, W_H, 0x7bff, 0x4800, LOOP, 0x7bff},
  {"f16 sub+sub",        SW_ADD_F16, W_H, 0x0001, 0x0001, LLC, 0x0002},
  {"f16 -sub+sub",       SW_ADD_F16, W_H, 0x8001, 0x0001, LOOP, 0x0000},
  {"f16 inf-inf",        SW_ADD_F16, W_H, 0x7c00, 0xfc00, LOOP, F16_QNAN},
  {"f16 nan",            SW_ADD_F16, W_H, 0x7e01, 0x3c00, LOOP, F16_QNAN},
  {"f16 tie to even",    SW_ADD_F16, W_H, 0x3c00, 0x1000, LOOP, 0x3c00},
  {"f16 tie up",         SW_ADD_F16, W_H, 0x3c00, 0x1600, LOOP, 0x3c02},
  {"f16 masks old",      SW_ADD_F16, W_H, 0xffff3c00ull, 0x3c00, LOOP, 0x4000},
  {"f16 sub to normal",  SW_ADD_F16, W_H, 0x03ff, 0x0001, LOOP, 0x0400},

  {"bf16 1+1",           SW_ADD_BF16, W_H, 0x3f80, 0x3f80, LOOP, 0x4000},
  {"bf16 tie to even",   SW_ADD_BF16, W_H, 0x3f80, 0x3b80, LOOP, 0x3f80},
  {"bf16 tie up",        SW_ADD_BF16, W_H, 0x3f80, 0x3c40, LOOP, 0x3f82},
  {"bf16 sub+sub",       SW_ADD_BF16, W_H, 0x0001, 0x0001, LLC, 0x0002},
  {"bf16 inf-inf",       SW_ADD_BF16, W_H, 0x7f80, 0xff80, LOOP, BF16_QNAN},
  {"bf16 overflow",      SW_ADD_BF16, W_H, 0x7f7f, 0x7f7f, LOOP, 0x7f80},
  {"bf16 nan",           SW_ADD_BF16, W_H, 0x7fc0, 0x3f80, LOOP, BF16_QNAN},
  {"bf16 -0 + -0",       SW_ADD_BF16, W_H, 0x8000, 0x8000, LOOP, 0x8000},

  {"f16x2",              OP_FADD2, W_W, 0x7c003c00, 0xfc003c00, LOOP, (uint64_t)F16_QNAN << 16 | 0x4000},
  {"f16x2 sub lanes",    OP_FADD2, W_W, 0x00010001, 0x00010001, LLC, 0x00020002},
  {"f32x2 ftz",          OP_FADD2, W_D, 0x3f80000000000001ull, 0x4000000000000001ull, LLC, 0x4040000000000000ull},
  {"f32x2 noftz",        OP_FADD2, W_D, 0x3f80000000000001ull, 0x4000000000000001ull, LOOP, 0x4040000000000002ull},
  {"f32x2 nan hi",       OP_FADD2, W_D, 0x7f80000000000000ull, 0xff80000000000000ull, LOOP, (uint64_t)F32_QNAN << 32},
  {"bf16x2",             OP_FADD2_BF, W_W, 0x3f803f80, 0x3f80bf80, LOOP, 0x40000000},

  {"fmin2 f16",          OP_FMIN2, W_W, 0x3c008000, 0x40000000, LOOP, 0x3c008000},
  {"fmax2 f16",          OP_FMAX2, W_W, 0x3c008000, 0x40000000, LOOP, 0x40000000},
  {"fmin2 f16 +0,-0",    OP_FMIN2, W_W, 0x00000000, 0x80008000, LOOP, 0x80008000},
  {"fmax2 f16 neg",      OP_FMAX2, W_W, 0xbc00c000, 0xc000bc00, LOOP, 0xbc00bc00},
  {"fmin2 f16 neg",      OP_FMIN2, W_W, 0xbc00c000, 0xc000bc00, LOOP, 0xc000c000},
  {"fmin2 f16 nan",      OP_FMIN2, W_W, 0x7e007e00, 0x7e013c00, LOOP, (uint64_t)F16_QNAN << 16 | 0x3c00},
  {"fmax2 f16 inf",      OP_FMAX2, W_W, 0x7c00fc00, 0x3c003c00, LOOP, 0x7c003c00},
  {"fmin2 bf16",         OP_FMIN2_BF, W_W, 0xbf803f80, 0x3f807fc0, LOOP, 0xbf803f80},
  {"fmax2 bf16",         OP_FMAX2_BF, W_W, 0xbf803f80, 0x3f807fc0, LOOP, 0x3f803f80},
  {"fmax2 bf16 nans",    OP_FMAX2_BF, W_W, 0x7fc17fc0, 0xffc10000, LOOP, (uint64_t)BF16_QNAN << 16 | 0x0000},
};

void run_vectors() {
  for (auto& v : kVectors)
    expect(v.name, run(v.op, v.width, v.old, v.rhs, v.site), v.want);

  expect("cas w hit",   cas(W_W, 5, 5, 7), 7);
  expect("cas w miss",  cas(W_W, 5, 6, 7), 5);
  expect("cas w masks", cas(W_W, 0xffffffff00000005ull, 5, 7), 7);
  expect("cas h hit",   cas(W_H, 0x1234, 0x1234, 0xabcd), 0xabcd);
  expect("cas h miss",  cas(W_H, 0x1234, 0x1235, 0xabcd), 0x1234);
  expect("cas d hit",   cas(W_D, 0x8000000000000001ull, 0x8000000000000001ull, 2), 2);
  expect("apply cas",   apply(OP_CAS, W_W, 5, 7, LOOP), 5);

  expect("ret w",  ret_value(W_W, 0x80000000), 0xffffffff80000000ull);
  expect("ret h",  ret_value(W_H, 0x8000), 0xffffffffffff8000ull);
  expect("ret h+", ret_value(W_H, 0x7fff), 0x7fff);
  expect("ret d",  ret_value(W_D, 0x8000000000000000ull), 0x8000000000000000ull);

  struct L { uint32_t op, w; bool ok; };
  const L legal_cases[] = {
    {OP_INC, W_W, true}, {OP_INC, W_D, false}, {OP_INC, W_H, false}, {OP_DEC, W_W, true},
    {OP_FADD, W_W, true}, {OP_FADD, W_D, true}, {OP_FADD, 0, false},
    {OP_FADD_BF, W_W, false}, {OP_FADD_BF, W_D, false},
    {OP_FADD2, W_W, true}, {OP_FADD2, W_D, true}, {OP_FADD2, W_H, false},
    {OP_FADD2_BF, W_W, true}, {OP_FADD2_BF, W_D, false},
    {OP_FMIN2, W_W, true}, {OP_FMIN2, W_D, false}, {OP_FMAX2_BF, W_W, true}, {OP_FMAX2_BF, W_H, false},
    {OP_CAS, W_W, true}, {OP_CAS, W_D, true},
    {26, W_W, false}, {31, W_W, false}, {15, W_D, false}, {0, W_W, false},
  };
  for (auto& c : legal_cases) {
    char what[64];
    std::snprintf(what, sizeof(what), "legal(%u,%u)", c.op, c.w);
    expect(what, legal(c.op, c.w), c.ok);
  }
  expect("is_fp inc",  is_fp(OP_INC), 0);
  expect("is_fp fadd", is_fp(OP_FADD), 1);
  expect("is_fp fmax2_bf", is_fp(OP_FMAX2_BF), 1);
  expect("is_fp 26",   is_fp(26), 0);
  expect("illegal returns old", apply(OP_FADD_BF, W_W, 0x3f803f80, 0x3f803f80, LOOP), 0x3f803f80);
  expect("legal fadd h", legal(OP_FADD, W_H), 0);
  expect("legal fadd_bf h", legal(OP_FADD_BF, W_H), 0);
  expect("legal cas h", legal(OP_CAS, W_H), 0);
  expect("fadd h returns old", apply(OP_FADD, W_H, 0x3c00, 0x3c00, LLC), 0x3c00);
}

// Every measured row against amo_ref, bit for bit over the operand bytes,
// under the site rule of the row's space; the returned value must be the old
// one and the guard bytes intact.
int run_probe(const std::string& path) {
  std::vector<probe_csv::Row> rows;
  std::string msg;
  if (!probe_csv::load(path, rows, msg)) {
    std::printf("probe CSV: FAILED (%s)\n", msg.c_str());
    return 1;
  }
  struct G { int rows = 0, ok = 0, ret_bad = 0, guard_bad = 0; Site site = LOOP; };
  std::map<std::string, G> groups;
  std::map<std::string, int> unmapped, excluded;
  int fails = 0, checked = 0;
  for (auto& r : rows) {
    const std::string key = r.form + " " + r.op + " " + r.space + " " + r.variant;
    if (r.f.kind == probe_csv::K_NONE) {
      ++unmapped[r.form + " " + r.op];
      continue;
    }
    if (probe_csv::excluded(r)) {
      ++excluded[key];
      continue;
    }
    G& g = groups[key];
    g.site = probe_csv::row_site(r);
    ++g.rows;
    ++checked;
    const uint32_t nb = r.f.bytes;
    const uint64_t mlo = (nb >= 8) ? ~0ull : ((1ull << (8 * nb)) - 1);
    const uint64_t mhi = (nb >= 16) ? ~0ull : (nb > 8 ? ((1ull << (8 * (nb - 8))) - 1) : 0);
    auto eq = [&](const probe_csv::V128& x, const probe_csv::V128& y) {
      return ((x.lo ^ y.lo) & mlo) == 0 && ((x.hi ^ y.hi) & mhi) == 0;
    };
    const auto w = probe_csv::expect_mem(r, g.site);
    const bool mem_ok = eq(r.mem, w);
    const bool ret_ok = !r.has_ret || eq(r.ret, r.old);
    g.ok += mem_ok;
    g.ret_bad += !ret_ok;
    g.guard_bad += !r.guard_ok;
    if (!(mem_ok && ret_ok && r.guard_ok)) {
      if (fails < 40) {
        std::printf("FAIL probe %s: old %016" PRIx64 "%016" PRIx64 " rhs %016" PRIx64 "%016" PRIx64
                    " H100 mem %016" PRIx64 "%016" PRIx64 " amo_ref %016" PRIx64 "%016" PRIx64 "%s%s\n",
                    key.c_str(), r.old.hi, r.old.lo, r.rhs.hi, r.rhs.lo, r.mem.hi, r.mem.lo, w.hi, w.lo,
                    ret_ok ? "" : " (ret != old)", r.guard_ok ? "" : " (guard)");
      }
      ++fails;
    }
  }
  std::printf("probe CSV %s: %zu rows\n", path.c_str(), rows.size());
  for (auto& kv : groups) {
    const G& g = kv.second;
    std::printf("  %-48s rows %5d  match %5d  site %s%s%s\n", kv.first.c_str(), g.rows, g.ok,
                g.site == LLC ? "llc" : "loop", g.ret_bad ? "  RET MISMATCH" : "", g.guard_bad ? "  GUARD" : "");
  }
  int n_excl = 0, n_unm = 0;
  for (auto& e : excluded) {
    std::printf("  excluded (no atomic on the H100): %s (%d rows)\n", e.first.c_str(), e.second);
    n_excl += e.second;
  }
  for (auto& u : unmapped) {
    std::printf("  not modelled: %s (%d rows)\n", u.first.c_str(), u.second);
    n_unm += u.second;
  }
  std::printf("probe CSV: %d rows checked, %d match, %d mismatch; %d excluded, %d not modelled\n",
              checked, checked - fails, fails, n_excl, n_unm);
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
  std::string csv = PROBE_CSV;  // "" or --csv= skips the measured rows
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--csv=", 6) == 0)
      csv = argv[i] + 6;
  }
  run_vectors();
  std::printf("hand vectors: %d passed, %d failed\n", g_pass, g_fail);
  int fails = g_fail;
  if (!csv.empty()) {
    const std::string path = probe_csv::locate(csv);
    fails += run_probe(path.empty() ? csv : path);
  }
  std::printf("amo_ref: %s\n", fails ? "FAILED" : "PASSED");
  return fails ? 1 : 0;
}
