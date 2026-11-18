#include <vortex2.h>
#include <uniform_ref.h>
#include <vx_fe_isa.h>
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)


namespace {

const char* kernel_file = "kernel.vxbin";
bool verbose = false;
int errors = 0;

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;
vx_kernel_h kernel = nullptr;
uint32_t num_threads = 0, num_warps = 0;
const uint32_t xlen = VX_CFG_XLEN;

void fail(const char* fmt, uint64_t a, uint64_t b, uint64_t c) {
  if (++errors <= 20) {
    std::printf("FAIL ");
    std::printf(fmt, (unsigned long long)a, (unsigned long long)b, (unsigned long long)c);
    std::printf("\n");
  }
}

// One launch of a block of `warps` warps; returns the output buffer.
std::vector<uint64_t> launch(ur_arg_t arg, const std::vector<ur_case_t>& cases, uint32_t warps, size_t out_words) {
  std::vector<uint64_t> out(out_words, UR_SENTINEL);
  vx_buffer_h cbuf = nullptr, obuf = nullptr;
  const uint64_t cbytes = std::max<uint64_t>(sizeof(ur_case_t), cases.size() * sizeof(ur_case_t));
  CHECK(vx_buffer_create(dev, cbytes, VX_MEM_READ, &cbuf));
  CHECK(vx_buffer_create(dev, out.size() * 8, VX_MEM_READ_WRITE, &obuf));
  if (!cases.empty())
    CHECK(vx_enqueue_write(q, cbuf, 0, cases.data(), cases.size() * sizeof(ur_case_t), 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, obuf, 0, out.data(), out.size() * 8, 0, nullptr, nullptr));
  CHECK(vx_buffer_address(cbuf, &arg.cases_addr));
  CHECK(vx_buffer_address(obuf, &arg.out_addr));

  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.grid_dim[1]  = 1;
  li.grid_dim[2]  = 1;
  li.block_dim[0] = warps * num_threads;
  li.block_dim[1] = 1;
  li.block_dim[2] = 1;
  vx_event_h ev = nullptr, rd = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
  CHECK(vx_enqueue_read(q, out.data(), obuf, 0, out.size() * 8, 1, &ev, &rd));
  CHECK(vx_event_wait_value(rd, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(rd);
  vx_event_release(ev);
  CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
  vx_buffer_release(cbuf);
  vx_buffer_release(obuf);
  return out;
}


uint64_t xmask() { return (xlen == 64) ? ~0ull : 0xffffffffull; }

// Every U-ALU/U-CMP op, R2UR+UR2R and URZ on seeded values and active masks (full, one
// lane, random), then S2UR of every selector on full and partial masks.
void run_alu(uint32_t seed, uint32_t ncases) {
  std::mt19937_64 rng(seed);
  const uint64_t full = (num_threads == 64) ? ~0ull : ((1ull << num_threads) - 1);
  std::vector<ur_case_t> cases;
  for (uint32_t c = 0; c < ncases; ++c) {
    ur_case_t t;
    std::memset(&t, 0, sizeof(t));
    t.code = (c < UR_NUM_CODES - 1) ? c : uint32_t(rng() % (UR_NUM_CODES - 1));
    switch (c % 3) {
    case 0: t.tmask = full; break;
    case 1: t.tmask = 1ull << (rng() % num_threads); break;
    default: t.tmask = rng() & full; if (!t.tmask) t.tmask = full; break;
    }
    for (uint32_t i = 0; i < num_threads; ++i) {
      t.a[i] = rng();
      // small shift amounts and equal pairs are common
      t.b[i] = (rng() & 3) == 0 ? t.a[i] : ((rng() & 1) ? (rng() & 0x7f) : rng());
    }
    cases.push_back(t);
  }
  for (uint64_t m : {full, uint64_t(1) << (num_threads - 1), uint64_t(full & 0xaaaaaaaaaaaaaaaaull)}) {
    ur_case_t t;
    std::memset(&t, 0, sizeof(t));
    t.code = CASE_S2UR;
    t.tmask = m;
    cases.push_back(t);
  }
  ur_arg_t arg = {};
  arg.mode = MODE_ALU;
  arg.ncases = (uint32_t)cases.size();
  const size_t stride = (size_t)num_threads * S2UR_WORDS;
  const auto out = launch(arg, cases, num_warps, cases.size() * stride);
  for (size_t c = 0; c < cases.size(); ++c) {
    const auto& t = cases[c];
    const uint64_t* o = out.data() + c * stride;
    uint32_t first = 0;
    while (!((t.tmask >> first) & 1))
      ++first;
    if (t.code == CASE_S2UR) {
      for (uint32_t i = 0; i < num_threads; ++i) {
        if (!((t.tmask >> i) & 1))
          continue;
        const uint64_t* l = o + i * S2UR_WORDS;
        if (!(l[0] <= l[1] && l[1] <= l[2]))
          fail("S2UR clock not between two cycle reads: case %llu lane %llu s2ur %llu", c, i, l[1]);
        if (xlen == 32 && l[3] > l[54])
          fail("S2UR clock high after the CSR: case %llu lane %llu s2ur %llu", c, i, l[3]);
        for (uint32_t s = 2; s < vortex::fe::S2UR_NUM_SELECTORS; ++s) {
          // the first active lane's CSR value
          const uint64_t want = o[first * S2UR_WORDS + 2 * s + 1];
          if ((l[2 * s] & xmask()) != (want & xmask()))
            fail("S2UR selector %llu lane %llu: 0x%llx", s, i, l[2 * s]);
        }
      }
      continue;
    }
    uint64_t want = (t.code == CASE_URZ) ? 0 : uniform_ref::execute(t.code, xlen, t.a[first], t.b[first]);
    for (uint32_t i = 0; i < num_threads; ++i) {
      const bool on = (t.tmask >> i) & 1;
      const uint64_t got = o[i];
      if (on ? ((got & xmask()) != want) : (got != UR_SENTINEL)) {
        fail("case %llu lane %llu code %llu", c, i, t.code);
        if (errors <= 20)
          std::printf("  got 0x%llx want 0x%llx (first lane %u)\n", (unsigned long long)got,
                      (unsigned long long)want, first);
      }
    }
  }
  std::printf("alu: %zu cases, seed %u\n", cases.size(), seed);
}

// Every ALU, M and FPU class with a UR in a source slot equals its GPR form; the prefix
// issues nothing (equal instret for the prefixed and the plain block).
void run_operand(uint64_t u) {
  ur_arg_t arg = {};
  arg.mode = MODE_OPERAND;
  arg.u = u;
  const size_t words = (size_t)num_threads * OPD_MAX * 2 + 3;
  const auto out = launch(arg, {}, 1, words);
  const uint32_t nent = (xlen == 64) ? 35 : 33;
  for (uint32_t i = 0; i < num_threads; ++i) {
    for (uint32_t k = 0; k < nent; ++k) {
      const uint64_t g = out[(i * OPD_MAX + k) * 2], p = out[(i * OPD_MAX + k) * 2 + 1];
      if (g == UR_SENTINEL || g != p) {
        fail("operand entry %llu lane %llu: GPR form 0x%llx", k, i, g);
        if (errors <= 20)
          std::printf("  UR form 0x%llx\n", (unsigned long long)p);
      }
    }
  }
  const uint64_t* c = out.data() + (size_t)num_threads * OPD_MAX * 2;
  std::printf("operand: %u entries x %u lanes, instret prefixed %llu plain %llu\n", nent, num_threads,
              (unsigned long long)c[0], (unsigned long long)c[1]);
  if (c[0] != c[1])
    fail("the UPFX block retired %llu instructions, the plain block %llu", c[0], c[1], 0);
}

// Per-op cycles of a dependent U-ALU chain, S2UR -> dependent U-ALU, and independent
// S2URs of one warp, against the same chains on the vector ALU (add) and the CSR unit
// (csrr -> add). The U-ALU has VX_alu_int's one output register and S2UR is the CSR
// unit's read, so each uniform chain steps as its counterpart; the steps compared are
// differential (32-op body minus 16-op body), so the loop glue, which lands on the
// vector ALU, cancels. Independent S2URs pipeline (no per-warp interval): spaced less
// than a dependent S2UR step. H100 values are printed.
void run_latency(uint32_t iters, uint32_t warps) {
  constexpr double kTol = 1;
  ur_arg_t arg = {};
  arg.mode = MODE_LATENCY;
  arg.iters = iters;
  const auto out = launch(arg, {}, warps, (size_t)warps * LAT_RECS);
  const double ops = double(iters) * UR_CHAIN;
  for (uint32_t w = 0; w < warps; ++w) {
    const uint64_t* rec = out.data() + w * LAT_RECS;
    const double ualu = rec[0] / ops, s2ur = rec[1] / ops, b2b = rec[2] / ops, add = rec[4] / ops,
                 csr = rec[5] / ops;
    auto diff = [&](int i2, int i1) { return (double(rec[i2]) - double(rec[i1])) / ops; };
    const double d_ualu = diff(6, 0), d_s2ur = diff(7, 1), d_add = diff(8, 4), d_csr = diff(9, 5);
    std::printf("LAT: warps=%u warp=%u ualu=%.3f s2ur=%.3f b2b=%.3f add=%.3f csrr=%.3f\n", warps, w, ualu, s2ur,
                b2b, add, csr);
    std::printf("LAT: warps=%u warp=%u differential ualu=%.3f s2ur=%.3f add=%.3f csrr=%.3f\n", warps, w, d_ualu,
                d_s2ur, d_add, d_csr);
    if (w == 0) {
      std::printf("H100REF metric=ualu_dep_step h100=10.0 vortex=%.3f\n", ualu);
      std::printf("H100REF metric=s2ur_dep_step h100=7.5 vortex=%.3f\n", s2ur);
      std::printf("H100REF metric=s2ur_spacing_w%u h100=8.0 vortex=%.3f\n", warps, b2b);
    }
    if (std::abs(d_ualu - d_add) > kTol)
      fail("U-ALU step %llu milli-cycles, the vector add's %llu", (uint64_t)(d_ualu * 1000),
           (uint64_t)(d_add * 1000), 0);
    if (std::abs(d_s2ur - d_csr) > kTol)
      fail("S2UR step %llu milli-cycles, csrr's %llu", (uint64_t)(d_s2ur * 1000), (uint64_t)(d_csr * 1000), 0);
    if (b2b >= s2ur)
      fail("independent S2URs %llu milli-cycles apart, not below the dependent step %llu", (uint64_t)(b2b * 1000),
           (uint64_t)(s2ur * 1000), 0);
  }
}

} // namespace

int main(int argc, char** argv) {
  char mode = 'a';
  uint32_t seed = 1, ncases = 128, iters = 16, warps = 1;
  int c;
  while ((c = getopt(argc, argv, "m:s:n:i:w:vh")) != -1) {
    switch (c) {
    case 'm': mode = optarg[0]; break;
    case 's': seed = (uint32_t)std::atoi(optarg); break;
    case 'n': ncases = (uint32_t)std::atoi(optarg); break;
    case 'i': iters = (uint32_t)std::atoi(optarg); break;
    case 'w': warps = (uint32_t)std::atoi(optarg); break;
    case 'v': verbose = true; break;
    default:
      std::printf("usage: %s [-m a|o|l|x] [-s seed] [-n cases] [-i iters] [-w warps]\n", argv[0]);
      return (c == 'h') ? 0 : 1;
    }
  }

  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
  CHECK(vx_queue_create(dev, &qi, &q));
  uint64_t nt = 0, nw = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  num_threads = (uint32_t)nt;
  num_warps = (uint32_t)nw;
  if (num_threads > UR_MAX_LANES || warps > num_warps) {
    std::printf("uniform_regs: %u threads (at most %u), %u warps asked of %u\nFAILED!\n", num_threads,
                UR_MAX_LANES, warps, num_warps);
    return 1;
  }
  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "ur_main", &kernel));

  const uint64_t u = 0x0123456789abcdefull & ((xlen == 64) ? ~0ull : 0x7fffffffull);
  switch (mode) {
  case 'a': run_alu(seed, ncases); break;
  case 'o': run_operand(u); break;
  case 'l': run_latency(iters, warps); break;
  case 'x': run_alu(seed, 32); run_operand(u); break;
  default: std::printf("unknown mode %c\n", mode); return 1;
  }

  vx_kernel_release(kernel);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);
  if (errors) {
    std::printf("Found %d errors!\nFAILED!\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
