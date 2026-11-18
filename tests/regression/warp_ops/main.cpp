// ELECT, MATCH and REDUX as warp instructions:
//   -m b  bit-exact: seeded ops on random active masks, every active lane's
//         result equal to the golden (sw/common/warp_ops_ref.h), inactive lanes untouched;
//   -m l  latency: dependent chains of one warp with the same glue for every op;
//         MATCH.ANY's step grows by exactly one cycle (one comparator-row pass) per
//         distinct value, and every other op steps as MATCH.ANY of its width with k
//         = the op's unit latency (warp_ops_ref::latency: both leave the same unit
//         through the same port), within one cycle, plus for a UR result (REDUX)
//         the UR2R hop, ELECT.L's step over ELECT.P's;
//   -m r  rate: WOP_ILP independent ops per warp, one warp and every warp of the
//         SM; never above the units' width (one warp op per cycle per unit), and
//         MATCH.ANY at k >= 4 within 5% of one op per k cycles of the shared unit;
//         a per-block unit at most k + 1 cycles per op (its result port is shared
//         round robin with the block's ALU results, so a last pass waits at most one).
// H100 measurements are printed as H100REF lines for task G, not checked.

#include <vortex2.h>
#include <warp_ops_ref.h>
#include "common.h"
#include "../../unittest/warp_ops/stream.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

namespace wop = vortex::warp_ops_ref;
namespace ws = warp_ops_stream;

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
std::vector<uint64_t> launch(wop_arg_t arg, const std::vector<wop_case_t>& cases, uint32_t warps, size_t out_words) {
  std::vector<uint64_t> out(out_words, WOP_SENTINEL);
  vx_buffer_h cbuf = nullptr, obuf = nullptr;
  const uint64_t cbytes = std::max<uint64_t>(sizeof(wop_case_t), cases.size() * sizeof(wop_case_t));
  CHECK(vx_buffer_create(dev, cbytes, VX_MEM_READ, &cbuf));
  CHECK(vx_buffer_create(dev, out.size() * 8, VX_MEM_READ_WRITE, &obuf));
  if (!cases.empty())
    CHECK(vx_enqueue_write(q, cbuf, 0, cases.data(), cases.size() * sizeof(wop_case_t), 0, nullptr, nullptr));
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

void run_bitexact(uint32_t seed, uint32_t ncases) {
  const auto ops = ws::make_stream(seed, ncases, num_threads, 1, xlen);
  std::vector<wop_case_t> cases(ops.size());
  for (size_t c = 0; c < ops.size(); ++c) {
    std::memset(&cases[c], 0, sizeof(wop_case_t));
    cases[c].code = ops[c].code;
    cases[c].tmask = ops[c].tmask;
    for (uint32_t i = 0; i < num_threads; ++i) {
      cases[c].rs1[i] = ops[c].rs1[i];
      cases[c].rs2[i] = ops[c].rs2[i];
    }
  }
  wop_arg_t arg = {};
  arg.mode = MODE_BITEXACT;
  arg.ncases = ncases;
  const auto out = launch(arg, cases, num_warps, (size_t)ncases * num_threads);
  const uint64_t xmask = (xlen == 64) ? ~0ull : 0xffffffffull;
  for (size_t c = 0; c < ops.size(); ++c) {
    std::vector<uint64_t> want(num_threads, WOP_SENTINEL);
    wop::execute(ops[c].code, xlen, num_threads, ops[c].tmask, ops[c].rs1.data(), ops[c].rs2.data(), want.data());
    for (uint32_t i = 0; i < num_threads; ++i) {
      const bool on = (ops[c].tmask >> i) & 1;
      const uint64_t got = out[c * num_threads + i];
      if (on ? ((got & xmask) != (want[i] & xmask)) : (got != WOP_SENTINEL))
        fail("case %llu lane %llu code 0x%llx", c, i, ops[c].code), std::printf("  got 0x%llx want 0x%llx\n",
             (unsigned long long)got, (unsigned long long)want[i]);
    }
  }
  std::printf("bitexact: %u cases, seed %u\n", ncases, seed);
}

double measure(uint32_t mode, uint32_t code, uint32_t k, uint32_t iters, uint32_t warps) {
  wop_arg_t arg = {};
  arg.mode = mode;
  arg.code = code;
  arg.k = k;
  arg.iters = iters;
  const auto out = launch(arg, {}, warps, (size_t)warps * 4);
  uint64_t t0 = ~0ull, t1 = 0;
  for (uint32_t w = 0; w < (mode == MODE_LATENCY ? 1u : warps); ++w) {
    t0 = std::min(t0, out[w * 4 + 0]);
    t1 = std::max(t1, out[w * 4 + 1]);
  }
  if (mode == MODE_RATE && warps > 1) {
    uint64_t s0 = ~0ull, s1 = 0, d0 = ~0ull, d1 = 0;
    for (uint32_t w = 0; w < warps; ++w) {
      s0 = std::min(s0, out[w * 4 + 0] - t0);
      s1 = std::max(s1, out[w * 4 + 0] - t0);
      d0 = std::min(d0, out[w * 4 + 1] - t0);
      d1 = std::max(d1, out[w * 4 + 1] - t0);
    }
    std::printf("rate: code 0x%02x W %u: warps start +%llu..+%llu, done +%llu..+%llu cycles\n", code, warps,
                (unsigned long long)s0, (unsigned long long)s1, (unsigned long long)d0, (unsigned long long)d1);
  }
  return double(t1 - t0);
}

// H100 per-step latencies of a dependent chain (02 §4.7), MATCH.ANY as base + per value * k.
uint32_t h100_latency(uint32_t code, uint32_t k) {
  if (wop::is_elect(code))
    return 4;
  if (wop::is_redux(code))
    return 38;
  if (wop::is_match_all(code))
    return 28;
  return (code == wop::MATCH_ANY_B64) ? 19 + 13 * k : 17 + 11 * k;
}

void run_latency(uint32_t iters, int64_t c0) {
  if (iters == 0 || iters % WOP_CHAIN_UNROLL)
    fail("-i %llu: the chain runs %llu steps per loop pass", iters, WOP_CHAIN_UNROLL, 0), std::exit(1);
  const uint32_t ks[] = {1, 2, 4, 8, 16, 32};
  // One warp's fetch and dependent glue set a floor per step that hides the unit's
  // latency for small k; above it, each further pass adds one cycle per step.
  for (uint32_t code : {(uint32_t)wop::MATCH_ANY_B32, (uint32_t)wop::MATCH_ANY_B64}) {
    uint64_t prev = 0, first = 0;
    uint32_t prev_k = 0, sloped = 0;
    for (uint32_t k : ks) {
      if (k > num_threads)
        break;
      const uint64_t cyc = (uint64_t)measure(MODE_LATENCY, code, k, iters, 1);
      const uint32_t unit = wop::latency(num_threads, code, k);
      std::printf("latency: code 0x%02x k %2u: %llu cycles / %u = %.2f per step (unit %u)\n", code, k,
                  (unsigned long long)cyc, iters, double(cyc) / iters, unit);
      std::printf("H100REF metric=warpop_latency_0x%02x_k%u h100=%u vortex=%.2f\n", code, k,
                  h100_latency(code, k), double(cyc) / iters);
      if (!prev_k)
        first = cyc;
      if (prev_k && cyc < prev)
        fail("MATCH.ANY: %llu cycles with %llu values, fewer than with %llu", cyc, k, prev_k);
      sloped += (prev_k && prev > first);
      if (prev_k && prev > first && cyc - prev != (uint64_t)iters * (k - prev_k))
        fail("MATCH.ANY slope: %llu cycles over %llu more values, want 1 per value per step", cyc - prev,
             k - prev_k, 1);
      if (cyc < (uint64_t)iters * unit)
        fail("step below the unit's latency: %llu cycles, %llu steps, unit %llu", cyc, iters, unit);
      if (c0 >= 0 && cyc > (uint64_t)iters * (unit + c0))
        fail("step above unit + c0: %llu cycles, %llu steps, unit %llu", cyc, iters, unit);
      prev = cyc;
      prev_k = k;
    }
    if (num_threads >= 32 && !sloped)
      fail("MATCH.ANY 0x%llx: no k above the per-step floor up to %llu values (%llu pairs)", code, num_threads, sloped);
  }
  // ELECT.P and ELECT.L differ only in the result's register file: ELECT.L's (and
  // REDUX's) goes to a UR and the intrinsic's UR2R moves it, one more dependent hop.
  int64_t ur_hop = -1;
  uint64_t elect_p = 0;
  for (uint32_t code : {(uint32_t)wop::MATCH_ALL_B32, (uint32_t)wop::MATCH_ALL_B64, (uint32_t)wop::ELECT_P,
                        (uint32_t)wop::ELECT_L, (uint32_t)wop::REDUX_ADD}) {
    const uint64_t cyc = (uint64_t)measure(MODE_LATENCY, code, 1, iters, 1);
    const uint32_t unit = wop::latency(num_threads, code, 1);
    std::printf("latency: code 0x%02x: %.2f per step (unit %u)\n", code, double(cyc) / iters, unit);
    std::printf("H100REF metric=warpop_latency_0x%02x h100=%u vortex=%.2f\n", code, h100_latency(code, 1),
                double(cyc) / iters);
    if (cyc < (uint64_t)iters * unit)
      fail("step below the unit's latency: %llu cycles, %llu steps, unit %llu", cyc, iters, unit);
    if (c0 >= 0 && cyc > (uint64_t)iters * (unit + c0))
      fail("step above unit + c0: %llu cycles, %llu steps, unit %llu", cyc, iters, unit);
    if (code == wop::ELECT_P)
      elect_p = cyc;
    if (code == wop::ELECT_L) {
      ur_hop = (int64_t)cyc - (int64_t)elect_p;
      std::printf("latency: UR2R hop %.2f per step\n", double(ur_hop) / iters);
      if (ur_hop < (int64_t)iters)
        fail("ELECT.L %llu cycles, ELECT.P %llu: no UR2R hop", cyc, elect_p, 0);
      continue;
    }
    const uint32_t any = wop::is_b64(code) ? (uint32_t)wop::MATCH_ANY_B64 : (uint32_t)wop::MATCH_ANY_B32;
    if (unit <= num_threads) {
      const uint64_t ref = (uint64_t)measure(MODE_LATENCY, any, unit, iters, 1);
      const int64_t want = (int64_t)ref + (wop::writes_ur(code) ? ur_hop : 0);
      std::printf("latency: code 0x%02x against MATCH.ANY 0x%02x k %u: %.2f per step, want %.2f\n", code, any,
                  unit, double(ref) / iters, double(want) / iters);
      if (std::llabs((int64_t)cyc - want) > (int64_t)iters)
        fail("code 0x%llx: %llu cycles, want MATCH.ANY at k = its unit latency (+ UR2R) %llu", code, cyc,
             (uint64_t)want);
    }
  }
}

void run_rate(uint32_t iters) {
  const uint32_t units = VX_CFG_WARP_OP_SHARED ? 1 : VX_CFG_NUM_ALU_BLOCKS;
  const double unit_lanes = double(num_threads) * units;
  // H100: one warp's REDUX every 11.5 cycles, MATCH.ALL every 8; 16 lanes/clk/SM, MATCH.ANY 16/k.
  struct Form { uint32_t code, k; double h100_w1; };
  const Form forms[] = {{wop::REDUX_ADD, 1, 11.5}, {wop::MATCH_ALL_B32, 1, 8.0},
                        {wop::MATCH_ANY_B32, 4, 0}, {wop::MATCH_ANY_B32, 8, 0}};
  for (auto& f : forms) {
    const double ops = double(iters) * WOP_ILP;
    const double c1 = measure(MODE_RATE, f.code, f.k, iters, 1);
    std::printf("rate: code 0x%02x k %u W 1: %.2f cycles per instruction\n", f.code, f.k, c1 / ops);
    if (f.h100_w1 > 0)
      std::printf("H100REF metric=warpop_w1_cpi_0x%02x h100=%.1f vortex=%.2f\n", f.code, f.h100_w1, c1 / ops);
    const double cw = measure(MODE_RATE, f.code, f.k, iters, num_warps);
    const double lanes = ops * num_warps * num_threads / cw;
    const bool any = wop::is_match_any(f.code);
    const double cap = unit_lanes / (any ? f.k : 1);
    std::printf("rate: code 0x%02x k %u W %u: %.2f lanes/clk/SM (units' width %.2f)\n", f.code, f.k, num_warps,
                lanes, cap);
    std::printf("H100REF metric=warpop_lanes_per_clk_0x%02x_k%u h100=%.2f vortex=%.2f\n", f.code, f.k,
                16.0 / (any ? f.k : 1), lanes);
    if (lanes > cap * 1.0001)
      fail("rate above the units' width: code 0x%llx, %llu milli-lanes/clk, width %llu", f.code,
           (uint64_t)(lanes * 1000), (uint64_t)(cap * 1000));
    if (any && f.k >= 4 && VX_CFG_WARP_OP_SHARED && lanes < cap * 0.95)
      fail("MATCH.ANY below one op per k cycles: code 0x%llx, %llu milli-lanes/clk, want %llu", f.code,
           (uint64_t)(lanes * 1000), (uint64_t)(cap * 1000));
    // A per-block unit's last pass can lose its block's result port to one ALU result
    // (VX_pe_switch round robin), never two in a row.
    if (any && f.k >= 4 && !VX_CFG_WARP_OP_SHARED && lanes * (f.k + 1) <= cap * f.k)
      fail("MATCH.ANY above k + 1 cycles per op: code 0x%llx, %llu milli-lanes/clk, width %llu", f.code,
           (uint64_t)(lanes * 1000), (uint64_t)(cap * 1000));
  }
}

} // namespace

int main(int argc, char** argv) {
  char mode = 'b';
  uint32_t seed = 1, ncases = 256, iters = 64;
  int64_t c0 = -1;
  int c;
  while ((c = getopt(argc, argv, "m:s:n:i:C:vh")) != -1) {
    switch (c) {
    case 'm': mode = optarg[0]; break;
    case 's': seed = (uint32_t)std::atoi(optarg); break;
    case 'n': ncases = (uint32_t)std::atoi(optarg); break;
    case 'i': iters = (uint32_t)std::atoi(optarg); break;
    case 'C': c0 = std::atoll(optarg); break;
    case 'v': verbose = true; break;
    default:
      std::printf("usage: %s [-m b|l|r] [-s seed] [-n cases] [-i iters] [-C c0]\n", argv[0]);
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
  if (num_threads > WOP_MAX_LANES) {
    std::printf("warp_ops: %u threads per warp, at most %u supported\nFAILED!\n", num_threads, WOP_MAX_LANES);
    return 1;
  }
  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "wop_main", &kernel));

  switch (mode) {
  case 'b': run_bitexact(seed, ncases); break;
  case 'l': run_latency(iters, c0); break;
  case 'r': run_rate(iters); break;
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
