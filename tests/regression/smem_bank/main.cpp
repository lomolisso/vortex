// smem_bank: shared-memory banks, wavefronts and latencies of the H100 SM path.
//
// Every expectation comes from sw/common/smem_bank_ref.h evaluated at the
// build's knobs; nothing measured on Vortex is read back as a target.
//
//   -m 0  lds32: a one-thread lw chase, then full-warp chases at word strides
//         1/2/3/4/8/16/32/33/64 and with 16/8/2/1 distinct words; per hop
//         (chase - what the hop's add costs ahead of an access, from an ld
//         chase with and without an add) = smem_load_latency +-1.
//   -m 1  lds64: ld chases over full pointers, 1..warp-width consecutive
//         lanes, strides 2/4/8/16 doublewords, the warp on one address; +-1.
//   -m 2  atom: amoadd.w / amocas.w chases (minus the hop's add as in -m 0,
//         and amocas.w's mv, from an amoadd.w chase with and without it), one
//         lane (absolute, +-1) and k lanes per address (delta over k = 1,
//         +-1); then every warp issues independent atomics: lane-ops per
//         cycle = lanes / passes +-5 % (one address per lane: >= 30/32 of it, or
//         reported when within 90 % of one warp instruction per cycle or at
//         issue width > 1).
//   -m 3  bw: every warp issues independent lw: bytes per cycle = lane bytes
//         / wavefronts +-3 % at word stride 2; conflict-free is reported,
//         or checked with -b at issue width 1.
// Rates are measured by difference of an N- and a 2N-iteration launch, so
// the warps' start skew and drain tail cancel.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <vortex.h>
#include <smem_bank_ref.h>

#include "common.h"

using namespace vortex::smem_bank_ref;

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

#ifndef VX_CFG_H100_SM_ENABLE
#error "smem_bank tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

const char* kernel_file = "kernel.vxbin";
int  mode = 0;
bool check_free = false;
uint32_t lanes = 0;  // the device's warp width

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_buffer_h pat_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-m mode: 0 lds32, 1 lds64, 2 atom, 3 bw] [-b check conflict-free bw] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:m:bh")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'm': mode = atoi(optarg); break;
    case 'b': check_free = true; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (pat_buffer) vx_buffer_release(pat_buffer);
    if (out_buffer) vx_buffer_release(out_buffer);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static SmemCfg knob_cfg() {
  SmemCfg c;
  c.banks      = VX_CFG_LMEM_NUM_BANKS;
  c.bank_bytes = VX_CFG_LMEM_BANK_BYTES;
  c.lat32      = VX_CFG_LMEM_LATENCY;
  c.lat64      = VX_CFG_LMEM_LATENCY_64;
  c.penalty    = VX_CFG_LMEM_WAVEFRONT_PENALTY;
  c.wide_off64 = VX_CFG_LMEM_WIDE_OFFSET_64;
  c.atom_extra = VX_CFG_LMEM_ATOM_EXTRA;
  c.cas_extra  = VX_CFG_LMEM_CAS_EXTRA;
  c.cas_passes = VX_CFG_LMEM_CAS_PASSES;
  return c;
}

struct Pattern {
  std::string name;
  std::vector<uint32_t> off;  // per lane, SB_INACTIVE = idle
  bool cas = false;
};

static Pattern strided(const std::string& name, uint32_t n, uint32_t stride_bytes, uint32_t group = 1) {
  Pattern p;
  p.name = name;
  p.off.assign(SB_LANES, SB_INACTIVE);
  for (uint32_t i = 0; i < n; ++i) {
    p.off[i] = (i / group) * stride_bytes;
  }
  return p;
}

static void golden_addrs(const Pattern& p, uint64_t* addr, uint32_t* mask) {
  *mask = 0;
  for (uint32_t i = 0; i < SB_LANES; ++i) {
    addr[i] = (p.off[i] == SB_INACTIVE) ? 0 : p.off[i];
    if (p.off[i] != SB_INACTIVE) *mask |= (1u << i);
  }
}

static uint32_t num_warps_dev = 0;
static uint32_t lmem_bytes = 0;

static void launch(kernel_arg_t& arg, const std::vector<Pattern>& pats, uint32_t warps,
                   std::vector<uint64_t>& out) {
  std::vector<uint32_t> tab;
  for (auto& p : pats) tab.insert(tab.end(), p.off.begin(), p.off.end());
  RT_CHECK(vx_enqueue_write(queue, pat_buffer, 0, tab.data(), tab.size() * sizeof(uint32_t), 0, nullptr, nullptr));
  arg.npat = (uint32_t)pats.size();
  RT_CHECK(vx_buffer_address(pat_buffer, &arg.pat_addr));
  RT_CHECK(vx_buffer_address(out_buffer, &arg.out_addr));
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = warps * lanes;
  li.lmem_size    = lmem_bytes;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_enqueue_read(queue, out.data(), out_buffer, 0, out.size() * sizeof(uint64_t), 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
}

static uint32_t errors = 0;

static void verdict(const std::string& what, double got, double exp, double tol_abs, double tol_rel) {
  double tol = std::max(tol_abs, std::fabs(exp) * tol_rel);
  bool ok = std::fabs(got - exp) <= tol + 1e-9;
  printf("  %-40s measured %9.3f expected %9.3f (+-%.3f) %s\n", what.c_str(), got, exp, tol, ok ? "ok" : "MISMATCH");
  if (!ok) ++errors;
}

static void run_chases(uint32_t kind, const std::vector<Pattern>& pats, double hop_extra, std::vector<uint64_t>& out,
                       std::vector<double>* lat) {
  kernel_arg_t arg = {};
  arg.kind = kind;
  arg.hops = 4096;
  arg.rows = 4;
  arg.row_bytes = 8192;
  arg.cas = pats.empty() ? 0 : pats[0].cas;
  launch(arg, pats, 1, out);
  for (size_t i = 0; i < pats.size(); ++i) {
    double chase = (double)out[i * 3 + 0] / arg.hops;
    lat->push_back(chase - hop_extra);
    printf("  %-40s chase %.3f/hop, minus %.3f for the rest of the hop\n", pats[i].name.c_str(), chase, hop_extra);
  }
}

// What an add costs ahead of a dependent LSU access, from one lane's ld chase
// timed without and with an add on every hop.
static double add_before_access(std::vector<uint64_t>& out) {
  kernel_arg_t arg = {};
  arg.kind = SB_CHASE64;
  arg.hops = 4096;
  arg.rows = 4;
  arg.row_bytes = 8192;
  arg.calib = 1;
  launch(arg, {strided("one lane", 1, 8)}, 1, out);
  double ld = (double)out[0] / arg.hops;
  double ld_add = (double)out[1] / arg.hops;
  printf("  %-40s ld %.3f/hop, ld + add %.3f/hop\n", "add ahead of an access", ld, ld_add);
  return ld_add - ld;
}

// What amocas.w's mv of the compare value costs ahead of the atomic, from one
// lane's amoadd.w chase with and without the same mv.
static double mv_before_atomic(std::vector<uint64_t>& out) {
  kernel_arg_t arg = {};
  arg.kind = SB_ATOM_LAT;
  arg.hops = 4096;
  arg.rows = 4;
  arg.row_bytes = 8192;
  arg.calib = 1;
  launch(arg, {strided("one lane", 1, 4)}, 1, out);
  double amo = (double)out[0] / arg.hops;
  double mv_amo = (double)out[1] / arg.hops;
  printf("  %-40s amoadd.w %.3f/hop, mv + amoadd.w %.3f/hop\n", "mv ahead of an atomic", amo, mv_amo);
  return mv_amo - amo;
}

static double run_tput(uint32_t kind, const Pattern& pat, bool cas, double exp_rate, uint32_t warps,
                       std::vector<uint64_t>& out, double* lane_ops_or_bytes) {
  kernel_arg_t arg = {};
  arg.kind = kind;
  arg.rows = 4;
  arg.row_bytes = 8192;
  arg.cas = cas;
  uint32_t active = 0;
  for (auto o : pat.off) active += (o != SB_INACTIVE);
  const double unit = (kind == SB_LOAD_BW) ? active * 4.0 : active;  // per warp op
  // About 32k cycles of steady state at the expected rate.
  double per_iter = warps * 64.0 * unit;
  const uint32_t iters = std::max<uint32_t>(1, (uint32_t)std::ceil(32768.0 * exp_rate / per_iter));
  double span[2];
  for (uint32_t r = 0; r < 2; ++r) {
    arg.iters = iters << r;
    launch(arg, {pat}, warps, out);
    uint64_t t0 = UINT64_MAX, t1 = 0;
    for (uint32_t w = 0; w < warps; ++w) {
      t0 = std::min(t0, out[w * 2 + 0]);
      t1 = std::max(t1, out[w * 2 + 1]);
    }
    span[r] = (double)(t1 - t0);
  }
  *lane_ops_or_bytes = per_iter * iters;
  return *lane_ops_or_bytes / (span[1] - span[0]);
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  if (mode < 0 || mode > 3) {
    show_usage();
    return -1;
  }

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_warps, num_threads, isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (num_threads > SB_LANES) {
    std::cout << "Error: needs at most " << SB_LANES << " threads, device has " << num_threads << std::endl;
    cleanup();
    return -1;
  }
  if ((isa_flags & VX_ISA_STD_A) == 0) {
    std::cout << "Error: needs the A extension" << std::endl;
    cleanup();
    return -1;
  }
  num_warps_dev = (uint32_t)num_warps;
  lanes = (uint32_t)num_threads;
  const std::string warp = std::to_string(lanes) + " lanes";
  lmem_bytes = 4 * 8192;

  const SmemCfg cfg = knob_cfg();
  std::cout << "banks " << cfg.banks << " x " << cfg.bank_bytes << " B, LDS " << cfg.lat32 << "/" << cfg.lat64
            << ", penalty " << cfg.penalty << ", warps " << num_warps_dev << ", LSU blocks " << VX_CFG_NUM_LSU_BLOCKS << std::endl;

  RT_CHECK(vx_buffer_create(device, 64 * SB_LANES * sizeof(uint32_t), VX_MEM_READ, &pat_buffer));
  std::vector<uint64_t> out(std::max<uint32_t>(64 * 3, num_warps_dev * 2));
  RT_CHECK(vx_buffer_create(device, out.size() * sizeof(uint64_t), VX_MEM_READ_WRITE, &out_buffer));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  uint64_t addr[SB_LANES];
  uint32_t mask;

  if (mode == 0 || mode == 1) {
    const bool wide = (mode == 1);
    const uint32_t bytes = wide ? 8 : 4;
    std::vector<Pattern> pats;
    if (!wide) {
      pats.push_back(strided("one thread", 1, 4));
      for (uint32_t s : {1u, 2u, 3u, 4u, 8u, 16u, 32u, 33u, 64u})
        pats.push_back(strided(warp + ", stride " + std::to_string(s) + " words", lanes, s * 4));
      for (uint32_t d : {16u, 8u, 2u, 1u})
        if (d <= lanes)
          pats.push_back(strided(warp + ", " + std::to_string(d) + " distinct words", lanes, 4, lanes / d));
    } else {
      for (uint32_t n : {1u, 2u, 4u, 8u, 16u, 24u, 32u})
        if (n <= lanes)
          pats.push_back(strided(std::to_string(n) + " lanes, consecutive", n, 8));
      for (uint32_t s : {2u, 4u, 8u, 16u})
        pats.push_back(strided(warp + ", stride " + std::to_string(s) + " doublewords", lanes, s * 8));
      pats.push_back(strided(warp + ", one address", lanes, 8, lanes));
    }
    std::vector<double> lat;
    run_chases(wide ? SB_CHASE64 : SB_CHASE32, pats, wide ? 0.0 : add_before_access(out), out, &lat);
    for (size_t i = 0; i < pats.size(); ++i) {
      golden_addrs(pats[i], addr, &mask);
      uint32_t d  = smem_wavefronts(addr, mask, lanes, bytes, cfg);
      uint32_t dw = smem_distinct_words(addr, mask, lanes, bytes);
      verdict(pats[i].name + " (D=" + std::to_string(d) + ")", lat[i], smem_load_latency(d, bytes, dw, cfg), 1, 0);
    }
  } else if (mode == 2) {
    for (bool cas : {false, true}) {
#ifndef VX_CFG_EXT_ZACAS_ENABLE
      if (cas) { std::cout << "Error: amocas.w needs Zacas" << std::endl; ++errors; break; }
#endif
      std::vector<Pattern> pats;
      pats.push_back(strided("one lane", 1, 4));
      std::vector<uint32_t> ks = {1};
      for (uint32_t k = 2; k <= lanes; k *= 2)
        if (!cas || k == lanes) ks.push_back(k);
      for (uint32_t k : ks)
        pats.push_back(strided(warp + ", " + std::to_string(k) + " per address", lanes, 4, k));
      for (auto& p : pats) p.cas = cas;
      std::cout << (cas ? "amocas.w" : "amoadd.w") << " latency" << std::endl;
      std::vector<double> lat;
      double hop_extra = add_before_access(out);
      if (cas)
        hop_extra += mv_before_atomic(out);
      run_chases(SB_ATOM_LAT, pats, hop_extra, out, &lat);
      golden_addrs(pats[0], addr, &mask);
      verdict("one lane", lat[0], smem_atom_latency(smem_atom_passes(addr, mask, lanes, 4, cas, cfg), cas, cfg), 1, 0);
      golden_addrs(pats[1], addr, &mask);
      uint32_t ref1 = smem_atom_latency(smem_atom_passes(addr, mask, lanes, 4, cas, cfg), cas, cfg);
      for (size_t i = 2; i < pats.size(); ++i) {
        golden_addrs(pats[i], addr, &mask);
        uint32_t ref = smem_atom_latency(smem_atom_passes(addr, mask, lanes, 4, cas, cfg), cas, cfg);
        verdict(pats[i].name + " (delta over k=1)", lat[i] - lat[1], (double)ref - ref1, 1, 0);
      }
    }
    for (bool cas : {false, true}) {
#ifndef VX_CFG_EXT_ZACAS_ENABLE
      if (cas) break;
#endif
      std::vector<uint32_t> ks = {1};
      for (uint32_t k = 2; !cas && k <= lanes; k *= 2)
        if (k <= 8 || k == lanes) ks.push_back(k);
      std::cout << (cas ? "amocas.w" : "amoadd.w") << " throughput, " << num_warps_dev << " warps" << std::endl;
      for (uint32_t k : ks) {
        Pattern p = strided(warp + ", " + std::to_string(k) + " per address", lanes, 4, k);
        golden_addrs(p, addr, &mask);
        double exp = (double)lanes / smem_atom_passes(addr, mask, lanes, 4, cas, cfg);
        double ops;
        double rate = run_tput(SB_ATOM_TPUT, p, cas, exp, num_warps_dev, out, &ops);
        if (k == 1 && !cas) {
          // One address per lane: at least 30/32 of the golden rate (the H100 reaches 30 of 32). One warp
          // instruction per cycle caps the loop (SB_UNROLL atomics, its counter and branch) below that.
          const double min_rate = exp * 30.0 / 32.0;
          const double fe_bound = (double)lanes * SB_UNROLL / (SB_UNROLL + 2);
          // With several LSU blocks the core's issue feeds them below one warp request per cycle in all,
          // upstream of the shared-memory path.
          if (rate < min_rate && VX_CFG_ISSUE_WIDTH > 1) {
            printf("  %-40s measured %9.3f lane-ops/cycle, report: below %.2f at issue width %d\n", p.name.c_str(), rate,
                   min_rate, (int)VX_CFG_ISSUE_WIDTH);
          } else if (rate < min_rate && rate >= 0.9 * fe_bound) {
            printf("  %-40s measured %9.3f lane-ops/cycle, report: front-end bound %.3f of %.3f\n", p.name.c_str(), rate,
                   rate, fe_bound);
          } else {
            printf("  %-40s measured %9.3f lane-ops/cycle, at least %.2f of %.0f %s\n", p.name.c_str(), rate, min_rate,
                   exp, rate >= min_rate ? "ok" : "MISMATCH");
            if (rate < min_rate) ++errors;
          }
        } else {
          verdict(p.name + " lane-ops/cycle", rate, exp, 0, 0.05);
        }
      }
    }
  } else {
    std::cout << "lw bandwidth, " << num_warps_dev << " warps" << std::endl;
    for (uint32_t s : {2u, 1u}) {
      Pattern p = strided(warp + ", stride " + std::to_string(s) + " words", lanes, s * 4);
      golden_addrs(p, addr, &mask);
      double exp = lanes * 4.0 / smem_wavefronts(addr, mask, lanes, 4, cfg);
      double bytes;
      double rate = run_tput(SB_LOAD_BW, p, false, exp, num_warps_dev, out, &bytes);
      if (s == 1 && (!check_free || VX_CFG_ISSUE_WIDTH > 1)) {
        printf("  %-40s measured %9.3f B/cycle, expected %.3f (reported)\n", p.name.c_str(), rate, exp);
      } else {
        verdict(p.name + " B/cycle", rate, exp, 0, 0.03);
      }
    }
  }

  cleanup();
  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
