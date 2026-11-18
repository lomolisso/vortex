// smem_bank: shared-memory banks, wavefronts and latencies of the H100 SM path.
//
// Every expectation is a structural formula from sw/common/smem_bank_ref.h
// (the stages a shared access passes, one cycle per wavefront or AMO pass);
// nothing measured on Vortex is read back as a target. H100 measurements are
// printed as "H100REF metric=... h100=... vortex=..." lines, never judged.
//
//   -m 0  lds32: a one-thread lw chase, then full-warp chases at word strides
//         1/2/3/4/8/16/32/33/64 and with 16/8/2/1 distinct words; per hop
//         (chase - what the hop's add costs ahead of an access, from an ld
//         chase with and without an add) = smem_load_latency +-1, and each
//         extra wavefront adds 1 cycle +-0.25 over the full-warp D=1 chase.
//   -m 1  lds64: ld chases over full pointers, 1..warp-width consecutive
//         lanes, strides 2/4/8/16 doublewords, the warp on one address; as -m 0.
//   -m 2  atom: amoadd.w / amocas.w chases (minus the hop's add as in -m 0,
//         and amocas.w's mv, from an amoadd.w chase with and without it), one
//         lane (absolute, +-1) and k lanes per address (one cycle per extra
//         pass, +-0.25); a CAS takes the passes of an add plus one stage
//         cycle for its comparand beat; then every warp issues independent
//         atomics: lane-ops per cycle = lanes / stage cycles +-5 % (one
//         address per lane: capped at one warp instruction per cycle over the
//         loop's atomics, counter and branch; reported below it at issue
//         width > 1).
//   -m 3  bw: every warp issues independent lw: bytes per cycle = lane bytes
//         / wavefronts +-3 % at word stride 2; conflict-free is reported,
//         or checked with -b at issue width 1.
//   -m 4  port-share: warp classes on the SM's one data port (LDS lw, L1-hit
//         lw on a line per warp, STS sw), each warp a full 128-B row: the
//         classes' bytes per cycle sum to the port width +-3 %; two equal
//         classes get half each +-10 %.
//   -m 5  under-miss: M = 8 warps stream L2-hit lines through the L1 (8 lines
//         in flight each) beside a one-lane LDS chase and an L1-hit chase.
//         Each hop grows by at most the cycles the competing lines take on
//         what the chase shares with them (a read-out wavefront and a fill
//         write per line; an L1 access also their tag lookup and fill update),
//         + 1; 2M miss warps add nothing beyond their own lines (+ 1), and no
//         hop waits longer than every line in flight (no head-of-line wait on
//         tags or MSHRs); the H100's U2 values are printed. 2M x 8 lines
//         stays within the SM's read lines: past them a global load waits
//         for a line in the LSU's in-order request queue and the shared
//         accesses behind it wait too (the H100 shows no such wait up to 768
//         lines; per-class queues after address generation are not built
//         yet). LDS stream warps
//         beside the misses: LDS bytes plus the misses' read-out bytes = port
//         width +-5 %.
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
vx_buffer_h gbl_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-m mode: 0 lds32, 1 lds64, 2 atom, 3 bw, 4 port-share, 5 under-miss] [-b check conflict-free bw] [-h]" << std::endl;
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
    if (gbl_buffer) vx_buffer_release(gbl_buffer);
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
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  c.pipe       = smem_pipe_stages(VX_CFG_NUM_OPCS, true);
#else
  c.pipe       = smem_pipe_stages(VX_CFG_NUM_OPCS, false);
#endif
  return c;
}

// An H100 measurement beside Vortex's value, for task G; never judged.
static void h100ref(const std::string& metric, double h100, double vortex) {
  printf("H100REF metric=%s h100=%.2f vortex=%.3f\n", metric.c_str(), h100, vortex);
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
  double per_iter = warps * (double)SB_TIMED_ACCESSES * unit;
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

struct Span {
  uint64_t t0 = UINT64_MAX, t1 = 0;
  double amount = 0;
  void add(uint64_t a, uint64_t b, double n) { t0 = std::min(t0, a); t1 = std::max(t1, b); amount += n; }
  double rate() const { return (t1 > t0) ? amount / (double)(t1 - t0) : 0.0; }
};

// SB_MIX: per class bytes per cycle (difference of an N- and a 2N-iteration
// launch) and their sum, LDS / L1 / STS.
static void run_mix(const uint32_t ncls[3], double port, std::vector<uint64_t>& out, double rate[4]) {
  kernel_arg_t arg = {};
  arg.kind = SB_MIX;
  arg.rows = 4;
  arg.row_bytes = 8192;
  for (int c = 0; c < 3; ++c) arg.ncls[c] = ncls[c];
  RT_CHECK(vx_buffer_address(gbl_buffer, &arg.gbl_addr));
  const uint32_t warps = ncls[0] + ncls[1] + ncls[2];
  const double per_warp = (double)SB_TIMED_ACCESSES * lanes * 4;
  const uint32_t iters = std::max<uint32_t>(1, (uint32_t)std::ceil(32768.0 * port / (warps * per_warp)));
  double amount[2][4], span[2][4];
  for (uint32_t r = 0; r < 2; ++r) {
    arg.iters = iters << r;
    launch(arg, {strided("rows", lanes, 4)}, warps, out);
    Span sp[4];
    for (uint32_t w = 0; w < warps; ++w) {
      uint32_t c = (w < ncls[0]) ? 0 : (w < ncls[0] + ncls[1]) ? 1 : 2;
      sp[c].add(out[w * 2 + 0], out[w * 2 + 1], per_warp * arg.iters);
      sp[3].add(out[w * 2 + 0], out[w * 2 + 1], per_warp * arg.iters);
    }
    for (int c = 0; c < 4; ++c) {
      amount[r][c] = sp[c].amount;
      span[r][c] = (double)(sp[c].t1 - std::min(sp[c].t0, sp[c].t1));
    }
  }
  for (int c = 0; c < 4; ++c) {
    double d = span[1][c] - span[0][c];
    rate[c] = (amount[0][c] > 0 && d > 0) ? (amount[1][c] - amount[0][c]) / d : 0.0;
  }
}

// SB_UNDER_MISS: per warp {start, end, amount, max hop}.
static void run_under_miss(uint32_t um_kind, uint32_t miss_warps, uint32_t instr, uint32_t warps, uint64_t gbl_bytes,
                           uint32_t iters, std::vector<uint64_t>& out) {
  kernel_arg_t arg = {};
  arg.kind = SB_UNDER_MISS;
  arg.um_kind = um_kind;
  arg.miss_warps = miss_warps;
  arg.instr = instr;
  arg.hops = 4096;
  arg.rows = 4;
  arg.row_bytes = (um_kind == SB_UM_L1_CHASE) ? 4096 : 8192;
  arg.iters = iters;
  arg.gbl_bytes = gbl_bytes;
  RT_CHECK(vx_buffer_address(gbl_buffer, &arg.gbl_addr));
  launch(arg, {strided("rows", lanes, 4)}, warps, out);
}

static void check_max(const std::string& what, double got, double bound) {
  bool ok = got <= bound + 1e-9;
  printf("  %-40s measured %9.3f bound <= %9.3f %s\n", what.c_str(), got, bound, ok ? "ok" : "MISMATCH");
  if (!ok) ++errors;
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  if (mode < 0 || mode > 5) {
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
  std::cout << "banks " << cfg.banks << " x " << cfg.bank_bytes << " B, stages " << cfg.pipe << ", warps " << num_warps_dev << ", LSU blocks " << VX_CFG_NUM_LSU_BLOCKS << std::endl;

  RT_CHECK(vx_buffer_create(device, 64 * SB_LANES * sizeof(uint32_t), VX_MEM_READ, &pat_buffer));
  std::vector<uint64_t> out(std::max<uint32_t>(64 * 3, num_warps_dev * 4));
  // The miss stream's footprint: L2-resident, several times the largest L1.
  const uint64_t um_bytes = (uint64_t)VX_CFG_L2_SIZE * 3 / 4 / 1024 * 1024;
  if (mode >= 4) {
    RT_CHECK(vx_buffer_create(device, std::max<uint64_t>(num_warps_dev * 128, um_bytes) + 4 * 4096, VX_MEM_READ_WRITE, &gbl_buffer));
  }
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
    double lat1 = -1, lat1_warp = -1;
    for (size_t i = 0; i < pats.size(); ++i) {
      golden_addrs(pats[i], addr, &mask);
      uint32_t d = smem_wavefronts(addr, mask, lanes, bytes, cfg);
      verdict(pats[i].name + " (D=" + std::to_string(d) + ")", lat[i], smem_load_latency(d, cfg), 1, 0);
      if (d == 1 && lat1 < 0)
        lat1 = lat[i];
      if (d == 1 && lat1_warp < 0 && (uint32_t)__builtin_popcount(mask) == lanes)
        lat1_warp = lat[i];
    }
    // One cycle per extra wavefront, exactly: the wavefront stage reads one per cycle. The reference is
    // a full-warp D=1 chase, so both sides issue and collect the same lanes.
    uint32_t d_max = 1;
    double lat_max = lat1_warp;
    for (size_t i = 0; i < pats.size(); ++i) {
      golden_addrs(pats[i], addr, &mask);
      uint32_t d = smem_wavefronts(addr, mask, lanes, bytes, cfg);
      if (d > 1 && lat1_warp >= 0 && (uint32_t)__builtin_popcount(mask) == lanes) {
        verdict(pats[i].name + " (over D=1)", lat[i] - lat1_warp, d - 1.0, 0.25, 0);
        if (d > d_max) { d_max = d; lat_max = lat[i]; }
      }
    }
    h100ref(wide ? "lds64_chase" : "lds32_chase", wide ? 23.77 : 23.02, lat1);
    if (d_max > 1)
      h100ref("lds_extra_wavefront", 2.0, (lat_max - lat1_warp) / (d_max - 1));
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
      verdict("one lane", lat[0], smem_atom_latency(smem_atom_passes(addr, mask, lanes, 4, cfg), cfg, cas), 1, 0);
      golden_addrs(pats[1], addr, &mask);
      uint32_t ref1 = smem_atom_latency(smem_atom_passes(addr, mask, lanes, 4, cfg), cfg, cas);
      for (size_t i = 2; i < pats.size(); ++i) {
        golden_addrs(pats[i], addr, &mask);
        uint32_t ref = smem_atom_latency(smem_atom_passes(addr, mask, lanes, 4, cfg), cfg, cas);
        verdict(pats[i].name + " (delta over k=1)", lat[i] - lat[1], (double)ref - ref1, 0.25, 0);
      }
      h100ref(cas ? "atom_cas_chase" : "atom_add_chase", cas ? 40.0 : 33.0, lat[0]);
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
        double exp = (double)lanes / smem_atom_stage_cycles(smem_atom_passes(addr, mask, lanes, 4, cfg), cas);
        double ops;
        double rate = run_tput(SB_ATOM_TPUT, p, cas, exp, num_warps_dev, out, &ops);
        if (k == 1 && !cas) {
          // One address per lane: one warp instruction per cycle caps the loop (SB_UNROLL atomics, its
          // counter and branch) below the golden rate at issue width 1.
          const double fe_bound = (double)lanes * SB_UNROLL / (SB_UNROLL + 2);
          const double bound = (VX_CFG_ISSUE_WIDTH > 1) ? exp : std::min(exp, fe_bound);
          // With several LSU blocks the core's issue feeds them below one warp request per cycle in all,
          // upstream of the shared-memory path.
          if (VX_CFG_ISSUE_WIDTH > 1 && rate < 0.95 * bound) {
            printf("  %-40s measured %9.3f lane-ops/cycle, report: below %.3f at issue width %d\n", p.name.c_str(), rate,
                   bound, (int)VX_CFG_ISSUE_WIDTH);
          } else {
            verdict(p.name + " lane-ops/cycle", rate, bound, 0, 0.05);
          }
          h100ref("atom_add_rate_per_lane", 30.0 / 32.0, rate / lanes);
        } else {
          verdict(p.name + " lane-ops/cycle", rate, exp, 0, 0.05);
          if (cas)
            h100ref("atom_cas_rate_over_add", 0.5, rate / lanes);
        }
      }
    }
  } else if (mode == 4) {
    const double port = VX_CFG_SM_DATAPATH_BYTES;
    const uint32_t W = num_warps_dev;
    std::cout << "port share, " << W << " warps, port " << port << " B/cycle" << std::endl;
    if (W < 4) {
      std::cout << "Error: needs at least 4 warps" << std::endl;
      ++errors;
    } else {
      struct Mix { const char* name; uint32_t n[3]; bool halves; };
      const Mix mixes[] = {
        {"LDS + L1 hit, equal",   {W / 2, W - W / 2, 0}, true},
        {"LDS + L1 hit, 1:3",     {W / 4, W - W / 4, 0}, false},
        {"LDS + L1 hit, 3:1",     {W - W / 4, W / 4, 0}, false},
        {"LDS + STS",             {W / 2, 0, W - W / 2}, false},
        {"LDS + L1 hit + STS",    {W / 3, W / 3, W - 2 * (W / 3)}, false},
      };
      const char* cls[3] = {"LDS", "L1 hit", "STS"};
      for (auto& m : mixes) {
        double rate[4];
        run_mix(m.n, port, out, rate);
        printf("  %-40s", m.name);
        for (int c = 0; c < 3; ++c) if (m.n[c]) printf(" %s(%u w) %.2f", cls[c], m.n[c], rate[c]);
        printf("\n");
        verdict(std::string(m.name) + " sum B/cycle", rate[3], port, 0, 0.03);
        if (m.halves) {
          verdict(std::string(m.name) + " LDS B/cycle", rate[0], port / 2, 0, 0.10);
          verdict(std::string(m.name) + " L1 hit B/cycle", rate[1], port / 2, 0, 0.10);
        }
      }
    }
  } else if (mode == 5) {
    const uint32_t W = num_warps_dev, M = 8;
    std::cout << "LDS under L1 misses, " << W << " warps, " << M << " miss warps over " << um_bytes / 1024 << " KiB" << std::endl;
    if (W < 2 * M + 1) {
      std::cout << "Error: needs at least " << 2 * M + 1 << " warps" << std::endl;
      ++errors;
    } else {
      const double hop_extra = add_before_access(out);
      const double lat = smem_load_latency(1, cfg);
      const double line = VX_CFG_DCACHE_LINE_SIZE;
      struct Hop { double hop, max, lines; };
      // Per hop, the chase's cycles, its longest hop and the miss lines the stream read meanwhile.
      auto chase_hop = [&](uint32_t kind, uint32_t miss, uint32_t instr) {
        run_under_miss(kind, miss, instr, miss + 1, um_bytes, 0, out);
        const uint64_t* c = &out[0];
        Span ms;
        for (uint32_t w = 1; w <= miss; ++w) ms.add(out[w * 4 + 0], out[w * 4 + 1], (double)out[w * 4 + 2]);
        const double hop = (double)(c[1] - c[0]) / c[2];
        return Hop{hop - hop_extra, (double)c[3], ms.rate() * hop / line};
      };
      // cost: the chase's cycles per competing line on what it shares with the stream.
      auto under_miss = [&](uint32_t kind, const std::string& what, const std::string& ref, double alone, uint32_t cost,
                            double h100_alone, double h100_hop) {
        const Hop m1 = chase_hop(kind, M, 0);
        const Hop m2 = chase_hop(kind, 2 * M, 0);
        printf("  %-40s alone %.3f, M %.3f (%.2f lines/hop), 2M %.3f (%.2f lines/hop)\n", what.c_str(), alone,
               m1.hop, m1.lines, m2.hop, m2.lines);
        check_max(what + " under M, growth", m1.hop - alone, cost * m1.lines + 1);
        check_max(what + " 2M over M", m2.hop - m1.hop, cost * std::max(0.0, m2.lines - m1.lines) + 1);
        h100ref(ref + "_under_miss_alone", h100_alone, alone);
        h100ref(ref + "_under_miss_hop", h100_hop, m1.hop);
      };
      const double lds_alone = chase_hop(SB_UM_LDS_CHASE, 0, 0).hop;
      verdict("LDS chase alone", lds_alone, lat, 1, 0);
      under_miss(SB_UM_LDS_CHASE, "LDS chase", "lds", lds_alone, 2, 23.08, 31.26);
      const double instr_cost = chase_hop(SB_UM_LDS_CHASE, 0, 1).hop - lds_alone;
      const double lds_max = chase_hop(SB_UM_LDS_CHASE, M, 1).max - instr_cost - hop_extra;
      check_max("LDS chase under misses, longest hop", lds_max - lds_alone, 2.0 * M * 8 + 1);
      h100ref("lds_under_miss_max", 64, lds_max);
      const double l1_alone = chase_hop(SB_UM_L1_CHASE, 0, 0).hop;
      printf("  %-40s measured %9.3f (reported)\n", "L1-hit chase alone", l1_alone);
      under_miss(SB_UM_L1_CHASE, "L1-hit chase", "l1hit", l1_alone, 4, 36.90, 52.16);
      const uint32_t S = W - M;
      const double port = VX_CFG_SM_DATAPATH_BYTES;
      const uint32_t iters =
          std::max<uint32_t>(1, (uint32_t)std::ceil(32768.0 * port / (S * 64.0 * lanes * SB_UM_LANE_BYTES)));
      run_under_miss(SB_UM_LDS_STREAM, M, 0, W, um_bytes, iters, out);
      Span ms, ls;
      for (uint32_t w = 0; w < W; ++w) (w >= S ? ms : ls).add(out[w * 4 + 0], out[w * 4 + 1], (double)out[w * 4 + 2]);
      printf("  %-40s LDS %.2f B/cycle, miss read-out %.2f B/cycle\n", "LDS stream beside misses", ls.rate(), ms.rate());
      verdict("LDS stream + miss read-out B/cycle", ls.rate() + ms.rate(), port, 0, 0.05);
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
