// sm_port: the H100 SM path's egress port, SM-to-SM messages and ingress.
//
// Every expected rate is the steady state of sw/common/sm_port_ref.h's
// SmEgressRef fed the test's packet stream at the build's knobs: one 32-B flit
// per cycle per FLIT bytes of link, each packet its flits by the port's format
// (a write line = touched sectors + 1 mask flit if any is partial; every SM-to-SM
// message = 1 flit). H100 rates of the same streams are printed as H100REF lines
// (SMPORT_flit_h100.md), never judged.
//
//   -m 0 st      full-line global stores (4 full sectors per line)            +-3 %
//   -m 1 st1s    one full sector per line (1 flit per packet)                 +-3 %
//   -m 2 stmask  one 4-B lane per sector: 4 masked sectors per line, then 1   +-5 %
//   -m 3 rw      every warp's stores, then warp 0's load: usable no earlier than their drain (in-order path)
//   (-m 0..3: a rate the levels below the port cannot take, measured with -m 1's stream, is reported, not counted)
//   -m 4 inject  rank 0 sd's into rank 1                 within the link; rate reported
//   -m 5 eject   ranks 1..3 sd into rank 0              within the receiver's ejection link; rate reported
//   -m 6 pkt1s   one sector per destination line         within the link; rate reported
//   -m 7 ld      rank 0 ld's rank 1's lines              rank 1's replies within its link; rate reported
//   -m 8 atom    rank 0 amoadd.w's rank 1's lines        within the link; rate reported
//   -m 9 mix     rank 0 sd's into rank 1 and to global memory: the sum of their flits within the link
//   -m 10 tma    each rank's warp 0 bulk-loads L2-resident lines (TMA), each rank alone, then both:
//                each SM keeps > 0.75 of its own alone rate (one engine and port per SM; shared would give <= 0.5)
//   -m 11 fillwr rank 0's warps 1.. stream LDS / L1-hit lw / STS beside fills of the SM's array (warp 0's TMA
//                bulk loads, or rank 1's sd arrivals); fills share the write port only, ahead of STS:
//                LDS and L1 hits next to TMA >= 15/16 of the datapath, STS next to TMA = datapath - TMA rate +-5 %,
//                STS next to DSMEM = datapath x (1 - arrival rate / SM_SRAM_DSMEM_WR_BYTES) +-10 %,
//                LDS next to DSMEM = datapath +-3 %

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

#include <vortex.h>
#include <sm_port_ref.h>

#include "common.h"

using namespace vortex::sm_port_ref;

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
#error "sm_port tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t LINE = VX_CFG_DCACHE_LINE_SIZE;
constexpr uint32_t SPAN = 65536;

const char* kernel_file = "kernel.vxbin";
std::vector<int> modes;

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_buffer_h pat_buffer = nullptr;
vx_buffer_h gmem_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
uint64_t num_warps = 0, num_cores = 0, max_cluster = 0;
uint32_t errors = 0;
uint32_t warp_lanes = 0;  // the device's warp width

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-m mode[,mode...] 0..11] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:m:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'm': {
      std::string a(optarg);
      size_t pos = 0;
      while (pos <= a.size()) {
        size_t e = a.find(',', pos);
        if (e == std::string::npos) e = a.size();
        modes.push_back(atoi(a.substr(pos, e - pos).c_str()));
        pos = e + 1;
      }
      break;
    }
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (pat_buffer) vx_buffer_release(pat_buffer);
    if (gmem_buffer) vx_buffer_release(gmem_buffer);
    if (out_buffer) vx_buffer_release(out_buffer);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static void verdict(const std::string& what, double got, double exp, double tol_rel) {
  double tol = std::fabs(exp) * tol_rel;
  bool ok = std::fabs(got - exp) <= tol + 1e-9;
  printf("  %-48s measured %9.3f expected %9.3f (+-%.3f) %s\n", what.c_str(), got, exp, tol, ok ? "ok" : "MISMATCH");
  if (!ok) ++errors;
}

// An SM-to-SM stream's rate depends on more than this port (the sender's lane
// serialiser, the network, the receiver's array), so it is printed beside the
// golden and the H100's rate, and only its bound by the link is judged.
static void report(const std::string& what, double got, double golden, const char* metric, double h100) {
  printf("  %-48s measured %9.3f golden %9.3f (report)\n", what.c_str(), got, golden);
  printf("H100REF metric=%s h100=%.3f vortex=%.3f\n", metric, h100, got);
}

static void within(const std::string& what, double got, double bound) {
  // The kernel's end time can precede the drain of the stores still queued
  // below the warp, hence the 5 % margin.
  bool ok = got <= bound * 1.05;
  printf("  %-48s %.3f <= %.3f %s\n", what.c_str(), got, bound, ok ? "ok" : "MISMATCH");
  if (!ok) ++errors;
}

static SmPortCfg knob_cfg() {
  SmPortCfg c{};
  c.egress_bytes  = VX_CFG_SM_EGRESS_BYTES;
  c.flit_bytes    = VX_CFG_SM_EGRESS_BEAT;
  c.eject_bytes   = VX_CFG_SM2SM_EJECT_BYTES;
  c.ingress_bytes = VX_CFG_SM_INGRESS_BYTES;
  c.w_fill        = VX_CFG_SM_INGRESS_FILL_WEIGHT;
  c.w_dsmem       = VX_CFG_SM_INGRESS_DSMEM_WEIGHT;
#if VX_CFG_EXT_DLMEM_ENABLED
  c.rsp_slots     = VX_CFG_DLMEM_INGRESS_SLOTS;
#else
  c.rsp_slots     = 1;
#endif
  return c;
}

enum class Kind { GStore, DStore, DLoadRsp, DAtom };

// One egress packet of the stream and the payload it carries.
struct Item {
  SmEgressItem it;
  bool rsp;
  double payload;
  int cls;
  uint32_t blocks;  // memory blocks it sends to the L2
};

static std::vector<uint8_t> line_bytes(const std::vector<uint32_t>& offs, uint32_t size) {
  std::vector<uint8_t> be(LINE, 0);
  for (auto o : offs)
    for (uint32_t i = 0; i < size; ++i) be[(o % LINE) + i] = 1;
  return be;
}

// A global write line: one packet. An SM-to-SM item: one message per lane.
static Item make_item(Kind kind, const std::vector<uint8_t>& be, double payload, int cls, const SmPortCfg& cfg) {
  Item x;
  x.it.l2 = (kind == Kind::GStore);
  x.it.flits = x.it.l2 ? sm_packet_flits(SmItem::WriteLine, be.data(), LINE, cfg) : 1;
  x.rsp = (kind == Kind::DLoadRsp);
  x.payload = payload;
  x.cls = cls;
  x.blocks = 0;
  if (x.it.l2) {
    for (uint32_t b = 0; b < LINE; b += VX_CFG_MEM_BLOCK_SIZE) {
      uint32_t n = 0;
      for (uint32_t i = 0; i < VX_CFG_MEM_BLOCK_SIZE; ++i) n += be[b + i];
      x.blocks += (n != 0);
    }
  }
  return x;
}

// One warp instruction's packets: a global write per line its lanes touch, in
// line order, carrying its bytes; an SM-to-SM message per active lane,
// carrying its bytes (an atomic: one lane-op).
static std::vector<Item> items_of(Kind kind, const std::vector<uint32_t>& pat, uint32_t size, int cls,
                                  const SmPortCfg& cfg) {
  std::vector<Item> v;
  if (kind != Kind::GStore) {
    for (auto o : pat) {
      if (o != 0xFFFFFFFFu)
        v.push_back(make_item(kind, line_bytes({o}, size), kind == Kind::DAtom ? 1.0 : size, cls, cfg));
    }
    return v;
  }
  std::map<uint32_t, std::vector<uint32_t>> lines;
  for (auto o : pat)
    if (o != 0xFFFFFFFFu) lines[o / LINE].push_back(o % LINE);
  for (auto& l : lines) {
    auto be = line_bytes(l.second, size);
    uint32_t bytes = 0;
    for (auto b : be) bytes += b;
    v.push_back(make_item(kind, be, bytes, cls, cfg));
  }
  return v;
}

static double flits_of(const std::vector<Item>& items) {
  double f = 0;
  for (auto& x : items) f += x.it.flits;
  return f;
}

// Payload of one instruction: the sum of its items'.
static double payload_of(const std::vector<Item>& items) {
  double p = 0;
  for (auto& x : items) p += x.payload;
  return p;
}

// Bytes the pattern spans, whole lines: the kernel's per-instruction step.
static uint32_t span_of(const std::vector<uint32_t>& pat, uint32_t size) {
  uint32_t end = 0;
  for (auto o : pat)
    if (o != 0xFFFFFFFFu) end = std::max(end, o + size);
  return (end + LINE - 1) / LINE * LINE;
}

// The kernel's store loops step by 64, 128, 256, 512 or 1024 B.
static uint32_t store_step(const std::vector<uint32_t>& pat, uint32_t size) {
  const uint32_t s = span_of(pat, size);
  if (s != 64 && s != 128 && s != 256 && s != 512 && s != 1024) {
    std::cout << "Error: no kernel store loop steps by " << s << " B" << std::endl;
    cleanup();
    exit(-1);
  }
  return s;
}

// Steady-state payload per cycle of each class when the stream repeats forever,
// the global side taking at most `accept` memory blocks per cycle (0: no cap).
static std::vector<double> golden_rates(const std::vector<Item>& stream, const SmPortCfg& cfg, double accept = 0) {
  SmEgressRef eg(cfg);
  std::vector<double> got(2, 0.0);
  double credit = 1.0;
  const uint32_t warm = 4000, cycles = 40000;
  size_t next = 0;
  for (uint32_t t = 0; t < warm + cycles; ++t) {
    const Item& x = stream[next];
    if (x.rsp ? eg.rsp_can_push() : eg.req_can_push()) {
      SmEgressItem it = x.it;
      it.id = next;
      if (x.rsp) eg.rsp_push(it); else eg.req_push(it);
      next = (next + 1) % stream.size();
    }
    credit = std::min(1.0, credit + accept);
    const bool l2_ready = accept <= 0 || credit >= 1.0 - 1e-9;
    auto p = eg.tick(l2_ready, true, true);
    if (accept > 0 && p.req && stream[p.req_item.id].it.l2) credit -= stream[p.req_item.id].blocks;
    if (t >= warm) {
      if (p.req) got[stream[p.req_item.id].cls] += stream[p.req_item.id].payload;
      if (p.rsp) got[stream[p.rsp_item.id].cls] += stream[p.rsp_item.id].payload;
    }
  }
  for (auto& g : got) g /= cycles;
  return got;
}

struct Run {
  uint32_t blocks = 1, cluster = 1, warps = 1;
  kernel_arg_t arg = {};
};

static std::vector<uint64_t> launch(Run& r, const std::vector<uint32_t>& pat) {
  RT_CHECK(vx_enqueue_write(queue, pat_buffer, 0, pat.data(), pat.size() * 4, 0, nullptr, nullptr));
  RT_CHECK(vx_buffer_address(pat_buffer, &r.arg.pat_addr));
  RT_CHECK(vx_buffer_address(gmem_buffer, &r.arg.gmem_addr));
  r.arg.gmem_addr = (r.arg.gmem_addr + 4095) & ~4095ull;
  RT_CHECK(vx_buffer_address(out_buffer, &r.arg.out_addr));
  r.arg.span = SPAN;
  std::vector<uint64_t> out((size_t)r.blocks * num_warps * 2 + 8, 0);
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &r.arg;
  li.args_size    = sizeof(r.arg);
  li.ndim         = 1;
  li.grid_dim[0]  = r.blocks;
  li.grid_dim[1]  = 1;
  li.grid_dim[2]  = 1;
  li.block_dim[0] = r.warps * warp_lanes;
  li.block_dim[1] = 1;
  li.block_dim[2] = 1;
  li.lmem_size    = r.arg.lmem_bytes;
  if (r.cluster > 1) {
    li.cluster_dim[0] = r.cluster;
    li.cluster_dim[1] = 1;
    li.cluster_dim[2] = 1;
    li.cluster_policy = VX_CLUSTER_POLICY_SPREAD;
  }
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_enqueue_read(queue, out.data(), out_buffer, 0, out.size() * 8, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  return out;
}

// Payload per cycle over the warps `sel` picks: their bytes over their span.
template <typename Sel>
static double measured_rate(const std::vector<uint64_t>& out, const Run& r, double per_warp, Sel sel) {
  uint64_t t0 = UINT64_MAX, t1 = 0;
  uint32_t n = 0;
  for (uint32_t b = 0; b < r.blocks; ++b) {
    for (uint32_t w = 0; w < r.warps; ++w) {
      if (!sel(b, w)) continue;
      size_t i = ((size_t)b * num_warps + w) * 2;
      t0 = std::min(t0, out[i]);
      t1 = std::max(t1, out[i + 1]);
      ++n;
    }
  }
  return (n == 0 || t1 <= t0) ? 0.0 : n * per_warp / (double)(t1 - t0);
}

static uint32_t iters_for(double rate, uint32_t warps, double payload_per_instr) {
  uint32_t it = (uint32_t)std::ceil(30000.0 * rate / (warps * payload_per_instr));
  return std::max<uint32_t>(32, (it + 31) & ~31u);
}

// A remote request stream is sized at the transport's one lane per cycle, not at
// the H100 rate: rank 1 waits at the closing barrier for the whole stream, and
// sized at the H100 rate that wait outlasts the scheduler watchdog.
static uint32_t remote_iters(double exp, uint32_t warps, double per_instr) {
  return iters_for(std::min(exp, per_instr / warp_lanes), warps, per_instr);
}

// Lane i at (i / group) * gstride + (i % group) * stride; group 0 = all lanes one group.
static std::vector<uint32_t> pattern(uint32_t lanes, uint32_t stride, uint32_t group = 0, uint32_t gstride = 0) {
  std::vector<uint32_t> p(SP_LANES, 0xFFFFFFFFu);
  const uint32_t g = group ? group : lanes;
  for (uint32_t i = 0; i < lanes; ++i) p[i] = (i / g) * gstride + (i % g) * stride;
  return p;
}

// Memory blocks per cycle the levels below the port take, measured with -m 1's
// stream (one single-block item per sector: the most blocks per cycle the port hands down).
static double lower_levels_accept(Run r, const SmPortCfg& cfg) {
  static double accept = -1.0;
  if (accept < 0) {
    auto pat = pattern(8, 4);
    auto items = items_of(Kind::GStore, pat, 4, 0, cfg);
    double payload = payload_of(items);
    r.arg.kind = SP_GSTORE;
    r.arg.step = store_step(pat, 4);
    r.arg.iters = iters_for(golden_rates(items, cfg)[0], r.warps, payload);
    auto out = launch(r, pat);
    double blocks = 0;
    for (auto& x : items) blocks += x.blocks;
    accept = measured_rate(out, r, r.arg.iters * payload, [](uint32_t, uint32_t) { return true; })
           * blocks / payload;
    printf("  lower levels accept %.3f blocks/cycle\n", accept);
  }
  return accept;
}

// Share of the run after the first warp finishes: the core front end serves the
// remaining warps alone, so the port no longer sees every warp's stream.
static double finish_spread(const std::vector<uint64_t>& out, const Run& r) {
  uint64_t t0 = UINT64_MAX, f = UINT64_MAX, t1 = 0;
  for (uint32_t b = 0; b < r.blocks; ++b) {
    for (uint32_t w = 0; w < r.warps; ++w) {
      size_t i = ((size_t)b * num_warps + w) * 2;
      t0 = std::min(t0, out[i]);
      f  = std::min(f, out[i + 1]);
      t1 = std::max(t1, out[i + 1]);
    }
  }
  return (t1 <= t0) ? 0.0 : (double)(t1 - f) / (double)(t1 - t0);
}

// The SM path's own rate is judged only where the levels below take what it
// needs and every warp stores until the end; otherwise the unit bench
// (hw/unittest/sm_port) is the criterion.
static void rate_line(const std::string& what, double got, double exp, double tol_rel, double bound, double accept,
                      double spread) {
  if (bound < exp * (1 - tol_rel)) {
    printf("  %-48s measured %9.3f golden %9.3f lower levels accept %.3f blocks/cycle: %.3f "
           "(report: lower-level bound; unit bench is the criterion)\n", what.c_str(), got, exp, accept, bound);
    return;
  }
  if (spread > tol_rel) {
    printf("  %-48s measured %9.3f golden %9.3f warps finish over the last %.1f %% of the run "
           "(report: front-end tail; unit bench is the criterion)\n", what.c_str(), got, exp, 100.0 * spread);
    return;
  }
  verdict(what, got, exp, tol_rel);
}

static void run_mode(int mode, const SmPortCfg& cfg) {
  std::cout << "mode " << mode << std::endl;
  Run r;
  r.warps = (uint32_t)num_warps;
  auto all = [](uint32_t, uint32_t) { return true; };

  const bool cluster_mode = (mode >= 4);
  if (cluster_mode) {
#if !VX_CFG_EXT_DLMEM_ENABLED
    std::cout << "Error: the SM-to-SM modes need -DVX_CFG_EXT_DLMEM_ENABLE" << std::endl;
    ++errors;
    return;
#endif
    r.cluster = (mode == 5) ? 4 : 2;
    r.blocks = r.cluster;
    r.arg.lmem_bytes = SPAN;
    if (max_cluster < r.cluster || num_cores < r.cluster) {
      std::cout << "Error: needs a cluster of " << r.cluster << " on " << r.cluster << " cores" << std::endl;
      ++errors;
      return;
    }
  }

  switch (mode) {
  case 0: case 1: case 2: {
    struct Case { std::string name; std::vector<uint32_t> pat; };
    std::vector<Case> cases;
    if (mode == 0)
      cases.push_back({warp_lanes * 4 >= LINE ? "full-line stores B/cycle"
                                              : std::to_string(warp_lanes) + " 4-B lanes per line B/cycle",
                       pattern(warp_lanes, 4)});
    if (mode == 1) cases.push_back({"one full sector per line B/cycle", pattern(8, 4)});
    if (mode == 2) {
      cases.push_back({"4-B lane in each sector: payload B/cycle", pattern(LINE / cfg.flit_bytes, cfg.flit_bytes)});
      cases.push_back({"4-B lane in one sector: payload B/cycle", pattern(1, 4)});
    }
    const double accept = lower_levels_accept(r, cfg);
    for (auto& c : cases) {
      auto items = items_of(Kind::GStore, c.pat, 4, 0, cfg);
      double payload = payload_of(items);
      double exp = golden_rates(items, cfg)[0];
      double bound = golden_rates(items, cfg, accept)[0];
      r.arg.kind = SP_GSTORE;
      r.arg.step = store_step(c.pat, 4);
      r.arg.iters = iters_for(exp, r.warps, payload);
      auto out = launch(r, c.pat);
      double got = measured_rate(out, r, r.arg.iters * payload, all);
      printf("  %u packet(s) of %.0f flits per instruction\n", (uint32_t)items.size(), flits_of(items) / items.size());
      rate_line(c.name, got, exp, mode == 2 ? 0.05 : 0.03, bound, accept, finish_spread(out, r));
      if (mode == 0)
        printf("H100REF metric=sm_port_full_line_store_Bpc h100=31.92 vortex=%.3f\n", got);
      if (mode == 1)
        printf("H100REF metric=sm_port_1sector_store_rpc h100=0.617 vortex=%.3f\n", got / cfg.flit_bytes);
    }
    break;
  }
  case 3: {
    // One full-line store per instruction (lane i at 4 i), 8 lines per warp per round.
    const double accept = lower_levels_accept(r, cfg);
    r.arg.kind = SP_RW;
    r.arg.n = (256 + r.warps * 8 - 1) / (r.warps * 8) * (r.warps * 8);
    auto out = launch(r, pattern(warp_lanes, 4));
    std::vector<uint32_t> offs;
    for (uint32_t i = 0; i < warp_lanes; ++i) offs.push_back(i * 4);
    auto items = items_of(Kind::GStore, offs, 4, 0, cfg);
    double flits = flits_of(items), blocks = 0;
    for (auto& x : items) blocks += x.blocks;
    SmEgressRef eg(cfg);
    double drain = (double)r.arg.n * flits / eg.units();
    printf("  load alone %llu cycles; %u full-line stores issued in %llu cycles; the load behind them usable after %llu\n",
           (unsigned long long)out[0], r.arg.n, (unsigned long long)out[1], (unsigned long long)out[2]);
    // The stores leave at the port's rate, and the load waits for them (one in-order FIFO).
    double below = (double)r.arg.n * blocks / accept;
    if (below <= drain * 1.05) {
      verdict("store issue cycles (the queued flits' drain)", (double)out[1], drain, 0.05);
    } else {
      printf("  %-48s measured %9.0f golden %9.0f lower levels accept %.3f blocks/cycle: %.0f "
             "(report: lower-level bound; unit bench is the criterion)\n",
             "store issue cycles (the queued flits' drain)", (double)out[1], drain, accept, below);
    }
    bool ok = out[2] >= 0.97 * drain;
    printf("  %-48s %llu >= %.0f %s\n", "load behind queued stores", (unsigned long long)out[2], 0.97 * drain,
           ok ? "ok" : "MISMATCH");
    if (!ok) ++errors;
    break;
  }
  case 4: case 5: case 6: {
    std::vector<uint32_t> pat = (mode == 6) ? pattern(warp_lanes, 8, 4, LINE) : pattern(warp_lanes, 8);
    auto stream = items_of(Kind::DStore, pat, 8, 0, cfg);
    double per_instr = payload_of(stream);
    double exp = golden_rates(stream, cfg)[0];
    // One message of 8 B per flit: the link's bound, and the receiver's ejection link's.
    double bound = 8.0 * cfg.egress_bytes / cfg.flit_bytes;
    uint32_t senders = 1;
    if (mode == 5) {
      senders = r.cluster - 1;
      bound = 8.0 * cfg.eject_bytes / cfg.flit_bytes;
      exp = std::min(exp * senders, bound);
    }
    r.arg.kind = SP_DSTORE;
    r.arg.recv = (mode == 5) ? 0 : 1;
    r.arg.step = store_step(pat, 8);
    r.arg.iters = iters_for(exp / senders, r.warps, per_instr);
    auto out = launch(r, pat);
    uint32_t recv = r.arg.recv;
    double got = measured_rate(out, r, r.arg.iters * per_instr, [&](uint32_t b, uint32_t) { return b != recv; });
    const char* name = (mode == 4) ? "rank 0 -> rank 1 full lines B/cycle"
                     : (mode == 5) ? "ranks 1..3 -> rank 0 B/cycle at the receiver" : "one sector per line: unique B/cycle";
    report(name, got, exp, mode == 5 ? "sm_port_dsmem_eject_Bpc" : "sm_port_dsmem_store_Bpc", 21.3);
    within(std::string(name) + " within the link", got, bound);
    break;
  }
  case 7: {
    // The served replies: one 8-B message per lane of the requester's instruction.
    auto pat = pattern(warp_lanes, 8);
    auto stream = items_of(Kind::DLoadRsp, pat, 8, 0, cfg);
    double per_instr = payload_of(stream);
    double exp = golden_rates(stream, cfg)[0];
    r.arg.kind = SP_DLOAD;
    r.arg.step = span_of(pat, 8);
    r.arg.iters = remote_iters(exp, r.warps, per_instr);
    auto out = launch(r, pat);
    double got = measured_rate(out, r, r.arg.iters * per_instr, [](uint32_t b, uint32_t) { return b == 0; });
    // H100: a reply is 1 slot per sector (32 B) at the sender's 2/3 slot per clock.
    report("rank 1's replies to rank 0's ld, B/cycle", got, exp, "sm_port_dsmem_reply_Bpc", 21.3);
    within("rank 1's replies within its link", got, 8.0 * cfg.egress_bytes / cfg.flit_bytes);
    break;
  }
  case 8: {
    auto pat = pattern(warp_lanes, 4);
    auto stream = items_of(Kind::DAtom, pat, 4, 0, cfg);
    double per_instr = payload_of(stream);
    double exp = golden_rates(stream, cfg)[0];
    r.arg.kind = SP_DATOM;
    r.arg.step = span_of(pat, 4);
    r.arg.iters = remote_iters(exp, r.warps, per_instr);
    auto out = launch(r, pat);
    double got = measured_rate(out, r, r.arg.iters * per_instr, [](uint32_t b, uint32_t) { return b == 0; });
    // H100: a 32-lane line is one 5-slot packet; single-op atomics 0.5 per clock.
    report("remote amoadd.w lane-ops/cycle", got, exp, "sm_port_dsmem_atom_lane_ops_pc", 32.0 * 2 / 3 / 5);
    within("remote amoadd.w lane-ops within the link", got, (double)cfg.egress_bytes / cfg.flit_bytes);
    break;
  }
  case 10: {
#if !(VX_CFG_EXT_DXA_ENABLED && VX_CFG_EXT_MBAR_ENABLED)
    std::cout << "Error: the TMA mode needs DXA and MBAR" << std::endl;
    ++errors;
#else
    r.arg.kind = SP_TMA;
    r.arg.iters = 16;
    r.warps = 1;
    auto out = launch(r, pattern(warp_lanes, 4));
    const double bytes = r.arg.iters * 4.0 * 4096;
    auto rate = [&](uint32_t ph, uint32_t rank) {
      size_t i = (ph * 2 + rank) * 2;
      return bytes / (double)(out[i + 1] - out[i]);
    };
    // A TMA engine or SM port shared by the TPC's two SMs halves each SM's rate when both run (<= 0.5);
    // one per SM keeps it up to the shared L2/DRAM's contention. 0.75 separates the two structures.
    // H100 reference: two SMs of one TPC each keep their one-SM port rate (SMPORT_flit_h100.md w_tpc2/w_sm1).
    for (uint32_t rank = 0; rank < 2; ++rank) {
      double alone = rate(rank, rank);
      double ratio = rate(2, rank) / alone;
      bool ok = ratio > 0.75;
      printf("  rank %u: alone %.3f B/cycle, beside the other SM %.3f B/cycle, ratio %.3f (> 0.75) %s\n",
             rank, alone, rate(2, rank), ratio, ok ? "ok" : "MISMATCH");
      if (!ok) ++errors;
      printf("H100REF metric=tpc2_ratio_rank%u h100=%.3f vortex=%.3f\n", rank, 31.91 / 31.92, ratio);
    }
#endif
    break;
  }
  case 11: {
#if !(VX_CFG_EXT_DXA_ENABLED && VX_CFG_EXT_MBAR_ENABLED)
    std::cout << "Error: the fill-write mode needs DXA and MBAR" << std::endl;
    ++errors;
#else
    const double D = VX_CFG_SM_DATAPATH_BYTES;
    const double per_instr = warp_lanes * 8.0;
    if (r.warps < 2) {
      std::cout << "Error: the fill-write mode needs at least 2 warps" << std::endl;
      ++errors;
      break;
    }
    r.arg.kind = SP_FILLWR;
    r.arg.extra = r.blocks * (uint32_t)num_warps * 2;
    r.arg.iters = std::max<uint32_t>(16, (uint32_t)(12000 * D / ((r.warps - 1) * 16 * per_instr)));
    auto stream_of = [&](const std::vector<uint64_t>& out) {
      return measured_rate(out, r, r.arg.iters * 16.0 * per_instr, [](uint32_t b, uint32_t w) { return b == 0 && w != 0; });
    };
    auto run = [&](uint32_t cls, uint32_t bg, double* fill) {
      r.arg.recv = cls;
      r.arg.n = bg;
      auto out = launch(r, pattern(warp_lanes, 4));
      *fill = 0;
      uint64_t w0 = UINT64_MAX, w1 = 0, s1 = UINT64_MAX, e0 = 0;
      for (uint32_t w = 1; w < r.warps; ++w) {
        w0 = std::min(w0, out[w * 2]);
        s1 = std::min(s1, out[w * 2 + 1]);
        e0 = std::max(e0, out[w * 2]);
        w1 = std::max(w1, out[w * 2 + 1]);
      }
      printf("    stream warps start %llu..%llu, end %llu..%llu", (unsigned long long)w0, (unsigned long long)e0,
             (unsigned long long)s1, (unsigned long long)w1);
      if (bg == SP_BG_TMA)
        printf("; TMA phases %llu, first done %llu, last done %llu", (unsigned long long)out[r.arg.extra + 3],
               (unsigned long long)out[r.arg.extra + 4], (unsigned long long)out[r.arg.extra + 5]);
      printf("\n");
      if (bg == SP_BG_TMA && out[r.arg.extra + 2] != 0) {
        *fill = (double)out[r.arg.extra + 1] / (double)out[r.arg.extra + 2];
      } else if (bg == SP_BG_DSMEM) {
        uint64_t t0 = UINT64_MAX, t1 = 0;
        for (uint32_t w = 0; w < r.warps; ++w) {
          t0 = std::min(t0, out[(num_warps + w) * 2]);
          t1 = std::max(t1, out[(num_warps + w) * 2 + 1]);
        }
        *fill = (t1 > t0) ? (double)out[r.arg.extra] / (double)(t1 - t0) : 0;
      }
      return stream_of(out);
    };
    auto at_least = [&](const std::string& what, double got, double lo) {
      bool ok = got >= lo - 1e-9;
      printf("  %-48s measured %9.3f expected >= %.3f %s\n", what.c_str(), got, lo, ok ? "ok" : "MISMATCH");
      if (!ok) ++errors;
    };
    double f = 0;
    printf("  LDS stream alone: %.3f B/cycle\n", run(SP_FW_LDS, SP_BG_NONE, &f));
    double lds = run(SP_FW_LDS, SP_BG_TMA, &f);
    printf("  TMA fills beside LDS: %.3f B/cycle (H100: LDS 122.4 beside 82)\n", f);
    at_least("LDS B/cycle beside TMA fills", lds, D * 15 / 16);
    double l1 = run(SP_FW_L1, SP_BG_TMA, &f);
    printf("  TMA fills beside L1 hits: %.3f B/cycle\n", f);
    at_least("L1-hit B/cycle beside TMA fills", l1, D * 15 / 16);
    double sts = run(SP_FW_STS, SP_BG_TMA, &f);
    printf("  TMA fills beside STS: %.3f B/cycle (H100: STS 30, gap logged)\n", f);
    verdict("STS B/cycle beside TMA fills", sts, D - f, 0.05);
    sts = run(SP_FW_STS, SP_BG_DSMEM, &f);
    printf("  DSMEM arrivals beside STS: %.3f B/cycle (H100: STS 66 beside 32)\n", f);
    verdict("STS B/cycle beside DSMEM arrivals", sts, D * (1.0 - f / VX_CFG_SM_SRAM_DSMEM_WR_BYTES), 0.10);
    lds = run(SP_FW_LDS, SP_BG_DSMEM, &f);
    printf("  DSMEM arrivals beside LDS: %.3f B/cycle\n", f);
    verdict("LDS B/cycle beside DSMEM arrivals", lds, D, 0.03);
#endif
    break;
  }
  case 9: {
    // Even warps' DSMEM stores and odd warps' global stores share the one in-order FIFO.
    auto pat = pattern(warp_lanes, 8);
    auto dsm_items = items_of(Kind::DStore, pat, 8, 0, cfg);
    auto l2_items = items_of(Kind::GStore, pat, 8, 1, cfg);
    std::vector<Item> stream = dsm_items;
    stream.insert(stream.end(), l2_items.begin(), l2_items.end());
    auto g = golden_rates(stream, cfg);
    double per_instr = payload_of(dsm_items);
    r.arg.kind = SP_MIX;
    r.arg.step = store_step(pat, 8);
    r.arg.iters = iters_for(g[0] + g[1], r.warps, per_instr);
    auto out = launch(r, pat);
    double dsm = measured_rate(out, r, r.arg.iters * per_instr, [](uint32_t b, uint32_t w) { return b == 0 && (w & 1) == 0; });
    double l2  = measured_rate(out, r, r.arg.iters * per_instr, [](uint32_t b, uint32_t w) { return b == 0 && (w & 1) == 1; });
    printf("  DSMEM %.3f + global %.3f B/cycle (golden: %.3f + %.3f)\n", dsm, l2, g[0], g[1]);
    // Flits per cycle: a DSMEM message is one flit for 8 B, a global line one flit per 32-B sector.
    double fl = dsm / 8.0 + l2 * flits_of(l2_items) / payload_of(l2_items);
    report("DSMEM + global flits/cycle", fl, g[0] / 8.0 + g[1] * flits_of(l2_items) / payload_of(l2_items),
           "sm_port_mix_flits_pc", 1.0);
    within("DSMEM + global flits within the link", fl, (double)cfg.egress_bytes / cfg.flit_bytes);
    break;
  }
  }

}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  if (modes.empty()) modes.push_back(0);
  for (int m : modes) {
    if (m < 0 || m > 11) {
      show_usage();
      return -1;
    }
  }
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  uint64_t num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  if (num_threads > SP_LANES) {
    std::cout << "Error: needs at most " << SP_LANES << " threads, device has " << num_threads << std::endl;
    cleanup();
    return -1;
  }
  warp_lanes = (uint32_t)num_threads;
#if VX_CFG_EXT_DLMEM_ENABLED
  RT_CHECK(vx_device_query(device, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
#endif
  const SmPortCfg cfg = knob_cfg();
  std::cout << "egress " << cfg.egress_bytes << " B/cycle, flit " << cfg.flit_bytes << " B, ejection "
            << cfg.eject_bytes << " B/cycle; " << num_cores
            << " cores, " << num_warps << " warps" << std::endl;

  RT_CHECK(vx_buffer_create(device, SP_LANES * 4, VX_MEM_READ, &pat_buffer));
  RT_CHECK(vx_buffer_create(device, 4 * SPAN + 3 * 4096, VX_MEM_READ_WRITE, &gmem_buffer));
  RT_CHECK(vx_buffer_create(device, (num_cores * num_warps * 2 + 8) * 8, VX_MEM_READ_WRITE, &out_buffer));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  for (int m : modes) {
    run_mode(m, cfg);
  }

  cleanup();
  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
