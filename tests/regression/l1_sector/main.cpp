// l1_sector: the H100 SM path's L1 (128-B lines, 32-B sectors, word-valid
// store allocate, hashed index, slots on the age list, invalidation).
//
// Expectations come from sw/common/l1_slot_ref.h at the build's knobs; nothing
// measured on Vortex is read back as a target. A timed load counts as a miss
// when it takes more than the midpoint of a timed L1 hit and a timed L2 hit of
// one line, measured at start-up. H100 measurements are printed as
// "H100REF metric=... h100=... vortex=..." lines, never judged.
//
//   -m 0 hitlat     one-thread ld chase on L1-resident lines: l1_hit_latency (l1_slot_ref.h) +-1
//   -m 1 l2hit      ld chase over L2-resident, L1-cold lines (gpu fence after warming): above the
//                   L1 hit's stages plus the L1 miss path's own (reported against the H100's)
//   -m 2 multiline  multi-lane ld chases on resident lines: l1_hit_latency over the lines' slot rows +-1
//   -m 3 fill       4-B miss at offset 0, then a timed load at +X: hit/miss as the golden's sectors
//   -m 4 wpolicy    store/load sequences of 05 sec 4.5: hit/miss as the golden's byte-valid lines
//   -m 5 assoc      every resident line of the launch's config at each pow2 stride 128 B..64 KiB: no misses
//   -m 6 repl       at the largest config and the no-hint config: cyclic passes over 1.1/1.2/1.4/2.5 x
//                   the L1, and an overfill probe: miss rates within the golden's range over the LFSR phases +-2 points
//   -m 7 inval      gpu and cluster fences and the next kernel invalidate, a cta fence does not
//   -m 8 bw         every warp loads one resident line per request: the add twin's issue rate
//                   (capped at one line per cycle) +-5 %
//   -m 9 banks      warp-wide ld hits on L lines (offset 0 / diagonal / sector j), lines 128 B, 4 KiB, 64 KiB apart:
//                   l1_hit_latency over the lines' slot rows +-1
//   -m 10 refill    per carve-out config: an L2-hit lw stream over min(4 x L1, 3/4 L2) lines (a full L1,
//                   age-list victims) >= 0.95 x the same stream over half the L1 after an invalidate;
//                   configs whose L1 holds <= 2 x SM_MAX_READ_LINES lines are reported only

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include <vortex.h>
#include <l1_slot_ref.h>
#include <sm_port_ref.h>
#include <smem_bank_ref.h>
#include <smem_carveout.h>

#include "common.h"

using namespace vortex::l1_ref;
namespace sr = vortex::l1_slot_ref;
namespace sb = vortex::smem_bank_ref;

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
#error "l1_sector tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t LINE = VX_CFG_DCACHE_LINE_SIZE;
uint64_t miss_thr = 0;     // calibrate_miss_thr()

const char* kernel_file = "kernel.vxbin";
std::vector<int> modes;
bool quick = false;

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_buffer_h data_buffer = nullptr;
vx_buffer_h tab_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
uint64_t num_warps = 0;
uint32_t lanes = 0;  // the device's warp width
uint64_t data_base = 0;   // line-aligned device address inside data_buffer
uint64_t tab_addr = 0;
uint64_t out_addr = 0;
uint32_t errors = 0;
uint32_t carve_field = 0;  // vx_launch_info_t::smem_carveout: 0 = no preference
SmemCarveCfg carve;

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-m mode[,mode...] 0..10] [-q shorter repl run] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:m:qh")) != -1) {
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
    case 'q': quick = true; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (data_buffer) vx_buffer_release(data_buffer);
    if (tab_buffer) vx_buffer_release(tab_buffer);
    if (out_buffer) vx_buffer_release(out_buffer);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static void verdict(const std::string& what, double got, double exp, double tol_abs, double tol_rel) {
  double tol = std::max(tol_abs, std::fabs(exp) * tol_rel);
  bool ok = std::fabs(got - exp) <= tol + 1e-9;
  printf("  %-44s measured %9.3f expected %9.3f (+-%.3f) %s\n", what.c_str(), got, exp, tol, ok ? "ok" : "MISMATCH");
  if (!ok) ++errors;
}

static void verdict_range(const std::string& what, double got, double lo, double hi, double tol_abs) {
  bool ok = got >= lo - tol_abs - 1e-9 && got <= hi + tol_abs + 1e-9;
  printf("  %-44s measured %9.3f expected [%.3f, %.3f] (+-%.3f) %s\n", what.c_str(), got, lo, hi, tol_abs,
         ok ? "ok" : "MISMATCH");
  if (!ok) ++errors;
}

static void verdict_bool(const std::string& what, bool hit, bool exp_hit, uint64_t cycles) {
  bool ok = (hit == exp_hit);
  printf("  %-44s %s (%llu cycles), expected %s %s\n", what.c_str(), hit ? "hit " : "miss",
         (unsigned long long)cycles, exp_hit ? "hit" : "miss", ok ? "ok" : "MISMATCH");
  if (!ok) ++errors;
}

static uint64_t data_buf_addr = 0;

// The queue reads the host buffer when the write runs, so wait for it.
static void write_sync(vx_buffer_h buf, uint64_t off, const void* src, size_t bytes) {
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_write(queue, buf, off, src, bytes, 0, nullptr, &ev));
  RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
}

static void write_at(uint64_t dev_addr, const void* src, size_t bytes) {
  write_sync(data_buffer, dev_addr - data_buf_addr, src, bytes);
}

static void launch(kernel_arg_t& arg, uint32_t warps, std::vector<uint64_t>& out) {
  arg.out_addr = out_addr;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = warps * lanes;
  li.lmem_size    = 0;
  li.smem_carveout = carve_field;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_enqueue_read(queue, out.data(), out_buffer, 0, out.size() * sizeof(uint64_t), 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
}

// Per-lane pointer rings: lane i's word in row k points to its word in row k+1.
static void build_rings(const std::vector<uint32_t>& off, uint32_t rows, uint32_t row_bytes,
                        std::vector<uint64_t>& starts) {
  std::vector<uint64_t> img((size_t)rows * row_bytes / 8, 0);
  starts.assign(LS_LANES, 0);
  for (size_t i = 0; i < off.size(); ++i) {
    for (uint32_t k = 0; k < rows; ++k) {
      uint64_t a = (uint64_t)k * row_bytes + off[i];
      img[a / 8] = data_base + (uint64_t)((k + 1) % rows) * row_bytes + off[i];
    }
    starts[i] = data_base + off[i];
  }
  write_at(data_base, img.data(), img.size() * 8);
  write_sync(tab_buffer, 0, starts.data(), starts.size() * 8);
}

static double chase(const std::vector<uint32_t>& off, uint32_t rows, uint32_t row_bytes, uint32_t warm,
                    uint32_t hops, uint32_t fence, std::vector<uint64_t>& out) {
  std::vector<uint64_t> starts;
  build_rings(off, rows, row_bytes, starts);
  kernel_arg_t arg = {};
  arg.kind = LS_CHASE;
  arg.lanes = (uint32_t)off.size();
  arg.warm = warm;
  arg.hops = hops;
  arg.fence = fence;
  arg.start_addr = tab_addr;
  launch(arg, 1, out);
  return (double)out[0] / hops;
}

struct Op { uint32_t op; uint64_t addr; };

static std::vector<uint64_t> run_ops(const std::vector<Op>& ops, std::vector<uint64_t>& out) {
  std::vector<uint64_t> tab;
  for (auto& o : ops) { tab.push_back(o.op); tab.push_back(o.addr); }
  write_sync(tab_buffer, 0, tab.data(), tab.size() * 8);
  kernel_arg_t arg = {};
  arg.kind = LS_OPS;
  arg.nops = (uint32_t)ops.size();
  arg.start_addr = tab_addr;
  launch(arg, 1, out);
  return std::vector<uint64_t>(out.begin(), out.begin() + ops.size());
}

static uint64_t sweep(uint64_t base, uint32_t n, uint32_t stride, uint32_t laps, uint32_t timed_laps,
                      uint32_t tail_from, std::vector<uint64_t>& out) {
  kernel_arg_t arg = {};
  arg.kind = LS_SWEEP;
  arg.n = n;
  arg.stride = stride;
  arg.laps = laps;
  arg.timed_laps = timed_laps;
  arg.tail_from = tail_from;
  arg.miss_thr = (uint32_t)miss_thr;
  arg.start_addr = base;
  launch(arg, 1, out);
  return out[0];
}

// The launch's shared-memory config: the hint when one is set, else the no-hint rule.
static uint32_t launch_s(uint32_t warps) {
  return (uint32_t)smem_config_choose(0, warps * lanes, false, (int32_t)carve_field - 1, carve);
}

static uint32_t usable_lines(uint32_t warps) {
  return smem_usable_l1(launch_s(warps), carve) / LINE;
}

static sr::L1SlotCfg slot_cfg(uint32_t s_bytes) {
  sr::L1SlotCfg c{};
  c.s_bytes = s_bytes;
  c.sram_bytes = VX_CFG_SM_SRAM_BYTES;
  c.reserve_bytes = VX_CFG_L1_RESERVE_BYTES;
  c.row_bytes = LINE;
  c.valid_bytes = VX_CFG_L1_VALID_BYTES;
  c.sector_bytes = VX_CFG_DCACHE_SECTOR_SIZE;
  c.tag_sets = VX_CFG_L1_TAG_SETS;
  c.tag_ways = VX_CFG_L1_TAG_WAYS;
  c.tag_banks = VX_CFG_L1_TAG_BANKS;
  c.index_hash = VX_CFG_L1_INDEX_HASH;
  c.protect_pct = VX_CFG_L1_REPL_PROTECT_PCT;
  c.mshr_lines = VX_CFG_SM_MAX_READ_LINES;
  c.max_lines = VX_CFG_L1_MAX_LINES;
  return c;
}

// The slot golden with one access in flight: a miss allocates, fills its
// sectors and retires before the next access starts.
struct SlotSerial {
  sr::L1SlotDir d;

  explicit SlotSerial(uint32_t s_bytes) { d.configure(slot_cfg(s_bytes)); }

  void alloc(uint64_t line, uint32_t sectors) {
    for (;;) {
      auto a = d.allocate(line, true, 0, sectors);
      if (a.ok) {
        if (sectors) {
          d.fill(a.slot, sectors);
          d.fill_done(a.slot);
          d.replay_done(a.slot);
        }
        return;
      }
      if (a.wait == sr::L1SlotDir::W_SET_IN_FLIGHT || a.wait == sr::L1SlotDir::W_ENTRY_OPEN)
        std::abort();
      d.tick();
    }
  }

  void settle() {
    d.tick();
    d.tick();
  }

  bool load(uint64_t addr, uint32_t bytes) {
    const auto& c = d.cfg();
    const uint64_t line = addr / LINE;
    const uint32_t off = (uint32_t)(addr % LINE);
    const bool hit = d.lookup(line, sr::l1_touched_words(off, bytes, c)).hit;
    if (!hit) {
      uint32_t sec = 0;
      for (uint32_t k = off / c.sector_bytes; k <= (off + bytes - 1) / c.sector_bytes; ++k) sec |= 1u << k;
      alloc(line, sec);
    }
    settle();
    return hit;
  }

  void store(uint64_t addr, uint32_t bytes) {
    const uint64_t line = addr / LINE;
    if (!d.lookup(line, 0).tag) {
      alloc(line, 0);
      d.tick();
    }
    d.store(line, sr::l1_full_words((uint32_t)(addr % LINE), bytes, d.cfg()));
    settle();
  }
};

static uint64_t golden_sweep_misses(SlotSerial& ref, uint64_t base, uint32_t n, uint32_t stride, uint32_t laps,
                                    uint32_t timed_laps, uint32_t tail_from) {
  uint64_t misses = 0;
  for (uint32_t l = 0; l < laps; ++l) {
    bool timed = (l + timed_laps >= laps);
    for (uint32_t i = (timed ? tail_from : 0); i < n; ++i) {
      bool hit = ref.load(base + (uint64_t)i * stride, 4);
      if (timed && !hit) ++misses;
    }
  }
  return misses;
}

// The kernel's earlier misses step the L1's LFSR (never reseeded), so the
// replacement golden is the range over its starting phases.
static std::pair<uint64_t, uint64_t> golden_sweep_range(uint32_t s_bytes, uint64_t base, uint32_t n, uint32_t stride,
                                                         uint32_t laps, uint32_t timed_laps, uint32_t tail_from) {
  const uint32_t phases = 256;
  sr::L1Lfsr g;
  uint64_t lo = UINT64_MAX, hi = 0;
  for (uint32_t k = 0; k < phases; ++k) {
    SlotSerial ref(s_bytes);
    for (uint32_t i = 0; i < 251; ++i) g.next();
    ref.d.set_lfsr_state(g.s);
    uint64_t m = golden_sweep_misses(ref, base, n, stride, laps, timed_laps, tail_from);
    lo = std::min(lo, m);
    hi = std::max(hi, m);
  }
  return {lo, hi};
}

// One warp load hitting the L1, `bytes` per lane: wavefronts over the lines'
// slot rows (distinct lines sit in distinct rows), plus tag cycles.
static uint32_t slot_hit_latency(const std::vector<uint64_t>& addr, uint32_t bytes) {
  const uint32_t n = (uint32_t)addr.size();
  const uint32_t bank_bytes = VX_CFG_LMEM_BANK_BYTES, banks = VX_CFG_LMEM_NUM_BANKS;
  uint32_t row[LS_LANES], lo[LS_LANES], nb[LS_LANES];
  uint64_t lines[LS_LANES];
  uint32_t mask = 0;
  for (uint32_t i = 0; i < n; ++i) {
    lines[i] = addr[i] / LINE;
    row[i] = (uint32_t)lines[i];
    lo[i] = (uint32_t)((addr[i] % LINE) / bank_bytes % banks);
    nb[i] = (bytes + bank_bytes - 1) / bank_bytes;
    mask |= 1u << i;
  }
  const uint32_t w = sb::sram_row_wavefronts(row, lo, nb, mask, n);
  sr::L1SlotCfg c = slot_cfg(launch_s(1));
  uint32_t tc = sr::l1_tag_cycles(lines, n, VX_CFG_L1_TAG_BANKS, c);
  return sr::l1_hit_latency(w, tc, sr::l1_pipe_stages(VX_CFG_NUM_OPCS));
}

// The miss classifier: the midpoint of one line's timed L2 hit (loaded, then a gpu fence clears the L1)
// and its timed L1 hit right after, with the same timed-load code the classified loads use.
static void calibrate_miss_thr(std::vector<uint64_t>& out) {
  const uint64_t a = data_base + 49152;
  auto r = run_ops({{OP_LOAD, a}, {OP_FENCE_GPU, 0}, {OP_TIMED, a}, {OP_TIMED, a}}, out);
  miss_thr = (r[2] + r[3]) / 2;
  printf("timed load: L2 hit %llu, L1 hit %llu cycles, miss threshold %llu %s\n", (unsigned long long)r[2],
         (unsigned long long)r[3], (unsigned long long)miss_thr, r[2] > r[3] ? "ok" : "MISMATCH (hit not below miss)");
  if (r[2] <= r[3]) ++errors;
}

static void run_mode(int mode, std::vector<uint64_t>& out) {
  std::cout << "mode " << mode << std::endl;
  auto hit_lat = [&](const std::vector<uint32_t>& off) {
    std::vector<uint64_t> addr;
    for (auto o : off) addr.push_back(data_base + o);
    return slot_hit_latency(addr, 8);
  };

  switch (mode) {
  case 0: {
    std::vector<uint32_t> off = {0};
    double got = chase(off, 64, LINE, 64, 4096, 0, out);
    verdict("one-thread ld chase, L1 hit", got, hit_lat(off), 1, 0);
    printf("H100REF metric=l1hit_chase h100=32.02 vortex=%.3f\n", got);
    break;
  }
  case 1: {
    std::vector<uint32_t> off = {0};
    double got = chase(off, 4096, LINE, 4096, 4096, OP_FENCE_GPU, out);
    // The miss path's stages and the SM port's output and input registers; the L2 adds at least a cycle.
    const uint32_t min_miss = sr::l1_miss_pipe_stages(VX_CFG_NUM_OPCS) + vortex::sm_port_ref::SM_PORT_OUT_STAGES
                            + vortex::sm_port_ref::SM_PORT_IN_STAGES;
    bool ok = got > min_miss;
    printf("  %-44s measured %9.3f per hop, above the miss-path and SM port stages %u %s\n",
           "one-thread ld chase, L1 miss, L2 hit", got, min_miss, ok ? "ok" : "MISMATCH");
    if (!ok) ++errors;
    printf("H100REF metric=l2hit_chase h100=279 vortex=%.3f\n", got);
    break;
  }
  case 2: {
    struct Row { std::string name; uint32_t lanes, stride, group; };
    const std::string warp = std::to_string(lanes) + " lanes";
    std::vector<Row> rows = {
      {"1 lane", 1, 8, 1}, {"16 lanes, one line", 16, 8, 1}, {"17 lanes, one address", 17, 8, lanes},
      {warp + ", one address", lanes, 8, lanes}, {warp + ", stride 8 B", lanes, 8, 1}, {warp + ", stride 16 B", lanes, 16, 1},
      {warp + ", stride 32 B", lanes, 32, 1}, {warp + ", stride 64 B", lanes, 64, 1}, {warp + ", stride 128 B", lanes, 128, 1},
      {"2 lanes, stride 128 B", 2, 128, 1}, {"4 lanes, stride 128 B", 4, 128, 1}, {"8 lanes, stride 128 B", 8, 128, 1},
      {"16 lanes, stride 128 B", 16, 128, 1}, {"17 lanes, stride 128 B", 17, 128, 1}, {"24 lanes, stride 128 B", 24, 128, 1}};
    double one = -1, all = -1;
    for (auto& r : rows) {
      if (r.lanes > lanes)
        continue;
      std::vector<uint32_t> off;
      for (uint32_t i = 0; i < r.lanes; ++i) off.push_back((i / r.group) * r.stride);
      double got = chase(off, 4, 4096, 4, 1024, 0, out);
      auto e = hit_lat(off);
      verdict(r.name, got, e, 1, 0);
      if (r.lanes == 1) one = got;
      if (r.lanes == lanes && r.stride == 128) all = got;
    }
    if (one > 0 && all > 0 && lanes > 1)
      printf("H100REF metric=l1hit_extra_line h100=2.23 vortex=%.3f\n", (all - one) / (lanes - 1));
    break;
  }
  case 3: {
    SlotSerial ref(launch_s(1));
    std::vector<Op> ops;
    std::vector<uint32_t> xs = {4, 8, 12, 16, 20, 24, 28, 32, 36, 60, 64, 96, 124};
    std::vector<bool> exp;
    for (size_t i = 0; i < xs.size(); ++i) {
      uint64_t a = data_base + (uint64_t)(i + 1) * 8192;
      ops.push_back({OP_LOAD, a});
      ops.push_back({OP_TIMED, a + xs[i]});
      ref.load(a, 4);
      exp.push_back(ref.load(a + xs[i], 4));
    }
    auto c = run_ops(ops, out);
    for (size_t i = 0; i < xs.size(); ++i) {
      verdict_bool("4-B miss at +0, then +" + std::to_string(xs[i]), c[2 * i + 1] <= miss_thr, exp[i], c[2 * i + 1]);
    }
    break;
  }
  case 4: {
    SlotSerial ref(launch_s(1));
    struct Seq { std::string name; std::vector<std::pair<uint32_t, uint32_t>> steps; };  // (op, offset)
    std::vector<Seq> seqs = {
      {"ld A; st A -> ld A", {{OP_LOAD, 0}, {OP_STORE, 0}, {OP_TIMED, 0}}},
      {"ld A; st A+4 -> ld A+4", {{OP_LOAD, 0}, {OP_STORE, 4}, {OP_TIMED, 4}}},
      {"st A -> ld A", {{OP_STORE, 0}, {OP_TIMED, 0}}},
      {"st A -> ld A+4", {{OP_STORE, 0}, {OP_TIMED, 4}}},
      {"whole sector stored -> ld A", {{OP_STORE, 0}, {OP_STORE, 4}, {OP_STORE, 8}, {OP_STORE, 12}, {OP_STORE, 16},
                                       {OP_STORE, 20}, {OP_STORE, 24}, {OP_STORE, 28}, {OP_TIMED, 0}}},
      {"7 of 8 words stored -> ld the 8th", {{OP_STORE, 0}, {OP_STORE, 4}, {OP_STORE, 8}, {OP_STORE, 12},
                                             {OP_STORE, 16}, {OP_STORE, 20}, {OP_STORE, 24}, {OP_TIMED, 28}}},
      {"st A; ld A+4 -> ld A", {{OP_STORE, 0}, {OP_LOAD, 4}, {OP_TIMED, 0}}}};
    std::vector<Op> ops;
    std::vector<size_t> idx;
    std::vector<bool> exp;
    for (size_t s = 0; s < seqs.size(); ++s) {
      uint64_t a = data_base + (uint64_t)(s + 1) * 8192;
      for (auto& st : seqs[s].steps) {
        uint64_t x = a + st.second;
        ops.push_back({st.first, x});
        if (st.first == OP_STORE) {
          ref.store(x, 4);
        } else {
          bool h = ref.load(x, 4);
          if (st.first == OP_TIMED) { idx.push_back(ops.size() - 1); exp.push_back(h); }
        }
      }
    }
    auto c = run_ops(ops, out);
    for (size_t s = 0; s < seqs.size(); ++s) {
      verdict_bool(seqs[s].name, c[idx[s]] <= miss_thr, exp[s], c[idx[s]]);
    }
    break;
  }
  case 5: {
    // Every resident line of the launch's config, at any power-of-two stride, fits its sets.
    const uint32_t n = usable_lines(1);
    const sr::L1SlotCfg c = slot_cfg(launch_s(1));
    for (uint32_t stride = LINE; stride <= 65536; stride *= 2) {
      std::vector<uint32_t> per_set(c.tag_sets, 0);
      uint32_t fullest = 0;
      for (uint32_t i = 0; i < n; ++i) {
        uint32_t set = sr::l1_set((data_base + (uint64_t)i * stride) / LINE, c.tag_sets, c.index_hash);
        fullest = std::max(fullest, ++per_set[set]);
      }
      uint64_t got = sweep(data_base, n, stride, 2, 1, 0, out);
      printf("  (%u lines, stride %u B: fullest set %u of %u ways)\n", n, stride, fullest, c.tag_ways);
      verdict(std::to_string(n) + " lines, stride " + std::to_string(stride) + " B: misses", (double)got, 0, 0, 0);
    }
    break;
  }
  case 6: {
    const std::vector<uint32_t> pcts = quick ? std::vector<uint32_t>{110} : std::vector<uint32_t>{110, 120, 140, 250};
    const uint32_t laps = quick ? 3 : 4;
    const uint32_t largest = carve.configs_kb[SMEM_CARVE_NUM_CONFIGS - 1];
    const uint32_t saved = carve_field;
    for (uint32_t field : {1u + largest * 100 / largest, 0u}) {
      carve_field = field;
      const uint32_t s = launch_s(1), cap = usable_lines(1);
      printf("  config %u KiB (%s): L1 %u lines%s\n", s / 1024, field ? "hint" : "no hint", cap,
             s == largest * 1024 ? "; H100 at 228 KiB: 9/27/50/100 % (05 sec 4.6)" : "");
      for (uint32_t pct : pcts) {
        uint32_t n = cap * pct / 100;
        auto exp = golden_sweep_range(s, data_base, n, LINE, laps, laps - 1, 0);
        uint64_t got = sweep(data_base, n, LINE, laps, laps - 1, 0, out);
        double denom = (laps - 1.0) * n;
        verdict_range("cyclic over " + std::to_string(pct) + " % of the L1: miss %", 100.0 * got / denom,
                      100.0 * exp.first / denom, 100.0 * exp.second / denom, 2);
      }
      uint32_t n = cap + cap / 4;
      auto exp = golden_sweep_range(s, data_base, n, LINE, 2, 1, cap);
      uint64_t got = sweep(data_base, n, LINE, 2, 1, cap, out);
      double denom = n - cap;
      verdict_range("overfill probe: newest lines' hit %", 100.0 - 100.0 * got / denom,
                    100.0 - 100.0 * exp.second / denom, 100.0 - 100.0 * exp.first / denom, 2);
    }
    carve_field = saved;
    break;
  }
  case 7: {
    std::vector<Op> ops;
    uint64_t a = data_base + 8192, b = data_base + 16384, c = data_base + 24576, d = data_base + 32768;
    ops.push_back({OP_LOAD, a}); ops.push_back({OP_FENCE_GPU, 0}); ops.push_back({OP_TIMED, a});
    ops.push_back({OP_LOAD, b}); ops.push_back({OP_FENCE_CLUSTER, 0}); ops.push_back({OP_TIMED, b});
#if VX_CFG_EXT_DLMEM_ENABLED
    ops.push_back({OP_LOAD, c}); ops.push_back({OP_FENCE_CTA, 0}); ops.push_back({OP_TIMED, c});
#endif
    ops.push_back({OP_LOAD, d});
    auto r = run_ops(ops, out);
    verdict_bool("ld A; gpu fence -> ld A", r[2] <= miss_thr, false, r[2]);
    verdict_bool("ld B; cluster fence -> ld B", r[5] <= miss_thr, false, r[5]);
#if VX_CFG_EXT_DLMEM_ENABLED
    verdict_bool("ld C; cta fence -> ld C", r[8] <= miss_thr, true, r[8]);
#else
    (void)c;
    std::cout << "  (cta fence: DLMEM builds only)" << std::endl;
#endif
    auto r2 = run_ops({{OP_TIMED, d}}, out);
    verdict_bool("ld D; next kernel -> ld D", r2[0] <= miss_thr, false, r2[0]);
    break;
  }
  case 8: {
    // The front end issues one warp instruction per cycle per core, below the
    // L1's line per cycle, so the L1 is judged by serving every lw the front end
    // issues: the lw loop runs as fast as its add twin, and never above a line per cycle.
    kernel_arg_t arg = {};
    arg.start_addr = data_base;
    const double line_rate = (double)LINE;
    arg.n = std::max<uint32_t>(2, (uint32_t)std::ceil(32768.0 * line_rate / (num_warps * 16.0 * lanes * 4)));
    auto span = [&](uint32_t kind) {
      arg.kind = kind;
      launch(arg, (uint32_t)num_warps, out);
      uint64_t t0 = UINT64_MAX, t1 = 0;
      for (uint32_t w = 0; w < num_warps; ++w) {
        t0 = std::min(t0, out[w * 2]);
        t1 = std::max(t1, out[w * 2 + 1]);
      }
      return (double)(t1 - t0);
    };
    double bytes = (double)num_warps * arg.n * 16 * lanes * 4;
    double lw_cycles = span(LS_BW), add_cycles = span(LS_BW_ALU);
    double got = bytes / lw_cycles, issue = std::min(bytes / add_cycles, line_rate);
    printf("  %llu warps, %u LSU blocks, %.0f bytes in %.0f cycles; the add twin in %.0f cycles\n",
           (unsigned long long)num_warps, VX_CFG_NUM_LSU_BLOCKS, bytes, lw_cycles, add_cycles);
    verdict("L1-hit lw B/cycle vs the front end's issue rate", got, issue, 0, 0.05);
    printf("  %-44s %.3f (H100: one line per cycle per SM)\n", "issue rate x line", issue);
    break;
  }
  case 9: {
    // Warp-wide 8-B hits on L lines (lanes l and l + L share one), 16 rows of lines.
    const char* kinds[3] = {"offset 0", "diagonal", "sector j mod sectors"};
    const uint32_t sectors = LINE / VX_CFG_DCACHE_SECTOR_SIZE;
    for (uint32_t spacing : {LINE, 4096u, 65536u}) {
      for (uint32_t L : {32u, 16u, 8u, 4u, 2u, 1u}) {
        if (L > lanes)
          continue;
        for (uint32_t k = 0; k < 3; ++k) {
          if (L == 1 && k > 0)
            continue;
          std::vector<uint32_t> off;
          for (uint32_t i = 0; i < lanes; ++i) {
            uint32_t j = i % L;
            uint32_t in = (k == 0) ? 0 : (k == 1) ? (8 * i) % LINE : (j % sectors) * VX_CFG_DCACHE_SECTOR_SIZE;
            off.push_back(j * spacing + in);
          }
          double got = chase(off, 16, L * spacing, 16, 1024, 0, out);
          verdict("L = " + std::to_string(L) + ", " + kinds[k] + ", lines " + std::to_string(spacing) + " B apart",
                  got, hit_lat(off), 1, 0);
        }
      }
    }
    break;
  }
  case 10: {
    // A full L1 refills through age-list victims: an L2-hit lw stream over more lines than the
    // L1 holds keeps the rate of the same stream over half the L1 after an invalidate (bump slots).
    const uint32_t W = (uint32_t)num_warps, step = W * LS_STREAM_UNROLL;
    const uint32_t l2_lines = (uint32_t)((uint64_t)VX_CFG_L2_SIZE * 3 / 4 / LINE);
    auto stream = [&](uint32_t n, uint32_t fence) {
      kernel_arg_t arg = {};
      arg.kind = LS_STREAM;
      arg.n = n / step * step;
      arg.laps = 2;
      arg.fence = fence;
      arg.warps = W;
      arg.start_addr = data_base;
      launch(arg, W, out);
      uint64_t t0 = UINT64_MAX, t1 = 0;
      for (uint32_t w = 0; w < W; ++w) {
        t0 = std::min(t0, out[w * 2]);
        t1 = std::max(t1, out[w * 2 + 1]);
      }
      return (double)arg.n * LINE / (double)(t1 - t0);
    };
    const uint32_t saved = carve_field;
    for (uint32_t i = 0; i < SMEM_CARVE_NUM_CONFIGS; ++i) {
      const uint32_t c = carve.configs_kb[i];
      if (i > 0 && c == carve.configs_kb[i - 1])
        continue;
      carve_field = 1 + c * 100 / carve.configs_kb[SMEM_CARVE_NUM_CONFIGS - 1];
      if (launch_s(W) != c * 1024) {
        printf("  config %u KiB: the hint picks %u B\n", c, launch_s(W));
        ++errors;
        continue;
      }
      const uint32_t lines = usable_lines(W);
      const uint32_t full_n = std::min(4 * lines, l2_lines);
      const double cold = stream(lines / 2, OP_FENCE_GPU);
      const double full = stream(full_n, 0);
      const std::string what = "config " + std::to_string(c) + " KiB, " + std::to_string(lines) + " lines: full-L1 B/cycle";
      if (lines > 2 * VX_CFG_SM_MAX_READ_LINES && full_n >= 2 * lines) {
        verdict_range(what + " (stream over " + std::to_string(full_n) + " lines) vs cold " + std::to_string(cold).substr(0, 7),
                      full, 0.95 * cold, 1e9, 0);
      } else {
        printf("  %-60s %9.3f, cold %.3f (reported)\n", what.c_str(), full, cold);
      }
    }
    carve_field = saved;
    break;
  }
  }

}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  if (modes.empty()) modes.push_back(0);
  bool wide = false, rows64k = false;
  for (int m : modes) {
    if (m < 0 || m > 10) {
      show_usage();
      return -1;
    }
    wide |= (m == 5);
    rows64k |= (m == 9);
  }
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  uint64_t num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  if (num_threads > LS_LANES) {
    std::cout << "Error: needs at most " << LS_LANES << " threads, device has " << num_threads << std::endl;
    cleanup();
    return -1;
  }
  lanes = (uint32_t)num_threads;
  std::cout << "SM array " << VX_CFG_SM_SRAM_BYTES << " B, L1 tags " << VX_CFG_L1_TAG_SETS << " sets x "
            << VX_CFG_L1_TAG_WAYS << " ways, line " << LINE
            << ", sector " << VX_CFG_DCACHE_SECTOR_SIZE << std::endl;

  carve = smem_carve_cfg();
  // Mode 5 strides up to 64 KiB over every resident line; mode 9 spans 16 rows of 32 lines 64 KiB apart.
  uint64_t data_bytes = wide ? (uint64_t)usable_lines(1) * 65536 + 65536 : 16ull << 20;
  if (rows64k) data_bytes = std::max<uint64_t>(data_bytes, 16ull * 32 * 65536 + 65536);
  RT_CHECK(vx_buffer_create(device, data_bytes, VX_MEM_READ_WRITE, &data_buffer));
  RT_CHECK(vx_buffer_address(data_buffer, &data_buf_addr));
  data_base = (data_buf_addr + 65535) & ~65535ull;
  RT_CHECK(vx_buffer_create(device, 1 << 20, VX_MEM_READ, &tab_buffer));
  RT_CHECK(vx_buffer_address(tab_buffer, &tab_addr));
  std::vector<uint64_t> out(8192);
  RT_CHECK(vx_buffer_create(device, out.size() * 8, VX_MEM_READ_WRITE, &out_buffer));
  RT_CHECK(vx_buffer_address(out_buffer, &out_addr));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  calibrate_miss_thr(out);
  for (int m : modes) {
    run_mode(m, out);
  }

  cleanup();
  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
