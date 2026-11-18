// carveout: the per-launch split of the SM's unified L1 / shared-memory array.
//
// Expectations come from sw/common/smem_carveout.h at the build's knobs. A
// launch asks for configuration c through the hint floor(c * 100 / largest),
// which the hint rule maps back to c; a timed load counts as a miss when it
// takes more than the midpoint of the launch's fastest timed L2 hit and L1 hit
// over the ring's first pass (the kernel measures both).
//
//   -m 0 capacity   per c: a random-order line chase over smem_usable_l1(c) bytes has no second-pass
//                   miss; one step more (2 KiB, 1 KiB at the largest c) misses
//   -m 1 regions    kernel 0 (largest c) fills its shared memory, kernel 1 (c) fills every L1 slot with
//                   tagged lines, kernel 2 (largest c) reads its uninitialised shared memory: the first
//                   L1 row is c + L1_RESERVE_BYTES, the rows below keep kernel 0's data
//   -m 2 switch     launches at 32, 228, 8, 100 KiB back to back: the previous kernel's shared words are
//                   seen at the same index, the first line touched misses, each holds its own capacity;
//                   the gap between two launches at different configs and a same-config pair's are printed
//   -m 3 occupancy  resident blocks per SM (overlapping block lifetimes on one core) == smem_blocks_per_sm
//                   at the no-hint configuration, unclustered and in clusters of 2; a zero-initialised
//                   launch descriptor gets the no-hint count
//   -m 4 default    a launch with no carve-out preference gets the no-hint configuration on both launch
//                   paths: a zero-initialised descriptor and the legacy vx_start (one block of one warp,
//                   no shared memory) each hold that configuration's L1 capacity and miss one step above
//   -m 5 route      launches hinted to each listed c with the same per-block shared size: every SM that
//                   runs blocks holds smem_blocks_per_sm(c) of them at once, so each SM's dispatcher and
//                   the cluster admission saw that launch's DCR value
//   -c a,b,...      configurations (KiB) for modes 0, 1 and 5 (default: the configured list)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include <vortex.h>
#include <smem_carveout.h>

#include "common.h"

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
#error "carveout tests the H100 SM's unified array: build with -DVX_CFG_H100_SM_ENABLE"
#endif

constexpr uint32_t LINE = VX_CFG_DCACHE_LINE_SIZE;
constexpr uint32_t MARK_WORDS_BYTES = CV_MARK_WORDS * 4;
constexpr uint32_t TAG = 0x50000000u;

const char* kernel_file = "kernel.vxbin";
const char* legacy_file = "legacy.vxbin";
std::vector<int> modes;
std::vector<uint32_t> cfg_list;

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_buffer_h data_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
uint64_t data_buf_addr = 0, data_base = 0, out_addr = 0;
uint32_t lanes = 0, num_cores = 0;
uint32_t errors = 0;
SmemCarveCfg carve;

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-m mode[,mode...] 0..5] [-c KiB[,KiB...]] [-h]" << std::endl;
}

static std::vector<uint32_t> parse_list(const std::string& a) {
  std::vector<uint32_t> v;
  size_t pos = 0;
  while (pos <= a.size()) {
    size_t e = a.find(',', pos);
    if (e == std::string::npos) e = a.size();
    v.push_back((uint32_t)atoi(a.substr(pos, e - pos).c_str()));
    pos = e + 1;
  }
  return v;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:m:c:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'm': for (uint32_t m : parse_list(optarg)) modes.push_back((int)m); break;
    case 'c': cfg_list = parse_list(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (data_buffer) vx_buffer_release(data_buffer);
    if (out_buffer) vx_buffer_release(out_buffer);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static void check(const std::string& what, bool ok, const std::string& detail) {
  printf("  %-52s %s %s\n", what.c_str(), detail.c_str(), ok ? "ok" : "MISMATCH");
  if (!ok) ++errors;
}

static uint32_t largest_kb() { return carve.configs_kb[SMEM_CARVE_NUM_CONFIGS - 1]; }

// The launch descriptor's field: 0 = no preference, 1..101 = hint 0..100 %.
static uint32_t hint_field(uint32_t c_kb) { return 1 + c_kb * 100 / largest_kb(); }

struct Launch {
  uint32_t grid = 1, block = 0, lmem = 0, carveout = 0, cluster = 1;
};

static void enqueue(kernel_arg_t& arg, const Launch& l, vx_event_h* launch_ev) {
  arg.out_addr = out_addr;
  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kernel;
  li.args_host      = &arg;
  li.args_size      = sizeof(arg);
  li.ndim           = 1;
  li.grid_dim[0]    = l.grid;
  li.block_dim[0]   = l.block ? l.block : lanes;
  li.lmem_size      = l.lmem;
  li.smem_carveout  = l.carveout;
  if (l.cluster > 1) {
    li.cluster_dim[0] = l.cluster;
    li.cluster_dim[1] = 1;
    li.cluster_dim[2] = 1;
  }
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, launch_ev));
}

static void launch(kernel_arg_t& arg, const Launch& l, void* out, size_t out_bytes) {
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  enqueue(arg, l, &launch_ev);
  RT_CHECK(vx_enqueue_read(queue, out, out_buffer, 0, out_bytes, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
}

static void write_sync(uint64_t dev_addr, const void* src, size_t bytes) {
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_write(queue, data_buffer, dev_addr - data_buf_addr, src, bytes, 0, nullptr, &ev));
  RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
}

// A launch at configuration c: the marker region as its shared memory, the hint that picks c.
static Launch at_config(uint32_t c_kb) {
  Launch l;
  l.lmem = MARK_WORDS_BYTES;
  l.carveout = hint_field(c_kb);
  int32_t s = smem_config_choose(l.lmem, lanes, false, (int32_t)l.carveout - 1, carve);
  if (s != (int32_t)(c_kb * 1024)) {
    printf("Error: hint %u picks %d B, not %u KiB\n", l.carveout - 1, s, c_kb);
    cleanup();
    exit(-1);
  }
  return l;
}

// A ring over n contiguous lines from data_base, visited in a seeded random order.
static void build_ring(uint32_t n) {
  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(n);
  std::shuffle(order.begin() + 1, order.end(), rng);
  std::vector<uint64_t> img((size_t)n * LINE / 8, 0);
  for (uint32_t i = 0; i < n; ++i)
    img[(size_t)order[i] * LINE / 8] = data_base + (uint64_t)order[(i + 1) % n] * LINE;
  write_sync(data_base, img.data(), img.size() * 8);
}

struct ChaseResult {
  uint64_t misses, cycles, first, bad, thr;
};

static kernel_arg_t chase_arg(uint32_t n, uint32_t mark_old, uint32_t mark_new) {
  build_ring(n);
  kernel_arg_t arg = {};
  arg.kind = CV_CHASE;
  arg.n = n;
  arg.mark_old = mark_old;
  arg.mark_new = mark_new;
  arg.start_addr = data_base;
  return arg;
}

static ChaseResult chase_with(const Launch& l, uint32_t n, uint32_t mark_old, uint32_t mark_new) {
  kernel_arg_t arg = chase_arg(n, mark_old, mark_new);
  uint64_t o[5];
  launch(arg, l, o, sizeof(o));
  return {o[0], o[1], o[2], o[3], o[4]};
}

static ChaseResult chase(uint32_t c_kb, uint32_t n, uint32_t mark_old, uint32_t mark_new) {
  return chase_with(at_config(c_kb), n, mark_old, mark_new);
}

static uint32_t l1_lines_at(uint32_t c_kb) { return smem_usable_l1(c_kb * 1024, carve) / LINE; }

static void mode_capacity() {
  for (uint32_t c : cfg_list) {
    uint32_t cap = l1_lines_at(c);
    uint32_t step = ((c == largest_kb()) ? 1024 : 2048) / LINE;
    printf("config %u KiB: L1 %u B (%u lines), H100 %u B\n", c, cap * LINE, cap, (248 - c) * 1024);
    auto at = chase(c, cap, 0, 1);
    check("second pass at capacity: misses", at.misses == 0, std::to_string(at.misses) + " (want 0)");
    auto over = chase(c, cap + step, 0, 1);
    check("second pass at capacity + one step: misses", over.misses > 0,
          std::to_string(over.misses) + " (want > 0), threshold " + std::to_string(over.thr) + ", first " +
            std::to_string(over.first) + ", mean " + std::to_string(over.cycles / (cap + step)));
  }
}

static void mode_regions() {
  const uint32_t big_lmem = largest_kb() * 1024;
  const uint32_t words = big_lmem / 4;
  std::vector<uint32_t> dump(words);
  for (uint32_t c : cfg_list) {
    uint32_t first_l1 = c * 1024 + carve.reserve_bytes;
    if (first_l1 + LINE > big_lmem)
      continue;
    printf("config %u KiB: first L1 row expected at %u B (H100: %u B, after a 4 KiB gap)\n", c, first_l1,
           c * 1024 + 4096);
    const uint32_t P0 = 0xA0000000u;
    kernel_arg_t a0 = {};
    a0.kind = CV_FILL;
    a0.lmem_words = words;
    a0.mark_new = P0;
    Launch big;
    big.lmem = big_lmem;
    big.carveout = hint_field(largest_kb());
    uint64_t sink;
    launch(a0, big, &sink, sizeof(sink));

    uint32_t n = l1_lines_at(c);
    std::vector<uint32_t> img((size_t)n * LINE / 4);
    for (uint32_t L = 0; L < n; ++L)
      for (uint32_t w = 0; w < LINE / 4; ++w)
        img[(size_t)L * LINE / 4 + w] = TAG | (L << 5) | w;
    write_sync(data_base, img.data(), img.size() * 4);
    kernel_arg_t a1 = {};
    a1.kind = CV_TOUCH;
    a1.n = n;
    a1.line = LINE;
    a1.start_addr = data_base;
    launch(a1, at_config(c), &sink, sizeof(sink));

    kernel_arg_t a2 = {};
    a2.kind = CV_DUMP;
    a2.lmem_words = words;
    launch(a2, big, dump.data(), (size_t)words * 4);

    uint32_t first = words;
    for (uint32_t i = 0; i < words; ++i) {
      if (dump[i] != (P0 ^ i)) {
        first = i;
        break;
      }
    }
    check("first row not kernel 0's", first * 4 == first_l1,
          std::to_string(first * 4) + " B (want " + std::to_string(first_l1) + ")");
    uint32_t rows = 0, tagged = 0;
    for (uint32_t r = first_l1 / LINE; (r + 1) * LINE <= big_lmem; ++r, ++rows) {
      const uint32_t* w = &dump[(size_t)r * LINE / 4];
      bool whole = ((w[0] & 0xF000001Fu) == TAG);
      for (uint32_t k = 1; whole && k < LINE / 4; ++k)
        whole = (w[k] == (w[0] | k));
      tagged += whole;
    }
    // A few rows hold kernel 1's own argument and stack lines.
    check("rows above: whole tagged lines of kernel 1", tagged + 16 >= rows,
          std::to_string(tagged) + " of " + std::to_string(rows));
  }
}

// Cycles from the first launch's last instruction (after its stores drained) to the second launch's first,
// both enqueued while the first one spins.
static uint64_t launch_gap(uint32_t c0, uint32_t c1) {
  kernel_arg_t a0 = {}, a1 = {};
  a0.kind = a1.kind = CV_STAMP;
  a0.n = 0;
  a1.n = 1;
  a0.spin = 100000;
  a1.spin = 1000;
  Launch l0, l1;
  l0.carveout = hint_field(c0);
  l1.carveout = hint_field(c1);
  l0.lmem = l1.lmem = sizeof(uint64_t);
  vx_event_h e0 = nullptr, e1 = nullptr, read_ev = nullptr;
  uint64_t o[4] = {};
  enqueue(a0, l0, &e0);
  enqueue(a1, l1, &e1);
  RT_CHECK(vx_enqueue_read(queue, o, out_buffer, 0, sizeof(o), 1, &e1, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(e1);
  vx_event_release(e0);
  return o[2];
}

static void mode_switch() {
  const uint32_t seq[] = {32, 228, 8, 100};
  uint32_t mark = 0;
  for (uint32_t c : seq) {
    bool listed = false;
    for (uint32_t i = 0; i < SMEM_CARVE_NUM_CONFIGS; ++i)
      listed |= (carve.configs_kb[i] == c);
    if (!listed)
      continue;
    uint32_t next = 0x30000000u + c;
    auto r = chase(c, l1_lines_at(c), mark, next);
    printf("config %u KiB (%u lines): first hop %llu cycles, timed pass %llu cycles\n", c, l1_lines_at(c),
           (unsigned long long)r.first, (unsigned long long)r.cycles);
    if (mark)
      check("previous kernel's shared words at the same index", r.bad == 0,
            std::to_string(r.bad) + " mismatches");
    check("first line touched misses", r.first > r.thr, std::to_string(r.first) + " cycles, threshold " + std::to_string(r.thr));
    check("its own capacity: second-pass misses", r.misses == 0, std::to_string(r.misses));
    mark = next;
  }
  // U4a: the gaps are reported, not checked: each launch's CP command fetch, argument and TLS accesses reach
  // DRAM, whose latency follows the bank history, so two pairs differ by hundreds of cycles at one config.
  std::vector<uint32_t> listed;
  for (uint32_t c : seq) {
    for (uint32_t i = 0; i < SMEM_CARVE_NUM_CONFIGS; ++i) {
      if (carve.configs_kb[i] == c) {
        listed.push_back(c);
        break;
      }
    }
  }
  for (size_t i = 0; i + 1 < listed.size(); ++i) {
    const uint32_t a = listed[i], b = listed[i + 1];
    // A warm-up pair first, so both measured pairs follow the same queue and cache history.
    const uint64_t warm = launch_gap(b, b);
    const uint64_t sw = launch_gap(a, b);
    const uint64_t same = launch_gap(b, b);
    printf("  launch gap %u -> %u KiB: %llu cycles, %u -> %u KiB: %llu (warm-up pair %llu)\n", a, b,
           (unsigned long long)sw, b, b, (unsigned long long)same, (unsigned long long)warm);
  }
}

// Most blocks alive at once on each core.
static std::vector<uint32_t> resident_per_core(const std::vector<uint64_t>& o, uint32_t blocks) {
  std::vector<uint32_t> res(num_cores, 0);
  for (uint32_t core = 0; core < num_cores; ++core) {
    uint32_t best = 0;
    std::vector<std::pair<uint64_t, int>> ev;
    for (uint32_t b = 0; b < blocks; ++b) {
      if (o[b * 3] != core) continue;
      ev.push_back({o[b * 3 + 1], 1});
      ev.push_back({o[b * 3 + 2], -1});
    }
    std::sort(ev.begin(), ev.end(), [](auto& a, auto& b) { return a.first < b.first || (a.first == b.first && a.second < b.second); });
    int cur = 0;
    for (auto& e : ev) {
      cur += e.second;
      best = std::max(best, (uint32_t)cur);
    }
    res[core] = best;
  }
  return res;
}

// Resident blocks per core; the period doubles until the expected count is reached (on every core, or on the
// busiest), since blocks whose start-up straddles a period boundary end at different boundaries.
static std::vector<uint32_t> occupancy_run(const Launch& l, uint32_t exp, bool every_core,
                                           std::vector<uint64_t>* last = nullptr, uint32_t* period = nullptr) {
  kernel_arg_t arg = {};
  arg.kind = CV_OCC;
  arg.start_addr = data_base;
  arg.line = LINE;
  std::vector<uint64_t> o((size_t)l.grid * 3);
  std::vector<uint32_t> res;
  for (arg.spin = 4000;; arg.spin *= 2) {
    launch(arg, l, o.data(), o.size() * 8);
    res = resident_per_core(o, l.grid);
    uint32_t r = every_core ? *std::min_element(res.begin(), res.end()) : *std::max_element(res.begin(), res.end());
    if (r >= exp || arg.spin >= 4000u * 256)
      break;
  }
  if (last)
    *last = o;
  if (period)
    *period = arg.spin;
  return res;
}

// A failing count's evidence: each block of core 0 as start / end cycle relative to the first start.
static void print_blocks(const std::vector<uint64_t>& o, uint32_t blocks, uint32_t period) {
  std::vector<std::pair<uint64_t, uint64_t>> v;
  for (uint32_t b = 0; b < blocks; ++b)
    if (o[b * 3] == 0)
      v.push_back({o[b * 3 + 1], o[b * 3 + 2]});
  if (v.empty())
    return;
  std::sort(v.begin(), v.end());
  printf("    period %u; core 0 blocks (start..end - %llu):", period, (unsigned long long)v[0].first);
  for (auto& p : v)
    printf(" %llu..%llu", (unsigned long long)(p.first - v[0].first), (unsigned long long)(p.second - v[0].first));
  printf("\n");
}

static void occupancy_case(uint32_t bt, uint32_t lmem, uint32_t cluster, bool zero_desc) {
  bool cl = cluster > 1;
  uint32_t other = smem_blocks_other(bt, cl, carve);
  if (other == 0 || bt > lanes * carve.warps_per_sm)
    return;
  int32_t s = smem_config_choose(lmem, bt, cl, -1, carve);
  if (s < 0)
    return;
  uint32_t exp = smem_blocks_per_sm((uint32_t)s, lmem, other, carve);
  // On one SM every block of a cluster shares it, so whole clusters only.
  if (num_cores == 1)
    exp = exp / cluster * cluster;
  uint32_t grid = std::min<uint32_t>(4 * exp * num_cores, 1024);
  grid = (grid + cluster - 1) / cluster * cluster;
  Launch l;
  l.grid = grid;
  l.block = bt;
  l.lmem = lmem;
  l.cluster = cluster;
  l.carveout = 0;
  std::vector<uint64_t> o;
  uint32_t period = 0;
  auto res = occupancy_run(l, exp, false, &o, &period);
  uint32_t got = *std::max_element(res.begin(), res.end());
  char what[96];
  snprintf(what, sizeof(what), "%s%u threads, %u B, cluster %u (S %d KiB)", zero_desc ? "zero descriptor: " : "", bt,
           lmem, cluster, s / 1024);
  check(what, got == exp, std::to_string(got) + " resident (want " + std::to_string(exp) + ")");
  if (got != exp)
    print_blocks(o, l.grid, period);
}

static void mode_occupancy() {
  const uint32_t kb256[] = {0, 16, 48, 76, 100, 113};
  const uint32_t kb_small[] = {0, 4, 16};
  for (uint32_t cluster : {1u, 2u}) {
    for (uint32_t k : kb256) occupancy_case(256, k * 1024, cluster, false);
    for (uint32_t bt : {32u, 1024u})
      for (uint32_t k : kb_small) occupancy_case(bt, k * 1024, cluster, false);
  }
  occupancy_case(lanes, 4096, 1, true);
}

// The legacy entry point runs a legacy image: grid = one block per core, block = one warp, no shared memory.
static ChaseResult legacy_chase(uint32_t n) {
  kernel_arg_t arg = chase_arg(n, 0, 0);
  arg.out_addr = out_addr;
  vx_buffer_h krnl = nullptr, args = nullptr;
  RT_CHECK(vx_upload_kernel_file(device, legacy_file, &krnl));
  RT_CHECK(vx_upload_bytes(device, &arg, sizeof(arg), &args));
  RT_CHECK(vx_start(device, krnl, args));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  uint64_t o[5];
  RT_CHECK(vx_copy_from_dev(o, out_buffer, 0, sizeof(o)));
  vx_mem_free(args);
  vx_mem_free(krnl);
  return {o[0], o[1], o[2], o[3], o[4]};
}

static ChaseResult legacy_at, legacy_over;
static uint32_t legacy_cap, legacy_step;

static uint32_t step_lines(uint32_t s) {
  return ((s == largest_kb() * 1024) ? 1024 : 2048) / LINE;
}

// vx_start cannot declare a footprint, so it runs at the largest configuration.
static uint32_t legacy_lines(uint32_t& step) {
  uint32_t s = largest_kb() * 1024;
  step = step_lines(s);
  return smem_usable_l1(s, carve) / LINE;
}

static void mode_default() {
  int32_t s = smem_config_choose(0, lanes, false, -1, carve);
  if (s < 0) {
    check("no-hint configuration of a one-warp block", false, "refused");
    return;
  }
  uint32_t cap = smem_usable_l1((uint32_t)s, carve) / LINE;
  uint32_t step = step_lines((uint32_t)s);
  printf("no hint, one warp, no shared memory: S %d KiB, L1 %u lines\n", s / 1024, cap);
  Launch zero;
  auto at = chase_with(zero, cap, 0, 0);
  check("zero descriptor, at capacity: misses", at.misses == 0, std::to_string(at.misses) + " (want 0)");
  auto over = chase_with(zero, cap + step, 0, 0);
  check("zero descriptor, capacity + one step: misses", over.misses > 0,
        std::to_string(over.misses) + " (want > 0), threshold " + std::to_string(over.thr) + ", first " +
          std::to_string(over.first) + ", mean " + std::to_string(over.cycles / (cap + step)));
  printf("vx_start: S %u KiB, L1 %u lines\n", largest_kb(), legacy_cap);
  check("vx_start, at capacity: misses", legacy_at.misses == 0, std::to_string(legacy_at.misses) + " (want 0)");
  check("vx_start, capacity + one step: misses", legacy_over.misses > 0,
        std::to_string(legacy_over.misses) + " (want > 0), threshold " + std::to_string(legacy_over.thr) + ", first " +
          std::to_string(legacy_over.first) + ", mean " +
          std::to_string(legacy_over.cycles / (legacy_cap + legacy_step)));
}

static void mode_route() {
  const uint32_t other = smem_blocks_other(lanes, false, carve);
  const uint32_t lmem = cfg_list.front() * 1024 / 2;
  for (uint32_t c : cfg_list) {
    Launch l;
    l.lmem = lmem;
    l.carveout = hint_field(c);
    l.block = lanes;
    int32_t s = smem_config_choose(lmem, lanes, false, (int32_t)l.carveout - 1, carve);
    if (s != (int32_t)(c * 1024)) {
      printf("config %u KiB: hint %u picks %d B with %u B per block, skipped\n", c, l.carveout - 1, s, lmem);
      continue;
    }
    uint32_t exp = smem_blocks_per_sm((uint32_t)s, lmem, other, carve);
    l.grid = std::min<uint32_t>(4 * exp * num_cores, 1024);
    auto res = occupancy_run(l, exp, true);
    uint32_t bad = 0;
    for (uint32_t core = 0; core < num_cores; ++core)
      bad += (res[core] != exp);
    char what[96];
    snprintf(what, sizeof(what), "config %u KiB, %u B per block: SMs not at %u resident", c, lmem, exp);
    check(what, bad == 0, std::to_string(bad) + " of " + std::to_string(num_cores) + " (core 0: " +
          std::to_string(res[0]) + ")");
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  if (modes.empty()) modes.push_back(0);
  for (int m : modes) {
    if (m < 0 || m > 5) {
      show_usage();
      return -1;
    }
  }
  carve = smem_carve_cfg();
  if (cfg_list.empty())
    cfg_list.assign(carve.configs_kb, carve.configs_kb + SMEM_CARVE_NUM_CONFIGS);
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  uint64_t v;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &v));
  lanes = (uint32_t)v;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &v));
  num_cores = (uint32_t)v;
  printf("array %u B, reserve %u B, configs", carve.sram_bytes, carve.reserve_bytes);
  for (uint32_t i = 0; i < SMEM_CARVE_NUM_CONFIGS; ++i) printf(" %u", carve.configs_kb[i]);
  printf(" KiB\n");

  // Tag-set aligned so contiguous lines spread over every set.
  const uint64_t align = (uint64_t)VX_CFG_L1_TAG_SETS * LINE;
  uint64_t data_bytes = (uint64_t)carve.sram_bytes + 2 * align;
  RT_CHECK(vx_buffer_create(device, data_bytes, VX_MEM_READ_WRITE, &data_buffer));
  RT_CHECK(vx_buffer_address(data_buffer, &data_buf_addr));
  data_base = (data_buf_addr + align - 1) / align * align;
  RT_CHECK(vx_buffer_create(device, carve.sram_bytes, VX_MEM_READ_WRITE, &out_buffer));
  RT_CHECK(vx_buffer_address(out_buffer, &out_addr));
  // The legacy path uploads the image at its load address, so it runs before the module occupies it.
  if (std::find(modes.begin(), modes.end(), 4) != modes.end()) {
    legacy_cap = legacy_lines(legacy_step);
    legacy_at = legacy_chase(legacy_cap);
    legacy_over = legacy_chase(legacy_cap + legacy_step);
  }
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  for (int m : modes) {
    std::cout << "mode " << m << std::endl;
    switch (m) {
    case 0: mode_capacity(); break;
    case 1: mode_regions(); break;
    case 2: mode_switch(); break;
    case 3: mode_occupancy(); break;
    case 4: mode_default(); break;
    case 5: mode_route(); break;
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
