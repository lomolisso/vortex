// DRAM microbenchmarks (E17): pointer chase (DRAM miss over an L2 hit), concurrent pairs placed with the golden
// DRAM map, read/write streams in one channel or all, sparse line patterns, and an allocation at the top of global
// memory. Every mode checks its data; the PERF lines feed the configuration sweep.
//
// Usage: dram_bench <chase|pair|stream|sparse|top> [-n count] [-t conflict|samerow|otherbank]
//                   [-w writes per 8] [-c one|all|both] [-p ld64|ld32|st32|st64|st16] [-g gap] [-k kernel]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>
#include <vortex2.h>
#include <VX_types.h>
#include <vx_dram_map.h>
#include "common.h"

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = _expr;                                         \
    if (0 == _ret) break;                                     \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);  \
    cleanup();                                                \
    exit(-1);                                                 \
  } while (false)

#define WRITE_MARK 0x5a5a5a5a5a5a5a5aull

namespace {

constexpr uint32_t log2u(uint64_t x) { return x <= 1 ? 0 : 1 + log2u(x >> 1); }

constexpr uint64_t LINE = VX_CFG_DRAM_LINE_BYTES;
constexpr uint32_t LG_LINE = log2u(LINE);
constexpr uint32_t LG_CH = log2u(VX_CFG_DRAM_CHANNELS);
constexpr uint32_t LG_SEL = log2u(VX_CFG_DRAM_PORT_DATA_SIZE) + VX_CFG_DRAM_INTERLEAVE_SHIFT;
constexpr uint64_t SEL = uint64_t(1) << LG_SEL;

const char* kernel_file = "kernel.vxbin";
std::string mode_name;
uint32_t count = 0;
std::string pair_type = "conflict";
uint32_t writes8 = 0;
std::string place = "all";
std::string pattern = "ld64";
uint32_t gap = 600;

vx_device_h device = nullptr;
vx_queue_h queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
std::vector<vx_buffer_h> buffers;

void cleanup() {
  if (device) {
    for (auto b : buffers)
      vx_buffer_release(b);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
    device = nullptr;
  }
}

void usage() {
  printf("Usage: dram_bench <chase|pair|stream|sparse|top> [-n count] [-t conflict|samerow|otherbank] "
         "[-w writes/8] [-c one|all|both] [-p ld64|ld32|st32|st64|st16] [-g gap] [-k kernel]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:w:c:p:g:k:h")) != -1) {
    switch (c) {
    case 'n': count = atoi(optarg); break;
    case 't': pair_type = optarg; break;
    case 'w': writes8 = atoi(optarg); break;
    case 'c': place = optarg; break;
    case 'p': pattern = optarg; break;
    case 'g': gap = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': usage(); exit(0);
    default: usage(); exit(-1);
    }
  }
  if (optind < argc)
    mode_name = argv[optind];
}

struct Buf {
  vx_buffer_h h = nullptr;
  uint64_t addr = 0;   // aligned device address
  uint64_t base = 0;   // allocation start
  uint64_t size = 0;   // allocation size
};

// A buffer whose start is aligned to `align` (the channel-select period), so a channel-local placement holds.
Buf alloc(uint64_t size, uint64_t align) {
  Buf b;
  b.size = size + align;
  RT_CHECK(vx_buffer_create(device, b.size, VX_MEM_READ_WRITE, &b.h));
  buffers.push_back(b.h);
  RT_CHECK(vx_buffer_address(b.h, &b.base));
  b.addr = (b.base + align - 1) / align * align;
  return b;
}

void upload(const Buf& b, const std::vector<uint64_t>& data) {
  RT_CHECK(vx_enqueue_write(queue, b.h, b.addr - b.base, data.data(), data.size() * 8, 0, nullptr, nullptr));
}

std::vector<uint64_t> download(const Buf& b, uint64_t bytes, vx_event_h after) {
  std::vector<uint64_t> v((bytes + 7) / 8);
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, v.data(), b.h, b.addr - b.base, bytes, after ? 1 : 0, after ? &after : nullptr, &ev));
  RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
  return v;
}

vx_event_h launch(kernel_arg_t& arg, uint32_t grid, uint32_t block, uint32_t lmem) {
  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = kernel;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = grid;
  li.block_dim[0] = block;
  li.lmem_size = lmem;
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &ev));
  return ev;
}

uint32_t pct(std::vector<uint32_t> v, uint32_t p) {
  std::sort(v.begin(), v.end());
  return v[std::min<size_t>(v.size() - 1, v.size() * p / 100)];
}

// Channel-local DRAM line of a device address and its (bank, row, col).
const uint64_t g_hb[] = VX_DRAM_HB_H100;
const vx_dram_geom_t g_geom = {log2u(VX_CFG_DRAM_ROW_BYTES / LINE), log2u(VX_CFG_DRAM_BANKS), g_hb};
uint64_t local_line(uint64_t addr) { return vx_dram_interim_local(addr, LG_SEL, LG_CH) >> LG_LINE; }
vx_dram_loc_t loc_of(uint64_t addr) { return vx_dram_map(local_line(addr), &g_geom); }

int run_chase(kernel_arg_t& arg) {
  uint32_t n = count ? count : 256;
  arg.count = n;
  arg.passes = 2;
  // one node per DRAM line, visited in a random cycle; the span exceeds the L1 and fits the L2
  Buf b = alloc(n * LINE, LINE);
  std::vector<uint32_t> perm(n);
  for (uint32_t i = 0; i < n; ++i) perm[i] = i;
  std::mt19937 rng(7);
  std::shuffle(perm.begin() + 1, perm.end(), rng);
  std::vector<uint64_t> data(n * LINE / 8, 0);
  for (uint32_t i = 0; i < n; ++i)
    data[perm[i] * LINE / 8] = b.addr + perm[(i + 1) % n] * LINE;
  upload(b, data);
  Buf out = alloc(4ull * 2 * n + 8, 64);
  arg.list_addr = b.addr + perm[0] * LINE;
  arg.out_addr = out.addr;
  auto ev = launch(arg, 1, 1, 4 * 2 * n);
  auto res = download(out, 4ull * 2 * n + 8, ev);
  vx_event_release(ev);
  auto lat = reinterpret_cast<const uint32_t*>(res.data());
  std::vector<uint32_t> miss(lat, lat + n), hit(lat + n, lat + 2 * n);
  uint64_t end = res.back();
  printf("PERF: chase dram_miss median=%u p10=%u p90=%u\n", pct(miss, 50), pct(miss, 10), pct(miss, 90));
  printf("PERF: chase l2_hit median=%u p10=%u p90=%u\n", pct(hit, 50), pct(hit, 10), pct(hit, 90));
  printf("PERF: chase offset=%d (DRAM miss over L2 hit, SM cycles)\n", int(pct(miss, 50)) - int(pct(hit, 50)));
  if (end != arg.list_addr) {
    printf("error: chase ended at 0x%lx, expected 0x%lx\n", end, arg.list_addr);
    return 1;
  }
  return 0;
}

enum Rel { CONFLICT, SAMEROW, OTHERBANK };

int run_pair(kernel_arg_t& arg) {
  uint32_t trials = count ? count : 31;
  std::vector<Rel> kinds;
  if (pair_type == "conflict") kinds = {CONFLICT, OTHERBANK};
  else if (pair_type == "samerow") kinds = {SAMEROW, OTHERBANK};
  else if (pair_type == "otherbank") kinds = {OTHERBANK};
  else { usage(); return 1; }
  // 4096 DRAM lines of channel 0 (16 rows of every bank); aligned so line j's row group is j's
  const uint64_t ch_lines = 4096;
  const uint64_t row_lines = uint64_t(1) << (g_geom.lg_lines_per_row + g_geom.lg_banks);
  Buf b = alloc((ch_lines * LINE) << LG_CH, (row_lines * LINE) << LG_CH);
  auto line = [&](uint64_t j) { return b.addr + vx_dram_interim_addr(0, j * LINE, LG_SEL, LG_CH); };
  std::mt19937_64 rng(11);
  std::set<uint64_t> used;
  auto key = [](vx_dram_loc_t l) { return (l.row << 8) | l.bank; };
  auto fresh = [&]() {
    for (;;) {
      uint64_t j = rng() % ch_lines;
      if (used.insert(key(loc_of(line(j)))).second) return j;
    }
  };
  std::vector<uint64_t> list;
  std::vector<Rel> seq;
  for (uint32_t i = 0; i < trials; ++i) {
    for (Rel r : kinds) {
      uint64_t ja = fresh(), a = line(ja), bb = 0;
      auto la = loc_of(a);
      for (int t = 0; t < 1000000 && !bb; ++t) {
        uint64_t jc = r == SAMEROW ? (ja & ~(row_lines - 1)) | (rng() & (row_lines - 1)) : rng() % ch_lines;
        uint64_t c = line(jc);
        if (c == a) continue;
        auto lc = loc_of(c);
        bool ok = r == CONFLICT ? (lc.bank == la.bank && lc.row != la.row && !used.count(key(lc)))
                : r == SAMEROW  ? (lc.bank == la.bank && lc.row == la.row && lc.col != la.col)
                                : (lc.bank != la.bank && !used.count(key(lc)));
        if (ok) bb = c;
      }
      if (!bb) { printf("error: no partner line\n"); return 1; }
      used.insert(key(loc_of(bb)));
      list.push_back(a);
      list.push_back(bb);
      seq.push_back(r);
    }
  }
  std::vector<uint64_t> data((ch_lines * LINE << LG_CH) / 8, 0);
  for (auto a : list) data[(a - b.addr) / 8] = a;
  upload(b, data);
  Buf lb = alloc(list.size() * 8, 64);
  upload(lb, list);
  Buf out = alloc(4ull * (seq.size() + 1) + 16, 64);
  arg.count = seq.size();
  arg.gap = gap;
  arg.list_addr = lb.addr;
  arg.out_addr = out.addr;
  auto ev = launch(arg, 1, 2, 4 * seq.size());
  auto res = download(out, 4ull * (seq.size() + 1) + 16, ev);
  vx_event_release(ev);
  auto lat = reinterpret_cast<const uint32_t*>(res.data());
  std::map<Rel, std::vector<uint32_t>> by;
  uint64_t sums[2] = {0, 0};
  for (size_t i = 0; i < seq.size(); ++i) {
    by[seq[i]].push_back(lat[i]);
    sums[0] += list[2 * i];
    sums[1] += list[2 * i + 1];
  }
  auto sres = reinterpret_cast<const uint64_t*>(lat + seq.size() + (seq.size() & 1));
  const char* names[] = {"conflict", "samerow", "otherbank"};
  for (auto& kv : by)
    printf("PERF: pair %s median=%u p10=%u p90=%u\n", names[kv.first], pct(kv.second, 50), pct(kv.second, 10),
           pct(kv.second, 90));
  int errors = 0;
  if (sres[0] != sums[0] || sres[1] != sums[1]) {
    printf("error: pair data sums 0x%lx 0x%lx, expected 0x%lx 0x%lx\n", sres[0], sres[1], sums[0], sums[1]);
    ++errors;
  }
  if (kinds.size() == 2) {
    int d = int(pct(by[kinds[0]], 50)) - int(pct(by[OTHERBANK], 50));
    printf("PERF: pair %s_minus_otherbank=%d\n", names[kinds[0]], d);
#ifdef VX_CFG_DRAM_MODEL_H100
    // 07 §4: a concurrent same-bank different-row pair costs one row cycle, ~70 SM cycles
    if (kinds[0] == CONFLICT && (d < 60 || d > 80)) {
      printf("error: conflict - otherbank = %d, expected 70 +-10\n", d);
      ++errors;
    }
#endif
  }
  return errors;
}

uint32_t total_threads() {
  uint64_t cores, warps, threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &threads));
  return uint32_t(cores * warps * threads);
}

int run_lines(kernel_arg_t& arg, bool is_stream) {
  uint32_t n = count ? (count + BENCH_ILP - 1) / BENCH_ILP * BENCH_ILP : 256;
  arg.count = n;
  uint32_t req = total_threads();
  uint32_t grid = 0, block = 0;
  RT_CHECK(vx_device_max_occupancy_grid(device, 1, &req, &grid, &block));
  uint64_t T = uint64_t(grid) * block;
  uint64_t lines = uint64_t(n) * T;
  uint32_t phases = 1u << PLACE_ALL;
  if (is_stream) {
    arg.param = writes8;
    phases = place == "one" ? 1u << PLACE_ONE : place == "both" ? 3u : 1u << PLACE_ALL;
  } else {
    const char* pats[] = {"ld64", "ld32", "st32", "st64", "st16"};
    arg.param = 5;
    for (uint32_t i = 0; i < 5; ++i)
      if (pattern == pats[i]) arg.param = i;
    if (arg.param == 5) { usage(); return 1; }
  }
  arg.phases = phases;
  // region of the one-channel placement, then the every-channel one
  uint64_t bytes = ((lines << LG_CH) << LG_SEL) + (lines << LG_SEL);
  Buf b = alloc(bytes, SEL << LG_CH);
  std::vector<uint64_t> data(bytes / 8);
  for (uint64_t k = 0; k < data.size(); ++k)
    data[k] = b.addr + 8 * k;
  upload(b, data);
  Buf out = alloc(8 * T * LINES_OUT_WORDS, 64);
  arg.buf_addr = b.addr;
  arg.out_addr = out.addr;
  auto ev = launch(arg, grid, block, 0);
  auto mem = download(b, bytes, ev);
  auto res = download(out, 8 * T * LINES_OUT_WORDS, nullptr);
  vx_event_release(ev);

  auto addr_of = [&](uint32_t pl, uint64_t j) {
    if (pl == PLACE_ONE) return b.addr + vx_dram_interim_addr(0, j << LG_SEL, LG_SEL, LG_CH);
    return b.addr + ((lines << LG_CH) << LG_SEL) + (j << LG_SEL);
  };
  int errors = 0;
  uint64_t reads = 0, writes = 0;
  for (uint64_t t = 0; t < T; ++t) {
    uint64_t sum = 0;
    for (uint32_t pl = PLACE_ONE; pl <= PLACE_ALL; ++pl) {
      if (!(phases & (1u << pl))) continue;
      for (uint32_t i = 0; i < n; ++i) {
        uint64_t a = addr_of(pl, uint64_t(i) * T + t);
        uint64_t w = (a - b.addr) / 8;
        uint32_t nst = 0;
        if (is_stream) {
          uint32_t slot = i % 8;
          bool wr = writes8 && ((slot * writes8) % 8) < writes8;
          if (wr) nst = 1; else sum += a;
        } else {
          switch (arg.param) {
          case SP_LD64: sum += a + a + 32; break;
          case SP_LD32: sum += a; break;
          case SP_ST16: nst = 2; break;
          case SP_ST32: nst = 4; break;
          default: nst = 8; break;
          }
        }
        if (nst) ++writes; else ++reads;
        for (uint32_t o = 0; o < nst; ++o) {
          if (mem[w + o] != (a ^ WRITE_MARK)) {
            if (errors < 5) printf("error: line 0x%lx word %u = 0x%lx\n", a, o, mem[w + o]);
            ++errors;
          }
        }
      }
    }
    if (res[t * LINES_OUT_WORDS] != sum) {
      if (errors < 5) printf("error: thread %lu read sum 0x%lx, expected 0x%lx\n", t, res[t * LINES_OUT_WORDS], sum);
      ++errors;
    }
  }
  const char* mode = is_stream ? "stream" : "sparse";
  const char* name = is_stream ? place.c_str() : pattern.c_str();
  printf("PERF: %s %s lines=%lu reads=%lu writes=%lu threads=%lu\n", mode, name, reads + writes, reads, writes, T);
  // a phase lasts from its first thread's start to its last thread's end
  const char* pl_names[] = {"one", "all"};
  for (uint32_t pl = PLACE_ONE; pl <= PLACE_ALL; ++pl) {
    if (!(phases & (1u << pl))) continue;
    uint64_t t0 = UINT64_MAX, t1 = 0;
    for (uint64_t t = 0; t < T; ++t) {
      t0 = std::min(t0, res[t * LINES_OUT_WORDS + 1 + 2 * pl]);
      t1 = std::max(t1, res[t * LINES_OUT_WORDS + 2 + 2 * pl]);
    }
    printf("PERF: %s %s phase=%s cycles=%lu lines_per_kcycle=%.1f\n", mode, name, pl_names[pl], t1 - t0,
           1000.0 * lines / double(t1 - t0));
  }
  return errors;
}

int run_top(kernel_arg_t& arg) {
  uint64_t cores, warps, threads, gmem;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_GLOBAL_MEM_SIZE, &gmem));
  const uint64_t size = 1ull << 20;
  uint64_t stacks = cores * warps * threads << VX_MEM_STACK_LOG2_SIZE;
  uint64_t end = uint64_t(VX_MEM_STACK_BASE_ADDR) - stacks;
  uint64_t start = end - size;
  printf("top: global memory 0x%lx, stack base 0x%lx, buffer [0x%lx, 0x%lx)\n", gmem, uint64_t(VX_MEM_STACK_BASE_ADDR),
         start, end);
  int errors = 0;
  if (gmem != uint64_t(VX_CFG_GLOBAL_MEM_SIZE) || start < gmem / 2 || end > uint64_t(VX_MEM_LMEM_BASE_ADDR)) {
    printf("error: the top buffer is not just below the stacks of a 0x%lx-byte memory\n", gmem);
    ++errors;
  }
#if VX_CFG_XLEN == 64
  if (uint64_t(VX_MEM_DLMEM_BASE_ADDR) < gmem) {
    printf("error: the DLMEM window 0x%lx starts inside global memory\n", uint64_t(VX_MEM_DLMEM_BASE_ADDR));
    ++errors;
  }
#endif
  Buf b;
  RT_CHECK(vx_buffer_reserve(device, start, size, VX_MEM_READ_WRITE, &b.h));
  buffers.push_back(b.h);
  b.base = b.addr = start;
  b.size = size;
  uint32_t req = total_threads(), grid = 0, block = 0;
  RT_CHECK(vx_device_max_occupancy_grid(device, 1, &req, &grid, &block));
  uint64_t T = uint64_t(grid) * block;
  Buf out = alloc(8 * T, 64);
  arg.count = uint32_t(size / 8);
  arg.buf_addr = start;
  arg.out_addr = out.addr;
  auto ev = launch(arg, grid, block, 0);
  auto mem = download(b, size, ev);
  auto res = download(out, 8 * T, nullptr);
  vx_event_release(ev);
  for (uint64_t t = 0; t < T; ++t)
    if (res[t]) { printf("error: thread %lu read back %lu wrong words\n", t, res[t]); ++errors; }
  for (uint64_t k = 0; k < size / 8; ++k)
    if (mem[k] != ((start + 8 * k) ^ WRITE_MARK)) {
      if (errors < 5) printf("error: word 0x%lx = 0x%lx\n", start + 8 * k, mem[k]);
      ++errors;
    }
  return errors;
}

}  // namespace

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  kernel_arg_t arg = {};
  arg.lg_channels = LG_CH;
  arg.lg_sel = LG_SEL;
  if (mode_name == "chase") arg.mode = MODE_CHASE;
  else if (mode_name == "pair") arg.mode = MODE_PAIR;
  else if (mode_name == "stream") arg.mode = MODE_STREAM;
  else if (mode_name == "sparse") arg.mode = MODE_SPARSE;
  else if (mode_name == "top") arg.mode = MODE_TOP;
  else { usage(); return -1; }

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = {sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0};
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));
  printf("dram_bench %s: %u channels, select granule %lu B, DRAM line %lu B\n", mode_name.c_str(),
         VX_CFG_DRAM_CHANNELS, SEL, LINE);

  int errors = 0;
  switch (arg.mode) {
  case MODE_CHASE:  errors = run_chase(arg); break;
  case MODE_PAIR:   errors = run_pair(arg); break;
  case MODE_STREAM: errors = run_lines(arg, true); break;
  case MODE_SPARSE: errors = run_lines(arg, false); break;
  default:          errors = run_top(arg); break;
  }
  cleanup();
  if (errors) {
    printf("Found %d errors!\nFAILED!\n", errors);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
