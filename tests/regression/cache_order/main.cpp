// cache_order: per-word access order in the data cache hierarchy.
//
// Every mode races one hart's (or one warp's) accesses to a word against the
// fill of that word's line, so a younger access reaches the cache while older
// accesses to the same word still wait in the line's miss chain. The cache
// must apply the accesses to a word in the order it accepted them, which on
// one hart is program order. Each mode checks every value by host-side model:
//   raw        store, then a load of the same word        load sees the store
//   war        load, then a store to the same word        load misses the store
//   store_amo  store, then amoadd on the same word        amoadd sees the store
//   store_lr   store, then lr.w on the same word          lr.w sees the store
//   amo_store  amoadd by one warp, store by another       one of the two orders
//   amo_load   amoadd, then a load of the same word       load sees the amoadd
//   amo_storm  every hart: atomics on cold and resident lines, none lost
// The atomic modes need the A extension, amo_store two warps and amo_load
// four threads per warp; without -t a mode the device cannot run is skipped.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <vortex2.h>
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

struct ModeInfo {
  const char* name;
  bool        atomic;
  uint32_t    iters;  // default per-hart iterations (rounds)
};

static const ModeInfo g_modes[NUM_MODES] = {
  {"raw",       false, 256},
  {"war",       false, 256},
  {"store_amo", true,  32},
  {"store_lr",  true,  32},
  {"amo_store", true,  64},
  {"amo_load",  true,  5000},
  {"amo_storm", true,  128},
};

const char* kernel_file = "kernel.vxbin";
uint32_t iters_opt = 0;
uint32_t seed = 1;
std::vector<int> selected;

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
vx_buffer_h buf_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;

uint64_t num_cores = 0, num_warps = 0, num_threads = 0;

static void show_usage() {
  std::cout << "Usage: [-t<mode>: select a mode, repeatable] [-n<iters per hart>] [-s<seed>] [-k<kernel>] [-h]"
            << std::endl << "modes:";
  for (auto& m : g_modes) {
    std::cout << " " << m.name;
  }
  std::cout << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "t:n:s:k:h")) != -1) {
    switch (c) {
    case 't': {
      int found = -1;
      for (int m = 0; m < NUM_MODES; ++m) {
        if (strcmp(optarg, g_modes[m].name) == 0) {
          found = m;
        }
      }
      if (found < 0) {
        std::cout << "unknown mode: " << optarg << std::endl;
        show_usage();
        exit(-1);
      }
      selected.push_back(found);
      break;
    }
    case 'n': iters_opt = (uint32_t)atoi(optarg); break;
    case 's': seed = (uint32_t)atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

static void release_buffers() {
  if (buf_buffer) {
    vx_buffer_release(buf_buffer);
    buf_buffer = nullptr;
  }
  if (out_buffer) {
    vx_buffer_release(out_buffer);
    out_buffer = nullptr;
  }
}

void cleanup() {
  if (device) {
    release_buffers();
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
    device = nullptr;
  }
}

// Allocates the two buffers, writes their initial images, runs the kernel on
// the given grid and reads both images back.
static void launch(kernel_arg_t& arg, std::vector<uint32_t>& buf, uint64_t buf_align,
                   std::vector<uint32_t>& out, uint32_t grid, uint32_t block, uint64_t* skew) {
  size_t buf_bytes = buf.size() * sizeof(uint32_t);
  size_t out_bytes = out.size() * sizeof(uint32_t);
  RT_CHECK(vx_buffer_create(device, buf_bytes + buf_align, VX_MEM_READ_WRITE, &buf_buffer));
  uint64_t raw = 0;
  RT_CHECK(vx_buffer_address(buf_buffer, &raw));
  arg.buf_addr = (raw + buf_align - 1) & ~(buf_align - 1);
  *skew = arg.buf_addr - raw;
  RT_CHECK(vx_buffer_create(device, out_bytes, VX_MEM_READ_WRITE, &out_buffer));
  RT_CHECK(vx_buffer_address(out_buffer, &arg.out_addr));
  RT_CHECK(vx_enqueue_write(queue, buf_buffer, *skew, buf.data(), buf_bytes, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, out_buffer, 0, out.data(), out_bytes, 0, nullptr, nullptr));

  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = grid;
  li.block_dim[0] = block;
  vx_event_h launch_ev = nullptr, rd0 = nullptr, rd1 = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_enqueue_read(queue, out.data(), out_buffer, 0, out_bytes, 1, &launch_ev, &rd0));
  RT_CHECK(vx_enqueue_read(queue, buf.data(), buf_buffer, *skew, buf_bytes, 1, &rd0, &rd1));
  RT_CHECK(vx_event_wait_value(rd1, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(rd1);
  vx_event_release(rd0);
  vx_event_release(launch_ev);
}

static void occupancy_grid(uint32_t harts, uint32_t* grid, uint32_t* block) {
  uint32_t n = harts;
  RT_CHECK(vx_device_max_occupancy_grid(device, 1, &n, grid, block));
}

static int report(const char* name, int checked, int errors) {
  std::cout << name << ": " << checked << " checks, " << errors << " errors" << std::endl;
  return errors;
}

// raw/war: every hart, every line: the load's value and the line's final image.
static int run_probes(int mode, uint32_t iters) {
  bool war = (mode == MODE_WAR);
  uint32_t harts = (uint32_t)(num_cores * num_warps * num_threads);
  uint32_t lines = iters * harts;
  const uint32_t wpl = PROBE_LINE / 4;
  std::vector<uint32_t> buf((size_t)lines * wpl), out(lines, 0xffffffffu);
  for (uint32_t ln = 0; ln < lines; ++ln) {
    for (uint32_t w = 0; w < wpl; ++w) {
      buf[(size_t)ln * wpl + w] = (w == 0) ? war_init(ln) : (0x0C000000u | (ln << 4) | w);
    }
  }
  kernel_arg_t arg = {};
  arg.mode = mode;
  arg.iters = iters;
  arg.num_harts = harts;
  uint32_t grid, block;
  occupancy_grid(harts, &grid, &block);
  uint64_t skew;
  launch(arg, buf, PROBE_LINE, out, grid, block, &skew);

  int errors = 0, checked = 0;
  for (uint32_t h = 0; h < harts; ++h) {
    for (uint32_t i = 0; i < iters; ++i) {
      uint32_t ln = probe_line(h, i, harts);
      uint32_t v = probe_variant(h, i);
      uint32_t want = war ? war_init(ln) : raw_value(ln);
      uint32_t fin0 = war ? war_marker(ln) : raw_value(ln);
      uint32_t got = out[ln];
      uint32_t w0 = buf[(size_t)ln * wpl];
      uint32_t w4 = buf[(size_t)ln * wpl + 4];
      uint32_t want_w4 = war ? war_side(ln) : (0x0C000000u | (ln << 4) | 4);
      ++checked;
      if (got != want || w0 != fin0 || w4 != want_w4) {
        if (++errors <= 16) {
          const char* what = (got == want) ? "final image"
                           : war ? ((got == war_marker(ln)) ? "load returned the younger store" : "load mismatch")
                                 : ((got == war_init(ln)) ? "load missed the older store" : "load mismatch");
          printf("%s: hart %u iter %u variant %u line 0x%llx: %s: load 0x%08x (want 0x%08x), word0 0x%08x "
                 "(want 0x%08x), word4 0x%08x (want 0x%08x)\n", g_modes[mode].name, h, i, v,
                 (unsigned long long)(arg.buf_addr + (uint64_t)ln * PROBE_LINE), what, got, want, w0, fin0,
                 w4, want_w4);
        }
      }
    }
  }
  release_buffers();
  return report(g_modes[mode].name, checked, errors);
}

static int run_store_amo(int mode, uint32_t iters) {
  bool lr = (mode == MODE_STORE_LR);
  const uint32_t lb = VX_CFG_DCACHE_LINE_SIZE;
  const uint32_t dw = VX_CFG_DCACHE_WORD_SIZE;
  // one dcache word of the line per thread: more threads would share words
  const uint32_t nt = std::min<uint32_t>((uint32_t)num_threads, lb / dw);
  std::vector<uint32_t> buf((size_t)iters * lb / 4), out((size_t)iters * nt + nt, 0xdeadbeefu);
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = 0x11000000u + (uint32_t)i;
  }
  kernel_arg_t arg = {};
  arg.mode = mode;
  arg.iters = iters;
  uint64_t skew;
  launch(arg, buf, lb, out, 1, nt, &skew);

  int errors = 0, checked = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    for (uint32_t t = 0; t < nt; ++t) {
      uint32_t s_off = (t * dw) % lb;
      uint32_t stored = STORE_AMO_BASE + i * 4 + t;
      uint32_t fin = buf[((size_t)i * lb + s_off) / 4];
      uint32_t got = out[(size_t)i * nt + t];
      uint32_t want_fin = lr ? stored : stored + 1;
      ++checked;
      if (got != stored || fin != want_fin) {
        if (++errors <= 16) {
          printf("%s: iter %u thread %u: %s returned 0x%08x (want 0x%08x), final word 0x%08x (want 0x%08x)\n",
                 g_modes[mode].name, i, t, lr ? "lr.w" : "amoadd.w", got, stored, fin, want_fin);
        }
      }
    }
  }
  release_buffers();
  return report(g_modes[mode].name, checked, errors);
}

static int run_amo_store(uint32_t iters) {
  std::vector<uint32_t> buf((size_t)iters * SLOT_STRIDE_WORDS, 0), out(iters, 0);
  for (uint32_t i = 0; i < iters; ++i) {
    buf[(size_t)i * SLOT_STRIDE_WORDS] = slot_init(i);
  }
  kernel_arg_t arg = {};
  arg.mode = MODE_AMO_STORE;
  arg.iters = iters;
  uint64_t skew;
  launch(arg, buf, 64, out, 1, 2 * (uint32_t)num_threads, &skew);

  int errors = 0, amo_first = 0, store_first = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    uint32_t init = slot_init(i), inc = slot_inc(i), val = slot_val(i);
    uint32_t fin = buf[(size_t)i * SLOT_STRIDE_WORDS];
    uint32_t r = out[i];
    if (r == init && fin == val) {
      ++amo_first;
    } else if (r == val && fin == val + inc) {
      ++store_first;
    } else if (++errors <= 16) {
      printf("amo_store: slot %u: amoadd returned 0x%08x, final 0x%08x: neither order explains it "
             "(initial 0x%08x, store 0x%08x, add 0x%x)\n", i, r, fin, init, val, inc);
    }
  }
  std::cout << "amo_store: atomic first " << amo_first << ", store first " << store_first << std::endl;
  release_buffers();
  return report("amo_store", (int)iters, errors);
}

static int run_amo_load(uint32_t iters) {
  const uint32_t pool_lines = 8192;
  std::vector<uint32_t> buf((size_t)pool_lines * 64 / 4, 0), out(AMO_LOAD_LANES * AMO_LOAD_RES, 0xeeeeeeeeu);
  kernel_arg_t arg = {};
  arg.mode = MODE_AMO_LOAD;
  arg.iters = iters;
  arg.seed = seed;
  arg.pool_lines = pool_lines;
  uint64_t skew;
  launch(arg, buf, 64, out, 1, AMO_LOAD_LANES, &skew);

  int errors = 0;
  for (uint32_t l = 0; l < AMO_LOAD_LANES; ++l) {
    const uint32_t* r = &out[l * AMO_LOAD_RES];
    if (r[0] != 0) {
      errors += (int)r[0];
      printf("amo_load: lane %u: %u load(s) missed the lane's own atomic; first at round %u: amoadd old=%u, "
             "then lw=%u (want %u)\n", l, r[0], r[1], r[2], r[3], r[2] + 1);
    }
  }
  release_buffers();
  return report("amo_load", (int)(iters * AMO_LOAD_LANES), errors);
}

static int run_amo_storm(uint32_t iters) {
  uint32_t harts = (uint32_t)(num_cores * num_warps * num_threads);
  const uint32_t wpl = STORM_LINE / 4;
  size_t lines = (size_t)harts + (size_t)iters * harts;
  std::vector<uint32_t> buf(lines * wpl, 0), out(harts, 0xffffffffu);
  kernel_arg_t arg = {};
  arg.mode = MODE_AMO_STORM;
  arg.iters = iters;
  arg.num_harts = harts;
  arg.seed = seed;
  uint32_t grid, block;
  occupancy_grid(harts, &grid, &block);
  uint64_t skew;
  launch(arg, buf, STORM_LINE, out, grid, block, &skew);

  int errors = 0, checked = 0;
  for (uint32_t h = 0; h < harts; ++h) {
    uint32_t z = buf[(size_t)h * wpl];
    checked += 2;
    if (out[h] != 0 || z != iters) {
      ++errors;
      printf("amo_storm: hart %u: %u wrong atomic results, resident counter %u (want %u)\n", h, out[h], z, iters);
    }
    for (uint32_t r = 0; r < iters; ++r) {
      uint32_t b = buf[((size_t)harts + (size_t)r * harts + h) * wpl];
      ++checked;
      if (b != 1 && ++errors <= 16) {
        printf("amo_storm: hart %u round %u: cold-line counter %u (want 1)\n", h, r, b);
      }
    }
  }
  release_buffers();
  return report("amo_storm", checked, errors);
}

// The launch shape a mode needs that the device lacks, or "".
static std::string missing_shape(int mode) {
  if (mode == MODE_AMO_STORE && num_warps < 2) {
    return "two warps";
  }
  if (mode == MODE_AMO_LOAD && num_threads < AMO_LOAD_LANES) {
    return std::to_string(AMO_LOAD_LANES) + " threads per warp";
  }
  return "";
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  uint64_t isa_flags = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  bool has_a = (isa_flags & VX_ISA_STD_A) != 0;

  bool explicit_modes = !selected.empty();
  if (!explicit_modes) {
    for (int m = 0; m < NUM_MODES; ++m) {
      if (g_modes[m].atomic && !has_a) {
        continue;
      }
      std::string need = missing_shape(m);
      if (!need.empty()) {
        std::cout << g_modes[m].name << ": skipped, needs " << need << std::endl;
        continue;
      }
      selected.push_back(m);
    }
  }

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  int errors = 0;
  for (int m : selected) {
    uint32_t iters = iters_opt ? iters_opt : g_modes[m].iters;
    if (g_modes[m].atomic && !has_a) {
      std::cout << g_modes[m].name << ": the device has no A extension" << std::endl;
      ++errors;
      continue;
    }
    std::string need = missing_shape(m);
    if (!need.empty()) {
      std::cout << g_modes[m].name << ": needs " << need << std::endl;
      ++errors;
      continue;
    }
    std::cout << "mode " << g_modes[m].name << ": " << iters << " iterations" << std::endl;
    switch (m) {
    case MODE_RAW:
    case MODE_WAR:       errors += run_probes(m, iters); break;
    case MODE_STORE_AMO:
    case MODE_STORE_LR:  errors += run_store_amo(m, iters); break;
    case MODE_AMO_STORE: errors += run_amo_store(iters); break;
    case MODE_AMO_LOAD:  errors += run_amo_load(iters); break;
    default:             errors += run_amo_storm(iters); break;
    }
  }

  cleanup();
  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
