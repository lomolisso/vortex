// lmem_amo_dma: local-memory atomics racing DXA tile writes on the same banks.
//
// Warp 0 of every CTA streams back-to-back DXA copies into region A of local
// memory while the other warps hammer atomics on counters in region B until
// the copies finish. Every counter has a closed form in the iteration counts
// the harts report, and region A must hold the source tile of the last copy.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <vortex.h>
#include <dxa.h>

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

constexpr uint32_t kDescSrc = 0;

const char* kernel_file = "kernel.vxbin";
uint32_t num_copies    = 32;
uint32_t tile_elems    = 1024;
uint32_t num_src_tiles = 3;
uint32_t min_iters     = 16;

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h tile_out_buffer = nullptr;
vx_buffer_h hammer_out_buffer = nullptr;
vx_buffer_h lane_out_buffer = nullptr;
vx_buffer_h overlap_out_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-r copies] [-e tile elems] [-s source tiles]"
            << " [-n min hammer iters] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:r:e:s:n:h")) != -1) {
    switch (c) {
    case 'k': kernel_file   = optarg; break;
    case 'r': num_copies    = (uint32_t)atoi(optarg); break;
    case 'e': tile_elems    = (uint32_t)atoi(optarg); break;
    case 's': num_src_tiles = (uint32_t)atoi(optarg); break;
    case 'n': min_iters     = (uint32_t)atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (src_buffer) {
      vx_buffer_release(src_buffer);
    }
    if (tile_out_buffer) {
      vx_buffer_release(tile_out_buffer);
    }
    if (hammer_out_buffer) {
      vx_buffer_release(hammer_out_buffer);
    }
    if (lane_out_buffer) {
      vx_buffer_release(lane_out_buffer);
    }
    if (overlap_out_buffer) {
      vx_buffer_release(overlap_out_buffer);
    }
    if (kernel) {
      vx_kernel_release(kernel);
    }
    if (module_) {
      vx_module_release(module_);
    }
    if (queue) {
      vx_queue_release(queue);
    }
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static uint32_t src_value(uint32_t idx) {
  uint32_t x = (idx + 1) * 0x85EBCA6Bu;
  return x ^ (x >> 13) ^ 0xA5A5A5A5u;
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  if (tile_elems == 0 || num_src_tiles == 0) {
    std::cout << "Error: tile elems and source tiles must be > 0" << std::endl;
    return -1;
  }

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_STD_A) == 0 || (isa_flags & VX_ISA_EXT_DXA) == 0) {
    std::cout << "Error: needs both the A and the DXA extensions" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  if (num_warps < 2) {
    std::cout << "Error: needs a copy warp and at least one hammer warp" << std::endl;
    cleanup();
    return -1;
  }

  const uint32_t threads    = (uint32_t)num_threads;
  const uint32_t block_size = (uint32_t)(num_warps * num_threads);
  const uint32_t hammers    = block_size - threads;
  const uint32_t num_ctas   = (uint32_t)num_cores;
  const uint32_t tile_bytes = tile_elems * sizeof(uint32_t);
  // Counters, one bank word each: add and swap per hammer, add and xor per
  // lane, and the done flag.
  const uint32_t num_slots  = 2 * hammers + 2 * threads + 1;
  const uint32_t lmem_bytes = tile_bytes + num_slots * (VX_CFG_XLEN / 8);

  std::cout << "ctas: " << num_ctas << ", block: " << block_size
            << ", hammers/cta: " << hammers << std::endl;
  std::cout << "copies: " << num_copies << ", tile: " << tile_bytes
            << "B, source tiles: " << num_src_tiles
            << ", min iters: " << min_iters << std::endl;

  RT_CHECK(vx_check_occupancy(device, block_size, lmem_bytes));

  const uint32_t src_elems = num_src_tiles * tile_elems;
  std::vector<uint32_t> h_src(src_elems);
  for (uint32_t i = 0; i < src_elems; ++i) {
    h_src[i] = src_value(i);
  }

  kernel_arg_t kernel_arg = {};
  kernel_arg.num_copies    = num_copies;
  kernel_arg.tile_elems    = tile_elems;
  kernel_arg.num_src_tiles = num_src_tiles;
  kernel_arg.min_iters     = min_iters;

  uint64_t src_addr = 0;
  const size_t tile_out_bytes    = (size_t)num_ctas * tile_bytes;
  const size_t hammer_out_bytes  = (size_t)num_ctas * hammers * sizeof(hammer_out_t);
  const size_t lane_out_bytes    = (size_t)num_ctas * threads * sizeof(lane_out_t);
  const size_t overlap_out_bytes = (size_t)num_ctas * sizeof(uint32_t);
  RT_CHECK(vx_buffer_create(device, src_elems * sizeof(uint32_t), VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_create(device, tile_out_bytes,    VX_MEM_READ_WRITE, &tile_out_buffer));
  RT_CHECK(vx_buffer_create(device, hammer_out_bytes,  VX_MEM_READ_WRITE, &hammer_out_buffer));
  RT_CHECK(vx_buffer_create(device, lane_out_bytes,    VX_MEM_READ_WRITE, &lane_out_buffer));
  RT_CHECK(vx_buffer_create(device, overlap_out_bytes, VX_MEM_READ_WRITE, &overlap_out_buffer));
  RT_CHECK(vx_buffer_address(src_buffer,         &src_addr));
  RT_CHECK(vx_buffer_address(tile_out_buffer,    &kernel_arg.tile_out_addr));
  RT_CHECK(vx_buffer_address(hammer_out_buffer,  &kernel_arg.hammer_out_addr));
  RT_CHECK(vx_buffer_address(lane_out_buffer,    &kernel_arg.lane_out_addr));
  RT_CHECK(vx_buffer_address(overlap_out_buffer, &kernel_arg.overlap_out_addr));

  // Poison the outputs so anything the kernel never wrote cannot pass; zero
  // is the failing overlap.
  std::vector<uint32_t> h_tile_out(tile_out_bytes / sizeof(uint32_t), 0xDEADBEEF);
  std::vector<hammer_out_t> h_hammer_out((size_t)num_ctas * hammers);
  std::vector<lane_out_t> h_lane_out((size_t)num_ctas * threads);
  std::vector<uint32_t> h_overlap_out(num_ctas, 0);
  std::memset(h_hammer_out.data(), 0xCC, hammer_out_bytes);
  std::memset(h_lane_out.data(), 0xCC, lane_out_bytes);

  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), src_elems * sizeof(uint32_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, tile_out_buffer, 0, h_tile_out.data(), tile_out_bytes, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, hammer_out_buffer, 0, h_hammer_out.data(), hammer_out_bytes, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, lane_out_buffer, 0, h_lane_out.data(), lane_out_bytes, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, overlap_out_buffer, 0, h_overlap_out.data(), overlap_out_bytes, 0, nullptr, nullptr));

  RT_CHECK(vortex::dxa::program_1d(device, kDescSrc, src_addr,
    src_elems, tile_elems, sizeof(uint32_t)));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "start" << std::endl;
  vx_event_h launch_ev = nullptr, rd_ev0 = nullptr, rd_ev1 = nullptr, rd_ev2 = nullptr, rd_ev3 = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = num_ctas;
    li.block_dim[0] = block_size;
    li.lmem_size    = lmem_bytes;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  RT_CHECK(vx_enqueue_read(queue, h_tile_out.data(), tile_out_buffer, 0, tile_out_bytes, 1, &launch_ev, &rd_ev0));
  RT_CHECK(vx_enqueue_read(queue, h_hammer_out.data(), hammer_out_buffer, 0, hammer_out_bytes, 1, &launch_ev, &rd_ev1));
  RT_CHECK(vx_enqueue_read(queue, h_lane_out.data(), lane_out_buffer, 0, lane_out_bytes, 1, &launch_ev, &rd_ev2));
  RT_CHECK(vx_enqueue_read(queue, h_overlap_out.data(), overlap_out_buffer, 0, overlap_out_bytes, 1, &launch_ev, &rd_ev3));
  RT_CHECK(vx_event_wait_value(rd_ev3, 1, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_event_wait_value(rd_ev2, 1, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_event_wait_value(rd_ev1, 1, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_event_wait_value(rd_ev0, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(rd_ev3);
  vx_event_release(rd_ev2);
  vx_event_release(rd_ev1);
  vx_event_release(rd_ev0);
  vx_event_release(launch_ev);

  struct Check {
    const char* fmt;
    uint32_t count;
    std::string first;
  };
  Check tile_check     = {"cta %u tile[%u]: expected=0x%08x actual=0x%08x", 0, {}};
  Check iters_check    = {"cta %u hammer %u iterations: expected>=%u actual=%u", 0, {}};
  Check add_check      = {"cta %u hammer %u add counter: expected=%u actual=%u", 0, {}};
  Check swap_check     = {"cta %u hammer %u swap word: expected=%u actual=%u", 0, {}};
  Check stale_check    = {"cta %u hammer %u stale old values: expected=%u actual=%u", 0, {}};
  Check lane_add_check = {"cta %u lane %u shared add: expected=0x%08x actual=0x%08x", 0, {}};
  Check lane_xor_check = {"cta %u lane %u shared xor: expected=0x%08x actual=0x%08x", 0, {}};
  auto mismatch = [](Check& c, uint32_t cta, uint32_t idx, uint32_t exp, uint32_t act) {
    if (c.count++ == 0) {
      char buf[128];
      std::snprintf(buf, sizeof(buf), c.fmt, cta, idx, exp, act);
      c.first = buf;
    }
  };

  uint64_t total_iters = 0, lost_adds = 0;
  uint32_t min_seen = UINT32_MAX, max_seen = 0;
  uint32_t no_overlap = 0;
  std::string overlaps;

  for (uint32_t cta = 0; cta < num_ctas; ++cta) {
    if (num_copies != 0) {
      const uint32_t last_tile = (num_copies - 1 + cta) % num_src_tiles;
      const uint32_t* exp = h_src.data() + last_tile * tile_elems;
      const uint32_t* act = h_tile_out.data() + (size_t)cta * tile_elems;
      for (uint32_t e = 0; e < tile_elems; ++e) {
        if (act[e] != exp[e]) {
          mismatch(tile_check, cta, e, exp[e], act[e]);
        }
      }
      no_overlap += (h_overlap_out[cta] == 0);
    }
    overlaps += " " + std::to_string(h_overlap_out[cta]);

    std::vector<uint32_t> lane_add(threads, 0), lane_xor(threads, 0);
    for (uint32_t h = 0; h < hammers; ++h) {
      const hammer_out_t& o = h_hammer_out[(size_t)cta * hammers + h];
      if (o.iters < min_iters || o.iters == 0xCCCCCCCCu) {
        mismatch(iters_check, cta, h, min_iters, o.iters);
        continue;
      }
      total_iters += o.iters;
      min_seen = std::min(min_seen, o.iters);
      max_seen = std::max(max_seen, o.iters);
      if (o.add != o.iters) {
        mismatch(add_check, cta, h, o.iters, o.add);
        if (o.add < o.iters) {
          lost_adds += o.iters - o.add;
        }
      }
      if (o.swap != o.iters) {
        mismatch(swap_check, cta, h, o.iters, o.swap);
      }
      if (o.bad_old != 0) {
        mismatch(stale_check, cta, h, 0, o.bad_old);
      }
      const uint32_t lane = h % threads;
      lane_add[lane] += o.iters * (h + 1);
      for (uint32_t i = 0; i < o.iters; ++i) {
        lane_xor[lane] ^= xor_pattern(h, i);
      }
    }

    for (uint32_t l = 0; l < threads; ++l) {
      const lane_out_t& o = h_lane_out[(size_t)cta * threads + l];
      if (o.add != lane_add[l]) {
        mismatch(lane_add_check, cta, l, lane_add[l], o.add);
      }
      if (o.xor_acc != lane_xor[l]) {
        mismatch(lane_xor_check, cta, l, lane_xor[l], o.xor_acc);
      }
    }
  }

  std::cout << "hammer iterations: total=" << total_iters
            << " min=" << min_seen << " max=" << max_seen << std::endl;
  std::cout << "hammer iterations done when the last copy landed, per cta:" << overlaps << std::endl;

  cleanup();

  uint32_t errors = 0;
  for (const Check* c : {&tile_check, &iters_check, &add_check, &swap_check,
                         &stale_check, &lane_add_check, &lane_xor_check}) {
    if (c->count != 0) {
      std::cout << "*** " << c->count << " mismatches, first: " << c->first << std::endl;
      errors += c->count;
    }
  }
  if (no_overlap != 0) {
    std::cout << "*** " << no_overlap << " cta(s) finished every copy before any hammer"
              << " iteration completed: the atomics never raced the DXA writes" << std::endl;
    errors += no_overlap;
  }
  if (errors) {
    std::cout << "Found " << errors << " errors (" << lost_adds
              << " private add updates lost)!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
