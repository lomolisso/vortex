// One warp with every lane on its own line of one L1 tag set (sw/common/l1_slot_ref.h's l1_set at the
// build's hash), so a warp access touches more lines of the set than the set has ways and the lines it
// holds itself (joined misses, pinned hits) can fill every way: a ld ring and a ld/sd ring must complete
// with the right pointers and stored words.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <vortex.h>
#include <l1_slot_ref.h>

#include "common.h"

#ifndef VX_CFG_H100_SM_ENABLE
#error "l1_setover tests the H100 SM path: build with -DVX_CFG_H100_SM_ENABLE"
#endif

namespace sr = vortex::l1_slot_ref;

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

constexpr uint32_t LINE = VX_CFG_DCACHE_LINE_SIZE;
constexpr uint64_t DATA_BYTES = 16ull << 20;

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_buffer_h data_buffer = nullptr;
vx_buffer_h tab_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
uint64_t data_addr = 0, tab_addr = 0, out_addr = 0;
uint32_t lanes = 0, errors = 0;

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

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    default:
      std::cout << "Usage: [-k kernel] [-h]" << std::endl;
      exit(c == 'h' ? 0 : -1);
    }
  }
}

static void wait_release(vx_event_h ev) {
  RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
}

static void write_sync(vx_buffer_h buf, uint64_t off, const void* src, size_t bytes) {
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_write(queue, buf, off, src, bytes, 0, nullptr, &ev));
  wait_release(ev);
}

static void read_sync(void* dst, vx_buffer_h buf, uint64_t off, size_t bytes) {
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, dst, buf, off, bytes, 0, nullptr, &ev));
  wait_release(ev);
}

static uint64_t run(uint32_t kind, uint32_t hops) {
  kernel_arg_t arg = {};
  arg.kind = kind;
  arg.lanes = lanes;
  arg.hops = hops;
  arg.start_addr = tab_addr;
  arg.out_addr = out_addr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = lanes;
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &ev));
  wait_release(ev);
  uint64_t last = 0;
  read_sync(&last, out_buffer, 0, sizeof(last));
  return last;
}

static void check(const char* what, bool ok, uint64_t got, uint64_t exp) {
  printf("  %-44s 0x%llx, expected 0x%llx %s\n", what, (unsigned long long)got, (unsigned long long)exp,
         ok ? "ok" : "MISMATCH");
  if (!ok) ++errors;
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  uint64_t num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  if (num_threads > SO_LANES) {
    std::cout << "Error: needs at most " << SO_LANES << " threads, device has " << num_threads << std::endl;
    cleanup();
    return -1;
  }
  lanes = (uint32_t)num_threads;
  RT_CHECK(vx_buffer_create(device, DATA_BYTES, VX_MEM_READ_WRITE, &data_buffer));
  RT_CHECK(vx_buffer_address(data_buffer, &data_addr));
  RT_CHECK(vx_buffer_create(device, SO_LANES * 8, VX_MEM_READ, &tab_buffer));
  RT_CHECK(vx_buffer_address(tab_buffer, &tab_addr));
  RT_CHECK(vx_buffer_create(device, 64, VX_MEM_READ_WRITE, &out_buffer));
  RT_CHECK(vx_buffer_address(out_buffer, &out_addr));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  const uint64_t l0 = (data_addr + LINE - 1) / LINE, lend = (data_addr + DATA_BYTES) / LINE;
  const uint32_t set = sr::l1_set(l0, VX_CFG_L1_TAG_SETS, VX_CFG_L1_INDEX_HASH);
  std::vector<uint64_t> ls;
  for (uint64_t l = l0; l < lend && ls.size() < lanes; ++l) {
    if (sr::l1_set(l, VX_CFG_L1_TAG_SETS, VX_CFG_L1_INDEX_HASH) == set)
      ls.push_back(l);
  }
  if (ls.size() < lanes) {
    std::cout << "Error: only " << ls.size() << " lines of set " << set << " in the buffer" << std::endl;
    cleanup();
    return -1;
  }
  const uint32_t n = lanes;
  printf("%u lanes on %u lines of tag set %u (%u sets x %u ways, hash %u)\n", n, n, set,
         (uint32_t)VX_CFG_L1_TAG_SETS, (uint32_t)VX_CFG_L1_TAG_WAYS, (uint32_t)VX_CFG_L1_INDEX_HASH);
  std::vector<uint64_t> img(LINE / 8), starts(SO_LANES, 0);
  for (uint32_t i = 0; i < n; ++i) {
    img.assign(LINE / 8, 0);
    img[0] = ls[(i + 1) % n] * LINE;
    img[1] = ~0ull;
    write_sync(data_buffer, ls[i] * LINE - data_addr, img.data(), LINE);
    starts[i] = ls[i] * LINE;
  }
  write_sync(tab_buffer, 0, starts.data(), starts.size() * 8);

  const uint32_t hops = 3 * n + 1;
  const uint64_t exp_end = ls[hops % n] * LINE;
  uint64_t got = run(SO_CHASE, hops);
  check("ld ring: lane 0's last pointer", got == exp_end, got, exp_end);
  got = run(SO_MIX, hops);
  check("ld/sd ring: lane 0's last pointer", got == exp_end, got, exp_end);
  uint32_t bad = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint64_t w[2] = {0, 0};
    read_sync(w, data_buffer, ls[i] * LINE - data_addr, sizeof(w));
    if (w[0] != ls[(i + 1) % n] * LINE || w[1] != hops - 1)
      ++bad;
  }
  check("ld/sd ring: lines with a wrong word", bad == 0, bad, 0);

  cleanup();
  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
