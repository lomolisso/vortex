// sm_hol: what a store burst costs the accesses issued behind it in the same
// LSU block. Warp 0 of rank 0 times a DSMEM load from rank 1 and a shared
// load, each alone and each right behind a burst of full-line global stores
// (-n lines, 128 B each). Correctness only is checked; the cycles are printed
// for comparison across builds (H100: >= ~26 KiB of stores observed queued
// ahead of a read, doc 18 S1/R1 [I, low]).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <vortex.h>

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

const char* kernel_file = "kernel.vxbin";
uint32_t lines = 256;

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_buffer_h gmem_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:n:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'n': lines = (uint32_t)atoi(optarg); break;
    default:
      std::cout << "Usage: [-k kernel] [-n lines, a multiple of 8] [-h]" << std::endl;
      exit(c == 'h' ? 0 : -1);
    }
  }
}

void cleanup() {
  if (device) {
    if (gmem_buffer) vx_buffer_release(gmem_buffer);
    if (out_buffer) vx_buffer_release(out_buffer);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  if (lines == 0 || (lines % 8) != 0) {
    std::cout << "Error: -n must be a positive multiple of 8" << std::endl;
    return -1;
  }
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  uint64_t num_threads = 0, max_cluster = 0, num_cores = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  if (num_threads != 32 || max_cluster < 2 || num_cores < 2) {
    std::cout << "Error: needs 32 threads and a cluster of 2 on 2 cores" << std::endl;
    cleanup();
    return -1;
  }
  RT_CHECK(vx_buffer_create(device, (uint64_t)lines * 128 + 4096, VX_MEM_READ_WRITE, &gmem_buffer));
  RT_CHECK(vx_buffer_create(device, 8 * 8, VX_MEM_READ_WRITE, &out_buffer));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  kernel_arg_t arg = {};
  arg.lines = lines;
  RT_CHECK(vx_buffer_address(gmem_buffer, &arg.gmem_addr));
  arg.gmem_addr = (arg.gmem_addr + 4095) & ~4095ull;
  RT_CHECK(vx_buffer_address(out_buffer, &arg.out_addr));
  std::vector<uint64_t> out(8, ~0ull);
  RT_CHECK(vx_enqueue_write(queue, out_buffer, 0, out.data(), 64, 0, nullptr, nullptr));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kernel;
  li.args_host      = &arg;
  li.args_size      = sizeof(arg);
  li.ndim           = 1;
  li.grid_dim[0]    = 2;
  li.grid_dim[1]    = 1;
  li.grid_dim[2]    = 1;
  li.block_dim[0]   = (uint32_t)num_threads;
  li.block_dim[1]   = 1;
  li.block_dim[2]   = 1;
  li.lmem_size      = 4096;
  li.cluster_dim[0] = 2;
  li.cluster_dim[1] = 1;
  li.cluster_dim[2] = 1;
  li.cluster_policy = VX_CLUSTER_POLICY_SPREAD;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_enqueue_read(queue, out.data(), out_buffer, 0, 64, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  printf("burst %u lines (%u KiB), port drain %u flits\n", lines, lines * 128 / 1024, lines * 4);
  printf("HOL dsmem_ld_alone=%llu shared_ld_alone=%llu\n", (unsigned long long)out[0], (unsigned long long)out[1]);
  printf("HOL burst_issue=%llu dsmem_ld_behind=%llu\n", (unsigned long long)out[2], (unsigned long long)out[3]);
  printf("HOL burst_issue=%llu shared_ld_behind=%llu burst_to_drain=%llu\n", (unsigned long long)out[4],
         (unsigned long long)out[5], (unsigned long long)out[6]);
  bool ok = true;
  for (int i = 0; i < 7; ++i)
    ok &= (out[i] != ~0ull);
  cleanup();
  if (!ok) {
    std::cout << "Error: the kernel did not record every time\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
