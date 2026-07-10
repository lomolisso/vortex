// Host driver for dsmem_pingpong — cross-core DSMEM addressing (Task 3b).
//
// Launches one CTA cluster of num_ranks single-warp CTAs; the DSMEM-mode
// dispatcher places one rank per core. Verifies (A) remote loads observe the
// peer's published pattern and (B) the cluster barrier drains remote stores
// (each rank's local read sees what its peer wrote — no stale data).

#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>

#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret) break;                                       \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

static const char* kernel_file = "kernel.vxbin";
static uint32_t    num_ranks   = 2;
static uint32_t    warps_per_cta = 1;

static vx_device_h  device      = nullptr;
static vx_buffer_h  dstA_buffer = nullptr;
static vx_buffer_h  dstB_buffer = nullptr;
static vx_buffer_h  dstC_buffer = nullptr;
static vx_queue_h   queue       = nullptr;
static vx_module_h  module_     = nullptr;
static vx_kernel_h  kernel      = nullptr;
static kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-n num_ranks] [-w warps_per_cta] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:w:k:h")) != -1) {
    switch (c) {
      case 'n': num_ranks = atoi(optarg); break;
      case 'w': warps_per_cta = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (dstA_buffer) vx_buffer_release(dstA_buffer);
    if (dstB_buffer) vx_buffer_release(dstB_buffer);
    if (dstC_buffer) vx_buffer_release(dstC_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores = 0, num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  if (num_cores < num_ranks) {
    std::cout << "Device does not have enough cores (need " << num_ranks
              << " for one rank per core)\n";
    cleanup();
    return -1;
  }

  const uint32_t buf_elems = warps_per_cta * (uint32_t)num_threads;
  const uint32_t local_mem = 2 * buf_elems * sizeof(uint32_t); // phases A+B
  const uint32_t dst_bytes = num_ranks * buf_elems * sizeof(uint32_t);

  std::cout << "dsmem_pingpong (cross-core DSMEM)\n";
  std::cout << "  num_ranks=" << num_ranks << ", warps_per_cta=" << warps_per_cta
            << ", buf_elems=" << buf_elems
            << ", local_mem=" << local_mem << " bytes\n";

  RT_CHECK(vx_check_occupancy(device, buf_elems, local_mem));

  RT_CHECK(vx_buffer_create(device, dst_bytes, VX_MEM_WRITE, &dstA_buffer));
  RT_CHECK(vx_buffer_address(dstA_buffer, &kernel_arg.dstA_addr));
  RT_CHECK(vx_buffer_create(device, dst_bytes, VX_MEM_WRITE, &dstB_buffer));
  RT_CHECK(vx_buffer_address(dstB_buffer, &kernel_arg.dstB_addr));
  RT_CHECK(vx_buffer_create(device, dst_bytes, VX_MEM_WRITE, &dstC_buffer));
  RT_CHECK(vx_buffer_address(dstC_buffer, &kernel_arg.dstC_addr));

  kernel_arg.num_ranks = num_ranks;
  kernel_arg.buf_elems = buf_elems;

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::vector<uint32_t> h_dstA(num_ranks * buf_elems, 0);
  std::vector<uint32_t> h_dstB(num_ranks * buf_elems, 0);
  std::vector<uint32_t> h_dstC(num_ranks * buf_elems, 0);

  std::cout << "start\n";
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = num_ranks;   // one cluster spans the whole grid
    li.block_dim[0] = buf_elems;
    li.lmem_size    = local_mem;
    li.cluster_dim[0] = num_ranks; // gang: one rank per core (DSMEM placement)
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  vx_event_h readA_ev = nullptr, readB_ev = nullptr, readC_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_dstA.data(), dstA_buffer, 0, dst_bytes, 1, &launch_ev, &readA_ev));
  RT_CHECK(vx_enqueue_read(queue, h_dstB.data(), dstB_buffer, 0, dst_bytes, 1, &readA_ev, &readB_ev));
  RT_CHECK(vx_enqueue_read(queue, h_dstC.data(), dstC_buffer, 0, dst_bytes, 1, &readB_ev, &readC_ev));
  RT_CHECK(vx_event_wait_value(readC_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(readC_ev);
  vx_event_release(readB_ev);
  vx_event_release(readA_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t r = 0; r < num_ranks; ++r) {
    uint32_t peer = (r + 1) % num_ranks;
    for (uint32_t t = 0; t < buf_elems; ++t) {
      uint32_t expA = (peer + 1) * 1000 + t;  // remote load of peer's pattern
      uint32_t gotA = h_dstA[r * buf_elems + t];
      if (gotA != expA) {
        if (errors < 20)
          printf("*** error A: rank=%u t=%u expected=%u actual=%u\n", r, t, expA, gotA);
        ++errors;
      }
      // Rank r's slice was remote-written by the rank whose peer is r.
      uint32_t writer = (r + num_ranks - 1) % num_ranks;
      uint32_t expB = (writer + 1) * 7777 + t;
      uint32_t gotB = h_dstB[r * buf_elems + t];
      if (gotB != expB) {
        if (errors < 20)
          printf("*** error B (drain): rank=%u t=%u expected=%u actual=%u\n", r, t, expB, gotB);
        ++errors;
      }
      // Phase C: plain lw of the peer's A pattern; the 0xDEAD0000 high bits
      // flag a failed region-0 self-map identity/roundtrip.
      uint32_t expC = expA;
      uint32_t gotC = h_dstC[r * buf_elems + t];
      if (gotC != expC) {
        if (errors < 20)
          printf("*** error C (plain-lw/self-map): rank=%u t=%u expected=%u actual=%u\n",
                 r, t, expC, gotC);
        ++errors;
      }
    }
  }

  cleanup();
  if (errors) {
    std::cout << "FAILED (" << errors << " errors)\n";
    return errors;
  }
  std::cout << "PASSED\n";
  return 0;
}
