// Host driver for dxa_multicast — cross-core TMA multicast (Task 3c).
//
// Verifies the three faithfulness criteria:
//  (a) every receiver rank's scratchpad holds the correct tile bytes;
//  (b) every receiver's barrier fired (the kernel only reaches its dst store
//      past arrive_and_wait on its own slice);
//  (c) THE LOAD-BEARING CHECK: the GMEM read traffic for the N-way multicast
//      is ~1x the tile, not Nx — byte-correctness alone must not pass.

#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>
#include <dxa.h>

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
static uint32_t    tile_rows   = 2;
static uint32_t    tile_cols   = 32;
static uint32_t    src_rows    = 2;
static uint32_t    src_cols    = 32;

static vx_device_h device     = nullptr;
static vx_buffer_h src_buffer = nullptr;
static vx_buffer_h dst_buffer = nullptr;
static vx_queue_h  queue      = nullptr;
static vx_module_h module_    = nullptr;
static vx_kernel_h kernel     = nullptr;
static kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-r src_rows] [-c src_cols] "
               "[-R tile_rows] [-C tile_cols] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:R:C:k:h")) != -1) {
    switch (c) {
      case 'r': src_rows  = atoi(optarg); break;
      case 'c': src_cols  = atoi(optarg); break;
      case 'R': tile_rows = atoi(optarg); break;
      case 'C': tile_cols = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (src_buffer) vx_buffer_release(src_buffer);
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores = 0, num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  if (num_cores < 2) {
    std::cout << "Device does not have enough cores for a cross-core multicast\n";
    cleanup();
    return -1;
  }
  if (tile_cols != (uint32_t)num_threads) {
    std::cout << "Error: tile_cols (" << tile_cols
              << ") must equal VX_CFG_NUM_THREADS (" << num_threads
              << ") for single-warp CTAs\n";
    cleanup();
    return -1;
  }

  const uint32_t num_recv   = (uint32_t)num_cores;  // one rank per core
  const uint32_t tile_elems = tile_rows * tile_cols;
  const uint32_t local_mem  = tile_elems * sizeof(TYPE);
  const uint32_t src_elems  = src_rows * src_cols;
  const uint32_t src_bytes  = src_elems * sizeof(TYPE);
  const uint32_t dst_bytes  = num_recv * tile_elems * sizeof(TYPE);

  std::cout << "dxa_multicast (cross-core TMA multicast)\n";
  std::cout << "  source: " << src_rows << " x " << src_cols
            << ", tile: " << tile_rows << " x " << tile_cols << "\n";
  std::cout << "  num_recv=" << num_recv << " (one rank per core), local_mem="
            << local_mem << " bytes\n";

  RT_CHECK(vx_check_occupancy(device, tile_cols, local_mem));

  RT_CHECK(vx_buffer_create(device, src_bytes, VX_MEM_READ,  &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_buffer_create(device, dst_bytes, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  std::vector<TYPE> h_src(src_elems);
  for (uint32_t i = 0; i < src_elems; ++i)
    h_src[i] = static_cast<TYPE>(i + 1);
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), src_bytes, 0, nullptr, nullptr));

  constexpr uint32_t kDescSrc = 0;
  RT_CHECK(vortex::dxa::program_2d(device, kDescSrc, kernel_arg.src_addr,
    /*size0=*/src_cols, /*size1=*/src_rows,
    /*stride0_bytes=*/src_cols * sizeof(TYPE),
    /*tile0=*/tile_cols, /*tile1=*/tile_rows,
    /*elem_bytes=*/sizeof(TYPE)));
  // Receiver slices resolve through the CTA residency table in DSMEM mode;
  // the stride field is set for descriptor completeness only.
  RT_CHECK(vortex::dxa::set_multicast(device, kDescSrc, local_mem));

  kernel_arg.tile_rows      = tile_rows;
  kernel_arg.tile_cols      = tile_cols;
  kernel_arg.src_row_stride = src_cols;
  kernel_arg.num_recv       = num_recv;

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::vector<TYPE> h_dst(num_recv * tile_elems, 0);

  std::cout << "start\n";
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = num_recv;   // one cluster spans the whole grid
    li.block_dim[0] = tile_cols;
    li.lmem_size    = local_mem;
    li.cluster_dim[0] = num_recv; // gang: one rank per core (DSMEM placement)
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, dst_bytes, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;

  // (a)+(b) byte-correctness on every receiver (reached only past each
  // receiver's own completion barrier).
  for (uint32_t rcv = 0; rcv < num_recv; ++rcv) {
    for (uint32_t r = 0; r < tile_rows; ++r) {
      for (uint32_t c = 0; c < tile_cols; ++c) {
        TYPE expected = h_src[r * src_cols + c];
        TYPE actual   = h_dst[rcv * tile_elems + r * tile_cols + c];
        if (expected != actual) {
          if (errors < 20)
            printf("*** error: recv=%u [r=%u,c=%u] expected=%f actual=%f\n",
                   rcv, r, c, (float)expected, (float)actual);
          ++errors;
        }
      }
    }
  }

  // (c) THE LOAD-BEARING CHECK — one global read, not N. Sum the per-core
  // DXA gmem_reads counters; the whole grid must have read the tile from
  // GMEM approximately once (cache-line granularity), not num_recv times.
  {
    uint64_t total_gmem_reads = 0, total_lmem_writes = 0;
    for (uint32_t c = 0; c < num_recv; ++c) {
      uint64_t v = 0;
      RT_CHECK(vx_mpm_query(device, VX_DCR_MPM_CLASS_DXA, VX_CSR_MPM_DXA_GMEM_READS, c, &v));
      total_gmem_reads += v;
      RT_CHECK(vx_mpm_query(device, VX_DCR_MPM_CLASS_DXA, VX_CSR_MPM_DXA_LMEM_WRITES, c, &v));
      total_lmem_writes += v;
    }
    const uint64_t tile_lines = (uint64_t(tile_elems) * sizeof(TYPE)
                                 + 63) / 64; // GMEM cache-line requests for 1x tile
    std::cout << "dxa: gmem_reads=" << total_gmem_reads
              << " (1x tile = " << tile_lines << " lines)"
              << ", lmem_writes=" << total_lmem_writes
              << " (fan-out = " << num_recv << "x)\n";
    if (total_gmem_reads == 0 || total_gmem_reads >= 2 * tile_lines) {
      printf("*** error: multicast GMEM traffic is not ~1x the tile "
             "(reads=%llu, tile=%llu lines) — the single-global-read "
             "property was lost\n",
             (unsigned long long)total_gmem_reads,
             (unsigned long long)tile_lines);
      ++errors;
    }
    if (total_lmem_writes < num_recv * tile_lines) {
      printf("*** error: expected ~%ux fan-out of SMEM writes (got %llu)\n",
             num_recv, (unsigned long long)total_lmem_writes);
      ++errors;
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
