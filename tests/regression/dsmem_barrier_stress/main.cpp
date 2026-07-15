// Host driver for dsmem_barrier_stress — F2 cluster-barrier ordering + suspend.
// Verifies each phase's read observes exactly this phase's write from `writer`
// across many barriers, multiple warps, and multiple co-resident clusters.

#include <cstdlib>
#include <cstdint>
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
static uint32_t    cluster_size = 4;   // K
static uint32_t    num_clusters = 2;   // G
static uint32_t    phases       = 8;   // N
static uint32_t    warps_per_cta = 1;

static vx_device_h  device      = nullptr;
static vx_buffer_h  dst_buffer  = nullptr;
static vx_queue_h   queue       = nullptr;
static vx_module_h  module_     = nullptr;
static vx_kernel_h  kernel      = nullptr;
static kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-n K] [-g num_clusters] [-p phases] [-w warps] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:g:p:w:k:h")) != -1) {
    switch (c) {
      case 'n': cluster_size = atoi(optarg); break;
      case 'g': num_clusters = atoi(optarg); break;
      case 'p': phases = atoi(optarg); break;
      case 'w': warps_per_cta = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static uint64_t rd(uint32_t cls, uint32_t csr, uint32_t core) {
  uint64_t v = 0;
  if (vx_mpm_query(device, cls, csr, core, &v) != 0) return 0;
  return v;
}

static int dump_counters(const char* what) {
  uint64_t mcycle  = rd(VX_DCR_MPM_CLASS_CORE, VX_CSR_MCYCLE, 0);
  uint64_t load_lt = rd(VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_LOAD_LT, 0xffffffff);
  uint64_t st_lsu  = rd(VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALL_LSU, 0xffffffff);
  uint64_t stl_wp  = rd(VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_STALLED_WARPS, 0xffffffff);
  uint64_t sidle   = rd(VX_DCR_MPM_CLASS_CORE, VX_CSR_MPM_SCHED_IDLE, 0xffffffff);
  uint64_t lm_rd   = rd(VX_DCR_MPM_CLASS_MEM,  VX_CSR_MPM_LMEM_READS, 0xffffffff);
  uint64_t lm_wr   = rd(VX_DCR_MPM_CLASS_MEM,  VX_CSR_MPM_LMEM_WRITES, 0xffffffff);
  printf("COUNTERS mcycle=%lu load_lt=%lu stall_lsu=%lu stalled_warps=%lu sched_idle=%lu lmem_reads=%lu lmem_writes=%lu\n",
         (unsigned long)mcycle, (unsigned long)load_lt, (unsigned long)st_lsu,
         (unsigned long)stl_wp, (unsigned long)sidle, (unsigned long)lm_rd, (unsigned long)lm_wr);
  int e = 0;
  if (mcycle > 0) { // profiling active — warps must have suspended on the barriers
    if (stl_wp == 0) { printf("*** counter error (%s): stalled_warps==0 (no barrier suspends)\n", what); ++e; }
  }
  return e;
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores = 0, num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  if (num_cores < cluster_size) {
    std::cout << "Device does not have enough cores (need " << cluster_size << ")\n";
    cleanup();
    return -1;
  }

  const uint32_t buf_elems  = warps_per_cta * (uint32_t)num_threads;
  const uint32_t local_mem  = buf_elems * sizeof(uint32_t);
  const uint32_t total_ctas = num_clusters * cluster_size;
  const uint32_t dst_words  = total_ctas * phases * buf_elems;
  const uint32_t dst_bytes  = dst_words * sizeof(uint32_t);

  std::cout << "dsmem_barrier_stress (F2 barrier ordering)\n";
  std::cout << "  K=" << cluster_size << ", G=" << num_clusters << ", phases=" << phases
            << ", warps/CTA=" << warps_per_cta << ", cores=" << num_cores << "\n";

  RT_CHECK(vx_check_occupancy(device, buf_elems, local_mem));

  RT_CHECK(vx_buffer_create(device, dst_bytes, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));
  kernel_arg.cluster_size = cluster_size;
  kernel_arg.buf_elems    = buf_elems;
  kernel_arg.phases       = phases;

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::vector<uint32_t> h_dst(dst_words, 0xffffffffu);

  std::cout << "start\n";
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = total_ctas;
    li.block_dim[0] = buf_elems;
    li.lmem_size    = local_mem;
    li.cluster_dim[0] = cluster_size;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, dst_bytes, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t c = 0; c < num_clusters; ++c) {
    for (uint32_t r = 0; r < cluster_size; ++r) {
      uint32_t writer = (r + cluster_size - 1) % cluster_size;
      for (uint32_t p = 0; p < phases; ++p) {
        for (uint32_t t = 0; t < buf_elems; ++t) {
          uint32_t exp = DSMEM_BS_PATTERN(p, writer, t);
          uint32_t got = h_dst[((c * cluster_size + r) * phases + p) * buf_elems + t];
          if (got != exp) {
            if (errors < 20)
              printf("*** error: cluster=%u rank=%u phase=%u t=%u expected=%u actual=%u\n", c, r, p, t, exp, got);
            ++errors;
          }
        }
      }
    }
  }

  errors += dump_counters("barrier_stress");

  cleanup();
  if (errors) {
    std::cout << "FAILED (" << errors << " errors)\n";
    return errors;
  }
  std::cout << "PASSED\n";
  return 0;
}
