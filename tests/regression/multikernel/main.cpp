// Multi-entry .vxbin test — host side, C-runtime-prologue edition.
//
// Loads a single .vxbin holding three independent kernels, resolves each
// by name via vx_module_get_kernel, and launches all three. Each kernel
// depends on state set up by the __vx_cta_entry prologue:
//
//   add_k / mul_k  read g_cubes[], a table filled by an .init_array
//                  constructor (__libc_init_array).
//   acc_k          sums g_cubes[] through a .tbss thread_local accumulator
//                  and folds in a .tdata thread_local tag (__init_tls +
//                  per-hart tp stride).
//
// So this exercises three things at once and pins a failure to one:
//   - the multi-entry mechanism — VXSYMTAB footer parse, per-name
//     resolution, vx_enqueue_launch's per-kernel PC;
//   - __libc_init_array — if it never runs, g_cubes[] stays zero and
//     add_k/mul_k produce src+0 / src*0;
//   - __init_tls — if tp is unset or not per-thread, acc_k collides.

#include <vortex2.h>
#include "common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

#define FAIL(msg) do {                                                \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
    std::exit(1);                                                     \
  } while (0)

namespace {
const char* kernel_file  = "kernel.vxbin";
uint32_t    size         = 256;
bool        cluster_mode = false;

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:ch")) != -1) {
    switch (c) {
      case 'n': size         = (uint32_t)std::atoi(optarg); break;
      case 'k': kernel_file  = optarg;                      break;
      case 'c': cluster_mode = true;                        break;
      default:
        std::printf("Usage: [-k kernel] [-n size] [-c: cluster ranks across launches] [-h]\n");
        std::exit(c == 'h' ? 0 : -1);
    }
  }
}

// Mirror of the device-side g_cubes[] table: g_cubes[idx] = (idx+1)^3.
int32_t cube(uint32_t idx) {
  int32_t k = (int32_t)((idx & 15) + 1);
  return k * k * k;
}

// Launches rank_k over `num_ctas` CTAs in clusters of `k` and returns the
// mismatches between the slot-derived cluster rank and the block-index one.
int check_cluster_ranks(vx_queue_h q, vx_kernel_h kernel, vx_buffer_h buf,
                        uint32_t num_ctas, uint32_t k, uint32_t block,
                        const char* tag) {
  const uint64_t bytes = (uint64_t)num_ctas * RANK_REC_WORDS * sizeof(uint32_t);
  // A CTA that never ran leaves the fill, which matches no field it writes.
  std::vector<uint32_t> recs(num_ctas * RANK_REC_WORDS, 0xffffffffu);
  CHECK(vx_enqueue_write(q, buf, 0, recs.data(), bytes, 0, nullptr, nullptr));

  rank_arg_t arg = {};
  arg.cluster_k = k;
  CHECK(vx_buffer_address(buf, &arg.dst_addr));

  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kernel;
  li.args_host      = &arg;
  li.args_size      = sizeof(arg);
  li.ndim           = 1;
  li.grid_dim[0]    = num_ctas;
  li.block_dim[0]   = block;
  li.cluster_dim[0] = k;

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, recs.data(), buf, 0, bytes, 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t b = 0; b < num_ctas; ++b) {
    const uint32_t* rec = &recs[b * RANK_REC_WORDS];
    if (rec[RANK_REC_CLUSTER_RANK] == rec[RANK_REC_INDEX]
     && rec[RANK_REC_INDEX] == b % k
     && rec[RANK_REC_CLUSTER_SIZE] == k)
      continue;
    if (errors < 8)
      std::printf("*** %s cta %u: slot %u, cluster rank %u, index %u, cluster size %u;"
                  " expected rank %u of %u\n",
                  tag, b, rec[RANK_REC_CTA_ID], rec[RANK_REC_CLUSTER_RANK],
                  rec[RANK_REC_INDEX], rec[RANK_REC_CLUSTER_SIZE], b % k, k);
    ++errors;
  }
  return errors;
}

// -c: two launches in one process. The first, one CTA without clusters,
// moves the dispatcher's round-robin slot pointer to 1. Software takes a CTA's
// cluster rank from its slot (CTA_ID % K), so each launch must place its first
// cluster from slot 0 again: a window opening at slot 1 hands rank 1 to the
// cluster's first CTA.
int run_cluster_ranks(vx_device_h dev, vx_queue_h q) {
  const uint32_t k = 2;
  uint64_t num_cores = 0, num_warps = 0, num_threads = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,   &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &num_warps));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &num_threads));
  // With fewer slots the window from slot 1 overruns and wraps to slot 0.
  if (num_warps < k + 1) {
    std::printf("Error: cluster ranks need %u warps per core, the device has %u\n",
                k + 1, (uint32_t)num_warps);
    return 1;
  }
  // Two rounds of clusters per core, so the core that ran the first launch
  // takes a cluster whatever the distribution.
  const uint32_t num_ctas = 2 * (uint32_t)(num_cores * num_warps);
  std::printf("multikernel: cluster ranks, 1 CTA then %u CTAs in clusters of %u\n",
              num_ctas, k);

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, "rank_k", &kernel));
  vx_buffer_h buf = nullptr;
  CHECK(vx_buffer_create(dev, (uint64_t)num_ctas * RANK_REC_WORDS * sizeof(uint32_t),
                         VX_MEM_READ_WRITE, &buf));

  int errors = check_cluster_ranks(q, kernel, buf, 1, 1, (uint32_t)num_threads,
                                   "launch 1");
  errors += check_cluster_ranks(q, kernel, buf, num_ctas, k, (uint32_t)num_threads,
                                "launch 2");

  vx_buffer_release(buf);
  vx_kernel_release(kernel);
  vx_module_release(mod);
  return errors;
}
} // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);
  const uint64_t bytes = (uint64_t)size * sizeof(int32_t);
  if (!cluster_mode)
    std::printf("multikernel: size=%u\n", size);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));

  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  if (cluster_mode) {
    int errors = run_cluster_ranks(dev, q);
    vx_queue_release(q);
    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);
    if (errors) {
      std::printf("Found %d errors!\nFAILED!\n", errors);
      return 1;
    }
    std::printf("PASSED!\n");
    return 0;
  }

  vx_buffer_h src_buf = nullptr;
  vx_buffer_h add_buf = nullptr, mul_buf = nullptr, acc_buf = nullptr;
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ,  &src_buf));
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_WRITE, &add_buf));
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_WRITE, &mul_buf));
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_WRITE, &acc_buf));

  // Load the multi-entry module and resolve all three kernels by name.
  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h k_add = nullptr, k_mul = nullptr, k_acc = nullptr;
  CHECK(vx_module_get_kernel(mod, "add_k", &k_add));
  CHECK(vx_module_get_kernel(mod, "mul_k", &k_mul));
  CHECK(vx_module_get_kernel(mod, "acc_k", &k_acc));
  if (k_add == k_mul || k_add == k_acc || k_mul == k_acc)
    FAIL("named kernels resolved to the same handle");

  // A name not in the footer must NOT resolve — proves real footer
  // lookup, not a single-'main' fallback that would accept anything.
  vx_kernel_h k_bogus = nullptr;
  if (vx_module_get_kernel(mod, "no_such_kernel", &k_bogus) == VX_SUCCESS)
    FAIL("a kernel name absent from the module resolved");

  std::vector<int32_t> h_src(size);
  std::vector<int32_t> h_add(size, 0), h_mul(size, 0), h_acc(size, 0);
  for (uint32_t i = 0; i < size; ++i)
    h_src[i] = (int32_t)i;

  uint32_t grid[1] = {1}, block[1] = {1};
  const uint32_t global_dim[1] = { size };
  CHECK(vx_device_max_occupancy_grid(dev, 1, global_dim, grid, block));

  kernel_arg_t arg_add = {}, arg_mul = {}, arg_acc = {};
  arg_add.count = size;
  arg_mul.count = size;
  arg_acc.count = size;
  CHECK(vx_buffer_address(src_buf, &arg_add.src_addr));
  CHECK(vx_buffer_address(add_buf, &arg_add.dst_addr));
  CHECK(vx_buffer_address(src_buf, &arg_mul.src_addr));
  CHECK(vx_buffer_address(mul_buf, &arg_mul.dst_addr));
  CHECK(vx_buffer_address(src_buf, &arg_acc.src_addr));
  CHECK(vx_buffer_address(acc_buf, &arg_acc.dst_addr));

  CHECK(vx_enqueue_write(q, src_buf, 0, h_src.data(), bytes,
                         0, nullptr, nullptr));

  // One launch per named kernel, all from the same module. The
  // single-worker queue keeps them in submit order.
  vx_launch_info_t li_add = {}, li_mul = {}, li_acc = {};
  li_add.struct_size = sizeof(li_add);
  li_add.kernel      = k_add;
  li_add.args_host   = &arg_add;
  li_add.args_size   = sizeof(arg_add);
  li_add.ndim        = 1;
  li_add.grid_dim[0] = grid[0];
  li_add.block_dim[0]= block[0];
  li_mul = li_add;
  li_mul.kernel    = k_mul;
  li_mul.args_host = &arg_mul;
  li_acc = li_add;
  li_acc.kernel    = k_acc;
  li_acc.args_host = &arg_acc;

  vx_event_h add_ev = nullptr, mul_ev = nullptr, acc_ev = nullptr;
  vx_event_h r_add  = nullptr, r_mul  = nullptr, r_acc  = nullptr;
  CHECK(vx_enqueue_launch(q, &li_add, 0, nullptr, &add_ev));
  CHECK(vx_enqueue_launch(q, &li_mul, 0, nullptr, &mul_ev));
  CHECK(vx_enqueue_launch(q, &li_acc, 0, nullptr, &acc_ev));
  CHECK(vx_enqueue_read(q, h_add.data(), add_buf, 0, bytes,
                        1, &add_ev, &r_add));
  CHECK(vx_enqueue_read(q, h_mul.data(), mul_buf, 0, bytes,
                        1, &mul_ev, &r_mul));
  CHECK(vx_enqueue_read(q, h_acc.data(), acc_buf, 0, bytes,
                        1, &acc_ev, &r_acc));
  CHECK(vx_event_wait_value(r_add, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(r_mul, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(r_acc, 1, VX_TIMEOUT_INFINITE));

  int errors = 0;
  for (uint32_t i = 0; i < size; ++i) {
    int32_t want_add = h_src[i] + cube(i);   // add_k
    int32_t want_mul = h_src[i] * cube(i);   // mul_k
    int32_t sum = 0;                         // acc_k
    for (uint32_t j = 0; j < 8; ++j)
      sum += cube(i + j);
    int32_t want_acc = h_src[i] + sum + 0x5A5A;   // 0x5A5A = t_tag

    if (h_add[i] != want_add) {
      if (errors < 8)
        std::printf("*** add_k[%u]: got %d, expected %d\n",
                    i, h_add[i], want_add);
      ++errors;
    }
    if (h_mul[i] != want_mul) {
      if (errors < 8)
        std::printf("*** mul_k[%u]: got %d, expected %d\n",
                    i, h_mul[i], want_mul);
      ++errors;
    }
    if (h_acc[i] != want_acc) {
      if (errors < 8)
        std::printf("*** acc_k[%u]: got %d, expected %d\n",
                    i, h_acc[i], want_acc);
      ++errors;
    }
  }

  vx_event_release(r_acc);
  vx_event_release(r_mul);
  vx_event_release(r_add);
  vx_event_release(acc_ev);
  vx_event_release(mul_ev);
  vx_event_release(add_ev);
  vx_kernel_release(k_acc);
  vx_kernel_release(k_mul);
  vx_kernel_release(k_add);
  vx_module_release(mod);
  vx_buffer_release(acc_buf);
  vx_buffer_release(mul_buf);
  vx_buffer_release(add_buf);
  vx_buffer_release(src_buf);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::printf("Found %d errors!\nFAILED!\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
