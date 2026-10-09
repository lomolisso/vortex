// fence: FENCE ordering and cost.
//
//   -t vecadd (default): every thread adds its words, then fences.
//   -t storm: the first warp of the device times -n iterations of a load and
//             an AMO while the other cores' warps (on one core, the timer's
//             siblings) fence in a loop until they see the timer's done flag.
//             The timer waits up to -b/8 cycles for every fencer to start,
//             and must finish within -b cycles. A fencer gives up after -b
//             cycles, so a timer starved by the fences fails the run instead
//             of hanging it, and so does a done flag that a fence never makes
//             visible.

#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex2.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";
uint32_t count = 0;
uint32_t mode = FENCE_MODE_VECADD;
uint32_t budget = 20000;

vx_device_h device = nullptr;
vx_buffer_h src0_buffer = nullptr;
vx_buffer_h src1_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h ctrl_buffer = nullptr;
vx_buffer_h rec_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-t vecadd|storm] [-n words (vecadd) or timed iterations (storm)]"
             << " [-b storm fencer budget in cycles, default 20000] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:t:b:h")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 't':
      if (strcmp(optarg, "vecadd") == 0) {
        mode = FENCE_MODE_VECADD;
      } else if (strcmp(optarg, "storm") == 0) {
        mode = FENCE_MODE_STORM;
      } else {
        show_usage();
        exit(-1);
      }
      break;
    case 'b':
      budget = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (src0_buffer) vx_buffer_release(src0_buffer);
    if (src1_buffer) vx_buffer_release(src1_buffer);
    if (dst_buffer)  vx_buffer_release(dst_buffer);
    if (ctrl_buffer) vx_buffer_release(ctrl_buffer);
    if (rec_buffer)  vx_buffer_release(rec_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

// Launches one block per core over total_threads and waits for the read of
// `out` that is chained after it.
static void launch_and_read(uint32_t total_threads, void* out, vx_buffer_h out_buffer,
                            uint64_t out_size) {
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  uint32_t n = total_threads;
  uint32_t grid_dim[1], block_dim[1];
  RT_CHECK(vx_device_max_occupancy_grid(device, 1, &n, grid_dim, block_dim));
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &kernel_arg;
  li.args_size    = sizeof(kernel_arg);
  li.ndim         = 1;
  li.grid_dim[0]  = grid_dim[0];
  li.block_dim[0] = block_dim[0];
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));

  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_enqueue_read(queue, out, out_buffer, 0, out_size, 1, &launch_ev, &read_ev));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
}

static int run_vecadd(uint32_t total_threads) {
  if (count == 0) {
    count = 1;
  }

  uint32_t num_points = count * total_threads;
  uint32_t buf_size   = num_points * sizeof(int32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  kernel_arg.mode      = FENCE_MODE_VECADD;
  kernel_arg.num_tasks = total_threads;
  kernel_arg.task_size = count;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src0_buffer));
  RT_CHECK(vx_buffer_address(src0_buffer, &kernel_arg.src0_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src1_buffer));
  RT_CHECK(vx_buffer_address(src1_buffer, &kernel_arg.src1_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  std::cout << "dev_src0=0x" << std::hex << kernel_arg.src0_addr << std::endl;
  std::cout << "dev_src1=0x" << std::hex << kernel_arg.src1_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::dec << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<int32_t> h_src0(num_points);
  std::vector<int32_t> h_src1(num_points);
  std::vector<int32_t> h_dst(num_points);

  // generate source data
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = i-1;
    h_src1[i] = i+1;
  }

  // upload source buffer0
  std::cout << "upload source buffer0" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, src0_buffer, 0, h_src0.data(), buf_size, 0, nullptr, nullptr));

  // upload source buffer1
  std::cout << "upload source buffer1" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, src1_buffer, 0, h_src1.data(), buf_size, 0, nullptr, nullptr));

  std::cout << "launch kernel" << std::endl;
  launch_and_read(total_threads, h_dst.data(), dst_buffer, buf_size);

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    int ref = i + i;
    int cur = h_dst[i];
    if (cur != ref) {
      std::cout << "error at result #" << std::dec << i
                << std::hex << ": actual 0x" << cur << ", expected 0x" << ref << std::dec << std::endl;
      ++errors;
    }
  }
  return errors;
}

static int run_storm(uint32_t num_cores, uint32_t num_warps, uint32_t num_threads) {
  uint64_t isa_flags = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_STD_A) == 0) {
    std::cout << "Error: storm needs the A extension" << std::endl;
    return 1;
  }
  if (num_threads > STORM_ITER_WORDS) {
    std::cout << "Error: storm supports at most " << STORM_ITER_WORDS << " threads per warp" << std::endl;
    return 1;
  }

  // One block per core; on several cores the rest of the timer's block idles.
  const uint32_t iters = count ? count : 16;
  const uint32_t total_warps = num_cores * num_warps;
  const uint32_t total_threads = total_warps * num_threads;
  const uint32_t idle = (num_cores > 1) ? num_warps - 1 : 0;
  const uint32_t fencers = total_warps - 1 - idle;

  std::cout << "storm: " << total_warps << " warps (1 timer, " << fencers << " fencers, " << idle
            << " idle), " << iters << " timed iterations, fencer budget " << budget << " cycles"
            << std::endl;

  kernel_arg.mode          = FENCE_MODE_STORM;
  kernel_arg.storm_iters   = iters;
  kernel_arg.storm_fencers = fencers;
  kernel_arg.storm_budget  = budget;

  std::vector<uint32_t> h_data(iters * STORM_ITER_WORDS, 0);
  for (uint32_t i = 0; i < iters; ++i) {
    for (uint32_t lane = 0; lane < num_threads; ++lane) {
      h_data[i * STORM_ITER_WORDS + lane] = storm_value(i, lane);
    }
  }
  std::vector<uint32_t> h_ctrl(STORM_CTRL_WORDS, 0);
  std::vector<uint32_t> h_rec(total_threads * STORM_REC_WORDS, STORM_UNWRITTEN);
  const uint64_t data_size = h_data.size() * sizeof(uint32_t);
  const uint64_t ctrl_size = h_ctrl.size() * sizeof(uint32_t);
  const uint64_t rec_size  = h_rec.size() * sizeof(uint32_t);

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, data_size, VX_MEM_READ, &src0_buffer));
  RT_CHECK(vx_buffer_address(src0_buffer, &kernel_arg.src0_addr));
  RT_CHECK(vx_buffer_create(device, ctrl_size, VX_MEM_READ_WRITE, &ctrl_buffer));
  RT_CHECK(vx_buffer_address(ctrl_buffer, &kernel_arg.ctrl_addr));
  RT_CHECK(vx_buffer_create(device, rec_size, VX_MEM_READ_WRITE, &rec_buffer));
  RT_CHECK(vx_buffer_address(rec_buffer, &kernel_arg.rec_addr));

  std::cout << "upload buffers" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, src0_buffer, 0, h_data.data(), data_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, ctrl_buffer, 0, h_ctrl.data(), ctrl_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, rec_buffer, 0, h_rec.data(), rec_size, 0, nullptr, nullptr));

  std::cout << "launch kernel" << std::endl;
  launch_and_read(total_threads, h_rec.data(), rec_buffer, rec_size);
  RT_CHECK(vx_enqueue_read(queue, h_ctrl.data(), ctrl_buffer, 0, ctrl_size, 0, nullptr, nullptr));
  RT_CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));

  std::cout << "verify result" << std::endl;
  int errors = 0;
  const uint32_t final_count = iters * num_threads;

  uint32_t timer_cycles = 0, timer_maxlat = 0, timer_wait = 0, timer_total = 0, running = UINT32_MAX;
  for (uint32_t lane = 0; lane < num_threads; ++lane) {
    const uint32_t* r = &h_rec[lane * STORM_REC_WORDS];
    if (r[STORM_REC_KIND] != STORM_KIND_TIMER) {
      std::cout << "error: timer lane " << lane << " did not run" << std::endl;
      ++errors;
      continue;
    }
    running = std::min(running, r[STORM_REC_FLAG]);
    if (r[STORM_REC_ERRORS] != 0) {
      std::cout << "error: timer lane " << lane << " loaded " << r[STORM_REC_ERRORS]
                << " wrong values" << std::endl;
      ++errors;
    }
    timer_cycles = std::max(timer_cycles, r[STORM_REC_CYCLES]);
    timer_maxlat = std::max(timer_maxlat, r[STORM_REC_MAXLAT]);
    timer_wait   = std::max(timer_wait, r[STORM_REC_WAIT]);
    timer_total  = std::max(timer_total, r[STORM_REC_WAIT] + r[STORM_REC_CYCLES]);
  }
  std::cout << "timer: " << iters << " iterations in " << timer_cycles << " cycles, slowest "
            << timer_maxlat << ", after waiting " << timer_wait << " cycles for the fencers ("
            << running << " of " << fencers * num_threads << " fencer lanes running)" << std::endl;
  if (timer_total > budget) {
    std::cout << "error: the timer took " << timer_total << " cycles, over the budget" << std::endl;
    ++errors;
  }

  if (h_ctrl[STORM_COUNTER] != final_count) {
    std::cout << "error: counter is " << h_ctrl[STORM_COUNTER] << ", expected " << final_count << std::endl;
    ++errors;
  }
  if (h_ctrl[STORM_STARTED] != fencers * num_threads) {
    std::cout << "error: " << h_ctrl[STORM_STARTED] << " fencer lanes started, expected "
              << fencers * num_threads << std::endl;
    ++errors;
  }

  uint32_t min_fences = UINT32_MAX, max_fences = 0, max_cycles = 0, gave_up = 0, stale = 0;
  for (uint32_t t = num_threads; t < total_threads; ++t) {
    const uint32_t* r = &h_rec[t * STORM_REC_WORDS];
    const uint32_t w = t / num_threads, lane = t % num_threads;
    if (w <= idle) {
      if (r[STORM_REC_KIND] != STORM_KIND_IDLE) {
        std::cout << "error: idle warp " << w << " lane " << lane << " did not run" << std::endl;
        ++errors;
      }
      continue;
    }
    if (r[STORM_REC_KIND] != STORM_KIND_FENCER) {
      std::cout << "error: fencer warp " << w << " lane " << lane << " did not run" << std::endl;
      ++errors;
      continue;
    }
    min_fences = std::min(min_fences, r[STORM_REC_COUNT]);
    max_fences = std::max(max_fences, r[STORM_REC_COUNT]);
    max_cycles = std::max(max_cycles, r[STORM_REC_CYCLES]);
    if (r[STORM_REC_FLAG] == 0) {
      if (lane == 0) {
        std::cout << "error: fencer warp " << w << " gave up after " << r[STORM_REC_COUNT]
                  << " fences in " << r[STORM_REC_CYCLES] << " cycles without seeing the done flag"
                  << " (counter read " << r[STORM_REC_LAST] << " in its last pass)" << std::endl;
      }
      ++gave_up;
      ++errors;
    } else if (r[STORM_REC_SEEN] != final_count) {
      if (lane == 0) {
        std::cout << "error: fencer warp " << w << " read counter " << r[STORM_REC_SEEN]
                  << " after its acquire fence, expected " << final_count << std::endl;
      }
      ++stale;
      ++errors;
    }
  }
  if (fencers != 0) {
    std::cout << "fencers: " << min_fences << ".." << max_fences << " fences per warp, the longest ran "
              << max_cycles << " cycles" << std::endl;
  }
  if (gave_up != 0) {
    std::cout << gave_up << " fencer lanes gave up: "
              << ((timer_total >= budget)
                    ? "the fences starved the timer past the budget"
                    : "the timer finished in time, so a fence left the done flag stale")
              << std::endl;
  }
  if (stale != 0) {
    std::cout << stale << " fencer lanes read a stale counter after a fence" << std::endl;
  }
  return errors;
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  const uint32_t total_warps = num_cores * num_warps;

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  int errors = (mode == FENCE_MODE_STORM)
             ? run_storm(num_cores, num_warps, num_threads)
             : run_vecadd(total_warps * num_threads);

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
