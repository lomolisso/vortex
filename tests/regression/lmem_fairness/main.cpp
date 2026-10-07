// lmem_fairness: a local-memory bank must not starve one LSU block.
//
// The warps of one LSU block poll a local-memory flag with atomics while one
// warp of another block sets it. Every poll lands on the flag's bank, so the
// setter's store gets in only if the bank arbitrates fairly between blocks.
// Pollers give up after a bounded number of polls, so starvation fails the run
// instead of hanging it.
//
//   mode 0: block 0 polls, once per other block, a warp of which sets the flag:
//           a block sharing block 0's bank-arbiter slice and a block behind
//           the slice join can starve independently.
//   mode 1: the control, roles swapped: the last block polls, warp 0 sets it.
//
// A poller that sees the flag on its first poll never contended with the
// setter, so that proves nothing and fails the run too.

#include <algorithm>
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

constexpr uint32_t kUnwritten = 0xFFFFFFFFu;

const char* kernel_file = "kernel.vxbin";
int      only_mode = -1;
uint32_t delay     = 128;
uint32_t bound     = 256;

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_buffer_h polls_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-m mode (0: block 0 polls, 1: control; default both)]"
            << " [-d producer delay] [-b poll bound] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:m:d:b:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'm': only_mode   = atoi(optarg); break;
    case 'd': delay       = (uint32_t)atoi(optarg); break;
    case 'b': bound       = (uint32_t)atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (polls_buffer) {
      vx_buffer_release(polls_buffer);
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

struct Launch {
  uint32_t num_ctas;
  uint32_t warps;
  uint32_t threads;
  uint32_t lmem_bytes;
  kernel_arg_t arg;
  std::vector<uint32_t> polls;
};

// Runs one poll/produce pairing and returns its error count.
static uint32_t run(Launch& ln, uint32_t mode, uint32_t poll_block, uint32_t producer_wid) {
  const uint32_t num_blocks     = ln.arg.num_blocks;
  const uint32_t producer_block = producer_wid % num_blocks;
  const size_t   polls_bytes    = ln.polls.size() * sizeof(uint32_t);
  ln.arg.poll_block   = poll_block;
  ln.arg.producer_wid = producer_wid;

  std::cout << "mode " << mode << (mode ? " (control)" : "") << ": block " << poll_block
            << " polls, warp " << producer_wid << " (block " << producer_block
            << ") sets the flag after " << delay << " iterations; bound " << bound
            << " polls" << std::endl;

  std::fill(ln.polls.begin(), ln.polls.end(), kUnwritten);
  RT_CHECK(vx_enqueue_write(queue, polls_buffer, 0, ln.polls.data(), polls_bytes, 0, nullptr, nullptr));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &ln.arg;
    li.args_size    = sizeof(ln.arg);
    li.ndim         = 1;
    li.grid_dim[0]  = ln.num_ctas;
    li.block_dim[0] = ln.warps * ln.threads;
    li.lmem_size    = ln.lmem_bytes;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  RT_CHECK(vx_enqueue_read(queue, ln.polls.data(), polls_buffer, 0, polls_bytes, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  uint32_t pollers = 0, starved = 0, missing = 0;
  uint32_t min_polls = UINT32_MAX, max_polls = 0;
  for (uint32_t cta = 0; cta < ln.num_ctas; ++cta) {
    for (uint32_t w = 0; w < ln.warps; ++w) {
      if (w == producer_wid || w % num_blocks != poll_block) {
        continue;
      }
      for (uint32_t l = 0; l < ln.threads; ++l) {
        const uint32_t p = ln.polls[((size_t)cta * ln.warps + w) * ln.threads + l];
        ++pollers;
        if (p == kUnwritten) {
          if (missing++ == 0) {
            std::cout << "*** cta " << cta << " warp " << w << " lane " << l
                      << " never reported" << std::endl;
          }
        } else if (p == 0) {
          if (starved++ == 0) {
            std::cout << "*** cta " << cta << " warp " << w << " lane " << l
                      << " polled " << bound << " times without seeing the flag" << std::endl;
          }
        } else {
          min_polls = std::min(min_polls, p);
          max_polls = std::max(max_polls, p);
        }
      }
    }
  }

  std::cout << "mode " << mode << ", producer block " << producer_block << ": ";
  if (pollers == 0) {
    std::cout << "no poller in block " << poll_block << std::endl;
    return 1;
  }
  if (starved + missing != 0) {
    std::cout << starved << " of " << pollers << " pollers exhausted the bound, "
              << missing << " never reported" << std::endl;
    return starved + missing;
  }
  std::cout << "all " << pollers << " pollers saw the flag after "
            << min_polls << ".." << max_polls << " polls" << std::endl;
  if (min_polls < 2) {
    std::cout << "*** a poller saw the flag on its first poll: producer stored before the"
              << " pollers contended; raise -d" << std::endl;
    return 1;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  if (only_mode > 1 || bound == 0) {
    show_usage();
    return -1;
  }

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_STD_A) == 0) {
    std::cout << "Error: needs the A extension" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t num_cores, num_warps, num_threads, issue_width;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_ISSUE_WIDTH, &issue_width));
  if (issue_width != VX_CFG_ISSUE_WIDTH) {
    std::cout << "Error: built for issue width " << VX_CFG_ISSUE_WIDTH
              << ", device has " << issue_width << "; rebuild with the device's CONFIGS" << std::endl;
    cleanup();
    return -1;
  }

  Launch ln = {};
  ln.num_ctas   = (uint32_t)num_cores;
  ln.warps      = (uint32_t)num_warps;
  ln.threads    = (uint32_t)num_threads;
  ln.lmem_bytes = sizeof(uint32_t);
  const uint32_t num_blocks = VX_CFG_NUM_LSU_BLOCKS;
  std::cout << "cores: " << num_cores << ", warps: " << ln.warps << ", threads: " << ln.threads
            << ", LSU blocks: " << num_blocks << std::endl;
  if (num_blocks < 2) {
    cleanup();
    std::cout << "SKIPPED (a single LSU block has no other block to starve)" << std::endl;
    return 0;
  }

  RT_CHECK(vx_check_occupancy(device, ln.warps * ln.threads, ln.lmem_bytes));

  ln.polls.resize((size_t)ln.num_ctas * ln.warps * ln.threads);
  RT_CHECK(vx_buffer_create(device, ln.polls.size() * sizeof(uint32_t), VX_MEM_READ_WRITE, &polls_buffer));

  ln.arg.num_blocks = num_blocks;
  ln.arg.delay      = delay;
  ln.arg.bound      = bound;
  RT_CHECK(vx_buffer_address(polls_buffer, &ln.arg.polls_addr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  uint32_t errors = 0;
  if (only_mode != 1) {
    for (uint32_t b = 1; b < num_blocks; ++b) {
      errors += run(ln, 0, 0, b);
    }
  }
  if (only_mode != 0) {
    errors += run(ln, 1, num_blocks - 1, 0);
  }

  cleanup();

  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
