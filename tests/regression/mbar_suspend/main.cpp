// mbarrier try_wait timing of the H100 front end (12/spec.md §4.2, §4.4), one block on one core:
//   -m0 time limit: warp 0, lanes [0, L), R try_waits on a phase that never completes, hint H ns (-t);
//   -m1 the op alone: R try_waits on a completed phase;
//   -m2 wake-up: warp 0 loops on try_wait (hint H unless none), warp 1 lane 0 arrives D cycles later; arrive clock
//       to waiter clock;
//   -m3 as -m2 with a test_wait spin.
// Prints the median and every round; mbar_suspend_check.py holds the pass conditions.
#include <vortex2.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>
#include "common.h"

#define CHECK(expr)                                                                     \
  do {                                                                                  \
    vx_result_t _r = (expr);                                                            \
    if (_r != VX_SUCCESS) {                                                             \
      std::printf("*** %s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,             \
                  vx_result_string(_r));                                                \
      std::exit(-1);                                                                    \
    }                                                                                   \
  } while (0)

int main(int argc, char** argv) {
  sus_arg_t arg = {};
  arg.mode = M_LIMIT;
  arg.lanes = 0;
  arg.hint = NO_HINT;
  arg.rounds = 3;
  arg.delay = 2000;
  int c;
  while ((c = getopt(argc, argv, "m:l:t:r:d:h")) != -1) {
    switch (c) {
    case 'm': arg.mode = atoi(optarg); break;
    case 'l': arg.lanes = atoi(optarg); break;
    case 't': arg.hint = (uint32_t)strtoul(optarg, nullptr, 0); break;
    case 'r': arg.rounds = atoi(optarg); break;
    case 'd': arg.delay = atoi(optarg); break;
    default:
      std::printf("usage: %s [-m mode] [-l lanes] [-t hint_ns] [-r rounds] [-d delay]\n", argv[0]);
      return -1;
    }
  }

  vx_device_h dev;
  vx_queue_h q;
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  CHECK(vx_device_open(0, &dev));
  CHECK(vx_queue_create(dev, &qi, &q));
  uint64_t nt = 0, nw = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  if (arg.lanes == 0 || arg.lanes > nt)
    arg.lanes = (uint32_t)nt;
  const uint32_t warps = (arg.mode >= M_WAKE) ? 2 : 1;
  if (arg.mode > M_SPIN || arg.rounds < 1 || arg.rounds > MAX_ROUNDS || warps > nw) {
    std::printf("*** bad arguments\n");
    return -1;
  }

  vx_buffer_h obuf;
  CHECK(vx_buffer_create(dev, O_WORDS * 4, VX_MEM_READ_WRITE, &obuf));
  CHECK(vx_buffer_address(obuf, &arg.out_addr));
  std::vector<uint32_t> out(O_WORDS, 0);

  vx_module_h mod;
  vx_kernel_h kernel;
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  CHECK(vx_module_get_kernel(mod, "sus_k", &kernel));

  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = kernel;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = li.grid_dim[1] = li.grid_dim[2] = 1;
  li.block_dim[0] = warps * (uint32_t)nt;
  li.block_dim[1] = li.block_dim[2] = 1;
  li.lmem_size = LM_WAKE + 8 * MAX_ROUNDS;
  li.cluster_dim[0] = li.cluster_dim[1] = li.cluster_dim[2] = 1;
  li.cluster_policy = VX_CLUSTER_POLICY_DEFAULT;
  vx_event_h ev = nullptr, oev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
  CHECK(vx_enqueue_read(q, out.data(), obuf, 0, O_WORDS * 4, 1, &ev, &oev));
  CHECK(vx_event_wait_value(oev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(oev);
  vx_event_release(ev);

  std::printf("knobs suspend=%u fe=%u threads=%u\n", out[O_KNOBS], out[O_KNOBS + 1], (uint32_t)nt);
  std::vector<uint32_t> t(arg.rounds);
  for (uint32_t r = 0; r < arg.rounds; ++r) {
    t[r] = (arg.mode >= M_WAKE) ? out[O_TW + r] - out[O_TA + r] : out[O_T + r];
    std::printf("round %u t=%u\n", r, t[r]);
  }
  std::vector<uint32_t> s(t);
  std::sort(s.begin(), s.end());
  std::printf("result mode=%u lanes=%u hint=%d rounds=%u median=%u\n", arg.mode, arg.lanes,
              (arg.hint == NO_HINT) ? -1 : (int)arg.hint, arg.rounds, s[arg.rounds / 2]);

  int errors = 0;
  if (out[O_ERR]) {
    std::printf("*** a try_wait returned the wrong answer\n");
    ++errors;
  }
  if (arg.mode >= M_WAKE) {
    for (uint32_t r = 0; r < arg.rounds; ++r) {
      if ((int32_t)(out[O_TW + r] - out[O_TA + r]) <= 0) {
        std::printf("*** round %u: the waiter left before the arrive\n", r);
        ++errors;
      }
    }
  }

  vx_buffer_release(obuf);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);
  if (errors) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
