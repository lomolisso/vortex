// mbar_init_fence: PTX fence.mbarrier_init.release.cluster on a DLMEM device
// with mbarriers (RV64). A SPREAD cluster of two blocks: each round rank 0
// initialises a fresh mbarrier with a plain store, fences, and passes it to
// rank 1 with a relaxed window store; rank 1's remote arrive must find the
// object initialised, or the round's arrive is lost.
//
//   -n R      rounds (default 256, at most 256)
//   -f F      the fence: none, cta, cluster, init (default init,
//             vx_fence_mbarrier_init_release_cluster())
//   -H 0|1    rank 0's other warps hammer its banks with atomics (default 1)
//   -t T      try_wait calls before a round counts as lost (default 8)
//
// Passes when both ranks ran and no arrive was lost. Without options on a
// device with one core it prints SKIPPED.

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace {

const char* const kFenceNames[] = {"none", "cta", "cluster", "init"};

uint32_t rounds = 256, fence = FENCE_INIT, hammer = 1, tries = 8;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

void usage() {
  std::printf("Usage: [-n rounds] [-f none|cta|cluster|init] [-H 0|1] [-t tries] [-h]\n");
}

uint32_t parse_count(const char* s) {
  char* end = nullptr;
  errno = 0;
  const unsigned long v = std::strtoul(s, &end, 10);
  if (*s < '0' || *s > '9' || *end != '\0' || errno != 0 || v > UINT32_MAX) {
    usage();
    std::exit(-1);
  }
  return (uint32_t)v;
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:f:H:t:h")) != -1) {
    switch (c) {
      case 'n': rounds = parse_count(optarg); break;
      case 'H': hammer = parse_count(optarg); break;
      case 't': tries  = parse_count(optarg); break;
      case 'f':
        fence = 4;
        for (uint32_t i = 0; i < 4; ++i)
          if (!std::strcmp(optarg, kFenceNames[i]))
            fence = i;
        if (fence == 4) {
          usage();
          std::exit(-1);
        }
        break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || rounds == 0 || rounds > MAX_ROUNDS || hammer > 1 || tries == 0) {
    usage();
    std::exit(-1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  uint64_t nt = 0, nw = 0, cores = 0, max_cluster = 0, socket_size = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,         &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,         &cores));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_SOCKET_SIZE,       &socket_size));
  std::printf("mbar_init_fence: %u rounds, fence %s, hammer %u, %u tries, %u cores, %u per socket, %u warps\n",
              rounds, kFenceNames[fence], hammer, tries, (uint32_t)cores, (uint32_t)socket_size, (uint32_t)nw);
  if (cores < 2 || max_cluster < 2 || (hammer && nw < 2)) {
    // A run launched without options on a device too small is skipped.
    std::printf("%s: needs a cluster of 2 on 2 cores%s\n%s", argc > 1 ? "Error" : "SKIPPED",
                hammer ? ", 2 warps" : "", argc > 1 ? "FAILED!\n" : "");
    vx_queue_release(q);
    vx_device_release(dev);
    return argc > 1 ? 1 : 0;
  }

  std::vector<uint32_t> out(2 * OUT_WORDS, 0);
  vx_buffer_h ob = nullptr;
  fence_arg_t a = {};
  CHECK(vx_buffer_create(dev, out.size() * 4, VX_MEM_READ_WRITE, &ob));
  CHECK(vx_buffer_address(ob, &a.out_addr));
  CHECK(vx_enqueue_write(q, ob, 0, out.data(), out.size() * 4, 0, nullptr, nullptr));
  a.rounds = rounds;
  a.fence  = fence;
  a.hammer = hammer;
  a.tries  = tries;

  vx_module_h mod = nullptr;
  vx_kernel_h k = nullptr;
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  CHECK(vx_module_get_kernel(mod, "main", &k));
  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = k;
  li.args_host      = &a;
  li.args_size      = sizeof(a);
  li.ndim           = 1;
  li.grid_dim[0]    = 2;
  li.grid_dim[1]    = 1;
  li.grid_dim[2]    = 1;
  li.block_dim[0]   = (uint32_t)(nt * nw);
  li.block_dim[1]   = 1;
  li.block_dim[2]   = 1;
  li.lmem_size      = LM_WORDS * 8;
  li.cluster_dim[0] = 2;
  li.cluster_dim[1] = 1;
  li.cluster_dim[2] = 1;
  li.cluster_policy = VX_CLUSTER_POLICY_SPREAD;

  vx_event_h le = nullptr, re = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &le));
  CHECK(vx_enqueue_read(q, out.data(), ob, 0, out.size() * 4, 1, &le, &re));
  CHECK(vx_event_wait_value(re, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(re);
  vx_event_release(le);

  const uint32_t* r0 = &out[0];
  const uint32_t* r1 = &out[OUT_WORDS];
  std::printf("mbar_init_fence: rank 0 on core %u, rank 1 on core %u; %u hammer atomics; lost %u, first %u\n",
              r0[OUT_CORE], r1[OUT_CORE], r0[OUT_HAM], r0[OUT_LOST], r0[OUT_FIRST]);
  int errors = 0;
  if (r0[OUT_RAN] != 1 || r1[OUT_RAN] != 1 || r1[OUT_RANK] != 1) {
    std::printf("*** a rank did not finish\n");
    ++errors;
  }
  if (r0[OUT_LOST] != 0) {
    std::printf("*** %u arrives lost, the first in round %u\n", r0[OUT_LOST], r0[OUT_FIRST] - 1);
    ++errors;
  }
  if (hammer && r0[OUT_HAM] == 0) {
    std::printf("*** the hammer warps made no atomics\n");
    ++errors;
  }

  vx_kernel_release(k);
  vx_module_release(mod);
  vx_buffer_release(ob);
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
