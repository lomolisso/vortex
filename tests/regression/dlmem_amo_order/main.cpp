// dlmem_amo_order: release and acquire ordering of AMOs on the local memory of
// blocks on other cores, on a DLMEM device with the A extension (RV64). One
// cluster of four SPREAD blocks, rank r on core r (see kernel.cpp).
//
//   -m mp    message passing (default): rank 0 stores DATA into rank 1, then
//            swaps FLAG into rank 2 with amoswap.d.rl; rank 2 polls FLAG with
//            plain loads and reads DATA on each new value. Any read of DATA
//            older than its FLAG fails the run.
//   -m mpaq  rank 0 stores DATA into rank 2, then swaps FLAG into rank 1
//            with amoswap.d.rl; rank 2 reads the remote FLAG with amoor.d.aq
//            and its own DATA with a load issued before it branches on the flag
//   -m lat   fixed-count latency probes of rank 0 against rank 1, no spin:
//            an .aq AMO must hold the warp until it is answered, and an .rl
//            AMO must wait for an earlier remote store's ack
//   -m wide  two warps per block, every lane: .aqrl, .aq and .rl AMOs with
//            local and remote lanes, lr.d.aq/sc.d.rl and a global amoadd.d.aqrl
//            for N rounds; every counter is checked
//   -n N     producer rounds (default 2000)
//   -f       control: vx_fence() and a plain amoswap.d instead of .rl
//   -F       rank 3 floods rank 1 with window stores and rank 1 hammers its
//            own banks with atomics
//   -D       rank 1 streams DXA copies into its own local memory instead
//   -v       print every lat sample

#include <vortex2.h>
#include <dxa.h>
#include <VX_types.h>
#include "common.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint32_t K = 4;
constexpr uint32_t kSrcTiles = 3;
constexpr uint32_t kTileElems = 1024;

const char* kernel_file = "kernel.vxbin";
uint32_t mode    = MODE_MP;
uint32_t iters   = 2000;
uint32_t fence   = 0;
uint32_t flood   = 0;
uint32_t dxa     = 0;
bool     verbose = false;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

void usage() {
  std::printf("Usage: [-m mp|mpaq|lat|wide] [-n rounds] [-f] [-F] [-D] [-v] [-h]\n");
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
  while ((c = getopt(argc, argv, "m:n:fFDvh")) != -1) {
    switch (c) {
      case 'm':
        if (!std::strcmp(optarg, "mp")) {
          mode = MODE_MP;
        } else if (!std::strcmp(optarg, "mpaq")) {
          mode = MODE_MPAQ;
        } else if (!std::strcmp(optarg, "lat")) {
          mode = MODE_LAT;
        } else if (!std::strcmp(optarg, "wide")) {
          mode = MODE_WIDE;
        } else {
          usage();
          std::exit(-1);
        }
        break;
      case 'n': iters   = parse_count(optarg); break;
      case 'f': fence   = 1;                   break;
      case 'F': flood   = 1;                   break;
      case 'D': dxa     = 1;                   break;
      case 'v': verbose = true;                break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || iters == 0 || ((mode == MODE_LAT || mode == MODE_WIDE) && (fence || flood || dxa))) {
    usage();
    std::exit(-1);
  }
}

const char* mode_name() {
  switch (mode) {
    case MODE_MPAQ: return "mpaq";
    case MODE_LAT:  return "lat";
    case MODE_WIDE: return "wide";
    default:        return "mp";
  }
}

struct Stat {
  uint32_t min, med, max;
};

Stat stat_of(const uint32_t* s) {
  std::vector<uint32_t> v(s, s + LAT_ITERS);
  std::sort(v.begin(), v.end());
  return {v.front(), v[LAT_ITERS / 2], v.back()};
}

// The medians against thresholds halfway between no wait and the expected one:
// .aq waits for the AMO's whole round trip R, so A must reach R/2; .rl waits for
// a store's ack, which skips the read-modify-write and the data return and
// takes about R/2, so L must reach B + R/4. S also pays the fence's cache
// drain, so it only bounds L from above and is not a threshold.
int check_lat(const uint32_t* lat) {
  static const char* names[LAT_KINDS] = {"R", "A", "C", "B", "S", "L", "L0"};
  static const char* what[LAT_KINDS] = {
    "amoadd.d rd to rank 1, then a use of rd",
    "amoadd.d.aq zero to rank 1",
    "amoadd.d zero to rank 1 (control)",
    "amoswap.d rd on the own word, then a use of rd",
    "store to rank 1, fence, then B (reference)",
    "store to rank 1, then B with amoswap.d.rl",
    "store to rank 1, then B (control)",
  };
  Stat st[LAT_KINDS];
  for (uint32_t k = 0; k < LAT_KINDS; ++k) {
    st[k] = stat_of(lat + k * LAT_ITERS);
    std::printf("dlmem_amo_order: %-2s min %4u median %4u max %4u cycles: %s\n", names[k],
                st[k].min, st[k].med, st[k].max, what[k]);
    if (verbose) {
      std::printf("  ");
      for (uint32_t i = 0; i < LAT_ITERS; ++i)
        std::printf(" %u", lat[k * LAT_ITERS + i]);
      std::printf("\n");
    }
  }
  int errors = 0;
  const uint32_t aq_thr = st[LAT_R].med / 2;
  const uint32_t rl_thr = st[LAT_B].med + st[LAT_R].med / 4;
  std::printf("dlmem_amo_order: .aq threshold %u (R/2), .rl threshold %u (B+R/4)\n", aq_thr, rl_thr);
  if (st[LAT_C].med >= aq_thr) {
    std::printf("*** control C (%u) is not below R/2 (%u): the probe cannot tell a held warp\n",
                st[LAT_C].med, aq_thr);
    ++errors;
  }
  if (st[LAT_L0].med >= rl_thr) {
    std::printf("*** control L0 (%u) is not below B+R/4 (%u): the probe cannot tell a held AMO\n",
                st[LAT_L0].med, rl_thr);
    ++errors;
  }
  if (errors)
    return errors;
  if (st[LAT_A].med < aq_thr) {
    std::printf("*** .aq does not hold the warp: A = %u cycles, below R/2 = %u\n", st[LAT_A].med, aq_thr);
    ++errors;
  }
  if (st[LAT_L].med < rl_thr) {
    std::printf("*** .rl does not wait for the earlier remote store: L = %u cycles, below B+R/4 = %u\n",
                st[LAT_L].med, rl_thr);
    ++errors;
  }
  return errors;
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

  uint64_t nt = 0, cores = 0, max_cluster = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,         &cores));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &isa));
  std::printf("dlmem_amo_order: %s mode, %u cores, largest cluster %u, %u rounds, %s release%s%s\n",
              mode_name(), (uint32_t)cores, (uint32_t)max_cluster, iters,
              fence ? "fence + amoswap.d" : "amoswap.d.rl", flood ? ", flood" : "", dxa ? ", DXA on rank 1" : "");

  if (!(isa & VX_ISA_EXT_DLMEM) || !(isa & VX_ISA_STD_A) || (dxa && !(isa & VX_ISA_EXT_DXA))) {
    std::printf("Error: the device needs DLMEM and the A extension%s\nFAILED!\n", dxa ? ", and DXA" : "");
    return 1;
  }
  if (max_cluster < K) {
    std::printf("Error: a cluster of %u needs VX_CAPS_DLMEM_MAX_CLUSTER >= %u, the device has %u\nFAILED!\n",
                K, K, (uint32_t)max_cluster);
    return 1;
  }

  // Alive until the queue has run the upload.
  std::vector<uint32_t> h_src;
  vx_buffer_h src = nullptr;
  if (dxa) {
    h_src.resize(kSrcTiles * kTileElems);
    for (size_t i = 0; i < h_src.size(); ++i)
      h_src[i] = (uint32_t)i;
    uint64_t src_addr = 0;
    CHECK(vx_buffer_create(dev, h_src.size() * sizeof(uint32_t), VX_MEM_READ, &src));
    CHECK(vx_buffer_address(src, &src_addr));
    CHECK(vx_enqueue_write(q, src, 0, h_src.data(), h_src.size() * sizeof(uint32_t), 0, nullptr, nullptr));
    if (vortex::dxa::program_1d(dev, 0, src_addr, (uint32_t)h_src.size(), kTileElems, sizeof(uint32_t)) != 0) {
      std::printf("Error: cannot program the DXA descriptor\nFAILED!\n");
      return 1;
    }
  }

  std::vector<uint32_t> out(K * OUT_WORDS + LAT_KINDS * LAT_ITERS, 0);
  const uint64_t out_bytes = out.size() * sizeof(uint32_t);
  vx_buffer_h out_buf = nullptr;
  CHECK(vx_buffer_create(dev, out_bytes, VX_MEM_READ_WRITE, &out_buf));
  CHECK(vx_enqueue_write(q, out_buf, 0, out.data(), out_bytes, 0, nullptr, nullptr));

  amo_arg_t arg = {};
  CHECK(vx_buffer_address(out_buf, &arg.out_addr));
  arg.mode       = mode;
  arg.iters      = iters;
  arg.fence      = fence;
  arg.flood      = flood;
  arg.dxa        = dxa;
  arg.tile_elems = kTileElems;

  vx_module_h mod = nullptr;
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kernel));

  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kernel;
  li.args_host      = &arg;
  li.args_size      = sizeof(arg);
  li.ndim           = 1;
  li.grid_dim[0]    = K;
  li.grid_dim[1]    = 1;
  li.grid_dim[2]    = 1;
  const uint32_t nb = (uint32_t)nt * (mode == MODE_WIDE ? 2 : 1);
  li.block_dim[0]   = nb;
  li.block_dim[1]   = 1;
  li.block_dim[2]   = 1;
  li.lmem_size      = (mode == MODE_WIDE) ? WIDE_WORDS(nb) * sizeof(uint64_t)
                    : AO_TILE * sizeof(uint64_t) + (dxa ? kTileElems * sizeof(uint32_t) : 0);
  li.cluster_dim[0] = K;
  li.cluster_dim[1] = 1;
  li.cluster_dim[2] = 1;
  li.cluster_policy = VX_CLUSTER_POLICY_SPREAD;

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, out.data(), out_buf, 0, out_bytes, 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t b = 0; b < K; ++b) {
    const uint32_t* o = &out[b * OUT_WORDS];
    if (o[OUT_RAN] != 1 || o[OUT_RANK] != b) {
      std::printf("*** block %u: ran %u, rank %u\n", b, o[OUT_RAN], o[OUT_RANK]);
      ++errors;
    }
  }
  const uint32_t* o0 = &out[0];
  const uint32_t* o1 = &out[1 * OUT_WORDS];
  const uint32_t* o2 = &out[2 * OUT_WORDS];
  const uint32_t* o3 = &out[3 * OUT_WORDS];
  std::printf("dlmem_amo_order: ranks on cores %u %u %u %u\n",
              o0[OUT_CORE], o1[OUT_CORE], o2[OUT_CORE], o3[OUT_CORE]);
  if (!errors && (o1[OUT_CORE] == o0[OUT_CORE] || o2[OUT_CORE] == o0[OUT_CORE])) {
    std::printf("*** ranks 1 and 2 must sit on other cores than rank 0: nothing would cross the network\n");
    ++errors;
  }

  if (!errors) {
    if (mode == MODE_LAT) {
      errors += check_lat(&out[K * OUT_WORDS]);
    } else if (mode == MODE_WIDE) {
      uint64_t g = 0;
      std::memcpy(&g, &out[K * OUT_WORDS], sizeof(g));
      const uint64_t want = (uint64_t)iters * K * nb;
      std::printf("dlmem_amo_order: wide, %u lanes per block, wrong words per rank %u %u %u %u,"
                  " global count %lu of %lu\n", nb, o0[OUT_VIOL], o1[OUT_VIOL], o2[OUT_VIOL],
                  o3[OUT_VIOL], (unsigned long)g, (unsigned long)want);
      if (o0[OUT_VIOL] || o1[OUT_VIOL] || o2[OUT_VIOL] || o3[OUT_VIOL]) {
        std::printf("*** a counter or an .aq AMO's old value is wrong, or an SC kept failing\n");
        ++errors;
      }
      if (g != want) {
        std::printf("*** the global amoadd.d.aqrl count is wrong\n");
        ++errors;
      }
    } else {
      std::printf("dlmem_amo_order: %u violations in %u flag observations, first at FLAG %u with DATA %u;"
                  " rank 1 work %u, rank 3 flood stores %u\n",
                  o2[OUT_VIOL], o2[OUT_READS], o2[OUT_FIRST_F], o2[OUT_FIRST_D],
                  o1[OUT_WORK], o3[OUT_WORK]);
      if (o2[OUT_VIOL]) {
        std::printf("*** the consumer read DATA older than the FLAG it had seen\n");
        ++errors;
      }
      if (o2[OUT_READS] == 0) {
        std::printf("*** the consumer observed no flag value\n");
        ++errors;
      }
    }
  }

  vx_kernel_release(kernel);
  vx_module_release(mod);
  vx_buffer_release(out_buf);
  if (src)
    vx_buffer_release(src);
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
