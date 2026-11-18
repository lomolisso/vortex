// tmap_cache: the tensor-map cache levels as the H100 shows them (doc 14 §4, §8; E13_h100.md M1, M2, M5).
//   -t rr -M m -R r -e L    M maps round robin, r rounds: the last round's median is level L (1, 2 or 3)
//   -t lru                  maps 0-7, 0, 8, then 0: level 1 with LRU, level 2 with round robin
//   -t inval                fence acquire drops a map from both levels; without it level 2 holds it
//   -t prefetch -S s        prefetch.tensormap, s idle cycles, use = level 1; an unused map = level 3
//   -t param -S s           the argument's map: first use = a level-2 hit (the launch loaded the window), then level 1; prefetched = level 1
//   -t relaunch             -t param after a launch without a copy whose argument maps another tensor: same costs, data
//   -t launch -M m          launch 1's first use of launch 0's maps is level 3
//   -t share -M m -e L      block 1 (another SM) uses block 0's maps: level 2 in one cluster, 3 across clusters
//   -t plru -M m -R i       map m every i-th hop of a stream of m maps: never level 3
// Levels: 1 = the base within +-TOL; 2 = above it by at most the level-2 trip (common.h) + TOL; 3 = above that:
// a map read from GMEM crosses the SM port and the L2, which the level-2 trip does not. The H100's much larger
// level-3 cost comes from the L2-slice pipeline and crossbar Vortex does not build yet (task G reference only).
// The H100's level costs over a level-1 hit (E13_h100.md M1, M5; 14/spec.md section 4) are printed as H100REF lines.
#include <vortex2.h>
#include <dxa.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

namespace dh = vortex::dxa;

static const char* kTests[] = {"rr", "lru", "inval", "prefetch", "param", "launch", "share", "plru", "relaunch"};
static const int TOL = 2;
static const int H100_L2_OVER_L1 = 43;
static const int H100_PARAM_FIRST_USE = 34;
static const int H100_L3_OVER_L1 = 215;

static int errors = 0;
static uint32_t g_base = 0, g_l2 = 0;

static int level_of(uint32_t t) {
  const int d = (int)t - (int)g_base;
  if (std::abs(d) <= TOL) return 1;
  if (d > TOL && d <= (int)g_l2 + TOL) return 2;
  if (d > (int)g_l2 + TOL) return 3;
  return 0;
}

static void h100ref(const char* metric, int h100, int vortex) {
  std::printf("H100REF metric=%s h100=%d vortex=%d\n", metric, h100, vortex);
}

static void expect_level(const char* what, uint32_t t, int want) {
  const int got = level_of(t);
  std::printf("%s: %u cycles (base %u, %+d) level %d, expected %d\n", what, t, g_base, (int)t - (int)g_base, got, want);
  if (got != want)
    ++errors;
  if (want == 2)
    h100ref("tmap_l2_over_l1", H100_L2_OVER_L1, (int)t - (int)g_base);
  if (want == 3)
    h100ref("tmap_l3_over_l1", H100_L3_OVER_L1, (int)t - (int)g_base);
}

int main(int argc, char** argv) {
  tc_arg_t arg = {};
  arg.test = T_RR;
  arg.nmaps = 8;
  arg.rounds = 3;
  arg.spin = 3000;
  int want = 1;
  int c;
  while ((c = getopt(argc, argv, "t:M:R:S:e:h")) != -1) {
    switch (c) {
    case 't': {
      arg.test = ~0u;
      for (uint32_t i = 0; i < sizeof(kTests) / sizeof(kTests[0]); ++i)
        if (!std::strcmp(optarg, kTests[i])) arg.test = i;
      break;
    }
    case 'M': arg.nmaps = atoi(optarg); break;
    case 'R': arg.rounds = atoi(optarg); break;
    case 'S': arg.spin = atoi(optarg); break;
    case 'e': want = atoi(optarg); break;
    default:
      std::printf("usage: %s [-t rr|lru|inval|prefetch|param|launch|share|plru|relaunch] [-M maps] [-R rounds] [-S spin] [-e level]\n",
                  argv[0]);
      return -1;
    }
  }
  if (arg.test == ~0u || arg.nmaps < 1 || arg.rounds < 1 || want < 1 || want > 3) {
    std::printf("*** bad arguments\n");
    return -1;
  }

  vx_device_h dev;
  vx_queue_h q;
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  CHECK(vx_device_open(0, &dev));
  CHECK(vx_queue_create(dev, &qi, &q));
  uint64_t nt = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));

  const uint32_t nmaps = arg.nmaps + 64;
  vx_buffer_h tbuf, mbuf, obuf, fbuf;
  CHECK(vx_buffer_create(dev, 64, VX_MEM_READ, &tbuf));
  CHECK(vx_buffer_create(dev, (uint64_t)nmaps * 128, VX_MEM_READ, &mbuf));
  CHECK(vx_buffer_create(dev, 2 * O_WORDS * 4, VX_MEM_READ_WRITE, &obuf));
  CHECK(vx_buffer_create(dev, 64, VX_MEM_READ_WRITE, &fbuf));
  uint64_t taddr = 0;
  CHECK(vx_buffer_address(tbuf, &taddr));
  CHECK(vx_buffer_address(mbuf, &arg.maps_addr));
  CHECK(vx_buffer_address(obuf, &arg.out_addr));
  CHECK(vx_buffer_address(fbuf, &arg.flag_addr));

  // One 16-B row of four u32 and its whole-tensor box; every map is the same map at its own address.
  const uint64_t gdim[2] = {4, 1}, gstr[1] = {16};
  const uint32_t box[2] = {4, 1}, ones[2] = {1, 1};
  vx_tensor_map_t m;
  if (dh::encode_tiled(&m, dh::TensorMapDataType::UInt32, 2, taddr, gdim, gstr, box, ones,
                       dh::TensorMapInterleave::None, dh::TensorMapSwizzle::None, dh::TensorMapL2Promotion::None,
                       dh::TensorMapFloatOOBfill::None) != VX_SUCCESS) {
    std::printf("*** encode_tiled failed\n");
    return -1;
  }
  arg.map = m;
  arg.map2 = m;
  std::vector<vx_tensor_map_t> maps(nmaps, m);
  const uint32_t tdata[16] = {0x11, 0x22, 0x33, 0x44};
  std::vector<uint32_t> zeros(2 * O_WORDS, 0);
  CHECK(vx_enqueue_write(q, tbuf, 0, tdata, 64, 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, mbuf, 0, maps.data(), (uint64_t)nmaps * 128, 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, fbuf, 0, zeros.data(), 64, 0, nullptr, nullptr));

  vx_module_h mod;
  vx_kernel_h kernel;
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  CHECK(vx_module_get_kernel(mod, "tc_k", &kernel));

  std::vector<uint32_t> out(O_WORDS, 0);
  // Launch 0's argument names zeros (past tdata's row): a window line it leaves
  // in flight that reached launch 1 would show as wrong data.
  vx_tensor_map_t decoy;
  const uint64_t daddr = taddr + 32;
  if (dh::encode_tiled(&decoy, dh::TensorMapDataType::UInt32, 2, daddr, gdim, gstr, box, ones,
                       dh::TensorMapInterleave::None, dh::TensorMapSwizzle::None, dh::TensorMapL2Promotion::None,
                       dh::TensorMapFloatOOBfill::None) != VX_SUCCESS) {
    std::printf("*** encode_tiled failed\n");
    return -1;
  }
  const uint32_t launches = (arg.test == T_LAUNCH || arg.test == T_RELAUNCH) ? 2 : 1;
  for (uint32_t l = 0; l < launches; ++l) {
    arg.launch = l;
    arg.map = arg.map2 = (arg.test == T_RELAUNCH && l == 0) ? decoy : m;
    CHECK(vx_enqueue_write(q, obuf, 0, zeros.data(), 2 * O_WORDS * 4, 0, nullptr, nullptr));
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = kernel;
    li.args_host = &arg;
    li.args_size = sizeof(arg);
    li.ndim = 1;
    li.grid_dim[0] = (arg.test == T_SHARE) ? 2 : 1;
    li.grid_dim[1] = li.grid_dim[2] = 1;
    li.block_dim[0] = 2 * (uint32_t)nt;
    li.block_dim[1] = li.block_dim[2] = 1;
    li.lmem_size = LM_DATA + 64;
    li.cluster_dim[0] = li.cluster_dim[1] = li.cluster_dim[2] = 1;
    li.cluster_policy = VX_CLUSTER_POLICY_DEFAULT;
    vx_event_h ev = nullptr, oev = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
    CHECK(vx_enqueue_read(q, out.data(), obuf, 0, O_WORDS * 4, 1, &ev, &oev));
    CHECK(vx_event_wait_value(oev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(oev);
    vx_event_release(ev);
  }

  const uint32_t* k = out.data() + O_KNOBS;
  std::printf("knobs l2_trip=%u param_l2_trip=%u l1_entries=%u l1_repl=%u l2_entries=%u l2_ways=%u\n", k[0], k[1],
              k[2], k[3], k[4], k[5]);
  if (k[0] == 0) {
    std::printf("*** the front-end switch is off: no tensor-map cache levels to test\n");
    return -1;
  }
  if (out[O_ERR]) {
    std::printf("*** %s (%u)\n", out[O_ERR] == 3 ? "the argument map's copy read wrong data" : "an mbarrier or flag wait gave up",
                out[O_ERR]);
    ++errors;
  }
  g_base = out[O_BASE];
  g_l2 = std::max(k[0], k[1]);
  const uint32_t* t = out.data() + O_T;
  std::vector<uint32_t> v(out.begin() + O_VALS, out.begin() + O_VALS + std::min<uint32_t>(out[O_N], MAX_VALS));
  std::vector<uint32_t> sorted = v;
  std::sort(sorted.begin(), sorted.end());
  const uint32_t median = sorted.empty() ? 0 : sorted[sorted.size() / 2];

  switch (arg.test) {
  case T_RR:
  case T_SHARE:
  case T_LAUNCH: {
    if (arg.test == T_SHARE) {
      std::printf("share: block 0 on core %u, block 1 on core %u\n", out[O_CORE], out[O_CORE + 1]);
      if (out[O_CORE] == out[O_CORE + 1]) {
        std::printf("*** both blocks ran on one SM\n");
        ++errors;
      }
    }
    if (v.size() != arg.nmaps)
      ++errors;
    const std::string what = std::string(kTests[arg.test]) + " M=" + std::to_string(arg.nmaps) + " median";
    expect_level(what.c_str(), median, arg.test == T_LAUNCH ? 3 : want);
    std::printf("%s min=%u max=%u n=%zu\n", kTests[arg.test], sorted.empty() ? 0 : sorted.front(),
                sorted.empty() ? 0 : sorted.back(), v.size());
    break;
  }
  case T_LRU:
    expect_level("lru: map 0 after maps 0-7, 0, 8", t[0], k[3] ? 1 : 2);
    break;
  case T_INVAL:
    expect_level("inval: in level 1, fence, use", t[0], 3);
    expect_level("inval: only in level 2, fence, use", t[1], 3);
    expect_level("inval: only in level 2, no fence, use", t[2], 2);
    break;
  case T_PREFETCH:
    expect_level("prefetch: prefetched, idle, use", t[0], 1);
    expect_level("prefetch: never used, use", t[1], 3);
    break;
  case T_PARAM:
  case T_RELAUNCH:
    std::printf("param: first use: %u cycles (base %u, %+d) level %d, expected 2\n", t[0], g_base,
                (int)t[0] - (int)g_base, level_of(t[0]));
    if (level_of(t[0]) != 2)
      ++errors;
    h100ref("tmap_param_first_use", H100_PARAM_FIRST_USE, (int)t[0] - (int)g_base);
    expect_level("param: second use", t[1], 1);
    expect_level("param: prefetched, idle, use", t[2], 1);
    break;
  case T_PLRU: {
    uint32_t l3 = 0;
    for (uint32_t x : v)
      l3 += (level_of(x) == 3);
    std::printf("plru: %zu probe hops, %u at level 3, median %u (base %u)\n", v.size(), l3, median, g_base);
    if (v.empty() || l3)
      ++errors;
    break;
  }
  }

  vx_buffer_release(fbuf);
  vx_buffer_release(obuf);
  vx_buffer_release(mbuf);
  vx_buffer_release(tbuf);
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
