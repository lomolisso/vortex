// cf_rmsnorm: ClusterFusion's RMSNormKernel as the H100 ran it (variant
// cluster2: a cluster of 2 blocks of 512 threads per row of 8192 fp16
// values, y = x * rsqrt(mean(x^2) + 1e-6) * w), one block per core.
//
//   -b <batch>  rows, one cluster each (default 1; the H100 ran 64)
//   -r <reps>   launches, each one checked (default 3)
//   -h          this help
//
// Exit 0 when every launch passes, 1 on a failure or a bad option.
//
// Inputs, as the H100 wrapper: x [batch, 8192] and w [8192] fp16 drawn from
// N(0, 1), here by mt19937_64 seeded 20261110 + i for launch i, so every
// launch has a new input and its own reference (the models are
// deterministic: a launch on the same input would find the previous launch's
// correct values in memory). The reference is the wrapper's fp32 formula
// computed on the host; an element passes when its error is at most 1e-2
// (the wrapper's max-abs tolerance) or, where one fp16 ulp of the output is
// larger than 1e-2 (|y| >= 16), at most one ulp; the line of each launch
// counts the elements that needed the ulp bound. A launch also fails if a
// 256-byte guard tail after the output or the smid buffer changes, an
// element is left unwritten, or a cluster's two blocks are not on distinct
// cores.

#include <vortex2.h>
#include <vx_cuda_runtime.h>
#include <vx_cuda_fp.h>
#include <VX_types.h>
#include "common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <unistd.h>
#include <vector>

namespace {

namespace vc = vx::cuda;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "Error %s:%d: '%s' returned %s\n",         \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;
int batch = 1, reps = 3;
constexpr double kEps = 1e-6;
constexpr double kTolMaxAbs = 1e-2;

void usage() { std::printf("Usage: [-b batch] [-r reps] [-h]\n"); }

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "b:r:h")) != -1) {
    switch (c) {
    case 'b': batch = std::atoi(optarg); break;
    case 'r': reps = std::atoi(optarg); break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(1);
    }
  }
  if (optind < argc || batch < 1 || reps < 1) {
    usage();
    std::exit(1);
  }
}

struct Buf {
  vx_buffer_h h = nullptr;
  uint64_t addr = 0, size = 0;
  explicit Buf(uint64_t n) : size(n) {
    CHECK(vx_buffer_create(dev, size, VX_MEM_READ_WRITE, &h));
    CHECK(vx_buffer_address(h, &addr));
  }
  ~Buf() { vx_buffer_release(h); }
  void write(const void* p) {
    vx_event_h e = nullptr;
    CHECK(vx_enqueue_write(q, h, 0, p, size, 0, nullptr, &e));
    CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e);
  }
  void read(void* p) const {
    vx_event_h e = nullptr;
    CHECK(vx_enqueue_read(q, p, h, 0, size, 0, nullptr, &e));
    CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e);
  }
};

constexpr size_t kGuardBytes = 256;
constexpr uint32_t kGuardWord = 0x9e3779b9u;

// One fp16 ulp at the magnitude of v (the spacing of the binade holding |v|).
double f16_ulp(double v) {
  const double a = std::fabs(v);
  if (a < 6.103515625e-05)
    return std::ldexp(1.0, -24);
  int e = 0;
  std::frexp(a, &e);
  return std::ldexp(1.0, e - 11);
}

// Reports the first changed guard word after `body` bytes of a read-back buffer.
bool guard_intact(const std::vector<uint8_t>& v, size_t body, const char* what, int rep) {
  for (size_t i = body; i + 4 <= v.size(); i += 4) {
    uint32_t w;
    std::memcpy(&w, &v[i], 4);
    if (w != kGuardWord) {
      std::printf("Error: rep %d: guard word %zu after %s changed: 0x%08x\n", rep, (i - body) / 4, what, w);
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  parse_args(argc, argv);

  CHECK(vx_device_open(0, &dev));
  uint64_t num_cores = 0, nw = 0, nt = 0, max_cluster = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));
  const uint32_t smem = sizeof(RMSNormSmem);
  std::printf("cf_rmsnorm: batch=%d hidden=%d cluster=%d block=%d reps=%d; %llu cores, %llu warps x %llu threads; "
              "%u bytes of local memory per block\n",
              batch, HIDDEN_DIM, CLUSTER_SIZE, BLOCK_SIZE, reps, (unsigned long long)num_cores,
              (unsigned long long)nw, (unsigned long long)nt, smem);
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA | VX_ISA_EXT_L2CACHE;
  const char* bad_dev = nullptr;
  if ((isa & need) != need)
    bad_dev = "the device needs DLMEM, the A extension, the DXA and an L2";
  else if (nw * nt != (uint64_t)BLOCK_SIZE)
    bad_dev = "a block is 512 threads: the device needs NUM_WARPS x NUM_THREADS = 512";
  else if (num_cores < CLUSTER_SIZE || max_cluster < CLUSTER_SIZE)
    bad_dev = "the device needs 2 cores and a cluster of 2";
  if (bad_dev) {
    std::printf("Error: %s\nFAILED!\n", bad_dev);
    return 1;
  }

  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority = VX_QUEUE_PRIORITY_NORMAL;
  CHECK(vx_queue_create(dev, &qi, &q));
  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  vx_kernel_h k = nullptr;
  CHECK(vx_module_get_kernel(mod, "RMSNormKernel", &k));

  vc::launch_config_t cfg;
  cfg.grid = vc::dim3((uint32_t)(batch * CLUSTER_SIZE), 1, 1);
  cfg.block = vc::dim3(BLOCK_SIZE, 1, 1);
  cfg.dyn_smem = smem;
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)smem));
  int max_active = 0;
  CHECK(vc::occupancy_max_active_clusters<RMSNormKernel_cluster_dims>(dev, k, cfg, &max_active));
  std::printf("max_active_clusters = %d\n", max_active);
  if (max_active < 1) {
    std::printf("Error: the cluster cannot be placed\nFAILED!\n");
    return 1;
  }

  const size_t n = (size_t)batch * HIDDEN_DIM;
  const size_t nblocks = (size_t)batch * CLUSTER_SIZE;
  const size_t out_bytes = n * sizeof(__half);
  std::vector<__half> h_x(n), h_w(HIDDEN_DIM);
  std::vector<uint8_t> raw(out_bytes + kGuardBytes);
  const size_t smid_bytes = nblocks * sizeof(uint32_t);
  std::vector<uint8_t> raw_smid(smid_bytes + kGuardBytes);
  Buf d_x(n * sizeof(__half)), d_w(HIDDEN_DIM * sizeof(__half)), d_out(out_bytes + kGuardBytes),
      d_smid(smid_bytes + kGuardBytes);
  RMSNormArgs args = {d_out.addr, d_x.addr, d_w.addr, d_smid.addr};

  int failed = 0;
  for (int rep = 0; rep < reps; ++rep) {
    std::mt19937_64 g(20261110ull + (uint64_t)rep);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : h_x)
      v = __half(nd(g));
    for (auto& v : h_w)
      v = __half(nd(g));
    d_x.write(h_x.data());
    d_w.write(h_w.data());
    std::memset(raw.data(), 0xff, out_bytes);
    for (size_t i = out_bytes; i + 4 <= raw.size(); i += 4)
      std::memcpy(&raw[i], &kGuardWord, 4);
    d_out.write(raw.data());
    std::memset(raw_smid.data(), 0xff, smid_bytes);
    for (size_t i = smid_bytes; i + 4 <= raw_smid.size(); i += 4)
      std::memcpy(&raw_smid[i], &kGuardWord, 4);
    d_smid.write(raw_smid.data());

    vx_event_h ev = nullptr;
    CHECK(vc::launch_kernel_ex<RMSNormKernel_cluster_dims>(q, k, cfg, args, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    d_out.read(raw.data());
    d_smid.read(raw_smid.data());
    const uint32_t* h_smid = reinterpret_cast<const uint32_t*>(raw_smid.data());

    const bool out_guard = guard_intact(raw, out_bytes, "the output", rep);
    const bool guard = guard_intact(raw_smid, smid_bytes, "smid", rep) && out_guard;
    bool cores = true;
    for (size_t c = 0; c < (size_t)batch; ++c) {
      std::set<uint32_t> seen;
      for (int b = 0; b < CLUSTER_SIZE; ++b) {
        const uint32_t s = h_smid[c * CLUSTER_SIZE + b];
        if (s >= num_cores || !seen.insert(s).second) {
          std::printf("Error: rep %d: cluster %zu block %d reports smid %u\n", rep, c, b, s);
          cores = false;
        }
      }
    }

    const __half* y = reinterpret_cast<const __half*>(raw.data());
    long long bad = 0, ulp_bound = 0;
    double max_abs = 0, max_ref = 0;
    for (int r = 0; r < batch; ++r) {
      double ss = 0;
      for (int i = 0; i < HIDDEN_DIM; ++i) {
        const double xv = (float)h_x[(size_t)r * HIDDEN_DIM + i];
        ss += xv * xv;
      }
      const float rs = 1.0f / std::sqrt((float)(ss / HIDDEN_DIM) + (float)kEps);
      for (int i = 0; i < HIDDEN_DIM; ++i) {
        const size_t j = (size_t)r * HIDDEN_DIM + i;
        const double ref = (double)((float)h_x[j] * rs * (float)h_w[i]);
        const double got = (float)y[j];
        const double err = std::fabs(got - ref);
        const double ulp = f16_ulp(ref);
        const bool use_ulp = ulp > kTolMaxAbs;
        ulp_bound += use_ulp;
        if (!(err <= (use_ulp ? ulp : kTolMaxAbs)))
          ++bad;
        if (!(err <= max_abs))
          max_abs = err;
        max_ref = std::max(max_ref, std::fabs(ref));
      }
    }
    const bool pass = guard && cores && bad == 0;
    failed += !pass;
    std::printf("rep %d: %s bad=%lld max_abs=%.6g max_ref=%.4g ulp_bound=%lld\n", rep, pass ? "PASS" : "FAIL", bad,
                max_abs, max_ref, ulp_bound);
  }

  vx_kernel_release(k);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);
  if (failed) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
