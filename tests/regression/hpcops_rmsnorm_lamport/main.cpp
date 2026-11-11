// hpcops_rmsnorm_lamport: the cluster RMSNorm of hpc-ops' fused all-reduce +
// RMSNorm (rmsNormLamport<bf16, bf16, 1>), one cluster per token, launched as
// the hpc-ops host launches it: grid (tokens, C), cluster (1, C, 1), block,
// C and the dynamic shared memory from adjustGridConfig (H = 8192 with one
// token gives C = 8 blocks of 128 threads), one block per core.
// adjustGridConfig takes the SM count; here it is the device's core count,
// which can choose a different shape than the H100's 132 SMs for -t > 1
// (the 1-token shape is the same).
//
//   -t <tokens>  tokens, one cluster each (default 1)
//   -d <dim>     hidden size H, a multiple of 8 (default 8192)
//   -r <reps>    launches, each one checked (default 3)
//   -h           this help
//
// Exit 0 when every launch passes, 1 on a failure or a bad option.
//
// Inputs: the all-reduced input, the residual and gamma are bf16 values of
// N(0, 1) drawn by mt19937_64 seeded 20261110 + i for launch i (a new input
// and reference per launch: the models are deterministic, and a launch on the
// same input would find the previous launch's correct values in memory);
// launch i scales the input and the residual by 1 + i, so a block sum left
// over from the previous launch is off by a large factor; eps 1e-6.
// Checks, against h = bf16(input + residual) computed on the host with the
// same bf16 addition:
// - the upstream test's rule, atol = rtol = 0.1 (|got - ref| <= 0.1 + 0.1
//   |ref|), on residual_out and on out = h * rsqrt(mean(h^2) + eps) * gamma
//   (reference in double);
// - residual_out bit-exact (the sum is rounded once, as on the host);
// - a tight gate on out, which a missing or stale block sum fails:
//   |got - ref| <= 2^-7 |ref| + 1e-3 per element (one bf16 rounding is at most
//   2^-8 |ref|); `tight` on each rep line is the worst error over that limit.
// A launch also fails if a 256-byte guard tail after an output or the smid
// buffer changes, an element is left unwritten, or a cluster's blocks are not
// on distinct cores.

#include <vortex2.h>
#include <vx_cuda_runtime.h>
#include <vx_cuda_fp.h>
#include <VX_types.h>
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace {

namespace vc = vx::cuda;
typedef __nv_bfloat16 T;

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
int tokens = 1, dim = 8192, reps = 3;
constexpr float kEps = 1e-6f;
constexpr double kAtol = 0.1, kRtol = 0.1;

void usage() { std::printf("Usage: [-t tokens] [-d dim] [-r reps] [-h]\n"); }

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "t:d:r:h")) != -1) {
    switch (c) {
    case 't': tokens = std::atoi(optarg); break;
    case 'd': dim = std::atoi(optarg); break;
    case 'r': reps = std::atoi(optarg); break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(1);
    }
  }
  if (optind < argc || tokens < 1 || dim < 8 || dim % 8 != 0 || reps < 1) {
    usage();
    std::exit(1);
  }
}

int ceil_div(int x, int y) { return (x + y - 1) / y; }

// hpc-ops' adjustGridConfig: (block size, cluster size, loads per thread),
// with the device's core count as the SM count.
std::tuple<int, int, int> adjust_grid_config(int num_tokens, int d, int elts_per_thread, int sm_count) {
  int cluster_size = 8;
  int block_size = 128;
  const int threads_needed = ceil_div(d, elts_per_thread);
  const int loads_per_thread = 1;
  while (threads_needed % cluster_size != 0 && cluster_size > 1)
    cluster_size /= 2;
  block_size = ceil_div(threads_needed, cluster_size);
  while (block_size < 128 && cluster_size >= 2) {
    block_size *= 2;
    cluster_size /= 2;
  }
  while (num_tokens * cluster_size > sm_count && cluster_size > 1 && block_size <= 512) {
    block_size *= 2;
    cluster_size /= 2;
  }
  while (block_size > 1024) {
    if (cluster_size < 8)
      cluster_size = cluster_size << 1;
    else
      break;
    block_size = ceil_div(threads_needed, cluster_size * loads_per_thread);
  }
  return {block_size, cluster_size, loads_per_thread};
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

// An output whose body starts as 0xffff (a bf16 NaN, which the kernel never
// writes for finite inputs), then the guard tail.
struct Output {
  Buf buf;
  size_t body;
  const char* name;
  std::vector<uint8_t> raw;
  Output(size_t body_, const char* name_) : buf(body_ + kGuardBytes), body(body_), name(name_), raw(body_ + kGuardBytes) {}
  void reset() {
    std::fill(raw.begin(), raw.begin() + body, 0xff);
    for (size_t i = body; i + 4 <= raw.size(); i += 4)
      std::memcpy(&raw[i], &kGuardWord, 4);
    buf.write(raw.data());
  }
  bool fetch(int rep) {
    buf.read(raw.data());
    for (size_t i = body; i + 4 <= raw.size(); i += 4) {
      uint32_t w;
      std::memcpy(&w, &raw[i], 4);
      if (w != kGuardWord) {
        std::printf("Error: rep %d: guard word %zu after %s changed: 0x%08x\n", rep, (i - body) / 4, name, w);
        return false;
      }
    }
    return true;
  }
  const T* data() const { return reinterpret_cast<const T*>(raw.data()); }
};

struct Tally {
  long long bad = 0;
  double max_err = 0, worst = 0;
  void add(double got, double ref) {
    const double err = std::fabs(got - ref), lim = kAtol + kRtol * std::fabs(ref);
    if (!(err <= lim))
      ++bad;
    if (!(err <= max_err))
      max_err = err;
    if (!(err / lim <= worst))
      worst = err / lim;
  }
};

} // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  parse_args(argc, argv);

  CHECK(vx_device_open(0, &dev));
  uint64_t num_cores = 0, nw = 0, nt = 0, lmem_cap = 0, max_cluster = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));

  const int elts = (int)(16 / sizeof(T));
  int block = 0, cluster = 0, loads = 0;
  std::tie(block, cluster, loads) = adjust_grid_config(tokens, dim, elts, (int)num_cores);
  const int num_threads = cluster * block;
  const int dim_padded = ceil_div(dim, elts * num_threads) * elts * num_threads;
  const int iters = dim_padded / num_threads;
  const uint32_t smem = kDynOffset + (uint32_t)(3 * block * iters * sizeof(T));
  std::printf("hpcops_rmsnorm_lamport: tokens=%d dim=%d block=%d cluster=%d loads=%d reps=%d; %llu cores, %llu warps "
              "x %llu threads, %llu bytes of local memory; %u bytes of local memory per block\n",
              tokens, dim, block, cluster, loads, reps, (unsigned long long)num_cores, (unsigned long long)nw,
              (unsigned long long)nt, (unsigned long long)lmem_cap, smem);
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA | VX_ISA_EXT_L2CACHE;
  const char* bad_dev = nullptr;
  if ((isa & need) != need)
    bad_dev = "the device needs DLMEM, the A extension, the DXA and an L2";
  else if (cluster < 2)
    bad_dev = "this H gives no cluster (the kernel's cluster path needs at least 2 blocks)";
  else if (nw * nt != (uint64_t)block || block % 32 != 0)
    bad_dev = "the device's NUM_WARPS x NUM_THREADS must equal the block size (a whole number of warps)";
  else if (num_cores < (uint64_t)cluster || max_cluster < (uint64_t)cluster)
    bad_dev = "the device needs C cores and a cluster of C";
  else if (smem > lmem_cap)
    bad_dev = "the block's local memory exceeds a core's";
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
  CHECK(vx_module_get_kernel(mod, "rmsNormLamport", &k));

  vc::launch_config_t cfg;
  cfg.grid = vc::dim3((uint32_t)tokens, (uint32_t)cluster, 1);
  cfg.block = vc::dim3((uint32_t)block, 1, 1);
  cfg.cluster = vc::dim3(1, (uint32_t)cluster, 1);
  cfg.dyn_smem = smem;
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)smem));
  int max_active = 0;
  CHECK(vc::occupancy_max_active_clusters(dev, k, cfg, &max_active));
  std::printf("max_active_clusters = %d\n", max_active);
  if (max_active < 1) {
    std::printf("Error: the cluster cannot be placed\nFAILED!\n");
    return 1;
  }

  const size_t n = (size_t)tokens * dim;
  const size_t nblocks = (size_t)tokens * cluster;
  std::vector<T> h_in(n), h_res(n), h_gamma(dim);
  Buf d_in(n * sizeof(T)), d_res(n * sizeof(T)), d_gamma((size_t)dim * sizeof(T)), d_flags(64);
  Output pre(n * sizeof(T), "residual_out"), out(n * sizeof(T), "output"), smid(nblocks * sizeof(uint32_t), "smid");
  std::vector<uint8_t> zero_flags(64, 0);
  d_flags.write(zero_flags.data());

  LamportArgs args;
  std::memset(&args, 0, sizeof(args));
  args.outputPreNorm = pre.buf.addr;
  args.outputNorm = out.buf.addr;
  args.bufferInput = d_in.addr;
  args.gamma = d_gamma.addr;
  args.epsilon = kEps;
  args.residual = d_res.addr;
  args.numTokens = (uint32_t)tokens;
  args.dim = (uint32_t)dim;
  args.worldSize = 1;
  args.bufferFlags = d_flags.addr;
  args.smid = smid.buf.addr;

  int failed = 0;
  for (int rep = 0; rep < reps; ++rep) {
    std::mt19937_64 g(20261110ull + (uint64_t)rep);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const float scale = 1.0f + (float)rep;
    for (auto& v : h_in)
      v = T(nd(g) * scale);
    for (auto& v : h_res)
      v = T(nd(g) * scale);
    for (auto& v : h_gamma)
      v = T(nd(g));
    d_in.write(h_in.data());
    d_res.write(h_res.data());
    d_gamma.write(h_gamma.data());
    pre.reset();
    out.reset();
    smid.reset();

    vx_event_h ev = nullptr;
    CHECK(vc::launch_kernel_ex(q, k, cfg, args, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    const bool g1 = pre.fetch(rep), g2 = out.fetch(rep), g3 = smid.fetch(rep);
    const uint32_t* h_smid = reinterpret_cast<const uint32_t*>(smid.raw.data());

    bool cores = true;
    for (int t = 0; t < tokens; ++t) {
      std::set<uint32_t> seen;
      for (int b = 0; b < cluster; ++b) {
        const uint32_t s = h_smid[(size_t)t * cluster + b];
        if (s >= num_cores || !seen.insert(s).second) {
          std::printf("Error: rep %d: cluster %d block %d reports smid %u\n", rep, t, b, s);
          cores = false;
        }
      }
    }

    Tally t_pre, t_out;
    long long pre_bits = 0;
    double tight = 0;
    const T* got_pre = pre.data();
    const T* got_out = out.data();
    for (int t = 0; t < tokens; ++t) {
      std::vector<double> h(dim);
      std::vector<T> hb(dim);
      double ss = 0;
      for (int i = 0; i < dim; ++i) {
        const size_t j = (size_t)t * dim + i;
        hb[i] = h_in[j] + h_res[j];
        h[i] = (double)(float)hb[i];
        ss += h[i] * h[i];
      }
      const double rcp = 1.0 / std::sqrt(ss / dim + (double)kEps);
      for (int i = 0; i < dim; ++i) {
        const size_t j = (size_t)t * dim + i;
        const double ref = h[i] * rcp * (double)(float)h_gamma[i];
        t_pre.add((float)got_pre[j], h[i]);
        t_out.add((float)got_out[j], ref);
        pre_bits += got_pre[j].__x != hb[i].__x;
        const double ratio = std::fabs((double)(float)got_out[j] - ref) / (0x1p-7 * std::fabs(ref) + 1e-3);
        if (!(ratio <= tight))
          tight = ratio;
      }
    }
    const bool pass = g1 && g2 && g3 && cores && t_pre.bad == 0 && t_out.bad == 0 && pre_bits == 0 && tight <= 1.0;
    failed += !pass;
    std::printf("rep %d: %s bad_residual=%lld bad_out=%lld residual_bits_differ=%lld max_err_out=%.4g "
                "worst_out_err/tol=%.4g tight=%.4g\n",
                rep, pass ? "PASS" : "FAIL", t_pre.bad, t_out.bad, pre_bits, t_out.max_err, t_out.worst, tight);
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
