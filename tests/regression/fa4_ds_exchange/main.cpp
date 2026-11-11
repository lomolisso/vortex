// fa4_ds_exchange: the FlashAttention-4 two-CTA dS exchange
// (h100-dsmem-collectives/kernels/flash_bwd_2cta_ds_exchange__fa4.cu), one
// cluster of two 256-thread blocks on two cores.
//
//   -r <reps>   launches, each one checked (default 3)
//   -h          this help
//
// Launch i draws dS and K uniformly in [-1, 1] (mt19937_64 seeded
// 20261110 + i), rounded to f16, and zeroes dQ. Reference, in double:
//   dQ[64r + q][d] = sum over kv < 256 of dS[kv][64r + q] * K[kv][d].
// Each f16 x f16 product is exact in f32, so the only error is the f32
// accumulation over 256 terms (and the atomic add to zero, which is exact):
// |dQ - ref| <= 256 * 2^-24 * sum over kv of |dS[kv][.] * K[kv][d]|.
// A launch also fails if the 256-byte guard after dQ changes or if the two
// ranks do not report distinct cores of the device.
//
// Exit 0 when every launch passes, 1 otherwise or on a bad option.

#include <vortex2.h>
#include <vx_cuda_runtime.h>
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

constexpr int KV = 2 * TILE_N;
constexpr size_t kGuardBytes = 256;
constexpr uint32_t kGuardWord = 0x9e3779b9u;

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;
int reps = 3;

void usage() { std::printf("Usage: [-r reps] [-h]\n"); }

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:h")) != -1) {
    switch (c) {
    case 'r':
      reps = std::atoi(optarg);
      if (reps < 1) { usage(); std::exit(1); }
      break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(1);
    }
  }
  if (optind < argc) {
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

// Zero `body` bytes, then fill the rest of the buffer with the guard word.
std::vector<uint8_t> guarded(size_t total, size_t body) {
  std::vector<uint8_t> v(total, 0);
  for (size_t i = body; i + 4 <= total; i += 4)
    std::memcpy(&v[i], &kGuardWord, 4);
  return v;
}

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
  uint64_t num_cores = 0, nw = 0, nt = 0, lmem_cap = 0, max_cluster = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));
  const uint32_t smem = (uint32_t)sizeof(SharedStorage);
  std::printf("fa4_ds_exchange: reps %d; %llu cores, %llu warps x %llu threads, %llu bytes of local memory; "
              "%u bytes of shared memory per block\n",
              reps, (unsigned long long)num_cores, (unsigned long long)nw, (unsigned long long)nt,
              (unsigned long long)lmem_cap, smem);
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA | VX_ISA_EXT_L2CACHE;
  const char* bad_dev = nullptr;
  if ((isa & need) != need)
    bad_dev = "the device needs DLMEM, the A extension, the DXA and an L2";
  else if (nw * nt != kBlockThreads)
    bad_dev = "a block is 256 threads: the device needs NUM_WARPS x NUM_THREADS = 256";
  else if (num_cores < kClusterSize || max_cluster < kClusterSize)
    bad_dev = "the device needs 2 cores and a cluster of 2";
  else if (smem > lmem_cap)
    bad_dev = "the shared storage exceeds the local memory of a core";
  if (bad_dev) {
    std::printf("Error: %s\nFAILED!\n", bad_dev);
    vx_device_release(dev);
    return 1;
  }

  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority = VX_QUEUE_PRIORITY_NORMAL;
  CHECK(vx_queue_create(dev, &qi, &q));
  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  vx_kernel_h k = nullptr;
  CHECK(vx_module_get_kernel(mod, "flash_bwd_2cta_ds_exchange", &k));

  vc::launch_config_t cfg;
  cfg.grid = vc::dim3(kClusterSize, 1, 1);
  cfg.block = vc::dim3(kBlockThreads, 1, 1);
  cfg.dyn_smem = smem;
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)smem));
  int max_active = 0;
  CHECK(vc::occupancy_max_active_clusters<flash_bwd_2cta_ds_exchange_cluster_dims>(dev, k, cfg, &max_active));
  std::printf("max_active_clusters = %d\n", max_active);
  if (max_active < 1) {
    std::printf("Error: the cluster cannot be placed\nFAILED!\n");
    return 1;
  }

  const size_t ds_elems = (size_t)KV * TILE_M, k_elems = (size_t)KV * HEAD_D, dq_elems = (size_t)TILE_M * HEAD_D;
  const size_t dq_bytes = dq_elems * sizeof(float), smid_bytes = kClusterSize * sizeof(uint32_t);
  std::vector<__half> h_ds(ds_elems), h_k(k_elems);
  std::vector<float> h_dq(dq_elems);
  std::vector<uint32_t> h_smid(kClusterSize);
  Buf d_ds(ds_elems * sizeof(__half)), d_k(k_elems * sizeof(__half)), d_dq(dq_bytes + kGuardBytes),
      d_smid(smid_bytes + kGuardBytes);
  std::vector<uint8_t> raw_dq(d_dq.size), raw_smid(d_smid.size);

  kernel_arg_t arg;
  std::memset(&arg, 0, sizeof(arg));
  arg.dS_in = d_ds.addr;
  arg.K_in = d_k.addr;
  arg.dQ_out = d_dq.addr;
  arg.smid = d_smid.addr;

  int failed = 0;
  for (int rep = 0; rep < reps; ++rep) {
    std::mt19937_64 g(20261110ull + (unsigned long long)rep);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (auto& x : h_ds)
      x = __float2half_rn((float)u(g));
    for (auto& x : h_k)
      x = __float2half_rn((float)u(g));
    d_ds.write(h_ds.data());
    d_k.write(h_k.data());
    d_dq.write(guarded(d_dq.size, dq_bytes).data());
    std::vector<uint8_t> s0 = guarded(d_smid.size, smid_bytes);
    std::memset(s0.data(), 0xff, smid_bytes);
    d_smid.write(s0.data());

    vx_event_h ev = nullptr;
    CHECK(vc::launch_kernel_ex<flash_bwd_2cta_ds_exchange_cluster_dims>(q, k, cfg, arg, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    d_dq.read(raw_dq.data());
    d_smid.read(raw_smid.data());
    std::memcpy(h_dq.data(), raw_dq.data(), dq_bytes);
    std::memcpy(h_smid.data(), raw_smid.data(), smid_bytes);
    bool ok = guard_intact(raw_dq, dq_bytes, "dQ", rep) && guard_intact(raw_smid, smid_bytes, "smid", rep);

    long long bad = 0;
    double max_abs = 0, max_ratio = 0;
    for (int row = 0; row < TILE_M; ++row) {
      for (int d = 0; d < HEAD_D; ++d) {
        double ref = 0, mag = 0;
        for (int kv = 0; kv < KV; ++kv) {
          const double p = (double)__half2float(h_ds[(size_t)kv * TILE_M + row]) *
                           (double)__half2float(h_k[(size_t)kv * HEAD_D + d]);
          ref += p;
          mag += std::fabs(p);
        }
        const double got = h_dq[(size_t)row * HEAD_D + d];
        const double err = std::fabs(got - ref);
        const double tol = 256.0 * std::ldexp(1.0, -24) * mag;
        if (!(err <= tol)) {
          if (bad < 5)
            std::printf("Error: rep %d: dQ[%d][%d] = %.9g, expected %.9g (tolerance %.3g)\n", rep, row, d, got, ref,
                        tol);
          ++bad;
        }
        if (!(err <= max_abs))
          max_abs = err;
        if (tol > 0 && err / tol > max_ratio)
          max_ratio = err / tol;
      }
    }
    std::set<uint32_t> cores;
    for (uint32_t r = 0; r < kClusterSize; ++r) {
      if (h_smid[r] >= num_cores) {
        std::printf("Error: rep %d: rank %u reports smid %u, not one of the %llu cores\n", rep, r, h_smid[r],
                    (unsigned long long)num_cores);
        ok = false;
      }
      cores.insert(h_smid[r]);
    }
    const bool pass = ok && bad == 0 && cores.size() == kClusterSize;
    failed += !pass;
    std::printf("rep %d: %s bad=%lld max_abs=%.3g max_err_over_tol=%.3g smid_distinct=%zu\n", rep,
                pass ? "PASS" : "FAIL", bad, max_abs, max_ratio, cores.size());
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
