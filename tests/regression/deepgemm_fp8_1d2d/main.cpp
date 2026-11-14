// deepgemm_fp8_1d2d: DeepGEMM's SM90 FP8 1D2D m-grouped contiguous GEMM
// (sm90_fp8_gemm_1d2d_impl, block 128x192x128, 4 stages, multicast 2 on B),
// D[M][N] (bf16) = A[M][K] (e4m3, 1x128 fp32 scales) . B[g][N][K]^T (e4m3,
// 128x128 fp32 scales), one cluster of two blocks on two cores, persistent.
//
//   -m <rows>   rows per group, a multiple of 128 (default 256)
//   -r <reps>   launches, each with new inputs and checked (default 3)
//   -s <seed>   seed of rep 0; rep i uses seed + i (default 0)
//   -N          negative controls on rep 0's output (host only)
//   -h          this help
//
// N, K, the group count and the SM count are build-time (DG_SHAPE_N,
// DG_SHAPE_K, DG_NUM_GROUPS, DG_NUM_SMS).
//
// Inputs as the wrapper makes them: A and B bf16 standard normal, A cast to
// e4m3 per 1x128 block and B per 128x128 block (amax clamped to 1e-4,
// sf = amax / 448, x * fp32(1 / sf) rounded to fp32, then to e4m3), SFA
// moved once to the MN-major TMA-aligned layout, m_indices = group of each
// row. Checks per rep:
//   - the wrapper's: every output finite and calc_diff(D, ref) < 1e-3, ref =
//     the grouped product of the unquantised bf16 inputs (fp64, rounded to bf16);
//   - per element, against the fp8-dequantised fp64 reference ref8:
//     |D - ref8| <= ulp_bf16(max(|ref8|, 2^-126)) + 2^-14 * S, S = the sum of
//     |sfa * sfb * a8 * b8| over k;
//   - D's guard tail, and the smid and trace buffers' guards;
//   - the two blocks on distinct cores;
//   - the producer's tile trace equal to the host's schedule.
// -N corrupts the reference four ways (a k-block dropped, the B-scale split
// off by 8 column groups, block 1's tiles with the previous tile's B, all
// zeros); the per-element check must fail for each.
//
// Exit 0 when every rep passes (and every negative control is detected), 1
// otherwise or on a bad option.
//
// Departures from the H100 host: DeepGEMM's heuristic is not run (the H100's
// config is fixed and only N, K, G and the SM count shrink); SFA is moved to
// the TMA layout here, once, as the wrapper does; our own RNG instead of
// torch's; the reference is fp64 rounded to bf16 instead of cuBLAS bf16.

#include <vortex2.h>
#include <vx_cuda_runtime.h>
#include <VX_types.h>
#include "common.h"
#include <vx_cuda_fp.h>

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <unistd.h>
#include <vector>

namespace {

namespace vc = vx::cuda;
using namespace dg;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "Error %s:%d: '%s' returned %s\n",         \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

constexpr size_t kGuardBytes = 256;
constexpr uint32_t kGuardWord = 0x9e3779b9u;
constexpr uint16_t kBf16NaN = 0x7FC0;
constexpr uint32_t kKBlocks = kShapeK / kBlockK;
constexpr uint32_t kNumNBlocks = ceil_div(kShapeN, kBlockN);
constexpr uint32_t kShapeNSfb = ceil_div(kShapeN, kBlockK);
constexpr double kMaxDiff = 1e-3;

static_assert(kShapeK % kBlockK == 0, "K is a multiple of 128");
static_assert(kShapeN % 64 == 0, "N is a multiple of the 64-column store box");
static_assert(kNumGroups >= 1 && kNumSMs == 2, "one cluster of 2 blocks");

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;
uint32_t rows_per_group = 256;
int reps = 3;
uint64_t seed0 = 0;
bool negatives = false;

void usage() { std::printf("Usage: [-m rows per group] [-r reps] [-s seed] [-N] [-h]\n"); }

// Decimal digits only, the whole string, no overflow.
bool parse_dec(const char* s, uint64_t max, uint64_t* v) {
  if (!std::isdigit((unsigned char)*s))
    return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long x = std::strtoull(s, &end, 10);
  if (*end || errno == ERANGE || x > max)
    return false;
  *v = x;
  return true;
}

void parse_args(int argc, char** argv) {
  int c;
  bool ok = true;
  uint64_t v = 0;
  while ((c = getopt(argc, argv, "m:r:s:Nh")) != -1) {
    switch (c) {
    case 'm':
      ok = ok && parse_dec(optarg, 1000000, &v) && v >= 128 && v % 128 == 0;
      rows_per_group = (uint32_t)v;
      break;
    case 'r':
      ok = ok && parse_dec(optarg, 1000000, &v) && v >= 1;
      reps = (int)v;
      break;
    case 's': ok = ok && parse_dec(optarg, UINT64_MAX, &seed0); break;
    case 'N': negatives = true; break;
    case 'h': usage(); std::exit(0);
    default:  ok = false; break;
    }
  }
  if (!ok || optind < argc) {
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

// The body filled with `fill` bytes (or left as given), the tail with the guard word.
std::vector<uint8_t> guarded(size_t total, size_t body, const void* src, int fill) {
  std::vector<uint8_t> v(total, 0);
  if (src)
    std::memcpy(v.data(), src, body);
  else
    std::memset(v.data(), fill, body);
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

uint16_t bf16_bits(float f) {
  const __nv_bfloat16 h(f);
  return __builtin_bit_cast(uint16_t, h);
}
float bf16_float(uint16_t b) { return (float)__builtin_bit_cast(__nv_bfloat16, b); }
uint8_t e4m3_bits(float f) {
  const __nv_fp8_e4m3 v(f);
  return __builtin_bit_cast(uint8_t, v);
}
float e4m3_float(uint8_t b) { return (float)__builtin_bit_cast(__nv_fp8_e4m3, b); }

// One unit in the last place of a bf16 of magnitude x (x >= 2^-126).
double ulp_bf16(double x) {
  int e = 0;
  std::frexp(x, &e);
  return std::ldexp(1.0, e - 8);
}

// The tiles block `cta` runs, in order (scheduler/gemm.cuh, m-grouped
// contiguous, multicast on B).
struct Tile {
  uint32_t m, n, mcast;
};
constexpr uint32_t blocks_per_group() {
  const uint32_t u8 = 8 * kBlockM + ceil_div(kNumSMs, 8) * kBlockN;
  const uint32_t u16 = 16 * kBlockM + ceil_div(kNumSMs, 16) * kBlockN;
  return u16 < u8 ? 16 : 8;
}
std::vector<Tile> schedule(uint32_t cta, uint32_t shape_m, const std::vector<int32_t>& layout) {
  const uint32_t num_m = ceil_div(shape_m, kBlockM), num_n = kNumNBlocks, num_blocks = num_m * num_n;
  const uint32_t width = blocks_per_group();
  std::vector<Tile> t;
  for (uint32_t iter = 0;; ++iter) {
    const uint32_t idx = iter * kNumSMs + cta;
    if (idx >= num_blocks)
      break;
    const uint32_t per_group = num_n * width;
    uint32_t first = (idx / per_group) * width, in = idx % per_group;
    uint32_t nbig = std::min(width, num_m - first);
    if (nbig % 2 != 0) {
      if (in < (nbig ^ 1) * num_n) {
        nbig ^= 1;
      } else {
        in -= (nbig ^ 1) * num_n;
        first += nbig ^ 1;
        nbig = 1;
      }
    }
    Tile x;
    x.m = first + in % nbig;
    x.n = in / nbig;
    x.mcast = (nbig != 1 && layout[x.m * kBlockM] == layout[(x.m ^ 1) * kBlockM]) ? 1u : 0u;
    t.push_back(x);
  }
  return t;
}

struct Problem {
  uint32_t M = 0, Mal = 0;
  std::vector<uint16_t> a, b;          // bf16 [M][K], [G][N][K]
  std::vector<uint8_t> a8, b8;         // e4m3, same shapes
  std::vector<float> a8f, b8f;         // their values
  std::vector<float> sfa;              // [M][K/128]
  std::vector<float> sfa_tma;          // [K/128][Mal]
  std::vector<float> sfb;              // [G][N/128][K/128]
  std::vector<int32_t> layout;         // [M]
};

void make_inputs(Problem& p, uint64_t seed) {
  const uint32_t M = p.M, N = kShapeN, K = kShapeK, G = kNumGroups;
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  p.a.resize((size_t)M * K);
  p.b.resize((size_t)G * N * K);
  for (auto& x : p.a)
    x = bf16_bits((float)nd(rng));
  for (auto& x : p.b)
    x = bf16_bits((float)nd(rng));

  // amax in fp32, clamped; sf = amax / 448; x * fp32(1 / sf) to e4m3.
  auto quant_block = [](float amax, float& sf, float& inv) {
    amax = std::max(amax, 1e-4f);
    sf = amax / 448.0f;
    inv = 1.0f / sf;
  };
  p.a8.resize(p.a.size());
  p.a8f.resize(p.a.size());
  p.sfa.assign((size_t)M * kKBlocks, 0.f);
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t kb = 0; kb < kKBlocks; ++kb) {
      float amax = 0.f;
      for (uint32_t k = kb * kBlockK; k < (kb + 1) * kBlockK; ++k)
        amax = std::max(amax, std::fabs(bf16_float(p.a[(size_t)m * K + k])));
      float sf, inv;
      quant_block(amax, sf, inv);
      p.sfa[(size_t)m * kKBlocks + kb] = sf;
      for (uint32_t k = kb * kBlockK; k < (kb + 1) * kBlockK; ++k) {
        const size_t i = (size_t)m * K + k;
        const float prod = bf16_float(p.a[i]) * inv;
        p.a8[i] = e4m3_bits(prod);
        p.a8f[i] = e4m3_float(p.a8[i]);
      }
    }
  }
  p.b8.resize(p.b.size());
  p.b8f.resize(p.b.size());
  p.sfb.assign((size_t)G * kShapeNSfb * kKBlocks, 0.f);
  for (uint32_t g = 0; g < G; ++g) {
    for (uint32_t nb = 0; nb < kShapeNSfb; ++nb) {
      for (uint32_t kb = 0; kb < kKBlocks; ++kb) {
        float amax = 0.f;
        for (uint32_t n = nb * kBlockK; n < std::min(N, (nb + 1) * kBlockK); ++n)
          for (uint32_t k = kb * kBlockK; k < (kb + 1) * kBlockK; ++k)
            amax = std::max(amax, std::fabs(bf16_float(p.b[((size_t)g * N + n) * K + k])));
        float sf, inv;
        quant_block(amax, sf, inv);
        p.sfb[((size_t)g * kShapeNSfb + nb) * kKBlocks + kb] = sf;
        for (uint32_t n = nb * kBlockK; n < std::min(N, (nb + 1) * kBlockK); ++n) {
          for (uint32_t k = kb * kBlockK; k < (kb + 1) * kBlockK; ++k) {
            const size_t i = ((size_t)g * N + n) * K + k;
            const float prod = bf16_float(p.b[i]) * inv;
            p.b8[i] = e4m3_bits(prod);
            p.b8f[i] = e4m3_float(p.b8[i]);
          }
        }
      }
    }
  }
  p.sfa_tma.assign((size_t)kKBlocks * p.Mal, 0.f);
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t kb = 0; kb < kKBlocks; ++kb)
      p.sfa_tma[(size_t)kb * p.Mal + m] = p.sfa[(size_t)m * kKBlocks + kb];
  p.layout.resize(M);
  for (uint32_t m = 0; m < M; ++m)
    p.layout[m] = (int32_t)(m / rows_per_group);
}

// Per (m, n, kb): the exact fp64 dot product of the e4m3 values over the
// k-block and the sum of the magnitudes of its terms.
struct Partials {
  std::vector<double> dot, mag;
};

void make_partials(const Problem& p, Partials& r) {
  const uint32_t M = p.M, N = kShapeN, K = kShapeK;
  r.dot.assign((size_t)M * N * kKBlocks, 0.0);
  r.mag.assign((size_t)M * N * kKBlocks, 0.0);
  for (uint32_t m = 0; m < M; ++m) {
    const uint32_t g = (uint32_t)p.layout[m];
    const float* ar = &p.a8f[(size_t)m * K];
    for (uint32_t n = 0; n < N; ++n) {
      const float* br = &p.b8f[((size_t)g * N + n) * K];
      for (uint32_t kb = 0; kb < kKBlocks; ++kb) {
        double s = 0, a = 0;
        for (uint32_t k = kb * kBlockK; k < (kb + 1) * kBlockK; ++k) {
          const double t = (double)ar[k] * (double)br[k];
          s += t;
          a += std::fabs(t);
        }
        r.dot[((size_t)m * N + n) * kKBlocks + kb] = s;
        r.mag[((size_t)m * N + n) * kKBlocks + kb] = a;
      }
    }
  }
}

// The wrapper's reference: the grouped bf16 product, in fp64, rounded to bf16.
void make_ref_bf16(const Problem& p, std::vector<double>& ref) {
  const uint32_t M = p.M, N = kShapeN, K = kShapeK;
  std::vector<float> af(p.a.size()), bf(p.b.size());
  for (size_t i = 0; i < af.size(); ++i)
    af[i] = bf16_float(p.a[i]);
  for (size_t i = 0; i < bf.size(); ++i)
    bf[i] = bf16_float(p.b[i]);
  ref.assign((size_t)M * N, 0.0);
  for (uint32_t m = 0; m < M; ++m) {
    const uint32_t g = (uint32_t)p.layout[m];
    for (uint32_t n = 0; n < N; ++n) {
      double s = 0;
      const float* ar = &af[(size_t)m * K];
      const float* br = &bf[((size_t)g * N + n) * K];
      for (uint32_t k = 0; k < K; ++k)
        s += (double)ar[k] * (double)br[k];
      ref[(size_t)m * N + n] = (double)bf16_float(bf16_bits((float)s));
    }
  }
}

// The B-scale row column n of n-block nb takes, for a given split point
// `former` (in 8-column groups), as the kernel picks it.
uint32_t sfb_row(uint32_t n, uint32_t former) {
  const uint32_t nb = n / kBlockN, gi = (n % kBlockN) / 8;
  return (nb * kBlockN) / kBlockK + (gi < former ? 0 : 1);
}
uint32_t former_iters(uint32_t nb) {
  return std::min(kBlockN, kBlockK - nb * kBlockN % kBlockK) / 8;
}

// ref8 and S for element (m, n), with optional corruptions.
struct Corrupt {
  int drop_kb = -1;          // k-block left out
  int former_shift = 0;      // added to num_former_iters
};
void ref8_elem(const Problem& p, const Partials& r, uint32_t m, uint32_t n, const Corrupt& c, double& ref,
               double& s) {
  const uint32_t g = (uint32_t)p.layout[m];
  const uint32_t row = sfb_row(n, (uint32_t)((int)former_iters(n / kBlockN) + c.former_shift));
  ref = 0;
  s = 0;
  for (uint32_t kb = 0; kb < kKBlocks; ++kb) {
    if ((int)kb == c.drop_kb)
      continue;
    const double sc = (double)p.sfa[(size_t)m * kKBlocks + kb] *
                      (double)p.sfb[((size_t)g * kShapeNSfb + row) * kKBlocks + kb];
    const size_t i = ((size_t)m * kShapeN + n) * kKBlocks + kb;
    ref += sc * r.dot[i];
    s += std::fabs(sc) * r.mag[i];
  }
}

double bound_of(double ref, double s) {
  return ulp_bf16(std::max(std::fabs(ref), std::ldexp(1.0, -126))) + std::ldexp(1.0, -14) * s;
}

// Row `rowg` of B's TMA view [G*N][K] (zero past the last group).
const float* b_row(const Problem& p, uint32_t rowg, const std::vector<float>& zero) {
  return rowg < kNumGroups * kShapeN ? &p.b8f[(size_t)rowg * kShapeK] : zero.data();
}

// Count of elements of `out` outside the per-element bound against a
// reference built by `refof(m, n, ref, s)`.
template <typename F>
long long count_bad(const Problem& p, const std::vector<double>& out, F refof, double* max_ratio, int rep,
                    bool report) {
  long long bad = 0;
  double mr = 0;
  for (uint32_t m = 0; m < p.M; ++m) {
    for (uint32_t n = 0; n < kShapeN; ++n) {
      double ref, s;
      refof(m, n, ref, s);
      const double got = out[(size_t)m * kShapeN + n];
      const double bnd = bound_of(ref, s);
      const double err = std::fabs(got - ref);
      if (!(err <= bnd)) {
        if (report && bad < 5)
          std::printf("Error: rep %d: D[%u][%u] = %.9g, ref8 %.9g, bound %.3g\n", rep, m, n, got, ref, bnd);
        ++bad;
      }
      const double ratio = err / bnd;
      if (!std::isnan(mr) && !(ratio <= mr))
        mr = ratio;
    }
  }
  if (max_ratio)
    *max_ratio = mr;
  return bad;
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

  Problem p;
  p.M = rows_per_group * kNumGroups;
  p.Mal = align_up(p.M, 4);
  const uint32_t M = p.M, N = kShapeN, K = kShapeK;
  const uint32_t smem = kSmemSize;
  std::printf("deepgemm_fp8_1d2d: m %u x %u groups (M %u), N %u, K %u, %u SMs; reps %d, seed %llu%s; "
              "%llu cores, %llu warps x %llu threads, %llu bytes of local memory; %u bytes of shared memory\n",
              rows_per_group, kNumGroups, M, N, K, kNumSMs, reps, (unsigned long long)seed0,
              negatives ? ", negative controls" : "", (unsigned long long)num_cores, (unsigned long long)nw,
              (unsigned long long)nt, (unsigned long long)lmem_cap, smem);
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA | VX_ISA_EXT_L2CACHE;
  const char* bad_dev = nullptr;
  if ((isa & need) != need)
    bad_dev = "the device needs DLMEM, the A extension, the DXA and an L2";
  else if (nt != 32 || nw * nt < kNumThreads)
    bad_dev = "a block is 896 threads of 32-thread warps";
  else if (num_cores < kNumSMs || max_cluster < kNumTMAMulticast)
    bad_dev = "the device needs 2 cores and a cluster of 2";
  else if (smem > lmem_cap)
    bad_dev = "the shared memory exceeds the local memory of a core";
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
  CHECK(vx_module_get_kernel(mod, "sm90_fp8_gemm_1d2d", &k));

  // The H100 host passes the cluster with the launch (the kernel has no
  // __cluster_dims__).
  vc::launch_config_t cfg;
  cfg.grid = vc::dim3(kNumSMs, 1, 1);
  cfg.block = vc::dim3(kNumThreads, 1, 1);
  cfg.cluster = vc::dim3(kNumTMAMulticast, 1, 1);
  cfg.dyn_smem = smem;
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)smem));
  int max_active = 0;
  CHECK(vc::occupancy_max_active_clusters(dev, k, cfg, &max_active));
  std::printf("max_active_clusters = %d\n", max_active);
  if (max_active < 1) {
    std::printf("Error: the cluster cannot be placed\nFAILED!\n");
    return 1;
  }

  const size_t a_bytes = (size_t)M * K, b_bytes = (size_t)kNumGroups * N * K;
  const size_t sfa_bytes = (size_t)kKBlocks * p.Mal * 4, sfb_bytes = (size_t)kNumGroups * kShapeNSfb * kKBlocks * 4;
  const size_t lay_bytes = (size_t)M * 4, d_bytes = (size_t)M * N * 2;
  const size_t smid_bytes = kNumSMs * 4, trace_bytes = (size_t)kNumSMs * kTraceCap * sizeof(trace_rec_t);
  Buf d_a(a_bytes), d_b(b_bytes), d_sfa(sfa_bytes), d_sfb(sfb_bytes), d_lay(lay_bytes);
  Buf d_d(d_bytes + kGuardBytes), d_smid(smid_bytes + kGuardBytes), d_trace(trace_bytes + kGuardBytes);
  std::vector<uint8_t> raw_d(d_d.size), raw_smid(d_smid.size), raw_trace(d_trace.size);

  kernel_arg_t arg;
  std::memset(&arg, 0, sizeof(arg));
  {
    namespace c = vx::cuda;
    const uint32_t es[2] = {1, 1};
    const uint64_t a_dim[2] = {K, M}, a_str[1] = {K};
    const uint32_t a_box[2] = {kBlockK, kBlockM};
    CHECK(c::tensor_map_encode_tiled(&arg.tensor_map_a, c::CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, d_a.addr, a_dim, a_str,
                                     a_box, es, c::CU_TENSOR_MAP_INTERLEAVE_NONE, c::CU_TENSOR_MAP_SWIZZLE_128B,
                                     c::CU_TENSOR_MAP_L2_PROMOTION_L2_256B, c::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    const uint64_t b_dim[2] = {K, (uint64_t)N * kNumGroups}, b_str[1] = {K};
    const uint32_t b_box[2] = {kBlockK, kBlockN};
    CHECK(c::tensor_map_encode_tiled(&arg.tensor_map_b, c::CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, d_b.addr, b_dim, b_str,
                                     b_box, es, c::CU_TENSOR_MAP_INTERLEAVE_NONE, c::CU_TENSOR_MAP_SWIZZLE_128B,
                                     c::CU_TENSOR_MAP_L2_PROMOTION_L2_256B, c::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    // The inner box is the 128-byte swizzle span: 64 bf16 of the 192.
    const uint64_t d_dim[2] = {N, M}, d_str[1] = {(uint64_t)N * 2};
    const uint32_t d_box[2] = {kSwizzleDMode / 2, kBlockM};
    CHECK(c::tensor_map_encode_tiled(&arg.tensor_map_d, c::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, d_d.addr, d_dim,
                                     d_str, d_box, es, c::CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     c::CU_TENSOR_MAP_SWIZZLE_128B, c::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     c::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    const uint64_t s_dim[2] = {p.Mal, kKBlocks}, s_str[1] = {(uint64_t)p.Mal * 4};
    const uint32_t s_box[2] = {kBlockM, 1};
    CHECK(c::tensor_map_encode_tiled(&arg.tensor_map_sfa, c::CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, d_sfa.addr, s_dim,
                                     s_str, s_box, es, c::CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     c::CU_TENSOR_MAP_SWIZZLE_NONE, c::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                     c::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  arg.sfb = d_sfb.addr;
  arg.grouped_layout = d_lay.addr;
  arg.shape_m = M;
  arg.shape_n = N;
  arg.shape_k = K;
  arg.smid = d_smid.addr;
  arg.trace = d_trace.addr;

  // The schedule depends on the layout only, the same in every rep.
  std::vector<int32_t> lay0(M);
  for (uint32_t m = 0; m < M; ++m)
    lay0[m] = (int32_t)(m / rows_per_group);
  std::vector<Tile> sched[kNumSMs];
  for (uint32_t c = 0; c < kNumSMs; ++c) {
    sched[c] = schedule(c, M, lay0);
    std::printf("schedule cta %u:", c);
    for (size_t i = 0; i < sched[c].size(); ++i)
      std::printf(" (%u,%u)%s", sched[c][i].m, sched[c][i].n, sched[c][i].mcast ? "" : "*");
    std::printf("   (* = no multicast)\n");
    if (sched[c].size() > kTraceCap) {
      std::printf("Error: %zu tiles exceed the trace of %u\nFAILED!\n", sched[c].size(), kTraceCap);
      return 1;
    }
  }

  std::vector<double> out((size_t)M * N), ref;
  Partials parts;
  int failed = 0;
  bool neg_ok = true;
  for (int rep = 0; rep < reps; ++rep) {
    make_inputs(p, seed0 + (uint64_t)rep);
    d_a.write(p.a8.data());
    d_b.write(p.b8.data());
    d_sfa.write(p.sfa_tma.data());
    d_sfb.write(p.sfb.data());
    d_lay.write(p.layout.data());
    {
      std::vector<uint16_t> nan_body((size_t)M * N, kBf16NaN);
      d_d.write(guarded(d_d.size, d_bytes, nan_body.data(), 0).data());
    }
    d_smid.write(guarded(d_smid.size, smid_bytes, nullptr, 0xff).data());
    d_trace.write(guarded(d_trace.size, trace_bytes, nullptr, 0xff).data());

    vx_event_h ev = nullptr;
    CHECK(vc::launch_kernel_ex(q, k, cfg, arg, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    d_d.read(raw_d.data());
    d_smid.read(raw_smid.data());
    d_trace.read(raw_trace.data());
    bool ok = guard_intact(raw_d, d_bytes, "D", rep) & guard_intact(raw_smid, smid_bytes, "smid", rep) &
              guard_intact(raw_trace, trace_bytes, "trace", rep);

    bool finite = true;
    for (size_t i = 0; i < out.size(); ++i) {
      uint16_t bits;
      std::memcpy(&bits, &raw_d[2 * i], 2);
      out[i] = (double)bf16_float(bits);
      finite = finite && std::isfinite(out[i]);
    }
    if (!finite)
      std::printf("Error: rep %d: D has non-finite elements\n", rep);

    make_ref_bf16(p, ref);
    double sxy = 0, sxx = 0;
    for (size_t i = 0; i < out.size(); ++i) {
      sxy += out[i] * ref[i];
      sxx += out[i] * out[i] + ref[i] * ref[i];
    }
    const double diff = sxx == 0 ? 0.0 : 1.0 - 2.0 * sxy / sxx;

    make_partials(p, parts);
    double max_ratio = 0;
    const Corrupt none;
    const long long bad = count_bad(p, out, [&](uint32_t m, uint32_t n, double& r, double& s) {
      ref8_elem(p, parts, m, n, none, r, s);
    }, &max_ratio, rep, true);

    uint32_t smid[kNumSMs];
    std::memcpy(smid, raw_smid.data(), smid_bytes);
    for (uint32_t c = 0; c < kNumSMs; ++c) {
      if (smid[c] >= num_cores) {
        std::printf("Error: rep %d: block %u reports smid %u, not one of the %llu cores\n", rep, c, smid[c],
                    (unsigned long long)num_cores);
        ok = false;
      }
    }
    if (smid[0] == smid[1]) {
      std::printf("Error: rep %d: both blocks report core %u\n", rep, smid[0]);
      ok = false;
    }

    uint32_t mc_got = 0, tiles = 0;
    const trace_rec_t* tr = (const trace_rec_t*)raw_trace.data();
    for (uint32_t c = 0; c < kNumSMs; ++c) {
      for (uint32_t i = 0; i < kTraceCap; ++i) {
        const trace_rec_t& r = tr[c * kTraceCap + i];
        if (i < sched[c].size()) {
          const Tile& e = sched[c][i];
          ++tiles;
          mc_got += (r.mcast == 1);
          if (r.iter != i || r.m_block != e.m || r.n_block != e.n || r.mcast != e.mcast) {
            std::printf("Error: rep %d: block %u tile %u traced {%u, %u, %u, %u}, expected {%u, %u, %u, %u}\n", rep,
                        c, i, r.iter, r.m_block, r.n_block, r.mcast, i, e.m, e.n, e.mcast);
            ok = false;
          }
        } else if (r.iter != 0xffffffffu || r.m_block != 0xffffffffu || r.n_block != 0xffffffffu ||
                   r.mcast != 0xffffffffu) {
          std::printf("Error: rep %d: block %u traced an extra tile %u\n", rep, c, i);
          ok = false;
        }
      }
    }

    const bool pass = ok && finite && diff < kMaxDiff && bad == 0;
    failed += !pass;
    std::printf("rep %d: %s calc_diff=%.3e max_err_over_bound=%.3g mcast=%u/%u\n", rep, pass ? "PASS" : "FAIL",
                diff, max_ratio, mc_got, tiles);

    if (negatives && rep == 0) {
      // (1) one k-block dropped (the last).
      Corrupt c1;
      c1.drop_kb = (int)kKBlocks - 1;
      const long long b1 = count_bad(p, out, [&](uint32_t m, uint32_t n, double& r, double& s) {
        ref8_elem(p, parts, m, n, c1, r, s);
      }, nullptr, rep, false);
      // (2) num_former_iters 8 column groups larger: one n-third per n-block
      // takes the other B-scale row.
      Corrupt c2;
      c2.former_shift = 8;
      const long long b2 = count_bad(p, out, [&](uint32_t m, uint32_t n, double& r, double& s) {
        ref8_elem(p, parts, m, n, c2, r, s);
      }, nullptr, rep, false);
      // (3) block 1's tiles after its first with the previous tile's B box
      // (a stale multicast), with the tile's own scales.
      std::vector<double> ref3((size_t)M * N, NAN), s3((size_t)M * N, NAN);
      const std::vector<float> zero(K, 0.f);
      for (size_t j = 1; j < sched[1].size(); ++j) {
        const Tile& cur = sched[1][j];
        const Tile& prev = sched[1][j - 1];
        const uint32_t gp = (uint32_t)lay0[prev.m * kBlockM];
        for (uint32_t mm = 0; mm < kBlockM; ++mm) {
          const uint32_t m = cur.m * kBlockM + mm;
          const uint32_t g = (uint32_t)p.layout[m];
          for (uint32_t cc = 0; cc < kBlockN; ++cc) {
            const uint32_t n = cur.n * kBlockN + cc;
            if (n >= N)
              continue;
            const float* br = b_row(p, gp * N + prev.n * kBlockN + cc, zero);
            const float* ar = &p.a8f[(size_t)m * K];
            const uint32_t row = sfb_row(n, former_iters(cur.n));
            double r = 0, s = 0;
            for (uint32_t kb = 0; kb < kKBlocks; ++kb) {
              double dot = 0, mag = 0;
              for (uint32_t kk = kb * kBlockK; kk < (kb + 1) * kBlockK; ++kk) {
                const double t = (double)ar[kk] * (double)br[kk];
                dot += t;
                mag += std::fabs(t);
              }
              const double sc = (double)p.sfa[(size_t)m * kKBlocks + kb] *
                                (double)p.sfb[((size_t)g * kShapeNSfb + row) * kKBlocks + kb];
              r += sc * dot;
              s += std::fabs(sc) * mag;
            }
            ref3[(size_t)m * N + n] = r;
            s3[(size_t)m * N + n] = s;
          }
        }
      }
      const long long b3 = count_bad(p, out, [&](uint32_t m, uint32_t n, double& r, double& s) {
        const size_t i = (size_t)m * N + n;
        if (std::isnan(ref3[i]))
          ref8_elem(p, parts, m, n, none, r, s);
        else {
          r = ref3[i];
          s = s3[i];
        }
      }, nullptr, rep, false);
      // (4) all zeros.
      const long long b4 = count_bad(p, out, [&](uint32_t, uint32_t, double& r, double& s) {
        r = 0;
        s = 0;
      }, nullptr, rep, false);
      const long long nb[4] = {b1, b2, b3, b4};
      for (int i = 0; i < 4; ++i) {
        std::printf("neg %d: %s (%lld elements outside the bound)\n", i + 1, nb[i] > 0 ? "detected" : "MISSED",
                    nb[i]);
        neg_ok = neg_ok && nb[i] > 0;
      }
    }
  }

  vx_kernel_release(k);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);
  if (failed || !neg_ok) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
