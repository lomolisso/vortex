// dak_gemv_multicast: DAK's multicast GEMV kernel (arXiv 2604.26074) as the
// H100 ran it, C[T][M] = X[T][K] . W[K][M] in fp16 with fp16 accumulation,
// one cluster of 2 blocks; rank 0 multicasts the weight tiles to both blocks
// and each block loads its own 256 tokens. Checked as the H100 harness checks
// it (h100-workloads harness/wrappers/dak_gemm_multicast.py), plus a bit-exact
// check against the tensor unit's per-step fp16 model.
//
//   -t T     tokens (only 512: two 256-token tiles, one cluster)
//   -m M     out features, a multiple of 128 (default 256)
//   -k K     in features, a multiple of 64 (default 448)
//   -r R     launches, each on new inputs and checked (default 3)
//   -s S     seed of launch 0; launch i uses S + i (default 0)
//   -N       negative controls on launch 0's output (needs K >= 256)
//   -d       debug: the kernel dumps every descriptor it builds for slots 0-2
//            and the host checks each one
//   -h       this help
//
// Inputs (the wrapper's distributions): x ~ N(0, 1) rounded to fp16 [T][K];
// W = (U(0,1) * 2 - 1) / sqrt(K) in fp32 rounded once to fp16, drawn as
// [M][K] (nn.Linear) and stored [K][M].
//
// Checks per launch:
// - the wrapper's rule: every output finite and, for every 256 (tokens) x 64
//   (m) tile, ||C - ref||_F / ||ref||_F <= 4 sqrt(K / 16) 2^-11, with ref =
//   x . W in fp32 and NaNs counted as 1e4;
// - bit-exact: out[t][m] equals the fp16 model of the tensor unit, c = 0 then,
//   k-tile by k-tile, k16 step by k16 step and k8 half by k8 half,
//   c = FEDP(W[k0..k0+7][m], X[t][k0..k0+7], c) rounded to fp16;
// - before the launch the output body is fp16 NaN, the placement records are
//   0xff bytes, and every output has a tail of guard words, which must survive;
// - the two blocks run on distinct cores, each with thread 0 on a hardware warp
//   that is a multiple of 4 (warpgroups 4-aligned).
// -N corrupts the references of launch 0 and requires each check to catch
// it: (a) fp32 accumulation rounded once (bit-exact), (b) the two k8 halves of
// every k16 step swapped (bit-exact), (c) one k-tile dropped and one reused
// from a stale slot (both), (d) rank 1's tokens with the other group's A rows
// (both), (e) all zeros (both).
// Exit 0 only when every launch passes and no control is missed; 1 on a
// failure or a bad option.

#include <vortex2.h>
#include <vx_cuda_runtime.h>
#include <vx_cuda_fp.h>
#include <VX_types.h>
#include <tcu_fedp_ref.h>
#include "common.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace vc = vx::cuda;
namespace tf = vortex::tcu_fedp_ref;
using namespace dak;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "Error %s:%d: '%s' returned %s\n",         \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

constexpr uint32_t kGrid = 2;
constexpr unsigned kRefThreads = 4;

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;
uint64_t num_cores = 0;

struct Opts {
  uint32_t T = 512, M = 256, K = 448;
  int reps = 3;
  unsigned long long seed = 0;
  bool neg = false, dump = false;
} o;

void usage() {
  std::printf("Usage: [-t 512] [-m M (multiple of 128)] [-k K (multiple of 64)] [-r reps] [-s seed] [-N] [-d] [-h]\n");
}

// Plain decimal digits only: strtoull alone accepts leading blanks, a sign
// (negating "-1" modulo 2^64) and saturates on overflow.
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

bool parse_u32(const char* s, uint32_t* v) {
  uint64_t x = 0;
  if (!parse_dec(s, 1000000, &x) || x == 0)
    return false;
  *v = (uint32_t)x;
  return true;
}

void parse_args(int argc, char** argv) {
  int c;
  bool ok = true;
  while ((c = getopt(argc, argv, "t:m:k:r:s:Ndh")) != -1) {
    switch (c) {
    case 't': ok &= parse_u32(optarg, &o.T); break;
    case 'm': ok &= parse_u32(optarg, &o.M); break;
    case 'k': ok &= parse_u32(optarg, &o.K); break;
    case 'r': {
      uint32_t r = 0;
      ok &= parse_u32(optarg, &r);
      o.reps = (int)r;
      break;
    }
    case 's': {
      uint64_t s = 0;
      ok &= parse_dec(optarg, UINT64_MAX, &s);
      o.seed = s;
      break;
    }
    case 'N': o.neg = true; break;
    case 'd': o.dump = true; break;
    case 'h': usage(); std::exit(0);
    default: ok = false; break;
    }
  }
  // Grid = (T / TILE_N) x num_copy_blocks with one cluster: T is 512.
  ok &= optind == argc && o.T == 2 * (uint32_t)TILE_N && o.M % (2 * TILE_M) == 0 && o.K % TILE_K == 0
        && o.reps >= 1;
  // The stale-slot control reuses the k-tile three uses back.
  ok &= !o.neg || o.K >= 4 * (uint32_t)TILE_K;
  if (!ok) {
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
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
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

// Output buffers carry a tail of guard words, rewritten before every launch.
constexpr size_t kGuardBytes = 256;
constexpr uint32_t kGuardWord = 0x9e3779b9u;
constexpr uint32_t kHalfNaN2 = 0x7e007e00u;
constexpr uint32_t kNoRecord = 0xffffffffu;

struct Out {
  size_t body;
  Buf buf;
  std::vector<uint8_t> raw;
  explicit Out(size_t b) : body(b), buf(b + kGuardBytes), raw(b + kGuardBytes) {}
  void reset(uint32_t fill) {
    for (size_t i = 0; i + 4 <= body; i += 4)
      std::memcpy(&raw[i], &fill, 4);
    for (size_t i = body; i + 4 <= raw.size(); i += 4)
      std::memcpy(&raw[i], &kGuardWord, 4);
    buf.write(raw.data());
  }
  bool fetch(const char* what, int rep) {
    buf.read(raw.data());
    for (size_t i = body; i + 4 <= raw.size(); i += 4) {
      uint32_t w;
      std::memcpy(&w, &raw[i], 4);
      if (w != kGuardWord) {
        std::printf("Error: rep %d: guard word %zu after %s changed: 0x%08x\n", rep, (i - body) / 4, what, w);
        return false;
      }
    }
    return true;
  }
  template <typename T>
  std::vector<T> as() const {
    std::vector<T> v(body / sizeof(T));
    std::memcpy(v.data(), raw.data(), body);
    return v;
  }
};

void parallel_rows(uint32_t rows, const std::function<void(uint32_t)>& fn) {
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < kRefThreads; ++i)
    pool.emplace_back([&, i] {
      for (uint32_t r = i; r < rows; r += kRefThreads)
        fn(r);
    });
  for (auto& t : pool)
    t.join();
}

inline float h2f(uint16_t b) { return (float)__half(__half_raw{b}); }

// One problem: inputs as fp16 bits, x [T][K] and W [K][M], and the k8 words
// the tensor unit consumes (two halves per word, lowest k in bits 15:0).
struct Problem {
  uint32_t T, M, K, KT;
  std::vector<uint16_t> x, w;
  std::vector<uint32_t> xp, wp;   // xp[t][k8][4], wp[m][k8][4]
  std::vector<float> xf, wf;      // as floats, x [T][K], W [M][K]

  Problem(uint32_t t, uint32_t m, uint32_t k) : T(t), M(m), K(k), KT(k / TILE_K) {}

  void generate(unsigned long long seed) {
    std::mt19937_64 g(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    x.assign((size_t)T * K, 0);
    w.assign((size_t)K * M, 0);
    for (auto& v : x)
      v = __half(nd(g)).__x;
    const float sk = (float)std::sqrt((double)K);
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t k = 0; k < K; ++k)
        w[(size_t)k * M + m] = __half((ud(g) * 2.0f - 1.0f) / sk).__x;
    const uint32_t k8s = K / 8;
    xp.assign((size_t)T * k8s * 4, 0);
    wp.assign((size_t)M * k8s * 4, 0);
    xf.resize((size_t)T * K);
    wf.resize((size_t)M * K);
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t k = 0; k < K; ++k) {
        xp[((size_t)t * k8s + k / 8) * 4 + (k % 8) / 2] |= (uint32_t)x[(size_t)t * K + k] << (16 * (k % 2));
        xf[(size_t)t * K + k] = h2f(x[(size_t)t * K + k]);
      }
    for (uint32_t m = 0; m < M; ++m)
      for (uint32_t k = 0; k < K; ++k) {
        wp[((size_t)m * k8s + k / 8) * 4 + (k % 8) / 2] |= (uint32_t)w[(size_t)k * M + m] << (16 * (k % 2));
        wf[(size_t)m * K + k] = h2f(w[(size_t)k * M + m]);
      }
  }

  // kmap[kt]: the k-tile whose operands the kernel uses at k-tile kt (-1: none).
  std::vector<int> identity() const {
    std::vector<int> v(KT);
    for (uint32_t i = 0; i < KT; ++i)
      v[i] = (int)i;
    return v;
  }

  // The tensor unit's fp16 model; swap exchanges the k8 halves of every k16 step.
  std::vector<uint16_t> ref16(const std::vector<int>& kmap, bool swap) const {
    std::vector<uint16_t> r((size_t)T * M);
    const uint32_t k8s = K / 8;
    parallel_rows(T, [&](uint32_t t) {
      for (uint32_t m = 0; m < M; ++m) {
        uint32_t c = 0;
        for (uint32_t kt = 0; kt < KT; ++kt) {
          if (kmap[kt] < 0)
            continue;
          for (uint32_t kk = 0; kk < (uint32_t)TILE_K / 16; ++kk)
            for (uint32_t hk = 0; hk < 2; ++hk) {
              const uint32_t k8 = (uint32_t)kmap[kt] * (TILE_K / 8) + 2 * kk + (swap ? 1 - hk : hk);
              c = tf::fedp(tf::FMT_FP16, tf::FMT_FP16, &wp[((size_t)m * k8s + k8) * 4],
                           &xp[((size_t)t * k8s + k8) * 4], c, 4);
            }
        }
        r[(size_t)t * M + m] = (uint16_t)c;
      }
    });
    return r;
  }

  // x . W with the products summed in double, as an fp32 tensor.
  std::vector<float> ref32(const std::vector<int>& kmap) const {
    std::vector<float> r((size_t)T * M);
    parallel_rows(T, [&](uint32_t t) {
      for (uint32_t m = 0; m < M; ++m) {
        double acc = 0;
        for (uint32_t kt = 0; kt < KT; ++kt) {
          if (kmap[kt] < 0)
            continue;
          const size_t k0 = (size_t)kmap[kt] * TILE_K;
          for (uint32_t k = 0; k < (uint32_t)TILE_K; ++k)
            acc += (double)xf[(size_t)t * K + k0 + k] * (double)wf[(size_t)m * K + k0 + k];
        }
        r[(size_t)t * M + m] = (float)acc;
      }
    });
    return r;
  }

  // The products accumulated in fp32 in k order, rounded once to fp16.
  std::vector<uint16_t> ref16_fp32_acc() const {
    std::vector<uint16_t> r((size_t)T * M);
    parallel_rows(T, [&](uint32_t t) {
      for (uint32_t m = 0; m < M; ++m) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < K; ++k)
          acc += xf[(size_t)t * K + k] * wf[(size_t)m * K + k];
        r[(size_t)t * M + m] = __half(acc).__x;
      }
    });
    return r;
  }
};

struct Result {
  double max_tile_rel = 0;
  bool finite = true;
  long long mismatch = 0;
  bool wrapper_ok(double tol) const { return finite && max_tile_rel <= tol; }
};

Result check(const std::vector<uint16_t>& out, const std::vector<uint16_t>& r16, const std::vector<float>& r32,
             uint32_t T, uint32_t M) {
  Result res;
  for (uint32_t tt = 0; tt < T / TILE_N; ++tt) {
    for (uint32_t mt = 0; mt < M / TILE_M; ++mt) {
      double d2 = 0, n2 = 0;
      for (uint32_t t = tt * TILE_N; t < (tt + 1) * TILE_N; ++t) {
        for (uint32_t m = mt * TILE_M; m < (mt + 1) * TILE_M; ++m) {
          const size_t i = (size_t)t * M + m;
          float c = h2f(out[i]);
          if (std::isnan(c))
            c = 1e4f;
          else if (std::isinf(c))
            c = c > 0 ? FLT_MAX : -FLT_MAX;
          const double d = (double)c - (double)r32[i];
          d2 += d * d;
          n2 += (double)r32[i] * (double)r32[i];
        }
      }
      // A NaN tile error (0 / 0) stays the maximum: it never passes.
      const double e = std::sqrt(d2 / n2);
      if (!std::isnan(res.max_tile_rel) && !(e <= res.max_tile_rel))
        res.max_tile_rel = e;
    }
  }
  for (size_t i = 0; i < out.size(); ++i) {
    if ((out[i] & 0x7c00u) == 0x7c00u)
      res.finite = false;
    res.mismatch += out[i] != r16[i];
  }
  return res;
}

// The descriptor the kernel must build: swizzle 128B, SBO 1024, base offset 0.
uint64_t want_desc(uint32_t start, uint32_t lbo) {
  return (1ull << 62) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)(lbo >> 4) << 16) | (uint64_t)(start >> 4);
}
uint64_t want_a(uint32_t g, uint32_t s, uint32_t kk) {
  return want_desc((g ? LM_A_2 : LM_A) + s * CHUNK_SIZE_A * 2 + kk * 2048, 0);
}
uint64_t want_b(uint32_t nh, uint32_t s, uint32_t kk) {
  return want_desc(LM_B + s * CHUNK_SIZE_B * 2 + nh * 16384 + kk * 32, 16);
}

void encode(vc::CUtensorMap* map, uint32_t rank, uint64_t addr, const uint64_t* dims, const uint64_t* strides,
            const uint32_t* box) {
  const uint32_t es[4] = {1, 1, 1, 1};
  CHECK(vc::tensor_map_encode_tiled(map, vc::CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, addr, dims, strides, box, es,
                                    vc::CU_TENSOR_MAP_INTERLEAVE_NONE, vc::CU_TENSOR_MAP_SWIZZLE_128B,
                                    vc::CU_TENSOR_MAP_L2_PROMOTION_L2_256B, vc::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int run(vx_kernel_h k) {
  const uint32_t T = o.T, M = o.M, K = o.K;
  const double tol = 4.0 * std::sqrt((double)K / 16.0) * std::pow(2.0, -11);

  Buf d_x((size_t)T * K * 2), d_w((size_t)K * M * 2);
  Out c_out((size_t)T * M * 2), smid_out(kGrid * 4), wid0_out(kGrid * 4);
  Out desc_out(o.dump ? (size_t)kGrid * DESC_PER_CTA * 8 : 4);

  kernel_arg_t arg;
  std::memset(&arg, 0, sizeof(arg));
  {
    // A = W [K][M] (build_tma_wgmma_mn(W, 64, 64, debug)): 64 m, 8 k rows,
    // M/64 m-atoms, K/8 k-groups; box one 64x64 tile.
    const uint64_t dims[4] = {64, 8, M / 64, K / 8};
    const uint64_t strides[3] = {2ull * M, 128, 16ull * M};
    const uint32_t box[4] = {64, 8, 1, 8};
    encode(&arg.a_map, 4, d_w.addr, dims, strides, box);
  }
  {
    // B = X [T][K] (build_tma_wgmma_k(X, 64, 256)): 64 k, T tokens, K/64.
    const uint64_t dims[3] = {64, T, K / 64};
    const uint64_t strides[2] = {2ull * K, 128};
    const uint32_t box[3] = {64, 256, 1};
    encode(&arg.b_map, 3, d_x.addr, dims, strides, box);
  }
  {
    // C = out [T][M] (build_tma_wgmma_mn(out, 64, 256, debug)).
    const uint64_t dims[4] = {64, 8, M / 64, T / 8};
    const uint64_t strides[3] = {2ull * M, 128, 16ull * M};
    const uint32_t box[4] = {64, 8, 1, 32};
    encode(&arg.c_map, 4, c_out.buf.addr, dims, strides, box);
  }
  arg.M = (int32_t)M;
  arg.N = (int32_t)T;
  arg.K = (int32_t)K;
  arg.num_blocks = (int32_t)kGrid;
  arg.sms_per_row = 1;
  arg.smid = smid_out.buf.addr;
  arg.wid0 = wid0_out.buf.addr;
  arg.desc = o.dump ? desc_out.buf.addr : 0;

  vc::launch_config_t cfg;
  cfg.grid = vc::dim3(kGrid, 1, 1);
  cfg.block = vc::dim3(BLOCK_THREADS, 1, 1);
  cfg.dyn_smem = LMEM_BYTES;
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)LMEM_BYTES));
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::non_portable_cluster_size_allowed, 1));
  int clusters = 0;
  CHECK(vc::occupancy_max_active_clusters<gemv_multicast_cluster_dims>(dev, k, cfg, &clusters));
  std::printf("max_active_clusters = %d\n", clusters);
  if (clusters < 1) {
    std::printf("Error: the cluster cannot be placed\n");
    return o.reps;
  }

  Problem p(T, M, K);
  int failed = 0;
  std::vector<std::string> neg_lines;
  bool neg_missed = false;
  for (int rep = 0; rep < o.reps; ++rep) {
    p.generate(o.seed + (unsigned long long)rep);
    const std::vector<int> ident = p.identity();
    const std::vector<uint16_t> r16 = p.ref16(ident, false);
    const std::vector<float> r32 = p.ref32(ident);

    d_x.write(p.x.data());
    d_w.write(p.w.data());
    c_out.reset(kHalfNaN2);
    smid_out.reset(kNoRecord);
    wid0_out.reset(kNoRecord);
    desc_out.reset(kNoRecord);

    vx_event_h ev = nullptr;
    CHECK(vc::launch_kernel_ex<gemv_multicast_cluster_dims>(q, k, cfg, arg, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);

    bool guards = c_out.fetch("C", rep);
    guards &= smid_out.fetch("smid", rep);
    guards &= wid0_out.fetch("wid0", rep);
    guards &= desc_out.fetch("descriptors", rep);
    const std::vector<uint16_t> out = c_out.as<uint16_t>();
    const std::vector<uint32_t> smid = smid_out.as<uint32_t>(), wid0 = wid0_out.as<uint32_t>();

    bool placed = true;
    for (uint32_t b = 0; b < kGrid; ++b) {
      if (smid[b] >= num_cores) {
        std::printf("Error: rep %d: block %u reports smid %u, not one of the %llu cores\n", rep, b, smid[b],
                    (unsigned long long)num_cores);
        placed = false;
      }
      if (wid0[b] % 4 != 0) {
        std::printf("Error: rep %d: block %u starts at hardware warp %u, not a warpgroup boundary\n", rep, b,
                    wid0[b]);
        placed = false;
      }
    }
    if (smid[0] == smid[1]) {
      std::printf("Error: rep %d: both blocks of the cluster report core %u\n", rep, smid[0]);
      placed = false;
    }

    long long bad_desc = 0;
    if (o.dump) {
      const std::vector<uint64_t> d = desc_out.as<uint64_t>();
      for (uint32_t b = 0; b < kGrid; ++b)
        for (uint32_t wg = 0; wg < DESC_WGS; ++wg)
          for (uint32_t s = 0; s < (uint32_t)BUFFER_SLOTS; ++s)
            for (uint32_t kk = 0; kk < (uint32_t)TILE_K / 16; ++kk) {
              const uint32_t g = wg / WGS_PER_GROUP, nh = wg % WGS_PER_GROUP;
              const uint64_t* e = &d[(size_t)b * DESC_PER_CTA + desc_index(wg, s, kk, 0)];
              const uint64_t wa = want_a(g, s, kk), wb = want_b(nh, s, kk);
              if (e[0] != wa || e[1] != wb) {
                if (bad_desc < 8)
                  std::printf("Error: rep %d: block %u warpgroup %u slot %u k-step %u: descriptors %#llx %#llx, "
                              "expected %#llx %#llx\n", rep, b, wg, s, kk, (unsigned long long)e[0],
                              (unsigned long long)e[1], (unsigned long long)wa, (unsigned long long)wb);
                ++bad_desc;
              }
            }
      std::printf("rep %d descriptors: %u checked, %lld wrong; A g0/g1 and B nh0/nh1, slots 0-2, k-steps 0-3:\n",
                  rep, kGrid * DESC_PER_CTA, bad_desc);
      for (uint32_t g = 0; g < 2; ++g) {
        std::printf("  A g%u:", g);
        for (uint32_t s = 0; s < (uint32_t)BUFFER_SLOTS; ++s)
          for (uint32_t kk = 0; kk < (uint32_t)TILE_K / 16; ++kk)
            std::printf(" %#llx", (unsigned long long)d[desc_index(g * WGS_PER_GROUP, s, kk, 0)]);
        std::printf("\n");
      }
      for (uint32_t nh = 0; nh < 2; ++nh) {
        std::printf("  B nh%u:", nh);
        for (uint32_t s = 0; s < (uint32_t)BUFFER_SLOTS; ++s)
          for (uint32_t kk = 0; kk < (uint32_t)TILE_K / 16; ++kk)
            std::printf(" %#llx", (unsigned long long)d[desc_index(nh, s, kk, 1)]);
        std::printf("\n");
      }
    }

    const Result res = check(out, r16, r32, T, M);
    const bool pass = res.wrapper_ok(tol) && res.mismatch == 0 && guards && placed && bad_desc == 0;
    failed += !pass;
    std::printf("rep %d: %s max_tile_rel=%.6e tol=%.6f bitexact_mismatch=%lld\n", rep, pass ? "PASS" : "FAIL",
                res.max_tile_rel, tol, res.mismatch);
    std::printf("  rep %d placement: block 0 core %u warp %u, block 1 core %u warp %u; finite=%d guards=%d\n", rep,
                smid[0], wid0[0], smid[1], wid0[1], res.finite ? 1 : 0, guards ? 1 : 0);

    if (o.neg && rep == 0) {
      // Each corrupted reference must fail the check(s) named.
      auto both = [&](const char* name, const std::vector<uint16_t>& n16, const std::vector<float>& n32) {
        const Result r = check(out, n16, n32, T, M);
        const bool det = !r.wrapper_ok(tol) && r.mismatch > 0;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s wrapper=%s max_tile_rel=%.3e bitexact_mismatch=%lld", name,
                      r.wrapper_ok(tol) ? "pass" : "FAIL", r.max_tile_rel, r.mismatch);
        return std::make_pair(det, std::string(buf));
      };
      auto exact = [&](const char* name, const std::vector<uint16_t>& n16) {
        const Result r = check(out, n16, r32, T, M);
        const bool det = r.mismatch > 0;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s bitexact_mismatch=%lld", name, r.mismatch);
        return std::make_pair(det, std::string(buf));
      };
      auto line = [&](const char* id, bool det, const std::string& why) {
        neg_missed |= !det;
        neg_lines.push_back(std::string("neg ") + id + ": " + (det ? "detected" : "MISSED") + " (" + why + ")");
      };
      {
        const auto a = exact("fp32 accumulation rounded once:", p.ref16_fp32_acc());
        line("a", a.first, a.second);
      }
      {
        const auto b = exact("k8 halves swapped:", p.ref16(ident, true));
        line("b", b.first, b.second);
      }
      {
        std::vector<int> drop = ident, reuse = ident;
        drop[p.KT - 1] = -1;
        reuse[p.KT - 1] = (int)p.KT - 1 - BUFFER_SLOTS;
        char nd[64], nr[64];
        std::snprintf(nd, sizeof(nd), "k-tile %u dropped:", p.KT - 1);
        std::snprintf(nr, sizeof(nr), "k-tile %u from k-tile %u:", p.KT - 1, p.KT - 1 - BUFFER_SLOTS);
        const auto d = both(nd, p.ref16(drop, false), p.ref32(drop));
        const auto r = both(nr, p.ref16(reuse, false), p.ref32(reuse));
        line("c", d.first && r.first, d.second + "; " + r.second);
      }
      {
        // Rank 1 holds tokens TILE_N..2 TILE_N-1; group g's m-tile is 2 tile_im + g.
        std::vector<uint16_t> n16 = r16;
        std::vector<float> n32 = r32;
        for (uint32_t t = TILE_N; t < T; ++t)
          for (uint32_t m = 0; m < M; ++m) {
            n16[(size_t)t * M + m] = r16[(size_t)t * M + (m ^ TILE_M)];
            n32[(size_t)t * M + m] = r32[(size_t)t * M + (m ^ TILE_M)];
          }
        const auto d = both("rank 1 with the other group's A rows:", n16, n32);
        line("d", d.first, d.second);
      }
      {
        const auto z = both("all zeros:", std::vector<uint16_t>(out.size(), 0), std::vector<float>(out.size(), 0.0f));
        line("e", z.first, z.second);
      }
    }
  }
  for (const auto& l : neg_lines)
    std::printf("%s\n", l.c_str());
  return failed + (neg_missed ? 1 : 0);
}

} // namespace

int main(int argc, char** argv) {
  // A hung launch is stopped from outside: keep what was printed.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  parse_args(argc, argv);

  CHECK(vx_device_open(0, &dev));
  uint64_t nw = 0, nt = 0, lmem_cap = 0, max_cluster = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));
  std::printf("dak_gemv_multicast: T=%u M=%u K=%u, %u m-tile pairs x %u k-tiles per block, reps %d, seed %llu%s%s; "
              "device %llu cores, %llu warps x %llu threads, %llu bytes of local memory, clusters up to %llu\n",
              o.T, o.M, o.K, o.M / (2 * TILE_M), o.K / TILE_K, o.reps, o.seed, o.neg ? ", negative controls" : "",
              o.dump ? ", descriptor dump" : "", (unsigned long long)num_cores, (unsigned long long)nw,
              (unsigned long long)nt, (unsigned long long)lmem_cap, (unsigned long long)max_cluster);
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA | VX_ISA_EXT_L2CACHE;
  const char* bad_dev = nullptr;
  if ((isa & need) != need)
    bad_dev = "the device needs DLMEM, the A extension, the DXA and an L2";
  else if (nt != 32 || nw * nt < (uint64_t)BLOCK_THREADS)
    bad_dev = "a block is 576 threads of 32-thread warps";
  else if (num_cores < kGrid || max_cluster < (uint64_t)CLUSTER_SIZE)
    bad_dev = "the device needs 2 cores and a cluster of 2";
  else if (LMEM_BYTES > lmem_cap)
    bad_dev = "the block's shared memory exceeds the local memory of a core";
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
  CHECK(vx_module_get_kernel(mod, "gemv_multicast", &k));

  const int failed = run(k);

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
