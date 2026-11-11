// cf_decoder: ClusterFusion's fused decode-step kernels (NeurIPS 2025) as the
// H100 ran them, one cluster of 4 blocks of 128 threads per attention head,
// one block per core, against the unfused sequence of the H100 harness
// (h100-workloads harness/wrappers/clusterfusion_{llama2_7b_decoder,
// deepseek_v2_lite_mla}.py) computed on the host.
//
//   -k llama|mla  the kernel (default llama):
//                 llama  Llama-2 attention block (RMSNorm with residual, QKV,
//                        NEOX RoPE, flash decoding over the cache + the new
//                        token, O projection); returns o, residual_out, k, v
//                 mla    DeepSeek-V2-Lite weight-absorbed MLA; returns o
//   -s <len>      llama: cached tokens S (default 250); the cache has S + 1
//                 rows, the kernel's maps cover rows 0..S-1
//   -r <reps>     launches, each on new inputs and checked (default 3)
//   -h            this help
//
// Head counts, the MLA hidden size and its sequence length are compile-time
// (CF_LLAMA_HEADS, CF_MLA_HEADS, CF_MLA_HIDDEN, CF_MLA_SEQ in the Makefile).
//
// Inputs (the wrappers' distributions, from a fixed seed; launch i uses seed
// + i, since a deterministic model would otherwise let a late read find the
// previous launch's correct values): llama N(0, 0.1^2) for every tensor;
// mla x and the cache N(0, 1), norm weights 1 + N(0, 0.1^2), projection
// weights N(0, 1/fan_in); RoPE angles uniform in [0, 2 pi).
//
// Reference: the unfused sequence in fp32 (double here), rounded to fp16
// where that sequence stores fp16. Tolerances are the wrappers':
//   llama o: max_abs <= 2e-2, MAE <= 1e-3, rel_L2 <= 1e-2; k_new, v_new:
//         rel_L2 <= 1e-2, rel_max <= 2e-2; residual_out bit-exact
//         half(input + residual); input and residual unchanged
//   mla   o: rel_L2 <= 1e-2, rel_max <= 2e-2, finite; x and cache unchanged
// Before each launch o is zeroed (the kernels add into it), the other output
// bodies are NaN-filled and the per-block core ids set to 0xffffffff, so an
// element the kernel does not write fails. A launch also fails if a guard
// word after an output changes or a cluster's blocks do not run on distinct
// cores. Exit 0 when every launch passes, 1 on
// a failure or a bad option.

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
#include <string>
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

constexpr uint64_t kSeed = 20261110ull;
constexpr uint32_t kBlockThreads = 128;
constexpr uint32_t kClusterSize = 4;
constexpr float kEps = 1e-6f;

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;
uint64_t num_cores = 0;

std::string opt_kernel = "llama";
uint32_t opt_seq = 250;
int reps = 3;

void usage() {
  std::printf("Usage: [-k llama|mla] [-s seq_len] [-r reps] [-h]\n"
              "  llama: %u heads, hidden %u (-s: cached tokens, default 250)\n"
              "  mla:   %u heads, hidden %u, seq_len %u\n",
              cf_llama::HEAD_NUM, cf_llama::HIDDEN_DIM, cf_mla::HEAD_NUM, cf_mla::HIDDEN_DIM, cf_mla::SEQ_LEN);
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:s:r:h")) != -1) {
    switch (c) {
    case 'k': opt_kernel = optarg; break;
    case 's': {
      const int s = std::atoi(optarg);
      if (s < 1) { usage(); std::exit(1); }
      opt_seq = (uint32_t)s;
      break;
    }
    case 'r':
      reps = std::atoi(optarg);
      if (reps < 1) { usage(); std::exit(1); }
      break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(1);
    }
  }
  if (optind < argc || (opt_kernel != "llama" && opt_kernel != "mla")) {
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
  template <typename T>
  void write(const std::vector<T>& v) {
    if (v.size() * sizeof(T) != size) {
      std::fprintf(stderr, "Error: buffer write of %zu bytes into %llu\n", v.size() * sizeof(T),
                   (unsigned long long)size);
      std::exit(1);
    }
    write(v.data());
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
// Before a launch: o is zero (the kernels add into it over the heads); every
// other output body is two f16 NaNs per word, and smid is 0xff bytes (no
// core), so an element or a block's smid the kernel does not write fails.
constexpr uint32_t kHalfNaN2 = 0x7fff7fffu;
constexpr uint32_t kNoCore = 0xffffffffu;

struct Out {
  size_t body;
  Buf buf;
  std::vector<uint8_t> raw;
  explicit Out(size_t b) : body(b), buf(b + kGuardBytes), raw(b + kGuardBytes) {}
  // The body filled with `fill`, guard words after it.
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

typedef std::vector<half> hvec;

struct Rng {
  std::mt19937_64 g;
  std::normal_distribution<double> n{0.0, 1.0};
  explicit Rng(uint64_t s) : g(s) {}
  hvec normal(size_t count, double stddev, double mean = 0.0) {
    hvec v(count);
    for (auto& x : v)
      x = half(mean + stddev * n(g));
    return v;
  }
  std::vector<float> angles(size_t count) {
    std::uniform_real_distribution<double> u(0.0, 2.0 * 3.14159265358979323846);
    std::vector<float> a(count);
    for (auto& x : a)
      x = (float)u(g);
    return a;
  }
};

inline double f(half h) { return (double)(float)h; }
inline half h16(double x) { return half(x); }

struct Err {
  double max_abs = 0, mae = 0, rel_l2 = 0, rel_max = 0, ref_max = 0;
  bool finite = true;
};

Err errs(const hvec& got, const hvec& ref) {
  Err e;
  double d2 = 0, r2 = 0, sum = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = f(got[i]), r = f(ref[i]);
    if (!std::isfinite(g))
      e.finite = false;
    const double d = std::fabs(g - r);
    if (!(d <= e.max_abs))
      e.max_abs = d;
    sum += d;
    d2 += d * d;
    r2 += r * r;
    e.ref_max = std::max(e.ref_max, std::fabs(r));
  }
  e.mae = sum / (double)got.size();
  e.rel_l2 = std::sqrt(d2) / std::sqrt(r2);
  e.rel_max = e.max_abs / std::max(e.ref_max, 1e-30);
  if (!e.finite) {
    e.max_abs = e.mae = e.rel_l2 = e.rel_max = INFINITY;
  }
  return e;
}

bool ok_o_llama(const Err& e) { return e.finite && e.max_abs <= 2e-2 && e.mae <= 1e-3 && e.rel_l2 <= 1e-2; }
bool ok_rel(const Err& e) { return e.finite && e.rel_l2 <= 1e-2 && e.rel_max <= 2e-2; }

void print_err(const char* name, const Err& e) {
  std::printf(" %s: max_abs=%.3e mae=%.3e rel_l2=%.3e rel_max=%.3e", name, e.max_abs, e.mae, e.rel_l2, e.rel_max);
}

// half(x * rsqrt(mean(x^2) + eps) * w), the norms' fp32 maths in double.
hvec rmsnorm(const std::vector<double>& x, const hvec& w) {
  double ms = 0;
  for (double v : x)
    ms += v * v;
  const double r = 1.0 / std::sqrt(ms / (double)x.size() + (double)kEps);
  hvec y(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    y[i] = h16(x[i] * r * f(w[i]));
  return y;
}

// Every cluster on distinct cores of the device.
bool placement_ok(const std::vector<uint32_t>& smid, int rep) {
  bool ok = true;
  for (size_t c = 0; c * kClusterSize < smid.size(); ++c) {
    std::set<uint32_t> s;
    for (uint32_t r = 0; r < kClusterSize; ++r) {
      const uint32_t id = smid[c * kClusterSize + r];
      if (id >= num_cores) {
        std::printf("Error: rep %d: block %zu reports smid %u, not one of the %llu cores\n", rep,
                    c * kClusterSize + r, id, (unsigned long long)num_cores);
        ok = false;
      }
      s.insert(id);
    }
    if (s.size() != kClusterSize) {
      std::printf("Error: rep %d: cluster %zu runs on %zu distinct cores\n", rep, c, s.size());
      ok = false;
    }
  }
  return ok;
}

void encode(vc::CUtensorMap* m, uint64_t addr, uint64_t inner, uint64_t outer, uint32_t box0, uint32_t box1) {
  const uint64_t dims[2] = {inner, outer};
  const uint64_t strides[1] = {inner * 2};
  const uint32_t box[2] = {box0, box1};
  const uint32_t es[2] = {1, 1};
  CHECK(vc::tensor_map_encode_tiled(m, vc::CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, addr, dims, strides, box, es,
                                    vc::CU_TENSOR_MAP_INTERLEAVE_NONE, vc::CU_TENSOR_MAP_SWIZZLE_NONE,
                                    vc::CU_TENSOR_MAP_L2_PROMOTION_NONE, vc::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

template <typename Dims, typename Arg>
bool launch(vx_kernel_h k, const Arg& arg, uint32_t grid, uint32_t lmem) {
  vc::launch_config_t cfg;
  cfg.grid = vc::dim3(grid, 1, 1);
  cfg.block = vc::dim3(kBlockThreads, 1, 1);
  cfg.dyn_smem = lmem;
  vx_event_h ev = nullptr;
  CHECK(vc::launch_kernel_ex<Dims>(q, k, cfg, arg, &ev));
  CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
  return true;
}

template <typename Dims>
bool prepare(vx_kernel_h k, uint32_t grid, uint32_t lmem) {
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::non_portable_cluster_size_allowed, 1));
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)lmem));
  vc::launch_config_t cfg;
  cfg.grid = vc::dim3(grid, 1, 1);
  cfg.block = vc::dim3(kBlockThreads, 1, 1);
  cfg.dyn_smem = lmem;
  int n = 0;
  CHECK(vc::occupancy_max_active_clusters<Dims>(dev, k, cfg, &n));
  std::printf("max_active_clusters = %d\n", n);
  if (n < 1) {
    std::printf("Error: the cluster cannot be placed\n");
    return false;
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////
// Llama

int run_llama(vx_kernel_h k) {
  using namespace cf_llama;
  const uint32_t H = HEAD_NUM, D = HEAD_DIM, hid = HIDDEN_DIM, S = opt_seq;
  const uint32_t kv_dim = ((S + CLUSTER_SIZE - 1) / CLUSTER_SIZE + (TMA_LOAD_ONCE_ATTN - 1)) & ~(TMA_LOAD_ONCE_ATTN - 1);
  const uint32_t grid = H * CLUSTER_SIZE;
  std::printf("cf_decoder: llama heads %u hidden %u head_dim %u seq_len %u (cache %u rows) kv_dim_per_block %u; "
              "grid %u x %u threads, clusters of %u, %u bytes of local memory per block\n",
              H, hid, D, S, S + 1, kv_dim, grid, kBlockThreads, CLUSTER_SIZE, LMEM_BYTES);
  if (!prepare<LlamaDecoderLayerKernel_cluster_dims>(k, grid, LMEM_BYTES))
    return 1;

  Buf d_in(hid * 2), d_res(hid * 2), d_wrms(hid * 2), d_wqkv((uint64_t)3 * hid * hid * 2),
      d_wo((uint64_t)hid * hid * 2), d_kc((uint64_t)(S + 1) * hid * 2), d_vc((uint64_t)(S + 1) * hid * 2),
      d_cos(D * 4), d_sin(D * 4);
  Out o_out(hid * 2), k_out(H * D * 2), v_out(H * D * 2), r_out(hid * 2), smid_out(grid * 4);

  llama_arg_t arg;
  std::memset(&arg, 0, sizeof(arg));
  // W_qkv [3 hid, hid] and W_o [hid, hid] as nn.Linear [out, in]; the caches
  // as S rows of hid (row S, the append slot, lies outside the maps).
  encode(&arg.tensor_map, d_wqkv.addr, hid, 3 * hid, TMA_LOAD_ONCE, HEAD_DIM);
  encode(&arg.tensor_map_k_cache, d_kc.addr, hid, S, HEAD_DIM, TMA_LOAD_ONCE / 2);
  encode(&arg.tensor_map_v_cache, d_vc.addr, hid, S, HEAD_DIM, TMA_LOAD_ONCE / 2);
  encode(&arg.tensor_map_weight_o, d_wo.addr, hid, hid, HEAD_DIM, TMA_LOAD_ONCE);
  arg.output = o_out.buf.addr;
  arg.k_output = k_out.buf.addr;
  arg.v_output = v_out.buf.addr;
  arg.input = d_in.addr;
  arg.residual = d_res.addr;
  arg.residual_output = r_out.buf.addr;
  arg.w_rms_input = d_wrms.addr;
  arg.cos = d_cos.addr;
  arg.sin = d_sin.addr;
  arg.smid = smid_out.buf.addr;
  arg.eps = kEps;
  arg.seq_len = S;
  arg.kv_dim_per_block = kv_dim;

  int failed = 0;
  for (int rep = 0; rep < reps; ++rep) {
    Rng rng(kSeed + (uint64_t)rep);
    const hvec inp = rng.normal(hid, 0.1), res = rng.normal(hid, 0.1), w_rms = rng.normal(hid, 0.1);
    const hvec wqkv = rng.normal((size_t)3 * hid * hid, 0.1), wo = rng.normal((size_t)hid * hid, 0.1);
    const hvec kc = rng.normal((size_t)(S + 1) * hid, 0.1), vc_ = rng.normal((size_t)(S + 1) * hid, 0.1);
    const std::vector<float> ang = rng.angles(D / 2);
    std::vector<float> cosv(D), sinv(D);
    for (uint32_t d = 0; d < D; ++d) {
      cosv[d] = std::cos(ang[d % (D / 2)]);
      sinv[d] = std::sin(ang[d % (D / 2)]);
    }

    // The unfused sequence: fused add + RMSNorm, QKV, NEOX RoPE, the new
    // token appended at row S, attention over S + 1 rows, O projection.
    hvec x(hid);
    std::vector<double> xd(hid);
    for (uint32_t i = 0; i < hid; ++i) {
      x[i] = h16(f(inp[i]) + f(res[i]));
      xd[i] = f(x[i]);
    }
    const hvec xn = rmsnorm(xd, w_rms);
    hvec qkv((size_t)3 * hid);
    for (size_t r = 0; r < (size_t)3 * hid; ++r) {
      double acc = 0;
      for (uint32_t i = 0; i < hid; ++i)
        acc += f(xn[i]) * f(wqkv[r * hid + i]);
      qkv[r] = h16(acc);
    }
    hvec k_ref((size_t)H * D), v_ref((size_t)H * D), attn((size_t)H * D);
    for (uint32_t h = 0; h < H; ++h) {
      const half* qh = &qkv[(size_t)h * D];
      const half* kh = &qkv[(size_t)hid + h * D];
      const half* vh = &qkv[(size_t)2 * hid + h * D];
      std::vector<double> qr(D);
      for (uint32_t d = 0; d < D; ++d) {
        const double rq = d < D / 2 ? -f(qh[d + D / 2]) : f(qh[d - D / 2]);
        const double rk = d < D / 2 ? -f(kh[d + D / 2]) : f(kh[d - D / 2]);
        qr[d] = f(h16(f(qh[d]) * cosv[d] + rq * sinv[d]));
        k_ref[(size_t)h * D + d] = h16(f(kh[d]) * cosv[d] + rk * sinv[d]);
        v_ref[(size_t)h * D + d] = vh[d];
      }
      std::vector<double> sc(S + 1);
      double mx = -INFINITY;
      for (uint32_t t = 0; t <= S; ++t) {
        double s = 0;
        for (uint32_t d = 0; d < D; ++d) {
          const double kv = t < S ? f(kc[(size_t)t * hid + h * D + d]) : f(k_ref[(size_t)h * D + d]);
          s += qr[d] * kv;
        }
        sc[t] = s / std::sqrt((double)D);
        mx = std::max(mx, sc[t]);
      }
      double den = 0;
      for (auto& s : sc) {
        s = std::exp(s - mx);
        den += s;
      }
      for (uint32_t d = 0; d < D; ++d) {
        double a = 0;
        for (uint32_t t = 0; t <= S; ++t)
          a += sc[t] * (t < S ? f(vc_[(size_t)t * hid + h * D + d]) : f(v_ref[(size_t)h * D + d]));
        attn[(size_t)h * D + d] = h16(a / den);
      }
    }
    hvec o_ref(hid);
    for (uint32_t r = 0; r < hid; ++r) {
      double acc = 0;
      for (uint32_t j = 0; j < hid; ++j)
        acc += f(attn[j]) * f(wo[(size_t)r * hid + j]);
      o_ref[r] = h16(acc);
    }

    d_in.write(inp);
    d_res.write(res);
    d_wrms.write(w_rms);
    d_wqkv.write(wqkv);
    d_wo.write(wo);
    d_kc.write(kc);
    d_vc.write(vc_);
    d_cos.write(cosv);
    d_sin.write(sinv);
    o_out.reset(0);
    k_out.reset(kHalfNaN2);
    v_out.reset(kHalfNaN2);
    r_out.reset(kHalfNaN2);
    smid_out.reset(kNoCore);

    launch<LlamaDecoderLayerKernel_cluster_dims>(k, arg, grid, LMEM_BYTES);

    bool guards = o_out.fetch("o", rep);
    guards &= k_out.fetch("k", rep);
    guards &= v_out.fetch("v", rep);
    guards &= r_out.fetch("residual_out", rep);
    guards &= smid_out.fetch("smid", rep);
    const hvec o = o_out.as<half>(), kn = k_out.as<half>(), vn = v_out.as<half>(), ro = r_out.as<half>();
    hvec in_after(hid), res_after(hid);
    d_in.read(in_after.data());
    d_res.read(res_after.data());
    long long res_bad = 0, in_changed = 0;
    for (uint32_t i = 0; i < hid; ++i) {
      res_bad += ro[i].__x != x[i].__x;
      in_changed += (in_after[i].__x != inp[i].__x) + (res_after[i].__x != res[i].__x);
    }
    const bool placed = placement_ok(smid_out.as<uint32_t>(), rep);
    const Err eo = errs(o, o_ref), ek = errs(kn, k_ref), ev = errs(vn, v_ref);
    const bool pass = ok_o_llama(eo) && ok_rel(ek) && ok_rel(ev) && res_bad == 0 && in_changed == 0 && guards &&
                      placed;
    failed += !pass;
    std::printf("rep %d: %s", rep, pass ? "PASS" : "FAIL");
    print_err("o", eo);
    print_err("k_new", ek);
    print_err("v_new", ev);
    std::printf(" residual_out_mismatch=%lld inputs_changed=%lld max|o_ref|=%.3e\n", res_bad, in_changed,
                eo.ref_max);
  }
  return failed;
}

///////////////////////////////////////////////////////////////////////////////
// MLA

int run_mla(vx_kernel_h k) {
  using namespace cf_mla;
  const uint32_t H = HEAD_NUM, Dm = HIDDEN_DIM, N = NOPE_HEAD_DIM, R = ROPE_HEAD_DIM, L = KV_LORA_RANK, S = SEQ_LEN;
  const uint32_t W = MLA_HEAD_DIM;
  const uint32_t grid = H * CLUSTER_SIZE;
  std::printf("cf_decoder: mla heads %u hidden %u nope %u rope %u lora %u seq_len %u (attention over %u cached "
              "tokens + the new one); grid %u x %u threads, clusters of %u, %u bytes of local memory per block\n",
              H, Dm, N, R, L, S, S - 1, grid, kBlockThreads, CLUSTER_SIZE, LMEM_BYTES);
  if (!prepare<DeepSeekDecoderLayerKernel_cluster_dims>(k, grid, LMEM_BYTES))
    return 1;

  Buf d_x(Dm * 2), d_win(Dm * 2), d_wckv(L * 2), d_wq((uint64_t)Dm * H * N * 2), d_wqpe((uint64_t)Dm * H * R * 2),
      d_wkv((uint64_t)Dm * L * 2), d_wkpe((uint64_t)Dm * R * 2), d_wuk((uint64_t)N * H * L * 2),
      d_wuv((uint64_t)L * H * N * 2), d_wo((uint64_t)H * N * Dm * 2), d_cache((uint64_t)S * W * 2), d_cos(R * 4),
      d_sin(R * 4);
  Out o_out(Dm * 2), smid_out(grid * 4);

  mla_arg_t arg;
  std::memset(&arg, 0, sizeof(arg));
  // Weights [in, out] row-major; the cache [S, 576] = latent (512) | k_pe (64).
  encode(&arg.tensor_map_weight_q, d_wq.addr, H * N, Dm, N, TMA_LOAD_ONCE);
  encode(&arg.tensor_map_weight_q_pe, d_wqpe.addr, H * R, Dm, TMA_LOAD_ONCE, 128);
  encode(&arg.tensor_map_weight_uk, d_wuk.addr, H * L, N, 128, TMA_LOAD_ONCE);
  encode(&arg.tensor_map_weight_kv, d_wkv.addr, L, Dm, 128, TMA_LOAD_ONCE);
  encode(&arg.tensor_map_weight_k_pe, d_wkpe.addr, R, Dm, TMA_LOAD_ONCE, 128);
  encode(&arg.tensor_map_kv_cache, d_cache.addr, W, S, 256, TMA_LOAD_ONCE_ATTN);
  encode(&arg.tensor_map_kv_cache_, d_cache.addr, W, S, 256, TMA_LOAD_ONCE_ATTN);
  encode(&arg.tensor_map_weight_uv, d_wuv.addr, H * N, L, 128, TMA_LOAD_ONCE);
  encode(&arg.tensor_map_weight_o, d_wo.addr, Dm, H * N, TMA_LOAD_ONCE, N);
  arg.output = o_out.buf.addr;
  arg.input = d_x.addr;
  arg.w_rms_input = d_win.addr;
  arg.w_rms_ckv = d_wckv.addr;
  arg.cos = d_cos.addr;
  arg.sin = d_sin.addr;
  arg.smid = smid_out.buf.addr;

  const double sm_scale = 1.0 / std::sqrt((double)(N + R));
  int failed = 0;
  for (int rep = 0; rep < reps; ++rep) {
    Rng rng(kSeed + 1000 + (uint64_t)rep);
    const hvec x = rng.normal(Dm, 1.0), w_in = rng.normal(Dm, 0.1, 1.0), w_ckv = rng.normal(L, 0.1, 1.0);
    const hvec wq = rng.normal((size_t)Dm * H * N, 1.0 / std::sqrt((double)Dm));
    const hvec wqpe = rng.normal((size_t)Dm * H * R, 1.0 / std::sqrt((double)Dm));
    const hvec wkv = rng.normal((size_t)Dm * L, 1.0 / std::sqrt((double)Dm));
    const hvec wkpe = rng.normal((size_t)Dm * R, 1.0 / std::sqrt((double)Dm));
    const hvec wuk = rng.normal((size_t)N * H * L, 1.0 / std::sqrt((double)N));
    const hvec wuv = rng.normal((size_t)L * H * N, 1.0 / std::sqrt((double)L));
    const hvec wo = rng.normal((size_t)H * N * Dm, 1.0 / std::sqrt((double)(H * N)));
    const hvec cache = rng.normal((size_t)S * W, 1.0);
    const std::vector<float> ang = rng.angles(R / 2);
    std::vector<float> cosv(R), sinv(R);
    for (uint32_t d = 0; d < R; ++d) {
      cosv[d] = std::cos(ang[d % (R / 2)]);
      sinv[d] = std::sin(ang[d % (R / 2)]);
    }

    // The wrapper's unfused sequence, without the rope score term and the
    // cache append, as the kernel: RMSNorm, q_nope, c_kv, its RMSNorm,
    // absorption, attention over rows 0..S-2 and the new latent, W_UV, W_O.
    std::vector<double> xd(Dm);
    for (uint32_t i = 0; i < Dm; ++i)
      xd[i] = f(x[i]);
    const hvec xn = rmsnorm(xd, w_in);
    hvec q_nope((size_t)H * N);
    for (size_t c = 0; c < (size_t)H * N; ++c) {
      double acc = 0;
      for (uint32_t i = 0; i < Dm; ++i)
        acc += f(xn[i]) * f(wq[(size_t)i * H * N + c]);
      q_nope[c] = h16(acc);
    }
    std::vector<double> ckv(L);
    for (uint32_t j = 0; j < L; ++j) {
      double acc = 0;
      for (uint32_t i = 0; i < Dm; ++i)
        acc += f(xn[i]) * f(wkv[(size_t)i * L + j]);
      ckv[j] = f(h16(acc));
    }
    const hvec cn = rmsnorm(ckv, w_ckv);
    hvec a((size_t)H * N);
    for (uint32_t h = 0; h < H; ++h) {
      std::vector<double> ql(L);
      for (uint32_t j = 0; j < L; ++j) {
        double acc = 0;
        for (uint32_t n = 0; n < N; ++n)
          acc += f(q_nope[(size_t)h * N + n]) * f(wuk[(size_t)n * H * L + h * L + j]);
        ql[j] = f(h16(acc));
      }
      auto Krow = [&](uint32_t t, uint32_t j) { return t + 1 < S ? f(cache[(size_t)t * W + j]) : f(cn[j]); };
      std::vector<double> sc(S);
      double mx = -INFINITY;
      for (uint32_t t = 0; t < S; ++t) {
        double s = 0;
        for (uint32_t j = 0; j < L; ++j)
          s += ql[j] * Krow(t, j);
        sc[t] = s * sm_scale;
        mx = std::max(mx, sc[t]);
      }
      double den = 0;
      for (auto& s : sc) {
        s = std::exp(s - mx);
        den += s;
      }
      std::vector<double> al(L);
      for (uint32_t j = 0; j < L; ++j) {
        double acc = 0;
        for (uint32_t t = 0; t < S; ++t)
          acc += sc[t] * Krow(t, j);
        al[j] = f(h16(acc / den));
      }
      for (uint32_t n = 0; n < N; ++n) {
        double acc = 0;
        for (uint32_t j = 0; j < L; ++j)
          acc += al[j] * f(wuv[(size_t)j * H * N + h * N + n]);
        a[(size_t)h * N + n] = h16(acc);
      }
    }
    hvec o_ref(Dm);
    for (uint32_t c = 0; c < Dm; ++c) {
      double acc = 0;
      for (size_t r = 0; r < (size_t)H * N; ++r)
        acc += f(a[r]) * f(wo[r * Dm + c]);
      o_ref[c] = h16(acc);
    }

    d_x.write(x);
    d_win.write(w_in);
    d_wckv.write(w_ckv);
    d_wq.write(wq);
    d_wqpe.write(wqpe);
    d_wkv.write(wkv);
    d_wkpe.write(wkpe);
    d_wuk.write(wuk);
    d_wuv.write(wuv);
    d_wo.write(wo);
    d_cache.write(cache);
    d_cos.write(cosv);
    d_sin.write(sinv);
    o_out.reset(0);
    smid_out.reset(kNoCore);

    launch<DeepSeekDecoderLayerKernel_cluster_dims>(k, arg, grid, LMEM_BYTES);

    bool guards = o_out.fetch("o", rep);
    guards &= smid_out.fetch("smid", rep);
    const hvec o = o_out.as<half>();
    hvec x_after(Dm), cache_after((size_t)S * W);
    d_x.read(x_after.data());
    d_cache.read(cache_after.data());
    long long in_changed = 0;
    for (uint32_t i = 0; i < Dm; ++i)
      in_changed += x_after[i].__x != x[i].__x;
    for (size_t i = 0; i < cache.size(); ++i)
      in_changed += cache_after[i].__x != cache[i].__x;
    const bool placed = placement_ok(smid_out.as<uint32_t>(), rep);
    const Err eo = errs(o, o_ref);
    const bool pass = ok_rel(eo) && in_changed == 0 && guards && placed;
    failed += !pass;
    std::printf("rep %d: %s", rep, pass ? "PASS" : "FAIL");
    print_err("o", eo);
    std::printf(" inputs_changed=%lld max|o_ref|=%.3e\n", in_changed, eo.ref_max);
  }
  return failed;
}

} // namespace

int main(int argc, char** argv) {
  // A hung launch is killed by the runner's timeout: keep what was printed.
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
  std::printf("device: %llu cores, %llu warps x %llu threads, %llu bytes of local memory, clusters up to %llu\n",
              (unsigned long long)num_cores, (unsigned long long)nw, (unsigned long long)nt,
              (unsigned long long)lmem_cap, (unsigned long long)max_cluster);
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA | VX_ISA_EXT_L2CACHE;
  const bool is_llama = opt_kernel == "llama";
  const uint32_t lmem = is_llama ? cf_llama::LMEM_BYTES : cf_mla::LMEM_BYTES;
  const char* bad_dev = nullptr;
  if ((isa & need) != need)
    bad_dev = "the device needs DLMEM, the A extension, the DXA and an L2";
  else if (nw * nt != kBlockThreads)
    bad_dev = "a block is 128 threads: the device needs NUM_WARPS x NUM_THREADS = 128";
  else if (num_cores < kClusterSize || max_cluster < kClusterSize)
    bad_dev = "the device needs 4 cores and a cluster of 4";
  else if (lmem > lmem_cap)
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
  CHECK(vx_module_get_kernel(mod, is_llama ? "LlamaDecoderLayerKernel" : "DeepSeekDecoderLayerKernel", &k));

  const int failed = is_llama ? run_llama(k) : run_mla(k);

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
