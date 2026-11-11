// credit_reduction: the six CREDIT cluster kernels the H100 ran (CREDIT
// commit c8f2ad2, src/workloads/*.cu with the roofline_hooks patch), one row
// of `cols` elements per cluster of P blocks of 256 threads, one block per
// core; block b of a cluster stages slice b (ceil(cols / P) elements).
//
//   -k <kernel>  layernorm | weighted_var | pearson | softmax | lars | rowwise
//                (required)
//   -c <cols>    row length (default 4096, the H100 shape n4k)
//   -n <rows>    rows, one cluster each (default 3)
//   -p <P>       cluster size, 2 to 8 (default 8, the profiled size)
//   -r <reps>    launches, each one checked (default 3)
//   -h           this help
//
// Exit 0 when every launch passes, 1 on a failure or a bad option.
//
// Inputs follow the H100 init formulas (the init_*_kernel LCG hashes of the
// element index, gamma and grad from the column and row), computed here on the
// host. Launch i hashes index + i * rows * cols, so every launch gets a new
// input and its own reference: the models are deterministic, and a launch on
// the same input would find the previous launch's correct values still in
// memory. layernorm adds 0.25 k to x in row r of launch i, k = i * rows + r (k
// = 0 is the plain formula), so a slot left over from another row or launch
// is off by its mean.
//
// The reported rule is the H100 harness's, over every element (the H100 checks
// 3 rows x 4 columns), in double, except rowwise, whose reference is the H100
// harness's fp32 quantizer. Where outputs are far below its absolute term, a
// tight gate is added that a broken cluster reduction fails (a missing or
// stale peer, or all zeros):
//   layernorm, pearson, softmax: per row ||got - ref|| / ||ref|| <= 1e-4;
//   lars: per element |(w_out - w) - (-step u)| <= 2e-5 |step u| + ulp(w_out),
//         and momentum_out per row rel L2 <= 1e-4.
// `tight` on each rep line is the worst gate value over its limit.
// A launch also fails if a 256-byte guard tail after an output or the smid
// buffer changes, an element is left unwritten, or a cluster's blocks are not
// on distinct cores of the device.

#include <vortex2.h>
#include <vx_cuda_runtime.h>
#include <VX_types.h>
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace vc = vx::cuda;
using credit::kBlockThreads;
using credit::kMaxClusterSize;

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
uint64_t num_cores = 0;

std::string opt_kernel;
int cols = 4096, rows = 3, P = 8, reps = 3;

const char* const kKernels[] = {"layernorm", "weighted_var", "pearson", "softmax", "lars", "rowwise"};

void usage() {
  std::printf("Usage: -k layernorm|weighted_var|pearson|softmax|lars|rowwise [-c cols] [-n rows] [-p P] "
              "[-r reps] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:c:n:p:r:h")) != -1) {
    switch (c) {
    case 'k': opt_kernel = optarg; break;
    case 'c': cols = std::atoi(optarg); break;
    case 'n': rows = std::atoi(optarg); break;
    case 'p': P = std::atoi(optarg); break;
    case 'r': reps = std::atoi(optarg); break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(1);
    }
  }
  bool known = false;
  for (const char* k : kKernels)
    known |= opt_kernel == k;
  if (optind < argc || !known || cols < 1 || rows < 1 || reps < 1 || P < 2 || P > kMaxClusterSize) {
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

// An output buffer: `body` bytes the kernel writes, then the guard tail. The
// body starts as `fill` bytes, a value the kernel never writes (NaN for
// floats, -128 for the int8 quantizer, which clamps to [-127, 127]).
struct Output {
  Buf buf;
  size_t body;
  uint8_t fill;
  const char* name;
  std::vector<uint8_t> raw;
  Output(size_t body_, uint8_t fill_, const char* name_)
      : buf(body_ + kGuardBytes), body(body_), fill(fill_), name(name_), raw(body_ + kGuardBytes) {}
  void reset() {
    std::fill(raw.begin(), raw.begin() + body, fill);
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
  template <typename T>
  const T* data() const { return reinterpret_cast<const T*>(raw.data()); }
};

// The init kernels' hashes of an element index.
inline unsigned hash_a(uint64_t idx) { return (unsigned)((idx * 1103515245ull + 12345ull) >> 16); }
inline unsigned hash_b(uint64_t idx) { return (unsigned)((idx * 1664525ull + 1013904223ull) >> 16); }
inline unsigned hash_c(uint64_t idx) { return (unsigned)((idx * 22695477ull + 1ull) >> 16); }

// Error tally of one output against its reference: err <= atol + rtol |ref|.
struct Tally {
  long long bad = 0;
  double max_err = 0, worst = 0;
  void add(double got, double ref, double atol, double rtol) {
    const double err = std::fabs(got - ref);
    const double lim = atol + rtol * std::fabs(ref);
    if (!(err <= lim))
      ++bad;
    if (!(err <= max_err))
      max_err = err;
    const double ratio = lim > 0 ? err / lim : (err > 0 ? std::numeric_limits<double>::infinity() : 0.0);
    if (!(ratio <= worst))
      worst = ratio;
  }
  // The tight gate: worst value over limit; above 1 fails.
  double tight = 0;
  void gate(double value, double lim) {
    const double ratio = value / lim;
    if (!(ratio <= tight))
      tight = ratio;
  }
};

// ||got - ref|| / ||ref|| over one row of one output.
struct RelL2 {
  double num = 0, den = 0;
  void add(double got, double ref) {
    num += (got - ref) * (got - ref);
    den += ref * ref;
  }
  double value() const { return den > 0 ? std::sqrt(num / den) : (num > 0 ? HUGE_VAL : 0.0); }
};
constexpr double kRelL2 = 1e-4;

// One unit in the last place of the float nearest v.
double f32_ulp(double v) {
  const double a = std::fabs(v);
  if (a < 0x1p-126)
    return 0x1p-149;
  int e = 0;
  std::frexp(a, &e);
  return std::ldexp(1.0, e - 24);
}

// What one kernel needs from the host: its inputs per launch, its argument
// block, and the check of its outputs.
struct Run {
  virtual ~Run() {}
  virtual int tile_arrays() const = 0;
  virtual uint32_t vars_bytes() const = 0;
  virtual void prepare(int rep) = 0;
  virtual vx_result_t launch(vx_kernel_h k, const vc::launch_config_t& cfg, uint64_t smid, int slice) = 0;
  virtual bool check(int rep, Tally* t) = 0;
};

std::vector<float> make_vec(size_t n) { return std::vector<float>(n); }

size_t elems() { return (size_t)rows * cols; }

struct LayerNorm : Run {
  static constexpr float kEps = 1e-5f;
  std::vector<float> x = make_vec(elems()), dy = make_vec(elems()), gamma = make_vec(cols);
  Buf x_in{elems() * 4}, dy_in{elems() * 4}, gamma_in{(uint64_t)cols * 4};
  Output dx{elems() * 4, 0xff, "dx"};
  int tile_arrays() const override { return layernorm::kTileArrays; }
  uint32_t vars_bytes() const override {
    return credit::tile_offset<layernorm::Smem<kBlockThreads, kMaxClusterSize>>();
  }
  void prepare(int rep) override {
    const uint64_t base = (uint64_t)rep * elems();
    for (size_t i = 0; i < elems(); ++i) {
      const float k = (float)((size_t)rep * rows + i / cols);
      x[i] = (float)(hash_a(base + i) & 2047u) * (1.0f / 1024.0f) - 1.0f + 0.25f * k;
      dy[i] = (float)(hash_b(base + i) & 2047u) * (1.0f / 1024.0f) - 1.0f;
    }
    for (int c = 0; c < cols; ++c)
      gamma[c] = 0.75f + (float)(c % 31) * 0.015625f;
    x_in.write(x.data());
    dy_in.write(dy.data());
    gamma_in.write(gamma.data());
    dx.reset();
  }
  vx_result_t launch(vx_kernel_h k, const vc::launch_config_t& cfg, uint64_t smid, int slice) override {
    layernorm::Args a = {x_in.addr, dy_in.addr, gamma_in.addr, dx.buf.addr, smid, rows, cols, kEps, slice};
    vx_event_h ev = nullptr;
    vx_result_t r = vc::launch_kernel_ex(q, k, cfg, a, &ev);
    if (r == VX_SUCCESS) {
      CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(ev);
    }
    return r;
  }
  bool check(int rep, Tally* t) override {
    if (!dx.fetch(rep))
      return false;
    const float* got = dx.data<float>();
    for (int r = 0; r < rows; ++r) {
      const size_t rb = (size_t)r * cols;
      double sum = 0, sumsq = 0;
      for (int c = 0; c < cols; ++c) {
        sum += (double)x[rb + c];
        sumsq += (double)x[rb + c] * (double)x[rb + c];
      }
      const double mean = sum / cols;
      const double var = std::max(sumsq / cols - mean * mean, 0.0);
      const double inv_std = 1.0 / std::sqrt(var + kEps);
      double sum_dyg = 0, sum_dyg_xhat = 0;
      for (int c = 0; c < cols; ++c) {
        const double xhat = ((double)x[rb + c] - mean) * inv_std;
        const double dyg = (double)dy[rb + c] * (double)gamma[c];
        sum_dyg += dyg;
        sum_dyg_xhat += dyg * xhat;
      }
      const double mean_dyg = sum_dyg / cols, mean_dyg_xhat = sum_dyg_xhat / cols;
      RelL2 rl;
      for (int c = 0; c < cols; ++c) {
        const double xhat = ((double)x[rb + c] - mean) * inv_std;
        const double dyg = (double)dy[rb + c] * (double)gamma[c];
        const double ref = (dyg - mean_dyg - xhat * mean_dyg_xhat) * inv_std;
        t->add(got[rb + c], ref, 2e-2, 2e-2);
        rl.add(got[rb + c], ref);
      }
      t->gate(rl.value(), kRelL2);
    }
    return true;
  }
};

struct WeightedVar : Run {
  std::vector<float> in = make_vec(elems()), w = make_vec(elems()), dy = make_vec(elems());
  Buf in_d{elems() * 4}, w_d{elems() * 4}, dy_d{elems() * 4};
  Output out{elems() * 4, 0xff, "output"};
  int tile_arrays() const override { return weighted_var::kTileArrays; }
  uint32_t vars_bytes() const override {
    return credit::tile_offset<weighted_var::Smem<kBlockThreads, kMaxClusterSize>>();
  }
  void prepare(int rep) override {
    const uint64_t base = (uint64_t)rep * elems();
    for (size_t i = 0; i < elems(); ++i) {
      const float b = (float)(hash_a(base + i) & 4095u) * (1.0f / 1024.0f) - 2.0f;
      const float tiny = (float)(hash_b(base + i) & 255u) * 1.0e-7f;
      in[i] = b + tiny;
      w[i] = 0.5f + (float)(hash_b(base + i) & 4095u) * (1.0f / 4096.0f);
      dy[i] = (float)(hash_c(base + i) & 4095u) * (1.0f / 2048.0f) - 1.0f;
    }
    in_d.write(in.data());
    w_d.write(w.data());
    dy_d.write(dy.data());
    out.reset();
  }
  vx_result_t launch(vx_kernel_h k, const vc::launch_config_t& cfg, uint64_t smid, int slice) override {
    weighted_var::Args a = {in_d.addr, w_d.addr, dy_d.addr, out.buf.addr, smid, rows, cols, slice};
    vx_event_h ev = nullptr;
    vx_result_t r = vc::launch_kernel_ex(q, k, cfg, a, &ev);
    if (r == VX_SUCCESS) {
      CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(ev);
    }
    return r;
  }
  bool check(int rep, Tally* t) override {
    if (!out.fetch(rep))
      return false;
    const float* got = out.data<float>();
    const double eps = weighted_var::kEps;
    for (int r = 0; r < rows; ++r) {
      const size_t rb = (size_t)r * cols;
      double sum_w = 0, sum_wx = 0;
      for (int c = 0; c < cols; ++c) {
        sum_w += (double)w[rb + c];
        sum_wx += (double)w[rb + c] * (double)in[rb + c];
      }
      const double mean = sum_wx / (sum_w + eps);
      double wvar = 0;
      for (int c = 0; c < cols; ++c) {
        const double diff = (double)in[rb + c] - mean;
        wvar += (double)w[rb + c] * diff * diff;
      }
      const double var = std::max(wvar / (sum_w + eps), 0.0);
      const double inv_std = 1.0 / std::sqrt(var + eps);
      const double inv_sum_w = 1.0 / (sum_w + eps);
      double sum_dy = 0, sum_dy_centered = 0;
      for (int c = 0; c < cols; ++c) {
        sum_dy += (double)dy[rb + c];
        sum_dy_centered += (double)dy[rb + c] * ((double)in[rb + c] - mean);
      }
      for (int c = 0; c < cols; ++c) {
        const double centered = (double)in[rb + c] - mean, weight = (double)w[rb + c];
        const double ref = inv_std * ((double)dy[rb + c] - weight * inv_sum_w * sum_dy -
                                      weight * centered * inv_std * inv_std * inv_sum_w * sum_dy_centered);
        t->add(got[rb + c], ref, 2.0e-4, 2.0e-4);
      }
    }
    return true;
  }
};

struct Pearson : Run {
  static constexpr float kEps = 1e-5f;
  std::vector<float> x = make_vec(elems()), y = make_vec(elems()), grad = make_vec(rows);
  Buf x_d{elems() * 4}, y_d{elems() * 4}, grad_d{(uint64_t)rows * 4};
  Output dx{elems() * 4, 0xff, "dx"}, dy{elems() * 4, 0xff, "dy"};
  int tile_arrays() const override { return pearson::kTileArrays; }
  uint32_t vars_bytes() const override {
    return credit::tile_offset<pearson::Smem<kBlockThreads, kMaxClusterSize>>();
  }
  void prepare(int rep) override {
    const uint64_t base = (uint64_t)rep * elems();
    for (size_t i = 0; i < elems(); ++i) {
      x[i] = (float)(hash_a(base + i) & 2047u) * (1.0f / 1024.0f) - 1.0f;
      y[i] = (float)(hash_b(base + i) & 2047u) * (1.0f / 1024.0f) - 1.0f;
    }
    for (int r = 0; r < rows; ++r)
      grad[r] = 0.75f + (float)(r % 17) * 0.03125f;
    x_d.write(x.data());
    y_d.write(y.data());
    grad_d.write(grad.data());
    dx.reset();
    dy.reset();
  }
  vx_result_t launch(vx_kernel_h k, const vc::launch_config_t& cfg, uint64_t smid, int slice) override {
    pearson::Args a = {x_d.addr, y_d.addr, grad_d.addr, dx.buf.addr, dy.buf.addr, smid, rows, cols, kEps, slice};
    vx_event_h ev = nullptr;
    vx_result_t r = vc::launch_kernel_ex(q, k, cfg, a, &ev);
    if (r == VX_SUCCESS) {
      CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(ev);
    }
    return r;
  }
  bool check(int rep, Tally* t) override {
    const bool g1 = dx.fetch(rep), g2 = dy.fetch(rep);
    if (!g1 || !g2)
      return false;
    const float* gdx = dx.data<float>();
    const float* gdy = dy.data<float>();
    for (int r = 0; r < rows; ++r) {
      const size_t rb = (size_t)r * cols;
      double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
      for (int c = 0; c < cols; ++c) {
        const double xv = x[rb + c], yv = y[rb + c];
        sx += xv;
        sy += yv;
        sxx += xv * xv;
        syy += yv * yv;
        sxy += xv * yv;
      }
      const double mean_x = sx / cols, mean_y = sy / cols;
      const double var_x = std::max(sxx - sx * mean_x, 0.0), var_y = std::max(syy - sy * mean_y, 0.0);
      const double cov = sxy - sx * mean_y;
      const double inv_x = 1.0 / std::sqrt(var_x + kEps), inv_y = 1.0 / std::sqrt(var_y + kEps);
      const double inv_xy = inv_x * inv_y;
      const double coeff_x = cov * inv_x * inv_x * inv_x * inv_y;
      const double coeff_y = cov * inv_x * inv_y * inv_y * inv_y;
      const double g = grad[r];
      RelL2 rx, ry;
      for (int c = 0; c < cols; ++c) {
        const double xc = (double)x[rb + c] - mean_x, yc = (double)y[rb + c] - mean_y;
        const double ref_dx = g * (yc * inv_xy - xc * coeff_x), ref_dy = g * (xc * inv_xy - yc * coeff_y);
        t->add(gdx[rb + c], ref_dx, 2e-2, 2e-2);
        t->add(gdy[rb + c], ref_dy, 2e-2, 2e-2);
        rx.add(gdx[rb + c], ref_dx);
        ry.add(gdy[rb + c], ref_dy);
      }
      t->gate(rx.value(), kRelL2);
      t->gate(ry.value(), kRelL2);
    }
    return true;
  }
};

struct Softmax : Run {
  std::vector<float> logits = make_vec(elems()), dy = make_vec(elems());
  Buf logits_d{elems() * 4}, dy_d{elems() * 4};
  Output dx{elems() * 4, 0xff, "dx"};
  int tile_arrays() const override { return softmax::kTileArrays; }
  uint32_t vars_bytes() const override {
    return credit::tile_offset<softmax::Smem<kBlockThreads, kMaxClusterSize>>();
  }
  void prepare(int rep) override {
    const uint64_t base = (uint64_t)rep * elems();
    for (size_t i = 0; i < elems(); ++i) {
      logits[i] = ((float)(hash_a(base + i) & 4095u) - 2048.0f) * (1.0f / 512.0f);
      dy[i] = (float)(hash_b(base + i) & 2047u) * (1.0f / 1024.0f) - 1.0f;
    }
    logits_d.write(logits.data());
    dy_d.write(dy.data());
    dx.reset();
  }
  vx_result_t launch(vx_kernel_h k, const vc::launch_config_t& cfg, uint64_t smid, int slice) override {
    softmax::Args a = {logits_d.addr, dy_d.addr, dx.buf.addr, smid, rows, cols, slice};
    vx_event_h ev = nullptr;
    vx_result_t r = vc::launch_kernel_ex(q, k, cfg, a, &ev);
    if (r == VX_SUCCESS) {
      CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(ev);
    }
    return r;
  }
  bool check(int rep, Tally* t) override {
    if (!dx.fetch(rep))
      return false;
    const float* got = dx.data<float>();
    for (int r = 0; r < rows; ++r) {
      const size_t rb = (size_t)r * cols;
      double row_max = -std::numeric_limits<double>::infinity();
      for (int c = 0; c < cols; ++c)
        row_max = std::max(row_max, (double)logits[rb + c]);
      double row_sum = 0;
      for (int c = 0; c < cols; ++c)
        row_sum += std::exp((double)logits[rb + c] - row_max);
      double row_dot = 0;
      for (int c = 0; c < cols; ++c)
        row_dot += std::exp((double)logits[rb + c] - row_max) / row_sum * (double)dy[rb + c];
      RelL2 rl;
      for (int c = 0; c < cols; ++c) {
        const double p = std::exp((double)logits[rb + c] - row_max) / row_sum;
        const double ref = p * ((double)dy[rb + c] - row_dot);
        t->add(got[rb + c], ref, 2e-2, 2e-2);
        rl.add(got[rb + c], ref);
      }
      t->gate(rl.value(), kRelL2);
    }
    return true;
  }
};

struct Lars : Run {
  std::vector<float> w = make_vec(elems()), g = make_vec(elems()), m = make_vec(elems());
  Buf w_d{elems() * 4}, g_d{elems() * 4}, m_d{elems() * 4};
  Output w_out{elems() * 4, 0xff, "weight_out"}, m_out{elems() * 4, 0xff, "momentum_out"};
  int tile_arrays() const override { return lars::kTileArrays; }
  uint32_t vars_bytes() const override {
    return credit::tile_offset<lars::Smem<kBlockThreads, kMaxClusterSize>>();
  }
  void prepare(int rep) override {
    const uint64_t base = (uint64_t)rep * elems();
    for (size_t i = 0; i < elems(); ++i) {
      w[i] = (float)(hash_a(base + i) & 2047u) * (1.0f / 1024.0f) - 1.0f;
      g[i] = (float)(hash_b(base + i) & 2047u) * (1.0f / 1024.0f) - 1.0f;
      m[i] = (float)(hash_c(base + i) & 2047u) * (1.0f / 1024.0f) - 1.0f;
    }
    w_d.write(w.data());
    g_d.write(g.data());
    m_d.write(m.data());
    w_out.reset();
    m_out.reset();
  }
  vx_result_t launch(vx_kernel_h k, const vc::launch_config_t& cfg, uint64_t smid, int slice) override {
    lars::Args a = {w_d.addr, g_d.addr, m_d.addr, w_out.buf.addr, m_out.buf.addr, smid, rows, cols, slice};
    vx_event_h ev = nullptr;
    vx_result_t r = vc::launch_kernel_ex(q, k, cfg, a, &ev);
    if (r == VX_SUCCESS) {
      CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(ev);
    }
    return r;
  }
  bool check(int rep, Tally* t) override {
    const bool g1 = w_out.fetch(rep), g2 = m_out.fetch(rep);
    if (!g1 || !g2)
      return false;
    const float* gw = w_out.data<float>();
    const float* gm = m_out.data<float>();
    for (int r = 0; r < rows; ++r) {
      const size_t rb = (size_t)r * cols;
      double wss = 0, uss = 0;
      for (int c = 0; c < cols; ++c) {
        const double wv = w[rb + c];
        const double u = (double)lars::kMomentumCoeff * m[rb + c] + (double)g[rb + c] + (double)lars::kWeightDecay * wv;
        wss += wv * wv;
        uss += u * u;
      }
      const double trust = (double)lars::kTrustCoeff * std::sqrt(wss) / (std::sqrt(uss) + lars::kEps);
      const double step = (double)lars::kLearningRate * trust;
      RelL2 ru;
      for (int c = 0; c < cols; ++c) {
        const double wv = w[rb + c];
        const double u = (double)lars::kMomentumCoeff * m[rb + c] + (double)g[rb + c] + (double)lars::kWeightDecay * wv;
        t->add(gw[rb + c], wv - step * u, 2e-4, 2e-4);
        t->add(gm[rb + c], u, 2e-4, 2e-4);
        // The update itself: w_out - w is exact in double; the device rounds it once, to w_out.
        const double upd = (double)gw[rb + c] - wv, ref_upd = -step * u;
        t->gate(std::fabs(upd - ref_upd), 2e-5 * std::fabs(ref_upd) + f32_ulp(gw[rb + c]));
        ru.add(gm[rb + c], u);
      }
      t->gate(ru.value(), kRelL2);
    }
    return true;
  }
};

// The H100 harness's host quantizer (fp32), the reference of the int8 values.
int8_t quantize_ref(float value, float inv_scale) {
  float qv = std::nearbyint(value * inv_scale);
  qv = std::fmin(rowwise::kQMax, std::fmax(-rowwise::kQMax, qv));
  return (int8_t)qv;
}

struct Rowwise : Run {
  std::vector<float> in = make_vec(elems());
  Buf in_d{elems() * 4};
  Output out{elems(), 0x80, "output"}, scales{(size_t)rows * 4, 0xff, "scales"};
  int tile_arrays() const override { return rowwise::kTileArrays; }
  uint32_t vars_bytes() const override {
    return credit::tile_offset<rowwise::Smem<kBlockThreads, kMaxClusterSize>>();
  }
  void prepare(int rep) override {
    const uint64_t base = (uint64_t)rep * elems();
    for (size_t i = 0; i < elems(); ++i) {
      const float b = (float)(hash_a(base + i) & 4095u) * (1.0f / 2048.0f) - 1.0f;
      const float tiny = (float)(hash_b(base + i) & 255u) * 1.0e-7f;
      in[i] = b + tiny;
    }
    in_d.write(in.data());
    out.reset();
    scales.reset();
  }
  vx_result_t launch(vx_kernel_h k, const vc::launch_config_t& cfg, uint64_t smid, int slice) override {
    rowwise::Args a = {in_d.addr, out.buf.addr, scales.buf.addr, smid, rows, cols, slice};
    vx_event_h ev = nullptr;
    vx_result_t r = vc::launch_kernel_ex(q, k, cfg, a, &ev);
    if (r == VX_SUCCESS) {
      CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(ev);
    }
    return r;
  }
  // int8 values exact; the scale within 1e-6 (both as the H100 harness).
  bool check(int rep, Tally* t) override {
    const bool g1 = out.fetch(rep), g2 = scales.fetch(rep);
    if (!g1 || !g2)
      return false;
    const int8_t* gq = out.data<int8_t>();
    const float* gs = scales.data<float>();
    for (int r = 0; r < rows; ++r) {
      const size_t rb = (size_t)r * cols;
      float max_abs = 0.0f;
      for (int c = 0; c < cols; ++c)
        max_abs = std::max(max_abs, std::fabs(in[rb + c]));
      const float scale = max_abs / rowwise::kQMax;
      const float inv_scale = max_abs > 0.0f ? rowwise::kQMax / max_abs : 0.0f;
      t->add(gs[r], scale, 1.0e-6, 0.0);
      for (int c = 0; c < cols; ++c)
        t->add(gq[rb + c], quantize_ref(in[rb + c], inv_scale), 0.0, 0.0);
    }
    return true;
  }
};

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

  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority = VX_QUEUE_PRIORITY_NORMAL;
  CHECK(vx_queue_create(dev, &qi, &q));

  Run* run = nullptr;
  if (opt_kernel == "layernorm")
    run = new LayerNorm();
  else if (opt_kernel == "weighted_var")
    run = new WeightedVar();
  else if (opt_kernel == "pearson")
    run = new Pearson();
  else if (opt_kernel == "softmax")
    run = new Softmax();
  else if (opt_kernel == "lars")
    run = new Lars();
  else
    run = new Rowwise();

  const int slice = (cols + P - 1) / P;
  const uint64_t smem = (uint64_t)run->vars_bytes() + (uint64_t)run->tile_arrays() * slice * sizeof(float);
  std::printf("credit_reduction: %s cols=%d rows=%d P=%d slice=%d reps=%d; %llu cores, %llu warps x %llu threads, "
              "%llu bytes of local memory; %llu bytes of local memory per block\n",
              opt_kernel.c_str(), cols, rows, P, slice, reps, (unsigned long long)num_cores,
              (unsigned long long)nw, (unsigned long long)nt, (unsigned long long)lmem_cap,
              (unsigned long long)smem);
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA | VX_ISA_EXT_L2CACHE;
  const char* bad_dev = nullptr;
  if ((isa & need) != need)
    bad_dev = "the device needs DLMEM, the A extension, the DXA and an L2";
  else if (nw * nt != (uint64_t)kBlockThreads)
    bad_dev = "a block is 256 threads: the device needs NUM_WARPS x NUM_THREADS = 256";
  else if (num_cores < (uint64_t)P || max_cluster < (uint64_t)P)
    bad_dev = "the device needs P cores and a cluster of P";
  else if (smem > lmem_cap)
    bad_dev = "the block's local memory exceeds a core's";
  if (bad_dev) {
    std::printf("Error: %s\nFAILED!\n", bad_dev);
    return 1;
  }

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  vx_kernel_h k = nullptr;
  CHECK(vx_module_get_kernel(mod, opt_kernel.c_str(), &k));

  vc::launch_config_t cfg;
  cfg.grid = vc::dim3((uint32_t)(rows * P), 1, 1);
  cfg.block = vc::dim3(kBlockThreads, 1, 1);
  cfg.cluster = vc::dim3((uint32_t)P, 1, 1);
  cfg.dyn_smem = (uint32_t)smem;
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)smem));
  int max_active = 0;
  CHECK(vc::occupancy_max_active_clusters(dev, k, cfg, &max_active));
  std::printf("max_active_clusters = %d\n", max_active);
  if (max_active < 1) {
    std::printf("Error: the cluster cannot be placed\nFAILED!\n");
    return 1;
  }

  const size_t nblocks = (size_t)rows * P;
  Output d_smid(nblocks * sizeof(uint32_t), 0xff, "smid");

  int failed = 0;
  for (int rep = 0; rep < reps; ++rep) {
    run->prepare(rep);
    d_smid.reset();
    CHECK(run->launch(k, cfg, d_smid.buf.addr, slice));
    Tally t;
    const bool out_guards = run->check(rep, &t);
    const bool guards = d_smid.fetch(rep) && out_guards;
    const uint32_t* h_smid = d_smid.data<uint32_t>();
    bool cores = true;
    for (int r = 0; r < rows; ++r) {
      std::set<uint32_t> seen;
      for (int b = 0; b < P; ++b) {
        const uint32_t s = h_smid[(size_t)r * P + b];
        if (s >= num_cores || !seen.insert(s).second) {
          std::printf("Error: rep %d: cluster %d block %d reports smid %u (not a distinct core below %llu)\n", rep,
                      r, b, s, (unsigned long long)num_cores);
          cores = false;
        }
      }
    }
    const bool pass = guards && cores && t.bad == 0 && t.tight <= 1.0;
    failed += !pass;
    std::printf("rep %d: %s bad=%lld max_err=%.6g worst_err/tol=%.4g tight=%.4g\n", rep, pass ? "PASS" : "FAIL",
                t.bad, t.max_err, t.worst, t.tight);
  }

  delete run;
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
