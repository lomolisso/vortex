// hpcops_topk: exact top-2048 indices of m rows of n f32 logits through the
// hpc-ops cluster-8 kernel (topk_filtered_cluster8_exact_kernel<VEC, 2048, 8,
// 1024>), one 8-CTA cluster of 1024-thread blocks per row, one block per core.
// VEC follows the row stride as in hpc-ops: 4 when n % 4 == 0, 2 when
// n % 4 == 2 (VEC 1 is not ported).
//
//   -n <n>       row length (default 8194, the VEC 2 route the H100 profiled)
//   -m <rows>    rows (default 1)
//   -p <input>   randn (default): normal(0, 1) from a fixed seed, as the
//                wrapper's torch.manual_seed(20 + m) randn rows;
//                ties (n >= 32768): a row whose boundary bucket holds 4500
//                scores with repeated values, 2400 of them in rank 5's
//                segment, so that the boundary overflows the primary's buffers
//                (the VEC 2 spilled route) with exact ties at the cut
//   -e <route>   fail unless every row took this route: short, large,
//                register, resident or overflow
//   -r <reps>    launches, each one checked (default 3)
//   -h           this help
//
// Launch i draws new rows from mt19937_64 seeded 20 + m + 1000 i (a fresh
// seed per launch; launch 0 keeps the wrapper's 20 + m). Check per row, as the
// wrapper's _validate: k indices, all in [0, n), unique, equal as a set to the
// CPU top-k, or under ties with the same sorted values. A row of n <= k takes
// the kernel's short route, which writes indices 0..n-1 and -1 after them;
// that is checked exactly. The route each row took (recorded by the kernel)
// must be the one the host derives from the coarse histogram, and on VEC 2
// each rank's count of candidates spilled past its buffers must be the one
// the host derives from its segment (-p ties: some rank must spill). A launch
// also fails if a guard tail after the output, spill or record buffer
// changes, or if a cluster's eight CTAs are not on distinct cores of the
// device.
//
// Exit 0 when every launch passes, 1 otherwise or on a bad option.

#include <vortex2.h>
#include <vx_cuda_runtime.h>
#include <VX_types.h>
#include <vx_cuda_fp.h>
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

using namespace hpcops_topk;

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

constexpr size_t kGuardBytes = 256;
constexpr uint32_t kGuardWord = 0x9e3779b9u;
const char* const kRouteNames[] = {"none", "short", "large", "register", "resident", "overflow"};

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;
int opt_n = 8194, opt_m = 1, reps = 3;
std::string pattern = "randn";
int expect_route = 0;

void usage() {
  std::printf("Usage: [-n n] [-m rows] [-p randn|ties] [-e short|large|register|resident|overflow] [-r reps] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:m:p:e:r:h")) != -1) {
    switch (c) {
    case 'n': opt_n = std::atoi(optarg); break;
    case 'm': opt_m = std::atoi(optarg); break;
    case 'p': pattern = optarg; break;
    case 'e':
      expect_route = 0;
      for (int i = 1; i < 6; ++i)
        if (!std::strcmp(optarg, kRouteNames[i]))
          expect_route = i;
      if (!expect_route) { usage(); std::exit(1); }
      break;
    case 'r': reps = std::atoi(optarg); break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(1);
    }
  }
  if (optind < argc || opt_n < 1 || opt_m < 1 || reps < 1 || (pattern != "randn" && pattern != "ties") ||
      (pattern == "ties" && opt_n < 32768)) {
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

// `body` bytes of the byte `fill`, then the guard word to the end.
std::vector<uint8_t> guarded(size_t total, size_t body, uint8_t fill) {
  std::vector<uint8_t> v(total, fill);
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

// The ties row: 1200 scores in [4, 8) are certainly selected; the boundary
// bucket (f16 coarse key of 1.0) holds 700 scores 1 + j 2^-12 (j < 96) and
// 3800 scores 1 - k 2^-20 (0 < k < 256, just below 1.0 but rounding to it in
// f16), so 848 of 4500 boundary scores are taken and the cut falls inside the
// 1 - k 2^-20 group, among equal values; the rest lie in [-4, 0.98). 2400 of
// the 3800 sit in rank 5's segment (more than one block's candidate buffer),
// the others are spread at random.
void fill_ties(float* row, int n, std::mt19937_64& g) {
  enum { kRest, kBig, kAbove, kBelow };
  const int seg_len = (n + kSplits - 1) / kSplits;
  const int seg5_lo = 5 * seg_len, seg5_hi = std::min(seg5_lo + seg_len, n);
  std::vector<int> label(n, -1);
  std::vector<int> seg5(seg5_hi - seg5_lo);
  std::iota(seg5.begin(), seg5.end(), seg5_lo);
  std::shuffle(seg5.begin(), seg5.end(), g);
  for (int i = 0; i < 2400; ++i)
    label[seg5[i]] = kBelow;
  std::vector<int> pool;
  pool.insert(pool.end(), 1200, kBig);
  pool.insert(pool.end(), 700, kAbove);
  pool.insert(pool.end(), 3800 - 2400, kBelow);
  pool.insert(pool.end(), (size_t)n - 2400 - pool.size(), kRest);
  std::shuffle(pool.begin(), pool.end(), g);
  size_t p = 0;
  for (int i = 0; i < n; ++i)
    if (label[i] < 0)
      label[i] = pool[p++];
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int i = 0; i < n; ++i) {
    switch (label[i]) {
    case kBig:   row[i] = (float)(4.0 + 4.0 * u(g)); break;
    case kAbove: row[i] = (float)(1.0 + std::ldexp((double)(g() % 96), -12)); break;
    case kBelow: row[i] = (float)(1.0 - std::ldexp((double)(1 + g() % 255), -20)); break;
    default:     row[i] = (float)(-4.0 + 4.98 * u(g)); break;
    }
  }
}

uint16_t to_coarse_key(float x) {
  const uint16_t bits = __half_as_ushort(__float2half_rn(x));
  const uint16_t key = (bits & 0x8000) ? bits : static_cast<uint16_t>((bits ^ 0xFFFF) & 0x7FFF);
  return static_cast<uint16_t>(key >> 5);
}

// The route the kernel takes for one row, from its coarse histogram, and on
// VEC 2 each rank's spill: its segment's boundary-bucket scores past the
// candidate buffer (collected only while slots remain to fill).
int predict_route(const float* row, int n, int vec, int* boundary_count, std::vector<int>* spill) {
  *boundary_count = 0;
  spill->assign(kSplits, 0);
  if (n <= kTopK)
    return kRouteShortRow;
  std::vector<int> h(RADIX, 0);
  for (int i = 0; i < n; ++i)
    ++h[to_coarse_key(row[i])];
  int above = 0, t = 0;
  while (above + h[t] <= kTopK)
    above += h[t++];
  const int remaining = kTopK - above;
  *boundary_count = h[t];
  const int seg_len = (n + kSplits - 1) / kSplits;
  for (int r = 0; r < kSplits && remaining != 0; ++r) {
    const int lo = std::min(r * seg_len, n), hi = std::min(lo + seg_len, n);
    int cnt = 0;
    for (int i = lo; i < hi; ++i)
      cnt += to_coarse_key(row[i]) == t;
    (*spill)[r] = std::max(0, cnt - COARSE_CANDIDATE_CAPACITY);
  }
  if (vec == 4)
    return (remaining != 0 && h[t] > kBlockThreads) ? kRouteLargeBoundary : kRouteRegister;
  return (remaining != 0 && h[t] > COARSE_CANDIDATE_CAPACITY) ? kRouteOverflow : kRouteResident;
}

struct RowCheck {
  bool ok = true;
  bool tie = false;
  std::string msg = "ok";
};

RowCheck check_row(const float* row, int n, const int32_t* idx) {
  RowCheck c;
  char buf[160];
  if (n <= kTopK) {
    for (int i = 0; i < kTopK; ++i) {
      if (idx[i] != (i < n ? i : -1)) {
        std::snprintf(buf, sizeof(buf), "slot %d holds %d, expected %d", i, idx[i], i < n ? i : -1);
        c.ok = false, c.msg = buf;
        return c;
      }
    }
    return c;
  }
  std::vector<int> ref(n);
  std::iota(ref.begin(), ref.end(), 0);
  std::partial_sort(ref.begin(), ref.begin() + kTopK, ref.end(),
                    [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
  ref.resize(kTopK);
  std::vector<int> got;
  for (int i = 0; i < kTopK; ++i)
    if (idx[i] >= 0)
      got.push_back(idx[i]);
  if ((int)got.size() != kTopK) {
    std::snprintf(buf, sizeof(buf), "%zu valid indices, expected %d", got.size(), kTopK);
    c.ok = false, c.msg = buf;
    return c;
  }
  for (int v : got) {
    if (v >= n) {
      std::snprintf(buf, sizeof(buf), "index %d out of range", v);
      c.ok = false, c.msg = buf;
      return c;
    }
  }
  std::set<int> gs(got.begin(), got.end()), rs(ref.begin(), ref.end());
  if ((int)gs.size() != kTopK) {
    c.ok = false, c.msg = "duplicate indices";
    return c;
  }
  if (gs == rs)
    return c;
  c.tie = true;
  std::vector<float> gv, rv;
  for (int v : got)
    gv.push_back(row[v]);
  for (int v : ref)
    rv.push_back(row[v]);
  std::sort(gv.begin(), gv.end(), std::greater<float>());
  std::sort(rv.begin(), rv.end(), std::greater<float>());
  if (std::memcmp(gv.data(), rv.data(), gv.size() * sizeof(float)) != 0)
    c.ok = false, c.msg = "top-k values differ from the CPU top-k";
  return c;
}

} // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  parse_args(argc, argv);
  const int n = opt_n, m = opt_m;
  const int vec = (n % 4 == 0) ? 4 : (n % 2 == 0) ? 2 : 1;
  if (vec == 1) {
    std::printf("Error: n = %d is odd: the VEC 1 instantiation is not ported\n", n);
    return 1;
  }

  CHECK(vx_device_open(0, &dev));
  uint64_t num_cores = 0, nw = 0, nt = 0, lmem_cap = 0, max_cluster = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));
  const uint32_t smem = (uint32_t)sizeof(ClusterSmem);
  std::printf("hpcops_topk: n=%d m=%d k=%d VEC=%d input %s reps %d; %llu cores, %llu warps x %llu threads, "
              "%llu bytes of local memory; %u bytes of shared memory per block\n",
              n, m, kTopK, vec, pattern.c_str(), reps, (unsigned long long)num_cores, (unsigned long long)nw,
              (unsigned long long)nt, (unsigned long long)lmem_cap, smem);
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_L2CACHE;
  const char* bad_dev = nullptr;
  if ((isa & need) != need)
    bad_dev = "the device needs DLMEM, the A extension and an L2";
  else if (nw * nt != (uint64_t)kBlockThreads)
    bad_dev = "a block is 1024 threads: the device needs NUM_WARPS x NUM_THREADS = 1024";
  else if (num_cores < (uint64_t)kSplits || max_cluster < (uint64_t)kSplits)
    bad_dev = "the device needs 8 cores and a cluster of 8";
  else if (smem > lmem_cap)
    bad_dev = "RowSmem exceeds the local memory of a core";
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
  CHECK(vx_module_get_kernel(mod, vec == 4 ? "topk_filtered_cluster8_exact_kernel_v4"
                                           : "topk_filtered_cluster8_exact_kernel_v2", &k));

  vc::launch_config_t cfg;
  cfg.grid = vc::dim3(kSplits, (uint32_t)m, 1);
  cfg.block = vc::dim3(kBlockThreads, 1, 1);
  cfg.cluster = vc::dim3(kSplits, 1, 1);
  cfg.dyn_smem = smem;
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)smem));
  int max_active = 0;
  CHECK(vc::occupancy_max_active_clusters(dev, k, cfg, &max_active));
  std::printf("max_active_clusters = %d\n", max_active);
  if (max_active < 1) {
    std::printf("Error: the cluster cannot be placed\nFAILED!\n");
    return 1;
  }

  const size_t logits_elems = (size_t)m * n, out_elems = (size_t)m * kTopK, spill_elems = (size_t)m * 2 * n,
               info_elems = (size_t)m * kSplits * kInfoWords;
  const size_t out_bytes = out_elems * 4, spill_bytes = spill_elems * 4, info_bytes = info_elems * 4;
  std::vector<float> h_logits(logits_elems);
  std::vector<int32_t> h_out(out_elems), h_ke(m, n), h_nvr(1, m);
  std::vector<uint32_t> h_info(info_elems);
  Buf d_logits(logits_elems * 4), d_out(out_bytes + kGuardBytes), d_ke((size_t)m * 4), d_nvr(4),
      d_spill(spill_bytes + kGuardBytes), d_info(info_bytes + kGuardBytes);
  std::vector<uint8_t> raw_out(d_out.size), raw_spill(d_spill.size), raw_info(d_info.size);
  d_ke.write(h_ke.data());
  d_nvr.write(h_nvr.data());

  kernel_arg_t arg;
  std::memset(&arg, 0, sizeof(arg));
  arg.input = d_logits.addr;
  arg.output = d_out.addr;
  arg.ke = d_ke.addr;
  arg.spill_buffer = d_spill.addr;
  arg.num_valid_rows = d_nvr.addr;
  arg.info = d_info.addr;
  arg.row_stride = (uint32_t)n;
  arg.out_stride = (uint32_t)kTopK;
  arg.n_cols = (uint32_t)n;

  int failed = 0;
  for (int rep = 0; rep < reps; ++rep) {
    std::mt19937_64 g(20ull + (unsigned long long)m + 1000ull * (unsigned long long)rep);
    if (pattern == "randn") {
      std::normal_distribution<double> nd(0.0, 1.0);
      for (auto& x : h_logits)
        x = (float)nd(g);
    } else {
      for (int r = 0; r < m; ++r)
        fill_ties(&h_logits[(size_t)r * n], n, g);
    }
    d_logits.write(h_logits.data());
    d_out.write(guarded(d_out.size, out_bytes, 0xff).data());  // -1 in every slot
    d_spill.write(guarded(d_spill.size, spill_bytes, 0xa5).data());
    d_info.write(guarded(d_info.size, info_bytes, 0xff).data());

    vx_event_h ev = nullptr;
    CHECK(vc::launch_kernel_ex(q, k, cfg, arg, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    d_out.read(raw_out.data());
    d_spill.read(raw_spill.data());
    d_info.read(raw_info.data());
    std::memcpy(h_out.data(), raw_out.data(), out_bytes);
    std::memcpy(h_info.data(), raw_info.data(), info_bytes);
    bool ok = guard_intact(raw_out, out_bytes, "out", rep) && guard_intact(raw_spill, spill_bytes, "spill", rep) &&
              guard_intact(raw_info, info_bytes, "info", rep);

    int ties = 0;
    std::string routes;
    size_t min_distinct = kSplits;
    for (int r = 0; r < m; ++r) {
      const float* row = &h_logits[(size_t)r * n];
      const RowCheck c = check_row(row, n, &h_out[(size_t)r * kTopK]);
      if (!c.ok) {
        std::printf("Error: rep %d: row %d: %s\n", rep, r, c.msg.c_str());
        ok = false;
      }
      ties += c.tie;
      int boundary = 0;
      std::vector<int> want_spill;
      const int want = predict_route(row, n, vec, &boundary, &want_spill);
      const uint32_t got = h_info[((size_t)r * kSplits) * kInfoWords + 1];
      const char* got_name = got < 6 ? kRouteNames[got] : "invalid";
      if ((int)got != want || (expect_route && (int)got != expect_route)) {
        std::printf("Error: rep %d: row %d took route %s; the coarse histogram (boundary bucket %d) gives %s%s%s\n",
                    rep, r, got_name, boundary, kRouteNames[want], expect_route ? ", -e asks for " : "",
                    expect_route ? kRouteNames[expect_route] : "");
        ok = false;
      }
      routes += (r ? "," : "") + std::string(got_name) + "(" + std::to_string(boundary) + ")";
      if (vec == 2 && want != kRouteShortRow) {
        int total = 0;
        for (int c2 = 0; c2 < kSplits; ++c2) {
          const uint32_t sp = h_info[((size_t)r * kSplits + c2) * kInfoWords + 2];
          if ((int)sp != want_spill[c2]) {
            std::printf("Error: rep %d: row %d rank %d spilled %d candidates; its segment gives %d\n", rep, r, c2,
                        (int)sp, want_spill[c2]);
            ok = false;
          }
          total += want_spill[c2];
          if (want_spill[c2])
            routes += " spill1[" + std::to_string(c2) + "]=" + std::to_string((int)sp);
        }
        if (pattern == "ties" && total == 0) {
          std::printf("Error: rep %d: row %d: the ties row spilled on no rank\n", rep, r);
          ok = false;
        }
      }
      std::set<uint32_t> cores;
      for (int c2 = 0; c2 < kSplits; ++c2) {
        const uint32_t smid = h_info[((size_t)r * kSplits + c2) * kInfoWords];
        if (smid >= num_cores) {
          std::printf("Error: rep %d: row %d rank %d reports smid %u, not one of the %llu cores\n", rep, r, c2, smid,
                      (unsigned long long)num_cores);
          ok = false;
        }
        cores.insert(smid);
      }
      min_distinct = std::min(min_distinct, cores.size());
    }
    const bool pass = ok && min_distinct == (size_t)kSplits;
    failed += !pass;
    std::printf("rep %d: %s route=%s rows_with_tie_resolution=%d smid_distinct=%zu\n", rep, pass ? "PASS" : "FAIL",
                routes.c_str(), ties, min_distinct);
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
