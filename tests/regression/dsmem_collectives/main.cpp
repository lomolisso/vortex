// dsmem_collectives: the 29 DSMEM collectives of the H100 study
// (h100-dsmem-collectives/collectives, harness/main.cu at commit 2b9a97e),
// one cluster of CLUSTER_N blocks of 256 threads, one block per core.
// N, M and T are compile-time (COLL_N, COLL_M, COLL_T in the Makefile).
//
//   -a <name>   the collective, <op>__<alg> (required)
//   -p A|B      input pattern of launch i: A the small integers
//               T((k*7+3+i) % 17 - 8) (default), B floats drawn uniformly
//               in [-1, 1] by mt19937_64 seeded 20261025 + i, float only
//   -r <reps>   launches, each one checked (default 3)
//   -s          per rank the core and every stamp relative to the earliest
//               entry stamp, and the latency max exit - min entry
//   -h          this help
//
// Exit 0 when every launch passes; 1 on a failure, an unknown name or a bad
// option; 2 when the build violates the collective's constraints
// (SKIP:static_assert); 3 when its shared memory exceeds the local memory of
// a core (SKIP:smem).
//
// Launch 0 uses the H100 harness input; every later launch gets a new one
// and its own reference. The H100 harness finds races through run-to-run
// timing variation, which the deterministic Vortex models lack, and a launch
// on the same input would find the previous launch's correct values still in
// shared memory. Pattern A stays within [-8, 8], so its sums are exact in f32
// and i32. A launch also fails if a guard tail after out or ts changes, or if
// a rank's smid is not a distinct core of the device.

#include <vortex2.h>
#include <vx_cuda_runtime.h>
#include <VX_types.h>
#include "common.h"
#include "dsmem_primitives.h"

#include "coll/allgather__bruck__mpich.h"
#include "coll/allgather__direct_pull__generic.h"
#include "coll/allgather__direct_replicated_push__credit.h"
#include "coll/allgather__direct_shift__clusterfusion.h"
#include "coll/allgather__gather_bcast_binomial__mpich.h"
#include "coll/allgather__recursive_doubling__clusterfusion.h"
#include "coll/allgather__ring__nccl.h"
#include "coll/allreduce__2d_mesh__generic.h"
#include "coll/allreduce__atomic_all_push__generic.h"
#include "coll/allreduce__direct_replicated_push__credit.h"
#include "coll/allreduce__one_shot_pull__trtllm.h"
#include "coll/allreduce__pull_reduce_scatter_push_allgather__hpcops.h"
#include "coll/allreduce__rabenseifner__mpich.h"
#include "coll/allreduce__recursive_doubling__clusterfusion.h"
#include "coll/allreduce__reduce_bcast_binomial__mpich.h"
#include "coll/allreduce__ring__nccl.h"
#include "coll/allreduce__shift_accumulate_atomic__clusterfusion.h"
#include "coll/allreduce__shift_accumulate_bulk__clusterfusion.h"
#include "coll/allreduce__two_shot_push__trtllm.h"
#include "coll/reducescatter__direct_pull__generic.h"
#include "coll/reducescatter__direct_push__generic.h"
#include "coll/reducescatter__recursive_halving__mpich.h"
#include "coll/reducescatter__ring__nccl.h"
#include "coll/alltoall__bruck__mpich.h"
#include "coll/alltoall__direct_pull__generic.h"
#include "coll/alltoall__direct_push__generic.h"
#include "coll/alltoall__pairwise_exchange__fa4.h"
#include "coll/alltoall__pairwise_xor__mpich.h"
#include "coll/alltoall__shift__generic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <type_traits>
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

typedef ELEM_T T;
constexpr int N = CLUSTER_N, M = MSG_ELEMS;
constexpr uint32_t kBlockThreads = 256;
constexpr bool kIsInt = std::is_same<T, int>::value;
static_assert(kIsInt || std::is_same<T, float>::value, "ELEM_T must be float or int");
const char* const kOpNames[] = {"allreduce", "allgather", "reducescatter", "alltoall"};

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;
vx_module_h mod = nullptr;

typedef vx_result_t (*launch_fn)(vx_kernel_h, const vc::launch_config_t&, const kernel_arg_t&, vx_event_h*);
typedef vx_result_t (*occupancy_fn)(vx_kernel_h, const vc::launch_config_t&, int*);

template <typename D>
vx_result_t launch_as(vx_kernel_h k, const vc::launch_config_t& cfg, const kernel_arg_t& arg, vx_event_h* ev) {
  return vc::launch_kernel_ex<D>(q, k, cfg, arg, ev);
}

template <typename D>
vx_result_t occupancy_as(vx_kernel_h k, const vc::launch_config_t& cfg, int* n) {
  return vc::occupancy_max_active_clusters<D>(dev, k, cfg, n);
}

// What the host needs of each collective: legality and shared memory at this
// build's N, M, T, the op, and the launch through its __cluster_dims__.
struct Alg {
  const char* name;
  int op;
  bool type_agnostic;
  const char* illegal;
  size_t smem;
  launch_fn launch;
  occupancy_fn occupancy;
};

#define DSMEM_COLL_ROW(n)                                                              \
  {#n, n::kOp, n::kTypeAgnostic, n::illegal<CLUSTER_N, MSG_ELEMS, ELEM_T>(),           \
   sizeof(n::CollectiveSmem), &launch_as<n##_cluster_dims>, &occupancy_as<n##_cluster_dims>},
const Alg kAlgs[] = {DSMEM_COLL_ALGS(DSMEM_COLL_ROW)};
#undef DSMEM_COLL_ROW
static_assert(sizeof(kAlgs) / sizeof(kAlgs[0]) == (size_t)kNumAlgs, "one row per collective");

const char* opt_alg = nullptr;
char pattern = 'A';
int reps = 3;
bool stamps = false;

void usage() {
  std::printf("Usage: -a <op>__<alg> [-p A|B] [-r reps] [-s] [-h]\n  N=%d M=%d T=%s; collectives:\n", N, M,
              kIsInt ? "i32" : "f32");
  for (int i = 0; i < kNumAlgs; ++i)
    std::printf("    %s\n", kAlgNames[i]);
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "a:p:r:sh")) != -1) {
    switch (c) {
    case 'a': opt_alg = optarg; break;
    case 'p':
      if ((optarg[0] != 'A' && optarg[0] != 'B') || optarg[1]) { usage(); std::exit(1); }
      pattern = optarg[0];
      break;
    case 'r':
      reps = std::atoi(optarg);
      if (reps < 1) { usage(); std::exit(1); }
      break;
    case 's': stamps = true; break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(1);
    }
  }
  if (optind < argc || !opt_alg) {
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
  void fill(uint8_t b) {
    std::vector<uint8_t> v(size, b);
    write(v.data());
  }
  void read(void* p) const {
    vx_event_h e = nullptr;
    CHECK(vx_enqueue_read(q, p, h, 0, size, 0, nullptr, &e));
    CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e);
  }
};

void fill_input(std::vector<T>& in, int rep) {
  if (pattern == 'A') {
    for (size_t k = 0; k < in.size(); ++k)
      in[k] = T((long long)((k * 7 + 3 + (size_t)rep) % 17) - 8);
  } else {
    std::mt19937_64 g(20261025ull + (unsigned long long)rep);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (auto& x : in)
      x = T(u(g));
  }
}

// Expected outputs per common.h, computed in double; sums of pattern A and of
// int inputs are exact.
std::vector<double> reference(int op, const std::vector<T>& in, size_t out_elems) {
  std::vector<double> ref(out_elems, 0.0);
  for (int b = 0; b < N; ++b) {
    for (int i = 0; i < M; ++i) {
      if (op == 0) {
        double s = 0;
        for (int r = 0; r < N; ++r)
          s += (double)in[(size_t)r * M + i];
        ref[(size_t)b * M + i] = s;
      } else if (op == 1) {
        for (int r = 0; r < N; ++r)
          ref[(size_t)b * N * M + (size_t)r * M + i] = (double)in[(size_t)r * M + i];
      } else if (op == 2) {
        double s = 0;
        for (int r = 0; r < N; ++r)
          s += (double)in[(size_t)r * N * M + (size_t)b * M + i];
        ref[(size_t)b * M + i] = s;
      } else {
        for (int r = 0; r < N; ++r)
          ref[(size_t)b * N * M + (size_t)r * M + i] = (double)in[(size_t)r * N * M + (size_t)b * M + i];
      }
    }
  }
  return ref;
}

// Guard words after the end of out and ts catch a store past either buffer.
constexpr size_t kGuardBytes = 256;
constexpr uint32_t kGuardWord = 0x9e3779b9u;

// Zero `body` bytes, then fill the rest of the buffer with the guard word.
void reset_guarded(Buf& b, size_t body) {
  std::vector<uint8_t> v(b.size, 0);
  for (size_t i = body; i + 4 <= b.size; i += 4)
    std::memcpy(&v[i], &kGuardWord, 4);
  b.write(v.data());
}

// Reports the first changed guard word of a buffer read back after a launch.
bool guard_intact(const std::vector<uint8_t>& v, size_t body, const char* what, int rep) {
  for (size_t i = body; i + 4 <= v.size(); i += 4) {
    uint32_t w;
    std::memcpy(&w, &v[i], 4);
    if (w != kGuardWord) {
      std::printf("Error: rep %d: guard word %zu after %s changed: 0x%08x (want 0x%08x)\n", rep, (i - body) / 4,
                  what, w, kGuardWord);
      return false;
    }
  }
  return true;
}

struct Result {
  long long bad = 0;
  double max_abs = 0, max_rel = 0;
};

// Exact in the element type, except a tolerance of 1e-5 (1 + |ref|) ceil(log2 N)
// for reductions of pattern B.
Result check(bool reduce, const std::vector<T>& out, const std::vector<double>& ref) {
  Result c;
  const bool tol = reduce && !kIsInt && pattern == 'B';
  const double ln = std::ceil(std::log2((double)N));
  for (size_t k = 0; k < out.size(); ++k) {
    const double got = (double)out[k];
    const double want = kIsInt ? ref[k] : (double)(float)ref[k];
    const double err = std::fabs(got - ref[k]);
    const bool bad = tol ? !(err <= 1e-5 * (1.0 + std::fabs(ref[k])) * ln) : !(got == want);
    c.bad += bad;
    if (!(err <= c.max_abs))
      c.max_abs = err;
    const double rel = err / (1.0 + std::fabs(ref[k]));
    if (!(rel <= c.max_rel))
      c.max_rel = rel;
  }
  return c;
}

void print_stamps(const std::vector<uint64_t>& ts, const std::vector<uint32_t>& smid) {
  uint64_t t0 = ~0ull, t1 = 0;
  for (int r = 0; r < N; ++r) {
    t0 = std::min<uint64_t>(t0, ts[(size_t)r * dsmem::kStamps + 0]);
    t1 = std::max<uint64_t>(t1, ts[(size_t)r * dsmem::kStamps + 1]);
  }
  for (int r = 0; r < N; ++r) {
    std::printf("  rank %2d smid %3u:", r, smid[r]);
    for (int k = 0; k < dsmem::kStamps; ++k) {
      const uint64_t t = ts[(size_t)r * dsmem::kStamps + k];
      if (!t)
        continue;
      const long long d = (long long)(t - t0);
      if (k == 0)
        std::printf(" entry=%lld", d);
      else if (k == 1)
        std::printf(" exit=%lld", d);
      else
        std::printf(" round%d=%lld", k - 2, d);
    }
    std::printf("\n");
  }
  std::printf("  latency_ns=%lld\n", (long long)(t1 - t0));
}

} // namespace

int main(int argc, char** argv) {
  // A hung launch is killed by the runner's timeout: keep what was printed.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  parse_args(argc, argv);

  const Alg* alg = nullptr;
  for (const Alg& a : kAlgs)
    if (!std::strcmp(a.name, opt_alg))
      alg = &a;
  if (!alg) {
    std::printf("Error: unknown collective '%s'\n", opt_alg);
    usage();
    return 1;
  }
  if (pattern == 'B' && kIsInt) {
    std::printf("Error: pattern B is float only\n");
    return 1;
  }
  if (kIsInt && alg->type_agnostic) {
    std::printf("Error: %s moves data only and is built for f32 only\n", alg->name);
    return 1;
  }
  if (alg->illegal) {
    std::printf("SKIP:static_assert %s N=%d M=%d: %s\n", alg->name, N, M, alg->illegal);
    return 2;
  }

  const int op = alg->op;
  const bool reduce = (op == 0 || op == 2);
  const size_t in_elems = (op == 0 || op == 1) ? (size_t)N * M : (size_t)N * N * M;
  const size_t out_elems = (op == 0 || op == 2) ? (size_t)N * M : (size_t)N * N * M;
  const uint32_t smem = (uint32_t)alg->smem;

  CHECK(vx_device_open(0, &dev));
  uint64_t num_cores = 0, nw = 0, nt = 0, lmem_cap = 0, max_cluster = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));
  if (alg->smem > lmem_cap) {
    std::printf("SKIP:smem %s N=%d M=%d: %zu > %llu\n", alg->name, N, M, alg->smem, (unsigned long long)lmem_cap);
    vx_device_release(dev);
    return 3;
  }
  std::printf("dsmem_collectives: %s (%s) N=%d M=%d T=%s pattern %c reps %d; %llu cores, %llu warps x %llu "
              "threads, %llu bytes of local memory; %u bytes of shared memory per block\n",
              alg->name, kOpNames[op], N, M, kIsInt ? "i32" : "f32", pattern, reps,
              (unsigned long long)num_cores, (unsigned long long)nw, (unsigned long long)nt,
              (unsigned long long)lmem_cap, smem);
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA | VX_ISA_EXT_L2CACHE;
  const char* bad_dev = nullptr;
  if ((isa & need) != need)
    bad_dev = "the device needs DLMEM, the A extension, the DXA and an L2";
  else if (nw * nt != kBlockThreads)
    bad_dev = "a block is 256 threads: the device needs NUM_WARPS x NUM_THREADS = 256";
  else if (num_cores < (uint64_t)N || max_cluster < (uint64_t)N)
    bad_dev = "the device needs N cores and a cluster of N";
  if (bad_dev) {
    std::printf("Error: %s\nFAILED!\n", bad_dev);
    vx_device_release(dev);
    return 1;
  }

  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority = VX_QUEUE_PRIORITY_NORMAL;
  CHECK(vx_queue_create(dev, &qi, &q));
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  vx_kernel_h k = nullptr;
  CHECK(vx_module_get_kernel(mod, alg->name, &k));

  vc::launch_config_t cfg;
  cfg.grid = vc::dim3(N, 1, 1);
  cfg.block = vc::dim3(kBlockThreads, 1, 1);
  cfg.cluster = vc::dim3(N, 1, 1);
  cfg.dyn_smem = smem;
  cfg.non_portable = N > 8;
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)smem));
  if (N > 8)
    CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::non_portable_cluster_size_allowed, 1));
  int max_active = 0;
  CHECK(alg->occupancy(k, cfg, &max_active));
  std::printf("max_active_clusters = %d\n", max_active);
  if (max_active < 1) {
    std::printf("Error: the cluster cannot be placed\nFAILED!\n");
    return 1;
  }

  std::vector<T> h_in(in_elems), h_out(out_elems);
  std::vector<uint64_t> h_ts((size_t)N * dsmem::kStamps);
  std::vector<uint32_t> h_smid(N);
  const size_t out_bytes = out_elems * sizeof(T), ts_bytes = h_ts.size() * sizeof(uint64_t);

  Buf d_in(in_elems * sizeof(T)), d_out(out_bytes + kGuardBytes), d_ts(ts_bytes + kGuardBytes),
      d_smid(h_smid.size() * sizeof(uint32_t));
  std::vector<uint8_t> raw_out(d_out.size), raw_ts(d_ts.size);

  kernel_arg_t arg;
  std::memset(&arg, 0, sizeof(arg));
  arg.in = d_in.addr;
  arg.out = d_out.addr;
  arg.ts = d_ts.addr;
  arg.smid = d_smid.addr;
  for (int i = 0; i < kNumAlgs; ++i)
    if (&kAlgs[i] == alg)
      arg.alg = (uint32_t)i;

  int failed = 0;
  for (int rep = 0; rep < reps; ++rep) {
    fill_input(h_in, rep);
    const std::vector<double> ref = reference(op, h_in, out_elems);
    d_in.write(h_in.data());
    reset_guarded(d_out, out_bytes);
    reset_guarded(d_ts, ts_bytes);
    d_smid.fill(0xff);
    vx_event_h ev = nullptr;
    CHECK(alg->launch(k, cfg, arg, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    d_out.read(raw_out.data());
    d_ts.read(raw_ts.data());
    d_smid.read(h_smid.data());
    std::memcpy(h_out.data(), raw_out.data(), out_bytes);
    std::memcpy(h_ts.data(), raw_ts.data(), ts_bytes);
    const bool out_guard = guard_intact(raw_out, out_bytes, "out", rep);
    const bool ts_guard = guard_intact(raw_ts, ts_bytes, "ts", rep);
    bool cores = true;
    for (int r = 0; r < N; ++r) {
      if (h_smid[r] >= num_cores) {
        std::printf("Error: rep %d: rank %d reports smid %u, not one of the %llu cores\n", rep, r, h_smid[r],
                    (unsigned long long)num_cores);
        cores = false;
      }
    }
    const Result c = check(reduce, h_out, ref);
    const size_t distinct = std::set<uint32_t>(h_smid.begin(), h_smid.end()).size();
    const bool pass = c.bad == 0 && distinct == (size_t)N && cores && out_guard && ts_guard;
    failed += !pass;
    std::printf("rep %d: %s bad=%lld max_abs=%.6g max_rel=%.6g smid_distinct=%zu\n", rep, pass ? "PASS" : "FAIL",
                c.bad, c.max_abs, c.max_rel, distinct);
    if (stamps)
      print_stamps(h_ts, h_smid);
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
