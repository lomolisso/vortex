// cluster_sync: the hardware cluster barrier (cluster.sync(), and the split
// barrier_arrive() / barrier_wait()) on a DLMEM device, RV64.
//
//   default  three launches in one process: DEFAULT (same-core) clusters of
//            K = min(4, NUM_WARPS / 2) blocks of 2 warps, SPREAD with K = 1,
//            and SPREAD of the device's largest K
//   -k K     one launch in clusters of K x 1 x 1 instead
//   -w W     warps per block (default 1)
//   -n G     clusters per launch; the grid is G * K blocks (default 6)
//   -s / -d  SPREAD (default) / DEFAULT policy for the -k launch
//   -i P     barrier phases per block (default 4, 64 with -p)
//   -l B     local-memory bytes per block (default 0)
//   -m mode  sync (default), split, skew, mixed or exit
//   -r       sync with the fence-free arrive and wait instead of cluster.sync()
//   -p       parity mode: no atomics and no rank-dependent work, so the
//            instruction count does not depend on timing
//   -v       print every cluster's placement (core:slot by rank)
//   -x       pass a launch whose clusters cannot be co-resident to the
//            runtime, which must refuse it with VX_ERR_INVALID_VALUE
//
// With the A extension, every warp adds itself to its cluster's counter for
// the phase before arriving and reads the counter back after the release; it
// must hold every warp of the cluster. Every arrive must return the phase
// token of its phase. The host checks the kernel's error counts, the counters,
// that each block ran once, its rank, and that a DEFAULT cluster stays on one
// core. A DEFAULT cluster must fit on one core at once (K * W warps and K
// slots), or its barrier would deadlock: the app refuses such a launch unless
// -x leaves the refusal to the runtime.

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <unistd.h>
#include <utility>
#include <vector>

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

namespace {

constexpr uint32_t kSpin       = 128;
constexpr uint32_t kParityWork = 16;

const char* kernel_file = "kernel.vxbin";
uint32_t    req_k       = 0;
uint32_t    req_w       = 1;
uint32_t    req_g       = 6;
uint32_t    req_p       = 0;    // 0: not given
uint32_t    req_lmem    = 0;
uint32_t    req_mode    = MODE_SYNC;
int         req_policy  = -1;   // -1: not given
bool        relaxed     = false;
bool        parity_mode = false;
bool        verbose     = false;
bool        expect_reject = false;

const char* const kModeNames[] = {"sync", "split", "skew", "mixed", "exit"};

struct Caps {
  uint64_t num_cores = 0, num_clusters = 0, num_warps = 0, num_threads = 0;
  uint64_t lmem_size = 0, max_cluster = 0, isa = 0;
  bool dlmem() const { return (isa & VX_ISA_EXT_DLMEM) != 0; }
  bool atomics() const { return (isa & VX_ISA_STD_A) != 0; }
  uint32_t ncc() const { return (uint32_t)(num_cores / num_clusters); }
};

struct Launch {
  uint32_t k, g, w;
  vx_cluster_policy_e policy;
  const char* name;
};

void usage() {
  std::printf("Usage: [-k cluster size] [-w warps per block] [-n clusters] [-s: SPREAD | -d: DEFAULT]"
              " [-i phases] [-l local-memory bytes] [-m sync|split|skew|mixed|exit] [-r: relaxed]"
              " [-p: parity] [-v: placement] [-x: expect the runtime to refuse] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  bool bad = false;
  while ((c = getopt(argc, argv, "k:w:n:sdi:l:m:rpvxh")) != -1) {
    switch (c) {
      case 'k': req_k       = (uint32_t)std::atoi(optarg); break;
      case 'w': req_w       = (uint32_t)std::atoi(optarg); break;
      case 'n': req_g       = (uint32_t)std::atoi(optarg); break;
      case 's': req_policy  = VX_CLUSTER_POLICY_SPREAD;    break;
      case 'd': req_policy  = VX_CLUSTER_POLICY_DEFAULT;   break;
      case 'i': req_p       = (uint32_t)std::atoi(optarg); bad |= (req_p == 0); break;
      case 'l': req_lmem    = (uint32_t)std::atoi(optarg); break;
      case 'm': {
        req_mode = ~0u;
        for (uint32_t m = 0; m < sizeof(kModeNames) / sizeof(kModeNames[0]); ++m) {
          if (std::strcmp(optarg, kModeNames[m]) == 0)
            req_mode = m;
        }
        bad |= (req_mode == ~0u);
        break;
      }
      case 'r': relaxed     = true;                        break;
      case 'p': parity_mode = true;                        break;
      case 'v': verbose     = true;                        break;
      case 'x': expect_reject = true;                      break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  // Rank-dependent work and the exit mode's lone arrivals would make the
  // instruction count depend on timing.
  bad |= parity_mode && req_mode != MODE_SYNC;
  bad |= relaxed && req_mode != MODE_SYNC;
  bad |= expect_reject && req_k == 0 && req_policy < 0;
  if (bad || optind < argc || req_w == 0 || req_g == 0 || req_lmem % sizeof(uint32_t) != 0) {
    usage();
    std::exit(-1);
  }
}

const char* policy_name(uint32_t p) {
  return (p == VX_CLUSTER_POLICY_SPREAD) ? "SPREAD" : (p == VX_CLUSTER_POLICY_DEFAULT) ? "DEFAULT" : "?";
}

// Slots a block of the launch's local-memory size leaves on a core.
uint32_t usable_slots(const Caps& caps) {
  const uint64_t stride = ((uint64_t)req_lmem + VX_CFG_MEM_BLOCK_SIZE - 1)
                        / VX_CFG_MEM_BLOCK_SIZE * VX_CFG_MEM_BLOCK_SIZE;
  if (stride == 0)
    return (uint32_t)caps.num_warps;
  return (uint32_t)std::min<uint64_t>(caps.num_warps, caps.lmem_size / stride);
}

// Why the launch's clusters cannot all be resident at once, or nullptr. Their
// barrier would never complete.
const char* not_coresident(const Caps& caps, const Launch& L) {
  if (L.w > caps.num_warps)
    return "a block is wider than a core";
  if (L.policy == VX_CLUSTER_POLICY_DEFAULT) {
    if (L.k * L.w > caps.num_warps)
      return "a DEFAULT cluster needs more warps than a core has";
    if (L.k > usable_slots(caps))
      return "a DEFAULT cluster needs more slots than a core has";
  } else if (L.k > caps.max_cluster) {
    return "K is above VX_CAPS_DLMEM_MAX_CLUSTER";
  }
  return nullptr;
}

struct Job {
  vx_buffer_h rec_buf = nullptr, err_buf = nullptr, count_buf = nullptr;
  std::vector<uint32_t> recs, errs, counts;
  sync_arg_t arg = {};
  uint32_t num_blocks = 0, phases = 0;
};

template <typename T>
void upload(vx_device_h dev, vx_queue_h q, const std::vector<T>& v, vx_buffer_h* buf,
            uint64_t* addr) {
  const uint64_t bytes = v.size() * sizeof(T);
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, buf));
  CHECK(vx_enqueue_write(q, *buf, 0, v.data(), bytes, 0, nullptr, nullptr));
  CHECK(vx_buffer_address(*buf, addr));
}

template <typename T>
vx_event_h download(vx_queue_h q, std::vector<T>& v, vx_buffer_h buf, vx_event_h after) {
  vx_event_h ev = nullptr;
  CHECK(vx_enqueue_read(q, v.data(), buf, 0, v.size() * sizeof(T), 1, &after, &ev));
  return ev;
}

// The mismatches of one finished launch.
int check(const Caps& caps, const Launch& L, const Job& j) {
  const uint32_t k = L.k, w = L.w, ncc = caps.ncc();
  int errors = 0;
  auto rec_of = [&](uint32_t b) { return &j.recs[(size_t)b * REC_WORDS]; };
  auto report = [&](uint32_t b, const char* what, uint64_t got, uint64_t want) {
    if (errors < 16) {
      const uint32_t* rec = rec_of(b);
      std::printf("*** %s: block %u (cluster %u, core %u, slot %u): %s = %llu, expected %llu\n",
                  L.name, b, b / k, rec[REC_CORE], rec[REC_CTA_ID], what,
                  (unsigned long long)got, (unsigned long long)want);
    }
    ++errors;
  };

  for (uint32_t b = 0; b < j.num_blocks; ++b) {
    const uint32_t* rec = rec_of(b);
    if (rec[REC_RUNS] != 1)
      report(b, "times run", rec[REC_RUNS], 1);
    if (rec[REC_BLOCK] != b) {
      report(b, "blockIdx.x", rec[REC_BLOCK], b);
      continue;
    }
    if (rec[REC_RANK] != b % k)
      report(b, "cluster rank", rec[REC_RANK], b % k);
    if (rec[REC_SIZE] != k)
      report(b, "cluster size", rec[REC_SIZE], k);
    if (rec[REC_PHASES] != j.phases)
      report(b, "phases", rec[REC_PHASES], j.phases);
    if (rec[REC_CORE] >= caps.num_cores)
      report(b, "core id", rec[REC_CORE], caps.num_cores - 1);
    for (uint32_t i = 0; i < w; ++i) {
      if (j.errs[(size_t)b * w + i] != 0)
        report(b, "kernel errors (bad counter or token) in a warp", j.errs[(size_t)b * w + i], 0);
    }
  }
  for (uint32_t g = 0; g < L.g && !j.counts.empty(); ++g) {
    for (uint32_t i = 0; i < j.phases; ++i) {
      const uint32_t got = j.counts[(size_t)g * j.phases + i];
      if (got != k * w) {
        if (errors < 16)
          std::printf("*** %s: cluster %u phase %u: %u arrivals counted, expected %u\n",
                      L.name, g, i, got, k * w);
        ++errors;
      }
    }
  }
  if (errors)
    return errors;

  for (uint32_t g = 0; g < L.g; ++g) {
    const uint32_t* first = rec_of(g * k);
    std::set<std::pair<uint32_t, uint32_t>> slots;
    for (uint32_t r = 0; r < k; ++r) {
      const uint32_t b = g * k + r;
      const uint32_t* rec = rec_of(b);
      if (!slots.insert({rec[REC_CORE], rec[REC_CTA_ID]}).second)
        report(b, "slot already held by a block of its cluster", rec[REC_CTA_ID], ~0u);
      if (rec[REC_CORE] / ncc != first[REC_CORE] / ncc)
        report(b, "hardware cluster", rec[REC_CORE] / ncc, first[REC_CORE] / ncc);
      if (L.policy == VX_CLUSTER_POLICY_DEFAULT && rec[REC_CORE] != first[REC_CORE])
        report(b, "core (DEFAULT keeps a cluster on one core)", rec[REC_CORE], first[REC_CORE]);
    }
    if (verbose) {
      std::printf("%s: cluster %u:", L.name, g);
      for (uint32_t r = 0; r < k; ++r)
        std::printf(" %u:%u", rec_of(g * k + r)[REC_CORE], rec_of(g * k + r)[REC_CTA_ID]);
      std::printf("\n");
    }
  }
  return errors;
}

// One launch end to end: the mismatches, or 1 when it could not run.
int run_launch(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps,
               const Launch& L) {
  const bool counting = caps.atomics() && !parity_mode;
  Job j;
  j.num_blocks = L.g * L.k;
  j.phases = req_p ? req_p : (parity_mode ? 64 : 4);
  std::printf("%s: %s, K=%u, %u clusters of %u-warp blocks, %u phases of %s%s%s, %u local-memory bytes\n",
              L.name, policy_name(L.policy), L.k, L.g, L.w, j.phases, kModeNames[req_mode],
              relaxed ? " (relaxed)" : "", counting ? " with arrival counters" : "", req_lmem);
  const char* why = not_coresident(caps, L);
  if (expect_reject && !why) {
    std::printf("*** %s: -x needs a launch whose clusters cannot be co-resident\n", L.name);
    return 1;
  }
  if (why && !expect_reject) {
    std::printf("*** %s: refused, its clusters cannot be co-resident: %s\n", L.name, why);
    return 1;
  }

  j.recs.assign((size_t)j.num_blocks * REC_WORDS, 0xffffffffu);
  for (uint32_t b = 0; b < j.num_blocks; ++b)
    j.recs[(size_t)b * REC_WORDS + REC_RUNS] = 0;
  j.errs.assign((size_t)j.num_blocks * L.w, 0);
  upload(dev, q, j.recs, &j.rec_buf, &j.arg.rec_addr);
  upload(dev, q, j.errs, &j.err_buf, &j.arg.err_addr);
  if (counting) {
    j.counts.assign((size_t)L.g * j.phases, 0);
    upload(dev, q, j.counts, &j.count_buf, &j.arg.count_addr);
  }
  j.arg.k           = L.k;
  j.arg.phases      = j.phases;
  j.arg.mode        = req_mode;
  j.arg.spin        = parity_mode ? kParityWork : kSpin;
  j.arg.num_threads = (uint32_t)caps.num_threads;
  j.arg.relaxed     = relaxed;

  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kernel;
  li.args_host      = &j.arg;
  li.args_size      = sizeof(j.arg);
  li.ndim           = 1;
  li.grid_dim[0]    = j.num_blocks;
  li.grid_dim[1]    = 1;
  li.grid_dim[2]    = 1;
  li.block_dim[0]   = L.w * (uint32_t)caps.num_threads;
  li.block_dim[1]   = 1;
  li.block_dim[2]   = 1;
  li.lmem_size      = req_lmem;
  li.cluster_dim[0] = L.k;
  li.cluster_dim[1] = 1;
  li.cluster_dim[2] = 1;
  li.cluster_policy = L.policy;
  vx_event_h launch_ev = nullptr;
  const vx_result_t r = vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev);
  if (expect_reject && r == VX_ERR_INVALID_VALUE) {
    std::printf("reject: %s: %s (expected VX_ERR_INVALID_VALUE)\n", L.name, vx_result_string(r));
    if (j.count_buf)
      vx_buffer_release(j.count_buf);
    vx_buffer_release(j.err_buf);
    vx_buffer_release(j.rec_buf);
    return 0;
  }
  if (r != VX_SUCCESS) {
    std::printf("*** %s: the launch returned %s%s\n", L.name, vx_result_string(r),
                expect_reject ? ", expected VX_ERR_INVALID_VALUE" : "");
    return 1;
  }
  // An accepted launch that cannot be co-resident deadlocks below.
  if (expect_reject) {
    std::printf("*** %s: the launch was accepted, expected VX_ERR_INVALID_VALUE (%s)\n",
                L.name, why);
    std::fflush(stdout);
  }
  std::vector<vx_event_h> evs = {download(q, j.recs, j.rec_buf, launch_ev),
                                 download(q, j.errs, j.err_buf, launch_ev)};
  if (counting)
    evs.push_back(download(q, j.counts, j.count_buf, launch_ev));
  for (vx_event_h ev : evs) {
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }
  vx_event_release(launch_ev);

  const int errors = check(caps, L, j) + (expect_reject ? 1 : 0);
  if (j.count_buf)
    vx_buffer_release(j.count_buf);
  vx_buffer_release(j.err_buf);
  vx_buffer_release(j.rec_buf);
  return errors;
}

}  // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  Caps caps;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,         &caps.num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CLUSTERS,      &caps.num_clusters));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,         &caps.num_warps));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &caps.num_threads));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE,    &caps.lmem_size));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &caps.max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &caps.isa));
  std::printf("cluster_sync: %u cores in %u clusters, %u warps, DLMEM %s (largest cluster %u),"
              " atomics %s\n", (uint32_t)caps.num_cores, (uint32_t)caps.num_clusters,
              (uint32_t)caps.num_warps, caps.dlmem() ? "on" : "off",
              (uint32_t)caps.max_cluster, caps.atomics() ? "on" : "off");
  if (!caps.dlmem()) {
    std::printf("*** the device has no DLMEM\nFAILED!\n");
    return 1;
  }
  if (!caps.atomics() && !parity_mode)
    std::printf("no A extension: the arrival counters are left out\n");

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, "sync_k", &kernel));

  std::vector<Launch> launches;
  if (req_k != 0 || req_policy >= 0) {
    const auto policy = (vx_cluster_policy_e)((req_policy < 0) ? VX_CLUSTER_POLICY_SPREAD : req_policy);
    uint32_t k = req_k;
    if (k == 0) {
      k = (policy == VX_CLUSTER_POLICY_SPREAD)
        ? (uint32_t)caps.max_cluster
        : std::max<uint32_t>(1, std::min<uint32_t>(4, (uint32_t)caps.num_warps / req_w));
    }
    launches.push_back({k, req_g, req_w, policy, "launch"});
  } else {
    // A same-core cluster first, then K = 1 and the largest K, so the barrier
    // state is reset by the slot admissions of each new launch.
    const uint32_t w = std::min<uint32_t>(2, (uint32_t)caps.num_warps);
    const uint32_t k = std::max<uint32_t>(1, std::min<uint32_t>(4, (uint32_t)caps.num_warps / w));
    launches.push_back({k, req_g, w, VX_CLUSTER_POLICY_DEFAULT, "default"});
    launches.push_back({1, req_g, req_w, VX_CLUSTER_POLICY_SPREAD, "spread-1"});
    launches.push_back({(uint32_t)caps.max_cluster, req_g, req_w, VX_CLUSTER_POLICY_SPREAD,
                        "spread-max"});
  }
  int errors = 0;
  for (const Launch& L : launches)
    errors += run_launch(dev, q, kernel, caps, L);

  vx_kernel_release(kernel);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::printf("Found %d errors!\nFAILED!\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
