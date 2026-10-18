// cta_gang: thread-block clusters spread over the cores of one hardware cluster
// (VX_CLUSTER_POLICY_SPREAD, a DLMEM device, RV64).
//
//   default  three launches in one process: SPREAD clusters of the device's
//            largest K, SPREAD with K = 1, and DEFAULT (same-core) clusters of
//            min(2, NUM_WARPS); on a device without DLMEM, SPREAD must be
//            refused and only the DEFAULT launch runs
//   -k K     one launch in clusters of K x 1 x 1 instead
//   -w W     warps per block (default 1)
//   -l B     local-memory bytes per block (default 0)
//   -n G     clusters per launch; the grid is G * K blocks (default 6)
//   -s / -d  SPREAD (default) / DEFAULT policy for the -k launch
//   -p       parity mode: fixed ALU work and no rendezvous, so the instruction
//            count does not depend on timing
//   -r       rejection mode: descriptors the runtime must refuse, an old-size
//            descriptor it must run as DEFAULT, a SPREAD batch launch, and
//            DEFAULT clusters that exactly fill a core's warps or slots
//   -v       print every cluster's placement (core:slot by rank), to diff models
//
// Every block records its core, slot, rank and local-memory base; the host checks
// that each block ran once, that a cluster stays inside one hardware cluster
// (on one core under DEFAULT), that ranks follow the walk, that the first
// SPREAD cluster of a launch, placed on an idle machine, lands rank r on core r,
// and that the base is the slot times the aligned footprint. With the A
// extension each cluster also meets at an atomic counter, which proves its
// blocks were resident together.

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#include <algorithm>
#include <cstddef>
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

constexpr uint32_t kRendezvousBound = 100000;
constexpr uint32_t kParityWork      = 64;

const char* kernel_file = "kernel.vxbin";
uint32_t    req_k       = 0;
uint32_t    req_w       = 1;
uint32_t    req_lmem    = 0;
uint32_t    req_g       = 6;
int         req_policy  = -1;   // -1: not given
bool        parity_mode = false;
bool        reject_mode = false;
bool        verbose     = false;

struct Caps {
  uint64_t num_cores = 0, num_clusters = 0, num_warps = 0, num_threads = 0;
  uint64_t lmem_size = 0, max_cluster = 0, isa = 0;
  bool dlmem() const { return (isa & VX_ISA_EXT_DLMEM) != 0; }
  bool atomics() const { return (isa & VX_ISA_STD_A) != 0; }
  uint32_t ncc() const { return (uint32_t)(num_cores / num_clusters); }
};

enum class Submit { Launch, Commands, OldSize };

struct Launch {
  uint32_t k, g, w, lmem;
  vx_cluster_policy_e policy;
  Submit how;
  const char* name;
};

void usage() {
  std::printf("Usage: [-k cluster size] [-w warps per block] [-l local-memory bytes]"
              " [-n clusters] [-s: SPREAD | -d: DEFAULT] [-p: parity] [-r: rejections] [-v: placement] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:w:l:n:sdprvh")) != -1) {
    switch (c) {
      case 'k': req_k       = (uint32_t)std::atoi(optarg); break;
      case 'w': req_w       = (uint32_t)std::atoi(optarg); break;
      case 'l': req_lmem    = (uint32_t)std::atoi(optarg); break;
      case 'n': req_g       = (uint32_t)std::atoi(optarg); break;
      case 's': req_policy  = VX_CLUSTER_POLICY_SPREAD;    break;
      case 'd': req_policy  = VX_CLUSTER_POLICY_DEFAULT;   break;
      case 'p': parity_mode = true;                        break;
      case 'r': reject_mode = true;                        break;
      case 'v': verbose     = true;                        break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || req_w == 0 || req_g == 0 || req_lmem % sizeof(uint32_t) != 0) {
    usage();
    std::exit(-1);
  }
}

const char* policy_name(uint32_t p) {
  return (p == VX_CLUSTER_POLICY_SPREAD) ? "SPREAD" : (p == VX_CLUSTER_POLICY_DEFAULT) ? "DEFAULT" : "?";
}

// The launch's buffers and its descriptor, built once so a rejected and an
// accepted submission see the same memory.
struct Job {
  vx_buffer_h rec_buf = nullptr, arrive_buf = nullptr;
  std::vector<uint32_t> recs;
  gang_arg_t arg = {};
  vx_launch_info_t li = {};
  uint32_t num_blocks = 0;
};

void job_create(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps,
                const Launch& L, bool rendezvous, Job& j) {
  j.num_blocks = L.g * L.k;
  const uint64_t rec_bytes = (uint64_t)j.num_blocks * REC_WORDS * sizeof(uint32_t);
  CHECK(vx_buffer_create(dev, rec_bytes, VX_MEM_READ_WRITE, &j.rec_buf));
  // A block that never ran leaves the fill, which no field it writes can equal;
  // the run counter starts at 0.
  j.recs.assign((size_t)j.num_blocks * REC_WORDS, 0xffffffffu);
  for (uint32_t b = 0; b < j.num_blocks; ++b)
    j.recs[(size_t)b * REC_WORDS + REC_RUNS] = 0;
  CHECK(vx_enqueue_write(q, j.rec_buf, 0, j.recs.data(), rec_bytes, 0, nullptr, nullptr));
  CHECK(vx_buffer_address(j.rec_buf, &j.arg.rec_addr));
  if (rendezvous) {
    const uint64_t arrive_bytes = (uint64_t)L.g * sizeof(uint32_t);
    std::vector<uint32_t> zeros(L.g, 0);
    CHECK(vx_buffer_create(dev, arrive_bytes, VX_MEM_READ_WRITE, &j.arrive_buf));
    CHECK(vx_enqueue_write(q, j.arrive_buf, 0, zeros.data(), arrive_bytes, 0, nullptr, nullptr));
    CHECK(vx_buffer_address(j.arrive_buf, &j.arg.arrive_addr));
  }
  j.arg.k          = L.k;
  j.arg.lmem_bytes = L.lmem;
  j.arg.bound      = kRendezvousBound;
  j.arg.work       = parity_mode ? kParityWork : 0;

  vx_launch_info_t& li = j.li;
  li = {};
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
  li.lmem_size      = L.lmem;
  li.cluster_dim[0] = L.k;
  li.cluster_dim[1] = 1;
  li.cluster_dim[2] = 1;
  li.cluster_policy = L.policy;
  if (L.how == Submit::OldSize) {
    // A caller built before cluster_policy passes its sizeof (88 on LP64), whose tail
    // padding is now reserved0; the bytes there and past it are whatever its stack
    // held, here a SPREAD request the runtime must not see.
    constexpr size_t v1_size = (offsetof(vx_launch_info_t, reserved0) + alignof(vx_launch_info_t) - 1)
                             / alignof(vx_launch_info_t) * alignof(vx_launch_info_t);
    li.struct_size    = v1_size;
    li.reserved0      = 0xdeadbeefu;
    li.cluster_policy = VX_CLUSTER_POLICY_SPREAD;
  }
}

void job_release(Job& j) {
  if (j.arrive_buf)
    vx_buffer_release(j.arrive_buf);
  vx_buffer_release(j.rec_buf);
  j = Job();
}

vx_result_t submit(vx_queue_h q, const vx_launch_info_t& li, Submit how, vx_event_h* ev) {
  if (how == Submit::Commands) {
    vx_command_t c = {};
    c.type = VX_COMMAND_LAUNCH;
    c.data.launch = &li;
    return vx_enqueue_commands(q, &c, 1, 0, nullptr, ev);
  }
  return vx_enqueue_launch(q, &li, 0, nullptr, ev);
}

// Runs a submitted launch to completion and reads the records back.
void job_wait(vx_queue_h q, Job& j, vx_event_h launch_ev) {
  const uint64_t rec_bytes = (uint64_t)j.num_blocks * REC_WORDS * sizeof(uint32_t);
  vx_event_h rec_ev = nullptr;
  CHECK(vx_enqueue_read(q, j.recs.data(), j.rec_buf, 0, rec_bytes, 1, &launch_ev, &rec_ev));
  CHECK(vx_event_wait_value(rec_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(rec_ev);
  vx_event_release(launch_ev);
}

// The checks on one finished launch run with `policy`, the policy the runtime
// must have applied.
int check_records(const Caps& caps, const Launch& L, uint32_t policy, const Job& j,
                  bool rendezvous) {
  const uint32_t k = L.k;
  const uint32_t ncc = caps.ncc();
  const uint64_t stride = ((uint64_t)L.lmem + VX_CFG_MEM_BLOCK_SIZE - 1)
                        / VX_CFG_MEM_BLOCK_SIZE * VX_CFG_MEM_BLOCK_SIZE;
  int errors = 0;
  auto rec_of = [&](uint32_t b) { return &j.recs[(size_t)b * REC_WORDS]; };
  auto report = [&](uint32_t b, const char* what, uint64_t got, uint64_t want) {
    if (errors < 16) {
      const uint32_t* rec = rec_of(b);
      std::printf("*** %s: block %u (core %u, slot %u): %s = %#llx, expected %#llx\n",
                  L.name, b, rec[REC_CORE], rec[REC_CTA_ID], what,
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
    if (rec[REC_CORE] >= caps.num_cores)
      report(b, "core id", rec[REC_CORE], caps.num_cores - 1);
    const uint64_t lmem = ((uint64_t)rec[REC_LMEM_HI] << 32) | rec[REC_LMEM_LO];
    const uint64_t want_lmem = (uint64_t)VX_MEM_LMEM_BASE_ADDR + rec[REC_CTA_ID] * stride;
    if (lmem != want_lmem)
      report(b, "local-memory base", lmem, want_lmem);
    if (lmem + L.lmem > (uint64_t)VX_MEM_LMEM_BASE_ADDR + caps.lmem_size)
      report(b, "local-memory end", lmem + L.lmem, VX_MEM_LMEM_BASE_ADDR + caps.lmem_size);
    if (rec[REC_LMEM_OK] != 1)
      report(b, "local-memory readback", rec[REC_LMEM_OK], 1);
    if (rendezvous && rec[REC_POLLS] == 0)
      report(b, "rendezvous (0 = the cluster never met)", 0, 1);
  }
  if (errors)
    return errors;

  uint32_t shared = 0;
  for (uint32_t g = 0; g < L.g; ++g) {
    const uint32_t* first = rec_of(g * k);
    std::set<uint32_t> cores;
    std::set<std::pair<uint32_t, uint32_t>> slots;
    for (uint32_t r = 0; r < k; ++r) {
      const uint32_t b = g * k + r;
      const uint32_t* rec = rec_of(b);
      cores.insert(rec[REC_CORE]);
      if (!slots.insert({rec[REC_CORE], rec[REC_CTA_ID]}).second)
        report(b, "slot already held by a block of its cluster", rec[REC_CTA_ID], ~0u);
      if (rec[REC_CORE] / ncc != first[REC_CORE] / ncc)
        report(b, "hardware cluster", rec[REC_CORE] / ncc, first[REC_CORE] / ncc);
      if (policy == VX_CLUSTER_POLICY_DEFAULT && rec[REC_CORE] != first[REC_CORE])
        report(b, "core (DEFAULT keeps a cluster on one core)", rec[REC_CORE], first[REC_CORE]);
      // The first cluster of a launch meets an empty mirror and a reset round-robin.
      if (policy == VX_CLUSTER_POLICY_SPREAD && g == 0 && rec[REC_CORE] % ncc != r)
        report(b, "core in its hardware cluster (first cluster, idle machine)",
               rec[REC_CORE] % ncc, r);
    }
    if (cores.size() < std::min<size_t>(k, ncc))
      ++shared;
    if (verbose) {
      std::printf("%s: cluster %u:", L.name, g);
      for (uint32_t r = 0; r < k; ++r)
        std::printf(" %u:%u", rec_of(g * k + r)[REC_CORE], rec_of(g * k + r)[REC_CTA_ID]);
      std::printf("\n");
    }
  }
  if (policy == VX_CLUSTER_POLICY_SPREAD && k > 1)
    std::printf("%s: %u of %u clusters shared a core\n", L.name, shared, L.g);
  return errors;
}

// One launch end to end: the mismatches, or -1 when it could not run.
int run_launch(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps,
               const Launch& L) {
  const uint32_t applied = (L.how == Submit::OldSize) ? VX_CLUSTER_POLICY_DEFAULT : L.policy;
  const bool rendezvous = caps.atomics() && !parity_mode;
  Job j;
  job_create(dev, q, kernel, caps, L, rendezvous, j);
  std::printf("%s: %s, K=%u, %u clusters, %u warps and %u local-memory bytes per block%s%s"
              " (struct_size %zu)\n",
              L.name, policy_name(applied), L.k, L.g, L.w, L.lmem,
              rendezvous ? ", with a rendezvous" : "",
              L.how == Submit::Commands ? ", as a batch" :
              L.how == Submit::OldSize ? ", from an old-size descriptor" : "",
              (size_t)j.li.struct_size);
  vx_event_h ev = nullptr;
  const vx_result_t r = submit(q, j.li, L.how, &ev);
  if (r != VX_SUCCESS) {
    std::printf("*** %s: the launch returned %s\n", L.name, vx_result_string(r));
    job_release(j);
    return -1;
  }
  job_wait(q, j, ev);
  const int errors = check_records(caps, L, applied, j, rendezvous);
  job_release(j);
  return errors;
}

// A descriptor the runtime must refuse with `want`, before anything runs.
int expect_reject(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps,
                  const Launch& L, vx_result_t want, const char* what,
                  void (*edit)(vx_launch_info_t&) = nullptr) {
  Job j;
  job_create(dev, q, kernel, caps, L, false, j);
  if (edit)
    edit(j.li);
  vx_event_h ev = nullptr;
  const vx_result_t r = submit(q, j.li, L.how, &ev);
  int errors = 0;
  if (r == VX_SUCCESS) {
    // A launch that cannot be co-resident may never finish.
    std::printf("*** %s%s: accepted, expected %s\n", what,
                L.how == Submit::Commands ? " (batch)" : "", vx_result_string(want));
    std::fflush(stdout);
    job_wait(q, j, ev);
    ++errors;
  } else if (r != want) {
    ++errors;
  }
  std::printf("%s %s%s: %s (expected %s)\n", errors ? "***" : "reject:", what,
              L.how == Submit::Commands ? " (batch)" : "", vx_result_string(r),
              vx_result_string(want));
  job_release(j);
  return errors;
}

int run_rejects(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps) {
  const uint32_t k2 = std::min<uint32_t>(2, (uint32_t)caps.num_warps);
  int errors = 0;
  for (Submit how : {Submit::Launch, Submit::Commands}) {
    Launch L = {k2, 1, 1, 0, VX_CLUSTER_POLICY_SPREAD, how, "reject"};
    if (!caps.dlmem()) {
      errors += expect_reject(dev, q, kernel, caps, L, VX_ERR_NOT_SUPPORTED,
                              "SPREAD on a device without DLMEM");
    } else {
      Launch big = L;
      big.k = (uint32_t)caps.max_cluster + 1;
      errors += expect_reject(dev, q, kernel, caps, big, VX_ERR_INVALID_VALUE,
                              "SPREAD with K above VX_CAPS_DLMEM_MAX_CLUSTER");
      // The KMU's rank counter is sized for the largest cluster of either policy.
      Launch over = L;
      over.policy = VX_CLUSTER_POLICY_DEFAULT;
      over.k = (uint32_t)std::max(caps.num_warps, caps.max_cluster) + 1;
      errors += expect_reject(dev, q, kernel, caps, over, VX_ERR_INVALID_VALUE,
                              "DEFAULT with K above max(NUM_WARPS, MAX_CLUSTER)");
      // DEFAULT puts a whole cluster on one core, so its K blocks must fit there at once.
      if (caps.num_warps >= 2) {
        Launch warps = L;
        warps.policy = VX_CLUSTER_POLICY_DEFAULT;
        warps.k = 2;
        warps.w = (uint32_t)caps.num_warps / 2 + 1;
        errors += expect_reject(dev, q, kernel, caps, warps, VX_ERR_INVALID_VALUE,
                                "DEFAULT with K * W above NUM_WARPS");
        Launch slots = L;
        slots.policy = VX_CLUSTER_POLICY_DEFAULT;
        slots.k = 2;
        slots.lmem = (uint32_t)caps.lmem_size / 2 + VX_CFG_MEM_BLOCK_SIZE;
        errors += expect_reject(dev, q, kernel, caps, slots, VX_ERR_INVALID_VALUE,
                                "DEFAULT with K above the local-memory slots");
      }
      // Pre-programmed geometry cannot be checked against the gang's placement bound.
      errors += expect_reject(dev, q, kernel, caps, L, VX_ERR_INVALID_VALUE,
                              "SPREAD with ndim 0",
                              [](vx_launch_info_t& li) { li.ndim = 0; });
      if (how == Submit::Commands) {
        // The batch path skips the block-size check, so the placement bound catches it.
        Launch wide = L;
        wide.w = (uint32_t)caps.num_warps + 1;
        errors += expect_reject(dev, q, kernel, caps, wide, VX_ERR_INVALID_VALUE,
                                "SPREAD with a block wider than a core");
      }
    }
    errors += expect_reject(dev, q, kernel, caps, L, VX_ERR_INVALID_VALUE, "policy 2",
                            [](vx_launch_info_t& li) {
                              li.cluster_policy = (vx_cluster_policy_e)2;
                            });
    errors += expect_reject(dev, q, kernel, caps, L, VX_ERR_INVALID_INFO, "struct_size 80",
                            [](vx_launch_info_t& li) { li.struct_size = 80; });
  }

  // A version-1 descriptor still runs, as DEFAULT whatever its stack tail holds.
  const Launch old = {k2, 3, 1, 0, VX_CLUSTER_POLICY_DEFAULT, Submit::OldSize, "old-size"};
  int e = run_launch(dev, q, kernel, caps, old);
  errors += (e < 0) ? 1 : e;
  if (caps.dlmem()) {
    // The batch carries the policy in its queue-memory descriptor.
    const uint32_t k = std::min<uint32_t>(2, (uint32_t)caps.max_cluster);
    const Launch batch = {k, 3, 1, 0, VX_CLUSTER_POLICY_SPREAD, Submit::Commands, "batch"};
    e = run_launch(dev, q, kernel, caps, batch);
    errors += (e < 0) ? 1 : e;
    // DEFAULT clusters that exactly fill a core's warps, or its local-memory slots.
    const uint32_t w = std::max<uint32_t>(1, (uint32_t)caps.num_warps / 2);
    const Launch full_warps = {(uint32_t)caps.num_warps / w, 2, w, 0, VX_CLUSTER_POLICY_DEFAULT,
                               Submit::Launch, "default-full-warps"};
    e = run_launch(dev, q, kernel, caps, full_warps);
    errors += (e < 0) ? 1 : e;
    const Launch full_slots = {k2, 2, 1, (uint32_t)caps.lmem_size / k2, VX_CLUSTER_POLICY_DEFAULT,
                               Submit::Commands, "default-full-slots"};
    e = run_launch(dev, q, kernel, caps, full_slots);
    errors += (e < 0) ? 1 : e;
  }
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
  std::printf("cta_gang: %u cores in %u clusters, %u warps, DLMEM %s (largest cluster %u),"
              " atomics %s\n", (uint32_t)caps.num_cores, (uint32_t)caps.num_clusters,
              (uint32_t)caps.num_warps, caps.dlmem() ? "on" : "off",
              (uint32_t)caps.max_cluster, caps.atomics() ? "on" : "off");

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, "gang_k", &kernel));

  int errors = 0;
  if (reject_mode) {
    errors += run_rejects(dev, q, kernel, caps);
  } else {
    std::vector<Launch> launches;
    if (req_k != 0) {
      const auto policy = (vx_cluster_policy_e)((req_policy < 0) ? VX_CLUSTER_POLICY_SPREAD : req_policy);
      launches.push_back({req_k, req_g, req_w, req_lmem, policy, Submit::Launch, "launch"});
    } else {
      // A cluster of the largest K, then the K = 1 and DEFAULT launches after it,
      // so the round-robin and epoch resets at each new launch are exercised.
      const uint32_t k2 = std::min<uint32_t>(2, (uint32_t)caps.num_warps);
      const uint32_t singles = caps.ncc() * (uint32_t)caps.num_warps + 3;
      if (caps.dlmem()) {
        launches.push_back({(uint32_t)caps.max_cluster, req_g, req_w, req_lmem,
                            VX_CLUSTER_POLICY_SPREAD, Submit::Launch, "spread-max"});
        launches.push_back({1, singles, req_w, req_lmem, VX_CLUSTER_POLICY_SPREAD,
                            Submit::Launch, "spread-1"});
      } else {
        const Launch L = {k2, 1, 1, 0, VX_CLUSTER_POLICY_SPREAD, Submit::Launch, "no-dlmem"};
        errors += expect_reject(dev, q, kernel, caps, L, VX_ERR_NOT_SUPPORTED,
                                "SPREAD on a device without DLMEM");
      }
      launches.push_back({k2, req_g, req_w, req_lmem, VX_CLUSTER_POLICY_DEFAULT,
                          Submit::Launch, "default"});
    }
    for (const Launch& L : launches) {
      const int e = run_launch(dev, q, kernel, caps, L);
      errors += (e < 0) ? 1 : e;
    }
  }

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
