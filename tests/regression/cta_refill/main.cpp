// cta_refill: when blocks start and end, from thread 0's first and last clock
// reads (the H100 method: %globaltimer read first and last), with the SM and
// slot each block ran on.
//
//   -H       one GPC never blocks another: n clusters of K = the SMs of a GPC
//            (up to the largest cluster), each block a whole SM; cluster 0
//            runs long, the others short. Every short cluster must start and
//            end while cluster 0 still runs, in the GPCs cluster 0 leaves free:
//            a cluster that waits for cluster 0's GPC must not hold back the
//            clusters behind it that another GPC can take now.
//   -R       refill gaps: waves of one whole-SM block per SM, as plain blocks,
//            clusters of 1 and clusters of 2. The gap from a block's last clock
//            read to the next block's first on its SM is the old block's exit,
//            the retirement, the dispatch (for a cluster also the done event's
//            trip to the admission and the placement) and the new block's start;
//            nothing holds a freed slot. With -t the gaps after an SM's first
//            block are at most that kind's tail. A launch starts with cold caches,
//            so the first block's exit path fetches its code from DRAM and its gap
//            carries a DRAM latency the refill does not. Each kind prints its gap
//            next to the H100's block-to-block refill (H100REF).
//   -M       first wave: 1.5 x M1 + 1 long clusters of one-warp blocks for K = 1,
//            2 and the largest K; the clusters that start before the first one
//            ends must number M1 = vx_occupancy_max_active_clusters = the sum
//            over GPCs of floor(SMs x c / K), c = min(VX_CFG_MAX_CTAS_PER_SM_CLUSTER,
//            NUM_WARPS).
//   -n N     clusters (-H, default 8) or waves (-R, default 3, at least 3)
//   -L I     ALU iterations of cluster 0 (-H, default 4000) or of every block (-M, default 4000)
//   -S I     ALU iterations of the other clusters (-H) or of every block (-R) (default 16)
//   -t P,C1,C2  -R: the tails of plain blocks, clusters of 1 and clusters of 2, in cycles (0: not checked)
//   -v       print every cluster's GPC, SMs and clock window

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <unistd.h>
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

const char* kernel_file = "kernel.vxbin";
char        mode        = 0;
bool        verbose     = false;
uint32_t    req_n       = 0;
uint32_t    long_work   = 4000;
uint32_t    short_work  = 16;
uint64_t    tails[3]    = {0, 0, 0};

struct Caps {
  uint64_t num_cores = 0, num_clusters = 0, num_warps = 0, num_threads = 0;
  uint64_t max_cluster = 0, isa = 0;
  bool dlmem() const { return (isa & VX_ISA_EXT_DLMEM) != 0; }
  uint32_t sms_per_gpc() const { return (uint32_t)(num_cores / num_clusters); }
};

void usage() {
  std::printf("Usage: -H|-R|-M [-n clusters or waves] [-L long iterations] [-S short iterations]"
              " [-t plain,cluster1,cluster2 tail cycles] [-v] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "HRMn:L:S:t:vh")) != -1) {
    switch (c) {
      case 'H': case 'R': case 'M': mode = (char)c;         break;
      case 't': {
        char* p = optarg;
        for (int i = 0; i < 3 && *p; ++i) {
          tails[i] = std::strtoull(p, &p, 0);
          if (*p == ',')
            ++p;
        }
        break;
      }
      case 'n': req_n      = (uint32_t)std::atoi(optarg);   break;
      case 'L': long_work  = (uint32_t)std::atoi(optarg);   break;
      case 'S': short_work = (uint32_t)std::atoi(optarg);   break;
      case 'v': verbose    = true;                           break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (req_n == 0)
    req_n = (mode == 'R') ? 3 : 8;
  if (mode == 0 || optind < argc || req_n < ((mode == 'R') ? 3u : 2u)) {
    usage();
    std::exit(-1);
  }
}

struct Rec {
  uint32_t block, smid, slot, rank, size, runs;
  uint64_t t0, t1;
};

struct Run {
  uint32_t k = 1, w = 1, n = 1;
  bool cluster = true;  // false: a plain launch, no cluster attribute
  cta_arg_t arg = {};
  std::vector<Rec> recs;
};

// One launch end to end; the records come back in `run.recs`.
void launch(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps,
            vx_cluster_policy_e policy, Run& run) {
  const uint32_t blocks = run.n * run.k;
  const uint64_t rec_bytes = (uint64_t)blocks * REC_WORDS * sizeof(uint32_t);
  std::vector<uint32_t> raw((size_t)blocks * REC_WORDS, 0xffffffffu);
  for (uint32_t b = 0; b < blocks; ++b)
    raw[(size_t)b * REC_WORDS + REC_RUNS] = 0;
  vx_buffer_h buf = nullptr;
  CHECK(vx_buffer_create(dev, rec_bytes, VX_MEM_READ_WRITE, &buf));
  CHECK(vx_enqueue_write(q, buf, 0, raw.data(), rec_bytes, 0, nullptr, nullptr));
  CHECK(vx_buffer_address(buf, &run.arg.rec_addr));
  run.arg.k = run.k;

  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kernel;
  li.args_host      = &run.arg;
  li.args_size      = sizeof(run.arg);
  li.ndim           = 1;
  li.grid_dim[0]    = blocks;
  li.grid_dim[1]    = 1;
  li.grid_dim[2]    = 1;
  li.block_dim[0]   = run.w * (uint32_t)caps.num_threads;
  li.block_dim[1]   = 1;
  li.block_dim[2]   = 1;
  li.cluster_dim[0] = run.cluster ? run.k : 0;
  li.cluster_dim[1] = run.cluster ? 1 : 0;
  li.cluster_dim[2] = run.cluster ? 1 : 0;
  li.cluster_policy = policy;
  li.cluster_flags  = (run.k > VX_CLUSTER_MAX_PORTABLE) ? VX_CLUSTER_FLAG_NON_PORTABLE : 0;
  vx_event_h ev = nullptr, rd = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
  CHECK(vx_enqueue_read(q, raw.data(), buf, 0, rec_bytes, 1, &ev, &rd));
  CHECK(vx_event_wait_value(rd, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(rd);
  vx_event_release(ev);
  CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
  vx_buffer_release(buf);

  run.recs.resize(blocks);
  for (uint32_t b = 0; b < blocks; ++b) {
    const uint32_t* r = &raw[(size_t)b * REC_WORDS];
    run.recs[b] = {r[REC_BLOCK], r[REC_SMID], r[REC_SLOT], r[REC_RANK], r[REC_SIZE], r[REC_RUNS],
                   ((uint64_t)r[REC_T0_HI] << 32) | r[REC_T0_LO],
                   ((uint64_t)r[REC_T1_HI] << 32) | r[REC_T1_LO]};
  }
}

// Every block ran once, with its rank, on an SM of the device; each cluster sits
// in one GPC on distinct SMs.
int check_placement(const Caps& caps, const Run& run) {
  const uint32_t per_gpc = caps.sms_per_gpc();
  int errors = 0;
  auto report = [&](uint32_t b, const char* what, uint64_t got, uint64_t want) {
    if (errors < 16)
      std::printf("*** block %u: %s = %llu, expected %llu\n", b, what,
                  (unsigned long long)got, (unsigned long long)want);
    ++errors;
  };
  for (uint32_t b = 0; b < run.recs.size(); ++b) {
    const Rec& r = run.recs[b];
    if (r.runs != 1)               report(b, "times run", r.runs, 1);
    else if (r.block != b)         report(b, "blockIdx.x", r.block, b);
    else if (r.rank != b % run.k)  report(b, "cluster rank", r.rank, b % run.k);
    else if (r.size != run.k)      report(b, "cluster size", r.size, run.k);
    else if (r.smid >= caps.num_cores) report(b, "%smid", r.smid, caps.num_cores - 1);
    else if (r.t1 < r.t0)          report(b, "last clock read before the first", r.t1, r.t0);
  }
  if (errors)
    return errors;
  for (uint32_t g = 0; g < run.n; ++g) {
    std::set<uint32_t> sms;
    const uint32_t gpc = run.recs[g * run.k].smid / per_gpc;
    for (uint32_t r = 0; r < run.k; ++r) {
      const Rec& rec = run.recs[g * run.k + r];
      sms.insert(rec.smid);
      if (rec.smid / per_gpc != gpc)
        report(rec.block, "GPC (a cluster stays in one GPC)", rec.smid / per_gpc, gpc);
    }
    if (sms.size() != run.k)
      report(g * run.k, "distinct SMs of the cluster", sms.size(), run.k);
  }
  return errors;
}

struct Window { uint32_t gpc; uint64_t start, end, first_end; };

Window window_of(const Caps& caps, const Run& run, uint32_t g) {
  Window w = {run.recs[g * run.k].smid / caps.sms_per_gpc(), ~0ull, 0, ~0ull};
  for (uint32_t r = 0; r < run.k; ++r) {
    const Rec& rec = run.recs[g * run.k + r];
    w.start     = std::min(w.start, rec.t0);
    w.end       = std::max(w.end, rec.t1);
    w.first_end = std::min(w.first_end, rec.t1);
  }
  return w;
}

int run_h4(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps) {
  const uint32_t k = std::min<uint32_t>(caps.sms_per_gpc(), (uint32_t)caps.max_cluster);
  if (!caps.dlmem() || caps.num_clusters < 2 || k < 2) {
    std::printf("*** -H needs a DLMEM device with 2 GPCs of at least 2 SMs\n");
    return 1;
  }
  Run run;
  run.k = k;
  run.w = (uint32_t)caps.num_warps;
  run.n = req_n;
  run.arg.work         = short_work;
  run.arg.long_work    = long_work;
  run.arg.long_cluster = 0;
  std::printf("h4: %u clusters of K=%u, %u warps per block (whole SMs), cluster 0 runs %u"
              " iterations, the others %u\n", run.n, k, run.w, long_work, short_work);
  launch(dev, q, kernel, caps, VX_CLUSTER_POLICY_SPREAD, run);
  int errors = check_placement(caps, run);
  if (errors)
    return errors;

  const Window w0 = window_of(caps, run, 0);
  for (uint32_t g = 0; g < run.n; ++g) {
    const Window w = window_of(caps, run, g);
    if (verbose) {
      std::printf("h4: cluster %u: GPC %u, SMs", g, w.gpc);
      for (uint32_t r = 0; r < k; ++r)
        std::printf(" %u", run.recs[g * k + r].smid);
      std::printf(", cycles %llu..%llu\n", (unsigned long long)w.start, (unsigned long long)w.end);
    }
    if (g == 0)
      continue;
    if (w.start >= w0.first_end || w.end >= w0.first_end) {
      if (errors < 16)
        std::printf("*** h4: cluster %u (GPC %u) runs %llu..%llu, not inside cluster 0's run"
                    " (GPC %u, ends at %llu)\n", g, w.gpc, (unsigned long long)w.start,
                    (unsigned long long)w.end, w0.gpc, (unsigned long long)w0.first_end);
      ++errors;
    }
  }
  std::printf("h4: cluster 0 ends at %llu; %s\n", (unsigned long long)w0.first_end,
              errors ? "a short cluster waited for it" : "every short cluster ran beside it");
  return errors;
}

// The gaps between consecutive blocks on each SM: thread 0's last clock read of
// one block to the first read of the next; warm_max leaves out the gap after the
// SM's first block.
struct Gaps { uint64_t min = ~0ull, max = 0, count = 0, warm_max = 0, warm_count = 0; };

Gaps gaps_of(const Caps& caps, const Run& run) {
  std::vector<std::vector<const Rec*>> per_sm(caps.num_cores);
  for (const Rec& r : run.recs)
    per_sm[r.smid].push_back(&r);
  Gaps g;
  for (auto& v : per_sm) {
    std::sort(v.begin(), v.end(), [](const Rec* a, const Rec* b) { return a->t0 < b->t0; });
    for (size_t i = 1; i < v.size(); ++i) {
      const uint64_t gap = (v[i]->t0 > v[i - 1]->t1) ? v[i]->t0 - v[i - 1]->t1 : 0;
      g.min = std::min(g.min, gap);
      g.max = std::max(g.max, gap);
      ++g.count;
      if (i >= 2) {
        g.warm_max = std::max(g.warm_max, gap);
        ++g.warm_count;
      }
    }
  }
  return g;
}

int run_refill(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps) {
  // H100 block-to-block refill per launch kind (03/spec.md §5), in ns
  struct Kind { const char* name; bool cluster; uint32_t k; uint64_t h100_ns, tail; };
  std::vector<Kind> kinds = {
    {"plain", false, 1, VX_CFG_H100_CTA_REFILL_NS, tails[0]},
    {"cluster-1", true, 1, VX_CFG_H100_CTA_REFILL_CLUSTER1_NS, tails[1]},
  };
  if (caps.dlmem() && caps.sms_per_gpc() >= 2 && caps.max_cluster >= 2)
    kinds.push_back({"cluster-2", true, 2, VX_CFG_H100_CTA_REFILL_CLUSTER_NS, tails[2]});
  int errors = 0;
  for (const Kind& kd : kinds) {
    Run run;
    run.cluster = kd.cluster;
    run.k = kd.k;
    run.w = (uint32_t)caps.num_warps;
    run.n = req_n * (uint32_t)caps.num_cores / kd.k;
    run.arg.work         = short_work;
    run.arg.long_cluster = ~0u;
    launch(dev, q, kernel, caps, VX_CLUSTER_POLICY_DEFAULT, run);
    const int e = check_placement(caps, run);
    errors += e;
    if (e)
      continue;
    const Gaps g = gaps_of(caps, run);
    bool ok = g.warm_count != 0 && (kd.tail == 0 || g.warm_max <= kd.tail);
    std::printf("%s refill: %s, %llu gaps of %llu..%llu cycles, %llu warm up to %llu, tail %llu\n",
                ok ? "" : "***", kd.name, (unsigned long long)g.count, (unsigned long long)g.min,
                (unsigned long long)g.max, (unsigned long long)g.warm_count, (unsigned long long)g.warm_max,
                (unsigned long long)kd.tail);
    std::printf("H100REF metric=cta_refill_%s_cycles h100=%.0f vortex=%llu\n", kd.name,
                kd.h100_ns * 1e-3 * VX_CFG_H100_SM_CLOCK_MHZ, (unsigned long long)g.warm_max);
    errors += ok ? 0 : 1;
  }
  return errors;
}

int run_m1(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps) {
  const uint32_t per_gpc = caps.sms_per_gpc();
  const uint32_t c = std::min<uint32_t>(VX_CFG_MAX_CTAS_PER_SM_CLUSTER, (uint32_t)caps.num_warps);
  std::vector<uint32_t> ks = {1};
  if (caps.dlmem() && caps.max_cluster >= 2) {
    ks.push_back(2);
    if (caps.max_cluster > 2)
      ks.push_back((uint32_t)caps.max_cluster);
  }
  int errors = 0;
  for (uint32_t k : ks) {
    Run run;
    run.k = k;
    run.w = 1;
    run.arg.long_work    = long_work;
    run.arg.long_cluster = 0;
    run.arg.work         = long_work;
    vx_launch_info_t li = {};
    li.struct_size    = sizeof(li);
    li.ndim           = 1;
    li.grid_dim[0]    = k;
    li.block_dim[0]   = (uint32_t)caps.num_threads;
    li.cluster_dim[0] = k;
    li.cluster_policy = VX_CLUSTER_POLICY_DEFAULT;
    li.cluster_flags  = (k > VX_CLUSTER_MAX_PORTABLE) ? VX_CLUSTER_FLAG_NON_PORTABLE : 0;
    uint32_t m1 = 0;
    CHECK(vx_occupancy_max_active_clusters(dev, &li, &m1));
    const uint32_t want = (uint32_t)caps.num_clusters * ((per_gpc >= k) ? per_gpc * c / k : 0);
    run.n = m1 + m1 / 2 + 1;
    launch(dev, q, kernel, caps, VX_CLUSTER_POLICY_DEFAULT, run);
    int e = check_placement(caps, run);
    if (e == 0) {
      uint64_t first_end = ~0ull;
      for (const Rec& r : run.recs)
        first_end = std::min(first_end, r.t1);
      uint32_t wave = 0;
      for (uint32_t g = 0; g < run.n; ++g)
        wave += (window_of(caps, run, g).start < first_end) ? 1 : 0;
      const bool ok = (m1 == want) && (wave == m1);
      std::printf("%s m1: K=%u, %u clusters, first wave %u, max_active_clusters %u, expected %u\n",
                  ok ? "" : "***", k, run.n, wave, m1, want);
      e = ok ? 0 : 1;
    }
    errors += e;
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
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &caps.max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &caps.isa));
  std::printf("cta_refill: %u SMs in %u GPCs, %u warps, largest cluster %u\n",
              (uint32_t)caps.num_cores, (uint32_t)caps.num_clusters,
              (uint32_t)caps.num_warps, (uint32_t)caps.max_cluster);

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, "cta_t", &kernel));

  const int errors = (mode == 'H') ? run_h4(dev, q, kernel, caps)
                   : (mode == 'R') ? run_refill(dev, q, kernel, caps)
                   : run_m1(dev, q, kernel, caps);

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
