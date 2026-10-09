// dlmem_local: the distributed local memory window between blocks of a cluster
// that share a core (DEFAULT clusters, and SPREAD clusters of one block), on a
// DLMEM device (RV64).
//
//   default  three launches in one process: SPREAD K = 1 with 1 KB of local
//            memory per block, DEFAULT K = min(2, MAX) with 64 B and DEFAULT
//            K = MAX with SRAM / K, where MAX is VX_CAPS_DLMEM_MAX_CLUSTER
//   -k K     clusters of K x 1 x 1 instead (K <= MAX: the window names ranks
//            up to MAX only), with 64 B, 1 KB and SRAM / K per block
//   -s / -d  SPREAD / DEFAULT (the default) policy for the -k launch
//   -n G     clusters per launch (default: one round of the machine, plus one)
//   -w W     warps per block (default 1)
//   -l B     local-memory bytes per block, a multiple of 64; repeat for a sweep
//   -t mask  tests to run, each as its own launch of the shape, bit t-1 for
//            test t (default all; test 3 needs the A extension):
//            1 own write, sync, read every peer through the window
//            2 stores into every peer through the window
//            3 amoadd into rank 0 through the window
//            4 global, own and window lanes in one store and one load
//            5 sub-word stores at the slice and line edges of every peer
//            6 the self window aliases the plain pointer
//   -c       sync with cluster_group::sync() instead of a local barrier
//   -e       exit stress: many clusters, each block ending with window stores
//            to its own rank, read back by the slot's next block
//   -p       parity mode: one launch of fixed window loops on the next rank's
//            slice (K = 1, or -k), no spin and no data-dependent trip count
//   -b       burst mode: one-thread blocks (K = 1), each keeping eight one-lane
//            global loads in flight, more than the LSU holds at once
//   -x mode  a run that must stop the simulator: oob (rank 2 of a 2-block
//            cluster, needs MAX >= 3) or remote (SPREAD K = 2, rank 0 storing to
//            rank 1 on another core)
//   -v       print every cluster's placement (core:slot by rank)
//
// Until the hardware cluster barrier is used (-c), a DEFAULT cluster syncs on a
// local barrier of its base slot: its blocks sit in consecutive slots of one core.
// A default launch the device cannot hold prints SKIPPED; a requested one fails.

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
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

constexpr uint32_t kGlobFill   = 0xDEADBEEFu;
constexpr uint32_t kParityLmem = 256;
constexpr uint32_t kExitLmem   = 64;
constexpr uint32_t kExitG      = 256;

const char* kernel_file = "kernel.vxbin";
uint32_t req_k      = 0;
uint32_t req_g      = 0;
uint32_t req_w      = 1;
int      req_policy = -1;
uint32_t req_tests  = (1u << NUM_TESTS) - 1;
bool     use_cbar   = false;
bool     exit_mode  = false;
bool     parity     = false;
bool     burst      = false;
uint32_t neg_mode   = 0;
bool     verbose    = false;
std::vector<uint32_t> req_lmem;

struct Caps {
  uint64_t num_cores = 0, num_clusters = 0, num_warps = 0, num_threads = 0;
  uint64_t lmem_size = 0, max_cluster = 0, isa = 0;
  bool atomics() const { return (isa & VX_ISA_STD_A) != 0; }
};

enum class Kind { Local, Exit, Parity, Burst, Neg };

struct Launch {
  Kind kind;
  uint32_t policy, k, g, w, lmem;
  bool requested;
};

void usage() {
  std::printf("Usage: [-k K] [-s | -d] [-n clusters] [-w warps per block] [-l bytes]..."
              " [-t mask] [-c] [-e] [-p] [-b] [-x oob|remote] [-v] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:sdn:w:l:t:cepbx:vh")) != -1) {
    switch (c) {
      case 'k': req_k      = (uint32_t)std::atoi(optarg); break;
      case 's': req_policy = VX_CLUSTER_POLICY_SPREAD;    break;
      case 'd': req_policy = VX_CLUSTER_POLICY_DEFAULT;   break;
      case 'n': req_g      = (uint32_t)std::atoi(optarg); break;
      case 'w': req_w      = (uint32_t)std::atoi(optarg); break;
      case 'l': req_lmem.push_back((uint32_t)std::atoi(optarg)); break;
      case 't': req_tests  = (uint32_t)std::strtoul(optarg, nullptr, 0); break;
      case 'c': use_cbar   = true;                        break;
      case 'e': exit_mode  = true;                        break;
      case 'p': parity     = true;                        break;
      case 'b': burst      = true;                        break;
      case 'x':
        if (!std::strcmp(optarg, "oob")) {
          neg_mode = NEG_OOB;
        } else if (!std::strcmp(optarg, "remote")) {
          neg_mode = NEG_REMOTE;
        } else {
          usage();
          std::exit(-1);
        }
        break;
      case 'v': verbose    = true;                        break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || req_w == 0 || (exit_mode + parity + burst + (neg_mode != 0)) > 1) {
    usage();
    std::exit(-1);
  }
}

const char* policy_name(uint32_t p) {
  return (p == VX_CLUSTER_POLICY_SPREAD) ? "SPREAD" : "DEFAULT";
}

uint64_t footprint(uint32_t lmem) {
  return ((uint64_t)lmem + VX_CFG_MEM_BLOCK_SIZE - 1) / VX_CFG_MEM_BLOCK_SIZE * VX_CFG_MEM_BLOCK_SIZE;
}

uint32_t block_threads(const Caps& caps, const Launch& L) {
  return (L.kind == Kind::Burst) ? 1 : L.w * (uint32_t)caps.num_threads;
}

// Blocks a core holds at once, by warps and by local memory.
uint32_t slots_per_core(const Caps& caps, const Launch& L) {
  const uint32_t by_warps = (uint32_t)caps.num_warps / L.w;
  const uint32_t by_lmem  = (uint32_t)std::min<uint64_t>(caps.num_warps, caps.lmem_size / footprint(L.lmem));
  return std::min(by_warps, by_lmem);
}

// Why the device cannot run L, or empty.
std::string infeasible(const Caps& caps, const Launch& L) {
  char buf[200];
  const uint32_t nt = block_threads(caps, L);
  if (L.k == 0 || L.k > caps.max_cluster) {
    std::snprintf(buf, sizeof(buf), "K=%u is outside [1, MAX=%u]", L.k, (uint32_t)caps.max_cluster);
    return buf;
  }
  if (L.lmem < 64 || L.lmem % 64 != 0 || L.lmem > caps.lmem_size) {
    std::snprintf(buf, sizeof(buf), "%u bytes of local memory: needs a multiple of 64 up to %u",
                  L.lmem, (uint32_t)caps.lmem_size);
    return buf;
  }
  if (L.w > caps.num_warps) {
    std::snprintf(buf, sizeof(buf), "%u warps per block, the core has %u", L.w, (uint32_t)caps.num_warps);
    return buf;
  }
  if (L.kind == Kind::Parity && L.lmem < 2 * L.k * nt * sizeof(uint32_t))
    return "the parity loops need two words per thread of the cluster";
  if (L.kind == Kind::Neg) {
    if (neg_mode == NEG_OOB && caps.max_cluster < 3)
      return "rank 2 lies past the window unless MAX >= 3";
    return "";
  }
  if (L.policy == VX_CLUSTER_POLICY_SPREAD && L.k > 1)
    return "SPREAD with K > 1 places peers on other cores (remote access)";
  // A same-core cluster syncs on one core, so all its blocks must fit there.
  if (L.policy == VX_CLUSTER_POLICY_DEFAULT && slots_per_core(caps, L) < L.k) {
    std::snprintf(buf, sizeof(buf), "a core holds %u such blocks, the cluster needs %u",
                  slots_per_core(caps, L), L.k);
    return buf;
  }
  return "";
}

// One device buffer per launch, the records, errors and global words back to
// back, so a launch costs one write and one read.
struct Run {
  std::vector<uint32_t> mem;
  uint32_t* errs = nullptr;
  uint32_t* recs = nullptr;
  uint32_t* glob = nullptr;
  uint32_t num_blocks = 0, nt = 0;
};

// The mismatches of one launch.
int run_launch(vx_device_h dev, vx_queue_h q, vx_module_h mod, const Caps& caps,
               const Launch& L, uint32_t test, uint32_t seed) {
  char kname[16];
  if (L.kind == Kind::Local)
    std::snprintf(kname, sizeof(kname), "local%u_k", test);
  else
    std::snprintf(kname, sizeof(kname), "%s", (L.kind == Kind::Exit) ? "exit_k"
                  : (L.kind == Kind::Parity) ? "parity_k"
                  : (L.kind == Kind::Burst) ? "burst_k" : "neg_k");
  Run r;
  r.num_blocks = L.g * L.k;
  r.nt = block_threads(caps, L);
  const uint32_t tests = (L.kind == Kind::Neg) ? neg_mode : (1u << (test - 1));
  std::printf("dlmem_local: %s, %s K=%u, %u clusters, %u warps and %u bytes of local memory"
              " per block, seed %#x%s\n", kname, policy_name(L.policy), L.k, L.g,
              L.w, L.lmem, seed, use_cbar ? ", cluster_group::sync()" : "");

  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, kname, &kernel));

  const uint32_t threads = r.num_blocks * r.nt;
  const size_t err_words  = (size_t)threads * ERR_WORDS;
  const size_t rec_words  = (size_t)r.num_blocks * REC_WORDS;
  const size_t glob_words = (size_t)threads * ((L.kind == Kind::Parity) ? 2
                          : (L.kind == Kind::Burst) ? BURST_WORDS : 1);
  r.mem.assign(err_words + rec_words + glob_words, 0);
  r.errs = r.mem.data();
  r.recs = r.errs + err_words;
  r.glob = r.recs + rec_words;
  for (uint32_t t = 0; t < threads; ++t)
    r.errs[(size_t)t * ERR_WORDS + ERR_COUNT] = ~0u;
  std::fill(r.recs, r.glob, ~0u);
  for (uint32_t b = 0; b < r.num_blocks; ++b)
    r.recs[(size_t)b * REC_WORDS + REC_RAN] = 0;
  std::fill(r.glob, r.glob + glob_words, kGlobFill);
  if (L.kind == Kind::Burst) {
    for (size_t i = 0; i < glob_words; ++i)
      r.glob[i] = pattern(seed, TEST_BURST, 0, 0, (uint32_t)i);
  }
  const uint64_t bytes = r.mem.size() * sizeof(uint32_t);
  vx_buffer_h buf = nullptr;
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &buf));
  CHECK(vx_enqueue_write(q, buf, 0, r.mem.data(), bytes, 0, nullptr, nullptr));

  local_arg_t arg = {};
  uint64_t base = 0;
  CHECK(vx_buffer_address(buf, &base));
  arg.err_addr   = base;
  arg.rec_addr   = base + err_words * sizeof(uint32_t);
  arg.glob_addr  = arg.rec_addr + rec_words * sizeof(uint32_t);
  arg.k          = L.k;
  arg.lmem_bytes = L.lmem;
  arg.seed       = seed;
  arg.tests      = tests;
  arg.flags      = (caps.atomics() ? FLAG_ATOMICS : 0) | (use_cbar ? FLAG_CBAR : 0);

  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kernel;
  li.args_host      = &arg;
  li.args_size      = sizeof(arg);
  li.ndim           = 1;
  li.grid_dim[0]    = r.num_blocks;
  li.grid_dim[1]    = 1;
  li.grid_dim[2]    = 1;
  li.block_dim[0]   = r.nt;
  li.block_dim[1]   = 1;
  li.block_dim[2]   = 1;
  li.lmem_size      = L.lmem;
  li.cluster_dim[0] = L.k;
  li.cluster_dim[1] = 1;
  li.cluster_dim[2] = 1;
  li.cluster_policy = (vx_cluster_policy_e)L.policy;

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, r.mem.data(), buf, 0, bytes, 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  vx_buffer_release(buf);
  vx_kernel_release(kernel);

  int errors = 0;
  auto rec_of = [&](uint32_t b) { return &r.recs[(size_t)b * REC_WORDS]; };
  auto report = [&](uint32_t b, const char* what, uint64_t got, uint64_t want) {
    if (errors < 16) {
      const uint32_t* rec = rec_of(b);
      std::printf("*** block %u (cluster %u, core %u, slot %u): %s = %#llx, expected %#llx\n",
                  b, b / L.k, rec[REC_CORE], rec[REC_SLOT], what,
                  (unsigned long long)got, (unsigned long long)want);
    }
    ++errors;
  };

  if (L.kind == Kind::Neg) {
    std::printf("Error: the %s window store did not stop the simulation\n",
                neg_mode == NEG_OOB ? "out-of-bounds" : "remote");
    return 1;
  }

  for (uint32_t b = 0; b < r.num_blocks; ++b) {
    const uint32_t* rec = rec_of(b);
    if (rec[REC_RAN] != 1) {
      report(b, "ran", rec[REC_RAN], 1);
      continue;
    }
    if (rec[REC_RANK] != b % L.k)
      report(b, "vx_cluster_block_rank()", rec[REC_RANK], b % L.k);
    if (rec[REC_SIZE] != L.k)
      report(b, "vx_cluster_num_blocks()", rec[REC_SIZE], L.k);
    const uint32_t* first = rec_of(b - b % L.k);
    if (L.policy == VX_CLUSTER_POLICY_DEFAULT && first[REC_RAN] == 1) {
      if (rec[REC_CORE] != first[REC_CORE])
        report(b, "core (a DEFAULT cluster shares one)", rec[REC_CORE], first[REC_CORE]);
      if (rec[REC_SLOT] != first[REC_SLOT] + b % L.k)
        report(b, "slot (base + rank)", rec[REC_SLOT], first[REC_SLOT] + b % L.k);
    }
    for (uint32_t t = 0; t < r.nt; ++t) {
      const uint32_t* e = &r.errs[((size_t)b * r.nt + t) * ERR_WORDS];
      if (e[ERR_COUNT] == 0)
        continue;
      if (e[ERR_COUNT] == ~0u) {
        report(b, "thread never finished", t, 0);
        continue;
      }
      if (errors < 16)
        std::printf("*** block %u (cluster %u rank %u, core %u, slot %u) thread %u: %u mismatches,"
                    " the first in test %u at %#x (got %#x)\n", b, b / L.k, rec[REC_RANK],
                    rec[REC_CORE], rec[REC_SLOT], t, e[ERR_COUNT], e[ERR_TEST], e[ERR_INDEX], e[ERR_GOT]);
      errors += (int)e[ERR_COUNT];
    }
  }

  const uint32_t n = L.lmem / sizeof(uint32_t);
  if (L.kind == Kind::Local && (tests & (1u << (TEST_MIXED - 1)))) {
    // The global lanes' words, and no other lane wrote global memory.
    for (uint32_t b = 0; b < r.num_blocks; ++b) {
      const uint32_t rank = b % L.k;
      for (uint32_t t = 0; t < r.nt; ++t) {
        const uint32_t got = r.glob[(size_t)b * r.nt + t];
        const uint32_t want = (mixed_kind(rank, t, r.nt, n) == 0)
                            ? pattern(seed, TEST_MIXED, b / L.k, rank, t) : kGlobFill;
        if (got != want)
          report(b, "test 4 global word of thread", got, want);
      }
    }
  }

  if (L.kind == Kind::Parity) {
    for (uint32_t b = 0; b < r.num_blocks; ++b) {
      for (uint32_t t = 0; t < r.nt; ++t) {
        const uint32_t* g = &r.glob[2 * ((size_t)b * r.nt + t)];
        if (g[0] != PARITY_ITERS)
          report(b, "parity window load/store count", g[0], PARITY_ITERS);
        const uint32_t want = caps.atomics() ? PARITY_ITERS : 0;
        if (g[1] != want)
          report(b, "parity window amoadd count", g[1], want);
      }
    }
  }

  if (L.kind == Kind::Exit) {
    // Every block's slot was left either by no block (the slot's first one) or,
    // whole, by one earlier block of the same core and slot, each found once.
    const uint32_t words = std::min(n, r.nt);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> firsts;
    std::set<uint32_t> found;
    for (uint32_t b = 0; b < r.num_blocks; ++b) {
      const uint32_t* rec = rec_of(b);
      if (rec[REC_RAN] != 1)
        continue;
      const uint32_t* f = &r.glob[(size_t)b * r.nt];
      const uint32_t prev = (f[0] >> 8) & 0xFFFFFu;
      bool whole = (f[0] >> 28) == EXIT_MAGIC && prev < r.num_blocks && prev != b;
      for (uint32_t t = 0; whole && t < words; ++t)
        whole = (f[t] == exit_word(prev, t));
      const auto where = std::make_pair(rec[REC_CORE], rec[REC_SLOT]);
      if (!whole) {
        if (++firsts[where] > 1)
          report(b, "slot left by no whole earlier block, word 0", f[0], exit_word(0, 0));
        continue;
      }
      const uint32_t* prec = rec_of(prev);
      if (prec[REC_CORE] != rec[REC_CORE] || prec[REC_SLOT] != rec[REC_SLOT])
        report(b, "slot left by a block of another core:slot", prev, b);
      if (!found.insert(prev).second)
        report(b, "a block's exit stores found by two successors", prev, b);
    }
    std::printf("dlmem_local: %zu core:slot pairs, %zu exits found by the next block\n",
                firsts.size(), found.size());
  }

  if (verbose && L.kind != Kind::Parity) {
    for (uint32_t c = 0; c < L.g; ++c) {
      std::printf("cluster %u:", c);
      for (uint32_t k = 0; k < L.k; ++k)
        std::printf(" %u:%u", rec_of(c * L.k + k)[REC_CORE], rec_of(c * L.k + k)[REC_SLOT]);
      std::printf("\n");
    }
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
  std::printf("dlmem_local: %u cores, %u warps of %u threads, %u bytes of local memory,"
              " largest cluster %u, atomics %s\n", (uint32_t)caps.num_cores,
              (uint32_t)caps.num_warps, (uint32_t)caps.num_threads, (uint32_t)caps.lmem_size,
              (uint32_t)caps.max_cluster, caps.atomics() ? "on" : "off");

  int errors = 0;
  if (!(caps.isa & VX_ISA_EXT_DLMEM)) {
    std::printf("*** the device has no DLMEM\n");
    ++errors;
  }

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));

  const uint32_t D = VX_CLUSTER_POLICY_DEFAULT, S = VX_CLUSTER_POLICY_SPREAD;
  const uint32_t max_k = (uint32_t)caps.max_cluster;
  std::vector<Launch> launches;
  auto lmems = [&](uint32_t dflt) {
    return req_lmem.empty() ? std::vector<uint32_t>{dflt} : req_lmem;
  };
  auto sram_per = [&](uint32_t k) { return (uint32_t)(caps.lmem_size / k) & ~63u; };
  auto add = [&](Kind kind, uint32_t policy, uint32_t k, uint32_t dflt_lmem, bool requested) {
    for (uint32_t lmem : lmems(dflt_lmem)) {
      Launch L = {kind, policy, k, req_g, req_w, lmem, requested};
      for (const Launch& o : launches) {
        if (o.policy == L.policy && o.k == L.k && o.lmem == L.lmem)
          L.k = 0;
      }
      if (L.k != 0)
        launches.push_back(L);
    }
  };
  if (neg_mode != 0) {
    if (neg_mode == NEG_OOB)
      add(Kind::Neg, D, 2, 64, true);
    else
      add(Kind::Neg, S, 2, 64, true);
  } else if (parity) {
    add(Kind::Parity, D, req_k ? req_k : 1, kParityLmem, true);
  } else if (burst) {
    add(Kind::Burst, D, 1, 64, true);
  } else if (exit_mode) {
    add(Kind::Exit, (req_policy < 0) ? D : (uint32_t)req_policy,
        req_k ? req_k : std::min<uint32_t>(2, max_k), kExitLmem, true);
  } else if (req_k != 0 || req_policy >= 0) {
    const uint32_t k = req_k ? req_k : 1;
    if (req_lmem.empty()) {
      for (uint32_t lmem : {64u, 1024u, sram_per(k)})
        add(Kind::Local, (req_policy < 0) ? D : (uint32_t)req_policy, k, lmem, true);
    } else {
      add(Kind::Local, (req_policy < 0) ? D : (uint32_t)req_policy, k, 0, true);
    }
  } else {
    add(Kind::Local, S, 1, 1024, false);
    add(Kind::Local, D, std::min<uint32_t>(2, max_k), 64, false);
    add(Kind::Local, D, max_k, sram_per(max_k), false);
  }

  const uint32_t tests = req_tests & (caps.atomics() ? ~0u : ~(1u << (TEST_AMO - 1)));
  if (tests != req_tests)
    std::printf("dlmem_local: no A extension, test %u skipped\n", TEST_AMO);
  for (size_t i = 0; i < launches.size(); ++i) {
    Launch& L = launches[i];
    const std::string why = infeasible(caps, L);
    if (!why.empty()) {
      std::printf("%s: %s K=%u, %u warps and %u bytes per block: %s\n",
                  L.requested ? "Error" : "SKIPPED", policy_name(L.policy), L.k, L.w,
                  L.lmem, why.c_str());
      errors += L.requested ? 1 : 0;
      continue;
    }
    if (L.g == 0) {
      if (L.kind == Kind::Exit) {
        L.g = kExitG;
      } else if (L.kind == Kind::Neg) {
        L.g = 1;
      } else {
        const uint32_t per_core = std::max<uint32_t>(1, slots_per_core(caps, L) / L.k);
        L.g = (uint32_t)caps.num_cores * per_core + 1;
      }
    }
    for (uint32_t t = 1; t <= NUM_TESTS; ++t) {
      if (L.kind == Kind::Local && !(tests & (1u << (t - 1))))
        continue;
      errors += run_launch(dev, q, mod, caps, L, t, hash3(0x5EEDu, (uint32_t)i, t));
      if (L.kind != Kind::Local)
        break;
    }
  }

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
