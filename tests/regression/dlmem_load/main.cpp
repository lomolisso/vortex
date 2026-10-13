// dlmem_load: loads from the local memory of blocks on other cores, through
// the distributed local-memory window, on a DLMEM device (RV64). Every launch
// is SPREAD, so the ranks of a cluster sit on different cores whenever the
// hardware cluster has room.
//
//   default  clusters of K = 2 and K = 4 (each up to VX_CAPS_DLMEM_MAX_CLUSTER),
//            each with 1 KB and SRAM / K of local memory per block
//   -k K     clusters of K x 1 x 1 instead
//   -n G     clusters per launch (default: one round of the machine, plus one)
//   -w W     warps per block (default 1)
//   -l B     local-memory bytes per block, a multiple of 64; repeat for a sweep
//   -t mask  tests to run, each as its own launch, bit t-1 for test t (default all):
//            1 each block fills its own slice, cluster.sync(), then reads
//              every slice as doublewords, words, halves and bytes
//            2 ping-pong: ranks 2i and 2i + 1 poll each other's slice
//            3 one word load and one doubleword load whose lanes are global,
//              own local memory and the window on another core
//            4 a thread's window store then its load of the same bytes
//   -r R     ping-pong and store-load rounds (default 8)
//   -p       parity mode: one launch of fixed window-load loops from the next
//            rank (K = 2, or -k), no spin and no data-dependent trip count
//   -c       control run: DEFAULT (same-core) clusters through the same
//            window, so nothing crosses the network
//   -v       print every cluster's placement (core:slot by rank)
//
// A launch needs K >= 2 and a device whose largest cluster holds it: a default
// launch the device cannot hold prints SKIPPED, a requested one fails. A launch
// in which no cluster spans two cores fails too: nothing would cross the network.

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
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

constexpr uint32_t kParityLmem   = 256;
constexpr uint32_t kParityRounds = 64;

const char* kernel_file = "kernel.vxbin";
uint32_t req_k      = 0;
uint32_t req_g      = 0;
uint32_t req_w      = 1;
uint32_t req_tests  = (1u << NUM_TESTS) - 1;
uint32_t req_rounds = 8;
bool     parity     = false;
bool     verbose    = false;
bool     control    = false;
std::vector<uint32_t> req_lmem;

struct Caps {
  uint64_t num_cores = 0, num_clusters = 0, num_warps = 0, num_threads = 0;
  uint64_t lmem_size = 0, max_cluster = 0, isa = 0;
};

struct Launch {
  uint32_t test, k, g, w, lmem;
  bool requested;
};

void usage() {
  std::printf("Usage: [-k K] [-n clusters] [-w warps per block] [-l bytes]... [-t mask]"
              " [-r rounds] [-p] [-c] [-v] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:n:w:l:t:r:pcvh")) != -1) {
    switch (c) {
      case 'k': req_k      = (uint32_t)std::atoi(optarg); break;
      case 'n': req_g      = (uint32_t)std::atoi(optarg); break;
      case 'w': req_w      = (uint32_t)std::atoi(optarg); break;
      case 'l': req_lmem.push_back((uint32_t)std::atoi(optarg)); break;
      case 't': req_tests  = (uint32_t)std::strtoul(optarg, nullptr, 0); break;
      case 'r': req_rounds = (uint32_t)std::atoi(optarg); break;
      case 'p': parity     = true;                        break;
      case 'c': control    = true;                        break;
      case 'v': verbose    = true;                        break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || req_w == 0 || req_rounds == 0 || req_rounds > 255) {
    usage();
    std::exit(-1);
  }
}

uint64_t footprint(uint32_t lmem) {
  return ((uint64_t)lmem + VX_CFG_MEM_BLOCK_SIZE - 1) / VX_CFG_MEM_BLOCK_SIZE * VX_CFG_MEM_BLOCK_SIZE;
}

uint32_t block_threads(const Caps& caps, const Launch& L) {
  return L.w * (uint32_t)caps.num_threads;
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
  if (L.k < 2 || L.k > caps.max_cluster) {
    std::snprintf(buf, sizeof(buf), "K=%u is outside [2, MAX=%u]", L.k, (uint32_t)caps.max_cluster);
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
  const uint32_t n = L.lmem / sizeof(uint32_t);
  if (L.test == TEST_PARITY && (n < nt || n % nt != 0))
    return "the parity loops need a slice of a multiple of the block's threads in words";
  // A same-core cluster syncs on one core, so all its blocks must fit there.
  if (control && slots_per_core(caps, L) < L.k)
    return "a core cannot hold the whole same-core cluster";
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

const char* kernel_name(uint32_t test) {
  return (test == TEST_ALL) ? "all_k" : (test == TEST_PP) ? "pingpong_k"
       : (test == TEST_MIXED) ? "mixed_k" : (test == TEST_RAW) ? "raw_k" : "parity_k";
}

// The mismatches of one launch.
int run_launch(vx_device_h dev, vx_queue_h q, vx_module_h mod, const Caps& caps,
               const Launch& L, uint32_t seed) {
  const char* kname = kernel_name(L.test);
  const uint32_t rounds = (L.test == TEST_PARITY) ? kParityRounds : req_rounds;
  Run r;
  r.num_blocks = L.g * L.k;
  r.nt = block_threads(caps, L);
  const uint32_t n = L.lmem / sizeof(uint32_t);
  std::printf("dlmem_load: %s, %s K=%u, %u clusters, %u warps and %u bytes of local memory"
              " per block, seed %#x\n", kname, control ? "DEFAULT" : "SPREAD", L.k, L.g, L.w,
              L.lmem, seed);

  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, kname, &kernel));

  const uint32_t threads = r.num_blocks * r.nt;
  const size_t err_words  = (size_t)threads * ERR_WORDS;
  const size_t rec_words  = (size_t)r.num_blocks * REC_WORDS;
  const size_t glob_words = (L.test == TEST_MIXED) ? 2 * (size_t)threads
                          : (L.test == TEST_PARITY) ? (size_t)threads : 1;
  r.mem.assign(err_words + rec_words + glob_words, 0);
  r.errs = r.mem.data();
  r.recs = r.errs + err_words;
  r.glob = r.recs + rec_words;
  for (uint32_t t = 0; t < threads; ++t)
    r.errs[(size_t)t * ERR_WORDS + ERR_COUNT] = ~0u;
  std::fill(r.recs, r.glob, ~0u);
  for (uint32_t b = 0; b < r.num_blocks; ++b)
    r.recs[(size_t)b * REC_WORDS + REC_RAN] = 0;
  for (size_t i = 0; i < glob_words; ++i)
    r.glob[i] = glob_word(seed, (uint32_t)i);
  const uint64_t bytes = r.mem.size() * sizeof(uint32_t);
  vx_buffer_h buf = nullptr;
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &buf));
  CHECK(vx_enqueue_write(q, buf, 0, r.mem.data(), bytes, 0, nullptr, nullptr));

  load_arg_t arg = {};
  uint64_t base = 0;
  CHECK(vx_buffer_address(buf, &base));
  arg.err_addr   = base;
  arg.rec_addr   = base + err_words * sizeof(uint32_t);
  arg.glob_addr  = arg.rec_addr + rec_words * sizeof(uint32_t);
  arg.k          = L.k;
  arg.lmem_bytes = L.lmem;
  arg.seed       = seed;
  arg.rounds     = rounds;

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
  li.cluster_policy = control ? VX_CLUSTER_POLICY_DEFAULT : VX_CLUSTER_POLICY_SPREAD;

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
                    " the first in test %u at %#x: got %#llx, expected %#llx\n",
                    b, b / L.k, rec[REC_RANK], rec[REC_CORE], rec[REC_SLOT], t, e[ERR_COUNT],
                    e[ERR_TEST], e[ERR_INDEX],
                    (unsigned long long)(((uint64_t)e[ERR_GOT + 1] << 32) | e[ERR_GOT]),
                    (unsigned long long)(((uint64_t)e[ERR_WANT + 1] << 32) | e[ERR_WANT]));
      errors += (int)e[ERR_COUNT];
    }
  }

  if (L.test == TEST_PARITY) {
    for (uint32_t b = 0; b < r.num_blocks; ++b) {
      const uint32_t c = b / L.k, next = (b % L.k + 1) % L.k;
      for (uint32_t t = 0; t < r.nt; ++t) {
        uint32_t want = 0;
        for (uint32_t j = 0; j < rounds; ++j)
          want += parity_fill(seed, c, next, (j * r.nt + t) % n);
        const uint32_t got = r.glob[(size_t)b * r.nt + t];
        if (got != want)
          report(b, "parity load sum of thread", ((uint64_t)t << 32) | got, want);
      }
    }
  }

  // Whether any cluster's ranks sit on two cores, so the run crossed the network.
  uint32_t spanning = 0;
  for (uint32_t c = 0; c < L.g; ++c) {
    std::set<uint32_t> cores;
    for (uint32_t k = 0; k < L.k; ++k)
      cores.insert(rec_of(c * L.k + k)[REC_CORE]);
    spanning += (cores.size() > 1);
    if (verbose) {
      std::printf("cluster %u:", c);
      for (uint32_t k = 0; k < L.k; ++k)
        std::printf(" %u:%u", rec_of(c * L.k + k)[REC_CORE], rec_of(c * L.k + k)[REC_SLOT]);
      std::printf("\n");
    }
  }
  std::printf("dlmem_load: %u of %u clusters span several cores\n", spanning, L.g);
  if (errors == 0 && spanning == 0 && !control) {
    std::printf("*** no cluster spans two cores: nothing crossed the network\n");
    ++errors;
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
  std::printf("dlmem_load: %u cores, %u warps of %u threads, %u bytes of local memory,"
              " largest cluster %u\n", (uint32_t)caps.num_cores, (uint32_t)caps.num_warps,
              (uint32_t)caps.num_threads, (uint32_t)caps.lmem_size, (uint32_t)caps.max_cluster);

  int errors = 0;
  if (!(caps.isa & VX_ISA_EXT_DLMEM)) {
    std::printf("*** the device has no DLMEM\n");
    ++errors;
  }

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));

  std::vector<Launch> launches;
  auto sram_per = [&](uint32_t k) { return (uint32_t)(caps.lmem_size / k) & ~63u; };
  if (parity) {
    for (uint32_t lmem : req_lmem.empty() ? std::vector<uint32_t>{kParityLmem} : req_lmem)
      launches.push_back({TEST_PARITY, req_k ? req_k : 2, req_g, req_w, lmem, true});
  } else {
    const bool requested = (req_k != 0);
    for (uint32_t k : requested ? std::vector<uint32_t>{req_k} : std::vector<uint32_t>{2, 4}) {
      std::vector<uint32_t> lmems = req_lmem.empty()
          ? std::vector<uint32_t>{1024, sram_per(k)} : req_lmem;
      lmems.erase(std::unique(lmems.begin(), lmems.end()), lmems.end());
      for (uint32_t lmem : lmems) {
        for (uint32_t t = 1; t <= NUM_TESTS; ++t) {
          if (req_tests & (1u << (t - 1)))
            launches.push_back({t, k, req_g, req_w, lmem, requested});
        }
      }
    }
  }

  for (size_t i = 0; i < launches.size(); ++i) {
    Launch& L = launches[i];
    const std::string why = infeasible(caps, L);
    if (!why.empty()) {
      std::printf("%s: %s, K=%u, %u warps and %u bytes per block: %s\n",
                  L.requested ? "Error" : "SKIPPED", kernel_name(L.test), L.k, L.w,
                  L.lmem, why.c_str());
      errors += L.requested ? 1 : 0;
      continue;
    }
    if (L.g == 0) {
      const uint32_t slots = (uint32_t)caps.num_cores * slots_per_core(caps, L);
      L.g = std::max<uint32_t>(1, slots / L.k) + 1;
    }
    errors += run_launch(dev, q, mod, caps, L, hash3(0x5EEDu, (uint32_t)i, L.test));
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
