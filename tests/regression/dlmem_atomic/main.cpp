// dlmem_atomic: atomics on the local memory of blocks on other cores, through
// the distributed local-memory window, on a DLMEM device with the A extension
// (RV64). Every launch is SPREAD, so the ranks of a cluster sit on different
// cores whenever the hardware cluster has room.
//
//   default  clusters of K = 2 and K = 4 (each up to VX_CAPS_DLMEM_MAX_CLUSTER)
//   -k K     clusters of K x 1 x 1 instead
//   -n G     clusters per launch (default: one round of the machine, plus one)
//   -w W     warps per block (default 1)
//   -l B     local-memory bytes per block, a multiple of 64 (default 1024)
//   -i I     operations per thread, lock rounds per warp (default 4)
//   -t mask  tests to run, each as its own launch, bit t-1 for test t (default all):
//            1 amoadd.w/.d histogram into rank 0, and one amoadd ticket of each
//              width per thread from rank K-1: every value exactly once
//            2 an amoswap spinlock in rank K-1 around a plain window
//              read-modify-write of a counter in rank 0
//            3 amo{add,xor,and,or,min,max,minu,maxu}.{w,d} into every slice
//            4 compare-and-swap counters (only with VX_CFG_EXT_ZACAS_ENABLE)
//   -r       the lock's release is amoswap.w.rl alone, without the vx_fence()
//            before it: the release must wait for the counter's window store
//   -p       parity mode: one launch of fixed window-atomic loops into the next
//            rank (K = 2, or -k), no spin and no data-dependent trip count
//   -f       with -p, a fence after each iteration's atomics, which must wait
//            for their remote answers
//   -x lrsc  a run that must stop the simulator: rank 0 takes a reservation
//            (lr.w) on a word of rank 1, on another core
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

constexpr uint32_t kParityIters = 32;
constexpr uint32_t TEST_LRSC = 6;

const char* kernel_file = "kernel.vxbin";
uint32_t req_k     = 0;
uint32_t req_g     = 0;
uint32_t req_w     = 1;
uint32_t req_lmem  = 1024;
uint32_t req_iters = 4;
uint32_t req_tests = (1u << NUM_TESTS) - 1;
bool     parity    = false;
bool     pfence    = false;
bool     neg_lrsc  = false;
bool     verbose   = false;
bool     control   = false;
bool     rl_only   = false;

struct Caps {
  uint64_t num_cores = 0, num_clusters = 0, num_warps = 0, num_threads = 0;
  uint64_t lmem_size = 0, max_cluster = 0, isa = 0;
  bool atomics() const { return (isa & VX_ISA_STD_A) != 0; }
};

struct Launch {
  uint32_t test, k, g, w, lmem;
  bool requested;
};

void usage() {
  std::printf("Usage: [-k K] [-n clusters] [-w warps per block] [-l bytes] [-i iters] [-t mask]"
              " [-p [-f]] [-r] [-x lrsc] [-c] [-v] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:n:w:l:i:t:pfrx:cvh")) != -1) {
    switch (c) {
      case 'k': req_k     = (uint32_t)std::atoi(optarg); break;
      case 'n': req_g     = (uint32_t)std::atoi(optarg); break;
      case 'w': req_w     = (uint32_t)std::atoi(optarg); break;
      case 'l': req_lmem  = (uint32_t)std::atoi(optarg); break;
      case 'i': req_iters = (uint32_t)std::atoi(optarg); break;
      case 't': req_tests = (uint32_t)std::strtoul(optarg, nullptr, 0); break;
      case 'p': parity    = true;                        break;
      case 'f': pfence    = true;                        break;
      case 'r': rl_only   = true;                        break;
      case 'x':
        if (std::strcmp(optarg, "lrsc") != 0) {
          usage();
          std::exit(-1);
        }
        neg_lrsc = true;
        break;
      case 'c': control   = true;                        break;
      case 'v': verbose   = true;                        break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || req_w == 0 || req_iters == 0 || req_iters > 255 || (parity && neg_lrsc)
   || (pfence && !parity) || (rl_only && (parity || neg_lrsc))) {
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

const char* kernel_name(uint32_t test) {
  switch (test) {
  case TEST_HIST: return "hist_k";
  case TEST_LOCK: return rl_only ? "lock_rl_k" : "lock_k";
  case TEST_OPS:  return "ops_k";
  case TEST_CAS:  return "cas_k";
  case TEST_LRSC: return "lrsc_k";
  default:        return pfence ? "parity_fence_k" : "parity_k";
  }
}

// The local-memory bytes a kernel's layout needs.
uint32_t lmem_needed(uint32_t test, uint32_t nt) {
  switch (test) {
  case TEST_HIST:   return HIST_LMEM;
  case TEST_LOCK:   return LOCK_LMEM;
  case TEST_OPS:    return OPS_LMEM;
  case TEST_CAS:    return CAS_LMEM;
  case TEST_PARITY: return 16 * nt;
  default:          return 4;
  }
}

// Why the device cannot run L, or empty.
std::string infeasible(const Caps& caps, const Launch& L) {
  char buf[200];
  const uint32_t nt = block_threads(caps, L);
  if (!caps.atomics())
    return "the device has no A extension";
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
  if (L.lmem < lmem_needed(L.test, nt)) {
    std::snprintf(buf, sizeof(buf), "%s needs %u bytes of local memory per block",
                  kernel_name(L.test), lmem_needed(L.test, nt));
    return buf;
  }
  // A same-core cluster syncs on one core, so all its blocks must fit there.
  if (control && slots_per_core(caps, L) < L.k)
    return "a core cannot hold the whole same-core cluster";
  return "";
}

// Global words per thread and per block.
uint32_t thread_words(uint32_t test) {
  return (test == TEST_HIST) ? HIST_TWORDS : (test == TEST_PARITY) ? PARITY_TWORDS : 0;
}
uint32_t block_words(uint32_t test, uint32_t nt) {
  switch (test) {
  case TEST_HIST:   return HIST_BWORDS;
  case TEST_LOCK:   return LOCK_BWORDS;
  case TEST_OPS:    return OPS_BWORDS;
  case TEST_CAS:    return CAS_BWORDS;
  case TEST_PARITY: return PARITY_BWORDS(nt);
  default:          return 0;
  }
}

uint64_t dword(const uint32_t* p) {
  return ((uint64_t)p[1] << 32) | p[0];
}

// The mismatches of one launch.
int run_launch(vx_device_h dev, vx_queue_h q, vx_module_h mod, const Caps& caps,
               const Launch& L, uint32_t seed) {
  const char* kname = kernel_name(L.test);
  const uint32_t iters = (L.test == TEST_PARITY) ? kParityIters : req_iters;
  const uint32_t num_blocks = L.g * L.k;
  const uint32_t nt = block_threads(caps, L);
  const uint32_t threads = num_blocks * nt;
  std::printf("dlmem_atomic: %s, %s K=%u, %u clusters, %u warps and %u bytes of local memory"
              " per block, %u iterations, seed %#x\n", kname, control ? "DEFAULT" : "SPREAD",
              L.k, L.g, L.w, L.lmem, iters, seed);

  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, kname, &kernel));

  const size_t err_words  = (size_t)threads * ERR_WORDS;
  const size_t rec_words  = (size_t)num_blocks * REC_WORDS;
  const size_t tw = thread_words(L.test), bw = block_words(L.test, nt);
  const size_t glob_words = std::max<size_t>(1, threads * tw + num_blocks * bw);
  std::vector<uint32_t> mem(err_words + rec_words + glob_words, 0);
  uint32_t* errs = mem.data();
  uint32_t* recs = errs + err_words;
  uint32_t* glob = recs + rec_words;
  for (uint32_t t = 0; t < threads; ++t)
    errs[(size_t)t * ERR_WORDS + ERR_COUNT] = ~0u;
  std::fill(recs, glob, ~0u);
  for (uint32_t b = 0; b < num_blocks; ++b)
    recs[(size_t)b * REC_WORDS + REC_RAN] = 0;
  std::fill(glob, glob + glob_words, 0xDEADBEEFu);
  const uint64_t bytes = mem.size() * sizeof(uint32_t);
  vx_buffer_h buf = nullptr;
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &buf));
  CHECK(vx_enqueue_write(q, buf, 0, mem.data(), bytes, 0, nullptr, nullptr));

  atomic_arg_t arg = {};
  uint64_t base = 0;
  CHECK(vx_buffer_address(buf, &base));
  arg.err_addr    = base;
  arg.rec_addr    = base + err_words * sizeof(uint32_t);
  arg.glob_addr   = arg.rec_addr + rec_words * sizeof(uint32_t);
  arg.k           = L.k;
  arg.lmem_bytes  = L.lmem;
  arg.seed        = seed;
  arg.iters       = iters;
  arg.num_threads = (uint32_t)caps.num_threads;

  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kernel;
  li.args_host      = &arg;
  li.args_size      = sizeof(arg);
  li.ndim           = 1;
  li.grid_dim[0]    = num_blocks;
  li.grid_dim[1]    = 1;
  li.grid_dim[2]    = 1;
  li.block_dim[0]   = nt;
  li.block_dim[1]   = 1;
  li.block_dim[2]   = 1;
  li.lmem_size      = L.lmem;
  li.cluster_dim[0] = L.k;
  li.cluster_dim[1] = 1;
  li.cluster_dim[2] = 1;
  li.cluster_policy = control ? VX_CLUSTER_POLICY_DEFAULT : VX_CLUSTER_POLICY_SPREAD;

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, mem.data(), buf, 0, bytes, 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  vx_buffer_release(buf);
  vx_kernel_release(kernel);

  if (L.test == TEST_LRSC) {
    std::printf("Error: the remote LR did not stop the simulation\n");
    return 1;
  }

  int errors = 0;
  auto rec_of = [&](uint32_t b) { return &recs[(size_t)b * REC_WORDS]; };
  auto report = [&](uint32_t b, const char* what, uint64_t got, uint64_t want) {
    if (errors < 16) {
      const uint32_t* rec = rec_of(b);
      std::printf("*** block %u (cluster %u, core %u, slot %u): %s = %#llx, expected %#llx\n",
                  b, b / L.k, rec[REC_CORE], rec[REC_SLOT], what,
                  (unsigned long long)got, (unsigned long long)want);
    }
    ++errors;
  };

  for (uint32_t b = 0; b < num_blocks; ++b) {
    const uint32_t* rec = rec_of(b);
    if (rec[REC_RAN] != 1) {
      report(b, "ran", rec[REC_RAN], 1);
      continue;
    }
    if (rec[REC_RANK] != b % L.k)
      report(b, "vx_cluster_block_rank()", rec[REC_RANK], b % L.k);
    if (rec[REC_SIZE] != L.k)
      report(b, "vx_cluster_num_blocks()", rec[REC_SIZE], L.k);
    for (uint32_t t = 0; t < nt; ++t) {
      const uint32_t* e = &errs[((size_t)b * nt + t) * ERR_WORDS];
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
                    e[ERR_TEST], e[ERR_INDEX], (unsigned long long)dword(e + ERR_GOT),
                    (unsigned long long)dword(e + ERR_WANT));
      errors += (int)e[ERR_COUNT];
    }
  }

  const uint32_t* per_block = glob + threads * tw;
  auto out_of = [&](uint32_t b) { return per_block + (size_t)b * bw; };
  const uint32_t total = L.k * nt;
  for (uint32_t c = 0; c < L.g; ++c) {
    const uint32_t b0 = c * L.k, bl = b0 + L.k - 1;
    switch (L.test) {
    case TEST_HIST: {
      uint32_t bins_w[HIST_BINS] = {};
      uint64_t bins_d[HIST_BINS] = {};
      for (uint32_t r = 0; r < L.k; ++r) {
        for (uint32_t t = 0; t < nt; ++t) {
          for (uint32_t j = 0; j < iters; ++j) {
            const uint32_t h = hist_hash(seed, c, r, t, j);
            bins_w[hist_bin(h)] += hist_w(h);
            bins_d[hist_bin(h)] += hist_d(h);
          }
        }
      }
      const uint32_t* out = out_of(b0);
      for (uint32_t i = 0; i < HIST_BINS; ++i) {
        if (out[i] != bins_w[i])
          report(b0, "word bin", ((uint64_t)i << 32) | out[i], bins_w[i]);
        if (dword(out + HIST_BINS + 2 * i) != bins_d[i])
          report(b0, "doubleword bin", dword(out + HIST_BINS + 2 * i), bins_d[i]);
      }
      // Each ticket value drawn once, the word ones sign-extended as returned.
      std::vector<uint64_t> tws, tds;
      for (uint32_t b = b0; b <= bl; ++b) {
        for (uint32_t t = 0; t < nt; ++t) {
          const uint32_t* m = glob + ((size_t)b * nt + t) * HIST_TWORDS;
          tws.push_back(dword(m));
          tds.push_back(dword(m + 2));
        }
      }
      // In draw order, which crosses the sign bit of the word.
      const uint32_t w0 = ticket_w0(total);
      std::sort(tws.begin(), tws.end(), [w0](uint64_t a, uint64_t b) {
        return (uint32_t)a - w0 < (uint32_t)b - w0;
      });
      std::sort(tds.begin(), tds.end());
      for (uint32_t i = 0; i < total; ++i) {
        const uint64_t ww = (uint64_t)(int64_t)(int32_t)(ticket_w0(total) + i);
        if (tws[i] != ww)
          report(b0, "word ticket (sorted)", tws[i], ww);
        if (tds[i] != ticket_d0(total) + i)
          report(b0, "doubleword ticket (sorted)", tds[i], ticket_d0(total) + i);
      }
      const uint32_t* outl = out_of(bl);
      if (outl[3 * HIST_BINS] != ticket_w0(total) + total)
        report(bl, "final word ticket", outl[3 * HIST_BINS], ticket_w0(total) + total);
      if (dword(outl + 3 * HIST_BINS + 2) != ticket_d0(total) + total)
        report(bl, "final doubleword ticket", dword(outl + 3 * HIST_BINS + 2), ticket_d0(total) + total);
      break;
    }
    case TEST_LOCK: {
      const uint64_t want = (uint64_t)L.k * L.w * iters;
      if (dword(out_of(b0)) != want)
        report(b0, "counter under the lock", dword(out_of(b0)), want);
      break;
    }
    case TEST_OPS: {
      for (uint32_t qr = 0; qr < L.k; ++qr) {
        const uint32_t* out = out_of(b0 + qr);
        for (uint32_t o = 0; o < 8; ++o) {
          uint64_t vw = (uint32_t)ops_init(seed, c, qr, o);
          uint64_t vd = ops_init(seed, c, qr, 8 + o);
          for (uint32_t r = 0; r < L.k; ++r) {
            for (uint32_t t = 0; t < nt; ++t) {
              vw = ops_apply(o, 0, vw, ops_operand(seed, c, r, t, qr, o, 0));
              vd = ops_apply(o, 1, vd, ops_operand(seed, c, r, t, qr, o, 1));
            }
          }
          if (out[o] != (uint32_t)vw)
            report(b0 + qr, "word after operation", ((uint64_t)o << 32) | out[o], vw);
          if (dword(out + 8 + 2 * o) != vd)
            report(b0 + qr, "doubleword after operation", dword(out + 8 + 2 * o), vd);
        }
      }
      break;
    }
    case TEST_CAS: {
      const uint32_t ww = 0x7FFFFFF0u + total * iters;
      const uint64_t wd = 0xFFFFFFF0ull + (uint64_t)total * iters;
      if (out_of(b0)[0] != ww)
        report(b0, "compare-and-swap word counter", out_of(b0)[0], ww);
      if (dword(out_of(bl) + 2) != wd)
        report(bl, "compare-and-swap doubleword counter", dword(out_of(bl) + 2), wd);
      break;
    }
    default: {
      for (uint32_t r = 0; r < L.k; ++r) {
        const uint32_t b = b0 + r, prev = (r + L.k - 1) % L.k;
        for (uint32_t t = 0; t < nt; ++t) {
          uint64_t sum = 0, cell_d = 0;
          uint32_t cell_w = 0;
          for (uint32_t j = 0; j < iters; ++j) {
            sum += (uint64_t)(int64_t)(int32_t)cell_w;
            cell_w += j + 1;
            sum ^= cell_d;
            cell_d = std::max(cell_d, parity_d(seed, c, r, t, j));
          }
          const uint64_t got = dword(glob + ((size_t)b * nt + t) * PARITY_TWORDS);
          if (got != sum)
            report(b, "parity checksum of thread", got, sum);
          // The block's cells hold the previous rank's totals.
          uint64_t pd = 0;
          for (uint32_t j = 0; j < iters; ++j)
            pd = std::max(pd, parity_d(seed, c, prev, t, j));
          const uint32_t* out = out_of(b);
          if (out[t] != cell_w)
            report(b, "parity word cell", out[t], cell_w);
          if (dword(out + nt + 2 * t) != pd)
            report(b, "parity doubleword cell", dword(out + nt + 2 * t), pd);
        }
      }
      break;
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
  std::printf("dlmem_atomic: %u of %u clusters span several cores\n", spanning, L.g);
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
  std::printf("dlmem_atomic: %u cores, %u warps of %u threads, %u bytes of local memory,"
              " largest cluster %u, atomics %s, compare-and-swap %s\n", (uint32_t)caps.num_cores,
              (uint32_t)caps.num_warps, (uint32_t)caps.num_threads, (uint32_t)caps.lmem_size,
              (uint32_t)caps.max_cluster, caps.atomics() ? "on" : "off",
              VX_CFG_EXT_ZACAS_ENABLED ? "on" : "off");

  int errors = 0;
  if (!(caps.isa & VX_ISA_EXT_DLMEM)) {
    std::printf("*** the device has no DLMEM\n");
    ++errors;
  }

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));

  std::vector<Launch> launches;
  if (neg_lrsc) {
    launches.push_back({TEST_LRSC, 2, 1, req_w, req_lmem, true});
  } else if (parity) {
    launches.push_back({TEST_PARITY, req_k ? req_k : 2, req_g, req_w, req_lmem, true});
  } else {
    const bool requested = (req_k != 0);
    uint32_t tests = req_tests;
    if (!VX_CFG_EXT_ZACAS_ENABLED && (tests & (1u << (TEST_CAS - 1)))) {
      std::printf("dlmem_atomic: no Zacas, test %u skipped\n", TEST_CAS);
      tests &= ~(1u << (TEST_CAS - 1));
    }
    for (uint32_t k : requested ? std::vector<uint32_t>{req_k} : std::vector<uint32_t>{2, 4}) {
      for (uint32_t t = 1; t <= NUM_TESTS; ++t) {
        if (tests & (1u << (t - 1)))
          launches.push_back({t, k, req_g, req_w, req_lmem, requested});
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
