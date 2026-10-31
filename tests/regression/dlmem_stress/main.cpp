// dlmem_stress: a seeded random mix of window loads, stores and atomics across
// the cores of a cluster, on a DLMEM device with the A extension (RV64). Every
// launch is SPREAD, so the ranks of a cluster sit on different cores whenever
// the hardware cluster has room.
//
// Each thread owns ST_CELLS doublewords spread over the slices of its cluster
// and keeps a model of them: every load and atomic it issues must return what
// the model says. Each step is one instruction for the whole warp (a store of
// 1-8 bytes, a doubleword or narrow load, amoadd.d, amoxor.w, amoswap.d,
// amomaxu.w, a store then a load, a fence, or an amoadd.w into a counter every
// thread shares). A phase ends with a signature store into the next rank and
// cluster.sync(), the fence-free arrive/wait, or a ring of flags with a
// bounded spin; each thread then checks the signature the previous rank left
// it. The host checks every cell and counter at the end against its own model.
//
//   default  clusters of K = 2 and K = 4 (each up to VX_CAPS_DLMEM_MAX_CLUSTER)
//   -k K     clusters of K x 1 x 1 instead
//   -n G     clusters per launch (default: one round of the machine, plus one)
//   -w W     warps per block (default 1)
//   -l B     local-memory bytes per block (default: what the layout needs)
//   -i P     phases (default 8)
//   -s S     steps per phase (default 32)
//   -r seed  the seed (default 1)
//   -d       a DXA copy into each block's own local memory every phase
//            (a device built with VX_CFG_EXT_DXA_ENABLE); built with
//            VX_CFG_EXT_MBAR_ENABLE too, also a copy into the next rank's
//            local memory through the window, completing on an mbarrier
//            there, in the first phase and after each cluster.sync() phase
//            end; the receiver waits for it and checks it before its phase end
//   -c       control run: DEFAULT (same-core) clusters through the same
//            window, so nothing crosses the network
//   -v       print every cluster's placement (core:slot by rank)
//
// A launch needs K >= 2 and a device whose largest cluster holds it: a default
// launch the device cannot hold prints SKIPPED, a requested one fails. A launch
// in which no cluster spans two cores fails too: nothing would cross the network.

#include <vortex2.h>
#include <VX_types.h>
#ifdef VX_CFG_EXT_DXA_ENABLE
#include <dxa.h>
#endif
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

const char* kernel_file = "kernel.vxbin";
uint32_t req_k      = 0;
uint32_t req_g      = 0;
uint32_t req_w      = 1;
uint32_t req_lmem   = 0;
uint32_t req_phases = 8;
uint32_t req_steps  = 32;
uint32_t req_seed   = 1;
bool     use_dxa    = false;
bool     verbose    = false;
bool     control    = false;

struct Caps {
  uint64_t num_cores = 0, num_clusters = 0, num_warps = 0, num_threads = 0;
  uint64_t lmem_size = 0, max_cluster = 0, isa = 0;
  bool atomics() const { return (isa & VX_ISA_STD_A) != 0; }
};

struct Launch {
  uint32_t k, g, w, lmem;
  bool requested;
};

void usage() {
  std::printf("Usage: [-k K] [-n clusters] [-w warps per block] [-l bytes] [-i phases]"
              " [-s steps] [-r seed] [-d] [-c] [-v] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:n:w:l:i:s:r:dcvh")) != -1) {
    switch (c) {
      case 'k': req_k      = (uint32_t)std::atoi(optarg); break;
      case 'n': req_g      = (uint32_t)std::atoi(optarg); break;
      case 'w': req_w      = (uint32_t)std::atoi(optarg); break;
      case 'l': req_lmem   = (uint32_t)std::atoi(optarg); break;
      case 'i': req_phases = (uint32_t)std::atoi(optarg); break;
      case 's': req_steps  = (uint32_t)std::atoi(optarg); break;
      case 'r': req_seed   = (uint32_t)std::strtoul(optarg, nullptr, 0); break;
      case 'd': use_dxa    = true;                        break;
      case 'c': control    = true;                        break;
      case 'v': verbose    = true;                        break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  // The generators pack the phase in 8 bits and the step in 12.
  if (optind < argc || req_w == 0 || req_phases == 0 || req_phases > 255
      || req_steps == 0 || req_steps > 4095) {
    usage();
    std::exit(-1);
  }
}

uint32_t dxa_flags() {
#ifdef VX_CFG_EXT_MBAR_ENABLE
  return use_dxa ? FLAG_DXA | FLAG_RDXA : 0;
#else
  return use_dxa ? FLAG_DXA : 0;
#endif
}

uint64_t footprint(uint32_t lmem) {
  return ((uint64_t)lmem + VX_CFG_MEM_BLOCK_SIZE - 1) / VX_CFG_MEM_BLOCK_SIZE * VX_CFG_MEM_BLOCK_SIZE;
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
  const uint32_t nt = L.w * (uint32_t)caps.num_threads;
  if (!caps.atomics())
    return "the device has no A extension";
  if (use_dxa && !(caps.isa & VX_ISA_EXT_DXA))
    return "-d needs a device with DXA";
  if (L.k < 2 || L.k > caps.max_cluster) {
    std::snprintf(buf, sizeof(buf), "K=%u is outside [2, MAX=%u]", L.k, (uint32_t)caps.max_cluster);
    return buf;
  }
  if (L.w > caps.num_warps) {
    std::snprintf(buf, sizeof(buf), "%u warps per block, the core has %u", L.w, (uint32_t)caps.num_warps);
    return buf;
  }
  const uint32_t need = lmem_needed(L.k, nt, L.w, dxa_flags());
  if (L.lmem < need || L.lmem % 64 != 0 || L.lmem > caps.lmem_size) {
    std::snprintf(buf, sizeof(buf), "%u bytes of local memory: needs a multiple of 64 from %u up to %u",
                  L.lmem, need, (uint32_t)caps.lmem_size);
    return buf;
  }
  // A same-core cluster syncs on one core, so all its blocks must fit there.
  if (control && slots_per_core(caps, L) < L.k)
    return "a core cannot hold the whole same-core cluster";
  return "";
}

uint64_t dword(const uint32_t* p) {
  return ((uint64_t)p[1] << 32) | p[0];
}

// The mismatches of one launch.
int run_launch(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps,
               const Launch& L, uint32_t seed) {
  const uint32_t num_blocks = L.g * L.k;
  const uint32_t nt = L.w * (uint32_t)caps.num_threads;
  const uint32_t threads = num_blocks * nt;
  const uint32_t dw = dump_words(L.k, nt);
  std::printf("dlmem_stress: %s K=%u, %u clusters, %u warps and %u bytes of local memory per"
              " block, %u phases of %u steps, seed %#x%s\n", control ? "DEFAULT" : "SPREAD", L.k,
              L.g, L.w, L.lmem, req_phases, req_steps, seed,
              ((dxa_flags() & FLAG_RDXA) ? ", DXA copies, also into the next rank" : use_dxa ? ", DXA copies" : ""));

  const size_t err_words  = (size_t)threads * ERR_WORDS;
  const size_t rec_words  = (size_t)num_blocks * REC_WORDS;
  const size_t dump_total = (size_t)num_blocks * dw;
  std::vector<uint32_t> mem(err_words + rec_words + dump_total, 0);
  uint32_t* errs = mem.data();
  uint32_t* recs = errs + err_words;
  uint32_t* dump = recs + rec_words;
  for (uint32_t t = 0; t < threads; ++t)
    errs[(size_t)t * ERR_WORDS + ERR_COUNT] = ~0u;
  std::fill(recs, dump, ~0u);
  for (uint32_t b = 0; b < num_blocks; ++b)
    recs[(size_t)b * REC_WORDS + REC_RAN] = 0;
  std::fill(dump, dump + dump_total, 0xDEADBEEFu);
  const uint64_t bytes = mem.size() * sizeof(uint32_t);
  vx_buffer_h buf = nullptr;
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &buf));
  CHECK(vx_enqueue_write(q, buf, 0, mem.data(), bytes, 0, nullptr, nullptr));

  stress_arg_t arg = {};
  uint64_t base = 0;
  CHECK(vx_buffer_address(buf, &base));
  arg.err_addr    = base;
  arg.rec_addr    = base + err_words * sizeof(uint32_t);
  arg.dump_addr   = arg.rec_addr + rec_words * sizeof(uint32_t);
  arg.k           = L.k;
  arg.lmem_bytes  = L.lmem;
  arg.seed        = seed;
  arg.phases      = req_phases;
  arg.steps       = req_steps;
  arg.num_threads = (uint32_t)caps.num_threads;
  arg.flags       = dxa_flags();

  vx_buffer_h dxa_buf = nullptr;
  // The queued write reads it when the queue runs it, so it lives until the read completes.
  std::vector<uint32_t> src;
  if (use_dxa) {
#ifdef VX_CFG_EXT_DXA_ENABLE
    src.resize(DXA_TILES * DXA_WORDS);
    for (uint32_t i = 0; i < src.size(); ++i)
      src[i] = dxa_word(seed, i);
    const uint64_t src_bytes = src.size() * sizeof(uint32_t);
    CHECK(vx_buffer_create(dev, src_bytes, VX_MEM_READ, &dxa_buf));
    CHECK(vx_enqueue_write(q, dxa_buf, 0, src.data(), src_bytes, 0, nullptr, nullptr));
    CHECK(vx_buffer_address(dxa_buf, &arg.dxa_addr));
    if (vortex::dxa::program_1d(dev, 0, arg.dxa_addr, DXA_TILES * DXA_WORDS, DXA_WORDS, sizeof(uint32_t))) {
      std::printf("*** programming the DXA descriptor failed\n");
      vx_queue_finish(q, VX_TIMEOUT_INFINITE);
      return 1;
    }
#endif
  }

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
  if (dxa_buf)
    vx_buffer_release(dxa_buf);

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

  // Every thread's cells and the shared counters, from the same plan the kernel ran.
  const uint32_t cells_per_slice = L.k * nt * ST_CELLS;
  uint64_t ops = 0;
  for (uint32_t c = 0; c < L.g; ++c) {
    std::vector<uint64_t> img((size_t)L.k * cells_per_slice, 0);
    std::vector<uint32_t> ctrs((size_t)L.k * ST_CTRS, 0);
    for (uint32_t r = 0; r < L.k; ++r) {
      for (uint32_t t = 0; t < nt; ++t) {
        uint64_t cells[ST_CELLS] = {};
        const uint32_t w = t / (uint32_t)caps.num_threads;
        for (uint32_t p = 0; p < req_phases; ++p) {
          for (uint32_t s = 0; s < req_steps; ++s) {
            st_step_t st;
            st_plan(&st, st_warp_rnd(seed, c, p, s, r, w), st_lane_rnd(seed, c, p, s, r, t),
                    st_value(seed, c, p, s, r, t), L.k, cells);
            if (st.op == OPC_CTR)
              ctrs[(size_t)st.ctr_rank * ST_CTRS + st.ctr_idx] += (uint32_t)st.v;
            ++ops;
          }
        }
        for (uint32_t j = 0; j < ST_CELLS; ++j)
          img[(size_t)cell_rank(r, j, L.k) * cells_per_slice + cell_index(r, t, j, nt)] = cells[j];
      }
    }
    for (uint32_t q = 0; q < L.k; ++q) {
      const uint32_t b = c * L.k + q;
      const uint32_t* d = dump + (size_t)b * dw;
      for (uint32_t i = 0; i < cells_per_slice; ++i) {
        const uint64_t want = img[(size_t)q * cells_per_slice + i];
        if (dword(d + 2 * i) != want)
          report(b, "cell (index in the high word)", ((uint64_t)i << 32) ^ dword(d + 2 * i), want);
      }
      for (uint32_t i = 0; i < ST_CTRS; ++i) {
        const uint32_t want = ctrs[(size_t)q * ST_CTRS + i];
        if (d[2 * cells_per_slice + i] != want)
          report(b, "shared counter", ((uint64_t)i << 32) | d[2 * cells_per_slice + i], want);
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
  std::printf("dlmem_stress: %llu operations, %u of %u clusters span several cores\n",
              (unsigned long long)ops, spanning, L.g);
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
  std::printf("dlmem_stress: %u cores, %u warps of %u threads, %u bytes of local memory,"
              " largest cluster %u, atomics %s\n", (uint32_t)caps.num_cores,
              (uint32_t)caps.num_warps, (uint32_t)caps.num_threads, (uint32_t)caps.lmem_size,
              (uint32_t)caps.max_cluster, caps.atomics() ? "on" : "off");

  int errors = 0;
  if (!(caps.isa & VX_ISA_EXT_DLMEM)) {
    std::printf("*** the device has no DLMEM\n");
    ++errors;
  }
#ifndef VX_CFG_EXT_DXA_ENABLE
  if (use_dxa) {
    std::printf("*** -d needs the test built with VX_CFG_EXT_DXA_ENABLE\nFAILED!\n");
    return 1;
  }
#endif

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, "stress_k", &kernel));

  const bool requested = (req_k != 0);
  std::vector<Launch> launches;
  for (uint32_t k : requested ? std::vector<uint32_t>{req_k} : std::vector<uint32_t>{2, 4}) {
    const uint32_t nt = req_w * (uint32_t)caps.num_threads;
    const uint32_t lmem = req_lmem ? req_lmem : ((lmem_needed(k, nt, req_w, dxa_flags()) + 63) & ~63u);
    launches.push_back({k, req_g, req_w, lmem, requested});
  }
  for (size_t i = 0; i < launches.size(); ++i) {
    Launch& L = launches[i];
    const std::string why = infeasible(caps, L);
    if (!why.empty()) {
      std::printf("%s: K=%u, %u warps and %u bytes per block: %s\n",
                  L.requested ? "Error" : "SKIPPED", L.k, L.w, L.lmem, why.c_str());
      errors += L.requested ? 1 : 0;
      continue;
    }
    if (L.g == 0) {
      const uint32_t slots = (uint32_t)caps.num_cores * slots_per_core(caps, L);
      L.g = std::max<uint32_t>(1, slots / L.k) + 1;
    }
    errors += run_launch(dev, q, kernel, caps, L, hash3(req_seed, (uint32_t)i, L.k));
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
