// mbar_remote: mbarrier ops on the objects of the other blocks of a
// thread-block cluster, through the distributed local-memory window, on a
// DLMEM device with the A extension and mbarriers (RV64). Clusters are SPREAD,
// so the ranks sit on different cores and their ops cross the network.
//
//   -m all     every thread arrives on every rank's object each phase, then
//              waits on its own; rank p % K arrives late in phase p, so its
//              remote arrive completes the others' phases while they sleep
//              (default)
//   -m data    a ring: window stores into the next rank's buffer, then a
//              remote arrive on its FULL object; the next rank waits, reads
//              and arrives on the producer's EMPTY object
//   -m global  the ring over global buffers, with
//              vx_mbar_arrive_release_cluster() and the waiter's
//              vx_mbar_try_wait_acquire_cluster(): the consumer reads its
//              buffer into its L1 before each wait
//   -m tx      remote complete_tx, expect_tx and arrive_tx: every thread
//              completes bytes on every peer's object, the expecting arrives
//              come late, so tx goes negative in between
//   -m self    -m all in DEFAULT clusters: every peer is on the caller's core
//   -k K       blocks per cluster (default 2)
//   -n G       clusters (default 2)
//   -w W       warps per block (default 2)
//   -r R       phases (default 16, at most 64)
//   -d D       idle iterations of the phase's late rank (default 8; global 256;
//              parity 4)
//   -e E       data, global: doublewords per buffer (default 64)
//   -b B       data, global: stores per element and phase (default 2)
//   -R         all, self: relaxed arrives (but each rank's last of a phase);
//              then a launch that times a relaxed and a release arrive
//              behind window stores, and the relaxed one must be cheaper
//              every time; data -p: a relaxed arrive behind the stores,
//              then the release one
//   -i I       -R: measurements of each kind (default 8)
//   -s S       -R: window stores in front of each arrive (default 4, at most 8)
//   -F         global: the plain arrive and wait instead of the _cluster
//              forms; the run then fails with the count of stale reads
//   -p         data: parity, a fixed schedule with an exact instruction
//              count (a cluster barrier per phase, one producer rank per phase,
//              each wait one try_wait that the remote arrive wakes)
//   -x wait    a run that must stop the simulator: a try_wait on a peer's
//              object
//   -v         print every thread with an error and the placement
//
// The host checks each block's placement and every kernel check, replays
// every object with mbar_ref against its final word, and checks the
// coverage: the op that completed each phase (each block records the op its
// returned word shows to complete one) is the one the schedule makes last,
// and in every phase whose completion comes late from a peer every warp of
// the waiting block slept and was woken before the time limit. SPREAD runs
// need such a peer on another core, self runs need every peer on the
// caller's core. tx runs must see a negative tx.

#include <vortex2.h>
#include <VX_types.h>
#include <mbar_ref.h>
#include "common.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace {

namespace mr = vortex::mbar_ref;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

const char* const kModeNames[] = {"all", "data", "global", "tx", "self"};
// all, data, tx and self delay a rank that already knows the others are about
// to wait; global delays a producer that does not, past the consumer's refill
// of its L1, so that a wait without the acquire reads stale lines.
const uint32_t kDefDelay[] = {8, 8, 256, 8, 8};
// A parity producer starts with the waiters, at the cluster barrier: the delay
// only has to cover their try_wait, and its whole fill must beat the time limit.
const uint32_t kParityDelay = 4;

const char* kernel_file = "kernel.vxbin";
uint32_t mode    = MODE_ALL;
uint32_t req_k   = 2;
uint32_t req_g   = 2;
uint32_t req_w   = 2;
uint32_t rounds  = 16;
uint32_t delay   = ~0u;
uint32_t elems   = 64;
uint32_t burst   = 2;
uint32_t iters   = 8;
uint32_t stores  = 4;
bool     relaxed = false;
bool     unfenced = false;
bool     parity  = false;
bool     neg_wait = false;
bool     verbose = false;

void usage() {
  std::printf("Usage: [-m all|data|global|tx|self] [-k K] [-n clusters] [-w warps] [-r phases] [-d delay]"
              " [-e elems] [-b burst] [-R [-i iters] [-s stores]] [-F] [-p] [-x wait] [-v] [-h]\n");
}

uint32_t parse_count(const char* s) {
  char* end = nullptr;
  errno = 0;
  const unsigned long v = std::strtoul(s, &end, 10);
  if (*s < '0' || *s > '9' || *end != '\0' || errno != 0 || v > UINT32_MAX) {
    usage();
    std::exit(-1);
  }
  return (uint32_t)v;
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "m:k:n:w:r:d:e:b:i:s:RFpx:vh")) != -1) {
    switch (c) {
      case 'm':
        for (mode = 0; mode < NUM_MODES && std::strcmp(optarg, kModeNames[mode]); ++mode) {}
        if (mode == NUM_MODES) {
          usage();
          std::exit(-1);
        }
        break;
      case 'k': req_k  = parse_count(optarg); break;
      case 'n': req_g  = parse_count(optarg); break;
      case 'w': req_w  = parse_count(optarg); break;
      case 'r': rounds = parse_count(optarg); break;
      case 'd': delay  = parse_count(optarg); break;
      case 'e': elems  = parse_count(optarg); break;
      case 'b': burst  = parse_count(optarg); break;
      case 'i': iters  = parse_count(optarg); break;
      case 's': stores = parse_count(optarg); break;
      case 'R': relaxed  = true; break;
      case 'F': unfenced = true; break;
      case 'p': parity   = true; break;
      case 'x':
        if (std::strcmp(optarg, "wait") != 0) {
          usage();
          std::exit(-1);
        }
        neg_wait = true;
        break;
      case 'v': verbose = true; break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (delay == ~0u)
    delay = parity ? kParityDelay : kDefDelay[mode];
  const bool rings = (mode == MODE_DATA || mode == MODE_GLOBAL);
  if (optind < argc || req_k < 2 || req_k > MAX_CLUSTER || req_g == 0 || req_w == 0 || rounds == 0 || rounds > MAX_ROUNDS
      || elems == 0 || burst == 0 || iters == 0 || iters > (uint32_t)(mr::COUNT_MAX / 2)
      || stores > MAX_STORES
      || (relaxed && mode != MODE_ALL && mode != MODE_SELF && !parity)
      || (unfenced && mode != MODE_GLOBAL)
      || (parity && (mode != MODE_DATA || rounds < req_k))
      || (rings && rounds < 2)) {
    usage();
    std::exit(-1);
  }
}

struct Caps {
  uint64_t num_cores = 0, num_warps = 0, num_threads = 0, lmem_size = 0, max_cluster = 0, isa = 0;
};

void print_word(const char* what, uint64_t w) {
  std::printf("%s %#018lx (phase %u, pending %u, expected %u, tx %ld)", what, (unsigned long)w,
              mr::phase(w), mr::pending(w), mr::expected(w), (long)mr::tx(w));
}

// The word of an object of `count` after `phases` completed phases.
uint64_t after_phases(uint32_t count, uint32_t phases) {
  uint64_t w = mr::init(count);
  for (uint32_t p = 0; p < phases; ++p)
    w = mr::apply(mr::OP_ARRIVE, w, count).new_word;
  return w;
}

// Phases of FULL[s] (and EMPTY[s]) in R phases that alternate between the two.
uint32_t phases_of(uint32_t s, uint32_t R) {
  return (R + 1 - s) / 2;
}

struct Run {
  uint32_t k, g, w, nt, blocks, threads;
  vx_cluster_policy_e policy;
  std::vector<uint32_t> recs, errs, comps;
  std::vector<uint64_t> words;
  const uint32_t* err(uint32_t t) const { return &errs[(size_t)t * ERR_WORDS]; }
  const uint32_t* rec(uint32_t b) const { return &recs[(size_t)b * REC_WORDS]; }
};

template <typename T>
vx_buffer_h upload(vx_device_h dev, vx_queue_h q, const std::vector<T>& v, uint64_t* addr) {
  vx_buffer_h buf = nullptr;
  const uint64_t bytes = v.size() * sizeof(T);
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &buf));
  CHECK(vx_enqueue_write(q, buf, 0, v.data(), bytes, 0, nullptr, nullptr));
  CHECK(vx_buffer_address(buf, addr));
  return buf;
}

template <typename T>
vx_event_h download(vx_queue_h q, std::vector<T>& v, vx_buffer_h buf, vx_event_h after) {
  vx_event_h ev = nullptr;
  CHECK(vx_enqueue_read(q, v.data(), buf, 0, v.size() * sizeof(T), 1, &after, &ev));
  return ev;
}

// One launch end to end; returns false if the launch could not run.
bool launch(vx_device_h dev, vx_queue_h q, vx_module_h mod, const char* kname, const Caps& caps,
            rem_arg_t arg, Run& run, uint32_t stride) {
  run.blocks  = run.g * run.k;
  run.threads = run.w * run.nt;
  run.recs.assign((size_t)run.blocks * REC_WORDS, ~0u);
  for (uint32_t b = 0; b < run.blocks; ++b)
    run.recs[(size_t)b * REC_WORDS + REC_RAN] = 0;
  run.errs.assign((size_t)run.blocks * run.threads * ERR_WORDS, 0);
  for (size_t t = 0; t < (size_t)run.blocks * run.threads; ++t)
    run.errs[t * ERR_WORDS + ERR_COUNT] = ~0u;
  run.comps.assign((size_t)run.blocks * run.k * arg.rounds, 0);
  run.words.assign((size_t)run.blocks * NUM_OBJS, ~uint64_t(0));
  std::vector<uint64_t> glob(stride ? (size_t)run.blocks * 2 * stride : 1, 0);

  vx_buffer_h rec_buf  = upload(dev, q, run.recs, &arg.rec_addr);
  vx_buffer_h err_buf  = upload(dev, q, run.errs, &arg.err_addr);
  vx_buffer_h comp_buf = upload(dev, q, run.comps, &arg.comp_addr);
  vx_buffer_h word_buf = upload(dev, q, run.words, &arg.word_addr);
  vx_buffer_h glob_buf = upload(dev, q, glob, &arg.glob_addr);
  arg.k = run.k;
  arg.stride = stride;

  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, kname, &kernel));
  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kernel;
  li.args_host      = &arg;
  li.args_size      = sizeof(arg);
  li.ndim           = 1;
  li.grid_dim[0]    = run.blocks;
  li.grid_dim[1]    = 1;
  li.grid_dim[2]    = 1;
  li.block_dim[0]   = run.threads;
  li.block_dim[1]   = 1;
  li.block_dim[2]   = 1;
  li.lmem_size      = (uint32_t)((LM_DATA + 2 * (size_t)elems) * sizeof(uint64_t));
  li.cluster_dim[0] = run.k;
  li.cluster_dim[1] = 1;
  li.cluster_dim[2] = 1;
  li.cluster_policy = run.policy;
  if (li.lmem_size > caps.lmem_size) {
    std::printf("*** %u bytes of local memory per block, the core has %lu\n", li.lmem_size,
                (unsigned long)caps.lmem_size);
    return false;
  }
  vx_event_h launch_ev = nullptr;
  const vx_result_t r = vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev);
  if (r != VX_SUCCESS) {
    std::printf("*** %s: the launch returned %s\n", kname, vx_result_string(r));
    return false;
  }
  std::vector<vx_event_h> evs = {download(q, run.recs, rec_buf, launch_ev),
                                 download(q, run.errs, err_buf, launch_ev),
                                 download(q, run.comps, comp_buf, launch_ev),
                                 download(q, run.words, word_buf, launch_ev)};
  for (vx_event_h ev : evs) {
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }
  vx_event_release(launch_ev);
  vx_kernel_release(kernel);
  vx_buffer_release(glob_buf);
  vx_buffer_release(word_buf);
  vx_buffer_release(comp_buf);
  vx_buffer_release(err_buf);
  vx_buffer_release(rec_buf);
  return true;
}

// Placement, per-thread checks, and the final words against `want`
// (NUM_OBJS per block).
int check_common(const Caps& caps, const Run& run, const std::vector<uint64_t>& want, const char* name) {
  int errors = 0;
  for (uint32_t b = 0; b < run.blocks; ++b) {
    const uint32_t* rec = run.rec(b);
    if (rec[REC_RAN] != 1 || rec[REC_BLOCK] != b || rec[REC_RANK] != b % run.k || rec[REC_SIZE] != run.k
        || rec[REC_CORE] >= caps.num_cores) {
      std::printf("*** %s: block %u ran %u times as block %u, rank %u of %u, core %u\n", name, b,
                  rec[REC_RAN], rec[REC_BLOCK], rec[REC_RANK], rec[REC_SIZE], rec[REC_CORE]);
      ++errors;
    }
  }
  if (verbose) {
    for (uint32_t g = 0; g < run.g; ++g) {
      std::printf("%s: cluster %u:", name, g);
      for (uint32_t r = 0; r < run.k; ++r)
        std::printf(" %u:%u", run.rec(g * run.k + r)[REC_CORE], run.rec(g * run.k + r)[REC_SLOT]);
      std::printf("\n");
    }
  }
  uint32_t failed = 0, unfinished = 0;
  uint64_t checks = 0;
  for (uint32_t t = 0; t < run.blocks * run.threads; ++t) {
    const uint32_t* e = run.err(t);
    if (e[ERR_COUNT] == ~0u) {
      ++unfinished;
      continue;
    }
    if (e[ERR_COUNT] == 0)
      continue;
    checks += e[ERR_COUNT];
    if (verbose || failed == 0)
      std::printf("*** %s: block %u thread %u: %u failed checks, first in phase %u, check %u\n", name,
                  t / run.threads, t % run.threads, e[ERR_COUNT], e[ERR_FIRST] - 1, e[ERR_WHAT]);
    ++failed;
  }
  std::printf("%s: %u threads with %lu failed checks, %u unfinished threads\n", name, failed,
              (unsigned long)checks, unfinished);
  errors += (failed != 0) + (unfinished != 0);
  for (uint32_t b = 0; b < run.blocks; ++b) {
    for (uint32_t i = 0; i < NUM_OBJS; ++i) {
      const uint64_t got = run.words[(size_t)b * NUM_OBJS + i], exp = want[(size_t)b * NUM_OBJS + i];
      if (got == exp)
        continue;
      if (errors < 16) {
        std::printf("*** %s: block %u object %u:", name, b, i);
        print_word(" final", got);
        print_word(", mbar_ref", exp);
        std::printf("\n");
      }
      ++errors;
    }
  }
  return errors;
}

// The coverage of the completions and wake-ups. `last(T, p)` is the rank
// whose op must complete block T's FULL phase p, or -1 for any (exactly one
// op must); `late(T, p)` says whether that op comes late from a peer, so T's
// warps must have slept and been woken by it.
template <typename Last, typename Late>
int check_wakes(const Run& run, uint32_t R, Last last, Late late, bool same_core, const char* name) {
  int errors = 0;
  uint32_t eligible = 0, remote = 0, woken = 0;
  for (uint32_t T = 0; T < run.blocks; ++T) {
    const uint32_t base = T - T % run.k;
    for (uint32_t p = 0; p < R; ++p) {
      // The ops of each block of the cluster that completed T's phase p.
      uint32_t ops = 0, from = 0;
      for (uint32_t r = 0; r < run.k; ++r) {
        const uint32_t c = run.comps[((size_t)(base + r) * run.k + T % run.k) * R + p];
        ops += c;
        if (c)
          from = base + r;
      }
      const int lr = last(T, p);
      const uint32_t want = lr < 0 ? from : base + (uint32_t)lr;
      if (ops != 1 || from != want) {
        if (errors < 16)
          std::printf("*** %s: block %u phase %u: completed by %u op(s), the last from block %u,"
                      " expected one from block %u\n", name, T, p, ops, from, want);
        ++errors;
        continue;
      }
      if (!late(T, p))
        continue;
      ++eligible;
      const bool cross = run.rec(want)[REC_CORE] != run.rec(T)[REC_CORE];
      if (cross == same_core) {
        if (errors < 16)
          std::printf("*** %s: block %u phase %u: completed from block %u on core %u, the waiter is on core %u\n",
                      name, T, p, want, run.rec(want)[REC_CORE], run.rec(T)[REC_CORE]);
        ++errors;
      }
      remote += cross;
      bool all = true;
      for (uint32_t w = 0; w < run.w; ++w) {
        const uint32_t* e = run.err(T * run.threads + w * run.nt);
        const uint64_t mask = ((uint64_t)e[ERR_WOKE_HI] << 32) | e[ERR_WOKE_LO];
        all &= ((mask >> p) & 1) != 0;
      }
      if (!all) {
        if (errors < 16)
          std::printf("*** %s: block %u phase %u: a warp did not sleep until block %u's late op woke it\n",
                      name, T, p, want);
        ++errors;
      }
      woken += all;
    }
  }
  uint64_t waits = 0, slept = 0, woke = 0, timeouts = 0;
  for (uint32_t t = 0; t < run.blocks * run.threads; ++t) {
    waits    += run.err(t)[ERR_WAITS];
    slept    += run.err(t)[ERR_SLEPT];
    woke     += run.err(t)[ERR_WOKE];
    timeouts += run.err(t)[ERR_TIMEOUTS];
  }
  std::printf("%s: %u late completions from a peer (%u from another core), %u with every warp woken;"
              " %lu thread waits, %lu suspending try_waits, %lu woken, %lu time-outs\n",
              name, eligible, remote, woken, (unsigned long)waits, (unsigned long)slept, (unsigned long)woke,
              (unsigned long)timeouts);
  if (eligible == 0 || (!same_core && remote == 0)) {
    std::printf("*** %s: no phase was completed late by a peer%s\n", name, same_core ? "" : " on another core");
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
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,         &caps.num_warps));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &caps.num_threads));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE,    &caps.lmem_size));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &caps.max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &caps.isa));

  const uint32_t k = neg_wait ? 2 : req_k;
  const bool self = (mode == MODE_SELF);
  std::printf("mbar_remote: %s%s%s%s%s, %u cores of %u warps x %u threads, %u clusters of %u blocks of %u warps,"
              " %u phases, delay %u\n",
              neg_wait ? "-x wait" : kModeNames[mode], relaxed ? " (relaxed)" : "", unfenced ? " (unfenced)" : "",
              parity ? " (parity)" : "", self ? " DEFAULT" : " SPREAD", (uint32_t)caps.num_cores,
              (uint32_t)caps.num_warps, (uint32_t)caps.num_threads, req_g, k, req_w, rounds, delay);
  if (!(caps.isa & VX_ISA_EXT_DLMEM) || !(caps.isa & VX_ISA_STD_A)) {
    std::printf("Error: the device needs DLMEM and the A extension\nFAILED!\n");
    return 1;
  }
  if (self ? k * req_w > caps.num_warps : (k > caps.max_cluster || caps.num_cores < 2 || req_w > caps.num_warps)) {
    std::printf("Error: %s clusters of %u blocks of %u warps do not fit %u cores of %u warps (largest cluster %u)\n"
                "FAILED!\n", self ? "DEFAULT" : "SPREAD", k, req_w, (uint32_t)caps.num_cores,
                (uint32_t)caps.num_warps, (uint32_t)caps.max_cluster);
    return 1;
  }

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));

  rem_arg_t arg = {};
  arg.mode    = mode;
  arg.rounds  = rounds;
  arg.delay   = delay;
  arg.elems   = elems;
  arg.burst   = burst;
  arg.relaxed = relaxed;
  arg.fenced  = !unfenced;
  arg.parity  = parity;
  arg.iters   = iters;
  arg.stores  = stores;

  Run run;
  run.k = k;
  run.g = neg_wait ? 1 : req_g;
  run.w = req_w;
  run.nt = (uint32_t)caps.num_threads;
  run.policy = self ? VX_CLUSTER_POLICY_DEFAULT : VX_CLUSTER_POLICY_SPREAD;
  const uint32_t R = rounds;

  int errors = 0;
  if (neg_wait) {
    if (launch(dev, q, mod, "neg_k", caps, arg, run, 0))
      std::printf("*** the wait on a peer's object ran to completion\n");
    errors = 1;
  } else {
    const char* const kKernels[] = {"all_k", "data_k", "global_k", "tx_k", "all_k"};
    const char* kname = parity ? (relaxed ? "pdata_rlx_k" : "pdata_k") : kKernels[mode];
    const uint32_t line_words = VX_CFG_MEM_BLOCK_SIZE / sizeof(uint64_t);
    const uint32_t stride = (mode == MODE_GLOBAL) ? (elems + line_words - 1) / line_words * line_words : 0;
    if (!launch(dev, q, mod, kname, caps, arg, run, stride)) {
      errors = 1;
    } else {
      const uint32_t threads = run.threads;
      std::vector<uint64_t> want((size_t)run.blocks * NUM_OBJS, 0);
      for (uint32_t b = 0; b < run.blocks; ++b) {
        uint64_t* w = &want[(size_t)b * NUM_OBJS];
        const uint32_t rank = b % k;
        switch (mode) {
        case MODE_ALL:
        case MODE_SELF:
          for (uint32_t s = 0; s < 2; ++s)
            w[OBJ_FULL + s] = after_phases(k * threads, phases_of(s, R));
          break;
        case MODE_TX:
          for (uint32_t s = 0; s < 2; ++s)
            w[OBJ_FULL + s] = after_phases(k * threads, phases_of(s, R));
          break;
        default:
          if (parity) {
            w[OBJ_FULL] = after_phases((relaxed ? 2 : 1) * threads, R - (R + k - 1 - rank) / k);
          } else {
            for (uint32_t s = 0; s < 2; ++s) {
              w[OBJ_FULL + s]  = after_phases(threads, phases_of(s, R));
              w[OBJ_EMPTY + s] = after_phases(threads, phases_of(s, R));
            }
          }
          break;
        }
      }
      const char* name = kModeNames[mode];
      errors += check_common(caps, run, want, name);

      // Who completes block T's FULL phase p, and whether it comes late from a peer.
      switch (mode) {
      case MODE_ALL:
      case MODE_SELF:
        errors += check_wakes(run, R, [&](uint32_t, uint32_t p) { return (int)(p % k); },
                              [&](uint32_t T, uint32_t p) { return T % k != p % k; }, self, name);
        break;
      case MODE_TX:
        // Odd phase p comes late on rank (p / 2) % k only.
        errors += check_wakes(run, R,
                              [&](uint32_t T, uint32_t p) {
                                return !(p & 1) ? (int)(T % k)
                                     : ((p >> 1) % k == T % k ? (int)((T % k + 1) % k) : -1);
                              },
                              [&](uint32_t T, uint32_t p) { return (p & 1) && (p >> 1) % k == T % k; },
                              false, name);
        break;
      case MODE_DATA:
        if (parity) {
          uint64_t waits = 0, woke = 0;
          for (uint32_t t = 0; t < run.blocks * threads; ++t) {
            waits += run.err(t)[ERR_WAITS];
            woke  += run.err(t)[ERR_WOKE];
          }
          const uint64_t want_waits = (uint64_t)run.blocks * threads * R * (k - 1) / k;
          std::printf("%s: %lu thread waits, %lu woken by the remote arrive\n", name, (unsigned long)waits,
                      (unsigned long)woke);
          if (waits != want_waits || woke != waits) {
            std::printf("*** %s: expected %lu waits, each woken\n", name, (unsigned long)want_waits);
            ++errors;
          }
        } else {
          errors += check_wakes(run, R, [&](uint32_t T, uint32_t) { return (int)((T % k + k - 1) % k); },
                                [&](uint32_t T, uint32_t p) { return T % k == (p + 1) % k; }, false, name);
        }
        break;
      case MODE_GLOBAL: {
        uint64_t stale = 0, reads = (uint64_t)run.blocks * R * elems;
        for (uint32_t t = 0; t < run.blocks * threads; ++t)
          stale += run.err(t)[ERR_STALE];
        uint32_t wrong = 0;
        for (uint32_t T = 0; T < run.blocks; ++T) {
          const uint32_t base = T - T % k, prev = base + (T % k + k - 1) % k;
          for (uint32_t p = 0; p < R; ++p) {
            for (uint32_t r = 0; r < k; ++r)
              wrong += (run.comps[((size_t)(base + r) * k + T % k) * R + p] != (base + r == prev ? 1u : 0u));
          }
        }
        std::printf("%s: %lu stale global reads of %lu\n", name, (unsigned long)stale, (unsigned long)reads);
        if (stale != 0) {
          std::printf("*** %s: %lu stale global reads%s\n", name, (unsigned long)stale,
                      unfenced ? " (expected without the cluster-scope release and acquire)" : "");
          ++errors;
        }
        if (wrong != 0) {
          std::printf("*** %s: %u phases not completed by exactly one arrive of the previous rank\n", name, wrong);
          ++errors;
        }
        break;
      }
      }
      if (mode == MODE_TX) {
        uint64_t negs = 0;
        for (uint32_t t = 0; t < run.blocks * threads; ++t)
          negs += run.err(t)[ERR_NEG];
        std::printf("%s: %lu returned words with a negative tx\n", name, (unsigned long)negs);
        if (negs == 0) {
          std::printf("*** %s: tx never went negative\n", name);
          ++errors;
        }
      }

      if (relaxed && !parity && errors == 0) {
        Run cost;
        cost.k = 2;
        cost.g = 1;
        cost.w = 1;
        cost.nt = run.nt;
        cost.policy = run.policy;
        if (!launch(dev, q, mod, "cost_k", caps, arg, cost, 0)) {
          ++errors;
        } else {
          std::vector<uint64_t> cwant((size_t)2 * NUM_OBJS, 0);
          cwant[OBJ_COST] = mr::init(2 * iters);
          cwant[NUM_OBJS + OBJ_COST] = after_phases(2 * iters, 1);
          errors += check_common(caps, cost, cwant, "cost");
          const uint32_t* e = cost.err(0);
          const bool cross = cost.rec(0)[REC_CORE] != cost.rec(1)[REC_CORE];
          std::printf("cost: %u window stores then an arrive, %u times each, %s: relaxed %u cycles (slowest %u),"
                      " release %u cycles (fastest %u)\n", stores, iters, cross ? "across cores" : "on one core",
                      e[ERR_CYC_RLX], e[ERR_MAX_RLX], e[ERR_CYC_REL], e[ERR_MIN_REL]);
          if (cross == self) {
            std::printf("*** cost: the two blocks are %s\n", cross ? "on two cores" : "on one core");
            ++errors;
          }
          if (stores != 0 && !(e[ERR_MAX_RLX] < e[ERR_MIN_REL])) {
            std::printf("*** cost: a relaxed arrive took as long as a release arrive\n");
            ++errors;
          }
        }
      }
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
