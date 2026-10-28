// mbar_wait: waits on mbarriers with try_wait, which suspends the warp until
// the phase completes or a time limit (VX_CFG_MBAR_SUSPEND_CYCLES) passes, on
// a DLMEM device with the A extension and mbarriers (RV64). One block fills
// the core (see kernel.cpp).
//
//   -m late   per phase, the arriver arrives after an idle delay while the
//             other warps wait on the phase (default)
//   -m sleep  the producer sleeps through R time limits on an object nobody
//             arrives on, so every warp sleeps for longer than the
//             scheduler's stall timeout, then arrives for the waiting warps
//   -m many   every lane of every warp waits, on one object (-o 1) or on
//             several, by warp or by lane (-L)
//   -m mixed  the arrive lands while a waiter warp's lanes read the object, so
//             the first poll splits the warp; at least one phase must
//   -r R      phases; sleep: the producer's time-outs (default per mode)
//   -d D      idle iterations of the arriver before an arrive (default per mode)
//   -o N      many: objects (default 1, at most 16)
//   -L        many: the lanes of a warp wait on different objects
//   -s        poll with test_wait instead of try_wait (late, many, mixed)
//   -p        parity (late): a fixed schedule with an exact instruction count
//   -D S      S warps stream DXA copies into their own tiles meanwhile (default 0)
//   -e E      32-bit elements per DXA tile, even (default 512)
//   -v        print every thread with an error
//
// The host checks the kernel's per-thread checks (a try_wait on one object
// returns false only after the time limit), every final object word against
// mbar_ref, and the time-outs against the arrive's lag: none when every
// arrive is answered at least kSlack cycles within the time limit (a wake-up
// was lost otherwise; mixed and one-object many runs must stay within it),
// one per phase and waiter when every arrive lands after two. A try_wait run
// must suspend a warp at least once.

#include <vortex2.h>
#include <dxa.h>
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

constexpr uint32_t kSrcTiles = 3;
// Barrier ids 0 (__syncthreads) and 1 (the workers) are taken; streamer w uses 2 + w.
constexpr uint32_t kMaxStreamers = VX_CFG_NUM_BARRIERS - 2;
constexpr uint32_t kLimit = VX_CFG_MBAR_SUSPEND_CYCLES;
// An arrive's answer leaves the bank as its phase flips, and a waiter's time
// limit starts after its poll, after it left the meeting: a few cycles apart
// from the arriver, one per warp the scheduler picks first. When every arrive
// is answered kSlack cycles before the limit, no waiter can time out.
constexpr uint32_t kSlack = 64;

const char* kernel_file = "kernel.vxbin";
uint32_t mode       = MODE_LATE;
uint32_t rounds     = 0;
uint32_t delay      = ~0u;
uint32_t objs       = 1;
uint32_t streamers  = 0;
uint32_t tile_elems = 512;
bool     lanes      = false;
bool     spin       = false;
bool     parity     = false;
bool     verbose    = false;

const char* const kModeNames[]    = {"late", "sleep", "many", "mixed"};
const uint32_t    kDefRounds[]    = {16, 100, 8, 96};
const uint32_t    kDefDelay[]     = {100, 0, 2, 8};
const uint32_t    kParityDelay    = 16;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

void usage() {
  std::printf("Usage: [-m late|sleep|many|mixed] [-r rounds] [-d delay] [-o objects] [-L] [-s] [-p]"
              " [-D streamers] [-e tile elements] [-v] [-h]\n");
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
  bool named = false;
  while ((c = getopt(argc, argv, "m:r:d:o:D:e:Lspvh")) != -1) {
    switch (c) {
      case 'm':
        for (mode = 0; mode < 4 && std::strcmp(optarg, kModeNames[mode]); ++mode) {}
        named = (mode < 4);
        if (!named) {
          usage();
          std::exit(-1);
        }
        break;
      case 'r': rounds     = parse_count(optarg); break;
      case 'd': delay      = parse_count(optarg); break;
      case 'o': objs       = parse_count(optarg); break;
      case 'D': streamers  = parse_count(optarg); break;
      case 'e': tile_elems = parse_count(optarg); break;
      case 'L': lanes      = true;                break;
      case 's': spin       = true;                break;
      case 'p': parity     = true;                break;
      case 'v': verbose    = true;                break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (rounds == 0)
    rounds = kDefRounds[mode];
  if (delay == ~0u)
    delay = parity ? kParityDelay : kDefDelay[mode];
  if (optind < argc || tile_elems == 0 || (tile_elems & 1) || streamers > kMaxStreamers
      || objs == 0 || objs > MAX_OBJS || ((objs > 1 || lanes) && mode != MODE_MANY)
      || (parity && (mode != MODE_LATE || spin || streamers))
      || (mode == MODE_SLEEP && (spin || streamers))
      || (mode == MODE_MIXED && spin)) {
    usage();
    std::exit(-1);
  }
}

void print_word(const char* what, uint64_t w) {
  std::printf("%s %#018lx (phase %u, pending %u, expected %u, tx %ld)", what, (unsigned long)w,
              mr::phase(w), mr::pending(w), mr::expected(w), (long)mr::tx(w));
}

// The word of an object of count 1 after `phases` single arrivals.
uint64_t after_phases(uint32_t phases) {
  uint64_t w = mr::init(1);
  for (uint32_t p = 0; p < phases; ++p)
    w = mr::apply(mr::OP_ARRIVE, w, 1).new_word;
  return w;
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

  uint64_t nw = 0, nt = 0, lmem_cap = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,      &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,    &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,      &isa));

  const uint32_t block   = (uint32_t)(nw * nt);
  const uint32_t workers = (uint32_t)nw - streamers;
  const uint32_t tile_word = LM_TILES;
  const uint64_t lmem_bytes = (uint64_t)tile_word * sizeof(uint64_t)
                            + (uint64_t)streamers * tile_elems * sizeof(uint32_t);
  std::printf("mbar_wait: %s mode%s%s, %u warps of %u threads, %u rounds, delay %u, %u object%s%s,"
              " %u DXA streamers, time limit %u\n",
              kModeNames[mode], spin ? " (test_wait spin)" : "", parity ? " (parity)" : "",
              (uint32_t)nw, (uint32_t)nt, rounds, delay, objs, objs > 1 ? "s" : "",
              lanes ? " by lane" : "", streamers, kLimit);

  if (!(isa & VX_ISA_EXT_DLMEM) || !(isa & VX_ISA_STD_A) || (streamers && !(isa & VX_ISA_EXT_DXA))) {
    std::printf("Error: the device needs DLMEM and the A extension%s\nFAILED!\n", streamers ? ", and DXA" : "");
    return 1;
  }
  if (workers < 2) {
    std::printf("Error: %u warps leave %d workers, the mode needs 2\nFAILED!\n", (uint32_t)nw,
                (int)nw - (int)streamers);
    return 1;
  }
  if (lmem_bytes > lmem_cap) {
    std::printf("Error: %lu bytes of local memory, the core has %lu\nFAILED!\n",
                (unsigned long)lmem_bytes, (unsigned long)lmem_cap);
    return 1;
  }

  // Every host buffer stays alive until the queue has run its upload.
  std::vector<uint32_t> h_src;
  vx_buffer_h src = nullptr;
  if (streamers) {
    h_src.resize(tile_elems * kSrcTiles);
    for (size_t i = 0; i < h_src.size(); ++i)
      h_src[i] = (uint32_t)(i * 2654435761u);
    uint64_t src_addr = 0;
    CHECK(vx_buffer_create(dev, h_src.size() * sizeof(uint32_t), VX_MEM_READ, &src));
    CHECK(vx_buffer_address(src, &src_addr));
    CHECK(vx_enqueue_write(q, src, 0, h_src.data(), h_src.size() * sizeof(uint32_t), 0, nullptr, nullptr));
    if (vortex::dxa::program_1d(dev, 0, src_addr, (uint32_t)h_src.size(), tile_elems, sizeof(uint32_t)) != 0) {
      std::printf("Error: cannot program the DXA descriptor\nFAILED!\n");
      return 1;
    }
  }

  std::vector<uint32_t> err((size_t)block * ERR_WORDS, 0);
  for (uint32_t t = 0; t < block; ++t)
    err[(size_t)t * ERR_WORDS + ERR_COUNT] = ~0u;
  const uint64_t err_bytes = err.size() * sizeof(uint32_t);
  vx_buffer_h err_buf = nullptr;
  CHECK(vx_buffer_create(dev, err_bytes, VX_MEM_READ_WRITE, &err_buf));
  CHECK(vx_enqueue_write(q, err_buf, 0, err.data(), err_bytes, 0, nullptr, nullptr));

  std::vector<uint64_t> words(NUM_OBJS, ~uint64_t(0));
  const uint64_t word_bytes = words.size() * sizeof(uint64_t);
  vx_buffer_h word_buf = nullptr;
  CHECK(vx_buffer_create(dev, word_bytes, VX_MEM_READ_WRITE, &word_buf));
  CHECK(vx_enqueue_write(q, word_buf, 0, words.data(), word_bytes, 0, nullptr, nullptr));

  mbar_arg_t arg = {};
  CHECK(vx_buffer_address(err_buf,  &arg.err_addr));
  CHECK(vx_buffer_address(word_buf, &arg.word_addr));
  arg.mode          = mode;
  arg.rounds        = rounds;
  arg.delay         = delay;
  arg.objs          = objs;
  arg.lanes         = lanes;
  arg.spin          = spin;
  arg.parity        = parity;
  arg.streamers     = streamers;
  arg.tile_word     = tile_word;
  arg.tile_elems    = tile_elems;
  arg.num_src_tiles = kSrcTiles;

  const char* const kKernels[] = {"late_k", "sleep_k", "many_k", "mixed_k"};
  vx_module_h mod = nullptr;
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, kKernels[mode], &kernel));

  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.grid_dim[1]  = 1;
  li.grid_dim[2]  = 1;
  li.block_dim[0] = block;
  li.block_dim[1] = 1;
  li.block_dim[2] = 1;
  li.lmem_size    = (uint32_t)lmem_bytes;

  vx_event_h launch_ev = nullptr, read_ev = nullptr, word_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, err.data(), err_buf, 0, err_bytes, 1, &launch_ev, &read_ev));
  CHECK(vx_enqueue_read(q, words.data(), word_buf, 0, word_bytes, 1, &launch_ev, &word_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(word_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(word_ev);
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;

  // Per-thread checks and counts. The waiters are the worker threads outside
  // the arriver's warp, except in many mode, where every worker thread waits.
  const uint32_t w0 = streamers * (uint32_t)nt;
  const uint32_t a0 = w0;   // the arriver: lane 0 of worker warp 0
  const uint32_t first_waiter = (mode == MODE_MANY) ? w0 : w0 + (uint32_t)nt;
  uint32_t failed = 0, unfinished = 0, waiters = 0, few = 0;
  uint64_t calls = 0, timeouts = 0, early = 0, slept = 0, mixed = 0, checks = 0;
  uint32_t minfalse = ~0u, elapsed = 0, min_waiter_timeouts = ~0u;
  for (uint32_t t = 0; t < block; ++t) {
    const uint32_t* e = &err[(size_t)t * ERR_WORDS];
    if (e[ERR_COUNT] == ~0u) {
      ++unfinished;
      continue;
    }
    if (t < w0)
      continue;
    if (e[ERR_COUNT] != 0) {
      checks += e[ERR_COUNT];
      ++failed;
      if (verbose || failed == 1)
        std::printf("*** thread %u (warp %u lane %u): %u failed checks, first in round %u, check %u\n",
                    t, t / (uint32_t)nt, t % (uint32_t)nt, e[ERR_COUNT], e[ERR_FIRST] - 1, e[ERR_WHAT]);
    }
    elapsed = std::max(elapsed, e[ERR_ELAPSED]);
    mixed += e[ERR_MIXED];
    if (t < first_waiter)
      continue;
    ++waiters;
    calls    += e[ERR_CALLS];
    timeouts += e[ERR_TIMEOUTS];
    early    += e[ERR_EARLY];
    slept    += e[ERR_SLEPT];
    minfalse = std::min(minfalse, e[ERR_MINFALSE]);
    min_waiter_timeouts = std::min(min_waiter_timeouts, e[ERR_TIMEOUTS]);
    few += (e[ERR_TIMEOUTS] < rounds);
  }
  const uint32_t* ea = &err[(size_t)a0 * ERR_WORDS];
  const uint32_t lagmin = ea[ERR_LAGMIN], lagmax = ea[ERR_LAGMAX];
  std::printf("mbar_wait: %u threads with %lu failed checks, %u unfinished threads\n",
              failed, (unsigned long)checks, unfinished);
  errors += (failed != 0) + (unfinished != 0);

  const uint64_t waits = (uint64_t)waiters * (mode == MODE_SLEEP ? 1 : rounds);
  std::printf("mbar_wait: %u waiter threads, %lu waits, %lu %s (%.2f per wait), %lu time-outs, %lu early returns,"
              " %lu suspending polls, fewest cycles of a false poll %d, %lu split first polls, arrive lag %u..%u,"
              " %u cycles\n",
              waiters, (unsigned long)waits, (unsigned long)calls, spin ? "test_wait polls" : "wait calls",
              waits ? (double)calls / (double)waits : 0.0, (unsigned long)timeouts, (unsigned long)early,
              (unsigned long)slept, minfalse == ~0u ? -1 : (int)minfalse, (unsigned long)mixed,
              parity ? 0 : lagmin, parity ? 0 : lagmax, elapsed);

  // Late runs arrive after the time limit by design, and many runs on several
  // objects arrive on them in turn, the later ones after it: their waiters time
  // out legitimately, so the check needs every arrive within the bound.
  const bool bounded = lagmax + kSlack < kLimit;
  if (!spin && !parity && mode != MODE_SLEEP && bounded && timeouts != 0) {
    std::printf("*** %lu time-outs, though every arrive was answered within %u cycles of the meeting:"
                " a wake-up was lost\n", (unsigned long)timeouts, lagmax);
    ++errors;
  }
  if (!spin && (mode == MODE_MIXED || (mode == MODE_MANY && objs == 1)) && !bounded) {
    std::printf("*** an arrive was answered %u cycles after the meeting, not %u cycles within the time limit:"
                " no lost wake-up can be told from a time-out\n", lagmax, kSlack);
    ++errors;
  }
  if (!spin && mode == MODE_LATE && !parity && lagmin >= 2 * kLimit && few != 0) {
    std::printf("*** %u waiter threads timed out fewer than %u times, though every arrive came %u cycles or more"
                " after the meeting\n", few, rounds, lagmin);
    ++errors;
  }
  if (mode == MODE_SLEEP) {
    const uint32_t* ep = &err[(size_t)w0 * ERR_WORDS];
    if (ep[ERR_TIMEOUTS] != rounds || ep[ERR_ELAPSED] < (uint64_t)rounds * kLimit) {
      std::printf("*** the producer timed out %u times in %u cycles, expected %u in at least %lu\n",
                  ep[ERR_TIMEOUTS], ep[ERR_ELAPSED], rounds, (unsigned long)rounds * kLimit);
      ++errors;
    }
    if (min_waiter_timeouts == ~0u || min_waiter_timeouts < rounds / 2) {
      std::printf("*** a waiter timed out only %u times while the producer slept %u time limits\n",
                  min_waiter_timeouts, rounds);
      ++errors;
    }
  }
  if (!spin && !parity && mode != MODE_SLEEP && slept == 0) {
    std::printf("*** no try_wait suspended its warp: every arrive came before the waiters read the phase\n");
    ++errors;
  }
  if (mode == MODE_MIXED && mixed == 0) {
    std::printf("*** no first poll split a warp: the arrive never landed among a warp's lanes\n");
    ++errors;
  }

  // The final object words.
  std::vector<uint64_t> want(NUM_OBJS, mr::init(1));
  if (mode == MODE_MANY) {
    for (uint32_t j = 0; j < objs; ++j)
      want[OBJ_MANY + j] = after_phases(rounds);
  } else {
    want[OBJ_BAR] = after_phases(mode == MODE_SLEEP ? 1 : rounds);
  }
  for (uint32_t i = 0; i < NUM_OBJS; ++i) {
    if (words[i] == want[i])
      continue;
    std::printf("*** object %u:", i);
    print_word(" final", words[i]);
    print_word(", mbar_ref", want[i]);
    std::printf("\n");
    ++errors;
  }

  vx_kernel_release(kernel);
  vx_module_release(mod);
  vx_buffer_release(word_buf);
  vx_buffer_release(err_buf);
  if (src)
    vx_buffer_release(src);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
