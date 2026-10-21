// mbar_local: mbarriers in a block's own local memory, on a DLMEM device with
// the A extension and mbarriers (RV64). One block fills the core (see kernel.cpp).
//
//   -m basic  every worker thread arrives with 1 and waits on its token, over
//             many phases; the data stored before each arrive must be visible
//             after the wait (default)
//   -m pc     producer and consumer warps over a double buffer, with FULL and
//             EMPTY barriers per buffer; the consumers check every element
//   -m tx     a scripted op sequence whose every returned word is checked,
//             then expect/complete/arrive_tx by every worker warp per phase,
//             with completes landing before the expect on odd phases
//   -m count  lane 0 of each worker warp arrives for the whole warp
//   -m glob   pc over a global double buffer: the producers' global stores
//             must be visible to the consumers after the wait (no streamers)
//   -b B      glob: stores per element and fill, the last one with the tag
//             (default 4)
//   -r R      phases, or buffer fills in pc (default 32)
//   -D S      S warps stream DXA copies into their own tiles meanwhile (default 0)
//   -e E      32-bit elements per DXA tile, even (default 512)
//   -w D      idle iterations that skew the warps (default 64)
//   -p        parity: no spin, so the instruction count does not depend on
//             timing (the workers meet in a register barrier where they would
//             wait, and the phase must then be complete); no streamers
//   -x neg    a run that must stop the simulator: underflow (arrive past
//             pending), txrange (tx past its range), global (an arrive on a
//             global address), remote (an arrive on another block's object;
//             a SPREAD cluster of 2, so 2 cores)
//   -v        print every thread with an error
//
// The host replays each object's ops with mbar_ref and checks the final words,
// the kernel's per-thread checks, and in tx mode every returned word.

#include <vortex2.h>
#include <dxa.h>
#include <VX_types.h>
#include <mbar_ref.h>
#include "common.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace mr = vortex::mbar_ref;

constexpr uint32_t kSrcTiles = 3;
// Barrier ids 0 (__syncthreads) and 1 (the workers) are taken; streamer w uses 2 + w.
constexpr uint32_t kMaxStreamers = VX_CFG_NUM_BARRIERS - 2;

const char* kernel_file = "kernel.vxbin";
uint32_t mode       = MODE_BASIC;
uint32_t neg        = NEG_NONE;
uint32_t rounds     = 32;
uint32_t streamers  = 0;
uint32_t tile_elems = 512;
uint32_t delay      = 64;
uint32_t burst      = 4;
bool     verbose    = false;
bool     parity     = false;

const char* const kModeNames[] = {"basic", "pc", "tx", "count", "glob"};
const char* const kNegNames[]  = {"", "underflow", "txrange", "global", "remote"};

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

void usage() {
  std::printf("Usage: [-m basic|pc|tx|count|glob] [-r rounds] [-D streamers] [-e tile elements] [-w delay] [-b burst] [-p]"
              " [-x underflow|txrange|global|remote] [-v] [-h]\n");
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

uint32_t parse_name(const char* s, const char* const* names, uint32_t n, uint32_t first) {
  for (uint32_t i = first; i < n; ++i) {
    if (!std::strcmp(s, names[i]))
      return i;
  }
  usage();
  std::exit(-1);
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "m:r:D:e:w:b:x:pvh")) != -1) {
    switch (c) {
      case 'm': mode       = parse_name(optarg, kModeNames, 5, 0); break;
      case 'x': neg        = parse_name(optarg, kNegNames, 5, 1);  break;
      case 'r': rounds     = parse_count(optarg); break;
      case 'D': streamers  = parse_count(optarg); break;
      case 'e': tile_elems = parse_count(optarg); break;
      case 'w': delay      = parse_count(optarg); break;
      case 'b': burst      = parse_count(optarg); break;
      case 'v': verbose    = true;                break;
      case 'p': parity     = true;                break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || rounds == 0 || tile_elems == 0 || (tile_elems & 1)
      || streamers > kMaxStreamers || (neg && (streamers || parity)) || (parity && streamers)
      || burst == 0 || (mode == MODE_GLOB && streamers)) {
    usage();
    std::exit(-1);
  }
}

typedef std::vector<std::pair<uint32_t, uint64_t>> Ops;

// Applies `phases` repetitions of `ops`, which must complete the phase with
// their last op and only then; false on any error or misplaced flip.
bool replay_phases(uint64_t& w, uint32_t phases, const Ops& ops) {
  for (uint32_t p = 0; p < phases; ++p) {
    for (size_t i = 0; i < ops.size(); ++i) {
      const mr::Result r = mr::apply(ops[i].first, w, ops[i].second);
      if (r.err || r.flip != (i + 1 == ops.size()))
        return false;
      w = r.new_word;
    }
  }
  return true;
}

void add(Ops& ops, uint32_t op, uint64_t v, uint32_t times = 1) {
  for (uint32_t i = 0; i < times; ++i)
    ops.push_back({op, v});
}

// Exercises every op and edge: a complete before its expect (negative tx), a
// flip by expect_tx, tx at both ends of its range, arrive_tx of 0, the largest
// count, and tokens of both phases.
Ops make_script() {
  const uint64_t tmax = (uint64_t)mr::TX_MAX;
  const uint64_t ph0 = 0, ph1 = uint64_t(1) << 63;
  Ops s;
  add(s, SCRIPT_STORE, mr::init(2));
  add(s, mr::OP_COMPLETE_TX, 64);
  add(s, mr::OP_ARRIVE, 1);
  add(s, mr::OP_TEST_WAIT, ph0);
  add(s, mr::OP_ARRIVE, 1);
  add(s, mr::OP_TEST_WAIT, ph0);
  add(s, mr::OP_EXPECT_TX, 64);
  add(s, mr::OP_TEST_WAIT, ph0);
  add(s, mr::OP_TEST_WAIT, ph1);
  add(s, mr::OP_EXPECT_TX, 100);
  add(s, mr::OP_ARRIVE_TX, 28);
  add(s, mr::OP_COMPLETE_TX, 100);
  add(s, mr::OP_ARRIVE, 1);
  add(s, mr::OP_COMPLETE_TX, 28);
  add(s, mr::OP_ARRIVE, 2);
  add(s, mr::OP_ARRIVE_TX, 0);
  add(s, mr::OP_ARRIVE_TX, 0);
  add(s, mr::OP_EXPECT_TX, tmax);
  add(s, mr::OP_COMPLETE_TX, tmax);
  add(s, mr::OP_COMPLETE_TX, tmax);
  add(s, mr::OP_ARRIVE, 2);
  add(s, mr::OP_EXPECT_TX, tmax);
  add(s, mr::OP_TEST_WAIT, ph0);
  add(s, SCRIPT_STORE, 0);
  add(s, SCRIPT_STORE, mr::init((uint32_t)mr::COUNT_MAX));
  add(s, mr::OP_ARRIVE, mr::COUNT_MAX - 1);
  add(s, mr::OP_ARRIVE_TX, 5);
  add(s, mr::OP_TEST_WAIT, ph0);
  add(s, mr::OP_COMPLETE_TX, 5);
  add(s, mr::OP_TEST_WAIT, ph0);
  add(s, mr::OP_ARRIVE, mr::COUNT_MAX);
  add(s, mr::OP_TEST_WAIT, ph1);
  add(s, SCRIPT_STORE, mr::init(3));
  add(s, mr::OP_ARRIVE, 1, 3);
  add(s, mr::OP_TEST_WAIT, ph1);
  return s;
}

// Expected returned words of the script, and the object's final word.
bool replay_script(const Ops& s, std::vector<uint64_t>& ret, uint64_t& w, uint32_t& flips) {
  w = 0;
  flips = 0;
  for (const auto& op : s) {
    if (op.first == SCRIPT_STORE) {
      w = op.second;
      ret.push_back(0);
      continue;
    }
    const mr::Result r = mr::apply(op.first, w, op.second);
    if (r.err)
      return false;
    ret.push_back(r.ret_word);
    w = r.new_word;
    flips += r.flip;
  }
  return true;
}

void print_word(const char* what, uint64_t w) {
  std::printf("%s %#018lx (phase %u, pending %u, expected %u, tx %ld)", what, (unsigned long)w,
              mr::phase(w), mr::pending(w), mr::expected(w), (long)mr::tx(w));
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

  uint64_t nw = 0, nt = 0, cores = 0, max_cluster = 0, lmem_cap = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,         &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,         &cores));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE,    &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &isa));

  const uint32_t block   = (uint32_t)(nw * nt);
  const uint32_t workers = (uint32_t)nw - streamers;
  const uint32_t grid    = (neg == NEG_REMOTE) ? 2 : 1;
  const uint32_t tile_word = LM_DATA + DATA_WORDS(block);
  const uint64_t lmem_bytes = (uint64_t)tile_word * sizeof(uint64_t)
                            + (uint64_t)streamers * tile_elems * sizeof(uint32_t);
  if (neg) {
    std::printf("mbar_local: negative run '%s', %u warps of %u threads\n",
                kNegNames[neg], (uint32_t)nw, (uint32_t)nt);
  } else {
    std::printf("mbar_local: %s mode%s, %u warps of %u threads, %u rounds, %u DXA streamers, delay %u\n",
                kModeNames[mode], parity ? " (parity)" : "", (uint32_t)nw, (uint32_t)nt, rounds, streamers, delay);
  }

  if (!(isa & VX_ISA_EXT_DLMEM) || !(isa & VX_ISA_STD_A) || (streamers && !(isa & VX_ISA_EXT_DXA))) {
    std::printf("Error: the device needs DLMEM and the A extension%s\nFAILED!\n", streamers ? ", and DXA" : "");
    return 1;
  }
  const bool two_sided = (mode == MODE_PC || mode == MODE_GLOB);
  if (nw <= streamers || (two_sided && workers < 2)) {
    std::printf("Error: %u warps leave %d workers, %s needs %u\nFAILED!\n", (uint32_t)nw,
                (int)nw - (int)streamers, kModeNames[mode], two_sided ? 2 : 1);
    return 1;
  }
  if (lmem_bytes > lmem_cap) {
    std::printf("Error: %lu bytes of local memory, the core has %lu\nFAILED!\n",
                (unsigned long)lmem_bytes, (unsigned long)lmem_cap);
    return 1;
  }
  if (neg == NEG_REMOTE && (cores < 2 || max_cluster < 2)) {
    std::printf("Error: a remote arrive needs a cluster of 2 on 2 cores (%u cores, largest cluster %u)\nFAILED!\n",
                (uint32_t)cores, (uint32_t)max_cluster);
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

  const Ops script = (mode == MODE_TX && !neg) ? make_script() : Ops();
  if (script.size() > MAX_SCRIPT) {
    std::printf("Error: a script of %zu ops, at most %u fit\nFAILED!\n", script.size(), MAX_SCRIPT);
    return 1;
  }
  std::vector<uint64_t> h_script(2 * script.size() + 2, 0);
  for (size_t i = 0; i < script.size(); ++i) {
    h_script[2 * i]     = script[i].first;
    h_script[2 * i + 1] = script[i].second;
  }
  const uint64_t script_bytes = h_script.size() * sizeof(uint64_t);
  vx_buffer_h script_buf = nullptr;
  CHECK(vx_buffer_create(dev, script_bytes, VX_MEM_READ, &script_buf));
  CHECK(vx_enqueue_write(q, script_buf, 0, h_script.data(), script_bytes, 0, nullptr, nullptr));

  const uint32_t threads = grid * block;
  std::vector<uint32_t> err((size_t)threads * ERR_WORDS, 0);
  for (uint32_t t = 0; t < threads; ++t)
    err[(size_t)t * ERR_WORDS + ERR_COUNT] = ~0u;
  const uint64_t err_bytes = err.size() * sizeof(uint32_t);
  vx_buffer_h err_buf = nullptr;
  CHECK(vx_buffer_create(dev, err_bytes, VX_MEM_READ_WRITE, &err_buf));
  CHECK(vx_enqueue_write(q, err_buf, 0, err.data(), err_bytes, 0, nullptr, nullptr));

  std::vector<uint64_t> words(NUM_OBJS + MAX_SCRIPT, ~uint64_t(0));
  const uint64_t word_bytes = words.size() * sizeof(uint64_t);
  vx_buffer_h word_buf = nullptr;
  CHECK(vx_buffer_create(dev, word_bytes, VX_MEM_READ_WRITE, &word_buf));
  CHECK(vx_enqueue_write(q, word_buf, 0, words.data(), word_bytes, 0, nullptr, nullptr));

  std::vector<uint64_t> h_glob;
  vx_buffer_h glob_buf = nullptr;
  if (mode == MODE_GLOB && !neg) {
    h_glob.assign(2 * PC_ELEMS, 0);
    const uint64_t glob_bytes = h_glob.size() * sizeof(uint64_t);
    CHECK(vx_buffer_create(dev, glob_bytes, VX_MEM_READ_WRITE, &glob_buf));
    CHECK(vx_enqueue_write(q, glob_buf, 0, h_glob.data(), glob_bytes, 0, nullptr, nullptr));
  }

  mbar_arg_t arg = {};
  CHECK(vx_buffer_address(err_buf,    &arg.err_addr));
  CHECK(vx_buffer_address(word_buf,   &arg.word_addr));
  CHECK(vx_buffer_address(script_buf, &arg.script_addr));
  if (glob_buf)
    CHECK(vx_buffer_address(glob_buf, &arg.glob_addr));
  arg.mode          = mode;
  arg.neg           = neg;
  arg.rounds        = rounds;
  arg.streamers     = streamers;
  arg.tile_word     = tile_word;
  arg.tile_elems    = tile_elems;
  arg.num_src_tiles = kSrcTiles;
  arg.script_len    = (uint32_t)script.size();
  arg.delay         = delay;
  arg.parity        = parity;
  arg.burst         = burst;

  const char* const kKernels[] = {"basic_k", "pc_k", "tx_k", "count_k", "glob_k"};
  vx_module_h mod = nullptr;
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, neg ? "neg_k" : kKernels[mode], &kernel));

  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = grid;
  li.grid_dim[1]  = 1;
  li.grid_dim[2]  = 1;
  li.block_dim[0] = block;
  li.block_dim[1] = 1;
  li.block_dim[2] = 1;
  li.lmem_size    = (uint32_t)lmem_bytes;
  if (neg == NEG_REMOTE) {
    li.cluster_dim[0] = 2;
    li.cluster_dim[1] = 1;
    li.cluster_dim[2] = 1;
    li.cluster_policy = VX_CLUSTER_POLICY_SPREAD;
  }

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
  if (neg) {
    std::printf("*** the negative run '%s' completed: the simulator must stop it\n", kNegNames[neg]);
    ++errors;
  } else {
    // Per-thread checks of the workers.
    const uint32_t w0 = streamers * (uint32_t)nt;
    uint32_t failed = 0, unfinished = 0;
    uint64_t checks = 0, negs = 0;
    for (uint32_t t = 0; t < block; ++t) {
      const uint32_t* e = &err[(size_t)t * ERR_WORDS];
      if (e[ERR_COUNT] == ~0u) {
        ++unfinished;
        continue;
      }
      negs += e[ERR_NEG];
      if (t < w0 || e[ERR_COUNT] == 0)
        continue;
      checks += e[ERR_COUNT];
      ++failed;
      if (verbose || failed == 1)
        std::printf("*** thread %u (warp %u lane %u): %u failed checks, first in round %u, check %u\n",
                    t, t / (uint32_t)nt, t % (uint32_t)nt, e[ERR_COUNT], e[ERR_FIRST] - 1, e[ERR_WHAT]);
    }
    std::printf("mbar_local: %u threads with %lu failed checks, %u unfinished threads, %lu negative-tx words seen\n",
                failed, (unsigned long)checks, unfinished, (unsigned long)negs);
    errors += (failed != 0) + (unfinished != 0);

    // The final object words.
    std::vector<uint64_t> want(NUM_OBJS, 0);
    const uint32_t nww = workers;
    const uint32_t nwt = nww * (uint32_t)nt;
    bool model_ok = true;
    if (mode == MODE_BASIC) {
      Ops ops;
      add(ops, mr::OP_ARRIVE, 1, nwt);
      want[OBJ_BAR] = mr::init(nwt);
      model_ok &= replay_phases(want[OBJ_BAR], rounds, ops);
    } else if (mode == MODE_COUNT) {
      Ops ops;
      for (uint32_t w = 0; w < nww; ++w) {
        if ((w & 1) && nt >= 2) {
          add(ops, mr::OP_ARRIVE, nt / 2);
          add(ops, mr::OP_ARRIVE, nt - nt / 2);
        } else {
          add(ops, mr::OP_ARRIVE, nt);
        }
      }
      want[OBJ_BAR] = mr::init(nwt);
      model_ok &= replay_phases(want[OBJ_BAR], rounds, ops);
    } else if (two_sided) {
      const uint32_t P = nww / 2, C = nww - P;
      for (uint32_t s = 0; s < 2; ++s) {
        const uint32_t fills = (rounds + 1 - s) / 2;
        Ops full, empty;
        add(full, mr::OP_ARRIVE, 1, P * (uint32_t)nt);
        add(empty, mr::OP_ARRIVE, 1, C);
        want[OBJ_FULL + s] = mr::init(P * (uint32_t)nt);
        want[OBJ_EMPTY + s] = mr::init(C);
        model_ok &= replay_phases(want[OBJ_FULL + s], fills, full);
        model_ok &= replay_phases(want[OBJ_EMPTY + s], fills, empty);
      }
    } else {
      const uint32_t total = TX_CHUNK * (nww * (nww - 1) / 2);
      Ops ops;
      for (uint32_t w = 1; w < nww; ++w) {
        add(ops, mr::OP_COMPLETE_TX, TX_CHUNK * w);
        add(ops, mr::OP_ARRIVE, 1);
      }
      add(ops, mr::OP_ARRIVE_TX, total);
      want[OBJ_BAR] = mr::init(nww);
      model_ok &= replay_phases(want[OBJ_BAR], rounds, ops);

      std::vector<uint64_t> ret;
      uint32_t flips = 0;
      model_ok &= replay_script(script, ret, want[OBJ_SCRIPT], flips);
      uint32_t bad = 0;
      for (size_t i = 0; i < ret.size(); ++i) {
        if (words[NUM_OBJS + i] == ret[i])
          continue;
        if (bad++ < 4 || verbose) {
          std::printf("*** script op %zu (op %u, operand %#lx):", i, script[i].first,
                      (unsigned long)script[i].second);
          print_word(" returned", words[NUM_OBJS + i]);
          print_word(", mbar_ref", ret[i]);
          std::printf("\n");
        }
      }
      std::printf("mbar_local: script of %zu ops, %u phase flips, %u wrong returned words\n",
                  script.size(), flips, bad);
      errors += (bad != 0);
      if (nww >= 2 && rounds >= 2 && delay > 0 && negs == 0) {
        std::printf("*** no returned word showed a negative tx: the completes never came first\n");
        ++errors;
      }
    }
    if (!model_ok) {
      std::printf("*** mbar_ref rejects the expected op sequence (app bug)\n");
      ++errors;
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
    std::printf("mbar_local: final object words");
    for (uint32_t i = 0; i < NUM_OBJS; ++i)
      std::printf(" %#lx", (unsigned long)words[i]);
    std::printf("\n");
  }

  vx_kernel_release(kernel);
  vx_module_release(mod);
  vx_buffer_release(word_buf);
  vx_buffer_release(err_buf);
  vx_buffer_release(script_buf);
  if (glob_buf)
    vx_buffer_release(glob_buf);
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
