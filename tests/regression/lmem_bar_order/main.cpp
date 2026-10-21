// lmem_bar_order: a local-memory store must reach its bank before the barrier
// or fence that orders it releases a reader, while DXA copies keep the banks
// busy. One block fills the core: warps [0, S) stream DXA tile copies, warp S
// produces X each round, the remaining warps read it back (see kernel.cpp).
//
//   -m bar       the group barrier orders the store (default)
//   -m fence     vx_fence() orders the store before a flag store the consumers poll
//   -m fencebar  vx_fence() and then the group barrier, with no polling
//   -m rl        amoswap.d.rl of a flag the consumers poll (needs the A extension)
//   -o st        the producer stores the round into X (default)
//   -o amo       the producer swaps the round into X with amoswap.d (needs A)
//   -o ld        load buffering: the producer loads X before its ordering point
//                and must see the previous round, which the consumers overwrite
//                with the round once released
//   -r R         rounds (default 200)
//   -b N         stores per lane and round into X's bank, from every producer and
//                consumer lane, the producer's last one carrying X (default 8, >= 1)
//   -d 0|1       DXA streamers on (default 1); -d 0 in bar or fencebar mode has
//                no spin loop, so both models retire the same instruction count
//   -e E         32-bit elements per DXA tile, even (default 768)
//   -s S         streamer warps (default 1); the core needs S + 2 warps
//   -x W         doubleword index of X past the tiles (default 0)
//   -w D         producer idle iterations before its stores (default 0)
//   -v           print every consumer thread with an error and each streamer's copies
//
// -o st with bar, fence or fencebar runs kernel_main; the other combinations
// run order_k, so kernel_main's instruction stream does not depend on them.

#include <vortex2.h>
#include <dxa.h>
#include <VX_types.h>
#include "common.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint32_t kDescSrc  = 0;
constexpr uint32_t kSrcTiles = 3;
// Barrier ids 0 (__syncthreads) and 2 (the group) are taken; streamer w uses 3 + w.
constexpr uint32_t kMaxStreamers = VX_CFG_NUM_BARRIERS - 3;

const char* kernel_file = "kernel.vxbin";
uint32_t mode       = MODE_BAR;
uint32_t x_access   = ACC_ST;
uint32_t rounds     = 200;
uint32_t burst      = 8;
uint32_t dxa_on     = 1;
uint32_t tile_elems = 768;
uint32_t streamers  = 1;
uint32_t x_word     = 0;
uint32_t delay      = 0;
bool     verbose    = false;

vx_device_h dev     = nullptr;
vx_queue_h  q       = nullptr;
vx_buffer_h src     = nullptr;
vx_buffer_h res_buf = nullptr;
vx_module_h mod     = nullptr;
vx_kernel_h kernel  = nullptr;

void cleanup() {
  if (kernel)  vx_kernel_release(kernel);
  if (mod)     vx_module_release(mod);
  if (res_buf) vx_buffer_release(res_buf);
  if (src)     vx_buffer_release(src);
  if (q)       vx_queue_release(q);
  if (dev)     vx_device_release(dev);
  kernel = nullptr;
  mod = nullptr;
  res_buf = nullptr;
  src = nullptr;
  q = nullptr;
  dev = nullptr;
}

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      cleanup();                                                      \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

void usage() {
  std::printf("Usage: [-m bar|fence|fencebar|rl] [-o st|amo|ld] [-r rounds] [-b burst] [-d 0|1]"
              " [-e tile elements] [-s streamers] [-x X word] [-w delay] [-v] [-h]\n");
}

// A decimal count; anything else, a sign included, is a usage error.
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
  while ((c = getopt(argc, argv, "m:o:r:b:d:e:s:x:w:vh")) != -1) {
    switch (c) {
      case 'm':
        if (!std::strcmp(optarg, "bar")) {
          mode = MODE_BAR;
        } else if (!std::strcmp(optarg, "fence")) {
          mode = MODE_FENCE;
        } else if (!std::strcmp(optarg, "fencebar")) {
          mode = MODE_FENCEBAR;
        } else if (!std::strcmp(optarg, "rl")) {
          mode = MODE_RL;
        } else {
          usage();
          std::exit(-1);
        }
        break;
      case 'o':
        if (!std::strcmp(optarg, "st")) {
          x_access = ACC_ST;
        } else if (!std::strcmp(optarg, "amo")) {
          x_access = ACC_AMO;
        } else if (!std::strcmp(optarg, "ld")) {
          x_access = ACC_LD;
        } else {
          usage();
          std::exit(-1);
        }
        break;
      case 'r': rounds     = parse_count(optarg); break;
      case 'b': burst      = parse_count(optarg); break;
      case 'd': dxa_on     = parse_count(optarg); break;
      case 'e': tile_elems = parse_count(optarg); break;
      case 's': streamers  = parse_count(optarg); break;
      case 'x': x_word     = parse_count(optarg); break;
      case 'w': delay      = parse_count(optarg); break;
      case 'v': verbose    = true;                break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || rounds == 0 || burst == 0 || dxa_on > 1 || tile_elems == 0 || (tile_elems & 1)
      || streamers == 0 || streamers > kMaxStreamers) {
    usage();
    std::exit(-1);
  }
}

const char* mode_name() {
  switch (mode) {
    case MODE_FENCE:    return "fence";
    case MODE_FENCEBAR: return "fencebar";
    case MODE_RL:       return "rl";
    default:            return "bar";
  }
}

const char* access_name() {
  switch (x_access) {
    case ACC_AMO: return "amo";
    case ACC_LD:  return "ld";
    default:      return "st";
  }
}

bool use_order_k() {
  return x_access != ACC_ST || mode == MODE_RL;
}

}  // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
  CHECK(vx_queue_create(dev, &qi, &q));

  uint64_t nw = 0, nt = 0, lmem_cap = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,      &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,    &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,      &isa));

  const uint32_t banks = VX_CFG_LMEM_NUM_BANKS;
  const uint32_t block = (uint32_t)(nw * nt);
  // X, the flag and done, then each lane's burst word on X's bank.
  const uint64_t ctrl_words = (uint64_t)x_word + (uint64_t)banks * (SINK_ROWS(banks) + nt);
  const uint64_t lmem_bytes = (uint64_t)streamers * tile_elems * sizeof(uint32_t) + ctrl_words * sizeof(uint64_t);
  std::printf("lmem_bar_order: %s mode, %u warps of %u threads, %u rounds, burst %u, DXA %s,"
              " %u streamers of %u-byte tiles, X at word %u, delay %u, %u banks\n",
              mode_name(), (uint32_t)nw, (uint32_t)nt, rounds, burst, dxa_on ? "on" : "off",
              streamers, tile_elems * (uint32_t)sizeof(uint32_t), x_word, delay, banks);
  if (use_order_k())
    std::printf("lmem_bar_order: producer access %s, order_k\n", access_name());

  if (!(isa & VX_ISA_EXT_DLMEM) || !(isa & VX_ISA_EXT_DXA)) {
    std::printf("Error: the device needs DLMEM and DXA\nFAILED!\n");
    cleanup();
    return 1;
  }
  if ((x_access == ACC_AMO || mode == MODE_RL) && !(isa & VX_ISA_STD_A)) {
    std::printf("Error: -o amo and -m rl need the A extension\nFAILED!\n");
    cleanup();
    return 1;
  }
  if (nw < streamers + 2) {
    std::printf("Error: %u streamers need at least %u warps\nFAILED!\n", streamers, streamers + 2);
    cleanup();
    return 1;
  }
  if (lmem_bytes > lmem_cap) {
    std::printf("Error: %lu bytes of local memory, the core has %lu\nFAILED!\n",
                (unsigned long)lmem_bytes, (unsigned long)lmem_cap);
    cleanup();
    return 1;
  }

  std::vector<uint32_t> h_src(tile_elems * kSrcTiles);
  for (size_t i = 0; i < h_src.size(); ++i)
    h_src[i] = (uint32_t)(i * 2654435761u);
  uint64_t src_addr = 0;
  CHECK(vx_buffer_create(dev, h_src.size() * sizeof(uint32_t), VX_MEM_READ, &src));
  CHECK(vx_buffer_address(src, &src_addr));
  CHECK(vx_enqueue_write(q, src, 0, h_src.data(), h_src.size() * sizeof(uint32_t), 0, nullptr, nullptr));
  if (vortex::dxa::program_1d(dev, kDescSrc, src_addr, (uint32_t)h_src.size(), tile_elems,
                              sizeof(uint32_t)) != 0) {
    std::printf("Error: cannot program the DXA descriptor\nFAILED!\n");
    cleanup();
    return 1;
  }

  // One buffer: RES_WORDS per thread, then one copy count per streamer warp.
  std::vector<uint32_t> res((size_t)block * RES_WORDS + streamers, 0);
  for (uint32_t t = 0; t < block; ++t)
    res[(size_t)t * RES_WORDS + RES_STALE] = ~0u;
  const uint64_t res_bytes = res.size() * sizeof(uint32_t);
  CHECK(vx_buffer_create(dev, res_bytes, VX_MEM_READ_WRITE, &res_buf));
  CHECK(vx_enqueue_write(q, res_buf, 0, res.data(), res_bytes, 0, nullptr, nullptr));

  kernel_arg_t arg = {};
  CHECK(vx_buffer_address(res_buf, &arg.res_addr));
  arg.mode          = use_order_k() ? (mode | (x_access << ACC_SHIFT)) : mode;
  arg.rounds        = rounds;
  arg.dxa_on        = dxa_on;
  arg.tile_elems    = tile_elems;
  arg.num_src_tiles = kSrcTiles;
  arg.streamers     = streamers;
  arg.x_word        = x_word;
  arg.delay         = delay;
  arg.burst         = burst;
  arg.banks         = banks;

  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, use_order_k() ? "order_k" : "main", &kernel));

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

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, res.data(), res_buf, 0, res_bytes, 1, &launch_ev, &read_ev));
  CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  uint64_t stale = 0, timeouts = 0, copies = 0;
  uint32_t unfinished = 0, threads_hit = 0, first_round = 0;
  // kernel_main records the consumers; order_k the producer too, whose lane 0 checks its loads in ld mode.
  const uint32_t c0 = (streamers + (use_order_k() ? 0 : 1)) * (uint32_t)nt;
  for (uint32_t t = c0; t < block; ++t) {
    const uint32_t* e = &res[(size_t)t * RES_WORDS];
    if (e[RES_STALE] == ~0u) {
      ++unfinished;
      continue;
    }
    stale += e[RES_STALE];
    timeouts += e[RES_TIMEOUT];
    if (e[RES_STALE] || e[RES_TIMEOUT]) {
      ++threads_hit;
      if (e[RES_FIRST] && (!first_round || e[RES_FIRST] < first_round))
        first_round = e[RES_FIRST];
      if (verbose)
        std::printf("  thread %u (warp %u lane %u): %u stale, first in round %u, %u timeouts\n",
                    t, t / (uint32_t)nt, t % (uint32_t)nt, e[RES_STALE], e[RES_FIRST], e[RES_TIMEOUT]);
    }
  }
  for (uint32_t w = 0; w < streamers; ++w) {
    const uint32_t n = res[(size_t)block * RES_WORDS + w];
    copies += n;
    if (verbose)
      std::printf("  streamer warp %u: %u DXA copies\n", w, n);
  }
  const uint64_t reads = (x_access == ACC_LD) ? rounds : (uint64_t)rounds * (block - (streamers + 1) * (uint32_t)nt);
  std::printf("lmem_bar_order: %lu stale of %lu %s reads, %lu flag timeouts, %u threads hit,"
              " first stale round %u, %u unfinished threads, %lu DXA copies\n",
              (unsigned long)stale, (unsigned long)reads, (x_access == ACC_LD) ? "producer" : "consumer",
              (unsigned long)timeouts, threads_hit,
              first_round, unfinished, (unsigned long)copies);

  vx_kernel_release(kernel);
  kernel = nullptr;
  vx_module_release(mod);
  mod = nullptr;
  vx_buffer_release(res_buf);
  res_buf = nullptr;
  vx_buffer_release(src);
  src = nullptr;
  vx_queue_release(q);
  q = nullptr;
  vx_device_dump_perf(dev, stdout);
  cleanup();

  int errors = 0;
  errors += (stale != 0) + (timeouts != 0) + (unfinished != 0);
  if (dxa_on && copies == 0) {
    std::printf("*** the streamers completed no DXA copy\n");
    ++errors;
  }
  if (errors) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
