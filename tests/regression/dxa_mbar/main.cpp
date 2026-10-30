// dxa_mbar: DXA copies into local memory that complete on an mbarrier, on a
// DLMEM device with the A extension, mbarriers and the DXA (RV64). One block
// fills the core (see kernel.cpp).
//
//   (default) per round, warp 0 arrives on the mbarrier declaring the box
//             bytes and issues the copy; warps 0 and 1 wait with try_wait,
//             which the copy's complete_tx must wake; then the tile's last
//             word must already hold the copied value (spins=0) while the
//             other warps load, store and add atomically on its bank
//   -l        the same with the copy completing on a barrier event
//   -q        QUEUE_COPIES copies a round, issued back to back by both
//             waiter warps, each into the next tile and completing on the
//             one mbarrier (no wake-up is required), while the
//             other warps hammer the mbarrier's bank instead: the complete_tx
//             requests lose the bank's round-robin, so with several DXA
//             workers the core's completion queue fills and holds the next
//             copy's last beat back
//   -p        parity: each contention thread runs a fixed number of
//             iterations, so the instruction count does not depend on timing
//             as long as no try_wait times out
//   -r R      rounds (default 8)
//   -e E      32-bit elements per copy (default 256)
//   -x neg    a run that must stop: expect (the arrive declares 16 bytes more
//             than the box, so the phase never completes and the app gives up
//             with "mbarrier phase did not complete"), retired (the block
//             exits before its copy lands), bounds (the mbarrier lies past
//             the block's local memory), mcast (a multicast mask)
//   -v        print every thread with an error
//
// The host checks the kernel's per-thread checks, that every waiter warp was
// woken by a completion at least once (mbarrier mode), that the contention
// warps ran, and the final mbarrier word against mbar_ref.

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
#include <vector>

namespace {

namespace mr = vortex::mbar_ref;

constexpr uint32_t kSrcTiles = 3;
constexpr uint32_t kExtraTx  = 16;
constexpr uint32_t kLinger   = 20000;
constexpr uint32_t kParityIters = 64;

const char* kernel_file = "kernel.vxbin";
uint32_t rounds     = 8;
uint32_t tile_elems = 256;
uint32_t copies     = 1;
uint32_t neg        = NEG_NONE;
bool     legacy     = false;
bool     parity     = false;
bool     verbose    = false;

const char* const kNegNames[] = {"", "expect", "retired", "bounds", "mcast"};
const char* const kChecks[]   = {"", "wait", "data", "probe"};

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

void usage() {
  std::printf("Usage: [-l] [-p] [-q] [-r rounds] [-e tile elements] [-x expect|retired|bounds|mcast] [-v] [-h]\n");
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
  while ((c = getopt(argc, argv, "lpqr:e:x:vh")) != -1) {
    switch (c) {
      case 'l': legacy     = true;                break;
      case 'p': parity     = true;                break;
      case 'q': copies     = QUEUE_COPIES;        break;
      case 'r': rounds     = parse_count(optarg); break;
      case 'e': tile_elems = parse_count(optarg); break;
      case 'v': verbose    = true;                break;
      case 'x': {
        neg = NEG_NONE;
        for (uint32_t i = 1; i < 5; ++i) {
          if (!std::strcmp(optarg, kNegNames[i]))
            neg = i;
        }
        if (neg == NEG_NONE) {
          usage();
          std::exit(-1);
        }
        break;
      }
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || rounds == 0 || tile_elems == 0 || tile_elems > 0xFFFF || (neg && (legacy || parity))
      || (copies > 1 && (legacy || neg))) {
    usage();
    std::exit(-1);
  }
}

} // namespace

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

  if (!(isa & VX_ISA_EXT_DLMEM) || !(isa & VX_ISA_STD_A) || !(isa & VX_ISA_EXT_DXA)) {
    std::printf("Error: the device needs DLMEM, the A extension and the DXA\nFAILED!\n");
    return 1;
  }
  if (nw < NUM_WAITERS) {
    std::printf("Error: %u warps, the test needs at least %u\nFAILED!\n", (uint32_t)nw, NUM_WAITERS);
    return 1;
  }
  if (neg) {
    rounds = 1;
  }

  const uint32_t nb        = VX_CFG_LMEM_NUM_BANKS;
  const uint32_t row       = nb * sizeof(uint64_t);
  const uint32_t block     = (uint32_t)(nw * nt);
  const uint32_t contend   = block - NUM_WAITERS * (uint32_t)nt;
  const uint32_t box       = tile_elems * sizeof(uint32_t);
  const uint32_t probe_off = LM_TILE + copies * box - sizeof(uint32_t);
  const uint32_t mbank     = (LM_MBAR / sizeof(uint64_t)) % nb;
  const uint32_t pbank     = (copies > 1) ? mbank : (probe_off / sizeof(uint64_t)) % nb;
  const uint32_t scr_off   = (LM_TILE + copies * box + row - 1) / row * row;
  uint64_t lmem_bytes      = (uint64_t)scr_off + (uint64_t)(contend + 1) * row;
  // retired: the block takes all of local memory, so the next one reuses its slot.
  if (neg == NEG_RETIRED)
    lmem_bytes = lmem_cap;
  const uint32_t stride = (uint32_t)((lmem_bytes + VX_CFG_MEM_BLOCK_SIZE - 1) / VX_CFG_MEM_BLOCK_SIZE
                                     * VX_CFG_MEM_BLOCK_SIZE);
  if (lmem_bytes > lmem_cap || (neg == NEG_BOUNDS && (uint64_t)stride * 2 > lmem_cap)) {
    std::printf("Error: %lu bytes of local memory, the core has %lu\nFAILED!\n",
                (unsigned long)lmem_bytes, (unsigned long)lmem_cap);
    return 1;
  }

  if (neg) {
    std::printf("dxa_mbar: negative run '%s', %u warps of %u threads, %u-byte box\n",
                kNegNames[neg], (uint32_t)nw, (uint32_t)nt, box);
  } else {
    std::printf("dxa_mbar: %s completion%s, %u warps of %u threads, %u rounds of %u copies, %u-byte box,"
                " contention on bank %u of %u (mbarrier bank %u), %u contention threads\n",
                legacy ? "barrier" : "mbarrier", parity ? " (parity)" : "", (uint32_t)nw, (uint32_t)nt, rounds,
                copies, box, pbank, nb, mbank, contend);
  }

  // Every host buffer stays alive until the queue has run its upload.
  std::vector<uint32_t> h_src((size_t)tile_elems * kSrcTiles);
  for (uint32_t i = 0; i < h_src.size(); ++i)
    h_src[i] = src_value(i);
  const uint64_t src_bytes = h_src.size() * sizeof(uint32_t);
  vx_buffer_h src = nullptr;
  uint64_t src_addr = 0;
  CHECK(vx_buffer_create(dev, src_bytes, VX_MEM_READ, &src));
  CHECK(vx_buffer_address(src, &src_addr));
  CHECK(vx_enqueue_write(q, src, 0, h_src.data(), src_bytes, 0, nullptr, nullptr));
  if (vortex::dxa::program_1d(dev, 0, src_addr, (uint32_t)h_src.size(), tile_elems, sizeof(uint32_t)) != 0
      || (neg == NEG_MCAST && vortex::dxa::set_multicast(dev, 0, stride) != 0)) {
    std::printf("Error: cannot program the DXA descriptor\nFAILED!\n");
    return 1;
  }

  const uint32_t grid = (neg == NEG_RETIRED) ? 2 : 1;
  const uint32_t threads = grid * block;
  std::vector<uint32_t> out((size_t)threads * OUT_WORDS, 0);
  const uint64_t out_bytes = out.size() * sizeof(uint32_t);
  vx_buffer_h out_buf = nullptr;
  CHECK(vx_buffer_create(dev, out_bytes, VX_MEM_READ_WRITE, &out_buf));
  CHECK(vx_enqueue_write(q, out_buf, 0, out.data(), out_bytes, 0, nullptr, nullptr));

  uint64_t word = ~uint64_t(0);
  vx_buffer_h word_buf = nullptr;
  CHECK(vx_buffer_create(dev, sizeof(word), VX_MEM_READ_WRITE, &word_buf));
  CHECK(vx_enqueue_write(q, word_buf, 0, &word, sizeof(word), 0, nullptr, nullptr));

  kernel_arg_t arg = {};
  CHECK(vx_buffer_address(out_buf,  &arg.out_addr));
  CHECK(vx_buffer_address(word_buf, &arg.word_addr));
  arg.legacy        = legacy;
  arg.neg           = neg;
  arg.rounds        = rounds;
  arg.tile_elems    = tile_elems;
  arg.num_src_tiles = kSrcTiles;
  arg.tx_bytes      = copies * box + (neg == NEG_EXPECT ? kExtraTx : 0);
  arg.scr_off       = scr_off;
  arg.probe_bank    = pbank;
  arg.lmem_stride   = stride;
  arg.linger        = kLinger;
  arg.iters         = parity ? kParityIters : 0;
  arg.copies        = copies;

  vx_module_h mod = nullptr;
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  const bool probe = (neg == NEG_NONE || neg == NEG_EXPECT);
  CHECK(vx_module_get_kernel(mod, probe ? "probe_k" : "neg_k", &kernel));

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

  vx_event_h launch_ev = nullptr, out_ev = nullptr, word_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, out.data(), out_buf, 0, out_bytes, 1, &launch_ev, &out_ev));
  CHECK(vx_enqueue_read(q, &word, word_buf, 0, sizeof(word), 1, &launch_ev, &word_ev));
  CHECK(vx_event_wait_value(out_ev, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(word_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(word_ev);
  vx_event_release(out_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  if (!probe) {
    std::printf("*** the negative run '%s' completed: the simulator must stop it\n", kNegNames[neg]);
    ++errors;
  } else {
    const uint32_t nwt = NUM_WAITERS * (uint32_t)nt;
    uint32_t failed = 0, unfinished = 0, gave_up = 0;
    uint64_t spins = 0, timeouts = 0, iters = 0;
    uint64_t wakes[NUM_WAITERS] = {};
    for (uint32_t t = 0; t < block; ++t) {
      const uint32_t* o = &out[(size_t)t * OUT_WORDS];
      if (o[OUT_DONE] != 1) {
        ++unfinished;
        continue;
      }
      if (t < nwt) {
        spins    += o[OUT_SPINS];
        timeouts += o[OUT_TIMEOUTS];
        wakes[t / nt] += o[OUT_WAKES];
      } else {
        iters += o[OUT_ITERS];
      }
      if (o[OUT_ERRS] == 0)
        continue;
      const uint32_t chk = o[OUT_FIRST] >> 16;
      gave_up += (chk == CHK_WAIT);
      ++failed;
      if (verbose || failed == 1)
        std::printf("*** thread %u (warp %u lane %u): %u failed checks, first in round %u, check %s\n",
                    t, t / (uint32_t)nt, t % (uint32_t)nt, o[OUT_ERRS], o[OUT_FIRST] & 0xFFFF,
                    chk < 4 ? kChecks[chk] : "?");
    }
    std::printf("dxa_mbar: spins=%lu, wakes by completion %lu/%lu (warps 0/1), %lu try_wait time-outs,"
                " %lu contention iterations, %u threads with failed checks, %u unfinished\n",
                (unsigned long)spins, (unsigned long)wakes[0], (unsigned long)wakes[1],
                (unsigned long)timeouts, (unsigned long)iters, failed, unfinished);
    if (gave_up)
      std::printf("*** mbarrier phase did not complete: %u threads gave up after %u try_waits\n",
                  gave_up, TRY_LIMIT);
    errors += (failed != 0) + (unfinished != 0) + (spins != 0);

    if (neg == NEG_EXPECT) {
      if (!gave_up) {
        std::printf("*** the phase completed although the arrive declared %u bytes more than the box\n",
                    kExtraTx);
        ++errors;
      }
    } else {
      if (!legacy && copies == 1 && wakes[1] == 0) {
        std::printf("*** warp 1 was never woken from a try_wait by a copy's completion\n");
        ++errors;
      }
      if (contend && iters == 0) {
        std::printf("*** the contention warps never ran\n");
        ++errors;
      }
      // Per round: the issuer's arrive_tx of every copy's bytes, then each
      // copy's complete_tx, the last of which completes the phase.
      uint64_t want = mr::init(1);
      bool model_ok = true;
      for (uint32_t r = 0; r < rounds && !legacy; ++r) {
        const mr::Result a = mr::apply(mr::OP_ARRIVE_TX, want, copies * box);
        model_ok &= !a.err && !a.flip;
        want = a.new_word;
        for (uint32_t i = 0; i < copies; ++i) {
          const mr::Result c = mr::apply(mr::OP_COMPLETE_TX, want, box);
          model_ok &= !c.err && (c.flip == (i + 1 == copies));
          want = c.new_word;
        }
      }
      if (!model_ok || word != want) {
        std::printf("*** mbarrier word %#018lx, expected %#018lx\n",
                    (unsigned long)word, (unsigned long)want);
        ++errors;
      }
    }
  }

  vx_buffer_release(word_buf);
  vx_buffer_release(out_buf);
  vx_buffer_release(src);
  vx_kernel_release(kernel);
  vx_module_release(mod);
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
