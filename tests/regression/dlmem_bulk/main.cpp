// dlmem_bulk: DXA bulk copies from a block's own local memory into the local
// memory of a block of its thread-block cluster, the issuer's own included,
// completing on an mbarrier in the destination block, on a DLMEM device with
// the A extension, mbarriers and the DXA (RV64). Like CUDA's
// cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes, with
// fence.proxy.async ordering the source's stores before the copy.
//
//   -m ring   block r copies into rank (r + 1) % K through the window (default)
//   -m many   every rank copies into rank 0, back to back, each into its own
//             slots; rank 0's own copies use the window address of itself
//   -m self   every block copies into its own local memory through local
//             pointers
//   -m race   rounds in which warp 0 stores the round's words into its source
//             and copies it into its own local memory at once, while the other
//             warps load and store their own words on every bank. Without -f
//             the run only reports the bytes that hold the previous round's
//             value (stale); with -f a stale byte is an error
//   -f        race: vx_fence_proxy_async() between the stores and the copy
//   -F        the other modes: leave out the vx_fence_proxy_async() that
//             orders the source's fill before the cluster barrier
//   -p spread|default
//             the cluster policy: SPREAD (default) puts the ranks on
//             different cores; DEFAULT keeps the cluster on one core, so
//             every window destination is another slot of the issuer's core
//   -k K      blocks per cluster (default 2)
//   -n G      clusters (default 2)
//   -w W      warps per block (default 2)
//   -c C      copies each block issues, back to back (default 4)
//   -s S      bytes per copy, a multiple of 16; 0 (default) is a mix of 16,
//             48, 208 and 720 bytes over the copies and ranks (race: 256)
//   -o A      source misalignment base, in 16-byte units, 0-3 (default 1)
//   -O A      destination misalignment base, in 16-byte units, 0-3 (default 0)
//   -r R      race rounds (default 8)
//   -g H      race: rank 0 of each cluster is a hog that copies H bytes of its
//             unchanged source every round once the other ranks are ready, and
//             only then lets them store theirs, one word per thread of warp 0
//             (-s 4 x threads), so that, in a DEFAULT cluster with two engine
//             workers, their stores wait behind its copy at the banks
//   -b D      the mbarrier sits D doublewords into the block, 0-6 (default 0),
//             so in local-memory bank D modulo the banks
//   -S        the reference path: the issuing warp copies with its own loads
//             and stores, fences them and applies complete_tx itself, which
//             checks the layout and the host's checker without the DXA
//   -x neg    a run that must stop the simulator: size (not a multiple of
//             16), align (a source 8 bytes off), dalign (a destination 8
//             bytes off), src (a source running past the issuer's block), rank
//             (the destination and its mbarrier lie outside the cluster), mbar
//             (the mbarrier is in the issuer's block, not the destination's),
//             box (a copy running past the destination block), retired (the
//             destination block has exited when the copy reaches it)
//   -v        print every cluster's placement
//
// Copy i of rank r reads 16 * ((i + r + A) % 4) bytes into its source slot
// and writes 16 * ((3i + r + A') % 4) bytes into its destination slot, so the
// defaults give source and destination offsets that differ by an odd
// multiple of 16 bytes. Each receiver arms its mbarrier with arrive_tx of the
// bytes it will receive before the cluster barrier, then waits for the phase.
// The host checks every block's whole local memory byte for byte: each
// source range, each destination range against its sender's source, every
// other byte its canary; and the final mbarrier word against mbar_ref. SPREAD
// runs must have a copy cross cores, DEFAULT runs must keep every copy on one
// core.
//
// A device without the extensions, or that the clusters do not fit, prints
// SKIPPED when the run was launched without options, and fails when any
// option asked for a run.

#include <vortex2.h>
#include <VX_types.h>
#include <mbar_ref.h>
#include "common.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
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

const char* const kModeNames[] = {"ring", "many", "self", "race"};
const char* const kNegNames[]  = {"", "size", "align", "dalign", "src", "rank", "mbar", "box", "retired"};

constexpr uint32_t kLinger   = 20000;
constexpr uint32_t kRaceSize = 256;
constexpr uint32_t kNegSize  = 64;

const char* kernel_file = "kernel.vxbin";
uint32_t mode    = MODE_RING;
uint32_t neg     = NEG_NONE;
uint32_t req_k   = 2;
uint32_t req_g   = 2;
uint32_t req_w   = 2;
uint32_t copies  = 4;
uint32_t size    = 0;
uint32_t salign  = 1;
uint32_t dalign  = 0;
uint32_t rounds  = 8;
uint32_t mbar_dw = 0;
uint32_t hog     = 0;
bool     race_fence = false;
bool     no_fence   = false;
bool     soft       = false;
bool     spread  = true;
bool     verbose = false;
bool     requested = false;

void usage() {
  std::printf("Usage: [-m ring|many|self|race] [-f] [-F] [-p spread|default] [-k K] [-n clusters] [-w warps]"
              " [-c copies] [-s bytes] [-o src align] [-O dst align] [-r rounds] [-g hog bytes] [-b mbar doubleword] [-S]"
              " [-x size|align|dalign|src|rank|mbar|box|retired] [-v] [-h]\n");
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

uint32_t parse_name(const char* s, const char* const* names, uint32_t first, uint32_t n) {
  for (uint32_t i = first; i < n; ++i) {
    if (!std::strcmp(s, names[i]))
      return i;
  }
  usage();
  std::exit(-1);
}

void parse_args(int argc, char** argv) {
  requested = (argc > 1);
  int c;
  while ((c = getopt(argc, argv, "m:fFp:k:n:w:c:s:o:O:r:g:b:Sx:vh")) != -1) {
    switch (c) {
      case 'm': mode = parse_name(optarg, kModeNames, 0, NUM_MODES); break;
      case 'f': race_fence = true; break;
      case 'F': no_fence = true; break;
      case 'p': {
        const char* const kPolicies[] = {"spread", "default"};
        spread = parse_name(optarg, kPolicies, 0, 2) == 0;
        break;
      }
      case 'k': req_k  = parse_count(optarg); break;
      case 'n': req_g  = parse_count(optarg); break;
      case 'w': req_w  = parse_count(optarg); break;
      case 'c': copies = parse_count(optarg); break;
      case 's': size   = parse_count(optarg); break;
      case 'o': salign = parse_count(optarg); break;
      case 'O': dalign = parse_count(optarg); break;
      case 'r': rounds = parse_count(optarg); break;
      case 'b': mbar_dw = parse_count(optarg); break;
      case 'g': hog    = parse_count(optarg); break;
      case 'S': soft   = true; break;
      case 'x': neg    = parse_name(optarg, kNegNames, 1, NUM_NEGS); break;
      case 'v': verbose = true; break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  const bool local = (mode == MODE_SELF || mode == MODE_RACE);
  if (optind < argc || req_k < 1 || req_k > MAX_CLUSTER || req_g == 0 || req_w == 0 || copies == 0
      || copies > MAX_COPIES || size % 16 != 0 || salign > 3 || dalign > 3 || rounds == 0
      || rounds > MAX_ROUNDS || mbar_dw > MAX_MBAR_DW || (!local && req_k < 2) || (race_fence && mode != MODE_RACE)
      || (no_fence && mode == MODE_RACE) || (soft && neg)
      || (hog && (mode != MODE_RACE || hog % 16 != 0 || req_k < 2 || soft))) {
    usage();
    std::exit(-1);
  }
}

// A racer of a race with a hog stores one word per thread of warp 0.
void check_hog_size(uint32_t nt) {
  if (hog && size != 4 * nt) {
    usage();
    std::exit(-1);
  }
}

struct Caps {
  uint64_t num_cores = 0, num_warps = 0, num_threads = 0, lmem_size = 0, max_cluster = 0, isa = 0;
};

uint32_t clog2(uint64_t x) {
  uint32_t n = 0;
  while ((uint64_t(1) << n) < x)
    ++n;
  return n;
}

uint32_t round_up(uint32_t x, uint32_t a) {
  return (x + a - 1) / a * a;
}

template <typename T>
vx_buffer_h upload(vx_device_h dev, vx_queue_h q, const std::vector<T>& v, uint64_t* addr, uint32_t flags) {
  vx_buffer_h buf = nullptr;
  const uint64_t bytes = v.size() * sizeof(T);
  CHECK(vx_buffer_create(dev, bytes, flags, &buf));
  CHECK(vx_enqueue_write(q, buf, 0, v.data(), bytes, 0, nullptr, nullptr));
  CHECK(vx_buffer_address(buf, addr));
  return buf;
}

uint8_t canary_byte(uint32_t b, uint32_t o) { return word_byte(canary_word(b, o / 4), o); }
uint8_t src_byte(uint32_t b, uint32_t o)    { return word_byte(src_word(b, o / 4), o); }
uint8_t race_byte(uint32_t b, uint32_t j, uint32_t o) { return word_byte(race_word(b, j, o / 4), o); }

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

  Caps caps;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,         &caps.num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,         &caps.num_warps));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &caps.num_threads));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE,    &caps.lmem_size));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &caps.max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &caps.isa));

  if (neg) {
    mode = MODE_RING;
    spread = true;
    req_k = 2;
    req_g = 1;
    req_w = 1;
    copies = 1;
    size = kNegSize;
  }
  if (mode == MODE_RACE) {
    copies = 1;
    if (size == 0)
      size = kRaceSize;
  }
  const bool race = (mode == MODE_RACE);
  const uint32_t k = req_k, nt = (uint32_t)caps.num_threads, threads = req_w * nt;
  check_hog_size(nt);

  bulk_arg_t arg = {};
  arg.mode         = mode;
  arg.neg          = neg;
  arg.k            = k;
  arg.copies       = copies;
  arg.size         = size;
  arg.salign       = salign;
  arg.dalign       = dalign;
  arg.rounds       = race ? rounds : 0;
  arg.fence        = race ? race_fence : !no_fence;
  arg.warp_threads = nt;
  arg.soft         = soft;
  arg.pitch        = round_up(std::max(size ? size : MIX_MAX, hog), 64) + 128;
  arg.linger       = kLinger;
  arg.lm_mbar      = 8 * mbar_dw;
  arg.hog          = race ? hog : 0;
  const uint32_t ham_words = race ? threads - nt : 0;
  arg.lmem_bytes   = round_up(lm_ham(&arg) + ham_words * 4, 256);
  arg.neg_off      = arg.lmem_bytes - 32;

  uint64_t total = 0;
  for (uint32_t r = 0; r < k; ++r)
    for (uint32_t i = 0; i < copies; ++i)
      total += copy_size(&arg, r, i);
  if (neg) {
    std::printf("dlmem_bulk: negative run '%s', SPREAD cluster of 2 blocks, %u cores of %u warps x %u threads\n",
                kNegNames[neg], (uint32_t)caps.num_cores, (uint32_t)caps.num_warps, nt);
  } else if (race) {
    std::printf("dlmem_bulk: race, %s, %s, %u clusters of %u blocks of %u warps x %u threads, %u cores of %u warps,"
                " %u rounds of %u bytes%s, hog of %u bytes, mbarrier at byte %u, %u bytes of local memory"
                " per block\n", race_fence ? "fenced" : "unfenced", spread ? "SPREAD" : "DEFAULT", req_g, k, req_w, nt,
                (uint32_t)caps.num_cores, (uint32_t)caps.num_warps, rounds, size, soft ? " (reference path)" : "",
                hog, arg.lm_mbar, arg.lmem_bytes);
  } else {
    std::printf("dlmem_bulk: %s, %s, %u clusters of %u blocks of %u warps x %u threads, %u cores of %u warps,"
                " %u copies per block (%s, %lu bytes per cluster%s), alignment bases %u/%u, %s, mbarrier at byte %u,"
                " %u bytes of local memory per block\n", kModeNames[mode], spread ? "SPREAD" : "DEFAULT", req_g, k,
                req_w, nt, (uint32_t)caps.num_cores, (uint32_t)caps.num_warps, copies, size ? "fixed size" : "size mix",
                (unsigned long)total, soft ? ", reference path" : "", salign, dalign,
                no_fence ? "no proxy fence" : "proxy fence", arg.lm_mbar, arg.lmem_bytes);
  }
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA;
  if ((caps.isa & need) != need) {
    std::printf("%s: the device needs DLMEM, the A extension and the DXA\n", requested ? "Error" : "SKIPPED");
    vx_queue_release(q);
    vx_device_release(dev);
    std::printf(requested ? "FAILED!\n" : "PASSED!\n");
    return requested ? 1 : 0;
  }
  const bool fits = nt >= 4
      && (spread ? (k <= caps.max_cluster && caps.num_cores >= 2 && req_w <= caps.num_warps
                    && arg.lmem_bytes <= caps.lmem_size)
                 : (k * req_w <= caps.num_warps && (uint64_t)k * arg.lmem_bytes <= caps.lmem_size));
  if (!fits) {
    std::printf("%s: %s clusters of %u blocks of %u warps and %u bytes do not fit %u cores of %u warps x %u"
                " threads and %u bytes (largest cluster %u)\n", requested ? "Error" : "SKIPPED",
                spread ? "SPREAD" : "DEFAULT", k, req_w, arg.lmem_bytes, (uint32_t)caps.num_cores,
                (uint32_t)caps.num_warps, nt, (uint32_t)caps.lmem_size, (uint32_t)caps.max_cluster);
    vx_queue_release(q);
    vx_device_release(dev);
    std::printf(requested ? "FAILED!\n" : "PASSED!\n");
    return requested ? 1 : 0;
  }
  if (neg == NEG_RANK) {
    // Rank k when the window's rank field holds it, else rank 1 one block stride out.
    const uint32_t lmem_log = clog2(caps.lmem_size);
    if (k < (1u << clog2(caps.max_cluster)))
      arg.neg_win = (uint64_t)k << lmem_log;
    else
      arg.neg_win = ((uint64_t)1 << lmem_log) + arg.lmem_bytes;
  }

  // Every host buffer stays alive until the queue has finished.
  const uint32_t blocks = req_g * k;
  std::vector<uint32_t> recs((size_t)blocks * REC_WORDS, ~0u);
  for (uint32_t b = 0; b < blocks; ++b)
    recs[(size_t)b * REC_WORDS + REC_RAN] = 0;
  std::vector<uint32_t> errs((size_t)blocks * threads * ERR_WORDS, 0);
  std::vector<uint32_t> dump((size_t)blocks * arg.lmem_bytes / sizeof(uint32_t), 0xDEADBEEFu);
  vx_buffer_h rec_buf  = upload(dev, q, recs, &arg.rec_addr, VX_MEM_READ_WRITE);
  vx_buffer_h err_buf  = upload(dev, q, errs, &arg.err_addr, VX_MEM_READ_WRITE);
  vx_buffer_h dump_buf = upload(dev, q, dump, &arg.dump_addr, VX_MEM_READ_WRITE);
  std::vector<uint32_t> sync(2 * req_g, 0);
  vx_buffer_h sync_buf = upload(dev, q, sync, &arg.sync_addr, VX_MEM_READ_WRITE);

  vx_module_h mod = nullptr;
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, neg ? "neg_k" : "copy_k", &kernel));
  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kernel;
  li.args_host      = &arg;
  li.args_size      = sizeof(arg);
  li.ndim           = 1;
  li.grid_dim[0]    = blocks;
  li.grid_dim[1]    = 1;
  li.grid_dim[2]    = 1;
  li.block_dim[0]   = threads;
  li.block_dim[1]   = 1;
  li.block_dim[2]   = 1;
  li.lmem_size      = arg.lmem_bytes;
  li.cluster_dim[0] = k;
  li.cluster_dim[1] = 1;
  li.cluster_dim[2] = 1;
  li.cluster_policy = spread ? VX_CLUSTER_POLICY_SPREAD : VX_CLUSTER_POLICY_DEFAULT;

  vx_event_h launch_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  std::vector<vx_event_h> evs(3, nullptr);
  CHECK(vx_enqueue_read(q, recs.data(), rec_buf, 0, recs.size() * sizeof(uint32_t), 1, &launch_ev, &evs[0]));
  CHECK(vx_enqueue_read(q, errs.data(), err_buf, 0, errs.size() * sizeof(uint32_t), 1, &launch_ev, &evs[1]));
  CHECK(vx_enqueue_read(q, dump.data(), dump_buf, 0, dump.size() * sizeof(uint32_t), 1, &launch_ev, &evs[2]));
  for (vx_event_h ev : evs) {
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }
  vx_event_release(launch_ev);

  int errors = 0;
  auto rec = [&](uint32_t b) { return &recs[(size_t)b * REC_WORDS]; };
  for (uint32_t b = 0; b < blocks; ++b) {
    const uint32_t* rc = rec(b);
    if (rc[REC_RAN] != 1 || rc[REC_BLOCK] != b || rc[REC_RANK] != b % k || rc[REC_SIZE] != k
        || rc[REC_CORE] >= caps.num_cores) {
      std::printf("*** block %u ran %u times as block %u, rank %u of %u, core %u\n", b, rc[REC_RAN],
                  rc[REC_BLOCK], rc[REC_RANK], rc[REC_SIZE], rc[REC_CORE]);
      ++errors;
    }
  }
  if (verbose || neg) {
    for (uint32_t g = 0; g < req_g; ++g) {
      std::printf("cluster %u (core:slot by rank):", g);
      for (uint32_t r = 0; r < k; ++r)
        std::printf(" %u:%u", rec(g * k + r)[REC_CORE], rec(g * k + r)[REC_SLOT]);
      std::printf("\n");
    }
  }

  if (neg) {
    std::printf("*** the negative run '%s' completed: the simulator must stop it\n", kNegNames[neg]);
    ++errors;
  } else {
    uint32_t unfinished = 0, gave_up = 0;
    uint64_t tries = 0;
    for (size_t t = 0; t < (size_t)blocks * threads; ++t) {
      const uint32_t* e = &errs[t * ERR_WORDS];
      if (e[ERR_DONE] != 1) {
        ++unfinished;
        continue;
      }
      gave_up += e[ERR_WAIT];
      tries += e[ERR_TRIES];
    }
    std::printf("dlmem_bulk: %lu try_waits, %u threads gave up, %u unfinished\n", (unsigned long)tries, gave_up,
                unfinished);
    if (gave_up)
      std::printf("*** mbarrier phase did not complete: %u threads gave up after %u try_waits\n", gave_up,
                  WAIT_LIMIT);
    errors += (gave_up != 0) + (unfinished != 0);

    // The expected image of every block: canaries, its source ranges, the
    // copies it receives and its mbarrier word.
    const uint32_t L = arg.lmem_bytes;
    std::vector<std::vector<uint8_t>> want(blocks, std::vector<uint8_t>(L));
    std::vector<uint64_t> mbar_want(blocks, mr::init(1));
    for (uint32_t b = 0; b < blocks; ++b) {
      for (uint32_t o = 0; o < L; ++o)
        want[b][o] = canary_byte(b, o);
    }
    uint32_t cross = 0, same = 0, stale = 0;
    if (race) {
      for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t r = b % k, so = src_off(&arg, r, 0);
        const uint8_t* got = reinterpret_cast<const uint8_t*>(dump.data()) + (size_t)b * L;
        const uint32_t one = 1;
        std::memcpy(&want[b][lm_flag(&arg)], &one, sizeof(one));
        for (uint32_t h = 0; h < ham_words; ++h) {
          const uint32_t v = ham_word(b, h);
          std::memcpy(&want[b][lm_ham(&arg) + 4 * h], &v, sizeof(v));
        }
        if (is_hog(&arg, r)) {
          // Every round copies the same canaries into slot 0.
          const uint32_t d = dst_off(&arg, r, 0);
          for (uint32_t x = 0; x < hog; ++x)
            want[b][d + x] = canary_byte(b, so + x);
          for (uint32_t j = 0; j < rounds; ++j) {
            mbar_want[b] = mr::apply(mr::OP_ARRIVE_TX, mbar_want[b], hog).new_word;
            mbar_want[b] = mr::apply(mr::OP_COMPLETE_TX, mbar_want[b], hog).new_word;
          }
          continue;
        }
        for (uint32_t x = 0; x < size; ++x)
          want[b][so + x] = race_byte(b, rounds - 1, so + x);
        for (uint32_t j = 0; j < rounds; ++j) {
          const uint32_t d = dst_off(&arg, r, j);
          for (uint32_t x = 0; x < size; ++x) {
            const uint8_t now = race_byte(b, j, so + x);
            const uint8_t old = j ? race_byte(b, j - 1, so + x) : canary_byte(b, so + x);
            want[b][d + x] = now;
            if (got[d + x] != now && got[d + x] == old) {
              ++stale;
              if (!race_fence)
                want[b][d + x] = old;
            }
          }
          mbar_want[b] = mr::apply(mr::OP_ARRIVE_TX, mbar_want[b], size).new_word;
          mbar_want[b] = mr::apply(mr::OP_COMPLETE_TX, mbar_want[b], size).new_word;
        }
      }
    } else {
      for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t r = b % k;
        mbar_want[b] = mr::apply(mr::OP_ARRIVE_TX, mbar_want[b], incoming(&arg, r)).new_word;
        for (uint32_t i = 0; i < copies; ++i) {
          const uint32_t so = src_off(&arg, r, i), n = copy_size(&arg, r, i);
          for (uint32_t x = 0; x < n; ++x)
            want[b][so + x] = src_byte(b, so + x);
        }
      }
      for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t r = b % k, dst = b - r + dst_rank(mode, r, k);
        if (rec(b)[REC_CORE] == rec(dst)[REC_CORE])
          ++same;
        else
          ++cross;
        for (uint32_t i = 0; i < copies; ++i) {
          const uint32_t so = src_off(&arg, r, i), d = dst_off(&arg, r, i), n = copy_size(&arg, r, i);
          for (uint32_t x = 0; x < n; ++x)
            want[dst][d + x] = src_byte(b, so + x);
          mbar_want[dst] = mr::apply(mr::OP_COMPLETE_TX, mbar_want[dst], n).new_word;
        }
      }
    }

    uint32_t bad_bytes = 0, bad_blocks = 0;
    for (uint32_t b = 0; b < blocks; ++b) {
      const uint8_t* got = reinterpret_cast<const uint8_t*>(dump.data()) + (size_t)b * L;
      uint64_t got_w;
      std::memcpy(&got_w, got + arg.lm_mbar, sizeof(got_w));
      if (got_w != mbar_want[b]) {
        std::printf("*** block %u: mbarrier word %#018lx, expected %#018lx\n", b, (unsigned long)got_w,
                    (unsigned long)mbar_want[b]);
        ++errors;
      }
      uint32_t bad = 0;
      for (uint32_t o = 0; o < L; ++o) {
        if (o / 8 == arg.lm_mbar / 8 || got[o] == want[b][o])
          continue;
        if (bad++ < 4 && bad_blocks < 4) {
          const uint32_t ld = lm_dst(&arg), lh = lm_ham(&arg);
          const char* where = o < LM_SRC ? "header" : o < ld ? "source slots" : o < lh ? "destination slots"
                            : "hammer words";
          const uint32_t base = o < LM_SRC ? 0 : o < ld ? LM_SRC : o < lh ? ld : lh;
          std::printf("*** block %u (rank %u, core %u) byte %u (%s, slot %u, offset %u): %#04x, expected %#04x\n",
                      b, b % k, rec(b)[REC_CORE], o, where, (o - base) / arg.pitch, (o - base) % arg.pitch,
                      got[o], want[b][o]);
        }
      }
      if (bad) {
        ++bad_blocks;
        bad_bytes += bad;
      }
    }
    if (race) {
      std::printf("dlmem_bulk: %u stale bytes in %u copies (%s); %u wrong bytes in %u blocks\n", stale,
                  (hog ? blocks - req_g : blocks) * rounds, race_fence ? "fenced, must be 0" : "unfenced, reported only", bad_bytes,
                  bad_blocks);
    } else {
      std::printf("dlmem_bulk: %u copies across cores, %u on the issuer's core; %u wrong bytes in %u blocks\n",
                  cross * copies, same * copies, bad_bytes, bad_blocks);
    }
    errors += (bad_bytes != 0);
    if ((mode == MODE_RING || mode == MODE_MANY) && spread && cross == 0) {
      std::printf("*** no copy crossed cores in a SPREAD launch\n");
      ++errors;
    }
    if ((mode == MODE_RING || mode == MODE_MANY) && !spread && cross != 0) {
      std::printf("*** a DEFAULT launch put a cluster on several cores\n");
      ++errors;
    }
  }

  vx_buffer_release(sync_buf);
  vx_buffer_release(dump_buf);
  vx_buffer_release(err_buf);
  vx_buffer_release(rec_buf);
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
