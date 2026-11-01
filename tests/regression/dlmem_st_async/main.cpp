// dlmem_st_async: st.async and red.async into the local memory of a block of
// the thread-block cluster, through the DLMEM window, each completing its 4
// or 8 bytes on an mbarrier in the destination block, on a DLMEM device with
// the A extension and mbarriers (RV64). Like PTX st.async / red.async
// .shared::cluster .mbarrier::complete_tx::bytes.
//
//   -m ring   stores: block r into rank (r + 1) % K; adds: every rank into
//             rank 0's counters, rank 0 into its own (default)
//   -m self   every block into itself, through its own window address
//   -p spread|default
//             the cluster policy: SPREAD (default) puts the ranks on
//             different cores; DEFAULT keeps the cluster on one core
//   -k K      blocks per cluster (default 2)
//   -n G      clusters (default 2)
//   -w W      warps per block (default 2)
//   -e N      ops per thread, op and round (default 2)
//   -r R      rounds (default 2)
//   -o MASK   the ops: bit 0 st.async.b32, 1 st.async.b64, 2 red.async.add.u32,
//             3 red.async.add.u64 (default 15)
//   -x neg    a run that must stop the simulator, from rank 0 of a SPREAD
//             cluster of 2: local (the destination is a local pointer), mbar
//             (the mbarrier is in the issuer's block), mbarlocal (the
//             mbarrier is a local pointer), rank (the destination lies outside
//             the cluster), off (the destination lies past the peer's block)
//   -v        print every cluster's placement
//
// Each round, every receiving block arms its mbarriers with arrive_tx of the
// bytes coming in before the cluster barrier; the senders issue with no fence;
// every thread waits for its own block's phases and checks at once that every
// stored word holds the round's value (no stale data) and every counter the
// exact total so far. The host then checks every block's whole local memory
// byte for byte (the last round's words, the counters, canaries everywhere
// else) and each mbarrier word against mbar_ref. SPREAD runs must have ops
// cross cores, DEFAULT runs must keep them on one core.
//
// A device the clusters do not fit prints SKIPPED when the run was launched
// without options, and fails when any option asked for a run.

#include <vortex2.h>
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

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

const char* const kModeNames[] = {"ring", "self"};
const char* const kNegNames[]  = {"", "local", "mbar", "mbarlocal", "rank", "off"};
const char* const kOpNames[]   = {"st.async.b32", "st.async.b64", "red.async.add.u32", "red.async.add.u64"};

constexpr uint32_t kLinger = 20000;

const char* kernel_file = "kernel.vxbin";
uint32_t mode    = MODE_RING;
uint32_t neg     = NEG_NONE;
uint32_t req_k   = 2;
uint32_t req_g   = 2;
uint32_t req_w   = 2;
uint32_t elems   = 2;
uint32_t rounds  = 2;
uint32_t ops     = 0xF;
bool     spread  = true;
bool     verbose = false;
bool     requested = false;

void usage() {
  std::printf("Usage: [-m ring|self] [-p spread|default] [-k K] [-n clusters] [-w warps] [-e ops per thread]"
              " [-r rounds] [-o op mask] [-x local|mbar|mbarlocal|rank|off] [-v] [-h]\n");
}

uint32_t parse_count(const char* s) {
  char* end = nullptr;
  errno = 0;
  const unsigned long v = std::strtoul(s, &end, 0);
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
  while ((c = getopt(argc, argv, "m:p:k:n:w:e:r:o:x:vh")) != -1) {
    switch (c) {
      case 'm': mode = parse_name(optarg, kModeNames, 0, NUM_MODES); break;
      case 'p': {
        const char* const kPolicies[] = {"spread", "default"};
        spread = parse_name(optarg, kPolicies, 0, 2) == 0;
        break;
      }
      case 'k': req_k  = parse_count(optarg); break;
      case 'n': req_g  = parse_count(optarg); break;
      case 'w': req_w  = parse_count(optarg); break;
      case 'e': elems  = parse_count(optarg); break;
      case 'r': rounds = parse_count(optarg); break;
      case 'o': ops    = parse_count(optarg); break;
      case 'x': neg    = parse_name(optarg, kNegNames, 1, NUM_NEGS); break;
      case 'v': verbose = true; break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || req_k < 1 || req_k > 8 || req_g == 0 || req_w == 0 || elems == 0 || elems > 64
      || rounds == 0 || rounds > 64 || ops == 0 || ops > 0xF) {
    usage();
    std::exit(-1);
  }
}

uint32_t clog2(uint64_t x) {
  uint32_t n = 0;
  while ((uint64_t(1) << n) < x)
    ++n;
  return n;
}

uint64_t footprint(uint32_t lmem) {
  return ((uint64_t)lmem + VX_CFG_MEM_BLOCK_SIZE - 1) / VX_CFG_MEM_BLOCK_SIZE * VX_CFG_MEM_BLOCK_SIZE;
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

template <typename T>
void put(std::vector<uint8_t>& img, uint32_t off, T v) {
  std::memcpy(&img[off], &v, sizeof(v));
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

  uint64_t num_cores = 0, num_warps = 0, num_threads = 0, lmem_size = 0, max_cluster = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,         &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,         &num_warps));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &num_threads));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE,    &lmem_size));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &isa));

  if (neg) {
    mode = MODE_RING; spread = true; req_k = 2; req_g = 1; req_w = 1; elems = 1; rounds = 1; ops = 1;
  }
  const uint32_t k = req_k, nt = (uint32_t)num_threads, threads = req_w * nt;
  if (threads * elems > 1024 * 64 || threads > 1024) {
    usage();
    return -1;
  }

  sa_arg_t arg = {};
  arg.mode       = mode;
  arg.neg        = neg;
  arg.k          = k;
  arg.ops        = ops;
  arg.rounds     = rounds;
  arg.elems      = elems;
  arg.lmem_bytes = lm_bytes(threads, elems);
  arg.linger     = kLinger;

  if (neg) {
    std::printf("dlmem_st_async: negative run '%s', SPREAD cluster of 2 blocks, %u cores of %u warps x %u threads\n",
                kNegNames[neg], (uint32_t)num_cores, (uint32_t)num_warps, nt);
  } else {
    std::printf("dlmem_st_async: %s, %s, %u clusters of %u blocks of %u warps x %u threads, %u cores of %u warps,"
                " %u rounds of %u ops per thread of", kModeNames[mode], spread ? "SPREAD" : "DEFAULT", req_g, k,
                req_w, nt, (uint32_t)num_cores, (uint32_t)num_warps, rounds, elems);
    for (uint32_t o = 0; o < NUM_OPS; ++o) {
      if (ops & (1u << o))
        std::printf(" %s", kOpNames[o]);
    }
    std::printf(", %u bytes of local memory per block\n", arg.lmem_bytes);
  }
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A;
  if ((isa & need) != need) {
    std::printf("Error: the device needs DLMEM and the A extension\nFAILED!\n");
    return 1;
  }
  const bool fits = spread ? (k <= max_cluster && num_cores >= 2 && req_w <= num_warps
                              && arg.lmem_bytes <= lmem_size)
                           : (k * req_w <= num_warps && k * footprint(arg.lmem_bytes) <= lmem_size);
  if (!fits) {
    std::printf("%s: %s clusters of %u blocks of %u warps and %u bytes do not fit %u cores of %u warps and"
                " %u bytes (largest cluster %u)\n", requested ? "Error" : "SKIPPED", spread ? "SPREAD" : "DEFAULT",
                k, req_w, arg.lmem_bytes, (uint32_t)num_cores, (uint32_t)num_warps, (uint32_t)lmem_size,
                (uint32_t)max_cluster);
    vx_queue_release(q);
    vx_device_release(dev);
    std::printf(requested ? "FAILED!\n" : "PASSED!\n");
    return requested ? 1 : 0;
  }
  if (neg == NEG_RANK) {
    // Rank k when the window's rank field holds it, else rank 1 one block stride out.
    const uint32_t lmem_log = clog2(lmem_size);
    if (k < (1u << clog2(max_cluster)))
      arg.neg_win = (uint64_t)k << lmem_log;
    else
      arg.neg_win = ((uint64_t)1 << lmem_log) + footprint(arg.lmem_bytes);
  }
  if (neg == NEG_OFF)
    arg.neg_win = footprint(arg.lmem_bytes);

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

  vx_module_h mod = nullptr;
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, neg ? "neg_k" : "st_k", &kernel));
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
    if (rc[REC_RAN] != 1 || rc[REC_BLOCK] != b || rc[REC_RANK] != b % k || rc[REC_CORE] >= num_cores) {
      std::printf("*** block %u ran %u times as block %u, rank %u, core %u\n", b, rc[REC_RAN], rc[REC_BLOCK],
                  rc[REC_RANK], rc[REC_CORE]);
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
    uint32_t unfinished = 0, gave_up = 0, stale = 0, bad_red = 0;
    uint64_t tries = 0;
    for (size_t t = 0; t < (size_t)blocks * threads; ++t) {
      const uint32_t* e = &errs[t * ERR_WORDS];
      if (e[ERR_DONE] != 1) {
        ++unfinished;
        continue;
      }
      gave_up += e[ERR_WAIT];
      stale   += e[ERR_STALE];
      bad_red += e[ERR_RED];
      tries   += e[ERR_TRIES];
    }
    std::printf("dlmem_st_async: %lu try_waits, %u waits gave up, %u stale words, %u wrong counters, %u unfinished\n",
                (unsigned long)tries, gave_up, stale, bad_red, unfinished);
    if (gave_up)
      std::printf("*** mbarrier phase did not complete: %u waits gave up after %u try_waits\n", gave_up, WAIT_LIMIT);
    if (stale)
      std::printf("*** %u stored words did not hold the round's value right after the wait\n", stale);
    if (bad_red)
      std::printf("*** %u counters did not hold the round's total right after the wait\n", bad_red);
    errors += (gave_up != 0) + (stale != 0) + (bad_red != 0) + (unfinished != 0);

    // The expected image of every block.
    const uint32_t st64_off = lm_st64(threads, elems);
    uint32_t cross = 0, same = 0, bad_bytes = 0, bad_blocks = 0;
    for (uint32_t b = 0; b < blocks; ++b) {
      const uint32_t r = b % k, base = b - r;
      std::vector<uint8_t> want(arg.lmem_bytes, 0);
      for (uint32_t w = LM_CANARY; w < arg.lmem_bytes / 4; ++w)
        put<uint32_t>(want, w * 4, canary_word(b, w));
      const uint32_t src = base + (mode == MODE_RING ? (r + k - 1) % k : r);
      const uint32_t p = rounds - 1;
      for (uint32_t t = 0; t < threads; ++t) {
        for (uint32_t i = 0; i < elems; ++i) {
          const uint32_t w = i * threads + t;
          if (ops & (1u << OP_ST32))
            put<uint32_t>(want, LM_ST32 + w * 4, st32_value(p, src, t, i));
          if (ops & (1u << OP_ST64))
            put<uint64_t>(want, st64_off + w * 8, st64_value(p, src, t, i));
        }
      }
      uint32_t c32 = 0;
      uint64_t c64 = 0;
      for (uint32_t s = 0; s < k; ++s) {
        if (red_dst(mode, s) != r)
          continue;
        for (uint32_t pp = 0; pp < rounds; ++pp) {
          for (uint32_t t = 0; t < threads; ++t) {
            for (uint32_t i = 0; i < elems; ++i) {
              if (ops & (1u << OP_RED32))
                c32 += red32_value(pp, base + s, t, i);
              if (ops & (1u << OP_RED64))
                c64 += red64_value(pp, base + s, t, i);
            }
          }
        }
      }
      put<uint64_t>(want, LM_CTR32, c32);
      put<uint64_t>(want, LM_CTR64, c64);
      const uint32_t in[NUM_OPS] = {threads * elems, threads * elems,
                                    red_senders(mode, r, k) * threads * elems,
                                    red_senders(mode, r, k) * threads * elems};
      for (uint32_t o = 0; o < NUM_OPS; ++o) {
        uint64_t w = mr::init(1);
        if ((ops & (1u << o)) && in[o]) {
          for (uint32_t pp = 0; pp < rounds; ++pp) {
            w = mr::apply(mr::OP_ARRIVE_TX, w, in[o] * op_bytes(o)).new_word;
            for (uint32_t j = 0; j < in[o]; ++j)
              w = mr::apply(mr::OP_COMPLETE_TX, w, op_bytes(o)).new_word;
          }
        }
        put<uint64_t>(want, LM_MBAR + o * 8, w);
      }

      const uint32_t q_st = base + st_dst(mode, r, k);
      if (rec(b)[REC_CORE] == rec(q_st)[REC_CORE])
        ++same;
      else
        ++cross;

      const uint8_t* got = reinterpret_cast<const uint8_t*>(dump.data()) + (size_t)b * arg.lmem_bytes;
      uint32_t bad = 0;
      for (uint32_t i = 0; i < arg.lmem_bytes; ++i) {
        if (got[i] == want[i])
          continue;
        if (bad++ < 4 && bad_blocks < 4) {
          const char* where = i < LM_CTR32 ? "mbarrier" : i < LM_CTR64 ? "counter32"
                              : i < LM_CTR64 + 8 ? "counter64" : i < LM_ST32 ? "gap"
                              : i < st64_off ? "st32 area" : i < st64_off + threads * elems * 8 ? "st64 area"
                              : "tail";
          std::printf("*** block %u (rank %u, core %u) byte %u (%s): %#04x, expected %#04x\n", b, r,
                      rec(b)[REC_CORE], i, where, got[i], want[i]);
        }
      }
      if (bad) {
        ++bad_blocks;
        bad_bytes += bad;
      }
    }
    std::printf("dlmem_st_async: %u blocks store across cores, %u on their own core; %u wrong bytes in %u blocks\n",
                cross, same, bad_bytes, bad_blocks);
    errors += (bad_bytes != 0);
    if (mode == MODE_RING && spread && cross == 0) {
      std::printf("*** no op crossed cores in a SPREAD launch\n");
      ++errors;
    }
    if (!spread && cross != 0) {
      std::printf("*** a DEFAULT launch put a cluster on several cores\n");
      ++errors;
    }
  }

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
