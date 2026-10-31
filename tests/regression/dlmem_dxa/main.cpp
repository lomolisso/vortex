// dlmem_dxa: DXA copies into the local memory of another block of the
// thread-block cluster, through the DLMEM window, completing on an mbarrier
// in the destination block, on a DLMEM device with the A extension,
// mbarriers and the DXA (RV64). Like CUDA's
// cp.async.bulk.tensor.shared::cluster with .mbarrier::complete_tx::bytes.
//
//   -m ring   block r copies a tile into rank (r + 1) % K (default)
//   -m many   every rank copies into rank 0, back to back, each into its own
//             tile slots; rank 0's own copies use the window address of
//             itself
//   -m self   control: every block copies into its own local memory through
//             local pointers (the single-block path)
//   -p spread|default
//             the cluster policy: SPREAD (default) puts the ranks on
//             different cores; DEFAULT keeps the cluster on one core, so
//             every window destination is another slot of the issuer's core
//   -k K      blocks per cluster (default 2)
//   -n G      clusters (default 2)
//   -w W      warps per block (default 2)
//   -d N      tensor rank of the copies, 1-5 (default 1)
//   -c C      copies each block issues, back to back (default 1)
//   -b E      bytes per element: 1, 2, 4 or 8 (default 4)
//   -x neg    a run that must stop the simulator: rank (the destination and
//             its mbarrier lie outside the cluster), mbar (the mbarrier is in
//             the issuer's block, not the destination's), legacy (a window
//             destination completing on a barrier event), mcast (a window
//             destination with a multicast mask), retired (the destination
//             block has exited when the copy reaches it), box (the box starts
//             inside the destination block and runs past its end)
//   -v        print every cluster's placement
//
// Each receiver arms its mbarrier with arrive_tx of the bytes it will
// receive before the cluster barrier, then waits for the phase. The host
// checks every block's whole local memory byte for byte: each tile slot
// holds its sender's source tile, every other byte its canary; and the final
// mbarrier word against mbar_ref. SPREAD runs must have a copy cross cores,
// DEFAULT runs must keep every copy on one core.
//
// A device the clusters do not fit prints SKIPPED when the run was launched
// without options, and fails when any option asked for a run.

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

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

const char* const kModeNames[] = {"ring", "many", "self"};
const char* const kNegNames[]  = {"", "rank", "mbar", "legacy", "mcast", "retired", "box"};

// Tile extents and source tiles per dimension, by tensor rank. Odd extents
// give rows that end inside a bank word.
const uint32_t kTiles[MAX_DIMS][MAX_DIMS] = {
  {99, 1, 1, 1, 1}, {13, 5, 1, 1, 1}, {5, 3, 2, 1, 1}, {3, 2, 2, 2, 1}, {3, 2, 2, 2, 2}};
const uint32_t kGrids[MAX_DIMS][MAX_DIMS] = {
  {4, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, {2, 2, 2, 1, 1}, {2, 2, 1, 2, 1}, {2, 1, 2, 1, 2}};

constexpr uint32_t kLinger = 20000;

const char* kernel_file = "kernel.vxbin";
uint32_t mode    = MODE_RING;
uint32_t neg     = NEG_NONE;
uint32_t req_k   = 2;
uint32_t req_g   = 2;
uint32_t req_w   = 2;
uint32_t ndim    = 1;
uint32_t copies  = 1;
uint32_t elem    = 4;
bool     spread  = true;
bool     verbose = false;
bool     requested = false;

void usage() {
  std::printf("Usage: [-m ring|many|self] [-p spread|default] [-k K] [-n clusters] [-w warps] [-d rank]"
              " [-c copies] [-b elem bytes] [-x rank|mbar|legacy|mcast|retired|box] [-v] [-h]\n");
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
  while ((c = getopt(argc, argv, "m:p:k:n:w:d:c:b:x:vh")) != -1) {
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
      case 'd': ndim   = parse_count(optarg); break;
      case 'c': copies = parse_count(optarg); break;
      case 'b': elem   = parse_count(optarg); break;
      case 'x': neg    = parse_name(optarg, kNegNames, 1, NUM_NEGS); break;
      case 'v': verbose = true; break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || req_k < 1 || req_k > MAX_CLUSTER || req_g == 0 || req_w == 0 || ndim < 1
      || ndim > MAX_DIMS || copies == 0 || copies > 16 || (elem != 1 && elem != 2 && elem != 4 && elem != 8)
      || (mode != MODE_SELF && req_k < 2)) {
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

uint64_t footprint(uint32_t lmem) {
  return ((uint64_t)lmem + VX_CFG_MEM_BLOCK_SIZE - 1) / VX_CFG_MEM_BLOCK_SIZE * VX_CFG_MEM_BLOCK_SIZE;
}

// The source tile `t` as the bytes the copy writes, row-major, dimension 0 fastest.
std::vector<uint8_t> tile_bytes(const dxa_arg_t& a, uint32_t t) {
  uint32_t base[MAX_DIMS], sizes[MAX_DIMS], elems = 1;
  for (uint32_t d = 0; d < MAX_DIMS; ++d) {
    base[d]  = (t % a.grids[d]) * a.tiles[d];
    t       /= a.grids[d];
    sizes[d] = a.tiles[d] * a.grids[d];
    elems   *= a.tiles[d];
  }
  std::vector<uint8_t> out((size_t)elems * elem);
  for (uint32_t e = 0; e < elems; ++e) {
    uint64_t g = 0, mult = 1;
    uint32_t rem = e;
    for (uint32_t d = 0; d < MAX_DIMS; ++d) {
      g += (uint64_t)(base[d] + rem % a.tiles[d]) * mult;
      rem /= a.tiles[d];
      mult *= sizes[d];
    }
    for (uint32_t i = 0; i < elem; ++i)
      out[(size_t)e * elem + i] = src_byte((uint32_t)(g * elem + i));
  }
  return out;
}

int program_desc(vx_device_h dev, const dxa_arg_t& a, uint64_t src) {
  uint32_t s[MAX_DIMS], st[MAX_DIMS - 1] = {};
  for (uint32_t d = 0; d < MAX_DIMS; ++d)
    s[d] = a.tiles[d] * a.grids[d];
  uint32_t acc = s[0] * elem;
  for (uint32_t d = 0; d + 1 < MAX_DIMS; ++d) {
    st[d] = acc;
    acc *= s[d + 1];
  }
  const uint32_t* t = a.tiles;
  namespace dx = vortex::dxa;
  switch (a.ndim) {
  case 1:  return dx::program_1d(dev, DESC_SLOT, src, s[0], t[0], elem);
  case 2:  return dx::program_2d(dev, DESC_SLOT, src, s[0], s[1], st[0], t[0], t[1], elem);
  case 3:  return dx::program_3d(dev, DESC_SLOT, src, s[0], s[1], s[2], st[0], st[1], t[0], t[1], t[2], elem);
  case 4:  return dx::program_4d(dev, DESC_SLOT, src, s[0], s[1], s[2], s[3], st[0], st[1], st[2],
                                 t[0], t[1], t[2], t[3], elem);
  default: return dx::program_5d(dev, DESC_SLOT, src, s[0], s[1], s[2], s[3], s[4], st[0], st[1], st[2], st[3],
                                 t[0], t[1], t[2], t[3], t[4], elem);
  }
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
    ndim = 1;
    copies = 1;
  }
  const uint32_t k = req_k, nt = (uint32_t)caps.num_threads, threads = req_w * nt;

  dxa_arg_t arg = {};
  arg.mode   = mode;
  arg.neg    = neg;
  arg.k      = k;
  arg.ndim   = ndim;
  arg.copies = copies;
  arg.ntiles = 1;
  uint32_t elems = 1;
  for (uint32_t d = 0; d < MAX_DIMS; ++d) {
    arg.tiles[d] = kTiles[ndim - 1][d];
    arg.grids[d] = kGrids[ndim - 1][d];
    arg.ntiles  *= arg.grids[d];
    elems       *= arg.tiles[d];
  }
  arg.box        = elems * elem;
  arg.tstride    = (arg.box + 63) / 64 * 64 + 64;
  arg.lmem_bytes = LM_TILE + num_slots(mode, k, copies) * arg.tstride;
  arg.linger     = kLinger;

  if (neg) {
    std::printf("dlmem_dxa: negative run '%s', SPREAD cluster of 2 blocks, %u cores of %u warps x %u threads\n",
                kNegNames[neg], (uint32_t)caps.num_cores, (uint32_t)caps.num_warps, nt);
  } else {
    std::printf("dlmem_dxa: %s, %s, %u clusters of %u blocks of %u warps x %u threads, %u cores of %u warps,"
                " %u copies per block of a rank-%u box of %u bytes (%u-byte elements), %u bytes of local memory"
                " per block\n", kModeNames[mode], spread ? "SPREAD" : "DEFAULT", req_g, k, req_w, nt,
                (uint32_t)caps.num_cores, (uint32_t)caps.num_warps, copies, ndim, arg.box, elem, arg.lmem_bytes);
  }
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA;
  if ((caps.isa & need) != need) {
    std::printf("Error: the device needs DLMEM, the A extension and the DXA\nFAILED!\n");
    return 1;
  }
  const bool fits = spread ? (k <= caps.max_cluster && caps.num_cores >= 2 && req_w <= caps.num_warps
                              && arg.lmem_bytes <= caps.lmem_size)
                           : (k * req_w <= caps.num_warps && k * footprint(arg.lmem_bytes) <= caps.lmem_size);
  if (!fits) {
    std::printf("%s: %s clusters of %u blocks of %u warps and %u bytes do not fit %u cores of %u warps and"
                " %u bytes (largest cluster %u)\n", requested ? "Error" : "SKIPPED", spread ? "SPREAD" : "DEFAULT",
                k, req_w, arg.lmem_bytes, (uint32_t)caps.num_cores, (uint32_t)caps.num_warps,
                (uint32_t)caps.lmem_size, (uint32_t)caps.max_cluster);
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
      arg.neg_win = ((uint64_t)1 << lmem_log) + footprint(arg.lmem_bytes);
  }
  if (neg == NEG_BOX) {
    // 64 bytes before the end of the block, the slot pitch.
    arg.neg_win = footprint(arg.lmem_bytes) - 64;
  }

  // Every host buffer stays alive until the queue has finished.
  uint64_t src_elems = 1;
  for (uint32_t d = 0; d < MAX_DIMS; ++d)
    src_elems *= (uint64_t)arg.tiles[d] * arg.grids[d];
  std::vector<uint8_t> src(src_elems * elem);
  for (uint32_t i = 0; i < src.size(); ++i)
    src[i] = src_byte(i);
  uint64_t src_addr = 0;
  vx_buffer_h src_buf = upload(dev, q, src, &src_addr, VX_MEM_READ);
  if (program_desc(dev, arg, src_addr) != 0
      || (neg == NEG_MCAST && vortex::dxa::set_multicast(dev, DESC_SLOT, (uint32_t)footprint(arg.lmem_bytes)) != 0)) {
    std::printf("Error: cannot program the DXA descriptor\nFAILED!\n");
    return 1;
  }

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
    std::printf("dlmem_dxa: %lu try_waits, %u threads gave up, %u unfinished\n", (unsigned long)tries, gave_up,
                unfinished);
    if (gave_up)
      std::printf("*** mbarrier phase did not complete: %u threads gave up after %u try_waits\n", gave_up,
                  WAIT_LIMIT);
    errors += (gave_up != 0) + (unfinished != 0);

    // The expected image of every block: canaries, the tiles it receives and its mbarrier word.
    std::vector<std::vector<uint8_t>> tiles(arg.ntiles);
    for (uint32_t t = 0; t < arg.ntiles; ++t)
      tiles[t] = tile_bytes(arg, t);
    uint32_t cross = 0, same = 0, bad_bytes = 0, bad_blocks = 0;
    std::vector<std::vector<uint8_t>> want(blocks);
    for (uint32_t b = 0; b < blocks; ++b) {
      want[b].resize(arg.lmem_bytes);
      for (uint32_t i = 8; i < arg.lmem_bytes; ++i)
        want[b][i] = canary_byte(b, i);
    }
    for (uint32_t b = 0; b < blocks; ++b) {
      const uint32_t r = b % k, dst = b - r + dst_rank(mode, r, k);
      if (rec(b)[REC_CORE] == rec(dst)[REC_CORE])
        ++same;
      else
        ++cross;
      for (uint32_t i = 0; i < copies; ++i) {
        const std::vector<uint8_t>& t = tiles[src_tile(b, i, copies, arg.ntiles)];
        std::memcpy(&want[dst][LM_TILE + dst_slot(mode, r, i, copies) * arg.tstride], t.data(), t.size());
      }
    }
    for (uint32_t b = 0; b < blocks; ++b) {
      uint64_t w = mr::init(1);
      const uint32_t n = incoming(mode, b % k, k, copies);
      w = mr::apply(mr::OP_ARRIVE_TX, w, n * arg.box).new_word;
      for (uint32_t i = 0; i < n; ++i)
        w = mr::apply(mr::OP_COMPLETE_TX, w, arg.box).new_word;
      std::memcpy(want[b].data(), &w, sizeof(w));
      const uint8_t* got = reinterpret_cast<const uint8_t*>(dump.data()) + (size_t)b * arg.lmem_bytes;
      uint64_t got_w;
      std::memcpy(&got_w, got, sizeof(got_w));
      if (got_w != w) {
        std::printf("*** block %u: mbarrier word %#018lx, expected %#018lx\n", b, (unsigned long)got_w,
                    (unsigned long)w);
        ++errors;
      }
      uint32_t bad = 0;
      for (uint32_t i = 8; i < arg.lmem_bytes; ++i) {
        if (got[i] == want[b][i])
          continue;
        if (bad++ < 4 && bad_blocks < 4) {
          const uint32_t s = i < LM_TILE ? ~0u : (i - LM_TILE) / arg.tstride;
          const uint32_t o = i < LM_TILE ? i : (i - LM_TILE) % arg.tstride;
          std::printf("*** block %u (rank %u, core %u) byte %u (slot %d, offset %u%s): %#04x, expected %#04x\n", b,
                      b % k, rec(b)[REC_CORE], i, (int)s, o, o >= arg.box ? ", past the tile" : "", got[i],
                      want[b][i]);
        }
      }
      if (bad) {
        ++bad_blocks;
        bad_bytes += bad;
      }
    }
    std::printf("dlmem_dxa: %u copies across cores, %u on the issuer's core; %u wrong bytes in %u blocks\n",
                cross * copies, same * copies, bad_bytes, bad_blocks);
    errors += (bad_bytes != 0);
    if (mode != MODE_SELF && spread && cross == 0) {
      std::printf("*** no copy crossed cores in a SPREAD launch\n");
      ++errors;
    }
    if (mode != MODE_SELF && !spread && cross != 0) {
      std::printf("*** a DEFAULT launch put a cluster on several cores\n");
      ++errors;
    }
  }

  vx_buffer_release(dump_buf);
  vx_buffer_release(err_buf);
  vx_buffer_release(rec_buf);
  vx_buffer_release(src_buf);
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
