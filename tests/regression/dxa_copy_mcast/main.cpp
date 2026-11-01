// Host driver for dxa_copy_mcast — DXA multicast of one tile into several
// blocks.
//
// With no mode flag: the legacy intra-core multicast, VX_CFG_NUM_WARPS
// single-warp CTAs co-resident on one core, each receiving the same tile.
//
// Checked modes, on a DLMEM device with the A extension and mbarriers. Each
// block's whole local memory is checked byte for byte: every receiver holds
// the copies in its tile slots, every other byte (and every byte of a
// non-receiver) its canary.
//   -m MASK   legacy barrier multicast with any mask: a DEFAULT cluster of
//             -s blocks (default: the warps of a core), rank 0 issues, and
//             bit j selects the block in the issuer's slot + j
//   -M MASK   cluster multicast on mbarriers: bit r selects cluster rank r,
//             the data and the completion land at the issuer's block offsets
//             in every masked rank; the issuer may be outside the mask, and
//             then its own mbarrier must stay untouched
//   -s K      blocks per cluster (-M default 4)
//   -g G      clusters (default 1)
//   -w W      warps per block (default 1)
//   -d N      tensor rank of the copies, 1-5 (default 2)
//   -n C      copies the issuer multicasts, each into the next tile slot (default 1)
//   -b E      bytes per element: 1, 2, 4 or 8 (default 4)
//   -i R      -M: the issuing rank (default 0)
//   -p spread|default
//             -M: the cluster policy (default spread)
//   -H        -M: issue through vortex::dxa_multicast_cluster_Nd
//   -x neg    a run that must stop the simulator, in a SPREAD cluster of two
//             blocks: rank (a mask bit past the cluster), spread (a barrier
//             multicast from a SPREAD cluster), box (the box runs past the
//             end of the receivers' blocks), retired (a receiver has exited)
//   -v        print every cluster's placement

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>
#include <dxa.h>
#include <mbar_ref.h>

#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret) break;                                       \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

static const char* kernel_file = "kernel.vxbin";
static uint32_t    tile_rows   = 16;
static uint32_t    tile_cols   = 4;
static uint32_t    src_rows    = 16;
static uint32_t    src_cols    = 16;

static vx_device_h device     = nullptr;
static vx_buffer_h src_buffer = nullptr;
static vx_buffer_h dst_buffer = nullptr;
static vx_queue_h  queue      = nullptr;
static vx_module_h module_    = nullptr;
static vx_kernel_h kernel     = nullptr;
static kernel_arg_t kernel_arg = {};

static bool     checked  = false;
static uint32_t mc_mode  = MC_LEGACY;
static uint32_t mc_neg   = NEG_NONE;
static uint32_t mc_mask  = 0;
static uint32_t mc_k     = 0;
static uint32_t mc_g     = 1;
static uint32_t mc_w     = 1;
static uint32_t mc_ndim  = 2;
static uint32_t mc_copies = 1;
static uint32_t mc_elem  = 4;
static uint32_t mc_issuer = 0;
static bool     mc_spread = true;
static bool     mc_helper = false;
static bool     mc_verbose = false;

static const char* const kNegNames[] = {"", "rank", "spread", "box", "retired"};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-r src_rows] [-c src_cols] "
               "[-R tile_rows] [-C tile_cols] [-h]\n"
               "       -m MASK | -M MASK | -x rank|spread|box|retired, with [-s K] [-g clusters] [-w warps]"
               " [-d rank] [-n copies] [-b elem bytes] [-i issuer] [-p spread|default] [-H] [-v]\n";
}

static uint32_t parse_num(const char* s) {
  char* end = nullptr;
  errno = 0;
  const unsigned long v = std::strtoul(s, &end, 0);
  if (*s < '0' || *s > '9' || *end != '\0' || errno != 0 || v > UINT32_MAX) {
    show_usage();
    exit(-1);
  }
  return (uint32_t)v;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:R:C:k:m:M:s:g:w:d:n:b:i:p:x:Hvh")) != -1) {
    switch (c) {
      case 'r': src_rows  = atoi(optarg); break;
      case 'c': src_cols  = atoi(optarg); break;
      case 'R': tile_rows = atoi(optarg); break;
      case 'C': tile_cols = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'm': checked = true; mc_mode = MC_LEGACY;  mc_mask = parse_num(optarg); break;
      case 'M': checked = true; mc_mode = MC_CLUSTER; mc_mask = parse_num(optarg); break;
      case 's': mc_k      = parse_num(optarg); break;
      case 'g': mc_g      = parse_num(optarg); break;
      case 'w': mc_w      = parse_num(optarg); break;
      case 'd': mc_ndim   = parse_num(optarg); break;
      case 'n': mc_copies = parse_num(optarg); break;
      case 'b': mc_elem   = parse_num(optarg); break;
      case 'i': mc_issuer = parse_num(optarg); break;
      case 'p':
        if (!strcmp(optarg, "spread") || !strcmp(optarg, "default")) {
          mc_spread = !strcmp(optarg, "spread");
          break;
        }
        show_usage(); exit(-1);
      case 'x': {
        checked = true;
        mc_mode = MC_CLUSTER;
        uint32_t n = 1;
        while (n < NUM_NEGS && strcmp(optarg, kNegNames[n]))
          ++n;
        if (n == NUM_NEGS) {
          show_usage(); exit(-1);
        }
        mc_neg = n;
        break;
      }
      case 'H': mc_helper = true; break;
      case 'v': mc_verbose = true; break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
    }
  }
}

static int run_checked();

void cleanup() {
  if (device) {
    if (src_buffer) vx_buffer_release(src_buffer);
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  if (checked)
    return run_checked();
  std::srand(42);

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Single-warp CTAs: block = (tile_cols, 1) so each CTA uses exactly one
  // warp. num_recv = VX_CFG_NUM_WARPS so all receivers fit co-resident on one core.
  uint64_t num_warps = 0, num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  if (tile_cols != (uint32_t)num_threads) {
    std::cout << "Error: tile_cols (" << tile_cols
              << ") must equal VX_CFG_NUM_THREADS (" << num_threads
              << ") for single-warp CTAs\n";
    cleanup();
    return -1;
  }

  const uint32_t num_recv   = (uint32_t)num_warps;
  const uint32_t tile_elems = tile_rows * tile_cols;
  const uint32_t local_mem  = tile_elems * sizeof(TYPE);
  const uint32_t src_elems  = src_rows * src_cols;
  const uint32_t src_bytes  = src_elems * sizeof(TYPE);
  const uint32_t dst_bytes  = num_recv * tile_elems * sizeof(TYPE);

  std::cout << "dxa_copy_mcast (intra-core multicast)\n";
  std::cout << "  source: " << src_rows << " x " << src_cols
            << ", tile: " << tile_rows << " x " << tile_cols << "\n";
  std::cout << "  block=" << tile_cols << "x1 (single warp), grid="
            << num_recv << "x1, num_recv=" << num_recv << "\n";
  std::cout << "  local_mem=" << local_mem << " bytes\n";

  RT_CHECK(vx_check_occupancy(device, tile_cols, local_mem));

  RT_CHECK(vx_buffer_create(device, src_bytes, VX_MEM_READ,  &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_buffer_create(device, dst_bytes, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  // Distinctive source pattern (linear index + 1).
  std::vector<TYPE> h_src(src_elems);
  for (uint32_t i = 0; i < src_elems; ++i)
    h_src[i] = static_cast<TYPE>(i + 1);
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), src_bytes, 0, nullptr, nullptr));

  // Program DXA descriptor for the tile.
  constexpr uint32_t kDescSrc = 0;
  RT_CHECK(vortex::dxa::program_2d(device, kDescSrc, kernel_arg.src_addr,
    /*size0=*/src_cols, /*size1=*/src_rows,
    /*stride0_bytes=*/src_cols * sizeof(TYPE),
    /*tile0=*/tile_cols, /*tile1=*/tile_rows,
    /*elem_bytes=*/sizeof(TYPE)));
  // Multicast stride = local_mem (per-CTA SMEM size). Dispatcher allocates
  // sequential LMEM regions so receiver k's region starts at k*local_mem.
  RT_CHECK(vortex::dxa::set_multicast(device, kDescSrc, local_mem));

  kernel_arg.tile_rows      = tile_rows;
  kernel_arg.tile_cols      = tile_cols;
  kernel_arg.src_row_stride = src_cols;
  kernel_arg.num_recv       = num_recv;

  uint32_t grid_dim[2]  = { num_recv, 1 };
  uint32_t block_dim[2] = { tile_cols, 1 };

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // Host result buffer — must outlive the async read enqueued below.
  std::vector<TYPE> h_dst(num_recv * tile_elems, 0);

  std::cout << "start\n";
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = grid_dim[0];
    li.grid_dim[1]  = grid_dim[1];
    li.block_dim[0] = block_dim[0];
    li.block_dim[1] = block_dim[1];
    li.lmem_size    = local_mem;
    // Multicast group: all num_recv receiver CTAs share one B tile and must be
    // co-resident on one core in contiguous CTA slots. The grid is num_recv×1,
    // so the group spans the X axis. Without this, cluster_size defaults to 1,
    // get_cluster_rank() is 0 for every CTA, and every CTA (not just rank-0)
    // fires the multicast — over-releasing receivers' event barriers and
    // walking the barrier-id space past its bounds.
    li.cluster_dim[0] = num_recv;
    li.cluster_dim[1] = 1;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  // Verify each receiver got the same tile.
  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, dst_bytes, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t rcv = 0; rcv < num_recv; ++rcv) {
    for (uint32_t r = 0; r < tile_rows; ++r) {
      for (uint32_t c = 0; c < tile_cols; ++c) {
        TYPE expected = h_src[r * src_cols + c];
        TYPE actual   = h_dst[rcv * tile_elems + r * tile_cols + c];
        if (expected != actual) {
          if (errors < 20)
            printf("*** error: recv=%u [r=%u,c=%u] expected=%f actual=%f\n",
                   rcv, r, c, (float)expected, (float)actual);
          ++errors;
        }
      }
    }
  }

  cleanup();
  if (errors) {
    std::cout << "FAILED (" << errors << " errors)\n";
    return errors;
  }
  std::cout << "PASSED\n";
  return 0;
}

// ---------------------------------------------------------------------------
// The checked modes.

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

// Tile extents and source tiles per dimension, by tensor rank. Odd extents
// give rows that end inside a bank word.
const uint32_t kTiles[MAX_DIMS][MAX_DIMS] = {
  {99, 1, 1, 1, 1}, {13, 5, 1, 1, 1}, {5, 3, 2, 1, 1}, {3, 2, 2, 2, 1}, {3, 2, 2, 2, 2}};
const uint32_t kGrids[MAX_DIMS][MAX_DIMS] = {
  {4, 1, 1, 1, 1}, {2, 2, 1, 1, 1}, {2, 2, 2, 1, 1}, {2, 2, 1, 2, 1}, {2, 1, 2, 1, 2}};

constexpr uint32_t kLinger = 20000;

uint64_t footprint(uint32_t lmem) {
  return ((uint64_t)lmem + VX_CFG_MEM_BLOCK_SIZE - 1) / VX_CFG_MEM_BLOCK_SIZE * VX_CFG_MEM_BLOCK_SIZE;
}

// The source tile `t` as the bytes the copy writes, row-major, dimension 0 fastest.
std::vector<uint8_t> tile_bytes(const mc_arg_t& a, uint32_t t, uint32_t elem) {
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

int program_desc(vx_device_h dev, const mc_arg_t& a, uint64_t src, uint32_t elem) {
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

static int run_checked() {
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

  const bool legacy = (mc_mode == MC_LEGACY);
  if (mc_neg) {
    mc_k = 2; mc_g = 1; mc_w = 1; mc_ndim = 1; mc_copies = 1; mc_issuer = 0;
    mc_spread = true; mc_helper = false;
    mc_mask = (mc_neg == NEG_RANK) ? 0x4 : 0x3;
  }
  if (mc_k == 0)
    mc_k = legacy ? (uint32_t)num_warps / (mc_w ? mc_w : 1) : 4;
  const uint32_t k = mc_k, nt = (uint32_t)num_threads, threads = mc_w * nt, elem = mc_elem;
  if (k < 1 || k > 32 || mc_g == 0 || mc_w == 0 || mc_ndim < 1 || mc_ndim > MAX_DIMS || mc_copies == 0
      || mc_copies > 8 || (elem != 1 && elem != 2 && elem != 4 && elem != 8)
      || (!mc_neg && (mc_mask == 0 || (k < 32 && (mc_mask >> k) != 0))) || mc_issuer >= k
      || (legacy && (mc_helper || __builtin_popcount(mc_mask) < 2))) {
    // A one-bit legacy mask is no multicast: it copies into the issuer.
    show_usage();
    return -1;
  }

  mc_arg_t arg = {};
  arg.mode    = mc_mode;
  arg.neg     = mc_neg;
  arg.k       = k;
  arg.mask    = mc_mask;
  arg.issuer  = mc_issuer;
  arg.helper  = mc_helper;
  arg.ndim    = mc_ndim;
  arg.copies  = mc_copies;
  arg.ntiles  = 1;
  uint32_t elems = 1;
  for (uint32_t d = 0; d < MAX_DIMS; ++d) {
    arg.tiles[d] = kTiles[mc_ndim - 1][d];
    arg.grids[d] = kGrids[mc_ndim - 1][d];
    arg.ntiles  *= arg.grids[d];
    elems       *= arg.tiles[d];
  }
  arg.box        = elems * elem;
  arg.tstride    = (arg.box + 63) / 64 * 64 + 64;
  arg.lmem_bytes = LM_TILE + mc_copies * arg.tstride;
  arg.linger     = kLinger;
  arg.neg_off    = (uint32_t)footprint(arg.lmem_bytes) - 64;   // the last slot pitch of the block

  const bool spread = legacy ? false : mc_spread;
  if (mc_neg) {
    std::printf("dxa_copy_mcast: negative run '%s', SPREAD cluster of 2 blocks, %u cores of %u warps x %u threads\n",
                kNegNames[mc_neg], (uint32_t)num_cores, (uint32_t)num_warps, nt);
  } else {
    std::printf("dxa_copy_mcast: %s multicast, mask %#x, %s, %u clusters of %u blocks of %u warps x %u threads,"
                " issuer rank %u%s, %u cores of %u warps, %u copies of a rank-%u box of %u bytes (%u-byte"
                " elements), %u bytes of local memory per block\n", legacy ? "legacy" : "cluster", mc_mask,
                spread ? "SPREAD" : "DEFAULT", mc_g, k, mc_w, nt, legacy ? 0 : mc_issuer,
                mc_helper ? " (helper)" : "", (uint32_t)num_cores, (uint32_t)num_warps, mc_copies, mc_ndim,
                arg.box, elem, arg.lmem_bytes);
  }
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA;
  if ((isa & need) != need) {
    std::printf("Error: the checked modes need DLMEM, the A extension, mbarriers and the DXA\nFAILED!\n");
    return 1;
  }
  const bool fits = spread ? (k <= max_cluster && num_cores >= 2 && mc_w <= num_warps
                              && arg.lmem_bytes <= lmem_size)
                           : (k * mc_w <= num_warps && k * footprint(arg.lmem_bytes) <= lmem_size);
  if (!fits) {
    std::printf("Error: %s clusters of %u blocks of %u warps and %u bytes do not fit %u cores of %u warps and"
                " %u bytes (largest cluster %u)\nFAILED!\n", spread ? "SPREAD" : "DEFAULT", k, mc_w,
                arg.lmem_bytes, (uint32_t)num_cores, (uint32_t)num_warps, (uint32_t)lmem_size,
                (uint32_t)max_cluster);
    return 1;
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
  const bool barrier_mc = legacy || mc_neg == NEG_SPREAD;
  if (program_desc(dev, arg, src_addr, elem) != 0
      || (barrier_mc && vortex::dxa::set_multicast(dev, DESC_SLOT, (uint32_t)footprint(arg.lmem_bytes)) != 0)) {
    std::printf("Error: cannot program the DXA descriptor\nFAILED!\n");
    return 1;
  }

  const uint32_t blocks = mc_g * k;
  std::vector<uint32_t> recs((size_t)blocks * REC_WORDS, ~0u);
  for (uint32_t b = 0; b < blocks; ++b)
    recs[(size_t)b * REC_WORDS + REC_RAN] = 0;
  std::vector<uint32_t> errs((size_t)blocks * threads * ERR_WORDS, 0);
  std::vector<uint32_t> dump((size_t)blocks * arg.lmem_bytes / sizeof(uint32_t), 0xDEADBEEFu);
  vx_buffer_h rec_buf  = upload(dev, q, recs, &arg.rec_addr, VX_MEM_READ_WRITE);
  vx_buffer_h err_buf  = upload(dev, q, errs, &arg.err_addr, VX_MEM_READ_WRITE);
  vx_buffer_h dump_buf = upload(dev, q, dump, &arg.dump_addr, VX_MEM_READ_WRITE);

  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, mc_neg ? "neg_k" : legacy ? "legacy_k" : "cluster_k", &kern));
  vx_launch_info_t li = {};
  li.struct_size    = sizeof(li);
  li.kernel         = kern;
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
  if (mc_verbose || mc_neg) {
    for (uint32_t g = 0; g < mc_g; ++g) {
      std::printf("cluster %u (core:slot by rank):", g);
      for (uint32_t r = 0; r < k; ++r)
        std::printf(" %u:%u", rec(g * k + r)[REC_CORE], rec(g * k + r)[REC_SLOT]);
      std::printf("\n");
    }
  }

  if (mc_neg) {
    std::printf("*** the negative run '%s' completed: the simulator must stop it\n", kNegNames[mc_neg]);
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
    std::printf("dxa_copy_mcast: %lu waits, %u threads gave up, %u unfinished\n", (unsigned long)tries, gave_up,
                unfinished);
    if (gave_up)
      std::printf("*** mbarrier phase did not complete: %u threads gave up after %u waits\n", gave_up, WAIT_LIMIT);
    errors += (gave_up != 0) + (unfinished != 0);

    // The legacy contract needs each cluster in consecutive slots of one core.
    if (legacy) {
      for (uint32_t g = 0; g < mc_g; ++g) {
        for (uint32_t r = 1; r < k; ++r) {
          const uint32_t* r0 = rec(g * k);
          const uint32_t* rr = rec(g * k + r);
          if (rr[REC_CORE] != r0[REC_CORE] || rr[REC_SLOT] != r0[REC_SLOT] + r) {
            std::printf("*** cluster %u: rank %u at %u:%u, not in the slot after rank 0 at %u:%u\n", g, r,
                        rr[REC_CORE], rr[REC_SLOT], r0[REC_CORE], r0[REC_SLOT]);
            ++errors;
          }
        }
      }
    }

    std::vector<std::vector<uint8_t>> tiles(arg.ntiles);
    for (uint32_t t = 0; t < arg.ntiles; ++t)
      tiles[t] = tile_bytes(arg, t, elem);
    uint32_t cross = 0, same = 0, bad_bytes = 0, bad_blocks = 0;
    for (uint32_t b = 0; b < blocks; ++b) {
      const uint32_t r = b % k, g = b / k;
      const bool recv = (mc_mask >> r) & 1u;
      std::vector<uint8_t> want(arg.lmem_bytes);
      for (uint32_t i = 0; i < arg.lmem_bytes; ++i)
        want[i] = canary_byte(b, i);
      if (recv) {
        const uint32_t issuer = g * k + (legacy ? 0 : mc_issuer);
        if (rec(b)[REC_CORE] == rec(issuer)[REC_CORE])
          ++same;
        else
          ++cross;
        for (uint32_t i = 0; i < mc_copies; ++i) {
          const std::vector<uint8_t>& t = tiles[src_tile(g, i, mc_copies, arg.ntiles)];
          std::memcpy(&want[LM_TILE + i * arg.tstride], t.data(), t.size());
        }
      }
      const uint8_t* got = reinterpret_cast<const uint8_t*>(dump.data()) + (size_t)b * arg.lmem_bytes;
      uint32_t first = 0;
      if (!legacy) {
        // A receiver's object completed one phase; a non-receiver's is as initialized.
        const uint32_t count = mc_helper ? mc_copies : 1;
        uint64_t w = mr::init(count);
        if (recv) {
          for (uint32_t i = 0; i < mc_copies; ++i) {
            if (mc_helper || i == 0)
              w = mr::apply(mr::OP_ARRIVE_TX, w, (mc_helper ? 1 : mc_copies) * arg.box).new_word;
          }
          for (uint32_t i = 0; i < mc_copies; ++i)
            w = mr::apply(mr::OP_COMPLETE_TX, w, arg.box).new_word;
        }
        uint64_t got_w;
        std::memcpy(&got_w, got, sizeof(got_w));
        if (got_w != w) {
          std::printf("*** block %u (rank %u, %s): mbarrier word %#018lx, expected %#018lx\n", b, r,
                      recv ? "receiver" : "not a receiver", (unsigned long)got_w, (unsigned long)w);
          ++errors;
        }
        first = 8;
      }
      uint32_t bad = 0;
      for (uint32_t i = first; i < arg.lmem_bytes; ++i) {
        if (got[i] == want[i])
          continue;
        if (bad++ < 4 && bad_blocks < 4) {
          const uint32_t s = i < LM_TILE ? ~0u : (i - LM_TILE) / arg.tstride;
          const uint32_t o = i < LM_TILE ? i : (i - LM_TILE) % arg.tstride;
          std::printf("*** block %u (rank %u, %s, core %u slot %u) byte %u (tile slot %d, offset %u%s): %#04x,"
                      " expected %#04x\n", b, r, recv ? "receiver" : "not a receiver", rec(b)[REC_CORE],
                      rec(b)[REC_SLOT], i, (int)s, o, o >= arg.box ? ", past the tile" : "", got[i], want[i]);
        }
      }
      if (bad) {
        ++bad_blocks;
        bad_bytes += bad;
      }
    }
    std::printf("dxa_copy_mcast: %u receivers on other cores than the issuer, %u on its core; %u wrong bytes in"
                " %u blocks\n", cross, same, bad_bytes, bad_blocks);
    errors += (bad_bytes != 0);
    if (spread && cross == 0 && (mc_mask & ~(1u << mc_issuer)) != 0) {
      std::printf("*** no receiver on another core than the issuer in a SPREAD launch\n");
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
  vx_buffer_release(src_buf);
  vx_kernel_release(kern);
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
