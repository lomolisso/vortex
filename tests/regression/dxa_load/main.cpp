// dxa_load: PTX tensor loads on the DXA (cp.async.bulk.tensor.{1..5}d
// .shared::cluster.global.{tile,im2col}.mbarrier::complete_tx::bytes, and
// cp.async.bulk.prefetch.tensor), on a DLMEM device with the A extension,
// mbarriers and the DXA (RV64). Every block fills its local memory with
// canaries, arms its mbarrier with the bytes it will receive, and after the
// cluster barrier warp 0 of each issuing rank issues the run's loads; the
// block waits for the phase and dumps its local memory. The host checks every
// byte of every block against sw/common/dxa_ref.h (box, fill, swizzle, im2col
// walk, canaries) and the final mbarrier word against mbar_ref. Runs whose
// boxes do not fit one local memory or 16 descriptors are split over launches.
//
//   -m edge     2D boxes that pass the end of dim 0 and dim 1 (zero fill)
//   -m halo     2D boxes at coordinate -1 (and further) in dims 0 and 1
//   -m oob5     5D boxes larger than the tensor, out of bounds on both sides
//               of every dim
//   -m nan      OOB-NaN fill, f16, f32 and f64
//   -m es       5D element strides 1-8 in dims 1-4, with fill
//   -m swz      swizzle -s 32|64|128 (default 128) x element 1/2/4/8 x inner
//               widths 16 B up to the span, in bounds and with fill
//   -m il16     16B interleave, dim-0 element strides 1, 2 and 4
//   -m il32     32B interleave with 32B swizzle, dim-0 element stride 2
//   -m i2c3|i2c4|i2c5  im2col at rank 3, 4, 5: corners, offsets, traversal
//               strides, channel and image fill, 64B swizzle (i2c4)
//   -m i2c-ptx  the three im2col examples of PTX ISA §5.5.4
//   -m mc       cluster multicast with 128B swizzle (cluster of 2 on 2 cores),
//               masks 0b11 and 0b10 (issuer left out)
//   -m win      a window destination and raw window mbarrier on the peer
//   -m selfwin  each block loads into itself through its own window address
//   -m pf       tensor prefetch (tile and im2col), then the same loads
//   -m swz-oob  two 2D boxes with 128B swizzle and fill, before and past the
//               tensor (the parity case)
//   -x neg      a run that must stop the simulator: mbar-align (an mbarrier 4
//               bytes off), swz-align (a 128B-swizzled box 16 bytes off a
//               128-byte boundary), swz-span (a 256-byte inner box with 128B
//               swizzle), es0 (a dim-0 element stride without interleave), il32
//               (32B interleave without swizzle), layout (K-major with swizzle)
//   -p          perf (the parity cells): no canaries and no dump; warp 0
//               issues the mode's loads 8 times, one mbarrier phase each, and
//               times the loop; thread 0 checks spot bytes of the last repeat
//               against the host image and the final mbarrier word. Modes of
//               one block only.
//   -v          print every load
//
// A device without the extensions, or with too few cores for the mode, prints
// SKIPPED when the run was launched without options, and fails otherwise.

#include <vortex2.h>
#include <dxa.h>
#include <VX_types.h>
#include <dxa_ref.h>
#include <mbar_ref.h>
#include "common.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace dr = vortex::dxa_ref;
namespace mr = vortex::mbar_ref;
namespace dh = vortex::dxa;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

enum Mode { M_EDGE, M_HALO, M_OOB5, M_NAN, M_ES, M_SWZ, M_IL16, M_IL32, M_I2C3, M_I2C4, M_I2C5, M_I2CPTX,
            M_MC, M_WIN, M_SELFWIN, M_PF, M_SWZOOB, NUM_MODES };
const char* const kModeNames[] = {"edge", "halo", "oob5", "nan", "es", "swz", "il16", "il32", "i2c3", "i2c4",
                                  "i2c5", "i2c-ptx", "mc", "win", "selfwin", "pf", "swz-oob"};
enum Neg { N_NONE, N_MBAR_ALIGN, N_SWZ_ALIGN, N_SWZ_SPAN, N_ES0, N_IL32, N_LAYOUT, NUM_NEGS };
const char* const kNegNames[] = {"", "mbar-align", "swz-align", "swz-span", "es0", "il32", "layout"};

constexpr uint32_t kGBytes = 512 * 1024;
constexpr uint32_t kLinger = 20000;
constexpr uint32_t kGap    = 64;    // canary bytes after every box at least
constexpr uint32_t kSlots  = VX_DCR_DXA_DESC_COUNT;
constexpr uint32_t kPerfRepeats = 8;
constexpr uint32_t kSpots  = 64;    // -p: spot bytes per launch

const char* kernel_file = "kernel.vxbin";
uint32_t mode = M_EDGE;
uint32_t neg  = N_NONE;
uint32_t swz_span = 128;
bool verbose = false;
bool perf = false;
bool requested = false;

void usage() {
  std::printf("Usage: [-m edge|halo|oob5|nan|es|swz|il16|il32|i2c3|i2c4|i2c5|i2c-ptx|mc|win|selfwin|pf|swz-oob]"
              " [-s 32|64|128] [-x mbar-align|swz-align|swz-span|es0|il32|layout] [-p] [-v] [-h]\n");
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
  while ((c = getopt(argc, argv, "m:s:x:pvh")) != -1) {
    switch (c) {
      case 'm': mode = parse_name(optarg, kModeNames, 0, NUM_MODES); break;
      case 's': swz_span = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'x': neg = parse_name(optarg, kNegNames, 1, NUM_NEGS); break;
      case 'p': perf = true; break;
      case 'v': verbose = true; break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || (swz_span != 32 && swz_span != 64 && swz_span != 128) || (perf && neg)) {
    usage();
    std::exit(-1);
  }
}

uint32_t round_up(uint32_t x, uint32_t a) {
  return (x + a - 1) / a * a;
}

uint32_t swz_mode(uint32_t span) {
  return span == 32 ? dr::SWZ_32B : span == 64 ? dr::SWZ_64B : dr::SWZ_128B;
}

// One load of the run, with its decoded descriptor.
struct Load {
  dr::Desc d;
  dr::Req  q;
  job_t    j;
};

struct Batch {
  std::vector<dr::Desc> descs;  // by slot
  std::vector<Load> loads;
  uint32_t cursor = LM_DATA;
  uint32_t lmem_bytes = 0;
};

struct Plan {
  uint32_t k = 1;              // blocks per cluster
  std::vector<Batch> batches;
  uint32_t cap = 0;            // local memory per block
  uint64_t gbase = 0;
  uint32_t next_tensor = 0;    // tensor placement counter
  std::string error;

  Batch& cur() {
    if (batches.empty())
      batches.emplace_back();
    return batches.back();
  }

  // The slot of d in batch b, or -1 when the batch has none free.
  int slot_of(Batch& b, const dr::Desc& d) {
    uint32_t w[dr::DESC_WORDS] = {}, v[dr::DESC_WORDS] = {};
    dr::encode(d, w);
    for (uint32_t s = 0; s < b.descs.size(); ++s) {
      dr::encode(b.descs[s], v);
      if (!std::memcmp(w, v, sizeof(w)))
        return (int)s;
    }
    if (b.descs.size() >= kSlots)
      return -1;
    b.descs.push_back(d);
    return (int)b.descs.size() - 1;
  }
};

// Lays a tensor's elements out densely (rows padded to 16 bytes) and places
// its base in the global buffer; every in-bounds byte stays inside it.
void place_tensor(Plan& p, dr::Desc& d, uint32_t row_pad = 16) {
  d.stride[0] = round_up(d.size[0] * d.elem + row_pad, 16);
  for (uint32_t i = 1; i + 1 < d.rank; ++i)
    d.stride[i] = d.stride[i - 1] * d.size[i];
  uint64_t extent = (uint64_t)d.size[0] * d.elem;
  for (uint32_t i = 1; i < d.rank; ++i)
    extent += (uint64_t)(d.size[i] - 1) * d.stride[i - 1];
  uint64_t off = (uint64_t)(p.next_tensor++ * 4112u) % (kGBytes / 2);
  off = off / 16 * 16;
  if (off + extent > kGBytes)
    off = 0;
  if (extent > kGBytes)
    p.error = "a tensor is larger than the global buffer";
  d.base = p.gbase + off;
}

dr::Desc tile_desc(uint32_t rank, uint32_t elem, std::initializer_list<uint32_t> size,
                   std::initializer_list<uint32_t> tile) {
  dr::Desc d;
  d.rank = rank;
  d.elem = elem;
  uint32_t i = 0;
  for (uint32_t s : size)
    d.size[i++] = s;
  i = 0;
  for (uint32_t t : tile)
    d.tile[i++] = t;
  for (uint32_t r = rank; r < 5; ++r)
    d.tile[r] = 0;
  return d;
}

// Adds a load into the current batch, opening a new one when its slots or
// local memory run out. dst_align: 0 = 128 with swizzle (an odd multiple, so
// the swizzle starts mid-pattern), else 16.
void add(Plan& p, const dr::Desc& d, std::initializer_list<int32_t> coord, uint32_t kind = JOB_TLOAD,
         std::initializer_list<uint32_t> offs = {}, uint32_t issuer = 0, uint32_t dst_rank = DST_LOCAL,
         uint32_t mask = 0) {
  Load l;
  l.d = d;
  uint32_t i = 0;
  for (int32_t c : coord)
    l.q.coord[i++] = c;
  i = 0;
  for (uint32_t o : offs)
    l.q.offs[i++] = o;
  l.j = {};
  l.j.kind = kind;
  l.j.rank = d.rank;
  l.j.im2col = d.im2col;
  l.j.issuer = issuer;
  l.j.dst_rank = dst_rank;
  l.j.mask = mask;
  for (uint32_t c = 0; c < 5; ++c)
    l.j.coord[c] = l.q.coord[c];
  for (uint32_t c = 0; c < 3; ++c)
    l.j.offs[c] = l.q.offs[c];
  for (int attempt = 0; attempt < 2; ++attempt) {
    Batch& b = p.cur();
    const int slot = p.slot_of(b, d);
    uint32_t dst = b.cursor, lo = 0, hi = 0;
    if (d.swizzle != dr::SWZ_NONE) {
      dst = round_up(dst, 128);
      if ((dst / 128) % 2 == 0)
        dst += 128;
    } else {
      dst = round_up(dst, 16);
    }
    if (kind == JOB_TLOAD)
      dr::footprint(d, dst, &lo, &hi);
    else
      hi = dst;
    if (slot >= 0 && (hi + kGap <= p.cap || b.loads.empty())) {
      l.j.slot = (uint32_t)slot;
      l.j.dst_off = dst;
      if (kind == JOB_TLOAD)
        b.cursor = hi + kGap;
      b.loads.push_back(l);
      return;
    }
    if (slot >= 0 && b.descs.size() - 1 == (uint32_t)slot && !b.loads.empty()) {
      bool used = false;
      for (const Load& x : b.loads)
        used |= (x.j.slot == (uint32_t)slot);
      if (!used)
        b.descs.pop_back();
    }
    p.batches.emplace_back();
  }
  p.error = "a load does not fit one local memory";
}

void plan_mode(Plan& p) {
  switch (mode) {
  case M_EDGE: {
    // Dim 0 runs 3 elements past size0; then both dims past their end.
    dr::Desc d = tile_desc(2, 4, {13, 6}, {16, 4});
    place_tensor(p, d);
    add(p, d, {0, 0});
    add(p, d, {8, 3});
    dr::Desc e = tile_desc(2, 1, {37, 5}, {32, 2});
    place_tensor(p, e);
    add(p, e, {16, 4});
    break;
  }
  case M_HALO: {
    // Signed coordinates: a box starting below 0 loads its in-bounds part.
    dr::Desc d = tile_desc(2, 2, {20, 10}, {16, 8});
    place_tensor(p, d);
    add(p, d, {-1, -1});
    add(p, d, {-3, 4});
    add(p, d, {8, -1});
    add(p, d, {-15, -7});
    add(p, d, {-16, 0});
    break;
  }
  case M_OOB5: {
    dr::Desc d = tile_desc(5, 1, {5, 3, 3, 2, 2}, {8, 5, 5, 4, 4});
    place_tensor(p, d);
    add(p, d, {-2, -1, -1, -1, -1});
    dr::Desc e = tile_desc(5, 2, {6, 2, 2, 2, 2}, {8, 3, 3, 3, 3});
    place_tensor(p, e);
    add(p, e, {-1, -1, -1, -1, -1});
    add(p, e, {3, 1, 1, 1, 1});
    break;
  }
  case M_NAN: {
    dr::Desc h = tile_desc(2, 2, {10, 5}, {16, 8});
    h.cfill = dr::nan_fill(2);
    place_tensor(p, h);
    add(p, h, {-2, -1});
    dr::Desc f = tile_desc(3, 4, {6, 3, 2}, {8, 4, 3});
    f.cfill = dr::nan_fill(4);
    place_tensor(p, f);
    add(p, f, {-1, -1, -1});
    dr::Desc g = tile_desc(2, 8, {3, 3}, {4, 4});
    g.cfill = dr::nan_fill(8);
    place_tensor(p, g);
    add(p, g, {-1, 0});
    break;
  }
  case M_ES: {
    const uint32_t es[3][5] = {{1, 2, 3, 4, 5}, {1, 6, 7, 8, 1}, {1, 8, 8, 8, 8}};
    const uint32_t tl[3][5] = {{4, 8, 7, 5, 4}, {4, 8, 7, 8, 4}, {4, 9, 9, 9, 9}};
    for (uint32_t v = 0; v < 3; ++v) {
      dr::Desc d = tile_desc(5, 4, {4, 12, 10, 9, 5}, {tl[v][0], tl[v][1], tl[v][2], tl[v][3], tl[v][4]});
      for (uint32_t i = 0; i < 5; ++i)
        d.estride[i] = es[v][i];
      place_tensor(p, d);
      add(p, d, {0, 1, 0, 2, 0});
      add(p, d, {-1, -3, 2, -4, 1});
    }
    break;
  }
  case M_SWZ: {
    const uint32_t repeat = swz_span * 8;  // 256 / 512 / 1024 bytes
    for (uint32_t elem = 1; elem <= 8; elem *= 2) {
      for (uint32_t ib = 16; ib <= swz_span; ib *= 2) {
        const uint32_t t0 = ib / elem, rows = repeat / ib;
        dr::Desc d = (ib == 32) ? tile_desc(3, elem, {t0 + 3, rows / 2 + 3, 3}, {t0, rows / 2, 2})
                                : tile_desc(2, elem, {t0 + 3, rows + 3}, {t0, rows});
        d.swizzle = swz_mode(swz_span);
        place_tensor(p, d);
        if (d.rank == 3) {
          add(p, d, {1, 2, 0});
          add(p, d, {-2, (int32_t)(rows / 2 - 1), 2});
        } else {
          add(p, d, {1, 2});
          add(p, d, {-2, (int32_t)rows - 1});
        }
      }
    }
    break;
  }
  case M_IL16: {
    dr::Desc a = tile_desc(3, 2, {20, 4, 3}, {16, 3, 2});
    a.interleave = dr::IL_16B;
    a.estride[0] = 2;
    place_tensor(p, a);
    add(p, a, {-2, 0, 0});
    add(p, a, {7, 2, 2});
    dr::Desc b = tile_desc(3, 2, {12, 4, 3}, {8, 3, 2});
    b.interleave = dr::IL_16B;
    place_tensor(p, b);
    add(p, b, {0, 1, -1});
    dr::Desc c = tile_desc(4, 1, {70, 3, 2, 2}, {64, 2, 2, 2});
    c.interleave = dr::IL_16B;
    c.estride[0] = 4;
    place_tensor(p, c);
    add(p, c, {9, 2, 0, 1});
    break;
  }
  case M_IL32: {
    dr::Desc a = tile_desc(4, 2, {40, 5, 3, 2}, {32, 4, 2, 2});
    a.interleave = dr::IL_32B;
    a.swizzle = dr::SWZ_32B;
    a.estride[0] = 2;
    place_tensor(p, a);
    add(p, a, {-3, 0, 0, 0});
    add(p, a, {12, 3, 2, 1});
    dr::Desc b = tile_desc(4, 2, {20, 9, 2, 1}, {16, 8, 2, 1});
    b.interleave = dr::IL_32B;
    b.swizzle = dr::SWZ_32B;
    place_tensor(p, b);
    add(p, b, {0, 2, 0, 0});
    dr::Desc c = tile_desc(5, 1, {66, 3, 2, 2, 2}, {64, 2, 2, 2, 2});
    c.interleave = dr::IL_32B;
    c.swizzle = dr::SWZ_32B;
    c.estride[0] = 2;
    place_tensor(p, c);
    add(p, c, {4, 2, 1, 0, 1});
    break;
  }
  case M_I2C3: {
    dr::Desc a;
    a.rank = 3; a.elem = 2; a.im2col = true;
    a.size[0] = 24; a.size[1] = 10; a.size[2] = 3;
    a.channels = 16; a.pixels = 40;
    a.lower[0] = -2; a.upper[0] = 1;
    a.estride[1] = 3;
    place_tensor(p, a);
    add(p, a, {12, 5, 1}, JOB_TLOAD, {2});
    dr::Desc b = a;
    b.estride[1] = 1;
    b.pixels = 30;
    b.channels = 8;
    place_tensor(p, b);
    add(p, b, {-4, -2, 0}, JOB_TLOAD, {0});
    break;
  }
  case M_I2C4: {
    dr::Desc a;
    a.rank = 4; a.elem = 4; a.im2col = true;
    a.size[0] = 16; a.size[1] = 7; a.size[2] = 6; a.size[3] = 2;
    a.channels = 8; a.pixels = 48;
    a.lower[0] = -1; a.lower[1] = -2;
    a.upper[0] = 0;  a.upper[1] = 1;
    a.estride[1] = 2;
    place_tensor(p, a);
    add(p, a, {4, 3, -1, 0}, JOB_TLOAD, {1, 0});
    dr::Desc b = a;
    b.channels = 16;
    b.pixels = 24;
    b.estride[1] = 1;
    b.estride[2] = 3;
    b.swizzle = dr::SWZ_64B;
    place_tensor(p, b);
    add(p, b, {0, -1, 2, 1}, JOB_TLOAD, {2, 1});
    break;
  }
  case M_I2C5: {
    dr::Desc a;
    a.rank = 5; a.elem = 2; a.im2col = true;
    a.size[0] = 8; a.size[1] = 5; a.size[2] = 4; a.size[3] = 3; a.size[4] = 2;
    a.channels = 8; a.pixels = 64;
    a.lower[0] = -1; a.lower[1] = -1; a.lower[2] = 0;
    a.upper[0] = 0;  a.upper[1] = 1;  a.upper[2] = -1;
    a.estride[3] = 2;
    place_tensor(p, a);
    add(p, a, {0, -1, 0, 1, 0}, JOB_TLOAD, {1, 0, 1});
    break;
  }
  case M_I2CPTX: {
    // PTX ISA §5.5.4: NHWC, C = 64, H = 14 (N = 9 here: the examples reach image 8).
    for (uint32_t ex = 0; ex < 3; ++ex) {
      dr::Desc d;
      d.rank = 4; d.elem = 2; d.im2col = true;
      d.size[0] = 64; d.size[1] = ex == 2 ? 8 : 9; d.size[2] = 14; d.size[3] = 9;
      d.channels = ex == 2 ? 16 : 8;
      d.pixels = ex == 2 ? 32 : 64;
      d.lower[0] = ex == 1 ? 0 : -1;  d.lower[1] = ex == 1 ? 0 : -1;
      d.upper[0] = ex == 1 ? -2 : -1; d.upper[1] = ex == 1 ? -2 : -1;
      d.estride[1] = d.estride[2] = ex == 2 ? 2 : 1;
      place_tensor(p, d, 0);
      const uint32_t o = ex == 0 ? 0 : ex == 1 ? 2 : 1;
      add(p, d, {0, ex == 2 ? 5 : 4, 7, 7}, JOB_TLOAD, {o, o});
    }
    break;
  }
  case M_MC: {
    p.k = 2;
    dr::Desc d = tile_desc(2, 2, {70, 12}, {64, 8});
    d.swizzle = dr::SWZ_128B;
    place_tensor(p, d);
    add(p, d, {-3, -2}, JOB_TLOAD, {}, 0, DST_LOCAL, 0x3);
    dr::Desc e = tile_desc(3, 4, {20, 9, 2}, {16, 4, 2});
    e.swizzle = dr::SWZ_64B;
    place_tensor(p, e);
    add(p, e, {5, 6, 0}, JOB_TLOAD, {}, 0, DST_LOCAL, 0x2);
    break;
  }
  case M_WIN: {
    p.k = 2;
    dr::Desc d = tile_desc(2, 4, {20, 7}, {16, 8});
    place_tensor(p, d);
    add(p, d, {-1, 2}, JOB_TLOAD, {}, 0, 1);
    dr::Desc e = tile_desc(2, 2, {66, 9}, {64, 8});
    e.swizzle = dr::SWZ_128B;
    place_tensor(p, e);
    add(p, e, {3, -1}, JOB_TLOAD, {}, 1, 0);
    break;
  }
  case M_SELFWIN: {
    p.k = 2;
    dr::Desc d = tile_desc(3, 2, {12, 5, 3}, {8, 4, 2});
    place_tensor(p, d);
    add(p, d, {-1, 2, 0}, JOB_TLOAD, {}, 0, 0);
    dr::Desc e = tile_desc(2, 1, {130, 9}, {128, 8});
    e.swizzle = dr::SWZ_128B;
    place_tensor(p, e);
    add(p, e, {5, 3}, JOB_TLOAD, {}, 1, 1);
    break;
  }
  case M_PF: {
    dr::Desc d = tile_desc(2, 4, {40, 30}, {16, 8});
    place_tensor(p, d);
    add(p, d, {30, -2}, JOB_TPREFETCH);
    add(p, d, {30, -2});
    dr::Desc a;
    a.rank = 4; a.elem = 2; a.im2col = true;
    a.size[0] = 16; a.size[1] = 6; a.size[2] = 5; a.size[3] = 2;
    a.channels = 8; a.pixels = 32;
    a.lower[0] = -1; a.lower[1] = -1;
    a.upper[0] = 0;  a.upper[1] = 0;
    place_tensor(p, a);
    add(p, a, {4, 1, 2, 0}, JOB_TPREFETCH, {1, 1});
    add(p, a, {4, 1, 2, 0}, JOB_TLOAD, {1, 1});
    break;
  }
  case M_SWZOOB: {
    dr::Desc d = tile_desc(2, 2, {70, 12}, {64, 8});
    d.swizzle = dr::SWZ_128B;
    place_tensor(p, d);
    add(p, d, {-3, -2});
    add(p, d, {10, 6});
    break;
  }
  default:
    break;
  }
}

void plan_neg(Plan& p) {
  dr::Desc d = tile_desc(2, 2, {64, 8}, {64, 4});
  d.swizzle = dr::SWZ_128B;
  switch (neg) {
  case N_MBAR_ALIGN:
    d.swizzle = dr::SWZ_NONE;
    break;
  case N_SWZ_SPAN:
    d.size[0] = 200;
    d.tile[0] = 128;
    break;
  case N_ES0:
    d.swizzle = dr::SWZ_NONE;
    d.estride[0] = 2;
    break;
  case N_IL32:
    d = tile_desc(3, 2, {16, 4, 2}, {16, 4, 2});
    d.interleave = dr::IL_32B;
    break;
  case N_LAYOUT:
    d.swizzle = dr::SWZ_32B;
    d.layout = dr::LAYOUT_KMAJOR;
    d.tile[0] = 16;
    break;
  default:
    break;
  }
  place_tensor(p, d);
  add(p, d, {0, 0});
  Load& l = p.batches.back().loads.back();
  if (neg == N_MBAR_ALIGN)
    l.j.mbar_skew = 4;
  if (neg == N_SWZ_ALIGN)
    l.j.dst_off += 16;
}

int program(vx_device_h dev, uint32_t slot, const dr::Desc& d) {
  int rc = 0;
  if (d.im2col) {
    const int32_t* lo = d.lower;
    const int32_t* up = d.upper;
    switch (d.rank) {
    case 3: rc = dh::program_im2col_3d(dev, slot, d.base, d.size, d.stride, d.channels, d.pixels, lo, up, d.elem); break;
    case 4: rc = dh::program_im2col_4d(dev, slot, d.base, d.size, d.stride, d.channels, d.pixels, lo, up, d.elem); break;
    default: rc = dh::program_im2col_5d(dev, slot, d.base, d.size, d.stride, d.channels, d.pixels, lo, up, d.elem); break;
    }
  } else {
    const uint32_t* s = d.size;
    const uint32_t* st = d.stride;
    const uint32_t* t = d.tile;
    switch (d.rank) {
    case 1: rc = dh::program_1d(dev, slot, d.base, s[0], t[0], d.elem); break;
    case 2: rc = dh::program_2d(dev, slot, d.base, s[0], s[1], st[0], t[0], t[1], d.elem); break;
    case 3: rc = dh::program_3d(dev, slot, d.base, s[0], s[1], s[2], st[0], st[1], t[0], t[1], t[2], d.elem); break;
    case 4: rc = dh::program_4d(dev, slot, d.base, s[0], s[1], s[2], s[3], st[0], st[1], st[2], t[0], t[1], t[2],
                                t[3], d.elem); break;
    default: rc = dh::program_5d(dev, slot, d.base, s[0], s[1], s[2], s[3], s[4], st[0], st[1], st[2], st[3], t[0],
                                 t[1], t[2], t[3], t[4], d.elem); break;
    }
  }
  if (rc)
    return rc;
  rc = dh::set_tensor_options(dev, slot, d.rank, d.elem, static_cast<dh::Swizzle>(d.swizzle),
                              static_cast<dh::Interleave>(d.interleave), d.cfill != 0, d.estride, d.im2col);
  if (rc)
    return rc;
  // The host API keeps the row-major layout; a negative run sets META directly.
  if (d.layout != dr::LAYOUT_ROW)
    rc = vx_dcr_write(dev, VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE + VX_DCR_DXA_DESC_META_OFF,
                      dr::pack_meta(d));
  return rc;
}

// The ranks a load lands in.
uint32_t receivers(const Load& l) {
  if (l.j.kind == JOB_TPREFETCH || l.j.kind == JOB_BPREFETCH)
    return 0;
  if (l.j.mask)
    return l.j.mask;
  return 1u << (l.j.dst_rank == DST_LOCAL ? l.j.issuer : l.j.dst_rank);
}

uint8_t canary_byte(uint32_t b, uint32_t o) {
  return (uint8_t)(canary_word(b, o / 4) >> (8 * (o & 3)));
}

// -p: block 0's image after the last repeat, as spot bytes spread over the
// bytes the loads write (rendered over two backgrounds: a byte both renders
// agree on is written), and the final mbarrier word.
uint64_t perf_image(const Plan& p, const Batch& bt, const std::vector<uint8_t>& gmem, std::vector<spot_t>* spots) {
  const uint32_t L = bt.lmem_bytes;
  std::vector<uint8_t> img0(L, 0x00), img1(L, 0xFF);
  auto rd = [&](uint64_t a) -> uint8_t {
    return (a >= p.gbase && a < p.gbase + kGBytes) ? gmem[a - p.gbase] : 0;
  };
  uint32_t tx = 0;
  std::vector<uint32_t> txs;
  for (const Load& l : bt.loads) {
    if (!(receivers(l) & 1u))
      continue;
    dr::render(l.d, l.q, l.j.dst_off, rd, img0.data(), L);
    txs.push_back(dr::render(l.d, l.q, l.j.dst_off, rd, img1.data(), L));
    tx += txs.back();
  }
  std::vector<uint32_t> written;
  for (uint32_t o = 8; o < L; ++o) {
    if (img0[o] == img1[o])
      written.push_back(o);
  }
  const size_t n = std::min<size_t>(kSpots, written.size());
  for (size_t i = 0; i < n; ++i) {
    const uint32_t o = written[i * written.size() / n];
    spots->push_back({o, img0[o]});
  }
  uint64_t mw = mr::init(1);
  for (uint32_t r = 0; r < kPerfRepeats && tx; ++r) {
    mw = mr::apply(mr::OP_ARRIVE_TX, mw, tx).new_word;
    for (uint32_t t : txs)
      mw = mr::apply(mr::OP_COMPLETE_TX, mw, t).new_word;
  }
  return mw;
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

  uint64_t num_cores = 0, nw = 0, nt = 0, lmem_cap = 0, max_cluster = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,         &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,         &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE,    &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &isa));

  // Every host buffer stays alive until the queue has finished.
  std::vector<uint8_t> gmem(kGBytes);
  for (uint32_t i = 0; i < kGBytes; ++i)
    gmem[i] = (uint8_t)(mix32(0x6D0000u, i / 4) >> (8 * (i & 3)));
  vx_buffer_h gbuf = nullptr;
  Plan plan;
  CHECK(vx_buffer_create(dev, kGBytes, VX_MEM_READ, &gbuf));
  CHECK(vx_buffer_address(gbuf, &plan.gbase));
  CHECK(vx_enqueue_write(q, gbuf, 0, gmem.data(), kGBytes, 0, nullptr, nullptr));
  plan.cap = (uint32_t)lmem_cap;

  if (neg)
    plan_neg(plan);
  else
    plan_mode(plan);
  const uint32_t k = plan.k;
  const uint32_t threads = (uint32_t)nt * 2;

  if (neg)
    std::printf("dxa_load: negative run '%s', %u warps x %u threads\n", kNegNames[neg], 2, (uint32_t)nt);
  else
    std::printf("dxa_load: %s%s, %zu launches, cluster of %u, %u cores of %u warps x %u threads, %u bytes of"
                " local memory\n", kModeNames[mode], mode == M_SWZ ? (" " + std::to_string(swz_span) + "B").c_str()
                : "", plan.batches.size(), k, (uint32_t)num_cores, (uint32_t)nw, (uint32_t)nt, (uint32_t)lmem_cap);

  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA;
  const char* skip = nullptr;
  if ((isa & need) != need)
    skip = "the device needs DLMEM, the A extension and the DXA";
  else if (nw < 2 || nt < 4 || (k > 1 && (num_cores < k || max_cluster < k)))
    skip = "the device has too few cores, warps or threads for the mode";
  else if (!plan.error.empty())
    skip = plan.error.c_str();
  else if (perf && k > 1)
    skip = "-p needs a mode of one block";
  if (skip) {
    std::printf("%s: %s\n", requested ? "Error" : "SKIPPED", skip);
    vx_buffer_release(gbuf);
    vx_queue_release(q);
    vx_device_release(dev);
    std::printf(requested ? "FAILED!\n" : "PASSED!\n");
    return requested ? 1 : 0;
  }

  vx_module_h mod = nullptr;
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "load_k", &kernel));

  int errors = 0;
  uint32_t total_loads = 0, total_bytes = 0;
  for (size_t bi = 0; bi < plan.batches.size(); ++bi) {
    Batch& bt = plan.batches[bi];
    uint32_t hi = LM_DATA;
    for (const Load& l : bt.loads) {
      if (l.j.kind != JOB_TLOAD)
        continue;
      uint32_t lo = 0, h = 0;
      dr::footprint(l.d, l.j.dst_off, &lo, &h);
      hi = std::max(hi, h);
    }
    bt.lmem_bytes = round_up(hi + kGap, 256);
    if (bt.lmem_bytes > lmem_cap)
      bt.lmem_bytes = (uint32_t)lmem_cap;
    for (uint32_t s = 0; s < bt.descs.size(); ++s) {
      if (program(dev, s, bt.descs[s]) != 0) {
        std::printf("Error: cannot program DXA descriptor %u\nFAILED!\n", s);
        return 1;
      }
    }

    load_arg_t arg = {};
    arg.num_jobs   = (uint32_t)bt.loads.size();
    arg.lmem_bytes = bt.lmem_bytes;
    arg.neg        = neg;
    arg.linger     = kLinger;
    arg.repeats    = perf ? kPerfRepeats : 0;
    std::vector<job_t> jobs;
    for (const Load& l : bt.loads) {
      jobs.push_back(l.j);
      const uint32_t rx = receivers(l);
      const uint32_t tx = l.j.kind == JOB_TLOAD ? dr::tx_bytes(l.d) : l.j.size;
      for (uint32_t r = 0; r < k && !neg; ++r) {
        if ((rx >> r) & 1u)
          arg.tx_in[r] += tx;
      }
      if (verbose || neg) {
        std::printf("  launch %zu: %s %uD%s slot %u issuer %u dst %s%u+%u mask %#x coords (%d, %d, %d, %d, %d)"
                    " offs (%u, %u, %u) elem %u swizzle %u interleave %u tx %u\n", bi,
                    l.j.kind == JOB_TLOAD ? "load" : "prefetch", l.d.rank, l.d.im2col ? " im2col" : "", l.j.slot,
                    l.j.issuer, l.j.dst_rank == DST_LOCAL ? "local " : "rank ",
                    l.j.dst_rank == DST_LOCAL ? l.j.issuer : l.j.dst_rank, l.j.dst_off, l.j.mask, l.q.coord[0],
                    l.q.coord[1], l.q.coord[2], l.q.coord[3], l.q.coord[4], l.q.offs[0], l.q.offs[1], l.q.offs[2],
                    l.d.elem, l.d.swizzle, l.d.interleave, tx);
      }
    }
    const uint32_t blocks = k;
    std::vector<uint32_t> errs((size_t)blocks * threads * ERR_WORDS, 0);
    std::vector<uint32_t> dump(perf ? 1 : (size_t)blocks * bt.lmem_bytes / 4, 0xDEADBEEFu);
    std::vector<spot_t> spots;
    uint64_t perf_mw = 0;
    if (perf)
      perf_mw = perf_image(plan, bt, gmem, &spots);
    arg.num_spots = (uint32_t)spots.size();
    vx_buffer_h jobs_buf = nullptr, err_buf = nullptr, dump_buf = nullptr, spot_buf = nullptr;
    CHECK(vx_buffer_create(dev, jobs.size() * sizeof(job_t), VX_MEM_READ, &jobs_buf));
    CHECK(vx_buffer_address(jobs_buf, &arg.jobs_addr));
    CHECK(vx_enqueue_write(q, jobs_buf, 0, jobs.data(), jobs.size() * sizeof(job_t), 0, nullptr, nullptr));
    CHECK(vx_buffer_create(dev, errs.size() * 4, VX_MEM_READ_WRITE, &err_buf));
    CHECK(vx_buffer_address(err_buf, &arg.err_addr));
    CHECK(vx_enqueue_write(q, err_buf, 0, errs.data(), errs.size() * 4, 0, nullptr, nullptr));
    CHECK(vx_buffer_create(dev, dump.size() * 4, VX_MEM_READ_WRITE, &dump_buf));
    CHECK(vx_buffer_address(dump_buf, &arg.dump_addr));
    CHECK(vx_enqueue_write(q, dump_buf, 0, dump.data(), dump.size() * 4, 0, nullptr, nullptr));
    if (perf && !spots.empty()) {
      CHECK(vx_buffer_create(dev, spots.size() * sizeof(spot_t), VX_MEM_READ, &spot_buf));
      CHECK(vx_buffer_address(spot_buf, &arg.spots_addr));
      CHECK(vx_enqueue_write(q, spot_buf, 0, spots.data(), spots.size() * sizeof(spot_t), 0, nullptr, nullptr));
    }

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
    li.lmem_size      = bt.lmem_bytes;
    li.cluster_dim[0] = k;
    li.cluster_dim[1] = 1;
    li.cluster_dim[2] = 1;
    li.cluster_policy = k > 1 ? VX_CLUSTER_POLICY_SPREAD : VX_CLUSTER_POLICY_DEFAULT;
    li.cluster_flags  = 0;

    vx_event_h launch_ev = nullptr, err_ev = nullptr, dump_ev = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, errs.data(), err_buf, 0, errs.size() * 4, 1, &launch_ev, &err_ev));
    CHECK(vx_enqueue_read(q, dump.data(), dump_buf, 0, dump.size() * 4, 1, &launch_ev, &dump_ev));
    CHECK(vx_event_wait_value(err_ev, 1, VX_TIMEOUT_INFINITE));
    CHECK(vx_event_wait_value(dump_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(dump_ev);
    vx_event_release(err_ev);
    vx_event_release(launch_ev);

    if (neg) {
      std::printf("*** the negative run '%s' completed: the simulator must stop it\n", kNegNames[neg]);
      ++errors;
    } else if (perf) {
      uint32_t unfinished = 0, gave_up = 0;
      for (size_t t = 0; t < (size_t)blocks * threads; ++t) {
        unfinished += (errs[t * ERR_WORDS + ERR_DONE] != 1);
        gave_up += errs[t * ERR_WORDS + ERR_WAIT];
      }
      const uint32_t cycles = errs[ERR_CYCLES];
      const uint64_t mw = ((uint64_t)errs[ERR_MBAR_HI] << 32) | errs[ERR_MBAR_LO];
      std::printf("dxa_load: launch %zu: %u loads x %u repeats, %u cycles from the first issue to the last"
                  " phase (%u a repeat)\n", bi, (uint32_t)bt.loads.size(), kPerfRepeats, cycles,
                  cycles / kPerfRepeats);
      if (unfinished || gave_up) {
        std::printf("*** launch %zu: %u threads unfinished, %u gave up waiting for the phase after %u try_waits\n",
                    bi, unfinished, gave_up, WAIT_LIMIT);
        ++errors;
      }
      if (errs[ERR_SPOTS]) {
        std::printf("*** launch %zu: %u of %zu spot bytes wrong\n", bi, errs[ERR_SPOTS], spots.size());
        ++errors;
      }
      if (mw != perf_mw) {
        std::printf("*** launch %zu: mbarrier word %#018lx, expected %#018lx\n", bi, (unsigned long)mw,
                    (unsigned long)perf_mw);
        ++errors;
      }
      for (const Load& l : bt.loads) {
        if (receivers(l)) {
          total_loads += kPerfRepeats;
          total_bytes += kPerfRepeats * (l.j.kind == JOB_TLOAD ? dr::tx_bytes(l.d) : l.j.size);
        }
      }
    } else {
      uint32_t unfinished = 0, gave_up = 0;
      for (size_t t = 0; t < (size_t)blocks * threads; ++t) {
        unfinished += (errs[t * ERR_WORDS + ERR_DONE] != 1);
        gave_up += errs[t * ERR_WORDS + ERR_WAIT];
      }
      if (unfinished || gave_up) {
        std::printf("*** launch %zu: %u threads unfinished, %u gave up waiting for the phase after %u try_waits\n",
                    bi, unfinished, gave_up, WAIT_LIMIT);
        ++errors;
      }
      const uint32_t L = bt.lmem_bytes;
      for (uint32_t b = 0; b < blocks; ++b) {
        std::vector<uint8_t> want(L);
        for (uint32_t o = 0; o < L; ++o)
          want[o] = canary_byte(b, o);
        uint64_t mw = mr::init(1);
        if (arg.tx_in[b])
          mw = mr::apply(mr::OP_ARRIVE_TX, mw, arg.tx_in[b]).new_word;
        bool bad_read = false;
        for (const Load& l : bt.loads) {
          if (!((receivers(l) >> b) & 1u))
            continue;
          auto rd = [&](uint64_t a) -> uint8_t {
            if (a < plan.gbase || a >= plan.gbase + kGBytes) {
              bad_read = true;
              return 0;
            }
            return gmem[a - plan.gbase];
          };
          const uint32_t tx = dr::render(l.d, l.q, l.j.dst_off, rd, want.data(), L);
          mw = mr::apply(mr::OP_COMPLETE_TX, mw, tx).new_word;
          ++total_loads;
          total_bytes += tx;
        }
        if (bad_read) {
          std::printf("*** block %u: the model read outside the global buffer\n", b);
          ++errors;
        }
        const uint8_t* got = reinterpret_cast<const uint8_t*>(dump.data()) + (size_t)b * L;
        uint64_t gw;
        std::memcpy(&gw, got + LM_MBAR, sizeof(gw));
        if (gw != mw) {
          std::printf("*** launch %zu block %u: mbarrier word %#018lx, expected %#018lx\n", bi, b,
                      (unsigned long)gw, (unsigned long)mw);
          ++errors;
        }
        uint32_t bad = 0;
        for (uint32_t o = 8; o < L; ++o) {
          if (got[o] == want[o])
            continue;
          if (bad++ < 8) {
            const char* where = "canary";
            for (const Load& l : bt.loads) {
              uint32_t lo = 0, h = 0;
              if (l.j.kind != JOB_TLOAD || !((receivers(l) >> b) & 1u))
                continue;
              dr::footprint(l.d, l.j.dst_off, &lo, &h);
              if (o >= lo && o < h)
                where = "box";
            }
            std::printf("*** launch %zu block %u byte %u (%s): %#04x, expected %#04x\n", bi, b, o, where, got[o],
                        want[o]);
          }
        }
        if (bad) {
          std::printf("*** launch %zu block %u: %u wrong bytes\n", bi, b, bad);
          ++errors;
        }
      }
    }
    if (spot_buf)
      vx_buffer_release(spot_buf);
    vx_buffer_release(dump_buf);
    vx_buffer_release(err_buf);
    vx_buffer_release(jobs_buf);
  }
  if (!neg)
    std::printf("dxa_load: %u boxes landed, %u bytes, %s\n", total_loads, total_bytes,
                perf ? "spot-checked" : "checked byte for byte");

  vx_kernel_release(kernel);
  vx_module_release(mod);
  vx_buffer_release(gbuf);
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
