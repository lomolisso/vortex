// dxa_store: PTX TMA stores and reductions on the DXA (cp.async.bulk.global
// .shared::cta.bulk_group, cp.async.bulk.tensor.{1..5}d.global.shared::cta
// .{tile,im2col_no_offs}.bulk_group, cp.reduce.async.bulk{,.tensor}.global
// .shared::cta.bulk_group.<op>.<type>), the bulk groups (commit_group,
// wait_group, wait_group.read) and fence.proxy.async.global, on a DLMEM
// device with the A extension, mbarriers and the DXA (RV64). One block fills
// its local memory with a pattern (and typed values for the reductions),
// fences it for the async proxy, and warp 0 issues the run's jobs, ending
// with commit + wait_group 0. The host keeps a model of the whole global
// buffer (canaries everywhere, typed values under the reductions) and checks
// it byte for byte against sw/common/dxa_ref.h after the run. Runs that need
// more than 16 descriptors use several launches.
//
//   -m bst      bulk stores of 16 B up to the local memory (64 KB with -z)
//   -m tst      1D-5D tile stores in bounds
//   -m edge     boxes past the end of the tensor in every dimension: the
//               elements outside are not written
//   -m swz      -s 32|64|128 (default 128) x element 1/2/4/8, in bounds and
//               past the end
//   -m swz-oob  two 2D boxes with 128B swizzle, one past both ends (the parity case)
//   -m es       5D element strides 1-5
//   -m il16     16B interleave, dim-0 element strides 1, 2 and 4
//   -m il32     32B interleave with 32B swizzle
//   -m i2c3|i2c4|i2c5  im2col_no_offs stores at rank 3, 4, 5
//   -m rg       bulk global reductions, -o op and -t type (default all: every
//               legal pair; with -t all the float, inc and dec pairs need FPAMO)
//   -m rt       tensor reductions with out-of-bounds edges, several pairs
//   -m grp      four groups of 1, 2, 3 and 1 bulk stores, then wait_group 3,
//               2, 1, 0, each followed by fence.proxy.async.global and a
//               generic load of the groups it completes; wait_group 31 and
//               wait_group.read 8 (at once); ten groups back to back (the
//               8-group stall); one group of 70 operations (the stall at 63
//               pending operations in the open group)
//   -m rdreuse  a group of a bulk store, a swizzled tensor store (and a bulk
//               reduction with an L2), wait_group.read 0, the sources
//               overwritten, wait_group 0: global holds the old bytes; then a
//               store of the new bytes
//   -m i7a      generic global stores, fence.proxy.async.global, then a bulk
//               load of them: it sees the new words
//   -m i7b      the region in the data cache, a bulk store over it, wait_group
//               0, fence.proxy.async.global, then a generic load: it sees the
//               stored bytes
//   -m i7c      as i7b, with the region's first 16 bytes stored generically
//               (a dirty line in a write-back data cache) and the rest by the
//               bulk store: the line's write-back keeps the bulk-stored bytes
//   -m exit     a 64 KB bulk store, wait_group.read 0, and the kernel ends:
//               the host still finds every byte written
//   -x neg      a run that must stop the simulator: neg (a tile store at
//               coordinate -1), nol2 (a global reduction without an L2 or L3),
//               nofp (add.f32 without FPAMO), badred (min.f32), tsize (add.u64
//               on 4-byte elements), enc (a STORE of kind 5), pfence (proxy
//               fence selector 3), layout (a tile store with a K-major
//               descriptor)
//   -p          perf (the parity cells): the mode's jobs repeated (-r, default
//               per mode), each repeat into fresh global regions and closed by
//               commit + wait_group 0; warp 0 times the loop. No fill of the
//               local memory and no check of the global buffer: the checked
//               runs are the other modes
//   -r n        -p: the repeats
//   -v          print every job
//
// A device without the extensions prints SKIPPED when the run was launched
// without options, and fails otherwise.

#include <vortex2.h>
#include <dxa.h>
#include <VX_types.h>
#include <dxa_ref.h>
#include "common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace dr = vortex::dxa_ref;
namespace dh = vortex::dxa;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

enum Mode { M_BST, M_TST, M_EDGE, M_SWZ, M_SWZOOB, M_ES, M_IL16, M_IL32, M_I2C3, M_I2C4, M_I2C5, M_RG, M_RT,
            M_GRP, M_RDREUSE, M_I7A, M_I7B, M_I7C, M_EXIT, NUM_MODES };
const char* const kModeNames[] = {"bst", "tst", "edge", "swz", "swz-oob", "es", "il16", "il32", "i2c3", "i2c4",
                                  "i2c5", "rg", "rt", "grp", "rdreuse", "i7a", "i7b", "i7c", "exit"};
enum Neg { N_NONE, N_NEG, N_NOL2, N_NOFP, N_BADRED, N_TSIZE, N_ENC, N_PFENCE, N_LAYOUT, NUM_NEGS };
const char* const kNegNames[] = {"", "neg", "nol2", "nofp", "badred", "tsize", "enc", "pfence", "layout"};

constexpr uint32_t kGBytes  = 1u << 20;
constexpr uint32_t kGap     = 64;     // canary bytes after every global region at least
constexpr uint32_t kLinger  = 20000;
constexpr uint32_t kSlots   = VX_DCR_DXA_DESC_COUNT;
constexpr uint32_t kMaxOps  = 32;     // operations per group the planner lets accumulate
constexpr uint32_t kI7Bytes = 4096;
constexpr uint32_t kExitBytes = 65536;

#ifdef VX_CFG_EXT_FPAMO_ENABLE
constexpr bool kFpamo = true;
#else
constexpr bool kFpamo = false;
#endif

const char* kernel_file = "kernel.vxbin";
uint32_t mode = M_BST;
uint32_t neg  = N_NONE;
uint32_t swz_span = 128;
uint32_t only_size = 0;
int32_t  sel_op = -1, sel_type = -1;
bool verbose = false;
bool perf = false;
uint32_t repeats = 0;
bool requested = false;

bool special_mode(uint32_t m) {
  return m == M_I7A || m == M_I7B || m == M_I7C || m == M_EXIT;
}

// -p: the repeats that make the timed loop most of the run.
uint32_t default_repeats(uint32_t m) {
  switch (m) {
  case M_RG: return (sel_op >= 0 && sel_type >= 0) ? 8 : 2;
  case M_RT: return 8;
  default:   return 32;
  }
}

void usage() {
  std::printf("Usage: [-m bst|tst|edge|swz|swz-oob|es|il16|il32|i2c3|i2c4|i2c5|rg|rt|grp|rdreuse|i7a|i7b|i7c|exit]"
              " [-s 32|64|128] [-z bytes] [-o add|min|max|inc|dec|and|or|xor|all]"
              " [-t u32|s32|u64|s64|f32|f64|f16|bf16|b32|b64|all]"
              " [-x neg|nol2|nofp|badred|tsize|enc|pfence|layout] [-p] [-r repeats] [-v] [-h]\n");
}

uint32_t parse_name(const char* s, const char* const* names, uint32_t first, uint32_t n) {
  for (uint32_t i = first; i < n; ++i) {
    if (!std::strcmp(s, names[i]))
      return i;
  }
  usage();
  std::exit(-1);
}

int32_t parse_op(const char* s) {
  if (!std::strcmp(s, "all"))
    return -1;
  for (uint32_t op = 0; op < dr::RED_OPS; ++op) {
    if (!std::strcmp(s, dr::red_op_name(op)))
      return (int32_t)op;
  }
  usage();
  std::exit(-1);
}

int32_t parse_type(const char* s) {
  if (!std::strcmp(s, "all"))
    return -1;
  if (!std::strcmp(s, "b32"))
    return dr::RT_U32;
  if (!std::strcmp(s, "b64"))
    return dr::RT_U64;
  for (uint32_t t = 0; t < dr::RT_TYPES; ++t) {
    if (!std::strcmp(s, dr::red_type_name(t)))
      return (int32_t)t;
  }
  usage();
  std::exit(-1);
}

void parse_args(int argc, char** argv) {
  requested = (argc > 1);
  int c;
  while ((c = getopt(argc, argv, "m:s:z:o:t:x:pr:vh")) != -1) {
    switch (c) {
      case 'm': mode = parse_name(optarg, kModeNames, 0, NUM_MODES); break;
      case 's': swz_span = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'z': only_size = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'o': sel_op = parse_op(optarg); break;
      case 't': sel_type = parse_type(optarg); break;
      case 'x': neg = parse_name(optarg, kNegNames, 1, NUM_NEGS); break;
      case 'p': perf = true; break;
      case 'r': repeats = (uint32_t)std::strtoul(optarg, nullptr, 10); break;
      case 'v': verbose = true; break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  const bool special = (mode == M_GRP || mode == M_RDREUSE || special_mode(mode));
  if (optind < argc || (swz_span != 32 && swz_span != 64 && swz_span != 128) || (perf && (neg || special))
      || (only_size && (only_size % 16 || mode != M_BST)) || (repeats && !perf)) {
    usage();
    std::exit(-1);
  }
  if (perf && repeats == 0)
    repeats = default_repeats(mode);
}

uint32_t round_up(uint32_t x, uint32_t a) {
  return (x + a - 1) / a * a;
}

uint32_t swz_mode(uint32_t span) {
  return span == 32 ? dr::SWZ_32B : span == 64 ? dr::SWZ_64B : dr::SWZ_128B;
}

struct Job {
  job_t    j = {};
  dr::Desc d;
  dr::Req  q;
};

struct Batch {
  std::vector<dr::Desc> descs;   // by slot
  std::vector<Job> jobs;
  std::vector<fill_t> fills;
  uint32_t lmem_hi = LM_DATA;
  uint32_t ops = 0;              // operations since the last commit
  uint32_t groups = 0;           // commits so far
  uint32_t repeats = 0;          // -p: the repeats in the launch
};

// A typed global region: initial values of a reduction's destination.
struct Typed {
  uint64_t off;
  uint32_t size;
  uint32_t type;
  uint32_t seed;
};

struct Plan {
  std::vector<Batch> batches;
  uint32_t cap = 0;              // local memory of the block
  uint64_t gbase = 0;
  uint32_t gcur = 0;             // next free byte of the global buffer
  uint32_t lcur = LM_DATA;       // next source offset
  uint32_t seed = 1;
  bool llc = false;
  bool raw = false;              // -x: checks of the plan are the hardware's
  std::vector<Typed> typed;
  std::string error;

  Batch& cur() {
    if (batches.empty())
      batches.emplace_back();
    return batches.back();
  }

  // A global region of `bytes`, `skew` bytes past a 64-byte boundary.
  uint64_t galloc(uint64_t bytes, uint32_t skew = 0) {
    const uint32_t off = round_up(gcur, 64) + skew;
    if ((uint64_t)off + bytes + kGap > kGBytes) {
      error = "the run does not fit the global buffer";
      return gbase;
    }
    gcur = (uint32_t)(off + bytes + kGap);
    return gbase + off;
  }

  // A local-memory source of `bytes`, `align`-aligned; 128-aligned sources
  // sit at odd multiples of 128 (the swizzle starts mid-pattern). When the
  // local memory is full, plain sources wrap and overlap (they are only read);
  // a typed source (`fresh`) opens a new launch instead, so that no later fill
  // overwrites it.
  uint32_t lalloc(uint32_t bytes, uint32_t align, uint32_t skew = 0, bool fresh = false) {
    for (int attempt = 0; attempt < 2; ++attempt) {
      uint32_t o = round_up(lcur, align);
      if (align == 128 && (o / 128) % 2 == 0)
        o += 128;
      o += skew;
      if (o + bytes <= cap) {
        lcur = o + bytes;
        return o;
      }
      lcur = LM_DATA;
      if (fresh && !batches.empty() && !batches.back().jobs.empty())
        batches.emplace_back();
    }
    error = "a source does not fit the local memory";
    return LM_DATA;
  }

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

  void push(Job& x) {
    Batch* b = &cur();
    if (x.j.kind == J_TSTORE || x.j.kind == J_TRED) {
      int slot = slot_of(*b, x.d);
      if (slot < 0) {
        batches.emplace_back();
        b = &batches.back();
        slot = slot_of(*b, x.d);
      }
      x.j.slot = (uint32_t)slot;
      uint32_t lo = 0, hi = 0;
      dr::footprint(x.d, x.j.src, &lo, &hi);
      b->lmem_hi = std::max(b->lmem_hi, hi);
    } else if (x.j.kind == J_BSTORE || x.j.kind == J_BRED || x.j.kind == J_REWRITE) {
      b->lmem_hi = std::max(b->lmem_hi, x.j.src + x.j.size);
    }
    const bool op = (x.j.kind <= J_TRED);
    if (op && b->ops == kMaxOps && !perf)
      commit();
    if (op) {
      x.j.group = b->groups;
      ++b->ops;
    }
    b->jobs.push_back(x);
  }

  void commit() {
    Job c;
    c.j.kind = J_COMMIT;
    Batch& b = cur();
    b.jobs.push_back(c);
    b.ops = 0;
    ++b.groups;
  }

  void wait(uint32_t kind, uint32_t n, bool check) {
    Job w;
    w.j.kind = kind;
    w.j.n = n;
    w.j.check = check;
    cur().jobs.push_back(w);
  }

  void fill(uint32_t src, uint32_t size, uint32_t type) {
    cur().fills.push_back({src, size, type, 0xF1000u + seed++});
  }
};

// Lays the tensor's rows out with 16 bytes of padding and places its base.
void place_tensor(Plan& p, dr::Desc& d, uint32_t skew = 0) {
  d.stride[0] = round_up(d.size[0] * d.elem + 16, 16);
  for (uint32_t i = 1; i + 1 < d.rank; ++i)
    d.stride[i] = d.stride[i - 1] * d.size[i];
  uint64_t extent = (uint64_t)d.size[0] * d.elem;
  for (uint32_t i = 1; i < d.rank; ++i)
    extent += (uint64_t)(d.size[i] - 1) * d.stride[i - 1];
  d.base = p.galloc(extent, skew);
}

uint64_t tensor_extent(const dr::Desc& d) {
  uint64_t extent = (uint64_t)d.size[0] * d.elem;
  for (uint32_t i = 1; i < d.rank; ++i)
    extent += (uint64_t)(d.size[i] - 1) * d.stride[i - 1];
  return extent;
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

// A tensor store (or reduction with op/type) of a fresh tensor of d's shape.
void add_tensor(Plan& p, dr::Desc d, std::initializer_list<int32_t> coord, int32_t op = -1, uint32_t type = 0) {
  place_tensor(p, d);
  Job x;
  x.d = d;
  uint32_t i = 0;
  for (int32_t c : coord)
    x.q.coord[i++] = c;
  x.j.kind = op < 0 ? J_TSTORE : J_TRED;
  x.j.op = op < 0 ? 0 : (uint32_t)op;
  x.j.type = type;
  x.j.rank = d.rank;
  x.j.im2col = d.im2col;
  for (uint32_t c = 0; c < 5; ++c)
    x.j.coord[c] = x.q.coord[c];
  x.j.gdst = d.base;
  uint32_t lo = 0, hi = 0;
  const uint32_t align = d.swizzle != dr::SWZ_NONE ? 128 : 16;
  dr::footprint(d, 0, &lo, &hi);
  const uint32_t span = d.swizzle != dr::SWZ_NONE ? round_up(hi, 128) + 128 : hi;
  x.j.src = p.lalloc(span, align, 0, op >= 0);
  p.push(x);
  if (op >= 0) {
    p.typed.push_back({d.base - p.gbase, (uint32_t)tensor_extent(d), type, 0x7E000u + p.seed++});
    uint32_t flo = 0, fhi = 0;
    dr::footprint(d, x.j.src, &flo, &fhi);
    p.fill(flo, (fhi - flo) / d.elem * d.elem, type);
  }
}

void add_bulk(Plan& p, uint32_t kind, uint32_t size, uint32_t skew_l, uint32_t skew_g, uint32_t op = 0,
              uint32_t type = 0) {
  Job x;
  x.j.kind = kind;
  x.j.op = op;
  x.j.type = type;
  x.j.size = size;
  x.j.src = p.lalloc(size, 64, skew_l, kind == J_BRED);
  x.j.gdst = p.galloc(size, skew_g);
  if (kind == J_BRED) {
    p.typed.push_back({x.j.gdst - p.gbase, size, type, 0x7E000u + p.seed++});
    p.fill(x.j.src, size, type);
  }
  p.push(x);
}

// The im2col descriptors of dxa_load, stored with all offsets 0.
dr::Desc i2c_desc(uint32_t rank, uint32_t v) {
  dr::Desc a;
  a.im2col = true;
  a.rank = rank;
  if (rank == 3) {
    a.elem = 2;
    a.size[0] = 24; a.size[1] = 10; a.size[2] = 3;
    a.channels = v ? 8 : 16;
    a.pixels = v ? 30 : 40;
    a.lower[0] = -2; a.upper[0] = 1;
    a.estride[1] = v ? 1 : 3;
  } else if (rank == 4) {
    a.elem = 4;
    a.size[0] = 16; a.size[1] = 7; a.size[2] = 6; a.size[3] = 2;
    a.channels = v ? 16 : 8;
    a.pixels = v ? 24 : 48;
    a.lower[0] = -1; a.lower[1] = -2;
    a.upper[0] = 0;  a.upper[1] = 1;
    a.estride[1] = v ? 1 : 2;
    a.estride[2] = v ? 3 : 1;
    if (v)
      a.swizzle = dr::SWZ_64B;
  } else {
    a.elem = 2;
    a.size[0] = 8; a.size[1] = 5; a.size[2] = 4; a.size[3] = 3; a.size[4] = 2;
    a.channels = 8;
    a.pixels = 64;
    a.lower[0] = -1; a.lower[1] = -1; a.lower[2] = 0;
    a.upper[0] = 0;  a.upper[1] = 1;  a.upper[2] = -1;
    a.estride[3] = 2;
  }
  return a;
}

std::string pair_name(uint32_t op, uint32_t type) {
  return std::string(dr::red_op_name(op)) + "." + dr::red_type_name(type, op);
}

uint32_t skipped_fpamo = 0;

bool usable(uint32_t op, uint32_t type) {
  if (dr::red_needs_fpamo(op, type) && !kFpamo) {
    ++skipped_fpamo;
    return false;
  }
  return true;
}

void plan_mode(Plan& p) {
  switch (mode) {
  case M_BST: {
    const uint32_t sizes[] = {16, 32, 48, 208, 720, 1024, 4096, 8192, 16384, 32768, 65536};
    uint32_t k = 0;
    for (uint32_t s : sizes) {
      if (only_size && s != only_size)
        continue;
      if (LM_DATA + 64 + s > p.cap)
        continue;
      add_bulk(p, J_BSTORE, s, 16 * (k % 4), 16 * ((3 * k + 1) % 4));
      ++k;
      add_bulk(p, J_BSTORE, s, 16 * (k % 4), 16 * ((3 * k + 1) % 4));
      ++k;
    }
    if (only_size && k == 0)
      p.error = "the size does not fit the local memory";
    break;
  }
  case M_TST:
    add_tensor(p, tile_desc(1, 4, {64}, {32}), {16});
    add_tensor(p, tile_desc(2, 2, {40, 12}, {16, 8}), {3, 2});
    add_tensor(p, tile_desc(3, 1, {70, 6, 4}, {32, 4, 2}), {17, 1, 2});
    add_tensor(p, tile_desc(4, 8, {9, 5, 4, 3}, {4, 3, 2, 2}), {5, 2, 1, 1});
    add_tensor(p, tile_desc(5, 4, {6, 5, 4, 3, 3}, {4, 2, 3, 2, 2}), {1, 3, 0, 1, 1});
    break;
  case M_EDGE:
    add_tensor(p, tile_desc(1, 4, {37}, {16}), {30});
    add_tensor(p, tile_desc(2, 4, {13, 6}, {16, 4}), {0, 0});
    add_tensor(p, tile_desc(2, 4, {13, 6}, {16, 4}), {8, 3});
    add_tensor(p, tile_desc(2, 1, {37, 5}, {32, 2}), {16, 4});
    add_tensor(p, tile_desc(3, 2, {10, 7, 3}, {8, 4, 2}), {6, 5, 2});
    add_tensor(p, tile_desc(5, 1, {5, 3, 3, 2, 2}, {8, 5, 5, 4, 4}), {0, 0, 0, 0, 0});
    add_tensor(p, tile_desc(5, 2, {6, 2, 2, 2, 2}, {8, 3, 3, 3, 3}), {3, 1, 1, 1, 1});
    break;
  case M_SWZ: {
    const uint32_t repeat = swz_span * 8;
    for (uint32_t elem = 1; elem <= 8; elem *= 2) {
      for (uint32_t ib = 16; ib <= swz_span; ib *= 2) {
        const uint32_t t0 = ib / elem, rows = repeat / ib;
        dr::Desc d = (ib == 32) ? tile_desc(3, elem, {t0 + 3, rows / 2 + 3, 3}, {t0, rows / 2, 2})
                                : tile_desc(2, elem, {t0 + 3, rows + 3}, {t0, rows});
        d.swizzle = swz_mode(swz_span);
        if (d.rank == 3) {
          add_tensor(p, d, {1, 2, 0});
          add_tensor(p, d, {4, (int32_t)(rows / 2 - 1), 1});
        } else {
          add_tensor(p, d, {1, 2});
          add_tensor(p, d, {4, (int32_t)rows - 1});
        }
      }
    }
    break;
  }
  case M_SWZOOB: {
    dr::Desc d = tile_desc(2, 2, {70, 12}, {64, 8});
    d.swizzle = dr::SWZ_128B;
    add_tensor(p, d, {0, 0});
    add_tensor(p, d, {10, 6});
    break;
  }
  case M_ES: {
    const uint32_t es[3][5] = {{1, 2, 3, 4, 5}, {1, 6, 7, 8, 1}, {1, 8, 8, 8, 8}};
    const uint32_t tl[3][5] = {{4, 8, 7, 5, 4}, {4, 8, 7, 8, 4}, {4, 9, 9, 9, 9}};
    for (uint32_t v = 0; v < 3; ++v) {
      dr::Desc d = tile_desc(5, 4, {4, 12, 10, 9, 5}, {tl[v][0], tl[v][1], tl[v][2], tl[v][3], tl[v][4]});
      for (uint32_t i = 0; i < 5; ++i)
        d.estride[i] = es[v][i];
      add_tensor(p, d, {0, 1, 0, 2, 0});
      add_tensor(p, d, {1, 3, 2, 4, 1});
    }
    break;
  }
  case M_IL16: {
    dr::Desc a = tile_desc(3, 2, {20, 4, 3}, {16, 3, 2});
    a.interleave = dr::IL_16B;
    a.estride[0] = 2;
    add_tensor(p, a, {0, 0, 0});
    add_tensor(p, a, {7, 2, 2});
    dr::Desc b = tile_desc(3, 2, {12, 4, 3}, {8, 3, 2});
    b.interleave = dr::IL_16B;
    add_tensor(p, b, {0, 1, 1});
    dr::Desc c = tile_desc(4, 1, {70, 3, 2, 2}, {64, 2, 2, 2});
    c.interleave = dr::IL_16B;
    c.estride[0] = 4;
    add_tensor(p, c, {9, 2, 0, 1});
    break;
  }
  case M_IL32: {
    dr::Desc a = tile_desc(4, 2, {40, 5, 3, 2}, {32, 4, 2, 2});
    a.interleave = dr::IL_32B;
    a.swizzle = dr::SWZ_32B;
    a.estride[0] = 2;
    add_tensor(p, a, {0, 0, 0, 0});
    add_tensor(p, a, {12, 3, 2, 1});
    dr::Desc b = tile_desc(4, 2, {20, 9, 2, 1}, {16, 8, 2, 1});
    b.interleave = dr::IL_32B;
    b.swizzle = dr::SWZ_32B;
    add_tensor(p, b, {0, 2, 0, 0});
    dr::Desc c = tile_desc(5, 1, {66, 3, 2, 2, 2}, {64, 2, 2, 2, 2});
    c.interleave = dr::IL_32B;
    c.swizzle = dr::SWZ_32B;
    c.estride[0] = 2;
    add_tensor(p, c, {4, 2, 1, 0, 1});
    break;
  }
  case M_I2C3:
    add_tensor(p, i2c_desc(3, 0), {12, 5, 1});
    add_tensor(p, i2c_desc(3, 1), {-4, -2, 0});
    break;
  case M_I2C4:
    add_tensor(p, i2c_desc(4, 0), {4, 3, -1, 0});
    add_tensor(p, i2c_desc(4, 1), {0, -1, 2, 1});
    break;
  case M_I2C5:
    add_tensor(p, i2c_desc(5, 0), {0, -1, 0, 1, 0});
    break;
  case M_RG: {
    const bool one = sel_op >= 0 && sel_type >= 0;
    const uint32_t many[] = {16, 208, 1024};
    const uint32_t few[]  = {16, 48, 720, 4096};
    uint32_t k = 0, pairs = 0;
    for (uint32_t op = 0; op < dr::RED_OPS; ++op) {
      for (uint32_t type = 0; type < dr::RT_TYPES; ++type) {
        if ((sel_op >= 0 && op != (uint32_t)sel_op) || (sel_type >= 0 && type != (uint32_t)sel_type))
          continue;
        if (!dr::red_legal(dr::ST_RED_GLOBAL, op, type) || !usable(op, type))
          continue;
        ++pairs;
        for (uint32_t s : one ? std::vector<uint32_t>(few, few + 4) : std::vector<uint32_t>(many, many + 3)) {
          add_bulk(p, J_BRED, s, 16 * (k % 4), 16 * ((k + 2) % 4), op, type);
          ++k;
        }
      }
    }
    if (pairs == 0)
      p.error = "no reduction pair to run (FPAMO off?)";
    break;
  }
  case M_RT: {
    struct { uint32_t op, type; } r[] = {
      {dr::RED_ADD, dr::RT_F32}, {dr::RED_MIN, dr::RT_S32}, {dr::RED_ADD, dr::RT_F16}, {dr::RED_MAX, dr::RT_BF16},
      {dr::RED_XOR, dr::RT_U64}, {dr::RED_ADD, dr::RT_U64}, {dr::RED_INC, dr::RT_U32}, {dr::RED_ADD, dr::RT_F64},
      {dr::RED_MAX, dr::RT_U32}, {dr::RED_AND, dr::RT_U32}, {dr::RED_MAX, dr::RT_S64},
      {dr::RED_ADD, dr::RT_F16}, {dr::RED_MAX, dr::RT_BF16},
    };
    for (uint32_t i = 0; i < sizeof(r) / sizeof(*r); ++i) {
      const uint32_t op = r[i].op, type = r[i].type, E = dr::type_bytes(type);
      if (!usable(op, type))
        continue;
      switch (i) {
      case 0: add_tensor(p, tile_desc(2, E, {13, 6}, {16, 4}), {8, 3}, op, type); break;
      case 1: {
        dr::Desc d = tile_desc(3, E, {20, 9, 3}, {16, 4, 2});
        d.swizzle = dr::SWZ_64B;
        add_tensor(p, d, {5, 6, 2}, op, type);
        break;
      }
      case 2: {
        dr::Desc d = tile_desc(2, E, {70, 12}, {64, 8});
        d.swizzle = dr::SWZ_128B;
        add_tensor(p, d, {10, 6}, op, type);
        break;
      }
      case 3: add_tensor(p, tile_desc(3, E, {20, 7, 3}, {16, 4, 2}), {6, 5, 2}, op, type); break;
      case 4: add_tensor(p, tile_desc(5, E, {5, 3, 3, 2, 2}, {8, 4, 4, 3, 3}), {1, 1, 1, 1, 1}, op, type); break;
      case 5: {
        dr::Desc d = tile_desc(2, E, {9, 12}, {8, 8});
        d.estride[1] = 2;
        add_tensor(p, d, {3, 1}, op, type);
        break;
      }
      case 6: add_tensor(p, tile_desc(1, E, {37}, {16}), {30}, op, type); break;
      case 7: add_tensor(p, tile_desc(2, E, {7, 5}, {8, 4}), {2, 3}, op, type); break;
      case 8: {
        dr::Desc d = i2c_desc(4, 0);
        add_tensor(p, d, {4, 3, -1, 0}, op, type);
        break;
      }
      case 9: {
        dr::Desc d = tile_desc(3, E, {12, 4, 3}, {4, 3, 2});
        d.interleave = dr::IL_16B;
        add_tensor(p, d, {9, 2, 2}, op, type);
        break;
      }
      case 10: add_tensor(p, tile_desc(4, E, {5, 4, 3, 3}, {4, 4, 2, 2}), {3, 2, 2, 2}, op, type); break;
      default:
        // An odd dim-0 start and an odd in-bounds run: half pairs at both ends of every row.
        add_tensor(p, tile_desc(2, E, {22, 9}, {16, 8}), {(int32_t)(i == 11 ? 3 : 5), 2}, op, type);
        break;
      }
    }
    break;
  }
  case M_GRP: {
    // Four groups of 1, 2, 3 and 1 stores; each wait checks the groups it completes.
    const uint32_t big = std::min<uint32_t>(4096, (p.cap - LM_DATA) / 2 / 64 * 64);
    const uint32_t ops[4] = {1, 2, 3, 1};
    uint32_t k = 0;
    for (uint32_t g = 0; g < 4; ++g) {
      for (uint32_t o = 0; o < ops[g]; ++o, ++k)
        add_bulk(p, J_BSTORE, big - 16 * (k % 4), 16 * (k % 4), 16 * ((k + 1) % 4));
      p.commit();
    }
    for (uint32_t n = 4; n-- > 0;)
      p.wait(J_WAIT, n, true);
    // N >= 8 is satisfied at once; an empty group.
    add_bulk(p, J_BSTORE, 1024, 0, 16);
    p.commit();
    p.wait(J_WAIT, 31, false);
    p.wait(J_WAITR, 8, false);
    p.commit();
    p.wait(J_WAIT, 0, true);
    // Ten groups back to back: the ninth commit finds eight pending.
    for (uint32_t g = 0; g < 10; ++g) {
      add_bulk(p, J_BSTORE, 2048, 16 * (g % 4), 16 * ((g + 3) % 4));
      p.commit();
    }
    p.wait(J_WAIT, 0, true);
    // One group of 70 operations: the 64th waits until one of them has completed.
    for (uint32_t o = 0; o < 70; ++o) {
      Job x;
      x.j.kind = J_BSTORE;
      x.j.size = 16 + 16 * (o % 4);
      x.j.src = p.lalloc(x.j.size, 16);
      x.j.gdst = p.galloc(x.j.size, 16 * (o % 4));
      Batch& b = p.cur();
      b.lmem_hi = std::max(b.lmem_hi, x.j.src + x.j.size);
      x.j.group = b.groups;
      b.jobs.push_back(x);
    }
    p.commit();
    p.wait(J_WAIT, 0, true);
    break;
  }
  case M_RDREUSE: {
    const uint32_t big = std::min<uint32_t>(8192, (p.cap - LM_DATA) / 4 / 64 * 64);
    p.lcur = LM_DATA;
    const uint32_t lo = p.lcur;
    add_bulk(p, J_BSTORE, big, 0, 16);
    dr::Desc d = tile_desc(2, 2, {70, 12}, {64, 8});
    d.swizzle = dr::SWZ_128B;
    add_tensor(p, d, {3, 2});
    if (p.llc)
      add_bulk(p, J_BRED, 1024, 16, 32, dr::RED_ADD, dr::RT_U32);
    p.commit();
    p.wait(J_WAITR, 0, false);
    Job rw;
    rw.j.kind = J_REWRITE;
    rw.j.src = lo;
    rw.j.size = round_up(p.lcur, 4) - lo;
    p.push(rw);
    p.wait(J_WAIT, 0, false);
    // The rewritten bytes reach global memory too.
    Job x;
    x.j.kind = J_BSTORE;
    x.j.size = big;
    x.j.src = lo;
    x.j.gdst = p.galloc(big, 32);
    p.push(x);
    p.commit();
    p.wait(J_WAIT, 0, false);
    break;
  }
  default:
    break;
  }
}

void plan_neg(Plan& p) {
  p.raw = true;
  switch (neg) {
  case N_NEG:
    add_tensor(p, tile_desc(2, 4, {16, 8}, {8, 4}), {-1, 0});
    break;
  case N_NOL2:
    add_bulk(p, J_BRED, 64, 0, 0, dr::RED_ADD, dr::RT_U32);
    break;
  case N_NOFP:
    add_bulk(p, J_BRED, 64, 0, 0, dr::RED_ADD, dr::RT_F32);
    break;
  case N_BADRED:
    add_bulk(p, J_BRED, 64, 0, 0, dr::RED_MIN, dr::RT_F32);
    break;
  case N_TSIZE:
    add_tensor(p, tile_desc(2, 4, {16, 8}, {8, 4}), {0, 0}, dr::RED_ADD, dr::RT_U64);
    break;
  case N_LAYOUT: {
    dr::Desc d = tile_desc(2, 4, {16, 8}, {8, 4});
    d.layout = dr::LAYOUT_KMAJOR;
    add_tensor(p, d, {0, 0});
    break;
  }
  default: {
    Job x;
    x.j.kind = J_RAW;
    x.j.n = neg == N_ENC ? 0 : 1;
    x.j.size = 64;
    x.j.src = p.lalloc(64, 16);
    x.j.gdst = p.galloc(64);
    p.push(x);
    break;
  }
  }
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

// -p: `repeats` copies of the mode's jobs, each with the same local-memory
// sources and fresh global regions, closed by commit + wait_group 0. A launch
// takes as many whole repeats as its descriptor slots hold.
void plan_perf(Plan& p) {
  const uint32_t skipped0 = skipped_fpamo;
  Plan probe = p;
  plan_mode(probe);
  const uint32_t skipped1 = skipped_fpamo;
  uint32_t descs = 0;
  for (const Batch& b : probe.batches)
    descs += (uint32_t)b.descs.size();
  const uint32_t per_launch = probe.batches.size() > 1 ? 1 : descs ? std::max(1u, kSlots / descs) : repeats;
  for (uint32_t r = 0; r < repeats && p.error.empty(); ++r) {
    if (r > 0 && r % per_launch == 0)
      p.batches.emplace_back();
    p.lcur = LM_DATA;
    plan_mode(p);
    p.commit();
    p.wait(J_WAIT, 0, false);
    ++p.cur().repeats;
  }
  skipped_fpamo = skipped1 - skipped0;
}

uint8_t canary_byte(uint64_t o) {
  return (uint8_t)(mix32(0xCA0000u, (uint32_t)(o / 4)) >> (8 * (o & 3)));
}

void put_le(uint8_t* p, uint32_t n, uint64_t v) {
  for (uint32_t i = 0; i < n; ++i)
    p[i] = (uint8_t)(v >> (8 * i));
}

// The block's local memory after the pattern and the fills.
std::vector<uint8_t> lmem_image(const Batch& b, uint32_t L) {
  std::vector<uint8_t> img(L, 0);
  for (uint32_t w = 2; w < L / 4; ++w)
    put_le(&img[4 * w], 4, lmem_word(w));
  for (const fill_t& f : b.fills) {
    const uint32_t E = type_size(f.type);
    for (uint32_t e = 0; e < f.size / E; ++e)
      put_le(&img[f.src + e * E], E, typed_val(f.type, f.seed, e));
  }
  return img;
}

// Applies a batch's jobs to the global model, as the device runs them.
void model_batch(const Plan& p, const Batch& b, uint32_t L, std::vector<uint8_t>& g, bool* bad_addr) {
  std::vector<uint8_t> lm = lmem_image(b, L);
  auto gptr = [&](uint64_t a) -> uint8_t* {
    static uint8_t sink;
    if (a < p.gbase || a >= p.gbase + kGBytes) {
      *bad_addr = true;
      return &sink;
    }
    return &g[a - p.gbase];
  };
  {
    for (const Job& x : b.jobs) {
      const job_t& j = x.j;
      switch (j.kind) {
      case J_BSTORE:
        for (uint32_t i = 0; i < j.size; ++i)
          *gptr(j.gdst + i) = lm[j.src + i];
        break;
      case J_TSTORE:
        dr::store_box(x.d, x.q, j.src, lm.data(), lm.size(), gptr);
        break;
      case J_BRED:
        if (j.gdst < p.gbase || j.gdst + j.size > p.gbase + kGBytes)
          *bad_addr = true;
        else
          dr::reduce_bytes(j.op, j.type, &g[j.gdst - p.gbase], &lm[j.src], j.size);
        break;
      case J_TRED:
        dr::reduce_box(x.d, x.q, j.src, j.op, j.type, lm.data(), lm.size(), gptr);
        break;
      case J_REWRITE:
        for (uint32_t w = j.src / 4; w < (j.src + j.size) / 4; ++w)
          put_le(&lm[4 * w], 4, lmem_word2(w));
        break;
      default:
        break;
      }
    }
  }
}

const char* job_name(uint32_t k) {
  static const char* const n[] = {"bulk store", "tensor store", "bulk reduce", "tensor reduce", "commit", "wait",
                                  "wait.read", "rewrite", "raw"};
  return k < 9 ? n[k] : "?";
}

void print_job(size_t bi, const Job& x) {
  const job_t& j = x.j;
  std::printf("  launch %zu: %s", bi, job_name(j.kind));
  if (j.kind == J_BRED || j.kind == J_TRED)
    std::printf(" %s", pair_name(j.op, j.type).c_str());
  if (j.kind == J_TSTORE || j.kind == J_TRED)
    std::printf(" %uD%s slot %u src %u coords (%d, %d, %d, %d, %d) elem %u swizzle %u interleave %u", x.d.rank,
                x.d.im2col ? " im2col" : "", j.slot, j.src, j.coord[0], j.coord[1], j.coord[2], j.coord[3],
                j.coord[4], x.d.elem, x.d.swizzle, x.d.interleave);
  if (j.kind == J_BSTORE || j.kind == J_BRED || j.kind == J_RAW)
    std::printf(" src %u gdst +%#lx size %u", j.src, (unsigned long)(j.gdst & 0xFFFFFu), j.size);
  if (j.kind == J_WAIT || j.kind == J_WAITR)
    std::printf(" %u%s", j.n, j.check ? " (checked)" : "");
  if (j.kind <= J_TRED)
    std::printf(" group %u", j.group);
  std::printf("\n");
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

  // Every host buffer stays alive until the queue has finished.
  std::vector<uint8_t> gmem(kGBytes);
  for (uint32_t i = 0; i < kGBytes; ++i)
    gmem[i] = canary_byte(i);
  vx_buffer_h gbuf = nullptr;
  Plan plan;
  CHECK(vx_buffer_create(dev, kGBytes, VX_MEM_READ_WRITE, &gbuf));
  CHECK(vx_buffer_address(gbuf, &plan.gbase));
  plan.cap = (uint32_t)lmem_cap;
  plan.llc = (isa & (VX_ISA_EXT_L2CACHE | VX_ISA_EXT_L3CACHE)) != 0;

  const bool special = special_mode(mode);
  const uint32_t region_bytes = mode == M_EXIT ? kExitBytes : kI7Bytes;
  const uint32_t region_lmem = LM_DATA + (mode == M_I7A ? 2 : 1) * region_bytes;
  if (neg)
    plan_neg(plan);
  else if (perf)
    plan_perf(plan);
  else if (!special)
    plan_mode(plan);
  for (const Typed& t : plan.typed) {
    const uint32_t E = type_size(t.type);
    for (uint32_t e = 0; e < t.size / E; ++e)
      put_le(&gmem[t.off + e * E], E, typed_val(t.type, t.seed, e));
  }
  uint64_t region = 0;
  if (special)
    region = plan.galloc(region_bytes, 16);
  CHECK(vx_enqueue_write(q, gbuf, 0, gmem.data(), kGBytes, 0, nullptr, nullptr));

  const uint32_t threads = (uint32_t)nt * 2;
  if (neg)
    std::printf("dxa_store: negative run '%s', %u warps x %u threads\n", kNegNames[neg], 2, (uint32_t)nt);
  else
    std::printf("dxa_store: %s%s%s, %zu launches, %u warps x %u threads, %u bytes of local memory, %s%s\n",
                kModeNames[mode], mode == M_SWZ ? (" " + std::to_string(swz_span) + "B").c_str() : "",
                perf ? " (perf)" : "", special ? (size_t)1 : plan.batches.size(), 2, (uint32_t)nt,
                (uint32_t)lmem_cap, plan.llc ? "L2/L3" : "no L2/L3", kFpamo ? ", FPAMO" : "");
  if (skipped_fpamo)
    std::printf("dxa_store: %u reduction pairs need FPAMO and were left out\n", skipped_fpamo);

  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA;
  const char* skip = nullptr;
  if ((isa & need) != need)
    skip = "the device needs DLMEM, the A extension and the DXA";
  else if (nw < 2 || nt < 4)
    skip = "the device has too few warps or threads";
  else if (!plan.error.empty())
    skip = plan.error.c_str();
  else if (!neg && !plan.llc && (mode == M_RG || mode == M_RT))
    skip = "global reductions need an L2 or L3";
  else if (special && region_lmem > lmem_cap)
    skip = "the local memory is too small";
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
  CHECK(vx_module_get_kernel(mod, "store_k", &kernel));

  if (special) {
    // One launch, no jobs: the kernel's own sequence over `region`.
    plan.batches.assign(1, Batch());
    plan.batches[0].lmem_hi = region_lmem;
  }

  int errors = 0;
  std::vector<uint8_t> model = gmem;
  bool bad_addr = false;
  uint32_t total_ops = 0, total_cycles = 0;
  for (size_t bi = 0; bi < plan.batches.size(); ++bi) {
    Batch& bt = plan.batches[bi];
    const uint32_t L = std::min<uint32_t>(round_up(bt.lmem_hi + kGap, 256), (uint32_t)lmem_cap);
    for (uint32_t s = 0; s < bt.descs.size(); ++s) {
      if (program(dev, s, bt.descs[s]) != 0) {
        std::printf("Error: cannot program DXA descriptor %u\nFAILED!\n", s);
        return 1;
      }
    }
    store_arg_t arg = {};
    std::vector<job_t> jobs;
    for (const Job& x : bt.jobs) {
      jobs.push_back(x.j);
      total_ops += (x.j.kind <= J_TRED);
      if (verbose || neg)
        print_job(bi, x);
    }
    if (jobs.empty())
      jobs.push_back(job_t{});
    arg.num_jobs   = special ? 0 : (uint32_t)jobs.size();
    arg.num_fills  = (uint32_t)bt.fills.size();
    arg.lmem_bytes = L;
    arg.kernel     = mode == M_I7A ? K_I7A : mode == M_I7B ? K_I7B : mode == M_I7C ? K_I7C
                   : mode == M_EXIT ? K_EXIT : K_JOBS;
    arg.size       = region_bytes;
    arg.gaddr      = region;
    arg.neg        = neg;
    arg.linger     = kLinger;
    arg.repeats    = perf ? bt.repeats : 0;
    std::vector<fill_t> fills = bt.fills;
    if (fills.empty())
      fills.push_back(fill_t{});
    std::vector<uint32_t> errs((size_t)threads * ERR_WORDS, 0);
    vx_buffer_h jobs_buf = nullptr, fill_buf = nullptr, err_buf = nullptr;
    CHECK(vx_buffer_create(dev, jobs.size() * sizeof(job_t), VX_MEM_READ, &jobs_buf));
    CHECK(vx_buffer_address(jobs_buf, &arg.jobs_addr));
    CHECK(vx_enqueue_write(q, jobs_buf, 0, jobs.data(), jobs.size() * sizeof(job_t), 0, nullptr, nullptr));
    CHECK(vx_buffer_create(dev, fills.size() * sizeof(fill_t), VX_MEM_READ, &fill_buf));
    CHECK(vx_buffer_address(fill_buf, &arg.fills_addr));
    CHECK(vx_enqueue_write(q, fill_buf, 0, fills.data(), fills.size() * sizeof(fill_t), 0, nullptr, nullptr));
    CHECK(vx_buffer_create(dev, errs.size() * 4, VX_MEM_READ_WRITE, &err_buf));
    CHECK(vx_buffer_address(err_buf, &arg.err_addr));
    CHECK(vx_enqueue_write(q, err_buf, 0, errs.data(), errs.size() * 4, 0, nullptr, nullptr));

    vx_launch_info_t li = {};
    li.struct_size    = sizeof(li);
    li.kernel         = kernel;
    li.args_host      = &arg;
    li.args_size      = sizeof(arg);
    li.ndim           = 1;
    li.grid_dim[0]    = 1;
    li.grid_dim[1]    = 1;
    li.grid_dim[2]    = 1;
    li.block_dim[0]   = threads;
    li.block_dim[1]   = 1;
    li.block_dim[2]   = 1;
    li.lmem_size      = L;
    li.cluster_dim[0] = 1;
    li.cluster_dim[1] = 1;
    li.cluster_dim[2] = 1;
    li.cluster_policy = VX_CLUSTER_POLICY_DEFAULT;
    li.cluster_flags  = 0;

    vx_event_h launch_ev = nullptr, err_ev = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, errs.data(), err_buf, 0, errs.size() * 4, 1, &launch_ev, &err_ev));
    CHECK(vx_event_wait_value(err_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(err_ev);
    vx_event_release(launch_ev);

    if (neg) {
      std::printf("*** the negative run '%s' completed: the simulator must stop it\n", kNegNames[neg]);
      ++errors;
    }
    uint32_t unfinished = 0, bad_words = 0, gave_up = 0;
    for (uint32_t t = 0; t < threads; ++t) {
      unfinished += (errs[t * ERR_WORDS + ERR_DONE] != 1);
      bad_words += errs[t * ERR_WORDS + ERR_BAD];
      gave_up += errs[t * ERR_WORDS + ERR_WAIT];
    }
    if (unfinished || gave_up) {
      std::printf("*** launch %zu: %u threads unfinished, %u gave up waiting for an mbarrier phase\n", bi,
                  unfinished, gave_up);
      ++errors;
    }
    if (bad_words) {
      std::printf("*** launch %zu: generic loads found %u wrong words%s\n", bi, bad_words,
                  mode == M_GRP ? " after a wait_group" : "");
      ++errors;
    }
    if (perf) {
      const uint32_t cycles = errs[ERR_CYCLES];
      total_cycles += cycles;
      std::printf("dxa_store: launch %zu: %zu jobs in %u repeats, %u cycles (%u a repeat)\n", bi, jobs.size(),
                  bt.repeats, cycles, cycles / std::max(1u, bt.repeats));
    } else if (mode == M_I7A) {
      for (uint32_t w = 0; w < kI7Bytes / 4; ++w)
        put_le(&model[region - plan.gbase + 4 * w], 4, new_word(w));
    } else if (mode == M_I7B || mode == M_I7C || mode == M_EXIT) {
      for (uint32_t w = 0; w < region_bytes / 4; ++w)
        put_le(&model[region - plan.gbase + 4 * w], 4, lmem_word(LM_DATA / 4 + w));
    } else {
      model_batch(plan, bt, L, model, &bad_addr);
    }
    vx_buffer_release(err_buf);
    vx_buffer_release(fill_buf);
    vx_buffer_release(jobs_buf);
  }

  std::vector<uint8_t> got(kGBytes, 0);
  if (!perf) {
    vx_event_h rd_ev = nullptr;
    CHECK(vx_enqueue_read(q, got.data(), gbuf, 0, kGBytes, 0, nullptr, &rd_ev));
    CHECK(vx_event_wait_value(rd_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(rd_ev);
  }
  if (bad_addr) {
    std::printf("*** the model wrote outside the global buffer\n");
    ++errors;
  }
  uint32_t bad = 0;
  for (uint32_t o = 0; o < kGBytes && !perf; ++o) {
    if (got[o] == model[o])
      continue;
    if (bad++ < 12) {
      const char* what = model[o] == gmem[o] ? "must be untouched" : "written";
      std::printf("*** global byte +%#x (%s): %#04x, expected %#04x (initially %#04x)\n", o, what, got[o],
                  model[o], gmem[o]);
    }
  }
  if (bad) {
    std::printf("*** %u wrong bytes in the global buffer\n", bad);
    ++errors;
  }
  if (perf) {
    std::printf("dxa_store: %u operations in %u repeats, %u timed cycles (global buffer not checked)\n",
                total_ops, repeats, total_cycles);
  } else if (!neg) {
    uint32_t written = 0;
    for (uint32_t o = 0; o < kGBytes; ++o)
      written += (model[o] != gmem[o]);
    std::printf("dxa_store: %u operations, %u global bytes changed, checked byte for byte\n", total_ops,
                written);
  }

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
