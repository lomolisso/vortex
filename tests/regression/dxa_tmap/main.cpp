// dxa_tmap: PTX tensor maps in memory (the CUtensorMap model) on the DXA, on
// a DLMEM device with the A extension, mbarriers and the DXA (RV64). The host
// encodes maps with vortex::dxa::encode_tiled / encode_im2col (dxa.h) into the
// kernel argument (the __grid_constant__ form) and a global maps buffer; the
// block's warps run a job list of map-form tensor loads, stores and
// reductions, tensormap prefetches, acquires and releases, and the
// tensormap.replace / cp_fenceproxy helpers of vx_dxa.h. The host replays
// the jobs on its own images through sw/common/dxa_ref.h and checks every
// byte of every block's local memory, of the global buffer and of the maps
// buffer.
//
//   -m param          a map in the argument: a 2D tile load (with fill) and a
//                     tensor store; the same box through a descriptor slot
//                     lands the same bytes
//   -m global         maps in a global buffer: a 3D swizzled load with element
//                     strides and fill, a 2D store
//   -m im2col         a 4D im2col load through a map
//   -m mcast          a multicast load through the argument's map (cluster of
//                     2 on 2 cores)
//   -m red            map-typed tensor reductions: f32 add, u32 max (an L2,
//                     and FPAMO for the f32 add)
//   -m relaunch       two launches with the same argument address and
//                     different maps: the second uses its own
//   -m evict          12 maps used in rotation, each result checked
//   -m prefetch       8 loads through maps not yet cached, then
//                     prefetch.tensormap of 8 others and the loads through
//                     them, then the same loads again: the prefetched loads
//                     are faster than the cold ones, as fast as cached ones
//   -m replace-smem   a copy in shared memory, replace global_address,
//                     cp_fenceproxy, acquire, __syncthreads, then another warp's
//                     load reads the new tensor
//   -m replace-global PTX §9.7.13.4: replace on the global map, release,
//                     acquire, and the load reads the new tensor
//   -m replace-zero   every field from a zero buffer in shared memory, then
//                     cp_fenceproxy, acquire and a load; the global map equals
//                     the host encoder's word for word
//   -m replace-partial  replace-zero with cp_fenceproxy from a partial warp
//                     (lanes 0 and 2 of a diverged branch)
//   -x unaligned|smem|zero|wide|redtype   a run the simulator must stop
//   -p miss|hit       timing (the parity cells), no canaries and no dump: a
//                     first batch caches 8 maps; the timed batch issues 48
//                     prefetch.tensormap back to back, through 48 maps not yet
//                     cached or through the 8 cached ones, then one load of a
//                     16-byte box behind them, and waits once
//   -v                print every job
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
namespace dt = vortex::dxa_tmap;
namespace dh = vortex::dxa;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

#ifdef VX_CFG_EXT_FPAMO_ENABLE
constexpr bool kFpamo = true;
#else
constexpr bool kFpamo = false;
#endif

enum Mode { M_PARAM, M_GLOBAL, M_IM2COL, M_MCAST, M_RED, M_RELAUNCH, M_EVICT, M_PREFETCH, M_RSMEM, M_RGLOBAL,
            M_RZERO, M_RPARTIAL, NUM_MODES };
const char* const kModeNames[] = {"param", "global", "im2col", "mcast", "red", "relaunch", "evict", "prefetch",
                                  "replace-smem", "replace-global", "replace-zero", "replace-partial"};
enum Neg { N_NONE, N_UNALIGNED, N_SMEM, N_ZERO, N_WIDE, N_REDTYPE, NUM_NEGS };
const char* const kNegNames[] = {"", "unaligned", "smem", "zero", "wide", "redtype"};
const char* const kNegExpect[] = {"", "DXA tensor map is not 64-byte aligned",
                                  "DXA tensor map must be in global memory", "DXA tensor map rank must be 1-5",
                                  "DXA tensor stride needs 32 bits",
                                  "DXA tensor reduction type not supported by the map"};

constexpr uint32_t kGBytes   = 256 * 1024;
constexpr uint32_t kMaps     = 64;           // maps buffer entries
constexpr uint32_t kLmem     = 4096;
constexpr uint32_t kLinger   = 20000;
constexpr uint32_t kRepeats  = PERF_PREFETCHES;
constexpr uint32_t kPfIdle   = 4000;
constexpr uint32_t kPfMinSave = 20;

const char* kernel_file = "kernel.vxbin";
uint32_t mode = M_PARAM;
uint32_t neg = N_NONE;
int perf = -1;            // -p: 1 miss, 0 hit
bool verbose = false;
bool requested = false;

void usage() {
  std::printf("Usage: [-m param|global|im2col|mcast|red|relaunch|evict|prefetch|replace-smem|replace-global|"
              "replace-zero|replace-partial] [-x unaligned|smem|zero|wide|redtype] [-p miss|hit] [-v] [-h]\n");
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
  while ((c = getopt(argc, argv, "m:x:p:vh")) != -1) {
    switch (c) {
    case 'm': mode = parse_name(optarg, kModeNames, 0, NUM_MODES); break;
    case 'x': neg = parse_name(optarg, kNegNames, 1, NUM_NEGS); break;
    case 'p':
      if (!std::strcmp(optarg, "miss")) perf = 1;
      else if (!std::strcmp(optarg, "hit")) perf = 0;
      else { usage(); std::exit(-1); }
      break;
    case 'v': verbose = true; break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || (perf >= 0 && neg)) {
    usage();
    std::exit(-1);
  }
}

uint32_t round_up(uint32_t x, uint32_t a) {
  return (x + a - 1) / a * a;
}

uint8_t gbyte(uint32_t o) {
  return (uint8_t)(mix32(0x3C0000u, o / 4) >> (8 * (o & 3)));
}

// One launch: its argument map, jobs, spec, and the slots it programs.
struct Launch {
  Launch() { std::memset(&pmap, 0, sizeof(pmap)); }
  vx_tensor_map_t pmap;
  rspec_t spec = {};
  std::vector<job_t> jobs;
  std::vector<std::pair<uint32_t, vx_tensor_map_t>> slots;  // slot, the descriptor's words
  uint32_t blocks = 1;
  uint32_t warps = 2;
};

struct Plan {
  uint64_t gbase = 0, mbase = 0;
  uint32_t gnext = 0;
  std::vector<vx_tensor_map_t> maps = std::vector<vx_tensor_map_t>(kMaps);
  std::vector<Launch> launches;
  std::string error;
  bool mcast = false;

  Plan() {
    for (auto& m : maps)
      std::memset(&m, 0, sizeof(m));
  }

  uint64_t galloc(uint32_t bytes) {
    const uint64_t a = gbase + gnext;
    gnext = round_up(gnext + bytes, 256);
    if (gnext > kGBytes)
      error = "the global buffer is too small";
    return a;
  }

  uint32_t map_off(uint32_t idx) const {
    return idx * 128u;
  }

  uint64_t map_addr(uint32_t idx) const {
    return mbase + idx * 128u;
  }
};

// A dense row-major tensor (rows padded to 16 bytes) and its tile map, into
// *out. The map is encoded in a local and copied: GCC 11 can hand a by-value
// return slot of a 64-byte aligned type to the callee misaligned, which the
// encoder rejects.
void tile_map(Plan& p, vx_tensor_map_t* out, dh::TensorMapDataType t, uint32_t rank, const uint32_t* dims,
                         const uint32_t* box, const uint32_t* es = nullptr,
                         dh::TensorMapSwizzle swz = dh::TensorMapSwizzle::None, bool nan = false,
                         uint64_t* base_out = nullptr) {
  const uint32_t E = dt::elem_bytes(t == dh::TensorMapDataType::Float32   ? TMAP_ET_F32
                                    : t == dh::TensorMapDataType::BFloat16 ? TMAP_ET_BF16
                                    : t == dh::TensorMapDataType::Float16  ? TMAP_ET_F16
                                    : t == dh::TensorMapDataType::UInt8    ? TMAP_ET_U8
                                                                           : TMAP_ET_U32);
  uint64_t gdim[5] = {1, 1, 1, 1, 1}, gstr[4] = {0, 0, 0, 0};
  uint32_t ones[5] = {1, 1, 1, 1, 1};
  for (uint32_t i = 0; i < rank; ++i)
    gdim[i] = dims[i];
  uint64_t bytes = round_up(dims[0] * E, 16);
  for (uint32_t i = 1; i < rank; ++i) {
    gstr[i - 1] = bytes;
    bytes *= dims[i];
  }
  const uint64_t base = p.galloc((uint32_t)bytes);
  if (base_out)
    *base_out = base;
  vx_tensor_map_t m;
  const int r = dh::encode_tiled(&m, t, rank, base, gdim, gstr, box, es ? es : ones,
                                 dh::TensorMapInterleave::None, swz, dh::TensorMapL2Promotion::L2_128B,
                                 nan ? dh::TensorMapFloatOOBfill::NanRequestZeroFma
                                     : dh::TensorMapFloatOOBfill::None);
  if (r != VX_SUCCESS)
    p.error = "encode_tiled failed: " + std::to_string(r);
  if (out)
    *out = m;
}

job_t mk(uint32_t kind, uint32_t warp = 0, uint32_t sel = SEL_GLOBAL, uint32_t map_off = 0) {
  job_t j = {};
  j.kind = kind;
  j.warp = warp;
  j.sel = sel;
  j.map_off = map_off;
  return j;
}

job_t load(uint32_t sel, uint32_t map_off, uint32_t rank, uint32_t dst, std::initializer_list<int32_t> c,
           uint32_t warp = 0) {
  job_t j = mk(J_LOAD, warp, sel, map_off);
  j.rank = rank;
  j.dst = dst;
  uint32_t i = 0;
  for (int32_t v : c)
    j.coord[i++] = v;
  return j;
}

job_t store(uint32_t kind, uint32_t sel, uint32_t map_off, uint32_t rank, uint32_t src,
            std::initializer_list<int32_t> c) {
  job_t j = load(sel, map_off, rank, src, c);
  j.kind = kind;
  return j;
}

void plan_mode(Plan& p) {
  using T = dh::TensorMapDataType;
  p.launches.emplace_back();
  Launch& L = p.launches.back();
  auto& J = L.jobs;
  switch (mode) {
  case M_PARAM: {
    const uint32_t dims[2] = {40, 24}, box[2] = {16, 8};
    uint64_t base = 0;
    tile_map(p, &L.pmap, T::Float32, 2, dims, box, nullptr, dh::TensorMapSwizzle::None, true, &base);
    L.slots.push_back({0, L.pmap});
    J.push_back(load(SEL_PARAM, 0, 2, LM_DATA, {32, 20}));
    job_t s = load(SEL_SLOT, 0, 2, LM_DATA + 512, {32, 20});
    J.push_back(s);
    J.push_back(store(J_STORE, SEL_PARAM, 0, 2, LM_DATA + 1024, {0, 0}));
    J.push_back(mk(J_BWAIT));
    J.push_back(store(J_STORE, SEL_SLOT, 0, 2, LM_DATA + 1024, {32, 16}));
    J.push_back(mk(J_BWAIT));
    break;
  }
  case M_GLOBAL: {
    const uint32_t d3[3] = {24, 10, 6}, b3[3] = {16, 4, 3}, e3[3] = {1, 2, 1};
    tile_map(p, &p.maps[0], T::BFloat16, 3, d3, b3, e3, dh::TensorMapSwizzle::B32, true);
    const uint32_t d2[2] = {24, 12}, b2[2] = {8, 4};
    tile_map(p, &p.maps[1], T::Float32, 2, d2, b2);
    J.push_back(load(SEL_GLOBAL, p.map_off(0), 3, LM_DATA, {8, -1, 4}));
    J.push_back(store(J_STORE, SEL_GLOBAL, p.map_off(1), 2, LM_DATA + 1024, {8, 2}));
    J.push_back(mk(J_BWAIT));
    J.push_back(store(J_STORE, SEL_GLOBAL, p.map_off(1), 2, LM_DATA + 1280, {20, 10}));
    J.push_back(mk(J_BWAIT));
    break;
  }
  case M_IM2COL: {
    const uint64_t gdim[4] = {16, 10, 8, 2};
    const uint64_t row = 16 * 2;
    const uint64_t gstr[3] = {row, row * 10, row * 80};
    const int32_t lower[2] = {-1, -1}, upper[2] = {-1, -1};
    const uint32_t es[4] = {1, 2, 1, 1};
    const uint64_t base = p.galloc(16 * 10 * 8 * 2 * 2);
    if (dh::encode_im2col(&p.maps[2], T::Float16, 4, base, gdim, gstr, lower, upper, 16, 24, es,
                          dh::TensorMapInterleave::None, dh::TensorMapSwizzle::B32,
                          dh::TensorMapL2Promotion::None, dh::TensorMapFloatOOBfill::None))
      p.error = "encode_im2col failed";
    job_t j = load(SEL_GLOBAL, p.map_off(2), 4, LM_DATA, {0, -1, -1, 0});
    j.im2col = 1;
    j.offs[0] = 1;
    j.offs[1] = 1;
    J.push_back(j);
    j = load(SEL_GLOBAL, p.map_off(2), 4, LM_DATA + 1024, {0, 3, 2, 1});
    j.im2col = 1;
    J.push_back(j);
    break;
  }
  case M_MCAST: {
    const uint32_t dims[2] = {32, 16}, box[2] = {16, 4};
    tile_map(p, &L.pmap, T::Float32, 2, dims, box);
    L.blocks = 2;
    p.mcast = true;
    break;
  }
  case M_RED: {
    const uint32_t d[2] = {20, 6}, b[2] = {8, 4};
    tile_map(p, &p.maps[0], T::Float32, 2, d, b);
    tile_map(p, &p.maps[1], T::UInt32, 2, d, b);
    if (kFpamo) {
      job_t j = store(J_RED, SEL_GLOBAL, p.map_off(0), 2, LM_DATA, {16, 4});
      j.op = DXA_RED_ADD;
      J.push_back(j);
      J.push_back(mk(J_BWAIT));
    }
    job_t j = store(J_RED, SEL_GLOBAL, p.map_off(1), 2, LM_DATA + 512, {4, 1});
    j.op = DXA_RED_MAX;
    J.push_back(j);
    J.push_back(mk(J_BWAIT));
    j.coord[0] = 0;
    j.coord[1] = 0;
    J.push_back(j);
    J.push_back(mk(J_BWAIT));
    break;
  }
  case M_RELAUNCH: {
    const uint32_t d0[2] = {32, 8}, b0[2] = {16, 4};
    tile_map(p, &L.pmap, T::Float32, 2, d0, b0);
    J.push_back(load(SEL_PARAM, 0, 2, LM_DATA, {4, 2}));
    p.launches.emplace_back();
    Launch& L2 = p.launches.back();
    const uint32_t d1[2] = {48, 8}, b1[2] = {32, 2};
    tile_map(p, &L2.pmap, T::Float16, 2, d1, b1, nullptr, dh::TensorMapSwizzle::None, true);
    L2.jobs.push_back(load(SEL_PARAM, 0, 2, LM_DATA, {40, 1}));
    break;
  }
  case M_EVICT: {
    const uint32_t d[2] = {8, 4}, b[2] = {8, 2};
    for (uint32_t i = 0; i < 12; ++i)
      tile_map(p, &p.maps[i], T::UInt32, 2, d, b);
    for (uint32_t k = 0; k < 24; ++k)
      J.push_back(load(SEL_GLOBAL, p.map_off(k % 12), 2, LM_DATA + 64 * k, {0, (int32_t)(k / 12) * 2}));
    break;
  }
  case M_PREFETCH: {
    const uint32_t d[2] = {32, 8}, b[2] = {16, 4};
    tile_map(p, &p.maps[0], T::Float32, 2, d, b);
    for (uint32_t r = 1; r < 2 * DXA_TMAP_ENTRIES; ++r)
      p.maps[r] = p.maps[0];
    J.push_back(load(SEL_GLOBAL, 0, 2, LM_DATA, {4, 2}));
    L.warps = 1;
    break;
  }
  case M_RSMEM:
  case M_RGLOBAL: {
    const uint32_t d[2] = {32, 8}, b[2] = {16, 4};
    uint64_t tb = 0;
    tile_map(p, &p.maps[0], T::Float32, 2, d, b);
    tile_map(p, nullptr, T::Float32, 2, d, b, nullptr, dh::TensorMapSwizzle::None, false, &tb);
    J.push_back(load(SEL_GLOBAL, p.map_off(0), 2, LM_DATA, {4, 2}));
    if (mode == M_RSMEM) {
      job_t c = mk(J_COPY, 0, SEL_GLOBAL, p.map_off(0));
      c.dst = LM_SMAP;
      J.push_back(c);
      job_t r = mk(J_RADDR, 0, SEL_LMEM, LM_SMAP);
      r.val = tb;
      J.push_back(r);
      c = mk(J_CP, 0, SEL_GLOBAL, p.map_off(0));
      c.dst = LM_SMAP;
      J.push_back(c);
      J.push_back(mk(J_ACQ, 0, SEL_GLOBAL, p.map_off(0)));
      J.push_back(mk(J_SYNC));
      J.push_back(load(SEL_GLOBAL, p.map_off(0), 2, LM_DATA + 512, {4, 2}, 1));
    } else {
      job_t r = mk(J_RADDR, 0, SEL_GLOBAL, p.map_off(0));
      r.val = tb;
      J.push_back(r);
      J.push_back(mk(J_REL));
      J.push_back(mk(J_ACQ, 0, SEL_GLOBAL, p.map_off(0)));
      J.push_back(load(SEL_GLOBAL, p.map_off(0), 2, LM_DATA + 512, {4, 2}));
    }
    break;
  }
  case M_RZERO:
  case M_RPARTIAL: {
    const uint32_t d0[2] = {32, 8}, b0[2] = {16, 4};
    tile_map(p, &p.maps[2], T::Float32, 2, d0, b0);  // the stale content, cached first
    rspec_t& s = L.spec;
    s.rank = 3;
    const uint32_t dims[3] = {20, 6, 4}, box[3] = {16, 4, 2};
    uint64_t tc = 0;
    tile_map(p, nullptr, T::BFloat16, 3, dims, box, nullptr, dh::TensorMapSwizzle::None, false, &tc);
    s.gaddr = tc;
    for (uint32_t i = 0; i < 3; ++i) {
      s.box[i] = box[i];
      s.gdim[i] = dims[i];
      s.es[i] = i == 2 ? 2 : 1;
    }
    s.gstride[0] = 48;
    s.gstride[1] = 48 * 6;
    s.elemtype = TMAP_ET_BF16;
    s.swizzle = 1;
    s.fill = TMAP_FILL_NAN;
    J.push_back(load(SEL_GLOBAL, p.map_off(2), 2, LM_DATA, {0, 0}));
    job_t z = mk(J_ZERO);
    z.dst = LM_SMAP;
    J.push_back(z);
    J.push_back(mk(J_RALL, 0, SEL_LMEM, LM_SMAP));
    job_t c = mk(J_CP, 0, SEL_GLOBAL, p.map_off(2));
    c.dst = LM_SMAP;
    c.mask = mode == M_RPARTIAL ? 0x5 : 0;
    J.push_back(c);
    J.push_back(mk(J_ACQ, 0, SEL_GLOBAL, p.map_off(2)));
    J.push_back(load(SEL_GLOBAL, p.map_off(2), 3, LM_DATA + 512, {8, 4, 1}));
    break;
  }
  default:
    break;
  }
}

void plan_neg(Plan& p) {
  using T = dh::TensorMapDataType;
  p.launches.emplace_back();
  auto& J = p.launches.back().jobs;
  const uint32_t d[2] = {32, 8}, b[2] = {16, 4};
  tile_map(p, &p.maps[0], T::Float32, 2, d, b);
  switch (neg) {
  case N_UNALIGNED:
    J.push_back(load(SEL_GLOBAL, p.map_off(0) + 32, 2, LM_DATA, {0, 0}));
    break;
  case N_SMEM: {
    job_t c = mk(J_COPY, 0, SEL_GLOBAL, p.map_off(0));
    c.dst = LM_SMAP;
    J.push_back(c);
    J.push_back(load(SEL_LMEM, LM_SMAP, 2, LM_DATA, {0, 0}));
    break;
  }
  case N_ZERO:
    J.push_back(load(SEL_GLOBAL, p.map_off(5), 2, LM_DATA, {0, 0}));
    break;
  case N_WIDE: {
    job_t r = mk(J_RSTRIDE, 0, SEL_GLOBAL, p.map_off(0));
    r.arg = 0;
    r.val = (1ull << 32) | 128;
    J.push_back(r);
    J.push_back(mk(J_REL));
    J.push_back(mk(J_ACQ, 0, SEL_GLOBAL, p.map_off(0)));
    J.push_back(load(SEL_GLOBAL, p.map_off(0), 2, LM_DATA, {0, 0}));
    break;
  }
  default: {
    tile_map(p, &p.maps[1], T::UInt8, 2, d, b);
    job_t j = store(J_RED, SEL_GLOBAL, p.map_off(1), 2, LM_DATA, {0, 0});
    j.op = DXA_RED_ADD;
    J.push_back(j);
    J.push_back(mk(J_BWAIT));
    break;
  }
  }
}

// -p: a 1D box of 16 bytes through copies of one map.
void plan_perf(Plan& p) {
  using T = dh::TensorMapDataType;
  p.launches.emplace_back();
  const uint32_t d[1] = {64}, b[1] = {4};
  tile_map(p, &p.maps[0], T::Float32, 1, d, b);
  for (uint32_t r = 1; r < DXA_TMAP_ENTRIES + kRepeats; ++r)
    p.maps[r] = p.maps[0];
  p.launches.back().jobs.push_back(load(SEL_GLOBAL, 0, 1, LM_DATA, {4}));
  p.launches.back().warps = 1;
}

// The host's replay: one block's local memory, the global buffer and the maps.
struct Model {
  std::vector<uint8_t> g;
  std::vector<vx_tensor_map_t> maps;
  uint64_t gbase = 0, mbase = 0;
  bool bad = false;

  uint8_t* gptr(uint64_t a) {
    static uint8_t sink;
    if (a < gbase || a >= gbase + g.size()) {
      bad = true;
      return &sink;
    }
    return &g[a - gbase];
  }
};

uint32_t* map_words(Model& M, Launch& L, std::vector<uint8_t>& lm, const job_t& j) {
  switch (j.sel) {
  case SEL_PARAM:  return L.pmap.w;
  case SEL_GLOBAL: return M.maps[j.map_off / 128].w;
  default:         return reinterpret_cast<uint32_t*>(&lm[j.map_off]);
  }
}

dr::Req req_of(const job_t& j) {
  dr::Req q;
  for (uint32_t i = 0; i < 5; ++i)
    q.coord[i] = j.coord[i];
  for (uint32_t i = 0; i < 3; ++i)
    q.offs[i] = j.offs[i];
  return q;
}

void print_job(uint32_t li, const job_t& j) {
  static const char* const names[] = {"load", "store", "red", "bwait", "prefetch.tensormap", "acquire", "release",
                                      "copy", "zero", "replace.global_address", "replace.global_stride",
                                      "replace (all fields)", "cp_fenceproxy", "sync"};
  static const char* const sels[] = {"param", "global", "lmem", "slot"};
  std::printf("  launch %u warp %u: %s %s +%#x rank %u dst %#x coords (%d,%d,%d,%d,%d) tx %u\n", li, j.warp,
              names[j.kind], sels[j.sel], j.map_off, j.rank, j.dst, j.coord[0], j.coord[1], j.coord[2],
              j.coord[3], j.coord[4], j.tx);
}

// Replays the launch's jobs in order; fills each load's complete_tx bytes.
// Returns the first plan error (an assert the model trips), or empty.
std::string model_launch(Model& M, Launch& L, std::vector<std::vector<uint8_t>>& lms) {
  auto read = [&](uint64_t a) { return *M.gptr(a); };
  if (L.blocks == 2) {
    const dr::Desc d = dr::decode(L.pmap.w);
    dr::Req q;
    q.coord[0] = 4;
    q.coord[1] = 2;
    for (auto& lm : lms)
      dr::render(d, q, LM_DATA, read, lm.data(), lm.size());
  }
  std::vector<uint8_t>& lm = lms[0];
  for (job_t& j : L.jobs) {
    const bool uses_map = (j.kind == J_LOAD || j.kind == J_STORE || j.kind == J_RED) && j.sel != SEL_SLOT;
    if (uses_map && j.sel == SEL_GLOBAL && (j.map_off & 63))
      return dr::MSG_TMAP_ALIGN;
    if (uses_map && j.sel == SEL_LMEM)
      return dr::MSG_TMAP_GLOBAL;
    switch (j.kind) {
    case J_LOAD: {
      const uint32_t* w = j.sel == SEL_SLOT ? L.slots[0].second.w : map_words(M, L, lm, j);
      if (const char* m = dr::check_map_words(w))
        return m;
      const dr::Desc d = dr::decode(w);
      if (const char* m = dr::check_tensor(d, j.dst))
        return m;
      j.im2col = d.im2col;
      j.tx = dr::render(d, req_of(j), j.dst, read, lm.data(), lm.size());
      break;
    }
    case J_STORE: {
      const uint32_t* w = j.sel == SEL_SLOT ? L.slots[0].second.w : map_words(M, L, lm, j);
      const dr::Desc d = dr::decode(w);
      if (const char* m = dr::check_tensor_store(d, req_of(j), j.dst))
        return m;
      dr::store_box(d, req_of(j), j.dst, lm.data(), lm.size(), [&](uint64_t a) { return M.gptr(a); });
      break;
    }
    case J_RED: {
      const uint32_t* w = map_words(M, L, lm, j);
      uint32_t type = 0;
      if (const char* m = dr::map_red_type(j.op, w, &type))
        return m;
      const dr::Desc d = dr::decode(w);
      dr::reduce_box(d, req_of(j), j.dst, j.op, type, lm.data(), lm.size(), [&](uint64_t a) { return M.gptr(a); });
      break;
    }
    case J_COPY:
      std::memcpy(&lm[j.dst], M.maps[j.map_off / 128].w, 128);
      break;
    case J_ZERO:
      std::memset(&lm[j.dst], 0, 128);
      break;
    case J_RADDR:
      dt::replace_global_address(map_words(M, L, lm, j), j.val);
      break;
    case J_RSTRIDE:
      dt::replace_global_stride(map_words(M, L, lm, j), j.arg, j.val);
      break;
    case J_RALL: {
      uint32_t* w = map_words(M, L, lm, j);
      const rspec_t& s = L.spec;
      dt::replace_global_address(w, s.gaddr);
      dt::replace_rank(w, s.rank - 1);
      for (uint32_t i = 0; i < s.rank; ++i) {
        dt::replace_box_dim(w, i, s.box[i]);
        dt::replace_global_dim(w, i, s.gdim[i]);
      }
      for (uint32_t i = 0; i + 1 < s.rank; ++i)
        dt::replace_global_stride(w, i, s.gstride[i]);
      for (uint32_t i = 0; i < s.rank; ++i)
        dt::replace_element_stride(w, i, s.es[i]);
      dt::replace_elemtype(w, s.elemtype);
      dt::replace_interleave_layout(w, s.interleave);
      dt::replace_swizzle_mode(w, s.swizzle);
      dt::replace_fill_mode(w, s.fill);
      break;
    }
    case J_CP:
      std::memcpy(M.maps[j.map_off / 128].w, &lm[j.dst], 128);
      break;
    default:
      break;
    }
  }
  return std::string();
}

std::vector<uint8_t> lmem_image(uint32_t b) {
  std::vector<uint8_t> lm(kLmem, 0);
  for (uint32_t w = 2; w < kLmem / 4; ++w) {
    const uint32_t v = canary_word(b, w);
    std::memcpy(&lm[4 * w], &v, 4);
  }
  return lm;
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

  std::vector<uint8_t> gmem(kGBytes);
  for (uint32_t i = 0; i < kGBytes; ++i)
    gmem[i] = gbyte(i);
  vx_buffer_h gbuf = nullptr, mbuf = nullptr;
  Plan plan;
  CHECK(vx_buffer_create(dev, kGBytes, VX_MEM_READ_WRITE, &gbuf));
  CHECK(vx_buffer_address(gbuf, &plan.gbase));
  CHECK(vx_buffer_create(dev, kMaps * 128, VX_MEM_READ_WRITE, &mbuf));
  CHECK(vx_buffer_address(mbuf, &plan.mbase));

  if (neg)
    plan_neg(plan);
  else if (perf >= 0)
    plan_perf(plan);
  else
    plan_mode(plan);

  const bool llc = (isa & (VX_ISA_EXT_L2CACHE | VX_ISA_EXT_L3CACHE)) != 0;
  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA;
  const char* skip = nullptr;
  if ((isa & need) != need)
    skip = "the device needs DLMEM, the A extension and the DXA";
  else if (nw < 2 || nt < 4)
    skip = "the device has too few warps or threads";
  else if (lmem_cap < kLmem)
    skip = "the local memory is too small";
  else if (plan.mcast && (num_cores < 2 || max_cluster < 2))
    skip = "multicast needs 2 cores and a cluster of 2";
  else if (!llc && (mode == M_RED || neg == N_REDTYPE) && perf < 0)
    skip = "tensor reductions need an L2 or L3";
  else if ((plan.mbase & 63) != 0)
    skip = "the maps buffer is not 64-byte aligned";
  else if (!plan.error.empty())
    skip = plan.error.c_str();

  std::printf("dxa_tmap: %s%s%s, %zu launches, %u bytes of local memory%s\n",
              neg ? "negative " : perf >= 0 ? "perf " : "",
              neg ? kNegNames[neg] : perf >= 0 ? (perf ? "miss" : "hit") : kModeNames[mode],
              mode == M_RED && !kFpamo ? " (u32 only, no FPAMO)" : "", plan.launches.size(), kLmem,
              llc ? ", L2/L3" : "");
  if (skip) {
    std::printf("%s: %s\n", requested ? "Error" : "SKIPPED", skip);
    vx_buffer_release(mbuf);
    vx_buffer_release(gbuf);
    vx_queue_release(q);
    vx_device_release(dev);
    std::printf(requested ? "FAILED!\n" : "PASSED!\n");
    return requested ? 1 : 0;
  }

  // The host replay, ahead of the launches: it fills each load's complete_tx.
  Model M;
  M.g = gmem;
  M.maps = plan.maps;
  M.gbase = plan.gbase;
  M.mbase = plan.mbase;
  std::vector<std::vector<std::vector<uint8_t>>> lm_model;
  for (uint32_t li = 0; li < plan.launches.size(); ++li) {
    Launch& L = plan.launches[li];
    std::vector<std::vector<uint8_t>> lms;
    for (uint32_t b = 0; b < L.blocks; ++b)
      lms.push_back(lmem_image(b));
    std::string err = model_launch(M, L, lms);
    if (!neg && !err.empty()) {
      std::printf("Error: the plan trips '%s'\nFAILED!\n", err.c_str());
      return 1;
    }
    if (neg && li == 0 && err != kNegExpect[neg]) {
      std::printf("Error: the negative plan trips '%s', not '%s'\nFAILED!\n", err.c_str(), kNegExpect[neg]);
      return 1;
    }
    lm_model.push_back(lms);
  }
  if (perf >= 0 || (mode == M_PREFETCH && !neg)) {
    // Every repeat lands the same box at the same place.
    job_t& j = plan.launches[0].jobs[0];
    const dr::Desc d = dr::decode(plan.maps[0].w);
    auto read = [&](uint64_t a) { return M.g[a - plan.gbase]; };
    lm_model[0][0] = lmem_image(0);
    j.tx = dr::render(d, req_of(j), j.dst, read, lm_model[0][0].data(), kLmem);
  }

  CHECK(vx_enqueue_write(q, gbuf, 0, gmem.data(), kGBytes, 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, mbuf, 0, plan.maps.data(), kMaps * 128, 0, nullptr, nullptr));

  vx_module_h mod = nullptr;
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "tmap_k", &kernel));

  int errors = 0;
  uint64_t arg_addr0 = 0;
  uint32_t total_cycles = 0;
  for (uint32_t li = 0; li < plan.launches.size(); ++li) {
    Launch& L = plan.launches[li];
    for (auto& s : L.slots) {
      // The DCR helpers of dxa.h, which must give the map's words 0-21.
      const dr::Desc d = dr::decode(s.second.w);
      if (dh::program_2d(dev, s.first, d.base, d.size[0], d.size[1], d.stride[0], d.tile[0], d.tile[1], d.elem)
          || dh::set_tensor_options(dev, s.first, 2, d.elem, dh::Swizzle::None, dh::Interleave::None,
                                    d.cfill != 0, d.estride)) {
        std::printf("Error: cannot program DXA descriptor %u\nFAILED!\n", s.first);
        return 1;
      }
    }
    if (verbose || neg) {
      for (const job_t& j : L.jobs)
        print_job(li, j);
    }
    const uint32_t threads = L.warps * (uint32_t)nt;
    tmap_arg_t arg;
    std::memset(&arg, 0, sizeof(arg));
    arg.map        = L.pmap;
    arg.spec       = L.spec;
    arg.maps_addr  = plan.mbase;
    arg.num_jobs   = (uint32_t)L.jobs.size();
    arg.kernel     = perf >= 0 ? K_PERF : mode == M_PREFETCH && !neg ? K_PF : L.blocks == 2 ? K_MCAST : K_JOBS;
    arg.lmem_bytes = kLmem;
    arg.perf_miss  = perf == 1;
    arg.neg        = neg;
    arg.linger     = arg.kernel == K_PF ? kPfIdle : kLinger;
    arg.mc_tx      = 16 * 4 * 4;
    arg.mc_coord[0] = 4;
    arg.mc_coord[1] = 2;
    arg.mc_dst     = LM_DATA;
    std::vector<job_t> jobs = L.jobs;
    if (jobs.empty())
      jobs.push_back(job_t{});
    std::vector<uint32_t> errs((size_t)L.blocks * threads * ERR_WORDS, 0);
    std::vector<uint8_t> dump((size_t)L.blocks * kLmem, 0);
    vx_buffer_h jobs_buf = nullptr, err_buf = nullptr, dump_buf = nullptr;
    CHECK(vx_buffer_create(dev, jobs.size() * sizeof(job_t), VX_MEM_READ, &jobs_buf));
    CHECK(vx_buffer_address(jobs_buf, &arg.jobs_addr));
    CHECK(vx_enqueue_write(q, jobs_buf, 0, jobs.data(), jobs.size() * sizeof(job_t), 0, nullptr, nullptr));
    CHECK(vx_buffer_create(dev, errs.size() * 4, VX_MEM_READ_WRITE, &err_buf));
    CHECK(vx_buffer_address(err_buf, &arg.err_addr));
    CHECK(vx_enqueue_write(q, err_buf, 0, errs.data(), errs.size() * 4, 0, nullptr, nullptr));
    CHECK(vx_buffer_create(dev, dump.size(), VX_MEM_READ_WRITE, &dump_buf));
    CHECK(vx_buffer_address(dump_buf, &arg.dump_addr));

    vx_launch_info_t li_ = {};
    li_.struct_size    = sizeof(li_);
    li_.kernel         = kernel;
    li_.args_host      = &arg;
    li_.args_size      = sizeof(arg);
    li_.ndim           = 1;
    li_.grid_dim[0]    = L.blocks;
    li_.grid_dim[1]    = 1;
    li_.grid_dim[2]    = 1;
    li_.block_dim[0]   = threads;
    li_.block_dim[1]   = 1;
    li_.block_dim[2]   = 1;
    li_.lmem_size      = kLmem;
    li_.cluster_dim[0] = L.blocks;
    li_.cluster_dim[1] = 1;
    li_.cluster_dim[2] = 1;
    li_.cluster_policy = L.blocks > 1 ? VX_CLUSTER_POLICY_SPREAD : VX_CLUSTER_POLICY_DEFAULT;
    li_.cluster_flags  = 0;

    vx_event_h launch_ev = nullptr, err_ev = nullptr, dump_ev = nullptr;
    CHECK(vx_enqueue_launch(q, &li_, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, errs.data(), err_buf, 0, errs.size() * 4, 1, &launch_ev, &err_ev));
    CHECK(vx_enqueue_read(q, dump.data(), dump_buf, 0, dump.size(), 1, &launch_ev, &dump_ev));
    CHECK(vx_event_wait_value(err_ev, 1, VX_TIMEOUT_INFINITE));
    CHECK(vx_event_wait_value(dump_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(dump_ev);
    vx_event_release(err_ev);
    vx_event_release(launch_ev);

    if (neg) {
      std::printf("*** the negative run '%s' completed: the simulator must stop it with '%s'\n", kNegNames[neg],
                  kNegExpect[neg]);
      ++errors;
    }
    uint32_t unfinished = 0, gave_up = 0;
    for (uint32_t t = 0; t < L.blocks * threads; ++t) {
      unfinished += (errs[t * ERR_WORDS + ERR_DONE] != 1);
      gave_up += errs[t * ERR_WORDS + ERR_WAIT];
    }
    if (unfinished || gave_up) {
      std::printf("*** launch %u: %u threads unfinished, %u gave up waiting for an mbarrier phase\n", li,
                  unfinished, gave_up);
      ++errors;
    }
    const uint64_t arg_addr = ((uint64_t)errs[ERR_ARG_HI] << 32) | errs[ERR_ARG_LO];
    if (arg_addr & 63) {
      std::printf("*** launch %u: the argument map at %#llx is not 64-byte aligned\n", li,
                  (unsigned long long)arg_addr);
      ++errors;
    }
    if (li == 0)
      arg_addr0 = arg_addr;
    else if (mode == M_RELAUNCH && arg_addr != arg_addr0) {
      std::printf("*** launch %u: the argument moved (%#llx, then %#llx): the relaunch premise does not hold\n",
                  li, (unsigned long long)arg_addr0, (unsigned long long)arg_addr);
      ++errors;
    }
    for (uint32_t b = 0; b < L.blocks && perf < 0; ++b) {
      const std::vector<uint8_t>& want = lm_model[li][b];
      uint32_t bad = 0;
      for (uint32_t o = 8; o < kLmem; ++o) {
        const uint8_t got = dump[(size_t)b * kLmem + o];
        if (got == want[o])
          continue;
        if (bad++ < 12)
          std::printf("*** launch %u block %u: local byte +%#x: %#04x, expected %#04x\n", li, b, o, got, want[o]);
      }
      if (bad) {
        std::printf("*** launch %u block %u: %u wrong local-memory bytes\n", li, b, bad);
        ++errors;
      }
    }
    if (perf >= 0) {
      const uint32_t* box = reinterpret_cast<const uint32_t*>(&lm_model[li][0][L.jobs[0].dst]);
      uint32_t tmo = 0;
      for (uint32_t t = 0; t < threads; ++t) {
        tmo += errs[t * ERR_WORDS + ERR_TMO];
        const uint32_t got = errs[t * ERR_WORDS + ERR_BOX];
        if (got != box[t & 3]) {
          std::printf("*** thread %u: box word %u %#x, expected %#x\n", t, t & 3, got, box[t & 3]);
          ++errors;
        }
      }
      total_cycles = errs[ERR_T0];
      std::printf("dxa_tmap: %u prefetch.tensormap through %s and a load, %u cycles (%u a prefetch), "
                  "%u try_waits timed out\n", kRepeats, perf ? "maps not yet cached" : "8 cached maps",
                  total_cycles, total_cycles / kRepeats, tmo);
    }
    if (mode == M_PARAM && perf < 0 && !neg) {
      if (std::memcmp(&dump[LM_DATA], &dump[LM_DATA + 512], 512)) {
        std::printf("*** the map load and the slot load of the same box differ\n");
        ++errors;
      }
    }
    if (arg.kernel == K_PF) {
      // A descriptor fetch is at least two memory reads: the prefetched and
      // cached loads must each save kPfMinSave cycles a load.
      const uint32_t n = DXA_TMAP_ENTRIES, cold = errs[ERR_T0], pref = errs[ERR_T0 + 1], hit = errs[ERR_T0 + 2];
      std::printf("dxa_tmap: %u loads: through maps not cached %u cycles, prefetched %u, cached %u\n", n, cold,
                  pref, hit);
      if (pref + n * kPfMinSave > cold || hit + n * kPfMinSave > cold) {
        std::printf("*** the prefetched or cached loads are not faster than the cold ones\n");
        ++errors;
      }
    }
    vx_buffer_release(dump_buf);
    vx_buffer_release(err_buf);
    vx_buffer_release(jobs_buf);
  }

  std::vector<uint8_t> got(kGBytes, 0);
  std::vector<vx_tensor_map_t> got_maps(kMaps);
  {
    vx_event_h e1 = nullptr, e2 = nullptr;
    CHECK(vx_enqueue_read(q, got.data(), gbuf, 0, kGBytes, 0, nullptr, &e1));
    CHECK(vx_enqueue_read(q, got_maps.data(), mbuf, 0, kMaps * 128, 0, nullptr, &e2));
    CHECK(vx_event_wait_value(e1, 1, VX_TIMEOUT_INFINITE));
    CHECK(vx_event_wait_value(e2, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e1);
    vx_event_release(e2);
  }
  if (M.bad) {
    std::printf("*** the model wrote outside the global buffer\n");
    ++errors;
  }
  uint32_t bad = 0, changed = 0;
  for (uint32_t o = 0; o < kGBytes; ++o) {
    changed += (M.g[o] != gmem[o]);
    if (got[o] == M.g[o])
      continue;
    if (bad++ < 12)
      std::printf("*** global byte +%#x (%s): %#04x, expected %#04x\n", o,
                  M.g[o] == gmem[o] ? "must be untouched" : "written", got[o], M.g[o]);
  }
  if (bad) {
    std::printf("*** %u wrong bytes in the global buffer\n", bad);
    ++errors;
  }
  uint32_t bad_words = 0;
  for (uint32_t i = 0; i < kMaps; ++i) {
    for (uint32_t k = 0; k < TMAP_WORDS; ++k) {
      if (got_maps[i].w[k] == M.maps[i].w[k])
        continue;
      if (bad_words++ < 12)
        std::printf("*** map %u word %u: %#x, expected %#x\n", i, k, got_maps[i].w[k], M.maps[i].w[k]);
    }
  }
  if (bad_words) {
    std::printf("*** %u wrong words in the maps buffer\n", bad_words);
    ++errors;
  }
  if ((mode == M_RZERO || mode == M_RPARTIAL) && perf < 0 && !neg) {
    // The kernel's map from a zero buffer equals the host encoder's.
    const rspec_t& s = plan.launches[0].spec;
    uint64_t gdim[5], gstr[4];
    for (uint32_t i = 0; i < 5; ++i)
      gdim[i] = s.gdim[i];
    for (uint32_t i = 0; i < 4; ++i)
      gstr[i] = s.gstride[i];
    alignas(64) vx_tensor_map_t e;
    const int r = dh::encode_tiled(&e, dh::TensorMapDataType::BFloat16, s.rank, s.gaddr, gdim, gstr, s.box, s.es,
                                   dh::TensorMapInterleave::None, dh::TensorMapSwizzle::B32,
                                   dh::TensorMapL2Promotion::None, dh::TensorMapFloatOOBfill::NanRequestZeroFma);
    uint32_t diff = 0;
    for (uint32_t k = 0; k < TMAP_WORDS; ++k) {
      if (got_maps[2].w[k] != e.w[k] && diff++ < 8)
        std::printf("*** replace-zero word %u: %#x, the encoder gives %#x\n", k, got_maps[2].w[k], e.w[k]);
    }
    if (r != VX_SUCCESS || diff) {
      std::printf("*** the replaced map differs from the encoder's (%u words, encode %d)\n", diff, r);
      ++errors;
    }
  }
  if (perf < 0 && !neg)
    std::printf("dxa_tmap: %zu launches, %u global bytes changed, every byte checked\n", plan.launches.size(),
                changed);

  vx_kernel_release(kernel);
  vx_module_release(mod);
  vx_buffer_release(mbuf);
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
