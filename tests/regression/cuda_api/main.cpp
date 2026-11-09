// cuda_api: the CUDA porting layer (vx_cuda.h, vx_cuda_fp.h, vx_cuda_runtime.h)
// at the ports' shape: XLEN 64, 32-thread warps, DLMEM, A, Zacas, mbarriers, the
// DXA and FPAMO, an L2. Every kernel uses CUDA names only, and is launched
// through vx::cuda::launch_kernel_ex; every result is checked byte-exact
// against a host reference written from the CUDA / PTX semantics.
//
//   -m cg       cooperative-groups identity: grid, block, cluster, tile ranks
//               and sizes, map_shared_rank, invoke_one, for a 1D and a 2D
//               cluster and a launch without one (-c x,y,z: that shape only)
//   -m warp     shuffles (4 kinds x 6 widths x 6 lanes x 5 types), votes in
//               divergent branches, elect_sync, __reduce_*_sync,
//               __match_*_sync, lane masks, thread_block_tile<32> members
//   -m atomic   every atomic op and type on global, own and peer local memory
//               from every thread: final values and the returned old values
//               (a serialization must exist); _block and _system forms
//   -m barrier  cuda::barrier phases with arrive_and_drop, barrier_arrive_tx /
//               barrier_expect_tx with memcpy_async_tx, cuda::memcpy_async
//               (-s mask: 1 phases, 2 tx copies, 4 memcpy_async, 8 the group
//               form, the per-lane bulk loop and a byte copy)
//   -m ptx      every cuda::ptx name with its effect
//   -m sync     __syncthreads_count/and/or, __syncwarp ordering
//   -m types    the number types on the device equal the host's, bit for bit
//   -m launch   launch_kernel_ex with __cluster_dims__, func_set_attribute,
//               occupancy and tensor-map wrappers
//   -p          the parity run: warp reductions, shuffles, integer atomics
//   -r <n>      rounds of -p (default P_ROUNDS)
//   -x 1|2|3    runs that must fail: the kernel entry check traps a launch
//               whose cluster is not the kernel's __cluster_dims__ (1);
//               launch_kernel_ex refuses dynamic shared memory past the
//               recorded limit (2) and a cluster other than __cluster_dims__ (3)
//   -v          print every mismatch

#include <vortex2.h>
#include <vx_cuda_runtime.h>
#include <VX_types.h>
#include <vx_cuda_fp.h>
#include "common.h"
#include "types_ops.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace vc = vx::cuda;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

enum Mode { M_CG, M_WARP, M_ATOMIC, M_BARRIER, M_PTX, M_SYNC, M_TYPES, M_LAUNCH, NUM_MODES };
const char* const kModeNames[] = {"cg", "warp", "atomic", "barrier", "ptx", "sync", "types", "launch"};

uint32_t mode = M_CG;
int neg = 0;
bool perf = false;
uint32_t rounds = P_ROUNDS;
uint32_t parts = ~0u;
bool verbose = false;
bool have_cluster = false;
vc::dim3 opt_cluster;

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;
vx_module_h mod = nullptr;
uint64_t num_cores = 0, nw = 0, nt = 0, lmem_cap = 0, max_cluster = 0, isa = 0;
int failures = 0;

void usage() {
  std::printf("Usage: [-m cg|warp|atomic|barrier|ptx|sync|types|launch] [-c x,y,z] [-s parts] [-p] [-r rounds] "
              "[-x 1|2|3] [-v] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  bool have_mode = false;
  while ((c = getopt(argc, argv, "m:c:pr:s:x:vh")) != -1) {
    switch (c) {
    case 'm': {
      uint32_t i = 0;
      while (i < NUM_MODES && std::strcmp(optarg, kModeNames[i]))
        ++i;
      if (i == NUM_MODES) { usage(); std::exit(-1); }
      mode = i;
      have_mode = true;
      break;
    }
    case 'c': {
      unsigned x = 0, y = 0, z = 0;
      if (std::sscanf(optarg, "%u,%u,%u", &x, &y, &z) != 3 || !x || !y || !z) { usage(); std::exit(-1); }
      opt_cluster = vc::dim3(x, y, z);
      have_cluster = true;
      break;
    }
    case 'p': perf = true; break;
    case 'r': rounds = (uint32_t)std::atoi(optarg); break;
    case 's': parts = (uint32_t)std::strtoul(optarg, nullptr, 0); break;
    case 'x':
      neg = std::atoi(optarg);
      if (neg < 1 || neg > 3) { usage(); std::exit(-1); }
      break;
    case 'v': verbose = true; break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || (perf && (neg || have_mode)) || (neg && have_mode) || (have_cluster && mode != M_CG)
      || (parts != ~0u && mode != M_BARRIER)) {
    usage();
    std::exit(-1);
  }
}

const char* prefix() { return perf ? "perf" : neg ? "negative" : kModeNames[mode]; }

void report(const std::string& name, uint64_t bad, uint64_t checked) {
  if (bad) {
    std::printf("FAIL %s/%s: %llu of %llu wrong\n", prefix(), name.c_str(), (unsigned long long)bad,
                (unsigned long long)checked);
    ++failures;
  } else {
    std::printf("PASS %s/%s (%llu checked)\n", prefix(), name.c_str(), (unsigned long long)checked);
  }
}

// A per-check mismatch counter that prints the first few.
struct Check {
  std::string name;
  uint64_t bad = 0, n = 0;
  explicit Check(const std::string& s) : name(s) {}
  void eq(uint64_t got, uint64_t want, const char* what, uint32_t i, uint32_t j = 0) {
    ++n;
    if (got == want)
      return;
    if (bad++ < 8 || verbose)
      std::printf("  *** %s %s [%u,%u]: %#llx, expected %#llx\n", name.c_str(), what, i, j,
                  (unsigned long long)got, (unsigned long long)want);
  }
  ~Check() { report(name, bad, n); }
};

struct Buf {
  vx_buffer_h h = nullptr;
  uint64_t addr = 0, size = 0;
  explicit Buf(uint64_t n) : size(n ? n : 64) {
    CHECK(vx_buffer_create(dev, size, VX_MEM_READ_WRITE, &h));
    CHECK(vx_buffer_address(h, &addr));
  }
  ~Buf() { vx_buffer_release(h); }
  void write(const void* p, uint64_t n) {
    vx_event_h e = nullptr;
    CHECK(vx_enqueue_write(q, h, 0, p, n, 0, nullptr, &e));
    CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e);
  }
  void zero() {
    std::vector<uint8_t> z(size, 0);
    write(z.data(), size);
  }
  template <typename T>
  std::vector<T> read() const {
    std::vector<T> v(size / sizeof(T));
    vx_event_h e = nullptr;
    CHECK(vx_enqueue_read(q, v.data(), h, 0, v.size() * sizeof(T), 0, nullptr, &e));
    CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e);
    return v;
  }
};

vx_kernel_h kernel(const char* name) {
  vx_kernel_h k = nullptr;
  CHECK(vx_module_get_kernel(mod, name, &k));
  return k;
}

vc::launch_config_t config(vc::dim3 grid, vc::dim3 block, vc::dim3 cluster, uint32_t dyn) {
  vc::launch_config_t c;
  c.grid = grid;
  c.block = block;
  c.cluster = cluster;
  c.dyn_smem = dyn;
  return c;
}

// Launches and waits; returns the launch's own result.
template <typename D = vc::detail::no_cluster_dims>
vx_result_t run(vx_kernel_h k, const vc::launch_config_t& cfg, const cuda_arg_t& arg) {
  vx_event_h ev = nullptr;
  vx_result_t r = vc::launch_kernel_ex<D>(q, k, cfg, arg, &ev);
  if (r != VX_SUCCESS)
    return r;
  CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);
  return VX_SUCCESS;
}

cuda_arg_t new_arg() {
  cuda_arg_t a;
  std::memset(&a, 0, sizeof(a));
  return a;
}

uint32_t vol(const vc::dim3& d) { return d.x * d.y * d.z; }

// __match_all_sync over a warp with exited lanes returns the live part of the mask.
constexpr bool kMatchAllLive = true;

// ── cg ───────────────────────────────────────────────────────────────
void cg_launch(vx_kernel_h k, vc::dim3 cluster, vc::dim3 grid, const char* tag) {
  const vc::dim3 block(CG_BLOCK_X, CG_BLOCK_Y, CG_BLOCK_Z);
  const uint32_t bt = vol(block), nb = vol(grid), threads = bt * nb;
  const vc::dim3 c = (cluster == vc::dim3(0, 0, 0)) ? vc::dim3(1, 1, 1) : cluster;
  const vc::dim3 ncl(grid.x / c.x, grid.y / c.y, grid.z / c.z);
  Buf out((uint64_t)threads * CG_WORDS * 8), aux(nb * 8), dump(64);
  out.zero();
  aux.zero();
  cuda_arg_t arg = new_arg();
  arg.out = out.addr;
  arg.aux = aux.addr;
  arg.dump = dump.addr;
  arg.stride = CG_WORDS;
  arg.lmem = CG_LMEM;
  CHECK(run(k, config(grid, block, cluster, CG_LMEM), arg));
  const std::vector<uint64_t> o = out.read<uint64_t>();
  const std::vector<uint32_t> cnt = aux.read<uint32_t>();
  Check ck(std::string("identity-") + tag);
  for (uint32_t bz = 0; bz < grid.z; ++bz)
  for (uint32_t by = 0; by < grid.y; ++by)
  for (uint32_t bx = 0; bx < grid.x; ++bx) {
    const uint32_t br = bx + grid.x * (by + grid.y * bz);
    const vc::dim3 ci(bx % c.x, by % c.y, bz % c.z), cid(bx / c.x, by / c.y, bz / c.z);
    const uint32_t crank = ci.x + c.x * (ci.y + c.y * ci.z), clrank = cid.x + ncl.x * (cid.y + ncl.y * cid.z);
    const uint32_t peer_rank = (crank + 1) % vol(c);
    const vc::dim3 pi(peer_rank % c.x, (peer_rank / c.x) % c.y, peer_rank / (c.x * c.y));
    const uint32_t peer_br = (cid.x * c.x + pi.x) + grid.x * ((cid.y * c.y + pi.y) + grid.y * (cid.z * c.z + pi.z));
    for (uint32_t tz = 0; tz < block.z; ++tz)
    for (uint32_t ty = 0; ty < block.y; ++ty)
    for (uint32_t tx = 0; tx < block.x; ++tx) {
      const uint32_t tr = tx + block.x * (ty + block.y * tz), g = br * bt + tr;
      const uint64_t want[] = {
        g, threads, br, nb, grid.x, grid.y, grid.z, bx, by, bz, clrank, vol(ncl), ncl.x, ncl.y, ncl.z, 1,
        tr, bt, bt, bx, by, bz, tx, ty, tz, block.x, block.y, block.z, block.x, block.y, block.z,
        crank, vol(c), c.x, c.y, c.z, ci.x, ci.y, ci.z, crank * bt + tr, bt * vol(c),
        block.x * c.x, block.y * c.y, block.z * c.z, 1,
        0x1000u + peer_br, peer_rank,
        tr % 32, 32, tr / 32, (bt + 31) / 32, threads, 0xC0DE};
      static const char* const names[] = {
        "grid.thread_rank", "grid.num_threads", "grid.block_rank", "grid.num_blocks", "grid.dim_blocks.x",
        "grid.dim_blocks.y", "grid.dim_blocks.z", "grid.block_index.x", "grid.block_index.y",
        "grid.block_index.z", "grid.cluster_rank", "grid.num_clusters", "grid.dim_clusters.x",
        "grid.dim_clusters.y", "grid.dim_clusters.z", "grid.is_valid", "block.thread_rank",
        "block.num_threads", "block.size", "block.group_index.x", "block.group_index.y", "block.group_index.z",
        "block.thread_index.x", "block.thread_index.y", "block.thread_index.z", "block.dim_threads.x",
        "block.dim_threads.y", "block.dim_threads.z", "block.group_dim.x", "block.group_dim.y",
        "block.group_dim.z", "cluster.block_rank", "cluster.num_blocks", "cluster.dim_blocks.x",
        "cluster.dim_blocks.y", "cluster.dim_blocks.z", "cluster.block_index.x", "cluster.block_index.y",
        "cluster.block_index.z", "cluster.thread_rank", "cluster.num_threads", "cluster.dim_threads.x",
        "cluster.dim_threads.y", "cluster.dim_threads.z", "cluster.is_valid", "map_shared_rank(peer)",
        "query_shared_rank", "tile.thread_rank", "tile.size", "tile.meta_group_rank", "tile.meta_group_size",
        "grid.size", "done"};
      static_assert(sizeof(want) / sizeof(want[0]) == sizeof(names) / sizeof(names[0]), "one name a word");
      for (uint32_t w = 0; w < sizeof(want) / sizeof(want[0]); ++w)
        ck.eq(o[(uint64_t)g * CG_WORDS + w], want[w], names[w], g, w);
    }
  }
  Check ci(std::string("invoke_one-") + tag);
  for (uint32_t b = 0; b < nb; ++b) {
    ci.eq(cnt[2 * b], 1, "block", b);
    ci.eq(cnt[2 * b + 1], (bt + 31) / 32, "tiles", b);
  }
}

void mode_cg() {
  vx_kernel_h k = kernel("cg_k");
  if (have_cluster) {
    const vc::dim3 c = opt_cluster;
    char tag[32];
    std::snprintf(tag, sizeof(tag), "%ux%ux%u", c.x, c.y, c.z);
    cg_launch(k, c, vc::dim3(2 * c.x, c.y, c.z > 1 ? c.z : 2), tag);
  } else {
    cg_launch(k, vc::dim3(2, 1, 1), vc::dim3(4, 1, 1), "1d");
    cg_launch(k, vc::dim3(1, 2, 1), vc::dim3(2, 2, 1), "2d");
    cg_launch(k, vc::dim3(0, 0, 0), vc::dim3(2, 1, 1), "none");
  }
  vx_kernel_release(k);
}

// ── warp ─────────────────────────────────────────────────────────────
// PTX shfl.sync: the source lane and whether it is in range.
uint32_t shfl_src(uint32_t kind, uint32_t lane, uint32_t b, uint32_t width) {
  const uint32_t segmask = (32 - width) & 0x1f, clamp = kind == 1 ? 0 : 0x1f;
  const int32_t max_lane = (int32_t)((lane & segmask) | (clamp & ~segmask)), min_lane = (int32_t)(lane & segmask);
  const int32_t bv = (int32_t)(b & 0x1f), l = (int32_t)lane;
  int32_t j;
  bool ok;
  switch (kind) {
  case 0:  j = min_lane | (bv & ~(int32_t)segmask); ok = j <= max_lane; break;
  case 1:  j = l - bv; ok = j >= max_lane; break;
  case 2:  j = l + bv; ok = j <= max_lane; break;
  default: j = l ^ bv; ok = j <= max_lane; break;
  }
  return ok ? (uint32_t)j : lane;
}

// Threads 48-63 of a 64-thread block return at the kernel's start: warp 1
// keeps 16 live lanes. Every full-mask warp op acts over the live lanes.
void mode_warp_tail() {
  vx_kernel_h k = kernel("tail_k");
  Buf out((uint64_t)2 * WARP * WT_WORDS * 8), dump(64);
  out.zero();
  cuda_arg_t arg = new_arg();
  arg.out = out.addr;
  arg.dump = dump.addr;
  arg.stride = WT_WORDS;
  CHECK(run(k, config(vc::dim3(1), vc::dim3(2 * WARP), vc::dim3(0, 0, 0), 0), arg));
  const std::vector<uint64_t> o = out.read<uint64_t>();
  Check ck("exited lanes (threads 48-63 returned)");
  static const char* const rn[] = {"reduce add u32", "reduce add s32", "reduce min u32", "reduce min s32",
                                   "reduce max u32", "reduce max s32", "reduce and", "reduce or", "reduce xor"};
  for (uint32_t t = 0; t < WT_THREADS; ++t) {
    const uint32_t lane = t % WARP, base = t - lane;
    const uint32_t live = WT_THREADS - base >= WARP ? 0xffffffffu : (1u << (WT_THREADS - base)) - 1;
    auto at = [&](uint32_t w) { return o[(uint64_t)t * WT_WORDS + w]; };
    auto alive = [&](uint32_t l) { return ((live >> l) & 1) != 0; };
    for (uint32_t op = 0; op < W_NROP; ++op) {
      uint32_t r = 0;
      bool first = true;
      for (uint32_t l = 0; l < WARP; ++l) {
        if (!alive(l))
          continue;
        const uint32_t v = w_rvalue(0, base + l);
        if (first) { r = v; first = false; continue; }
        switch (op) {
        case 0: case 1: r += v; break;
        case 2: r = std::min(r, v); break;
        case 3: r = (uint32_t)std::min((int32_t)r, (int32_t)v); break;
        case 4: r = std::max(r, v); break;
        case 5: r = (uint32_t)std::max((int32_t)r, (int32_t)v); break;
        case 6: r &= v; break;
        case 7: r |= v; break;
        default: r ^= v; break;
        }
      }
      ck.eq(at(WT_RED + op), r, rn[op], t);
    }
    uint32_t any = 0, ballot = 0;
    for (uint32_t l = 0; l < WARP; ++l) {
      if (!alive(l))
        continue;
      any |= (l % 3 == lane % 3) ? (1u << l) : 0;
      ballot |= w_pred(3, base + l) ? (1u << l) : 0;
    }
    ck.eq(at(WT_MATCH + 0), any, "match_any 32", t);
    ck.eq(at(WT_MATCH + 1), any, "match_any 64", t);
    ck.eq(at(WT_MATCH + 2), kMatchAllLive ? live : 0xffffffffu, "match_all equal", t);
    ck.eq(at(WT_MATCH + 3), 1, "match_all equal pred", t);
    ck.eq(at(WT_MATCH + 4), 0, "match_all distinct", t);
    ck.eq(at(WT_MATCH + 5), 0, "match_all distinct pred", t);
    auto v = [&](uint32_t l) { return (uint32_t)w_value(0, base + l); };
    auto src = [&](uint32_t l) { return l < WARP && alive(l) ? v(l) : v(lane); };
    ck.eq(at(WT_VOTE + 0), ballot, "ballot", t);
    ck.eq(at(WT_VOTE + 1), live, "activemask", t);
    ck.eq(at(WT_VOTE + 2), src((lane + 5) % WARP), "shfl from an exited lane: own value", t);
    ck.eq(at(WT_VOTE + 3), src(lane ^ 1), "shfl_xor", t);
    ck.eq(at(WT_VOTE + 4), src(lane + 8), "shfl_down", t);
  }
  vx_kernel_release(k);
}

void mode_warp() {
  vx_kernel_h k = kernel("warp_k");
  const uint32_t threads = W_WARPS * WARP;
  Buf out((uint64_t)threads * W_WORDS * 8), dump(64);
  out.zero();
  cuda_arg_t arg = new_arg();
  arg.out = out.addr;
  arg.dump = dump.addr;
  arg.stride = W_WORDS;
  CHECK(run(k, config(vc::dim3(1), vc::dim3(threads), vc::dim3(0, 0, 0), 0), arg));
  const std::vector<uint64_t> o = out.read<uint64_t>();
  auto at = [&](uint32_t t, uint32_t w) { return o[(uint64_t)t * W_WORDS + w]; };
  static const char* const kinds[] = {"__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync"};
  static const char* const types[] = {"unsigned", "unsigned long long", "float", "double", "__half2"};
  for (uint32_t kind = 0; kind < W_NSHFL; ++kind) {
    Check ck(std::string("shuffle-") + kinds[kind]);
    for (uint32_t type = 0; type < W_NTYPE; ++type)
    for (uint32_t wi = 0; wi < W_NWIDTH; ++wi)
    for (uint32_t di = 0; di < W_NDELTA; ++di) {
      const uint32_t w = w_width(wi), d = w_delta(di, w);
      for (uint32_t t = 0; t < threads; ++t) {
        const uint32_t lane = t % WARP, base = t - lane;
        const uint64_t want = w_value(type, base + shfl_src(kind, lane, d, w));
        char what[96];
        std::snprintf(what, sizeof(what), "%s width %u lane/delta %u", types[type], w, d);
        ck.eq(at(t, W_SHFL + ((kind * W_NWIDTH + wi) * W_NDELTA + di) * W_NTYPE + type), want, what, t);
      }
    }
  }
  {
    Check ck("vote");
    static const char* const vn[] = {"ballot", "any", "all", "activemask", "all(true)", "any(false)"};
    for (uint32_t c = 0; c < W_NVCASE; ++c) {
      const uint32_t m = w_vcase_mask(c);
      for (uint32_t t = 0; t < threads; ++t) {
        const uint32_t lane = t % WARP, base = t - lane;
        uint32_t ballot = 0;
        for (uint32_t l = 0; l < WARP; ++l)
          ballot |= ((m >> l) & 1) && w_pred(c, base + l) ? (1u << l) : 0;
        const bool in = (m >> lane) & 1;
        const uint64_t want[W_VOTE_N] = {ballot, ballot != 0, ballot == m, m, 1, 0};
        for (uint32_t i = 0; i < W_VOTE_N; ++i)
          ck.eq(at(t, W_VOTE + c * W_VOTE_N + i), in ? want[i] : ~0ull, vn[i], t, c);
      }
    }
  }
  {
    Check ck("elect_sync");
    for (uint32_t t = 0; t < threads; ++t) {
      const uint32_t lane = t % WARP;
      ck.eq(at(t, W_ELECT + 0), lane == 0, "full mask", t);
      ck.eq(at(t, W_ELECT + 1), (lane & 1) ? lane == 1 : 2, "odd lanes", t);
      ck.eq(at(t, W_ELECT + 2), (lane >= 4 && lane < 8) ? lane == 4 : 2, "lanes 4-7", t);
      ck.eq(at(t, W_ELECT + 3), lane == 0, "elect_one_sync", t);
    }
  }
  {
    Check ck("reduce");
    static const char* const rn[] = {"add u32", "add s32", "min u32", "min s32", "max u32", "max s32", "and",
                                     "or", "xor"};
    for (uint32_t op = 0; op < W_NROP; ++op)
    for (uint32_t c = 0; c < W_NRCASE; ++c) {
      const uint32_t m = w_rmask(c);
      for (uint32_t t = 0; t < threads; ++t) {
        const uint32_t lane = t % WARP, base = t - lane;
        uint32_t r = 0;
        bool first = true;
        for (uint32_t l = 0; l < WARP; ++l) {
          if (!((m >> l) & 1))
            continue;
          const uint32_t v = w_rvalue(c, base + l);
          if (first) { r = v; first = false; continue; }
          switch (op) {
          case 0: case 1: r += v; break;
          case 2: r = std::min(r, v); break;
          case 3: r = (uint32_t)std::min((int32_t)r, (int32_t)v); break;
          case 4: r = std::max(r, v); break;
          case 5: r = (uint32_t)std::max((int32_t)r, (int32_t)v); break;
          case 6: r &= v; break;
          case 7: r |= v; break;
          default: r ^= v; break;
          }
        }
        const bool in = (m >> lane) & 1;
        ck.eq(at(t, W_RED + op * W_NRCASE + c), in ? r : ~0ull, rn[op], t, c);
      }
    }
  }
  {
    Check ck("match");
    for (uint32_t wide = 0; wide < 2; ++wide)
    for (uint32_t c = 0; c < W_NMCASE; ++c)
    for (uint32_t t = 0; t < threads; ++t) {
      const uint32_t lane = t % WARP;
      uint32_t any = 0;
      for (uint32_t l = 0; l < WARP; ++l)
        any |= w_key(wide, c, l) == w_key(wide, c, lane) ? (1u << l) : 0;
      const bool all = any == 0xffffffffu;
      const uint32_t base = W_MATCH + (wide * W_NMCASE + c) * W_MATCH_N;
      ck.eq(at(t, base + 0), any, wide ? "match_any 64" : "match_any 32", t, c);
      ck.eq(at(t, base + 1), all ? 0xffffffffu : 0, wide ? "match_all 64" : "match_all 32", t, c);
      ck.eq(at(t, base + 2), all ? 1 : 0, "match_all pred", t, c);
    }
  }
  {
    Check ck("lanemask");
    for (uint32_t t = 0; t < threads; ++t) {
      const uint32_t l = t % WARP;
      const uint32_t eq = 1u << l, lt = eq - 1, le = lt | eq;
      const uint64_t want[W_LANE_N] = {eq, lt, le, ~le, ~lt, l};
      static const char* const ln[] = {"eq", "lt", "le", "gt", "ge", "laneid"};
      for (uint32_t i = 0; i < W_LANE_N; ++i)
        ck.eq(at(t, W_LANE + i), want[i], ln[i], t);
    }
  }
  {
    Check ck("tile");
    for (uint32_t t = 0; t < threads; ++t) {
      const uint32_t l = t % WARP, base = t - l;
      auto v = [&](uint32_t x) { return (uint32_t)w_value(0, base + x); };
      uint32_t ballot = 0;
      for (uint32_t x = 0; x < WARP; ++x)
        ballot |= w_pred(3, base + x) ? (1u << x) : 0;
      const uint64_t want[W_TILE_N] = {v((l + 5) % WARP), l >= 2 ? v(l - 2) : v(l), l + 3 < WARP ? v(l + 3) : v(l),
                                       v(l ^ 6), ballot != 0, ballot == 0xffffffffu, ballot, l, v(5)};
      static const char* const tn[] = {"shfl", "shfl_up", "shfl_down", "shfl_xor", "any", "all", "ballot",
                                       "thread_rank", "__shfl_sync width 32"};
      for (uint32_t i = 0; i < W_TILE_N; ++i)
        ck.eq(at(t, W_TILE + i), want[i], tn[i], t);
    }
  }
  vx_kernel_release(k);
  mode_warp_tail();
}

// ── atomic ───────────────────────────────────────────────────────────
float f32(uint64_t b) { float f; uint32_t x = (uint32_t)b; std::memcpy(&f, &x, 4); return f; }
uint64_t bf32(float f) { uint32_t x; std::memcpy(&x, &f, 4); return x; }
double f64(uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }
uint64_t bf64(double d) { uint64_t x; std::memcpy(&x, &d, 8); return x; }
uint16_t h_add(uint16_t a, uint16_t b) {
  return __half_as_ushort(__float2half(__half2float(__ushort_as_half(a)) + __half2float(__ushort_as_half(b))));
}
uint16_t bh_add(uint16_t a, uint16_t b) {
  return __bfloat16_as_ushort(__float2bfloat16(__bfloat162float(__ushort_as_bfloat16(a))
                                               + __bfloat162float(__ushort_as_bfloat16(b))));
}

// The slot after one atomic of `op` with value v (CAS: compare = init).
uint64_t a_apply(uint32_t op, uint64_t s, uint64_t v, uint64_t init) {
  const uint32_t s32 = (uint32_t)s, v32 = (uint32_t)v;
  const uint64_t hi = s & 0xFFFFFFFF00000000ull;
  switch (op) {
  case A_ADD_S32: case A_ADD_U32: return hi | (uint32_t)(s32 + v32);
  case A_ADD_U64: return s + v;
  case A_ADD_F32: return hi | bf32(f32(s) + f32(v));
  case A_ADD_F64: return bf64(f64(s) + f64(v));
  case A_ADD_F16: return (s & ~0xFFFFull) | h_add((uint16_t)s, (uint16_t)v);
  case A_ADD_F16X2: return hi | h_add((uint16_t)s, (uint16_t)v) | ((uint32_t)h_add((uint16_t)(s >> 16), (uint16_t)(v >> 16)) << 16);
  case A_ADD_BF16: return (s & ~0xFFFFull) | bh_add((uint16_t)s, (uint16_t)v);
  case A_ADD_BF16X2: return hi | bh_add((uint16_t)s, (uint16_t)v) | ((uint32_t)bh_add((uint16_t)(s >> 16), (uint16_t)(v >> 16)) << 16);
  case A_SUB_S32: case A_SUB_U32: return hi | (uint32_t)(s32 - v32);
  case A_EXCH_S32: case A_EXCH_U32: case A_EXCH_F32: return hi | v32;
  case A_EXCH_U64: return v;
  case A_MIN_S32: return hi | (uint32_t)std::min((int32_t)s32, (int32_t)v32);
  case A_MIN_U32: return hi | std::min(s32, v32);
  case A_MIN_S64: return (uint64_t)std::min((int64_t)s, (int64_t)v);
  case A_MIN_U64: return std::min(s, v);
  case A_MAX_S32: return hi | (uint32_t)std::max((int32_t)s32, (int32_t)v32);
  case A_MAX_U32: return hi | std::max(s32, v32);
  case A_MAX_S64: return (uint64_t)std::max((int64_t)s, (int64_t)v);
  case A_MAX_U64: return std::max(s, v);
  case A_INC_U32: return hi | (s32 >= v32 ? 0u : s32 + 1);
  case A_DEC_U32: return hi | ((s32 == 0 || s32 > v32) ? v32 : s32 - 1);
  case A_AND_S32: case A_AND_U32: return hi | (s32 & v32);
  case A_AND_U64: return s & v;
  case A_OR_S32: case A_OR_U32: return hi | (s32 | v32);
  case A_OR_U64: return s | v;
  case A_XOR_S32: case A_XOR_U32: return hi | (s32 ^ v32);
  case A_XOR_U64: return s ^ v;
  case A_CAS_S32: case A_CAS_U32: return s32 == (uint32_t)init ? (hi | v32) : s;
  case A_CAS_U64: return s == init ? v : s;
  case A_CAS_U16: return (uint16_t)s == (uint16_t)init ? ((s & ~0xFFFFull) | (uint16_t)v) : s;
  default: return (uint16_t)(s >> 16) == (uint16_t)init ? ((s & ~0xFFFF0000ull) | ((uint64_t)(uint16_t)v << 16)) : s;
  }
}

// The bits an atomic returns (the old value of its own width).
uint64_t a_old_bits(uint32_t op, uint64_t s) {
  switch (op) {
  case A_ADD_U64: case A_ADD_F64: case A_EXCH_U64: case A_MIN_S64: case A_MIN_U64: case A_MAX_S64:
  case A_MAX_U64: case A_AND_U64: case A_OR_U64: case A_XOR_U64: case A_CAS_U64:
    return s;
  case A_ADD_F16: case A_ADD_BF16: case A_CAS_U16:
    return s & 0xFFFF;
  case A_CAS_U16H:
    return (s >> 16) & 0xFFFF;
  default:
    return s & 0xFFFFFFFF;
  }
}

// A serialization of the hits that returns each hit's old value and ends at
// `final`: hits that leave the slot unchanged are applied at once, the
// others are tried one distinct value at a time.
bool serializable(uint32_t op, uint64_t init, uint64_t state, std::vector<std::pair<uint64_t, uint64_t>>& left,
                  uint64_t final) {
  for (bool again = true; again;) {
    again = false;
    for (size_t i = 0; i < left.size(); ++i) {
      if (left[i].second == a_old_bits(op, state) && a_apply(op, state, left[i].first, init) == state) {
        left[i] = left.back();
        left.pop_back();
        again = true;
        break;
      }
    }
  }
  if (left.empty())
    return state == final;
  std::vector<uint64_t> tried;
  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i].second != a_old_bits(op, state))
      continue;
    const uint64_t v = left[i].first;
    if (std::find(tried.begin(), tried.end(), v) != tried.end())
      continue;
    tried.push_back(v);
    std::vector<std::pair<uint64_t, uint64_t>> rest = left;
    rest[i] = rest.back();
    rest.pop_back();
    if (serializable(op, init, a_apply(op, state, v, init), rest, final))
      return true;
  }
  return false;
}

void mode_atomic() {
  vx_kernel_h k = kernel("atomic_k");
  const uint32_t bt = A_WARPS * WARP, threads = 2 * bt;
  Buf out((uint64_t)threads * A_TARGETS * 8), aux(A_TARGETS * 8), dump(2 * 1024);
  out.zero();
  dump.zero();
  std::vector<uint64_t> g0(A_TARGETS, 0);
  for (uint32_t tg = 0; tg < A_TARGETS; ++tg)
    g0[tg] = a_slot_init(a_op(tg), tg);
  aux.write(g0.data(), g0.size() * 8);
  cuda_arg_t arg = new_arg();
  arg.out = out.addr;
  arg.aux = aux.addr;
  arg.dump = dump.addr;
  arg.stride = A_TARGETS;
  arg.lmem = 1024;
  CHECK(run(k, config(vc::dim3(2), vc::dim3(bt), vc::dim3(2, 1, 1), 1024), arg));
  const std::vector<uint64_t> o = out.read<uint64_t>(), g = aux.read<uint64_t>(), d = dump.read<uint64_t>();
  static const char* const names[A_NUM_OPS] = {
    "atomicAdd int", "atomicAdd unsigned", "atomicAdd ull", "atomicAdd float", "atomicAdd double",
    "atomicAdd __half", "atomicAdd __half2", "atomicAdd bf16", "atomicAdd bf162", "atomicSub int",
    "atomicSub unsigned", "atomicExch int", "atomicExch unsigned", "atomicExch ull", "atomicExch float",
    "atomicMin int", "atomicMin unsigned", "atomicMin ll", "atomicMin ull", "atomicMax int", "atomicMax unsigned",
    "atomicMax ll", "atomicMax ull", "atomicInc", "atomicDec", "atomicAnd int", "atomicAnd unsigned",
    "atomicAnd ull", "atomicOr int", "atomicOr unsigned", "atomicOr ull", "atomicXor int", "atomicXor unsigned",
    "atomicXor ull", "atomicCAS int", "atomicCAS unsigned", "atomicCAS ull", "atomicCAS ushort",
    "atomicCAS ushort at 2 mod 4"};
  static const char* const spaces[] = {"global", "local", "peer"};
  static const char* const scoped[A_NSCOPE_OPS] = {"Add", "Sub", "Exch", "Min", "Max", "Inc", "Dec", "And",
                                                   "Or", "Xor", "CAS"};
  auto check_target = [&](Check& ck, const std::string& what, uint32_t tg, uint32_t op, uint32_t blk_lo,
                          uint32_t blk_hi, uint64_t final) {
    std::vector<std::pair<uint64_t, uint64_t>> hits;
    for (uint32_t b = blk_lo; b < blk_hi; ++b)
      for (uint32_t t = 0; t < bt; ++t)
        hits.push_back({a_value(op, tg, b, t), o[(uint64_t)(b * bt + t) * A_TARGETS + tg]});
    const uint64_t init = a_init(op, tg);
    ++ck.n;
    if (!serializable(op, init, a_slot_init(op, tg), hits, final)) {
      if (ck.bad++ < 8 || verbose)
        std::printf("  *** %s: final %#llx and the %zu returned values have no serialization\n", what.c_str(),
                    (unsigned long long)final, (size_t)(blk_hi - blk_lo) * bt);
    }
  };
  for (uint32_t sp = 0; sp < 3; ++sp) {
    Check ck(std::string("ops-") + spaces[sp]);
    for (uint32_t op = 0; op < A_NUM_OPS; ++op) {
      const uint32_t tg = sp * A_NUM_OPS + op;
      if (sp == 0) {
        check_target(ck, std::string(names[op]) + " global", tg, op, 0, 2, g[tg]);
        continue;
      }
      for (uint32_t b = 0; b < 2; ++b) {
        // Block b's local memory: its own threads' targets, then the peer's.
        const uint64_t final = d[b * 128 + (sp == 1 ? 0 : A_NUM_OPS) + op];
        const uint32_t hitter = sp == 1 ? b : b ^ 1;
        check_target(ck, std::string(names[op]) + " " + spaces[sp] + " of block " + std::to_string(b), tg, op,
                     hitter, hitter + 1, final);
      }
    }
  }
  {
    Check ck("scoped");
    for (uint32_t kk = 0; kk < 2 * A_NSCOPE_OPS; ++kk) {
      const uint32_t tg = A_SCOPED + kk;
      check_target(ck, std::string("atomic") + scoped[kk % A_NSCOPE_OPS] + (kk < A_NSCOPE_OPS ? "_block" : "_system"),
                   tg, a_op(tg), 0, 2, g[tg]);
    }
  }
  vx_kernel_release(k);
}

// ── barrier ──────────────────────────────────────────────────────────
void mode_barrier() {
  vx_kernel_h k = kernel("barrier_k");
  Buf out((uint64_t)B_THREADS * B_WORDS * 8), in(32 * 1024), dump(B_LMEM);
  out.zero();
  dump.zero();
  std::vector<uint8_t> src(32 * 1024);
  for (uint32_t i = 0; i < src.size(); ++i)
    src[i] = (uint8_t)(mix32(i / 4, 0x5A) >> (8 * (i & 3)));
  in.write(src.data(), src.size());
  cuda_arg_t arg = new_arg();
  arg.out = out.addr;
  arg.in = in.addr;
  arg.dump = dump.addr;
  arg.stride = B_WORDS;
  arg.lmem = B_LMEM;
  arg.sub = parts;
  CHECK(run(k, config(vc::dim3(1), vc::dim3(B_THREADS), vc::dim3(0, 0, 0), B_LMEM), arg));
  const std::vector<uint64_t> o = out.read<uint64_t>();
  const std::vector<uint8_t> lm = dump.read<uint8_t>();
  {
    Check ck("phases");
    for (uint32_t t = 0; t < B_THREADS; ++t) {
      const bool dropped = t / WARP == B_WARPS - 1;
      for (uint32_t p = 0; p < B_PHASES && (parts & 1); ++p) {
        const uint32_t live = p <= B_DROP ? B_THREADS : (B_WARPS - 1) * WARP;
        const uint64_t want = dropped && p >= B_DROP ? ~0ull : b_data(p, (t + 1) % live);
        ck.eq(o[t * B_WORDS + p], want, p > B_DROP ? "after arrive_and_drop" : "arrive/wait", t, p);
      }
      ck.eq(o[t * B_WORDS + B_PHASES], 1, "barrier_native_handle", t);
      ck.eq(o[t * B_WORDS + B_PHASES + 1], 0xC0DE, "done", t);
    }
  }
  auto cmp = [&](Check& ck, uint32_t dst, uint32_t s, uint32_t n, const char* what) {
    for (uint32_t i = 0; i < n; ++i)
      ck.eq(lm[dst + i], src[s + i], what, i);
  };
  if (parts & 2) {
    Check ck("memcpy_async_tx");
    for (uint32_t i = 0; i < B_NTX; ++i)
      cmp(ck, B_LM_TX + i * 4096, i * 4096, b_tx_size(i),
          i == B_NTX - 1 ? "barrier_expect_tx + memcpy_async_tx" : "memcpy_async_tx + barrier_arrive_tx");
  }
  if (parts & 4) {
    Check ck("memcpy_async");
    cmp(ck, B_LM_MC, 4 * 4096, B_MC_ALIGNED, "aligned");
    cmp(ck, B_MC_DST_UN, B_MC_SRC_UN, B_MC_UNALIGNED, "unaligned");
    cmp(ck, B_MC_DST_SZ, B_MC_SRC_SZ, B_MC_SIZED, "aligned_size_t<16>");
  }
  if (parts & 8) {
    Check ck("memcpy_async group, per-lane bulk, byte copy");
    std::vector<uint8_t> want(4096, 0);
    auto put = [&](uint32_t d, uint32_t s0, uint32_t n) {
      for (uint32_t i = 0; i < n; ++i)
        want[d + i] = src[s0 + i];
    };
    put(B_G_AL, B_G_AL_SRC, B_G_AL_N);
    put(B_G_UN, B_G_UN_SRC, B_G_UN_N);
    for (uint32_t l = 0; l < 4; ++l)
      put(B_G_PL + B_G_PL_N * l, B_G_PL_SRC + B_G_PL_N * l, B_G_PL_N);
    put(B_G_ODD, B_G_ODD_SRC, B_G_ODD_N);
    for (uint32_t i = 0; i < 4096; ++i)
      ck.eq(lm[B_LM_G + i], want[i], i < 1024 ? "group aligned" : i < 2048 ? "group unaligned"
                                    : i < 3072 ? "per-lane bulk (4 lanes)" : "odd thread-form copy", i);
  }
  vx_kernel_release(k);
}

// ── sync ─────────────────────────────────────────────────────────────
void mode_sync() {
  vx_kernel_h k = kernel("sync_k");
  const uint32_t lmem = S_LMEM;
  Buf out((uint64_t)S_THREADS * S_WORDS * 8), aux(S_SWROUNDS * S_THREADS * 4), dump(64);
  out.zero();
  aux.zero();
  cuda_arg_t arg = new_arg();
  arg.out = out.addr;
  arg.aux = aux.addr;
  arg.dump = dump.addr;
  arg.stride = S_WORDS;
  arg.lmem = lmem;
  CHECK(run(k, config(vc::dim3(1), vc::dim3(S_THREADS), vc::dim3(0, 0, 0), lmem), arg));
  const std::vector<uint64_t> o = out.read<uint64_t>();
  {
    Check ck("__syncthreads_count/and/or");
    static const char* const pn[] = {"all true", "all false", "one thread", "half"};
    for (uint32_t t = 0; t < S_THREADS; ++t) {
      uint32_t kk = 0;
      for (uint32_t r = 0; r < S_ROUNDS; ++r)
      for (uint32_t p = 0; p < S_NPRED; ++p) {
        uint32_t count = 0;
        for (uint32_t u = 0; u < S_THREADS; ++u)
          count += s_pred(p, r, u);
        ck.eq(o[t * S_WORDS + kk++], count, pn[p], t, r);
        ck.eq(o[t * S_WORDS + kk++], count == S_THREADS, pn[p], t, r);
        ck.eq(o[t * S_WORDS + kk++], count != 0, pn[p], t, r);
      }
    }
  }
  {
    Check ck("__syncwarp");
    for (uint32_t t = 0; t < S_THREADS; ++t)
      for (uint32_t r = 0; r < S_SWROUNDS; ++r) {
        const uint32_t base = S_ROUNDS * S_NPRED * 3 + 2 * r;
        ck.eq(o[t * S_WORDS + base], s_sw_value(r, t ^ 1, 0), "global", t, r);
        ck.eq(o[t * S_WORDS + base + 1], s_sw_value(r, t ^ 1, 1), "local", t, r);
      }
  }
  vx_kernel_release(k);

  vx_kernel_h k8 = kernel("sync8_k");
  Buf out8((uint64_t)S_THREADS * S8_WORDS * 8);
  out8.zero();
  arg.out = out8.addr;
  arg.stride = S8_WORDS;
  arg.lmem = S8_LMEM;
  CHECK(run(k8, config(vc::dim3(1), vc::dim3(S_THREADS), vc::dim3(0, 0, 0), S8_LMEM), arg));
  const std::vector<uint64_t> o8 = out8.read<uint64_t>();
  {
    Check ck("8 bytes of own data, a cuda::barrier at offset 0");
    for (uint32_t t = 0; t < S_THREADS; ++t) {
      uint32_t kk = 0;
      for (uint32_t r = 0; r < S_ROUNDS; ++r)
      for (uint32_t p = 0; p < S_NPRED; ++p) {
        uint32_t count = 0;
        for (uint32_t u = 0; u < S_THREADS; ++u)
          count += s_pred(p, r, u);
        ck.eq(o8[t * S8_WORDS + kk++], count, "count", t, r);
        ck.eq(o8[t * S8_WORDS + kk++], count == S_THREADS, "and", t, r);
        ck.eq(o8[t * S8_WORDS + kk++], count != 0, "or", t, r);
      }
      // After two whole phases: pending (bits 19:0) and expected (39:20) are the block, tx 0.
      const uint64_t w = o8[t * S8_WORDS + kk];
      ck.eq(w & 0xFFFFF, S_THREADS, "barrier pending", t);
      ck.eq((w >> 20) & 0xFFFFF, S_THREADS, "barrier expected", t);
      ck.eq((w >> 40) & 0x1FFFFF, 0, "barrier tx", t);
      ck.eq(o8[t * S8_WORDS + kk + 1], 0xC0DE, "done", t);
    }
  }
  vx_kernel_release(k8);
}

// ── types ────────────────────────────────────────────────────────────
void mode_types() {
  vx_kernel_h k = kernel("types_k");
  Buf aux((uint64_t)T_INPUTS * T_OPS * 4), out(64), dump(64);
  aux.zero();
  cuda_arg_t arg = new_arg();
  arg.aux = aux.addr;
  arg.out = out.addr;
  arg.dump = dump.addr;
  CHECK(run(k, config(vc::dim3(2), vc::dim3(T_WARPS * WARP), vc::dim3(0, 0, 0), 0), arg));
  const std::vector<uint32_t> o = aux.read<uint32_t>();
  Check ck("device-equals-host");
  uint32_t want[T_OPS];
  for (uint32_t i = 0; i < T_INPUTS; ++i) {
    types_ops::eval_input(i, want);
    for (uint32_t j = 0; j < T_OPS; ++j)
      ck.eq(o[(uint64_t)i * T_OPS + j], want[j], "op", i, j);
  }
}

// ── launch ───────────────────────────────────────────────────────────
void check_launch(Check& ck, const Buf& aux, uint32_t blocks, uint32_t bt, uint32_t dyn, const char* what) {
  std::vector<uint32_t> w = aux.read<uint32_t>();
  for (uint32_t b = 0; b < blocks; ++b) {
    const uint32_t* x = &w[b * L_WORDS];
    ck.eq(x[0], 2, what, b, 0);
    ck.eq(x[1], 1, what, b, 1);
    ck.eq(x[2], 1, what, b, 2);
    ck.eq(x[3], b % 2, what, b, 3);
    ck.eq(x[4], 1, what, b, 4);
    ck.eq(x[5], dyn, what, b, 5);
    ck.eq(x[6], bt, what, b, 6);
    ck.eq(x[7], 1, what, b, 7);
  }
}

void mode_launch() {
  vx_kernel_h k = kernel("launch_k");
  const uint32_t bt = 64, blocks = 4;
  Buf aux(blocks * L_WORDS * 4), out(64), dump(64);
  cuda_arg_t arg = new_arg();
  arg.aux = aux.addr;
  arg.out = out.addr;
  arg.dump = dump.addr;
  {
    Check ck("launch_kernel_ex<__cluster_dims__>");
    aux.zero();
    vc::launch_config_t cfg = config(vc::dim3(blocks), vc::dim3(bt), vc::dim3(0, 0, 0), 1024);
    ck.eq(run<launch_k_cluster_dims>(k, cfg, arg), VX_SUCCESS, "unset cluster", 0);
    check_launch(ck, aux, blocks, bt, 1024, "unset cluster");
    aux.zero();
    cfg.cluster = vc::dim3(1, 1, 1);
    ck.eq(run<launch_k_cluster_dims>(k, cfg, arg), VX_SUCCESS, "cluster (1,1,1)", 0);
    check_launch(ck, aux, blocks, bt, 1024, "cluster (1,1,1)");
    aux.zero();
    cfg.cluster = vc::dim3(2, 1, 1);
    ck.eq(run<launch_k_cluster_dims>(k, cfg, arg), VX_SUCCESS, "the same explicit cluster", 0);
    check_launch(ck, aux, blocks, bt, 1024, "the same explicit cluster");
    cfg.grid = vc::dim3(3);
    ck.eq(run<launch_k_cluster_dims>(k, cfg, arg), VX_ERR_INVALID_VALUE, "grid not a multiple of the cluster", 0);
  }
  {
    Check ck("func_set_attribute");
    const int over_default = (int)vc::kDefaultMaxDynamicSharedMemorySize + 16;
    vc::launch_config_t cfg = config(vc::dim3(blocks), vc::dim3(bt), vc::dim3(0, 0, 0), (uint32_t)over_default);
    ck.eq(vc::func_max_dynamic_shared_memory_size(k), vc::kDefaultMaxDynamicSharedMemorySize, "default limit", 0);
    ck.eq(run<launch_k_cluster_dims>(k, cfg, arg), VX_ERR_INVALID_VALUE, "past the default 48 KB", 0);
    ck.eq(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)lmem_cap + 1),
          VX_ERR_INVALID_VALUE, "past the local memory", 0);
    ck.eq(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, -1),
          VX_ERR_INVALID_VALUE, "negative", 0);
    ck.eq(vc::func_set_attribute(nullptr, k, vc::func_attribute::max_dynamic_shared_memory_size, 1024),
          VX_ERR_INVALID_VALUE, "no device", 0);
    ck.eq(vc::func_set_attribute(dev, nullptr, vc::func_attribute::max_dynamic_shared_memory_size, 1024),
          VX_ERR_INVALID_HANDLE, "no kernel", 0);
    ck.eq(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)lmem_cap),
          VX_SUCCESS, "the local memory", 0);
    ck.eq(vc::func_max_dynamic_shared_memory_size(k), lmem_cap, "recorded limit", 0);
    aux.zero();
    cfg.dyn_smem = (uint32_t)lmem_cap;
    ck.eq(run<launch_k_cluster_dims>(k, cfg, arg), VX_SUCCESS, "at the limit", 0);
    check_launch(ck, aux, blocks, bt, (uint32_t)lmem_cap, "at the limit");
    ck.eq(vc::func_set_attribute(dev, k, vc::func_attribute::non_portable_cluster_size_allowed, 2),
          VX_ERR_INVALID_VALUE, "non_portable 2", 0);
    ck.eq(vc::func_set_attribute(dev, k, vc::func_attribute::non_portable_cluster_size_allowed, 1), VX_SUCCESS,
          "non_portable 1", 0);
  }
  {
    Check ck("occupancy");
    const vc::launch_config_t cfg = config(vc::dim3(blocks), vc::dim3(bt), vc::dim3(0, 0, 0), 1024);
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = k;
    li.ndim = 1;
    li.grid_dim[0] = blocks;
    li.grid_dim[1] = li.grid_dim[2] = 1;
    li.block_dim[0] = bt;
    li.block_dim[1] = li.block_dim[2] = 1;
    li.lmem_size = 1024;
    li.cluster_dim[0] = 2;
    li.cluster_dim[1] = li.cluster_dim[2] = 1;
    li.cluster_policy = VX_CLUSTER_POLICY_DEFAULT;
    li.cluster_flags = VX_CLUSTER_FLAG_NON_PORTABLE;
    uint32_t want_active = 0, want_size = 0;
    CHECK(vx_occupancy_max_active_clusters(dev, &li, &want_active));
    CHECK(vx_occupancy_max_potential_cluster_size(dev, &li, &want_size));
    int got = -1;
    ck.eq(vc::occupancy_max_active_clusters<launch_k_cluster_dims>(dev, k, cfg, &got), VX_SUCCESS, "rc", 0);
    ck.eq((uint64_t)got, want_active, "max_active_clusters<__cluster_dims__>", 0);
    vc::launch_config_t c2 = cfg;
    c2.cluster = vc::dim3(2, 1, 1);
    got = -1;
    ck.eq(vc::occupancy_max_active_clusters(dev, k, c2, &got), VX_SUCCESS, "rc", 1);
    ck.eq((uint64_t)got, want_active, "max_active_clusters", 1);
    got = -1;
    ck.eq(vc::occupancy_max_potential_cluster_size(dev, k, cfg, &got), VX_SUCCESS, "rc", 2);
    ck.eq((uint64_t)got, want_size, "max_potential_cluster_size", 2);
    ck.eq(want_active >= 1 && want_size >= 2, 1, "plausible", 0);
    got = -1;
    ck.eq(vc::occupancy_max_potential_cluster_size<launch_k_cluster_dims>(dev, k, cfg, &got), VX_SUCCESS, "rc", 3);
    ck.eq((uint64_t)got, 2, "max_potential_cluster_size<__cluster_dims__> = the required size", 3);
    got = -1;
    ck.eq(vc::occupancy_max_active_clusters(dev, k, cfg, &got), VX_ERR_INVALID_VALUE,
          "max_active_clusters without a cluster", 4);
    ck.eq((uint64_t)got, (uint64_t)-1, "untouched on the error", 4);
    std::printf("cuda_api: occupancy: %u active clusters of 2, potential cluster size %u\n", want_active,
                want_size);
  }
  {
    Check ck("tensor_map");
    alignas(64) vc::CUtensorMap a, b;
    const uint64_t gaddr = 0x40000000ull;
    const uint64_t gdim[3] = {64, 32, 4}, gstr[2] = {256, 256 * 32};
    const uint32_t box[3] = {16, 8, 2}, es[3] = {1, 1, 1};
    std::memset(&a, 0xA5, sizeof(a));
    std::memset(&b, 0xA5, sizeof(b));
    const vx_result_t r1 = vc::tensor_map_encode_tiled(
        &a, vc::CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, gaddr, gdim, gstr, box, es, vc::CU_TENSOR_MAP_INTERLEAVE_NONE,
        vc::CU_TENSOR_MAP_SWIZZLE_64B, vc::CU_TENSOR_MAP_L2_PROMOTION_L2_128B, vc::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    const int r2 = vortex::dxa::encode_tiled(&b, vortex::dxa::TensorMapDataType::UInt32, 3, gaddr, gdim, gstr, box,
                                             es, vortex::dxa::TensorMapInterleave::None,
                                             vortex::dxa::TensorMapSwizzle::B64,
                                             vortex::dxa::TensorMapL2Promotion::L2_128B,
                                             vortex::dxa::TensorMapFloatOOBfill::None);
    ck.eq(r1, VX_SUCCESS, "encode_tiled rc", 0);
    ck.eq(r1, (uint64_t)r2, "encode_tiled rc", 1);
    for (uint32_t w = 0; w < TMAP_WORDS; ++w)
      ck.eq(a.w[w], b.w[w], "encode_tiled word", w);
    ck.eq(vc::tensor_map_encode_tiled(&a, vc::CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, gaddr + 4, gdim, gstr, box, es,
                                      vc::CU_TENSOR_MAP_INTERLEAVE_NONE, vc::CU_TENSOR_MAP_SWIZZLE_NONE,
                                      vc::CU_TENSOR_MAP_L2_PROMOTION_NONE, vc::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          VX_ERR_INVALID_VALUE, "encode_tiled misaligned address", 0);
    ck.eq(vc::tensor_map_encode_tiled(&a, vc::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3, gaddr, gdim, gstr, box, es,
                                      vc::CU_TENSOR_MAP_INTERLEAVE_NONE, vc::CU_TENSOR_MAP_SWIZZLE_NONE,
                                      vc::CU_TENSOR_MAP_L2_PROMOTION_NONE, vc::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
          VX_ERR_NOT_SUPPORTED, "encode_tiled CC 10.0 type", 0);
    const uint64_t idim[4] = {16, 9, 9, 2}, istr[3] = {32, 32 * 9, 32 * 81};
    const int lo[2] = {-1, -1}, hi[2] = {-1, -1};
    const uint32_t ies[4] = {1, 1, 1, 1};
    const vx_result_t r3 = vc::tensor_map_encode_im2col(
        &a, vc::CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, gaddr, idim, istr, lo, hi, 16, 64, ies,
        vc::CU_TENSOR_MAP_INTERLEAVE_NONE, vc::CU_TENSOR_MAP_SWIZZLE_32B, vc::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        vc::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    const int r4 = vortex::dxa::encode_im2col(&b, vortex::dxa::TensorMapDataType::Float16, 4, gaddr, idim, istr, lo,
                                              hi, 16, 64, ies, vortex::dxa::TensorMapInterleave::None,
                                              vortex::dxa::TensorMapSwizzle::B32,
                                              vortex::dxa::TensorMapL2Promotion::None,
                                              vortex::dxa::TensorMapFloatOOBfill::NanRequestZeroFma);
    ck.eq(r3, VX_SUCCESS, "encode_im2col rc", 0);
    ck.eq(r3, (uint64_t)r4, "encode_im2col rc", 1);
    for (uint32_t w = 0; w < TMAP_WORDS; ++w)
      ck.eq(a.w[w], b.w[w], "encode_im2col word", w);
    ck.eq(vc::tensor_map_replace_address(&a, gaddr + 0x1000), VX_SUCCESS, "replace_address rc", 0);
    ck.eq(vortex::dxa::replace_address(&b, gaddr + 0x1000), VX_SUCCESS, "replace_address rc", 1);
    for (uint32_t w = 0; w < TMAP_WORDS; ++w)
      ck.eq(a.w[w], b.w[w], "replace_address word", w);
    ck.eq(vc::tensor_map_replace_address(&a, gaddr + 8), VX_ERR_INVALID_VALUE, "replace_address misaligned", 0);
  }
  {
    Check ck("templated kernel, two instantiations by name");
    for (uint32_t kk = 2; kk <= 3; ++kk) {
      vx_kernel_h kt = kernel(kk == 2 ? "scale2_k" : "scale3_k");
      Buf o2(2 * 64 * 8);
      o2.zero();
      cuda_arg_t a2 = new_arg();
      a2.out = o2.addr;
      CHECK(run(kt, config(vc::dim3(2), vc::dim3(64), vc::dim3(0, 0, 0), 0), a2));
      const std::vector<uint64_t> w = o2.read<uint64_t>();
      for (uint32_t t = 0; t < 128; ++t)
        ck.eq(w[t], kk * 1000 + t, kk == 2 ? "scale_k<2>" : "scale_k<3>", t);
      vx_kernel_release(kt);
    }
  }
  {
    Check ck("other CUDA names");
    vx_kernel_h kn = kernel("names_k");
    Buf on((uint64_t)N_THREADS * N_WORDS * 8), an(64), in(1024);
    on.zero();
    an.zero();
    std::vector<uint32_t> src(256);
    for (uint32_t i = 0; i < 256; ++i)
      src[i] = mix32(i, 0x9A);
    in.write(src.data(), 1024);
    cuda_arg_t a2 = new_arg();
    a2.out = on.addr;
    a2.aux = an.addr;
    a2.in = in.addr;
    a2.stride = N_WORDS;
    a2.lmem = N_LMEM;
    CHECK(run(kn, config(vc::dim3(1), vc::dim3(N_THREADS), vc::dim3(0, 0, 0), N_LMEM), a2));
    const std::vector<uint64_t> w = on.read<uint64_t>();
    const std::vector<float> f4 = an.read<float>();
    static const char* const nn[] = {"__popc", "__popcll", "__ffs", "__clz", "__brev", "max int", "min unsigned",
                                     "max long long", "min float", "max double", "uint4", "__cvta_generic_to_shared",
                                     "__cvta_generic_to_global", "clock64 / fences / __nanosleep",
                                     "__pipeline_memcpy_async", "__align__ on the stack", "done"};
    for (uint32_t t = 0; t < N_THREADS; ++t) {
      const uint32_t x = n_word(t), xs = x >> (t % 32);
      uint32_t brev = 0;
      for (uint32_t b = 0; b < 32; ++b)
        brev |= ((x >> b) & 1u) << (31 - b);
      const float mf = std::min(1.5f, (float)t);
      const double md = std::max(2.5, (double)t);
      uint32_t mfb; std::memcpy(&mfb, &mf, 4);
      uint64_t mdb; std::memcpy(&mdb, &md, 8);
      const uint64_t want[] = {
        (uint64_t)__builtin_popcount(x), (uint64_t)(2 * __builtin_popcount(x)), (uint64_t)__builtin_ffs((int)x),
        (uint64_t)(xs ? __builtin_clz(xs) : 32), brev, (uint64_t)(int64_t)std::max((int)x, -5),
        std::min(x, 1000u), (uint64_t)std::max((long long)t, 7ll), mfb, mdb, (uint32_t)(4 * x + 6), 64, 1, 1,
        src[(t % 64)], 1, 0xC0DE};
      for (uint32_t i = 0; i < sizeof(want) / sizeof(want[0]); ++i)
        ck.eq(w[t * N_WORDS + i], want[i], nn[i], t, i);
    }
    float sw = 0;
    for (uint32_t t = 0; t < N_THREADS; ++t)
      sw += (float)(t % 4);
    const float wf[4] = {(float)N_THREADS, 2.0f * N_THREADS, 0.5f * N_THREADS, sw};
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t gb, wb;
      std::memcpy(&gb, &f4[i], 4);
      std::memcpy(&wb, &wf[i], 4);
      ck.eq(gb, wb, "atomicAdd(float4*)", i);
    }
    vx_kernel_release(kn);
  }
  {
    // The attributes belong to the kernel, not to a handle the runtime may hand out again.
    Check ck("attributes after release");
    const void* old_handle = k;
    vx_kernel_release(k);
    k = kernel("cg_k");
    ck.eq(vc::func_max_dynamic_shared_memory_size(k), vc::kDefaultMaxDynamicSharedMemorySize,
          "another kernel starts from the default", 0);
    std::printf("cuda_api: the next kernel %s the released handle\n", (const void*)k == old_handle ? "reuses" : "does not reuse");
  }
  vx_kernel_release(k);
}

// ── ptx ──────────────────────────────────────────────────────────────
void mode_ptx() {
  const uint32_t threads = 2 * X_THREADS;
  Buf out((uint64_t)threads * X_WORDS * 8), in(XG_IN), aux(XA_SIZE), dump(2 * X_LMEM);
  std::vector<uint32_t> gin(XG_IN / 4), gaux(XA_SIZE / 4);
  for (uint32_t w = 0; w < gin.size(); ++w)
    gin[w] = x_in(w);
  for (uint32_t w = 0; w < gaux.size(); ++w)
    gaux[w] = x_aux(w);
  in.write(gin.data(), XG_IN);
  aux.write(gaux.data(), XA_SIZE);
  if ((aux.addr & 127) || (in.addr & 15)) {
    std::printf("FAIL ptx: buffers not aligned (%#llx, %#llx)\n", (unsigned long long)aux.addr,
                (unsigned long long)in.addr);
    ++failures;
    return;
  }
  cuda_arg_t arg = new_arg();
  const uint64_t gdim[2] = {X_TW, X_TH}, gstr[1] = {X_TW * 4}, adim[2] = {X_TW, 8};
  const uint32_t box[2] = {X_BX, X_BY}, es[2] = {1, 1};
  CHECK(vc::tensor_map_encode_tiled(&arg.map[0], vc::CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, in.addr + XG_T0, gdim, gstr,
                                    box, es, vc::CU_TENSOR_MAP_INTERLEAVE_NONE, vc::CU_TENSOR_MAP_SWIZZLE_NONE,
                                    vc::CU_TENSOR_MAP_L2_PROMOTION_NONE, vc::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CHECK(vc::tensor_map_encode_tiled(&arg.map[1], vc::CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, aux.addr + XA_TENS, adim,
                                    gstr, box, es, vc::CU_TENSOR_MAP_INTERLEAVE_NONE, vc::CU_TENSOR_MAP_SWIZZLE_NONE,
                                    vc::CU_TENSOR_MAP_L2_PROMOTION_NONE, vc::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  arg.out = out.addr;
  arg.in = in.addr;
  arg.aux = aux.addr;
  arg.dump = dump.addr;
  arg.stride = X_WORDS;
  arg.lmem = X_LMEM;
  const vc::launch_config_t cfg = config(vc::dim3(2), vc::dim3(X_THREADS), vc::dim3(0, 0, 0), X_LMEM);
  std::vector<uint64_t> o;
  std::vector<uint32_t> ga, lm32;
  auto launch = [&](const char* name, auto dims) {
    vx_kernel_h k = kernel(name);
    out.zero();
    dump.zero();
    CHECK(run<decltype(dims)>(k, cfg, arg));
    o = out.read<uint64_t>();
    ga = aux.read<uint32_t>();
    lm32 = dump.read<uint32_t>();
    vx_kernel_release(k);
    Check ck(std::string("done-") + name);
    for (uint32_t t = 0; t < threads; ++t)
      ck.eq(o[t * X_WORDS + 6], 0xC0DE, "done", t);
  };
  auto lw = [&](uint32_t b, uint32_t off) { return lm32[(b * X_LMEM + off) / 4]; };
  auto lq = [&](uint32_t b, uint32_t off) { return (uint64_t)lw(b, off) | ((uint64_t)lw(b, off + 4) << 32); };
  auto tile_of = [&](uint32_t base, int32_t cx, int32_t cy, uint32_t i) {
    return gin[(base + (cy + i / X_BX) * X_TW * 4 + (cx + i % X_BX) * 4) / 4];
  };
  launch("ptx_mbar_k", ptx_mbar_k_cluster_dims());
  {
    Check ck("mbarrier");
    static const char* const fn[] = {"arrive/try_wait/test_wait", "release/acquire arrive and waits",
                                     "relaxed arrive with count, parity waits", "cluster-scope waits, hints",
                                     "arrive with count 32, parity hint", "getctarank", "done"};
    for (uint32_t t = 0; t < threads; ++t)
      for (uint32_t i = 0; i < 6; ++i)
        ck.eq(o[t * X_WORDS + i], 1, fn[i], t, i);
  }
  launch("ptx_copy_k", ptx_copy_k_cluster_dims());
  {
    Check ck("loads");
    for (uint32_t b = 0; b < 2; ++b) {
      for (uint32_t w = 0; w < 128; ++w)
        ck.eq(lw(b, XL_G2S + 4 * w), gin[(XG_G2S + b * 512) / 4 + w], "cp_async_bulk global->shared", b, w);
      for (uint32_t i = 0; i < X_BX * X_BY; ++i)
        ck.eq(lw(b, XL_TENS + 4 * i), tile_of(XG_T0, 8 + 16 * b, 4 + b, i), "cp_async_bulk_tensor 2d", b, i);
      for (uint32_t w = 0; w < 64; ++w)
        ck.eq(lw(b, XL_MC + 4 * w), gin[XG_MC / 4 + w], "cp_async_bulk multicast", b, w);
      for (uint32_t i = 0; i < X_BX * X_BY; ++i)
        ck.eq(lw(b, XL_TMC + 4 * i), tile_of(XG_T0, 0, 8, i), "cp_async_bulk_tensor multicast", b, i);
    }
  }
  {
    Check ck("cluster copies");
    for (uint32_t b = 0; b < 2; ++b) {
      const uint32_t w = b ^ 1;
      for (uint32_t i = 0; i < 64; ++i)
        ck.eq(lw(b, XL_S2C + 4 * i), x_src(w, i), "cp_async_bulk shared::cta->shared::cluster", b, i);
      for (uint32_t i = 0; i < 64; ++i)
        ck.eq(lw(b, XL_RC + 4 * i), (uint32_t)(x_rc_init(i) + x_src(w, i)), "cp_reduce_async_bulk shared::cluster",
              b, i);
    }
  }
  launch("ptx_async_k", ptx_async_k_cluster_dims());
  {
    Check ck("st_async/red_async");
    for (uint32_t b = 0; b < 2; ++b) {
      const uint32_t w = b ^ 1;
      for (uint32_t l = 0; l < WARP; ++l)
        ck.eq(lw(b, XL_ST + 4 * l), mix32(w * 100 + l, 0x51), "st_async b32", b, l);
      for (uint32_t l = 0; l < 8; ++l)
        ck.eq(lq(b, XL_ST + 128 + 8 * l), mix64(w * 100 + l, 0x52), "st_async b64", b, l);
      for (uint32_t l = 0; l < 4; ++l)
        for (uint32_t e = 0; e < 2; ++e)
          ck.eq(lw(b, XL_ST + 192 + 8 * l + 4 * e), mix32(w * 100 + l, 0x53 + e), "st_async v2.b32", b, l);
      for (uint32_t l = 0; l < 2; ++l)
        for (uint32_t e = 0; e < 4; ++e)
          ck.eq(lw(b, XL_ST + 224 + 16 * l + 4 * e), mix32(w * 100 + l, 0x55 + e), "st_async v4.b32", b, l);
      static const char* const rn[] = {"red_async add u32", "min u32", "max u32", "inc", "dec", "min s32",
                                       "max s32", "add s32", "and", "or", "xor", "add u64", "add s64"};
      for (uint32_t kk = 0; kk < 13; ++kk) {
        uint64_t s = x_red_init(kk);
        for (uint32_t l = 0; l < WARP; ++l) {
          const uint64_t v = x_red_value(kk, w, l);
          const uint32_t s32 = (uint32_t)s, v32 = (uint32_t)v;
          switch (kk) {
          case 0: case 7: s = (uint32_t)(s32 + v32); break;
          case 1: s = std::min(s32, v32); break;
          case 2: s = std::max(s32, v32); break;
          case 3: s = s32 >= v32 ? 0 : s32 + 1; break;
          case 4: s = (s32 == 0 || s32 > v32) ? v32 : s32 - 1; break;
          case 5: s = (uint32_t)std::min((int32_t)s32, (int32_t)v32); break;
          case 6: s = (uint32_t)std::max((int32_t)s32, (int32_t)v32); break;
          case 8: s = s32 & v32; break;
          case 9: s = s32 | v32; break;
          case 10: s = s32 ^ v32; break;
          default: s += v; break;
          }
        }
        const uint64_t got = kk < 11 ? lw(b, XL_RED + 4 * kk) : lq(b, XL_RED + 48 + 8 * (kk - 11));
        ck.eq(got, kk < 11 ? (uint32_t)s : s, rn[kk], b, kk);
      }
    }
  }
  launch("ptx_global_k", ptx_global_k_cluster_dims());
  {
    Check ck("global");
    std::vector<uint32_t> want = gaux;
    for (uint32_t b = 0; b < 2; ++b) {
      for (uint32_t w = 0; w < 128; ++w)
        want[(XA_STORE + b * 512) / 4 + w] = x_src(b, w);
      for (uint32_t w = 0; w < 64; ++w)
        want[XA_RED / 4 + w] += x_src(b, w);
      for (uint32_t i = 0; i < X_BX * X_BY; ++i) {
        const uint32_t x = 16 * b + i % X_BX, y = i / X_BX;
        want[XA_TENS / 4 + y * X_TW + x] = x_src(b, i);
        const uint32_t xr = 32 + 16 * b + i % X_BX, yr = 4 + i / X_BX;
        want[XA_TENS / 4 + yr * X_TW + xr] += x_src(b, 64 + i);
      }
      uint32_t sum = 0;
      for (uint32_t w = 0; w < 128; ++w)
        sum = sum * 31 + x_src(b, w);
      want[XA_FENCE / 4 + b] = sum;
    }
    for (uint32_t w = 0; w < XA_MAP / 4; ++w)
      ck.eq(ga[w], want[w], w < XA_RED / 4 ? "cp_async_bulk shared->global" : w < XA_TENS / 4
                            ? "cp_reduce_async_bulk global add" : "cp_async_bulk_tensor store / reduce", w);
    for (uint32_t b = 0; b < 2; ++b)
      ck.eq(ga[XA_FENCE / 4 + b], want[XA_FENCE / 4 + b], "fence_proxy_async global read back", b);
  }
  {
    Check ck("tensormap");
    alignas(64) vc::CUtensorMap e2, e3;
    CHECK(vc::tensor_map_encode_tiled(&e2, vc::CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, in.addr + XG_T2, gdim, gstr, box,
                                      es, vc::CU_TENSOR_MAP_INTERLEAVE_NONE, vc::CU_TENSOR_MAP_SWIZZLE_NONE,
                                      vc::CU_TENSOR_MAP_L2_PROMOTION_NONE, vc::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    e3 = e2;
    CHECK(vc::tensor_map_replace_address(&e3, in.addr + XG_T3));
    for (uint32_t w = 0; w < TMAP_WORDS; ++w) {
      ck.eq(lw(0, XL_MAP + 4 * w), e2.w[w], "tensormap_replace_* in shared memory = the encoder", w);
      ck.eq(ga[XA_MAP / 4 + w], e3.w[w], "tensormap_cp_fenceproxy + replace in global = the encoder", w);
    }
    for (uint32_t i = 0; i < X_BX * X_BY; ++i) {
      ck.eq(lw(0, XL_TR + 4 * i), tile_of(XG_T2, 16, 2, i), "load through the field-built map", i);
      ck.eq(lw(0, XL_TR2 + 4 * i), tile_of(XG_T3, 16, 2, i), "load after the global replace", i);
    }
  }
}

// ── negatives ────────────────────────────────────────────────────────
int run_negative() {
  vx_kernel_h k = kernel("launch_k");
  Buf aux(4 * L_WORDS * 4), out(64), dump(64);
  aux.zero();
  cuda_arg_t arg = new_arg();
  arg.aux = aux.addr;
  arg.out = out.addr;
  arg.dump = dump.addr;
  vc::launch_config_t cfg = config(vc::dim3(4), vc::dim3(64), vc::dim3(1, 1, 1), 1024);
  // The compliant twin of x2 / x3 first: the same kernel and grid must launch and run.
  auto compliant = [&](vc::launch_config_t c, const char* what) {
    aux.zero();
    const vx_result_t r = run<launch_k_cluster_dims>(k, c, arg);
    const std::vector<uint32_t> w = aux.read<uint32_t>();
    if (r != VX_SUCCESS || w[7] != 1 || w[5] != c.dyn_smem) {
      std::printf("*** the compliant launch (%s) did not run: %s, ran %u, dynamic smem %u\n", what,
                  vx_result_string(r), w[7], w[5]);
      return false;
    }
    std::printf("cuda_api: the compliant launch (%s) ran\n", what);
    return true;
  };
  vx_result_t r = VX_SUCCESS;
  switch (neg) {
  case 1:
    // Flushed: the trap stops the process.
    std::printf("cuda_api: x1 marker: launch_k (__cluster_dims__ 2, 1, 1) launched with cluster (1, 1, 1)\n");
    std::fflush(stdout);
    r = run(k, cfg, arg);
    std::printf("*** the launch %s: the kernel entry check must stop it\n",
                r == VX_SUCCESS ? "completed" : vx_result_string(r));
    return 1;
  case 2:
    CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, 4096));
    cfg.cluster = vc::dim3(0, 0, 0);
    cfg.dyn_smem = 4096;
    if (!compliant(cfg, "dynamic shared memory 4096 at the limit 4096"))
      return 1;
    cfg.dyn_smem = 8192;
    r = run<launch_k_cluster_dims>(k, cfg, arg);
    if (r == VX_ERR_INVALID_VALUE) {
      std::printf("Error: launch_kernel_ex: dynamic shared memory 8192 over the kernel's limit 4096: %s\n",
                  vx_result_string(r));
      return 1;
    }
    break;
  default:
    cfg.cluster = vc::dim3(2, 1, 1);
    if (!compliant(cfg, "cluster (2, 1, 1) = __cluster_dims__"))
      return 1;
    cfg.cluster = vc::dim3(4, 1, 1);
    r = run<launch_k_cluster_dims>(k, cfg, arg);
    if (r == VX_ERR_INVALID_VALUE) {
      std::printf("Error: launch_kernel_ex: cluster (4, 1, 1) differs from __cluster_dims__ (2, 1, 1): %s\n",
                  vx_result_string(r));
      return 1;
    }
    break;
  }
  std::printf("*** launch_kernel_ex returned %s, not VX_ERR_INVALID_VALUE\n", vx_result_string(r));
  vx_kernel_release(k);
  return 1;
}

// ── perf (-p) ────────────────────────────────────────────────────────
void mode_perf() {
  vx_kernel_h k = kernel("perf_k");
  const uint32_t bt = P_WARPS * WARP, blocks = 2, threads = bt * blocks;
  Buf out((uint64_t)threads * P_WORDS * 8), aux((16 + blocks * 16) * 4), dump(64);
  out.zero();
  aux.zero();
  cuda_arg_t arg = new_arg();
  arg.out = out.addr;
  arg.aux = aux.addr;
  arg.dump = dump.addr;
  arg.stride = P_WORDS;
  arg.rounds = rounds;
  CHECK(run(k, config(vc::dim3(blocks), vc::dim3(bt), vc::dim3(0, 0, 0), 64), arg));
  const std::vector<uint64_t> o = out.read<uint64_t>();
  const std::vector<uint32_t> g = aux.read<uint32_t>();
  std::vector<uint32_t> wg(16 + blocks * 16, 0);
  Check ck("perf");
  for (uint32_t b = 0; b < blocks; ++b) {
    std::vector<uint32_t> lm(16, 0);
    for (uint32_t w = 0; w < P_WARPS; ++w) {
      uint32_t v[WARP], acc = 0, mx = 0;
      for (uint32_t l = 0; l < WARP; ++l)
        v[l] = mix32(b * bt + w * WARP + l, 1);
      for (uint32_t r = 0; r < rounds; ++r) {
        uint32_t n[WARP], m[WARP];
        const uint32_t x = 1 + r % 31;
        for (uint32_t l = 0; l < WARP; ++l)
          n[l] = v[l ^ x] + r;
        for (uint32_t l = 0; l < WARP; ++l)
          m[l] = n[l] ^ (l + 3 < WARP ? n[l + 3] : n[l]);
        uint32_t sum = 0;
        for (uint32_t l = 0; l < WARP; ++l)
          sum += m[l];
        acc += sum;
        mx = 0;
        for (uint32_t l = 0; l < WARP; ++l)
          mx = std::max(mx, m[l] ^ acc);
        for (uint32_t l = 0; l < WARP; ++l) {
          v[l] = m[l];
          wg[r % 8] += m[l] & 0xff;
          wg[8 + r % 8] = std::max(wg[8 + r % 8], m[l]);
          lm[r % 8] += m[l] & 0xff;
          lm[8 + r % 8] ^= mx ^ m[l];
        }
      }
      for (uint32_t l = 0; l < WARP; ++l) {
        const uint64_t* x = &o[(uint64_t)(b * bt + w * WARP + l) * P_WORDS];
        ck.eq(x[0], acc, "acc", b * bt + w * WARP + l);
        ck.eq(x[1], mx, "max", b * bt + w * WARP + l);
        ck.eq(x[2], v[l], "value", b * bt + w * WARP + l);
      }
    }
    for (uint32_t i = 0; i < 16; ++i)
      wg[16 + b * 16 + i] = lm[i];
  }
  for (uint32_t i = 0; i < wg.size(); ++i)
    ck.eq(g[i], wg[i], "atomics", i);
  vx_kernel_release(k);
}

} // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority = VX_QUEUE_PRIORITY_NORMAL;
  CHECK(vx_queue_create(dev, &qi, &q));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));

  const uint64_t need = VX_ISA_EXT_DLMEM | VX_ISA_STD_A | VX_ISA_EXT_DXA | VX_ISA_EXT_L2CACHE;
  const char* skip = nullptr;
  if ((isa & need) != need)
    skip = "the device needs DLMEM, the A extension, the DXA and an L2";
  else if (nt != WARP)
    skip = "the device needs 32-thread warps";
  else if (nw < 4)
    skip = "the device needs 4 warps a core";
  else if (num_cores < 2 || max_cluster < 2)
    skip = "the device needs 2 cores and a cluster of 2";
  else if (lmem_cap < B_LMEM)
    skip = "the local memory is too small";
  std::printf("cuda_api: %s, %llu cores, %llu warps x %llu threads, %llu bytes of local memory\n",
              neg ? "negative" : perf ? "perf" : kModeNames[mode], (unsigned long long)num_cores,
              (unsigned long long)nw, (unsigned long long)nt, (unsigned long long)lmem_cap);
  if (skip) {
    std::printf("Error: %s\nFAILED!\n", skip);
    return 1;
  }
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));

  if (neg)
    return run_negative();
  if (perf) {
    mode_perf();
  } else {
    switch (mode) {
    case M_CG:      mode_cg(); break;
    case M_WARP:    mode_warp(); break;
    case M_ATOMIC:  mode_atomic(); break;
    case M_BARRIER: mode_barrier(); break;
    case M_PTX:     mode_ptx(); break;
    case M_SYNC:    mode_sync(); break;
    case M_TYPES:   mode_types(); break;
    default:        mode_launch(); break;
    }
  }

  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);
  if (failures) {
    std::printf("Found %d failing checks!\nFAILED!\n", failures);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
