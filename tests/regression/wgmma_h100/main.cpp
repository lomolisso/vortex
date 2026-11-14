// wgmma_h100: the tensor unit's WGMMA as on the H100, blocks of one or two
// warpgroups (32 threads per warp, ISSUE_WIDTH 4).
//
//   -l none|128|64|32   swizzle mode of both operands (default 128)
//   -a k|mn, -b k|mn    A / B major (MN-major needs a 16-bit type; default k)
//   -t f16f32|f16f16|bf16f32|e4m3f32|e5m2f32   type pair (default f16f32)
//   -n N                16/32/64 (fp32 D), 32/64/128 (f16 D) (default 64 or 128)
//   -o                  operands above the first 64 KB of the block
//   -L B, -S B          bytes added to the LBO and SBO strides (multi-atom
//                       layouts with gaps; multiples of 16, of 128 with -g tma)
//   -g sw|tma           fill by stores of an image made from the descriptor
//                       golden, or by DXA tensor loads (one box per swizzle
//                       atom) on an mbarrier (default sw)
//   -w G                warpgroups in the block (1..6), each with its own B (default 1)
//   -A                  with -w 2 or more, every warpgroup also has its own A
//   -K                  LBO 0 in swizzled K-major descriptors (as DeepGEMM; default 16, as CuTe)
//   -m M                blocks (1..4), each with its own operands, all resident at
//                       once, so that several share a core and get distinct
//                       local-memory slots; a shape that does not fit is refused
//                       (default 1)
//   -C                  the blocks form clusters of 2 (M even)
//   -R                  ring: 2 stages filled by DXA tensor loads from a producer
//                       warp (the block's 5th warp) through full/empty mbarriers;
//                       wgmma i uses stage i % 2 with fresh data; warps 1-3 start
//                       each wgmma late (-k) and only thread 0 frees
//                       the stage after wait_group 0, so a wait that covers only
//                       the own warp lets the producer overwrite operands that
//                       warps 1-3 have not read (needs -c 3 or more)
//   -k cycles           the -R skew of warps 1-3 (default 20000)
//   -c C                wgmmas chained along k, with commit/wait groups (1..8, default 2)
//   -z                  the first wgmma has scale_d = 0
//   -r R                launches, each with fresh inputs (default 2)
//
// Every accumulator register of every thread is checked bit for bit against
// the TFR datapath model (tcu_fedp_ref: one FEDP call per k8 of 16-bit or
// k16 of 8-bit inputs, the result rounded to the D format each time), with
// each (thread, register, half) mapped to D(m, n) by the PTX fragment layout
// (tcu_desc_ref); bits 63:32 of every accumulator register must be the NaN
// box. The descriptors the kernel built must equal the golden encoding, the
// library's row/column helpers must equal the golden, outputs are prefilled
// with a NaN pattern, and guard words after every output must survive. With
// -m on one core, some block must have run at a nonzero local-memory slot;
// with -C, the two blocks of a cluster must run on distinct cores.

#include <vortex2.h>
#include <vx_cuda_runtime.h>
#include <VX_types.h>
#include <tcu_desc_ref.h>
#include <tcu_fedp_ref.h>
#include "common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace vc = vx::cuda;
namespace td = vortex::tcu_desc_ref;
namespace tf = vortex::tcu_fedp_ref;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "Error %s:%d: '%s' returned %s\n",         \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

constexpr size_t kGuardBytes = 256;
constexpr uint32_t kGuardWord = 0x9e3779b9u;
constexpr uint32_t kNaNFill = 0xffffffffu;
constexpr uint64_t kNaNFill64 = 0x7ff4a5a5a5a5a5a5ull;   // a NaN whose high word is no NaN box
constexpr uint32_t kM = 64;

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;

struct Opts {
  uint32_t layout = td::LAYOUT_128B;
  bool a_mn = false, b_mn = false;
  uint32_t pair = PAIR_F16F32;
  uint32_t n = 0;
  bool high = false;
  uint32_t pad_l = 0, pad_s = 0;
  bool tma = false;
  uint32_t groups = 1, chain = 2;
  bool zero_first = false;
  bool distinct_a = false, lbo0 = false, cluster = false, ring = false;
  uint32_t skew = 20000;
  uint32_t ctas = 1;
  int reps = 2;
} o;

const char* kPairNames[PAIR_COUNT] = {"f16f32", "f16f16", "bf16f32", "e4m3f32", "e5m2f32"};

uint32_t fmt_s(uint32_t pair) {
  switch (pair) {
  case PAIR_BF16F32: return tf::FMT_BF16;
  case PAIR_E4M3F32: return tf::FMT_FP8;
  case PAIR_E5M2F32: return tf::FMT_BF8;
  default: return tf::FMT_FP16;
  }
}
bool packed_d(uint32_t pair) { return pair == PAIR_F16F16; }
uint32_t esize(uint32_t pair) { return (pair == PAIR_E4M3F32 || pair == PAIR_E5M2F32) ? 1 : 2; }

void usage() {
  std::printf("Usage: [-l none|128|64|32] [-a k|mn] [-b k|mn] [-t f16f32|f16f16|bf16f32|e4m3f32|e5m2f32] [-n N]\n"
              "       [-o] [-L bytes] [-S bytes] [-g sw|tma] [-w 1..6] [-A] [-K] [-m blocks] [-C] [-R] [-k cycles] [-c chain] [-z]\n"
              "       [-r reps] [-h]\n");
}

bool parse_major(const char* s, bool* mn) {
  if (!std::strcmp(s, "k")) { *mn = false; return true; }
  if (!std::strcmp(s, "mn")) { *mn = true; return true; }
  return false;
}

void parse_args(int argc, char** argv) {
  int c;
  bool ok = true;
  while ((c = getopt(argc, argv, "l:a:b:t:n:oL:S:g:w:AKm:CRk:c:zr:h")) != -1) {
    switch (c) {
    case 'l':
      if (!std::strcmp(optarg, "none")) o.layout = td::LAYOUT_NONE;
      else if (!std::strcmp(optarg, "128")) o.layout = td::LAYOUT_128B;
      else if (!std::strcmp(optarg, "64")) o.layout = td::LAYOUT_64B;
      else if (!std::strcmp(optarg, "32")) o.layout = td::LAYOUT_32B;
      else ok = false;
      break;
    case 'a': ok &= parse_major(optarg, &o.a_mn); break;
    case 'b': ok &= parse_major(optarg, &o.b_mn); break;
    case 't': {
      uint32_t p = 0;
      while (p < PAIR_COUNT && std::strcmp(optarg, kPairNames[p])) ++p;
      ok &= p < PAIR_COUNT;
      o.pair = p;
      break;
    }
    case 'n': o.n = (uint32_t)std::atoi(optarg); break;
    case 'o': o.high = true; break;
    case 'L': o.pad_l = (uint32_t)std::atoi(optarg); break;
    case 'S': o.pad_s = (uint32_t)std::atoi(optarg); break;
    case 'g':
      if (!std::strcmp(optarg, "sw")) o.tma = false;
      else if (!std::strcmp(optarg, "tma")) o.tma = true;
      else ok = false;
      break;
    case 'w': o.groups = (uint32_t)std::atoi(optarg); break;
    case 'A': o.distinct_a = true; break;
    case 'K': o.lbo0 = true; break;
    case 'm': o.ctas = (uint32_t)std::atoi(optarg); break;
    case 'C': o.cluster = true; break;
    case 'R': o.ring = true; break;
    case 'k': o.skew = (uint32_t)std::atoi(optarg); break;
    case 'c': o.chain = (uint32_t)std::atoi(optarg); break;
    case 'z': o.zero_first = true; break;
    case 'r': o.reps = std::atoi(optarg); break;
    case 'h': usage(); std::exit(0);
    default: ok = false; break;
    }
  }
  const bool packed = packed_d(o.pair);
  if (o.n == 0)
    o.n = packed ? 128 : 64;
  const uint32_t nmin = packed ? 32 : 16;
  ok &= (o.n == nmin || o.n == 2 * nmin || o.n == 4 * nmin);
  ok &= !((o.a_mn || o.b_mn) && esize(o.pair) != 2);
  ok &= o.groups >= 1 && o.groups <= kMaxGroups && o.chain >= 1 && o.chain <= kMaxChain && o.reps >= 1;
  ok &= !o.distinct_a || o.groups >= 2;
  ok &= o.ctas >= 1 && o.ctas <= kMaxCtas && (!o.cluster || o.ctas % 2 == 0);
  ok &= !o.ring || (o.groups == 1 && o.chain >= 3);
  if (o.ring)
    o.tma = true;
  const uint32_t pad_unit = (o.tma && o.layout != td::LAYOUT_NONE) ? 128 : 16;
  ok &= (o.pad_l % pad_unit) == 0 && (o.pad_s % pad_unit) == 0;
  if (!ok || optind < argc) {
    usage();
    std::exit(1);
  }
}

struct Buf {
  vx_buffer_h h = nullptr;
  uint64_t addr = 0, size = 0;
  explicit Buf(uint64_t n) : size(n) {
    CHECK(vx_buffer_create(dev, size, VX_MEM_READ_WRITE, &h));
    CHECK(vx_buffer_address(h, &addr));
  }
  ~Buf() { vx_buffer_release(h); }
  void write(const void* p) {
    vx_event_h e = nullptr;
    CHECK(vx_enqueue_write(q, h, 0, p, size, 0, nullptr, &e));
    CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e);
  }
  void read(void* p) const {
    vx_event_h e = nullptr;
    CHECK(vx_enqueue_read(q, p, h, 0, size, 0, nullptr, &e));
    CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e);
  }
};

// body bytes of fill, then guard words.
std::vector<uint8_t> guarded(size_t body, uint32_t fill) {
  std::vector<uint8_t> v(body + kGuardBytes);
  for (size_t i = 0; i + 4 <= body; i += 4)
    std::memcpy(&v[i], &fill, 4);
  for (size_t i = body; i + 4 <= v.size(); i += 4)
    std::memcpy(&v[i], &kGuardWord, 4);
  return v;
}

bool guard_intact(const std::vector<uint8_t>& v, size_t body, const char* what, int rep) {
  for (size_t i = body; i + 4 <= v.size(); i += 4) {
    uint32_t w;
    std::memcpy(&w, &v[i], 4);
    if (w != kGuardWord) {
      std::printf("Error: rep %d: guard word %zu after %s changed: 0x%08x\n", rep, (i - body) / 4, what, w);
      return false;
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////
// Operand placement: swizzle atoms (8 rows of W 16-byte chunks) tiled MN
// first, then K (CuTe tile_to_shape order), with the -L/-S gaps.

struct Box { uint32_t inner, outer, dst; };

struct Plan {
  uint32_t mn = 0, k = 0, e = 0, layout = 0;
  bool mn_major = false;
  uint32_t lbo = 0, sbo = 0;
  uint32_t start[kMaxChain] = {};
  uint32_t atom_el = 0;               // box inner elements
  std::vector<Box> boxes;
  uint32_t lo = 0, hi = 0;            // byte range the operand occupies

  uint64_t desc(uint32_t i) const { return td::encode(start[i], lbo, sbo, layout); }
  uint32_t addr(uint32_t i, uint32_t m, uint32_t kk) const {
    return td::operand_addr(desc(i), mn_major, e, m, kk);
  }
};

Plan plan(uint32_t MN, uint32_t K, bool mn_major, uint32_t layout, uint32_t E, uint32_t base, uint32_t chain) {
  Plan p;
  p.mn = MN;
  p.k = K;
  p.e = E;
  p.layout = layout;
  p.mn_major = mn_major;
  const uint32_t W = td::swizzle_chunks(layout), KI = td::inst_k(E);
  const bool none = (layout == td::LAYOUT_NONE);
  if (!mn_major) {
    p.atom_el = W * 16 / E;
    p.sbo = (none ? 128 : 8 * W * 16) + o.pad_s;                 // next 8 rows
    const uint32_t kstride = (MN / 8) * p.sbo + o.pad_l;          // next k atom
    p.lbo = none ? kstride : (o.lbo0 ? 0 : 16);
    for (uint32_t i = 0; i < chain; ++i) {
      const uint32_t k0 = i * KI;
      p.start[i] = base + (k0 / p.atom_el) * kstride + (k0 % p.atom_el) * E;
    }
    for (uint32_t ka = 0; ka < (K + p.atom_el - 1) / p.atom_el; ++ka)   // a partial atom: zero fill past K
      for (uint32_t rg = 0; rg < MN / 8; ++rg)
        p.boxes.push_back({ka * p.atom_el, rg * 8, base + ka * kstride + rg * p.sbo});
  } else {
    p.atom_el = W * 16 / E;
    const uint32_t n_atoms = (MN + p.atom_el - 1) / p.atom_el;
    uint32_t kgstride;
    if (none) {
      p.sbo = 128 + o.pad_s;                                       // next 8 mn
      p.lbo = n_atoms * p.sbo + o.pad_l;                           // next 8 k
      kgstride = p.lbo;
      for (uint32_t kg = 0; kg < K / 8; ++kg)
        for (uint32_t a = 0; a < n_atoms; ++a)
          p.boxes.push_back({a * p.atom_el, kg * 8, base + kg * kgstride + a * p.sbo});
    } else {
      const uint32_t astride = 8 * W * 16 + o.pad_l;               // next mn atom
      p.sbo = n_atoms * astride + o.pad_s;                         // next 8 k
      p.lbo = (n_atoms == 1 && o.pad_l == 0) ? 0 : astride;        // unused for one atom (CuTe writes 0)
      kgstride = p.sbo;
      for (uint32_t kg = 0; kg < K / 8; ++kg)
        for (uint32_t a = 0; a < n_atoms; ++a)
          p.boxes.push_back({a * p.atom_el, kg * 8, base + kg * kgstride + a * astride});
    }
    for (uint32_t i = 0; i < chain; ++i)
      p.start[i] = base + (i * KI / 8) * kgstride;
  }
  p.lo = ~0u;
  p.hi = 0;
  for (uint32_t i = 0; i < chain; ++i) {
    for (uint32_t m = 0; m < MN; ++m) {
      for (uint32_t kk = 0; kk < KI; ++kk) {
        const uint32_t a = p.addr(i, m, kk);
        p.lo = std::min(p.lo, a);
        p.hi = std::max(p.hi, a + E);
      }
    }
  }
  for (const Box& b : p.boxes) {
    p.lo = std::min(p.lo, b.dst);
    p.hi = std::max(p.hi, b.dst + 8 * W * 16);
  }
  return p;
}

uint32_t align_up(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }

///////////////////////////////////////////////////////////////////////////////
// Inputs: raw encodings, finite, a few zeros and subnormals.

uint32_t rand_input(std::mt19937_64& g, uint32_t pair) {
  const uint32_t r = (uint32_t)g();
  const uint32_t s = r >> 31, sel = (r >> 24) & 15;
  switch (pair) {
  case PAIR_BF16F32: {
    const uint32_t e = sel == 0 ? 0 : 123 + (r >> 8) % 7;
    return (s << 15) | (e << 7) | (r & 0x7f);
  }
  case PAIR_E4M3F32: {
    uint32_t v = r & 0xff;
    if ((v & 0x7f) == 0x7f) v ^= 1;                                // no NaN
    return v;
  }
  case PAIR_E5M2F32: {
    const uint32_t e = (r >> 8) % 31;                              // no inf / NaN
    return (s << 7) | (e << 2) | (r & 3);
  }
  default: {
    const uint32_t e = sel == 0 ? 0 : 11 + (r >> 8) % 7;
    return (s << 15) | (e << 10) | (r & 0x3ff);
  }
  }
}

uint32_t rand_c(std::mt19937_64& g, bool packed) {
  const uint32_t r = (uint32_t)g();
  if (packed) {
    const uint32_t e = 12 + (r >> 8) % 5;
    return ((r >> 31) << 15) | (e << 10) | (r & 0x3ff);
  }
  const uint32_t e = 120 + (r >> 8) % 12;
  return ((r >> 31) << 31) | (e << 23) | (r & 0x7fffff);
}

} // namespace


int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  parse_args(argc, argv);
  const bool packed = packed_d(o.pair);
  const uint32_t E = esize(o.pair), KI = td::inst_k(E), K = o.chain * KI;
  const uint32_t NRC = td::acc_regs(o.n, packed), N = o.n, G = o.groups, GA = o.distinct_a ? o.groups : 1;
  const uint32_t M = o.ctas, threads = o.ring ? 160 : G * 128;

  CHECK(vx_device_open(0, &dev));
  uint64_t num_cores = 0, nw = 0, nt = 0, lmem_cap = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));

  // Local memory of a block: the mbarrier at 0, then A (one per warpgroup
  // with -A), B per warpgroup. The first start has the 1024-byte phase of
  // DAK's buffers on the H100 (0x500).
  const uint32_t base0 = o.high ? 0x10500 : 0x500;
  const uint32_t layout = o.layout;
  Plan pa[kMaxGroups], pb[kMaxGroups];
  uint32_t next = base0;
  for (uint32_t ga = 0; ga < GA; ++ga) {
    pa[ga] = plan(kM, K, o.a_mn, layout, E, next, o.chain);
    next = align_up(pa[ga].hi, 128) + 0x180;
  }
  for (uint32_t gi = 0; gi < G; ++gi) {
    pb[gi] = plan(N, K, o.b_mn, layout, E, next, o.chain);
    next = align_up(pb[gi].hi, 128) + 0x80;
  }
  // Ring mode: two stages, each one wgmma's A and B, at their own offsets
  // (the plans above then only shape the tensors).
  Plan sa[2], sb[2];
  if (o.ring) {
    next = base0;
    for (uint32_t st = 0; st < 2; ++st) {
      sa[st] = plan(kM, KI, o.a_mn, layout, E, next, 1);
      next = align_up(sa[st].hi, 128) + 0x180;
      sb[st] = plan(N, KI, o.b_mn, layout, E, next, 1);
      next = align_up(sb[st].hi, 128) + 0x80;
    }
  }
  const uint32_t fill_lo = (o.ring ? sa[0].lo : pa[0].lo) & ~15u;
  uint32_t fill_hi = 0;
  for (uint32_t gi = 0; gi < G; ++gi)
    fill_hi = std::max(fill_hi, align_up(pb[gi].hi, 16));
  if (o.ring)
    fill_hi = align_up(sb[1].hi, 16);
  // The descriptors each wgmma must use.
  uint64_t want_a[kMaxGroups][kMaxChain], want_b[kMaxGroups][kMaxChain];
  for (uint32_t gi = 0; gi < G; ++gi) {
    for (uint32_t i = 0; i < o.chain; ++i) {
      want_a[gi][i] = o.ring ? sa[i & 1].desc(0) : pa[o.distinct_a ? gi : 0].desc(i);
      want_b[gi][i] = o.ring ? sb[i & 1].desc(0) : pb[gi].desc(i);
    }
  }
  const uint32_t smem = fill_hi;
  const uint32_t img_stride = align_up(fill_hi - fill_lo + 16, 64);

  std::printf("wgmma_h100: %s m64n%uk%u x %u, A %s B %s, swizzle %s, %u warpgroup(s)%s, %u block(s)%s, fill %s, "
              "A at %#x, B at %#x, %u bytes of local memory per block, reps %d%s%s; device %llu cores, "
              "%llu warps x %llu threads, %llu bytes\n",
              kPairNames[o.pair], N, KI, o.chain, o.a_mn ? "MN" : "K", o.b_mn ? "MN" : "K",
              layout == td::LAYOUT_NONE ? "none" : layout == td::LAYOUT_128B ? "128B" : layout == td::LAYOUT_64B ? "64B" : "32B",
              G, o.ring ? " fed by a 2-stage ring" : o.distinct_a ? " with their own A" : "", M, o.cluster ? " in clusters of 2" : "", o.tma ? "tma" : "sw",
              pa[0].start[0], pb[0].start[0], smem, o.reps, o.zero_first ? ", scale_d 0 first" : "",
              o.lbo0 ? ", K-major LBO 0" : "", (unsigned long long)num_cores, (unsigned long long)nw,
              (unsigned long long)nt, (unsigned long long)lmem_cap);
  const uint64_t need = VX_ISA_EXT_DXA;
  const char* bad_dev = nullptr;
  if ((isa & need) != need)
    bad_dev = "the device needs the DXA";
  else if (nt != 32 || nw * nt < threads)
    bad_dev = "the device needs 32-thread warps and enough warps for the warpgroups";
  else if (smem > lmem_cap)
    bad_dev = "the operands exceed the local memory";
  else if (M > 1 && (M + num_cores - 1) / num_cores * align_up(smem, 256) > lmem_cap)
    bad_dev = "the blocks cannot all be resident at once: their local memory exceeds a core's";
  else if (M > 1 && (M + num_cores - 1) / num_cores * (threads / 32) > nw)
    bad_dev = "the blocks cannot all be resident at once: they need more warps than a core has";
  else if (o.cluster && num_cores < 2)
    bad_dev = "clusters of 2 need 2 cores";
  if (bad_dev) {
    std::printf("Error: %s\nFAILED!\n", bad_dev);
    vx_device_release(dev);
    return 1;
  }

  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority = VX_QUEUE_PRIORITY_NORMAL;
  CHECK(vx_queue_create(dev, &qi, &q));
  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  vx_kernel_h k = nullptr;
  CHECK(vx_module_get_kernel(mod, o.cluster ? "wgmma_h100_c2" : "wgmma_h100", &k));
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)smem));
  vc::launch_config_t cfg;
  cfg.grid = vc::dim3(M, 1, 1);
  cfg.block = vc::dim3(threads, 1, 1);
  cfg.dyn_smem = smem;

  // Logical operands: A[b][ga][m][k], B[b][g][n][k], C[b][g][m][n] as raw encodings.
  const size_t a_el = (size_t)kM * K, b_el = (size_t)N * K, c_el = (size_t)kM * N;
  std::vector<uint32_t> A(M * GA * a_el), B(M * G * b_el), C(M * G * c_el);
  auto Aat = [&](uint32_t b, uint32_t ga, uint32_t m, uint32_t kk) -> uint32_t& { return A[(b * GA + ga) * a_el + (size_t)m * K + kk]; };
  auto Bat = [&](uint32_t b, uint32_t g, uint32_t n, uint32_t kk) -> uint32_t& { return B[(b * G + g) * b_el + (size_t)n * K + kk]; };
  auto Cat = [&](uint32_t b, uint32_t g, uint32_t m, uint32_t n) -> uint32_t& { return C[(b * G + g) * c_el + (size_t)m * N + n]; };

  const size_t slots = (size_t)M * kSlotsPerCta, descs = (size_t)M * kDescsPerCta;
  Buf d_img((uint64_t)img_stride * M), d_c(slots * 4), d_d(slots * 8 + kGuardBytes), d_frag(slots * 4 + kGuardBytes),
      d_desc(descs * 8 + kGuardBytes), d_place(M * sizeof(place_t) + kGuardBytes);
  // Global tensors for the tensor loads: per operand, the blocks' tensors
  // stacked along the outer dimension (K-major [b][mn][k], MN-major [b][k][mn]).
  std::vector<std::unique_ptr<Buf>> d_t(MAP_COUNT);
  for (uint32_t mi = 0; mi < MAP_COUNT; ++mi)
    d_t[mi].reset(new Buf(M * (mi < kMaxGroups ? a_el : b_el) * E));
  std::vector<box_t> boxes;
  Buf d_box(kMaxBoxes * sizeof(box_t));

  kernel_arg_t arg;
  std::memset(&arg, 0, sizeof(arg));
  arg.img_addr = d_img.addr;
  arg.box_addr = d_box.addr;
  arg.c_addr = d_c.addr;
  arg.d_addr = d_d.addr;
  arg.desc_addr = d_desc.addr;
  arg.frag_addr = d_frag.addr;
  arg.place_addr = d_place.addr;
  arg.pair = o.pair;
  arg.n = N;
  arg.tnsp_a = o.a_mn;
  arg.tnsp_b = o.b_mn;
  arg.groups = G;
  arg.chain = o.chain;
  arg.zero_first = o.zero_first;
  arg.tma = o.tma;
  arg.layout = layout;
  arg.lbo_a = o.ring ? sa[0].lbo : pa[0].lbo;
  arg.sbo_a = o.ring ? sa[0].sbo : pa[0].sbo;
  arg.lbo_b = o.ring ? sb[0].lbo : pb[0].lbo;
  arg.sbo_b = o.ring ? sb[0].sbo : pb[0].sbo;
  arg.ring = o.ring;
  arg.ki = KI;
  arg.skew = o.skew;
  arg.fill_lo = fill_lo;
  arg.fill_hi = fill_hi;
  arg.img_stride = img_stride;
  arg.mbar_off = 0;
  arg.distinct_a = o.distinct_a;
  for (uint32_t i = 0; i < o.chain; ++i) {
    for (uint32_t gi = 0; gi < G; ++gi) {
      arg.a_start[gi][i] = o.ring ? sa[i & 1].start[0] : pa[o.distinct_a ? gi : 0].start[i];
      arg.b_start[gi][i] = o.ring ? sb[i & 1].start[0] : pb[gi].start[i];
    }
  }
  for (uint32_t gi = 1; gi < G; ++gi) {
    if (pb[gi].lbo != pb[0].lbo || pb[gi].sbo != pb[0].sbo || pa[GA - 1].lbo != pa[0].lbo || pa[GA - 1].sbo != pa[0].sbo) {
      std::printf("Error: the warpgroups' operands differ in LBO/SBO\nFAILED!\n");
      return 1;
    }
  }

  // The operand behind each map, and the tensor maps and boxes (fixed across reps).
  const Plan* map_plan[MAP_COUNT] = {};
  for (uint32_t ga = 0; ga < GA; ++ga)
    map_plan[map_a(ga)] = &pa[ga];
  for (uint32_t gi = 0; gi < G; ++gi)
    map_plan[map_b(gi)] = &pb[gi];
  if (o.tma) {
    const vc::CUtensorMapDataType dt = o.pair == PAIR_BF16F32 ? vc::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
                                     : E == 1 ? vc::CU_TENSOR_MAP_DATA_TYPE_UINT8
                                     : vc::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    const vc::CUtensorMapSwizzle sw = layout == td::LAYOUT_128B ? vc::CU_TENSOR_MAP_SWIZZLE_128B
                                    : layout == td::LAYOUT_64B ? vc::CU_TENSOR_MAP_SWIZZLE_64B
                                    : layout == td::LAYOUT_32B ? vc::CU_TENSOR_MAP_SWIZZLE_32B
                                    : vc::CU_TENSOR_MAP_SWIZZLE_NONE;
    uint32_t tx = 0;
    for (uint32_t mi = 0; mi < MAP_COUNT; ++mi) {
      const Plan* p = map_plan[mi];
      if (!p)
        continue;
      const uint64_t inner = p->mn_major ? p->mn : p->k, outer = p->mn_major ? p->k : p->mn;
      const uint64_t dims[2] = {inner, outer * M};
      const uint64_t strides[1] = {inner * E};
      const uint32_t box[2] = {p->atom_el, 8};
      const uint32_t es[2] = {1, 1};
      CHECK(vc::tensor_map_encode_tiled(&arg.map[mi], dt, 2, d_t[mi]->addr, dims, strides, box, es,
                                        vc::CU_TENSOR_MAP_INTERLEAVE_NONE, sw, vc::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                        vc::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
      arg.blk_outer[mi] = (uint32_t)outer;
      arg.kdim[mi] = p->mn_major ? 1 : 0;
      if (o.ring)
        continue;
      for (const Box& b : p->boxes) {
        boxes.push_back({mi, b.inner, b.outer, b.dst});
        tx += p->atom_el * 8 * E;
      }
    }
    if (o.ring) {
      for (uint32_t st = 0; st < 2; ++st) {
        arg.ring_box_lo[st] = (uint32_t)boxes.size();
        uint32_t stx = 0;
        for (const Box& b : sa[st].boxes) {
          boxes.push_back({map_a(0), b.inner, b.outer, b.dst});
          stx += sa[st].atom_el * 8 * E;
        }
        for (const Box& b : sb[st].boxes) {
          boxes.push_back({map_b(0), b.inner, b.outer, b.dst});
          stx += sb[st].atom_el * 8 * E;
        }
        arg.ring_box_n[st] = (uint32_t)boxes.size() - arg.ring_box_lo[st];
        arg.ring_tx[st] = stx;
      }
    }
    if (boxes.size() > kMaxBoxes) {
      std::printf("Error: %zu boxes, at most %u\nFAILED!\n", boxes.size(), kMaxBoxes);
      return 1;
    }
    arg.nboxes = (uint32_t)boxes.size();
    arg.tx_bytes = tx;
    boxes.resize(kMaxBoxes);
    d_box.write(boxes.data());
  }

  std::vector<uint64_t> h_d(slots), h_desc(descs);
  std::vector<uint32_t> h_frag(slots);
  std::vector<place_t> h_place(M);
  std::vector<uint8_t> raw_d(d_d.size), raw_frag(d_frag.size), raw_desc(d_desc.size), raw_place(d_place.size);
  int failed = 0;
  for (int rep = 0; rep < o.reps; ++rep) {
    std::mt19937_64 g(20261112ull * 131 + (unsigned long long)rep * 7919 + o.pair);
    for (auto& x : A) x = rand_input(g, o.pair);
    for (auto& x : B) x = rand_input(g, o.pair);
    for (auto& x : C) x = rand_c(g, packed);

    // Initial accumulators per (b, g, r, t), as the fragment layout places C.
    std::vector<uint32_t> c_slots(slots, kNaNFill);
    for (uint32_t b = 0; b < M; ++b) {
      for (uint32_t gi = 0; gi < G; ++gi) {
        for (uint32_t t = 0; t < 128; ++t) {
          for (uint32_t r = 0; r < NRC; ++r) {
            const uint32_t m = td::acc_row(t, r, packed);
            const uint32_t v = packed ? Cat(b, gi, m, td::acc_col(t, r, 0, true)) | (Cat(b, gi, m, td::acc_col(t, r, 1, true)) << 16)
                                      : Cat(b, gi, m, td::acc_col(t, r, 0, false));
            c_slots[(size_t)b * kSlotsPerCta + ((size_t)gi * kMaxRegs + r) * 128 + t] = v;
          }
        }
      }
    }
    d_c.write(c_slots.data());

    if (!o.tma) {
      std::vector<uint8_t> img(d_img.size, 0xff);
      for (uint32_t b = 0; b < M; ++b) {
        uint8_t* bi = img.data() + (size_t)b * img_stride;
        auto put = [&](uint32_t a, uint32_t v) {
          for (uint32_t x = 0; x < E; ++x)
            bi[a - fill_lo + x] = (uint8_t)(v >> (8 * x));
        };
        for (uint32_t i = 0; i < o.chain; ++i) {
          for (uint32_t kk = 0; kk < KI; ++kk) {
            for (uint32_t ga = 0; ga < GA; ++ga)
              for (uint32_t m = 0; m < kM; ++m)
                put(pa[ga].addr(i, m, kk), Aat(b, ga, m, i * KI + kk));
            for (uint32_t gi = 0; gi < G; ++gi)
              for (uint32_t n = 0; n < N; ++n)
                put(pb[gi].addr(i, n, kk), Bat(b, gi, n, i * KI + kk));
          }
        }
      }
      d_img.write(img.data());
    } else {
      for (uint32_t mi = 0; mi < MAP_COUNT; ++mi) {
        const Plan* p = map_plan[mi];
        if (!p)
          continue;
        const bool is_a = mi < kMaxGroups;
        const uint32_t sub = is_a ? mi : mi - kMaxGroups;
        std::vector<uint8_t> t((size_t)M * p->mn * p->k * E);
        for (uint32_t b = 0; b < M; ++b) {
          for (uint32_t m = 0; m < p->mn; ++m) {
            for (uint32_t kk = 0; kk < p->k; ++kk) {
              const size_t idx = (size_t)b * p->mn * p->k + (p->mn_major ? (size_t)kk * p->mn + m : (size_t)m * p->k + kk);
              const uint32_t v = is_a ? Aat(b, sub, m, kk) : Bat(b, sub, m, kk);
              for (uint32_t x = 0; x < E; ++x)
                t[idx * E + x] = (uint8_t)(v >> (8 * x));
            }
          }
        }
        d_t[mi]->write(t.data());
      }
    }
    {
      std::vector<uint8_t> v(d_d.size);
      for (size_t i = 0; i < slots; ++i)
        std::memcpy(&v[i * 8], &kNaNFill64, 8);
      for (size_t i = slots * 8; i + 4 <= v.size(); i += 4)
        std::memcpy(&v[i], &kGuardWord, 4);
      d_d.write(v.data());
    }
    d_frag.write(guarded(slots * 4, kNaNFill).data());
    d_desc.write(guarded(descs * 8, kNaNFill).data());
    d_place.write(guarded(M * sizeof(place_t), kNaNFill).data());

    vx_event_h ev = nullptr;
    if (o.cluster)
      CHECK(vc::launch_kernel_ex<wgmma_h100_c2_cluster_dims>(q, k, cfg, arg, &ev));
    else
      CHECK(vc::launch_kernel_ex(q, k, cfg, arg, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    d_d.read(raw_d.data());
    d_frag.read(raw_frag.data());
    d_desc.read(raw_desc.data());
    d_place.read(raw_place.data());
    bool ok = guard_intact(raw_d, slots * 8, "D", rep) && guard_intact(raw_frag, slots * 4, "fragment map", rep)
           && guard_intact(raw_desc, descs * 8, "descriptors", rep)
           && guard_intact(raw_place, M * sizeof(place_t), "placement", rep);
    std::memcpy(h_d.data(), raw_d.data(), slots * 8);
    std::memcpy(h_frag.data(), raw_frag.data(), slots * 4);
    std::memcpy(h_desc.data(), raw_desc.data(), descs * 8);
    std::memcpy(h_place.data(), raw_place.data(), M * sizeof(place_t));

    // Placement: blocks sharing a core sit in distinct slots, some nonzero;
    // the two blocks of a cluster run on distinct cores.
    long long bad_place = 0;
    bool shared_core = false, nonzero_slot = false;
    for (uint32_t b = 0; b < M; ++b) {
      if (h_place[b].core >= num_cores) {
        std::printf("Error: rep %d: block %u reports core %u\n", rep, b, h_place[b].core);
        ++bad_place;
      }
      nonzero_slot |= (h_place[b].slot != 0);
      for (uint32_t b2 = 0; b2 < b; ++b2) {
        if (h_place[b2].core == h_place[b].core) {
          shared_core = true;
          if (h_place[b2].slot == h_place[b].slot) {
            std::printf("Error: rep %d: blocks %u and %u share core %u and slot %#x\n", rep, b2, b, h_place[b].core,
                        h_place[b].slot);
            ++bad_place;
          }
        }
      }
      if (h_place[b].warp % 4 != 0) {
        std::printf("Error: rep %d: block %u starts at warp %u, not a warpgroup boundary\n", rep, b, h_place[b].warp);
        ++bad_place;
      }
      if (o.cluster && (b & 1) && h_place[b].core == h_place[b - 1].core) {
        std::printf("Error: rep %d: cluster %u runs both blocks on core %u\n", rep, b / 2, h_place[b].core);
        ++bad_place;
      }
    }
    if (M > 1 && num_cores == 1 && !(shared_core && nonzero_slot)) {
      std::printf("Error: rep %d: no block ran at a nonzero local-memory slot of a shared core\n", rep);
      ++bad_place;
    }

    long long bad_desc = 0, bad_frag = 0, bad_val = 0, bad_box = 0;
    for (uint32_t b = 0; b < M; ++b) {
      for (uint32_t gi = 0; gi < kMaxGroups; ++gi) {
        for (uint32_t i = 0; i < kMaxChain; ++i) {
          const uint64_t* dd = &h_desc[(size_t)b * kDescsPerCta + (gi * kMaxChain + i) * 2];
          const bool used = gi < G && i < o.chain;
          const uint64_t wa = used ? want_a[gi][i] : 0xffffffffffffffffull;
          const uint64_t wb = used ? want_b[gi][i] : 0xffffffffffffffffull;
          if (dd[0] != wa || dd[1] != wb) {
            if (bad_desc < 4)
              std::printf("Error: rep %d: block %u group %u wgmma %u descriptors %#llx %#llx, expected %#llx %#llx\n",
                          rep, b, gi, i, (unsigned long long)dd[0], (unsigned long long)dd[1], (unsigned long long)wa,
                          (unsigned long long)wb);
            ++bad_desc;
          }
        }
      }
    }
    // Reference per (b, g, m, n): k-step halves in order, C in the D format.
    const uint32_t fs = fmt_s(o.pair), fd = packed ? tf::FMT_FP16 : tf::FMT_FP32;
    const uint32_t per_word = 4 / E, half_k = KI / 2;
    for (uint32_t b = 0; b < M; ++b) {
      for (uint32_t gi = 0; gi < G; ++gi) {
        const uint32_t ga = o.distinct_a ? gi : 0;
        std::vector<uint32_t> ref(c_el);
        for (uint32_t m = 0; m < kM; ++m) {
          for (uint32_t n = 0; n < N; ++n) {
            uint32_t c = Cat(b, gi, m, n);
            for (uint32_t i = 0; i < o.chain; ++i) {
              if (i == 0 && o.zero_first)
                c = 0;
              for (uint32_t hk = 0; hk < 2; ++hk) {
                uint32_t aw[4] = {0, 0, 0, 0}, bw[4] = {0, 0, 0, 0};
                for (uint32_t x = 0; x < half_k; ++x) {
                  const uint32_t kk = i * KI + hk * half_k + x;
                  aw[x / per_word] |= Aat(b, ga, m, kk) << (8 * E * (x % per_word));
                  bw[x / per_word] |= Bat(b, gi, n, kk) << (8 * E * (x % per_word));
                }
                c = tf::fedp(fs, fd, aw, bw, c, 4);
              }
            }
            ref[(size_t)m * N + n] = c;
          }
        }
        for (uint32_t t = 0; t < 128; ++t) {
          for (uint32_t r = 0; r < NRC; ++r) {
            const size_t s = (size_t)b * kSlotsPerCta + ((size_t)gi * kMaxRegs + r) * 128 + t;
            const uint32_t m = td::acc_row(t, r, packed);
            const uint32_t want_frag = (m << 16) | td::acc_col(t, r, 0, packed);
            if (h_frag[s] != want_frag) {
              if (bad_frag < 4)
                std::printf("Error: rep %d: block %u thread %u register %u: library row/col %#x, expected %#x\n", rep, b,
                            t, r, h_frag[s], want_frag);
              ++bad_frag;
            }
            if ((h_d[s] >> 32) != 0xffffffffull) {
              if (bad_box < 4)
                std::printf("Error: rep %d: block %u group %u thread %u register %u: bits 63:32 = %#llx, not the NaN box\n",
                            rep, b, gi, t, r, (unsigned long long)(h_d[s] >> 32));
              ++bad_box;
            }
            const uint32_t lo = (uint32_t)h_d[s];
            for (uint32_t hh = 0; hh < (packed ? 2u : 1u); ++hh) {
              const uint32_t n = td::acc_col(t, r, hh, packed);
              const uint32_t want = ref[(size_t)m * N + n];
              const uint32_t got = packed ? (lo >> (16 * hh)) & 0xffffu : lo;
              if (got != want) {
                if (bad_val < 8)
                  std::printf("Error: rep %d: block %u group %u thread %u register %u%s = D(%u, %u): %#x, expected %#x\n",
                              rep, b, gi, t, r, packed ? (hh ? " hi" : " lo") : "", m, n, got, want);
                ++bad_val;
              }
            }
          }
        }
      }
    }
    const bool pass = ok && !bad_desc && !bad_frag && !bad_val && !bad_box && !bad_place;
    failed += !pass;
    std::printf("rep %d: %s bad_values=%lld bad_nan_box=%lld bad_descriptors=%lld bad_fragment_map=%lld bad_placement=%lld",
                rep, pass ? "PASS" : "FAIL", bad_val, bad_box, bad_desc, bad_frag, bad_place);
    for (uint32_t b = 0; b < M; ++b)
      std::printf(" [block %u: core %u slot %#x warp %u]", b, h_place[b].core, h_place[b].slot, h_place[b].warp);
    std::printf("\n");
  }

  vx_kernel_release(k);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);
  if (failed) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
