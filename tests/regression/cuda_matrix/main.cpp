// cuda_matrix: stmatrix / ldmatrix .m8n8.b16 (x1, x2, x4, with and without
// .trans) and named barriers (bar.sync ids 1-15, bar.arrive) of the CUDA porting layer, one
// block of 8 warps of 32 threads on one core (NUM_BARRIERS 16).
//
//   -r <reps>   launches, each with fresh data (default 2)
//   -p <mask>   parts to run: 1 stmatrix/ldmatrix, 2 DAK epilogue, 4 DeepGEMM
//               epilogue, 8 named barriers (default 15)
//   -h          this help
//
// Byte model (PTX ISA, verified against CuTe on the H100 toolchain): lanes
// 8q..8q+7 give matrix q's row addresses; lane L's register q, half e is
//   non-trans: addr[8q + L/4] + 2(2(L%4) + e),  trans: addr[8q + 2(L%4) + e] + 2(L/4).
// A launch passes when the whole local memory at the end equals the initial
// image with exactly the modelled stores applied (untouched rows included),
// every ldmatrix register equals the modelled load, DAK's stmatrix.x4.trans
// epilogue leaves its 64 x 256 f16 tile at the 128B-swizzled sC position of
// each fragment element (and supplies the row addresses CuTe printed),
// DeepGEMM's stmatrix.x2 epilogue leaves its 128 x 192 bf16 tile at the
// TMA store box positions, and every barrier round's reader sees the
// writer's word (the writer syncing, or arriving with bar.arrive). Exit 0
// when every launch passes.

#include <vortex2.h>
#include <vx_cuda_runtime.h>
#include <VX_types.h>
#include "common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unistd.h>
#include <vector>

namespace {

namespace vc = vx::cuda;

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

vx_device_h dev = nullptr;
vx_queue_h q = nullptr;
int reps = 2;
uint32_t phases = PHASE_ALL;

void usage() { std::printf("Usage: [-r reps] [-p mask] [-h]\n"); }

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:p:h")) != -1) {
    switch (c) {
    case 'p':
      phases = (uint32_t)std::strtoul(optarg, nullptr, 0);
      if (phases == 0 || phases > PHASE_ALL) { usage(); std::exit(1); }
      break;
    case 'r':
      reps = std::atoi(optarg);
      if (reps < 1) { usage(); std::exit(1); }
      break;
    case 'h': usage(); std::exit(0);
    default:  usage(); std::exit(1);
    }
  }
  if (optind < argc) {
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

std::vector<uint8_t> guarded(size_t body, uint8_t fill) {
  std::vector<uint8_t> v(body + kGuardBytes, fill);
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

void put16(std::vector<uint8_t>& img, uint32_t a, uint32_t v) {
  img[a] = (uint8_t)v;
  img[a + 1] = (uint8_t)(v >> 8);
}
uint32_t get16(const std::vector<uint8_t>& img, uint32_t a) { return img[a] | (img[a + 1] << 8); }

// The byte address of lane L's register q, half e.
uint32_t model_addr(const uint32_t* rows, bool trans, uint32_t L, uint32_t q, uint32_t e) {
  return trans ? rows[8 * q + 2 * (L % 4) + e] + 2 * (L / 4) : rows[8 * q + L / 4] + 2 * (2 * (L % 4) + e);
}

uint32_t sw128(uint32_t x) { return x ^ (((x >> 7) & 7u) << 4); }

} // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  parse_args(argc, argv);

  CHECK(vx_device_open(0, &dev));
  uint64_t nw = 0, nt = 0, lmem_cap = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));
  std::printf("cuda_matrix: reps %d; %llu warps x %llu threads, %llu bytes of local memory, %u bytes used, "
              "%d named barriers\n",
              reps, (unsigned long long)nw, (unsigned long long)nt, (unsigned long long)lmem_cap, LM_BYTES,
              VX_CFG_NUM_BARRIERS);
  const char* bad_dev = nullptr;
  if (nt != 32 || nw < kWarps)
    bad_dev = "the device needs 32-thread warps and 8 warps per core";
  else if (lmem_cap < LM_BYTES)
    bad_dev = "the device needs 256 KB of local memory";
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
  CHECK(vx_module_get_kernel(mod, "cuda_matrix", &k));
  CHECK(vc::func_set_attribute(dev, k, vc::func_attribute::max_dynamic_shared_memory_size, (int)LM_BYTES));
  vc::launch_config_t cfg;
  cfg.grid = vc::dim3(1, 1, 1);
  cfg.block = vc::dim3(kThreads, 1, 1);
  cfg.dyn_smem = LM_BYTES;

  const size_t mat_n = (size_t)kVariants * kWarps * 32, ld_words = (size_t)6 * kWarps * 32 * 4;
  const size_t dak_n = 128 * 64, dak_rec = 128 * kDakCalls, dg_n = (size_t)kThreads * kDgCalls * 2;
  Buf d_init(LM_BYTES), d_image(LM_BYTES + kGuardBytes), d_mat(mat_n * sizeof(mat_entry_t)),
      d_ld(ld_words * 4 + kGuardBytes), d_dak(dak_n * 4), d_dak_rec(dak_rec * 4 + kGuardBytes), d_dg(dg_n * 4),
      d_bar(kBarRounds * 4 + kGuardBytes);

  kernel_arg_t arg;
  std::memset(&arg, 0, sizeof(arg));
  arg.init_addr = d_init.addr;
  arg.image_addr = d_image.addr;
  arg.mat_addr = d_mat.addr;
  arg.ld_addr = d_ld.addr;
  arg.dak_regs = d_dak.addr;
  arg.dak_addr = d_dak_rec.addr;
  arg.dg_words = d_dg.addr;
  arg.bar_addr = d_bar.addr;
  arg.phases = phases;

  int failed = 0;
  for (int rep = 0; rep < reps; ++rep) {
    std::mt19937_64 g(20261112ull + 1009ull * (unsigned long long)rep);
    std::vector<uint8_t> init(LM_BYTES);
    for (auto& b : init)
      b = (uint8_t)g();
    std::memset(&init[LM_BAR], 0, LM_BYTES - LM_BAR);
    arg.bar_tokens = (uint32_t)g() | 1u;

    // Row offsets: distinct random rows below the poison rows for the lanes
    // that give addresses, poison rows for the others.
    std::vector<mat_entry_t> mat(mat_n);
    for (uint32_t v = 0; v < kVariants; ++v) {
      const uint32_t x = (v % 3 == 0) ? 1 : (v % 3 == 1) ? 2 : 4;
      for (uint32_t w = 0; w < kWarps; ++w) {
        std::vector<uint32_t> rows(kPoisonRow);
        for (uint32_t i = 0; i < kPoisonRow; ++i) rows[i] = i;
        std::shuffle(rows.begin(), rows.end(), g);
        const uint32_t region = LM_MATRIX + (v * kWarps + w) * kRegion;
        for (uint32_t L = 0; L < 32; ++L) {
          mat_entry_t& e = mat[(v * kWarps + w) * 32 + L];
          e.row = region + 16 * (L < 8 * x ? rows[L] : kPoisonRow + (uint32_t)(g() % (kRows - kPoisonRow)));
          for (uint32_t r = 0; r < 4; ++r) e.r[r] = (uint32_t)g();
        }
      }
    }
    std::vector<uint32_t> dak(dak_n), dg(dg_n);
    for (auto& x : dak) x = (uint32_t)g();
    for (auto& x : dg) x = (uint32_t)g();

    // The expected local memory and ldmatrix registers.
    std::vector<uint8_t> want = init;
    std::vector<uint32_t> want_ld(ld_words, 0);
    for (uint32_t v = 0; v < kVariants && (phases & PHASE_MATRIX); ++v) {
      const uint32_t x = (v % 3 == 0) ? 1 : (v % 3 == 1) ? 2 : 4;
      const bool trans = (v % 6) >= 3;
      for (uint32_t w = 0; w < kWarps; ++w) {
        uint32_t rows[32];
        for (uint32_t L = 0; L < 32; ++L) rows[L] = mat[(v * kWarps + w) * 32 + L].row;
        for (uint32_t L = 0; L < 32; ++L) {
          const mat_entry_t& e = mat[(v * kWarps + w) * 32 + L];
          for (uint32_t qq = 0; qq < x; ++qq) {
            for (uint32_t h = 0; h < 2; ++h) {
              const uint32_t a = model_addr(rows, trans, L, qq, h);
              if (v < 6)
                put16(want, a, e.r[qq] >> (16 * h));
              else
                want_ld[(((v - 6) * kWarps + w) * 32 + L) * 4 + qq] |= get16(init, a) << (16 * h);
            }
          }
        }
      }
    }
    // DAK: element (m, n) of the packed fragment at sC(m, n) = n*128 + ((m/8) ^ (n%8))*16 + (m%8)*2.
    for (uint32_t t = 0; t < 128 && (phases & PHASE_DAK); ++t) {
      for (uint32_t r = 0; r < 64; ++r) {
        for (uint32_t h = 0; h < 2; ++h) {
          const uint32_t m = 16 * (t / 32) + (t % 32) / 4 + 8 * (r & 1), n = 8 * (r / 2) + 2 * (t % 4) + h;
          put16(want, LM_DAK + n * 128 + (((m / 8) ^ (n % 8)) * 16) + (m % 8) * 2, dak[t * 64 + r] >> (16 * h));
        }
      }
    }
    // DeepGEMM: call i, word q, half e of thread t = accumulator 4i + 2q + e of
    // the fp32 fragment, at (n/64)*16384 + sw128(m*128 + 2(n%64)); warps 4-7
    // hold rows 64-127.
    for (uint32_t t = 0; t < kThreads && (phases & PHASE_DG); ++t) {
      for (uint32_t i = 0; i < kDgCalls; ++i) {
        for (uint32_t qq = 0; qq < 2; ++qq) {
          for (uint32_t e = 0; e < 2; ++e) {
            const uint32_t m = 16 * (t / 32) + (t % 32) / 4 + 8 * qq, n = 8 * i + 2 * (t % 4) + e;
            put16(want, LM_DG + (n / 64) * 16384 + sw128(m * 128 + 2 * (n % 64)), dg[(t * kDgCalls + i) * 2 + qq] >> (16 * e));
          }
        }
      }
    }
    for (uint32_t r = 0; r < kBarRounds && (phases & PHASE_BAR); ++r) {
      const uint32_t tok = arg.bar_tokens + r;
      std::memcpy(&want[LM_BAR + 4 * r], &tok, 4);
    }

    d_init.write(init.data());
    d_mat.write(mat.data());
    d_dak.write(dak.data());
    d_dg.write(dg.data());
    d_image.write(guarded(LM_BYTES, 0xA5).data());
    d_ld.write(guarded(ld_words * 4, 0xA5).data());
    d_dak_rec.write(guarded(dak_rec * 4, 0xA5).data());
    d_bar.write(guarded(kBarRounds * 4, 0).data());

    vx_event_h ev = nullptr;
    CHECK(vc::launch_kernel_ex(q, k, cfg, arg, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);

    std::vector<uint8_t> image(d_image.size), ld(d_ld.size), rec(d_dak_rec.size), bar(d_bar.size);
    d_image.read(image.data());
    d_ld.read(ld.data());
    d_dak_rec.read(rec.data());
    d_bar.read(bar.data());
    bool ok = guard_intact(image, LM_BYTES, "the image", rep) && guard_intact(ld, ld_words * 4, "ldmatrix", rep)
           && guard_intact(rec, dak_rec * 4, "DAK addresses", rep) && guard_intact(bar, kBarRounds * 4, "barriers", rep);

    long long bad_mat = 0, bad_dak = 0, bad_dg = 0, bad_ld = 0, bad_rec = 0, bad_bar = 0;
    for (uint32_t a = 0; a < LM_BYTES; ++a) {
      if (image[a] == want[a])
        continue;
      long long& cnt = a < LM_DAK ? bad_mat : a < LM_DG ? bad_dak : a < LM_BAR ? bad_dg : bad_bar;
      if (cnt < 4)
        std::printf("Error: rep %d: local memory byte %#x = %#04x, expected %#04x\n", rep, a, image[a], want[a]);
      ++cnt;
    }
    for (size_t i = 0; i < ld_words && (phases & PHASE_MATRIX); ++i) {
      uint32_t got;
      std::memcpy(&got, &ld[4 * i], 4);
      const uint32_t L = (uint32_t)(i / 4) % 32, qq = (uint32_t)(i % 4), v = (uint32_t)(i / 4 / 32 / kWarps);
      const uint32_t x = (v % 3 == 0) ? 1 : (v % 3 == 1) ? 2 : 4;
      if (qq >= x)
        continue;
      if (got != want_ld[i]) {
        if (bad_ld < 4)
          std::printf("Error: rep %d: ldmatrix variant %u warp %zu lane %u register %u = %#x, expected %#x\n", rep, v,
                      (i / 4 / 32) % kWarps, L, qq, got, want_ld[i]);
        ++bad_ld;
      }
    }
    // Row addresses DAK's lanes supplied: the formula, and the rows CuTe printed.
    struct Sample { uint32_t t, c, addr; };
    const Sample samples[] = {
      {0, 0, 0x00000}, {0, 1, 0x00800}, {0, 2, 0x01000}, {0, 3, 0x01800}, {1, 0, 0x00090}, {1, 1, 0x00890},
      {4, 0, 0x00240}, {4, 3, 0x01a40}, {37, 0, 0x002f0}, {37, 1, 0x00af0}, {37, 5, 0x02af0},
    };
    for (const Sample& s : samples) {
      if (!(phases & PHASE_DAK))
        break;
      uint32_t got;
      std::memcpy(&got, &rec[4 * (s.t * kDakCalls + s.c)], 4);
      if (got != s.addr) {
        std::printf("Error: rep %d: DAK thread %u call %u row %#x, CuTe gives %#x\n", rep, s.t, s.c, got, s.addr);
        ++bad_rec;
      }
    }
    for (uint32_t r = 0; r < kBarRounds && (phases & PHASE_BAR); ++r) {
      uint32_t got;
      std::memcpy(&got, &bar[4 * r], 4);
      if (got != arg.bar_tokens + r) {
        std::printf("Error: rep %d: barrier round %u (bar %u, %u threads, writer %s): reader saw %#x, expected %#x\n",
                    rep, r, kBarRound[r].id, kBarRound[r].nthreads, kBarRound[r].arrive ? "arrives" : "syncs", got,
                    arg.bar_tokens + r);
        ++bad_bar;
      }
    }
    const bool pass = ok && !bad_mat && !bad_dak && !bad_dg && !bad_ld && !bad_rec && !bad_bar;
    failed += !pass;
    std::printf("rep %d: %s matrix_bytes=%lld ldmatrix=%lld dak_bytes=%lld dak_rows=%lld deepgemm_bytes=%lld "
                "barriers=%lld\n",
                rep, pass ? "PASS" : "FAIL", bad_mat, bad_ld, bad_dak, bad_rec, bad_dg, bad_bar);
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
