// lmem_lrsc_dma: a DXA write into local memory breaks the LR/SC reservations
// on the bytes it writes.
//
// RISC-V requires a store-conditional to fail when another device writes the
// reserved bytes between the load-reserved and the store-conditional. Each
// round, lanes of the one warp reserve local-memory words, the warp copies a
// tile into local memory with the DXA, and once the copy is visible the lanes
// try their store-conditionals:
//   R1  the copy writes the reserved word: the SC must fail and the word must
//       hold the copied data.
//   R2  the reserved word lies outside the copy: the SC must succeed.
//   R3  the reserved word shares the copy's partial last bank row but sits in a
//       bank the copy does not write: the SC must succeed.
//   R4  the lane reserves the copied word again before its SC: it must succeed.
//   R5  (RV64) lr.d/sc.d on a word the copy writes only the upper half of: the
//       SC must fail, and succeed after a re-LR.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <vortex.h>
#include <dxa.h>

#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

constexpr uint32_t kDescSrc    = 0;
constexpr uint32_t kWordBytes  = VX_CFG_XLEN / 8;  // local-memory bank word
constexpr uint32_t kNumBanks   = VX_CFG_LMEM_NUM_BANKS;
constexpr uint32_t kRowBytes   = kNumBanks * kWordBytes;
constexpr uint32_t kBlockBytes = VX_CFG_MEM_BLOCK_SIZE;
constexpr uint32_t kSpanBytes  = std::max(kBlockBytes, kRowBytes);

// Two blocks plus two bank words: the copy has words at block bases, interior
// words, and a last bank row it only partly writes.
constexpr uint32_t kTileBytes = 2 * kBlockBytes + 2 * kWordBytes;
constexpr uint32_t kTileElems = kTileBytes / 4;
// Source tiles start on a cache line, the common DXA read path.
constexpr uint32_t kSrcStride = (kTileElems + kBlockBytes / 4 - 1) / (kBlockBytes / 4) * (kBlockBytes / 4);

constexpr uint32_t align_up(uint32_t x, uint32_t a) {
  return (x + a - 1) / a * a;
}

constexpr uint32_t kTileOff = 0;
constexpr uint32_t kCtlOff  = align_up(kTileOff + kTileBytes, kSpanBytes);
// Not a block base: only the bytes the copy writes tie it to this word, not
// the block they fall in.
constexpr uint32_t kHalfOff = kCtlOff + kSpanBytes + kWordBytes;
constexpr uint32_t kHalfDst = kHalfOff + 4;
constexpr uint32_t kLmemUsed = align_up(kHalfDst + kTileBytes, kBlockBytes);

enum class Expect { Fail, Succeed, Either };

struct Target {
  uint32_t    group;
  const char* what;
  uint32_t    word_off;
  uint32_t    dword;
  uint32_t    re_lr;
  uint32_t    pref_lane;
};

struct Round {
  uint32_t group;
  uint32_t sub;
  uint32_t dst_off;
  int32_t  target[MAX_LANES];
  std::vector<bool> bank_used;
};

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h plan_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (src_buffer) {
      vx_buffer_release(src_buffer);
    }
    if (plan_buffer) {
      vx_buffer_release(plan_buffer);
    }
    if (out_buffer) {
      vx_buffer_release(out_buffer);
    }
    if (kernel) {
      vx_kernel_release(kernel);
    }
    if (module_) {
      vx_module_release(module_);
    }
    if (queue) {
      vx_queue_release(queue);
    }
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static uint32_t src_value(uint32_t idx) {
  return 0xDA000000u | idx;
}

static uint32_t bank_of(uint32_t off) {
  return (off / kWordBytes) % kNumBanks;
}

static const char* expect_name(Expect e) {
  switch (e) {
  case Expect::Fail:    return "fail";
  case Expect::Succeed: return "succeed";
  default:              return "either";
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_STD_A) == 0 || (isa_flags & VX_ISA_EXT_DXA) == 0) {
    std::cout << "Error: needs both the A and the DXA extensions" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  const uint32_t threads = (uint32_t)num_threads;
  // A DXA issue gathers its operands from lanes 0-3 (vx_dxa.h).
  if (threads < 4) {
    std::cout << "Error: needs at least 4 threads per warp for the DXA issue" << std::endl;
    cleanup();
    return -1;
  }
  const uint32_t lanes = std::min<uint32_t>({MAX_LANES, VX_CFG_NUM_LSU_LANES, kNumBanks, threads});

  // Words, by byte offset in the CTA's local memory. Lane 0 of the one CTA
  // is hart 0 of core 0.
  std::vector<Target> targets = {
    {1, "block base",             kTileOff,                                     0, 0, 0},
    {1, "other block base",       kTileOff + kBlockBytes,                       0, 0, 1},
    {1, "interior word",          kTileOff + kBlockBytes + 3 * kWordBytes - 4,  0, 0, 2},
    {1, "last tile element",      kTileOff + kTileBytes - 4,                    0, 0, 3},
  };
  for (uint32_t l = 0; l < lanes; ++l) {
    targets.push_back({2, "outside the copy", kCtlOff + l * kWordBytes, 0, 0, l});
  }
  const uint32_t tail = kTileBytes % kRowBytes;
  const uint32_t tail_row = kTileOff + kTileBytes - tail;
  for (uint32_t b = tail / kWordBytes, l = 0; tail != 0 && b < kNumBanks && l < lanes; ++b, ++l) {
    targets.push_back({3, "unwritten bank, copy row", tail_row + b * kWordBytes, 0, 0, l});
  }
  for (uint32_t i = 0; i < 4; ++i) {
    Target t = targets[i];
    t.group = 4;
    t.re_lr = 1;
    targets.push_back(t);
  }
#if VX_CFG_XLEN == 64
  // The re-LR reads back exactly which half the copy wrote.
  targets.push_back({5, "upper half copied (.d)",   kHalfOff, 1, 0, 1});
  targets.push_back({5, "upper half copied, re-LR", kHalfOff, 1, 1, 2});
#endif

  // A bank's reservation stations are indexed by row, so two live
  // reservations in one bank can evict each other and fail an SC for a reason
  // unrelated to the copy. Each round therefore keeps one reservation per bank
  // and per lane; a target that does not fit opens another round of its group.
  std::vector<Round> rounds;
  uint32_t group_first = 0;
  for (uint32_t i = 0; i < targets.size(); ++i) {
    const Target& t = targets[i];
    if (i == 0 || targets[i - 1].group != t.group) {
      group_first = (uint32_t)rounds.size();
    }
    const uint32_t lane = t.pref_lane % lanes;
    const uint32_t bank = bank_of(t.word_off);
    uint32_t r = group_first;
    while (r < rounds.size() && (rounds[r].target[lane] >= 0 || rounds[r].bank_used[bank])) {
      ++r;
    }
    if (r == rounds.size()) {
      Round nr;
      nr.group   = t.group;
      nr.sub     = r - group_first;
      nr.dst_off = (t.group == 5) ? kHalfDst : kTileOff;
      std::fill(std::begin(nr.target), std::end(nr.target), -1);
      nr.bank_used.assign(kNumBanks, false);
      rounds.push_back(nr);
    }
    rounds[r].target[lane] = (int32_t)i;
    rounds[r].bank_used[bank] = true;
  }
  const uint32_t num_rounds = (uint32_t)rounds.size();

  const uint32_t lmem_bytes = kLmemUsed + LMEM_ALIGN;
  const uint32_t src_elems  = num_rounds * kSrcStride;

  std::cout << "xlen: " << VX_CFG_XLEN << ", lmem banks: " << kNumBanks
            << ", bank word: " << kWordBytes << "B, lanes: " << lanes
            << ", rounds: " << num_rounds << ", tile: " << kTileBytes << "B" << std::endl;
  if (tail == 0) {
    std::cout << "R3 skipped: the " << kTileBytes << "B copy fills whole " << kRowBytes
              << "B bank rows, so no copy row leaves a bank unwritten" << std::endl;
  }

  RT_CHECK(vx_check_occupancy(device, threads, lmem_bytes));

  std::vector<uint32_t> h_src(src_elems);
  for (uint32_t i = 0; i < src_elems; ++i) {
    h_src[i] = src_value(i);
  }

  std::vector<round_plan_t> h_plan(num_rounds);
  std::memset(h_plan.data(), 0, num_rounds * sizeof(round_plan_t));
  for (uint32_t r = 0; r < num_rounds; ++r) {
    round_plan_t& p = h_plan[r];
    p.dst_off   = rounds[r].dst_off;
    p.src_coord = r * kSrcStride;
    p.probe_off = p.dst_off + kTileBytes - 4;
    p.probe_val = src_value(p.src_coord + kTileElems - 1);
    for (uint32_t l = 0; l < MAX_LANES; ++l) {
      if (rounds[r].target[l] < 0) {
        continue;
      }
      const Target& t = targets[rounds[r].target[l]];
      p.lanes[l].active   = 1;
      p.lanes[l].word_off = t.word_off;
      p.lanes[l].dword    = t.dword;
      p.lanes[l].re_lr    = t.re_lr;
    }
  }

  const size_t out_bytes = (size_t)num_rounds * MAX_LANES * sizeof(lane_out_t);
  std::vector<lane_out_t> h_out((size_t)num_rounds * MAX_LANES);
  std::memset(h_out.data(), 0xCC, out_bytes);

  kernel_arg_t kernel_arg = {};
  kernel_arg.num_rounds = num_rounds;

  uint64_t src_addr = 0;
  RT_CHECK(vx_buffer_create(device, src_elems * sizeof(uint32_t), VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_create(device, num_rounds * sizeof(round_plan_t), VX_MEM_READ, &plan_buffer));
  RT_CHECK(vx_buffer_create(device, out_bytes, VX_MEM_READ_WRITE, &out_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &src_addr));
  RT_CHECK(vx_buffer_address(plan_buffer, &kernel_arg.plan_addr));
  RT_CHECK(vx_buffer_address(out_buffer, &kernel_arg.out_addr));

  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), src_elems * sizeof(uint32_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, plan_buffer, 0, h_plan.data(), num_rounds * sizeof(round_plan_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, out_buffer, 0, h_out.data(), out_bytes, 0, nullptr, nullptr));

  RT_CHECK(vortex::dxa::program_1d(device, kDescSrc, src_addr,
    src_elems, kTileElems, sizeof(uint32_t)));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "start" << std::endl;
  vx_event_h launch_ev = nullptr, rd_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = 1;
    li.block_dim[0] = threads;
    li.lmem_size    = lmem_bytes;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  RT_CHECK(vx_enqueue_read(queue, h_out.data(), out_buffer, 0, out_bytes, 1, &launch_ev, &rd_ev));
  RT_CHECK(vx_event_wait_value(rd_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(rd_ev);
  vx_event_release(launch_ev);

  cleanup();

  enum { kNoOutput, kLost, kBadOld, kBadReLr, kScOverCopy, kCopyLost,
         kSpuriousFail, kReLrFail, kStoreLost, kInconsistent, kNumKinds };
  static const char* const kind_msg[kNumKinds] = {
    "lane never reported",
    "copy never became visible after its barrier",
    "LR did not return the stored sentinel",
    "re-LR did not return the copied data",
    "SC succeeded although the DXA wrote the reserved bytes",
    "SC failed but the word does not hold the copied data",
    "SC failed although the DXA did not write the reserved bytes",
    "SC failed after re-reserving the copied word",
    "SC succeeded but the word does not hold its value",
    "SC outcome and memory disagree",
  };
  uint32_t kind_count[kNumKinds] = {};
  uint32_t lane0_hart = UINT32_MAX;
  const uint32_t poison = 0xCCCCCCCCu;

  for (uint32_t r = 0; r < num_rounds; ++r) {
    const Round& rd = rounds[r];
    const round_plan_t& p = h_plan[r];
    const uint32_t copy_lo = p.dst_off;
    const uint32_t copy_hi = p.dst_off + kTileBytes;
    uint32_t spins = 0;
    for (uint32_t l = 0; l < MAX_LANES; ++l) {
      if (rd.target[l] >= 0 && h_out[r * MAX_LANES + l].spins != poison) {
        spins = std::max(spins, h_out[r * MAX_LANES + l].spins);
      }
    }
    printf("round %u: R%u.%u copy -> [0x%03x, 0x%03x) from element %u, probe spins %u\n",
           r, rd.group, rd.sub, copy_lo, copy_hi, p.src_coord, spins);

    for (uint32_t l = 0; l < MAX_LANES; ++l) {
      if (rd.target[l] < 0) {
        continue;
      }
      const Target& t = targets[rd.target[l]];
      const lane_out_t& o = h_out[r * MAX_LANES + l];
      const uint32_t size = t.dword ? 8 : 4;
      const uint64_t mask = t.dword ? ~0ull : 0xFFFFFFFFull;
      const uint64_t sent = sentinel_value(r, l) & mask;
      const uint64_t mark = marker_value(r, l) & mask;

      // The word as it must read once the copy has landed over the sentinel.
      bool written = false, word_touched = false;
      uint64_t image = 0;
      for (uint32_t b = 0; b < size; ++b) {
        const uint32_t a = t.word_off + b;
        uint8_t v = (uint8_t)(sent >> (8 * b));
        if (a >= copy_lo && a < copy_hi) {
          const uint32_t sb = p.src_coord * 4 + (a - copy_lo);
          v = (uint8_t)(h_src[sb / 4] >> (8 * (sb % 4)));
          written = true;
        }
        image |= (uint64_t)v << (8 * b);
      }
      const uint32_t wbase = t.word_off & ~(kWordBytes - 1);
      word_touched = (wbase < copy_hi) && (wbase + kWordBytes > copy_lo);

      Expect expect = Expect::Succeed;
      if (!t.re_lr) {
        if (written) {
          expect = Expect::Fail;
        } else if (word_touched) {
          expect = Expect::Either;
        }
      }

      int kind = -1;
      const uint64_t old   = o.old & mask;
      const uint64_t old2  = o.old2 & mask;
      const uint64_t final = o.final & mask;
      if (o.hart == poison || o.rc == poison) {
        kind = kNoOutput;
      } else if (o.spins >= SPIN_BOUND) {
        kind = kLost;
      } else if (old != sent) {
        kind = kBadOld;
      } else if (t.re_lr && old2 != image) {
        kind = kBadReLr;
      } else if (expect == Expect::Fail) {
        if (o.rc == 0) {
          kind = kScOverCopy;
        } else if (final != image) {
          kind = kCopyLost;
        }
      } else if (expect == Expect::Succeed) {
        if (o.rc != 0) {
          kind = t.re_lr ? kReLrFail : kSpuriousFail;
        } else if (final != mark) {
          kind = kStoreLost;
        }
      } else if (final != (o.rc == 0 ? mark : image)) {
        kind = kInconsistent;
      }

      if (l == 0 && o.hart != poison && lane0_hart == UINT32_MAX) {
        lane0_hart = o.hart;
      }
      if (kind >= 0) {
        ++kind_count[kind];
      }
      char relr[32] = "";
      if (t.re_lr) {
        std::snprintf(relr, sizeof(relr), " re-lr=0x%0*llx", (int)(2 * size), (unsigned long long)old2);
      }
      printf("  R%u lane %u hart %u %-24s %s @0x%03x bank %u:%s rc=%u final=0x%0*llx expect %s -> %s\n",
             t.group, l, o.hart, t.what, t.dword ? "lr.d" : "lr.w", t.word_off,
             bank_of(t.word_off), relr, o.rc, (int)(2 * size), (unsigned long long)final,
             expect_name(expect), kind < 0 ? "ok" : kind_msg[kind]);
    }
  }

  std::cout << "hart of lane 0: " << lane0_hart << std::endl;

  uint32_t errors = 0;
  for (int k = 0; k < kNumKinds; ++k) {
    if (kind_count[k] != 0) {
      std::cout << "*** " << kind_count[k] << " x " << kind_msg[k] << std::endl;
      errors += kind_count[k];
    }
  }
  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
