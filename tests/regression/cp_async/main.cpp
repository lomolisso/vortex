// cp_async: asynchronous copies from global into local memory (PTX
// cp.async), on a DLMEM device with mbarriers (RV64).
//
//   -m zfill   one copy per case, byte-exact: 4, 8 and 16 bytes (.ca, and
//              .cg for 16) with every src-size from 0 to the copy size and
//              ignore-src false and true, the 4- and 8-byte ones also with the
//              source and destination at different offsets in their
//              doubleword; nothing outside the copy may change (default)
//   -m groups  per thread 4 groups of one copy each: after wait_group 2 the
//              first 2 groups must be visible, after wait_all all 4
//   -m bar     the H100 probe's cp.async.mbarrier.arrive modes: 0 noinc by
//              every thread completes the phase; 1 the tx form and an arrive
//              by every thread, the pending count never raised; 2 and 3 one
//              thread's noinc or tx form and an arrive on 2 or 1 expected.
//              Known difference: on the H100 a warp's 32 arrives in mode 1
//              are one update and all return 32; in Vortex each lane is its
//              own update, so the counts are T..1. The values read right
//              after the arrive in modes 2 and 3 depend on whether the copy
//              has landed: they are printed, and only checked for consistency
//   -m stream  every block streams rows of 16 bytes per thread into a double
//              buffer, each row completing on its buffer's mbarrier (noinc,
//              or the tx form and an arrive), and sums another warp's part
//   -m pipe    an RMSNorm-style double buffer with __pipeline_memcpy_async,
//              __pipeline_commit and __pipeline_wait_prior(1), the row's
//              tail zero-filled
//   -m orphan  each block, one per core at a time, exits with copies and a
//              noinc arrive in flight over its local memory; the next block
//              on those warps marks the same bytes and mbarrier, waits, and
//              must find its marks and an untouched mbarrier
//   -m mix     per row a copy and 4 global loads per thread in flight
//              together, wait_group 1 (exact instruction count)
//   -r R       stream, pipe, mix: rows (default 16)
//   -g G       stream: blocks, one per core (default: the cores); orphan:
//              blocks (default 4 per core)
//   -e E       pipe: valid elements per row (default all but 3)
//   -p         stream: no prefetch; per row wait_all and a block barrier,
//              then one try_wait and one test_wait (exact instruction count)
//   -x neg     a run that must stop the simulator: remote-dst, global-dst,
//              misaligned, srcsize, remote-mbar, local-src, encoding
//
// The host checks every byte, value and mbarrier word against the zero-fill
// rule, the H100's values and mbar_ref.

#include <vortex2.h>
#include <VX_types.h>
#include <mbar_ref.h>
#include "common.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace {

namespace mr = vortex::mbar_ref;

const char* const kModeNames[] = {"zfill", "groups", "bar", "stream", "pipe", "orphan", "mix"};
const char* const kNegNames[]  = {"", "remote-dst", "global-dst", "misaligned", "srcsize", "remote-mbar",
                                  "local-src", "encoding"};

uint32_t mode   = MODE_ZFILL;
uint32_t neg    = NEG_NONE;
uint32_t rows   = 16;
uint32_t grid   = 0;
int32_t  elems  = -1;
bool     parity = false;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

void usage() {
  std::printf("Usage: [-m zfill|groups|bar|stream|pipe|orphan|mix] [-r rows] [-g blocks] [-e elements] [-p]"
              " [-x remote-dst|global-dst|misaligned|srcsize|remote-mbar|local-src|encoding] [-h]\n");
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

uint32_t parse_name(const char* s, const char* const* names, uint32_t n, uint32_t first) {
  for (uint32_t i = first; i < n; ++i) {
    if (!std::strcmp(s, names[i]))
      return i;
  }
  usage();
  std::exit(-1);
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "m:x:r:g:e:ph")) != -1) {
    switch (c) {
      case 'm': mode   = parse_name(optarg, kModeNames, 7, 0); break;
      case 'x': neg    = parse_name(optarg, kNegNames, 8, 1); break;
      case 'r': rows   = parse_count(optarg); break;
      case 'g': grid   = parse_count(optarg); break;
      case 'e': elems  = (int32_t)parse_count(optarg); break;
      case 'p': parity = true; break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc || rows == 0) {
    usage();
    std::exit(-1);
  }
}

// The H100 probe's source pattern.
uint8_t src_byte(size_t i) {
  return (uint8_t)((i * 7 + 1) | 1);
}

// The bytes a slot must hold after its case: the copy at the destination
// offset, its tail past src-size zero, the fill elsewhere.
void zfill_expect(uint32_t c, const std::vector<uint8_t>& src, uint8_t* slot) {
  std::memset(slot, ZF_FILL, ZF_SLOT);
  const uint32_t size = ZF_SIZE(c), ss = ZF_IGN(c) ? 0 : ZF_SS(c);
  for (uint32_t j = 0; j < size; ++j)
    slot[ZF_DOFF(c) + j] = (j < ss) ? src[ZF_SOFF(c) + j] : 0;
}

// The rule against values the H100 recorded (src and dst at +16): {size, src-size, ignore-src, lo, hi} with lo/hi the 16 bytes at dst.
bool rule_matches_h100(const std::vector<uint8_t>& src) {
  struct H { uint32_t size, ss, ign; uint64_t lo, hi; };
  const H h[] = {
    {4, 0, 0, 12297829379609722880ull, 12297829382473034410ull},
    {4, 2, 0, 12297829379609753969ull, 12297829382473034410ull},
    {4, 4, 1, 12297829379609722880ull, 12297829382473034410ull},
    {8, 5, 0, 607863667057ull, 12297829382473034410ull},
    {8, 8, 0, 11789180884668610929ull, 12297829382473034410ull},
    {16, 9, 0, 11789180884668610929ull, 169ull},
    {16, 13, 0, 11789180884668610929ull, 849325044137ull},
    {16, 16, 0, 11789180884668610929ull, 15840230563600904617ull},
    {16, 16, 1, 0ull, 0ull},
  };
  for (const H& x : h) {
    uint8_t slot[ZF_SLOT];
    zfill_expect(ZF_CASE(x.size, x.ss, x.ign, 16, 16), src, slot);
    uint64_t lo, hi;
    std::memcpy(&lo, slot + 16, 8);
    std::memcpy(&hi, slot + 24, 8);
    if (lo != x.lo || hi != x.hi) {
      std::printf("*** the zero-fill rule disagrees with the H100: size %u src-size %u ignore %u\n", x.size, x.ss,
                  x.ign);
      return false;
    }
  }
  return true;
}

std::vector<uint32_t> zfill_cases() {
  std::vector<uint32_t> cases;
  const uint32_t offs4[][2] = {{16, 16}, {20, 16}, {16, 20}, {20, 20}};
  const uint32_t offs8[][2] = {{16, 16}, {24, 16}, {16, 24}, {24, 24}};
  for (const auto& o : offs4) {
    for (uint32_t ss = 0; ss <= 4; ++ss)
      cases.push_back(ZF_CASE(4, ss, 0, o[0], o[1]));
    cases.push_back(ZF_CASE(4, 4, 1, o[0], o[1]));
  }
  for (const auto& o : offs8) {
    for (uint32_t ss = 0; ss <= 8; ++ss)
      cases.push_back(ZF_CASE(8, ss, 0, o[0], o[1]));
    cases.push_back(ZF_CASE(8, 8, 1, o[0], o[1]));
  }
  for (uint32_t cg = 0; cg < 2; ++cg) {
    for (uint32_t ss = 0; ss <= 16; ++ss)
      cases.push_back(ZF_CASE(16, ss, 0, 16, 16) | (cg ? ZF_CG : 0));
    cases.push_back(ZF_CASE(16, 16, 1, 16, 16) | (cg ? ZF_CG : 0));
  }
  return cases;
}

vx_buffer_h make_buffer(vx_device_h dev, vx_queue_h q, const void* data, uint64_t bytes, uint64_t* addr) {
  vx_buffer_h b = nullptr;
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &b));
  CHECK(vx_buffer_address(b, addr));
  CHECK(vx_enqueue_write(q, b, 0, data, bytes, 0, nullptr, nullptr));
  return b;
}

uint64_t rd64(const std::vector<uint8_t>& v, size_t off) {
  uint64_t x;
  std::memcpy(&x, v.data() + off, 8);
  return x;
}

uint32_t rd32(const std::vector<uint8_t>& v, size_t off) {
  uint32_t x;
  std::memcpy(&x, v.data() + off, 4);
  return x;
}

// Replays ops on init(count) with mbar_ref; false on an error.
bool replay(uint64_t& w, const std::vector<std::pair<uint32_t, uint64_t>>& ops) {
  for (const auto& op : ops) {
    const mr::Result r = mr::apply(op.first, w, op.second);
    if (r.err)
      return false;
    w = r.new_word;
  }
  return true;
}

void add(std::vector<std::pair<uint32_t, uint64_t>>& ops, uint32_t op, uint64_t v, uint32_t n = 1) {
  for (uint32_t i = 0; i < n; ++i)
    ops.push_back({op, v});
}

}  // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  uint64_t nt = 0, nw = 0, cores = 0, max_cluster = 0, lmem_cap = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,         &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,         &cores));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE,    &lmem_cap));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &isa));
  if (!(isa & VX_ISA_EXT_DLMEM) || !(isa & VX_ISA_STD_A)) {
    std::printf("Error: the device needs DLMEM and the A extension\nFAILED!\n");
    return 1;
  }

  const bool cluster_neg = (neg == NEG_REMOTE_DST || neg == NEG_REMOTE_MBAR);
  // One block fills the core; the cluster negatives put two one-warp blocks on it.
  const uint32_t T = cluster_neg ? (uint32_t)nt : (uint32_t)(nw * nt);
  uint32_t blocks = cluster_neg ? 2 : (mode == MODE_STREAM && !neg ? (grid ? grid : (uint32_t)cores) : 1);
  if (mode == MODE_ORPHAN && !neg)
    blocks = grid ? grid : 4 * (uint32_t)cores;
  const uint32_t D = 4 * T;
  const uint32_t valid = (elems < 0) ? D - 3 : (uint32_t)elems;
  if (neg) {
    std::printf("cp_async: negative run '%s'\n", kNegNames[neg]);
  } else {
    std::printf("cp_async: %s mode%s, %u blocks of %u threads, %u rows\n", kModeNames[mode],
                parity ? " (parity)" : "", blocks, T, rows);
  }
  if ((cluster_neg && (nw < 2 || max_cluster < 2)) || valid > D) {
    std::printf("Error: the configuration does not fit the run\nFAILED!\n");
    return 1;
  }

  // Global source: the H100 pattern, large enough for every mode.
  size_t src_bytes = 64 * 1024;
  if (mode == MODE_STREAM)
    src_bytes = std::max(src_bytes, (size_t)blocks * rows * 16 * T);
  if (mode == MODE_PIPE)
    src_bytes = std::max(src_bytes, (size_t)rows * 4 * D);
  if (mode == MODE_ORPHAN)
    src_bytes = std::max(src_bytes, (size_t)blocks * OR_COPIES * T * OR_STRIDE);
  if (mode == MODE_MIX)
    src_bytes = std::max(src_bytes, (size_t)rows * 16 * T);
  std::vector<uint8_t> src(src_bytes);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = src_byte(i);

  std::vector<uint32_t> cases;
  std::vector<uint8_t> weights;
  std::vector<uint8_t> aux(64, 0);
  size_t out_bytes = 64;
  uint32_t count = 0;
  switch (mode) {
  case MODE_ZFILL:
    cases = zfill_cases();
    count = (uint32_t)cases.size();
    aux.assign((uint8_t*)cases.data(), (uint8_t*)(cases.data() + cases.size()));
    out_bytes = (size_t)count * ZF_SLOT;
    break;
  case MODE_GROUPS: out_bytes = (size_t)T * 4; break;
  case MODE_BAR:    out_bytes = (BAR_PC + T) * 8; break;
  case MODE_STREAM: count = rows; out_bytes = (size_t)blocks * rows * T * 8; break;
  case MODE_ORPHAN:
    count = 64;
    out_bytes = (size_t)blocks * OR_OUT * 8;
    aux.assign((size_t)blocks * T * OR_STORES * 64, 0);
    break;
  case MODE_MIX:
    count = rows;
    out_bytes = (size_t)rows * T * 8;
    aux.resize((size_t)rows * MIX_LOADS * T * 8);
    for (size_t i = 0; i < aux.size(); ++i)
      aux[i] = (uint8_t)(i * 29 + 3);
    break;
  default:
    count = rows;
    weights.resize(4 * D);
    for (size_t i = 0; i < weights.size(); ++i)
      weights[i] = (uint8_t)(i * 13 + 5);
    aux = weights;
    out_bytes = (size_t)rows * D * 4;
    break;
  }
  if (count > ZF_MAX && mode == MODE_ZFILL) {
    std::printf("Error: %u zfill cases, at most %u\nFAILED!\n", count, ZF_MAX);
    return 1;
  }
  size_t lmem_bytes = 4096;
  if (mode == MODE_ZFILL) lmem_bytes = (size_t)count * ZF_SLOT;
  if (mode == MODE_GROUPS) lmem_bytes = (size_t)GR_GROUPS * 16 * T;
  if (mode == MODE_BAR) lmem_bytes = 64 + BAR_MODES * 16 * T;
  if (mode == MODE_STREAM) lmem_bytes = 64 + 2 * 16 * T;
  if (mode == MODE_PIPE) lmem_bytes = 3 * 4 * D;
  // All of it, so every block takes the one slot and the same bytes.
  if (mode == MODE_ORPHAN) lmem_bytes = lmem_cap;
  if (mode == MODE_MIX) lmem_bytes = 2 * 16 * T;
  if (lmem_bytes > lmem_cap) {
    std::printf("Error: %zu bytes of local memory, the core has %lu\nFAILED!\n", lmem_bytes, (unsigned long)lmem_cap);
    return 1;
  }

  // bar and orphan count their bad elements into zeroed words.
  std::vector<uint8_t> out(out_bytes, (mode == MODE_BAR || mode == MODE_ORPHAN) ? 0 : 0x5c);
  std::vector<uint64_t> info((size_t)blocks * INFO_WORDS, 0);
  cpa_arg_t arg = {};
  vx_buffer_h src_buf  = make_buffer(dev, q, src.data(), src.size(), &arg.src_addr);
  vx_buffer_h out_buf  = make_buffer(dev, q, out.data(), out.size(), &arg.out_addr);
  vx_buffer_h aux_buf  = make_buffer(dev, q, aux.data(), aux.size(), &arg.aux_addr);
  vx_buffer_h info_buf = make_buffer(dev, q, info.data(), info.size() * 8, &arg.info_addr);
  arg.mode       = mode;
  arg.neg        = neg;
  arg.count      = count;
  arg.parity     = parity;
  arg.row_elems  = valid;
  arg.spin_limit = 1u << 12;

  vx_module_h mod = nullptr;
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kernel));

  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = blocks;
  li.grid_dim[1]  = 1;
  li.grid_dim[2]  = 1;
  li.block_dim[0] = T;
  li.block_dim[1] = 1;
  li.block_dim[2] = 1;
  li.lmem_size    = (uint32_t)lmem_bytes;
  if (cluster_neg) {
    li.cluster_dim[0] = 2;
    li.cluster_dim[1] = 1;
    li.cluster_dim[2] = 1;
    li.cluster_policy = VX_CLUSTER_POLICY_SINGLE_CORE;
  }

  vx_event_h launch_ev = nullptr, ev0 = nullptr, ev1 = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, out.data(), out_buf, 0, out.size(), 1, &launch_ev, &ev0));
  CHECK(vx_enqueue_read(q, info.data(), info_buf, 0, info.size() * 8, 1, &launch_ev, &ev1));
  CHECK(vx_event_wait_value(ev0, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev1, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev1);
  vx_event_release(ev0);
  vx_event_release(launch_ev);

  int errors = 0;
  if (neg) {
    std::printf("*** the negative run '%s' completed: the simulator must stop it\n", kNegNames[neg]);
    ++errors;
  } else {
    uint64_t spins = 0;
    for (uint32_t b = 0; b < blocks; ++b) {
      if (info[b * INFO_WORDS + INFO_DONE] != 1) {
        std::printf("*** block %u did not finish\n", b);
        ++errors;
      }
      spins += info[b * INFO_WORDS + INFO_SPINS];
    }
    if (spins >= arg.spin_limit) {
      std::printf("*** a wait gave up after %u polls\n", arg.spin_limit);
      ++errors;
    }

    if (mode == MODE_ZFILL) {
      errors += !rule_matches_h100(src);
      uint32_t bad = 0;
      for (uint32_t c = 0; c < count; ++c) {
        uint8_t want[ZF_SLOT];
        zfill_expect(cases[c], src, want);
        const uint8_t* got = out.data() + (size_t)c * ZF_SLOT;
        if (std::memcmp(got, want, ZF_SLOT) == 0)
          continue;
        if (bad++ < 8) {
          std::printf("*** case %u: size %u src-size %u ignore %u%s src +%u dst +%u:", c, ZF_SIZE(cases[c]),
                      ZF_SS(cases[c]), ZF_IGN(cases[c]), (cases[c] & ZF_CG) ? " cg" : "", ZF_SOFF(cases[c]),
                      ZF_DOFF(cases[c]));
          for (uint32_t i = 0; i < ZF_SLOT; ++i)
            if (got[i] != want[i])
              std::printf(" [%u] %02x/%02x", i, got[i], want[i]);
          std::printf("\n");
        }
      }
      std::printf("cp_async: %u zfill cases, %u wrong\n", count, bad);
      errors += (bad != 0);
    } else if (mode == MODE_GROUPS) {
      uint32_t bad = 0, early = 0;
      for (uint32_t t = 0; t < T; ++t) {
        const uint32_t seen = rd32(out, 4 * t);
        bad += (seen & 0x3) != 0x3 || (seen >> 4) != 0xf;
        early += __builtin_popcount(seen & 0xc);
      }
      std::printf("cp_async: groups: %u threads wrong; %u of %u later groups already visible after wait_group 2\n",
                  bad, early, 2 * T);
      errors += (bad != 0);
    } else if (mode == MODE_BAR) {
      // The single-thread records of the H100.
      const uint64_t h100[4][5] = {{0}, {0}, {2, 0, 1, 2, 0}, {1, 0, 1, 1, 1}};
      for (uint32_t m = 0; m < BAR_MODES; ++m) {
        const size_t o = BAR_OUT(m) * 8;
        const uint64_t bad = rd64(out, o), word = rd64(out, o + 8);
        uint64_t want = mr::init(m == 2 ? 2 : (m == 3 ? 1 : T));
        std::vector<std::pair<uint32_t, uint64_t>> ops;
        if (m == 0) {
          add(ops, mr::OP_ARRIVE, 1, T);
        } else if (m == 1) {
          add(ops, mr::OP_EXPECT_TX, 1, T);
          add(ops, mr::OP_ARRIVE, 1, T);
          add(ops, mr::OP_COMPLETE_TX, 1, T);
        } else if (m == 2) {
          add(ops, mr::OP_ARRIVE, 1, 3);
        } else {
          add(ops, mr::OP_EXPECT_TX, 1);
          add(ops, mr::OP_ARRIVE, 1);
          add(ops, mr::OP_COMPLETE_TX, 1);
          add(ops, mr::OP_ARRIVE, 1);
        }
        const bool model_ok = replay(want, ops);
        std::printf("cp_async: bar mode %u: %lu bad elements, word %#lx", m, (unsigned long)bad, (unsigned long)word);
        if (m >= 2) {
          std::printf(", records");
          for (uint32_t i = 0; i < 5; ++i)
            std::printf(" %lu", (unsigned long)rd64(out, o + 16 + 8 * i));
        }
        std::printf("\n");
        if (bad != 0 || !model_ok || word != want) {
          std::printf("*** bar mode %u: data or word wrong (mbar_ref %#lx)\n", m, (unsigned long)want);
          ++errors;
        }
        if (m >= 2) {
          // Records 0 and 1 (pending and test_wait right after the arrive)
          // depend on whether the copy has landed, except mode 3's pending:
          // the tx form never raises it. The rest hold whatever the timing.
          uint64_t rec[5];
          for (uint32_t i = 0; i < 5; ++i)
            rec[i] = rd64(out, o + 16 + 8 * i);
          bool ok = rec[2] == h100[m][2] && rec[3] == h100[m][3] && rec[4] == h100[m][4];
          if (m == 2) {
            // The noinc arrival already counted: the thread's arrive completes the phase.
            ok &= (rec[0] == 2 || rec[0] == 1) && rec[1] <= 1 && (rec[0] == 2 || rec[1] == 1);
          } else {
            ok &= rec[0] == h100[m][0] && rec[1] <= 1;
          }
          if (rec[0] != h100[m][0] || rec[1] != h100[m][1])
            std::printf("cp_async: bar mode %u: the copy landed before the arrive (the H100 read %lu %lu)\n", m,
                        (unsigned long)h100[m][0], (unsigned long)h100[m][1]);
          if (!ok) {
            std::printf("*** bar mode %u: records inconsistent with the mbarrier rules (H100: %lu %lu %lu %lu %lu)\n",
                        m, (unsigned long)h100[m][0], (unsigned long)h100[m][1], (unsigned long)h100[m][2],
                        (unsigned long)h100[m][3], (unsigned long)h100[m][4]);
            ++errors;
          }
        }
      }
      // Mode 1: each lane is its own arrive, so the counts are T..1, never
      // above T: cp.async.mbarrier.arrive does not raise pending. (On the H100
      // the warp's 32 arrives are one update and all read 32.)
      std::vector<uint32_t> pcs(T);
      for (uint32_t t = 0; t < T; ++t)
        pcs[t] = (uint32_t)rd64(out, (BAR_PC + t) * 8);
      std::sort(pcs.begin(), pcs.end());
      bool pc_ok = true;
      for (uint32_t t = 0; t < T; ++t)
        pc_ok &= (pcs[t] == t + 1);
      std::printf("cp_async: bar mode 1: pending counts %u..%u over %u threads\n", pcs[0], pcs[T - 1], T);
      if (!pc_ok) {
        std::printf("*** bar mode 1: the pending counts are not exactly 1..%u\n", T);
        ++errors;
      }
    } else if (mode == MODE_ORPHAN) {
      uint32_t bad = 0;
      const uint64_t want = mr::init(T);
      for (uint32_t b = 0; b < blocks; ++b) {
        const uint64_t marks = rd64(out, (size_t)b * OR_OUT * 8), word = rd64(out, ((size_t)b * OR_OUT + 1) * 8);
        if (marks != 0 || word != want) {
          std::printf("*** block %u: %lu marked words overwritten, mbarrier word %#lx (want %#lx)\n", b,
                      (unsigned long)marks, (unsigned long)word, (unsigned long)want);
          ++bad;
        }
      }
      std::printf("cp_async: orphan: %u of %u blocks found the previous block's copies or arrive landing late\n",
                  bad, blocks);
      errors += (bad != 0);
    } else if (mode == MODE_MIX) {
      uint32_t bad = 0;
      const uint32_t ntw = (uint32_t)nt;
      for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t t = 0; t < T; ++t) {
          const uint32_t j = (t + ntw) % T;
          uint64_t want = 0;
          for (uint32_t k = 0; k < MIX_LOADS; ++k)
            want += rd64(aux, (((size_t)r * MIX_LOADS + k) * T + t) * 8);
          const size_t s = (size_t)r * 16 * T + 16 * j;
          want += rd64(src, s) + rd64(src, s + 8);
          bad += rd64(out, ((size_t)r * T + t) * 8) != want;
        }
      }
      std::printf("cp_async: mix: %u wrong of %u\n", bad, rows * T);
      errors += (bad != 0);
    } else if (mode == MODE_STREAM) {
      uint32_t bad = 0;
      const uint32_t ntw = (uint32_t)nt;
      for (uint32_t b = 0; b < blocks; ++b) {
        for (uint32_t r = 0; r < rows; ++r) {
          for (uint32_t t = 0; t < T; ++t) {
            const uint32_t j = (t + ntw) % T;
            const size_t s = ((size_t)b * rows + r) * 16 * T + 16 * j;
            const uint64_t want = rd64(src, s) + rd64(src, s + 8);
            bad += rd64(out, (((size_t)b * rows + r) * T + t) * 8) != want;
          }
        }
        // Rows with bit 1 set use the tx form.
        for (uint32_t s = 0; s < 2; ++s) {
          uint64_t want = mr::init(T);
          std::vector<std::pair<uint32_t, uint64_t>> ops;
          for (uint32_t r = s; r < rows; r += 2) {
            if ((r >> 1) & 1) {
              add(ops, mr::OP_EXPECT_TX, 1, T);
              add(ops, mr::OP_ARRIVE, 1, T);
              add(ops, mr::OP_COMPLETE_TX, 1, T);
            } else {
              add(ops, mr::OP_ARRIVE, 1, T);
            }
          }
          const uint64_t got = info[b * INFO_WORDS + INFO_WORD0 + s];
          if (!replay(want, ops) || got != want) {
            std::printf("*** block %u FULL[%u]: word %#lx, mbar_ref %#lx\n", b, s, (unsigned long)got,
                        (unsigned long)want);
            ++errors;
          }
        }
        if (parity && info[b * INFO_WORDS + INFO_OPEN] != 0) {
          std::printf("*** block %u: %lu rows whose phase was still open after its copies landed\n", b,
                      (unsigned long)info[b * INFO_WORDS + INFO_OPEN]);
          ++errors;
        }
      }
      std::printf("cp_async: stream: %u wrong sums of %u\n", bad, blocks * rows * T);
      errors += (bad != 0);
    } else {
      uint32_t bad = 0;
      for (uint32_t r = 0; r < rows; ++r) {
        uint32_t ss = 0;
        for (uint32_t k = 0; k < valid; ++k) {
          const uint32_t x = rd32(src, ((size_t)r * D + k) * 4);
          ss += x * x;
        }
        for (uint32_t i = 0; i < D; ++i) {
          const uint32_t x = (i < valid) ? rd32(src, ((size_t)r * D + i) * 4) : 0;
          const uint32_t want = x * rd32(weights, 4 * i) + ss;
          bad += rd32(out, ((size_t)r * D + i) * 4) != want;
        }
      }
      std::printf("cp_async: pipe: %u rows of %u elements (%u valid), %u wrong\n", rows, D, valid, bad);
      errors += (bad != 0);
    }
  }

  vx_kernel_release(kernel);
  vx_module_release(mod);
  vx_buffer_release(info_buf);
  vx_buffer_release(aux_buf);
  vx_buffer_release(out_buf);
  vx_buffer_release(src_buf);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
