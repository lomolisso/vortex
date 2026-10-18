// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// DLMEM aperture translation testbench: one core's peer table and its per-block
// translation stages, checked lane by lane against dlmem_ref::translate.
//
// --mode=directed: the shared hand-computed vectors that fit this build's shape
// and core, one request each, with exact outputs.
// --mode=random: launches, each a drain, a new stride and a rewrite of every
// row, then traffic with row rewrites at random cycles. Lanes mix in-window,
// out-of-bounds and outside addresses, masked lanes and fences, under
// back-pressure phases. Every request must leave once, in order per block,
// one cycle after it enters while the output is ready, translated with the row
// current when it entered; untouched fields stay bit-equal, except the hart id
// of every local lane, which becomes the issuer's. Models of the core's two
// trackers count the remote lanes and the local plain-store lanes leaving the
// stage and drive drained and lmem_st_drained, registered like the trackers,
// with acknowledgements and bank commits (several per cycle) at random rates:
// a fence must not leave while its warp is undrained on either, and one right
// behind a remote access or a local store of its warp must be held. Each warp
// uses one block, as in the core.
// --mode=sched: random traffic into a model of the LSU scheduler's input queue
// (random acceptance and drain), filling it and handing requests over as the
// stage empties. The model is ideal, so the empty fold VX_core relies on rests
// on the exact per-cycle empty checked in every mode.
// --mode=oob: one out-of-bounds lane; a build with the bound assert must abort.
// --mode=remote: one lane to a peer on another core leaves flagged remote.
// --mode=fence: a remote store, then a remote load, each with its warp's fence
// right behind it and a later request of the block: the fence is held with
// drained low from its first cycle, the later request waits behind it, and the
// fence leaves in the cycle the completion makes drained high. Then a local
// store through the window (own rank) and a plain local-memory store, each
// held the same way on lmem_st_drained until its bank commit; a local load,
// a local AMO and a fence count nothing; a drained warp's fence is not held.
// --mode=all: directed, random, sched, then fence.

#include "vl_simulator.h"
#include "VVX_dlmem_xlate_top.h"
#include "VX_config.h"
#include "VX_types.h"
#include "dlmem_ref.h"

#if __has_include("vectors.h")
#include "vectors.h"
#define XLATE_HAVE_VECTORS 1
#else
#define XLATE_HAVE_VECTORS 0
#endif

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace vortex;
namespace ref = vortex::dlmem_ref;

///////////////////////////////////////////////////////////////////////////////

using Bits = std::vector<uint32_t>;

// Bit-field access that works on every Verilator port type (CData..QData,
// VlWide) and on Bits.
template <typename T>
static void set_bits(T& sig, uint32_t lsb, uint32_t width, uint64_t value) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    uint64_t v = static_cast<uint64_t>(sig);
    v = (v & ~(mask << lsb)) | ((value & mask) << lsb);
    sig = static_cast<T>(v);
  } else {
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      uint32_t chunk = static_cast<uint32_t>(value >> i) & m;
      sig[w] = (sig[w] & ~(m << o)) | (chunk << o);
      i += n;
    }
  }
}

template <typename T>
static uint64_t get_bits(const T& sig, uint32_t lsb, uint32_t width) {
  if constexpr (std::is_integral_v<T>) {
    uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
    return (static_cast<uint64_t>(sig) >> lsb) & mask;
  } else {
    uint64_t value = 0;
    uint32_t i = 0;
    while (i < width) {
      uint32_t bit = lsb + i;
      uint32_t w = bit / 32;
      uint32_t o = bit % 32;
      uint32_t n = std::min(32 - o, width - i);
      uint32_t m = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
      value |= static_cast<uint64_t>((sig[w] >> o) & m) << i;
      i += n;
    }
    return value;
  }
}

static Bits bits_zero(uint32_t width) {
  return Bits((width + 31) / 32, 0);
}

template <typename T>
static void put_field(T& sig, uint32_t lsb, uint32_t width, const Bits& v) {
  for (uint32_t i = 0; i < width; i += 32) {
    set_bits(sig, lsb + i, std::min(32u, width - i), v[i / 32]);
  }
}

template <typename T>
static Bits get_field(const T& sig, uint32_t lsb, uint32_t width) {
  Bits v = bits_zero(width);
  for (uint32_t i = 0; i < width; i += 32) {
    v[i / 32] = static_cast<uint32_t>(get_bits(sig, lsb + i, std::min(32u, width - i)));
  }
  return v;
}

static std::string hex(const Bits& v, uint32_t width) {
  std::string s;
  for (uint32_t w = (width + 31) / 32; w-- > 0;) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08x", v[w]);
    s += buf;
  }
  return s;
}

static std::string strf(const char* fmt, ...) {
  char buf[1024];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

static uint32_t popcount(uint64_t x) {
  return static_cast<uint32_t>(__builtin_popcountll(x));
}

static const char* kind_name(int k) {
  switch (k) {
  case ref::NONE:   return "NONE";
  case ref::LOCAL:  return "LOCAL";
  case ref::REMOTE: return "REMOTE";
  case ref::OOB:    return "OOB";
  default:          return "masked";
  }
}

///////////////////////////////////////////////////////////////////////////////

struct Options {
  std::string mode = "all";
  uint32_t seed     = 1;
  uint32_t launches = 40;
  uint32_t cycles   = 3000;   // traffic cycles per launch
  uint32_t watchdog = 2000;
  bool     oob      = true;   // generate out-of-bounds lanes
};

// A peer-table row as driven: all MAX_CLUSTER peers, valid or not.
struct TblRow {
  uint32_t k = 0;
  std::vector<ref::Peer> peers;
};

struct Request {
  uint64_t seq = 0;
  bool     rw = false;
  uint64_t mask = 0;
  bool     fence = false;
  uint32_t wid = 0;
  uint32_t cta = 0;
  uint32_t pid = 0;
  std::vector<uint32_t> byteen;
  std::vector<uint64_t> addr;   // word addresses
  std::vector<Bits>     attr;
  std::vector<uint64_t> data;
  Bits     tag;                 // raw; the top merges wid/cta/pid/fence into it

  // Captured when the stage accepts it.
  uint64_t entry_cycle = 0;
  uint32_t row_k = 0;
  uint32_t stride = 0;
  Bits     tag_eff;
  std::vector<uint64_t> exp_addr;
  std::vector<Bits>     exp_attr;
  std::vector<int>      kind;   // -1: not translated (masked lane or fence)
  uint64_t exp_remote = 0;
  uint64_t exp_oob = 0;
  uint32_t exp_lst = 0;          // local plain-store lanes
};

enum class ReadyKind { ALWAYS, RANDOM, STALL };

struct ReadyPhase {
  ReadyKind kind = ReadyKind::ALWAYS;
  uint32_t  pct = 100;
  uint32_t  left = 0;
};

enum LaneClass {
  LC_IN, LC_OOB_RANK, LC_OOB_OFF, LC_EDGE,
  LC_GLOBAL, LC_LMEM, LC_IO, LC_BELOW, LC_END, LC_PAST, LC_ALIAS, LC_FLIP,
  LC_COUNT
};

class Testbench {
public:
  explicit Testbench(const Options& opts) : opts_(opts), rng_(opts.seed) {
    auto* t = sim_.operator->();
    t->eval();
    ncc_      = t->cfg_ncc;
    nw_       = t->cfg_nw;
    maxc_     = t->cfg_max_cluster;
    lmem_log_ = t->cfg_lmem_log;
    nb_       = t->cfg_blocks;
    nl_       = t->cfg_lanes;
    aw_       = t->cfg_addr_w;
    attr_w_   = t->cfg_attr_w;
    tag_w_    = t->cfg_tag_w;
    rsp_w_    = t->cfg_rsp_w;
    pid_w_    = t->cfg_pid_w;
    nw_w_     = t->cfg_nw_w;
    k_w_      = t->cfg_k_w;
    core_w_   = t->cfg_core_w;
    hart_w_   = t->cfg_hart_w;
    hart_lsb_core_ = t->cfg_hart_core_lsb;
    nt_bits_  = t->cfg_nt_bits;
    core_id_  = t->cfg_core_id;
    cnt_w_    = ref::clog2(nl_ + 1);
    my_core_  = core_id_ % ncc_;
    rb_       = ref::clog2(maxc_);
    pid_bits_ = nt_bits_ - ref::clog2(nl_);

    Bits m_local = get_field(t->cfg_attr_local, 0, attr_w_);
    Bits m_io    = get_field(t->cfg_attr_io, 0, attr_w_);
    Bits m_hart  = get_field(t->cfg_attr_hart, 0, attr_w_);
    Bits m_remote = get_field(t->cfg_attr_remote, 0, attr_w_);
    attr_local_pos_  = this->single_bit(m_local, "is_addr_local");
    attr_io_pos_     = this->single_bit(m_io, "is_addr_io");
    attr_remote_pos_ = this->single_bit(m_remote, "is_addr_remote");
    attr_hart_lsb_  = attr_w_;
    uint32_t hart_bits = 0;
    for (uint32_t i = 0; i < attr_w_; ++i) {
      if (get_bits(m_hart, i, 1)) {
        attr_hart_lsb_ = std::min(attr_hart_lsb_, i);
        ++hart_bits;
      }
    }
    if (hart_bits != hart_w_ || get_bits(m_hart, attr_hart_lsb_, hart_w_) != ref::low_mask(hart_w_)) {
      std::printf("FATAL: hart_id attribute field is not %u contiguous bits\n", hart_w_);
      std::exit(2);
    }

    // The top's shape must be the one VX_config.h describes.
    uint32_t sockets = (VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE) > 0 ? (VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE) : 1;
    if (ncc_ != sockets * VX_CFG_SOCKET_SIZE || nw_ != VX_CFG_NUM_WARPS || maxc_ != uint32_t(VX_CFG_DLMEM_MAX_CLUSTER)
     || lmem_log_ != uint32_t(VX_CFG_LMEM_LOG_SIZE) || nl_ != uint32_t(VX_CFG_NUM_LSU_LANES)
     || aw_ != uint32_t(VX_CFG_MEM_ADDR_WIDTH) - 3 || nt_bits_ != ref::clog2(VX_CFG_NUM_THREADS)) {
      std::printf("FATAL: top shape differs from VX_config.h\n");
      std::exit(2);
    }

    cfg_.ncc         = ncc_;
    cfg_.nw          = nw_;
    cfg_.nt          = VX_CFG_NUM_THREADS;
    cfg_.max_cluster = maxc_;
    cfg_.lmem_log    = lmem_log_;
    cfg_.num_gangs   = 0;
    cfg_.socket_size = VX_CFG_SOCKET_SIZE;
    cfg_.num_sockets = sockets;
    cfg_.dlmem_base  = VX_MEM_DLMEM_BASE_ADDR;
    cfg_.lmem_base   = VX_MEM_LMEM_BASE_ADDR;
    win_bytes_       = uint64_t(1) << (lmem_log_ + rb_);

    rows_.assign(nw_, TblRow{});
    for (auto& r : rows_) {
      r.peers.assign(maxc_, ref::Peer{0, 0});
    }
    stride_ = 0;
    offer_.resize(nb_);
    queue_.resize(nb_);
    ready_.assign(nb_, true);
    ready_prev_.assign(nb_, false);
    phase_.resize(nb_);
    entered_prev_.assign(nb_, std::nullopt);
    valid_prev_.assign(nb_, false);
    rsp_valid_.assign(nb_, false);
    rsp_ready_.assign(nb_, false);
    rsp_data_.assign(nb_, bits_zero(rsp_w_));
    sched_q_.resize(nb_);
    sched_phase_.resize(nb_);
    sched_pop_.assign(nb_, false);
    fired_prev_.assign(nb_, false);
    wt_cnt_.assign(nw_, 0);
    drained_.assign(nw_, true);
    lst_cnt_.assign(nw_, 0);
    lst_drained_.assign(nw_, true);
    rem_st_prev_.assign(nb_, std::nullopt);

    t->reset = 1;
    this->drive();
    for (int i = 0; i < 8; ++i) {
      tick_ = sim_.step(tick_, 2);
    }
    t->reset = 0;
  }

  int run() {
    std::printf("dlmem_xlate: ncc=%u nw=%u max_cluster=%u lmem_log=%u blocks=%u lanes=%u core_id=%u (core %u) "
                "rank_bits=%u addr_w=%u attr_w=%u tag_w=%u mode=%s seed=%u oob=%d\n",
                ncc_, nw_, maxc_, lmem_log_, nb_, nl_, core_id_, my_core_, rb_, aw_, attr_w_, tag_w_,
                opts_.mode.c_str(), opts_.seed, opts_.oob);
    if (opts_.mode == "directed" || opts_.mode == "all") {
      this->run_directed();
    }
    if (opts_.mode == "random" || opts_.mode == "all") {
      this->run_random();
    }
    if (opts_.mode == "sched" || opts_.mode == "all") {
      this->run_sched();
    }
    if (opts_.mode == "oob") {
      this->run_oob();
    }
    if (opts_.mode == "remote") {
      this->run_remote();
    }
    if (opts_.mode == "fence" || opts_.mode == "all") {
      this->run_fence();
    }
    if (failures_ != 0) {
      std::printf("FAILED: %u check(s) failed\n", failures_);
      return 1;
    }
    std::printf("PASSED\n");
    return 0;
  }

private:
  ///////////////////////////////////////////////////////////////////////////
  // helpers

  uint32_t single_bit(const Bits& m, const char* what) {
    uint32_t pos = attr_w_, n = 0;
    for (uint32_t i = 0; i < attr_w_; ++i) {
      if (get_bits(m, i, 1)) {
        pos = i;
        ++n;
      }
    }
    if (n != 1) {
      std::printf("FATAL: attribute field %s is not one bit\n", what);
      std::exit(2);
    }
    return pos;
  }

  uint64_t rnd() {
    return rng_();
  }

  uint32_t below(uint64_t n) {
    return static_cast<uint32_t>(rng_() % n);
  }

  bool chance(uint32_t pct) {
    return below(100) < pct;
  }

  Bits rand_bits(uint32_t width) {
    Bits v = bits_zero(width);
    for (auto& w : v) {
      w = static_cast<uint32_t>(rng_());
    }
    if (width % 32) {
      v.back() &= (1u << (width % 32)) - 1;
    }
    return v;
  }

  void fail(const std::string& msg) {
    ++failures_;
    if (failures_ <= 20) {
      std::printf("FAIL [cycle %" PRIu64 "] %s\n", cycle_, msg.c_str());
    } else if (failures_ == 21) {
      std::printf("... further failure messages suppressed\n");
    }
  }

  uint32_t usable(uint32_t stride) const {
    return ref::usable_slots(cfg_, stride);
  }

  ref::Row golden_row(const TblRow& r) const {
    ref::Row g;
    g.k = r.k;
    g.peers.assign(r.peers.begin(), r.peers.begin() + r.k);
    return g;
  }

  uint64_t hart_id(uint32_t wid, uint32_t pid, uint32_t lane) const {
    uint64_t tid = (pid_bits_ != 0) ? (uint64_t(pid) * nl_ + lane) : lane;
    uint64_t h = (uint64_t(core_id_) << hart_lsb_core_) | (uint64_t(wid) << nt_bits_) | tid;
    return h & ref::low_mask(hart_w_);
  }

  ///////////////////////////////////////////////////////////////////////////
  // stimulus

  // Peers name slots that fit the launch's stride.
  TblRow gen_row(uint32_t stride) {
    TblRow r;
    uint32_t u = this->usable(stride);
    uint32_t pick = below(100);
    r.k = (pick < 10) ? 0 : (pick < 25) ? maxc_ : 1 + below(maxc_);
    r.peers.resize(maxc_);
    bool legacy = chance(20);
    uint32_t base = below(u);
    for (uint32_t i = 0; i < maxc_; ++i) {
      if (legacy) {
        r.peers[i] = {my_core_, (base + i) % u};
      } else {
        r.peers[i] = {chance(30) ? my_core_ : below(ncc_), below(u)};
      }
    }
    return r;
  }

  uint32_t gen_stride() {
    uint64_t sram = uint64_t(1) << lmem_log_;
    uint32_t pick = below(100);
    if (pick < 5)  return 0;
    if (pick < 15) return 64;
    if (pick < 25) return uint32_t(sram);
    if (pick < 40) return uint32_t(std::max<uint64_t>(64, sram / nw_));
    return uint32_t(64 * (1 + below(sram / 64)));
  }

  uint64_t word_off(uint64_t lo, uint64_t hi) {   // byte offset in [lo, hi), 8-aligned
    uint64_t n = (hi - lo) / 8;
    return lo + 8 * (rng_() % n);
  }

  uint64_t gen_lane(LaneClass c, const TblRow& row) {
    uint64_t sram = uint64_t(1) << lmem_log_;
    uint64_t ranks = uint64_t(1) << rb_;
    uint64_t base = cfg_.dlmem_base;
    if (!opts_.oob && (row.k == 0 || stride_ == 0) && (c == LC_IN || c == LC_EDGE)) {
      c = LC_GLOBAL;
    }
    switch (c) {
    case LC_OOB_RANK:
      if (row.k < ranks) {
        uint64_t rank = row.k + rng_() % (ranks - row.k);
        return base + (rank << lmem_log_) + word_off(0, sram);
      }
      [[fallthrough]];
    case LC_OOB_OFF:
      if (stride_ < sram) {
        uint64_t rank = rng_() % ranks;
        uint64_t off = chance(30) ? stride_ : word_off(stride_, sram);
        return base + (rank << lmem_log_) + off;
      }
      [[fallthrough]];
    case LC_IN: {
      if (row.k == 0 || stride_ == 0) {
        return base + word_off(0, sram);
      }
      uint64_t rank = rng_() % row.k;
      uint64_t pick = below(10);
      uint64_t off = (pick == 0) ? 0 : (pick == 1) ? stride_ - 8 : word_off(0, stride_);
      return base + (rank << lmem_log_) + off;
    }
    case LC_EDGE:
      return chance(50) ? base : base + win_bytes_ - 8;
    case LC_GLOBAL:
      return (rng_() & ref::low_mask(aw_)) << 3;
    case LC_LMEM:
      return cfg_.lmem_base + word_off(0, sram);
    case LC_IO:
      return 0x40 + 8 * below(64);
    case LC_BELOW:
      return base - 8 * (1 + below(4));
    case LC_END:
      return base + win_bytes_ + 8 * below(4);
    case LC_FLIP: {
      // One window-prefix bit flipped: every prefix bit must take part in the decode.
      uint32_t lo = lmem_log_ + rb_;
      uint32_t bit = lo + below(aw_ + 3 - lo);
      return (base ^ (uint64_t(1) << bit)) + word_off(0, win_bytes_);
    }
    case LC_ALIAS:
      // A remote encoding past the window must stay an ordinary address.
      if (ncc_ > ranks) {
        uint64_t core = ranks + rng_() % (ncc_ - ranks);
        return base + (core << lmem_log_) + word_off(0, sram);
      }
      [[fallthrough]];
    case LC_PAST:
    default:
      return base + win_bytes_ + word_off(0, uint64_t(1) << (lmem_log_ + 4));
    }
  }

  LaneClass pick_class() {
    static const uint32_t weights_oob[LC_COUNT]   = {40, 7, 7, 4, 10, 8, 3, 4, 5, 4, 6, 5};
    static const uint32_t weights_nooob[LC_COUNT] = {50, 0, 0, 0, 10, 8, 3, 4, 5, 4, 6, 5};
    const uint32_t* w = opts_.oob ? weights_oob : weights_nooob;
    uint32_t total = 0;
    for (uint32_t i = 0; i < LC_COUNT; ++i) {
      total += w[i];
    }
    uint32_t x = below(total);
    for (uint32_t i = 0; i < LC_COUNT; ++i) {
      if (x < w[i]) {
        return LaneClass(i);
      }
      x -= w[i];
    }
    return LC_GLOBAL;
  }

  // A warp of block b, as the core assigns them.
  uint32_t block_wid(uint32_t b) {
    return (b + nb_ * below((nw_ - b + nb_ - 1) / nb_)) % nw_;
  }

  Request gen_request(uint32_t b) {
    Request q;
    q.seq   = next_seq_++;
    q.cta   = below(this->usable(stride_));
    q.wid   = this->block_wid(b);
    q.pid   = below(uint64_t(1) << pid_w_);
    q.fence = chance(4);
    q.rw    = chance(40);
    uint32_t mpick = below(100);
    uint64_t full = ref::low_mask(nl_);
    q.mask = (mpick < 3) ? 0 : (mpick < 25) ? full : (rnd() & full);
    q.tag  = this->rand_bits(tag_w_);
    const TblRow& row = rows_[q.cta];
    bool uniform = chance(30);
    LaneClass uc = this->pick_class();
    for (uint32_t i = 0; i < nl_; ++i) {
      static const uint32_t sizes[] = {1, 2, 4, 8};
      uint32_t size = sizes[below(4)];
      uint32_t at = size * below(8 / size);
      q.byteen.push_back(((1u << size) - 1) << at);
      LaneClass c = uniform ? uc : this->pick_class();
      q.addr.push_back(this->gen_lane(c, row) >> 3);
      q.attr.push_back(this->rand_bits(attr_w_));
      q.data.push_back(rnd());
    }
    return q;
  }

  // The expected output, from the row and stride current when the stage accepts it.
  void expect(Request& q) {
    const TblRow& row = rows_[q.cta];
    ref::Row g = this->golden_row(row);
    uint64_t lmem_end = cfg_.lmem_base + (uint64_t(1) << lmem_log_);
    q.exp_addr = q.addr;
    q.exp_attr = q.attr;
    q.kind.assign(nl_, -1);
    q.row_k = row.k;
    q.stride = stride_;
    q.exp_remote = 0;
    q.exp_oob = 0;
    q.exp_lst = 0;
    // A local lane carries the issuer's hart id, and a local store is counted.
    auto local_lane = [&](uint32_t i) {
      set_bits(q.exp_attr[i], attr_hart_lsb_, hart_w_, this->hart_id(q.wid, q.pid, i));
      q.exp_lst += q.rw;
    };
    for (uint32_t i = 0; i < nl_; ++i) {
      set_bits(q.exp_attr[i], attr_remote_pos_, 1, 0);
      if (!((q.mask >> i) & 1) || q.fence) {
        ++n_untranslated_;
        continue;
      }
      uint64_t a = q.addr[i] << 3;
      ref::Xlate x = ref::translate(cfg_, a, g, my_core_, q.cta, stride_);
      q.kind[i] = x.kind;
      ++n_kind_[x.kind];
      if ((x.out_addr & 63) != (a & 63)) {
        this->fail(strf("golden: low six address bits changed (0x%" PRIx64 " -> 0x%" PRIx64 ")", a, x.out_addr));
      }
      if (x.kind == ref::LOCAL || x.kind == ref::OOB) {
        if (x.out_addr < cfg_.lmem_base || x.out_addr >= lmem_end) {
          this->fail(strf("golden: %s lane 0x%" PRIx64 " -> 0x%" PRIx64 " leaves the local window",
                          kind_name(x.kind), a, x.out_addr));
        }
      }
      if (x.kind == ref::REMOTE) {
        uint32_t c;
        uint64_t lb;
        ref::decode_remote(cfg_, x.out_addr, &c, &lb);
        if (c != x.core || lb != x.lmem_byte || (x.out_addr >= cfg_.lmem_base && x.out_addr < lmem_end)) {
          this->fail(strf("golden: remote 0x%" PRIx64 " does not decode to core %u offset 0x%" PRIx64,
                          x.out_addr, x.core, x.lmem_byte));
        }
      }
      uint64_t enc_end = cfg_.dlmem_base + (uint64_t(ncc_) << lmem_log_);
      if (x.kind == ref::REMOTE && x.out_addr >= cfg_.dlmem_base + win_bytes_) {
        ++n_remote_past_;
      }
      if (x.kind == ref::NONE && a >= cfg_.dlmem_base + win_bytes_ && a < enc_end) {
        ++n_alias_none_;
      }
      if (x.kind == ref::NONE) {
        if (get_bits(q.exp_attr[i], attr_local_pos_, 1)) {
          local_lane(i);
        }
        continue;
      }
      q.exp_addr[i] = x.out_addr >> 3;
      Bits& at = q.exp_attr[i];
      set_bits(at, attr_local_pos_, 1, x.kind != ref::REMOTE);
      set_bits(at, attr_io_pos_, 1, 0);
      if (x.kind == ref::REMOTE) {
        set_bits(at, attr_hart_lsb_, hart_w_, this->hart_id(q.wid, q.pid, i));
        set_bits(at, attr_remote_pos_, 1, 1);
        q.exp_remote |= uint64_t(1) << i;
      }
      if (x.kind == ref::OOB) {
        q.exp_oob |= uint64_t(1) << i;
      }
      if (x.kind != ref::REMOTE) {
        local_lane(i);
      }
    }
  }

  ///////////////////////////////////////////////////////////////////////////
  // DUT access

  void drive() {
    auto* t = sim_.operator->();
    set_bits(t->wr_valid, 0, 1, wr_.has_value());
    if (wr_) {
      set_bits(t->wr_slot, 0, nw_w_, wr_->first);
      set_bits(t->wr_k, 0, k_w_, wr_->second.k);
      for (uint32_t r = 0; r < maxc_; ++r) {
        set_bits(t->wr_core, r * core_w_, core_w_, wr_->second.peers[r].core);
        set_bits(t->wr_pslot, r * nw_w_, nw_w_, wr_->second.peers[r].slot);
      }
      set_bits(t->wr_stride, 0, lmem_log_ + 1, wr_stride_);
    }
    for (uint32_t w = 0; w < nw_; ++w) {
      set_bits(t->drained, w, 1, drained_[w]);
      set_bits(t->lmem_st_drained, w, 1, lst_drained_[w]);
    }
    for (uint32_t b = 0; b < nb_; ++b) {
      const auto& o = offer_[b];
      set_bits(t->in_valid, b, 1, o.has_value());
      set_bits(t->out_ready, b, 1, ready_[b]);
      set_bits(t->mem_rsp_valid, b, 1, rsp_valid_[b]);
      put_field(t->mem_rsp_data, b * rsp_w_, rsp_w_, rsp_data_[b]);
      set_bits(t->core_rsp_ready, b, 1, rsp_ready_[b]);
      if (!o) {
        continue;
      }
      set_bits(t->in_rw, b, 1, o->rw);
      set_bits(t->in_mask, b * nl_, nl_, o->mask);
      set_bits(t->in_fence, b, 1, o->fence);
      set_bits(t->in_wid, b * nw_w_, nw_w_, o->wid);
      set_bits(t->in_cta, b * nw_w_, nw_w_, o->cta);
      set_bits(t->in_pid, b * pid_w_, pid_w_, o->pid);
      put_field(t->in_tag, b * tag_w_, tag_w_, o->tag);
      for (uint32_t i = 0; i < nl_; ++i) {
        uint32_t l = b * nl_ + i;
        set_bits(t->in_byteen, l * 8, 8, o->byteen[i]);
        set_bits(t->in_addr, l * aw_, aw_, o->addr[i]);
        put_field(t->in_attr, l * attr_w_, attr_w_, o->attr[i]);
        set_bits(t->in_data, l * 64, 64, o->data[i]);
      }
    }
  }

  void check_output(uint32_t b, const Request& q) {
    auto* t = sim_.operator->();
    std::string id = strf("block %u seq %" PRIu64, b, q.seq);
    if (get_bits(t->out_rw, b, 1) != q.rw) {
      this->fail(id + ": rw changed");
    }
    if (get_bits(t->out_mask, b * nl_, nl_) != q.mask) {
      this->fail(id + ": mask changed");
    }
    Bits tag = get_field(t->out_tag, b * tag_w_, tag_w_);
    if (tag != q.tag_eff) {
      this->fail(strf("%s: tag changed\n  expected %s\n  got      %s", id.c_str(),
                      hex(q.tag_eff, tag_w_).c_str(), hex(tag, tag_w_).c_str()));
    }
    uint64_t remote = get_bits(t->out_remote, b * nl_, nl_);
    uint64_t oob = get_bits(t->out_oob, b * nl_, nl_);
    if (remote != q.exp_remote || oob != q.exp_oob) {
      this->fail(strf("%s: remote/oob flags 0x%" PRIx64 "/0x%" PRIx64 ", expected 0x%" PRIx64 "/0x%" PRIx64,
                      id.c_str(), remote, oob, q.exp_remote, q.exp_oob));
    }
    for (uint32_t i = 0; i < nl_; ++i) {
      uint32_t l = b * nl_ + i;
      uint64_t addr = get_bits(t->out_addr, l * aw_, aw_);
      if (addr != q.exp_addr[i]) {
        this->fail(strf("%s lane %u (%s): addr 0x%" PRIx64 " -> 0x%" PRIx64 ", expected 0x%" PRIx64
                        " (cta %u, k %u, stride %u)",
                        id.c_str(), i, kind_name(q.kind[i]), q.addr[i] << 3, addr << 3, q.exp_addr[i] << 3,
                        q.cta, q.row_k, q.stride));
      }
      if ((addr & 7) != (q.addr[i] & 7)) {
        this->fail(strf("%s lane %u: low six address bits changed", id.c_str(), i));
      }
      Bits attr = get_field(t->out_attr, l * attr_w_, attr_w_);
      if (attr != q.exp_attr[i]) {
        this->fail(strf("%s lane %u (%s): attr\n  expected %s\n  got      %s", id.c_str(), i,
                        kind_name(q.kind[i]), hex(q.exp_attr[i], attr_w_).c_str(), hex(attr, attr_w_).c_str()));
      }
      if (get_bits(t->out_byteen, l * 8, 8) != q.byteen[i]) {
        this->fail(strf("%s lane %u: byteen changed", id.c_str(), i));
      }
      if (get_bits(t->out_data, l * 64, 64) != q.data[i]) {
        this->fail(strf("%s lane %u: data changed", id.c_str(), i));
      }
    }
  }

  // One cycle: drive, sample every handshake before the edge, update the table model, clock.
  void clock() {
    auto* t = sim_.operator->();
    for (uint32_t b = 0; b < nb_; ++b) {
      rsp_valid_[b] = chance(50);
      rsp_ready_[b] = chance(50);
      rsp_data_[b]  = this->rand_bits(rsp_w_);
    }
    // The tracker takes at most one acknowledgement per cycle.
    std::optional<uint32_t> ack;
    if (chance(ack_pct_)) {
      std::vector<uint32_t> pend;
      for (uint32_t w = 0; w < nw_; ++w) {
        if (wt_cnt_[w] != 0) {
          pend.push_back(w);
        }
      }
      if (!pend.empty()) {
        ack = pend[below(pend.size())];
      }
    }
    std::vector<uint64_t> wt_inc(nw_, 0);
    // Up to one bank commit per local-memory bank per cycle.
    std::vector<uint64_t> lst_inc(nw_, 0), lst_dec(nw_, 0);
    for (uint32_t k = 0; k < VX_CFG_LMEM_NUM_BANKS; ++k) {
      if (!chance(commit_pct_)) {
        continue;
      }
      std::vector<uint32_t> pend;
      for (uint32_t w = 0; w < nw_; ++w) {
        if (lst_cnt_[w] > lst_dec[w]) {
          pend.push_back(w);
        }
      }
      if (!pend.empty()) {
        ++lst_dec[pend[below(pend.size())]];
      }
    }
    this->drive();
    t->eval();

    bool progress = false;

    // The table as written up to the previous cycle.
    if (get_bits(t->tbl_stride, 0, lmem_log_ + 1) != stride_) {
      this->fail(strf("table stride %" PRIu64 ", expected %u", get_bits(t->tbl_stride, 0, lmem_log_ + 1), stride_));
    }
    for (uint32_t s = 0; s < nw_; ++s) {
      uint64_t want = (uint64_t(s) * stride_) & ref::low_mask(lmem_log_);
      uint64_t got = get_bits(t->tbl_slot_base, s * lmem_log_, lmem_log_);
      if (got != want) {
        this->fail(strf("slot_base[%u] = 0x%" PRIx64 ", expected 0x%" PRIx64, s, got, want));
      }
    }

    for (uint32_t b = 0; b < nb_; ++b) {
      auto& q = queue_[b];
      bool empty = get_bits(t->empty, b, 1);
      if (empty != q.empty()) {
        this->fail(strf("block %u: empty=%d with %zu request(s) in the stage", b, empty, q.size()));
      }
      if (sched_) {
        bool sched_empty = sched_q_[b].empty();
        n_fold_busy_ += !(q.empty() && sched_empty);
        n_handoff_ += empty && !sched_empty && fired_prev_[b];
      }

      // Responses are wires.
      if (get_bits(t->core_rsp_valid, b, 1) != rsp_valid_[b]
       || get_field(t->core_rsp_data, b * rsp_w_, rsp_w_) != rsp_data_[b]
       || get_bits(t->mem_rsp_ready, b, 1) != rsp_ready_[b]) {
        this->fail(strf("block %u: response path is not a passthrough", b));
      }

      bool valid = get_bits(t->out_valid, b, 1);
      bool held = !q.empty() && q.front().fence && !(drained_[q.front().wid] && lst_drained_[q.front().wid]);
      if (valid_prev_[b] && !ready_prev_[b] && !valid) {
        this->fail(strf("block %u: output dropped while stalled", b));
      }
      if (entered_prev_[b] && ready_prev_[b]) {
        // Accepted last cycle while the output drained: it is the head now.
        if (valid == held || q.empty() || q.front().seq != *entered_prev_[b]) {
          this->fail(strf("block %u: seq %" PRIu64 " is not at the output one cycle after entering", b,
                          *entered_prev_[b]));
        }
      }
      if (held) {
        const Request& h = q.front();
        if (valid) {
          this->fail(strf("block %u seq %" PRIu64 ": fence of warp %u left with the warp undrained", b, h.seq, h.wid));
        }
        ++n_held_cycles_;
        auto& fo = fobs_[h.seq];
        if (fo.held_cycles++ == 0) {
          ++n_fence_held_;
        }
      }
      if (!q.empty() && q.front().fence && rem_st_prev_[b] == q.front().wid) {
        // The fence right behind its warp's remote access sees that access counted.
        if (!held) {
          this->fail(strf("block %u seq %" PRIu64 ": fence behind a remote access of warp %u not held", b,
                          q.front().seq, q.front().wid));
        }
        ++n_b2b_;
      }
      if (!q.empty() && q.front().fence && lst_prev_[b] == q.front().wid) {
        // The same for a local store.
        if (!held) {
          this->fail(strf("block %u seq %" PRIu64 ": fence behind a local store of warp %u not held", b,
                          q.front().seq, q.front().wid));
        }
        ++n_b2b_lst_;
      }
      rem_st_prev_[b].reset();
      lst_prev_[b].reset();
      bool fire = false;
      if (valid) {
        if (q.empty()) {
          this->fail(strf("block %u: output valid with nothing in the stage", b));
        } else {
          this->check_output(b, q.front());
          fire = ready_[b];
        }
      }
      bool wt_valid = get_bits(t->wt_inc_valid, b, 1);
      bool lst_valid = get_bits(t->lst_inc_valid, b, 1);
      if (fire) {
        const Request& r = q.front();
        bool want = (r.exp_remote != 0);
        if (wt_valid != want) {
          this->fail(strf("block %u seq %" PRIu64 ": wt_inc_valid=%d, expected %d", b, r.seq, wt_valid, want));
        }
        if (want) {
          uint64_t wid = get_bits(t->wt_inc_wid, b * nw_w_, nw_w_);
          uint64_t cnt = get_bits(t->wt_inc_cnt, b * cnt_w_, cnt_w_);
          if (wid != r.wid || cnt != popcount(r.exp_remote)) {
            this->fail(strf("block %u seq %" PRIu64 ": wt_inc wid %" PRIu64 " cnt %" PRIu64 ", expected %u %u",
                            b, r.seq, wid, cnt, r.wid, popcount(r.exp_remote)));
          }
          ++n_wt_inc_;
          (r.rw ? n_wt_stores_ : n_wt_reads_) += popcount(r.exp_remote);
          wt_inc[r.wid] += popcount(r.exp_remote);
          rem_st_prev_[b] = r.wid;
        }
        bool want_lst = (r.exp_lst != 0);
        if (lst_valid != want_lst) {
          this->fail(strf("block %u seq %" PRIu64 ": lst_inc_valid=%d, expected %d", b, r.seq, lst_valid, want_lst));
        }
        if (want_lst) {
          uint64_t wid = get_bits(t->lst_inc_wid, b * nw_w_, nw_w_);
          uint64_t cnt = get_bits(t->lst_inc_cnt, b * cnt_w_, cnt_w_);
          if (wid != r.wid || cnt != r.exp_lst) {
            this->fail(strf("block %u seq %" PRIu64 ": lst_inc wid %" PRIu64 " cnt %" PRIu64 ", expected %u %u",
                            b, r.seq, wid, cnt, r.wid, r.exp_lst));
          }
          ++n_lst_inc_;
          n_lst_lanes_ += r.exp_lst;
          lst_inc[r.wid] += r.exp_lst;
          lst_prev_[b] = r.wid;
        }
        if (r.fence) {
          fobs_[r.seq].fire_cycle = cycle_;
        }
        if (lane0_) {
          last_addr0_   = get_bits(t->out_addr, b * nl_ * aw_, aw_);
          last_remote0_ = get_bits(t->out_remote, b * nl_, 1);
          last_oob0_    = get_bits(t->out_oob, b * nl_, 1);
        }
        if (sched_) {
          if (sched_q_[b].size() >= sched_cap_) {
            this->fail(strf("block %u: output fired into a full scheduler queue", b));
          }
          sched_push_.push_back(b);
        }
        q.pop_front();
        ++n_out_;
        progress = true;
      } else if (wt_valid || lst_valid) {
        this->fail(strf("block %u: wt_inc_valid or lst_inc_valid without an output handshake", b));
      }

      // Full throughput: a stage drained last cycle accepts this cycle.
      bool in_ready = get_bits(t->in_ready, b, 1);
      if (ready_prev_[b] && !in_ready) {
        this->fail(strf("block %u: not ready after a cycle with the output ready", b));
      }
      auto& o = offer_[b];
      entered_prev_[b].reset();
      if (o) {
        if (get_bits(t->row_slot, b * nw_w_, nw_w_) != o->cta) {
          this->fail(strf("block %u: row_slot is not the request's cta_id", b));
        }
        if (in_ready) {
          o->entry_cycle = cycle_;
          o->tag_eff = get_field(t->in_tag_eff, b * tag_w_, tag_w_);
          if (get_bits(o->tag_eff, 0, 1) != o->fence) {
            this->fail(strf("block %u: merged tag lost the fence bit", b));
          }
          this->expect(*o);
          ++n_in_;
          n_fence_ += o->fence;
          entered_prev_[b] = o->seq;
          q.push_back(*o);
          max_occ_ = std::max<size_t>(max_occ_, q.size());
          o.reset();
          progress = true;
        }
      }
      if (q.size() > 2) {
        this->fail(strf("block %u: %zu requests in a two-entry stage", b, q.size()));
      }
      valid_prev_[b] = valid && !fire;
      fired_prev_[b] = fire;
      // What the stage's buffer sees: a held fence keeps it from draining.
      ready_prev_[b] = ready_[b] && !held;
      n_stall_ += valid && !ready_[b];
    }

    // The scheduler queue forwards its head and takes the stage's output at the edge.
    for (uint32_t b = 0; b < nb_ && sched_; ++b) {
      if (sched_pop_[b] && !sched_q_[b].empty()) {
        sched_q_[b].pop_front();
        progress = true;
      }
    }
    for (uint32_t b : sched_push_) {
      sched_q_[b].push_back(cycle_);
      sched_max_ = std::max<size_t>(sched_max_, sched_q_[b].size());
    }
    sched_push_.clear();

    // The tracker's count and its registered drained update at the edge.
    for (uint32_t w = 0; w < nw_; ++w) {
      wt_cnt_[w] += wt_inc[w];
      if (ack == w) {
        --wt_cnt_[w];
        ++n_acks_;
        progress = true;
      }
      drained_[w] = (wt_cnt_[w] == 0);
      if (!lst_drained_[w] && lst_dec[w] == lst_cnt_[w] && lst_inc[w] == 0) {
        lst_rise_[w] = cycle_ + 1;
      }
      lst_cnt_[w] += lst_inc[w];
      lst_cnt_[w] -= lst_dec[w];
      n_commits_ += lst_dec[w];
      progress |= (lst_dec[w] != 0);
      lst_drained_[w] = (lst_cnt_[w] == 0);
    }

    // A write lands at the edge: visible to requests from the next cycle.
    if (wr_) {
      rows_[wr_->first] = wr_->second;
      stride_ = wr_stride_;
      ++n_writes_;
      wr_.reset();
    }

    tick_ = sim_.step(tick_, 2);
    ++cycle_;
    if (progress) {
      last_progress_ = cycle_;
    }
  }

  bool busy() const {
    for (uint32_t b = 0; b < nb_; ++b) {
      if (offer_[b] || !queue_[b].empty() || !sched_q_[b].empty()) {
        return true;
      }
    }
    return false;
  }

  bool watchdog() {
    if (this->busy() && cycle_ - last_progress_ > opts_.watchdog) {
      this->fail(strf("no progress for %u cycles", opts_.watchdog));
      return true;
    }
    return false;
  }

  // The scheduler model accepts while its queue has room, at a random rate,
  // and drains at another, each held for a random phase.
  void sched_ready(uint32_t b) {
    auto& p = sched_phase_[b];
    if (p.left == 0) {
      static const uint32_t accept[] = {100, 100, 75, 50};
      static const uint32_t drain[]  = {0, 25, 50, 100};
      p.accept = accept[below(4)];
      p.drain  = drain[below(4)];
      p.left   = (p.drain == 0) ? 5 + below(40) : 50 + below(300);
    }
    --p.left;
    ready_[b] = sched_q_[b].size() < sched_cap_ && chance(p.accept);
    sched_pop_[b] = chance(p.drain);
  }

  void next_ready(uint32_t b) {
    if (sched_) {
      this->sched_ready(b);
      return;
    }
    auto& p = phase_[b];
    if (p.left == 0) {
      uint32_t pick = below(100);
      if (pick < 35) {
        p = {ReadyKind::ALWAYS, 100, 50 + below(300)};
      } else if (pick < 80) {
        static const uint32_t pcts[] = {25, 50, 75, 90};
        p = {ReadyKind::RANDOM, pcts[below(4)], 50 + below(300)};
      } else {
        p = {ReadyKind::STALL, 0, 10 + below(60)};
      }
    }
    --p.left;
    ready_[b] = (p.kind == ReadyKind::ALWAYS) || (p.kind == ReadyKind::RANDOM && chance(p.pct));
  }

  ///////////////////////////////////////////////////////////////////////////
  // modes

  // Until the stages are empty and every counted access has completed.
  void drain() {
    ack_pct_ = 100;
    commit_pct_ = 100;
    auto undrained = [&]() {
      return std::any_of(wt_cnt_.begin(), wt_cnt_.end(), [](auto n) { return n != 0; })
          || std::any_of(drained_.begin(), drained_.end(), [](bool d) { return !d; })
          || std::any_of(lst_cnt_.begin(), lst_cnt_.end(), [](auto n) { return n != 0; })
          || std::any_of(lst_drained_.begin(), lst_drained_.end(), [](bool d) { return !d; });
    };
    while ((this->busy() || undrained()) && !this->watchdog()) {
      for (uint32_t b = 0; b < nb_; ++b) {
        this->next_ready(b);
      }
      this->clock();
    }
  }

  void write_all_rows(uint32_t stride) {
    for (uint32_t s = 0; s < nw_; ++s) {
      wr_stride_ = stride;
      wr_ = std::make_pair(s, this->gen_row(stride));
      for (uint32_t b = 0; b < nb_; ++b) {
        this->next_ready(b);
      }
      this->clock();
    }
  }

  void run_random() {
    uint32_t offer_pct = 100;
    for (uint32_t l = 0; l < opts_.launches && failures_ == 0; ++l) {
      this->drain();
      this->write_all_rows(this->gen_stride());
      n_strides_.push_back(stride_);
      for (uint32_t c = 0; c < opts_.cycles; ++c) {
        if (c % 200 == 0) {
          static const uint32_t pcts[] = {100, 100, 60, 25};
          offer_pct = pcts[below(4)];
          static const uint32_t acks[] = {0, 10, 50, 100};
          ack_pct_ = acks[below(4)];
          commit_pct_ = acks[below(4)];
        }
        for (uint32_t b = 0; b < nb_; ++b) {
          this->next_ready(b);
          if (!offer_[b] && chance(offer_pct)) {
            offer_[b] = this->gen_request(b);
          }
        }
        // A rewrite between generation and entry could make an in-bounds lane out of bounds.
        if (opts_.oob && chance(10)) {
          wr_stride_ = stride_;
          wr_ = std::make_pair(below(nw_), this->gen_row(stride_));
        }
        this->clock();
        if (this->watchdog()) {
          break;
        }
      }
    }
    this->drain();
    std::printf("random: %" PRIu64 " cycles, %" PRIu64 " requests in, %" PRIu64 " out, %" PRIu64 " fences, "
                "%" PRIu64 " row writes, %" PRIu64 " stalled output cycles, max occupancy %zu\n",
                cycle_, n_in_, n_out_, n_fence_, n_writes_, n_stall_, max_occ_);
    std::printf("random: %" PRIu64 " fences held for %" PRIu64 " cycles, %" PRIu64 " right behind their warp's "
                "remote access, %" PRIu64 " acknowledgements\n", n_fence_held_, n_held_cycles_, n_b2b_, n_acks_);
    std::printf("random: lanes NONE %" PRIu64 ", LOCAL %" PRIu64 ", REMOTE %" PRIu64 ", OOB %" PRIu64
                ", untranslated %" PRIu64 "; remote-access events %" PRIu64 " (%" PRIu64 " store, %" PRIu64 " load/AMO lanes)\n",
                n_kind_[ref::NONE], n_kind_[ref::LOCAL], n_kind_[ref::REMOTE], n_kind_[ref::OOB],
                n_untranslated_, n_wt_inc_, n_wt_stores_, n_wt_reads_);
    std::printf("random: remote lanes encoded past the window %" PRIu64 ", outside lanes inside the remote "
                "encoding range %" PRIu64 "\n", n_remote_past_, n_alias_none_);
    std::printf("random: %" PRIu64 " local-store events (%" PRIu64 " lanes), %" PRIu64 " bank commits, %" PRIu64
                " fences right behind their warp's local store\n", n_lst_inc_, n_lst_lanes_, n_commits_, n_b2b_lst_);
    std::string s;
    for (uint32_t v : n_strides_) {
      s += strf(" %u", v);
    }
    std::printf("random: strides%s\n", s.c_str());
    if (n_in_ != n_out_) {
      this->fail(strf("%" PRIu64 " requests entered, %" PRIu64 " left", n_in_, n_out_));
    }
    if (n_kind_[ref::LOCAL] == 0 || n_kind_[ref::NONE] == 0 || (opts_.oob && n_kind_[ref::OOB] == 0)
     || (ncc_ > 1 && n_kind_[ref::REMOTE] == 0) || (ncc_ > 1 && (n_wt_stores_ == 0 || n_wt_reads_ == 0))
     || (ncc_ > (1u << rb_) && (n_remote_past_ == 0 || n_alias_none_ == 0))) {
      this->fail("random traffic missed a lane kind");
    }
    if (ncc_ > 1 && (n_fence_held_ == 0 || n_b2b_ == 0)) {
      this->fail("random traffic never held a fence, or never put one right behind its warp's remote access");
    }
    if (n_lst_lanes_ == 0 || n_b2b_lst_ == 0 || n_fence_held_ == 0) {
      this->fail("random traffic never counted a local store, or never held a fence right behind one");
    }
  }

  // One request, lane 0 only, through an always-ready stage.
  void one_request(uint32_t cta, uint64_t byte_addr) {
    Request q;
    q.seq = next_seq_++;
    q.cta = cta;
    q.wid = this->block_wid(0);
    q.pid = below(uint64_t(1) << pid_w_);
    q.rw = chance(50);
    q.mask = 1;
    q.tag = this->rand_bits(tag_w_);
    for (uint32_t i = 0; i < nl_; ++i) {
      q.byteen.push_back(0xff);
      q.addr.push_back(i == 0 ? (byte_addr >> 3) : (rnd() & ref::low_mask(aw_)));
      q.attr.push_back(this->rand_bits(attr_w_));
      q.data.push_back(rnd());
    }
    offer_[0] = q;
    ready_.assign(nb_, true);
    for (uint32_t c = 0; c < 4; ++c) {
      this->clock();
    }
    if (this->busy()) {
      this->fail("directed request did not leave the stage");
    }
  }

  void run_directed() {
#if XLATE_HAVE_VECTORS
    using namespace dlmem_xlate_vec;
    if (kDlmemBase != cfg_.dlmem_base || kLmemBase != cfg_.lmem_base) {
      this->fail("the vector table's DLMEM/LMEM bases differ from VX_types.h");
      return;
    }
    uint32_t ran = 0, skipped = 0;
    lane0_ = true;
    for (uint32_t n = 0; n < kNumVectors; ++n) {
      const Vector& v = kVectors[n];
      const Shape& sh = kShapes[v.shape];
      ref::Cfg vc = cfg_;
      vc.ncc         = sh.ncc;
      vc.nw          = sh.nw;
      vc.max_cluster = sh.max_cluster;
      vc.lmem_log    = sh.lmem_log;
      ref::Row row;
      row.k = v.k;
      for (uint32_t r = 0; r < v.k; ++r) {
        row.peers.push_back({v.peers[r].core, v.peers[r].slot});
      }
      // The hand-computed table must agree with the golden model, whatever the build.
      ref::Xlate g = ref::translate(vc, v.addr, row, sh.my_core, v.my_slot, v.stride);
      if (uint32_t(g.kind) != v.kind || g.out_addr != v.out_addr || (v.kind != K_NONE && g.core != v.core)) {
        this->fail(strf("vector %s: dlmem_ref gives %s core %u 0x%" PRIx64 ", the table %s core %u 0x%" PRIx64,
                        v.name, kind_name(g.kind), g.core, g.out_addr, kind_name(v.kind), v.core, v.out_addr));
      }
      bool fits = sh.ncc == ncc_ && sh.max_cluster == maxc_ && sh.lmem_log == lmem_log_ && sh.my_core == my_core_
               && v.my_slot < nw_;
      for (const auto& p : row.peers) {
        fits = fits && p.slot < nw_;
      }
      if (!fits) {
        ++skipped;
        continue;
      }
      TblRow tr;
      tr.k = v.k;
      for (uint32_t r = 0; r < maxc_; ++r) {
        tr.peers.push_back(r < v.k ? ref::Peer{v.peers[r].core, v.peers[r].slot}
                                   : ref::Peer{below(ncc_), below(nw_)});
      }
      wr_stride_ = v.stride;
      wr_ = std::make_pair(v.my_slot, tr);
      ready_.assign(nb_, true);
      this->clock();
      uint32_t before = failures_;
      uint64_t outs = n_out_;
      this->one_request(v.my_slot, v.addr);
      if (n_out_ != outs + 1) {
        this->fail(strf("vector %s: request did not leave the stage", v.name));
      } else if (last_addr0_ != (v.out_addr >> 3) || last_remote0_ != (v.kind == K_REMOTE)
              || last_oob0_ != (v.kind == K_OOB)) {
        this->fail(strf("vector %s: got 0x%" PRIx64 " remote=%d oob=%d, expected %s 0x%" PRIx64,
                        v.name, last_addr0_ << 3, last_remote0_, last_oob0_, kind_name(v.kind), v.out_addr));
      }
      std::printf("  vector %-26s %-6s 0x%09" PRIx64 " -> 0x%09" PRIx64 " %s\n", v.name, kind_name(v.kind),
                  v.addr, v.out_addr, (failures_ == before) ? "ok" : "FAIL");
      ++ran;
    }
    lane0_ = false;
    std::printf("directed: %u vector(s) ran, %u belong to other shapes or cores\n", ran, skipped);
    if (ran == 0) {
      this->fail("no directed vector fits this build");
    }
#else
    this->fail("directed mode needs tests/unittest/dlmem_xlate/vectors.h");
#endif
  }

  void run_sched() {
    sched_ = true;
    uint64_t in0 = n_in_;
    this->run_random();
    sched_ = false;
    std::printf("sched: queue capacity %u, %" PRIu64 " requests, %" PRIu64 " cycles with a request in the stage or "
                "the queue, %" PRIu64 " handoffs into a non-empty queue with the stage empty, max queue %zu\n",
                sched_cap_, n_in_ - in0, n_fold_busy_, n_handoff_, sched_max_);
    if (n_handoff_ == 0 || sched_max_ < sched_cap_) {
      this->fail("scheduler model never saw a handoff or never filled");
    }
  }

  void run_oob() {
    TblRow r;
    r.k = 1;
    r.peers.assign(maxc_, ref::Peer{my_core_, 0});
    wr_stride_ = 64;
    wr_ = std::make_pair(0u, r);
    this->clock();
    std::printf("oob: issuing an out-of-bounds lane; a bound-checking build must abort here\n");
    std::fflush(stdout);
    this->one_request(0, cfg_.dlmem_base + 64);
  }

  void run_remote() {
    if (ncc_ < 2 || maxc_ < 2) {
      this->fail("remote mode needs a cluster of at least two cores");
      return;
    }
    TblRow r;
    r.k = 2;
    r.peers.assign(maxc_, ref::Peer{my_core_, 0});
    r.peers[1] = ref::Peer{(my_core_ + 1) % ncc_, 1};
    wr_stride_ = 64;
    wr_ = std::make_pair(0u, r);
    this->clock();
    std::printf("remote: issuing a lane to rank 1 on core %u\n", r.peers[1].core);
    std::fflush(stdout);
    lane0_ = true;
    this->one_request(0, cfg_.dlmem_base + (uint64_t(1) << lmem_log_) + 8);
    lane0_ = false;
    if (!last_remote0_) {
      this->fail("remote: the lane was not flagged remote");
    }
  }

  Request fence_req(uint32_t wid, bool fence, bool rw, uint64_t byte_addr) {
    Request q;
    q.seq   = next_seq_++;
    q.cta   = 0;
    q.wid   = wid;
    q.pid   = 0;
    q.fence = fence;
    q.rw    = rw;
    q.mask  = 1;
    q.tag   = this->rand_bits(tag_w_);
    for (uint32_t i = 0; i < nl_; ++i) {
      q.byteen.push_back(0xff);
      q.addr.push_back(byte_addr >> 3);
      q.attr.push_back(this->rand_bits(attr_w_));
      q.data.push_back(rnd());
    }
    return q;
  }

  // Offers each request on block 0 as soon as the previous one is taken, output always ready.
  void offer_all(std::deque<Request> reqs, uint32_t cycles) {
    ready_.assign(nb_, true);
    for (uint32_t c = 0; c < cycles; ++c) {
      if (!offer_[0] && !reqs.empty()) {
        offer_[0] = reqs.front();
        reqs.pop_front();
      }
      this->clock();
    }
  }

  void run_fence() {
    this->drain();
    const bool two = (ncc_ >= 2 && maxc_ >= 2);
    TblRow r;
    r.k = two ? 2 : 1;
    r.peers.assign(maxc_, ref::Peer{my_core_, 0});
    if (two) {
      r.peers[1] = ref::Peer{(my_core_ + 1) % ncc_, 1};
    }
    wr_stride_ = 64;
    wr_ = std::make_pair(0u, r);
    this->clock();
    uint64_t remote = cfg_.dlmem_base + (uint64_t(1) << lmem_log_) + 8;
    uint64_t local  = cfg_.dlmem_base + 16;
    uint32_t w  = 0;
    uint32_t w2 = (nw_ > nb_) ? nb_ : 0;
    const uint32_t hold = 20;

    // A local store through the window (own rank), then a plain local-memory
    // store: each with its warp's fence and a later local load behind it,
    // every bank commit withheld.
    for (bool plain : {false, true}) {
      const char* what = plain ? "plain local-memory store" : "window store to the own rank";
      ack_pct_ = 100;
      commit_pct_ = 0;
      Request st = this->fence_req(w, false, true, plain ? cfg_.lmem_base + 16 : local);
      if (plain) {
        for (auto& at : st.attr) {
          set_bits(at, attr_local_pos_, 1, 1);
        }
      }
      Request fe = this->fence_req(w, true, false, 0);
      Request ld = this->fence_req(w2, false, false, local);
      uint64_t c0 = cycle_;
      this->offer_all({st, fe, ld}, hold);
      const FenceObs& fo = fobs_[fe.seq];
      if (fo.fire_cycle != ~uint64_t(0) || fo.held_cycles + 3 < hold || queue_[0].size() != 2
       || queue_[0].front().seq != fe.seq) {
        this->fail(strf("fence after a %s: held %" PRIu64 " of %u cycles, fired at %" PRId64
                        ", %zu request(s) in the stage", what, fo.held_cycles, hold, int64_t(fo.fire_cycle),
                        queue_[0].size()));
      }
      if (lst_cnt_[w] != 1 || wt_cnt_[w] != 0) {
        this->fail(strf("fence: tracker models count %" PRIu64 " local stores and %" PRIu64 " remote accesses for "
                        "warp %u after a %s, expected 1 and 0", lst_cnt_[w], wt_cnt_[w], w, what));
      }
      // One bank commit: lmem_st_drained rises at the edge, and the fence leaves in that cycle.
      commit_pct_ = 100;
      uint64_t c_commit = cycle_;
      this->offer_all({}, 4);
      if (fo.fire_cycle != c_commit + 1 || lst_rise_[w] != fo.fire_cycle || this->busy()) {
        this->fail(strf("fence after a %s: committed in cycle %" PRIu64 ", lmem_st_drained rose in %" PRId64
                        ", fired in %" PRId64 " (expected %" PRIu64 ")", what, c_commit - c0,
                        int64_t(lst_rise_[w] - c0), int64_t(fo.fire_cycle - c0), c_commit + 1 - c0));
      }
      std::printf("fence: %s, fence held %" PRIu64 " cycles with the commit withheld and the load behind it, "
                  "then left %" PRId64 " cycle(s) after the commit (the cycle lmem_st_drained rose)\n",
                  what, fo.held_cycles, int64_t(fo.fire_cycle - c_commit));
    }

    // A local load, a local AMO and a store-flagged fence count nothing: the fence behind them is not held.
    {
      commit_pct_ = 0;
      uint64_t lanes0 = n_lst_lanes_;
      Request ld = this->fence_req(w, false, false, local);
      Request amo = this->fence_req(w, false, false, cfg_.lmem_base + 24);
      for (auto& at : amo.attr) {
        set_bits(at, attr_local_pos_, 1, 1);
        set_bits(at, 4, 1, 1);   // amo_valid (MEM_ATTR_AMO_OFFS); the LSU sends an AMO with rw=0
      }
      Request fs = this->fence_req(w, true, true, local);
      Request fe = this->fence_req(w, true, false, 0);
      uint64_t c3 = cycle_;
      this->offer_all({ld, amo, fs, fe}, 8);
      const FenceObs& fo3 = fobs_[fe.seq];
      if (n_lst_lanes_ != lanes0 || lst_cnt_[w] != 0 || fo3.held_cycles != 0 || fo3.fire_cycle == ~uint64_t(0)) {
        this->fail(strf("fence: a local load, AMO and store-flagged fence counted %" PRIu64 " local stores; the "
                        "fence behind them held %" PRIu64 " cycles, fired in %" PRId64, n_lst_lanes_ - lanes0,
                        fo3.held_cycles, int64_t(fo3.fire_cycle - c3)));
      }
      commit_pct_ = 100;
      std::printf("fence: a local load, a local AMO and a store-flagged fence counted no local store\n");
    }

    if (!two) {
      std::printf("fence: the remote cases need a cluster of at least two cores, skipped\n");
      return;
    }

    // A remote store, then a remote load: each with its warp's fence and a
    // later local load of block 0 behind it, every completion withheld.
    for (bool rw : {true, false}) {
      const char* what = rw ? "store" : "load";
      ack_pct_ = 0;
      Request ac = this->fence_req(w, false, rw, remote);
      Request fe = this->fence_req(w, true, false, 0);
      Request ld = this->fence_req(w2, false, false, local);
      uint64_t c0 = cycle_;
      this->offer_all({ac, fe, ld}, hold);
      const FenceObs& fo = fobs_[fe.seq];
      if (fo.fire_cycle != ~uint64_t(0) || fo.held_cycles + 3 < hold || queue_[0].size() != 2
       || queue_[0].front().seq != fe.seq) {
        this->fail(strf("fence after a remote %s: held %" PRIu64 " of %u cycles, fired at %" PRId64
                        ", %zu request(s) in the stage", what, fo.held_cycles, hold, int64_t(fo.fire_cycle),
                        queue_[0].size()));
      }
      if (wt_cnt_[w] != 1) {
        this->fail(strf("fence: tracker model counts %" PRIu64 " accesses for warp %u after a remote %s, expected 1",
                        wt_cnt_[w], w, what));
      }
      // One completion: drained rises at the edge, and the fence leaves in that cycle.
      ack_pct_ = 100;
      uint64_t c_ack = cycle_;
      this->offer_all({}, 4);
      if (fo.fire_cycle != c_ack + 1 || this->busy()) {
        this->fail(strf("fence after a remote %s: completed in cycle %" PRIu64 ", fired in %" PRId64
                        " (expected %" PRIu64 ")", what, c_ack - c0, int64_t(fo.fire_cycle - c0), c_ack + 1 - c0));
      }
      std::printf("fence: remote %s fired, fence held %" PRIu64 " cycles with the completion withheld and the "
                  "load behind it, then left %" PRId64 " cycle(s) after the completion\n",
                  what, fo.held_cycles, int64_t(fo.fire_cycle - c_ack));
    }
    // A drained warp's fence goes straight through.
    Request f2 = this->fence_req(w2, true, false, 0);
    uint64_t c2 = cycle_;
    this->offer_all({f2}, 4);
    const FenceObs& fo2 = fobs_[f2.seq];
    if (fo2.held_cycles != 0 || fo2.fire_cycle != c2 + 1) {
      this->fail(strf("fence: a drained warp's fence held %" PRIu64 " cycles, fired in %" PRId64 " (expected %" PRIu64 ")",
                      fo2.held_cycles, int64_t(fo2.fire_cycle - c2), uint64_t(1)));
    }
  }

  ///////////////////////////////////////////////////////////////////////////

  Options opts_;
  std::mt19937_64 rng_;
  vl_simulator<VVX_dlmem_xlate_top> sim_;
  uint64_t tick_ = 0;
  uint64_t cycle_ = 0;
  uint64_t last_progress_ = 0;
  uint32_t failures_ = 0;

  uint32_t ncc_, nw_, maxc_, lmem_log_, nb_, nl_, aw_, attr_w_, tag_w_, rsp_w_, pid_w_, nw_w_, k_w_, core_w_;
  uint32_t hart_w_, hart_lsb_core_, nt_bits_, core_id_, cnt_w_, my_core_, rb_, pid_bits_;
  uint32_t attr_local_pos_, attr_io_pos_, attr_hart_lsb_, attr_remote_pos_;
  uint64_t win_bytes_;
  ref::Cfg cfg_{};

  std::vector<TblRow> rows_;
  uint32_t stride_ = 0;
  std::optional<std::pair<uint32_t, TblRow>> wr_;
  uint32_t wr_stride_ = 0;

  std::vector<std::optional<Request>> offer_;
  std::vector<std::deque<Request>> queue_;
  std::vector<bool> ready_, ready_prev_, valid_prev_;
  std::vector<ReadyPhase> phase_;
  std::vector<std::optional<uint64_t>> entered_prev_;
  std::vector<bool> rsp_valid_, rsp_ready_;
  std::vector<Bits> rsp_data_;

  struct SchedPhase {
    uint32_t accept = 100;
    uint32_t drain = 100;
    uint32_t left = 0;
  };
  bool sched_ = false;
  uint32_t sched_cap_ = VX_CFG_LSU_QUEUE_IN_SIZE;
  std::vector<std::deque<uint64_t>> sched_q_;
  std::vector<SchedPhase> sched_phase_;
  std::vector<bool> sched_pop_;
  std::vector<uint32_t> sched_push_;
  std::vector<bool> fired_prev_;
  uint64_t n_fold_busy_ = 0, n_handoff_ = 0;
  size_t sched_max_ = 0;

  // Store tracker model.
  std::vector<uint64_t> wt_cnt_;
  std::vector<bool> drained_;
  std::vector<std::optional<uint32_t>> rem_st_prev_;
  uint32_t ack_pct_ = 100;

  // Local-store tracker model.
  std::vector<uint64_t> lst_cnt_;
  std::vector<bool> lst_drained_;
  std::map<uint32_t, uint64_t> lst_rise_;   // per warp, the last cycle lmem_st_drained rose
  std::vector<std::optional<uint32_t>> lst_prev_ = std::vector<std::optional<uint32_t>>(VX_CFG_NUM_LSU_BLOCKS);
  uint32_t commit_pct_ = 100;
  uint64_t n_lst_inc_ = 0, n_lst_lanes_ = 0, n_commits_ = 0, n_b2b_lst_ = 0;

  struct FenceObs {
    uint64_t held_cycles = 0;
    uint64_t fire_cycle = ~uint64_t(0);
  };
  std::map<uint64_t, FenceObs> fobs_;
  uint64_t n_fence_held_ = 0, n_held_cycles_ = 0, n_b2b_ = 0, n_acks_ = 0;

  bool     lane0_ = false;
  uint64_t last_addr0_ = 0;
  bool     last_remote0_ = false;
  bool     last_oob0_ = false;
  uint64_t next_seq_ = 0;

  uint64_t n_in_ = 0, n_out_ = 0, n_fence_ = 0, n_writes_ = 0, n_stall_ = 0;
  uint64_t n_kind_[4] = {0, 0, 0, 0};
  uint64_t n_untranslated_ = 0, n_wt_inc_ = 0, n_wt_stores_ = 0, n_wt_reads_ = 0;
  uint64_t n_remote_past_ = 0, n_alias_none_ = 0;
  size_t max_occ_ = 0;
  std::vector<uint32_t> n_strides_;
};

static void usage(const char* prog) {
  std::printf("usage: %s [--mode=all|directed|random|sched|oob|remote|fence] [--seed=N] [--launches=N] [--cycles=N] "
              "[--watchdog=N] [--no-oob]\n", prog);
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&](const char* key) -> const char* {
      size_t n = std::strlen(key);
      return (a.compare(0, n, key) == 0) ? a.c_str() + n : nullptr;
    };
    if (const char* v = val("--mode=")) {
      opts.mode = v;
    } else if (const char* v = val("--seed=")) {
      opts.seed = std::strtoul(v, nullptr, 0);
    } else if (const char* v = val("--launches=")) {
      opts.launches = std::strtoul(v, nullptr, 0);
    } else if (const char* v = val("--cycles=")) {
      opts.cycles = std::strtoul(v, nullptr, 0);
    } else if (const char* v = val("--watchdog=")) {
      opts.watchdog = std::strtoul(v, nullptr, 0);
    } else if (a == "--no-oob") {
      opts.oob = false;
    } else if (a.rfind("+verilator", 0) == 0) {
      continue;
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (opts.mode != "all" && opts.mode != "directed" && opts.mode != "random" && opts.mode != "sched"
   && opts.mode != "oob" && opts.mode != "remote" && opts.mode != "fence") {
    usage(argv[0]);
    return 2;
  }
  Testbench tb(opts);
  return tb.run();
}
