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
// out-of-bounds and outside addresses, masked lanes, fences and release AMOs
// (an AMO request with the .rl tag bit), under
// back-pressure phases. Every request must leave once, in order per block,
// one cycle after it enters while the output is ready, translated with the row
// current when it entered; untouched fields stay bit-equal, except the hart id
// of every local lane, which becomes the issuer's. Models of the core's two
// trackers count the remote lanes and the local lanes (loads, stores, AMOs)
// leaving the stage and drive drained and lmem_acc_drained, registered like the
// trackers, with acknowledgements and bank commits (several per cycle) at
// random rates: a fence or a release AMO must not leave while its warp is
// undrained on either, and one right behind a remote or local access of its
// warp must be held. Each warp uses one block, as in the core.
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
// store and a local load through the window (own rank), a plain local-memory
// store and load, and a local AMO, each held the same way on lmem_acc_drained
// until its bank commit; a fence counts nothing; a drained warp's fence is not
// held.
// --mode=rl: a release AMO right behind its warp's uncommitted local store, load
// and AMO, then behind its unacknowledged remote store, is held and leaves in
// the cycle lmem_acc_drained or drained rises; a plain AMO in the same spot is
// not held, nor is a drained warp's release AMO.
// --mode=mbar (mbarrier builds): mbarrier ops on local lanes are counted; an
// arrive is held like a release AMO, the other ops are not.
// --mode=mbar-remote (mbarrier builds, two cores): arrive, arrive_tx,
// expect_tx and complete_tx on remote lanes leave as remote AMOs, flagged
// remote, at the peer's address, opcode unchanged, counted by the remote-access
// tracker; a release arrive is held behind its warp's unacknowledged remote
// store, a relaxed one is not.
// --mode=mbar-global / --mode=mbar-remote-wait / --mode=mbar-remote-trywait:
// an arrive on a global lane, or a wait on a remote lane; the stage's assert
// must abort the run.
// --mode=all: directed, random, sched, fence, then rl (and mbar).

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
  bool     rl = false;              // release AMO
  uint32_t wid = 0;
  uint32_t cta = 0;
  uint32_t pid = 0;
  std::vector<uint32_t> byteen;
  std::vector<uint64_t> addr;   // word addresses
  std::vector<Bits>     attr;
  std::vector<uint64_t> data;
  Bits     tag;                 // raw; the top merges wid/cta/pid/fence/rl into it

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
  uint32_t exp_lacc = 0;         // local lanes
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
    attr_lacc_pos_   = this->single_bit(get_field(t->cfg_attr_lacc, 0, attr_w_), "is_lacc");
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
    lacc_cnt_.assign(nw_, 0);
    lacc_drained_.assign(nw_, true);
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
    if (opts_.mode == "rl" || opts_.mode == "all") {
      this->run_rl();
    }
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    if (opts_.mode == "mbar" || opts_.mode == "all") {
  #else
    if (opts_.mode == "mbar") {
  #endif
      this->run_mbar();
    }
    if (opts_.mode == "mbar-remote") {
      this->run_mbar_remote();
    }
    if (opts_.mode == "mbar-global") {
      this->run_mbar_neg(false, kMbarArrive);
    }
    if (opts_.mode == "mbar-remote-wait" || opts_.mode == "mbar-remote-trywait") {
      this->run_mbar_neg(true, opts_.mode == "mbar-remote-wait" ? kMbarTestWait : kMbarTestWait + 1);
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

  // The LSU sends an AMO with rw=0 and amo_valid set on its lanes.
  void mark_amo(Request& q) {
    for (auto& at : q.attr) {
      set_bits(at, kAttrAmoValid, 1, 1);
    }
  }

  void mark_mbar(Request& q, uint32_t op) {
    for (auto& at : q.attr) {
      set_bits(at, kAttrAmoValid, 1, 1);
      set_bits(at, kAttrAmoOp, 4, op);
    }
  }

  // Random attributes. An mbarrier build asserts on an mbarrier op outside the
  // local memory, so random traffic keeps its AMO opcodes below them (an AMO
  // marked later keeps the opcode drawn here).
  Bits rand_attr() {
    Bits at = this->rand_bits(attr_w_);
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    if (get_bits(at, kAttrAmoOp, 4) >= kMbarArrive) {
      set_bits(at, kAttrAmoOp + 3, 1, 0);
    }
  #endif
    return at;
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
    q.rl    = !q.fence && !q.rw && chance(10);
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
      q.attr.push_back(this->rand_attr());
      q.data.push_back(rnd());
    }
    if (q.rl) {
      this->mark_amo(q);
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
    q.exp_lacc = 0;
    // A local lane carries the issuer's hart id and is counted, whatever the
    // access; is_lacc marks exactly the counted lanes for their banks.
    auto local_lane = [&](uint32_t i) {
      set_bits(q.exp_attr[i], attr_hart_lsb_, hart_w_, this->hart_id(q.wid, q.pid, i));
      set_bits(q.exp_attr[i], attr_lacc_pos_, 1, 1);
      ++q.exp_lacc;
    };
    for (uint32_t i = 0; i < nl_; ++i) {
      set_bits(q.exp_attr[i], attr_remote_pos_, 1, 0);
      set_bits(q.exp_attr[i], attr_lacc_pos_, 1, 0);
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
      set_bits(t->lmem_acc_drained, w, 1, lacc_drained_[w]);
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
      set_bits(t->in_rl, b, 1, o->rl);
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
    std::vector<uint64_t> lacc_inc(nw_, 0), lacc_dec(nw_, 0);
    for (uint32_t k = 0; k < VX_CFG_LMEM_NUM_BANKS; ++k) {
      if (!chance(commit_pct_)) {
        continue;
      }
      std::vector<uint32_t> pend;
      for (uint32_t w = 0; w < nw_; ++w) {
        if (lacc_cnt_[w] > lacc_dec[w]) {
          pend.push_back(w);
        }
      }
      if (!pend.empty()) {
        ++lacc_dec[pend[below(pend.size())]];
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
      bool ordered = !q.empty() && (q.front().fence || q.front().rl);
      bool held = ordered && !(drained_[q.front().wid] && lacc_drained_[q.front().wid]);
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
        const char* what = h.fence ? "fence" : "release AMO";
        if (valid) {
          this->fail(strf("block %u seq %" PRIu64 ": %s of warp %u left with the warp undrained", b, h.seq, what,
                          h.wid));
        }
        ++n_held_cycles_;
        auto& fo = fobs_[h.seq];
        if (fo.held_cycles++ == 0) {
          ++(h.fence ? n_fence_held_ : n_rl_held_);
        }
      }
      if (ordered && rem_st_prev_[b] == q.front().wid) {
        // The fence or release AMO right behind its warp's remote access sees that access counted.
        if (!held) {
          this->fail(strf("block %u seq %" PRIu64 ": %s behind a remote access of warp %u not held", b,
                          q.front().seq, q.front().fence ? "fence" : "release AMO", q.front().wid));
        }
        ++(q.front().fence ? n_b2b_ : n_b2b_rl_);
      }
      if (ordered && lacc_prev_[b] == q.front().wid) {
        // The same for a local access.
        if (!held) {
          this->fail(strf("block %u seq %" PRIu64 ": %s behind a local access of warp %u not held", b,
                          q.front().seq, q.front().fence ? "fence" : "release AMO", q.front().wid));
        }
        ++(q.front().fence ? n_b2b_lacc_ : n_b2b_rl_);
      }
      rem_st_prev_[b].reset();
      lacc_prev_[b].reset();
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
      bool lacc_valid = get_bits(t->lacc_inc_valid, b, 1);
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
        bool want_lacc = (r.exp_lacc != 0);
        if (lacc_valid != want_lacc) {
          this->fail(strf("block %u seq %" PRIu64 ": lacc_inc_valid=%d, expected %d", b, r.seq, lacc_valid, want_lacc));
        }
        if (want_lacc) {
          uint64_t wid = get_bits(t->lacc_inc_wid, b * nw_w_, nw_w_);
          uint64_t cnt = get_bits(t->lacc_inc_cnt, b * cnt_w_, cnt_w_);
          if (wid != r.wid || cnt != r.exp_lacc) {
            this->fail(strf("block %u seq %" PRIu64 ": lacc_inc wid %" PRIu64 " cnt %" PRIu64 ", expected %u %u",
                            b, r.seq, wid, cnt, r.wid, r.exp_lacc));
          }
          ++n_lacc_inc_;
          n_lacc_lanes_ += r.exp_lacc;
          n_lacc_rd_lanes_ += r.rw ? 0 : r.exp_lacc;
          lacc_inc[r.wid] += r.exp_lacc;
          lacc_prev_[b] = r.wid;
        }
        if (r.fence || r.rl || fobs_.count(r.seq)) {
          fobs_[r.seq].fire_cycle = cycle_;
        }
        if (lane0_) {
          last_addr0_   = get_bits(t->out_addr, b * nl_ * aw_, aw_);
          last_remote0_ = get_bits(t->out_remote, b * nl_, 1);
          last_oob0_    = get_bits(t->out_oob, b * nl_, 1);
          last_attr0_   = get_field(t->out_attr, b * nl_ * attr_w_, attr_w_);
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
      } else if (wt_valid || lacc_valid) {
        this->fail(strf("block %u: wt_inc_valid or lacc_inc_valid without an output handshake", b));
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
          if (get_bits(o->tag_eff, 0, 1) != o->fence || get_bits(o->tag_eff, 1, 1) != o->rl) {
            this->fail(strf("block %u: merged tag lost the fence or release bit", b));
          }
          this->expect(*o);
          ++n_in_;
          n_fence_ += o->fence;
          n_rl_ += o->rl;
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
      if (!lacc_drained_[w] && lacc_dec[w] == lacc_cnt_[w] && lacc_inc[w] == 0) {
        lacc_rise_[w] = cycle_ + 1;
      }
      lacc_cnt_[w] += lacc_inc[w];
      lacc_cnt_[w] -= lacc_dec[w];
      n_commits_ += lacc_dec[w];
      progress |= (lacc_dec[w] != 0);
      lacc_drained_[w] = (lacc_cnt_[w] == 0);
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
          || std::any_of(lacc_cnt_.begin(), lacc_cnt_.end(), [](auto n) { return n != 0; })
          || std::any_of(lacc_drained_.begin(), lacc_drained_.end(), [](bool d) { return !d; });
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
    std::printf("random: %" PRIu64 " local-access events (%" PRIu64 " lanes, %" PRIu64 " of loads or AMOs), %" PRIu64
                " bank commits, %" PRIu64 " fences right behind their warp's local access\n", n_lacc_inc_,
                n_lacc_lanes_, n_lacc_rd_lanes_, n_commits_, n_b2b_lacc_);
    std::printf("random: %" PRIu64 " release AMOs, %" PRIu64 " held, %" PRIu64 " right behind their warp's remote "
                "access or local access\n", n_rl_, n_rl_held_, n_b2b_rl_);
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
    if (n_lacc_lanes_ == 0 || n_lacc_rd_lanes_ == 0 || n_b2b_lacc_ == 0 || n_fence_held_ == 0) {
      this->fail("random traffic never counted a local store and a local load, or never held a fence right behind one");
    }
    if (n_rl_held_ == 0 || n_b2b_rl_ == 0) {
      this->fail("random traffic never held a release AMO, or never put one right behind its warp's access");
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
      q.attr.push_back(this->rand_attr());
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
      q.attr.push_back(this->rand_attr());
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

    // Local accesses through the window (own rank) and to the plain local
    // memory, and a local AMO: each with its warp's fence and a later local
    // load behind it, every bank commit withheld.
    enum { LA_WIN_ST, LA_PLAIN_ST, LA_WIN_LD, LA_PLAIN_LD, LA_AMO, NUM_LA };
    static const char* la_names[NUM_LA] = {"window store to the own rank", "plain local-memory store",
                                           "window load from the own rank", "plain local-memory load", "local AMO"};
    for (int k = 0; k < NUM_LA; ++k) {
      const char* what = la_names[k];
      ack_pct_ = 100;
      commit_pct_ = 0;
      const bool plain = (k == LA_PLAIN_ST || k == LA_PLAIN_LD);
      Request st = (k == LA_AMO) ? this->amo_req(w, false, cfg_.lmem_base + 24)
                                 : this->fence_req(w, false, k == LA_WIN_ST || k == LA_PLAIN_ST,
                                                   plain ? cfg_.lmem_base + 16 : local);
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
      if (lacc_cnt_[w] != 1 || wt_cnt_[w] != 0) {
        this->fail(strf("fence: tracker models count %" PRIu64 " local accesses and %" PRIu64 " remote accesses for "
                        "warp %u after a %s, expected 1 and 0", lacc_cnt_[w], wt_cnt_[w], w, what));
      }
      // One bank commit: lmem_acc_drained rises at the edge, and the fence leaves in that cycle.
      commit_pct_ = 100;
      uint64_t c_commit = cycle_;
      this->offer_all({}, 4);
      if (fo.fire_cycle != c_commit + 1 || lacc_rise_[w] != fo.fire_cycle) {
        this->fail(strf("fence after a %s: committed in cycle %" PRIu64 ", lmem_acc_drained rose in %" PRId64
                        ", fired in %" PRId64 " (expected %" PRIu64 ")", what, c_commit - c0,
                        int64_t(lacc_rise_[w] - c0), int64_t(fo.fire_cycle - c0), c_commit + 1 - c0));
      }
      this->drain();
      std::printf("fence: %s, fence held %" PRIu64 " cycles with the commit withheld and the load behind it, "
                  "then left %" PRId64 " cycle(s) after the commit (the cycle lmem_acc_drained rose)\n",
                  what, fo.held_cycles, int64_t(fo.fire_cycle - c_commit));
    }

    // A store-flagged fence counts nothing: the fence behind it is not held.
    {
      commit_pct_ = 0;
      uint64_t lanes0 = n_lacc_lanes_;
      Request fs = this->fence_req(w, true, true, local);
      Request fe = this->fence_req(w, true, false, 0);
      uint64_t c3 = cycle_;
      this->offer_all({fs, fe}, 6);
      const FenceObs& fo3 = fobs_[fe.seq];
      if (n_lacc_lanes_ != lanes0 || lacc_cnt_[w] != 0 || fo3.held_cycles != 0 || fo3.fire_cycle == ~uint64_t(0)) {
        this->fail(strf("fence: a store-flagged fence counted %" PRIu64 " local accesses; the fence behind it held %"
                        PRIu64 " cycles, fired in %" PRId64, n_lacc_lanes_ - lanes0, fo3.held_cycles,
                        int64_t(fo3.fire_cycle - c3)));
      }
      commit_pct_ = 100;
      std::printf("fence: a store-flagged fence counted no local access\n");
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

  Request amo_req(uint32_t wid, bool rl, uint64_t byte_addr) {
    Request q = this->fence_req(wid, false, false, byte_addr);
    q.rl = rl;
    for (auto& at : q.attr) {
      set_bits(at, attr_local_pos_, 1, 1);
    }
    this->mark_amo(q);
    return q;
  }

  // An AMO right behind its warp's access (kind 0 store, 1 load, 2 AMO) with
  // the access's completion withheld: a release AMO is held until the
  // completion, then leaves in the cycle the tracker's flag rises; a plain one
  // leaves right behind the access. rl_op/plain_op make them mbarrier ops;
  // remote_amo sends both AMOs to rank 1.
  void rl_case(const char* what, bool remote_st, uint64_t st_addr, uint32_t w, uint32_t w2, int kind = 0,
               uint32_t rl_op = 0, uint32_t plain_op = 0, bool remote_amo = false) {
    const uint32_t hold = 20;
    const uint64_t amo_addr = remote_amo ? cfg_.dlmem_base + (uint64_t(1) << lmem_log_) + 24 : cfg_.lmem_base + 24;
    auto withhold = [&](bool on) {
      (remote_st ? ack_pct_ : commit_pct_) = on ? 0 : 100;
    };
    auto pending = [&]() { return remote_st ? wt_cnt_[w] : lacc_cnt_[w]; };
    auto store = [&]() {
      Request st = (kind == 2) ? this->amo_req(w, false, st_addr) : this->fence_req(w, false, kind == 0, st_addr);
      if (!remote_st) {
        for (auto& at : st.attr) {
          set_bits(at, attr_local_pos_, 1, 1);
        }
      }
      return st;
    };

    withhold(true);
    Request am = this->amo_req(w, true, amo_addr);
    if (rl_op != 0) {
      this->mark_mbar(am, rl_op);
    }
    Request ld = this->fence_req(w2, false, false, cfg_.dlmem_base + 16);
    uint64_t c0 = cycle_;
    this->offer_all({store(), am, ld}, hold);
    const FenceObs& ho = fobs_[am.seq];
    if (ho.fire_cycle != ~uint64_t(0) || ho.held_cycles + 3 < hold || queue_[0].size() != 2
     || queue_[0].front().seq != am.seq || pending() != 1) {
      this->fail(strf("rl: release AMO after a %s: held %" PRIu64 " of %u cycles, fired at %" PRId64
                      ", %zu request(s) in the stage, %" PRIu64 " pending", what, ho.held_cycles, hold,
                      int64_t(ho.fire_cycle), queue_[0].size(), pending()));
    }
    withhold(false);
    uint64_t c_done = cycle_;
    this->offer_all({}, 2);
    // Read before the AMO's own commit raises the flag again.
    bool rise_ok = remote_st || lacc_rise_[w] == ho.fire_cycle;
    this->offer_all({}, 2);
    if (ho.fire_cycle != c_done + 1 || !rise_ok || this->busy()) {
      this->fail(strf("rl: release AMO after a %s: completed in cycle %" PRIu64 ", fired in %" PRId64
                      " (expected %" PRIu64 ")", what, c_done - c0, int64_t(ho.fire_cycle - c0), c_done + 1 - c0));
    }
    std::printf("rl: release AMO after a %s held %" PRIu64 " cycles with the completion withheld and a load "
                "behind it, then left %" PRId64 " cycle(s) after the completion\n",
                what, ho.held_cycles, int64_t(ho.fire_cycle - c_done));

    withhold(true);
    Request pa = this->amo_req(w, false, amo_addr);
    if (plain_op != 0) {
      this->mark_mbar(pa, plain_op);
    }
    const FenceObs& po = fobs_[pa.seq];   // watched, so its fire cycle is recorded
    uint64_t c1 = cycle_;
    this->offer_all({store(), pa}, 6);
    // The plain AMO counts beside the access when both are local or both remote.
    const uint64_t want_pending = (remote_st == remote_amo) ? 2 : 1;
    if (po.held_cycles != 0 || po.fire_cycle != c1 + 2 || pending() != want_pending) {
      this->fail(strf("rl: plain AMO after a %s: held %" PRIu64 " cycles, fired in %" PRId64 " (expected 2), %"
                      PRIu64 " pending (expected %" PRIu64 ")", what, po.held_cycles, int64_t(po.fire_cycle - c1),
                      pending(), want_pending));
    }
    this->drain();
    std::printf("rl: plain AMO after a %s left right behind it\n", what);
  }

  void run_rl() {
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
    uint32_t w  = 0;
    uint32_t w2 = (nw_ > nb_) ? nb_ : 0;

    this->rl_case("plain local-memory store", false, cfg_.lmem_base + 16, w, w2);
    this->rl_case("plain local-memory load", false, cfg_.lmem_base + 16, w, w2, 1);
    this->rl_case("local AMO", false, cfg_.lmem_base + 32, w, w2, 2);
    if (two) {
      this->rl_case("remote store", true, remote, w, w2);
    } else {
      std::printf("rl: the remote case needs a cluster of at least two cores, skipped\n");
    }

    // A drained warp's release AMO goes straight through, to any address.
    Request a2 = this->amo_req(w2, true, two ? remote : cfg_.lmem_base + 24);
    uint64_t c2 = cycle_;
    this->offer_all({a2}, 4);
    const FenceObs& ao = fobs_[a2.seq];
    if (ao.held_cycles != 0 || ao.fire_cycle != c2 + 1) {
      this->fail(strf("rl: a drained warp's release AMO held %" PRIu64 " cycles, fired in %" PRId64 " (expected 1)",
                      ao.held_cycles, int64_t(ao.fire_cycle - c2)));
    }
    this->drain();
    std::printf("rl: a drained warp's release AMO was not held\n");
  }

  void rl_table(bool two) {
    TblRow r;
    r.k = two ? 2 : 1;
    r.peers.assign(maxc_, ref::Peer{my_core_, 0});
    if (two) {
      r.peers[1] = ref::Peer{(my_core_ + 1) % ncc_, 1};
    }
    wr_stride_ = 64;
    wr_ = std::make_pair(0u, r);
    this->clock();
  }

  // mbarrier ops on local lanes count like any local access; an arrive is
  // held behind its warp's earlier local access like a release AMO, and the
  // other ops are not.
  void run_mbar() {
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    this->drain();
    this->rl_table(false);
    const uint32_t w = 0;
    const uint32_t w2 = (nw_ > nb_) ? nb_ : 0;
    for (uint32_t op = kMbarArrive; op <= kMbarArrive + 5; ++op) {
      for (bool plain : {false, true}) {
        Request q = this->fence_req(w, false, false, 0);
        q.mask = ref::low_mask(nl_);
        q.rl = (op <= kMbarArrive + 1);
        for (uint32_t i = 0; i < nl_; ++i) {
          q.addr[i] = ((plain ? cfg_.lmem_base : cfg_.dlmem_base) + 64 + 8 * i) >> 3;
          if (plain) {
            set_bits(q.attr[i], attr_local_pos_, 1, 1);
          }
        }
        this->mark_mbar(q, op);
        const uint64_t lanes0 = n_lacc_lanes_, outs0 = n_out_;
        this->offer_all({q}, 4);
        if (n_lacc_lanes_ - lanes0 != nl_ || n_out_ != outs0 + 1 || this->busy()) {
          this->fail(strf("mbar: op %u on %u %s lanes counted %" PRIu64 " local accesses", op, nl_,
                          plain ? "local-memory" : "window", n_lacc_lanes_ - lanes0));
        }
        this->drain();
      }
    }
    std::printf("mbar: ops 10..15 on every lane, through the window and to the local memory, counted per lane\n");
    this->rl_case("plain local-memory load, then an arrive", false, cfg_.lmem_base + 16, w, w2, 1,
                  kMbarArrive, kMbarExpectTx);
    this->rl_case("local AMO, then an arrive_tx", false, cfg_.lmem_base + 32, w, w2, 2,
                  kMbarArrive + 1, kMbarArrive + 4);
    this->rl_case("plain local-memory store, then an arrive", false, cfg_.lmem_base + 16, w, w2, 0,
                  kMbarArrive, kMbarArrive + 3);
  #else
    this->fail("mbar mode needs an mbarrier build");
  #endif
  }

  void run_mbar_remote() {
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    if (ncc_ < 2 || maxc_ < 2) {
      this->fail("mbar-remote mode needs a cluster of at least two cores");
      return;
    }
    this->drain();
    this->rl_table(true);
    const uint32_t peer = (my_core_ + 1) % ncc_;
    const uint64_t rank1 = cfg_.dlmem_base + (uint64_t(1) << lmem_log_);
    // Rank 1 is slot 1 of the peer core: slot base 64 at the table's stride.
    const uint64_t want0 = (cfg_.dlmem_base | (uint64_t(peer) << lmem_log_) | 64) >> 3;
    uint32_t n = 0;
    for (uint32_t op = kMbarArrive; op <= kMbarArrive + 3; ++op) {
      for (bool rl : {true, false}) {
        if (rl && op > kMbarArrive + 1) {
          continue;
        }
        Request q = this->fence_req(0, false, false, 0);
        q.mask = ref::low_mask(nl_);
        q.rl = rl;
        for (uint32_t i = 0; i < nl_; ++i) {
          q.addr[i] = (rank1 + 8 * i) >> 3;
          set_bits(q.attr[i], attr_local_pos_, 1, 0);
        }
        this->mark_mbar(q, op);
        const uint64_t rd0 = n_wt_reads_, lanes0 = n_lacc_lanes_, outs0 = n_out_;
        last_remote0_ = false;
        lane0_ = true;
        this->offer_all({q}, 4);
        lane0_ = false;
        uint32_t got_op = uint32_t(get_bits(last_attr0_, kAttrAmoOp, 4));
        if (n_out_ != outs0 + 1 || this->busy() || n_wt_reads_ - rd0 != nl_ || n_lacc_lanes_ != lanes0
         || !last_remote0_ || !get_bits(last_attr0_, attr_remote_pos_, 1)
         || !get_bits(last_attr0_, kAttrAmoValid, 1) || got_op != op || last_addr0_ != want0) {
          this->fail(strf("mbar-remote: %s op %u on %u remote lanes: %" PRIu64 " left, %" PRIu64
                          " remote and %" PRIu64 " local lanes counted, lane 0 remote=%d op %u addr 0x%" PRIx64
                          " (expected 0x%" PRIx64 ")", rl ? "release" : "relaxed", op, nl_, n_out_ - outs0,
                          n_wt_reads_ - rd0, n_lacc_lanes_ - lanes0, last_remote0_, got_op, last_addr0_ << 3,
                          want0 << 3));
        }
        this->drain();
        ++n;
      }
    }
    std::printf("mbar-remote: %u arrive/tx ops (release and relaxed) on every lane left as remote AMOs to core %u, "
                "opcode unchanged, counted as remote accesses\n", n, peer);
    const uint32_t w2 = (nw_ > nb_) ? nb_ : 0;
    this->rl_case("remote store, then a remote arrive", true, rank1 + 8, 0, w2, 0, kMbarArrive, kMbarArrive, true);
    this->rl_case("remote store, then a remote arrive_tx", true, rank1 + 8, 0, w2, 0, kMbarArrive + 1,
                  kMbarArrive + 1, true);
    this->rl_case("remote store, then a remote arrive and expect_tx", true, rank1 + 8, 0, w2, 0, kMbarArrive,
                  kMbarExpectTx, true);
  #else
    this->fail("mbar modes need an mbarrier build");
  #endif
  }

  // One mbarrier op on a global or a remote lane; the build's assert must abort.
  void run_mbar_neg(bool remote, uint32_t op) {
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    const bool two = (ncc_ >= 2 && maxc_ >= 2);
    if (remote && !two) {
      this->fail("mbar-remote-wait modes need a cluster of at least two cores");
      return;
    }
    this->rl_table(two);
    Request q = this->fence_req(0, false, false, remote ? cfg_.dlmem_base + (uint64_t(1) << lmem_log_) + 8 : 0x1000);
    for (auto& at : q.attr) {
      set_bits(at, attr_local_pos_, 1, 0);
    }
    this->mark_mbar(q, op);
    std::printf("mbar: issuing mbarrier op %u on a %s lane; the stage's assert must abort here\n", op,
                remote ? "remote" : "global");
    std::fflush(stdout);
    this->offer_all({q}, 6);
    this->fail("the mbarrier op on a " + std::string(remote ? "remote" : "global") + " lane did not abort the run");
  #else
    (void)remote;
    (void)op;
    this->fail("mbar modes need an mbarrier build");
  #endif
  }

  ///////////////////////////////////////////////////////////////////////////

  static constexpr uint32_t kAttrAmoValid = 4;   // MEM_ATTR_AMO_OFFS
  static constexpr uint32_t kAttrAmoOp    = 5;   // amo_req_t.amo_op, above amo_valid
  static constexpr uint32_t kMbarArrive   = 10;  // amo_op_e AMO_OP_MBAR_ARRIVE
  static constexpr uint32_t kMbarExpectTx = 12;
  static constexpr uint32_t kMbarTestWait = 14;

  Options opts_;
  std::mt19937_64 rng_;
  vl_simulator<VVX_dlmem_xlate_top> sim_;
  uint64_t tick_ = 0;
  uint64_t cycle_ = 0;
  uint64_t last_progress_ = 0;
  uint32_t failures_ = 0;

  uint32_t ncc_, nw_, maxc_, lmem_log_, nb_, nl_, aw_, attr_w_, tag_w_, rsp_w_, pid_w_, nw_w_, k_w_, core_w_;
  uint32_t hart_w_, hart_lsb_core_, nt_bits_, core_id_, cnt_w_, my_core_, rb_, pid_bits_;
  uint32_t attr_local_pos_, attr_io_pos_, attr_hart_lsb_, attr_remote_pos_, attr_lacc_pos_;
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

  // Local-access tracker model.
  std::vector<uint64_t> lacc_cnt_;
  std::vector<bool> lacc_drained_;
  std::map<uint32_t, uint64_t> lacc_rise_;   // per warp, the last cycle lmem_acc_drained rose
  std::vector<std::optional<uint32_t>> lacc_prev_ = std::vector<std::optional<uint32_t>>(VX_CFG_NUM_LSU_BLOCKS);
  uint32_t commit_pct_ = 100;
  uint64_t n_lacc_inc_ = 0, n_lacc_lanes_ = 0, n_lacc_rd_lanes_ = 0, n_commits_ = 0, n_b2b_lacc_ = 0;

  struct FenceObs {
    uint64_t held_cycles = 0;
    uint64_t fire_cycle = ~uint64_t(0);
  };
  std::map<uint64_t, FenceObs> fobs_;
  uint64_t n_fence_held_ = 0, n_held_cycles_ = 0, n_b2b_ = 0, n_acks_ = 0;
  uint64_t n_rl_ = 0, n_rl_held_ = 0, n_b2b_rl_ = 0;

  bool     lane0_ = false;
  uint64_t last_addr0_ = 0;
  bool     last_remote0_ = false;
  bool     last_oob0_ = false;
  Bits     last_attr0_;
  uint64_t next_seq_ = 0;

  uint64_t n_in_ = 0, n_out_ = 0, n_fence_ = 0, n_writes_ = 0, n_stall_ = 0;
  uint64_t n_kind_[4] = {0, 0, 0, 0};
  uint64_t n_untranslated_ = 0, n_wt_inc_ = 0, n_wt_stores_ = 0, n_wt_reads_ = 0;
  uint64_t n_remote_past_ = 0, n_alias_none_ = 0;
  size_t max_occ_ = 0;
  std::vector<uint32_t> n_strides_;
};

static void usage(const char* prog) {
  std::printf("usage: %s [--mode=all|directed|random|sched|oob|remote|fence|rl|mbar|mbar-global|mbar-remote|"
              "mbar-remote-wait|mbar-remote-trywait] [--seed=N] [--launches=N] [--cycles=N] [--watchdog=N] "
              "[--no-oob]\n", prog);
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
   && opts.mode != "oob" && opts.mode != "remote" && opts.mode != "fence" && opts.mode != "rl"
   && opts.mode != "mbar" && opts.mode != "mbar-global" && opts.mode != "mbar-remote"
   && opts.mode != "mbar-remote-wait" && opts.mode != "mbar-remote-trywait") {
    usage(argv[0]);
    return 2;
  }
  Testbench tb(opts);
  return tb.run();
}
