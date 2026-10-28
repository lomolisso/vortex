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

// VX_local_mem testbench: LSU lanes and the DMA row port share the bank
// SRAMs. Directed runs inject one DMA access at every cycle offset of an
// atomic's bank occupancy and across an LR/SC window; a randomized run mixes
// atomics, loads, stores and DMA traffic. Everything is checked against a
// golden memory model. DLMEM builds add an input whose stores are
// acknowledged once committed, as the DLMEM ingress uses it, and report every
// access of the other inputs on a per-bank commit port with its warp: a load or
// store once the bank performs it, an atomic once it writes back. mbarrier
// builds run the mbarrier ops against mbar_ref, directed and racing DMA and
// stores, and check every bank's waiter events: an insert {wid, row} one cycle
// after a try_wait is taken that reads the phase not complete, a flip {row}
// two cycles after a writing op is taken that completes it, nothing for any
// other op; in op order per bank. --neg=<underflow|txrange|align> issues one
// erroneous op, and the bank's assert must abort the run.

#include "vl_simulator.h"
#include "VVX_local_mem_top.h"
#include "VX_config.h"

#include <algorithm>
#include <array>
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
#include <unordered_map>
#include <vector>

#ifdef VX_CFG_EXT_MBAR_ENABLE
#include "mbar_ref.h"
namespace mbar = vortex::mbar_ref;
#endif

constexpr uint32_t XLEN       = VX_CFG_XLEN;
constexpr uint32_t WORD_SIZE  = XLEN / 8;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
// The top's last input acknowledges its stores (STORE_ACK_MASK).
constexpr uint32_t NUM_REQS   = VX_CFG_NUM_LSU_BLOCKS * VX_CFG_NUM_LSU_LANES + 1;
constexpr int      ACK_LANE   = NUM_REQS - 1;
#else
constexpr uint32_t NUM_REQS   = VX_CFG_NUM_LSU_BLOCKS * VX_CFG_NUM_LSU_LANES;
constexpr int      ACK_LANE   = -1;
#endif
constexpr uint32_t NUM_BANKS  = VX_CFG_LMEM_NUM_BANKS;
// Inputs [0, NUM_LSU_REQS) are the LSU lanes.
constexpr uint32_t NUM_LSU_REQS = VX_CFG_NUM_LSU_BLOCKS * VX_CFG_NUM_LSU_LANES;
constexpr uint32_t NUM_WARPS  = VX_CFG_NUM_WARPS;
constexpr uint32_t NT_BITS    = __CLOG2(VX_CFG_NUM_THREADS);
constexpr uint32_t NW_BITS    = __CLOG2(VX_CFG_NUM_WARPS);
constexpr uint32_t LMEM_SIZE  = 1u << VX_CFG_LMEM_LOG_SIZE;
constexpr uint32_t NUM_ROWS   = LMEM_SIZE / (NUM_BANKS * WORD_SIZE);
constexpr uint32_t NUM_WORDS  = NUM_ROWS * NUM_BANKS;
constexpr uint32_t BANK_BITS  = __CLOG2(NUM_BANKS);
constexpr uint32_t ROW_BITS   = __CLOG2(NUM_ROWS);
constexpr uint32_t ADDR_BITS  = ROW_BITS + BANK_BITS;
constexpr bool     AMO_EN     = VX_CFG_EXT_A_ENABLED;
constexpr bool     CAS_EN     = VX_CFG_EXT_ZACAS_ENABLED;

// Must match the VX_local_mem_top localparams.
constexpr uint32_t TAG_BITS     = 16;
constexpr uint32_t DMA_TAG_BITS = 8;
constexpr uint32_t AMO_OP_BITS  = 4;
constexpr uint32_t HART_BITS    = 16;

constexpr uint64_t WORD_MASK = (XLEN == 64) ? ~0ull : 0xffffffffull;
constexpr uint64_t MASK32    = 0xffffffffull;
constexpr uint32_t FULL_BE   = (1u << WORD_SIZE) - 1;

static_assert((NUM_BANKS & (NUM_BANKS - 1)) == 0, "NUM_BANKS must be a power of two");

static bool acks_stores(uint32_t lane) {
  return static_cast<int>(lane) == ACK_LANE;
}

#ifdef VX_CFG_EXT_DLMEM_ENABLE
constexpr bool COMMIT_EN = true;
#else
constexpr bool COMMIT_EN = false;
#endif

#ifdef VX_CFG_EXT_MBAR_ENABLE
constexpr bool MBAR_EN = true;
#else
constexpr bool MBAR_EN = false;
#endif

// An LSU input reports the commit of every access the translation counted.
static bool reports_commits(uint32_t lane, bool counted) {
  return COMMIT_EN && lane < NUM_LSU_REQS && counted;
}
static_assert(WORD_SIZE == 4 || WORD_SIZE == 8, "unsupported XLEN");

// VX_gpu_pkg amo_op_e
enum AmoOp : uint8_t {
  AMO_LR = 0, AMO_SC = 1, AMO_ADD = 2, AMO_SWAP = 3, AMO_XOR = 4,
  AMO_OR = 5, AMO_AND = 6, AMO_MIN = 7, AMO_MAX = 8, AMO_CAS = 9,
  AMO_MBAR_ARRIVE = 10, AMO_MBAR_ARRIVE_TX = 11, AMO_MBAR_EXPECT_TX = 12,
  AMO_MBAR_COMPLETE_TX = 13, AMO_MBAR_TEST_WAIT = 14, AMO_MBAR_TRY_WAIT = 15
};

static const char* amo_name(uint8_t op, bool uns) {
  switch (op) {
  case AMO_LR:   return "LR";
  case AMO_SC:   return "SC";
  case AMO_ADD:  return "AMOADD";
  case AMO_SWAP: return "AMOSWAP";
  case AMO_XOR:  return "AMOXOR";
  case AMO_OR:   return "AMOOR";
  case AMO_AND:  return "AMOAND";
  case AMO_MIN:  return uns ? "AMOMINU" : "AMOMIN";
  case AMO_MAX:  return uns ? "AMOMAXU" : "AMOMAX";
  case AMO_CAS:  return "AMOCAS";
  case AMO_MBAR_ARRIVE:      return "MBAR_ARRIVE";
  case AMO_MBAR_ARRIVE_TX:   return "MBAR_ARRIVE_TX";
  case AMO_MBAR_EXPECT_TX:   return "MBAR_EXPECT_TX";
  case AMO_MBAR_COMPLETE_TX: return "MBAR_COMPLETE_TX";
  case AMO_MBAR_TEST_WAIT:   return "MBAR_TEST_WAIT";
  case AMO_MBAR_TRY_WAIT:    return "MBAR_TRY_WAIT";
  default:       return "AMO?";
  }
}

///////////////////////////////////////////////////////////////////////////////

// Bit-field access that works on every Verilator port type (CData..QData
// and VlWide), so the testbench is independent of the configured widths.
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

static std::string hex(uint64_t v) {
  char buf[24];
  std::snprintf(buf, sizeof(buf), "0x%0*" PRIx64, static_cast<int>(XLEN / 4), v);
  return buf;
}

static std::string strf(const char* fmt, ...) {
  char buf[1024];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

///////////////////////////////////////////////////////////////////////////////
// Golden model

struct LsuReq {
  bool     rw     = false;
  uint32_t byteen = 0;
  uint32_t addr   = 0; // word address: {row, bank}
  uint64_t data   = 0;
  uint32_t tag    = 0;
  bool     amo    = false;
  uint8_t  op     = 0;
  bool     uns    = false;
  uint64_t cmp    = 0;
  int      wid    = -1; // issuing warp; -1: the hart id is the lane index
  bool     counted = true; // DLMEM: the translation counted it (attr is_lacc)
};

// hart id = {core, wid, tid}, with core 0 and the lane as the thread.
static uint32_t req_hart_id(uint32_t lane, const LsuReq& r) {
  if (r.wid < 0) {
    return lane;
  }
  uint32_t tid = lane & ((1u << NT_BITS) - 1);
  return (static_cast<uint32_t>(r.wid) << NT_BITS) | tid;
}

static uint32_t req_wid(uint32_t lane, const LsuReq& r) {
  return (req_hart_id(lane, r) >> NT_BITS) & ((1u << NW_BITS) - 1);
}

static bool is_plain_store(const LsuReq& r) {
  return r.rw && !r.amo;
}

static uint32_t req_bank(const LsuReq& r) {
  return r.addr & (NUM_BANKS - 1);
}

struct DmaReq {
  bool     rw  = false;
  uint32_t row = 0;
  std::array<uint32_t, NUM_BANKS> be{};
  std::array<uint64_t, NUM_BANKS> data{};
  uint32_t tag = 0;
};

using Row = std::array<uint64_t, NUM_BANKS>;

static uint32_t word_addr(uint32_t row, uint32_t bank) {
  return (row << BANK_BITS) | bank;
}

static uint64_t byte_mask(uint32_t byteen) {
  uint64_t m = 0;
  for (uint32_t b = 0; b < WORD_SIZE; ++b) {
    if ((byteen >> b) & 1) {
      m |= 0xffull << (8 * b);
    }
  }
  return m;
}

static uint64_t merge(uint64_t old, uint64_t data, uint32_t byteen) {
  uint64_t m = byte_mask(byteen);
  return ((old & ~m) | (data & m)) & WORD_MASK;
}

// Lane placement mirrors the bank: a .W on a 64-bit word is selected by
// byteen, operands arrive lane-positioned, the comparand right-aligned.
static uint64_t amo_new_word(const LsuReq& r, uint64_t old) {
  bool lane_hi = (XLEN == 64) && !(r.byteen & 1);
  bool is_w = (XLEN == 32) || (r.byteen != FULL_BE);
  uint32_t sh = lane_hi ? 32 : 0;
  uint64_t a = old >> sh;
  uint64_t b = r.data >> sh;
  uint64_t c = r.cmp;
  if (is_w) {
    a &= MASK32;
    b &= MASK32;
    c &= MASK32;
  }
  bool lt_s = is_w ? (static_cast<int32_t>(a) < static_cast<int32_t>(b))
                   : (static_cast<int64_t>(a) < static_cast<int64_t>(b));
  bool lt_u = a < b;
  bool lt = r.uns ? lt_u : lt_s;
  uint64_t res = a;
  switch (r.op) {
  case AMO_LR:   res = a; break;
  case AMO_SC:   res = b; break;
  case AMO_SWAP: res = b; break;
  case AMO_ADD:  res = a + b; break;
  case AMO_XOR:  res = a ^ b; break;
  case AMO_OR:   res = a | b; break;
  case AMO_AND:  res = a & b; break;
  case AMO_MIN:  res = lt ? a : b; break;
  case AMO_MAX:  res = lt ? b : a; break;
  case AMO_CAS:  res = (a == c) ? b : a; break;
  default:       break;
  }
  if (is_w) {
    res &= MASK32;
  }
  return merge(old, res << sh, r.byteen);
}

static uint64_t dma_merge(const DmaReq& d, uint32_t bank, uint64_t old) {
  return merge(old, d.data[bank], d.be[bank]);
}

static uint64_t sc_rsp_word(const LsuReq& r, bool fail) {
  bool lane_hi = (XLEN == 64) && !(r.byteen & 1);
  return static_cast<uint64_t>(fail ? 1 : 0) << (lane_hi ? 32 : 0);
}

///////////////////////////////////////////////////////////////////////////////
// DUT driver

struct Events {
  std::array<bool, NUM_REQS>     lsu_acc{};
  std::array<bool, NUM_REQS>     lsu_rsp{};
  std::array<uint64_t, NUM_REQS> rsp_data{};
  std::array<uint32_t, NUM_REQS> rsp_tag{};
  bool     dma_acc   = false;
  bool     dma_stall = false;
  bool     dma_rsp   = false;
  Row      dma_rsp_data{};
  uint32_t dma_rsp_tag = 0;
  std::array<LsuReq, NUM_REQS>    acc_req{};
  std::array<bool, NUM_BANKS>     commit{};
  std::array<uint32_t, NUM_BANKS> commit_wid{};
#ifdef VX_CFG_EXT_MBAR_ENABLE
  std::array<bool, NUM_BANKS>     mbar_evt{};
  std::array<bool, NUM_BANKS>     mbar_flip{};
  std::array<uint32_t, NUM_BANKS> mbar_wid{};
  std::array<uint32_t, NUM_BANKS> mbar_row{};
#endif
  uint32_t num_commits() const {
    uint32_t n = 0;
    for (bool c : commit) {
      n += c;
    }
    return n;
  }
  bool any() const {
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      if (lsu_acc[i] || lsu_rsp[i]) {
        return true;
      }
    }
    return dma_acc || dma_rsp;
  }
};

class Dut {
public:
  std::array<std::optional<LsuReq>, NUM_REQS> lane;
  std::array<bool, NUM_REQS> rsp_ready;
  std::optional<DmaReq> dma;
  bool dma_rsp_ready = true;

  Dut() {
    rsp_ready.fill(true);
    auto* t = sim_.operator->();
    t->reset = 1;
    for (int i = 0; i < 8; ++i) {
      this->clock();
    }
    t->reset = 0;
    cycle_ = 0;
  }

  uint64_t cycle() const {
    return cycle_;
  }

  // Inputs are applied, outputs sampled, then one rising edge.
  Events clock() {
    auto* t = sim_.operator->();
    this->drive(t);
    t->eval();
    Events ev;
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      ev.lsu_acc[i] = lane[i].has_value() && get_bits(t->mem_req_ready, i, 1);
      if (ev.lsu_acc[i]) {
        ev.acc_req[i] = *lane[i];
      }
      ev.lsu_rsp[i] = rsp_ready[i] && get_bits(t->mem_rsp_valid, i, 1);
      ev.rsp_data[i] = get_bits(t->mem_rsp_data, i * XLEN, XLEN);
      ev.rsp_tag[i] = static_cast<uint32_t>(get_bits(t->mem_rsp_tag, i * TAG_BITS, TAG_BITS));
    }
    bool dma_ready = t->dma_req_ready;
    ev.dma_acc   = dma.has_value() && dma_ready;
    ev.dma_stall = dma.has_value() && !dma_ready;
    ev.dma_rsp   = dma_rsp_ready && t->dma_rsp_valid;
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      ev.dma_rsp_data[b] = get_bits(t->dma_rsp_data, b * XLEN, XLEN);
    }
    ev.dma_rsp_tag = static_cast<uint32_t>(t->dma_rsp_tag);
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      ev.commit[b] = get_bits(t->lmem_commit_valid, b, 1);
      ev.commit_wid[b] = static_cast<uint32_t>(get_bits(t->lmem_commit_wid, b * 8, 8));
    }
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      ev.mbar_evt[b] = get_bits(t->mbar_evt_valid, b, 1);
      ev.mbar_flip[b] = get_bits(t->mbar_evt_flip, b, 1);
      ev.mbar_wid[b] = static_cast<uint32_t>(get_bits(t->mbar_evt_wid, b * 8, 8));
      ev.mbar_row[b] = static_cast<uint32_t>(get_bits(t->mbar_evt_row, b * ROW_BITS, ROW_BITS));
    }
    this->mbar_account(ev);
#endif
    this->account(ev);
    tick_ = sim_.step(tick_, 2);
    ++cycle_;
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      if (ev.lsu_acc[i]) {
        lane[i].reset();
      }
    }
    if (ev.dma_acc) {
      dma.reset();
    }
    return ev;
  }

private:
  void drive(VVX_local_mem_top* t) {
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      LsuReq r = lane[i].value_or(LsuReq{});
      set_bits(t->mem_req_valid, i, 1, lane[i].has_value());
      set_bits(t->mem_req_rw, i, 1, r.rw);
      set_bits(t->mem_req_byteen, i * WORD_SIZE, WORD_SIZE, r.byteen);
      set_bits(t->mem_req_addr, i * ADDR_BITS, ADDR_BITS, r.addr);
      set_bits(t->mem_req_data, i * XLEN, XLEN, r.data);
      set_bits(t->mem_req_tag, i * TAG_BITS, TAG_BITS, r.tag);
      set_bits(t->mem_req_amo_valid, i, 1, r.amo);
      set_bits(t->mem_req_amo_op, i * AMO_OP_BITS, AMO_OP_BITS, r.op);
      set_bits(t->mem_req_amo_unsigned, i, 1, r.uns);
      set_bits(t->mem_req_amo_cmp, i * XLEN, XLEN, r.cmp);
      set_bits(t->mem_req_hart_id, i * HART_BITS, HART_BITS, req_hart_id(i, r));
#ifdef VX_CFG_EXT_DLMEM_ENABLE
      set_bits(t->mem_req_lacc, i, 1, r.counted);
#endif
      set_bits(t->mem_rsp_ready, i, 1, rsp_ready[i]);
    }
    DmaReq d = dma.value_or(DmaReq{});
    t->dma_req_valid = dma.has_value();
    t->dma_req_rw = d.rw;
    t->dma_req_addr = d.row;
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      set_bits(t->dma_req_byteen, b * WORD_SIZE, WORD_SIZE, d.be[b]);
      set_bits(t->dma_req_data, b * XLEN, XLEN, d.data[b]);
    }
    t->dma_req_tag = d.tag;
    t->dma_rsp_ready = dma_rsp_ready;
  }

  // Per bank and warp, LSU-input accesses accepted and not yet reported
  // committed: it must never go negative, and must return to zero once the
  // banks drain.
  void account(const Events& ev) {
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      if (ev.lsu_acc[i] && reports_commits(i, ev.acc_req[i].counted)) {
        uint32_t w = req_wid(i, ev.acc_req[i]);
        ++uncommitted[w];
        ++bank_uncommitted[req_bank(ev.acc_req[i])][w];
        ++accepted;
      }
    }
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      if (!ev.commit[b]) {
        continue;
      }
      ++commits;
      uint32_t w = ev.commit_wid[b];
      if (w >= NUM_WARPS || bank_uncommitted[b][w] == 0) {
        commit_errors.push_back(strf("[cycle %" PRIu64 "] bank %u reported a commit of warp %u with no "
                                     "uncommitted access of that warp to the bank", cycle_, b, w));
        continue;
      }
      --uncommitted[w];
      --bank_uncommitted[b][w];
    }
  }

#ifdef VX_CFG_EXT_MBAR_ENABLE
  void mbar_account(const Events& ev) {
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      if (ev.mbar_evt[b]) {
        mbar_evs[b].push_back({cycle_, ev.mbar_flip[b], ev.mbar_wid[b], ev.mbar_row[b]});
      }
    }
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      if (ev.lsu_acc[i] && ev.acc_req[i].amo && ev.acc_req[i].op >= mbar::OP_BASE) {
        MbarOp o;
        o.lane = i;
        o.req = ev.acc_req[i];
        o.acc = cycle_;
        mbar_ops[req_bank(o.req)].push_back(o);
      }
    }
    for (uint32_t i = 0; i < NUM_REQS; ++i) {
      if (!ev.lsu_rsp[i]) {
        continue;
      }
      for (auto& q : mbar_ops) {
        for (MbarOp& o : q) {
          if (!o.known && o.lane == i && o.req.tag == ev.rsp_tag[i]) {
            o.known = true;
            o.rsp = cycle_;
            uint64_t d = ev.rsp_data[i];
            if (o.req.op == mbar::OP_TRY_WAIT) {
              o.owes = (d == 0) ? 1 : 0;
            } else if (o.req.op != mbar::OP_TEST_WAIT) {
              o.owes = mbar::apply(o.req.op, d, o.req.data).flip ? 2 : 0;
            }
          }
        }
      }
    }
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      auto& q = mbar_ops[b];
      while (!q.empty() && q.front().known) {
        const MbarOp& o = q.front();
        if (o.owes != 0) {
          const bool flip = (o.owes == 2);
          const uint32_t wid = req_wid(o.lane, o.req);
          const uint32_t row = o.req.addr >> BANK_BITS;
          std::string what = strf("%s lane %u wid %u row 0x%x (offered at %" PRIu64 ", answered at %" PRIu64 ")",
                                  amo_name(o.req.op, false), o.lane, wid, row, o.acc, o.rsp);
          if (mbar_evs[b].empty()) {
            mbar_errors.push_back(strf("bank %u: no %s for %s", b, flip ? "flip" : "insert", what.c_str()));
          } else {
            const MbarEv e = mbar_evs[b].front();
            mbar_evs[b].pop_front();
            if (e.flip != flip || e.row != row || (!flip && e.wid != wid) || e.cycle <= o.acc || e.cycle > o.rsp) {
              mbar_errors.push_back(strf("bank %u: %s at cycle %" PRIu64 " (wid %u, row 0x%x), expected the %s of %s",
                                         b, e.flip ? "flip" : "insert", e.cycle, e.wid, e.row,
                                         flip ? "flip" : "insert", what.c_str()));
            }
            (flip ? mbar_flips : mbar_inserts) += 1;
          }
        }
        q.pop_front();
      }
    }
  }
#endif

public:
  std::array<uint64_t, NUM_WARPS> uncommitted{};
  std::array<std::array<uint64_t, NUM_WARPS>, NUM_BANKS> bank_uncommitted{};
  uint64_t accepted = 0;
  uint64_t commits = 0;
  std::vector<std::string> commit_errors;

#ifdef VX_CFG_EXT_MBAR_ENABLE
  // Waiter events, per bank in op order. An mbarrier op taken by the bank owes
  // the event its response tells: a try_wait answering 0 an insert, a writing
  // op whose returned old word completes the phase a flip, any other none. The
  // owed events of each bank must come out in its ops' order, after the op was
  // offered and no later than its response; none may be left over.
  struct MbarOp {
    uint32_t lane = 0;
    LsuReq   req;
    uint64_t acc = 0;
    bool     known = false;
    int      owes = 0; // 0 none, 1 insert, 2 flip
    uint64_t rsp = 0;
  };
  struct MbarEv {
    uint64_t cycle = 0;
    bool     flip = false;
    uint32_t wid = 0;
    uint32_t row = 0;
  };
  std::array<std::deque<MbarOp>, NUM_BANKS> mbar_ops;
  std::array<std::deque<MbarEv>, NUM_BANKS> mbar_evs;
  uint64_t mbar_inserts = 0;
  uint64_t mbar_flips = 0;
  std::vector<std::string> mbar_errors;

  // Every op answered and every event matched.
  void mbar_drained(const char* ctx) {
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      if (!mbar_ops[b].empty()) {
        mbar_errors.push_back(strf("%s: bank %u has %zu mbarrier op(s) unanswered", ctx, b, mbar_ops[b].size()));
        mbar_ops[b].clear();
      }
      for (const MbarEv& e : mbar_evs[b]) {
        mbar_errors.push_back(strf("%s: bank %u %s at cycle %" PRIu64 " (wid %u, row 0x%x) owed by no op", ctx, b,
                                   e.flip ? "flip" : "insert", e.cycle, e.wid, e.row));
      }
      mbar_evs[b].clear();
    }
  }
#endif

private:
  vl_simulator<VVX_local_mem_top> sim_;
  uint64_t tick_  = 0;
  uint64_t cycle_ = 0;
};

///////////////////////////////////////////////////////////////////////////////

struct Options {
  uint32_t seed          = 1;
  uint32_t stress_seeds  = 4;
  uint32_t stress_cycles = 20000;
  uint32_t max_report    = 3;
  bool     verbose       = false;
  std::string neg;
};

class Testbench {
public:
  explicit Testbench(const Options& opts)
    : opts_(opts)
    , rng_(opts.seed)
    , gold_(NUM_WORDS, 0) {
  }

  int run() {
    std::printf("local_mem: XLEN=%u lanes=%u banks=%u rows=%u size=%u amo=%d zacas=%d mbar=%d seed=%u\n",
                XLEN, NUM_REQS, NUM_BANKS, NUM_ROWS, LMEM_SIZE, AMO_EN, CAS_EN, MBAR_EN, opts_.seed);
    if (!opts_.neg.empty()) {
    #ifdef VX_CFG_EXT_MBAR_ENABLE
      return this->mbar_negative(opts_.neg);
    #else
      std::printf("--neg needs an mbarrier build\n");
      return 2;
    #endif
    }
    this->init_memory();
    if (!this->calibrate()) {
      return 1;
    }
    if (ACK_LANE >= 0) {
      this->store_ack();
    }
    if (COMMIT_EN) {
      this->store_commit();
      this->commit();
    }
    if (AMO_EN) {
      this->directed();
      this->lrsc_window();
    #ifdef VX_CFG_EXT_MBAR_ENABLE
      this->mbar_directed();
      this->mbar_events();
      for (uint32_t s = 0; s < opts_.stress_seeds; ++s) {
        this->mbar_stress(opts_.seed * 104729u + s);
      }
    #endif
    } else {
      std::printf("atomics disabled (VX_CFG_EXT_A_ENABLE not set): directed atomic runs skipped\n");
    }
    for (uint32_t s = 0; s < opts_.stress_seeds; ++s) {
      this->stress(opts_.seed * 7919u + s);
    }
    this->final_check();
    std::printf("total cycles: %" PRIu64 "\n", dut_.cycle());
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

  uint64_t rand_word() {
    return rng_() & WORD_MASK;
  }

  uint32_t rand_below(uint32_t n) {
    return static_cast<uint32_t>(rng_() % n);
  }

  uint64_t& gold(uint32_t row, uint32_t bank) {
    return gold_.at(row * NUM_BANKS + bank);
  }

  uint32_t next_tag() {
    tag_ = (tag_ + 1) & ((1u << TAG_BITS) - 1);
    return tag_;
  }

  uint32_t next_dma_tag() {
    dma_tag_ = (dma_tag_ + 1) & ((1u << DMA_TAG_BITS) - 1);
    return dma_tag_;
  }

  void fail(const std::string& msg) {
    ++failures_;
    if (failures_ <= 200) {
      std::printf("%s\n", msg.c_str());
    } else if (failures_ == 201) {
      std::printf("... further failure messages suppressed\n");
    }
  }

  void idle(uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
      Events ev = dut_.clock();
      if (ev.any()) {
        this->fail(strf("FAIL [cycle %" PRIu64 "] unexpected activity while idle", dut_.cycle()));
      }
    }
  }

  // Blocking DMA access, used outside the timed windows.
  Row dma_access(const DmaReq& req) {
    DmaReq d = req;
    d.tag = this->next_dma_tag();
    dut_.dma = d;
    bool accepted = false;
    for (uint32_t i = 0; i < 64; ++i) {
      Events ev = dut_.clock();
      accepted |= ev.dma_acc;
      if (ev.dma_rsp) {
        if (ev.dma_rsp_tag != d.tag) {
          this->fail(strf("FAIL [cycle %" PRIu64 "] DMA response tag 0x%x, expected 0x%x",
                          dut_.cycle(), ev.dma_rsp_tag, d.tag));
        }
        return ev.dma_rsp_data;
      }
      if (accepted && d.rw) {
        return Row{};
      }
    }
    this->fail(strf("FAIL [cycle %" PRIu64 "] DMA %s row 0x%x timed out (accepted=%d)",
                    dut_.cycle(), d.rw ? "write" : "read", d.row, accepted));
    dut_.dma.reset();
    return Row{};
  }

  Row dma_read(uint32_t row) {
    DmaReq d;
    d.rw = false;
    d.row = row;
    return this->dma_access(d);
  }

  void dma_write(uint32_t row, const Row& data) {
    DmaReq d;
    d.rw = true;
    d.row = row;
    d.be.fill(FULL_BE);
    d.data = data;
    this->dma_access(d);
  }

  // Blocking LSU access; returns the response word (reads only).
  std::optional<uint64_t> lsu_access(uint32_t lane, LsuReq req) {
    req.tag = this->next_tag();
    dut_.lane[lane] = req;
    bool expects_rsp = !req.rw || acks_stores(lane);
    bool accepted = false;
    for (uint32_t i = 0; i < 64; ++i) {
      Events ev = dut_.clock();
      accepted |= ev.lsu_acc[lane];
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        if (!ev.lsu_rsp[l]) {
          continue;
        }
        if (l != lane || ev.rsp_tag[l] != req.tag) {
          this->fail(strf("FAIL [cycle %" PRIu64 "] stray LSU response lane=%u tag=0x%x (waiting lane=%u tag=0x%x)",
                          dut_.cycle(), l, ev.rsp_tag[l], lane, req.tag));
          continue;
        }
        return ev.rsp_data[l];
      }
      if (accepted && !expects_rsp) {
        return std::nullopt;
      }
    }
    this->fail(strf("FAIL [cycle %" PRIu64 "] LSU request lane=%u addr=0x%x timed out (accepted=%d)",
                    dut_.cycle(), lane, req.addr, accepted));
    dut_.lane[lane].reset();
    return std::nullopt;
  }

  bool check_rows(const char* ctx, const std::vector<uint32_t>& rows) {
    bool ok = true;
    for (uint32_t row : rows) {
      Row got = this->dma_read(row);
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        if (got[b] != this->gold(row, b)) {
          this->fail(strf("FAIL %s [cycle %" PRIu64 "] word (row 0x%x, bank %u): expected %s, got %s",
                          ctx, dut_.cycle(), row, b, hex(this->gold(row, b)).c_str(), hex(got[b]).c_str()));
          ok = false;
        }
      }
    }
    return ok;
  }

  ///////////////////////////////////////////////////////////////////////////

  void init_memory() {
    for (uint32_t row = 0; row < NUM_ROWS; ++row) {
      Row data;
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        data[b] = this->rand_word();
        this->gold(row, b) = data[b];
      }
      this->dma_write(row, data);
    }
    this->idle(2);
    std::vector<uint32_t> rows;
    for (uint32_t row = 0; row < NUM_ROWS; row += std::max(1u, NUM_ROWS / 16)) {
      rows.push_back(row);
    }
    this->check_rows("init", rows);
  }

  // Finds the cycle a request reaches its bank, relative to its LSU port
  // handshake, without looking inside the DUT: a plain store and a DMA read
  // of the same word are swept against each other, and the first DMA offset
  // that observes the new value is one past the store's SRAM write.
  bool calibrate() {
    const uint32_t lane = 0;
    const uint32_t row = 3 % NUM_ROWS;
    const uint32_t bank = NUM_BANKS - 1;
    int first_new = -1;
    std::string trace;
    for (int off = 0; off <= 8; ++off) {
      uint64_t old_val = this->gold(row, bank);
      uint64_t new_val = (~old_val) & WORD_MASK;
      LsuReq st;
      st.rw = true;
      st.byteen = FULL_BE;
      st.addr = word_addr(row, bank);
      st.data = new_val;
      st.tag = this->next_tag();
      DmaReq rd;
      rd.rw = false;
      rd.row = row;
      rd.tag = this->next_dma_tag();
      dut_.lane[lane] = st;
      uint64_t c0 = dut_.cycle();
      std::optional<uint64_t> seen;
      bool st_acc_at_0 = false;
      for (uint32_t i = 0; i < 32 && !seen; ++i) {
        int rel = static_cast<int>(dut_.cycle() - c0);
        if (rel == off) {
          dut_.dma = rd;
        }
        Events ev = dut_.clock();
        if (ev.lsu_acc[lane] && rel == 0) {
          st_acc_at_0 = true;
        }
        if (ev.dma_rsp) {
          seen = ev.dma_rsp_data[bank];
        }
      }
      this->gold(row, bank) = new_val;
      this->idle(2);
      if (!st_acc_at_0 || !seen) {
        this->fail(strf("FAIL calibrate: store accepted at port cycle 0: %d, DMA response: %d",
                        st_acc_at_0, seen.has_value()));
        return false;
      }
      bool is_new = (*seen == new_val);
      if (!is_new && *seen != old_val) {
        this->fail(strf("FAIL calibrate: DMA read at +%d returned %s, neither old %s nor new %s",
                        off, hex(*seen).c_str(), hex(old_val).c_str(), hex(new_val).c_str()));
        return false;
      }
      trace += is_new ? "N" : "o";
      if (is_new && first_new < 0) {
        first_new = off;
      }
      if (!is_new && first_new >= 0) {
        this->fail(strf("FAIL calibrate: store visibility not monotonic over DMA offsets (%s)", trace.c_str()));
        return false;
      }
    }
    if (first_new < 1) {
      this->fail(strf("FAIL calibrate: could not locate the bank acceptance cycle (%s)", trace.c_str()));
      return false;
    }
    bank_lat_ = first_new - 1;
    std::printf("calibrate: store visibility over DMA offsets 0..8 = %s -> request reaches its bank at port+%d\n",
                trace.c_str(), bank_lat_);

    LsuReq ld;
    ld.byteen = FULL_BE;
    ld.addr = word_addr(row, bank);
    ld.tag = this->next_tag();
    dut_.lane[lane] = ld;
    int ld_acc = -1, ld_rsp = -1;
    for (int rel = 0; rel < 32 && ld_rsp < 0; ++rel) {
      Events ev = dut_.clock();
      if (ev.lsu_acc[lane]) {
        ld_acc = rel;
      }
      if (ev.lsu_rsp[lane] && ev.rsp_tag[lane] == ld.tag) {
        ld_rsp = rel;
        if (ev.rsp_data[lane] != this->gold(row, bank)) {
          this->fail(strf("FAIL calibrate: load returned %s, expected %s",
                          hex(ev.rsp_data[lane]).c_str(), hex(this->gold(row, bank)).c_str()));
          return false;
        }
      }
    }
    this->idle(2);
    if (ld_acc != 0 || ld_rsp <= ld_acc + bank_lat_) {
      this->fail(strf("FAIL calibrate: load accepted at port +%d, responded at +%d", ld_acc, ld_rsp));
      return false;
    }
    rsp_lat_ = ld_rsp - ld_acc - bank_lat_;
    std::printf("calibrate: load responds at port+%d -> response leaves the bank %d cycle(s) after acceptance\n",
                ld_rsp, rsp_lat_);
    return true;
  }

  ///////////////////////////////////////////////////////////////////////////
  // Store-ack input (the DLMEM ingress): a store's acknowledgement must not
  // come before its write is visible, it keeps its place among the input's
  // responses, it breaks other harts' reservations, and atomics run as on any
  // other input.

  // Issues `first` on `lane_a` at relative cycle 0 and `second` on `lane_b`
  // from relative cycle `off` on; returns the handshake cycles and data.
  struct PairRun {
    int acc[2] = {-1, -1};
    int rsp[2] = {-1, -1};
    uint64_t data[2] = {0, 0};
    int rsp_order = 0; // 1: first answered before second, 2: second before first
    bool ok = true;
  };

  PairRun pair_run(uint32_t lane_a, LsuReq first, uint32_t lane_b, LsuReq second, int off, uint32_t rsp_ready_pct) {
    PairRun pr;
    first.tag = this->next_tag();
    second.tag = this->next_tag();
    const LsuReq* reqs[2] = {&first, &second};
    const uint32_t lanes[2] = {lane_a, lane_b};
    bool sent = false;
    dut_.lane[lane_a] = first;
    for (int rel = 0; rel < 128 && (pr.rsp[0] < 0 || pr.rsp[1] < 0); ++rel) {
      if (!sent && rel >= off && !dut_.lane[lane_b]) {
        dut_.lane[lane_b] = second;
        sent = true;
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        dut_.rsp_ready[l] = this->rand_below(100) < rsp_ready_pct;
      }
      Events ev = dut_.clock();
      // On a shared lane the second request is staged only once the first left it.
      if (ev.lsu_acc[lane_a] && pr.acc[0] < 0) {
        pr.acc[0] = rel;
      } else if (sent && ev.lsu_acc[lane_b] && pr.acc[1] < 0) {
        pr.acc[1] = rel;
      }
      if (lane_a != lane_b && sent && ev.lsu_acc[lane_b] && pr.acc[1] < 0) {
        pr.acc[1] = rel;
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        if (!ev.lsu_rsp[l]) {
          continue;
        }
        int k = -1;
        for (int j = 0; j < 2; ++j) {
          if (l == lanes[j] && ev.rsp_tag[l] == reqs[j]->tag && pr.rsp[j] < 0) {
            k = j;
          }
        }
        if (k < 0) {
          this->fail(strf("FAIL store-ack [cycle %" PRIu64 "] stray or repeated response lane=%u tag=0x%x",
                          dut_.cycle(), l, ev.rsp_tag[l]));
          pr.ok = false;
          continue;
        }
        pr.rsp[k] = rel;
        pr.data[k] = ev.rsp_data[l];
        if (pr.rsp_order == 0) {
          pr.rsp_order = k + 1;
        }
      }
    }
    dut_.rsp_ready.fill(true);
    if (pr.rsp[0] < 0 || pr.rsp[1] < 0) {
      this->fail(strf("FAIL store-ack [cycle %" PRIu64 "] responses did not arrive within 128 cycles "
                      "(lane %u: acc %d rsp %d, lane %u: acc %d rsp %d)",
                      dut_.cycle(), lane_a, pr.acc[0], pr.rsp[0], lane_b, pr.acc[1], pr.rsp[1]));
      dut_.lane[lane_a].reset();
      dut_.lane[lane_b].reset();
      pr.ok = false;
    }
    this->idle(2);
    return pr;
  }

  void store_ack() {
    const uint32_t ack = static_cast<uint32_t>(ACK_LANE);
    const uint32_t other = 0;
    const int ld_lat = bank_lat_ + rsp_lat_;
    const uint32_t before = failures_;

    // A load on another input to the stored word, issued at every offset
    // around the acknowledgement.
    uint32_t runs = 0, at_ack = 0, after_ack = 0;
    std::vector<uint32_t> rows;
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (int off = 0; off <= ld_lat + 2; ++off) {
        uint32_t row = this->rand_below(NUM_ROWS);
        rows.push_back(row);
        uint64_t old_val = this->gold(row, bank);
        uint64_t new_val;
        do {
          new_val = this->rand_word();
        } while (new_val == old_val);
        LsuReq st;
        st.rw = true;
        st.byteen = FULL_BE;
        st.addr = word_addr(row, bank);
        st.data = new_val;
        LsuReq ld;
        ld.byteen = FULL_BE;
        ld.addr = st.addr;
        PairRun pr = this->pair_run(ack, st, other, ld, off, 100);
        this->gold(row, bank) = new_val;
        ++runs;
        if (!pr.ok) {
          continue;
        }
        const int st_acc = pr.acc[0], ack_rel = pr.rsp[0], ld_acc = pr.acc[1];
        std::string ctx = strf("bank %u, load from +%d: store accepted +%d, ack +%d, load accepted +%d, load returned %s",
                               bank, off, st_acc, ack_rel, ld_acc, hex(pr.data[1]).c_str());
        if (ack_rel - st_acc != ld_lat) {
          this->fail(strf("FAIL store-ack %s: the ack took %d cycle(s) from acceptance, a load response takes %d",
                          ctx.c_str(), ack_rel - st_acc, ld_lat));
        }
        if (ld_acc >= ack_rel && pr.data[1] != new_val) {
          this->fail(strf("FAIL store-ack %s: a load accepted in or after the ack cycle missed the store (expected %s)",
                          ctx.c_str(), hex(new_val).c_str()));
        }
        if (ld_acc < st_acc && pr.data[1] != old_val) {
          this->fail(strf("FAIL store-ack %s: a load accepted before the store saw it (expected %s)",
                          ctx.c_str(), hex(old_val).c_str()));
        }
        if (pr.data[1] != old_val && pr.data[1] != new_val) {
          this->fail(strf("FAIL store-ack %s: neither old %s nor new %s",
                          ctx.c_str(), hex(old_val).c_str(), hex(new_val).c_str()));
        }
        at_ack += (ld_acc == ack_rel);
        after_ack += (ld_acc > ack_rel);
      }
    }
    if (at_ack == 0) {
      this->fail("FAIL store-ack: no load was accepted in the ack cycle, so commit-before-ack went untested");
    }
    std::printf("store-ack: ack at store acceptance +%d (= load response latency); %u runs, loads accepted in the "
                "ack cycle %u, after it %u\n", ld_lat, runs, at_ack, after_ack);

    // A load from the same input, right behind the store and under response
    // back-pressure: the ack comes first and the load sees the store.
    uint32_t order_runs = 0;
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (uint32_t pct : {100u, 50u, 20u}) {
        uint32_t row = this->rand_below(NUM_ROWS);
        rows.push_back(row);
        uint64_t new_val = this->rand_word();
        uint32_t be = (pct == 100) ? FULL_BE : (1 + this->rand_below(FULL_BE));
        LsuReq st;
        st.rw = true;
        st.byteen = be;
        st.addr = word_addr(row, bank);
        st.data = new_val;
        LsuReq ld;
        ld.byteen = FULL_BE;
        ld.addr = st.addr;
        uint64_t expect = merge(this->gold(row, bank), new_val, be);
        PairRun pr = this->pair_run(ack, st, ack, ld, 1, pct);
        this->gold(row, bank) = expect;
        ++order_runs;
        if (!pr.ok) {
          continue;
        }
        if (pr.rsp_order != 1) {
          this->fail(strf("FAIL store-ack bank %u, rsp ready %u%%: the load behind the store was answered before "
                          "the store's ack (ack +%d, load +%d)", bank, pct, pr.rsp[0], pr.rsp[1]));
        }
        if (pr.data[1] != expect) {
          this->fail(strf("FAIL store-ack bank %u, rsp ready %u%%: the load behind the store returned %s, expected %s",
                          bank, pct, hex(pr.data[1]).c_str(), hex(expect).c_str()));
        }
      }
    }
    std::printf("store-ack: store then load on the ack input, %u runs under response back-pressure\n", order_runs);

    if (AMO_EN) {
      // LR on another hart's input; an acknowledged store to the reserved
      // word must fail its SC, one to another word of the bank must not.
      uint32_t lr_runs = 0;
      for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
        for (int width = 0; width < NUM_WIDTHS; ++width) {
          for (int hit = 0; hit < 2; ++hit) {
            uint32_t row = this->rand_below(NUM_ROWS);
            uint32_t row2 = this->other_row(row);
            rows.push_back(row);
            rows.push_back(row2);
            uint32_t be = lane_byteen(width);
            LsuReq lr;
            lr.amo = true;
            lr.op = AMO_LR;
            lr.byteen = be;
            lr.addr = word_addr(row, bank);
            std::optional<uint64_t> lr_ret = this->lsu_access(other, lr);
            if (lr_ret && *lr_ret != this->gold(row, bank)) {
              this->fail(strf("FAIL store-ack LR bank %u %s: returned %s, expected %s", bank, width_name(width),
                              hex(*lr_ret).c_str(), hex(this->gold(row, bank)).c_str()));
            }
            uint32_t st_row = hit ? row : row2;
            LsuReq st;
            st.rw = true;
            st.byteen = FULL_BE;
            st.addr = word_addr(st_row, bank);
            st.data = this->rand_word();
            this->lsu_access(ack, st);
            this->gold(st_row, bank) = st.data;
            LsuReq sc;
            sc.amo = true;
            sc.op = AMO_SC;
            sc.byteen = be;
            sc.addr = word_addr(row, bank);
            sc.data = lane_position(width, this->rand_word());
            std::optional<uint64_t> sc_ret = this->lsu_access(other, sc);
            bool want_fail = (hit != 0);
            if (!sc_ret || *sc_ret != sc_rsp_word(sc, want_fail)) {
              this->fail(strf("FAIL store-ack SC bank %u %s after an ack-input store to %s: returned %s, expected %s",
                              bank, width_name(width), hit ? "the reserved word" : "another word of the bank",
                              sc_ret ? hex(*sc_ret).c_str() : "nothing", hex(sc_rsp_word(sc, want_fail)).c_str()));
            }
            if (!want_fail) {
              this->gold(row, bank) = merge(this->gold(row, bank), sc.data, be);
            }
            ++lr_runs;
          }
        }
      }
      std::printf("store-ack: LR/SC across an ack-input store, %u runs\n", lr_runs);

      // Every private atomic on the ack input, checked against the model.
      static const uint8_t ops[] = {AMO_ADD, AMO_SWAP, AMO_XOR, AMO_OR, AMO_AND, AMO_MIN, AMO_MAX, AMO_CAS};
      const uint32_t num_ops = CAS_EN ? 8 : 7;
      uint32_t amo_runs = 0;
      for (uint32_t k = 0; k < num_ops; ++k) {
        for (int width = 0; width < NUM_WIDTHS; ++width) {
          for (int uns = 0; uns < ((ops[k] == AMO_MIN || ops[k] == AMO_MAX) ? 2 : 1); ++uns) {
            uint32_t row = this->rand_below(NUM_ROWS);
            uint32_t bank = this->rand_below(NUM_BANKS);
            rows.push_back(row);
            LsuReq r;
            r.amo = true;
            r.op = ops[k];
            r.uns = (uns != 0);
            r.byteen = lane_byteen(width);
            r.addr = word_addr(row, bank);
            r.data = lane_position(width, this->rand_word());
            uint64_t& g = this->gold(row, bank);
            if (r.op == AMO_CAS) {
              uint64_t cur = (XLEN == 64 && width == W_HI) ? (g >> 32) : g;
              if (XLEN == 32 || width != W_D) {
                cur &= MASK32;
              }
              r.cmp = (amo_runs & 1) ? cur : this->rand_word();
            }
            std::optional<uint64_t> ret = this->lsu_access(ack, r);
            if (!ret || *ret != g) {
              this->fail(strf("FAIL store-ack %s.%s on the ack input (row 0x%x, bank %u): returned %s, expected %s",
                              amo_name(r.op, r.uns), width_name(width), row, bank,
                              ret ? hex(*ret).c_str() : "nothing", hex(g).c_str()));
            }
            g = amo_new_word(r, g);
            ++amo_runs;
          }
        }
      }
      std::printf("store-ack: %u atomics on the ack input\n", amo_runs);
    }

    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    this->check_rows("store-ack", rows);
    std::printf("store-ack: %s\n\n", (failures_ == before) ? "ok" : "FAILED");
  }

  ///////////////////////////////////////////////////////////////////////////
  // Store commits: a plain store of an LSU input is reported on its bank's
  // commit port, with the warp of its hart id, the cycle after the SRAM write,
  // as every access of an LSU input is (see the access commits below); the
  // ack input and DMA writes are never reported. With the store written at
  // bank cycle T, the report is valid at T+1, which is bank_lat_ + 1 cycles
  // after an uncontended port handshake and rsp_lat_ - 1 cycles before the
  // acknowledgement the ack input would get for the same store.

  struct Issue {
    int      at = 0;          // first relative cycle it is offered
    bool     dma = false;
    uint32_t lane = 0;
    LsuReq   req;
    DmaReq   dreq;
  };

  struct CommitEv {
    int      rel;
    uint32_t bank;
    uint32_t wid;
  };

#ifdef VX_CFG_EXT_MBAR_ENABLE
  struct MbarEvRel {
    int      rel;
    uint32_t bank;
    bool     flip;
    uint32_t wid;
    uint32_t row;
  };
#endif

  struct ScnRun {
    std::vector<int> acc;
    std::vector<int> rsp;
    std::vector<uint64_t> data;
    std::vector<CommitEv> commits;
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    std::vector<MbarEvRel> mbar;
  #endif
    bool ok = true;
  };

  // Offers the issues in order per port, each from its cycle on; runs until
  // every response arrived, then `tail` more cycles.
  ScnRun run_issues(std::vector<Issue> issues, bool rsp_ready, int tail) {
    ScnRun sr;
    const size_t n = issues.size();
    sr.acc.assign(n, -1);
    sr.rsp.assign(n, -1);
    sr.data.assign(n, 0);
    std::vector<bool> staged(n, false), want_rsp(n, false);
    for (size_t k = 0; k < n; ++k) {
      Issue& is = issues[k];
      if (is.dma) {
        is.dreq.tag = this->next_dma_tag();
        want_rsp[k] = !is.dreq.rw;
      } else {
        is.req.tag = this->next_tag();
        want_rsp[k] = !is.req.rw || acks_stores(is.lane);
      }
    }
    dut_.rsp_ready.fill(rsp_ready);
    int done_at = -1;
    for (int rel = 0; rel < 256; ++rel) {
      for (size_t k = 0; k < n; ++k) {
        const Issue& is = issues[k];
        if (staged[k] || rel < is.at) {
          continue;
        }
        bool before_pending = false;
        for (size_t j = 0; j < k; ++j) {
          before_pending |= !staged[j] && (issues[j].dma == is.dma) && (is.dma || issues[j].lane == is.lane);
        }
        if (before_pending) {
          continue;
        }
        if (is.dma && !dut_.dma) {
          dut_.dma = is.dreq;
          staged[k] = true;
        } else if (!is.dma && !dut_.lane[is.lane]) {
          dut_.lane[is.lane] = is.req;
          staged[k] = true;
        }
      }
      Events ev = dut_.clock();
      for (size_t k = 0; k < n; ++k) {
        const Issue& is = issues[k];
        if (!staged[k] || sr.acc[k] >= 0) {
          continue;
        }
        bool acc = is.dma ? (ev.dma_acc && !dut_.dma) : (ev.lsu_acc[is.lane] && ev.acc_req[is.lane].tag == is.req.tag);
        if (acc) {
          sr.acc[k] = rel;
        }
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        if (!ev.lsu_rsp[l]) {
          continue;
        }
        bool found = false;
        for (size_t k = 0; k < n; ++k) {
          if (!issues[k].dma && issues[k].lane == l && issues[k].req.tag == ev.rsp_tag[l] && sr.rsp[k] < 0) {
            sr.rsp[k] = rel;
            sr.data[k] = ev.rsp_data[l];
            found = true;
          }
        }
        if (!found) {
          this->fail(strf("FAIL store-commit [cycle %" PRIu64 "] stray response lane=%u tag=0x%x",
                          dut_.cycle(), l, ev.rsp_tag[l]));
          sr.ok = false;
        }
      }
      if (ev.dma_rsp) {
        for (size_t k = 0; k < n; ++k) {
          if (issues[k].dma && want_rsp[k] && sr.rsp[k] < 0 && issues[k].dreq.tag == ev.dma_rsp_tag) {
            sr.rsp[k] = rel;
            break;
          }
        }
      }
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        if (ev.commit[b]) {
          sr.commits.push_back({rel, b, ev.commit_wid[b]});
        }
      #ifdef VX_CFG_EXT_MBAR_ENABLE
        if (ev.mbar_evt[b]) {
          sr.mbar.push_back({rel, b, ev.mbar_flip[b], ev.mbar_wid[b], ev.mbar_row[b]});
        }
      #endif
      }
      bool all = true;
      for (size_t k = 0; k < n; ++k) {
        all = all && sr.acc[k] >= 0 && (!want_rsp[k] || sr.rsp[k] >= 0);
      }
      if (all && done_at < 0) {
        done_at = rel;
      }
      if (done_at >= 0 && rel >= done_at + tail) {
        break;
      }
    }
    dut_.rsp_ready.fill(true);
    if (done_at < 0) {
      this->fail(strf("FAIL store-commit [cycle %" PRIu64 "] scenario did not complete within 256 cycles",
                      dut_.cycle()));
      dut_.dma.reset();
      for (auto& l : dut_.lane) {
        l.reset();
      }
      sr.ok = false;
    }
    if (!rsp_ready) {
      // Let anything held by response back-pressure drain.
      for (uint32_t i = 0; i < 16; ++i) {
        dut_.clock();
      }
    }
    this->idle(2);
    return sr;
  }

  Issue lsu_issue(int at, uint32_t lane, const LsuReq& r) {
    Issue is;
    is.at = at;
    is.lane = lane;
    is.req = r;
    return is;
  }

  LsuReq plain_store(uint32_t row, uint32_t bank, uint32_t wid, uint32_t be) {
    LsuReq st;
    st.rw = true;
    st.byteen = be;
    st.addr = word_addr(row, bank);
    st.data = this->rand_word();
    st.wid = static_cast<int>(wid);
    this->gold(row, bank) = merge(this->gold(row, bank), st.data, be);
    return st;
  }

  void check_commit_errors(const char* ctx) {
    for (const std::string& e : dut_.commit_errors) {
      this->fail(strf("FAIL %s: %s", ctx, e.c_str()));
    }
    dut_.commit_errors.clear();
  }

#ifdef VX_CFG_EXT_MBAR_ENABLE
  void check_mbar_errors(const char* ctx) {
    dut_.mbar_drained(ctx);
    for (const std::string& e : dut_.mbar_errors) {
      this->fail(strf("FAIL mbar event stream %s: %s", ctx, e.c_str()));
    }
    dut_.mbar_errors.clear();
  }
#endif

  void store_commit() {
    const uint32_t before = failures_;
    const int commit_lat = bank_lat_ + 1;
    std::vector<uint32_t> rows;

    // One store per LSU input and bank: one report, bank_lat_ + 1 after the handshake, with its warp.
    uint32_t single_runs = 0;
    for (uint32_t lane = 0; lane < NUM_LSU_REQS; ++lane) {
      for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
        uint32_t row = this->rand_below(NUM_ROWS);
        uint32_t wid = this->rand_below(NUM_WARPS);
        rows.push_back(row);
        LsuReq st = this->plain_store(row, bank, wid, 1 + this->rand_below(FULL_BE));
        ScnRun sr = this->run_issues({this->lsu_issue(0, lane, st)}, true, 6);
        ++single_runs;
        if (!sr.ok) {
          continue;
        }
        if (sr.commits.size() != 1 || sr.commits[0].rel != sr.acc[0] + commit_lat
         || sr.commits[0].bank != bank || sr.commits[0].wid != wid) {
          std::string got;
          for (const auto& c : sr.commits) {
            got += strf(" {+%d bank %u wid %u}", c.rel, c.bank, c.wid);
          }
          this->fail(strf("FAIL store-commit lane %u bank %u wid %u: store accepted +%d, expected one report at +%d, "
                          "got%s", lane, bank, wid, sr.acc[0], sr.acc[0] + commit_lat,
                          got.empty() ? " none" : got.c_str()));
        }
      }
    }
    std::printf("store-commit: report at store handshake +%d (bank write +%d, then one register); %u single stores\n",
                commit_lat, bank_lat_, single_runs);

    // A load of the stored word from another input at every offset around the
    // store: handshaken after the store it returns the store's data, before it
    // the old data; a load reaching the bank in the report cycle T+1 (stalled
    // by the read-during-write hazard) sees the store.
    const uint32_t ld_lane = (NUM_LSU_REQS > 1) ? 1 : static_cast<uint32_t>(ACK_LANE);
    const int ld_lat = bank_lat_ + rsp_lat_;
    uint32_t sweep_runs = 0, at_report = 0, after_store = 0;
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (int off = 0; off <= ld_lat + 2; ++off) {
        uint32_t row = this->rand_below(NUM_ROWS);
        rows.push_back(row);
        uint64_t old_val = this->gold(row, bank);
        uint32_t wid = this->rand_below(NUM_WARPS);
        LsuReq st = this->plain_store(row, bank, wid, FULL_BE);
        if (st.data == old_val) {
          st.data = (~old_val) & WORD_MASK;
          this->gold(row, bank) = st.data;
        }
        LsuReq ld;
        ld.byteen = FULL_BE;
        ld.addr = st.addr;
        // The load reports too, under another warp.
        ld.wid = static_cast<int>((wid + 1) % NUM_WARPS);
        ScnRun sr = this->run_issues({this->lsu_issue(0, 0, st), this->lsu_issue(off, ld_lane, ld)}, true, 6);
        ++sweep_runs;
        if (!sr.ok) {
          continue;
        }
        const int st_acc = sr.acc[0], ld_acc = sr.acc[1];
        std::vector<CommitEv> st_reps;
        for (const CommitEv& c : sr.commits) {
          if (c.wid == wid) {
            st_reps.push_back(c);
          }
        }
        const int rep = st_reps.empty() ? -1 : st_reps[0].rel;
        std::string ctx = strf("bank %u, load from +%d: store accepted +%d, reported +%d, load accepted +%d, "
                               "returned %s", bank, off, st_acc, rep, ld_acc, hex(sr.data[1]).c_str());
        const size_t want_reps = (reports_commits(ld_lane, ld.counted) && NUM_WARPS > 1) ? 2 : 1;
        if (st_reps.size() != 1 || rep != st_acc + commit_lat || sr.commits.size() != want_reps) {
          this->fail(strf("FAIL store-commit %s: expected one report of warp %u at +%d", ctx.c_str(), wid,
                          st_acc + commit_lat));
        }
        if (ld_acc > st_acc && sr.data[1] != st.data) {
          this->fail(strf("FAIL store-commit %s: a load handshaken after the store missed it (expected %s)",
                          ctx.c_str(), hex(st.data).c_str()));
        }
        if (ld_acc < st_acc && sr.data[1] != old_val) {
          this->fail(strf("FAIL store-commit %s: a load handshaken before the store saw it (expected %s)",
                          ctx.c_str(), hex(old_val).c_str()));
        }
        after_store += (ld_acc > st_acc);
        at_report += (ld_acc + bank_lat_ == rep);
      }
    }
    if (at_report == 0 || after_store == 0) {
      this->fail("FAIL store-commit: no load reached the bank in the report cycle, or none followed the store");
    }
    std::printf("store-commit: load sweep %u runs, %u loads after the store, %u reaching the bank in the report "
                "cycle\n", sweep_runs, after_store, at_report);

    // The ack input's stores and DMA writes are not reported.
    uint32_t silent_runs = 0;
    auto expect_silent = [&](const char* what, std::vector<Issue> issues) {
      ScnRun sr = this->run_issues(std::move(issues), true, 8);
      ++silent_runs;
      if (sr.ok && !sr.commits.empty()) {
        this->fail(strf("FAIL store-commit: %s was reported (bank %u, +%d)", what, sr.commits[0].bank,
                        sr.commits[0].rel));
      }
    };
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      uint32_t row = this->rand_below(NUM_ROWS);
      rows.push_back(row);
      LsuReq st = this->plain_store(row, bank, 0, FULL_BE);
      expect_silent("a store of the ack input", {this->lsu_issue(0, ACK_LANE, st)});
      DmaReq dw;
      dw.rw = true;
      dw.row = row;
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        dw.be[b] = FULL_BE;
        dw.data[b] = this->rand_word();
        this->gold(row, b) = dw.data[b];
      }
      Issue di;
      di.dma = true;
      di.dreq = dw;
      expect_silent("a DMA write", {di});
    }
    std::printf("store-commit: %u runs of ack-input stores and DMA writes without a report\n", silent_runs);

    // Stalled stores: the report moves with the SRAM write. Each scenario runs
    // twice, the victim store once on an LSU input (its report) and once on the
    // ack input (its acknowledgement), which must stay rsp_lat_ - 1 apart.
    std::vector<uint32_t> aux_a, aux_b;
    for (uint32_t l = 1; l < NUM_REQS; ++l) {
      aux_a.push_back(l);
    }
    for (uint32_t l = 0; l < NUM_LSU_REQS; ++l) {
      aux_b.push_back(l);
    }
    enum { ST_DMA, ST_AMO, ST_RDW, NUM_ST };
    static const char* st_names[NUM_ST] = {"DMA write stream", "atomic owning the bank", "read-during-write stall"};
    uint32_t twin_runs = 0;
    for (int kind = 0; kind < NUM_ST; ++kind) {
      if (kind == ST_AMO && !AMO_EN) {
        continue;
      }
      if (aux_b.size() < 2 && kind == ST_RDW) {
        continue;
      }
      for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
        int delay[2] = {0, 0}, acc[2] = {0, 0}, evt[2] = {0, 0};
        int offered = 1;
        bool ok = true;
        uint32_t row = this->rand_below(NUM_ROWS);
        uint32_t row2 = this->other_row(row);
        rows.push_back(row);
        rows.push_back(row2);
        uint32_t wid = this->rand_below(NUM_WARPS);
        // The aux requests report too; their warp tells them from the victim.
        const int aux_wid = static_cast<int>((wid + 1) % NUM_WARPS);
        for (int twin = 0; twin < 2; ++twin) {
          const std::vector<uint32_t>& aux = twin ? aux_b : aux_a;
          const uint32_t victim = twin ? static_cast<uint32_t>(ACK_LANE) : 0;
          std::vector<Issue> issues;
          int victim_at = 1;
          if (kind == ST_DMA) {
            for (int k = 0; k < 6; ++k) {
              Issue di;
              di.dma = true;
              di.at = 0;
              di.dreq.rw = true;
              di.dreq.row = row2;
              di.dreq.be.fill(0);
              di.dreq.be[bank] = FULL_BE;
              di.dreq.data[bank] = this->rand_word();
              this->gold(row2, bank) = di.dreq.data[bank];
              issues.push_back(di);
            }
          } else if (kind == ST_AMO) {
            LsuReq amo;
            amo.amo = true;
            amo.op = AMO_ADD;
            amo.byteen = FULL_BE;
            amo.addr = word_addr(row2, bank);
            amo.data = this->rand_word();
            amo.wid = aux_wid;
            this->gold(row2, bank) = amo_new_word(amo, this->gold(row2, bank));
            issues.push_back(this->lsu_issue(0, aux[0], amo));
          } else {
            LsuReq s1 = this->plain_store(row2, bank, aux_wid, FULL_BE);
            LsuReq ld;
            ld.byteen = FULL_BE;
            ld.addr = s1.addr;
            ld.wid = aux_wid;
            issues.push_back(this->lsu_issue(0, aux[0], s1));
            issues.push_back(this->lsu_issue(1, aux[1], ld));
            victim_at = 2;
          }
          LsuReq st = this->plain_store(row, bank, wid, FULL_BE);
          issues.push_back(this->lsu_issue(victim_at, victim, st));
          const size_t v = issues.size() - 1;
          offered = victim_at;
          ScnRun sr = this->run_issues(issues, true, 6);
          ++twin_runs;
          if (!sr.ok) {
            ok = false;
            break;
          }
          acc[twin] = sr.acc[v];
          if (twin == 0) {
            if (sr.commits.empty() || sr.commits.back().bank != bank
             || (sr.commits.back().wid != wid && NUM_WARPS > 1)) {
              this->fail(strf("FAIL store-commit %s bank %u: no report of the stalled store's warp %u",
                              st_names[kind], bank, wid));
              ok = false;
              break;
            }
            evt[0] = sr.commits.back().rel;
          } else {
            evt[1] = sr.rsp[v];
            if (NUM_WARPS > 1 && !sr.commits.empty() && sr.commits.back().wid == wid
             && sr.commits.back().rel >= sr.acc[v]) {
              this->fail(strf("FAIL store-commit %s bank %u: the ack input's store was reported",
                              st_names[kind], bank));
            }
          }
          delay[twin] = evt[twin] - victim_at;
        }
        if (!ok) {
          continue;
        }
        std::string ctx = strf("%s bank %u: store offered +%d; LSU input store accepted +%d, reported +%d; ack "
                               "input store accepted +%d, acknowledged +%d", st_names[kind], bank, offered, acc[0],
                               evt[0], acc[1], evt[1]);
        if (acc[0] != acc[1] || evt[0] != evt[1] - (rsp_lat_ - 1)) {
          this->fail(strf("FAIL store-commit %s: expected the report %d cycle(s) before the acknowledgement",
                          ctx.c_str(), rsp_lat_ - 1));
        }
        if (delay[0] <= commit_lat) {
          this->fail(strf("FAIL store-commit %s: the stall did not delay the report", ctx.c_str()));
        }
        if (opts_.verbose || bank == 0) {
          std::printf("store-commit: %s\n", ctx.c_str());
        }
      }
    }
    std::printf("store-commit: %u twin runs, report = acknowledgement - %d under every stall\n", twin_runs,
                rsp_lat_ - 1);

    // Reports need no response credit: stores keep committing with every response port stalled.
    {
      std::vector<Issue> issues;
      uint32_t n_st = 0;
      for (int at = 0; at < 24; ++at) {
        for (uint32_t lane = 0; lane < NUM_LSU_REQS; ++lane) {
          uint32_t row = this->rand_below(NUM_ROWS);
          uint32_t bank = this->rand_below(NUM_BANKS);
          rows.push_back(row);
          issues.push_back(this->lsu_issue(at, lane, this->plain_store(row, bank, this->rand_below(NUM_WARPS),
                                                                       1 + this->rand_below(FULL_BE))));
          ++n_st;
        }
      }
      uint64_t c0 = dut_.commits;
      ScnRun sr = this->run_issues(issues, false, 6);
      if (sr.ok && sr.commits.size() != n_st) {
        this->fail(strf("FAIL store-commit: %u stores with every response port stalled, %zu reported", n_st,
                        sr.commits.size()));
      }
      std::printf("store-commit: %u stores reported with every response port stalled (%" PRIu64 " reports)\n",
                  n_st, dut_.commits - c0);
    }

    // One store per bank from distinct inputs in one cycle: every bank reports in the same cycle.
    {
      const uint32_t n = std::min(NUM_LSU_REQS, NUM_BANKS);
      std::vector<Issue> issues;
      std::vector<uint32_t> wids(n);
      for (uint32_t lane = 0; lane < n; ++lane) {
        uint32_t row = this->rand_below(NUM_ROWS);
        rows.push_back(row);
        wids[lane] = (lane + this->rand_below(NUM_WARPS)) % NUM_WARPS;
        issues.push_back(this->lsu_issue(0, lane, this->plain_store(row, lane, wids[lane], FULL_BE)));
      }
      ScnRun sr = this->run_issues(issues, true, 6);
      bool good = sr.ok && sr.commits.size() == n;
      for (uint32_t k = 0; good && k < n; ++k) {
        const CommitEv& c = sr.commits[k];
        good = (c.rel == commit_lat) && (sr.acc[c.bank] == 0) && (c.wid == wids[c.bank]);
      }
      if (!good) {
        this->fail(strf("FAIL store-commit: %u stores to %u banks in one cycle did not all report at +%d "
                        "(%zu reports)", n, n, commit_lat, sr.commits.size()));
      }
      std::printf("store-commit: %u banks reported in one cycle\n", n);
    }

    this->check_commit_errors("store-commit");
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    this->check_rows("store-commit", rows);
    std::printf("store-commit: %s\n\n", (failures_ == before) ? "ok" : "FAILED");
  }

  ///////////////////////////////////////////////////////////////////////////
  // Access commits: every access of an LSU input is reported once, on its
  // bank's commit port with its warp. A load or plain store reports the cycle
  // after the bank performs it, bank_lat_ + 1 after an uncontended handshake;
  // an atomic (LR, SC whether it fails or not, every mbarrier op) the cycle
  // after its write-back stage, bank_lat_ + 3. The ack input never reports.

  static int commit_delay(const LsuReq& r) {
    return r.amo ? 3 : 1;
  }

#ifdef VX_CFG_EXT_MBAR_ENABLE
  // The waiter event r owes, alone on an idle memory, at its exact cycle: an
  // insert in the cycle after the bank takes it, a flip in the one after that.
  void expect_mbar_event(const std::string& what, uint32_t lane, const LsuReq& r, int acc, uint64_t rsp,
                         const std::vector<MbarEvRel>& evs) {
    int owes = 0;
    if (r.amo && r.op == mbar::OP_TRY_WAIT) {
      owes = (rsp == 0) ? 1 : 0;
    } else if (r.amo && r.op >= mbar::OP_BASE && r.op != mbar::OP_TEST_WAIT) {
      owes = mbar::apply(r.op, rsp, r.data).flip ? 2 : 0;
    }
    std::string got;
    for (const auto& e : evs) {
      got += strf(" {+%d bank %u %s wid %u row 0x%x}", e.rel, e.bank, e.flip ? "flip" : "insert", e.wid, e.row);
    }
    if (owes == 0) {
      if (!evs.empty()) {
        this->fail(strf("FAIL mbar event %s lane %u: expected no event, got%s", what.c_str(), lane, got.c_str()));
      }
      return;
    }
    const int want = acc + bank_lat_ + owes;
    const bool flip = (owes == 2);
    const uint32_t row = r.addr >> BANK_BITS;
    if (evs.size() != 1 || evs[0].rel != want || evs[0].bank != req_bank(r) || evs[0].flip != flip
     || evs[0].row != row || (!flip && evs[0].wid != req_wid(lane, r))) {
      this->fail(strf("FAIL mbar event %s lane %u bank %u: accepted +%d, expected one %s (wid %u, row 0x%x) at +%d, "
                      "got%s", what.c_str(), lane, req_bank(r), acc, flip ? "flip" : "insert", req_wid(lane, r), row,
                      want, got.empty() ? " none" : got.c_str()));
    }
    ++mbar_event_checks_[owes];
  }
#endif

  // r alone on an idle memory: one report at the expected cycle, bank and
  // warp from an LSU input, none from the ack input. Returns the response.
  std::optional<uint64_t> expect_report(const std::string& what, uint32_t lane, const LsuReq& r) {
    ScnRun sr = this->run_issues({this->lsu_issue(0, lane, r)}, true, 6);
    ++commit_runs_;
    if (!sr.ok) {
      return std::nullopt;
    }
    std::string got;
    for (const auto& c : sr.commits) {
      got += strf(" {+%d bank %u wid %u}", c.rel, c.bank, c.wid);
    }
    if (!reports_commits(lane, r.counted)) {
      if (!sr.commits.empty()) {
        this->fail(strf("FAIL commit %s on lane %u (%s) was reported:%s", what.c_str(), lane,
                        acks_stores(lane) ? "the ack input" : "not counted by the translation", got.c_str()));
      }
    } else {
      const int want = sr.acc[0] + bank_lat_ + commit_delay(r);
      const uint32_t wid = req_wid(lane, r);
      if (sr.commits.size() != 1 || sr.commits[0].rel != want || sr.commits[0].bank != req_bank(r)
       || sr.commits[0].wid != wid) {
        this->fail(strf("FAIL commit %s lane %u bank %u wid %u: accepted +%d, expected one report at +%d, got%s",
                        what.c_str(), lane, req_bank(r), wid, sr.acc[0], want, got.empty() ? " none" : got.c_str()));
      }
    }
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    this->expect_mbar_event(what, lane, r, sr.acc[0], sr.data[0], sr.mbar);
  #endif
    if (r.rw && !acks_stores(lane)) {
      return std::nullopt;
    }
    return sr.data[0];
  }

  void check_ret(const std::string& what, std::optional<uint64_t> got, uint64_t want) {
    if (!got || *got != want) {
      this->fail(strf("FAIL %s: returned %s, expected %s", what.c_str(), got ? hex(*got).c_str() : "nothing",
                      hex(want).c_str()));
    }
  }

  void commit() {
    const uint32_t before = failures_;
    const uint32_t runs0 = commit_runs_;
    std::vector<uint32_t> rows;

    // Loads, from every LSU input to every bank, and from the ack input.
    for (uint32_t lane = 0; lane < NUM_REQS; ++lane) {
      for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
        uint32_t row = this->rand_below(NUM_ROWS);
        LsuReq ld;
        ld.byteen = FULL_BE;
        ld.addr = word_addr(row, bank);
        ld.wid = static_cast<int>(this->rand_below(NUM_WARPS));
        auto ret = this->expect_report("load", lane, ld);
        this->check_ret(strf("commit load lane %u bank %u", lane, bank), ret, this->gold(row, bank));
      }
    }
    std::printf("commit: loads report at handshake +%d\n", bank_lat_ + 1);

    if (AMO_EN) {
      // Atomics, LR, SC and a failing SC, at every width, from the first and
      // last LSU input and the ack input.
      std::vector<uint32_t> lanes = {0};
      if (NUM_LSU_REQS > 1) {
        lanes.push_back(NUM_LSU_REQS - 1);
      }
      if (ACK_LANE >= 0) {
        lanes.push_back(static_cast<uint32_t>(ACK_LANE));
      }
      for (uint32_t lane : lanes) {
        for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
          for (int width = 0; width < NUM_WIDTHS; ++width) {
            uint32_t row = this->rand_below(NUM_ROWS);
            rows.push_back(row);
            uint64_t& g = this->gold(row, bank);
            const std::string at = strf("lane %u bank %u %s", lane, bank, width_name(width));
            LsuReq a;
            a.amo = true;
            a.byteen = lane_byteen(width);
            a.addr = word_addr(row, bank);
            a.wid = static_cast<int>(this->rand_below(NUM_WARPS));
            for (uint8_t op : {AMO_ADD, AMO_SWAP, AMO_CAS}) {
              if (op == AMO_CAS && !CAS_EN) {
                continue;
              }
              a.op = op;
              a.data = lane_position(width, this->rand_word());
              if (op == AMO_CAS) {
                uint64_t cur = (XLEN == 64 && width == W_HI) ? (g >> 32) : g;
                a.cmp = (XLEN == 32 || width != W_D) ? (cur & MASK32) : cur;
              }
              auto ret = this->expect_report(amo_name(op, false), lane, a);
              this->check_ret(strf("commit %s %s", amo_name(op, false), at.c_str()), ret, g);
              g = amo_new_word(a, g);
            }
            a.op = AMO_LR;
            a.data = 0;
            this->check_ret("commit LR " + at, this->expect_report("LR", lane, a), g);
            a.op = AMO_SC;
            a.data = lane_position(width, this->rand_word());
            this->check_ret("commit SC " + at, this->expect_report("SC", lane, a), sc_rsp_word(a, false));
            g = merge(g, a.data, a.byteen);
            a.data = lane_position(width, this->rand_word());
            this->check_ret("commit failing SC " + at, this->expect_report("failing SC", lane, a),
                            sc_rsp_word(a, true));
          }
        }
      }
      std::printf("commit: atomics, LR, SC and failing SC report at handshake +%d\n", bank_lat_ + 3);
    }

    // Accesses the translation did not count (the TCU metadata loads share LSU
    // input 0 and bypass it): performed, never reported.
    if (COMMIT_EN) {
      uint32_t silent = 0;
      std::vector<uint32_t> lanes = {0};
      if (NUM_LSU_REQS > 1) {
        lanes.push_back(NUM_LSU_REQS - 1);
      }
      for (uint32_t lane : lanes) {
        for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
          uint32_t row = this->rand_below(NUM_ROWS);
          rows.push_back(row);
          uint64_t& g = this->gold(row, bank);
          const std::string at = strf("uncounted lane %u bank %u", lane, bank);
          LsuReq r;
          r.counted = false;
          r.byteen = FULL_BE;
          r.addr = word_addr(row, bank);
          r.wid = static_cast<int>(this->rand_below(NUM_WARPS));
          this->check_ret("commit load " + at, this->expect_report("uncounted load", lane, r), g);
          r.rw = true;
          r.data = this->rand_word();
          this->expect_report("uncounted store", lane, r);
          g = r.data;
          r.rw = false;
          silent += 2;
          if (AMO_EN) {
            r.amo = true;
            r.op = AMO_ADD;
            r.data = this->rand_word();
            this->check_ret("commit AMOADD " + at, this->expect_report("uncounted AMOADD", lane, r), g);
            g = amo_new_word(r, g);
            r.op = AMO_LR;
            r.data = 0;
            this->check_ret("commit LR " + at, this->expect_report("uncounted LR", lane, r), g);
            r.op = AMO_SC;
            r.data = this->rand_word();
            this->check_ret("commit SC " + at, this->expect_report("uncounted SC", lane, r), sc_rsp_word(r, false));
            g = merge(g, r.data, r.byteen);
            silent += 3;
          #ifdef VX_CFG_EXT_MBAR_ENABLE
            const uint64_t token = this->rand_word();
            LsuReq w = mbar_req(AMO_MBAR_TEST_WAIT, row, bank, token, r.wid);
            w.counted = false;
            this->check_ret("commit MBAR_TEST_WAIT " + at, this->expect_report("uncounted MBAR_TEST_WAIT", lane, w),
                            mbar::apply(AMO_MBAR_TEST_WAIT, g, token).ret_word);
            ++silent;
          #endif
          }
        }
      }
      std::printf("commit: %u uncounted accesses from LSU inputs, none reported\n", silent);
    }

    // Two LSU inputs to one bank, offered together: the reports follow the
    // bank order, each its own delay after its access, an atomic holding the
    // bank for three cycles and a load or store for one.
    if (NUM_LSU_REQS > 1) {
      enum { K_LD, K_ST, K_AMO, NUM_K_ACC };
      static const char* knames[NUM_K_ACC] = {"load", "store", "atomic"};
      uint32_t pair_runs = 0;
      for (int ka = 0; ka < NUM_K_ACC; ++ka) {
        for (int kb = 0; kb < NUM_K_ACC; ++kb) {
          if ((ka == K_AMO || kb == K_AMO) && !AMO_EN) {
            continue;
          }
          for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
            uint32_t rowv[2];
            rowv[0] = this->rand_below(NUM_ROWS);
            rowv[1] = this->other_row(rowv[0]);
            const int kinds[2] = {ka, kb};
            LsuReq req[2];
            uint64_t expect[2] = {0, 0};
            const uint32_t wid0 = this->rand_below(NUM_WARPS);
            for (int j = 0; j < 2; ++j) {
              rows.push_back(rowv[j]);
              LsuReq& r = req[j];
              r.byteen = FULL_BE;
              r.addr = word_addr(rowv[j], bank);
              r.wid = static_cast<int>((wid0 + static_cast<uint32_t>(j)) % NUM_WARPS);
              uint64_t& g = this->gold(rowv[j], bank);
              if (kinds[j] == K_ST) {
                r.rw = true;
                r.data = this->rand_word();
                g = r.data;
              } else if (kinds[j] == K_AMO) {
                r.amo = true;
                r.op = AMO_ADD;
                r.data = this->rand_word();
                expect[j] = g;
                g = amo_new_word(r, g);
              } else {
                expect[j] = g;
              }
            }
            ScnRun sr = this->run_issues({this->lsu_issue(0, 0, req[0]), this->lsu_issue(0, 1, req[1])}, true, 8);
            ++pair_runs;
            if (!sr.ok) {
              continue;
            }
            const int f = (sr.acc[0] <= sr.acc[1]) ? 0 : 1;
            const int s = 1 - f;
            const int b1 = sr.acc[f] + bank_lat_;
            const int b2 = std::max(sr.acc[s] + bank_lat_, b1 + commit_delay(req[f]));
            const int rep[2] = {f == 0 ? b1 + commit_delay(req[0]) : b2 + commit_delay(req[0]),
                                f == 1 ? b1 + commit_delay(req[1]) : b2 + commit_delay(req[1])};
            const CommitEv want[2] = {{rep[f], bank, req_wid(f, req[f])}, {rep[s], bank, req_wid(s, req[s])}};
            bool good = sr.commits.size() == 2;
            for (size_t k = 0; good && k < 2; ++k) {
              good = sr.commits[k].rel == want[k].rel && sr.commits[k].bank == bank && sr.commits[k].wid == want[k].wid;
            }
            if (!good) {
              std::string got;
              for (const auto& c : sr.commits) {
                got += strf(" {+%d bank %u wid %u}", c.rel, c.bank, c.wid);
              }
              this->fail(strf("FAIL commit pair %s then %s bank %u: accepted +%d/+%d, expected reports {+%d wid %u} "
                              "{+%d wid %u}, got%s", knames[kinds[f]], knames[kinds[s]], bank, sr.acc[f], sr.acc[s],
                              want[0].rel, want[0].wid, want[1].rel, want[1].wid, got.empty() ? " none" : got.c_str()));
            }
            for (int j = 0; j < 2; ++j) {
              if (kinds[j] != K_ST && sr.data[j] != expect[j]) {
                this->fail(strf("FAIL commit pair bank %u: %s returned %s, expected %s", bank, knames[kinds[j]],
                                hex(sr.data[j]).c_str(), hex(expect[j]).c_str()));
              }
            }
          }
        }
      }
      std::printf("commit: %u two-input runs on one bank, reports in bank order\n", pair_runs);
    }

    this->check_commit_errors("commit");
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    this->check_rows("commit", rows);
    std::printf("commit: %u runs: %s\n\n", commit_runs_ - runs0, (failures_ == before) ? "ok" : "FAILED");
  }

#ifdef VX_CFG_EXT_MBAR_ENABLE
  ///////////////////////////////////////////////////////////////////////////
  // mbarrier ops: amo_op 10..15 on a full doubleword, checked against
  // mbar_ref::apply. A writing op returns the old word and writes the updated
  // object back; a wait returns 1 once the token's phase has completed and
  // writes nothing, and is no reservation-breaking store.

  static LsuReq mbar_req(uint32_t op, uint32_t row, uint32_t bank, uint64_t operand, int wid) {
    LsuReq r;
    r.amo = true;
    r.op = static_cast<uint8_t>(op);
    r.byteen = FULL_BE;
    r.addr = word_addr(row, bank);
    r.data = operand;
    r.wid = wid;
    return r;
  }

  void put_word(uint32_t row, uint32_t bank, uint64_t v) {
    DmaReq d;
    d.rw = true;
    d.row = row;
    d.be.fill(0);
    d.be[bank] = FULL_BE;
    d.data[bank] = v;
    this->dma_access(d);
    this->gold(row, bank) = v;
  }

  uint64_t get_word(uint32_t row, uint32_t bank) {
    return this->dma_read(row)[bank];
  }

  // A well-formed object with random fields.
  uint64_t rand_mbar_word() {
    uint64_t exp = 1 + this->rand_below(this->rand_below(4) ? 16 : mbar::COUNT_MAX);
    uint64_t pend = this->rand_below(static_cast<uint32_t>(exp) + 1);
    int64_t tx = 0;
    switch (this->rand_below(4)) {
    case 0: tx = 0; break;
    case 1: tx = static_cast<int64_t>(this->rand_below(512)) - 256; break;
    default: tx = static_cast<int64_t>(this->rand_below(2 * mbar::TX_MAX + 1)) - mbar::TX_MAX; break;
    }
    return mbar::pack(pend, exp, tx, this->rand_below(2));
  }

  // An operand that keeps the op on `w` free of errors.
  uint64_t rand_mbar_operand(uint32_t op, uint64_t w) {
    const int64_t tx = mbar::tx(w);
    const uint32_t pend = mbar::pending(w);
    auto upto = [&](uint64_t n) -> uint64_t {
      if (n == 0) {
        return 0;
      }
      return this->rand_below(4) ? this->rand_below(static_cast<uint32_t>(std::min<uint64_t>(n, 300)) + 1)
                                 : (this->rand_below(2) ? n : this->rand_below(static_cast<uint32_t>(n) + 1));
    };
    switch (op) {
    case mbar::OP_ARRIVE:      return this->rand_below(3) ? std::min<uint64_t>(pend, 1 + this->rand_below(2)) : upto(pend);
    case mbar::OP_ARRIVE_TX:
    case mbar::OP_EXPECT_TX:   return upto(static_cast<uint64_t>(mbar::TX_MAX - tx));
    case mbar::OP_COMPLETE_TX: return (tx > 0 && this->rand_below(2)) ? static_cast<uint64_t>(tx)
                                                                     : upto(static_cast<uint64_t>(tx + mbar::TX_MAX));
    default: {
      uint64_t tok = this->rand_word();
      return this->rand_below(2) ? tok : ((tok & ~(1ull << 63)) | (w & (1ull << 63)));
    }
    }
  }

  // A random op, the arrivals only while something is pending.
  uint32_t rand_mbar_op(uint64_t w) {
    for (;;) {
      uint32_t op = mbar::OP_BASE + this->rand_below(6);
      if ((op == mbar::OP_ARRIVE || op == mbar::OP_ARRIVE_TX) && mbar::pending(w) == 0) {
        continue;
      }
      return op;
    }
  }

  // One op on an idle memory: its returned word, its commit report and the word it leaves.
  void mbar_one(const std::string& what, uint32_t lane, uint32_t row, uint32_t bank, uint32_t op, uint64_t old,
                uint64_t operand) {
    mbar::Result res = mbar::apply(op, old, operand);
    if (res.err != mbar::ERR_NONE) {
      this->fail(strf("FAIL mbar %s: the testbench generated an erroneous op", what.c_str()));
      return;
    }
    this->put_word(row, bank, old);
    LsuReq r = mbar_req(op, row, bank, operand, static_cast<int>(this->rand_below(NUM_WARPS)));
    std::string ctx = strf("mbar %s: %s old %s operand %s lane %u", what.c_str(), amo_name(r.op, false),
                           hex(old).c_str(), hex(operand).c_str(), lane);
    auto ret = this->expect_report(ctx, lane, r);
    this->check_ret(ctx, ret, res.ret_word);
    uint64_t want = res.writes ? res.new_word : old;
    uint64_t got = this->get_word(row, bank);
    if (got != want) {
      this->fail(strf("FAIL %s: left %s, expected %s", ctx.c_str(), hex(got).c_str(), hex(want).c_str()));
    }
    this->gold(row, bank) = want;
    ++mbar_ops_[op - mbar::OP_BASE];
    mbar_flips_ += res.flip;
  }

  void mbar_directed() {
    const uint32_t before = failures_;
    using namespace mbar;
    const int64_t TM = TX_MAX;
    struct Vec { uint32_t op; uint64_t old; uint64_t operand; const char* what; };
    const Vec vecs[] = {
      {OP_ARRIVE,      init(3),                      1,      "arrive, phase open"},
      {OP_ARRIVE,      pack(1, 3, 0, 0),             1,      "last arrival flips 0->1"},
      {OP_ARRIVE,      pack(1, 3, 0, 1),             1,      "last arrival flips 1->0"},
      {OP_ARRIVE,      init(3),                      3,      "count 3 completes"},
      {OP_ARRIVE,      init(5),                      2,      "count 2 of 5"},
      {OP_ARRIVE,      init(2),                      0,      "count 0"},
      {OP_ARRIVE,      pack(1, 3, 64, 0),            1,      "last arrival, tx outstanding"},
      {OP_ARRIVE,      init(uint32_t(COUNT_MAX)),    COUNT_MAX, "count COUNT_MAX completes"},
      {OP_ARRIVE,      pack(2, 3, 0, 0) | (3ull << 61), 1,   "reserved bits written zero"},
      {OP_ARRIVE_TX,   init(2),                      128,    "arrive_tx, phase open"},
      {OP_ARRIVE_TX,   pack(1, 2, -64, 0),           64,     "arrive_tx completes a negative tx"},
      {OP_ARRIVE_TX,   pack(1, 2, 0, 1),             0,      "arrive_tx 0 bytes completes"},
      {OP_EXPECT_TX,   init(2),                      256,    "expect_tx"},
      {OP_EXPECT_TX,   pack(0, 2, -64, 1),           64,     "expect after complete flips 1->0"},
      {OP_EXPECT_TX,   pack(0, 4, 0, 0),             0,      "expect 0 on a finished phase flips"},
      {OP_EXPECT_TX,   pack(1, 1, TM - 1, 0),        1,      "tx reaches TX_MAX"},
      {OP_EXPECT_TX,   init(1),                      uint64_t(TM), "expect TX_MAX"},
      {OP_COMPLETE_TX, init(1),                      64,     "complete before expect: negative tx"},
      {OP_COMPLETE_TX, pack(0, 1, 64, 0),            64,     "complete_tx flips"},
      {OP_COMPLETE_TX, pack(1, 1, 100, 0),           100,    "complete_tx, arrival outstanding"},
      {OP_COMPLETE_TX, pack(1, 1, -(TM - 1), 0),     1,      "tx reaches -TX_MAX"},
      {OP_COMPLETE_TX, pack(1, 1, TM, 0),            uint64_t(2 * TM), "complete 2*TX_MAX"},
      {OP_TEST_WAIT,   pack(1, 2, 0, 0),             0,      "wait, phase 0 open"},
      {OP_TEST_WAIT,   pack(1, 2, 0, 0),             1ull << 63, "wait, phase 0 completed"},
      {OP_TEST_WAIT,   pack(2, 2, 5, 1),             0,      "wait, phase 1 completed"},
      {OP_TEST_WAIT,   pack(2, 2, 5, 1),             ~0ull,  "wait, phase 1 open, token all ones"},
      {OP_TEST_WAIT,   pack(0, 3, -7, 0) | (1ull << 62), 0x7fffffffffffffffull, "wait ignores token low bits"},
      {OP_TRY_WAIT,    pack(1, 2, 0, 0),             1ull << 63, "op 15 waits like op 14"},
      {OP_TRY_WAIT,    pack(1, 2, 0, 1),             1ull << 63, "op 15, phase open"},
    };
    uint32_t n = 0;
    for (const Vec& v : vecs) {
      uint32_t lane = n % NUM_REQS;
      uint32_t row = this->rand_below(NUM_ROWS);
      uint32_t bank = n % NUM_BANKS;
      this->mbar_one(v.what, lane, row, bank, v.op, v.old, v.operand);
      ++n;
    }
    std::printf("mbar: %u directed vectors\n", n);

    for (uint32_t i = 0; i < 400; ++i) {
      uint64_t w = this->rand_mbar_word();
      uint32_t op = this->rand_mbar_op(w);
      uint64_t operand = this->rand_mbar_operand(op, w);
      this->mbar_one("random vector", this->rand_below(NUM_REQS), this->rand_below(NUM_ROWS),
                     this->rand_below(NUM_BANKS), op, w, operand);
    }
    std::printf("mbar: 400 random vectors\n");

    // Reservations: a wait between LR and SC leaves the reservation, a writing op breaks it.
    uint32_t res_runs = 0;
    const uint32_t other = (NUM_LSU_REQS > 1) ? 1 : static_cast<uint32_t>(ACK_LANE);
    for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
      for (uint32_t op : {uint32_t(OP_TEST_WAIT), uint32_t(OP_TRY_WAIT), uint32_t(OP_ARRIVE), uint32_t(OP_EXPECT_TX)}) {
        uint32_t row = this->rand_below(NUM_ROWS);
        uint64_t w = init(4);
        this->put_word(row, bank, w);
        LsuReq lr = mbar_req(AMO_LR, row, bank, 0, 0);
        this->check_ret("mbar LR", this->lsu_access(0, lr), w);
        uint64_t operand = this->rand_mbar_operand(op, w);
        mbar::Result res = mbar::apply(op, w, operand);
        this->check_ret(strf("mbar %s between LR and SC", amo_name(op, false)),
                        this->lsu_access(other, mbar_req(op, row, bank, operand, 1)), res.ret_word);
        if (res.writes) {
          w = res.new_word;
        }
        LsuReq sc = mbar_req(AMO_SC, row, bank, this->rand_word(), 0);
        this->check_ret(strf("mbar SC after %s", amo_name(op, false)), this->lsu_access(0, sc),
                        sc_rsp_word(sc, res.writes));
        if (!res.writes) {
          w = sc.data;
        }
        this->gold(row, bank) = w;
        if (this->get_word(row, bank) != w) {
          this->fail(strf("FAIL mbar reservation run %s bank %u: word %s, expected %s", amo_name(op, false), bank,
                          hex(this->get_word(row, bank)).c_str(), hex(w).c_str()));
        }
        ++res_runs;
      }
    }
    std::printf("mbar: %u LR/SC runs around waits and writing ops\n", res_runs);

    this->mbar_wait_hold();
    this->check_commit_errors("mbar");
    std::printf("mbar directed: %s\n", (failures_ == before) ? "ok" : "FAILED");
  }

  // Waiter events of several ops, each placed by its op's commit report (an
  // atomic reports two cycles after its bank takes it, at write-back + 1):
  // insert = report - 2, flip = report - 1.
  struct MbarScn {
    const char* what;
    uint64_t    word;
    uint32_t    first_op;
    uint64_t    first_operand;
    uint32_t    second_op;
    uint64_t    second_operand;
  };

  void mbar_events() {
    const uint32_t before = failures_;
    using namespace mbar;
    const uint64_t P1 = 1ull << 63;
    // Two ops on one object from one LSU input, so the bank takes them back to back.
    const MbarScn scns[] = {
      {"try_wait open, then the completing arrive", pack(1, 2, 0, 0), OP_TRY_WAIT, 0, OP_ARRIVE, 1},
      {"completing arrive, then try_wait (reads it done)", pack(1, 2, 0, 0), OP_ARRIVE, 1, OP_TRY_WAIT, 0},
      {"try_wait open, then an arrive that leaves it open", pack(2, 2, 0, 1), OP_TRY_WAIT, P1, OP_ARRIVE, 1},
      {"test_wait open, then the completing arrive", pack(1, 2, 0, 0), OP_TEST_WAIT, 0, OP_ARRIVE, 1},
      {"try_wait done, then a completing complete_tx", pack(0, 2, 64, 1), OP_TRY_WAIT, 0, OP_COMPLETE_TX, 64},
      {"completing expect_tx, then try_wait open", pack(0, 2, -32, 0), OP_EXPECT_TX, 32, OP_TRY_WAIT, P1},
      {"two try_waits open", pack(1, 1, 0, 1), OP_TRY_WAIT, P1, OP_TRY_WAIT, P1},
    };
    uint32_t runs = 0;
    for (const MbarScn& c : scns) {
      for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
        const uint32_t row = this->rand_below(NUM_ROWS);
        this->put_word(row, bank, c.word);
        const int w0 = static_cast<int>(this->rand_below(NUM_WARPS));
        const int w1 = static_cast<int>(this->rand_below(NUM_WARPS));
        LsuReq a = mbar_req(c.first_op, row, bank, c.first_operand, w0);
        LsuReq b = mbar_req(c.second_op, row, bank, c.second_operand, w1);
        ScnRun sr = this->run_issues({this->lsu_issue(0, 0, a), this->lsu_issue(0, 0, b)}, true, 6);
        ++runs;
        if (!sr.ok) {
          continue;
        }
        const Result ra = apply(c.first_op, c.word, c.first_operand);
        const uint64_t mid = ra.writes ? ra.new_word : c.word;
        const Result rb = apply(c.second_op, mid, c.second_operand);
        const uint64_t fin = rb.writes ? rb.new_word : mid;
        this->check_ret(strf("mbar events %s bank %u: first", c.what, bank), sr.data[0], ra.ret_word);
        this->check_ret(strf("mbar events %s bank %u: second", c.what, bank), sr.data[1], rb.ret_word);
        std::vector<int> rep;
        for (const auto& e : sr.commits) {
          if (e.bank == bank) {
            rep.push_back(e.rel);
          }
        }
        this->gold(row, bank) = fin;
        // An op right behind a write to its word waits a cycle more (read-during-write).
        const int gap = ra.writes ? 4 : 3;
        if (rep.size() != 2 || rep[1] != rep[0] + gap) {
          this->fail(strf("FAIL mbar events %s bank %u: %zu commit reports, expected two %d cycles apart",
                          c.what, bank, rep.size(), gap));
          continue;
        }
        std::vector<MbarEvRel> want;
        const Result* rs[2] = {&ra, &rb};
        const LsuReq* rq[2] = {&a, &b};
        for (int k = 0; k < 2; ++k) {
          if (rq[k]->op == OP_TRY_WAIT && rs[k]->ret_word == 0) {
            want.push_back({rep[k] - 2, bank, false, req_wid(0, *rq[k]), row});
          } else if (rs[k]->flip) {
            want.push_back({rep[k] - 1, bank, true, 0, row});
          }
        }
        bool ok = (sr.mbar.size() == want.size());
        for (size_t k = 0; ok && k < want.size(); ++k) {
          const auto& g = sr.mbar[k];
          ok = g.rel == want[k].rel && g.bank == want[k].bank && g.flip == want[k].flip && g.row == want[k].row
            && (g.flip || g.wid == want[k].wid);
        }
        if (!ok) {
          std::string got, exp;
          for (const auto& e : sr.mbar) {
            got += strf(" {+%d bank %u %s wid %u row 0x%x}", e.rel, e.bank, e.flip ? "flip" : "insert", e.wid, e.row);
          }
          for (const auto& e : want) {
            exp += strf(" {+%d bank %u %s wid %u row 0x%x}", e.rel, e.bank, e.flip ? "flip" : "insert", e.wid, e.row);
          }
          this->fail(strf("FAIL mbar events %s bank %u: got%s, expected%s", c.what, bank,
                          got.empty() ? " none" : got.c_str(), exp.empty() ? " none" : exp.c_str()));
        }
        if (this->get_word(row, bank) != fin) {
          this->fail(strf("FAIL mbar events %s bank %u: word %s, expected %s", c.what, bank,
                          hex(this->get_word(row, bank)).c_str(), hex(fin).c_str()));
        }
      }
    }

    // One try_wait per bank from as many LSU inputs at once, then one
    // completing arrive per bank: the banks' inserts, then their flips, land in
    // the same cycles.
    const uint32_t nl = std::min(NUM_LSU_REQS, NUM_BANKS);
    uint32_t same_insert = 0, same_flip = 0;
    for (uint32_t round = 0; round < 4 && nl > 1; ++round) {
      std::vector<uint32_t> rows(nl);
      for (uint32_t k = 0; k < nl; ++k) {
        rows[k] = this->rand_below(NUM_ROWS);
        this->put_word(rows[k], k, pack(1, 3, 0, 0));
      }
      for (int pass = 0; pass < 2; ++pass) {
        std::vector<Issue> is;
        std::vector<LsuReq> rq;
        for (uint32_t k = 0; k < nl; ++k) {
          rq.push_back(pass == 0 ? mbar_req(OP_TRY_WAIT, rows[k], k, 0, static_cast<int>(k % NUM_WARPS))
                                 : mbar_req(OP_ARRIVE, rows[k], k, 1, static_cast<int>(k % NUM_WARPS)));
          is.push_back(this->lsu_issue(0, k, rq.back()));
        }
        ScnRun sr = this->run_issues(is, true, 6);
        ++runs;
        if (!sr.ok) {
          continue;
        }
        std::map<int, uint32_t> per_cycle;
        for (uint32_t k = 0; k < nl; ++k) {
          int rep = -1;
          for (const auto& e : sr.commits) {
            if (e.bank == k) {
              rep = e.rel;
            }
          }
          const bool flip = (pass == 1);
          bool found = false;
          for (const auto& e : sr.mbar) {
            if (e.bank == k) {
              found = !found && rep >= 0 && e.rel == rep - (flip ? 1 : 2) && e.flip == flip && e.row == rows[k]
                   && (flip || e.wid == req_wid(k, rq[k]));
              ++per_cycle[e.rel];
            }
          }
          if (!found) {
            this->fail(strf("FAIL mbar events %s per bank, bank %u: no single event at its report%s",
                            flip ? "flips" : "inserts", k, rep >= 0 ? strf(" (+%d)", rep).c_str() : ""));
          }
        }
        for (const auto& pc : per_cycle) {
          if (pc.second > 1) {
            (pass == 0 ? same_insert : same_flip) += 1;
          }
        }
      }
      for (uint32_t k = 0; k < nl; ++k) {
        this->gold(rows[k], k) = apply(OP_ARRIVE, pack(1, 3, 0, 0), 1).new_word;
      }
    }
    if (nl > 1 && (same_insert == 0 || same_flip == 0)) {
      this->fail(strf("FAIL mbar events: no cycle with inserts (%u) or flips (%u) on two banks", same_insert, same_flip));
    }
    this->check_mbar_errors("mbar events");
    std::printf("mbar events: %u runs, %u cycles with inserts and %u with flips on several banks: %s\n", runs,
                same_insert, same_flip, (failures_ == before) ? "ok" : "FAILED");
  }

  // A wait's response held by back-pressure keeps its 0/1 outcome while DMA
  // traffic re-drives the bank and rewrites the object. k loads ahead of it on
  // the same port fill the response path, so the wait's own response can stall
  // in the bank. The object is rewritten once the wait's commit report shows it
  // performed.
  void mbar_wait_hold() {
    const uint32_t lane = 0;
    const int stall_cycles = 30;
    uint32_t runs = 0, stalled = 0;
    for (uint32_t k = 0; k <= 4; ++k) {
      for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
        const uint32_t row = this->rand_below(NUM_ROWS);
        uint64_t w = this->rand_mbar_word() | 0x123ull;
        this->put_word(row, bank, w);
        const uint64_t token = this->rand_below(2) ? 0 : (1ull << 63);
        const uint64_t want = mbar::apply(mbar::OP_TEST_WAIT, w, token).ret_word;
        std::deque<std::pair<LsuReq, uint64_t>> todo, sent;
        for (uint32_t i = 0; i < k; ++i) {
          uint32_t r2 = this->other_row(row);
          LsuReq ld;
          ld.byteen = FULL_BE;
          ld.addr = word_addr(r2, bank);
          ld.wid = 1;
          todo.push_back({ld, this->gold(r2, bank)});
        }
        todo.push_back({mbar_req(mbar::OP_TEST_WAIT, row, bank, token, 0), want});
        const uint64_t w2 = (w ^ (1ull << 63)) ^ 0xff00ull;
        int wait_rep = -1, release = stall_cycles;
        bool dma_done = false;
        std::vector<uint64_t> got;
        for (int rel = 0; rel < 200 && got.size() < k + 1; ++rel) {
          if (!dut_.lane[lane] && !todo.empty()) {
            todo.front().first.tag = this->next_tag();
            dut_.lane[lane] = todo.front().first;
            sent.push_back(todo.front());
            todo.pop_front();
          }
          if (wait_rep >= 0 && !dma_done && !dut_.dma) {
            DmaReq d;
            d.rw = true;
            d.row = row;
            d.be.fill(0);
            d.be[bank] = FULL_BE;
            d.data[bank] = w2;
            d.tag = this->next_dma_tag();
            dut_.dma = d;
            dma_done = true;
          } else if (dma_done && !dut_.dma && rel < release) {
            DmaReq d;
            d.rw = false;
            d.row = row;
            d.tag = this->next_dma_tag();
            dut_.dma = d;
          }
          dut_.rsp_ready[lane] = (rel >= release);
          Events ev = dut_.clock();
          if (ev.commit[bank] && ev.commit_wid[bank] == 0 && wait_rep < 0) {
            wait_rep = rel;
            if (rel < release) {
              release = rel + 10;
              ++stalled;
            }
          }
          if (ev.lsu_rsp[lane]) {
            got.push_back(ev.rsp_data[lane]);
          }
        }
        dut_.rsp_ready[lane] = true;
        for (uint32_t i = 0; i < 8; ++i) {
          dut_.clock();
        }
        dut_.dma.reset();
        this->idle(2);
        if (dma_done) {
          this->gold(row, bank) = w2;
        }
        ++runs;
        if (got.size() != k + 1) {
          this->fail(strf("FAIL mbar wait hold k=%u bank %u: %zu responses of %u", k, bank, got.size(), k + 1));
          continue;
        }
        for (uint32_t i = 0; i <= k; ++i) {
          if (got[i] != sent[i].second) {
            this->fail(strf("FAIL mbar wait hold k=%u bank %u: response %u (%s) is %s, expected %s", k, bank, i,
                            i == k ? "the wait" : "a load", hex(got[i]).c_str(), hex(sent[i].second).c_str()));
          }
        }
      }
    }
    if (stalled == 0) {
      this->fail("FAIL mbar wait hold: no wait performed while its port's responses were stalled");
    }
    std::printf("mbar: %u waits behind stalled responses (%u performed during the stall), the object rewritten "
                "behind them\n", runs, stalled);
  }

  // Random op streams on a few objects from every input, racing each other,
  // DMA writes of the same objects, DMA reads of their rows and loads and
  // stores of the neighbouring words. Ops on one bank perform in handshake
  // order, so each is modelled when its port takes it; an object takes at most
  // one staged op at a time, so its operands are always chosen error-free. A
  // DMA write racing an object's op is resolved by the op's returned word: the
  // DMA word has the other phase, so the two orders return different words.
  struct MbarObj {
    uint32_t row = 0, bank = 0;
    uint64_t gold = 0;
    int      staged = -1;          // lane holding an op to it, not yet taken
    uint32_t outstanding = 0;      // taken, not yet answered (ops with a response)
    bool     unconfirmed_store = false;
    uint64_t store_cycle = 0;
    bool     dma_pending = false;  // a DMA write of it is offered
    bool     racing = false;       // a DMA write landed with one op outstanding
    uint64_t dma_word = 0;
  };
  struct MbarPend {
    uint32_t lane = 0;
    int      obj = -1;             // -1: a neighbouring word
    uint32_t op = 0;
    uint64_t operand = 0;
    uint64_t before = 0;
    uint64_t exp_ret = 0;
    uint64_t taken = 0;
    bool     race = false;
    uint64_t alt_ret = 0, final_a = 0, final_b = 0;
  };

  void mbar_stress(uint32_t seed) {
    const uint32_t before = failures_;
    std::mt19937_64 rng(seed);
    auto rnd = [&](uint32_t n) { return static_cast<uint32_t>(rng() % n); };
    auto chance = [&](uint32_t pct) { return rnd(100) < pct; };
    const uint32_t OBJ_ROWS = 2, NB_ROWS = 4, DMA_ROWS = 2;
    const uint32_t nb_row0 = OBJ_ROWS, dma_row0 = OBJ_ROWS + NB_ROWS;

    std::vector<MbarObj> objs;
    for (uint32_t row = 0; row < OBJ_ROWS; ++row) {
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        MbarObj o;
        o.row = row;
        o.bank = b;
        o.gold = mbar::init(1 + rnd(6));
        this->put_word(row, b, o.gold);
        objs.push_back(o);
      }
    }
    std::array<std::vector<uint32_t>, NUM_REQS> priv;
    uint32_t next_owner = 0;
    for (uint32_t row = nb_row0; row < nb_row0 + NB_ROWS; ++row) {
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        priv[next_owner++ % NUM_REQS].push_back(row * NUM_BANKS + b);
      }
    }

    std::unordered_map<uint32_t, MbarPend> pend;
    std::array<std::optional<MbarPend>, NUM_REQS> staged;
    std::deque<DmaPending> dma_pend;
    std::optional<int> dma_obj;   // the object the offered DMA write targets
    uint64_t n_ops[6] = {}, n_flips = 0, n_ld = 0, n_st = 0, n_init = 0, n_dma_wr = 0, n_dma_rd = 0;
    uint64_t n_race_op_first = 0, n_race_dma_first = 0, n_dma_obj = 0;
    uint32_t errors = 0;
    const uint64_t start = dut_.cycle();
    uint64_t last_progress = start;
    auto report = [&](const std::string& m) {
      ++errors;
      if (errors <= 10) {
        this->fail(strf("FAIL mbar stress seed=%u [cycle %" PRIu64 "] %s", seed, dut_.cycle(), m.c_str()));
      } else {
        ++failures_;
      }
    };

    auto gen_lane = [&](uint32_t l) {
      LsuReq r;
      r.wid = static_cast<int>(rnd(NUM_WARPS));
      MbarPend p;
      p.lane = l;
      uint32_t kind = rnd(100);
      if (kind < 60) {
        std::vector<int> free;
        for (size_t j = 0; j < objs.size(); ++j) {
          const MbarObj& o = objs[j];
          if (o.staged < 0 && !o.dma_pending && !o.racing) {
            free.push_back(static_cast<int>(j));
          }
        }
        if (free.empty()) {
          return;
        }
        int j = free[rnd(static_cast<uint32_t>(free.size()))];
        MbarObj& o = objs[j];
        p.obj = j;
        if (chance(8)) {
          // Re-initialise it with a plain store, as software does.
          r.rw = true;
          r.byteen = FULL_BE;
          r.addr = word_addr(o.row, o.bank);
          r.data = mbar::init(1 + rnd(6));
          p.op = 0;
        } else {
          uint32_t op;
          do {
            op = mbar::OP_BASE + rnd(6);
          } while ((op == mbar::OP_ARRIVE || op == mbar::OP_ARRIVE_TX) && mbar::pending(o.gold) == 0);
          const int64_t tx = mbar::tx(o.gold);
          uint64_t operand;
          switch (op) {
          case mbar::OP_ARRIVE:
            operand = chance(70) ? 1 : 1 + rnd(mbar::pending(o.gold));
            break;
          case mbar::OP_ARRIVE_TX:
          case mbar::OP_EXPECT_TX:
            operand = rnd(static_cast<uint32_t>(std::min<int64_t>(mbar::TX_MAX - tx, 256)) + 1);
            break;
          case mbar::OP_COMPLETE_TX:
            operand = (tx > 0 && chance(50)) ? static_cast<uint64_t>(tx)
                                             : rnd(static_cast<uint32_t>(std::min<int64_t>(tx + mbar::TX_MAX, 256)) + 1);
            break;
          default:
            operand = (rng() & ~(1ull << 63)) | (uint64_t(rnd(2)) << 63);
            break;
          }
          r = mbar_req(op, o.row, o.bank, operand, r.wid);
          p.op = op;
          p.operand = operand;
        }
        o.staged = static_cast<int>(l);
      } else {
        const uint32_t w = priv[l][rnd(static_cast<uint32_t>(priv[l].size()))];
        r.addr = word_addr(w / NUM_BANKS, w % NUM_BANKS);
        r.byteen = FULL_BE;
        if (chance(50)) {
          p.exp_ret = gold_[w];
          p.op = 1;
          ++n_ld;
        } else {
          r.rw = true;
          r.byteen = 1 + rnd(FULL_BE);
          r.data = rng() & WORD_MASK;
          gold_[w] = merge(gold_[w], r.data, r.byteen);
          p.op = 2;
          ++n_st;
        }
      }
      r.tag = this->next_tag();
      dut_.lane[l] = r;
      staged[l] = p;
    };

    // The offered request on lane l was taken: model it in handshake order.
    auto taken = [&](uint32_t l, const LsuReq& r) {
      MbarPend p = *staged[l];
      staged[l].reset();
      p.taken = dut_.cycle();
      if (p.obj >= 0) {
        MbarObj& o = objs[p.obj];
        o.staged = -1;
        if (r.rw) {
          o.gold = r.data;
          o.unconfirmed_store = true;
          o.store_cycle = p.taken;
          ++n_init;
          if (acks_stores(l)) {
            MbarPend ack = p;
            ack.obj = -1;
            ack.op = 2;
            pend[r.tag] = ack;
          }
          return;
        }
        mbar::Result res = mbar::apply(p.op, o.gold, p.operand);
        if (res.err != mbar::ERR_NONE) {
          report("the testbench generated an erroneous op");
        }
        p.before = o.gold;
        p.exp_ret = res.ret_word;
        if (res.writes) {
          o.gold = res.new_word;
        }
        ++o.outstanding;
        ++n_ops[p.op - mbar::OP_BASE];
        n_flips += res.flip;
        pend[r.tag] = p;
      } else if (!r.rw || acks_stores(l)) {
        pend[r.tag] = p;
      }
    };

    auto gen_dma = [&]() {
      DmaReq d;
      d.tag = this->next_dma_tag();
      uint32_t kind = rnd(100);
      if (kind < 35) {
        // One object, with one op or none outstanding.
        std::vector<int> ok;
        for (size_t j = 0; j < objs.size(); ++j) {
          const MbarObj& o = objs[j];
          if (o.staged < 0 && !o.racing && !o.dma_pending && !o.unconfirmed_store && o.outstanding <= 1) {
            ok.push_back(static_cast<int>(j));
          }
        }
        if (ok.empty()) {
          return;
        }
        int j = ok[rnd(static_cast<uint32_t>(ok.size()))];
        MbarObj& o = objs[j];
        // The other phase, and enough pending for the op it may race.
        uint64_t need = 1, ph = mbar::phase(o.gold);
        for (const auto& kv : pend) {
          if (kv.second.obj == j) {
            ph = mbar::phase(kv.second.before);
            if (kv.second.op == mbar::OP_ARRIVE) {
              need = std::max<uint64_t>(need, kv.second.operand);
            }
          }
        }
        const uint64_t pnd = std::min<uint64_t>(need + rnd(4), mbar::COUNT_MAX);
        o.dma_word = mbar::pack(pnd, std::max<uint64_t>(pnd, 1 + rnd(16)), 0, static_cast<uint32_t>(ph ^ 1));
        o.dma_pending = true;
        dma_obj = j;
        d.rw = true;
        d.row = o.row;
        d.be.fill(0);
        d.be[o.bank] = FULL_BE;
        d.data[o.bank] = o.dma_word;
        ++n_dma_obj;
      } else if (kind < 60) {
        d.rw = true;
        d.row = dma_row0 + rnd(DMA_ROWS);
        for (uint32_t b = 0; b < NUM_BANKS; ++b) {
          d.data[b] = rng() & WORD_MASK;
          d.be[b] = rnd(FULL_BE + 1);
          this->gold(d.row, b) = dma_merge(d, b, this->gold(d.row, b));
        }
        ++n_dma_wr;
      } else {
        d.rw = false;
        d.row = rnd(dma_row0 + DMA_ROWS);
        DmaPending dp{d.tag, d.row, {}, dut_.cycle()};
        for (uint32_t b = 0; b < NUM_BANKS; ++b) {
          dp.expect[b] = this->gold(d.row, b);
        }
        dma_pend.push_back(dp);
        ++n_dma_rd;
      }
      dut_.dma = d;
    };

    auto dma_landed = [&]() {
      if (!dma_obj) {
        return;
      }
      MbarObj& o = objs[*dma_obj];
      o.dma_pending = false;
      if (o.outstanding == 0) {
        o.gold = o.dma_word;
      } else {
        for (auto& kv : pend) {
          MbarPend& p = kv.second;
          if (p.obj != *dma_obj) {
            continue;
          }
          mbar::Result alt = mbar::apply(p.op, o.dma_word, p.operand);
          if (alt.err != mbar::ERR_NONE) {
            report("the racing DMA word makes the op erroneous");
          }
          p.race = true;
          p.alt_ret = alt.ret_word;
          p.final_a = o.dma_word;
          p.final_b = alt.writes ? alt.new_word : o.dma_word;
        }
        o.racing = true;
      }
      dma_obj.reset();
    };

    auto busy = [&]() {
      if (!pend.empty() || !dma_pend.empty() || dut_.dma) {
        return true;
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        if (dut_.lane[l]) {
          return true;
        }
      }
      return false;
    };

    bool draining = false;
    while (true) {
      const uint64_t now = dut_.cycle();
      if (!draining && now - start >= opts_.stress_cycles / 2) {
        draining = true;
      }
      if (draining && !busy()) {
        break;
      }
      if (now - last_progress > 2000) {
        report("no progress for 2000 cycles (deadlock)");
        dut_.dma.reset();
        for (auto& l : dut_.lane) {
          l.reset();
        }
        break;
      }
      if (!draining) {
        for (uint32_t l = 0; l < NUM_REQS; ++l) {
          if (!dut_.lane[l] && chance(50)) {
            gen_lane(l);
          }
        }
        if (!dut_.dma && chance(40)) {
          gen_dma();
        }
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        dut_.rsp_ready[l] = chance(80);
      }
      dut_.dma_rsp_ready = chance(70);
      Events ev = dut_.clock();
      if (ev.any()) {
        last_progress = dut_.cycle();
      }
      if (ev.dma_acc) {
        dma_landed();
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        if (ev.lsu_acc[l]) {
          taken(l, ev.acc_req[l]);
        }
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        if (!ev.lsu_rsp[l]) {
          continue;
        }
        auto it = pend.find(ev.rsp_tag[l]);
        if (it == pend.end()) {
          report(strf("lane %u: response with unknown tag 0x%x", l, ev.rsp_tag[l]));
          continue;
        }
        MbarPend p = it->second;
        pend.erase(it);
        const uint64_t got = ev.rsp_data[l];
        if (p.obj < 0) {
          if (p.op == 1 && got != p.exp_ret) {
            report(strf("lane %u load of a neighbouring word: expected %s, got %s", l, hex(p.exp_ret).c_str(),
                        hex(got).c_str()));
          }
          continue;
        }
        MbarObj& o = objs[p.obj];
        --o.outstanding;
        if (o.unconfirmed_store && p.taken > o.store_cycle) {
          o.unconfirmed_store = false;
        }
        const std::string what = strf("lane %u %s operand %s on object (row %u, bank %u), word before %s", l,
                                      amo_name(p.op, false), hex(p.operand).c_str(), o.row, o.bank,
                                      hex(p.before).c_str());
        if (p.race) {
          o.racing = false;
          if (got == p.exp_ret) {
            o.gold = p.final_a;
            ++n_race_op_first;
          } else if (got == p.alt_ret) {
            o.gold = p.final_b;
            ++n_race_dma_first;
          } else {
            report(strf("%s, racing a DMA write of %s: returned %s, expected %s (op first) or %s (DMA first)",
                        what.c_str(), hex(o.dma_word).c_str(), hex(got).c_str(), hex(p.exp_ret).c_str(),
                        hex(p.alt_ret).c_str()));
          }
        } else if (got != p.exp_ret) {
          report(strf("%s: returned %s, expected %s", what.c_str(), hex(got).c_str(), hex(p.exp_ret).c_str()));
        }
      }
      if (ev.dma_rsp) {
        if (dma_pend.empty()) {
          report("unexpected DMA response");
        } else {
          DmaPending dp = dma_pend.front();
          dma_pend.pop_front();
          for (uint32_t b = 0; b < NUM_BANKS && dp.row >= dma_row0; ++b) {
            if (ev.dma_rsp_data[b] != dp.expect[b]) {
              report(strf("DMA read (row 0x%x, bank %u): expected %s, got %s", dp.row, b, hex(dp.expect[b]).c_str(),
                          hex(ev.dma_rsp_data[b]).c_str()));
            }
          }
        }
      }
    }
    dut_.rsp_ready.fill(true);
    dut_.dma_rsp_ready = true;
    this->idle(8);
    for (const MbarObj& o : objs) {
      this->gold(o.row, o.bank) = o.gold;
    }
    std::vector<uint32_t> rows;
    for (uint32_t row = 0; row < dma_row0 + DMA_ROWS; ++row) {
      rows.push_back(row);
    }
    this->check_rows("mbar stress", rows);
    for (uint32_t w = 0; w < NUM_WARPS; ++w) {
      if (dut_.uncommitted[w] != 0) {
        report(strf("warp %u: %" PRIu64 " LSU-input access(es) never reported committed", w, dut_.uncommitted[w]));
      }
    }
    this->check_commit_errors("mbar stress");
    if (n_race_op_first + n_race_dma_first == 0) {
      report("no DMA write raced an op on its object");
    }
    std::printf("mbar stress seed=%u: %" PRIu64 " cycles, ops arrive %" PRIu64 " arrive_tx %" PRIu64 " expect_tx %"
                PRIu64 " complete_tx %" PRIu64 " test_wait %" PRIu64 " op15 %" PRIu64 ", %" PRIu64 " flips, %" PRIu64
                " re-inits; %" PRIu64 " DMA writes of objects, races resolved op-first %" PRIu64 " / DMA-first %"
                PRIu64 "; neighbours %" PRIu64 " ld + %" PRIu64 " st; DMA %" PRIu64 " wr + %" PRIu64 " rd; %s\n",
                seed, dut_.cycle() - start, n_ops[0], n_ops[1], n_ops[2], n_ops[3], n_ops[4], n_ops[5], n_flips,
                n_init, n_dma_obj, n_race_op_first, n_race_dma_first, n_ld, n_st, n_dma_wr, n_dma_rd,
                (failures_ == before) ? "ok" : "FAILED");
  }

  // One erroneous op; the bank must abort the run with its assert.
  int mbar_negative(const std::string& kind) {
    const uint32_t row = 5 % NUM_ROWS, bank = NUM_BANKS - 1;
    LsuReq r;
    if (kind == "underflow") {
      this->put_word(row, bank, mbar::init(1));
      r = mbar_req(mbar::OP_ARRIVE, row, bank, 2, 0);
    } else if (kind == "txrange") {
      this->put_word(row, bank, mbar::init(1));
      r = mbar_req(mbar::OP_EXPECT_TX, row, bank, uint64_t(mbar::TX_MAX) + 1, 0);
    } else if (kind == "align") {
      this->put_word(row, bank, mbar::init(5));
      r = mbar_req(mbar::OP_ARRIVE, row, bank, 1, 0);
      r.byteen = 0x0f;
    } else {
      std::printf("unknown --neg=%s\n", kind.c_str());
      return 2;
    }
    std::printf("neg %s: issuing %s; the bank's assert must abort the run here\n", kind.c_str(),
                amo_name(r.op, false));
    std::fflush(stdout);
    this->lsu_access(0, r);
    this->idle(8);
    std::printf("FAILED: neg %s did not abort\n", kind.c_str());
    return 1;
  }

  uint64_t mbar_ops_[6] = {};
  uint64_t mbar_flips_ = 0;
  uint64_t mbar_event_checks_[3] = {};
#endif

  ///////////////////////////////////////////////////////////////////////////
  // Directed: one atomic on one bank and one DMA access injected at a fixed
  // offset k from the cycle t the atomic reaches that bank.

  enum Scn {
    S_WR_BANK,
    S_RD_BANK,
    S_WR_WORD,
    S_RD_WORD,
    S_SC_WR_BANK,
    S_WR_OTHER,
    NUM_SCN
  };

  static const char* scn_name(int s) {
    static const char* names[NUM_SCN] = {
      "dma-wr same-bank/other-word",
      "dma-rd same-bank/other-word",
      "dma-wr same-word",
      "dma-rd same-word",
      "sc + dma-wr same-bank",
      "dma-wr other-banks (control)"
    };
    return names[s];
  }

  static constexpr int K_MIN = -1;
  static constexpr int K_MAX = 4;
  static constexpr int NUM_K = K_MAX - K_MIN + 1;

  struct Cell {
    uint32_t runs  = 0;
    uint32_t fails = 0;
    uint32_t stall_max = 0;
    uint32_t sc_ok = 0;
  };

  // A 64-bit bank word holds two .W lanes; a 32-bit word is always W_LO.
  enum Width { W_LO, W_HI, W_D };
  static constexpr int NUM_WIDTHS = (XLEN == 64) ? 3 : 1;

  static uint32_t lane_byteen(int width) {
    if (XLEN == 32) {
      return FULL_BE;
    }
    return (width == W_LO) ? 0x0f : (width == W_HI) ? 0xf0 : 0xff;
  }

  static const char* width_name(int width) {
    if (XLEN == 32) {
      return "W";
    }
    return (width == W_LO) ? "W.lo" : (width == W_HI) ? "W.hi" : "D";
  }

  static uint64_t lane_position(int width, uint64_t v) {
    if (XLEN == 64 && width == W_LO) {
      return v & MASK32;
    }
    if (XLEN == 64 && width == W_HI) {
      return v & ~MASK32;
    }
    return v & WORD_MASK;
  }

  uint64_t lane_data(int width) {
    uint64_t v = lane_position(width, this->rand_word());
    return v ? v : lane_position(width, 0x0000000100000001ull);
  }

  uint32_t other_row(uint32_t row) {
    uint32_t r = this->rand_below(NUM_ROWS - 1);
    return (r >= row) ? r + 1 : r;
  }

  DmaReq make_dma_write(int scn, uint32_t row, uint32_t bank) {
    DmaReq d;
    d.rw = true;
    d.row = row;
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      d.data[b] = this->rand_word();
    }
    if (scn == S_WR_OTHER) {
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        d.be[b] = (b == bank) ? 0 : this->rand_below(FULL_BE + 1);
      }
      d.be[(bank + 1) % NUM_BANKS] = FULL_BE;
      return d;
    }
    switch (this->rand_below(3)) {
    case 0: // full row
      d.be.fill(FULL_BE);
      break;
    case 1: // only the atomic's bank
      d.be.fill(0);
      d.be[bank] = FULL_BE;
      break;
    default: // random bytes, at least one in the atomic's bank
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        d.be[b] = this->rand_below(FULL_BE + 1);
      }
      if (d.be[bank] == 0 || (scn == S_WR_WORD && d.be[bank] == FULL_BE)) {
        d.be[bank] = 1 + this->rand_below(FULL_BE - 1);
      }
      break;
    }
    return d;
  }

  void directed() {
    std::array<std::array<Cell, NUM_K>, NUM_SCN> cells{};
    std::array<std::array<uint32_t, NUM_K>, NUM_SCN> reported{};
    uint32_t lane_rr = 0;
    for (int s = 0; s < NUM_SCN; ++s) {
      if (s == S_WR_OTHER && NUM_BANKS < 2) {
        continue;
      }
      for (int k = K_MIN; k <= K_MAX; ++k) {
        for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
          for (int width = 0; width < NUM_WIDTHS; ++width) {
            for (int rep = 0; rep < 2; ++rep) {
              uint32_t lane = (lane_rr++) % NUM_REQS;
              Cell& cell = cells[s][k - K_MIN];
              uint32_t& nrep = reported[s][k - K_MIN];
              std::string msg;
              bool ok = this->directed_run(s, k, bank, lane, width, cell, msg);
              ++cell.runs;
              if (!ok) {
                ++cell.fails;
                ++failures_;
                if (nrep < opts_.max_report) {
                  std::printf("%s", msg.c_str());
                  ++nrep;
                }
              } else if (opts_.verbose) {
                std::printf("%s", msg.c_str());
              }
            }
          }
        }
      }
    }

    std::printf("\ndirected summary (t = cycle the atomic reaches its bank; cells = failed/runs):\n");
    std::printf("  %-34s", "scenario");
    for (int k = K_MIN; k <= K_MAX; ++k) {
      std::printf(" %9s", (k == 0) ? "t" : strf("t%+d", k).c_str());
    }
    std::printf("\n");
    for (int s = 0; s < NUM_SCN; ++s) {
      std::printf("  %-34s", scn_name(s));
      for (int k = K_MIN; k <= K_MAX; ++k) {
        const Cell& c = cells[s][k - K_MIN];
        if (c.runs == 0) {
          std::printf(" %9s", "-");
        } else if (c.fails == 0) {
          std::printf(" %9s", strf("ok %u", c.runs).c_str());
        } else {
          std::printf(" %9s", strf("FAIL %u/%u", c.fails, c.runs).c_str());
        }
      }
      std::printf("\n");
    }
    std::printf("  %-34s", "max DMA stall cycles (any scn)");
    for (int k = K_MIN; k <= K_MAX; ++k) {
      uint32_t m = 0;
      for (int s = 0; s < NUM_SCN; ++s) {
        m = std::max(m, cells[s][k - K_MIN].stall_max);
      }
      std::printf(" %9u", m);
    }
    std::printf("\n");
    uint32_t sc_ok = 0;
    uint32_t sc_runs = 0;
    for (int k = 0; k < NUM_K; ++k) {
      sc_ok += cells[S_SC_WR_BANK][k].sc_ok;
      sc_runs += cells[S_SC_WR_BANK][k].runs;
    }
    std::printf("  store-conditionals that succeeded: %u/%u\n\n", sc_ok, sc_runs);
    if (sc_runs != 0 && sc_ok == 0) {
      this->fail("FAIL: no store-conditional succeeded, so the LR/SC scenario exercised nothing");
    }
  }

  bool directed_run(int scn, int k, uint32_t bank, uint32_t lane, int width, Cell& cell, std::string& msg) {
    const bool same_word = (scn == S_WR_WORD || scn == S_RD_WORD);
    const bool dma_wr = (scn != S_RD_BANK && scn != S_RD_WORD);
    const uint32_t row_w = this->rand_below(NUM_ROWS);
    const uint32_t row_d = same_word ? row_w : this->other_row(row_w);
    const uint64_t old0 = this->gold(row_w, bank);
    const uint64_t cyc_base = dut_.cycle();

    std::string head = strf("%s @%s lane=%u bank=%u width=%s row_w=0x%x row_d=0x%x",
                            scn_name(scn), (k == 0) ? "t" : strf("t%+d", k).c_str(),
                            lane, bank, width_name(width), row_w, row_d);
    std::vector<std::string> errs;

    LsuReq amo;
    amo.rw = false;
    amo.amo = true;
    amo.byteen = this->lane_byteen(width);
    amo.addr = word_addr(row_w, bank);
    amo.data = this->lane_data(width);
    amo.op = (scn == S_SC_WR_BANK) ? AMO_SC : AMO_ADD;

    if (scn == S_SC_WR_BANK) {
      LsuReq lr = amo;
      lr.op = AMO_LR;
      lr.data = 0;
      std::optional<uint64_t> got = this->lsu_access(lane, lr);
      if (!got || *got != old0) {
        errs.push_back(strf("LR returned %s, expected %s", got ? hex(*got).c_str() : "nothing",
                            hex(old0).c_str()));
      }
      this->idle(2);
    }

    DmaReq dma;
    if (dma_wr) {
      dma = this->make_dma_write(scn, row_d, bank);
    } else {
      dma.rw = false;
      dma.row = row_d;
    }
    dma.tag = this->next_dma_tag();
    amo.tag = this->next_tag();

    const Row row_d_before = [&] {
      Row r;
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        r[b] = this->gold(row_d, b);
      }
      return r;
    }();

    // Timed window. All cycle numbers are relative to the cycle the atomic
    // is presented at its LSU port.
    const int dma_start = bank_lat_ + k;
    int c_acc = -1, d_acc = -1, d_rsp = -1, a_rsp = -1;
    uint32_t stalls = 0;
    uint64_t amo_ret = 0;
    Row dma_row{};
    const uint64_t c0 = dut_.cycle();
    dut_.lane[lane] = amo;
    for (int rel = 0; rel < 64; ++rel) {
      if (rel == dma_start) {
        dut_.dma = dma;
      }
      Events ev = dut_.clock();
      if (ev.lsu_acc[lane]) {
        c_acc = rel;
      }
      if (ev.dma_acc) {
        d_acc = rel;
      }
      if (ev.dma_stall) {
        ++stalls;
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        if (!ev.lsu_rsp[l]) {
          continue;
        }
        if (l != lane || ev.rsp_tag[l] != amo.tag || a_rsp >= 0) {
          errs.push_back(strf("stray LSU response at +%d: lane=%u tag=0x%x", rel, l, ev.rsp_tag[l]));
          continue;
        }
        a_rsp = rel;
        amo_ret = ev.rsp_data[l];
      }
      if (ev.dma_rsp) {
        if (dma_wr || d_rsp >= 0 || ev.dma_rsp_tag != dma.tag) {
          errs.push_back(strf("unexpected DMA response at +%d tag=0x%x", rel, ev.dma_rsp_tag));
        } else {
          d_rsp = rel;
          dma_row = ev.dma_rsp_data;
        }
      }
      bool dma_done = (d_acc >= 0) && (dma_wr || d_rsp >= 0);
      if (c_acc >= 0 && a_rsp >= 0 && dma_done) {
        break;
      }
    }
    cell.stall_max = std::max(cell.stall_max, stalls);
    std::string timing = strf("    timing (port cycle %" PRIu64 " = +0): atomic accepted +%d, rsp +%d; "
                              "DMA valid +%d, accepted +%d (stalled %u)%s\n",
                              c0, c_acc, a_rsp, dma_start, d_acc, stalls,
                              dma_wr ? "" : strf(", rsp +%d", d_rsp).c_str());
    if (c_acc < 0 || a_rsp < 0 || d_acc < 0 || (!dma_wr && d_rsp < 0)) {
      errs.push_back("handshake did not complete within 64 cycles (deadlock?)");
      dut_.lane[lane].reset();
      dut_.dma.reset();
    }
    this->idle(4);

    // Read back every row the window could have touched.
    std::vector<uint32_t> rows{row_w};
    if (row_d != row_w) {
      rows.push_back(row_d);
    }
    std::vector<Row> after;
    for (uint32_t row : rows) {
      after.push_back(this->dma_read(row));
    }
    auto actual = [&](uint32_t row, uint32_t b) {
      return after[(row == row_w) ? 0 : 1][b];
    };

    // Real time rules out a serial order only when the two operations cannot
    // overlap: a DMA access takes effect in its acceptance cycle, which can
    // precede the atomic reaching its bank; an atomic that already responded
    // has completed before a later DMA is accepted.
    const bool allow_ad = !(d_acc >= 0 && c_acc >= 0 && d_acc < c_acc + bank_lat_);
    const bool allow_da = !(a_rsp >= 0 && a_rsp < (d_acc >= 0 ? d_acc : dma_start));

    auto expect_word = [&](uint32_t row, uint32_t b, uint64_t exp, const char* what) {
      uint64_t got = actual(row, b);
      if (got != exp) {
        errs.push_back(strf("%s (row 0x%x, bank %u): expected %s, got %s", what, row, b,
                            hex(exp).c_str(), hex(got).c_str()));
      }
    };

    if (scn == S_SC_WR_BANK) {
      uint64_t ok_word = sc_rsp_word(amo, false);
      uint64_t fail_word = sc_rsp_word(amo, true);
      bool sc_ok = (amo_ret == ok_word);
      if (!sc_ok && amo_ret != fail_word) {
        errs.push_back(strf("SC returned %s, expected %s (success) or %s (failure)",
                            hex(amo_ret).c_str(), hex(ok_word).c_str(), hex(fail_word).c_str()));
      }
      cell.sc_ok += sc_ok;
      uint64_t exp_w = sc_ok ? amo_new_word(amo, old0) : old0;
      uint64_t got_w = actual(row_w, bank);
      if (got_w != exp_w) {
        errs.push_back(strf("SC %s but word (row 0x%x, bank %u) is %s, expected %s%s",
                            sc_ok ? "reported success" : "reported failure", row_w, bank,
                            hex(got_w).c_str(), hex(exp_w).c_str(),
                            (sc_ok && got_w == old0) ? " -> store-conditional write-back lost" : ""));
      }
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        expect_word(row_d, b, dma_merge(dma, b, row_d_before[b]), "DMA-written word");
      }
    } else if (!same_word) {
      uint64_t exp_w = amo_new_word(amo, old0);
      if (amo_ret != old0) {
        errs.push_back(strf("%s returned %s, expected old value %s", amo_name(amo.op, amo.uns),
                            hex(amo_ret).c_str(), hex(old0).c_str()));
      }
      uint64_t got_w = actual(row_w, bank);
      if (got_w != exp_w) {
        const char* why = "";
        if (got_w == old0) {
          why = " -> atomic write-back lost";
        }
        errs.push_back(strf("atomic word (row 0x%x, bank %u): expected %s, got %s%s",
                            row_w, bank, hex(exp_w).c_str(), hex(got_w).c_str(), why));
      }
      if (!dma_wr) {
        for (uint32_t b = 0; b < NUM_BANKS; ++b) {
          if (dma_row[b] != row_d_before[b]) {
            errs.push_back(strf("DMA read data (row 0x%x, bank %u): expected %s, got %s",
                                row_d, b, hex(row_d_before[b]).c_str(), hex(dma_row[b]).c_str()));
          }
        }
      }
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        uint64_t exp = dma_wr ? dma_merge(dma, b, row_d_before[b]) : row_d_before[b];
        uint64_t got = actual(row_d, b);
        if (got != exp) {
          const char* why = "";
          if (!dma_wr && b == bank && got == exp_w) {
            why = " -> atomic result written into the DMA-read word";
          } else if (dma_wr) {
            why = " -> DMA write corrupted";
          } else {
            why = " -> DMA-read row corrupted";
          }
          errs.push_back(strf("%s word (row 0x%x, bank %u): expected %s, got %s%s",
                              dma_wr ? "DMA-written" : "DMA-read", row_d, b,
                              hex(exp).c_str(), hex(got).c_str(), why));
        }
      }
    } else {
      // Same word: the outcome must equal one of the two serial orders.
      struct Order { const char* name; bool allowed; uint64_t ret; uint64_t fin; uint64_t dma_seen; };
      std::vector<Order> orders;
      uint64_t new_ad = amo_new_word(amo, old0);
      if (dma_wr) {
        uint64_t w1 = dma_merge(dma, bank, old0);
        orders.push_back({"atomic-then-DMA", allow_ad, old0, dma_merge(dma, bank, new_ad), 0});
        orders.push_back({"DMA-then-atomic", allow_da, w1, amo_new_word(amo, w1), 0});
      } else {
        orders.push_back({"atomic-then-DMA", allow_ad, old0, new_ad, new_ad});
        orders.push_back({"DMA-then-atomic", allow_da, old0, new_ad, old0});
      }
      uint64_t got_w = actual(row_w, bank);
      uint64_t seen_w = dma_wr ? 0 : dma_row[bank];
      const Order* match = nullptr;
      for (const Order& o : orders) {
        if (o.allowed && o.ret == amo_ret && o.fin == got_w && (dma_wr || o.dma_seen == seen_w)) {
          match = &o;
          break;
        }
      }
      if (!match) {
        std::string e = strf("atomic word (row 0x%x, bank %u) matches no serial order: atomic returned %s, final %s%s",
                             row_w, bank, hex(amo_ret).c_str(), hex(got_w).c_str(),
                             dma_wr ? "" : strf(", DMA read %s", hex(seen_w).c_str()).c_str());
        for (const Order& o : orders) {
          e += strf("\n      %s%s: returns %s, final %s%s", o.name, o.allowed ? "" : " (excluded by real time)",
                    hex(o.ret).c_str(), hex(o.fin).c_str(),
                    dma_wr ? "" : strf(", DMA read %s", hex(o.dma_seen).c_str()).c_str());
        }
        if (dma_wr && amo_ret == old0 && got_w == new_ad) {
          e += "\n      -> atomicity violated: the atomic overwrote a DMA write that landed between its read and write-back";
        } else if (dma_wr && amo_ret == old0 && got_w == dma_merge(dma, bank, old0)) {
          e += "\n      -> atomic write-back lost: the DMA write replaced it, bytes the DMA did not cover were reverted";
        }
        errs.push_back(e);
      } else if (opts_.verbose) {
        head += strf(" [order %s]", match->name);
      }
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        if (b == bank) {
          continue;
        }
        uint64_t exp = dma_wr ? dma_merge(dma, b, row_d_before[b]) : row_d_before[b];
        if (!dma_wr && dma_row[b] != exp) {
          errs.push_back(strf("DMA read data (row 0x%x, bank %u): expected %s, got %s",
                              row_d, b, hex(exp).c_str(), hex(dma_row[b]).c_str()));
        }
        expect_word(row_d, b, exp, dma_wr ? "DMA-written word" : "untouched word");
      }
    }

    // Keep the golden model in step with the DUT so a failure does not
    // cascade into the next run.
    for (size_t i = 0; i < rows.size(); ++i) {
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        this->gold(rows[i], b) = after[i][b];
      }
    }

    bool ok = errs.empty();
    msg = (ok ? "ok   " : "FAIL ") + head + strf(" [cycle %" PRIu64 "]\n", cyc_base);
    if (!ok || opts_.verbose) {
      msg += timing;
    }
    for (const std::string& e : errs) {
      msg += "    " + e + "\n";
    }
    return ok;
  }

  ///////////////////////////////////////////////////////////////////////////
  // LR/SC window: an LR, its SC issued once the LR has responded, and one
  // DMA access swept across the window. A DMA write to a reserved byte that
  // lands between the LR and the SC must make the SC fail; every other DMA
  // access must leave the reservation alone. An atomic's bank cycle is its
  // response cycle minus the calibrated response latency, which places the
  // DMA in real time against the LR and SC occupancies.

  enum LrscMode {
    L_FULL_ROW,
    L_BANK_ONLY,
    L_ONE_BYTE,
    L_OTHER_LANE,
    L_OTHER_ROW,
    L_OTHER_BANKS,
    L_DMA_READ,
    L_TWO_A,
    L_TWO_B,
    L_RERESERVE,
    NUM_LMODES
  };

  static const char* lrsc_name(int m) {
    static const char* names[NUM_LMODES] = {
      "dma-wr full row",
      "dma-wr reserved bank only",
      "dma-wr one byte of the lane",
      "dma-wr other lane of the word",
      "dma-wr other row, same bank",
      "dma-wr other banks, same row",
      "dma-rd same row",
      "two harts: A, DMA covers W_A",
      "two harts: B, other bank",
      "re-reserve: LR, DMA, LR, SC"
    };
    return names[m];
  }

  // Reservations are kept per bank word, so a write to the other .W lane of
  // a 64-bit word may break one, though it does not have to.
  static const char* lrsc_expect(int m) {
    switch (m) {
    case L_FULL_ROW:
    case L_BANK_ONLY:
    case L_ONE_BYTE:
    case L_TWO_A:
      return "fail in LR..SC";
    case L_OTHER_LANE:
      return "either in LR..SC";
    default:
      return "succeed";
    }
  }

  enum Pos { P_PRE, P_AT_LR, P_MID, P_AT_SC, P_POST, NUM_POS };

  static const char* pos_name(int p) {
    static const char* names[NUM_POS] = {"pre-LR", "@LR", "LR..SC", "@SC", "post-SC"};
    return names[p];
  }

  static constexpr int J_MIN = -2;
  static constexpr int J_MAX = 11;

  struct LrscCell {
    uint32_t runs  = 0;
    uint32_t fails = 0;
    uint32_t sc_ok = 0;
    uint32_t port_runs  = 0; // DMA accepted between the LR response and the SC port handshake
    uint32_t port_fails = 0;
  };

  using LrscCells = std::array<std::array<LrscCell, NUM_POS>, NUM_LMODES>;
  using LrscReported = std::array<std::array<uint32_t, NUM_POS>, NUM_LMODES>;

  struct LrscHart {
    uint32_t lane  = 0;
    uint32_t bank  = 0;
    uint32_t row   = 0;
    int      width = W_LO;
    bool     rereserve = false;
    LsuReq   lr, lr2, sc;
    uint64_t old = 0;
    int      lr_acc = -1, lr_rsp = -1, lr2_acc = -1, lr2_rsp = -1, sc_acc = -1, sc_rsp = -1;
    uint64_t lr_ret = 0, lr2_ret = 0, sc_ret = 0;
    int      stage = 0;   // request in flight: 0 LR, 1 second LR, 2 SC, 3 done
    int      next_at = 0; // cycle the stage's request is presented, -1 once presented
    bool     wait_dma = false;
  };

  struct LrscResult {
    int  pos   = P_PRE;
    bool sc_ok = false;
    bool port_between = false;
    std::vector<std::string> errs;
  };

  // Every byte the DMA writes differs from the word it replaces, so the LR's
  // return value and the final word show which side of the DMA each atomic
  // fell on.
  uint64_t fresh_bytes(uint64_t cur, uint32_t be) {
    uint64_t v = this->rand_word();
    for (uint32_t k = 0; k < WORD_SIZE; ++k) {
      if ((be >> k) & 1) {
        uint64_t b = ((cur >> (8 * k)) & 0xff) ^ (1 + this->rand_below(255));
        v = (v & ~(0xffull << (8 * k))) | (b << (8 * k));
      }
    }
    return v & WORD_MASK;
  }

  uint64_t sc_bytes(uint32_t byteen, uint64_t old, uint64_t dma_word) {
    uint64_t v = 0;
    for (uint32_t k = 0; k < WORD_SIZE; ++k) {
      if (!((byteen >> k) & 1)) {
        continue;
      }
      uint64_t o = (old >> (8 * k)) & 0xff;
      uint64_t d = (dma_word >> (8 * k)) & 0xff;
      uint64_t b;
      do {
        b = this->rand_below(256);
      } while (b == o || b == d);
      v |= b << (8 * k);
    }
    return v;
  }

  void lrsc_window() {
    LrscCells cells{};
    LrscReported reported{};
    uint32_t rep = 0;
    for (int mode = 0; mode < NUM_LMODES; ++mode) {
      if (mode == L_TWO_B) {
        continue; // tallied from the two-hart runs
      }
      if ((mode == L_OTHER_BANKS || mode == L_TWO_A) && NUM_BANKS < 2) {
        continue;
      }
      if (mode == L_TWO_A && NUM_REQS < 2) {
        continue;
      }
      const int j_min = (mode == L_RERESERVE) ? 0 : J_MIN;
      const int j_max = (mode == L_RERESERVE) ? 0 : J_MAX;
      for (int j = j_min; j <= j_max; ++j) {
        for (int gap : {1, 3}) {
          for (uint32_t bank = 0; bank < NUM_BANKS; ++bank) {
            for (int width = 0; width < NUM_WIDTHS; ++width) {
              if (mode == L_OTHER_LANE && lane_byteen(width) == FULL_BE) {
                continue;
              }
              for (uint32_t lane = 0; lane < NUM_REQS; ++lane) {
                this->lrsc_run(mode, j, gap, bank, lane, width, rep++, cells, reported);
              }
            }
          }
        }
      }
    }

    std::printf("\nLR/SC window summary (DMA acceptance against the LR and SC bank occupancies; cells = failed/runs):\n");
    std::printf("  %-31s %-17s", "mode", "SC must");
    for (int p = 0; p < NUM_POS; ++p) {
      std::printf(" %10s", pos_name(p));
    }
    std::printf("   SC ok in LR..SC   rsp(LR)<DMA<acc(SC)\n");
    for (int m = 0; m < NUM_LMODES; ++m) {
      std::printf("  %-31s %-17s", lrsc_name(m), lrsc_expect(m));
      for (int p = 0; p < NUM_POS; ++p) {
        const LrscCell& c = cells[m][p];
        if (c.runs == 0) {
          std::printf(" %10s", "-");
        } else if (c.fails == 0) {
          std::printf(" %10s", strf("ok %u", c.runs).c_str());
        } else {
          std::printf(" %10s", strf("FAIL %u/%u", c.fails, c.runs).c_str());
        }
      }
      const LrscCell& mid = cells[m][P_MID];
      std::printf("   %-17s", (mid.runs == 0) ? "-" : strf("%u/%u", mid.sc_ok, mid.runs).c_str());
      if (mid.port_runs == 0) {
        std::printf("   -\n");
      } else if (mid.port_fails == 0) {
        std::printf("   ok %u\n", mid.port_runs);
      } else {
        std::printf("   FAIL %u/%u\n", mid.port_fails, mid.port_runs);
      }
    }
    std::printf("\n");

    const bool two_harts = (NUM_BANKS >= 2 && NUM_REQS >= 2);
    auto require = [&](int m, int p) {
      if (cells[m][p].runs == 0) {
        this->fail(strf("FAIL: the LR/SC sweep never placed the DMA %s for '%s'", pos_name(p), lrsc_name(m)));
      }
    };
    for (int m : {L_FULL_ROW, L_BANK_ONLY, L_ONE_BYTE, L_TWO_A}) {
      if (m == L_TWO_A && !two_harts) {
        continue;
      }
      for (int p : {P_PRE, P_MID, P_POST}) {
        require(m, p);
      }
    }
    for (int m : {L_OTHER_ROW, L_OTHER_BANKS, L_DMA_READ, L_TWO_B}) {
      if ((m == L_OTHER_BANKS && NUM_BANKS < 2) || (m == L_TWO_B && !two_harts)) {
        continue;
      }
      require(m, P_MID);
    }
    if (WORD_SIZE == 8) {
      require(L_OTHER_LANE, P_MID);
    }
    require(L_RERESERVE, P_PRE);
  }

  void lrsc_run(int mode, int j, int gap, uint32_t bank, uint32_t lane, int width, uint32_t rep,
                LrscCells& cells, LrscReported& reported) {
    const bool two = (mode == L_TWO_A);
    std::vector<LrscHart> harts(two ? 2 : 1);
    harts[0].lane = lane;
    harts[0].bank = bank;
    harts[0].width = width;
    harts[0].row = this->rand_below(NUM_ROWS);
    harts[0].rereserve = (mode == L_RERESERVE);
    if (two) {
      harts[1].lane = (lane + 1) % NUM_REQS;
      harts[1].bank = (bank + 1 + this->rand_below(NUM_BANKS - 1)) % NUM_BANKS;
      harts[1].width = (width + 1) % NUM_WIDTHS;
      harts[1].row = (rep & 1) ? harts[0].row : this->rand_below(NUM_ROWS);
    }
    const uint32_t row_w = harts[0].row;
    const uint32_t lane_be = lane_byteen(width);

    DmaReq dma;
    dma.rw = (mode != L_DMA_READ);
    dma.row = row_w;
    switch (mode) {
    case L_FULL_ROW:
      dma.be.fill(FULL_BE);
      break;
    case L_BANK_ONLY:
      dma.be[bank] = FULL_BE;
      break;
    case L_ONE_BYTE: {
      uint32_t k;
      do {
        k = this->rand_below(WORD_SIZE);
      } while (!((lane_be >> k) & 1));
      dma.be[bank] = 1u << k;
      break;
    }
    case L_OTHER_LANE: {
      const uint32_t other = FULL_BE & ~lane_be;
      uint32_t be = other;
      if (rep & 1) {
        do {
          be = this->rand_below(FULL_BE + 1) & other;
        } while (be == 0);
      }
      dma.be[bank] = be;
      break;
    }
    case L_OTHER_ROW:
      // Flipping only the top row bit keeps the reservation station index,
      // so the write must be told apart by the full row address.
      dma.row = (rep & 1) ? (row_w ^ (NUM_ROWS / 2)) : this->other_row(row_w);
      dma.be.fill(FULL_BE);
      break;
    case L_OTHER_BANKS:
      dma.be.fill(FULL_BE);
      dma.be[bank] = 0;
      break;
    case L_TWO_A:
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        dma.be[b] = this->rand_below(FULL_BE + 1);
      }
      dma.be[bank] = FULL_BE;
      if (harts[1].row == row_w) {
        dma.be[harts[1].bank] = 0;
      }
      break;
    case L_RERESERVE:
      if (rep & 1) {
        dma.be.fill(FULL_BE);
      } else {
        dma.be[bank] = FULL_BE;
      }
      break;
    default:
      break;
    }
    for (uint32_t b = 0; b < NUM_BANKS; ++b) {
      dma.data[b] = this->fresh_bytes(this->gold(dma.row, b), dma.be[b]);
    }
    dma.tag = this->next_dma_tag();

    for (LrscHart& h : harts) {
      h.old = this->gold(h.row, h.bank);
      h.lr.amo = true;
      h.lr.op = AMO_LR;
      h.lr.byteen = lane_byteen(h.width);
      h.lr.addr = word_addr(h.row, h.bank);
      h.lr.tag = this->next_tag();
      h.lr2 = h.lr;
      h.lr2.tag = this->next_tag();
      h.sc = h.lr;
      h.sc.op = AMO_SC;
      const uint64_t dma_word = (dma.rw && dma.row == h.row) ? dma.data[h.bank] : h.old;
      h.sc.data = this->sc_bytes(h.lr.byteen, h.old, dma_word);
      h.sc.tag = this->next_tag();
    }

    std::vector<uint32_t> rows;
    auto add_row = [&](uint32_t r) {
      if (std::find(rows.begin(), rows.end(), r) == rows.end()) {
        rows.push_back(r);
      }
    };
    for (const LrscHart& h : harts) {
      add_row(h.row);
    }
    add_row(dma.row);
    auto row_idx = [&](uint32_t r) {
      return static_cast<size_t>(std::find(rows.begin(), rows.end(), r) - rows.begin());
    };
    std::vector<Row> before(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        before[i][b] = this->gold(rows[i], b);
      }
    }

    // Timed window; cycles are relative to the cycle the LR is presented.
    // The re-reserve DMA follows the first LR's response instead of j.
    const uint64_t cyc_base = dut_.cycle();
    const int rel0 = std::min(j, 0);
    std::optional<int> dma_at;
    if (mode != L_RERESERVE) {
      dma_at = j;
    }
    bool dma_sent = false;
    int d_acc = 0, d_rsp = 0;
    bool d_accepted = false, d_responded = false;
    Row d_data{};
    std::vector<std::string> common;
    auto stage_req = [](LrscHart& h) -> LsuReq& {
      return (h.stage == 0) ? h.lr : (h.stage == 1) ? h.lr2 : h.sc;
    };
    bool done = false;
    for (int rel = rel0; rel < rel0 + 128 && !done; ++rel) {
      if (!dma_sent && dma_at && rel >= *dma_at) {
        dut_.dma = dma;
        dma_sent = true;
      }
      for (LrscHart& h : harts) {
        if (h.stage < 3 && h.next_at >= 0 && rel >= h.next_at) {
          dut_.lane[h.lane] = stage_req(h);
          h.next_at = -1;
        }
      }
      Events ev = dut_.clock();
      if (ev.dma_acc) {
        d_acc = rel;
        d_accepted = true;
      }
      if (ev.dma_rsp) {
        if (dma.rw || d_responded || ev.dma_rsp_tag != dma.tag) {
          common.push_back(strf("unexpected DMA response at %+d tag=0x%x", rel, ev.dma_rsp_tag));
        } else {
          d_rsp = rel;
          d_responded = true;
          d_data = ev.dma_rsp_data;
        }
      }
      for (LrscHart& h : harts) {
        if (ev.lsu_acc[h.lane] && h.stage < 3) {
          ((h.stage == 0) ? h.lr_acc : (h.stage == 1) ? h.lr2_acc : h.sc_acc) = rel;
        }
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        if (!ev.lsu_rsp[l]) {
          continue;
        }
        LrscHart* h = nullptr;
        for (LrscHart& x : harts) {
          if (x.lane == l) {
            h = &x;
          }
        }
        if (!h || h->stage > 2 || ev.rsp_tag[l] != stage_req(*h).tag) {
          common.push_back(strf("stray LSU response at %+d: lane=%u tag=0x%x", rel, l, ev.rsp_tag[l]));
          continue;
        }
        if (h->stage == 0) {
          h->lr_rsp = rel;
          h->lr_ret = ev.rsp_data[l];
          if (h->rereserve) {
            h->stage = 1;
            h->wait_dma = true;
            dma_at = rel + 1;
          } else {
            h->stage = 2;
            h->next_at = rel + gap;
          }
        } else if (h->stage == 1) {
          h->lr2_rsp = rel;
          h->lr2_ret = ev.rsp_data[l];
          h->stage = 2;
          h->next_at = rel + gap;
        } else {
          h->sc_rsp = rel;
          h->sc_ret = ev.rsp_data[l];
          h->stage = 3;
        }
      }
      for (LrscHart& h : harts) {
        if (h.wait_dma && d_accepted) {
          h.wait_dma = false;
          h.next_at = rel + 1;
        }
      }
      done = d_accepted && (dma.rw || d_responded);
      for (const LrscHart& h : harts) {
        done = done && (h.stage == 3);
      }
    }
    if (!done) {
      common.push_back("handshakes did not complete within 128 cycles (deadlock?)");
      for (const LrscHart& h : harts) {
        dut_.lane[h.lane].reset();
      }
      dut_.dma.reset();
    }
    this->idle(4);

    std::vector<Row> after;
    for (uint32_t r : rows) {
      after.push_back(this->dma_read(r));
    }

    std::vector<Row> exp = before;
    if (dma.rw) {
      Row& er = exp[row_idx(dma.row)];
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        er[b] = dma_merge(dma, b, er[b]);
      }
    }

    std::array<LrscResult, 2> res;
    for (size_t k = 0; k < harts.size(); ++k) {
      const LrscHart& h = harts[k];
      LrscResult& r = res[k];
      const char* who = two ? ((k == 0) ? "A: " : "B: ") : "";
      const int t_lr1 = h.lr_rsp - rsp_lat_;
      const int t_lr2 = h.lr2_rsp - rsp_lat_;
      const int t_sc  = h.sc_rsp - rsp_lat_;
      const int t_lr  = h.rereserve ? t_lr2 : t_lr1;
      const int d = d_acc;
      r.pos = (d < t_lr) ? P_PRE : (d <= t_lr + 2) ? P_AT_LR : (d < t_sc) ? P_MID : (d <= t_sc + 2) ? P_AT_SC : P_POST;
      r.port_between = (r.pos == P_MID) && d > (h.rereserve ? h.lr2_rsp : h.lr_rsp) && d < h.sc_acc;
      if (!done) {
        r.errs.push_back(strf("%sincomplete window", who));
        continue;
      }
      const bool touches = !dma.rw || dma.be[h.bank] != 0;
      const bool writes = dma.rw && dma.row == h.row && dma.be[h.bank] != 0;
      const bool covers = writes && (dma.be[h.bank] & h.lr.byteen) != 0;
      auto D = [&](uint64_t x) { return writes ? dma_merge(dma, h.bank, x) : x; };
      auto S = [&](uint64_t x) { return amo_new_word(h.sc, x); };

      // A DMA beat must never be accepted on a bank while an atomic owns it.
      struct Atomic { const char* name; int acc; int t; };
      std::vector<Atomic> atomics{{"LR", h.lr_acc, t_lr1}};
      if (h.rereserve) {
        atomics.push_back({"second LR", h.lr2_acc, t_lr2});
      }
      atomics.push_back({"SC", h.sc_acc, t_sc});
      bool a1 = false;
      for (const Atomic& at : atomics) {
        if (at.t < at.acc + bank_lat_) {
          r.errs.push_back(strf("%s%s responded at %+d, too early for a request accepted at %+d",
                                who, at.name, at.t + rsp_lat_, at.acc));
          a1 = true;
        }
        if (touches && d >= at.t && d <= at.t + 2) {
          r.errs.push_back(strf("%sDMA %s accepted at %+d while the %s owned bank %u (%+d..%+d)",
                                who, dma.rw ? "write" : "read", d, at.name, h.bank, at.t, at.t + 2));
          a1 = true;
        }
      }
      if (h.rereserve && !(d > t_lr1 + 2 && d < t_lr2)) {
        r.errs.push_back(strf("%sDMA at %+d is not between the first LR (bank %+d) and the second (bank %+d)",
                              who, d, t_lr1, t_lr2));
      }

      const uint64_t ok_word = sc_rsp_word(h.sc, false);
      const uint64_t fail_word = sc_rsp_word(h.sc, true);
      r.sc_ok = (h.sc_ret == ok_word);
      if (!r.sc_ok && h.sc_ret != fail_word) {
        r.errs.push_back(strf("%sSC returned %s, expected %s (success) or %s (failure)", who,
                              hex(h.sc_ret).c_str(), hex(ok_word).c_str(), hex(fail_word).c_str()));
      }
      uint64_t fin = h.old;
      if (writes && d < t_sc) {
        fin = D(fin);
      }
      if (r.sc_ok) {
        fin = S(fin);
      }
      if (writes && d > t_sc + 2) {
        fin = D(fin);
      }
      exp[row_idx(h.row)][h.bank] = fin;
      if (a1) {
        continue;
      }

      if (h.rereserve && h.lr_ret != h.old) {
        r.errs.push_back(strf("%sfirst LR returned %s, expected %s", who, hex(h.lr_ret).c_str(), hex(h.old).c_str()));
      }
      const uint64_t lr_exp = (writes && d < t_lr) ? D(h.old) : h.old;
      const uint64_t lr_got = h.rereserve ? h.lr2_ret : h.lr_ret;
      if (lr_got != lr_exp) {
        r.errs.push_back(strf("%s%s returned %s, expected %s", who, h.rereserve ? "second LR" : "LR",
                              hex(lr_got).c_str(), hex(lr_exp).c_str()));
      }

      const bool between = (r.pos == P_MID);
      const uint64_t got_w = after[row_idx(h.row)][h.bank];
      if (covers && between && r.sc_ok) {
        r.errs.push_back(strf("%sSC succeeded although a DMA write to its reserved bytes (byteen 0x%x) was accepted at %+d, "
                              "after the LR (bank %+d) and before the SC (bank %+d); word is %s, the DMA data alone is %s%s",
                              who, dma.be[h.bank] & h.lr.byteen, d, t_lr, t_sc, hex(got_w).c_str(),
                              hex(D(h.old)).c_str(), (got_w == S(D(h.old))) ? " -> the SC overwrote the DMA write" : ""));
      } else if (!(writes && between) && !r.sc_ok) {
        r.errs.push_back(strf("%sSC failed although no DMA write to its word landed between the LR (bank %+d) and the SC "
                              "(bank %+d): DMA %s row 0x%x byteen 0x%x accepted at %+d -> reservation lost to an access "
                              "that does not cover it",
                              who, t_lr, t_sc, dma.rw ? "write" : "read", dma.row, dma.rw ? dma.be[h.bank] : 0, d));
      }

      if (!dma.rw && dma.row == h.row) {
        const uint64_t seen = (r.sc_ok && d > t_sc + 2) ? S(h.old) : h.old;
        if (d_data[h.bank] != seen) {
          r.errs.push_back(strf("%sDMA read of the reserved word returned %s, expected %s", who,
                                hex(d_data[h.bank]).c_str(), hex(seen).c_str()));
        }
      }
    }

    if (!dma.rw && done) {
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        bool reserved = false;
        for (const LrscHart& h : harts) {
          reserved |= (h.row == dma.row && h.bank == b);
        }
        uint64_t exp_rd = before[row_idx(dma.row)][b];
        if (!reserved && d_data[b] != exp_rd) {
          common.push_back(strf("DMA read (row 0x%x, bank %u): expected %s, got %s",
                                dma.row, b, hex(exp_rd).c_str(), hex(d_data[b]).c_str()));
        }
      }
    }

    for (size_t i = 0; i < rows.size(); ++i) {
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        if (after[i][b] == exp[i][b]) {
          continue;
        }
        std::string e = strf("word (row 0x%x, bank %u): expected %s, got %s",
                             rows[i], b, hex(exp[i][b]).c_str(), hex(after[i][b]).c_str());
        bool owned = false;
        for (size_t k = 0; k < harts.size(); ++k) {
          if (harts[k].row == rows[i] && harts[k].bank == b) {
            res[k].errs.push_back(e);
            owned = true;
          }
        }
        if (!owned) {
          common.push_back(e);
        }
      }
    }

    for (size_t i = 0; i < rows.size(); ++i) {
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        this->gold(rows[i], b) = after[i][b];
      }
    }

    bool run_ok = true;
    bool report = opts_.verbose;
    for (size_t k = 0; k < harts.size(); ++k) {
      const int m = (k == 0) ? mode : L_TWO_B;
      const LrscResult& r = res[k];
      LrscCell& c = cells[m][r.pos];
      const bool failed = !r.errs.empty() || !common.empty();
      ++c.runs;
      if (r.pos == P_MID && r.sc_ok) {
        ++c.sc_ok;
      }
      if (r.port_between) {
        ++c.port_runs;
        c.port_fails += failed;
      }
      if (failed) {
        ++c.fails;
        run_ok = false;
        uint32_t& nrep = reported[m][r.pos];
        if (nrep < opts_.max_report) {
          ++nrep;
          report = true;
        }
      }
    }
    if (!run_ok) {
      ++failures_;
    }
    if (!report) {
      return;
    }

    std::string msg = strf("%s lr/sc %s @%s lane=%u bank=%u width=%s gap=%d j=%+d row=0x%x",
                           run_ok ? "ok  " : "FAIL", lrsc_name(mode), pos_name(res[0].pos), lane, bank,
                           width_name(width), gap, j, row_w);
    if (two) {
      msg += strf(" | B @%s lane=%u bank=%u width=%s row=0x%x", pos_name(res[1].pos), harts[1].lane,
                  harts[1].bank, width_name(harts[1].width), harts[1].row);
    }
    msg += strf(" [cycle %" PRIu64 "]\n", cyc_base);
    for (size_t k = 0; k < harts.size(); ++k) {
      const LrscHart& h = harts[k];
      msg += strf("    timing %s(LR presented = +0): LR acc %+d rsp %+d (bank %+d)", two ? ((k == 0) ? "A " : "B ") : "",
                  h.lr_acc, h.lr_rsp, h.lr_rsp - rsp_lat_);
      if (h.rereserve) {
        msg += strf("; second LR acc %+d rsp %+d (bank %+d)", h.lr2_acc, h.lr2_rsp, h.lr2_rsp - rsp_lat_);
      }
      msg += strf("; SC acc %+d rsp %+d (bank %+d)\n", h.sc_acc, h.sc_rsp, h.sc_rsp - rsp_lat_);
    }
    msg += strf("    DMA %s row 0x%x", dma.rw ? "write" : "read", dma.row);
    if (dma.rw) {
      msg += " byteen";
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        msg += strf(" %0*x", static_cast<int>((WORD_SIZE + 3) / 4), dma.be[b]);
      }
    }
    msg += d_accepted ? strf(": accepted %+d", d_acc) : std::string(": not accepted");
    if (!dma.rw) {
      msg += d_responded ? strf(", rsp %+d", d_rsp) : std::string(", no response");
    }
    msg += "\n";
    for (const std::string& e : common) {
      msg += "    " + e + "\n";
    }
    for (size_t k = 0; k < harts.size(); ++k) {
      for (const std::string& e : res[k].errs) {
        msg += "    " + e + "\n";
      }
    }
    std::printf("%s", msg.c_str());
  }

  ///////////////////////////////////////////////////////////////////////////
  // Randomized stress. Every word is owned by exactly one agent so each
  // response has a single correct value no matter how the bank arbitration
  // interleaves: DMA owns half of each row, every lane owns a private set
  // of words, and a few shared counters see only AMOADD +1 from all lanes.

  struct Pending {
    uint32_t lane;
    int      counter; // shared counter index, or -1
    uint64_t expect;
    uint64_t issued;
    std::string desc;
    bool     ack = false; // a store acknowledgement: the data is don't-care
  };

  struct DmaPending {
    uint32_t tag;
    uint32_t row;
    Row      expect;
    uint64_t issued;
  };

  // A DMA write has no response to wait on, so it stalls only behind an
  // atomic that already owns one of its banks, for at most that atomic's
  // capture and write-back cycles.
  static constexpr uint32_t DMA_WR_STALL_BOUND = 2;

  static bool dma_owned(uint32_t row, uint32_t bank) {
    return ((row + bank) & 1) == 0;
  }

  void stress(uint32_t seed) {
    std::mt19937_64 rng(seed);
    auto rnd = [&](uint32_t n) { return static_cast<uint32_t>(rng() % n); };
    auto chance = [&](uint32_t pct) { return rnd(100) < pct; };

    const uint32_t rows = std::min<uint32_t>(NUM_ROWS, 16);
    std::vector<uint32_t> shared;
    std::array<std::vector<uint32_t>, NUM_REQS> priv;
    uint32_t next_owner = 0;
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t b = 0; b < NUM_BANKS; ++b) {
        if (dma_owned(row, b)) {
          continue;
        }
        uint32_t w = row * NUM_BANKS + b;
        if (row < 2 && AMO_EN) {
          shared.push_back(w);
        } else {
          priv[next_owner++ % NUM_REQS].push_back(w);
        }
      }
    }
    std::vector<uint64_t> sh_init(shared.size());
    std::vector<uint32_t> sh_count(shared.size(), 0);
    std::vector<std::vector<uint64_t>> sh_olds(shared.size());
    for (size_t j = 0; j < shared.size(); ++j) {
      sh_init[j] = gold_[shared[j]];
    }

    std::unordered_map<uint32_t, Pending> pend;
    std::deque<DmaPending> dma_pend;
    std::array<uint32_t, NUM_REQS> outstanding{};
    uint32_t errors = 0;
    uint64_t n_amo = 0, n_shared = 0, n_ld = 0, n_st = 0, n_dma_rd = 0, n_dma_wr = 0;
    uint64_t n_ack_lane_amo = 0, n_ack_lane_st = 0;
    uint64_t dma_stalls = 0, dma_wr_stalls = 0;
    uint32_t wr_stall_run = 0, wr_stall_max = 0;
    const uint64_t start = dut_.cycle();
    uint64_t last_progress = start;
    const uint64_t st0 = dut_.accepted, cm0 = dut_.commits;

    auto report = [&](const std::string& m) {
      ++errors;
      if (errors <= 10) {
        this->fail(strf("FAIL stress seed=%u [cycle %" PRIu64 "] %s", seed, dut_.cycle(), m.c_str()));
      } else {
        ++failures_;
      }
    };

    static const uint8_t priv_ops[] = {AMO_ADD, AMO_SWAP, AMO_XOR, AMO_OR, AMO_AND, AMO_MIN, AMO_MAX, AMO_CAS};
    const uint32_t num_priv_ops = CAS_EN ? 8 : 7;

    auto gen_lane = [&](uint32_t l) {
      if (priv[l].empty() && shared.empty()) {
        return;
      }
      LsuReq r;
      r.tag = this->next_tag();
      if (reports_commits(l, true)) {
        r.wid = static_cast<int>(rnd(NUM_WARPS));
        r.counted = rnd(4) != 0;
      }
      Pending p{l, -1, 0, dut_.cycle(), ""};
      uint32_t kind = rnd(100);
      if (AMO_EN && !shared.empty() && (kind < 25 || priv[l].empty())) {
        uint32_t j = rnd(static_cast<uint32_t>(shared.size()));
        uint32_t w = shared[j];
        r.amo = true;
        r.op = AMO_ADD;
        r.byteen = FULL_BE;
        r.addr = word_addr(w / NUM_BANKS, w % NUM_BANKS);
        r.data = 1;
        p.counter = static_cast<int>(j);
        p.desc = strf("AMOADD +1 shared counter %u (row 0x%x, bank %u)", j, w / NUM_BANKS, w % NUM_BANKS);
        ++sh_count[j];
        ++n_shared;
      } else {
        uint32_t w = priv[l][rnd(static_cast<uint32_t>(priv[l].size()))];
        uint64_t& g = gold_[w];
        r.addr = word_addr(w / NUM_BANKS, w % NUM_BANKS);
        if (AMO_EN && kind < 65) {
          int width = static_cast<int>(rnd(NUM_WIDTHS));
          r.amo = true;
          r.op = priv_ops[rnd(num_priv_ops)];
          r.uns = (r.op == AMO_MIN || r.op == AMO_MAX) && chance(50);
          r.byteen = this->lane_byteen(width);
          r.data = lane_position(width, rng());
          if (r.op == AMO_CAS) {
            uint64_t cur = (XLEN == 64 && width == W_HI) ? (g >> 32) : g;
            if (XLEN == 32 || width != W_D) {
              cur &= MASK32;
            }
            r.cmp = chance(50) ? cur : (rng() & WORD_MASK);
          }
          p.expect = g;
          p.desc = strf("%s.%s private (row 0x%x, bank %u)", amo_name(r.op, r.uns), width_name(width),
                        w / NUM_BANKS, w % NUM_BANKS);
          g = amo_new_word(r, g);
          ++n_amo;
          n_ack_lane_amo += acks_stores(l);
        } else if (kind < 82) {
          r.byteen = FULL_BE;
          p.expect = g;
          p.desc = strf("load private (row 0x%x, bank %u)", w / NUM_BANKS, w % NUM_BANKS);
          ++n_ld;
        } else {
          r.rw = true;
          r.byteen = 1 + rnd(FULL_BE);
          r.data = rng() & WORD_MASK;
          g = merge(g, r.data, r.byteen);
          ++n_st;
          n_ack_lane_st += acks_stores(l);
        }
      }
      if (!r.rw || acks_stores(l)) {
        p.ack = r.rw;
        if (p.ack) {
          p.desc = strf("store ack (row 0x%x, bank %u)", r.addr >> BANK_BITS, r.addr & (NUM_BANKS - 1));
        }
        pend[r.tag] = p;
        ++outstanding[l];
      }
      dut_.lane[l] = r;
    };

    auto gen_dma = [&]() {
      uint32_t row = rnd(rows);
      DmaReq d;
      d.row = row;
      d.tag = this->next_dma_tag();
      d.rw = chance(50);
      if (d.rw) {
        bool any = false;
        for (uint32_t b = 0; b < NUM_BANKS; ++b) {
          d.data[b] = rng() & WORD_MASK;
          d.be[b] = (dma_owned(row, b) && chance(60)) ? rnd(FULL_BE + 1) : 0;
          any |= (d.be[b] != 0);
        }
        if (!any) {
          uint32_t b = (row & 1) % NUM_BANKS;
          if (!dma_owned(row, b)) {
            return;
          }
          d.be[b] = FULL_BE;
        }
        for (uint32_t b = 0; b < NUM_BANKS; ++b) {
          this->gold(row, b) = dma_merge(d, b, this->gold(row, b));
        }
        ++n_dma_wr;
      } else {
        DmaPending dp{d.tag, row, {}, dut_.cycle()};
        for (uint32_t b = 0; b < NUM_BANKS; ++b) {
          dp.expect[b] = this->gold(row, b);
        }
        dma_pend.push_back(dp);
        ++n_dma_rd;
      }
      dut_.dma = d;
    };

    auto busy = [&]() {
      if (!pend.empty() || !dma_pend.empty() || dut_.dma) {
        return true;
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        if (dut_.lane[l]) {
          return true;
        }
      }
      return false;
    };

    bool draining = false;
    while (true) {
      uint64_t now = dut_.cycle();
      if (!draining && now - start >= opts_.stress_cycles) {
        draining = true;
      }
      if (draining && !busy()) {
        break;
      }
      if (now - last_progress > 2000) {
        std::string st;
        for (uint32_t l = 0; l < NUM_REQS; ++l) {
          st += strf(" lane%u{valid=%d out=%u}", l, dut_.lane[l].has_value(), outstanding[l]);
        }
        report(strf("no progress for 2000 cycles (deadlock): dma valid=%d, dma reads pending=%zu,%s",
                    dut_.dma.has_value(), dma_pend.size(), st.c_str()));
        dut_.dma.reset();
        for (uint32_t l = 0; l < NUM_REQS; ++l) {
          dut_.lane[l].reset();
        }
        break;
      }
      if (!draining) {
        for (uint32_t l = 0; l < NUM_REQS; ++l) {
          if (!dut_.lane[l] && outstanding[l] < 4 && chance(60)) {
            gen_lane(l);
          }
        }
        if (!dut_.dma && chance(45)) {
          gen_dma();
        }
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        dut_.rsp_ready[l] = chance(80);
      }
      dut_.dma_rsp_ready = chance(70);

      const bool dma_is_wr = dut_.dma && dut_.dma->rw;
      Events ev = dut_.clock();
      if (ev.any()) {
        last_progress = dut_.cycle();
      }
      dma_stalls += ev.dma_stall;
      dma_wr_stalls += ev.dma_stall && dma_is_wr;
      wr_stall_run = (ev.dma_stall && dma_is_wr) ? wr_stall_run + 1 : 0;
      wr_stall_max = std::max(wr_stall_max, wr_stall_run);
      if (wr_stall_run == DMA_WR_STALL_BOUND + 1) {
        report(strf("DMA write to row 0x%x stalled more than %u cycles: a pending DMA must block new atomics "
                    "on its banks so the in-flight ones drain", dut_.dma->row, DMA_WR_STALL_BOUND));
      }
      for (uint32_t l = 0; l < NUM_REQS; ++l) {
        if (!ev.lsu_rsp[l]) {
          continue;
        }
        auto it = pend.find(ev.rsp_tag[l]);
        if (it == pend.end()) {
          report(strf("lane %u: response with unknown tag 0x%x", l, ev.rsp_tag[l]));
          continue;
        }
        Pending p = it->second;
        pend.erase(it);
        --outstanding[p.lane];
        if (p.lane != l) {
          report(strf("tag 0x%x issued on lane %u answered on lane %u", ev.rsp_tag[l], p.lane, l));
        }
        if (p.ack) {
          continue;
        }
        if (p.counter >= 0) {
          sh_olds[p.counter].push_back(ev.rsp_data[l]);
        } else if (ev.rsp_data[l] != p.expect) {
          report(strf("lane %u %s issued at cycle %" PRIu64 ": expected %s, got %s",
                      l, p.desc.c_str(), p.issued, hex(p.expect).c_str(), hex(ev.rsp_data[l]).c_str()));
        }
      }
      if (ev.dma_rsp) {
        if (dma_pend.empty()) {
          report(strf("unexpected DMA response tag 0x%x", ev.dma_rsp_tag));
        } else {
          DmaPending dp = dma_pend.front();
          dma_pend.pop_front();
          if (ev.dma_rsp_tag != dp.tag) {
            report(strf("DMA response tag 0x%x, expected 0x%x (out of order)", ev.dma_rsp_tag, dp.tag));
          }
          for (uint32_t b = 0; b < NUM_BANKS; ++b) {
            if (dma_owned(dp.row, b) && ev.dma_rsp_data[b] != dp.expect[b]) {
              report(strf("DMA read (row 0x%x, bank %u) issued at cycle %" PRIu64 ": expected %s, got %s",
                          dp.row, b, dp.issued, hex(dp.expect[b]).c_str(), hex(ev.dma_rsp_data[b]).c_str()));
            }
          }
        }
      }
    }
    dut_.rsp_ready.fill(true);
    dut_.dma_rsp_ready = true;
    this->idle(4);
    if (COMMIT_EN) {
      this->idle(4);
      for (uint32_t w = 0; w < NUM_WARPS; ++w) {
        if (dut_.uncommitted[w] != 0) {
          report(strf("warp %u: %" PRIu64 " LSU-input access(es) never reported committed", w, dut_.uncommitted[w]));
        }
      }
      this->check_commit_errors("stress");
    }

    for (size_t j = 0; j < shared.size(); ++j) {
      uint64_t mask = WORD_MASK;
      std::vector<uint64_t> offs;
      for (uint64_t v : sh_olds[j]) {
        offs.push_back((v - sh_init[j]) & mask);
      }
      std::sort(offs.begin(), offs.end());
      for (size_t i = 0; i < offs.size(); ++i) {
        if (offs[i] != i) {
          report(strf("shared counter %zu (row 0x%x, bank %u): AMOADD returned old value init+%" PRIu64
                      " where init+%zu was due; %u increments issued, %zu answered",
                      j, shared[j] / NUM_BANKS, shared[j] % NUM_BANKS, offs[i], i, sh_count[j], offs.size()));
          break;
        }
      }
      gold_[shared[j]] = (sh_init[j] + sh_count[j]) & mask;
    }

    std::printf("stress seed=%u: %" PRIu64 " cycles, lsu %" PRIu64 " amo + %" PRIu64 " shared-amo + %" PRIu64
                " ld + %" PRIu64 " st, dma %" PRIu64 " rd + %" PRIu64 " wr, dma stall cycles %" PRIu64
                " (writes %" PRIu64 ", longest %u), %u error(s)\n",
                seed, dut_.cycle() - start, n_amo, n_shared, n_ld, n_st, n_dma_rd, n_dma_wr,
                dma_stalls, dma_wr_stalls, wr_stall_max, errors);
    if (ACK_LANE >= 0) {
      std::printf("stress seed=%u: store-ack lane %d: %" PRIu64 " acknowledged stores, %" PRIu64 " private atomics\n",
                  seed, ACK_LANE, n_ack_lane_st, n_ack_lane_amo);
      if (n_ack_lane_st == 0) {
        report("the store-ack lane issued no store");
      }
    }
    if (COMMIT_EN) {
      std::printf("stress seed=%u: %" PRIu64 " LSU-input accesses, %" PRIu64
                  " commits reported in total, per-warp balance 0\n",
                  seed, dut_.accepted - st0, dut_.commits - cm0);
    }
  }

  void final_check() {
    std::vector<uint32_t> rows(NUM_ROWS);
    for (uint32_t r = 0; r < NUM_ROWS; ++r) {
      rows[r] = r;
    }
    uint32_t before = failures_;
    this->check_commit_errors("final");
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    this->check_mbar_errors("final");
    std::printf("mbar event stream: %" PRIu64 " inserts and %" PRIu64 " flips matched to their ops; "
                "%" PRIu64 " inserts and %" PRIu64 " flips timed alone\n", dut_.mbar_inserts, dut_.mbar_flips,
                mbar_event_checks_[1], mbar_event_checks_[2]);
    if (dut_.mbar_inserts == 0 || dut_.mbar_flips == 0 || mbar_event_checks_[1] == 0 || mbar_event_checks_[2] == 0) {
      this->fail("FAIL mbar event stream: no insert or no flip was checked");
    }
  #endif
    this->check_rows("final memory", rows);
    std::printf("final memory check: %u mismatching word(s) over %u rows\n", failures_ - before, NUM_ROWS);
  }

  Options opts_;
  std::mt19937_64 rng_;
  std::vector<uint64_t> gold_;
  Dut dut_;
  int bank_lat_ = 1;
  int rsp_lat_ = 2;
  uint32_t tag_ = 0;
  uint32_t dma_tag_ = 0;
  uint32_t failures_ = 0;
  uint32_t commit_runs_ = 0;
};

static void usage(const char* prog) {
  std::printf("usage: %s [--seed=N] [--stress-seeds=N] [--stress-cycles=N] [--max-report=N] [--verbose] "
              "[--neg=underflow|txrange|align]\n", prog);
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  Options opts;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (a[0] == '+') {
      continue;
    }
    if (std::strncmp(a, "--seed=", 7) == 0) {
      opts.seed = static_cast<uint32_t>(std::strtoul(a + 7, nullptr, 0));
    } else if (std::strncmp(a, "--stress-seeds=", 15) == 0) {
      opts.stress_seeds = static_cast<uint32_t>(std::strtoul(a + 15, nullptr, 0));
    } else if (std::strncmp(a, "--stress-cycles=", 16) == 0) {
      opts.stress_cycles = static_cast<uint32_t>(std::strtoul(a + 16, nullptr, 0));
    } else if (std::strncmp(a, "--max-report=", 13) == 0) {
      opts.max_report = static_cast<uint32_t>(std::strtoul(a + 13, nullptr, 0));
    } else if (std::strcmp(a, "--verbose") == 0) {
      opts.verbose = true;
    } else if (std::strncmp(a, "--neg=", 6) == 0) {
      opts.neg = a + 6;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  Testbench tb(opts);
  return tb.run();
}
