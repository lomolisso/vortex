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

#include <VX_config.h>

#ifdef VX_CFG_H100_SM_ENABLE

#include "sm_sram.h"
#include "mem_block_pool.h"
#include "amo_ops.h"
#include <l1_slot_ref.h>
#include <bitmanip.h>
#include <array>
#include <cstring>
#include <deque>
#include <sstream>
#include <stdexcept>

using namespace vortex;
namespace sbr = vortex::smem_bank_ref;

namespace {

constexpr uint32_t BANKS      = SmSram::BANKS;
constexpr uint32_t BANK_BYTES = SmSram::BANK_BYTES;
constexpr uint32_t BANK_W     = BANK_BYTES * 8;
constexpr uint32_t ROW_BYTES  = SmSram::ROW_BYTES;
constexpr uint32_t ROWS       = SmSram::ROWS;
constexpr uint32_t LANES      = SmSram::LANES;
constexpr uint32_t WORD       = SmSram::WORD;
constexpr uint32_t HPW        = SmSram::HPW;
constexpr uint32_t NH         = SmSram::NH;
constexpr uint32_t PAIRS      = SmSram::PAIRS;
constexpr uint32_t RD_ROWS    = SmSram::RD_ROWS;
constexpr uint32_t WR_ROWS    = SmSram::WR_ROWS;
constexpr uint32_t RD_AMO     = RD_ROWS;
constexpr uint32_t RD_LSU     = RD_ROWS + 1;
constexpr uint32_t RD_N       = RD_ROWS + 2;
constexpr uint32_t WR_CPA     = WR_ROWS;
constexpr uint32_t WR_LSU     = WR_ROWS + 1;
constexpr uint32_t WR_N       = WR_ROWS + 2;
constexpr uint32_t NAC        = sbr::AMO_N;
constexpr uint32_t A_LSU      = sbr::AMO_LSU;
constexpr uint32_t A_MBAR     = sbr::AMO_MBAR;
constexpr uint64_t BANK_MASK  = (BANK_W >= 64) ? ~0ull : ((1ull << BANK_W) - 1);
constexpr uint32_t BMASK      = VX_CFG_MEM_BLOCK_SIZE - 1;
constexpr uint64_t WIN_MASK   = (1ull << VX_CFG_LMEM_LOG_SIZE) - 1;

static_assert(LANES <= 32 && NH <= 64, "the array takes at most 32 lanes");
static_assert(SmSram::RS <= SmSram::ROWS, "the LR/SC stations of a bank are indexed by row");

struct AmoAttr {
  bool     valid = false;
  MemOp    op = MemOp::LD;
  bool     uns = false;
  uint64_t cmp = 0;
  uint32_t hart = 0;
  uint32_t wid = 0;
};

// One warp access in the stage's lane form (VX_sm_sram wf_item_t).
struct WfItem {
  bool     global = false;
  bool     rw = false;
  uint32_t blk = 0;
  uint64_t tag = 0;
  uint32_t seq = 0;
  uint32_t mask = 0;
  std::array<uint32_t, LANES> row{};
  std::array<uint32_t, LANES> bank{};
  std::array<uint32_t, LANES> byteen{};
  std::array<uint64_t, LANES> data{};
  std::array<AmoAttr, LANES>  attr{};
  std::array<uint32_t, LANES> boff{};  // the lane word's byte offset in its memory block
  bool     amo = false;
  bool     mbar = false;
  bool     cas = false;   // its comparands follow as a second beat
  bool     cpa = false;   // a copy unit's mbarrier op: answered on its port
  uint32_t cid = 0;
  uint32_t wid = 0;
  uint64_t uuid = 0;
  uint32_t lacc = 0;  // lanes the DLMEM translation counted

  uint32_t half_row(uint32_t h) const { return row[h / HPW]; }
  uint32_t half_bank(uint32_t h) const { return (bank[h / HPW] + h % HPW) % BANKS; }
  uint32_t half_be(uint32_t h) const { return (byteen[h / HPW] >> ((h % HPW) * BANK_BYTES)) & ((1u << BANK_BYTES) - 1); }
  uint64_t half_data(uint32_t h) const { return (data[h / HPW] >> ((h % HPW) * BANK_W)) & BANK_MASK; }
  uint64_t touched() const {
    uint64_t rem = 0;
    for (uint32_t h = 0; h < NH; ++h) {
      if (((mask >> (h / HPW)) & 1) && this->half_be(h) != 0)
        rem |= 1ull << h;
    }
    return rem;
  }
};

// One wavefront engine's grant: per bank the row of its first pending bank
// word, every pending word of that row served with it.
struct Eng {
  std::array<bool, BANKS>     sel{};
  std::array<uint32_t, BANKS> sel_row{};
  std::array<uint32_t, BANKS> wren{};
  std::array<uint64_t, BANKS> wdata{};
  uint64_t served = 0;

  void eval(const WfItem& it, uint64_t rem) {
    for (int h = int(NH) - 1; h >= 0; --h) {
      if ((rem >> h) & 1) {
        sel[it.half_bank(h)] = true;
        sel_row[it.half_bank(h)] = it.half_row(h);
      }
    }
    for (uint32_t h = 0; h < NH; ++h) {
      if (((rem >> h) & 1) && sel_row[it.half_bank(h)] == it.half_row(h)) {
        served |= 1ull << h;
        uint32_t bk = it.half_bank(h);
        uint32_t be = it.half_be(h);
        uint64_t d  = it.half_data(h);
        for (uint32_t k = 0; k < BANK_BYTES; ++k) {
          if ((be >> k) & 1) {
            wren[bk] |= 1u << k;
            wdata[bk] = (wdata[bk] & ~(0xffull << (8 * k))) | (d & (0xffull << (8 * k)));
          }
        }
      }
    }
  }
};

struct Rsp {
  bool     global = false;
  bool     cpa = false;
  uint32_t blk = 0;
  uint64_t tag = 0;
  uint32_t mask = 0;
  std::array<uint64_t, LANES> data{};
  std::array<uint32_t, LANES> boff{};
  uint32_t cid = 0;
  uint64_t uuid = 0;
};

struct Wb {
  bool     v = false;
  uint32_t row = 0;
  uint32_t be = 0;
  uint64_t data = 0;
};

struct ClientWr {
  uint32_t row = 0;
  uint32_t be = 0;
};

// The AMO path's compute-stage register of one bank (VX_sm_sram c_*).
struct CReg {
  bool     v = false;
  bool     wide = false;
  uint32_t row = 0;
  uint32_t cl = 0;
  uint32_t ln = 0;
  uint64_t rhs = 0;
  AmoAttr  attr;
  uint64_t old = 0;  // the bank word read at the grant, forwarding applied
};

struct HeadLane {
  uint32_t cnt = 0;
  uint32_t fbank = 0;
  bool     wide = false;
  bool     ride = false;  // taken with its leader's pass, never picked itself
  uint32_t row = 0;
  uint64_t data = 0;
  AmoAttr  attr;
};

struct Rsv {
  bool     v = false;
  bool     wide = false;
  uint32_t row = 0;
  uint32_t hart = 0;
  uint32_t hold = 0;
};

struct FinRec {
  bool     v = false;
  bool     cpa = false;
  uint32_t blk = 0;
  uint64_t tag = 0;
  uint32_t mask = 0;
  std::array<uint32_t, LANES> boff{};
  uint32_t cid = 0;
  uint64_t uuid = 0;
  uint32_t wid = 0;
  uint32_t lacc = 0;
};

uint64_t load_word(const std::shared_ptr<mem_block_t>& blk, uint32_t off) {
  uint64_t w = 0;
  if (blk)
    std::memcpy(&w, blk->data() + off, WORD);
  return w;
}

std::shared_ptr<mem_block_t> word_block(uint64_t w, uint32_t off) {
  auto blk = make_mem_block();
  std::memset(blk->data(), 0, blk->size());
  std::memcpy(blk->data() + off, &w, WORD);
  return blk;
}

} // namespace

class SmSram::Impl {
public:
  Impl(SmSram* simobject, const Config& config)
    : simobject_(simobject)
    , config_(config)
    , mem_(size_t(ROWS) * ROW_BYTES, 0)
  {
    cfg_ = SmSram::cfg_of(VX_CFG_SMEM_PER_SM_MAX);
    rd_lvl_ = {VX_CFG_SM_SRAM_RD_PRI_TCU, VX_CFG_SM_SRAM_RD_PRI_DXA, VX_CFG_SM_SRAM_RD_PRI_DSMEM,
               VX_CFG_SM_SRAM_RD_PRI_AMO, VX_CFG_SM_SRAM_RD_PRI_LSU};
    wr_lvl_ = {VX_CFG_SM_SRAM_WR_PRI_L1FILL, VX_CFG_SM_SRAM_WR_PRI_DXA, VX_CFG_SM_SRAM_WR_PRI_DSMEM,
               VX_CFG_SM_SRAM_WR_PRI_CPASYNC, VX_CFG_SM_SRAM_WR_PRI_LSU};
    wf_lvl_ = {VX_CFG_SM_SRAM_WF_PRI_SHARED, VX_CFG_SM_SRAM_WF_PRI_GLOBAL};
    amo_lvl_ = {VX_CFG_SM_SRAM_AMO_PRI_DXA, VX_CFG_SM_SRAM_AMO_PRI_DSMEM,
                VX_CFG_SM_SRAM_AMO_PRI_MBAR, VX_CFG_SM_SRAM_AMO_PRI_LSU};
    if (BANKS * BANK_BYTES != ROW_BYTES)
      throw std::runtime_error("the banks must span one array row");
    if (WORD % BANK_BYTES != 0 || HPW != 2)
      throw std::runtime_error("the AMO units sit on even/odd bank pairs of one lane word");
    if (MRG_BYTES < 1 || MRG_BYTES > ROW_BYTES)
      throw std::runtime_error("VX_CFG_SM_SRAM_DSMEM_WR_BYTES must be 1..one array row");
    if ((uint64_t(ROWS) * ROW_BYTES) > (1ull << VX_CFG_LMEM_LOG_SIZE))
      throw std::runtime_error("the shared-memory window must cover the array");
  }

  void reset() {
    cfg_ = SmSram::cfg_of(VX_CFG_SMEM_PER_SM_MAX);
    sh_rr_ = wf_rr_ = rd_rr_ = wr_rr_ = cpa_rr_ = 0;
    pair_rr_.fill(0);
    hold_v_.fill(false);
    hold_c_.fill(0);
    stg_valid_ = false;
    stg_rem_ = 0;
    lsu_used_.fill(0);
    mbq_used_.fill(0);
    cmp_have_ = false;
    cmp_.fill(0);
    acc_.fill(0);
    for (auto& q : rspq_) q.clear();
    mbq_.clear();
    mrq_.clear();
    mrg_valid_ = mrg_open_ = false;
    mrg_busy_sent_ = false;
    mrg_row_ = 0;
    mrg_bytes_ = 0;
    mrg_be_.assign(ROW_BYTES, 0);
    mrg_data_.assign(ROW_BYTES, 0);
    cpa_valid_ = false;
    cpa_rem_ = 0;
    ack_valid_ = false;
    for (auto& c : creg_) c = CReg();
    cfol_ = 0;
    cfol_ln_ = 0;
    wb_cur_.fill(Wb());
    bw_prev_.fill(ClientWr());
    for (auto& b : rsv_) for (auto& r : b) r = Rsv();
    rs_acc_ = {};
    cap_push_.clear();
    lsu_fin_ = FinRec();
    mbq_fin_ = FinRec();
    amo_rsp_v_.fill(false);
    perf_stats_ = PerfStats();
  }

  void tick() {
    if (!simobject_->CfgIn.empty()) {
      cfg_ = simobject_->CfgIn.peek();
      simobject_->CfgIn.pop();
    }

    // Registered state seen this cycle.
    const bool sh_room = rspq_[0].size() < RSP_DEPTH - 2;
    const bool gl_room = rspq_[1].size() < RSP_DEPTH - 2;
    const bool mbq_empty = mbq_.empty();
    const bool mbq_full = (mbq_.size() == MBQ_DEPTH);
    const bool mbq_go = !mbq_empty && (mrq_.size() + (mbq_fin_.v ? 1 : 0) < MRQ_DEPTH);
    const bool stg_valid = stg_valid_;
    const bool ack_valid = ack_valid_;

    this->release();

    // C: the AMO units on the words read at the previous grant.
    std::array<Wb, BANKS> wb_next{};
    std::array<bool, BANKS> c_store{}, c_own{};
    this->compute(wb_next, c_store, c_own);

    for (auto& r : cap_push_)
      rspq_[r.global ? 1 : 0].push_back(r);
    cap_push_.clear();

    const uint32_t nblk = config_.num_blocks;

    // cp.async input: picked while its stage is free. A copy unit's mbarrier
    // op (cp.async.mbarrier.arrive) goes to the mbarrier op queue instead,
    // ahead of the LSU's: the warps' phase checks wait for it.
    uint32_t cpa_idx = nblk;
    uint32_t cpa_rr = cpa_rr_;
    WfItem cpa_in;
    if (!cpa_valid_) {
      bool req[32] = {};
      uint8_t lvl[32] = {};
      for (uint32_t b = 0; b < nblk; ++b)
        req[b] = !simobject_->CpaReqIn.at(b).empty();
      cpa_idx = sbr::level_pick(lvl, req, nblk, cpa_rr);
      if (cpa_idx < nblk) {
        cpa_in = this->shared_item(cpa_idx, simobject_->CpaReqIn.at(cpa_idx).peek());
        cpa_in.cpa = true;
      }
    }
    const bool cpa_mb = (cpa_idx < nblk) && cpa_in.mbar;
    const bool cpa_take = (cpa_idx < nblk) && (!cpa_mb || !mbq_full);

    // Shared head and the stage's pick.
    uint32_t sh_idx = nblk;
    {
      bool req[32] = {};
      for (uint32_t b = 0; b < nblk; ++b)
        req[b] = !simobject_->LsuReqIn.at(b).empty();
      uint8_t lvl[32] = {};
      uint32_t rr = sh_rr_;
      sh_idx = sbr::level_pick(lvl, req, nblk, rr);
    }
    WfItem sh_item;
    if (sh_idx < nblk)
      sh_item = this->shared_item(sh_idx, simobject_->LsuReqIn.at(sh_idx).peek());
    const bool mbq_take_ok = !mbq_full && !cpa_mb;

    bool pick = false;
    bool pick_gl = false;
    if (!stg_valid) {
      bool req[2] = {sh_idx < nblk && sh_room && (!sh_item.mbar || mbq_take_ok),
                     !simobject_->GlReqIn.empty() && gl_room};
      uint32_t w = sbr::level_pick(wf_lvl_.data(), req, 2, wf_rr_);
      if (w < 2) {
        pick = true;
        pick_gl = (w == 1);
        cur_ = pick_gl ? this->global_item(simobject_->GlReqIn.peek()) : sh_item;
        this->check_pick(cur_);
      }
    }
    const bool cur_valid = stg_valid || pick;
    const uint64_t cur_rem = !cur_valid ? 0 : (stg_valid ? stg_rem_ : cur_.touched());

    // A compare-and-swap crosses the 128-B LSU path twice (VX_mem_unit
    // g_cas_beats): the picked beat brings its swap values, the block's next
    // beat its comparands; its lanes reach the AMO path once those are in.
    const bool cmp_take = stg_valid && cur_.cas && !cmp_have_ && !simobject_->LsuReqIn.at(cur_.blk).empty();
    const bool cmp_in = !cur_valid || !cur_.cas || cmp_have_ || cmp_take;
    std::array<uint64_t, LANES> cmp_cur = cmp_;
    if (cmp_take) {
      const auto& beat = simobject_->LsuReqIn.at(cur_.blk).peek();
      for (uint32_t i = 0; i < LANES; ++i)
        cmp_cur[i] = beat.amo_cmp.at(i);
    }

    Eng eng0, eng1;
    eng0.eval(cur_, cur_rem);
    eng1.eval(cpa_, cpa_valid_ ? cpa_rem_ : 0);

    // AMO path: the clients' head ops and each pair's winner.
    std::array<std::array<HeadLane, LANES>, NAC> ah{};
    uint32_t mb_lead = LANES, mb_ride = 0;
    this->amo_heads(ah, cur_valid && cmp_in, cmp_cur, stg_valid, !mbq_go, mb_lead, mb_ride);

    std::array<bool, PAIRS> pr_v{}, pr_ev{}, pr_ov{};
    std::array<uint32_t, PAIRS> pr_c{}, pr_e{}, pr_o{};
    bool amo_rd_req = false;
    for (uint32_t p = 0; p < PAIRS; ++p) {
      std::array<uint32_t, NAC> on_e{}, on_o{};
      bool has[NAC] = {};
      for (uint32_t a = 0; a < NAC; ++a) {
        for (uint32_t i = 0; i < LANES; ++i) {
          const auto& l = ah[a][i];
          if (l.cnt == 0 || l.ride)
            continue;
          if (l.fbank == 2 * p)
            on_e[a] |= 1u << i;
          if (l.fbank == 2 * p + 1 || (l.wide && l.fbank == 2 * p))
            on_o[a] |= 1u << i;
        }
        has[a] = (on_e[a] | on_o[a]) != 0;
      }
      uint32_t pk = NAC;
      if (!hold_v_[p])
        pk = sbr::level_pick(amo_lvl_.data(), has, NAC, pair_rr_[p]);
      pr_v[p] = hold_v_[p] || pk < NAC;
      pr_c[p] = hold_v_[p] ? hold_c_[p] : (pk < NAC ? pk : 0);
      const uint32_t ce = on_e[pr_c[p]], co = on_o[pr_c[p]];
      const bool e_v = ce != 0, o_v = co != 0;
      const uint32_t e_idx = e_v ? __builtin_ctz(ce) : 0;
      const uint32_t o_idx = o_v ? __builtin_ctz(co) : 0;
      const bool w_e = ah[pr_c[p]][e_idx].wide, w_o = ah[pr_c[p]][o_idx].wide;
      const bool split = e_v && o_v && e_idx != o_idx && (w_e || w_o);
      pr_ev[p] = e_v && !(split && o_idx < e_idx);
      pr_ov[p] = o_v && !(split && e_idx < o_idx) && !(pr_ev[p] && e_idx == o_idx);
      pr_e[p] = e_idx;
      pr_o[p] = (o_v && e_v && e_idx == o_idx) ? e_idx : o_idx;
      amo_rd_req |= pr_v[p];
    }

    // Read port and write port.
    uint32_t rd_gnt;
    {
      bool req[RD_N];
      for (uint32_t c = 0; c < RD_ROWS; ++c)
        req[c] = !simobject_->RowRdIn.at(c).empty();
      req[RD_AMO] = amo_rd_req;
      req[RD_LSU] = cur_valid && !cur_.rw && !cur_.amo && !cur_.mbar;
      rd_gnt = sbr::level_pick(rd_lvl_.data(), req, RD_N, rd_rr_);
    }
    bool amo_wb_now = false;
    for (auto& w : wb_cur_)
      amo_wb_now |= w.v;
    uint32_t wr_gnt = WR_N;
    if (!amo_wb_now) {
      bool req[WR_N];
      req[0] = !simobject_->WrIn.at(0).empty();
      req[1] = !simobject_->WrIn.at(1).empty();
      req[2] = mrg_valid_ && !mrg_open_;
      req[WR_CPA] = cpa_valid_ && cpa_rem_ != 0 && !ack_valid;
      req[WR_LSU] = cur_valid && cur_.rw && !cur_.amo && !cur_.mbar && cur_rem != 0;
      wr_gnt = sbr::level_pick(wr_lvl_.data(), req, WR_N, wr_rr_);
    }
    const bool amo_gnt = (rd_gnt == RD_AMO);

    // This cycle's client write per bank.
    std::array<ClientWr, BANKS> bw{};
    const SmSramFill* fill_in = nullptr;
    if (wr_gnt < 2) {
      fill_in = &simobject_->WrIn.at(wr_gnt).peek();
      if (wr_gnt == 0 && fill_in->row < ROWS - cfg_.l1_lines)
        this->fail("refill row outside the L1", fill_in->row);
      if (wr_gnt == 1 && fill_in->row >= cfg_.s_rows)
        this->fail("DXA row outside shared memory", fill_in->row);
    }
    for (uint32_t bk = 0; bk < BANKS; ++bk) {
      if (fill_in) {
        bw[bk].row = fill_in->row;
        for (uint32_t k = 0; k < BANK_BYTES; ++k)
          bw[bk].be |= (fill_in->byteen.at(bk * BANK_BYTES + k) ? 1u : 0u) << k;
      } else if (wr_gnt == 2) {
        bw[bk].row = mrg_row_;
        for (uint32_t k = 0; k < BANK_BYTES; ++k)
          bw[bk].be |= (mrg_be_.at(bk * BANK_BYTES + k) ? 1u : 0u) << k;
      } else if (wr_gnt == WR_CPA && eng1.sel[bk]) {
        bw[bk] = {eng1.sel_row[bk], eng1.wren[bk]};
      } else if (wr_gnt == WR_LSU && eng0.sel[bk]) {
        bw[bk] = {eng0.sel_row[bk], eng0.wren[bk]};
      }
    }

    // W: the write-backs skip bytes a client wrote at G or C (logically later).
    for (uint32_t bk = 0; bk < BANKS; ++bk) {
      uint32_t late = 0;
      if (bw_prev_[bk].row == creg_[bk].row) late |= bw_prev_[bk].be;
      if (bw[bk].row == creg_[bk].row) late |= bw[bk].be;
      wb_next[bk].be = ~late & ((1u << BANK_BYTES) - 1);
      wb_next[bk].row = creg_[bk].row;
    }

    // Reads: the array before this cycle's write, with the AMO write-backs
    // of this cycle and the next (granted before the read).
    auto bank_read = [&](uint32_t bk, uint32_t row) {
      uint64_t d = 0;
      std::memcpy(&d, &mem_[size_t(row) * ROW_BYTES + bk * BANK_BYTES], BANK_BYTES);
      for (const Wb* w : {&wb_cur_[bk], &wb_next[bk]}) {
        if (!w->v || w->row != row)
          continue;
        for (uint32_t k = 0; k < BANK_BYTES; ++k) {
          if ((w->be >> k) & 1)
            d = (d & ~(0xffull << (8 * k))) | (w->data & (0xffull << (8 * k)));
        }
      }
      return d;
    };

    if (rd_gnt < RD_ROWS) {
      auto& in = simobject_->RowRdIn.at(rd_gnt);
      const auto& rq = in.peek();
      if (rq.row >= cfg_.s_rows)
        this->fail("row read outside shared memory", rq.row);
      SmSramRowRsp rsp;
      rsp.tag = rq.tag;
      rsp.data.resize(ROW_BYTES);
      for (uint32_t bk = 0; bk < BANKS; ++bk) {
        uint64_t d = bank_read(bk, rq.row);
        std::memcpy(&rsp.data[bk * BANK_BYTES], &d, BANK_BYTES);
      }
      simobject_->RowRspOut.at(rd_gnt).send(rsp, SM_SRAM_BANK_STAGES + SM_SRAM_RSP_STAGES);
      in.pop();
      ++perf_stats_.reads;
    }
    if (rd_gnt == RD_LSU) {
      for (uint32_t h = 0; h < NH; ++h) {
        if ((eng0.served >> h) & 1) {
          uint32_t i = h / HPW, k = h % HPW;
          uint64_t d = bank_read(cur_.half_bank(h), cur_.half_row(h));
          acc_[i] = (acc_[i] & ~(BANK_MASK << (k * BANK_W))) | (d << (k * BANK_W));
        }
      }
      ++perf_stats_.reads;
    }

    // G: the AMO grant's bank reads and the compute-stage registers.
    std::array<std::array<bool, LANES>, NAC> tk{};
    std::array<CReg, BANKS> cnext{};
    for (uint32_t p = 0; p < PAIRS; ++p) {
      const uint32_t a = pr_c[p];
      if (amo_gnt && pr_v[p]) {
        if (pr_ev[p]) tk[a][pr_e[p]] = true;
        if (pr_ov[p]) tk[a][pr_o[p]] = true;
      }
      for (uint32_t k = 0; k < 2; ++k) {
        const uint32_t bk = 2 * p + k;
        const uint32_t l = (k == 0) ? pr_e[p] : (pr_ov[p] ? pr_o[p] : pr_e[p]);
        const auto& hl = ah[a][l];
        auto& c = cnext[bk];
        bool bv = false;
        if (pr_v[p] && pr_ev[p] && (k == 0 || ah[a][pr_e[p]].wide))
          bv = true;
        if (pr_v[p] && pr_ov[p] && k == 1)
          bv = true;
        c.v    = amo_gnt && bv;
        c.row  = hl.row;
        c.cl   = a;
        c.ln   = l;
        c.attr = hl.attr;
        c.wide = (k == 0) && pr_ev[p] && hl.wide;
        c.rhs  = (k == 0) ? hl.data : (hl.data >> 32);
        c.old  = c.v ? bank_read(bk, hl.row) : 0;
      }
    }
    const bool mb_lead_tk = (mb_lead < LANES) && tk[A_MBAR][mb_lead];
    for (uint32_t i = 0; i < LANES; ++i) {
      if (mb_lead_tk && ((mb_ride >> i) & 1))
        tk[A_MBAR][i] = true;
    }
    if (amo_gnt)
      ++perf_stats_.amo_passes;

    // Writes.
    if (amo_wb_now) {
      for (uint32_t bk = 0; bk < BANKS; ++bk) {
        if (wb_cur_[bk].v)
          this->bank_write(bk, wb_cur_[bk].row, wb_cur_[bk].be, wb_cur_[bk].data);
      }
    } else if (wr_gnt < WR_N) {
      for (uint32_t bk = 0; bk < BANKS; ++bk) {
        uint64_t d = 0;
        if (fill_in) {
          std::memcpy(&d, &fill_in->data[bk * BANK_BYTES], BANK_BYTES);
        } else if (wr_gnt == 2) {
          std::memcpy(&d, &mrg_data_[bk * BANK_BYTES], BANK_BYTES);
        } else if (wr_gnt == WR_CPA) {
          d = eng1.wdata[bk];
        } else {
          d = eng0.wdata[bk];
        }
        if (bw[bk].be != 0)
          this->bank_write(bk, bw[bk].row, bw[bk].be, d);
      }
      ++perf_stats_.writes;
      if (wr_gnt == 0)
        simobject_->FillDoneOut.send(fill_in->row, 0);
      if (wr_gnt < 2)
        simobject_->WrIn.at(wr_gnt).pop();
    }

    // LR/SC reservations: a committed write to a reserved row breaks it, an SC
    // gives up its own, then a load-reserved claims its station unless another
    // hart's live one holds it, which spends one refusal. An 8-B reservation
    // lives in the even bank's station and also breaks on a write to the odd
    // bank's word.
    for (uint32_t bk = 0; bk < BANKS; ++bk) {
      auto& tb = rsv_[bk];
      const auto& c = creg_[bk];
      const uint32_t bx = bw[bk].row & (SmSram::RS - 1);
      const uint32_t cx = c.row & (SmSram::RS - 1);
      const Rsv ob = tb[bx], oc = tb[cx];
      bool o_brk_b = false, o_brk_c = false;
      uint32_t ox = 0, oy = 0;
      if ((bk & 1) == 0) {
        const auto& co = creg_[bk + 1];
        ox = bw[bk + 1].row & (SmSram::RS - 1);
        oy = co.row & (SmSram::RS - 1);
        const Rsv obo = tb[ox], oco = tb[oy];
        o_brk_b = bw[bk + 1].be != 0 && !wb_cur_[bk + 1].v && obo.v && obo.wide && obo.row == bw[bk + 1].row;
        o_brk_c = c_store[bk + 1] && oco.v && oco.wide && oco.row == co.row;
      }
      if (bw[bk].be != 0 && !wb_cur_[bk].v && ob.v && ob.row == bw[bk].row)
        tb[bx].v = false;
      const bool c_hit = oc.v && oc.row == c.row;
      const bool st = ((bk & 1) && creg_[bk - 1].wide) ? c_store[bk - 1] : c_store[bk];
      if (st && c_hit)
        tb[cx].v = false;
      if (o_brk_b)
        tb[ox].v = false;
      if (o_brk_c)
        tb[oy].v = false;
      if (!c_own[bk])
        continue;
      if (c.attr.op == MemOp::AMO_SC && c_hit && oc.hart == c.attr.hart)
        tb[cx].v = false;
      if (c.attr.op == MemOp::AMO_LR) {
        if (!oc.v || oc.hold == 0 || oc.hart == c.attr.hart)
          tb[cx] = {true, (bk & 1) == 0 && c.wide, c.row, c.attr.hart, SmSram::RS_HOLD};
        else
          tb[cx].hold = oc.hold - 1;
      }
    }

    // AMO client progress.
    uint32_t lsu_left = 0, mbq_left = 0;
    bool tk_lsu = false, tk_mbq = false;
    for (uint32_t i = 0; i < LANES; ++i) {
      if (ah[A_LSU][i].cnt - (tk[A_LSU][i] ? 1 : 0) != 0) lsu_left |= 1u << i;
      if (ah[A_MBAR][i].cnt - (tk[A_MBAR][i] ? 1 : 0) != 0) mbq_left |= 1u << i;
      tk_lsu |= tk[A_LSU][i];
      tk_mbq |= tk[A_MBAR][i];
    }
    const bool mbq_last = mbq_go && mbq_left == 0 && tk_mbq;

    const bool cur_grant = (rd_gnt == RD_LSU && !cur_.rw) || (wr_gnt == WR_LSU && cur_.rw);
    const uint64_t cur_left = cur_rem & ~(cur_grant ? eng0.served : 0);
    const bool cur_done = cur_valid
      && (cur_.mbar || (cur_.amo ? (cmp_in && lsu_left == 0) : (cur_left == 0 && (cur_.rw || cur_grant))));
    if (cur_grant)
      ++perf_stats_.wavefronts;


    for (uint32_t i = 0; i < LANES; ++i) {
      lsu_used_[i] = (cur_done || !cur_valid) ? 0 : ((stg_valid ? lsu_used_[i] : 0) + (tk[A_LSU][i] ? 1 : 0));
      mbq_used_[i] = mbq_last ? 0 : (mbq_used_[i] + (tk[A_MBAR][i] ? 1 : 0));
    }

    for (uint32_t a = 0; a < 2; ++a) {
      amo_rsp_v_[a] = tk[a][0];
      if (tk[a][0]) {
        const auto& rq = simobject_->AmoIn.at(a).peek();
        amo_rsp_req_[a] = {rq.tag, rq.hart_id, rq.uuid};
        amo_rsp_boff_[a] = (rq.addr & WIN_MASK & ~uint64_t(WORD - 1)) & BMASK;
        simobject_->AmoIn.at(a).pop();
      }
    }

    // The op whose last lane was granted answers at C.
    lsu_fin_.v = cur_valid && cur_.amo && cur_done;
    if (lsu_fin_.v) {
      lsu_fin_.blk = cur_.blk; lsu_fin_.tag = cur_.tag; lsu_fin_.mask = cur_.mask;
      lsu_fin_.boff = cur_.boff; lsu_fin_.cid = cur_.cid; lsu_fin_.uuid = cur_.uuid;
      lsu_fin_.wid = cur_.wid; lsu_fin_.lacc = cur_.lacc;
    }
    mbq_fin_.v = mbq_last;
    if (mbq_last) {
      const auto& hd = mbq_.front();
      mbq_fin_.cpa = hd.cpa;
      mbq_fin_.blk = hd.blk; mbq_fin_.tag = hd.tag; mbq_fin_.mask = hd.mask;
      mbq_fin_.boff = hd.boff; mbq_fin_.cid = hd.cid; mbq_fin_.uuid = hd.uuid;
      mbq_fin_.wid = hd.wid; mbq_fin_.lacc = hd.lacc;
      mbq_.pop_front();
    }

    for (uint32_t p = 0; p < PAIRS; ++p) {
      hold_v_[p] = pr_v[p] && !amo_gnt;
      hold_c_[p] = pr_c[p];
    }

    // Counted lanes leave a cycle after their final register: a wavefront
    // at its done cycle, an atomic and an mbarrier op at their C stage.
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    if (cur_done && !cur_.global && !cur_.amo && !cur_.mbar && cur_.lacc != 0)
      simobject_->CommitOut.at(0).send(WtInc{cur_.wid, cur_.lacc}, 1);
    if (lsu_fin_.v && lsu_fin_.lacc != 0)
      simobject_->CommitOut.at(1).send(WtInc{lsu_fin_.wid, lsu_fin_.lacc}, 2);
    if (mbq_fin_.v && mbq_fin_.lacc != 0)
      simobject_->CommitOut.at(2).send(WtInc{mbq_fin_.wid, mbq_fin_.lacc}, 2);
#endif

    // Stage progress.
    if (cur_done && cur_.global)
      simobject_->WfRetiredOut.send(cur_.seq, 0);
    if (rd_gnt == RD_LSU && cur_done) {
      Rsp r;
      r.global = cur_.global; r.blk = cur_.blk; r.tag = cur_.tag; r.mask = cur_.mask;
      r.data = acc_; r.boff = cur_.boff;
      r.cid = cur_.cid; r.uuid = cur_.uuid;
      cap_push_.push_back(r);
      acc_.fill(0);
    }
    if (cmp_take) {
      simobject_->LsuReqIn.at(cur_.blk).pop();
      cmp_ = cmp_cur;
    }
    cmp_have_ = !cur_done && (cmp_have_ || cmp_take);
    if (pick) {
      if (pick_gl) {
        simobject_->GlReqIn.pop();
      } else {
        if (!cur_.cas)
          simobject_->LsuReqIn.at(sh_idx).pop();
        sh_rr_ = (sh_idx + 1) % nblk;
      }
      if (!pick_gl && cur_.mbar) {
        mbq_.push_back(cur_);
        if (mbq_.size() > MBQ_DEPTH)
          this->fail("mbarrier op queue overflow", 0);
      }
    }
    stg_valid_ = cur_valid && !cur_done;
    if (cur_valid) {
      stg_rem_ = cur_left;
    }

    // DSMEM inputs of one row merge into writes of up to MRG_BYTES; an input
    // is one lane's message or a whole slot, so the merge counts bytes.
    {
      auto& in = simobject_->WrIn.at(2);
      const bool wv = !in.empty();
      uint32_t in_bytes = 0;
      if (wv) {
        for (auto b : in.peek().byteen)
          in_bytes += (b != 0);
      }
      const bool fit = wv && mrg_valid_ && mrg_open_ && in.peek().row == mrg_row_ && mrg_bytes_ + in_bytes <= MRG_BYTES;
      const bool mfree = !mrg_valid_ || wr_gnt == 2;
      if (wv && (fit || mfree)) {
        const auto& s = in.peek();
        mrg_valid_ = true;
        if (fit) {
          for (uint32_t k = 0; k < ROW_BYTES; ++k) {
            if (s.byteen[k]) {
              mrg_be_[k] = 1;
              mrg_data_[k] = s.data[k];
            }
          }
          mrg_bytes_ += in_bytes;
        } else {
          if (s.row >= cfg_.s_rows)
            this->fail("DSMEM row outside shared memory", s.row);
          mrg_row_ = s.row;
          mrg_be_ = s.byteen;
          mrg_data_ = s.data;
          mrg_bytes_ = in_bytes;
        }
        mrg_open_ = !s.last && mrg_bytes_ < MRG_BYTES;
        in.pop();
      } else if (wr_gnt == 2) {
        mrg_valid_ = false;
      } else if (mrg_valid_ && mrg_open_) {
        mrg_open_ = false;
      }
    }
    if (mrg_valid_ != mrg_busy_sent_) {
      simobject_->DsmemBusyOut.send(mrg_valid_, 1);
      mrg_busy_sent_ = mrg_valid_;
    }

    // cp.async stage: one warp write at a time, acknowledged once written.
    {
      const uint64_t cpa_left = cpa_rem_ & ~(wr_gnt == WR_CPA ? eng1.served : 0);
      if (!cpa_valid_) {
        if (cpa_take) {
          simobject_->CpaReqIn.at(cpa_idx).pop();
          cpa_rr_ = cpa_rr;
          if (cpa_mb) {
            mbq_.push_back(cpa_in);
            if (mbq_.size() > MBQ_DEPTH)
              this->fail("mbarrier op queue overflow", 0);
          } else {
            cpa_ = cpa_in;
            cpa_rem_ = cpa_.touched();
            cpa_valid_ = true;
          }
        }
      } else if (cpa_left == 0 && !ack_valid) {
        cpa_valid_ = false;
        ack_valid_ = true;
        ack_ = cpa_;
      } else {
        cpa_rem_ = cpa_left;
      }
    }

    creg_ = cnext;
    cfol_ = (amo_gnt && mb_lead_tk) ? mb_ride : 0;
    cfol_ln_ = (mb_lead < LANES) ? mb_lead : 0;
    wb_cur_ = wb_next;
    bw_prev_ = bw;
  }

  const PerfStats& perf_stats() const { return perf_stats_; }

  void read(void* data, uint64_t addr, uint32_t size) const {
    if (addr + size > mem_.size())
      throw std::out_of_range("SM array read out of range");
    std::memcpy(data, &mem_[addr], size);
  }

  void write(const void* data, uint64_t addr, uint32_t size) {
    if (addr + size > mem_.size())
      throw std::out_of_range("SM array write out of range");
    std::memcpy(&mem_[addr], data, size);
  }

private:
  [[noreturn]] void fail(const char* what, uint32_t row) const {
    std::ostringstream ss;
    ss << simobject_->name() << ": " << what << " (row 0x" << std::hex << row << ")";
    throw std::runtime_error(ss.str());
  }

  void bank_write(uint32_t bk, uint32_t row, uint32_t be, uint64_t d) {
    uint8_t* p = &mem_[size_t(row) * ROW_BYTES + bk * BANK_BYTES];
    for (uint32_t k = 0; k < BANK_BYTES; ++k) {
      if ((be >> k) & 1)
        p[k] = uint8_t(d >> (8 * k));
    }
  }

  WfItem shared_item(uint32_t b, const LsuReq& req) const {
    WfItem it;
    it.blk = b;
    it.tag = req.tag;
    it.rw = req.is_write();
    it.cid = req.cid;
    it.wid = req.wid;
    it.uuid = req.uuid;
    bool any_amo = false, any_mbar = false;
    for (uint32_t i = 0; i < LANES; ++i) {
      if (!req.mask.test(i))
        continue;
      it.mask |= 1u << i;
      const uint64_t la = req.addrs.at(i) & WIN_MASK;
      const uint64_t wbase = la & ~uint64_t(WORD - 1);
      it.row[i] = uint32_t(la / ROW_BYTES) & ROW_MASK;
      it.bank[i] = uint32_t((la / WORD) * HPW) % BANKS;
      it.boff[i] = uint32_t(wbase & BMASK);
      it.byteen[i] = uint32_t(req.byteen.at(i) >> it.boff[i]) & ((1u << WORD) - 1);
      it.data[i] = load_word(req.data.at(i), it.boff[i]);
      auto& at = it.attr[i];
#ifdef VX_CFG_EXT_MBAR_ENABLE
      const bool mbar = memop_is_mbar(req.op);
#else
      const bool mbar = false;
#endif
      at.valid = req.is_amo() || mbar;
      at.op = req.op;
      at.uns = req.flags.amo_unsigned;
      it.cas |= at.valid && req.op == MemOp::AMO_CAS;
      at.wid = req.wid;
      at.hart = req.wid * VX_CFG_NUM_THREADS + req.tids.at(i);
      any_amo |= at.valid;
      any_mbar |= at.valid && mbar;
    }
    it.amo = any_amo && !any_mbar;
    it.mbar = any_mbar;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    it.lacc = req.flags.lacc ? uint32_t(__builtin_popcount(it.mask)) : 0;
#endif
    return it;
  }

  WfItem global_item(const SmSramReq& req) const {
    if (req.op != SmSramOp::READ && req.op != SmSramOp::WRITE)
      this->fail("global AMO before the AMO path", 0);
    WfItem it;
    it.global = true;
    it.rw = (req.op == SmSramOp::WRITE);
    it.tag = req.tag;
    it.seq = req.seq;
    for (uint32_t i = 0; i < LANES; ++i) {
      if (req.mask.test(i))
        it.mask |= 1u << i;
      const auto& l = req.lanes.at(i);
      it.row[i] = l.row;
      it.bank[i] = (l.boff / BANK_BYTES) % BANKS;
      it.byteen[i] = l.byteen;
      it.data[i] = l.data;
    }
    return it;
  }

  void check_pick(const WfItem& it) const {
    for (uint32_t i = 0; i < LANES; ++i) {
      if (!((it.mask >> i) & 1))
        continue;
      const bool ok = it.global ? (it.row[i] >= ROWS - cfg_.l1_lines) : (it.row[i] < cfg_.s_rows);
      if (!ok)
        this->fail("lane row outside its region", it.row[i]);
    }
  }

  void amo_heads(std::array<std::array<HeadLane, LANES>, NAC>& ah, bool cur_valid,
                 const std::array<uint64_t, LANES>& cmp_cur, bool stg_valid, bool mbq_empty,
                 uint32_t& mb_lead, uint32_t& mb_ride) {
    auto fbank_of = [](uint32_t bank, uint32_t byteen) {
      return (bank + (((byteen & ((1u << BANK_BYTES) - 1)) == 0) ? 1 : 0)) % BANKS;
    };
    for (uint32_t a = 0; a < 2; ++a) {
      auto& in = simobject_->AmoIn.at(a);
      if (in.empty())
        continue;
      const auto& rq = in.peek();
      const uint64_t la = rq.addr & WIN_MASK;
      const uint64_t wbase = la & ~uint64_t(WORD - 1);
      const uint32_t boff = uint32_t(wbase & BMASK);
      const uint32_t be = uint32_t(rq.byteen >> boff) & ((1u << WORD) - 1);
      auto& l = ah[a][0];
      l.cnt = 1;
      l.fbank = fbank_of(uint32_t((la / WORD) * HPW) % BANKS, be);
      l.wide = (be == (1u << WORD) - 1);
      l.row = uint32_t(la / ROW_BYTES) & ROW_MASK;
      l.data = load_word(rq.data, boff);
      l.attr.valid = true;
      l.attr.op = rq.op;
      l.attr.uns = rq.flags.amo_unsigned;
      l.attr.cmp = rq.amo_cmp;
      l.attr.hart = rq.hart_id;
      l.attr.wid = (rq.hart_id / VX_CFG_NUM_THREADS) % VX_CFG_NUM_WARPS;
    }
    const WfItem* mh = mbq_empty ? nullptr : &mbq_.front();
    for (uint32_t i = 0; i < LANES; ++i) {
      if (mh) {
        auto& l = ah[A_MBAR][i];
        const uint32_t need = (((mh->mask >> i) & 1) && mh->attr[i].valid) ? 1 : 0;
        l.cnt = need - mbq_used_[i];
        l.fbank = fbank_of(mh->bank[i], mh->byteen[i]);
        l.wide = (mh->byteen[i] == (1u << WORD) - 1);
        l.row = mh->row[i];
        l.data = mh->data[i];
        l.attr = mh->attr[i];
      }
      auto& l = ah[A_LSU][i];
      const uint32_t need = (cur_valid && cur_.amo && ((cur_.mask >> i) & 1) && cur_.attr[i].valid) ? 1 : 0;
      l.cnt = need - (stg_valid ? lsu_used_[i] : 0);
      l.fbank = fbank_of(cur_.bank[i], cur_.byteen[i]);
      l.wide = (cur_.byteen[i] == (1u << WORD) - 1);
      l.row = cur_.row[i];
      l.data = cur_.data[i];
      l.attr = cur_.attr[i];
      l.attr.cmp = cmp_cur.at(i);
    }
#ifdef VX_CFG_EXT_MBAR_ENABLE
    // Phase checks of one word read it once, as a broadcast load does: the lowest
    // pending check leads, the checks of the same word, op and token phase ride on it.
    auto is_chk = [](const HeadLane& l) {
      return l.cnt != 0 && (l.attr.op == MemOp::MBAR_TEST_WAIT || l.attr.op == MemOp::MBAR_TRY_WAIT);
    };
    for (uint32_t i = 0; i < LANES; ++i) {
      auto& l = ah[A_MBAR][i];
      if (!is_chk(l))
        continue;
      if (mb_lead == LANES) {
        mb_lead = i;
        continue;
      }
      const auto& d = ah[A_MBAR][mb_lead];
      if (l.attr.op == d.attr.op && l.row == d.row && l.fbank == d.fbank && l.wide == d.wide
          && ((l.data ^ d.data) >> (8 * WORD - 1)) == 0) {
        l.ride = true;
        mb_ride |= 1u << i;
      }
    }
#else
    (void)mb_lead;
    (void)mb_ride;
#endif
  }

  // C: per pair a 64-bit unit on the even bank and a 32-bit one on the odd;
  // an 8-B lane is the even bank's op over both banks.
  void compute(std::array<Wb, BANKS>& wb_next, std::array<bool, BANKS>& c_store, std::array<bool, BANKS>& c_own) {
    std::array<std::array<uint64_t, LANES>, NAC> acc = rs_acc_;
    for (uint32_t a = 0; a < 2; ++a)
      acc[a].fill(0);
    std::array<uint64_t, BANKS> new_w{};
    for (uint32_t p = 0; p < PAIRS; ++p) {
      const bool is_wide = creg_[2 * p].wide;
      c_own[2 * p] = creg_[2 * p].v;
      c_own[2 * p + 1] = creg_[2 * p + 1].v && !is_wide;
      for (uint32_t k = 0; k < 2; ++k) {
        const uint32_t bk = 2 * p + k;
        const auto& c = creg_[bk];
        const auto& at = c.attr;
        const bool wide_op = (k == 0) && is_wide;
        const uint64_t old = wide_op ? ((creg_[2 * p + 1].old << 32) | (c.old & 0xffffffffull)) : (c.old & 0xffffffffull);
        const uint8_t width = wide_op ? 3 : 2;
        const bool fin = c_own[bk];
        const bool is_lr = (at.op == MemOp::AMO_LR);
        const bool is_sc = (at.op == MemOp::AMO_SC);
        bool is_wait = false, wait_done = false, flip = false;
        AmoComputeResult rmw{old, old};
        if (fin) {
#ifdef VX_CFG_EXT_MBAR_ENABLE
          if (memop_is_mbar(at.op)) {
            if (!wide_op)
              this->fail("mbarrier op is not a naturally aligned doubleword", c.row);
            const auto r = mbar_ref::apply(mbar_op_code(at.op), old, c.rhs);
            if (r.err != mbar_ref::ERR_NONE)
              this->fail("mbarrier op error", c.row);
            is_wait = mbar_ref::is_wait(mbar_op_code(at.op));
            wait_done = is_wait && ((old >> 63) != (c.rhs >> 63));
            flip = r.flip;
            rmw = {r.new_word, r.ret_word};
          } else
#endif
          {
#ifdef VX_CFG_EXT_FPAMO_ENABLE
            rmw = amo_compute_ptx(at.op, width, old, c.rhs, at.uns, at.cmp, amo_ref::SITE_LOOP);
#else
            rmw = amo_compute(at.op, width, old, c.rhs, at.uns, at.cmp);
#endif
          }
        }
        const auto& rv = rsv_[bk][c.row & (SmSram::RS - 1)];
        const bool sc_ok = rv.v && rv.row == c.row && rv.hart == at.hart;
        c_store[bk] = fin && !is_lr && !is_wait && !(is_sc && !sc_ok);
        new_w[bk] = rmw.new_word;
        const uint64_t ret_v = is_sc ? (sc_ok ? 0 : 1) : (is_wait ? (wait_done ? 1 : 0) : rmw.ret_word);
        const uint64_t ret = (k == 0) ? (wide_op ? ret_v : (ret_v & 0xffffffffull)) : ((ret_v & 0xffffffffull) << 32);
        if (fin) {
          acc[c.cl][c.ln] |= ret;
          if (c.cl == A_MBAR && c.ln == cfol_ln_) {
            for (uint32_t i = 0; i < LANES; ++i) {
              if ((cfol_ >> i) & 1)
                acc[A_MBAR][i] |= ret;
            }
          }
        }
#ifdef VX_CFG_EXT_MBAR_ENABLE
        const bool evt_v = fin && ((at.op == MemOp::MBAR_TRY_WAIT && !wait_done) || flip);
        if (evt_v)
          simobject_->MbarEvtOut.at(bk).send(MbarBankEvt{flip, at.wid, c.row}, 0);
#else
        (void)flip;
#endif
      }
      for (uint32_t k = 0; k < 2; ++k) {
        const uint32_t bk = 2 * p + k;
        wb_next[bk].v = (k == 1 && is_wide) ? c_store[2 * p] : c_store[bk];
        wb_next[bk].data = ((k == 1 && is_wide) ? (new_w[2 * p] >> 32) : new_w[bk]) & BANK_MASK;
      }
    }

    for (uint32_t a = 0; a < 2; ++a) {
      if (!amo_rsp_v_[a])
        continue;
      MemRsp rsp{amo_rsp_req_[a].tag, amo_rsp_req_[a].hart_id, amo_rsp_req_[a].uuid,
                 word_block(acc[a][0], amo_rsp_boff_[a])};
      simobject_->AmoRspOut.at(a).send(rsp, 0);
    }
    if (lsu_fin_.v) {
      Rsp r;
      r.blk = lsu_fin_.blk; r.tag = lsu_fin_.tag; r.mask = lsu_fin_.mask;
      r.data = acc[A_LSU]; r.boff = lsu_fin_.boff;
      r.cid = lsu_fin_.cid; r.uuid = lsu_fin_.uuid;
      rspq_[0].push_back(r);
      acc[A_LSU].fill(0);
    }
    if (mbq_fin_.v) {
      Rsp r;
      r.cpa = mbq_fin_.cpa;
      r.blk = mbq_fin_.blk; r.tag = mbq_fin_.tag; r.mask = mbq_fin_.mask;
      r.data = acc[A_MBAR]; r.boff = mbq_fin_.boff;
      r.cid = mbq_fin_.cid; r.uuid = mbq_fin_.uuid;
      mrq_.push_back(r);
      acc[A_MBAR].fill(0);
    }
    rs_acc_ = acc;
    for (uint32_t a = 0; a < 2; ++a)
      rs_acc_[a].fill(0);
  }

  LsuRsp lsu_rsp(const Rsp& r) const {
    LsuRsp out(LANES);
    out.tag = r.tag;
    out.cid = r.cid;
    out.uuid = r.uuid;
    for (uint32_t i = 0; i < LANES; ++i) {
      if ((r.mask >> i) & 1) {
        out.mask.set(i);
        out.data.at(i) = word_block(r.data[i], r.boff[i]);
      }
    }
    return out;
  }

  // Responses, per class in order; the in-order shared responses first,
  // then the mbarrier ops'. A copy unit's mbarrier op answers on its port
  // when no write acknowledgement leaves there.
  void release() {
    const bool sh_v = !rspq_[0].empty();
    const bool m_cpa = !mrq_.empty() && mrq_.front().cpa;
    if (sh_v || (!mrq_.empty() && !m_cpa)) {
      auto& q = sh_v ? rspq_[0] : mrq_;
      auto& out = simobject_->LsuRspOut.at(q.front().blk);
      if (!out.full()) {
        out.send(this->lsu_rsp(q.front()), 0);
        q.pop_front();
      }
    }
    if (!rspq_[1].empty() && !simobject_->GlRspOut.full()) {
      const auto& r = rspq_[1].front();
      SmSramRsp out(LANES);
      out.tag = r.tag;
      for (uint32_t i = 0; i < LANES; ++i) {
        if ((r.mask >> i) & 1)
          out.mask.set(i);
        out.data.at(i) = r.data[i];
      }
      simobject_->GlRspOut.send(out, 0);
      rspq_[1].pop_front();
    }
    if (ack_valid_) {
      auto& out = simobject_->CpaRspOut.at(ack_.blk);
      if (!out.full()) {
        LsuRsp rsp(LANES);
        rsp.tag = ack_.tag;
        rsp.cid = ack_.cid;
        rsp.uuid = ack_.uuid;
        for (uint32_t i = 0; i < LANES; ++i) {
          if ((ack_.mask >> i) & 1)
            rsp.mask.set(i);
        }
        out.send(rsp, 0);
        ack_valid_ = false;
      }
    } else if (m_cpa) {
      auto& out = simobject_->CpaRspOut.at(mrq_.front().blk);
      if (!out.full()) {
        out.send(this->lsu_rsp(mrq_.front()), 0);
        mrq_.pop_front();
      }
    }
  }

  SmSram*   simobject_;
  Config    config_;
  std::vector<uint8_t> mem_;
  SmSramCfg cfg_;
  PerfStats perf_stats_;

  std::array<uint8_t, RD_N> rd_lvl_;
  std::array<uint8_t, WR_N> wr_lvl_;
  std::array<uint8_t, 2>    wf_lvl_;
  std::array<uint8_t, NAC>  amo_lvl_;
  uint32_t sh_rr_ = 0, wf_rr_ = 0, rd_rr_ = 0, wr_rr_ = 0, cpa_rr_ = 0;
  std::array<uint32_t, PAIRS> pair_rr_{};
  std::array<bool, PAIRS>     hold_v_{};
  std::array<uint32_t, PAIRS> hold_c_{};

  bool     stg_valid_ = false;
  WfItem   cur_;
  uint64_t stg_rem_ = 0;
  std::array<uint32_t, LANES> lsu_used_{}, mbq_used_{};
  bool     cmp_have_ = false;
  std::array<uint64_t, LANES> cmp_{};
  std::array<uint64_t, LANES> acc_{};
  std::array<std::deque<Rsp>, 2> rspq_;
  std::vector<Rsp> cap_push_;

  std::deque<WfItem> mbq_;
  std::deque<Rsp> mrq_;

  bool     mrg_valid_ = false, mrg_open_ = false;
  bool     mrg_busy_sent_ = false;
  uint32_t mrg_row_ = 0, mrg_bytes_ = 0;
  std::vector<uint8_t> mrg_be_, mrg_data_;

  bool     cpa_valid_ = false;
  WfItem   cpa_;
  uint64_t cpa_rem_ = 0;
  bool     ack_valid_ = false;
  WfItem   ack_;

  std::array<CReg, BANKS> creg_{};
  uint32_t cfol_ = 0, cfol_ln_ = 0;  // the phase checks riding on the C-stage pass of lane cfol_ln_
  std::array<Wb, BANKS> wb_cur_{};
  std::array<ClientWr, BANKS> bw_prev_{};
  std::array<std::array<Rsv, SmSram::RS>, BANKS> rsv_{};
  std::array<std::array<uint64_t, LANES>, NAC> rs_acc_{};
  FinRec lsu_fin_, mbq_fin_;
  std::array<bool, 2> amo_rsp_v_{};
  std::array<MemRsp, 2> amo_rsp_req_{};
  std::array<uint32_t, 2> amo_rsp_boff_{};
};

///////////////////////////////////////////////////////////////////////////////

SmSram::SmSram(const SimContext& ctx, const char* name, const Config& config)
  : SimObject<SmSram>(ctx, name)
  , CfgIn(this)
  , LsuReqIn(config.num_blocks, this)
  , LsuRspOut(config.num_blocks, this)
  , GlReqIn(this)
  , GlRspOut(this)
  , WfRetiredOut(this)
  , RowRdIn(RD_ROWS, this)
  , RowRspOut(RD_ROWS, this)
  , WrIn(WR_ROWS, this)
  , FillDoneOut(this)
  , CpaReqIn(config.num_blocks, this)
  , CpaRspOut(config.num_blocks, this)
  , AmoIn(2, this)
  , AmoRspOut(2, this)
#ifdef VX_CFG_EXT_MBAR_ENABLE
  , MbarEvtOut(BANKS, this)
#endif
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  , CommitOut(3, this)
#endif
  , DsmemBusyOut(this)
  , impl_(new Impl(this, config))
{
  if (config.num_blocks == 0 || config.num_blocks > 32)
    throw std::runtime_error("the SM array takes 1 to 32 LSU blocks");
}

SmSramCfg SmSram::cfg_of(uint32_t s_bytes) {
  l1_slot_ref::L1SlotCfg c{};
  c.s_bytes = s_bytes;
  c.sram_bytes = VX_CFG_SM_SRAM_BYTES;
  c.reserve_bytes = VX_CFG_L1_RESERVE_BYTES;
  c.row_bytes = ROW_BYTES;
  c.protect_pct = VX_CFG_L1_REPL_PROTECT_PCT;
  c.mshr_lines = VX_CFG_SM_MAX_READ_LINES;
  return {s_bytes / ROW_BYTES, l1_slot_ref::l1_lines(c), l1_slot_ref::l1_protect(c)};
}

SmSram::~SmSram() {
  delete impl_;
}

void SmSram::on_reset() {
  impl_->reset();
}

void SmSram::on_tick() {
  impl_->tick();
}

const SmSram::PerfStats& SmSram::perf_stats() const {
  return impl_->perf_stats();
}

void SmSram::read(void* data, uint64_t addr, uint32_t size) const {
  impl_->read(data, addr, size);
}

void SmSram::write(const void* data, uint64_t addr, uint32_t size) {
  impl_->write(data, addr, size);
}

///////////////////////////////////////////////////////////////////////////////
// LocalMem with the H100 SM: the clients other than the LSU blocks and
// cp.async on the SM's one array (VX_mem_unit.sv), on LocalMem's port
// numbers. A port access is one memory block of an array row; the heads of a
// client's ports in one row and op go to the array as one row access.

class LocalMem::Impl {
public:
  static constexpr uint32_t BLK  = VX_CFG_MEM_BLOCK_SIZE;
  static constexpr uint32_t ROWB = SmSram::ROW_BYTES;
  static constexpr uint32_t RD_Q = 4;  // row reads a client may have between grant and reply
  static constexpr uint32_t DSQ  = 4;  // DSMEM replies owed, in request order
  static_assert(ROWB % BLK == 0, "an array row must hold whole memory blocks");

  Impl(LocalMem* simobject, const Config& config)
    : simobject_(simobject)
    , config_(config) {
    uint32_t n_dxa = 0;
#ifdef VX_CFG_EXT_DXA_ENABLE
    n_dxa = (VX_CFG_LMEM_PORT_BYTES > BLK) ? (VX_CFG_LMEM_PORT_BYTES / BLK) : 1;
#endif
    dxa_end_ = config.dma_begin + n_dxa;
#ifdef DXA_MBAR_ENABLE
    tcu_end_ = config.dxa_row_port;
#else
    tcu_end_ = config.num_reqs;
#endif
    if (dxa_end_ > tcu_end_ || tcu_end_ > config.num_reqs)
      throw std::invalid_argument("LocalMem: port layout");
    this->reset();
  }

  void reset() {
    for (auto& g : rd_grp_)
      g.clear();
    dxa_wr_.clear();
    dxa_wr_peer_ = false;
    dxa_free_ = UINT32_MAX;
    finals_ = 0;
    finals_done_ = 0;
    ds_q_.clear();
    ds_busy_ = false;
    perf_stats_ = PerfStats();
  }

  void tick() {
    auto& ap = simobject_->ArrayPorts;
    finals_ -= finals_done_;
    finals_done_ = 0;

    for (uint32_t c = 0; c < 3; ++c) {
      auto& in = ap.RowRspIn.at(c);
      while (!in.empty()) {
        this->row_reply(c, in.peek());
        in.pop();
      }
    }
    for (uint32_t a = 0; a < 2; ++a) {
      auto& in = ap.AmoRspIn.at(a);
      while (!in.empty()) {
        this->amo_reply(a, in.peek());
        in.pop();
      }
    }
    while (!ap.DsmemBusyIn.empty()) {
      ds_busy_ = ap.DsmemBusyIn.peek();
      ap.DsmemBusyIn.pop();
    }

    // The array took the row write of the last cycle: a peer's flit leaves
    // its port, a local copy's writes reach the completion unit.
    if (!dxa_wr_.empty() && ap.WrOut.at(0).size() == 0) {
#ifdef DXA_MBAR_ENABLE
      if (dxa_wr_peer_) {
        simobject_->Inputs.at(config_.dxa_row_port).pop();
      } else
#endif
      {
        this->dxa_commit();
      }
      dxa_wr_.clear();
      dxa_wr_peer_ = false;
    }

    this->issue_rows(0, dxa_end_, tcu_end_);
    this->issue_dxa();
    this->issue_dsmem();
    this->issue_dxa_compl();
    this->send_dsmem_reply();
  }

  void dxa_free(uint32_t free) {
    dxa_free_ = free;
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:
  struct Member {
    uint32_t port;
    uint32_t boff;
    MemReq   req;
  };
  using Group = std::vector<Member>;

  enum class DsKind : uint8_t { READ, WRITE, AMO };
  struct DsEntry {
    DsKind kind;
    bool   ready;
    MemRsp rsp;
    uint32_t boff;
  };

  static uint64_t win(uint64_t addr) { return addr & WIN_MASK; }
  static uint32_t row_of(uint64_t addr) { return uint32_t(win(addr) / ROWB); }
  static uint32_t boff_of(uint64_t addr) { return uint32_t(win(addr) % ROWB) & ~(BLK - 1); }

  // The heads of ports [begin, end) in the lowest busy port's row and op.
  Group gather(uint32_t begin, uint32_t end, bool* is_write) const {
    Group g;
    for (uint32_t p = begin; p < end; ++p) {
      const auto& ch = simobject_->Inputs.at(p);
      if (ch.empty())
        continue;
      const auto& rq = ch.peek();
      if (!g.empty()) {
        const auto& h = g.front().req;
        if (row_of(rq.addr) != row_of(h.addr) || rq.is_write() != h.is_write())
          continue;
#ifdef VX_CFG_EXT_DXA_ENABLE
        if (rq.is_write() && rq.dxa.beat != h.dxa.beat)
          continue;
#endif
      }
      g.push_back({p, boff_of(rq.addr), rq});
    }
    if (!g.empty())
      *is_write = g.front().req.is_write();
    return g;
  }

  bool issue_read(uint32_t c, Group& g) {
    auto& ap = simobject_->ArrayPorts;
    if (ap.RowRdOut.at(c).size() != 0 || rd_grp_.at(c).size() >= RD_Q)
      return false;
    SmSramRowRd rd;
    rd.row = row_of(g.front().req.addr);
    ap.RowRdOut.at(c).send(rd, 0);
    for (auto& m : g)
      simobject_->Inputs.at(m.port).pop();
    rd_grp_.at(c).push_back(std::move(g));
    return true;
  }

  void issue_rows(uint32_t c, uint32_t begin, uint32_t end) {
    if (begin == end)
      return;
    bool wr = false;
    auto g = this->gather(begin, end, &wr);
    if (g.empty())
      return;
    if (wr)
      throw std::runtime_error("LocalMem: tensor-unit write to shared memory");
    this->issue_read(c, g);
  }

  static SmSramFill fill_of(const Group& g, SmSramSrc src) {
    SmSramFill f;
    f.row = row_of(g.front().req.addr);
    f.src = src;
    for (const auto& m : g) {
      for (uint32_t k = 0; k < BLK; ++k) {
        if ((m.req.byteen >> k) & 1) {
          f.byteen[m.boff + k] = 1;
          f.data[m.boff + k] = (*m.req.data)[k];
        }
      }
    }
    return f;
  }

  void issue_dxa() {
#ifdef VX_CFG_EXT_DXA_ENABLE
    auto& ap = simobject_->ArrayPorts;
    const bool wr_free = dxa_wr_.empty() && ap.WrOut.at(0).size() == 0;
#ifdef DXA_MBAR_ENABLE
    // A peer's copy rows go first; the ingress sends its next flit once this
    // one is in the array.
    {
      const auto& ch = simobject_->Inputs.at(config_.dxa_row_port);
      if (!ch.empty()) {
        if (wr_free) {
          Group g{{config_.dxa_row_port, boff_of(ch.peek().addr), ch.peek()}};
          ap.WrOut.at(0).send(fill_of(g, SmSramSrc::TMA), 0);
          dxa_wr_ = std::move(g);
          dxa_wr_peer_ = true;
        }
        return;
      }
    }
#endif
    bool wr = false;
    auto g = this->gather(config_.dma_begin, dxa_end_, &wr);
    if (g.empty())
      return;
    if (!wr) {
      this->issue_read(1, g);
      return;
    }
    if (!wr_free)
      return;
    // A copy's final writes wait while the completion queue has no entry.
    bool ends = false;
    for (const auto& m : g)
      ends |= m.req.flags.dxa_notify_done;
    if (ends && dxa_free_ != UINT32_MAX && int64_t(dxa_free_) - int64_t(finals_) <= 0)
      return;
    ap.WrOut.at(0).send(fill_of(g, SmSramSrc::TMA), 0);
    for (auto& m : g)
      simobject_->Inputs.at(m.port).pop();
    finals_ += ends;
    dxa_wr_ = std::move(g);
#endif
  }

  void dxa_commit() {
#ifdef VX_CFG_EXT_DXA_ENABLE
    for (const auto& m : dxa_wr_) {
      DxaWrCommit c;
      c.tag    = m.req.dxa;
      c.notify = m.req.flags.dxa_notify_done;
      c.bar_id = m.req.flags.dxa_notify_bar_id;
      c.uuid   = m.req.uuid;
      simobject_->DxaWrOut.send(c, 0);
      finals_done_ += c.notify;
    }
#endif
  }

  void issue_dxa_compl() {
#ifdef DXA_MBAR_ENABLE
    auto& ap = simobject_->ArrayPorts;
    auto& ch = simobject_->Inputs.at(config_.dxa_compl_port);
    if (!ch.empty() && ap.AmoOut.at(0).size() == 0) {
      ap.AmoOut.at(0).send(ch.peek(), 0);
      ch.pop();
    }
#endif
  }

  void issue_dsmem() {
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    auto& ap = simobject_->ArrayPorts;
    auto& ch = simobject_->Inputs.at(config_.commit_ports);
    if (ch.empty() || ds_q_.size() >= DSQ)
      return;
    const auto& rq = ch.peek();
    bool amo = memop_is_atomic(rq.op);
#ifdef VX_CFG_EXT_MBAR_ENABLE
    amo |= memop_is_mbar(rq.op);
#endif
    // Reads and atomics wait until the slots written before them are in the
    // array; a write or an atomic waits for the earlier replies.
    const bool wr_busy = ds_busy_ || ap.WrOut.at(1).size() != 0;
    bool rsp_busy = false;
    for (const auto& e : ds_q_)
      rsp_busy |= (e.kind != DsKind::WRITE) && !e.ready;
    MemRsp rsp(rq.tag, rq.hart_id, rq.uuid);
    if (amo) {
      if (wr_busy || rsp_busy || ap.AmoOut.at(1).size() != 0)
        return;
      ap.AmoOut.at(1).send(rq, 0);
      ds_q_.push_back({DsKind::AMO, false, rsp, 0});
      ch.pop();
    } else if (rq.is_write()) {
      if (rsp_busy || ap.WrOut.at(1).size() != 0)
        return;
      Group g{{config_.commit_ports, boff_of(rq.addr), rq}};
      ap.WrOut.at(1).send(fill_of(g, SmSramSrc::DSMEM), 0);
      if (rq.flags.strsp)
        ds_q_.push_back({DsKind::WRITE, true, rsp, 0});
      ch.pop();
    } else {
      if (wr_busy)
        return;
      Group g{{config_.commit_ports, boff_of(rq.addr), rq}};
      if (this->issue_read(2, g))
        ds_q_.push_back({DsKind::READ, false, rsp, boff_of(rq.addr)});
    }
#endif
  }

  void send_dsmem_reply() {
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    if (ds_q_.empty() || !ds_q_.front().ready)
      return;
    simobject_->Outputs.at(config_.commit_ports).send(ds_q_.front().rsp, 1);
    ds_q_.pop_front();
#endif
  }

  void row_reply(uint32_t c, const SmSramRowRsp& r) {
    auto& q = rd_grp_.at(c);
    if (q.empty())
      throw std::runtime_error("LocalMem: row reply without a request");
    for (const auto& m : q.front()) {
      MemRsp rsp(m.req.tag, m.req.hart_id, m.req.uuid);
      rsp.data = make_mem_block();
      std::memcpy(rsp.data->data(), &r.data[m.boff], BLK);
      if (c == 2) {
        for (auto& e : ds_q_) {
          if (e.kind == DsKind::READ && !e.ready) {
            e.rsp.data = rsp.data;
            e.ready = true;
            break;
          }
        }
      } else {
        simobject_->Outputs.at(m.port).send(rsp, 1);
      }
    }
    q.pop_front();
  }

  void amo_reply(uint32_t a, const MemRsp& r) {
    if (a == 0) {
#ifdef DXA_MBAR_ENABLE
      simobject_->Outputs.at(config_.dxa_compl_port).send(r, 0);
#endif
      return;
    }
    for (auto& e : ds_q_) {
      if (e.kind == DsKind::AMO && !e.ready) {
        e.rsp.data = r.data;
        e.ready = true;
        return;
      }
    }
    throw std::runtime_error("LocalMem: DSMEM atomic reply without a request");
  }

  LocalMem* simobject_;
  Config    config_;
  uint32_t  dxa_end_ = 0;
  uint32_t  tcu_end_ = 0;
  std::array<std::deque<Group>, 3> rd_grp_;
  Group     dxa_wr_;          // the row write in the array's input
  bool      dxa_wr_peer_ = false;
  uint32_t  dxa_free_ = UINT32_MAX;
  uint32_t  finals_ = 0;      // final writes sent whose commits the free count may not show yet
  uint32_t  finals_done_ = 0;
  std::deque<DsEntry> ds_q_;
  bool      ds_busy_ = false;
  PerfStats perf_stats_;
};

LocalMem::LocalMem(const SimContext& ctx, const char* name, const Config& config)
  : SimObject<LocalMem>(ctx, name)
  , Inputs(config.num_reqs, this)
  , Outputs(config.num_reqs, this)
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  , CommitOut(1 << config.B, this)
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
  , MbarEvtOut(1 << config.B, this)
#endif
#ifdef VX_CFG_EXT_DXA_ENABLE
  , DxaWrOut(this, 2u << config.B)
  , DxaFreeIn(this)
#endif
  , impl_(new Impl(this, config))
{
#ifdef VX_CFG_EXT_DXA_ENABLE
  DxaFreeIn.bind(this, &LocalMem::on_dxa_free);
#endif
}

LocalMem::~LocalMem() {
  delete impl_;
}

void LocalMem::on_reset() {
  impl_->reset();
}

void LocalMem::on_tick() {
  impl_->tick();
}

const LocalMem::PerfStats& LocalMem::perf_stats() const {
  return impl_->perf_stats();
}

#ifdef VX_CFG_EXT_DXA_ENABLE
void LocalMem::on_dxa_free(const uint32_t& free) {
  impl_->dxa_free(free);
}
#endif

#endif
