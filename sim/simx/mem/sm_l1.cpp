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

#include "sm_l1.h"
#include "sm_port.h"
#include "mem_block_pool.h"
#include <l1_slot_ref.h>
#include <array>
#include <bitset>
#include <cassert>
#include <cstring>
#include <deque>
#include <map>
#include <sstream>
#include <stdexcept>

using namespace vortex;

static_assert(L1_PIPE_OVERHEAD == l1_slot_ref::l1_pipe_stages(VX_CFG_NUM_OPCS),
              "l1_pipe_stages() must count the stages the model builds");
static_assert(L1_MISS_PIPE_OVERHEAD == l1_slot_ref::l1_miss_pipe_stages(VX_CFG_NUM_OPCS),
              "l1_miss_pipe_stages() must count the stages the model builds");

namespace {

constexpr uint32_t clog2(uint64_t x) {
  uint32_t n = 0;
  while ((uint64_t(1) << n) < x)
    ++n;
  return n;
}

constexpr uint32_t LANES      = LSU_CHANNELS;
constexpr uint32_t WORD       = LSU_WORD_SIZE;
constexpr uint32_t LINE       = VX_CFG_DCACHE_LINE_SIZE;
constexpr uint32_t LWORDS     = LINE / WORD;
constexpr uint32_t BLOCK      = VX_CFG_MEM_BLOCK_SIZE;
constexpr uint32_t LBLOCKS    = (LINE + BLOCK - 1) / BLOCK;
constexpr uint32_t TB         = VX_CFG_L1_TAG_BANKS;
constexpr uint32_t SETS       = VX_CFG_L1_TAG_SETS;
constexpr uint32_t SPB        = SETS / TB;
constexpr uint32_t WAYS       = VX_CFG_L1_TAG_WAYS;
constexpr uint32_t VBYTES     = VX_CFG_L1_VALID_BYTES;
constexpr uint32_t VB         = LINE / VBYTES;
constexpr uint32_t NSEC       = LINE / VX_CFG_DCACHE_SECTOR_SIZE;
constexpr uint32_t SVB        = VX_CFG_DCACHE_SECTOR_SIZE / VBYTES;
constexpr uint32_t MSHR       = VX_CFG_SM_MAX_READ_LINES;
constexpr uint32_t RQ_DEPTH   = VX_CFG_LSU_PENDING_SIZE * VX_CFG_NUM_LSU_BLOCKS;
constexpr uint32_t EG_DEPTH   = 4;
constexpr uint32_t ALQ        = 2;
constexpr uint32_t ROWS       = VX_CFG_SM_SRAM_BYTES / VX_CFG_SM_DATAPATH_BYTES;
constexpr uint32_t MAX_LINES  = VX_CFG_L1_MAX_LINES;
constexpr uint32_t PROTECT_MAX = (VX_CFG_L1_REPL_PROTECT_PCT * VX_CFG_L1_MAX_LINES / 100 > VX_CFG_SM_MAX_READ_LINES)
                               ? (VX_CFG_L1_REPL_PROTECT_PCT * VX_CFG_L1_MAX_LINES / 100) : VX_CFG_SM_MAX_READ_LINES;
constexpr uint32_t YOUNG_DEPTH = PROTECT_MAX + 1;
constexpr uint32_t FREE_DEPTH = VX_CFG_SM_MAX_READ_LINES;
constexpr uint32_t OLD_DEPTH  = 1u << clog2(VX_CFG_L1_MAX_LINES);
constexpr uint32_t UPD_SRCS   = 3;
constexpr uint32_t UPD_DEPTH  = 2 * UPD_SRCS;
constexpr uint32_t SEQ_W      = clog2(2 * (2 * VX_CFG_LSU_PENDING_SIZE) * VX_CFG_NUM_LSU_BLOCKS);  // VX_gpu_pkg L1_SEQ_W: both LSU credit classes
constexpr uint32_t SEQ_MASK   = (1u << SEQ_W) - 1;
constexpr uint32_t LFSR_SEED  = 0xACE1;
constexpr uint32_t LFSR_TAPS  = 0xB400;

static_assert(LANES <= 64 && VB <= 64 && TB <= 32 && LBLOCKS <= 16, "mask widths");

using MMask = std::bitset<MSHR>;
static_assert(LINE == VX_CFG_SM_DATAPATH_BYTES, "an L1 line must be one array row");
static_assert(SETS % TB == 0, "tag sets must spread evenly over the tag banks");

enum Kind : uint8_t { K_LOAD, K_STORE, K_AMO, K_FENCE };
enum Cls : uint8_t { C_NONE, C_COMMIT, C_REPLAY };
enum Res : uint8_t { R_WAIT, R_HIT, R_JOIN, R_NONE, R_ACT };
enum Phase : uint8_t { P_LINES, P_EMIT, P_INV, P_PUSH };
enum PoolOp : uint8_t { OP_BORROW, OP_YOUNG, OP_DRAW };
enum PoolKind : uint8_t { KIND_FREE = 1, KIND_BUMP = 2, KIND_AGE = 4, KIND_BORROW = 5 };
enum PoolWait : uint8_t { WAIT_NONE, WAIT_YOUNG_FULL, WAIT_OLD_EMPTY, WAIT_VICTIM_IN_FLIGHT, WAIT_DRAW_PENDING };
enum UpdSrc : uint32_t { U_STORE = 0, U_VICTIM = 1, U_FILL = 2 };

inline uint64_t bit(uint32_t i) { return uint64_t(1) << i; }

inline uint32_t seq_add(uint32_t a, uint32_t b) { return (a + b) & SEQ_MASK; }

inline uint32_t set_of(uint64_t line) {
  return l1_slot_ref::l1_set(line, SETS, VX_CFG_L1_INDEX_HASH);
}
inline uint32_t bank_of(uint32_t set) { return l1_slot_ref::l1_tag_bank(set, TB); }
inline uint32_t trow_of(uint32_t set) { return l1_slot_ref::l1_tag_row(set, TB); }

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

// One warp request at the input register (l1_req_t).
struct Req {
  Kind kind = K_LOAD;
  bool nc = false;
  uint32_t blk = 0;
  uint32_t tag = 0;
  uint32_t seq = 0;
  uint64_t mask = 0;
  uint64_t lead = 0;
  bool host = false;
  std::array<uint32_t, LANES> lead_idx{};
  std::array<uint64_t, LANES> waddr{};
  std::array<uint64_t, LANES> data{};
  std::array<uint32_t, LANES> byteen{};
  std::shared_ptr<LsuReq> src;

  uint64_t line(uint32_t i) const { return waddr[i] / LWORDS; }
  uint32_t wi(uint32_t i) const { return uint32_t(waddr[i] % LWORDS); }
};

struct RespCtx {
  uint32_t cid = 0;
  uint64_t uuid = 0;
  uint64_t left = 0;
  std::array<uint32_t, LANES> boff{};
};

struct GlItem {
  SmSramReq req;
  MMask wait;
  GlItem() : req(LANES) {}
};

enum EgKind : uint8_t { EG_READ, EG_WRITE, EG_LANE };

struct EgItem {
  EgKind kind = EG_READ;
  uint64_t line = 0;
  std::array<uint8_t, LINE> byteen{};
  std::array<uint8_t, LINE> data{};
  uint32_t idx = 0;
  uint32_t lane = 0;
  std::shared_ptr<LsuReq> src;
};

struct Upd {
  uint32_t row = 0;
  uint32_t way = 0;
  uint64_t mask = 0;
  bool set = false;
  bool inval = false;
  uint32_t src = 0;
  uint32_t id = 0;
};

struct TagWord {
  bool valid = false;
  uint64_t line = 0;
  uint32_t slot = 0;
};

struct SlotRef {
  uint32_t slot = 0;
  uint32_t set = 0;
  uint32_t way = 0;
};

struct MshrEnt {
  uint64_t line = 0;
  uint32_t sectors = 0;
  uint32_t slot = 0;
  uint32_t set = 0;
  uint32_t way = 0;
  uint64_t vmiss = 0;
  bool cacheable = true;
  bool ilk_valid = false;
};

// Where a reused slot's fill blockers (the global items older than its
// request) are: FIN's commit item, the array's item in flight. The request in
// the tag stage never reuses a slot its own hits read, so a fill never waits
// on the tag stage, which may itself wait on fills.
enum : uint8_t { ILK_FIN = 1, ILK_ARR = 2 };

} // namespace

class SmL1::Impl {
public:
  Impl(SmL1* simobject, const Config& config)
    : simobject_(simobject)
    , config_(config)
    , tag_mem_(size_t(TB) * SPB * WAYS)
    , vm_mem_(size_t(TB) * SPB * WAYS, 0)
    , ctx_(config.num_blocks)
    , open_(MAX_LINES, false)
    , young_mem_(YOUNG_DEPTH)
    , free_mem_(FREE_DEPTH, 0)
    , old_mem_(OLD_DEPTH)
  {
    if (config_.num_blocks == 0 || config_.num_blocks > 32)
      throw std::runtime_error("the SM L1 takes 1 to 32 LSU blocks");
    if (UPD_DEPTH < 2)
      throw std::runtime_error("the tag update queues need room for a fill beside an update");
    if (uint64_t(SETS) * WAYS < MAX_LINES)
      throw std::runtime_error("the L1 tags must cover every L1 line");
    if (uint64_t(SETS) * WAYS < MSHR)
      throw std::runtime_error("L1_TAGS_COVER: the tags must hold the lines in flight");
    if (MAX_LINES >= 65536)
      throw std::runtime_error("the victim draw takes a 16-bit LFSR");
    this->reset();
  }

  void reset() {
    cfg_ = SmSram::cfg_of(VX_CFG_SMEM_PER_SM_MAX);
    rr_last_ = config_.num_blocks - 1;
    in_valid_ = false;
    in_todo_ = 0;
    in_cta_ = false;
    in_inv_ = false;
    t2_inv_ = false;
    hf_req_ = false;
    hf_sent_ = false;
    hf_done_ = false;
    seq_ctr_ = 0;
    t2_valid_ = false;
    t2_first_ = false;
    t2_last_ = false;
    t2_bv_ = 0;
    t2_bl_.fill(0);
    t2_res_ = 0;
    t2_stale_ = 0;
    t2_phase_ = P_LINES;
    t2_emit_ = 0;
    inv_wait_ = false;
    acc_cls_.fill(C_NONE);
    acc_slot_.fill(0);
    acc_wait_.reset();
    fin_c_valid_ = false;
    fin_r_valid_ = false;
    fin_wait_.reset();
    a_valid_ = false;
    a_rep_ = false;
    a_seq_ = 0;
    rq_resv_ = 0;
    rq_.clear();
    alq_n_ = 0;
    fn_valid_ = false;
    fn_host_ = false;
    fn_got_ = false;
    fn_mask_ = 0;
    amo_out_ = 0;
    fl_valid_ = false;
    fl_id_ = 0;
    fl_data_.fill(0);
    done_.reset();
    draw_on_ = false;
    rs_valid_ = false;
    eg_.clear();
    gl_off_ = false;
    fill_off_ = false;
    for (auto& c : ctx_)
      c.clear();
    for (auto& a : asm_)
      a = Asm{};
    // tags: VX_sm_l1_tags clears every row after reset before the first lookup
    inval_r_ = true;
    inval_row_ = 0;
    for (auto& q : tq_)
      q.clear();
    for (auto& t : tag_mem_)
      t = TagWord{};
    std::fill(vm_mem_.begin(), vm_mem_.end(), 0);
    for (uint32_t b = 0; b < TB; ++b) {
      for (uint32_t w = 0; w < WAYS; ++w) {
        t_rd_[b][w] = TagWord{};
        v_rd_[b][w] = 0;
        byp_s_[b][w] = byp_c_[b][w] = 0;
        byp_inv_[b][w] = false;
      }
    }
    // slots
    lfsr_ = LFSR_SEED;
    this->slots_clear();
    // mshr
    for (auto& e : mshr_)
      e = MshrEnt{};
    m_valid_.reset();
    m_filled_.reset();
    m_store_.reset();
    m_rpc_.fill(0);
    m_ilk_.fill(0);
    a_wait_.reset();
    perf_stats_ = PerfStats();
  }

  const PerfStats& perf_stats() const { return perf_stats_; }

  // the host flush's own work is not busy (VX_sm_l1 busy)
  bool busy() const {
    return (in_valid_ || t2_valid_ || fin_c_valid_ || fin_r_valid_ || !rq_.empty() || a_valid_
        || m_valid_.any() || fl_valid_ || !eg_.empty() || alq_n_ != 0 || fn_valid_ || amo_out_ != 0 || rs_valid_)
        && !(hf_sent_ && !hf_done_);
  }

  void flush_begin() {
    hf_req_ = true;
    hf_sent_ = false;
    hf_done_ = false;
  }

  bool flush_done() const { return hf_done_; }

  void tick() {
    auto* so = simobject_;

    if (!so->CfgIn.empty()) {
      cfg_ = so->CfgIn.peek();
      so->CfgIn.pop();
    }

    // The array's outputs of this cycle.
    const bool gl_fire = gl_off_ && so->GlReqOut.size() == 0;
    const bool take_rep = gl_off_rep_;
    bool wf_valid = false;
    if (!so->WfRetiredIn.empty()) {
      wf_valid = true;
      // a one-wavefront item can retire in the cycle it is taken
      if (gl_fire ? so->WfRetiredIn.peek() != gl_off_seq_
                  : (!a_valid_ || so->WfRetiredIn.peek() != a_seq_))
        throw std::runtime_error(std::string(so->name()) + ": global items retired out of order");
      so->WfRetiredIn.pop();
    }
    bool fill_fire = false;
    if (fill_off_ && !so->FillDoneIn.empty()) {
      fill_fire = true;
      so->FillDoneIn.pop();
    }

    ///////////////////////////////////////////////////////////////////////////
    // Tag macro outputs (registered at the previous cycle's lookups).

    std::array<std::array<bool, WAYS>, TB> rd_valid;
    std::array<std::array<uint64_t, WAYS>, TB> rd_vmask;
    for (uint32_t b = 0; b < TB; ++b) {
      for (uint32_t w = 0; w < WAYS; ++w) {
        rd_valid[b][w] = t_rd_[b][w].valid && !byp_inv_[b][w];
        rd_vmask[b][w] = (v_rd_[b][w] & ~byp_c_[b][w]) | byp_s_[b][w];
      }
    }

    // T2: the batch's lines.
    std::array<uint64_t, TB> t2_line{};
    for (uint32_t k = 0; k < TB; ++k)
      t2_line[k] = t2_r_.line(t2_bl_[k]);

    // MSHR lookups (registered entries).
    std::array<bool, TB> m_hit{}, m_store{};
    std::array<uint32_t, TB> m_id{}, m_secs{}, m_slot{};
    for (uint32_t k = 0; k < TB; ++k) {
      for (uint32_t i = 0; i < MSHR; ++i) {
        if (m_valid_.test(i) && mshr_[i].line == t2_line[k] && !m_hit[k]) {
          m_hit[k] = true;
          m_id[k] = i;
        }
      }
      m_secs[k] = mshr_[m_id[k]].sectors;
      m_store[k] = m_store_.test(m_id[k]);
      m_slot[k] = mshr_[m_id[k]].slot;
    }

    // Bytes, words and sectors the batch line's lanes touch.
    std::array<std::array<uint8_t, LINE>, TB> k_bm{};
    std::array<std::array<uint8_t, LINE>, TB> k_wdata{};
    std::array<uint64_t, TB> k_need{}, k_full{};
    std::array<uint32_t, TB> k_secs{};
    for (uint32_t k = 0; k < TB; ++k) {
      if (!t2_valid_)
        break;
      for (uint32_t j = 0; j < LANES; ++j) {
        if (((t2_r_.mask >> j) & 1) && t2_r_.lead_idx[j] == t2_bl_[k]) {
          for (uint32_t b = 0; b < WORD; ++b) {
            if ((t2_r_.byteen[j] >> b) & 1) {
              const uint32_t o = t2_r_.wi(j) * WORD + b;
              k_bm[k][o] = 1;
              k_wdata[k][o] = uint8_t(t2_r_.data[j] >> (8 * b));
            }
          }
        }
      }
      for (uint32_t v = 0; v < VB; ++v) {
        bool any = false, all = true;
        for (uint32_t b = 0; b < VBYTES; ++b) {
          any |= k_bm[k][v * VBYTES + b] != 0;
          all &= k_bm[k][v * VBYTES + b] != 0;
        }
        if (any) k_need[k] |= bit(v);
        if (all) k_full[k] |= bit(v);
      }
      for (uint32_t s = 0; s < NSEC; ++s) {
        if ((k_need[k] >> (s * SVB)) & ((SVB >= 64) ? ~uint64_t(0) : (bit(SVB) - 1)))
          k_secs[k] |= 1u << s;
      }
    }

    // A lookup that may join its line's open entry keeps the entry from
    // retiring in this cycle; one that waits for the entry to retire (other
    // sectors, an atomic) must not.
    MMask looked;
    for (uint32_t k = 0; k < TB; ++k) {
      const bool hold = t2_valid_ && ((t2_bv_ >> k) & 1) && !((t2_res_ >> k) & 1)
        && ((t2_r_.kind == K_LOAD && (k_secs[k] & ~m_secs[k]) == 0) || t2_r_.kind == K_STORE);
      for (uint32_t i = 0; i < MSHR; ++i) {
        if (hold && m_valid_.test(i) && mshr_[i].line == t2_line[k])
          looked.set(i);
      }
    }

    // Tag compare.
    std::array<bool, TB> th{}, tinv{};
    std::array<uint32_t, TB> tway{}, tslot{}, tinv_way{};
    std::array<uint64_t, TB> tvm{};
    for (uint32_t k = 0; k < TB; ++k) {
      for (int w = int(WAYS) - 1; w >= 0; --w) {
        if (rd_valid[k][w] && t_rd_[k][w].line == t2_line[k]) {
          th[k] = true; tway[k] = w; tslot[k] = t_rd_[k][w].slot; tvm[k] = rd_vmask[k][w];
        }
        if (!rd_valid[k][w]) {
          tinv[k] = true; tinv_way[k] = w;
        }
      }
    }

    std::array<Res, TB> k_cls;
    for (uint32_t k = 0; k < TB; ++k) {
      k_cls[k] = R_WAIT;
      if (!(t2_valid_ && ((t2_bv_ >> k) & 1) && !((t2_res_ >> k) & 1) && !((t2_stale_ >> k) & 1)))
        continue;
      const bool words_ok = (tvm[k] & k_need[k]) == k_need[k];
      switch (t2_r_.kind) {
      case K_LOAD:
        if (m_hit[k]) {
          if (!m_store[k] && th[k] && words_ok)
            k_cls[k] = R_HIT;
          else if ((k_secs[k] & ~m_secs[k]) == 0)
            k_cls[k] = R_JOIN;
        } else if (th[k] && words_ok) {
          k_cls[k] = R_HIT;
        } else {
          k_cls[k] = R_ACT;
        }
        break;
      case K_STORE:
        k_cls[k] = R_ACT;
        break;
      case K_AMO:
        if (!m_hit[k])
          k_cls[k] = th[k] ? R_ACT : R_NONE;
        break;
      default:
        break;
      }
    }

    bool act_any = false;
    uint32_t act_k = 0;
    for (int k = int(TB) - 1; k >= 0; --k) {
      if (k_cls[k] == R_ACT) {
        act_any = true;
        act_k = k;
      }
    }
    const uint64_t a_line = t2_line[act_k];
    const uint32_t a_set = set_of(a_line);
    const uint32_t a_row = trow_of(a_set);
    const uint32_t a_lane = t2_bl_[act_k];

    // Slots the load in the tag stage reads at its commit: its earlier hits
    // and this cycle's.
    auto pinned = [&](uint32_t slot) {
      if (!t2_valid_ || t2_r_.kind != K_LOAD)
        return false;
      for (uint32_t k = 0; k < TB; ++k) {
        if (k_cls[k] == R_HIT && tslot[k] == slot)
          return true;
      }
      if (t2_first_)
        return false;
      for (uint32_t j = 0; j < LANES; ++j) {
        if (acc_cls_[j] == C_COMMIT && acc_slot_[j] == slot)
          return true;
      }
      return false;
    };

    // In-set candidates: ways whose slot is neither in flight nor pinned.
    uint64_t cand = 0;
    uint32_t ncand = 0;
    for (uint32_t w = 0; w < WAYS; ++w) {
      if (!open_[t_rd_[act_k][w].slot] && !pinned(t_rd_[act_k][w].slot)) {
        cand |= bit(w);
        ++ncand;
      }
    }
    const uint32_t lfsr_next = this->lfsr_n();
    uint32_t cand_way = 0;
    {
      const uint32_t pick = (ncand != 0) ? (lfsr_next % ncand) : 0;
      uint32_t n = 0;
      for (uint32_t w = 0; w < WAYS; ++w) {
        if ((cand >> w) & 1) {
          if (n == pick) cand_way = w;
          ++n;
        }
      }
    }

    const bool fin_c_take = gl_fire && !take_rep;
    const bool fin_free = !fin_c_valid_ || fin_c_take;
    const bool alq_free = alq_n_ < ALQ;
    const bool eg_full = eg_.size() >= EG_DEPTH;
    uint32_t mal_id = 0;
    bool mal_ready = false;
    for (uint32_t i = 0; i < MSHR; ++i) {
      if (!m_valid_.test(i)) {
        mal_id = i;
        mal_ready = true;
        break;
      }
    }
    const SlotRef vref = this->victim_ref();
    const bool v_room = this->q_cnt(bank_of(vref.set)) < UPD_DEPTH;
    // a victim clear would land before a queued allocation write of the same
    // way, which would then revive the cleared line on the slot it gives away
    bool v_alq = false;
    for (uint32_t q = 0; q < alq_n_; ++q)
      v_alq = v_alq || alq_[q].slot == vref.slot;

    const bool act_mem = act_any && fin_free;
    const bool ld_alloc = t2_r_.kind == K_LOAD && !th[act_k] && mal_ready && !eg_full;
    const bool st_alloc = t2_r_.kind == K_STORE && !t2_r_.nc && !m_hit[act_k] && !th[act_k] && !eg_full;
    const bool free_avail = this->free_avail();
    const bool use_borrow = ld_alloc && t2_r_.nc && free_avail && !draw_on_;
    const bool use_pool = act_mem && (use_borrow || ((ld_alloc || st_alloc) && alq_free && tinv[act_k]));
    PoolOp pool_op = use_borrow ? OP_BORROW : ((free_avail && !draw_on_) ? OP_YOUNG : OP_DRAW);
    const bool pool_valid = use_pool && (pool_op != OP_DRAW || !draw_on_ || v_room);
    Pool pool = this->pool_eval(pool_valid, pool_op, a_set, tinv_way[act_k], v_alq || pinned(vref.slot));
    const bool pool_draw = pool_valid && pool_op == OP_DRAW;
    const bool draw_start = pool_draw && !pool.ok
      && (pool.wait == WAIT_DRAW_PENDING || pool.wait == WAIT_VICTIM_IN_FLIGHT);
    const bool draw_done = pool_draw && pool.ok;

    if (act_mem && t2_r_.kind == K_LOAD && !mal_ready)
      ++perf_stats_.mshr_stalls;

    // The one allocating action.
    bool act_fire = false;
    Res act_res = R_NONE;
    uint32_t act_slot = 0, act_wid = 0;
    bool act_al = false;
    uint32_t act_al_way = 0, act_al_slot = 0;
    uint64_t act_al_vm = 0;
    bool lfsr_step = false;
    bool mal_valid = false;
    MshrEnt mal;
    mal.line = a_line;
    mal.sectors = k_secs[act_k];
    mal.set = a_set;
    bool open_valid = false;
    uint32_t open_slot = 0;
    bool u0_valid = false, u0_set = true;
    uint64_t u0_mask = k_full[act_k];
    bool aeg_push = false;
    EgItem aeg;
    if (act_mem) {
      switch (t2_r_.kind) {
      case K_LOAD:
        if (mal_ready && !eg_full) {
          if (th[act_k]) {
            act_fire = true;
            mal.slot = tslot[act_k];
            mal.way = tway[act_k];
            mal.vmiss = tvm[act_k];
            // a store-allocate or an atomic may have left older readers of
            // other data on the slot, with no fill of its own to wait behind
            mal.ilk_valid = true;
          } else if (use_borrow) {
            act_fire = true;
            mal.slot = pool.slot;
            mal.cacheable = false;
          } else if (alq_free) {
            if (tinv[act_k]) {
              if (pool_valid && pool.ok) {
                act_fire = true;
                mal.slot = pool.slot;
                mal.way = tinv_way[act_k];
                mal.ilk_valid = (pool.kind == KIND_AGE);
                act_al = true;
              }
            } else if (ncand != 0) {
              act_fire = true;
              lfsr_step = true;
              mal.slot = t_rd_[act_k][cand_way].slot;
              mal.way = cand_way;
              mal.ilk_valid = true;
              act_al = true;
            }
          }
          if (act_fire) {
            mal_valid = true;
            open_valid = true;
            open_slot = mal.slot;
            aeg_push = true;
            aeg.kind = EG_READ;
            for (uint32_t b = 0; b < LINE; ++b)
              aeg.byteen[b] = (k_secs[act_k] >> (b / (LINE / NSEC))) & 1;
            aeg.idx = mal_id;
            act_res = R_JOIN;
            act_slot = mal.slot;
            act_wid = mal_id;
            act_al_way = mal.way;
            act_al_slot = mal.slot;
          }
        }
        break;
      case K_STORE:
        aeg.kind = EG_WRITE;
        aeg.byteen = k_bm[act_k];
        aeg.data = k_wdata[act_k];
        if (!eg_full) {
          if (t2_r_.nc) {
            act_fire = true;
            act_res = R_NONE;
          } else if (m_hit[act_k] || th[act_k]) {
            u0_valid = th[act_k] && k_full[act_k] != 0;
            if (!u0_valid || this->q_cnt(act_k) < UPD_DEPTH) {
              act_fire = true;
              act_res = m_hit[act_k] ? R_JOIN : R_HIT;
              act_slot = th[act_k] ? tslot[act_k] : m_slot[act_k];
              act_wid = m_id[act_k];
            } else {
              u0_valid = false;
            }
          } else if (alq_free) {
            if (tinv[act_k]) {
              if (pool_valid && pool.ok) {
                act_fire = true;
                act_al_way = tinv_way[act_k];
                act_al_slot = pool.slot;
              }
            } else if (ncand != 0) {
              act_fire = true;
              lfsr_step = true;
              act_al_way = cand_way;
              act_al_slot = t_rd_[act_k][cand_way].slot;
            }
            if (act_fire) {
              act_al = true;
              act_al_vm = k_full[act_k];
              act_res = R_HIT;
              act_slot = act_al_slot;
            }
          }
          aeg_push = act_fire;
        }
        break;
      case K_AMO:
        u0_mask = ~uint64_t(0) >> (64 - VB);
        u0_set = false;
        if (this->q_cnt(act_k) < UPD_DEPTH) {
          u0_valid = true;
          act_fire = true;
          act_res = R_NONE;
        }
        break;
      default:
        break;
      }
    }
    if (aeg_push) {
      aeg.line = a_line;
      aeg.lane = a_lane;
      aeg.src = t2_r_.src;
    }

    // Line results.
    std::array<bool, TB> res_now{};
    std::array<Res, TB> res_cls{};
    std::array<uint32_t, TB> res_slot{}, res_wid{};
    std::array<bool, TB> jn_valid{}, jn_store{};
    for (uint32_t k = 0; k < TB; ++k) {
      res_cls[k] = k_cls[k];
      res_slot[k] = tslot[k];
      res_wid[k] = m_id[k];
      switch (k_cls[k]) {
      case R_HIT:
        res_now[k] = fin_free;
        break;
      case R_JOIN:
        res_now[k] = fin_free;
        res_slot[k] = th[k] ? tslot[k] : m_slot[k];
        jn_valid[k] = fin_free;
        break;
      case R_NONE:
        res_now[k] = fin_free;
        break;
      case R_ACT:
        if (act_fire && act_k == k) {
          res_now[k] = true;
          res_cls[k] = act_res;
          res_slot[k] = act_slot;
          res_wid[k] = act_wid;
          jn_valid[k] = t2_r_.kind == K_STORE && m_hit[k] && !t2_r_.nc;
          jn_store[k] = true;
        }
        break;
      default:
        break;
      }
    }
    uint32_t res_mask = 0;
    for (uint32_t k = 0; k < TB; ++k)
      res_mask |= uint32_t(res_now[k]) << k;
    const uint32_t res_next = t2_res_ | res_mask;
    const bool lines_done = t2_valid_ && (t2_bv_ & ~res_next) == 0;
    const bool req_lines_done = lines_done && t2_last_;

    for (uint32_t k = 0; k < TB; ++k) {
      if (!res_now[k])
        continue;
      if (t2_r_.kind == K_LOAD)
        ++perf_stats_.reads;
      else if (t2_r_.kind == K_STORE)
        ++perf_stats_.writes;
    }
    if (mal_valid)
      ++perf_stats_.misses;

    auto acc_cls_n = t2_first_ ? std::array<Cls, LANES>{} : acc_cls_;
    auto acc_slot_n = acc_slot_;
    MMask acc_wait_n = t2_first_ ? MMask() : acc_wait_;
    for (uint32_t k = 0; k < TB; ++k) {
      if (!res_now[k])
        continue;
      const uint32_t l = t2_bl_[k];
      if (res_cls[k] == R_HIT) {
        acc_cls_n[l] = C_COMMIT;
      } else if (res_cls[k] == R_JOIN) {
        acc_cls_n[l] = C_REPLAY;
        acc_wait_n.set(res_wid[k]);
      } else {
        acc_cls_n[l] = C_NONE;
      }
      acc_slot_n[l] = res_slot[k];
    }
    // a store allocation may take the slot of a line the same request
    // already wrote: that line has left the L1 (its write-through stays),
    // so its bytes stay out of the slot, where the new line's must land
    if (act_fire && act_al && t2_r_.kind == K_STORE) {
      for (uint32_t j = 0; j < LANES; ++j) {
        if (j != t2_bl_[act_k] && acc_cls_n[j] == C_COMMIT && acc_slot_n[j] == act_al_slot)
          acc_cls_n[j] = C_NONE;
      }
    }

    // The request's commit and replay items.
    uint64_t bld_cm = 0, bld_rm = 0;
    for (uint32_t j = 0; j < LANES; ++j) {
      if (!((t2_r_.mask >> j) & 1))
        continue;
      const Cls c = acc_cls_n[t2_r_.lead_idx[j]];
      if (c == C_COMMIT) bld_cm |= bit(j);
      if (c == C_REPLAY) bld_rm |= bit(j);
    }

    // A request whose next line waits only on what the request itself holds
    // (every way of the set, every MSHR entry, or the drawn victim of an old
    // list it could hold whole) would never get it: what it has resolved
    // commits now, and its other lines go on as a new pass.
    auto self_held = [&](uint32_t slot) {
      if (pinned(slot))
        return true;
      for (uint32_t k = 0; k < TB; ++k) {
        if (k_cls[k] == R_JOIN && (th[k] ? tslot[k] : m_slot[k]) == slot)
          return true;
      }
      if (t2_first_)
        return false;
      for (uint32_t j = 0; j < LANES; ++j) {
        if (acc_cls_[j] == C_REPLAY && acc_slot_[j] == slot)
          return true;
      }
      return false;
    };
    bool way_held = !tinv[act_k] && !th[act_k]
      && ((t2_r_.kind == K_LOAD && !(t2_r_.nc && free_avail))
          || (t2_r_.kind == K_STORE && !t2_r_.nc && !m_hit[act_k]));
    for (uint32_t w = 0; way_held && w < WAYS; ++w)
      way_held = self_held(t_rd_[act_k][w].slot);
    const bool mshr_held = t2_r_.kind == K_LOAD && !mal_ready && acc_wait_n.all();
    const bool draw_held = pool_draw && pool.wait == WAIT_VICTIM_IN_FLIGHT && self_held(vref.slot)
      && n_old_ <= LANES;
    const bool split = t2_valid_ && t2_phase_ == P_LINES && (t2_r_.kind == K_LOAD || t2_r_.kind == K_STORE)
      && act_any && !act_fire && fin_free && (way_held || mshr_held || draw_held)
      && (bld_rm == 0 || rq_resv_ < RQ_DEPTH);
    const bool split_rq = split && bld_rm != 0;

    uint32_t emit_lane = 0;
    for (int j = int(LANES) - 1; j >= 0; --j) {
      if ((t2_emit_ >> j) & 1)
        emit_lane = j;
    }
    const bool amo_emit_last = (t2_emit_ & ~bit(emit_lane)) == 0;
    const bool t2_is_mem = t2_r_.kind == K_LOAD || t2_r_.kind == K_STORE;
    const bool t2_fin_now = req_lines_done && t2_is_mem;
    bool t2_done = false, emit_push = false, fence_push = false, inval_start = false;
    if (t2_valid_) {
      switch (t2_phase_) {
      case P_LINES: t2_done = lines_done && (!t2_last_ || (t2_is_mem && fin_free)); break;
      case P_EMIT:
        emit_push = !eg_full;
        t2_done = emit_push && amo_emit_last;
        break;
      case P_INV:
        inval_start = !inv_wait_;
        t2_done = t2_inv_ && inv_wait_ && !inval_r_;
        break;
      case P_PUSH:
        fence_push = !eg_full;
        t2_done = fence_push;
        break;
      }
    }
    const bool req_end = t2_valid_ && t2_done && t2_last_;
    // an invalidate ends after the valid clear and answers here
    const bool inv_end = t2_valid_ && t2_done && t2_phase_ == P_INV;

    bool eg_push = aeg_push;
    EgItem eg_in = aeg;
    if (emit_push || fence_push) {
      eg_push = true;
      eg_in = EgItem{};
      eg_in.kind = EG_LANE;
      eg_in.line = t2_r_.line(emit_lane);
      eg_in.lane = emit_lane;
      eg_in.idx = emit_lane;
      if (emit_push)
        eg_in.idx |= this->amo_key(t2_r_.blk, t2_r_.tag) << 16;
      eg_in.src = t2_r_.src;
    }

    // T1.
    const bool mshr_empty = m_valid_.none();
    const bool drained = !t2_valid_ && !fin_c_valid_ && !fin_r_valid_ && rq_.empty() && !a_valid_
      && mshr_empty && !fl_valid_ && alq_n_ == 0 && !fn_valid_ && amo_out_ == 0;
    const bool in_gate = (in_r_.kind == K_FENCE || (in_r_.kind == K_LOAD && in_cta_)) ? drained
      : ((in_r_.kind == K_AMO) ? !fn_valid_ : true);
    const bool t2_free = !t2_valid_ || t2_done;
    const bool t1_issue = in_valid_ && t2_free && in_gate && !inval_r_;
    std::array<bool, TB> iss_v{};
    std::array<uint32_t, TB> iss_l{};
    for (uint32_t k = 0; k < TB; ++k) {
      for (int i = int(LANES) - 1; i >= 0; --i) {
        if (((in_todo_ >> i) & 1) && bank_of(set_of(in_r_.line(i))) == k) {
          iss_v[k] = true;
          iss_l[k] = i;
        }
      }
    }
    uint64_t iss_mask = 0;
    uint32_t iss_bv = 0;
    for (uint32_t k = 0; k < TB; ++k) {
      if (iss_v[k]) {
        iss_mask |= bit(iss_l[k]);
        iss_bv |= 1u << k;
      }
    }
    const bool in_last = (in_todo_ & ~iss_mask) == 0;
    const uint32_t held = (t2_valid_ && !t2_done) ? (t2_bv_ & ~res_next) : 0;
    std::array<bool, TB> lk_v{};
    std::array<uint32_t, TB> lk_r{};
    for (uint32_t k = 0; k < TB; ++k) {
      if (t1_issue) {
        lk_v[k] = iss_v[k];
        lk_r[k] = trow_of(set_of(in_r_.line(iss_l[k])));
      } else {
        lk_v[k] = (held >> k) & 1;
        lk_r[k] = trow_of(set_of(t2_line[k]));
      }
    }

    // The fill (offered from registers in the previous cycle).
    const uint32_t fl_id = fl_id_;
    const MshrEnt fe = mshr_[fl_id];
    const bool fe_cacheable = fe.cacheable;
    const bool inval_busy = inval_r_;
    uint64_t fe_words = 0;
    for (uint32_t s = 0; s < NSEC; ++s) {
      if ((fe.sectors >> s) & 1) {
        for (uint32_t v = 0; v < SVB; ++v)
          fe_words |= bit(s * SVB + v);
      }
    }

    // Tag update sources, admitted in source order.
    std::array<bool, UPD_SRCS> up_valid{};
    std::array<uint32_t, UPD_SRCS> up_bank{};
    std::array<Upd, UPD_SRCS> up{};
    up_valid[U_STORE] = u0_valid;
    up_bank[U_STORE] = act_k;
    up[U_STORE] = Upd{a_row, tway[act_k], u0_mask, u0_set, false, U_STORE, 0};
    up_valid[U_VICTIM] = pool.victim;
    up_bank[U_VICTIM] = bank_of(vref.set);
    up[U_VICTIM] = Upd{trow_of(vref.set), vref.way, ~uint64_t(0) >> (64 - VB), false, true, U_VICTIM, 0};
    up_valid[U_FILL] = fill_fire && fe.cacheable;
    up_bank[U_FILL] = bank_of(fe.set);
    up[U_FILL] = Upd{trow_of(fe.set), fe.way, fe_words, true, false, U_FILL, fl_id_};
    std::array<bool, UPD_SRCS> up_ok{};
    for (uint32_t s = 0; s < UPD_SRCS; ++s) {
      uint32_t used = 0;
      for (uint32_t p = 0; p < s; ++p) {
        if (up_valid[p] && up_ok[p] && up_bank[p] == up_bank[s])
          ++used;
      }
      up_ok[s] = this->q_cnt(up_bank[s]) + used + 1 <= UPD_DEPTH;
    }
    if (up_valid[U_FILL] && !up_ok[U_FILL])
      throw std::runtime_error(std::string(so->name()) + ": fill valid-set without room");

    std::array<bool, TB> hz{};
    for (uint32_t k = 0; k < TB; ++k) {
      for (uint32_t q = 0; q < alq_n_; ++q)
        hz[k] = hz[k] || (alq_[q].bank == k && alq_[q].row == lk_r[k]);
      hz[k] = hz[k] || (act_al && act_k == k && a_row == lk_r[k]);
      for (uint32_t s = 0; s < 2; ++s)
        hz[k] = hz[k] || (up_valid[s] && up_ok[s] && up_bank[s] == k && up[s].row == lk_r[k]);
    }
    uint32_t hz_mask = 0;
    for (uint32_t k = 0; k < TB; ++k)
      hz_mask |= uint32_t(hz[k]) << k;

    // IN: round robin over the blocks.
    const bool in_free = !in_valid_ || (t1_issue && in_last);
    int arb_idx = -1;
    for (uint32_t n = 1; n <= config_.num_blocks; ++n) {
      const uint32_t b = (rr_last_ + n) % config_.num_blocks;
      if (!so->LsuReqIn.at(b).empty()) {
        arb_idx = int(b);
        break;
      }
    }
    const bool hf_sel = hf_req_ && !hf_sent_ && !hf_done_ && arb_idx < 0;
    const bool in_load = (arb_idx >= 0 || hf_sel) && in_free && rq_resv_ + uint32_t(split_rq) < RQ_DEPTH;


    // Memory responses; an atomic lane's reply goes straight to the LSU, so the next
    // warp atomic never waits for the replies of the previous one.
    const bool rs_free = !rs_valid_ || so->LsuRspOut.at(rs_blk_).size() == 0;
    bool mem_rsp_fire = false, rsp_special = false, rsp_amo = false, rsp_line_done = false;
    uint32_t rsp_idx = 0;
    if (!so->MemRspIn.empty()) {
      const auto& r = so->MemRspIn.peek();
      rsp_special = (r.tag >> 15) & 1;
      rsp_idx = uint32_t(r.tag & 0x7fff);
      if (rsp_special) {
        rsp_amo = !fn_valid_;
        mem_rsp_fire = !rsp_amo || rs_free;
      } else {
        const uint32_t blk = uint32_t(r.tag >> 16);
        auto& as = asm_.at(rsp_idx);
        const bool last = (as.got | (1u << blk)) == as.need;
        rsp_line_done = last;
        mem_rsp_fire = !last || !fl_valid_ || fill_fire;
      }
    }

    // Responses to the LSU.
    const bool fn_done = fn_valid_ && fn_got_;
    const bool amo_rsp = mem_rsp_fire && rsp_amo;
    const bool gl_rsp_ready = rs_free && !fn_done && !amo_rsp;
    const bool gl_rsp_fire = gl_rsp_ready && !so->GlRspIn.empty();

    // MSHR retirement: filled, and no replay of it still to leave the array.
    bool rt_valid = false;
    uint32_t rt_id = 0;
    for (uint32_t i = 0; i < MSHR; ++i) {
      if (m_valid_.test(i) && m_filled_.test(i) && m_rpc_[i] == 0 && !looked.test(i)) {
        rt_valid = true;
        rt_id = i;
        break;
      }
    }
    const uint32_t rt_slot = mshr_[rt_id].slot;
    const bool rt_cacheable = mshr_[rt_id].cacheable;

    ///////////////////////////////////////////////////////////////////////////
    // Clock edge.

    // Egress head (present at the start of the cycle) to the SM port.
    this->egress_send();

    // Tags: lookups read the macros before this cycle's writes.
    std::array<bool, TB> tw_valid{};
    std::array<Upd, TB> tw_head{};
    {
      std::array<std::array<uint64_t, WAYS>, TB> bs{}, bc{};
      std::array<std::array<bool, WAYS>, TB> bi{};
      for (uint32_t b = 0; b < TB; ++b) {
        for (const auto& e : tq_[b]) {
          if (e.row != lk_r[b])
            continue;
          if (e.set) {
            bs[b][e.way] |= e.mask;
          } else {
            bs[b][e.way] &= ~e.mask;
            bc[b][e.way] |= e.mask;
          }
          if (e.inval)
            bi[b][e.way] = true;
        }
      }
      for (uint32_t b = 0; b < TB; ++b) {
        if (lk_v[b]) {
          for (uint32_t w = 0; w < WAYS; ++w) {
            t_rd_[b][w] = this->tag_at(b, lk_r[b], w);
            v_rd_[b][w] = this->vm_at(b, lk_r[b], w);
          }
        }
        byp_s_[b] = bs[b];
        byp_c_[b] = bc[b];
        byp_inv_[b] = bi[b];
      }
      bool granted = false;
      for (uint32_t b = 0; b < TB; ++b) {
        auto& q = tq_[b];
        for (uint32_t s = 0; s < UPD_SRCS; ++s) {
          if (up_valid[s] && up_ok[s] && up_bank[s] == b)
            q.push_back(up[s]);
        }
        if (inval_r_) {
          for (uint32_t w = 0; w < WAYS; ++w) {
            this->tag_at(b, inval_row_, w) = TagWord{};
            this->vm_at(b, inval_row_, w) = 0;
          }
        } else if (!q.empty()) {
          const Upd h = q.front();
          q.pop_front();
          if (h.inval) {
            this->tag_at(b, h.row, h.way) = TagWord{};
            this->vm_at(b, h.row, h.way) = 0;
          } else if (h.set) {
            this->vm_at(b, h.row, h.way) |= h.mask;
          } else {
            this->vm_at(b, h.row, h.way) &= ~h.mask;
          }
          tw_valid[b] = true;
          tw_head[b] = h;
        } else if (alq_n_ != 0 && alq_[0].bank == b) {
          const auto& a = alq_[0];
          this->tag_at(b, a.row, a.way) = TagWord{true, a.line, a.slot};
          this->vm_at(b, a.row, a.way) = a.vmask;
          granted = true;
        }
      }
      if (inval_start && !inval_r_) {
        inval_r_ = true;
        inval_row_ = 0;
      } else if (inval_r_) {
        if (inval_row_ == SPB - 1)
          inval_r_ = false;
        ++inval_row_;
      }
      // the allocation-write queue
      if (granted) {
        for (uint32_t q = 0; q + 1 < ALQ; ++q)
          alq_[q] = alq_[q + 1];
        --alq_n_;
      }
      if (act_fire && act_al) {
        alq_[alq_n_] = Alq{act_k, a_row, act_al_way, a_line, act_al_slot, act_al_vm};
        ++alq_n_;
      }
    }

    // Slots.
    this->slots_edge(pool, inval_start, lfsr_step, open_valid, open_slot,
                     rt_valid, rt_slot, !rt_cacheable);

    // MSHR.
    const MMask take_wait = (gl_fire && take_rep) ? rq_.front().wait : MMask();
    auto ilk_next = [&](uint8_t m) {
      uint8_t n = 0;
      if ((m & ILK_FIN) && !fin_c_take)
        n |= ILK_FIN;
      if (gl_fire ? ((m & ILK_FIN) && !take_rep && !wf_valid) : ((m & ILK_ARR) && !wf_valid))
        n |= ILK_ARR;
      return n;
    };
    MMask rp_inc, rp_dec;
    for (uint32_t k = 0; k < TB; ++k) {
      if (res_now[k] && res_cls[k] == R_JOIN)
        rp_inc.set(res_wid[k]);
    }
    if (wf_valid)
      rp_dec = gl_fire ? take_wait : (a_rep_ ? a_wait_ : MMask());
    if (rt_valid)
      m_valid_.reset(rt_id);
    if (fill_fire)
      m_filled_.set(fl_id);
    for (uint32_t i = 0; i < MSHR; ++i) {
      if (rp_dec.test(i) && m_rpc_[i] == 0)
        throw std::runtime_error(std::string(so->name()) + ": replay count underflow");
      m_rpc_[i] = m_rpc_[i] + uint32_t(rp_inc.test(i)) - uint32_t(rp_dec.test(i));
      if (m_rpc_[i] > RQ_DEPTH)
        throw std::runtime_error(std::string(so->name()) + ": replay count overflow");
      m_ilk_[i] = ilk_next(m_ilk_[i]);
    }
    for (uint32_t k = 0; k < TB; ++k) {
      if (jn_valid[k] && jn_store[k])
        m_store_.set(m_id[k]);
    }
    if (mal_valid && mal_ready) {
      // an entry with no sector to fetch is never filled and never retires
      if (mal.sectors == 0)
        throw std::runtime_error(std::string(so->name()) + ": line read with no sector");
      mshr_[mal_id] = mal;
      m_valid_.set(mal_id);
      m_filled_.reset(mal_id);
      m_store_.reset(mal_id);
      m_rpc_[mal_id] = uint32_t(rp_inc.test(mal_id));
      m_ilk_[mal_id] = mal.ilk_valid
        ? ilk_next((fin_c_valid_ ? ILK_FIN : 0) | ((a_valid_ && !a_rep_) ? ILK_ARR : 0))
        : 0;
      auto& as = asm_.at(mal_id);
      as.need = 0;
      as.got = 0;
      for (uint32_t b = 0; b < LINE; ++b) {
        if (aeg.byteen[b])
          as.need |= 1u << (b / BLOCK);
      }
    }

    // Replay FIFO, egress FIFO.
    if (fin_r_valid_) {
      GlItem it = fin_r_;
      it.wait = fin_wait_;
      rq_.push_back(it);
    }
    const bool rq_pop = gl_fire && take_rep;
    if (rq_pop)
      rq_.pop_front();
    if (eg_push)
      eg_.push_back(eg_in);

    // T2 register, its accumulators and the atomic/fence register (old values).
    const Req& t2_prev = t2_r_;
    const auto bld_c = this->build(t2_prev, acc_slot_n, bld_cm);
    const auto bld_r = this->build(t2_prev, acc_slot_n, bld_rm);
    if (t2_valid_ && (t2_first_ || res_mask != 0)) {
      acc_cls_ = acc_cls_n;
      acc_slot_ = acc_slot_n;
      acc_wait_ = acc_wait_n;
    }
    const bool fn_set = t2_valid_ && t2_phase_ == P_LINES && lines_done && t2_last_ && t2_r_.kind == K_FENCE;
    const Req fn_req = fn_set ? t2_prev : Req{};
    if (draw_start) draw_on_ = true;
    if (draw_done) draw_on_ = false;

    // FIN.
    if (fin_c_take) fin_c_valid_ = false;
    if (fin_r_valid_) fin_r_valid_ = false;
    if ((t2_fin_now && t2_done) || split) {
      fin_c_valid_ = bld_cm != 0;
      fin_r_valid_ = bld_rm != 0;
      fin_c_ = bld_c;
      fin_r_ = bld_r;
      fin_wait_ = acc_wait_n;
    }
    rq_resv_ = rq_resv_ + uint32_t(in_load) + uint32_t(split_rq)
             - uint32_t(req_end && !(t2_is_mem && bld_rm != 0))
             - uint32_t(rq_pop);

    if (t1_issue) {
      t2_valid_ = true;
      t2_first_ = (in_todo_ == in_r_.lead);
      t2_last_ = in_last;
      t2_r_ = in_r_;
      t2_inv_ = in_inv_;
      t2_bv_ = iss_bv;
      t2_bl_ = iss_l;
      t2_res_ = 0;
      t2_stale_ = hz_mask;
      t2_phase_ = P_LINES;
      t2_emit_ = in_r_.mask;
    } else if (t2_done) {
      t2_valid_ = false;
    } else if (t2_valid_) {
      const Phase ph = t2_phase_;
      t2_res_ = res_next;
      t2_stale_ = hz_mask & held;
      t2_first_ = split;
      if (ph == P_LINES && lines_done && t2_last_ && !t2_is_mem)
        t2_phase_ = (t2_r_.kind == K_AMO) ? P_EMIT : P_INV;
      if (emit_push)
        t2_emit_ &= ~bit(emit_lane);
      if (ph == P_INV) {
        if (inval_start) {
          inv_wait_ = true;
        } else if (!inval_busy) {
          inv_wait_ = false;
          t2_phase_ = P_PUSH;
        }
      }
    }

    // IN register.
    if (in_load && hf_sel) {
      LsuReq f(LANES);
      f.op = MemOp::FLUSH;
      f.flags.fence = 1;
      f.mask.set(0);
      this->load_in(0, f, true);
      hf_sent_ = true;
      seq_ctr_ = seq_add(seq_ctr_, 1);
    } else if (in_load) {
      rr_last_ = uint32_t(arb_idx);
      auto& in = so->LsuReqIn.at(arb_idx);
      this->load_in(uint32_t(arb_idx), in.peek(), false);
      in.pop();
      seq_ctr_ = seq_add(seq_ctr_, 1);
    } else if (t1_issue) {
      in_todo_ &= ~iss_mask;
      if (in_last)
        in_valid_ = false;
    }

    if (wf_valid) a_valid_ = false;
    if (gl_fire) {
      a_valid_ = !wf_valid;
      a_rep_ = take_rep;
      a_seq_ = gl_off_seq_;
      a_wait_ = take_wait;
      gl_off_ = false;
    }

    // Memory responses and the fill register.
    if (fill_fire) {
      fl_valid_ = false;
      fill_off_ = false;
    }
    if (mem_rsp_fire && !rsp_amo) {
      const auto& r = so->MemRspIn.peek();
      if (rsp_special) {
        fn_got_ = true;
      } else {
        const uint32_t blk = uint32_t(r.tag >> 16);
        auto& as = asm_.at(rsp_idx);
        as.got |= 1u << blk;
        if (r.data)
          std::memcpy(as.data.data() + blk * BLOCK, r.data->data(), std::min(BLOCK, LINE - blk * BLOCK));
        if (rsp_line_done) {
          fl_valid_ = true;
          fl_id_ = rsp_idx;
          fl_data_ = as.data;
        }
      }
      so->MemRspIn.pop();
    }
    for (uint32_t b = 0; b < TB; ++b) {
      if (tw_valid[b] && tw_head[b].src == U_FILL)
        done_.set(tw_head[b].id);
    }
    if (fill_fire && !fe_cacheable)
      done_.set(fl_id);
    if (mal_valid && mal_ready)
      done_.reset(mal_id);

    // Response register.
    if (rs_free) rs_valid_ = false;
    if (fn_done && rs_free && fn_host_) {
      hf_done_ = true;
      hf_req_ = false;
    } else if (fn_done && rs_free) {
      LsuRsp out(LANES);
      out.tag = fn_tag_;
      out.cid = fn_src_ ? fn_src_->cid : 0;
      out.uuid = fn_src_ ? fn_src_->uuid : 0;
      for (uint32_t i = 0; i < LANES; ++i) {
        if ((fn_mask_ >> i) & 1) {
          out.mask.set(i);
          out.data.at(i) = word_block(0, 0);
        }
      }
      this->respond(fn_blk_, out);
    } else if (amo_rsp) {
      const auto& r = so->MemRspIn.peek();
      const uint32_t key = r.tag >> 16;
      const uint32_t blk = key % uint32_t(ctx_.size());
      const uint32_t tag = key / uint32_t(ctx_.size());
      auto it = ctx_.at(blk).find(tag);
      if (it == ctx_.at(blk).end())
        throw std::runtime_error(std::string(so->name()) + ": atomic response with no request");
      auto& c = it->second;
      LsuRsp out(LANES);
      out.tag = tag;
      out.cid = c.cid;
      out.uuid = c.uuid;
      out.mask.set(rsp_idx);
      out.data.at(rsp_idx) = r.data ? r.data : word_block(0, 0);
      c.left &= ~bit(rsp_idx);
      if (c.left == 0)
        ctx_.at(blk).erase(it);
      this->respond(blk, out);
      so->MemRspIn.pop();
    } else if (gl_rsp_fire) {
      const auto& g = so->GlRspIn.peek();
      const uint32_t blk = uint32_t(g.tag >> 32);
      const uint32_t tag = uint32_t(g.tag);
      auto it = ctx_.at(blk).find(tag);
      if (it == ctx_.at(blk).end())
        throw std::runtime_error(std::string(so->name()) + ": global response with no request");
      auto& c = it->second;
      LsuRsp out(LANES);
      out.tag = tag;
      out.cid = c.cid;
      out.uuid = c.uuid;
      for (uint32_t i = 0; i < LANES; ++i) {
        if (g.mask.test(i)) {
          out.mask.set(i);
          out.data.at(i) = word_block(g.data.at(i), c.boff[i]);
          c.left &= ~bit(i);
        }
      }
      if (c.left == 0)
        ctx_.at(blk).erase(it);
      this->respond(blk, out);
      so->GlRspIn.pop();
    }
    if (inv_end) {
      inv_wait_ = false;
      fn_got_ = true;
    }
    if (fn_done && rs_free) fn_valid_ = false;
    if (fn_set) {
      fn_valid_ = true;
      fn_blk_ = fn_req.blk;
      fn_tag_ = fn_req.tag;
      fn_mask_ = fn_req.mask;
      fn_got_ = false;
      fn_src_ = fn_req.src;
      fn_host_ = fn_req.host;
    }
    amo_out_ = amo_out_ + uint32_t(emit_push) - uint32_t(amo_rsp);

    ///////////////////////////////////////////////////////////////////////////
    // Offers for the next cycle, from registers.

    if (!gl_off_) {
      const bool rq_ready = !rq_.empty() && (rq_.front().wait & ~done_).none();
      if (rq_ready || fin_c_valid_) {
        const auto& item = rq_ready ? rq_.front() : fin_c_;
        so->GlReqOut.send(item.req, 0);
        gl_off_ = true;
        gl_off_rep_ = rq_ready;
        gl_off_seq_ = item.req.seq;
      }
    }
    if (fl_valid_ && !fill_off_) {
      const MshrEnt& f = mshr_[fl_id_];
      const bool blocked = m_ilk_[fl_id_] != 0;
      if (!blocked && (!f.cacheable || this->q_cnt(bank_of(f.set)) + 2 <= UPD_DEPTH)) {
        SmSramFill fill;
        fill.row = ROWS - 1 - f.slot;
        fill.src = SmSramSrc::L1;
        uint64_t words = 0;
        for (uint32_t s = 0; s < NSEC; ++s) {
          if ((f.sectors >> s) & 1) {
            for (uint32_t v = 0; v < SVB; ++v)
              words |= bit(s * SVB + v);
          }
        }
        for (uint32_t b = 0; b < LINE; ++b) {
          const uint32_t v = b / VBYTES;
          fill.byteen.at(b) = ((words >> v) & 1) && !((f.vmiss >> v) & 1);
          fill.data.at(b) = fl_data_[b];
        }
        so->FillOut.send(fill, 0);
        fill_off_ = true;
      }
    }
  }

private:
  struct Alq {
    uint32_t bank = 0;
    uint32_t row = 0;
    uint32_t way = 0;
    uint64_t line = 0;
    uint32_t slot = 0;
    uint64_t vmask = 0;
  };

  struct Asm {
    uint32_t need = 0;
    uint32_t got = 0;
    std::array<uint8_t, LINE> data{};
  };

  struct Pool {
    bool ok = false;
    PoolWait wait = WAIT_NONE;
    PoolKind kind = KIND_FREE;
    uint32_t slot = 0;
    bool victim = false;
    // edge effects
    bool take_free = false, take_bump = false, push_young = false, pop_swap = false;
    SlotRef push_ref;
    bool start_draw = false, want_migrate = false, draw_lfsr = false;
    bool is_draw = false;
  };

  TagWord& tag_at(uint32_t b, uint32_t row, uint32_t w) {
    return tag_mem_[(size_t(b) * SPB + row) * WAYS + w];
  }
  uint64_t& vm_at(uint32_t b, uint32_t row, uint32_t w) {
    return vm_mem_[(size_t(b) * SPB + row) * WAYS + w];
  }
  uint32_t q_cnt(uint32_t b) const { return uint32_t(tq_[b].size()); }

  // ---- slots (VX_sm_l1_slots) ----

  void slots_clear() {
    bump_ = 0;
    n_old_ = 0;
    draw_r_ = false;
    draw_idx_ = 0;
    last_wen_ = false;
    last_widx_ = 0;
    y_head_ = y_tail_ = y_cnt_ = 0;
    f_head_ = f_tail_ = f_cnt_ = 0;
    std::fill(open_.begin(), open_.end(), false);
  }

  uint32_t lfsr_n() const {
    return (lfsr_ >> 1) ^ ((lfsr_ & 1) ? LFSR_TAPS : 0);
  }

  bool free_avail() const { return f_cnt_ != 0 || bump_ < cfg_.l1_lines; }

  SlotRef victim_ref() const {
    return (last_wen_ && last_widx_ == draw_idx_) ? last_wdata_ : old_rdata_;
  }

  Pool pool_eval(bool valid, PoolOp op, uint32_t set, uint32_t way, bool v_busy) const {
    Pool p;
    const bool is_borrow = valid && op == OP_BORROW;
    const bool is_young = valid && op == OP_YOUNG;
    p.is_draw = valid && op == OP_DRAW;
    const bool young_full = y_cnt_ >= YOUNG_DEPTH;
    const uint32_t free_head = free_mem_[f_head_];
    p.slot = free_head;
    p.push_ref = SlotRef{free_head, set, way};
    if (is_borrow || is_young) {
      if (is_young && young_full) {
        p.wait = WAIT_YOUNG_FULL;
      } else {
        p.ok = true;
        if (f_cnt_ != 0) {
          p.take_free = true;
          p.kind = is_borrow ? KIND_BORROW : KIND_FREE;
          p.slot = free_head;
        } else {
          p.take_bump = true;
          p.kind = is_borrow ? KIND_BORROW : KIND_BUMP;
          p.slot = bump_;
        }
        p.push_young = is_young;
        p.push_ref = SlotRef{p.slot, set, way};
      }
    } else if (p.is_draw) {
      const SlotRef v = this->victim_ref();
      if (draw_r_ && !open_[v.slot] && !v_busy) {
        p.ok = true;
        p.kind = KIND_AGE;
        p.slot = v.slot;
        p.victim = true;
        p.pop_swap = true;
        p.push_young = true;
        p.push_ref = SlotRef{v.slot, set, way};
      } else if (young_full) {
        p.wait = WAIT_YOUNG_FULL;
      } else if (n_old_ == 0) {
        p.want_migrate = true;
        p.wait = WAIT_OLD_EMPTY;
      } else {
        p.start_draw = true;
        p.draw_lfsr = true;
        p.wait = draw_r_ ? WAIT_VICTIM_IN_FLIGHT : WAIT_DRAW_PENDING;
      }
    }
    return p;
  }

  void slots_edge(const Pool& p, bool invalidate, bool lfsr_step, bool open_valid, uint32_t open_slot,
                  bool retire_valid, uint32_t retire_slot, bool retire_borrowed) {
    const uint32_t lfsr_next = this->lfsr_n();
    const uint32_t draw_idx_n = uint32_t((uint64_t(lfsr_next) * n_old_) >> 16) & (OLD_DEPTH - 1);
    const uint32_t y_cnt_req = y_cnt_ + uint32_t(p.push_young && !p.pop_swap);
    const bool migrate = !p.pop_swap && y_cnt_req != 0
      && (y_cnt_req > cfg_.l1_protect || p.want_migrate);
    const bool pop_young = p.pop_swap || migrate;
    const uint32_t old_widx = p.pop_swap ? draw_idx_ : n_old_;
    const SlotRef young_head = young_mem_[y_head_];
    const bool retire_free = retire_valid && retire_borrowed;
    if (retire_free && f_cnt_ == FREE_DEPTH)
      throw std::runtime_error(std::string(simobject_->name()) + ": free FIFO overflow");

    // old table: read (before write), then write
    if (p.start_draw)
      old_rdata_ = old_mem_[draw_idx_n];
    if (pop_young)
      old_mem_[old_widx] = young_head;
    if (p.push_young)
      young_mem_[y_tail_] = p.push_ref;
    if (retire_free)
      free_mem_[f_tail_] = retire_slot;

    if (p.draw_lfsr || lfsr_step)
      lfsr_ = lfsr_next;

    if (invalidate) {
      this->slots_clear();
      return;
    }
    if (p.take_bump)
      ++bump_;
    if (p.start_draw) {
      draw_r_ = true;
      draw_idx_ = draw_idx_n;
    } else if (p.is_draw) {
      draw_r_ = false;
    }
    if (migrate)
      ++n_old_;
    if (p.push_young)
      y_tail_ = (y_tail_ + 1) % YOUNG_DEPTH;
    if (pop_young)
      y_head_ = (y_head_ + 1) % YOUNG_DEPTH;
    y_cnt_ = y_cnt_ + uint32_t(p.push_young) - uint32_t(pop_young);
    if (retire_free)
      f_tail_ = (f_tail_ + 1) % FREE_DEPTH;
    if (p.take_free)
      f_head_ = (f_head_ + 1) % FREE_DEPTH;
    f_cnt_ = f_cnt_ + uint32_t(retire_free) - uint32_t(p.take_free);
    last_wen_ = pop_young;
    last_widx_ = old_widx;
    last_wdata_ = young_head;
    if (retire_valid)
      open_[retire_slot] = false;
    if (open_valid)
      open_[open_slot] = true;
  }

  // ---- IN ----

  void load_in(uint32_t blk, const LsuReq& req, bool host) {
    Req r;
    r.blk = blk;
    r.host = host;
    r.tag = req.tag;
    r.seq = seq_ctr_;
    r.src = std::make_shared<LsuReq>(req);
    int first = -1;
    for (uint32_t i = 0; i < LANES; ++i) {
      if (!req.mask.test(i))
        continue;
      if (first < 0)
        first = int(i);
      r.mask |= bit(i);
      const uint64_t a = req.addrs.at(i);
      const uint32_t boff = uint32_t((a & ~uint64_t(WORD - 1)) % BLOCK);
      r.waddr[i] = a / WORD;
      r.byteen[i] = uint32_t(req.byteen.at(i) >> boff) & ((1u << WORD) - 1);
      r.data[i] = load_word(req.data.at(i), boff);
    }
    const bool fence = (req.flags.fence || req.op == MemOp::FLUSH);
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    const bool cta = fence && req.flags.fence_cta;
#else
    const bool cta = false;
#endif
    Kind kind = (fence && !cta) ? K_FENCE : (req.is_amo() ? K_AMO : (req.is_write() ? K_STORE : K_LOAD));
    r.kind = cta ? K_LOAD : kind;
    in_cta_ = cta;
    in_inv_ = fence && req.flags.inval;
    r.nc = (first >= 0) && get_addr_type(req.addrs.at(first)) == AddrType::IO;
    for (uint32_t i = 0; i < LANES; ++i) {
      r.lead_idx[i] = i;
      bool dup = false;
      for (int j = int(i) - 1; j >= 0; --j) {
        if (((r.mask >> j) & 1) && r.line(j) == r.line(i)) {
          r.lead_idx[i] = j;
          dup = true;
        }
      }
      if (((r.mask >> i) & 1) && !dup && kind != K_FENCE)
        r.lead |= bit(i);
    }
    if (r.kind == K_LOAD || r.kind == K_AMO) {
      RespCtx c;
      c.cid = req.cid;
      c.uuid = req.uuid;
      c.left = r.mask;
      for (uint32_t i = 0; i < LANES; ++i)
        c.boff[i] = uint32_t((req.addrs.at(i) & ~uint64_t(WORD - 1)) % BLOCK);
      ctx_.at(blk)[req.tag] = c;
    }
    in_valid_ = true;
    in_r_ = r;
    in_todo_ = r.lead;
  }

  GlItem build(const Req& t, const std::array<uint32_t, LANES>& slot, uint64_t mask) const {
    GlItem g;
    g.req.op = (t.kind == K_STORE) ? SmSramOp::WRITE : SmSramOp::READ;
    g.req.tag = (uint64_t(t.blk) << 32) | t.tag;
    g.req.seq = t.seq;
    for (uint32_t j = 0; j < LANES; ++j) {
      auto& l = g.req.lanes.at(j);
      l.row = ROWS - 1 - slot[t.lead_idx[j]];
      l.boff = t.wi(j) * WORD;
      l.byteen = t.byteen[j];
      l.data = t.data[j];
      if ((mask >> j) & 1)
        g.req.mask.set(j);
    }
    return g;
  }

  // an atomic lane's miss-bus tag carries its LSU block and tag (VX_sm_l1 eg_tag)
  uint32_t amo_key(uint32_t blk, uint32_t tag) const {
    const uint32_t key = tag * uint32_t(ctx_.size()) + blk;
    if (key >= (1u << 16))
      throw std::runtime_error(std::string(simobject_->name()) + ": atomic tag does not fit the miss-bus tag");
    return key;
  }

  void respond(uint32_t blk, const LsuRsp& out) {
    simobject_->LsuRspOut.at(blk).send(out, 0);
    rs_valid_ = true;
    rs_blk_ = blk;
  }

  // ---- egress to the SM port ----

  void egress_send() {
    if (eg_.empty())
      return;
    const EgItem& e = eg_.front();
    auto& out = simobject_->MemReqOut;
    std::vector<MemReq> reqs;
    const auto& lr = *e.src;
    const uint32_t lane = e.lane;
    const uint32_t hart = make_hart_id(lr.cid, lr.wid, lr.tids.at(lane));
    if (e.kind == EG_LANE) {
      MemReq m;
      m.op = lr.op;
      m.addr = lr.addrs.at(lane);
      m.data = lr.data.at(lane);
      m.byteen = lr.byteen.at(lane);
      // The comparand travels in the other 16-B half of the operand's sector,
      // at the operand's byte position and width.
      if (memop_is_amo(m.op) && m.data) {
        uint32_t off = uint32_t(m.addr % VX_CFG_MEM_BLOCK_SIZE) ^ 16;
        uint32_t width = (__builtin_popcountll(m.byteen) >= 8) ? 8 : 4;
        m.data = std::make_shared<mem_block_t>(*m.data);
        std::memcpy(m.data->data() + off, &lr.amo_cmp.at(lane), width);
      }
      m.amo_cmp = 0;
      m.flags = lr.flags;
      m.hart_id = hart;
      m.uuid = lr.uuid;
      m.tag = (1u << 15) | e.idx;
      auto t = get_addr_type(m.addr);
      m.flags.io = (t == AddrType::IO);
      m.flags.local = (t == AddrType::Shared);
      reqs.push_back(m);
    } else {
      for (uint32_t b = 0; b < LBLOCKS; ++b) {
        uint64_t be = 0;
        auto blk = make_mem_block();
        std::memset(blk->data(), 0, blk->size());
        for (uint32_t o = 0; o < BLOCK && b * BLOCK + o < LINE; ++o) {
          if (e.byteen[b * BLOCK + o]) {
            be |= bit(o);
            (*blk)[o] = e.data[b * BLOCK + o];
          }
        }
        if (be == 0)
          continue;
        MemReq m;
        m.op = (e.kind == EG_WRITE) ? MemOp::ST : MemOp::LD;
        m.addr = e.line * LINE + b * BLOCK;
        m.byteen = be;
        if (e.kind == EG_WRITE)
          m.data = blk;
        m.flags = lr.flags;
        m.flags.fence = 0;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
        m.flags.fence_cta = 0;
#endif
        auto t = get_addr_type(m.addr);
        m.flags.io = (t == AddrType::IO);
        m.flags.local = (t == AddrType::Shared);
        m.hart_id = hart;
        m.uuid = lr.uuid;
        m.tag = (b << 16) | ((e.kind == EG_WRITE) ? 0 : e.idx);
        reqs.push_back(m);
      }
      for (size_t i = 0; i + 1 < reqs.size(); ++i)
        sm_port_more(reqs[i]);
    }
    if (out.size() + reqs.size() > out.capacity())
      return;
    for (auto& m : reqs)
      out.send(m, 0);
    eg_.pop_front();
  }

  SmL1* simobject_;
  Config config_;
  SmSramCfg cfg_{};
  PerfStats perf_stats_;

  // IN / T2 / FIN
  uint32_t rr_last_ = 0;
  bool in_valid_ = false;
  Req in_r_;
  uint64_t in_todo_ = 0;
  bool in_cta_ = false;
  bool in_inv_ = false;
  bool t2_inv_ = false;
  bool hf_req_ = false;
  bool hf_sent_ = false;
  bool hf_done_ = false;
  bool fn_host_ = false;
  uint32_t seq_ctr_ = 0;
  bool t2_valid_ = false, t2_first_ = false, t2_last_ = false;
  Req t2_r_;
  uint32_t t2_bv_ = 0;
  std::array<uint32_t, TB> t2_bl_{};
  uint32_t t2_res_ = 0, t2_stale_ = 0;
  Phase t2_phase_ = P_LINES;
  uint64_t t2_emit_ = 0;
  bool inv_wait_ = false;
  std::array<Cls, LANES> acc_cls_{};
  std::array<uint32_t, LANES> acc_slot_{};
  MMask acc_wait_;
  bool fin_c_valid_ = false, fin_r_valid_ = false;
  GlItem fin_c_, fin_r_;
  MMask fin_wait_;
  bool a_valid_ = false, a_rep_ = false;
  uint32_t a_seq_ = 0;
  MMask a_wait_;
  uint32_t rq_resv_ = 0;
  std::deque<GlItem> rq_;
  std::array<Alq, ALQ> alq_{};
  uint32_t alq_n_ = 0;
  bool draw_on_ = false;
  bool gl_off_ = false, gl_off_rep_ = false;
  uint32_t gl_off_seq_ = 0;

  // fence, atomics in flight, fill, responses
  bool fn_valid_ = false, fn_got_ = false;
  uint32_t fn_blk_ = 0, fn_tag_ = 0;
  uint64_t fn_mask_ = 0;
  std::shared_ptr<LsuReq> fn_src_;
  uint32_t amo_out_ = 0;
  bool fl_valid_ = false;
  uint32_t fl_id_ = 0;
  std::array<uint8_t, LINE> fl_data_{};
  bool fill_off_ = false;
  MMask done_;
  std::array<Asm, MSHR> asm_{};
  bool rs_valid_ = false;
  uint32_t rs_blk_ = 0;
  std::deque<EgItem> eg_;

  // tags
  std::vector<TagWord> tag_mem_;
  std::vector<uint64_t> vm_mem_;
  std::array<std::deque<Upd>, TB> tq_;
  std::array<std::array<TagWord, WAYS>, TB> t_rd_{};
  std::array<std::array<uint64_t, WAYS>, TB> v_rd_{};
  std::array<std::array<uint64_t, WAYS>, TB> byp_s_{}, byp_c_{};
  std::array<std::array<bool, WAYS>, TB> byp_inv_{};
  bool inval_r_ = false;
  uint32_t inval_row_ = 0;

  // response contexts per block and LSU tag
  std::vector<std::map<uint32_t, RespCtx>> ctx_;

  // slots
  uint32_t lfsr_ = LFSR_SEED;
  uint32_t bump_ = 0, n_old_ = 0;
  bool draw_r_ = false;
  uint32_t draw_idx_ = 0;
  bool last_wen_ = false;
  uint32_t last_widx_ = 0;
  SlotRef last_wdata_{}, old_rdata_{};
  std::vector<bool> open_;
  std::vector<SlotRef> young_mem_;
  std::vector<uint32_t> free_mem_;
  std::vector<SlotRef> old_mem_;
  uint32_t y_head_ = 0, y_tail_ = 0, y_cnt_ = 0;
  uint32_t f_head_ = 0, f_tail_ = 0, f_cnt_ = 0;

  // mshr
  std::array<MshrEnt, MSHR> mshr_{};
  MMask m_valid_, m_filled_, m_store_;
  std::array<uint32_t, MSHR> m_rpc_{};
  std::array<uint8_t, MSHR> m_ilk_{};
};

///////////////////////////////////////////////////////////////////////////////

SmL1::SmL1(const SimContext& ctx, const char* name, const Config& config)
  : SimObject<SmL1>(ctx, name)
  , CfgIn(this)
  , LsuReqIn(config.num_blocks, SimChannel<LsuReq>(this, VX_CFG_LSU_PENDING_SIZE))
  , LsuRspOut(config.num_blocks, this)
  , GlReqOut(this)
  , GlRspIn(this, 1)
  , WfRetiredIn(this)
  , FillOut(this)
  , FillDoneIn(this)
  , MemReqOut(this)
  , MemRspIn(this)
  , impl_(new Impl(this, config))
{}

SmL1::~SmL1() {
  delete impl_;
}

void SmL1::on_reset() {
  impl_->reset();
}

void SmL1::on_tick() {
  impl_->tick();
}

const SmL1::PerfStats& SmL1::perf_stats() const {
  return impl_->perf_stats();
}

bool SmL1::busy() const {
  return impl_->busy();
}

void SmL1::flush_begin() {
  impl_->flush_begin();
}

bool SmL1::flush_done() const {
  return impl_->flush_done();
}

#endif // VX_CFG_H100_SM_ENABLE
