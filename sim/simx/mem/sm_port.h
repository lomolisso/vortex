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

#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <deque>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "types.h"
#include "sm_port_ref.h"
#ifdef VX_CFG_EXT_DLMEM_ENABLE
#include "dlmem/dlmem_types.h"
#include "dlmem/dlmem_switch.h"
#endif

namespace vortex {

// A MemReq holds one memory block, so a source sends a line request as its
// touched blocks, lowest first, every one but the last marked here. MemFlags
// gives this bit to no field; the port clears it before the memory.
constexpr uint32_t SM_PORT_MORE_BIT = 1u << 31;

inline void sm_port_more(MemReq& req) {
  req.flags.raw |= SM_PORT_MORE_BIT;
}

inline bool sm_port_has_more(const MemReq& req) {
  return (req.flags.raw & SM_PORT_MORE_BIT) != 0;
}

inline sm_port_ref::SmPortCfg sm_port_cfg() {
  sm_port_ref::SmPortCfg c{};
  c.egress_bytes     = VX_CFG_SM_EGRESS_BYTES;
  c.beat_bytes       = VX_CFG_SM_EGRESS_BEAT;
  c.hb_wr_sector     = VX_CFG_SM_HB_WRITE_SECTOR;
  c.hb_wr_masked     = VX_CFG_SM_HB_WRITE_MASKED;
  c.hb_rd_line       = VX_CFG_SM_HB_READ_LINE;
  c.hb_pf_line       = VX_CFG_SM_HB_PREFETCH_LINE;
  c.hb_atom_lane     = VX_CFG_SM_HB_ATOM_LANE;
  c.hb_dsmem_slot    = VX_CFG_SM_HB_DSMEM_SLOT;
  c.hb_dsmem_ld_line = VX_CFG_SM_HB_DSMEM_LD_LINE;
  c.hb_tma_st_sector = VX_CFG_SM_HB_TMA_ST_SECTOR;
  c.hb_mbar          = VX_CFG_SM_HB_MBAR;
  c.min_slots        = VX_CFG_N5_MIN_SLOTS;
  c.atom_extra_slots = VX_CFG_DSMEM_ATOM_EXTRA_SLOTS;
  c.inj_credits      = VX_CFG_SM2SM_INJ_CREDITS;
  c.inj_return       = VX_CFG_SM2SM_INJ_RETURN;
  c.eject_bytes      = VX_CFG_SM2SM_EJECT_BYTES;
  c.ingress_bytes    = VX_CFG_SM_INGRESS_BYTES;
  c.w_fill           = VX_CFG_SM_INGRESS_FILL_WEIGHT;
  c.w_dsmem          = VX_CFG_SM_INGRESS_DSMEM_WEIGHT;
  c.issue_w_pm       = VX_CFG_SM_SECTOR_ISSUE_W_PERMILLE;
  c.issue_r_pm       = VX_CFG_SM_SECTOR_ISSUE_R_PERMILLE;
  c.req_fifo_bytes   = VX_CFG_SM_REQ_FIFO_BYTES;
  c.rsp_slots        = VX_CFG_SM_RSP_QUEUE_SLOTS;
  return c;
}

// VX_sm_port + VX_sm_mem_adapter: one SM's ports to the GPC, its decisions
// those of sm_port_ref's SmEgressRef and SmIngressRef. Request-class items
// (the L1's and the TMA engine's line requests, this SM's SM-to-SM packets)
// enter one in-order FIFO, one per cycle round robin over the sources; the
// replies this SM serves wait in the response queue. A paid L2-side item
// passes the output register, then leaves as one memory block per cycle; its
// reply blocks are gathered into one line, pass the input register and the
// ingress gate with the SM-to-SM arrivals.
//
// The memory tag carries what the reply needs on the way back: the source,
// the blocks sent and the sectors the reply brings into the SM.
class SmPort : public SimObject<SmPort> {
public:
  using Ptr = std::shared_ptr<SmPort>;

  enum { SRC_L1 = 0, SRC_TMA = 1, MAX_SRCS = 2 };

  struct Config {
    uint32_t num_srcs;
    uint32_t line_bytes;
  };

  std::vector<SimChannel<MemReq>> SrcReqIn;
  std::vector<SimChannel<MemRsp>> SrcRspOut;
  SimChannel<MemReq> MemReqOut;
  SimChannel<MemRsp> MemRspIn;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  SimChannel<DlmemReqPkt> DsmReqIn;    // this SM's SM-to-SM requests
  SimChannel<DlmemReqPkt> DsmReqOut;
  SimChannel<DlmemRspPkt> DsmRspIn;    // the replies this SM serves
  SimChannel<DlmemRspPkt> DsmRspOut;
  SimChannel<DlmemReqPkt> ArrReqIn;    // requests arriving for this SM
  SimChannel<DlmemReqPkt> ArrReqOut;
  SimChannel<DlmemRspPkt> ArrRspIn;    // replies arriving for this SM
  SimChannel<DlmemRspPkt> ArrRspOut;
#endif

  SmPort(const SimContext& ctx, const char* name, const Config& config)
    : SimObject<SmPort>(ctx, name)
    , SrcReqIn(config.num_srcs, this)
    , SrcRspOut(config.num_srcs, this)
    , MemReqOut(this)
    , MemRspIn(this)
  #ifdef VX_CFG_EXT_DLMEM_ENABLE
    , DsmReqIn(this)
    , DsmReqOut(this)
    , DsmRspIn(this)
    , DsmRspOut(this)
    , ArrReqIn(this)
    , ArrReqOut(this)
    , ArrRspIn(this)
    , ArrRspOut(this)
  #endif
    , config_(config)
    , cfg_(sm_port_cfg())
    , egress_(cfg_)
    , ingress_(cfg_)
    , sectors_(config.line_bytes / cfg_.beat_bytes)
    , blocks_(config.line_bytes / VX_CFG_MEM_BLOCK_SIZE) {
    if (config.num_srcs < 1 || config.num_srcs > MAX_SRCS)
      throw std::runtime_error("sm port: one L1 and at most one TMA source");
    if (cfg_.beat_bytes != VX_CFG_DCACHE_SECTOR_SIZE)
      throw std::runtime_error("sm port: the egress beat must be the L1 sector");
    if ((cfg_.egress_bytes % cfg_.beat_bytes) != 0 || (cfg_.ingress_bytes % cfg_.beat_bytes) != 0)
      throw std::runtime_error("sm port: the egress and ingress must be whole beats");
    if ((config.line_bytes % VX_CFG_MEM_BLOCK_SIZE) != 0 || (config.line_bytes % cfg_.beat_bytes) != 0)
      throw std::runtime_error("sm port: the line must be whole memory blocks and sectors");
    if (cfg_.inj_credits < 1 || cfg_.inj_credits > cfg_.inj_return)
      throw std::runtime_error("sm port: injection credits must be 1..return");
    if (cfg_.eject_bytes > cfg_.ingress_bytes)
      throw std::runtime_error("sm port: ejection above the ingress");
    if (blocks_ > (1u << TAG_BLK_BITS))
      throw std::runtime_error("sm port: the line has more memory blocks than the memory tag holds");
  }

  // Replies still awaited from the memory are not this port's work.
  bool empty() const {
    for (auto& s : staged_) {
      if (!s.empty())
        return false;
    }
    // A packet already waiting in an input sends no wake-up, so a port that
    // slept on it would never take it.
    for (auto& in : SrcReqIn) {
      if (!in.empty())
        return false;
    }
    return egress_.empty() && req_items_.empty() && rsp_items_.empty() && out_reg_.empty()
        && in_reg_.empty() && MemReqOut.size() == 0 && MemRspIn.empty()
      #ifdef VX_CFG_EXT_DLMEM_ENABLE
        && DsmReqOut.size() == 0 && DsmRspOut.size() == 0
        && ArrReqOut.size() == 0 && ArrRspOut.size() == 0
        && DsmReqIn.empty() && DsmRspIn.empty() && ArrReqIn.empty() && ArrRspIn.empty()
      #endif
        ;
  }

protected:
  void on_reset() {
    egress_.reset();
    ingress_.reset();
    for (auto& s : staged_)
      s.clear();
    req_items_.clear();
    rsp_items_.clear();
    out_reg_.clear();
    pend_.clear();
    pend_free_.clear();
    in_reg_.clear();
    adapter_blk_ = 0;
    rr_last_ = 0;
    idle_ = 0;
  }

  void on_tick() {
    uint64_t now = SimPlatform::instance().cycles();
    this->collect_blocks(now);
    this->admit(now);
    this->drain(now);
    this->adapter_send(now);
    this->gather_replies(now);
    this->ingress_pass(now);
    if (this->empty() && egress_.credits() == cfg_.inj_credits) {
      // Credits and issue tokens refill within inj_return cycles of idling.
      if (++idle_ > cfg_.inj_return)
        this->tick_sleep();
    } else {
      idle_ = 0;
    }
  }

private:
  enum { DSM = MAX_SRCS };
  static constexpr uint32_t TAG_BLK_BITS = 2;
  static constexpr uint32_t OUT_DEPTH = sm_port_ref::SM_PORT_REG_DEPTH;
  static constexpr uint32_t IN_DEPTH  = sm_port_ref::SM_PORT_REG_DEPTH;

  struct Staged {
    MemReq req;
    uint64_t arrival;
  };

  struct Item {
    uint32_t src = 0;
    std::vector<MemReq> blocks;
    uint32_t rsp_sect = 0;
    bool expects = false;
  #ifdef VX_CFG_EXT_DLMEM_ENABLE
    DlmemReqPkt dsm;
  #endif
  };

  struct RegItem {
    uint64_t valid_at;
    Item item;
  };

  // A line item awaiting its reply blocks.
  struct Pending {
    uint32_t src = 0;
    uint32_t sect = 0;
    uint32_t got = 0;
    std::vector<MemRsp> blocks;
  };

  struct Reply {
    uint64_t valid_at;
    uint32_t src;
    uint32_t sectors;
    std::vector<MemRsp> blocks;
  };

  void collect_blocks(uint64_t now) {
    for (uint32_t s = 0; s < SrcReqIn.size(); ++s) {
      auto& in = SrcReqIn.at(s);
      while (!in.empty() && staged_[s].size() < blocks_ + 1) {
        DT(3, this->name() << " stage src" << s << " more=" << sm_port_has_more(in.peek()) << ": " << in.peek());
        staged_[s].push_back(Staged{in.peek(), now});
        in.pop();
      }
    }
  }

  uint32_t block_of(const MemReq& r) const {
    return uint32_t(r.addr % config_.line_bytes) / VX_CFG_MEM_BLOCK_SIZE;
  }

  static bool plain(const MemReq& r) {
    return (r.op == MemOp::LD || r.op == MemOp::ST) && !r.flags.fence;
  }

  // Blocks of the source's head line item, 0 while it is incomplete: the
  // marked blocks and the one after them, as long as they share the line and
  // the operation. The L1 sends a line as its blocks from its block banks,
  // the higher one a cycle later at most, so an unmarked plain L1 request and
  // the next block of its line with the same operation form one item too.
  uint32_t item_blocks(uint32_t s, uint64_t now) const {
    auto& st = staged_[s];
    if (st.empty())
      return 0;
    const MemReq& h = st[0].req;
    auto same = [&](const MemReq& p) {
      return p.op == h.op && p.flags.fence == h.flags.fence
          && (p.addr / config_.line_bytes) == (h.addr / config_.line_bytes);
    };
    if (sm_port_has_more(h)) {
      uint32_t n = 1;
      while (n < st.size() && same(st[n].req)) {
        bool more = sm_port_has_more(st[n].req);
        ++n;
        if (!more)
          return n;
      }
      if (n < st.size() || n == blocks_)
        return n;
      // A source sends a line's blocks back to back, so the rest of a marked
      // line is never more than a block per cycle behind the one before it.
      if (SrcReqIn.at(s).empty() && now > st.back().arrival + 2 * blocks_) {
        std::ostringstream os;
        os << this->name() << ": marked block " << h << " lost the rest of its line";
        throw std::runtime_error(os.str());
      }
      return 0;
    }
    if (s != SRC_L1 || !plain(h) || block_of(h) + 1 == blocks_)
      return 1;
    if (st.size() >= 2) {
      const MemReq& p = st[1].req;
      return (plain(p) && same(p) && !sm_port_has_more(p) && block_of(p) > block_of(h)) ? 2 : 1;
    }
    return (now > st[0].arrival) ? 1 : 0;
  }

  Item src_item(uint32_t s, uint32_t n, sm_port_ref::SmEgressItem* eit) const {
    Item it;
    it.src = s;
    std::vector<uint8_t> be(config_.line_bytes, 0);
    for (uint32_t k = 0; k < n; ++k) {
      const MemReq& b = staged_[s][k].req;
      it.blocks.push_back(b);
      uint32_t off = block_of(b) * VX_CFG_MEM_BLOCK_SIZE;
      for (uint32_t i = 0; i < VX_CFG_MEM_BLOCK_SIZE; ++i)
        be[off + i] = (b.byteen >> i) & 1;
    }
    const MemReq& h = it.blocks.front();
    uint32_t touched = sm_port_ref::touched_sectors(be.data(), config_.line_bytes, cfg_.beat_bytes);
    bool is_amo = memop_is_amo(h.op);
    bool is_rd = !h.is_write() && !h.flags.fence && !is_amo;
    bool is_wr = h.is_write() && !is_amo;
    it.rsp_sect = is_rd ? (touched ? touched : sectors_) : (is_amo ? 1 : 0);
    it.expects = (h.op != MemOp::ST) || h.flags.fence || h.flags.strsp;
    eit->slots = 0;
    eit->l2 = true;
    eit->single = 0;
    if (h.flags.fence) {
      eit->hb = cfg_.hb_rd_line;
    } else if (is_amo) {
      eit->hb = sm_port_ref::sm_item_halfbeats(sm_port_ref::SmItem::AtomicLanes, be.data(), config_.line_bytes, 1, 0, cfg_);
    } else if (is_wr) {
      auto kind = (s == SRC_L1) ? sm_port_ref::SmItem::WriteLine : sm_port_ref::SmItem::TmaStoreLine;
      eit->hb = sm_port_ref::sm_item_halfbeats(kind, be.data(), config_.line_bytes, 0, 0, cfg_);
      if (s == SRC_L1 && touched == 1)
        eit->single = 1;
    } else {
      eit->hb = sm_port_ref::sm_item_halfbeats(sm_port_ref::SmItem::ReadLine, be.data(), config_.line_bytes, 0, 0, cfg_);
      if (s == SRC_L1 && touched == 1)
        eit->single = 2;
    }
    return it;
  }

  // One push per cycle into the request FIFO, round robin over the admissible
  // sources; the response queue takes one reply per cycle.
  void admit(uint64_t now) {
    uint32_t nsrc = SrcReqIn.size();
  #ifdef VX_CFG_EXT_DLMEM_ENABLE
    uint32_t nin = nsrc + 1;
  #else
    uint32_t nin = nsrc;
  #endif
    for (uint32_t k = 1; k <= nin; ++k) {
      uint32_t j = (rr_last_ + k) % nin;
      sm_port_ref::SmEgressItem eit;
      Item it;
      uint32_t nblk = 0;
      if (j < nsrc) {
        nblk = this->item_blocks(j, now);
        if (nblk == 0)
          continue;
        it = this->src_item(j, nblk, &eit);
      } else {
      #ifdef VX_CFG_EXT_DLMEM_ENABLE
        if (DsmReqIn.empty())
          continue;
        it.src = DSM;
        it.dsm = DsmReqIn.peek();
        eit.hb = it.dsm.pkt_hb;
        eit.slots = (it.dsm.pkt_hb + 1) / 2;
        eit.l2 = false;
        eit.single = 0;
      #endif
      }
      if (!egress_.req_can_push(eit))
        continue;
      egress_.req_push(eit);
      DT(3, this->name() << " push src" << j << " hb=" << eit.hb << " slots=" << eit.slots << " blocks=" << nblk);
      if (j < nsrc) {
        staged_[j].erase(staged_[j].begin(), staged_[j].begin() + nblk);
      } else {
      #ifdef VX_CFG_EXT_DLMEM_ENABLE
        DsmReqIn.pop();
      #endif
      }
      req_items_.push_back(std::move(it));
      rr_last_ = j;
      break;
    }
  #ifdef VX_CFG_EXT_DLMEM_ENABLE
    if (!DsmRspIn.empty()) {
      auto& rsp = DsmRspIn.peek();
      sm_port_ref::SmEgressItem eit;
      eit.hb = rsp.pkt_hb;
      eit.slots = (eit.hb + 1) / 2;
      eit.l2 = false;
      if (egress_.rsp_can_push(eit)) {
        egress_.rsp_push(eit);
        rsp_items_.push_back(rsp);
        DsmRspIn.pop();
      }
    }
  #endif
  }

  template <typename T>
  static bool room(const SimChannel<T>& ch, uint32_t n) {
    return ch.size() + n <= std::min<uint32_t>(ch.capacity(), OUT_DEPTH);
  }

  void drain(uint64_t now) {
    bool l2_ready = out_reg_.size() < OUT_DEPTH;
  #ifdef VX_CFG_EXT_DLMEM_ENABLE
    bool dsm_ready = room(DsmReqOut, 1);
    bool rsp_ready = room(DsmRspOut, 1);
  #else
    bool dsm_ready = false;
    bool rsp_ready = false;
  #endif
    auto pops = egress_.tick(l2_ready, dsm_ready, rsp_ready);
    if (pops.req) {
      Item it = std::move(req_items_.front());
      req_items_.pop_front();
      if (pops.req_item.l2) {
        out_reg_.push_back(RegItem{now + sm_port_ref::SM_PORT_OUT_STAGES, std::move(it)});
      } else {
      #ifdef VX_CFG_EXT_DLMEM_ENABLE
        DsmReqOut.send(it.dsm, 1);
      #endif
      }
    }
  #ifdef VX_CFG_EXT_DLMEM_ENABLE
    if (pops.rsp) {
      DsmRspOut.send(rsp_items_.front(), 1);
      rsp_items_.pop_front();
    }
  #else
    __unused(pops);
  #endif
  }

  // One memory block per cycle from the output register's head line;
  // an item that answers holds a pending entry, named by the blocks' tags.
  void adapter_send(uint64_t now) {
    if (out_reg_.empty() || out_reg_.front().valid_at > now)
      return;
    if (MemReqOut.full())
      return;
    auto& d = out_reg_.front();
    uint32_t nblk = d.item.blocks.size();
    if (adapter_blk_ == 0 && d.item.expects) {
      if (pend_free_.empty()) {
        pend_free_.push_back(pend_.size());
        pend_.emplace_back();
      }
      adapter_idx_ = pend_free_.back();
      pend_free_.pop_back();
      auto& p = pend_.at(adapter_idx_);
      p.src = d.item.src;
      p.sect = d.item.rsp_sect;
      p.got = 0;
      p.blocks = std::vector<MemRsp>(nblk);
      tags_.resize(pend_.size());
      tags_.at(adapter_idx_).clear();
      for (auto& b : d.item.blocks)
        tags_.at(adapter_idx_).push_back(b.tag);
    }
    MemReq req = d.item.blocks.at(adapter_blk_);
    req.flags.raw &= ~SM_PORT_MORE_BIT;
    if (d.item.expects)
      req.tag = (adapter_idx_ << TAG_BLK_BITS) | adapter_blk_;
    DT(3, this->name() << " mem-req blk" << adapter_blk_ << "/" << nblk << ": " << req);
    MemReqOut.send(req, 1);
    if (++adapter_blk_ == nblk) {
      adapter_blk_ = 0;
      out_reg_.pop_front();
    }
  }

  // The input register takes replies while it has room, as the RTL's in_buf
  // drives the L2's response ready.
  void gather_replies(uint64_t now) {
    while (!MemRspIn.empty() && in_reg_.size() < IN_DEPTH) {
      MemRsp rsp = MemRspIn.peek();
      MemRspIn.pop();
      uint32_t idx = uint32_t(rsp.tag >> TAG_BLK_BITS);
      uint32_t blk = uint32_t(rsp.tag & ((1u << TAG_BLK_BITS) - 1));
      DT(3, this->name() << " mem-rsp: " << rsp);
      if (idx >= pend_.size() || blk >= pend_.at(idx).blocks.size()) {
        std::cerr << SimPlatform::instance().cycles() << ": *** " << this->name()
                  << " reply for no request: " << rsp << std::endl;
        std::abort();
      }
      auto& p = pend_.at(idx);
      rsp.tag = tags_.at(idx).at(blk);
      p.blocks.at(blk) = rsp;
      if (++p.got == p.blocks.size()) {
        in_reg_.push_back(Reply{now + sm_port_ref::SM_PORT_IN_STAGES, p.src, p.sect, std::move(p.blocks)});
        p.blocks.clear();
        pend_free_.push_back(idx);
      }
    }
  }

  void ingress_pass(uint64_t now) {
    using G = sm_port_ref::SmIngressRef;
    bool valid[G::NSRC] = {false, false, false};
    uint32_t sectors[G::NSRC] = {0, 0, 0};
    bool ready[G::NSRC] = {false, false, false};
    bool pass[G::NSRC];
    if (!in_reg_.empty() && in_reg_.front().valid_at <= now) {
      auto& r = in_reg_.front();
      valid[G::FILL] = true;
      sectors[G::FILL] = r.sectors;
      ready[G::FILL] = room(SrcRspOut.at(r.src), r.blocks.size());
    }
  #ifdef VX_CFG_EXT_DLMEM_ENABLE
    if (!ArrReqIn.empty()) {
      valid[G::DSM_REQ] = true;
      sectors[G::DSM_REQ] = (ArrReqIn.peek().pkt_hb + 1) / 2;
      ready[G::DSM_REQ] = room(ArrReqOut, 1);
    }
    if (!ArrRspIn.empty()) {
      valid[G::DSM_RSP] = true;
      sectors[G::DSM_RSP] = (ArrRspIn.peek().pkt_hb + 1) / 2;
      ready[G::DSM_RSP] = room(ArrRspOut, 1);
    }
  #endif
    ingress_.tick(valid, sectors, ready, pass);
    if (pass[G::FILL]) {
      auto& r = in_reg_.front();
      for (auto& b : r.blocks)
        SrcRspOut.at(r.src).send(b, 1);
      in_reg_.pop_front();
    }
  #ifdef VX_CFG_EXT_DLMEM_ENABLE
    if (pass[G::DSM_REQ]) {
      ArrReqOut.send(ArrReqIn.peek(), 1);
      ArrReqIn.pop();
    }
    if (pass[G::DSM_RSP]) {
      ArrRspOut.send(ArrRspIn.peek(), 1);
      ArrRspIn.pop();
    }
  #endif
  }

  Config config_;
  sm_port_ref::SmPortCfg cfg_;
  sm_port_ref::SmEgressRef egress_;
  sm_port_ref::SmIngressRef ingress_;
  uint32_t sectors_;
  uint32_t blocks_;
  std::deque<Staged> staged_[MAX_SRCS];
  std::deque<Item> req_items_;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  std::deque<DlmemRspPkt> rsp_items_;
#else
  std::deque<int> rsp_items_;
#endif
  std::deque<RegItem> out_reg_;
  std::vector<Pending> pend_;
  std::vector<std::vector<uint32_t>> tags_;
  std::vector<uint32_t> pend_free_;
  std::deque<Reply> in_reg_;
  uint32_t adapter_blk_ = 0;
  uint32_t adapter_idx_ = 0;
  uint32_t rr_last_ = 0;
  uint32_t idle_ = 0;

  friend class SimObject<SmPort>;
};

// The TMA engine's memory port: each request goes to the port of the SM its
// tag names (the engine's worker index, which is the SM), each reply back.
class SmTmaSplit : public SimObject<SmTmaSplit> {
public:
  using Ptr = std::shared_ptr<SmTmaSplit>;

  SimChannel<MemReq> ReqIn;
  SimChannel<MemRsp> RspOut;
  std::vector<SimChannel<MemReq>> ReqOut;
  std::vector<SimChannel<MemRsp>> RspIn;

  SmTmaSplit(const SimContext& ctx, const char* name, uint32_t num_sms)
    : SimObject<SmTmaSplit>(ctx, name)
    , ReqIn(this)
    , RspOut(this)
    , ReqOut(num_sms, this)
    , RspIn(num_sms, this)
    , mask_(num_sms - 1) {
    if (num_sms == 0 || (num_sms & (num_sms - 1)) != 0)
      throw std::runtime_error("sm tma split: the SM count must be a power of two");
  }

protected:
  void on_reset() {
    rr_ = 0;
  }

  // As the RTL's bus switch: a request waits for room at its SM's port, and
  // one reply per cycle passes, round robin over the SMs.
  void on_tick() {
    if (!ReqIn.empty()) {
      auto& req = ReqIn.peek();
      if (ReqOut.at(req.tag & mask_).try_send(req, 1))
        ReqIn.pop();
    }
    if (RspOut.full())
      return;
    uint32_t n = RspIn.size();
    for (uint32_t k = 1; k <= n; ++k) {
      uint32_t i = (rr_ + k) % n;
      if (!RspIn.at(i).empty()) {
        RspOut.send(RspIn.at(i).peek(), 1);
        RspIn.at(i).pop();
        rr_ = i;
        break;
      }
    }
  }

private:
  uint32_t mask_;
  uint32_t rr_ = 0;

  friend class SimObject<SmTmaSplit>;
};

} // namespace vortex
