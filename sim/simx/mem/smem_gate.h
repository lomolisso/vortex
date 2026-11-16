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

#ifdef VX_CFG_H100_SM_ENABLE

#include "types.h"
#include "cache.h"
#include "mem_block_pool.h"
#include <smem_bank_ref.h>
#include <cstring>
#include <deque>
#include <unordered_map>
#include <stdexcept>

namespace vortex {

// A local atomic whose 8-B lanes each carry two 32-bit operations, one per half.
constexpr uint32_t SMEM_AMO_PAIR_BIT = 1u << 30;

inline bool smem_amo_pair(const MemReq& req) {
  return (req.flags.raw & SMEM_AMO_PAIR_BIT) != 0;
}

// The H100 shared-memory port of one SM, between the LSU blocks' local legs
// and the banks (VX_smem_gate.sv): one wavefront per cycle in total, every
// other local-memory client ahead of the LSU, and each read answered at the
// golden time after its issue. The banks below stay as they are; same-word
// read lanes reach them once.
class SmemGate : public SimObject<SmemGate> {
public:
  using Ptr = std::shared_ptr<SmemGate>;

  // Pre-commit buffering the gate adds, in warp requests per LSU block.
  static constexpr uint32_t REQ_STAGES = 1;
  // A load's cycles outside the hold (VX_smem_gate's): the core's from its
  // issue to the LSU's request and from the LSU's response to the dependent
  // instruction's issue, the LSU memory switch's local leg buffers, and the
  // DLMEM address translation's request register (DlmemXlate) when present.
  static constexpr uint32_t LMEM_CORE_STAGES   = 12 + OPC_ROUTE_STAGES;
  static constexpr uint32_t LMEM_SWITCH_STAGES = 2;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  static constexpr uint32_t LMEM_XLATE_STAGES  = 1;
#else
  static constexpr uint32_t LMEM_XLATE_STAGES  = 0;
#endif
  static constexpr uint32_t PIPE_OVERHEAD      = LMEM_CORE_STAGES + LMEM_SWITCH_STAGES + LMEM_XLATE_STAGES;
  // Responses held per block: a stream of one-wavefront reads, one release a
  // cycle, never fills it.
  static constexpr uint32_t RSP_DEPTH = 1u << log2ceil(VX_CFG_LMEM_LATENCY_64 + VX_CFG_LMEM_WIDE_OFFSET_64);

  std::vector<SimChannel<LsuReq>> ReqIn;
  std::vector<SimChannel<LsuReq>> ReqOut;
  std::vector<SimChannel<LsuRsp>> RspIn;
  std::vector<SimChannel<LsuRsp>> RspOut;
  // Another local-memory client first presented an access in the previous cycle.
  SimChannel<bool> ExtIn;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // Lanes a merged or paired counted access performs besides its own, so the
  // tracker's per-lane count drains with the one bank commit that serves them.
  std::vector<SimChannel<WtInc>> FoldOut;
#endif

  SmemGate(const SimContext& ctx, const char* name, uint32_t num_blocks)
    : SimObject<SmemGate>(ctx, name)
    , ReqIn(num_blocks, this)
    , ReqOut(num_blocks, this)
    , RspIn(num_blocks, this)
    , RspOut(num_blocks, this)
    , ExtIn(this)
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    , FoldOut(num_blocks, this)
#endif
    , blocks_(num_blocks)
  {
    cfg_ = {VX_CFG_LMEM_NUM_BANKS, VX_CFG_LMEM_BANK_BYTES, VX_CFG_LMEM_LATENCY, VX_CFG_LMEM_LATENCY_64,
            VX_CFG_LMEM_WAVEFRONT_PENALTY, VX_CFG_LMEM_WIDE_OFFSET_64, VX_CFG_LMEM_ATOM_EXTRA,
            VX_CFG_LMEM_CAS_EXTRA, VX_CFG_LMEM_CAS_PASSES};
    if (PIPE_OVERHEAD > VX_CFG_LMEM_LATENCY)
      throw std::runtime_error("VX_CFG_LMEM_LATENCY is below the shared-memory pipeline overhead");
    if (VX_CFG_LMEM_NUM_BANKS * VX_CFG_LMEM_BANK_BYTES != VX_CFG_SM_DATAPATH_BYTES)
      throw std::runtime_error("the shared-memory banks must span the SM datapath");
    if (LSU_CHANNELS > 32 || (LSU_WORD_SIZE % VX_CFG_LMEM_BANK_BYTES) != 0)
      throw std::runtime_error("the shared-memory port takes at most 32 lanes of whole bank words");
  }

protected:
  struct Pending {
    uint64_t    due;
    std::vector<int32_t> leader;   // per lane: the lane whose read serves it, or -1
    BitVector<> expect;            // lanes sent to the banks and not yet answered
  };

  struct Held {
    LsuRsp   rsp;
    uint64_t due;
  };

  struct Block {
    std::unordered_map<uint64_t, Pending> pending;
    std::deque<Held> held;
  };

  void on_reset() {
    debt_ = 0;
    rr_ = blocks_.size() - 1;
    for (auto& b : blocks_) {
      b.pending.clear();
      b.held.clear();
    }
  }

  void on_tick() {
    const uint64_t now = SimPlatform::instance().cycles();
    bool ext = false;
    while (!ExtIn.empty()) {
      ext = true;
      ExtIn.pop();
    }
    const uint32_t nb = blocks_.size();

    // Release the held responses whose time has come, one per block.
    for (uint32_t b = 0; b < nb; ++b) {
      auto& held = blocks_[b].held;
      if (held.empty() || held.front().due > now || RspOut[b].full())
        continue;
      DT(4, this->name() << "-rsp" << b << ": " << held.front().rsp);
      RspOut[b].send(held.front().rsp, 0);
      held.pop_front();
    }

    // Issue one warp request when the port is free.
    if (debt_ == 0 && !ext) {
      for (uint32_t k = 1; k <= nb; ++k) {
        uint32_t b = (rr_ + k) % nb;
        if (ReqIn[b].empty() || ReqOut[b].full())
          continue;
        this->issue(b, ReqIn[b].peek(), now);
        ReqIn[b].pop();
        rr_ = b;
        break;
      }
    } else if (debt_ != 0 && !ext) {
      --debt_;
    }

    // Take the banks' responses into the hold, serving merged lanes.
    for (uint32_t b = 0; b < nb; ++b) {
      auto& blk = blocks_[b];
      if (RspIn[b].empty() || blk.held.size() >= RSP_DEPTH)
        continue;
      LsuRsp rsp = RspIn[b].peek();
      RspIn[b].pop();
      auto it = blk.pending.find(rsp.tag);
      __assert(it != blk.pending.end(), this->name() << ": response without a request, tag=" << rsp.tag);
      auto& p = it->second;
      for (uint32_t j = 0; j < rsp.mask.size(); ++j) {
        if (rsp.mask.test(j))
          p.expect.reset(j);
      }
      for (uint32_t j = 0; j < p.leader.size(); ++j) {
        int32_t l = p.leader[j];
        if (l >= 0 && rsp.mask.test(l)) {
          rsp.mask.set(j);
          rsp.data[j] = rsp.data[l];
          p.leader[j] = -1;
        }
      }
      // A bank conflict below returns a request in parts: the parts before
      // the last go on at once, the last one at the request's time.
      blk.held.push_back({rsp, p.expect.none() ? p.due : now});
      if (p.expect.none()) {
        blk.pending.erase(it);
      }
    }
  }

private:
  struct LaneView {
    uint64_t addr[32];
    uint32_t mask;
    uint32_t bytes;
  };

  LaneView view(const LsuReq& req) const {
    LaneView v{};
    for (uint32_t i = 0; i < req.mask.size(); ++i) {
      if (!req.mask.test(i))
        continue;
      uint64_t en = req.byteen.at(i);
      uint32_t off = en ? (uint32_t)__builtin_ctzll(en) : 0;
      uint32_t n = en ? (uint32_t)__builtin_popcountll(en) : LSU_WORD_SIZE;
      v.addr[i] = (req.addrs.at(i) & ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1)) + off;
      v.mask |= 1u << i;
      if (n > v.bytes)
        v.bytes = n;
    }
    return v;
  }

  void issue(uint32_t b, const LsuReq& req, uint64_t now) {
    using namespace smem_bank_ref;
    const uint32_t n = req.mask.size();
    LaneView v = this->view(req);
    const bool atomic = memop_is_atomic(req.op);
#ifdef VX_CFG_EXT_MBAR_ENABLE
    const bool mbar = memop_is_mbar(req.op);
#else
    const bool mbar = false;
#endif
    uint32_t cost = 1;
    uint32_t lat = 0;
    if (atomic || mbar) {
      const bool cas = (req.op == MemOp::AMO_CAS);
      cost = smem_atom_passes(v.addr, v.mask, n, v.bytes, cas, cfg_);
      if (!mbar)
        lat = smem_atom_latency(cost, cas, cfg_);
    } else if (req.op == MemOp::LD || req.op == MemOp::ST) {
      cost = smem_wavefronts(v.addr, v.mask, n, v.bytes, cfg_);
      if (req.op == MemOp::LD)
        lat = smem_load_latency(cost, v.bytes, smem_distinct_words(v.addr, v.mask, n, v.bytes), cfg_);
    }
    if (cost == 0)
      cost = 1;
    debt_ = cost - 1;

    LsuReq out = req;
    if (req.op != MemOp::ST) {
      Pending p;
      p.due = now + ((lat > PIPE_OVERHEAD) ? (lat - PIPE_OVERHEAD) : 0);
      p.leader.assign(n, -1);
      // Lanes reading one word are served by its first lane.
      if (req.op == MemOp::LD) {
        for (uint32_t j = 0; j < n; ++j) {
          if (!out.mask.test(j))
            continue;
          for (uint32_t i = 0; i < j; ++i) {
            if (req.mask.test(i) && (req.addrs.at(i) / LSU_WORD_SIZE) == (req.addrs.at(j) / LSU_WORD_SIZE)) {
              p.leader[j] = i;
              out.mask.reset(j);
              break;
            }
          }
        }
      }
      if (atomic && this->pairable(req, v))
        this->pair_halves(req, v, out, p.leader);
      p.expect = out.mask;
      if (this->counted(req) && out.mask.count() < req.mask.count())
        this->fold(b, req.wid, req.mask.count() - out.mask.count());
      auto& pend = blocks_[b].pending;
      __assert(pend.count(req.tag) == 0, this->name() << ": tag in flight twice, tag=" << req.tag);
      pend.emplace(req.tag, std::move(p));
    }
    DT(4, this->name() << "-req" << b << ": cost=" << cost << ", lat=" << lat << ", " << out);
    ReqOut[b].send(out, 0);
  }

  // A 32-bit read-modify-write on each half of one word goes down as one
  // access on the even bank of the pair, so a bank updates a
  // lane per golden bank a cycle. The m-th lane on the low half pairs with
  // the m-th on the high half; the high lane is answered from the low one.
  bool pairable(const LsuReq& req, const LaneView& v) const {
    if (v.bytes != 4 || req.op < MemOp::AMO_SWAP || req.op > MemOp::AMO_MAX)
      return false;
#ifdef VX_CFG_EXT_MBAR_ENABLE
    if (req.flags.async)
      return false;
#endif
    return true;
  }

  void pair_halves(const LsuReq& req, const LaneView& v, LsuReq& out, std::vector<int32_t>& leader) const {
    const uint32_t n = req.mask.size();
    uint32_t taken = 0;
    bool paired = false;
    for (uint32_t j = 0; j < n; ++j) {
      if (!((v.mask >> j) & 1) || (v.addr[j] & 7) != 4)
        continue;
      for (uint32_t i = 0; i < n; ++i) {
        if (!((v.mask >> i) & 1) || ((taken >> i) & 1) || (v.addr[i] & 7) != 0 || (v.addr[i] >> 3) != (v.addr[j] >> 3))
          continue;
        taken |= 1u << i;
        auto d = make_mem_block();
        std::memcpy(d->data(), req.data.at(i)->data(), d->size());
        const uint32_t off = (uint32_t)(v.addr[j] & (VX_CFG_MEM_BLOCK_SIZE - 1));
        std::memcpy(d->data() + off, req.data.at(j)->data() + off, 4);
        out.data.at(i) = d;
        out.byteen.at(i) |= req.byteen.at(j);
        out.mask.reset(j);
        leader[j] = i;
        paired = true;
        break;
      }
    }
    if (paired)
      out.flags.raw |= SMEM_AMO_PAIR_BIT;
  }

  void fold(uint32_t b, uint32_t wid, uint32_t lanes) {
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    FoldOut.at(b).send(WtInc{wid, lanes}, 1);
#else
    (void)b, (void)wid, (void)lanes;
#endif
  }

  bool counted(const LsuReq& req) const {
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    return req.flags.lacc;
#else
    (void)req;
    return false;
#endif
  }

  smem_bank_ref::SmemCfg cfg_;
  std::vector<Block> blocks_;
  uint32_t debt_ = 0;
  uint32_t rr_ = 0;

  friend class SimObject<SmemGate>;
};

}

#endif
