// Copyright © 2019-2026
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

#include "dlmem_ingress.h"
#ifdef VX_CFG_EXT_MBAR_ENABLE
#include <cstring>
#include "mem_block_pool.h"
#else
#line 14
#endif

using namespace vortex;

DlmemIngress::DlmemIngress(const SimContext& ctx, const char* name, const Config& config)
  : SimObject<DlmemIngress>(ctx, name)
  , NetReqIn(this)
  , LmemReqOut(this)
  , LmemRspIn(this)
  , NetRspOut(this)
#ifdef DXA_MBAR_ENABLE
  , LmemDmaOut(this)
  , SlotIn(this, 4 * VX_CFG_NUM_WARPS + 8)
#endif
  , config_(config)
  , slots_(config.num_slots)
#ifdef DXA_MBAR_ENABLE
  , slot_valid_(VX_CFG_NUM_WARPS, false)
#endif
{
  __assert(config.num_slots != 0, "ingress without slots");
}

void DlmemIngress::on_reset() {
  for (auto& s : slots_) {
    s = Slot();
  }
  done_.clear();
#ifdef DXA_MBAR_ENABLE
  std::fill(slot_valid_.begin(), slot_valid_.end(), false);
#endif
}

uint32_t DlmemIngress::busy_slots() const {
  uint32_t n = 0;
  for (auto& s : slots_) {
    n += s.busy;
  }
  return n;
}

bool DlmemIngress::empty() const {
  return this->busy_slots() == 0
      && LmemReqOut.size() == 0
#ifdef DXA_MBAR_ENABLE
      && LmemDmaOut.size() == 0
#endif
      && NetRspOut.size() == 0;
}

#ifdef DXA_MBAR_ENABLE
void DlmemIngress::check_slot(const DlmemReqPkt& pkt) const {
  if (pkt.tag < slot_valid_.size() && slot_valid_.at(pkt.tag))
    return;
  std::cerr << SimPlatform::instance().cycles() << ": *** " << this->name()
            << " DXA write into a retired CTA slot: " << pkt << std::endl;
  std::abort();
}

// The RTL ingress hands a DATA word straight to the local memory's DMA port,
// whose acceptance is the bank commit, and takes nothing behind it until then.
// Here the word leaves in the cycle it is taken, and the next flit waits until
// the local memory's crossbar has moved it into its bank's queue, ahead of
// any later access to that bank; the crossbar holds it while an atomic owns
// the bank, and holds the COMPL's complete_tx until the words are written.
bool DlmemIngress::accept_dxa(const DlmemReqPkt& pkt) {
  const bool compl_flit = pkt.flags.amo_unsigned;
  if (LmemDmaOut.size() != 0)
    return false;
  if (!compl_flit) {
    this->check_slot(pkt);
    MemReq req(MemOp::ST, config_.lmem_base + pkt.lmem_addr, pkt.data, pkt.byteen, 0, 0, pkt.uuid);
    req.flags.dma = 1;
    req.dxa.remote = true;
    req.dxa.slot   = pkt.tag;
    LmemDmaOut.send(req, 0);
    DT(3, this->name() << " dxa data in: " << pkt);
    return true;
  }
  if (LmemReqOut.full() || this->cx_valid())
    return false;
  int32_t s = -1;
  for (uint32_t i = 0; i < slots_.size(); ++i) {
    if (!slots_[i].busy) {
      s = i;
      break;
    }
  }
  if (s < 0)
    return false;
  this->check_slot(pkt);
  auto& slot = slots_[s];
  slot = Slot();
  slot.busy   = true;
  slot.posted = true;
  slot.uuid   = pkt.uuid;
  MemReq req(MemOp::MBAR_COMPLETE_TX, config_.lmem_base + pkt.lmem_addr, pkt.data, pkt.byteen,
             uint32_t(s), 0, pkt.uuid);
  req.flags.local = 1;
  req.dxa.remote  = true;
  LmemReqOut.send(req, 1);
  DT(3, this->name() << " dxa compl in, slot" << s << ": " << pkt);
  return true;
}
#endif

void DlmemIngress::on_tick() {
  // All three stages read the state registered at the start of the cycle: a
  // slot freed by the injection is reusable next cycle, and a completion
  // pushed by the local-memory response is injected next cycle.
#ifdef DXA_MBAR_ENABLE
  while (!SlotIn.empty()) {
    const auto& e = SlotIn.peek();
    if (e.kind == CtaSlotEvt::ADMIT) {
      slot_valid_.at(e.slot) = true;
    } else if (e.kind == CtaSlotEvt::FREE) {
      slot_valid_.at(e.slot) = false;
    }
    SlotIn.pop();
  }
  if (!NetReqIn.empty() && NetReqIn.peek().dxa) {
    if (this->accept_dxa(NetReqIn.peek())) {
      NetReqIn.pop();
    }
  } else
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
  // The head completion's complete_tx takes the request buffer first.
  if (this->cx_valid()) {
    // a new request waits
  } else
#endif
#line 141
  // Accept a request into the lowest free slot, its response space reserved.
  if (!NetReqIn.empty() && !LmemReqOut.full()) {
    int32_t s = -1;
    for (uint32_t i = 0; i < slots_.size(); ++i) {
      if (!slots_[i].busy) {
        s = i;
        break;
      }
    }
    if (s >= 0) {
      auto& pkt = NetReqIn.peek();
      auto& slot = slots_[s];
      slot.busy = true;
      slot.src  = pkt.src;
      slot.tag  = pkt.tag;
      slot.ack  = (pkt.op == MemOp::ST);
      slot.data = nullptr;
      slot.uuid = pkt.uuid;
      MemReq req(pkt.op, config_.lmem_base + pkt.lmem_addr, pkt.data, pkt.byteen,
                 uint32_t(s), pkt.hart_id, pkt.uuid);
      req.amo_cmp = pkt.amo_cmp;
      req.flags = pkt.flags;
      req.flags.local = 1;
      // The store's response is its bank-commit ack.
      req.flags.strsp = slot.ack;
    #ifdef VX_CFG_EXT_MBAR_ENABLE
      if (pkt.mbar_valid) {
        slot.mbar      = true;
        slot.mbar_addr = pkt.mbar_addr;
        slot.bytes     = __builtin_popcountll(pkt.byteen);
        // A red.async is acked as a store: it returns nothing.
        slot.ack       = true;
        req.flags.strsp = (pkt.op == MemOp::ST);
      }
    #else
    #line 165
    #endif
      LmemReqOut.send(req, 1);
      DT(4, this->name() << " accept slot" << s << ": " << pkt);
      NetReqIn.pop();
    }
  }

  // Inject the oldest completion; its slot frees with the push.
#ifdef VX_CFG_EXT_MBAR_ENABLE
  // An async slot sends its complete_tx, then its ack, which may leave in
  // the same cycle.
  bool cx_ok = true;
  if (this->cx_valid()) {
    cx_ok = false;
    if (!LmemReqOut.full()) {
      const uint32_t s = done_.front();
      auto req = this->complete_tx(s);
      LmemReqOut.send(req, 1);
      DT(4, this->name() << " complete_tx slot" << s << ": " << req);
      slots_.at(s).cx_sent = true;
      cx_ok = true;
    }
  }
#endif
#ifdef DXA_MBAR_ENABLE
  // A COMPL's complete_tx sends nothing: its slot frees in turn.
  if (!done_.empty() && slots_.at(done_.front()).posted) {
    DT(4, this->name() << " retire posted slot" << done_.front());
    slots_.at(done_.front()) = Slot();
    done_.pop_front();
  } else
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
  if (cx_ok && !done_.empty() && !NetRspOut.full()) {
#else
#line 181
  if (!done_.empty() && !NetRspOut.full()) {
#endif
    uint32_t s = done_.front();
    auto& slot = slots_.at(s);
    DlmemRspPkt rsp;
    rsp.dst  = slot.src;
    rsp.ack  = slot.ack;
    rsp.tag  = slot.tag;
    rsp.data = slot.ack ? nullptr : slot.data;
    rsp.uuid = slot.uuid;
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    rsp.mbar = slot.mbar;
  #else
  #line 189
  #endif
    NetRspOut.send(rsp, 1);
    DT(4, this->name() << " inject slot" << s << ": " << rsp);
    slot = Slot();
    done_.pop_front();
  }

  // Local-memory responses are never refused: their slot is already held.
  if (!LmemRspIn.empty()) {
    auto& rsp = LmemRspIn.peek();
    uint32_t s = uint32_t(rsp.tag);
  #ifdef VX_CFG_EXT_MBAR_ENABLE
    // An async access's complete_tx answers too, past the slot tags; it is dropped.
    if (s >= slots_.size()) {
      LmemRspIn.pop();
      __assert(LmemRspIn.empty(), "ingress: two local-memory responses in one cycle");
      return;
    }
  #else
  #line 199
  #endif
    __assert(s < slots_.size() && slots_[s].busy, "ingress: response for a free slot");
    slots_[s].data = rsp.data;
    done_.push_back(s);
    __assert(done_.size() <= slots_.size(), "ingress: completion FIFO overflow");
    LmemRspIn.pop();
    __assert(LmemRspIn.empty(), "ingress: two local-memory responses in one cycle");
  }

#ifdef DXA_MBAR_ENABLE
  if (SlotIn.size() != 0 || LmemDmaOut.size() != 0) {
    return;
  }
#endif
  if (NetReqIn.size() == 0 && LmemRspIn.size() == 0 && done_.empty()) {
    this->tick_sleep();
  }
}

#ifdef VX_CFG_EXT_MBAR_ENABLE
// The completion FIFO's head is an async access whose complete_tx has not left.
bool DlmemIngress::cx_valid() const {
  if (done_.empty())
    return false;
  const auto& slot = slots_.at(done_.front());
  return slot.mbar && !slot.cx_sent;
}

// Its response, tagged past the slots, is dropped.
MemReq DlmemIngress::complete_tx(uint32_t s) const {
  const auto& slot = slots_.at(s);
  const uint32_t off = uint32_t(slot.mbar_addr & (VX_CFG_MEM_BLOCK_SIZE - 1));
  auto blk = make_mem_block();
  std::memset(blk->data(), 0, blk->size());
  const uint64_t bytes = slot.bytes;
  std::memcpy(blk->data() + off, &bytes, sizeof(bytes));
  MemReq req(MemOp::MBAR_COMPLETE_TX, config_.lmem_base + slot.mbar_addr, blk, uint64_t(0xff) << off,
             uint32_t(slots_.size() + s), 0, slot.uuid);
  req.flags.local = 1;
  return req;
}
#endif
