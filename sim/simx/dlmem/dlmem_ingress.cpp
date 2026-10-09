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

using namespace vortex;

DlmemIngress::DlmemIngress(const SimContext& ctx, const char* name, const Config& config)
  : SimObject<DlmemIngress>(ctx, name)
  , NetReqIn(this)
  , LmemReqOut(this)
  , LmemRspIn(this)
  , NetRspOut(this)
  , config_(config)
  , slots_(config.num_slots) {
  __assert(config.num_slots != 0, "ingress without slots");
}

void DlmemIngress::on_reset() {
  for (auto& s : slots_) {
    s = Slot();
  }
  done_.clear();
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
      && NetRspOut.size() == 0;
}

void DlmemIngress::on_tick() {
  // All three stages read the state registered at the start of the cycle: a
  // slot freed by the injection is reusable next cycle, and a completion
  // pushed by the local-memory response is injected next cycle.

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
      LmemReqOut.send(req, 1);
      DT(4, this->name() << " accept slot" << s << ": " << pkt);
      NetReqIn.pop();
    }
  }

  // Inject the oldest completion; its slot frees with the push.
  if (!done_.empty() && !NetRspOut.full()) {
    uint32_t s = done_.front();
    auto& slot = slots_.at(s);
    DlmemRspPkt rsp;
    rsp.dst  = slot.src;
    rsp.ack  = slot.ack;
    rsp.tag  = slot.tag;
    rsp.data = slot.ack ? nullptr : slot.data;
    rsp.uuid = slot.uuid;
    NetRspOut.send(rsp, 1);
    DT(4, this->name() << " inject slot" << s << ": " << rsp);
    slot = Slot();
    done_.pop_front();
  }

  // Local-memory responses are never refused: their slot is already held.
  if (!LmemRspIn.empty()) {
    auto& rsp = LmemRspIn.peek();
    uint32_t s = uint32_t(rsp.tag);
    __assert(s < slots_.size() && slots_[s].busy, "ingress: response for a free slot");
    slots_[s].data = rsp.data;
    done_.push_back(s);
    __assert(done_.size() <= slots_.size(), "ingress: completion FIFO overflow");
    LmemRspIn.pop();
    __assert(LmemRspIn.empty(), "ingress: two local-memory responses in one cycle");
  }

  if (NetReqIn.size() == 0 && LmemRspIn.size() == 0 && done_.empty()) {
    this->tick_sleep();
  }
}
