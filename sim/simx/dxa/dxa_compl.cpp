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

#include "dxa_compl.h"
#include <VX_types.h>
#include <cstring>
#include <iostream>
#include "mem_block_pool.h"
#include "debug.h"

using namespace vortex;

DxaCompl::DxaCompl(const SimContext& ctx, const char* name, uint32_t num_banks)
  : SimObject<DxaCompl>(ctx, name)
  , WrIn(this, 2 * num_banks)
  , TxbarOut(this)
  , FreeOut(this)
  , ReleaseIn(this, VX_CFG_NUM_WARPS * VX_CFG_NUM_DXA_CORES)
#ifdef DXA_MBAR_ENABLE
  , ReqOut(this)
  , RspIn(this)
  , SlotIn(this, 4 * VX_CFG_NUM_WARPS + 8)
  , slot_valid_(VX_CFG_NUM_WARPS, false)
  , slot_gen_(VX_CFG_NUM_WARPS, 0)
  , slot_stride_(0)
  , sent_(false)
#endif
  , landing_(0)
  , free_sent_(UINT32_MAX)
{}

DxaCompl::~DxaCompl() {}

void DxaCompl::on_reset() {
  fifo_.clear();
  xfers_.clear();
  landing_ = 0;
  free_sent_ = UINT32_MAX;
#ifdef DXA_MBAR_ENABLE
  std::fill(slot_valid_.begin(), slot_valid_.end(), false);
  std::fill(slot_gen_.begin(), slot_gen_.end(), 0);
  slot_stride_ = 0;
  sent_ = false;
#endif
}

bool DxaCompl::entry_is_mbar(const Entry& e) const {
#ifdef DXA_MBAR_ENABLE
  return e.tag.mode;
#else
  __unused(e);
  return false;
#endif
}

// The RTL completion unit queues a last beat in the cycle the bank takes it,
// and presents the queue head from the next cycle: a barrier event to the
// barrier unit, or a complete_tx at the local memory's input, where it waits
// for its round-robin grant like an LSU request. It pops one entry per cycle.
void DxaCompl::on_tick() {
  const uint64_t now = SimPlatform::instance().cycles();

  while (!ReleaseIn.empty()) {
    const uint32_t bar = bar_decode_id(ReleaseIn.peek(), VX_CFG_NUM_BARRIERS);
    DT(2, this->name() << " release without a write: bar=" << bar);
    TxbarOut.send(bar, 1);
    ReleaseIn.pop();
  }

  bool popped = false;
#ifdef DXA_MBAR_ENABLE
  while (!RspIn.empty()) {
    RspIn.pop();
  }
  while (!SlotIn.empty()) {
    const auto& e = SlotIn.peek();
    switch (e.kind) {
    case CtaSlotEvt::ADMIT:
      slot_valid_.at(e.slot) = true;
      slot_gen_.at(e.slot)   = e.gen;
      slot_stride_           = e.stride;
      break;
    case CtaSlotEvt::FREE:
      slot_valid_.at(e.slot) = false;
      break;
    default:
      break;
    }
    SlotIn.pop();
  }

  // The local memory's input took the head's complete_tx this cycle.
  if (sent_ && ReqOut.size() == 0) {
    this->check_complete_tx(fifo_.front());
    DT(2, this->name() << " complete_tx taken (#" << fifo_.front().uuid << ")");
    fifo_.pop_front();
    sent_ = false;
    popped = true;
  }
#endif

  if (!popped && !fifo_.empty() && fifo_.front().push < now && !this->entry_is_mbar(fifo_.front())) {
    const auto& e = fifo_.front();
    uint32_t bar = bar_decode_id(e.bar_id, VX_CFG_NUM_BARRIERS);
    DT(2, this->name() << " release: bar=" << bar << " (#" << e.uuid << ")");
    TxbarOut.send(bar, 1);
    fifo_.pop_front();
  }

  while (!WrIn.empty()) {
    const auto& c = WrIn.peek();
  #ifdef DXA_MBAR_ENABLE
    this->check_live(c.tag.slot, c.tag.gen, c.uuid, "write");
  #endif
    auto& x = xfers_[c.tag.xfer];
    ++x.committed;
    if (c.notify) {
      x.writes = c.tag.writes;
      x.bar_id = c.bar_id;
      ++landing_;
    }
    if (x.writes != 0 && x.committed == x.writes) {
      __assert(fifo_.size() < DEPTH, this->name() << ": completion queue overflow (#" << c.uuid << ")");
      fifo_.push_back({c.tag, x.bar_id, c.uuid, now});
      DT(2, this->name() << " copy landed: writes=" << x.writes << " (#" << c.uuid << ")");
      xfers_.erase(c.tag.xfer);
      --landing_;
    }
    WrIn.pop();
  }

#ifdef DXA_MBAR_ENABLE
  if (!sent_ && !fifo_.empty() && fifo_.front().tag.mode && ReqOut.size() == 0) {
    this->send_complete_tx(fifo_.front());
    sent_ = true;
  }
#endif

  this->send_free();
}

// A copy whose final write has committed holds its entry already. The head's
// barrier release leaves next cycle for sure; its complete_tx only if it wins
// its bank, which the local memory's input sees for itself.
void DxaCompl::send_free() {
  uint32_t used = uint32_t(fifo_.size()) + landing_;
  if (!fifo_.empty() && !this->entry_is_mbar(fifo_.front())) {
    --used;
  }
  const uint32_t free = (used < DEPTH) ? (DEPTH - used) : 0;
  if (free != free_sent_) {
    FreeOut.send(free, 1);
    free_sent_ = free;
  }
}

#ifdef DXA_MBAR_ENABLE
void DxaCompl::check_live(uint32_t slot, uint32_t gen, uint64_t uuid, const char* what) const {
  if (slot_valid_.at(slot) && slot_gen_.at(slot) == gen)
    return;
  std::cerr << SimPlatform::instance().cycles() << ": *** " << this->name()
            << " DXA write into a retired CTA slot: " << what << " slot=" << slot << " gen=" << gen
            << " (slot valid=" << slot_valid_.at(slot) << " gen=" << slot_gen_.at(slot) << ")"
            << " (#" << uuid << ")" << std::endl;
  std::abort();
}

// Checked when the local memory's input takes the request, as the RTL does.
void DxaCompl::check_complete_tx(const Entry& e) const {
  this->check_live(e.tag.slot, e.tag.gen, e.uuid, "complete_tx");
  const uint64_t lo   = uint64_t(e.tag.slot) * slot_stride_;
  const uint64_t hi   = lo + slot_stride_;
  const uint64_t addr = uint64_t(e.tag.mbar_off) << 3;
  if (addr < lo || addr >= hi) {
    std::cerr << SimPlatform::instance().cycles() << ": *** " << this->name()
              << " mbarrier offset outside the block: byte offset 0x" << std::hex << addr
              << std::dec << ", slot=" << e.tag.slot << " block=[0x" << std::hex << lo
              << ", 0x" << hi << ")" << std::dec << " (#" << e.uuid << ")" << std::endl;
    std::abort();
  }
}

void DxaCompl::send_complete_tx(const Entry& e) {
  const uint64_t addr = uint64_t(e.tag.mbar_off) << 3;
  const uint32_t off = uint32_t(addr & (VX_CFG_MEM_BLOCK_SIZE - 1));
  auto data = make_mem_block();
  std::memset(data->data(), 0, data->size());
  const uint64_t bytes = e.tag.tx_bytes;
  std::memcpy(data->data() + off, &bytes, sizeof(bytes));

  MemReq req;
  req.op      = MemOp::MBAR_COMPLETE_TX;
  req.addr    = VX_MEM_LMEM_BASE_ADDR + addr;
  req.data    = data;
  req.byteen  = uint64_t(0xff) << off;
  req.tag     = 0;
  req.hart_id = 0;
  req.uuid    = e.uuid;
  ReqOut.send(req, 1);
  DT(2, this->name() << " complete_tx: off=0x" << std::hex << e.tag.mbar_off << std::dec
     << ", bytes=" << e.tag.tx_bytes << ", slot=" << e.tag.slot << " (#" << e.uuid << ")");
}
#endif
