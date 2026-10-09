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

#include "dlmem_xlate.h"
#include <algorithm>
#include <iostream>

using namespace vortex;

// Every kind keeps the address's 64-byte line offset, which per-lane data and byteen are relative to.
static_assert(VX_CFG_MEM_BLOCK_SIZE == 64, "DLMEM requires a 64-byte memory block");

DlmemXlate::DlmemXlate(const SimContext& ctx,
                       const char* name,
                       const dlmem_ref::Cfg& cfg,
                       uint32_t core_id,
                       uint32_t num_blocks,
                       bool abort_on_oob,
                       bool abort_on_remote)
  : SimObject<DlmemXlate>(ctx, name)
  , ReqIn(num_blocks, SimChannel<LsuReq>(this, 2))
  , ReqOut(num_blocks, SimChannel<LsuReq>(this))
  , PeerWrIn(this)
  , WtIncOut(num_blocks, SimChannel<WtInc>(this))
  , cfg_(cfg)
  , my_core_(0)
  , num_blocks_(num_blocks)
  , abort_on_oob_(abort_on_oob)
  , abort_on_remote_(abort_on_remote)
  , rows_(cfg.nw, std::make_shared<const dlmem_ref::Row>(dlmem_ref::Row{0, {}}))
  , entry_rows_(num_blocks)
  , stride_(0) {
  __assert(num_blocks != 0, "no LSU block");
  __assert(cfg.ncc != 0 && cfg.nw != 0, "empty DLMEM cluster shape");
  __assert(cfg.max_cluster != 0, "DLMEM max cluster is zero");
  my_core_ = core_id % cfg.ncc;
  // Delivery into ReqIn is the RTL's entry into its skid buffer, where the row is captured.
  for (uint32_t b = 0; b < num_blocks; ++b) {
    ReqIn.at(b).tx_callback([this, b](const LsuReq& req, uint64_t /*cycle*/) {
      __assert(req.cta_id < cfg_.nw, "cta_id " << req.cta_id << " out of range");
      entry_rows_.at(b).push_back(rows_.at(req.cta_id));
    });
  }
}

bool DlmemXlate::empty() const {
  for (auto& in : ReqIn) {
    if (in.size() != 0) {
      return false;
    }
  }
  return true;
}

void DlmemXlate::on_reset() {
  // Every window access is out of bounds until the dispatcher writes a row.
  auto empty = std::make_shared<const dlmem_ref::Row>(dlmem_ref::Row{0, {}});
  std::fill(rows_.begin(), rows_.end(), empty);
  for (auto& rows : entry_rows_) {
    rows.clear();
  }
  stride_ = 0;
}

void DlmemXlate::on_tick() {
  // A row written last cycle is captured by the requests entering after this tick.
  while (!PeerWrIn.empty()) {
    auto& wr = PeerWrIn.peek();
    __assert(wr.slot < cfg_.nw, "peer row for slot " << wr.slot << " out of range");
    __assert(wr.row.k <= cfg_.max_cluster && wr.row.peers.size() == wr.row.k,
             "malformed peer row: k=" << wr.row.k << " peers=" << wr.row.peers.size());
    for (auto& p : wr.row.peers) {
      __assert(p.core < cfg_.ncc && p.slot < cfg_.nw,
               "peer {core=" << p.core << ", slot=" << p.slot << "} out of range");
    }
    // The low line-offset bits must survive translation untouched.
    __assert((wr.stride % VX_CFG_MEM_BLOCK_SIZE) == 0
          && uint64_t(wr.stride) <= (uint64_t(1) << cfg_.lmem_log),
             "bad DLMEM slot stride " << wr.stride);
    rows_.at(wr.slot) = std::make_shared<const dlmem_ref::Row>(wr.row);
    stride_ = wr.stride;
    PeerWrIn.pop();
  }

  // size() also counts packets still in flight toward a channel.
  bool busy = (PeerWrIn.size() != 0);
  for (uint32_t b = 0; b < num_blocks_; ++b) {
    auto& in = ReqIn.at(b);
    auto& out = ReqOut.at(b);
    if (!in.empty() && !out.full()) {
      auto& entry_rows = entry_rows_.at(b);
      __assert(!entry_rows.empty(), "block " << b << " request without its entry row");
      LsuReq req = in.peek();
      uint32_t remote = this->translate(req, *entry_rows.front());
      // AMOs return data, so only plain stores need a remote-ack count.
      if (remote != 0 && req.op == MemOp::ST) {
        WtIncOut.at(b).send(WtInc{req.wid, remote}, 1);
      }
      DT(3, this->name() << " block" << b << " xlate: " << req);
      out.send(std::move(req), 1);
      in.pop();
      entry_rows.pop_front();
    }
    busy |= (in.size() != 0);
  }
  if (!busy) {
    this->tick_sleep();
  }
}

uint32_t DlmemXlate::translate(LsuReq& req, const dlmem_ref::Row& row) {
  uint32_t remote = 0;
  int oob_lane = -1;
  uint64_t oob_addr = 0;
  int remote_lane = -1;
  uint64_t remote_addr = 0;
  uint32_t remote_core = 0;
  for (uint32_t i = 0; i < req.mask.size(); ++i) {
    req.dlmem_remote.reset(i);
    req.dlmem_oob.reset(i);
    req.dlmem_core.at(i) = 0;
    // A fence token's lanes carry no address.
    if (!req.mask.test(i) || req.flags.fence)
      continue;
    uint64_t addr = req.addrs.at(i);
    auto x = dlmem_ref::translate(cfg_, addr, row, my_core_, req.cta_id, stride_);
    if (x.kind == dlmem_ref::NONE)
      continue;
    req.addrs.at(i) = x.out_addr;
    req.dlmem_core.at(i) = x.core;
    if (x.kind == dlmem_ref::REMOTE) {
      req.dlmem_remote.set(i);
      ++remote;
      if (remote_lane < 0) {
        remote_lane = i;
        remote_addr = addr & ~uint64_t(LSU_WORD_SIZE - 1);
        remote_core = x.core;
      }
    } else if (x.kind == dlmem_ref::OOB) {
      req.dlmem_oob.set(i);
      if (oob_lane < 0) {
        oob_lane = i;
        // The RTL holds word addresses, so its message shows the word.
        oob_addr = addr & ~uint64_t(LSU_WORD_SIZE - 1);
      }
    }
  }
  if (oob_lane >= 0 && abort_on_oob_) {
    std::cerr << SimPlatform::instance().cycles() << ": " << this->name()
              << " DLMEM access out of bounds: wid=" << req.wid << " cta=" << req.cta_id
              << " lane=" << oob_lane << " addr=0x" << std::hex << oob_addr << std::dec
              << " (#" << req.uuid << ")" << std::endl;
    std::abort();
  }
  if (remote_lane >= 0 && abort_on_remote_) {
    std::cerr << SimPlatform::instance().cycles() << ": " << this->name()
              << " remote DLMEM access is not supported yet: wid=" << req.wid << " cta=" << req.cta_id
              << " lane=" << remote_lane << " addr=0x" << std::hex << remote_addr << std::dec
              << " core=" << remote_core << " (#" << req.uuid << ")" << std::endl;
    std::abort();
  }
  return remote;
}
