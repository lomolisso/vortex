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

#include "dlmem_egress.h"
#include <cstring>
#include <dlmem_ref.h>
#include "mem_block_pool.h"

using namespace vortex;

DlmemEgress::DlmemEgress(const SimContext& ctx, const char* name, const Config& config)
  : SimObject<DlmemEgress>(ctx, name)
  , ReqIn(config.num_blocks, SimChannel<LsuReq>(this, 1))
  , RspOut(config.num_blocks, this)
  , NetReqOut(this)
  , NetRspIn(this)
  , AckOut(this)
  , DoneOut(this)
  , config_(config)
  , entries_(config.pending_size)
  , credits_(config.store_credits)
  , locked_(false)
  , lock_block_(0)
  , rem_mask_(config.num_lanes)
  , rr_last_(config.num_blocks - 1) {
  __assert(config.num_blocks != 0 && config.num_lanes != 0, "empty egress");
  __assert(config.store_credits != 0 && config.pending_size != 0, "egress without credits or entries");
  __assert(config.core_id < config.ncc, "egress core outside its cluster");
}

void DlmemEgress::on_reset() {
  for (auto& e : entries_) {
    e = Entry();
  }
  credits_ = config_.store_credits;
  locked_ = false;
  lock_block_ = 0;
  rem_mask_.reset();
  // VX_rr_arbiter starts with its last grant at the top input.
  rr_last_ = config_.num_blocks - 1;
}

uint32_t DlmemEgress::pending() const {
  uint32_t n = 0;
  for (auto& e : entries_) {
    n += e.valid;
  }
  return n;
}

bool DlmemEgress::empty() const {
  if (locked_ || credits_ != config_.store_credits || this->pending() != 0) {
    return false;
  }
  if (NetReqOut.size() != 0 || AckOut.size() != 0 || DoneOut.size() != 0) {
    return false;
  }
  for (auto& out : RspOut) {
    if (out.size() != 0) {
      return false;
    }
  }
  return true;
}

void DlmemEgress::on_tick() {
  // Every stage reads the state registered at the start of the cycle: an entry
  // delivered now is reusable next cycle, and data or a credit returned now is
  // visible next cycle.
  std::vector<uint32_t> freed;
  this->deliver(freed);
  this->issue();
  for (auto i : freed) {
    entries_.at(i) = Entry();
  }
  this->sink();

  bool idle = (NetRspIn.size() == 0);
  for (auto& in : ReqIn) {
    idle = idle && (in.size() == 0);
  }
  for (auto& e : entries_) {
    idle = idle && !e.data_valid;
  }
  if (idle) {
    this->tick_sleep();
  }
}

// Per block, the lowest entry holding its data goes to the LSU, one lane.
void DlmemEgress::deliver(std::vector<uint32_t>& freed) {
  for (uint32_t b = 0; b < config_.num_blocks; ++b) {
    if (RspOut.at(b).full()) {
      continue;
    }
    for (uint32_t i = 0; i < entries_.size(); ++i) {
      auto& e = entries_[i];
      if (!e.valid || !e.data_valid || e.block != b) {
        continue;
      }
      LsuRsp rsp(config_.num_lanes);
      rsp.mask.set(e.lane);
      rsp.data.at(e.lane) = e.data;
      rsp.tag  = e.lsu_tag;
      rsp.cid  = e.cid;
      rsp.uuid = e.uuid;
      RspOut.at(b).send(rsp, 1);
      DT(4, this->name() << " deliver entry" << i << " block" << b << ": " << rsp);
      freed.push_back(i);
      break;
    }
  }
}

void DlmemEgress::issue() {
  uint32_t B = config_.num_blocks;
  uint32_t b = lock_block_;
  if (!locked_) {
    int32_t g = -1;
    for (uint32_t k = 1; k <= B; ++k) {
      uint32_t i = (rr_last_ + k) % B;
      if (!ReqIn.at(i).empty()) {
        g = i;
        break;
      }
    }
    if (g < 0) {
      return;
    }
    // The grant takes the lock and moves the round-robin pointer even when
    // its first lane stalls.
    b = g;
    rr_last_ = b;
  }
  auto& req = ReqIn.at(b).peek();
  __assert(req.mask.size() == config_.num_lanes, "egress: request lane count mismatch");
  BitVector<> mask = locked_ ? rem_mask_ : req.mask;
  __assert(mask.any(), "egress: request without remote lanes");
  locked_ = true;
  lock_block_ = b;
  rem_mask_ = mask;
  uint32_t l = 0;
  while (!mask.test(l)) {
    ++l;
  }
  __assert(req.dlmem_remote.test(l), "egress: lane is not marked remote");

  uint32_t dst = 0;
  uint64_t lb = 0;
  {
    dlmem_ref::Cfg cfg{};
    cfg.ncc = config_.ncc;
    cfg.lmem_log = config_.lmem_log;
    cfg.dlmem_base = config_.dlmem_base;
    uint64_t a = req.addrs.at(l);
    uint64_t window = uint64_t(config_.ncc) << config_.lmem_log;
    __assert(a >= config_.dlmem_base && a - config_.dlmem_base < window,
             "egress: remote lane outside the DLMEM window");
    dlmem_ref::decode_remote(cfg, a, &dst, &lb);
  }
  __assert(dst < config_.ncc && dst != config_.core_id, "egress: remote lane targets itself");

  bool is_st = (req.op == MemOp::ST);
  bool is_lr = (req.op == MemOp::AMO_LR);
  bool is_sc = (req.op == MemOp::AMO_SC);

  int32_t e = -1;
  if (!is_st) {
    for (uint32_t i = 0; i < entries_.size(); ++i) {
      if (!entries_[i].valid) {
        e = i;
        break;
      }
    }
    if (e < 0) {
      return;
    }
  } else if (credits_ == 0) {
    return;
  }
  if (!is_sc && NetReqOut.full()) {
    return;
  }
  // An SC yields to an arriving response so that an ack and a done never coincide.
  if (is_sc && !NetRspIn.empty()) {
    return;
  }

  if (is_lr || is_sc) {
    __assert(!config_.lrsc_assert, "remote LR/SC is not supported");
  }

  if (is_sc) {
    // A remote SC fails locally: no reservation can exist on another core.
    auto& ent = entries_.at(e);
    auto data = make_mem_block();
    std::memset(data->data(), 0, data->size());
    // The failure code 1 sits in the bytes the SC addressed, as a bank returns it.
    uint64_t byteen = req.byteen.at(l);
    __assert(byteen != 0, "egress: SC without byte enables");
    uint32_t off = __builtin_ctzll(byteen);
    uint32_t width = (__builtin_popcountll(byteen) >= 8) ? 8 : 4;
    uint64_t one = 1;
    std::memcpy(data->data() + off, &one, width);
    ent.valid = true;
    ent.data_valid = true;
    ent.block = b;
    ent.lane = l;
    ent.wid = req.wid;
    ent.lsu_tag = req.tag;
    ent.cid = req.cid;
    ent.uuid = req.uuid;
    ent.data = data;
    DoneOut.send(req.wid, 1);
  } else {
    DlmemReqPkt pkt;
    pkt.dst = dst;
    pkt.src = config_.core_id;
    pkt.op = is_lr ? MemOp::LD : req.op;
    pkt.lmem_addr = lb;
    pkt.data = (req.op == MemOp::LD || is_lr) ? nullptr : req.data.at(l);
    pkt.byteen = req.byteen.at(l);
    pkt.amo_cmp = req.amo_cmp.at(l);
    pkt.hart_id = make_hart_id(req.cid, req.wid, req.tids.at(l));
    pkt.flags.amo_unsigned = req.flags.amo_unsigned;
    pkt.uuid = req.uuid;
    if (is_st) {
      pkt.tag = req.wid;
      --credits_;
    } else {
      pkt.tag = e;
      auto& ent = entries_.at(e);
      ent.valid = true;
      ent.data_valid = false;
      ent.block = b;
      ent.lane = l;
      ent.wid = req.wid;
      ent.lsu_tag = req.tag;
      ent.cid = req.cid;
      ent.uuid = req.uuid;
      ent.data = nullptr;
    }
    NetReqOut.send(pkt, 1);
    DT(4, this->name() << " issue block" << b << " lane" << l << ": " << pkt);
  }

  rem_mask_.reset(l);
  if (rem_mask_.none()) {
    locked_ = false;
    ReqIn.at(b).pop();
  }
}

// The response sink never refuses: every response owns an entry or a credit.
void DlmemEgress::sink() {
  if (NetRspIn.empty()) {
    return;
  }
  auto& rsp = NetRspIn.peek();
  __assert(rsp.dst == config_.core_id, "egress: response for another core");
  if (rsp.ack) {
    __assert(credits_ < config_.store_credits, "egress: store credit overflow");
    ++credits_;
    AckOut.send(rsp.tag, 1);
  } else {
    __assert(rsp.tag < entries_.size(), "egress: response tag out of range");
    auto& e = entries_.at(rsp.tag);
    __assert(e.valid && !e.data_valid, "egress: response for an idle entry");
    e.data = rsp.data;
    e.data_valid = true;
    DoneOut.send(e.wid, 1);
  }
  DT(4, this->name() << " response: " << rsp);
  NetRspIn.pop();
  __assert(NetRspIn.empty(), "egress: two responses in one cycle");
}
