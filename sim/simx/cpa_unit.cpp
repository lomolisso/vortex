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

#include "cpa_unit.h"
#include <cstring>
#include "debug.h"
#include "mem_block_pool.h"

using namespace vortex;

CpaStatus CpaStatus::idle() {
  CpaStatus s;
  s.free_slots = (CpaUnit::SLOTS == 64) ? ~uint64_t(0) : ((uint64_t(1) << CpaUnit::SLOTS) - 1);
  return s;
}

CpaUnit::CpaUnit(const SimContext& ctx, const char* name, uint32_t block)
  : SimObject<CpaUnit>(ctx, name)
  , IssueIn(this)
  , RspIn(this)
  , ReqOut(this)
  , AckIn(this)
  , StatusOut(this)
  , UnlockOut(this)
  , block_(block)
  , slots_(SLOTS)
  , warps_(VX_CFG_NUM_WARPS)
{}

void CpaUnit::on_reset() {
  slots_.assign(SLOTS, Slot{});
  warps_.assign(VX_CFG_NUM_WARPS, Warp{});
  sent_status_ = CpaStatus::idle();
  held_.valid = false;
  arr_inflight_ = 0;
}

bool CpaUnit::idle(uint32_t wid) const {
  const auto& wp = warps_.at(wid);
  return wp.fifo.empty() && !wp.arrive && arr_inflight_ == 0;
}

// The boundaries that still have an entry before them at or past `head`.
uint32_t CpaUnit::beyond(const Warp& w, uint32_t head) {
  uint32_t n = 0;
  for (auto b : w.bounds) {
    n += int32_t(b - head) > 0;
  }
  return n;
}

bool CpaUnit::satisfied(const Warp& w, uint32_t head, uint32_t n) const {
  return n >= GROUPS || beyond(w, head) <= n;
}

void CpaUnit::on_tick() {
  const uint32_t num_warps = VX_CFG_NUM_WARPS;
  bool has_evt = !IssueIn.empty();
  CpaIssue evt;
  if (has_evt) {
    evt = IssueIn.peek();
    IssueIn.pop();
    __assert(IssueIn.empty(), this->name() << ": two issue events in one cycle");
  }
  std::vector<uint32_t> head_r(num_warps);
  for (uint32_t w = 0; w < num_warps; ++w) {
    head_r[w] = warps_[w].head;
  }

  // Waits see the registered head: the waiting warps, then a wait issued now.
  for (uint32_t w = 0; w < num_warps; ++w) {
    auto& wp = warps_[w];
    if (wp.waiting && this->satisfied(wp, head_r[w], wp.wait_n)) {
      wp.waiting = false;
      wp.ready = true;
    }
  }
  if (has_evt && evt.kind == CpaIssue::WAIT) {
    auto& wp = warps_.at(evt.wid);
    __assert(!wp.waiting && !wp.ready, this->name() << ": second cp.async wait of a warp: wid=" << evt.wid);
    DT(2, this->name() << " wait_group: wid=" << evt.wid << ", n=" << evt.n
       << ", pending=" << beyond(wp, head_r[evt.wid]));
    if (this->satisfied(wp, head_r[evt.wid], evt.n)) {
      wp.ready = true;
    } else {
      wp.waiting = true;
      wp.wait_n = evt.n;
    }
  }

  // One release a cycle, lowest warp first: the pulse register, then the
  // scheduler's stall register.
  for (uint32_t w = 0; w < num_warps; ++w) {
    if (warps_[w].ready) {
      warps_[w].ready = false;
      DT(2, this->name() << " wait-unlock: wid=" << w);
      UnlockOut.send(w, 2);
      break;
    }
  }

  // The head pops one entry a cycle, on the registered counts.
  for (uint32_t w = 0; w < num_warps; ++w) {
    auto& wp = warps_[w];
    if (!wp.fifo.empty() && wp.fifo.front().closed && wp.fifo.front().count == 0) {
      wp.fifo.pop_front();
      ++wp.head;
      while (!wp.bounds.empty() && int32_t(wp.bounds.front() - wp.head) <= 0) {
        wp.bounds.pop_front();
      }
    }
  }

  while (!AckIn.empty()) {
    this->ack(AckIn.peek());
    AckIn.pop();
  }

  // One request a cycle: an eligible arrive first, then landed lanes. The
  // adapter takes a request once all its lanes are out, so the next one is
  // chosen only when the last has left it.
  if (ReqOut.size() == 0 && (this->select_arrive(head_r) || this->select_write())) {
    this->accept_held();
  }

  if (has_evt && evt.kind != CpaIssue::WAIT) {
    this->apply_issue(evt, head_r);
  }

  // Responses land in their cycle; they leave from the next.
  while (!RspIn.empty()) {
    this->land(RspIn.peek());
    RspIn.pop();
  }

  auto st = this->status();
  if (st != sent_status_) {
    StatusOut.send(st, 1);
    sent_status_ = st;
  }

  bool idle = (IssueIn.size() == 0) && (RspIn.size() == 0) && (AckIn.size() == 0);
  for (auto& s : slots_) {
    idle &= !s.valid;
  }
  for (auto& wp : warps_) {
    idle &= wp.fifo.empty() && !wp.waiting && !wp.ready && !wp.arrive;
  }
  idle &= !held_.valid && (arr_inflight_ == 0);
  if (idle) {
    this->tick_sleep();
  }
}

void CpaUnit::apply_issue(const CpaIssue& e, const std::vector<uint32_t>& head_r) {
  __unused(head_r);
  auto& wp = warps_.at(e.wid);
  switch (e.kind) {
  case CpaIssue::COPY: {
    auto& s = slots_.at(e.slot);
    if (e.sop) {
      __assert(!s.valid, this->name() << ": copy into a busy slot " << e.slot);
      __assert(wp.fifo.size() < DEPTH, this->name() << ": cp.async FIFO overflow: wid=" << e.wid);
      s.valid = true;
      s.wid   = e.wid;
      s.seq   = wp.tail;
      s.size  = e.size;
      s.uuid  = e.uuid;
      s.lanes.assign(NUM_PIDS * VX_CFG_NUM_LSU_LANES, Slot::Lane{});
      wp.fifo.push_back(Entry{});
      ++wp.tail;
    }
    __assert(s.valid && s.wid == e.wid && !wp.fifo.empty(),
             this->name() << ": copy packet without its open slot: wid=" << e.wid << ", slot=" << e.slot);
    auto& entry = wp.fifo.at(s.seq - wp.head);
    uint32_t count = 0;
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      if (!((e.tmask >> t) & 1))
        continue;
      s.lanes[t].active = true;
      s.lanes[t].info = e.lanes.at(t);
      ++count;
    }
    __assert(count != 0, this->name() << ": copy without a lane");
    entry.count += count;
    entry.closed = e.eop;
    DT(2, this->name() << " copy: wid=" << e.wid << ", slot=" << e.slot << ", seq=" << s.seq
       << ", lanes=" << count << ", sop=" << e.sop << ", eop=" << e.eop << " (#" << e.uuid << ")");
  } break;
  case CpaIssue::COMMIT:
    __assert(beyond(wp, wp.head) < GROUPS, this->name() << ": cp.async group overflow: wid=" << e.wid);
    wp.bounds.push_back(wp.tail);
    DT(2, this->name() << " commit_group: wid=" << e.wid << ", tail=" << wp.tail);
    break;
  case CpaIssue::ARRIVE:
    __assert(!wp.arrive, this->name() << ": second pending cp.async arrive: wid=" << e.wid);
    wp.arrive       = true;
    wp.noinc        = e.noinc;
    wp.arrive_tail  = wp.tail;
    wp.arrive_pid   = 0;
    wp.arrive_mask  = e.tmask;
    wp.arrive_uuid  = e.uuid;
    wp.arrive_addrs = e.addrs;
    DT(2, this->name() << " arrive: wid=" << e.wid << ", tail=" << wp.tail << ", noinc=" << e.noinc);
    break;
  default:
    std::abort();
  }
}

// Presents the landed, unsent lanes of the lowest (slot, pid) as one write packet.
bool CpaUnit::select_write() {
  constexpr uint32_t L = VX_CFG_NUM_LSU_LANES;
  for (uint32_t si = 0; si < SLOTS; ++si) {
    auto& s = slots_[si];
    if (!s.valid)
      continue;
    for (uint32_t pid = 0; pid < NUM_PIDS; ++pid) {
      LsuReq req(L);
      uint64_t lanes = 0;
      for (uint32_t i = 0; i < L; ++i) {
        auto& lane = s.lanes[pid * L + i];
        if (!lane.landed || lane.sent)
          continue;
        const uint64_t dst = lane.info.dst;
        const uint32_t off = uint32_t(dst & (VX_CFG_MEM_BLOCK_SIZE - 1));
        auto block = make_mem_block();
        std::memset(block->data(), 0, block->size());
        for (uint32_t k = 0; k < s.size; ++k) {
          (*block)[off + k] = uint8_t(lane.data >> (8 * k));
        }
        req.mask.set(i);
        req.addrs.at(i)  = dst;
        req.data.at(i)   = block;
        req.byteen.at(i) = ((uint64_t(1) << s.size) - 1) << off;
        lanes |= uint64_t(1) << i;
      }
      if (lanes == 0)
        continue;
      req.op = MemOp::ST;
      req.flags.strsp = 1;
      req.tag = si;
      req.uuid = s.uuid;
      held_.valid  = true;
      held_.arrive = false;
      held_.slot   = si;
      held_.pid    = pid;
      held_.lanes  = lanes;
      held_.req    = req;
      return true;
    }
  }
  return false;
}

// Presents the next packet of the lowest warp's arrive whose tail the head passed.
bool CpaUnit::select_arrive(const std::vector<uint32_t>& head_r) {
  constexpr uint32_t L = VX_CFG_NUM_LSU_LANES;
  for (uint32_t w = 0; w < warps_.size(); ++w) {
    auto& wp = warps_[w];
    if (!wp.arrive || int32_t(wp.arrive_tail - head_r[w]) > 0)
      continue;
    auto pid_lanes = [&](uint32_t pid) {
      return (wp.arrive_mask >> (pid * L)) & ((L >= 64) ? ~uint64_t(0) : ((uint64_t(1) << L) - 1));
    };
    while (wp.arrive_pid < NUM_PIDS && pid_lanes(wp.arrive_pid) == 0) {
      ++wp.arrive_pid;
    }
    __assert(wp.arrive_pid < NUM_PIDS, this->name() << ": arrive without a lane: wid=" << w);
    const uint32_t pid = wp.arrive_pid;
    LsuReq req(L);
    req.op = wp.noinc ? MemOp::MBAR_ARRIVE : MemOp::MBAR_COMPLETE_TX;
    for (uint32_t i = 0; i < L; ++i) {
      if (!((pid_lanes(pid) >> i) & 1))
        continue;
      const uint64_t a = wp.arrive_addrs.at(pid * L + i);
      const uint32_t off = uint32_t(a & (VX_CFG_MEM_BLOCK_SIZE - 1));
      auto block = make_mem_block();
      std::memset(block->data(), 0, block->size());
      (*block)[off] = 1;
      req.mask.set(i);
      req.addrs.at(i)  = a;
      req.data.at(i)   = block;
      req.byteen.at(i) = uint64_t(0xff) << off;
    }
    req.tag = ARRIVE_TAG;
    req.uuid = wp.arrive_uuid;
    held_.valid  = true;
    held_.arrive = true;
    held_.wid    = w;
    held_.pid    = pid;
    held_.req    = req;
    return true;
  }
  return false;
}

// The port takes the held request: only its lanes leave the landed set.
void CpaUnit::accept_held() {
  constexpr uint32_t L = VX_CFG_NUM_LSU_LANES;
  if (held_.arrive) {
    auto& wp = warps_.at(held_.wid);
    DT(3, this->name() << " arrive-req: wid=" << held_.wid << ", " << held_.req);
    wp.arrive_pid = held_.pid + 1;
    const uint64_t lane_mask = (L >= 64) ? ~uint64_t(0) : ((uint64_t(1) << L) - 1);
    while (wp.arrive_pid < NUM_PIDS && ((wp.arrive_mask >> (wp.arrive_pid * L)) & lane_mask) == 0) {
      ++wp.arrive_pid;
    }
    if (wp.arrive_pid >= NUM_PIDS) {
      wp.arrive = false;
    }
    arr_inflight_ += held_.req.mask.count();
  } else {
    auto& s = slots_.at(held_.slot);
    for (uint32_t i = 0; i < L; ++i) {
      if ((held_.lanes >> i) & 1) {
        s.lanes.at(held_.pid * L + i).sent = true;
      }
    }
    DT(3, this->name() << " write: slot=" << held_.slot << ", pid=" << held_.pid << ", " << held_.req);
  }
  ReqOut.send(held_.req, 0);
  held_.valid = false;
}

void CpaUnit::land(const LsuRsp& rsp) {
  constexpr uint32_t L = VX_CFG_NUM_LSU_LANES;
  const uint32_t si = uint32_t(rsp.tag) / NUM_PIDS;
  const uint32_t pid = uint32_t(rsp.tag) % NUM_PIDS;
  __assert(si < SLOTS && slots_[si].valid, this->name() << ": response for a free slot, tag=" << rsp.tag);
  auto& s = slots_[si];
  for (uint32_t i = 0; i < L; ++i) {
    if (!rsp.mask.test(i))
      continue;
    auto& lane = s.lanes.at(pid * L + i);
    __assert(lane.active && !lane.landed, this->name() << ": stray response lane " << i << ", tag=" << rsp.tag);
    lane.landed = true;
    lane.data = 0;
    const uint8_t* src = rsp.data.at(i)->data() + lane.info.src_off;
    for (uint32_t k = 0; k < lane.info.ss; ++k) {
      lane.data |= uint64_t(src[k]) << (8 * k);
    }
  }
  DT(3, this->name() << " land: slot=" << si << ", pid=" << pid << ", " << rsp);
}

void CpaUnit::ack(const LsuRsp& rsp) {
  if (rsp.tag == ARRIVE_TAG) {
    __assert(arr_inflight_ >= rsp.mask.count(), this->name() << ": arrive response underflow");
    arr_inflight_ -= rsp.mask.count();
    return;
  }
  const uint32_t si = uint32_t(rsp.tag);
  __assert(si < SLOTS && slots_[si].valid, this->name() << ": ack for a free slot " << si);
  auto& s = slots_[si];
  auto& wp = warps_.at(s.wid);
  auto& entry = wp.fifo.at(s.seq - wp.head);
  const uint32_t n = rsp.mask.count();
  __assert(entry.count >= n, this->name() << ": too many acks for slot " << si);
  entry.count -= n;
  if (entry.count == 0 && entry.closed) {
    s.valid = false;
    DT(2, this->name() << " done: wid=" << s.wid << ", slot=" << si << ", seq=" << s.seq);
  }
}

CpaStatus CpaUnit::status() const {
  CpaStatus st;
  for (uint32_t w = 0; w < warps_.size(); ++w) {
    auto& wp = warps_[w];
    st.fifo_full[w]      = wp.fifo.size() >= DEPTH;
    st.groups_full[w]    = beyond(wp, wp.head) >= GROUPS;
    st.arrive_pending[w] = wp.arrive;
  }
  for (uint32_t si = 0; si < SLOTS; ++si) {
    if (!slots_[si].valid) {
      st.free_slots |= uint64_t(1) << si;
    }
  }
  return st;
}
