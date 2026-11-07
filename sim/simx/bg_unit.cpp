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

#include "bg_unit.h"
#include "debug.h"

#ifdef DXA_MBAR_ENABLE

using namespace vortex;

BgUnit::BgUnit(const SimContext& ctx, const char* name)
  : SimObject<BgUnit>(ctx, name)
  , EvtIn(this, 4)
  , ComplIn(this, 4)
  , StateOut(this)
  , UnlockOut(this)
  , warps_(VX_CFG_NUM_WARPS)
  , sent_(VX_CFG_NUM_WARPS)
{}

void BgUnit::on_reset() {
  warps_.assign(VX_CFG_NUM_WARPS, Warp{});
  sent_.resize(VX_CFG_NUM_WARPS);
  for (uint32_t w = 0; w < VX_CFG_NUM_WARPS; ++w) {
    sent_[w] = state(w, warps_[w]);
  }
}

// Every committed group but the n newest has nothing pending of the kind.
bool BgUnit::satisfied(const Warp& w, bool rd, uint32_t n) {
  if (n >= DXA_BG_GROUPS || w.ncommit <= n)
    return true;
  const uint32_t head = (w.seq - w.ncommit) & SEQ_MASK;
  for (uint32_t i = 0; i < w.ncommit - n; ++i) {
    const auto& g = w.groups[(head + i) & SEQ_MASK];
    if ((rd ? g.rd : g.wr) != 0)
      return false;
  }
  return true;
}

bool BgUnit::rd_pending(uint32_t wid) const {
  for (const auto& g : warps_.at(wid).groups) {
    if (g.rd != 0)
      return true;
  }
  return false;
}

bool BgUnit::wr_pending() const {
  for (const auto& wp : warps_) {
    for (const auto& g : wp.groups) {
      if (g.wr != 0)
        return true;
    }
  }
  return false;
}

DxaBgState BgUnit::state(uint32_t wid, const Warp& w) {
  DxaBgState s;
  s.wid          = wid;
  s.seq          = w.seq;
  s.stall_commit = (w.ncommit == DXA_BG_GROUPS);
  // The open group's slot counts at most DXA_BG_MAX_OPS operations not yet
  // performed; a full ring leaves the open group no slot of its own.
  s.stall_op     = (w.groups[w.seq].wr == DXA_BG_MAX_OPS) || s.stall_commit;
  return s;
}

void BgUnit::on_tick() {
  const uint32_t num_warps = VX_CFG_NUM_WARPS;
  bool has_evt = !EvtIn.empty();
  DxaBgEvt evt;
  if (has_evt) {
    evt = EvtIn.peek();
    EvtIn.pop();
    __assert(EvtIn.empty(), this->name() << ": two group events in one cycle");
  }

  // Waits see the registered counts: the waiting warps, then a wait issued now.
  for (uint32_t w = 0; w < num_warps; ++w) {
    auto& wp = warps_[w];
    if (wp.waiting && satisfied(wp, wp.wait_rd, wp.wait_n)) {
      wp.waiting = false;
      wp.ready = true;
    }
  }
  if (has_evt && (evt.kind == DxaBgEvt::WAIT || evt.kind == DxaBgEvt::WAIT_READ)) {
    auto& wp = warps_.at(evt.wid);
    __assert(!wp.waiting && !wp.ready, this->name() << ": second bulk wait of a warp: wid=" << evt.wid);
    const bool rd = (evt.kind == DxaBgEvt::WAIT_READ);
    DT(2, this->name() << " wait_group" << (rd ? ".read" : "") << ": wid=" << evt.wid
       << ", n=" << evt.n << ", committed=" << wp.ncommit);
    if (satisfied(wp, rd, evt.n)) {
      wp.ready = true;
    } else {
      wp.waiting = true;
      wp.wait_rd = rd;
      wp.wait_n  = evt.n;
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

  // The head retires one group a cycle, on the registered counts.
  for (uint32_t w = 0; w < num_warps; ++w) {
    auto& wp = warps_[w];
    if (wp.ncommit == 0)
      continue;
    const auto& g = wp.groups[(wp.seq - wp.ncommit) & SEQ_MASK];
    if (g.rd == 0 && g.wr == 0) {
      --wp.ncommit;
      DT(3, this->name() << " retire: wid=" << w << ", committed=" << wp.ncommit);
    }
  }

  while (!ComplIn.empty()) {
    this->apply_compl(ComplIn.peek());
    ComplIn.pop();
  }

  if (has_evt && (evt.kind == DxaBgEvt::OP || evt.kind == DxaBgEvt::COMMIT)) {
    this->apply_evt(evt);
  }

  for (uint32_t w = 0; w < num_warps; ++w) {
    auto st = state(w, warps_[w]);
    auto& prev = sent_[w];
    if (st.seq != prev.seq || st.stall_op != prev.stall_op || st.stall_commit != prev.stall_commit) {
      StateOut.send(st, 1);
      prev = st;
    }
  }

  bool idle = EvtIn.empty() && ComplIn.empty();
  for (auto& wp : warps_) {
    idle &= !wp.waiting && !wp.ready && (wp.ncommit == 0);
    for (auto& g : wp.groups) {
      idle &= (g.rd == 0) && (g.wr == 0);
    }
  }
  if (idle) {
    this->tick_sleep();
  }
}

void BgUnit::apply_evt(const DxaBgEvt& e) {
  auto& wp = warps_.at(e.wid);
  if (e.kind == DxaBgEvt::OP) {
    auto& g = wp.groups[wp.seq];
    __assert(g.wr < DXA_BG_MAX_OPS && wp.ncommit < DXA_BG_GROUPS,
             this->name() << ": bulk operation into a full group: wid=" << e.wid);
    ++g.rd;
    ++g.wr;
    DT(3, this->name() << " op: wid=" << e.wid << ", seq=" << wp.seq << ", pending=" << g.wr);
  } else {
    __assert(wp.ncommit < DXA_BG_GROUPS, this->name() << ": bulk group overflow: wid=" << e.wid);
    ++wp.ncommit;
    wp.seq = (wp.seq + 1) & SEQ_MASK;
    DT(2, this->name() << " commit_group: wid=" << e.wid << ", seq=" << wp.seq
       << ", committed=" << wp.ncommit);
  }
}

void BgUnit::apply_compl(const DxaBgCompl& c) {
  auto& wp = warps_.at(c.wid);
  auto& g = wp.groups.at(c.seq & SEQ_MASK);
  if (c.read_done) {
    __assert(g.rd != 0, this->name() << ": read_done without a pending read: " << c);
    --g.rd;
  }
  if (c.write_done) {
    __assert(g.wr != 0, this->name() << ": write_done without a pending write: " << c);
    --g.wr;
  }
  DT(3, this->name() << " compl: " << c << ", rd=" << g.rd << ", wr=" << g.wr);
}

#endif
