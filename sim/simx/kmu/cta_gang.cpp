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

#include "cta_gang.h"
#include <algorithm>
#include <VX_config.h>
#include <VX_types.h>
#include "debug.h"

using namespace vortex;
using dlmem_ref::check;

dlmem_ref::Cfg vortex::dlmem_cfg() {
  dlmem_ref::Cfg cfg;
  cfg.ncc         = NUM_SOCKETS * VX_CFG_SOCKET_SIZE;
  cfg.nw          = VX_CFG_NUM_WARPS;
  cfg.nt          = VX_CFG_NUM_THREADS;
  cfg.max_cluster = VX_CFG_DLMEM_MAX_CLUSTER;
  cfg.lmem_log    = VX_CFG_LMEM_LOG_SIZE;
  cfg.num_gangs   = std::max<uint32_t>(2, (cfg.ncc * cfg.nw) / 2);
  cfg.socket_size = VX_CFG_SOCKET_SIZE;
  cfg.num_sockets = NUM_SOCKETS;
  cfg.dlmem_base  = VX_MEM_DLMEM_BASE_ADDR;
  cfg.lmem_base   = VX_MEM_LMEM_BASE_ADDR;
  return cfg;
}

namespace {

uint32_t cluster_size(const kmu_req_t& req) {
  uint32_t k = req.cluster_dim[0] * req.cluster_dim[1] * req.cluster_dim[2];
  return (k == 0) ? 1 : k;
}

// The dispatcher's slot pitch: the footprint rounded up to a memory block.
uint32_t slot_stride(const kmu_req_t& req) {
  return (req.lmem_size + VX_CFG_MEM_BLOCK_SIZE - 1u) & ~uint32_t(VX_CFG_MEM_BLOCK_SIZE - 1u);
}

} // namespace

CtaGang::CtaGang(const SimContext& ctx, const char* name, const dlmem_ref::Cfg& cfg)
  : SimObject<CtaGang>(ctx, name)
  // Two entries keep a pass-through stream at one beat per cycle behind the
  // lane's credit loop.
  , bus_in(this, 2)
  , credit_out(this)
  , bus_out(cfg.ncc, this)
  , credit_in(cfg.ncc, this)
  , cfg_(cfg)
  , blocks_(0)
  , state_(State::IDLE)
  , step_(0)
  , fail_step_(0)
  , cur_gang_(0)
  , cur_epoch_(0)
  , failed_(false)
  , dirty_(false)
  , ctx_(0)
  , policy_(KMU_POLICY_DEFAULT)
  , k_(1)
  , w_(1)
  , rr_core_(0)
  , locked_(false)
  , lock_core_(0)
  , inflight_(cfg.ncc, 0)
{
  dlmem_ref::reset(cfg_, mirror_);
  dlmem_ref::reset(cfg_, gids_);
}

void CtaGang::on_reset() {
  // A launch is not a device reset: ctx_ and the per-launch latches persist,
  // as in the RTL. Nothing is in flight here (busy() held the last launch),
  // so the placement state is the one the next launch's first beat would
  // set; the legacy steering restarts with the KMU's round robin.
  dlmem_ref::reset(cfg_, mirror_);
  dlmem_ref::reset(cfg_, gids_);
  blocks_    = 0;
  state_     = State::IDLE;
  step_      = 0;
  fail_step_ = 0;
  members_.clear();
  failed_    = false;
  dirty_     = false;
  done_q_.clear();
  rr_core_   = 0;
  locked_    = false;
  lock_core_ = 0;
  inflight_.assign(inflight_.size(), 0);
}

void CtaGang::cta_done(uint32_t core, uint32_t slot, uint32_t gang_id, uint32_t epoch) {
  done_q_.push_back({core, slot, gang_id, epoch});
  this->tick_wake();
}

bool CtaGang::busy() const {
  return (state_ != State::IDLE) || (blocks_ != 0) || !done_q_.empty() || (bus_in.size() != 0);
}

bool CtaGang::lane_ready(uint32_t core) const {
  return !bus_out.at(core).full() && inflight_.at(core) < Kmu::LAUNCH_CREDITS;
}

void CtaGang::send_beat(uint32_t core, const kmu_req_t& beat) {
  // The lane's own registered stage is the RTL unit's output register, so the
  // unit adds one cycle to the launch path, as in hardware.
  bus_out.at(core).send(beat, 0);
  ++inflight_.at(core);
  bus_in.pop();
  credit_out.send(1, 1);
}

void CtaGang::apply_done(const Done& d) {
  // Same-core clusters emit no done events, so every event is a SPREAD block's.
  check(policy_ == KMU_POLICY_SPREAD, "cta_gang: done event outside a SPREAD launch");
  check(d.core < cfg_.ncc && blocks_ != 0, "cta_gang: done event for no placed block");
  DT(2, this->name() << " cta-done: core_id=" << d.core << ", slot=" << d.slot
        << ", gang=" << d.gang_id << ", epoch=" << d.epoch);
  dlmem_ref::release(mirror_, d.core, d.slot, w_);
  --blocks_;
  if (k_ > 1) {
    dlmem_ref::done(gids_, d.gang_id, d.epoch);
  }
}

void CtaGang::forward_legacy() {
  uint32_t target = cfg_.ncc;
  if (locked_) {
    if (this->lane_ready(lock_core_)) {
      target = lock_core_;
    }
  } else if (this->lane_ready(rr_core_)) {
    target = rr_core_;
  }
  if (target == cfg_.ncc)
    return;
  kmu_req_t beat = bus_in.peek();
  bool last = (beat.cluster_rank + 1 == cluster_size(beat));
  this->send_beat(target, beat);
  locked_    = !last;
  lock_core_ = target;
  if (!locked_) {
    rr_core_ = (target + 1) % cfg_.ncc;
  }
}

void CtaGang::start_attempt(const kmu_req_t& head) {
  uint32_t K = cluster_size(head);
  uint32_t W = dlmem_ref::warps_of(head.block_size, cfg_.nt);
  uint32_t U = dlmem_ref::usable_slots(cfg_, slot_stride(head));
  check(head.is_first_of_cluster && head.cluster_rank == 0, "cta_gang: a SPREAD message must start at rank 0");
  check(K == k_ && W == w_, "cta_gang: cluster shape changed inside a launch");
  check(K <= cfg_.max_cluster && W <= cfg_.nw, "cta_gang: SPREAD gang can never fit");
  snap_ = mirror_;
  if (dlmem_ref::place(cfg_, snap_, K, W, U, members_)) {
    fail_step_ = K;
  } else {
    // The RTL walks one greedy step per cycle and stops at the first step
    // with no capable core: the longest prefix that still fits.
    fail_step_ = 0;
    for (uint32_t r = 1; r < K; ++r) {
      dlmem_ref::Mirror m = mirror_;
      std::vector<dlmem_ref::Peer> prefix;
      if (!dlmem_ref::place(cfg_, m, r, W, U, prefix))
        break;
      fail_step_ = r;
    }
  }
  step_  = 0;
  // The snapshot already holds this cycle's done event.
  dirty_ = false;
  state_ = State::PLACE;
}

void CtaGang::commit() {
  // The placement took only slots free in its snapshot, and the live mirror
  // has since only released, so applying it to the live copy is exact.
  for (const auto& p : members_) {
    check(((mirror_.free.at(p.core) >> p.slot) & 1) != 0, "cta_gang: committed slot is not free");
    mirror_.free[p.core] &= ~(uint64_t(1) << p.slot);
    mirror_.used[p.core] += w_;
    check(mirror_.used[p.core] <= cfg_.nw, "cta_gang: core over-reserved");
  }
  mirror_.rr = snap_.rr;
  blocks_ += members_.size();
  cur_gang_  = 0;
  cur_epoch_ = 0;
  if (members_.size() > 1) {
    int id = dlmem_ref::alloc(gids_, members_.size());
    check(id >= 0, "cta_gang: gang-id pool exhausted at commit");
    cur_gang_  = uint32_t(id);
    cur_epoch_ = gids_.epoch[id];
  }
  DT(1, this->name() << " place: ctx=" << ctx_ << ", K=" << members_.size() << ", W=" << w_
        << ", gang=" << cur_gang_ << ", last={" << members_.back().core << "," << members_.back().slot << "}");
  failed_ = false;
  step_   = 0;
  state_  = State::EMIT;
}

void CtaGang::emit() {
  if (bus_in.empty())
    return;
  const auto& p = members_.at(step_);
  if (!this->lane_ready(p.core))
    return;
  kmu_req_t beat = bus_in.peek();
  check(beat.cluster_rank == step_, "cta_gang: SPREAD beat out of rank order");
  beat.pinned      = true;
  beat.dest_core   = p.core;
  beat.forced_slot = p.slot;
  beat.gang_id     = cur_gang_;
  beat.epoch       = cur_epoch_;
  beat.row.k       = members_.size();
  beat.row.peers   = members_;
  this->send_beat(p.core, beat);
  if (++step_ == members_.size()) {
    state_ = State::IDLE;
  }
}

void CtaGang::on_tick() {
  for (uint32_t c = 0; c < cfg_.ncc; ++c) {
    auto& credits = credit_in.at(c);
    while (!credits.empty()) {
      credits.pop();
      check(inflight_[c] != 0, "cta_gang: credit without a launch in flight");
      --inflight_[c];
    }
  }

  // dirty_ is a register: an event lands in this cycle's placement snapshot,
  // but the retry gate sees it only from the next cycle.
  bool dirty_q = dirty_;
  if (!done_q_.empty()) {
    this->apply_done(done_q_.front());
    done_q_.pop_front();
    dirty_ = true;
  }

  switch (state_) {
  case State::IDLE: {
    if (bus_in.empty())
      break;
    const auto& head = bus_in.peek();
    if (head.ctx_id != ctx_) {
      ctx_    = head.ctx_id;
      policy_ = head.policy;
      k_      = cluster_size(head);
      w_      = dlmem_ref::warps_of(head.block_size, cfg_.nt);
      dlmem_ref::new_launch(cfg_, mirror_, gids_);
      failed_ = false;
    }
    if (head.policy != KMU_POLICY_SPREAD) {
      this->forward_legacy();
      break;
    }
    bool id_free = (k_ == 1)
                || (std::find(gids_.live.begin(), gids_.live.end(), false) != gids_.live.end());
    if (id_free && (!failed_ || dirty_q)) {
      this->start_attempt(head);
    }
  } break;
  case State::PLACE:
    if (step_ == fail_step_) {
      failed_ = true;
      state_  = State::IDLE;
    } else if (step_ + 1 == members_.size()) {
      this->commit();
    } else {
      ++step_;
    }
    break;
  case State::EMIT:
    this->emit();
    break;
  }

  if (state_ == State::IDLE && done_q_.empty() && bus_in.size() == 0) {
    this->tick_sleep();
  }
}
