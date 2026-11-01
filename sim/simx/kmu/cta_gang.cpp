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
#include "barrier_unit.h"
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
#if defined(VX_CFG_EXT_DXA_ENABLE) && (VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8) > VX_CFG_MEM_BLOCK_SIZE)
  // A DXA build starts every slot on a DMA word, as the dispatcher does.
  constexpr uint32_t kSlotAlign = VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);
  return (req.lmem_size + kSlotAlign - 1u) & ~(kSlotAlign - 1u);
#else
  return (req.lmem_size + VX_CFG_MEM_BLOCK_SIZE - 1u) & ~uint32_t(VX_CFG_MEM_BLOCK_SIZE - 1u);
#endif
#line 49
}

} // namespace

CtaGang::CtaGang(const SimContext& ctx, const char* name, const dlmem_ref::Cfg& cfg)
  : SimObject<CtaGang>(ctx, name)
  // Two entries keep a pass-through stream at one beat per cycle behind the
  // lane's credit loop.
  , bus_in(this, 2)
  , credit_out(this)
  , bus_out(this)
  , credit_in(this)
  , release_out(cfg.ncc, this)
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
  , out_({LaunchTree::Level(1, true)})
  , direct_(cfg.num_sockets * cfg.socket_size == 1)
  // From the cycle a release leaves: the gbar unit's response register, then
  // one response-fork stage per fan-out level that exists.
  , rel_delay_(1 + (cfg.num_sockets > 1 ? 1 : 0) + (cfg.socket_size > 1 ? 1 : 0))
{
  dlmem_ref::reset(cfg_, mirror_);
  dlmem_ref::reset(cfg_, gids_);
}

void CtaGang::on_reset() {
  // A launch is not a device reset: ctx_ and the per-launch latches persist,
  // as in the RTL. Nothing is in flight here (busy() held the last launch),
  // so the placement state is the one the next launch's first beat would
  // set.
  dlmem_ref::reset(cfg_, mirror_);
  dlmem_ref::reset(cfg_, gids_);
  blocks_    = 0;
  state_     = State::IDLE;
  step_      = 0;
  fail_step_ = 0;
  members_.clear();
  failed_    = false;
  dirty_     = false;
  evt_q_.clear();
  out_.reset();
}

void CtaGang::cta_event(const GangEvent& evt) {
  evt_q_.push_back(evt);
  this->tick_wake();
}

bool CtaGang::busy() const {
  bool rel_pend = std::find(gids_.rel_pend.begin(), gids_.rel_pend.end(), true) != gids_.rel_pend.end();
  return (state_ != State::IDLE) || (blocks_ != 0) || !evt_q_.empty() || (bus_in.size() != 0) || rel_pend;
}

// An id is reusable only once its last release left.
bool CtaGang::id_free() const {
  for (size_t i = 0; i < gids_.live.size(); ++i) {
    if (!gids_.live[i] && !gids_.rel_pend[i])
      return true;
  }
  return false;
}

bool CtaGang::out_ready(const LaunchTree::Beat& beat) const {
  return out_.ready(beat);
}

void CtaGang::send_beat(const LaunchTree::Beat& beat) {
  // The skid shows the beat at its output the next cycle: through the core
  // lane's registered stage, or a cycle's delay into the fan-out.
  out_.push(beat, SimPlatform::instance().cycles(), [this](uint32_t, const kmu_req_t& req) {
    bus_out.send(req, direct_ ? 0 : 1);
  });
  bus_in.pop();
  credit_out.send(1, 1);
}

void CtaGang::apply_done(const GangEvent& d) {
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

void CtaGang::apply_cbar(const GangEvent& a) {
  check(policy_ == KMU_POLICY_SPREAD && k_ > 1, "cta_gang: cluster-barrier arrival outside a SPREAD gang");
  check(a.count == k_, "cta_gang: cluster-barrier arrival with a block count other than the gang's");
  DT(2, this->name() << " cbar-arrive: core_id=" << a.core << ", gang=" << a.gang_id << ", epoch=" << a.epoch);
  dlmem_ref::cbar_arrive(gids_, a.gang_id, a.epoch, k_);
}

void CtaGang::forward_legacy() {
  // The cluster's fan-out picks the core, and keeps a same-core cluster on it.
  LaunchTree::Beat beat{};
  beat.req = bus_in.peek();
  beat.eop = (beat.req.cluster_rank + 1 == cluster_size(beat.req));
  if (this->out_ready(beat)) {
    this->send_beat(beat);
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
  // A placed block is a whole message, routed to its core.
  LaunchTree::Beat beat{};
  beat.eop    = true;
  beat.pinned = true;
  beat.dest   = p.core;
  if (!this->out_ready(beat)) {
    return;
  }
  beat.req = bus_in.peek();
  check(beat.req.cluster_rank == step_, "cta_gang: SPREAD beat out of rank order");
  beat.req.pinned      = true;
  beat.req.dest_core   = p.core;
  beat.req.forced_slot = p.slot;
  beat.req.gang_id     = cur_gang_;
  beat.req.epoch       = cur_epoch_;
  beat.req.row.k       = members_.size();
  beat.req.row.peers   = members_;
  this->send_beat(beat);
  if (++step_ == members_.size()) {
    state_ = State::IDLE;
  }
}

void CtaGang::on_tick() {
  while (!credit_in.empty()) {
    credit_in.pop();
    out_.credit(0);
  }
  out_.begin_cycle();

  // dirty_ is a register: an event lands in this cycle's placement snapshot,
  // but the retry gate sees it only from the next cycle.
  bool dirty_q = dirty_;
  bool has_evt = !evt_q_.empty();
  GangEvent evt{};
  if (has_evt) {
    evt = evt_q_.front();
    evt_q_.pop_front();
  }
  // a cluster-barrier arrival frees nothing, so it cannot wake a failed placement
  if (has_evt && evt.kind == GangEvent::CTA_DONE) {
    this->apply_done(evt);
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
    bool id_free = (k_ == 1) || this->id_free();
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

  // The pending releases are a register: the lowest leaves this cycle, after
  // this cycle's placement saw it pending, and an arrival that completes a
  // phase now sets its release for the next cycle.
  int rel = dlmem_ref::cbar_pop(gids_);
  if (rel >= 0) {
    GbarResume msg{0};
    msg.kind    = GBAR_KIND_CBAR;
    msg.gang_id = uint32_t(rel);
    msg.epoch   = gids_.epoch[rel];
    DT(2, this->name() << " cbar-release: gang=" << rel << ", epoch=" << msg.epoch);
    // RTL also holds a release while a global barrier response occupies gbar_unit's register; not modelled.
    for (auto& link : release_out) {
      link.send(msg, rel_delay_);
    }
  }
  if (has_evt && evt.kind == GangEvent::CBAR) {
    this->apply_cbar(evt);
  }

  bool rel_pend = std::find(gids_.rel_pend.begin(), gids_.rel_pend.end(), true) != gids_.rel_pend.end();
  if (state_ == State::IDLE && evt_q_.empty() && bus_in.size() == 0 && credit_in.size() == 0 && !rel_pend) {
    this->tick_sleep();
  }
}
