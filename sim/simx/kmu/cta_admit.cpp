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

#include "cta_admit.h"
#include <VX_config.h>
#ifdef VX_CFG_H100_SM_ENABLE
#include <smem_carveout.h>
#else
#line 15
#endif
#include "debug.h"

using namespace vortex;
using dlmem_ref::check;
using dlmem_ref::low_mask;

namespace {

uint32_t cluster_size(const kmu_req_t& req) {
  uint32_t k = req.cluster_dim[0] * req.cluster_dim[1] * req.cluster_dim[2];
  return (k == 0) ? 1 : k;
}

uint32_t popcount64(uint64_t x) {
  return uint32_t(__builtin_popcountll(x));
}

} // namespace

namespace {
constexpr uint32_t kSkidSize = 2;
} // namespace

CtaAdmit::CtaAdmit(const dlmem_ref::Cfg& cfg, bool skid)
  : cfg_(cfg)
  , skid_(skid)
  , gpc_(cfg.num_gpcs)
  , ctx_(0)
{
  check(cfg.gpc_start < cfg.num_gpcs, "cta_admit: CLUSTER_GPC_START must name a GPC");
  check(dlmem_ref::slots(cfg) <= cfg.nw && dlmem_ref::slots(cfg) <= 64, "cta_admit: CTA slots exceed the warps or 64");
  this->reset();
}

void CtaAdmit::reset() {
  for (auto& g : gpc_) {
    g.free.assign(cfg_.ncc, low_mask(dlmem_ref::slots(cfg_)));
    g.sfree.assign(cfg_.ncc, 0);
    g.used.assign(cfg_.ncc, 0);
    g.sused.assign(cfg_.ncc, 0);
    g.visited.assign(cfg_.ncc, false);
    g.live.assign(cfg_.num_gangs, false);
    g.cnt.assign(cfg_.num_gangs, 0);
    g.epoch.assign(cfg_.num_gangs, 0);
    g.members.clear();
    g.rr = 0;
    g.start = 0;
    g.out = false;
  }
  q_.clear();
  gptr_      = cfg_.gpc_start;
  state_     = State::IDLE;
  step_      = 0;
  win_       = 0;
  cur_gang_  = 0;
  cur_epoch_ = 0;
  failed_    = false;
  dirty_     = false;
  placed_    = false;
  k_         = 1;
  w_         = 0;
  u_         = dlmem_ref::slots(cfg_);
}

bool CtaAdmit::busy() const {
  if (state_ != State::IDLE || !q_.empty())
    return true;
  for (const auto& g : gpc_) {
    for (uint32_t c = 0; c < cfg_.ncc; ++c) {
      if (g.free[c] != low_mask(dlmem_ref::slots(cfg_)))
        return true;
    }
    for (bool l : g.live) {
      if (l)
        return true;
    }
  }
  return false;
}

CtaAdmit::Outputs CtaAdmit::tick(const Inputs& in) {
  const uint32_t G = cfg_.num_gpcs;
  const uint32_t nslots = dlmem_ref::slots(cfg_);
  const uint32_t cap = dlmem_ref::cluster_cap(cfg_);
  Outputs out;

  // the skid's ready is a register: its occupancy at the start of the cycle
  bool next_ready = in.out_ready;
  bool skid_pop = false;
  if (skid_) {
    next_ready = (q_.size() < kSkidSize);
    if (!q_.empty()) {
      const auto& h = q_.front();
      out.out_valid  = true;
      out.out        = h.req;
      out.out_dest   = h.dest;
      out.out_pinned = h.pinned;
      out.out_eop    = h.eop;
      skid_pop = in.out_ready;
    }
  }
  SkidBeat beat{};

  // ---- events: applied before this cycle's decision ----
  struct Next {
    std::vector<uint64_t> free;
    std::vector<uint32_t> used;
    std::vector<bool>     live;
    std::vector<uint32_t> cnt;
  };
  std::vector<Next> nx(G);
  bool any_done = false;
  for (uint32_t g = 0; g < G; ++g) {
    auto& gp = gpc_[g];
    nx[g] = {gp.free, gp.used, gp.live, gp.cnt};
    const AdmitDone* d = in.done.at(g);
    if (!d)
      continue;
    any_done = true;
    check(placed_, "cta_admit: CTA_DONE outside a cluster launch");
    check(d->core < cfg_.ncc && ((gp.free[d->core] >> d->slot) & 1) == 0 && gp.used[d->core] >= w_,
          "cta_admit: CTA_DONE for a slot that holds no placed block");
    nx[g].free[d->core] |= uint64_t(1) << d->slot;
    nx[g].used[d->core] -= w_;
    if (k_ > 1) {
      check(d->gang_id < cfg_.num_gangs && gp.live[d->gang_id] && gp.epoch[d->gang_id] == d->epoch
            && gp.cnt[d->gang_id] != 0, "cta_admit: CTA_DONE for a gang that is not live");
      nx[g].cnt[d->gang_id] -= 1;
      if (gp.cnt[d->gang_id] == 1) {
        nx[g].live[d->gang_id] = false;
      }
    }
  }

  // lowest free gang id per GPC: neither live nor holding back a release
  std::vector<bool>     id_free(G, false);
  std::vector<uint32_t> free_id(G, 0);
  bool any_id_free = false;
  for (uint32_t g = 0; g < G; ++g) {
    for (uint32_t i = 0; i < cfg_.num_gangs; ++i) {
      if (!nx[g].live[i] && !in.rel_pend.at(g).at(i)) {
        id_free[g] = true;
        free_id[g] = i;
        break;
      }
    }
    any_id_free = any_id_free || id_free[g];
  }

  // ---- head beat ----
  const kmu_req_t* head = in.head;
  bool head_valid = (head != nullptr);
  bool in_placed = false, in_new = false;
  uint32_t in_k = 1, in_w = 0, in_u = nslots;
  if (head_valid) {
    in_placed = (head->policy == KMU_POLICY_SPREAD)
             || (VX_CFG_CLUSTER_DISTINCT_SMS_ENABLED && head->explicit_cluster);
    in_new = (head->ctx_id != ctx_);
    in_k = cluster_size(*head);
    in_w = dlmem_ref::warps_of(head->block_size, cfg_.nt);
#ifdef VX_CFG_H100_SM_ENABLE
    in_u = smem_usable_slots(head->smem_bytes, slot_stride(*head), dlmem_ref::slots(cfg_), smem_carve_cfg());
#else
#line 178
    in_u = dlmem_ref::usable_slots(cfg_, slot_stride(*head));
#line 178
#endif
  }

  // ---- one greedy placement step per GPC on the scratch copies ----
  std::vector<bool>     any_cand(G, false);
  std::vector<uint32_t> sel_core(G, 0), sel_slot(G, 0);
  if (state_ == State::PLACE) {
    uint64_t umask = low_mask(u_);
    for (uint32_t g = 0; g < G; ++g) {
      auto& gp = gpc_[g];
      if (gp.out)
        continue;
      int best = -1;
      uint32_t best_res = 0;
      for (uint32_t i = 0; i < cfg_.ncc; ++i) {
        // H100: least resident, then lowest; SPREAD: first at or after the start core
        uint32_t c = (cfg_.policy == dlmem_ref::PLACE_H100) ? i : (gp.start + i) % cfg_.ncc;
        uint32_t res = popcount64(~gp.sfree[c] & low_mask(nslots));
        bool cand = ((gp.sfree[c] & umask) != 0) && (gp.sused[c] + w_ <= cfg_.nw) && (res < cap) && !gp.visited[c];
        if (!cand)
          continue;
        if (cfg_.policy == dlmem_ref::PLACE_H100) {
          if (best < 0 || res < best_res) {
            best = int(c);
            best_res = res;
          }
        } else if (best < 0) {
          best = int(c);
        }
      }
      if (best < 0)
        continue;
      any_cand[g] = true;
      sel_core[g] = uint32_t(best);
      uint64_t avail = gp.sfree[best] & umask;
      while (((avail >> sel_slot[g]) & 1) == 0) {
        ++sel_slot[g];
      }
    }
  }
  bool any_gpc = false;
  uint32_t win = gptr_;
  for (uint32_t i = 0; i < G; ++i) {
    uint32_t g = (gptr_ + i) % G;
    if (any_cand[g]) {
      win = g;
      any_gpc = true;
      break;
    }
  }
  bool is_last = (step_ + 1 == k_);

  // ---- control ----
  bool idle   = (state_ == State::IDLE);
  bool emit   = (state_ == State::EMIT);
  bool is_new = idle && head_valid && in_new;
  bool start  = idle && head_valid && in_placed
             && ((in_k == 1) || any_id_free)
             && (!(failed_ && !is_new) || dirty_);
  bool fwd = (idle && !in_placed) || emit;

  out.in_ready = head_valid && fwd && next_ready;
  if (head_valid && fwd) {
    beat.req = *head;
    beat.eop = emit || (head->cluster_rank + 1 >= in_k);
    beat.pinned = false;
    beat.dest = 0;
    if (emit) {
      const auto& p = gpc_[win_].members.at(step_);
      beat.pinned = true;
      beat.eop    = true;
      beat.dest   = win_ * cfg_.ncc + p.core;
      beat.req.pinned      = true;
      beat.req.policy      = KMU_POLICY_SPREAD;
      beat.req.dest_core   = p.core;
      beat.req.forced_slot = p.slot;
      beat.req.gang_id     = cur_gang_;
      beat.req.epoch       = cur_epoch_;
      beat.req.row.k       = k_;
      beat.req.row.peers   = gpc_[win_].members;
      check(head->cluster_rank == step_, "cta_admit: cluster beat out of rank order");
    }
    if (!skid_) {
      out.out_valid  = true;
      out.out        = beat.req;
      out.out_dest   = beat.dest;
      out.out_pinned = beat.pinned;
      out.out_eop    = beat.eop;
    }
  }
  if (skid_) {
    if (skid_pop) {
      q_.pop_front();
    }
    if (out.in_ready) {
      q_.push_back(beat);
    }
  }
  if (idle && head_valid && in_placed) {
    check(head->is_first_of_cluster && head->cluster_rank == 0, "cta_admit: a cluster message must start at rank 0");
    check(in_k <= cfg_.max_cluster && in_w <= cfg_.nw, "cta_admit: cluster can never fit");
  }

  // ---- registers ----
  for (uint32_t g = 0; g < G; ++g) {
    gpc_[g].free = nx[g].free;
    gpc_[g].used = nx[g].used;
    gpc_[g].live = nx[g].live;
    gpc_[g].cnt  = nx[g].cnt;
  }
  bool dirty_n = !start && (dirty_ || any_done);

  switch (state_) {
  case State::IDLE:
    if (head_valid) {
      ctx_    = head->ctx_id;
      placed_ = in_placed;
      k_      = in_k;
      w_      = in_w;
      u_      = in_u;
      if (in_new) {
        for (auto& gp : gpc_) {
          gp.rr = 0;
          std::fill(gp.epoch.begin(), gp.epoch.end(), 0);
        }
        gptr_   = cfg_.gpc_start;
        failed_ = false;
      }
    }
    if (start) {
      for (uint32_t g = 0; g < G; ++g) {
        auto& gp = gpc_[g];
        gp.sfree = nx[g].free;
        gp.sused = nx[g].used;
        gp.visited.assign(cfg_.ncc, false);
        gp.members.assign(in_k, dlmem_ref::Peer{0, 0});
        gp.out   = !((in_k == 1) || id_free[g]);
        gp.start = in_new ? 0 : gp.rr;
      }
      step_  = 0;
      state_ = State::PLACE;
    }
    break;
  case State::PLACE:
    if (!any_gpc) {
      failed_ = true;
      state_  = State::IDLE;
      break;
    }
    for (uint32_t g = 0; g < G; ++g) {
      auto& gp = gpc_[g];
      if (any_cand[g]) {
        gp.members.at(step_) = {sel_core[g], sel_slot[g]};
        gp.sfree[sel_core[g]] &= ~(uint64_t(1) << sel_slot[g]);
        gp.sused[sel_core[g]] += w_;
        gp.visited[sel_core[g]] = true;
      } else {
        gp.out = true;
      }
    }
    if (is_last) {
      // the slots were free in the snapshot, and the live mirror has only released since
      auto& gp = gpc_[win];
      for (const auto& p : gp.members) {
        check(((gp.free[p.core] >> p.slot) & 1) != 0, "cta_admit: committed slot is not free");
        gp.free[p.core] &= ~(uint64_t(1) << p.slot);
        gp.used[p.core] += w_;
        check(gp.used[p.core] <= cfg_.nw, "cta_admit: core over-reserved");
      }
      gp.rr  = (sel_core[win] + 1) % cfg_.ncc;
      gptr_  = (win + 1) % G;
      win_   = win;
      if (k_ > 1) {
        uint32_t id = free_id[win];
        gp.live[id]  = true;
        gp.cnt[id]   = k_;
        gp.epoch[id] ^= 1;
        cur_gang_    = id;
        cur_epoch_   = gp.epoch[id];
        out.alloc       = true;
        out.alloc_gpc   = win;
        out.alloc_gang  = id;
        out.alloc_epoch = gp.epoch[id];
        out.alloc_k     = k_;
      } else {
        cur_gang_  = 0;
        cur_epoch_ = 0;
      }
      DT(1, "cta-admit place: ctx=" << uint32_t(ctx_) << ", K=" << k_ << ", W=" << w_ << ", gpc=" << win
            << ", gang=" << cur_gang_ << ", last={" << sel_core[win] << "," << sel_slot[win] << "}");
      failed_ = false;
      step_   = 0;
      state_  = State::EMIT;
    } else {
      ++step_;
    }
    break;
  case State::EMIT:
    if (out.in_ready) {
      if (is_last) {
        step_  = 0;
        state_ = State::IDLE;
      } else {
        ++step_;
      }
    }
    break;
  }
  dirty_ = dirty_n;
  return out;
}

///////////////////////////////////////////////////////////////////////////////

namespace {

std::vector<LaunchTree::Level> admit_levels(uint32_t num_gpcs) {
  // the output skid, then the registered per-GPC fan-out when there are several
  std::vector<LaunchTree::Level> levels{LaunchTree::Level(1, true)};
  if (num_gpcs > 1) {
    levels.emplace_back(num_gpcs);
  }
  return levels;
}

constexpr uint64_t kBoundary = VX_CFG_H100_ORG_ENABLED ? VX_CFG_DOMAIN_BOUNDARY_DELAY : 0;

} // namespace

AdmitUnit::AdmitUnit(const SimContext& ctx, const char* name, const dlmem_ref::Cfg& cfg, bool direct)
  : SimObject<AdmitUnit>(ctx, name)
  // Two entries keep a pass-through stream at one beat per cycle behind the
  // lane's credit loop.
  , bus_in(this, 2)
  , credit_out(this)
  , bus_out(cfg.num_gpcs, this)
  , credit_in(cfg.num_gpcs, this)
  , evt_in(cfg.num_gpcs, SimChannel<GangEvent>(this, 64))
  , relp_in(cfg.num_gpcs, SimChannel<GangRelPend>(this, 64))
  , alloc_out(cfg.num_gpcs, this)
  , cfg_(cfg)
  , direct_(direct)
  , admit_(cfg, false)
  , out_(admit_levels(cfg.num_gpcs))
  , evt_q_(cfg.num_gpcs)
  , rel_pend_(cfg.num_gpcs, std::vector<bool>(cfg.num_gangs, false))
  , mode_(false)
  , link_delay_(1 + kBoundary)
{}

void AdmitUnit::on_reset() {
  // A launch is not a device reset: the per-launch latches persist, as in the
  // RTL; nothing is in flight here (busy() held the last launch).
  out_.reset();
}

bool AdmitUnit::busy() const {
  // The RTL's busy reads the pending-release register the admission sees this
  // cycle: the copy delivered for the next tick, not the one taken at the last.
  bool rel = false;
  for (uint32_t g = 0; g < cfg_.num_gpcs; ++g) {
    const auto& pend = relp_in[g].empty() ? rel_pend_[g] : relp_in[g].peek().pend;
    for (bool r : pend) {
      rel = rel || r;
    }
  }
  bool evt = false;
  for (uint32_t g = 0; g < cfg_.num_gpcs; ++g) {
    evt = evt || !evt_q_[g].empty() || (evt_in[g].size() != 0);
  }
  return admit_.busy() || (bus_in.size() != 0) || !out_.empty() || rel || evt;
}

void AdmitUnit::on_tick() {
  uint64_t now = SimPlatform::instance().cycles();
  const uint32_t G = cfg_.num_gpcs;
  for (uint32_t g = 0; g < G; ++g) {
    while (!credit_in[g].empty()) {
      credit_in[g].pop();
      out_.credit(g);
    }
  }
  auto sink = [this](uint32_t lane, const kmu_req_t& req) {
    bus_out.at(lane).send(req, direct_ ? 0 : 1);
  };
  out_.begin_cycle();
  out_.advance(now, sink);

  // the gang tables' registered pending-release bits
  for (uint32_t g = 0; g < G; ++g) {
    while (!relp_in[g].empty()) {
      rel_pend_[g] = relp_in[g].peek().pend;
      relp_in[g].pop();
    }
  }

  // one gang event per GPC and cycle, in the gang table's queue order
  std::vector<AdmitDone> done(G);
  CtaAdmit::Inputs in;
  in.done.assign(G, nullptr);
  in.rel_pend = rel_pend_;
  for (uint32_t g = 0; g < G; ++g) {
    while (!evt_in[g].empty()) {
      evt_q_[g].push_back(evt_in[g].peek());
      evt_in[g].pop();
    }
    if (evt_q_[g].empty())
      continue;
    const auto& e = evt_q_[g].front();
    if (e.kind == GangEvent::CTA_DONE) {
      done[g] = {e.core, e.slot, e.gang_id, e.epoch};
      in.done[g] = &done[g];
    }
    evt_q_[g].pop_front();
  }

  LaunchTree::Beat probe{};
  in.head = bus_in.empty() ? nullptr : &bus_in.peek();
  in.out_ready = out_.ready(probe);
  auto o = admit_.tick(in);
  if (o.in_ready) {
    LaunchTree::Beat beat{o.out, o.out_eop, o.out_pinned, o.out_dest / cfg_.ncc};
    out_.push(beat, now, sink);
    bus_in.pop();
    credit_out.send(1, 1);
  }
  if (o.alloc) {
    alloc_out.at(o.alloc_gpc).send({false, o.alloc_gang, o.alloc_epoch, o.alloc_k}, link_delay_);
  }
  if (admit_.gang_mode() != mode_) {
    mode_ = admit_.gang_mode();
    for (auto& a : alloc_out) {
      a.send({true, 0, 0, mode_ ? 2u : 1u}, link_delay_);
    }
  }

  bool waiting = false;
  for (uint32_t g = 0; g < G; ++g) {
    waiting = waiting || (credit_in[g].size() != 0) || (relp_in[g].size() != 0);
  }
  if (!this->busy() && !waiting) {
    this->tick_sleep();
  }
}
