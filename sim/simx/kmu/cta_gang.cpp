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
  cfg.num_gangs   = VX_CFG_DLMEM_NUM_GANGS;
  cfg.socket_size = VX_CFG_SOCKET_SIZE;
  cfg.num_sockets = NUM_SOCKETS;
  cfg.dlmem_base  = VX_MEM_DLMEM_BASE_ADDR;
  cfg.lmem_base   = VX_MEM_LMEM_BASE_ADDR;
  cfg.nslots      = VX_CFG_MAX_CTAS_PER_SM;
  cfg.max_cluster_ctas = VX_CFG_MAX_CTAS_PER_SM_CLUSTER;
  cfg.num_gpcs    = VX_CFG_NUM_CLUSTERS;
#ifdef VX_CFG_CLUSTER_PLACEMENT_H100
  cfg.policy      = dlmem_ref::PLACE_H100;
#else
  cfg.policy      = dlmem_ref::PLACE_SPREAD;
#endif
  cfg.gpc_start   = VX_CFG_CLUSTER_GPC_START;
  return cfg;
}

uint32_t vortex::slot_stride(const kmu_req_t& req) {
#if defined(VX_CFG_EXT_DXA_ENABLE) && (VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8) > VX_CFG_MEM_BLOCK_SIZE)
  // A DXA build starts every slot on a DMA word, as the dispatcher does.
  constexpr uint32_t kSlotAlign = VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);
  return (req.lmem_size + kSlotAlign - 1u) & ~(kSlotAlign - 1u);
#else
  return (req.lmem_size + VX_CFG_MEM_BLOCK_SIZE - 1u) & ~uint32_t(VX_CFG_MEM_BLOCK_SIZE - 1u);
#endif
}

namespace {
constexpr uint64_t kBoundary = VX_CFG_H100_ORG_ENABLED ? VX_CFG_DOMAIN_BOUNDARY_DELAY : 0;
} // namespace

CtaGang::CtaGang(const SimContext& ctx, const char* name, const dlmem_ref::Cfg& cfg)
  : SimObject<CtaGang>(ctx, name)
  , release_out(cfg.ncc, this)
  , alloc_in(this, 64)
  , relp_out(this)
  , cfg_(cfg)
  , gk_(cfg.num_gangs, 0)
  , mode_(false)
  , relp_sent_(cfg.num_gangs, false)
  // From the cycle a release leaves: the gbar unit's response register, then
  // one response-fork stage per fan-out level that exists.
  , rel_delay_(1 + (cfg.num_sockets > 1 ? 1 : 0) + (cfg.socket_size > 1 ? 1 : 0))
  // the admission reads the pending bits as a register, across the GPC boundary in H100 builds
  , relp_delay_(1 + kBoundary)
{
  dlmem_ref::reset(cfg_, gids_);
  gph_.assign(cfg_.num_gangs, false);
  sph_.assign(cfg_.ncc, std::vector<bool>(dlmem_ref::slots(cfg_), false));
}

void CtaGang::on_reset() {
  // A launch is not a device reset; nothing is in flight here (busy() held
  // the last launch), so the state is the one the next launch would find.
  dlmem_ref::reset(cfg_, gids_);
  evt_q_.clear();
  gph_.assign(cfg_.num_gangs, false);
  sph_.assign(cfg_.ncc, std::vector<bool>(dlmem_ref::slots(cfg_), false));
  relp_sent_.assign(cfg_.num_gangs, false);
}

void CtaGang::cta_event(const GangEvent& evt) {
  evt_q_.push_back(evt);
  this->tick_wake();
}

bool CtaGang::busy() const {
  bool rel_pend = std::find(gids_.rel_pend.begin(), gids_.rel_pend.end(), true) != gids_.rel_pend.end();
  // A grant in flight is still live at the admission, and the gang-mode
  // message is a level the RTL's busy does not count.
  return !evt_q_.empty() || rel_pend;
}

// A block that ends leaves the gang's live members, and the open phase's
// arrivals if it arrived in it; it can complete that phase, whose release
// then leaves next cycle as an arrival's does. The live count drops before
// the cycle's release; the phase is a register.
void CtaGang::done_phase(const GangEvent& d, bool before) {
  if (!mode_)
    return;
  auto& g = gids_;
  uint32_t id = d.gang_id;
  if (before) {
    check(id < g.live.size() && g.live[id], "done: gang id not live");
    check(g.epoch[id] == d.epoch, "done: stale epoch");
    check(g.cnt[id] > 0, "done: too many done events");
    done_cnt_ = g.cnt[id];
    if (--g.cnt[id] == 0) {
      g.live[id] = false;
    }
    return;
  }
  bool arr = d.arrived && (sph_.at(d.core).at(d.slot) == gph_.at(id));
  check(!arr || g.arrived[id] != 0, "done: the block's arrival is not held by its gang");
  uint32_t left = g.arrived[id] - (arr ? 1 : 0);
  if (left != 0 && left == done_cnt_ - 1) {
    g.arrived[id]  = 0;
    g.rel_pend[id] = true;
    gph_[id] = !gph_[id];
  } else {
    g.arrived[id] = left;
  }
}

void CtaGang::apply_cbar(const GangEvent& a) {
  check(mode_, "cta_gang: cluster-barrier arrival outside a gang launch");
  auto& g = gids_;
  uint32_t id = a.gang_id;
  DT(2, this->name() << " cbar-arrive: core_id=" << a.core << ", slot=" << a.slot << ", gang=" << id << ", epoch=" << a.epoch);
  check(id < g.live.size() && g.live[id], "cbar_arrive: gang id not live");
  check(a.count == gk_.at(id), "cta_gang: cluster-barrier arrival with a block count other than the gang's");
  check(g.epoch[id] == a.epoch, "cbar_arrive: stale epoch");
  check(!g.rel_pend[id], "cbar_arrive: arrival before the previous release left");
  check(g.arrived[id] < g.cnt[id], "cbar_arrive: too many arrivals");
  sph_.at(a.core).at(a.slot) = gph_.at(id);
  // the phase completes when every live member arrived
  if (++g.arrived[id] == g.cnt[id]) {
    g.arrived[id]  = 0;
    g.rel_pend[id] = true;
    gph_[id] = !gph_[id];
  }
}

void CtaGang::on_tick() {
  while (!alloc_in.empty()) {
    const auto& a = alloc_in.peek();
    if (a.mode) {
      mode_ = (a.k > 1);
    } else {
      auto& g = gids_;
      check(a.gang < g.live.size() && !g.live[a.gang] && !g.rel_pend[a.gang] && g.arrived[a.gang] == 0,
            "cta_gang: gang id taken while still live or releasing");
      g.live[a.gang]  = true;
      g.cnt[a.gang]   = a.k;
      g.epoch[a.gang] = a.epoch;
      gk_[a.gang]     = a.k;
    }
    alloc_in.pop();
  }

  bool has_evt = !evt_q_.empty();
  GangEvent evt{};
  if (has_evt) {
    evt = evt_q_.front();
    evt_q_.pop_front();
  }
  if (has_evt && evt.kind == GangEvent::CTA_DONE) {
    DT(2, this->name() << " cta-done: core_id=" << evt.core << ", slot=" << evt.slot
          << ", gang=" << evt.gang_id << ", epoch=" << evt.epoch << ", arrived=" << evt.arrived);
    this->done_phase(evt, true);
  }

  // The pending releases are a register: the lowest leaves this cycle, and
  // an arrival that completes a phase now sets its release for the next cycle.
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
  if (has_evt && evt.kind == GangEvent::CTA_DONE) {
    this->done_phase(evt, false);
  }

  if (gids_.rel_pend != relp_sent_) {
    relp_sent_ = gids_.rel_pend;
    relp_out.send({relp_sent_}, relp_delay_);
  }

  bool rel_pend = std::find(gids_.rel_pend.begin(), gids_.rel_pend.end(), true) != gids_.rel_pend.end();
  if (evt_q_.empty() && alloc_in.size() == 0 && !rel_pend) {
    this->tick_sleep();
  }
}
