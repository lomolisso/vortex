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

#include "barrier_unit.h"
#include <cassert>
#include "core.h"
#include "scheduler.h"
#include "cluster.h"
#include "debug.h"

using namespace vortex;

BarrierUnit::BarrierUnit(const SimContext& ctx, const char* name, Core* core, Scheduler* scheduler)
  : SimObject<BarrierUnit>(ctx, name)
  , core_(core)
  , scheduler_(scheduler)
  , barriers_(VX_CFG_NUM_WARPS * VX_CFG_NUM_BARRIERS)
{}

BarrierUnit::~BarrierUnit() {}

void BarrierUnit::on_reset() {
  for (auto& barrier : barriers_) {
    barrier.reset();
  }
}

uint32_t BarrierUnit::get_phase(uint32_t bar_id) const {
  auto bar_index = bar_id & 0x3fffffff;
  return barriers_.at(bar_index).phase;
}

void BarrierUnit::arrive(uint32_t bar_id, uint32_t count, uint32_t wid, bool is_sync_bar) {
  bool is_global = (bar_id >> 31);
  auto bar_index = bar_id & 0x3fffffff;

  auto& barrier = barriers_.at(bar_index);
  const auto& active_warps = scheduler_->active_warps();

  DP(4, "*** Barrier arrive: core #" << core_->id() << ", warp #" << wid << " at barrier #" << bar_id << ", phase=" << barrier.phase << ", wait_mask=" << barrier.wait_mask << ", count=" << barrier.count << ", events=" << barrier.events);

#ifdef VX_CFG_DSMEM_ENABLE
  if (!is_global && ((bar_id >> 30) & 1)) {
    // CTA-cluster barrier: local arrivals complete one rank's worth (count =
    // the CTA's warps), then escalate a rank arrival to the cluster, which
    // counts ranks against the residency table and broadcast-resumes.
    barrier.count = count;
    barrier.arrival_mask.set(wid);
    if (is_sync_bar) {
      barrier.wait_mask.set(wid);
    }
    // Drain obligation: outstanding remote (DSMEM) stores become barrier
    // events, released one-per-ack as they commit in the peers' LMEM banks.
    // The warp is suspended past this point, so no new stores can slip in.
    uint32_t drain = core_->dsmem_store_count(wid);
    if (drain != 0) {
      barrier.events += drain;
      core_->dsmem_arm_drain(wid, bar_id);
    }
    if (barrier.arrival_mask.count() >= barrier.count && barrier.events == 0) {
      this->cluster_escalate(bar_id, wid);
    }
    return;
  }
#endif

  if (is_global) {
    barrier.count = count;
    // Track arrivals separately from waiters. Async arrive does NOT suspend
    // the warp, so it must not appear in wait_mask — otherwise global_resume
    // would wake it spuriously while it's mid-stalling-instruction.
    barrier.arrival_mask.set(wid);
    if (is_sync_bar) {
      barrier.wait_mask.set(wid);
    }
    if (barrier.arrival_mask.count() == active_warps.count() && barrier.events == 0) {
      // wait_mask is preserved across the async hop (holds warps resumed by
      // global_resume) and cleared there. arrival_mask is local-only and reset here.
      core_->cluster()->global_barrier_arrive(bar_id, count, core_->id());
      barrier.arrival_mask.reset();
      barrier.count  = 0;
      barrier.events = 0;
    }
  } else {
    if (is_sync_bar) {
      barrier.wait_mask.set(wid);
    }
    // local barrier handling
    auto barrier_count_p1 = barrier.count + 1;
    if (barrier_count_p1 == count && barrier.events == 0) {
      // resume waiting warps
      for (uint32_t i = 0; i < VX_CFG_NUM_WARPS; ++i) {
        if (barrier.wait_mask.test(i)) {
          DP(3, "*** Resume core #" << core_->id() << ", warp #" << i << " at barrier #" << bar_id);
          scheduler_->resume(i);
        }
      }
      // reset barrier and advance phase
      barrier.wait_mask.reset();
      ++barrier.phase;
    }
    // update count and wrap around
    if (count == 0) {
      std::cout << "Error: barrier_arrive count=0: core=" << core_->id() << " wid=" << wid << " bar_id=0x" << std::hex << bar_id << std::dec << " bar_index=" << bar_index << " is_sync=" << is_sync_bar << std::endl;
      std::abort();
    }
    barrier.count = barrier_count_p1 % count;
  }
}

bool BarrierUnit::wait(uint32_t bar_id, uint32_t phase, uint32_t wid) {
  auto bar_index = bar_id & 0x3fffffff;

  auto& barrier = barriers_.at(bar_index);
  bool wait = (barrier.phase == phase);

  DP(4, "*** Barrier wait: core #" << core_->id() << ", warp #" << wid << " at barrier #" << bar_id << ", wait=" << wait);

  if (wait) {
    // add warp to wait list
    barrier.wait_mask.set(wid);
    DP(3, "*** Suspend core #" << core_->id() << ", warp #" << wid << " at barrier #" << bar_id);
  }

  return wait;
}

void BarrierUnit::global_resume(uint32_t bar_id) {
  auto bar_index = bar_id & 0x3fffffff;
  auto& barrier = barriers_.at(bar_index);
  // Only resume warps that actually suspended on this barrier (sync_bar or
  // bar_wait). Async bar_arrive leaves wait_mask empty so this is a no-op,
  // which is the correct semantics for fire-and-forget global barriers.
  for (uint32_t i = 0; i < VX_CFG_NUM_WARPS; ++i) {
    if (barrier.wait_mask.test(i)) {
      DP(3, "*** Resume core #" << core_->id() << ", warp #" << i << " at barrier #" << bar_id);
      scheduler_->resume(i);
    }
  }
  barrier.wait_mask.reset();
  ++barrier.phase;
}

#ifdef VX_CFG_DSMEM_ENABLE
void BarrierUnit::cluster_escalate(uint32_t bar_id, uint32_t wid) {
  auto bar_index = bar_id & 0x3fffffff;
  auto& barrier = barriers_.at(bar_index);
  auto& csrs = scheduler_->warp(wid).cta_csrs;
  DP(3, "*** Cluster-barrier escalate: core #" << core_->id() << ", cluster #"
     << csrs.cluster_id << " at barrier #" << bar_id);
  core_->cluster()->cluster_barrier_arrive(csrs.cluster_id, bar_id);
  // One rank's worth of arrivals consumed; further ranks on this core (if
  // any) accumulate afresh. wait_mask persists until the cluster resume.
  barrier.arrival_mask.reset();
}
#endif

void BarrierUnit::event_attach(uint32_t bar_id, uint32_t count) {
  auto bar_index = bar_id & 0x3fffffff;
  auto& barrier = barriers_.at(bar_index);
  barrier.events += count;
}

void BarrierUnit::event_release(uint32_t bar_id) {
  bool is_global = (bar_id >> 31);
  auto bar_index = bar_id & 0x3fffffff;
  auto& barrier = barriers_.at(bar_index);
  const auto& active_warps = scheduler_->active_warps();
  assert(barrier.events > 0);
  --barrier.events;
  if (barrier.events == 0) {
#ifdef VX_CFG_DSMEM_ENABLE
    if (!is_global && ((bar_id >> 30) & 1)) {
      if (barrier.count != 0 && barrier.arrival_mask.count() >= barrier.count) {
        // Last drain ack / expected transaction landed after all local
        // arrivals — escalate using any arrived warp's cluster identity.
        for (uint32_t i = 0; i < VX_CFG_NUM_WARPS; ++i) {
          if (barrier.arrival_mask.test(i)) {
            this->cluster_escalate(bar_id, i);
            break;
          }
        }
      }
      return;
    }
#endif
    if (is_global) {
      if (barrier.arrival_mask.count() == active_warps.count()) {
        uint32_t num_cores = barrier.count;
        // notify global barrier
        core_->cluster()->global_barrier_arrive(bar_id, num_cores, core_->id());
        barrier.arrival_mask.reset();
        barrier.count  = 0;
        barrier.events = 0;
      }
    } else {
      if (barrier.count == 0) {
        // resume waiting warps
        for (uint32_t i = 0; i < VX_CFG_NUM_WARPS; ++i) {
          if (barrier.wait_mask.test(i)) {
            DP(3, "*** Resume core #" << core_->id() << ", warp #" << i << " at barrier #" << bar_id);
            scheduler_->resume(i);
          }
        }
        // reset barrier and advance phase
        barrier.wait_mask.reset();
        ++barrier.phase;
      }
    }
  }
}
