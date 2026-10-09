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
#include <algorithm>
#include "core.h"
#include "scheduler.h"
#include "socket.h"
#include "debug.h"

using namespace vortex;

BarrierUnit::BarrierUnit(const SimContext& ctx, const char* name, Core* core, Scheduler* scheduler)
  : SimObject<BarrierUnit>(ctx, name)
  , gbar_send_next_(0)
  , core_(core)
  , scheduler_(scheduler)
  , barriers_(VX_CFG_NUM_WARPS * VX_CFG_NUM_BARRIERS)
{}

BarrierUnit::~BarrierUnit() {}

// One request leaves the queue per cycle, so a completion that lands while the
// queue is still draining is delivered on the next free cycle rather than
// immediately. Mirrors the RTL request FIFO's drain rate.
void BarrierUnit::gbar_send(uint32_t bar_id, uint32_t count) {
  uint64_t now = SimPlatform::instance().cycles();
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  this->book_dones(now);
  // the arrival takes the queue's write port: a done event due to enter this
  // cycle, and those behind it, enter a cycle later
  uint64_t next_push = now + 1;
  for (auto& d : done_q_) {
    d.push = std::max(d.push, next_push);
    next_push = d.push + 1;
  }
#endif
  uint64_t due = std::max(now + 1, gbar_send_next_);
  gbar_send_next_ = due + 1;
  core_->gbar_arrive_out.send({bar_id, count, core_->id()}, due - now);
}

#ifdef VX_CFG_EXT_DLMEM_ENABLE
// RTL path of a done event, from the last warp's exit: the dispatcher's
// retirement pipe (2) and its pending register (1) push it into the request
// queue; after the queue, one arbiter stage per fan-in level that exists, and
// the gbar unit's output register.
static constexpr uint64_t CTA_DONE_PUSH_DELAY = 3;
static constexpr uint64_t CTA_DONE_LINK_DELAY = 1 + (VX_CFG_SOCKET_SIZE > 1 ? 1 : 0) + (NUM_SOCKETS > 1 ? 1 : 0);

void BarrierUnit::send_cta_done(const CtaDone& done) {
  uint64_t now  = SimPlatform::instance().cycles();
  uint64_t push = now + CTA_DONE_PUSH_DELAY;
  if (!done_q_.empty()) {
    push = std::max(push, done_q_.back().push + 1);
  }
  done_q_.push_back({push, done.slot, done.gang_id, done.epoch});
  this->tick_wake();
}

// Books, in push order, the done events whose push cycle has passed: every
// arrival that could have entered the queue ahead of them is known by now.
void BarrierUnit::book_dones(uint64_t now) {
  while (!done_q_.empty() && done_q_.front().push < now) {
    const auto& d = done_q_.front();
    uint64_t due = std::max(d.push + 1, gbar_send_next_);
    assert(due >= now);
    gbar_send_next_ = due + 1;
    GbarArrive msg{0, 0, core_->id()};
    msg.kind    = GBAR_KIND_CTA_DONE;
    msg.gang_id = d.gang_id;
    msg.epoch   = d.epoch;
    msg.slot    = d.slot;
    core_->gbar_arrive_out.send(msg, (due - now) + CTA_DONE_LINK_DELAY);
    done_q_.pop_front();
  }
}

void BarrierUnit::on_tick() {
  this->book_dones(SimPlatform::instance().cycles());
  if (done_q_.empty()) {
    this->tick_sleep();
  }
}
#endif

void BarrierUnit::on_reset() {
  gbar_send_next_ = 0;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  done_q_.clear();
#endif
  for (auto& barrier : barriers_) {
    barrier.reset();
  }
}

uint32_t BarrierUnit::get_phase(uint32_t bar_id) const {
  auto bar_index = bar_id & 0x7fffffff;
  return barriers_.at(bar_index).phase;
}

void BarrierUnit::arrive(uint32_t bar_id, uint32_t count, uint32_t wid, bool is_sync_bar) {
  bool is_global = (bar_id >> 31);
  auto bar_index = bar_id & 0x7fffffff;

  auto& barrier = barriers_.at(bar_index);
  const auto& active_warps = scheduler_->active_warps();

  DP(4, "*** Barrier arrive: core #" << core_->id() << ", warp #" << wid << " at barrier #" << bar_id << ", phase=" << barrier.phase << ", wait_mask=" << barrier.wait_mask << ", count=" << barrier.count << ", events=" << barrier.events);

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
      this->gbar_send(bar_id, count);
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
      barrier.phase ^= 1u;
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
  auto bar_index = bar_id & 0x7fffffff;

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
  auto bar_index = bar_id & 0x7fffffff;
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
  barrier.phase ^= 1u;
}

void BarrierUnit::event_attach(uint32_t bar_id, uint32_t count) {
  auto bar_index = bar_id & 0x7fffffff;
  auto& barrier = barriers_.at(bar_index);
  barrier.events += count;
}

void BarrierUnit::event_release(uint32_t bar_id) {
  bool is_global = (bar_id >> 31);
  auto bar_index = bar_id & 0x7fffffff;
  auto& barrier = barriers_.at(bar_index);
  const auto& active_warps = scheduler_->active_warps();
  assert(barrier.events > 0);
  --barrier.events;
  if (barrier.events == 0) {
    if (is_global) {
      if (barrier.arrival_mask.count() == active_warps.count()) {
        uint32_t num_cores = barrier.count;
        this->gbar_send(bar_id, num_cores);
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
        barrier.phase ^= 1u;
      }
    }
  }
}
