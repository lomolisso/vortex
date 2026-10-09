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

#pragma once

#include <deque>
#include <vector>
#include "types.h"

namespace vortex {

class Core;
class Scheduler;

#ifdef VX_CFG_EXT_DLMEM_ENABLE
struct CtaDone;

// GbarArrive kinds, the GBAR_KIND_* of VX_gpu_pkg.
constexpr uint32_t GBAR_KIND_BAR      = 0;
constexpr uint32_t GBAR_KIND_CTA_DONE = 1;
constexpr uint32_t GBAR_KIND_CBAR     = 2;
#endif

// Owns per-warp barrier state; resumes waiting warps via the parent Scheduler;
// participates in cross-core global barriers via Socket.
class BarrierUnit : public SimObject<BarrierUnit> {
public:
  BarrierUnit(const SimContext& ctx, const char* name, Core* core, Scheduler* scheduler);
  ~BarrierUnit();

  uint32_t get_phase(uint32_t bar_id) const;
  void arrive(uint32_t bar_id, uint32_t count, uint32_t wid, bool is_sync_bar);
  bool wait(uint32_t bar_id, uint32_t phase, uint32_t wid);
  void global_resume(uint32_t bar_id);
  void event_attach(uint32_t bar_id, uint32_t count = 1);
  void event_release(uint32_t bar_id);
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // Sends a SPREAD block's done event to the cluster's gang unit, through the
  // same request queue the global barrier arrivals use.
  void send_cta_done(const CtaDone& done);
#endif

protected:
  void on_reset();
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  void on_tick();
#endif

private:
  // RTL holds completed global arrivals in a depth-VX_CFG_NUM_BARRIERS queue
  // that drains at most one entry per cycle onto the gbar bus. Schedule each
  // send on its own cycle so back-to-back completions serialize here too.
  uint64_t gbar_send_next_;

  void gbar_send(uint32_t bar_id, uint32_t count);

#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // A done event enters the RTL queue only cycles after the warp exit, and a
  // barrier arrival in between enters first, so its departure is booked only
  // once its push cycle has passed.
  struct PendingDone {
    uint64_t push;
    uint32_t slot;
    uint32_t gang_id;
    uint32_t epoch;
  };
  std::deque<PendingDone> done_q_;

  void book_dones(uint64_t now);
#endif

  Core*       core_;
  Scheduler*  scheduler_;
  std::vector<warp_barrier_t> barriers_;

  friend class SimObject<BarrierUnit>;
};

}
