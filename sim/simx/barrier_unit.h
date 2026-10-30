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

// rs1 bit 30 of a BAR names the block's cluster barrier; the low bits are its CTA slot.
constexpr uint32_t BAR_CLUSTER_BIT = 1u << 30;

// A CTA slot's cluster-barrier record, written by the dispatcher at admission.
struct CbarSlot {
  bool     gang    = false;  // a SPREAD gang of K > 1 completes through the gang unit
  uint32_t gang_id = 0;
  uint32_t epoch   = 0;
  uint32_t base    = 0;      // otherwise the cluster's first slot on this core
  uint32_t k       = 1;      // and its block count
};
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
#ifdef VX_CFG_EXT_DXA_ENABLE
  // A DXA copy's completion: one event of a barrier (its flat id) is released.
  SimEventLink<uint32_t> TxbarIn;
private:
  void on_txbar(const uint32_t& bar_id);
public:
#else
#line 58
#endif
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  // Sends a SPREAD block's done event to the cluster's gang unit, through the
  // same request queue the global barrier arrivals use.
  void send_cta_done(const CtaDone& done);
  // The gang unit released a cluster barrier phase.
  void cluster_release(uint32_t gang_id, uint32_t epoch);
  // A block was admitted into the slot: its cluster-barrier state restarts.
  void slot_admit(uint32_t slot, const CbarSlot& info);
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
  // Books a request queue departure; returns its cycle.
  uint64_t gbar_book(uint64_t now);
  void cbar_send(const CbarSlot& info);

  uint32_t cluster_phase(uint32_t bar_id) const;
  void cluster_arrive(uint32_t bar_id, uint32_t count, uint32_t wid, bool is_sync_bar);
  bool cluster_wait(uint32_t bar_id, uint32_t phase, uint32_t wid);
  void cluster_complete(uint32_t first, uint32_t count);

  // Per-slot cluster-barrier state, apart from the barrier store: the slot's
  // __syncthreads barrier stays usable between a cluster arrive and its wait.
  std::vector<CbarSlot> cinfo_;
  std::vector<uint32_t> ccount_;   // warps of the block arrived
  std::vector<uint32_t> cblk_;     // blocks arrived, at a same-core cluster's base slot
  std::vector<WarpMask> cwait_;    // suspended warps
  std::vector<bool>     cpend_;    // arrived, waiting for the gang unit's release
  std::vector<uint32_t> cphase_;

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
