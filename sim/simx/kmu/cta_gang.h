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

#pragma once

#include <cstdint>
#include <deque>
#include <vector>
#include <dlmem_ref.h>
#include "types.h"
#include "kmu.h"
#include "launch_tree.h"

namespace vortex {

// The build's hardware-cluster shape, as the placement model sees it.
dlmem_ref::Cfg dlmem_cfg();

// An event of a SPREAD block toward its gang unit, in its core's queue order.
struct GangEvent {
  enum Kind { CTA_DONE, CBAR };
  Kind     kind;
  uint32_t core;
  uint32_t slot;      // CTA_DONE: the freed slot
  uint32_t gang_id;
  uint32_t epoch;
  uint32_t count;     // CBAR: the gang's block count
};

// Gang unit of one hardware cluster (VX_cta_gang): it sits between the
// cluster's launch lane and its cores.
//
// A DEFAULT launch passes through; the cluster's fan-out picks its core. A SPREAD
// message's K blocks are placed all or nothing, spread first, on an exact
// mirror of every core's free CTA slots and used warps, one greedy step per
// cycle, and then leave as K pinned beats, one per cycle. A slot returns to
// the mirror only when its block's CTA_DONE event arrives, so the slot a beat
// forces is always free when the beat reaches its core.
//
// It also counts the gang's cluster-barrier arrivals (CBAR events) and, when
// all K blocks arrived, broadcasts the phase's release to every core.
class CtaGang : public SimObject<CtaGang> {
public:
  // The cluster's launch-lane endpoint, and its admission credit to the KMU.
  SimChannel<kmu_req_t> bus_in;
  SimChannel<uint8_t>   credit_out;

  // The unit's output skid, into the cluster's launch fan-out (or the core of
  // a one-core cluster), and one credit per beat taken from it.
  SimChannel<kmu_req_t> bus_out;
  SimChannel<uint8_t>   credit_in;

  // Cluster-barrier releases, one link per core.
  std::vector<SimEventLink<GbarResume>> release_out;

  CtaGang(const SimContext& ctx, const char* name, const dlmem_ref::Cfg& cfg);

  // Always accepted, applied at most one per cycle like the RTL's single event port.
  void cta_event(const GangEvent& evt);

  // Busy until every placed block has reported done and every release has
  // left, so the next launch starts on an empty mirror.
  bool busy() const;

  // Cycles from the release leaving this unit to its arrival at a core's barrier unit.
  uint64_t release_delay() const { return rel_delay_; }

protected:
  void on_reset();
  void on_tick();

private:
  enum class State { IDLE, PLACE, EMIT };

  bool out_ready(const LaunchTree::Beat& beat) const;
  void send_beat(const LaunchTree::Beat& beat);
  void apply_done(const GangEvent& d);
  void apply_cbar(const GangEvent& a);
  bool id_free() const;
  void forward_legacy();
  void start_attempt(const kmu_req_t& head);
  void commit();
  void emit();

  dlmem_ref::Cfg cfg_;

  // Live mirror and gang-id pool.
  dlmem_ref::Mirror  mirror_;
  dlmem_ref::GangIds gids_;
  uint32_t           blocks_;     // placed blocks still owing their done event

  State    state_;
  uint32_t step_;                 // PLACE: greedy step; EMIT: member being sent
  uint32_t fail_step_;            // PLACE: the step that fails, or K on success
  std::vector<dlmem_ref::Peer> members_;
  dlmem_ref::Mirror  snap_;       // the mirror after a successful placement
  uint32_t cur_gang_;
  uint32_t cur_epoch_;
  bool     failed_;
  bool     dirty_;                // a block finished since the last attempt started

  // Per-launch latches: the done events of a launch use its K and W.
  uint8_t  ctx_;
  uint32_t policy_;
  uint32_t k_;
  uint32_t w_;

  std::deque<GangEvent> evt_q_;

  LaunchTree out_;
  bool       direct_;             // no fan-out: the output skid is the core's

  uint64_t rel_delay_;

  friend class SimObject<CtaGang>;
};

} // namespace vortex
