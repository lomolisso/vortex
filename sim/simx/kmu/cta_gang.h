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

namespace vortex {

// The build's GPC shape, as the placement model sees it.
dlmem_ref::Cfg dlmem_cfg();

// The dispatcher's slot pitch: the footprint rounded up to a memory block.
uint32_t slot_stride(const kmu_req_t& req);

// An event of a placed block toward its GPC's gang table and the admission, in its core's queue order.
struct GangEvent {
  enum Kind { CTA_DONE, CBAR };
  Kind     kind;
  uint32_t core;
  uint32_t slot;      // CTA_DONE: the freed slot
  uint32_t gang_id;
  uint32_t epoch;
  uint32_t count;     // CBAR: the gang's block count
  bool     arrived = false;  // CTA_DONE: the block's arrival counts in the open phase
};

// A gang table's pending-release register, as the admission reads it.
struct GangRelPend {
  std::vector<bool> pend;
};

// From the admission to a gang table: a gang id taken, or the launch's gang mode.
struct GangAlloc {
  bool     mode;    // a mode update: k > 1 is gang mode on
  uint32_t gang;
  uint32_t epoch;
  uint32_t k;
};

// Gang table of one GPC (VX_cta_gang), beside its gbar unit: the
// cluster-barrier half of gang admission; the placement half is the chip's
// admission (CtaAdmit, next to the KMU), which takes the gang ids and tells
// this table one cycle later.
//
// It counts the gangs' cluster-barrier arrivals (CBAR events) and, when every
// live block arrived, broadcasts the phase's release to every core; a block
// that ends counts as arrived. An id stays taken until its last release has
// left, which the admission learns from relp_out.
class CtaGang : public SimObject<CtaGang> {
public:
  // Cluster-barrier releases, one link per core.
  std::vector<SimEventLink<GbarResume>> release_out;

  SimChannel<GangAlloc>   alloc_in;
  SimChannel<GangRelPend> relp_out;

  CtaGang(const SimContext& ctx, const char* name, const dlmem_ref::Cfg& cfg);

  // Always accepted, applied at most one per cycle like the RTL's single event port.
  void cta_event(const GangEvent& evt);

  // Busy until every release has left.
  bool busy() const;

  // Cycles from the release leaving this unit to its arrival at a core's barrier unit.
  uint64_t release_delay() const { return rel_delay_; }

protected:
  void on_reset();
  void on_tick();

private:
  void apply_cbar(const GangEvent& a);
  void done_phase(const GangEvent& d, bool before);

  dlmem_ref::Cfg     cfg_;
  dlmem_ref::GangIds gids_;
  std::vector<uint32_t> gk_;       // each gang's block count
  bool               mode_;        // the launch's blocks belong to gangs
  std::deque<GangEvent> evt_q_;
  std::vector<bool>  relp_sent_;   // the pending-release bits the admission was last told
  uint64_t           rel_delay_;
  uint64_t           relp_delay_;
  uint32_t           done_cnt_ = 0; // the live count a done event found

  // Each gang's phase parity, flipped by its releases, and the parity of each
  // block's last arrival: a block that ends after arriving still counts only
  // while that phase is open.
  std::vector<bool> gph_;
  std::vector<std::vector<bool>> sph_;

  friend class SimObject<CtaGang>;
};

} // namespace vortex
