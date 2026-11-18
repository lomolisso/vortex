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

#include <cstdint>
#include <deque>
#include <vector>
#include <dlmem_ref.h>
#include "types.h"
#include "kmu.h"
#include "cta_gang.h"
#include "launch_tree.h"

namespace vortex {

// A placed block's CTA_DONE as the admission mirror of its GPC receives it,
// after the CTA-done link (VX_CFG_CTA_DONE_LINK_DELAY stages from the gbar unit).
struct AdmitDone {
  uint32_t core;     // in-GPC index
  uint32_t slot;
  uint32_t gang_id;  // gang_id and epoch: valid only for K > 1
  uint32_t epoch;
};

// Cluster admission of the chip (VX_cta_admit), one tick() per cycle: the
// chip-wide in-order queue of the KMU's stream and one exact mirror per GPC. A plain beat passes through. A cluster message
// (SPREAD policy, or an explicit-cluster launch with distinct SMs) is placed
// all or nothing in the first GPC from the round-robin pointer that can place
// it now (dlmem_ref::place_chip), one greedy step per cycle in every GPC at
// once, and leaves as K pinned beats. The pointer restarts at
// CLUSTER_GPC_START with every launch.
class CtaAdmit {
public:
  struct Inputs {
    const kmu_req_t* head = nullptr;            // the beat on the input, or none
    bool out_ready = false;                     // skid: the consumer takes the output beat; else the next stage takes a beat
    std::vector<const AdmitDone*> done;         // per GPC: this cycle's CTA_DONE, or none
    std::vector<std::vector<bool>> rel_pend;    // per GPC: its gang table's pending releases as of this cycle
  };

  struct Outputs {
    bool      in_ready = false;                 // the head beat is taken this cycle
    bool      out_valid = false;                // skid: the output beat; else a beat offered to the next stage
    kmu_req_t out{};
    uint32_t  out_dest = 0;                     // global SM index of a pinned beat
    bool      out_pinned = false;
    bool      out_eop = false;
    bool      alloc = false;                    // a gang id was taken this cycle (in alloc_gpc)
    uint32_t  alloc_gpc = 0;
    uint32_t  alloc_gang = 0;
    uint32_t  alloc_epoch = 0;
    uint32_t  alloc_k = 0;
  };

  // skid: the outputs are VX_cta_admit's ports, behind its two-entry output
  // skid (OUT_BUF=3); without it, the owner models the skid.
  explicit CtaAdmit(const dlmem_ref::Cfg& cfg, bool skid = true);

  void reset();

  Outputs tick(const Inputs& in);

  // The launch's placed blocks belong to gangs (K >= 2).
  bool gang_mode() const { return placed_ && k_ > 1; }

  // Mirrors not empty, a gang live, a placement in progress or a beat in the skid.
  bool busy() const;

private:
  enum class State { IDLE, PLACE, EMIT };

  struct SkidBeat {
    kmu_req_t req;
    uint32_t  dest;
    bool      pinned;
    bool      eop;
  };

  struct Gpc {
    std::vector<uint64_t> free, sfree;
    std::vector<uint32_t> used, sused;
    std::vector<bool>     visited;
    std::vector<bool>     live;
    std::vector<uint32_t> cnt;
    std::vector<uint32_t> epoch;
    std::vector<dlmem_ref::Peer> members;
    uint32_t rr = 0;
    uint32_t start = 0;
    bool     out = false;
  };

  dlmem_ref::Cfg   cfg_;
  bool             skid_;
  std::deque<SkidBeat> q_;
  std::vector<Gpc> gpc_;
  uint32_t         gptr_;
  State            state_;
  uint32_t         step_;
  uint32_t         win_;
  uint32_t         cur_gang_;
  uint32_t         cur_epoch_;
  bool             failed_;
  bool             dirty_;
  // per-launch latches
  uint8_t          ctx_;
  bool             placed_;
  uint32_t         k_;
  uint32_t         w_;
  uint32_t         u_;
};

// The admission in the simulator: KMU lane in, one launch lane per GPC out
// (its output skid, then the 1 -> NUM_CLUSTERS fan-out, as the RTL's kmu_arb),
// and per GPC the gang table's events, pending releases and gang-id grants.

class AdmitUnit : public SimObject<AdmitUnit> {
public:
  SimChannel<kmu_req_t> bus_in;
  SimChannel<uint8_t>   credit_out;

  std::vector<SimChannel<kmu_req_t>> bus_out;
  std::vector<SimChannel<uint8_t>>   credit_in;

  // per GPC: every gang event (CTA_DONE and CBAR, in its queue order), its
  // gang table's pending-release changes, and the grants to the table
  std::vector<SimChannel<GangEvent>>   evt_in;
  std::vector<SimChannel<GangRelPend>> relp_in;
  std::vector<SimChannel<GangAlloc>>   alloc_out;

  // direct: a GPC of one core, whose lane is the core's own registered stage
  AdmitUnit(const SimContext& ctx, const char* name, const dlmem_ref::Cfg& cfg, bool direct);

  bool busy() const;

protected:
  void on_reset();
  void on_tick();

private:
  dlmem_ref::Cfg cfg_;
  bool           direct_;
  CtaAdmit       admit_;
  LaunchTree     out_;
  std::vector<std::deque<GangEvent>> evt_q_;
  std::vector<std::vector<bool>>     rel_pend_;
  bool           mode_;
  uint64_t       link_delay_;

  friend class SimObject<AdmitUnit>;
};

} // namespace vortex
