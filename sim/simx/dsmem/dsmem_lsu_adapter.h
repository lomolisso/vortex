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

#include <simobject.h>
#include "types.h"

namespace vortex {

class Core;

// LSU-side DSMEM adapter (future VX_dsmem_lsu_adapter in VX_core.sv): turns
// the LocalMemSwitch's remote-shared lanes — still in (region, offset)
// aperture form — into per-lane MemReqs on the core's network egress port.
// Per lane it resolves region → absolute cluster rank → {core, slice base}
// via the CTA residency table, rebases the address to a plain shared address
// on the peer, and requests a store ack (strsp) used for the cluster
// barrier's drain accounting. Lane and warp indices ride the low tag bits so
// load data returns to the right lane and store acks decrement the right
// warp's pending counter.
class DsmemLsuAdapter : public SimObject<DsmemLsuAdapter> {
public:
  SimChannel<LsuReq> ReqIn;   // from LocalMemSwitch (remote-shared lanes)
  SimChannel<LsuRsp> RspOut;  // load data back to the switch

  SimChannel<MemReq> ReqOut;  // to the core's DSMEM egress (network)
  SimChannel<MemRsp> RspIn;   // from the network (load data + store acks)

  DsmemLsuAdapter(const SimContext& ctx, const char* name, Core* core);

protected:
  void on_reset();
  void on_tick();

private:
  Core*       core_;
  uint32_t    lg2_lanes_;
  uint32_t    lg2_warps_;
  BitVector<> pending_mask_;

  friend class SimObject<DsmemLsuAdapter>;
};

} // namespace vortex
