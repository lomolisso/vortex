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

// Ideal SM-to-SM transport — an explicit modeling PLACEHOLDER, deliberately
// NOT a future RTL module. It gives the DSMEM endpoints (LSU adapters,
// LocalMem remote ports, DXA multicast) point-to-point connectivity with a
// fixed delay and no topology, contention, or link accounting: NoC modeling
// for every Vortex fabric is delegated to the Omelet co-simulator
// (docs/proposals/dsmem/5__omelet_cosimulation.md). This class remains the
// fallback transport when co-simulation is off, so standalone simx keeps
// working without gem5.
//
// Contract (shared with the future Omelet bridge): requests arrive carrying
// their destination in MemFlags::dsmem_target and are delivered to that
// core's LocalMem remote port; responses back-route by the source index the
// transport appends to the request tag.
class IdealTransport : public SimObject<IdealTransport> {
public:
  std::vector<SimChannel<MemReq>> ReqIn;   // [core] from the core's DSMEM egress
  std::vector<SimChannel<MemRsp>> RspOut;  // [core] back to the requesting core

  std::vector<SimChannel<MemReq>> ReqOut;  // [core] into the core's LocalMem remote port
  std::vector<SimChannel<MemRsp>> RspIn;   // [core] from the core's LocalMem remote port

  IdealTransport(const SimContext& ctx, const char* name, uint32_t num_cores);

protected:
  void on_reset();
  void on_tick();

private:
  uint32_t num_cores_;
  uint32_t lg2_cores_;
  std::vector<uint32_t> req_rr_;  // per-destination rotating pick (fairness only)
  std::vector<uint32_t> rsp_rr_;  // per-requester rotating pick (fairness only)

  friend class SimObject<IdealTransport>;
};

} // namespace vortex
