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

#include <vector>
#include "types.h"
#include "socket.h"

namespace vortex {

class Cluster;

// One CPC row of a GPC: it creates and holds its TPCs (Socket), in TPC-slot
// order; TPC slot t is the GPC's socket t * CPCS_PER_GPC + cpc. A container
// only: no state, no tick, no cycle.
class Cpc : public SimObject<Cpc> {
public:
  Cpc(const SimContext& ctx, const char* name, uint32_t cpc_id, Cluster* cluster);

  uint32_t id() const { return cpc_id_; }

  const Socket::Ptr& socket(uint32_t tpc_slot) const { return sockets_.at(tpc_slot); }

  uint32_t num_sockets() const { return uint32_t(sockets_.size()); }

private:
  uint32_t cpc_id_;
  std::vector<Socket::Ptr> sockets_;

  friend class SimObject<Cpc>;
};

} // namespace vortex
