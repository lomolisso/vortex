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

#include "cpc.h"
#include "cluster.h"
#include "constants.h"

using namespace vortex;

Cpc::Cpc(const SimContext& ctx, const char* name, uint32_t cpc_id, Cluster* cluster)
  : SimObject<Cpc>(ctx, name)
  , cpc_id_(cpc_id)
  , sockets_(NUM_SOCKETS / VX_CFG_CPCS_PER_GPC)
{
  char sname[100];
  for (uint32_t t = 0; t < sockets_.size(); ++t) {
    uint32_t i = t * VX_CFG_CPCS_PER_GPC + cpc_id;
    uint32_t socket_id = cluster->id() * NUM_SOCKETS + i;
    snprintf(sname, 100, "%s-socket%d", cluster->name().c_str(), i);
    sockets_.at(t) = Socket::Create(sname, socket_id, cluster);
  }
}
