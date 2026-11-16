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

#include "partition.h"
#include "constants.h"
#include "exec_domains.h"

using namespace vortex;

Partition::Partition(const SimContext& ctx, const char* name, uint32_t partition_id, ProcessorImpl* processor)
  : SimObject<Partition>(ctx, name)
  , partition_id_(partition_id)
  , clusters_(VX_CFG_NUM_CLUSTERS / VX_CFG_NUM_PARTITIONS)
{
  char sname[100];
  for (uint32_t g = 0; g < clusters_.size(); ++g) {
    uint32_t cluster_id = partition_id * clusters_.size() + g;
    snprintf(sname, 100, "cluster%d", cluster_id);
    // the GPC's own execution domain in H100 builds, everything it builds included
    domains_.emplace_back(DomainLevel::Gpc);
    auto s = domains_.back().enter();
    clusters_.at(g) = Cluster::Create(sname, cluster_id, processor);
  }
}
