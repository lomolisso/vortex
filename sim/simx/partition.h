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
#include "cluster.h"
#include "exec_domains.h"

namespace vortex {

class ProcessorImpl;

// One partition: it creates and holds its GPCs (Cluster); GPC g is the
// chip's cluster partition_id * GPCS_PER_PARTITION + g. A container only: no
// state, no tick, no cycle.
class Partition : public SimObject<Partition> {
public:
  Partition(const SimContext& ctx, const char* name, uint32_t partition_id, ProcessorImpl* processor);

  uint32_t id() const { return partition_id_; }

  const Cluster::Ptr& cluster(uint32_t g) const { return clusters_.at(g); }

  uint32_t num_clusters() const { return uint32_t(clusters_.size()); }

  // The execution domain of GPC g, for binding the channels that leave it.
  const ExecDomain& gpc_domain(uint32_t g) const { return domains_.at(g); }

private:
  uint32_t partition_id_;
  std::vector<Cluster::Ptr> clusters_;
  std::vector<ExecDomain>   domains_;

  friend class SimObject<Partition>;
};

} // namespace vortex
