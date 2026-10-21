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
#include <memory>
#include <vector>
#include "types.h"
#include "VX_types.h"
#ifdef VX_CFG_EXT_DLMEM_ENABLE
#include <dlmem_ref.h>
#endif

namespace vortex {

struct kmu_req_t {
  uint64_t PC;
  uint64_t entry;
  uint64_t param;
  uint32_t cta_id;
  uint32_t block_idx[3];
  uint32_t block_dim[3];
  uint32_t grid_dim[3];
  uint32_t lmem_size;
  uint32_t block_size;
  uint32_t warp_step[3];
  uint32_t cluster_dim[3];
  bool     is_first_of_cluster;
  // Launch identity: bumped on every start(), so consecutive launches of the
  // same kernel are still distinct contexts.
  uint8_t  ctx_id;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  uint32_t cluster_rank;
  uint32_t policy;          // KMU_POLICY_* from the launch's cluster-policy DCR
  // Stamped by the cluster's gang unit on a SPREAD block.
  bool     pinned;          // routed to dest_core, not balanced
  uint32_t dest_core;       // core within the hardware cluster
  uint32_t forced_slot;     // the CTA slot the gang reserved on dest_core
  uint32_t gang_id;         // gang_id and epoch: valid only for K > 1
  uint32_t epoch;
  dlmem_ref::Row row;       // the gang's peer row, the same in all K beats
#endif
};

#ifdef VX_CFG_EXT_DLMEM_ENABLE
// Same encodings as VX_gpu_pkg.
constexpr uint32_t KMU_POLICY_DEFAULT = 0;
constexpr uint32_t KMU_POLICY_SPREAD  = 1;
#endif

class LaunchTree;

// Kernel Management Unit. Holds the in-flight kernel descriptor, walks the
// grid, and fires one CTA per cycle into the launch bus.
class Kmu : public SimObject<Kmu> {
public:
  // Launch lanes, one per endpoint of the launch bus (a core, or with DLMEM a
  // cluster's gang unit), each bound to a registered boundary stage into the
  // endpoint.
  std::vector<SimChannel<kmu_req_t>> bus_out;

  // One credit per beat the endpoint takes.
  std::vector<SimChannel<uint8_t>> credit_in;

  Kmu(const SimContext& ctx, const char* name, uint32_t num_lanes);
  ~Kmu();

  void dcr_write(uint32_t addr, uint32_t value);

  // Called by ProcessorImpl::run() to arm a kernel launch (host context,
  // between cycles).
  void start();

  // True while CTAs remain to be issued or wait in the launch bus's skids.
  bool running() const;

  // A grid-less launch is a delegated draw launch: the KMU walks no CTAs and
  // the processor forwards the frame kick to the raster engines.
  bool launch_delegated() const {
    return !(grid_dim_[0] && grid_dim_[1] && grid_dim_[2]);
  }

  // Program image base PC (KMU STARTUP_ADDR) — where every warp begins
  // executing (__vx_cta_entry). Persists across reset; the Fragment Work
  // Distributor reuses it as the launch PC for injected fragment warps.
  uint64_t startup_pc() const { return PC_; }

protected:
  void on_reset();
  void on_tick();

private:
  // Fill *req with the next CTA and advance the walk.
  void next_cta(kmu_req_t* req);

  bool credits_idle() const;

  uint64_t PC_;
  uint64_t entry_;
  uint64_t param_;
  uint32_t block_dim_[3];
  uint32_t grid_dim_[3];
  uint32_t lmem_size_;
  uint32_t block_size_;
  uint32_t warp_step_[3];
  uint32_t cluster_dim_[3];
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  uint32_t policy_;
#endif
  bool     running_;
  uint32_t cta_id_;
  uint8_t  ctx_id_;
  // Nested counter: group_origin advances by cluster_dim per axis;
  // intra_offset walks within the cluster.
  uint32_t group_origin_[3];
  uint32_t intra_offset_[3];
  // A cluster is one message: its members stay on the output its first took.
  std::unique_ptr<LaunchTree> tree_;

  friend class SimObject<Kmu>;
};

} // namespace vortex
