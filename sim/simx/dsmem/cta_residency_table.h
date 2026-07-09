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
#include <vector>
#include <VX_config.h>

namespace vortex {

// CTA-cluster residency table (future VX_cta_table in VX_cluster.sv): records,
// for each in-flight CTA cluster, which core hosts each rank and at what LMEM
// base. Written by the per-core CtaDispatchers at admission; read by the DSMEM
// remote-address path, the DXA multicast fan-out, and the cluster barrier.
class CtaResidencyTable {
public:
  struct RankSlot {
    bool     valid     = false;
    uint32_t core_id   = 0;  // global core id hosting this rank
    uint32_t cta_slot  = 0;  // rank's local CTA slot on its core (bar_id cta_no)
    uint64_t lmem_base = 0;  // rank's slice base within its core's LMEM window
  };

  explicit CtaResidencyTable(uint32_t num_entries);

  // Admission handshake for the dispatcher's placement gate: true when the
  // cluster already has an open entry or a free entry exists (the RTL
  // table_notfull condition; a full table backpressures rank 0).
  bool can_admit(uint32_t cluster_id) const;

  void admit(uint32_t cluster_id, uint32_t rank, uint32_t num_ranks,
             uint32_t core_id, uint32_t cta_slot, uint64_t lmem_base);

  // Called when the last warp of a rank's CTA exits; the entry frees once
  // every admitted rank has retired.
  void retire(uint32_t cluster_id, uint32_t rank);

  // nullptr while the rank is not resident.
  const RankSlot* resolve(uint32_t cluster_id, uint32_t rank) const;

  // Bitmask of cores hosting at least one live rank of the cluster.
  uint64_t participant_mask(uint32_t cluster_id) const;

  uint32_t num_ranks(uint32_t cluster_id) const;

  void reset();

private:
  struct Entry {
    bool     valid      = false;
    uint32_t cluster_id = 0;
    uint32_t num_ranks  = 0;
    uint32_t admitted   = 0;
    uint32_t retired    = 0;
    std::vector<RankSlot> ranks;
  };

  const Entry* find(uint32_t cluster_id) const;
  Entry* find(uint32_t cluster_id);

  std::vector<Entry> entries_;
};

} // namespace vortex
