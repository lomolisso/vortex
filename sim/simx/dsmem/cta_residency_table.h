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

// One directory entry: which core hosts a cluster rank and at what LMEM base.
struct CtaRankSlot {
  bool     valid     = false;
  uint32_t core_id   = 0;  // global core id hosting this rank
  uint32_t cta_slot  = 0;  // rank's local CTA slot on its core (bar_id cta_no)
  uint64_t lmem_base = 0;  // rank's slice base within its core's LMEM window
};

// CTA-cluster residency MASTER directory (one per hardware cluster; future
// VX_cta_table in VX_cluster.sv): records, for each in-flight CTA cluster, which
// core hosts each rank and at what LMEM base. Written by the per-core
// CtaDispatchers at admission (via Cluster::cta_admit). The master owns the
// control plane — admission gate (`can_admit`), participant mask + rank count
// for the cluster barrier — and, once a cluster is fully placed, its entries are
// broadcast to the participating SMs' CtaReplica caches (Task 5). Hot-path
// resolves (DSMEM remote access, DXA fan-out) read the local replica, NOT this
// master, so there is no shared-structure read contention.
class CtaResidencyTable {
public:
  using RankSlot = CtaRankSlot;

  explicit CtaResidencyTable(uint32_t num_entries);

  // Admission handshake for the dispatcher's placement gate: true when the
  // cluster already has an open entry or a free entry exists (the RTL
  // table_notfull condition; a full table backpressures rank 0).
  bool can_admit(uint32_t cluster_id) const;

  void admit(uint32_t cluster_id, uint32_t rank, uint32_t num_ranks,
             uint32_t core_id, uint32_t cta_slot, uint64_t lmem_base);

  // True once every rank of the cluster has been admitted (placement complete);
  // this is when the master broadcasts the entry set to the replicas.
  bool fully_admitted(uint32_t cluster_id) const;

  // Called when the last warp of a rank's CTA exits; the entry frees once
  // every admitted rank has retired. Returns true when that final retire frees
  // the cluster entry (so the caller can evict the replicas).
  bool retire(uint32_t cluster_id, uint32_t rank);

  // nullptr while the rank is not resident. (Master-side lookup — used only by
  // the broadcast and control plane; the hot path uses CtaReplica::resolve.)
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

// Per-SM READ replica of the residency directory (future VX_cta_replica in
// VX_core.sv). Holds only the clusters this SM participates in; filled by the
// master's broadcast at full-cluster admission (Cluster::cta_admit) and evicted
// at cluster teardown. `resolve` is a purely local read — a small associative
// lookup with a fixed latency modeled by the consumer
// (VX_CFG_CTA_TABLE_LOOKUP_LATENCY, charged on the DSMEM egress). Because every
// participating SM holds its own copy and entries are immutable admit->retire,
// there is no cross-SM read contention and replication is trivially coherent.
class CtaReplica {
public:
  using RankSlot = CtaRankSlot;

  explicit CtaReplica(uint32_t num_entries);

  // Broadcast fill: install rank's slot for cluster_id (allocates the cluster's
  // slot in the replica on first rank).
  void install(uint32_t cluster_id, uint32_t rank, uint32_t num_ranks,
               const RankSlot& slot);

  // Drop a whole cluster's entry (called at cluster teardown).
  void evict(uint32_t cluster_id);

  // Local associative read; nullptr when the rank is not resident here.
  const RankSlot* resolve(uint32_t cluster_id, uint32_t rank) const;

  void reset();

private:
  struct Entry {
    bool     valid      = false;
    uint32_t cluster_id = 0;
    uint32_t num_ranks  = 0;
    std::vector<RankSlot> ranks;
  };
  const Entry* find(uint32_t cluster_id) const;
  Entry* find(uint32_t cluster_id);

  std::vector<Entry> entries_;
};

} // namespace vortex
