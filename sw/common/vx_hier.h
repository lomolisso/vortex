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

// Compute levels and the global SM index (%smid), shared by the host, SimX and
// the RTL benches; VX_gpu_pkg's sm_* functions decode the same way. The shape
// comes from the caller (the device caps or the build), never VX_config.h.

#include <cstdint>

namespace vortex {
namespace hier {

struct Shape {
  uint32_t partitions;          // P
  uint32_t gpcs_per_partition;  // G
  uint32_t cpcs_per_gpc;        // C
  uint32_t tpcs_per_cpc;        // T
  uint32_t sms_per_tpc;         // S2

  uint32_t num_gpcs() const { return partitions * gpcs_per_partition; }
  uint32_t sms_per_gpc() const { return cpcs_per_gpc * tpcs_per_cpc * sms_per_tpc; }
  uint32_t num_sms() const { return this->num_gpcs() * this->sms_per_gpc(); }
};

// Today's knobs as levels: NUM_CLUSTERS GPCs of NUM_CORES SMs in SOCKET_SIZE-SM TPCs.
inline Shape shape_of(uint32_t partitions, uint32_t num_clusters, uint32_t cpcs_per_gpc,
                      uint32_t num_cores, uint32_t socket_size) {
  Shape s;
  s.partitions         = partitions;
  s.gpcs_per_partition = num_clusters / partitions;
  s.cpcs_per_gpc       = cpcs_per_gpc;
  s.tpcs_per_cpc       = num_cores / (socket_size * cpcs_per_gpc);
  s.sms_per_tpc        = socket_size;
  return s;
}

struct Pos {
  uint32_t partition;
  uint32_t gpc;        // in its partition
  uint32_t cpc;        // CPC row
  uint32_t tpc_slot;
  uint32_t parity;     // SM in its TPC (lsm0 / lsm1)

  bool operator==(const Pos& o) const {
    return partition == o.partition && gpc == o.gpc && cpc == o.cpc
        && tpc_slot == o.tpc_slot && parity == o.parity;
  }
};

inline uint32_t gpc_global(const Shape& s, uint32_t sm) {
  return sm / s.sms_per_gpc();
}

// Index inside the GPC: TPC-slot-major, CPC-minor, so the lowest indices take one TPC of every CPC first.
inline uint32_t in_gpc(const Shape& s, const Pos& p) {
  return (p.tpc_slot * s.cpcs_per_gpc + p.cpc) * s.sms_per_tpc + p.parity;
}

inline uint32_t encode(const Shape& s, const Pos& p) {
  uint32_t gpc = p.partition * s.gpcs_per_partition + p.gpc;
  return gpc * s.sms_per_gpc() + in_gpc(s, p);
}

inline Pos decode(const Shape& s, uint32_t sm) {
  uint32_t gpc = gpc_global(s, sm);
  uint32_t i   = sm % s.sms_per_gpc();
  uint32_t tpc = i / s.sms_per_tpc;
  Pos p;
  p.partition = gpc / s.gpcs_per_partition;
  p.gpc       = gpc % s.gpcs_per_partition;
  p.cpc       = tpc % s.cpcs_per_gpc;
  p.tpc_slot  = tpc / s.cpcs_per_gpc;
  p.parity    = i % s.sms_per_tpc;
  return p;
}

// TPC slots between an SM and the L2 block / SM-to-SM hub: the highest slot is nearest (01/spec.md §7.2).
inline uint32_t slot_dist(const Shape& s, uint32_t sm) {
  return s.tpcs_per_cpc - 1 - decode(s, sm).tpc_slot;
}

} // namespace hier
} // namespace vortex
