// Copyright © 2019-2026
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
#include <string>
#include "regslice.h"

namespace vortex {

// The containers that can own an execution domain. H100 builds run each GPC
// (its CPCs, TPCs, SMs, L1s, L2, gang table and DLMEM crossbar) as one
// domain; other builds keep one domain per socket. Domain 0 holds the memory
// side, the KMU and its admission mirrors.
enum class DomainLevel { Gpc, Socket };

inline constexpr DomainLevel kDomainLevel =
  VX_CFG_H100_ORG_ENABLED ? DomainLevel::Gpc : DomainLevel::Socket;

static_assert(VX_CFG_DOMAIN_BOUNDARY_DELAY >= 1,
              "VX_CFG_DOMAIN_BOUNDARY_DELAY must be at least one cycle");

// Built where a container of `level` is created, in the enclosing domain's
// scope. Every channel between the container and the outside goes through
// stage() or link() below, so a container that owns a domain in this build
// gets registered crossings and one that does not gets today's direct binds.
class ExecDomain {
public:
  explicit ExecDomain(DomainLevel level)
    : outer_(SimPlatform::instance().build_domain())
    , inner_((level == kDomainLevel) ? SimPlatform::instance().alloc_domain() : outer_)
  {}

  bool boundary() const { return inner_ != outer_; }

  // Scope for the container and everything it builds.
  SimPlatform::DomainScope enter() const {
    return SimPlatform::DomainScope(inner_);
  }

  // Scope for objects the container builds on behalf of its parent.
  SimPlatform::DomainScope enter_outer() const {
    return SimPlatform::DomainScope(outer_);
  }

  // A crossing whose producer sends with no registered delay of its own (a
  // memory bus): a stage of the boundary delay owned by the producer's side,
  // the same register VX_gpc_boundary adds in RTL. `inbound` is the outside
  // to container direction.
  template <typename Pkt>
  void stage(const std::string& name, SimChannel<Pkt>& src, SimChannel<Pkt>& dst, bool inbound) const {
    if (!this->boundary()) {
      src.bind(&dst);
      return;
    }
    typename RegSlice<Pkt>::Ptr slice;
    {
      SimPlatform::DomainScope scope(inbound ? outer_ : inner_);
      slice = RegSlice<Pkt>::Create(name.c_str(), VX_CFG_DOMAIN_BOUNDARY_DELAY);
    }
    src.bind(&slice->In);
    slice->Out.bind(&dst);
  }

  // A crossing whose producer already sends with a registered delay (a
  // credit, a done event): that delay is the boundary stage, no cycle is
  // added. `src` belongs to the producer's side. The output of a register
  // stage is already the crossing, with its own credit pool on the endpoint's
  // pops: a second pool would take over that endpoint's credit return.
  template <typename Pkt>
  void link(SimChannel<Pkt>& src, SimChannel<Pkt>& dst) const {
    src.bind(&dst);
    if (this->boundary() && !src.boundary_stage()) {
      src.mark_crossing(VX_CFG_DOMAIN_BOUNDARY_DELAY);
    }
  }

private:
  uint32_t outer_;
  uint32_t inner_;
};

} // namespace vortex
