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

#include <memory>
#include <simobject.h>
#include "types.h"
#include "dxa_unit.h"

namespace vortex {

class Core;
class Cluster;

// Per-core (per-SM) DXA engine. Consumes DxaReq packets from this core's
// DxaUnit, dispatches to VX_CFG_NUM_DXA_UNITS workers, and drives GMEM reads
// (through the core's L1→L2 arbiter) and LMEM writes (into this core's own
// LocalMem, carrying real MemReq packets with TLM payload + completion flag).
class DxaCore : public SimObject<DxaCore> {
public:
  using Ptr = std::shared_ptr<DxaCore>;

  struct PerfStats {
    uint64_t transfers      = 0;
    uint64_t gmem_reads     = 0;
    uint64_t gmem_dedup     = 0;
    uint64_t lmem_writes    = 0;
    uint64_t total_latency  = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      transfers     += rhs.transfers;
      gmem_reads    += rhs.gmem_reads;
      gmem_dedup    += rhs.gmem_dedup;
      lmem_writes   += rhs.lmem_writes;
      total_latency += rhs.total_latency;
      return *this;
    }
  };

  // DxaReq input (size = 1, per-core). The owning Core binds its
  // SfuUnit::dxa_req_out here.
  std::vector<SimChannel<DxaReq>>  dxa_req_in;

  // GMEM ports to L2 (size = kDxaMemPorts). Internally fed by gmem_arb_
  // from VX_CFG_NUM_DXA_UNITS worker outputs. The owning Core binds these
  // into its L1→L2 arbiter (below the demand traffic).
  std::vector<SimChannel<MemReq>>  gmem_req_out;
  std::vector<SimChannel<MemRsp>>  gmem_rsp_in;
  MemArbiter::Ptr                  gmem_arb_;

  // LMEM write port (size = 1, per-core). The owning Core binds its own
  // LocalMem::Inputs[port_dxa] here. Write-only — no rsp.
  std::vector<SimChannel<MemReq>>  lmem_req_out;

  DxaCore(const SimContext& ctx, const char* name, Core* core);
  virtual ~DxaCore();

  int dcr_write(uint32_t addr, uint32_t value);

  const PerfStats& perf_stats() const;

protected:
  void on_reset();
  void on_tick();

private:
  class Impl;
  Impl* impl_;

  friend class SimObject<DxaCore>;
};

} // namespace vortex
