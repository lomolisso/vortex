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
#include "types.h"
#include "dxa_unit.h"
#ifdef DXA_MBAR_ENABLE
#include "dlmem/dlmem_types.h"
#else
#line 18
#endif

namespace vortex {

class Core;
class Socket;

// Socket-resident DXA engine. Aggregates DxaReq packets from per-core
// DxaUnits, dispatches to VX_CFG_NUM_DXA_CORES workers, and drives GMEM reads
// (through L2) and LMEM writes (per-core direct channels carrying real
// MemReq packets with TLM payload + completion flag).
class DxaCore : public SimObject<DxaCore> {
public:
  using Ptr = std::shared_ptr<DxaCore>;

  struct PerfStats {
    uint64_t transfers      = 0;
    uint64_t gmem_reads     = 0;
    uint64_t gmem_dedup     = 0;
    uint64_t lmem_writes    = 0;
    uint64_t total_latency  = 0;
    uint64_t noslot_stalls  = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      transfers     += rhs.transfers;
      gmem_reads    += rhs.gmem_reads;
      gmem_dedup    += rhs.gmem_dedup;
      lmem_writes   += rhs.lmem_writes;
      total_latency += rhs.total_latency;
      noslot_stalls += rhs.noslot_stalls;
      return *this;
    }
  };

  // Per-core DxaReq inputs (size = VX_CFG_SOCKET_SIZE). The socket binds
  // each core's DxaUnit::req_out here.
  std::vector<SimChannel<DxaReq>>  dxa_req_in;

  // GMEM ports to L2 (size = kDxaMemPorts). Internally fed by gmem_arb_
  // from VX_CFG_NUM_DXA_CORES worker outputs.
  std::vector<SimChannel<MemReq>>  gmem_req_out;
  std::vector<SimChannel<MemRsp>>  gmem_rsp_in;
  MemArbiter::Ptr                  gmem_arb_;

  // LMEM row width the RTL drains per cycle: one DXA write beat covers the
  // full banked row (DXA_LMEM_WORD_SIZE = LMEM_NUM_BANKS * XLEN/8). The
  // LocalMem model's byteen scope is one VX_CFG_MEM_BLOCK_SIZE block, so a
  // row beat is emitted as LMEM_PORTS_PER_CORE same-cycle block writes on
  // adjacent input ports (one per row half).
  static constexpr uint32_t LMEM_ROW_SIZE =
      VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);
  static constexpr uint32_t LMEM_PORTS_PER_CORE =
      (LMEM_ROW_SIZE > VX_CFG_MEM_BLOCK_SIZE)
          ? (LMEM_ROW_SIZE / VX_CFG_MEM_BLOCK_SIZE) : 1;

  // Per-core LMEM write ports (size = VX_CFG_SOCKET_SIZE *
  // LMEM_PORTS_PER_CORE, core-major). The socket binds each core's
  // LocalMem::Inputs[port_dxa + p] here; a bulk copy's reads answer below.
  std::vector<SimChannel<MemReq>>  lmem_req_out;
#ifdef VX_CFG_EXT_DXA_ENABLE

  // Per core, the raw barriers of copies finished without a write, to its
  // DXA completion unit.
  std::vector<SimChannel<uint32_t>> compl_release_out;
#ifdef DXA_MBAR_ENABLE

  // The row egress: the flits of copies into a block on another core of the
  // cluster, posted into the socket's DLMEM request switch, one per cycle.
  SimChannel<DlmemReqPkt> row_out;

  // A bulk copy's row reads of its source, answered by the issuing core's
  // LocalMem on the DXA ports of lmem_req_out (same indexing).
  std::vector<SimChannel<MemRsp>>  lmem_rsp_in;

  // The line reader's entries per worker (VX_dxa_lmem_rd DEPTH), apart from
  // the DXA_MAX_INFLIGHT tags of its lines.
  static constexpr uint32_t LMEM_RD_DEPTH = 7;
#else
#line 81
#endif
#else
#line 76
#endif

  DxaCore(const SimContext& ctx, const char* name, Socket* socket);
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
