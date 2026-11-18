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

#include "types.h"

namespace vortex {

#ifdef VX_CFG_EXT_MBAR_ENABLE
// A bank's mbarrier event, keyed by the word's (bank, row): a try_wait that
// read an incomplete phase (insert, from warp `wid`) or an op whose write-back
// completed one (flip).
struct MbarBankEvt {
  bool     flip;
  uint32_t wid;
  uint32_t row;
};
#endif
#ifdef VX_CFG_H100_SM_ENABLE
}
#include "sm_sram.h"
namespace vortex {
// The core latches a launch's carve-out into its array; the hook shares a
// line with Core::dcr_write, so a build without the H100 SM preprocesses to
// the same text.
#define LMEM_SMEM_CONFIG_HOOK(addr, value) if ((addr) == VX_DCR_KMU_SMEM_CONFIG) impl_->smem_config(value);
#define LMEM_SMEM_CFG_RESET smem_cfg_dirty_ = true;
#define LMEM_SMEM_CFG_TICK this->smem_cfg_send();
// The L1's queued misses and write-throughs hold the core, as its busy does
// in the RTL; the core has one dcache port, the L1's.
#define LMEM_SM_L1_RUNNING if (sm_l1_->busy()) return true;
#define CORE_DCACHE_PORTS 1
#else
#define LMEM_SMEM_CONFIG_HOOK(addr, value)
#define LMEM_SMEM_CFG_RESET
#define LMEM_SMEM_CFG_TICK
#define LMEM_SM_L1_RUNNING
#define CORE_DCACHE_PORTS VX_CFG_DCACHE_NUM_REQS
#line 29
#endif

class LocalMem : public SimObject<LocalMem> {
public:
  struct Config {
    uint32_t capacity;
    uint32_t line_size;
    uint32_t num_reqs;
    uint32_t dma_begin; // ports [dma_begin, num_reqs) are DMA ports
    uint32_t B; // log2 number of banks
    bool write_reponse;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
    uint32_t commit_ports = 0; // counted accesses (flags.lacc) from ports [0, commit_ports) pulse CommitOut
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
    uint32_t cpa_ports = 0;    // ports [num_reqs, num_reqs + cpa_ports): the cp.async writers, arbitrated apart
#endif
#ifdef DXA_MBAR_ENABLE
    uint32_t dxa_compl_port = 0; // the DXA completion unit's complete_tx requests
    uint32_t dxa_row_port = 0;   // a peer's DXA words from the DLMEM ingress: the last DMA port
#else
#line 42
#endif
  };

  struct PerfStats {
    uint64_t reads = 0;
    uint64_t writes = 0;
    uint64_t bank_stalls = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      this->reads += rhs.reads;
      this->writes += rhs.writes;
      this->bank_stalls += rhs.bank_stalls;
      return *this;
    }
  };

  std::vector<SimChannel<MemReq>> Inputs;
  std::vector<SimChannel<MemRsp>> Outputs;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
#ifdef VX_CFG_H100_SM_ENABLE
  // Per bank: the warp and lanes of each such access when it commits, 1 cycle after the
  // bank takes a load or store and 3 cycles after it takes an atomic.
  std::vector<SimChannel<WtInc>> CommitOut;
#else
#line 70
  // Per bank: the warp of each such access when it commits, 1 cycle after the
  // bank takes a load or store and 3 cycles after it takes an atomic.
  std::vector<SimChannel<uint32_t>> CommitOut;
#endif
#line 73
#endif
#ifdef VX_CFG_EXT_MBAR_ENABLE
  // Per bank, at most one per cycle: an insert 1 cycle after the bank takes the
  // try_wait (its read), a flip 2 cycles after it takes the op (its write-back).
  std::vector<SimChannel<MbarBankEvt>> MbarEvtOut;
#endif
#ifdef VX_CFG_EXT_DXA_ENABLE
  // The DXA writes the banks commit, in the cycle they commit, toward the
  // core's DXA completion unit.
  SimChannel<DxaWrCommit> DxaWrOut;

  // The free entries the completion queue will have next cycle. A copy's
  // final beat waits at the DXA input while it has none, so none is dropped.
  SimEventLink<uint32_t> DxaFreeIn;
private:
  void on_dxa_free(const uint32_t& free);
public:
#else
#line 69
#endif

  LocalMem(const SimContext& ctx, const char* name, const Config& config);
  virtual ~LocalMem();

  const PerfStats& perf_stats() const;

protected:

  void on_reset();
  void on_tick();

  class Impl;
  Impl* impl_;

  friend class SimObject<LocalMem>;
#ifdef VX_CFG_H100_SM_ENABLE
public:
  // Toward the SM's one array: row reads, row writes, one-lane atomics, the DSMEM merge state.
  SmSramClientPorts ArrayPorts{this};
#endif
};

}
