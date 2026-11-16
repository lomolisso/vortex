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
  // A client other than the LSU blocks presented an access, toward the shared-memory port.
  SimChannel<bool> ExtOut{this};
#endif
};

}

#ifdef VX_CFG_H100_SM_ENABLE
#include "smem_gate.h"
#endif

// The bytes a bank interleaves by inside LocalMem: the H100's 4-B banks (an 8-B
// lane on the even bank of its pair), otherwise one lane word.
#ifdef VX_CFG_H100_SM_ENABLE
#define LMEM_BANK_GRAIN VX_CFG_LMEM_BANK_BYTES
#else
#define LMEM_BANK_GRAIN config_.line_size
#endif

// The H100's bank pairs (lmem_pair.h); each use shares a line with existing
// code, so a build without the H100 SM preprocesses to the same text.
#ifdef VX_CFG_H100_SM_ENABLE
#include "lmem_pair.h"
#define LMEM_PAIR_MEMBER LmemPairHold pair_hold_;
#define LMEM_PAIR_RESET pair_hold_.reset(1u << config_.B);
#define LMEM_PAIR_HELD(bank, req) \
	if (pair_hold_.held(bank, (req).is_write() && !memop_is_atomic((req).op), (req).addr, (req).byteen, SimPlatform::instance().cycles())) \
		continue;
#define LMEM_PAIR_TAKE(bank, req) pair_hold_.take(bank, (req).addr, (req).byteen, SimPlatform::instance().cycles());
#else
#define LMEM_PAIR_MEMBER
#define LMEM_PAIR_RESET
#define LMEM_PAIR_HELD(bank, req)
#define LMEM_PAIR_TAKE(bank, req)
#endif
