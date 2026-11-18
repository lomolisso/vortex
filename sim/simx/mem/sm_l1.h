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

#include <VX_config.h>

#ifdef VX_CFG_H100_SM_ENABLE

#include "types.h"
#include "sm_sram.h"

namespace vortex {

// The SM's L1 on the unified array (VX_sm_l1.sv with VX_sm_l1_tags,
// VX_sm_l1_slots and VX_sm_l1_mshr): decoupled tags pointing at data slots
// in the array's top rows, the miss path and the in-order global stream.
// Ticks after SmSram: the array's same-cycle outputs (takes, retirements,
// fill writes, responses) are inputs of this cycle, and the requests to the
// array are offered for the next cycle from registers only.
class SmL1 : public SimObject<SmL1> {
public:
  using Ptr = std::shared_ptr<SmL1>;

  struct Config {
    uint32_t num_blocks;
  };

  struct PerfStats {
    uint64_t reads = 0;
    uint64_t writes = 0;
    uint64_t misses = 0;
    uint64_t mshr_stalls = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      reads += rhs.reads;
      writes += rhs.writes;
      misses += rhs.misses;
      mshr_stalls += rhs.mshr_stalls;
      return *this;
    }
  };

  SimChannel<SmSramCfg> CfgIn;

  std::vector<SimChannel<LsuReq>> LsuReqIn;
  std::vector<SimChannel<LsuRsp>> LsuRspOut;

  // The global stream to the array: one offer at a time, held until taken.
  SimChannel<SmSramReq> GlReqOut;
  SimChannel<SmSramRsp> GlRspIn;
  SimChannel<uint32_t>  WfRetiredIn;

  // Refill rows: one offer at a time, written when its row comes back.
  SimChannel<SmSramFill> FillOut;
  SimChannel<uint32_t>   FillDoneIn;

  // Line reads, write-through, atomics and fences, as memory blocks.
  SimChannel<MemReq> MemReqOut;
  SimChannel<MemRsp> MemRspIn;

  SmL1(const SimContext& ctx, const char* name, const Config& config);
  virtual ~SmL1();

  const PerfStats& perf_stats() const;

  bool busy() const;

  // Host flush: a fence entering after the LSU legs (drain, invalidate, one
  // fence on the miss bus); done once its reply is back.
  void flush_begin();
  bool flush_done() const;

protected:
  void on_reset();
  void on_tick();

  class Impl;
  Impl* impl_;

  friend class SimObject<SmL1>;
};

}

#endif // VX_CFG_H100_SM_ENABLE
