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

#include "func_unit.h"
#include <array>
#include <vector>

namespace vortex {

class AluUnit : public FuncUnit<VX_CFG_NUM_ALU_BLOCKS> {
public:
  AluUnit(const SimContext& ctx, const char* name, Core*);

protected:
  void on_tick() override;

private:
  // Per-unit functional execution. Called only from this unit's tick().
  // Reads operands from trace->src_data, populates trace->dst_data,
  // mutates warp.PC for branches.
  void execute(instr_trace_t* trace);

  uint32_t latency_of(const instr_trace_t* trace) const;

  // A packet narrower than 4 lanes holds part of a WGATHER group of 4.
  static constexpr bool kWgatherSplit = (VX_CFG_NUM_ALU_LANES < 4) && (VX_CFG_NUM_ALU_LANES < VX_CFG_NUM_THREADS);

  void wgather_split(instr_trace_t* trace, uint32_t src_offset);

  // Per issue slice: the source thread's operands from an earlier packet.
  struct WgatherStage {
    bool active = false;
    std::array<reg_data_t, 3> ops = {};
  };
  std::vector<WgatherStage> wg_stages_ = std::vector<WgatherStage>(VX_CFG_ISSUE_WIDTH);
};

}
