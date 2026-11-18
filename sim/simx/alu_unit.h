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
#ifdef VX_CFG_H100_FE_ENABLE
#include "warp_ops.h"
#endif
#line 19

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
#ifdef VX_CFG_H100_FE_ENABLE
  static_assert(VX_CFG_NUM_ALU_LANES == VX_CFG_NUM_THREADS, "a warp op sees the whole warp in one packet");

  void on_reset() override;

  static constexpr uint32_t kWopUnits = VX_CFG_WARP_OP_SHARED ? 1 : VX_CFG_NUM_ALU_BLOCKS;
  static constexpr uint32_t kWopPorts = VX_CFG_WARP_OP_SHARED ? VX_CFG_NUM_ALU_BLOCKS : 1;

  struct WopTag {
    instr_trace_t* trace = nullptr;
    uint32_t block = 0;
  };

  using BlockFlags = std::array<bool, VX_CFG_NUM_ALU_BLOCKS>;

  // One clock of every warp-op unit: the offered result to its block's result port when the
  // port grants it, then the round-robin pick among the unit's blocks whose head is a warp op
  // (a block whose input already issued this cycle is skipped). True while busy.
  bool warp_op_tick(const BlockFlags& issued, BlockFlags& granted);

  // A block's result port takes one result a cycle (VX_pe_switch): round robin between its
  // ALU results, each ready at the cycle it reaches the port, and its warp-op unit's.
  struct AluRsp {
    uint64_t at;
    instr_trace_t* trace;
  };
  std::vector<std::vector<AluRsp>> alu_rsp_ = std::vector<std::vector<AluRsp>>(VX_CFG_NUM_ALU_BLOCKS);
  BlockFlags port_last_wop_ = {};

  // The oldest ALU result of block `b` at the port by `now`, or -1.
  int alu_rsp_ready(uint32_t b, uint64_t now) const;

  static bool is_warp_op(const instr_trace_t* trace) {
    return std::get_if<WarpOpType>(&trace->op_type) != nullptr;
  }

  std::vector<warp_ops_ref::Pipe<WopTag>> wop_units_ = std::vector<warp_ops_ref::Pipe<WopTag>>(
    kWopUnits, warp_ops_ref::Pipe<WopTag>(VX_CFG_NUM_THREADS));
  std::vector<uint32_t> wop_rr_ = std::vector<uint32_t>(kWopUnits, kWopPorts - 1);
#endif
#line 48
};

}
