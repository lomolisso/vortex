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

`include "VX_define.vh"

module VX_alu_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`VX_CFG_ISSUE_WIDTH],
    VX_branch_ctl_if.master branch_ctl_if [`VX_CFG_NUM_ALU_BLOCKS]
);

    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BLOCK_SIZE   = `VX_CFG_NUM_ALU_BLOCKS;
    localparam NUM_LANES    = `VX_CFG_NUM_ALU_LANES;
    localparam PARTIAL_BW   = (BLOCK_SIZE != `VX_CFG_ISSUE_WIDTH) || (NUM_LANES != `VX_CFG_SIMD_WIDTH);
`ifdef VX_CFG_H100_FE_ENABLE
    localparam PE_COUNT     = 3 + `VX_CFG_EXT_M_ENABLED;
`else
    localparam PE_COUNT     = 1 + `VX_CFG_EXT_M_ENABLED;
`endif
    localparam PE_SEL_BITS  = `CLOG2(PE_COUNT);
    localparam PE_IDX_INT   = 0;
    localparam PE_IDX_MDV   = PE_IDX_INT + `VX_CFG_EXT_M_ENABLED;
`ifdef VX_CFG_H100_FE_ENABLE
    localparam PE_IDX_UNI   = PE_IDX_MDV + 1;
    localparam PE_IDX_WOP   = PE_IDX_UNI + 1;
`endif

    VX_execute_if #(
        .data_t (alu_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_result_if #(
        .data_t (alu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0),
        .EX_TYPE    (EX_ALU)
    ) lane_dispatch (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

`ifdef VX_CFG_H100_FE_ENABLE
    VX_execute_if #(
        .data_t (alu_execute_t)
    ) wop_execute_if[BLOCK_SIZE]();
    VX_result_if #(
        .data_t (alu_result_t)
    ) wop_result_if[BLOCK_SIZE]();

    localparam WOP_PORTS = `VX_CFG_WARP_OP_SHARED ? BLOCK_SIZE : 1;
    for (genvar u = 0; u < BLOCK_SIZE / WOP_PORTS; ++u) begin : g_wop
        VX_alu_warp #(
            .INSTANCE_ID (`SFORMATF(("%s-warp%0d", INSTANCE_ID, u))),
            .NUM_LANES   (NUM_LANES),
            .NUM_PORTS   (WOP_PORTS)
        ) warp_unit (
            .clk        (clk),
            .reset      (reset),
            .execute_if (wop_execute_if[u * WOP_PORTS +: WOP_PORTS]),
            .result_if  (wop_result_if[u * WOP_PORTS +: WOP_PORTS])
        );
    end
`endif

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks

        VX_execute_if #(
            .data_t (alu_execute_t)
        ) pe_execute_if[PE_COUNT]();

        VX_result_if#(
            .data_t (alu_result_t)
        ) pe_result_if[PE_COUNT]();

        reg [`UP(PE_SEL_BITS)-1:0] pe_select;
        always @(*) begin
            pe_select = PE_IDX_INT;
            if (`VX_CFG_EXT_M_ENABLED && (per_block_execute_if[block_idx].data.op_args.alu.xtype == ALU_TYPE_MULDIV))
                pe_select = PE_IDX_MDV;
        `ifdef VX_CFG_H100_FE_ENABLE
            if (alu_is_uniform(per_block_execute_if[block_idx].data.op_args.alu.xtype, per_block_execute_if[block_idx].data.op_type))
                pe_select = PE_IDX_UNI;
            if (alu_is_warp(per_block_execute_if[block_idx].data.op_args.alu.xtype, per_block_execute_if[block_idx].data.op_type))
                pe_select = PE_IDX_WOP;
        `endif
        end

        VX_pe_switch #(
            .PE_COUNT    (PE_COUNT),
            .NUM_LANES   (NUM_LANES),
            .ARBITER     ("R"),
            .REQ_OUT_BUF (0),
            .RSP_OUT_BUF (PARTIAL_BW ? 1 : 3)
        ) pe_switch (
            .clk            (clk),
            .reset          (reset),
            .pe_sel         (pe_select),
            .execute_in_if  (per_block_execute_if[block_idx]),
            .result_out_if  (per_block_result_if[block_idx]),
            .execute_out_if (pe_execute_if),
            .result_in_if   (pe_result_if)
        );

        VX_alu_int #(
            .INSTANCE_ID (`SFORMATF(("%s-int%0d", INSTANCE_ID, block_idx))),
            .BLOCK_IDX (block_idx),
            .NUM_LANES (NUM_LANES)
        ) alu_int (
            .clk        (clk),
            .reset      (reset),
            .execute_if (pe_execute_if[PE_IDX_INT]),
            .branch_ctl_if (branch_ctl_if[block_idx]),
            .result_if  (pe_result_if[PE_IDX_INT])
        );

    `ifdef VX_CFG_EXT_M_ENABLE
        VX_alu_muldiv #(
            .INSTANCE_ID (`SFORMATF(("%s-muldiv%0d", INSTANCE_ID, block_idx))),
            .NUM_LANES (NUM_LANES)
        ) muldiv_unit (
            .clk        (clk),
            .reset      (reset),
            .execute_if (pe_execute_if[PE_IDX_MDV]),
            .result_if  (pe_result_if[PE_IDX_MDV])
        );
    `endif

    `ifdef VX_CFG_H100_FE_ENABLE
        VX_alu_uniform #(
            .INSTANCE_ID (`SFORMATF(("%s-uniform%0d", INSTANCE_ID, block_idx))),
            .NUM_LANES (NUM_LANES)
        ) uniform_unit (
            .clk        (clk),
            .reset      (reset),
            .execute_if (pe_execute_if[PE_IDX_UNI]),
            .result_if  (pe_result_if[PE_IDX_UNI])
        );

        `ASSIGN_VX_IF (wop_execute_if[block_idx], pe_execute_if[PE_IDX_WOP]);
        `ASSIGN_VX_IF (pe_result_if[PE_IDX_WOP], wop_result_if[block_idx]);
    `endif
    end

    VX_lane_gather #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) lane_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule
