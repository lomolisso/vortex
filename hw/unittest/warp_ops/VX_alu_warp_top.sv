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

`ifndef WOP_BENCH_PORTS
`define WOP_BENCH_PORTS 1
`endif

module VX_alu_warp_top import VX_gpu_pkg::*; #(
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,
    parameter NUM_WARPS = `VX_CFG_NUM_WARPS,
    parameter NUM_PORTS = `WOP_BENCH_PORTS
) (
    input wire                              clk,
    input wire                              reset,

    input wire [NUM_PORTS-1:0]              in_valid,
    output wire [NUM_PORTS-1:0]             in_ready,
    input wire [NUM_PORTS-1:0][INST_WOP_BITS-1:0] in_code,
    input wire [NUM_PORTS-1:0][NW_WIDTH-1:0] in_wid,
    input wire [NUM_PORTS-1:0][UUID_WIDTH-1:0] in_uuid,
    input wire [NUM_PORTS-1:0][NUM_LANES-1:0] in_tmask,
    input wire [NUM_PORTS-1:0][NUM_LANES-1:0][`VX_CFG_XLEN-1:0] in_rs1,
    input wire [NUM_PORTS-1:0][NUM_LANES-1:0][`VX_CFG_XLEN-1:0] in_rs2,

    output wire [NUM_PORTS-1:0]             out_valid,
    input wire [NUM_PORTS-1:0]              out_ready,
    output wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] out_data,
    output wire [NUM_LANES-1:0]             out_tmask,
    output wire [NW_WIDTH-1:0]              out_wid,
    output wire [UUID_WIDTH-1:0]            out_uuid
);
    `UNUSED_PARAM (NUM_WARPS)

    VX_execute_if #(
        .data_t (alu_execute_t)
    ) execute_if[NUM_PORTS]();

    VX_result_if #(
        .data_t (alu_result_t)
    ) result_if[NUM_PORTS]();

    alu_result_t [NUM_PORTS-1:0] res_data;
    alu_result_t out_res;
    always @(*) begin
        out_res = '0;
        for (integer p = 0; p < NUM_PORTS; ++p) begin
            if (out_valid[p])
                out_res = res_data[p];
        end
    end

    for (genvar p = 0; p < NUM_PORTS; ++p) begin : g_ports
        alu_execute_t exe;
        always @(*) begin
            exe = '0;
            exe.header.wid   = in_wid[p];
            exe.header.uuid  = in_uuid[p];
            exe.header.tmask = in_tmask[p];
            exe.header.sop   = 1'b1;
            exe.header.eop   = 1'b1;
            exe.op_type      = INST_OP_BITS'(INST_ALU_FE_WARP);
            exe.op_args.alu.xtype = ALU_TYPE_OTHER;
            exe.op_args.alu.imm20 = 20'(in_code[p]);
            exe.rs1_data     = in_rs1[p];
            exe.rs2_data     = in_rs2[p];
        end

        assign execute_if[p].valid = in_valid[p];
        assign execute_if[p].data  = exe;
        assign in_ready[p]         = execute_if[p].ready;

        assign out_valid[p]        = result_if[p].valid;
        assign res_data[p]         = result_if[p].data;
        assign result_if[p].ready  = out_ready[p];
    end

    assign out_data  = out_res.data;
    assign out_tmask = out_res.header.tmask;
    assign out_wid   = out_res.header.wid;
    assign out_uuid  = out_res.header.uuid;
    `UNUSED_VAR (out_res.header)

    VX_alu_warp #(
        .INSTANCE_ID ("alu-warp"),
        .NUM_LANES   (NUM_LANES),
        .NUM_PORTS   (NUM_PORTS)
    ) alu_warp (
        .clk        (clk),
        .reset      (reset),
        .execute_if (execute_if),
        .result_if  (result_if)
    );

endmodule
