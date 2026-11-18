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

`ifdef VX_CFG_H100_FE_ENABLE

`include "VX_define.vh"

// Uniform datapath: UR sources arrive broadcast in every lane, so each lane computes the
// same value and writeback keeps the first active lane's. R2UR reads that lane's GPR.
module VX_alu_uniform import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1
) (
    input wire              clk,
    input wire              reset,

    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (execute_if.data.op_type)
    `UNUSED_VAR (execute_if.data.rs3_data)

    localparam SHAMT_BITS = `CLOG2(`VX_CFG_XLEN);
    localparam LANE_BITS  = `LOG2UP(NUM_LANES);

    wire [INST_UOP_BITS-1:0] uop = alu_uniform_code(execute_if.data.op_args);

    wire [LANE_BITS-1:0] first_lane;
    wire first_valid;
    VX_priority_encoder #(
        .N (NUM_LANES)
    ) first_active (
        .data_in   (execute_if.data.header.tmask),
        .index_out (first_lane),
        `UNUSED_PIN (onehot_out),
        .valid_out (first_valid)
    );
    `UNUSED_VAR (first_valid)

    wire [`VX_CFG_XLEN-1:0] r2ur_value = execute_if.data.rs1_data[first_lane];

    logic [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] uresult;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lanes
        wire [`VX_CFG_XLEN-1:0] a = execute_if.data.rs1_data[i];
        wire [`VX_CFG_XLEN-1:0] b = execute_if.data.rs2_data[i];
        wire [SHAMT_BITS-1:0] shamt = b[SHAMT_BITS-1:0];
        always @(*) begin
            case (uop)
            INST_UOP_ADD:  uresult[i] = a + b;
            INST_UOP_SUB:  uresult[i] = a - b;
            INST_UOP_AND:  uresult[i] = a & b;
            INST_UOP_OR:   uresult[i] = a | b;
            INST_UOP_XOR:  uresult[i] = a ^ b;
            INST_UOP_SLL:  uresult[i] = a << shamt;
            INST_UOP_SRL:  uresult[i] = a >> shamt;
            INST_UOP_SRA:  uresult[i] = `VX_CFG_XLEN'($signed(a) >>> shamt);
            INST_UOP_SLT:  uresult[i] = `VX_CFG_XLEN'($signed(a) < $signed(b));
            INST_UOP_SLTU: uresult[i] = `VX_CFG_XLEN'(a < b);
            INST_UOP_SEQ:  uresult[i] = `VX_CFG_XLEN'(a == b);
            INST_UOP_R2UR: uresult[i] = r2ur_value;
            default:       uresult[i] = a; // UR2R
            endcase
        end
    end

    // One output register, as VX_alu_int.
    VX_elastic_buffer #(
        .DATAW ($bits(alu_header_t) + NUM_LANES * `VX_CFG_XLEN)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (execute_if.valid),
        .ready_in  (execute_if.ready),
        .data_in   ({execute_if.data.header, uresult}),
        .data_out  ({result_if.data.header, result_if.data.data}),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

endmodule

`endif
