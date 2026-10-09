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

// Output side 0 is the global (dcache) port, side 1 the local-memory port.
module VX_lmem_switch_top import VX_gpu_pkg::*; #(
    // Defaults match the VX_mem_unit instance.
    parameter GLOBAL_OUT_BUF    = 1,
    parameter LOCAL_OUT_BUF     = 1,
    parameter RSP_OUT_BUF       = 1,
    parameter `STRING ARBITER   = "P",

    localparam NUM_LANES        = `VX_CFG_NUM_LSU_LANES,
    localparam WORD_SIZE        = LSU_WORD_SIZE,
    localparam WORD_WIDTH       = LSU_WORD_SIZE * 8,
    localparam ADDR_WIDTH       = LSU_ADDR_WIDTH,
    localparam TAG_VALUE_WIDTH  = LSU_TAG_WIDTH - UUID_WIDTH,
    localparam AMO_OP_WIDTH     = 4
) (
    input wire clk,
    input wire reset,

    // LSU request
    input  wire                                     in_req_valid,
    input  wire                                     in_req_rw,
    input  wire [NUM_LANES-1:0]                     in_req_mask,
    input  wire [NUM_LANES-1:0][ADDR_WIDTH-1:0]     in_req_addr,
    input  wire [NUM_LANES-1:0][WORD_WIDTH-1:0]     in_req_data,
    input  wire [NUM_LANES-1:0][WORD_SIZE-1:0]      in_req_byteen,
    input  wire [NUM_LANES-1:0]                     in_req_local,
    input  wire [NUM_LANES-1:0]                     in_req_amo,
    input  wire [NUM_LANES-1:0][AMO_OP_WIDTH-1:0]   in_req_amo_op,
    input  wire [TAG_VALUE_WIDTH-1:0]               in_req_tag,
    output wire                                     in_req_ready,

    // LSU response
    output wire                                     in_rsp_valid,
    output wire [NUM_LANES-1:0]                     in_rsp_mask,
    output wire [NUM_LANES-1:0][WORD_WIDTH-1:0]     in_rsp_data,
    output wire [TAG_VALUE_WIDTH-1:0]               in_rsp_tag,
    input  wire                                     in_rsp_ready,

    // Memory-side requests
    output wire [1:0]                                   out_req_valid,
    output wire [1:0]                                   out_req_rw,
    output wire [1:0][NUM_LANES-1:0]                    out_req_mask,
    output wire [1:0][NUM_LANES-1:0][ADDR_WIDTH-1:0]    out_req_addr,
    output wire [1:0][NUM_LANES-1:0][WORD_WIDTH-1:0]    out_req_data,
    output wire [1:0][NUM_LANES-1:0][WORD_SIZE-1:0]     out_req_byteen,
    output wire [1:0][NUM_LANES-1:0]                    out_req_local,
    output wire [1:0][NUM_LANES-1:0]                    out_req_amo,
    output wire [1:0][NUM_LANES-1:0][AMO_OP_WIDTH-1:0]  out_req_amo_op,
    output wire [1:0][TAG_VALUE_WIDTH-1:0]              out_req_tag,
    input  wire [1:0]                                   out_req_ready,

    // Memory-side responses
    input  wire [1:0]                                   out_rsp_valid,
    input  wire [1:0][NUM_LANES-1:0]                    out_rsp_mask,
    input  wire [1:0][NUM_LANES-1:0][WORD_WIDTH-1:0]    out_rsp_data,
    input  wire [1:0][TAG_VALUE_WIDTH-1:0]              out_rsp_tag,
    output wire [1:0]                                   out_rsp_ready
);
    // main.cpp derives these widths from VX_config.h.
    `STATIC_ASSERT (ADDR_WIDTH == `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(`VX_CFG_XLEN / 8),
        ("lmem_switch_top: address width differs from the testbench's"))
    `STATIC_ASSERT (TAG_VALUE_WIDTH == `CLOG2(`VX_CFG_LSU_PENDING_SIZE),
        ("lmem_switch_top: tag width differs from the testbench's"))
    `STATIC_ASSERT ($bits(amo_op_e) == AMO_OP_WIDTH,
        ("lmem_switch_top: amo_op_e is %0d bits, the testbench packs %0d", $bits(amo_op_e), AMO_OP_WIDTH))

    VX_lsu_mem_if #(
        .NUM_LANES (NUM_LANES),
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_in_if();

    VX_lsu_mem_if #(
        .NUM_LANES (NUM_LANES),
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) out_if[2]();

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_in_user
        mem_bus_attr_t attr;
        always @(*) begin
            attr = '0;
            attr.is_addr_local = in_req_local[i];
            attr.amo.amo_valid = in_req_amo[i];
            attr.amo.amo_op    = amo_op_e'(in_req_amo_op[i]);
            attr.amo.hart_id   = HART_ID_WIDTH'(i);
        end
        assign lsu_in_if.req_data.user[i] = attr;
    end

    assign lsu_in_if.req_valid       = in_req_valid;
    assign lsu_in_if.req_data.rw     = in_req_rw;
    assign lsu_in_if.req_data.mask   = in_req_mask;
    assign lsu_in_if.req_data.addr   = in_req_addr;
    assign lsu_in_if.req_data.data   = in_req_data;
    assign lsu_in_if.req_data.byteen = in_req_byteen;
    assign lsu_in_if.req_data.tag    = {UUID_WIDTH'(0), in_req_tag};
    assign in_req_ready = lsu_in_if.req_ready;

    assign in_rsp_valid = lsu_in_if.rsp_valid;
    assign in_rsp_mask  = lsu_in_if.rsp_data.mask;
    assign in_rsp_data  = lsu_in_if.rsp_data.data;
    assign in_rsp_tag   = lsu_in_if.rsp_data.tag.value;
    assign lsu_in_if.rsp_ready = in_rsp_ready;
    `UNUSED_VAR (lsu_in_if.rsp_data.tag.uuid)

    for (genvar s = 0; s < 2; ++s) begin : g_out
        assign out_req_valid[s]  = out_if[s].req_valid;
        assign out_req_rw[s]     = out_if[s].req_data.rw;
        assign out_req_mask[s]   = out_if[s].req_data.mask;
        assign out_req_addr[s]   = out_if[s].req_data.addr;
        assign out_req_data[s]   = out_if[s].req_data.data;
        assign out_req_byteen[s] = out_if[s].req_data.byteen;
        assign out_req_tag[s]    = out_if[s].req_data.tag.value;
        assign out_if[s].req_ready = out_req_ready[s];
        `UNUSED_VAR (out_if[s].req_data.tag.uuid)

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_user
            mem_bus_attr_t attr;
            assign attr = mem_bus_attr_t'(out_if[s].req_data.user[i]);
            assign out_req_local[s][i]  = attr.is_addr_local;
            assign out_req_amo[s][i]    = attr.amo.amo_valid;
            assign out_req_amo_op[s][i] = attr.amo.amo_op;
            `UNUSED_VAR (attr)
        end

        assign out_if[s].rsp_valid          = out_rsp_valid[s];
        assign out_if[s].rsp_data.mask      = out_rsp_mask[s];
        assign out_if[s].rsp_data.data      = out_rsp_data[s];
        assign out_if[s].rsp_data.tag       = {UUID_WIDTH'(0), out_rsp_tag[s]};
        assign out_rsp_ready[s] = out_if[s].rsp_ready;
    end

    VX_lmem_switch #(
        .GLOBAL_OUT_BUF (GLOBAL_OUT_BUF),
        .LOCAL_OUT_BUF  (LOCAL_OUT_BUF),
        .RSP_OUT_BUF    (RSP_OUT_BUF),
        .ARBITER        (ARBITER)
    ) lmem_switch (
        .clk           (clk),
        .reset         (reset),
        .lsu_in_if     (lsu_in_if),
        .global_out_if (out_if[0]),
        .local_out_if  (out_if[1])
    );

endmodule
