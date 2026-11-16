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

module VX_smem_gate_top import VX_gpu_pkg::*; #(
    parameter NUM_BLOCKS = `VX_CFG_NUM_LSU_BLOCKS,
    parameter NUM_LANES  = `VX_CFG_NUM_LSU_LANES,
    parameter WORD_SIZE  = LSU_WORD_SIZE,
    parameter ADDR_WIDTH = 16,

    // Lane widths main.cpp packs the vector ports at (uuid is tied to zero)
    localparam TAG_VALUE_WIDTH = 6,
    localparam AMO_OP_WIDTH    = $bits(amo_op_e),
    localparam TAG_WIDTH       = TAG_VALUE_WIDTH + UUID_WIDTH,
    localparam WORD_WIDTH      = WORD_SIZE * 8
) (
    input wire clk,
    input wire reset,

    input  wire                                           ext_valid,

    // LSU blocks' warp requests
    input  wire [NUM_BLOCKS-1:0]                          in_valid,
    input  wire [NUM_BLOCKS-1:0]                          in_rw,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]           in_mask,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][ADDR_WIDTH-1:0] in_addr,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_SIZE-1:0]  in_byteen,
    input  wire [NUM_BLOCKS-1:0]                          in_amo,
    input  wire [NUM_BLOCKS-1:0][AMO_OP_WIDTH-1:0]        in_amo_op,
    input  wire [NUM_BLOCKS-1:0]                          in_lacc,
    input  wire [NUM_BLOCKS-1:0][TAG_VALUE_WIDTH-1:0]     in_tag,
    output wire [NUM_BLOCKS-1:0]                          in_ready,

    output wire [NUM_BLOCKS-1:0]                          in_rsp_valid,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]           in_rsp_mask,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_WIDTH-1:0] in_rsp_data,
    output wire [NUM_BLOCKS-1:0][TAG_VALUE_WIDTH-1:0]     in_rsp_tag,
    input  wire [NUM_BLOCKS-1:0]                          in_rsp_ready,

    // Toward the banks
    output wire [NUM_BLOCKS-1:0]                          out_valid,
    output wire [NUM_BLOCKS-1:0]                          out_rw,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]           out_mask,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][ADDR_WIDTH-1:0] out_addr,
    output wire [NUM_BLOCKS-1:0][TAG_VALUE_WIDTH-1:0]     out_tag,
    input  wire [NUM_BLOCKS-1:0]                          out_ready,

    input  wire [NUM_BLOCKS-1:0]                          out_rsp_valid,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]           out_rsp_mask,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_WIDTH-1:0] out_rsp_data,
    input  wire [NUM_BLOCKS-1:0][TAG_VALUE_WIDTH-1:0]     out_rsp_tag,
    output wire [NUM_BLOCKS-1:0]                          out_rsp_ready
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    ,
    output wire [NUM_BLOCKS-1:0]                          fold_valid,
    output wire [NUM_BLOCKS-1:0][`CLOG2(NUM_LANES+1)-1:0] fold_cnt
`endif
);
    VX_lsu_mem_if #(
        .NUM_LANES  (NUM_LANES),
        .DATA_SIZE  (WORD_SIZE),
        .TAG_WIDTH  (TAG_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) in_if[NUM_BLOCKS]();

    VX_lsu_mem_if #(
        .NUM_LANES  (NUM_LANES),
        .DATA_SIZE  (WORD_SIZE),
        .TAG_WIDTH  (TAG_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) out_if[NUM_BLOCKS]();

    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_blk
        mem_bus_attr_t attr;
        always @(*) begin
            attr = '0;
            attr.is_addr_local = 1'b1;
            attr.amo.amo_valid = in_amo[b];
            attr.amo.amo_op    = amo_op_e'(in_amo_op[b]);
        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            attr.is_lacc       = in_lacc[b];
        `endif
        end
    `ifndef VX_CFG_EXT_DLMEM_ENABLE
        `UNUSED_VAR (in_lacc[b])
    `endif

        assign in_if[b].req_valid        = in_valid[b];
        assign in_if[b].req_data.mask    = in_mask[b];
        assign in_if[b].req_data.rw      = in_rw[b];
        assign in_if[b].req_data.addr    = in_addr[b];
        assign in_if[b].req_data.data    = '0;
        assign in_if[b].req_data.byteen  = in_byteen[b];
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_user
            assign in_if[b].req_data.user[i] = attr;
        end
        assign in_if[b].req_data.tag     = TAG_WIDTH'(in_tag[b]);
        assign in_ready[b]               = in_if[b].req_ready;

        assign in_rsp_valid[b]           = in_if[b].rsp_valid;
        assign in_rsp_mask[b]            = in_if[b].rsp_data.mask;
        assign in_rsp_data[b]            = in_if[b].rsp_data.data;
        assign in_rsp_tag[b]             = in_if[b].rsp_data.tag.value;
        assign in_if[b].rsp_ready        = in_rsp_ready[b];
        `UNUSED_VAR (in_if[b].rsp_data.tag.uuid)

        assign out_valid[b]              = out_if[b].req_valid;
        assign out_rw[b]                 = out_if[b].req_data.rw;
        assign out_mask[b]               = out_if[b].req_data.mask;
        assign out_addr[b]               = out_if[b].req_data.addr;
        assign out_tag[b]                = out_if[b].req_data.tag.value;
        assign out_if[b].req_ready       = out_ready[b];
        `UNUSED_VAR (out_if[b].req_data.data)
        `UNUSED_VAR (out_if[b].req_data.byteen)
        `UNUSED_VAR (out_if[b].req_data.user)
        `UNUSED_VAR (out_if[b].req_data.tag.uuid)

        assign out_if[b].rsp_valid       = out_rsp_valid[b];
        assign out_if[b].rsp_data.mask   = out_rsp_mask[b];
        assign out_if[b].rsp_data.data   = out_rsp_data[b];
        assign out_if[b].rsp_data.tag    = TAG_WIDTH'(out_rsp_tag[b]);
        assign out_rsp_ready[b]          = out_if[b].rsp_ready;
    end

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    wire [NUM_BLOCKS-1:0][NW_WIDTH-1:0] fold_wid;
    `UNUSED_VAR (fold_wid)
`endif

    VX_smem_gate #(
        .INSTANCE_ID  ("smem_gate"),
        .NUM_BLOCKS   (NUM_BLOCKS),
        .NUM_LANES    (NUM_LANES),
        .WORD_SIZE    (WORD_SIZE),
        .TAG_WIDTH    (TAG_WIDTH),
        .TAG_SEL_BITS (TAG_VALUE_WIDTH),
        .ADDR_WIDTH   (ADDR_WIDTH)
    ) gate (
        .clk       (clk),
        .reset     (reset),
        .ext_valid (ext_valid),
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        .fold_valid(fold_valid),
        .fold_wid  (fold_wid),
        .fold_cnt  (fold_cnt),
    `endif
        .in_if     (in_if),
        .out_if    (out_if)
    );

endmodule
