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

// One core's DXA completion unit, as VX_mem_unit wires it: the DMA port's
// bank writes and their attribute in, the barrier release and the local-memory
// complete_tx request out. main.cpp packs the per-slot fields in fixed lanes.
module VX_dxa_completion_top import VX_gpu_pkg::*; #(
    parameter NUM_BANKS = `VX_CFG_LMEM_NUM_BANKS
) (
    input  wire                         clk,
    input  wire                         reset,

    input  wire [NUM_BANKS-1:0]         bank_wr_fire,
    input  wire                         in_last,
    input  wire                         in_mode,
    input  wire [31:0]                  in_payload,
    input  wire [31:0]                  in_tx_bytes,
    input  wire [7:0]                   in_slot,
    input  wire [7:0]                   in_gen,

    input  wire [`VX_CFG_NUM_WARPS-1:0] slot_valid,
    input  wire [`VX_CFG_NUM_WARPS-1:0][7:0] slot_gen,
    input  wire [31:0]                  slot_stride,

    output wire                         full,

    output wire                         txbar_valid,
    output wire [31:0]                  txbar_addr,
    output wire                         txbar_is_done,
    input  wire                         txbar_ready,

    output wire                         compl_valid,
    output wire                         compl_rw,
    output wire [63:0]                  compl_addr,
    output wire [63:0]                  compl_data,
    output wire [7:0]                   compl_byteen,
    output wire                         compl_amo_valid,
    output wire [7:0]                   compl_amo_op,
    output wire                         compl_amo_unsigned,
    output wire [31:0]                  compl_hart_id,
    output wire                         compl_is_lacc,
    input  wire                         compl_ready,
    input  wire                         compl_rsp_valid,
    output wire                         compl_rsp_ready
);
    `STATIC_ASSERT (LSU_WORD_SIZE == 8, ("dxa_completion_top: mbarriers need 64-bit words"))
    `STATIC_ASSERT (DXA_ATTR_PAYLOAD_W <= 32, ("dxa_completion_top: payload is packed in 32 bits"))
    `STATIC_ASSERT (DXA_CTA_GEN_W <= 8, ("dxa_completion_top: generations are packed in 8 bits"))

    dxa_lmem_attr_t attr;
    assign attr.last     = in_last;
    assign attr.mode     = in_mode;
    assign attr.payload  = DXA_ATTR_PAYLOAD_W'(in_payload);
    assign attr.tx_bytes = DXA_TX_BYTES_W'(in_tx_bytes);
    assign attr.slot     = NW_WIDTH'(in_slot);
    assign attr.gen      = DXA_CTA_GEN_W'(in_gen);
    assign attr.remote   = 1'b0;
    assign attr.dst_core = '0;

    wire [`VX_CFG_NUM_WARPS-1:0][DXA_CTA_GEN_W-1:0] slot_gen_w;
    for (genvar i = 0; i < `VX_CFG_NUM_WARPS; ++i) begin : g_gen
        assign slot_gen_w[i] = DXA_CTA_GEN_W'(slot_gen[i]);
    end
    `UNUSED_VAR ({in_payload, in_tx_bytes, in_slot, in_gen, slot_gen, slot_stride})

    VX_txbar_bus_if txbar_bus_if();

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) compl_bus_if();

    VX_dxa_completion #(
        .INSTANCE_ID ("dxa_completion"),
        .NUM_BANKS   (NUM_BANKS),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W)
    ) completion (
        .clk          (clk),
        .reset        (reset),
        .bank_wr_fire (bank_wr_fire),
        .bank_wr_attr (attr),
        .full         (full),
        .slot_valid   (slot_valid),
        .slot_gen     (slot_gen_w),
        .slot_stride  ((`VX_CFG_LMEM_LOG_SIZE+1)'(slot_stride)),
        .compl_bus_if (compl_bus_if),
        .txbar_bus_if (txbar_bus_if)
    );

    assign txbar_valid   = txbar_bus_if.valid;
    assign txbar_addr    = 32'(txbar_bus_if.data.addr);
    assign txbar_is_done = txbar_bus_if.data.is_done;
    assign txbar_bus_if.ready = txbar_ready;

    mem_bus_attr_t compl_attr;
    assign compl_attr = mem_bus_attr_t'(compl_bus_if.req_data.attr);

    assign compl_valid        = compl_bus_if.req_valid;
    assign compl_rw           = compl_bus_if.req_data.rw;
    assign compl_addr         = 64'(compl_bus_if.req_data.addr);
    assign compl_data         = 64'(compl_bus_if.req_data.data);
    assign compl_byteen       = 8'(compl_bus_if.req_data.byteen);
    assign compl_amo_valid    = compl_attr.amo.amo_valid;
    assign compl_amo_op       = 8'(compl_attr.amo.amo_op);
    assign compl_amo_unsigned = compl_attr.amo.amo_unsigned;
    assign compl_hart_id      = 32'(compl_attr.amo.hart_id);
    assign compl_is_lacc      = compl_attr.is_lacc;
    assign compl_bus_if.req_ready = compl_ready;
    `UNUSED_VAR (compl_attr)
    `UNUSED_VAR (compl_bus_if.req_data.tag)

    assign compl_bus_if.rsp_valid = compl_rsp_valid;
    assign compl_bus_if.rsp_data  = '0;
    assign compl_rsp_ready = compl_bus_if.rsp_ready;

endmodule
