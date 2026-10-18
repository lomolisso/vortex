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

module VX_local_mem_top import VX_gpu_pkg::*; #(
    parameter `STRING  INSTANCE_ID = "",

    // Size of local memory in bytes
    parameter SIZE              = (1 << `VX_CFG_LMEM_LOG_SIZE),

    // Number of Word requests per cycle
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // The last input stands in for the DLMEM ingress, as in VX_mem_unit.
    parameter NUM_REQS          = LSU_NUM_REQS + 1,
`else
    parameter NUM_REQS          = LSU_NUM_REQS,
`endif
    // Number of banks
    parameter NUM_BANKS         = `VX_CFG_LMEM_NUM_BANKS,

    // Size of a word in bytes
    parameter WORD_SIZE         = `VX_CFG_XLEN/8,

    // Lane widths main.cpp packs the vector ports at (uuid is tied to zero)
    localparam TAG_VALUE_WIDTH  = 16,
    localparam DMA_TAG_VALUE_WIDTH = 8,
    localparam AMO_OP_WIDTH     = 4,
    localparam HART_IN_WIDTH    = 16,

    parameter TAG_WIDTH         = TAG_VALUE_WIDTH + UUID_WIDTH,
    parameter DMA_TAG_WIDTH     = DMA_TAG_VALUE_WIDTH + UUID_WIDTH,

    // Address width
    parameter NUM_WORDS         = SIZE / WORD_SIZE,
    parameter WORDS_PER_BANK    = NUM_WORDS / NUM_BANKS,
    parameter BANK_ADDR_WIDTH   = `CLOG2(WORDS_PER_BANK),
    parameter ADDR_WIDTH        = BANK_ADDR_WIDTH + `CLOG2(NUM_BANKS),
    parameter WORD_WIDTH        = WORD_SIZE * 8,
    parameter ROW_SIZE          = NUM_BANKS * WORD_SIZE
 ) (
    input wire clk,
    input wire reset,

    // Core request
    input  wire [NUM_REQS-1:0]                 mem_req_valid,
    input  wire [NUM_REQS-1:0]                 mem_req_rw,
    input  wire [NUM_REQS-1:0][WORD_SIZE-1:0]  mem_req_byteen,
    input  wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] mem_req_addr,
    input  wire [NUM_REQS-1:0][WORD_WIDTH-1:0] mem_req_data,
    input  wire [NUM_REQS-1:0][TAG_VALUE_WIDTH-1:0] mem_req_tag,
    input  wire [NUM_REQS-1:0]                 mem_req_amo_valid,
    input  wire [NUM_REQS-1:0][AMO_OP_WIDTH-1:0] mem_req_amo_op,
    input  wire [NUM_REQS-1:0]                 mem_req_amo_unsigned,
    input  wire [NUM_REQS-1:0][`VX_CFG_XLEN-1:0] mem_req_amo_cmp,
    input  wire [NUM_REQS-1:0][HART_IN_WIDTH-1:0] mem_req_hart_id,
    output wire [NUM_REQS-1:0]                 mem_req_ready,

    // Core response
    output wire [NUM_REQS-1:0]                 mem_rsp_valid,
    output wire [NUM_REQS-1:0][WORD_WIDTH-1:0] mem_rsp_data,
    output wire [NUM_REQS-1:0][TAG_VALUE_WIDTH-1:0] mem_rsp_tag,
    input  wire [NUM_REQS-1:0]                 mem_rsp_ready,

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // Plain-store commits of the LSU inputs, one per bank
    output wire [NUM_BANKS-1:0]                st_commit_valid,
    output wire [NUM_BANKS-1:0][7:0]           st_commit_wid,
`endif

    // DMA request: one row of NUM_BANKS words at a bank address
    input  wire                                dma_req_valid,
    input  wire                                dma_req_rw,
    input  wire [BANK_ADDR_WIDTH-1:0]          dma_req_addr,
    input  wire [ROW_SIZE-1:0]                 dma_req_byteen,
    input  wire [ROW_SIZE*8-1:0]               dma_req_data,
    input  wire [DMA_TAG_VALUE_WIDTH-1:0]      dma_req_tag,
    output wire                                dma_req_ready,

    // DMA response
    output wire                                dma_rsp_valid,
    output wire [ROW_SIZE*8-1:0]               dma_rsp_data,
    output wire [DMA_TAG_VALUE_WIDTH-1:0]      dma_rsp_tag,
    input  wire                                dma_rsp_ready
);
    `STATIC_ASSERT ($bits(amo_op_e) == AMO_OP_WIDTH,
        ("local_mem_top: amo_op_e is %0d bits, the testbench packs %0d", $bits(amo_op_e), AMO_OP_WIDTH))
    `STATIC_ASSERT (TAG_WIDTH == TAG_VALUE_WIDTH + UUID_WIDTH,
        ("local_mem_top: TAG_WIDTH must be TAG_VALUE_WIDTH + UUID_WIDTH"))
    `STATIC_ASSERT (DMA_TAG_WIDTH == DMA_TAG_VALUE_WIDTH + UUID_WIDTH,
        ("local_mem_top: DMA_TAG_WIDTH must be DMA_TAG_VALUE_WIDTH + UUID_WIDTH"))
    `STATIC_ASSERT (HART_ID_WIDTH <= HART_IN_WIDTH,
        ("local_mem_top: hart id is %0d bits, the testbench packs %0d", HART_ID_WIDTH, HART_IN_WIDTH))

    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) lsu_bus_if[NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (ROW_SIZE),
        .TAG_WIDTH (DMA_TAG_WIDTH),
        .ADDR_WIDTH(BANK_ADDR_WIDTH)
    ) dma_bus_if();

    // LSU request
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_lsu_req
        mem_bus_attr_t req_attr;
        always @(*) begin
            req_attr = '0;
            req_attr.is_addr_local    = 1'b1;
            req_attr.amo.amo_valid    = mem_req_amo_valid[i];
            req_attr.amo.amo_op       = amo_op_e'(mem_req_amo_op[i]);
            req_attr.amo.amo_unsigned = mem_req_amo_unsigned[i];
            req_attr.amo.hart_id      = HART_ID_WIDTH'(mem_req_hart_id[i]);
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            req_attr.amo.amo_cmp      = mem_req_amo_cmp[i];
        `endif
        end
    `ifndef VX_CFG_EXT_ZACAS_ENABLE
        `UNUSED_VAR (mem_req_amo_cmp[i])
    `endif
        if (HART_IN_WIDTH > HART_ID_WIDTH) begin : g_hart_hi
            `UNUSED_VAR (mem_req_hart_id[i][HART_IN_WIDTH-1:HART_ID_WIDTH])
        end
        assign lsu_bus_if[i].req_valid = mem_req_valid[i];
        assign lsu_bus_if[i].req_data.rw = mem_req_rw[i];
        assign lsu_bus_if[i].req_data.byteen = mem_req_byteen[i];
        assign lsu_bus_if[i].req_data.addr = mem_req_addr[i];
        assign lsu_bus_if[i].req_data.attr = req_attr;
        assign lsu_bus_if[i].req_data.data = mem_req_data[i];
        assign lsu_bus_if[i].req_data.tag = TAG_WIDTH'(mem_req_tag[i]);
        assign mem_req_ready[i] = lsu_bus_if[i].req_ready;
    end

    // LSU response
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_lsu_rsp
        assign mem_rsp_valid[i] = lsu_bus_if[i].rsp_valid;
        assign mem_rsp_data[i] = lsu_bus_if[i].rsp_data.data;
        assign mem_rsp_tag[i] = lsu_bus_if[i].rsp_data.tag.value;
        assign lsu_bus_if[i].rsp_ready = mem_rsp_ready[i];
        `UNUSED_VAR (lsu_bus_if[i].rsp_data.tag.uuid)
    end

    // DMA port
    assign dma_bus_if.req_valid       = dma_req_valid;
    assign dma_bus_if.req_data.rw     = dma_req_rw;
    assign dma_bus_if.req_data.addr   = dma_req_addr;
    assign dma_bus_if.req_data.byteen = dma_req_byteen;
    assign dma_bus_if.req_data.data   = dma_req_data;
    assign dma_bus_if.req_data.attr   = '0;
    assign dma_bus_if.req_data.tag    = DMA_TAG_WIDTH'(dma_req_tag);
    assign dma_req_ready = dma_bus_if.req_ready;

    assign dma_rsp_valid = dma_bus_if.rsp_valid;
    assign dma_rsp_data  = dma_bus_if.rsp_data.data;
    assign dma_rsp_tag   = dma_bus_if.rsp_data.tag.value;
    assign dma_bus_if.rsp_ready = dma_rsp_ready;
    `UNUSED_VAR (dma_bus_if.rsp_data.tag.uuid)

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    wire [NUM_BANKS-1:0][NW_WIDTH-1:0] st_commit_wid_w;
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_st_commit_wid
        assign st_commit_wid[i] = 8'(st_commit_wid_w[i]);
    end
`endif

    VX_local_mem #(
        .INSTANCE_ID  (INSTANCE_ID),
        .SIZE         (SIZE),
        .NUM_REQS     (NUM_REQS),
        .NUM_BANKS    (NUM_BANKS),
        .WORD_SIZE    (WORD_SIZE),
        .ADDR_WIDTH   (ADDR_WIDTH),
        .TAG_WIDTH    (TAG_WIDTH),
        .DMA_ENABLE   (1),
        .DMA_TAG_WIDTH(DMA_TAG_WIDTH),
        .AMO_ENABLE   (`VX_CFG_EXT_A_ENABLED),
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        .STORE_ACK_MASK (NUM_REQS'(1) << (NUM_REQS - 1)),
        .COMMIT_MASK  ((NUM_REQS'(1) << (NUM_REQS - 1)) - NUM_REQS'(1)),
    `endif
        .OUT_BUF      (3)
    ) local_mem (
        .clk        (clk),
        .reset      (reset),
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        .st_commit_valid (st_commit_valid),
        .st_commit_wid   (st_commit_wid_w),
    `endif
        .lsu_bus_if (lsu_bus_if),
        .dma_bus_if (dma_bus_if)
    );

endmodule
