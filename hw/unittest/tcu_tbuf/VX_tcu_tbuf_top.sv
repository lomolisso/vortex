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

// Flat-port wrapper around VX_tcu_tbuf: the per-block micro-op requests in,
// the arbitrated local-memory read port out, the operand buses and ready out.

module VX_tcu_tbuf_top import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter BLOCK_SIZE      = `VX_CFG_NUM_TCU_BLOCKS,
    parameter NUM_BANKS       = `VX_CFG_LMEM_NUM_BANKS,
    parameter BANK_ADDR_WIDTH = `VX_CFG_LMEM_LOG_SIZE
                              - $clog2(`VX_CFG_XLEN / 8)
                              - $clog2(`VX_CFG_LMEM_NUM_BANKS)
) (
    input  wire clk,
    input  wire reset,

    input  wire [BLOCK_SIZE-1:0]                                req_valid,
    input  wire [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0]                req_cta_id,
    input  wire [BLOCK_SIZE-1:0][3:0]                           req_step_m,
    input  wire [BLOCK_SIZE-1:0][4:0]                           req_step_n,
    input  wire [BLOCK_SIZE-1:0][3:0]                           req_step_k,
    input  wire [BLOCK_SIZE-1:0][1:0]                           req_cd_nregs,
    input  wire [BLOCK_SIZE-1:0][63:0]                          req_desc_a,
    input  wire [BLOCK_SIZE-1:0][63:0]                          req_desc_b,
    input  wire [BLOCK_SIZE-1:0]                                req_is_first_uop,
    input  wire [BLOCK_SIZE-1:0]                                req_is_last_uop,
    input  wire [BLOCK_SIZE-1:0][`VX_CFG_LMEM_LOG_SIZE-1:0]     req_slot_base,
    input  wire [BLOCK_SIZE-1:0]                                req_esize16,

    output wire                                                 lmem_req_valid,
    input  wire                                                 lmem_req_ready,
    output wire [BANK_ADDR_WIDTH-1:0]                           lmem_req_addr,
    output wire                                                 lmem_req_rw,
    output wire [TCU_LMEM_TAG_W-1:0]                            lmem_req_tag,
    input  wire                                                 lmem_rsp_valid,
    input  wire [NUM_BANKS*`VX_CFG_XLEN-1:0]                    lmem_rsp_data,
    input  wire [TCU_LMEM_TAG_W-1:0]                            lmem_rsp_tag,
    output wire                                                 lmem_rsp_ready,

    output wire [BLOCK_SIZE-1:0][TCU_WG_A_DATA_SIZE-1:0][31:0]  rs1_data,
    output wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][31:0]    rs2_data,
    output wire [BLOCK_SIZE-1:0]                                ready
);
    `STATIC_ASSERT (`VX_CFG_XLEN == 64, ("tcu_tbuf_top: XLEN 64 only"))

    tcu_tbuf_req_t [BLOCK_SIZE-1:0] req;
    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_req
        always_comb begin
            req[b]              = '0;
            req[b].valid        = req_valid[b];
            req[b].wid          = NW_WIDTH'(b);
            req[b].cta_id       = req_cta_id[b];
            req[b].step_m       = req_step_m[b];
            req[b].step_n       = $bits(req[b].step_n)'(req_step_n[b]);
            req[b].step_k       = req_step_k[b];
            req[b].cd_nregs     = req_cd_nregs[b];
            req[b].desc_a       = req_desc_a[b];
            req[b].desc_b       = req_desc_b[b];
            req[b].a_is_smem    = 1'b1;
            req[b].is_first_uop = req_is_first_uop[b];
            req[b].is_last_uop  = req_is_last_uop[b];
            req[b].setup_fire   = 1'b0;
        end
    end

    VX_mem_bus_if #(
        .DATA_SIZE  (NUM_BANKS * (`VX_CFG_XLEN / 8)),
        .TAG_WIDTH  (TCU_LMEM_TAG_W),
        .ATTR_WIDTH (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH (BANK_ADDR_WIDTH)
    ) lmem_if();

    assign lmem_req_valid        = lmem_if.req_valid;
    assign lmem_if.req_ready     = lmem_req_ready;
    assign lmem_req_addr         = lmem_if.req_data.addr;
    assign lmem_req_rw           = lmem_if.req_data.rw;
    assign lmem_req_tag          = lmem_if.req_data.tag;
    assign lmem_if.rsp_valid     = lmem_rsp_valid;
    assign lmem_if.rsp_data.data = lmem_rsp_data;
    assign lmem_if.rsp_data.tag  = lmem_rsp_tag;
    assign lmem_rsp_ready        = lmem_if.rsp_ready;
    `UNUSED_VAR ({lmem_if.req_data.data, lmem_if.req_data.byteen, lmem_if.req_data.attr})

    wire [BLOCK_SIZE-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] rs1_w;
    wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   rs2_w;

    VX_tcu_tbuf #(
        .INSTANCE_ID     ("tbuf"),
        .NUM_BANKS       (NUM_BANKS),
        .BANK_ADDR_WIDTH (BANK_ADDR_WIDTH),
        .BLOCK_SIZE      (BLOCK_SIZE)
    ) tbuf (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        `UNUSED_PIN (tbuf_stalls),
        `UNUSED_PIN (tbuf_cache_hits),
        `UNUSED_PIN (lmem_reads),
    `endif
        .req           (req),
        .req_slot_base (req_slot_base),
        .req_esize16   (req_esize16),
        .tcu_lmem_if   (lmem_if),
        .tbuf_rs1_data (rs1_w),
        .tbuf_rs2_data (rs2_w),
        .tbuf_ready    (ready)
    );

    for (genvar b = 0; b < BLOCK_SIZE; ++b) begin : g_out
        for (genvar l = 0; l < TCU_WG_A_DATA_SIZE; ++l) begin : g_a
            assign rs1_data[b][l] = rs1_w[b][l][31:0];
            `UNUSED_VAR (rs1_w[b][l][`VX_CFG_XLEN-1:32])
        end
        for (genvar l = 0; l < TCU_WG_RS2_WIDTH; ++l) begin : g_b
            assign rs2_data[b][l] = rs2_w[b][l][31:0];
            `UNUSED_VAR (rs2_w[b][l][`VX_CFG_XLEN-1:32])
        end
    end

endmodule
