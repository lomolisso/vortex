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

// One cluster's DLMEM endpoints: per core an egress, an ingress and a store
// tracker, joined by a request network and a separate response network.
// Ports are flat; an LSU port index is p = core * NB + block. Field containers
// are fixed (64-bit word addresses, 32-bit hart ids, 8-bit tags/wids/counts) so
// the testbench needs only NCC, NB, NL and NW from VX_config.h; the top checks
// that every real width fits its container.
module VX_dlmem_ep_top import VX_gpu_pkg::*, VX_dlmem_pkg::*; #(
    parameter LRSC_ASSERT = 0,
    parameter OUT_BUF     = 3,

    localparam NCC = DLMEM_NCC,
    localparam NB  = `VX_CFG_NUM_LSU_BLOCKS,
    localparam NL  = `VX_CFG_NUM_LSU_LANES,
    localparam NW  = `VX_CFG_NUM_WARPS,
    localparam NP  = NCC * NB,
    localparam WS  = LSU_WORD_SIZE,
    localparam XW  = `VX_CFG_XLEN,
    localparam NS  = NUM_SOCKETS,
    localparam DMA_DW = LMEM_DMA_DATA_SIZE * 8,
    localparam DMA_BW = LMEM_DMA_DATA_SIZE
) (
    input wire clk,
    input wire reset,

    // LSU requests: remote lanes only, word addresses in the DLMEM aperture
    input  wire [NP-1:0]                    lsu_req_valid,
    input  wire [NP-1:0]                    lsu_req_rw,
    input  wire [NP-1:0][NL-1:0]            lsu_req_mask,
    input  wire [NP-1:0][NL-1:0][63:0]      lsu_req_addr,
    input  wire [NP-1:0][NL-1:0][XW-1:0]    lsu_req_data,
    input  wire [NP-1:0][NL-1:0][WS-1:0]    lsu_req_byteen,
    input  wire [NP-1:0][NL-1:0]            lsu_req_amo,
    input  wire [NP-1:0][NL-1:0][3:0]       lsu_req_amo_op,
    input  wire [NP-1:0][NL-1:0]            lsu_req_amo_unsigned,
    input  wire [NP-1:0][NL-1:0][XW-1:0]    lsu_req_amo_cmp,
    input  wire [NP-1:0][NL-1:0][31:0]      lsu_req_hart_id,
    input  wire [NP-1:0][7:0]               lsu_req_tag,
    output wire [NP-1:0]                    lsu_req_ready,

    // LSU responses: one lane per beat
    output wire [NP-1:0]                    lsu_rsp_valid,
    output wire [NP-1:0][NL-1:0]            lsu_rsp_mask,
    output wire [NP-1:0][NL-1:0][XW-1:0]    lsu_rsp_data,
    output wire [NP-1:0][7:0]               lsu_rsp_tag,
    input  wire [NP-1:0]                    lsu_rsp_ready,

    // Local-memory port of each core's ingress (word address within the core)
    output wire [NCC-1:0]                   lmem_req_valid,
    output wire [NCC-1:0]                   lmem_req_rw,
    output wire [NCC-1:0][31:0]             lmem_req_addr,
    output wire [NCC-1:0][XW-1:0]           lmem_req_data,
    output wire [NCC-1:0][WS-1:0]           lmem_req_byteen,
    output wire [NCC-1:0]                   lmem_req_local,
    output wire [NCC-1:0]                   lmem_req_amo,
    output wire [NCC-1:0][3:0]              lmem_req_amo_op,
    output wire [NCC-1:0]                   lmem_req_amo_unsigned,
    output wire [NCC-1:0][XW-1:0]           lmem_req_amo_cmp,
    output wire [NCC-1:0][31:0]             lmem_req_hart_id,
    output wire [NCC-1:0][7:0]              lmem_req_tag,
    input  wire [NCC-1:0]                   lmem_req_ready,

    // Every read, AMO and store gets one response, the store's after it committed
    input  wire [NCC-1:0]                   lmem_rsp_valid,
    input  wire [NCC-1:0][XW-1:0]           lmem_rsp_data,
    input  wire [NCC-1:0][7:0]              lmem_rsp_tag,
    output wire [NCC-1:0]                   lmem_rsp_ready,

    // Store tracker: increments as the translation stage would issue them
    input  wire [NP-1:0]                    wt_inc_valid,
    input  wire [NP-1:0][7:0]               wt_inc_wid,
    input  wire [NP-1:0][7:0]               wt_inc_cnt,
    output wire [NCC-1:0][NW-1:0]           drained,

    output wire [NCC-1:0]                   egress_empty,
    output wire [NCC-1:0]                   ingress_empty,
    output wire [NCC-1:0]                   wtrack_empty,

    // Observation of the endpoint/network handshakes. An ingress slot frees
    // when its response enters the ingress output buffer, so req_ej - rsp_inj
    // can exceed the slot count by that buffer's two entries.
    output wire [NCC-1:0]                   ack_valid,      // egress -> tracker decrement
    output wire [NCC-1:0][7:0]              ack_wid,
    output wire [NCC-1:0]                   req_inj_fire,   // egress -> request network
    output wire [NCC-1:0][1:0]              req_inj_op,
    output wire [NCC-1:0][7:0]              req_inj_dst,
    output wire [NCC-1:0]                   req_ej_fire,    // request network -> ingress
    output wire [NCC-1:0]                   rsp_inj_fire,   // ingress -> response network
    output wire [NCC-1:0]                   rsp_inj_ack,
    output wire [NCC-1:0]                   rsp_ej_fire,    // response network -> egress
    output wire [NCC-1:0]                   rsp_ej_ack,
    output wire [NCC-1:0]                   rsp_ej_ready

`ifdef DXA_MBAR_ENABLE
    ,
    // Posted row flits of each socket's DXA egress, as packet fields: a data
    // word, or with row_compl the copy's complete_tx.
    input  wire [NS-1:0]                    row_valid,
    input  wire [NS-1:0][7:0]               row_dst,
    input  wire [NS-1:0][7:0]               row_src,
    input  wire [NS-1:0]                    row_compl,
    input  wire [NS-1:0][31:0]              row_addr,
    input  wire [NS-1:0][WS-1:0]            row_byteen,
    input  wire [NS-1:0][XW-1:0]            row_data,
    input  wire [NS-1:0][7:0]               row_slot,
    output wire [NS-1:0]                    row_ready,

    // Each core's occupied CTA slots, and its local memory's DMA port.
    input  wire [NCC-1:0][NW-1:0]           slot_valid,
    output wire [NCC-1:0]                   dma_valid,
    output wire [NCC-1:0][31:0]             dma_addr,
    output wire [NCC-1:0][DMA_DW-1:0]       dma_data,
    output wire [NCC-1:0][DMA_BW-1:0]       dma_byteen,
    output wire [NCC-1:0][7:0]              dma_slot,
    output wire [NCC-1:0]                   dma_remote,
    output wire [NCC-1:0]                   dma_last,
    input  wire [NCC-1:0]                   dma_ready,

    output wire [31:0]                      cfg_amo_complete_tx
`endif
);
`ifndef DXA_MBAR_ENABLE
    `UNUSED_PARAM (NS)
    `UNUSED_PARAM (DMA_DW)
    `UNUSED_PARAM (DMA_BW)
`endif
    localparam TAGV_W = LSU_TAG_WIDTH - UUID_WIDTH;
    localparam CNT_W  = `CLOG2(NL + 1);

    `STATIC_ASSERT (XW == 64, ("dlmem_ep_top: DLMEM is RV64 only"))
    `STATIC_ASSERT (LSU_ADDR_WIDTH <= 64, ("dlmem_ep_top: address container too small"))
    `STATIC_ASSERT (HART_ID_WIDTH <= 32, ("dlmem_ep_top: hart id container too small"))
    `STATIC_ASSERT (TAGV_W <= 8, ("dlmem_ep_top: tag container too small"))
    `STATIC_ASSERT (NW_WIDTH <= 8 && CNT_W <= 8, ("dlmem_ep_top: wid/count container too small"))
    `STATIC_ASSERT (DLMEM_LMEM_AW <= 32, ("dlmem_ep_top: local address container too small"))
    `STATIC_ASSERT ($bits(amo_op_e) == 4, ("dlmem_ep_top: amo_op_e width changed"))
    `STATIC_ASSERT (DLMEM_CORE_W <= 8, ("dlmem_ep_top: dst container too small"))

    // Request network
    wire [NCC-1:0]                       reqn_in_valid,  reqn_in_ready;
    wire [NCC-1:0][DLMEM_REQ_DATAW-1:0]  reqn_in_data;
    wire [NCC-1:0][DLMEM_CORE_W-1:0]     reqn_in_dst;
    wire [NCC-1:0]                       reqn_out_valid, reqn_out_ready;
    wire [NCC-1:0][DLMEM_REQ_DATAW-1:0]  reqn_out_data;
    wire [NCC-1:0][DLMEM_CORE_W-1:0]     reqn_out_dst;

    // Response network
    wire [NCC-1:0]                       rspn_in_valid,  rspn_in_ready;
    wire [NCC-1:0][DLMEM_RSP_DATAW-1:0]  rspn_in_data;
    wire [NCC-1:0][DLMEM_CORE_W-1:0]     rspn_in_dst;
    wire [NCC-1:0]                       rspn_out_valid, rspn_out_ready;
    wire [NCC-1:0][DLMEM_RSP_DATAW-1:0]  rspn_out_data;
    wire [NCC-1:0][DLMEM_CORE_W-1:0]     rspn_out_dst;

`ifdef DXA_MBAR_ENABLE
    `STATIC_ASSERT (DXA_CTA_GEN_W <= 8 && NW_WIDTH <= 8, ("dlmem_ep_top: slot container too small"))
    assign cfg_amo_complete_tx = 32'(AMO_OP_MBAR_COMPLETE_TX);

    wire [NS-1:0][DLMEM_REQ_DATAW-1:0] row_pkt_data;
    wire [NS-1:0][DLMEM_CORE_W-1:0]    row_pkt_dst;
    for (genvar s = 0; s < NS; ++s) begin : g_row_pkts
        dlmem_req_pkt_t pkt;
        always @(*) begin
            pkt              = '0;
            pkt.src          = DLMEM_CORE_W'(row_src[s]);
            pkt.op           = DLMEM_OP_DXA;
            pkt.addr         = DLMEM_LMEM_AW'(row_addr[s]);
            pkt.byteen       = row_byteen[s];
            pkt.data         = row_data[s];
            pkt.amo_op       = row_compl[s] ? AMO_OP_MBAR_COMPLETE_TX : amo_op_e'(0);
            pkt.amo_unsigned = row_compl[s];
            pkt.tag          = DLMEM_TAG_W'(row_slot[s]);
        end
        assign row_pkt_data[s] = pkt;
        assign row_pkt_dst[s]  = DLMEM_CORE_W'(row_dst[s]);
        `UNUSED_VAR (row_src[s])
        `UNUSED_VAR (row_dst[s])
        `UNUSED_VAR (row_addr[s])
        `UNUSED_VAR (row_slot[s])
    end
`endif

    VX_dlmem_noc_top #(
        .DATAW   (DLMEM_REQ_DATAW),
        .OUT_BUF (OUT_BUF)
    ) req_net (
        .clk       (clk),
        .reset     (reset),
        .in_valid  (reqn_in_valid),
        .in_data   (reqn_in_data),
        .in_dst    (reqn_in_dst),
        .in_ready  (reqn_in_ready),
    `ifdef DXA_MBAR_ENABLE
        .row_valid (row_valid),
        .row_data  (row_pkt_data),
        .row_dst   (row_pkt_dst),
        .row_ready (row_ready),
    `endif
        .out_valid (reqn_out_valid),
        .out_data  (reqn_out_data),
        .out_dst   (reqn_out_dst),
        .out_ready (reqn_out_ready)
    );

    VX_dlmem_noc_top #(
        .DATAW   (DLMEM_RSP_DATAW),
        .OUT_BUF (OUT_BUF)
    ) rsp_net (
        .clk       (clk),
        .reset     (reset),
        .in_valid  (rspn_in_valid),
        .in_data   (rspn_in_data),
        .in_dst    (rspn_in_dst),
        .in_ready  (rspn_in_ready),
    `ifdef DXA_MBAR_ENABLE
        .row_valid ('0),
        .row_data  ('0),
        .row_dst   ('0),
        `UNUSED_PIN (row_ready),
    `endif
        .out_valid (rspn_out_valid),
        .out_data  (rspn_out_data),
        .out_dst   (rspn_out_dst),
        .out_ready (rspn_out_ready)
    );

    for (genvar c = 0; c < NCC; ++c) begin : g_cores

        // LSU ports

        VX_lsu_mem_if #(
            .NUM_LANES (NL),
            .DATA_SIZE (WS),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) lsu_if[NB]();

        for (genvar b = 0; b < NB; ++b) begin : g_blocks
            localparam p = c * NB + b;

            for (genvar l = 0; l < NL; ++l) begin : g_lanes
                mem_bus_attr_t attr;
                always @(*) begin
                    attr = '0;
                    attr.amo.amo_valid    = lsu_req_amo[p][l];
                    attr.amo.amo_op       = amo_op_e'(lsu_req_amo_op[p][l]);
                    attr.amo.amo_unsigned = lsu_req_amo_unsigned[p][l];
                    attr.amo.hart_id      = HART_ID_WIDTH'(lsu_req_hart_id[p][l]);
                `ifdef VX_CFG_EXT_ZACAS_ENABLE
                    attr.amo.amo_cmp      = lsu_req_amo_cmp[p][l];
                `endif
                end
                assign lsu_if[b].req_data.user[l] = attr;
                assign lsu_if[b].req_data.addr[l] = LSU_ADDR_WIDTH'(lsu_req_addr[p][l]);
                `UNUSED_VAR (lsu_req_addr[p][l])
                `UNUSED_VAR (lsu_req_hart_id[p][l])
            `ifndef VX_CFG_EXT_ZACAS_ENABLE
                `UNUSED_VAR (lsu_req_amo_cmp[p][l])
            `endif
            end

            assign lsu_if[b].req_valid       = lsu_req_valid[p];
            assign lsu_if[b].req_data.rw     = lsu_req_rw[p];
            assign lsu_if[b].req_data.mask   = lsu_req_mask[p];
            assign lsu_if[b].req_data.data   = lsu_req_data[p];
            assign lsu_if[b].req_data.byteen = lsu_req_byteen[p];
            assign lsu_if[b].req_data.tag    = {UUID_WIDTH'(0), TAGV_W'(lsu_req_tag[p])};
            assign lsu_req_ready[p] = lsu_if[b].req_ready;
            `UNUSED_VAR (lsu_req_tag[p])

            assign lsu_rsp_valid[p] = lsu_if[b].rsp_valid;
            assign lsu_rsp_mask[p]  = lsu_if[b].rsp_data.mask;
            assign lsu_rsp_data[p]  = lsu_if[b].rsp_data.data;
            assign lsu_rsp_tag[p]   = 8'(lsu_if[b].rsp_data.tag.value);
            assign lsu_if[b].rsp_ready = lsu_rsp_ready[p];
            `UNUSED_VAR (lsu_if[b].rsp_data.tag.uuid)
        end

        // Network attachments

        VX_dlmem_bus_if #(
            .DATAW (DLMEM_REQ_DATAW),
            .DST_W (DLMEM_CORE_W)
        ) req_out_if();

        VX_dlmem_bus_if #(
            .DATAW (DLMEM_REQ_DATAW),
            .DST_W (DLMEM_CORE_W)
        ) req_in_if();

        VX_dlmem_bus_if #(
            .DATAW (DLMEM_RSP_DATAW),
            .DST_W (DLMEM_CORE_W)
        ) rsp_out_if();

        VX_dlmem_bus_if #(
            .DATAW (DLMEM_RSP_DATAW),
            .DST_W (DLMEM_CORE_W)
        ) rsp_in_if();

        assign reqn_in_valid[c]   = req_out_if.valid;
        assign reqn_in_data[c]    = req_out_if.data;
        assign reqn_in_dst[c]     = req_out_if.dst;
        assign req_out_if.ready   = reqn_in_ready[c];

        assign req_in_if.valid    = reqn_out_valid[c];
        assign req_in_if.data     = reqn_out_data[c];
        assign req_in_if.dst      = reqn_out_dst[c];
        assign reqn_out_ready[c]  = req_in_if.ready;

        assign rspn_in_valid[c]   = rsp_out_if.valid;
        assign rspn_in_data[c]    = rsp_out_if.data;
        assign rspn_in_dst[c]     = rsp_out_if.dst;
        assign rsp_out_if.ready   = rspn_in_ready[c];

        assign rsp_in_if.valid    = rspn_out_valid[c];
        assign rsp_in_if.data     = rspn_out_data[c];
        assign rsp_in_if.dst      = rspn_out_dst[c];
        assign rspn_out_ready[c]  = rsp_in_if.ready;

        // Endpoints

        wire                ack_valid_w;
        wire [NW_WIDTH-1:0] ack_wid_w;

        VX_dlmem_egress #(
            .INSTANCE_ID   ($sformatf("dlmem-egress%0d", c)),
            .CORE_ID       (c),
            .NUM_BLOCKS    (NB),
            .NUM_LANES     (NL),
            .LRSC_ASSERT   (LRSC_ASSERT),
            .OUT_BUF       (OUT_BUF)
        ) egress (
            .clk        (clk),
            .reset      (reset),
            .lsu_in_if  (lsu_if),
            .req_out_if (req_out_if),
            .rsp_in_if  (rsp_in_if),
            .ack_valid  (ack_valid_w),
            .ack_wid    (ack_wid_w),
            // The bench counts stores only, so the tracker takes acks alone.
            `UNUSED_PIN (done_valid),
            `UNUSED_PIN (done_wid),
            .empty      (egress_empty[c])
        );

        VX_mem_bus_if #(
            .DATA_SIZE  (WS),
            .TAG_WIDTH  (LSU_TAG_WIDTH),
            .ADDR_WIDTH (DLMEM_LMEM_AW)
        ) lmem_bus_if();

    `ifdef DXA_MBAR_ENABLE
        VX_mem_bus_if #(
            .DATA_SIZE  (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH  (LMEM_DMA_IN_TAG_MAX),
            .ATTR_WIDTH (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH (LMEM_DMA_ADDR_WIDTH)
        ) dma_if();

        dxa_lmem_attr_t dma_attr;
        assign dma_attr = dxa_lmem_attr_t'(dma_if.req_data.attr);

        assign dma_valid[c]    = dma_if.req_valid;
        assign dma_addr[c]     = 32'(dma_if.req_data.addr);
        assign dma_data[c]     = dma_if.req_data.data;
        assign dma_byteen[c]   = dma_if.req_data.byteen;
        assign dma_slot[c]     = 8'(dma_attr.slot);
        assign dma_remote[c]   = dma_attr.remote && dma_if.req_data.rw;
        assign dma_last[c]     = dma_attr.last;
        assign dma_if.req_ready = dma_ready[c];
        assign dma_if.rsp_valid = 1'b0;
        assign dma_if.rsp_data  = '0;
        `UNUSED_VAR (dma_attr)
        `UNUSED_VAR (dma_if.req_data.tag)
        `UNUSED_VAR (dma_if.rsp_ready)
    `endif

        VX_dlmem_ingress #(
            .INSTANCE_ID ($sformatf("dlmem-ingress%0d", c)),
            .OUT_BUF     (OUT_BUF)
        ) ingress (
            .clk         (clk),
            .reset       (reset),
            .req_in_if   (req_in_if),
            .lmem_bus_if (lmem_bus_if),
            .rsp_out_if  (rsp_out_if),
        `ifdef DXA_MBAR_ENABLE
            .slot_valid  (slot_valid[c]),
            .dma_out_if  (dma_if),
        `endif
            .empty       (ingress_empty[c])
        );

        mem_bus_attr_t lmem_attr;
        assign lmem_attr = mem_bus_attr_t'(lmem_bus_if.req_data.attr);

        assign lmem_req_valid[c]        = lmem_bus_if.req_valid;
        assign lmem_req_rw[c]           = lmem_bus_if.req_data.rw;
        assign lmem_req_addr[c]         = 32'(lmem_bus_if.req_data.addr);
        assign lmem_req_data[c]         = lmem_bus_if.req_data.data;
        assign lmem_req_byteen[c]       = lmem_bus_if.req_data.byteen;
        assign lmem_req_local[c]        = lmem_attr.is_addr_local;
        assign lmem_req_amo[c]          = lmem_attr.amo.amo_valid;
        assign lmem_req_amo_op[c]       = lmem_attr.amo.amo_op;
        assign lmem_req_amo_unsigned[c] = lmem_attr.amo.amo_unsigned;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        assign lmem_req_amo_cmp[c]      = lmem_attr.amo.amo_cmp;
    `else
        assign lmem_req_amo_cmp[c]      = '0;
    `endif
        assign lmem_req_hart_id[c]      = 32'(lmem_attr.amo.hart_id);
        assign lmem_req_tag[c]          = 8'(lmem_bus_if.req_data.tag.value);
        assign lmem_bus_if.req_ready    = lmem_req_ready[c];
        `UNUSED_VAR (lmem_attr)
        `UNUSED_VAR (lmem_bus_if.req_data.tag.uuid)

        assign lmem_bus_if.rsp_valid          = lmem_rsp_valid[c];
        assign lmem_bus_if.rsp_data.data      = lmem_rsp_data[c];
        assign lmem_bus_if.rsp_data.tag       = {UUID_WIDTH'(0), TAGV_W'(lmem_rsp_tag[c])};
        assign lmem_rsp_ready[c] = lmem_bus_if.rsp_ready;
        `UNUSED_VAR (lmem_rsp_tag[c])

        // Store tracker

        wire [NB-1:0]                inc_valid_w;
        wire [NB-1:0][NW_WIDTH-1:0]  inc_wid_w;
        wire [NB-1:0][CNT_W-1:0]     inc_cnt_w;
        for (genvar b = 0; b < NB; ++b) begin : g_inc
            assign inc_valid_w[b] = wt_inc_valid[c * NB + b];
            assign inc_wid_w[b]   = NW_WIDTH'(wt_inc_wid[c * NB + b]);
            assign inc_cnt_w[b]   = CNT_W'(wt_inc_cnt[c * NB + b]);
            `UNUSED_VAR (wt_inc_wid[c * NB + b])
            `UNUSED_VAR (wt_inc_cnt[c * NB + b])
        end

        VX_dlmem_wtrack #(
            .NUM_INC (NB)
        ) wtrack (
            .clk       (clk),
            .reset     (reset),
            .inc_valid (inc_valid_w),
            .inc_wid   (inc_wid_w),
            .inc_cnt   (inc_cnt_w),
            .dec_valid (ack_valid_w),
            .dec_wid   (ack_wid_w),
            .drained   (drained[c]),
            .empty     (wtrack_empty[c])
        );

        // Observation

        dlmem_req_pkt_t req_inj_pkt;
        dlmem_rsp_pkt_t rsp_inj_pkt, rsp_ej_pkt;
        assign req_inj_pkt = dlmem_req_pkt_t'(reqn_in_data[c]);
        assign rsp_inj_pkt = dlmem_rsp_pkt_t'(rspn_in_data[c]);
        assign rsp_ej_pkt  = dlmem_rsp_pkt_t'(rspn_out_data[c]);
        `UNUSED_VAR (req_inj_pkt)
        `UNUSED_VAR (rsp_inj_pkt)
        `UNUSED_VAR (rsp_ej_pkt)

        assign ack_valid[c]    = ack_valid_w;
        assign ack_wid[c]      = 8'(ack_wid_w);
        assign req_inj_fire[c] = reqn_in_valid[c] && reqn_in_ready[c];
        assign req_inj_op[c]   = req_inj_pkt.op;
        assign req_inj_dst[c]  = 8'(reqn_in_dst[c]);
        assign req_ej_fire[c]  = reqn_out_valid[c] && reqn_out_ready[c];
        assign rsp_inj_fire[c] = rspn_in_valid[c] && rspn_in_ready[c];
        assign rsp_inj_ack[c]  = rsp_inj_pkt.ack;
        assign rsp_ej_fire[c]  = rspn_out_valid[c] && rspn_out_ready[c];
        assign rsp_ej_ack[c]   = rsp_ej_pkt.ack;
        assign rsp_ej_ready[c] = rspn_out_ready[c];
    end

endmodule
