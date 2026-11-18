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

module VX_mem_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

`ifdef VX_CFG_H100_SM_ENABLE
    input smem_cfg_t        smem_cfg,
`endif

`ifdef PERF_ENABLE
    output lmem_perf_t      lmem_perf,
    output coalescer_perf_t coalescer_perf,
`endif

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_mem_bus_if.slave     dxa_lmem_bus_if,
    VX_txbar_bus_if.master  dxa_txbar_bus_if,
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // TCU LMEM read port
    VX_mem_bus_if.slave     tcu_lmem_if,
`endif

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // Remote lanes toward the DLMEM egress, and the ingress's local-memory port
    VX_lsu_mem_if.master    dlmem_remote_if [`VX_CFG_NUM_LSU_BLOCKS],
    VX_mem_bus_if.slave     dlmem_lmem_if,
`ifdef VX_CFG_H100_SM_ENABLE
    // The array's counted shared-memory lanes once final, and their warps
    output wire [2:0]                lmem_commit_valid,
    output wire [2:0][NW_WIDTH-1:0]  lmem_commit_wid,
    output wire [2:0][`CLOG2(`VX_CFG_NUM_LSU_LANES+1)-1:0] lmem_commit_cnt,
`else
    // Per local-memory bank, an LSU access performed in the previous cycle, and its warp
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0] lmem_commit_valid,
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0][NW_WIDTH-1:0] lmem_commit_wid,
`endif
`endif

`ifdef VX_CFG_EXT_MBAR_ENABLE
    // Per LSU block, the retire of an mbarrier try_wait; the warps it releases
    input wire [`VX_CFG_NUM_LSU_BLOCKS-1:0]               mbar_retire_valid,
    input wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][NW_WIDTH-1:0] mbar_retire_wid,
    output wire                                          mbar_unlock_valid,
    output wire [`VX_CFG_NUM_WARPS-1:0]                  mbar_unlock_mask,
    // Per LSU block, the cp.async unit's writes and arrives
    VX_lsu_mem_if.slave                                  cpa_lmem_if [`VX_CFG_NUM_LSU_BLOCKS],
`endif

`ifdef DXA_MBAR_ENABLE
    // The core's CTA slots, for the DXA's retired-slot and bounds checks
    input wire [`VX_CFG_NUM_WARPS-1:0]                    cta_slot_valid,
    input wire [`VX_CFG_NUM_WARPS-1:0][DXA_CTA_GEN_W-1:0] cta_slot_gen,
    input wire [`VX_CFG_LMEM_LOG_SIZE:0]                  cta_slot_stride,
    // Data words of remote DXA copies, from the DLMEM ingress
    VX_mem_bus_if.slave                                   dlmem_dma_if,
`endif

    output wire             empty,

    VX_lsu_mem_if.slave     lsu_mem_if [`VX_CFG_NUM_LSU_BLOCKS],
    VX_dcr_flush_if.slave   dcr_flush_if,
`ifdef VX_CFG_H100_SM_ENABLE
`ifdef PERF_ENABLE
    output cache_perf_t     dcache_perf,
`endif
    VX_mem_bus_if.master    dcache_bus_if [1]
`else
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS]
`endif
);
`ifndef VX_CFG_H100_SM_ENABLE
    // Per-block empty: high when the global-store path holds no in-flight
    // write. Folded into core `busy` so the end-of-kernel cache flush is
    // ordered strictly behind the last store. Two terms: the coalescer/passthru
    // producer, and the dcache adapter's request buffer (which carries state
    // when REQ_OUT_BUF > 0 and would otherwise hide a still-settling store).
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] per_block_path_empty;
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] per_block_adapter_empty;
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] per_block_empty;
    assign per_block_empty = per_block_path_empty & per_block_adapter_empty;
`endif
    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_dcache_if[`VX_CFG_NUM_LSU_BLOCKS]();

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    `STATIC_ASSERT(`VX_CFG_LMEM_ENABLED, ("TCU_WGMMA_ENABLE requires LMEM_ENABLE"))
`ifdef VX_CFG_TCU_H100_ENABLE
    // H100 descriptors carry 14-bit fields of 16-byte units: 256 KB.
    `STATIC_ASSERT(`VX_CFG_LMEM_LOG_SIZE <= 18, ("LMEM exceeds the 256 KB reach of the wgmma descriptor"))
`else
    // The WGMMA shared-memory descriptor carries the LMEM byte offset in a 16-bit
    // field, so a WGMMA-visible LMEM cannot exceed 64KB; beyond that the offset
    // wraps into the adjacent stride field.
    `STATIC_ASSERT(`VX_CFG_LMEM_LOG_SIZE <= 16, ("LMEM exceeds the 16-bit smem descriptor offset field"))
`endif
`endif

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    `STATIC_ASSERT(`VX_CFG_LMEM_ENABLED, ("DLMEM_ENABLE requires LMEM_ENABLE"))
`endif

`ifdef VX_CFG_LMEM_ENABLE

    `STATIC_ASSERT(`IS_DIVISBLE((1 << `VX_CFG_LMEM_LOG_SIZE), `VX_CFG_MEM_BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT(0 == (`VX_MEM_LMEM_BASE_ADDR % (1 << `VX_CFG_LMEM_LOG_SIZE)), ("invalid parameter"))

    localparam LMEM_ADDR_WIDTH = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE);

    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_lmem_if[`VX_CFG_NUM_LSU_BLOCKS]();

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_lmem_switches
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        // Round-robin response merge, so no leg waits behind the other two.
        VX_lmem_switch #(
            .GLOBAL_OUT_BUF(1),
            .LOCAL_OUT_BUF(1),
            .REMOTE_OUT_BUF(1),
            .RSP_OUT_BUF  (1),
        `ifdef VX_CFG_H100_SM_ENABLE
            .GLOBAL_DEPTH (`VX_CFG_LSU_PENDING_SIZE),
            .LOCAL_DEPTH  (LMEM_SW_LOCAL_DEPTH),
        `endif
            .ARBITER      ("R")
        ) lmem_switch (
            .clk          (clk),
            .reset        (reset),
            .lsu_in_if    (lsu_mem_if[i]),
            .global_out_if(lsu_dcache_if[i]),
            .remote_out_if(dlmem_remote_if[i]),
            .local_out_if (lsu_lmem_if[i])
        );
    `else
        VX_lmem_switch #(
            .GLOBAL_OUT_BUF(1),
            .LOCAL_OUT_BUF(1),
            .RSP_OUT_BUF  (1),
        `ifdef VX_CFG_H100_SM_ENABLE
            .GLOBAL_DEPTH (`VX_CFG_LSU_PENDING_SIZE),
            .LOCAL_DEPTH  (LMEM_SW_LOCAL_DEPTH),
        `endif
            .ARBITER      ("P")
        ) lmem_switch (
            .clk          (clk),
            .reset        (reset),
            .lsu_in_if    (lsu_mem_if[i]),
            .global_out_if(lsu_dcache_if[i]),
            .local_out_if (lsu_lmem_if[i])
        );
    `endif
    end

`ifdef VX_CFG_H100_SM_ENABLE
    `STATIC_ASSERT((64'(1) << `VX_CFG_LMEM_LOG_SIZE) <= 64'(`VX_CFG_LMEM_DESC_REACH_BYTES), ("LMEM exceeds the wgmma descriptor reach"))

    // The local leg crosses the array's 128-B LSU path one request per beat; a
    // compare-and-swap takes two beats, its comparands as the second's data.
    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_sram_if[`VX_CFG_NUM_LSU_BLOCKS]();

    for (genvar b = 0; b < `VX_CFG_NUM_LSU_BLOCKS; ++b) begin : g_cas_beats
        reg cmp_beat_r;
        wire [`VX_CFG_NUM_LSU_LANES-1:0] lane_cas;
        wire [`VX_CFG_NUM_LSU_LANES-1:0][LSU_WORD_SIZE*8-1:0] beat_data;
        for (genvar i = 0; i < `VX_CFG_NUM_LSU_LANES; ++i) begin : g_lanes
            mem_bus_attr_t lane_attr;
            assign lane_attr = mem_bus_attr_t'(lsu_lmem_if[b].req_data.user[i]);
            assign lane_cas[i] = lsu_lmem_if[b].req_data.mask[i] && lane_attr.amo.amo_valid
                              && (lane_attr.amo.amo_op == AMO_OP_CAS);
        `ifdef VX_CFG_EXT_ZACAS_ENABLE
            // The comparand sits where the lane's data does in its word.
            wire hi_only = (lsu_lmem_if[b].req_data.byteen[i][LSU_WORD_SIZE/2-1:0] == '0);
            wire [LSU_WORD_SIZE*8-1:0] cmp_w = hi_only
                ? {lane_attr.amo.amo_cmp[LSU_WORD_SIZE*4-1:0], (LSU_WORD_SIZE*4)'(0)}
                : (LSU_WORD_SIZE*8)'(lane_attr.amo.amo_cmp);
            assign beat_data[i] = cmp_beat_r ? cmp_w : lsu_lmem_if[b].req_data.data[i];
        `else
            assign beat_data[i] = lsu_lmem_if[b].req_data.data[i];
        `endif
            `UNUSED_VAR (lane_attr)
        end
        wire is_cas = (|lane_cas);
        wire fire   = lsu_lmem_if[b].req_valid && lsu_sram_if[b].req_ready;

        assign lsu_sram_if[b].req_valid      = lsu_lmem_if[b].req_valid;
        assign lsu_sram_if[b].req_data.mask   = lsu_lmem_if[b].req_data.mask;
        assign lsu_sram_if[b].req_data.rw     = lsu_lmem_if[b].req_data.rw;
        assign lsu_sram_if[b].req_data.addr   = lsu_lmem_if[b].req_data.addr;
        assign lsu_sram_if[b].req_data.data   = beat_data;
        assign lsu_sram_if[b].req_data.byteen = lsu_lmem_if[b].req_data.byteen;
        assign lsu_sram_if[b].req_data.user   = lsu_lmem_if[b].req_data.user;
        assign lsu_sram_if[b].req_data.tag    = lsu_lmem_if[b].req_data.tag;
        assign lsu_lmem_if[b].req_ready       = lsu_sram_if[b].req_ready && (~is_cas || cmp_beat_r);

        always @(posedge clk) begin
            if (reset) begin
                cmp_beat_r <= 1'b0;
            end else if (fire && is_cas) begin
                cmp_beat_r <= ~cmp_beat_r;
            end
        end

        assign lsu_lmem_if[b].rsp_valid = lsu_sram_if[b].rsp_valid;
        assign lsu_lmem_if[b].rsp_data  = lsu_sram_if[b].rsp_data;
        assign lsu_sram_if[b].rsp_ready = lsu_lmem_if[b].rsp_ready;
    end

    // Every client lands on a port of the SM's one array; port accesses of
    // LMEM_DMA_DATA_SIZE take their part of an array row.
    localparam ROW_BYTES      = SM_SRAM_ROW_BYTES;
    localparam LMEM_ROW_PARTS = ROW_BYTES / LMEM_DMA_DATA_SIZE;
    localparam LMEM_PART_BITS = `CLOG2(LMEM_ROW_PARTS);
    localparam PB             = `UP(LMEM_PART_BITS);
    localparam ROW_WORDS      = ROW_BYTES / LSU_WORD_SIZE;
    localparam ROW_WORD_BITS  = `CLOG2(ROW_WORDS);
    localparam HPW_BITS       = `CLOG2(LSU_WORD_SIZE / `VX_CFG_LMEM_BANK_BYTES);
    localparam BANK_BITS      = `CLOG2(`VX_CFG_LMEM_NUM_BANKS);
    localparam LMEM_ROW_WIDTH = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(ROW_BYTES);
    `STATIC_ASSERT(ROW_BYTES % LMEM_DMA_DATA_SIZE == 0, ("the shared-memory port must divide an array row"))
    `STATIC_ASSERT(LMEM_ROW_WIDTH <= SM_SRAM_ROW_W, ("the shared-memory window exceeds the array"))
    `UNUSED_PARAM (LMEM_ADDR_WIDTH)
    `UNUSED_PARAM (PB)
    `UNUSED_PARAM (HPW_BITS)
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    localparam RT_TCU_W = TCU_LMEM_TAG_W + PB;
`else
    localparam RT_TCU_W = 1;
`endif
`ifdef VX_CFG_EXT_DXA_ENABLE
    localparam RT_DXA_W = DXA_LMEM_OUT_TAG_W + PB;
`else
    localparam RT_DXA_W = 1;
`endif
    localparam ROW_TAG_W = `MAX(`MAX(RT_TCU_W, RT_DXA_W), LSU_TAG_WIDTH + ROW_WORD_BITS);
    localparam RD_Q = 4;  // row-read responses a client may have between grant and pop

    // Row reads: per client a response queue; a client presents only with room for its reply.
    wire [2:0]                         rd_c_req;
    wire [2:0][SM_SRAM_ROW_W-1:0]      rd_c_row;
    wire [2:0][ROW_TAG_W-1:0]          rd_c_tag;
    wire [2:0]                         rd_valid, rd_ready, rd_rsp_valid;
    wire [2:0][SM_SRAM_ROW_W-1:0]      rd_row;
    wire [2:0][ROW_TAG_W-1:0]          rd_tag;
    wire [ROW_BYTES*8-1:0]             rd_rsp_data;
    wire [ROW_TAG_W-1:0]               rd_rsp_tag;
    wire [2:0]                         rd_c_fire, rd_c_rsp_valid, rd_c_rsp_ready;
    wire [2:0][ROW_BYTES*8-1:0]        rd_c_rsp_data;
    wire [2:0][ROW_TAG_W-1:0]          rd_c_rsp_tag;
    wire [2:0][`CLOG2(RD_Q+1)-1:0]     rd_c_out;

    for (genvar c = 0; c < 3; ++c) begin : g_rd_clients
        reg [`CLOG2(RD_Q+1)-1:0] infl_r;
        wire [`CLOG2(RD_Q+1)-1:0] q_size;
        wire q_empty;
        assign rd_c_out[c]  = infl_r + q_size;
        assign rd_valid[c]  = rd_c_req[c] && (rd_c_out[c] < `CLOG2(RD_Q+1)'(RD_Q));
        assign rd_row[c]    = rd_c_row[c];
        assign rd_tag[c]    = rd_c_tag[c];
        assign rd_c_fire[c] = rd_valid[c] && rd_ready[c];
        always @(posedge clk) begin
            if (reset) begin
                infl_r <= '0;
            end else begin
                infl_r <= infl_r + `CLOG2(RD_Q+1)'(rd_c_fire[c]) - `CLOG2(RD_Q+1)'(rd_rsp_valid[c]);
            end
        end
        VX_fifo_queue #(
            .DATAW (ROW_TAG_W + ROW_BYTES * 8),
            .DEPTH (RD_Q)
        ) rsp_q (
            .clk      (clk),
            .reset    (reset),
            .push     (rd_rsp_valid[c]),
            .pop      (rd_c_rsp_valid[c] && rd_c_rsp_ready[c]),
            .data_in  ({rd_rsp_tag, rd_rsp_data}),
            .data_out ({rd_c_rsp_tag[c], rd_c_rsp_data[c]}),
            .empty    (q_empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            .size     (q_size)
        );
        assign rd_c_rsp_valid[c] = ~q_empty;
        `RUNTIME_ASSERT(~rd_rsp_valid[c] || (q_size < `CLOG2(RD_Q+1)'(RD_Q)), ("%t: *** %s row-read queue %0d overflow", $time, INSTANCE_ID, c))
    end

    // Row writes (SramWrClient order: L1 refill, DXA, DSMEM).
    wire [2:0]          wr_valid, wr_ready;
    sm_sram_fill_t [2:0] wr_data;
    wire                wr_dsmem_busy;
    wire                wr_dsmem_last;
    `UNUSED_VAR (wr_ready)

    // One-lane atomics (DXA completion, DSMEM).
    wire [1:0]                         amo_valid, amo_ready, amo_rsp_valid;
    wire [1:0][SM_SRAM_ROW_W-1:0]      amo_row;
    wire [1:0][BANK_BITS-1:0]          amo_bank;
    wire [1:0][LSU_WORD_SIZE-1:0]      amo_byteen;
    wire [1:0][LSU_WORD_SIZE*8-1:0]    amo_wdata, amo_rsp_data;
    amo_req_t [1:0]                    amo_attr;
    wire [1:0][ROW_TAG_W-1:0]          amo_tag, amo_rsp_tag;

    // The L1's global stream and refill rows.
    wire                    gl_req_valid, gl_req_ready, gl_rsp_ready;
    sm_sram_req_t           gl_req_data;

    // TCU tile reads.
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire [PB-1:0] tcu_part = PB'(tcu_lmem_if.req_data.addr);
    assign rd_c_req[0] = tcu_lmem_if.req_valid;
    assign rd_c_row[0] = SM_SRAM_ROW_W'(tcu_lmem_if.req_data.addr >> LMEM_PART_BITS);
    assign rd_c_tag[0] = ROW_TAG_W'({tcu_lmem_if.req_data.tag, tcu_part});
    assign tcu_lmem_if.req_ready = rd_c_fire[0];
    `RUNTIME_ASSERT(~tcu_lmem_if.req_valid || ~tcu_lmem_if.req_data.rw, ("%t: *** %s TCU write to shared memory", $time, INSTANCE_ID))
    `UNUSED_VAR ({tcu_lmem_if.req_data.data, tcu_lmem_if.req_data.byteen, tcu_lmem_if.req_data.attr})
    wire [PB-1:0] tcu_rsp_part = PB'(rd_c_rsp_tag[0]);
    `UNUSED_VAR (tcu_rsp_part)
    assign tcu_lmem_if.rsp_valid     = rd_c_rsp_valid[0];
    assign tcu_lmem_if.rsp_data.data = rd_c_rsp_data[0][((LMEM_ROW_PARTS > 1) ? 32'(tcu_rsp_part) : 32'd0) * LMEM_DMA_DATA_SIZE * 8 +: LMEM_DMA_DATA_SIZE * 8];
    assign tcu_lmem_if.rsp_data.tag  = TCU_LMEM_TAG_W'(rd_c_rsp_tag[0] >> PB);
    assign rd_c_rsp_ready[0]         = tcu_lmem_if.rsp_ready;
`else
    assign rd_c_req[0] = 1'b0;
    assign rd_c_row[0] = '0;
    assign rd_c_tag[0] = '0;
    assign rd_c_rsp_ready[0] = 1'b1;
    `UNUSED_VAR ({rd_c_rsp_valid[0], rd_c_rsp_data[0], rd_c_rsp_tag[0]})
`endif

    // TMA rows: reads (stores to global) and writes (loads into shared
    // memory, local or from a peer's copy); their completions after the rows.
`ifdef VX_CFG_EXT_DXA_ENABLE
`ifdef DXA_MBAR_ENABLE
    wire dxa_compl_full;
    dxa_lmem_attr_t dxa_in_attr, peer_attr;
    assign dxa_in_attr = dxa_lmem_attr_t'(DXA_LMEM_ATTR_W'(dxa_lmem_bus_if.req_data.attr));
    assign peer_attr   = dxa_lmem_attr_t'(DXA_LMEM_ATTR_W'(dlmem_dma_if.req_data.attr));
    wire dxa_last_hold = dxa_in_attr.last && dxa_compl_full;
    `UNUSED_VAR (dxa_in_attr)
    // A peer's copy rows go first: the ingress takes its next flit only once this one is written.
    wire peer_wr = dlmem_dma_if.req_valid;
    `RUNTIME_ASSERT(~dlmem_dma_if.req_valid || dlmem_dma_if.req_data.rw, ("%t: *** %s read on the peer copy path", $time, INSTANCE_ID))
    wire peer_last_hold = peer_attr.last && dxa_compl_full;
    `UNUSED_VAR (peer_attr)
`else
    wire dxa_last_hold = 1'b0;
    wire peer_wr = 1'b0;
    wire peer_last_hold = 1'b0;
`endif
    wire dxa_v  = dxa_lmem_bus_if.req_valid && ~dxa_last_hold;
    wire dxa_rw = dxa_lmem_bus_if.req_data.rw;
    wire [PB-1:0] dxa_part = PB'(dxa_lmem_bus_if.req_data.addr);

    assign rd_c_req[1] = dxa_v && ~dxa_rw;
    assign rd_c_row[1] = SM_SRAM_ROW_W'(dxa_lmem_bus_if.req_data.addr >> LMEM_PART_BITS);
    assign rd_c_tag[1] = ROW_TAG_W'({dxa_lmem_bus_if.req_data.tag, dxa_part});

    wire [LMEM_DMA_ADDR_WIDTH-1:0]   wd_addr;
    wire [LMEM_DMA_DATA_SIZE-1:0]    wd_byteen;
    wire [LMEM_DMA_DATA_SIZE*8-1:0]  wd_data;
    wire [LMEM_DMA_ATTR_W-1:0]       wd_attr;
`ifdef DXA_MBAR_ENABLE
    assign wd_addr   = peer_wr ? dlmem_dma_if.req_data.addr   : dxa_lmem_bus_if.req_data.addr;
    assign wd_byteen = peer_wr ? dlmem_dma_if.req_data.byteen : dxa_lmem_bus_if.req_data.byteen;
    assign wd_data   = peer_wr ? dlmem_dma_if.req_data.data   : dxa_lmem_bus_if.req_data.data;
    assign wd_attr   = peer_wr ? LMEM_DMA_ATTR_W'(dlmem_dma_if.req_data.attr) : LMEM_DMA_ATTR_W'(dxa_lmem_bus_if.req_data.attr);
    assign dlmem_dma_if.req_ready = wr_ready[1] && ~peer_last_hold;
    assign dlmem_dma_if.rsp_valid = 1'b0;
    assign dlmem_dma_if.rsp_data  = '0;
    `UNUSED_VAR ({dlmem_dma_if.req_data.tag, dlmem_dma_if.rsp_ready})
`else
    assign wd_addr   = dxa_lmem_bus_if.req_data.addr;
    assign wd_byteen = dxa_lmem_bus_if.req_data.byteen;
    assign wd_data   = dxa_lmem_bus_if.req_data.data;
    assign wd_attr   = LMEM_DMA_ATTR_W'(dxa_lmem_bus_if.req_data.attr);
`endif
    wire [PB-1:0] wd_part = PB'(wd_addr);
    `UNUSED_VAR (wd_part)
    assign wr_valid[1]       = (peer_wr && ~peer_last_hold) || (~peer_wr && dxa_v && dxa_rw);
    assign wr_data[1].row    = SM_SRAM_ROW_W'(wd_addr >> LMEM_PART_BITS);
    assign wr_data[1].byteen = ROW_BYTES'(wd_byteen) << (((LMEM_ROW_PARTS > 1) ? 32'(wd_part) : 32'd0) * LMEM_DMA_DATA_SIZE);
    assign wr_data[1].data   = {LMEM_ROW_PARTS{wd_data}};
    assign wr_data[1].src    = SM_SRAM_SRC_TMA;
    wire wd_fire = wr_valid[1] && wr_ready[1];

    assign dxa_lmem_bus_if.req_ready = dxa_v && (dxa_rw ? (wr_ready[1] && ~peer_wr) : rd_c_fire[1]);

    wire [PB-1:0] dxa_rsp_part = PB'(rd_c_rsp_tag[1]);
    `UNUSED_VAR (dxa_rsp_part)
    assign dxa_lmem_bus_if.rsp_valid     = rd_c_rsp_valid[1];
    assign dxa_lmem_bus_if.rsp_data.data = rd_c_rsp_data[1][((LMEM_ROW_PARTS > 1) ? 32'(dxa_rsp_part) : 32'd0) * LMEM_DMA_DATA_SIZE * 8 +: LMEM_DMA_DATA_SIZE * 8];
    assign dxa_lmem_bus_if.rsp_data.tag  = DXA_LMEM_OUT_TAG_W'(rd_c_rsp_tag[1] >> PB);
    assign rd_c_rsp_ready[1]             = dxa_lmem_bus_if.rsp_ready;

    wire [`VX_CFG_LMEM_NUM_BANKS-1:0] dxa_bank_wr_fire;
    for (genvar i = 0; i < `VX_CFG_LMEM_NUM_BANKS; ++i) begin : g_dxa_bank_wr_fire
        assign dxa_bank_wr_fire[i] = wd_fire && (|wr_data[1].byteen[i * `VX_CFG_LMEM_BANK_BYTES +: `VX_CFG_LMEM_BANK_BYTES]);
    end

`ifdef DXA_MBAR_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) dxa_compl_if();
`endif

    VX_dxa_completion #(
        .INSTANCE_ID (`SFORMATF(("%s-lmem-dma-compl_det", INSTANCE_ID))),
        .NUM_BANKS   (`VX_CFG_LMEM_NUM_BANKS),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W)
    ) dxa_completion_detect (
        .clk           (clk),
        .reset         (reset),
        .bank_wr_fire  (dxa_bank_wr_fire),
        .bank_wr_attr  (DXA_LMEM_ATTR_W'(wd_attr)),
    `ifdef DXA_MBAR_ENABLE
        .full          (dxa_compl_full),
        .slot_valid    (cta_slot_valid),
        .slot_gen      (cta_slot_gen),
        .slot_stride   (cta_slot_stride),
        .compl_bus_if  (dxa_compl_if),
    `endif
        .txbar_bus_if  (dxa_txbar_bus_if)
    );

`ifdef DXA_MBAR_ENABLE
    // complete_tx on the mbarrier word, through the bank pair's AMO path.
    wire [ROW_WORD_BITS + SM_SRAM_ROW_W - 1:0] dc_word = (ROW_WORD_BITS + SM_SRAM_ROW_W)'(dxa_compl_if.req_data.addr);
    assign amo_valid[0]  = dxa_compl_if.req_valid;
    assign amo_row[0]    = dc_word[ROW_WORD_BITS +: SM_SRAM_ROW_W];
    assign amo_bank[0]   = BANK_BITS'(dc_word << HPW_BITS);
    assign amo_byteen[0] = dxa_compl_if.req_data.byteen;
    assign amo_wdata[0]  = dxa_compl_if.req_data.data;
    mem_bus_attr_t dc_attr;
    assign dc_attr = mem_bus_attr_t'(dxa_compl_if.req_data.attr);
    assign amo_attr[0]   = dc_attr.amo;
    `UNUSED_VAR (dc_attr)
    assign amo_tag[0]    = ROW_TAG_W'(dxa_compl_if.req_data.tag);
    assign dxa_compl_if.req_ready     = amo_ready[0];
    assign dxa_compl_if.rsp_valid     = amo_rsp_valid[0];
    assign dxa_compl_if.rsp_data.data = amo_rsp_data[0];
    assign dxa_compl_if.rsp_data.tag  = LSU_TAG_WIDTH'(amo_rsp_tag[0]);
    `RUNTIME_ASSERT(~amo_rsp_valid[0] || dxa_compl_if.rsp_ready, ("%t: *** %s DXA completion response dropped", $time, INSTANCE_ID))
    `UNUSED_VAR (dxa_compl_if.req_data.rw)
`else
    assign amo_valid[0] = 1'b0;
    assign {amo_row[0], amo_bank[0], amo_byteen[0], amo_wdata[0], amo_attr[0], amo_tag[0]} = '0;
    `UNUSED_VAR ({amo_ready[0], amo_rsp_valid[0], amo_rsp_data[0], amo_rsp_tag[0]})
`endif
`else
    assign rd_c_req[1] = 1'b0;
    assign rd_c_row[1] = '0;
    assign rd_c_tag[1] = '0;
    assign rd_c_rsp_ready[1] = 1'b1;
    `UNUSED_VAR ({rd_c_rsp_valid[1], rd_c_rsp_data[1], rd_c_rsp_tag[1]})
    assign wr_valid[1] = 1'b0;
    assign wr_data[1]  = '0;
    assign amo_valid[0] = 1'b0;
    assign {amo_row[0], amo_bank[0], amo_byteen[0], amo_wdata[0], amo_attr[0], amo_tag[0]} = '0;
    `UNUSED_VAR ({amo_ready[0], amo_rsp_valid[0], amo_rsp_data[0], amo_rsp_tag[0]})
`endif

    // DSMEM: one SM-to-SM slot per ingress request. Reads and atomics wait
    // until the slots written before them are in the array; replies leave in
    // request order (an earlier reply never arrives later than a later one).
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    localparam SLOT_BYTES = LMEM_INGRESS_DATA_SIZE;
    localparam ING_WORDS  = SLOT_BYTES / LSU_WORD_SIZE;
    localparam ING_OFF_BITS = `CLOG2(ING_WORDS);
    localparam SLOT_BITS  = `CLOG2(ROW_BYTES / SLOT_BYTES);
    localparam DSQ        = 4;
    `STATIC_ASSERT(SLOT_BYTES % LSU_WORD_SIZE == 0, ("the ingress slot must hold whole LSU words"))
    `STATIC_ASSERT(ROW_BYTES % SLOT_BYTES == 0, ("an array row must hold whole ingress slots"))
    `STATIC_ASSERT($bits(dlmem_lmem_if.req_data.tag) == LSU_TAG_WIDTH, ("invalid parameter"))

    wire [ING_WORDS-1:0] ing_word_en;
    for (genvar k = 0; k < ING_WORDS; ++k) begin : g_ing_word_en
        assign ing_word_en[k] = |dlmem_lmem_if.req_data.byteen[k * LSU_WORD_SIZE +: LSU_WORD_SIZE];
    end
    wire [`UP(ING_OFF_BITS)-1:0] ing_off;
    VX_priority_encoder #(.N (ING_WORDS)) ing_word_enc (
        .data_in (ing_word_en), .index_out (ing_off), `UNUSED_PIN (onehot_out), `UNUSED_PIN (valid_out)
    );
    mem_bus_attr_t ing_attr;
    assign ing_attr = mem_bus_attr_t'(dlmem_lmem_if.req_data.attr);
    wire ing_amo = ing_attr.amo.amo_valid;
    wire ing_rw  = dlmem_lmem_if.req_data.rw;
    wire [SLOT_BITS + SM_SRAM_ROW_W - 1:0] ing_slot = (SLOT_BITS + SM_SRAM_ROW_W)'(dlmem_lmem_if.req_data.addr);
    wire [`UP(SLOT_BITS)-1:0] ing_sidx = `UP(SLOT_BITS)'(ing_slot);
    wire [ROW_WORD_BITS + SM_SRAM_ROW_W - 1:0] ing_word;
    if (ING_OFF_BITS != 0) begin : g_ing_word
        assign ing_word = (ROW_WORD_BITS + SM_SRAM_ROW_W)'({ing_slot, ing_off[`UP(ING_OFF_BITS)-1:0]});
    end else begin : g_ing_word1
        assign ing_word = (ROW_WORD_BITS + SM_SRAM_ROW_W)'(ing_slot);
    end
    `RUNTIME_ASSERT(~dlmem_lmem_if.req_valid || ~ing_amo || ($countones(ing_word_en) <= 1), ("%t: *** %s DSMEM atomic over more than one word", $time, INSTANCE_ID))

    // Replies in flight: kind 0 none, 1 atomic (+1 cycle), 2 row read (+2).
    reg [`CLOG2(DSQ+1)-1:0] ds_infl_r;
    reg [1:0]               ds_rd_infl_r;  // row reads granted, reply not yet queued
    wire [`CLOG2(DSQ+1)-1:0] ds_q_size;
    wire ds_room = (ds_infl_r + ds_q_size) < `CLOG2(DSQ+1)'(DSQ);
    wire ing_v   = dlmem_lmem_if.req_valid && ds_room;
    reg  amo_rsp_pend;  // an atomic granted last cycle replies this cycle
    // An earlier read or atomic reply not yet queued keeps a write ack or an atomic behind it.
    wire ds_rd_busy = (ds_rd_infl_r != '0) || rd_c_rsp_valid[2];

    assign rd_c_req[2] = ing_v && ~ing_rw && ~ing_amo && ~wr_dsmem_busy;
    assign rd_c_row[2] = ing_slot[SLOT_BITS +: SM_SRAM_ROW_W];
    assign rd_c_tag[2] = ROW_TAG_W'({dlmem_lmem_if.req_data.tag, ROW_WORD_BITS'(SLOT_BITS != 0 ? 32'(ing_sidx) : 32'd0)});

    assign wr_valid[2]       = ing_v && ing_rw && ~ing_amo && ~ds_rd_busy && ~amo_rsp_pend;
    assign wr_data[2].row    = ing_slot[SLOT_BITS +: SM_SRAM_ROW_W];
    assign wr_data[2].byteen = ROW_BYTES'(dlmem_lmem_if.req_data.byteen) << (((SLOT_BITS != 0) ? 32'(ing_sidx) : 32'd0) * SLOT_BYTES);
    assign wr_data[2].data   = {(ROW_BYTES / SLOT_BYTES){dlmem_lmem_if.req_data.data}};
    assign wr_data[2].src    = SM_SRAM_SRC_DSMEM;

    assign amo_valid[1]  = ing_v && ing_amo && ~wr_dsmem_busy && ~ds_rd_busy;
    assign amo_row[1]    = ing_word[ROW_WORD_BITS +: SM_SRAM_ROW_W];
    assign amo_bank[1]   = BANK_BITS'(ing_word << HPW_BITS);
    assign amo_byteen[1] = dlmem_lmem_if.req_data.byteen[((ING_OFF_BITS != 0) ? 32'(ing_off) : 32'd0) * LSU_WORD_SIZE +: LSU_WORD_SIZE];
    assign amo_wdata[1]  = dlmem_lmem_if.req_data.data[((ING_OFF_BITS != 0) ? 32'(ing_off) : 32'd0) * LSU_WORD_SIZE * 8 +: LSU_WORD_SIZE * 8];
    assign amo_attr[1]   = ing_attr.amo;
    assign amo_tag[1]    = ROW_TAG_W'(dlmem_lmem_if.req_data.tag);

    wire ds_wr_fire  = wr_valid[2] && wr_ready[2];
    wire ds_amo_fire = amo_valid[1] && amo_ready[1];
    assign dlmem_lmem_if.req_ready = ds_wr_fire || ds_amo_fire || rd_c_fire[2];
    assign wr_dsmem_last = 1'b0;

    always @(posedge clk) begin
        if (reset) begin
            ds_infl_r    <= '0;
            ds_rd_infl_r <= '0;
            amo_rsp_pend <= 1'b0;
        end else begin
            ds_infl_r    <= ds_infl_r + `CLOG2(DSQ+1)'(ds_amo_fire) + `CLOG2(DSQ+1)'(rd_c_fire[2])
                                      - `CLOG2(DSQ+1)'(amo_rsp_valid[1]) - `CLOG2(DSQ+1)'(rd_c_rsp_valid[2] && rd_c_rsp_ready[2]);
            ds_rd_infl_r <= ds_rd_infl_r + 2'(rd_c_fire[2]) - 2'(rd_rsp_valid[2]);
            amo_rsp_pend <= ds_amo_fire;
        end
    end

    // Replies: write acks at accept, atomics a cycle after the grant, row reads from their queue.
    wire ds_rd_out = rd_c_rsp_valid[2];
    wire [`UP(SLOT_BITS)-1:0] ds_rd_sidx = `UP(SLOT_BITS)'(rd_c_rsp_tag[2]);
    wire [SLOT_BYTES*8-1:0]     ds_rsp_data = amo_rsp_valid[1] ? {ING_WORDS{amo_rsp_data[1]}}
                                            : ds_rd_out ? rd_c_rsp_data[2][((SLOT_BITS != 0) ? 32'(ds_rd_sidx) : 32'd0) * SLOT_BYTES * 8 +: SLOT_BYTES * 8]
                                            : '0;
    wire [LSU_TAG_WIDTH-1:0]    ds_rsp_tag  = amo_rsp_valid[1] ? LSU_TAG_WIDTH'(amo_rsp_tag[1])
                                            : ds_rd_out ? LSU_TAG_WIDTH'(rd_c_rsp_tag[2] >> ROW_WORD_BITS)
                                            : dlmem_lmem_if.req_data.tag;
    assign rd_c_rsp_ready[2] = ~ds_wr_fire && ~amo_rsp_valid[1];
    wire ds_q_empty;
    VX_fifo_queue #(
        .DATAW (LSU_TAG_WIDTH + SLOT_BYTES * 8),
        .DEPTH (DSQ)
    ) ds_rsp_q (
        .clk      (clk),
        .reset    (reset),
        .push     (ds_wr_fire || amo_rsp_valid[1] || (ds_rd_out && rd_c_rsp_ready[2])),
        .pop      (dlmem_lmem_if.rsp_valid && dlmem_lmem_if.rsp_ready),
        .data_in  ({ds_rsp_tag, ds_rsp_data}),
        .data_out ({dlmem_lmem_if.rsp_data.tag, dlmem_lmem_if.rsp_data.data}),
        .empty    (ds_q_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        .size     (ds_q_size)
    );
    assign dlmem_lmem_if.rsp_valid = ~ds_q_empty;
    `RUNTIME_ASSERT(~(ds_wr_fire && amo_rsp_valid[1]), ("%t: *** %s DSMEM ack beside an atomic reply", $time, INSTANCE_ID))
    `UNUSED_VAR (ing_attr)
`else
    assign rd_c_req[2] = 1'b0;
    assign rd_c_row[2] = '0;
    assign rd_c_tag[2] = '0;
    assign rd_c_rsp_ready[2] = 1'b1;
    `UNUSED_VAR ({rd_c_rsp_valid[2], rd_c_rsp_data[2], rd_c_rsp_tag[2]})
    assign wr_valid[2] = 1'b0;
    assign wr_data[2]  = '0;
    assign wr_dsmem_last = 1'b0;
    assign amo_valid[1] = 1'b0;
    assign {amo_row[1], amo_bank[1], amo_byteen[1], amo_wdata[1], amo_attr[1], amo_tag[1]} = '0;
    `UNUSED_VAR ({amo_ready[1], amo_rsp_valid[1], amo_rsp_data[1], amo_rsp_tag[1], wr_dsmem_busy})
`endif

    // cp.async lanes, through the array's fill class.
`ifndef VX_CFG_EXT_MBAR_ENABLE
    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) cpa_lmem_if[`VX_CFG_NUM_LSU_BLOCKS]();
    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_cpa_sram
        assign cpa_lmem_if[i].req_valid = 1'b0;
        assign cpa_lmem_if[i].req_data  = '0;
        assign cpa_lmem_if[i].rsp_ready = 1'b1;
        `UNUSED_VAR ({cpa_lmem_if[i].req_ready, cpa_lmem_if[i].rsp_valid, cpa_lmem_if[i].rsp_data})
    end
`endif

`ifdef VX_CFG_EXT_MBAR_ENABLE
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0]                     mbar_evt_valid;
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0]                     mbar_evt_flip;
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0][NW_WIDTH-1:0]       mbar_evt_wid;
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0][LMEM_ROW_WIDTH-1:0] mbar_evt_row;
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0][SM_SRAM_ROW_W-1:0]  mbar_evt_arow;
    for (genvar i = 0; i < `VX_CFG_LMEM_NUM_BANKS; ++i) begin : g_mbar_evt_row
        // mbarrier words are shared-memory words: their rows are below the window's top.
        assign mbar_evt_row[i] = LMEM_ROW_WIDTH'(mbar_evt_arow[i]);
    end
`endif

    wire [2:0]                                             acc_commit_valid;
    wire [2:0][NW_WIDTH-1:0]                               acc_commit_wid;
    wire [2:0][`CLOG2(`VX_CFG_NUM_LSU_LANES+1)-1:0]        acc_commit_cnt;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    assign lmem_commit_valid = acc_commit_valid;
    assign lmem_commit_wid   = acc_commit_wid;
    assign lmem_commit_cnt   = acc_commit_cnt;
`else
    `UNUSED_VAR ({acc_commit_valid, acc_commit_wid, acc_commit_cnt})
`endif

    wire                    gl_rsp_valid, wf_retired_valid, fill_done_valid;
    sm_sram_rsp_t           gl_rsp_data;
    wire [L1_SEQ_W-1:0]     wf_retired_seq;
    wire [SM_SRAM_ROW_W-1:0] fill_done_row;
    `UNUSED_VAR ({fill_done_valid, fill_done_row})

    VX_sm_sram #(
        .INSTANCE_ID      (`SFORMATF(("%s-sm-sram", INSTANCE_ID))),
        .NUM_BLOCKS       (`VX_CFG_NUM_LSU_BLOCKS),
        .ROW_TAG_W        (ROW_TAG_W)
    ) sm_sram (
        .clk              (clk),
        .reset            (reset),
        .cfg              (smem_cfg),
        .lsu_if           (lsu_sram_if),
        .gl_req_valid     (gl_req_valid),
        .gl_req_data      (gl_req_data),
        .gl_req_ready     (gl_req_ready),
        .gl_rsp_valid     (gl_rsp_valid),
        .gl_rsp_data      (gl_rsp_data),
        .gl_rsp_ready     (gl_rsp_ready),
        .wf_retired_valid (wf_retired_valid),
        .wf_retired_seq   (wf_retired_seq),
        .rd_valid         (rd_valid),
        .rd_row           (rd_row),
        .rd_tag           (rd_tag),
        .rd_ready         (rd_ready),
        .rd_rsp_valid     (rd_rsp_valid),
        .rd_rsp_data      (rd_rsp_data),
        .rd_rsp_tag       (rd_rsp_tag),
        .wr_valid         (wr_valid),
        .wr_data          (wr_data),
        .wr_dsmem_last    (wr_dsmem_last),
        .wr_ready         (wr_ready),
        .wr_dsmem_busy    (wr_dsmem_busy),
        .fill_done_valid  (fill_done_valid),
        .fill_done_row    (fill_done_row),
        .cpa_if           (cpa_lmem_if),
        .amo_valid        (amo_valid),
        .amo_row          (amo_row),
        .amo_bank         (amo_bank),
        .amo_byteen       (amo_byteen),
        .amo_wdata        (amo_wdata),
        .amo_attr         (amo_attr),
        .amo_tag          (amo_tag),
        .amo_ready        (amo_ready),
        .amo_rsp_valid    (amo_rsp_valid),
        .amo_rsp_data     (amo_rsp_data),
        .amo_rsp_tag      (amo_rsp_tag),
        .acc_commit_valid (acc_commit_valid),
        .acc_commit_wid   (acc_commit_wid),
        .acc_commit_cnt   (acc_commit_cnt)
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        ,
        .mbar_evt_valid   (mbar_evt_valid),
        .mbar_evt_flip    (mbar_evt_flip),
        .mbar_evt_wid     (mbar_evt_wid),
        .mbar_evt_row     (mbar_evt_arow)
    `endif
    );

    wire l1_busy;

    VX_sm_l1 #(
        .INSTANCE_ID      (`SFORMATF(("%s-sm-l1", INSTANCE_ID))),
        .NUM_BLOCKS       (`VX_CFG_NUM_LSU_BLOCKS),
        .MEM_TAG_WIDTH    (SM_L1_MEM_TAG_WIDTH)
    ) sm_l1 (
        .clk              (clk),
        .reset            (reset),
        .cfg              (smem_cfg),
        .lsu_if           (lsu_dcache_if),
        .gl_req_valid     (gl_req_valid),
        .gl_req_data      (gl_req_data),
        .gl_req_ready     (gl_req_ready),
        .gl_rsp_valid     (gl_rsp_valid),
        .gl_rsp_data      (gl_rsp_data),
        .gl_rsp_ready     (gl_rsp_ready),
        .wf_retired_valid (wf_retired_valid),
        .wf_retired_seq   (wf_retired_seq),
        .fill_valid       (wr_valid[0]),
        .fill_data        (wr_data[0]),
        .fill_ready       (wr_ready[0]),
        .mem_bus_if       (dcache_bus_if[0]),
        .dcr_flush_if     (dcr_flush_if),
    `ifdef PERF_ENABLE
        .perf             (dcache_perf),
    `endif
        .busy             (l1_busy)
    );

    assign empty = ~l1_busy;

`ifdef PERF_ENABLE
    assign lmem_perf = '0;
    assign coalescer_perf = '0;
`endif

`else
    // Per-block local memory adapters to avoid deadlock when NUM_LSU_BLOCKS > 1.
    // Each block gets dedicated ports into local memory, eliminating the circular
    // dependency that arises from sharing a single adapter's unpack/pack buffers.

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // The DLMEM ingress is one more input after the LSU lanes, in their round-robin.
    localparam LMEM_INGRESS_IDX = LSU_NUM_REQS;

`ifdef DXA_MBAR_ENABLE
    // The DXA completion's complete_tx requests are one more input, after the ingress.
    localparam LMEM_DXA_COMPL_IDX = LSU_NUM_REQS + 1;
    // Then the cp.async units' lanes, block-major.
    localparam LMEM_CPA_IDX  = LSU_NUM_REQS + 2;
    localparam LMEM_NUM_REQS = LMEM_CPA_IDX + LSU_NUM_REQS;

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_adapt_if[LMEM_NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) dxa_compl_if();

    `ASSIGN_VX_MEM_BUS_IF (lmem_adapt_if[LMEM_DXA_COMPL_IDX], dxa_compl_if);
`elsif VX_CFG_EXT_MBAR_ENABLE
    // The cp.async units' lanes, block-major, after the ingress.
    localparam LMEM_CPA_IDX  = LSU_NUM_REQS + 1;
    localparam LMEM_NUM_REQS = LMEM_CPA_IDX + LSU_NUM_REQS;

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_adapt_if[LMEM_NUM_REQS]();
`else
    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_adapt_if[LSU_NUM_REQS + 1]();
`endif

    `STATIC_ASSERT($bits(dlmem_lmem_if.req_data.addr) == LMEM_ADDR_WIDTH, ("invalid parameter"))
    `STATIC_ASSERT($bits(dlmem_lmem_if.req_data.tag) == LSU_TAG_WIDTH, ("invalid parameter"))

    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_valid       = dlmem_lmem_if.req_valid;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.rw     = dlmem_lmem_if.req_data.rw;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.addr   = LSU_ADDR_WIDTH'(dlmem_lmem_if.req_data.addr);
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.data   = dlmem_lmem_if.req_data.data;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.byteen = dlmem_lmem_if.req_data.byteen;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.attr   = dlmem_lmem_if.req_data.attr;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.tag    = dlmem_lmem_if.req_data.tag;
    assign dlmem_lmem_if.req_ready = lmem_adapt_if[LMEM_INGRESS_IDX].req_ready;

    assign dlmem_lmem_if.rsp_valid = lmem_adapt_if[LMEM_INGRESS_IDX].rsp_valid;
    assign dlmem_lmem_if.rsp_data  = lmem_adapt_if[LMEM_INGRESS_IDX].rsp_data;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].rsp_ready = dlmem_lmem_if.rsp_ready;
`else
    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_adapt_if[LSU_NUM_REQS]();
`endif

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_lmem_adapters

        VX_mem_bus_if #(
            .DATA_SIZE (LSU_WORD_SIZE),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) lmem_block_if[`VX_CFG_NUM_LSU_LANES]();

        VX_lsu_adapter #(
            .NUM_LANES    (`VX_CFG_NUM_LSU_LANES),
            .DATA_SIZE    (LSU_WORD_SIZE),
            .TAG_WIDTH    (LSU_TAG_WIDTH),
            .TAG_SEL_BITS (LSU_TAG_WIDTH - UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (3),
            .RSP_OUT_BUF  (0)
        ) lmem_adapter (
            .clk        (clk),
            .reset      (reset),
            .lsu_mem_if (lsu_lmem_if[i]),
            .mem_bus_if (lmem_block_if)
        );

        for (genvar j = 0; j < `VX_CFG_NUM_LSU_LANES; ++j) begin : g_lmem_adapt_if
            `ASSIGN_VX_MEM_BUS_IF (lmem_adapt_if[i * `VX_CFG_NUM_LSU_LANES + j], lmem_block_if[j]);
        end
    end

`ifdef VX_CFG_EXT_MBAR_ENABLE
    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_cpa_adapters

        VX_mem_bus_if #(
            .DATA_SIZE (LSU_WORD_SIZE),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) cpa_block_if[`VX_CFG_NUM_LSU_LANES]();

        VX_lsu_adapter #(
            .NUM_LANES    (`VX_CFG_NUM_LSU_LANES),
            .DATA_SIZE    (LSU_WORD_SIZE),
            .TAG_WIDTH    (LSU_TAG_WIDTH),
            .TAG_SEL_BITS (LSU_TAG_WIDTH - UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (3),
            .RSP_OUT_BUF  (0)
        ) cpa_adapter (
            .clk        (clk),
            .reset      (reset),
            .lsu_mem_if (cpa_lmem_if[i]),
            .mem_bus_if (cpa_block_if)
        );

        for (genvar j = 0; j < `VX_CFG_NUM_LSU_LANES; ++j) begin : g_cpa_adapt_if
            `ASSIGN_VX_MEM_BUS_IF (lmem_adapt_if[LMEM_CPA_IDX + i * `VX_CFG_NUM_LSU_LANES + j], cpa_block_if[j]);
        end
    end
`endif

    // DMA arbiter: merge DXA writes and/or TCU reads onto single DMA port.

    VX_mem_bus_if #(
        .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
        .TAG_WIDTH   (LMEM_DMA_TAG_WIDTH),
        .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
    ) lmem_dma_if();

    localparam LMEM_DMA_IN_TAG_W = LMEM_DMA_IN_TAG_MAX;

    if (LMEM_DMA_INPUTS > 0) begin : g_lmem_dma

        VX_mem_bus_if #(
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (LMEM_DMA_IN_TAG_W),
            .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
        ) dma_arb_in_if[LMEM_DMA_INPUTS]();

        VX_mem_bus_if #(
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (LMEM_DMA_TAG_WIDTH),
            .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
        ) dma_arb_out_if[1]();

        // Wire DXA and/or TCU into the arbiter input array.
    `ifdef VX_CFG_EXT_DXA_ENABLE
        // DXA issues LMEM-relative word addresses directly (issuer_lmem_base + intra,
        // with multicast replay adding r×smem_stride per beat), so no per-slot
        // address translation is needed here.
    `ifdef DXA_MBAR_ENABLE
        // A last beat waits here while the completion queue is full, so no
        // completion is ever dropped; it reaches neither the arbiter nor a bank.
        wire dxa_compl_full;
        dxa_lmem_attr_t dxa_in_attr;
        assign dxa_in_attr = dxa_lmem_attr_t'(dxa_lmem_bus_if.req_data.attr);
        wire dxa_last_hold = dxa_in_attr.last && dxa_compl_full;
        `UNUSED_VAR (dxa_in_attr)

        VX_mem_bus_if #(
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
            .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
        ) dxa_held_if();

        assign dxa_held_if.req_valid     = dxa_lmem_bus_if.req_valid && ~dxa_last_hold;
        assign dxa_held_if.req_data      = dxa_lmem_bus_if.req_data;
        assign dxa_lmem_bus_if.req_ready = dxa_held_if.req_ready && ~dxa_last_hold;
        assign dxa_lmem_bus_if.rsp_valid = dxa_held_if.rsp_valid;
        assign dxa_lmem_bus_if.rsp_data  = dxa_held_if.rsp_data;
        assign dxa_held_if.rsp_ready     = dxa_lmem_bus_if.rsp_ready;

        `ASSIGN_VX_MEM_BUS_IF_EX (dma_arb_in_if[LMEM_DMA_DXA_IDX], dxa_held_if, LMEM_DMA_IN_TAG_W, DXA_LMEM_OUT_TAG_W, UUID_WIDTH);
    `else
        `ASSIGN_VX_MEM_BUS_IF_EX (dma_arb_in_if[LMEM_DMA_DXA_IDX], dxa_lmem_bus_if, LMEM_DMA_IN_TAG_W, DXA_LMEM_OUT_TAG_W, UUID_WIDTH);
    `endif
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        `ASSIGN_VX_MEM_BUS_IF_EX (dma_arb_in_if[LMEM_DMA_TCU_IDX], tcu_lmem_if, LMEM_DMA_IN_TAG_W, TCU_LMEM_TAG_W, UUID_WIDTH);
    `endif
    `ifdef DXA_MBAR_ENABLE
        // No buffer: the ingress takes the next flit only once the bank has
        // committed this one, which keeps a copy's completion behind its data.
        `ASSIGN_VX_MEM_BUS_IF (dma_arb_in_if[LMEM_DMA_ROW_IDX], dlmem_dma_if);
    `endif

        VX_mem_bus_arb #(
            .NUM_INPUTS  (LMEM_DMA_INPUTS),
            .NUM_OUTPUTS (1),
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (LMEM_DMA_IN_TAG_W),
            .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH),
            .ARBITER     ("P")
        ) lmem_dma_arb (
            .clk         (clk),
            .reset       (reset),
            .bus_in_if   (dma_arb_in_if),
            .bus_out_if  (dma_arb_out_if)
        );

        `ASSIGN_VX_MEM_BUS_IF (lmem_dma_if, dma_arb_out_if[0]);

    `ifdef VX_CFG_EXT_DXA_ENABLE
        wire [`VX_CFG_LMEM_NUM_BANKS-1:0] dxa_bank_wr_fire;
        for (genvar i = 0; i < `VX_CFG_LMEM_NUM_BANKS; ++i) begin : g_dxa_bank_wr_fire
            assign dxa_bank_wr_fire[i] = lmem_dma_if.req_valid
                                      && lmem_dma_if.req_ready
                                      && lmem_dma_if.req_data.rw
                                      && (|lmem_dma_if.req_data.byteen[i*LSU_WORD_SIZE +: LSU_WORD_SIZE]);
        end

        VX_dxa_completion #(
            .INSTANCE_ID (`SFORMATF(("%s-lmem-dma-compl_det", INSTANCE_ID))),
            .NUM_BANKS   (`VX_CFG_LMEM_NUM_BANKS),
            .ATTR_WIDTH  (DXA_LMEM_ATTR_W)
        ) dxa_completion_detect (
            .clk           (clk),
            .reset         (reset),
            .bank_wr_fire  (dxa_bank_wr_fire),
            .bank_wr_attr  (DXA_LMEM_ATTR_W'(lmem_dma_if.req_data.attr)),
        `ifdef DXA_MBAR_ENABLE
            .full          (dxa_compl_full),
            .slot_valid    (cta_slot_valid),
            .slot_gen      (cta_slot_gen),
            .slot_stride   (cta_slot_stride),
            .compl_bus_if  (dxa_compl_if),
        `endif
            .txbar_bus_if  (dxa_txbar_bus_if)
        );

        `ifdef DBG_TRACE_DXA
        always @(posedge clk) begin
            if (dxa_txbar_bus_if.valid && !reset) begin
                `TRACE(2, ("%t: %s-lmem-dma: dxa_txbar valid=1, bar_addr=0x%0h, ready=%b\n",
                    $time, INSTANCE_ID, dxa_txbar_bus_if.data.addr, dxa_txbar_bus_if.ready))
            end
        end
        `endif

    `endif

    end else begin : g_no_lmem_dma

        assign lmem_dma_if.req_valid = 1'b0;
        assign lmem_dma_if.req_data  = '0;
        `UNUSED_VAR (lmem_dma_if.req_ready)

        `UNUSED_VAR (lmem_dma_if.rsp_valid)
        `UNUSED_VAR (lmem_dma_if.rsp_data)
        assign lmem_dma_if.rsp_ready = 1'b0;

    end

`ifdef VX_CFG_EXT_MBAR_ENABLE
    localparam LMEM_ROW_WIDTH = LMEM_ADDR_WIDTH - `CLOG2(`VX_CFG_LMEM_NUM_BANKS);
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0]                     mbar_evt_valid;
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0]                     mbar_evt_flip;
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0][NW_WIDTH-1:0]       mbar_evt_wid;
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0][LMEM_ROW_WIDTH-1:0] mbar_evt_row;
`endif


    VX_local_mem #(
        .INSTANCE_ID (`SFORMATF(("%s-lmem", INSTANCE_ID))),
        .SIZE        (1 << `VX_CFG_LMEM_LOG_SIZE),
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        // The cp.async lanes' stores are acknowledged; they report no commits.
        .NUM_REQS    (LMEM_NUM_REQS),
        .STORE_ACK_MASK ((LMEM_NUM_REQS'(1) << LMEM_INGRESS_IDX)
                       | (((LMEM_NUM_REQS'(1) << LSU_NUM_REQS) - LMEM_NUM_REQS'(1)) << LMEM_CPA_IDX)),
        .COMMIT_MASK ((LMEM_NUM_REQS'(1) << LMEM_INGRESS_IDX) - LMEM_NUM_REQS'(1)),
        .LOW_PRI_REQS (LSU_NUM_REQS),
    `elsif VX_CFG_EXT_DLMEM_ENABLE
        .NUM_REQS    (LSU_NUM_REQS + 1),
        .STORE_ACK_MASK ((LSU_NUM_REQS + 1)'(1) << LMEM_INGRESS_IDX),
        .COMMIT_MASK (((LSU_NUM_REQS + 1)'(1) << LMEM_INGRESS_IDX) - (LSU_NUM_REQS + 1)'(1)),
    `else
        .NUM_REQS    (LSU_NUM_REQS),
    `endif
        .NUM_BANKS   (`VX_CFG_LMEM_NUM_BANKS),
        .WORD_SIZE   (LSU_WORD_SIZE),
        .ADDR_WIDTH  (LMEM_ADDR_WIDTH),
        .TAG_WIDTH   (LSU_TAG_WIDTH),
        .DMA_ENABLE  (LMEM_DMA_EN),
        .DMA_TAG_WIDTH (LMEM_DMA_TAG_WIDTH),
        .AMO_ENABLE  (`VX_CFG_EXT_A_ENABLED),
        .OUT_BUF     (3)
    ) local_mem (
        .clk         (clk),
        .reset       (reset),
    `ifdef PERF_ENABLE
        .lmem_perf   (lmem_perf),
    `endif
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        .lmem_commit_valid (lmem_commit_valid),
        .lmem_commit_wid   (lmem_commit_wid),
    `endif
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        .mbar_evt_valid    (mbar_evt_valid),
        .mbar_evt_flip     (mbar_evt_flip),
        .mbar_evt_wid      (mbar_evt_wid),
        .mbar_evt_row      (mbar_evt_row),
    `endif
        .dma_bus_if  (lmem_dma_if),
        .lsu_bus_if  (lmem_adapt_if)
    );

`endif
`ifdef VX_CFG_EXT_MBAR_ENABLE
    VX_mbar_waiters #(
        .INSTANCE_ID (`SFORMATF(("%s-mbar-waiters", INSTANCE_ID))),
        .NUM_BANKS   (`VX_CFG_LMEM_NUM_BANKS),
        .NUM_RETIRES (`VX_CFG_NUM_LSU_BLOCKS),
        .ROW_WIDTH   (LMEM_ROW_WIDTH)
    ) mbar_waiters (
        .clk          (clk),
        .reset        (reset),
        .evt_valid    (mbar_evt_valid),
        .evt_flip     (mbar_evt_flip),
        .evt_wid      (mbar_evt_wid),
        .evt_row      (mbar_evt_row),
        .retire_valid (mbar_retire_valid),
        .retire_wid   (mbar_retire_wid),
        .unlock_valid (mbar_unlock_valid),
        .unlock_mask  (mbar_unlock_mask)
    );
`endif

`else

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_lsu_dcache_if
        `ASSIGN_VX_MEM_BUS_IF (lsu_dcache_if[i], lsu_mem_if[i]);
    end

`ifdef PERF_ENABLE
    assign lmem_perf = '0;
`endif

`endif

`ifndef VX_CFG_H100_SM_ENABLE
    VX_lsu_mem_if #(
        .NUM_LANES (DCACHE_CHANNELS),
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH)
    ) dcache_coalesced_if[`VX_CFG_NUM_LSU_BLOCKS]();

`ifdef PERF_ENABLE
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][PERF_CTR_BITS-1:0] per_block_coalescer_misses;
    wire [PERF_CTR_BITS-1:0] coalescer_misses;
    VX_reduce_tree #(
        .IN_W (PERF_CTR_BITS),
        .N    (`VX_CFG_NUM_LSU_BLOCKS),
        .OP   ("+")
    ) coalescer_reduce (
        .data_in  (per_block_coalescer_misses),
        .data_out (coalescer_misses)
    );
    `BUFFER(coalescer_perf.misses, coalescer_misses);
`endif

    if ((`VX_CFG_NUM_LSU_LANES > 1) && (DCACHE_WORD_SIZE > LSU_WORD_SIZE)) begin : g_coalescing

        for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_coalescers

            VX_mem_coalescer #(
                .INSTANCE_ID    (`SFORMATF(("%s-coalescer%0d", INSTANCE_ID, i))),
                .NUM_REQS       (`VX_CFG_NUM_LSU_LANES),
                .DATA_IN_SIZE   (LSU_WORD_SIZE),
                .DATA_OUT_SIZE  (DCACHE_WORD_SIZE),
                .ADDR_WIDTH     (LSU_ADDR_WIDTH),
                .USER_WIDTH     (MEM_ATTR_WIDTH),
                .TAG_WIDTH      (LSU_TAG_WIDTH),
                .UUID_WIDTH     (UUID_WIDTH),
                .QUEUE_SIZE     (LSU_QUEUE_OUT_SIZE),
                .PERF_CTR_BITS  (PERF_CTR_BITS)
            ) mem_coalescer (
                .clk            (clk),
                .reset          (reset),

            `ifdef PERF_ENABLE
                .misses         (per_block_coalescer_misses[i]),
            `else
                `UNUSED_PIN (misses),
            `endif

                .empty          (per_block_path_empty[i]),

                // Input request
                .in_req_valid   (lsu_dcache_if[i].req_valid),
                .in_req_mask    (lsu_dcache_if[i].req_data.mask),
                .in_req_rw      (lsu_dcache_if[i].req_data.rw),
                .in_req_byteen  (lsu_dcache_if[i].req_data.byteen),
                .in_req_addr    (lsu_dcache_if[i].req_data.addr),
                .in_req_user    (lsu_dcache_if[i].req_data.user),
                .in_req_no_merge(lsu_dcache_if[i].req_data.user[0][MEM_ATTR_AMO_OFFS]),
                .in_req_data    (lsu_dcache_if[i].req_data.data),
                .in_req_tag     (lsu_dcache_if[i].req_data.tag),
                .in_req_ready   (lsu_dcache_if[i].req_ready),

                // Input response
                .in_rsp_valid   (lsu_dcache_if[i].rsp_valid),
                .in_rsp_mask    (lsu_dcache_if[i].rsp_data.mask),
                .in_rsp_data    (lsu_dcache_if[i].rsp_data.data),
                .in_rsp_tag     (lsu_dcache_if[i].rsp_data.tag),
                .in_rsp_ready   (lsu_dcache_if[i].rsp_ready),

                // Output request
                .out_req_valid  (dcache_coalesced_if[i].req_valid),
                .out_req_mask   (dcache_coalesced_if[i].req_data.mask),
                .out_req_rw     (dcache_coalesced_if[i].req_data.rw),
                .out_req_byteen (dcache_coalesced_if[i].req_data.byteen),
                .out_req_addr   (dcache_coalesced_if[i].req_data.addr),
                .out_req_user   (dcache_coalesced_if[i].req_data.user),
                .out_req_data   (dcache_coalesced_if[i].req_data.data),
                .out_req_tag    (dcache_coalesced_if[i].req_data.tag),
                .out_req_ready  (dcache_coalesced_if[i].req_ready),

                // Output response
                .out_rsp_valid  (dcache_coalesced_if[i].rsp_valid),
                .out_rsp_mask   (dcache_coalesced_if[i].rsp_data.mask),
                .out_rsp_data   (dcache_coalesced_if[i].rsp_data.data),
                .out_rsp_tag    (dcache_coalesced_if[i].rsp_data.tag),
                .out_rsp_ready  (dcache_coalesced_if[i].rsp_ready)
            );
        end

    end else begin : g_no_coalescing

        // The passthrough narrows the tag from LSU_TAG_WIDTH to
        // DCACHE_CORE_TAG_WIDTH: the dcache tag-id space must cover the
        // mem-scheduler slot id or responses alias.
        `STATIC_ASSERT (DCACHE_CORE_TAG_WIDTH >= LSU_TAG_WIDTH, ("dcache tag-id space cannot hold the LSU outstanding slot id"))

        for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_dcache_coalesced_if
            `ASSIGN_VX_MEM_BUS_IF (dcache_coalesced_if[i], lsu_dcache_if[i]);
        `ifdef PERF_ENABLE
            assign per_block_coalescer_misses[i] = '0;
        `endif
            // No coalescer: a store still in flight upstream of the adapter is
            // one being presented to the adapter input. The adapter's own buffer
            // is accounted for separately by per_block_adapter_empty.
            assign per_block_path_empty[i] = ~dcache_coalesced_if[i].req_valid;
        end

    end

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_dcache_adapters

        VX_mem_bus_if #(
            .DATA_SIZE (DCACHE_WORD_SIZE),
            .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH)
        ) dcache_bus_tmp_if[DCACHE_CHANNELS]();

        VX_lsu_adapter #(
            .NUM_LANES    (DCACHE_CHANNELS),
            .DATA_SIZE    (DCACHE_WORD_SIZE),
            .TAG_WIDTH    (DCACHE_CORE_TAG_WIDTH),
            .TAG_SEL_BITS (DCACHE_CORE_TAG_WIDTH - UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (3), // register dcache-request master boundary on all channels (passthru ports 1+)
            .RSP_OUT_BUF  (0)
        ) dcache_adapter (
            .clk        (clk),
            .reset      (reset),
            .lsu_mem_if (dcache_coalesced_if[i]),
            .mem_bus_if (dcache_bus_tmp_if)
        );

        // A store buffered in the adapter's REQ_OUT_BUF has left dcache_coalesced_if
        // but not yet reached the flush-injection point; keep the block non-empty
        // until every channel's request has drained past the adapter output.
        wire [DCACHE_CHANNELS-1:0] adapter_req_valid;
        for (genvar j = 0; j < DCACHE_CHANNELS; ++j) begin : g_adapter_req_valid
            assign adapter_req_valid[j] = dcache_bus_tmp_if[j].req_valid;
        end
        assign per_block_adapter_empty[i] = ~(| adapter_req_valid);

        for (genvar j = 0; j < DCACHE_CHANNELS; ++j) begin : g_dcache_bus_if
            if (i == 0 && j == 0) begin : g_flush_port
                // Port 0: route through VX_dcr_flush to inject flush requests
                VX_dcr_flush #(
                    .WORD_SIZE (DCACHE_WORD_SIZE),
                    .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH),
                    .REQ_OUT_BUF (3) // register core dcache-request boundary; rsp already registered by L1 CORE_OUT_BUF
                ) dcr_flush (
                    .clk          (clk),
                    .reset        (reset),
                    .dcr_flush_if (dcr_flush_if),
                    .core_bus_if  (dcache_bus_tmp_if[0]),
                    .cache_bus_if (dcache_bus_if[0])
                );
            end else begin : g_passthru_port
                // Ports 1+: pass through; tag is zero-extended from
                // DCACHE_CORE_TAG_WIDTH to DCACHE_TAG_WIDTH on request,
                // and MSB-stripped on response.
                `ASSIGN_VX_MEM_BUS_IF_EX (dcache_bus_if[i * DCACHE_CHANNELS + j], dcache_bus_tmp_if[j], DCACHE_TAG_WIDTH_BASE, DCACHE_CORE_TAG_WIDTH, 0);
            end
        end

    end

    assign empty = (& per_block_empty);
`endif

`ifdef DBG_TRACE_MEM
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // A CTA fence copy reaching the dcache: the drain it starts covers the requests already accepted.
`ifdef VX_CFG_H100_SM_ENABLE
    for (genvar i = 0; i < 1; ++i) begin : g_trace_cta
`else
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin : g_trace_cta
`endif
        wire is_cta = dcache_bus_if[i].req_valid && dcache_bus_if[i].req_data.attr[MEM_ATTR_FENCE_CTA_OFFS];
        reg waiting_r;
        always @(posedge clk) begin
            waiting_r <= ~reset && is_cta && ~dcache_bus_if[i].req_ready;
            if (~reset && is_cta && ~waiting_r) begin
                `TRACE(2, ("%t: %s dcache-cta-fence[%0d]: arrive (#%0d)\n", $time, INSTANCE_ID, i, dcache_bus_if[i].req_data.tag.uuid))
            end
        end
    end
`endif
`endif

endmodule
