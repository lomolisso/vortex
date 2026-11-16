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

// LSU-side scheduler shared across FU clients.
//
// Owns the VX_mem_scheduler + dcache port driver. Multiple FU clients
// (LSU per-lane AGU, TCU_LD warp-level AGU, future RTX/TEX/OM
// preload AGUs) share a single scheduler and a single dcache port.
//
// Round-robin arbiter for NUM_CLIENTS > 1. The chosen client's tag is
// extended with a client_id sideband (CLIENT_ID_BITS bits prepended)
// so responses can be demuxed back to the originator.
// For NUM_CLIENTS=1 the arbiter degenerates to pass-through with no
// tag extension (zero-cost path for the LSU-only case).

module VX_lsu_scheduler import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_CLIENTS    = 1,
    parameter NUM_LANES      = `VX_CFG_NUM_LSU_LANES,
    parameter CORE_QUEUE_SIZE= `VX_CFG_LSU_QUEUE_IN_SIZE,
    parameter MEM_QUEUE_SIZE = LSU_QUEUE_OUT_SIZE,
    parameter PENDING_SIZE   = `VX_CFG_LSU_PENDING_SIZE
) (
    input wire clk,
    input wire reset,

    VX_lsu_sched_if.slave  client_if [NUM_CLIENTS],

    output wire             empty,

    VX_lsu_mem_if.master    lsu_mem_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam CLIENT_ID_BITS  = (NUM_CLIENTS > 1) ? $clog2(NUM_CLIENTS) : 0;
    localparam SCHED_TAG_WIDTH = LSU_CLIENT_TAG_WIDTH + CLIENT_ID_BITS;

    // -----------------------------------------------------------------------
    // Arbiter (round-robin) — feeds the single scheduler input from the
    // multi-client request channels.
    // -----------------------------------------------------------------------
    wire                            sched_req_valid;
    wire                            sched_req_rw;
    wire [NUM_LANES-1:0]            sched_req_mask;
    wire [NUM_LANES-1:0][LSU_WORD_SIZE-1:0]    sched_req_byteen;
    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0]   sched_req_addr;
    wire [NUM_LANES-1:0][MEM_ATTR_WIDTH-1:0]   sched_req_attr;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] sched_req_data;
    wire [SCHED_TAG_WIDTH-1:0]      sched_req_tag;
    wire                            sched_req_ready;

    wire                            sched_rsp_valid;
    wire [NUM_LANES-1:0]            sched_rsp_mask;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] sched_rsp_data;
    wire [SCHED_TAG_WIDTH-1:0]      sched_rsp_tag;
    wire                            sched_rsp_sop;
    wire                            sched_rsp_eop;
    wire                            sched_rsp_ready;

    // Flatten interface array into per-signal vectors so the arbiter
    // logic stays index-friendly (SystemVerilog interface arrays can't
    // be indexed inside expressions).
    wire [NUM_CLIENTS-1:0] cli_req_valid;
    lsu_req_data_t         cli_req_data [NUM_CLIENTS];
    wire [NUM_CLIENTS-1:0] cli_req_ready;
    wire [NUM_CLIENTS-1:0] cli_rsp_valid;
    lsu_rsp_data_t         cli_rsp_data [NUM_CLIENTS];
    wire [NUM_CLIENTS-1:0] cli_rsp_ready;

    for (genvar i = 0; i < NUM_CLIENTS; ++i) begin : g_cli_flat
        assign cli_req_valid[i] = client_if[i].req_valid;
        assign cli_req_data[i]  = client_if[i].req_data;
        assign client_if[i].req_ready = cli_req_ready[i];

        assign client_if[i].rsp_valid = cli_rsp_valid[i];
        assign client_if[i].rsp_data  = cli_rsp_data[i];
        assign cli_rsp_ready[i] = client_if[i].rsp_ready;
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        // Consumed by the DLMEM translation ahead of the scheduler.
        `UNUSED_VAR (cli_req_data[i].is_async)
        `UNUSED_VAR (cli_req_data[i].mbar_addr)
    `endif
    end

    if (NUM_CLIENTS == 1) begin : g_no_arb
        // Pure pass-through — no client_id, no rotation, no tag extension.
        assign sched_req_valid    = cli_req_valid[0];
        assign sched_req_rw       = cli_req_data[0].rw;
        assign sched_req_mask     = cli_req_data[0].mask;
        assign sched_req_byteen   = cli_req_data[0].byteen;
        assign sched_req_addr     = cli_req_data[0].addr;
        assign sched_req_attr     = cli_req_data[0].attr;
        assign sched_req_data     = cli_req_data[0].data;
        assign sched_req_tag      = cli_req_data[0].tag;
        assign cli_req_ready[0]   = sched_req_ready;

        assign cli_rsp_valid[0]   = sched_rsp_valid;
        assign cli_rsp_data[0].mask  = sched_rsp_mask;
        assign cli_rsp_data[0].data  = sched_rsp_data;
        assign cli_rsp_data[0].tag   = sched_rsp_tag;
        assign cli_rsp_data[0].sop   = sched_rsp_sop;
        assign cli_rsp_data[0].eop   = sched_rsp_eop;
        assign sched_rsp_ready    = cli_rsp_ready[0];
    end else begin : g_arb
        // Round-robin priority. rr_ptr rotates after every accepted request.
        // For NUM_CLIENTS=2 this trivially alternates; the generic loop
        // generalizes to wider clients later.
        reg  [CLIENT_ID_BITS-1:0] rr_ptr_r;

        // Priority chain starting at rr_ptr (rotated), pick lowest set bit.
        wire [(2*NUM_CLIENTS)-1:0] rotated_double = {cli_req_valid, cli_req_valid} >> rr_ptr_r;
        wire [NUM_CLIENTS-1:0]     rotated_valid  = rotated_double[NUM_CLIENTS-1:0];
        `UNUSED_VAR (rotated_double[(2*NUM_CLIENTS)-1:NUM_CLIENTS])

        logic [CLIENT_ID_BITS-1:0] rotated_grant_idx;
        logic                       any_valid;
        always_comb begin
            rotated_grant_idx = '0;
            any_valid         = 1'b0;
            for (int i = NUM_CLIENTS-1; i >= 0; i--) begin
                if (rotated_valid[i]) begin
                    rotated_grant_idx = CLIENT_ID_BITS'(i);
                    any_valid         = 1'b1;
                end
            end
        end

        wire [CLIENT_ID_BITS-1:0] grant_idx = rr_ptr_r + rotated_grant_idx;
        wire                       grant_fire = any_valid && sched_req_ready;

        always @(posedge clk) begin
            if (reset) begin
                rr_ptr_r <= '0;
            end else if (grant_fire) begin
                rr_ptr_r <= grant_idx + CLIENT_ID_BITS'(1);
            end
        end

        // Mux selected client's request to the scheduler. Tag is extended
        // with grant_idx so the response demux can route back.
        assign sched_req_valid  = any_valid;
        assign sched_req_rw     = cli_req_data[grant_idx].rw;
        assign sched_req_mask   = cli_req_data[grant_idx].mask;
        assign sched_req_byteen = cli_req_data[grant_idx].byteen;
        assign sched_req_addr   = cli_req_data[grant_idx].addr;
        assign sched_req_attr   = cli_req_data[grant_idx].attr;
        assign sched_req_data   = cli_req_data[grant_idx].data;
        assign sched_req_tag    = {grant_idx, cli_req_data[grant_idx].tag};

        // Drive per-client ready: only the granted client sees ready high.
        for (genvar i = 0; i < NUM_CLIENTS; ++i) begin : g_creq_rdy
            assign cli_req_ready[i] = sched_req_ready && (grant_idx == CLIENT_ID_BITS'(i));
        end

        // Demux response: extract client_id from extended tag.
        wire [CLIENT_ID_BITS-1:0] rsp_cid = sched_rsp_tag[SCHED_TAG_WIDTH-1 -: CLIENT_ID_BITS];
        for (genvar i = 0; i < NUM_CLIENTS; ++i) begin : g_crsp
            assign cli_rsp_valid[i]    = sched_rsp_valid && (rsp_cid == CLIENT_ID_BITS'(i));
            assign cli_rsp_data[i].mask = sched_rsp_mask;
            assign cli_rsp_data[i].data = sched_rsp_data;
            assign cli_rsp_data[i].tag  = sched_rsp_tag[LSU_CLIENT_TAG_WIDTH-1:0];
            assign cli_rsp_data[i].sop  = sched_rsp_sop;
            assign cli_rsp_data[i].eop  = sched_rsp_eop;
        end
        // sched_rsp_ready is driven by the granted client's ready.
        logic sched_rsp_ready_l;
        always_comb begin
            sched_rsp_ready_l = 1'b0;
            for (int i = 0; i < NUM_CLIENTS; i++) begin
                if (rsp_cid == CLIENT_ID_BITS'(i)) sched_rsp_ready_l = cli_rsp_ready[i];
            end
        end
        assign sched_rsp_ready = sched_rsp_ready_l;
    end

    // -----------------------------------------------------------------------
    // VX_mem_scheduler — single instance, shared across clients.
    // -----------------------------------------------------------------------
    wire                                       lsu_mem_req_valid;
    wire                                       lsu_mem_req_rw;
    wire [NUM_LANES-1:0]                       lsu_mem_req_mask;
    wire [NUM_LANES-1:0][LSU_WORD_SIZE-1:0]    lsu_mem_req_byteen;
    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0]   lsu_mem_req_addr;
    wire [NUM_LANES-1:0][MEM_ATTR_WIDTH-1:0]   lsu_mem_req_attr;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] lsu_mem_req_data;
    wire [LSU_TAG_WIDTH-1:0]                   lsu_mem_req_tag;
    wire                                       lsu_mem_req_ready;

    wire                                       lsu_mem_rsp_valid;
    wire [NUM_LANES-1:0]                       lsu_mem_rsp_mask;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] lsu_mem_rsp_data;
    wire [LSU_TAG_WIDTH-1:0]                   lsu_mem_rsp_tag;
    wire                                       lsu_mem_rsp_ready;

    wire                                       sched_req_queue_empty;

    VX_mem_scheduler #(
        .INSTANCE_ID (`SFORMATF(("%s-memsched", INSTANCE_ID))),
        .CORE_REQS   (NUM_LANES),
        .MEM_CHANNELS(NUM_LANES),
        .WORD_SIZE   (LSU_WORD_SIZE),
        .LINE_SIZE   (LSU_WORD_SIZE),
        .ADDR_WIDTH  (LSU_ADDR_WIDTH),
        .USER_WIDTH  (MEM_ATTR_WIDTH),
        .TAG_WIDTH   (SCHED_TAG_WIDTH),
        .CORE_QUEUE_SIZE (CORE_QUEUE_SIZE),
        .MEM_QUEUE_SIZE (MEM_QUEUE_SIZE),
        .PENDING_SIZE   (PENDING_SIZE),
        .UUID_WIDTH  (UUID_WIDTH),
        .RSP_PARTIAL (1),
        .MEM_OUT_BUF (0),
        .CORE_OUT_BUF(0),
        .LUTRAM      (1)
    ) mem_scheduler (
        .clk            (clk),
        .reset          (reset),

        .core_req_valid (sched_req_valid),
        .core_req_rw    (sched_req_rw),
        .core_req_mask  (sched_req_mask),
        .core_req_byteen(sched_req_byteen),
        .core_req_addr  (sched_req_addr),
        .core_req_user  (sched_req_attr),
        .core_req_data  (sched_req_data),
        .core_req_tag   (sched_req_tag),
        .core_req_ready (sched_req_ready),
        .req_queue_empty(sched_req_queue_empty),
        `UNUSED_PIN (req_queue_rw_notify),

        .core_rsp_valid (sched_rsp_valid),
        .core_rsp_mask  (sched_rsp_mask),
        .core_rsp_data  (sched_rsp_data),
        .core_rsp_tag   (sched_rsp_tag),
        .core_rsp_sop   (sched_rsp_sop),
        .core_rsp_eop   (sched_rsp_eop),
        .core_rsp_ready (sched_rsp_ready),

        .mem_req_valid  (lsu_mem_req_valid),
        .mem_req_rw     (lsu_mem_req_rw),
        .mem_req_mask   (lsu_mem_req_mask),
        .mem_req_byteen (lsu_mem_req_byteen),
        .mem_req_addr   (lsu_mem_req_addr),
        .mem_req_user   (lsu_mem_req_attr),
        .mem_req_data   (lsu_mem_req_data),
        .mem_req_tag    (lsu_mem_req_tag),
        .mem_req_ready  (lsu_mem_req_ready),

        .mem_rsp_valid  (lsu_mem_rsp_valid),
        .mem_rsp_mask   (lsu_mem_rsp_mask),
        .mem_rsp_data   (lsu_mem_rsp_data),
        .mem_rsp_tag    (lsu_mem_rsp_tag),
        .mem_rsp_ready  (lsu_mem_rsp_ready)
    );

    // -----------------------------------------------------------------------
    // Drive the dcache port.
    // -----------------------------------------------------------------------
`ifdef VX_CFG_H100_SM_ENABLE
    // A global load completes no earlier than its L1 hit time after it left
    // here: l1_hit_latency(P) per dependent hop (05 sec 4.3, l1_ref.h), net of
    // the core stages before this point and after the response
    // (L1_CORE_STAGES). A response that would complete it earlier waits in a
    // queue, in arrival order, so it blocks neither the L1 nor the responses
    // behind it; a longer path (a miss) is not delayed.
    localparam HOLD_W      = 16;
    localparam ID_BITS     = `LOG2UP(PENDING_SIZE);
    localparam PASS_LANES  = NUM_LANES / `VX_CFG_L1_ADDR_PASSES;
    localparam LINE_ASHIFT = `CLOG2(`VX_CFG_DCACHE_LINE_SIZE / LSU_WORD_SIZE);
    localparam REQ_ASHIFT  = `CLOG2(LSU_WORD_SIZE);
    `STATIC_ASSERT(NUM_LANES == `VX_CFG_NUM_THREADS, ("invalid parameter: the address passes span a warp"))
    `STATIC_ASSERT(L1_PIPE_OVERHEAD <= `VX_CFG_L1_HIT_LATENCY, ("invalid parameter: L1_HIT_LATENCY below the pipeline's own hit time"))

    wire [NUM_LANES-1:0] gload_lane;
    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-LINE_ASHIFT-1:0] lane_line;
    wire [NUM_LANES-1:0] lane_amo;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_gload_lane
        mem_bus_attr_t lane_attr;
        assign lane_attr = mem_bus_attr_t'(lsu_mem_req_attr[i]);
        `UNUSED_VAR (lane_attr)
        wire [`VX_CFG_XLEN-1:0] lane_baddr = `VX_CFG_XLEN'(lsu_mem_req_addr[i]) << REQ_ASHIFT;
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        localparam WIN_LSB = `VX_CFG_LMEM_LOG_SIZE + DLMEM_RANK_BITS;
        wire lane_in_win = ((lane_baddr >> WIN_LSB) == (`VX_CFG_XLEN'(`VX_MEM_DLMEM_BASE_ADDR) >> WIN_LSB));
    `else
        wire lane_in_win = 1'b0;
    `endif
        assign gload_lane[i] = lsu_mem_req_mask[i] && ~lane_attr.is_addr_local && ~lane_attr.is_addr_io && ~lane_in_win;
        assign lane_line[i]  = lsu_mem_req_addr[i][LSU_ADDR_WIDTH-1:LINE_ASHIFT];
        assign lane_amo[i]   = lane_attr.amo.amo_valid || lane_attr.is_fence;
    end

    // P: per address pass, the distinct lines its lanes touch
    wire [NUM_LANES-1:0] lane_first;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane_first
        localparam PASS_LO = (i / PASS_LANES) * PASS_LANES;
        wire [NUM_LANES-1:0] dup;
        for (genvar j = 0; j < NUM_LANES; ++j) begin : g_j
            if ((j >= PASS_LO) && (j < i)) begin : g_prev
                assign dup[j] = gload_lane[j] && (lane_line[j] == lane_line[i]);
            end else begin : g_none
                assign dup[j] = 1'b0;
            end
        end
        assign lane_first[i] = gload_lane[i] && ~(| dup);
    end
    wire [`CLOG2(NUM_LANES+1)-1:0] req_passes;
    `POP_COUNT(req_passes, lane_first);

    wire req_gload = ~lsu_mem_req_rw && (| gload_lane) && ~(| lane_amo);
    wire [HOLD_W-1:0] req_hit_lat = HOLD_W'((`VX_CFG_L1_LINE_BASE_CENTICYCLES + `VX_CFG_L1_LINE_STEP_CENTICYCLES * 32'(req_passes) + 50) / 100);
    wire [HOLD_W-1:0] req_hold = req_hit_lat - HOLD_W'(L1_CORE_STAGES);

    reg [HOLD_W-1:0] now_r;
    reg [PENDING_SIZE-1:0] hold_valid;
    reg [HOLD_W-1:0] hold_until [PENDING_SIZE];
    reg [NUM_LANES-1:0] hold_rem [PENDING_SIZE];

    wire [ID_BITS-1:0] req_id = lsu_mem_req_tag[ID_BITS-1:0];
    wire [ID_BITS-1:0] rsp_id = lsu_mem_if.rsp_data.tag[ID_BITS-1:0];
    wire lsu_mem_req_fire = lsu_mem_req_valid && lsu_mem_req_ready;

    always @(posedge clk) begin
        if (reset) begin
            now_r      <= '0;
            hold_valid <= '0;
        end else begin
            now_r <= now_r + HOLD_W'(1);
            if (lsu_mem_req_fire) begin
                hold_valid[req_id] <= req_gload;
            end
        end
        if (lsu_mem_req_fire) begin
            hold_until[req_id] <= now_r + req_hold;
            hold_rem[req_id]   <= lsu_mem_req_mask;
        end
        if (lsu_mem_if.rsp_valid && lsu_mem_if.rsp_ready) begin
            hold_rem[rsp_id] <= hold_rem[rsp_id] & ~lsu_mem_if.rsp_data.mask;
        end
    end

    wire [HOLD_W-1:0] rsp_wait = hold_until[rsp_id] - now_r;
    wire rsp_last = ((hold_rem[rsp_id] & ~lsu_mem_if.rsp_data.mask) == '0);
    wire rsp_early = hold_valid[rsp_id] && rsp_last && (rsp_wait != '0) && ~rsp_wait[HOLD_W-1];

    localparam HQ_DATAW = NUM_LANES + (NUM_LANES * LSU_WORD_SIZE * 8) + LSU_TAG_WIDTH;
    wire hq_empty, hq_full, hq_ready, hq_push, hq_pop;
    wire [NUM_LANES-1:0] hq_mask;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] hq_data;
    wire [LSU_TAG_WIDTH-1:0] hq_tag;

    VX_fifo_queue #(
        .DATAW (HQ_DATAW),
        .DEPTH (`VX_CFG_L1_HIT_LATENCY)
    ) hold_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (hq_push),
        .pop      (hq_pop),
        .data_in  ({lsu_mem_if.rsp_data.mask, lsu_mem_if.rsp_data.data, lsu_mem_if.rsp_data.tag}),
        .data_out ({hq_mask, hq_data, hq_tag}),
        .empty    (hq_empty),
        `UNUSED_PIN (alm_empty),
        .full     (hq_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    wire [HOLD_W-1:0] hq_wait = hold_until[hq_tag[ID_BITS-1:0]] - now_r;
    assign hq_ready = ~hq_empty && ((hq_wait == '0) || hq_wait[HOLD_W-1]);
    assign hq_push  = lsu_mem_if.rsp_valid && rsp_early && ~hq_full;
    assign hq_pop   = hq_ready && lsu_mem_rsp_ready;

`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (lsu_mem_req_fire && req_gload) begin
            `TRACE(2, ("%t: %s l1-hold req: id=%0d, passes=%0d, until=%0d\n", $time, INSTANCE_ID, req_id, req_passes, now_r + req_hold))
        end
        if (lsu_mem_rsp_valid && lsu_mem_rsp_ready) begin
            `TRACE(2, ("%t: %s l1-hold rsp: id=%0d, held=%b\n", $time, INSTANCE_ID, lsu_mem_rsp_tag[ID_BITS-1:0], hq_ready))
        end
    end
`endif
`endif

    assign lsu_mem_if.req_valid       = lsu_mem_req_valid;
    assign lsu_mem_if.req_data.mask   = lsu_mem_req_mask;
    assign lsu_mem_if.req_data.rw     = lsu_mem_req_rw;
    assign lsu_mem_if.req_data.byteen = lsu_mem_req_byteen;
    assign lsu_mem_if.req_data.addr   = lsu_mem_req_addr;
    assign lsu_mem_if.req_data.user   = lsu_mem_req_attr;
    assign lsu_mem_if.req_data.data   = lsu_mem_req_data;
    assign lsu_mem_if.req_data.tag    = lsu_mem_req_tag;
    assign lsu_mem_req_ready          = lsu_mem_if.req_ready;

`ifdef VX_CFG_H100_SM_ENABLE
    // the held response whose time has come goes first
    assign lsu_mem_rsp_valid          = hq_ready || (lsu_mem_if.rsp_valid && ~rsp_early);
    assign lsu_mem_rsp_mask           = hq_ready ? hq_mask : lsu_mem_if.rsp_data.mask;
    assign lsu_mem_rsp_data           = hq_ready ? hq_data : lsu_mem_if.rsp_data.data;
    assign lsu_mem_rsp_tag            = hq_ready ? hq_tag : lsu_mem_if.rsp_data.tag;
    assign lsu_mem_if.rsp_ready       = rsp_early ? ~hq_full : (~hq_ready && lsu_mem_rsp_ready);
`else
    assign lsu_mem_rsp_valid          = lsu_mem_if.rsp_valid;
    assign lsu_mem_rsp_mask           = lsu_mem_if.rsp_data.mask;
    assign lsu_mem_rsp_data           = lsu_mem_if.rsp_data.data;
    assign lsu_mem_rsp_tag            = lsu_mem_if.rsp_data.tag;
    assign lsu_mem_if.rsp_ready       = lsu_mem_rsp_ready;
`endif

    assign empty = sched_req_queue_empty;

endmodule
