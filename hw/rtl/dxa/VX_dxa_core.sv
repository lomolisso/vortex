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

module VX_dxa_core import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_REQS = 1,
    parameter GMEM_OUT_PORTS = 1
) (
    input wire clk,
    input wire reset,
`ifdef PERF_ENABLE
    output dxa_perf_t dxa_perf,
`endif
    VX_dcr_bus_if.slave dcr_bus_if,
    VX_dxa_req_bus_if.slave req_bus_if[NUM_REQS],
    VX_mem_bus_if.master gmem_bus_if[GMEM_OUT_PORTS],
    VX_mem_bus_if.master smem_bus_if[1],
`ifdef DXA_MBAR_ENABLE
    // Rows of copies into another core's block, as DLMEM request flits.
    VX_dlmem_bus_if.master row_out_if,
    // Each requesting core's slot stride, a launch constant.
    input wire [`VX_CFG_SOCKET_SIZE-1:0][`VX_CFG_LMEM_LOG_SIZE:0] slot_stride,
`endif
    output wire busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
`ifdef DXA_MBAR_ENABLE
    // A transfer's core id indexes slot_stride by its socket-local bits.
    `STATIC_ASSERT (NUM_REQS == `VX_CFG_SOCKET_SIZE, ("invalid parameter"))
`endif

    // ================================================================
    // Descriptor table (single read port, driven by dispatch queue)
    // ================================================================
    wire [DXA_DESC_SLOT_W-1:0] desc_read_addr;
    dxa_desc_t desc_read_data;

    VX_dxa_desc_table desc_table (
        .clk        (clk),
        .reset      (reset),
        .dcr_bus_if (dcr_bus_if),
        .read_addr  (desc_read_addr),
        .read_desc  (desc_read_data)
    );

    // ================================================================
    // Request path: sockets → arb(N:1) → queue → dispatch(1:N) → workers
    // ================================================================
    localparam REQ_DATAW = $bits(dxa_req_data_t);

    VX_dxa_req_bus_if arb_out_bus_if[1]();

    VX_dxa_req_arb #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (1),
        // Register the arbitration output whenever it actually arbitrates
        // (>1 input). 1:1 stays a passthrough (no area/latency cost).
        .OUT_BUF     ((NUM_REQS > 1) ? 3 : 0)
    ) req_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (req_bus_if),
        .bus_out_if (arb_out_bus_if)
    );

    VX_dxa_req_bus_if queue_out_bus_if[1]();

    VX_elastic_buffer #(
        .DATAW  (REQ_DATAW),
        .SIZE   (`VX_CFG_DXA_QUEUE_SIZE),
        // LUTRAM=0: the wide-shallow queue is BRAM-eligible, moving it off the
        // LUT pool onto an (abundant) BRAM tile.
        .LUTRAM (0)
    ) req_queue (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (arb_out_bus_if[0].req_valid),
        .ready_in  (arb_out_bus_if[0].req_ready),
        .data_in   (arb_out_bus_if[0].req_data),
        .valid_out (queue_out_bus_if[0].req_valid),
        .ready_out (queue_out_bus_if[0].req_ready),
        .data_out  (queue_out_bus_if[0].req_data)
    );

    // Read desc_table using the queued request's desc_slot. desc_store is now
    // a registered-read BRAM (OUT_REG=1), so desc_read_data lands one cycle
    // after desc_read_addr. A 1-deep fetch register absorbs that latency: the
    // queue head's slot drives the address combinationally; the next cycle the
    // captured request and its descriptor are presented together to dispatch.
    // DXA launches are infrequent (one per multi-hundred-cycle transfer), so
    // the extra launch cycle is throughput-irrelevant.
    assign desc_read_addr = DXA_DESC_SLOT_W'(queue_out_bus_if[0].req_data.meta[DXA_DESC_SLOT_W-1:0]);

    // Bundle request + descriptor into dispatch input
    VX_dxa_worker_req_if dispatch_in_if[1]();

    // desc_store is a registered-read BRAM: desc_read_data lags desc_read_addr
    // by one cycle. A 2-state fetch FSM waits that cycle, and crucially holds
    // the queue head (so desc_read_addr — and therefore desc_read_data — stays
    // stable) until dispatch consumes. The queue is popped only on consume, so
    // a request and its descriptor are always presented in alignment, even when
    // dispatch back-pressures during streamed back-to-back transfers. Launch
    // latency is +1 cycle, negligible against a multi-hundred-cycle drain.
    localparam FETCH_IDLE = 1'b0, FETCH_PRESENT = 1'b1;
    reg fetch_state_r;

    // Pop only when presenting AND dispatch accepts; the head (hence the read
    // address) stays put through IDLE→PRESENT and across any dispatch stall.
    assign queue_out_bus_if[0].req_ready =
        (fetch_state_r == FETCH_PRESENT) && dispatch_in_if[0].ready;

    always @(posedge clk) begin
        if (reset) begin
            fetch_state_r <= FETCH_IDLE;
        end else begin
            case (fetch_state_r)
                FETCH_IDLE:    if (queue_out_bus_if[0].req_valid) fetch_state_r <= FETCH_PRESENT;
                FETCH_PRESENT: if (dispatch_in_if[0].ready)       fetch_state_r <= FETCH_IDLE;
            endcase
        end
    end

    assign dispatch_in_if[0].valid     = (fetch_state_r == FETCH_PRESENT);
    assign dispatch_in_if[0].req_data  = queue_out_bus_if[0].req_data;
`ifdef DXA_MBAR_ENABLE
    // A bulk copy reads no descriptor: it runs as a rank-1 box of size / 8
    // doublewords at its source, the issuer core's local-memory byte address.
    dxa_desc_t bulk_desc;
    always @(*) begin
        bulk_desc           = '0;
        bulk_desc.base_addr = `VX_CFG_MEM_ADDR_WIDTH'(queue_out_bus_if[0].req_data.src_addr);
        bulk_desc.meta      = (32'd1 << DXA_DESC_META_DIM_LSB) | (32'd3 << DXA_DESC_META_ELEMSZ_LSB);
        bulk_desc.tile01    = 32'(queue_out_bus_if[0].req_data.size >> 3);
    end
    assign dispatch_in_if[0].desc_data = queue_out_bus_if[0].req_data.bulk ? bulk_desc : desc_read_data;
`else
    assign dispatch_in_if[0].desc_data = desc_read_data;
`endif

    VX_dxa_worker_req_if worker_req_if[`VX_CFG_NUM_DXA_CORES]();

    VX_dxa_dispatch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (`VX_CFG_NUM_DXA_CORES)
    ) issue_dispatch (
        .clk       (clk),
        .reset     (reset),
        .req_in    (dispatch_in_if),
        .req_out   (worker_req_if)
    );

    // ================================================================
    // Workers
    // ================================================================
    localparam GMEM_ARB_SEL_BITS = `ARB_SEL_BITS(`VX_CFG_NUM_DXA_CORES, GMEM_OUT_PORTS);
    localparam WORKER_GMEM_TAG_WIDTH = SOCKET_MEM_TAG_WIDTH - GMEM_ARB_SEL_BITS;

    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (WORKER_GMEM_TAG_WIDTH)
    ) worker_gmem_bus_if[`VX_CFG_NUM_DXA_CORES]();

    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) worker_smem_bus_if[`VX_CFG_NUM_DXA_CORES]();

`ifdef PERF_ENABLE
    dxa_perf_t worker_dxa_perf [`VX_CFG_NUM_DXA_CORES];
`endif

    for (genvar i = 0; i < `VX_CFG_NUM_DXA_CORES; ++i) begin : g_workers
        VX_dxa_worker #(
            .INSTANCE_ID(`SFORMATF(("%s-worker%0d", INSTANCE_ID, i))),
            .WORKER_ID  (i),
            .GMEM_TAG_WIDTH (WORKER_GMEM_TAG_WIDTH)
        ) worker (
            .clk            (clk),
            .reset          (reset),
        `ifdef PERF_ENABLE
            .dxa_perf       (worker_dxa_perf[i]),
        `endif
            .req_if         (worker_req_if[i]),
            .gmem_bus_if    (worker_gmem_bus_if[i]),
        `ifdef DXA_MBAR_ENABLE
            .slot_stride    (slot_stride),
        `endif
            .smem_bus_if    (worker_smem_bus_if[i])
        );
    end

    // ================================================================
    // Output arbitration
    // ================================================================
    VX_mem_bus_arb #(
        .NUM_INPUTS  (`VX_CFG_NUM_DXA_CORES),
        .NUM_OUTPUTS (GMEM_OUT_PORTS),
        .DATA_SIZE   (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH   (WORKER_GMEM_TAG_WIDTH),
        .ARBITER     ("R"),
        .REQ_OUT_BUF ((`VX_CFG_NUM_DXA_CORES > 1) ? 3 : 0)
    ) gmem_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (worker_gmem_bus_if),
        .bus_out_if (gmem_bus_if)
    );

`ifdef DXA_MBAR_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) lmem_arb_out_if[1]();
`endif

    VX_mem_bus_arb #(
        .NUM_INPUTS  (`VX_CFG_NUM_DXA_CORES),
        .NUM_OUTPUTS (1),
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_TAG_W),
        .TAG_SEL_IDX (DXA_LMEM_TAG_W - UUID_WIDTH),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W),
        .ARBITER     ("R"),
        // Register the SMEM write request at the DXA boundary: the tiled
        // dest-address is a deep cone whose sink is the far-away core LMEM BRAM,
        // so terminate it at a local flop to keep placement compact and isolate
        // the long DXA→core route into its own cycle.
        .REQ_OUT_BUF (3)
    ) lmem_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (worker_smem_bus_if),
    `ifdef DXA_MBAR_ENABLE
        .bus_out_if (lmem_arb_out_if)
    `else
        .bus_out_if (smem_bus_if)
    `endif
    );

`ifdef DXA_MBAR_ENABLE
    // A remote copy's beats leave through the row egress, the rest to the cores.
    dxa_lmem_attr_t arb_out_attr;
    assign arb_out_attr = dxa_lmem_attr_t'(lmem_arb_out_if[0].req_data.attr);
    wire arb_out_remote = arb_out_attr.remote;
    wire row_beat_ready;

    assign smem_bus_if[0].req_valid = lmem_arb_out_if[0].req_valid && ~arb_out_remote;
    assign smem_bus_if[0].req_data  = lmem_arb_out_if[0].req_data;
    assign lmem_arb_out_if[0].req_ready = arb_out_remote ? row_beat_ready : smem_bus_if[0].req_ready;
    assign lmem_arb_out_if[0].rsp_valid = smem_bus_if[0].rsp_valid;
    assign lmem_arb_out_if[0].rsp_data  = smem_bus_if[0].rsp_data;
    assign smem_bus_if[0].rsp_ready = lmem_arb_out_if[0].rsp_ready;

    VX_dlmem_row_egress #(
        .INSTANCE_ID (`SFORMATF(("%s-row-egress", INSTANCE_ID))),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W)
    ) row_egress (
        .clk         (clk),
        .reset       (reset),
        .beat_valid  (lmem_arb_out_if[0].req_valid && arb_out_remote),
        .beat_addr   (lmem_arb_out_if[0].req_data.addr),
        .beat_data   (lmem_arb_out_if[0].req_data.data),
        .beat_byteen (lmem_arb_out_if[0].req_data.byteen),
        .beat_attr   (arb_out_attr),
        .beat_tag    (lmem_arb_out_if[0].req_data.tag),
        .beat_ready  (row_beat_ready),
        .row_out_if  (row_out_if)
    );
`endif

    // ================================================================
    // Busy / perf
    // ================================================================
    wire [NUM_REQS-1:0] req_bus_valid;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_valid
        assign req_bus_valid[i] = req_bus_if[i].req_valid;
    end

    wire [`VX_CFG_NUM_DXA_CORES-1:0] worker_idle;
    for (genvar i = 0; i < `VX_CFG_NUM_DXA_CORES; ++i) begin : g_worker_idle
        assign worker_idle[i] = worker_req_if[i].ready;
    end

    reg dxa_busy_r;
    always @(posedge clk) begin
        if (reset)
            dxa_busy_r <= 1'b0;
        else
            dxa_busy_r <= dcr_bus_if.req_valid | (|req_bus_valid) | ~(&worker_idle);
    end
    assign busy = dxa_busy_r | dcr_bus_if.req_valid | (|req_bus_valid);

`ifdef PERF_ENABLE
    always_comb begin
        dxa_perf = '0;
        for (int w = 0; w < `VX_CFG_NUM_DXA_CORES; ++w) begin
            dxa_perf.transfers   += worker_dxa_perf[w].transfers;
            dxa_perf.gmem_reads  += worker_dxa_perf[w].gmem_reads;
            dxa_perf.gmem_dedup  += worker_dxa_perf[w].gmem_dedup;
            dxa_perf.lmem_writes += worker_dxa_perf[w].lmem_writes;
            dxa_perf.gmem_latency += worker_dxa_perf[w].gmem_latency;
            dxa_perf.noslot_stalls += worker_dxa_perf[w].noslot_stalls;
        end
    end
`endif

`ifdef DBG_TRACE_DXA
    // Interface-array indexing must be elaboration-time constant; use a
    // genvar loop instead of an integer loop.
    for (genvar w = 0; w < `VX_CFG_NUM_DXA_CORES; ++w) begin : g_trace_worker
        always @(posedge clk) begin
            if (~reset && worker_req_if[w].valid) begin
                `TRACE(1, ("%t: %s dispatch-issue: worker=%0d, core=%0d, wid=%0d, meta=0x%0h\n",
                    $time, INSTANCE_ID, w,
                    worker_req_if[w].req_data.core_id,
                    worker_req_if[w].req_data.wid,
                    worker_req_if[w].req_data.meta))
            end
        end
    end
`endif

endmodule
