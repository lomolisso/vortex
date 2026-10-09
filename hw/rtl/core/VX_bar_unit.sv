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

module VX_bar_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire                clk,
    input wire                reset,

    // local barrier interface
    input wire                req_valid,
    input wire [NW_WIDTH-1:0] req_wid,
    input barrier_t           req_data,
    input wire [BAR_ADDR_W-1:0] read_addr, // valid one cycle before req_valid
    output wire               read_phase,  // asynchronous phase bit return at read_addr

    // global barrier interface
    VX_gbar_bus_if.master gbar_bus_if,

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // with read_addr: the read targets the slot's cluster barrier
    input wire                read_cluster,

    // a CTA slot's cluster-barrier record, written when its block is admitted
    input wire                cbar_wr_valid,
    input wire [NW_WIDTH-1:0] cbar_wr_slot,
    input cbar_slot_t         cbar_wr_info,

    // CTA_DONE events of this core's SPREAD blocks, toward the gang unit
    input wire                cta_done_valid,
    input dlmem_cta_done_t    cta_done_data,
    output wire               cta_done_ready,
`endif

    // scheduler interface
    input wire [`VX_CFG_NUM_WARPS-1:0] active_warps,
    output wire               unlock_valid, // unlock stalled warps
    output wire [`VX_CFG_NUM_WARPS-1:0] unlock_mask // warps to unlock
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    //                    warp mask + warp count + event count
    localparam EVENT_WIDTH = `CLOG2(`VX_CFG_MAX_BAR_EVENTS + 1);
    localparam BAR_STATEW = `VX_CFG_NUM_WARPS + NW_WIDTH + EVENT_WIDTH;
    localparam GBAR_REQW = NB_WIDTH + NC_WIDTH;
    localparam USE_GBAR = (`VX_CFG_NUM_CORES > 1);

    logic [`VX_CFG_NUM_WARPS-1:0] mask_r, mask_n;
    logic [NW_WIDTH-1:0]   count_r, count_n;
    logic [EVENT_WIDTH-1:0] events_r, events_n;
    logic                  phase_r, phase_n;

    logic                  unlock_valid_n;
    logic [`VX_CFG_NUM_WARPS-1:0] unlock_mask_n;

    // A response carries only its id after the barrier-store read port has moved on.
    logic [`VX_CFG_NUM_BARRIERS-1:0] gbar_pending_r, gbar_pending_n;
    logic [`VX_CFG_NUM_BARRIERS-1:0] gbar_pending_phase_r, gbar_pending_phase_n;
    logic [`VX_CFG_NUM_BARRIERS-1:0][`VX_CFG_NUM_WARPS-1:0] gbar_waiters_r, gbar_waiters_n;

    logic gbar_enqueue;
    logic [NB_WIDTH-1:0] gbar_enqueue_id;
    logic [NC_WIDTH-1:0] gbar_enqueue_size_m1;

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // A cluster-barrier release touches no store port, so it is taken in any cycle.
    wire gbar_rsp_ready = ~req_valid | gbar_bus_if.rsp_data.kind;
    wire gbar_rsp_fire = gbar_bus_if.rsp_valid && gbar_rsp_ready;
    wire gbar_rsp_apply = USE_GBAR && gbar_rsp_fire && ~gbar_bus_if.rsp_data.kind && gbar_pending_r[gbar_bus_if.rsp_data.id];
`else
    wire gbar_rsp_ready = ~req_valid;
    wire gbar_rsp_fire = gbar_bus_if.rsp_valid && gbar_rsp_ready;
    wire gbar_rsp_apply = USE_GBAR && gbar_rsp_fire && gbar_pending_r[gbar_bus_if.rsp_data.id];
`endif

    wire [`VX_CFG_NUM_WARPS-1:0] wait_mask = ((`VX_CFG_NUM_WARPS)'(1) << req_wid) | mask_r;
    wire [NW_WIDTH-1:0] next_count  = count_r + NW_WIDTH'(1);
    wire next_phase  = ~phase_r;

    always @(*) begin
        mask_n  = mask_r;
        count_n = count_r;
        events_n = events_r;
        phase_n = phase_r;
        unlock_valid_n = 0;
        unlock_mask_n = 'x;
        gbar_pending_n = gbar_pending_r;
        gbar_pending_phase_n = gbar_pending_phase_r;
        gbar_waiters_n = gbar_waiters_r;
        gbar_enqueue = 0;
        gbar_enqueue_id = 'x;
        gbar_enqueue_size_m1 = 'x;

        // local barrier scheduling
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        if (req_valid && ~req_data.is_global && ~req_data.is_cluster) begin
    `else
        if (req_valid && ~req_data.is_global) begin
    `endif
            if (req_data.is_event) begin
                // event tracking
                if (req_data.phase) begin
                    // attach/expect_tx: increment by (size_m1 + 1) so that
                    // a software expect_tx(N) adds N events, while ordinary
                    // single-event paths (size_m1=0) add 1.
                    events_n = events_r + EVENT_WIDTH'(req_data.size_m1) + EVENT_WIDTH'(1);
                end else begin
                    events_n = events_r - EVENT_WIDTH'(1);
                end
                // unlock warps if decrementing event to 0 and all all warps have arrived
                if ((req_data.phase == 0) && (events_r == EVENT_WIDTH'(1)) && (count_r == 0)) begin
                    mask_n = '0;
                    unlock_valid_n = 1; // release waiting warps
                    unlock_mask_n = mask_r;
                    phase_n = next_phase; // advance phase
                end
            end else if (req_data.is_arrive) begin
                // barrier arrival
                if (count_r == NW_WIDTH'(req_data.size_m1)) begin
                    count_n = '0;
                    if (events_r == 0) begin
                        mask_n = '0;
                        unlock_valid_n = 1; // release waiting warps
                        unlock_mask_n = req_data.is_sync ? wait_mask : mask_r;
                        phase_n = next_phase; // advance phase
                    end else if (req_data.is_sync) begin
                        // Add arriving warp to wait mask
                        mask_n = wait_mask;
                    end
                end else begin
                    count_n = next_count;
                    if (req_data.is_sync) begin
                        mask_n = wait_mask;
                    end
                end
            end else begin
                // barrier waiting
                if (req_data.phase != phase_r) begin
                    unlock_valid_n = 1; // release warp
                    unlock_mask_n = (`VX_CFG_NUM_WARPS)'(1) << req_wid;
                end else begin
                    // add warp to wait list
                    mask_n = wait_mask;
                end
            end
        end
        if (USE_GBAR) begin
            // global barrier scheduling
            if (req_valid && req_data.is_global) begin
                if (req_data.is_event) begin
                    // event tracking
                    if (req_data.phase) begin
                        // expect_tx(N): add (size_m1 + 1)
                        events_n = events_r + EVENT_WIDTH'(req_data.size_m1) + EVENT_WIDTH'(1);
                    end else begin
                        events_n = events_r - EVENT_WIDTH'(1);
                    end
                    // unlock warps if decrementing event to 0 and all warps have arrived
                    if ((req_data.phase == 0) && (events_r == EVENT_WIDTH'(1)) && (wait_mask == active_warps)) begin
                        mask_n = '0;
                        gbar_enqueue = 1; // notify global barrier
                        gbar_enqueue_id = req_data.id;
                        gbar_enqueue_size_m1 = NC_WIDTH'(count_r); // was saved in barrier_arrive
                    end
                end else if (req_data.is_arrive) begin
                    // barrier arrival
                    count_n = NW_WIDTH'(req_data.size_m1); // store participating number of cores
                    if (req_data.is_sync) begin
                        gbar_waiters_n[req_data.id][req_wid] = 1;
                    end
                    if (wait_mask == active_warps && events_r == 0) begin
                        mask_n = '0;
                        gbar_enqueue = 1; // notify global barrier
                        gbar_enqueue_id = req_data.id;
                        gbar_enqueue_size_m1 = NC_WIDTH'(req_data.size_m1);
                    end else begin
                        // Add arriving warp to arrival mask
                        mask_n = wait_mask;
                    end
                end else begin
                    // barrier waiting
                    if (req_data.phase != phase_r) begin
                        unlock_valid_n = 1; // release warp
                        unlock_mask_n = (`VX_CFG_NUM_WARPS)'(1) << req_wid;
                    end else begin
                        // add warp to wait list
                        gbar_waiters_n[req_data.id][req_wid] = 1;
                    end
                end
            end

            // global barrier response handling
            if (gbar_rsp_apply) begin
                unlock_valid_n = (gbar_waiters_r[gbar_bus_if.rsp_data.id] != '0);
                unlock_mask_n = gbar_waiters_r[gbar_bus_if.rsp_data.id];
                gbar_waiters_n[gbar_bus_if.rsp_data.id] = '0;
                gbar_pending_n[gbar_bus_if.rsp_data.id] = 0;
            end

            if (gbar_enqueue) begin
                gbar_pending_n[gbar_enqueue_id] = 1;
                gbar_pending_phase_n[gbar_enqueue_id] = phase_r;
            end
        end
    end

    // Barriers store
    wire [BAR_STATEW-1:0] store_state_rdata;
    wire                  store_phase_rdata;
    wire [BAR_ADDR_W-1:0] store_raddr = read_addr;
    reg [BAR_ADDR_W-1:0]  store_waddr;
    wire [BAR_STATEW-1:0] store_state_wdata = {mask_n, count_n, events_n};
    wire                  store_state_write = req_valid;
    wire [BAR_ADDR_W-1:0] store_phase_waddr = gbar_rsp_apply ? BAR_ADDR_W'(gbar_bus_if.rsp_data.id) : store_waddr;
    wire                  store_phase_wdata = gbar_rsp_apply ? ~gbar_pending_phase_r[gbar_bus_if.rsp_data.id] : phase_n;
    wire                  store_phase_write = req_valid || gbar_rsp_apply;

    VX_dp_ram #(
        .DATAW    (BAR_STATEW),
        .SIZE     (1 << BAR_ADDR_BITS),
        .RDW_MODE ("W"),
        .OUT_REG  (1)
    ) barrier_state_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (store_state_write),
        .wren  (1'b1),
        .raddr (store_raddr),
        .waddr (store_waddr),
        .wdata (store_state_wdata),
        .rdata (store_state_rdata)
    );

    VX_dp_ram #(
        .DATAW    (1),
        .SIZE     (1 << BAR_ADDR_BITS),
        .RDW_MODE ("W"),
        .RADDR_REG(1)
    ) barrier_phase_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (store_phase_write),
        .wren  (1'b1),
        .raddr (store_raddr),
        .waddr (store_phase_waddr),
        .wdata (store_phase_wdata),
        .rdata (store_phase_rdata)
    );

    // Store reset handling
    reg [(1 << BAR_ADDR_BITS)-1:0] store_valids;
    wire is_phase_rdw_hazard = store_phase_write && (store_phase_waddr == store_raddr);

    wire store_phase_rdata_v = store_valids[store_raddr] ? store_phase_rdata : '0;

    always @(posedge clk) begin
        if (reset) begin
            store_valids <= '0;
            phase_r <= '0;
        end else begin
            if (store_state_write) begin
                store_valids[store_waddr] <= 1'b1;
            end
            phase_r <= is_phase_rdw_hazard ? store_phase_wdata : store_phase_rdata_v;
        end
        store_waddr <= store_raddr;
    end

    assign {mask_r, count_r, events_r} = store_valids[store_waddr] ? store_state_rdata : '0;

    wire phase_async = is_phase_rdw_hazard ? store_phase_wdata : store_phase_rdata_v;

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // Cluster barriers: one per CTA slot, in flops beside the store, whose
    // {slot, id 0} entry belongs to __syncthreads, which a block may run between
    // a cluster arrive and its wait. A block's arrival counts once all of its
    // warps arrived. A gang's block then sends one CBAR to the gang unit and
    // waits for its release; a cluster inside this core (legacy, or K=1) counts
    // its blocks at the base slot and releases all of them with the last.
    localparam CK_W = $bits(cbar_wr_info.k);

    cbar_slot_t [`VX_CFG_NUM_WARPS-1:0]                cinfo_r;
    reg [`VX_CFG_NUM_WARPS-1:0][NW_WIDTH-1:0]          ccount_r, ccount_n;   // warps arrived, per block
    reg [`VX_CFG_NUM_WARPS-1:0][CK_W-1:0]              cblk_r, cblk_n;       // blocks arrived, at a base slot
    reg [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_NUM_WARPS-1:0] cwait_r, cwait_n;     // suspended warps
    reg [`VX_CFG_NUM_WARPS-1:0]                        cpend_r, cpend_n;     // arrived, waiting for the gang
    reg [`VX_CFG_NUM_WARPS-1:0]                        cphase_r, cphase_n;

    // the slot is the high half of a barrier address
    wire [NW_WIDTH-1:0] creq_slot, crd_slot;
    if (NW_BITS != 0) begin : g_cslot
        assign creq_slot = store_waddr[NB_BITS +: NW_BITS];
        assign crd_slot  = read_addr[NB_BITS +: NW_BITS];
    end else begin : g_cslot0
        assign creq_slot = '0;
        assign crd_slot  = '0;
    end

    wire creq = req_valid && req_data.is_cluster;
    wire crsp = gbar_bus_if.rsp_valid && gbar_bus_if.rsp_ready && gbar_bus_if.rsp_data.kind;
    // the request slot's record, and the block count of its cluster's base slot
    cbar_slot_t cinfo_s;
    assign cinfo_s = cinfo_r[creq_slot];
    wire [CK_W-1:0] cbase_k = cinfo_r[cinfo_s.base].k;
    wire cblk_last = (32'(cblk_r[cinfo_s.base]) + 1) == 32'(cbase_k);

    logic                              cbar_push;
    logic [`VX_CFG_NUM_WARPS-1:0]      crel_mask;
    logic [`VX_CFG_NUM_WARPS-1:0]      cunlock_mask;

    always @(*) begin
        ccount_n     = ccount_r;
        cblk_n       = cblk_r;
        cwait_n      = cwait_r;
        cpend_n      = cpend_r;
        cphase_n     = cphase_r;
        cbar_push    = 0;
        crel_mask    = '0;
        cunlock_mask = '0;
        if (creq) begin
            if (req_data.is_arrive) begin
                if (req_data.is_sync) begin
                    cwait_n[creq_slot][req_wid] = 1;
                end
                if (ccount_r[creq_slot] != NW_WIDTH'(req_data.size_m1)) begin
                    ccount_n[creq_slot] = ccount_r[creq_slot] + NW_WIDTH'(1);
                end else begin
                    ccount_n[creq_slot] = '0;
                    if (cinfo_s.gang) begin
                        cbar_push = 1;
                        cpend_n[creq_slot] = 1;
                    end else if (cblk_last) begin
                        cblk_n[cinfo_s.base] = '0;
                        for (integer i = 0; i < `VX_CFG_NUM_WARPS; ++i) begin
                            crel_mask[i] = (32'(i) >= 32'(cinfo_s.base)) && (32'(i) < (32'(cinfo_s.base) + 32'(cbase_k)));
                        end
                    end else begin
                        cblk_n[cinfo_s.base] = cblk_r[cinfo_s.base] + CK_W'(1);
                    end
                end
            end else if (req_data.phase != cphase_r[creq_slot]) begin
                cunlock_mask[req_wid] = 1;
            end else begin
                cwait_n[creq_slot][req_wid] = 1;
            end
        end
        // a gang's release frees its pending slots; one for an exited or re-admitted slot matches nothing
        if (crsp) begin
            for (integer i = 0; i < `VX_CFG_NUM_WARPS; ++i) begin
                if (cpend_r[i] && cinfo_r[i].gang && (cinfo_r[i].gang_id == gbar_bus_if.rsp_data.gang_id)
                 && (cinfo_r[i].epoch == gbar_bus_if.rsp_data.epoch)) begin
                    crel_mask[i] = 1;
                end
            end
        end
        for (integer i = 0; i < `VX_CFG_NUM_WARPS; ++i) begin
            if (crel_mask[i]) begin
                cunlock_mask |= cwait_n[i];
                cwait_n[i]  = '0;
                cpend_n[i]  = 0;
                cphase_n[i] = ~cphase_r[i];
            end
        end
        // a newly admitted block starts on fresh state
        if (cbar_wr_valid) begin
            ccount_n[cbar_wr_slot] = '0;
            cblk_n[cbar_wr_slot]   = '0;
            cwait_n[cbar_wr_slot]  = '0;
            cpend_n[cbar_wr_slot]  = 0;
            cphase_n[cbar_wr_slot] = 0;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            ccount_r <= '0;
            cblk_r   <= '0;
            cwait_r  <= '0;
            cpend_r  <= '0;
            cphase_r <= '0;
        end else begin
            ccount_r <= ccount_n;
            cblk_r   <= cblk_n;
            cwait_r  <= cwait_n;
            cpend_r  <= cpend_n;
            cphase_r <= cphase_n;
        end
        if (cbar_wr_valid) begin
            cinfo_r[cbar_wr_slot] <= cbar_wr_info;
        end
    end

    `RUNTIME_ASSERT(~creq || (~req_data.is_event && ~req_data.is_global && (req_data.id == '0)),
        ("%t: %s cluster barrier request with event, global or id bits set: id=%0d", $time, INSTANCE_ID, req_data.id))
    `RUNTIME_ASSERT(~(creq && req_data.is_arrive) || ~cpend_r[creq_slot],
        ("%t: %s cluster barrier arrival before its phase completed: slot=%0d, wid=%0d", $time, INSTANCE_ID, creq_slot, req_wid))

    reg unlock_valid_r;
    reg [`VX_CFG_NUM_WARPS-1:0] unlock_mask_r;

    // cluster unlocks join the local or global unlock of the same cycle
    always @(posedge clk) begin
        if (reset) begin
            unlock_valid_r <= 0;
        end else begin
            unlock_valid_r <= unlock_valid_n || (cunlock_mask != '0);
        end
        unlock_mask_r <= (unlock_valid_n ? unlock_mask_n : '0) | cunlock_mask;
    end

    // same timing as the store's read, with the same-cycle write visible
    assign read_phase   = read_cluster ? cphase_n[crd_slot] : phase_async;
`else
    reg unlock_valid_r;
    reg [`VX_CFG_NUM_WARPS-1:0] unlock_mask_r;

    always @(posedge clk) begin
        if (reset) begin
            unlock_valid_r <= 0;
        end else begin
            unlock_valid_r <= unlock_valid_n;
        end
        unlock_mask_r <= unlock_mask_n;
    end

    assign read_phase   = phase_async;
`endif
    assign unlock_valid = unlock_valid_r;
    assign unlock_mask  = unlock_mask_r;

    always @(posedge clk) begin
        if (reset) begin
            gbar_pending_r <= '0;
            gbar_pending_phase_r <= '0;
            gbar_waiters_r <= '0;
        end else begin
            gbar_pending_r <= gbar_pending_n;
            gbar_pending_phase_r <= gbar_pending_phase_n;
            gbar_waiters_r <= gbar_waiters_n;
        end
    end

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // CTA_DONE events and cluster-barrier arrivals share the global-barrier
    // request queue, so it exists even at one core and each core's events leave
    // in order. A barrier holds at most one entry per id, a slot at most one
    // cluster-barrier arrival, and a done event enters only below NUM_WARPS
    // entries, so the queue never overflows.
    if (1) begin : g_gbar

        localparam GQ_DATAW = 2 + GANG_ID_W + 1 + NW_WIDTH + GBAR_REQW;
        localparam GQ_DEPTH = 1 << `CLOG2((1 << NB_WIDTH) + 2 * `VX_CFG_NUM_WARPS);
        localparam GQ_SIZEW = `CLOG2(GQ_DEPTH + 1);

        wire done_push = cta_done_valid && cta_done_ready;
        wire req_push  = gbar_enqueue || cbar_push || done_push;

        wire [GQ_DATAW-1:0] req_queue_in = gbar_enqueue
            ? {GBAR_KIND_BAR, GANG_ID_W'(0), 1'b0, NW_WIDTH'(0), gbar_enqueue_id, gbar_enqueue_size_m1}
            : cbar_push
            ? {GBAR_KIND_CBAR, cinfo_s.gang_id, cinfo_s.epoch, creq_slot, NB_WIDTH'(0), NC_WIDTH'(cinfo_s.k - CK_W'(1))}
            : {GBAR_KIND_CTA_DONE, cta_done_data.gang_id, cta_done_data.epoch, cta_done_data.slot,
               NB_WIDTH'(0), NC_WIDTH'(0)};
        wire [GQ_DATAW-1:0] req_queue_data;
        wire [GQ_SIZEW-1:0] req_size;
        wire req_empty, req_alm_empty, req_full;
        wire req_pop = ~req_empty && gbar_bus_if.req_ready;

        VX_fifo_queue #(
            .DATAW  (GQ_DATAW),
            .DEPTH  (GQ_DEPTH),
            .LUTRAM (1)
        ) req_queue (
            .clk     (clk),
            .reset   (reset),
            .push    (req_push),
            .pop     (req_pop),
            .data_in (req_queue_in),
            .data_out(req_queue_data),
            .empty   (req_empty),
            .alm_empty(req_alm_empty),
            .full    (req_full),
            `UNUSED_PIN (alm_full),
            .size    (req_size)
        );

        // a barrier arrival takes the queue's write port first
        assign cta_done_ready = ~(gbar_enqueue || cbar_push) && (req_size < GQ_SIZEW'(`VX_CFG_NUM_WARPS));

        // Same registered valid as without DLMEM: one cycle after the push.
        reg gbar_req_valid_r;

        always @(posedge clk) begin
            if (reset) begin
                gbar_req_valid_r <= 0;
            end else begin
                gbar_req_valid_r <= req_push | (req_pop ? ~req_alm_empty : ~req_empty);
            end
        end

        assign gbar_bus_if.req_valid        = gbar_req_valid_r;
        assign {gbar_bus_if.req_data.kind, gbar_bus_if.req_data.gang_id, gbar_bus_if.req_data.epoch,
                gbar_bus_if.req_data.slot, gbar_bus_if.req_data.id, gbar_bus_if.req_data.size_m1} = req_queue_data;
        assign gbar_bus_if.req_data.core_id = NC_WIDTH'(CORE_ID % `VX_CFG_NUM_CORES);
        // at one core no global barrier ever answers
        assign gbar_bus_if.rsp_ready        = USE_GBAR ? gbar_rsp_ready : 1'b1;
        // the event's core is this one, sent as core_id
        `UNUSED_VAR (cta_done_data.core)

        `RUNTIME_ASSERT(~req_push || ~req_full, ("%s global barrier request queue overflow", INSTANCE_ID))
        `RUNTIME_ASSERT(~gbar_enqueue || ~gbar_pending_r[gbar_enqueue_id], ("%s duplicate global barrier generation: id=%0d", INSTANCE_ID, gbar_enqueue_id))
        `RUNTIME_ASSERT(~gbar_enqueue || (store_waddr == BAR_ADDR_W'(gbar_enqueue_id)), ("%s invalid global barrier slot: id=%0d, slot=%0d", INSTANCE_ID, gbar_enqueue_id, store_waddr))
    end
`else
    if (USE_GBAR) begin : g_gbar

        wire [GBAR_REQW-1:0] req_queue_data;
        wire req_empty, req_alm_empty;
        wire req_pop = ~req_empty && gbar_bus_if.req_ready;

        VX_fifo_queue #(
            .DATAW  (GBAR_REQW),
            .DEPTH  (1 << NB_WIDTH),
            .LUTRAM (1)
        ) req_queue (
            .clk     (clk),
            .reset   (reset),
            .push    (gbar_enqueue),
            .pop     (req_pop),
            .data_in ({gbar_enqueue_id, gbar_enqueue_size_m1}),
            .data_out(req_queue_data),
            .empty   (req_empty),
            .alm_empty(req_alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (full),
            `UNUSED_PIN (size)
        );

        // The queue's empty flag is already a register, but inverting it for the
        // outgoing valid leaves a combinational cell on a net that crosses the
        // core, socket and cluster boundaries. Track non-emptiness in its own
        // flop: same next state on the same edge, so the request still leaves
        // one cycle after the arrival that produced it.
        reg gbar_req_valid_r;

        always @(posedge clk) begin
            if (reset) begin
                gbar_req_valid_r <= 0;
            end else begin
                gbar_req_valid_r <= gbar_enqueue | (req_pop ? ~req_alm_empty : ~req_empty);
            end
        end

        assign gbar_bus_if.req_valid        = gbar_req_valid_r;
        assign {gbar_bus_if.req_data.id, gbar_bus_if.req_data.size_m1} = req_queue_data;
        assign gbar_bus_if.req_data.core_id = NC_WIDTH'(CORE_ID % `VX_CFG_NUM_CORES);
        assign gbar_bus_if.rsp_ready        = gbar_rsp_ready;

        `RUNTIME_ASSERT(~gbar_enqueue || ~gbar_pending_r[gbar_enqueue_id], ("%s duplicate global barrier generation: id=%0d", INSTANCE_ID, gbar_enqueue_id))
        `RUNTIME_ASSERT(~gbar_enqueue || (store_waddr == BAR_ADDR_W'(gbar_enqueue_id)), ("%s invalid global barrier slot: id=%0d, slot=%0d", INSTANCE_ID, gbar_enqueue_id, store_waddr))
    end else begin : g_nogbar

        assign gbar_bus_if.req_valid = 0;
        assign gbar_bus_if.req_data  = 'x;
        assign gbar_bus_if.rsp_ready = 0;

        `UNUSED_VAR (gbar_bus_if.req_ready)
        `UNUSED_VAR (gbar_enqueue_size_m1)

    end
`endif

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (req_valid) begin
            `TRACE(2, ("%t: %s req: wid=%0d, bar_id=%0d, is_global=%b, is_event=%b, is_arrive=%b, is_sync=%b, phase=%b, size_m1=%0d\n",
                $time, INSTANCE_ID, req_wid, req_data.id, req_data.is_global, req_data.is_event, req_data.is_arrive, req_data.is_sync, req_data.phase, req_data.size_m1))
        end
        if (USE_GBAR && gbar_bus_if.req_valid && gbar_bus_if.req_ready) begin
            `TRACE(2, ("%t: %s global-req: bar_id=%0d, size_m1=%0d\n",
                $time, INSTANCE_ID, gbar_bus_if.req_data.id, gbar_bus_if.req_data.size_m1))
        end
        if (USE_GBAR && gbar_bus_if.rsp_valid && gbar_rsp_ready) begin
            `TRACE(2, ("%t: %s global-rsp: bar_id=%0d\n",
                $time, INSTANCE_ID, gbar_bus_if.rsp_data.id))
        end
        if (unlock_valid_n) begin
            `TRACE(2, ("%t: %s unlock: mask=%b\n",
                $time, INSTANCE_ID, unlock_mask_n))
        end
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        if (creq) begin
            `TRACE(2, ("%t: %s cluster-req: wid=%0d, slot=%0d, is_arrive=%b, is_sync=%b, phase=%b, push=%b, local_release=%b\n",
                $time, INSTANCE_ID, req_wid, creq_slot, req_data.is_arrive, req_data.is_sync, req_data.phase, cbar_push, |crel_mask))
        end
        if (crsp) begin
            `TRACE(2, ("%t: %s cluster-rsp: gang=%0d, epoch=%0d, slots=%b\n",
                $time, INSTANCE_ID, gbar_bus_if.rsp_data.gang_id, gbar_bus_if.rsp_data.epoch, crel_mask))
        end
        if (cunlock_mask != '0) begin
            `TRACE(2, ("%t: %s cluster-unlock: mask=%b\n", $time, INSTANCE_ID, cunlock_mask))
        end
    `endif
    end
`endif

endmodule
