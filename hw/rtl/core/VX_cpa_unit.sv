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

// Gated on the knob so the file preprocesses to nothing with it off:
// VX_define.vh itself emits DPI imports.
`ifdef VX_CFG_EXT_MBAR_ENABLE

`include "VX_define.vh"

// cp.async completion for the warps of one LSU block.
//
// A copy uop takes a slot at its first packet; the slot keeps each lane's
// destination and src-size, lands the lane's global data when its response
// arrives (always accepted, so the shared LSU response path never waits on
// the local memory), and counts the lane writes not yet acknowledged. Per
// warp, the uops form an in-order FIFO of CPA_DEPTH entries; the head pops one
// completed entry per cycle. commit_group records the FIFO tail as a group
// boundary (CPA_GROUPS per warp); wait_group N releases its warp once at most
// N boundaries lie beyond the head. One cp.async.mbarrier.arrive per warp
// waits for the head to reach the tail it recorded, then sends COMPLETE_TX(1)
// (or ARRIVE(1) for .noinc) per lane to the lane's mbarrier. The head moves by
// one entry, so it meets that tail exactly; the arrive stays eligible from then.
//
// Per cycle: the head pops on the registered completion; waits are evaluated
// on the registered head and boundaries; acks and issue events update the
// counts; one request leaves on the local-memory port: one packet of the
// lowest eligible warp's arrive first, else the landed lanes of the lowest
// (slot, pid); a request not taken is presented again unchanged. An arrive
// goes first because a warp holds at most one, while its copies can keep the
// writer busy for as long as it issues them.
//
// A store ack at T makes its entry poppable at T+1; a satisfied wait at T
// pulses the unlock at T+1, one warp per cycle, lowest first. A warp is idle
// when its FIFO is empty, it has no arrive pending and no arrive of the unit
// is still in the local memory: only then may it exit.
//
// Write tags are slot ids, so the slots fit the LSU tag; the all-ones id
// marks arrive responses, which are dropped.

module VX_cpa_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    localparam NUM_LANES = `VX_CFG_NUM_LSU_LANES,
    localparam PID_BITS  = `CLOG2(`VX_CFG_NUM_THREADS / NUM_LANES),
    localparam PID_W     = `UP(PID_BITS),
    localparam SLOT_W    = `LOG2UP(`VX_CFG_LSU_PENDING_SIZE),
    localparam LMEM_LOG  = `VX_CFG_LMEM_LOG_SIZE
) (
    input  wire                             clk,
    input  wire                             reset,

    // The LSU's cp.async instruction presented this cycle; iss_ready is its stall.
    input  wire                             iss_valid,
    input  wire [CPA_OP_BITS-1:0]           iss_op,
    input  wire                             iss_noinc,
    input  wire [NW_WIDTH-1:0]              iss_wid,
    input  wire [PID_W-1:0]                 iss_pid,
    input  wire                             iss_sop,
    input  wire                             iss_eop,
    input  wire [NUM_LANES-1:0]             iss_tmask,
    input  wire                             iss_size8,
    input  wire [4:0]                       iss_n,
    // Per lane, a copy's destination or an arrive's mbarrier: a local-memory byte offset.
    input  wire [NUM_LANES-1:0][LMEM_LOG-1:0] iss_addr,
    input  wire [NUM_LANES-1:0][3:0]        iss_ss,
    output wire                             iss_ready,
    output wire [SLOT_W-1:0]                iss_slot,

    // Global read responses of copy uops.
    input  wire                             rsp_valid,
    input  wire [SLOT_W-1:0]                rsp_slot,
    input  wire [PID_W-1:0]                 rsp_pid,
    input  wire [NUM_LANES-1:0]             rsp_mask,
    input  wire [NUM_LANES-1:0][63:0]       rsp_data,
    input  wire [NUM_LANES-1:0][2:0]        rsp_align,

    VX_lsu_mem_if.master                    lmem_if,
    VX_sched_unlock_if.master               unlock_if,
    output wire [`VX_CFG_NUM_WARPS-1:0]     idle
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `STATIC_ASSERT (`VX_CFG_XLEN == 64, ("cp.async needs a 64-bit LSU word"))

    localparam NW       = `VX_CFG_NUM_WARPS;
    localparam NUM_PIDS = 1 << PID_BITS;
    localparam DEPTH    = CPA_DEPTH;
    localparam GROUPS   = CPA_GROUPS;
    localparam ENT_W    = `CLOG2(DEPTH);
    localparam GRP_W    = `CLOG2(GROUPS);
    // Sequence numbers: entries in flight stay within DEPTH, so the signed difference is exact.
    localparam SEQ_W    = ENT_W + 2;
    localparam CNT_W    = `CLOG2(`VX_CFG_NUM_THREADS + 1);
    localparam LCNT_W   = `CLOG2(NUM_LANES + 1);
    localparam TAG_ID_W = LSU_TAG_WIDTH - UUID_WIDTH;
    localparam SLOTS    = CPA_SLOTS;
    localparam LMEM_AW  = LMEM_LOG - 3;
    localparam [TAG_ID_W-1:0] ARRIVE_TAG = '1;

    `STATIC_ASSERT (SLOTS < (1 << TAG_ID_W), ("cp.async slots must leave the arrive tag free"))
    `STATIC_ASSERT (SLOTS <= (1 << SLOT_W), ("cp.async slots must fit the LSU packet slot field"))

    function automatic logic [LCNT_W-1:0] lane_count(input logic [NUM_LANES-1:0] m);
        lane_count = '0;
        for (integer j = 0; j < NUM_LANES; ++j) begin
            lane_count = lane_count + LCNT_W'(m[j]);
        end
    endfunction

    // ---------------------------------------------------------------------
    // State

    reg [SLOTS-1:0]              slot_valid_r, slot_closed_r, slot_size8_r;
    reg [NW_WIDTH-1:0]           slot_wid_r   [SLOTS];
    reg [ENT_W-1:0]              slot_ent_r   [SLOTS];
    reg [CNT_W-1:0]              slot_cnt_r   [SLOTS];

    reg [LMEM_LOG-1:0]           tbl_addr_r   [SLOTS][NUM_PIDS][NUM_LANES];
    reg [3:0]                    tbl_ss_r     [SLOTS][NUM_PIDS][NUM_LANES];
    reg [63:0]                   tbl_data_r   [SLOTS][NUM_PIDS][NUM_LANES];
    reg [NUM_LANES-1:0]          tbl_landed_r [SLOTS][NUM_PIDS];

    reg [SEQ_W-1:0]              head_r       [NW];
    reg [SEQ_W-1:0]              tail_r       [NW];
    reg [DEPTH-1:0]              done_r       [NW];
    reg [SLOT_W-1:0]             open_slot_r  [NW];

    reg [SEQ_W-1:0]              bnd_r        [NW][GROUPS];
    reg [GRP_W-1:0]              bhead_r      [NW];
    reg [GRP_W:0]                bsize_r      [NW];

    reg [NW-1:0]                 waiting_r, ready_r;
    reg [4:0]                    wait_n_r     [NW];

    reg [NW-1:0]                 arr_valid_r, arr_closed_r, arr_noinc_r, arr_rdy_r;
    reg [SEQ_W-1:0]              arr_tail_r   [NW];
    reg [NUM_PIDS-1:0][NUM_LANES-1:0] arr_mask_r [NW];
    reg [LMEM_AW-1:0]            arr_addr_r   [NW][NUM_PIDS][NUM_LANES];

    reg                          unlock_valid_r;
    reg [NW_WIDTH-1:0]           unlock_wid_r;

    // Arrive lanes presented to the port and not yet answered by the local memory.
    localparam AFL_W = `CLOG2(NW * `VX_CFG_NUM_THREADS + 1) + 1;
    reg [AFL_W-1:0]              arr_inflight_r;

    // ---------------------------------------------------------------------
    // Registered views

    wire [NW-1:0]      fifo_full, bnd_full, head_pop, arr_elig;
    wire [GRP_W:0]     pend_cnt [NW];

    for (genvar w = 0; w < NW; ++w) begin : g_warp
        wire [SEQ_W-1:0] fill = tail_r[w] - head_r[w];
        assign fifo_full[w] = (fill == SEQ_W'(DEPTH));
        assign head_pop[w]  = (fill != '0) && done_r[w][head_r[w][ENT_W-1:0]];

        // Boundaries are monotonic, so the pending ones are a suffix of those held.
        reg [GRP_W:0] cnt;
        always @(*) begin
            cnt = '0;
            for (integer j = 0; j < GROUPS; ++j) begin
                logic [SEQ_W-1:0] d;
                d = bnd_r[w][GRP_W'(bhead_r[w] + GRP_W'(j))] - head_r[w];
                if (((GRP_W+1)'(j) < bsize_r[w]) && (d != '0) && ~d[SEQ_W-1]) begin
                    cnt = cnt + (GRP_W+1)'(1);
                end
            end
        end
        assign pend_cnt[w] = cnt;
        assign bnd_full[w] = (cnt == (GRP_W+1)'(GROUPS));

        assign arr_elig[w] = arr_valid_r[w] && arr_closed_r[w] && arr_rdy_r[w];
        assign idle[w] = (fill == '0) && ~arr_valid_r[w] && (arr_inflight_r == '0);
    end

    wire [SLOTS-1:0] slot_free = ~slot_valid_r;
    wire             slot_full = ~(| slot_free);
    wire [`LOG2UP(SLOTS)-1:0] alloc_idx;
    VX_priority_encoder #(
        .N (SLOTS)
    ) alloc_enc (
        .data_in    (slot_free),
        .index_out  (alloc_idx),
        `UNUSED_PIN (onehot_out),
        `UNUSED_PIN (valid_out)
    );
    wire [SLOT_W-1:0] alloc_slot = SLOT_W'(alloc_idx);

    // ---------------------------------------------------------------------
    // Issue

    wire is_copy   = (iss_op == CPA_OP_COPY);
    wire is_commit = (iss_op == CPA_OP_COMMIT);
    wire is_wait   = (iss_op == CPA_OP_WAIT);
    wire is_arrive = (iss_op == CPA_OP_ARRIVE);

    assign iss_ready = ~(is_copy && iss_sop && (fifo_full[iss_wid] || slot_full))
                    && ~(is_commit && bnd_full[iss_wid])
                    && ~(is_arrive && iss_sop && arr_valid_r[iss_wid]);
    assign iss_slot = iss_sop ? alloc_slot : open_slot_r[iss_wid];

    wire iss_fire   = iss_valid && iss_ready;
    wire copy_fire  = iss_fire && is_copy;
    wire [SLOT_W-1:0] cur_slot = iss_slot;
    wire [LCNT_W-1:0] iss_lanes = lane_count(iss_tmask);

    // ---------------------------------------------------------------------
    // Local-memory requests: landed data first, then arrives

    wire [SLOTS*NUM_PIDS-1:0] landed_any;
    for (genvar s = 0; s < SLOTS; ++s) begin : g_landed_s
        for (genvar p = 0; p < NUM_PIDS; ++p) begin : g_landed_p
            assign landed_any[s * NUM_PIDS + p] = (| tbl_landed_r[s][p]);
        end
    end
    localparam SP_W = `LOG2UP(SLOTS * NUM_PIDS);
    wire [SP_W-1:0] f_wr_sp;
    wire            f_wr_valid;
    VX_priority_encoder #(
        .N (SLOTS * NUM_PIDS)
    ) wr_enc (
        .data_in    (landed_any),
        .index_out  (f_wr_sp),
        `UNUSED_PIN (onehot_out),
        .valid_out  (f_wr_valid)
    );
    wire [SLOT_W-1:0] f_wr_slot = SLOT_W'(f_wr_sp >> PID_BITS);
    wire [PID_W-1:0]  f_wr_pid  = PID_W'(f_wr_sp & SP_W'(NUM_PIDS - 1));

    wire [NW_WIDTH-1:0] f_ar_wid;
    wire                f_ar_valid;
    VX_priority_encoder #(
        .N (NW)
    ) ar_enc (
        .data_in    (arr_elig),
        .index_out  (f_ar_wid),
        `UNUSED_PIN (onehot_out),
        .valid_out  (f_ar_valid)
    );
    wire [NUM_PIDS-1:0] ar_pids;
    for (genvar p = 0; p < NUM_PIDS; ++p) begin : g_ar_pids
        assign ar_pids[p] = (| arr_mask_r[f_ar_wid][p]);
    end
    wire [PID_W-1:0] f_ar_pid;
    VX_priority_encoder #(
        .N (NUM_PIDS)
    ) ar_pid_enc (
        .data_in    (ar_pids),
        .index_out  (f_ar_pid),
        `UNUSED_PIN (onehot_out),
        `UNUSED_PIN (valid_out)
    );

    // The adapter takes a packet's lanes as their outputs free up, so a request
    // stays as presented until it is taken.
    reg                 hold_r, hold_arr_r;
    reg [SLOT_W-1:0]    hold_slot_r;
    reg [PID_W-1:0]     hold_pid_r;
    reg [NW_WIDTH-1:0]  hold_wid_r;
    reg [NUM_LANES-1:0] hold_mask_r;

    wire send_arrive = hold_r ? hold_arr_r : f_ar_valid;
    wire [SLOT_W-1:0]   wr_slot = hold_r ? hold_slot_r : f_wr_slot;
    wire [PID_W-1:0]    wr_pid  = hold_r ? hold_pid_r : f_wr_pid;
    wire [NW_WIDTH-1:0] ar_wid  = hold_r ? hold_wid_r : f_ar_wid;
    wire [PID_W-1:0]    ar_pid  = hold_r ? hold_pid_r : f_ar_pid;
    wire [NUM_LANES-1:0] wr_mask = hold_r ? hold_mask_r : tbl_landed_r[f_wr_slot][f_wr_pid];

    mem_bus_attr_t arr_attr;
    always @(*) begin
        arr_attr = '0;
        arr_attr.amo.amo_valid = 1'b1;
        arr_attr.amo.amo_op    = arr_noinc_r[ar_wid] ? AMO_OP_MBAR_ARRIVE : AMO_OP_MBAR_COMPLETE_TX;
    end

    wire wr_size8 = slot_size8_r[wr_slot];
    for (genvar j = 0; j < NUM_LANES; ++j) begin : g_req
        wire [LMEM_LOG-1:0] dst = tbl_addr_r[wr_slot][wr_pid][j];
        wire [63:0] wdata = (~wr_size8 && dst[2]) ? {tbl_data_r[wr_slot][wr_pid][j][31:0], 32'h0}
                                                  : tbl_data_r[wr_slot][wr_pid][j];
        wire [7:0] wbe = wr_size8 ? 8'hff : (dst[2] ? 8'hf0 : 8'h0f);
        `UNUSED_VAR (dst[1:0])
        assign lmem_if.req_data.addr[j]   = send_arrive ? LSU_ADDR_WIDTH'(arr_addr_r[ar_wid][ar_pid][j])
                                                        : LSU_ADDR_WIDTH'(dst[LMEM_LOG-1:3]);
        assign lmem_if.req_data.data[j]   = send_arrive ? 64'd1 : wdata;
        assign lmem_if.req_data.byteen[j] = send_arrive ? 8'hff : wbe;
        assign lmem_if.req_data.user[j]   = send_arrive ? MEM_ATTR_WIDTH'(arr_attr) : '0;
    end
    assign lmem_if.req_valid          = hold_r || f_wr_valid || f_ar_valid;
    assign lmem_if.req_data.rw        = ~send_arrive;
    assign lmem_if.req_data.mask      = send_arrive ? arr_mask_r[ar_wid][ar_pid] : wr_mask;
    assign lmem_if.req_data.tag.value = send_arrive ? ARRIVE_TAG : TAG_ID_W'(wr_slot);
    assign lmem_if.req_data.tag.uuid  = '0;

    wire req_fire = lmem_if.req_valid && lmem_if.req_ready;
    wire wr_fire  = req_fire && ~send_arrive;
    wire ar_fire  = req_fire && send_arrive;
    wire ar_present = lmem_if.req_valid && send_arrive && ~hold_r;

    // Store acks; arrive responses are dropped.
    assign lmem_if.rsp_ready = 1'b1;
    wire ack_valid = lmem_if.rsp_valid && (lmem_if.rsp_data.tag.value != ARRIVE_TAG);
    wire arr_ack   = lmem_if.rsp_valid && (lmem_if.rsp_data.tag.value == ARRIVE_TAG);
    wire [SLOT_W-1:0] ack_slot = SLOT_W'(lmem_if.rsp_data.tag.value);
    wire [LCNT_W-1:0] ack_lanes = lane_count(lmem_if.rsp_data.mask);
    `UNUSED_VAR (lmem_if.rsp_data.data)
    `UNUSED_VAR (lmem_if.rsp_data.tag.uuid)

    // ---------------------------------------------------------------------
    // Slot counts and completions

    reg [SLOTS-1:0]  slot_valid_n, slot_closed_n, slot_done;
    reg [CNT_W-1:0]  slot_cnt_n [SLOTS];
    always @(*) begin
        for (integer s = 0; s < SLOTS; ++s) begin
            logic is_iss, is_ack;
            is_iss = copy_fire && (cur_slot == SLOT_W'(s));
            is_ack = ack_valid && (ack_slot == SLOT_W'(s));
            slot_valid_n[s]  = slot_valid_r[s] || (is_iss && iss_sop);
            slot_closed_n[s] = (slot_closed_r[s] && ~(is_iss && iss_sop)) || (is_iss && iss_eop);
            slot_cnt_n[s]    = ((is_iss && iss_sop) ? '0 : slot_cnt_r[s])
                             + (is_iss ? CNT_W'(iss_lanes) : '0)
                             - (is_ack ? CNT_W'(ack_lanes) : '0);
            slot_done[s]     = slot_valid_n[s] && slot_closed_n[s] && (slot_cnt_n[s] == '0);
        end
    end

    // ---------------------------------------------------------------------
    // Waits and unlocks

    reg [NW-1:0] ready_n, pick, waiting_n;
    always @(*) begin
        ready_n   = ready_r;
        waiting_n = waiting_r;
        for (integer w = 0; w < NW; ++w) begin
            if (waiting_r[w] && (pend_cnt[w] <= (GRP_W+1)'(wait_n_r[w]))) begin
                ready_n[w]   = 1'b1;
                waiting_n[w] = 1'b0;
            end
        end
        // N >= CPA_GROUPS is always satisfied.
        if (iss_fire && is_wait && iss_eop) begin
            if ((iss_n >= 5'(GROUPS)) || (pend_cnt[iss_wid] <= (GRP_W+1)'(iss_n))) begin
                ready_n[iss_wid] = 1'b1;
            end else begin
                waiting_n[iss_wid] = 1'b1;
            end
        end
        pick = '0;
        for (integer w = NW-1; w >= 0; --w) begin
            if (ready_n[w]) pick = NW'(1) << w;
        end
    end

    wire [NW_WIDTH-1:0] pick_wid;
    VX_onehot_encoder #(
        .N (NW)
    ) pick_enc (
        .data_in  (pick),
        .data_out (pick_wid),
        `UNUSED_PIN (valid_out)
    );

    // ---------------------------------------------------------------------
    // Registers

    always @(posedge clk) begin
        if (reset) begin
            slot_valid_r   <= '0;
            slot_closed_r  <= '0;
            waiting_r      <= '0;
            ready_r        <= '0;
            arr_valid_r    <= '0;
            arr_closed_r   <= '0;
            arr_rdy_r      <= '0;
            arr_inflight_r <= '0;
            unlock_valid_r <= 1'b0;
            hold_r         <= 1'b0;
            for (integer w = 0; w < NW; ++w) begin
                head_r[w]     <= '0;
                tail_r[w]     <= '0;
                done_r[w]     <= '0;
                bhead_r[w]    <= '0;
                bsize_r[w]    <= '0;
                arr_mask_r[w] <= '0;
            end
            for (integer s = 0; s < SLOTS; ++s) begin
                for (integer p = 0; p < NUM_PIDS; ++p) begin
                    tbl_landed_r[s][p] <= '0;
                end
            end
        end else begin
            // the head pops on the registered completion; drop the boundaries it passed
            for (integer w = 0; w < NW; ++w) begin
                if (head_pop[w]) begin
                    head_r[w] <= head_r[w] + SEQ_W'(1);
                    done_r[w][head_r[w][ENT_W-1:0]] <= 1'b0;
                end
                bhead_r[w] <= bhead_r[w] + GRP_W'(bsize_r[w] - pend_cnt[w]);
                bsize_r[w] <= pend_cnt[w];
            end

            // slots
            for (integer s = 0; s < SLOTS; ++s) begin
                slot_valid_r[s]  <= slot_valid_n[s] && ~slot_done[s];
                slot_closed_r[s] <= slot_closed_n[s];
                slot_cnt_r[s]    <= slot_cnt_n[s];
                if (slot_done[s]) begin
                    done_r[slot_wid_r[s]][slot_ent_r[s]] <= 1'b1;
                end
            end
            if (copy_fire && iss_sop) begin
                tail_r[iss_wid]      <= tail_r[iss_wid] + SEQ_W'(1);
                open_slot_r[iss_wid] <= alloc_slot;
            end

            // landed lanes
            for (integer s = 0; s < SLOTS; ++s) begin
                for (integer p = 0; p < NUM_PIDS; ++p) begin
                    logic [NUM_LANES-1:0] l;
                    l = tbl_landed_r[s][p];
                    if (wr_fire && (wr_slot == SLOT_W'(s)) && (wr_pid == PID_W'(p))) begin
                        l = l & ~wr_mask;
                    end
                    if (rsp_valid && (rsp_slot == SLOT_W'(s)) && (rsp_pid == PID_W'(p))) begin
                        l = l | rsp_mask;
                    end
                    tbl_landed_r[s][p] <= l;
                end
            end

            // commit pushes the tail as a boundary
            if (iss_fire && is_commit && iss_eop) begin
                bsize_r[iss_wid] <= pend_cnt[iss_wid] + (GRP_W+1)'(1);
            end

            hold_r <= lmem_if.req_valid && ~lmem_if.req_ready;

            // waits
            waiting_r      <= waiting_n;
            ready_r        <= ready_n & ~pick;
            unlock_valid_r <= (pick != '0);

            // arrives
            for (integer w = 0; w < NW; ++w) begin
                if (arr_valid_r[w] && head_pop[w] && ((head_r[w] + SEQ_W'(1)) == arr_tail_r[w])) begin
                    arr_rdy_r[w] <= 1'b1;
                end
            end
            if (iss_fire && is_arrive) begin
                if (iss_sop) begin
                    arr_valid_r[iss_wid] <= 1'b1;
                    arr_rdy_r[iss_wid]   <= (tail_r[iss_wid] == (head_r[iss_wid] + SEQ_W'(head_pop[iss_wid])));
                end
                arr_closed_r[iss_wid] <= iss_eop;
            end
            // Counted when first presented: the adapter can send a lane, and
            // its answer come back, before it takes the packet's last lane.
            arr_inflight_r <= arr_inflight_r
                            + (ar_present ? AFL_W'(lane_count(lmem_if.req_data.mask)) : '0)
                            - (arr_ack ? AFL_W'(ack_lanes) : '0);
            if (ar_fire) begin
                if (arr_mask_r[ar_wid] == ((NUM_PIDS*NUM_LANES)'(arr_mask_r[ar_wid][ar_pid]) << (ar_pid * NUM_LANES))) begin
                    arr_valid_r[ar_wid]  <= 1'b0;
                    arr_closed_r[ar_wid] <= 1'b0;
                end
            end
            for (integer w = 0; w < NW; ++w) begin
                logic [NUM_PIDS-1:0][NUM_LANES-1:0] m;
                m = arr_mask_r[w];
                if (ar_fire && (ar_wid == NW_WIDTH'(w))) begin
                    m[ar_pid] = '0;
                end
                if (iss_fire && is_arrive && (iss_wid == NW_WIDTH'(w))) begin
                    if (iss_sop) begin
                        m = '0;
                    end
                    m[iss_pid] = iss_tmask;
                end
                arr_mask_r[w] <= m;
            end
        end

        if (copy_fire && iss_sop) begin
            slot_wid_r[cur_slot]   <= iss_wid;
            slot_ent_r[cur_slot]   <= tail_r[iss_wid][ENT_W-1:0];
            slot_size8_r[cur_slot] <= iss_size8;
        end
        if (copy_fire) begin
            for (integer j = 0; j < NUM_LANES; ++j) begin
                tbl_addr_r[cur_slot][iss_pid][j] <= iss_addr[j];
                tbl_ss_r[cur_slot][iss_pid][j]   <= iss_ss[j];
            end
        end
        // A lane's bytes at and above its src-size are written as zeros.
        if (rsp_valid) begin
            for (integer j = 0; j < NUM_LANES; ++j) begin
                if (rsp_mask[j]) begin
                    logic [63:0] v, keep;
                    logic [3:0]  ss;
                    v    = rsp_data[j] >> {rsp_align[j], 3'b000};
                    ss   = tbl_ss_r[rsp_slot][rsp_pid][j];
                    keep = (ss >= 4'd8) ? '1 : ~(64'(-1) << {ss[2:0], 3'b000});
                    tbl_data_r[rsp_slot][rsp_pid][j] <= v & keep;
                end
            end
        end
        if (iss_fire && is_commit && iss_eop) begin
            // after the drop: at bhead + (bsize - pending) + pending
            bnd_r[iss_wid][GRP_W'(bhead_r[iss_wid] + GRP_W'(bsize_r[iss_wid]))] <= tail_r[iss_wid];
        end
        if (iss_fire && is_wait && iss_eop) begin
            wait_n_r[iss_wid] <= iss_n;
        end
        if (iss_fire && is_arrive) begin
            if (iss_sop) begin
                arr_tail_r[iss_wid]  <= tail_r[iss_wid];
                arr_noinc_r[iss_wid] <= iss_noinc;
            end
            for (integer j = 0; j < NUM_LANES; ++j) begin
                arr_addr_r[iss_wid][iss_pid][j] <= iss_addr[j][LMEM_LOG-1:3];
            end
        end
        unlock_wid_r <= pick_wid;
        hold_arr_r   <= send_arrive;
        hold_slot_r  <= wr_slot;
        hold_pid_r   <= send_arrive ? ar_pid : wr_pid;
        hold_wid_r   <= ar_wid;
        hold_mask_r  <= wr_mask;
    end

    assign unlock_if.valid = unlock_valid_r;
    assign unlock_if.wid   = unlock_wid_r;

    // ---------------------------------------------------------------------
    // Checks

    `RUNTIME_ASSERT (~(ack_valid && ~slot_valid_r[ack_slot]),
        ("%t: *** %s cp.async store ack for a free slot: slot=%0d", $time, INSTANCE_ID, ack_slot))
    `RUNTIME_ASSERT (~(ack_valid && (CNT_W'(ack_lanes) > slot_cnt_r[ack_slot])),
        ("%t: *** %s cp.async store ack underflow: slot=%0d", $time, INSTANCE_ID, ack_slot))
    `RUNTIME_ASSERT (~(rsp_valid && ~slot_valid_r[rsp_slot]),
        ("%t: *** %s cp.async response for a free slot: slot=%0d", $time, INSTANCE_ID, rsp_slot))
    `RUNTIME_ASSERT (~(rsp_valid && (| (tbl_landed_r[rsp_slot][rsp_pid] & rsp_mask))),
        ("%t: *** %s cp.async response for a lane already landed: slot=%0d pid=%0d", $time, INSTANCE_ID, rsp_slot, rsp_pid))
    `RUNTIME_ASSERT (~(copy_fire && ~iss_sop && ~slot_valid_r[cur_slot]),
        ("%t: *** %s cp.async packet without an open slot: wid=%0d", $time, INSTANCE_ID, iss_wid))
    `RUNTIME_ASSERT (~(arr_ack && (AFL_W'(ack_lanes) > arr_inflight_r)),
        ("%t: *** %s cp.async arrive response underflow", $time, INSTANCE_ID))
    `RUNTIME_ASSERT (~(iss_fire && is_wait && iss_eop && (waiting_r[iss_wid] || ready_r[iss_wid])),
        ("%t: *** %s second cp.async.wait_group of a warp: wid=%0d", $time, INSTANCE_ID, iss_wid))

`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (~reset && iss_fire) begin
            `TRACE(2, ("%t: %s cpa-issue: wid=%0d, op=%0d, pid=%0d, sop=%b, eop=%b, tmask=%b, slot=%0d, n=%0d\n",
                $time, INSTANCE_ID, iss_wid, iss_op, iss_pid, iss_sop, iss_eop, iss_tmask, cur_slot, iss_n))
        end
        if (~reset && rsp_valid) begin
            `TRACE(2, ("%t: %s cpa-land: slot=%0d, pid=%0d, mask=%b\n", $time, INSTANCE_ID, rsp_slot, rsp_pid, rsp_mask))
        end
        if (~reset && wr_fire) begin
            `TRACE(2, ("%t: %s cpa-write: slot=%0d, pid=%0d, mask=%b\n", $time, INSTANCE_ID, wr_slot, wr_pid, lmem_if.req_data.mask))
        end
        if (~reset && ar_fire) begin
            `TRACE(2, ("%t: %s cpa-arrive: wid=%0d, pid=%0d, mask=%b, noinc=%b\n", $time, INSTANCE_ID, ar_wid, ar_pid,
                lmem_if.req_data.mask, arr_noinc_r[ar_wid]))
        end
        if (~reset && unlock_valid_r) begin
            `TRACE(2, ("%t: %s cpa-unlock: wid=%0d\n", $time, INSTANCE_ID, unlock_wid_r))
        end
    end
`endif

endmodule

`endif
