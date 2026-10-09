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

// Gang admission of one hardware cluster, between its launch input and its
// socket distributor.
//
// A DEFAULT beat passes through unchanged. A SPREAD message (one thread-block
// cluster of K blocks) is placed all or nothing on an exact mirror of every
// core's free CTA slots and reserved warps: one greedy step per cycle, spread
// first from a round-robin start core (sw/common/dlmem_ref.h place()). The K
// beats then leave as pinned messages, each routed by `dest` to its core with
// its forced slot and the gang's peer row. A slot returns to the mirror only
// when its block's CTA_DONE event arrives, so a forced slot is always free when
// its beat reaches the core.
//
// Timing: a head beat visible in cycle t is placed in cycles t+1..t+K; beat e
// is taken at the earliest in cycle t+K+1+e and leaves one cycle later.
module VX_cta_gang import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CLUSTER_ID = 0,
    parameter OUT_BUF    = 3
) (
    input wire              clk,
    input wire              reset,

    VX_kmu_bus_if.slave     bus_in_if,
    VX_kmu_bus_if.master    bus_out_if,

    // CTA completions of SPREAD blocks, always accepted
    input wire              done_valid,
    input dlmem_cta_done_t  done_data,

    output wire             busy
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam NCC      = DLMEM_NCC;
    localparam NW       = `VX_CFG_NUM_WARPS;
    localparam MAXC     = DLMEM_MAX_CLUSTER;
    localparam NG       = DLMEM_NUM_GANGS;
    localparam CORE_W   = DLMEM_CORE_W;
    localparam USED_W   = NW_WIDTH + 1;
    localparam STEP_W   = `UP(`CLOG2(MAXC));
    localparam LMEM_LOG = `VX_CFG_LMEM_LOG_SIZE;
    localparam PROD_W   = LMEM_LOG + NW_WIDTH + 2;
    localparam SS_W     = `CLOG2(`VX_CFG_SOCKET_SIZE);
    localparam NS_W     = `CLOG2(NUM_SOCKETS);
    // packed beat: {pin, kind, eop, dest, data}
    localparam PW       = 3 + KMU_DEST_W + KMU_DATAW;

    `STATIC_ASSERT ((((((CLUSTER_ID << NS_W) | (NUM_SOCKETS - 1)) << SS_W) | (`VX_CFG_SOCKET_SIZE - 1)) >> KMU_DEST_W) == 0,
        ("%s: the launch dest field cannot name every core of cluster %0d", INSTANCE_ID, CLUSTER_ID))

    typedef enum logic [1:0] {
        S_IDLE  = 0,
        S_PLACE = 1,
        S_EMIT  = 2
    } state_t;

    state_t                     state;

    // live mirror and gang-id pool
    reg [NCC-1:0][NW-1:0]       free_r;
    reg [NCC-1:0][USED_W-1:0]   used_r;
    reg [NG-1:0]                live_r;
    reg [NG-1:0][DLMEM_K_W-1:0] cnt_r;
    reg [NG-1:0]                epoch_r;
    reg [CORE_W-1:0]            rr_r;

    // per-launch latches: a launch's done events use its policy, K and W
    reg [7:0]                   ctx_r;
    reg                         policy_r;
    reg [CLUSTER_SIZE_W-1:0]    k_r;
    reg [USED_W-1:0]            w_r;
    reg [NW-1:0]                umask_r;

    // placement in progress: a scratch copy of the mirror taken at its start
    reg [NCC-1:0][NW-1:0]       sfree_r;
    reg [NCC-1:0][USED_W-1:0]   sused_r;
    reg [NCC-1:0][NW-1:0]       amask_r;
    reg [NCC-1:0][USED_W-1:0]   aused_r;
    reg [NCC-1:0]               visited_r;
    reg [CORE_W-1:0]            start_r;
    reg [STEP_W-1:0]            step_r;
    dlmem_peer_t [MAXC-1:0]     members_r;
    reg [GANG_ID_W-1:0]         cur_gang_r;
    reg                         cur_epoch_r;
    // A placement that failed is retried only after a block has finished.
    reg                         failed_r;
    reg                         dirty_r;

    // ---- head beat ---------------------------------------------------------

    kmu_req_t in_req;
    assign in_req = kmu_req_t'(bus_in_if.data);

    wire                      head_valid = bus_in_if.valid;
    wire                      in_spread  = (in_req.args.compute.dlmem.policy == KMU_POLICY_SPREAD);
    wire                      in_new     = (in_req.ctx_id != ctx_r);
    wire [CLUSTER_SIZE_W-1:0] in_k       = in_req.args.compute.cluster_size;
    wire [CTA_TID_WIDTH:0]    in_bs      = in_req.args.compute.block_size;
    wire [LMEM_LOG:0]         in_stride  = in_req.aligned_lmem_size;

    // warps per block: the dispatcher's ceiling division by NUM_THREADS
    wire [USED_W-1:0] in_w;
    if (NT_BITS > 0) begin : g_w
        assign in_w = USED_W'(in_bs[CTA_TID_WIDTH:NT_BITS]) + USED_W'(|in_bs[NT_BITS-1:0]);
    end else begin : g_w1
        assign in_w = USED_W'(in_bs);
    end

    // usable slots as a mask: the dispatcher's occupancy bound, slot 0 always
    wire [NW-1:0] in_umask;
    for (genvar i = 0; i < NW; ++i) begin : g_umask
        assign in_umask[i] = (in_stride == 0) || (i == 0)
                          || ((PROD_W'(i + 1) * PROD_W'(in_stride)) <= PROD_W'(1 << LMEM_LOG));
    end

    // ---- done events: applied before this cycle's decision ----------------

    wire done_spread = done_valid && (policy_r == KMU_POLICY_SPREAD);
    wire done_gang   = done_spread && (k_r > CLUSTER_SIZE_W'(1));

    logic [NCC-1:0][NW-1:0]       free_n;
    logic [NCC-1:0][USED_W-1:0]   used_n;
    logic [NG-1:0]                live_n;
    logic [NG-1:0][DLMEM_K_W-1:0] cnt_n;

    always @(*) begin
        free_n = free_r;
        used_n = used_r;
        live_n = live_r;
        cnt_n  = cnt_r;
        if (done_spread) begin
            free_n[done_data.core][done_data.slot] = 1'b1;
            used_n[done_data.core] = used_r[done_data.core] - w_r;
        end
        if (done_gang) begin
            cnt_n[done_data.gang_id] = cnt_r[done_data.gang_id] - DLMEM_K_W'(1);
            if (cnt_r[done_data.gang_id] == DLMEM_K_W'(1)) begin
                live_n[done_data.gang_id] = 1'b0;
            end
        end
    end

    // lowest free gang id
    wire id_free = ~(&live_n);
    logic [GANG_ID_W-1:0] free_id;
    always @(*) begin
        free_id = '0;
        for (integer i = NG - 1; i >= 0; --i) begin
            if (~live_n[i]) begin
                free_id = GANG_ID_W'(i);
            end
        end
    end

    // ---- one greedy placement step on the scratch copy ---------------------

    wire [NCC-1:0] capable;
    for (genvar c = 0; c < NCC; ++c) begin : g_capable
        assign capable[c] = (|(sfree_r[c] & umask_r))
                         && ((USED_W+1)'(sused_r[c]) + (USED_W+1)'(w_r) <= (USED_W+1)'(NW));
    end
    wire           any_capable = (|capable);
    wire [NCC-1:0] cand_new    = capable & ~visited_r;
    // every capable core already holds a member: start a new round
    wire           new_round   = ~(|cand_new);
    wire [NCC-1:0] cand        = new_round ? capable : cand_new;

    // the first candidate at or after the start core, cyclically
    logic [CORE_W-1:0] sel_core;
    always @(*) begin
        sel_core = '0;
        for (integer i = NCC - 1; i >= 0; --i) begin
            if (cand[i]) begin
                sel_core = CORE_W'(i);
            end
        end
        for (integer i = NCC - 1; i >= 0; --i) begin
            if (cand[i] && (CORE_W'(i) >= start_r)) begin
                sel_core = CORE_W'(i);
            end
        end
    end

    wire [NW-1:0] sel_avail = sfree_r[sel_core] & umask_r;
    logic [NW_WIDTH-1:0] sel_slot;
    always @(*) begin
        sel_slot = '0;
        for (integer s = NW - 1; s >= 0; --s) begin
            if (sel_avail[s]) begin
                sel_slot = NW_WIDTH'(s);
            end
        end
    end

    logic [NCC-1:0][NW-1:0]     amask_n;
    logic [NCC-1:0][USED_W-1:0] aused_n;
    always @(*) begin
        amask_n = amask_r;
        aused_n = aused_r;
        amask_n[sel_core][sel_slot] = 1'b1;
        aused_n[sel_core] = aused_r[sel_core] + w_r;
    end

    wire [CORE_W-1:0] rr_next = (sel_core == CORE_W'(NCC - 1)) ? CORE_W'(0) : (sel_core + CORE_W'(1));
    wire              is_last = ((CLUSTER_SIZE_W'(step_r) + CLUSTER_SIZE_W'(1)) == k_r);

    // ---- control -----------------------------------------------------------

    wire idle   = (state == S_IDLE);
    wire emit   = (state == S_EMIT);
    wire is_new = idle && head_valid && in_new;
    wire start  = idle && head_valid && in_spread
               && ((in_k == CLUSTER_SIZE_W'(1)) || id_free)
               && (~(failed_r && ~is_new) || dirty_r);

    // DEFAULT beats pass through; SPREAD beats leave only once placed.
    wire fwd          = (idle && ~in_spread) || emit;
    wire buf_valid_in = head_valid && fwd;
    wire buf_ready_in;
    assign bus_in_if.ready = buf_ready_in && fwd;
    wire in_fire = head_valid && bus_in_if.ready;

    // ---- output beat -------------------------------------------------------

    wire [NCC-1:0][KMU_DEST_W-1:0] dest_lut;
    for (genvar c = 0; c < NCC; ++c) begin : g_dest
        assign dest_lut[c] = KMU_DEST_W'((((CLUSTER_ID << NS_W) | (c / `VX_CFG_SOCKET_SIZE)) << SS_W)
                                         | (c % `VX_CFG_SOCKET_SIZE));
    end

    dlmem_peer_t emit_peer;
    assign emit_peer = members_r[step_r];

    kmu_req_t out_req;
    always @(*) begin
        out_req = in_req;
        if (emit) begin
            out_req.args.compute.dlmem.policy      = KMU_POLICY_SPREAD;
            out_req.args.compute.dlmem.epoch       = cur_epoch_r;
            out_req.args.compute.dlmem.gang_id     = cur_gang_r;
            out_req.args.compute.dlmem.forced_slot = emit_peer.slot;
            out_req.args.compute.dlmem.row.k       = DLMEM_K_W'(k_r);
            out_req.args.compute.dlmem.row.peers   = members_r;
        end
    end

    wire                  out_eop  = emit || bus_in_if.eop;
    wire [KMU_DEST_W-1:0] out_dest = emit ? dest_lut[emit_peer.core] : bus_in_if.dest;
    wire [PW-1:0]         buf_out;

    VX_elastic_buffer #(
        .DATAW   (PW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
        .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (buf_valid_in),
        .ready_in  (buf_ready_in),
        .data_in   ({emit, bus_in_if.kind, out_eop, out_dest, KMU_DATAW'(out_req)}),
        .data_out  (buf_out),
        .valid_out (bus_out_if.valid),
        .ready_out (bus_out_if.ready)
    );

    assign {bus_out_if.pin, bus_out_if.kind, bus_out_if.eop, bus_out_if.dest, bus_out_if.data} = buf_out;

    // ---- state -------------------------------------------------------------

    always @(posedge clk) begin
        if (reset) begin
            state       <= S_IDLE;
            free_r      <= '1;
            used_r      <= '0;
            live_r      <= '0;
            cnt_r       <= '0;
            epoch_r     <= '0;
            rr_r        <= '0;
            ctx_r       <= '0;
            policy_r    <= KMU_POLICY_DEFAULT;
            k_r         <= CLUSTER_SIZE_W'(1);
            w_r         <= '0;
            umask_r     <= '1;
            sfree_r     <= '0;
            sused_r     <= '0;
            amask_r     <= '0;
            aused_r     <= '0;
            visited_r   <= '0;
            start_r     <= '0;
            step_r      <= '0;
            members_r   <= '0;
            cur_gang_r  <= '0;
            cur_epoch_r <= 1'b0;
            failed_r    <= 1'b0;
            dirty_r     <= 1'b0;
        end else begin
            free_r  <= free_n;
            used_r  <= used_n;
            live_r  <= live_n;
            cnt_r   <= cnt_n;
            dirty_r <= ~start && (dirty_r || done_valid);

            case (state)
            S_IDLE: begin
                if (head_valid) begin
                    ctx_r    <= in_req.ctx_id;
                    policy_r <= in_spread;
                    k_r      <= in_k;
                    w_r      <= in_w;
                    umask_r  <= in_umask;
                    if (in_new) begin
                        // nothing is in flight at a launch boundary
                        rr_r     <= '0;
                        epoch_r  <= '0;
                        failed_r <= 1'b0;
                    end
                end
                if (start) begin
                    sfree_r   <= free_n;
                    sused_r   <= used_n;
                    amask_r   <= '0;
                    aused_r   <= '0;
                    visited_r <= '0;
                    start_r   <= in_new ? CORE_W'(0) : rr_r;
                    step_r    <= '0;
                    members_r <= '0;
                    state     <= S_PLACE;
                end
            end
            S_PLACE: begin
                if (~any_capable) begin
                    failed_r <= 1'b1;
                    state    <= S_IDLE;
                end else begin
                    members_r[step_r] <= '{core: sel_core, slot: sel_slot};
                    sfree_r[sel_core][sel_slot] <= 1'b0;
                    sused_r[sel_core] <= sused_r[sel_core] + w_r;
                    visited_r <= (new_round ? '0 : visited_r) | (NCC'(1) << sel_core);
                    amask_r   <= amask_n;
                    aused_r   <= aused_n;
                    if (is_last) begin
                        // commit: the slots were free in the snapshot, and the
                        // live mirror has only released since, so they are free
                        for (integer c = 0; c < NCC; ++c) begin
                            free_r[c] <= free_n[c] & ~amask_n[c];
                            used_r[c] <= used_n[c] + aused_n[c];
                        end
                        rr_r <= rr_next;
                        if (k_r > CLUSTER_SIZE_W'(1)) begin
                            live_r[free_id]  <= 1'b1;
                            cnt_r[free_id]   <= DLMEM_K_W'(k_r);
                            epoch_r[free_id] <= ~epoch_r[free_id];
                            cur_gang_r       <= free_id;
                            cur_epoch_r      <= ~epoch_r[free_id];
                        end else begin
                            cur_gang_r  <= '0;
                            cur_epoch_r <= 1'b0;
                        end
                        failed_r <= 1'b0;
                        step_r   <= '0;
                        state    <= S_EMIT;
                    end else begin
                        step_r <= step_r + STEP_W'(1);
                    end
                end
            end
            S_EMIT: begin
                if (in_fire) begin
                    if (is_last) begin
                        step_r <= '0;
                        state  <= S_IDLE;
                    end else begin
                        step_r <= step_r + STEP_W'(1);
                    end
                end
            end
            default:;
            endcase
        end
    end

    // The device stays busy until every placed block has reported done, so the
    // next launch starts on an empty mirror.
    assign busy = head_valid | bus_out_if.valid | ~idle | ~(&free_r) | (|live_r);

    // ---- checks (dlmem_ref's asserts) ---------------------------------------

    `RUNTIME_ASSERT (~done_valid || (policy_r == KMU_POLICY_SPREAD),
        ("%t: %s: CTA_DONE outside a SPREAD launch", $time, INSTANCE_ID))
    `RUNTIME_ASSERT (~done_spread || ((32'(done_data.core) < NCC) && ~free_r[done_data.core][done_data.slot]
                                      && (used_r[done_data.core] >= w_r)),
        ("%t: %s: CTA_DONE for core %0d slot %0d, which holds no placed block", $time, INSTANCE_ID,
         done_data.core, done_data.slot))
    `RUNTIME_ASSERT (~done_gang || ((32'(done_data.gang_id) < NG) && live_r[done_data.gang_id]
                                    && (epoch_r[done_data.gang_id] == done_data.epoch)
                                    && (cnt_r[done_data.gang_id] != 0)),
        ("%t: %s: CTA_DONE for gang %0d epoch %0d, which is not live", $time, INSTANCE_ID,
         done_data.gang_id, done_data.epoch))
    `RUNTIME_ASSERT (~is_new || ((&free_n) && ~(|live_n)),
        ("%t: %s: a new launch began with blocks of the last one still placed", $time, INSTANCE_ID))
    `RUNTIME_ASSERT (~(idle && head_valid && in_spread)
                  || (in_req.args.compute.is_first_of_cluster && (in_req.args.compute.cluster_rank == '0)
                      && (in_k != '0) && (32'(in_k) <= MAXC) && (32'(in_w) <= NW)),
        ("%t: %s: a SPREAD message must start at rank 0 and fit the cluster (K=%0d, W=%0d)", $time, INSTANCE_ID,
         in_k, in_w))
    `RUNTIME_ASSERT (~(emit && head_valid)
                  || (in_spread && ~in_new && (in_req.args.compute.cluster_rank == RANK_W'(step_r))
                      && (bus_in_if.eop == is_last)),
        ("%t: %s: SPREAD beat out of step: rank %0d, expected %0d", $time, INSTANCE_ID,
         in_req.args.compute.cluster_rank, step_r))

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (~reset && (state == S_PLACE) && ~any_capable) begin
            `TRACE(2, ("%t: %s place-fail: ctx=%0d, K=%0d, W=%0d, step=%0d\n", $time, INSTANCE_ID, ctx_r, k_r, w_r, step_r))
        end
        if (~reset && (state == S_PLACE) && any_capable && is_last) begin
            `TRACE(1, ("%t: %s place: ctx=%0d, K=%0d, W=%0d, gang=%0d, last={%0d,%0d}\n", $time, INSTANCE_ID,
                ctx_r, k_r, w_r, (k_r > CLUSTER_SIZE_W'(1)) ? free_id : '0, sel_core, sel_slot))
        end
    end
`endif

endmodule
