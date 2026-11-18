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

// Cluster admission of the chip, between the KMU and the per-GPC launch lanes.
//
// One in-order queue (the KMU's stream) and one exact mirror per GPC of every
// SM's free CTA slots and reserved warps. A plain beat passes through. A
// cluster message (SPREAD policy, or an explicit-cluster launch with distinct
// SMs) is placed all or nothing on one GPC: every GPC runs the in-GPC policy
// one greedy step per cycle in parallel (SPREAD: round-robin from the core
// after its last placement; H100: least resident CTAs, then the lowest in-GPC
// index), and the first GPC from the round-robin pointer that placed all K
// members wins; the pointer moves past it and restarts at CLUSTER_GPC_START
// with every launch (sw/common/dlmem_ref.h place_chip). A head no GPC can
// place waits until a block finishes, and nothing behind it issues. The K
// beats then leave as pinned messages routed by their global SM index, with
// the forced slot and the gang's peer row; a gang of K >= 2 takes the lowest
// id its GPC's gang table has free, which the table learns from `alloc`.
//
// Timing: a head beat visible in cycle t is placed in cycles t+1..t+K; beat e
// is taken at the earliest in cycle t+K+1+e and leaves one cycle later.
module VX_cta_admit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_GPCS = 1,
    parameter OUT_BUF  = 3
) (
    input wire              clk,
    input wire              reset,

    VX_kmu_bus_if.slave     bus_in_if,
    VX_kmu_bus_if.master    bus_out_if,

    // per GPC: CTA_DONE events of placed blocks, after the CTA-done link
    input wire [NUM_GPCS-1:0]                       done_valid,
    input dlmem_cta_done_t [NUM_GPCS-1:0]           done_data,

    // per GPC: the gang table's pending cluster-barrier releases
    input wire [NUM_GPCS-1:0][DLMEM_NUM_GANGS-1:0]  rel_pend,

    // per GPC: a gang id taken, one cycle after the placement
    output wire [NUM_GPCS-1:0]                      alloc_valid,
    output wire [GANG_ID_W-1:0]                     alloc_gang,
    output wire                                     alloc_epoch,
    output wire [DLMEM_K_W-1:0]                     alloc_k,

    // the current launch's placed blocks belong to gangs (K >= 2)
    output wire                                     gang_mode,

    output wire             busy
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam G        = NUM_GPCS;
    localparam G_W      = `UP(`CLOG2(G));
    localparam NCC      = DLMEM_NCC;
    localparam NW       = `VX_CFG_NUM_WARPS;
    localparam NS       = NUM_CTA_MAX;
    localparam MAXC     = DLMEM_MAX_CLUSTER;
    localparam NG       = DLMEM_NUM_GANGS;
    localparam CORE_W   = DLMEM_CORE_W;
    localparam USED_W   = NW_WIDTH + 1;
    localparam RES_W    = `CLOG2(NS + 1);
    localparam STEP_W   = `UP(`CLOG2(MAXC));
    localparam LMEM_LOG = `VX_CFG_LMEM_LOG_SIZE;
    localparam PROD_W   = LMEM_LOG + NW_WIDTH + 2;
    localparam SS_W     = `CLOG2(`VX_CFG_SOCKET_SIZE);
    localparam NS_W     = `CLOG2(NUM_SOCKETS);
    localparam CAP      = `VX_CFG_MAX_CTAS_PER_SM_CLUSTER;
    localparam H100     = `VX_CFG_CLUSTER_DISTINCT_SMS_ENABLED;
`ifdef VX_CFG_CLUSTER_PLACEMENT_H100
    localparam POLICY_H100 = 1;
`else
    localparam POLICY_H100 = 0;
`endif
    // packed beat: {pin, kind, eop, dest, data}
    localparam PW       = 3 + KMU_DEST_W + KMU_DATAW;

    `STATIC_ASSERT (((((((G - 1) << NS_W) | (NUM_SOCKETS - 1)) << SS_W) | (`VX_CFG_SOCKET_SIZE - 1)) >> KMU_DEST_W) == 0,
        ("%s: the launch dest field cannot name every SM", INSTANCE_ID))
    `STATIC_ASSERT (NS <= NW, ("%s: more CTA slots than warps", INSTANCE_ID))
    `STATIC_ASSERT ((`VX_CFG_CLUSTER_GPC_START >= 0) && (`VX_CFG_CLUSTER_GPC_START < G),
        ("%s: CLUSTER_GPC_START must name a GPC", INSTANCE_ID))

    typedef enum logic [1:0] {
        S_IDLE  = 0,
        S_PLACE = 1,
        S_EMIT  = 2
    } state_t;

    state_t                         state;

    // live mirrors and gang-id pools
    reg [G-1:0][NCC-1:0][NS-1:0]    free_r;
    reg [G-1:0][NCC-1:0][USED_W-1:0] used_r;
    reg [G-1:0][NG-1:0]             live_r;
    reg [G-1:0][NG-1:0][DLMEM_K_W-1:0] cnt_r;
    reg [G-1:0][NG-1:0]             epoch_r;
    reg [G-1:0][CORE_W-1:0]         rr_r;
    reg [G_W-1:0]                   gptr_r;

    // per-launch latches: a launch's done events use its K and W
    reg [7:0]                       ctx_r;
    reg                             placed_r;
    reg [CLUSTER_SIZE_W-1:0]        k_r;
    reg [USED_W-1:0]                w_r;
    reg [NS-1:0]                    umask_r;

    // placement in progress: a scratch copy of every mirror taken at its start
    reg [G-1:0][NCC-1:0][NS-1:0]    sfree_r;
    reg [G-1:0][NCC-1:0][USED_W-1:0] sused_r;
    reg [G-1:0][NCC-1:0]            visited_r;
    reg [G-1:0][NCC-1:0][NS-1:0]    amask_r;    // slots this placement took
    reg [G-1:0][NCC-1:0][USED_W-1:0] aused_r;   // warps this placement took
    reg [G-1:0]                     out_r;      // GPC dropped out of this placement
    reg [G-1:0][CORE_W-1:0]         start_r;
    reg [STEP_W-1:0]                step_r;
    dlmem_peer_t [G-1:0][MAXC-1:0]  members_r;
    reg [G_W-1:0]                   win_r;
    reg [GANG_ID_W-1:0]             cur_gang_r;
    reg                             cur_epoch_r;
    // A placement that failed is retried only after a block has finished.
    reg                             failed_r;
    reg                             dirty_r;

    reg [G-1:0]                     alloc_v_r;
    reg [GANG_ID_W-1:0]             alloc_g_r;
    reg                             alloc_e_r;
    reg [DLMEM_K_W-1:0]             alloc_k_r;

    // ---- head beat ---------------------------------------------------------

    kmu_req_t in_req;
    assign in_req = kmu_req_t'(bus_in_if.data);

    wire                      head_valid = bus_in_if.valid;
    wire                      in_placed  = (in_req.args.compute.dlmem.policy == KMU_POLICY_SPREAD)
                                        || ((H100 != 0) && in_req.args.compute.dlmem.explicit_cluster);
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
    wire [NS-1:0] in_umask;
`ifdef VX_CFG_H100_SM_ENABLE
    // Each slot is charged its pitch against the launch's carve-out S.
    wire [PROD_W-1:0] in_fp = PROD_W'(in_stride);
    for (genvar i = 0; i < NS; ++i) begin : g_umask
        assign in_umask[i] = (in_fp == 0) || (i == 0)
                          || ((PROD_W'(i + 1) * in_fp) <= PROD_W'(in_req.smem_bytes));
    end
`else
    for (genvar i = 0; i < NS; ++i) begin : g_umask
        assign in_umask[i] = (in_stride == 0) || (i == 0)
                          || ((PROD_W'(i + 1) * PROD_W'(in_stride)) <= PROD_W'(1 << LMEM_LOG));
    end
`endif

    // ---- events: applied before this cycle's decision ---------------------

    wire [G-1:0] done_placed;
    wire [G-1:0] done_gang;
    logic [G-1:0][NCC-1:0][NS-1:0]       free_n;
    logic [G-1:0][NCC-1:0][USED_W-1:0]   used_n;
    logic [G-1:0][NG-1:0]                live_n;
    logic [G-1:0][NG-1:0][DLMEM_K_W-1:0] cnt_n;

    for (genvar g = 0; g < G; ++g) begin : g_done
        assign done_placed[g] = done_valid[g] && placed_r;
        assign done_gang[g]   = done_placed[g] && (k_r > CLUSTER_SIZE_W'(1));
        always @(*) begin
            free_n[g] = free_r[g];
            used_n[g] = used_r[g];
            live_n[g] = live_r[g];
            cnt_n[g]  = cnt_r[g];
            if (done_placed[g]) begin
                free_n[g][done_data[g].core][done_data[g].slot] = 1'b1;
                used_n[g][done_data[g].core] = used_r[g][done_data[g].core] - w_r;
            end
            if (done_gang[g]) begin
                cnt_n[g][done_data[g].gang_id] = cnt_r[g][done_data[g].gang_id] - DLMEM_K_W'(1);
                if (cnt_r[g][done_data[g].gang_id] == DLMEM_K_W'(1)) begin
                    live_n[g][done_data[g].gang_id] = 1'b0;
                end
            end
        end
    end

    // lowest free gang id per GPC: neither live nor holding back a release
    wire  [G-1:0]                id_free;
    logic [G-1:0][GANG_ID_W-1:0] free_id;
    for (genvar g = 0; g < G; ++g) begin : g_ids
        wire [NG-1:0] id_avail = ~live_n[g] & ~rel_pend[g];
        assign id_free[g] = (|id_avail);
        always @(*) begin
            free_id[g] = '0;
            for (integer i = NG - 1; i >= 0; --i) begin
                if (id_avail[i]) begin
                    free_id[g] = GANG_ID_W'(i);
                end
            end
        end
    end

    // ---- one greedy placement step per GPC on the scratch copies ------------

    wire  [G-1:0]              any_cand;
    logic [G-1:0][CORE_W-1:0]  sel_core;
    logic [G-1:0][NW_WIDTH-1:0] sel_slot;

    for (genvar g = 0; g < G; ++g) begin : g_step
        wire [NCC-1:0] cand;
        wire [NCC-1:0][RES_W-1:0] res;
        for (genvar c = 0; c < NCC; ++c) begin : g_cand
            wire [NS-1:0] taken = ~sfree_r[g][c];
            `POP_COUNT(res[c], taken);
            assign cand[c] = (|(sfree_r[g][c] & umask_r))
                          && ((USED_W+1)'(sused_r[g][c]) + (USED_W+1)'(w_r) <= (USED_W+1)'(NW))
                          && (32'(res[c]) < CAP)
                          && ~visited_r[g][c];
        end
        assign any_cand[g] = (|cand) && ~out_r[g];

        if (POLICY_H100 != 0) begin : g_h100
            `UNUSED_VAR (start_r[g])
            // least resident CTAs first, then the lowest in-GPC index
            always @(*) begin
                sel_core[g] = '0;
                for (integer i = NCC - 1; i >= 0; --i) begin
                    if (cand[i] && (~cand[sel_core[g]] || (res[i] <= res[sel_core[g]]))) begin
                        sel_core[g] = CORE_W'(i);
                    end
                end
            end
        end else begin : g_spread
            // the first candidate at or after the start core, cyclically
            always @(*) begin
                sel_core[g] = '0;
                for (integer i = NCC - 1; i >= 0; --i) begin
                    if (cand[i]) begin
                        sel_core[g] = CORE_W'(i);
                    end
                end
                for (integer i = NCC - 1; i >= 0; --i) begin
                    if (cand[i] && (CORE_W'(i) >= start_r[g])) begin
                        sel_core[g] = CORE_W'(i);
                    end
                end
            end
        end

        wire [NS-1:0] sel_avail = sfree_r[g][sel_core[g]] & umask_r;
        always @(*) begin
            sel_slot[g] = '0;
            for (integer s = NS - 1; s >= 0; --s) begin
                if (sel_avail[s]) begin
                    sel_slot[g] = NW_WIDTH'(s);
                end
            end
        end
    end

    logic [G-1:0][NCC-1:0][NS-1:0]     amask_n;
    logic [G-1:0][NCC-1:0][USED_W-1:0] aused_n;
    always @(*) begin
        amask_n = amask_r;
        aused_n = aused_r;
        for (integer g = 0; g < G; ++g) begin
            amask_n[g][sel_core[g]][sel_slot[g]] = 1'b1;
            aused_n[g][sel_core[g]] = aused_r[g][sel_core[g]] + w_r;
        end
    end

    wire is_last = ((CLUSTER_SIZE_W'(step_r) + CLUSTER_SIZE_W'(1)) == k_r);

    // the first GPC from the pointer whose placement is still whole
    logic [G_W-1:0] win;
    always @(*) begin
        win = gptr_r;
        for (integer k = G - 1; k >= 0; --k) begin
            automatic logic [G_W-1:0] idx = G_W'((32'(gptr_r) + k) % G);
            if (any_cand[idx]) begin
                win = idx;
            end
        end
    end
    wire any_gpc = (|any_cand);

    // ---- control -----------------------------------------------------------

    wire idle   = (state == S_IDLE);
    wire emit   = (state == S_EMIT);
    wire is_new = idle && head_valid && in_new;
    wire start  = idle && head_valid && in_placed
               && ((in_k == CLUSTER_SIZE_W'(1)) || (|id_free))
               && (~(failed_r && ~is_new) || dirty_r);

    // Plain beats pass through; cluster beats leave only once placed.
    wire fwd          = (idle && ~in_placed) || emit;
    wire buf_valid_in = head_valid && fwd;
    wire buf_ready_in;
    assign bus_in_if.ready = buf_ready_in && fwd;
    wire in_fire = head_valid && bus_in_if.ready;

    // ---- output beat -------------------------------------------------------

    dlmem_peer_t emit_peer;
    assign emit_peer = members_r[win_r][step_r];

    wire [KMU_DEST_W-1:0] emit_dest = KMU_DEST_W'((((32'(win_r) << NS_W) | (32'(emit_peer.core) / `VX_CFG_SOCKET_SIZE)) << SS_W)
                                                  | (32'(emit_peer.core) % `VX_CFG_SOCKET_SIZE));

    kmu_req_t out_req;
    always @(*) begin
        out_req = in_req;
        if (emit) begin
            out_req.args.compute.dlmem.policy      = KMU_POLICY_SPREAD;
            out_req.args.compute.dlmem.epoch       = cur_epoch_r;
            out_req.args.compute.dlmem.gang_id     = cur_gang_r;
            out_req.args.compute.dlmem.forced_slot = emit_peer.slot;
            out_req.args.compute.dlmem.row.k       = DLMEM_K_W'(k_r);
            out_req.args.compute.dlmem.row.peers   = members_r[win_r];
        end
    end

    wire                  out_eop  = emit || bus_in_if.eop;
    wire [KMU_DEST_W-1:0] out_dest = emit ? emit_dest : bus_in_if.dest;
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

    for (genvar g = 0; g < G; ++g) begin : g_alloc
        assign alloc_valid[g] = alloc_v_r[g];
    end
    assign alloc_gang  = alloc_g_r;
    assign alloc_epoch = alloc_e_r;
    assign alloc_k     = alloc_k_r;
    assign gang_mode   = placed_r && (k_r > CLUSTER_SIZE_W'(1));

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
            gptr_r      <= G_W'(`VX_CFG_CLUSTER_GPC_START);
            ctx_r       <= '0;
            placed_r    <= 1'b0;
            k_r         <= CLUSTER_SIZE_W'(1);
            w_r         <= '0;
            umask_r     <= '1;
            sfree_r     <= '0;
            sused_r     <= '0;
            visited_r   <= '0;
            amask_r     <= '0;
            aused_r     <= '0;
            out_r       <= '0;
            start_r     <= '0;
            step_r      <= '0;
            members_r   <= '0;
            win_r       <= '0;
            cur_gang_r  <= '0;
            cur_epoch_r <= 1'b0;
            failed_r    <= 1'b0;
            dirty_r     <= 1'b0;
            alloc_v_r   <= '0;
            alloc_g_r   <= '0;
            alloc_e_r   <= 1'b0;
            alloc_k_r   <= '0;
        end else begin
            free_r    <= free_n;
            used_r    <= used_n;
            live_r    <= live_n;
            cnt_r     <= cnt_n;
            dirty_r   <= ~start && (dirty_r || (|done_valid));
            alloc_v_r <= '0;

            case (state)
            S_IDLE: begin
                if (head_valid) begin
                    ctx_r    <= in_req.ctx_id;
                    placed_r <= in_placed;
                    k_r      <= in_k;
                    w_r      <= in_w;
                    umask_r  <= in_umask;
                    if (in_new) begin
                        // nothing is in flight at a launch boundary
                        rr_r     <= '0;
                        epoch_r  <= '0;
                        gptr_r   <= G_W'(`VX_CFG_CLUSTER_GPC_START);
                        failed_r <= 1'b0;
                    end
                end
                if (start) begin
                    sfree_r   <= free_n;
                    sused_r   <= used_n;
                    visited_r <= '0;
                    amask_r   <= '0;
                    aused_r   <= '0;
                    for (integer g = 0; g < G; ++g) begin
                        out_r[g]   <= ~((in_k == CLUSTER_SIZE_W'(1)) || id_free[g]);
                        start_r[g] <= in_new ? CORE_W'(0) : rr_r[g];
                    end
                    step_r    <= '0;
                    members_r <= '0;
                    state     <= S_PLACE;
                end
            end
            S_PLACE: begin
                if (~any_gpc) begin
                    failed_r <= 1'b1;
                    state    <= S_IDLE;
                end else begin
                    for (integer g = 0; g < G; ++g) begin
                        if (any_cand[g]) begin
                            members_r[g][step_r] <= '{core: sel_core[g], slot: sel_slot[g]};
                            sfree_r[g][sel_core[g]][sel_slot[g]] <= 1'b0;
                            sused_r[g][sel_core[g]] <= sused_r[g][sel_core[g]] + w_r;
                            visited_r[g] <= visited_r[g] | (NCC'(1) << sel_core[g]);
                            amask_r[g]   <= amask_n[g];
                            aused_r[g]   <= aused_n[g];
                        end else begin
                            out_r[g] <= 1'b1;
                        end
                    end
                    if (is_last) begin
                        // commit: the slots were free in the snapshot, and the
                        // live mirror has only released since, so they are free
                        for (integer c = 0; c < NCC; ++c) begin
                            free_r[win][c] <= free_n[win][c] & ~amask_n[win][c];
                            used_r[win][c] <= used_n[win][c] + aused_n[win][c];
                        end
                        rr_r[win] <= (sel_core[win] == CORE_W'(NCC - 1)) ? CORE_W'(0) : (sel_core[win] + CORE_W'(1));
                        gptr_r    <= (32'(win) == (G - 1)) ? G_W'(0) : (win + G_W'(1));
                        win_r     <= win;
                        if (k_r > CLUSTER_SIZE_W'(1)) begin
                            live_r[win][free_id[win]]  <= 1'b1;
                            cnt_r[win][free_id[win]]   <= DLMEM_K_W'(k_r);
                            epoch_r[win][free_id[win]] <= ~epoch_r[win][free_id[win]];
                            cur_gang_r       <= free_id[win];
                            cur_epoch_r      <= ~epoch_r[win][free_id[win]];
                            alloc_v_r[win]   <= 1'b1;
                            alloc_g_r        <= free_id[win];
                            alloc_e_r        <= ~epoch_r[win][free_id[win]];
                            alloc_k_r        <= DLMEM_K_W'(k_r);
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

    // The device stays busy until every placed block has reported done and
    // every release has left, so the next launch starts on an empty mirror.
    assign busy = head_valid | bus_out_if.valid | ~idle | ~(&free_r) | (|live_r) | (|rel_pend);

    // ---- checks (dlmem_ref's asserts) ---------------------------------------

    for (genvar g = 0; g < G; ++g) begin : g_checks
        `RUNTIME_ASSERT (~done_valid[g] || placed_r,
            ("%t: %s: CTA_DONE in GPC %0d outside a cluster launch", $time, INSTANCE_ID, g))
        `RUNTIME_ASSERT (~done_placed[g] || ((32'(done_data[g].core) < NCC) && ~free_r[g][done_data[g].core][done_data[g].slot]
                                          && (used_r[g][done_data[g].core] >= w_r)),
            ("%t: %s: CTA_DONE for GPC %0d core %0d slot %0d, which holds no placed block", $time, INSTANCE_ID, g,
             done_data[g].core, done_data[g].slot))
        `RUNTIME_ASSERT (~done_gang[g] || ((32'(done_data[g].gang_id) < NG) && live_r[g][done_data[g].gang_id]
                                        && (epoch_r[g][done_data[g].gang_id] == done_data[g].epoch)
                                        && (cnt_r[g][done_data[g].gang_id] != 0)),
            ("%t: %s: CTA_DONE for GPC %0d gang %0d epoch %0d, which is not live", $time, INSTANCE_ID, g,
             done_data[g].gang_id, done_data[g].epoch))
    end
    `RUNTIME_ASSERT (~is_new || ((&free_n) && ~(|live_n) && ~(|rel_pend)),
        ("%t: %s: a new launch began with blocks of the last one still placed", $time, INSTANCE_ID))
    `RUNTIME_ASSERT (~(idle && head_valid && in_placed)
                  || (in_req.args.compute.is_first_of_cluster && (in_req.args.compute.cluster_rank == '0)
                      && (in_k != '0) && (32'(in_k) <= MAXC) && (32'(in_w) <= NW)),
        ("%t: %s: a cluster message must start at rank 0 and fit the cluster (K=%0d, W=%0d)", $time, INSTANCE_ID,
         in_k, in_w))
    `RUNTIME_ASSERT (~(emit && head_valid)
                  || (in_placed && ~in_new && (in_req.args.compute.cluster_rank == RANK_W'(step_r))
                      && (bus_in_if.eop == is_last)),
        ("%t: %s: cluster beat out of step: rank %0d, expected %0d", $time, INSTANCE_ID,
         in_req.args.compute.cluster_rank, step_r))

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (~reset && (state == S_PLACE) && ~any_gpc) begin
            `TRACE(2, ("%t: %s place-fail: ctx=%0d, K=%0d, W=%0d, step=%0d\n", $time, INSTANCE_ID, ctx_r, k_r, w_r, step_r))
        end
        if (~reset && (state == S_PLACE) && any_gpc && is_last) begin
            `TRACE(1, ("%t: %s place: ctx=%0d, K=%0d, W=%0d, gpc=%0d, gang=%0d, last={%0d,%0d}\n", $time, INSTANCE_ID,
                ctx_r, k_r, w_r, win, (k_r > CLUSTER_SIZE_W'(1)) ? free_id[win] : '0, sel_core[win], sel_slot[win]))
        end
    end
`endif

endmodule
