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

`ifdef VX_CFG_H100_SM_ENABLE

`include "VX_define.vh"

// The L1's tag store, cycle for cycle l1_slot_ref::L1TagPortRef: per tag bank
// one lookup read and one write per cycle. Tag macros hold {valid, tag, slot}
// per way, word-valid macros a bit per word; both are 1R1W and written only
// under masks (no read-modify-write), so a store's or fill's valid bits are
// set, and a victim's or atomic's cleared, through a per-bank update queue.
// A lookup reads the macros as of its cycle and applies the masks queued for
// its row before that cycle, so a queued update is never missed.
module VX_sm_l1_tags import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter BANK_W = `LOG2UP(L1_TAG_BANKS),
    parameter ROW_W  = `LOG2UP(L1_SETS_PER_BANK),
    parameter WAY_W  = `LOG2UP(L1_TAG_WAYS),
    parameter ID_W   = L1_SLOT_W
) (
    input wire                      clk,
    input wire                      reset,

    // whole-L1 clear: one row per bank per cycle, owns every write port
    input wire                      inval_start,
    output wire                     inval_busy,

    // lookup (one per bank per cycle); the ways come out one cycle later
    input wire [L1_TAG_BANKS-1:0]   lk_valid,
    input wire [L1_TAG_BANKS-1:0][ROW_W-1:0] lk_row,
    output wire [L1_TAG_BANKS-1:0][L1_TAG_WAYS-1:0] rd_valid,
    output wire [L1_TAG_BANKS-1:0][L1_TAG_WAYS-1:0][L1_TAG_W-1:0] rd_tag,
    output wire [L1_TAG_BANKS-1:0][L1_TAG_WAYS-1:0][L1_SLOT_W-1:0] rd_slot,
    output wire [L1_TAG_BANKS-1:0][L1_TAG_WAYS-1:0][L1_VALID_BITS-1:0] rd_vmask,

    // masked updates, entered all-or-nothing in source order (lookup, victim, fill)
    input wire [L1_TAG_UPD_SRCS-1:0] up_valid,
    input wire [L1_TAG_UPD_SRCS-1:0][BANK_W-1:0] up_bank,
    input wire [L1_TAG_UPD_SRCS-1:0][ROW_W-1:0] up_row,
    input wire [L1_TAG_UPD_SRCS-1:0][WAY_W-1:0] up_way,
    input wire [L1_TAG_UPD_SRCS-1:0][L1_VALID_BITS-1:0] up_mask,
    input wire [L1_TAG_UPD_SRCS-1:0] up_set,   // 1: set the bits, 0: clear them
    input wire [L1_TAG_UPD_SRCS-1:0] up_inval, // also clears the way's tag
    input wire [L1_TAG_UPD_SRCS-1:0][ID_W-1:0] up_id,
    output wire [L1_TAG_UPD_SRCS-1:0] up_ready,
    // a bank's queue has room as of the cycle start
    output wire [L1_TAG_BANKS-1:0]  q_room,
    output wire [L1_TAG_BANKS-1:0]  q_room2,

    // allocation write of a whole way; granted when its bank's queue is empty
    input wire                      al_valid,
    input wire [BANK_W-1:0]         al_bank,
    input wire [ROW_W-1:0]          al_row,
    input wire [WAY_W-1:0]          al_way,
    input wire [L1_TAG_W-1:0]       al_tag,
    input wire [L1_SLOT_W-1:0]      al_slot,
    input wire [L1_VALID_BITS-1:0]  al_vmask,
    output wire                     al_grant,

    // an update written this cycle, per bank (fill_done for source 2)
    output wire [L1_TAG_BANKS-1:0]  wr_valid,
    output wire [L1_TAG_BANKS-1:0][1:0] wr_src,
    output wire [L1_TAG_BANKS-1:0][ID_W-1:0] wr_id
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam DEPTH  = L1_TAG_UPD_DEPTH;
    localparam CNT_W  = `CLOG2(DEPTH+1);
    localparam NSRC   = L1_TAG_UPD_SRCS;
    localparam VB     = L1_VALID_BITS;
    localparam WB     = L1_TAG_WAY_BITS;
    localparam TPM    = L1_TAG_WAYS_PER_MACRO;
    localparam VPM    = L1_VALID_WAYS_PER_MACRO;
    localparam N_TM   = L1_TAG_WAYS / TPM;
    localparam N_VM   = L1_TAG_WAYS / VPM;

    `STATIC_ASSERT(L1_TAG_SETS * L1_TAG_WAYS >= `VX_CFG_L1_MAX_LINES, ("the tags must cover every L1 line"))
    `STATIC_ASSERT((L1_TAG_SETS % L1_TAG_BANKS) == 0, ("tag sets must divide among the tag banks"))
    `STATIC_ASSERT(WB <= `VX_CFG_SRAM_MACRO_MAX_BITS, ("one way's tag bits must fit a macro"))
    `STATIC_ASSERT(VB <= `VX_CFG_SRAM_MACRO_MAX_BITS, ("one way's valid bits must fit a macro"))
    `STATIC_ASSERT((L1_TAG_WAYS % TPM) == 0 && (L1_TAG_WAYS % VPM) == 0, ("macro split"))

    typedef struct packed {
        logic [ROW_W-1:0] row;
        logic [WAY_W-1:0] way;
        logic [VB-1:0]    mask;
        logic             set;
        logic             inval;
        logic [1:0]       src;
        logic [ID_W-1:0]  id;
    } upd_t;

    // ---- invalidate sequencer ----
    reg              inval_r;
    reg [ROW_W-1:0]  inval_row_r;
    always @(posedge clk) begin
        if (reset) begin
            // the tag macros power up undefined: clear every row before the first lookup
            inval_r     <= 1;
            inval_row_r <= '0;
        end else if (inval_start && !inval_r) begin
            inval_r     <= 1;
            inval_row_r <= '0;
        end else if (inval_r) begin
            inval_row_r <= inval_row_r + 1;
            if (inval_row_r == ROW_W'(L1_SETS_PER_BANK-1)) begin
                inval_r <= 0;
            end
        end
    end
    assign inval_busy = inval_r;

    // ---- update admission: all-or-nothing per source, space as of the cycle start ----
    upd_t [NSRC-1:0] up_ent;
    for (genvar s = 0; s < NSRC; ++s) begin : g_up_ent
        assign up_ent[s] = '{row: up_row[s], way: up_way[s], mask: up_mask[s], set: up_set[s],
                             inval: up_inval[s], src: 2'(s), id: up_id[s]};
    end

    wire [L1_TAG_BANKS-1:0][CNT_W-1:0] q_cnt;
    reg [NSRC-1:0] up_ok;
    always @(*) begin
        integer used;
        reg [NSRC-1:0] ok;
        ok = '0;
        for (integer s = 0; s < NSRC; ++s) begin
            used = 0;
            for (integer p = 0; p < s; ++p) begin
                if (up_valid[p] && ok[p] && up_bank[p] == up_bank[s]) begin
                    used = used + 1;
                end
            end
            ok[s] = (32'(q_cnt[up_bank[s]]) + used + 1) <= DEPTH;
        end
        up_ok = ok;
    end
    assign up_ready = up_ok;
    for (genvar b = 0; b < L1_TAG_BANKS; ++b) begin : g_q_room
        assign q_room[b]  = (32'(q_cnt[b]) < DEPTH);
        assign q_room2[b] = (32'(q_cnt[b]) + 2 <= DEPTH);
    end
    wire [NSRC-1:0] up_fire = up_valid & up_ok;

    wire [L1_TAG_BANKS-1:0] alloc_g;

    for (genvar b = 0; b < L1_TAG_BANKS; ++b) begin : g_bank

        // ---- queue ----
        upd_t [DEPTH-1:0] q_r;
        reg [CNT_W-1:0]   cnt;
        assign q_cnt[b] = cnt;

        reg [NSRC-1:0] pushed;
        always @(*) begin
            for (integer s = 0; s < NSRC; ++s) begin
                pushed[s] = up_fire[s] && (up_bank[s] == BANK_W'(b));
            end
        end

        // the queue after this cycle's pushes, before the write port's pop
        upd_t [DEPTH-1:0] q_push;
        reg [CNT_W-1:0]   cnt_push;
        always @(*) begin
            q_push   = q_r;
            cnt_push = cnt;
            for (integer s = 0; s < NSRC; ++s) begin
                if (pushed[s]) begin
                    q_push[cnt_push] = up_ent[s];
                    cnt_push = cnt_push + 1;
                end
            end
        end

        wire do_inval = inval_r;
        wire do_upd   = !inval_r && (cnt_push != 0);
        wire do_alloc = !inval_r && (cnt_push == 0) && al_valid && (al_bank == BANK_W'(b));
        upd_t head;
        assign head = q_push[0];
        assign alloc_g[b] = do_alloc;

        always @(posedge clk) begin
            if (reset) begin
                cnt <= '0;
            end else begin
                if (do_upd) begin
                    for (integer i = 0; i < DEPTH-1; ++i) begin
                        q_r[i] <= q_push[i+1];
                    end
                    cnt <= cnt_push - 1;
                end else begin
                    q_r <= q_push;
                    cnt <= cnt_push;
                end
            end
        end

        assign wr_valid[b] = do_upd;
        assign wr_src[b]   = head.src;
        assign wr_id[b]    = head.id;

        // ---- write port ----
        wire [ROW_W-1:0] w_row = do_inval ? inval_row_r : (do_upd ? head.row : al_row);

        // ---- lookup bypass: the masks queued as of the lookup cycle, folded per way ----
        reg [L1_TAG_WAYS-1:0][VB-1:0] byp_s, byp_c;
        reg [L1_TAG_WAYS-1:0]         byp_inv;
        always @(*) begin
            byp_s   = '0;
            byp_c   = '0;
            byp_inv = '0;
            for (integer i = 0; i < DEPTH; ++i) begin
                if (CNT_W'(i) < cnt && q_r[i].row == lk_row[b]) begin
                    if (q_r[i].set) begin
                        byp_s[q_r[i].way] = byp_s[q_r[i].way] | q_r[i].mask;
                    end else begin
                        byp_s[q_r[i].way] = byp_s[q_r[i].way] & ~q_r[i].mask;
                        byp_c[q_r[i].way] = byp_c[q_r[i].way] | q_r[i].mask;
                    end
                    if (q_r[i].inval) begin
                        byp_inv[q_r[i].way] = 1;
                    end
                end
            end
        end

        reg [L1_TAG_WAYS-1:0][VB-1:0] byp_s_r, byp_c_r;
        reg [L1_TAG_WAYS-1:0]         byp_inv_r;
        always @(posedge clk) begin
            byp_s_r   <= byp_s;
            byp_c_r   <= byp_c;
            byp_inv_r <= byp_inv;
        end

        // ---- tag macros ----
        wire [L1_TAG_WAYS-1:0][WB-1:0] t_rdata;
        for (genvar m = 0; m < N_TM; ++m) begin : g_tm
            reg [TPM-1:0]          wren;
            reg [TPM-1:0][WB-1:0]  wdata;
            always @(*) begin
                wren  = '0;
                wdata = '0;
                for (integer k = 0; k < TPM; ++k) begin
                    if (do_inval) begin
                        wren[k] = 1;
                    end else if (do_upd) begin
                        wren[k] = head.inval && (32'(head.way) == m * TPM + k);
                    end else if (do_alloc) begin
                        wren[k] = (32'(al_way) == m * TPM + k);
                        wdata[k] = {1'b1, al_tag, al_slot};
                    end
                end
            end
            VX_dp_ram #(
                .DATAW    (TPM * WB),
                .SIZE     (L1_SETS_PER_BANK),
                .WRENW    (TPM),
                .OUT_REG  (1),
                .RDW_MODE ("R")
            ) tag_ram (
                .clk   (clk),
                .reset (reset),
                .read  (lk_valid[b]),
                .write (|wren),
                .wren  (wren),
                .waddr (w_row),
                .wdata (wdata),
                .raddr (lk_row[b]),
                .rdata (t_rdata[m * TPM +: TPM])
            );
        end

        // ---- word-valid macros (bit-masked writes) ----
        wire [L1_TAG_WAYS-1:0][VB-1:0] v_rdata;
        for (genvar m = 0; m < N_VM; ++m) begin : g_vm
            reg [VPM-1:0][VB-1:0] wren;
            reg [VPM-1:0][VB-1:0] wdata;
            always @(*) begin
                wren  = '0;
                wdata = '0;
                for (integer k = 0; k < VPM; ++k) begin
                    if (do_inval) begin
                        wren[k] = '1;
                    end else if (do_upd) begin
                        if (32'(head.way) == m * VPM + k) begin
                            wren[k]  = head.inval ? '1 : head.mask;
                            wdata[k] = (head.set && !head.inval) ? '1 : '0;
                        end
                    end else if (do_alloc) begin
                        if (32'(al_way) == m * VPM + k) begin
                            wren[k]  = '1;
                            wdata[k] = al_vmask;
                        end
                    end
                end
            end
            VX_dp_ram #(
                .DATAW    (VPM * VB),
                .SIZE     (L1_SETS_PER_BANK),
                .WRENW    (VPM * VB),
                .OUT_REG  (1),
                .RDW_MODE ("R")
            ) valid_ram (
                .clk   (clk),
                .reset (reset),
                .read  (lk_valid[b]),
                .write (|wren),
                .wren  (wren),
                .waddr (w_row),
                .wdata (wdata),
                .raddr (lk_row[b]),
                .rdata (v_rdata[m * VPM +: VPM])
            );
        end

        for (genvar w = 0; w < L1_TAG_WAYS; ++w) begin : g_out
            assign rd_valid[b][w] = t_rdata[w][WB-1] && !byp_inv_r[w];
            assign rd_tag[b][w]   = t_rdata[w][L1_SLOT_W +: L1_TAG_W];
            assign rd_slot[b][w]  = t_rdata[w][L1_SLOT_W-1:0];
            assign rd_vmask[b][w] = (v_rdata[w] & ~byp_c_r[w]) | byp_s_r[w];
        end
    end

    assign al_grant = |alloc_g;

endmodule

`endif // VX_CFG_H100_SM_ENABLE
