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

// The L1's slot pool on the unified array, cycle for cycle the pool half of
// l1_slot_ref::L1SlotDir: a bump pointer, a free FIFO of returned borrowed
// slots, and the age list (young FIFO in allocation order + old table). A full
// L1 draws a uniform old entry (registered read at t, victim at t + 1), swaps
// young's head into its place and pushes the reused slot to young, so one
// line per cycle is allocated with one old-table read and one write.
// Per cycle: at most one pool request (the tag stage's allocation); retires
// and opens take effect from the next cycle; migration runs last.
module VX_sm_l1_slots import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter SET_W = `LOG2UP(L1_TAG_SETS),
    parameter WAY_W = `LOG2UP(L1_TAG_WAYS)
) (
    input wire                      clk,
    input wire                      reset,

    input wire [L1_SLOT_W:0]        cfg_lines,
    input wire [`CLOG2(L1_PROTECT_MAX+1)-1:0] cfg_protect,
    // pool reset; the caller has drained every open entry
    input wire                      invalidate,

    // in-flight state of a set's slots (the in-set victim candidates)
    input wire [L1_TAG_WAYS-1:0][L1_SLOT_W-1:0] q_slot,
    output wire [L1_TAG_WAYS-1:0]   q_open,

    output wire                     free_avail,
    output wire [15:0]              lfsr_next,
    // the tag stage consumed lfsr_next (an in-set victim pick)
    input wire                      lfsr_step,

    input wire                      pool_valid,
    input wire [1:0]                pool_op,
    input wire [SET_W-1:0]          pool_set,
    input wire [WAY_W-1:0]          pool_way,
    output wire                     pool_ok,
    output wire [2:0]               pool_wait,
    output wire [2:0]               pool_kind,
    output wire [L1_SLOT_W-1:0]     pool_slot,
    output wire                     victim_valid,
    // the victim's slot has an allocation write still queued
    input wire                      victim_busy,
    output wire [L1_SLOT_W-1:0]     victim_slot,
    output wire [SET_W-1:0]         victim_set,
    output wire [WAY_W-1:0]         victim_way,

    input wire                      open_valid,
    input wire [L1_SLOT_W-1:0]      open_slot,

    input wire                      retire_valid,
    input wire [L1_SLOT_W-1:0]      retire_slot,
    input wire                      retire_borrowed,

    output wire [`CLOG2(L1_PROTECT_MAX+2)-1:0] young_count,
    output wire [L1_SLOT_W:0]       old_count,
    output wire [L1_SLOT_W:0]       bump_count,
    output wire [`CLOG2(L1_FREE_DEPTH+1)-1:0] free_count
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam MAX_LINES = `VX_CFG_L1_MAX_LINES;
    localparam YOUNG_DEPTH = L1_PROTECT_MAX + 1;
    localparam YOUNG_AW = `LOG2UP(YOUNG_DEPTH);
    localparam YOUNG_CW = `CLOG2(L1_PROTECT_MAX+2);
    localparam FREE_AW  = `LOG2UP(L1_FREE_DEPTH);
    localparam FREE_CW  = `CLOG2(L1_FREE_DEPTH+1);
    localparam OLD_AW   = `LOG2UP(L1_OLD_DEPTH);
    localparam REF_W    = L1_SLOT_W + SET_W + WAY_W;
    localparam [15:0] LFSR_SEED = 16'hACE1;
    localparam [15:0] LFSR_TAPS = 16'hB400;

    `STATIC_ASSERT(MAX_LINES < 65536, ("16-bit victim draw"))
    `STATIC_ASSERT(L1_OLD_DEPTH >= MAX_LINES, ("old table depth"))

    // l1_slot_ref::L1SlotDir::Kind and ::Wait encodings
    localparam KIND_FREE = 3'd1, KIND_BUMP = 3'd2, KIND_AGE = 3'd4, KIND_BORROW = 3'd5;
    localparam WAIT_NONE = 3'd0, WAIT_YOUNG_FULL = 3'd1, WAIT_OLD_EMPTY = 3'd2,
               WAIT_VICTIM_IN_FLIGHT = 3'd3, WAIT_DRAW_PENDING = 3'd4;
    localparam OP_BORROW = 2'd0, OP_YOUNG = 2'd1, OP_DRAW = 2'd2;

    // ---- state ----
    reg [L1_SLOT_W:0]       bump_r;
    reg [MAX_LINES-1:0]     open_r;
    reg [15:0]              lfsr_r;
    reg [L1_SLOT_W:0]       n_old_r;
    reg                     draw_r;
    reg [OLD_AW-1:0]        draw_idx_r;
    reg [OLD_AW-1:0]        last_widx_r;
    reg                     last_wen_r;
    reg [REF_W-1:0]         last_wdata_r;

    reg [YOUNG_AW-1:0]      y_head_r, y_tail_r;
    reg [YOUNG_CW-1:0]      y_cnt_r;
    reg [FREE_AW-1:0]       f_head_r, f_tail_r;
    reg [FREE_CW-1:0]       f_cnt_r;

    wire [REF_W-1:0] young_head;
    wire [L1_SLOT_W-1:0] free_head;
    wire [REF_W-1:0] old_rdata;

    for (genvar i = 0; i < L1_TAG_WAYS; ++i) begin : g_q_open
        assign q_open[i] = open_r[q_slot[i]];
    end

    wire [15:0] lfsr_n = (lfsr_r >> 1) ^ (lfsr_r[0] ? LFSR_TAPS : 16'h0);
    assign lfsr_next = lfsr_n;

    wire free_fifo_nz = (f_cnt_r != 0);
    wire bump_avail   = (bump_r < cfg_lines);
    assign free_avail = free_fifo_nz || bump_avail;
    wire young_full   = (y_cnt_r >= YOUNG_CW'(YOUNG_DEPTH));

    // victim of a completing draw; a same-index write in the draw cycle is forwarded
    wire [REF_W-1:0] victim_ref = (last_wen_r && last_widx_r == draw_idx_r) ? last_wdata_r : old_rdata;
    wire [L1_SLOT_W-1:0] v_slot = victim_ref[REF_W-1 -: L1_SLOT_W];
    wire [SET_W-1:0]     v_set  = victim_ref[WAY_W +: SET_W];
    wire [WAY_W-1:0]     v_way  = victim_ref[WAY_W-1:0];
    wire v_open = open_r[v_slot];

    wire is_borrow = pool_valid && (pool_op == OP_BORROW);
    wire is_young  = pool_valid && (pool_op == OP_YOUNG);
    wire is_draw   = pool_valid && (pool_op == OP_DRAW);

    // ---- pool request ----
    reg              ok;
    reg [2:0]        wcode, kind;
    reg [L1_SLOT_W-1:0] slot;
    reg              take_free, take_bump, push_young, pop_young_swap;
    reg [REF_W-1:0]  push_ref;
    reg              start_draw, want_migrate, draw_lfsr;
    reg              vict;

    always @(*) begin
        ok             = 0;
        wcode          = WAIT_NONE;
        kind           = KIND_FREE;
        slot           = free_head;
        take_free      = 0;
        take_bump      = 0;
        push_young     = 0;
        pop_young_swap = 0;
        push_ref       = {free_head, pool_set, pool_way};
        start_draw     = 0;
        want_migrate   = 0;
        draw_lfsr      = 0;
        vict           = 0;
        if (is_borrow || is_young) begin
            if (is_young && young_full) begin
                wcode = WAIT_YOUNG_FULL;
            end else begin
                ok = 1;
                if (free_fifo_nz) begin
                    take_free = 1;
                    kind = is_borrow ? KIND_BORROW : KIND_FREE;
                    slot = free_head;
                end else begin
                    take_bump = 1;
                    kind = is_borrow ? KIND_BORROW : KIND_BUMP;
                    slot = L1_SLOT_W'(bump_r);
                end
                push_young = is_young;
                push_ref = {slot, pool_set, pool_way};
            end
        end else if (is_draw) begin
            if (draw_r && !v_open && !victim_busy) begin
                ok             = 1;
                kind           = KIND_AGE;
                slot           = v_slot;
                vict           = 1;
                pop_young_swap = 1;
                push_young     = 1;
                push_ref       = {v_slot, pool_set, pool_way};
            end else begin
                if (young_full) begin
                    wcode = WAIT_YOUNG_FULL;
                end else if (n_old_r == 0) begin
                    want_migrate = 1;
                    wcode = WAIT_OLD_EMPTY;
                end else begin
                    start_draw = 1;
                    draw_lfsr  = 1;
                    wcode = draw_r ? WAIT_VICTIM_IN_FLIGHT : WAIT_DRAW_PENDING;
                end
            end
        end
    end

    assign pool_ok      = ok;
    assign pool_wait    = wcode;
    assign pool_kind    = kind;
    assign pool_slot    = slot;
    assign victim_valid = vict;
    assign victim_slot  = v_slot;
    assign victim_set   = v_set;
    assign victim_way   = v_way;

    wire [OLD_AW-1:0] draw_idx_n = OLD_AW'((32'(lfsr_n) * 32'(n_old_r)) >> 16);

    // ---- young FIFO occupancy after this cycle's request, then migration ----
    wire [YOUNG_CW-1:0] y_cnt_req = y_cnt_r + YOUNG_CW'(push_young && !pop_young_swap);
    wire migrate = !pop_young_swap && (y_cnt_req != 0)
                && ((y_cnt_req > YOUNG_CW'(cfg_protect)) || want_migrate);
    wire pop_young = pop_young_swap || migrate;

    wire old_wen = pop_young;
    wire [OLD_AW-1:0] old_widx = pop_young_swap ? draw_idx_r : OLD_AW'(n_old_r);

    `RUNTIME_ASSERT(!(is_young && !free_avail), ("%t: *** %s young allocation without a free slot", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(!(migrate && push_young && y_cnt_r == 0), ("%t: *** %s young head pushed and migrated in one cycle", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(!(retire_valid && retire_borrowed && f_cnt_r == FREE_CW'(L1_FREE_DEPTH)), ("%t: *** %s free FIFO overflow", $time, INSTANCE_ID))

    // the FIFO heads' next values address the RAMs, so their reads are registered (SRAM macros)
    wire [YOUNG_AW-1:0] y_head_n = (reset || invalidate) ? '0
                                 : (pop_young ? ((y_head_r == YOUNG_AW'(YOUNG_DEPTH-1)) ? '0 : y_head_r + 1) : y_head_r);
    wire [FREE_AW-1:0]  f_head_n = (reset || invalidate) ? '0
                                 : (take_free ? ((f_head_r == FREE_AW'(L1_FREE_DEPTH-1)) ? '0 : f_head_r + 1) : f_head_r);

    always @(posedge clk) begin
        y_head_r <= y_head_n;
        f_head_r <= f_head_n;
        if (reset || invalidate) begin
            bump_r     <= '0;
            n_old_r    <= '0;
            draw_r     <= 0;
            y_tail_r   <= '0;
            y_cnt_r    <= '0;
            f_tail_r   <= '0;
            f_cnt_r    <= '0;
            last_wen_r <= 0;
        end else begin
            if (take_bump) begin
                bump_r <= bump_r + 1;
            end
            if (start_draw) begin
                draw_r     <= 1;
                draw_idx_r <= draw_idx_n;
            end else if (is_draw) begin
                draw_r <= 0;
            end
            if (migrate) begin
                n_old_r <= n_old_r + 1;
            end
            if (push_young) begin
                y_tail_r <= (y_tail_r == YOUNG_AW'(YOUNG_DEPTH-1)) ? '0 : y_tail_r + 1;
            end
            y_cnt_r <= y_cnt_r + YOUNG_CW'(push_young) - YOUNG_CW'(pop_young);
            if (retire_valid && retire_borrowed) begin
                f_tail_r <= (f_tail_r == FREE_AW'(L1_FREE_DEPTH-1)) ? '0 : f_tail_r + 1;
            end
            f_cnt_r <= f_cnt_r + FREE_CW'(retire_valid && retire_borrowed) - FREE_CW'(take_free);
            last_wen_r   <= old_wen;
            last_widx_r  <= old_widx;
            last_wdata_r <= young_head;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            lfsr_r <= LFSR_SEED;
        end else if (draw_lfsr || lfsr_step) begin
            lfsr_r <= lfsr_n;
        end
    end

    always @(posedge clk) begin
        if (reset || invalidate) begin
            open_r <= '0;
        end else begin
            if (retire_valid) begin
                open_r[retire_slot] <= 0;
            end
            if (open_valid) begin
                open_r[open_slot] <= 1;
            end
        end
    end

    `RUNTIME_ASSERT(!(open_valid && retire_valid && open_slot == retire_slot), ("%t: *** %s slot opened and retired in one cycle", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(!(lfsr_step && draw_lfsr), ("%t: *** %s two LFSR draws in one cycle", $time, INSTANCE_ID))

    VX_dp_ram #(
        .DATAW    (REF_W),
        .SIZE     (YOUNG_DEPTH),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) young_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (push_young),
        .wren  (1'b1),
        .waddr (y_tail_r),
        .wdata (push_ref),
        .raddr (y_head_n),
        .rdata (young_head)
    );

    VX_dp_ram #(
        .DATAW    (L1_SLOT_W),
        .SIZE     (L1_FREE_DEPTH),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) free_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (retire_valid && retire_borrowed),
        .wren  (1'b1),
        .waddr (f_tail_r),
        .wdata (retire_slot),
        .raddr (f_head_n),
        .rdata (free_head)
    );

    VX_dp_ram #(
        .DATAW    (REF_W),
        .SIZE     (L1_OLD_DEPTH),
        .OUT_REG  (1),
        .RDW_MODE ("R")
    ) old_ram (
        .clk   (clk),
        .reset (reset),
        .read  (start_draw),
        .write (old_wen),
        .wren  (1'b1),
        .waddr (old_widx),
        .wdata (young_head),
        .raddr (draw_idx_n),
        .rdata (old_rdata)
    );

    assign young_count = y_cnt_r;
    assign old_count   = n_old_r;
    assign bump_count  = bump_r;
    assign free_count  = f_cnt_r;

endmodule

`endif // VX_CFG_H100_SM_ENABLE
