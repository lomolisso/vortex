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

// The L1's miss entries: one per line in flight, holding the line's slot while
// any request that joined it may still read the slot: an entry retires once it
// is filled and every replay that waited on it has left the wavefront stage.
// Counts and stage flags, not sequence numbers: a fill can outlast any
// sequence window, since hits keep passing the requests that wait on it.
module VX_sm_l1_mshr import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter SIZE    = `VX_CFG_SM_MAX_READ_LINES,
    parameter NUM_LK  = L1_TAG_BANKS,
    parameter LINE_W  = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(`VX_CFG_DCACHE_LINE_SIZE),
    parameter NSEC    = `VX_CFG_DCACHE_LINE_SIZE / `VX_CFG_DCACHE_SECTOR_SIZE,
    parameter SET_W   = `LOG2UP(L1_TAG_SETS),
    parameter WAY_W   = `LOG2UP(L1_TAG_WAYS),
    parameter RPC_MAX = `VX_CFG_LSU_PENDING_SIZE * `VX_CFG_NUM_LSU_BLOCKS,
    parameter ID_W    = `LOG2UP(SIZE)
) (
    input wire                      clk,
    input wire                      reset,

    // line lookup, one per tag bank per cycle; a held lookup's entry does not retire that cycle
    input wire [NUM_LK-1:0]         lk_hold,
    input wire [NUM_LK-1:0][LINE_W-1:0] lk_line,
    output wire [NUM_LK-1:0]        lk_hit,
    output wire [NUM_LK-1:0][ID_W-1:0] lk_id,
    output wire [NUM_LK-1:0][NSEC-1:0] lk_sectors,
    output wire [NUM_LK-1:0]        lk_store,
    output wire [NUM_LK-1:0][L1_SLOT_W-1:0] lk_slot,

    // a new entry; its allocating request is its first joiner
    input wire                      al_valid,
    input wire [LINE_W-1:0]         al_line,
    input wire [NSEC-1:0]           al_sectors,
    input wire [L1_SLOT_W-1:0]      al_slot,
    input wire [SET_W-1:0]          al_set,
    input wire [WAY_W-1:0]          al_way,
    input wire [L1_VALID_BITS-1:0]  al_vmiss,
    input wire                      al_cacheable,
    input wire [1:0]                al_ilk,      // a reused slot's fill blockers now: {array, FIN}
    output wire                     al_ready,
    output wire [ID_W-1:0]          al_id,

    // requests joining open entries (stores mark the entry)
    input wire [NUM_LK-1:0]         jn_valid,
    input wire [NUM_LK-1:0][ID_W-1:0] jn_id,
    input wire [NUM_LK-1:0]         jn_store,

    // replays queued on an entry (allocation included) and replays that left the wavefront stage
    input wire [SIZE-1:0]           rp_inc,
    input wire [SIZE-1:0]           rp_dec,

    // where the global items older than or equal to a slot reuse's request move this cycle
    input wire                      ilk_fin_take,  // FIN's commit item goes to the array
    input wire                      ilk_arr_take,  // the array takes an item
    input wire                      ilk_arr_rep,   // ... and it is a replay
    input wire                      ilk_arr_ret,   // the array retires an item

    // the fill's entry
    input wire [ID_W-1:0]           fl_id,
    output wire [LINE_W-1:0]        fl_line,
    output wire [NSEC-1:0]          fl_sectors,
    output wire [L1_SLOT_W-1:0]     fl_slot,
    output wire [SET_W-1:0]         fl_set,
    output wire [WAY_W-1:0]         fl_way,
    output wire [L1_VALID_BITS-1:0] fl_vmiss,
    output wire                     fl_cacheable,
    output wire                     fl_ilk_block,
    input wire                      fl_done,     // the fill's row and valid bits are written

    // one retirement per cycle
    output wire                     rt_valid,
    output wire [ID_W-1:0]          rt_id,
    output wire [L1_SLOT_W-1:0]     rt_slot,
    output wire                     rt_cacheable,

    output wire                     empty,
    output wire                     full
);
    `UNUSED_SPARAM (INSTANCE_ID)

    typedef struct packed {
        logic [LINE_W-1:0]        line;
        logic [NSEC-1:0]          sectors;
        logic [L1_SLOT_W-1:0]     slot;
        logic [SET_W-1:0]         set;
        logic [WAY_W-1:0]         way;
        logic [L1_VALID_BITS-1:0] vmiss;
        logic                     cacheable;
    } entry_t;

    localparam CNT_W = `CLOG2(RPC_MAX + 1);

    entry_t [SIZE-1:0]          ent_r;
    reg [SIZE-1:0]              valid_r;
    reg [SIZE-1:0]              filled_r;
    reg [SIZE-1:0]              store_r;
    reg [SIZE-1:0][CNT_W-1:0]   cnt_r;
    reg [SIZE-1:0][1:0]         ilk_r;

    // ---- allocation: lowest free entry ----
    wire [ID_W-1:0] free_id;
    wire            free_found;
    VX_priority_encoder #(
        .N (SIZE)
    ) free_enc (
        .data_in   (~valid_r),
        .index_out (free_id),
        `UNUSED_PIN (onehot_out),
        .valid_out (free_found)
    );
    assign al_ready = free_found;
    assign al_id    = free_id;
    wire al_fire = al_valid && free_found;

    // ---- lookup ----
    wire [NUM_LK-1:0][SIZE-1:0] lk_match;
    for (genvar l = 0; l < NUM_LK; ++l) begin : g_lk
        wire [SIZE-1:0] match;
        for (genvar i = 0; i < SIZE; ++i) begin : g_m
            assign match[i] = valid_r[i] && (ent_r[i].line == lk_line[l]);
        end
        assign lk_match[l] = lk_hold[l] ? match : '0;
        VX_priority_encoder #(
            .N (SIZE)
        ) lk_enc (
            .data_in   (match),
            .index_out (lk_id[l]),
            `UNUSED_PIN (onehot_out),
            .valid_out (lk_hit[l])
        );
        assign lk_sectors[l] = ent_r[lk_id[l]].sectors;
        assign lk_store[l]   = store_r[lk_id[l]];
        assign lk_slot[l]    = ent_r[lk_id[l]].slot;
    end

    // ---- fill ----
    assign fl_line      = ent_r[fl_id].line;
    assign fl_sectors   = ent_r[fl_id].sectors;
    assign fl_slot      = ent_r[fl_id].slot;
    assign fl_set       = ent_r[fl_id].set;
    assign fl_way       = ent_r[fl_id].way;
    assign fl_vmiss     = ent_r[fl_id].vmiss;
    assign fl_cacheable = ent_r[fl_id].cacheable;
    assign fl_ilk_block = (ilk_r[fl_id] != '0);

    // ---- fill interlock: the blockers move with the items; index SIZE is the allocation ----
    wire [SIZE:0][1:0] ilk_cur = {al_ilk, ilk_r};
    wire [SIZE:0][1:0] ilk_nxt;
    for (genvar i = 0; i <= SIZE; ++i) begin : g_ilk
        assign ilk_nxt[i][0] = ilk_cur[i][0] && ~ilk_fin_take;
        assign ilk_nxt[i][1] = ilk_arr_take ? (ilk_cur[i][0] && ~ilk_arr_rep && ~ilk_arr_ret)
                                            : (ilk_cur[i][1] && ~ilk_arr_ret);
    end

    // ---- retirement: filled, and no replay of it still to leave the wavefront stage ----
    reg [SIZE-1:0] looked;
    always @(*) begin
        looked = '0;
        for (integer l = 0; l < NUM_LK; ++l) begin
            looked = looked | lk_match[l];
        end
    end
    wire [SIZE-1:0] can_rt;
    for (genvar i = 0; i < SIZE; ++i) begin : g_rt
        assign can_rt[i] = valid_r[i] && filled_r[i] && (cnt_r[i] == '0) && !looked[i];
    end
    VX_priority_encoder #(
        .N (SIZE)
    ) rt_enc (
        .data_in   (can_rt),
        .index_out (rt_id),
        `UNUSED_PIN (onehot_out),
        .valid_out (rt_valid)
    );
    assign rt_slot      = ent_r[rt_id].slot;
    assign rt_cacheable = ent_r[rt_id].cacheable;

    always @(posedge clk) begin
        if (reset) begin
            valid_r  <= '0;
            filled_r <= '0;
            store_r  <= '0;
            cnt_r    <= '0;
            ilk_r    <= '0;
        end else begin
            for (integer i = 0; i < SIZE; ++i) begin
                cnt_r[i] <= cnt_r[i] + CNT_W'(rp_inc[i]) - CNT_W'(rp_dec[i]);
                ilk_r[i] <= ilk_nxt[i];
            end
            if (rt_valid) begin
                valid_r[rt_id] <= 0;
            end
            if (fl_done) begin
                filled_r[fl_id] <= 1;
            end
            for (integer l = 0; l < NUM_LK; ++l) begin
                if (jn_valid[l] && jn_store[l]) begin
                    store_r[jn_id[l]] <= 1;
                end
            end
            if (al_fire) begin
                valid_r[free_id]  <= 1;
                filled_r[free_id] <= 0;
                store_r[free_id]  <= 0;
                cnt_r[free_id]    <= CNT_W'(rp_inc[free_id]);
                ilk_r[free_id]    <= ilk_nxt[SIZE];
                ent_r[free_id]    <= '{line: al_line, sectors: al_sectors, slot: al_slot, set: al_set, way: al_way,
                                       vmiss: al_vmiss, cacheable: al_cacheable};
            end
        end
    end

    assign empty = ~(|valid_r);
    assign full  = ~free_found;

`ifdef SIMULATION
    always @(posedge clk) begin
        if (!reset) begin
            if (fl_done && !valid_r[fl_id]) begin
                `ASSERT(0, ("%t: *** %s fill of a closed entry %0d", $time, INSTANCE_ID, fl_id));
            end
            for (integer i = 0; i < SIZE; ++i) begin
                if (rp_dec[i] && (cnt_r[i] == '0)) begin
                    `ASSERT(0, ("%t: *** %s replay count underflow on entry %0d", $time, INSTANCE_ID, i));
                end
                if (rp_inc[i] && !rp_dec[i] && (32'(cnt_r[i]) == RPC_MAX)) begin
                    `ASSERT(0, ("%t: *** %s replay count overflow on entry %0d", $time, INSTANCE_ID, i));
                end
            end
            for (integer l = 0; l < NUM_LK; ++l) begin
                if (jn_valid[l] && !valid_r[jn_id[l]]) begin
                    `ASSERT(0, ("%t: *** %s join of a closed entry %0d", $time, INSTANCE_ID, jn_id[l]));
                end
            end
        end
    end
`endif

endmodule

`endif // VX_CFG_H100_SM_ENABLE
