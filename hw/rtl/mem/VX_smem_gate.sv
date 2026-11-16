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

// The H100 shared-memory port of one SM, between the LSU blocks' local legs
// and the banks (SimX SmemGate): one wavefront per cycle in total, every other
// local-memory client ahead of the LSU, and each read answered at the golden
// time after its issue (sw/common/smem_bank_ref.h). The banks below stay as
// they are; same-word read lanes reach them once.
module VX_smem_gate import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BLOCKS   = 1,
    parameter NUM_LANES    = 1,
    parameter WORD_SIZE    = 8,
    parameter TAG_WIDTH    = 1,
    parameter TAG_SEL_BITS = 1,
    parameter ADDR_WIDTH   = 1
) (
    input wire              clk,
    input wire              reset,

    // Another local-memory client first presents an access this cycle.
    input wire              ext_valid,

    VX_lsu_mem_if.slave     in_if [NUM_BLOCKS],
    VX_lsu_mem_if.master    out_if [NUM_BLOCKS]
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // Counted lanes a merged or paired access performs besides its own, so the
    // local-access tracker drains with the one bank commit that serves them.
    , output wire [NUM_BLOCKS-1:0]                              fold_valid
    , output wire [NUM_BLOCKS-1:0][NW_WIDTH-1:0]                fold_wid
    , output wire [NUM_BLOCKS-1:0][`CLOG2(NUM_LANES+1)-1:0]     fold_cnt
`endif
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam BANKS      = `VX_CFG_LMEM_NUM_BANKS;
    localparam BANK_BYTES = `VX_CFG_LMEM_BANK_BYTES;
    localparam BANK_BITS  = `CLOG2(BANKS);
    localparam HPW        = WORD_SIZE / BANK_BYTES;     // bank words per lane word
    localparam HPW_BITS   = `CLOG2(HPW);
    localparam NH         = NUM_LANES * HPW;
    localparam HW_WIDTH   = ADDR_WIDTH + HPW_BITS;
    localparam CNT_W      = `CLOG2(2 * NH + 1);
    localparam LAT_W      = 16;
    localparam TIME_W     = 32;
    localparam LANE_BITS  = `UP(`CLOG2(NUM_LANES));
    localparam NUM_TAGS   = 1 << TAG_SEL_BITS;
    localparam RSP_DATAW  = NUM_LANES + NUM_LANES * WORD_SIZE * 8 + TAG_WIDTH + TIME_W;
    localparam BLK_BITS   = `UP(`CLOG2(NUM_BLOCKS));
    // A load's cycles outside the hold (SimX SmemGate's): the core's from its
    // issue to the LSU's request and from the LSU's response to the dependent
    // instruction's issue, the LSU memory switch's local leg buffers, and the
    // DLMEM address translation's request register (VX_dlmem_xlate) when present.
    localparam LMEM_CORE_STAGES   = 12 + OPC_ROUTE_STAGES;
    localparam LMEM_SWITCH_STAGES = 2;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    localparam LMEM_XLATE_STAGES  = 1;
`else
    localparam LMEM_XLATE_STAGES  = 0;
`endif
    localparam PIPE_OVERHEAD      = LMEM_CORE_STAGES + LMEM_SWITCH_STAGES + LMEM_XLATE_STAGES;
    // Responses held per block: a stream of one-wavefront reads, one release a cycle, never fills it.
    localparam RSP_DEPTH  = 1 << `CLOG2(`VX_CFG_LMEM_LATENCY_64 + `VX_CFG_LMEM_WIDE_OFFSET_64);

    `STATIC_ASSERT(BANKS * BANK_BYTES == `VX_CFG_SM_DATAPATH_BYTES, ("the shared-memory banks must span the SM datapath"))
    `STATIC_ASSERT(WORD_SIZE % BANK_BYTES == 0, ("a lane word must hold whole bank words"))
    `STATIC_ASSERT(PIPE_OVERHEAD <= `VX_CFG_LMEM_LATENCY, ("VX_CFG_LMEM_LATENCY is below the shared-memory pipeline overhead"))
    `STATIC_ASSERT(NUM_LANES <= 32, ("the shared-memory port takes at most 32 lanes"))

    reg [TIME_W-1:0] now_r;
    reg              ext_r;
    reg [LAT_W-1:0]  debt_r;
    always @(posedge clk) begin
        if (reset) begin
            now_r <= '0;
            ext_r <= 1'b0;
        end else begin
            now_r <= now_r + TIME_W'(1);
            ext_r <= ext_valid;
        end
    end

    // Per block: the request's port cost and response time.
    wire [NUM_BLOCKS-1:0][LAT_W-1:0]     blk_cost;
    wire [NUM_BLOCKS-1:0][LAT_W-1:0]     blk_hold;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0] blk_out_mask;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0] blk_dup;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][LANE_BITS-1:0] blk_leader;
    wire [NUM_BLOCKS-1:0]                blk_reads;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0] blk_pair_lo;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][LANE_BITS-1:0] blk_partner;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_SIZE-1:0] blk_byteen;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    localparam FOLD_W = `CLOG2(NUM_LANES + 1);
    wire [NUM_BLOCKS-1:0][FOLD_W-1:0]    blk_fold_cnt;
    wire [NUM_BLOCKS-1:0][NW_WIDTH-1:0]  blk_wid;
`endif

    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_cost
        wire [NUM_LANES-1:0] lane_valid = in_if[b].req_data.mask;

        // The lowest active lane's sideband stands for the warp.
        wire [LANE_BITS-1:0] lead_idx;
        VX_priority_encoder #(
            .N (NUM_LANES)
        ) lead_enc (
            .data_in  (lane_valid),
            .index_out(lead_idx),
            `UNUSED_PIN (onehot_out),
            `UNUSED_PIN (valid_out)
        );
        mem_bus_attr_t lead_attr;
        assign lead_attr = mem_bus_attr_t'(in_if[b].req_data.user[lead_idx]);

        wire is_amo  = lead_attr.amo.amo_valid;
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        wire is_mbar = is_amo && (lead_attr.amo.amo_op inside {AMO_OP_MBAR_ARRIVE, AMO_OP_MBAR_ARRIVE_TX, AMO_OP_MBAR_EXPECT_TX,
                                                                AMO_OP_MBAR_COMPLETE_TX, AMO_OP_MBAR_TEST_WAIT, AMO_OP_MBAR_TRY_WAIT});
    `else
        wire is_mbar = 1'b0;
    `endif
        wire is_cas  = is_amo && (lead_attr.amo.amo_op == AMO_OP_CAS);
        wire is_rd   = ~in_if[b].req_data.rw;
        wire is_ld   = is_rd && ~is_amo;
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        wire counted = lead_attr.is_lacc;
    `else
        wire counted = 1'b0;
    `endif
        `UNUSED_VAR (lead_attr)

        // Access size: the widest active lane.
        wire [NUM_LANES-1:0][`CLOG2(WORD_SIZE+1)-1:0] lane_bytes;
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_bytes
            assign lane_bytes[i] = lane_valid[i] ? $countones(in_if[b].req_data.byteen[i]) : '0;
        end
        reg [`CLOG2(WORD_SIZE+1)-1:0] acc_bytes;
        always @(*) begin
            acc_bytes = '0;
            for (integer i = 0; i < NUM_LANES; ++i) begin
                if (lane_bytes[i] > acc_bytes)
                    acc_bytes = lane_bytes[i];
            end
        end
        wire wide = (acc_bytes == `CLOG2(WORD_SIZE+1)'(8));

        // Bank words the active lanes touch, and the first lane on each.
        wire [NH-1:0]                touched;
        wire [NH-1:0][HW_WIDTH-1:0]  hword;
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lanes
            for (genvar k = 0; k < HPW; ++k) begin : g_halves
                assign touched[i * HPW + k] = lane_valid[i] && (|in_if[b].req_data.byteen[i][k * BANK_BYTES +: BANK_BYTES]);
                if (HPW > 1) begin : g_hw
                    assign hword[i * HPW + k] = {in_if[b].req_data.addr[i], HPW_BITS'(k)};
                end else begin : g_hw1
                    assign hword[i * HPW + k] = in_if[b].req_data.addr[i];
                end
            end
        end
        wire [NH-1:0] first;
        for (genvar h = 0; h < NH; ++h) begin : g_first
            wire [NH-1:0] same;
            for (genvar g = 0; g < NH; ++g) begin : g_same
                if (g < h) begin : g_lt
                    assign same[g] = touched[g] && (hword[g] == hword[h]);
                end else begin : g_ge
                    assign same[g] = 1'b0;
                end
            end
            assign first[h] = touched[h] && ~(|same);
        end

        // D: distinct words per bank; passes: lane-updates per bank.
        reg [CNT_W-1:0] wavefronts, passes, nd, nt;
        always @(*) begin
            wavefronts = '0;
            passes     = '0;
            nd         = '0;
            nt         = '0;
            for (integer bk = 0; bk < BANKS; ++bk) begin
                nd = '0;
                nt = '0;
                for (integer h = 0; h < NH; ++h) begin
                    if (BANK_BITS == 0 || hword[h][`UP(BANK_BITS)-1:0] == `UP(BANK_BITS)'(bk)) begin
                        nd = nd + CNT_W'(first[h]);
                        nt = nt + CNT_W'(touched[h]);
                    end
                end
                if (nd > wavefronts)
                    wavefronts = nd;
                if (nt > passes)
                    passes = nt;
            end
            if (is_cas)
                passes = passes * CNT_W'(`VX_CFG_LMEM_CAS_PASSES);
        end

        // Lanes on one lane word: the first one serves the others' reads.
        wire [NUM_LANES-1:0]                 dup;
        wire [NUM_LANES-1:0][LANE_BITS-1:0]  leader;
        for (genvar j = 0; j < NUM_LANES; ++j) begin : g_dup
            wire [NUM_LANES-1:0] match;
            for (genvar i = 0; i < NUM_LANES; ++i) begin : g_match
                if (i < j) begin : g_lt
                    assign match[i] = lane_valid[i] && (in_if[b].req_data.addr[i] == in_if[b].req_data.addr[j]);
                end else begin : g_ge
                    assign match[i] = 1'b0;
                end
            end
            assign dup[j] = lane_valid[j] && (|match);
            VX_priority_encoder #(
                .N (NUM_LANES)
            ) leader_enc (
                .data_in  (match),
                .index_out(leader[j]),
                `UNUSED_PIN (onehot_out),
                `UNUSED_PIN (valid_out)
            );
        end
        wire [`CLOG2(NUM_LANES+1)-1:0] distinct_words = $countones(lane_valid & ~dup);

        // A load's leader reaches every bank word its lane word's lanes touch.
        wire [NUM_LANES-1:0][WORD_SIZE-1:0] word_byteen;
        for (genvar j = 0; j < NUM_LANES; ++j) begin : g_word_byteen
            reg [WORD_SIZE-1:0] be;
            always @(*) begin
                be = '0;
                for (integer i = 0; i < NUM_LANES; ++i) begin
                    if (lane_valid[i] && (in_if[b].req_data.addr[i] == in_if[b].req_data.addr[j]))
                        be = be | in_if[b].req_data.byteen[i];
                end
            end
            assign word_byteen[j] = be;
        end

        // A 32-bit read-modify-write on each half of one word goes down as one
        // access on the even bank of the pair. The m-th lane on a
        // word's high half rides with the m-th on its low half and is answered
        // from it.
        wire [NUM_LANES-1:0]                 pair_hi;
        wire [NUM_LANES-1:0]                 pair_lo;
        wire [NUM_LANES-1:0][LANE_BITS-1:0]  pair_partner;
        if (HPW == 2) begin : g_pair
            wire pair_op = is_amo && (lead_attr.amo.amo_op >= AMO_OP_ADD) && (lead_attr.amo.amo_op <= AMO_OP_MAX)
                        && (acc_bytes == `CLOG2(WORD_SIZE+1)'(4));
            wire [NUM_LANES-1:0] lo, hi;
            for (genvar i = 0; i < NUM_LANES; ++i) begin : g_half
                assign lo[i] = pair_op && lane_valid[i] && in_if[b].req_data.byteen[i][0];
                assign hi[i] = pair_op && lane_valid[i] && ~(|in_if[b].req_data.byteen[i][BANK_BYTES-1:0]);
            end
            wire [NUM_LANES-1:0][LANE_BITS:0] rank;
            for (genvar j = 0; j < NUM_LANES; ++j) begin : g_rank
                wire [NUM_LANES-1:0] earlier;
                for (genvar i = 0; i < NUM_LANES; ++i) begin : g_earlier
                    if (i < j) begin : g_lt
                        assign earlier[i] = (lo[i] == lo[j]) && (hi[i] == hi[j])
                                        && (in_if[b].req_data.addr[i] == in_if[b].req_data.addr[j]);
                    end else begin : g_ge
                        assign earlier[i] = 1'b0;
                    end
                end
                assign rank[j] = (LANE_BITS+1)'($countones(earlier));
            end
            for (genvar j = 0; j < NUM_LANES; ++j) begin : g_mate
                wire [NUM_LANES-1:0] mate;
                for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mi
                    assign mate[i] = ((hi[j] && lo[i]) || (lo[j] && hi[i])) && (rank[i] == rank[j])
                                  && (in_if[b].req_data.addr[i] == in_if[b].req_data.addr[j]);
                end
                VX_priority_encoder #(
                    .N (NUM_LANES)
                ) mate_enc (
                    .data_in  (mate),
                    .index_out(pair_partner[j]),
                    `UNUSED_PIN (onehot_out),
                    `UNUSED_PIN (valid_out)
                );
                assign pair_hi[j] = hi[j] && (|mate);
                assign pair_lo[j] = lo[j] && (|mate);
            end
        end else begin : g_no_pair
            assign pair_hi      = '0;
            assign pair_lo      = '0;
            assign pair_partner = '0;
        end

        wire [CNT_W-1:0] d1 = (wavefronts == 0) ? CNT_W'(1) : wavefronts;
        wire [CNT_W-1:0] p1 = is_cas ? CNT_W'(`VX_CFG_LMEM_CAS_PASSES) : CNT_W'(1);
        wire [CNT_W-1:0] pp = (passes < p1) ? p1 : passes;

        reg [LAT_W-1:0] cost, lat;
        always @(*) begin
            cost = LAT_W'(1);
            lat  = '0;
            if (is_amo) begin
                cost = (passes == 0) ? LAT_W'(1) : LAT_W'(passes);
                if (~is_mbar) begin
                    lat = LAT_W'(`VX_CFG_LMEM_LATENCY)
                        + (is_cas ? LAT_W'(`VX_CFG_LMEM_CAS_EXTRA) : LAT_W'(`VX_CFG_LMEM_ATOM_EXTRA))
                        + LAT_W'(`VX_CFG_LMEM_WAVEFRONT_PENALTY) * LAT_W'(pp - p1);
                end
            end else begin
                cost = LAT_W'(d1);
                if (is_rd) begin
                    lat = (wide ? LAT_W'(`VX_CFG_LMEM_LATENCY_64) : LAT_W'(`VX_CFG_LMEM_LATENCY))
                        + LAT_W'(`VX_CFG_LMEM_WAVEFRONT_PENALTY) * LAT_W'(d1 - CNT_W'(1))
                        + ((wide && distinct_words > 2) ? LAT_W'(`VX_CFG_LMEM_WIDE_OFFSET_64) : LAT_W'(0));
                end
            end
        end

        assign blk_cost[b]     = cost;
        assign blk_hold[b]     = (lat > LAT_W'(PIPE_OVERHEAD)) ? (lat - LAT_W'(PIPE_OVERHEAD)) : '0;
        assign blk_dup[b]      = is_ld ? dup : pair_hi;
        assign blk_out_mask[b] = lane_valid & ~blk_dup[b];
        assign blk_leader[b]   = is_ld ? leader : pair_partner;
        assign blk_reads[b]    = is_rd;
        assign blk_pair_lo[b]  = pair_lo;
        assign blk_partner[b]  = pair_partner;
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_blk_byteen
            assign blk_byteen[b][i] = is_ld ? word_byteen[i]
                                            : (in_if[b].req_data.byteen[i] | (pair_lo[i] ? in_if[b].req_data.byteen[pair_partner[i]] : '0));
        end
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        assign blk_fold_cnt[b] = counted ? FOLD_W'($countones(blk_dup[b])) : '0;
        if (NW_BITS != 0) begin : g_wid
            assign blk_wid[b] = lead_attr.amo.hart_id[NT_BITS +: NW_BITS];
        end else begin : g_wid0
            assign blk_wid[b] = '0;
        end
    `else
        `UNUSED_VAR (counted)
    `endif
    end

    // One warp request a cycle when the port is free, round robin over the blocks.
    wire can_issue = (debt_r == 0) && ~ext_r;
    wire [NUM_BLOCKS-1:0] blk_req;
    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_blk_req
        assign blk_req[b] = can_issue && in_if[b].req_valid && out_if[b].req_ready;
    end

    wire [BLK_BITS-1:0]   grant_idx;
    wire [NUM_BLOCKS-1:0] grant_onehot;
    wire                  grant_valid;
    VX_generic_arbiter #(
        .NUM_REQS (NUM_BLOCKS),
        .TYPE     ("R")
    ) issue_arb (
        .clk         (clk),
        .reset       (reset),
        .requests    (blk_req),
        .grant_index (grant_idx),
        .grant_onehot(grant_onehot),
        .grant_valid (grant_valid),
        .grant_ready (1'b1)
    );

    always @(posedge clk) begin
        if (reset) begin
            debt_r <= '0;
        end else if (grant_valid) begin
            debt_r <= blk_cost[grant_idx] - LAT_W'(1);
        end else if (debt_r != 0 && ~ext_r) begin
            debt_r <= debt_r - LAT_W'(1);
        end
    end

    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_blocks
        wire issue = grant_onehot[b] && grant_valid;

        assign out_if[b].req_valid        = issue;
        assign out_if[b].req_data.mask    = blk_out_mask[b];
        assign out_if[b].req_data.rw      = in_if[b].req_data.rw;
        assign out_if[b].req_data.addr    = in_if[b].req_data.addr;
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_merge
            wire [LANE_BITS-1:0] mate = blk_partner[b][i];
            mem_bus_attr_t attr;
            always @(*) begin
                attr = mem_bus_attr_t'(in_if[b].req_data.user[i]);
                attr.amo.amo_pair = blk_pair_lo[b][i];
            end
            assign out_if[b].req_data.data[i]   = blk_pair_lo[b][i]
                ? {in_if[b].req_data.data[mate][WORD_SIZE*8-1:BANK_BYTES*8], in_if[b].req_data.data[i][BANK_BYTES*8-1:0]}
                : in_if[b].req_data.data[i];
            assign out_if[b].req_data.byteen[i] = blk_byteen[b][i];
            assign out_if[b].req_data.user[i]   = attr;
        end
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        assign fold_valid[b] = issue && (blk_fold_cnt[b] != '0);
        assign fold_wid[b]   = blk_wid[b];
        assign fold_cnt[b]   = blk_fold_cnt[b];
    `endif
        assign out_if[b].req_data.tag     = in_if[b].req_data.tag;
        assign in_if[b].req_ready         = issue;

        // Per in-flight read, by its tag: its response time, merged lanes and
        // the lanes the banks still owe. A bank conflict below returns a
        // request in parts; the parts before the last go on at once and the
        // last one at the request's time.
        wire [TAG_SEL_BITS-1:0] req_tsel = in_if[b].req_data.tag[TAG_SEL_BITS-1:0];
        wire [TAG_SEL_BITS-1:0] rsp_tsel = out_if[b].rsp_data.tag[TAG_SEL_BITS-1:0];
        reg [NUM_TAGS-1:0][TIME_W-1:0]  due_tbl;
        reg [NUM_TAGS-1:0][NUM_LANES-1:0] dup_tbl;
        reg [NUM_TAGS-1:0][NUM_LANES-1:0][LANE_BITS-1:0] leader_tbl;
        reg [NUM_TAGS-1:0][NUM_LANES-1:0] owed_tbl;

        wire fifo_full, fifo_empty;
        wire push = out_if[b].rsp_valid && ~fifo_full;
        wire rsp_last = ~(|(owed_tbl[rsp_tsel] & ~out_if[b].rsp_data.mask));
        wire [TIME_W-1:0] rsp_due = rsp_last ? due_tbl[rsp_tsel] : now_r;

        always @(posedge clk) begin
            if (issue && blk_reads[b]) begin
                due_tbl[req_tsel]    <= now_r + TIME_W'(blk_hold[b]);
                dup_tbl[req_tsel]    <= blk_dup[b];
                leader_tbl[req_tsel] <= blk_leader[b];
                owed_tbl[req_tsel]   <= blk_out_mask[b];
            end
            if (push) begin
                owed_tbl[rsp_tsel]   <= owed_tbl[rsp_tsel] & ~out_if[b].rsp_data.mask;
            end
        end

        wire [NUM_LANES-1:0]                     rsp_dup    = dup_tbl[rsp_tsel];
        wire [NUM_LANES-1:0][LANE_BITS-1:0]      rsp_leader = leader_tbl[rsp_tsel];
        wire [NUM_LANES-1:0]                     rsp_mask;
        wire [NUM_LANES-1:0][WORD_SIZE*8-1:0]    rsp_data;
        for (genvar j = 0; j < NUM_LANES; ++j) begin : g_expand
            wire from_leader = rsp_dup[j] && out_if[b].rsp_data.mask[rsp_leader[j]];
            assign rsp_mask[j] = out_if[b].rsp_data.mask[j] || from_leader;
            assign rsp_data[j] = from_leader ? out_if[b].rsp_data.data[rsp_leader[j]] : out_if[b].rsp_data.data[j];
        end

        wire [RSP_DATAW-1:0] fifo_out;
        wire [NUM_LANES-1:0]                  held_mask;
        wire [NUM_LANES-1:0][WORD_SIZE*8-1:0] held_data;
        wire [TAG_WIDTH-1:0]                  held_tag;
        wire [TIME_W-1:0]                     held_due;
        assign {held_mask, held_data, held_tag, held_due} = fifo_out;

        wire held_ready = ~fifo_empty && ($signed(now_r - held_due) >= 0);
        wire pop  = held_ready && in_if[b].rsp_ready;

        VX_fifo_queue #(
            .DATAW  (RSP_DATAW),
            .DEPTH  (RSP_DEPTH),
            .LUTRAM (1)
        ) rsp_hold (
            .clk      (clk),
            .reset    (reset),
            .push     (push),
            .pop      (pop),
            .data_in  ({rsp_mask, rsp_data, out_if[b].rsp_data.tag, rsp_due}),
            .data_out (fifo_out),
            .empty    (fifo_empty),
            .full     (fifo_full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

        assign out_if[b].rsp_ready    = ~fifo_full;
        assign in_if[b].rsp_valid     = held_ready;
        assign in_if[b].rsp_data.mask = held_mask;
        assign in_if[b].rsp_data.data = held_data;
        assign in_if[b].rsp_data.tag  = held_tag;
    end

endmodule

`endif
