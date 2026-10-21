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

// Gated on the knobs, not TCU_WG_GROUPS_ENABLE, so the file preprocesses to
// nothing with them off: VX_define.vh itself emits DPI imports.
`ifdef VX_CFG_EXT_MBAR_ENABLE
`ifdef VX_CFG_TCU_WGMMA_ENABLE

`include "VX_define.vh"

// wgmma commit/wait groups of the warps of one TCU block.
//
// Per warp: `issued` counts WGMMAs whose first compute uop was accepted,
// `done` those whose last compute result retired. commit_group pushes
// `issued` into a FIFO of group boundaries; a group is pending while its
// boundary exceeds `done`. wait_group N releases its warp once at most N
// groups are pending. A warp always uses the same block and its uops arrive
// in order, so the counts of one block are complete for its warps.
// Each retire, wait and commit drops the warp's boundaries at or below
// `done`, so the ones held stay within [done, issued].
//
// Timing: satisfied in cycle T (the wait's accept or the satisfying retire)
// -> unlock pulse at T+1 -> warp schedulable at T+2. One pulse per cycle;
// simultaneous releases go out lowest warp first.

module VX_tcu_groups import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter BLOCK_ID = 0
) (
    input wire                  clk,
    input wire                  reset,

    input wire                  issue_valid,
    input wire [NW_WIDTH-1:0]   issue_wid,

    input wire                  done_valid,
    input wire [NW_WIDTH-1:0]   done_wid,

    // an accepted commit_group or wait_group
    input wire                  req_valid,
    input wire                  req_is_wait,
    input wire [NW_WIDTH-1:0]   req_wid,
    input wire [2:0]            req_n,

    VX_sched_unlock_if.master   unlock_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam PW     = PER_ISSUE_WARPS;
    localparam DEPTH  = 8;
    localparam PTR_W  = `CLOG2(DEPTH);
    localparam SIZE_W = `CLOG2(DEPTH + 1);
    // The held boundaries and `issued` stay within [done, done + WGMMAs in
    // flight], far below 2^(CNT_W-1), so the modular compare is exact.
    localparam CNT_W  = 6;

    // The warps of one block are those of one issue slice.
    `STATIC_ASSERT (`VX_CFG_NUM_TCU_BLOCKS == `VX_CFG_ISSUE_WIDTH,
        ("wgmma groups need one TCU block per issue slice"))

    reg [CNT_W-1:0]  issued_r [PW];
    reg [CNT_W-1:0]  done_r   [PW];
    reg [CNT_W-1:0]  fifo_r   [PW][DEPTH];
    reg [PTR_W-1:0]  head_r   [PW];
    reg [SIZE_W-1:0] size_r   [PW];
    reg [PW-1:0]     waiting_r;
    reg [2:0]        wait_n_r [PW];
    reg [PW-1:0]     ready_r;
    reg              unlock_valid_r;
    reg [NW_WIDTH-1:0] unlock_wid_r;

    wire [ISSUE_WIS_W-1:0] issue_wis = wid_to_wis(issue_wid);
    wire [ISSUE_WIS_W-1:0] done_wis  = wid_to_wis(done_wid);
    wire [ISSUE_WIS_W-1:0] req_wis   = wid_to_wis(req_wid);

    // boundaries of warp w above d
    function automatic logic [SIZE_W-1:0] pending(input logic [ISSUE_WIS_W-1:0] w, input logic [CNT_W-1:0] d);
        logic [CNT_W-1:0] diff;
        pending = '0;
        for (integer j = 0; j < DEPTH; ++j) begin
            diff = fifo_r[w][PTR_W'(head_r[w] + PTR_W'(j))] - d;
            if ((SIZE_W'(j) < size_r[w]) && (diff != '0) && ~diff[CNT_W-1]) begin
                pending = pending + SIZE_W'(1);
            end
        end
    endfunction

    wire [CNT_W-1:0] done_inc = done_r[done_wis] + CNT_W'(1);
    wire [CNT_W-1:0] req_done_n = done_r[req_wis] + CNT_W'(done_valid && (done_wis == req_wis));

    wire [SIZE_W-1:0] req_pending  = pending(req_wis, req_is_wait ? req_done_n : done_r[req_wis]);
    wire [SIZE_W-1:0] done_pending = pending(done_wis, done_inc);

    // Boundaries the request's warp keeps: a same-warp retire drops its own.
    wire [SIZE_W-1:0] req_live = (done_valid && (done_wis == req_wis)) ? done_pending : req_pending;

    wire req_wait = req_valid && req_is_wait;
    wire req_commit = req_valid && ~req_is_wait;
    wire sat_req  = req_wait && (req_pending <= SIZE_W'(req_n));
    wire sat_done = done_valid && waiting_r[done_wis] && (done_pending <= SIZE_W'(wait_n_r[done_wis]));

    reg [PW-1:0] ready_n;
    reg [PW-1:0] pick;
    always @(*) begin
        ready_n = ready_r;
        if (sat_req)  ready_n[req_wis]  = 1'b1;
        if (sat_done) ready_n[done_wis] = 1'b1;
        pick = '0;
        for (integer w = PW-1; w >= 0; --w) begin
            if (ready_n[w]) pick = PW'(1) << w;
        end
    end

    wire [ISSUE_WIS_W-1:0] pick_wis;
    VX_onehot_encoder #(
        .N (PW)
    ) pick_enc (
        .data_in  (pick),
        .data_out (pick_wis),
        `UNUSED_PIN (valid_out)
    );

    always @(posedge clk) begin
        if (reset) begin
            for (integer w = 0; w < PW; ++w) begin
                issued_r[w] <= '0;
                done_r[w]   <= '0;
                head_r[w]   <= '0;
                size_r[w]   <= '0;
            end
            waiting_r      <= '0;
            ready_r        <= '0;
            unlock_valid_r <= 1'b0;
        end else begin
            if (issue_valid) begin
                issued_r[issue_wis] <= issued_r[issue_wis] + CNT_W'(1);
            end
            if (done_valid) begin
                done_r[done_wis] <= done_inc;
                head_r[done_wis] <= head_r[done_wis] + PTR_W'(size_r[done_wis] - done_pending);
                size_r[done_wis] <= done_pending;
            end
            if (req_valid) begin
                // drop the retired boundaries; a commit then pushes
                head_r[req_wis] <= head_r[req_wis] + PTR_W'(size_r[req_wis] - req_live);
                size_r[req_wis] <= req_live + SIZE_W'(req_commit);
            end
            if (req_wait && ~sat_req) begin
                waiting_r[req_wis] <= 1'b1;
            end
            if (sat_done) begin
                waiting_r[done_wis] <= 1'b0;
            end
            ready_r        <= ready_n & ~pick;
            unlock_valid_r <= (pick != '0);
        end
        if (req_commit) begin
            fifo_r[req_wis][PTR_W'(head_r[req_wis] + PTR_W'(size_r[req_wis]))] <= issued_r[req_wis];
        end
        if (req_wait) begin
            wait_n_r[req_wis] <= req_n;
        end
        unlock_wid_r <= wis_to_wid(pick_wis, ISSUE_ISW_W'(BLOCK_ID));
    end

    assign unlock_if.valid = unlock_valid_r;
    assign unlock_if.wid   = unlock_wid_r;

    `RUNTIME_ASSERT(~req_commit || (req_pending != SIZE_W'(DEPTH)),
        ("%t: %s wgmma group overflow: wid=%0d", $time, INSTANCE_ID, req_wid))
    `RUNTIME_ASSERT(~req_wait || ~waiting_r[req_wis],
        ("%t: %s second wgmma wait of a warp: wid=%0d", $time, INSTANCE_ID, req_wid))
    `RUNTIME_ASSERT(~done_valid || (done_r[done_wis] != issued_r[done_wis]),
        ("%t: %s wgmma retire without an issue: wid=%0d", $time, INSTANCE_ID, done_wid))
    `RUNTIME_ASSERT(~issue_valid || ((issued_r[issue_wis] - done_r[issue_wis]) < CNT_W'(1 << (CNT_W-1))),
        ("%t: %s too many wgmmas in flight: wid=%0d", $time, INSTANCE_ID, issue_wid))
    `RUNTIME_ASSERT(~req_valid || (wid_to_isw(req_wid) == ISSUE_ISW_W'(BLOCK_ID)),
        ("%t: %s wgmma group op of another block: wid=%0d", $time, INSTANCE_ID, req_wid))

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (~reset && req_commit) begin
            `TRACE(2, ("%t: %s commit_group: wid=%0d, issued=%0d, done=%0d, pending=%0d\n",
                $time, INSTANCE_ID, req_wid, issued_r[req_wis], done_r[req_wis], req_pending + SIZE_W'(1)))
        end
        if (~reset && req_wait) begin
            `TRACE(2, ("%t: %s wait_group: wid=%0d, n=%0d, pending=%0d\n",
                $time, INSTANCE_ID, req_wid, req_n, req_pending))
        end
        if (~reset && done_valid) begin
            `TRACE(2, ("%t: %s wgmma-retire: wid=%0d, done=%0d\n", $time, INSTANCE_ID, done_wid, done_inc))
        end
        if (~reset && unlock_valid_r) begin
            `TRACE(2, ("%t: %s wait-unlock: wid=%0d\n", $time, INSTANCE_ID, unlock_wid_r))
        end
    end
`endif

endmodule

`endif
`endif
