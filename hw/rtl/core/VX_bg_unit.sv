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
`ifdef VX_CFG_EXT_DXA_ENABLE
`ifdef VX_CFG_EXT_MBAR_ENABLE

`include "VX_define.vh"

// Bulk async-groups of the DXA stores and reductions, per warp.
//
// Each warp has GROUPS slots of {read, write} pending-operation counts,
// indexed by group sequence number: the open group's seq and the ncommit
// committed groups before it, the oldest (the head) at seq - ncommit. An
// operation's issue event counts it in the open group's slot, and its
// completions from the DXA, always accepted, uncount it in the slot of the
// seq it carried. A commit closes the open group, even an empty one. The head
// retires, one group per cycle, once both its counts are 0.
//
// wait_group N releases its warp once every committed group but the N newest
// has no write pending; wait_group.read N once none has a read pending.
//
// An operation stalls at issue while its open group has 63 writes pending or
// 8 groups are committed; a commit stalls while 8 are. A group's read is done
// no later than its write, so the read count never exceeds the write count.
//
// Timing: events and completions at T change the state seen at T+1; the
// stalls, the open seq and the pending flags are functions of the state
// alone; a head that reaches zero at T+1 retires into the state of T+2. A
// wait satisfied by the state of cycle T (its own event's cycle included)
// pulses the unlock at T+1, one warp per cycle, lowest first.

module VX_bg_unit import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input  wire                             clk,
    input  wire                             reset,

    // one per DXA instruction of a warp, at its send
    input  wire                             evt_valid,
    input  dxa_bg_evt_t                     evt,

    // operation progress from the socket's DXA
    input  wire                             compl_valid,
    input  dxa_bg_compl_t                   compl,

    // per warp, to the DXA issue
    output wire [`VX_CFG_NUM_WARPS-1:0]     stall_op,
    output wire [`VX_CFG_NUM_WARPS-1:0]     stall_commit,
    output wire [`VX_CFG_NUM_WARPS-1:0][DXA_BG_SEQ_W-1:0] seq,

    // per warp, an operation's source not yet read (the exit waits for it);
    // any warp's write not yet performed (the core stays busy)
    output wire [`VX_CFG_NUM_WARPS-1:0]     rd_pending,
    output wire                             wr_pending,

    VX_sched_unlock_if.master               unlock_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam NW      = `VX_CFG_NUM_WARPS;
    localparam GROUPS  = DXA_BG_GROUPS;
    localparam SEQ_W   = DXA_BG_SEQ_W;
    localparam NC_W    = SEQ_W + 1;
    localparam CNT_W   = 6;
    localparam [CNT_W-1:0] CNT_MAX = '1;

    `STATIC_ASSERT (GROUPS == (1 << SEQ_W), ("bulk groups must fill the sequence space"))

    reg [SEQ_W-1:0]  seq_r     [NW];
    reg [NC_W-1:0]   ncommit_r [NW];
    reg [CNT_W-1:0]  rd_cnt_r  [NW][GROUPS];
    reg [CNT_W-1:0]  wr_cnt_r  [NW][GROUPS];

    reg [NW-1:0]     waiting_r, ready_r;
    reg [4:0]        wait_n_r  [NW];
    reg [NW-1:0]     wait_rd_r;

    reg              unlock_valid_r;
    reg [NW_WIDTH-1:0] unlock_wid_r;

    wire is_op     = evt_valid && (evt.kind == DXA_BG_KIND_OP);
    wire is_commit = evt_valid && (evt.kind == DXA_BG_KIND_COMMIT);
    wire is_wait   = evt_valid && ((evt.kind == DXA_BG_KIND_WAIT) || (evt.kind == DXA_BG_KIND_WAIT_READ));
    wire evt_rd    = (evt.kind == DXA_BG_KIND_WAIT_READ);

    // ---------------------------------------------------------------------
    // Per warp: head retire and wait checks on the registered state

    reg [NW-1:0] retire;
    always @(*) begin
        for (integer w = 0; w < NW; ++w) begin
            logic [SEQ_W-1:0] head;
            head = seq_r[w] - SEQ_W'(ncommit_r[w]);
            retire[w] = (ncommit_r[w] != '0) && (rd_cnt_r[w][head] == '0) && (wr_cnt_r[w][head] == '0);
        end
    end

    // Whether warp w's committed groups but the n newest have no read (rd) or write pending.
    function automatic logic groups_done(input logic [NW_WIDTH-1:0] w, input logic [4:0] n, input logic rd);
        groups_done = 1'b1;
        for (integer i = 0; i < GROUPS; ++i) begin
            logic [SEQ_W-1:0] s;
            s = seq_r[w] - SEQ_W'(ncommit_r[w]) + SEQ_W'(i);
            if ((32'(i) + 32'(n)) < 32'(ncommit_r[w])) begin
                if ((rd ? rd_cnt_r[w][s] : wr_cnt_r[w][s]) != '0) begin
                    groups_done = 1'b0;
                end
            end
        end
    endfunction

    reg [NW-1:0] ready_n, waiting_n, pick;
    always @(*) begin
        ready_n   = ready_r;
        waiting_n = waiting_r;
        for (integer w = 0; w < NW; ++w) begin
            if (waiting_r[w] && groups_done(NW_WIDTH'(w), wait_n_r[w], wait_rd_r[w])) begin
                ready_n[w]   = 1'b1;
                waiting_n[w] = 1'b0;
            end
        end
        if (is_wait) begin
            if (groups_done(evt.wid, evt.n, evt_rd)) begin
                ready_n[evt.wid] = 1'b1;
            end else begin
                waiting_n[evt.wid] = 1'b1;
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
    // State update

    always @(posedge clk) begin
        if (reset) begin
            for (integer w = 0; w < NW; ++w) begin
                seq_r[w]     <= '0;
                ncommit_r[w] <= '0;
                for (integer g = 0; g < GROUPS; ++g) begin
                    rd_cnt_r[w][g] <= '0;
                    wr_cnt_r[w][g] <= '0;
                end
            end
            waiting_r      <= '0;
            ready_r        <= '0;
            unlock_valid_r <= 1'b0;
        end else begin
            for (integer w = 0; w < NW; ++w) begin
                logic ev_w;
                ev_w = (evt.wid == NW_WIDTH'(w));
                for (integer g = 0; g < GROUPS; ++g) begin
                    logic inc, c_rd, c_wr;
                    inc  = is_op && ev_w && (seq_r[w] == SEQ_W'(g));
                    c_rd = compl_valid && (compl.wid == NW_WIDTH'(w)) && (compl.seq == SEQ_W'(g)) && compl.read_done;
                    c_wr = compl_valid && (compl.wid == NW_WIDTH'(w)) && (compl.seq == SEQ_W'(g)) && compl.write_done;
                    rd_cnt_r[w][g] <= rd_cnt_r[w][g] + CNT_W'(inc) - CNT_W'(c_rd);
                    wr_cnt_r[w][g] <= wr_cnt_r[w][g] + CNT_W'(inc) - CNT_W'(c_wr);
                end
                if (is_commit && ev_w) begin
                    seq_r[w] <= seq_r[w] + SEQ_W'(1);
                end
                ncommit_r[w] <= ncommit_r[w] + NC_W'(is_commit && ev_w) - NC_W'(retire[w]);
            end
            waiting_r      <= waiting_n;
            ready_r        <= ready_n & ~pick;
            unlock_valid_r <= (pick != '0);
        end
        if (is_wait) begin
            wait_n_r[evt.wid]  <= evt.n;
            wait_rd_r[evt.wid] <= evt_rd;
        end
        unlock_wid_r <= pick_wid;
    end

    wire [NW-1:0] wr_pending_w;
    for (genvar w = 0; w < NW; ++w) begin : g_out
        assign stall_op[w]     = (wr_cnt_r[w][seq_r[w]] == CNT_MAX) || (ncommit_r[w] == NC_W'(GROUPS));
        assign stall_commit[w] = (ncommit_r[w] == NC_W'(GROUPS));
        assign seq[w]          = seq_r[w];
        wire [GROUPS-1:0] rd_nz, wr_nz;
        for (genvar g = 0; g < GROUPS; ++g) begin : g_nz
            assign rd_nz[g] = (rd_cnt_r[w][g] != '0);
            assign wr_nz[g] = (wr_cnt_r[w][g] != '0);
        end
        assign rd_pending[w]   = (| rd_nz);
        assign wr_pending_w[w] = (| wr_nz);
    end
    assign wr_pending = (| wr_pending_w);

    assign unlock_if.valid = unlock_valid_r;
    assign unlock_if.wid   = unlock_wid_r;

    // ---------------------------------------------------------------------
    // Checks

    `RUNTIME_ASSERT (~(is_op && stall_op[evt.wid]),
        ("%t: *** %s bulk_group operation past its stall: wid=%0d", $time, INSTANCE_ID, evt.wid))
    `RUNTIME_ASSERT (~(is_commit && stall_commit[evt.wid]),
        ("%t: *** %s bulk commit_group past its stall: wid=%0d", $time, INSTANCE_ID, evt.wid))
    `RUNTIME_ASSERT (~(is_wait && (waiting_r[evt.wid] || ready_r[evt.wid])),
        ("%t: *** %s second bulk wait_group of a warp: wid=%0d", $time, INSTANCE_ID, evt.wid))
    `RUNTIME_ASSERT (~(is_op && (rd_cnt_r[evt.wid][seq_r[evt.wid]] == CNT_MAX)),
        ("%t: *** %s bulk group read count overflow: wid=%0d", $time, INSTANCE_ID, evt.wid))
    `RUNTIME_ASSERT (~(compl_valid && compl.read_done && (rd_cnt_r[compl.wid][compl.seq] == '0)),
        ("%t: *** %s bulk group read completion underflow: wid=%0d seq=%0d", $time, INSTANCE_ID, compl.wid, compl.seq))
    `RUNTIME_ASSERT (~(compl_valid && compl.write_done && (wr_cnt_r[compl.wid][compl.seq] == '0)),
        ("%t: *** %s bulk group write completion underflow: wid=%0d seq=%0d", $time, INSTANCE_ID, compl.wid, compl.seq))
    `RUNTIME_ASSERT (~(compl_valid && ~compl.read_done && ~compl.write_done),
        ("%t: *** %s empty bulk group completion: wid=%0d seq=%0d", $time, INSTANCE_ID, compl.wid, compl.seq))

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && evt_valid) begin
            `TRACE(2, ("%t: %s bg-evt: wid=%0d, kind=%0d, n=%0d, seq=%0d, ncommit=%0d\n",
                $time, INSTANCE_ID, evt.wid, evt.kind, evt.n, seq_r[evt.wid], ncommit_r[evt.wid]))
        end
        if (~reset && compl_valid) begin
            `TRACE(2, ("%t: %s bg-compl: wid=%0d, seq=%0d, rd=%b, wr=%b\n",
                $time, INSTANCE_ID, compl.wid, compl.seq, compl.read_done, compl.write_done))
        end
        if (~reset && unlock_valid_r) begin
            `TRACE(2, ("%t: %s bg-unlock: wid=%0d\n", $time, INSTANCE_ID, unlock_wid_r))
        end
    end
`endif

endmodule

`endif
`endif
