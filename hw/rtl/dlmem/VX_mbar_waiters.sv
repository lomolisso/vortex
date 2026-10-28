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

// Gated on the knob so the file preprocesses to nothing with it off: VX_define.vh
// itself emits DPI imports.
`ifdef VX_CFG_EXT_MBAR_ENABLE

`include "VX_define.vh"

// The warps of one core suspended in an mbarrier try_wait. An object lives in
// one bank, which orders every op on it: a phase flip either precedes a lane's
// read (which then sees the phase complete) or follows that lane's insert, so no
// wake-up is lost. A warp's retire comes after all of its lanes' inserts.

module VX_mbar_waiters import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BANKS      = `VX_CFG_LMEM_NUM_BANKS,
    parameter NUM_RETIRES    = `VX_CFG_NUM_LSU_BLOCKS,
    parameter ROW_WIDTH      = 1,
    parameter SUSPEND_CYCLES = `VX_CFG_MBAR_SUSPEND_CYCLES
) (
    input wire                                  clk,
    input wire                                  reset,

    // Per bank, an insert (flip = 0) or a flip (flip = 1) of the word at row
    input wire [NUM_BANKS-1:0]                  evt_valid,
    input wire [NUM_BANKS-1:0]                  evt_flip,
    input wire [NUM_BANKS-1:0][NW_WIDTH-1:0]    evt_wid,
    input wire [NUM_BANKS-1:0][ROW_WIDTH-1:0]   evt_row,

    // Per LSU block, the last response of a try_wait has left the block
    input wire [NUM_RETIRES-1:0]                retire_valid,
    input wire [NUM_RETIRES-1:0][NW_WIDTH-1:0]  retire_wid,

    output wire                                 unlock_valid,
    output wire [`VX_CFG_NUM_WARPS-1:0]         unlock_mask
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam BANK_WIDTH  = `UP(`CLOG2(NUM_BANKS));
    localparam TIMER_WIDTH = `UP(`CLOG2(SUSPEND_CYCLES + 1));

    `STATIC_ASSERT (`VX_CFG_MBAR_SUSPEND_CYCLES < STALL_TIMEOUT / 4,
        ("VX_CFG_MBAR_SUSPEND_CYCLES must stay well below the scheduler stall timeout"))

    reg [`VX_CFG_NUM_WARPS-1:0]                 valid_r;
    reg [`VX_CFG_NUM_WARPS-1:0][BANK_WIDTH-1:0] bank_r;
    reg [`VX_CFG_NUM_WARPS-1:0][ROW_WIDTH-1:0]  row_r;
    reg [`VX_CFG_NUM_WARPS-1:0]                 woke_r;
    reg [`VX_CFG_NUM_WARPS-1:0]                 retired_r;
    reg [`VX_CFG_NUM_WARPS-1:0][TIMER_WIDTH-1:0] timer_r;

    reg                         unlock_valid_r;
    reg [`VX_CFG_NUM_WARPS-1:0] unlock_mask_r;

    wire [`VX_CFG_NUM_WARPS-1:0] release_n;
    wire [`VX_CFG_NUM_WARPS-1:0] timeout_n;
    wire [`VX_CFG_NUM_WARPS-1:0] retire_w;

    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_warps
        reg                  ins;
        reg [BANK_WIDTH-1:0] ins_bank;
        reg [ROW_WIDTH-1:0]  ins_row;
        always @(*) begin
            ins      = 1'b0;
            ins_bank = bank_r[w];
            ins_row  = row_r[w];
            for (integer b = 0; b < NUM_BANKS; ++b) begin
                if (evt_valid[b] && ~evt_flip[b] && (evt_wid[b] == NW_WIDTH'(w))) begin
                    ins      = 1'b1;
                    ins_bank = BANK_WIDTH'(b);
                    ins_row  = evt_row[b];
                end
            end
        end

        wire valid_i = valid_r[w] || ins;

        reg flip_hit;
        always @(*) begin
            flip_hit = 1'b0;
            for (integer b = 0; b < NUM_BANKS; ++b) begin
                if (evt_valid[b] && evt_flip[b] && (ins_bank == BANK_WIDTH'(b)) && (evt_row[b] == ins_row)) begin
                    flip_hit = 1'b1;
                end
            end
        end

        wire woke_f = woke_r[w] || (valid_i && flip_hit);

        wire [NUM_RETIRES-1:0] ret_hit;
        for (genvar i = 0; i < NUM_RETIRES; ++i) begin : g_ret
            assign ret_hit[i] = retire_valid[i] && (retire_wid[i] == NW_WIDTH'(w));
        end
        wire ret = (| ret_hit);

        // A retire releases its warp at once when no lane is waiting or a flip already came.
        wire ret_release = ret && (~valid_i || woke_f);
        wire retired_n   = retired_r[w] || (ret && ~ret_release);
        wire [TIMER_WIDTH-1:0] timer_n = (ret && ~ret_release) ? TIMER_WIDTH'(SUSPEND_CYCLES) : timer_r[w];

        wire wake    = valid_i && retired_n && woke_f;
        wire timeout = valid_i && retired_n && ~woke_f && (timer_n == '0);

        assign release_n[w] = ret_release || wake || timeout;
        assign timeout_n[w] = timeout;
        assign retire_w[w]  = ret;

        always @(posedge clk) begin
            if (reset) begin
                valid_r[w]   <= 1'b0;
                woke_r[w]    <= 1'b0;
                retired_r[w] <= 1'b0;
            end else if (release_n[w]) begin
                valid_r[w]   <= 1'b0;
                woke_r[w]    <= 1'b0;
                retired_r[w] <= 1'b0;
            end else begin
                valid_r[w]   <= valid_i;
                woke_r[w]    <= woke_f;
                retired_r[w] <= retired_n;
            end
            bank_r[w]  <= ins_bank;
            row_r[w]   <= ins_row;
            timer_r[w] <= (valid_i && retired_n && (timer_n != '0)) ? (timer_n - TIMER_WIDTH'(1)) : timer_n;
        end

        `RUNTIME_ASSERT (~(ins && retired_r[w]),
            ("%t: *** %s mbarrier waiter insert after retire: wid=%0d", $time, INSTANCE_ID, w))
        `RUNTIME_ASSERT (~(ret && retired_r[w]),
            ("%t: *** %s mbarrier waiter retired twice: wid=%0d", $time, INSTANCE_ID, w))
        `RUNTIME_ASSERT ($countones(ret_hit) <= 1,
            ("%t: *** %s mbarrier waiter retires from two LSU blocks: wid=%0d", $time, INSTANCE_ID, w))
    end

    always @(posedge clk) begin
        if (reset) begin
            unlock_valid_r <= 1'b0;
        end else begin
            unlock_valid_r <= (| release_n);
        end
        unlock_mask_r <= release_n;
    end

    assign unlock_valid = unlock_valid_r;
    assign unlock_mask  = unlock_mask_r;

`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (~reset) begin
            for (integer b = 0; b < NUM_BANKS; ++b) begin
                if (evt_valid[b] && ~evt_flip[b]) begin
                    `TRACE(2, ("%t: %s mbar-insert: bank=%0d, wid=%0d, row=0x%0h\n", $time, INSTANCE_ID, b, evt_wid[b], evt_row[b]))
                end
                if (evt_valid[b] && evt_flip[b]) begin
                    `TRACE(2, ("%t: %s mbar-flip: bank=%0d, row=0x%0h\n", $time, INSTANCE_ID, b, evt_row[b]))
                end
            end
            for (integer w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin
                if (retire_w[w]) begin
                    `TRACE(2, ("%t: %s mbar-retire: wid=%0d\n", $time, INSTANCE_ID, w))
                end
                if (timeout_n[w]) begin
                    `TRACE(2, ("%t: %s mbar-timeout: wid=%0d\n", $time, INSTANCE_ID, w))
                end
            end
            if (unlock_valid_r) begin
                `TRACE(2, ("%t: %s mbar-unlock: mask=0x%0h\n", $time, INSTANCE_ID, unlock_mask_r))
            end
        end
    end
`endif
    `UNUSED_VAR (timeout_n)
    `UNUSED_VAR (retire_w)

endmodule

`endif
