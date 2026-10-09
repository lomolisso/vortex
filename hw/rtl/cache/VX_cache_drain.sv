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

`include "VX_cache_define.vh"

// Bank drain for a fence at the shared level: drain_end pulses once the bank
// has retired every request it held at drain_begin, while it keeps accepting
// new ones. Each step waits only for a snapshot of older work, so traffic
// arriving meanwhile cannot hold the drain off:
//   PIPE: the requests in the pipeline commit (PIPE_DEPTH advancing cycles);
//   MSHR: the misses they left in the MSHR are filled and leave it;
//   PIPE: the replays of those misses commit;
//   AMO : the atomic engine settles and its writebacks enter the pipeline (it
//         admits no core request while busy, so new work cannot keep it so);
//   PIPE: those writebacks commit;
//   MREQ: the memory requests queued by then are sent.
// A bank found empty ends the drain at once.
module VX_cache_drain import VX_gpu_pkg::*; #(
    parameter MSHR_SIZE  = 1,
    parameter PIPE_DEPTH = 2,
    parameter MREQ_SIZEW = 1
) (
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  drain_begin,
    output wire                  drain_end,
    input  wire                  bank_empty,
    input  wire                  pipe_advance,
    input  wire [MSHR_SIZE-1:0]  mshr_valid,
    input  wire [MSHR_SIZE-1:0]  mshr_free,
    input  wire                  amo_busy,
    input  wire [MREQ_SIZEW-1:0] mreq_size,
    input  wire                  mreq_pop
);
    localparam STATE_IDLE = 0;
    localparam STATE_PIPE = 1;
    localparam STATE_MSHR = 2;
    localparam STATE_AMO  = 3;
    localparam STATE_MREQ = 4;
    localparam STATE_DONE = 5;

    localparam CNT_W = `CLOG2(PIPE_DEPTH + 1);

    reg [2:0] state, state_n;
    reg [1:0] step, step_n;   // which PIPE wait: after begin, after MSHR, after AMO
    reg [CNT_W-1:0] count, count_n;
    reg [MSHR_SIZE-1:0] mshr_wait, mshr_wait_n;
    reg [MREQ_SIZEW-1:0] mreq_wait, mreq_wait_n;

    always @(*) begin
        state_n     = state;
        step_n      = step;
        count_n     = count;
        mshr_wait_n = mshr_wait & ~mshr_free;
        mreq_wait_n = mreq_wait - MREQ_SIZEW'(mreq_pop && (mreq_wait != 0));
        case (state)
        STATE_PIPE: begin
            if (pipe_advance) begin
                count_n = count + CNT_W'(1);
                if (count == CNT_W'(PIPE_DEPTH - 1)) begin
                    count_n = '0;
                    case (step)
                    2'd0: begin
                        state_n     = STATE_MSHR;
                        mshr_wait_n = mshr_valid & ~mshr_free;
                    end
                    2'd1: state_n = STATE_AMO;
                    default: begin
                        state_n     = STATE_MREQ;
                        mreq_wait_n = mreq_size - MREQ_SIZEW'(mreq_pop);
                    end
                    endcase
                end
            end
        end
        STATE_MSHR: begin
            if (mshr_wait_n == '0) begin
                state_n = STATE_PIPE;
                step_n  = 2'd1;
            end
        end
        STATE_AMO: begin
            if (~amo_busy) begin
                state_n = STATE_PIPE;
                step_n  = 2'd2;
            end
        end
        STATE_MREQ: begin
            if (mreq_wait_n == '0) begin
                state_n = STATE_DONE;
            end
        end
        STATE_DONE: begin
            state_n = STATE_IDLE;
        end
        default: begin
            if (drain_begin) begin
                state_n = STATE_PIPE;
                step_n  = 2'd0;
                count_n = '0;
            end
        end
        endcase
        if ((state != STATE_IDLE) && (state != STATE_DONE) && bank_empty) begin
            state_n = STATE_DONE;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= STATE_IDLE;
        end else begin
            state <= state_n;
        end
        step      <= step_n;
        count     <= count_n;
        mshr_wait <= mshr_wait_n;
        mreq_wait <= mreq_wait_n;
    end

    `RUNTIME_ASSERT (~drain_begin || (state == STATE_IDLE), ("%t: cache bank drain restarted while active", $time))

    assign drain_end = (state == STATE_DONE);

endmodule
