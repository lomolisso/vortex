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

`include "VX_platform.vh"

`TRACING_OFF
// The highest requesting level wins; equal levels share round robin from the
// input after the last winner (sw/common/smem_bank_ref.h level_pick).
module VX_level_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LEVEL_W      = 1,
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire [NUM_REQS-1:0]               requests,
    input  wire [NUM_REQS-1:0][LEVEL_W-1:0]  levels,
    output wire [LOG_NUM_REQS-1:0]           grant_index,
    output wire [NUM_REQS-1:0]               grant_onehot,
    output wire                              grant_valid,
    input  wire                              grant_ready
);
    if (NUM_REQS == 1) begin : g_passthru

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (levels)
        `UNUSED_VAR (grant_ready)
        assign grant_index  = '0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin : g_arbiter

        reg [LOG_NUM_REQS-1:0] rr_r;
        reg [LEVEL_W-1:0]      top;
        reg [NUM_REQS-1:0]     cand;
        reg [LOG_NUM_REQS-1:0] pick;
        reg                    found;

        always @(*) begin
            top = '0;
            for (integer i = 0; i < NUM_REQS; ++i) begin
                if (requests[i] && levels[i] > top)
                    top = levels[i];
            end
            for (integer i = 0; i < NUM_REQS; ++i) begin
                cand[i] = requests[i] && (levels[i] == top);
            end
            pick  = '0;
            found = 1'b0;
            for (integer k = 0; k < NUM_REQS; ++k) begin
                reg [LOG_NUM_REQS:0] idx;
                idx = (LOG_NUM_REQS+1)'(rr_r) + (LOG_NUM_REQS+1)'(k);
                if (idx >= (LOG_NUM_REQS+1)'(NUM_REQS))
                    idx = idx - (LOG_NUM_REQS+1)'(NUM_REQS);
                if (~found && cand[idx[LOG_NUM_REQS-1:0]]) begin
                    found = 1'b1;
                    pick  = idx[LOG_NUM_REQS-1:0];
                end
            end
        end

        always @(posedge clk) begin
            if (reset) begin
                rr_r <= '0;
            end else if (found && grant_ready) begin
                rr_r <= (pick == LOG_NUM_REQS'(NUM_REQS - 1)) ? '0 : (pick + LOG_NUM_REQS'(1));
            end
        end

        assign grant_index  = pick;
        assign grant_onehot = found ? (NUM_REQS'(1) << pick) : '0;
        assign grant_valid  = found;

    end

endmodule
`TRACING_ON

`endif
