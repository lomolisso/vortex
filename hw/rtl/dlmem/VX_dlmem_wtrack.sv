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

// A core's remote-store tracker: per warp, the remote stores counted at
// translation and not yet acknowledged. drained[w] means every earlier remote
// store of warp w has committed in its target bank.
module VX_dlmem_wtrack import VX_gpu_pkg::*, VX_dlmem_pkg::*; #(
    parameter NUM_INC = `VX_CFG_NUM_LSU_BLOCKS,

    localparam CNT_W = `CLOG2(`VX_CFG_NUM_LSU_LANES + 1)
) (
    input wire                                  clk,
    input wire                                  reset,

    input wire [NUM_INC-1:0]                    inc_valid,
    input wire [NUM_INC-1:0][NW_WIDTH-1:0]      inc_wid,
    input wire [NUM_INC-1:0][CNT_W-1:0]         inc_cnt,

    input wire                                  dec_valid,
    input wire [NW_WIDTH-1:0]                   dec_wid,

    output wire [`VX_CFG_NUM_WARPS-1:0]         drained,
    output wire                                 empty
);
    // Wide enough for a counter plus every same-cycle increment, so overflow is detectable.
    localparam SUM_W = DLMEM_WT_W + `CLOG2(NUM_INC + 1) + CNT_W;

    reg [`VX_CFG_NUM_WARPS-1:0][DLMEM_WT_W-1:0] cnt_r;
    reg [`VX_CFG_NUM_WARPS-1:0]                 drained_r;

    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_warps
        wire [NUM_INC-1:0][SUM_W-1:0] inc_w;
        for (genvar i = 0; i < NUM_INC; ++i) begin : g_inc
            assign inc_w[i] = (inc_valid[i] && (inc_wid[i] == NW_WIDTH'(w))) ? SUM_W'(inc_cnt[i]) : '0;
        end

        reg [SUM_W-1:0] inc_sum;
        always @(*) begin
            inc_sum = '0;
            for (integer i = 0; i < NUM_INC; ++i) begin
                inc_sum = inc_sum + inc_w[i];
            end
        end

        wire dec_w = dec_valid && (dec_wid == NW_WIDTH'(w));
        wire [SUM_W-1:0] cnt_inc = SUM_W'(cnt_r[w]) + inc_sum;
        wire [SUM_W-1:0] cnt_n   = cnt_inc - SUM_W'(dec_w);

        always @(posedge clk) begin
            if (reset) begin
                cnt_r[w]     <= '0;
                drained_r[w] <= 1'b1;
            end else begin
                cnt_r[w]     <= DLMEM_WT_W'(cnt_n);
                drained_r[w] <= (cnt_n == '0);
            end
        end

        `RUNTIME_ASSERT (cnt_inc < (SUM_W'(1) << DLMEM_WT_W),
            ("%t: *** dlmem wtrack counter overflow: wid=%0d count=%0d", $time, w, cnt_inc))
        `RUNTIME_ASSERT (~(dec_w && (cnt_r[w] == '0) && (inc_sum == '0)),
            ("%t: *** dlmem wtrack store ack without a counted store: wid=%0d", $time, w))
    end

    assign drained = drained_r;
    assign empty   = &drained_r;

endmodule
