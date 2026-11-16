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

`include "VX_cache_define.vh"

// Tags of the H100 L1: a per-byte valid mask per line, so a store allocates
// only its bytes and a load hits only if every byte it reads is valid. A set
// is read and written whole; the set written last cycle is forwarded, since
// the array read for the next request happens in the same cycle.
module VX_cache_tags_h100 import VX_gpu_pkg::*; #(
    parameter CACHE_SIZE = 1024,
    parameter LINE_SIZE  = 16,
    parameter NUM_BANKS  = 1,
    parameter NUM_WAYS   = 1,
    parameter WORD_SIZE  = 16
) (
    input wire                          clk,
    input wire                          reset,
    input wire                          stall,

    input wire                          init,
    input wire                          flush,
    input wire [`CS_LINE_SEL_BITS-1:0]  line_idx,
    input wire [`CS_LINE_SEL_BITS-1:0]  line_idx_n,
    input wire [`CS_TAG_SEL_BITS-1:0]   line_tag,
    input wire [LINE_SIZE-1:0]          req_bytes,

    input wire                          update,
    input wire                          upd_alloc,
    input wire [NUM_WAYS-1:0]           upd_way_oh,
    input wire [LINE_SIZE-1:0]          upd_bytes,
    input wire                          inval,

    output wire [NUM_WAYS-1:0]          tag_matches,
    output wire [NUM_WAYS-1:0]          line_present,
    output wire [NUM_WAYS-1:0]          way_valid,
    output wire [LINE_SIZE-1:0]         present_bytes
);
    `STATIC_ASSERT(`CS_LINE_SEL_BITS > 0, ("invalid parameter: the H100 L1 needs more than one set"))

    localparam TAGW = `CS_TAG_SEL_BITS;
    localparam WAYW = TAGW + LINE_SIZE;

    wire [NUM_WAYS-1:0][WAYW-1:0] ram_rdata;
    reg  [NUM_WAYS-1:0][WAYW-1:0] nxt_state;

    reg                           byp_valid;
    reg [`CS_LINE_SEL_BITS-1:0]   byp_set;
    reg [NUM_WAYS-1:0][WAYW-1:0]  byp_state;

    wire [NUM_WAYS-1:0][WAYW-1:0] cur_state = (byp_valid && (byp_set == line_idx)) ? byp_state : ram_rdata;

    wire [NUM_WAYS-1:0][LINE_SIZE-1:0] present_bv;

    for (genvar w = 0; w < NUM_WAYS; ++w) begin : g_way
        wire [TAGW-1:0]      tag_w = cur_state[w][LINE_SIZE +: TAGW];
        wire [LINE_SIZE-1:0] bv_w  = cur_state[w][LINE_SIZE-1:0];
        wire valid_w  = (| bv_w);
        wire tag_eq_w = valid_w && (tag_w == line_tag);

        assign way_valid[w]    = valid_w;
        assign line_present[w] = tag_eq_w;
        assign tag_matches[w]  = tag_eq_w && ((req_bytes & ~bv_w) == '0);
        assign present_bv[w]   = tag_eq_w ? bv_w : '0;

        always @(*) begin
            nxt_state[w] = cur_state[w];
            if (init || flush || (inval && tag_eq_w)) begin
                nxt_state[w][LINE_SIZE-1:0] = '0;
            end
            if (update && upd_way_oh[w]) begin
                nxt_state[w][LINE_SIZE-1:0] = (upd_alloc ? '0 : bv_w) | upd_bytes;
                if (upd_alloc) begin
                    nxt_state[w][LINE_SIZE +: TAGW] = line_tag;
                end
            end
        end
    end

    VX_reduce_tree #(
        .IN_W  (LINE_SIZE),
        .OUT_W (LINE_SIZE),
        .N     (NUM_WAYS),
        .OP    ("|")
    ) present_reduce (
        .data_in  (present_bv),
        .data_out (present_bytes)
    );

    wire state_write = ~stall && (init || flush || inval || update);

    always @(posedge clk) begin
        if (reset) begin
            byp_valid <= 1'b0;
        end else if (~stall) begin
            byp_valid <= state_write;
        end
        if (~stall) begin
            byp_set   <= line_idx;
            byp_state <= nxt_state;
        end
    end

    VX_dp_ram #(
        .DATAW    (NUM_WAYS * WAYW),
        .SIZE     (`CS_LINES_PER_BANK),
        .OUT_REG  (1),
        .RDW_MODE ("R")
    ) state_store (
        .clk   (clk),
        .reset (reset),
        .read  (~stall),
        .write (state_write),
        .wren  (1'b1),
        .waddr (line_idx),
        .raddr (line_idx_n),
        .wdata (nxt_state),
        .rdata (ram_rdata)
    );

endmodule

`endif
