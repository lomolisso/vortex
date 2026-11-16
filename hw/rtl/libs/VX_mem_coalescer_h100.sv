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

`include "VX_platform.vh"

`ifdef VX_CFG_H100_SM_ENABLE

`TRACING_OFF
// The H100 SM's line coalescer: one batch per cycle, each output channel one
// line. Channels whose seed lands on a line an earlier channel already took
// fold into it, so a warp request costs one L1 access per distinct line.
// Ports as VX_mem_coalescer.
module VX_mem_coalescer_h100 #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_REQS      = 1,
    parameter ADDR_WIDTH    = 32,
    parameter USER_WIDTH    = 0,
    parameter DATA_IN_SIZE  = 4,
    parameter DATA_OUT_SIZE = 64,
    parameter TAG_WIDTH     = 8,
    parameter UUID_WIDTH    = 0,
    parameter QUEUE_SIZE    = 8,
    parameter PERF_CTR_BITS = `CLOG2(NUM_REQS+1),

    parameter DATA_IN_WIDTH = DATA_IN_SIZE * 8,
    parameter DATA_OUT_WIDTH= DATA_OUT_SIZE * 8,
    parameter DATA_RATIO    = DATA_OUT_SIZE / DATA_IN_SIZE,
    parameter DATA_RATIO_W  = `LOG2UP(DATA_RATIO),
    parameter OUT_REQS      = NUM_REQS / DATA_RATIO,
    parameter OUT_ADDR_WIDTH= ADDR_WIDTH - DATA_RATIO_W,
    parameter QUEUE_ADDRW   = `CLOG2(QUEUE_SIZE),
    parameter OUT_TAG_WIDTH = UUID_WIDTH + QUEUE_ADDRW
) (
    input wire clk,
    input wire reset,

    output wire [PERF_CTR_BITS-1:0]     misses,

    output wire                         empty,

    input wire                          in_req_valid,
    input wire                          in_req_rw,
    input wire [NUM_REQS-1:0]           in_req_mask,
    input wire [NUM_REQS-1:0][DATA_IN_SIZE-1:0] in_req_byteen,
    input wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] in_req_addr,
    input wire [NUM_REQS-1:0][`UP(USER_WIDTH)-1:0] in_req_user,
    input wire [NUM_REQS-1:0][DATA_IN_WIDTH-1:0] in_req_data,
    input wire                          in_req_no_merge,
    input wire [TAG_WIDTH-1:0]          in_req_tag,
    output wire                         in_req_ready,

    output wire                         in_rsp_valid,
    output wire [NUM_REQS-1:0]          in_rsp_mask,
    output wire [NUM_REQS-1:0][DATA_IN_WIDTH-1:0] in_rsp_data,
    output wire [TAG_WIDTH-1:0]         in_rsp_tag,
    input wire                          in_rsp_ready,

    output wire                         out_req_valid,
    output wire                         out_req_rw,
    output wire [OUT_REQS-1:0]          out_req_mask,
    output wire [OUT_REQS-1:0][DATA_OUT_SIZE-1:0] out_req_byteen,
    output wire [OUT_REQS-1:0][OUT_ADDR_WIDTH-1:0] out_req_addr,
    output wire [OUT_REQS-1:0][`UP(USER_WIDTH)-1:0] out_req_user,
    output wire [OUT_REQS-1:0][DATA_OUT_WIDTH-1:0] out_req_data,
    output wire [OUT_TAG_WIDTH-1:0]     out_req_tag,
    input wire                          out_req_ready,

    input wire                          out_rsp_valid,
    input wire [OUT_REQS-1:0]           out_rsp_mask,
    input wire [OUT_REQS-1:0][DATA_OUT_WIDTH-1:0] out_rsp_data,
    input wire [OUT_TAG_WIDTH-1:0]      out_rsp_tag,
    output wire                         out_rsp_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `STATIC_ASSERT ((NUM_REQS > 1), ("invalid parameter"))
    `STATIC_ASSERT (`IS_DIVISBLE(NUM_REQS * DATA_IN_WIDTH, DATA_OUT_WIDTH), ("invalid parameter"))
    `STATIC_ASSERT ((NUM_REQS * DATA_IN_WIDTH >= DATA_OUT_WIDTH), ("invalid parameter"))
    `RUNTIME_ASSERT ((~in_req_valid || in_req_mask != 0), ("invalid request mask"))
    `RUNTIME_ASSERT ((~out_rsp_valid || out_rsp_mask != 0), ("invalid request mask"))

    localparam TAG_ID_WIDTH = TAG_WIDTH - UUID_WIDTH;
    localparam CH_W = `UP(`CLOG2(OUT_REQS));
    localparam IBUF_DATA_WIDTH = TAG_ID_WIDTH + NUM_REQS + (NUM_REQS * DATA_RATIO_W) + (OUT_REQS * CH_W);

    reg [NUM_REQS-1:0] rem_mask_r;
    wire [NUM_REQS-1:0] pend_mask = in_req_mask & rem_mask_r;

    wire [NUM_REQS-1:0][DATA_RATIO_W-1:0] in_addr_offset;
    wire [NUM_REQS-1:0][OUT_ADDR_WIDTH-1:0] in_addr_base;
    for (genvar i = 0; i < NUM_REQS; i++) begin : g_in_addr
        assign in_addr_offset[i] = in_req_addr[i][DATA_RATIO_W-1:0];
        assign in_addr_base[i]   = in_req_addr[i][ADDR_WIDTH-1:DATA_RATIO_W];
    end

    // per channel: its seed (first pending lane of its lane group)
    wire [OUT_REQS-1:0] has_seed;
    wire [OUT_REQS-1:0][DATA_RATIO_W-1:0] seed_idx;
    wire [OUT_REQS-1:0][OUT_ADDR_WIDTH-1:0] seed_addr;
    wire [OUT_REQS-1:0][`UP(USER_WIDTH)-1:0] seed_user;
    for (genvar i = 0; i < OUT_REQS; ++i) begin : g_seed
        VX_priority_encoder #(
            .N (DATA_RATIO)
        ) seed_sel (
            .data_in   (pend_mask[i * DATA_RATIO +: DATA_RATIO]),
            .index_out (seed_idx[i]),
            .valid_out (has_seed[i]),
            `UNUSED_PIN (onehot_out)
        );
        assign seed_addr[i] = in_addr_base[i * DATA_RATIO + 32'(seed_idx[i])];
        assign seed_user[i] = in_req_user[i * DATA_RATIO + 32'(seed_idx[i])];
    end

    // channel i folds into the lowest channel j whose seed has its line
    reg [OUT_REQS-1:0][CH_W-1:0] ch_alias;
    always @(*) begin
        for (integer i = 0; i < OUT_REQS; ++i) begin
            ch_alias[i] = CH_W'(i);
            for (integer j = OUT_REQS-1; j >= 0; --j) begin
                if ((j < i) && has_seed[j] && (seed_addr[j] == seed_addr[i]) && ~in_req_no_merge) begin
                    ch_alias[i] = CH_W'(j);
                end
            end
        end
    end
    wire [OUT_REQS-1:0] ch_valid;
    for (genvar i = 0; i < OUT_REQS; ++i) begin : g_ch_valid
        assign ch_valid[i] = has_seed[i] && (ch_alias[i] == CH_W'(i));
    end

    wire [NUM_REQS-1:0] cur_pmask;
    for (genvar i = 0; i < OUT_REQS; ++i) begin : g_match
        for (genvar j = 0; j < DATA_RATIO; ++j) begin : g_j
            assign cur_pmask[i * DATA_RATIO + j] = has_seed[i] && pend_mask[i * DATA_RATIO + j]
                && (in_addr_base[i * DATA_RATIO + j] == seed_addr[i])
                && (~in_req_no_merge || (DATA_RATIO_W'(j) == seed_idx[i]));
        end
    end

    reg [OUT_REQS-1:0][DATA_RATIO-1:0][DATA_IN_SIZE-1:0] byteen_merged;
    reg [OUT_REQS-1:0][DATA_RATIO-1:0][DATA_IN_WIDTH-1:0] data_merged;
    always @(*) begin
        byteen_merged = '0;
        data_merged = 'x;
        for (integer i = 0; i < OUT_REQS; ++i) begin
            for (integer j = 0; j < DATA_RATIO; ++j) begin
                for (integer k = 0; k < DATA_IN_SIZE; ++k) begin
                    if (cur_pmask[i * DATA_RATIO + j] && in_req_byteen[i * DATA_RATIO + j][k]) begin
                        byteen_merged[ch_alias[i]][in_addr_offset[i * DATA_RATIO + j]][k] = 1'b1;
                        data_merged[ch_alias[i]][in_addr_offset[i * DATA_RATIO + j]][k * 8 +: 8] = in_req_data[i * DATA_RATIO + j][k * 8 +: 8];
                    end
                end
            end
        end
    end

    wire is_last_batch = ~(| (pend_mask & ~cur_pmask));

    wire ibuf_full, ibuf_empty;
    wire [QUEUE_ADDRW-1:0] ibuf_waddr, ibuf_raddr;
    wire [IBUF_DATA_WIDTH-1:0] ibuf_dout;

    reg out_req_valid_r, out_req_rw_r;
    reg [OUT_REQS-1:0] out_req_mask_r;
    reg [OUT_REQS-1:0][OUT_ADDR_WIDTH-1:0] out_req_addr_r;
    reg [OUT_REQS-1:0][`UP(USER_WIDTH)-1:0] out_req_user_r;
    reg [OUT_REQS-1:0][DATA_OUT_SIZE-1:0] out_req_byteen_r;
    reg [OUT_REQS-1:0][DATA_OUT_WIDTH-1:0] out_req_data_r;
    reg [OUT_TAG_WIDTH-1:0] out_req_tag_r;

    wire out_req_fire = out_req_valid && out_req_ready;
    wire issue = in_req_valid && (~out_req_valid_r || out_req_ready) && ~ibuf_full;

    always @(posedge clk) begin
        if (reset) begin
            out_req_valid_r <= 1'b0;
            rem_mask_r      <= '1;
        end else begin
            if (issue) begin
                out_req_valid_r <= 1'b1;
                rem_mask_r      <= is_last_batch ? '1 : (rem_mask_r & ~cur_pmask);
            end else if (out_req_fire) begin
                out_req_valid_r <= 1'b0;
            end
        end
        if (issue) begin
            out_req_rw_r     <= in_req_rw;
            out_req_mask_r   <= ch_valid;
            out_req_addr_r   <= seed_addr;
            out_req_user_r   <= seed_user;
            out_req_byteen_r <= byteen_merged;
            out_req_data_r   <= data_merged;
            out_req_tag_r    <= {in_req_tag[TAG_WIDTH-1 -: UUID_WIDTH], ibuf_waddr};
        end
    end

    assign in_req_ready = issue && is_last_batch;

    assign out_req_valid  = out_req_valid_r;
    assign out_req_rw     = out_req_rw_r;
    assign out_req_mask   = out_req_mask_r;
    assign out_req_byteen = out_req_byteen_r;
    assign out_req_addr   = out_req_addr_r;
    if (USER_WIDTH != 0) begin : g_out_req_user
        assign out_req_user = out_req_user_r;
    end else begin : g_out_req_user_0
        `UNUSED_VAR (out_req_user_r)
        assign out_req_user = '0;
    end
    assign out_req_data   = out_req_data_r;
    assign out_req_tag    = out_req_tag_r;

    assign empty = ~out_req_valid_r && ~in_req_valid;

    wire out_rsp_fire = out_rsp_valid && out_rsp_ready;
    wire out_rsp_eop;
    wire ibuf_push = issue && ~in_req_rw;
    wire ibuf_pop  = out_rsp_fire && out_rsp_eop;
    assign ibuf_raddr = out_rsp_tag[QUEUE_ADDRW-1:0];

    VX_index_buffer #(
        .DATAW (IBUF_DATA_WIDTH),
        .SIZE  (QUEUE_SIZE)
    ) req_ibuf (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (ibuf_push),
        .write_addr   (ibuf_waddr),
        .write_data   ({in_req_tag[TAG_ID_WIDTH-1:0], cur_pmask, in_addr_offset, ch_alias}),
        .read_data    (ibuf_dout),
        .read_addr    (ibuf_raddr),
        .release_en   (ibuf_pop),
        .full         (ibuf_full),
        .empty        (ibuf_empty)
    );
    `UNUSED_VAR (ibuf_empty)

    reg [QUEUE_SIZE-1:0][OUT_REQS-1:0] rsp_rem_mask;
    wire [OUT_REQS-1:0] rsp_rem_mask_n = rsp_rem_mask[ibuf_raddr] & ~out_rsp_mask;
    assign out_rsp_eop = ~(| rsp_rem_mask_n);

    always @(posedge clk) begin
        if (ibuf_push) begin
            rsp_rem_mask[ibuf_waddr] <= ch_valid;
        end
        if (out_rsp_fire) begin
            rsp_rem_mask[ibuf_raddr] <= rsp_rem_mask_n;
        end
    end

    wire [TAG_ID_WIDTH-1:0] ibuf_dout_tag;
    wire [NUM_REQS-1:0] ibuf_dout_pmask;
    wire [NUM_REQS-1:0][DATA_RATIO_W-1:0] ibuf_dout_offset;
    wire [OUT_REQS-1:0][CH_W-1:0] ibuf_dout_alias;
    assign {ibuf_dout_tag, ibuf_dout_pmask, ibuf_dout_offset, ibuf_dout_alias} = ibuf_dout;

    for (genvar i = 0; i < OUT_REQS; ++i) begin : g_in_rsp
        wire [DATA_OUT_WIDTH-1:0] ch_data = out_rsp_data[ibuf_dout_alias[i]];
        wire ch_mask = out_rsp_mask[ibuf_dout_alias[i]];
        for (genvar j = 0; j < DATA_RATIO; ++j) begin : g_j
            assign in_rsp_data[i * DATA_RATIO + j] = ch_data[ibuf_dout_offset[i * DATA_RATIO + j] * DATA_IN_WIDTH +: DATA_IN_WIDTH];
            assign in_rsp_mask[i * DATA_RATIO + j] = ch_mask && ibuf_dout_pmask[i * DATA_RATIO + j];
        end
    end

    assign in_rsp_valid  = out_rsp_valid;
    assign in_rsp_tag    = {out_rsp_tag[OUT_TAG_WIDTH-1 -: UUID_WIDTH], ibuf_dout_tag};
    assign out_rsp_ready = in_rsp_ready;

    reg [PERF_CTR_BITS-1:0] misses_r;
    always @(posedge clk) begin
        if (reset) begin
            misses_r <= '0;
        end else begin
            misses_r <= misses_r + PERF_CTR_BITS'(issue && ~is_last_batch);
        end
    end
    assign misses = misses_r;

endmodule
`TRACING_ON

`endif
