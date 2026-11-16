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

// An SM port's line requests as memory-block requests: one block request per
// block the line's byteen touches (block 0 for a request with no byte), lowest
// first. A request that answers (a read, an atomic, a fence) holds an entry
// that gathers its blocks' replies, which may arrive in any order; the line
// reply leaves once all have, in the order the lines complete.
module VX_sm_mem_adapter import VX_gpu_pkg::*; #(
    parameter LINE_SIZE     = 128,
    parameter BLOCK_SIZE    = 64,
    parameter TAG_IN_WIDTH  = UUID_WIDTH + 1,
    parameter TAG_OUT_WIDTH = UUID_WIDTH + 1,
    parameter PENDING       = 256
) (
    input wire              clk,
    input wire              reset,

    VX_mem_bus_if.slave     line_bus_if,
    VX_mem_bus_if.master    mem_bus_if,

    output wire             empty
);
    localparam BLOCKS  = LINE_SIZE / BLOCK_SIZE;
    localparam BLK_W   = `LOG2UP(BLOCKS);
    localparam IDX_W   = `LOG2UP(PENDING);
    localparam CNT_W   = `CLOG2(BLOCKS + 1);
    localparam OUT_ADDR_W = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(BLOCK_SIZE);

    `STATIC_ASSERT (LINE_SIZE % BLOCK_SIZE == 0 && BLOCKS >= 1, ("sm adapter: the line must be whole blocks"))
    `STATIC_ASSERT (`IS_POW2(PENDING), ("sm adapter: pending entries must be a power of two"))
    `STATIC_ASSERT (TAG_OUT_WIDTH >= UUID_WIDTH + IDX_W + ((BLOCKS > 1) ? BLK_W : 0),
        ("sm adapter: memory tag too narrow"))

    wire [LINE_SIZE-1:0] in_be = line_bus_if.req_data.byteen;
    mem_bus_attr_t in_attr;
    assign in_attr = mem_bus_attr_t'(line_bus_if.req_data.attr);
    `UNUSED_VAR (in_attr)
    wire expects = ~line_bus_if.req_data.rw || in_attr.amo.amo_valid || in_attr.is_fence;

    wire [BLOCKS-1:0] touched;
    for (genvar b = 0; b < BLOCKS; ++b) begin : g_touched
        assign touched[b] = (| in_be[b * BLOCK_SIZE +: BLOCK_SIZE]);
    end
    wire [BLOCKS-1:0] blocks = (touched != '0) ? touched : BLOCKS'(1);
    wire [CNT_W-1:0]  n_blocks;
    `POP_COUNT(n_blocks, blocks);

    reg  [BLOCKS-1:0] sent_r;
    reg               started_r;
    reg  [IDX_W-1:0]  cur_idx_r;
    reg  [PENDING-1:0] pend_valid_r;

    wire [BLOCKS-1:0] left = blocks & ~sent_r;
    wire [BLK_W-1:0]  cur_blk;
    wire [BLOCKS-1:0] cur_onehot;
    VX_priority_encoder #(
        .N (BLOCKS)
    ) blk_sel (
        .data_in    (left),
        .onehot_out (cur_onehot),
        .index_out  (cur_blk),
        `UNUSED_PIN (valid_out)
    );
    wire last = ((left & ~cur_onehot) == '0);

    wire [IDX_W-1:0] free_idx;
    wire             free_valid;
    VX_priority_encoder #(
        .N (PENDING)
    ) free_sel (
        .data_in    (~pend_valid_r),
        `UNUSED_PIN (onehot_out),
        .index_out  (free_idx),
        .valid_out  (free_valid)
    );

    wire need_alloc = expects && ~started_r;
    wire [IDX_W-1:0] idx = started_r ? cur_idx_r : free_idx;

    wire out_valid = line_bus_if.req_valid && ~(need_alloc && ~free_valid);
    wire out_ready;
    wire out_fire  = out_valid && out_ready;
    wire alloc     = out_fire && need_alloc;

    assign line_bus_if.req_ready = out_fire && last;

    always @(posedge clk) begin
        if (reset) begin
            sent_r    <= '0;
            started_r <= 1'b0;
        end else if (out_fire) begin
            sent_r    <= last ? '0 : (sent_r | cur_onehot);
            started_r <= ~last && expects;
        end
        if (alloc) begin
            cur_idx_r <= free_idx;
        end
    end

    wire [LINE_SIZE*8-1:0] in_data = line_bus_if.req_data.data;
    wire [TAG_OUT_WIDTH-1:0] out_tag;
    if (BLOCKS > 1) begin : g_tag_blk
        assign out_tag = {line_bus_if.req_data.tag.uuid, (TAG_OUT_WIDTH - UUID_WIDTH - BLK_W)'(idx), cur_blk};
    end else begin : g_tag
        assign out_tag = {line_bus_if.req_data.tag.uuid, (TAG_OUT_WIDTH - UUID_WIDTH)'(idx)};
    end
    wire [OUT_ADDR_W-1:0] out_addr;
    if (BLOCKS > 1) begin : g_addr_blk
        assign out_addr = OUT_ADDR_W'({line_bus_if.req_data.addr, cur_blk});
    end else begin : g_addr
        assign out_addr = OUT_ADDR_W'(line_bus_if.req_data.addr);
    end

    VX_elastic_buffer #(
        .DATAW (1 + OUT_ADDR_W + BLOCK_SIZE * 8 + BLOCK_SIZE + MEM_ATTR_WIDTH + TAG_OUT_WIDTH),
        .SIZE  (2)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (out_valid),
        .ready_in  (out_ready),
        .data_in   ({line_bus_if.req_data.rw, out_addr,
                     in_data[cur_blk * BLOCK_SIZE * 8 +: BLOCK_SIZE * 8],
                     in_be[cur_blk * BLOCK_SIZE +: BLOCK_SIZE],
                     line_bus_if.req_data.attr, out_tag}),
        .data_out  ({mem_bus_if.req_data.rw, mem_bus_if.req_data.addr, mem_bus_if.req_data.data,
                     mem_bus_if.req_data.byteen, mem_bus_if.req_data.attr, mem_bus_if.req_data.tag}),
        .ready_out (mem_bus_if.req_ready),
        .valid_out (mem_bus_if.req_valid)
    );

    // Replies.

    reg [PENDING-1:0][CNT_W-1:0]     pend_left_r;
    reg [PENDING-1:0][TAG_IN_WIDTH-1:0] pend_tag_r;
    reg [LINE_SIZE*8-1:0]            pend_data_r [PENDING];

    wire [IDX_W-1:0] rsp_idx;
    wire [BLK_W-1:0] rsp_blk;
    if (BLOCKS > 1) begin : g_rsp_blk
        assign rsp_idx = mem_bus_if.rsp_data.tag.value[BLK_W +: IDX_W];
        assign rsp_blk = mem_bus_if.rsp_data.tag.value[BLK_W-1:0];
    end else begin : g_rsp
        assign rsp_idx = mem_bus_if.rsp_data.tag.value[IDX_W-1:0];
        assign rsp_blk = '0;
    end
    `UNUSED_VAR (mem_bus_if.rsp_data.tag)
    assign mem_bus_if.rsp_ready = 1'b1;

    wire rsp_fire = mem_bus_if.rsp_valid;
    wire rsp_done = rsp_fire && (pend_left_r[rsp_idx] == CNT_W'(1));

    wire             done_empty;
    wire [IDX_W-1:0] done_idx;
    wire             done_pop = ~done_empty && line_bus_if.rsp_ready;

    VX_fifo_queue #(
        .DATAW (IDX_W),
        .DEPTH (PENDING)
    ) done_q (
        .clk      (clk),
        .reset    (reset),
        .push     (rsp_done),
        .pop      (done_pop),
        .data_in  (rsp_idx),
        .data_out (done_idx),
        .empty    (done_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    always @(posedge clk) begin
        if (reset) begin
            pend_valid_r <= '0;
        end else begin
            if (alloc) begin
                pend_valid_r[free_idx] <= 1'b1;
            end
            if (done_pop) begin
                pend_valid_r[done_idx] <= 1'b0;
            end
        end
        if (alloc) begin
            pend_left_r[free_idx] <= n_blocks;
            pend_tag_r[free_idx]  <= line_bus_if.req_data.tag;
        end
        if (rsp_fire) begin
            pend_left_r[rsp_idx] <= pend_left_r[rsp_idx] - CNT_W'(1);
            pend_data_r[rsp_idx][rsp_blk * BLOCK_SIZE * 8 +: BLOCK_SIZE * 8] <= mem_bus_if.rsp_data.data;
        end
    end

    assign line_bus_if.rsp_valid     = ~done_empty;
    assign line_bus_if.rsp_data.data = pend_data_r[done_idx];
    assign line_bus_if.rsp_data.tag  = pend_tag_r[done_idx];

    assign empty = ~(| pend_valid_r) && ~mem_bus_if.req_valid && ~started_r;

    `RUNTIME_ASSERT (~(rsp_fire && ~pend_valid_r[rsp_idx]),
        ("%t: *** sm adapter: reply for a free entry %0d", $time, rsp_idx))

endmodule

`endif // VX_CFG_H100_SM_ENABLE
