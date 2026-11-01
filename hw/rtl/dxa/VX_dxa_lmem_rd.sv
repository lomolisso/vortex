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

// DXA local-memory line reader, for bulk copies: each line read gmem_req
// would send to global memory becomes the WORD_SIZE reads covering it, on the
// worker's local-memory bus to the issuer core, and their responses become one
// line response with the read's tag. The reads share the bus with the writer's
// beats and win it: a line's reads are sent as soon as it is taken, so the
// writer finds its next lines read ahead. Every read goes to one core and
// returns in order, so the lines fill and leave in order. A line is taken
// only into a free entry, which it holds until smem_wr takes the line, so
// every response has its room and the bus never waits on one; DEPTH entries
// cover the read latency, apart from the tags gmem_req keeps.

// With the knobs off the file is empty: VX_define.vh derives DXA_MBAR_ENABLE.
`ifdef VX_CFG_EXT_DXA_ENABLE
`ifdef VX_CFG_EXT_MBAR_ENABLE
`include "VX_define.vh"
`endif
`endif

`ifdef DXA_MBAR_ENABLE

module VX_dxa_lmem_rd import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter MAX_OUTSTANDING = 8,               // gmem_req's tags
    parameter DEPTH           = MAX_OUTSTANDING, // line entries
    parameter CL_SIZE         = `VX_CFG_L1_LINE_SIZE,
    parameter WORD_SIZE       = DXA_LMEM_WORD_SIZE,
    parameter LINE_ADDR_W     = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(CL_SIZE),
    localparam TAG_W          = `CLOG2(MAX_OUTSTANDING)
) (
    input  wire                        clk,
    input  wire                        reset,

    // The issuer, whose local memory holds the source.
    input  wire [NC_WIDTH-1:0]         core_id,
    input  wire [UUID_WIDTH-1:0]       uuid,

    // Line reads (from gmem_req) and their responses.
    input  wire                        line_req_valid,
    input  wire [LINE_ADDR_W-1:0]      line_req_addr,
    input  wire [TAG_W-1:0]            line_req_tag,
    output wire                        line_req_ready,
    output wire                        line_rsp_valid,
    output wire [CL_SIZE*8-1:0]        line_rsp_data,
    output wire [TAG_W-1:0]            line_rsp_tag,
    input  wire                        line_rsp_ready,

    // The writer's beats (smem_wr), and the worker's local-memory bus.
    VX_mem_bus_if.slave                wr_bus_if,
    VX_mem_bus_if.master               smem_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam LMEM_LOG  = `VX_CFG_LMEM_LOG_SIZE;
    localparam CL_LOG    = `CLOG2(CL_SIZE);
    localparam WORD_LOG  = `CLOG2(WORD_SIZE);
    localparam LADDR_W   = LMEM_LOG - CL_LOG;          // a line of the local memory
    localparam ADDR_W    = LMEM_LOG - WORD_LOG;        // a word of the local memory
    localparam NUM_RDS   = (CL_SIZE > WORD_SIZE) ? (CL_SIZE / WORD_SIZE) : 1;
    localparam RD_W      = `LOG2UP(NUM_RDS);
    localparam WORD_W    = WORD_SIZE * 8;
    localparam PTR_W     = `LOG2UP(DEPTH);

    localparam ENGINE_VALUE_W   = DXA_LMEM_ENGINE_TAG_W - UUID_WIDTH;
    localparam SMEM_TAG_VALUE_W = DXA_LMEM_TAG_W - UUID_WIDTH;

    `STATIC_ASSERT ((DEPTH >= 1) && (DEPTH <= MAX_OUTSTANDING), ("invalid parameter"))
    `STATIC_ASSERT (`IS_POW2(CL_SIZE) && `IS_POW2(WORD_SIZE), ("invalid parameter"))
    `STATIC_ASSERT (LADDR_W > 0, ("invalid parameter"))
    `STATIC_ASSERT (ADDR_W == DXA_LMEM_ADDR_W, ("invalid parameter"))

    // Line entries, a ring in order; each pointer carries a lap bit.
    reg [CL_SIZE*8-1:0] data_r [DEPTH];
    reg [TAG_W-1:0]     tag_r  [DEPTH];
    reg [LADDR_W-1:0]   laddr_r [DEPTH];

    reg [PTR_W:0] alloc_ptr_r;  // next entry to take
    reg [PTR_W:0] issue_ptr_r;  // entry whose reads are being sent
    reg [PTR_W:0] fill_ptr_r;   // entry the next response lands in
    reg [PTR_W:0] out_ptr_r;    // entry presented to gmem_req
    reg [RD_W-1:0] issue_k_r;   // read of the issuing entry
    reg [RD_W-1:0] fill_k_r;    // response of the filling entry

    wire [PTR_W-1:0] alloc_idx = alloc_ptr_r[PTR_W-1:0];
    wire [PTR_W-1:0] issue_idx = issue_ptr_r[PTR_W-1:0];
    wire [PTR_W-1:0] fill_idx  = fill_ptr_r[PTR_W-1:0];
    wire [PTR_W-1:0] out_idx   = out_ptr_r[PTR_W-1:0];

    function automatic [PTR_W:0] ptr_next(input [PTR_W:0] ptr);
        if (ptr[PTR_W-1:0] == PTR_W'(DEPTH - 1)) begin
            ptr_next = {~ptr[PTR_W], PTR_W'(0)};
        end else begin
            ptr_next = ptr + (PTR_W+1)'(1);
        end
    endfunction

    wire full = (alloc_ptr_r == {~out_ptr_r[PTR_W], out_ptr_r[PTR_W-1:0]});

    assign line_req_ready = ~full;
    wire line_req_fire = line_req_valid && ~full;

    // ---- reads ----
    wire rd_valid = (issue_ptr_r != alloc_ptr_r);
    wire rd_last  = (issue_k_r == RD_W'(NUM_RDS - 1));
    wire [ADDR_W-1:0] rd_addr;
    if (CL_SIZE > WORD_SIZE) begin : g_rd_split
        assign rd_addr = {laddr_r[issue_idx], issue_k_r};
    end else if (CL_SIZE == WORD_SIZE) begin : g_rd_same
        assign rd_addr = laddr_r[issue_idx];
        `UNUSED_VAR (issue_k_r)
    end else begin : g_rd_part
        assign rd_addr = laddr_r[issue_idx][LADDR_W-1:WORD_LOG-CL_LOG];
        `UNUSED_VAR (issue_k_r)
    end

    wire rd_fire = rd_valid && smem_bus_if.req_ready;

    assign smem_bus_if.req_valid = rd_valid || wr_bus_if.req_valid;
    assign smem_bus_if.req_data.rw     = rd_valid ? 1'b0               : wr_bus_if.req_data.rw;
    assign smem_bus_if.req_data.addr   = rd_valid ? rd_addr            : wr_bus_if.req_data.addr;
    assign smem_bus_if.req_data.data   = rd_valid ? '0                 : wr_bus_if.req_data.data;
    assign smem_bus_if.req_data.byteen = rd_valid ? {WORD_SIZE{1'b1}}  : wr_bus_if.req_data.byteen;
    assign smem_bus_if.req_data.attr   = rd_valid ? '0                 : wr_bus_if.req_data.attr;
    assign smem_bus_if.req_data.tag.uuid  = rd_valid ? uuid : wr_bus_if.req_data.tag.uuid;
    assign smem_bus_if.req_data.tag.value = rd_valid ? ((SMEM_TAG_VALUE_W'(core_id) << ENGINE_VALUE_W) | SMEM_TAG_VALUE_W'(1))
                                                     : wr_bus_if.req_data.tag.value;
    assign wr_bus_if.req_ready = smem_bus_if.req_ready && ~rd_valid;

    // Writes are posted.
    assign wr_bus_if.rsp_valid = 1'b0;
    assign wr_bus_if.rsp_data  = '0;
    `UNUSED_VAR (wr_bus_if.rsp_ready)

    // ---- responses ----
    assign smem_bus_if.rsp_ready = 1'b1;
    wire rsp_fire = smem_bus_if.rsp_valid;
    wire fill_last = (fill_k_r == RD_W'(NUM_RDS - 1));
    `UNUSED_VAR (smem_bus_if.rsp_data.tag)

    wire [CL_SIZE*8-1:0] rsp_line;
    if (CL_SIZE > WORD_SIZE) begin : g_rsp_split
        assign rsp_line = '0;
        `UNUSED_VAR (rsp_line)
    end else if (CL_SIZE == WORD_SIZE) begin : g_rsp_same
        assign rsp_line = smem_bus_if.rsp_data.data;
    end else begin : g_rsp_part
        localparam SEL_W = WORD_LOG - CL_LOG;
        wire [SEL_W-1:0] sel = laddr_r[fill_idx][SEL_W-1:0];
        assign rsp_line = smem_bus_if.rsp_data.data[sel * (CL_SIZE*8) +: (CL_SIZE*8)];
    end

    // ---- line output ----
    assign line_rsp_valid = (out_ptr_r != fill_ptr_r);
    assign line_rsp_data  = data_r[out_idx];
    assign line_rsp_tag   = tag_r[out_idx];
    wire line_rsp_fire = line_rsp_valid && line_rsp_ready;

    always @(posedge clk) begin
        if (reset) begin
            alloc_ptr_r <= '0;
            issue_ptr_r <= '0;
            fill_ptr_r  <= '0;
            out_ptr_r   <= '0;
            issue_k_r   <= '0;
            fill_k_r    <= '0;
        end else begin
            if (line_req_fire) begin
                alloc_ptr_r <= ptr_next(alloc_ptr_r);
            end
            if (rd_fire) begin
                issue_k_r <= rd_last ? '0 : (issue_k_r + RD_W'(1));
                if (rd_last) begin
                    issue_ptr_r <= ptr_next(issue_ptr_r);
                end
            end
            if (rsp_fire) begin
                fill_k_r <= fill_last ? '0 : (fill_k_r + RD_W'(1));
                if (fill_last) begin
                    fill_ptr_r <= ptr_next(fill_ptr_r);
                end
            end
            if (line_rsp_fire) begin
                out_ptr_r <= ptr_next(out_ptr_r);
            end
        end
        if (line_req_fire) begin
            tag_r[alloc_idx]   <= line_req_tag;
            laddr_r[alloc_idx] <= line_req_addr[LADDR_W-1:0];
        end
    end

    if (CL_SIZE > WORD_SIZE) begin : g_fill_split
        always @(posedge clk) begin
            if (rsp_fire) begin
                data_r[fill_idx][fill_k_r * WORD_W +: WORD_W] <= smem_bus_if.rsp_data.data;
            end
        end
    end else begin : g_fill_line
        always @(posedge clk) begin
            if (rsp_fire) begin
                data_r[fill_idx] <= rsp_line;
            end
        end
    end

    if (LINE_ADDR_W > LADDR_W) begin : g_line_addr_hi
        `UNUSED_VAR (line_req_addr[LINE_ADDR_W-1:LADDR_W])
    end

    `RUNTIME_ASSERT (~(rsp_fire && (fill_ptr_r == issue_ptr_r) && (fill_k_r >= issue_k_r)),
        ("%t: *** %s DXA local-memory read response without a read", $time, INSTANCE_ID))

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (rd_fire) begin
                $write("DXA_PIPE,%0d,LMEM_RD,core=%0d,addr=0x%0h,tag=%0d\n",
                    $time, core_id, rd_addr, tag_r[issue_idx]);
            end
            if (rsp_fire && fill_last) begin
                $write("DXA_PIPE,%0d,LMEM_LINE,tag=%0d\n", $time, tag_r[fill_idx]);
            end
        end
    end
`endif

endmodule

`endif
