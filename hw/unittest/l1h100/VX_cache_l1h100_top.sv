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

// The SM's L1 data cache as the socket builds it under VX_CFG_H100_SM_ENABLE:
// one line-wide input, one memory port of line requests.
module VX_cache_l1h100_top import VX_gpu_pkg::*; #(
    parameter LINE_SIZE  = `VX_CFG_DCACHE_LINE_SIZE,
    parameter MSHR_SIZE  = `VX_CFG_DCACHE_MSHR_SIZE,
    parameter TAG_VALUE_WIDTH = 8,

    localparam TAG_WIDTH     = TAG_VALUE_WIDTH + UUID_WIDTH,
    localparam ADDR_WIDTH    = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(LINE_SIZE),
    localparam MEM_TAG_WIDTH = `CACHE_MEM_TAG_WIDTH(MSHR_SIZE, 1, 1, UUID_WIDTH)
) (
    input  wire                     clk,
    input  wire                     reset,

    input  wire                     core_req_valid,
    input  wire                     core_req_rw,
    input  wire                     core_req_flush,
    input  wire [LINE_SIZE-1:0]     core_req_byteen,
    input  wire [ADDR_WIDTH-1:0]    core_req_addr,
    input  wire [LINE_SIZE*8-1:0]   core_req_data,
    input  wire [TAG_VALUE_WIDTH-1:0] core_req_tag,
    output wire                     core_req_ready,

    output wire                     core_rsp_valid,
    output wire [LINE_SIZE*8-1:0]   core_rsp_data,
    output wire [TAG_VALUE_WIDTH-1:0] core_rsp_tag,
    input  wire                     core_rsp_ready,

    output wire                     mem_req_valid,
    output wire                     mem_req_rw,
    output wire [LINE_SIZE-1:0]     mem_req_byteen,
    output wire [ADDR_WIDTH-1:0]    mem_req_addr,
    output wire [LINE_SIZE*8-1:0]   mem_req_data,
    output wire [MEM_TAG_WIDTH-1:0] mem_req_tag,
    output wire                     mem_req_fence,
    input  wire                     mem_req_ready,

    input  wire                     mem_rsp_valid,
    input  wire [LINE_SIZE*8-1:0]   mem_rsp_data,
    input  wire [MEM_TAG_WIDTH-1:0] mem_rsp_tag,
    output wire                     mem_rsp_ready
);
    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) core_bus_if[1]();

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (MEM_TAG_WIDTH)
    ) mem_bus_if[1]();

    // A fence of the LSU: is_flush with is_fence. Above the shared level it
    // walks the cache; the H100 L1 invalidates every line.
    mem_bus_attr_t attr;
    always @(*) begin
        attr = '0;
        attr.is_flush = core_req_flush;
        attr.is_fence = core_req_flush;
    end

    assign core_bus_if[0].req_valid       = core_req_valid;
    assign core_bus_if[0].req_data.rw     = core_req_rw;
    assign core_bus_if[0].req_data.byteen = core_req_byteen;
    assign core_bus_if[0].req_data.addr   = core_req_addr;
    assign core_bus_if[0].req_data.attr   = attr;
    assign core_bus_if[0].req_data.data   = core_req_data;
    assign core_bus_if[0].req_data.tag    = TAG_WIDTH'(core_req_tag);
    assign core_req_ready = core_bus_if[0].req_ready;

    assign core_rsp_valid = core_bus_if[0].rsp_valid;
    assign core_rsp_data  = core_bus_if[0].rsp_data.data;
    assign core_rsp_tag   = core_bus_if[0].rsp_data.tag.value;
    assign core_bus_if[0].rsp_ready = core_rsp_ready;
    `UNUSED_VAR (core_bus_if[0].rsp_data.tag.uuid)

    assign mem_req_valid  = mem_bus_if[0].req_valid;
    assign mem_req_rw     = mem_bus_if[0].req_data.rw;
    assign mem_req_byteen = mem_bus_if[0].req_data.byteen;
    assign mem_req_addr   = mem_bus_if[0].req_data.addr;
    assign mem_req_data   = mem_bus_if[0].req_data.data;
    assign mem_req_tag    = mem_bus_if[0].req_data.tag;
    mem_bus_attr_t mem_attr;
    assign mem_attr       = mem_bus_attr_t'(mem_bus_if[0].req_data.attr);
    assign mem_req_fence  = mem_attr.is_fence || mem_attr.is_flush;
    `UNUSED_VAR (mem_attr)
    assign mem_bus_if[0].req_ready = mem_req_ready;

    assign mem_bus_if[0].rsp_valid     = mem_rsp_valid;
    assign mem_bus_if[0].rsp_data.data = mem_rsp_data;
    assign mem_bus_if[0].rsp_data.tag  = mem_rsp_tag;
    assign mem_rsp_ready = mem_bus_if[0].rsp_ready;

    VX_cache_wrap #(
        .INSTANCE_ID  ("l1h100"),
        .CACHE_SIZE   (`VX_CFG_DCACHE_SIZE),
        .LINE_SIZE    (LINE_SIZE),
        .SECTOR_SIZE  (LINE_SIZE),
        .NUM_BANKS    (`VX_CFG_DCACHE_NUM_BANKS),
        .NUM_WAYS     (`VX_CFG_DCACHE_NUM_WAYS),
        .WORD_SIZE    (LINE_SIZE),
        .NUM_REQS     (1),
        .MEM_PORTS    (1),
        .CRSQ_SIZE    (`VX_CFG_DCACHE_CRSQ_SIZE),
        .MSHR_SIZE    (MSHR_SIZE),
        .MRSQ_SIZE    (`VX_CFG_DCACHE_MRSQ_SIZE),
        .MREQ_SIZE    (`VX_CFG_DCACHE_MREQ_SIZE),
        .LATENCY      (`VX_CFG_DCACHE_LATENCY),
        .TAG_WIDTH    (TAG_WIDTH),
        .WRITE_ENABLE (1),
        .WRITEBACK    (`VX_CFG_DCACHE_WRITEBACK),
        .DIRTY_BYTES  (`VX_CFG_DCACHE_DIRTYBYTES),
        .REPL_POLICY  (`VX_CFG_DCACHE_REPL_POLICY),
        .IS_LLC       (0),
        .H100_L1      (1),
        .CORE_OUT_BUF (3),
        .MEM_OUT_BUF  (3)
    ) cache (
    `ifdef PERF_ENABLE
        `UNUSED_PIN (cache_perf),
    `endif
        .clk          (clk),
        .reset        (reset),
        .core_bus_if  (core_bus_if),
        .mem_bus_if   (mem_bus_if)
    );

endmodule
