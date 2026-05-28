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

`ifdef ASIC_SYNTHESIS
// Hard-macro boundary: localparams (not parameters) so DC can't uniquify on parent overrides.
module VX_local_mem_top import VX_gpu_pkg::*; #(
    localparam SIZE       = (1 << `LMEM_LOG_SIZE),
    localparam NUM_REQS   = `NUM_LSU_LANES,
    localparam NUM_BANKS  = `LMEM_NUM_BANKS,
    localparam WORD_SIZE  = LSU_WORD_SIZE,
    localparam TAG_WIDTH  = LMEM_TAG_WIDTH,
    localparam ADDR_WIDTH = `LMEM_LOG_SIZE - `CLOG2(WORD_SIZE)
) (
    input wire clk,
    input wire reset,

    // Core request
    input  wire [NUM_REQS-1:0]                 mem_req_valid,
    input  wire [NUM_REQS-1:0]                 mem_req_rw,
    input  wire [NUM_REQS-1:0][WORD_SIZE-1:0]  mem_req_byteen,
    input  wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] mem_req_addr,
    input  wire [NUM_REQS-1:0][MEM_FLAGS_WIDTH-1:0] mem_req_flags,
    input  wire [NUM_REQS-1:0][WORD_SIZE*8-1:0] mem_req_data,
    input  wire [NUM_REQS-1:0][TAG_WIDTH-1:0]  mem_req_tag,
    output wire [NUM_REQS-1:0]                 mem_req_ready,

    // Core response
    output wire [NUM_REQS-1:0]                 mem_rsp_valid,
    output wire [NUM_REQS-1:0][WORD_SIZE*8-1:0] mem_rsp_data,
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0]  mem_rsp_tag,
    input  wire [NUM_REQS-1:0]                 mem_rsp_ready
);
    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) mem_bus_if[NUM_REQS]();

     // memory request
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign mem_bus_if[i].req_valid = mem_req_valid[i];
        assign mem_bus_if[i].req_data.rw = mem_req_rw[i];
        assign mem_bus_if[i].req_data.byteen = mem_req_byteen[i];
        assign mem_bus_if[i].req_data.addr = mem_req_addr[i];
        assign mem_bus_if[i].req_data.flags = mem_req_flags[i];
        assign mem_bus_if[i].req_data.data = mem_req_data[i];
        assign mem_bus_if[i].req_data.tag = mem_req_tag[i];
        assign mem_req_ready[i] = mem_bus_if[i].req_ready;
    end

    // memory response
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign mem_rsp_valid[i] = mem_bus_if[i].rsp_valid;
        assign mem_rsp_data[i] = mem_bus_if[i].rsp_data.data;
        assign mem_rsp_tag[i] = mem_bus_if[i].rsp_data.tag;
        assign mem_bus_if[i].rsp_ready = mem_rsp_ready[i];
    end

    VX_local_mem #(
        .SIZE       (SIZE),
        .NUM_REQS   (NUM_REQS),
        .NUM_BANKS  (NUM_BANKS),
        .WORD_SIZE  (WORD_SIZE),
        .ADDR_WIDTH (ADDR_WIDTH),
        .TAG_WIDTH  (TAG_WIDTH),
        .OUT_BUF    (3)
    ) local_mem (
        .clk        (clk),
        .reset      (reset),
        .mem_bus_if (mem_bus_if)
    );

endmodule
`endif
