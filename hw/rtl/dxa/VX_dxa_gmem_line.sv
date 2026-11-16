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

// A DXA block request onto the SM port's line-wide bus: the block's bytes of
// its line, its place in the line carried in the tag's top value bits to pick
// the reply.
module VX_dxa_gmem_line import VX_gpu_pkg::*; #(
    parameter IN_SIZE    = 64,
    parameter OUT_SIZE   = 128,
    parameter TAG_WIDTH  = 8
) (
    VX_mem_bus_if.slave  bus_in_if,
    VX_mem_bus_if.master bus_out_if
);
    localparam PARTS     = OUT_SIZE / IN_SIZE;
    localparam PART_BITS = `CLOG2(PARTS);
    localparam IN_ADDR_W = $bits(bus_in_if.req_data.addr);
    localparam VAL_W     = TAG_WIDTH - UUID_WIDTH;

    `STATIC_ASSERT (OUT_SIZE % IN_SIZE == 0 && (1 << PART_BITS) == PARTS, ("invalid parameter"))
    `STATIC_ASSERT ($bits(bus_out_if.req_data.tag) == TAG_WIDTH + PART_BITS, ("invalid parameter"))
    `STATIC_ASSERT ($bits(bus_out_if.req_data.addr) + PART_BITS == IN_ADDR_W, ("invalid parameter"))

    if (PARTS > 1) begin : g_parts
        wire [PART_BITS-1:0] req_part = bus_in_if.req_data.addr[PART_BITS-1:0];
        wire [PART_BITS-1:0] rsp_part = bus_out_if.rsp_data.tag.value[VAL_W +: PART_BITS];

        assign bus_out_if.req_valid            = bus_in_if.req_valid;
        assign bus_out_if.req_data.rw          = bus_in_if.req_data.rw;
        assign bus_out_if.req_data.addr        = bus_in_if.req_data.addr[IN_ADDR_W-1:PART_BITS];
        assign bus_out_if.req_data.data        = {PARTS{bus_in_if.req_data.data}};
        assign bus_out_if.req_data.byteen      = OUT_SIZE'(bus_in_if.req_data.byteen) << (32'(req_part) * IN_SIZE);
        assign bus_out_if.req_data.attr        = bus_in_if.req_data.attr;
        assign bus_out_if.req_data.tag.uuid    = bus_in_if.req_data.tag.uuid;
        assign bus_out_if.req_data.tag.value   = {req_part, bus_in_if.req_data.tag.value};
        assign bus_in_if.req_ready             = bus_out_if.req_ready;

        assign bus_in_if.rsp_valid             = bus_out_if.rsp_valid;
        assign bus_in_if.rsp_data.data         = bus_out_if.rsp_data.data[32'(rsp_part) * IN_SIZE * 8 +: IN_SIZE * 8];
        assign bus_in_if.rsp_data.tag.uuid     = bus_out_if.rsp_data.tag.uuid;
        assign bus_in_if.rsp_data.tag.value    = bus_out_if.rsp_data.tag.value[VAL_W-1:0];
        assign bus_out_if.rsp_ready            = bus_in_if.rsp_ready;
    end else begin : g_whole
        `ASSIGN_VX_MEM_BUS_IF (bus_out_if, bus_in_if);
    end

endmodule

`endif
