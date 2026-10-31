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

// One cluster's DLMEM network: a switch per socket plus the cluster crossbar,
// with flat injection (in_*) and ejection (out_*) ports indexed by core, and
// with DXA mbarriers a row injection port (row_*) per socket.
module VX_dlmem_noc_top import VX_gpu_pkg::*, VX_dlmem_pkg::*; #(
    parameter DATAW   = 160,
    parameter OUT_BUF = 3,

    localparam SOCKET_SIZE = `VX_CFG_SOCKET_SIZE,
    localparam NCC         = DLMEM_NCC,
    localparam DST_W       = DLMEM_CORE_W
) (
    input wire clk,
    input wire reset,

    input  wire [NCC-1:0]               in_valid,
    input  wire [NCC-1:0][DATAW-1:0]    in_data,
    input  wire [NCC-1:0][DST_W-1:0]    in_dst,
    output wire [NCC-1:0]               in_ready,

`ifdef DXA_MBAR_ENABLE
    // Per socket, the DXA row egress's input-only port.
    input  wire [NUM_SOCKETS-1:0]            row_valid,
    input  wire [NUM_SOCKETS-1:0][DATAW-1:0] row_data,
    input  wire [NUM_SOCKETS-1:0][DST_W-1:0] row_dst,
    output wire [NUM_SOCKETS-1:0]            row_ready,
`endif

    output wire [NCC-1:0]               out_valid,
    output wire [NCC-1:0][DATAW-1:0]    out_data,
    output wire [NCC-1:0][DST_W-1:0]    out_dst,
    input  wire [NCC-1:0]               out_ready
);
    // main.cpp derives the shape from VX_config.h.
    `STATIC_ASSERT (NCC == `UP(`VX_CFG_NUM_CORES / `VX_CFG_SOCKET_SIZE) * `VX_CFG_SOCKET_SIZE,
        ("dlmem_noc_top: %0d cores, the testbench expects a different count", NCC))

    VX_dlmem_bus_if #(
        .DATAW (DATAW),
        .DST_W (DST_W)
    ) uplink_up_if[NUM_SOCKETS]();

    VX_dlmem_bus_if #(
        .DATAW (DATAW),
        .DST_W (DST_W)
    ) uplink_down_if[NUM_SOCKETS]();

    for (genvar s = 0; s < NUM_SOCKETS; ++s) begin : g_sockets

        VX_dlmem_bus_if #(
            .DATAW (DATAW),
            .DST_W (DST_W)
        ) core_in_if[SOCKET_SIZE]();

        VX_dlmem_bus_if #(
            .DATAW (DATAW),
            .DST_W (DST_W)
        ) core_out_if[SOCKET_SIZE]();

        for (genvar c = 0; c < SOCKET_SIZE; ++c) begin : g_cores
            localparam i = s * SOCKET_SIZE + c;
            assign core_in_if[c].valid = in_valid[i];
            assign core_in_if[c].data  = in_data[i];
            assign core_in_if[c].dst   = in_dst[i];
            assign in_ready[i] = core_in_if[c].ready;

            assign out_valid[i] = core_out_if[c].valid;
            assign out_data[i]  = core_out_if[c].data;
            assign out_dst[i]   = core_out_if[c].dst;
            assign core_out_if[c].ready = out_ready[i];
        end

    `ifdef DXA_MBAR_ENABLE
        VX_dlmem_bus_if #(
            .DATAW (DATAW),
            .DST_W (DST_W)
        ) row_in_if();

        assign row_in_if.valid = row_valid[s];
        assign row_in_if.data  = row_data[s];
        assign row_in_if.dst   = row_dst[s];
        assign row_ready[s]    = row_in_if.ready;
    `endif

        VX_dlmem_socket_switch #(
            .SOCKET_SIZE (SOCKET_SIZE),
            .NUM_SOCKETS (NUM_SOCKETS),
            .SOCKET_IDX  (s),
            .DATAW       (DATAW),
            .DST_W       (DST_W),
        `ifdef DXA_MBAR_ENABLE
            .ROW_IN      (1),
        `endif
            .OUT_BUF     (OUT_BUF)
        ) socket_switch (
            .clk           (clk),
            .reset         (reset),
            .core_in_if    (core_in_if),
            .core_out_if   (core_out_if),
        `ifdef DXA_MBAR_ENABLE
            .row_in_if     (row_in_if),
        `endif
            .uplink_in_if  (uplink_down_if[s]),
            .uplink_out_if (uplink_up_if[s])
        );
    end

    if (NUM_SOCKETS > 1) begin : g_xbar
        VX_dlmem_xbar #(
            .SOCKET_SIZE (SOCKET_SIZE),
            .NUM_SOCKETS (NUM_SOCKETS),
            .DATAW       (DATAW),
            .DST_W       (DST_W),
            .OUT_BUF     (OUT_BUF)
        ) cluster_xbar (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (uplink_up_if),
            .bus_out_if (uplink_down_if)
        );
    end else begin : g_no_xbar
        assign uplink_down_if[0].valid = 1'b0;
        assign uplink_down_if[0].data  = '0;
        assign uplink_down_if[0].dst   = '0;
        assign uplink_up_if[0].ready   = 1'b0;
        `UNUSED_VAR (uplink_down_if[0].ready)
        `UNUSED_VAR (uplink_up_if[0].valid)
        `UNUSED_VAR (uplink_up_if[0].data)
        `UNUSED_VAR (uplink_up_if[0].dst)
    end

endmodule
