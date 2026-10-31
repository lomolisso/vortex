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

// Per-socket stage of a cluster's DLMEM network: SOCKET_SIZE core ports plus,
// with several sockets, one uplink to the cluster crossbar. Port order at the
// round-robin arbiters is cores 0..SOCKET_SIZE-1, then the uplink.
module VX_dlmem_socket_switch #(
    parameter SOCKET_SIZE = 1,
    parameter NUM_SOCKETS = 1,
    // This socket's index inside its cluster, not VX_socket's global SOCKET_ID.
    parameter SOCKET_IDX  = 0,
    parameter DATAW       = 1,
    parameter DST_W       = 1,
`ifdef DXA_MBAR_ENABLE
    // 1: the socket's DXA row egress is one more input, after the uplink.
    // Nothing is addressed to it, so it has no output.
    parameter ROW_IN      = 0,
`endif
    parameter OUT_BUF     = 3
) (
    input wire              clk,
    input wire              reset,

    VX_dlmem_bus_if.slave   core_in_if [SOCKET_SIZE],
    VX_dlmem_bus_if.master  core_out_if [SOCKET_SIZE],

`ifdef DXA_MBAR_ENABLE
    VX_dlmem_bus_if.slave   row_in_if,
`endif

    VX_dlmem_bus_if.slave   uplink_in_if,
    VX_dlmem_bus_if.master  uplink_out_if
);
    localparam NCC        = NUM_SOCKETS * SOCKET_SIZE;
    localparam HAS_UPLINK = (NUM_SOCKETS > 1);
    localparam NUM_PORTS  = SOCKET_SIZE + (HAS_UPLINK ? 1 : 0);
    localparam UPLINK     = SOCKET_SIZE;
    localparam SEL_W      = `LOG2UP(NUM_PORTS);
    localparam PKT_W      = DST_W + DATAW;
`ifdef DXA_MBAR_ENABLE
    localparam NUM_IN     = NUM_PORTS + ROW_IN;
    localparam ROW_PORT   = NUM_PORTS;
`endif

    `STATIC_ASSERT (SOCKET_IDX < NUM_SOCKETS, ("invalid parameter"))
    `STATIC_ASSERT (DST_W >= `CLOG2(NCC), ("invalid parameter"))

    function automatic logic [SEL_W-1:0] core_port(input logic [DST_W-1:0] dst);
        core_port = SEL_W'(32'(dst) % SOCKET_SIZE);
    endfunction

    function automatic logic in_socket(input logic [DST_W-1:0] dst);
        in_socket = ((32'(dst) / SOCKET_SIZE) == SOCKET_IDX);
    endfunction

`ifdef DXA_MBAR_ENABLE
    wire [NUM_IN-1:0]                valid_in;
    wire [NUM_IN-1:0][PKT_W-1:0]     data_in;
    wire [NUM_IN-1:0][SEL_W-1:0]     sel_in;
    wire [NUM_IN-1:0]                ready_in;
`else
    wire [NUM_PORTS-1:0]             valid_in;
    wire [NUM_PORTS-1:0][PKT_W-1:0]  data_in;
    wire [NUM_PORTS-1:0][SEL_W-1:0]  sel_in;
    wire [NUM_PORTS-1:0]             ready_in;
`endif

    wire [NUM_PORTS-1:0]             valid_out;
    wire [NUM_PORTS-1:0][PKT_W-1:0]  data_out;
    wire [NUM_PORTS-1:0]             ready_out;

    for (genvar i = 0; i < SOCKET_SIZE; ++i) begin : g_core_in
        wire [DST_W-1:0] dst = core_in_if[i].dst;
        assign valid_in[i] = core_in_if[i].valid;
        assign data_in[i]  = {dst, core_in_if[i].data};
        if (HAS_UPLINK) begin : g_route
            assign sel_in[i] = in_socket(dst) ? core_port(dst) : SEL_W'(UPLINK);
        end else begin : g_local
            assign sel_in[i] = core_port(dst);
        end
        assign core_in_if[i].ready = ready_in[i];

        `RUNTIME_ASSERT (~core_in_if[i].valid || (32'(dst) < NCC),
            ("%t: dlmem switch %0d: core %0d sends to core %0d, outside the cluster (%0d cores)", $time, SOCKET_IDX, i, dst, NCC))
    end

    for (genvar i = 0; i < SOCKET_SIZE; ++i) begin : g_core_out
        assign core_out_if[i].valid = valid_out[i];
        assign {core_out_if[i].dst, core_out_if[i].data} = data_out[i];
        assign ready_out[i] = core_out_if[i].ready;
    end

    // Uplink packets only ever leave on a core port, so the network is a tree
    // and adds no dependency cycle of its own.
    if (HAS_UPLINK) begin : g_uplink
        assign valid_in[UPLINK] = uplink_in_if.valid;
        assign data_in[UPLINK]  = {uplink_in_if.dst, uplink_in_if.data};
        assign sel_in[UPLINK]   = core_port(uplink_in_if.dst);
        assign uplink_in_if.ready = ready_in[UPLINK];

        assign uplink_out_if.valid = valid_out[UPLINK];
        assign {uplink_out_if.dst, uplink_out_if.data} = data_out[UPLINK];
        assign ready_out[UPLINK] = uplink_out_if.ready;

        `RUNTIME_ASSERT (~uplink_in_if.valid || in_socket(uplink_in_if.dst),
            ("%t: dlmem switch %0d: uplink delivers a packet for core %0d of another socket", $time, SOCKET_IDX, uplink_in_if.dst))
    end else begin : g_no_uplink
        assign uplink_in_if.ready  = 1'b0;
        assign uplink_out_if.valid = 1'b0;
        assign uplink_out_if.data  = '0;
        assign uplink_out_if.dst   = '0;
        `UNUSED_VAR (uplink_in_if.valid)
        `UNUSED_VAR (uplink_in_if.data)
        `UNUSED_VAR (uplink_in_if.dst)
        `UNUSED_VAR (uplink_out_if.ready)

        `RUNTIME_ASSERT (~uplink_in_if.valid,
            ("%t: dlmem switch %0d: packet on the uplink of a single-socket cluster", $time, SOCKET_IDX))
    end

`ifdef DXA_MBAR_ENABLE
    if (ROW_IN != 0) begin : g_row_in
        wire [DST_W-1:0] row_dst = row_in_if.dst;
        assign valid_in[ROW_PORT] = row_in_if.valid;
        assign data_in[ROW_PORT]  = {row_dst, row_in_if.data};
        if (HAS_UPLINK) begin : g_route
            assign sel_in[ROW_PORT] = in_socket(row_dst) ? core_port(row_dst) : SEL_W'(UPLINK);
        end else begin : g_local
            assign sel_in[ROW_PORT] = core_port(row_dst);
        end
        assign row_in_if.ready = ready_in[ROW_PORT];

        `RUNTIME_ASSERT (~row_in_if.valid || (32'(row_dst) < NCC),
            ("%t: dlmem switch %0d: DXA rows sent to core %0d, outside the cluster (%0d cores)", $time, SOCKET_IDX, row_dst, NCC))
    end else begin : g_no_row_in
        assign row_in_if.ready = 1'b0;
        `UNUSED_VAR (row_in_if.valid)
        `UNUSED_VAR (row_in_if.data)
        `UNUSED_VAR (row_in_if.dst)
    end
`endif

    // MAX_FANOUT 0: one flat arbiter per output at any port count, so a hop
    // stays one registered stage; the fan-out tree adds a stage above 12 ports.
    VX_stream_xbar #(
    `ifdef DXA_MBAR_ENABLE
        .NUM_INPUTS  (NUM_IN),
    `else
        .NUM_INPUTS  (NUM_PORTS),
    `endif
        .NUM_OUTPUTS (NUM_PORTS),
        .DATAW       (PKT_W),
        .ARBITER     ("R"),
        .MAX_FANOUT  (0),
        .OUT_BUF     (OUT_BUF)
    ) xbar (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .data_in    (data_in),
        .sel_in     (sel_in),
        .ready_in   (ready_in),
        .valid_out  (valid_out),
        .data_out   (data_out),
        `UNUSED_PIN (sel_out),
        .ready_out  (ready_out),
        `UNUSED_PIN (collisions)
    );

endmodule
