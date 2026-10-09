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

// Cluster stage of the DLMEM network: connects the socket switches' uplinks,
// routing on the destination core's socket. Port i is socket i of the cluster.
module VX_dlmem_xbar #(
    parameter SOCKET_SIZE = 1,
    parameter NUM_SOCKETS = 1,
    parameter DATAW       = 1,
    parameter DST_W       = 1,
    parameter OUT_BUF     = 3
) (
    input wire              clk,
    input wire              reset,

    VX_dlmem_bus_if.slave   bus_in_if [NUM_SOCKETS],
    VX_dlmem_bus_if.master  bus_out_if [NUM_SOCKETS]
);
    localparam SEL_W = `LOG2UP(NUM_SOCKETS);
    localparam PKT_W = DST_W + DATAW;

    `STATIC_ASSERT (DST_W >= `CLOG2(NUM_SOCKETS * SOCKET_SIZE), ("invalid parameter"))

    wire [NUM_SOCKETS-1:0]             valid_in;
    wire [NUM_SOCKETS-1:0][PKT_W-1:0]  data_in;
    wire [NUM_SOCKETS-1:0][SEL_W-1:0]  sel_in;
    wire [NUM_SOCKETS-1:0]             ready_in;

    wire [NUM_SOCKETS-1:0]             valid_out;
    wire [NUM_SOCKETS-1:0][PKT_W-1:0]  data_out;
    wire [NUM_SOCKETS-1:0]             ready_out;

    for (genvar i = 0; i < NUM_SOCKETS; ++i) begin : g_in
        wire [31:0] dst_socket = 32'(bus_in_if[i].dst) / SOCKET_SIZE;
        assign valid_in[i] = bus_in_if[i].valid;
        assign data_in[i]  = {bus_in_if[i].dst, bus_in_if[i].data};
        assign sel_in[i]   = SEL_W'(dst_socket);
        assign bus_in_if[i].ready = ready_in[i];

        `RUNTIME_ASSERT (~bus_in_if[i].valid || (dst_socket < NUM_SOCKETS),
            ("%t: dlmem xbar: socket %0d sends to core %0d, outside the cluster", $time, i, bus_in_if[i].dst))
    end

    for (genvar i = 0; i < NUM_SOCKETS; ++i) begin : g_out
        assign bus_out_if[i].valid = valid_out[i];
        assign {bus_out_if[i].dst, bus_out_if[i].data} = data_out[i];
        assign ready_out[i] = bus_out_if[i].ready;
    end

    // MAX_FANOUT 0: one flat arbiter per output at any port count, so a hop
    // stays one registered stage; the fan-out tree adds a stage above 12 ports.
    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_SOCKETS),
        .NUM_OUTPUTS (NUM_SOCKETS),
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
