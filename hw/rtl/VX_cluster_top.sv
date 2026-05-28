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

// Flat-port wrapper around VX_cluster.
//
// VX_cluster exposes SystemVerilog interface ports (VX_dcr_bus_if and
// VX_mem_bus_if[L2_MEM_PORTS]).  This wrapper exposes the same block
// through plain `wire` ports for the same reasons VX_socket_top exists
// at the socket level:
//   - DC writes a netlist whose top-level pins line up 1:1 with the
//     pins produced by PnR.
//   - The resulting .lef / .lib pairs cleanly with a flat blackbox stub.
//   - Higher-level integration (a future SoC top) can treat the cluster
//     as a hard macro.
//
// Mirrors the style of VX_socket_top one level up: the cluster (8 sockets
// + shared L2) is the unit being PnR'd, with VX_socket_top instances
// already linked from the 2d-single-core hard macro.

`ifdef ASIC_SYNTHESIS
module VX_cluster_top import VX_gpu_pkg::*; #(
    parameter CLUSTER_ID = 0
) (
    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // DCR write channel
    input  wire                             dcr_write_valid,
    input  wire [VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input  wire [VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    // L2 memory master (one set of signals per L2_MEM_PORTS port)
    output wire [`L2_MEM_PORTS-1:0]                                             mem_req_valid,
    output wire [`L2_MEM_PORTS-1:0]                                             mem_req_rw,
    output wire [`L2_MEM_PORTS-1:0][`L2_LINE_SIZE-1:0]                          mem_req_byteen,
    output wire [`L2_MEM_PORTS-1:0][`MEM_ADDR_WIDTH-`CLOG2(`L2_LINE_SIZE)-1:0]  mem_req_addr,
    output wire [`L2_MEM_PORTS-1:0][`L2_LINE_SIZE*8-1:0]                        mem_req_data,
    output wire [`L2_MEM_PORTS-1:0][MEM_FLAGS_WIDTH-1:0]                        mem_req_flags,
    output wire [`L2_MEM_PORTS-1:0][L2_MEM_TAG_WIDTH-1:0]                       mem_req_tag,
    input  wire [`L2_MEM_PORTS-1:0]                                             mem_req_ready,

    input  wire [`L2_MEM_PORTS-1:0]                                             mem_rsp_valid,
    input  wire [`L2_MEM_PORTS-1:0][`L2_LINE_SIZE*8-1:0]                        mem_rsp_data,
    input  wire [`L2_MEM_PORTS-1:0][L2_MEM_TAG_WIDTH-1:0]                       mem_rsp_tag,
    output wire [`L2_MEM_PORTS-1:0]                                             mem_rsp_ready,

    // Status
    output wire                             busy
);

    //-------------------------------------------------------------------
    // DCR bus: lift the three flat wires into a VX_dcr_bus_if instance.
    //-------------------------------------------------------------------
    VX_dcr_bus_if dcr_bus_if();

    assign dcr_bus_if.write_valid = dcr_write_valid;
    assign dcr_bus_if.write_addr  = dcr_write_addr;
    assign dcr_bus_if.write_data  = dcr_write_data;

    //-------------------------------------------------------------------
    // L2 memory bus: build an interface array and connect each port's
    // flat wires to the corresponding interface fields.
    //-------------------------------------------------------------------
    VX_mem_bus_if #(
        .DATA_SIZE (`L2_LINE_SIZE),
        .TAG_WIDTH (L2_MEM_TAG_WIDTH)
    ) mem_bus_if [`L2_MEM_PORTS]();

    for (genvar i = 0; i < `L2_MEM_PORTS; ++i) begin : g_mem_bus_if
        assign mem_req_valid[i]              = mem_bus_if[i].req_valid;
        assign mem_req_rw[i]                 = mem_bus_if[i].req_data.rw;
        assign mem_req_byteen[i]             = mem_bus_if[i].req_data.byteen;
        assign mem_req_addr[i]               = mem_bus_if[i].req_data.addr;
        assign mem_req_data[i]               = mem_bus_if[i].req_data.data;
        assign mem_req_flags[i]              = mem_bus_if[i].req_data.flags;
        assign mem_req_tag[i]                = mem_bus_if[i].req_data.tag;
        assign mem_bus_if[i].req_ready       = mem_req_ready[i];

        assign mem_bus_if[i].rsp_valid       = mem_rsp_valid[i];
        assign mem_bus_if[i].rsp_data.data   = mem_rsp_data[i];
        assign mem_bus_if[i].rsp_data.tag    = mem_rsp_tag[i];
        assign mem_rsp_ready[i]              = mem_bus_if[i].rsp_ready;
    end

`ifdef PERF_ENABLE
    // VX_cluster requires a sysmem_perf input when PERF_ENABLE is set.
    // The wrapper's ports are flat wires, so we drive it with zeros here.
    sysmem_perf_t sysmem_perf_stub;
    assign sysmem_perf_stub = '0;
`endif

    //-------------------------------------------------------------------
    // Instantiate the cluster itself.  INSTANCE_ID is hardcoded to
    // "cluster_top" — we don't forward it through the wrapper because the
    // wrapper is the synthesis/blackbox boundary.
    //-------------------------------------------------------------------
    VX_cluster #(
        .CLUSTER_ID  (CLUSTER_ID),
        .INSTANCE_ID (`SFORMATF(("cluster_top")))
    ) cluster (
        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .sysmem_perf    (sysmem_perf_stub),
    `endif

        .dcr_bus_if     (dcr_bus_if),
        .mem_bus_if     (mem_bus_if),
        .busy           (busy)
    );

endmodule
`endif
