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

// Flat-port wrapper around VX_socket.
//
// VX_socket exposes SystemVerilog interface ports (VX_dcr_bus_if,
// VX_mem_bus_if[L1_MEM_PORTS], VX_gbar_bus_if).  This wrapper exposes the
// same block through plain `wire` ports so that:
//   - Synopsys DC writes a netlist whose top-level pins line up 1:1 with
//     the physical pins produced by PnR.
//   - The resulting .lef / .lib is straightforward to pair with a matching
//     blackbox stub (no unpacked interface instances on the boundary).
//   - The GPGPU synthesis run (Vortex top) can later treat VX_socket_top
//     as a hard macro: same module name, same flat port list.
//
// Mirrors the style of VX_core_top in hw/rtl/core/VX_core_top.sv, one level
// up the hierarchy (socket = 1 VX_core + its private I$/D$/lmem/regfile
// when SOCKET_SIZE=1).

module VX_socket_top import VX_gpu_pkg::*; #(
    parameter SOCKET_ID = 0
) (
    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // DCR write channel
    input  wire                             dcr_write_valid,
    input  wire [VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input  wire [VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    // L1 memory master (one set of signals per L1_MEM_PORTS port)
    output wire [`L1_MEM_PORTS-1:0]                                             mem_req_valid,
    output wire [`L1_MEM_PORTS-1:0]                                             mem_req_rw,
    output wire [`L1_MEM_PORTS-1:0][`L1_LINE_SIZE-1:0]                          mem_req_byteen,
    output wire [`L1_MEM_PORTS-1:0][`MEM_ADDR_WIDTH-`CLOG2(`L1_LINE_SIZE)-1:0]  mem_req_addr,
    output wire [`L1_MEM_PORTS-1:0][`L1_LINE_SIZE*8-1:0]                        mem_req_data,
    output wire [`L1_MEM_PORTS-1:0][MEM_FLAGS_WIDTH-1:0]                        mem_req_flags,
    output wire [`L1_MEM_PORTS-1:0][L1_MEM_ARB_TAG_WIDTH-1:0]                   mem_req_tag,
    input  wire [`L1_MEM_PORTS-1:0]                                             mem_req_ready,

    input  wire [`L1_MEM_PORTS-1:0]                                             mem_rsp_valid,
    input  wire [`L1_MEM_PORTS-1:0][`L1_LINE_SIZE*8-1:0]                        mem_rsp_data,
    input  wire [`L1_MEM_PORTS-1:0][L1_MEM_ARB_TAG_WIDTH-1:0]                   mem_rsp_tag,
    output wire [`L1_MEM_PORTS-1:0]                                             mem_rsp_ready,

`ifdef GBAR_ENABLE
    // Global barrier master
    output wire                             gbar_req_valid,
    output wire [NB_WIDTH-1:0]              gbar_req_id,
    output wire [NC_WIDTH-1:0]              gbar_req_size_m1,
    output wire [NC_WIDTH-1:0]              gbar_req_core_id,
    input  wire                             gbar_req_ready,
    input  wire                             gbar_rsp_valid,
    input  wire [NB_WIDTH-1:0]              gbar_rsp_id,
`endif

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
    // L1 memory bus: build an interface array and connect each port's
    // flat wires to the corresponding interface fields.
    //-------------------------------------------------------------------
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) mem_bus_if [`L1_MEM_PORTS]();

    for (genvar i = 0; i < `L1_MEM_PORTS; ++i) begin : g_mem_bus_if
        assign mem_req_valid[i]             = mem_bus_if[i].req_valid;
        assign mem_req_rw[i]                = mem_bus_if[i].req_data.rw;
        assign mem_req_byteen[i]            = mem_bus_if[i].req_data.byteen;
        assign mem_req_addr[i]              = mem_bus_if[i].req_data.addr;
        assign mem_req_data[i]              = mem_bus_if[i].req_data.data;
        assign mem_req_flags[i]             = mem_bus_if[i].req_data.flags;
        assign mem_req_tag[i]               = mem_bus_if[i].req_data.tag;
        assign mem_bus_if[i].req_ready      = mem_req_ready[i];

        assign mem_bus_if[i].rsp_valid      = mem_rsp_valid[i];
        assign mem_bus_if[i].rsp_data.data  = mem_rsp_data[i];
        assign mem_bus_if[i].rsp_data.tag   = mem_rsp_tag[i];
        assign mem_rsp_ready[i]             = mem_bus_if[i].rsp_ready;
    end

`ifdef GBAR_ENABLE
    //-------------------------------------------------------------------
    // Global barrier bus
    //-------------------------------------------------------------------
    VX_gbar_bus_if gbar_bus_if();

    assign gbar_req_valid              = gbar_bus_if.req_valid;
    assign gbar_req_id                 = gbar_bus_if.req_data.id;
    assign gbar_req_size_m1            = gbar_bus_if.req_data.size_m1;
    assign gbar_req_core_id            = gbar_bus_if.req_data.core_id;
    assign gbar_bus_if.req_ready       = gbar_req_ready;

    assign gbar_bus_if.rsp_valid       = gbar_rsp_valid;
    assign gbar_bus_if.rsp_data.id     = gbar_rsp_id;
`endif

`ifdef PERF_ENABLE
    // VX_socket requires a sysmem_perf input when PERF_ENABLE is set.
    // The wrapper's ports are flat wires, so we drive it with zeros here.
    // Performance counter plumbing lives at the Vortex level for the full
    // GPGPU and is irrelevant for standalone socket synthesis runs.
    sysmem_perf_t sysmem_perf_stub;
    assign sysmem_perf_stub = '0;
`endif

    //-------------------------------------------------------------------
    // Instantiate the socket itself.  INSTANCE_ID is hardcoded to
    // "socket_top" — we don't forward it through the wrapper because the
    // wrapper is the synthesis/blackbox boundary; any surrounding
    // hierarchy uses module-name uniqueness (multiple VX_socket_top
    // instances at the VX_cluster level) rather than INSTANCE_ID strings.
    //-------------------------------------------------------------------
    VX_socket #(
        .SOCKET_ID   (SOCKET_ID),
        .INSTANCE_ID (`SFORMATF(("socket_top")))
    ) socket (
        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .sysmem_perf    (sysmem_perf_stub),
    `endif

        .dcr_bus_if     (dcr_bus_if),
        .mem_bus_if     (mem_bus_if),

    `ifdef GBAR_ENABLE
        .gbar_bus_if    (gbar_bus_if),
    `endif

        .busy           (busy)
    );

endmodule
