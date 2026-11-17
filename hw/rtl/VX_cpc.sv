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

// One CPC row of a GPC: its TPCs (VX_socket), wired through without a
// register, so the level adds no cycle. TPC slot t of CPC c is the GPC's
// socket t * CPCS_PER_GPC + c, the order the global SM index counts in.
module VX_cpc import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter CLUSTER_ID = 0,
    parameter CPC_ID     = 0,
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input  wire                 clk,
    input  wire                 reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t         sysmem_perf,
`endif

    VX_dcr_bus_if.slave         dcr_bus_if [TPCS_PER_CPC],

    VX_mem_bus_if.master        mem_bus_if [TPCS_PER_CPC * L1_MEM_PORTS],

`ifdef VX_CFG_VM_ENABLE
    VX_tlb_bus_if.master        cluster_tlb_bus_if [TPCS_PER_CPC],
`endif

`ifdef EXT_GFX_ANY_ENABLE
    VX_dcr_flush_if.master      cluster_flush_if [TPCS_PER_CPC],
`endif

    VX_kmu_bus_if.slave         kmu_bus_if [TPCS_PER_CPC],

    VX_gbar_bus_if.master       gbar_bus_if [TPCS_PER_CPC],

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    VX_dlmem_bus_if.master      dlmem_req_up_if [TPCS_PER_CPC],
    VX_dlmem_bus_if.slave       dlmem_req_down_if [TPCS_PER_CPC],
    VX_dlmem_bus_if.master      dlmem_rsp_up_if [TPCS_PER_CPC],
    VX_dlmem_bus_if.slave       dlmem_rsp_down_if [TPCS_PER_CPC],
`endif

    output wire [TPCS_PER_CPC-1:0] busy
);
    `UNUSED_SPARAM (INSTANCE_ID)

`ifdef SCOPE
    localparam scope_socket = 0;
    `SCOPE_IO_SWITCH (TPCS_PER_CPC);
`endif

    for (genvar t = 0; t < TPCS_PER_CPC; ++t) begin : g_sockets

        localparam SOCKET_IN_GPC = t * CPCS_PER_GPC + CPC_ID;

        // sv2v cannot resolve a slice of a port array as a modport binding, so
        // the socket's array ports go through local arrays.
        VX_mem_bus_if #(
            .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
            .TAG_WIDTH (L2_TAG_WIDTH)
        ) socket_mem_bus_if[L1_MEM_PORTS]();

        for (genvar m = 0; m < L1_MEM_PORTS; ++m) begin : g_mem
            `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[t * L1_MEM_PORTS + m], socket_mem_bus_if[m]);
        end

        VX_kmu_bus_if socket_kmu_bus_if[1]();
        assign socket_kmu_bus_if[0].valid = kmu_bus_if[t].valid;
        assign socket_kmu_bus_if[0].kind  = kmu_bus_if[t].kind;
        assign socket_kmu_bus_if[0].eop   = kmu_bus_if[t].eop;
        assign socket_kmu_bus_if[0].dest  = kmu_bus_if[t].dest;
        assign socket_kmu_bus_if[0].data  = kmu_bus_if[t].data;
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        assign socket_kmu_bus_if[0].pin   = kmu_bus_if[t].pin;
    `endif
        assign kmu_bus_if[t].ready = socket_kmu_bus_if[0].ready;

        VX_socket #(
            .SOCKET_ID   ((CLUSTER_ID * NUM_SOCKETS) + SOCKET_IN_GPC),
            .INSTANCE_ID (`SFORMATF(("%s-socket%0d", INSTANCE_ID, SOCKET_IN_GPC)))
        ) socket (
            `SCOPE_IO_BIND  (scope_socket + t)

            .clk            (clk),
            .reset          (reset),

        `ifdef PERF_ENABLE
            .sysmem_perf    (sysmem_perf),
        `endif

            .dcr_bus_if     (dcr_bus_if[t]),

            .mem_bus_if     (socket_mem_bus_if),

        `ifdef VX_CFG_VM_ENABLE
            .cluster_tlb_bus_if (cluster_tlb_bus_if[t]),
        `endif

        `ifdef EXT_GFX_ANY_ENABLE
            .cluster_flush_if (cluster_flush_if[t]),
        `endif

            .kmu_bus_if     (socket_kmu_bus_if),

            .gbar_bus_if    (gbar_bus_if[t]),

        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            .dlmem_req_up_if   (dlmem_req_up_if[t]),
            .dlmem_req_down_if (dlmem_req_down_if[t]),
            .dlmem_rsp_up_if   (dlmem_rsp_up_if[t]),
            .dlmem_rsp_down_if (dlmem_rsp_down_if[t]),
        `endif

            .busy           (busy[t])
        );
    end

endmodule
