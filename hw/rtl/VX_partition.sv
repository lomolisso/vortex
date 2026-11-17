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

// One partition: its GPCs (VX_cluster), wired through without a register, so
// the level adds no cycle. The memory side keeps each GPC's own L2 for now.
module VX_partition import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter PARTITION_ID = 0
) (
    `SCOPE_IO_DECL

    input  wire                 clk,
    input  wire                 reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t         sysmem_perf,
`endif

    VX_dcr_bus_if.slave         dcr_bus_if [GPCS_PER_PARTITION],

    VX_mem_bus_if.master        mem_bus_if [GPCS_PER_PARTITION * L2_MEM_PORTS],

    VX_kmu_bus_if.slave         kmu_bus_if [GPCS_PER_PARTITION],

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_launch_if.slave   raster_launch_if [GPCS_PER_PARTITION],
`endif

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    input  wire                 gang_mode,
    input  wire [GPCS_PER_PARTITION-1:0] alloc_valid,
    input  wire [GANG_ID_W-1:0] alloc_gang,
    input  wire                 alloc_epoch,
    input  wire [DLMEM_K_W-1:0] alloc_k,
    output wire [GPCS_PER_PARTITION-1:0] done_valid,
    output dlmem_cta_done_t [GPCS_PER_PARTITION-1:0] done_data,
    output wire [GPCS_PER_PARTITION-1:0][DLMEM_NUM_GANGS-1:0] rel_pend,
`endif

`ifdef VX_CFG_VM_ENABLE
    input  wire [`VX_CFG_XLEN-1:0] mmu_satp,
    VX_tlb_bus_if.master        dev_ptw_if [GPCS_PER_PARTITION],
    VX_mem_bus_if.slave         dev_ptw_mem_if [GPCS_PER_PARTITION],
    input  wire                 mmu_flush_req,
    output wire [GPCS_PER_PARTITION-1:0] mmu_flush_done,
`endif

    output wire [GPCS_PER_PARTITION-1:0] busy
);

`ifdef SCOPE
    localparam scope_cluster = 0;
    `SCOPE_IO_SWITCH (GPCS_PER_PARTITION);
`endif

    for (genvar g = 0; g < GPCS_PER_PARTITION; ++g) begin : g_clusters

        localparam CLUSTER_ID = PARTITION_ID * GPCS_PER_PARTITION + g;

        // A GPC is its own SimX execution domain in H100 builds: its memory
        // buses leave it through a register stage, as in SimX.
        VX_mem_bus_if #(
            .DATA_SIZE (L2_SECTOR_SIZE),
            .TAG_WIDTH (L3_TAG_WIDTH)
        ) gpc_mem_bus_if[L2_MEM_PORTS]();

        for (genvar m = 0; m < L2_MEM_PORTS; ++m) begin : g_mem_boundary
            VX_gpc_boundary #(
                .DATA_SIZE (L2_SECTOR_SIZE),
                .TAG_WIDTH (L3_TAG_WIDTH)
            ) mem_boundary (
                .clk        (clk),
                .reset      (reset),
                .bus_in_if  (gpc_mem_bus_if[m]),
                .bus_out_if (mem_bus_if[g * L2_MEM_PORTS + m])
            );
        end

    `ifdef VX_CFG_VM_ENABLE
        localparam PTW_MEM_DATA_SIZE = (PTW_ON_L2 != 0) ? `VX_CFG_L1_LINE_SIZE : L2_SECTOR_SIZE;
        localparam PTW_MEM_TAG_WIDTH = (PTW_ON_L2 != 0) ? L2_TAG_WIDTH : L3_TAG_WIDTH;

        VX_mem_bus_if #(
            .DATA_SIZE (PTW_MEM_DATA_SIZE),
            .TAG_WIDTH (PTW_MEM_TAG_WIDTH)
        ) gpc_ptw_mem_if();

        VX_gpc_boundary #(
            .DATA_SIZE (PTW_MEM_DATA_SIZE),
            .TAG_WIDTH (PTW_MEM_TAG_WIDTH)
        ) ptw_boundary (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (dev_ptw_mem_if[g]),
            .bus_out_if (gpc_ptw_mem_if)
        );
    `endif

        // sv2v cannot resolve a slice of a port array as a modport binding, so
        // the cluster's array ports go through local arrays.
        VX_kmu_bus_if gpc_kmu_bus_if[1]();
        assign gpc_kmu_bus_if[0].valid = kmu_bus_if[g].valid;
        assign gpc_kmu_bus_if[0].kind  = kmu_bus_if[g].kind;
        assign gpc_kmu_bus_if[0].eop   = kmu_bus_if[g].eop;
        assign gpc_kmu_bus_if[0].dest  = kmu_bus_if[g].dest;
        assign gpc_kmu_bus_if[0].data  = kmu_bus_if[g].data;
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        assign gpc_kmu_bus_if[0].pin   = kmu_bus_if[g].pin;
    `endif
        assign kmu_bus_if[g].ready = gpc_kmu_bus_if[0].ready;

    `ifdef VX_CFG_EXT_RASTER_ENABLE
        VX_raster_launch_if gpc_raster_launch_if[1]();
        assign gpc_raster_launch_if[0].valid = raster_launch_if[g].valid;
        assign raster_launch_if[g].ready = gpc_raster_launch_if[0].ready;
    `endif

        VX_cluster #(
            .CLUSTER_ID (CLUSTER_ID),
            .INSTANCE_ID (`SFORMATF(("cluster%0d", CLUSTER_ID)))
        ) cluster (
            `SCOPE_IO_BIND (scope_cluster + g)

            .clk                (clk),
            .reset              (reset),

        `ifdef PERF_ENABLE
            .sysmem_perf        (sysmem_perf),
        `endif

            .dcr_bus_if         (dcr_bus_if[g]),

            .mem_bus_if         (gpc_mem_bus_if),

            .kmu_bus_if         (gpc_kmu_bus_if),

        `ifdef VX_CFG_EXT_RASTER_ENABLE
            .raster_launch_if   (gpc_raster_launch_if),
        `endif

        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            .gang_mode          (gang_mode),
            .alloc_valid        (alloc_valid[g]),
            .alloc_gang         (alloc_gang),
            .alloc_epoch        (alloc_epoch),
            .alloc_k            (alloc_k),
            .done_valid         (done_valid[g]),
            .done_data          (done_data[g]),
            .rel_pend           (rel_pend[g]),
        `endif

        `ifdef VX_CFG_VM_ENABLE
            .mmu_satp           (mmu_satp),
            .dev_ptw_if         (dev_ptw_if[g]),
            .dev_ptw_mem_if     (gpc_ptw_mem_if),
            .mmu_flush_req      (mmu_flush_req),
            .mmu_flush_done     (mmu_flush_done[g]),
        `endif

            .busy               (busy[g])
        );
    end

endmodule
