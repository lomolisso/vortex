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

module Vortex import VX_gpu_pkg::*, VX_trace_pkg::*, VX_tlb_pkg::*; (
    `SCOPE_IO_DECL

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // Memory request
    output wire                             mem_req_valid [VX_MEM_PORTS],
    output wire                             mem_req_rw [VX_MEM_PORTS],
    output wire [VX_MEM_BYTEEN_WIDTH-1:0]   mem_req_byteen [VX_MEM_PORTS],
    output wire [VX_MEM_ADDR_WIDTH-1:0]     mem_req_addr [VX_MEM_PORTS],
    output wire [VX_MEM_DATA_WIDTH-1:0]     mem_req_data [VX_MEM_PORTS],
    output wire [VX_MEM_TAG_WIDTH-1:0]      mem_req_tag [VX_MEM_PORTS],
    input  wire                             mem_req_ready [VX_MEM_PORTS],

    // Memory response
    input wire                              mem_rsp_valid [VX_MEM_PORTS],
    input wire [VX_MEM_DATA_WIDTH-1:0]      mem_rsp_data [VX_MEM_PORTS],
    input wire [VX_MEM_TAG_WIDTH-1:0]       mem_rsp_tag [VX_MEM_PORTS],
    output wire                             mem_rsp_ready [VX_MEM_PORTS],

    // DCR write request
    input  wire                             dcr_req_valid,
    input  wire                             dcr_req_rw,
    input  wire [VX_DCR_ADDR_WIDTH-1:0]     dcr_req_addr,
    input  wire [VX_DCR_DATA_WIDTH-1:0]     dcr_req_data,

    // DCR read response
    output wire                             dcr_rsp_valid,
    output wire [VX_DCR_DATA_WIDTH-1:0]     dcr_rsp_data,

    // ctrl/status
    input  wire                             start,
    output wire                             busy
);
    // Check clustering configuration
    `STATIC_ASSERT(`IS_POW2(`VX_CFG_NUM_CLUSTERS), ("NUM_CLUSTERS must be a power of 2"));
    `STATIC_ASSERT(`IS_POW2(`VX_CFG_NUM_CORES), ("NUM_CORES must be a power of 2"));
    `STATIC_ASSERT(`IS_POW2(`VX_CFG_SOCKET_SIZE), ("SOCKET_SIZE must be a power of 2"));
    `STATIC_ASSERT(`IS_POW2(`VX_CFG_NUM_PARTITIONS), ("NUM_PARTITIONS must be a power of 2"));
    `STATIC_ASSERT(`IS_POW2(`VX_CFG_GPCS_PER_PARTITION), ("GPCS_PER_PARTITION must be a power of 2"));
    `STATIC_ASSERT(`IS_POW2(`VX_CFG_CPCS_PER_GPC), ("CPCS_PER_GPC must be a power of 2"));
    `STATIC_ASSERT(`IS_POW2(`VX_CFG_TPCS_PER_CPC), ("TPCS_PER_CPC must be a power of 2"));
    `STATIC_ASSERT(`IS_POW2(`VX_CFG_SMS_PER_TPC), ("SMS_PER_TPC must be a power of 2"));
    `STATIC_ASSERT((`VX_CFG_NUM_CLUSTERS % `VX_CFG_NUM_PARTITIONS) == 0, ("NUM_PARTITIONS must divide NUM_CLUSTERS"));
    `STATIC_ASSERT((NUM_SOCKETS % CPCS_PER_GPC) == 0, ("CPCS_PER_GPC must divide the TPCs of a GPC"));
    `STATIC_ASSERT(`VX_CFG_NUM_SMS <= 1024, ("NUM_SMS must be at most 1024"));
    `STATIC_ASSERT((`VX_CFG_CLUSTER_GPC_START >= 0) && (`VX_CFG_CLUSTER_GPC_START < `VX_CFG_NUM_CLUSTERS),
        ("CLUSTER_GPC_START must name a GPC"));
    `STATIC_ASSERT((NUM_CTA_MAX >= 1) && (NUM_CTA_MAX <= `VX_CFG_NUM_WARPS) && (NUM_CTA_MAX <= 64),
        ("MAX_CTAS_PER_SM must be between 1 and min(NUM_WARPS, 64)"));
    `STATIC_ASSERT((`VX_CFG_MAX_CTAS_PER_SM_CLUSTER >= 1) && (`VX_CFG_MAX_CTAS_PER_SM_CLUSTER <= NUM_CTA_MAX),
        ("MAX_CTAS_PER_SM_CLUSTER must be between 1 and MAX_CTAS_PER_SM"));
    `STATIC_ASSERT(`VX_CFG_DOMAIN_BOUNDARY_DELAY >= 1, ("DOMAIN_BOUNDARY_DELAY must be at least 1"));
    `STATIC_ASSERT(`VX_CFG_DOMAIN_BOUNDARY_DELAY <= 1, ("DOMAIN_BOUNDARY_DELAY must be at most 1, the launch and credit edges' one register stage"));

    // Every cache strictly above the LLC must be write-through.
    // A WB intermediate could absorb a hart-B store without the LLC
    // seeing it; a later SC from hart-A on the same line would
    // spuriously succeed. RVA permits spurious failure, not spurious success.
`ifdef VX_CFG_EXT_A_ENABLE
  `ifdef VX_CFG_L3_ENABLE
    `STATIC_ASSERT(`VX_CFG_DCACHE_WRITEBACK == 0, ("AMO requires write-through L1 (DCACHE_WRITEBACK=0) when L3 is the LLC"));
    `STATIC_ASSERT(`VX_CFG_L2_WRITEBACK == 0,     ("AMO requires write-through L2 (L2_WRITEBACK=0) when L3 is the LLC"));
  `elsif VX_CFG_L2_ENABLE
    `STATIC_ASSERT(`VX_CFG_DCACHE_WRITEBACK == 0, ("AMO requires write-through L1 (DCACHE_WRITEBACK=0) when L2 is the LLC"));
  `endif

    // RVA atomicity is device-wide: every hart's AMOs and reservations must meet in one LLC instance.
    `STATIC_ASSERT((`VX_CFG_L3_ENABLED != 0)
                || ((`VX_CFG_NUM_CLUSTERS * ((`VX_CFG_L2_ENABLED != 0) ? 1
                       : (NUM_SOCKETS * `VX_CFG_NUM_DCACHES * `VX_CFG_DCACHE_ENABLED))) == 1),
        ("the A extension needs one shared last-level cache: enable L2 (one cluster) or L3 (several clusters)"));
`endif

`ifdef VX_CFG_EXT_DLMEM_ENABLE
`ifndef SYNTHESIS
    // Rejected at elaboration as a hard error: the simulators build with
    // non-fatal warnings, which would demote an elaboration $error.
    function automatic integer dlmem_reject(input logic bad, input string msg);
        if (bad) begin
            $fatal(1, "%s", msg);
        end
        return 0;
    endfunction
    localparam DLMEM_CFG_CHECKED =
          dlmem_reject((`VX_CFG_DLMEM_MAX_CLUSTER < 1) || (`VX_CFG_DLMEM_MAX_CLUSTER > `VX_CFG_NUM_CORES),
                       "DLMEM_MAX_CLUSTER must be between 1 and the cores of a cluster")
        + dlmem_reject(`VX_CFG_SOCKET_SIZE > `VX_CFG_NUM_CORES, "DLMEM requires SOCKET_SIZE <= NUM_CORES")
        + dlmem_reject(`VX_CFG_LMEM_LOG_SIZE > 18, "DLMEM requires a local memory of at most 256 KB")
        + dlmem_reject(64'(`VX_CFG_GLOBAL_MEM_SIZE) > 64'(`VX_MEM_DLMEM_BASE_ADDR),
                       "DLMEM requires the DLMEM window above global memory (GLOBAL_MEM_SIZE <= DLMEM_BASE_ADDR)")
        + dlmem_reject((64'(`VX_MEM_DLMEM_BASE_ADDR) % (64'(1) << (`VX_CFG_LMEM_LOG_SIZE + DLMEM_RANK_BITS))) != 0,
                       "DLMEM requires DLMEM_BASE_ADDR aligned to LMEM_SIZE x 2^clog2(DLMEM_MAX_CLUSTER)")
        + dlmem_reject((`VX_CFG_NUM_CORES % `VX_CFG_SOCKET_SIZE) != 0,
                       "DLMEM requires NUM_CORES to be a multiple of SOCKET_SIZE")
        // a pinned launch names its core field by field: cluster, socket, core in socket
        + dlmem_reject((((((`VX_CFG_NUM_CLUSTERS - 1) << KMU_DEST_LSB_DEVICE) | ((NUM_SOCKETS - 1) << KMU_DEST_LSB_CLUSTER)
                         | (`VX_CFG_SOCKET_SIZE - 1)) >> KMU_DEST_W) != 0),
                       "DLMEM requires the launch dest field to name every core of every cluster");
    `UNUSED_PARAM (DLMEM_CFG_CHECKED)
`endif
`endif

    VX_dcr_bus_if dcr_bus_if();
    assign dcr_bus_if.req_valid = dcr_req_valid;
    assign dcr_bus_if.req_data.rw = dcr_req_rw;
    assign dcr_bus_if.req_data.addr = dcr_req_addr;
    assign dcr_bus_if.req_data.data = dcr_req_data;
    assign dcr_rsp_valid = dcr_bus_if.rsp_valid;
    assign dcr_rsp_data = dcr_bus_if.rsp_data;

    // Kernel Management Unit
    VX_kmu_bus_if kmu_bus_in[1]();
    wire kmu_busy;
`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_launch_if raster_launch_if();
`endif
    VX_kmu #(
        .INSTANCE_ID ("kmu")
    ) kmu (
        .clk        (clk),
        .reset      (reset),
        .start      (start),
        .busy       (kmu_busy),
        .dcr_req_valid (dcr_req_valid),
        .dcr_req_rw (dcr_req_rw),
        .dcr_req_addr(dcr_req_addr),
        .dcr_req_data(dcr_req_data),
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        .raster_launch_if (raster_launch_if),
    `endif
        .kmu_bus_if (kmu_bus_in[0])
    );

`ifdef SCOPE
    localparam scope_partition = 0;
    `SCOPE_IO_SWITCH (NUM_PARTITIONS);
`endif

`ifdef PERF_ENABLE
    cache_perf_t l3_perf;
    mem_perf_t mem_perf;
    sysmem_perf_t sysmem_perf;
    always @(*) begin
        sysmem_perf = '0;
        sysmem_perf.l3cache = l3_perf;
        sysmem_perf.mem = mem_perf;
    end
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (L2_SECTOR_SIZE),
        .TAG_WIDTH (L3_TAG_WIDTH)
    ) per_cluster_mem_bus_if[L3_NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (L3_SECTOR_SIZE),
        .TAG_WIDTH (L3_MEM_TAG_WIDTH)
    ) mem_bus_if[L3_MEM_PORTS]();

    VX_cache_wrap #(
        .INSTANCE_ID    ("l3cache"),
        .CACHE_SIZE     (`VX_CFG_L3_SIZE),
        .LINE_SIZE      (`VX_CFG_L3_LINE_SIZE),
        .SECTOR_SIZE    (L3_SECTOR_SIZE),
        .NUM_BANKS      (L3_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_L3_NUM_WAYS),
        .WORD_SIZE      (L3_WORD_SIZE),
        .NUM_REQS       (L3_NUM_REQS),
        .MEM_PORTS      (L3_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_L3_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_L3_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_L3_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_L3_MREQ_SIZE),
        .LATENCY        (`VX_CFG_L3_LATENCY),
        .TAG_WIDTH      (L3_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`VX_CFG_L3_WRITEBACK),
        .DIRTY_BYTES    (`VX_CFG_L3_DIRTYBYTES),
        .REPL_POLICY    (`VX_CFG_L3_REPL_POLICY),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`VX_CFG_L3_ENABLED),
        .IS_LLC         (L3_IS_LLC),
        // every cluster and the page walker meet in the L3: a fence drains it
        .SHARED_LEVEL   (`VX_CFG_L3_ENABLED),
        .AMO_ENABLE     (`VX_CFG_EXT_A_ENABLED)
    ) l3cache (
        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .cache_perf     (l3_perf),
    `endif

        .core_bus_if    (per_cluster_mem_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

    for (genvar i = 0; i < L3_MEM_PORTS; ++i) begin : g_mem_bus_if
        assign mem_req_valid[i]  = mem_bus_if[i].req_valid;
        assign mem_req_rw[i]     = mem_bus_if[i].req_data.rw;
        assign mem_req_byteen[i] = mem_bus_if[i].req_data.byteen;
        assign mem_req_addr[i]   = mem_bus_if[i].req_data.addr;
        assign mem_req_data[i]   = mem_bus_if[i].req_data.data;
        assign mem_req_tag[i]    = mem_bus_if[i].req_data.tag;
        `UNUSED_VAR (mem_bus_if[i].req_data.attr)
        assign mem_bus_if[i].req_ready = mem_req_ready[i];

        assign mem_bus_if[i].rsp_valid     = mem_rsp_valid[i];
        assign mem_bus_if[i].rsp_data.data = mem_rsp_data[i];
        assign mem_bus_if[i].rsp_data.tag  = mem_rsp_tag[i];
        assign mem_rsp_ready[i] = mem_bus_if[i].rsp_ready;
    end

    wire [`VX_CFG_NUM_CLUSTERS-1:0] per_cluster_busy;

    VX_kmu_bus_if per_cluster_kmu_bus_if[`VX_CFG_NUM_CLUSTERS]();

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // Cluster admission for the whole chip, ahead of the per-GPC launch lanes;
    // its mirrors learn of finished blocks from each GPC's gbar unit.
    localparam NGPC = `VX_CFG_NUM_CLUSTERS;
    // a GPC is its own SimX execution domain in H100 builds: every edge to it is registered
    localparam BD = `VX_CFG_H100_ORG_ENABLED ? `VX_CFG_DOMAIN_BOUNDARY_DELAY : 0;

    VX_kmu_bus_if admit_kmu_bus_if[1]();
    wire admit_busy;

    wire [NGPC-1:0]                      gpc_done_valid, admit_done_valid;
    dlmem_cta_done_t [NGPC-1:0]          gpc_done_data, admit_done_data;
    wire [NGPC-1:0][DLMEM_NUM_GANGS-1:0] gpc_rel_pend, admit_rel_pend;
    wire [NGPC-1:0]                      admit_alloc_valid, gpc_alloc_valid;
    wire [GANG_ID_W-1:0]                 admit_alloc_gang, gpc_alloc_gang;
    wire                                 admit_alloc_epoch, gpc_alloc_epoch;
    wire [DLMEM_K_W-1:0]                 admit_alloc_k, gpc_alloc_k;
    wire                                 admit_gang_mode, gpc_gang_mode;

    VX_cta_admit #(
        .INSTANCE_ID ("cta_admit"),
        .NUM_GPCS    (NGPC),
        .OUT_BUF     (3)
    ) cta_admit (
        .clk         (clk),
        .reset       (reset),
        .bus_in_if   (kmu_bus_in[0]),
        .bus_out_if  (admit_kmu_bus_if[0]),
        .done_valid  (admit_done_valid),
        .done_data   (admit_done_data),
        .rel_pend    (admit_rel_pend),
        .alloc_valid (admit_alloc_valid),
        .alloc_gang  (admit_alloc_gang),
        .alloc_epoch (admit_alloc_epoch),
        .alloc_k     (admit_alloc_k),
        .gang_mode   (admit_gang_mode),
        .busy        (admit_busy)
    );

    `BUFFER_EX({admit_done_valid, admit_done_data, admit_rel_pend},
               {gpc_done_valid, gpc_done_data, gpc_rel_pend}, 1'b1, 1, BD);
    `BUFFER_EX({gpc_alloc_valid, gpc_alloc_gang, gpc_alloc_epoch, gpc_alloc_k, gpc_gang_mode},
               {admit_alloc_valid, admit_alloc_gang, admit_alloc_epoch, admit_alloc_k, admit_gang_mode}, 1'b1, 1, BD);

    VX_kmu_bus_arb #(
        .NUM_INPUTS (1),
        .NUM_OUTPUTS (`VX_CFG_NUM_CLUSTERS),
        .DEST_LSB   (KMU_DEST_LSB_DEVICE),
        .OUT_BUF    ((`VX_CFG_NUM_CLUSTERS > 1) ? 3 : 0)  // register per-cluster kmu fan-out (SLR-crossing skid)
    ) kmu_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (admit_kmu_bus_if),
        .bus_out_if (per_cluster_kmu_bus_if)
    );
`else
    wire admit_busy = 1'b0;

    VX_kmu_bus_arb #(
        .NUM_INPUTS (1),
        .NUM_OUTPUTS (`VX_CFG_NUM_CLUSTERS),
        .DEST_LSB   (KMU_DEST_LSB_DEVICE),
        .OUT_BUF    ((`VX_CFG_NUM_CLUSTERS > 1) ? 3 : 0)  // register per-cluster kmu fan-out (SLR-crossing skid)
    ) kmu_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (kmu_bus_in),
        .bus_out_if (per_cluster_kmu_bus_if)
    );
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_launch_if per_cluster_raster_launch_if[`VX_CFG_NUM_CLUSTERS]();
    VX_raster_launch_fork #(
        .NUM_OUTPUTS (`VX_CFG_NUM_CLUSTERS)
    ) raster_launch_fork (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (raster_launch_if),
        .bus_out_if (per_cluster_raster_launch_if)
    );
`endif

    // The device MMU surface filters the DCR stream: it assembles the
    // page-table root, pulses the TLB flush, and answers fault reads before
    // the rest of the DCR traffic fans to the clusters.
    VX_dcr_bus_if dcr_cluster_src_if();
`ifdef VX_CFG_VM_ENABLE
    wire [`VX_CFG_XLEN-1:0] mmu_satp;
    wire                    mmu_flush_req;
    wire [`VX_CFG_NUM_CLUSTERS-1:0] cl_mmu_flush_done;

    // Per-cluster L2-TLB miss export buses (used by the device-level walker).
    VX_tlb_bus_if #(.ID_WIDTH (L2_TLB_SLOT_WIDTH)) per_cluster_dev_ptw_if [`VX_CFG_NUM_CLUSTERS] ();

    // One shared walker at the device: the clusters' L2-TLB miss buses arb
    // into it, and the flush done-tree gains the walker's leg. Structural
    // faults surface here.
    VX_tlb_bus_if #(.ID_WIDTH (TLB_DEV_ID_WIDTH)) dev_ptw_bus_if ();

    VX_tlb_bus_arb #(
        .NUM_INPUTS  (`VX_CFG_NUM_CLUSTERS),
        .ID_WIDTH_IN (L2_TLB_SLOT_WIDTH),
        .OUT_BUF     (3)
    ) tlb_dev_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_cluster_dev_ptw_if),
        .bus_out_if (dev_ptw_bus_if)
    );

    VX_tlb_flush_if dev_ptw_flush_if ();
    VX_mmu_fault_if dev_ptw_fault_if ();
    wire dev_ptw_empty;

    // PTE fetches ride the LLC: the L3 client slot normally, or the single
    // cluster's L2 when that L2 is the LLC (PTW_ON_L2).
    localparam PTW_MEM_DATA_SIZE = (PTW_ON_L2 != 0) ? `VX_CFG_L1_LINE_SIZE : L2_SECTOR_SIZE;
    localparam PTW_MEM_TAG_WIDTH = (PTW_ON_L2 != 0) ? L2_TAG_WIDTH : L3_TAG_WIDTH;

    VX_mem_bus_if #(
        .DATA_SIZE (PTW_MEM_DATA_SIZE),
        .TAG_WIDTH (PTW_MEM_TAG_WIDTH)
    ) dev_ptw_mem_if ();

    // One leg per cluster so every instance owns its port's drivers; only
    // cluster 0's leg carries traffic, and only under PTW_ON_L2.
    VX_mem_bus_if #(
        .DATA_SIZE (PTW_MEM_DATA_SIZE),
        .TAG_WIDTH (PTW_MEM_TAG_WIDTH)
    ) per_cluster_ptw_mem_if [`VX_CFG_NUM_CLUSTERS] ();

    VX_ptw #(
        .ID_WIDTH       (TLB_DEV_ID_WIDTH),
        .DATA_SIZE      (PTW_MEM_DATA_SIZE),
        .MEM_TAG_WIDTH  (PTW_MEM_TAG_WIDTH)
    ) dev_ptw (
        .clk        (clk),
        .reset      (reset),
        .satp       (mmu_satp),
        .miss_if    (dev_ptw_bus_if),
        .mem_bus_if (dev_ptw_mem_if),
        .flush_if   (dev_ptw_flush_if),
        .fault_if   (dev_ptw_fault_if),
        .empty      (dev_ptw_empty)
    );

    if (PTW_ON_L2 == 0) begin : g_ptw_llc_client
        `ASSIGN_VX_MEM_BUS_IF (per_cluster_mem_bus_if[L3_PTW_IDX], dev_ptw_mem_if);
        for (genvar c = 0; c < `VX_CFG_NUM_CLUSTERS; ++c) begin : g_ptw_mem_legs
            `INIT_VX_MEM_BUS_IF (per_cluster_ptw_mem_if[c])
        end
    end else begin : g_ptw_l2_client
        `ASSIGN_VX_MEM_BUS_IF (per_cluster_ptw_mem_if[0], dev_ptw_mem_if);
    end

    assign dev_ptw_flush_if.req = mmu_flush_req;

    // The walker is the only fault source and owns its own legs on the DCR
    // surface; the clusters report translation drains only.
    VX_mmu_dcr mmu_dcr (
        .clk                (clk),
        .reset              (reset),
        .dcr_bus_if         (dcr_bus_if),
        .dcr_bus_out_if     (dcr_cluster_src_if),
        .satp               (mmu_satp),
        .flush_req          (mmu_flush_req),
        .cluster_flush_done (cl_mmu_flush_done),
        .walker_flush_done  (dev_ptw_flush_if.done),
        .fault_valid        (dev_ptw_fault_if.valid),
        .fault_va           (dev_ptw_fault_if.va),
        .fault_access       (dev_ptw_fault_if.access),
        .fault_amo          (dev_ptw_fault_if.amo)
    );

    wire dev_mmu_busy = ~dev_ptw_empty;
`else
    assign dcr_cluster_src_if.req_valid = dcr_bus_if.req_valid;
    assign dcr_cluster_src_if.req_data  = dcr_bus_if.req_data;
    assign dcr_bus_if.rsp_valid = dcr_cluster_src_if.rsp_valid;
    assign dcr_bus_if.rsp_data  = dcr_cluster_src_if.rsp_data;

    wire dev_mmu_busy = 1'b0;
`endif

    VX_dcr_bus_if per_cluster_dcr_bus_if[`VX_CFG_NUM_CLUSTERS]();
    VX_dcr_arb #(
        .NUM_REQS    (`VX_CFG_NUM_CLUSTERS),
        .REQ_OUT_BUF ((`VX_CFG_NUM_CLUSTERS > 1) ? 1 : 0)
    ) dcr_cluster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (dcr_cluster_src_if),
        .bus_out_if (per_cluster_dcr_bus_if)
    );

    // Partitions of GPCs; GPC g of partition p is cluster p * GPCS_PER_PARTITION + g.
    for (genvar p = 0; p < NUM_PARTITIONS; ++p) begin : g_partitions

        localparam G0 = p * GPCS_PER_PARTITION;

        VX_partition #(
            .PARTITION_ID (p)
        ) partition (
            `SCOPE_IO_BIND (scope_partition + p)

            .clk                (clk),
            .reset              (reset),

        `ifdef PERF_ENABLE
            .sysmem_perf        (sysmem_perf),
        `endif

            .dcr_bus_if         (per_cluster_dcr_bus_if[G0 +: GPCS_PER_PARTITION]),

            .mem_bus_if         (per_cluster_mem_bus_if[G0 * L2_MEM_PORTS +: GPCS_PER_PARTITION * L2_MEM_PORTS]),

            .kmu_bus_if         (per_cluster_kmu_bus_if[G0 +: GPCS_PER_PARTITION]),

        `ifdef VX_CFG_EXT_RASTER_ENABLE
            .raster_launch_if   (per_cluster_raster_launch_if[G0 +: GPCS_PER_PARTITION]),
        `endif

        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            .gang_mode          (gpc_gang_mode),
            .alloc_valid        (gpc_alloc_valid[G0 +: GPCS_PER_PARTITION]),
            .alloc_gang         (gpc_alloc_gang),
            .alloc_epoch        (gpc_alloc_epoch),
            .alloc_k            (gpc_alloc_k),
            .done_valid         (gpc_done_valid[G0 +: GPCS_PER_PARTITION]),
            .done_data          (gpc_done_data[G0 +: GPCS_PER_PARTITION]),
            .rel_pend           (gpc_rel_pend[G0 +: GPCS_PER_PARTITION]),
        `endif

        `ifdef VX_CFG_VM_ENABLE
            .mmu_satp           (mmu_satp),
            .dev_ptw_if         (per_cluster_dev_ptw_if[G0 +: GPCS_PER_PARTITION]),
            .dev_ptw_mem_if     (per_cluster_ptw_mem_if[G0 +: GPCS_PER_PARTITION]),
            .mmu_flush_req      (mmu_flush_req),
            .mmu_flush_done     (cl_mmu_flush_done[G0 +: GPCS_PER_PARTITION]),
        `endif

            .busy               (per_cluster_busy[G0 +: GPCS_PER_PARTITION])
        );
    end
    // Launch liveness: a beat resident in a per-cluster output skid is folded into
    // busy combinationally. The device input
    // (kmu_bus_in) needs no separate term -- kmu_busy (combinational) covers the
    // presented cycle and the registered per_cluster_busy covers the cycles after.
    wire [`VX_CFG_NUM_CLUSTERS-1:0] per_cluster_kmu_valid;
    for (genvar c = 0; c < `VX_CFG_NUM_CLUSTERS; ++c) begin : g_kmu_link_valid
        assign per_cluster_kmu_valid[c] = per_cluster_kmu_bus_if[c].valid;
    end
    wire busy_r;
    `BUFFER_EX(busy_r, kmu_busy | dcr_bus_if.req_valid | (|per_cluster_busy) | admit_busy | dev_mmu_busy, 1'b1, 1, (`VX_CFG_NUM_CLUSTERS > 1));
    assign busy = busy_r | kmu_busy | dcr_bus_if.req_valid | (|per_cluster_kmu_valid);

`ifdef PERF_ENABLE

    localparam MEM_PORTS_CTR_W = `CLOG2(VX_MEM_PORTS+1);

    wire [VX_MEM_PORTS-1:0] mem_req_fire, mem_rsp_fire;
    wire [VX_MEM_PORTS-1:0] mem_rd_req_fire, mem_wr_req_fire;

    for (genvar i = 0; i < VX_MEM_PORTS; ++i) begin : g_perf_ctrs
        assign mem_req_fire[i] = mem_req_valid[i] & mem_req_ready[i];
        assign mem_rsp_fire[i] = mem_rsp_valid[i] & mem_rsp_ready[i];
        assign mem_rd_req_fire[i] = mem_req_fire[i] & ~mem_req_rw[i];
        assign mem_wr_req_fire[i] = mem_req_fire[i] & mem_req_rw[i];
    end

    wire [MEM_PORTS_CTR_W-1:0] perf_mem_reads_per_cycle;
    wire [MEM_PORTS_CTR_W-1:0] perf_mem_writes_per_cycle;
    wire [MEM_PORTS_CTR_W-1:0] perf_mem_rsps_per_cycle;

    `POP_COUNT(perf_mem_reads_per_cycle, mem_rd_req_fire);
    `POP_COUNT(perf_mem_writes_per_cycle, mem_wr_req_fire);
    `POP_COUNT(perf_mem_rsps_per_cycle, mem_rsp_fire);

    reg [PERF_CTR_BITS-1:0] perf_mem_pending_reads;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_pending_reads <= '0;
        end else begin
            perf_mem_pending_reads <= $signed(perf_mem_pending_reads) +
                PERF_CTR_BITS'($signed((MEM_PORTS_CTR_W+1)'(perf_mem_reads_per_cycle) - (MEM_PORTS_CTR_W+1)'(perf_mem_rsps_per_cycle)));
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            mem_perf <= '0;
        end else begin
            mem_perf.reads <= mem_perf.reads + PERF_CTR_BITS'(perf_mem_reads_per_cycle);
            mem_perf.writes <= mem_perf.writes + PERF_CTR_BITS'(perf_mem_writes_per_cycle);
            mem_perf.latency <= mem_perf.latency + perf_mem_pending_reads;
        end
    end

`endif

    // dump device configuration
    initial begin
        `TRACE(0, ("CONFIGS: num_threads=%0d, num_warps=%0d, num_cores=%0d, num_clusters=%0d, socket_size=%0d, local_mem_base=0x%0h, num_barriers=%0d\n",
                    `VX_CFG_NUM_THREADS, `VX_CFG_NUM_WARPS, `VX_CFG_NUM_CORES, `VX_CFG_NUM_CLUSTERS, `VX_CFG_SOCKET_SIZE, `VX_MEM_LMEM_BASE_ADDR, `VX_CFG_NUM_BARRIERS))
    end

`ifdef DBG_TRACE_MEM
    for (genvar i = 0; i < VX_MEM_PORTS; ++i) begin : g_trace
        always @(posedge clk) begin
            if (mem_bus_if[i].req_valid && mem_bus_if[i].req_ready) begin
                if (mem_bus_if[i].req_data.rw) begin
                    `TRACE(2, ("%t: MEM Wr Req[%0d]: addr=0x%0h, byteen=0x%h, data=0x%h, tag=0x%0h (#%0d)\n", $time, i, `TO_FULL_ADDR(mem_bus_if[i].req_data.addr), mem_bus_if[i].req_data.byteen, mem_bus_if[i].req_data.data, mem_bus_if[i].req_data.tag.value, mem_bus_if[i].req_data.tag.uuid))
                end else begin
                    `TRACE(2, ("%t: MEM Rd Req[%0d]: addr=0x%0h, byteen=0x%h, tag=0x%0h (#%0d)\n", $time, i, `TO_FULL_ADDR(mem_bus_if[i].req_data.addr), mem_bus_if[i].req_data.byteen, mem_bus_if[i].req_data.tag.value, mem_bus_if[i].req_data.tag.uuid))
                end
            end
            if (mem_bus_if[i].rsp_valid && mem_bus_if[i].rsp_ready) begin
                `TRACE(2, ("%t: MEM Rd Rsp[%0d]: data=0x%h, tag=0x%0h (#%0d)\n", $time, i, mem_bus_if[i].rsp_data.data, mem_bus_if[i].rsp_data.tag.value, mem_bus_if[i].rsp_data.tag.uuid))
            end
        end
    end
`endif

`ifdef SIMULATION
    always @(posedge clk) begin
        $fflush(); // flush stdout buffer
    end
`endif

endmodule
