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
`ifdef VX_CFG_H100_SM_ENABLE
`include "VX_cache_define.vh"
`endif

module VX_socket import VX_gpu_pkg::*, VX_tlb_pkg::*;
`ifdef VX_CFG_EXT_DXA_ENABLE
    import VX_dxa_pkg::*;
`endif
#(
    parameter SOCKET_ID = 0,
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    // Clock
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t     sysmem_perf,
`endif

    // DCRs
    VX_dcr_bus_if.slave     dcr_bus_if,

    // Memory
    VX_mem_bus_if.master    mem_bus_if [L1_MEM_PORTS],

`ifdef VX_CFG_VM_ENABLE
    // TLB miss/fill fabric up to the cluster walker (socket-arbitrated).
    VX_tlb_bus_if.master    cluster_tlb_bus_if,
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
`endif

`ifdef EXT_GFX_ANY_ENABLE
    VX_dcr_flush_if.master  cluster_flush_if,
`endif

    // KMU bus
    VX_kmu_bus_if.slave     kmu_bus_if[1],

    // Global barrier
    VX_gbar_bus_if.master   gbar_bus_if,

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // DLMEM networks: uplinks to the cluster crossbar
    VX_dlmem_bus_if.master  dlmem_req_up_if,
    VX_dlmem_bus_if.slave   dlmem_req_down_if,
    VX_dlmem_bus_if.master  dlmem_rsp_up_if,
    VX_dlmem_bus_if.slave   dlmem_rsp_down_if,
`endif

    // Status
    output wire             busy
);

`ifdef SCOPE
    localparam scope_core = 0;
    `SCOPE_IO_SWITCH (`VX_CFG_SOCKET_SIZE);
`endif

    VX_kmu_bus_if per_core_kmu_bus_if[`VX_CFG_SOCKET_SIZE]();

    VX_kmu_bus_arb #(
        .NUM_INPUTS (1),
        .NUM_OUTPUTS (`VX_CFG_SOCKET_SIZE),
        .DEST_LSB   (KMU_DEST_LSB_SOCKET),
        .OUT_BUF    ((`VX_CFG_SOCKET_SIZE > 1) ? 3 : 0)
    ) kmu_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (kmu_bus_if),
        .bus_out_if (per_core_kmu_bus_if)
    );

    VX_gbar_bus_if per_core_gbar_bus_if[`VX_CFG_SOCKET_SIZE]();

    VX_gbar_arb #(
        .NUM_REQS (`VX_CFG_SOCKET_SIZE),
        .REQ_OUT_BUF ((`VX_CFG_SOCKET_SIZE > 1) ? 3 : 0),
        .RSP_OUT_BUF ((`VX_CFG_SOCKET_SIZE > 1) ? 3 : 0)
    ) gbar_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_core_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );

    ///////////////////////////////////////////////////////////////////////////

`ifdef PERF_ENABLE
    cache_perf_t icache_perf, dcache_perf;
`ifdef VX_CFG_EXT_DXA_ENABLE
    dxa_perf_t dxa_core_perf;
`endif
`ifdef VX_CFG_EXT_TEX_ENABLE
    tex_perf_t   socket_tex_perf;
    cache_perf_t socket_tcache_perf;
`endif
    sysmem_perf_t sysmem_perf_tmp;
    always @(*) begin
        sysmem_perf_tmp = sysmem_perf;
        sysmem_perf_tmp.icache = icache_perf;
        sysmem_perf_tmp.dcache = dcache_perf;
    `ifdef VX_CFG_EXT_DXA_ENABLE
        sysmem_perf_tmp.dxa = dxa_core_perf;
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        sysmem_perf_tmp.tex    = socket_tex_perf;
        sysmem_perf_tmp.tcache = socket_tcache_perf;
    `endif
    end
`endif

    ///////////////////////////////////////////////////////////////////////////

    // Core-side (virtual under VM) and cache-side (physical) icache buses. The
    // socket iMMU translates between them; without VM they pass straight through.
    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) per_core_icache_bus_if[`VX_CFG_SOCKET_SIZE]();

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) icache_phys_bus_if[`VX_CFG_SOCKET_SIZE]();

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_LINE_SIZE),
        .TAG_WIDTH (ICACHE_MEM_TAG_WIDTH)
    ) icache_mem_bus_if[1]();

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("%s-icache", INSTANCE_ID))),
        .NUM_UNITS      (`VX_CFG_NUM_ICACHES),
        .NUM_INPUTS     (`VX_CFG_SOCKET_SIZE),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`VX_CFG_ICACHE_SIZE),
        .LINE_SIZE      (ICACHE_LINE_SIZE),
        .NUM_BANKS      (1),
        .NUM_WAYS       (`VX_CFG_ICACHE_NUM_WAYS),
        .WORD_SIZE      (ICACHE_WORD_SIZE),
        .NUM_REQS       (1),
        .MEM_PORTS      (1),
        .CRSQ_SIZE      (`VX_CFG_ICACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_ICACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_ICACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_ICACHE_MREQ_SIZE),
        .LATENCY        (`VX_CFG_ICACHE_LATENCY),
        .TAG_WIDTH      (ICACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .REPL_POLICY    (`VX_CFG_ICACHE_REPL_POLICY),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3)
    ) icache (
    `ifdef PERF_ENABLE
        .cache_perf     (icache_perf),
    `endif
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (icache_phys_bus_if),
        .mem_bus_if     (icache_mem_bus_if)
    );

    ///////////////////////////////////////////////////////////////////////////

    // Core-side (virtual under VM) and cache-side (physical) dcache buses. The
    // socket dMMU translates between them; without VM they pass straight through.
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) per_core_dcache_bus_if[`VX_CFG_SOCKET_SIZE * DCACHE_NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) dcache_phys_bus_if[`VX_CFG_SOCKET_SIZE * DCACHE_NUM_REQS]();

`ifdef VX_CFG_H100_SM_ENABLE
    // One L1 per SM, each with its own memory side: the SM's port (below).
    localparam SM_DCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_NC_MEM_TAG_WIDTH(`VX_CFG_DCACHE_MSHR_SIZE, DCACHE_NUM_BANKS,
        DCACHE_NUM_REQS, 1, DCACHE_MEM_DATA_SIZE, DCACHE_WORD_SIZE, DCACHE_TAG_WIDTH, 1, 1, UUID_WIDTH);
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_MEM_DATA_SIZE),
        .TAG_WIDTH (SM_DCACHE_MEM_TAG_WIDTH)
    ) sm_dcache_mem_bus_if[`VX_CFG_SOCKET_SIZE]();
`else
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_SECTOR_SIZE),
        .TAG_WIDTH (DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_bus_if[L1_MEM_PORTS]();
`endif

    // The dcache is the device's shared level, where a fence drains, only when
    // no other cache or memory port could hold or read data behind it: no L2
    // or L3, one socket with one dcache, and no unit with a memory port below
    // it. Otherwise a fence walks it.
    localparam DCACHE_SHARED_LEVEL = (`VX_CFG_DCACHE_ENABLED != 0)
                                  && (`VX_CFG_L2_ENABLED == 0) && (`VX_CFG_L3_ENABLED == 0)
                                  && (`VX_CFG_NUM_CLUSTERS == 1) && (NUM_SOCKETS == 1)
                                  && (`VX_CFG_NUM_DCACHES == 1)
                                  && (`VX_CFG_EXT_DXA_ENABLED == 0) && (`VX_CFG_EXT_TEX_ENABLED == 0)
                                  && (`VX_CFG_EXT_RTU_ENABLED == 0) && (`VX_CFG_EXT_RASTER_ENABLED == 0)
                                  && (`VX_CFG_EXT_OM_ENABLED == 0);

`ifdef VX_CFG_H100_SM_ENABLE
`ifdef PERF_ENABLE
    cache_perf_t sm_dcache_perf [`VX_CFG_SOCKET_SIZE];
    `PERF_CACHE_ADD (dcache_perf, sm_dcache_perf, `VX_CFG_SOCKET_SIZE)
`endif
    for (genvar c = 0; c < `VX_CFG_SOCKET_SIZE; ++c) begin : g_sm_dcache
        VX_mem_bus_if #(
            .DATA_SIZE (DCACHE_MEM_DATA_SIZE),
            .TAG_WIDTH (SM_DCACHE_MEM_TAG_WIDTH)
        ) mem_bus_tmp_if[1]();

        VX_cache_cluster #(
            .INSTANCE_ID    (`SFORMATF(("%s-dcache%0d", INSTANCE_ID, c))),
            .NUM_UNITS      (1),
            .NUM_INPUTS     (1),
            .TAG_SEL_IDX    (0),
            .CACHE_SIZE     (`VX_CFG_DCACHE_SIZE),
            .LINE_SIZE      (DCACHE_LINE_SIZE),
            .SECTOR_SIZE    (DCACHE_MEM_DATA_SIZE),
            .H100_L1        (1),
            .NUM_BANKS      (DCACHE_NUM_BANKS),
            .NUM_WAYS       (`VX_CFG_DCACHE_NUM_WAYS),
            .WORD_SIZE      (DCACHE_WORD_SIZE),
            .NUM_REQS       (DCACHE_NUM_REQS),
            .MEM_PORTS      (1),
            .CRSQ_SIZE      (`VX_CFG_DCACHE_CRSQ_SIZE),
            .MSHR_SIZE      (`VX_CFG_DCACHE_MSHR_SIZE),
            .MRSQ_SIZE      (`VX_CFG_DCACHE_MRSQ_SIZE),
            .MREQ_SIZE      (`VX_CFG_DCACHE_MREQ_SIZE),
            .LATENCY        (`VX_CFG_DCACHE_LATENCY),
            .TAG_WIDTH      (DCACHE_TAG_WIDTH),
            .WRITE_ENABLE   (1),
            .WRITEBACK      (`VX_CFG_DCACHE_WRITEBACK),
            .DIRTY_BYTES    (`VX_CFG_DCACHE_DIRTYBYTES),
            .REPL_POLICY    (`VX_CFG_DCACHE_REPL_POLICY),
            .NC_ENABLE      (1),
            .CORE_OUT_BUF   (3),
            .MEM_OUT_BUF    (3),
            .IS_LLC         (DCACHE_IS_LLC),
            .SHARED_LEVEL   (DCACHE_SHARED_LEVEL),
        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            .CTA_FENCE      (1),
        `endif
            .AMO_ENABLE     (`VX_CFG_EXT_A_ENABLED)
        ) dcache (
        `ifdef PERF_ENABLE
            .cache_perf     (sm_dcache_perf[c]),
        `endif
            .clk            (clk),
            .reset          (reset),
            .core_bus_if    (dcache_phys_bus_if[c * DCACHE_NUM_REQS +: DCACHE_NUM_REQS]),
            .mem_bus_if     (mem_bus_tmp_if)
        );

        `ASSIGN_VX_MEM_BUS_IF (sm_dcache_mem_bus_if[c], mem_bus_tmp_if[0]);
    end
`else
    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("%s-dcache", INSTANCE_ID))),
        .NUM_UNITS      (`VX_CFG_NUM_DCACHES),
        .NUM_INPUTS     (`VX_CFG_SOCKET_SIZE),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`VX_CFG_DCACHE_SIZE),
        .LINE_SIZE      (DCACHE_LINE_SIZE),
        .SECTOR_SIZE    (DCACHE_SECTOR_SIZE),
        .NUM_BANKS      (DCACHE_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_DCACHE_NUM_WAYS),
        .WORD_SIZE      (DCACHE_WORD_SIZE),
        .NUM_REQS       (DCACHE_NUM_REQS),
        .MEM_PORTS      (L1_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_DCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_DCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_DCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_DCACHE_MREQ_SIZE),
        .LATENCY        (`VX_CFG_DCACHE_LATENCY),
        .TAG_WIDTH      (DCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`VX_CFG_DCACHE_WRITEBACK),
        .DIRTY_BYTES    (`VX_CFG_DCACHE_DIRTYBYTES),
        .REPL_POLICY    (`VX_CFG_DCACHE_REPL_POLICY),
        .NC_ENABLE      (1),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .IS_LLC         (DCACHE_IS_LLC),
        .SHARED_LEVEL   (DCACHE_SHARED_LEVEL),
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        .CTA_FENCE      (1),
    `endif
        .AMO_ENABLE     (`VX_CFG_EXT_A_ENABLED)
    ) dcache (
    `ifdef PERF_ENABLE
        .cache_perf     (dcache_perf),
    `endif
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (dcache_phys_bus_if),
        .mem_bus_if     (dcache_mem_bus_if)
    );
`endif

    ///////////////////////////////////////////////////////////////////////////
    // Socket MMUs — one per L1 cache. Each translates the (virtual) addresses
    // its cores emit and feeds physical addresses to the cache. Misses ride the
    // socket->cluster tlb_bus to the shared walker. Without VM the core-side and
    // cache-side buses pass straight through.
    ///////////////////////////////////////////////////////////////////////////
`ifdef VX_CFG_VM_ENABLE
    // Socket MMU control tapped from the DCR chain: the module sits inline,
    // forwarding DCR to the cores via socket_dcr_bus_if, and derives the
    // translation-enable bit + self-times the L1 flush — so no satp/flush ports
    // cross the boundary.
    wire vm_active;
    VX_dcr_bus_if   socket_dcr_bus_if();
    VX_tlb_flush_if dmmu_flush_if();
    VX_tlb_flush_if immu_flush_if();
    VX_mmu_fault_if dmmu_fault_if();
    VX_mmu_fault_if immu_fault_if();

    VX_mmu_snoop mmu_snoop (
        .clk            (clk),
        .reset          (reset),
        .dcr_bus_in_if  (dcr_bus_if),
        .dcr_bus_out_if (socket_dcr_bus_if),
        .vm_active      (vm_active),
        .dmmu_flush_if  (dmmu_flush_if),
        .immu_flush_if  (immu_flush_if)
    );

    // dtlb + itlb each drive one tlb_bus into the socket arbiter below.
    VX_tlb_bus_if #(.ID_WIDTH (L1_TLB_ID_WIDTH))
        socket_tlb_bus_if [2] ();

    wire dmmu_empty, immu_empty;

`ifdef PERF_ENABLE
    mmu_perf_t dmmu_perf, immu_perf;
`endif

    VX_mmu #(
        .INSTANCE_ID ("dmmu"),
        .NUM_REQS    (`VX_CFG_SOCKET_SIZE * DCACHE_NUM_REQS),
        .TLB_SIZE    (`VX_CFG_DTLB_SIZE),
        .EXEC_SIDE   (0),
        .DATA_SIZE   (DCACHE_WORD_SIZE),
        .TAG_WIDTH   (DCACHE_TAG_WIDTH_BASE)
    ) dmmu (
        .clk         (clk),
        .reset       (reset),
        .vm_active   (vm_active),
    `ifdef PERF_ENABLE
        .mmu_perf    (dmmu_perf),
    `endif
        .core_bus_if (per_core_dcache_bus_if),
        .mem_bus_if  (dcache_phys_bus_if),
        .tlb_bus_if  (socket_tlb_bus_if[0]),
        .flush_if    (dmmu_flush_if),
        .fault_if    (dmmu_fault_if),
        .empty       (dmmu_empty)
    );

    VX_mmu #(
        .INSTANCE_ID ("immu"),
        .NUM_REQS    (`VX_CFG_SOCKET_SIZE),
        .TLB_SIZE    (`VX_CFG_ITLB_SIZE),
        .EXEC_SIDE   (1),
        .DATA_SIZE   (ICACHE_WORD_SIZE),
        .TAG_WIDTH   (ICACHE_TAG_WIDTH)
    ) immu (
        .clk         (clk),
        .reset       (reset),
        .vm_active   (vm_active),
    `ifdef PERF_ENABLE
        .mmu_perf    (immu_perf),
    `endif
        .core_bus_if (per_core_icache_bus_if),
        .mem_bus_if  (icache_phys_bus_if),
        .tlb_bus_if  (socket_tlb_bus_if[1]),
        .flush_if    (immu_flush_if),
        .fault_if    (immu_fault_if),
        .empty       (immu_empty)
    );

    // L1 MMU faults are not surfaced at the socket; the authoritative page-fault
    // report comes from the walker at the cluster level.
    `UNUSED_VAR (dmmu_fault_if.valid)
    `UNUSED_VAR (dmmu_fault_if.va)
    `UNUSED_VAR (dmmu_fault_if.access)
    `UNUSED_VAR (dmmu_fault_if.amo)
    `UNUSED_VAR (immu_fault_if.valid)
    `UNUSED_VAR (immu_fault_if.va)
    `UNUSED_VAR (immu_fault_if.access)
    `UNUSED_VAR (immu_fault_if.amo)

    // Socket MMU drain status, broadcast to every core so its busy/barrier logic
    // waits for in-flight translations. Shared MMU: a core waits for the whole
    // socket to drain (conservative, always correct).
    wire socket_mmu_drained = dmmu_empty && immu_empty;

`ifdef PERF_ENABLE
    // Surfacing these through sysmem_perf needs a dedicated mmu slot; until that
    // exists the socket MMU counters are computed but not consumed.
    `UNUSED_VAR (dmmu_perf)
    `UNUSED_VAR (immu_perf)
`endif

    // Merge the two MMU tlb buses onto the one cluster-facing bus.
    VX_tlb_bus_arb #(
        .NUM_INPUTS  (2),
        .ID_WIDTH_IN (L1_TLB_ID_WIDTH),
        .OUT_BUF     (3)
    ) tlb_socket_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (socket_tlb_bus_if),
        .bus_out_if (cluster_tlb_bus_if)
    );

`else
    // No-VM passthrough: cache-side buses mirror the core-side buses.
    for (genvar i = 0; i < `VX_CFG_SOCKET_SIZE * DCACHE_NUM_REQS; ++i) begin : g_dcache_phys_passthru
        `ASSIGN_VX_MEM_BUS_IF (dcache_phys_bus_if[i], per_core_dcache_bus_if[i]);
    end
    for (genvar i = 0; i < `VX_CFG_SOCKET_SIZE; ++i) begin : g_icache_phys_passthru
        `ASSIGN_VX_MEM_BUS_IF (icache_phys_bus_if[i], per_core_icache_bus_if[i]);
    end
`endif

    ///////////////////////////////////////////////////////////////////////////

    // DCR fan-out — cores first, then socket-resident unit consumers
    // (TEX units, DXA). Declared before its consumers below.
    localparam SOCKET_NUM_DCR_TEX  = `VX_CFG_EXT_TEX_ENABLED * `VX_CFG_NUM_TEX_CORES;
    localparam SOCKET_NUM_DCR_REQS = `VX_CFG_SOCKET_SIZE + SOCKET_NUM_DCR_TEX + `VX_CFG_EXT_DXA_ENABLED;
`ifdef VX_CFG_EXT_TEX_ENABLE
    localparam SOCKET_DCR_TEX_BASE = `VX_CFG_SOCKET_SIZE;
`endif
`ifdef VX_CFG_EXT_DXA_ENABLE
    localparam SOCKET_DCR_DXA_IDX  = `VX_CFG_SOCKET_SIZE + SOCKET_NUM_DCR_TEX;
`endif

    VX_dcr_bus_if per_core_dcr_bus_if[SOCKET_NUM_DCR_REQS]();
    VX_dcr_arb #(
        .NUM_REQS    (SOCKET_NUM_DCR_REQS),
        .REQ_OUT_BUF ((SOCKET_NUM_DCR_REQS > 1) ? 1 : 0),
        .RSP_OUT_BUF ((SOCKET_NUM_DCR_REQS > 1) ? 1 : 0)
    ) dcr_core_arb (
        .clk        (clk),
        .reset      (reset),
    `ifdef VX_CFG_VM_ENABLE
        .bus_in_if  (socket_dcr_bus_if),
    `else
        .bus_in_if  (dcr_bus_if),
    `endif
        .bus_out_if (per_core_dcr_bus_if)
    );

    ///////////////////////////////////////////////////////////////////////////
    // TEX — socket-resident texture units + private tcache
    ///////////////////////////////////////////////////////////////////////////

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (TEX_REQ_TAG_WIDTH)
    ) per_core_tex_bus_if[`VX_CFG_SOCKET_SIZE]();

    VX_tex_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (TEX_REQ_ARB1_TAG_WIDTH)
    ) tex_bus_if [`VX_CFG_NUM_TEX_CORES] ();

    VX_tex_bus_arb #(
        .NUM_INPUTS  (`VX_CFG_SOCKET_SIZE),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
        .NUM_OUTPUTS (`VX_CFG_NUM_TEX_CORES),
        .TAG_WIDTH   (TEX_REQ_TAG_WIDTH),
        .ARBITER     ("R"),
        .OUT_BUF_REQ ((`VX_CFG_SOCKET_SIZE != `VX_CFG_NUM_TEX_CORES) ? 3 : 0), // register request fan-in
        .OUT_BUF_RSP ((`VX_CFG_SOCKET_SIZE != `VX_CFG_NUM_TEX_CORES) ? 3 : 0)  // register response distribution (skid registers the core-facing rsp and its backward ready)
    ) tex_socket_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_core_tex_bus_if),
        .bus_out_if (tex_bus_if)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_WORD_SIZE),
        .TAG_WIDTH (TCACHE_TAG_WIDTH)
    ) tcache_bus_if [`VX_CFG_NUM_TEX_CORES * TCACHE_NUM_REQS] ();

`ifdef PERF_ENABLE
    VX_tex_perf_if per_core_tex_perf_if [`VX_CFG_NUM_TEX_CORES] ();
`endif

    for (genvar i = 0; i < `VX_CFG_NUM_TEX_CORES; ++i) begin : g_tex_core
        VX_tex_core #(
            .INSTANCE_ID (`SFORMATF(("%s-tex%0d", INSTANCE_ID, i))),
            .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
            .TAG_WIDTH   (TEX_REQ_ARB1_TAG_WIDTH)
        ) tex_core (
            .clk          (clk),
            .reset        (reset),
        `ifdef PERF_ENABLE
            .perf_tex_if  (per_core_tex_perf_if[i]),
        `endif
            .dcr_bus_if   (per_core_dcr_bus_if[SOCKET_DCR_TEX_BASE + i]),
            .tex_bus_if   (tex_bus_if[i]),
            .cache_bus_if (tcache_bus_if[i * TCACHE_NUM_REQS +: TCACHE_NUM_REQS])
        );
    end

`ifdef PERF_ENABLE
    // Sum per-unit TEX counters across the socket. Verilator forbids
    // dynamic indexing into an interface array, so first copy each interface
    // member into a packed wire array via a genvar, then sum.
    wire [`VX_CFG_NUM_TEX_CORES-1:0][PERF_CTR_BITS-1:0] tex_mr_w, tex_ml_w, tex_sc_w;
    for (genvar i = 0; i < `VX_CFG_NUM_TEX_CORES; ++i) begin : g_tex_perf_pack
        assign tex_mr_w[i] = per_core_tex_perf_if[i].mem_reads;
        assign tex_ml_w[i] = per_core_tex_perf_if[i].mem_latency;
        assign tex_sc_w[i] = per_core_tex_perf_if[i].stall_cycles;
    end
    tex_perf_t tex_perf_sum;
    always @(*) begin
        tex_perf_sum = '0;
        for (int i = 0; i < `VX_CFG_NUM_TEX_CORES; ++i) begin
            tex_perf_sum.mem_reads    = tex_perf_sum.mem_reads    + tex_mr_w[i];
            tex_perf_sum.mem_latency  = tex_perf_sum.mem_latency  + tex_ml_w[i];
            tex_perf_sum.stall_cycles = tex_perf_sum.stall_cycles + tex_sc_w[i];
        end
    end
    assign socket_tex_perf = tex_perf_sum;
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_LINE_SIZE),
        .TAG_WIDTH (TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_bus_tmp_if [TCACHE_MEM_PORTS] ();

    // Cache-side bus with the +1 flush-tag bit (port 0 carries it through
    // VX_dcr_flush; ports 1..N-1 zero-extend their tags into the same width).
    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_WORD_SIZE),
        .TAG_WIDTH (TCACHE_BUS_TAG_WIDTH)
    ) tcache_flushable_bus_if [`VX_CFG_NUM_TEX_CORES * TCACHE_NUM_REQS] ();

    VX_dcr_flush_if tcache_flush_if();

    VX_dcr_flush #(
        .WORD_SIZE   (TCACHE_WORD_SIZE),
        .TAG_WIDTH   (TCACHE_TAG_WIDTH),
        .REQ_OUT_BUF (3) // register cache-request master boundary; rsp registered by cache CORE_OUT_BUF
    ) tcache_dcr_flush (
        .clk          (clk),
        .reset        (reset),
        .dcr_flush_if (tcache_flush_if),
        .core_bus_if  (tcache_bus_if[0]),
        .cache_bus_if (tcache_flushable_bus_if[0])
    );

    for (genvar i = 1; i < `VX_CFG_NUM_TEX_CORES * TCACHE_NUM_REQS; ++i) begin : g_tcache_passthru
        `ASSIGN_VX_MEM_BUS_IF_EX (tcache_flushable_bus_if[i], tcache_bus_if[i],
                                  TCACHE_BUS_TAG_WIDTH, TCACHE_TAG_WIDTH, 0);
    end

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("%s-tcache", INSTANCE_ID))),
        .NUM_UNITS      (`VX_CFG_NUM_TCACHES),
        .NUM_INPUTS     (`VX_CFG_NUM_TEX_CORES),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`VX_CFG_TCACHE_SIZE),
        .LINE_SIZE      (TCACHE_LINE_SIZE),
        .NUM_BANKS      (`VX_CFG_TCACHE_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_TCACHE_NUM_WAYS),
        .WORD_SIZE      (TCACHE_WORD_SIZE),
        .NUM_REQS       (TCACHE_NUM_REQS),
        .MEM_PORTS      (TCACHE_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_TCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_TCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_TCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_TCACHE_MREQ_SIZE),
        .TAG_WIDTH      (TCACHE_BUS_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .WRITEBACK      (0),
        .DIRTY_BYTES    (0),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3)
    ) tcache (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .cache_perf     (socket_tcache_perf),
    `endif
        .core_bus_if    (tcache_flushable_bus_if),
        .mem_bus_if     (tcache_mem_bus_tmp_if)
    );
`endif // VX_CFG_EXT_TEX_ENABLE

    ///////////////////////////////////////////////////////////////////////////
    // RTU — socket-resident ray-traversal units + private rtcache
    ///////////////////////////////////////////////////////////////////////////

`ifdef VX_CFG_EXT_RTU_ENABLE
    // How many cores share one RTU. Its beats name their source core with this many
    // bits, which is what keeps two cores' identically numbered warps apart in the
    // RTU's ray staging (see VX_rtu_bus_if.req_data_t.src).
    localparam RTU_CORES_PER_RTU = `VX_CFG_SOCKET_SIZE / `VX_CFG_NUM_RTU_CORES;
    localparam RTU_SRC_WIDTH     = `UP(`ARB_SEL_BITS(RTU_CORES_PER_RTU, 1));

    VX_rtu_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (RTU_REQ_TAG_WIDTH),
        .SRC_WIDTH (1)   // a core does not know its own index; the arbiter fills it in
    ) per_core_rtu_bus_if[`VX_CFG_SOCKET_SIZE]();

    VX_rtu_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (RTU_REQ_ARB1_TAG_WIDTH),
        .SRC_WIDTH (RTU_SRC_WIDTH)
    ) rtu_bus_if [`VX_CFG_NUM_RTU_CORES] ();

    VX_rtu_bus_arb #(
        .NUM_INPUTS  (`VX_CFG_SOCKET_SIZE),
        .NUM_LANES   (`VX_CFG_NUM_SFU_LANES),
        .NUM_OUTPUTS (`VX_CFG_NUM_RTU_CORES),
        .TAG_WIDTH   (RTU_REQ_TAG_WIDTH),
        .ARBITER     ("R"),
        // The arm may be buffered: it does not mean "the RTU has taken your ray"
        // (the RTU stages a ray per {src, wid}), so registering it cannot let a
        // TRACE retire into somebody else's traversal. See VX_rtu_bus_slice.
        .OUT_BUF_ARM ((`VX_CFG_SOCKET_SIZE != `VX_CFG_NUM_RTU_CORES) ? 3 : 0),
        .OUT_BUF_REQ ((`VX_CFG_SOCKET_SIZE != `VX_CFG_NUM_RTU_CORES) ? 3 : 0),
        .OUT_BUF_WIN ((`VX_CFG_SOCKET_SIZE != `VX_CFG_NUM_RTU_CORES) ? 3 : 0)
    ) rtu_socket_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_core_rtu_bus_if),
        .bus_out_if (rtu_bus_if)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (RTCACHE_WORD_SIZE),
        .TAG_WIDTH (RTCACHE_TAG_WIDTH)
    ) rtcache_bus_if [`VX_CFG_NUM_RTU_CORES * RTCACHE_NUM_REQS] ();

    for (genvar i = 0; i < `VX_CFG_NUM_RTU_CORES; ++i) begin : g_rtu_core
        VX_rtu_core #(
            .INSTANCE_ID     (`SFORMATF(("%s-rtu%0d", INSTANCE_ID, i))),
            .NUM_LANES       (`VX_CFG_NUM_SFU_LANES),
            .NUM_SRCS        (RTU_CORES_PER_RTU),
            .TAG_WIDTH       (RTU_REQ_ARB1_TAG_WIDTH),
            .CACHE_DATA_SIZE (RTCACHE_WORD_SIZE),
            .CACHE_TAG_WIDTH (RTCACHE_TAG_WIDTH)
        ) rtu_core (
            .clk          (clk),
            .reset        (reset),
            .rtu_bus_if   (rtu_bus_if[i]),
            .cache_bus_if (rtcache_bus_if[i * RTCACHE_NUM_REQS])
        );
    end

    VX_mem_bus_if #(
        .DATA_SIZE (RTCACHE_LINE_SIZE),
        .TAG_WIDTH (RTCACHE_MEM_TAG_WIDTH)
    ) rtcache_mem_bus_tmp_if [RTCACHE_MEM_PORTS] ();

    VX_mem_bus_if #(
        .DATA_SIZE (RTCACHE_WORD_SIZE),
        .TAG_WIDTH (RTCACHE_BUS_TAG_WIDTH)
    ) rtcache_flushable_bus_if [`VX_CFG_NUM_RTU_CORES * RTCACHE_NUM_REQS] ();

    VX_dcr_flush_if rtcache_flush_if();

    VX_dcr_flush #(
        .WORD_SIZE   (RTCACHE_WORD_SIZE),
        .TAG_WIDTH   (RTCACHE_TAG_WIDTH),
        .REQ_OUT_BUF (3) // register cache-request master boundary; rsp registered by cache CORE_OUT_BUF
    ) rtcache_dcr_flush (
        .clk          (clk),
        .reset        (reset),
        .dcr_flush_if (rtcache_flush_if),
        .core_bus_if  (rtcache_bus_if[0]),
        .cache_bus_if (rtcache_flushable_bus_if[0])
    );

    for (genvar i = 1; i < `VX_CFG_NUM_RTU_CORES * RTCACHE_NUM_REQS; ++i) begin : g_rtcache_passthru
        `ASSIGN_VX_MEM_BUS_IF_EX (rtcache_flushable_bus_if[i], rtcache_bus_if[i],
                                  RTCACHE_BUS_TAG_WIDTH, RTCACHE_TAG_WIDTH, 0);
    end

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("%s-rtcache", INSTANCE_ID))),
        .NUM_UNITS      (`VX_CFG_NUM_RTCACHES),
        .NUM_INPUTS     (`VX_CFG_NUM_RTU_CORES),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`VX_CFG_RTCACHE_SIZE),
        .LINE_SIZE      (RTCACHE_LINE_SIZE),
        .NUM_BANKS      (`VX_CFG_RTCACHE_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_RTCACHE_NUM_WAYS),
        .WORD_SIZE      (RTCACHE_WORD_SIZE),
        .NUM_REQS       (RTCACHE_NUM_REQS),
        .MEM_PORTS      (RTCACHE_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_RTCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_RTCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_RTCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_RTCACHE_MREQ_SIZE),
        .TAG_WIDTH      (RTCACHE_BUS_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .WRITEBACK      (0),
        .DIRTY_BYTES    (0),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3)
    ) rtcache (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        `UNUSED_PIN     (cache_perf),
    `endif
        .core_bus_if    (rtcache_flushable_bus_if),
        .mem_bus_if     (rtcache_mem_bus_tmp_if)
    );
`endif // VX_CFG_EXT_RTU_ENABLE

    ///////////////////////////////////////////////////////////////////////////
    // DXA — socket-resident transfer engine
    ///////////////////////////////////////////////////////////////////////////

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if per_core_dxa_req_bus_if[`VX_CFG_SOCKET_SIZE]();

`ifdef VX_CFG_H100_SM_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE (DXA_GMEM_LINE_SIZE),
        .TAG_WIDTH (SOCKET_MEM_TAG_WIDTH)
    ) dxa_gmem_bus_if[1]();
`else
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (SOCKET_MEM_TAG_WIDTH)
    ) dxa_gmem_bus_if[1]();
`endif

    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) dxa_lmem_bus_if[1]();

    // Alias the DXA's DCR array element onto a scalar interface via signal
    // assigns. A constant array index in a modport binding is rejected by
    // sv2v; aliasing moves the index out of that context. Pure net joins.
    VX_dcr_bus_if dxa_dcr_bus_if();
    assign dxa_dcr_bus_if.req_valid                          = per_core_dcr_bus_if[SOCKET_DCR_DXA_IDX].req_valid;
    assign dxa_dcr_bus_if.req_data                           = per_core_dcr_bus_if[SOCKET_DCR_DXA_IDX].req_data;
    assign per_core_dcr_bus_if[SOCKET_DCR_DXA_IDX].rsp_valid = dxa_dcr_bus_if.rsp_valid;
    assign per_core_dcr_bus_if[SOCKET_DCR_DXA_IDX].rsp_data  = dxa_dcr_bus_if.rsp_data;

`ifdef DXA_MBAR_ENABLE
    // Each core's slot stride, for the engine's cluster multicasts.
    wire [`VX_CFG_SOCKET_SIZE-1:0][`VX_CFG_LMEM_LOG_SIZE:0] per_core_slot_stride;
    // The engine's rows into other cores' blocks, an input of the DLMEM request switch.
    VX_dlmem_bus_if #(
        .DATAW (VX_dlmem_pkg::DLMEM_REQ_DATAW),
        .DST_W (DLMEM_CORE_W)
    ) dxa_row_if();
    // Bulk-group completion events, per core.
    wire [`VX_CFG_SOCKET_SIZE-1:0] per_core_bg_compl_valid;
    dxa_bg_compl_t [`VX_CFG_SOCKET_SIZE-1:0] per_core_bg_compl;
`endif

    VX_dxa_core #(
        .INSTANCE_ID    (`SFORMATF(("%s-dxa", INSTANCE_ID))),
        .NUM_REQS       (`VX_CFG_SOCKET_SIZE),
        .GMEM_OUT_PORTS (1)
    ) dxa_core (
        .clk              (clk),
        .reset            (reset),
    `ifdef PERF_ENABLE
        .dxa_perf         (dxa_core_perf),
    `endif
        .dcr_bus_if       (dxa_dcr_bus_if),
        .req_bus_if       (per_core_dxa_req_bus_if),
        .smem_bus_if      (dxa_lmem_bus_if),
        .gmem_bus_if      (dxa_gmem_bus_if),
    `ifdef DXA_MBAR_ENABLE
        .row_out_if       (dxa_row_if),
        .slot_stride      (per_core_slot_stride),
        .bg_compl_valid   (per_core_bg_compl_valid),
        .bg_compl         (per_core_bg_compl),
    `endif
        `UNUSED_PIN (busy)
    );

    // Route DXA lmem requests to per-core buses using core_local_id from tag.
    // Tag value layout: {core_id[NC_BITS-1:0], engine_value[0]}
    // core_local_id = core_id[CORE_LOCAL_BITS-1:0]
    localparam DXA_LMEM_CORE_LOCAL_BITS = `CLOG2(`VX_CFG_SOCKET_SIZE);
    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) per_core_dxa_lmem_bus_if[`VX_CFG_SOCKET_SIZE]();

    wire [`UP(DXA_LMEM_CORE_LOCAL_BITS)-1:0] dxa_lmem_core_sel;
    if (`VX_CFG_SOCKET_SIZE > 1) begin : g_dxa_lmem_sel
        assign dxa_lmem_core_sel = dxa_lmem_bus_if[0].req_data.tag.value[1 +: DXA_LMEM_CORE_LOCAL_BITS];
    end else begin : g_dxa_lmem_sel
        assign dxa_lmem_core_sel = '0;
    end

    VX_mem_bus_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (`VX_CFG_SOCKET_SIZE),
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W),
        .REQ_OUT_BUF ((`VX_CFG_SOCKET_SIZE > 1) ? 3 : 0), // register the per-core DXA lmem request fan-out
        .RSP_OUT_BUF ((`VX_CFG_SOCKET_SIZE > 1) ? 3 : 0)  // register its response (skid registers rsp and its backward ready)
    ) dxa_lmem_core_switch (
        .clk        (clk),
        .reset      (reset),
        .bus_sel    (dxa_lmem_core_sel),
        .bus_in_if  (dxa_lmem_bus_if),
        .bus_out_if (per_core_dxa_lmem_bus_if)
    );
`endif // VX_CFG_EXT_DXA_ENABLE

    ///////////////////////////////////////////////////////////////////////////

`ifdef VX_CFG_H100_SM_ENABLE
    // One port per SM (VX_sm_port): its L1's line requests, its TMA engine's
    // and its SM-to-SM packets in one in-order FIFO, its L2 side split into
    // memory blocks. Instruction fetch and the texture caches are not L1TEX
    // traffic: they join SM 0's memory port behind its adapter.
    localparam SM_DXA_SRCS    = `VX_CFG_EXT_DXA_ENABLED;
    localparam SM_NUM_SRCS    = 1 + SM_DXA_SRCS;
    localparam SM_SRC_TAG_W   = `MAX(SM_DCACHE_MEM_TAG_WIDTH, SM_DXA_SRCS ? SOCKET_MEM_TAG_WIDTH : 0);
    localparam SM_SECT_W      = `CLOG2(SM_PORT_LINE_SIZE / `VX_CFG_SM_EGRESS_BEAT + 1);
    localparam SM_PORT_TAG_W  = SM_SRC_TAG_W + SM_SECT_W + `LOG2UP(SM_NUM_SRCS);
    localparam SM_PENDING     = 2 * `VX_CFG_SM_MAX_READ_LINES;
    localparam SM_BLOCKS      = SM_PORT_LINE_SIZE / `VX_CFG_L1_LINE_SIZE;
    localparam SM_MEM_TAG_W   = UUID_WIDTH + `CLOG2(SM_PENDING) + ((SM_BLOCKS > 1) ? `CLOG2(SM_BLOCKS) : 0);
    // The port's own stages on an L2 access: request FIFO and output register,
    // the adapter's request buffer and reply queue, input register and reply
    // buffer. The rest of the SM side beyond the core and L1 stages is a delay
    // split over the two directions.
    localparam SM_PORT_EG_STAGES   = 2;
    localparam SM_ADAPTER_STAGES   = 2;
    localparam SM_PORT_IG_STAGES   = 2;
    localparam SM_PORT_MISS_STAGES = SM_PORT_EG_STAGES + SM_ADAPTER_STAGES + SM_PORT_IG_STAGES;
    localparam SM_MISS_DELAY  = `VX_CFG_SM_L2_SIDE_LATENCY - L1_MISS_PIPE_OVERHEAD - SM_PORT_MISS_STAGES;
    `STATIC_ASSERT (SM_MISS_DELAY >= 0, ("sm port: VX_CFG_SM_L2_SIDE_LATENCY below the pipeline's own stages"))
    `STATIC_ASSERT (SM_MEM_TAG_W <= SOCKET_MEM_TAG_WIDTH, ("sm port: adapter tag wider than the socket memory tag"))
    `STATIC_ASSERT (`VX_CFG_XLEN == 64, ("the H100 SM path requires XLEN 64"))

`ifdef VX_CFG_EXT_DXA_ENABLE
    // The TMA engine's line requests go to the issuing SM's port.
    localparam SM_DXA_CORE_BITS = `CLOG2(`VX_CFG_SOCKET_SIZE);
    VX_mem_bus_if #(
        .DATA_SIZE (DXA_GMEM_LINE_SIZE),
        .TAG_WIDTH (SOCKET_MEM_TAG_WIDTH)
    ) sm_dxa_gmem_bus_if[`VX_CFG_SOCKET_SIZE]();

    wire [`UP(SM_DXA_CORE_BITS)-1:0] sm_dxa_sel;
    if (`VX_CFG_SOCKET_SIZE > 1) begin : g_sm_dxa_sel
        assign sm_dxa_sel = dxa_gmem_bus_if[0].req_data.tag.value[VX_dxa_pkg::DXA_GMEM_CORE_LSB +: SM_DXA_CORE_BITS];
    end else begin : g_sm_dxa_sel1
        assign sm_dxa_sel = '0;
    end

    VX_mem_bus_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (`VX_CFG_SOCKET_SIZE),
        .DATA_SIZE   (DXA_GMEM_LINE_SIZE),
        .TAG_WIDTH   (SOCKET_MEM_TAG_WIDTH),
        .REQ_OUT_BUF ((`VX_CFG_SOCKET_SIZE > 1) ? 3 : 0),
        .RSP_OUT_BUF ((`VX_CFG_SOCKET_SIZE > 1) ? 3 : 0)
    ) sm_dxa_switch (
        .clk        (clk),
        .reset      (reset),
        .bus_sel    (sm_dxa_sel),
        .bus_in_if  (dxa_gmem_bus_if),
        .bus_out_if (sm_dxa_gmem_bus_if)
    );
`endif

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // Each SM's SM-to-SM traffic between its core and the socket switches.
    VX_dlmem_bus_if #(
        .DATAW (VX_dlmem_pkg::DLMEM_REQ_DATAW),
        .DST_W (DLMEM_CORE_W)
    ) sm_dlmem_req_out_if[`VX_CFG_SOCKET_SIZE](), sm_dlmem_req_in_if[`VX_CFG_SOCKET_SIZE]();

    VX_dlmem_bus_if #(
        .DATAW (VX_dlmem_pkg::DLMEM_RSP_DATAW),
        .DST_W (DLMEM_CORE_W)
    ) sm_dlmem_rsp_out_if[`VX_CFG_SOCKET_SIZE](), sm_dlmem_rsp_in_if[`VX_CFG_SOCKET_SIZE]();
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (SM_MEM_TAG_W)
    ) sm_mem_bus_if[`VX_CFG_SOCKET_SIZE]();

    for (genvar i = 0; i < L1_MEM_PORTS; ++i) begin : g_mem_bus_if
        if (i == 0) begin : g_i0
            VX_mem_bus_if #(
                .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH (SOCKET_MEM_TAG_WIDTH)
            ) socket_arb_in_if[SOCKET_MEM_ARB_REQS]();

            VX_mem_bus_if #(
                .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH (SOCKET_MEM_ARB_TAG_WIDTH)
            ) socket_arb_out_if[1]();

            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[ICACHE_MEM_ARB_IDX], icache_mem_bus_if[0], SOCKET_MEM_TAG_WIDTH, ICACHE_MEM_TAG_WIDTH, UUID_WIDTH);
            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[DCACHE_MEM_ARB_IDX], sm_mem_bus_if[0], SOCKET_MEM_TAG_WIDTH, SM_MEM_TAG_W, UUID_WIDTH);
        `ifdef VX_CFG_EXT_TEX_ENABLE
            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[TCACHE_MEM_ARB_IDX], tcache_mem_bus_tmp_if[0], SOCKET_MEM_TAG_WIDTH, TCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
        `endif
        `ifdef VX_CFG_EXT_RTU_ENABLE
            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[RTCACHE_MEM_ARB_IDX], rtcache_mem_bus_tmp_if[0], SOCKET_MEM_TAG_WIDTH, RTCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
        `endif
        `ifdef VX_CFG_EXT_DXA_ENABLE
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].req_valid = 1'b0;
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].req_data  = '0;
            assign socket_arb_in_if[DXA_MEM_ARB_IDX].rsp_ready = 1'b1;
            `UNUSED_VAR (socket_arb_in_if[DXA_MEM_ARB_IDX].req_ready)
            `UNUSED_VAR (socket_arb_in_if[DXA_MEM_ARB_IDX].rsp_valid)
            `UNUSED_VAR (socket_arb_in_if[DXA_MEM_ARB_IDX].rsp_data)
        `endif

            VX_mem_bus_arb #(
                .NUM_INPUTS (SOCKET_MEM_ARB_REQS),
                .NUM_OUTPUTS(1),
                .DATA_SIZE  (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH  (SOCKET_MEM_TAG_WIDTH),
                .TAG_SEL_IDX(0),
                .ARBITER    ("P"),
                .REQ_OUT_BUF(3),
                .RSP_OUT_BUF(3)
            ) mem_arb (
                .clk        (clk),
                .reset      (reset),
                .bus_in_if  (socket_arb_in_if),
                .bus_out_if (socket_arb_out_if)
            );

            `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[0], socket_arb_out_if[0]);
        end else begin : g_i
            `ASSIGN_VX_MEM_BUS_IF_EX (mem_bus_if[i], sm_mem_bus_if[i], SOCKET_MEM_ARB_TAG_WIDTH, SM_MEM_TAG_W, UUID_WIDTH);
        end
    end
`else
    // L2-facing memory ports. Port 0 arbitrates icache and dcache with the
    // socket-resident units' memory ports (tcache, rtcache, DXA gmem) as
    // peers; priority order keeps icache first and DXA bulk traffic last so
    // it cannot starve core fetch/load traffic. Ports 1.. carry the extra
    // dcache banks straight through.
    for (genvar i = 0; i < L1_MEM_PORTS; ++i) begin : g_mem_bus_if
        if (i == 0) begin : g_i0
            VX_mem_bus_if #(
                .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH (SOCKET_MEM_TAG_WIDTH)
            ) socket_arb_in_if[SOCKET_MEM_ARB_REQS]();

            VX_mem_bus_if #(
                .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH (SOCKET_MEM_ARB_TAG_WIDTH)
            ) socket_arb_out_if[1]();

            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[ICACHE_MEM_ARB_IDX], icache_mem_bus_if[0], SOCKET_MEM_TAG_WIDTH, ICACHE_MEM_TAG_WIDTH, UUID_WIDTH);
            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[DCACHE_MEM_ARB_IDX], dcache_mem_bus_if[0], SOCKET_MEM_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
        `ifdef VX_CFG_EXT_TEX_ENABLE
            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[TCACHE_MEM_ARB_IDX], tcache_mem_bus_tmp_if[0], SOCKET_MEM_TAG_WIDTH, TCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
        `endif
        `ifdef VX_CFG_EXT_RTU_ENABLE
            `ASSIGN_VX_MEM_BUS_IF_EX (socket_arb_in_if[RTCACHE_MEM_ARB_IDX], rtcache_mem_bus_tmp_if[0], SOCKET_MEM_TAG_WIDTH, RTCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
        `endif
        `ifdef VX_CFG_EXT_DXA_ENABLE
            `ASSIGN_VX_MEM_BUS_IF (socket_arb_in_if[DXA_MEM_ARB_IDX], dxa_gmem_bus_if[0]);
        `endif

            VX_mem_bus_arb #(
                .NUM_INPUTS (SOCKET_MEM_ARB_REQS),
                .NUM_OUTPUTS(1),
                .DATA_SIZE  (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH  (SOCKET_MEM_TAG_WIDTH),
                .TAG_SEL_IDX(0),
                .ARBITER    ("P"), // icache first, DXA last
                .REQ_OUT_BUF(3),
                .RSP_OUT_BUF(3)
            ) mem_arb (
                .clk        (clk),
                .reset      (reset),
                .bus_in_if  (socket_arb_in_if),
                .bus_out_if (socket_arb_out_if)
            );

            `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[0], socket_arb_out_if[0]);
        end else begin : g_i
            VX_mem_bus_if #(
                .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
                .TAG_WIDTH (SOCKET_MEM_ARB_TAG_WIDTH)
            ) l1_mem_arb_bus_if();

            `ASSIGN_VX_MEM_BUS_IF_EX (l1_mem_arb_bus_if, dcache_mem_bus_if[i], SOCKET_MEM_ARB_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
            `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[i], l1_mem_arb_bus_if);
        end
    end

`endif

    ///////////////////////////////////////////////////////////////////////////

    wire [`VX_CFG_SOCKET_SIZE-1:0] per_core_busy;
`ifdef EXT_GFX_ANY_ENABLE
    // Core flush requests fan out to the socket-local gfx caches (tcache,
    // rtcache) and up to the cluster (rcache, ocache); each leg's done is
    // level-held (VX_dcr_flush) so the AND below is race-free.
    VX_dcr_flush_if per_core_cluster_flush_if [`VX_CFG_SOCKET_SIZE]();
    wire [`VX_CFG_SOCKET_SIZE-1:0] per_core_cluster_flush_req;
    wire socket_flush_req = (| per_core_cluster_flush_req);
    assign cluster_flush_if.req = socket_flush_req;

    wire tcache_flush_done_w;
    wire rtcache_flush_done_w;
`ifdef VX_CFG_EXT_TEX_ENABLE
    assign tcache_flush_if.req = socket_flush_req;
    assign tcache_flush_done_w = tcache_flush_if.done;
`else
    assign tcache_flush_done_w = 1'b1;
`endif
`ifdef VX_CFG_EXT_RTU_ENABLE
    assign rtcache_flush_if.req = socket_flush_req;
    assign rtcache_flush_done_w = rtcache_flush_if.done;
`else
    assign rtcache_flush_done_w = 1'b1;
`endif

    for (genvar c = 0; c < `VX_CFG_SOCKET_SIZE; ++c) begin : g_per_core_cluster_flush
        assign per_core_cluster_flush_req[c]      = per_core_cluster_flush_if[c].req;
        assign per_core_cluster_flush_if[c].done  = cluster_flush_if.done & tcache_flush_done_w & rtcache_flush_done_w;
    end
`endif

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // One switch per DLMEM network; requests and responses never share one.

    VX_dlmem_bus_if #(
        .DATAW (VX_dlmem_pkg::DLMEM_REQ_DATAW),
        .DST_W (DLMEM_CORE_W)
    ) per_core_dlmem_req_out_if[`VX_CFG_SOCKET_SIZE](), per_core_dlmem_req_in_if[`VX_CFG_SOCKET_SIZE]();

    VX_dlmem_bus_if #(
        .DATAW (VX_dlmem_pkg::DLMEM_RSP_DATAW),
        .DST_W (DLMEM_CORE_W)
    ) per_core_dlmem_rsp_out_if[`VX_CFG_SOCKET_SIZE](), per_core_dlmem_rsp_in_if[`VX_CFG_SOCKET_SIZE]();

    VX_dlmem_socket_switch #(
        .SOCKET_SIZE (`VX_CFG_SOCKET_SIZE),
        .NUM_SOCKETS (NUM_SOCKETS),
        .SOCKET_IDX  (SOCKET_ID % NUM_SOCKETS),
        .DATAW       (VX_dlmem_pkg::DLMEM_REQ_DATAW),
        .DST_W       (DLMEM_CORE_W),
    `ifdef DXA_MBAR_ENABLE
        .ROW_IN      (1),
    `endif
        .OUT_BUF     (3)
    ) dlmem_req_switch (
        .clk           (clk),
        .reset         (reset),
    `ifdef VX_CFG_H100_SM_ENABLE
        .core_in_if    (sm_dlmem_req_out_if),
        .core_out_if   (sm_dlmem_req_in_if),
    `else
        .core_in_if    (per_core_dlmem_req_out_if),
        .core_out_if   (per_core_dlmem_req_in_if),
    `endif
    `ifdef DXA_MBAR_ENABLE
        .row_in_if     (dxa_row_if),
    `endif
        .uplink_in_if  (dlmem_req_down_if),
        .uplink_out_if (dlmem_req_up_if)
    );

`ifdef DXA_MBAR_ENABLE
    // Responses have no row input.
    VX_dlmem_bus_if #(
        .DATAW (VX_dlmem_pkg::DLMEM_RSP_DATAW),
        .DST_W (DLMEM_CORE_W)
    ) dlmem_rsp_no_row_if();
    assign dlmem_rsp_no_row_if.valid = 1'b0;
    assign dlmem_rsp_no_row_if.data  = '0;
    assign dlmem_rsp_no_row_if.dst   = '0;
    `UNUSED_VAR (dlmem_rsp_no_row_if.ready)
`endif

    VX_dlmem_socket_switch #(
        .SOCKET_SIZE (`VX_CFG_SOCKET_SIZE),
        .NUM_SOCKETS (NUM_SOCKETS),
        .SOCKET_IDX  (SOCKET_ID % NUM_SOCKETS),
        .DATAW       (VX_dlmem_pkg::DLMEM_RSP_DATAW),
        .DST_W       (DLMEM_CORE_W),
        .OUT_BUF     (3)
    ) dlmem_rsp_switch (
        .clk           (clk),
        .reset         (reset),
    `ifdef VX_CFG_H100_SM_ENABLE
        .core_in_if    (sm_dlmem_rsp_out_if),
        .core_out_if   (sm_dlmem_rsp_in_if),
    `else
        .core_in_if    (per_core_dlmem_rsp_out_if),
        .core_out_if   (per_core_dlmem_rsp_in_if),
    `endif
    `ifdef DXA_MBAR_ENABLE
        .row_in_if     (dlmem_rsp_no_row_if),
    `endif
        .uplink_in_if  (dlmem_rsp_down_if),
        .uplink_out_if (dlmem_rsp_up_if)
    );
`endif

`ifdef VX_CFG_H100_SM_ENABLE
    wire [`VX_CFG_SOCKET_SIZE-1:0] sm_port_empty;

    for (genvar c = 0; c < `VX_CFG_SOCKET_SIZE; ++c) begin : g_sm_port
        VX_mem_bus_if #(
            .DATA_SIZE (SM_PORT_LINE_SIZE),
            .TAG_WIDTH (SM_SRC_TAG_W)
        ) src_bus_if[SM_NUM_SRCS]();

        VX_mem_bus_if #(
            .DATA_SIZE (SM_PORT_LINE_SIZE),
            .TAG_WIDTH (SM_PORT_TAG_W)
        ) line_bus_if();

        `ASSIGN_VX_MEM_BUS_IF_EX (src_bus_if[0], sm_dcache_mem_bus_if[c], SM_SRC_TAG_W, SM_DCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
    `ifdef VX_CFG_EXT_DXA_ENABLE
        `ASSIGN_VX_MEM_BUS_IF_EX (src_bus_if[1], sm_dxa_gmem_bus_if[c], SM_SRC_TAG_W, SOCKET_MEM_TAG_WIDTH, UUID_WIDTH);
    `endif

        wire port_empty, adapter_empty;

        VX_sm_port #(
            .INSTANCE_ID   (`SFORMATF(("%s-smport%0d", INSTANCE_ID, c))),
            .NUM_SRCS      (SM_NUM_SRCS),
            .SRC_TAG_WIDTH (SM_SRC_TAG_W),
            .LINE_SIZE     (SM_PORT_LINE_SIZE),
            .EGRESS_DELAY  ((SM_MISS_DELAY + 1) / 2),
            .INGRESS_DELAY (SM_MISS_DELAY / 2)
        ) sm_port (
            .clk            (clk),
            .reset          (reset),
            .src_bus_if     (src_bus_if),
            .mem_bus_if     (line_bus_if),
        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            .dsm_req_in_if  (per_core_dlmem_req_out_if[c]),
            .dsm_req_out_if (sm_dlmem_req_out_if[c]),
            .dsm_rsp_in_if  (per_core_dlmem_rsp_out_if[c]),
            .dsm_rsp_out_if (sm_dlmem_rsp_out_if[c]),
            .arr_req_in_if  (sm_dlmem_req_in_if[c]),
            .arr_req_out_if (per_core_dlmem_req_in_if[c]),
            .arr_rsp_in_if  (sm_dlmem_rsp_in_if[c]),
            .arr_rsp_out_if (per_core_dlmem_rsp_in_if[c]),
        `endif
            .empty          (port_empty)
        );

        VX_sm_mem_adapter #(
            .LINE_SIZE     (SM_PORT_LINE_SIZE),
            .BLOCK_SIZE    (`VX_CFG_L1_LINE_SIZE),
            .TAG_IN_WIDTH  (SM_PORT_TAG_W),
            .TAG_OUT_WIDTH (SM_MEM_TAG_W),
            .PENDING       (SM_PENDING)
        ) adapter (
            .clk         (clk),
            .reset       (reset),
            .line_bus_if (line_bus_if),
            .mem_bus_if  (sm_mem_bus_if[c]),
            .empty       (adapter_empty)
        );

        assign sm_port_empty[c] = port_empty && adapter_empty;
    end

`endif

    // Generate all cores
    for (genvar core_id = 0; core_id < `VX_CFG_SOCKET_SIZE; ++core_id) begin : g_cores
        /*
        wire core_clk;
        wire core_clk_en = reset
                        || per_core_kmu_bus_if[core_id].valid
                        || per_core_dcr_bus_if[core_id].req_valid
                        || per_core_busy[core_id];
        VX_clockgate core_icg (
            .clk_in (clk),
            .en     (core_clk_en),
            .clk_out(core_clk)
        );*/

        VX_core #(
            .CORE_ID  ((SOCKET_ID * `VX_CFG_SOCKET_SIZE) + core_id),
            .INSTANCE_ID (`SFORMATF(("%s-core%0d", INSTANCE_ID, core_id)))
        ) core (
            `SCOPE_IO_BIND  (scope_core + core_id)

            .clk            (clk),
            .reset          (reset),

        `ifdef PERF_ENABLE
            .sysmem_perf    (sysmem_perf_tmp),
        `endif

            .dcr_bus_if     (per_core_dcr_bus_if[core_id]),

            .dcache_bus_if  (per_core_dcache_bus_if[core_id * DCACHE_NUM_REQS +: DCACHE_NUM_REQS]),

            .icache_bus_if  (per_core_icache_bus_if[core_id]),

        `ifdef VX_CFG_EXT_DXA_ENABLE
            .dxa_req_bus_if (per_core_dxa_req_bus_if[core_id]),
            .dxa_lmem_bus_if(per_core_dxa_lmem_bus_if[core_id]),
        `endif
        `ifdef DXA_MBAR_ENABLE
            .dxa_slot_stride(per_core_slot_stride[core_id]),
            .dxa_bg_compl_valid (per_core_bg_compl_valid[core_id]),
            .dxa_bg_compl   (per_core_bg_compl[core_id]),
        `endif

        `ifdef VX_CFG_EXT_TEX_ENABLE
            .tex_bus_if     (per_core_tex_bus_if[core_id]),
        `endif

        `ifdef VX_CFG_EXT_OM_ENABLE
        `endif

        `ifdef VX_CFG_EXT_RASTER_ENABLE
        `endif

        `ifdef VX_CFG_EXT_RTU_ENABLE
            .rtu_bus_if     (per_core_rtu_bus_if[core_id]),
        `endif

        `ifdef VX_CFG_VM_ENABLE
            .mmu_drained    (socket_mmu_drained),
        `endif

        `ifdef EXT_GFX_ANY_ENABLE
            .cluster_flush_if (per_core_cluster_flush_if[core_id]),
        `endif

            .kmu_bus_if     (per_core_kmu_bus_if[core_id]),

            .gbar_bus_if    (per_core_gbar_bus_if[core_id]),

        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            .dlmem_req_out_if (per_core_dlmem_req_out_if[core_id]),
            .dlmem_req_in_if  (per_core_dlmem_req_in_if[core_id]),
            .dlmem_rsp_out_if (per_core_dlmem_rsp_out_if[core_id]),
            .dlmem_rsp_in_if  (per_core_dlmem_rsp_in_if[core_id]),
        `endif

            .busy           (per_core_busy[core_id])
        );
    end

    // Launch liveness: fold this level's launch-link `valid` at both ends into busy
    // combinationally -- the beat presented at the socket input and any beat resident
    // in a per-core output skid -- so an in-transit launch stays visible on the
    // presented cycle. The child (per_core_busy) aggregation stays registered.
    wire [`VX_CFG_SOCKET_SIZE-1:0] per_core_kmu_valid;
    for (genvar i = 0; i < `VX_CFG_SOCKET_SIZE; ++i) begin : g_kmu_link_valid
        assign per_core_kmu_valid[i] = per_core_kmu_bus_if[i].valid;
    end
    wire busy_r;
    `BUFFER_EX(busy_r, dcr_bus_if.req_valid | (|per_core_busy), 1'b1, 1, (`VX_CFG_SOCKET_SIZE > 1));
`ifdef VX_CFG_H100_SM_ENABLE
    assign busy = busy_r | dcr_bus_if.req_valid | kmu_bus_if[0].valid | (|per_core_kmu_valid) | ~(&sm_port_empty);
`else
    assign busy = busy_r | dcr_bus_if.req_valid | kmu_bus_if[0].valid | (|per_core_kmu_valid);
`endif

endmodule
