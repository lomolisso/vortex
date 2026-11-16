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

module VX_mem_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output lmem_perf_t      lmem_perf,
    output coalescer_perf_t coalescer_perf,
`endif

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_mem_bus_if.slave     dxa_lmem_bus_if,
    VX_txbar_bus_if.master  dxa_txbar_bus_if,
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // TCU LMEM read port
    VX_mem_bus_if.slave     tcu_lmem_if,
`endif

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // Remote lanes toward the DLMEM egress, and the ingress's local-memory port
    VX_lsu_mem_if.master    dlmem_remote_if [`VX_CFG_NUM_LSU_BLOCKS],
    VX_mem_bus_if.slave     dlmem_lmem_if,
    // Per local-memory bank, an LSU access performed in the previous cycle, and its warp
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0] lmem_commit_valid,
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0][NW_WIDTH-1:0] lmem_commit_wid,
`ifdef VX_CFG_H100_SM_ENABLE
    // Per LSU block, counted lanes the shared-memory port served with another lane's access
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] lmem_fold_valid,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][NW_WIDTH-1:0] lmem_fold_wid,
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][`CLOG2(`VX_CFG_NUM_LSU_LANES+1)-1:0] lmem_fold_cnt,
`endif
`endif

`ifdef VX_CFG_EXT_MBAR_ENABLE
    // Per LSU block, the retire of an mbarrier try_wait; the warps it releases
    input wire [`VX_CFG_NUM_LSU_BLOCKS-1:0]               mbar_retire_valid,
    input wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][NW_WIDTH-1:0] mbar_retire_wid,
    output wire                                          mbar_unlock_valid,
    output wire [`VX_CFG_NUM_WARPS-1:0]                  mbar_unlock_mask,
    // Per LSU block, the cp.async unit's writes and arrives
    VX_lsu_mem_if.slave                                  cpa_lmem_if [`VX_CFG_NUM_LSU_BLOCKS],
`endif

`ifdef DXA_MBAR_ENABLE
    // The core's CTA slots, for the DXA's retired-slot and bounds checks
    input wire [`VX_CFG_NUM_WARPS-1:0]                    cta_slot_valid,
    input wire [`VX_CFG_NUM_WARPS-1:0][DXA_CTA_GEN_W-1:0] cta_slot_gen,
    input wire [`VX_CFG_LMEM_LOG_SIZE:0]                  cta_slot_stride,
    // Data words of remote DXA copies, from the DLMEM ingress
    VX_mem_bus_if.slave                                   dlmem_dma_if,
`endif

    output wire             empty,

    VX_lsu_mem_if.slave     lsu_mem_if [`VX_CFG_NUM_LSU_BLOCKS],
    VX_dcr_flush_if.slave   dcr_flush_if,
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS]
);
    // Per-block empty: high when the global-store path holds no in-flight
    // write. Folded into core `busy` so the end-of-kernel cache flush is
    // ordered strictly behind the last store. Two terms: the coalescer/passthru
    // producer, and the dcache adapter's request buffer (which carries state
    // when REQ_OUT_BUF > 0 and would otherwise hide a still-settling store).
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] per_block_path_empty;
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] per_block_adapter_empty;
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] per_block_empty;
    assign per_block_empty = per_block_path_empty & per_block_adapter_empty;
    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_dcache_if[`VX_CFG_NUM_LSU_BLOCKS]();

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    `STATIC_ASSERT(`VX_CFG_LMEM_ENABLED, ("TCU_WGMMA_ENABLE requires LMEM_ENABLE"))
`ifdef VX_CFG_TCU_H100_ENABLE
    // H100 descriptors carry 14-bit fields of 16-byte units: 256 KB.
    `STATIC_ASSERT(`VX_CFG_LMEM_LOG_SIZE <= 18, ("LMEM exceeds the 256 KB reach of the wgmma descriptor"))
`else
    // The WGMMA shared-memory descriptor carries the LMEM byte offset in a 16-bit
    // field, so a WGMMA-visible LMEM cannot exceed 64KB; beyond that the offset
    // wraps into the adjacent stride field.
    `STATIC_ASSERT(`VX_CFG_LMEM_LOG_SIZE <= 16, ("LMEM exceeds the 16-bit smem descriptor offset field"))
`endif
`endif

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    `STATIC_ASSERT(`VX_CFG_LMEM_ENABLED, ("DLMEM_ENABLE requires LMEM_ENABLE"))
`endif

`ifdef VX_CFG_LMEM_ENABLE

    `STATIC_ASSERT(`IS_DIVISBLE((1 << `VX_CFG_LMEM_LOG_SIZE), `VX_CFG_MEM_BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT(0 == (`VX_MEM_LMEM_BASE_ADDR % (1 << `VX_CFG_LMEM_LOG_SIZE)), ("invalid parameter"))

    localparam LMEM_ADDR_WIDTH = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE);

    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_lmem_if[`VX_CFG_NUM_LSU_BLOCKS]();

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_lmem_switches
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        // Round-robin response merge, so no leg waits behind the other two.
        VX_lmem_switch #(
            .GLOBAL_OUT_BUF(1),
            .LOCAL_OUT_BUF(1),
            .REMOTE_OUT_BUF(1),
            .RSP_OUT_BUF  (1),
            .ARBITER      ("R")
        ) lmem_switch (
            .clk          (clk),
            .reset        (reset),
            .lsu_in_if    (lsu_mem_if[i]),
            .global_out_if(lsu_dcache_if[i]),
            .remote_out_if(dlmem_remote_if[i]),
            .local_out_if (lsu_lmem_if[i])
        );
    `else
        VX_lmem_switch #(
            .GLOBAL_OUT_BUF(1),
            .LOCAL_OUT_BUF(1),
            .RSP_OUT_BUF  (1),
            .ARBITER      ("P")
        ) lmem_switch (
            .clk          (clk),
            .reset        (reset),
            .lsu_in_if    (lsu_mem_if[i]),
            .global_out_if(lsu_dcache_if[i]),
            .local_out_if (lsu_lmem_if[i])
        );
    `endif
    end

`ifdef VX_CFG_H100_SM_ENABLE
    `STATIC_ASSERT((64'(1) << `VX_CFG_LMEM_LOG_SIZE) <= 64'(`VX_CFG_LMEM_DESC_REACH_BYTES), ("LMEM exceeds the wgmma descriptor reach"))

    // The SM's shared-memory port: one wavefront a cycle over the LSU blocks.
    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_lmem_g_if[`VX_CFG_NUM_LSU_BLOCKS]();

    wire lmem_ext_valid;

    VX_smem_gate #(
        .INSTANCE_ID  (`SFORMATF(("%s-smem-gate", INSTANCE_ID))),
        .NUM_BLOCKS   (`VX_CFG_NUM_LSU_BLOCKS),
        .NUM_LANES    (`VX_CFG_NUM_LSU_LANES),
        .WORD_SIZE    (LSU_WORD_SIZE),
        .TAG_WIDTH    (LSU_TAG_WIDTH),
        .TAG_SEL_BITS (LSU_TAG_WIDTH - UUID_WIDTH),
        .ADDR_WIDTH   (LSU_ADDR_WIDTH)
    ) smem_gate (
        .clk       (clk),
        .reset     (reset),
        .ext_valid (lmem_ext_valid),
        .in_if     (lsu_lmem_if),
        .out_if    (lsu_lmem_g_if)
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        , .fold_valid (lmem_fold_valid)
        , .fold_wid   (lmem_fold_wid)
        , .fold_cnt   (lmem_fold_cnt)
    `endif
    );

`endif
    // Per-block local memory adapters to avoid deadlock when NUM_LSU_BLOCKS > 1.
    // Each block gets dedicated ports into local memory, eliminating the circular
    // dependency that arises from sharing a single adapter's unpack/pack buffers.

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // The DLMEM ingress is one more input after the LSU lanes, in their round-robin.
    localparam LMEM_INGRESS_IDX = LSU_NUM_REQS;

`ifdef DXA_MBAR_ENABLE
    // The DXA completion's complete_tx requests are one more input, after the ingress.
    localparam LMEM_DXA_COMPL_IDX = LSU_NUM_REQS + 1;
    // Then the cp.async units' lanes, block-major.
    localparam LMEM_CPA_IDX  = LSU_NUM_REQS + 2;
    localparam LMEM_NUM_REQS = LMEM_CPA_IDX + LSU_NUM_REQS;

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_adapt_if[LMEM_NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) dxa_compl_if();

    `ASSIGN_VX_MEM_BUS_IF (lmem_adapt_if[LMEM_DXA_COMPL_IDX], dxa_compl_if);
`elsif VX_CFG_EXT_MBAR_ENABLE
    // The cp.async units' lanes, block-major, after the ingress.
    localparam LMEM_CPA_IDX  = LSU_NUM_REQS + 1;
    localparam LMEM_NUM_REQS = LMEM_CPA_IDX + LSU_NUM_REQS;

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_adapt_if[LMEM_NUM_REQS]();
`else
    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_adapt_if[LSU_NUM_REQS + 1]();
`endif

`ifdef VX_CFG_H100_SM_ENABLE
    // The ingress presents one SM-to-SM slot; the word its bytes select goes
    // to the banks, and the read word comes back at every word of the slot.
    localparam ING_WORDS    = LMEM_INGRESS_DATA_SIZE / LSU_WORD_SIZE;
    localparam ING_OFF_BITS = `CLOG2(ING_WORDS);
    `STATIC_ASSERT(LMEM_INGRESS_DATA_SIZE % LSU_WORD_SIZE == 0, ("the ingress slot must hold whole LSU words"))
    `STATIC_ASSERT(($bits(dlmem_lmem_if.req_data.addr) + ING_OFF_BITS) == LMEM_ADDR_WIDTH, ("invalid parameter"))
    `STATIC_ASSERT($bits(dlmem_lmem_if.req_data.tag) == LSU_TAG_WIDTH, ("invalid parameter"))

    wire [ING_WORDS-1:0] ing_word_en;
    for (genvar k = 0; k < ING_WORDS; ++k) begin : g_ing_word_en
        assign ing_word_en[k] = |dlmem_lmem_if.req_data.byteen[k * LSU_WORD_SIZE +: LSU_WORD_SIZE];
    end
    wire [`UP(ING_OFF_BITS)-1:0] ing_off;
    VX_priority_encoder #(
        .N (ING_WORDS)
    ) ing_word_enc (
        .data_in   (ing_word_en),
        .index_out (ing_off),
        `UNUSED_PIN (onehot_out),
        `UNUSED_PIN (valid_out)
    );
    `RUNTIME_ASSERT(~dlmem_lmem_if.req_valid || ($countones(ing_word_en) <= 1),
        ("%t: *** %s: an ingress slot request with more than one word", $time, INSTANCE_ID))

    wire [LMEM_ADDR_WIDTH-1:0] ing_word_addr;
    if (ING_OFF_BITS != 0) begin : g_ing_word_addr
        assign ing_word_addr = {dlmem_lmem_if.req_data.addr, ing_off[ING_OFF_BITS-1:0]};
    end else begin : g_ing_word_addr1
        assign ing_word_addr = dlmem_lmem_if.req_data.addr;
    end

    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_valid       = dlmem_lmem_if.req_valid;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.rw     = dlmem_lmem_if.req_data.rw;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.addr   = LSU_ADDR_WIDTH'(ing_word_addr);
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.data   = dlmem_lmem_if.req_data.data[ing_off * LSU_WORD_SIZE * 8 +: LSU_WORD_SIZE * 8];
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.byteen = dlmem_lmem_if.req_data.byteen[ing_off * LSU_WORD_SIZE +: LSU_WORD_SIZE];
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.attr   = dlmem_lmem_if.req_data.attr;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.tag    = dlmem_lmem_if.req_data.tag;
    assign dlmem_lmem_if.req_ready = lmem_adapt_if[LMEM_INGRESS_IDX].req_ready;

    assign dlmem_lmem_if.rsp_valid     = lmem_adapt_if[LMEM_INGRESS_IDX].rsp_valid;
    assign dlmem_lmem_if.rsp_data.data = {ING_WORDS{lmem_adapt_if[LMEM_INGRESS_IDX].rsp_data.data}};
    assign dlmem_lmem_if.rsp_data.tag  = lmem_adapt_if[LMEM_INGRESS_IDX].rsp_data.tag;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].rsp_ready = dlmem_lmem_if.rsp_ready;
`else
    `STATIC_ASSERT($bits(dlmem_lmem_if.req_data.addr) == LMEM_ADDR_WIDTH, ("invalid parameter"))
    `STATIC_ASSERT($bits(dlmem_lmem_if.req_data.tag) == LSU_TAG_WIDTH, ("invalid parameter"))

    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_valid       = dlmem_lmem_if.req_valid;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.rw     = dlmem_lmem_if.req_data.rw;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.addr   = LSU_ADDR_WIDTH'(dlmem_lmem_if.req_data.addr);
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.data   = dlmem_lmem_if.req_data.data;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.byteen = dlmem_lmem_if.req_data.byteen;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.attr   = dlmem_lmem_if.req_data.attr;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].req_data.tag    = dlmem_lmem_if.req_data.tag;
    assign dlmem_lmem_if.req_ready = lmem_adapt_if[LMEM_INGRESS_IDX].req_ready;

    assign dlmem_lmem_if.rsp_valid = lmem_adapt_if[LMEM_INGRESS_IDX].rsp_valid;
    assign dlmem_lmem_if.rsp_data  = lmem_adapt_if[LMEM_INGRESS_IDX].rsp_data;
    assign lmem_adapt_if[LMEM_INGRESS_IDX].rsp_ready = dlmem_lmem_if.rsp_ready;
`endif
`else
    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_adapt_if[LSU_NUM_REQS]();
`endif

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_lmem_adapters

        VX_mem_bus_if #(
            .DATA_SIZE (LSU_WORD_SIZE),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) lmem_block_if[`VX_CFG_NUM_LSU_LANES]();

        VX_lsu_adapter #(
            .NUM_LANES    (`VX_CFG_NUM_LSU_LANES),
            .DATA_SIZE    (LSU_WORD_SIZE),
            .TAG_WIDTH    (LSU_TAG_WIDTH),
            .TAG_SEL_BITS (LSU_TAG_WIDTH - UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (3),
            .RSP_OUT_BUF  (0)
        ) lmem_adapter (
            .clk        (clk),
            .reset      (reset),
        `ifdef VX_CFG_H100_SM_ENABLE
            .lsu_mem_if (lsu_lmem_g_if[i]),
        `else
            .lsu_mem_if (lsu_lmem_if[i]),
        `endif
            .mem_bus_if (lmem_block_if)
        );

        for (genvar j = 0; j < `VX_CFG_NUM_LSU_LANES; ++j) begin : g_lmem_adapt_if
            `ASSIGN_VX_MEM_BUS_IF (lmem_adapt_if[i * `VX_CFG_NUM_LSU_LANES + j], lmem_block_if[j]);
        end
    end

`ifdef VX_CFG_EXT_MBAR_ENABLE
    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_cpa_adapters

        VX_mem_bus_if #(
            .DATA_SIZE (LSU_WORD_SIZE),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) cpa_block_if[`VX_CFG_NUM_LSU_LANES]();

        VX_lsu_adapter #(
            .NUM_LANES    (`VX_CFG_NUM_LSU_LANES),
            .DATA_SIZE    (LSU_WORD_SIZE),
            .TAG_WIDTH    (LSU_TAG_WIDTH),
            .TAG_SEL_BITS (LSU_TAG_WIDTH - UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (3),
            .RSP_OUT_BUF  (0)
        ) cpa_adapter (
            .clk        (clk),
            .reset      (reset),
            .lsu_mem_if (cpa_lmem_if[i]),
            .mem_bus_if (cpa_block_if)
        );

        for (genvar j = 0; j < `VX_CFG_NUM_LSU_LANES; ++j) begin : g_cpa_adapt_if
            `ASSIGN_VX_MEM_BUS_IF (lmem_adapt_if[LMEM_CPA_IDX + i * `VX_CFG_NUM_LSU_LANES + j], cpa_block_if[j]);
        end
    end
`endif

    // DMA arbiter: merge DXA writes and/or TCU reads onto single DMA port.

    VX_mem_bus_if #(
        .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
        .TAG_WIDTH   (LMEM_DMA_TAG_WIDTH),
        .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
    ) lmem_dma_if();

    localparam LMEM_DMA_IN_TAG_W = LMEM_DMA_IN_TAG_MAX;

    if (LMEM_DMA_INPUTS > 0) begin : g_lmem_dma

        VX_mem_bus_if #(
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (LMEM_DMA_IN_TAG_W),
            .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
        ) dma_arb_in_if[LMEM_DMA_INPUTS]();

        VX_mem_bus_if #(
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (LMEM_DMA_TAG_WIDTH),
            .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
        ) dma_arb_out_if[1]();

        // Wire DXA and/or TCU into the arbiter input array.
    `ifdef VX_CFG_EXT_DXA_ENABLE
        // DXA issues LMEM-relative word addresses directly (issuer_lmem_base + intra,
        // with multicast replay adding r×smem_stride per beat), so no per-slot
        // address translation is needed here.
    `ifdef DXA_MBAR_ENABLE
        // A last beat waits here while the completion queue is full, so no
        // completion is ever dropped; it reaches neither the arbiter nor a bank.
        wire dxa_compl_full;
        dxa_lmem_attr_t dxa_in_attr;
        assign dxa_in_attr = dxa_lmem_attr_t'(dxa_lmem_bus_if.req_data.attr);
        wire dxa_last_hold = dxa_in_attr.last && dxa_compl_full;
        `UNUSED_VAR (dxa_in_attr)

        VX_mem_bus_if #(
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
            .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
        ) dxa_held_if();

        assign dxa_held_if.req_valid     = dxa_lmem_bus_if.req_valid && ~dxa_last_hold;
        assign dxa_held_if.req_data      = dxa_lmem_bus_if.req_data;
        assign dxa_lmem_bus_if.req_ready = dxa_held_if.req_ready && ~dxa_last_hold;
        assign dxa_lmem_bus_if.rsp_valid = dxa_held_if.rsp_valid;
        assign dxa_lmem_bus_if.rsp_data  = dxa_held_if.rsp_data;
        assign dxa_held_if.rsp_ready     = dxa_lmem_bus_if.rsp_ready;

        `ASSIGN_VX_MEM_BUS_IF_EX (dma_arb_in_if[LMEM_DMA_DXA_IDX], dxa_held_if, LMEM_DMA_IN_TAG_W, DXA_LMEM_OUT_TAG_W, UUID_WIDTH);
    `else
        `ASSIGN_VX_MEM_BUS_IF_EX (dma_arb_in_if[LMEM_DMA_DXA_IDX], dxa_lmem_bus_if, LMEM_DMA_IN_TAG_W, DXA_LMEM_OUT_TAG_W, UUID_WIDTH);
    `endif
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        `ASSIGN_VX_MEM_BUS_IF_EX (dma_arb_in_if[LMEM_DMA_TCU_IDX], tcu_lmem_if, LMEM_DMA_IN_TAG_W, TCU_LMEM_TAG_W, UUID_WIDTH);
    `endif
    `ifdef DXA_MBAR_ENABLE
        // No buffer: the ingress takes the next flit only once the bank has
        // committed this one, which keeps a copy's completion behind its data.
        `ASSIGN_VX_MEM_BUS_IF (dma_arb_in_if[LMEM_DMA_ROW_IDX], dlmem_dma_if);
    `endif

        VX_mem_bus_arb #(
            .NUM_INPUTS  (LMEM_DMA_INPUTS),
            .NUM_OUTPUTS (1),
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (LMEM_DMA_IN_TAG_W),
            .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH),
            .ARBITER     ("P")
        ) lmem_dma_arb (
            .clk         (clk),
            .reset       (reset),
            .bus_in_if   (dma_arb_in_if),
            .bus_out_if  (dma_arb_out_if)
        );

        `ASSIGN_VX_MEM_BUS_IF (lmem_dma_if, dma_arb_out_if[0]);

    `ifdef VX_CFG_EXT_DXA_ENABLE
        wire [`VX_CFG_LMEM_NUM_BANKS-1:0] dxa_bank_wr_fire;
        for (genvar i = 0; i < `VX_CFG_LMEM_NUM_BANKS; ++i) begin : g_dxa_bank_wr_fire
        `ifdef VX_CFG_H100_SM_ENABLE
            // A port access covers the banks of its part of the row.
            localparam BANKS_PER_PORT = LMEM_DMA_DATA_SIZE / LSU_WORD_SIZE;
            wire in_part = (lmem_dma_if.req_data.addr % (`VX_CFG_LMEM_NUM_BANKS / BANKS_PER_PORT)) == (i / BANKS_PER_PORT);
            assign dxa_bank_wr_fire[i] = lmem_dma_if.req_valid
                                      && lmem_dma_if.req_ready
                                      && lmem_dma_if.req_data.rw
                                      && in_part
                                      && (|lmem_dma_if.req_data.byteen[(i % BANKS_PER_PORT)*LSU_WORD_SIZE +: LSU_WORD_SIZE]);
        `else
            assign dxa_bank_wr_fire[i] = lmem_dma_if.req_valid
                                      && lmem_dma_if.req_ready
                                      && lmem_dma_if.req_data.rw
                                      && (|lmem_dma_if.req_data.byteen[i*LSU_WORD_SIZE +: LSU_WORD_SIZE]);
        `endif
        end

        VX_dxa_completion #(
            .INSTANCE_ID (`SFORMATF(("%s-lmem-dma-compl_det", INSTANCE_ID))),
            .NUM_BANKS   (`VX_CFG_LMEM_NUM_BANKS),
            .ATTR_WIDTH  (DXA_LMEM_ATTR_W)
        ) dxa_completion_detect (
            .clk           (clk),
            .reset         (reset),
            .bank_wr_fire  (dxa_bank_wr_fire),
            .bank_wr_attr  (DXA_LMEM_ATTR_W'(lmem_dma_if.req_data.attr)),
        `ifdef DXA_MBAR_ENABLE
            .full          (dxa_compl_full),
            .slot_valid    (cta_slot_valid),
            .slot_gen      (cta_slot_gen),
            .slot_stride   (cta_slot_stride),
            .compl_bus_if  (dxa_compl_if),
        `endif
            .txbar_bus_if  (dxa_txbar_bus_if)
        );

        `ifdef DBG_TRACE_DXA
        always @(posedge clk) begin
            if (dxa_txbar_bus_if.valid && !reset) begin
                `TRACE(2, ("%t: %s-lmem-dma: dxa_txbar valid=1, bar_addr=0x%0h, ready=%b\n",
                    $time, INSTANCE_ID, dxa_txbar_bus_if.data.addr, dxa_txbar_bus_if.ready))
            end
        end
        `endif

    `endif

    end else begin : g_no_lmem_dma

        assign lmem_dma_if.req_valid = 1'b0;
        assign lmem_dma_if.req_data  = '0;
        `UNUSED_VAR (lmem_dma_if.req_ready)

        `UNUSED_VAR (lmem_dma_if.rsp_valid)
        `UNUSED_VAR (lmem_dma_if.rsp_data)
        assign lmem_dma_if.rsp_ready = 1'b0;

    end

`ifdef VX_CFG_EXT_MBAR_ENABLE
`ifdef VX_CFG_H100_SM_ENABLE
    localparam LMEM_ROW_WIDTH = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(`VX_CFG_LMEM_NUM_BANKS * `VX_CFG_LMEM_BANK_BYTES);
`else
    localparam LMEM_ROW_WIDTH = LMEM_ADDR_WIDTH - `CLOG2(`VX_CFG_LMEM_NUM_BANKS);
`endif
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0]                     mbar_evt_valid;
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0]                     mbar_evt_flip;
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0][NW_WIDTH-1:0]       mbar_evt_wid;
    wire [`VX_CFG_LMEM_NUM_BANKS-1:0][LMEM_ROW_WIDTH-1:0] mbar_evt_row;
`endif

`ifdef VX_CFG_H100_SM_ENABLE
    // The port's LMEM_PORT_BYTES accesses into the banks' wider rows: the
    // access's part of the row by its low address bits, carried in the tag
    // to pick the read data.
    localparam LMEM_ROW_BYTES = `VX_CFG_LMEM_NUM_BANKS * `VX_CFG_LMEM_BANK_BYTES;
    localparam LMEM_ROW_PARTS = LMEM_ROW_BYTES / LMEM_DMA_DATA_SIZE;
    localparam LMEM_PART_BITS = `CLOG2(LMEM_ROW_PARTS);
    `STATIC_ASSERT(LMEM_ROW_BYTES % LMEM_DMA_DATA_SIZE == 0, ("the shared-memory port must divide a bank row"))

    VX_mem_bus_if #(
        .DATA_SIZE   (LMEM_ROW_BYTES),
        .TAG_WIDTH   (LMEM_DMA_TAG_WIDTH + LMEM_PART_BITS),
        .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH - LMEM_PART_BITS)
    ) lmem_row_if();

    if (LMEM_ROW_PARTS > 1) begin : g_lmem_row_parts
        wire [LMEM_PART_BITS-1:0] req_part = lmem_dma_if.req_data.addr[LMEM_PART_BITS-1:0];
        wire [LMEM_PART_BITS-1:0] rsp_part = lmem_row_if.rsp_data.tag[LMEM_PART_BITS-1:0];
        assign lmem_row_if.req_valid       = lmem_dma_if.req_valid;
        assign lmem_row_if.req_data.rw     = lmem_dma_if.req_data.rw;
        assign lmem_row_if.req_data.addr   = lmem_dma_if.req_data.addr[LMEM_DMA_ADDR_WIDTH-1:LMEM_PART_BITS];
        assign lmem_row_if.req_data.data   = {LMEM_ROW_PARTS{lmem_dma_if.req_data.data}};
        assign lmem_row_if.req_data.byteen = LMEM_ROW_BYTES'(lmem_dma_if.req_data.byteen) << (req_part * LMEM_DMA_DATA_SIZE);
        assign lmem_row_if.req_data.attr   = lmem_dma_if.req_data.attr;
        assign lmem_row_if.req_data.tag    = {lmem_dma_if.req_data.tag, req_part};
        assign lmem_dma_if.req_ready       = lmem_row_if.req_ready;
        assign lmem_dma_if.rsp_valid       = lmem_row_if.rsp_valid;
        assign lmem_dma_if.rsp_data.data   = lmem_row_if.rsp_data.data[rsp_part * LMEM_DMA_DATA_SIZE * 8 +: LMEM_DMA_DATA_SIZE * 8];
        assign lmem_dma_if.rsp_data.tag    = lmem_row_if.rsp_data.tag[LMEM_PART_BITS +: LMEM_DMA_TAG_WIDTH];
        assign lmem_row_if.rsp_ready       = lmem_dma_if.rsp_ready;
    end else begin : g_lmem_row_whole
        `ASSIGN_VX_MEM_BUS_IF (lmem_row_if, lmem_dma_if);
    end

    // Every other client's access takes the port for a cycle, charged when
    // it is first presented.
`ifdef VX_CFG_EXT_MBAR_ENABLE
    localparam LMEM_ALL_REQS = LMEM_NUM_REQS;
`elsif VX_CFG_EXT_DLMEM_ENABLE
    localparam LMEM_ALL_REQS = LSU_NUM_REQS + 1;
`else
    localparam LMEM_ALL_REQS = LSU_NUM_REQS;
`endif
    localparam LMEM_EXT_REQS = LMEM_ALL_REQS - LSU_NUM_REQS + 1;
    wire [LMEM_EXT_REQS-1:0] lmem_ext_req, lmem_ext_fire;
    assign lmem_ext_req[0]  = LMEM_DMA_EN && lmem_dma_if.req_valid;
    assign lmem_ext_fire[0] = LMEM_DMA_EN && lmem_dma_if.req_valid && lmem_dma_if.req_ready;
    for (genvar i = 1; i < LMEM_EXT_REQS; ++i) begin : g_lmem_ext
        assign lmem_ext_req[i]  = lmem_adapt_if[LSU_NUM_REQS + i - 1].req_valid;
        assign lmem_ext_fire[i] = lmem_adapt_if[LSU_NUM_REQS + i - 1].req_valid && lmem_adapt_if[LSU_NUM_REQS + i - 1].req_ready;
    end
    reg [LMEM_EXT_REQS-1:0] lmem_ext_wait_r;
    always @(posedge clk) begin
        if (reset) begin
            lmem_ext_wait_r <= '0;
        end else begin
            lmem_ext_wait_r <= lmem_ext_req & ~lmem_ext_fire;
        end
    end
    assign lmem_ext_valid = |(lmem_ext_req & ~lmem_ext_wait_r);
`endif

    VX_local_mem #(
        .INSTANCE_ID (`SFORMATF(("%s-lmem", INSTANCE_ID))),
        .SIZE        (1 << `VX_CFG_LMEM_LOG_SIZE),
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        // The cp.async lanes' stores are acknowledged; they report no commits.
        .NUM_REQS    (LMEM_NUM_REQS),
        .STORE_ACK_MASK ((LMEM_NUM_REQS'(1) << LMEM_INGRESS_IDX)
                       | (((LMEM_NUM_REQS'(1) << LSU_NUM_REQS) - LMEM_NUM_REQS'(1)) << LMEM_CPA_IDX)),
        .COMMIT_MASK ((LMEM_NUM_REQS'(1) << LMEM_INGRESS_IDX) - LMEM_NUM_REQS'(1)),
        .LOW_PRI_REQS (LSU_NUM_REQS),
    `elsif VX_CFG_EXT_DLMEM_ENABLE
        .NUM_REQS    (LSU_NUM_REQS + 1),
        .STORE_ACK_MASK ((LSU_NUM_REQS + 1)'(1) << LMEM_INGRESS_IDX),
        .COMMIT_MASK (((LSU_NUM_REQS + 1)'(1) << LMEM_INGRESS_IDX) - (LSU_NUM_REQS + 1)'(1)),
    `else
        .NUM_REQS    (LSU_NUM_REQS),
    `endif
        .NUM_BANKS   (`VX_CFG_LMEM_NUM_BANKS),
        .WORD_SIZE   (LSU_WORD_SIZE),
        .ADDR_WIDTH  (LMEM_ADDR_WIDTH),
        .TAG_WIDTH   (LSU_TAG_WIDTH),
        .DMA_ENABLE  (LMEM_DMA_EN),
    `ifdef VX_CFG_H100_SM_ENABLE
        .DMA_TAG_WIDTH (LMEM_DMA_TAG_WIDTH + LMEM_PART_BITS),
    `else
        .DMA_TAG_WIDTH (LMEM_DMA_TAG_WIDTH),
    `endif
        .AMO_ENABLE  (`VX_CFG_EXT_A_ENABLED),
        .OUT_BUF     (3)
    ) local_mem (
        .clk         (clk),
        .reset       (reset),
    `ifdef PERF_ENABLE
        .lmem_perf   (lmem_perf),
    `endif
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        .lmem_commit_valid (lmem_commit_valid),
        .lmem_commit_wid   (lmem_commit_wid),
    `endif
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        .mbar_evt_valid    (mbar_evt_valid),
        .mbar_evt_flip     (mbar_evt_flip),
        .mbar_evt_wid      (mbar_evt_wid),
        .mbar_evt_row      (mbar_evt_row),
    `endif
    `ifdef VX_CFG_H100_SM_ENABLE
        .dma_bus_if  (lmem_row_if),
    `else
        .dma_bus_if  (lmem_dma_if),
    `endif
        .lsu_bus_if  (lmem_adapt_if)
    );

`ifdef VX_CFG_EXT_MBAR_ENABLE
    VX_mbar_waiters #(
        .INSTANCE_ID (`SFORMATF(("%s-mbar-waiters", INSTANCE_ID))),
        .NUM_BANKS   (`VX_CFG_LMEM_NUM_BANKS),
        .NUM_RETIRES (`VX_CFG_NUM_LSU_BLOCKS),
        .ROW_WIDTH   (LMEM_ROW_WIDTH)
    ) mbar_waiters (
        .clk          (clk),
        .reset        (reset),
        .evt_valid    (mbar_evt_valid),
        .evt_flip     (mbar_evt_flip),
        .evt_wid      (mbar_evt_wid),
        .evt_row      (mbar_evt_row),
        .retire_valid (mbar_retire_valid),
        .retire_wid   (mbar_retire_wid),
        .unlock_valid (mbar_unlock_valid),
        .unlock_mask  (mbar_unlock_mask)
    );
`endif

`else

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_lsu_dcache_if
        `ASSIGN_VX_MEM_BUS_IF (lsu_dcache_if[i], lsu_mem_if[i]);
    end

`ifdef PERF_ENABLE
    assign lmem_perf = '0;
`endif

`endif

    VX_lsu_mem_if #(
        .NUM_LANES (DCACHE_CHANNELS),
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH)
    ) dcache_coalesced_if[`VX_CFG_NUM_LSU_BLOCKS]();

`ifdef PERF_ENABLE
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][PERF_CTR_BITS-1:0] per_block_coalescer_misses;
    wire [PERF_CTR_BITS-1:0] coalescer_misses;
    VX_reduce_tree #(
        .IN_W (PERF_CTR_BITS),
        .N    (`VX_CFG_NUM_LSU_BLOCKS),
        .OP   ("+")
    ) coalescer_reduce (
        .data_in  (per_block_coalescer_misses),
        .data_out (coalescer_misses)
    );
    `BUFFER(coalescer_perf.misses, coalescer_misses);
`endif

    if ((`VX_CFG_NUM_LSU_LANES > 1) && (DCACHE_WORD_SIZE > LSU_WORD_SIZE)) begin : g_coalescing

        for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_coalescers

        `ifdef VX_CFG_H100_SM_ENABLE
            VX_mem_coalescer_h100 #(
        `else
            VX_mem_coalescer #(
        `endif
                .INSTANCE_ID    (`SFORMATF(("%s-coalescer%0d", INSTANCE_ID, i))),
                .NUM_REQS       (`VX_CFG_NUM_LSU_LANES),
                .DATA_IN_SIZE   (LSU_WORD_SIZE),
                .DATA_OUT_SIZE  (DCACHE_WORD_SIZE),
                .ADDR_WIDTH     (LSU_ADDR_WIDTH),
                .USER_WIDTH     (MEM_ATTR_WIDTH),
                .TAG_WIDTH      (LSU_TAG_WIDTH),
                .UUID_WIDTH     (UUID_WIDTH),
                .QUEUE_SIZE     (LSU_QUEUE_OUT_SIZE),
                .PERF_CTR_BITS  (PERF_CTR_BITS)
            ) mem_coalescer (
                .clk            (clk),
                .reset          (reset),

            `ifdef PERF_ENABLE
                .misses         (per_block_coalescer_misses[i]),
            `else
                `UNUSED_PIN (misses),
            `endif

                .empty          (per_block_path_empty[i]),

                // Input request
                .in_req_valid   (lsu_dcache_if[i].req_valid),
                .in_req_mask    (lsu_dcache_if[i].req_data.mask),
                .in_req_rw      (lsu_dcache_if[i].req_data.rw),
                .in_req_byteen  (lsu_dcache_if[i].req_data.byteen),
                .in_req_addr    (lsu_dcache_if[i].req_data.addr),
                .in_req_user    (lsu_dcache_if[i].req_data.user),
                .in_req_no_merge(lsu_dcache_if[i].req_data.user[0][MEM_ATTR_AMO_OFFS]),
                .in_req_data    (lsu_dcache_if[i].req_data.data),
                .in_req_tag     (lsu_dcache_if[i].req_data.tag),
                .in_req_ready   (lsu_dcache_if[i].req_ready),

                // Input response
                .in_rsp_valid   (lsu_dcache_if[i].rsp_valid),
                .in_rsp_mask    (lsu_dcache_if[i].rsp_data.mask),
                .in_rsp_data    (lsu_dcache_if[i].rsp_data.data),
                .in_rsp_tag     (lsu_dcache_if[i].rsp_data.tag),
                .in_rsp_ready   (lsu_dcache_if[i].rsp_ready),

                // Output request
                .out_req_valid  (dcache_coalesced_if[i].req_valid),
                .out_req_mask   (dcache_coalesced_if[i].req_data.mask),
                .out_req_rw     (dcache_coalesced_if[i].req_data.rw),
                .out_req_byteen (dcache_coalesced_if[i].req_data.byteen),
                .out_req_addr   (dcache_coalesced_if[i].req_data.addr),
                .out_req_user   (dcache_coalesced_if[i].req_data.user),
                .out_req_data   (dcache_coalesced_if[i].req_data.data),
                .out_req_tag    (dcache_coalesced_if[i].req_data.tag),
                .out_req_ready  (dcache_coalesced_if[i].req_ready),

                // Output response
                .out_rsp_valid  (dcache_coalesced_if[i].rsp_valid),
                .out_rsp_mask   (dcache_coalesced_if[i].rsp_data.mask),
                .out_rsp_data   (dcache_coalesced_if[i].rsp_data.data),
                .out_rsp_tag    (dcache_coalesced_if[i].rsp_data.tag),
                .out_rsp_ready  (dcache_coalesced_if[i].rsp_ready)
            );
        end

    end else begin : g_no_coalescing

        // The passthrough narrows the tag from LSU_TAG_WIDTH to
        // DCACHE_CORE_TAG_WIDTH: the dcache tag-id space must cover the
        // mem-scheduler slot id or responses alias.
        `STATIC_ASSERT (DCACHE_CORE_TAG_WIDTH >= LSU_TAG_WIDTH, ("dcache tag-id space cannot hold the LSU outstanding slot id"))

        for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_dcache_coalesced_if
            `ASSIGN_VX_MEM_BUS_IF (dcache_coalesced_if[i], lsu_dcache_if[i]);
        `ifdef PERF_ENABLE
            assign per_block_coalescer_misses[i] = '0;
        `endif
            // No coalescer: a store still in flight upstream of the adapter is
            // one being presented to the adapter input. The adapter's own buffer
            // is accounted for separately by per_block_adapter_empty.
            assign per_block_path_empty[i] = ~dcache_coalesced_if[i].req_valid;
        end

    end

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_dcache_adapters

        VX_mem_bus_if #(
            .DATA_SIZE (DCACHE_WORD_SIZE),
            .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH)
        ) dcache_bus_tmp_if[DCACHE_CHANNELS]();

        VX_lsu_adapter #(
            .NUM_LANES    (DCACHE_CHANNELS),
            .DATA_SIZE    (DCACHE_WORD_SIZE),
            .TAG_WIDTH    (DCACHE_CORE_TAG_WIDTH),
            .TAG_SEL_BITS (DCACHE_CORE_TAG_WIDTH - UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (3), // register dcache-request master boundary on all channels (passthru ports 1+)
            .RSP_OUT_BUF  (0)
        ) dcache_adapter (
            .clk        (clk),
            .reset      (reset),
            .lsu_mem_if (dcache_coalesced_if[i]),
            .mem_bus_if (dcache_bus_tmp_if)
        );

        // A store buffered in the adapter's REQ_OUT_BUF has left dcache_coalesced_if
        // but not yet reached the flush-injection point; keep the block non-empty
        // until every channel's request has drained past the adapter output.
        wire [DCACHE_CHANNELS-1:0] adapter_req_valid;
        for (genvar j = 0; j < DCACHE_CHANNELS; ++j) begin : g_adapter_req_valid
            assign adapter_req_valid[j] = dcache_bus_tmp_if[j].req_valid;
        end
        assign per_block_adapter_empty[i] = ~(| adapter_req_valid);

        for (genvar j = 0; j < DCACHE_CHANNELS; ++j) begin : g_dcache_bus_if
            if (i == 0 && j == 0) begin : g_flush_port
                // Port 0: route through VX_dcr_flush to inject flush requests
                VX_dcr_flush #(
                    .WORD_SIZE (DCACHE_WORD_SIZE),
                    .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH),
                    .REQ_OUT_BUF (3) // register core dcache-request boundary; rsp already registered by L1 CORE_OUT_BUF
                ) dcr_flush (
                    .clk          (clk),
                    .reset        (reset),
                    .dcr_flush_if (dcr_flush_if),
                    .core_bus_if  (dcache_bus_tmp_if[0]),
                    .cache_bus_if (dcache_bus_if[0])
                );
            end else begin : g_passthru_port
                // Ports 1+: pass through; tag is zero-extended from
                // DCACHE_CORE_TAG_WIDTH to DCACHE_TAG_WIDTH on request,
                // and MSB-stripped on response.
                `ASSIGN_VX_MEM_BUS_IF_EX (dcache_bus_if[i * DCACHE_CHANNELS + j], dcache_bus_tmp_if[j], DCACHE_TAG_WIDTH_BASE, DCACHE_CORE_TAG_WIDTH, 0);
            end
        end

    end

    assign empty = (& per_block_empty);

`ifdef DBG_TRACE_MEM
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // A CTA fence copy reaching the dcache: the drain it starts covers the requests already accepted.
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin : g_trace_cta
        wire is_cta = dcache_bus_if[i].req_valid && dcache_bus_if[i].req_data.attr[MEM_ATTR_FENCE_CTA_OFFS];
        reg waiting_r;
        always @(posedge clk) begin
            waiting_r <= ~reset && is_cta && ~dcache_bus_if[i].req_ready;
            if (~reset && is_cta && ~waiting_r) begin
                `TRACE(2, ("%t: %s dcache-cta-fence[%0d]: arrive (#%0d)\n", $time, INSTANCE_ID, i, dcache_bus_if[i].req_data.tag.uuid))
            end
        end
    end
`endif
`endif

endmodule
