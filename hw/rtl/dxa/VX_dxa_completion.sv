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

module VX_dxa_completion import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BANKS = 1,
    parameter ATTR_WIDTH = DXA_LMEM_ATTR_W
) (
    input  wire                                clk,
    input  wire                                reset,

    // DXA bank writes (from dedicated DMA port, all are DXA by definition)
    input  wire [NUM_BANKS-1:0]                bank_wr_fire,
    input  wire [ATTR_WIDTH-1:0]               bank_wr_attr,  // the beat's DXA attribute: its last flag on top, the queue entry below

`ifdef DXA_MBAR_ENABLE
    // The queue cannot take another last beat: the DMA port holds it.
    output wire                                full,

    // The core's CTA slots, for the retired-slot and bounds checks.
    input  wire [`VX_CFG_NUM_WARPS-1:0]        slot_valid,
    input  wire [`VX_CFG_NUM_WARPS-1:0][DXA_CTA_GEN_W-1:0] slot_gen,
    input  wire [`VX_CFG_LMEM_LOG_SIZE:0]      slot_stride,

    // complete_tx requests into the local memory; their responses are dropped.
    VX_mem_bus_if.master                       compl_bus_if,
`endif

    VX_txbar_bus_if.master                     txbar_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    wire is_last = bank_wr_attr[ATTR_WIDTH-1];
    wire any_dxa_wr = |bank_wr_fire;

    wire done_fire = any_dxa_wr && is_last;
`ifdef DXA_MBAR_ENABLE
    `STATIC_ASSERT (ATTR_WIDTH == $bits(dxa_lmem_attr_t), ("invalid parameter"))
    // The queue entry is the attribute below its last bit.
    localparam ENTRY_W = ATTR_WIDTH - 1;
    dxa_lmem_attr_t wr_attr;
    assign wr_attr = dxa_lmem_attr_t'(bank_wr_attr);
    wire [ENTRY_W-1:0] done_fire_bar = bank_wr_attr[ENTRY_W-1:0];
`else
    wire [BAR_ADDR_W-1:0] done_fire_bar = bank_wr_attr[BAR_ADDR_W-1:0];
`endif

    // ════════════════════════════════════════════════════════════════════
    // Release-event FIFO.
    //
    // Multicast can fire popcount(cta_mask) back-to-back release events on
    // the LAST drain word — one per receiver. A single pending slot would
    // drop events if downstream (wctl_unit) is slower than the burst rate.
    //
    // Depth = NUM_WARPS handles the worst-case intra-core multicast burst
    // (NUM_WARPS receivers in NUM_WARPS consecutive cycles). The DXA worker
    // never fires releases faster than 1/cycle, so a per-cycle drain rate
    // ≥ 1/2 keeps the FIFO bounded. An assertion catches overflow.
    // ════════════════════════════════════════════════════════════════════
`ifdef DXA_MBAR_ENABLE
    wire [ENTRY_W-1:0] fifo_dout;
`else
    wire [BAR_ADDR_W-1:0] fifo_dout;
`endif
    wire fifo_empty;
    wire fifo_full;
    wire fifo_push = done_fire;
`ifdef DXA_MBAR_ENABLE
    dxa_lmem_attr_t head;
    assign head = dxa_lmem_attr_t'({1'b1, fifo_dout});
    wire compl_fire = compl_bus_if.req_valid && compl_bus_if.req_ready;
    wire fifo_pop  = (txbar_bus_if.valid && txbar_bus_if.ready) || compl_fire;
`else
    wire fifo_pop  = txbar_bus_if.valid && txbar_bus_if.ready;
`endif

    // DEPTH must be a power of 2 and >= 2 (the queue's ALM_FULL defaults to
    // DEPTH-1, which fails its `ALM_FULL > 0` static assert at DEPTH=1).
    // Bump to 2 when NUM_WARPS=1 — the depth doesn't affect correctness, just
    // backpressure granularity.
    localparam FIFO_DEPTH = (`VX_CFG_NUM_WARPS < 2) ? 2 : `VX_CFG_NUM_WARPS;
    VX_fifo_queue #(
    `ifdef DXA_MBAR_ENABLE
        .DATAW  (ENTRY_W),
    `else
        .DATAW  (BAR_ADDR_W),
    `endif
        .DEPTH  (FIFO_DEPTH),
        .LUTRAM (1)
    ) compl_fifo (
        .clk        (clk),
        .reset      (reset),
        .push       (fifo_push),
        .pop        (fifo_pop),
        .data_in    (done_fire_bar),
        .data_out   (fifo_dout),
        .empty      (fifo_empty),
        .full       (fifo_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

`ifdef DXA_MBAR_ENABLE
    assign full = fifo_full;

    // Legacy entries release their barrier; mbarrier entries become a
    // complete_tx on the mbarrier word, executed by the bank's atomic unit.
    assign txbar_bus_if.valid        = ~fifo_empty && ~head.mode;
    assign txbar_bus_if.data.addr    = head.payload[BAR_ADDR_W-1:0];
    assign txbar_bus_if.data.is_done = 1'b1;

    // An atomic of hart 0.
    localparam mem_bus_attr_t COMPL_ATTR = '{
        amo: '{amo_valid: 1'b1, amo_op: AMO_OP_MBAR_COMPLETE_TX, default: '0},
        default: '0
    };

    assign compl_bus_if.req_valid       = ~fifo_empty && head.mode;
    assign compl_bus_if.req_data.rw     = 1'b0;
    assign compl_bus_if.req_data.addr   = $bits(compl_bus_if.req_data.addr)'(head.payload[DXA_MBAR_OFF_W-1:0]);
    assign compl_bus_if.req_data.data   = $bits(compl_bus_if.req_data.data)'(head.tx_bytes);
    assign compl_bus_if.req_data.byteen = '1;
    assign compl_bus_if.req_data.attr   = $bits(compl_bus_if.req_data.attr)'(COMPL_ATTR);
    assign compl_bus_if.req_data.tag    = '0;
    assign compl_bus_if.rsp_ready       = 1'b1;
    `UNUSED_VAR (compl_bus_if.rsp_valid)
    `UNUSED_VAR (compl_bus_if.rsp_data)

    // Simulation-only checks: a block that leaves before its transfer lands,
    // and an mbarrier outside the issuing block's local memory.
    localparam BOUND_W = `VX_CFG_LMEM_LOG_SIZE + NW_WIDTH + 2;
    wire [BOUND_W-1:0] blk_lo  = BOUND_W'(head.slot) * BOUND_W'(slot_stride);
    wire [BOUND_W-1:0] blk_hi  = blk_lo + BOUND_W'(slot_stride);
    wire [BOUND_W-1:0] mbar_ba = BOUND_W'(head.payload[DXA_MBAR_OFF_W-1:0]) << 3;
    wire beat_live  = slot_valid[wr_attr.slot] && (slot_gen[wr_attr.slot] == wr_attr.gen);
    wire compl_live = slot_valid[head.slot] && (slot_gen[head.slot] == head.gen);
    `UNUSED_VAR (wr_attr)
    `UNUSED_VAR (head)

    `RUNTIME_ASSERT(~any_dxa_wr || beat_live,
        ("%t: *** %s DXA write into a retired CTA slot: slot=%0d gen=%0d (slot valid=%b gen=%0d)",
         $time, INSTANCE_ID, wr_attr.slot, wr_attr.gen, slot_valid[wr_attr.slot], slot_gen[wr_attr.slot]))
    `RUNTIME_ASSERT(~compl_fire || compl_live,
        ("%t: *** %s DXA write into a retired CTA slot: complete_tx slot=%0d gen=%0d (slot valid=%b gen=%0d)",
         $time, INSTANCE_ID, head.slot, head.gen, slot_valid[head.slot], slot_gen[head.slot]))
    `RUNTIME_ASSERT(~compl_fire || ((mbar_ba >= blk_lo) && (mbar_ba < blk_hi)),
        ("%t: *** %s mbarrier offset outside the block: byte offset 0x%0h, slot=%0d block=[0x%0h, 0x%0h)",
         $time, INSTANCE_ID, mbar_ba, head.slot, blk_lo, blk_hi))
`else
    assign txbar_bus_if.valid        = ~fifo_empty;
    assign txbar_bus_if.data.addr    = fifo_dout;
    assign txbar_bus_if.data.is_done = 1'b1;
`endif

    `RUNTIME_ASSERT(~(fifo_push && fifo_full),
        ("%t: %s overflow — multicast release FIFO full, event for bar=0x%0h would be dropped",
         $time, INSTANCE_ID, done_fire_bar))

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (any_dxa_wr && !reset) begin
            `TRACE(2, ("%t: %s: bank_wr_fire=%b, attr=0x%0h, is_last=%b, done_fire=%b, fifo_empty=%b, valid=%b, ready=%b\n",
                $time, INSTANCE_ID, bank_wr_fire, bank_wr_attr, is_last, done_fire,
                fifo_empty, txbar_bus_if.valid, txbar_bus_if.ready))
        end
    `ifdef DXA_MBAR_ENABLE
        if (txbar_bus_if.valid && txbar_bus_if.ready && !reset) begin
            `TRACE(2, ("%t: %s: ACCEPTED bar_addr=0x%0h\n",
                $time, INSTANCE_ID, txbar_bus_if.data.addr))
        end
        if (compl_fire && !reset) begin
            `TRACE(2, ("%t: %s: ACCEPTED complete_tx off=0x%0h, bytes=%0d\n",
                $time, INSTANCE_ID, head.payload[DXA_MBAR_OFF_W-1:0], head.tx_bytes))
        end
    `else
        if (fifo_pop && !reset) begin
            `TRACE(2, ("%t: %s: ACCEPTED bar_addr=0x%0h\n",
                $time, INSTANCE_ID, txbar_bus_if.data.addr))
        end
    `endif
    `ifdef DXA_MBAR_ENABLE
        if (~reset && txbar_bus_if.valid && txbar_bus_if.ready) begin
            $write("DXA_TL,%0d,DONE_DETECT,%s,bar=%0d\n",
                $time, INSTANCE_ID, txbar_bus_if.data.addr);
        end
        if (~reset && compl_fire) begin
            $write("DXA_TL,%0d,COMPLETE_TX,%s,off=0x%0h,bytes=%0d,slot=%0d\n",
                $time, INSTANCE_ID, head.payload[DXA_MBAR_OFF_W-1:0], head.tx_bytes, head.slot);
        end
        if (~reset && fifo_full) begin
            $write("DXA_TL,%0d,QUEUE_FULL,%s\n", $time, INSTANCE_ID);
        end
    `else
        if (~reset && fifo_pop) begin
            $write("DXA_TL,%0d,DONE_DETECT,%s,bar=%0d\n",
                $time, INSTANCE_ID, txbar_bus_if.data.addr);
        end
    `endif
    end
`endif
endmodule
