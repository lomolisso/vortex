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

// A core's DLMEM ingress: executes remote requests on the core's local memory
// and returns their responses. A request is accepted only into a free slot,
// which reserves its response, so the local-memory response port never stalls
// on the network; responses leave in local-memory completion order. A store's
// response is its acknowledgement, sent once the bank has committed it.
// With DXA mbarriers, a remote copy's posted row flits arrive in order: a data
// word goes straight to the local memory's DMA port, which commits it as it
// accepts it, so the copy's completion, a complete_tx taken in a slot after
// it, reaches its bank later; neither sends a response.
// With MBAR, an st.async/red.async keeps its slot through two local-memory
// requests on the same input: the access, then, once the access's response
// has returned (after its bank commit), a complete_tx of its bytes on its
// mbarrier, whose response is dropped. The ack follows the complete_tx's
// acceptance and carries no data.
module VX_dlmem_ingress import VX_gpu_pkg::*, VX_dlmem_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_SLOTS = DLMEM_INGRESS_SLOTS,
    parameter OUT_BUF   = 3
) (
    input wire              clk,
    input wire              reset,

    VX_dlmem_bus_if.slave   req_in_if,
    VX_mem_bus_if.master    lmem_bus_if,
    VX_dlmem_bus_if.master  rsp_out_if,

`ifdef DXA_MBAR_ENABLE
    // The core's occupied CTA slots, and its local memory's DMA port.
    input wire [`VX_CFG_NUM_WARPS-1:0] slot_valid,
    VX_mem_bus_if.master    dma_out_if,
`endif

    output wire             empty
);
    localparam SLOT_W     = `LOG2UP(NUM_SLOTS);
    localparam WORD_W     = LSU_WORD_SIZE * 8;
    localparam LMEM_DATAW = 1 + DLMEM_LMEM_AW + WORD_W + LSU_WORD_SIZE + MEM_ATTR_WIDTH + LSU_TAG_WIDTH;
    localparam RSP_DATAW  = DLMEM_CORE_W + DLMEM_RSP_DATAW;

    `STATIC_ASSERT (NUM_SLOTS >= 1, ("invalid parameter"))
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // One more tag bit marks the complete_tx whose response is dropped.
    `STATIC_ASSERT (SLOT_W + 1 <= LSU_TAG_WIDTH - UUID_WIDTH, ("dlmem ingress: local-memory tag too narrow"))
`else
    `STATIC_ASSERT (SLOT_W <= LSU_TAG_WIDTH - UUID_WIDTH, ("dlmem ingress: local-memory tag too narrow"))
`endif

    reg [NUM_SLOTS-1:0]                    busy_r;
    reg [NUM_SLOTS-1:0]                    done_r;
    reg [NUM_SLOTS-1:0]                    ack_r;
    reg [NUM_SLOTS-1:0][DLMEM_CORE_W-1:0]  src_r;
    reg [NUM_SLOTS-1:0][DLMEM_TAG_W-1:0]   tag_r;
    reg [NUM_SLOTS-1:0][WORD_W-1:0]        data_r;
`ifdef VX_CFG_EXT_MBAR_ENABLE
    reg [NUM_SLOTS-1:0]                    async_r;
    reg [NUM_SLOTS-1:0]                    cx_sent_r;
    reg [NUM_SLOTS-1:0][DLMEM_LMEM_AW-1:0] mbar_addr_r;
    reg [NUM_SLOTS-1:0][`CLOG2(LSU_WORD_SIZE+1)-1:0] bytes_r;
`endif

    // Accept into the lowest free slot.

    wire [SLOT_W-1:0] free_idx;
    wire              free_valid;

    VX_priority_encoder #(
        .N (NUM_SLOTS)
    ) free_sel (
        .data_in    (~busy_r),
        `UNUSED_PIN (onehot_out),
        .index_out  (free_idx),
        .valid_out  (free_valid)
    );

    dlmem_req_pkt_t req_pkt;
    assign req_pkt = dlmem_req_pkt_t'(req_in_if.data);

`ifdef DXA_MBAR_ENABLE
    wire is_row   = (req_pkt.op == DLMEM_OP_DXA);
    wire is_data  = is_row && ~req_pkt.amo_unsigned;
    wire is_compl = is_row && req_pkt.amo_unsigned;

    wire cx_valid;
    wire lmem_req_valid = req_in_if.valid && ~is_data && free_valid && ~cx_valid;
    wire lmem_req_ready;
    wire req_fire = lmem_req_valid && lmem_req_ready;

    assign req_in_if.ready = is_data ? dma_out_if.req_ready : (free_valid && lmem_req_ready && ~cx_valid);
    `UNUSED_VAR (req_in_if.dst)

    // A data word is one bank's lane of a DMA beat.
    localparam NUM_BANKS = `VX_CFG_LMEM_NUM_BANKS;
    localparam BANK_BITS = `CLOG2(NUM_BANKS);
    wire [`UP(BANK_BITS)-1:0] row_bank;
    if (BANK_BITS != 0) begin : g_row_bank
        assign row_bank = req_pkt.addr[BANK_BITS-1:0];
    end else begin : g_row_bank0
        assign row_bank = '0;
    end

    dxa_lmem_attr_t row_attr;
    always @(*) begin
        row_attr        = '0;
        row_attr.slot   = NW_WIDTH'(req_pkt.tag);
        row_attr.remote = 1'b1;
    end

    assign dma_out_if.req_valid       = req_in_if.valid && is_data;
    assign dma_out_if.req_data.rw     = 1'b1;
    assign dma_out_if.req_data.addr   = $bits(dma_out_if.req_data.addr)'(req_pkt.addr >> BANK_BITS);
    assign dma_out_if.req_data.data   = {NUM_BANKS{req_pkt.data}};
    assign dma_out_if.req_data.byteen = $bits(dma_out_if.req_data.byteen)'(req_pkt.byteen) << (32'(row_bank) * LSU_WORD_SIZE);
    assign dma_out_if.req_data.attr   = $bits(dma_out_if.req_data.attr)'(row_attr);
    assign dma_out_if.req_data.tag    = '0;
    assign dma_out_if.rsp_ready       = 1'b1;
    `UNUSED_VAR (dma_out_if.rsp_valid)
    `UNUSED_VAR (dma_out_if.rsp_data)
`elsif VX_CFG_EXT_MBAR_ENABLE
    wire cx_valid;
    wire lmem_req_valid = req_in_if.valid && free_valid && ~cx_valid;
    wire lmem_req_ready;
    wire req_fire = lmem_req_valid && lmem_req_ready;

    assign req_in_if.ready = free_valid && lmem_req_ready && ~cx_valid;
    `UNUSED_VAR (req_in_if.dst)
`else
    wire lmem_req_valid = req_in_if.valid && free_valid;
    wire lmem_req_ready;
    wire req_fire = lmem_req_valid && lmem_req_ready;

    assign req_in_if.ready = free_valid && lmem_req_ready;
    `UNUSED_VAR (req_in_if.dst)
`endif

    mem_bus_attr_t lmem_attr;
    always @(*) begin
        lmem_attr                  = '0;
        lmem_attr.is_addr_local    = 1'b1;
    `ifdef DXA_MBAR_ENABLE
        lmem_attr.amo.amo_valid    = (req_pkt.op == DLMEM_OP_AMO) || is_compl;
        lmem_attr.amo.amo_op       = req_pkt.amo_op;
        lmem_attr.amo.amo_unsigned = req_pkt.amo_unsigned && ~is_compl;
    `else
        lmem_attr.amo.amo_valid    = (req_pkt.op == DLMEM_OP_AMO);
        lmem_attr.amo.amo_op       = req_pkt.amo_op;
        lmem_attr.amo.amo_unsigned = req_pkt.amo_unsigned;
    `endif
        lmem_attr.amo.hart_id      = req_pkt.hart_id;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        lmem_attr.amo.amo_cmp      = req_pkt.amo_cmp;
    `endif
    end

    wire [LSU_TAG_WIDTH-1:0] lmem_tag = {UUID_WIDTH'(0), (LSU_TAG_WIDTH-UUID_WIDTH)'(free_idx)};

`ifdef VX_CFG_EXT_MBAR_ENABLE
    // The oldest completed slot, if async, sends its complete_tx first; new
    // requests wait, and the local memory never waits on the network.
    wire [SLOT_W-1:0] head_slot;
    wire              cpl_empty;
    assign cx_valid = ~cpl_empty && async_r[head_slot] && ~cx_sent_r[head_slot];
    wire cx_fire = cx_valid && lmem_req_ready;

    // An atomic of hart 0.
    localparam mem_bus_attr_t CX_ATTR = '{
        is_addr_local: 1'b1,
        amo: '{amo_valid: 1'b1, amo_op: AMO_OP_MBAR_COMPLETE_TX, default: '0},
        default: '0
    };
    wire [LSU_TAG_WIDTH-1:0] cx_tag = {UUID_WIDTH'(0), (LSU_TAG_WIDTH-UUID_WIDTH)'({1'b1, head_slot})};

    VX_elastic_buffer #(
        .DATAW   (LMEM_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
    ) lmem_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lmem_req_valid || cx_valid),
        .ready_in  (lmem_req_ready),
        .data_in   (cx_valid ? {
            1'b0,
            mbar_addr_r[head_slot],
            WORD_W'(bytes_r[head_slot]),
            {LSU_WORD_SIZE{1'b1}},
            MEM_ATTR_WIDTH'(CX_ATTR),
            cx_tag
        } : {
            (req_pkt.op == DLMEM_OP_ST),
            req_pkt.addr,
            req_pkt.data,
            req_pkt.byteen,
            MEM_ATTR_WIDTH'(lmem_attr),
            lmem_tag
        }),
        .data_out  (lmem_bus_if.req_data),
        .valid_out (lmem_bus_if.req_valid),
        .ready_out (lmem_bus_if.req_ready)
    );

    // Local-memory responses complete their slot; they are never refused.
    // A complete_tx's response completes nothing.

    assign lmem_bus_if.rsp_ready = 1'b1;

    wire [SLOT_W-1:0] rsp_slot = lmem_bus_if.rsp_data.tag.value[SLOT_W-1:0];
    wire rsp_drop = lmem_bus_if.rsp_data.tag.value[SLOT_W];
    wire lmem_rsp_fire = lmem_bus_if.rsp_valid && ~rsp_drop;
    `UNUSED_VAR (lmem_bus_if.rsp_data.tag)
`else
    VX_elastic_buffer #(
        .DATAW   (LMEM_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
    ) lmem_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lmem_req_valid),
        .ready_in  (lmem_req_ready),
        .data_in   ({
            (req_pkt.op == DLMEM_OP_ST),
            req_pkt.addr,
            req_pkt.data,
            req_pkt.byteen,
            MEM_ATTR_WIDTH'(lmem_attr),
            lmem_tag
        }),
        .data_out  (lmem_bus_if.req_data),
        .valid_out (lmem_bus_if.req_valid),
        .ready_out (lmem_bus_if.req_ready)
    );

    // Local-memory responses complete their slot; they are never refused.

    assign lmem_bus_if.rsp_ready = 1'b1;

    wire [SLOT_W-1:0] rsp_slot = lmem_bus_if.rsp_data.tag.value[SLOT_W-1:0];
    wire lmem_rsp_fire = lmem_bus_if.rsp_valid;
    `UNUSED_VAR (lmem_bus_if.rsp_data.tag)
`endif

`ifndef VX_CFG_EXT_MBAR_ENABLE
    wire [SLOT_W-1:0] head_slot;
    wire              cpl_empty, cpl_full;
`else
    wire              cpl_full;
`endif
    wire              rsp_buf_ready;
`ifdef DXA_MBAR_ENABLE
    // A completion's slot frees in turn without a response; an async slot's
    // ack waits for its complete_tx.
    reg [NUM_SLOTS-1:0] posted_r;
    wire              cx_ok    = ~async_r[head_slot] || cx_sent_r[head_slot] || cx_fire;
    wire              rsp_send = ~cpl_empty && ~posted_r[head_slot] && cx_ok;
    wire              rsp_fire = ~cpl_empty && (posted_r[head_slot] || (rsp_buf_ready && cx_ok));
`elsif VX_CFG_EXT_MBAR_ENABLE
    wire              cx_ok    = ~async_r[head_slot] || cx_sent_r[head_slot] || cx_fire;
    wire              rsp_send = ~cpl_empty && cx_ok;
    wire              rsp_fire = rsp_send && rsp_buf_ready;
`else
    wire              rsp_fire = ~cpl_empty && rsp_buf_ready;
`endif

    VX_fifo_queue #(
        .DATAW (SLOT_W),
        .DEPTH (NUM_SLOTS)
    ) cpl_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (lmem_rsp_fire),
        .pop      (rsp_fire),
        .data_in  (rsp_slot),
        .data_out (head_slot),
        .empty    (cpl_empty),
        .full     (cpl_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    for (genvar s = 0; s < NUM_SLOTS; ++s) begin : g_slots
        wire alloc_s = req_fire && (free_idx == SLOT_W'(s));
        wire done_s  = lmem_rsp_fire && (rsp_slot == SLOT_W'(s));
        wire free_s  = rsp_fire && (head_slot == SLOT_W'(s));
        always @(posedge clk) begin
            if (reset) begin
                busy_r[s] <= 1'b0;
                done_r[s] <= 1'b0;
            end else if (alloc_s) begin
                busy_r[s] <= 1'b1;
            end else if (done_s) begin
                done_r[s] <= 1'b1;
            end else if (free_s) begin
                busy_r[s] <= 1'b0;
                done_r[s] <= 1'b0;
            end
            if (alloc_s) begin
            `ifdef DXA_MBAR_ENABLE
                posted_r[s] <= is_compl;
            `endif
            `ifdef VX_CFG_EXT_MBAR_ENABLE
                async_r[s]     <= req_pkt.mbar_valid;
                mbar_addr_r[s] <= req_pkt.mbar_addr;
                bytes_r[s]     <= `CLOG2(LSU_WORD_SIZE+1)'($countones(req_pkt.byteen));
                ack_r[s]       <= (req_pkt.op == DLMEM_OP_ST) || req_pkt.mbar_valid;
            `else
                ack_r[s] <= (req_pkt.op == DLMEM_OP_ST);
            `endif
                src_r[s] <= req_pkt.src;
                tag_r[s] <= req_pkt.tag;
            end
        `ifdef VX_CFG_EXT_MBAR_ENABLE
            if (alloc_s) begin
                cx_sent_r[s] <= 1'b0;
            end else if (cx_fire && (head_slot == SLOT_W'(s))) begin
                cx_sent_r[s] <= 1'b1;
            end
        `endif
            if (done_s) begin
                data_r[s] <= lmem_bus_if.rsp_data.data;
            end
        end
    end

    // Inject completed responses back to their requesters.

    dlmem_rsp_pkt_t rsp_pkt;
    always @(*) begin
        rsp_pkt      = '0;
        rsp_pkt.ack  = ack_r[head_slot];
        rsp_pkt.tag  = tag_r[head_slot];
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        rsp_pkt.data = ack_r[head_slot] ? `VX_CFG_XLEN'(async_r[head_slot]) : data_r[head_slot];
    `else
        rsp_pkt.data = data_r[head_slot];
    `endif
    end

    VX_elastic_buffer #(
        .DATAW   (RSP_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        .valid_in  (rsp_send),
    `else
        .valid_in  (~cpl_empty),
    `endif
        .ready_in  (rsp_buf_ready),
        .data_in   ({src_r[head_slot], rsp_pkt}),
        .data_out  ({rsp_out_if.dst, rsp_out_if.data}),
        .valid_out (rsp_out_if.valid),
        .ready_out (rsp_out_if.ready)
    );

    assign empty = ~(|busy_r) && ~lmem_bus_if.req_valid && ~rsp_out_if.valid;

    `RUNTIME_ASSERT (~(lmem_rsp_fire && ((32'(rsp_slot) >= NUM_SLOTS) || ~busy_r[rsp_slot] || done_r[rsp_slot])),
        ("%t: *** %s local-memory response for a free or completed slot: slot=%0d", $time, INSTANCE_ID, rsp_slot))
    `RUNTIME_ASSERT (~(lmem_rsp_fire && cpl_full),
        ("%t: *** %s completion queue overflow", $time, INSTANCE_ID))
    `RUNTIME_ASSERT (~(rsp_fire && ~done_r[head_slot]),
        ("%t: *** %s response for an incomplete slot: slot=%0d", $time, INSTANCE_ID, head_slot))
`ifdef VX_CFG_EXT_MBAR_ENABLE
`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (~reset && req_fire && req_pkt.mbar_valid) begin
            `TRACE(2, ("%t: %s async-acc: src=%0d, addr=0x%0h, mbar=0x%0h, red=%0d, slot=%0d\n", $time, INSTANCE_ID,
                req_pkt.src, req_pkt.addr, req_pkt.mbar_addr, (req_pkt.op == DLMEM_OP_AMO), free_idx))
        end
        if (~reset && cx_fire) begin
            `TRACE(2, ("%t: %s async-cx: mbar=0x%0h, bytes=%0d, slot=%0d\n", $time, INSTANCE_ID,
                mbar_addr_r[head_slot], bytes_r[head_slot], head_slot))
        end
        if (~reset && rsp_fire && async_r[head_slot]) begin
            `TRACE(2, ("%t: %s async-ack: dst=%0d, slot=%0d\n", $time, INSTANCE_ID, src_r[head_slot], head_slot))
        end
    end
`endif
`endif
`ifdef DXA_MBAR_ENABLE
    wire row_fire = req_in_if.valid && req_in_if.ready && is_row;
    wire [NW_WIDTH-1:0] row_slot = NW_WIDTH'(req_pkt.tag);
    `RUNTIME_ASSERT (~(row_fire && ~slot_valid[row_slot]),
        ("%t: *** %s DXA write into a retired CTA slot: slot=%0d %s from core %0d, addr=0x%0h",
         $time, INSTANCE_ID, row_slot, (is_compl ? "completion" : "data"), req_pkt.src, req_pkt.addr))

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && row_fire && is_data) begin
            $write("DXA_TL,%0d,ROW_DATA_IN,%s,src=%0d,addr=0x%0h,byteen=0x%0h,slot=%0d\n",
                $time, INSTANCE_ID, req_pkt.src, req_pkt.addr, req_pkt.byteen, row_slot);
        end
        if (~reset && row_fire && is_compl) begin
            $write("DXA_TL,%0d,ROW_COMPL_IN,%s,src=%0d,off=0x%0h,bytes=%0d,slot=%0d\n",
                $time, INSTANCE_ID, req_pkt.src, req_pkt.addr, req_pkt.data, row_slot);
        end
    end
`endif
`else
    `RUNTIME_ASSERT (~(req_in_if.valid && (req_pkt.op == 2'd3)),
        ("%t: *** %s invalid request opcode", $time, INSTANCE_ID))
`endif

endmodule
