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

    output wire             empty
);
    localparam SLOT_W     = `LOG2UP(NUM_SLOTS);
    localparam WORD_W     = LSU_WORD_SIZE * 8;
    localparam LMEM_DATAW = 1 + DLMEM_LMEM_AW + WORD_W + LSU_WORD_SIZE + MEM_ATTR_WIDTH + LSU_TAG_WIDTH;
    localparam RSP_DATAW  = DLMEM_CORE_W + DLMEM_RSP_DATAW;

    `STATIC_ASSERT (NUM_SLOTS >= 1, ("invalid parameter"))
    `STATIC_ASSERT (SLOT_W <= LSU_TAG_WIDTH - UUID_WIDTH, ("dlmem ingress: local-memory tag too narrow"))

    reg [NUM_SLOTS-1:0]                    busy_r;
    reg [NUM_SLOTS-1:0]                    done_r;
    reg [NUM_SLOTS-1:0]                    ack_r;
    reg [NUM_SLOTS-1:0][DLMEM_CORE_W-1:0]  src_r;
    reg [NUM_SLOTS-1:0][DLMEM_TAG_W-1:0]   tag_r;
    reg [NUM_SLOTS-1:0][WORD_W-1:0]        data_r;

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

    wire lmem_req_valid = req_in_if.valid && free_valid;
    wire lmem_req_ready;
    wire req_fire = lmem_req_valid && lmem_req_ready;

    assign req_in_if.ready = free_valid && lmem_req_ready;
    `UNUSED_VAR (req_in_if.dst)

    mem_bus_attr_t lmem_attr;
    always @(*) begin
        lmem_attr                  = '0;
        lmem_attr.is_addr_local    = 1'b1;
        lmem_attr.amo.amo_valid    = (req_pkt.op == DLMEM_OP_AMO);
        lmem_attr.amo.amo_op       = req_pkt.amo_op;
        lmem_attr.amo.amo_unsigned = req_pkt.amo_unsigned;
        lmem_attr.amo.hart_id      = req_pkt.hart_id;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        lmem_attr.amo.amo_cmp      = req_pkt.amo_cmp;
    `endif
    end

    wire [LSU_TAG_WIDTH-1:0] lmem_tag = {UUID_WIDTH'(0), (LSU_TAG_WIDTH-UUID_WIDTH)'(free_idx)};

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

    wire [SLOT_W-1:0] head_slot;
    wire              cpl_empty, cpl_full;
    wire              rsp_buf_ready;
    wire              rsp_fire = ~cpl_empty && rsp_buf_ready;

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
                ack_r[s] <= (req_pkt.op == DLMEM_OP_ST);
                src_r[s] <= req_pkt.src;
                tag_r[s] <= req_pkt.tag;
            end
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
        rsp_pkt.data = data_r[head_slot];
    end

    VX_elastic_buffer #(
        .DATAW   (RSP_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (~cpl_empty),
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
    `RUNTIME_ASSERT (~(req_in_if.valid && (req_pkt.op == 2'd3)),
        ("%t: *** %s invalid request opcode", $time, INSTANCE_ID))

endmodule
