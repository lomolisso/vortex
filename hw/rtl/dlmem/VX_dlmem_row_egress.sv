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

`ifdef DXA_MBAR_ENABLE

// The socket's DXA row egress: turns each local-memory beat of a remote copy
// into posted DLMEM flits, one data word per bank with a nonzero byte mask, in
// bank order, one flit per cycle. After the copy's last beat it sends one
// completion flit carrying the mbarrier word offset and the box bytes. The
// flits of one socket follow one path to each core, so they arrive in order.
module VX_dlmem_row_egress import VX_gpu_pkg::*, VX_dlmem_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter TAG_WIDTH = 1,
    parameter OUT_BUF   = 3
) (
    input wire              clk,
    input wire              reset,

    // A beat of a remote copy: the destination's DMA word address, all banks.
    input  wire                                beat_valid,
    input  wire [LMEM_DMA_ADDR_WIDTH-1:0]      beat_addr,
    input  wire [LMEM_DMA_DATA_SIZE*8-1:0]     beat_data,
    input  wire [LMEM_DMA_DATA_SIZE-1:0]       beat_byteen,
    input  dxa_lmem_attr_t                     beat_attr,
    input  wire [TAG_WIDTH-1:0]                beat_tag,
    output wire                                beat_ready,

    VX_dlmem_bus_if.master                     row_out_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam NUM_BANKS = `VX_CFG_LMEM_NUM_BANKS;
    localparam BANK_BITS = `CLOG2(NUM_BANKS);
    localparam BANK_W    = `UP(BANK_BITS);
    localparam WORD_W    = LSU_WORD_SIZE * 8;
    localparam OUT_W     = DLMEM_CORE_W + DLMEM_REQ_DATAW;

    `STATIC_ASSERT (LMEM_DMA_DATA_SIZE == NUM_BANKS * LSU_WORD_SIZE, ("invalid parameter"))
    `STATIC_ASSERT (DXA_MBAR_OFF_W == DLMEM_LMEM_AW, ("an mbarrier offset is a DLMEM word address"))
    `STATIC_ASSERT (DXA_TX_BYTES_W <= `VX_CFG_XLEN, ("invalid parameter"))

    wire [NUM_BANKS-1:0] word_nz;
    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_word_nz
        assign word_nz[b] = (| beat_byteen[b * LSU_WORD_SIZE +: LSU_WORD_SIZE]);
    end

    // Banks of the head beat already sent.
    reg [NUM_BANKS-1:0] sent_r;
    wire [NUM_BANKS-1:0] rem = word_nz & ~sent_r;
    wire has_data = (| rem);

    wire [NUM_BANKS-1:0] bank_onehot;
    wire [BANK_W-1:0]    bank_idx;

    VX_priority_encoder #(
        .N (NUM_BANKS)
    ) bank_sel (
        .data_in    (rem),
        .onehot_out (bank_onehot),
        .index_out  (bank_idx),
        `UNUSED_PIN (valid_out)
    );

    wire last_data = (rem == bank_onehot);

    // The issuing core, the socket's local-memory route of the beat.
    wire [DLMEM_CORE_W-1:0] src_core;
    if (NC_BITS != 0) begin : g_src
        assign src_core = DLMEM_CORE_W'(beat_tag[1 +: NC_BITS]);
    end else begin : g_src0
        assign src_core = '0;
    end

    dlmem_req_pkt_t flit;
    always @(*) begin
        flit         = '0;
        flit.src     = src_core;
        flit.op      = DLMEM_OP_DXA;
        flit.tag     = DLMEM_TAG_W'(beat_attr.slot);
        if (has_data) begin
            flit.addr   = (DLMEM_LMEM_AW'(beat_addr) << BANK_BITS) | DLMEM_LMEM_AW'(bank_idx);
            flit.byteen = beat_byteen[bank_idx * LSU_WORD_SIZE +: LSU_WORD_SIZE];
            flit.data   = beat_data[bank_idx * WORD_W +: WORD_W];
        end else begin
            flit.addr         = DLMEM_LMEM_AW'(beat_attr.payload[DXA_MBAR_OFF_W-1:0]);
            flit.byteen       = '1;
            flit.data         = `VX_CFG_XLEN'(beat_attr.tx_bytes);
            flit.amo_op       = AMO_OP_MBAR_COMPLETE_TX;
            flit.amo_unsigned = 1'b1;
        end
    end

    wire flit_valid = beat_valid && (has_data || beat_attr.last);
    wire flit_ready;
    wire flit_fire  = flit_valid && flit_ready;

    // A beat leaves with its last flit; a beat without data or completion
    // carries nothing to send.
    assign beat_ready = beat_valid && (flit_fire ? (has_data ? (last_data && ~beat_attr.last) : 1'b1)
                                                 : ~flit_valid);

    always @(posedge clk) begin
        if (reset) begin
            sent_r <= '0;
        end else if (beat_ready) begin
            sent_r <= '0;
        end else if (flit_fire && has_data) begin
            sent_r <= sent_r | bank_onehot;
        end
    end

    VX_elastic_buffer #(
        .DATAW   (OUT_W),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (flit_valid),
        .ready_in  (flit_ready),
        .data_in   ({beat_attr.dst_core, flit}),
        .data_out  ({row_out_if.dst, row_out_if.data}),
        .valid_out (row_out_if.valid),
        .ready_out (row_out_if.ready)
    );

    `UNUSED_VAR (beat_attr)
    `UNUSED_VAR (beat_tag)

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && flit_fire) begin
            if (has_data) begin
                $write("DXA_TL,%0d,ROW_FLIT,%s,dst=%0d,slot=%0d,addr=0x%0h,byteen=0x%0h\n",
                    $time, INSTANCE_ID, beat_attr.dst_core, beat_attr.slot, flit.addr, flit.byteen);
            end else begin
                $write("DXA_TL,%0d,ROW_COMPL,%s,dst=%0d,slot=%0d,off=0x%0h,bytes=%0d\n",
                    $time, INSTANCE_ID, beat_attr.dst_core, beat_attr.slot, flit.addr, beat_attr.tx_bytes);
            end
        end
    end
`endif

endmodule

`endif
