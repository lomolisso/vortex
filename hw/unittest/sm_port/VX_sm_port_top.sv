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

// One SM port with its two L1TEX sources (the L1 and the TMA engine), a
// line-wide memory side and, in DLMEM builds, its SM-to-SM legs. Every
// payload is a pattern of its tag, so the *_ok outputs prove that the
// deserialiser rebuilt each request's address, enabled bytes and message bits.
module VX_sm_port_top import VX_gpu_pkg::*;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    import VX_dlmem_pkg::*;
`endif
#(
    parameter NUM_SRCS  = 2,
    parameter LINE_SIZE = SM_PORT_LINE_SIZE,

    localparam TAGV          = 8,
    localparam SRC_TAG_WIDTH = UUID_WIDTH + TAGV,
    localparam SECTORS       = LINE_SIZE / `VX_CFG_SM_EGRESS_BEAT,
    localparam SECT_W        = `CLOG2(SECTORS + 1),
    localparam SRC_W         = `LOG2UP(NUM_SRCS),
    localparam MEM_TAG_W     = SRC_TAG_WIDTH + SECT_W + SRC_W,
    localparam LADDR_W       = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(LINE_SIZE)
) (
    input  wire                         clk,
    input  wire                         reset,

    input  wire [NUM_SRCS-1:0]          src_valid,
    input  wire [NUM_SRCS-1:0]          src_rw,
    input  wire [NUM_SRCS-1:0]          src_amo,
    input  wire [NUM_SRCS-1:0][LINE_SIZE-1:0] src_byteen,
    input  wire [NUM_SRCS-1:0][TAGV-1:0] src_tag,
    output wire [NUM_SRCS-1:0]          src_ready,

    output wire [NUM_SRCS-1:0]          src_rsp_valid,
    output wire [NUM_SRCS-1:0][TAGV-1:0] src_rsp_tag,
    input  wire [NUM_SRCS-1:0]          src_rsp_ready,

    output wire                         mem_req_valid,
    output wire                         mem_req_ok,
    output wire                         mem_req_rw,
    output wire [LINE_SIZE-1:0]         mem_req_byteen,
    output wire [MEM_TAG_W-1:0]         mem_req_tag,
    input  wire                         mem_req_ready,

    input  wire                         mem_rsp_valid,
    input  wire [MEM_TAG_W-1:0]         mem_rsp_tag,
    output wire                         mem_rsp_ready,

    // SM-to-SM legs (DLMEM builds; tied off otherwise).
    input  wire                         dsm_req_in_valid,
    input  wire [TAGV-1:0]              dsm_req_in_tag,
    output wire                         dsm_req_in_ready,
    output wire                         dsm_req_out_valid,
    output wire                         dsm_req_out_ok,
    output wire [TAGV-1:0]              dsm_req_out_tag,
    input  wire                         dsm_req_out_ready,

    input  wire                         dsm_rsp_in_valid,
    input  wire [TAGV-1:0]              dsm_rsp_in_tag,
    output wire                         dsm_rsp_in_ready,
    output wire                         dsm_rsp_out_valid,
    output wire                         dsm_rsp_out_ok,
    output wire [TAGV-1:0]              dsm_rsp_out_tag,
    input  wire                         dsm_rsp_out_ready,

    input  wire                         arr_req_in_valid,
    output wire                         arr_req_in_ready,
    output wire                         arr_req_out_valid,
    input  wire                         arr_req_out_ready,

    input  wire                         arr_rsp_in_valid,
    output wire                         arr_rsp_in_ready,
    output wire                         arr_rsp_out_valid,
    input  wire                         arr_rsp_out_ready,

    output wire                         empty
);
    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (SRC_TAG_WIDTH)
    ) src_bus_if[NUM_SRCS]();

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (MEM_TAG_W)
    ) mem_bus_if();

    for (genvar i = 0; i < NUM_SRCS; ++i) begin : g_src
        mem_bus_attr_t attr;
        always @(*) begin
            attr = '0;
            attr.amo.amo_valid = src_amo[i];
            attr.amo.amo_op    = AMO_OP_ADD;
        end
        assign src_bus_if[i].req_valid       = src_valid[i];
        assign src_bus_if[i].req_data.rw     = src_rw[i];
        assign src_bus_if[i].req_data.byteen = src_byteen[i];
        assign src_bus_if[i].req_data.addr   = LADDR_W'(src_tag[i]);
        for (genvar b = 0; b < LINE_SIZE; ++b) begin : g_data
            assign src_bus_if[i].req_data.data[b * 8 +: 8] = src_tag[i] ^ 8'(b);
        end
        assign src_bus_if[i].req_data.attr   = attr;
        assign src_bus_if[i].req_data.tag    = SRC_TAG_WIDTH'(src_tag[i]);
        assign src_ready[i] = src_bus_if[i].req_ready;

        assign src_rsp_valid[i] = src_bus_if[i].rsp_valid;
        assign src_rsp_tag[i]   = src_bus_if[i].rsp_data.tag.value;
        assign src_bus_if[i].rsp_ready = src_rsp_ready[i];
        `UNUSED_VAR (src_bus_if[i].rsp_data)
    end

    assign mem_req_valid  = mem_bus_if.req_valid;
    assign mem_req_rw     = mem_bus_if.req_data.rw;
    assign mem_req_byteen = mem_bus_if.req_data.byteen;
    assign mem_req_tag    = mem_bus_if.req_data.tag;
    assign mem_bus_if.req_ready = mem_req_ready;
    wire [TAGV-1:0] mem_tag_v = mem_bus_if.req_data.tag[TAGV-1:0];
    mem_bus_attr_t mem_attr;
    assign mem_attr = mem_bus_attr_t'(mem_bus_if.req_data.attr);
    `UNUSED_VAR (mem_attr)
    // A read or fence names its sectors only; a write or atomic carries data.
    wire mem_has_data = mem_bus_if.req_data.rw || mem_attr.amo.amo_valid;
    reg mem_ok;
    always @(*) begin
        mem_ok = (mem_bus_if.req_data.addr == LADDR_W'(mem_tag_v));
        for (integer b = 0; b < LINE_SIZE; ++b) begin
            if (mem_has_data && mem_bus_if.req_data.byteen[b] && (mem_bus_if.req_data.data[b * 8 +: 8] != (mem_tag_v ^ 8'(b)))) begin
                mem_ok = 1'b0;
            end
        end
    end
    assign mem_req_ok = mem_ok;

    assign mem_bus_if.rsp_valid     = mem_rsp_valid;
    assign mem_bus_if.rsp_data.data = '0;
    assign mem_bus_if.rsp_data.tag  = mem_rsp_tag;
    assign mem_rsp_ready = mem_bus_if.rsp_ready;

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    VX_dlmem_bus_if #(.DATAW (DLMEM_REQ_DATAW), .DST_W (DLMEM_CORE_W)) dsm_req_in_if();
    VX_dlmem_bus_if #(.DATAW (DLMEM_REQ_DATAW), .DST_W (DLMEM_CORE_W)) dsm_req_out_if();
    VX_dlmem_bus_if #(.DATAW (DLMEM_RSP_DATAW), .DST_W (DLMEM_CORE_W)) dsm_rsp_in_if();
    VX_dlmem_bus_if #(.DATAW (DLMEM_RSP_DATAW), .DST_W (DLMEM_CORE_W)) dsm_rsp_out_if();
    VX_dlmem_bus_if #(.DATAW (DLMEM_REQ_DATAW), .DST_W (DLMEM_CORE_W)) arr_req_in_if();
    VX_dlmem_bus_if #(.DATAW (DLMEM_REQ_DATAW), .DST_W (DLMEM_CORE_W)) arr_req_out_if();
    VX_dlmem_bus_if #(.DATAW (DLMEM_RSP_DATAW), .DST_W (DLMEM_CORE_W)) arr_rsp_in_if();
    VX_dlmem_bus_if #(.DATAW (DLMEM_RSP_DATAW), .DST_W (DLMEM_CORE_W)) arr_rsp_out_if();

    dlmem_req_pkt_t dreq, areq, dreq_out;
    dlmem_rsp_pkt_t drsp, arsp, drsp_out;
    always @(*) begin
        dreq = '0;
        dreq.tag = DLMEM_TAG_W'(dsm_req_in_tag);
        dreq.addr = DLMEM_LMEM_AW'(~dreq.tag);
        dreq.data = `VX_CFG_XLEN'({(`VX_CFG_XLEN / DLMEM_TAG_W + 1){dreq.tag}});
        dreq.src = DLMEM_CORE_W'(dreq.tag);
        areq = '0;
        drsp = '0;
        drsp.tag = DLMEM_TAG_W'(dsm_rsp_in_tag);
        drsp.data = `VX_CFG_XLEN'({(`VX_CFG_XLEN / DLMEM_TAG_W + 1){drsp.tag}});
        drsp.ack = drsp.tag[0];
        arsp = '0;
    end

    assign dsm_req_in_if.valid = dsm_req_in_valid;
    assign dsm_req_in_if.data  = dreq;
    assign dsm_req_in_if.dst   = '0;
    assign dsm_req_in_ready    = dsm_req_in_if.ready;
    assign dreq_out            = dlmem_req_pkt_t'(dsm_req_out_if.data);
    assign dsm_req_out_valid   = dsm_req_out_if.valid;
    assign dsm_req_out_tag     = TAGV'(dreq_out.tag);
    assign dsm_req_out_ok      = (dreq_out.addr == DLMEM_LMEM_AW'(~dreq_out.tag))
                              && (dreq_out.data == `VX_CFG_XLEN'({(`VX_CFG_XLEN / DLMEM_TAG_W + 1){dreq_out.tag}}))
                              && (dreq_out.src == DLMEM_CORE_W'(dreq_out.tag));
    assign dsm_req_out_if.ready = dsm_req_out_ready;

    assign dsm_rsp_in_if.valid = dsm_rsp_in_valid;
    assign dsm_rsp_in_if.data  = drsp;
    assign dsm_rsp_in_if.dst   = '0;
    assign dsm_rsp_in_ready    = dsm_rsp_in_if.ready;
    assign drsp_out            = dlmem_rsp_pkt_t'(dsm_rsp_out_if.data);
    assign dsm_rsp_out_valid   = dsm_rsp_out_if.valid;
    assign dsm_rsp_out_tag     = TAGV'(drsp_out.tag);
    assign dsm_rsp_out_ok      = (drsp_out.data == `VX_CFG_XLEN'({(`VX_CFG_XLEN / DLMEM_TAG_W + 1){drsp_out.tag}}))
                              && (drsp_out.ack == drsp_out.tag[0]);
    assign dsm_rsp_out_if.ready = dsm_rsp_out_ready;

    assign arr_req_in_if.valid = arr_req_in_valid;
    assign arr_req_in_if.data  = areq;
    assign arr_req_in_if.dst   = '0;
    assign arr_req_in_ready    = arr_req_in_if.ready;
    assign arr_req_out_valid   = arr_req_out_if.valid;
    assign arr_req_out_if.ready = arr_req_out_ready;

    assign arr_rsp_in_if.valid = arr_rsp_in_valid;
    assign arr_rsp_in_if.data  = arsp;
    assign arr_rsp_in_if.dst   = '0;
    assign arr_rsp_in_ready    = arr_rsp_in_if.ready;
    assign arr_rsp_out_valid   = arr_rsp_out_if.valid;
    assign arr_rsp_out_if.ready = arr_rsp_out_ready;

    // The packets' tag fields may be narrower than the bench's ports.
    `UNUSED_VAR (dsm_req_in_tag)
    `UNUSED_VAR (dsm_rsp_in_tag)
    `UNUSED_VAR (dreq_out)
    `UNUSED_VAR (drsp_out)
    `UNUSED_VAR (dsm_req_out_if.dst)
    `UNUSED_VAR (dsm_rsp_out_if.dst)
    `UNUSED_VAR (arr_req_out_if.data)
    `UNUSED_VAR (arr_req_out_if.dst)
    `UNUSED_VAR (arr_rsp_out_if.data)
    `UNUSED_VAR (arr_rsp_out_if.dst)
`else
    assign dsm_req_in_ready  = 1'b0;
    assign dsm_req_out_valid = 1'b0;
    assign dsm_req_out_tag   = '0;
    assign dsm_rsp_in_ready  = 1'b0;
    assign dsm_rsp_out_valid = 1'b0;
    assign dsm_rsp_out_tag   = '0;
    assign dsm_req_out_ok    = 1'b0;
    assign dsm_rsp_out_ok    = 1'b0;
    assign arr_req_in_ready  = 1'b0;
    assign arr_req_out_valid = 1'b0;
    assign arr_rsp_in_ready  = 1'b0;
    assign arr_rsp_out_valid = 1'b0;
    `UNUSED_VAR ({dsm_req_in_valid, dsm_req_in_tag, dsm_req_out_ready, dsm_rsp_in_valid,
                  dsm_rsp_in_tag, dsm_rsp_out_ready, arr_req_in_valid, arr_req_out_ready,
                  arr_rsp_in_valid, arr_rsp_out_ready})
`endif

    VX_sm_port #(
        .INSTANCE_ID   ("sm_port"),
        .NUM_SRCS      (NUM_SRCS),
        .SRC_TAG_WIDTH (SRC_TAG_WIDTH),
        .LINE_SIZE     (LINE_SIZE)
    ) port (
        .clk            (clk),
        .reset          (reset),
        .src_bus_if     (src_bus_if),
        .mem_bus_if     (mem_bus_if),
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        .dsm_req_in_if  (dsm_req_in_if),
        .dsm_req_out_if (dsm_req_out_if),
        .dsm_rsp_in_if  (dsm_rsp_in_if),
        .dsm_rsp_out_if (dsm_rsp_out_if),
        .arr_req_in_if  (arr_req_in_if),
        .arr_req_out_if (arr_req_out_if),
        .arr_rsp_in_if  (arr_rsp_in_if),
        .arr_rsp_out_if (arr_rsp_out_if),
    `endif
        .empty          (empty)
    );

endmodule
