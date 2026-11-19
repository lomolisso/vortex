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

`ifdef VX_CFG_H100_SM_ENABLE

`include "VX_define.vh"

// One SM's ports to the GPC, cycle for cycle sm_port_ref::SmEgressRef and
// SmIngressRef.
//
// Egress: every request-class packet (the L1TEX sources' line requests and the
// SM's own SM-to-SM messages) enters one in-order FIFO, one per cycle round
// robin over the sources; the SM-to-SM replies this SM serves wait in a
// response queue. The two heads share one link of EGRESS_BYTES / FLIT flits
// per cycle, round robin per flit. A packet's header rides in the link's side
// field with its first flit: a write then sends its touched sectors and, when
// one is partial, the line's byte mask as one more flit; any other packet is
// one flit (a read's or a fence's header, an atomic's operand sector, an
// SM-to-SM message). The deserialiser at the far end rebuilds the request;
// a packet whose last flit is sent leaves once its sink is ready, and a head
// that cannot leave holds its queue.
//
// Ingress: L2 replies pass their input register, then the ingress gate
// with the SM-to-SM arrivals (VX_sm_ingress) and return to their source.
module VX_sm_port import VX_gpu_pkg::*;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    import VX_dlmem_pkg::*;
`endif
#(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_SRCS      = 1,    // L1TEX sources: 0 the L1, 1 the TMA engine
    parameter SRC_TAG_WIDTH = UUID_WIDTH + 1,
    parameter LINE_SIZE     = SM_PORT_LINE_SIZE
) (
    input wire              clk,
    input wire              reset,

    VX_mem_bus_if.slave     src_bus_if [NUM_SRCS],
    VX_mem_bus_if.master    mem_bus_if,

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    VX_dlmem_bus_if.slave   dsm_req_in_if,   // this SM's SM-to-SM requests
    VX_dlmem_bus_if.master  dsm_req_out_if,
    VX_dlmem_bus_if.slave   dsm_rsp_in_if,   // replies this SM serves
    VX_dlmem_bus_if.master  dsm_rsp_out_if,
    VX_dlmem_bus_if.slave   arr_req_in_if,   // requests arriving for this SM
    VX_dlmem_bus_if.master  arr_req_out_if,
    VX_dlmem_bus_if.slave   arr_rsp_in_if,   // replies arriving for this SM
    VX_dlmem_bus_if.master  arr_rsp_out_if,
`endif

    output wire             empty
);
    localparam FLIT       = `VX_CFG_SM_EGRESS_BEAT;
    localparam FLIT_BITS  = FLIT * 8;
    localparam SECTORS    = LINE_SIZE / FLIT;
    localparam SECT_W     = `CLOG2(SECTORS + 1);
    localparam SIDX_W     = `LOG2UP(SECTORS);
    localparam UNITS      = `VX_CFG_SM_EGRESS_BYTES / FLIT;
    localparam ING_UNITS  = `VX_CFG_SM_INGRESS_BYTES / FLIT;
    localparam EJECT      = `VX_CFG_SM2SM_EJECT_BYTES / FLIT;
    localparam SRC_W      = `LOG2UP(NUM_SRCS);
    localparam LADDR_W    = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(LINE_SIZE);
    localparam MEM_TAG_W  = SRC_TAG_WIDTH + SECT_W + SRC_W;
    localparam CNT_W      = `CLOG2(SECTORS + 2);
    // The serialiser's skid: the packet on the link and the next one, so the
    // link never idles on its own ready loop.
    localparam REQ_DEPTH  = 2;
    // A reply leaves its ingress slot when it enters this queue, so no more
    // than the ingress slots can wait.
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    localparam RSP_DEPTH  = DLMEM_INGRESS_SLOTS;
`else
    localparam RSP_DEPTH  = 1;
`endif
    // An L2 packet's header: rw, line address, attributes, tag, the sectors it
    // names, whether a byte-mask flit follows, and an atomic's operand bytes.
    localparam L2_HDR_W   = 1 + LADDR_W + MEM_ATTR_WIDTH + MEM_TAG_W + SECTORS + 1 + FLIT;
    localparam L2_PAYLOAD = 1 + LADDR_W + LINE_SIZE * 8 + LINE_SIZE + MEM_ATTR_WIDTH + MEM_TAG_W;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    localparam DSM_PAYLOAD = DLMEM_CORE_W + DLMEM_REQ_DATAW;
    localparam RSP_PAYLOAD = DLMEM_CORE_W + DLMEM_RSP_DATAW;
    localparam NUM_RSRCS   = NUM_SRCS + 1;
`else
    localparam DSM_PAYLOAD = 1;
    localparam RSP_PAYLOAD = 1;
    localparam NUM_RSRCS   = NUM_SRCS;
`endif
    localparam REQ_PAYLOAD = `MAX(L2_PAYLOAD, DSM_PAYLOAD);
    // An SM-to-SM message fills one flit and its side field.
    localparam HDR_W       = `MAX(L2_HDR_W, `MAX(DSM_PAYLOAD, RSP_PAYLOAD) - FLIT_BITS);
    localparam LINK_W      = HDR_W + FLIT_BITS;

`ifndef DBG_TRACE_MEM
    `UNUSED_SPARAM (INSTANCE_ID)
`endif

    `STATIC_ASSERT (FLIT == `VX_CFG_DCACHE_SECTOR_SIZE, ("sm port: the flit must be the L1 sector"))
    `STATIC_ASSERT (UNITS >= 1 && `VX_CFG_SM_EGRESS_BYTES % FLIT == 0 && `VX_CFG_SM_INGRESS_BYTES % FLIT == 0
                 && `VX_CFG_SM2SM_EJECT_BYTES % FLIT == 0,
        ("sm port: the egress, ejection and ingress must be whole flits"))
    `STATIC_ASSERT (LINE_SIZE % FLIT == 0, ("sm port: the line must be whole sectors"))
    `STATIC_ASSERT (LINE_SIZE <= FLIT_BITS, ("sm port: a line's byte mask must fit one flit"))
    `STATIC_ASSERT (`VX_CFG_SM2SM_EJECT_BYTES <= `VX_CFG_SM_INGRESS_BYTES, ("sm port: ejection above the ingress"))
    `STATIC_ASSERT (NUM_SRCS >= 1 && NUM_SRCS <= 2, ("sm port: one L1 and at most one TMA source"))

    ///////////////////////////////////////////////////////////////////////////
    // Request-class sources.

    wire [NUM_RSRCS-1:0]                  rs_valid;
    wire [NUM_RSRCS-1:0]                  rs_l2;
    wire [NUM_RSRCS-1:0][REQ_PAYLOAD-1:0] rs_payload;
    wire [NUM_RSRCS-1:0]                  rs_ready;

    for (genvar i = 0; i < NUM_SRCS; ++i) begin : g_src
        wire [LINE_SIZE-1:0] be = src_bus_if[i].req_data.byteen;
        mem_bus_attr_t attr;
        assign attr = mem_bus_attr_t'(src_bus_if[i].req_data.attr);
        `UNUSED_VAR (attr)

        wire [SECTORS-1:0] touched;
        for (genvar s = 0; s < SECTORS; ++s) begin : g_sect
            assign touched[s] = (| be[s * FLIT +: FLIT]);
        end
        wire [SECT_W-1:0] n_touched;
        `POP_COUNT(n_touched, touched);

        wire is_rd = ~src_bus_if[i].req_data.rw && ~attr.is_fence && ~attr.amo.amo_valid;

        // Sectors the reply brings back (0: no reply, or one that carries no fill).
        wire [SECT_W-1:0] rsp_sect = is_rd ? ((n_touched != '0) ? n_touched : SECT_W'(SECTORS))
                                   : (attr.amo.amo_valid ? SECT_W'(1) : '0);

        wire [MEM_TAG_W-1:0] out_tag = {src_bus_if[i].req_data.tag.uuid, rsp_sect, SRC_W'(i),
                                        src_bus_if[i].req_data.tag.value};

        assign rs_valid[i]   = src_bus_if[i].req_valid;
        assign rs_l2[i]      = 1'b1;
        assign rs_payload[i] = REQ_PAYLOAD'({src_bus_if[i].req_data.rw,
                                             src_bus_if[i].req_data.addr,
                                             src_bus_if[i].req_data.data,
                                             src_bus_if[i].req_data.byteen,
                                             src_bus_if[i].req_data.attr,
                                             out_tag});
        assign src_bus_if[i].req_ready = rs_ready[i];
    end

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    assign rs_valid[NUM_SRCS]   = dsm_req_in_if.valid;
    assign rs_l2[NUM_SRCS]      = 1'b0;
    assign rs_payload[NUM_SRCS] = REQ_PAYLOAD'({dsm_req_in_if.dst, dsm_req_in_if.data});
    assign dsm_req_in_if.ready  = rs_ready[NUM_SRCS];
`endif

    ///////////////////////////////////////////////////////////////////////////
    // Admission: one push per cycle, round robin over the sources.

    wire [`CLOG2(REQ_DEPTH+1)-1:0] req_size;
    wire req_room = (32'(req_size) < REQ_DEPTH);

    reg [`LOG2UP(NUM_RSRCS)-1:0] rr_last_r;
    reg [`LOG2UP(NUM_RSRCS)-1:0] push_idx;
    reg                          push_valid;
    always @(*) begin
        push_valid = 1'b0;
        push_idx   = '0;
        for (integer k = NUM_RSRCS; k >= 1; --k) begin
            if (rs_valid[(32'(rr_last_r) + k) % NUM_RSRCS] && req_room) begin
                push_valid = 1'b1;
                push_idx   = `LOG2UP(NUM_RSRCS)'((32'(rr_last_r) + k) % NUM_RSRCS);
            end
        end
    end

    for (genvar i = 0; i < NUM_RSRCS; ++i) begin : g_ready
        assign rs_ready[i] = push_valid && (push_idx == `LOG2UP(NUM_RSRCS)'(i));
    end

    ///////////////////////////////////////////////////////////////////////////
    // Request FIFO and response queue.

    wire req_pop, rsp_pop;
    wire req_empty, rsp_empty;
    wire [REQ_PAYLOAD-1:0] req_head;
    wire                   req_head_l2;

    VX_fifo_queue #(
        .DATAW (1 + REQ_PAYLOAD),
        .DEPTH (REQ_DEPTH)
    ) req_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (push_valid),
        .pop      (req_pop),
        .data_in  ({rs_l2[push_idx], rs_payload[push_idx]}),
        .data_out ({req_head_l2, req_head}),
        .empty    (req_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        .size     (req_size)
    );

    wire                   rsp_push;
    wire [RSP_PAYLOAD-1:0] rsp_head;

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    wire [`CLOG2(RSP_DEPTH+1)-1:0] rsp_size;
    assign rsp_push = dsm_rsp_in_if.valid && (32'(rsp_size) < RSP_DEPTH);
    assign dsm_rsp_in_if.ready = rsp_push;

    VX_fifo_queue #(
        .DATAW (RSP_PAYLOAD),
        .DEPTH (RSP_DEPTH)
    ) rsp_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (rsp_push),
        .pop      (rsp_pop),
        .data_in  ({dsm_rsp_in_if.dst, dsm_rsp_in_if.data}),
        .data_out (rsp_head),
        .empty    (rsp_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        .size     (rsp_size)
    );
`else
    assign rsp_push  = 1'b0;
    assign rsp_empty = 1'b1;
    assign rsp_head  = '0;
    `UNUSED_VAR (rsp_push)
    `UNUSED_PARAM (RSP_DEPTH)
`endif

    ///////////////////////////////////////////////////////////////////////////
    // Serialiser: the request head's packet as flits.

    wire                  h_rw;
    wire [LADDR_W-1:0]    h_addr;
    wire [LINE_SIZE*8-1:0] h_data;
    wire [LINE_SIZE-1:0]  h_be;
    wire [MEM_ATTR_WIDTH-1:0] h_attr_bits;
    wire [MEM_TAG_W-1:0]  h_tag;
    assign {h_rw, h_addr, h_data, h_be, h_attr_bits, h_tag} = req_head[L2_PAYLOAD-1:0];
    mem_bus_attr_t h_attr;
    assign h_attr = mem_bus_attr_t'(h_attr_bits);
    `UNUSED_VAR (h_attr)

    wire [SECTORS-1:0] h_touched, h_full;
    for (genvar s = 0; s < SECTORS; ++s) begin : g_hsect
        assign h_touched[s] = (| h_be[s * FLIT +: FLIT]);
        assign h_full[s]    = (& h_be[s * FLIT +: FLIT]);
    end
    wire [SECT_W-1:0] h_nt;
    `POP_COUNT(h_nt, h_touched);
    wire h_amo     = h_attr.amo.amo_valid;
    wire h_wr      = h_rw && ~h_amo;
    wire h_partial = h_wr && ((h_touched & ~h_full) != '0);

    reg [SIDX_W-1:0] h_amo_sect;
    always @(*) begin
        h_amo_sect = '0;
        for (integer s = SECTORS - 1; s >= 0; --s) begin
            if (h_touched[s]) h_amo_sect = SIDX_W'(s);
        end
    end

    wire [CNT_W-1:0] req_head_flits = ~req_head_l2 ? CNT_W'(1)
                                    : (h_wr ? `MAX(CNT_W'(h_nt) + CNT_W'(h_partial), CNT_W'(1)) : CNT_W'(1));
    wire [L2_HDR_W-1:0] h_hdr = {h_rw, h_addr, h_attr_bits, h_tag, h_touched, h_partial,
                                 h_be[h_amo_sect * FLIT +: FLIT]};

    // The sector the k-th data flit of a sector set carries.
    function automatic [SIDX_W-1:0] kth_sector(input [SECTORS-1:0] m, input [CNT_W-1:0] k);
        reg [CNT_W-1:0] c;
        kth_sector = '0;
        c = '0;
        for (integer s = 0; s < SECTORS; ++s) begin
            if (m[s]) begin
                if (c == k) kth_sector = SIDX_W'(s);
                c = c + CNT_W'(1);
            end
        end
    endfunction

    // Every flit of the head packet: data flits of the touched sectors, then
    // the byte mask; an atomic's operand sector; an SM-to-SM message whole.
    wire [SECTORS:0][LINK_W-1:0] req_flit;
    for (genvar k = 0; k <= SECTORS; ++k) begin : g_flit
        wire [FLIT_BITS-1:0] d = h_wr ? ((k < 32'(h_nt)) ? h_data[kth_sector(h_touched, CNT_W'(k)) * FLIT_BITS +: FLIT_BITS]
                                                         : FLIT_BITS'(h_be))
                               : (h_amo ? h_data[h_amo_sect * FLIT_BITS +: FLIT_BITS] : '0);
        assign req_flit[k] = req_head_l2 ? {HDR_W'(h_hdr), d} : LINK_W'(req_head[DSM_PAYLOAD-1:0]);
    end

    reg [CNT_W-1:0] sent_q_r, sent_p_r;
    reg             turn_r;   // 0: request FIFO next, 1: response queue next

    reg [CNT_W-1:0] sent_q_n, sent_p_n;
    reg             turn_n;
    reg [UNITS-1:0] lk_valid;
    reg [UNITS-1:0] lk_rsp;
    reg [UNITS-1:0][CNT_W-1:0] lk_idx;
    reg [UNITS-1:0][LINK_W-1:0] lk_data;

    always @(*) begin
        reg cq, cp, g;
        cq       = 1'b0;
        cp       = 1'b0;
        g        = 1'b0;
        sent_q_n = sent_q_r;
        sent_p_n = sent_p_r;
        turn_n   = turn_r;
        lk_valid = '0;
        lk_rsp   = '0;
        lk_idx   = '0;
        lk_data  = '0;
        for (integer u = 0; u < UNITS; ++u) begin
            cq = ~req_empty && (sent_q_n < req_head_flits);
            cp = ~rsp_empty && (sent_p_n == '0);
            if (cq || cp) begin
                g = (cq && cp) ? turn_n : cp;
                lk_valid[u] = 1'b1;
                lk_rsp[u]   = g;
                if (g) begin
                    lk_data[u] = LINK_W'(rsp_head);
                    sent_p_n   = sent_p_n + CNT_W'(1);
                end else begin
                    lk_idx[u]  = sent_q_n;
                    lk_data[u] = req_flit[sent_q_n];
                    sent_q_n   = sent_q_n + CNT_W'(1);
                end
                turn_n = ~g;
            end
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // Deserialiser: the far end of the link rebuilds each packet.

    reg [LINK_W-1:0]      rx_q0_r, rx_p_r;
    reg [LINE_SIZE*8-1:0] rx_data_r;
    reg [LINE_SIZE-1:0]   rx_be_r;

    reg [LINK_W-1:0]      rx_q0_n, rx_p_n;
    reg [LINE_SIZE*8-1:0] rx_data_n;
    reg [LINE_SIZE-1:0]   rx_be_n;

    wire [L2_HDR_W-1:0] rx_hdr = rx_q0_n[FLIT_BITS +: L2_HDR_W];
    wire                  rx_rw;
    wire [LADDR_W-1:0]    rx_addr;
    wire [MEM_ATTR_WIDTH-1:0] rx_attr_bits;
    wire [MEM_TAG_W-1:0]  rx_tag;
    wire [SECTORS-1:0]    rx_smask;
    wire                  rx_partial;
    wire [FLIT-1:0]       rx_sbe;
    assign {rx_rw, rx_addr, rx_attr_bits, rx_tag, rx_smask, rx_partial, rx_sbe} = rx_hdr;
    mem_bus_attr_t rx_attr;
    assign rx_attr = mem_bus_attr_t'(rx_attr_bits);
    `UNUSED_VAR (rx_attr)

    always @(*) begin
        reg [L2_HDR_W-1:0] hdr;
        reg [SECTORS-1:0] sm;
        reg [CNT_W-1:0] nt;
        reg wr, amo;
        `IGNORE_UNUSED_BEGIN
        mem_bus_attr_t at;
        `IGNORE_UNUSED_END
        hdr       = '0;
        sm        = '0;
        nt        = '0;
        wr        = 1'b0;
        amo       = 1'b0;
        at        = '0;
        rx_q0_n   = rx_q0_r;
        rx_p_n    = rx_p_r;
        rx_data_n = rx_data_r;
        rx_be_n   = rx_be_r;
        for (integer u = 0; u < UNITS; ++u) begin
            if (lk_valid[u] && lk_rsp[u]) begin
                rx_p_n = lk_data[u];
            end else if (lk_valid[u]) begin
                if (lk_idx[u] == '0) begin
                    rx_q0_n = lk_data[u];
                end
                hdr = rx_q0_n[FLIT_BITS +: L2_HDR_W];
                sm  = hdr[FLIT + 1 +: SECTORS];
                at  = mem_bus_attr_t'(hdr[MEM_TAG_W + SECTORS + 1 + FLIT +: MEM_ATTR_WIDTH]);
                amo = at.amo.amo_valid;
                wr  = hdr[L2_HDR_W-1] && ~amo;
                nt  = '0;
                for (integer s = 0; s < SECTORS; ++s) begin
                    nt = nt + CNT_W'(sm[s]);
                end
                if (wr && (lk_idx[u] < nt)) begin
                    rx_data_n[kth_sector(sm, lk_idx[u]) * FLIT_BITS +: FLIT_BITS] = lk_data[u][FLIT_BITS-1:0];
                end else if (wr) begin
                    rx_be_n = lk_data[u][LINE_SIZE-1:0];
                end else if (amo) begin
                    rx_data_n[kth_sector(sm, '0) * FLIT_BITS +: FLIT_BITS] = lk_data[u][FLIT_BITS-1:0];
                end
            end
        end
    end

    wire rx_wr  = rx_rw && ~rx_attr.amo.amo_valid;
    reg [LINE_SIZE-1:0] rx_byteen;
    always @(*) begin
        rx_byteen = '0;
        for (integer s = 0; s < SECTORS; ++s) begin
            if (rx_wr && rx_partial) begin
                rx_byteen[s * FLIT +: FLIT] = rx_be_n[s * FLIT +: FLIT];
            end else if (rx_attr.amo.amo_valid) begin
                rx_byteen[s * FLIT +: FLIT] = rx_smask[s] ? rx_sbe : '0;
            end else begin
                rx_byteen[s * FLIT +: FLIT] = {FLIT{rx_smask[s]}};
            end
        end
    end
    wire [L2_PAYLOAD-1:0] rx_l2_req = {rx_rw, rx_addr, rx_data_n, rx_byteen, rx_attr_bits, rx_tag};

    // Sinks: the output register (L2 packets) and the network buffers.
    wire l2_sink_ready, dsm_sink_ready, rsp_sink_ready;

    assign req_pop = ~req_empty && (sent_q_n >= req_head_flits)
                  && (req_head_l2 ? l2_sink_ready : dsm_sink_ready);
    assign rsp_pop = ~rsp_empty && (sent_p_n != '0) && rsp_sink_ready;

    always @(posedge clk) begin
        if (reset) begin
            sent_q_r  <= '0;
            sent_p_r  <= '0;
            turn_r    <= 1'b0;
            rr_last_r <= '0;
        end else begin
            sent_q_r  <= req_pop ? '0 : sent_q_n;
            sent_p_r  <= rsp_pop ? '0 : sent_p_n;
            turn_r    <= turn_n;
            if (push_valid) begin
                rr_last_r <= push_idx;
            end
        end
        rx_q0_r   <= rx_q0_n;
        rx_p_r    <= rx_p_n;
        rx_data_r <= rx_data_n;
        rx_be_r   <= rx_be_n;
    end

    ///////////////////////////////////////////////////////////////////////////
    // Output register toward the L2.

    wire                 l2_out_valid;
    wire [L2_PAYLOAD-1:0] l2_out_data;
    wire                 l2_out_ready;
    wire                 eg_empty;

    VX_elastic_buffer #(
        .DATAW (L2_PAYLOAD),
        .SIZE  (2)
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_pop && req_head_l2),
        .ready_in  (l2_sink_ready),
        .data_in   (rx_l2_req),
        .data_out  (l2_out_data),
        .ready_out (l2_out_ready),
        .valid_out (l2_out_valid)
    );
    assign eg_empty = ~l2_out_valid;

    assign mem_bus_if.req_valid = l2_out_valid;
    assign {mem_bus_if.req_data.rw,
            mem_bus_if.req_data.addr,
            mem_bus_if.req_data.data,
            mem_bus_if.req_data.byteen,
            mem_bus_if.req_data.attr,
            mem_bus_if.req_data.tag} = l2_out_data;
    assign l2_out_ready = mem_bus_if.req_ready;

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    VX_elastic_buffer #(
        .DATAW (DSM_PAYLOAD),
        .SIZE  (2)
    ) dsm_out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_pop && ~req_head_l2),
        .ready_in  (dsm_sink_ready),
        .data_in   (rx_q0_n[DSM_PAYLOAD-1:0]),
        .data_out  ({dsm_req_out_if.dst, dsm_req_out_if.data}),
        .ready_out (dsm_req_out_if.ready),
        .valid_out (dsm_req_out_if.valid)
    );

    VX_elastic_buffer #(
        .DATAW (RSP_PAYLOAD),
        .SIZE  (2)
    ) rsp_out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_pop),
        .ready_in  (rsp_sink_ready),
        .data_in   (rx_p_n[RSP_PAYLOAD-1:0]),
        .data_out  ({dsm_rsp_out_if.dst, dsm_rsp_out_if.data}),
        .ready_out (dsm_rsp_out_if.ready),
        .valid_out (dsm_rsp_out_if.valid)
    );
`else
    assign dsm_sink_ready = 1'b0;
    assign rsp_sink_ready = 1'b0;
    `UNUSED_VAR (rx_p_n)
`endif

    ///////////////////////////////////////////////////////////////////////////
    // Ingress: L2 replies after their input register, the SM-to-SM arrivals.

    wire                     fill_valid;
    wire [LINE_SIZE*8-1:0]   fill_data;
    wire [MEM_TAG_W-1:0]     fill_tag;
    wire                     fill_pass;
    wire                     ig_empty;

    VX_elastic_buffer #(
        .DATAW (LINE_SIZE * 8 + MEM_TAG_W),
        .SIZE  (2)
    ) in_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_bus_if.rsp_valid),
        .ready_in  (mem_bus_if.rsp_ready),
        .data_in   ({mem_bus_if.rsp_data.data, mem_bus_if.rsp_data.tag}),
        .data_out  ({fill_data, fill_tag}),
        .ready_out (fill_pass),
        .valid_out (fill_valid)
    );
    assign ig_empty = ~fill_valid;

    wire [SECT_W-1:0]        fill_sect = fill_tag[SRC_TAG_WIDTH - UUID_WIDTH + SRC_W +: SECT_W];
    wire [SRC_W-1:0]         fill_src  = fill_tag[SRC_TAG_WIDTH - UUID_WIDTH +: SRC_W];
    wire [SRC_TAG_WIDTH-1:0] fill_stag = {fill_tag[MEM_TAG_W-1 -: UUID_WIDTH], fill_tag[SRC_TAG_WIDTH-UUID_WIDTH-1:0]};

    // Each source's replies leave through a registered buffer, so a pass never
    // waits on its sink combinationally.
    wire [NUM_SRCS-1:0] src_buf_ready;
    wire [NUM_SRCS-1:0] src_buf_valid;
    for (genvar i = 0; i < NUM_SRCS; ++i) begin : g_src_rsp
        VX_elastic_buffer #(
            .DATAW (LINE_SIZE * 8 + SRC_TAG_WIDTH),
            .SIZE  (2)
        ) rsp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (fill_pass && (fill_src == SRC_W'(i))),
            .ready_in  (src_buf_ready[i]),
            .data_in   ({fill_data, fill_stag}),
            .data_out  ({src_bus_if[i].rsp_data.data, src_bus_if[i].rsp_data.tag}),
            .ready_out (src_bus_if[i].rsp_ready),
            .valid_out (src_bus_if[i].rsp_valid)
        );
        assign src_buf_valid[i] = src_bus_if[i].rsp_valid;
    end
    wire fill_ready = src_buf_ready[fill_src];

    wire dreq_valid, drsp_valid, dreq_pass, drsp_pass;
    wire [SECT_W-1:0] dreq_sect, drsp_sect;
    wire dreq_ready, drsp_ready;

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    assign dreq_valid = arr_req_in_if.valid;
    assign drsp_valid = arr_rsp_in_if.valid;
    assign dreq_sect  = SECT_W'(1);
    assign drsp_sect  = SECT_W'(1);

    VX_elastic_buffer #(
        .DATAW (DLMEM_CORE_W + DLMEM_REQ_DATAW),
        .SIZE  (2)
    ) arr_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (dreq_pass),
        .ready_in  (dreq_ready),
        .data_in   ({arr_req_in_if.dst, arr_req_in_if.data}),
        .data_out  ({arr_req_out_if.dst, arr_req_out_if.data}),
        .ready_out (arr_req_out_if.ready),
        .valid_out (arr_req_out_if.valid)
    );
    assign arr_req_in_if.ready = dreq_pass;

    VX_elastic_buffer #(
        .DATAW (DLMEM_CORE_W + DLMEM_RSP_DATAW),
        .SIZE  (2)
    ) arr_rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (drsp_pass),
        .ready_in  (drsp_ready),
        .data_in   ({arr_rsp_in_if.dst, arr_rsp_in_if.data}),
        .data_out  ({arr_rsp_out_if.dst, arr_rsp_out_if.data}),
        .ready_out (arr_rsp_out_if.ready),
        .valid_out (arr_rsp_out_if.valid)
    );
    assign arr_rsp_in_if.ready = drsp_pass;
`else
    assign dreq_valid = 1'b0;
    assign drsp_valid = 1'b0;
    assign dreq_sect  = '0;
    assign drsp_sect  = '0;
    assign dreq_ready = 1'b0;
    assign drsp_ready = 1'b0;
    `UNUSED_VAR (dreq_pass)
    `UNUSED_VAR (drsp_pass)
`endif

    VX_sm_ingress #(
        .SECT_W  (SECT_W),
        .UNITS   (ING_UNITS),
        .EJECT   (EJECT),
        .W_FILL  (`VX_CFG_SM_INGRESS_FILL_WEIGHT),
        .W_DSMEM (`VX_CFG_SM_INGRESS_DSMEM_WEIGHT)
    ) ingress (
        .clk          (clk),
        .reset        (reset),
        .fill_valid   (fill_valid),
        .fill_sectors (fill_sect),
        .fill_pass    (fill_pass),
        .fill_ready   (fill_ready),
        .dreq_valid   (dreq_valid),
        .dreq_sectors (dreq_sect),
        .dreq_pass    (dreq_pass),
        .dreq_ready   (dreq_ready),
        .drsp_valid   (drsp_valid),
        .drsp_sectors (drsp_sect),
        .drsp_pass    (drsp_pass),
        .drsp_ready   (drsp_ready)
    );

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    assign empty = req_empty && rsp_empty && eg_empty && ig_empty && ~(| src_buf_valid)
                && ~dsm_req_out_if.valid && ~dsm_rsp_out_if.valid
                && ~arr_req_out_if.valid && ~arr_rsp_out_if.valid;
`else
    assign empty = req_empty && eg_empty && ig_empty && ~(| src_buf_valid);
`endif

`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (~reset && push_valid) begin
            `TRACE(3, ("%t: %s push: src=%0d, l2=%0d\n", $time, INSTANCE_ID, push_idx, rs_l2[push_idx]))
        end
    end
`endif

endmodule

`endif // VX_CFG_H100_SM_ENABLE
