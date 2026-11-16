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
// Egress: every request-class item (the L1TEX sources' line requests and the
// SM's own SM-to-SM packets) enters one in-order FIFO, one item per cycle,
// round robin over the sources an item can be admitted from (room in half-beats
// and entries, an issue token for a single-sector request). The SM-to-SM
// replies this SM serves wait in a response queue. Each cycle the port deals
// 2 * EGRESS_BYTES / BEAT half-beats to the two heads round robin; a half-beat
// that starts a slot of an SM-to-SM item also takes an injection credit,
// returned INJ_RETURN cycles later. A paid head leaves when its sink is ready;
// a head that cannot leave holds its queue. L2-side items then wait
// EGRESS_DELAY cycles beyond their output register (the SM side of an L2
// access).
//
// Ingress: L2 replies wait INGRESS_DELAY cycles beyond their input register,
// then pass the ingress gate
// with the SM-to-SM arrivals (VX_sm_ingress) and return to their source.
module VX_sm_port import VX_gpu_pkg::*;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    import VX_dlmem_pkg::*;
`endif
#(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_SRCS      = 1,    // L1TEX sources: 0 the L1, 1 the TMA engine
    parameter SRC_TAG_WIDTH = UUID_WIDTH + 1,
    parameter LINE_SIZE     = SM_PORT_LINE_SIZE,
    parameter EGRESS_DELAY  = 0,
    parameter INGRESS_DELAY = 0
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
    localparam BEAT       = `VX_CFG_SM_EGRESS_BEAT;
    localparam SECTORS    = LINE_SIZE / BEAT;
    localparam SECT_W     = `CLOG2(SECTORS + 1);
    localparam UNITS      = 2 * `VX_CFG_SM_EGRESS_BYTES / BEAT;
    localparam ING_UNITS  = `VX_CFG_SM_INGRESS_BYTES / BEAT;
    localparam EJECT      = `VX_CFG_SM2SM_EJECT_BYTES / BEAT;
    localparam SRC_W      = `LOG2UP(NUM_SRCS);
    localparam LADDR_W    = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(LINE_SIZE);
    localparam MEM_TAG_W  = SRC_TAG_WIDTH + SECT_W + SRC_W;
    localparam L2_HB_MAX  = SECTORS * `MAX(`MAX(`VX_CFG_SM_HB_WRITE_SECTOR, `VX_CFG_SM_HB_WRITE_MASKED), `VX_CFG_SM_HB_TMA_ST_SECTOR)
                          + `MAX(`VX_CFG_SM_HB_READ_LINE, `VX_CFG_SM_HB_ATOM_LANE);
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    localparam DSM_HB_MAX = (1 << DLMEM_HB_W) - 1;
`else
    localparam DSM_HB_MAX = 0;
`endif
    localparam HB_W       = `CLOG2(`MAX(L2_HB_MAX, DSM_HB_MAX) + 1);
    localparam SLOT_W     = HB_W;
    localparam REQ_CAP    = 2 * `VX_CFG_SM_REQ_FIFO_BYTES / BEAT;
    localparam REQ_DEPTH  = 1 << `CLOG2(REQ_CAP);
    localparam RSP_CAP    = `VX_CFG_SM_RSP_QUEUE_SLOTS;
    localparam RSP_DEPTH  = 1 << `CLOG2(RSP_CAP);
    localparam OCC_W      = `CLOG2(`MAX(REQ_CAP, 2 * RSP_CAP) * 2 + 1);
    localparam CRED_W     = `CLOG2(`VX_CFG_SM2SM_INJ_CREDITS + 1);
    localparam RET        = `VX_CFG_SM2SM_INJ_RETURN;
    localparam RET_W      = `CLOG2(UNITS + 1);
    localparam TOK_W      = 12;
    localparam TOK_W_CAP  = 1000 + `VX_CFG_SM_SECTOR_ISSUE_W_PERMILLE;
    localparam TOK_R_CAP  = 1000 + `VX_CFG_SM_SECTOR_ISSUE_R_PERMILLE;
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
    localparam REQ_META    = HB_W + SLOT_W + 1;
    localparam RSP_META    = HB_W + SLOT_W;
    localparam TS_W        = 32;

`ifndef DBG_TRACE_MEM
    `UNUSED_SPARAM (INSTANCE_ID)
`endif

    `STATIC_ASSERT (BEAT == `VX_CFG_DCACHE_SECTOR_SIZE, ("sm port: the egress beat must be the L1 sector"))
    `STATIC_ASSERT (`VX_CFG_SM_EGRESS_BYTES % BEAT == 0 && `VX_CFG_SM_INGRESS_BYTES % BEAT == 0,
        ("sm port: the egress and ingress must be whole beats"))
    `STATIC_ASSERT (LINE_SIZE % BEAT == 0, ("sm port: the line must be whole sectors"))
    `STATIC_ASSERT (`VX_CFG_SM2SM_INJ_CREDITS >= 1 && `VX_CFG_SM2SM_INJ_CREDITS <= RET,
        ("sm port: injection credits must be 1..return"))
    `STATIC_ASSERT (`VX_CFG_SM2SM_EJECT_BYTES <= `VX_CFG_SM_INGRESS_BYTES, ("sm port: ejection above the ingress"))
    `STATIC_ASSERT (NUM_SRCS >= 1 && NUM_SRCS <= 2, ("sm port: one L1 and at most one TMA source"))

    ///////////////////////////////////////////////////////////////////////////
    // Request-class sources and their costs.

    wire [NUM_RSRCS-1:0]                 rs_valid;
    wire [NUM_RSRCS-1:0][HB_W-1:0]       rs_hb;
    wire [NUM_RSRCS-1:0][SLOT_W-1:0]     rs_slots;
    wire [NUM_RSRCS-1:0]                 rs_l2;
    wire [NUM_RSRCS-1:0][1:0]            rs_single;
    wire [NUM_RSRCS-1:0][REQ_PAYLOAD-1:0] rs_payload;
    wire [NUM_RSRCS-1:0]                 rs_ready;

    for (genvar i = 0; i < NUM_SRCS; ++i) begin : g_src
        wire [LINE_SIZE-1:0] be = src_bus_if[i].req_data.byteen;
        mem_bus_attr_t attr;
        assign attr = mem_bus_attr_t'(src_bus_if[i].req_data.attr);
        `UNUSED_VAR (attr)

        wire [SECTORS-1:0] touched, full;
        for (genvar s = 0; s < SECTORS; ++s) begin : g_sect
            assign touched[s] = (| be[s * BEAT +: BEAT]);
            assign full[s]    = (& be[s * BEAT +: BEAT]);
        end

        wire [SECT_W-1:0] n_touched, n_full;
        `POP_COUNT(n_touched, touched);
        `POP_COUNT(n_full, full);

        wire is_fence = attr.is_fence;
        wire is_amo   = attr.amo.amo_valid;
        wire is_rd    = ~src_bus_if[i].req_data.rw && ~is_fence && ~is_amo;
        wire is_wr    = src_bus_if[i].req_data.rw && ~is_amo;

        // Sectors the reply brings back (0: no reply, or one that carries no fill).
        wire [SECT_W-1:0] rsp_sect = is_rd ? ((n_touched != '0) ? n_touched : SECT_W'(SECTORS))
                                   : (is_amo ? SECT_W'(1) : '0);

        reg [HB_W-1:0] hb;
        reg [1:0]      single;
        always @(*) begin
            single = 2'd0;
            if (is_fence) begin
                hb = HB_W'(`VX_CFG_SM_HB_READ_LINE);
            end else if (is_amo) begin
                hb = HB_W'(`VX_CFG_SM_HB_ATOM_LANE);
            end else if (is_wr) begin
                if (i == 0) begin
                    hb = HB_W'(n_full) * HB_W'(`VX_CFG_SM_HB_WRITE_SECTOR)
                       + HB_W'(n_touched - n_full) * HB_W'(`VX_CFG_SM_HB_WRITE_MASKED);
                    single = (n_touched == SECT_W'(1)) ? 2'd1 : 2'd0;
                end else begin
                    hb = HB_W'(n_touched) * HB_W'(`VX_CFG_SM_HB_TMA_ST_SECTOR);
                end
            end else begin
                hb = HB_W'(`VX_CFG_SM_HB_READ_LINE);
                if (i == 0) begin
                    single = (n_touched == SECT_W'(1)) ? 2'd2 : 2'd0;
                end
            end
        end

        wire [MEM_TAG_W-1:0] out_tag = {src_bus_if[i].req_data.tag.uuid, rsp_sect, SRC_W'(i),
                                        src_bus_if[i].req_data.tag.value};

        assign rs_valid[i]   = src_bus_if[i].req_valid;
        assign rs_hb[i]      = hb;
        assign rs_slots[i]   = '0;
        assign rs_l2[i]      = 1'b1;
        assign rs_single[i]  = single;
        assign rs_payload[i] = REQ_PAYLOAD'({src_bus_if[i].req_data.rw,
                                             src_bus_if[i].req_data.addr,
                                             src_bus_if[i].req_data.data,
                                             src_bus_if[i].req_data.byteen,
                                             src_bus_if[i].req_data.attr,
                                             out_tag});
        assign src_bus_if[i].req_ready = rs_ready[i];
    end

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    dlmem_req_pkt_t dsm_req_pkt;
    assign dsm_req_pkt = dlmem_req_pkt_t'(dsm_req_in_if.data);
    `UNUSED_VAR (dsm_req_pkt)
    assign rs_valid[NUM_SRCS]   = dsm_req_in_if.valid;
    assign rs_hb[NUM_SRCS]      = HB_W'(dsm_req_pkt.pkt_hb);
    assign rs_slots[NUM_SRCS]   = SLOT_W'((HB_W'(dsm_req_pkt.pkt_hb) + HB_W'(1)) >> 1);
    assign rs_l2[NUM_SRCS]      = 1'b0;
    assign rs_single[NUM_SRCS]  = 2'd0;
    assign rs_payload[NUM_SRCS] = REQ_PAYLOAD'({dsm_req_in_if.dst, dsm_req_in_if.data});
    assign dsm_req_in_if.ready  = rs_ready[NUM_SRCS];
`endif

    ///////////////////////////////////////////////////////////////////////////
    // Admission: one push per cycle, round robin over the admissible sources.

    reg  [OCC_W-1:0] req_occ_r, rsp_occ_r;
    reg  [TOK_W-1:0] tok_w_r, tok_r_r;
    wire [`CLOG2(REQ_DEPTH+1)-1:0] req_size;
    wire [`CLOG2(RSP_DEPTH+1)-1:0] rsp_size;

    wire [NUM_RSRCS-1:0] rs_admit;
    for (genvar i = 0; i < NUM_RSRCS; ++i) begin : g_admit
        assign rs_admit[i] = rs_valid[i]
                          && ((req_occ_r + OCC_W'(rs_hb[i])) <= OCC_W'(REQ_CAP))
                          && (32'(req_size) < REQ_CAP)
                          && ~((rs_single[i] == 2'd1) && (tok_w_r < TOK_W'(1000)))
                          && ~((rs_single[i] == 2'd2) && (tok_r_r < TOK_W'(1000)));
    end

    reg [`LOG2UP(NUM_RSRCS)-1:0] rr_last_r;
    reg [`LOG2UP(NUM_RSRCS)-1:0] push_idx;
    reg                          push_valid;
    always @(*) begin
        push_valid = 1'b0;
        push_idx   = '0;
        for (integer k = NUM_RSRCS; k >= 1; --k) begin
            if (rs_admit[(32'(rr_last_r) + k) % NUM_RSRCS]) begin
                push_valid = 1'b1;
                push_idx   = `LOG2UP(NUM_RSRCS)'((32'(rr_last_r) + k) % NUM_RSRCS);
            end
        end
    end

    for (genvar i = 0; i < NUM_RSRCS; ++i) begin : g_ready
        assign rs_ready[i] = push_valid && (push_idx == `LOG2UP(NUM_RSRCS)'(i));
    end

    wire [HB_W-1:0]   push_hb     = rs_hb[push_idx];
    wire [1:0]        push_single = push_valid ? rs_single[push_idx] : 2'd0;

    ///////////////////////////////////////////////////////////////////////////
    // Request FIFO and response queue.

    wire req_pop, rsp_pop;
    wire req_empty, rsp_empty;
    wire [REQ_PAYLOAD-1:0] req_head;
    wire [REQ_META-1:0]    req_head_meta;
    wire [HB_W-1:0]        req_head_hb;
    wire [SLOT_W-1:0]      req_head_slots;
    wire                   req_head_l2;

    VX_fifo_queue #(
        .DATAW (REQ_META + REQ_PAYLOAD),
        .DEPTH (REQ_DEPTH)
    ) req_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (push_valid),
        .pop      (req_pop),
        .data_in  ({rs_hb[push_idx], rs_slots[push_idx], rs_l2[push_idx], rs_payload[push_idx]}),
        .data_out ({req_head_meta, req_head}),
        .empty    (req_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        .size     (req_size)
    );
    assign {req_head_hb, req_head_slots, req_head_l2} = req_head_meta;

    wire                   rsp_push;
    wire [HB_W-1:0]        rsp_in_hb;
    wire [RSP_PAYLOAD-1:0] rsp_head;
    wire [HB_W-1:0]        rsp_head_hb;
    wire [SLOT_W-1:0]      rsp_head_slots;

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    dlmem_rsp_pkt_t dsm_rsp_pkt;
    assign dsm_rsp_pkt = dlmem_rsp_pkt_t'(dsm_rsp_in_if.data);
    `UNUSED_VAR (dsm_rsp_pkt)
    assign rsp_in_hb = HB_W'(dsm_rsp_pkt.pkt_hb);
    assign rsp_push = dsm_rsp_in_if.valid
                   && ((rsp_occ_r + OCC_W'(rsp_in_hb)) <= OCC_W'(2 * RSP_CAP))
                   && (32'(rsp_size) < RSP_CAP);
    assign dsm_rsp_in_if.ready = rsp_push;

    VX_fifo_queue #(
        .DATAW (RSP_META + RSP_PAYLOAD),
        .DEPTH (RSP_DEPTH)
    ) rsp_fifo (
        .clk      (clk),
        .reset    (reset),
        .push     (rsp_push),
        .pop      (rsp_pop),
        .data_in  ({rsp_in_hb, SLOT_W'((rsp_in_hb + HB_W'(1)) >> 1), dsm_rsp_in_if.dst, dsm_rsp_in_if.data}),
        .data_out ({rsp_head_hb, rsp_head_slots, rsp_head}),
        .empty    (rsp_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        .size     (rsp_size)
    );
`else
    assign rsp_push       = 1'b0;
    assign rsp_in_hb      = '0;
    assign rsp_empty      = 1'b1;
    assign rsp_head       = '0;
    assign rsp_head_hb    = '0;
    assign rsp_head_slots = '0;
    assign rsp_size       = '0;
    `UNUSED_PARAM (RSP_META)
    `UNUSED_VAR (rsp_size)
    `UNUSED_VAR (rsp_head_slots)
`endif

    ///////////////////////////////////////////////////////////////////////////
    // Payment: UNITS half-beats per cycle, round robin over the two heads.

    reg  [HB_W-1:0]  paid_q_r, paid_p_r;
    reg              turn_r;   // 0: request FIFO next, 1: response queue next
    reg  [CRED_W-1:0] credits_r;
    reg  [RET-1:0][RET_W-1:0] ret_r;
    reg  [`LOG2UP(RET)-1:0]   ret_pos_r;

    reg  [HB_W-1:0]   paid_q_n, paid_p_n;
    reg               turn_n;
    reg  [CRED_W-1:0] credits_n;
    reg  [RET_W-1:0]  used_n;

    always @(*) begin
        reg cq, cp, nq, np, g;
        paid_q_n  = paid_q_r;
        paid_p_n  = paid_p_r;
        turn_n    = turn_r;
        credits_n = credits_r + CRED_W'(ret_r[ret_pos_r]);
        used_n    = '0;
        for (integer u = 0; u < UNITS; ++u) begin
            g  = 1'b0;
            nq = ~paid_q_n[0] && (SLOT_W'(paid_q_n >> 1) < req_head_slots);
            cq = ~req_empty && (paid_q_n < req_head_hb) && ~(nq && (credits_n == '0));
        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            np = ~paid_p_n[0] && (SLOT_W'(paid_p_n >> 1) < rsp_head_slots);
            cp = ~rsp_empty && (paid_p_n < rsp_head_hb) && ~(np && (credits_n == '0));
        `else
            np = 1'b0;
            cp = 1'b0;
        `endif
            if (cq || cp) begin
                g = (cq && cp) ? turn_n : cp;
                if (g ? np : nq) begin
                    credits_n = credits_n - CRED_W'(1);
                    used_n    = used_n + RET_W'(1);
                end
                if (g) begin
                    paid_p_n = paid_p_n + HB_W'(1);
                end else begin
                    paid_q_n = paid_q_n + HB_W'(1);
                end
                turn_n = ~g;
            end
        end
    end

    // Sinks: the egress delay line (L2 items) and the network buffers.
    wire l2_sink_ready, dsm_sink_ready, rsp_sink_ready;

    assign req_pop = ~req_empty && (paid_q_n >= req_head_hb)
                  && (req_head_l2 ? l2_sink_ready : dsm_sink_ready);
    assign rsp_pop = ~rsp_empty && (paid_p_n >= rsp_head_hb) && rsp_sink_ready;

    always @(posedge clk) begin
        if (reset) begin
            req_occ_r <= '0;
            rsp_occ_r <= '0;
            paid_q_r  <= '0;
            paid_p_r  <= '0;
            turn_r    <= 1'b0;
            credits_r <= CRED_W'(`VX_CFG_SM2SM_INJ_CREDITS);
            ret_r     <= '0;
            ret_pos_r <= '0;
            tok_w_r   <= TOK_W'(TOK_W_CAP);
            tok_r_r   <= TOK_W'(TOK_R_CAP);
            rr_last_r <= '0;
        end else begin
            req_occ_r <= req_occ_r + (push_valid ? OCC_W'(push_hb) : '0) - (req_pop ? OCC_W'(req_head_hb) : '0);
            rsp_occ_r <= rsp_occ_r + (rsp_push ? OCC_W'(rsp_in_hb) : '0) - (rsp_pop ? OCC_W'(rsp_head_hb) : '0);
            paid_q_r  <= req_pop ? '0 : paid_q_n;
            paid_p_r  <= rsp_pop ? '0 : paid_p_n;
            turn_r    <= turn_n;
            credits_r <= credits_n;
            ret_r[ret_pos_r] <= used_n;
            ret_pos_r <= (32'(ret_pos_r) == (RET - 1)) ? '0 : (ret_pos_r + 1);
            tok_w_r   <= `MIN(tok_w_r - ((push_single == 2'd1) ? TOK_W'(1000) : '0) + TOK_W'(`VX_CFG_SM_SECTOR_ISSUE_W_PERMILLE), TOK_W'(TOK_W_CAP));
            tok_r_r   <= `MIN(tok_r_r - ((push_single == 2'd2) ? TOK_W'(1000) : '0) + TOK_W'(`VX_CFG_SM_SECTOR_ISSUE_R_PERMILLE), TOK_W'(TOK_R_CAP));
            if (push_valid) begin
                rr_last_r <= push_idx;
            end
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // Egress delay line toward the L2.

    reg [TS_W-1:0] now_r;
    always @(posedge clk) begin
        if (reset) begin
            now_r <= '0;
        end else begin
            now_r <= now_r + TS_W'(1);
        end
    end

    wire                 l2_out_valid;
    wire [L2_PAYLOAD-1:0] l2_out_data;
    wire                 l2_out_ready;
    wire                 eg_empty;

    if (EGRESS_DELAY != 0) begin : g_eg_delay
        localparam DEPTH = 1 << `CLOG2(EGRESS_DELAY + 2);
        wire [TS_W-1:0] rel;
        wire full;
        VX_fifo_queue #(
            .DATAW (TS_W + L2_PAYLOAD),
            .DEPTH (DEPTH)
        ) delay_q (
            .clk      (clk),
            .reset    (reset),
            .push     (req_pop && req_head_l2),
            .pop      (l2_out_valid && l2_out_ready),
            .data_in  ({now_r + TS_W'(1 + EGRESS_DELAY), req_head[L2_PAYLOAD-1:0]}),
            .data_out ({rel, l2_out_data}),
            .empty    (eg_empty),
            `UNUSED_PIN (alm_empty),
            .full     (full),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
        assign l2_sink_ready = ~full;
        assign l2_out_valid  = ~eg_empty && ($signed(now_r - rel) >= 0);
    end else begin : g_eg_nodelay
        VX_elastic_buffer #(
            .DATAW (L2_PAYLOAD),
            .SIZE  (2)
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (req_pop && req_head_l2),
            .ready_in  (l2_sink_ready),
            .data_in   (req_head[L2_PAYLOAD-1:0]),
            .data_out  (l2_out_data),
            .ready_out (l2_out_ready),
            .valid_out (l2_out_valid)
        );
        assign eg_empty = ~l2_out_valid;
    end

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
        .data_in   (req_head[DSM_PAYLOAD-1:0]),
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
        .data_in   (rsp_head),
        .data_out  ({dsm_rsp_out_if.dst, dsm_rsp_out_if.data}),
        .ready_out (dsm_rsp_out_if.ready),
        .valid_out (dsm_rsp_out_if.valid)
    );
`else
    assign dsm_sink_ready = 1'b0;
    assign rsp_sink_ready = 1'b0;
    `UNUSED_VAR (rsp_head)
`endif

    ///////////////////////////////////////////////////////////////////////////
    // Ingress: L2 replies after the delay line, the SM-to-SM arrivals.

    wire                     fill_valid;
    wire [LINE_SIZE*8-1:0]   fill_data;
    wire [MEM_TAG_W-1:0]     fill_tag;
    wire                     fill_pass;
    wire                     ig_empty;

    if (INGRESS_DELAY != 0) begin : g_ig_delay
        localparam DEPTH = 1 << `CLOG2(INGRESS_DELAY + 2);
        wire [TS_W-1:0] rel;
        wire full;
        VX_fifo_queue #(
            .DATAW (TS_W + LINE_SIZE * 8 + MEM_TAG_W),
            .DEPTH (DEPTH)
        ) delay_q (
            .clk      (clk),
            .reset    (reset),
            .push     (mem_bus_if.rsp_valid && ~full),
            .pop      (fill_pass),
            .data_in  ({now_r + TS_W'(1 + INGRESS_DELAY), mem_bus_if.rsp_data.data, mem_bus_if.rsp_data.tag}),
            .data_out ({rel, fill_data, fill_tag}),
            .empty    (ig_empty),
            `UNUSED_PIN (alm_empty),
            .full     (full),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
        assign mem_bus_if.rsp_ready = ~full;
        assign fill_valid = ~ig_empty && ($signed(now_r - rel) >= 0);
    end else begin : g_ig_nodelay
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
    end

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
    dlmem_req_pkt_t arr_req_pkt;
    dlmem_rsp_pkt_t arr_rsp_pkt;
    assign arr_req_pkt = dlmem_req_pkt_t'(arr_req_in_if.data);
    assign arr_rsp_pkt = dlmem_rsp_pkt_t'(arr_rsp_in_if.data);
    `UNUSED_VAR (arr_req_pkt)
    `UNUSED_VAR (arr_rsp_pkt)
    assign dreq_valid = arr_req_in_if.valid;
    assign drsp_valid = arr_rsp_in_if.valid;
    assign dreq_sect  = SECT_W'((HB_W'(arr_req_pkt.pkt_hb) + HB_W'(1)) >> 1);
    assign drsp_sect  = SECT_W'((HB_W'(arr_rsp_pkt.pkt_hb) + HB_W'(1)) >> 1);

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
            `TRACE(3, ("%t: %s push: src=%0d, hb=%0d, l2=%0d\n", $time, INSTANCE_ID, push_idx, push_hb, rs_l2[push_idx]))
        end
    end
`endif

endmodule

`endif // VX_CFG_H100_SM_ENABLE
