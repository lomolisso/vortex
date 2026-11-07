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

// DXA store engine: the local-memory-to-global direction of a box walk.
// Each in-bounds token of the walk is read from the issuer's local memory,
// one word per read (one element per read with a dim-0 stride), its bytes
// rotated to the token's line offsets, then written as one global line write
// with its byte enables, or reduced as one atomic per word at the last-level
// cache, or sent as one DLMEM atomic packet per element to a cluster block.
// An out-of-bounds token costs its walk cycle only, and a byte outside the
// box is never written: a 16-bit pair at a box edge reduces its one half. A store ends with one
// fence request on the global port, whose response marks its writes
// performed; a global reduction with its last atomic's response.

// With the knobs off the file is empty: VX_define.vh derives DXA_MBAR_ENABLE.
`ifdef VX_CFG_EXT_DXA_ENABLE
`ifdef VX_CFG_EXT_MBAR_ENABLE
`include "VX_define.vh"
`endif
`endif

`ifdef DXA_MBAR_ENABLE

module VX_dxa_st import VX_gpu_pkg::*, VX_dxa_pkg::*, VX_dlmem_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CL_SIZE         = `VX_CFG_L1_LINE_SIZE,
    parameter WORD_SIZE       = DXA_LMEM_WORD_SIZE,
    parameter LINE_ADDR_W     = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(CL_SIZE),
    parameter GMEM_TAG_WIDTH  = SOCKET_MEM_TAG_WIDTH,
    parameter DEPTH           = 8,
    localparam CL_OFF_BITS    = `CLOG2(CL_SIZE)
) (
    input  wire                        clk,
    input  wire                        reset,

    // The transfer (stable from start to done).
    input  wire                        start,
    input  wire                        active,
    input  dxa_st_t                    st,
    input  wire                        bg,
    input  dxa_tma_wr_t                tma,
    input  wire [3:0]                  elem_bytes,
    input  wire [NC_WIDTH-1:0]         core_id,
    input  wire [UUID_WIDTH-1:0]       uuid,
    input  wire [DLMEM_CORE_W-1:0]     dst_core,
    input  wire [NW_WIDTH-1:0]         dst_slot,
    input  wire [DXA_MBAR_OFF_W-1:0]   mbar_off,

    // The walk's tokens.
    input  wire                        ag_valid,
    output wire                        ag_ready,
    input  wire [LINE_ADDR_W-1:0]      ag_cl_addr,
    input  wire [DXA_SMEM_ADDR_W-1:0]  ag_smem_byte_addr,
    input  wire [CL_OFF_BITS-1:0]      ag_byte_offset,
    input  wire [CL_OFF_BITS:0]        ag_valid_length,
    input  wire                        ag_oob,
    input  wire                        ag_last,

    // Local-memory word reads and their in-order responses (never refused).
    output wire                        rd_valid,
    output wire [DXA_LMEM_ADDR_W-1:0]  rd_addr,
    input  wire                        rd_ready,
    input  wire                        rsp_valid,
    input  wire [WORD_SIZE*8-1:0]      rsp_data,

    // Global writes, atomics and the fence.
    VX_mem_bus_if.master               gmem_bus_if,

    // Cluster-reduction packets to the row egress.
    output wire                        amo_valid,
    output dlmem_req_pkt_t             amo_pkt,
    output wire [DLMEM_CORE_W-1:0]     amo_dst,
    input  wire                        amo_ready,

    // Completion events of a bulk_group operation.
    output wire                        evt_valid,
    output wire                        evt_read_done,
    output wire                        evt_write_done,
    input  wire                        evt_ready,

    output wire                        done,
    output wire                        progress
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (st.en)
    `UNUSED_VAR (st.tsize)
    `UNUSED_VAR (tma.prefetch)

    localparam WORD_LOG   = `CLOG2(WORD_SIZE);
    localparam WORD_W     = WORD_SIZE * 8;
    localparam CL_W       = CL_SIZE * 8;
    localparam ADDR_W     = DXA_LMEM_ADDR_W;
    localparam DEPTH_W    = `CLOG2(DEPTH + 1);
    localparam NG4        = CL_SIZE / 4;               // 4-byte words of a line
    localparam NG4_W      = `LOG2UP(NG4);
    localparam TAG_VALUE_W = GMEM_TAG_WIDTH - UUID_WIDTH;
    localparam CNT_W      = 20;
    localparam SWZ_TOP    = (WORD_LOG < 7) ? WORD_LOG : 7;
    localparam SWZ_STAGES = (SWZ_TOP > 4) ? (SWZ_TOP - 4) : 0;

    `STATIC_ASSERT (`IS_POW2(CL_SIZE) && `IS_POW2(WORD_SIZE), ("invalid parameter"))
    `STATIC_ASSERT (CL_SIZE >= 8, ("invalid parameter"))
    `STATIC_ASSERT (LSU_WORD_SIZE == 8, ("invalid parameter"))

    wire is_store = (st.kind == DXA_ST_BULK) || (st.kind == DXA_ST_TENSOR);
    wire is_red_c = (st.kind == DXA_ST_RED_C);
    wire is_red_g = ~is_store && ~is_red_c;

    // ════════════════════════════════════════════════════════════════════
    // Reads: one per local-memory word of the token's source bytes, or one
    // per element of a gathered token, in order.
    // ════════════════════════════════════════════════════════════════════
    wire gather = tma.en && (tma.es0 != 4'd1);
    // A gathered element's line step; the walk may offer its first token the
    // cycle after start, so it is not registered.
    wire [6:0] rd_step = 7'(tma.es0) << ((elem_bytes == 4'd8) ? 3 : (elem_bytes == 4'd4) ? 2 : (elem_bytes == 4'd2) ? 1 : 0);

    reg                          mid_r;      // a token's reads are under way
    reg [DXA_SMEM_ADDR_W-1:0]    a_r;
    reg [CL_OFF_BITS:0]          p_r;
    reg [CL_OFF_BITS:0]          rem_r;
    reg                          walk_done_r;

    wire [DXA_SMEM_ADDR_W-1:0] a_cur   = mid_r ? a_r : ag_smem_byte_addr;
    wire [CL_OFF_BITS:0]       p_cur   = mid_r ? p_r : {1'b0, ag_byte_offset};
    wire [CL_OFF_BITS:0]       rem_cur = mid_r ? rem_r : ag_valid_length;

    wire [WORD_LOG:0] word_room = (WORD_LOG+1)'(WORD_SIZE) - (WORD_LOG+1)'(a_cur[WORD_LOG-1:0]);
    wire [CL_OFF_BITS:0] n_cur = gather ? (CL_OFF_BITS+1)'(elem_bytes)
                               : ((32'(word_room) < 32'(rem_cur)) ? (CL_OFF_BITS+1)'(word_room) : rem_cur);
    wire rd_last = (n_cur >= rem_cur);

    // Swizzle on the block-relative word, as the writer does it.
    wire [ADDR_W-1:0] word_l   = ADDR_W'(a_cur >> WORD_LOG);
    wire [ADDR_W-1:0] swz_bw   = ADDR_W'(tma.blk_base >> WORD_LOG);
    wire [ADDR_W-1:0] swz_rel  = word_l - swz_bw;
    wire [ADDR_W+WORD_LOG-1:0] swz_rel_b = {swz_rel, WORD_LOG'(0)};
    wire [2:0] swz_m  = (tma.swizzle == 2'd1) ? 3'b001 : (tma.swizzle == 2'd2) ? 3'b011 : (tma.swizzle == 2'd3) ? 3'b111 : 3'b000;
    wire [2:0] swz_xv = 3'(swz_rel_b >> 7) & swz_m;
    wire [6:0] swz_x  = {swz_xv, 4'b0000};
    wire [ADDR_W-1:0] word_p = swz_bw + (swz_rel ^ ADDR_W'(swz_x >> WORD_LOG));

    // What the assembly needs per read.
    typedef struct packed {
        logic [WORD_LOG-1:0]      rot;      // source byte address - line offset, mod WORD_SIZE
        logic [CL_OFF_BITS:0]     p0;       // first line offset
        logic [CL_OFF_BITS:0]     n;        // bytes
        logic [2:0]               swz;      // chunk swap bits 4..6
        logic                     tok_last;
        logic [LINE_ADDR_W-1:0]   cl_addr;
    } rd_info_t;

    rd_info_t rd_info;
    wire [DXA_SMEM_ADDR_W-1:0] rot_full = a_cur - DXA_SMEM_ADDR_W'(p_cur);
    assign rd_info.rot = rot_full[WORD_LOG-1:0];
    `UNUSED_VAR (rot_full[DXA_SMEM_ADDR_W-1:WORD_LOG])
    assign rd_info.p0       = p_cur;
    assign rd_info.n        = n_cur;
    assign rd_info.swz      = swz_xv;
    assign rd_info.tok_last = rd_last;
    assign rd_info.cl_addr  = ag_cl_addr;

    wire info_full, info_empty;
    wire tok_ib   = active && ag_valid && ~ag_oob && ~walk_done_r;
    assign rd_valid = tok_ib && ~info_full;
    assign rd_addr  = word_p;
    wire rd_fire  = rd_valid && rd_ready;
    wire oob_take = active && ag_valid && ag_oob && ~walk_done_r;
    assign ag_ready = oob_take || (rd_fire && rd_last);
    wire tok_take = ag_valid && ag_ready;

    always @(posedge clk) begin
        if (reset || start) begin
            mid_r       <= 1'b0;
            walk_done_r <= 1'b0;
        end else begin
            if (rd_fire) begin
                mid_r <= ~rd_last;
                a_r   <= a_cur + (gather ? DXA_SMEM_ADDR_W'(elem_bytes) : DXA_SMEM_ADDR_W'(n_cur));
                p_r   <= p_cur + (gather ? (CL_OFF_BITS+1)'(rd_step) : n_cur);
                rem_r <= rem_cur - n_cur;
            end
            if (tok_take && ag_last) begin
                walk_done_r <= 1'b1;
            end
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // Read info and response queues; the info entries are the credits.
    // ════════════════════════════════════════════════════════════════════
    rd_info_t asm_info;
    wire [WORD_W-1:0] asm_data;
    wire rsp_empty;
    wire asm_fire;
    wire [DEPTH_W-1:0] info_size;

    VX_fifo_queue #(
        .DATAW (($bits(rd_info_t))),
        .DEPTH (DEPTH)
    ) info_q (
        .clk      (clk),
        .reset    (reset),
        .push     (rd_fire),
        .pop      (asm_fire),
        .data_in  (rd_info),
        .data_out (asm_info),
        .empty    (info_empty),
        .full     (info_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        .size     (info_size)
    );
    `UNUSED_VAR (info_size)

    VX_fifo_queue #(
        .DATAW (WORD_W),
        .DEPTH (DEPTH)
    ) rsp_q (
        .clk      (clk),
        .reset    (reset),
        .push     (rsp_valid),
        .pop      (asm_fire),
        .data_in  (rsp_data),
        .data_out (asm_data),
        .empty    (rsp_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    // ════════════════════════════════════════════════════════════════════
    // Assembly: un-swizzle, rotate to the line offsets, merge.
    // ════════════════════════════════════════════════════════════════════
    // Bit t of the chunk swap moves 16-byte chunks 1 << t apart inside the word.
    wire [SWZ_STAGES:0][WORD_W-1:0] uswz_d;
    assign uswz_d[0] = asm_data;
    for (genvar t = 0; t < SWZ_STAGES; ++t) begin : g_uswz
        for (genvar i = 0; i < WORD_SIZE; ++i) begin : g_bytes
            localparam J = i ^ (1 << (4 + t));
            assign uswz_d[t+1][i*8 +: 8] = asm_info.swz[t] ? uswz_d[t][J*8 +: 8] : uswz_d[t][i*8 +: 8];
        end
    end
    if (SWZ_STAGES < 3) begin : g_uswz_x
        `UNUSED_VAR (asm_info.swz[2:SWZ_STAGES])
    end

    // rot_d[q] = word[(q + rot) mod WORD_SIZE]; line byte p takes rot_d[p mod WORD_SIZE].
    wire [2*WORD_W-1:0] dbl = {uswz_d[SWZ_STAGES], uswz_d[SWZ_STAGES]};
    wire [WORD_W-1:0] rot_d = WORD_W'(dbl >> {asm_info.rot, 3'b000});
    if (WORD_SIZE > CL_SIZE) begin : g_rot_hi
        `UNUSED_VAR (rot_d[WORD_W-1:CL_W])
    end
    wire [CL_W-1:0]   cand;
    wire [CL_SIZE-1:0] amask;
    for (genvar p = 0; p < CL_SIZE; ++p) begin : g_cand
        assign cand[p*8 +: 8] = rot_d[(p % WORD_SIZE)*8 +: 8];
        assign amask[p] = ((CL_OFF_BITS+2)'(p) >= (CL_OFF_BITS+2)'(asm_info.p0))
                       && ((CL_OFF_BITS+2)'(p) < ((CL_OFF_BITS+2)'(asm_info.p0) + (CL_OFF_BITS+2)'(asm_info.n)));
    end

    reg [CL_W-1:0]    acc_r;
    reg [CL_SIZE-1:0] acc_be_r;
    wire [CL_SIZE*8-1:0] amask_bits;
    for (genvar p = 0; p < CL_SIZE; ++p) begin : g_amask_bits
        assign amask_bits[p*8 +: 8] = {8{amask[p]}};
    end
    wire [CL_W-1:0]    acc_n    = acc_r | (cand & amask_bits);
    wire [CL_SIZE-1:0] acc_be_n = acc_be_r | amask;

    wire [CL_W-1:0]    tok_data = acc_n;
    wire [CL_SIZE-1:0] tok_be   = acc_be_n;

    // ════════════════════════════════════════════════════════════════════
    // Output slot and emission.
    // ════════════════════════════════════════════════════════════════════
    reg                    slot_valid_r;
    reg [CL_W-1:0]         slot_data_r;
    reg [CL_SIZE-1:0]      slot_be_r;
    reg [LINE_ADDR_W-1:0]  slot_addr_r;
    reg [NG4-1:0]          slot_rem_r;    // words still to emit (4-byte, or 8-byte on the even bits)

    wire [NG4-1:0] tok_words;
    for (genvar g = 0; g < NG4; ++g) begin : g_words
        if ((g % 2) == 0) begin : g_even
            assign tok_words[g] = st.w8 ? (| tok_be[g*4 +: 8]) : (| tok_be[g*4 +: 4]);
        end else begin : g_odd
            assign tok_words[g] = ~st.w8 && (| tok_be[g*4 +: 4]);
        end
    end

    wire [NG4_W-1:0] emit_idx;
    wire [NG4-1:0]   emit_onehot;
    VX_priority_encoder #(
        .N (NG4)
    ) emit_pe (
        .data_in    (slot_rem_r),
        .onehot_out (emit_onehot),
        .index_out  (emit_idx),
        `UNUSED_PIN (valid_out)
    );
    wire emit_last = is_store || (slot_rem_r == emit_onehot);

    wire gmem_req_ready_w;
    wire emit_ready = is_red_c ? amo_ready : gmem_req_ready_w;
    wire emit_fire  = slot_valid_r && emit_ready;
    wire slot_free  = emit_fire && emit_last;
    wire slot_take  = ~slot_valid_r || slot_free;

    wire asm_valid = ~info_empty && ~rsp_empty;
    assign asm_fire = asm_valid && (~asm_info.tok_last || slot_take);
    wire tok_done = asm_fire && asm_info.tok_last;

    always @(posedge clk) begin
        if (reset || start) begin
            acc_r        <= '0;
            acc_be_r     <= '0;
            slot_valid_r <= 1'b0;
        end else begin
            if (asm_fire) begin
                acc_r    <= tok_done ? '0 : acc_n;
                acc_be_r <= tok_done ? '0 : acc_be_n;
            end
            if (tok_done) begin
                slot_valid_r <= 1'b1;
            end else if (slot_free) begin
                slot_valid_r <= 1'b0;
            end
            if (tok_done) begin
                slot_data_r <= tok_data;
                slot_be_r   <= tok_be;
                slot_addr_r <= asm_info.cl_addr;
                slot_rem_r  <= tok_words;
            end else if (emit_fire) begin
                slot_rem_r  <= slot_rem_r & ~emit_onehot;
            end
        end
    end

    // The emitted word's bytes.
    wire [CL_OFF_BITS-1:0] w_byte = CL_OFF_BITS'(emit_idx) << 2;
    wire [CL_SIZE-1:0] w_mask = CL_SIZE'(st.w8 ? 8'hff : 8'h0f) << w_byte;

    // ════════════════════════════════════════════════════════════════════
    // Global port: writes, atomics, then the fence.
    // ════════════════════════════════════════════════════════════════════
    VX_mem_bus_if #(
        .DATA_SIZE (CL_SIZE),
        .TAG_WIDTH (GMEM_TAG_WIDTH)
    ) mem_bus_w ();

    reg                          wrote_r;
    reg                          fence_sent_r;
    reg                          fence_rsp_r;
    reg [LINE_ADDR_W-1:0]        last_line_r;
    reg [CNT_W-1:0]              amo_cnt_r;

    wire pipes_empty = info_empty && ~slot_valid_r;
    wire all_emitted = walk_done_r && pipes_empty;
    wire fence_valid = active && is_store && all_emitted && wrote_r && ~fence_sent_r;

    wire emit_g = slot_valid_r && ~is_red_c;

    mem_bus_attr_t g_attr;
    always @(*) begin
        g_attr = '0;
        if (fence_valid) begin
            g_attr.is_flush = 1'b1;
            g_attr.is_fence = 1'b1;
        end else if (is_red_g) begin
            g_attr.amo.amo_valid    = 1'b1;
            g_attr.amo.amo_op       = st.amo_op;
            g_attr.amo.amo_unsigned = st.amo_uns;
            g_attr.amo.hart_id      = st.hart_id;
        end
    end

    assign mem_bus_w.req_valid       = emit_g || fence_valid;
    assign mem_bus_w.req_data.rw     = emit_g && is_store;
    assign mem_bus_w.req_data.addr   = emit_g ? slot_addr_r : last_line_r;
    assign mem_bus_w.req_data.data   = slot_data_r;
    // A reduction word's bytes in the token: a 16-bit pair at a box edge sends its one half.
    assign mem_bus_w.req_data.byteen = emit_g ? (is_store ? slot_be_r : (w_mask & slot_be_r)) : {CL_SIZE{1'b1}};
    assign mem_bus_w.req_data.attr   = g_attr;
    assign mem_bus_w.req_data.tag.uuid  = uuid;
    assign mem_bus_w.req_data.tag.value = TAG_VALUE_W'(0);
    assign gmem_req_ready_w = mem_bus_w.req_ready;
    wire g_fire = mem_bus_w.req_valid && mem_bus_w.req_ready;
    wire fence_fire = g_fire && ~emit_g;
    wire amo_req_fire = g_fire && emit_g && is_red_g;

    assign mem_bus_w.rsp_ready = 1'b1;
    wire g_rsp = mem_bus_w.rsp_valid;
    `UNUSED_VAR (mem_bus_w.rsp_data)

    always @(posedge clk) begin
        if (reset || start) begin
            wrote_r      <= 1'b0;
            fence_sent_r <= 1'b0;
            fence_rsp_r  <= 1'b0;
            amo_cnt_r    <= '0;
        end else begin
            if (g_fire && emit_g && is_store) begin
                wrote_r     <= 1'b1;
                last_line_r <= slot_addr_r;
            end
            if (fence_fire) begin
                fence_sent_r <= 1'b1;
            end
            if (g_rsp && is_store) begin
                fence_rsp_r <= 1'b1;
            end
            amo_cnt_r <= amo_cnt_r + CNT_W'(amo_req_fire && is_red_g) - CNT_W'(g_rsp && is_red_g);
        end
    end

    VX_mem_bus_slice #(
        .DATA_SIZE   (CL_SIZE),
        .TAG_WIDTH   (GMEM_TAG_WIDTH),
        .REQ_OUT_BUF (3),
        .RSP_OUT_BUF (0)
    ) gmem_out_slice (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (mem_bus_w),
        .bus_out_if (gmem_bus_if)
    );

    // ════════════════════════════════════════════════════════════════════
    // Cluster-reduction packets: one per element, red.async's packet form
    // with its mbarrier, posted.
    // ════════════════════════════════════════════════════════════════════
    wire [DXA_SMEM_ADDR_W-1:0] e_byte = DXA_SMEM_ADDR_W'({slot_addr_r, CL_OFF_BITS'(0)}) + DXA_SMEM_ADDR_W'(w_byte);
    wire [CL_OFF_BITS-1:0] e_lane = CL_OFF_BITS'({w_byte[CL_OFF_BITS-1:3], 3'b000});
    always @(*) begin
        amo_pkt              = '0;
        amo_pkt.src          = DLMEM_CORE_W'(core_id);
        amo_pkt.op           = DLMEM_OP_DXA;
        amo_pkt.addr         = DLMEM_LMEM_AW'(e_byte >> 3);
        amo_pkt.byteen       = (st.w8 ? 8'hff : 8'h0f) << w_byte[2:0];
        amo_pkt.data         = slot_data_r[{e_lane, 3'b000} +: 64];
        amo_pkt.amo_op       = st.amo_op;
        amo_pkt.amo_unsigned = st.amo_uns;
        amo_pkt.hart_id      = st.hart_id;
        amo_pkt.mbar_valid   = 1'b1;
        amo_pkt.mbar_addr    = DLMEM_LMEM_AW'(mbar_off);
        amo_pkt.tag          = DLMEM_TAG_W'(dst_slot);
    end
    assign amo_valid = slot_valid_r && is_red_c;
    assign amo_dst   = dst_core;

    // ════════════════════════════════════════════════════════════════════
    // Completion.
    // ════════════════════════════════════════════════════════════════════
    reg rd_seen_r, wr_seen_r, evt_rd_r, evt_wr_r, done_r;
    wire rd_cond = active && walk_done_r && info_empty && ~rd_seen_r;
    wire wr_ok   = is_store ? (wrote_r ? fence_rsp_r : all_emitted)
                 : is_red_g ? (all_emitted && (amo_cnt_r == '0))
                 : all_emitted;
    wire wr_cond = active && wr_ok && ~wr_seen_r;
    wire evt_fire = evt_valid && evt_ready;

    always @(posedge clk) begin
        if (reset || start) begin
            rd_seen_r <= 1'b0;
            wr_seen_r <= 1'b0;
            evt_rd_r  <= 1'b0;
            evt_wr_r  <= 1'b0;
            done_r    <= 1'b0;
        end else begin
            if (rd_cond) rd_seen_r <= 1'b1;
            if (wr_cond) wr_seen_r <= 1'b1;
            evt_rd_r <= (evt_rd_r && ~evt_fire) || (rd_cond && bg);
            evt_wr_r <= (evt_wr_r && ~evt_fire) || (wr_cond && bg);
            if (done) done_r <= 1'b1;
        end
    end

    assign evt_valid      = evt_rd_r || evt_wr_r;
    assign evt_read_done  = evt_rd_r;
    assign evt_write_done = evt_wr_r;

    // Done once performed and the last event is taken.
    assign done = active && wr_seen_r && ~done_r && (~evt_valid || evt_fire);

    assign progress = rd_fire || asm_fire || emit_fire || fence_fire || g_rsp || tok_take;

    `RUNTIME_ASSERT (~(rsp_valid && info_empty),
        ("%t: *** %s DXA store read response without a read", $time, INSTANCE_ID))
    `RUNTIME_ASSERT (~(g_rsp && is_store && ~fence_sent_r),
        ("%t: *** %s DXA store response without a fence", $time, INSTANCE_ID))

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (rd_fire) begin
                $write("DXA_PIPE,%0d,ST_RD,addr=0x%0h,p0=%0d,n=%0d,last=%0d\n", $time, rd_addr, p_cur, n_cur, rd_last);
            end
            if (g_fire) begin
                $write("DXA_PIPE,%0d,ST_GMEM,rw=%0d,addr=0x%0h,byteen=0x%0h,fence=%0d\n", $time,
                    mem_bus_w.req_data.rw, mem_bus_w.req_data.addr, mem_bus_w.req_data.byteen, fence_fire);
            end
            if (amo_valid && amo_ready) begin
                $write("DXA_PIPE,%0d,ST_RAMO,dst=%0d,addr=0x%0h,byteen=0x%0h\n", $time, amo_dst, amo_pkt.addr, amo_pkt.byteen);
            end
        end
    end
`endif

endmodule

`endif
