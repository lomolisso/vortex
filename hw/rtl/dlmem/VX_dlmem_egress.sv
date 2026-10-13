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

// A core's DLMEM egress: serialises the remote lanes of LSU requests into
// request-network packets, one lane per cycle, and returns load/AMO data and
// store acknowledgements from the response network. A block's request is
// granted round-robin and held until its last lane issues; a store needs a
// credit, a load or AMO a pending entry that its response fills and the LSU
// delivery frees. Remote LR is sent as a load; remote SC fails locally without
// a packet, never in a cycle a response arrives (both assert unless
// LRSC_ASSERT=0). ack reports each committed store and done each load, AMO or
// SC that returned its data, both by warp, so the core's tracker sees every
// remote access performed.
//
// Deadlock freedom. Requests and responses use two separate network
// instances. The egress issue waits on a credit or a pending entry and on the
// request network; the request network waits on ingress acceptance; the
// ingress waits on a free slot and its local-memory port, which waits only on
// bank arbitration with that core's own LSU and finite DMA traffic; an ingress
// slot frees once its response enters the response network, whose sinks (this
// response port) never refuse, because every response was reserved at issue
// (a pending entry or a credit, at most STORE_CREDITS outstanding stores). By
// network liveness every response is therefore delivered, every slot frees,
// every ingress keeps accepting and the request network drains. The local
// memory never waits on a network (its ingress port is always ready for
// responses), and the LSU response path never waits on a DLMEM resource, so
// core A waiting on core B's ingress cannot close a cycle back to A.
//
// Ordering. A warp always uses one LSU block, the egress issues a block's
// requests in arrival order and a request's lanes in index order, the request
// network keeps per (source, destination) FIFO order, the ingress feeds one
// local-memory input in arrival order, and the local memory keeps each input's
// order per bank with its read-after-write stall. Same-address accesses from
// one hart therefore execute in program order; different addresses may
// complete out of order (RVWMO), and fences order them through ack and done:
// a store is acknowledged only after the bank committed it, and a load's or
// AMO's data exists only once the bank performed it.
// Responses are matched by tag, so their arrival order does not matter.
module VX_dlmem_egress import VX_gpu_pkg::*, VX_dlmem_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID       = 0,
    parameter NUM_BLOCKS    = `VX_CFG_NUM_LSU_BLOCKS,
    parameter NUM_LANES     = `VX_CFG_NUM_LSU_LANES,
    parameter STORE_CREDITS = `VX_CFG_DLMEM_STORE_CREDITS,
    parameter PENDING_SIZE  = `VX_CFG_DLMEM_PENDING_SIZE,
    parameter LRSC_ASSERT   = 1,
    parameter OUT_BUF       = 3
) (
    input wire              clk,
    input wire              reset,

    VX_lsu_mem_if.slave     lsu_in_if [NUM_BLOCKS],

    VX_dlmem_bus_if.master  req_out_if,
    VX_dlmem_bus_if.slave   rsp_in_if,

    output wire                 ack_valid,
    output wire [NW_WIDTH-1:0]  ack_wid,
    output wire                 done_valid,
    output wire [NW_WIDTH-1:0]  done_wid,

    output wire             empty
);
    localparam MY_CORE   = CORE_ID % DLMEM_NCC;
    localparam BLK_W     = `LOG2UP(NUM_BLOCKS);
    localparam LANE_W    = `LOG2UP(NUM_LANES);
    localparam PEND_W    = `LOG2UP(PENDING_SIZE);
    localparam CRED_W    = `CLOG2(STORE_CREDITS + 1);
    localparam WORD_W    = LSU_WORD_SIZE * 8;
    localparam REQ_DATAW = DLMEM_CORE_W + DLMEM_REQ_DATAW;
    localparam RSP_DATAW = NUM_LANES + NUM_LANES * WORD_W + LSU_TAG_WIDTH;

    // Remote lanes come from the core's own translation, which encodes
    // DLMEM_BASE | core << LMEM_LOG | lmem_byte; these bound that window.
    localparam [LSU_ADDR_WIDTH-1:0] WIN_BASE = LSU_ADDR_WIDTH'(`VX_MEM_DLMEM_BASE_ADDR >> DLMEM_WA);
    localparam [LSU_ADDR_WIDTH-1:0] WIN_END  = WIN_BASE + (LSU_ADDR_WIDTH'(DLMEM_NCC) << DLMEM_LMEM_AW);

    `STATIC_ASSERT (STORE_CREDITS >= 1 && PENDING_SIZE >= 1, ("invalid parameter"))
    `STATIC_ASSERT (PEND_W <= DLMEM_TAG_W && NW_WIDTH <= DLMEM_TAG_W, ("dlmem egress: packet tag too narrow"))
    `STATIC_ASSERT ((WIN_BASE & ((LSU_ADDR_WIDTH'(1) << (DLMEM_LMEM_AW + DLMEM_CORE_W)) - 1)) == 0,
        ("dlmem egress: DLMEM base not aligned to the core field"))

    // Flatten the interface array so the selected block can be indexed.

    wire [NUM_BLOCKS-1:0]                                   in_valid;
    wire [NUM_BLOCKS-1:0]                                   in_rw;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]                    in_mask;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0] in_addr;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_W-1:0]        in_data;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][LSU_WORD_SIZE-1:0] in_byteen;
    wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][MEM_ATTR_WIDTH-1:0] in_user;
    wire [NUM_BLOCKS-1:0][LSU_TAG_WIDTH-1:0]                in_tag;
    wire [NUM_BLOCKS-1:0]                                   in_ready;

    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_in
        assign in_valid[b]  = lsu_in_if[b].req_valid;
        assign in_rw[b]     = lsu_in_if[b].req_data.rw;
        assign in_mask[b]   = lsu_in_if[b].req_data.mask;
        assign in_addr[b]   = lsu_in_if[b].req_data.addr;
        assign in_data[b]   = lsu_in_if[b].req_data.data;
        assign in_byteen[b] = lsu_in_if[b].req_data.byteen;
        assign in_tag[b]    = lsu_in_if[b].req_data.tag;
        for (genvar l = 0; l < NUM_LANES; ++l) begin : g_user
            assign in_user[b][l] = MEM_ATTR_WIDTH'(lsu_in_if[b].req_data.user[l]);
        end
        assign lsu_in_if[b].req_ready = in_ready[b];
    end

    // Block selection: round-robin, held until the request's last lane issues.

    reg                  lock_r;
    reg [BLK_W-1:0]      lock_blk_r;
    reg [NUM_LANES-1:0]  rem_mask_r;

    wire [BLK_W-1:0] arb_index;
    wire             arb_valid;

    VX_generic_arbiter #(
        .NUM_REQS (NUM_BLOCKS),
        .TYPE     ("R")
    ) blk_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (in_valid),
        .grant_index  (arb_index),
        `UNUSED_PIN   (grant_onehot),
        .grant_valid  (arb_valid),
        .grant_ready  (~lock_r)
    );

    wire                 cur_valid = lock_r || arb_valid;
    wire [BLK_W-1:0]     cur_blk   = lock_r ? lock_blk_r : arb_index;
    wire [NUM_LANES-1:0] cur_mask  = lock_r ? rem_mask_r : in_mask[cur_blk];

    wire [NUM_LANES-1:0] lane_onehot;
    wire [LANE_W-1:0]    cur_lane;

    VX_priority_encoder #(
        .N (NUM_LANES)
    ) lane_sel (
        .data_in    (cur_mask),
        .onehot_out (lane_onehot),
        .index_out  (cur_lane),
        `UNUSED_PIN (valid_out)
    );

    // The selected lane.

    wire [LSU_ADDR_WIDTH-1:0] cur_addr   = in_addr[cur_blk][cur_lane];
    wire [LSU_WORD_SIZE-1:0]  cur_byteen = in_byteen[cur_blk][cur_lane];
    mem_bus_attr_t            cur_attr;
    assign cur_attr = mem_bus_attr_t'(in_user[cur_blk][cur_lane]);
    `UNUSED_VAR (cur_attr)

    wire cur_amo = cur_attr.amo.amo_valid;
    wire is_st   = in_rw[cur_blk];
    wire is_lr   = cur_amo && (cur_attr.amo.amo_op == AMO_OP_LR);
    wire is_sc   = cur_amo && (cur_attr.amo.amo_op == AMO_OP_SC);
    wire [1:0] cur_op = is_st ? DLMEM_OP_ST : ((cur_amo && ~is_lr) ? DLMEM_OP_AMO : DLMEM_OP_LD);

    wire [DLMEM_CORE_W-1:0] cur_dst = cur_addr[DLMEM_LMEM_AW +: DLMEM_CORE_W];

    wire [NW_WIDTH-1:0] cur_wid;
    if (NW_BITS != 0) begin : g_wid
        assign cur_wid = cur_attr.amo.hart_id[NT_BITS +: NW_BITS];
    end else begin : g_wid0
        assign cur_wid = '0;
    end

    // Resources.

    reg  [CRED_W-1:0]       credits_r;
    reg  [PENDING_SIZE-1:0] pend_valid_r;
    reg  [PENDING_SIZE-1:0] pend_data_valid_r;

    wire [PEND_W-1:0] free_idx;
    wire              free_valid;

    VX_priority_encoder #(
        .N (PENDING_SIZE)
    ) free_sel (
        .data_in    (~pend_valid_r),
        `UNUSED_PIN (onehot_out),
        .index_out  (free_idx),
        .valid_out  (free_valid)
    );

    wire credit_ok = (credits_r != '0);
    wire res_ok    = is_st ? credit_ok : free_valid;
    wire req_buf_ready;

    // An SC needs only its pending entry; everything else also needs the network.
    // An SC yields to an arriving response so that ack and done never coincide.
    wire pkt_valid = cur_valid && ~is_sc && res_ok;
    wire pkt_fire  = pkt_valid && req_buf_ready;
    wire sc_fire   = cur_valid && is_sc && free_valid && ~rsp_in_if.valid;
    wire lane_fire = pkt_fire || sc_fire;
    wire st_fire   = pkt_fire && is_st;
    wire alloc     = lane_fire && ~is_st;

    wire [NUM_LANES-1:0] rem_next = cur_mask & ~lane_onehot;
    wire req_done = lane_fire && (rem_next == '0);

    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_ready
        assign in_ready[b] = req_done && (cur_blk == BLK_W'(b));
    end

    always @(posedge clk) begin
        if (reset) begin
            lock_r <= 1'b0;
        end else if (lock_r) begin
            if (req_done) begin
                lock_r <= 1'b0;
            end
        end else if (arb_valid) begin
            lock_r <= ~req_done;
        end
        if (~lock_r) begin
            lock_blk_r <= arb_index;
        end
        if (lane_fire) begin
            rem_mask_r <= rem_next;
        end else if (~lock_r) begin
            rem_mask_r <= cur_mask;
        end
    end

    // Request packet.

    dlmem_req_pkt_t req_pkt;
    always @(*) begin
        req_pkt              = '0;
        req_pkt.src          = DLMEM_CORE_W'(MY_CORE);
        req_pkt.op           = cur_op;
        req_pkt.addr         = cur_addr[DLMEM_LMEM_AW-1:0];
        req_pkt.byteen       = cur_byteen;
        req_pkt.data         = in_data[cur_blk][cur_lane];
        req_pkt.amo_op       = cur_attr.amo.amo_op;
        req_pkt.amo_unsigned = cur_attr.amo.amo_unsigned;
        req_pkt.hart_id      = cur_attr.amo.hart_id;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        req_pkt.amo_cmp      = cur_attr.amo.amo_cmp;
    `endif
        req_pkt.tag          = is_st ? DLMEM_TAG_W'(cur_wid) : DLMEM_TAG_W'(free_idx);
    end

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (pkt_valid),
        .ready_in  (req_buf_ready),
        .data_in   ({cur_dst, req_pkt}),
        .data_out  ({req_out_if.dst, req_out_if.data}),
        .valid_out (req_out_if.valid),
        .ready_out (req_out_if.ready)
    );

    // An SC's outcome: 1 (failed) in the addressed 32-bit half, as the bank returns it.
    wire [WORD_W-1:0] sc_fail_word;
    if (WORD_W > 32) begin : g_sc_word64
        assign sc_fail_word = cur_byteen[0] ? WORD_W'(1) : (WORD_W'(1) << 32);
    end else begin : g_sc_word32
        assign sc_fail_word = WORD_W'(1);
    end

    // Response sink: every response was reserved at issue, so it never refuses.

    dlmem_rsp_pkt_t rsp_pkt;
    assign rsp_pkt = dlmem_rsp_pkt_t'(rsp_in_if.data);
    assign rsp_in_if.ready = 1'b1;

    wire rsp_ack  = rsp_in_if.valid && rsp_pkt.ack;
    wire rsp_fill = rsp_in_if.valid && ~rsp_pkt.ack;
    wire [PEND_W-1:0] rsp_idx = rsp_pkt.tag[PEND_W-1:0];

    reg [PENDING_SIZE-1:0][NW_WIDTH-1:0] pend_wid_r;

    reg                ack_valid_r;
    reg [NW_WIDTH-1:0] ack_wid_r;
    reg                done_valid_r;
    reg [NW_WIDTH-1:0] done_wid_r;

    always @(posedge clk) begin
        if (reset) begin
            credits_r    <= CRED_W'(STORE_CREDITS);
            ack_valid_r  <= 1'b0;
            done_valid_r <= 1'b0;
        end else begin
            credits_r    <= credits_r - CRED_W'(st_fire) + CRED_W'(rsp_ack);
            ack_valid_r  <= rsp_ack;
            done_valid_r <= rsp_fill || sc_fire;
        end
        ack_wid_r  <= rsp_pkt.tag[NW_WIDTH-1:0];
        done_wid_r <= rsp_fill ? pend_wid_r[rsp_idx] : cur_wid;
    end

    assign ack_valid  = ack_valid_r;
    assign ack_wid    = ack_wid_r;
    assign done_valid = done_valid_r;
    assign done_wid   = done_wid_r;

    // Pending table and delivery: each block returns its lowest filled entry,
    // one lane per response; the entry frees when the response is buffered.

    reg [PENDING_SIZE-1:0][BLK_W-1:0]         pend_blk_r;
    reg [PENDING_SIZE-1:0][LANE_W-1:0]        pend_lane_r;
    reg [PENDING_SIZE-1:0][LSU_TAG_WIDTH-1:0] pend_tag_r;
    reg [PENDING_SIZE-1:0][WORD_W-1:0]        pend_data_r;

    wire [NUM_BLOCKS-1:0][PENDING_SIZE-1:0] del_onehot;
    wire [NUM_BLOCKS-1:0] del_fire;
    wire [NUM_BLOCKS-1:0] rsp_buf_valid;

    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_deliver
        wire [PENDING_SIZE-1:0] cand;
        for (genvar i = 0; i < PENDING_SIZE; ++i) begin : g_cand
            assign cand[i] = pend_data_valid_r[i] && (pend_blk_r[i] == BLK_W'(b));
        end

        wire [PEND_W-1:0] del_idx;
        wire              del_valid;

        VX_priority_encoder #(
            .N (PENDING_SIZE)
        ) del_sel (
            .data_in    (cand),
            .onehot_out (del_onehot[b]),
            .index_out  (del_idx),
            .valid_out  (del_valid)
        );

        wire del_ready;
        assign del_fire[b] = del_valid && del_ready;

        wire [NUM_LANES-1:0] del_mask;
        for (genvar l = 0; l < NUM_LANES; ++l) begin : g_mask
            assign del_mask[l] = (pend_lane_r[del_idx] == LANE_W'(l));
        end

        VX_elastic_buffer #(
            .DATAW   (RSP_DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
        ) rsp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (del_valid),
            .ready_in  (del_ready),
            .data_in   ({del_mask, {NUM_LANES{pend_data_r[del_idx]}}, pend_tag_r[del_idx]}),
            .data_out  (lsu_in_if[b].rsp_data),
            .valid_out (lsu_in_if[b].rsp_valid),
            .ready_out (lsu_in_if[b].rsp_ready)
        );

        assign rsp_buf_valid[b] = lsu_in_if[b].rsp_valid;
    end

    wire [PENDING_SIZE-1:0] pend_free;
    for (genvar i = 0; i < PENDING_SIZE; ++i) begin : g_free
        wire [NUM_BLOCKS-1:0] hit;
        for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_hit
            assign hit[b] = del_fire[b] && del_onehot[b][i];
        end
        assign pend_free[i] = |hit;
    end

    for (genvar i = 0; i < PENDING_SIZE; ++i) begin : g_pend
        wire alloc_i = alloc && (free_idx == PEND_W'(i));
        wire fill_i  = rsp_fill && (rsp_idx == PEND_W'(i));
        always @(posedge clk) begin
            if (reset) begin
                pend_valid_r[i]      <= 1'b0;
                pend_data_valid_r[i] <= 1'b0;
            end else if (alloc_i) begin
                pend_valid_r[i]      <= 1'b1;
                pend_data_valid_r[i] <= sc_fire;
            end else if (fill_i) begin
                pend_data_valid_r[i] <= 1'b1;
            end else if (pend_free[i]) begin
                pend_valid_r[i]      <= 1'b0;
                pend_data_valid_r[i] <= 1'b0;
            end
            if (alloc_i) begin
                pend_blk_r[i]  <= cur_blk;
                pend_lane_r[i] <= cur_lane;
                pend_wid_r[i]  <= cur_wid;
                pend_tag_r[i]  <= in_tag[cur_blk];
                pend_data_r[i] <= sc_fail_word;
            end else if (fill_i) begin
                pend_data_r[i] <= rsp_pkt.data;
            end
        end
    end

    assign empty = ~lock_r
                && (credits_r == CRED_W'(STORE_CREDITS))
                && ~(|pend_valid_r)
                && ~req_out_if.valid
                && ~(|rsp_buf_valid)
                && ~ack_valid_r
                && ~done_valid_r;

    `UNUSED_VAR (rsp_in_if.dst)

    // Protocol checks.

    wire cur_in_window = (cur_addr >= WIN_BASE) && (cur_addr < WIN_END);

    `RUNTIME_ASSERT (~lock_r || in_valid[lock_blk_r],
        ("%t: *** %s locked LSU request withdrawn: block=%0d", $time, INSTANCE_ID, lock_blk_r))
    `RUNTIME_ASSERT (~(~lock_r && arb_valid && (in_mask[arb_index] == '0)),
        ("%t: *** %s LSU request without lanes: block=%0d", $time, INSTANCE_ID, arb_index))
    `RUNTIME_ASSERT (~(lane_fire && ~cur_in_window),
        ("%t: *** %s remote lane outside the DLMEM window: addr=0x%0h", $time, INSTANCE_ID, {cur_addr, DLMEM_WA'(0)}))
    `RUNTIME_ASSERT (~(lane_fire && (cur_dst == DLMEM_CORE_W'(MY_CORE))),
        ("%t: *** %s remote lane addressed to its own core: addr=0x%0h", $time, INSTANCE_ID, {cur_addr, DLMEM_WA'(0)}))
    `RUNTIME_ASSERT (~(cur_valid && is_st && cur_amo),
        ("%t: *** %s remote lane is both a store and an AMO", $time, INSTANCE_ID))
    `RUNTIME_ASSERT (~(LRSC_ASSERT != 0 && lane_fire && is_lr),
        ("%t: *** %s remote LR/SC is not supported: LR addr=0x%0h", $time, INSTANCE_ID, {cur_addr, DLMEM_WA'(0)}))
    `RUNTIME_ASSERT (~(LRSC_ASSERT != 0 && lane_fire && is_sc),
        ("%t: *** %s remote LR/SC is not supported: SC addr=0x%0h", $time, INSTANCE_ID, {cur_addr, DLMEM_WA'(0)}))
    `RUNTIME_ASSERT (~(rsp_ack && (credits_r == CRED_W'(STORE_CREDITS))),
        ("%t: *** %s store ack without an outstanding store: wid=%0d", $time, INSTANCE_ID, rsp_pkt.tag))
    `RUNTIME_ASSERT (~(rsp_fill && ((32'(rsp_pkt.tag) >= PENDING_SIZE) || ~pend_valid_r[rsp_idx] || pend_data_valid_r[rsp_idx])),
        ("%t: *** %s response for a free or filled pending entry: tag=%0d", $time, INSTANCE_ID, rsp_pkt.tag))
    `RUNTIME_ASSERT (~(rsp_in_if.valid && (rsp_in_if.dst != DLMEM_CORE_W'(MY_CORE))),
        ("%t: *** %s response for core %0d", $time, INSTANCE_ID, rsp_in_if.dst))

endmodule
