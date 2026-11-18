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

// One SRAM array per SM holding shared memory (rows from 0 up) and the L1's
// data slots (rows from the top down), with one read port and one write port
// over all banks (SimX SmSram, sw/common/smem_bank_ref.h SramPortRef). Every
// client takes whole rows or wavefronts; no line or fill buffer holds data.
module VX_sm_sram import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BLOCKS = `VX_CFG_NUM_LSU_BLOCKS,
    parameter ROW_TAG_W  = 1,
    localparam RD_ROWS = 3,  // TCU, DXA, DSMEM: SramRdClient order
    localparam WR_ROWS = 3   // L1 refill, DXA, DSMEM: SramWrClient order
) (
    input wire                                   clk,
    input wire                                   reset,

    input smem_cfg_t                             cfg,

    // Shared class: the LSU blocks' local legs (window word addresses).
    VX_lsu_mem_if.slave                          lsu_if [NUM_BLOCKS],

    // Global class: the L1's one in-order stream, rows resolved to slot rows.
    input  wire                                  gl_req_valid,
    input  sm_sram_req_t                         gl_req_data,
    output wire                                  gl_req_ready,
    output wire                                  gl_rsp_valid,
    output sm_sram_rsp_t                         gl_rsp_data,
    input  wire                                  gl_rsp_ready,
    output wire                                  wf_retired_valid,
    output wire [L1_SEQ_W-1:0]                   wf_retired_seq,

    // Row reads; answered SM_SRAM_BANK_STAGES + SM_SRAM_RSP_STAGES after the grant, no back-pressure.
    input  wire [RD_ROWS-1:0]                    rd_valid,
    input  wire [RD_ROWS-1:0][SM_SRAM_ROW_W-1:0] rd_row,
    input  wire [RD_ROWS-1:0][ROW_TAG_W-1:0]     rd_tag,
    output wire [RD_ROWS-1:0]                    rd_ready,
    output wire [RD_ROWS-1:0]                    rd_rsp_valid,
    output wire [SM_SRAM_ROW_BYTES*8-1:0]        rd_rsp_data,
    output wire [ROW_TAG_W-1:0]                  rd_rsp_tag,

    // Row writes; a DSMEM input is at most one SM-to-SM slot, wr_dsmem_last on its packet's last.
    input  wire [WR_ROWS-1:0]                    wr_valid,
    input  sm_sram_fill_t [WR_ROWS-1:0]          wr_data,
    input  wire                                  wr_dsmem_last,
    output wire [WR_ROWS-1:0]                    wr_ready,
    output wire                                  wr_dsmem_busy,  // DSMEM slots not yet in the array
    output wire                                  fill_done_valid,
    output wire [SM_SRAM_ROW_W-1:0]              fill_done_row,

    // cp.async lane writes, acknowledged once written; the copy unit's
    // mbarrier ops, answered from the mbarrier op queue.
    VX_lsu_mem_if.slave                          cpa_if [NUM_BLOCKS],

    // One-lane atomics (SramAmoClient order: DXA completion, DSMEM); answered
    // the cycle after their AMO-path grant, no back-pressure.
    input  wire [1:0]                            amo_valid,
    input  wire [1:0][SM_SRAM_ROW_W-1:0]         amo_row,
    input  wire [1:0][`CLOG2(`VX_CFG_LMEM_NUM_BANKS)-1:0] amo_bank,
    input  wire [1:0][LSU_WORD_SIZE-1:0]         amo_byteen,
    input  wire [1:0][LSU_WORD_SIZE*8-1:0]       amo_wdata,
    input  amo_req_t [1:0]                       amo_attr,
    input  wire [1:0][ROW_TAG_W-1:0]             amo_tag,
    output wire [1:0]                            amo_ready,
    output wire [1:0]                            amo_rsp_valid,
    output wire [1:0][LSU_WORD_SIZE*8-1:0]       amo_rsp_data,
    output wire [1:0][ROW_TAG_W-1:0]             amo_rsp_tag,

    // Shared-class lanes the DLMEM translation counted, reported the cycle
    // after they are final (SramCommit order: wavefront, LSU atomic, mbarrier op).
    output wire [2:0]                            acc_commit_valid,
    output wire [2:0][NW_WIDTH-1:0]              acc_commit_wid,
    output wire [2:0][`CLOG2(`VX_CFG_NUM_LSU_LANES+1)-1:0] acc_commit_cnt
`ifdef VX_CFG_EXT_MBAR_ENABLE
    ,
    // Per bank: a try_wait that found its phase open (flip = 0), or an
    // mbarrier op that completed the phase of the word at row (flip = 1).
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0]                    mbar_evt_valid,
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0]                    mbar_evt_flip,
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0][NW_WIDTH-1:0]      mbar_evt_wid,
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0][SM_SRAM_ROW_W-1:0] mbar_evt_row
`endif
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam BANKS      = `VX_CFG_LMEM_NUM_BANKS;
    localparam BANK_BYTES = `VX_CFG_LMEM_BANK_BYTES;
    localparam BANK_W     = BANK_BYTES * 8;
    localparam BANK_BITS  = `CLOG2(BANKS);
    localparam LANES      = `VX_CFG_NUM_LSU_LANES;
    localparam WORD       = LSU_WORD_SIZE;
    localparam HPW        = WORD / BANK_BYTES;
    localparam NH         = LANES * HPW;
    localparam ROW_W      = SM_SRAM_ROW_W;
    localparam BLK_W      = `UP(`CLOG2(NUM_BLOCKS));
    localparam CNT_L_W    = `CLOG2(LANES + 1);
    localparam MRG_BYTES  = `VX_CFG_SM_SRAM_DSMEM_WR_BYTES;
    localparam ROWB_W     = `CLOG2(SM_SRAM_ROW_BYTES + 1);
    // Output skid: the stage and the capture register each owe an entry when
    // a pick is decided; two more keep one response per cycle.
    localparam RSP_DEPTH  = 4;
    localparam RSP_DATAW  = 1 + BLK_W + LMEM_TAG_WIDTH + LANES + LANES * WORD * 8;
    localparam RD_LSU     = RD_ROWS + 1;
    localparam WR_CPA     = WR_ROWS;
    localparam WR_LSU     = WR_ROWS + 1;

    `STATIC_ASSERT(BANKS * BANK_BYTES == SM_SRAM_ROW_BYTES, ("the banks must span one array row"))
    `STATIC_ASSERT(WORD % BANK_BYTES == 0, ("a lane word must hold whole bank words"))
    `STATIC_ASSERT(MRG_BYTES >= 1 && MRG_BYTES <= SM_SRAM_ROW_BYTES, ("VX_CFG_SM_SRAM_DSMEM_WR_BYTES must be 1..one array row"))
    `STATIC_ASSERT(`VX_CFG_LMEM_LOG_SIZE >= ROW_W + SM_SRAM_BOFF_W, ("the shared-memory window must cover the array"))

    // A lane's window word address: its array row and first bank.
    localparam HPW_BITS = `CLOG2(HPW);
    `define SM_SRAM_LANE_ROW(a)  a[SM_SRAM_BOFF_W - `CLOG2(WORD) +: ROW_W]
    `define SM_SRAM_LANE_BANK(a) BANK_BITS'(a << HPW_BITS)

    // ---------------------------------------------------------------------
    // Requests in the stage's lane form.

    typedef struct packed {
        logic                     cls;
        logic                     rw;
        logic [BLK_W-1:0]         blk;
        logic [LMEM_TAG_WIDTH-1:0] tag;
        logic [L1_SEQ_W-1:0]      seq;
        logic [LANES-1:0]         mask;
        logic [LANES-1:0][ROW_W-1:0] row;
        logic [LANES-1:0][BANK_BITS-1:0] bank;
        logic [LANES-1:0][WORD-1:0] byteen;
        logic [LANES-1:0][WORD*8-1:0] data;
        logic                     amo;    // on the AMO path, not the ports
        logic                     mbar;   // through the mbarrier op queue
        logic                     cas;    // its comparands follow as a second beat
        logic                     cpa;    // a copy unit's mbarrier op: answered on cpa_if
        amo_req_t [LANES-1:0]     attr;
        logic [CNT_L_W-1:0]       lacc;   // counted lanes
        logic [NW_WIDTH-1:0]      wid;
    } wf_item_t;

    // The comparand reaches the stage only as a second beat's data.
    function automatic amo_req_t attr_no_cmp(input amo_req_t a);
        attr_no_cmp = a;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        attr_no_cmp.amo_cmp = '0;
    `endif
    endfunction

    // Shared requests, round robin over the blocks.
    wire [NUM_BLOCKS-1:0] sh_valid;
    wf_item_t [NUM_BLOCKS-1:0] sh_item;

    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_sh_in
        wire [LANES-1:0] lane_on = lsu_if[b].req_data.mask;

        assign sh_valid[b]       = lsu_if[b].req_valid;
        assign sh_item[b].cls    = SM_SRAM_CLS_SHARED;
        assign sh_item[b].rw     = lsu_if[b].req_data.rw;
        assign sh_item[b].blk    = BLK_W'(b);
        assign sh_item[b].tag    = LMEM_TAG_WIDTH'(lsu_if[b].req_data.tag);
        assign sh_item[b].seq    = '0;
        assign sh_item[b].mask   = lane_on;
        assign sh_item[b].byteen = lsu_if[b].req_data.byteen;
        assign sh_item[b].data   = lsu_if[b].req_data.data;
        wire [LANES-1:0] lane_amo;
        wire [LANES-1:0] lane_mbar;
        wire [LANES-1:0] lane_cas;
        wire [LANES-1:0] lane_lacc;
        for (genvar i = 0; i < LANES; ++i) begin : g_lanes
            mem_bus_attr_t lane_attr;
            assign lane_attr = mem_bus_attr_t'(lsu_if[b].req_data.user[i]);
        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            assign lane_lacc[i] = lane_on[i] && lane_attr.is_lacc;
        `else
            assign lane_lacc[i] = 1'b0;
        `endif
            assign sh_item[b].row[i]  = `SM_SRAM_LANE_ROW(lsu_if[b].req_data.addr[i]);
            assign sh_item[b].bank[i] = `SM_SRAM_LANE_BANK(lsu_if[b].req_data.addr[i]);
            assign lane_amo[i]  = lane_on[i] && lane_attr.amo.amo_valid;
            assign lane_cas[i]  = lane_amo[i] && (lane_attr.amo.amo_op == AMO_OP_CAS);
            assign sh_item[b].attr[i] = attr_no_cmp(lane_attr.amo);
        `ifdef VX_CFG_EXT_MBAR_ENABLE
            assign lane_mbar[i] = lane_amo[i] && (lane_attr.amo.amo_op >= AMO_OP_MBAR_ARRIVE);
        `else
            assign lane_mbar[i] = 1'b0;
        `endif
            `UNUSED_VAR (lane_attr)
        end
        assign sh_item[b].amo  = (|lane_amo) && ~(|lane_mbar);
        assign sh_item[b].mbar = (|lane_mbar);
        assign sh_item[b].cas  = (|lane_cas);
        assign sh_item[b].cpa  = 1'b0;
        assign sh_item[b].lacc = CNT_L_W'($countones(lane_lacc));
        reg [NW_WIDTH-1:0] first_wid;
        always @(*) begin
            first_wid = '0;
            for (integer i = LANES - 1; i >= 0; --i) begin
                if (lane_on[i])
                    first_wid = NW_WIDTH'(sh_item[b].attr[i].hart_id >> NT_BITS);
            end
        end
        assign sh_item[b].wid = first_wid;
    end

    wire [BLK_W-1:0]      sh_idx;
    wire [NUM_BLOCKS-1:0] sh_onehot;
    wire                  sh_any;
    wire                  sh_take;

    VX_rr_arbiter #(
        .NUM_REQS (NUM_BLOCKS)
    ) sh_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (sh_valid),
        .grant_index  (sh_idx),
        .grant_onehot (sh_onehot),
        .grant_valid  (sh_any),
        .grant_ready  (sh_take)
    );

    // Global requests from the L1.
    wf_item_t gl_item;
    assign gl_item.cls    = SM_SRAM_CLS_GLOBAL;
    assign gl_item.rw     = (gl_req_data.op == SM_SRAM_OP_WRITE);
    assign gl_item.blk    = '0;
    assign gl_item.tag    = gl_req_data.tag;
    assign gl_item.seq    = gl_req_data.seq;
    assign gl_item.mask   = gl_req_data.mask;
    assign gl_item.amo    = 1'b0;
    assign gl_item.mbar   = 1'b0;
    assign gl_item.cas    = 1'b0;
    assign gl_item.cpa    = 1'b0;
    assign gl_item.attr   = '0;
    assign gl_item.lacc   = '0;
    assign gl_item.wid    = '0;
    for (genvar i = 0; i < LANES; ++i) begin : g_gl_lanes
        assign gl_item.row[i]    = gl_req_data.lanes[i].row;
        assign gl_item.bank[i]   = gl_req_data.lanes[i].boff[`CLOG2(BANK_BYTES) +: BANK_BITS];
        assign gl_item.byteen[i] = gl_req_data.lanes[i].byteen;
        assign gl_item.data[i]   = gl_req_data.lanes[i].data;
    end
    `UNUSED_VAR (gl_req_data.cls)
    `UNUSED_VAR (gl_req_data.amo)

    // ---------------------------------------------------------------------
    // Wavefront engines: 0 = the LSU stage, 1 = the cp.async stage. A bank
    // serves per grant the row of its first pending bank word, every pending
    // word of that row with it, so an item takes sram_row_wavefronts grants.

    wire [1:0][NH-1:0]                  eng_rem;
    wire [1:0][NH-1:0][ROW_W-1:0]       eng_row;
    wire [1:0][NH-1:0][BANK_BITS-1:0]   eng_bank;
    wire [1:0][NH-1:0][BANK_BYTES-1:0]  eng_be;
    wire [1:0][NH-1:0][BANK_W-1:0]      eng_data;
    reg  [1:0][BANKS-1:0]               eng_sel;
    reg  [1:0][BANKS-1:0][ROW_W-1:0]    eng_sel_row;
    reg  [1:0][NH-1:0]                  eng_served;
    reg  [1:0][BANKS-1:0][BANK_BYTES-1:0] eng_wren;
    reg  [1:0][BANKS-1:0][BANK_W-1:0]   eng_wdata;

    always @(*) begin
        for (integer e = 0; e < 2; ++e) begin
            for (integer b = 0; b < BANKS; ++b) begin
                eng_sel[e][b]     = 1'b0;
                eng_sel_row[e][b] = '0;
                for (integer h = NH - 1; h >= 0; --h) begin
                    if (eng_rem[e][h] && eng_bank[e][h] == BANK_BITS'(b)) begin
                        eng_sel[e][b]     = 1'b1;
                        eng_sel_row[e][b] = eng_row[e][h];
                    end
                end
            end
            for (integer h = 0; h < NH; ++h) begin
                eng_served[e][h] = eng_rem[e][h] && (eng_sel_row[e][eng_bank[e][h]] == eng_row[e][h]);
            end
            for (integer b = 0; b < BANKS; ++b) begin
                eng_wren[e][b]  = '0;
                eng_wdata[e][b] = '0;
                for (integer h = 0; h < NH; ++h) begin
                    if (eng_served[e][h] && eng_bank[e][h] == BANK_BITS'(b)) begin
                        for (integer k = 0; k < BANK_BYTES; ++k) begin
                            if (eng_be[e][h][k]) begin
                                eng_wren[e][b][k] = 1'b1;
                                eng_wdata[e][b][k*8 +: 8] = eng_data[e][h][k*8 +: 8];
                            end
                        end
                    end
                end
            end
        end
    end

    // ---------------------------------------------------------------------
    // The LSU wavefront stage.

    reg                stg_valid;
    wf_item_t          stg_r;
    reg [NH-1:0]       stg_rem_r;

    wire sh_rsp_room, gl_rsp_room;
    wire mbq_take_ok;
    wire [1:0] wf_req  = {gl_req_valid && gl_rsp_room,
                          sh_any && sh_rsp_room && (~sh_item[sh_idx].mbar || mbq_take_ok)};
    wire [1:0][SM_SRAM_PRI_W-1:0] wf_lvl = {SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_WF_PRI_GLOBAL), SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_WF_PRI_SHARED)};
    wire       wf_pick_idx;
    wire       wf_pick_valid;

    VX_level_arbiter #(
        .NUM_REQS (2),
        .LEVEL_W  (SM_SRAM_PRI_W)
    ) wf_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (stg_valid ? 2'b00 : wf_req),
        .levels       (wf_lvl),
        .grant_index  (wf_pick_idx),
        `UNUSED_PIN   (grant_onehot),
        .grant_valid  (wf_pick_valid),
        .grant_ready  (1'b1)
    );

    wire      pick = ~stg_valid && wf_pick_valid;
    wf_item_t pick_item;
    assign    pick_item = wf_pick_idx ? gl_item : sh_item[sh_idx];
    assign    sh_take    = pick && ~wf_pick_idx;
    assign    gl_req_ready = pick && wf_pick_idx;
    // A compare-and-swap crosses the 128-B LSU path twice: the picked beat
    // brings its swap values, the block's next beat its comparands; its lanes
    // reach the AMO path once the comparands are in.
    wire cmp_take;
    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_sh_ready
        assign lsu_if[b].req_ready = (sh_take && sh_onehot[b]) || (cmp_take && (stg_r.blk == BLK_W'(b)));
    end

    wire      cur_valid = stg_valid || pick;
    wf_item_t cur;
    assign    cur = stg_valid ? stg_r : pick_item;

    reg                          cmp_have_r;
    reg [LANES-1:0][WORD*8-1:0]  cmp_r;
    wire [NUM_BLOCKS-1:0]        cmp_beat_v;
    wire [NUM_BLOCKS-1:0][LANES-1:0][WORD*8-1:0] cmp_beat_d;
    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_cmp_beat
        assign cmp_beat_v[b] = lsu_if[b].req_valid;
        assign cmp_beat_d[b] = lsu_if[b].req_data.data;
    end
    assign cmp_take = stg_valid && stg_r.cas && ~cmp_have_r && cmp_beat_v[stg_r.blk];
    wire cmp_in = ~cur.cas || cmp_have_r || cmp_take;
    wire [LANES-1:0][WORD*8-1:0] cmp_cur = cmp_have_r ? cmp_r : cmp_beat_d[stg_r.blk];
    wire [NH-1:0] cur_rem;
    for (genvar i = 0; i < LANES; ++i) begin : g_cur_lanes
        for (genvar k = 0; k < HPW; ++k) begin : g_half
            localparam h = i * HPW + k;
            wire touched = cur.mask[i] && (|cur.byteen[i][k * BANK_BYTES +: BANK_BYTES]);
            assign cur_rem[h]     = stg_valid ? stg_rem_r[h] : touched;
            assign eng_row[0][h]  = cur.row[i];
            assign eng_bank[0][h] = cur.bank[i] + BANK_BITS'(k);
            assign eng_be[0][h]   = cur.byteen[i][k * BANK_BYTES +: BANK_BYTES];
            assign eng_data[0][h] = cur.data[i][k * BANK_W +: BANK_W];
        end
    end
    assign eng_rem[0] = cur_valid ? cur_rem : '0;

    // ---------------------------------------------------------------------
    // Read port and write port.

    wire [RD_LSU:0] rd_req;
    assign rd_req[RD_ROWS-1:0] = rd_valid;
    wire   amo_rd_req;
    assign rd_req[RD_ROWS]     = amo_rd_req;
    assign rd_req[RD_LSU]      = cur_valid && ~cur.rw && ~cur.amo && ~cur.mbar;
    wire [RD_LSU:0][SM_SRAM_PRI_W-1:0] rd_lvl = {
        SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_RD_PRI_LSU),
        SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_RD_PRI_AMO),
        SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_RD_PRI_DSMEM),
        SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_RD_PRI_DXA),
        SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_RD_PRI_TCU)};
    wire [`CLOG2(RD_LSU+1)-1:0] rd_gnt_idx;
    wire [RD_LSU:0]             rd_gnt;

    VX_level_arbiter #(
        .NUM_REQS (RD_LSU + 1),
        .LEVEL_W  (SM_SRAM_PRI_W)
    ) rd_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (rd_req),
        .levels       (rd_lvl),
        .grant_index  (rd_gnt_idx),
        .grant_onehot (rd_gnt),
        `UNUSED_PIN   (grant_valid),
        .grant_ready  (1'b1)
    );
    assign rd_ready = rd_gnt[RD_ROWS-1:0];

    // DSMEM inputs of one row merge into writes of up to MRG_BYTES; an input
    // is one lane's message or a whole slot, so the merge counts bytes.
    reg                    mrg_valid, mrg_open;
    reg [ROW_W-1:0]        mrg_row;
    reg [SM_SRAM_ROW_BYTES-1:0]   mrg_be;
    reg [SM_SRAM_ROW_BYTES*8-1:0] mrg_data;
    reg [ROWB_W-1:0]       mrg_bytes;
    wire [ROWB_W-1:0]      mrg_in_bytes;

    // cp.async stage.
    wire [NUM_BLOCKS-1:0] cpa_valid;
    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_cpa_valid
        assign cpa_valid[b] = cpa_if[b].req_valid;
    end
    wire [BLK_W-1:0]      cpa_idx;
    wire [NUM_BLOCKS-1:0] cpa_onehot;
    wire                  cpa_any;
    reg                   cpa_stg_valid;
    reg  [BLK_W-1:0]      cpa_blk_r;
    reg  [NH-1:0]         cpa_rem_r;
    reg  [LANES-1:0][ROW_W-1:0]     cpa_row_r;
    reg  [LANES-1:0][BANK_BITS-1:0] cpa_bank_r;
    reg  [LANES-1:0][WORD-1:0]      cpa_be_r;
    reg  [LANES-1:0][WORD*8-1:0]    cpa_data_r;
    reg  [LANES-1:0]                cpa_mask_r;
    reg  [LSU_TAG_WIDTH-1:0]        cpa_tag_r;
    reg                   ack_valid_r;
    reg  [BLK_W-1:0]      ack_blk_r;
    reg  [LANES-1:0]      ack_mask_r;
    reg  [LSU_TAG_WIDTH-1:0] ack_tag_r;
    wire                  ack_pop;

    // A copy unit's mbarrier op (cp.async.mbarrier.arrive) goes to the
    // mbarrier op queue instead of the stage, ahead of the LSU's: the warps'
    // phase checks wait for it.
    wf_item_t [NUM_BLOCKS-1:0] cpa_item;
    wire                  cpa_mb;
    wire                  cpa_take;
    wire                  mbq_full_w;

    VX_rr_arbiter #(
        .NUM_REQS (NUM_BLOCKS)
    ) cpa_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (cpa_valid),
        .grant_index  (cpa_idx),
        .grant_onehot (cpa_onehot),
        .grant_valid  (cpa_any),
        .grant_ready  (cpa_take)
    );
    assign cpa_mb   = ~cpa_stg_valid && cpa_any && cpa_item[cpa_idx].mbar;
    assign cpa_take = ~cpa_stg_valid && (~cpa_item[cpa_idx].mbar || ~mbq_full_w);
    wire [NUM_BLOCKS-1:0][LANES-1:0]                cpa_in_mask;
    wire [NUM_BLOCKS-1:0][LSU_TAG_WIDTH-1:0]        cpa_in_tag;
    wire [NUM_BLOCKS-1:0][LANES-1:0][WORD-1:0]      cpa_in_be;
    wire [NUM_BLOCKS-1:0][LANES-1:0][WORD*8-1:0]    cpa_in_data;
    wire [NUM_BLOCKS-1:0][LANES-1:0][ROW_W-1:0]     cpa_in_row;
    wire [NUM_BLOCKS-1:0][LANES-1:0][BANK_BITS-1:0] cpa_in_bank;
    wire [NUM_BLOCKS-1:0]                           cpa_in_rsp_ready;
    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_cpa_in
        assign cpa_if[b].req_ready  = cpa_take && cpa_onehot[b];
        assign cpa_in_mask[b]       = cpa_if[b].req_data.mask;
        assign cpa_in_tag[b]        = LSU_TAG_WIDTH'(cpa_if[b].req_data.tag);
        assign cpa_in_be[b]         = cpa_if[b].req_data.byteen;
        assign cpa_in_data[b]       = cpa_if[b].req_data.data;
        assign cpa_in_rsp_ready[b]  = cpa_if[b].rsp_ready;
        for (genvar i = 0; i < LANES; ++i) begin : g_lanes
            assign cpa_in_row[b][i]  = `SM_SRAM_LANE_ROW(cpa_if[b].req_data.addr[i]);
            assign cpa_in_bank[b][i] = `SM_SRAM_LANE_BANK(cpa_if[b].req_data.addr[i]);
        end
        `UNUSED_VAR (cpa_if[b].req_data.rw)

        wire [LANES-1:0] lane_mbar;
        for (genvar i = 0; i < LANES; ++i) begin : g_attr
            mem_bus_attr_t lane_attr;
            assign lane_attr = mem_bus_attr_t'(cpa_if[b].req_data.user[i]);
        `ifdef VX_CFG_EXT_MBAR_ENABLE
            assign lane_mbar[i] = cpa_in_mask[b][i] && lane_attr.amo.amo_valid && (lane_attr.amo.amo_op >= AMO_OP_MBAR_ARRIVE);
        `else
            assign lane_mbar[i] = 1'b0;
        `endif
            assign cpa_item[b].attr[i] = attr_no_cmp(lane_attr.amo);
            `UNUSED_VAR (lane_attr)
        end
        assign cpa_item[b].cls    = SM_SRAM_CLS_SHARED;
        assign cpa_item[b].rw     = 1'b0;
        assign cpa_item[b].blk    = BLK_W'(b);
        assign cpa_item[b].tag    = LMEM_TAG_WIDTH'(cpa_in_tag[b]);
        assign cpa_item[b].seq    = '0;
        assign cpa_item[b].mask   = cpa_in_mask[b];
        assign cpa_item[b].row    = cpa_in_row[b];
        assign cpa_item[b].bank   = cpa_in_bank[b];
        assign cpa_item[b].byteen = cpa_in_be[b];
        assign cpa_item[b].data   = cpa_in_data[b];
        assign cpa_item[b].amo    = 1'b0;
        assign cpa_item[b].mbar   = (|lane_mbar);
        assign cpa_item[b].cas    = 1'b0;
        assign cpa_item[b].cpa    = 1'b1;
        assign cpa_item[b].lacc   = '0;
        assign cpa_item[b].wid    = '0;
    end

    for (genvar i = 0; i < LANES; ++i) begin : g_cpa_lanes
        for (genvar k = 0; k < HPW; ++k) begin : g_half
            localparam h = i * HPW + k;
            assign eng_row[1][h]  = cpa_row_r[i];
            assign eng_bank[1][h] = cpa_bank_r[i] + BANK_BITS'(k);
            assign eng_be[1][h]   = cpa_be_r[i][k * BANK_BYTES +: BANK_BYTES];
            assign eng_data[1][h] = cpa_data_r[i][k * BANK_W +: BANK_W];
        end
    end
    assign eng_rem[1] = cpa_stg_valid ? cpa_rem_r : '0;

    wire [WR_LSU:0] wr_req;
    assign wr_req[0]      = wr_valid[0];
    assign wr_req[1]      = wr_valid[1];
    assign wr_req[2]      = mrg_valid && ~mrg_open;
    assign wr_req[WR_CPA] = cpa_stg_valid && (|cpa_rem_r) && ~ack_valid_r;
    assign wr_req[WR_LSU] = cur_valid && cur.rw && ~cur.amo && ~cur.mbar && (|cur_rem);
    wire amo_wb_now;
    wire [WR_LSU:0][SM_SRAM_PRI_W-1:0] wr_lvl = {
        SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_WR_PRI_LSU),
        SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_WR_PRI_CPASYNC),
        SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_WR_PRI_DSMEM),
        SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_WR_PRI_DXA),
        SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_WR_PRI_L1FILL)};
    wire [`CLOG2(WR_LSU+1)-1:0] wr_gnt_idx;
    wire [WR_LSU:0]             wr_gnt;

    VX_level_arbiter #(
        .NUM_REQS (WR_LSU + 1),
        .LEVEL_W  (SM_SRAM_PRI_W)
    ) wr_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (amo_wb_now ? '0 : wr_req),
        .levels       (wr_lvl),
        .grant_index  (wr_gnt_idx),
        .grant_onehot (wr_gnt),
        `UNUSED_PIN   (grant_valid),
        .grant_ready  (1'b1)
    );

    VX_popcount #(
        .N (SM_SRAM_ROW_BYTES)
    ) mrg_in_count (
        .data_in  (wr_data[2].byteen),
        .data_out (mrg_in_bytes)
    );
    wire [ROWB_W:0] mrg_sum = (ROWB_W+1)'(mrg_bytes) + (ROWB_W+1)'(mrg_in_bytes);
    wire mrg_fit  = mrg_valid && mrg_open && (wr_data[2].row == mrg_row) && (mrg_sum <= (ROWB_W+1)'(MRG_BYTES));
    wire [ROWB_W-1:0] mrg_bytes_n = mrg_fit ? ROWB_W'(mrg_sum) : mrg_in_bytes;
    wire mrg_free = ~mrg_valid || wr_gnt[2];
    assign wr_ready[0] = wr_gnt[0];
    assign wr_ready[1] = wr_gnt[1];
    assign wr_ready[2] = mrg_fit || mrg_free;
    wire mrg_in = wr_valid[2] && wr_ready[2];
    assign wr_dsmem_busy = mrg_valid;
    `UNUSED_VAR (wr_data[2].src)
    `UNUSED_VAR (wr_data[1].src)
    `UNUSED_VAR (wr_data[0].src)

    always @(posedge clk) begin
        if (reset) begin
            mrg_valid <= 1'b0;
            mrg_open  <= 1'b0;
        end else if (mrg_in) begin
            mrg_valid <= 1'b1;
            mrg_bytes <= mrg_bytes_n;
            mrg_open  <= ~wr_dsmem_last && (mrg_bytes_n < ROWB_W'(MRG_BYTES));
            if (mrg_fit) begin
                mrg_be    <= mrg_be | wr_data[2].byteen;
                for (integer k = 0; k < SM_SRAM_ROW_BYTES; ++k) begin
                    if (wr_data[2].byteen[k])
                        mrg_data[k*8 +: 8] <= wr_data[2].data[k*8 +: 8];
                end
            end else begin
                mrg_row   <= wr_data[2].row;
                mrg_be    <= wr_data[2].byteen;
                mrg_data  <= wr_data[2].data;
            end
        end else if (wr_gnt[2]) begin
            mrg_valid <= 1'b0;
        end else if (mrg_valid && mrg_open) begin
            // A run of inputs ends at the first cycle with no input that fits.
            mrg_open <= 1'b0;
        end
    end

    assign fill_done_valid = wr_gnt[0];
    assign fill_done_row   = wr_data[0].row;

    // ---------------------------------------------------------------------
    // AMO path (SramPortRef): per bank pair the winner among the clients'
    // head ops (DXA, DSMEM, mbarrier queue, LSU) by the AMO_PRI levels, kept
    // until granted; the winners together are one RD_AMO read. Granted, a
    // pair updates per bank the first of the winner's remaining lanes there
    // (one lane when the two firsts differ and one is 8 B: the earlier). The
    // ALU works on the bank output the next cycle (C) and the write-back takes
    // the write port first SM_SRAM_AMO_STAGES after the grant (W).

    wire [BANKS-1:0][BANK_W-1:0]     bk_ram;
    wire [BANKS-1:0][BANK_W-1:0]     bk_rdata;
    wire [BANKS-1:0][ROW_W-1:0]      bw_row;   // this cycle's client write, AMO write-backs excluded
    wire [BANKS-1:0][BANK_BYTES-1:0] bw_be;
    wire [BANKS-1:0][ROW_W-1:0]      rrow_r;   // each bank's last read row
    wire [BANKS-1:0]                 aw_valid;
    wire [BANKS-1:0][ROW_W-1:0]      aw_row;
    wire [BANKS-1:0][BANK_BYTES-1:0] aw_be;
    wire [BANKS-1:0][BANK_W-1:0]     aw_data;
    wire cur_done;

    wire lsu_amo_left;

    localparam PAIRS  = BANKS / 2;
    localparam NAC    = 4;  // SramAmoClient: DXA, DSMEM, MBAR, LSU
    localparam AC_W   = 1;
    localparam LANE_W = `UP(`CLOG2(LANES));

    `STATIC_ASSERT(HPW == 2, ("the AMO units sit on even/odd bank pairs of one lane word"))
    `STATIC_ASSERT(SM_SRAM_AMO_STAGES == 2, ("the AMO path's forwarding covers a two-stage write-back"))

    // Head op of each client, per lane: passes left, first bank, both banks, row, operands.
    wire [NAC-1:0][LANES-1:0][AC_W-1:0]      ah_cnt;
    wire [NAC-1:0][LANES-1:0][BANK_BITS-1:0] ah_fbank;
    wire [NAC-1:0][LANES-1:0]                ah_wide;
    wire [NAC-1:0][LANES-1:0][ROW_W-1:0]     ah_row;
    wire [NAC-1:0][LANES-1:0][WORD*8-1:0]    ah_data;
    wire [NAC-1:0][LANES-1:0][WORD*8-1:0]    ah_cmp;   // at the data's position in the word
    amo_req_t [NAC-1:0][LANES-1:0]           ah_attr;


    // DXA and DSMEM: one lane, the input itself until granted.
    for (genvar a = 0; a < 2; ++a) begin : g_ah_one
        for (genvar i = 0; i < LANES; ++i) begin : g_lanes
            if (i == 0) begin : g_l0
                wire hi_only = ~(|amo_byteen[a][BANK_BYTES-1:0]);
                assign ah_cnt[a][i]   = amo_valid[a] ? AC_W'(1) : '0;
                assign ah_fbank[a][i] = amo_bank[a] + BANK_BITS'(hi_only);
                assign ah_wide[a][i]  = (&amo_byteen[a]);
                assign ah_row[a][i]   = amo_row[a];
                assign ah_data[a][i]  = amo_wdata[a];
                assign ah_attr[a][i]  = amo_attr[a];
            `ifdef VX_CFG_EXT_ZACAS_ENABLE
                assign ah_cmp[a][i]   = hi_only ? {amo_attr[a].amo_cmp[BANK_W-1:0], BANK_W'(0)} : (WORD*8)'(amo_attr[a].amo_cmp);
            `else
                assign ah_cmp[a][i]   = '0;
            `endif
            end else begin : g_ln
                assign ah_cnt[a][i]   = '0;
                assign ah_fbank[a][i] = '0;
                assign ah_wide[a][i]  = 1'b0;
                assign ah_row[a][i]   = '0;
                assign ah_data[a][i]  = '0;
                assign ah_attr[a][i]  = '0;
                assign ah_cmp[a][i]   = '0;
            end
        end
    end

    // mbarrier op queue: an op leaves the wavefront stage in one cycle and
    // waits here for the AMO path, so loads behind it are not held by its passes.
    // Two entries cover the push-to-pop turnaround of the registered full flag.
    localparam MBQ_DEPTH = `VX_CFG_MBAR_OP_QUEUE;
    // Response skid: the head goes to the AMO path only while the op in the
    // finish register and the head itself still fit.
    localparam MRQ_DEPTH = 4;
    wire                 mbq_push, mbq_pop, mbq_empty, mbq_full;
    wf_item_t            mbq_head;
    wire                 mrq_room;
    `UNUSED_VAR (mbq_head)

    VX_fifo_queue #(
        .DATAW ($bits(wf_item_t)),
        .DEPTH (MBQ_DEPTH)
    ) mbq (
        .clk      (clk),
        .reset    (reset),
        .push     (mbq_push),
        .pop      (mbq_pop),
        .data_in  (cpa_mb ? cpa_item[cpa_idx] : cur),
        .data_out (mbq_head),
        .empty    (mbq_empty),
        `UNUSED_PIN (alm_empty),
        .full     (mbq_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    assign mbq_full_w  = mbq_full;
    assign mbq_take_ok = ~mbq_full && ~cpa_mb;
    assign mbq_push    = (pick && ~wf_pick_idx && pick_item.mbar) || (cpa_mb && ~mbq_full);
    wire   mbq_go      = ~mbq_empty && mrq_room;

    // Passes taken so far by the stage's LSU atomic and by the queue's head.
    reg  [LANES-1:0][AC_W-1:0] lsu_used_r, mbq_used_r;
    for (genvar i = 0; i < LANES; ++i) begin : g_ah_lanes
        wire [AC_W-1:0] lsu_need = AC_W'(cur_valid && cur.amo && cmp_in && cur.mask[i] && cur.attr[i].amo_valid);
        wire [AC_W-1:0] mbq_need = AC_W'(mbq_go && mbq_head.mask[i] && mbq_head.attr[i].amo_valid);
        wire [AC_W-1:0] lsu_used = stg_valid ? lsu_used_r[i] : '0;
        assign ah_cnt[2][i]   = mbq_need - mbq_used_r[i];
        assign ah_cnt[3][i]   = lsu_need - lsu_used;
        assign ah_fbank[2][i] = mbq_head.bank[i] + BANK_BITS'(mbq_head.byteen[i][BANK_BYTES-1:0] == '0);
        assign ah_fbank[3][i] = cur.bank[i] + BANK_BITS'(cur.byteen[i][BANK_BYTES-1:0] == '0);
        assign ah_wide[2][i]  = (&mbq_head.byteen[i]);
        assign ah_wide[3][i]  = (&cur.byteen[i]);
        assign ah_row[2][i]   = mbq_head.row[i];
        assign ah_row[3][i]   = cur.row[i];
        assign ah_data[2][i]  = mbq_head.data[i];
        assign ah_data[3][i]  = cur.data[i];
        assign ah_attr[2][i]  = mbq_head.attr[i];
        assign ah_attr[3][i]  = cur.attr[i];
        assign ah_cmp[2][i]   = '0;
        assign ah_cmp[3][i]   = cmp_cur[i];
    end

    // Phase checks of one word read it once, as a broadcast load does: the lowest
    // pending check leads, the checks of the same word, op and token phase ride on it.
    wire [LANES-1:0]  mb_ride;
    wire [LANE_W-1:0] mb_lead;
    wire              mb_lead_v;
`ifdef VX_CFG_EXT_MBAR_ENABLE
    wire [LANES-1:0] mb_chk;
    for (genvar i = 0; i < LANES; ++i) begin : g_mb_chk
        assign mb_chk[i] = (ah_cnt[2][i] != '0)
                        && (ah_attr[2][i].amo_op == AMO_OP_MBAR_TEST_WAIT || ah_attr[2][i].amo_op == AMO_OP_MBAR_TRY_WAIT);
    end
    VX_priority_encoder #(.N (LANES)) mb_lead_enc (
        .data_in (mb_chk), .index_out (mb_lead), `UNUSED_PIN (onehot_out), .valid_out (mb_lead_v)
    );
    for (genvar i = 0; i < LANES; ++i) begin : g_mb_ride
        assign mb_ride[i] = mb_lead_v && mb_chk[i] && (LANE_W'(i) != mb_lead)
                         && (ah_attr[2][i].amo_op == ah_attr[2][mb_lead].amo_op)
                         && (ah_row[2][i] == ah_row[2][mb_lead])
                         && (ah_fbank[2][i] == ah_fbank[2][mb_lead])
                         && (ah_wide[2][i] == ah_wide[2][mb_lead])
                         && (ah_data[2][i][WORD*8-1] == ah_data[2][mb_lead][WORD*8-1]);
    end
`else
    assign mb_ride   = '0;
    assign mb_lead   = '0;
    assign mb_lead_v = 1'b0;
`endif

    // Per pair: the held winner and this cycle's lanes on its two banks.
    reg  [PAIRS-1:0]                 hold_v_r;
    reg  [PAIRS-1:0][1:0]            hold_c_r;
    wire [PAIRS-1:0]                 pr_v;
    wire [PAIRS-1:0][1:0]            pr_c;
    wire [PAIRS-1:0]                 pr_ev, pr_ov;     // a lane on the even / odd bank this grant
    wire [PAIRS-1:0][LANE_W-1:0]     pr_e, pr_o;
    wire amo_gnt = rd_gnt[RD_ROWS];
    assign amo_rd_req = (|pr_v);

    wire [NAC-1:0][LANES-1:0] tk;  // lanes taking a pass in this grant
    wire [PAIRS-1:0][NAC-1:0][LANES-1:0] tk_p;

    for (genvar p = 0; p < PAIRS; ++p) begin : g_pair
        wire [NAC-1:0][LANES-1:0] on_e, on_o;
        wire [NAC-1:0] has;
        for (genvar a = 0; a < NAC; ++a) begin : g_cl
            for (genvar i = 0; i < LANES; ++i) begin : g_ln
                wire live = (ah_cnt[a][i] != '0) && ~((a == 2) && mb_ride[i]);
                assign on_e[a][i] = live && (ah_fbank[a][i] == BANK_BITS'(2 * p));
                assign on_o[a][i] = live && ((ah_fbank[a][i] == BANK_BITS'(2 * p + 1))
                                          || (ah_wide[a][i] && ah_fbank[a][i] == BANK_BITS'(2 * p)));
            end
            assign has[a] = (|on_e[a]) || (|on_o[a]);
        end
        wire [1:0] pk_idx;
        wire       pk_v;
        VX_level_arbiter #(
            .NUM_REQS (NAC),
            .LEVEL_W  (SM_SRAM_PRI_W)
        ) pair_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (hold_v_r[p] ? '0 : has),
            .levels       ({SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_AMO_PRI_LSU), SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_AMO_PRI_MBAR),
                            SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_AMO_PRI_DSMEM), SM_SRAM_PRI_W'(`VX_CFG_SM_SRAM_AMO_PRI_DXA)}),
            .grant_index  (pk_idx),
            `UNUSED_PIN   (grant_onehot),
            .grant_valid  (pk_v),
            .grant_ready  (1'b1)
        );
        assign pr_v[p] = hold_v_r[p] || pk_v;
        assign pr_c[p] = hold_v_r[p] ? hold_c_r[p] : pk_idx;
        always @(posedge clk) begin
            if (reset) begin
                hold_v_r[p] <= 1'b0;
            end else begin
                hold_v_r[p] <= pr_v[p] && ~amo_gnt;
            end
            hold_c_r[p] <= pr_c[p];
        end

        wire [LANES-1:0] ce = on_e[pr_c[p]];
        wire [LANES-1:0] co = on_o[pr_c[p]];
        wire [LANE_W-1:0] e_idx, o_idx;
        wire e_v, o_v;
        VX_priority_encoder #(.N (LANES)) e_enc (
            .data_in (ce), .index_out (e_idx), `UNUSED_PIN (onehot_out), .valid_out (e_v)
        );
        VX_priority_encoder #(.N (LANES)) o_enc (
            .data_in (co), .index_out (o_idx), `UNUSED_PIN (onehot_out), .valid_out (o_v)
        );
        wire w_e = ah_wide[pr_c[p]][e_idx];
        wire w_o = ah_wide[pr_c[p]][o_idx];
        wire split = e_v && o_v && (e_idx != o_idx) && (w_e || w_o);
        assign pr_ev[p] = e_v && ~(split && (o_idx < e_idx));
        assign pr_ov[p] = o_v && ~(split && (e_idx < o_idx)) && ~(pr_ev[p] && e_idx == o_idx);
        assign pr_e[p]  = e_idx;
        assign pr_o[p]  = (o_v && e_v && e_idx == o_idx) ? e_idx : o_idx;
        for (genvar a = 0; a < NAC; ++a) begin : g_tk
            for (genvar i = 0; i < LANES; ++i) begin : g_ln
                assign tk_p[p][a][i] = amo_gnt && pr_v[p] && (pr_c[p] == 2'(a))
                                    && ((pr_ev[p] && pr_e[p] == LANE_W'(i)) || (pr_ov[p] && pr_o[p] == LANE_W'(i)));
            end
        end
    end
    wire [NAC-1:0][LANES-1:0] tk_pick;
    for (genvar a = 0; a < NAC; ++a) begin : g_tk_or
        for (genvar i = 0; i < LANES; ++i) begin : g_ln
            wire [PAIRS-1:0] any;
            for (genvar p = 0; p < PAIRS; ++p) begin : g_p
                assign any[p] = tk_p[p][a][i];
            end
            assign tk_pick[a][i] = (|any);
        end
    end
    wire mb_lead_tk = mb_lead_v && tk_pick[2][mb_lead];
    for (genvar a = 0; a < NAC; ++a) begin : g_tk_all
        for (genvar i = 0; i < LANES; ++i) begin : g_ln
            assign tk[a][i] = tk_pick[a][i] || ((a == 2) && mb_lead_tk && mb_ride[i]);
        end
    end

    // Grant (G): bank reads of the updated lanes.
    reg [BANKS-1:0]            ag_bv;
    reg [BANKS-1:0][ROW_W-1:0] ag_brow;
    always @(*) begin
        ag_bv   = '0;
        ag_brow = '0;
        for (integer p = 0; p < PAIRS; ++p) begin
            if (pr_v[p] && pr_ev[p]) begin
                ag_bv[2*p]   = 1'b1;
                ag_brow[2*p] = ah_row[pr_c[p]][pr_e[p]];
                if (ah_wide[pr_c[p]][pr_e[p]]) begin
                    ag_bv[2*p+1]   = 1'b1;
                    ag_brow[2*p+1] = ah_row[pr_c[p]][pr_e[p]];
                end
            end
            if (pr_v[p] && pr_ov[p]) begin
                ag_bv[2*p+1]   = 1'b1;
                ag_brow[2*p+1] = ah_row[pr_c[p]][pr_o[p]];
            end
        end
    end

    // Client progress.
    wire [LANES-1:0] lsu_left_n, mbq_left_n;
    for (genvar i = 0; i < LANES; ++i) begin : g_left
        assign lsu_left_n[i] = (ah_cnt[3][i] - AC_W'(tk[3][i])) != '0;
        assign mbq_left_n[i] = (ah_cnt[2][i] - AC_W'(tk[2][i])) != '0;
    end
    assign lsu_amo_left = (|lsu_left_n);
    wire mbq_last = mbq_go && ~(|mbq_left_n) && (|tk[2]);
    assign mbq_pop  = mbq_last;
    assign amo_ready = {tk[1][0], tk[0][0]};
    always @(posedge clk) begin
        for (integer i = 0; i < LANES; ++i) begin
            if (cur_done || ~cur_valid) begin
                lsu_used_r[i] <= '0;
            end else begin
                lsu_used_r[i] <= (stg_valid ? lsu_used_r[i] : '0) + AC_W'(tk[3][i]);
            end
            if (reset || mbq_pop) begin
                mbq_used_r[i] <= '0;
            end else begin
                mbq_used_r[i] <= mbq_used_r[i] + AC_W'(tk[2][i]);
            end
        end
    end

    // Compute stage (C) registers, per bank: the lane-op read at G.
    reg [BANKS-1:0]                c_v;
    reg [BANKS-1:0]                c_wide;    // one 64-bit op over the pair (even bank)
    reg [BANKS-1:0][ROW_W-1:0]     c_row;
    reg [BANKS-1:0][1:0]           c_cl;
    reg [BANKS-1:0][LANE_W-1:0]    c_ln;
    reg [BANKS-1:0][63:0]          c_rhs;
    amo_req_t [BANKS-1:0]          c_attr;
    reg [BANKS-1:0][63:0]          c_cmp;
    reg [LANES-1:0]                c_fol;     // the phase checks riding on the pass of lane c_fol_ln
    reg [LANE_W-1:0]               c_fol_ln;
    always @(posedge clk) begin
        if (reset) begin
            c_v   <= '0;
            c_fol <= '0;
        end else begin
            c_v   <= amo_gnt ? ag_bv : '0;
            c_fol <= mb_lead_tk ? mb_ride : '0;
        end
        c_fol_ln <= mb_lead;
        for (integer p = 0; p < PAIRS; ++p) begin
            for (integer k = 0; k < 2; ++k) begin
                logic [LANE_W-1:0] l;
                logic [1:0] a;
                l = (k == 0) ? pr_e[p] : (pr_ov[p] ? pr_o[p] : pr_e[p]);
                a = pr_c[p];
                c_row[2*p+k]   <= ah_row[a][l];
                c_cl[2*p+k]    <= a;
                c_ln[2*p+k]    <= l;
                c_attr[2*p+k]  <= ah_attr[a][l];
                c_wide[2*p+k]  <= (k == 0) && pr_ev[p] && ah_wide[a][l];
                c_rhs[2*p+k]   <= (k == 0) ? ah_data[a][l] : {32'h0, ah_data[a][l][63:32]};
                c_cmp[2*p+k]   <= (k == 0) ? ah_cmp[a][l] : {32'h0, ah_cmp[a][l][63:32]};
            end
        end
    end

    // Forwarding: a read at t returns the array before t's write, so data read
    // at t - 1 takes the AMO write-backs of t - 1 and t (granted before it).
    reg  [BANKS-1:0]                 awp_v;
    reg  [BANKS-1:0][ROW_W-1:0]      awp_row;
    reg  [BANKS-1:0][BANK_BYTES-1:0] awp_be;
    reg  [BANKS-1:0][BANK_W-1:0]     awp_data;
    reg  [BANKS-1:0][ROW_W-1:0]      bwp_row;
    reg  [BANKS-1:0][BANK_BYTES-1:0] bwp_be;
    reg  [BANKS-1:0][BANK_W-1:0]     fwd;
    always @(*) begin
        for (integer bk = 0; bk < BANKS; ++bk) begin
            fwd[bk] = bk_ram[bk];
            for (integer y = 0; y < BANK_BYTES; ++y) begin
                if (awp_v[bk] && awp_row[bk] == rrow_r[bk] && awp_be[bk][y])
                    fwd[bk][y*8 +: 8] = awp_data[bk][y*8 +: 8];
                if (aw_valid[bk] && aw_row[bk] == rrow_r[bk] && aw_be[bk][y])
                    fwd[bk][y*8 +: 8] = aw_data[bk][y*8 +: 8];
            end
        end
    end
    assign bk_rdata = fwd;
    always @(posedge clk) begin
        awp_v    <= reset ? '0 : aw_valid;
        awp_row  <= aw_row;
        awp_be   <= aw_be;
        awp_data <= aw_data;
        bwp_row  <= bw_row;
        bwp_be   <= reset ? '0 : bw_be;
    end

    // C: the ALUs (per pair a 64-bit unit on the even bank, a 32-bit one on
    // the odd); an 8-B lane is the even bank's op over both banks.
    wire [BANKS-1:0]        c_own;      // the bank's own lane-op (not the odd half of an 8-B one)
    wire [BANKS-1:0]        c_store;
    wire [BANKS-1:0][63:0]  c_ret;      // the lane's response word, at the op's position
    wire [BANKS-1:0]        c_evt_v, c_evt_flip, c_wait_done;

    // LR/SC stations per bank, direct mapped by row: room for every hart's one
    // reservation when the reserved words spread over the banks. A live station
    // refuses other harts' load-reserveds until it has refused one from each.
    localparam HARTS   = `VX_CFG_NUM_WARPS * `VX_CFG_NUM_THREADS;
    localparam RS      = 1 << `CLOG2((HARTS + BANKS - 1) / BANKS);
    localparam RS_W    = `UP(`CLOG2(RS));
    localparam RS_N    = 1 << RS_W;
    localparam RS_HOLD = HARTS - 1;
    localparam HOLD_W  = `UP(`CLOG2(RS_HOLD + 1));
    `STATIC_ASSERT(RS <= (1 << ROW_W), ("the LR/SC stations of a bank are indexed by row"))

    for (genvar p = 0; p < PAIRS; ++p) begin : g_alu
        wire [1:0]        sc_ok;
        wire [1:0][63:0]  new_w;
        wire is_wide = c_wide[2*p];
        assign c_own[2*p]   = c_v[2*p];
        assign c_own[2*p+1] = c_v[2*p+1] && ~is_wide;
        for (genvar k = 0; k < 2; ++k) begin : g_bank
            localparam bk = 2 * p + k;
            amo_req_t at;
            assign at = c_attr[bk];
            wire wide_op = (k == 0) && is_wide;
            wire [63:0] old = wide_op ? {fwd[2*p+1], fwd[2*p]} : 64'(fwd[bk]);
            wire [63:0] ret_w;
        `ifdef VX_CFG_EXT_MBAR_ENABLE
            wire mflip;
            wire [3:0] merr;
        `endif
            VX_amo_alu #(
                .DATA_WIDTH (k == 0 ? 64 : 32)
            `ifdef VX_CFG_EXT_MBAR_ENABLE
                , .MBAR_ENABLE (k == 0 ? 1 : 0)
            `endif
            ) alu (
                .op          (at.amo_op),
                .is_unsigned (at.amo_unsigned),
                .width       (wide_op ? 2'd3 : 2'd2),
                .old_word    (old),
                .rhs         (c_rhs[bk]),
                .cmp         (c_cmp[bk]),
            `ifdef VX_CFG_EXT_MBAR_ENABLE
                .mbar_flip   (mflip),
                .mbar_err    (merr),
            `endif
                .new_word    (new_w[k]),
                .ret_word    (ret_w)
            );
            wire fin   = c_own[bk];
            wire is_lr = (at.amo_op == AMO_OP_LR);
            wire is_sc = (at.amo_op == AMO_OP_SC);
        `ifdef VX_CFG_EXT_MBAR_ENABLE
            wire is_wait = (at.amo_op == AMO_OP_MBAR_TEST_WAIT) || (at.amo_op == AMO_OP_MBAR_TRY_WAIT);
            assign c_wait_done[bk] = is_wait && (old[63] != c_rhs[bk][63]);
            assign c_evt_v[bk]     = fin && ((at.amo_op == AMO_OP_MBAR_TRY_WAIT && ~c_wait_done[bk]) || mflip);
            assign c_evt_flip[bk]  = fin && mflip;
            `RUNTIME_ASSERT(~(fin && (at.amo_op >= AMO_OP_MBAR_ARRIVE) && ~wide_op),
                ("%t: *** %s bank%0d: mbarrier op is not a naturally aligned doubleword", $time, INSTANCE_ID, bk))
            `RUNTIME_ASSERT(~(fin && (|merr)),
                ("%t: *** %s bank%0d: mbarrier op error 0x%0h row=0x%0h", $time, INSTANCE_ID, bk, merr, c_row[bk]))
        `else
            wire is_wait = 1'b0;
            assign c_wait_done[bk] = 1'b0;
            assign c_evt_v[bk]     = 1'b0;
            assign c_evt_flip[bk]  = 1'b0;
        `endif
            `UNUSED_VAR (at)
            assign c_store[bk] = fin && ~is_lr && ~is_wait && ~(is_sc && ~sc_ok[k]);
            wire [63:0] ret_v = is_sc ? {63'b0, ~sc_ok[k]} : (is_wait ? 64'(c_wait_done[bk]) : ret_w);
            `UNUSED_VAR (ret_v)
            if (k == 0) begin : g_ret_e
                assign c_ret[bk] = wide_op ? ret_v : {32'h0, ret_v[31:0]};
            end else begin : g_ret_o
                assign c_ret[bk] = {ret_v[31:0], 32'h0};
            end
        end

        // Reservations: a committed write to a reserved row breaks it, an SC gives up
        // its own, then a load-reserved claims its station unless another hart's
        // live one holds it, which spends one refusal. An 8-B reservation lives in
        // the even bank's station and also breaks on a write to the odd bank's word.
        for (genvar k = 0; k < 2; ++k) begin : g_rsv
            localparam bk = 2 * p + k;
            reg [RS_N-1:0]                    rs_v;
            reg [RS_N-1:0]                    rs_wide;
            reg [RS_N-1:0][ROW_W-1:0]         rs_row;
            reg [RS_N-1:0][HART_ID_WIDTH-1:0] rs_hart;
            reg [RS_N-1:0][HOLD_W-1:0]        rs_hold;
            wire [RS_W-1:0] bx = RS_W'(bw_row[bk] & ROW_W'(RS - 1));
            wire [RS_W-1:0] cx = RS_W'(c_row[bk] & ROW_W'(RS - 1));
            wire st  = (k == 1 && is_wide) ? c_store[2*p] : c_store[bk];
            wire fin = c_own[bk];
            wire [HART_ID_WIDTH-1:0] hid = c_attr[bk].hart_id;
            wire c_hit = rs_v[cx] && rs_row[cx] == c_row[bk];
            wire c_mine = c_hit && rs_hart[cx] == hid;
            assign sc_ok[k] = c_mine;
            wire [RS_W-1:0] ox, oy;
            wire o_brk_b, o_brk_c;
            if (k == 0) begin : g_odd
                assign ox = RS_W'(bw_row[bk+1] & ROW_W'(RS - 1));
                assign oy = RS_W'(c_row[bk+1] & ROW_W'(RS - 1));
                assign o_brk_b = (|bw_be[bk+1]) && rs_v[ox] && rs_wide[ox] && rs_row[ox] == bw_row[bk+1];
                assign o_brk_c = c_store[bk+1] && rs_v[oy] && rs_wide[oy] && rs_row[oy] == c_row[bk+1];
            end else begin : g_no_odd
                assign ox = '0;
                assign oy = '0;
                assign o_brk_b = 1'b0;
                assign o_brk_c = 1'b0;
                `UNUSED_VAR (rs_wide)
            end
            always @(posedge clk) begin
                if (reset) begin
                    rs_v <= '0;
                end else begin
                    if ((|bw_be[bk]) && rs_v[bx] && rs_row[bx] == bw_row[bk])
                        rs_v[bx] <= 1'b0;
                    if (st && c_hit)
                        rs_v[cx] <= 1'b0;
                    if (o_brk_b)
                        rs_v[ox] <= 1'b0;
                    if (o_brk_c)
                        rs_v[oy] <= 1'b0;
                    if (fin && c_attr[bk].amo_op == AMO_OP_SC && c_mine)
                        rs_v[cx] <= 1'b0;
                    if (fin && c_attr[bk].amo_op == AMO_OP_LR) begin
                        if (~rs_v[cx] || rs_hold[cx] == '0 || rs_hart[cx] == hid) begin
                            rs_v[cx]    <= 1'b1;
                            rs_row[cx]  <= c_row[bk];
                            rs_wide[cx] <= (k == 0) && is_wide;
                            rs_hart[cx] <= hid;
                            rs_hold[cx] <= HOLD_W'(RS_HOLD);
                        end else begin
                            rs_hold[cx] <= rs_hold[cx] - HOLD_W'(1);
                        end
                    end
                end
            end
        end

        // W: the write-backs, minus bytes a client wrote at G or C (granted after the read).
        reg [1:0]                 wv_r;
        reg [1:0][ROW_W-1:0]      wrow_r;
        reg [1:0][BANK_BYTES-1:0] wbe_r;
        reg [1:0][BANK_W-1:0]     wdata_r;
        for (genvar k = 0; k < 2; ++k) begin : g_wb
            localparam bk = 2 * p + k;
            wire st = (k == 1 && is_wide) ? c_store[2*p] : c_store[bk];
            wire [BANK_BYTES-1:0] late = ((bwp_row[bk] == c_row[bk]) ? bwp_be[bk] : '0)
                                       | ((bw_row[bk] == c_row[bk]) ? bw_be[bk] : '0);
            always @(posedge clk) begin
                wv_r[k]    <= ~reset && st;
                wrow_r[k]  <= c_row[bk];
                wbe_r[k]   <= ~late;
                wdata_r[k] <= (k == 1 && is_wide) ? new_w[0][63:32] : new_w[k][BANK_W-1:0];
            end
            assign aw_valid[bk] = wv_r[k];
            assign aw_row[bk]   = wrow_r[k];
            assign aw_be[bk]    = wbe_r[k];
            assign aw_data[bk]  = wdata_r[k];
        end
        `UNUSED_VAR (new_w[1][63:32])
    end
    assign amo_wb_now = (|aw_valid);
    `UNUSED_VAR (c_wide)
`ifndef VX_CFG_EXT_MBAR_ENABLE
    `UNUSED_VAR (c_evt_v)
    `UNUSED_VAR (c_evt_flip)
`endif

`ifdef VX_CFG_EXT_MBAR_ENABLE
    for (genvar bk = 0; bk < BANKS; ++bk) begin : g_mbar_evt
        wire [HART_ID_WIDTH-1:0] hid = c_attr[bk].hart_id;
        assign mbar_evt_valid[bk] = c_evt_v[bk];
        assign mbar_evt_flip[bk]  = c_evt_flip[bk];
        if (NW_BITS != 0) begin : g_wid
            assign mbar_evt_wid[bk] = hid[NT_BITS +: NW_BITS];
        end else begin : g_wid0
            assign mbar_evt_wid[bk] = '0;
        end
        assign mbar_evt_row[bk] = c_row[bk];
        `UNUSED_VAR (hid)
    end
`endif

    // Responses: lanes' words gathered per client as their final pass computes.
    reg [NAC-1:0][LANES-1:0][63:0] rs_acc_r;
    reg [NAC-1:0][LANES-1:0][63:0] rs_acc;
    always @(*) begin
        rs_acc   = rs_acc_r;
        for (integer bk = 0; bk < BANKS; ++bk) begin
            if (c_own[bk]) begin
                rs_acc[c_cl[bk]][c_ln[bk]] = rs_acc[c_cl[bk]][c_ln[bk]] | c_ret[bk];
                if (c_cl[bk] == 2'd2 && c_ln[bk] == c_fol_ln) begin
                    for (integer j = 0; j < LANES; ++j) begin
                        if (c_fol[j])
                            rs_acc[2][j] = rs_acc[2][j] | c_ret[bk];
                    end
                end
            end
        end
    end
    // The op whose last lane was granted at G answers at C.
    reg              lsu_fin_r, mbq_fin_r;
    reg [BLK_W-1:0]  lsu_fin_blk, mbq_fin_blk;
    reg [LMEM_TAG_WIDTH-1:0] lsu_fin_tag, mbq_fin_tag;
    reg [LANES-1:0]  lsu_fin_mask, mbq_fin_mask;
    reg [CNT_L_W-1:0] lsu_fin_lacc, mbq_fin_lacc;
    reg [NW_WIDTH-1:0] lsu_fin_wid, mbq_fin_wid;
    reg              mbq_fin_cpa;
    always @(posedge clk) begin
        if (reset) begin
            lsu_fin_r <= 1'b0;
            mbq_fin_r <= 1'b0;
            rs_acc_r  <= '0;
        end else begin
            lsu_fin_r <= cur_valid && cur.amo && cur_done;
            mbq_fin_r <= mbq_last;
            rs_acc_r  <= rs_acc;
            if (lsu_fin_r) rs_acc_r[3] <= '0;
            if (mbq_fin_r) rs_acc_r[2] <= '0;
            rs_acc_r[1:0] <= '0;
        end
        lsu_fin_blk  <= cur.blk;
        lsu_fin_tag  <= cur.tag;
        lsu_fin_mask <= cur.mask;
        lsu_fin_lacc <= cur.lacc;
        lsu_fin_wid  <= cur.wid;
        mbq_fin_lacc <= mbq_head.lacc;
        mbq_fin_wid  <= mbq_head.wid;
        mbq_fin_blk  <= mbq_head.blk;
        mbq_fin_tag  <= mbq_head.tag;
        mbq_fin_mask <= mbq_head.mask;
        mbq_fin_cpa  <= mbq_head.cpa;
    end
    for (genvar a = 0; a < 2; ++a) begin : g_amo_rsp
        reg              v_r;
        reg [ROW_TAG_W-1:0] t_r;
        always @(posedge clk) begin
            v_r <= ~reset && tk[a][0];
            t_r <= amo_tag[a];
        end
        assign amo_rsp_valid[a] = v_r;
        assign amo_rsp_tag[a]   = t_r;
        assign amo_rsp_data[a]  = rs_acc[a][0];
    end

    localparam MRQ_W = 1 + BLK_W + LMEM_TAG_WIDTH + LANES + LANES * WORD * 8;
    wire               mrq_empty;
    wire [MRQ_W-1:0]   mrq_dout;
    wire [`CLOG2(MRQ_DEPTH+1)-1:0] mrq_size;
    wire               mrq_pop;
    reg  [LANES-1:0][WORD*8-1:0] mbq_rdata;
    always @(*) begin
        for (integer i = 0; i < LANES; ++i)
            mbq_rdata[i] = rs_acc[2][i];
    end
    VX_fifo_queue #(
        .DATAW (MRQ_W),
        .DEPTH (MRQ_DEPTH)
    ) mrq (
        .clk      (clk),
        .reset    (reset),
        .push     (mbq_fin_r),
        .pop      (mrq_pop),
        .data_in  ({mbq_fin_cpa, mbq_fin_blk, mbq_fin_tag, mbq_fin_mask, mbq_rdata}),
        .data_out (mrq_dout),
        .empty    (mrq_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        .size     (mrq_size)
    );
    assign mrq_room = (32'(mrq_size) + 32'(mbq_fin_r)) < MRQ_DEPTH;
    wire             mrq_cpa = mrq_dout[MRQ_W-1];
    wire [BLK_W-1:0] mrq_blk = mrq_dout[MRQ_W-2 -: BLK_W];

    reg [LANES-1:0][WORD*8-1:0] lsu_amo_rdata;
    always @(*) begin
        for (integer i = 0; i < LANES; ++i)
            lsu_amo_rdata[i] = rs_acc[3][i];
    end
    `RUNTIME_ASSERT(~(mbq_push && mbq_full), ("%t: *** %s mbarrier op queue overflow", $time, INSTANCE_ID))

    // ---------------------------------------------------------------------
    // The banks.

    for (genvar bk = 0; bk < BANKS; ++bk) begin : g_banks
        reg [ROW_W-1:0]      raddr, waddr;
        reg                  rd_en, wr_en;
        reg [BANK_BYTES-1:0] wren;
        reg [BANK_W-1:0]     wdata;
        always @(*) begin
            rd_en = 1'b0;
            raddr = '0;
            for (integer c = 0; c < RD_ROWS; ++c) begin
                if (rd_gnt[c]) begin
                    rd_en = 1'b1;
                    raddr = rd_row[c];
                end
            end
            if (rd_gnt[RD_LSU] && eng_sel[0][bk]) begin
                rd_en = 1'b1;
                raddr = eng_sel_row[0][bk];
            end
            if (rd_gnt[RD_ROWS] && ag_bv[bk]) begin
                rd_en = 1'b1;
                raddr = ag_brow[bk];
            end
            wr_en = 1'b0;
            waddr = '0;
            wren  = '0;
            wdata = '0;
            for (integer c = 0; c < 2; ++c) begin
                if (wr_gnt[c]) begin
                    waddr = wr_data[c].row;
                    wren  = wr_data[c].byteen[bk * BANK_BYTES +: BANK_BYTES];
                    wdata = wr_data[c].data[bk * BANK_W +: BANK_W];
                end
            end
            if (wr_gnt[2]) begin
                waddr = mrg_row;
                wren  = mrg_be[bk * BANK_BYTES +: BANK_BYTES];
                wdata = mrg_data[bk * BANK_W +: BANK_W];
            end
            if (wr_gnt[WR_CPA] && eng_sel[1][bk]) begin
                waddr = eng_sel_row[1][bk];
                wren  = eng_wren[1][bk];
                wdata = eng_wdata[1][bk];
            end
            if (wr_gnt[WR_LSU] && eng_sel[0][bk]) begin
                waddr = eng_sel_row[0][bk];
                wren  = eng_wren[0][bk];
                wdata = eng_wdata[0][bk];
            end
            if (aw_valid[bk]) begin
                waddr = aw_row[bk];
                wren  = aw_be[bk];
                wdata = aw_data[bk];
            end
            wr_en = (|wren);
        end
        assign bw_row[bk] = waddr;
        assign bw_be[bk]  = aw_valid[bk] ? '0 : wren;
        reg [ROW_W-1:0] rrow;
        always @(posedge clk) begin
            if (rd_en)
                rrow <= raddr;
        end
        assign rrow_r[bk] = rrow;

        VX_dp_ram #(
            .DATAW    (BANK_W),
            .SIZE     (SM_SRAM_ROWS),
            .WRENW    (BANK_BYTES),
            .OUT_REG  (1),
            .RDW_MODE ("R")
        ) bank_ram (
            .clk   (clk),
            .reset (reset),
            .read  (rd_en),
            .write (wr_en),
            .wren  (wren),
            .waddr (waddr),
            .wdata (wdata),
            .raddr (raddr),
            .rdata (bk_ram[bk])
        );
    end

    // Row-read responses: the bank stage, then the release register.
    reg [RD_ROWS-1:0]   rdb_valid, rdr_valid;
    reg [ROW_TAG_W-1:0] rdb_tag, rdr_tag;
    reg [SM_SRAM_ROW_BYTES*8-1:0] rdr_data;
    always @(posedge clk) begin
        if (reset) begin
            rdb_valid <= '0;
            rdr_valid <= '0;
        end else begin
            rdb_valid <= rd_gnt[RD_ROWS-1:0];
            rdr_valid <= rdb_valid;
        end
        for (integer c = 0; c < RD_ROWS; ++c) begin
            if (rd_gnt[c])
                rdb_tag <= rd_tag[c];
        end
        rdr_tag  <= rdb_tag;
        rdr_data <= bk_rdata;
    end
    assign rd_rsp_valid = rdr_valid;
    assign rd_rsp_data  = rdr_data;
    assign rd_rsp_tag   = rdr_tag;
    `UNUSED_VAR (wr_gnt_idx)
    `UNUSED_VAR (rd_gnt_idx)

    // ---------------------------------------------------------------------
    // Stage progress.

    wire cur_grant  = (rd_gnt[RD_LSU] && ~cur.rw) || (wr_gnt[WR_LSU] && cur.rw);
    wire [NH-1:0] cur_left = cur_rem & ~(cur_grant ? eng_served[0] : '0);
    assign cur_done = cur_valid && (cur.mbar || (cur.amo ? (cmp_in && ~lsu_amo_left) : ((cur_left == '0) && (cur.rw || cur_grant))));

    always @(posedge clk) begin
        if (reset) begin
            stg_valid <= 1'b0;
        end else begin
            stg_valid <= cur_valid && ~cur_done;
        end
        if (cur_valid) begin
            stg_r     <= cur;
            stg_rem_r <= cur_left;
        end
        if (reset || cur_done) begin
            cmp_have_r <= 1'b0;
        end else if (cmp_take) begin
            cmp_have_r <= 1'b1;
        end
        if (cmp_take) begin
            cmp_r <= cmp_beat_d[stg_r.blk];
        end
    end

    assign wf_retired_valid = cur_done && (cur.cls == SM_SRAM_CLS_GLOBAL);

    wire [2:0] cm_n;
    assign cm_n[0] = cur_done && (cur.cls == SM_SRAM_CLS_SHARED) && ~cur.amo && ~cur.mbar && (cur.lacc != '0);
    assign cm_n[1] = lsu_fin_r && (lsu_fin_lacc != '0);
    assign cm_n[2] = mbq_fin_r && (mbq_fin_lacc != '0);
    VX_pipe_register #(
        .DATAW  (3 + 3 * (NW_WIDTH + CNT_L_W)),
        .RESETW (3)
    ) acc_commit_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({cm_n, mbq_fin_wid, lsu_fin_wid, cur.wid, mbq_fin_lacc, lsu_fin_lacc, cur.lacc}),
        .data_out ({acc_commit_valid, acc_commit_wid, acc_commit_cnt})
    );
    assign wf_retired_seq   = cur.seq;

    // Read data: captured from the banks one cycle after each grant.
    reg              cap_valid, cap_last;
    reg [NH-1:0]     cap_served;
    reg [NH-1:0][BANK_BITS-1:0] cap_bank;
    reg              cap_cls;
    reg [BLK_W-1:0]  cap_blk;
    reg [LMEM_TAG_WIDTH-1:0] cap_tag;
    reg [LANES-1:0]  cap_mask;
    reg [LANES-1:0][WORD*8-1:0] acc_data;

    always @(posedge clk) begin
        if (reset) begin
            cap_valid <= 1'b0;
            cap_last  <= 1'b0;
        end else begin
            cap_valid <= rd_gnt[RD_LSU];
            cap_last  <= rd_gnt[RD_LSU] && cur_done;
        end
        cap_served <= eng_served[0];
        cap_bank   <= eng_bank[0];
        cap_cls    <= cur.cls;
        cap_blk    <= cur.blk;
        cap_tag    <= cur.tag;
        cap_mask   <= cur.mask;
    end

    reg [LANES-1:0][WORD*8-1:0] merged;
    always @(*) begin
        merged = acc_data;
        for (integer h = 0; h < NH; ++h) begin
            if (cap_valid && cap_served[h])
                merged[h / HPW][(h % HPW) * BANK_W +: BANK_W] = bk_rdata[cap_bank[h]];
        end
    end
    always @(posedge clk) begin
        if (reset || cap_last) begin
            acc_data <= '0;
        end else begin
            acc_data <= merged;
        end
    end

    // Responses, per class in order.
    wire [1:0]              rq_push, rq_pop, rq_empty, rq_alm_full;
    wire [1:0][RSP_DATAW-1:0] rq_dout;
    wire [RSP_DATAW-1:0]    rq_din = lsu_fin_r ? {SM_SRAM_CLS_SHARED, lsu_fin_blk, lsu_fin_tag, lsu_fin_mask, lsu_amo_rdata}
                                               : {cap_cls, cap_blk, cap_tag, cap_mask, merged};
    assign rq_push[0] = (cap_last && (cap_cls == SM_SRAM_CLS_SHARED)) || lsu_fin_r;
    `RUNTIME_ASSERT(~(cap_last && lsu_fin_r), ("%t: *** %s atomic response beside a read's", $time, INSTANCE_ID))
    assign rq_push[1] = cap_last && (cap_cls == SM_SRAM_CLS_GLOBAL);

    for (genvar q = 0; q < 2; ++q) begin : g_rspq
        VX_fifo_queue #(
            .DATAW    (RSP_DATAW),
            .DEPTH    (RSP_DEPTH),
            .ALM_FULL (RSP_DEPTH - 2)
        ) rspq (
            .clk      (clk),
            .reset    (reset),
            .push     (rq_push[q]),
            .pop      (rq_pop[q]),
            .data_in  (rq_din),
            .data_out (rq_dout[q]),
            .empty    (rq_empty[q]),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (full),
            .alm_full (rq_alm_full[q]),
            `UNUSED_PIN (size)
        );
    end
    // A read in the stage or in capture still owes its queue an entry.
    assign sh_rsp_room = ~rq_alm_full[0];
    assign gl_rsp_room = ~rq_alm_full[1];

    wire [1:0] rq_v = ~rq_empty;

    wire [BLK_W-1:0]       sq_blk  = rq_dout[0][LANES * WORD * 8 + LANES + LMEM_TAG_WIDTH +: BLK_W];
    wire [LMEM_TAG_WIDTH-1:0] sq_tag = rq_dout[0][LANES * WORD * 8 + LANES +: LMEM_TAG_WIDTH];
    wire [LANES-1:0]       sq_mask = rq_dout[0][LANES * WORD * 8 +: LANES];
    wire [LANES*WORD*8-1:0] sq_data = rq_dout[0][0 +: LANES * WORD * 8];
    wire [NUM_BLOCKS-1:0]  sh_rsp_ready;
    // The in-order shared responses first, then the mbarrier ops'; a copy
    // unit's op answers on cpa_if when no write acknowledgement leaves there.
    wire m_lsu = ~mrq_empty && ~mrq_cpa;
    wire m_cpa = ~mrq_empty && mrq_cpa && ~ack_valid_r;
    wire use_m = ~rq_v[0];
    wire [BLK_W-1:0] o_blk = use_m ? mrq_blk : sq_blk;
    wire o_valid = rq_v[0] || m_lsu;
    wire [LMEM_TAG_WIDTH-1:0] o_tag = use_m ? mrq_dout[LANES * WORD * 8 + LANES +: LMEM_TAG_WIDTH] : sq_tag;
    wire [LANES-1:0] o_mask = use_m ? mrq_dout[LANES * WORD * 8 +: LANES] : sq_mask;
    wire [LANES*WORD*8-1:0] o_data = use_m ? mrq_dout[0 +: LANES * WORD * 8] : sq_data;
    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_sh_rsp
        assign lsu_if[b].rsp_valid      = o_valid && (o_blk == BLK_W'(b));
        assign lsu_if[b].rsp_data.mask  = o_mask;
        assign lsu_if[b].rsp_data.data  = o_data;
        assign lsu_if[b].rsp_data.tag   = LSU_TAG_WIDTH'(o_tag);
        assign sh_rsp_ready[b]          = lsu_if[b].rsp_ready && (o_blk == BLK_W'(b));
    end
    assign rq_pop[0] = rq_v[0] && (|sh_rsp_ready);
    assign mrq_pop   = (use_m && m_lsu && (|sh_rsp_ready)) || (m_cpa && cpa_in_rsp_ready[mrq_blk]);
    `UNUSED_VAR (o_tag)

    assign gl_rsp_valid      = rq_v[1];
    assign gl_rsp_data.cls   = SM_SRAM_CLS_GLOBAL;
    assign gl_rsp_data.tag   = rq_dout[1][LANES * WORD * 8 + LANES +: LMEM_TAG_WIDTH];
    assign gl_rsp_data.mask  = rq_dout[1][LANES * WORD * 8 +: LANES];
    assign gl_rsp_data.data  = rq_dout[1][0 +: LANES * WORD * 8];
    assign rq_pop[1]         = rq_v[1] && gl_rsp_ready;
    `UNUSED_VAR (rq_dout[0][RSP_DATAW-1])
    `UNUSED_VAR (rq_dout[1][RSP_DATAW-1 -: (1 + BLK_W)])

    // ---------------------------------------------------------------------
    // cp.async stage: one warp write at a time, acknowledged once written.

    wire [NH-1:0] cpa_left = cpa_rem_r & ~(wr_gnt[WR_CPA] ? eng_served[1] : '0);
    wire          cpa_load = ~cpa_stg_valid && cpa_any && ~cpa_item[cpa_idx].mbar;
    assign ack_pop = ack_valid_r && cpa_in_rsp_ready[ack_blk_r];

    always @(posedge clk) begin
        if (reset) begin
            cpa_stg_valid <= 1'b0;
            ack_valid_r   <= 1'b0;
        end else begin
            if (cpa_load) begin
                cpa_stg_valid <= 1'b1;
            end else if (cpa_stg_valid && cpa_left == '0 && ~ack_valid_r) begin
                cpa_stg_valid <= 1'b0;
            end
            if (cpa_stg_valid && cpa_left == '0 && ~ack_valid_r) begin
                ack_valid_r <= 1'b1;
            end else if (ack_pop) begin
                ack_valid_r <= 1'b0;
            end
        end
        if (cpa_load) begin
            cpa_blk_r  <= cpa_idx;
            cpa_mask_r <= cpa_in_mask[cpa_idx];
            cpa_tag_r  <= cpa_in_tag[cpa_idx];
            cpa_be_r   <= cpa_in_be[cpa_idx];
            cpa_data_r <= cpa_in_data[cpa_idx];
            cpa_row_r  <= cpa_in_row[cpa_idx];
            cpa_bank_r <= cpa_in_bank[cpa_idx];
            for (integer i = 0; i < LANES; ++i) begin
                for (integer k = 0; k < HPW; ++k) begin
                    cpa_rem_r[i * HPW + k] <= cpa_in_mask[cpa_idx][i]
                                           && (|cpa_in_be[cpa_idx][i][k * BANK_BYTES +: BANK_BYTES]);
                end
            end
        end else if (cpa_stg_valid) begin
            cpa_rem_r <= cpa_left;
        end
        if (cpa_stg_valid && cpa_left == '0 && ~ack_valid_r) begin
            ack_blk_r  <= cpa_blk_r;
            ack_mask_r <= cpa_mask_r;
            ack_tag_r  <= cpa_tag_r;
        end
    end

    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_cpa_rsp
        assign cpa_if[b].rsp_valid     = (ack_valid_r && (ack_blk_r == BLK_W'(b))) || (m_cpa && (mrq_blk == BLK_W'(b)));
        assign cpa_if[b].rsp_data.mask = ack_valid_r ? ack_mask_r : mrq_dout[LANES * WORD * 8 +: LANES];
        assign cpa_if[b].rsp_data.data = ack_valid_r ? '0 : mrq_dout[0 +: LANES * WORD * 8];
        assign cpa_if[b].rsp_data.tag  = ack_valid_r ? ack_tag_r : LSU_TAG_WIDTH'(mrq_dout[LANES * WORD * 8 + LANES +: LMEM_TAG_WIDTH]);
    end

    // ---------------------------------------------------------------------
    // Region checks: shared clients below s_rows, the L1 above the shared and reserve rows.

    wire [ROW_W:0] l1_base = (ROW_W+1)'(SM_SRAM_ROWS) - (ROW_W+1)'(cfg.l1_lines);
    `UNUSED_VAR (cfg.l1_protect)
    for (genvar i = 0; i < LANES; ++i) begin : g_region
        `RUNTIME_ASSERT(~pick || ~pick_item.mask[i] || (pick_item.cls == SM_SRAM_CLS_GLOBAL
                        ? ((ROW_W+1)'(pick_item.row[i]) >= l1_base)
                        : ((ROW_W+1)'(pick_item.row[i]) < cfg.s_rows)),
            ("%t: *** %s lane %0d row 0x%0h outside its region", $time, INSTANCE_ID, i, pick_item.row[i]))
    end
    for (genvar c = 0; c < RD_ROWS; ++c) begin : g_rd_region
        `RUNTIME_ASSERT(~rd_gnt[c] || ((ROW_W+1)'(rd_row[c]) < cfg.s_rows),
            ("%t: *** %s row read 0x%0h outside shared memory", $time, INSTANCE_ID, rd_row[c]))
    end
    `RUNTIME_ASSERT(~wr_gnt[0] || ((ROW_W+1)'(wr_data[0].row) >= l1_base),
        ("%t: *** %s refill row 0x%0h outside the L1", $time, INSTANCE_ID, wr_data[0].row))
    `RUNTIME_ASSERT(~wr_gnt[1] || ((ROW_W+1)'(wr_data[1].row) < cfg.s_rows),
        ("%t: *** %s DXA row 0x%0h outside shared memory", $time, INSTANCE_ID, wr_data[1].row))
    `RUNTIME_ASSERT(~(mrg_in && ~mrg_fit) || ((ROW_W+1)'(wr_data[2].row) < cfg.s_rows),
        ("%t: *** %s DSMEM row 0x%0h outside shared memory", $time, INSTANCE_ID, wr_data[2].row))
    `RUNTIME_ASSERT(~pick || ~wf_pick_idx || (gl_req_data.op == SM_SRAM_OP_READ) || (gl_req_data.op == SM_SRAM_OP_WRITE),
        ("%t: *** %s global AMO before the AMO path", $time, INSTANCE_ID))

    `undef SM_SRAM_LANE_ROW
    `undef SM_SRAM_LANE_BANK

endmodule

`endif
