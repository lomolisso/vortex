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

// The SM's L1 on the slots of the unified array (SimX SmL1, golden
// sw/common/l1_slot_ref.h). The LSU blocks' global legs merge into one
// in-order tag stage; hits go to the array as one global item, lanes of lines
// with a miss in flight wait in an in-order replay FIFO until their fill is
// written, misses, write-throughs, atomics and fences leave in order on the
// miss bus. The array holds the only copy of line data.
module VX_sm_l1 import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BLOCKS    = `VX_CFG_NUM_LSU_BLOCKS,
    parameter MEM_TAG_WIDTH = SM_L1_MEM_TAG_WIDTH
) (
    input wire              clk,
    input wire              reset,

    input smem_cfg_t        cfg,

    // the LSU blocks' global legs
    VX_lsu_mem_if.slave     lsu_if [NUM_BLOCKS],

    // the array's global class: one in-order stream
    output wire             gl_req_valid,
    output sm_sram_req_t    gl_req_data,
    input  wire             gl_req_ready,
    input  wire             gl_rsp_valid,
    input  sm_sram_rsp_t    gl_rsp_data,
    output wire             gl_rsp_ready,
    input  wire             wf_retired_valid,
    input  wire [L1_SEQ_W-1:0] wf_retired_seq,

    // refill rows into the array's write port
    output wire             fill_valid,
    output sm_sram_fill_t   fill_data,
    input  wire             fill_ready,

    // the miss bus (line requests with sector masks, write-throughs, atomics, fences)
    VX_mem_bus_if.master    mem_bus_if,

    // host flush: a fence from IN when no leg requests; done held until req drops
    VX_dcr_flush_if.slave   dcr_flush_if,

`ifdef PERF_ENABLE
    output cache_perf_t     perf,
`endif
    output wire             busy
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam LANES      = `VX_CFG_NUM_LSU_LANES;
    localparam LANE_W     = `LOG2UP(LANES);
    localparam WORD       = LSU_WORD_SIZE;
    localparam LINE       = `VX_CFG_DCACHE_LINE_SIZE;
    localparam LWORDS     = LINE / WORD;
    localparam LWORD_W    = `CLOG2(LWORDS);
    localparam ADDRW      = LSU_ADDR_WIDTH;
    localparam LINE_W     = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(LINE);
    localparam SETS       = L1_TAG_SETS;
    localparam SET_BITS   = `CLOG2(SETS);
    localparam SET_W      = `LOG2UP(SETS);
    localparam TB         = L1_TAG_BANKS;
    localparam TBB        = `CLOG2(TB);
    localparam TB_W       = `LOG2UP(TB);
    localparam TROW_W     = `LOG2UP(L1_SETS_PER_BANK);
    localparam WAYS       = L1_TAG_WAYS;
    localparam WAY_W      = `LOG2UP(WAYS);
    localparam VB         = L1_VALID_BITS;
    localparam VBYTES     = `VX_CFG_L1_VALID_BYTES;
    localparam NSEC       = LINE / `VX_CFG_DCACHE_SECTOR_SIZE;
    localparam SVB        = L1_SECTOR_VALID_BITS;
    localparam MSHR       = `VX_CFG_SM_MAX_READ_LINES;
    localparam ID_W       = `LOG2UP(MSHR);
    localparam IDX_W      = MEM_TAG_WIDTH - UUID_WIDTH - 1;
    localparam ROWS       = SM_SRAM_ROWS;
    localparam RQ_DEPTH   = `VX_CFG_LSU_PENDING_SIZE * `VX_CFG_NUM_LSU_BLOCKS;
    localparam EG_DEPTH   = 4;
    localparam BLK_W      = `UP(`CLOG2(NUM_BLOCKS));
    localparam LTAG_W     = LSU_TAG_WIDTH;
    localparam LTID_W     = LTAG_W - UUID_WIDTH;
    localparam KEY_W      = LMEM_TAG_WIDTH - UUID_WIDTH;
    localparam AMO_OUT_W  = `CLOG2(LSU_PEND_POOL * NUM_BLOCKS * LANES + 1);
    localparam LINE_SH    = `CLOG2(LINE);
    localparam MEM_ADDRW  = `VX_CFG_MEM_ADDR_WIDTH - LINE_SH;

    `STATIC_ASSERT(LINE == SM_SRAM_ROW_BYTES, ("an L1 line must be one array row"))
    `STATIC_ASSERT(LINE_W >= SET_BITS + 1, ("invalid parameter"))
    `STATIC_ASSERT(L1_TAG_W == LINE_W - SET_BITS, ("tag width"))
    `STATIC_ASSERT(SETS % TB == 0, ("tag sets must spread evenly over the tag banks"))
    `STATIC_ASSERT(`IS_POW2(SETS) && `IS_POW2(TB), ("tag sets and banks must be powers of two"))
    `STATIC_ASSERT(IDX_W >= ID_W && IDX_W >= KEY_W + LWORD_W + LANE_W, ("miss-bus tag too narrow"))
    `STATIC_ASSERT(LMEM_TAG_WIDTH >= LTAG_W + `CLOG2(NUM_BLOCKS), ("array tag too narrow"))
    `STATIC_ASSERT(`VX_CFG_L1_TAG_SETS * `VX_CFG_L1_TAG_WAYS >= `VX_CFG_SM_MAX_READ_LINES, ("L1_TAGS_COVER: the tags must hold the lines in flight"))

    localparam K_LOAD = 2'd0, K_STORE = 2'd1, K_AMO = 2'd2, K_FENCE = 2'd3;
    localparam C_NONE = 2'd0, C_COMMIT = 2'd1, C_REPLAY = 2'd2;
    localparam OP_BORROW = 2'd0, OP_YOUNG = 2'd1, OP_DRAW = 2'd2;
    localparam KIND_AGE = 3'd4;
    localparam WAIT_VICTIM_IN_FLIGHT = 3'd3, WAIT_DRAW_PENDING = 3'd4;
    localparam ALQ = 2;

    function automatic [SET_W-1:0] set_of(input [LINE_W-1:0] line);
        reg [SET_W-1:0] idx;
        idx = '0;
        if (SET_BITS != 0) begin
            if (`VX_CFG_L1_INDEX_HASH == 0) begin
                idx = SET_W'(line);
            end else begin
                for (integer i = 0; i < LINE_W; i += `MAX(SET_BITS, 1)) begin
                    idx = idx ^ SET_W'(line >> i);
                end
            end
        end
        return idx;
    endfunction

    function automatic [TB_W-1:0] bank_of(input [SET_W-1:0] set);
        reg [TB_W-1:0] b;
        b = '0;
        if (TBB != 0) begin
            for (integer i = 0; i < SET_W; i += `MAX(TBB, 1)) begin
                b = b ^ TB_W'(set >> i);
            end
        end
        return b;
    endfunction

    function automatic [TROW_W-1:0] trow_of(input [SET_W-1:0] set);
        return TROW_W'(set >> TBB);
    endfunction

    function automatic [L1_TAG_W-1:0] tag_of(input [LINE_W-1:0] line);
        return L1_TAG_W'(line >> SET_BITS);
    endfunction

    // a line read: no atomic or fence flags
    function automatic [MEM_ATTR_WIDTH-1:0] rd_attr(input [MEM_ATTR_WIDTH-1:0] a);
        mem_bus_attr_t r;
        r = mem_bus_attr_t'(a);
        r.amo = '0;
        r.is_flush = 1'b0;
        r.is_fence = 1'b0;
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        r.is_fence_cta = 1'b0;
    `endif
        return MEM_ATTR_WIDTH'(r);
    endfunction

    // =====================================================================
    // A warp request in the tag stage's form.
    typedef struct packed {
        logic [1:0]                         kind;
        logic                               nc;
        logic [BLK_W-1:0]                   blk;
        logic [LTAG_W-1:0]                  tag;
        logic [L1_SEQ_W-1:0]                seq;
        logic [LANES-1:0]                   mask;
        logic [LANES-1:0]                   lead;
        logic [LANES-1:0][LANE_W-1:0]       lead_idx;
        logic [LANES-1:0][ADDRW-1:0]        addr;
        logic [LANES-1:0][WORD*8-1:0]       data;
        logic [LANES-1:0][WORD-1:0]         byteen;
        logic [LANES-1:0][MEM_ATTR_WIDTH-1:0] attr;
    } l1_req_t;

    // ---------------------------------------------------------------------
    // IN: round robin over the blocks' global legs into one register.
    wire [NUM_BLOCKS-1:0] blk_valid;
    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_blk_valid
        assign blk_valid[b] = lsu_if[b].req_valid;
    end

    wire [BLK_W-1:0] arb_idx;
    wire [NUM_BLOCKS-1:0] arb_onehot;
    wire arb_valid;
    wire in_load;
    wire lsu_load;

    reg  hf_sent, hf_done;
    wire hf_sel = dcr_flush_if.req && ~hf_sent && ~hf_done && ~arb_valid;

    VX_generic_arbiter #(
        .NUM_REQS (NUM_BLOCKS),
        .TYPE     ("R")
    ) in_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (blk_valid),
        .grant_index  (arb_idx),
        .grant_onehot (arb_onehot),
        .grant_valid  (arb_valid),
        .grant_ready  (lsu_load)
    );

    wire [LANES-1:0]                     sel_mask;
    wire                                 sel_rw;
    wire [LANES-1:0][ADDRW-1:0]          sel_addr;
    wire [LANES-1:0][WORD*8-1:0]         sel_data;
    wire [LANES-1:0][WORD-1:0]           sel_byteen;
    wire [LANES-1:0][MEM_ATTR_WIDTH-1:0] sel_attr;
    wire [LTAG_W-1:0]                    sel_tag;
    wire [NUM_BLOCKS-1:0][LANES-1:0]                     b_mask;
    wire [NUM_BLOCKS-1:0]                                b_rw;
    wire [NUM_BLOCKS-1:0][LANES-1:0][ADDRW-1:0]          b_addr;
    wire [NUM_BLOCKS-1:0][LANES-1:0][WORD*8-1:0]         b_data;
    wire [NUM_BLOCKS-1:0][LANES-1:0][WORD-1:0]           b_byteen;
    wire [NUM_BLOCKS-1:0][LANES-1:0][MEM_ATTR_WIDTH-1:0] b_attr;
    wire [NUM_BLOCKS-1:0][LTAG_W-1:0]                    b_tag;
    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_blk_req
        assign b_mask[b]   = lsu_if[b].req_data.mask;
        assign b_rw[b]     = lsu_if[b].req_data.rw;
        assign b_addr[b]   = lsu_if[b].req_data.addr;
        assign b_data[b]   = lsu_if[b].req_data.data;
        assign b_byteen[b] = lsu_if[b].req_data.byteen;
        assign b_attr[b]   = lsu_if[b].req_data.user;
        assign b_tag[b]    = lsu_if[b].req_data.tag;
    end
    assign sel_mask   = hf_sel ? LANES'(1) : b_mask[arb_idx];
    assign sel_rw     = hf_sel ? 1'b0 : b_rw[arb_idx];
    assign sel_addr   = hf_sel ? '0 : b_addr[arb_idx];
    assign sel_data   = hf_sel ? '0 : b_data[arb_idx];
    assign sel_byteen = hf_sel ? '0 : b_byteen[arb_idx];
    assign sel_attr   = hf_sel ? (LANES * MEM_ATTR_WIDTH)'(MEM_ATTR_WIDTH'(1) << MEM_ATTR_FLUSH_OFFS) : b_attr[arb_idx];
    assign sel_tag    = hf_sel ? '0 : b_tag[arb_idx];

    // the request kind from its first active lane
    reg [MEM_ATTR_WIDTH-1:0] sel_attr0;
    always @(*) begin
        sel_attr0 = '0;
        for (integer i = LANES-1; i >= 0; --i) begin
            if (sel_mask[i]) sel_attr0 = sel_attr[i];
        end
    end
    mem_bus_attr_t sel_a;
    assign sel_a = mem_bus_attr_t'(sel_attr0);
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    wire sel_cta   = sel_a.is_fence_cta;
`else
    wire sel_cta   = 1'b0;
`endif
    wire sel_fence = (sel_a.is_flush || sel_a.is_fence) && ~sel_cta;
`ifdef VX_CFG_H100_SM_ENABLE
    wire sel_inv   = sel_a.is_inval;
`endif
    `UNUSED_VAR (sel_a)
    wire [1:0] sel_kind = sel_fence ? K_FENCE : (sel_a.amo.amo_valid ? K_AMO : (sel_rw ? K_STORE : K_LOAD));

    // distinct lines in lane order; a lane's leader is the first lane of its line
    wire [LANES-1:0][LINE_W-1:0] sel_line;
    wire [LANES-1:0] sel_lead;
    wire [LANES-1:0][LANE_W-1:0] sel_lead_idx;
    for (genvar i = 0; i < LANES; ++i) begin : g_lead
        assign sel_line[i] = LINE_W'(sel_addr[i] >> LWORD_W);
        reg [LANE_W-1:0] li;
        reg dup;
        always @(*) begin
            li  = LANE_W'(i);
            dup = 1'b0;
            for (integer j = i - 1; j >= 0; --j) begin
                if (sel_mask[j] && (sel_line[j] == sel_line[i])) begin
                    li  = LANE_W'(j);
                    dup = 1'b1;
                end
            end
        end
        assign sel_lead[i]     = sel_mask[i] && ~dup && (sel_kind != K_FENCE);
        assign sel_lead_idx[i] = li;
    end

    // ---------------------------------------------------------------------
    // Tag stage state.
    reg              in_valid;
    l1_req_t         in_r;
    reg [LANES-1:0]  in_todo;
    reg [L1_SEQ_W-1:0] seq_ctr;
    reg              in_r_cta;
`ifdef VX_CFG_H100_SM_ENABLE
    // an invalidate ends after the valid clear and answers here
    reg              in_r_inv;
    reg              t2_inv;
`endif
    reg              in_r_host;
    reg              t2_host;

    reg              t2_valid;      // a batch in T2
    reg              t2_first;      // the batch opens a new request
    reg              t2_last;       // the request's last batch
    l1_req_t         t2_r;
    reg [TB-1:0]     t2_bv;         // batch line per tag bank
    reg [TB-1:0][LANE_W-1:0] t2_bl;
    reg [TB-1:0]     t2_res;        // resolved
    reg [TB-1:0]     t2_stale;      // read under a pending write: look again

    // per leader: class and slot; waited entries
    reg [LANES-1:0][1:0]          acc_cls;
    reg [LANES-1:0][L1_SLOT_W-1:0] acc_slot;
    reg [MSHR-1:0]                acc_wait;

    // AMO lane emission and fence phases
    localparam P_LINES = 2'd0, P_EMIT = 2'd1, P_INV = 2'd2, P_PUSH = 2'd3;
    reg [1:0]        t2_phase;
    reg [LANES-1:0]  t2_emit;
    reg              inv_wait;

    // FIN: the request's built items
    reg              fin_c_valid, fin_r_valid;
    sm_sram_req_t    fin_c, fin_r;
    reg [MSHR-1:0]   fin_wait;

    // in-flight global item in the array's wavefront stage
    reg              a_valid, a_rep;
    reg [L1_SEQ_W-1:0] a_seq;
    reg [MSHR-1:0]   a_wait;

    // replay reservations
    reg [`CLOG2(RQ_DEPTH+1)-1:0] rq_resv;

    // allocation writes waiting for their tag bank, in order
    reg [ALQ-1:0]              alq_valid;
    reg [ALQ-1:0][TB_W-1:0]    alq_bank;
    reg [ALQ-1:0][TROW_W-1:0]  alq_row;
    reg [ALQ-1:0][WAY_W-1:0]   alq_way;
    reg [ALQ-1:0][L1_TAG_W-1:0] alq_tag;
    reg [ALQ-1:0][L1_SLOT_W-1:0] alq_slot;
    reg [ALQ-1:0][VB-1:0]      alq_vmask;

    // outstanding fence; atomic lanes in flight
    reg              fn_valid_r;
    reg [BLK_W-1:0]  fn_blk;
    reg [LTAG_W-1:0] fn_tag;
    reg [LANES-1:0]  fn_mask;
    reg              fn_got;
    reg              fn_host;
    reg [AMO_OUT_W-1:0] amo_out;

    // fill register
    reg              fl_valid_r;
    reg [ID_W-1:0]   fl_id_r;
    reg [LINE*8-1:0] fl_data_r;

    reg [MSHR-1:0]   done_r;

    // ---------------------------------------------------------------------
    // Submodules.
    wire [TB-1:0]               lk_valid;
    wire [TB-1:0][TROW_W-1:0]   lk_row;
    wire [TB-1:0][WAYS-1:0]     rd_valid;
    wire [TB-1:0][WAYS-1:0][L1_TAG_W-1:0]  rd_tag;
    wire [TB-1:0][WAYS-1:0][L1_SLOT_W-1:0] rd_slot;
    wire [TB-1:0][WAYS-1:0][VB-1:0]        rd_vmask;

    reg [L1_TAG_UPD_SRCS-1:0]               up_valid;
    reg [L1_TAG_UPD_SRCS-1:0][TB_W-1:0]     up_bank;
    reg [L1_TAG_UPD_SRCS-1:0][TROW_W-1:0]   up_row;
    reg [L1_TAG_UPD_SRCS-1:0][WAY_W-1:0]    up_way;
    reg [L1_TAG_UPD_SRCS-1:0][VB-1:0]       up_mask;
    reg [L1_TAG_UPD_SRCS-1:0]               up_set;
    reg [L1_TAG_UPD_SRCS-1:0]               up_inval;
    reg [L1_TAG_UPD_SRCS-1:0][ID_W-1:0]     up_id;
    wire [L1_TAG_UPD_SRCS-1:0]              up_ready;
    wire [TB-1:0]                           q_room;
    wire [TB-1:0]                           q_room2;
    wire al_grant;
    wire [TB-1:0] tw_valid;
    wire [TB-1:0][1:0] tw_src;
    wire [TB-1:0][ID_W-1:0] tw_id;
    reg  inval_start;
    wire inval_busy;

    VX_sm_l1_tags #(
        .INSTANCE_ID (`SFORMATF(("%s-tags", INSTANCE_ID))),
        .ID_W        (ID_W)
    ) tags (
        .clk         (clk),
        .reset       (reset),
        .inval_start (inval_start),
        .inval_busy  (inval_busy),
        .lk_valid    (lk_valid),
        .lk_row      (lk_row),
        .rd_valid    (rd_valid),
        .rd_tag      (rd_tag),
        .rd_slot     (rd_slot),
        .rd_vmask    (rd_vmask),
        .up_valid    (up_valid),
        .up_bank     (up_bank),
        .up_row      (up_row),
        .up_way      (up_way),
        .up_mask     (up_mask),
        .up_set      (up_set),
        .up_inval    (up_inval),
        .up_id       (up_id),
        .up_ready    (up_ready),
        .q_room      (q_room),
        .q_room2     (q_room2),
        .al_valid    (alq_valid[0]),
        .al_bank     (alq_bank[0]),
        .al_row      (alq_row[0]),
        .al_way      (alq_way[0]),
        .al_tag      (alq_tag[0]),
        .al_slot     (alq_slot[0]),
        .al_vmask    (alq_vmask[0]),
        .al_grant    (al_grant),
        .wr_valid    (tw_valid),
        .wr_src      (tw_src),
        .wr_id       (tw_id)
    );

    reg  [WAYS-1:0][L1_SLOT_W-1:0] q_slot;
    wire [WAYS-1:0] q_open;
    wire free_avail;
    wire [15:0] lfsr_next;
    reg  lfsr_step;
    reg  pool_valid;
    reg  [1:0] pool_op;
    reg  [SET_W-1:0] pool_set;
    reg  [WAY_W-1:0] pool_way;
    wire pool_ok;
    wire [2:0] pool_wait, pool_kind;
    wire [L1_SLOT_W-1:0] pool_slot;
    wire victim_valid;
    wire [L1_SLOT_W-1:0] victim_slot;
    wire [SET_W-1:0] victim_set;
    wire [WAY_W-1:0] victim_way;
    wire [L1_SLOT_W:0] old_count;
    // a victim clear would land before a queued allocation write of the same
    // way, which would then revive the cleared line on the slot it gives away
    wire victim_pin;
    reg  victim_busy;
    always @(*) begin
        victim_busy = victim_pin;
        for (integer q = 0; q < ALQ; ++q) begin
            victim_busy |= alq_valid[q] && (alq_slot[q] == victim_slot);
        end
    end
    reg  open_valid;
    reg  [L1_SLOT_W-1:0] open_slot;
    wire rt_valid;
    wire [ID_W-1:0] rt_id;
    wire [L1_SLOT_W-1:0] rt_slot;
    wire rt_cacheable;

    VX_sm_l1_slots #(
        .INSTANCE_ID (`SFORMATF(("%s-slots", INSTANCE_ID)))
    ) slots (
        .clk            (clk),
        .reset          (reset),
        .cfg_lines      (cfg.l1_lines),
        .cfg_protect    (cfg.l1_protect),
        .invalidate     (inval_start),
        .q_slot         (q_slot),
        .q_open         (q_open),
        .free_avail     (free_avail),
        .lfsr_next      (lfsr_next),
        .lfsr_step      (lfsr_step),
        .pool_valid     (pool_valid),
        .pool_op        (pool_op),
        .pool_set       (pool_set),
        .pool_way       (pool_way),
        .pool_ok        (pool_ok),
        .pool_wait      (pool_wait),
        .pool_kind      (pool_kind),
        .pool_slot      (pool_slot),
        .victim_valid   (victim_valid),
        .victim_slot    (victim_slot),
        .victim_set     (victim_set),
        .victim_way     (victim_way),
        .victim_busy    (victim_busy),
        .open_valid     (open_valid),
        .open_slot      (open_slot),
        .retire_valid   (rt_valid),
        .retire_slot    (rt_slot),
        .retire_borrowed(~rt_cacheable),
        `UNUSED_PIN     (young_count),
        .old_count      (old_count),
        `UNUSED_PIN     (bump_count),
        `UNUSED_PIN     (free_count)
    );
    `UNUSED_VAR ({cfg, pool_wait})

    wire [TB-1:0] m_hit, m_store;
    wire [TB-1:0][ID_W-1:0] m_id;
    wire [TB-1:0][NSEC-1:0] m_secs;
    wire [TB-1:0][L1_SLOT_W-1:0] m_slot;
    reg  mal_valid;
    reg  [LINE_W-1:0] mal_line;
    reg  [NSEC-1:0] mal_secs;
    reg  [L1_SLOT_W-1:0] mal_slot;
    reg  [SET_W-1:0] mal_set;
    reg  [WAY_W-1:0] mal_way;
    reg  [VB-1:0] mal_vmiss;
    reg  mal_cacheable, mal_ilk;
    wire mal_ready;
    wire [ID_W-1:0] mal_id;
    reg  [TB-1:0] jn_valid;
    reg  [TB-1:0][ID_W-1:0] jn_id;
    reg  [TB-1:0] jn_store;
    wire [LINE_W-1:0] fe_line;
    wire [NSEC-1:0] fe_secs;
    wire [L1_SLOT_W-1:0] fe_slot;
    wire [SET_W-1:0] fe_set;
    wire [WAY_W-1:0] fe_way;
    wire [VB-1:0] fe_vmiss;
    wire fe_cacheable, fe_ilk_block;
    wire fill_fire;
    reg  [MSHR-1:0] rp_inc;
    wire [MSHR-1:0] rp_dec;
    wire ilk_fin_take;
    wire mshr_empty;
    wire [TB-1:0][LINE_W-1:0] t2_line;

    wire [TB-1:0] k_hold;

    VX_sm_l1_mshr #(
        .INSTANCE_ID (`SFORMATF(("%s-mshr", INSTANCE_ID))),
        .RPC_MAX     (RQ_DEPTH)
    ) mshr (
        .clk          (clk),
        .reset        (reset),
        .lk_hold      (k_hold),
        .lk_line      (t2_line),
        .lk_hit       (m_hit),
        .lk_id        (m_id),
        .lk_sectors   (m_secs),
        .lk_store     (m_store),
        .lk_slot      (m_slot),
        .al_valid     (mal_valid),
        .al_line      (mal_line),
        .al_sectors   (mal_secs),
        .al_slot      (mal_slot),
        .al_set       (mal_set),
        .al_way       (mal_way),
        .al_vmiss     (mal_vmiss),
        .al_cacheable (mal_cacheable),
        .al_ilk       (mal_ilk ? {a_valid && ~a_rep, fin_c_valid} : 2'b0),
        .al_ready     (mal_ready),
        .al_id        (mal_id),
        .jn_valid     (jn_valid),
        .jn_id        (jn_id),
        .jn_store     (jn_store),
        .rp_inc       (rp_inc),
        .rp_dec       (rp_dec),
        .ilk_fin_take (ilk_fin_take),
        .ilk_arr_take (gl_fire),
        .ilk_arr_rep  (take_rep),
        .ilk_arr_ret  (wf_retired_valid),
        .fl_id        (fl_id_r),
        .fl_line      (fe_line),
        .fl_sectors   (fe_secs),
        .fl_slot      (fe_slot),
        .fl_set       (fe_set),
        .fl_way       (fe_way),
        .fl_vmiss     (fe_vmiss),
        .fl_cacheable (fe_cacheable),
        .fl_ilk_block (fe_ilk_block),
        .fl_done      (fill_fire),
        .rt_valid     (rt_valid),
        .rt_id        (rt_id),
        .rt_slot      (rt_slot),
        .rt_cacheable (rt_cacheable),
        .empty        (mshr_empty),
        `UNUSED_PIN   (full)
    );
    `UNUSED_VAR ({fe_line, rt_id})

    // ---------------------------------------------------------------------
    // Egress FIFO (miss bus, in order).
    localparam EG_DATAW = 1 + MEM_ADDRW + LINE * 8 + LINE + MEM_ATTR_WIDTH + MEM_TAG_WIDTH;
    reg              eg_push;
    reg              eg_rw;
    reg [MEM_ADDRW-1:0] eg_addr;
    reg [LINE*8-1:0] eg_data;
    reg [LINE-1:0]   eg_byteen;
    reg [MEM_ATTR_WIDTH-1:0] eg_attr;
    reg [MEM_TAG_WIDTH-1:0]  eg_tag;
    wire eg_empty, eg_full;
    wire [EG_DATAW-1:0] eg_dout;
    wire eg_pop = ~eg_empty && mem_bus_if.req_ready;

    VX_fifo_queue #(
        .DATAW (EG_DATAW),
        .DEPTH (EG_DEPTH)
    ) egress (
        .clk      (clk),
        .reset    (reset),
        .push     (eg_push),
        .pop      (eg_pop),
        .data_in  ({eg_rw, eg_addr, eg_data, eg_byteen, eg_attr, eg_tag}),
        .data_out (eg_dout),
        .empty    (eg_empty),
        .full     (eg_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );
    assign mem_bus_if.req_valid = ~eg_empty;
    assign {mem_bus_if.req_data.rw,
            mem_bus_if.req_data.addr,
            mem_bus_if.req_data.data,
            mem_bus_if.req_data.byteen,
            mem_bus_if.req_data.attr,
            mem_bus_if.req_data.tag} = eg_dout;

    // ---------------------------------------------------------------------
    // Replay FIFO: whole items in tag-stage order, each with its waited entries.
    localparam RQ_DATAW = $bits(sm_sram_req_t) + MSHR;
    wire rq_push = fin_r_valid;
    wire rq_pop;
    wire rq_empty;
    wire [RQ_DATAW-1:0] rq_dout;
    VX_fifo_queue #(
        .DATAW (RQ_DATAW),
        .DEPTH (RQ_DEPTH)
    ) replay_q (
        .clk      (clk),
        .reset    (reset),
        .push     (rq_push),
        .pop      (rq_pop),
        .data_in  ({fin_r, fin_wait}),
        .data_out (rq_dout),
        .empty    (rq_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );
    sm_sram_req_t rq_head;
    wire [MSHR-1:0] rq_head_wait;
    assign {rq_head, rq_head_wait} = rq_dout;
    wire rq_ready = ~rq_empty && ((rq_head_wait & ~done_r) == '0);

    // merge point: a ready replay first, else the commit item; an offer
    // stays as it is until the array takes it
    reg  gl_off_c;
    wire take_rep = rq_ready && ~gl_off_c;
    assign gl_req_valid = rq_ready || fin_c_valid;
    assign gl_req_data  = take_rep ? rq_head : fin_c;
    wire gl_fire = gl_req_valid && gl_req_ready;
    assign rq_pop = gl_fire && take_rep;
    wire fin_c_take = gl_fire && ~take_rep;

    // ---------------------------------------------------------------------
    // T2: resolve the batch.
    for (genvar k = 0; k < TB; ++k) begin : g_t2_line
        assign t2_line[k] = LINE_W'(t2_r.addr[t2_bl[k]] >> LWORD_W);
    end
    `UNUSED_VAR (t2_r.lead)

    // per batch line: what its lanes need; a word-select crossbar per line
    // byte, the highest lane writing the byte gives its data
    reg [LANES-1:0][LWORDS-1:0] t2_lw;
    reg [TB-1:0][LANES-1:0] k_lanes;
    reg [TB-1:0][LINE-1:0] k_bm;        // bytes the line's lanes touch
    reg [TB-1:0][VB-1:0]   k_need, k_full;
    reg [TB-1:0][NSEC-1:0] k_secs;
    reg [TB-1:0][LINE*8-1:0] k_wdata;
    always @(*) begin
        for (integer j = 0; j < LANES; ++j) begin
            for (integer w = 0; w < LWORDS; ++w) begin
                t2_lw[j][w] = (LWORD_W'(t2_r.addr[j]) == LWORD_W'(w));
            end
        end
        for (integer k = 0; k < TB; ++k) begin
            for (integer j = 0; j < LANES; ++j) begin
                k_lanes[k][j] = t2_r.mask[j] && (t2_r.lead_idx[j] == t2_bl[k]);
            end
            k_bm[k] = '0;
            k_wdata[k] = '0;
            for (integer w = 0; w < LWORDS; ++w) begin
                for (integer b = 0; b < WORD; ++b) begin
                    for (integer j = 0; j < LANES; ++j) begin
                        if (k_lanes[k][j] && t2_lw[j][w] && t2_r.byteen[j][b]) begin
                            k_bm[k][w * WORD + b] = 1'b1;
                            k_wdata[k][(w * WORD + b) * 8 +: 8] = t2_r.data[j][b * 8 +: 8];
                        end
                    end
                end
            end
            for (integer v = 0; v < VB; ++v) begin
                k_need[k][v] = | k_bm[k][v * VBYTES +: VBYTES];
                k_full[k][v] = & k_bm[k][v * VBYTES +: VBYTES];
            end
            for (integer s = 0; s < NSEC; ++s) begin
                k_secs[k][s] = | k_need[k][s * SVB +: SVB];
            end
        end
    end

    // a lookup that may join its line's open entry keeps the entry from
    // retiring in this cycle; one that waits for the entry to retire (other
    // sectors, an atomic) must not
    for (genvar k = 0; k < TB; ++k) begin : g_k_hold
        assign k_hold[k] = t2_valid && t2_bv[k] && ~t2_res[k]
            && (((t2_r.kind == K_LOAD) && ((k_secs[k] & ~m_secs[k]) == '0)) || (t2_r.kind == K_STORE));
    end

    // tag compare
    reg [TB-1:0]             th;
    reg [TB-1:0][WAY_W-1:0]  tway;
    reg [TB-1:0][L1_SLOT_W-1:0] tslot;
    reg [TB-1:0][VB-1:0]     tvm;
    reg [TB-1:0]             tinv;
    reg [TB-1:0][WAY_W-1:0]  tinv_way;
    always @(*) begin
        for (integer k = 0; k < TB; ++k) begin
            th[k] = 1'b0; tway[k] = '0; tslot[k] = '0; tvm[k] = '0;
            tinv[k] = 1'b0; tinv_way[k] = '0;
            for (integer w = WAYS-1; w >= 0; --w) begin
                if (rd_valid[k][w] && (rd_tag[k][w] == tag_of(t2_line[k]))) begin
                    th[k] = 1'b1; tway[k] = WAY_W'(w); tslot[k] = rd_slot[k][w]; tvm[k] = rd_vmask[k][w];
                end
                if (~rd_valid[k][w]) begin
                    tinv[k] = 1'b1; tinv_way[k] = WAY_W'(w);
                end
            end
        end
    end

    // classification: resolve now, wait, or needs the cycle's one action
    localparam R_WAIT = 3'd0, R_HIT = 3'd1, R_JOIN = 3'd2, R_NONE = 3'd3, R_ACT = 3'd4;
    reg [TB-1:0][2:0] k_cls;
    always @(*) begin
        for (integer k = 0; k < TB; ++k) begin
            k_cls[k] = R_WAIT;
            if (t2_valid && t2_bv[k] && ~t2_res[k] && ~t2_stale[k]) begin
                case (t2_r.kind)
                K_LOAD: begin
                    if (m_hit[k]) begin
                        if (~m_store[k] && th[k] && ((tvm[k] & k_need[k]) == k_need[k]))
                            k_cls[k] = R_HIT;
                        else if ((k_secs[k] & ~m_secs[k]) == '0)
                            k_cls[k] = R_JOIN;
                    end else if (th[k] && ((tvm[k] & k_need[k]) == k_need[k])) begin
                        k_cls[k] = R_HIT;
                    end else begin
                        k_cls[k] = R_ACT;
                    end
                end
                K_STORE: k_cls[k] = R_ACT;
                K_AMO: begin
                    if (~m_hit[k])
                        k_cls[k] = th[k] ? R_ACT : R_NONE;
                end
                default:;
                endcase
            end
        end
    end

    // the action line: the lowest tag bank asking
    reg act_any;
    reg [TB_W-1:0] act_k;
    always @(*) begin
        act_any = 1'b0;
        act_k = '0;
        for (integer k = TB-1; k >= 0; --k) begin
            if (k_cls[k] == R_ACT) begin
                act_any = 1'b1;
                act_k = TB_W'(k);
            end
        end
    end

    wire [LINE_W-1:0] a_line   = t2_line[act_k];
    wire [SET_W-1:0]  a_set    = set_of(a_line);
    wire [TROW_W-1:0] a_row    = trow_of(a_set);
    wire [LANE_W-1:0] a_lane   = t2_bl[act_k];
    wire [MEM_ATTR_WIDTH-1:0] a_attr = t2_r.attr[a_lane];
    wire [UUID_WIDTH-1:0] a_uuid = t2_r.tag[LTAG_W-1 -: UUID_WIDTH];

    // in-set victim candidates of the action line's set
    always @(*) begin
        q_slot = rd_slot[act_k];
    end

    // slots the load in the tag stage reads at its commit: its earlier hits and this cycle's;
    // and the slots of the open entries the request in the tag stage joined
    reg [WAYS-1:0] q_pin, q_jn;
    reg            v_pin, v_jn;
    always @(*) begin
        q_pin = '0;
        v_pin = 1'b0;
        q_jn  = '0;
        v_jn  = 1'b0;
        if (t2_valid) begin
            for (integer k = 0; k < TB; ++k) begin
                if ((k_cls[k] == R_HIT) && (t2_r.kind == K_LOAD)) begin
                    for (integer w = 0; w < WAYS; ++w) begin
                        q_pin[w] |= (tslot[k] == q_slot[w]);
                    end
                    v_pin |= (tslot[k] == victim_slot);
                end
                if (k_cls[k] == R_JOIN) begin
                    for (integer w = 0; w < WAYS; ++w) begin
                        q_jn[w] |= ((th[k] ? tslot[k] : m_slot[k]) == q_slot[w]);
                    end
                    v_jn |= ((th[k] ? tslot[k] : m_slot[k]) == victim_slot);
                end
            end
            if (~t2_first) begin
                for (integer j = 0; j < LANES; ++j) begin
                    if ((acc_cls[j] == C_COMMIT) && (t2_r.kind == K_LOAD)) begin
                        for (integer w = 0; w < WAYS; ++w) begin
                            q_pin[w] |= (acc_slot[j] == q_slot[w]);
                        end
                        v_pin |= (acc_slot[j] == victim_slot);
                    end
                    if (acc_cls[j] == C_REPLAY) begin
                        for (integer w = 0; w < WAYS; ++w) begin
                            q_jn[w] |= (acc_slot[j] == q_slot[w]);
                        end
                        v_jn |= (acc_slot[j] == victim_slot);
                    end
                end
            end
        end
    end
    assign victim_pin = v_pin;

    reg [WAYS-1:0] cand;
    reg [WAY_W:0]  ncand;
    reg [WAY_W-1:0] cand_way;
    always @(*) begin
        reg [WAY_W:0] pick, n;
        cand = ~q_open & ~q_pin;
        ncand = '0;
        for (integer w = 0; w < WAYS; ++w) begin
            ncand = ncand + (WAY_W+1)'(cand[w]);
        end
        pick = (ncand != 0) ? (WAY_W+1)'(32'(lfsr_next) % 32'(ncand)) : '0;
        cand_way = '0;
        n = '0;
        for (integer w = 0; w < WAYS; ++w) begin
            if (cand[w]) begin
                if (n == pick) cand_way = WAY_W'(w);
                n = n + 1;
            end
        end
    end

    // a draw started by the tag stage completes on a later cycle
    reg draw_on;

    wire fin_free;
    wire alq_free = ~alq_valid[ALQ-1];
    wire v_room   = q_room[bank_of(victim_set)];

    // what the action needs from the pool (no dependence on the pool's answer)
    wire act_mem   = act_any && fin_free;
    wire ld_alloc  = (t2_r.kind == K_LOAD) && ~th[act_k] && mal_ready && ~eg_full;
    wire st_alloc  = (t2_r.kind == K_STORE) && ~t2_r.nc && ~m_hit[act_k] && ~th[act_k] && ~eg_full;
    wire use_borrow = ld_alloc && t2_r.nc && free_avail && ~draw_on;
    wire use_pool  = act_mem && (use_borrow || ((ld_alloc || st_alloc) && alq_free && tinv[act_k]));
    always @(*) begin
        pool_set = a_set;
        pool_way = tinv_way[act_k];
        if (use_borrow) begin
            pool_op = OP_BORROW;
        end else if (free_avail && ~draw_on) begin
            pool_op = OP_YOUNG;
        end else begin
            pool_op = OP_DRAW;
        end
        pool_valid = use_pool && ((pool_op != OP_DRAW) || ~draw_on || v_room);
    end
    wire pool_draw  = pool_valid && (pool_op == OP_DRAW);
    wire draw_start = pool_draw && ~pool_ok && ((pool_wait == WAIT_DRAW_PENDING) || (pool_wait == WAIT_VICTIM_IN_FLIGHT));
    wire draw_done  = pool_draw && pool_ok;

    reg act_fire;
    reg [2:0] act_res;          // R_HIT (commit) / R_JOIN (replay) / R_NONE
    reg [L1_SLOT_W-1:0] act_slot;
    reg [ID_W-1:0] act_wid;
    reg act_al;                 // new allocation write
    reg [WAY_W-1:0] act_al_way;
    reg [L1_SLOT_W-1:0] act_al_slot;
    reg [VB-1:0] act_al_vm;
    reg u0_valid, u0_set;
    reg [VB-1:0] u0_mask;
    reg [MEM_ATTR_WIDTH-1:0] aeg_attr;
    reg [LINE-1:0] aeg_byteen;
    reg aeg_rw, aeg_push;
    reg [IDX_W:0] aeg_idx;

    always @(*) begin
        act_fire = 1'b0;
        act_res = R_NONE;
        act_slot = '0;
        act_wid = '0;
        act_al = 1'b0;
        act_al_way = '0;
        act_al_slot = '0;
        act_al_vm = '0;
        lfsr_step = 1'b0;
        mal_valid = 1'b0;
        mal_line = a_line;
        mal_secs = k_secs[act_k];
        mal_slot = '0;
        mal_set = a_set;
        mal_way = '0;
        mal_vmiss = '0;
        mal_cacheable = 1'b1;
        mal_ilk = 1'b0;
        open_valid = 1'b0;
        open_slot = '0;
        u0_valid = 1'b0;
        u0_mask = k_full[act_k];
        u0_set = 1'b1;
        aeg_push = 1'b0;
        aeg_rw = 1'b0;
        aeg_byteen = '0;
        aeg_attr = a_attr;
        aeg_idx = {1'b0, IDX_W'(mal_id)};
        if (act_mem) begin
            case (t2_r.kind)
            K_LOAD: begin
                for (integer s = 0; s < NSEC; ++s) begin
                    aeg_byteen[s * (LINE / NSEC) +: (LINE / NSEC)] = {(LINE / NSEC){k_secs[act_k][s]}};
                end
                aeg_attr = rd_attr(a_attr);
                if (mal_ready && ~eg_full) begin
                    if (th[act_k]) begin
                        // the line keeps its way and slot; the missing words refill
                        act_fire = 1'b1;
                        mal_slot = tslot[act_k];
                        mal_way = tway[act_k];
                        mal_vmiss = tvm[act_k];
                        // a store-allocate or an atomic may have left older readers
                        // of other data on the slot, with no fill of its own to wait behind
                        mal_ilk = 1'b1;
                    end else if (use_borrow) begin
                        act_fire = 1'b1;
                        mal_slot = pool_slot;
                        mal_cacheable = 1'b0;
                    end else if (alq_free) begin
                        if (tinv[act_k]) begin
                            if (pool_valid && pool_ok) begin
                                act_fire = 1'b1;
                                mal_slot = pool_slot;
                                mal_way = tinv_way[act_k];
                                mal_ilk = (pool_kind == KIND_AGE);
                                act_al = 1'b1;
                            end
                        end else if (ncand != 0) begin
                            act_fire = 1'b1;
                            lfsr_step = 1'b1;
                            mal_slot = rd_slot[act_k][cand_way];
                            mal_way = cand_way;
                            mal_ilk = 1'b1;
                            act_al = 1'b1;
                        end
                    end
                    if (act_fire) begin
                        mal_valid = 1'b1;
                        open_valid = 1'b1;
                        open_slot = mal_slot;
                        aeg_push = 1'b1;
                        act_res = R_JOIN;
                        act_slot = mal_slot;
                        act_wid = mal_id;
                        act_al_way = mal_way;
                        act_al_slot = mal_slot;
                    end
                end
            end
            K_STORE: begin
                aeg_rw = 1'b1;
                aeg_byteen = k_bm[act_k];
                aeg_idx = '0;
                if (~eg_full) begin
                    if (t2_r.nc) begin
                        act_fire = 1'b1;
                        act_res = R_NONE;
                    end else if (m_hit[act_k] || th[act_k]) begin
                        u0_valid = th[act_k] && (k_full[act_k] != '0);
                        if (~u0_valid || q_room[act_k]) begin
                            act_fire = 1'b1;
                            act_res = m_hit[act_k] ? R_JOIN : R_HIT;
                            act_slot = th[act_k] ? tslot[act_k] : m_slot[act_k];
                            act_wid = m_id[act_k];
                        end else begin
                            u0_valid = 1'b0;
                        end
                    end else if (alq_free) begin
                        // store-allocate: a way and a slot, the fully written words valid
                        if (tinv[act_k]) begin
                            if (pool_valid && pool_ok) begin
                                act_fire = 1'b1;
                                act_al_way = tinv_way[act_k];
                                act_al_slot = pool_slot;
                            end
                        end else if (ncand != 0) begin
                            act_fire = 1'b1;
                            lfsr_step = 1'b1;
                            act_al_way = cand_way;
                            act_al_slot = rd_slot[act_k][cand_way];
                        end
                        if (act_fire) begin
                            act_al = 1'b1;
                            act_al_vm = k_full[act_k];
                            act_res = R_HIT;
                            act_slot = act_al_slot;
                        end
                    end
                    aeg_push = act_fire;
                end
            end
            K_AMO: begin
                // the probe clears the line's valid bits; tag and slot stay
                u0_mask = '1;
                u0_set = 1'b0;
                if (q_room[act_k]) begin
                    u0_valid = 1'b1;
                    act_fire = 1'b1;
                    act_res = R_NONE;
                end
            end
            default:;
            endcase
        end
    end

    // resolutions this cycle
    reg [TB-1:0] res_now;
    reg [TB-1:0][2:0] res_cls;
    reg [TB-1:0][L1_SLOT_W-1:0] res_slot;
    reg [TB-1:0][ID_W-1:0] res_wid;
    always @(*) begin
        for (integer k = 0; k < TB; ++k) begin
            res_now[k] = 1'b0;
            res_cls[k] = k_cls[k];
            res_slot[k] = tslot[k];
            res_wid[k] = m_id[k];
            jn_valid[k] = 1'b0;
            jn_id[k] = m_id[k];
            jn_store[k] = 1'b0;
            case (k_cls[k])
            R_HIT:  res_now[k] = fin_free;
            R_JOIN: begin
                res_now[k] = fin_free;
                res_slot[k] = th[k] ? tslot[k] : m_slot[k];
                jn_valid[k] = fin_free;
            end
            R_NONE: res_now[k] = fin_free;
            R_ACT: begin
                if (act_fire && (act_k == TB_W'(k))) begin
                    res_now[k] = 1'b1;
                    res_cls[k] = act_res;
                    res_slot[k] = act_slot;
                    res_wid[k] = act_wid;
                    jn_valid[k] = (t2_r.kind == K_STORE) && m_hit[k] && ~t2_r.nc;
                    jn_store[k] = 1'b1;
                end
            end
            default:;
            endcase
        end
    end

    wire [TB-1:0] res_next = t2_res | res_now;
    wire lines_done = t2_valid && ((t2_bv & ~res_next) == '0);
    wire req_lines_done = lines_done && t2_last;

    // the accumulator after this cycle
    reg [LANES-1:0][1:0]           acc_cls_n;
    reg [LANES-1:0][L1_SLOT_W-1:0] acc_slot_n;
    reg [MSHR-1:0]                 acc_wait_n;
    always @(*) begin
        acc_cls_n  = t2_first ? '0 : acc_cls;
        acc_slot_n = acc_slot;
        acc_wait_n = t2_first ? '0 : acc_wait;
        for (integer k = 0; k < TB; ++k) begin
            if (res_now[k]) begin
                case (res_cls[k])
                R_HIT:  acc_cls_n[t2_bl[k]] = C_COMMIT;
                R_JOIN: begin
                    acc_cls_n[t2_bl[k]] = C_REPLAY;
                    acc_wait_n[res_wid[k]] = 1'b1;
                end
                default: acc_cls_n[t2_bl[k]] = C_NONE;
                endcase
                acc_slot_n[t2_bl[k]] = res_slot[k];
            end
        end
        // a store allocation may take the slot of a line the same request
        // already wrote: that line has left the L1 (its write-through stays),
        // so its bytes stay out of the slot, where the new line's must land
        if (act_fire && act_al && (t2_r.kind == K_STORE)) begin
            for (integer j = 0; j < LANES; ++j) begin
                if ((LANE_W'(j) != t2_bl[act_k]) && (acc_cls_n[j] == C_COMMIT) && (acc_slot_n[j] == act_al_slot)) begin
                    acc_cls_n[j] = C_NONE;
                end
            end
        end
    end

    // build the request's items
    sm_sram_req_t bld_c, bld_r;
    reg [LANES-1:0] bld_cm, bld_rm;
    always @(*) begin
        bld_c = '0;
        bld_c.cls = SM_SRAM_CLS_GLOBAL;
        bld_c.op = (t2_r.kind == K_STORE) ? SM_SRAM_OP_WRITE : SM_SRAM_OP_READ;
        bld_c.tag = LMEM_TAG_WIDTH'({t2_r.blk, t2_r.tag});
        bld_c.seq = t2_r.seq;
        for (integer j = 0; j < LANES; ++j) begin
            bld_c.lanes[j].row = SM_SRAM_ROW_W'(ROWS - 1) - SM_SRAM_ROW_W'(acc_slot_n[t2_r.lead_idx[j]]);
            bld_c.lanes[j].boff = SM_SRAM_BOFF_W'(LWORD_W'(t2_r.addr[j]) * WORD);
            bld_c.lanes[j].byteen = t2_r.byteen[j];
            bld_c.lanes[j].data = t2_r.data[j];
            bld_cm[j] = t2_r.mask[j] && (acc_cls_n[t2_r.lead_idx[j]] == C_COMMIT);
            bld_rm[j] = t2_r.mask[j] && (acc_cls_n[t2_r.lead_idx[j]] == C_REPLAY);
        end
        bld_r = bld_c;
        bld_c.mask = bld_cm;
        bld_r.mask = bld_rm;
    end

    // a request whose next line waits only on what the request itself holds (every way of the
    // set, every MSHR entry, or the drawn victim of an old list it could hold whole) would never
    // get it: what it has resolved commits now, and its other lines go on as a new pass
    wire way_held  = ~tinv[act_k] && ~th[act_k] && (&(q_pin | q_jn))
                  && (((t2_r.kind == K_LOAD) && ~(t2_r.nc && free_avail))
                   || ((t2_r.kind == K_STORE) && ~t2_r.nc && ~m_hit[act_k]));
    wire mshr_held = (t2_r.kind == K_LOAD) && ~mal_ready && (&acc_wait_n);
    wire draw_held = pool_draw && (pool_wait == WAIT_VICTIM_IN_FLIGHT) && (v_pin || v_jn)
                  && (old_count <= (L1_SLOT_W+1)'(LANES));
    wire split     = t2_valid && (t2_phase == P_LINES) && ((t2_r.kind == K_LOAD) || (t2_r.kind == K_STORE)) && act_any && ~act_fire && fin_free
                  && (way_held || mshr_held || draw_held)
                  && ((bld_rm == '0) || (rq_resv < (`CLOG2(RQ_DEPTH+1))'(RQ_DEPTH)));
    wire split_rq  = split && (bld_rm != '0);

    // fences and atomics after their lines
    wire amo_emit_last;
    reg [LANE_W-1:0] emit_lane;
    always @(*) begin
        emit_lane = '0;
        for (integer j = LANES-1; j >= 0; --j) begin
            if (t2_emit[j]) emit_lane = LANE_W'(j);
        end
    end
    assign amo_emit_last = ((t2_emit & ~(LANES'(1) << emit_lane)) == '0);

    wire t2_is_mem = (t2_r.kind == K_LOAD) || (t2_r.kind == K_STORE);
    wire req_end;
    wire t2_fin_now = req_lines_done && t2_is_mem;

    reg t2_done;
    reg emit_push;
    reg fence_push;
    always @(*) begin
        t2_done = 1'b0;
        emit_push = 1'b0;
        fence_push = 1'b0;
        inval_start = 1'b0;
        if (t2_valid) begin
            case (t2_phase)
            P_LINES: t2_done = lines_done && (~t2_last || (t2_is_mem && fin_free));
            P_EMIT: begin
                emit_push = ~eg_full;
                t2_done = emit_push && amo_emit_last;
            end
        `ifdef VX_CFG_H100_SM_ENABLE
            P_INV: begin
                inval_start = ~inv_wait;
                t2_done = t2_inv && inv_wait && ~inval_busy;
            end
        `else
            P_INV: inval_start = ~inv_wait;
        `endif
            P_PUSH: begin
                fence_push = ~eg_full;
                t2_done = fence_push;
            end
            default:;
            endcase
        end
    end

    assign req_end = t2_valid && t2_done && t2_last;
    wire [LWORD_W-1:0] e_wi = LWORD_W'(t2_r.addr[emit_lane]);
    wire [KEY_W-1:0] e_key = KEY_W'({t2_r.blk, t2_r.tag[LTID_W-1:0]});
`ifdef VX_CFG_EXT_ZACAS_ENABLE
    mem_bus_attr_t eg_cas;
    reg [WORD*8-1:0] eg_cmp;
`endif
    always @(*) begin
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        eg_cas    = '0;
        eg_cmp    = '0;
    `endif
        eg_push   = aeg_push;
        eg_rw     = aeg_rw;
        eg_addr   = MEM_ADDRW'(a_line);
        eg_data   = k_wdata[act_k];
        eg_byteen = aeg_byteen;
        eg_attr   = aeg_attr;
        eg_tag    = {a_uuid, aeg_idx};
        if (emit_push || fence_push) begin
            eg_push   = 1'b1;
            eg_rw     = 1'b0;
            eg_addr   = MEM_ADDRW'(t2_r.addr[emit_lane] >> LWORD_W);
            eg_data   = '0;
            eg_byteen = '1;
            eg_attr   = t2_r.attr[emit_lane];
            eg_tag    = {t2_r.tag[LTAG_W-1 -: UUID_WIDTH], 1'b1, IDX_W'(emit_lane)};
            if (emit_push) begin
                eg_tag = {t2_r.tag[LTAG_W-1 -: UUID_WIDTH], 1'b1, IDX_W'({e_key, e_wi, emit_lane})};
                eg_data[e_wi * WORD * 8 +: WORD * 8] = t2_r.data[emit_lane];
                eg_byteen = '0;
                eg_byteen[e_wi * WORD +: WORD] = t2_r.byteen[emit_lane];
            `ifdef VX_CFG_EXT_ZACAS_ENABLE
                // The comparand travels in the other 16-B half of the operand's
                // sector, at the operand's byte position in its word.
                eg_cas = mem_bus_attr_t'(t2_r.attr[emit_lane]);
                eg_cmp = (t2_r.byteen[emit_lane][WORD/2-1:0] == '0)
                       ? {eg_cas.amo.amo_cmp[WORD*4-1:0], (WORD*4)'(0)}
                       : (WORD*8)'(eg_cas.amo.amo_cmp);
                eg_data[((e_wi * WORD) ^ 16) * 8 +: WORD * 8] = eg_cmp;
                eg_cas.amo.amo_cmp = '0;
                eg_attr = MEM_ATTR_WIDTH'(eg_cas);
            `endif
            end
        end
    end

    // ---------------------------------------------------------------------
    // T1: issue the next batch; re-read the held lines otherwise.
    wire drained = ~t2_valid && ~fin_c_valid && ~fin_r_valid && rq_empty && ~a_valid
                && mshr_empty && ~fl_valid_r && ~alq_valid[0] && ~fn_valid_r && (amo_out == '0);
    wire in_gate = (in_r.kind == K_FENCE || (in_r.kind == K_LOAD && in_r_cta)) ? drained
                 : ((in_r.kind == K_AMO) ? ~fn_valid_r : 1'b1);
    wire t2_free = ~t2_valid || t2_done;
    wire t1_issue = in_valid && t2_free && in_gate && ~inval_busy;

    reg [TB-1:0] iss_v;
    reg [TB-1:0][LANE_W-1:0] iss_l;
    always @(*) begin
        for (integer k = 0; k < TB; ++k) begin
            iss_v[k] = 1'b0;
            iss_l[k] = '0;
            for (integer i = LANES-1; i >= 0; --i) begin
                if (in_todo[i] && (bank_of(set_of(LINE_W'(in_r.addr[i] >> LWORD_W))) == TB_W'(k))) begin
                    iss_v[k] = 1'b1;
                    iss_l[k] = LANE_W'(i);
                end
            end
        end
    end
    reg [LANES-1:0] iss_mask;
    always @(*) begin
        iss_mask = '0;
        for (integer k = 0; k < TB; ++k) begin
            if (iss_v[k]) iss_mask[iss_l[k]] = 1'b1;
        end
    end
    wire in_last = ((in_todo & ~iss_mask) == '0);

    wire [TB-1:0] held = t2_bv & ~res_next & {TB{t2_valid && ~t2_done}};
    reg [TB-1:0] lk_v;
    reg [TB-1:0][TROW_W-1:0] lk_r;
    always @(*) begin
        for (integer k = 0; k < TB; ++k) begin
            if (t1_issue) begin
                lk_v[k] = iss_v[k];
                lk_r[k] = trow_of(set_of(LINE_W'(in_r.addr[iss_l[k]] >> LWORD_W)));
            end else begin
                lk_v[k] = held[k];
                lk_r[k] = trow_of(set_of(t2_line[k]));
            end
        end
    end
    assign lk_valid = lk_v;
    assign lk_row   = lk_r;

    // a read under a tag write not yet visible is looked at again
    reg [TB-1:0] hz;
    always @(*) begin
        for (integer k = 0; k < TB; ++k) begin
            hz[k] = (alq_valid[0] && (alq_bank[0] == TB_W'(k)) && (alq_row[0] == lk_r[k]))
                 || (alq_valid[1] && (alq_bank[1] == TB_W'(k)) && (alq_row[1] == lk_r[k]))
                 || (act_al && (act_k == TB_W'(k)) && (a_row == lk_r[k]))
                 || (up_valid[0] && up_ready[0] && (up_bank[0] == TB_W'(k)) && (up_row[0] == lk_r[k]))
                 || (up_valid[1] && up_ready[1] && (up_bank[1] == TB_W'(k)) && (up_row[1] == lk_r[k]));
        end
    end

    wire in_free = ~in_valid || (t1_issue && in_last);
    assign in_load  = (arb_valid || hf_sel) && in_free && ((rq_resv + (`CLOG2(RQ_DEPTH+1))'(split_rq)) < (`CLOG2(RQ_DEPTH+1))'(RQ_DEPTH));
    assign lsu_load = in_load && arb_valid;

    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_req_ready
        assign lsu_if[b].req_ready = lsu_load && arb_onehot[b];
    end

    // ---------------------------------------------------------------------
    // FIN and the commit item.
    assign fin_free = ~fin_c_valid || fin_c_take;

    // ---------------------------------------------------------------------
    // Fill: the entry's row into the array after its interlock, then its valid bits.
    wire ilk_block = fe_ilk_block;
    assign ilk_fin_take = fin_c_take;

    // replays queued on entries this cycle; a replay leaves when the array retires it
    always @(*) begin
        rp_inc = '0;
        for (integer k = 0; k < TB; ++k) begin
            if (res_now[k] && res_cls[k] == R_JOIN) begin
                rp_inc[res_wid[k]] = 1'b1;
            end
        end
    end
    assign rp_dec = ~wf_retired_valid ? '0
                  : gl_fire ? (take_rep ? rq_head_wait : '0)
                  : (a_rep ? a_wait : '0);
    reg [VB-1:0] fe_words;
    reg [LINE-1:0] fe_byteen;
    always @(*) begin
        for (integer s = 0; s < NSEC; ++s) begin
            fe_words[s * SVB +: SVB] = {SVB{fe_secs[s]}};
        end
        for (integer v = 0; v < VB; ++v) begin
            fe_byteen[v * VBYTES +: VBYTES] = {VBYTES{fe_words[v] && ~fe_vmiss[v]}};
        end
    end
    wire fl_go = fl_valid_r && ~ilk_block;
    // room for the fill's valid-set beside this cycle's one store or victim
    // update, so the offer depends on registers only and stays until taken
    assign fill_valid = fl_go && (~fe_cacheable || q_room2[bank_of(fe_set)]);
    `RUNTIME_ASSERT(~up_valid[2] || up_ready[2], ("%t: *** %s fill valid-set without room", $time, INSTANCE_ID))
    assign fill_data.row    = SM_SRAM_ROW_W'(ROWS - 1) - SM_SRAM_ROW_W'(fe_slot);
    assign fill_data.byteen = fe_byteen;
    assign fill_data.data   = fl_data_r;
    assign fill_data.src    = SM_SRAM_SRC_L1;
    assign fill_fire = fill_valid && fill_ready;
    always @(*) begin
        up_valid[0] = u0_valid;
        up_bank[0]  = act_k;
        up_row[0]   = a_row;
        up_way[0]   = tway[act_k];
        up_mask[0]  = u0_mask;
        up_set[0]   = u0_set;
        up_inval[0] = 1'b0;
        up_id[0]    = '0;
        up_valid[1] = victim_valid;
        up_bank[1]  = bank_of(victim_set);
        up_row[1]   = trow_of(victim_set);
        up_way[1]   = victim_way;
        up_mask[1]  = '1;
        up_set[1]   = 1'b0;
        up_inval[1] = 1'b1;
        up_id[1]    = '0;
        up_valid[2] = fl_go && fe_cacheable && fill_ready;
        up_bank[2]  = bank_of(fe_set);
        up_row[2]   = trow_of(fe_set);
        up_way[2]   = fe_way;
        up_mask[2]  = fe_words;
        up_set[2]   = 1'b1;
        up_inval[2] = 1'b0;
        up_id[2]    = fl_id_r;
    end


    // ---------------------------------------------------------------------
    // Responses to the LSU blocks.
    reg              rs_valid;
    reg [BLK_W-1:0]  rs_blk;
    reg [LANES-1:0]  rs_mask;
    reg [LANES-1:0][WORD*8-1:0] rs_data;
    reg [LTAG_W-1:0] rs_tag;
    wire [NUM_BLOCKS-1:0] blk_rsp_ready;
    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_rsp
        assign lsu_if[b].rsp_valid = rs_valid && (rs_blk == BLK_W'(b));
        assign lsu_if[b].rsp_data.mask = rs_mask;
        assign lsu_if[b].rsp_data.data = rs_data;
        assign lsu_if[b].rsp_data.tag  = rs_tag;
        assign blk_rsp_ready[b] = lsu_if[b].rsp_ready;
    end
    wire rs_free = ~rs_valid || blk_rsp_ready[rs_blk];

    // miss-bus responses: fills by entry; a fence's lane, or an atomic lane straight to its LSU block
    wire rsp_special = mem_bus_if.rsp_data.tag.value[IDX_W];
    wire [IDX_W-1:0] rsp_idx = mem_bus_if.rsp_data.tag.value[IDX_W-1:0];
    wire [LANE_W-1:0] rsp_lane = rsp_idx[LANE_W-1:0];
    wire [LWORD_W-1:0] rsp_wi = rsp_idx[LANE_W +: LWORD_W];
    wire [KEY_W-1:0] rsp_key = rsp_idx[LANE_W + LWORD_W +: KEY_W];
    wire rsp_amo = rsp_special && ~fn_valid_r;
    assign mem_bus_if.rsp_ready = rsp_special ? (~rsp_amo || rs_free) : (~fl_valid_r || fill_fire);
    wire mem_rsp_fire = mem_bus_if.rsp_valid && mem_bus_if.rsp_ready;
    wire amo_rsp = mem_rsp_fire && rsp_amo;
    wire [WORD*8-1:0] rsp_word = mem_bus_if.rsp_data.data[rsp_wi * WORD * 8 +: WORD * 8];

    wire fn_done = fn_valid_r && fn_got;
    assign gl_rsp_ready = rs_free && ~fn_done && ~amo_rsp;
    wire [LMEM_TAG_WIDTH-1:0] gl_tag = gl_rsp_data.tag;
    `UNUSED_VAR (gl_rsp_data.cls)

    // ---------------------------------------------------------------------
    // Sequential state.
    always @(posedge clk) begin
        if (reset) begin
            in_valid    <= 1'b0;
            in_todo     <= '0;
            in_r_cta    <= 1'b0;
        `ifdef VX_CFG_H100_SM_ENABLE
            in_r_inv    <= 1'b0;
            t2_inv      <= 1'b0;
        `endif
            in_r_host   <= 1'b0;
            t2_host     <= 1'b0;
            fn_host     <= 1'b0;
            hf_sent     <= 1'b0;
            hf_done     <= 1'b0;
            seq_ctr     <= '0;
            t2_valid    <= 1'b0;
            t2_res      <= '0;
            t2_stale    <= '0;
            t2_phase    <= P_LINES;
            inv_wait    <= 1'b0;
            fin_c_valid <= 1'b0;
            fin_r_valid <= 1'b0;
            a_valid     <= 1'b0;
            gl_off_c    <= 1'b0;
            rq_resv     <= '0;
            alq_valid   <= '0;
            fn_valid_r  <= 1'b0;
            amo_out     <= '0;
            fl_valid_r  <= 1'b0;
            done_r      <= '0;
            draw_on     <= 1'b0;
            rs_valid    <= 1'b0;
        end else begin
            // IN
            if (in_load) begin
                in_valid       <= 1'b1;
                in_r.kind      <= sel_cta ? K_LOAD : sel_kind;
                in_r_cta       <= sel_cta;
            `ifdef VX_CFG_H100_SM_ENABLE
                in_r_inv       <= sel_inv;
            `endif
                in_r_host      <= hf_sel;
                in_r.nc        <= sel_a.is_addr_io;
                in_r.blk       <= arb_idx;
                in_r.tag       <= sel_tag;
                in_r.seq       <= seq_ctr;
                in_r.mask      <= sel_mask;
                in_r.lead      <= sel_lead;
                in_r.lead_idx  <= sel_lead_idx;
                in_r.addr      <= sel_addr;
                in_r.data      <= sel_data;
                in_r.byteen    <= sel_byteen;
                in_r.attr      <= sel_attr;
                in_todo        <= sel_lead;
                seq_ctr        <= seq_ctr + L1_SEQ_W'(1);
            end else if (t1_issue) begin
                in_todo <= in_todo & ~iss_mask;
                if (in_last) in_valid <= 1'b0;
            end

            // T2
            if (t1_issue) begin
                t2_valid <= 1'b1;
                t2_first <= (in_todo == in_r.lead);
                t2_last  <= in_last;
                t2_r     <= in_r;
                t2_host  <= in_r_host;
            `ifdef VX_CFG_H100_SM_ENABLE
                t2_inv   <= in_r_inv;
            `endif
                t2_bv    <= iss_v;
                t2_bl    <= iss_l;
                t2_res   <= '0;
                t2_stale <= hz;
                t2_phase <= P_LINES;
                t2_emit  <= in_r.mask;
            end else if (t2_done) begin
                t2_valid <= 1'b0;
            end else if (t2_valid) begin
                t2_res   <= res_next;
                t2_stale <= hz & held;
                t2_first <= split;
                if (t2_phase == P_LINES && lines_done && t2_last && ~t2_is_mem) begin
                    t2_phase <= (t2_r.kind == K_AMO) ? P_EMIT : P_INV;
                end
                if (emit_push) begin
                    t2_emit[emit_lane] <= 1'b0;
                end
                if (t2_phase == P_INV) begin
                    if (inval_start) begin
                        inv_wait <= 1'b1;
                    end else if (~inval_busy) begin
                        inv_wait <= 1'b0;
                        t2_phase <= P_PUSH;
                    end
                end
            end
            if (t2_valid && (t2_first || res_now != '0)) begin
                acc_cls  <= acc_cls_n;
                acc_slot <= acc_slot_n;
                acc_wait <= acc_wait_n;
            end

            // draws in progress
            if (draw_start) draw_on <= 1'b1;
            if (draw_done)  draw_on <= 1'b0;

            // allocation writes
            begin
                reg [ALQ-1:0] v;
                v = alq_valid;
                if (al_grant) begin
                    for (integer q = 0; q < ALQ-1; ++q) begin
                        alq_bank[q]  <= alq_bank[q+1];
                        alq_row[q]   <= alq_row[q+1];
                        alq_way[q]   <= alq_way[q+1];
                        alq_tag[q]   <= alq_tag[q+1];
                        alq_slot[q]  <= alq_slot[q+1];
                        alq_vmask[q] <= alq_vmask[q+1];
                    end
                    v = v >> 1;
                end
                if (act_fire && act_al) begin
                    for (integer q = ALQ-1; q >= 0; --q) begin
                        if (v == ALQ'((1 << q) - 1)) begin
                            alq_bank[q]  <= act_k;
                            alq_row[q]   <= a_row;
                            alq_way[q]   <= act_al_way;
                            alq_tag[q]   <= tag_of(a_line);
                            alq_slot[q]  <= act_al_slot;
                            alq_vmask[q] <= act_al_vm;
                        end
                    end
                    v = (v << 1) | ALQ'(1);
                end
                alq_valid <= v;
            end

            // FIN
            if (fin_c_take) fin_c_valid <= 1'b0;
            if (rq_push)    fin_r_valid <= 1'b0;
            if ((t2_fin_now && t2_done) || split) begin
                fin_c_valid <= (bld_cm != '0);
                fin_r_valid <= (bld_rm != '0);
                fin_c       <= bld_c;
                fin_r       <= bld_r;
                fin_wait    <= acc_wait_n;
            end

            // replay reservations: taken at IN, given back without a replay item or when it leaves
            rq_resv <= rq_resv + (`CLOG2(RQ_DEPTH+1))'(in_load) + (`CLOG2(RQ_DEPTH+1))'(split_rq)
                     - (`CLOG2(RQ_DEPTH+1))'(req_end && ~(t2_is_mem && (bld_rm != '0)))
                     - (`CLOG2(RQ_DEPTH+1))'(rq_pop);

            // the array's global item in flight (a one-wavefront item retires in the cycle it is taken)
            if (gl_fire) a_valid <= ~wf_retired_valid;
            else if (wf_retired_valid) a_valid <= 1'b0;
            gl_off_c <= gl_req_valid && ~gl_fire && ~take_rep;
            if (gl_fire) begin
                a_rep   <= take_rep;
                a_seq   <= gl_req_data.seq;
                a_wait  <= take_rep ? rq_head_wait : '0;
            end

            // fence outstanding, atomic lanes in flight
            if (fn_done && rs_free) fn_valid_r <= 1'b0;
            if (t2_valid && (t2_phase == P_LINES) && lines_done && t2_last && (t2_r.kind == K_FENCE)) begin
                fn_valid_r <= 1'b1;
                fn_blk     <= t2_r.blk;
                fn_tag     <= t2_r.tag;
                fn_host    <= t2_host;
                fn_mask    <= t2_r.mask;
                fn_got     <= 1'b0;
            end
            if (mem_rsp_fire && rsp_special && ~rsp_amo) begin
                fn_got <= 1'b1;
            end
        `ifdef VX_CFG_H100_SM_ENABLE
            if (t2_valid && t2_done && (t2_phase == P_INV)) begin
                inv_wait <= 1'b0;
                fn_got   <= 1'b1;
            end
        `endif
            amo_out <= amo_out + AMO_OUT_W'(emit_push) - AMO_OUT_W'(amo_rsp);

            // fills
            if (fill_fire) fl_valid_r <= 1'b0;
            if (mem_rsp_fire && ~rsp_special) begin
                fl_valid_r <= 1'b1;
                fl_id_r    <= ID_W'(rsp_idx);
                fl_data_r  <= mem_bus_if.rsp_data.data;
            end
            for (integer b = 0; b < TB; ++b) begin
                if (tw_valid[b] && (tw_src[b] == 2'd2)) done_r[tw_id[b]] <= 1'b1;
            end
            if (fill_fire && ~fe_cacheable) done_r[fl_id_r] <= 1'b1;
            if (mal_valid && mal_ready) done_r[mal_id] <= 1'b0;

            // host flush
            if (~dcr_flush_if.req) begin
                hf_sent <= 1'b0;
                hf_done <= 1'b0;
            end else if (in_load && hf_sel) begin
                hf_sent <= 1'b1;
            end

            // responses
            if (rs_free) rs_valid <= 1'b0;
            if (fn_done && rs_free && fn_host) begin
                hf_done  <= 1'b1;
            end else if (fn_done && rs_free) begin
                rs_valid <= 1'b1;
                rs_blk   <= fn_blk;
                rs_mask  <= fn_mask;
                rs_data  <= '0;
                rs_tag   <= fn_tag;
            end else if (amo_rsp) begin
                rs_valid <= 1'b1;
                rs_blk   <= BLK_W'(rsp_key >> LTID_W);
                rs_mask  <= LANES'(1) << rsp_lane;
                rs_data  <= {LANES{rsp_word}};
                rs_tag   <= {mem_bus_if.rsp_data.tag.uuid, rsp_key[LTID_W-1:0]};
            end else if (gl_rsp_valid && gl_rsp_ready) begin
                rs_valid <= 1'b1;
                rs_blk   <= BLK_W'(gl_tag >> LTAG_W);
                rs_mask  <= gl_rsp_data.mask;
                rs_data  <= gl_rsp_data.data;
                rs_tag   <= LTAG_W'(gl_tag);
            end
        end
    end

    assign dcr_flush_if.done = hf_done;

    // the host flush's own work is not busy: the core raises the flush only when the L1 is idle
    assign busy = (in_valid || t2_valid || fin_c_valid || fin_r_valid || ~rq_empty || a_valid
                || ~mshr_empty || fl_valid_r || ~eg_empty || alq_valid[0] || fn_valid_r || (amo_out != '0) || rs_valid)
               && ~(hf_sent && ~hf_done);

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_reads, perf_writes, perf_misses, perf_mshr_stalls;
    wire [`CLOG2(TB+1)-1:0] res_cnt = `CLOG2(TB+1)'($countones(res_now));
    always @(posedge clk) begin
        if (reset) begin
            perf_reads       <= '0;
            perf_writes      <= '0;
            perf_misses      <= '0;
            perf_mshr_stalls <= '0;
        end else begin
            if (t2_r.kind == K_LOAD)  perf_reads  <= perf_reads + PERF_CTR_BITS'(res_cnt);
            if (t2_r.kind == K_STORE) perf_writes <= perf_writes + PERF_CTR_BITS'(res_cnt);
            if (mal_valid) perf_misses <= perf_misses + PERF_CTR_BITS'(1);
            if (act_mem && (t2_r.kind == K_LOAD) && ~mal_ready) perf_mshr_stalls <= perf_mshr_stalls + PERF_CTR_BITS'(1);
        end
    end
    assign perf.reads        = perf_reads;
    assign perf.writes       = perf_writes;
    assign perf.read_misses  = perf_misses;
    assign perf.write_misses = '0;
    assign perf.evictions    = '0;
    assign perf.bank_stalls  = '0;
    assign perf.mshr_stalls  = perf_mshr_stalls;
    assign perf.mem_stalls   = '0;
    assign perf.crsp_stalls  = '0;
`endif

    `RUNTIME_ASSERT(~wf_retired_valid || a_valid || gl_fire, ("%t: %s wavefront retirement with no global item", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(~wf_retired_valid || (wf_retired_seq == (gl_fire ? gl_req_data.seq : a_seq)), ("%t: %s global items retired out of order", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(~(mem_rsp_fire && ~rsp_special) || ~fl_valid_r || fill_fire, ("%t: %s fill register overrun", $time, INSTANCE_ID))
    // an entry with no sector to fetch is never filled and never retires
    `RUNTIME_ASSERT(~(mal_valid && mal_ready) || (mal_secs != '0), ("%t: %s line read with no sector", $time, INSTANCE_ID))

endmodule

`endif // VX_CFG_H100_SM_ENABLE
