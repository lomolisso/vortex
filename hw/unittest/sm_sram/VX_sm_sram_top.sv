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

module VX_sm_sram_top import VX_gpu_pkg::*; #(
    parameter NUM_BLOCKS = `VX_CFG_NUM_LSU_BLOCKS,
    parameter NUM_LANES  = `VX_CFG_NUM_LSU_LANES,
    parameter WORD_SIZE  = LSU_WORD_SIZE,
    // Word addresses wide enough for every shared row of the array
    localparam ADDR_WIDTH      = SM_SRAM_BOFF_W - `CLOG2(WORD_SIZE) + SM_SRAM_ROW_W,
    localparam TAG_VALUE_WIDTH = 6,
    localparam ROW_TAG_W       = 8,
    localparam RD_ROWS         = 3,
    localparam AMO_OP_WIDTH    = $bits(amo_op_e),
    localparam TAG_WIDTH       = LSU_TAG_WIDTH,
    localparam WORD_WIDTH      = WORD_SIZE * 8,
    localparam ROW_DATA_W      = SM_SRAM_ROW_BYTES * 8,
    localparam CNT_W           = `CLOG2(NUM_LANES + 1),
    localparam GL_ADDR_WIDTH   = LSU_ADDR_WIDTH,
    localparam LINE_SIZE       = SM_SRAM_ROW_BYTES,
    localparam MEM_ADDR_W      = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(LINE_SIZE),
    localparam MEM_TAG_W       = SM_L1_MEM_TAG_WIDTH,
    localparam FILL_W          = $bits(sm_sram_fill_t),
    localparam BANK_W          = `CLOG2(`VX_CFG_LMEM_NUM_BANKS)
) (
    input wire clk,
    input wire reset,

    input  wire [31:0]                                    cfg_s_rows,
    input  wire [31:0]                                    cfg_l1_lines,
    input  wire [31:0]                                    cfg_l1_protect,

    // LSU blocks' shared-memory warp requests
    input  wire [NUM_BLOCKS-1:0]                          in_valid,
    input  wire [NUM_BLOCKS-1:0]                          in_rw,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]           in_mask,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][ADDR_WIDTH-1:0] in_addr,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_SIZE-1:0]  in_byteen,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_WIDTH-1:0] in_data,
    input  wire [NUM_BLOCKS-1:0]                          in_amo,
    input  wire [NUM_BLOCKS-1:0][AMO_OP_WIDTH-1:0]        in_amo_op,
    input  wire [NUM_BLOCKS-1:0]                          in_lacc,
    input  wire [NUM_BLOCKS-1:0][TAG_VALUE_WIDTH-1:0]     in_tag,
    output wire [NUM_BLOCKS-1:0]                          in_ready,
    output wire [NUM_BLOCKS-1:0]                          in_rsp_valid,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]           in_rsp_mask,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_WIDTH-1:0] in_rsp_data,
    output wire [NUM_BLOCKS-1:0][TAG_VALUE_WIDTH-1:0]     in_rsp_tag,
    input  wire [NUM_BLOCKS-1:0]                          in_rsp_ready,

    // Row-read clients (TCU, DXA, DSMEM)
    input  wire [RD_ROWS-1:0]                             rd_valid,
    input  wire [RD_ROWS-1:0][31:0]                       rd_row,
    input  wire [RD_ROWS-1:0][ROW_TAG_W-1:0]              rd_tag,
    output wire [RD_ROWS-1:0]                             rd_ready,
    output wire [RD_ROWS-1:0]                             rd_rsp_valid,
    output wire [ROW_DATA_W-1:0]                          rd_rsp_data,
    output wire [ROW_TAG_W-1:0]                           rd_rsp_tag,

    // Counted lanes once final: wavefront, LSU atomic, mbarrier op
    output wire [2:0]                                     acc_commit_valid,
    output wire [2:0][31:0]                               acc_commit_wid,
    output wire [2:0][CNT_W-1:0]                          acc_commit_cnt,

    // LSU blocks' global warp requests into the L1
    input  wire [NUM_BLOCKS-1:0]                          gl_in_valid,
    input  wire [NUM_BLOCKS-1:0]                          gl_in_rw,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]           gl_in_mask,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][GL_ADDR_WIDTH-1:0] gl_in_addr,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_SIZE-1:0]  gl_in_byteen,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_WIDTH-1:0] gl_in_data,
    input  wire [NUM_BLOCKS-1:0][MEM_ATTR_WIDTH-1:0]      gl_in_attr,
    input  wire [NUM_BLOCKS-1:0][TAG_WIDTH-1:0]           gl_in_tag,
    output wire [NUM_BLOCKS-1:0]                          gl_in_ready,
    output wire [NUM_BLOCKS-1:0]                          gl_in_rsp_valid,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]           gl_in_rsp_mask,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_WIDTH-1:0] gl_in_rsp_data,
    output wire [NUM_BLOCKS-1:0][TAG_WIDTH-1:0]           gl_in_rsp_tag,
    input  wire [NUM_BLOCKS-1:0]                          gl_in_rsp_ready,
    output wire [15:0]                                    gl_tag_bits,
    output wire [15:0]                                    pkg_lmem_pipe,
    output wire [15:0]                                    pkg_l1_pipe,
    output wire [15:0]                                    pkg_l1_miss_pipe,

    // cp.async lane writes
    input  wire [NUM_BLOCKS-1:0]                          cpa_valid,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]           cpa_mask,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][ADDR_WIDTH-1:0] cpa_addr,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_SIZE-1:0]  cpa_byteen,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_WIDTH-1:0] cpa_data,
    input  wire [NUM_BLOCKS-1:0][TAG_WIDTH-1:0]           cpa_tag,
    input  wire [NUM_BLOCKS-1:0]                          cpa_amo,
    input  wire [NUM_BLOCKS-1:0][AMO_OP_WIDTH-1:0]        cpa_amo_op,
    output wire [NUM_BLOCKS-1:0]                          cpa_ready,
    output wire [NUM_BLOCKS-1:0]                          cpa_rsp_valid,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]           cpa_rsp_mask,
    output wire [NUM_BLOCKS-1:0][TAG_WIDTH-1:0]           cpa_rsp_tag,
    input  wire [NUM_BLOCKS-1:0]                          cpa_rsp_ready,

    // Row-write clients after the L1 refill (DXA, DSMEM)
    input  wire [1:0]                                     wr_valid,
    input  wire [1:0][FILL_W-1:0]                         wr_data,
    input  wire                                           wr_dsmem_last,
    output wire [1:0]                                     wr_ready,
    output wire                                           wr_dsmem_busy,
    output wire                                           fill_done_valid,
    output wire [SM_SRAM_ROW_W-1:0]                       fill_done_row,

    // One-lane atomic clients (DXA completion, DSMEM)
    input  wire [1:0]                                     amo_valid,
    input  wire [1:0][SM_SRAM_ROW_W-1:0]                  amo_row,
    input  wire [1:0][BANK_W-1:0]                         amo_bank,
    input  wire [1:0][LSU_WORD_SIZE-1:0]                  amo_byteen,
    input  wire [1:0][LSU_WORD_SIZE*8-1:0]                amo_wdata,
    input  wire [1:0][AMO_REQ_BITS-1:0]                   amo_attr,
    input  wire [1:0][ROW_TAG_W-1:0]                      amo_tag,
    output wire [1:0]                                     amo_ready,
    output wire [1:0]                                     amo_rsp_valid,
    output wire [1:0][LSU_WORD_SIZE*8-1:0]                amo_rsp_data,
    output wire [1:0][ROW_TAG_W-1:0]                      amo_rsp_tag,
    // Width of one amo_attr entry, for the bench's packing
    output wire [15:0]                                    amo_attr_bits,

`ifdef VX_CFG_EXT_MBAR_ENABLE
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0]              mbar_evt_valid,
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0]              mbar_evt_flip,
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0][NW_WIDTH-1:0] mbar_evt_wid,
    output wire [`VX_CFG_LMEM_NUM_BANKS-1:0][SM_SRAM_ROW_W-1:0] mbar_evt_row,
`endif

    // The L1's line port to the SM's egress / ingress
    output wire                                           mem_req_valid,
    output wire                                           mem_req_rw,
    output wire [MEM_ADDR_W-1:0]                          mem_req_addr,
    output wire [LINE_SIZE*8-1:0]                         mem_req_data,
    output wire [LINE_SIZE-1:0]                           mem_req_byteen,
    output wire [MEM_ATTR_WIDTH-1:0]                      mem_req_attr,
    output wire [MEM_TAG_W-1:0]                           mem_req_tag,
    input  wire                                           mem_req_ready,
    input  wire                                           mem_rsp_valid,
    input  wire [LINE_SIZE*8-1:0]                         mem_rsp_data,
    input  wire [MEM_TAG_W-1:0]                           mem_rsp_tag,
    output wire                                           mem_rsp_ready,

    input  wire                                           flush_req,
    output wire                                           flush_done,
    output wire                                           l1_busy,

    // the L1's refill rows as the array takes them (slot = rows - 1 - row)
    output wire                                           l1_fill_fire,
    output wire [SM_SRAM_ROW_W-1:0]                       l1_fill_row
);
    `STATIC_ASSERT (SM_SRAM_ROW_W <= 32, ("row index wider than the bench's row ports"))

    smem_cfg_t cfg;
    assign cfg.s_rows     = $bits(cfg.s_rows)'(cfg_s_rows);
    assign cfg.l1_lines   = $bits(cfg.l1_lines)'(cfg_l1_lines);
    assign cfg.l1_protect = $bits(cfg.l1_protect)'(cfg_l1_protect);
    `UNUSED_VAR (cfg_s_rows)
    `UNUSED_VAR (cfg_l1_lines)
    `UNUSED_VAR (cfg_l1_protect)

    VX_lsu_mem_if #(
        .NUM_LANES  (NUM_LANES),
        .DATA_SIZE  (WORD_SIZE),
        .TAG_WIDTH  (TAG_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) lsu_if[NUM_BLOCKS]();

    VX_lsu_mem_if #(
        .NUM_LANES  (NUM_LANES),
        .DATA_SIZE  (WORD_SIZE),
        .TAG_WIDTH  (TAG_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) cpa_if[NUM_BLOCKS]();

    VX_lsu_mem_if #(
        .NUM_LANES  (NUM_LANES),
        .DATA_SIZE  (WORD_SIZE),
        .TAG_WIDTH  (TAG_WIDTH),
        .ADDR_WIDTH (GL_ADDR_WIDTH)
    ) gl_lsu_if[NUM_BLOCKS]();

    for (genvar b = 0; b < NUM_BLOCKS; ++b) begin : g_blk
        mem_bus_attr_t attr;
        always @(*) begin
            attr = '0;
            attr.is_addr_local = 1'b1;
            attr.amo.amo_valid = in_amo[b];
            attr.amo.amo_op    = amo_op_e'(in_amo_op[b]);
        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            attr.is_lacc       = in_lacc[b];
        `endif
        end
    `ifndef VX_CFG_EXT_DLMEM_ENABLE
        `UNUSED_VAR (in_lacc[b])
    `endif
        assign lsu_if[b].req_valid        = in_valid[b];
        assign lsu_if[b].req_data.mask    = in_mask[b];
        assign lsu_if[b].req_data.rw      = in_rw[b];
        assign lsu_if[b].req_data.addr    = in_addr[b];
        assign lsu_if[b].req_data.data    = in_data[b];
        assign lsu_if[b].req_data.byteen  = in_byteen[b];
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_user
            assign lsu_if[b].req_data.user[i] = attr;
        end
        assign lsu_if[b].req_data.tag     = TAG_WIDTH'(in_tag[b]);
        assign in_ready[b]                = lsu_if[b].req_ready;
        assign in_rsp_valid[b]            = lsu_if[b].rsp_valid;
        assign in_rsp_mask[b]             = lsu_if[b].rsp_data.mask;
        assign in_rsp_data[b]             = lsu_if[b].rsp_data.data;
        assign in_rsp_tag[b]              = TAG_VALUE_WIDTH'(lsu_if[b].rsp_data.tag);
        assign lsu_if[b].rsp_ready        = in_rsp_ready[b];
        `UNUSED_VAR (lsu_if[b].rsp_data.tag)

        // The copy unit's mbarrier op: its attribute as VX_cpa_unit sends it.
        mem_bus_attr_t cpa_attr;
        always @(*) begin
            cpa_attr = '0;
            cpa_attr.amo.amo_valid = cpa_amo[b];
            cpa_attr.amo.amo_op    = amo_op_e'(cpa_amo_op[b]);
        end
        assign cpa_if[b].req_valid        = cpa_valid[b];
        assign cpa_if[b].req_data.mask    = cpa_mask[b];
        assign cpa_if[b].req_data.rw      = ~cpa_amo[b];
        assign cpa_if[b].req_data.addr    = cpa_addr[b];
        assign cpa_if[b].req_data.data    = cpa_data[b];
        assign cpa_if[b].req_data.byteen  = cpa_byteen[b];
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cpa_user
            assign cpa_if[b].req_data.user[i] = cpa_attr;
        end
        assign cpa_if[b].req_data.tag     = cpa_tag[b];
        assign cpa_ready[b]               = cpa_if[b].req_ready;
        assign cpa_rsp_valid[b]           = cpa_if[b].rsp_valid;
        assign cpa_rsp_mask[b]            = cpa_if[b].rsp_data.mask;
        assign cpa_rsp_tag[b]             = cpa_if[b].rsp_data.tag;
        assign cpa_if[b].rsp_ready        = cpa_rsp_ready[b];
        `UNUSED_VAR (cpa_if[b].rsp_data.data)

        assign gl_lsu_if[b].req_valid       = gl_in_valid[b];
        assign gl_lsu_if[b].req_data.mask   = gl_in_mask[b];
        assign gl_lsu_if[b].req_data.rw     = gl_in_rw[b];
        assign gl_lsu_if[b].req_data.addr   = gl_in_addr[b];
        assign gl_lsu_if[b].req_data.data   = gl_in_data[b];
        assign gl_lsu_if[b].req_data.byteen = gl_in_byteen[b];
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_gl_user
            assign gl_lsu_if[b].req_data.user[i] = gl_in_attr[b];
        end
        assign gl_lsu_if[b].req_data.tag    = gl_in_tag[b];
        assign gl_in_ready[b]               = gl_lsu_if[b].req_ready;
        assign gl_in_rsp_valid[b]           = gl_lsu_if[b].rsp_valid;
        assign gl_in_rsp_mask[b]            = gl_lsu_if[b].rsp_data.mask;
        assign gl_in_rsp_data[b]            = gl_lsu_if[b].rsp_data.data;
        assign gl_in_rsp_tag[b]             = gl_lsu_if[b].rsp_data.tag;
        assign gl_lsu_if[b].rsp_ready       = gl_in_rsp_ready[b];
    end

    wire [RD_ROWS-1:0][SM_SRAM_ROW_W-1:0] rd_row_w;
    for (genvar r = 0; r < RD_ROWS; ++r) begin : g_rd
        assign rd_row_w[r] = SM_SRAM_ROW_W'(rd_row[r]);
    end
    `UNUSED_VAR (rd_row)

    wire [2:0][NW_WIDTH-1:0] acc_wid;
    for (genvar k = 0; k < 3; ++k) begin : g_acc
        assign acc_commit_wid[k] = 32'(acc_wid[k]);
    end

    wire          gl_req_valid, gl_req_ready, gl_rsp_valid, gl_rsp_ready, wf_retired_valid;
    sm_sram_req_t gl_req_data;
    sm_sram_rsp_t gl_rsp_data;
    wire [L1_SEQ_W-1:0] wf_retired_seq;
    wire          l1_fill_valid, l1_fill_ready;
    sm_sram_fill_t l1_fill_data;

    sm_sram_fill_t [2:0] wr_data_w;
    wire [2:0] wr_ready_w;
    assign wr_data_w[0] = l1_fill_data;
    for (genvar w = 0; w < 2; ++w) begin : g_wr
        assign wr_data_w[w+1] = sm_sram_fill_t'(wr_data[w]);
    end
    assign l1_fill_ready = wr_ready_w[0];
    assign l1_fill_fire  = l1_fill_valid && l1_fill_ready;
    assign l1_fill_row   = l1_fill_data.row;
    assign wr_ready      = wr_ready_w[2:1];

    amo_req_t [1:0] amo_attr_w;
    for (genvar a = 0; a < 2; ++a) begin : g_amo
        assign amo_attr_w[a] = amo_req_t'(amo_attr[a]);
    end
    assign amo_attr_bits = 16'(AMO_REQ_BITS);
    assign gl_tag_bits   = 16'(TAG_WIDTH);
    assign pkg_lmem_pipe = 16'(LMEM_PIPE_OVERHEAD);
    assign pkg_l1_pipe   = 16'(L1_PIPE_OVERHEAD);
    assign pkg_l1_miss_pipe = 16'(L1_MISS_PIPE_OVERHEAD);

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (MEM_TAG_W)
    ) l1_mem_if();

    assign mem_req_valid  = l1_mem_if.req_valid;
    assign mem_req_rw     = l1_mem_if.req_data.rw;
    assign mem_req_addr   = l1_mem_if.req_data.addr;
    assign mem_req_data   = l1_mem_if.req_data.data;
    assign mem_req_byteen = l1_mem_if.req_data.byteen;
    assign mem_req_attr   = l1_mem_if.req_data.attr;
    assign mem_req_tag    = l1_mem_if.req_data.tag;
    assign l1_mem_if.req_ready = mem_req_ready;
    assign l1_mem_if.rsp_valid = mem_rsp_valid;
    assign l1_mem_if.rsp_data.data = mem_rsp_data;
    assign l1_mem_if.rsp_data.tag  = mem_rsp_tag;
    assign mem_rsp_ready  = l1_mem_if.rsp_ready;

    VX_dcr_flush_if l1_flush_if();
    assign l1_flush_if.req = flush_req;
    assign flush_done      = l1_flush_if.done;

`ifdef PERF_ENABLE
    cache_perf_t l1_perf;
    `UNUSED_VAR (l1_perf)
`endif

    VX_sm_l1 #(
        .INSTANCE_ID   ("sm_l1"),
        .NUM_BLOCKS    (NUM_BLOCKS),
        .MEM_TAG_WIDTH (MEM_TAG_W)
    ) sm_l1 (
        .clk              (clk),
        .reset            (reset),
        .cfg              (cfg),
        .lsu_if           (gl_lsu_if),
        .gl_req_valid     (gl_req_valid),
        .gl_req_data      (gl_req_data),
        .gl_req_ready     (gl_req_ready),
        .gl_rsp_valid     (gl_rsp_valid),
        .gl_rsp_data      (gl_rsp_data),
        .gl_rsp_ready     (gl_rsp_ready),
        .wf_retired_valid (wf_retired_valid),
        .wf_retired_seq   (wf_retired_seq),
        .fill_valid       (l1_fill_valid),
        .fill_data        (l1_fill_data),
        .fill_ready       (l1_fill_ready),
        .mem_bus_if       (l1_mem_if),
        .dcr_flush_if     (l1_flush_if),
    `ifdef PERF_ENABLE
        .perf             (l1_perf),
    `endif
        .busy             (l1_busy)
    );

    VX_sm_sram #(
        .INSTANCE_ID      ("sm_sram"),
        .NUM_BLOCKS       (NUM_BLOCKS),
        .ROW_TAG_W        (ROW_TAG_W)
    ) sm_sram (
        .clk              (clk),
        .reset            (reset),
        .cfg              (cfg),
        .lsu_if           (lsu_if),
        .gl_req_valid     (gl_req_valid),
        .gl_req_data      (gl_req_data),
        .gl_req_ready     (gl_req_ready),
        .gl_rsp_valid     (gl_rsp_valid),
        .gl_rsp_data      (gl_rsp_data),
        .gl_rsp_ready     (gl_rsp_ready),
        .wf_retired_valid (wf_retired_valid),
        .wf_retired_seq   (wf_retired_seq),
        .rd_valid         (rd_valid),
        .rd_row           (rd_row_w),
        .rd_tag           (rd_tag),
        .rd_ready         (rd_ready),
        .rd_rsp_valid     (rd_rsp_valid),
        .rd_rsp_data      (rd_rsp_data),
        .rd_rsp_tag       (rd_rsp_tag),
        .wr_valid         ({wr_valid, l1_fill_valid}),
        .wr_data          (wr_data_w),
        .wr_dsmem_last    (wr_dsmem_last),
        .wr_ready         (wr_ready_w),
        .wr_dsmem_busy    (wr_dsmem_busy),
        .fill_done_valid  (fill_done_valid),
        .fill_done_row    (fill_done_row),
        .cpa_if           (cpa_if),
        .amo_valid        (amo_valid),
        .amo_row          (amo_row),
        .amo_bank         (amo_bank),
        .amo_byteen       (amo_byteen),
        .amo_wdata        (amo_wdata),
        .amo_attr         (amo_attr_w),
        .amo_tag          (amo_tag),
        .amo_ready        (amo_ready),
        .amo_rsp_valid    (amo_rsp_valid),
        .amo_rsp_data     (amo_rsp_data),
        .amo_rsp_tag      (amo_rsp_tag),
        .acc_commit_valid (acc_commit_valid),
        .acc_commit_wid   (acc_wid),
        .acc_commit_cnt   (acc_commit_cnt)
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        ,
        .mbar_evt_valid   (mbar_evt_valid),
        .mbar_evt_flip    (mbar_evt_flip),
        .mbar_evt_wid     (mbar_evt_wid),
        .mbar_evt_row     (mbar_evt_row)
    `endif
    );

endmodule
