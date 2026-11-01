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

// The core's DLMEM aperture translation (peer table and per-LSU-block stages),
// the same wrapper VX_core instantiates, with flat ports. The request header
// fields the stage reads (wid, cta_id, pid) and the fence and release bits are
// driven separately and merged into the raw tag here, so the testbench needs
// no tag layout; in_tag_eff returns the merged tag. The cfg_* outputs are
// constants that give the testbench the package-derived widths and the
// attribute fields the stage may rewrite. drained and lmem_acc_drained stand in
// for the core's remote-access and local-access trackers.
module VX_dlmem_xlate_top import VX_gpu_pkg::*; #(
    parameter  CORE_ID       = 0,
    parameter  OOB_ASSERT    = 0,
    parameter  REMOTE_ASSERT = 0,

    localparam NB         = `VX_CFG_NUM_LSU_BLOCKS,
    localparam NL         = `VX_CFG_NUM_LSU_LANES,
    localparam NW         = `VX_CFG_NUM_WARPS,
    localparam MAXC       = DLMEM_MAX_CLUSTER,
    localparam LMEM_LOG   = `VX_CFG_LMEM_LOG_SIZE,
    localparam AW         = LSU_ADDR_WIDTH,
    localparam WS         = LSU_WORD_SIZE,
    localparam DW         = LSU_WORD_SIZE * 8,
    localparam ATTR_W     = MEM_ATTR_WIDTH,
    localparam TAG_W      = LSU_CLIENT_TAG_WIDTH,
    localparam HDR_W      = $bits(lsu_header_t),
    localparam PID_W      = `LOG2UP(`VX_CFG_NUM_THREADS / NL),
    localparam RSP_W      = $bits(lsu_rsp_data_t),
    localparam CORE_W     = DLMEM_CORE_W,
    localparam K_W        = DLMEM_K_W,
    localparam CNT_W      = `CLOG2(NL + 1)
) (
    input  wire                                 clk,
    input  wire                                 reset,

    // Peer-table write port.
    input  wire                                 wr_valid,
    input  wire [NW_WIDTH-1:0]                  wr_slot,
    input  wire [K_W-1:0]                       wr_k,
    input  wire [MAXC-1:0][CORE_W-1:0]          wr_core,
    input  wire [MAXC-1:0][NW_WIDTH-1:0]        wr_pslot,
    input  wire [LMEM_LOG:0]                    wr_stride,

    output wire [NW-1:0][LMEM_LOG-1:0]          tbl_slot_base,
    output wire [LMEM_LOG:0]                    tbl_stride,

`ifdef DXA_MBAR_ENABLE
    // The table's DXA read port.
    input  wire [NW_WIDTH-1:0]                  dxa_rd_slot,
    output wire [K_W-1:0]                       dxa_rd_k,
    output wire [MAXC-1:0][CORE_W-1:0]          dxa_rd_core,
    output wire [MAXC-1:0][NW_WIDTH-1:0]        dxa_rd_pslot,
    output wire [NW-1:0][LMEM_LOG-1:0]          dxa_slot_base,
    output wire [LMEM_LOG:0]                    dxa_stride,
`endif

    // Requests from the LSU, per block.
    input  wire [NB-1:0]                        in_valid,
    input  wire [NB-1:0]                        in_rw,
    input  wire [NB-1:0][NL-1:0]                in_mask,
    input  wire [NB-1:0][NL-1:0][WS-1:0]        in_byteen,
    input  wire [NB-1:0][NL-1:0][AW-1:0]        in_addr,
    input  wire [NB-1:0][NL-1:0][ATTR_W-1:0]    in_attr,
    input  wire [NB-1:0][NL-1:0][DW-1:0]        in_data,
    input  wire [NB-1:0][TAG_W-1:0]             in_tag,
    input  wire [NB-1:0][NW_WIDTH-1:0]          in_wid,
    input  wire [NB-1:0][NW_WIDTH-1:0]          in_cta,
    input  wire [NB-1:0][PID_W-1:0]             in_pid,
    input  wire [NB-1:0]                        in_fence,
    input  wire [NB-1:0]                        in_rl,
`ifdef DXA_MBAR_ENABLE
    input  wire [NB-1:0]                        in_pfence,   // a fence's op field names the proxy fence
`endif
    output wire [NB-1:0][TAG_W-1:0]             in_tag_eff,
    output wire [NB-1:0]                        in_ready,
    output wire [NB-1:0][NW_WIDTH-1:0]          row_slot,

    // Requests to the LSU scheduler, per block.
    output wire [NB-1:0]                        out_valid,
    output wire [NB-1:0]                        out_rw,
    output wire [NB-1:0][NL-1:0]                out_mask,
    output wire [NB-1:0][NL-1:0][WS-1:0]        out_byteen,
    output wire [NB-1:0][NL-1:0][AW-1:0]        out_addr,
    output wire [NB-1:0][NL-1:0][ATTR_W-1:0]    out_attr,
    output wire [NB-1:0][NL-1:0][DW-1:0]        out_data,
    output wire [NB-1:0][TAG_W-1:0]             out_tag,
    output wire [NB-1:0][NL-1:0]                out_remote,
    output wire [NB-1:0][NL-1:0]                out_oob,
    input  wire [NB-1:0]                        out_ready,

    output wire [NB-1:0]                        wt_inc_valid,
    output wire [NB-1:0][NW_WIDTH-1:0]          wt_inc_wid,
    output wire [NB-1:0][CNT_W-1:0]             wt_inc_cnt,
    output wire [NB-1:0]                        lacc_inc_valid,
    output wire [NB-1:0][NW_WIDTH-1:0]          lacc_inc_wid,
    output wire [NB-1:0][CNT_W-1:0]             lacc_inc_cnt,
    output wire [NB-1:0]                        empty,

    // Per-warp tracker states; a fence or a release AMO waits at the output
    // until both are set.
    input  wire [NW-1:0]                        drained,
    input  wire [NW-1:0]                        lmem_acc_drained,

    // Responses, scheduler side in, LSU side out.
    input  wire [NB-1:0]                        mem_rsp_valid,
    input  wire [NB-1:0][RSP_W-1:0]             mem_rsp_data,
    output wire [NB-1:0]                        mem_rsp_ready,
    output wire [NB-1:0]                        core_rsp_valid,
    output wire [NB-1:0][RSP_W-1:0]             core_rsp_data,
    input  wire [NB-1:0]                        core_rsp_ready,

    output wire [31:0]                          cfg_ncc,
    output wire [31:0]                          cfg_nw,
    output wire [31:0]                          cfg_max_cluster,
    output wire [31:0]                          cfg_lmem_log,
    output wire [31:0]                          cfg_blocks,
    output wire [31:0]                          cfg_lanes,
    output wire [31:0]                          cfg_addr_w,
    output wire [31:0]                          cfg_attr_w,
    output wire [31:0]                          cfg_tag_w,
    output wire [31:0]                          cfg_rsp_w,
    output wire [31:0]                          cfg_pid_w,
    output wire [31:0]                          cfg_nw_w,
    output wire [31:0]                          cfg_k_w,
    output wire [31:0]                          cfg_core_w,
    output wire [31:0]                          cfg_hart_w,
    output wire [31:0]                          cfg_hart_core_lsb,
    output wire [31:0]                          cfg_nt_bits,
    output wire [31:0]                          cfg_core_id,
    output wire [ATTR_W-1:0]                    cfg_attr_local,
    output wire [ATTR_W-1:0]                    cfg_attr_io,
    output wire [ATTR_W-1:0]                    cfg_attr_hart,
    output wire [ATTR_W-1:0]                    cfg_attr_remote,
    output wire [ATTR_W-1:0]                    cfg_attr_lacc
);
    assign cfg_ncc           = 32'(DLMEM_NCC);
    assign cfg_nw            = 32'(NW);
    assign cfg_max_cluster   = 32'(MAXC);
    assign cfg_lmem_log      = 32'(LMEM_LOG);
    assign cfg_blocks        = 32'(NB);
    assign cfg_lanes         = 32'(NL);
    assign cfg_addr_w        = 32'(AW);
    assign cfg_attr_w        = 32'(ATTR_W);
    assign cfg_tag_w         = 32'(TAG_W);
    assign cfg_rsp_w         = 32'(RSP_W);
    assign cfg_pid_w         = 32'(PID_W);
    assign cfg_nw_w          = 32'(NW_WIDTH);
    assign cfg_k_w           = 32'(K_W);
    assign cfg_core_w        = 32'(CORE_W);
    assign cfg_hart_w        = 32'(HART_ID_WIDTH);
    assign cfg_hart_core_lsb = 32'(NW_BITS + NT_BITS);
    assign cfg_nt_bits       = 32'(NT_BITS);
    assign cfg_core_id       = 32'(CORE_ID);

    function automatic mem_bus_attr_t attr_mask(input logic local_f, input logic io_f, input logic hart_f,
                                                input logic remote_f, input logic lacc_f);
        attr_mask = '0;
        attr_mask.is_addr_local  = local_f;
        attr_mask.is_addr_io     = io_f;
        attr_mask.amo.hart_id    = {HART_ID_WIDTH{hart_f}};
        attr_mask.is_addr_remote = remote_f;
        attr_mask.is_lacc        = lacc_f;
    endfunction

    assign cfg_attr_local  = attr_mask(1'b1, 1'b0, 1'b0, 1'b0, 1'b0);
    assign cfg_attr_io     = attr_mask(1'b0, 1'b1, 1'b0, 1'b0, 1'b0);
    assign cfg_attr_hart   = attr_mask(1'b0, 1'b0, 1'b1, 1'b0, 1'b0);
    assign cfg_attr_remote = attr_mask(1'b0, 1'b0, 1'b0, 1'b1, 1'b0);
    assign cfg_attr_lacc   = attr_mask(1'b0, 1'b0, 1'b0, 1'b0, 1'b1);

    dlmem_row_t wr_row;
    assign wr_row.k = wr_k;
    for (genvar r = 0; r < MAXC; ++r) begin : g_wr_row
        assign wr_row.peers[r].core = wr_core[r];
        assign wr_row.peers[r].slot = wr_pslot[r];
    end

`ifdef DXA_MBAR_ENABLE
    dlmem_row_t dxa_rd_row;
`endif

    VX_lsu_sched_if in_if  [NB]();
    VX_lsu_sched_if out_if [NB]();

    for (genvar b = 0; b < NB; ++b) begin : g_blocks
        lsu_header_t hdr;
        always @(*) begin
            hdr = in_tag[b][TAG_W-1 -: HDR_W];
            hdr.wid    = in_wid[b];
            hdr.cta_id = NCTA_WIDTH'(in_cta[b]);
            hdr.pid    = in_pid[b];
        end
    `ifdef DXA_MBAR_ENABLE
        // A random op field names the proxy fence, which the stage answers
        // itself, only when the bench asks for one.
        wire [INST_LSU_BITS-1:0] op_raw = in_tag[b][TAG_W-HDR_W-1 -: INST_LSU_BITS];
        wire [INST_LSU_BITS-1:0] op_eff = in_pfence[b] ? INST_LSU_PFENCE
                                        : ((op_raw == INST_LSU_PFENCE) ? INST_LSU_FENCE : op_raw);
        assign in_tag_eff[b] = {hdr, op_eff, in_tag[b][TAG_W-HDR_W-INST_LSU_BITS-1:2], in_rl[b], in_fence[b]};
    `else
        assign in_tag_eff[b] = {hdr, in_tag[b][TAG_W-HDR_W-1:2], in_rl[b], in_fence[b]};
    `endif
        `UNUSED_VAR (in_tag[b][1:0])

        assign in_if[b].req_valid       = in_valid[b];
        assign in_if[b].req_data.rw     = in_rw[b];
        assign in_if[b].req_data.mask   = in_mask[b];
        assign in_if[b].req_data.byteen = in_byteen[b];
        assign in_if[b].req_data.addr   = in_addr[b];
        assign in_if[b].req_data.attr   = in_attr[b];
        assign in_if[b].req_data.data   = in_data[b];
        assign in_if[b].req_data.tag    = in_tag_eff[b];
        assign in_ready[b]              = in_if[b].req_ready;

        assign core_rsp_valid[b] = in_if[b].rsp_valid;
        assign core_rsp_data[b]  = in_if[b].rsp_data;
        assign in_if[b].rsp_ready = core_rsp_ready[b];

        assign out_valid[b]  = out_if[b].req_valid;
        assign out_rw[b]     = out_if[b].req_data.rw;
        assign out_mask[b]   = out_if[b].req_data.mask;
        assign out_byteen[b] = out_if[b].req_data.byteen;
        assign out_addr[b]   = out_if[b].req_data.addr;
        assign out_attr[b]   = out_if[b].req_data.attr;
        assign out_data[b]   = out_if[b].req_data.data;
        assign out_tag[b]    = out_if[b].req_data.tag;
        assign out_if[b].req_ready = out_ready[b];

        assign out_if[b].rsp_valid = mem_rsp_valid[b];
        assign out_if[b].rsp_data  = mem_rsp_data[b];
        assign mem_rsp_ready[b]    = out_if[b].rsp_ready;
    end

    VX_dlmem_lsu_xlate #(
        .INSTANCE_ID   ("dlmem-xlate"),
        .CORE_ID       (CORE_ID),
        .OOB_ASSERT    (OOB_ASSERT),
        .REMOTE_ASSERT (REMOTE_ASSERT)
    ) lsu_xlate (
        .clk            (clk),
        .reset          (reset),
        .peer_wr_valid  (wr_valid),
        .peer_wr_slot   (wr_slot),
        .peer_wr_row    (wr_row),
        .peer_wr_stride (wr_stride),
        .drained        (drained),
        .lmem_acc_drained (lmem_acc_drained),
        .in_if          (in_if),
        .out_if         (out_if),
        .out_remote     (out_remote),
        .out_oob        (out_oob),
        .wt_inc_valid   (wt_inc_valid),
        .wt_inc_wid     (wt_inc_wid),
        .wt_inc_cnt     (wt_inc_cnt),
        .lacc_inc_valid (lacc_inc_valid),
        .lacc_inc_wid   (lacc_inc_wid),
        .lacc_inc_cnt   (lacc_inc_cnt),
    `ifdef DXA_MBAR_ENABLE
        .dxa_rd_slot    (dxa_rd_slot),
        .dxa_rd_row     (dxa_rd_row),
        .dxa_slot_base  (dxa_slot_base),
        .dxa_stride     (dxa_stride),
    `endif
        .empty          (empty)
    );

`ifdef DXA_MBAR_ENABLE
    assign dxa_rd_k = dxa_rd_row.k;
    for (genvar r = 0; r < MAXC; ++r) begin : g_dxa_rd_row
        assign dxa_rd_core[r]  = dxa_rd_row.peers[r].core;
        assign dxa_rd_pslot[r] = dxa_rd_row.peers[r].slot;
    end
`endif

    // Internal table state and row selects, observed by the testbench only.
`ifdef SYNTHESIS
    assign tbl_slot_base = '0;
    assign tbl_stride    = '0;
    assign row_slot      = '0;
`else
    assign tbl_slot_base = lsu_xlate.peer_tbl.slot_base;
    assign tbl_stride    = lsu_xlate.peer_tbl.stride;
    for (genvar b = 0; b < NB; ++b) begin : g_row_slot
        assign row_slot[b] = lsu_xlate.g_blocks[b].xlate.row_slot;
    end
`endif

endmodule
