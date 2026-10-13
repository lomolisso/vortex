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

// A core's DLMEM aperture translation: the peer table, written as blocks are
// admitted, and one translation stage per LSU block between the LSU and its
// scheduler. Each block's empty covers its stage only; the core folds it into
// that block's scheduler empty.
module VX_dlmem_lsu_xlate import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter  CORE_ID       = 0,
    parameter  OOB_ASSERT    = 1,
    parameter  REMOTE_ASSERT = 0,
    parameter  REQ_BUF_SIZE  = 2,

    localparam NB            = `VX_CFG_NUM_LSU_BLOCKS,
    localparam NUM_LANES     = `VX_CFG_NUM_LSU_LANES,
    localparam NW            = `VX_CFG_NUM_WARPS,
    localparam LMEM_LOG      = `VX_CFG_LMEM_LOG_SIZE,
    localparam CNT_W         = `CLOG2(NUM_LANES + 1)
) (
    input  wire                             clk,
    input  wire                             reset,

    input  wire                             peer_wr_valid,
    input  wire [NW_WIDTH-1:0]              peer_wr_slot,
    input  dlmem_row_t                      peer_wr_row,
    input  wire [LMEM_LOG:0]                peer_wr_stride,

    // Per warp, no remote access awaits its completion; fences wait for it.
    input  wire [NW-1:0]                    drained,

    VX_lsu_sched_if.slave                   in_if  [NB],
    VX_lsu_sched_if.master                  out_if [NB],

    // Per-lane flags, qualified by out_if[b].req_valid.
    output wire [NB-1:0][NUM_LANES-1:0]     out_remote,
    output wire [NB-1:0][NUM_LANES-1:0]     out_oob,

    // Remote accesses leaving each stage, per warp.
    output wire [NB-1:0]                    wt_inc_valid,
    output wire [NB-1:0][NW_WIDTH-1:0]      wt_inc_wid,
    output wire [NB-1:0][CNT_W-1:0]         wt_inc_cnt,

    output wire [NB-1:0]                    empty
);
    wire [NB-1:0][NW_WIDTH-1:0] row_slot;
    dlmem_row_t [NB-1:0]        row;
    wire [`VX_CFG_NUM_WARPS-1:0][LMEM_LOG-1:0] slot_base;
    wire [LMEM_LOG:0]           stride;

    VX_dlmem_peer_tbl #(
        .NUM_RD (NB)
    ) peer_tbl (
        .clk       (clk),
        .reset     (reset),
        .wr_valid  (peer_wr_valid),
        .wr_slot   (peer_wr_slot),
        .wr_row    (peer_wr_row),
        .wr_stride (peer_wr_stride),
        .rd_slot   (row_slot),
        .rd_row    (row),
        .slot_base (slot_base),
        .stride    (stride)
    );

    for (genvar b = 0; b < NB; ++b) begin : g_blocks
        VX_dlmem_xlate #(
            .INSTANCE_ID   (`SFORMATF(("%s%0d", INSTANCE_ID, b))),
            .CORE_ID       (CORE_ID),
            .OOB_ASSERT    (OOB_ASSERT),
            .REMOTE_ASSERT (REMOTE_ASSERT),
            .REQ_BUF_SIZE  (REQ_BUF_SIZE)
        ) xlate (
            .clk          (clk),
            .reset        (reset),
            .in_if        (in_if[b]),
            .out_if       (out_if[b]),
            .row_slot     (row_slot[b]),
            .row          (row[b]),
            .slot_base    (slot_base),
            .stride       (stride),
            .drained      (drained),
            .out_remote   (out_remote[b]),
            .out_oob      (out_oob[b]),
            .wt_inc_valid (wt_inc_valid[b]),
            .wt_inc_wid   (wt_inc_wid[b]),
            .wt_inc_cnt   (wt_inc_cnt[b]),
            .empty        (empty[b])
        );
    end

endmodule
