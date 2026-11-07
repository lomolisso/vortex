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

// The bulk-group unit with its structs and unlock flattened; warp ids travel at 8 bits.
module VX_bg_unit_top import VX_gpu_pkg::*, VX_dxa_pkg::*; (
    input  wire                         clk,
    input  wire                         reset,

    input  wire                         evt_valid,
    input  wire [7:0]                   evt_wid,
    input  wire [1:0]                   evt_kind,
    input  wire [4:0]                   evt_n,

    input  wire                         compl_valid,
    input  wire [7:0]                   compl_wid,
    input  wire [2:0]                   compl_seq,
    input  wire                         compl_read_done,
    input  wire                         compl_write_done,

    output wire [`VX_CFG_NUM_WARPS-1:0] stall_op,
    output wire [`VX_CFG_NUM_WARPS-1:0] stall_commit,
    output wire [`VX_CFG_NUM_WARPS-1:0][2:0] seq,
    output wire [`VX_CFG_NUM_WARPS-1:0] rd_pending,
    output wire                         wr_pending,

    output wire                         unlock_valid,
    output wire [7:0]                   unlock_wid
);
    dxa_bg_evt_t   evt;
    dxa_bg_compl_t compl;
    assign evt.wid   = NW_WIDTH'(evt_wid);
    assign evt.kind  = evt_kind;
    assign evt.n     = evt_n;
    assign compl.wid = NW_WIDTH'(compl_wid);
    assign compl.seq = DXA_BG_SEQ_W'(compl_seq);
    assign compl.read_done  = compl_read_done;
    assign compl.write_done = compl_write_done;

    wire [`VX_CFG_NUM_WARPS-1:0][DXA_BG_SEQ_W-1:0] seq_w;

    VX_sched_unlock_if unlock_if();

    VX_bg_unit #(
        .INSTANCE_ID ("bg")
    ) bg_unit (
        .clk          (clk),
        .reset        (reset),
        .evt_valid    (evt_valid),
        .evt          (evt),
        .compl_valid  (compl_valid),
        .compl        (compl),
        .stall_op     (stall_op),
        .stall_commit (stall_commit),
        .seq          (seq_w),
        .rd_pending   (rd_pending),
        .wr_pending   (wr_pending),
        .unlock_if    (unlock_if)
    );

    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_seq
        assign seq[w] = 3'(seq_w[w]);
    end

    assign unlock_valid = unlock_if.valid;
    assign unlock_wid   = 8'(unlock_if.wid);

    `UNUSED_VAR (evt_wid)
    `UNUSED_VAR (compl_wid)

endmodule
