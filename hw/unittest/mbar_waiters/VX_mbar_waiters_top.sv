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

// One core's waiter table, as VX_mem_unit wires it: one event port per local
// memory bank and one retire port per LSU block. main.cpp packs the warp ids
// in 8-bit lanes.
module VX_mbar_waiters_top import VX_gpu_pkg::*; #(
    parameter NUM_BANKS   = `VX_CFG_LMEM_NUM_BANKS,
    parameter NUM_RETIRES = `VX_CFG_NUM_LSU_BLOCKS,
    parameter ROW_WIDTH   = 8
) (
    input  wire                              clk,
    input  wire                              reset,

    input  wire [NUM_BANKS-1:0]              evt_valid,
    input  wire [NUM_BANKS-1:0]              evt_flip,
    input  wire [NUM_BANKS-1:0][7:0]         evt_wid,
    input  wire [NUM_BANKS-1:0][ROW_WIDTH-1:0] evt_row,

    input  wire [NUM_RETIRES-1:0]            retire_valid,
    input  wire [NUM_RETIRES-1:0][7:0]       retire_wid,

    output wire                              unlock_valid,
    output wire [`VX_CFG_NUM_WARPS-1:0]      unlock_mask
);
    `STATIC_ASSERT (NW_WIDTH <= 8, ("mbar_waiters_top: warp ids are packed in 8 bits"))

    wire [NUM_BANKS-1:0][NW_WIDTH-1:0]   evt_wid_w;
    wire [NUM_RETIRES-1:0][NW_WIDTH-1:0] retire_wid_w;
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_evt
        assign evt_wid_w[i] = NW_WIDTH'(evt_wid[i]);
    end
    for (genvar i = 0; i < NUM_RETIRES; ++i) begin : g_ret
        assign retire_wid_w[i] = NW_WIDTH'(retire_wid[i]);
    end
    `UNUSED_VAR (evt_wid)
    `UNUSED_VAR (retire_wid)

    VX_mbar_waiters #(
        .INSTANCE_ID ("mbar_waiters"),
        .NUM_BANKS   (NUM_BANKS),
        .NUM_RETIRES (NUM_RETIRES),
        .ROW_WIDTH   (ROW_WIDTH)
    ) waiters (
        .clk          (clk),
        .reset        (reset),
        .evt_valid    (evt_valid),
        .evt_flip     (evt_flip),
        .evt_wid      (evt_wid_w),
        .evt_row      (evt_row),
        .retire_valid (retire_valid),
        .retire_wid   (retire_wid_w),
        .unlock_valid (unlock_valid),
        .unlock_mask  (unlock_mask)
    );

endmodule
