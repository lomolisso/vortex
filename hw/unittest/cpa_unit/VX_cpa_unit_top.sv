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

// The cp.async unit with its local-memory port and unlock flattened; warp
// ids, tags and amo ops travel at 8 bits and addresses at 32.
module VX_cpa_unit_top import VX_gpu_pkg::*; #(
    localparam NL       = `VX_CFG_NUM_LSU_LANES,
    localparam PID_BITS = `CLOG2(`VX_CFG_NUM_THREADS / NL),
    localparam PID_W    = `UP(PID_BITS),
    localparam SLOT_W   = `LOG2UP(`VX_CFG_LSU_PENDING_SIZE),
    localparam LMEM_LOG = `VX_CFG_LMEM_LOG_SIZE
) (
    input  wire                         clk,
    input  wire                         reset,

    input  wire                         iss_valid,
    input  wire [1:0]                   iss_op,
    input  wire                         iss_noinc,
    input  wire [7:0]                   iss_wid,
    input  wire [PID_W-1:0]             iss_pid,
    input  wire                         iss_sop,
    input  wire                         iss_eop,
    input  wire [NL-1:0]                iss_tmask,
    input  wire                         iss_size8,
    input  wire [4:0]                   iss_n,
    input  wire [NL-1:0][LMEM_LOG-1:0]  iss_addr,
    input  wire [NL-1:0][3:0]           iss_ss,
    output wire                         iss_ready,
    output wire [SLOT_W-1:0]            iss_slot,

    input  wire                         rsp_valid,
    input  wire [SLOT_W-1:0]            rsp_slot,
    input  wire [PID_W-1:0]             rsp_pid,
    input  wire [NL-1:0]                rsp_mask,
    input  wire [NL-1:0][63:0]          rsp_data,
    input  wire [NL-1:0][2:0]           rsp_align,

    output wire                         req_valid,
    output wire                         req_rw,
    output wire [NL-1:0]                req_mask,
    output wire [NL-1:0][31:0]          req_addr,
    output wire [NL-1:0][63:0]          req_data,
    output wire [NL-1:0][7:0]           req_byteen,
    output wire [NL-1:0]                req_amo_valid,
    output wire [NL-1:0][7:0]           req_amo_op,
    output wire [7:0]                   req_tag,
    input  wire                         req_ready,

    input  wire                         ack_valid,
    input  wire [NL-1:0]                ack_mask,
    input  wire [7:0]                   ack_tag,

    output wire                         unlock_valid,
    output wire [7:0]                   unlock_wid,

    output wire [`VX_CFG_NUM_WARPS-1:0] idle
);
    localparam TAG_ID_W = LSU_TAG_WIDTH - UUID_WIDTH;

    VX_lsu_mem_if #(
        .NUM_LANES (NL),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_if();

    VX_sched_unlock_if unlock_if();

    VX_cpa_unit #(
        .INSTANCE_ID ("cpa")
    ) cpa_unit (
        .clk       (clk),
        .reset     (reset),
        .iss_valid (iss_valid),
        .iss_op    (CPA_OP_BITS'(iss_op)),
        .iss_noinc (iss_noinc),
        .iss_wid   (NW_WIDTH'(iss_wid)),
        .iss_pid   (iss_pid),
        .iss_sop   (iss_sop),
        .iss_eop   (iss_eop),
        .iss_tmask (iss_tmask),
        .iss_size8 (iss_size8),
        .iss_n     (iss_n),
        .iss_addr  (iss_addr),
        .iss_ss    (iss_ss),
        .iss_ready (iss_ready),
        .iss_slot  (iss_slot),
        .rsp_valid (rsp_valid),
        .rsp_slot  (rsp_slot),
        .rsp_pid   (rsp_pid),
        .rsp_mask  (rsp_mask),
        .rsp_data  (rsp_data),
        .rsp_align (rsp_align),
        .lmem_if   (lmem_if),
        .unlock_if (unlock_if),
        .idle      (idle)
    );

    for (genvar j = 0; j < NL; ++j) begin : g_lanes
        mem_bus_attr_t attr;
        assign attr = mem_bus_attr_t'(lmem_if.req_data.user[j]);
        assign req_addr[j]      = 32'(lmem_if.req_data.addr[j]);
        assign req_data[j]      = lmem_if.req_data.data[j];
        assign req_byteen[j]    = lmem_if.req_data.byteen[j];
        assign req_amo_valid[j] = attr.amo.amo_valid;
        assign req_amo_op[j]    = 8'(attr.amo.amo_op);
        `UNUSED_VAR (attr)
    end
    assign req_valid = lmem_if.req_valid;
    assign req_rw    = lmem_if.req_data.rw;
    assign req_mask  = lmem_if.req_data.mask;
    assign req_tag   = 8'(lmem_if.req_data.tag.value);
    `UNUSED_VAR (lmem_if.req_data.tag.uuid)
    assign lmem_if.req_ready = req_ready;

    assign lmem_if.rsp_valid          = ack_valid;
    assign lmem_if.rsp_data.mask      = ack_mask;
    assign lmem_if.rsp_data.data      = '0;
    assign lmem_if.rsp_data.tag.value = TAG_ID_W'(ack_tag);
    assign lmem_if.rsp_data.tag.uuid  = '0;
    `UNUSED_VAR (lmem_if.rsp_ready)
    `UNUSED_VAR (iss_wid)
    `UNUSED_VAR (ack_tag)

    assign unlock_valid = unlock_if.valid;
    assign unlock_wid   = 8'(unlock_if.wid);

endmodule
