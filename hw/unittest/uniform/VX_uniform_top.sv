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

// The issue stage with flat ports: the bench plays decode and the execute units. PC carries the
// bench's instruction tag (UUID_WIDTH is 1 in non-debug builds).
module VX_uniform_top import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "uniform"
) (
    input wire                              clk,
    input wire                              reset,

    input wire                              decode_valid,
    input wire [NW_WIDTH-1:0]               decode_wid,
    input wire [`VX_CFG_NUM_THREADS-1:0]    decode_tmask,
    input wire [PC_BITS-1:0]                decode_PC,
    input wire                              decode_wb,
    input wire                              decode_wr_ur,
    input wire [UR_SLOTS-1:0]               decode_ur_rs,
    input wire [UR_QUAD_SLOTS-1:0]          decode_ur_quad,
    input wire [NUM_SRC_OPDS-1:0]           decode_used_rs,
    input wire [NUM_REGS_BITS-1:0]          decode_rd,
    input wire [NUM_REGS_BITS-1:0]          decode_rs1,
    input wire [NUM_REGS_BITS-1:0]          decode_rs2,
    input wire [NUM_REGS_BITS-1:0]          decode_rs3,
    output wire                             decode_ready,

    input wire                              writeback_valid,
    input wire [NW_WIDTH-1:0]               writeback_wid,
    input wire [`VX_CFG_SIMD_WIDTH-1:0]     writeback_tmask,
    input wire [PC_BITS-1:0]                writeback_PC,
    input wire [NUM_REGS_BITS-1:0]          writeback_rd,
    input wire                              writeback_wb,
    input wire                              writeback_wr_ur,
    input wire [`VX_CFG_SIMD_WIDTH-1:0][`VX_CFG_XLEN-1:0] writeback_data,

    output wire                             dispatch_valid,
    output wire [PC_BITS-1:0]               dispatch_PC,
    output wire [`VX_CFG_SIMD_WIDTH-1:0]    dispatch_tmask,
    output wire [`VX_CFG_SIMD_WIDTH-1:0][`VX_CFG_XLEN-1:0] dispatch_rs1_data,
    output wire [`VX_CFG_SIMD_WIDTH-1:0][`VX_CFG_XLEN-1:0] dispatch_rs2_data,
    output wire [`VX_CFG_SIMD_WIDTH-1:0][`VX_CFG_XLEN-1:0] dispatch_rs3_data,
    output wire                             dispatch_sop,
    input wire                              dispatch_ready
);
    localparam NUM_DISPATCH = NUM_EX_UNITS * `VX_CFG_ISSUE_WIDTH;

    VX_decode_if    decode_if();
    VX_dispatch_if  dispatch_if[NUM_DISPATCH]();
    VX_writeback_if writeback_if[`VX_CFG_ISSUE_WIDTH]();
    VX_issue_sched_if issue_sched_if[`VX_CFG_ISSUE_WIDTH]();

    always @(*) begin
        decode_if.data = '0;
        decode_if.data.wid      = decode_wid;
        decode_if.data.tmask    = decode_tmask;
        decode_if.data.PC       = decode_PC;
        decode_if.data.ex_type  = EX_ALU;
        decode_if.data.wb       = decode_wb;
        decode_if.data.wr_xregs = NUM_XREGS'(decode_wr_ur) << XREG_UR;
        decode_if.data.ur_rs    = decode_ur_rs;
        decode_if.data.ur_quad  = decode_ur_quad;
        decode_if.data.used_rs  = decode_used_rs;
        decode_if.data.rd       = decode_rd;
        decode_if.data.rs1      = decode_rs1;
        decode_if.data.rs2      = decode_rs2;
        decode_if.data.rs3      = decode_rs3;
    end
    assign decode_if.valid = decode_valid;
    assign decode_ready = decode_if.ready;
    `UNUSED_VAR (decode_if.ibuf_pop)

    wire [ISSUE_ISW_W-1:0] wb_isw = wid_to_isw(writeback_wid);
    for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_writeback_if
        assign writeback_if[i].valid = writeback_valid && (wb_isw == ISSUE_ISW_W'(i));
        always @(*) begin
            writeback_if[i].data = '0;
            writeback_if[i].data.wis      = wid_to_wis(writeback_wid);
            writeback_if[i].data.eop_wis  = PER_ISSUE_WARPS'(1) << wid_to_wis(writeback_wid);
            writeback_if[i].data.tmask    = writeback_tmask;
            writeback_if[i].data.PC       = writeback_PC;
            writeback_if[i].data.rd       = writeback_rd;
            writeback_if[i].data.wb       = writeback_wb;
            writeback_if[i].data.wr_xregs = NUM_XREGS'(writeback_wr_ur) << XREG_UR;
            writeback_if[i].data.byteen   = '1;
            writeback_if[i].data.data     = writeback_data;
            writeback_if[i].data.sop      = 1'b1;
            writeback_if[i].data.eop      = 1'b1;
        end
    end

    // One dispatch port at a time, lowest index first.
    logic [NUM_DISPATCH-1:0] dsp_valids;
    logic [`LOG2UP(NUM_DISPATCH)-1:0] dsp_sel;
    always @(*) begin
        dsp_sel = '0;
        for (integer i = NUM_DISPATCH-1; i >= 0; --i) begin
            if (dsp_valids[i])
                dsp_sel = `LOG2UP(NUM_DISPATCH)'(i);
        end
    end
    dispatch_t dsp_data [NUM_DISPATCH];
    for (genvar i = 0; i < NUM_DISPATCH; ++i) begin : g_dispatch_if
        assign dsp_valids[i] = dispatch_if[i].valid;
        assign dsp_data[i] = dispatch_if[i].data;
        `UNUSED_VAR (dsp_data[i])
        assign dispatch_if[i].ready = dispatch_ready && (dsp_sel == `LOG2UP(NUM_DISPATCH)'(i));
    end
    assign dispatch_valid    = (| dsp_valids);
    assign dispatch_PC       = dsp_data[dsp_sel].PC;
    assign dispatch_tmask    = dsp_data[dsp_sel].tmask;
    assign dispatch_rs1_data = dsp_data[dsp_sel].rs1_data;
    assign dispatch_rs2_data = dsp_data[dsp_sel].rs2_data;
    assign dispatch_rs3_data = dsp_data[dsp_sel].rs3_data;
    assign dispatch_sop      = dsp_data[dsp_sel].sop;

    for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_issue_sched
        `UNUSED_VAR (issue_sched_if[i].valid)
        `UNUSED_VAR (issue_sched_if[i].wis)
    end

`ifdef PERF_ENABLE
    issue_perf_t issue_perf;
    `UNUSED_VAR (issue_perf)
`endif

`ifdef SCOPE
    wire [0:0] scope_reset_w = 1'b0;
    wire [0:0] scope_bus_in_w = 1'b0;
    wire [0:0] scope_bus_out_w;
    `UNUSED_VAR (scope_bus_out_w)
`endif

    VX_issue #(
        .INSTANCE_ID (INSTANCE_ID)
    ) issue (
        `SCOPE_IO_BIND (0)
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .issue_perf     (issue_perf),
    `endif
        .decode_if      (decode_if),
        .writeback_if   (writeback_if),
        .dispatch_if    (dispatch_if),
        .issue_sched_if (issue_sched_if)
    );

endmodule
