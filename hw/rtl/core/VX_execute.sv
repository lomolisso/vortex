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

module VX_execute
import VX_gpu_pkg::*;
`ifdef EXT_F_ENABLE
`ifdef ASIC_SYNTHESIS
import VX_fpu_pkg::*;
`endif
`endif
#(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t     sysmem_perf,
    input pipeline_perf_t   pipeline_perf,
`endif

    input base_dcrs_t       base_dcrs,

    // Dcache interface
    VX_lsu_mem_if.master    lsu_mem_if [`NUM_LSU_BLOCKS],

    // dispatch interface
    VX_dispatch_if.slave    dispatch_if [NUM_EX_UNITS * `ISSUE_WIDTH],

    // commit interface
    VX_commit_if.master     commit_if [NUM_EX_UNITS * `ISSUE_WIDTH],

    // scheduler interfaces
    VX_sched_csr_if.slave   sched_csr_if,
    VX_branch_ctl_if.master branch_ctl_if [`NUM_ALU_BLOCKS],
    VX_warp_ctl_if.master   warp_ctl_if,

    // commit interface
    VX_commit_csr_if.slave  commit_csr_if
);

`ifdef EXT_F_ENABLE
    VX_fpu_csr_if fpu_csr_if[`NUM_FPU_BLOCKS]();
`endif

    VX_alu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-alu", INSTANCE_ID)))
    ) alu_unit (
        .clk            (clk),
        .reset          (reset),
        .dispatch_if    (dispatch_if[EX_ALU * `ISSUE_WIDTH +: `ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_ALU * `ISSUE_WIDTH +: `ISSUE_WIDTH]),
        .branch_ctl_if  (branch_ctl_if)
    );

    `SCOPE_IO_SWITCH (1);

    VX_lsu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-lsu", INSTANCE_ID)))
    ) lsu_unit (
        `SCOPE_IO_BIND  (0)
        .clk            (clk),
        .reset          (reset),
        .dispatch_if    (dispatch_if[EX_LSU * `ISSUE_WIDTH +: `ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_LSU * `ISSUE_WIDTH +: `ISSUE_WIDTH]),
        .lsu_mem_if     (lsu_mem_if)
    );

`ifdef EXT_F_ENABLE
  `ifdef ASIC_SYNTHESIS
    // Flatten dispatch_if/commit_if/fpu_csr_if into the flat-port VX_fpu_top macro.
    wire [`ISSUE_WIDTH-1:0]                       fpu_dispatch_valid;
    dispatch_t [`ISSUE_WIDTH-1:0]                 fpu_dispatch_data;
    wire [`ISSUE_WIDTH-1:0]                       fpu_dispatch_ready;
    wire [`ISSUE_WIDTH-1:0]                       fpu_commit_valid;
    commit_t   [`ISSUE_WIDTH-1:0]                 fpu_commit_data;
    wire [`ISSUE_WIDTH-1:0]                       fpu_commit_ready;
    wire [`NUM_FPU_BLOCKS-1:0]                    fpu_csr_write_enable;
    wire [`NUM_FPU_BLOCKS-1:0][NW_WIDTH-1:0]      fpu_csr_write_wid;
    fflags_t   [`NUM_FPU_BLOCKS-1:0]              fpu_csr_write_fflags;
    wire [`NUM_FPU_BLOCKS-1:0][NW_WIDTH-1:0]      fpu_csr_read_wid;
    wire [`NUM_FPU_BLOCKS-1:0][INST_FRM_BITS-1:0] fpu_csr_read_frm;

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin : g_fpu_disp_commit
        assign fpu_dispatch_valid[i]                                       = dispatch_if[EX_FPU * `ISSUE_WIDTH + i].valid;
        assign fpu_dispatch_data[i]                                        = dispatch_if[EX_FPU * `ISSUE_WIDTH + i].data;
        assign dispatch_if[EX_FPU * `ISSUE_WIDTH + i].ready                = fpu_dispatch_ready[i];

        assign commit_if[EX_FPU * `ISSUE_WIDTH + i].valid                  = fpu_commit_valid[i];
        assign commit_if[EX_FPU * `ISSUE_WIDTH + i].data                   = fpu_commit_data[i];
        assign fpu_commit_ready[i]                                         = commit_if[EX_FPU * `ISSUE_WIDTH + i].ready;
    end

    for (genvar b = 0; b < `NUM_FPU_BLOCKS; ++b) begin : g_fpu_csr_flat
        assign fpu_csr_if[b].write_enable = fpu_csr_write_enable[b];
        assign fpu_csr_if[b].write_wid    = fpu_csr_write_wid[b];
        assign fpu_csr_if[b].write_fflags = fpu_csr_write_fflags[b];
        assign fpu_csr_if[b].read_wid     = fpu_csr_read_wid[b];
        assign fpu_csr_read_frm[b]        = fpu_csr_if[b].read_frm;
    end

    VX_fpu_top fpu_top (
        .clk              (clk),
        .reset            (reset),
        .dispatch_valid   (fpu_dispatch_valid),
        .dispatch_data    (fpu_dispatch_data),
        .dispatch_ready   (fpu_dispatch_ready),
        .commit_valid     (fpu_commit_valid),
        .commit_data      (fpu_commit_data),
        .commit_ready     (fpu_commit_ready),
        .csr_write_enable (fpu_csr_write_enable),
        .csr_write_wid    (fpu_csr_write_wid),
        .csr_write_fflags (fpu_csr_write_fflags),
        .csr_read_wid     (fpu_csr_read_wid),
        .csr_read_frm     (fpu_csr_read_frm)
    );
  `else
    VX_fpu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-fpu", INSTANCE_ID)))
    ) fpu_unit (
        .clk            (clk),
        .reset          (reset),
        .dispatch_if    (dispatch_if[EX_FPU * `ISSUE_WIDTH +: `ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_FPU * `ISSUE_WIDTH +: `ISSUE_WIDTH]),
        .fpu_csr_if     (fpu_csr_if)
    );
  `endif
`endif

`ifdef EXT_TCU_ENABLE
  `ifdef ASIC_SYNTHESIS
    // Flatten dispatch_if/commit_if into the flat-port VX_tcu_top macro.
    wire [`ISSUE_WIDTH-1:0]                       tcu_dispatch_valid;
    dispatch_t [`ISSUE_WIDTH-1:0]                 tcu_dispatch_data;
    wire [`ISSUE_WIDTH-1:0]                       tcu_dispatch_ready;
    wire [`ISSUE_WIDTH-1:0]                       tcu_commit_valid;
    commit_t   [`ISSUE_WIDTH-1:0]                 tcu_commit_data;
    wire [`ISSUE_WIDTH-1:0]                       tcu_commit_ready;

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin : g_tcu_disp_commit
        assign tcu_dispatch_valid[i]                                       = dispatch_if[EX_TCU * `ISSUE_WIDTH + i].valid;
        assign tcu_dispatch_data[i]                                        = dispatch_if[EX_TCU * `ISSUE_WIDTH + i].data;
        assign dispatch_if[EX_TCU * `ISSUE_WIDTH + i].ready                = tcu_dispatch_ready[i];

        assign commit_if[EX_TCU * `ISSUE_WIDTH + i].valid                  = tcu_commit_valid[i];
        assign commit_if[EX_TCU * `ISSUE_WIDTH + i].data                   = tcu_commit_data[i];
        assign tcu_commit_ready[i]                                         = commit_if[EX_TCU * `ISSUE_WIDTH + i].ready;
    end

    VX_tcu_top tcu_top (
        .clk            (clk),
        .reset          (reset),
        .dispatch_valid (tcu_dispatch_valid),
        .dispatch_data  (tcu_dispatch_data),
        .dispatch_ready (tcu_dispatch_ready),
        .commit_valid   (tcu_commit_valid),
        .commit_data    (tcu_commit_data),
        .commit_ready   (tcu_commit_ready)
    );
  `else
    VX_tcu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-tcu", INSTANCE_ID)))
    ) tcu_unit (
        .clk            (clk),
        .reset          (reset),
        .dispatch_if    (dispatch_if[EX_TCU * `ISSUE_WIDTH +: `ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_TCU * `ISSUE_WIDTH +: `ISSUE_WIDTH])
    );
  `endif
`endif

    VX_sfu_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-sfu", INSTANCE_ID))),
        .CORE_ID (CORE_ID)
    ) sfu_unit (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .sysmem_perf    (sysmem_perf),
        .pipeline_perf  (pipeline_perf),
    `endif
        .base_dcrs      (base_dcrs),
        .dispatch_if    (dispatch_if[EX_SFU * `ISSUE_WIDTH +: `ISSUE_WIDTH]),
        .commit_if      (commit_if[EX_SFU * `ISSUE_WIDTH +: `ISSUE_WIDTH]),
    `ifdef EXT_F_ENABLE
        .fpu_csr_if     (fpu_csr_if),
    `endif
        .commit_csr_if  (commit_csr_if),
        .sched_csr_if   (sched_csr_if),
        .warp_ctl_if    (warp_ctl_if)
    );

endmodule
