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
`include "VX_fpu_define.vh"

// Flat-port wrapper around VX_fpu_unit for the ASIC macro boundary.

`ifdef ASIC_SYNTHESIS
module VX_fpu_top import VX_gpu_pkg::*, VX_fpu_pkg::*; (
    input  wire                                         clk,
    input  wire                                         reset,

    // Dispatch (per issue slot)
    input  wire [`ISSUE_WIDTH-1:0]                      dispatch_valid,
    input  dispatch_t [`ISSUE_WIDTH-1:0]                dispatch_data,
    output wire [`ISSUE_WIDTH-1:0]                      dispatch_ready,

    // Commit (per issue slot)
    output wire [`ISSUE_WIDTH-1:0]                      commit_valid,
    output commit_t [`ISSUE_WIDTH-1:0]                  commit_data,
    input  wire [`ISSUE_WIDTH-1:0]                      commit_ready,

    // FPU CSR (per block, NUM_FPU_BLOCKS == ISSUE_WIDTH by default)
    output wire [`NUM_FPU_BLOCKS-1:0]                   csr_write_enable,
    output wire [`NUM_FPU_BLOCKS-1:0][NW_WIDTH-1:0]     csr_write_wid,
    output fflags_t [`NUM_FPU_BLOCKS-1:0]               csr_write_fflags,
    output wire [`NUM_FPU_BLOCKS-1:0][NW_WIDTH-1:0]     csr_read_wid,
    input  wire [`NUM_FPU_BLOCKS-1:0][INST_FRM_BITS-1:0] csr_read_frm
);
    VX_dispatch_if dispatch_if [`ISSUE_WIDTH]();
    VX_commit_if   commit_if   [`ISSUE_WIDTH]();
    VX_fpu_csr_if  fpu_csr_if  [`NUM_FPU_BLOCKS]();

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin : g_disp_commit
        assign dispatch_if[i].valid = dispatch_valid[i];
        assign dispatch_if[i].data  = dispatch_data[i];
        assign dispatch_ready[i]    = dispatch_if[i].ready;

        assign commit_valid[i]      = commit_if[i].valid;
        assign commit_data[i]       = commit_if[i].data;
        assign commit_if[i].ready   = commit_ready[i];
    end

    for (genvar b = 0; b < `NUM_FPU_BLOCKS; ++b) begin : g_fpu_csr
        assign csr_write_enable[b]     = fpu_csr_if[b].write_enable;
        assign csr_write_wid[b]        = fpu_csr_if[b].write_wid;
        assign csr_write_fflags[b]     = fpu_csr_if[b].write_fflags;
        assign csr_read_wid[b]         = fpu_csr_if[b].read_wid;
        assign fpu_csr_if[b].read_frm  = csr_read_frm[b];
    end

    VX_fpu_unit #(
        .INSTANCE_ID (`SFORMATF(("fpu_top")))
    ) fpu_unit (
        .clk         (clk),
        .reset       (reset),
        .dispatch_if (dispatch_if),
        .commit_if   (commit_if),
        .fpu_csr_if  (fpu_csr_if)
    );

endmodule
`endif
