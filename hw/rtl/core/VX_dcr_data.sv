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

module VX_dcr_data import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    // Inputs
    VX_dcr_bus_if.slave     dcr_bus_if,

    // DCR-triggered CSR read bypass (for MPM perf counter readback)
    VX_dcr_csr_if.master    dcr_csr_if,

    // DCR-triggered cache flush
    VX_dcr_flush_if.master  dcr_flush_if
`ifdef VX_CFG_H100_SM_ENABLE
    ,
    // The SM array's split for the current launch
    output smem_cfg_t       smem_cfg
`endif
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // MPM perf-counter readback via DCR interface ////////////////////////////

    wire [7:0]  mpm_class      = dcr_bus_if.req_data.data[22 +: 8];
    wire [5:0]  mpm_tag_idx    = dcr_bus_if.req_data.data[16 +: 6];
    wire [15:0] mpm_target_cid = dcr_bus_if.req_data.data[0  +:16];

    wire [`VX_CSR_ADDR_BITS-1:0] mpm_csr_addr = mpm_tag_idx[5]
        ? (`VX_CSR_ADDR_BITS'(`VX_CSR_MPM_BASE_H) + `VX_CSR_ADDR_BITS'(mpm_tag_idx[4:0]))
        : (`VX_CSR_ADDR_BITS'(`VX_CSR_MPM_BASE)   + `VX_CSR_ADDR_BITS'(mpm_tag_idx[4:0]));

    // Latch a read request when it is for this core and addr == MPM_VALUE
    reg                         dcr_csr_pending_r;
    reg [`VX_CSR_ADDR_BITS-1:0] dcr_csr_addr_r;
    reg [7:0]                   dcr_csr_class_r;

    wire is_mpm_read = dcr_bus_if.req_valid
                    && ~dcr_bus_if.req_data.rw
                    && (dcr_bus_if.req_data.addr == `VX_DCR_BASE_MPM_VALUE)
                    && (mpm_target_cid == 16'(CORE_ID))
                    && ~dcr_csr_pending_r;

    always @(posedge clk) begin
        if (reset) begin
            dcr_csr_pending_r <= 1'b0;
            dcr_csr_addr_r    <= '0;
            dcr_csr_class_r   <= '0;
        end else if (is_mpm_read) begin
            dcr_csr_pending_r <= 1'b1;
            dcr_csr_addr_r    <= mpm_csr_addr;
            dcr_csr_class_r   <= mpm_class;
        end else if (dcr_csr_if.ready) begin
            dcr_csr_pending_r <= 1'b0;
        end
    end

    assign dcr_csr_if.valid = dcr_csr_pending_r;
    assign dcr_csr_if.addr  = dcr_csr_addr_r;
    assign dcr_csr_if.mpm_class = dcr_csr_class_r;

    wire dcr_csr_if_fire = dcr_csr_if.valid && dcr_csr_if.ready;

    // Cache flush via DCR interface /////////////////////////////////////////

    reg flush_pending_r;

    wire is_flush_read = dcr_bus_if.req_valid
                      && ~dcr_bus_if.req_data.rw
                      && (dcr_bus_if.req_data.addr == `VX_DCR_BASE_CACHE_FLUSH)
                      && (mpm_target_cid == 16'(CORE_ID))
                      && ~flush_pending_r;

    always @(posedge clk) begin
        if (reset) begin
            flush_pending_r <= 1'b0;
        end else if (is_flush_read) begin
            flush_pending_r <= 1'b1;
        end else if (dcr_flush_if.done) begin
            flush_pending_r <= 1'b0;
        end
    end

    assign dcr_flush_if.req = flush_pending_r;

    wire flush_done = flush_pending_r && dcr_flush_if.done;

    // Return CSR data or flush-done on DCR response bus
    // Use intermediate registers — Vivado (Synth 8-453) forbids procedural
    // assignment to interface net signals; drive them with continuous assigns.
    reg         rsp_valid_r;
    dcr_rsp_t   rsp_data_r;

    always @(posedge clk) begin
        if (reset) begin
            rsp_valid_r <= 1'b0;
            rsp_data_r  <= '0;
        end else begin
            rsp_valid_r <= dcr_csr_if_fire || flush_done;
            if (dcr_csr_if_fire) begin
                rsp_data_r <= dcr_csr_if.value;
            end else if (flush_done) begin
                rsp_data_r <= '0;
            end
        end
    end

    assign dcr_bus_if.rsp_valid = rsp_valid_r;
    assign dcr_bus_if.rsp_data  = rsp_data_r;

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // The KMU's launch DCRs, kept here as the KMU keeps them, for the cluster identity CSRs.
    reg [2:0][CLUSTER_SIZE_W-1:0] cluster_dim_r;
    reg [`VX_CFG_LMEM_LOG_SIZE:0] lmem_size_r;
    reg                           cluster_explicit_r;

    wire dcr_write = dcr_bus_if.req_valid && dcr_bus_if.req_data.rw;

    always @(posedge clk) begin
        if (reset) begin
            cluster_dim_r      <= {3{CLUSTER_SIZE_W'(1)}};
            lmem_size_r        <= '0;
            cluster_explicit_r <= 1'b0;
        end else if (dcr_write) begin
            case (dcr_bus_if.req_data.addr)
            `VX_DCR_KMU_CLUSTER_DIM_X:  cluster_dim_r[0]   <= dcr_bus_if.req_data.data[CLUSTER_SIZE_W-1:0];
            `VX_DCR_KMU_CLUSTER_DIM_Y:  cluster_dim_r[1]   <= dcr_bus_if.req_data.data[CLUSTER_SIZE_W-1:0];
            `VX_DCR_KMU_CLUSTER_DIM_Z:  cluster_dim_r[2]   <= dcr_bus_if.req_data.data[CLUSTER_SIZE_W-1:0];
            `VX_DCR_KMU_LMEM_SIZE:      lmem_size_r        <= dcr_bus_if.req_data.data[`VX_CFG_LMEM_LOG_SIZE:0];
            `VX_DCR_KMU_CLUSTER_POLICY: cluster_explicit_r <= dcr_bus_if.req_data.data[1];
            default:;
            endcase
        end
    end

    assign dcr_csr_if.cluster_dim      = cluster_dim_r;
    assign dcr_csr_if.lmem_size        = lmem_size_r;
    assign dcr_csr_if.cluster_explicit = cluster_explicit_r;
`endif

`ifdef VX_CFG_H100_SM_ENABLE
    // A new split may only be written while the SM is idle and its L1 drained (the runtime writes it between launches).
    localparam S_W = `CLOG2(`VX_CFG_SM_SRAM_BYTES + 1);
    localparam L1_AVAIL_BYTES = `VX_CFG_SM_SRAM_BYTES - `VX_CFG_L1_RESERVE_BYTES;
    `STATIC_ASSERT(`VX_CFG_SMEM_PER_SM_MAX <= L1_AVAIL_BYTES, ("the shared-memory maximum exceeds the array"))

    function automatic smem_cfg_t smem_cfg_of(input [S_W-1:0] s_bytes);
        logic [S_W-1:0] lines, prot;
        smem_cfg_of.s_rows = (SM_SRAM_ROW_W+1)'(s_bytes / S_W'(SM_SRAM_ROW_BYTES));
        lines = (S_W'(L1_AVAIL_BYTES) - s_bytes) / S_W'(SM_SRAM_ROW_BYTES);
        prot  = S_W'((32'(lines) * 32'(`VX_CFG_L1_REPL_PROTECT_PCT)) / 32'd100);
        smem_cfg_of.l1_lines   = (L1_SLOT_W+1)'(lines);
        smem_cfg_of.l1_protect = $bits(smem_cfg_of.l1_protect)'((prot > S_W'(`VX_CFG_SM_MAX_READ_LINES)) ? prot : S_W'(`VX_CFG_SM_MAX_READ_LINES));
    endfunction

    wire smem_cfg_write = dcr_bus_if.req_valid && dcr_bus_if.req_data.rw
                       && (dcr_bus_if.req_data.addr == `VX_DCR_KMU_SMEM_CONFIG);
    wire [S_W-1:0] smem_cfg_s = S_W'(dcr_bus_if.req_data.data);

    smem_cfg_t smem_cfg_r;
    always @(posedge clk) begin
        if (reset) begin
            smem_cfg_r <= smem_cfg_of(S_W'(`VX_CFG_SMEM_PER_SM_MAX));
        end else if (smem_cfg_write) begin
            smem_cfg_r <= smem_cfg_of(smem_cfg_s);
        end
    end
    `RUNTIME_ASSERT(~smem_cfg_write || (smem_cfg_s <= S_W'(`VX_CFG_SMEM_PER_SM_MAX)
                                        && (smem_cfg_s % S_W'(SM_SRAM_ROW_BYTES)) == 0),
                    ("%t: *** %s invalid shared-memory split 0x%0h", $time, INSTANCE_ID, dcr_bus_if.req_data.data))
    assign smem_cfg = smem_cfg_r;
`endif

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (dcr_bus_if.req_valid && dcr_bus_if.req_data.rw) begin
            `TRACE(2, ("%t: %s: state=", $time, INSTANCE_ID))
            VX_trace_pkg::trace_base_dcr(1, dcr_bus_if.req_data.addr);
            `TRACE(2, (", rw=%b, data=0x%h\n", dcr_bus_if.req_data.rw, dcr_bus_if.req_data.data))
        end
    end
`endif

endmodule
