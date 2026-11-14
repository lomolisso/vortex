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

// The guard sits above the include so the file preprocesses to nothing
// without H100 WGMMA.
`ifdef VX_CFG_TCU_H100_ENABLE

`include "VX_define.vh"

// Local-memory byte address of operand element (mn, k) named by a 64-bit
// wgmma shared-memory descriptor:
//   [13:0] start>>4, [29:16] LBO>>4, [45:32] SBO>>4, [51:49] base offset,
//   [52] MN-major, [63:62] layout (0 none, 1 128B, 2 64B, 3 32B swizzle).
// The canonical layout gives an address relative to the CTA's window; the
// swizzle XOR is applied to that relative address, which is the origin the
// tensor copies swizzle from, and the CTA's slot base is added last.
// Combinational; the clock only times the descriptor checks.

module VX_tcu_smem_addr #(
    parameter LMEM_LOG = `VX_CFG_LMEM_LOG_SIZE
) (
    input  wire                clk,
    input  wire                reset,
    input  wire                check,      // the address is used this cycle
    input  wire [63:0]         desc,
    input  wire [LMEM_LOG-1:0] slot_base,
    input  wire                esize16,    // 16-bit elements (else 8-bit)
    input  wire [7:0]          mn,
    input  wire [5:0]          k,
    output wire [LMEM_LOG-1:0] addr
);
    `UNUSED_VAR ({clk, reset, check})

    // Byte offsets wrap at 256 KB, the reach of the 14-bit fields.
    localparam AW = 18;

    wire [AW-1:0] start = {desc[13:0], 4'b0};
    wire [AW-1:0] lbo   = {desc[29:16], 4'b0};
    wire [AW-1:0] sbo   = {desc[45:32], 4'b0};
    wire [2:0]    base_off = desc[51:49];
    wire          mn_major = desc[52];
    wire [1:0]    layout   = desc[63:62];
    `UNUSED_VAR (base_off)

    // log2 of the 16-byte chunks per swizzle row.
    logic [1:0] lg_w;
    always_comb begin
        case (layout)
            2'd1:    lg_w = 2'd3;
            2'd2:    lg_w = 2'd2;
            2'd3:    lg_w = 2'd1;
            default: lg_w = 2'd0;
        endcase
    end
    wire swizzled = (layout != 2'd0);

    // K-major: rows of 8 along MN, k contiguous inside a 16-byte chunk.
    wire [2:0]    km_row  = mn[2:0];
    wire [4:0]    km_grp  = mn[7:3];
    wire [3:0]    km_kc   = esize16 ? {1'b0, k[5:3]} : {2'b0, k[5:4]};
    wire [3:0]    km_kb   = esize16 ? {k[2:0], 1'b0} : k[3:0];
    wire [AW-1:0] km_row_off = AW'(km_row) << (4 + lg_w);
    wire [AW-1:0] km_k_off   = swizzled ? (AW'(km_kc) << 4) : (AW'(km_kc) * lbo);
    wire [AW-1:0] km_addr = start + km_row_off + AW'(km_grp) * sbo + km_k_off + AW'(km_kb);

    // MN-major (16-bit only): eight elements per chunk along MN, k rows of
    // one swizzle row each.
    wire [4:0]    mm_mc  = mn[7:3];
    wire [3:0]    mm_mb  = {mn[2:0], 1'b0};
    wire [4:0]    mm_mc_lo = mm_mc & ((5'd1 << lg_w) - 5'd1);
    wire [4:0]    mm_mc_hi = mm_mc >> lg_w;
    wire [AW-1:0] mm_sw_addr = start + AW'(mm_mb)
                             + (AW'(mm_mc_lo) << 4) + AW'(mm_mc_hi) * lbo
                             + (AW'(k[2:0]) << (4 + lg_w)) + AW'(k[5:3]) * sbo;
    wire [AW-1:0] mm_in_addr = start + AW'(mm_mb) + AW'(mm_mc) * sbo
                             + (AW'(k[2:0]) << 4) + AW'(k[5:3]) * lbo;
    wire [AW-1:0] mm_addr = swizzled ? mm_sw_addr : mm_in_addr;

    wire [AW-1:0] rel = mn_major ? mm_addr : km_addr;

    // 16-byte chunk bits [6:4] XOR row bits [9:7], masked to the mode's width.
    wire [2:0]    sw_mask = 3'((4'd1 << lg_w) - 4'd1);
    wire [AW-1:0] rel_sw  = rel ^ (AW'(rel[9:7] & sw_mask) << 4);

    assign addr = slot_base + LMEM_LOG'(rel_sw);
    `UNUSED_VAR (rel_sw)

`ifdef SIMULATION
    `RUNTIME_ASSERT(!check || (desc[51:49] == 3'b0),
        ("%t: wgmma descriptor base offset must be 0 (desc=0x%0h)", $time, desc))
    `RUNTIME_ASSERT(!check || ((desc[15:14] == 2'b0) && (desc[31:30] == 2'b0)
                            && (desc[48:46] == 3'b0) && (desc[61:53] == 9'b0)),
        ("%t: wgmma descriptor reserved bits must be 0 (desc=0x%0h)", $time, desc))
    `RUNTIME_ASSERT(!check || !mn_major || esize16,
        ("%t: wgmma MN-major needs a 16-bit type (desc=0x%0h)", $time, desc))
    `RUNTIME_ASSERT(!check || (slot_base[3:0] == 4'b0),
        ("%t: wgmma operand base must be 16-byte aligned (slot base=0x%0h)", $time, slot_base))
`endif

endmodule

`endif // VX_CFG_TCU_H100_ENABLE
