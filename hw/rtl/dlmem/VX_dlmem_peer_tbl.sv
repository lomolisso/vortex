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

// A core's DLMEM peer rows, one per CTA slot, plus the launch's per-slot local
// memory base (slot * stride). A row written in cycle w is read from cycle w+1.
module VX_dlmem_peer_tbl import VX_gpu_pkg::*; #(
    parameter  NUM_RD   = `VX_CFG_NUM_LSU_BLOCKS,

    localparam NW       = `VX_CFG_NUM_WARPS,
    localparam LMEM_LOG = `VX_CFG_LMEM_LOG_SIZE
) (
    input  wire                             clk,
    input  wire                             reset,

    input  wire                             wr_valid,
    input  wire [NW_WIDTH-1:0]              wr_slot,
    input  dlmem_row_t                      wr_row,
    input  wire [LMEM_LOG:0]                wr_stride,

    input  wire [NUM_RD-1:0][NW_WIDTH-1:0]  rd_slot,
    output dlmem_row_t [NUM_RD-1:0]         rd_row,

    output wire [NW-1:0][LMEM_LOG-1:0]      slot_base,
    output wire [LMEM_LOG:0]                stride
);
    localparam PROD_W = LMEM_LOG + 1 + NW_WIDTH;

    // Peers are don't-care while k == 0, so only k is reset.
    reg [NW-1:0][DLMEM_K_W-1:0]                         row_k_r;
    dlmem_peer_t [NW-1:0][DLMEM_MAX_CLUSTER-1:0]        row_peers_r;
    reg [NW-1:0][LMEM_LOG-1:0]                          slot_base_r;
    reg [LMEM_LOG:0]                                    stride_r;

    for (genvar s = 0; s < NW; ++s) begin : g_rows
        wire row_we = wr_valid && (wr_slot == NW_WIDTH'(s));

        always @(posedge clk) begin
            if (reset) begin
                row_k_r[s] <= '0;
            end else if (row_we) begin
                row_k_r[s] <= wr_row.k;
            end
        end

        always @(posedge clk) begin
            if (row_we) begin
                row_peers_r[s] <= wr_row.peers;
            end
        end

        // A slot past the launch's occupancy wraps; no in-bounds access reaches it.
        wire [PROD_W-1:0] slot_prod = PROD_W'(wr_stride) * PROD_W'(s);

        always @(posedge clk) begin
            if (reset) begin
                slot_base_r[s] <= '0;
            end else if (wr_valid) begin
                slot_base_r[s] <= slot_prod[LMEM_LOG-1:0];
            end
        end
        `UNUSED_VAR (slot_prod[PROD_W-1:LMEM_LOG])
    end

    always @(posedge clk) begin
        if (reset) begin
            stride_r <= '0;
        end else if (wr_valid) begin
            stride_r <= wr_stride;
        end
    end

    for (genvar b = 0; b < NUM_RD; ++b) begin : g_rd
        assign rd_row[b].k     = row_k_r[rd_slot[b]];
        assign rd_row[b].peers = row_peers_r[rd_slot[b]];
    end

    assign slot_base = slot_base_r;
    assign stride    = stride_r;

endmodule
