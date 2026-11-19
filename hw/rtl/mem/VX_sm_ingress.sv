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

`ifdef VX_CFG_H100_SM_ENABLE

`include "VX_define.vh"

// An SM's ingress gate, cycle for cycle sm_port_ref::SmIngressRef: each cycle
// it deals UNITS sectors to the heads of three sources, fills (L2 and TMA
// replies) against SM-to-SM arrivals by a weighted wheel, the two SM-to-SM
// sources round robin between them over the one ejection link of EJECT flits
// per cycle, a message being one flit. A head
// leaves once paid and accepted; a fill head may be prepaid one cycle ahead.
module VX_sm_ingress import VX_gpu_pkg::*; #(
    parameter SECT_W     = 1,
    parameter UNITS      = 4,
    parameter EJECT      = 1,
    parameter W_FILL     = 5,
    parameter W_DSMEM    = 1
) (
    input wire                  clk,
    input wire                  reset,

    input wire                  fill_valid,
    input wire [SECT_W-1:0]     fill_sectors,
    output wire                 fill_pass,
    input wire                  fill_ready,

    input wire                  dreq_valid,
    input wire [SECT_W-1:0]     dreq_sectors,
    output wire                 dreq_pass,
    input wire                  dreq_ready,

    input wire                  drsp_valid,
    input wire [SECT_W-1:0]     drsp_sectors,
    output wire                 drsp_pass,
    input wire                  drsp_ready
);
    localparam WHEEL  = W_FILL + W_DSMEM;
    localparam WPOS_W = `LOG2UP(WHEEL);
    localparam BANK_W = SECT_W + `CLOG2(UNITS) + 1;

    reg [BANK_W-1:0] bank_f_r, bank_q_r, bank_p_r;
    reg [WPOS_W-1:0] wpos_r;
    reg              dturn_r; // 0: request arrivals next, 1: reply arrivals next

    reg [BANK_W-1:0] bank_f_n, bank_q_n, bank_p_n;
    reg [WPOS_W-1:0] wpos_n;
    reg              dturn_n;

    always @(*) begin
        reg [`CLOG2(EJECT+1)-1:0] dsm_used;
        reg fc, rc, pc, dc, to_fill;
        bank_f_n = fill_valid ? bank_f_r : '0;
        bank_q_n = dreq_valid ? bank_q_r : '0;
        bank_p_n = drsp_valid ? bank_p_r : '0;
        wpos_n   = wpos_r;
        dturn_n  = dturn_r;
        dsm_used = '0;
        for (integer u = 0; u < UNITS; ++u) begin
            fc = fill_valid && (fill_sectors != '0)
              && (bank_f_n < (BANK_W'(fill_sectors) + BANK_W'(UNITS - 1)));
            rc = dreq_valid && (bank_q_n < BANK_W'(dreq_sectors)) && (32'(dsm_used) < EJECT);
            pc = drsp_valid && (bank_p_n < BANK_W'(drsp_sectors)) && (32'(dsm_used) < EJECT);
            dc = rc || pc;
            to_fill = fc && ((32'(wpos_n) < W_FILL) || ~dc);
            if (fc || dc) begin
                if (to_fill) begin
                    bank_f_n = bank_f_n + BANK_W'(1);
                end else if (rc && (~pc || ~dturn_n)) begin
                    bank_q_n = bank_q_n + BANK_W'(1);
                    dsm_used = dsm_used + 1;
                    dturn_n  = 1'b1;
                end else begin
                    bank_p_n = bank_p_n + BANK_W'(1);
                    dsm_used = dsm_used + 1;
                    dturn_n  = 1'b0;
                end
                wpos_n = (32'(wpos_n) == (WHEEL - 1)) ? '0 : (wpos_n + WPOS_W'(1));
            end
        end
    end

    assign fill_pass = fill_valid && (bank_f_n >= BANK_W'(fill_sectors)) && fill_ready;
    assign dreq_pass = dreq_valid && (bank_q_n >= BANK_W'(dreq_sectors)) && dreq_ready;
    assign drsp_pass = drsp_valid && (bank_p_n >= BANK_W'(drsp_sectors)) && drsp_ready;

    always @(posedge clk) begin
        if (reset) begin
            bank_f_r <= '0;
            bank_q_r <= '0;
            bank_p_r <= '0;
            wpos_r   <= '0;
            dturn_r  <= 1'b0;
        end else begin
            bank_f_r <= bank_f_n - (fill_pass ? BANK_W'(fill_sectors) : '0);
            bank_q_r <= bank_q_n - (dreq_pass ? BANK_W'(dreq_sectors) : '0);
            bank_p_r <= bank_p_n - (drsp_pass ? BANK_W'(drsp_sectors) : '0);
            wpos_r   <= wpos_n;
            dturn_r  <= dturn_n;
        end
    end

endmodule

`endif // VX_CFG_H100_SM_ENABLE
