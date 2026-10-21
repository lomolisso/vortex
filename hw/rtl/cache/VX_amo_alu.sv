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

// Pure combinational RVA RMW kernel. Given the cache line's current word
// at the AMO byte offset, the rs2 operand, and (for compare-and-swap) the
// comparand, derives:
//   - new_word: the value to write back for store-bearing AMOs (all except LR).
//   - ret_word: the original loaded value, sign-extended into rd.
//               For SC the bank overrides this with 0/1 outside this module.
// DATA_WIDTH is the synthesized operand width (= the cache word width, capped
// at 64): a 32-bit-word cache can only carry .W atomics, so the adder and
// comparators are built 32-bit rather than 64-bit. width selects .W vs .D and
// is only meaningful when DATA_WIDTH > 32. ARITH_WIDTH bounds every op except
// compare-and-swap: a 32-bit hart issues 64-bit operands only for the Zacas
// pair, so its adder and comparators shrink to 32 bits while the CAS equality
// keeps the full width.
module VX_amo_alu import VX_gpu_pkg::*; #(
    parameter DATA_WIDTH  = 64,
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // mbarrier ops on a 64-bit word; only the local memory banks run them.
    parameter MBAR_ENABLE = 0,
`endif
    parameter ARITH_WIDTH = DATA_WIDTH
) (
    input  amo_op_e     op,
    input  wire         is_unsigned, // selects MIN/MAX variant
    input  wire [1:0]   width,        // 2 = .W, 3 = .D
    input  wire [63:0]  old_word,
    input  wire [63:0]  rhs,
    input  wire [63:0]  cmp,         // compare-and-swap comparand
`ifdef VX_CFG_EXT_MBAR_ENABLE
    output wire         mbar_flip,   // the mbarrier op completed the phase
    output wire [1:0]   mbar_err,    // {tx out of range, pending underflow}
`endif
    output wire [63:0]  new_word,
    output wire [63:0]  ret_word
);
    localparam AW  = DATA_WIDTH;
    localparam ARW = (ARITH_WIDTH < AW) ? ARITH_WIDTH : AW;

    // .W and .D only differ when the datapath is wider than 32 bits;
    // a <= 32-bit operand width can only ever be a .W atomic. Non-CAS ops
    // additionally cap at ARW (a .D non-CAS atomic cannot reach an
    // ARITH_WIDTH=32 build; the bank asserts this).
    wire is_w = (AW > 32) ? (width == 2'd2) : 1'b1;
    wire arith_w = (ARW > 32) ? is_w : 1'b1;
    if (AW <= 32) begin : g_w_only
        `UNUSED_VAR (width)
    end
    if (AW < 64) begin : g_hi_unused
        `UNUSED_VAR (old_word[63:AW])
        `UNUSED_VAR (rhs[63:AW])
    end

    wire [AW-1:0] a = old_word[AW-1:0];
    wire [AW-1:0] b = rhs[AW-1:0];

    // Mask to width-sized values; sign-extend at the 32-bit boundary for MIN/MAX.
    wire [ARW-1:0] a_v = a[ARW-1:0];
    wire [ARW-1:0] b_v = b[ARW-1:0];
    wire [ARW-1:0] a_u = arith_w ? {{(ARW-32){1'b0}},  a_v[31:0]} : a_v;
    wire [ARW-1:0] b_u = arith_w ? {{(ARW-32){1'b0}},  b_v[31:0]} : b_v;
    wire signed [ARW-1:0] a_s = arith_w ? {{(ARW-32){a_v[31]}}, a_v[31:0]} : a_v;
    wire signed [ARW-1:0] b_s = arith_w ? {{(ARW-32){b_v[31]}}, b_v[31:0]} : b_v;

    reg [ARW-1:0] arith_res;
    always @(*) begin
        case (op)
            AMO_OP_LR:    arith_res = a_u;
            AMO_OP_SC:    arith_res = b_u;
            AMO_OP_SWAP:  arith_res = b_u;
            AMO_OP_ADD:   arith_res = a_u + b_u;
            AMO_OP_AND:   arith_res = a_u & b_u;
            AMO_OP_OR:    arith_res = a_u | b_u;
            AMO_OP_XOR:   arith_res = a_u ^ b_u;
            AMO_OP_MIN:   arith_res = is_unsigned ? ((a_u < b_u) ? a_u : b_u)
                                                  : ((a_s < b_s) ? a_s : b_s);
            AMO_OP_MAX:   arith_res = is_unsigned ? ((a_u > b_u) ? a_u : b_u)
                                                  : ((a_s > b_s) ? a_s : b_s);
            default:      arith_res = a_u;
        endcase
        if (arith_w) arith_res = {{(ARW-32){1'b0}}, arith_res[31:0]};
    end

`ifdef VX_CFG_EXT_ZACAS_ENABLE
    wire [AW-1:0] c = cmp[AW-1:0];
    if (AW < 64) begin : g_cmp_hi_unused
        `UNUSED_VAR (cmp[63:AW])
    end
    wire [AW-1:0] a_cu = is_w ? {{(AW-32){1'b0}}, a[31:0]} : a;
    wire [AW-1:0] b_cu = is_w ? {{(AW-32){1'b0}}, b[31:0]} : b;
    wire [AW-1:0] c_cu = is_w ? {{(AW-32){1'b0}}, c[31:0]} : c;
    // On a mismatch the old value is written back unchanged rather than the
    // store being suppressed; the commit still breaks other harts'
    // reservations on the word, which is the required behaviour and the
    // reason it is not treated as a no-op.
    wire [AW-1:0] cas_res = (a_cu == c_cu) ? b_cu : a_cu;
    wire [AW-1:0] res = (op == AMO_OP_CAS) ? cas_res : AW'(arith_res);
`else
    `UNUSED_VAR (cmp)
    if (ARW < AW) begin : g_arith_hi_unused
        `UNUSED_VAR (b[AW-1:ARW])
    end
    wire [AW-1:0] res = AW'(arith_res);
`endif

    // Zero-extend the AW-sized results back to the 64-bit port.
    // For SC the bank overrides ret_word with 0/1.
`ifdef VX_CFG_EXT_MBAR_ENABLE
    wire [63:0] rmw_new = 64'(res);
    wire [63:0] rmw_ret = 64'(is_w ? {{(AW-32){1'b0}}, a[31:0]} : a);

    if (MBAR_ENABLE != 0) begin : g_mbar
        `STATIC_ASSERT (DATA_WIDTH == 64, ("mbarrier ops need a 64-bit word"))

        // The object (sw/common/mbar_ref.h): pending[19:0] expected[39:20]
        // tx[60:40] (signed bytes) [62:61] zero phase[63]. Fields wrap at their
        // widths, so an erroneous op still leaves a defined word.
        localparam signed [22:0] TX_MAX = 23'sd1048575;

        wire        is_mbar = (op >= AMO_OP_MBAR_ARRIVE);
        wire        is_wait = (op == AMO_OP_MBAR_TEST_WAIT) || (op == AMO_OP_MBAR_TRY_WAIT);
        wire [19:0] pend    = old_word[19:0];
        wire [19:0] expd    = old_word[39:20];
        wire [20:0] tx      = old_word[60:40];
        wire        phase   = old_word[63];
        // A byte count of 2^21 or more cannot leave tx in range.
        wire        big     = (| rhs[63:21]);
        wire signed [22:0] tx_o  = {{2{tx[20]}}, tx};
        wire signed [22:0] bytes = {2'b00, rhs[20:0]};

        reg signed [22:0] tx_n;
        reg [19:0]        pend_n;
        reg               underflow, bad_bytes;
        always @(*) begin
            tx_n      = tx_o;
            pend_n    = pend;
            underflow = 1'b0;
            bad_bytes = 1'b0;
            case (op)
            AMO_OP_MBAR_ARRIVE: begin
                underflow = (rhs > 64'(pend));
                pend_n    = pend - rhs[19:0];
            end
            AMO_OP_MBAR_ARRIVE_TX: begin
                bad_bytes = big;
                tx_n      = tx_o + bytes;
                underflow = (pend == 20'd0);
                pend_n    = pend - 20'd1;
            end
            AMO_OP_MBAR_EXPECT_TX: begin
                bad_bytes = big;
                tx_n      = tx_o + bytes;
            end
            AMO_OP_MBAR_COMPLETE_TX: begin
                bad_bytes = big;
                tx_n      = tx_o - bytes;
            end
            default:;
            endcase
        end
        wire tx_range = bad_bytes || (tx_n > TX_MAX) || (tx_n < -TX_MAX);

        // Every writing op ends with the completion check, expect_tx included: a
        // complete_tx may come first and leave tx negative.
        wire        done   = (pend_n == 20'd0) && (tx_n[20:0] == 21'd0);
        wire [63:0] mbar_new = {phase ^ done, 2'b00, tx_n[20:0], expd, done ? expd : pend_n};
        wire        writes = is_mbar && ~is_wait;

        assign new_word  = writes ? mbar_new : (is_mbar ? old_word : rmw_new);
        assign ret_word  = (is_mbar && is_wait) ? {63'b0, phase != rhs[63]} : (is_mbar ? old_word : rmw_ret);
        assign mbar_flip = writes && done;
        assign mbar_err  = writes ? {tx_range, underflow} : 2'b00;
    end else begin : g_no_mbar
        assign new_word  = rmw_new;
        assign ret_word  = rmw_ret;
        assign mbar_flip = 1'b0;
        assign mbar_err  = 2'b00;
    end
`else
    assign new_word = 64'(res);
    assign ret_word = 64'(is_w ? {{(AW-32){1'b0}}, a[31:0]} : a);
`endif

endmodule
