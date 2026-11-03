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

// VX_config.vh preprocesses to nothing, so with the knob off this file is empty.
`include "VX_config.vh"

`ifdef VX_CFG_EXT_FPAMO_ENABLE
`include "VX_define.vh"
`include "HardFloat_consts.vi"

// Float atomic kernel (golden model: sw/common/amo_ref.h). old/rhs arrive
// right-aligned at the op width; new_word leaves LATENCY cycles later,
// right-aligned and zero above the width. Lane 0 adds at f64 width and lane 1
// (the upper element of an x2 form) at f32 width; either raw sum is rounded
// once, straight to the op's format, round to nearest even. The packed min/max
// are comparators. A NaN result of the 16-bit and f32 forms is the format's
// canonical NaN, and an invalid f64 add (inf - inf) gives F64_QNAN. The unit
// follows the last-level cache's rules (amo_ref::SITE_LLC): f32 and f32x2
// subnormal inputs and results are flushed to a sign-preserving zero, and an
// f64 NaN input comes back unchanged, rhs before old. The 16-bit forms and f64
// never flush.
module VX_amo_fpu import VX_gpu_pkg::*; #(
    parameter LATENCY = `VX_CFG_FPAMO_LATENCY
) (
    input  wire         clk,
    input  wire         reset,
    input  amo_op_e     op,
    input  wire [1:0]   width,      // 2 = .W, 3 = .D
    input  wire [63:0]  old_word,
    input  wire [63:0]  rhs,
    output wire [63:0]  new_word
);
    `STATIC_ASSERT (LATENCY >= 1, ("VX_amo_fpu: LATENCY must be at least 1"))

    // Canonical NaNs and the min/max rule; amo_ref.h holds the same values.
    localparam [15:0] F16_QNAN  = 16'h7fff;
    localparam [15:0] BF16_QNAN = 16'h7fff;
    localparam [31:0] F32_QNAN  = 32'h7fffffff;
    localparam [63:0] F64_QNAN  = 64'hfff8000000000000;
    // minimumNumber/maximumNumber: a NaN operand yields the other one, two NaNs
    // the canonical NaN, and -0 orders below +0.

    // Register stages sit between the HardFloat blocks, added in this order as
    // the latency grows: after the raw add, after the input recoding, after the
    // conversion to IEEE, at the output, after the rounding, then more at the
    // output. Each cut goes where it shortens the worst stage; the f64 add,
    // the longest block, is a stage of its own from two on. Below four, the
    // logic after the last cut (the NaN fixups and packing, and at one or two
    // the rounding and conversion too) ends in the consumer's register, so
    // new_word depends on registers only, never on the inputs.
    localparam D_ADD = 1;
    localparam D_REC = (LATENCY >= 2) ? 1 : 0;
    localparam D_CVT = (LATENCY >= 3) ? 1 : 0;
    localparam D_RND = (LATENCY >= 5) ? 1 : 0;
    localparam D_OUT = LATENCY - D_ADD - D_REC - D_CVT - D_RND;

    localparam FMT_F16  = 2'd0;
    localparam FMT_BF16 = 2'd1;
    localparam FMT_F32  = 2'd2;
    localparam FMT_F64  = 2'd3;

    wire control = `flControl_tininessAfterRounding;
    wire [2:0] rnd_mode = `round_near_even;
    // Both addends are values of the result's format, so a sum in its subnormal
    // range is exact and the rounding position never depends on the exponent.
    localparam RND_OPT = `flRoundOpt_subnormsAlwaysExact;

    // Op decode

    reg [1:0] fmt;
    reg       is_x2, is_mm, is_max;
    always @(*) begin
        fmt    = FMT_F32;
        is_x2  = 1'b0;
        is_mm  = 1'b0;
        is_max = 1'b0;
        case (op)
            AMO_OP_FADD:     fmt = (width == 2'd2) ? FMT_F32 : FMT_F64;
            AMO_OP_FADD2:    begin fmt = (width == 2'd2) ? FMT_F16 : FMT_F32; is_x2 = 1'b1; end
            AMO_OP_FADD2_BF: begin fmt = FMT_BF16; is_x2 = 1'b1; end
            AMO_OP_FMIN2:    begin fmt = FMT_F16;  is_x2 = 1'b1; is_mm = 1'b1; end
            AMO_OP_FMAX2:    begin fmt = FMT_F16;  is_x2 = 1'b1; is_mm = 1'b1; is_max = 1'b1; end
            AMO_OP_FMIN2_BF: begin fmt = FMT_BF16; is_x2 = 1'b1; is_mm = 1'b1; end
            AMO_OP_FMAX2_BF: begin fmt = FMT_BF16; is_x2 = 1'b1; is_mm = 1'b1; is_max = 1'b1; end
            default:;
        endcase
    end

    wire ftz = (fmt == FMT_F32);

    // An f64 NaN input bypasses the adder.
    wire old_nan64 = (old_word[62:52] == 11'h7ff) && (old_word[51:0] != 52'd0);
    wire rhs_nan64 = (rhs[62:52] == 11'h7ff) && (rhs[51:0] != 52'd0);
    wire f64_pass = (fmt == FMT_F64) && (old_nan64 || rhs_nan64);
    wire [63:0] f64_pass_val = rhs_nan64 ? rhs : old_word;

    // A subnormal f32 becomes a zero of its sign.
    function automatic logic [31:0] f32_ftz(input logic [31:0] x);
        return (x[30:23] == 8'd0) ? {x[31], 31'd0} : x;
    endfunction

    // A 32-bit element (f32, or bf16 at the top of an f32) under FTZ.
    function automatic logic [31:0] elem32(input logic [1:0] f, input logic do_ftz,
                                           input logic [31:0] w32, input logic [15:0] w16);
        logic [31:0] x;
        x = (f == FMT_BF16) ? {w16, 16'd0} : w32;
        return do_ftz ? f32_ftz(x) : x;
    endfunction

    // Packed 16-bit min/max

    function automatic logic is_nan16(input logic bf, input logic [14:0] x);
        return bf ? ((x[14:7] == 8'hff) && (x[6:0] != 7'd0))
                  : ((x[14:10] == 5'h1f) && (x[9:0] != 10'd0));
    endfunction

    // Order key of a non-NaN value: -0 below +0.
    function automatic logic [15:0] key16(input logic [15:0] x);
        return x[15] ? ~x : (x | 16'h8000);
    endfunction

    function automatic logic [15:0] minmax16(input logic bf, input logic mx,
                                             input logic [15:0] a, input logic [15:0] b);
        logic na, nb, a_lt_b;
        na = is_nan16(bf, a[14:0]);
        nb = is_nan16(bf, b[14:0]);
        a_lt_b = (key16(a) < key16(b));
        if (na && nb) begin
            return bf ? BF16_QNAN : F16_QNAN;
        end else if (na) begin
            return b;
        end else if (nb) begin
            return a;
        end
        return (a_lt_b != mx) ? a : b;
    endfunction

    wire is_bf = (fmt == FMT_BF16);
    wire [31:0] mm_res = {minmax16(is_bf, is_max, old_word[31:16], rhs[31:16]),
                          minmax16(is_bf, is_max, old_word[15:0],  rhs[15:0])};

    // A min/max or an f64 NaN pass skips the arithmetic: its result rides
    // beside the adders to the output.
    wire        byp     = is_mm || f64_pass;
    wire [63:0] byp_val = is_mm ? 64'(mm_res) : f64_pass_val;

    // Lane 0: every format widened exactly to recoded f64.

    wire [16:0] a0_r16, b0_r16;
    wire [32:0] a0_r32n, b0_r32n, a0_r32w, b0_r32w;
    wire [64:0] a0_r64n, b0_r64n, a0_r64w, b0_r64w;

    VX_amo_fpu_rec #(.EXPW(5), .SIGW(11)) a0_from16 (.in(old_word[15:0]), .out(a0_r16));
    VX_amo_fpu_rec #(.EXPW(5), .SIGW(11)) b0_from16 (.in(rhs[15:0]),      .out(b0_r16));
    VX_amo_fpu_rec #(.EXPW(8), .SIGW(24)) a0_from32 (.in(elem32(fmt, ftz, old_word[31:0], old_word[15:0])), .out(a0_r32n));
    VX_amo_fpu_rec #(.EXPW(8), .SIGW(24)) b0_from32 (.in(elem32(fmt, ftz, rhs[31:0], rhs[15:0])),           .out(b0_r32n));
    recFNToRecFN_unsafe #(.inExpWidth(5), .inSigWidth(11), .outExpWidth(8), .outSigWidth(24)) a0_16to32 (.in(a0_r16), .out(a0_r32w));
    recFNToRecFN_unsafe #(.inExpWidth(5), .inSigWidth(11), .outExpWidth(8), .outSigWidth(24)) b0_16to32 (.in(b0_r16), .out(b0_r32w));
    wire [32:0] a0_r32 = (fmt == FMT_F16) ? a0_r32w : a0_r32n;
    wire [32:0] b0_r32 = (fmt == FMT_F16) ? b0_r32w : b0_r32n;
    VX_amo_fpu_rec #(.EXPW(11), .SIGW(53)) a0_from64 (.in(old_word), .out(a0_r64n));
    VX_amo_fpu_rec #(.EXPW(11), .SIGW(53)) b0_from64 (.in(rhs),      .out(b0_r64n));
    recFNToRecFN_unsafe #(.inExpWidth(8), .inSigWidth(24), .outExpWidth(11), .outSigWidth(53)) a0_32to64 (.in(a0_r32), .out(a0_r64w));
    recFNToRecFN_unsafe #(.inExpWidth(8), .inSigWidth(24), .outExpWidth(11), .outSigWidth(53)) b0_32to64 (.in(b0_r32), .out(b0_r64w));
    wire [64:0] a0_rec = (fmt == FMT_F64) ? a0_r64n : a0_r64w;
    wire [64:0] b0_rec = (fmt == FMT_F64) ? b0_r64n : b0_r64w;

    // Lane 1: the upper element, widened exactly to recoded f32.

    wire [16:0] a1_r16, b1_r16;
    wire [32:0] a1_r32n, b1_r32n, a1_r32w, b1_r32w;

    VX_amo_fpu_rec #(.EXPW(5), .SIGW(11)) a1_from16 (.in(old_word[31:16]), .out(a1_r16));
    VX_amo_fpu_rec #(.EXPW(5), .SIGW(11)) b1_from16 (.in(rhs[31:16]),      .out(b1_r16));
    VX_amo_fpu_rec #(.EXPW(8), .SIGW(24)) a1_from32 (.in(elem32(fmt, ftz, old_word[63:32], old_word[31:16])), .out(a1_r32n));
    VX_amo_fpu_rec #(.EXPW(8), .SIGW(24)) b1_from32 (.in(elem32(fmt, ftz, rhs[63:32], rhs[31:16])),           .out(b1_r32n));
    recFNToRecFN_unsafe #(.inExpWidth(5), .inSigWidth(11), .outExpWidth(8), .outSigWidth(24)) a1_16to32 (.in(a1_r16), .out(a1_r32w));
    recFNToRecFN_unsafe #(.inExpWidth(5), .inSigWidth(11), .outExpWidth(8), .outSigWidth(24)) b1_16to32 (.in(b1_r16), .out(b1_r32w));
    wire [32:0] a1_rec = (fmt == FMT_F16) ? a1_r32w : a1_r32n;
    wire [32:0] b1_rec = (fmt == FMT_F16) ? b1_r32w : b1_r32n;

    // Stage: recode

    localparam CTL_W = 2 + 1 + 1 + 1 + 64; // fmt, is_x2, ftz, byp, byp_val

    wire [1:0]  c_fmt;
    wire        c_x2, c_ftz, c_byp;
    wire [63:0] c_byp_val;
    wire [64:0] c_a0, c_b0;
    wire [32:0] c_a1, c_b1;

    VX_pipe_register #(
        .DATAW (CTL_W + 2 * 65 + 2 * 33),
        .DEPTH (D_REC)
    ) rec_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({fmt,   is_x2, ftz,   byp,   byp_val,   a0_rec, b0_rec, a1_rec, b1_rec}),
        .data_out ({c_fmt, c_x2,  c_ftz, c_byp, c_byp_val, c_a0,   c_b0,   c_a1,   c_b1})
    );

    // Raw sums

    wire        s0_inv, s0_nan, s0_inf, s0_zero, s0_sign;
    wire signed [12:0] s0_sexp;
    wire [55:0] s0_sig;
    addRecFNToRaw #(.expWidth(11), .sigWidth(53)) add0 (
        .control      (control),
        .subOp        (1'b0),
        .a            (c_a0),
        .b            (c_b0),
        .roundingMode (rnd_mode),
        .invalidExc   (s0_inv),
        .out_isNaN    (s0_nan),
        .out_isInf    (s0_inf),
        .out_isZero   (s0_zero),
        .out_sign     (s0_sign),
        .out_sExp     (s0_sexp),
        .out_sig      (s0_sig)
    );

    wire        s1_inv, s1_nan, s1_inf, s1_zero, s1_sign;
    wire signed [9:0] s1_sexp;
    wire [26:0] s1_sig;
    addRecFNToRaw #(.expWidth(8), .sigWidth(24)) add1 (
        .control      (control),
        .subOp        (1'b0),
        .a            (c_a1),
        .b            (c_b1),
        .roundingMode (rnd_mode),
        .invalidExc   (s1_inv),
        .out_isNaN    (s1_nan),
        .out_isInf    (s1_inf),
        .out_isZero   (s1_zero),
        .out_sign     (s1_sign),
        .out_sExp     (s1_sexp),
        .out_sig      (s1_sig)
    );

    // Stage: add

    localparam RAW0_W = 5 + 13 + 56;
    localparam RAW1_W = 5 + 10 + 27;

    wire [1:0]  a_fmt;
    wire        a_x2, a_ftz, a_byp;
    wire [63:0] a_byp_val;
    wire        a0_inv, a0_nan, a0_inf, a0_zero, a0_sign;
    wire signed [12:0] a0_sexp;
    wire [55:0] a0_sig;
    wire        a1_inv, a1_nan, a1_inf, a1_zero, a1_sign;
    wire signed [9:0] a1_sexp;
    wire [26:0] a1_sig;

    VX_pipe_register #(
        .DATAW (CTL_W + RAW0_W + RAW1_W),
        .DEPTH (D_ADD)
    ) add_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({c_fmt, c_x2, c_ftz, c_byp, c_byp_val,
                    s0_inv, s0_nan, s0_inf, s0_zero, s0_sign, s0_sexp, s0_sig,
                    s1_inv, s1_nan, s1_inf, s1_zero, s1_sign, s1_sexp, s1_sig}),
        .data_out ({a_fmt, a_x2, a_ftz, a_byp, a_byp_val,
                    a0_inv, a0_nan, a0_inf, a0_zero, a0_sign, a0_sexp, a0_sig,
                    a1_inv, a1_nan, a1_inf, a1_zero, a1_sign, a1_sexp, a1_sig})
    );

    // Rounding, one rounder per output format

    wire [64:0] r0_rec64;
    wire [32:0] r0_rec32, r1_rec32;
    wire [16:0] r0_rec16, r0_recbf, r1_rec16, r1_recbf;

    roundAnyRawFNToRecFN #(
        .inExpWidth (11), .inSigWidth (55), .outExpWidth (11), .outSigWidth (53), .options (RND_OPT)
    ) rnd0_f64 (
        .control      (control),
        .invalidExc   (a0_inv),
        .infiniteExc  (1'b0),
        .in_isNaN     (a0_nan),
        .in_isInf     (a0_inf),
        .in_isZero    (a0_zero),
        .in_sign      (a0_sign),
        .in_sExp      (a0_sexp),
        .in_sig       (a0_sig),
        .roundingMode (rnd_mode),
        .out          (r0_rec64),
        `UNUSED_PIN   (exceptionFlags)
    );
    roundAnyRawFNToRecFN #(
        .inExpWidth (11), .inSigWidth (55), .outExpWidth (8), .outSigWidth (24), .options (RND_OPT)
    ) rnd0_f32 (
        .control      (control),
        .invalidExc   (a0_inv),
        .infiniteExc  (1'b0),
        .in_isNaN     (a0_nan),
        .in_isInf     (a0_inf),
        .in_isZero    (a0_zero),
        .in_sign      (a0_sign),
        .in_sExp      (a0_sexp),
        .in_sig       (a0_sig),
        .roundingMode (rnd_mode),
        .out          (r0_rec32),
        `UNUSED_PIN   (exceptionFlags)
    );
    roundAnyRawFNToRecFN #(
        .inExpWidth (11), .inSigWidth (55), .outExpWidth (5), .outSigWidth (11), .options (RND_OPT)
    ) rnd0_f16 (
        .control      (control),
        .invalidExc   (a0_inv),
        .infiniteExc  (1'b0),
        .in_isNaN     (a0_nan),
        .in_isInf     (a0_inf),
        .in_isZero    (a0_zero),
        .in_sign      (a0_sign),
        .in_sExp      (a0_sexp),
        .in_sig       (a0_sig),
        .roundingMode (rnd_mode),
        .out          (r0_rec16),
        `UNUSED_PIN   (exceptionFlags)
    );
    roundAnyRawFNToRecFN #(
        .inExpWidth (11), .inSigWidth (55), .outExpWidth (8), .outSigWidth (8), .options (RND_OPT)
    ) rnd0_bf16 (
        .control      (control),
        .invalidExc   (a0_inv),
        .infiniteExc  (1'b0),
        .in_isNaN     (a0_nan),
        .in_isInf     (a0_inf),
        .in_isZero    (a0_zero),
        .in_sign      (a0_sign),
        .in_sExp      (a0_sexp),
        .in_sig       (a0_sig),
        .roundingMode (rnd_mode),
        .out          (r0_recbf),
        `UNUSED_PIN   (exceptionFlags)
    );
    roundAnyRawFNToRecFN #(
        .inExpWidth (8), .inSigWidth (26), .outExpWidth (8), .outSigWidth (24), .options (RND_OPT)
    ) rnd1_f32 (
        .control      (control),
        .invalidExc   (a1_inv),
        .infiniteExc  (1'b0),
        .in_isNaN     (a1_nan),
        .in_isInf     (a1_inf),
        .in_isZero    (a1_zero),
        .in_sign      (a1_sign),
        .in_sExp      (a1_sexp),
        .in_sig       (a1_sig),
        .roundingMode (rnd_mode),
        .out          (r1_rec32),
        `UNUSED_PIN   (exceptionFlags)
    );
    roundAnyRawFNToRecFN #(
        .inExpWidth (8), .inSigWidth (26), .outExpWidth (5), .outSigWidth (11), .options (RND_OPT)
    ) rnd1_f16 (
        .control      (control),
        .invalidExc   (a1_inv),
        .infiniteExc  (1'b0),
        .in_isNaN     (a1_nan),
        .in_isInf     (a1_inf),
        .in_isZero    (a1_zero),
        .in_sign      (a1_sign),
        .in_sExp      (a1_sexp),
        .in_sig       (a1_sig),
        .roundingMode (rnd_mode),
        .out          (r1_rec16),
        `UNUSED_PIN   (exceptionFlags)
    );
    roundAnyRawFNToRecFN #(
        .inExpWidth (8), .inSigWidth (26), .outExpWidth (8), .outSigWidth (8), .options (RND_OPT)
    ) rnd1_bf16 (
        .control      (control),
        .invalidExc   (a1_inv),
        .infiniteExc  (1'b0),
        .in_isNaN     (a1_nan),
        .in_isInf     (a1_inf),
        .in_isZero    (a1_zero),
        .in_sign      (a1_sign),
        .in_sExp      (a1_sexp),
        .in_sig       (a1_sig),
        .roundingMode (rnd_mode),
        .out          (r1_recbf),
        `UNUSED_PIN   (exceptionFlags)
    );

    wire r0_is_nan = a0_nan || a0_inv;
    wire r1_is_nan = a1_nan || a1_inv;

    // Stage: round

    localparam REC_W = 65 + 33 + 17 + 17 + 33 + 17 + 17;

    wire [1:0]  n_fmt;
    wire        n_x2, n_ftz, n_byp, n0_nan, n1_nan;
    wire [63:0] n_byp_val;
    wire [64:0] n0_rec64;
    wire [32:0] n0_rec32, n1_rec32;
    wire [16:0] n0_rec16, n0_recbf, n1_rec16, n1_recbf;

    VX_pipe_register #(
        .DATAW (CTL_W + 2 + REC_W),
        .DEPTH (D_RND)
    ) rnd_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({a_fmt, a_x2, a_ftz, a_byp, a_byp_val, r0_is_nan, r1_is_nan,
                    r0_rec64, r0_rec32, r0_rec16, r0_recbf, r1_rec32, r1_rec16, r1_recbf}),
        .data_out ({n_fmt, n_x2, n_ftz, n_byp, n_byp_val, n0_nan, n1_nan,
                    n0_rec64, n0_rec32, n0_rec16, n0_recbf, n1_rec32, n1_rec16, n1_recbf})
    );

    // Recoded to IEEE, the op's format selected

    wire [63:0] r0_f64;
    wire [31:0] r0_f32, r1_f32;
    wire [15:0] r0_f16, r0_bf16, r1_f16, r1_bf16;
    recFNToFN #(.expWidth(11), .sigWidth(53)) to0_f64  (.in(n0_rec64), .out(r0_f64));
    recFNToFN #(.expWidth(8),  .sigWidth(24)) to0_f32  (.in(n0_rec32), .out(r0_f32));
    recFNToFN #(.expWidth(5),  .sigWidth(11)) to0_f16  (.in(n0_rec16), .out(r0_f16));
    recFNToFN #(.expWidth(8),  .sigWidth(8))  to0_bf16 (.in(n0_recbf), .out(r0_bf16));
    recFNToFN #(.expWidth(8),  .sigWidth(24)) to1_f32  (.in(n1_rec32), .out(r1_f32));
    recFNToFN #(.expWidth(5),  .sigWidth(11)) to1_f16  (.in(n1_rec16), .out(r1_f16));
    recFNToFN #(.expWidth(8),  .sigWidth(8))  to1_bf16 (.in(n1_recbf), .out(r1_bf16));

    reg [63:0] r0_sel;
    reg [31:0] r1_sel;
    always @(*) begin
        case (n_fmt)
            FMT_F16:  begin r0_sel = 64'(r0_f16);  r1_sel = 32'(r1_f16);  end
            FMT_BF16: begin r0_sel = 64'(r0_bf16); r1_sel = 32'(r1_bf16); end
            FMT_F32:  begin r0_sel = 64'(r0_f32);  r1_sel = r1_f32;       end
            default:  begin r0_sel = r0_f64;       r1_sel = r1_f32;       end
        endcase
    end

    // Stage: convert

    wire [1:0]  o_fmt;
    wire        o_x2, o_ftz, o_byp, o0_nan, o1_nan;
    wire [63:0] o_byp_val;
    wire [63:0] o0;
    wire [31:0] o1;

    VX_pipe_register #(
        .DATAW (CTL_W + 2 + 64 + 32),
        .DEPTH (D_CVT)
    ) cvt_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({n_fmt, n_x2, n_ftz, n_byp, n_byp_val, n0_nan, n1_nan, r0_sel, r1_sel}),
        .data_out ({o_fmt, o_x2, o_ftz, o_byp, o_byp_val, o0_nan, o1_nan, o0,     o1})
    );

    // Canonical NaN, FTZ of the result, packing

    reg [63:0] q0;
    reg [31:0] q1;
    always @(*) begin
        case (o_fmt)
            FMT_F16:  q0 = o0_nan ? 64'(F16_QNAN)  : o0;
            FMT_BF16: q0 = o0_nan ? 64'(BF16_QNAN) : o0;
            FMT_F32:  q0 = o0_nan ? 64'(F32_QNAN)  : (o_ftz ? 64'(f32_ftz(o0[31:0])) : o0);
            default:  q0 = o0_nan ? F64_QNAN       : o0;
        endcase
        case (o_fmt)
            FMT_F16:  q1 = o1_nan ? 32'(F16_QNAN)  : o1;
            FMT_BF16: q1 = o1_nan ? 32'(BF16_QNAN) : o1;
            default:  q1 = o1_nan ? F32_QNAN       : (o_ftz ? f32_ftz(o1) : o1);
        endcase
    end

    reg [63:0] result;
    always @(*) begin
        if (o_byp) begin
            result = o_byp_val;
        end else if (o_x2) begin
            result = (o_fmt == FMT_F32) ? {q1, q0[31:0]} : 64'({q1[15:0], q0[15:0]});
        end else begin
            result = q0;
        end
    end

    VX_pipe_register #(
        .DATAW (64),
        .DEPTH (D_OUT)
    ) out_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  (result),
        .data_out (new_word)
    );

endmodule
`endif
