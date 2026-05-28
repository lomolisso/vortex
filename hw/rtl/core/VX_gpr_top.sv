// Copyright © 2019-2023. Apache 2.0.
// Hard-macro wrapper around VX_opc_unit with flat ports for PnR boundary.

`include "VX_define.vh"

`ifdef ASIC_SYNTHESIS
// Hard-macro boundary: localparams (not parameters) so DC can't uniquify on parent overrides.
module VX_gpr_top import VX_gpu_pkg::*; #(
    localparam NUM_BANKS = `NUM_GPR_BANKS,
    localparam OUT_BUF   = 3
) (
    input  wire                            clk,
    input  wire                            reset,

    input  wire                            writeback_valid,
    input  wire [$bits(writeback_t)-1:0]   writeback_data,

    input  wire                            scoreboard_valid,
    input  wire [$bits(scoreboard_t)-1:0]  scoreboard_data,
    output wire                            scoreboard_ready,

    output wire                            operands_valid,
    output wire [$bits(operands_t)-1:0]    operands_data,
    input  wire                            operands_ready
);
    VX_writeback_if  writeback_if();
    VX_scoreboard_if scoreboard_if();
    VX_operands_if   operands_if();

    assign writeback_if.valid  = writeback_valid;
    assign writeback_if.data   = writeback_data;

    assign scoreboard_if.valid = scoreboard_valid;
    assign scoreboard_if.data  = scoreboard_data;
    assign scoreboard_ready    = scoreboard_if.ready;

    assign operands_valid      = operands_if.valid;
    assign operands_data       = operands_if.data;
    assign operands_if.ready   = operands_ready;

    VX_opc_unit #(
        .NUM_BANKS (NUM_BANKS),
        .OUT_BUF   (OUT_BUF)
    ) opc (
        .clk           (clk),
        .reset         (reset),
        .writeback_if  (writeback_if),
        .scoreboard_if (scoreboard_if),
        .operands_if   (operands_if)
    );

endmodule
`endif
