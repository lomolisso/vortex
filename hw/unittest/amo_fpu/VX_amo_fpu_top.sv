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

// The float atomic unit of the last-level cache.
module VX_amo_fpu_top import VX_gpu_pkg::*; (
    input  wire         clk,
    input  wire         reset,
    input  wire [7:0]   op,
    input  wire [1:0]   width,
    input  wire [63:0]  old_word,
    input  wire [63:0]  rhs,
    output wire [63:0]  new_word,
    output wire [7:0]   cfg_latency,
    output wire [7:0]   cfg_amo_op_width
);
    assign cfg_latency      = 8'(`VX_CFG_FPAMO_LATENCY);
    assign cfg_amo_op_width = 8'($bits(amo_op_e));

    wire [$bits(amo_op_e)-1:0] op_w = op[$bits(amo_op_e)-1:0];
    `UNUSED_VAR (op[7:$bits(amo_op_e)])

    VX_amo_fpu fpu (
        .clk      (clk),
        .reset    (reset),
        .op       (amo_op_e'(op_w)),
        .width    (width),
        .old_word (old_word),
        .rhs      (rhs),
        .new_word (new_word)
    );

endmodule
