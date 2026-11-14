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

// Flat-port wrapper around the WGMMA CTA gate.

module VX_tcu_lockstep_top import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter BLOCK_SIZE = 4
) (
    input  wire                                  clk,
    input  wire                                  reset,
    input  wire [BLOCK_SIZE-1:0]                 is_wgmma,
    input  wire [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0] new_cta,
    input  wire [BLOCK_SIZE-1:0]                 exec_fire,
    input  wire [BLOCK_SIZE-1:0]                 is_first_uop,
    input  wire [BLOCK_SIZE-1:0]                 is_last_uop,
    output wire [BLOCK_SIZE-1:0]                 cta_conflict
);
    VX_tcu_lockstep #(
        .INSTANCE_ID ("lockstep"),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) lockstep (
        .clk            (clk),
        .reset          (reset),
        .is_wgmma_b     (is_wgmma),
        .new_cta_b      (new_cta),
        .exec_fire_b    (exec_fire),
        .is_first_uop_b (is_first_uop),
        .is_last_uop_b  (is_last_uop),
        .cta_conflict   (cta_conflict)
    );
endmodule
