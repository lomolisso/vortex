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

module VX_alu_uniform_top import VX_gpu_pkg::*; #(
    parameter NUM_LANES = `VX_CFG_NUM_THREADS
) (
    input wire          clk,
    input wire          reset,

    input wire          execute_valid,
    input alu_execute_t execute_data,
    output wire         execute_ready,

    output wire         result_valid,
    output alu_result_t result_data,
    input wire          result_ready
);
    VX_execute_if #(
        .data_t (alu_execute_t)
    ) execute_if();

    VX_result_if #(
        .data_t (alu_result_t)
    ) result_if();

    assign execute_if.valid = execute_valid;
    assign execute_if.data  = execute_data;
    assign execute_ready    = execute_if.ready;

    assign result_valid     = result_if.valid;
    assign result_data      = result_if.data;
    assign result_if.ready  = result_ready;

    VX_alu_uniform #(
        .INSTANCE_ID ("alu-uniform"),
        .NUM_LANES   (NUM_LANES)
    ) alu_uniform (
        .clk        (clk),
        .reset      (reset),
        .execute_if (execute_if),
        .result_if  (result_if)
    );

endmodule
