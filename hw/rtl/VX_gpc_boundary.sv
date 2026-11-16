// Copyright © 2019-2026
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

// A memory bus that leaves a GPC: in H100 builds SimX runs the GPC as its own
// execution domain and registers the crossing for VX_CFG_DOMAIN_BOUNDARY_DELAY
// cycles each way (exec_domains.h, ExecDomain::stage), so the RTL carries the
// same stages. Other builds pass the bus through.
module VX_gpc_boundary import VX_gpu_pkg::*; #(
    parameter DATA_SIZE  = 1,
    parameter TAG_WIDTH  = 1,
    parameter ADDR_WIDTH = (`VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE)),
    parameter ATTR_WIDTH = MEM_ATTR_WIDTH
) (
    input wire              clk,
    input wire              reset,
    VX_mem_bus_if.slave     bus_in_if,
    VX_mem_bus_if.master    bus_out_if
);
    `STATIC_ASSERT((`VX_CFG_DOMAIN_BOUNDARY_DELAY >= 1), ("VX_CFG_DOMAIN_BOUNDARY_DELAY must be at least one cycle"));

    localparam STAGES = `VX_CFG_H100_ORG_ENABLED ? `VX_CFG_DOMAIN_BOUNDARY_DELAY : 0;

    VX_mem_bus_if #(
        .DATA_SIZE  (DATA_SIZE),
        .ATTR_WIDTH (ATTR_WIDTH),
        .TAG_WIDTH  (TAG_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) stage_if[STAGES + 1]();

    `ASSIGN_VX_MEM_BUS_IF (stage_if[0], bus_in_if);

    if (STAGES == 0) begin : g_passthru
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
    end

    for (genvar i = 0; i < STAGES; ++i) begin : g_stages
        VX_mem_bus_slice #(
            .DATA_SIZE   (DATA_SIZE),
            .TAG_WIDTH   (TAG_WIDTH),
            .REQ_OUT_BUF (3),
            .RSP_OUT_BUF (3),
            .ADDR_WIDTH  (ADDR_WIDTH),
            .ATTR_WIDTH  (ATTR_WIDTH)
        ) stage (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (stage_if[i]),
            .bus_out_if (stage_if[i + 1])
        );
    end

    `ASSIGN_VX_MEM_BUS_IF (bus_out_if, stage_if[STAGES]);

endmodule
