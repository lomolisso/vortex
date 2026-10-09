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

module VX_kmu_top import VX_gpu_pkg::*;
(
    input wire                              clk,
    input wire                              reset,

    // DCR request interface
    input  wire                             dcr_req_valid,
    input  wire                             dcr_req_rw,
    input  wire [VX_DCR_ADDR_WIDTH-1:0]     dcr_req_addr,
    input  wire [VX_DCR_DATA_WIDTH-1:0]     dcr_req_data,

    // Start trigger: pulse high for one cycle to begin CTA dispatch
    input  wire                             start,
    output wire                             busy,

    // Launch bus, with the compute descriptor unpacked for the bench
    input  wire                             bus_ready,
    output wire                             bus_valid,
    output wire                             bus_eop,
    output wire [7:0]                       bus_ctx_id,
    output wire [31:0]                      bus_block_idx_x,
    output wire [31:0]                      bus_block_idx_y,
    output wire [31:0]                      bus_block_idx_z,
    output wire [31:0]                      bus_cluster_size,
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    output wire [31:0]                      bus_cluster_rank,
    output wire                             bus_policy,
    output wire                             bus_pin,
    // any gang-unit field set (the KMU leaves them all to the gang)
    output wire                             bus_gang_fields,
`endif
    output wire                             bus_is_first
);

    VX_kmu_bus_if kmu_bus_if();

    kmu_req_t kmu_req;
    assign kmu_req = kmu_req_t'(kmu_bus_if.data);

    assign kmu_bus_if.ready = bus_ready;
    assign bus_valid        = kmu_bus_if.valid;
    assign bus_eop          = kmu_bus_if.eop;
    assign bus_ctx_id       = kmu_req.ctx_id;
    assign bus_block_idx_x  = kmu_req.args.compute.block_idx[0];
    assign bus_block_idx_y  = kmu_req.args.compute.block_idx[1];
    assign bus_block_idx_z  = kmu_req.args.compute.block_idx[2];
    assign bus_cluster_size = 32'(kmu_req.args.compute.cluster_size);
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    assign bus_cluster_rank = 32'(kmu_req.args.compute.cluster_rank);
    assign bus_policy       = kmu_req.args.compute.dlmem.policy;
    assign bus_pin          = kmu_bus_if.pin;
    kmu_dlmem_t gang_fields;
    always @(*) begin
        gang_fields = kmu_req.args.compute.dlmem;
        gang_fields.policy = 1'b0;
    end
    assign bus_gang_fields  = (|gang_fields);
`endif
    assign bus_is_first     = kmu_req.args.compute.is_first_of_cluster;

    `UNUSED_VAR (kmu_bus_if.kind)
    `UNUSED_VAR (kmu_bus_if.dest)
    `UNUSED_VAR (kmu_req)

    VX_kmu kmu (
        .clk           (clk),
        .reset         (reset),
        .dcr_req_valid (dcr_req_valid),
        .dcr_req_rw    (dcr_req_rw),
        .dcr_req_addr  (dcr_req_addr),
        .dcr_req_data  (dcr_req_data),
        .start         (start),
        .busy          (busy),
        .kmu_bus_if    (kmu_bus_if)
    );

endmodule
