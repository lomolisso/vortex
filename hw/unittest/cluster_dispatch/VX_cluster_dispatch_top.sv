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

// The chip-wide cluster admission with its launch beats flattened, the per-GPC
// CTA_DONE inputs as packed fields, no cluster-barrier releases pending, and a
// probe of the SM position functions.
module VX_cluster_dispatch_top import VX_gpu_pkg::*; #(
    localparam G = `VX_CFG_NUM_CLUSTERS
) (
    input  wire                             clk,
    input  wire                             reset,

    input  wire                             in_valid,
    output wire                             in_ready,
    input  wire [7:0]                       in_ctx_id,
    input  wire                             in_policy,
    input  wire                             in_explicit,
    input  wire [CLUSTER_SIZE_W-1:0]        in_cluster_size,
    input  wire [RANK_W-1:0]                in_cluster_rank,
    input  wire                             in_is_first,
    input  wire                             in_eop,
    input  wire [CTA_TID_WIDTH:0]           in_block_size,
    input  wire [`VX_CFG_LMEM_LOG_SIZE:0]   in_lmem_stride,

    output wire                             out_valid,
    input  wire                             out_ready,
    output wire                             out_pin,
    output wire                             out_eop,
    output wire [KMU_DEST_W-1:0]            out_dest,
    output wire [7:0]                       out_ctx_id,
    output wire [RANK_W-1:0]                out_cluster_rank,
    output wire                             out_epoch,
    output wire [GANG_ID_W-1:0]             out_gang_id,
    output wire [NW_WIDTH-1:0]              out_forced_slot,

    input  wire [G-1:0]                     done_valid,
    input  wire [G-1:0][DLMEM_CORE_W-1:0]   done_core,
    input  wire [G-1:0][NW_WIDTH-1:0]       done_slot,
    input  wire [G-1:0][GANG_ID_W-1:0]      done_gang_id,
    input  wire [G-1:0]                     done_epoch,

    output wire [G-1:0]                     alloc_valid,
    output wire [GANG_ID_W-1:0]             alloc_gang,
    output wire                             alloc_epoch,
    output wire [DLMEM_K_W-1:0]             alloc_k,
    output wire                             gang_mode,

    output wire                             busy,

    input  wire [15:0]                      pos_sm,
    output wire [7:0]                       pos_partition,
    output wire [7:0]                       pos_gpc,
    output wire [7:0]                       pos_cpc,
    output wire [7:0]                       pos_tpc_slot,
    output wire [7:0]                       pos_parity,
    output wire [7:0]                       pos_slot_dist
);
    VX_kmu_bus_if bus_in_if();
    VX_kmu_bus_if bus_out_if();

    kmu_req_t in_req;
    always @(*) begin
        in_req = '0;
        in_req.kind                             = KMU_KIND_COMPUTE;
        in_req.ctx_id                           = in_ctx_id;
        in_req.aligned_lmem_size                = in_lmem_stride;
    `ifdef VX_CFG_H100_SM_ENABLE
        in_req.smem_bytes                       = (`VX_CFG_LMEM_LOG_SIZE+1)'(`VX_CFG_SMEM_PER_SM_MAX);
    `endif
        in_req.args.compute.block_size          = in_block_size;
        in_req.args.compute.cluster_size        = in_cluster_size;
        in_req.args.compute.cluster_rank        = in_cluster_rank;
        in_req.args.compute.is_first_of_cluster = in_is_first;
        in_req.args.compute.dlmem.policy        = in_policy;
        in_req.args.compute.dlmem.explicit_cluster = in_explicit;
    end

    assign bus_in_if.valid = in_valid;
    assign bus_in_if.kind  = KMU_KIND_COMPUTE;
    assign bus_in_if.eop   = in_eop;
    assign bus_in_if.dest  = '0;
    assign bus_in_if.data  = KMU_DATAW'(in_req);
    assign bus_in_if.pin   = 1'b0;
    assign in_ready        = bus_in_if.ready;

    kmu_req_t out_req;
    assign out_req          = kmu_req_t'(bus_out_if.data);
    assign out_valid        = bus_out_if.valid;
    assign bus_out_if.ready = out_ready;
    assign out_pin          = bus_out_if.pin;
    assign out_eop          = bus_out_if.eop;
    assign out_dest         = bus_out_if.dest;
    assign out_ctx_id       = out_req.ctx_id;
    assign out_cluster_rank = out_req.args.compute.cluster_rank;
    assign out_epoch        = out_req.args.compute.dlmem.epoch;
    assign out_gang_id      = out_req.args.compute.dlmem.gang_id;
    assign out_forced_slot  = out_req.args.compute.dlmem.forced_slot;
    `UNUSED_VAR (out_req)

    dlmem_cta_done_t [G-1:0] done_data;
    for (genvar g = 0; g < G; ++g) begin : g_done
        assign done_data[g].core    = done_core[g];
        assign done_data[g].slot    = done_slot[g];
        assign done_data[g].gang_id = done_gang_id[g];
        assign done_data[g].epoch   = done_epoch[g];
        assign done_data[g].arrived = 1'b0;
    end

    VX_cta_admit #(
        .INSTANCE_ID ("cta_admit"),
        .NUM_GPCS    (G)
    ) cta_admit (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (bus_in_if),
        .bus_out_if (bus_out_if),
        .done_valid (done_valid),
        .done_data  (done_data),
        .rel_pend   ('0),
        .alloc_valid(alloc_valid),
        .alloc_gang (alloc_gang),
        .alloc_epoch(alloc_epoch),
        .alloc_k    (alloc_k),
        .gang_mode  (gang_mode),
        .busy       (busy)
    );

    wire [31:0] sm = 32'(pos_sm);
    assign pos_partition = 8'(sm_partition(sm));
    assign pos_gpc       = 8'(sm_gpc(sm));
    assign pos_cpc       = 8'(sm_cpc(sm));
    assign pos_tpc_slot  = 8'(sm_tpc_slot(sm));
    assign pos_parity    = 8'(sm_parity(sm));
    assign pos_slot_dist = 8'(sm_slot_dist(sm));

endmodule
