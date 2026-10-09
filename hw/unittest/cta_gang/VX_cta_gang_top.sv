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

// One hardware cluster's gang unit with its launch beats flattened. The input
// payload is a raw fill with the fields the gang reads overlaid on it, so every
// payload bit is driven (the bench checks them bit for bit and synthesis keeps
// the whole beat path).
module VX_cta_gang_top import VX_gpu_pkg::*; #(
    parameter CLUSTER_ID = 0
) (
    input  wire                             clk,
    input  wire                             reset,

    input  wire                             in_valid,
    output wire                             in_ready,
    input  wire [KMU_DATAW-1:0]             in_fill,
    input  wire [7:0]                       in_ctx_id,
    input  wire                             in_policy,
    input  wire [CLUSTER_SIZE_W-1:0]        in_cluster_size,
    input  wire [RANK_W-1:0]                in_cluster_rank,
    input  wire                             in_is_first,
    input  wire                             in_eop,
    input  wire [CTA_TID_WIDTH:0]           in_block_size,
    input  wire [`VX_CFG_LMEM_LOG_SIZE:0]   in_lmem_stride,
    // the beat as presented, with its dlmem field cleared
    output wire [KMU_DATAW-1:0]             in_word,

    output wire                             out_valid,
    input  wire                             out_ready,
    output wire                             out_pin,
    output wire                             out_kind,
    output wire                             out_eop,
    output wire [KMU_DEST_W-1:0]            out_dest,
    // the beat as delivered, with its dlmem field cleared
    output wire [KMU_DATAW-1:0]             out_word,
    output wire                             out_policy,
    output wire                             out_epoch,
    output wire [GANG_ID_W-1:0]             out_gang_id,
    output wire [NW_WIDTH-1:0]              out_forced_slot,
    output wire [$bits(dlmem_row_t)-1:0]    out_row,

    // events: a CTA_DONE, or a cluster-barrier arrival of one block (CBAR)
    input  wire                             evt_valid,
    input  wire [1:0]                       evt_kind,
    input  wire [DLMEM_CORE_W-1:0]          evt_core,
    input  wire [NW_WIDTH-1:0]              evt_slot,
    input  wire [GANG_ID_W-1:0]             evt_gang_id,
    input  wire                             evt_epoch,
    input  wire [NC_WIDTH-1:0]              evt_size_m1,

    output wire                             rel_valid,
    input  wire                             rel_ready,
    output wire [GANG_ID_W-1:0]             rel_gang,
    output wire                             rel_epoch,

    output wire                             busy,

    // payload width, so the bench masks the unused bits of the top word
    output wire [15:0]                      cfg_kmu_dataw
);
    assign cfg_kmu_dataw = 16'(KMU_DATAW);

    VX_kmu_bus_if bus_in_if();
    VX_kmu_bus_if bus_out_if();

    kmu_req_t in_req;
    always @(*) begin
        in_req = kmu_req_t'(in_fill);
        in_req.kind                             = KMU_KIND_COMPUTE;
        in_req.ctx_id                           = in_ctx_id;
        in_req.aligned_lmem_size                = in_lmem_stride;
        in_req.args.compute.block_size          = in_block_size;
        in_req.args.compute.cluster_size        = in_cluster_size;
        in_req.args.compute.cluster_rank        = in_cluster_rank;
        in_req.args.compute.is_first_of_cluster = in_is_first;
        in_req.args.compute.dlmem               = '0;
        in_req.args.compute.dlmem.policy        = in_policy;
    end

    kmu_req_t in_req_bare;
    always @(*) begin
        in_req_bare = in_req;
        in_req_bare.args.compute.dlmem = '0;
    end
    assign in_word = KMU_DATAW'(in_req_bare);

    assign bus_in_if.valid = in_valid;
    assign bus_in_if.kind  = KMU_KIND_COMPUTE;
    assign bus_in_if.eop   = in_eop;
    assign bus_in_if.dest  = '0;
    assign bus_in_if.data  = KMU_DATAW'(in_req);
    assign bus_in_if.pin   = 1'b0;
    assign in_ready        = bus_in_if.ready;

    kmu_req_t out_req, out_req_bare;
    assign out_req = kmu_req_t'(bus_out_if.data);
    always @(*) begin
        out_req_bare = out_req;
        out_req_bare.args.compute.dlmem = '0;
    end

    assign out_valid       = bus_out_if.valid;
    assign bus_out_if.ready = out_ready;
    assign out_pin         = bus_out_if.pin;
    assign out_kind        = bus_out_if.kind;
    assign out_eop         = bus_out_if.eop;
    assign out_dest        = bus_out_if.dest;
    assign out_word        = KMU_DATAW'(out_req_bare);
    assign out_policy      = out_req.args.compute.dlmem.policy;
    assign out_epoch       = out_req.args.compute.dlmem.epoch;
    assign out_gang_id     = out_req.args.compute.dlmem.gang_id;
    assign out_forced_slot = out_req.args.compute.dlmem.forced_slot;
    assign out_row         = out_req.args.compute.dlmem.row;

    dlmem_cta_done_t evt_data;
    assign evt_data.core    = evt_core;
    assign evt_data.slot    = evt_slot;
    assign evt_data.gang_id = evt_gang_id;
    assign evt_data.epoch   = evt_epoch;

    VX_cta_gang #(
        .INSTANCE_ID ("cta_gang"),
        .CLUSTER_ID  (CLUSTER_ID)
    ) cta_gang (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (bus_in_if),
        .bus_out_if (bus_out_if),
        .evt_valid  (evt_valid),
        .evt_kind   (evt_kind),
        .evt_data   (evt_data),
        .evt_size_m1(evt_size_m1),
        .rel_valid  (rel_valid),
        .rel_gang   (rel_gang),
        .rel_epoch  (rel_epoch),
        .rel_ready  (rel_ready),
        .busy       (busy)
    );

endmodule
