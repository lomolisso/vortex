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

`ifndef TB_CORE_ID
`define TB_CORE_ID 0
`endif

module VX_cta_dispatch_top import VX_gpu_pkg::*;
(
    input wire                          clk,
    input wire                          reset,

    // KMU bus input (struct fields flattened for C++ testbench driving)
    input  wire                         task_in_valid,
    output wire                         task_in_ready,
    input  wire [PC_BITS-1:0]           in_PC,
    input  wire [31:0]                  in_cta_id,
    input  wire [31:0]                  in_block_idx_x,
    input  wire [31:0]                  in_block_idx_y,
    input  wire [31:0]                  in_block_idx_z,
    input  wire [CTA_TID_WIDTH:0]       in_block_dim_x,
    input  wire [CTA_TID_WIDTH:0]       in_block_dim_y,
    input  wire [CTA_TID_WIDTH:0]       in_block_dim_z,
    input  wire [31:0]                  in_grid_dim_x,
    input  wire [31:0]                  in_grid_dim_y,
    input  wire [31:0]                  in_grid_dim_z,
    input  wire [`VX_CFG_MEM_ADDR_WIDTH-1:0]   in_param,
    input  wire [`VX_CFG_LMEM_LOG_SIZE:0]      in_lmem_size,
    input  wire [CTA_TID_WIDTH:0]       in_block_size,
    input  wire [CTA_TID_WIDTH-1:0]     in_warp_step_x,
    input  wire [CTA_TID_WIDTH-1:0]     in_warp_step_y,
    input  wire [CTA_TID_WIDTH-1:0]     in_warp_step_z,
    input  wire [7:0]                   in_ctx_id,
    input  wire [31:0]                  in_cluster_size,
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    input  wire [31:0]                  in_cluster_rank,
`endif
    input  wire                         in_is_first,
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // the gang unit's placement fields of a pinned (SPREAD) beat
    input  wire                         in_policy,
    input  wire                         in_pin,
    input  wire [NW_WIDTH-1:0]          in_forced_slot,
    input  wire [GANG_ID_W-1:0]         in_gang_id,
    input  wire                         in_epoch,
    input  wire [$bits(dlmem_row_t)-1:0] in_row,
`endif

    input  wire [`VX_CFG_NUM_WARPS-1:0]        active_warps,
    input  wire                         warp_done,
    input  wire [NW_WIDTH-1:0]          warp_done_wid,

    // Outputs
    output wire                         cta_fire,
    output wire [NW_WIDTH-1:0]          cta_wid,

    // CTA slot (the CTA_ID CSR value) of a launched warp
    input  wire [NW_WIDTH-1:0]          schedule_wid,
    output wire [NCTA_WIDTH-1:0]        schedule_cta_id,

    // CTA-CSR read-back of a slot, one cycle after its index
    input  wire [NW_WIDTH-1:0]          csr_rd_wid,
    input  wire [NCTA_WIDTH-1:0]        csr_rd_cta_id,
    output wire [31:0]                  rd_block_idx_x,
    output wire [31:0]                  rd_cluster_size,
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    output wire [31:0]                  rd_cluster_rank,
    output wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] rd_lmem_addr,

    // CTA_DONE events
    output wire                         cta_done_valid,
    input  wire                         cta_done_ready,
    output wire [DLMEM_CORE_W-1:0]      cta_done_core,
    output wire [NW_WIDTH-1:0]          cta_done_slot,
    output wire [GANG_ID_W-1:0]         cta_done_gang,
    output wire                         cta_done_epoch,

    // peer-row writes
    output wire                         peer_wr_valid,
    output wire [NW_WIDTH-1:0]          peer_wr_slot,
    output wire [$bits(dlmem_row_t)-1:0] peer_wr_row,
    output wire [`VX_CFG_LMEM_LOG_SIZE:0] peer_wr_stride,

    // cluster-barrier slot records
    output wire                         cbar_wr_valid,
    output wire [NW_WIDTH-1:0]          cbar_wr_slot,
    output wire                         cbar_wr_gang,
    output wire [GANG_ID_W-1:0]         cbar_wr_gang_id,
    output wire                         cbar_wr_epoch,
    output wire [NW_WIDTH-1:0]          cbar_wr_base,
    output wire [7:0]                   cbar_wr_k,

    output wire                         busy_out,
`endif
    output wire [31:0]                  rd_cta_id
);

    VX_kmu_bus_if kmu_bus();

    // The KMU is not instantiated in this unit test, so the bench supplies the
    // values the dispatcher expects precomputed on the bus:
    //   aligned_lmem_size  — in_lmem_size rounded up to MEM_BLOCK_SIZE.
    //   ctx_id, cluster_size, is_first_of_cluster — the KMU's launch id and
    //   cluster walk.
    wire [`VX_CFG_LMEM_LOG_SIZE:0] aligned_lmem_size_w =
        ((`VX_CFG_LMEM_LOG_SIZE+1)'(in_lmem_size) + (`VX_CFG_LMEM_LOG_SIZE+1)'(`VX_CFG_MEM_BLOCK_SIZE-1))
        & ~((`VX_CFG_LMEM_LOG_SIZE+1)'(`VX_CFG_MEM_BLOCK_SIZE-1));

    // The bus carries the launch as a flat data beat; the dispatcher casts it
    // back to kmu_req_t. CTA ids are allocated inside the dispatcher.
    kmu_req_t kmu_req;
    always_comb begin
        kmu_req = '0;
        kmu_req.kind              = KMU_KIND_COMPUTE;
        kmu_req.PC                = in_PC;
        kmu_req.entry             = in_PC;
        kmu_req.param             = in_param;
        kmu_req.ctx_id            = in_ctx_id;
        kmu_req.aligned_lmem_size = aligned_lmem_size_w;
        kmu_req.args.compute.grid_dim[0]  = in_grid_dim_x;
        kmu_req.args.compute.grid_dim[1]  = in_grid_dim_y;
        kmu_req.args.compute.grid_dim[2]  = in_grid_dim_z;
        kmu_req.args.compute.block_idx[0] = in_block_idx_x;
        kmu_req.args.compute.block_idx[1] = in_block_idx_y;
        kmu_req.args.compute.block_idx[2] = in_block_idx_z;
        kmu_req.args.compute.block_dim[0] = in_block_dim_x;
        kmu_req.args.compute.block_dim[1] = in_block_dim_y;
        kmu_req.args.compute.block_dim[2] = in_block_dim_z;
        kmu_req.args.compute.block_size   = in_block_size;
        kmu_req.args.compute.warp_step[0] = in_warp_step_x;
        kmu_req.args.compute.warp_step[1] = in_warp_step_y;
        kmu_req.args.compute.warp_step[2] = in_warp_step_z;
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        kmu_req.args.compute.cluster_size = CLUSTER_SIZE_W'(in_cluster_size);
        kmu_req.args.compute.cluster_rank = RANK_W'(in_cluster_rank);
    `else
        kmu_req.args.compute.cluster_size = (NW_WIDTH+1)'(in_cluster_size);
    `endif
        kmu_req.args.compute.is_first_of_cluster = in_is_first;
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        kmu_req.args.compute.dlmem.policy      = in_policy;
        kmu_req.args.compute.dlmem.forced_slot = in_forced_slot;
        kmu_req.args.compute.dlmem.gang_id     = in_gang_id;
        kmu_req.args.compute.dlmem.epoch       = in_epoch;
        kmu_req.args.compute.dlmem.row         = dlmem_row_t'(in_row);
    `endif
    end
    `UNUSED_VAR (in_cluster_size)
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    `UNUSED_VAR (in_cluster_rank)
`endif

    assign kmu_bus.valid = task_in_valid;
    assign kmu_bus.kind  = KMU_KIND_COMPUTE;
    assign kmu_bus.eop   = 1'b1;
    assign kmu_bus.dest  = '0;
    assign kmu_bus.data  = kmu_req;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    assign kmu_bus.pin   = in_pin;
`endif
    assign task_in_ready = kmu_bus.ready;
    `UNUSED_VAR (in_cta_id)

    wire [PC_BITS-1:0]      cta_PC;
    wire [`VX_CFG_NUM_THREADS-1:0] cta_tmask;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] cta_param;
    wire                    cta_init;
    wire                    busy;

    cta_csrs_t              cta_rd_csrs;
    cta_lane_t [`VX_CFG_NUM_THREADS-1:0] cta_rd_lane;

    assign rd_block_idx_x  = cta_rd_csrs.block_idx[0];
    assign rd_cluster_size = cta_rd_csrs.cluster_size;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    assign rd_cluster_rank = 32'(cta_rd_csrs.cluster_rank);
    assign rd_lmem_addr    = cta_rd_csrs.lmem_addr;
    assign busy_out        = busy;
`endif
    assign rd_cta_id       = 32'(cta_rd_csrs.cta_id);

    `UNUSED_VAR (cta_PC)
    `UNUSED_VAR (cta_tmask)
    `UNUSED_VAR (cta_param)
    `UNUSED_VAR (cta_init)
`ifndef VX_CFG_EXT_DLMEM_ENABLE
    `UNUSED_VAR (busy)
`endif
    `UNUSED_VAR (cta_rd_csrs)
    `UNUSED_VAR (cta_rd_lane)

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    dlmem_cta_done_t cta_done_data;
    dlmem_row_t      peer_wr_row_s;
    assign cta_done_core  = cta_done_data.core;
    assign cta_done_slot  = cta_done_data.slot;
    assign cta_done_gang  = cta_done_data.gang_id;
    assign cta_done_epoch = cta_done_data.epoch;
    assign peer_wr_row    = peer_wr_row_s;

    cbar_slot_t cbar_wr_info;
    assign cbar_wr_gang    = cbar_wr_info.gang;
    assign cbar_wr_gang_id = cbar_wr_info.gang_id;
    assign cbar_wr_epoch   = cbar_wr_info.epoch;
    assign cbar_wr_base    = cbar_wr_info.base;
    assign cbar_wr_k       = 8'(cbar_wr_info.k);
`endif

    VX_cta_dispatch #(
        .INSTANCE_ID ("cta_dispatch")
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        , .CORE_ID   (`TB_CORE_ID)
    `endif
    ) cta_dispatch (
        .clk           (clk),
        .reset         (reset),
        .kmu_bus_if    (kmu_bus),
        .active_warps  (active_warps),
        .warp_done     (warp_done),
        .warp_done_wid (warp_done_wid),
        .cta_fire      (cta_fire),
        .cta_wid       (cta_wid),
        .cta_PC        (cta_PC),
        .cta_tmask     (cta_tmask),
        .cta_param     (cta_param),
        .cta_init      (cta_init),
        .csr_rd_wid    (csr_rd_wid),
        .csr_rd_cta_id (csr_rd_cta_id),
        .cta_rd_csrs   (cta_rd_csrs),
        .cta_rd_lane   (cta_rd_lane),
        .schedule_wid  (schedule_wid),
        .schedule_cta_id (schedule_cta_id),
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        .cta_done_valid (cta_done_valid),
        .cta_done_data  (cta_done_data),
        .cta_done_ready (cta_done_ready),
        .peer_wr_valid  (peer_wr_valid),
        .peer_wr_slot   (peer_wr_slot),
        .peer_wr_row    (peer_wr_row_s),
        .peer_wr_stride (peer_wr_stride),
        .cbar_wr_valid  (cbar_wr_valid),
        .cbar_wr_slot   (cbar_wr_slot),
        .cbar_wr_info   (cbar_wr_info),
    `endif
        .busy          (busy)
    );

endmodule
