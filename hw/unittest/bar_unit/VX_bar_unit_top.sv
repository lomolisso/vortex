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

// One hardware cluster's barrier units, behind the socket and cluster request
// arbiters and the cluster's global barrier unit, wired as VX_socket and
// VX_cluster wire them. Barrier requests, slot records and CTA_DONE events come
// in per core; the gang unit's event port comes out and its cluster-barrier
// release port goes in (the bench models the gang unit).
module VX_bar_unit_top import VX_gpu_pkg::*; (
    input  wire                                     clk,
    input  wire                                     reset,

    input  wire [DLMEM_NCC-1:0]                     req_valid,
    input  wire [DLMEM_NCC-1:0][NW_WIDTH-1:0]       req_wid,
    input  wire [DLMEM_NCC-1:0][NB_WIDTH-1:0]       req_id,
    input  wire [DLMEM_NCC-1:0]                     req_is_global,
    input  wire [DLMEM_NCC-1:0]                     req_is_arrive,
    input  wire [DLMEM_NCC-1:0]                     req_is_sync,
    input  wire [DLMEM_NCC-1:0][BAR_SIZE_W-1:0]     req_size_m1,
    input  wire [DLMEM_NCC-1:0]                     req_is_cluster,
    input  wire [DLMEM_NCC-1:0]                     req_phase,
    input  wire [DLMEM_NCC-1:0][BAR_ADDR_W-1:0]     read_addr,
    input  wire [DLMEM_NCC-1:0]                     read_cluster,
    output wire [DLMEM_NCC-1:0]                     read_phase,

    input  wire [DLMEM_NCC-1:0]                     cbar_wr_valid,
    input  wire [DLMEM_NCC-1:0][NW_WIDTH-1:0]       cbar_wr_slot,
    input  wire [DLMEM_NCC-1:0]                     cbar_wr_gang,
    input  wire [DLMEM_NCC-1:0][GANG_ID_W-1:0]      cbar_wr_gang_id,
    input  wire [DLMEM_NCC-1:0]                     cbar_wr_epoch,
    input  wire [DLMEM_NCC-1:0][NW_WIDTH-1:0]       cbar_wr_base,
    input  wire [DLMEM_NCC-1:0][7:0]                cbar_wr_k,

    input  wire [DLMEM_NCC-1:0][`VX_CFG_NUM_WARPS-1:0] active_warps,
    output wire [DLMEM_NCC-1:0]                     unlock_valid,
    output wire [DLMEM_NCC-1:0][`VX_CFG_NUM_WARPS-1:0] unlock_mask,

    input  wire [DLMEM_NCC-1:0]                     done_in_valid,
    input  wire [DLMEM_NCC-1:0][NW_WIDTH-1:0]       done_in_slot,
    input  wire [DLMEM_NCC-1:0][GANG_ID_W-1:0]      done_in_gang,
    input  wire [DLMEM_NCC-1:0]                     done_in_epoch,
    output wire [DLMEM_NCC-1:0]                     done_in_ready,

    output wire                                     evt_valid,
    output wire [1:0]                               evt_kind,
    output wire [DLMEM_CORE_W-1:0]                  evt_core,
    output wire [NW_WIDTH-1:0]                      evt_slot,
    output wire [GANG_ID_W-1:0]                     evt_gang,
    output wire                                     evt_epoch,
    output wire [NC_WIDTH-1:0]                      evt_size_m1,

    input  wire                                     rel_valid,
    input  wire [GANG_ID_W-1:0]                     rel_gang,
    input  wire                                     rel_epoch,
    output wire                                     rel_ready,

    // a core holds back its barrier response this cycle
    output wire [DLMEM_NCC-1:0]                     gbar_rsp_stall,
    // a core takes a cluster-barrier release this cycle, or holds one back
    output wire [DLMEM_NCC-1:0]                     cbar_rsp_fire,
    output wire [DLMEM_NCC-1:0]                     cbar_rsp_stall
);
    localparam SS = `VX_CFG_SOCKET_SIZE;

    VX_gbar_bus_if per_socket_gbar_bus_if[NUM_SOCKETS]();
    VX_gbar_bus_if gbar_bus_if();

    for (genvar s = 0; s < NUM_SOCKETS; ++s) begin : g_sockets
        VX_gbar_bus_if per_core_gbar_bus_if[SS]();

        for (genvar i = 0; i < SS; ++i) begin : g_cores
            localparam C = s * SS + i;

            barrier_t req;
            assign req.id        = req_id[C];
            assign req.is_event  = 1'b0;
            assign req.is_global = req_is_global[C];
            assign req.is_arrive = req_is_arrive[C];
            assign req.is_sync   = req_is_sync[C];
            assign req.phase     = req_phase[C];
            assign req.size_m1   = req_size_m1[C];
            assign req.is_cluster = req_is_cluster[C];

            dlmem_cta_done_t done_in;
            assign done_in.core    = DLMEM_CORE_W'(C);
            assign done_in.slot    = done_in_slot[C];
            assign done_in.gang_id = done_in_gang[C];
            assign done_in.epoch   = done_in_epoch[C];

            cbar_slot_t cbar_info;
            localparam CK_W = $bits(cbar_info.k);
            assign cbar_info.gang    = cbar_wr_gang[C];
            assign cbar_info.gang_id = cbar_wr_gang_id[C];
            assign cbar_info.epoch   = cbar_wr_epoch[C];
            assign cbar_info.base    = cbar_wr_base[C];
            assign cbar_info.k       = CK_W'(cbar_wr_k[C]);

            VX_bar_unit #(
                .INSTANCE_ID (`SFORMATF(("core%0d-barrier", C))),
                .CORE_ID     (C)
            ) bar_unit (
                .clk            (clk),
                .reset          (reset),
                .req_valid      (req_valid[C]),
                .req_wid        (req_wid[C]),
                .req_data       (req),
                .read_addr      (read_addr[C]),
                .read_cluster   (read_cluster[C]),
                .read_phase     (read_phase[C]),
                .cbar_wr_valid  (cbar_wr_valid[C]),
                .cbar_wr_slot   (cbar_wr_slot[C]),
                .cbar_wr_info   (cbar_info),
                .gbar_bus_if    (per_core_gbar_bus_if[i]),
                .cta_done_valid (done_in_valid[C]),
                .cta_done_data  (done_in),
                .cta_done_ready (done_in_ready[C]),
                .active_warps   (active_warps[C]),
                .unlock_valid   (unlock_valid[C]),
                .unlock_mask    (unlock_mask[C])
            );

            assign gbar_rsp_stall[C] = per_core_gbar_bus_if[i].rsp_valid && ~per_core_gbar_bus_if[i].rsp_ready;
            assign cbar_rsp_fire[C]  = per_core_gbar_bus_if[i].rsp_valid && per_core_gbar_bus_if[i].rsp_ready
                                    && per_core_gbar_bus_if[i].rsp_data.kind;
            assign cbar_rsp_stall[C] = per_core_gbar_bus_if[i].rsp_valid && ~per_core_gbar_bus_if[i].rsp_ready
                                    && per_core_gbar_bus_if[i].rsp_data.kind;
        end

        VX_gbar_arb #(
            .NUM_REQS    (SS),
            .REQ_OUT_BUF ((SS > 1) ? 3 : 0),
            .RSP_OUT_BUF ((SS > 1) ? 3 : 0)
        ) gbar_arb (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (per_core_gbar_bus_if),
            .bus_out_if (per_socket_gbar_bus_if[s])
        );
    end

    VX_gbar_arb #(
        .NUM_REQS    (NUM_SOCKETS),
        .REQ_OUT_BUF ((NUM_SOCKETS > 1) ? 3 : 0),
        .RSP_OUT_BUF ((NUM_SOCKETS > 1) ? 3 : 0)
    ) gbar_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );

    dlmem_cta_done_t evt_data;

    VX_gbar_unit #(
        .INSTANCE_ID ("gbar0")
    ) gbar_unit (
        .clk         (clk),
        .reset       (reset),
        .gbar_bus_if (gbar_bus_if),
        .evt_valid   (evt_valid),
        .evt_kind    (evt_kind),
        .evt_data    (evt_data),
        .evt_size_m1 (evt_size_m1),
        .rel_valid   (rel_valid),
        .rel_gang    (rel_gang),
        .rel_epoch   (rel_epoch),
        .rel_ready   (rel_ready)
    );

    assign evt_core  = evt_data.core;
    assign evt_slot  = evt_data.slot;
    assign evt_gang  = evt_data.gang_id;
    assign evt_epoch = evt_data.epoch;

endmodule
