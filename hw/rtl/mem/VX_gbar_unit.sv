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

module VX_gbar_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    VX_gbar_bus_if.slave gbar_bus_if
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // CTA_DONE events and cluster-barrier arrivals, forwarded to the cluster's
    // gang unit, which always takes them
    , output wire                  evt_valid
    , output wire [1:0]            evt_kind
    , output dlmem_cta_done_t      evt_data
    , output wire [NC_WIDTH-1:0]   evt_size_m1
    // the gang unit's cluster-barrier releases, broadcast to the cores
    , input  wire                  rel_valid
    , input  wire [GANG_ID_W-1:0]  rel_gang
    , input  wire                  rel_epoch
    , output wire                  rel_ready
`endif
);
    `UNUSED_SPARAM (INSTANCE_ID)

    reg [`VX_CFG_NUM_BARRIERS-1:0][`VX_CFG_NUM_CORES-1:0] barrier_masks;
    wire [`CLOG2(`VX_CFG_NUM_CORES+1)-1:0] active_barrier_count;
    wire [NB_WIDTH-1:0] req_id = gbar_bus_if.req_data.id;
    wire [`VX_CFG_NUM_CORES-1:0] curr_barrier_mask = barrier_masks[req_id];

    `POP_COUNT(active_barrier_count, curr_barrier_mask);
    `UNUSED_VAR (active_barrier_count)

    reg rsp_valid;
    reg [NB_WIDTH-1:0] rsp_bar_id;

    wire req_file  = gbar_bus_if.req_valid && gbar_bus_if.req_ready;
    wire rsp_fire  = gbar_bus_if.rsp_valid && gbar_bus_if.rsp_ready;
    wire rsp_stall = gbar_bus_if.rsp_valid && ~gbar_bus_if.rsp_ready;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // A release takes the response register when no barrier completes into it.
    wire bar_completing = req_file && (gbar_bus_if.req_data.kind == GBAR_KIND_BAR)
                       && (active_barrier_count[NC_WIDTH-1:0] == gbar_bus_if.req_data.size_m1);
    wire rel_fire = rel_valid && rel_ready;
    assign rel_ready = (~rsp_valid || rsp_fire) && ~bar_completing;
`endif

    always @(posedge clk) begin
        if (reset) begin
            barrier_masks <= '0;
            rsp_valid <= 0;
        end else begin
            if (rsp_fire) begin
                rsp_valid <= 0;
            end
        `ifdef VX_CFG_EXT_DLMEM_ENABLE
            if (rel_fire) begin
                rsp_valid <= 1;
            end
            if (req_file && (gbar_bus_if.req_data.kind == GBAR_KIND_BAR)) begin
        `else
            if (req_file) begin
        `endif
                if (active_barrier_count[NC_WIDTH-1:0] == gbar_bus_if.req_data.size_m1) begin
                    barrier_masks[req_id] <= '0;
                    rsp_valid  <= 1;
                    rsp_bar_id <= req_id;
                end else begin
                    barrier_masks[req_id][gbar_bus_if.req_data.core_id] <= 1;
                end
            end
        end
    end

    // request ack
    assign gbar_bus_if.req_ready = ~rsp_stall;

    // response broadcast
    assign gbar_bus_if.rsp_valid = rsp_valid;
    assign gbar_bus_if.rsp_data.id = rsp_bar_id;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    reg                 rsp_kind_r;
    reg [GANG_ID_W-1:0] rsp_gang_r;
    reg                 rsp_epoch_r;

    always @(posedge clk) begin
        if (rel_fire) begin
            rsp_kind_r  <= 1'b1;
            rsp_gang_r  <= rel_gang;
            rsp_epoch_r <= rel_epoch;
        end else if (bar_completing) begin
            rsp_kind_r  <= 1'b0;
        end
    end

    assign gbar_bus_if.rsp_data.kind    = rsp_kind_r;
    assign gbar_bus_if.rsp_data.gang_id = rsp_gang_r;
    assign gbar_bus_if.rsp_data.epoch   = rsp_epoch_r;

    reg                evt_valid_r;
    reg [1:0]          evt_kind_r;
    dlmem_cta_done_t   evt_data_r;
    reg [NC_WIDTH-1:0] evt_size_m1_r;

    always @(posedge clk) begin
        if (reset) begin
            evt_valid_r <= 1'b0;
        end else begin
            evt_valid_r <= req_file && (gbar_bus_if.req_data.kind != GBAR_KIND_BAR);
        end
        evt_kind_r         <= gbar_bus_if.req_data.kind;
        evt_data_r.core    <= DLMEM_CORE_W'(gbar_bus_if.req_data.core_id);
        evt_data_r.slot    <= gbar_bus_if.req_data.slot;
        evt_data_r.gang_id <= gbar_bus_if.req_data.gang_id;
        evt_data_r.epoch   <= gbar_bus_if.req_data.epoch;
        evt_size_m1_r      <= gbar_bus_if.req_data.size_m1;
    end

    assign evt_valid   = evt_valid_r;
    assign evt_kind    = evt_kind_r;
    assign evt_data    = evt_data_r;
    assign evt_size_m1 = evt_size_m1_r;

    `RUNTIME_ASSERT(~req_file || (gbar_bus_if.req_data.kind == GBAR_KIND_BAR) || (gbar_bus_if.req_data.kind == GBAR_KIND_CTA_DONE)
                 || (gbar_bus_if.req_data.kind == GBAR_KIND_CBAR),
        ("%t: %s unknown request kind %0d", $time, INSTANCE_ID, gbar_bus_if.req_data.kind))
`endif

`ifdef DBG_TRACE_GBAR
    always @(posedge clk) begin
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        if (req_file && (gbar_bus_if.req_data.kind == GBAR_KIND_CTA_DONE)) begin
            `TRACE(2, ("%t: %s cta-done: core_id=%0d, slot=%0d, gang=%0d, epoch=%0d\n",
                $time, INSTANCE_ID, gbar_bus_if.req_data.core_id, gbar_bus_if.req_data.slot,
                gbar_bus_if.req_data.gang_id, gbar_bus_if.req_data.epoch))
        end
        if (req_file && (gbar_bus_if.req_data.kind == GBAR_KIND_CBAR)) begin
            `TRACE(2, ("%t: %s cluster-arrive: core_id=%0d, slot=%0d, gang=%0d, epoch=%0d, size=%0d\n",
                $time, INSTANCE_ID, gbar_bus_if.req_data.core_id, gbar_bus_if.req_data.slot,
                gbar_bus_if.req_data.gang_id, gbar_bus_if.req_data.epoch, gbar_bus_if.req_data.size_m1))
        end
        if (rel_fire) begin
            `TRACE(2, ("%t: %s cluster-release: gang=%0d, epoch=%0d\n", $time, INSTANCE_ID, rel_gang, rel_epoch))
        end
        if (req_file && (gbar_bus_if.req_data.kind == GBAR_KIND_BAR)) begin
    `else
        if (req_file) begin
    `endif
            `TRACE(2, ("%t: %s arrive: bar_id=%0d, size=%0d, core_id=%0d\n",
                $time, INSTANCE_ID, gbar_bus_if.req_data.id, gbar_bus_if.req_data.size_m1, gbar_bus_if.req_data.core_id))
        end
        if (rsp_fire) begin
            `TRACE(2, ("%t: %s resume: bar_id=%0d\n", $time, INSTANCE_ID, gbar_bus_if.rsp_data.id))
        end
    end
`endif

endmodule
