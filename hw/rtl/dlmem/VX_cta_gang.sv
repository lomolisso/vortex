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

// Gang table of one GPC, beside its gbar unit: the cluster-barrier half of
// gang admission (the placement half is VX_cta_admit, next to the KMU).
//
// A gang id is taken by VX_cta_admit, which places the gang; `alloc` tells
// this table one cycle later, long before the gang's first event. The event
// port carries the CTA_DONE events and the cluster-barrier arrivals of the
// gang's blocks, one per block and phase. A phase completes once every block
// still live has arrived: by the last arrival, or by the CTA_DONE of the last
// block missing. That raises the gang's release, which leaves lowest gang id
// first; the id stays taken until it has left (`rel_pend` tells the
// admission), so a release never reaches a later gang that reuses the id.
module VX_cta_gang import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CLUSTER_ID = 0
) (
    input wire              clk,
    input wire              reset,

    // the launch's blocks belong to gangs (a cluster launch of K >= 2)
    input wire              gang_mode,

    // a gang id taken by the admission
    input wire              alloc_valid,
    input wire [GANG_ID_W-1:0] alloc_gang,
    input wire              alloc_epoch,
    input wire [DLMEM_K_W-1:0] alloc_k,

    // CTA_DONE events and cluster-barrier arrivals of placed blocks, always accepted
    input wire              evt_valid,
    input wire [1:0]        evt_kind,
    input dlmem_cta_done_t  evt_data,
    input wire [NC_WIDTH-1:0] evt_size_m1,

    // cluster-barrier releases, one per gang and phase
    output wire             rel_valid,
    output wire [GANG_ID_W-1:0] rel_gang,
    output wire             rel_epoch,
    input wire              rel_ready,

    output wire [DLMEM_NUM_GANGS-1:0] rel_pend,

    output wire             busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CLUSTER_ID)

    localparam NCC = DLMEM_NCC;
    localparam NS  = NUM_CTA_MAX;
    localparam NG  = DLMEM_NUM_GANGS;

    reg [NG-1:0]                live_r;
    reg [NG-1:0][DLMEM_K_W-1:0] cnt_r;
    reg [NG-1:0]                epoch_r;
    reg [NG-1:0][DLMEM_K_W-1:0] k_r;

    // cluster barrier: arrivals of the current phase, and a release not yet sent
    reg [NG-1:0][DLMEM_K_W-1:0] arr_r;
    reg [NG-1:0]                relp_r;
    reg [NG-1:0]                rel_ep_r;
    // a phase parity per gang, flipped by each release, and the parity each
    // block's last arrival counted in: a block that ends after its arrival
    // still counts only if its phase has not been released
    reg [NG-1:0]                gph_r;
    reg [NCC-1:0][NS-1:0]       sph_r;

    wire             done_valid = evt_valid && (evt_kind == GBAR_KIND_CTA_DONE);
    wire             cbar_valid = evt_valid && (evt_kind == GBAR_KIND_CBAR);
    dlmem_cta_done_t done_data;
    assign done_data = evt_data;

    wire done_gang = done_valid && gang_mode;

    // A phase's arrival from its last live block raises the gang's release;
    // the release port sends the lowest pending one.
    wire [GANG_ID_W-1:0]  cbar_id   = evt_data.gang_id;
    wire [DLMEM_K_W-1:0]  cbar_next = arr_r[cbar_id] + DLMEM_K_W'(1);
    wire                  cbar_last = (cbar_next == cnt_r[cbar_id]);

    // a block that ends lowers the live count, and its arrival if it still counts
    wire                  done_arr  = done_gang && done_data.arrived
                                   && (sph_r[done_data.core][done_data.slot] == gph_r[cbar_id]);
    wire [DLMEM_K_W-1:0]  done_left = arr_r[cbar_id] - DLMEM_K_W'(done_arr);
    wire                  done_last = done_gang && (done_left != '0)
                                   && (done_left == (cnt_r[cbar_id] - DLMEM_K_W'(1)));

    logic [GANG_ID_W-1:0] rel_id;
    always @(*) begin
        rel_id = '0;
        for (integer i = NG - 1; i >= 0; --i) begin
            if (relp_r[i]) begin
                rel_id = GANG_ID_W'(i);
            end
        end
    end
    assign rel_valid = (|relp_r);
    assign rel_gang  = rel_id;
    assign rel_epoch = rel_ep_r[rel_id];
    assign rel_pend  = relp_r;
    wire rel_fire = rel_valid && rel_ready;

    always @(posedge clk) begin
        if (reset) begin
            live_r   <= '0;
            cnt_r    <= '0;
            epoch_r  <= '0;
            k_r      <= '0;
            arr_r    <= '0;
            relp_r   <= '0;
            rel_ep_r <= '0;
            gph_r    <= '0;
            sph_r    <= '0;
        end else begin
            if (done_gang) begin
                cnt_r[done_data.gang_id] <= cnt_r[done_data.gang_id] - DLMEM_K_W'(1);
                if (cnt_r[done_data.gang_id] == DLMEM_K_W'(1)) begin
                    live_r[done_data.gang_id] <= 1'b0;
                end
            end
            if (alloc_valid) begin
                live_r[alloc_gang]  <= 1'b1;
                cnt_r[alloc_gang]   <= alloc_k;
                epoch_r[alloc_gang] <= alloc_epoch;
                k_r[alloc_gang]     <= alloc_k;
            end

            if (rel_fire) begin
                relp_r[rel_id] <= 1'b0;
            end
            if (cbar_valid) begin
                sph_r[evt_data.core][evt_data.slot] <= gph_r[cbar_id];
                if (cbar_last) begin
                    arr_r[cbar_id]    <= '0;
                    relp_r[cbar_id]   <= 1'b1;
                    rel_ep_r[cbar_id] <= evt_data.epoch;
                    gph_r[cbar_id]    <= ~gph_r[cbar_id];
                end else begin
                    arr_r[cbar_id] <= cbar_next;
                end
            end
            if (done_last) begin
                arr_r[cbar_id]    <= '0;
                relp_r[cbar_id]   <= 1'b1;
                rel_ep_r[cbar_id] <= evt_data.epoch;
                gph_r[cbar_id]    <= ~gph_r[cbar_id];
            end else if (done_arr) begin
                arr_r[cbar_id] <= done_left;
            end
        end
    end

    assign busy = (|relp_r);

    // ---- checks (dlmem_ref's asserts) ---------------------------------------

    `RUNTIME_ASSERT (~alloc_valid || (~live_r[alloc_gang] && ~relp_r[alloc_gang] && (arr_r[alloc_gang] == '0)),
        ("%t: %s: gang %0d taken while still live or releasing", $time, INSTANCE_ID, alloc_gang))
    `RUNTIME_ASSERT (~done_gang || ((32'(done_data.gang_id) < NG) && live_r[done_data.gang_id]
                                    && (epoch_r[done_data.gang_id] == done_data.epoch)
                                    && (cnt_r[done_data.gang_id] != 0)),
        ("%t: %s: CTA_DONE for gang %0d epoch %0d, which is not live", $time, INSTANCE_ID,
         done_data.gang_id, done_data.epoch))
    `RUNTIME_ASSERT (~done_arr || (arr_r[done_data.gang_id] != '0),
        ("%t: %s: CTA_DONE of gang %0d epoch %0d counts an arrival the gang does not hold", $time, INSTANCE_ID,
         done_data.gang_id, done_data.epoch))
    `RUNTIME_ASSERT (~cbar_valid || (gang_mode
                                     && (32'(cbar_id) < NG) && live_r[cbar_id] && (epoch_r[cbar_id] == evt_data.epoch)
                                     && ~relp_r[cbar_id] && (arr_r[cbar_id] < cnt_r[cbar_id])
                                     && ((32'(evt_size_m1) + 1) == 32'(k_r[cbar_id]))),
        ("%t: %s: cluster-barrier arrival for gang %0d epoch %0d (size %0d) that is not live, not waiting for it, or of another size",
         $time, INSTANCE_ID, cbar_id, evt_data.epoch, 32'(evt_size_m1) + 1))

endmodule
