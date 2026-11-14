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

`ifdef VX_CFG_TCU_WGMMA_ENABLE

// CTA lockstep gate for the shared VX_tcu_tbuf bbuf.
//
// The shared bbuf assumes single-CTA occupancy across all blocks. A new
// WGMMA on block b is deferred while any other block holds an in-flight
// expansion for a different cta_id, so that all warps of the current CTA
// drain before a new CTA enters. Same-CTA across blocks (the production
// case for a warpgroup at the same uop) remains free.
//
// Per-block ownership matches the SimX model: a block records its CTA when
// its first compute uop enters WGMMA, and releases on its last compute uop.
// A block is deferred while another block is mid-WGMMA for a different CTA,
// or while a lower block starts a WGMMA of a different CTA in the same cycle.
//
// Contract: the consumer must AND `cta_conflict` into the request
// validity presented to bbuf and into any downstream ready that gates
// fire. A sim-only assertion below catches forgotten masks.

module VX_tcu_lockstep import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter         BLOCK_SIZE  = `VX_CFG_NUM_TCU_BLOCKS
) (
    input wire clk,
    input wire reset,

    // observation
    input  wire [BLOCK_SIZE-1:0]                 is_wgmma_b,
    input  wire [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0] new_cta_b,
    input  wire [BLOCK_SIZE-1:0]                 exec_fire_b,
    input  wire [BLOCK_SIZE-1:0]                 is_first_uop_b,
    input  wire [BLOCK_SIZE-1:0]                 is_last_uop_b,

    // gating output
    output wire [BLOCK_SIZE-1:0]                 cta_conflict
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    // Per-block "WGMMA expansion in progress" — first compute uop joins the
    // CTA-owned region; the last compute uop releases it. This persists
    // across LMEM-stall gaps so a different CTA cannot enter mid-expansion
    // and corrupt the shared bbuf state.
    reg [BLOCK_SIZE-1:0]      in_expansion_r;
    reg [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0] cta_owner_r;

    // -----------------------------------------------------------------------
    // Owner update.
    //
    // Leader-fire detection: any block joining a WGMMA expansion this cycle.
    // The owner cta is picked from the lowest-indexed leader-firing block.
    // Under same-CTA warpgroup fire (production path), every leader-firing
    // block carries the same cta_id, so the priority pick agrees with all.
    // -----------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0] leader_fire_b = exec_fire_b & is_first_uop_b & is_wgmma_b;
    // in_expansion_next reflects this-cycle updates (set on leader fire,
    // clear on last-uop fire).
    wire [BLOCK_SIZE-1:0] in_expansion_next;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_inexp_next
        wire is_wgmma_fire = exec_fire_b[bi] && is_wgmma_b[bi];
        assign in_expansion_next[bi] =
            (in_expansion_r[bi] || (is_wgmma_fire && is_first_uop_b[bi]))
            && !(is_wgmma_fire && is_last_uop_b[bi]);
    end

    always @(posedge clk) begin
        if (reset) begin
            in_expansion_r <= '0;
            cta_owner_r    <= '0;
        end else begin
            in_expansion_r <= in_expansion_next;
            for (int bi = 0; bi < BLOCK_SIZE; ++bi) begin
                if (leader_fire_b[bi]) begin
                    cta_owner_r[bi] <= new_cta_b[bi];
                end
            end
        end
    end

    // -----------------------------------------------------------------------
    // Per-block conflict.
    //
    // The registered test defers a block while another block is mid-WGMMA
    // for a different CTA. On its own it would let two blocks start WGMMAs of
    // different CTAs in the same cycle, after which each blocks the other
    // forever. So a block that starts this cycle (a leader) also defers the
    // higher blocks of other CTAs; the leaders are resolved lowest block
    // first, so a deferred block is not itself a leader.
    // -----------------------------------------------------------------------
    logic [BLOCK_SIZE-1:0] conflict_w;
    always_comb begin : p_gate
        logic [BLOCK_SIZE-1:0] lead_v;
        lead_v = '0;
        for (int bi = 0; bi < BLOCK_SIZE; ++bi) begin
            conflict_w[bi] = 1'b0;
            for (int k = 0; k < BLOCK_SIZE; ++k) begin
                if (k != bi && in_expansion_r[k] && (cta_owner_r[k] != new_cta_b[bi])) begin
                    conflict_w[bi] = 1'b1;
                end
            end
            for (int k = 0; k < bi; ++k) begin
                if (lead_v[k] && (new_cta_b[k] != new_cta_b[bi])) begin
                    conflict_w[bi] = 1'b1;
                end
            end
            lead_v[bi] = is_wgmma_b[bi] && is_first_uop_b[bi] && !in_expansion_r[bi] && !conflict_w[bi];
        end
    end
    assign cta_conflict = is_wgmma_b & conflict_w;

    // -----------------------------------------------------------------------
    // Sim-only contract assertion: a fired WGMMA uop must have its
    // cta_conflict held low. If a downstream consumer forgets to AND
    // cta_conflict into the gating ready, this fires.
    // -----------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0] lockstep_violation;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_lockstep_violation
        assign lockstep_violation[bi] = exec_fire_b[bi] && is_wgmma_b[bi] && cta_conflict[bi];
    end
    `RUNTIME_ASSERT (~|lockstep_violation,
        ("%s lockstep violation: a WGMMA uop fired with a cta_id that conflicts with another block's resident CTA",
         INSTANCE_ID))
`ifdef SIMULATION
    always @(posedge clk) begin
        if (!reset) begin
            for (int i = 0; i < BLOCK_SIZE; ++i) begin
                for (int j = i + 1; j < BLOCK_SIZE; ++j) begin
                    `ASSERT(!(in_expansion_r[i] && in_expansion_r[j] && (cta_owner_r[i] != cta_owner_r[j])),
                        ("%t: %s TCU blocks %0d and %0d inside WGMMAs of different CTAs", $time, INSTANCE_ID, i, j));
                end
            end
        end
    end
`endif

endmodule

`endif // VX_CFG_TCU_WGMMA_ENABLE
