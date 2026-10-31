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

module VX_dxa_unit import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,
`ifdef DXA_MBAR_ENABLE
    input wire [`VX_CFG_NUM_WARPS-1:0][NW_WIDTH-1:0] cta_warp_slot,
    input wire [`VX_CFG_NUM_WARPS-1:0][DXA_CTA_GEN_W-1:0] cta_slot_gen,
    // The issuing block's DLMEM peer row and the launch's slot bases.
    output wire [NW_WIDTH-1:0]                       peer_rd_slot,
    input  dlmem_row_t                               peer_row,
    input wire [`VX_CFG_NUM_WARPS-1:0][`VX_CFG_LMEM_LOG_SIZE-1:0] peer_slot_base,
    input wire [`VX_CFG_LMEM_LOG_SIZE:0]             peer_stride,
`endif
    VX_dxa_req_bus_if.master dxa_req_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (execute_if.data.rs3_data)

    localparam NUM_LANES = `VX_CFG_NUM_SFU_LANES;
    localparam DXA_LANES = 4;
    `STATIC_ASSERT (`VX_CFG_NUM_THREADS >= DXA_LANES, ("DXA requires at least 4 threads per warp"))
    // Packet carrying lane 3: it issues the request; later packets only retire.
    localparam SEND_PID  = (DXA_LANES - 1) / NUM_LANES;
    localparam PID_W     = $bits(execute_if.data.header.pid);

    wire is_send = (execute_if.data.header.pid == PID_W'(SEND_PID));

    // Wgather-based layout (lane index = thread_id & 3):
    //   Lane 0: rs1=smem_addr, rs2=coord2
    //   Lane 1: rs1=meta,      rs2=coord3
    //   Lane 2: rs1=coord0,    rs2=coord4
    //   Lane 3: rs1=coord1,    rs2=cta_mask (multicast)
    wire [DXA_LANES-1:0][`VX_CFG_XLEN-1:0] lane_rs1;
    wire [DXA_LANES-1:0][`VX_CFG_XLEN-1:0] lane_rs2;

    if (SEND_PID != 0) begin : g_stage
        localparam STAGE_LANES = SEND_PID * NUM_LANES;
        // Every packet of a DXA is dispatched, active or not, and a warp's
        // packets are contiguous within its issue slice, so one staging entry
        // per slice holds the lanes of its earlier packets.
        reg [`VX_CFG_ISSUE_WIDTH-1:0][STAGE_LANES-1:0][`VX_CFG_XLEN-1:0] stage_rs1;
        reg [`VX_CFG_ISSUE_WIDTH-1:0][STAGE_LANES-1:0][`VX_CFG_XLEN-1:0] stage_rs2;
        wire [ISSUE_ISW_W-1:0] isw = wid_to_isw(execute_if.data.header.wid);
        wire stage_fire = execute_if.valid && execute_if.ready;

        always @(posedge clk) begin
            for (integer p = 0; p < SEND_PID; ++p) begin
                if (stage_fire && (execute_if.data.header.pid == PID_W'(p))) begin
                    stage_rs1[isw][p * NUM_LANES +: NUM_LANES] <= execute_if.data.rs1_data;
                    stage_rs2[isw][p * NUM_LANES +: NUM_LANES] <= execute_if.data.rs2_data;
                end
            end
        end

        for (genvar k = 0; k < DXA_LANES; ++k) begin : g_lanes
            if (k < STAGE_LANES) begin : g_staged
                assign lane_rs1[k] = stage_rs1[isw][k];
                assign lane_rs2[k] = stage_rs2[isw][k];
            end else begin : g_packet
                assign lane_rs1[k] = execute_if.data.rs1_data[k - STAGE_LANES];
                assign lane_rs2[k] = execute_if.data.rs2_data[k - STAGE_LANES];
            end
        end
    end else begin : g_direct
        for (genvar k = 0; k < DXA_LANES; ++k) begin : g_lanes
            assign lane_rs1[k] = execute_if.data.rs1_data[k];
            assign lane_rs2[k] = execute_if.data.rs2_data[k];
        end
        if (NUM_LANES > DXA_LANES) begin : g_unused
            `UNUSED_VAR (execute_if.data.rs1_data)
            `UNUSED_VAR (execute_if.data.rs2_data)
        end
    end

    wire [`VX_CFG_XLEN-1:0] lane0_rs1 = lane_rs1[0];
    wire [`VX_CFG_XLEN-1:0] lane1_rs1 = lane_rs1[1];
    wire [`VX_CFG_XLEN-1:0] lane2_rs1 = lane_rs1[2];
    wire [`VX_CFG_XLEN-1:0] lane3_rs1 = lane_rs1[3];
    wire [`VX_CFG_XLEN-1:0] lane0_rs2 = lane_rs2[0];
    wire [`VX_CFG_XLEN-1:0] lane1_rs2 = lane_rs2[1];
    wire [`VX_CFG_XLEN-1:0] lane2_rs2 = lane_rs2[2];
    wire [`VX_CFG_XLEN-1:0] lane3_rs2 = lane_rs2[3];
    `UNUSED_VAR (lane3_rs2)

    // Cluster-contiguous LMEM placement guarantees receiver bases are
    // `issuer_base + r × smem_stride`, so the bus carries the issuer's
    // LMEM-relative byte address (issuer_lmem_base + intra). The per-beat
    // receiver address is `bus_addr + r × stride` and is computed in
    // VX_dxa_smem_wr's replay path. No issuer-side intra_offset, no
    // per-receiver translation in VX_mem_unit.

    // Strip the global LMEM base prefix to land in LMEM's own byte-address
    // space. Equivalently: `lane0_rs1[LMEM_BYTE_W-1:0]`, since `lane0_rs1`
    // is a kernel-supplied LMEM pointer whose high bits ARE VX_MEM_LMEM_BASE
    // when the kernel uses `__local_mem()` derivatives.
    wire [`VX_CFG_XLEN-1:0] lmem_rel_byte_addr =
        lane0_rs1 - `VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR);

    // Build dxa_req payload
    dxa_req_data_t dxa_req_data_in;
    assign dxa_req_data_in.core_id   = NC_WIDTH'(CORE_ID);
    assign dxa_req_data_in.uuid      = execute_if.data.header.uuid;
    assign dxa_req_data_in.wid       = execute_if.data.header.wid;
`ifndef DXA_MBAR_ENABLE
    assign dxa_req_data_in.smem_addr = lmem_rel_byte_addr[DXA_SMEM_ADDR_W-1:0];
`endif
    assign dxa_req_data_in.meta      = lane1_rs1[31:0];
    assign dxa_req_data_in.coords[0] = lane2_rs1[31:0];
    assign dxa_req_data_in.coords[1] = lane3_rs1[31:0];
    assign dxa_req_data_in.coords[2] = lane0_rs2[31:0];
    assign dxa_req_data_in.coords[3] = lane1_rs2[31:0];
    assign dxa_req_data_in.coords[4] = lane2_rs2[31:0];
    assign dxa_req_data_in.cta_mask  = lane3_rs2[`VX_CFG_NUM_WARPS-1:0];
`ifdef DXA_MBAR_ENABLE
    // A DLMEM window address names a block of the issuer's cluster, as the LSU's
    // aperture decodes it: rank r maps to peer r of the issuing block's row. A
    // block on this core takes the local path at its slot's base, any other
    // the DLMEM network. A window mbarrier (payload bit 25) carries its rank
    // in payload[24:20] and a block-relative word offset.
    localparam LMEM_LOG = `VX_CFG_LMEM_LOG_SIZE;
    localparam RB       = DLMEM_RANK_BITS;
    localparam RB_W     = `UP(RB);
    localparam WIN_LSB  = LMEM_LOG + RB;
    localparam MY_CORE  = CORE_ID % DLMEM_NCC;
    localparam [`VX_CFG_XLEN-1:0] DLMEM_BASE_B = `VX_CFG_XLEN'(`VX_MEM_DLMEM_BASE_ADDR);

    wire [NW_WIDTH-1:0] issue_slot = cta_warp_slot[execute_if.data.header.wid];
    assign peer_rd_slot = issue_slot;

    wire [RB_W-1:0] dst_rank;
    if (RB != 0) begin : g_dst_rank
        assign dst_rank = lane0_rs1[LMEM_LOG +: RB];
    end else begin : g_dst_rank0
        assign dst_rank = '0;
    end
    wire [LMEM_LOG-1:0] dst_off = lane0_rs1[LMEM_LOG-1:0];
    wire dst_win = ((lane0_rs1 >> WIN_LSB) == (DLMEM_BASE_B >> WIN_LSB));

    wire mbar_mode = lane1_rs1[30];
    wire mbar_win  = mbar_mode && lane1_rs1[29];
    wire [4:0] mbar_rank = lane1_rs1[28:24];
    wire [DXA_MBAR_OFF_W-1:0] mbar_off_w = lane1_rs1[4 +: DXA_MBAR_OFF_W];

    dlmem_peer_t dst_peer, mbar_peer;
    assign dst_peer  = peer_row.peers[dst_rank];
    assign mbar_peer = peer_row.peers[RB_W'(mbar_rank)];

    wire dst_remote = dst_win && (dst_peer.core != DLMEM_CORE_W'(MY_CORE));
    wire [NW_WIDTH-1:0] dst_slot = dst_win ? dst_peer.slot : issue_slot;
    wire [LMEM_LOG-1:0] dst_base = peer_slot_base[dst_peer.slot];
    wire [LMEM_LOG-1:0] mbar_base = peer_slot_base[mbar_peer.slot];
    `UNUSED_VAR (mbar_base[2:0])

    assign dxa_req_data_in.smem_addr = dst_win ? (dst_base + dst_off) : lmem_rel_byte_addr[DXA_SMEM_ADDR_W-1:0];
    assign dxa_req_data_in.slot      = dst_slot;
    assign dxa_req_data_in.gen       = cta_slot_gen[dst_slot];
    assign dxa_req_data_in.remote    = dst_remote;
    assign dxa_req_data_in.dst_core  = dst_peer.core;
    assign dxa_req_data_in.dst_slot  = dst_peer.slot;
    assign dxa_req_data_in.mbar_off  = mbar_win ? (mbar_base[LMEM_LOG-1:3] + mbar_off_w) : mbar_off_w;
    assign dxa_req_data_in.win       = dst_win;
    assign dxa_req_data_in.win_room  = peer_stride - (LMEM_LOG+1)'(dst_off);

    // Issue-time checks, simulation only.
    wire dst_oob  = dst_win && ((32'(dst_rank) >= 32'(peer_row.k)) || ({1'b0, dst_off} >= peer_stride));
    wire mbar_oob = mbar_win && ((32'(mbar_rank) >= 32'(peer_row.k))
                              || ({1'b0, mbar_off_w, 3'b0} >= (LMEM_LOG+1)'(peer_stride)));
    wire [DLMEM_CORE_W-1:0] dst_blk_core  = dst_win  ? dst_peer.core  : DLMEM_CORE_W'(MY_CORE);
    wire [DLMEM_CORE_W-1:0] mbar_blk_core = mbar_win ? mbar_peer.core : DLMEM_CORE_W'(MY_CORE);
    wire [NW_WIDTH-1:0]     mbar_blk_slot = mbar_win ? mbar_peer.slot : issue_slot;
    wire mbar_elsewhere = mbar_mode && ((dst_blk_core != mbar_blk_core) || (dst_slot != mbar_blk_slot));
`endif
    // smem_addr (LMEM byte width) and meta/coords (32-bit ABI) take low bits;
    // high bits unused when XLEN exceeds the field width
    `UNUSED_VAR (lmem_rel_byte_addr)
    `UNUSED_VAR (lane1_rs1)
    `UNUSED_VAR (lane2_rs1)
    `UNUSED_VAR (lane3_rs1)
    `UNUSED_VAR (lane0_rs2)
    `UNUSED_VAR (lane1_rs2)
    `UNUSED_VAR (lane2_rs2)

    // Output elastic buffer breaks the combinatorial path between
    // dxa_req_arb and this unit. Barrier transaction registration is
    // handled by software via vx_barrier.h::expect_tx.
    wire dxa_buf_ready, wb_ready;
    wire accept = (dxa_buf_ready || ~is_send) && wb_ready;
    wire fire   = execute_if.valid && accept;

    assign execute_if.ready = accept;

`ifdef DXA_MBAR_ENABLE
    wire issue_fire = fire && is_send;
    wire is_mcast   = ($countones(dxa_req_data_in.cta_mask) > 1);
    `RUNTIME_ASSERT(~(issue_fire && (dxa_req_data_in.meta[30] || dst_win) && is_mcast),
        ("%t: *** %s DXA multicast with an mbarrier is not supported: wid=%0d cta_mask=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, dxa_req_data_in.cta_mask))
    `RUNTIME_ASSERT(~(issue_fire && (dst_oob || mbar_oob)),
        ("%t: *** %s DXA destination rank outside the cluster: wid=%0d smem=0x%0h meta=0x%0h k=%0d stride=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1, peer_row.k, peer_stride))
    `RUNTIME_ASSERT(~(issue_fire && ~dst_oob && ~mbar_oob && mbar_elsewhere),
        ("%t: *** %s DXA mbarrier must be in the destination block: wid=%0d smem=0x%0h meta=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1))
    `RUNTIME_ASSERT(~(issue_fire && dst_win && ~mbar_mode),
        ("%t: *** %s DXA cluster destination needs an mbarrier: wid=%0d smem=0x%0h meta=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1))
`endif

    VX_elastic_buffer #(
        .DATAW   ($bits(dxa_req_data_t)),
        .SIZE    (2),
        .OUT_REG (1) // register the dxa-request master boundary
    ) dxa_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fire && is_send),
        .ready_in  (dxa_buf_ready),
        .data_in   (dxa_req_data_in),
        .valid_out (dxa_req_bus_if.req_valid),
        .ready_out (dxa_req_bus_if.req_ready),
        .data_out  (dxa_req_bus_if.req_data)
    );

    sfu_header_t header_out;

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_header_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fire),
        .ready_in  (wb_ready),
        .data_in   (execute_if.data.header),
        .data_out  (header_out),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

    assign result_if.data.header = header_out;
    assign result_if.data.data   = '0;

`ifdef DBG_TRACE_DXA
    always_ff @(posedge clk) begin
        if (~reset && dxa_req_bus_if.req_valid && dxa_req_bus_if.req_ready) begin
            `TRACE(1, ("%t: %s dxa-req: wid=%0d, smem=0x%0h, meta=0x%0h, c0=%0d, c1=%0d, c2=%0d, c3=%0d, c4=%0d\n",
                $time, INSTANCE_ID, dxa_req_bus_if.req_data.wid,
                dxa_req_bus_if.req_data.smem_addr, dxa_req_bus_if.req_data.meta,
                dxa_req_bus_if.req_data.coords[0], dxa_req_bus_if.req_data.coords[1],
                dxa_req_bus_if.req_data.coords[2], dxa_req_bus_if.req_data.coords[3],
                dxa_req_bus_if.req_data.coords[4]))
        `ifdef DXA_MBAR_ENABLE
            if (dxa_req_bus_if.req_data.remote) begin
                `TRACE(1, ("%t: %s dxa-req-remote: wid=%0d, dst_core=%0d, dst_slot=%0d, mbar_off=0x%0h\n",
                    $time, INSTANCE_ID, dxa_req_bus_if.req_data.wid, dxa_req_bus_if.req_data.dst_core,
                    dxa_req_bus_if.req_data.dst_slot, dxa_req_bus_if.req_data.mbar_off))
            end
        `endif
        end
    end
`endif

endmodule
