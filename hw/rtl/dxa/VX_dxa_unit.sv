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
`ifdef DXA_MBAR_ENABLE
    // A bulk copy names no descriptor and no box: an mbarrier-mode meta of
    // descriptor 0, zero coordinates and no multicast. A tensor load
    // (funct3 2) packs two coordinates per lane and its descriptor slot and
    // prefetch flag in lane 0 rs2[36:32]; its multicast mask is lane 1 rs2.
    wire is_bulk  = execute_if.data.op_args.dxa.bulk;
    wire is_tload = execute_if.data.op_args.dxa.tload;
    wire is_bload = execute_if.data.op_args.dxa.bload;
    wire is_bpf   = execute_if.data.op_args.dxa.bprefetch;
    wire is_lin   = is_bulk || is_bload || is_bpf;
    wire is_raw   = is_bulk || is_tload || is_bload;
    wire is_pf    = (is_tload && lane0_rs2[36]) || is_bpf;
    wire [31:0] tl_meta = {2'b11, 26'd0, lane0_rs2[35:32]};
    assign dxa_req_data_in.meta      = is_tload ? tl_meta : (is_lin ? 32'hC0000000 : lane1_rs1[31:0]);
    assign dxa_req_data_in.coords[0] = is_lin ? 32'd0 : lane2_rs1[31:0];
    assign dxa_req_data_in.coords[1] = is_lin ? 32'd0 : (is_tload ? lane2_rs1[63:32] : lane3_rs1[31:0]);
    assign dxa_req_data_in.coords[2] = is_lin ? 32'd0 : (is_tload ? lane3_rs1[31:0] : lane0_rs2[31:0]);
    assign dxa_req_data_in.coords[3] = is_lin ? 32'd0 : (is_tload ? lane3_rs1[63:32] : lane1_rs2[31:0]);
    assign dxa_req_data_in.coords[4] = is_lin ? 32'd0 : (is_tload ? lane0_rs2[31:0] : lane2_rs2[31:0]);
    assign dxa_req_data_in.cta_mask  = (is_lin || is_tload) ? '0 : lane3_rs2[`VX_CFG_NUM_WARPS-1:0];
    assign dxa_req_data_in.tload     = is_tload;
    assign dxa_req_data_in.bload     = is_bload;
    assign dxa_req_data_in.prefetch  = is_pf;
    assign dxa_req_data_in.offs[0]   = is_tload ? lane1_rs2[31:16] : 16'd0;
    assign dxa_req_data_in.offs[1]   = is_tload ? lane1_rs2[47:32] : 16'd0;
    assign dxa_req_data_in.offs[2]   = is_tload ? lane1_rs2[63:48] : 16'd0;
    assign dxa_req_data_in.gsrc      = (is_bload || is_bpf) ? lane2_rs1[`VX_CFG_MEM_ADDR_WIDTH-1:0] : '0;
`else
    assign dxa_req_data_in.meta      = lane1_rs1[31:0];
    assign dxa_req_data_in.coords[0] = lane2_rs1[31:0];
    assign dxa_req_data_in.coords[1] = lane3_rs1[31:0];
    assign dxa_req_data_in.coords[2] = lane0_rs2[31:0];
    assign dxa_req_data_in.coords[3] = lane1_rs2[31:0];
    assign dxa_req_data_in.coords[4] = lane2_rs2[31:0];
    assign dxa_req_data_in.cta_mask  = lane3_rs2[`VX_CFG_NUM_WARPS-1:0];
`endif
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
    // A prefetch has no destination.
    wire dst_win = ~is_pf && ((lane0_rs1 >> WIN_LSB) == (DLMEM_BASE_B >> WIN_LSB));

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

    // A bulk copy's lane 1 is a raw mbarrier pointer, decoded like lane 0:
    // a window rank of the issuer's cluster, or a local address.
    localparam [`VX_CFG_XLEN-1:0] LMEM_BASE_B = `VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR);
    wire [RB_W-1:0] bk_mb_rank;
    if (RB != 0) begin : g_bk_mb_rank
        assign bk_mb_rank = lane1_rs1[LMEM_LOG +: RB];
    end else begin : g_bk_mb_rank0
        assign bk_mb_rank = '0;
    end
    wire [LMEM_LOG-1:0] bk_mb_off = lane1_rs1[LMEM_LOG-1:0];
    wire bk_mb_win = ((lane1_rs1 >> WIN_LSB) == (DLMEM_BASE_B >> WIN_LSB));
    dlmem_peer_t bk_mb_peer;
    assign bk_mb_peer = peer_row.peers[bk_mb_rank];
    wire [LMEM_LOG-1:0] bk_mb_base = peer_slot_base[bk_mb_peer.slot];
    wire [`VX_CFG_XLEN-1:0] bk_mb_rel  = lane1_rs1 - LMEM_BASE_B;
    wire [`VX_CFG_XLEN-1:0] bk_src_rel = lane2_rs1 - LMEM_BASE_B;
    wire [`VX_CFG_XLEN-1:0] bk_size    = lane3_rs1;
    wire [LMEM_LOG-1:0] bk_mb_lmem = bk_mb_win ? (bk_mb_base + bk_mb_off) : bk_mb_rel[LMEM_LOG-1:0];
    `UNUSED_VAR (bk_mb_lmem[2:0])

    // A multicast in mbarrier mode names cluster ranks: every masked rank gets
    // the data and its mbarrier at the issuer's own block-relative offsets.
    // The raw forms (funct3 1..3) decode their mbarrier pointer as above.
    wire [31:0] mc_mask_w = (is_tload || is_bload) ? 32'(lane1_rs2[15:0]) : lane3_rs2[31:0];
    wire mbar_mode_e = is_raw || mbar_mode;
    wire mbar_win_e  = is_raw ? bk_mb_win : mbar_win;
    wire mc_mode    = ~is_bulk && ~is_pf && mbar_mode_e && (mc_mask_w != '0);
    wire mc_cluster = mc_mode && ~dst_win && ~mbar_win_e;
    wire [LMEM_LOG-1:0] own_base = peer_slot_base[issue_slot];
    wire [LMEM_LOG-1:0] mc_off   = lmem_rel_byte_addr[LMEM_LOG-1:0] - own_base;
    wire [DXA_MBAR_OFF_W-1:0] mc_mbar_rel = (is_raw ? bk_mb_lmem[LMEM_LOG-1:3] : mbar_off_w) - own_base[LMEM_LOG-1:3];
    wire [LMEM_LOG:0] mc_room = ({1'b0, mc_off} < peer_stride) ? (peer_stride - (LMEM_LOG+1)'(mc_off)) : '0;
    // A local destination's bytes up to the end of the issuer's block.
    wire [`VX_CFG_XLEN-1:0] own_off = lmem_rel_byte_addr - `VX_CFG_XLEN'(own_base);
    wire [LMEM_LOG:0] own_room = (own_off < `VX_CFG_XLEN'(peer_stride)) ? (peer_stride - (LMEM_LOG+1)'(own_off)) : '0;

    dxa_mc_peer_t [DLMEM_MAX_CLUSTER-1:0] mc_peers;
    wire [DLMEM_MAX_CLUSTER-1:0] row_other;
    for (genvar r = 0; r < DLMEM_MAX_CLUSTER; ++r) begin : g_mc_peers
        assign mc_peers[r].remote = (peer_row.peers[r].core != DLMEM_CORE_W'(MY_CORE));
        assign mc_peers[r].core   = peer_row.peers[r].core;
        assign mc_peers[r].slot   = peer_row.peers[r].slot;
        assign mc_peers[r].gen    = cta_slot_gen[peer_row.peers[r].slot];
        assign row_other[r] = (32'(r) < 32'(peer_row.k)) && mc_peers[r].remote;
    end

    assign dxa_req_data_in.smem_addr = mc_cluster ? mc_off
                                     : (dst_win ? (dst_base + dst_off) : lmem_rel_byte_addr[DXA_SMEM_ADDR_W-1:0]);
    assign dxa_req_data_in.blk_base  = mc_cluster ? '0 : (dst_win ? dst_base : own_base);
    assign dxa_req_data_in.slot      = dst_slot;
    assign dxa_req_data_in.gen       = cta_slot_gen[dst_slot];
    assign dxa_req_data_in.remote    = dst_remote;
    assign dxa_req_data_in.dst_core  = dst_peer.core;
    assign dxa_req_data_in.dst_slot  = dst_peer.slot;
    assign dxa_req_data_in.mbar_off  = mc_cluster ? mc_mbar_rel
                                     : is_raw ? bk_mb_lmem[LMEM_LOG-1:3]
                                     : (mbar_win ? (mbar_base[LMEM_LOG-1:3] + mbar_off_w) : mbar_off_w);
    assign dxa_req_data_in.win       = dst_win;
    assign dxa_req_data_in.win_room  = mc_cluster ? mc_room
                                     : (dst_win ? (peer_stride - (LMEM_LOG+1)'(dst_off)) : own_room);
    assign dxa_req_data_in.mc_cluster = mc_cluster;
    assign dxa_req_data_in.mc_mask   = mc_mask_w[DLMEM_MAX_CLUSTER-1:0];
    assign dxa_req_data_in.mc_peers  = mc_peers;
    assign dxa_req_data_in.bulk      = is_bulk;
    assign dxa_req_data_in.src_addr  = bk_src_rel[LMEM_LOG-1:0];
    // A bulk prefetch is a hint: it reads at most the local memory's size.
    wire [31:0] bpf_size = bk_size[31:0];
    wire bpf_clamp = (bpf_size > 32'(1 << LMEM_LOG));
    assign dxa_req_data_in.size      = (is_bpf && bpf_clamp) ? (LMEM_LOG+1)'(1 << LMEM_LOG) : bk_size[LMEM_LOG:0];

    // Issue-time checks, simulation only.
    wire dst_oob  = dst_win && ((32'(dst_rank) >= 32'(peer_row.k)) || ({1'b0, dst_off} >= peer_stride));
    wire mbar_oob = mbar_win && ((32'(mbar_rank) >= 32'(peer_row.k))
                              || ({1'b0, mbar_off_w, 3'b0} >= (LMEM_LOG+1)'(peer_stride)));
    wire [DLMEM_CORE_W-1:0] dst_blk_core  = dst_win  ? dst_peer.core  : DLMEM_CORE_W'(MY_CORE);
    wire [DLMEM_CORE_W-1:0] mbar_blk_core = mbar_win ? mbar_peer.core : DLMEM_CORE_W'(MY_CORE);
    wire [NW_WIDTH-1:0]     mbar_blk_slot = mbar_win ? mbar_peer.slot : issue_slot;
    wire mbar_elsewhere = mbar_mode && ((dst_blk_core != mbar_blk_core) || (dst_slot != mbar_blk_slot));

    // Bulk copy checks: a local address must lie in the issuer's own block.
    wire [`VX_CFG_XLEN-1:0] own_lo = `VX_CFG_XLEN'(own_base);
    wire [`VX_CFG_XLEN-1:0] own_hi = own_lo + `VX_CFG_XLEN'(peer_stride);
    wire [`VX_CFG_XLEN-1:0] bk_dst_off = dst_win ? `VX_CFG_XLEN'(dst_off) : (lmem_rel_byte_addr - own_lo);
    wire bk_src_in  = (bk_src_rel >= own_lo) && (bk_src_rel < own_hi);
    wire bk_mb_in   = (bk_mb_rel >= own_lo) && (bk_mb_rel < own_hi);
    wire bk_align   = (bk_size == '0) || (bk_size[3:0] != '0) || (bk_src_rel[3:0] != '0) || (bk_dst_off[3:0] != '0);
    wire bk_src_out = ~bk_src_in || (bk_size > (own_hi - bk_src_rel));
    wire bk_mb_oob  = bk_mb_win && ((32'(bk_mb_rank) >= 32'(peer_row.k)) || ({1'b0, bk_mb_off} >= peer_stride));
    wire bk_rank    = dst_oob || bk_mb_oob;
    wire [DLMEM_CORE_W-1:0] bk_mb_core = bk_mb_win ? bk_mb_peer.core : DLMEM_CORE_W'(MY_CORE);
    wire [NW_WIDTH-1:0]     bk_mb_slot = bk_mb_win ? bk_mb_peer.slot : issue_slot;
    wire bk_mbar    = (~bk_mb_win && ~bk_mb_in) || (dst_blk_core != bk_mb_core) || (dst_slot != bk_mb_slot);
    wire bk_box     = (bk_dst_off >= `VX_CFG_XLEN'(peer_stride))
                   || (bk_size > (`VX_CFG_XLEN'(peer_stride) - bk_dst_off));
    // A global source (bload, bulk prefetch) has the same 16-byte rules.
    wire gb_align   = (is_bpf ? (bpf_size == '0) : (bk_size == '0)) || (bk_size[3:0] != '0) || (lane2_rs1[3:0] != '0)
                   || (is_bload && (bk_dst_off[3:0] != '0));
    // An LMEM-to-LMEM copy by a local pointer or by its own rank names the issuer's block.
    wire bk_self    = ~dst_win || ((dst_peer.core == DLMEM_CORE_W'(MY_CORE)) && (dst_peer.slot == issue_slot));
    wire mb_misalign = (lane1_rs1[2:0] != '0);
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
    wire bulk_fire  = fire && is_send && is_bulk;
    wire issue_fire = fire && is_send && ~is_raw && ~is_bpf;
    wire ld_fire    = fire && is_send && (is_tload || is_bload) && ~is_pf;
    wire bpf_fire   = fire && is_send && is_bpf;
    wire is_mcast   = ($countones(dxa_req_data_in.cta_mask) > 1);
    wire mc_rank_oob = ((mc_mask_w >> peer_row.k) != '0);
    wire mc_mbar_oob = ({1'b0, mc_mbar_rel, 3'b0} >= (LMEM_LOG+1)'(peer_stride));
    `RUNTIME_ASSERT(~(issue_fire && mc_mode && (dst_win || mbar_win)),
        ("%t: *** %s DXA multicast needs a local address: wid=%0d smem=0x%0h meta=0x%0h cta_mask=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1, lane3_rs2))
    `RUNTIME_ASSERT(~(issue_fire && mc_mode && mc_rank_oob),
        ("%t: *** %s DXA multicast rank outside the cluster: wid=%0d cta_mask=0x%0h k=%0d",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane3_rs2, peer_row.k))
    `RUNTIME_ASSERT(~(issue_fire && mc_cluster && mc_mbar_oob),
        ("%t: *** %s mbarrier offset outside the block: wid=%0d meta=0x%0h stride=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane1_rs1, peer_stride))
    `RUNTIME_ASSERT(~(issue_fire && ~mbar_mode && is_mcast && (| row_other)),
        ("%t: *** %s DXA barrier multicast into a cluster spread over cores: wid=%0d cta_mask=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, dxa_req_data_in.cta_mask))
    `RUNTIME_ASSERT(~(issue_fire && ~mc_mode && (dst_oob || mbar_oob)),
        ("%t: *** %s DXA destination rank outside the cluster: wid=%0d smem=0x%0h meta=0x%0h k=%0d stride=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1, peer_row.k, peer_stride))
    `RUNTIME_ASSERT(~(issue_fire && ~mc_mode && ~dst_oob && ~mbar_oob && mbar_elsewhere),
        ("%t: *** %s DXA mbarrier must be in the destination block: wid=%0d smem=0x%0h meta=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1))
    `RUNTIME_ASSERT(~(issue_fire && dst_win && ~mbar_mode),
        ("%t: *** %s DXA cluster destination needs an mbarrier: wid=%0d smem=0x%0h meta=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1))
    `RUNTIME_ASSERT(~((bulk_fire || ld_fire) && mb_misalign),
        ("%t: *** %s DXA mbarrier must be 8-byte aligned: wid=%0d mbar=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane1_rs1))
    wire bk_ok0 = bulk_fire && ~mb_misalign;
    `RUNTIME_ASSERT(~(bk_ok0 && bk_align),
        ("%t: *** %s DXA bulk copy needs 16-byte size and alignment: wid=%0d dst=0x%0h src=0x%0h size=%0d",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane2_rs1, lane3_rs1))
    `RUNTIME_ASSERT(~(bk_ok0 && ~bk_align && bk_src_out),
        ("%t: *** %s DXA bulk copy source outside the block: wid=%0d src=0x%0h size=%0d stride=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane2_rs1, lane3_rs1, peer_stride))
    `RUNTIME_ASSERT(~(bk_ok0 && ~bk_align && ~bk_src_out && bk_rank),
        ("%t: *** %s DXA destination rank outside the cluster: wid=%0d dst=0x%0h mbar=0x%0h k=%0d stride=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1, peer_row.k, peer_stride))
    `RUNTIME_ASSERT(~(bk_ok0 && ~bk_align && ~bk_src_out && ~bk_rank && bk_self),
        ("%t: *** %s DXA bulk copy into the issuing block: wid=%0d dst=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1))
    `RUNTIME_ASSERT(~(bk_ok0 && ~bk_align && ~bk_src_out && ~bk_rank && ~bk_self && bk_mbar),
        ("%t: *** %s DXA mbarrier must be in the destination block: wid=%0d dst=0x%0h mbar=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1))
    `RUNTIME_ASSERT(~(bk_ok0 && ~bk_align && ~bk_src_out && ~bk_rank && ~bk_self && ~bk_mbar && bk_box),
        ("%t: *** %s DXA box exceeds the destination block: wid=%0d dst=0x%0h size=%0d stride=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane3_rs1, peer_stride))
    // Tensor loads and global-to-LMEM copies, with raw pointers.
    wire ld_ok0 = ld_fire && ~mb_misalign && ~(is_bload && gb_align);
    `RUNTIME_ASSERT(~((ld_fire && ~mb_misalign && is_bload && gb_align) || (bpf_fire && gb_align)),
        ("%t: *** %s DXA bulk copy needs 16-byte size and alignment: wid=%0d dst=0x%0h src=0x%0h size=%0d",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane2_rs1, lane3_rs1))
    `RUNTIME_ASSERT(~(ld_ok0 && mc_mode && (dst_win || bk_mb_win)),
        ("%t: *** %s DXA multicast needs a local address: wid=%0d dst=0x%0h mbar=0x%0h cta_mask=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1, mc_mask_w))
    `RUNTIME_ASSERT(~(ld_ok0 && mc_mode && mc_rank_oob),
        ("%t: *** %s DXA multicast rank outside the cluster: wid=%0d cta_mask=0x%0h k=%0d",
         $time, INSTANCE_ID, execute_if.data.header.wid, mc_mask_w, peer_row.k))
    `RUNTIME_ASSERT(~(ld_ok0 && mc_cluster && mc_mbar_oob),
        ("%t: *** %s mbarrier offset outside the block: wid=%0d mbar=0x%0h stride=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane1_rs1, peer_stride))
    `RUNTIME_ASSERT(~(ld_ok0 && ~mc_mode && bk_rank),
        ("%t: *** %s DXA destination rank outside the cluster: wid=%0d dst=0x%0h mbar=0x%0h k=%0d stride=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1, peer_row.k, peer_stride))
    `RUNTIME_ASSERT(~(ld_ok0 && ~mc_mode && ~bk_rank && bk_mbar),
        ("%t: *** %s DXA mbarrier must be in the destination block: wid=%0d dst=0x%0h mbar=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane1_rs1))
    `RUNTIME_ASSERT(~(ld_ok0 && is_bload && ~(mc_mode && (dst_win || bk_mb_win || mc_rank_oob || mc_mbar_oob))
                      && ~(~mc_mode && (bk_rank || bk_mbar)) && bk_box),
        ("%t: *** %s DXA box exceeds the destination block: wid=%0d dst=0x%0h size=%0d stride=0x%0h",
         $time, INSTANCE_ID, execute_if.data.header.wid, lane0_rs1, lane3_rs1, peer_stride))
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
            if (dxa_req_bus_if.req_data.mc_cluster) begin
                `TRACE(1, ("%t: %s dxa-req-mcast: wid=%0d, mask=0x%0h, off=0x%0h, mbar_rel=0x%0h\n",
                    $time, INSTANCE_ID, dxa_req_bus_if.req_data.wid, dxa_req_bus_if.req_data.mc_mask,
                    dxa_req_bus_if.req_data.smem_addr, dxa_req_bus_if.req_data.mbar_off))
            end
        `endif
        end
    end
`endif

endmodule
