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

// DLMEM aperture translation for one LSU block, one registered stage. Rank r
// of the window [DLMEM_BASE, +2^(LMEM_LOG + RANK_BITS)) maps to peer r of the
// issuing CTA's row: a peer on this core to the local memory window, any other
// to DLMEM_BASE | core << LMEM_LOG | offset, flagged on out_remote because that
// encoding can alias a global address once the core field outgrows the rank
// field. Out-of-bounds lanes go to the issuer's own slot base, keeping their
// 64-byte line offset like every other lane. A fence, and every packet of an
// AMO with .rl (a release mbarrier arrive included), waits at the stage output
// until its warp's remote accesses are performed (stores acknowledged, loads and
// AMOs answered) and its local accesses (loads, stores, AMOs, LR/SC, mbarrier
// ops) have committed in their banks, so it orders them before every later
// access of the warp. Earlier global accesses need a fence.
module VX_dlmem_xlate import VX_gpu_pkg::*, VX_dlmem_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    // Global core id (hart id core field); the core's index in its cluster is CORE_ID % DLMEM_NCC.
    parameter  CORE_ID       = 0,
    parameter  OOB_ASSERT    = 1,
    // For a core without the DLMEM network, where a remote lane has nowhere to go.
    parameter  REMOTE_ASSERT = 0,
    // Requests the stage holds; the core counts them against the LSU's outstanding reads.
    parameter  REQ_BUF_SIZE  = 2,

    localparam NUM_LANES    = `VX_CFG_NUM_LSU_LANES,
    localparam NW           = `VX_CFG_NUM_WARPS,
    localparam LMEM_LOG     = `VX_CFG_LMEM_LOG_SIZE,
    localparam CNT_W        = `CLOG2(NUM_LANES + 1)
) (
    input  wire                             clk,
    input  wire                             reset,

    VX_lsu_sched_if.slave                   in_if,
    VX_lsu_sched_if.master                  out_if,

    // Peer row of the incoming request's CTA slot.
    output wire [NW_WIDTH-1:0]              row_slot,
    input  dlmem_row_t                      row,
    input  wire [NW-1:0][LMEM_LOG-1:0]      slot_base,
    input  wire [LMEM_LOG:0]                stride,

    // Per warp, every remote access counted on wt_inc has been performed.
    input  wire [NW-1:0]                    drained,
    // Per warp, every local access counted on lacc_inc has committed.
    input  wire [NW-1:0]                    lmem_acc_drained,

    // Per-lane flags, qualified by out_if.req_valid.
    output wire [NUM_LANES-1:0]             out_remote,
    output wire [NUM_LANES-1:0]             out_oob,

    // Remote accesses leaving the stage, per warp.
    output wire                             wt_inc_valid,
    output wire [NW_WIDTH-1:0]              wt_inc_wid,
    output wire [CNT_W-1:0]                 wt_inc_cnt,

    // Local accesses leaving the stage, per warp.
    output wire                             lacc_inc_valid,
    output wire [NW_WIDTH-1:0]              lacc_inc_wid,
    output wire [CNT_W-1:0]                 lacc_inc_cnt,

    output wire                             empty
);
    localparam WA       = DLMEM_WA;
    localparam LMEM_AW  = DLMEM_LMEM_AW;
    localparam RB       = DLMEM_RANK_BITS;
    localparam RB_W     = `UP(RB);
    localparam WIN_LSB  = LMEM_AW + RB;
    localparam LINE_W   = 6 - WA;   // word bits of the 64-byte granule every kind keeps
    localparam MY_CORE  = CORE_ID % DLMEM_NCC;
    localparam PID_BITS = `CLOG2(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam HDR_W    = $bits(lsu_header_t);
    localparam ROW_W    = $bits(dlmem_row_t);
    localparam REQ_W    = $bits(lsu_req_data_t);

    localparam [LSU_ADDR_WIDTH-1:0] DLMEM_BASE_W = LSU_ADDR_WIDTH'(`VX_CFG_XLEN'(`VX_MEM_DLMEM_BASE_ADDR) >> WA);
    localparam [LSU_ADDR_WIDTH-1:0] LMEM_BASE_W  = LSU_ADDR_WIDTH'(`VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR) >> WA);

    `STATIC_ASSERT (WIN_LSB < LSU_ADDR_WIDTH, ("DLMEM window wider than the LSU address space"))
    `STATIC_ASSERT ((DLMEM_BASE_W & ((LSU_ADDR_WIDTH'(1) << (LMEM_AW + `MAX(RB, DLMEM_CORE_W))) - 1)) == 0,
        ("DLMEM base must be aligned to both the window and the remote core field"))
    `STATIC_ASSERT ((LMEM_BASE_W & ((LSU_ADDR_WIDTH'(1) << LMEM_AW) - 1)) == 0,
        ("LMEM base must be aligned to the local memory size"))
    // Slots are block-aligned, and the kept line offset must span exactly one block.
    `STATIC_ASSERT (`VX_CFG_MEM_BLOCK_SIZE == 64, ("DLMEM requires a 64-byte memory block"))
    `STATIC_ASSERT ((CORE_ID >> (HART_ID_BITS - NW_BITS - NT_BITS)) == 0,
        ("global core id does not fit the hart id core field"))
    `STATIC_ASSERT ($bits(dlmem_peer_t) == DLMEM_CORE_W + NW_WIDTH, ("invalid peer layout"))

    // Row select ahead of the register: the row is the one current when the request enters.

    lsu_header_t in_hdr;
    assign in_hdr   = in_if.req_data.tag[LSU_CLIENT_TAG_WIDTH-1 -: HDR_W];
    assign row_slot = NW_WIDTH'(in_hdr.cta_id);
    `UNUSED_VAR (in_hdr)

    lsu_req_data_t req_r;
    dlmem_row_t    row_r;
    wire           req_valid_r;
    wire           req_ready_r;

    VX_elastic_buffer #(
        .DATAW   (REQ_W + ROW_W),
        .SIZE    (REQ_BUF_SIZE),
        .OUT_REG (`TO_OUT_BUF_REG(3))
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (in_if.req_valid),
        .ready_in  (in_if.req_ready),
        .data_in   ({in_if.req_data, row}),
        .data_out  ({req_r, row_r}),
        .valid_out (req_valid_r),
        .ready_out (req_ready_r)
    );

    // Translation after the register drives the scheduler's request queue, which registers it.

    lsu_header_t hdr_r;
    assign hdr_r = req_r.tag[LSU_CLIENT_TAG_WIDTH-1 -: HDR_W];
    wire is_fence_r = req_r.tag[0];
    wire is_rl_r    = req_r.tag[1];
    wire [NW_WIDTH-1:0] my_slot = NW_WIDTH'(hdr_r.cta_id);
    wire [LMEM_AW-1:0] own_base = slot_base[my_slot][LMEM_LOG-1:WA];

    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0] addr_out;
    wire [NUM_LANES-1:0][MEM_ATTR_WIDTH-1:0] attr_out;
    wire [NUM_LANES-1:0][DLMEM_CORE_W-1:0]   lane_core;
    wire [NUM_LANES-1:0]                     lane_lacc;
`ifdef VX_CFG_EXT_MBAR_ENABLE
    wire [NUM_LANES-1:0]                     lane_mbar_global, lane_mbar_rwait;
`endif

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lanes
        wire [LSU_ADDR_WIDTH-1:0] a = req_r.addr[i];
        wire [RB_W-1:0] rank;
        if (RB != 0) begin : g_rank
            assign rank = a[LMEM_AW +: RB];
        end else begin : g_rank0
            assign rank = '0;
        end
        wire [LMEM_AW-1:0] offw = a[LMEM_AW-1:0];

        wire act  = req_r.mask[i] && ~is_fence_r;
        wire win  = act && (a[LSU_ADDR_WIDTH-1:WIN_LSB] == DLMEM_BASE_W[LSU_ADDR_WIDTH-1:WIN_LSB]);
        wire oob  = win && ((DLMEM_K_W'(rank) >= row_r.k) || ({1'b0, offw, WA'(0)} >= stride));

        dlmem_peer_t peer;
        assign peer = row_r.peers[rank];
        wire [LMEM_AW-1:0] peer_base = slot_base[peer.slot][LMEM_LOG-1:WA];
        wire [LMEM_AW-1:0] line_off = LMEM_AW'(offw[LINE_W-1:0]);
        wire [LMEM_AW-1:0] lw = oob ? (own_base + line_off) : (peer_base + offw);
        wire is_self = (peer.core == DLMEM_CORE_W'(MY_CORE));
        wire rem  = win && ~oob && ~is_self;

        wire [LSU_ADDR_WIDTH-1:0] rem_addr = DLMEM_BASE_W
                                           | (LSU_ADDR_WIDTH'(peer.core) << LMEM_AW)
                                           | LSU_ADDR_WIDTH'(lw);
        wire [LSU_ADDR_WIDTH-1:0] loc_addr = LMEM_BASE_W | LSU_ADDR_WIDTH'(lw);
        assign addr_out[i] = win ? (rem ? rem_addr : loc_addr) : a;

        // Lane i of packet pid carries warp thread pid * NUM_LANES + i.
        wire [HART_ID_WIDTH-1:0] lane_tid = (PID_BITS != 0)
            ? HART_ID_WIDTH'(hdr_r.pid * NUM_LANES + i)
            : HART_ID_WIDTH'(i);
        wire [HART_ID_WIDTH-1:0] hart_id = HART_ID_WIDTH'(
            (HART_ID_WIDTH'(CORE_ID) << (NW_BITS + NT_BITS))
          | (HART_ID_WIDTH'(hdr_r.wid) << NT_BITS)
          |  lane_tid
        );

        mem_bus_attr_t attr;
        always @(*) begin
            attr = mem_bus_attr_t'(req_r.attr[i]);
            attr.is_addr_remote = rem;
            if (win) begin
                attr.is_addr_local = ~rem;
                attr.is_addr_io    = 1'b0;
            end
            // A remote access carries its source warp to the target, a local
            // one to its bank, which reports the warp's accesses as committed.
            if (rem || (act && attr.is_addr_local)) begin
                attr.amo.hart_id = hart_id;
            end
            // VX_lmem_switch's local leg, every access: counted here and marked
            // for its bank, which reports the commit of marked accesses only.
            attr.is_lacc = act && attr.is_addr_local && ~rem;
        end
        assign attr_out[i] = attr;
        assign lane_lacc[i] = attr.is_lacc;
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        wire is_mbar = attr.amo.amo_valid && (attr.amo.amo_op >= AMO_OP_MBAR_ARRIVE);
        assign lane_mbar_global[i] = act && is_mbar && ~attr.is_addr_local && ~rem;
        assign lane_mbar_rwait[i]  = act && attr.amo.amo_valid && (attr.amo.amo_op >= AMO_OP_MBAR_TEST_WAIT) && rem;
    `endif

        assign out_remote[i] = rem;
        assign out_oob[i]    = oob;
        assign lane_core[i]  = peer.core;
    end

    // drained is registered after the count, so an access leaving in the cycle
    // before its fence is already visible to it.
    wire order_hold = (is_fence_r || is_rl_r) && ~(drained[hdr_r.wid] && lmem_acc_drained[hdr_r.wid]);

    assign out_if.req_valid       = req_valid_r && ~order_hold;
    assign out_if.req_data.rw     = req_r.rw;
    assign out_if.req_data.mask   = req_r.mask;
    assign out_if.req_data.byteen = req_r.byteen;
    assign out_if.req_data.addr   = addr_out;
    assign out_if.req_data.attr   = attr_out;
    assign out_if.req_data.data   = req_r.data;
    assign out_if.req_data.tag    = req_r.tag;
    assign req_ready_r            = out_if.req_ready && ~order_hold;

    assign in_if.rsp_valid  = out_if.rsp_valid;
    assign in_if.rsp_data   = out_if.rsp_data;
    assign out_if.rsp_ready = in_if.rsp_ready;

    wire out_fire = out_if.req_valid && out_if.req_ready;

    // A fence does not wait for responses, so loads and AMOs are tracked as stores are.
    assign wt_inc_valid = out_fire && (| out_remote);
    assign wt_inc_wid   = hdr_r.wid;
    `POP_COUNT(wt_inc_cnt, out_remote);

    assign lacc_inc_valid = out_fire && (| lane_lacc);
    assign lacc_inc_wid   = hdr_r.wid;
    `POP_COUNT(lacc_inc_cnt, lane_lacc);

    assign empty = ~req_valid_r;

    `UNUSED_VAR (hdr_r)

    // The first out-of-bounds and remote lanes, for the asserts' messages only.
    reg [`LOG2UP(NUM_LANES)-1:0] oob_lane, rem_lane;
    always @(*) begin
        oob_lane = '0;
        rem_lane = '0;
        for (integer i = NUM_LANES-1; i >= 0; --i) begin
            if (out_oob[i]) begin
                oob_lane = `LOG2UP(NUM_LANES)'(i);
            end
            if (out_remote[i]) begin
                rem_lane = `LOG2UP(NUM_LANES)'(i);
            end
        end
    end
    wire [LSU_ADDR_WIDTH-1:0] oob_addr = req_r.addr[oob_lane];
    wire [LSU_ADDR_WIDTH-1:0] rem_win_addr = req_r.addr[rem_lane];

    `RUNTIME_ASSERT (~((OOB_ASSERT != 0) && out_fire && (| out_oob)),
        ("%t: %s DLMEM access out of bounds: wid=%0d cta=%0d lane=%0d addr=0x%0h (#%0d)",
            $time, INSTANCE_ID, hdr_r.wid, my_slot, oob_lane, {oob_addr, WA'(0)}, hdr_r.uuid))

    `RUNTIME_ASSERT (~((REMOTE_ASSERT != 0) && out_fire && (| out_remote)),
        ("%t: %s remote DLMEM access is not supported yet: wid=%0d cta=%0d lane=%0d addr=0x%0h core=%0d (#%0d)",
            $time, INSTANCE_ID, hdr_r.wid, my_slot, rem_lane, {rem_win_addr, WA'(0)}, lane_core[rem_lane], hdr_r.uuid))

`ifdef VX_CFG_EXT_MBAR_ENABLE
    // An mbarrier lives in a local memory: the banks are the only place its ops run.
    `RUNTIME_ASSERT (~(out_fire && (| lane_mbar_global)),
        ("%t: %s mbarrier op on a global address: wid=%0d lanes=0x%0h (#%0d)",
            $time, INSTANCE_ID, hdr_r.wid, lane_mbar_global, hdr_r.uuid))
    // A wait reads its own block's object: the waiter table that releases a
    // suspended try_wait only sees its own core's banks.
    `RUNTIME_ASSERT (~(out_fire && (| lane_mbar_rwait)),
        ("%t: %s remote mbarrier wait is not supported: wid=%0d lanes=0x%0h (#%0d)",
            $time, INSTANCE_ID, hdr_r.wid, lane_mbar_rwait, hdr_r.uuid))
`endif

endmodule
