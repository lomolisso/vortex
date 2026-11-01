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

`ifndef VX_DLMEM_PKG_VH
`define VX_DLMEM_PKG_VH

`include "VX_define.vh"

`IGNORE_UNUSED_BEGIN

package VX_dlmem_pkg;

    import VX_gpu_pkg::*;

    localparam DLMEM_WA            = `CLOG2(LSU_WORD_SIZE);
    localparam DLMEM_LMEM_AW       = `VX_CFG_LMEM_LOG_SIZE - DLMEM_WA;   // word address in a core's local memory
    localparam DLMEM_PEND_W        = `UP(`CLOG2(`VX_CFG_DLMEM_PENDING_SIZE));
    localparam DLMEM_TAG_W         = `MAX(DLMEM_PEND_W, NW_WIDTH);
    localparam DLMEM_INGRESS_SLOTS = 4;
    localparam DLMEM_SLOT_W        = `UP(`CLOG2(DLMEM_INGRESS_SLOTS));
    // A warp's unperformed remote accesses: the egress credits and pending entries plus what the LSU queues can still hold.
    localparam DLMEM_WT_MAX        = `VX_CFG_DLMEM_STORE_CREDITS + `VX_CFG_DLMEM_PENDING_SIZE
                                   + (`VX_CFG_LSU_QUEUE_IN_SIZE + 3) * `VX_CFG_NUM_LSU_LANES;
    localparam DLMEM_WT_W          = `CLOG2(DLMEM_WT_MAX + 1);
    // A warp's local accesses between translation and their bank commit: what the
    // LSU queues and the per-lane buffers can hold, then per bank the local memory's
    // request crossbar, an atomic in its read and write-back stages, and the registered
    // commit. Each of the crossbar's arbiters buffers two entries, and VX_stream_arb
    // splits more than 1.5 * MAX_FANOUT inputs into groups of MAX_FANOUT under a joining
    // arbiter: G1 groups of the LSU lanes and the DLMEM ingress, and G2 groups of those
    // when G1 is too wide in turn.
    localparam DLMEM_LACC_XB_IN    = `VX_CFG_NUM_LSU_BLOCKS * `VX_CFG_NUM_LSU_LANES + 1;
    localparam DLMEM_LACC_XB_G1    = (DLMEM_LACC_XB_IN > (`MAX_FANOUT + `MAX_FANOUT / 2))
                                   ? `CDIV(DLMEM_LACC_XB_IN, `MAX_FANOUT) : 0;
    localparam DLMEM_LACC_XB_G2    = (DLMEM_LACC_XB_G1 > (`MAX_FANOUT + `MAX_FANOUT / 2))
                                   ? `CDIV(DLMEM_LACC_XB_G1, `MAX_FANOUT) : 0;
    localparam DLMEM_LACC_MAX      = (`VX_CFG_LSU_QUEUE_IN_SIZE + 3) * `VX_CFG_NUM_LSU_LANES
                                   + (5 + 2 * (DLMEM_LACC_XB_G1 + DLMEM_LACC_XB_G2)) * `VX_CFG_LMEM_NUM_BANKS;
    localparam DLMEM_LACC_W        = `CLOG2(DLMEM_LACC_MAX + 1);

    localparam [1:0] DLMEM_OP_LD  = 2'd0;
    localparam [1:0] DLMEM_OP_ST  = 2'd1;
    localparam [1:0] DLMEM_OP_AMO = 2'd2;
`ifdef DXA_MBAR_ENABLE
    // A posted row flit of a remote DXA copy: amo_unsigned 0 is a word of
    // data, 1 the copy's completion (a complete_tx on addr, data = box bytes).
    // tag carries the destination CTA slot, and nothing answers either kind.
    localparam [1:0] DLMEM_OP_DXA = 2'd3;
`endif
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // A translated st.async/red.async lane leaves for the egress with its
    // mbarrier word address above the core field and the top address bit set;
    // no real remote address reaches that bit.
    localparam DLMEM_ASYNC_BIT    = LSU_ADDR_WIDTH - 1;
    localparam DLMEM_ASYNC_MB_LSB = DLMEM_LMEM_AW + DLMEM_CORE_W;
`endif

    // Request network packet; the fabric dst is the target core.
    typedef struct packed {
        logic [DLMEM_CORE_W-1:0]   src;          // requesting core, the response's dst
        logic [1:0]                op;           // DLMEM_OP_*
        logic [DLMEM_LMEM_AW-1:0]  addr;         // word address in the target's local memory
        logic [LSU_WORD_SIZE-1:0]  byteen;
        logic [`VX_CFG_XLEN-1:0]   data;         // store data or AMO rhs, lane-positioned as the LSU sends it
        amo_op_e                   amo_op;
        logic                      amo_unsigned;
        logic [HART_ID_WIDTH-1:0]  hart_id;      // a remote store or AMO breaks other harts' reservations
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        logic [`VX_CFG_XLEN-1:0]   amo_cmp;
    `endif
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        // st.async/red.async: after the access, complete_tx(bytes) on this mbarrier word.
        logic                      mbar_valid;
        logic [DLMEM_LMEM_AW-1:0]  mbar_addr;
    `endif
        logic [DLMEM_TAG_W-1:0]    tag;          // LD/AMO: source pending index; ST: source warp id
    } dlmem_req_pkt_t;

    // Response network packet; the fabric dst is the requesting core.
    typedef struct packed {
        logic                      ack;          // 1: store committed, data unused (with MBAR, bit 0 marks an async access); 0: LD/AMO data
        logic [DLMEM_TAG_W-1:0]    tag;
        logic [`VX_CFG_XLEN-1:0]   data;         // full bank word, as VX_local_mem returns it
    } dlmem_rsp_pkt_t;

    localparam DLMEM_REQ_DATAW = $bits(dlmem_req_pkt_t);
    localparam DLMEM_RSP_DATAW = $bits(dlmem_rsp_pkt_t);

endpackage

`IGNORE_UNUSED_END

`endif // VX_DLMEM_PKG_VH
