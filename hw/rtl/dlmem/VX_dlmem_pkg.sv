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
    // A warp's unacknowledged remote stores: the egress credits plus what the LSU queues can still hold.
    localparam DLMEM_WT_MAX        = `VX_CFG_DLMEM_STORE_CREDITS + (`VX_CFG_LSU_QUEUE_IN_SIZE + 3) * `VX_CFG_NUM_LSU_LANES;
    localparam DLMEM_WT_W          = `CLOG2(DLMEM_WT_MAX + 1);

    localparam [1:0] DLMEM_OP_LD  = 2'd0;
    localparam [1:0] DLMEM_OP_ST  = 2'd1;
    localparam [1:0] DLMEM_OP_AMO = 2'd2;

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
        logic [DLMEM_TAG_W-1:0]    tag;          // LD/AMO: source pending index; ST: source warp id
    } dlmem_req_pkt_t;

    // Response network packet; the fabric dst is the requesting core.
    typedef struct packed {
        logic                      ack;          // 1: store committed, data unused; 0: LD/AMO data
        logic [DLMEM_TAG_W-1:0]    tag;
        logic [`VX_CFG_XLEN-1:0]   data;         // full bank word, as VX_local_mem returns it
    } dlmem_rsp_pkt_t;

    localparam DLMEM_REQ_DATAW = $bits(dlmem_req_pkt_t);
    localparam DLMEM_RSP_DATAW = $bits(dlmem_rsp_pkt_t);

endpackage

`IGNORE_UNUSED_END

`endif // VX_DLMEM_PKG_VH
