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

module VX_lsu_slice import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    // Inputs
    VX_execute_if.slave     execute_if,

    // Outputs
    VX_result_if.master     result_if,

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // Warps the scheduler activated this cycle.
    input wire [`VX_CFG_NUM_WARPS-1:0] warp_spawn,
    // Releases the warp an acquire AMO stalled at decode.
    VX_sched_unlock_if.master sched_unlock_if,
`endif
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // The last response of an mbarrier try_wait left the slice.
    output wire                       mbar_retire_valid,
    output wire [NW_WIDTH-1:0]        mbar_retire_wid,
    // cp.async: the copies' local-memory writes and arrives, and the release of a wait_group.
    VX_lsu_mem_if.master              cpa_lmem_if,
    VX_sched_unlock_if.master         cpa_unlock_if,
    // Per warp, no cp.async work left in the unit.
    output wire [`VX_CFG_NUM_WARPS-1:0] cpa_idle,
`endif

    // The slice exposes the memory-client side as a VX_lsu_sched_if master;
    // the parent wires it into VX_lsu_scheduler.
    VX_lsu_sched_if.master  client_if
);
    localparam NUM_LANES    = `VX_CFG_NUM_LSU_LANES;
    localparam PID_BITS     = `CLOG2(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam LSUQ_SIZEW   = `LOG2UP(`VX_CFG_LSU_PENDING_SIZE);
    localparam REQ_ASHIFT   = `CLOG2(LSU_WORD_SIZE);
    localparam MEM_ASHIFT   = `CLOG2(`VX_CFG_MEM_BLOCK_SIZE);
    localparam MEM_ADDRW    = `VX_CFG_MEM_ADDR_WIDTH - MEM_ASHIFT;
    // Window membership is an equality on the block-address bits above the
    // window: the LMEM base is asserted aligned to the capacity, so the prefix
    // compare is exact and needs no end address -- which is not representable
    // when the window ends at the top of the address space.
    localparam LMEM_BLK_LSB = `VX_CFG_LMEM_LOG_SIZE - MEM_ASHIFT;
    localparam [MEM_ADDRW-1:0] LMEM_BLK_BASE = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_LMEM_BASE_ADDR) >> MEM_ASHIFT);
    `STATIC_ASSERT (MEM_ADDRW > LMEM_BLK_LSB, ("LMEM window wider than the block address space"))
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    localparam DLMEM_WIN_BLK_LSB = `VX_CFG_LMEM_LOG_SIZE + DLMEM_RANK_BITS - MEM_ASHIFT;
    localparam [MEM_ADDRW-1:0] DLMEM_WIN_BLK_BASE = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_DLMEM_BASE_ADDR) >> MEM_ASHIFT);
    `STATIC_ASSERT (MEM_ADDRW > DLMEM_WIN_BLK_LSB, ("DLMEM window wider than the block address space"))
`endif
    `UNUSED_PARAM (CORE_ID)
    `UNUSED_SPARAM (INSTANCE_ID)

`ifdef VX_CFG_EXT_MBAR_ENABLE
    // tag_width = header + op_type + align + pkt_addr + gload + try_wait + cp.async + aq + rl + fence
    localparam TAG_WIDTH = $bits(lsu_header_t) + INST_LSU_BITS + (NUM_LANES * REQ_ASHIFT) + LSUQ_SIZEW + 1 + 1 + 1 + 2 + 1;
`elsif VX_CFG_EXT_DLMEM_ENABLE
    // tag_width = header + op_type + align + pkt_addr + gload + aq + rl + fence
    localparam TAG_WIDTH = $bits(lsu_header_t) + INST_LSU_BITS + (NUM_LANES * REQ_ASHIFT) + LSUQ_SIZEW + 1 + 2 + 1;
`else
    // tag_width = header + op_type + align + pkt_addr + fence
    localparam TAG_WIDTH = $bits(lsu_header_t) + INST_LSU_BITS + (NUM_LANES * REQ_ASHIFT) + LSUQ_SIZEW + 1;
`endif
    `STATIC_ASSERT (TAG_WIDTH == LSU_CLIENT_TAG_WIDTH, ("LSU client tag width mismatch — pkg/slice out of sync"))
`ifdef VX_CFG_EXT_A_ENABLE
    `STATIC_ASSERT ((CORE_ID >> (HART_ID_BITS - NW_BITS - NT_BITS)) == 0, ("global core id does not fit the hart id core field"))
`endif

    VX_result_if #(
        .data_t (lsu_result_t)
    ) result_rsp_if();

    VX_result_if #(
        .data_t (lsu_result_t)
    ) result_no_rsp_if();

    // The third operand slot is read only by compare-and-swap, which needs
    // the atomic extension to exist at all.
`ifdef VX_CFG_EXT_A_ENABLE
`ifndef VX_CFG_EXT_ZACAS_ENABLE
    `UNUSED_VAR (execute_if.data.rs3_data)
`endif
`else
    `UNUSED_VAR (execute_if.data.rs3_data)
`endif

    // full address calculation — per-lane AGU (all address forms live in
    // VX_lsu_agu; this slice contains no address arithmetic).

    wire req_is_fence, rsp_is_fence;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    wire req_is_cfence = (execute_if.data.op_type == INST_LSU_CFENCE);
`endif

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] full_addr;
`ifdef VX_CFG_EXT_MBAR_ENABLE
    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] agu_addr;
`endif
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_full_addr
        VX_lsu_agu lsu_agu (
            .base   (execute_if.data.rs1_data[i]),
            .stride (execute_if.data.rs2_data[i]),
            .offset (execute_if.data.op_args.lsu.offset),
            .pack   (execute_if.data.op_args.lsu.pack),
        `ifdef VX_CFG_EXT_MBAR_ENABLE
            .addr   (agu_addr[i])
        `else
            .addr   (full_addr[i])
        `endif
        );
    end

`ifdef VX_CFG_EXT_MBAR_ENABLE
    // cp.async: is_async without is_store. A copy is a global read that retires
    // at issue; its response goes to the cp.async unit, which writes the local
    // destination. Commit, wait and the arrives send nothing to memory.
    wire [INST_LSU_BITS-1:0] req_op = INST_LSU_BITS'(execute_if.data.op_type);
    wire req_cpa      = execute_if.data.op_args.lsu.is_async && ~execute_if.data.op_args.lsu.is_store;
    wire req_cpa4     = (req_op == INST_LSU_CPA4);
    wire req_cpa16    = (req_op == INST_LSU_CPA16H);
    wire req_cpa_copy = req_cpa && (req_cpa4 || (req_op == INST_LSU_CPA8) || req_cpa16);
    wire req_cpa_nomem = req_cpa && ~req_cpa_copy;
    wire req_cpa_arr  = req_cpa && ((req_op == INST_LSU_CPA_ARRIVE) || (req_op == INST_LSU_CPA_NOINC));
    // The high half of a 16-byte copy: offset 8 on source and destination.
    wire req_cpa_hi   = req_cpa16 && (execute_if.data.op_args.lsu.offset == 12'd8);

    wire [CPA_OP_BITS-1:0] cpa_op = req_cpa_copy ? CPA_OP_COPY
                                  : (req_op == INST_LSU_CPA_COMMIT) ? CPA_OP_COMMIT
                                  : (req_op == INST_LSU_CPA_WAIT)   ? CPA_OP_WAIT
                                                                    : CPA_OP_ARRIVE;

    localparam [`VX_CFG_XLEN-1:0] CPA_SAFE_ADDR = `VX_CFG_XLEN'(`VX_MEM_USER_BASE_ADDR);

    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] cpa_dst;
    wire [NUM_LANES-1:0][3:0]              cpa_ss;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cpa_lane
        wire [`VX_CFG_XLEN-1:0] ss = execute_if.data.rs2_data[i];
        // Each half of a 16-byte copy takes its share: low min(ss, 8), high max(ss - 8, 0).
        wire [3:0] ss_cap = req_cpa4 ? 4'd4 : 4'd8;
        wire [`VX_CFG_XLEN-1:0] ss_hi = (ss > `VX_CFG_XLEN'(8)) ? (ss - `VX_CFG_XLEN'(8)) : '0;
        wire [`VX_CFG_XLEN-1:0] ss_u  = req_cpa_hi ? ss_hi : ss;
        assign cpa_ss[i]  = (ss_u > `VX_CFG_XLEN'(ss_cap)) ? ss_cap : ss_u[3:0];
        assign cpa_dst[i] = execute_if.data.rs3_data[i] + `VX_CFG_XLEN'(execute_if.data.op_args.lsu.offset);
        // A lane with nothing to read reads a fixed global word instead of its source.
        assign full_addr[i] = (req_cpa_copy && (cpa_ss[i] == 4'd0)) ? CPA_SAFE_ADDR : agu_addr[i];
    end
`endif

    // address type + AMO classification — per-lane mem_bus_attr_t
    // carried through the scheduler's USER channel and out via the
    // lsu_mem_if's per-lane req_data.user field.

    mem_bus_attr_t [NUM_LANES-1:0] mem_req_attr_struct;
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    wire [NUM_LANES-1:0] lane_global;
`endif
    wire [NUM_LANES-1:0][MEM_ATTR_WIDTH-1:0] mem_req_attr;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_attr
        wire [MEM_ADDRW-1:0] block_addr = full_addr[i][MEM_ASHIFT +: MEM_ADDRW];
        // is I/O address
        wire [MEM_ADDRW-1:0] io_addr_start = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_IO_BASE_ADDR) >> MEM_ASHIFT);
        wire [MEM_ADDRW-1:0] io_addr_end = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_IO_END_ADDR) >> MEM_ASHIFT);
        assign mem_req_attr_struct[i].is_flush  = req_is_fence;
        assign mem_req_attr_struct[i].is_fence  = req_is_fence;
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        // The DLMEM translation stage sets them: lanes bound for a peer core, counted local lanes.
        assign mem_req_attr_struct[i].is_addr_remote = 1'b0;
        assign mem_req_attr_struct[i].is_lacc        = 1'b0;
        assign mem_req_attr_struct[i].is_fence_cta   = req_is_cfence;
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        // OM fragment-export aperture.
        //
        // An aperture store is presented to the cache hierarchy as an ORDINARY IO
        // STORE: is_addr_io is asserted alongside is_addr_om. That is not a hack —
        // uncached, bypassed and posted is exactly the behaviour an export wants,
        // and it means no cache level needs to know the aperture exists. The
        // caches carry req_data.attr through verbatim without decoding it, so the
        // is_addr_om bit rides along untouched and the cluster's OM steer peels
        // the store off the L1->L2 trunk on that bit alone.
        wire [MEM_ADDRW-1:0] om_addr_start = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_OM_BASE_ADDR) >> MEM_ASHIFT);
        wire [MEM_ADDRW-1:0] om_addr_end = MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_OM_END_ADDR) >> MEM_ASHIFT);
        wire lane_is_om = (block_addr >= om_addr_start) && (block_addr < om_addr_end);
        assign mem_req_attr_struct[i].is_addr_om = lane_is_om;
    `else
        wire lane_is_om = 1'b0;
        assign mem_req_attr_struct[i].is_addr_om = 1'b0;
    `endif
        assign mem_req_attr_struct[i].is_addr_io = ((block_addr >= io_addr_start) && (block_addr < io_addr_end))
                                                 || lane_is_om;
    `ifdef VX_CFG_LMEM_ENABLE
        // is local memory address
        assign mem_req_attr_struct[i].is_addr_local =
            (block_addr[MEM_ADDRW-1:LMEM_BLK_LSB] == LMEM_BLK_BASE[MEM_ADDRW-1:LMEM_BLK_LSB]);
    `else
        assign mem_req_attr_struct[i].is_addr_local = 1'b0;
    `endif
    `ifdef VX_CFG_EXT_DLMEM_ENABLE
        // The translation's window predicate: every window lane becomes local or remote.
        wire lane_in_win = (block_addr[MEM_ADDRW-1:DLMEM_WIN_BLK_LSB] == DLMEM_WIN_BLK_BASE[MEM_ADDRW-1:DLMEM_WIN_BLK_LSB]);
        // Stacks are thread-private: a CTA release has nothing to order there.
        wire lane_in_stack = (full_addr[i] >= STACK_WINDOW_BOTTOM) && (full_addr[i] < STACK_WINDOW_TOP);
        assign lane_global[i] = execute_if.data.header.tmask[i] && ~mem_req_attr_struct[i].is_addr_local
                             && ~lane_in_win && ~lane_in_stack;
    `endif
    `ifdef VX_CFG_EXT_A_ENABLE
        amo_req_t lane_amo;
        assign lane_amo.amo_valid    = execute_if.data.op_args.lsu.amo_valid;
        assign lane_amo.amo_op       = execute_if.data.op_args.lsu.amo_op;
        // amo_unsigned distinguishes signed/unsigned AMOMIN/MAX variants
        // (decoder collapses MINU/MAXU into MIN/MAX + this bit).
        assign lane_amo.amo_unsigned = execute_if.data.op_args.lsu.amo_unsigned;
    `ifdef VX_CFG_EXT_ZACAS_ENABLE
        // Compare-and-swap comparand. The decoder routes rd into the third
        // operand slot for it; every other atomic leaves the slot unread.
        assign lane_amo.amo_cmp      = execute_if.data.rs3_data[i];
    `endif
        // make_hart_id(cid, wid, tid) — packed concatenation, low bits = tid.
        // Lane i of packet pid carries warp thread pid * NUM_LANES + i.
        wire [HART_ID_WIDTH-1:0] lane_tid = (PID_BITS != 0)
            ? HART_ID_WIDTH'(execute_if.data.header.pid * NUM_LANES + i)
            : HART_ID_WIDTH'(i);
        assign lane_amo.hart_id      = HART_ID_WIDTH'(
            (HART_ID_WIDTH'(CORE_ID) << (NW_BITS + NT_BITS))
          | (HART_ID_WIDTH'(execute_if.data.header.wid) << NT_BITS)
          |  lane_tid
        );
        assign mem_req_attr_struct[i].amo = lane_amo;
    `else
        // EXT_A disabled: tie the AMO sideband to zero so the bits don't
        // propagate as 'x under --x-assign unique.
        assign mem_req_attr_struct[i].amo = '0;
    `endif
        assign mem_req_attr[i] = mem_req_attr_struct[i];
    end

    // schedule memory request

    wire                            mem_req_valid;
    wire [NUM_LANES-1:0]            mem_req_mask;
    wire                            mem_req_rw;
    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0] mem_req_addr;
    wire [NUM_LANES-1:0][LSU_WORD_SIZE-1:0] mem_req_byteen;
    reg  [NUM_LANES-1:0][LSU_WORD_SIZE*8-1:0] mem_req_data;
    wire [TAG_WIDTH-1:0]            mem_req_tag;
    wire                            mem_req_ready;

    wire                            mem_rsp_valid;
    wire [NUM_LANES-1:0]            mem_rsp_mask;
    wire [NUM_LANES-1:0][LSU_WORD_SIZE*8-1:0] mem_rsp_data;
    wire [TAG_WIDTH-1:0]            mem_rsp_tag;
    wire                            mem_rsp_sop;
    wire                            mem_rsp_eop;
    wire                            mem_rsp_ready;

    wire mem_req_fire = mem_req_valid && mem_req_ready;
    wire mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;

    wire mem_rsp_sop_pkt, mem_rsp_eop_pkt;
    wire no_rsp_buf_valid, no_rsp_buf_ready;
`ifdef VX_CFG_EXT_MBAR_ENABLE
    wire rsp_cpa;
`endif

    wire [LSUQ_SIZEW-1:0] pkt_waddr, pkt_raddr;

    // fence handling

    reg fence_lock;

    assign req_is_fence = inst_lsu_is_fence(execute_if.data.op_type);

    always @(posedge clk) begin
        if (reset) begin
            fence_lock <= 0;
        end else begin
            if (mem_req_fire && req_is_fence && execute_if.data.header.eop) begin
                fence_lock <= 1;
            end
            if (mem_rsp_fire && rsp_is_fence && mem_rsp_eop_pkt) begin
                fence_lock <= 0;
            end
        end
    end

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // Per warp: a global write was sent since the warp's last fence answered, and
    // the global read packets still waiting for their response. A CTA fence waits
    // for those reads, then drains the dcache only if the warp wrote globally.
    localparam GLOAD_W = `CLOG2(`VX_CFG_LSU_QUEUE_IN_SIZE + `VX_CFG_LSU_PENDING_SIZE + 2 + 1) + PID_BITS;

    reg [`VX_CFG_NUM_WARPS-1:0] gdirty;
    reg [`VX_CFG_NUM_WARPS-1:0][GLOAD_W-1:0] gload;

    wire req_has_global = (| lane_global) && ~req_is_fence;
`ifdef VX_CFG_EXT_A_ENABLE
    wire req_writes = (execute_if.data.op_args.lsu.is_store && ~execute_if.data.op_args.lsu.amo_valid)
                   || (execute_if.data.op_args.lsu.amo_valid && (execute_if.data.op_args.lsu.amo_op != AMO_OP_LR));
`else
    wire req_writes = execute_if.data.op_args.lsu.is_store;
`endif
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // cp.async reads are ordered by their groups and mbarriers, not by fences.
    wire req_gload = req_has_global && ~mem_req_rw && ~req_cpa;
`else
    wire req_gload = req_has_global && ~mem_req_rw;
`endif
    wire rsp_gload;

    wire cfence_hold = req_is_cfence && (gload[execute_if.data.header.wid] != '0);
    wire cfence_skip = req_is_cfence && ~gdirty[execute_if.data.header.wid];
`ifdef VX_CFG_EXT_MBAR_ENABLE
    wire req_skip = (req_is_fence && (~execute_if.data.header.eop || cfence_skip)) || req_cpa_nomem;
`else
    wire req_skip = req_is_fence && (~execute_if.data.header.eop || cfence_skip);
`endif
`else
    wire req_skip = req_is_fence && ~execute_if.data.header.eop;
`endif
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // A copy retires at issue like a store.
    wire no_rsp_buf_enable = (mem_req_rw && ~execute_if.data.header.wb) || req_skip || req_cpa_copy;
`else
    wire no_rsp_buf_enable = (mem_req_rw && ~execute_if.data.header.wb) || req_skip;
`endif

`ifdef VX_CFG_EXT_MBAR_ENABLE
    // The cp.async unit stalls the instruction it cannot take yet.
    wire cpa_iss_ready;
    wire cpa_stall = req_cpa && ~cpa_iss_ready;

    assign mem_req_valid = execute_if.valid
                        && ~req_skip
                        && ~(no_rsp_buf_enable && ~no_rsp_buf_ready)
                        && ~fence_lock
                        && ~cfence_hold
                        && ~cpa_stall;

    assign no_rsp_buf_valid = execute_if.valid
                           && no_rsp_buf_enable
                           && (req_skip || mem_req_ready)
                           && ~fence_lock
                           && ~cfence_hold
                           && ~cpa_stall;

    wire execute_ready_base = (mem_req_ready || req_skip)
                           && ~(no_rsp_buf_enable && ~no_rsp_buf_ready)
                           && ~fence_lock
                           && ~cfence_hold;

    assign execute_if.ready = execute_ready_base && ~cpa_stall;

`elsif VX_CFG_EXT_DLMEM_ENABLE
    assign mem_req_valid = execute_if.valid
                        && ~req_skip
                        && ~(no_rsp_buf_enable && ~no_rsp_buf_ready)
                        && ~fence_lock
                        && ~cfence_hold;

    assign no_rsp_buf_valid = execute_if.valid
                           && no_rsp_buf_enable
                           && (req_skip || mem_req_ready)
                           && ~fence_lock
                           && ~cfence_hold;

    assign execute_if.ready = (mem_req_ready || req_skip)
                           && ~(no_rsp_buf_enable && ~no_rsp_buf_ready)
                           && ~fence_lock
                           && ~cfence_hold;

`else
    assign mem_req_valid = execute_if.valid
                        && ~req_skip
                        && ~(no_rsp_buf_enable && ~no_rsp_buf_ready)
                        && ~fence_lock;

    assign no_rsp_buf_valid = execute_if.valid
                           && no_rsp_buf_enable
                           && (req_skip || mem_req_ready)
                           && ~fence_lock;

    assign execute_if.ready = (mem_req_ready || req_skip)
                           && ~(no_rsp_buf_enable && ~no_rsp_buf_ready)
                           && ~fence_lock;
`endif

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // A CTA fence token rides every lane, so each dcache channel carries a copy
    // behind the warp's older requests on it: the channels reach the dcache
    // independently, and its drain covers only the requests it has accepted.
    assign mem_req_mask = req_is_cfence ? {NUM_LANES{1'b1}} : execute_if.data.header.tmask;
`else
    assign mem_req_mask = execute_if.data.header.tmask;
`endif
`ifdef VX_CFG_EXT_A_ENABLE
    // AMO MemReq.rw is unconditionally 0: a missing line under SC must
    // miss-and-return-failure, not write-and-succeed.
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // A red.async is posted like a store; only its peer's ingress performs the add.
    assign mem_req_rw = execute_if.data.op_args.lsu.is_store
                     && (~execute_if.data.op_args.lsu.amo_valid || execute_if.data.op_args.lsu.is_async);
`else
    assign mem_req_rw = execute_if.data.op_args.lsu.is_store
                     && ~execute_if.data.op_args.lsu.amo_valid;
`endif
`else
    assign mem_req_rw = execute_if.data.op_args.lsu.is_store;
`endif

    // address formatting

    wire [NUM_LANES-1:0][REQ_ASHIFT-1:0] req_align;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_addr
        assign req_align[i] = full_addr[i][REQ_ASHIFT-1:0];
        assign mem_req_addr[i] = full_addr[i][`VX_CFG_MEM_ADDR_WIDTH-1:REQ_ASHIFT];
    end

    // byte enable formatting
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_byteen_w
        reg [LSU_WORD_SIZE-1:0] mem_req_byteen_w;
        always @(*) begin
            mem_req_byteen_w = '0;
            case (inst_lsu_wsize(execute_if.data.op_type))
                0: begin // 8-bit
                    mem_req_byteen_w[req_align[i]] = 1'b1;
                end
                1: begin // 16 bit
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:1], 1'b0}] = 1'b1;
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:1], 1'b1}] = 1'b1;
                end
            `ifdef VX_CFG_XLEN_64
                2: begin // 32 bit
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:2], 2'b00}] = 1'b1;
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:2], 2'b01}] = 1'b1;
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:2], 2'b10}] = 1'b1;
                    mem_req_byteen_w[{req_align[i][REQ_ASHIFT-1:2], 2'b11}] = 1'b1;
                end
            `endif
                // 3: 64 bit
                default : mem_req_byteen_w = {LSU_WORD_SIZE{1'b1}};
            endcase
        end
        assign mem_req_byteen[i] = mem_req_byteen_w;
    end

    // memory misalignment not supported!
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_missalign
        wire lsu_req_fire = execute_if.valid && execute_if.ready;
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        // cp.async lanes have their own checks, whose texts must win.
        `RUNTIME_ASSERT((~lsu_req_fire || ~execute_if.data.header.tmask[i] || req_is_fence || req_cpa || (full_addr[i] % (1 << inst_lsu_wsize(execute_if.data.op_type))) == 0),
            ("%t: misaligned memory access, wid=%0d, PC=0x%0h, addr=0x%0h, wsize=%0d! (#%0d)",
                $time, execute_if.data.header.wid, to_fullPC(execute_if.data.header.PC), full_addr[i], inst_lsu_wsize(execute_if.data.op_type), execute_if.data.header.uuid))
    `else
        `RUNTIME_ASSERT((~lsu_req_fire || ~execute_if.data.header.tmask[i] || req_is_fence || (full_addr[i] % (1 << inst_lsu_wsize(execute_if.data.op_type))) == 0),
            ("%t: misaligned memory access, wid=%0d, PC=0x%0h, addr=0x%0h, wsize=%0d! (#%0d)",
                $time, execute_if.data.header.wid, to_fullPC(execute_if.data.header.PC), full_addr[i], inst_lsu_wsize(execute_if.data.op_type), execute_if.data.header.uuid))
    `endif
    end

`ifdef VX_CFG_EXT_MBAR_ENABLE
    // A 16-byte st.async reaches the slice as two 8-byte halves; only the high one has offset 8.
    wire execute_fire = execute_if.valid && execute_if.ready;
    wire req_sq_hi    = execute_if.data.op_args.lsu.is_async && execute_if.data.op_args.lsu.is_store
                     && (execute_if.data.op_args.lsu.offset == 12'd8);
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_sq_align
        `RUNTIME_ASSERT(~(execute_fire && req_sq_hi && execute_if.data.header.tmask[i] && (full_addr[i][3:0] != 4'd8)),
            ("%t: st.async 16-byte form needs a 16-byte aligned address: wid=%0d, PC=0x%0h, addr=0x%0h (#%0d)",
                $time, execute_if.data.header.wid, to_fullPC(execute_if.data.header.PC), full_addr[i] - 8, execute_if.data.header.uuid))
    end
    `RUNTIME_ASSERT(~(execute_if.valid && execute_if.data.op_args.lsu.is_async && (execute_if.data.op_type == INST_LSU_SQ)),
        ("%t: unsplit 16-byte st.async reached the LSU: wid=%0d, PC=0x%0h (#%0d)",
            $time, execute_if.data.header.wid, to_fullPC(execute_if.data.header.PC), execute_if.data.header.uuid))

    // cp.async lanes, checked in this order; a lane reports its first failure only.
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cpa_check
        wire [MEM_ADDRW-1:0] src_blk = agu_addr[i][MEM_ASHIFT +: MEM_ADDRW];
        wire [MEM_ADDRW-1:0] dst_blk = cpa_dst[i][MEM_ASHIFT +: MEM_ADDRW];
        wire [MEM_ADDRW-1:0] arr_blk = full_addr[i][MEM_ASHIFT +: MEM_ADDRW];
        wire src_local = (src_blk[MEM_ADDRW-1:LMEM_BLK_LSB] == LMEM_BLK_BASE[MEM_ADDRW-1:LMEM_BLK_LSB]);
        wire dst_local = (dst_blk[MEM_ADDRW-1:LMEM_BLK_LSB] == LMEM_BLK_BASE[MEM_ADDRW-1:LMEM_BLK_LSB]);
        wire arr_local = (arr_blk[MEM_ADDRW-1:LMEM_BLK_LSB] == LMEM_BLK_BASE[MEM_ADDRW-1:LMEM_BLK_LSB]);
        `UNUSED_VAR (dst_blk)
        `UNUSED_VAR (arr_blk)
        wire src_io  = (src_blk >= MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_IO_BASE_ADDR) >> MEM_ASHIFT))
                    && (src_blk < MEM_ADDRW'(`VX_CFG_XLEN'(`VX_MEM_IO_END_ADDR) >> MEM_ASHIFT));
        wire src_win = (src_blk[MEM_ADDRW-1:DLMEM_WIN_BLK_LSB] == DLMEM_WIN_BLK_BASE[MEM_ADDRW-1:DLMEM_WIN_BLK_LSB]);
        wire src_global = ~src_local && ~src_win && ~src_io;
        // Alignment is to the instruction's cp-size: 16 for both halves of a 16-byte copy.
        wire [3:0] amask   = req_cpa4 ? 4'd3 : (req_cpa16 ? 4'd15 : 4'd7);
        wire [4:0] cp_size = req_cpa4 ? 5'd4 : (req_cpa16 ? 5'd16 : 5'd8);
        wire dst_mis = (| (execute_if.data.rs3_data[i][3:0] & amask));
        wire src_mis = (| (execute_if.data.rs1_data[i][3:0] & amask));
        wire ss_big  = (execute_if.data.rs2_data[i] > `VX_CFG_XLEN'(cp_size));
        wire act     = execute_fire && req_cpa_copy && execute_if.data.header.tmask[i];
        wire e_dst   = act && ~dst_local;
        wire e_dmis  = act && ~e_dst && dst_mis;
        wire e_ss    = act && ~e_dst && ~dst_mis && ss_big;
        wire src_on  = act && ~e_dst && ~dst_mis && ~ss_big && (cpa_ss[i] != 4'd0);
        wire e_src   = src_on && ~src_global;
        wire e_smis  = src_on && src_global && src_mis;
        wire arr_act = execute_fire && req_cpa_arr && execute_if.data.header.tmask[i];
        wire e_mbar  = arr_act && (~arr_local || (full_addr[i][2:0] != 3'd0));
        `RUNTIME_ASSERT(~e_dst,
            ("%t: cp.async destination must be the block's own shared memory: wid=%0d, PC=0x%0h, lane=%0d, dst=0x%0h (#%0d)",
                $time, execute_if.data.header.wid, to_fullPC(execute_if.data.header.PC), i, cpa_dst[i], execute_if.data.header.uuid))
        `RUNTIME_ASSERT(~(e_dmis || e_smis),
            ("%t: cp.async address must be aligned to its size: wid=%0d, PC=0x%0h, lane=%0d, src=0x%0h, dst=0x%0h, size=%0d (#%0d)",
                $time, execute_if.data.header.wid, to_fullPC(execute_if.data.header.PC), i, execute_if.data.rs1_data[i],
                execute_if.data.rs3_data[i], cp_size, execute_if.data.header.uuid))
        `RUNTIME_ASSERT(~e_ss,
            ("%t: cp.async src-size exceeds cp-size: wid=%0d, PC=0x%0h, lane=%0d, src-size=%0d, size=%0d (#%0d)",
                $time, execute_if.data.header.wid, to_fullPC(execute_if.data.header.PC), i, execute_if.data.rs2_data[i],
                cp_size, execute_if.data.header.uuid))
        `RUNTIME_ASSERT(~e_src,
            ("%t: cp.async source must be global memory: wid=%0d, PC=0x%0h, lane=%0d, src=0x%0h (#%0d)",
                $time, execute_if.data.header.wid, to_fullPC(execute_if.data.header.PC), i, agu_addr[i], execute_if.data.header.uuid))
        `RUNTIME_ASSERT(~e_mbar,
            ("%t: cp.async.mbarrier.arrive needs the block's own mbarrier: wid=%0d, PC=0x%0h, lane=%0d, addr=0x%0h (#%0d)",
                $time, execute_if.data.header.wid, to_fullPC(execute_if.data.header.PC), i, full_addr[i], execute_if.data.header.uuid))
    end
`endif

    // store data formatting
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_data
        always @(*) begin
            mem_req_data[i] = execute_if.data.rs2_data[i];
            case (req_align[i])
                1: mem_req_data[i][`VX_CFG_XLEN-1:8]  = execute_if.data.rs2_data[i][`VX_CFG_XLEN-9:0];
                2: mem_req_data[i][`VX_CFG_XLEN-1:16] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-17:0];
                3: mem_req_data[i][`VX_CFG_XLEN-1:24] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-25:0];
            `ifdef VX_CFG_XLEN_64
                4: mem_req_data[i][`VX_CFG_XLEN-1:32] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-33:0];
                5: mem_req_data[i][`VX_CFG_XLEN-1:40] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-41:0];
                6: mem_req_data[i][`VX_CFG_XLEN-1:48] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-49:0];
                7: mem_req_data[i][`VX_CFG_XLEN-1:56] = execute_if.data.rs2_data[i][`VX_CFG_XLEN-57:0];
            `endif
                default:;
            endcase
        end
    end

    // multi-packet load responses could return out-of-order.
    // we should track and flag SOP and EOP responses.

    if (PID_BITS != 0) begin : g_pid
        reg [`VX_CFG_LSU_PENDING_SIZE-1:0][PID_BITS:0] pkt_ctr;
        reg [`VX_CFG_LSU_PENDING_SIZE-1:0] pkt_sop, pkt_eop;

    `ifdef VX_CFG_EXT_MBAR_ENABLE
        // A cp.async read holds a cp.async slot, not a packet slot.
        wire mem_req_rd_fire     = mem_req_fire && ~mem_req_rw && ~req_cpa_copy;
        wire pkt_rsp_fire        = mem_rsp_fire && ~rsp_cpa;
        wire mem_req_rd_sop_fire = mem_req_rd_fire && execute_if.data.header.sop;
        wire mem_req_rd_eop_fire = mem_req_rd_fire && execute_if.data.header.eop;
        wire mem_rsp_eop_fire    = pkt_rsp_fire && mem_rsp_eop;
        wire mem_rsp_eop_pkt_fire= pkt_rsp_fire && mem_rsp_eop_pkt;
    `else
        wire mem_req_rd_fire     = mem_req_fire && ~mem_req_rw;
        wire mem_req_rd_sop_fire = mem_req_rd_fire && execute_if.data.header.sop;
        wire mem_req_rd_eop_fire = mem_req_rd_fire && execute_if.data.header.eop;
        wire mem_rsp_eop_fire    = mem_rsp_fire && mem_rsp_eop;
        wire mem_rsp_eop_pkt_fire= mem_rsp_fire && mem_rsp_eop_pkt;
    `endif
        wire full;

        VX_allocator #(
            .SIZE (`VX_CFG_LSU_PENDING_SIZE)
        ) pkt_allocator (
            .clk        (clk),
            .reset      (reset),
            .acquire_en (mem_req_rd_eop_fire),
            .acquire_addr(pkt_waddr),
            .release_en (mem_rsp_eop_pkt_fire),
            .release_addr(pkt_raddr),
            `UNUSED_PIN (empty),
            .full       (full)
        );

        wire rw_collision = mem_req_rd_fire && mem_rsp_eop_fire && (pkt_raddr == pkt_waddr);

        always @(posedge clk) begin
            if (reset) begin
                pkt_ctr <= '0;
                pkt_sop <= '0;
                pkt_eop <= '0;
            end else begin
                if (mem_req_rd_sop_fire) begin
                    pkt_sop[pkt_waddr] <= 1;
                end
                if (mem_req_rd_eop_fire) begin
                    pkt_eop[pkt_waddr] <= 1;
                end
            `ifdef VX_CFG_EXT_MBAR_ENABLE
                if (pkt_rsp_fire) begin
                    pkt_sop[pkt_raddr] <= 0;
                end
            `else
                if (mem_rsp_fire) begin
                    pkt_sop[pkt_raddr] <= 0;
                end
            `endif
                if (mem_rsp_eop_pkt_fire) begin
                    pkt_eop[pkt_raddr] <= 0;
                end
                if (~rw_collision) begin
                    if (mem_req_rd_fire) begin
                        pkt_ctr[pkt_waddr] <= pkt_ctr[pkt_waddr] + PID_BITS'(1);
                    end
                    if (mem_rsp_eop_fire) begin
                        pkt_ctr[pkt_raddr] <= pkt_ctr[pkt_raddr] - PID_BITS'(1);
                    end
                end
            end
        end

        assign mem_rsp_sop_pkt = pkt_sop[pkt_raddr];
        assign mem_rsp_eop_pkt = mem_rsp_eop && pkt_eop[pkt_raddr] && (pkt_ctr[pkt_raddr] == 1);
        `RUNTIME_ASSERT(~(mem_req_rd_fire && full), ("allocator full!"))
        `RUNTIME_ASSERT(~(mem_req_rd_sop_fire && pkt_ctr[pkt_waddr] != 0), ("oops! broken sop request!"))
        // pkt_ctr is guarded against a same-cycle same-slot request/response by
        // rw_collision; pkt_eop and pkt_sop are not, so the release wins and the
        // incoming packet loses the flag that would ever complete it.
        `RUNTIME_ASSERT(~(mem_req_rd_eop_fire && mem_rsp_eop_pkt_fire && (pkt_waddr == pkt_raddr)),
            ("%t: *** pkt_eop collision on slot %0d: incoming packet loses its eop flag", $time, pkt_waddr))
        `UNUSED_VAR (mem_rsp_sop)
    end else begin : g_no_pid
        assign pkt_waddr = 0;
        assign mem_rsp_sop_pkt = mem_rsp_sop;
        assign mem_rsp_eop_pkt = mem_rsp_eop;
        `UNUSED_VAR (pkt_raddr)
    end

    // pack memory request tag
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    wire req_aq, req_rl;
`ifdef DXA_MBAR_ENABLE
    // A proxy fence unlocks its warp as an acquire does, when it retires.
    assign req_aq = (execute_if.data.op_args.lsu.amo_valid && execute_if.data.op_args.lsu.amo_aq)
                 || (execute_if.data.op_type == INST_LSU_PFENCE);
    assign req_rl = execute_if.data.op_args.lsu.amo_valid && execute_if.data.op_args.lsu.amo_rl;
`elsif VX_CFG_EXT_A_ENABLE
    assign req_aq = execute_if.data.op_args.lsu.amo_valid && execute_if.data.op_args.lsu.amo_aq;
    assign req_rl = execute_if.data.op_args.lsu.amo_valid && execute_if.data.op_args.lsu.amo_rl;
`else
    assign req_aq = 1'b0;
    assign req_rl = 1'b0;
`endif
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // Marks a try_wait, whose retire the waiter table needs; rl and fence keep
    // the tag bits the DLMEM translation reads.
    wire req_tw = execute_if.data.op_args.lsu.amo_valid
               && (amo_op_e'(execute_if.data.op_args.lsu.amo_op) == AMO_OP_MBAR_TRY_WAIT);
    // A copy's packet field names its cp.async slot.
    wire [LSUQ_SIZEW-1:0] cpa_slot;
    assign mem_req_tag = {
        execute_if.data.header,
        execute_if.data.op_type,
        req_align,
        req_cpa_copy ? cpa_slot : pkt_waddr,
        req_gload,
        req_tw,
        req_cpa_copy,
        req_aq,
        req_rl,
        req_is_fence
    };
`else
    assign mem_req_tag = {
        execute_if.data.header,
        execute_if.data.op_type,
        req_align,
        pkt_waddr,
        req_gload,
        req_aq,
        req_rl,
        req_is_fence
    };
`endif
`else
    assign mem_req_tag = {
        execute_if.data.header,
        execute_if.data.op_type,
        req_align,
        pkt_waddr,
        req_is_fence
    };
`endif

    // Pack per-lane mem_req signals into the client_if request payload;
    // unpack the response payload into per-lane mem_rsp signals.
    assign client_if.req_valid       = mem_req_valid;
    assign client_if.req_data.rw     = mem_req_rw;
    assign client_if.req_data.mask   = mem_req_mask;
    assign client_if.req_data.byteen = mem_req_byteen;
    assign client_if.req_data.addr   = mem_req_addr;
    assign client_if.req_data.attr   = mem_req_attr;
    assign client_if.req_data.data   = mem_req_data;
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // A cp.async read is a plain global read past the slice.
    assign client_if.req_data.is_async = execute_if.data.op_args.lsu.is_async && execute_if.data.op_args.lsu.is_store;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mbar_addr
        assign client_if.req_data.mbar_addr[i] = execute_if.data.rs3_data[i][`VX_CFG_MEM_ADDR_WIDTH-1:REQ_ASHIFT];
    end
`endif
    assign client_if.req_data.tag    = mem_req_tag;
    assign mem_req_ready             = client_if.req_ready;

    assign mem_rsp_valid             = client_if.rsp_valid;
    assign mem_rsp_mask              = client_if.rsp_data.mask;
    assign mem_rsp_data              = client_if.rsp_data.data;
    assign mem_rsp_tag               = client_if.rsp_data.tag;
    assign mem_rsp_sop               = client_if.rsp_data.sop;
    assign mem_rsp_eop               = client_if.rsp_data.eop;
    assign client_if.rsp_ready       = mem_rsp_ready;

    lsu_header_t rsp_hdr;
    wire [INST_LSU_BITS-1:0] rsp_op_type;
    wire [NUM_LANES-1:0][REQ_ASHIFT-1:0] rsp_align;
    `UNUSED_VAR (rsp_op_type)

    // unpack memory response tag
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    wire rsp_aq, rsp_rl;
    `UNUSED_VAR (rsp_rl)
`ifdef VX_CFG_EXT_MBAR_ENABLE
    wire rsp_tw;
    assign {
        rsp_hdr,
        rsp_op_type,
        rsp_align,
        pkt_raddr,
        rsp_gload,
        rsp_tw,
        rsp_cpa,
        rsp_aq,
        rsp_rl,
        rsp_is_fence
    } = mem_rsp_tag;
`else
    assign {
        rsp_hdr,
        rsp_op_type,
        rsp_align,
        pkt_raddr,
        rsp_gload,
        rsp_aq,
        rsp_rl,
        rsp_is_fence
    } = mem_rsp_tag;
`endif

    always @(posedge clk) begin
        if (reset) begin
            gdirty <= '0;
            gload  <= '0;
        end else begin
            // A new warp has no earlier write for its first release to order.
            gdirty <= gdirty & ~warp_spawn;
        `ifdef DXA_MBAR_ENABLE
            // A proxy fence drains no cache, so the warp's global writes stay unfenced.
            if (mem_rsp_fire && rsp_is_fence && mem_rsp_eop_pkt && (rsp_op_type != INST_LSU_PFENCE)) begin
                gdirty[rsp_hdr.wid] <= 1'b0;
            end
        `else
            if (mem_rsp_fire && rsp_is_fence && mem_rsp_eop_pkt) begin
                gdirty[rsp_hdr.wid] <= 1'b0;
            end
        `endif
            if (mem_req_fire && req_has_global && req_writes) begin
                gdirty[execute_if.data.header.wid] <= 1'b1;
            end
            if (mem_req_fire && req_gload && mem_rsp_fire && mem_rsp_eop && rsp_gload
             && (execute_if.data.header.wid == rsp_hdr.wid)) begin
                gload[rsp_hdr.wid] <= gload[rsp_hdr.wid];
            end else begin
                if (mem_req_fire && req_gload) begin
                    gload[execute_if.data.header.wid] <= gload[execute_if.data.header.wid] + GLOAD_W'(1);
                end
                if (mem_rsp_fire && mem_rsp_eop && rsp_gload) begin
                    gload[rsp_hdr.wid] <= gload[rsp_hdr.wid] - GLOAD_W'(1);
                end
            end
        end
    end

    for (genvar w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin : g_spawn_gload
        `RUNTIME_ASSERT (~(warp_spawn[w] && (gload[w] != '0)),
            ("%t: %s global loads in flight at warp spawn, wid=%0d", $time, INSTANCE_ID, w))
    end
    `RUNTIME_ASSERT (~(mem_req_fire && req_gload && (gload[execute_if.data.header.wid] == '1)),
        ("%t: %s global load counter overflow, wid=%0d", $time, INSTANCE_ID, execute_if.data.header.wid))
    `RUNTIME_ASSERT (~(mem_rsp_fire && mem_rsp_eop && rsp_gload && (gload[rsp_hdr.wid] == '0)),
        ("%t: %s global load counter underflow, wid=%0d", $time, INSTANCE_ID, rsp_hdr.wid))
`else
    assign {
        rsp_hdr,
        rsp_op_type,
        rsp_align,
        pkt_raddr,
        rsp_is_fence
    } = mem_rsp_tag;
`endif

    // load response formatting

    reg [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] rsp_data;

`ifdef VX_CFG_XLEN_64
`ifdef VX_CFG_EXT_F_ENABLE
    // apply nan-boxing to flw outputs
    wire rsp_is_float = (get_reg_type(rsp_hdr.rd) == REG_TYPE_F);
`else
    wire rsp_is_float = 0;
`endif
`endif

    for (genvar i = 0; i < NUM_LANES; i++) begin : g_rsp_data
    `ifdef VX_CFG_XLEN_64
        wire [63:0] rsp_data64 = mem_rsp_data[i];
        wire [31:0] rsp_data32 = (rsp_align[i][2] ? mem_rsp_data[i][63:32] : mem_rsp_data[i][31:0]);
    `else
        wire [31:0] rsp_data32 = mem_rsp_data[i];
    `endif
        wire [15:0] rsp_data16 = rsp_align[i][1] ? rsp_data32[31:16] : rsp_data32[15:0];
        wire [7:0]  rsp_data8  = rsp_align[i][0] ? rsp_data16[15:8] : rsp_data16[7:0];

        always @(*) begin
            case (inst_lsu_fmt(rsp_op_type))
            LSU_FMT_B:  rsp_data[i] = `VX_CFG_XLEN'(signed'(rsp_data8));
            LSU_FMT_H:  rsp_data[i] = `VX_CFG_XLEN'(signed'(rsp_data16));
            LSU_FMT_BU: rsp_data[i] = `VX_CFG_XLEN'(unsigned'(rsp_data8));
            LSU_FMT_HU: rsp_data[i] = `VX_CFG_XLEN'(unsigned'(rsp_data16));
        `ifdef VX_CFG_XLEN_64
            LSU_FMT_W:  rsp_data[i] = rsp_is_float ? (`VX_CFG_XLEN'(rsp_data32) | 64'hffffffff00000000) : `VX_CFG_XLEN'(signed'(rsp_data32));
            LSU_FMT_WU: rsp_data[i] = `VX_CFG_XLEN'(unsigned'(rsp_data32));
            LSU_FMT_D:  rsp_data[i] = `VX_CFG_XLEN'(signed'(rsp_data64));
        `else
            LSU_FMT_W:  rsp_data[i] = `VX_CFG_XLEN'(signed'(rsp_data32));
        `endif
            default: rsp_data[i] = 'x;
            endcase
        end
    end

    // result

    lsu_header_t rsp_hdr2;
    always @(*) begin
        rsp_hdr2 = rsp_hdr;
        rsp_hdr2.tmask = mem_rsp_mask;
        rsp_hdr2.sop = mem_rsp_sop_pkt;
        rsp_hdr2.eop = mem_rsp_eop_pkt;
    end

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // An acquire AMO's aq bit travels beside its result to the unlock below.
    wire result_rsp_aq, result_aq;
`endif
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // So does a try_wait's mark, to its retire.
    wire result_rsp_tw, result_tw;
    // A copy's response never waits: the cp.async unit lands it.
    wire rsp_buf_ready;
    assign mem_rsp_ready = rsp_cpa || rsp_buf_ready;
`endif

    VX_elastic_buffer #(
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        .DATAW (2 + $bits(lsu_result_t)),
    `elsif VX_CFG_EXT_DLMEM_ENABLE
        .DATAW (1 + $bits(lsu_result_t)),
    `else
        .DATAW ($bits(lsu_result_t)),
    `endif
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        .valid_in  (mem_rsp_valid && ~rsp_cpa),
        .ready_in  (rsp_buf_ready),
    `else
        .valid_in  (mem_rsp_valid),
        .ready_in  (mem_rsp_ready),
    `endif
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        .data_in   ({rsp_tw, rsp_aq, rsp_hdr2, rsp_data}),
        .data_out  ({result_rsp_tw, result_rsp_aq, result_rsp_if.data}),
    `elsif VX_CFG_EXT_DLMEM_ENABLE
        .data_in   ({rsp_aq, rsp_hdr2, rsp_data}),
        .data_out  ({result_rsp_aq, result_rsp_if.data}),
    `else
        .data_in   ({rsp_hdr2, rsp_data}),
        .data_out  (result_rsp_if.data),
    `endif
        .valid_out (result_rsp_if.valid),
        .ready_out (result_rsp_if.ready)
    );

    VX_elastic_buffer #(
        .DATAW ($bits(lsu_header_t)),
        .SIZE  (2)
    ) no_rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (no_rsp_buf_valid),
        .ready_in  (no_rsp_buf_ready),
        .data_in   (execute_if.data.header),
        .data_out  (result_no_rsp_if.data.header),
        .valid_out (result_no_rsp_if.valid),
        .ready_out (result_no_rsp_if.ready)
    );
    assign result_no_rsp_if.data.data = result_rsp_if.data.data; // arbiter MUX optimization

    VX_stream_arb #(
        .NUM_INPUTS (2),
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        .DATAW      (2 + $bits(lsu_result_t)),
    `elsif VX_CFG_EXT_DLMEM_ENABLE
        .DATAW      (1 + $bits(lsu_result_t)),
    `else
        .DATAW      ($bits(lsu_result_t)),
    `endif
        .ARBITER    ("P"), // prioritize result_rsp_if
        .OUT_BUF    (3)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({result_no_rsp_if.valid, result_rsp_if.valid}),
        .ready_in  ({result_no_rsp_if.ready, result_rsp_if.ready}),
    `ifdef VX_CFG_EXT_MBAR_ENABLE
        .data_in   ({2'b00, result_no_rsp_if.data, result_rsp_tw, result_rsp_aq, result_rsp_if.data}),
        .data_out  ({result_tw, result_aq, result_if.data}),
    `elsif VX_CFG_EXT_DLMEM_ENABLE
        .data_in   ({1'b0, result_no_rsp_if.data, result_rsp_aq, result_rsp_if.data}),
        .data_out  ({result_aq, result_if.data}),
    `else
        .data_in   ({result_no_rsp_if.data, result_rsp_if.data}),
        .data_out  (result_if.data),
    `endif
        .valid_out (result_if.valid),
        .ready_out (result_if.ready),
        `UNUSED_PIN (sel_out)
    );

`ifdef VX_CFG_EXT_DLMEM_ENABLE
    // AMOs always take the response path, and its eop is the instruction's last packet.
    assign sched_unlock_if.valid = result_if.valid && result_if.ready && result_if.data.header.eop && result_aq;
    assign sched_unlock_if.wid   = result_if.data.header.wid;
`endif
`ifdef VX_CFG_EXT_MBAR_ENABLE
    // A try_wait never takes the unlock above: the waiter table releases its warp.
    assign mbar_retire_valid = result_if.valid && result_if.ready && result_if.data.header.eop && result_tw;
    assign mbar_retire_wid   = result_if.data.header.wid;
`endif

`ifdef VX_CFG_EXT_MBAR_ENABLE
    localparam LMEM_LOG = `VX_CFG_LMEM_LOG_SIZE;
    wire [NUM_LANES-1:0][LMEM_LOG-1:0] cpa_iss_addr;
    wire [NUM_LANES-1:0][63:0]         cpa_rsp_data;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_cpa_iss
        assign cpa_iss_addr[i] = req_cpa_copy ? cpa_dst[i][LMEM_LOG-1:0] : full_addr[i][LMEM_LOG-1:0];
        assign cpa_rsp_data[i] = 64'(mem_rsp_data[i]);
    end

    VX_cpa_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-cpa", INSTANCE_ID)))
    ) cpa_unit (
        .clk       (clk),
        .reset     (reset),
        .iss_valid (execute_if.valid && req_cpa && execute_ready_base),
        .iss_op    (cpa_op),
        .iss_noinc (req_op == INST_LSU_CPA_NOINC),
        .iss_wid   (execute_if.data.header.wid),
        .iss_pid   (`UP(PID_BITS)'(execute_if.data.header.pid)),
        .iss_sop   (execute_if.data.header.sop),
        .iss_eop   (execute_if.data.header.eop),
        .iss_tmask (execute_if.data.header.tmask),
        .iss_size8 (~req_cpa4),
        .iss_n     (execute_if.data.op_args.lsu.offset[4:0]),
        .iss_addr  (cpa_iss_addr),
        .iss_ss    (cpa_ss),
        .iss_ready (cpa_iss_ready),
        .iss_slot  (cpa_slot),
        .rsp_valid (mem_rsp_valid && rsp_cpa),
        .rsp_slot  (pkt_raddr),
        .rsp_pid   (`UP(PID_BITS)'(rsp_hdr.pid)),
        .rsp_mask  (mem_rsp_mask),
        .rsp_data  (cpa_rsp_data),
        .rsp_align (rsp_align),
        .lmem_if   (cpa_lmem_if),
        .unlock_if (cpa_unlock_if),
        .idle      (cpa_idle)
    );
`endif

`ifdef DBG_TRACE_MEM
    always @(posedge clk) begin
        if (execute_if.valid && fence_lock) begin
            `TRACE(2, ("%t: *** %s fence wait\n", $time, INSTANCE_ID))
        end
        if (mem_req_fire) begin
            if (mem_req_rw) begin
                `TRACE(2, ("%t: %s Wr Req: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b, addr=", $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.cta_id, to_fullPC(execute_if.data.header.PC), mem_req_mask))
                `TRACE_ARRAY1D(2, "0x%h", full_addr, NUM_LANES)
                `TRACE(2, (", attr="))
                `TRACE_ARRAY1D(2, "%b", mem_req_attr, NUM_LANES)
                `TRACE(2, (", byteen=0x%0h, data=", mem_req_byteen))
                `TRACE_ARRAY1D(2, "0x%0h", mem_req_data, NUM_LANES)
                `TRACE(2, (", sop=%b, pid=%0d, eop=%b, tag=0x%0h (#%0d)\n", execute_if.data.header.pid, execute_if.data.header.sop, execute_if.data.header.eop, mem_req_tag, execute_if.data.header.uuid))
            end else begin
                `TRACE(2, ("%t: %s Rd Req: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b, addr=", $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.cta_id, to_fullPC(execute_if.data.header.PC), mem_req_mask))
                `TRACE_ARRAY1D(2, "0x%h", full_addr, NUM_LANES)
                `TRACE(2, (", attr="))
                `TRACE_ARRAY1D(2, "%b", mem_req_attr, NUM_LANES)
                `TRACE(2, (", byteen=0x%0h, rd=%0d, pid=%0d, sop=%b, eop=%b, tag=0x%0h (#%0d)\n", mem_req_byteen, execute_if.data.header.rd, execute_if.data.header.pid, execute_if.data.header.sop, execute_if.data.header.eop, mem_req_tag, execute_if.data.header.uuid))
            end
        end
        if (mem_rsp_fire) begin
            `TRACE(2, ("%t: %s Rsp: wid=%0d, cta_id=%0d, PC=0x%0h, tmask=%b, rd=%0d, pid=%0d, sop=%b, eop=%b, data=",
                $time, INSTANCE_ID, rsp_hdr.wid, rsp_hdr.cta_id, to_fullPC(rsp_hdr.PC), mem_rsp_mask, rsp_hdr.rd, rsp_hdr.pid, mem_rsp_sop_pkt, mem_rsp_eop_pkt))
            `TRACE_ARRAY1D(2, "0x%0h", mem_rsp_data, NUM_LANES)
            `TRACE(2, (", tag=0x%0h (#%0d)\n", mem_rsp_tag, rsp_hdr.uuid))
        end
    end
`ifdef VX_CFG_EXT_DLMEM_ENABLE
    reg cfence_wait_r;
    always @(posedge clk) begin
        cfence_wait_r <= ~reset && execute_if.valid && cfence_hold;
        if (~reset && execute_if.valid && cfence_hold && ~cfence_wait_r) begin
            `TRACE(2, ("%t: %s cfence-wait: wid=%0d, gload=%0d (#%0d)\n", $time, INSTANCE_ID,
                execute_if.data.header.wid, gload[execute_if.data.header.wid], execute_if.data.header.uuid))
        end
        if (execute_if.valid && execute_if.ready && req_is_cfence && execute_if.data.header.eop) begin
            if (cfence_skip) begin
                `TRACE(2, ("%t: %s cfence-skip: wid=%0d (#%0d)\n", $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.uuid))
            end else begin
                `TRACE(2, ("%t: %s cfence-token: wid=%0d (#%0d)\n", $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.uuid))
            end
        end
    end
`endif
`endif


`ifdef SCOPE
`ifdef DBG_SCOPE_LSU
    `SCOPE_IO_SWITCH (1);
    wire reset_negedge;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP_EX (0, 3, 4, 2, (
            1 + NUM_LANES * (`VX_CFG_XLEN + LSU_WORD_SIZE + LSU_WORD_SIZE * 8) + UUID_WIDTH + NUM_LANES * LSU_WORD_SIZE * 8 + UUID_WIDTH
        ), {
            mem_req_valid,
            mem_req_ready,
            mem_rsp_valid,
            mem_rsp_ready
        }, {
            mem_req_fire,
            mem_rsp_fire
        }, {
            mem_req_rw,
            full_addr,
            mem_req_byteen,
            mem_req_data,
            execute_if.data.header.uuid,
            rsp_data,
            rsp_hdr.uuid
        },
        reset_negedge, 1'b0,	4096
    );
`else
    `SCOPE_IO_UNUSED(0)
`endif
`endif

`ifdef CHIPSCOPE
`ifdef DBG_SCOPE_LSU
    ila_lsu ila_lsu_inst (
        .clk    (clk),
        .probe0 ({execute_if.valid, execute_if.data, execute_if.ready}),
        .probe1 ({client_if.req_valid, client_if.req_data, client_if.req_ready}),
        .probe2 ({client_if.rsp_valid, client_if.rsp_data, client_if.rsp_ready})
    );
`endif
`endif

endmodule
