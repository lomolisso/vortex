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

module VX_dxa_core import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_REQS = 1,
    parameter GMEM_OUT_PORTS = 1
) (
    input wire clk,
    input wire reset,
`ifdef PERF_ENABLE
    output dxa_perf_t dxa_perf,
`endif
    VX_dcr_bus_if.slave dcr_bus_if,
    VX_dxa_req_bus_if.slave req_bus_if[NUM_REQS],
    VX_mem_bus_if.master gmem_bus_if[GMEM_OUT_PORTS],
    VX_mem_bus_if.master smem_bus_if[1],
`ifdef DXA_MBAR_ENABLE
    // Rows of copies into another core's block, as DLMEM request flits.
    VX_dlmem_bus_if.master row_out_if,
    // Each requesting core's slot stride, a launch constant.
    input wire [`VX_CFG_SOCKET_SIZE-1:0][`VX_CFG_LMEM_LOG_SIZE:0] slot_stride,
    // Bulk-group completion events, per requesting core, never refused.
    output wire [`VX_CFG_SOCKET_SIZE-1:0] bg_compl_valid,
    output dxa_bg_compl_t [`VX_CFG_SOCKET_SIZE-1:0] bg_compl,
`endif
    output wire busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
`ifdef DXA_MBAR_ENABLE
    // A transfer's core id indexes slot_stride by its socket-local bits.
    `STATIC_ASSERT (NUM_REQS == `VX_CFG_SOCKET_SIZE, ("invalid parameter"))
`endif

    // ================================================================
    // Descriptor table (single read port, driven by dispatch queue)
    // ================================================================
`ifdef DXA_MBAR_ENABLE
    localparam TM_STRIDE = `VX_DCR_DXA_DESC_STRIDE;
    wire [DXA_DESC_ADDR_W-1:0] desc_read_addr;
    dxa_desc_t desc_read_data;
    wire [31:0] desc_w22, desc_w23;
    wire                        tm_fill_valid, tm_fill_ready;
    wire [DXA_TMAP_IDX_W-1:0]   tm_fill_entry;
    wire [TM_STRIDE-1:0]        tm_fill_wren;
    wire [TM_STRIDE*32-1:0]     tm_fill_wdata;

    VX_dxa_desc_table desc_table (
        .clk        (clk),
        .reset      (reset),
        .dcr_bus_if (dcr_bus_if),
        .read_addr  (desc_read_addr),
        .fill_valid (tm_fill_valid),
        .fill_entry (tm_fill_entry),
        .fill_wren  (tm_fill_wren),
        .fill_wdata (tm_fill_wdata),
        .fill_ready (tm_fill_ready),
        .read_w22   (desc_w22),
        .read_w23   (desc_w23),
        .read_desc  (desc_read_data)
    );
`else
    wire [DXA_DESC_SLOT_W-1:0] desc_read_addr;
    dxa_desc_t desc_read_data;

    VX_dxa_desc_table desc_table (
        .clk        (clk),
        .reset      (reset),
        .dcr_bus_if (dcr_bus_if),
        .read_addr  (desc_read_addr),
        .read_desc  (desc_read_data)
    );
`endif

    // ================================================================
    // Request path: sockets → arb(N:1) → queue → dispatch(1:N) → workers
    // ================================================================
    localparam REQ_DATAW = $bits(dxa_req_data_t);

    VX_dxa_req_bus_if arb_out_bus_if[1]();

    VX_dxa_req_arb #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (1),
        // Register the arbitration output whenever it actually arbitrates
        // (>1 input). 1:1 stays a passthrough (no area/latency cost).
        .OUT_BUF     ((NUM_REQS > 1) ? 3 : 0)
    ) req_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (req_bus_if),
        .bus_out_if (arb_out_bus_if)
    );

    VX_dxa_req_bus_if queue_out_bus_if[1]();

    VX_elastic_buffer #(
        .DATAW  (REQ_DATAW),
        .SIZE   (`VX_CFG_DXA_QUEUE_SIZE),
        // LUTRAM=0: the wide-shallow queue is BRAM-eligible, moving it off the
        // LUT pool onto an (abundant) BRAM tile.
        .LUTRAM (0)
    ) req_queue (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (arb_out_bus_if[0].req_valid),
        .ready_in  (arb_out_bus_if[0].req_ready),
        .data_in   (arb_out_bus_if[0].req_data),
        .valid_out (queue_out_bus_if[0].req_valid),
        .ready_out (queue_out_bus_if[0].req_ready),
        .data_out  (queue_out_bus_if[0].req_data)
    );

`ifdef DXA_MBAR_ENABLE
    // The queue head reads its descriptor from a DCR slot or a cached tensor
    // map. A slot request and a map hit read the RAM in the IDLE cycle and are
    // presented the next (PRESENT). A map miss (MAP) reads the map's lines on
    // the GMEM port, writes each line's words into its victim entry as it
    // returns, then looks the head up again as a hit. A tensormap prefetch pops
    // on a hit, an invalidate in its IDLE cycle; neither reaches dispatch.
    localparam FETCH_IDLE = 2'd0, FETCH_PRESENT = 2'd1, FETCH_MAP = 2'd2;
    reg [1:0] fetch_state_r;

    VX_dxa_worker_req_if dispatch_in_if[1]();

    dxa_tm_req_t q_tm;
    assign q_tm = queue_out_bus_if[0].req_data.tm;
    wire q_valid = queue_out_bus_if[0].req_valid;
    wire q_map   = q_tm.en;
    wire q_pf    = (q_tm.op == DXA_TM_PREFETCH);
    wire q_inv   = (q_tm.op == DXA_TM_INVAL);

    reg [DXA_TMAP_ENTRIES-1:0]                      tm_valid_r;
    reg [DXA_TMAP_ENTRIES-1:0][DXA_TMAP_TAG_W-1:0]  tm_tag_r;
    reg [DXA_TMAP_IDX_W-1:0] tm_rr_r, tm_vict_r, tm_cur_r;

    wire [DXA_TMAP_ENTRIES-1:0] tm_match;
    for (genvar e = 0; e < DXA_TMAP_ENTRIES; ++e) begin : g_tm_match
        assign tm_match[e] = tm_valid_r[e] && (tm_tag_r[e] == q_tm.addr);
    end
    wire                      tm_hit = (| tm_match);
    wire [DXA_TMAP_IDX_W-1:0] tm_hit_idx, tm_free_idx;
    wire                      tm_has_free;

    VX_priority_encoder #(
        .N (DXA_TMAP_ENTRIES)
    ) tm_hit_enc (
        .data_in    (tm_match),
        .index_out  (tm_hit_idx),
        `UNUSED_PIN (onehot_out),
        `UNUSED_PIN (valid_out)
    );

    // The victim: the lowest invalid entry, else the round-robin pointer.
    VX_priority_encoder #(
        .N (DXA_TMAP_ENTRIES)
    ) tm_free_enc (
        .data_in    (~tm_valid_r),
        .index_out  (tm_free_idx),
        `UNUSED_PIN (onehot_out),
        .valid_out  (tm_has_free)
    );
    wire [DXA_TMAP_IDX_W-1:0] tm_victim = tm_has_free ? tm_free_idx : tm_rr_r;

    // Every launch writes the KMU state: maps are reread per launch and the
    // victim order restarts.
    wire tm_launch = dcr_bus_if.req_valid && dcr_bus_if.req_data.rw
                  && (dcr_bus_if.req_data.addr >= VX_DCR_ADDR_WIDTH'(`VX_DCR_KMU_STATE_BEGIN))
                  && (dcr_bus_if.req_data.addr < VX_DCR_ADDR_WIDTH'(`VX_DCR_KMU_STATE_END));
    // DXA work can outlive its kernel (its busy is not in the socket's), so a
    // fetch may straddle the next launch. A launch from the miss cycle through
    // the last response kills the fetch: the entry stays invalid, the head
    // misses again in the next IDLE cycle, and a tensormap prefetch pops there.
    reg tm_kill_r;

    wire idle_q     = (fetch_state_r == FETCH_IDLE) && q_valid;
    wire tm_inv_pop = idle_q && q_map && q_inv;
    wire tm_pf_pop  = idle_q && q_map && q_pf && (tm_hit || tm_kill_r);
    wire tm_miss    = idle_q && q_map && ~q_inv && ~tm_hit && ~(q_pf && tm_kill_r);
    wire tm_present = idle_q && (~q_map || (~q_inv && ~q_pf && tm_hit));

    // A map that fails its decode checks is held one PRESENT cycle, so its
    // assert fires before any check downstream sees the request.
    wire tm_hold;
    wire head_go = (fetch_state_r == FETCH_PRESENT) && ~tm_hold && dispatch_in_if[0].ready;

    assign queue_out_bus_if[0].req_ready = head_go || tm_inv_pop || tm_pf_pop;

    wire [DXA_TMAP_IDX_W-1:0] tm_rd_idx = (fetch_state_r == FETCH_IDLE) ? tm_hit_idx : tm_cur_r;
    assign desc_read_addr = q_map ? DXA_DESC_ADDR_W'(`VX_DCR_DXA_DESC_COUNT + 32'(tm_rd_idx))
                                  : DXA_DESC_ADDR_W'(queue_out_bus_if[0].req_data.meta[DXA_DESC_SLOT_W-1:0]);

    // ---- Map line reads: the map's 128 bytes from its first line on ----
    localparam TM_CL       = `VX_CFG_L1_LINE_SIZE;
    localparam TM_LG_CL    = `CLOG2(TM_CL);
    localparam TM_LINES    = (TM_CL > 64) ? 2 : (DXA_TMAP_BYTES / TM_CL);
    localparam TM_LINE_W   = `UP(`CLOG2(TM_LINES));
    localparam TM_CNT_W    = `CLOG2(TM_LINES + 1);
    localparam TM_LADDR_W  = `VX_CFG_MEM_ADDR_WIDTH - TM_LG_CL;
    localparam TM_PORT_TAG_W = SOCKET_MEM_TAG_WIDTH;
    localparam TM_VAL_W    = TM_PORT_TAG_W - UUID_WIDTH;
    localparam TM_ARB_SEL_BITS = `ARB_SEL_BITS(`VX_CFG_NUM_DXA_CORES, GMEM_OUT_PORTS);
    // A fetch tag sets the top value bit, which no worker tag reaches (a
    // worker's value is a line slot below DXA_MAX_INFLIGHT, or 0).
    `STATIC_ASSERT ((SOCKET_MEM_TAG_WIDTH - TM_ARB_SEL_BITS - UUID_WIDTH) > `CLOG2(`VX_CFG_DXA_MAX_INFLIGHT),
        ("DXA GMEM tag has no room for the tensor-map fetch"))
    `STATIC_ASSERT (TM_VAL_W > TM_LINE_W, ("DXA GMEM tag has no room for the tensor-map fetch"))

    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] tm_byte = {q_tm.addr, DXA_TMAP_ALIGN'(0)};
    wire [TM_LG_CL-1:0] tm_off;   // the map's byte offset in its first line
    if (TM_CL > 64) begin : g_tm_off
        assign tm_off = tm_byte[TM_LG_CL-1:0];
    end else begin : g_tm_off0
        assign tm_off = '0;
    end
    wire [TM_LADDR_W-1:0] tm_first  = tm_byte[`VX_CFG_MEM_ADDR_WIDTH-1:TM_LG_CL];
    wire [TM_CNT_W-1:0]   tm_nlines = TM_CNT_W'(((32'(tm_off) + DXA_TMAP_BYTES - 1) >> TM_LG_CL) + 1);

    reg [TM_CNT_W-1:0] tm_req_cnt_r, tm_rsp_cnt_r;
    wire tm_req_valid = (fetch_state_r == FETCH_MAP) && (tm_req_cnt_r < tm_nlines);
    wire tm_req_ready;
    wire tm_req_fire  = tm_req_valid && tm_req_ready;

    wire                     tm_rsp_valid;
    wire [TM_CL*8-1:0]       tm_rsp_data;
    wire [TM_LINE_W-1:0]     tm_rsp_line;
    wire tm_rsp_fire = tm_rsp_valid && tm_fill_ready;
    wire tm_rsp_last = tm_rsp_fire && ((tm_rsp_cnt_r + TM_CNT_W'(1)) == tm_nlines);

    // Word j of the map sits in line (off + 4j) / CL of the fetch.
    assign tm_fill_valid = tm_rsp_valid;
    assign tm_fill_entry = tm_vict_r;
    for (genvar j = 0; j < TM_STRIDE; ++j) begin : g_tm_fill
        wire [31:0] pos = 32'(tm_off) + 32'(j * 4);
        assign tm_fill_wren[j] = tm_rsp_valid && (TM_LINE_W'(pos >> TM_LG_CL) == tm_rsp_line);
        assign tm_fill_wdata[j*32 +: 32] = tm_rsp_data[((pos % TM_CL) / 4) * 32 +: 32];
    end

    always @(posedge clk) begin
        if (reset) begin
            fetch_state_r <= FETCH_IDLE;
            tm_valid_r    <= '0;
            tm_rr_r       <= '0;
            tm_kill_r     <= 1'b0;
        end else begin
            case (fetch_state_r)
                FETCH_IDLE: begin
                    if (tm_present) fetch_state_r <= FETCH_PRESENT;
                    if (tm_miss)    fetch_state_r <= FETCH_MAP;
                end
                FETCH_PRESENT: if (head_go) fetch_state_r <= FETCH_IDLE;
                default: if (tm_rsp_last) fetch_state_r <= FETCH_IDLE;
            endcase
            if (tm_miss) begin
                tm_valid_r[tm_victim] <= 1'b0;
                if (~tm_has_free)
                    tm_rr_r <= tm_rr_r + DXA_TMAP_IDX_W'(1);
            end
            if (tm_inv_pop)
                tm_valid_r <= tm_valid_r & ~tm_match;
            if (tm_rsp_last && ~tm_kill_r)
                tm_valid_r[tm_vict_r] <= 1'b1;
            if (tm_launch) begin
                tm_valid_r <= '0;
                tm_rr_r    <= '0;
            end
            if (tm_launch && (tm_miss || (fetch_state_r == FETCH_MAP)))
                tm_kill_r <= 1'b1;
            else if (fetch_state_r == FETCH_IDLE)
                tm_kill_r <= 1'b0;
        end
        if (tm_present)
            tm_cur_r <= tm_hit_idx;
        if (tm_miss) begin
            tm_vict_r    <= tm_victim;
            tm_req_cnt_r <= '0;
            tm_rsp_cnt_r <= '0;
        end
        if (tm_req_fire)
            tm_req_cnt_r <= tm_req_cnt_r + TM_CNT_W'(1);
        if (tm_rsp_fire)
            tm_rsp_cnt_r <= tm_rsp_cnt_r + TM_CNT_W'(1);
        if (tm_rsp_last)
            tm_tag_r[tm_vict_r] <= q_tm.addr;
    end

    // ---- Map decode at the head ----
    // A map's rank, its 40-bit strides and, for a map-typed reduction, its
    // element type. Invalid maps are undefined; they still run, clamped.
    wire [2:0] tm_rank     = desc_read_data.meta[DXA_DESC_META_DIM_LSB +: DXA_DESC_META_DIM_BITS];
    wire       tm_rank_bad = (tm_rank == 3'd0) || (tm_rank > 3'd5);
    wire       tm_wide     = (desc_w22 != '0);
    wire [3:0] tm_rtyp     = dxa_tmap_red_type(desc_w23[3:0]);
    wire       tm_rt_ok    = (tm_rtyp != DXA_RT_FROM_MAP) && dxa_red_ok_g(q_tm.rop, tm_rtyp);
    wire       tm_rt_nofp  = tm_rt_ok && dxa_red_fp(q_tm.rop, tm_rtyp) && (`VX_CFG_EXT_FPAMO_ENABLED == 0);
    wire       tm_rt_use   = tm_rt_ok && ~tm_rt_nofp;
    `UNUSED_VAR (desc_w23)
    `UNUSED_VAR (tm_st)

    dxa_desc_t tm_desc;
    dxa_st_t   tm_st;
    dxa_req_data_t head_req;
    always @(*) begin
        tm_desc = desc_read_data;
        if (tm_rank == 3'd0)
            tm_desc.meta[DXA_DESC_META_DIM_LSB +: DXA_DESC_META_DIM_BITS] = 3'd1;
        else if (tm_rank > 3'd5)
            tm_desc.meta[DXA_DESC_META_DIM_LSB +: DXA_DESC_META_DIM_BITS] = 3'd5;
        tm_st = dxa_red_st(tm_rt_use ? q_tm.rop : 4'd0, tm_rt_use ? tm_rtyp : 4'd0);
        head_req = queue_out_bus_if[0].req_data;
        if (q_map && q_tm.rt) begin
            head_req.st.amo_op  = tm_st.amo_op;
            head_req.st.amo_uns = tm_st.amo_uns;
            head_req.st.w8      = tm_st.w8;
            head_req.st.tsize   = tm_st.tsize;
        end
    end

    wire tm_fault = q_map && (tm_wide || tm_rank_bad || (q_tm.rt && ~tm_rt_use));
    reg  tm_held_r;
    always @(posedge clk) begin
        if (reset || (fetch_state_r != FETCH_PRESENT))
            tm_held_r <= 1'b0;
        else if (tm_fault)
            tm_held_r <= 1'b1;
    end
    assign tm_hold = tm_fault && ~tm_held_r;
    wire tm_dsp = (fetch_state_r == FETCH_PRESENT) && tm_hold;
    `RUNTIME_ASSERT(~(tm_dsp && tm_wide),
        ("%t: *** %s DXA tensor stride needs 32 bits: map=0x%0h stride_hi=0x%0h",
         $time, INSTANCE_ID, tm_byte, desc_w22))
    `RUNTIME_ASSERT(~(tm_dsp && ~tm_wide && tm_rank_bad),
        ("%t: *** %s DXA tensor map rank must be 1-5: map=0x%0h rank=%0d", $time, INSTANCE_ID, tm_byte, tm_rank))
    `RUNTIME_ASSERT(~(tm_dsp && ~tm_wide && ~tm_rank_bad && q_tm.rt && ~tm_rt_ok),
        ("%t: *** %s DXA tensor reduction type not supported by the map: map=0x%0h op=%0d type=%0d",
         $time, INSTANCE_ID, tm_byte, q_tm.rop, desc_w23[3:0]))
    `RUNTIME_ASSERT(~(tm_dsp && ~tm_wide && ~tm_rank_bad && q_tm.rt && tm_rt_nofp),
        ("%t: *** %s DXA reduction needs FPAMO: map=0x%0h op=%0d type=%0d",
         $time, INSTANCE_ID, tm_byte, q_tm.rop, desc_w23[3:0]))

    assign dispatch_in_if[0].valid     = (fetch_state_r == FETCH_PRESENT) && ~tm_hold;
    assign dispatch_in_if[0].req_data  = head_req;
    // A bulk copy reads no descriptor: it runs as a rank-1 box of size / 8
    // doublewords at its source, the issuer core's local-memory byte address,
    // or a global address (bload and its prefetch).
    wire is_gbulk = queue_out_bus_if[0].req_data.bload
                 || (queue_out_bus_if[0].req_data.prefetch && ~queue_out_bus_if[0].req_data.tload);
    // A bulk store or reduction walks its destination, gsrc, from the source.
    wire       q_st_en   = queue_out_bus_if[0].req_data.st.en;
    wire [2:0] q_st_kind = queue_out_bus_if[0].req_data.st.kind;
    wire is_sbulk = q_st_en && ((q_st_kind == DXA_ST_BULK) || (q_st_kind == DXA_ST_RED_G) || (q_st_kind == DXA_ST_RED_C));
    dxa_desc_t bulk_desc;
    always @(*) begin
        bulk_desc           = '0;
        bulk_desc.base_addr = (is_gbulk || is_sbulk) ? queue_out_bus_if[0].req_data.gsrc
                                       : `VX_CFG_MEM_ADDR_WIDTH'(queue_out_bus_if[0].req_data.src_addr);
        bulk_desc.meta      = (32'd1 << DXA_DESC_META_DIM_LSB) | (32'd3 << DXA_DESC_META_ELEMSZ_LSB);
        bulk_desc.tile01    = 32'(queue_out_bus_if[0].req_data.size >> 3);
        bulk_desc.size0     = 32'(queue_out_bus_if[0].req_data.size >> 3);
    end
    assign dispatch_in_if[0].desc_data = (queue_out_bus_if[0].req_data.bulk || is_gbulk || is_sbulk) ? bulk_desc
                                       : (q_map ? tm_desc : desc_read_data);
`else
    // Read desc_table using the queued request's desc_slot. desc_store is now
    // a registered-read BRAM (OUT_REG=1), so desc_read_data lands one cycle
    // after desc_read_addr. A 1-deep fetch register absorbs that latency: the
    // queue head's slot drives the address combinationally; the next cycle the
    // captured request and its descriptor are presented together to dispatch.
    // DXA launches are infrequent (one per multi-hundred-cycle transfer), so
    // the extra launch cycle is throughput-irrelevant.
    assign desc_read_addr = DXA_DESC_SLOT_W'(queue_out_bus_if[0].req_data.meta[DXA_DESC_SLOT_W-1:0]);

    // Bundle request + descriptor into dispatch input
    VX_dxa_worker_req_if dispatch_in_if[1]();

    // desc_store is a registered-read BRAM: desc_read_data lags desc_read_addr
    // by one cycle. A 2-state fetch FSM waits that cycle, and crucially holds
    // the queue head (so desc_read_addr — and therefore desc_read_data — stays
    // stable) until dispatch consumes. The queue is popped only on consume, so
    // a request and its descriptor are always presented in alignment, even when
    // dispatch back-pressures during streamed back-to-back transfers. Launch
    // latency is +1 cycle, negligible against a multi-hundred-cycle drain.
    localparam FETCH_IDLE = 1'b0, FETCH_PRESENT = 1'b1;
    reg fetch_state_r;

    // Pop only when presenting AND dispatch accepts; the head (hence the read
    // address) stays put through IDLE→PRESENT and across any dispatch stall.
    assign queue_out_bus_if[0].req_ready =
        (fetch_state_r == FETCH_PRESENT) && dispatch_in_if[0].ready;

    always @(posedge clk) begin
        if (reset) begin
            fetch_state_r <= FETCH_IDLE;
        end else begin
            case (fetch_state_r)
                FETCH_IDLE:    if (queue_out_bus_if[0].req_valid) fetch_state_r <= FETCH_PRESENT;
                FETCH_PRESENT: if (dispatch_in_if[0].ready)       fetch_state_r <= FETCH_IDLE;
            endcase
        end
    end

    assign dispatch_in_if[0].valid     = (fetch_state_r == FETCH_PRESENT);
    assign dispatch_in_if[0].req_data  = queue_out_bus_if[0].req_data;
    assign dispatch_in_if[0].desc_data = desc_read_data;
`endif

    VX_dxa_worker_req_if worker_req_if[`VX_CFG_NUM_DXA_CORES]();

    VX_dxa_dispatch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (`VX_CFG_NUM_DXA_CORES)
    ) issue_dispatch (
        .clk       (clk),
        .reset     (reset),
        .req_in    (dispatch_in_if),
        .req_out   (worker_req_if)
    );

    // ================================================================
    // Workers
    // ================================================================
    localparam GMEM_ARB_SEL_BITS = `ARB_SEL_BITS(`VX_CFG_NUM_DXA_CORES, GMEM_OUT_PORTS);
    localparam WORKER_GMEM_TAG_WIDTH = SOCKET_MEM_TAG_WIDTH - GMEM_ARB_SEL_BITS;

    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (WORKER_GMEM_TAG_WIDTH)
    ) worker_gmem_bus_if[`VX_CFG_NUM_DXA_CORES]();

    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) worker_smem_bus_if[`VX_CFG_NUM_DXA_CORES]();

`ifdef PERF_ENABLE
    dxa_perf_t worker_dxa_perf [`VX_CFG_NUM_DXA_CORES];
`endif
`ifdef DXA_MBAR_ENABLE
    localparam AMO_DATAW = DLMEM_CORE_W + VX_dlmem_pkg::DLMEM_REQ_DATAW;
    wire [`VX_CFG_NUM_DXA_CORES-1:0]                 w_amo_valid, w_amo_ready;
    wire [`VX_CFG_NUM_DXA_CORES-1:0][AMO_DATAW-1:0]  w_amo_data;
    wire [`VX_CFG_NUM_DXA_CORES-1:0]                 w_evt_valid, w_evt_ready;
    wire [`VX_CFG_NUM_DXA_CORES-1:0][NC_WIDTH-1:0]   w_evt_core;
    dxa_bg_compl_t [`VX_CFG_NUM_DXA_CORES-1:0]       w_evt;
`endif

    for (genvar i = 0; i < `VX_CFG_NUM_DXA_CORES; ++i) begin : g_workers
        VX_dxa_worker #(
            .INSTANCE_ID(`SFORMATF(("%s-worker%0d", INSTANCE_ID, i))),
            .WORKER_ID  (i),
            .GMEM_TAG_WIDTH (WORKER_GMEM_TAG_WIDTH)
        ) worker (
            .clk            (clk),
            .reset          (reset),
        `ifdef PERF_ENABLE
            .dxa_perf       (worker_dxa_perf[i]),
        `endif
            .req_if         (worker_req_if[i]),
            .gmem_bus_if    (worker_gmem_bus_if[i]),
        `ifdef DXA_MBAR_ENABLE
            .slot_stride    (slot_stride),
            .amo_valid      (w_amo_valid[i]),
            .amo_pkt        (w_amo_data[i][VX_dlmem_pkg::DLMEM_REQ_DATAW-1:0]),
            .amo_dst        (w_amo_data[i][VX_dlmem_pkg::DLMEM_REQ_DATAW +: DLMEM_CORE_W]),
            .amo_ready      (w_amo_ready[i]),
            .bg_evt_valid   (w_evt_valid[i]),
            .bg_evt_core    (w_evt_core[i]),
            .bg_evt         (w_evt[i]),
            .bg_evt_ready   (w_evt_ready[i]),
        `endif
            .smem_bus_if    (worker_smem_bus_if[i])
        );
    end

`ifdef DXA_MBAR_ENABLE
    // Cluster-reduction packets of the workers, into the row egress.
    wire                 amo_valid, amo_ready;
    wire [AMO_DATAW-1:0] amo_data;

    VX_stream_arb #(
        .NUM_INPUTS  (`VX_CFG_NUM_DXA_CORES),
        .NUM_OUTPUTS (1),
        .DATAW       (AMO_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     ((`VX_CFG_NUM_DXA_CORES > 1) ? 3 : 0)
    ) amo_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (w_amo_valid),
        .data_in   (w_amo_data),
        .ready_in  (w_amo_ready),
        .valid_out (amo_valid),
        .data_out  (amo_data),
        .ready_out (amo_ready),
        `UNUSED_PIN (sel_out)
    );

    // Completion events: a buffer per worker, then per core one event a cycle.
    localparam EVT_DATAW = NC_WIDTH + $bits(dxa_bg_compl_t);
    wire [`VX_CFG_NUM_DXA_CORES-1:0]                 eb_valid;
    wire [`VX_CFG_NUM_DXA_CORES-1:0][EVT_DATAW-1:0]  eb_data;
    wire [`VX_CFG_NUM_DXA_CORES-1:0]                 eb_ready;
    wire [`VX_CFG_SOCKET_SIZE-1:0][`VX_CFG_NUM_DXA_CORES-1:0] eb_ready_c;

    for (genvar i = 0; i < `VX_CFG_NUM_DXA_CORES; ++i) begin : g_evt_buf
        VX_elastic_buffer #(
            .DATAW   (EVT_DATAW),
            .SIZE    (2),
            .OUT_REG (1)
        ) evt_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (w_evt_valid[i]),
            .ready_in  (w_evt_ready[i]),
            .data_in   ({w_evt_core[i], w_evt[i]}),
            .data_out  (eb_data[i]),
            .valid_out (eb_valid[i]),
            .ready_out (eb_ready[i])
        );
        wire [NC_WIDTH-1:0] eb_core = eb_data[i][$bits(dxa_bg_compl_t) +: NC_WIDTH];
        wire [`UP(`CLOG2(`VX_CFG_SOCKET_SIZE))-1:0] eb_lc;
        if (`VX_CFG_SOCKET_SIZE > 1) begin : g_lc
            assign eb_lc = eb_core[`CLOG2(`VX_CFG_SOCKET_SIZE)-1:0];
        end else begin : g_lc0
            assign eb_lc = '0;
            `UNUSED_VAR (eb_core)
        end
        wire [`VX_CFG_SOCKET_SIZE-1:0] eb_rdy;
        for (genvar c = 0; c < `VX_CFG_SOCKET_SIZE; ++c) begin : g_rdy
            assign eb_rdy[c] = eb_ready_c[c][i] && (32'(eb_lc) == c);
        end
        assign eb_ready[i] = (| eb_rdy);
    end

    for (genvar c = 0; c < `VX_CFG_SOCKET_SIZE; ++c) begin : g_evt_core
        wire [`VX_CFG_NUM_DXA_CORES-1:0] cv;
        for (genvar i = 0; i < `VX_CFG_NUM_DXA_CORES; ++i) begin : g_cv
            wire [NC_WIDTH-1:0] cv_core = eb_data[i][$bits(dxa_bg_compl_t) +: NC_WIDTH];
            if (`VX_CFG_SOCKET_SIZE > 1) begin : g_m
                assign cv[i] = eb_valid[i] && (32'(cv_core[`CLOG2(`VX_CFG_SOCKET_SIZE)-1:0]) == c);
            end else begin : g_m0
                assign cv[i] = eb_valid[i];
                `UNUSED_VAR (cv_core)
            end
        end
        wire [EVT_DATAW-1:0] co_data;
        VX_stream_arb #(
            .NUM_INPUTS  (`VX_CFG_NUM_DXA_CORES),
            .NUM_OUTPUTS (1),
            .DATAW       (EVT_DATAW),
            .ARBITER     ("R"),
            .OUT_BUF     (3)
        ) evt_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (cv),
            .data_in   (eb_data),
            .ready_in  (eb_ready_c[c]),
            .valid_out (bg_compl_valid[c]),
            .data_out  (co_data),
            .ready_out (1'b1),
            `UNUSED_PIN (sel_out)
        );
        assign bg_compl[c] = co_data[$bits(dxa_bg_compl_t)-1:0];
        `UNUSED_VAR (co_data[$bits(dxa_bg_compl_t) +: NC_WIDTH])
    end
`endif

    // ================================================================
    // Output arbitration
    // ================================================================
`ifdef DXA_MBAR_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (TM_PORT_TAG_W)
    ) gmem_arb_out_if[GMEM_OUT_PORTS]();
`endif

    VX_mem_bus_arb #(
        .NUM_INPUTS  (`VX_CFG_NUM_DXA_CORES),
        .NUM_OUTPUTS (GMEM_OUT_PORTS),
        .DATA_SIZE   (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH   (WORKER_GMEM_TAG_WIDTH),
        .ARBITER     ("R"),
        .REQ_OUT_BUF ((`VX_CFG_NUM_DXA_CORES > 1) ? 3 : 0)
    ) gmem_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (worker_gmem_bus_if),
    `ifdef DXA_MBAR_ENABLE
        .bus_out_if (gmem_arb_out_if)
    `else
        .bus_out_if (gmem_bus_if)
    `endif
    );

`ifdef DXA_MBAR_ENABLE
    // Map line reads share port 0: the fetch wins while it has a request,
    // and its responses come back under the fetch tag bit.
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (TM_PORT_TAG_W)
    ) tm_bus_if();

    assign tm_bus_if.req_valid            = tm_req_valid;
    assign tm_bus_if.req_data.rw          = 1'b0;
    assign tm_bus_if.req_data.addr        = tm_first + TM_LADDR_W'(tm_req_cnt_r);
    assign tm_bus_if.req_data.data        = '0;
    assign tm_bus_if.req_data.byteen      = {`VX_CFG_L1_LINE_SIZE{1'b1}};
    assign tm_bus_if.req_data.attr        = '0;
    assign tm_bus_if.req_data.tag.uuid    = '0;
    assign tm_bus_if.req_data.tag.value   = {1'b1, (TM_VAL_W-1)'(tm_req_cnt_r)};

    wire gmem_rsp_tm = gmem_bus_if[0].rsp_data.tag.value[TM_VAL_W-1];

    assign gmem_bus_if[0].req_valid = tm_req_valid || gmem_arb_out_if[0].req_valid;
    assign gmem_bus_if[0].req_data  = tm_req_valid ? tm_bus_if.req_data : gmem_arb_out_if[0].req_data;
    assign gmem_arb_out_if[0].req_ready = gmem_bus_if[0].req_ready && ~tm_req_valid;
    assign tm_bus_if.req_ready = gmem_bus_if[0].req_ready;
    assign gmem_arb_out_if[0].rsp_valid = gmem_bus_if[0].rsp_valid && ~gmem_rsp_tm;
    assign gmem_arb_out_if[0].rsp_data  = gmem_bus_if[0].rsp_data;
    assign tm_bus_if.rsp_valid = gmem_bus_if[0].rsp_valid && gmem_rsp_tm;
    assign tm_bus_if.rsp_data  = gmem_bus_if[0].rsp_data;
    assign tm_bus_if.rsp_ready = tm_fill_ready;
    assign gmem_bus_if[0].rsp_ready = gmem_rsp_tm ? tm_bus_if.rsp_ready : gmem_arb_out_if[0].rsp_ready;

    assign tm_req_ready = tm_bus_if.req_ready;
    assign tm_rsp_valid = tm_bus_if.rsp_valid;
    assign tm_rsp_data  = tm_bus_if.rsp_data.data;
    assign tm_rsp_line  = TM_LINE_W'(tm_bus_if.rsp_data.tag.value);
    `UNUSED_VAR (tm_bus_if.rsp_data.tag)

    for (genvar i = 1; i < GMEM_OUT_PORTS; ++i) begin : g_gmem_pass
        assign gmem_bus_if[i].req_valid     = gmem_arb_out_if[i].req_valid;
        assign gmem_bus_if[i].req_data      = gmem_arb_out_if[i].req_data;
        assign gmem_arb_out_if[i].req_ready = gmem_bus_if[i].req_ready;
        assign gmem_arb_out_if[i].rsp_valid = gmem_bus_if[i].rsp_valid;
        assign gmem_arb_out_if[i].rsp_data  = gmem_bus_if[i].rsp_data;
        assign gmem_bus_if[i].rsp_ready     = gmem_arb_out_if[i].rsp_ready;
    end
`endif

`ifdef DXA_MBAR_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) lmem_arb_out_if[1]();
`endif

    VX_mem_bus_arb #(
        .NUM_INPUTS  (`VX_CFG_NUM_DXA_CORES),
        .NUM_OUTPUTS (1),
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_TAG_W),
        .TAG_SEL_IDX (DXA_LMEM_TAG_W - UUID_WIDTH),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W),
        .ARBITER     ("R"),
        // Register the SMEM write request at the DXA boundary: the tiled
        // dest-address is a deep cone whose sink is the far-away core LMEM BRAM,
        // so terminate it at a local flop to keep placement compact and isolate
        // the long DXA→core route into its own cycle.
        .REQ_OUT_BUF (3)
    ) lmem_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (worker_smem_bus_if),
    `ifdef DXA_MBAR_ENABLE
        .bus_out_if (lmem_arb_out_if)
    `else
        .bus_out_if (smem_bus_if)
    `endif
    );

`ifdef DXA_MBAR_ENABLE
    // A remote copy's beats leave through the row egress, the rest to the cores.
    dxa_lmem_attr_t arb_out_attr;
    assign arb_out_attr = dxa_lmem_attr_t'(lmem_arb_out_if[0].req_data.attr);
    wire arb_out_remote = arb_out_attr.remote;
    wire row_beat_ready;

    assign smem_bus_if[0].req_valid = lmem_arb_out_if[0].req_valid && ~arb_out_remote;
    assign smem_bus_if[0].req_data  = lmem_arb_out_if[0].req_data;
    assign lmem_arb_out_if[0].req_ready = arb_out_remote ? row_beat_ready : smem_bus_if[0].req_ready;
    assign lmem_arb_out_if[0].rsp_valid = smem_bus_if[0].rsp_valid;
    assign lmem_arb_out_if[0].rsp_data  = smem_bus_if[0].rsp_data;
    assign smem_bus_if[0].rsp_ready = lmem_arb_out_if[0].rsp_ready;

    VX_dlmem_row_egress #(
        .INSTANCE_ID (`SFORMATF(("%s-row-egress", INSTANCE_ID))),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W)
    ) row_egress (
        .clk         (clk),
        .reset       (reset),
        .beat_valid  (lmem_arb_out_if[0].req_valid && arb_out_remote),
        .beat_addr   (lmem_arb_out_if[0].req_data.addr),
        .beat_data   (lmem_arb_out_if[0].req_data.data),
        .beat_byteen (lmem_arb_out_if[0].req_data.byteen),
        .beat_attr   (arb_out_attr),
        .beat_tag    (lmem_arb_out_if[0].req_data.tag),
        .beat_ready  (row_beat_ready),
        .amo_valid   (amo_valid),
        .amo_pkt     (amo_data[VX_dlmem_pkg::DLMEM_REQ_DATAW-1:0]),
        .amo_dst     (amo_data[VX_dlmem_pkg::DLMEM_REQ_DATAW +: DLMEM_CORE_W]),
        .amo_ready   (amo_ready),
        .row_out_if  (row_out_if)
    );
`endif

    // ================================================================
    // Busy / perf
    // ================================================================
    wire [NUM_REQS-1:0] req_bus_valid;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_valid
        assign req_bus_valid[i] = req_bus_if[i].req_valid;
    end

    wire [`VX_CFG_NUM_DXA_CORES-1:0] worker_idle;
    for (genvar i = 0; i < `VX_CFG_NUM_DXA_CORES; ++i) begin : g_worker_idle
        assign worker_idle[i] = worker_req_if[i].ready;
    end

    reg dxa_busy_r;
    always @(posedge clk) begin
        if (reset)
            dxa_busy_r <= 1'b0;
        else
            dxa_busy_r <= dcr_bus_if.req_valid | (|req_bus_valid) | ~(&worker_idle);
    end
`ifdef DXA_MBAR_ENABLE
    assign busy = dxa_busy_r | dcr_bus_if.req_valid | (|req_bus_valid) | (fetch_state_r == FETCH_MAP);
`else
    assign busy = dxa_busy_r | dcr_bus_if.req_valid | (|req_bus_valid);
`endif

`ifdef PERF_ENABLE
`ifdef DXA_MBAR_ENABLE
    // A lookup counts once: the hit after a fill is the miss's own.
    reg [PERF_CTR_BITS-1:0] perf_tm_hits_r, perf_tm_misses_r;
    reg tm_refill_r;
    always @(posedge clk) begin
        if (reset) begin
            perf_tm_hits_r   <= '0;
            perf_tm_misses_r <= '0;
            tm_refill_r      <= 1'b0;
        end else begin
            if (idle_q)
                tm_refill_r <= 1'b0;
            if (tm_rsp_last)
                tm_refill_r <= 1'b1;
            if ((tm_pf_pop || (tm_present && q_map)) && ~tm_refill_r)
                perf_tm_hits_r <= perf_tm_hits_r + PERF_CTR_BITS'(1);
            if (tm_miss)
                perf_tm_misses_r <= perf_tm_misses_r + PERF_CTR_BITS'(1);
        end
    end
`endif
    always_comb begin
        dxa_perf = '0;
        for (int w = 0; w < `VX_CFG_NUM_DXA_CORES; ++w) begin
            dxa_perf.transfers   += worker_dxa_perf[w].transfers;
            dxa_perf.gmem_reads  += worker_dxa_perf[w].gmem_reads;
            dxa_perf.gmem_dedup  += worker_dxa_perf[w].gmem_dedup;
            dxa_perf.lmem_writes += worker_dxa_perf[w].lmem_writes;
            dxa_perf.gmem_latency += worker_dxa_perf[w].gmem_latency;
            dxa_perf.noslot_stalls += worker_dxa_perf[w].noslot_stalls;
        end
    `ifdef DXA_MBAR_ENABLE
        dxa_perf.tmap_hits   = perf_tm_hits_r;
        dxa_perf.tmap_misses = perf_tm_misses_r;
    `endif
    end
`endif

`ifdef DBG_TRACE_DXA
    // Interface-array indexing must be elaboration-time constant; use a
    // genvar loop instead of an integer loop.
    for (genvar w = 0; w < `VX_CFG_NUM_DXA_CORES; ++w) begin : g_trace_worker
        always @(posedge clk) begin
            if (~reset && worker_req_if[w].valid) begin
                `TRACE(1, ("%t: %s dispatch-issue: worker=%0d, core=%0d, wid=%0d, meta=0x%0h\n",
                    $time, INSTANCE_ID, w,
                    worker_req_if[w].req_data.core_id,
                    worker_req_if[w].req_data.wid,
                    worker_req_if[w].req_data.meta))
            end
        end
    end
`endif

endmodule
