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

// DXA worker: purely structural — wire declarations and module
// instantiations. All logic lives in the submodules:
//   setup → addr_gen → gmem_req → smem_wr + watchdog

`include "VX_define.vh"

module VX_dxa_worker import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter WORKER_ID = 0,
    parameter GMEM_TAG_WIDTH = SOCKET_MEM_TAG_WIDTH
) (
    input wire clk,
    input wire reset,
`ifdef PERF_ENABLE
    output dxa_perf_t dxa_perf,
`endif
    VX_dxa_worker_req_if.slave req_if,
    VX_mem_bus_if.master gmem_bus_if,
`ifdef DXA_MBAR_ENABLE
    input wire [`VX_CFG_SOCKET_SIZE-1:0][`VX_CFG_LMEM_LOG_SIZE:0] slot_stride,
    // Cluster-reduction packets, to the row egress.
    output wire                     amo_valid,
    output VX_dlmem_pkg::dlmem_req_pkt_t amo_pkt,
    output wire [DLMEM_CORE_W-1:0]  amo_dst,
    input  wire                     amo_ready,
    // Bulk-group completion events, to the issuing core.
    output wire                     bg_evt_valid,
    output wire [NC_WIDTH-1:0]      bg_evt_core,
    output dxa_bg_compl_t           bg_evt,
    input  wire                     bg_evt_ready,
`endif
    VX_mem_bus_if.master smem_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_SPARAM (WORKER_ID)

    localparam GMEM_BYTES      = `VX_CFG_L1_LINE_SIZE;
    localparam GMEM_DATAW      = GMEM_BYTES * 8;
    localparam GMEM_OFF_BITS   = `CLOG2(GMEM_BYTES);
    localparam GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - GMEM_OFF_BITS;

    localparam SMEM_BYTES      = DXA_LMEM_WORD_SIZE;
    localparam SMEM_ADDR_WIDTH = DXA_LMEM_ADDR_W;

    localparam MAX_OUTSTANDING = `VX_CFG_DXA_MAX_INFLIGHT;
    localparam TAG_W = `CLOG2(MAX_OUTSTANDING);
    localparam SEQ_W = `CLOG2(MAX_OUTSTANDING + 1);

    // ════════════════════════════════════════════════════════════════════
    // Inter-module wires
    // ════════════════════════════════════════════════════════════════════

    // setup → downstream
    wire                        transfer_active;
    wire                        pipeline_start;
    dxa_setup_params_t          setup_params;
    wire [NC_WIDTH-1:0]         active_core_id;
    wire [UUID_WIDTH-1:0]       active_uuid;
    wire [NW_WIDTH-1:0]         active_wid;
    wire [BAR_ADDR_W-1:0]       active_bar_addr;
    wire                        active_notify_smem_done;
`ifdef DXA_MBAR_ENABLE
    dxa_compl_t                 active_compl;
    wire                        active_bulk;
    dxa_st_t                    active_st;
    wire                        active_bg;
    wire [DXA_BG_SEQ_W-1:0]     active_bg_seq;
`endif
    wire                        active_is_multicast;
    wire [`VX_CFG_NUM_WARPS-1:0]       active_cta_mask;
    wire [31:0]                 active_smem_stride;

    // addr_gen → gmem_req
    wire                        ag_valid;
    wire                        ag_ready;
    wire [GMEM_ADDR_WIDTH-1:0]  ag_cl_addr;
    wire [DXA_SMEM_ADDR_W-1:0]  ag_smem_byte_addr;
    wire [GMEM_OFF_BITS-1:0]    ag_byte_offset;
    wire [GMEM_OFF_BITS:0]      ag_valid_length;
    wire                        ag_oob;
    wire                        ag_last;
    wire [31:0]                 ag_cfill;
    wire                        ag_dest_kmajor;
    wire [15:0]                 ag_per_lane_stride_bytes;
    wire [3:0]                  ag_elem_bytes;
    wire [1:0]                  ag_dest_mode;
    wire [3:0]                  ag_lg_ratio;
    wire [3:0]                  ag_lg_tcN;
    wire [3:0]                  ag_lg_nsteps;
    wire [3:0]                  ag_lg_bkK;
    wire [15:0]                 ag_k_row;
    wire [15:0]                 ag_n_base;
    wire [DXA_SMEM_ADDR_W-1:0]  ag_smem_base;
`ifdef DXA_MBAR_ENABLE
    wire                        ag_pf_end;
    dxa_tma_wr_t                ag_tma_wr;
    assign ag_tma_wr.en       = setup_params.tma.en;
    assign ag_tma_wr.prefetch = setup_params.tma.prefetch;
    assign ag_tma_wr.es0      = setup_params.tma.es0;
    assign ag_tma_wr.swizzle  = setup_params.tma.swizzle;
    assign ag_tma_wr.blk_base = setup_params.tma.blk_base;
`endif

    // gmem_req → smem_wr (direct-drain channel).
    wire                        sw_valid;
    wire                        sw_ready;
    wire [TAG_W-1:0]            sw_tag;
    wire [GMEM_DATAW-1:0]       sw_data;
    wire [DXA_SMEM_ADDR_W-1:0]  sw_smem_byte_addr;
    wire [GMEM_OFF_BITS-1:0]    sw_byte_offset;
    wire [GMEM_OFF_BITS:0]      sw_valid_length;
    wire                        sw_oob;
    wire                        sw_last;
    wire [15:0]                 sw_k_row;
    wire [15:0]                 sw_n_base;
    wire [SEQ_W-1:0]            sw_outstanding;

    // smem_wr → gmem_req (release)
    wire                        sw_release_en;
    wire [TAG_W-1:0]            sw_release_tag;

    // smem_wr → setup (completion)
    wire                        transfer_done;
`ifdef DXA_MBAR_ENABLE
    wire                        sw_transfer_done;
    wire                        st_done;
    wire                        st_progress;
    assign transfer_done = sw_transfer_done || st_done;
`endif
    wire [31:0]                 wr_done_count;
    wire                        smem_req_fire;

    // gmem_req → watchdog
    wire                        gmem_req_fire;
    wire                        stall_no_slot;

`ifdef DXA_MBAR_ENABLE
    // gmem_req ↔ lmem_rd (a bulk copy's line reads), smem_wr → lmem_rd (its beats)
    wire                        lr_req_valid;
    wire [GMEM_ADDR_WIDTH-1:0]  lr_req_addr;
    wire [TAG_W-1:0]            lr_req_tag;
    wire                        lr_req_ready;
    wire                        lr_rsp_valid;
    wire [GMEM_DATAW-1:0]       lr_rsp_data;
    wire [TAG_W-1:0]            lr_rsp_tag;
    wire                        lr_rsp_ready;
    wire                        st_rd_valid;
    wire [DXA_LMEM_ADDR_W-1:0]  st_rd_addr;
    wire                        st_rd_ready;
    wire                        st_rsp_valid;
    wire [SMEM_BYTES*8-1:0]     st_rsp_data;

    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) wr_bus_if();
`endif

`ifdef PERF_ENABLE
    wire [31:0]                 perf_gmem_reqs;
    wire [31:0]                 perf_gmem_span_cycles;
    wire [31:0]                 perf_lmem_writes;
`endif

    // ════════════════════════════════════════════════════════════════════
    // Stage 1: Setup
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_setup setup (
        .clk                  (clk),
        .reset                (reset),
        .req_valid            (req_if.valid),
        .req_ready            (req_if.ready),
        .req_data             (req_if.req_data),
        .desc_data            (req_if.desc_data),
        .transfer_done        (transfer_done),
        .transfer_active      (transfer_active),
        .pipeline_start       (pipeline_start),
        .setup_params         (setup_params),
        .active_core_id       (active_core_id),
        .active_uuid          (active_uuid),
        .active_wid           (active_wid),
        .active_bar_addr      (active_bar_addr),
        .active_notify_smem_done (active_notify_smem_done),
    `ifdef DXA_MBAR_ENABLE
        .active_compl         (active_compl),
        .active_bulk          (active_bulk),
        .active_st            (active_st),
        .active_bg            (active_bg),
        .active_bg_seq        (active_bg_seq),
    `endif
        .active_is_multicast  (active_is_multicast),
        .active_cta_mask      (active_cta_mask),
        .active_smem_stride   (active_smem_stride)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 2: Address Generator
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_addr_gen #(
        .GMEM_LINE_SIZE  (GMEM_BYTES),
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH)
    ) addr_gen (
        .clk                  (clk),
        .reset                (reset),
        .start                (pipeline_start),
        .setup_params         (setup_params),
        .out_valid            (ag_valid),
        .out_ready            (ag_ready),
        .out_cl_addr          (ag_cl_addr),
        .out_smem_byte_addr   (ag_smem_byte_addr),
        .out_byte_offset      (ag_byte_offset),
        .out_valid_length     (ag_valid_length),
        .out_oob              (ag_oob),
        .out_last             (ag_last),
        .out_cfill            (ag_cfill),
        .out_dest_kmajor      (ag_dest_kmajor),
        .out_per_lane_stride_bytes (ag_per_lane_stride_bytes),
        .out_elem_bytes       (ag_elem_bytes),
        .out_dest_mode        (ag_dest_mode),
        .out_lg_ratio         (ag_lg_ratio),
        .out_lg_tcN           (ag_lg_tcN),
        .out_lg_nsteps        (ag_lg_nsteps),
        .out_lg_bkK           (ag_lg_bkK),
        .out_k_row            (ag_k_row),
        .out_n_base           (ag_n_base),
        .out_smem_base        (ag_smem_base)
    `ifdef DXA_MBAR_ENABLE
      , .out_pf_end           (ag_pf_end)
    `endif
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 3: GMEM Request Issuer
    // ════════════════════════════════════════════════════════════════════

    // The request issuer registers its own gmem output (VX_mem_bus_slice), so
    // gmem_bus_if is already a clean flopped producer boundary here.
`ifdef DXA_MBAR_ENABLE
    // A store's tokens go to the store engine, its requests share the port.
    wire st_mode = active_st.en;
    wire gr_ag_ready, st_ag_ready;
    assign ag_ready = st_mode ? st_ag_ready : gr_ag_ready;

    VX_mem_bus_if #(
        .DATA_SIZE (GMEM_BYTES),
        .TAG_WIDTH (GMEM_TAG_WIDTH)
    ) gr_bus_if(), st_bus_if();

    assign gmem_bus_if.req_valid = gr_bus_if.req_valid || st_bus_if.req_valid;
    assign gmem_bus_if.req_data  = st_bus_if.req_valid ? st_bus_if.req_data : gr_bus_if.req_data;
    assign gr_bus_if.req_ready   = gmem_bus_if.req_ready;
    assign st_bus_if.req_ready   = gmem_bus_if.req_ready;
    assign gr_bus_if.rsp_valid   = gmem_bus_if.rsp_valid && ~st_mode;
    assign st_bus_if.rsp_valid   = gmem_bus_if.rsp_valid && st_mode;
    assign gr_bus_if.rsp_data    = gmem_bus_if.rsp_data;
    assign st_bus_if.rsp_data    = gmem_bus_if.rsp_data;
    assign gmem_bus_if.rsp_ready = st_mode ? st_bus_if.rsp_ready : gr_bus_if.rsp_ready;
`endif

    VX_dxa_gmem_req #(
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .GMEM_ADDR_WIDTH (GMEM_ADDR_WIDTH),
        .GMEM_TAG_WIDTH  (GMEM_TAG_WIDTH),
        .CL_OFF_BITS     (GMEM_OFF_BITS),
        .SMEM_ADDR_W     (DXA_SMEM_ADDR_W)
    ) gmem_req (
        .clk                (clk),
        .reset              (reset),
    `ifdef PERF_ENABLE
        .perf_gmem_reqs       (perf_gmem_reqs),
        .perf_gmem_span_cycles(perf_gmem_span_cycles),
    `endif
        .transfer_active    (transfer_active),
        .transfer_start     (pipeline_start),
        .active_uuid        (active_uuid),
    `ifdef DXA_MBAR_ENABLE
        .ag_valid           (ag_valid && ~st_mode),
        .ag_ready           (gr_ag_ready),
    `else
        .ag_valid           (ag_valid),
        .ag_ready           (ag_ready),
    `endif
        .ag_cl_addr         (ag_cl_addr),
        .ag_smem_byte_addr  (ag_smem_byte_addr),
        .ag_byte_offset     (ag_byte_offset),
        .ag_valid_length    (ag_valid_length),
        .ag_oob             (ag_oob),
        .ag_last            (ag_last),
        .ag_k_row           (ag_k_row),
        .ag_n_base          (ag_n_base),
    `ifdef DXA_MBAR_ENABLE
        .bulk               (active_bulk),
        .lmem_rd_valid      (lr_req_valid),
        .lmem_rd_addr       (lr_req_addr),
        .lmem_rd_tag        (lr_req_tag),
        .lmem_rd_ready      (lr_req_ready),
        .lmem_rsp_valid     (lr_rsp_valid),
        .lmem_rsp_data      (lr_rsp_data),
        .lmem_rsp_tag       (lr_rsp_tag),
        .lmem_rsp_ready     (lr_rsp_ready),
        .gmem_bus_if        (gr_bus_if),
    `else
        .gmem_bus_if        (gmem_bus_if),
    `endif
        .sw_valid           (sw_valid),
        .sw_ready           (sw_ready),
        .sw_tag             (sw_tag),
        .sw_data            (sw_data),
        .sw_smem_byte_addr  (sw_smem_byte_addr),
        .sw_byte_offset     (sw_byte_offset),
        .sw_valid_length    (sw_valid_length),
        .sw_oob             (sw_oob),
        .sw_last            (sw_last),
        .sw_k_row           (sw_k_row),
        .sw_n_base          (sw_n_base),
        .sw_outstanding     (sw_outstanding),
        .release_en         (sw_release_en),
        .release_tag        (sw_release_tag),
        .gmem_req_fire      (gmem_req_fire),
        .stall_no_slot      (stall_no_slot)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 4: SMEM Writer (gmem_req drives smem_wr directly via the
    // sw_* channel; smem_wr's pend slot + deferred-last register handle
    // out-of-order drain).
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_smem_wr #(
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .CL_SIZE         (GMEM_BYTES),
        .SMEM_WORD_SIZE  (SMEM_BYTES),
        .SMEM_ADDR_WIDTH (SMEM_ADDR_WIDTH),
        .GMEM_DATAW      (GMEM_DATAW)
    ) smem_wr (
        .clk                   (clk),
        .reset                 (reset),
    `ifdef PERF_ENABLE
        .perf_lmem_writes      (perf_lmem_writes),
    `endif
        .transfer_active       (transfer_active),
        .transfer_start        (pipeline_start),
        .cfill                 (ag_cfill),
        .active_core_id        (active_core_id),
        .active_uuid           (active_uuid),
        .active_bar_addr       (active_bar_addr),
        .active_notify_smem_done (active_notify_smem_done),
    `ifdef DXA_MBAR_ENABLE
        .active_compl          (active_compl),
        .slot_stride           (slot_stride),
        .tma                   (ag_tma_wr),
        .pf_end                (ag_pf_end),
    `endif
        .sw_valid              (sw_valid),
        .sw_ready              (sw_ready),
        .sw_tag                (sw_tag),
        .sw_data               (sw_data),
        .sw_smem_byte_addr     (sw_smem_byte_addr),
        .sw_byte_offset        (sw_byte_offset),
        .sw_valid_length       (sw_valid_length),
        .sw_oob                (sw_oob),
        .sw_last               (sw_last),
        .sw_k_row              (sw_k_row),
        .sw_n_base             (sw_n_base),
        .sw_outstanding        (sw_outstanding),
        .release_en            (sw_release_en),
        .release_tag           (sw_release_tag),
    `ifdef DXA_MBAR_ENABLE
        .smem_bus_if           (wr_bus_if),
    `else
        .smem_bus_if           (smem_bus_if),
    `endif
    `ifdef DXA_MBAR_ENABLE
        .transfer_done         (sw_transfer_done),
    `else
        .transfer_done         (transfer_done),
    `endif
        .wr_done_count         (wr_done_count),
        .smem_req_fire         (smem_req_fire),
        .is_multicast          (active_is_multicast),
        .cta_mask              (active_cta_mask),
        .smem_stride           (active_smem_stride),
        .dest_kmajor           (ag_dest_kmajor),
        .per_lane_stride_bytes (ag_per_lane_stride_bytes),
        .elem_bytes            (ag_elem_bytes),
        .dest_mode             (ag_dest_mode),
        .lg_ratio              (ag_lg_ratio),
        .lg_tcN                (ag_lg_tcN),
        .lg_nsteps             (ag_lg_nsteps),
        .lg_bkK                (ag_lg_bkK),
        .smem_base             (ag_smem_base)
    );

`ifdef DXA_MBAR_ENABLE
    // ════════════════════════════════════════════════════════════════════
    // Local-memory line reader (bulk copies); merges smem_wr's beats onto
    // the worker's bus.
    // ════════════════════════════════════════════════════════════════════

    // The line reader's entries, apart from the DXA_MAX_INFLIGHT line tags:
    // the fewest with the throughput of 8 at 4, 8 and 16 banks.
    localparam LMEM_RD_DEPTH = 7;

    VX_dxa_lmem_rd #(
        .INSTANCE_ID     (`SFORMATF(("%s-lmem-rd", INSTANCE_ID))),
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .DEPTH           (LMEM_RD_DEPTH),
        .CL_SIZE         (GMEM_BYTES),
        .WORD_SIZE       (SMEM_BYTES),
        .LINE_ADDR_W     (GMEM_ADDR_WIDTH)
    ) lmem_rd (
        .clk            (clk),
        .reset          (reset),
        .core_id        (active_core_id),
        .uuid           (active_uuid),
        .word_mode      (st_mode),
        .word_req_valid (st_rd_valid),
        .word_req_addr  (st_rd_addr),
        .word_req_ready (st_rd_ready),
        .word_rsp_valid (st_rsp_valid),
        .word_rsp_data  (st_rsp_data),
        .line_req_valid (lr_req_valid),
        .line_req_addr  (lr_req_addr),
        .line_req_tag   (lr_req_tag),
        .line_req_ready (lr_req_ready),
        .line_rsp_valid (lr_rsp_valid),
        .line_rsp_data  (lr_rsp_data),
        .line_rsp_tag   (lr_rsp_tag),
        .line_rsp_ready (lr_rsp_ready),
        .wr_bus_if      (wr_bus_if),
        .smem_bus_if    (smem_bus_if)
    );

    // ════════════════════════════════════════════════════════════════════
    // Store engine (funct3 5): local memory to global memory or a block.
    // ════════════════════════════════════════════════════════════════════
    wire st_evt_valid, st_evt_rd, st_evt_wr;

    VX_dxa_st #(
        .INSTANCE_ID    (`SFORMATF(("%s-st", INSTANCE_ID))),
        .CL_SIZE        (GMEM_BYTES),
        .WORD_SIZE      (SMEM_BYTES),
        .LINE_ADDR_W    (GMEM_ADDR_WIDTH),
        .GMEM_TAG_WIDTH (GMEM_TAG_WIDTH)
    ) st_eng (
        .clk               (clk),
        .reset             (reset),
        .start             (pipeline_start),
        .active            (transfer_active && st_mode),
        .st                (active_st),
        .bg                (active_bg),
        .tma               (ag_tma_wr),
        .elem_bytes        (ag_elem_bytes),
        .core_id           (active_core_id),
        .uuid              (active_uuid),
        .dst_core          (active_compl.dst_core),
        .dst_slot          (active_compl.slot),
        .mbar_off          (active_compl.mbar_off),
        .ag_valid          (ag_valid && st_mode),
        .ag_ready          (st_ag_ready),
        .ag_cl_addr        (ag_cl_addr),
        .ag_smem_byte_addr (ag_smem_byte_addr),
        .ag_byte_offset    (ag_byte_offset),
        .ag_valid_length   (ag_valid_length),
        .ag_oob            (ag_oob),
        .ag_last           (ag_last),
        .rd_valid          (st_rd_valid),
        .rd_addr           (st_rd_addr),
        .rd_ready          (st_rd_ready),
        .rsp_valid         (st_rsp_valid),
        .rsp_data          (st_rsp_data),
        .gmem_bus_if       (st_bus_if),
        .amo_valid         (amo_valid),
        .amo_pkt           (amo_pkt),
        .amo_dst           (amo_dst),
        .amo_ready         (amo_ready),
        .evt_valid         (st_evt_valid),
        .evt_read_done     (st_evt_rd),
        .evt_write_done    (st_evt_wr),
        .evt_ready         (bg_evt_ready),
        .done              (st_done),
        .progress          (st_progress)
    );

    assign bg_evt_valid    = st_evt_valid;
    assign bg_evt_core     = active_core_id;
    assign bg_evt.wid        = active_wid;
    assign bg_evt.seq        = active_bg_seq;
    assign bg_evt.read_done  = st_evt_rd;
    assign bg_evt.write_done = st_evt_wr;

`endif
    // ════════════════════════════════════════════════════════════════════
    // Watchdog
    // ════════════════════════════════════════════════════════════════════

    VX_dxa_watchdog #(
        .INSTANCE_ID (INSTANCE_ID)
    ) watchdog (
        .clk              (clk),
        .reset            (reset),
        .transfer_active  (transfer_active),
    `ifdef DXA_MBAR_ENABLE
        .gmem_req_fire    (gmem_req_fire || st_progress),
    `else
        .gmem_req_fire    (gmem_req_fire),
    `endif
        .gmem_rsp_valid   (sw_valid && sw_ready),  // any CL delivered to smem_wr
        .smem_req_fire    (smem_req_fire),
        .active_core_id   (active_core_id),
        .active_wid       (active_wid),
        .active_bar_addr  (active_bar_addr)
    );

    // ════════════════════════════════════════════════════════════════════
    // Perf counters
    // ════════════════════════════════════════════════════════════════════

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_transfers_r;
    reg [PERF_CTR_BITS-1:0] perf_gmem_reads_r;
    reg [PERF_CTR_BITS-1:0] perf_gmem_dedup_r;
    reg [PERF_CTR_BITS-1:0] perf_lmem_writes_r;
    reg [PERF_CTR_BITS-1:0] perf_gmem_lt_r;
    reg [PERF_CTR_BITS-1:0] perf_noslot_r;
    always @(posedge clk) begin
        if (reset) begin
            perf_transfers_r   <= '0;
            perf_gmem_reads_r  <= '0;
            perf_gmem_dedup_r  <= '0;
            perf_lmem_writes_r <= '0;
            perf_gmem_lt_r     <= '0;
            perf_noslot_r      <= '0;
        end else begin
            if (stall_no_slot) begin
                perf_noslot_r <= perf_noslot_r + PERF_CTR_BITS'(1);
            end
            if (transfer_active && transfer_done) begin
                perf_transfers_r   <= perf_transfers_r + PERF_CTR_BITS'(1);
                perf_gmem_reads_r  <= perf_gmem_reads_r + PERF_CTR_BITS'(perf_gmem_reqs);
                perf_lmem_writes_r <= perf_lmem_writes_r + PERF_CTR_BITS'(perf_lmem_writes);
                perf_gmem_lt_r     <= perf_gmem_lt_r + PERF_CTR_BITS'(perf_gmem_span_cycles);
            end
        end
    end
    assign dxa_perf.transfers    = perf_transfers_r;
    assign dxa_perf.gmem_reads   = perf_gmem_reads_r;
    assign dxa_perf.gmem_dedup   = perf_gmem_dedup_r;
    assign dxa_perf.lmem_writes  = perf_lmem_writes_r;
    assign dxa_perf.gmem_latency = perf_gmem_lt_r;
    assign dxa_perf.noslot_stalls = perf_noslot_r;
`else
    `UNUSED_VAR (stall_no_slot)
`endif
`ifndef DBG_TRACE_DXA
    `UNUSED_VAR (wr_done_count)
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (pipeline_start) begin
                `TRACE(1, ("%t: %s setup-done: core=%0d, wid=%0d, bar=%0d\n",
                    $time, INSTANCE_ID, active_core_id, active_wid, active_bar_addr))
                $write("DXA_TL,%0d,SETUP_DONE,core=%0d,wid=%0d,bar=%0d\n",
                    $time, active_core_id, active_wid, active_bar_addr);
            end
            if (transfer_active) begin
                if (gmem_req_fire) begin
                    `TRACE(2, ("%t: %s gmem-req-fire\n", $time, INSTANCE_ID))
                end
                if (sw_valid && sw_ready) begin
                    `TRACE(2, ("%t: %s sw-deliver: tag=%0d, oob=%0d, last=%0d\n",
                        $time, INSTANCE_ID, sw_tag, sw_oob, sw_last))
                end
                if (smem_req_fire) begin
                    `TRACE(2, ("%t: %s smem-wr-fire: count=%0d\n",
                        $time, INSTANCE_ID, wr_done_count))
                end
            end
            if (transfer_active && transfer_done) begin
                `TRACE(1, ("%t: %s done: core=%0d, wid=%0d, bar=%0d, wr_count=%0d\n",
                    $time, INSTANCE_ID, active_core_id, active_wid,
                    active_bar_addr, wr_done_count))
                $write("DXA_TL,%0d,XFER_DONE,core=%0d,wid=%0d,bar=%0d,wr_count=%0d\n",
                    $time, active_core_id, active_wid, active_bar_addr,
                    wr_done_count);
            end
        end
    end
`endif

endmodule
