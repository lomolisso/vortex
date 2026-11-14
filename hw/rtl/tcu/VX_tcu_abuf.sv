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

//
// Per-block A buffer (k-stripe storage, supports both block-major and
// row-major SMEM layouts).
//
// Holds the M_STEPS A-blocks for the current k-stripe of one warp slot.
// Storage is A_STRIPE_WORDS = M_STEPS × tcM × tcK 32-bit words, indexed
// by `row_idx * tcK + k_in` where row_idx ∈ [0, M_STEPS*tcM). The output
// mux is layout-agnostic — only the fetch path differs.
//
// Mode is selected by the descriptor's stride field (req_desc_a[31:16]):
//   stride == 0 → block-major. SMEM layout:
//     A_smem[(k_blk * M_STEPS + m_blk) * BLOCK_WORDS + i*TC_K + k_in]
//   stride != 0 → row-major (matches WGMMA SS descriptors and
//     DXA-loaded slabs). ldm_words = stride_bytes/4 = words per row.
//     A_smem[row * ldm_words + col_word].
//
// Block-major refill key = {desc_a, step_k}; one fetch reads M_STEPS
// adjacent bank-rows (one full stripe) starting at
//   base_a + step_k * M_STEPS * BLOCK_BANK_ROWS.
//
// Row-major refill key = {desc_a, step_k}; the FSM issues one LMEM read
// per row (M_STEPS*tcM total) at addr
//   base_a + (r * ldm_words + step_k * tcK) >> log2(BANK_ROW_WORDS)
// and extracts tcK consecutive words at lane
//   (r * ldm_words + step_k * tcK) & (BANK_ROW_WORDS-1).
//
// Output rs1_data carries one A-block per cycle, selected by step_m.
// Format-aware sub-word extraction (fp16/fp8 packing) happens
// downstream in the FEDP grid; abuf is word pass-through.
//

module VX_tcu_abuf import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         NUM_BANKS       = 4,
    parameter         BANK_ADDR_WIDTH = 12
`ifdef VX_CFG_TCU_H100_ENABLE
  , parameter         BLOCK_IDX       = 0
`endif
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] abuf_stalls,
    output wire [PERF_CTR_BITS-1:0] lmem_reads,
`endif

    // Execute-side observation (req_valid is already gated to WGMMA)
    input  wire [NW_WIDTH-1:0]      req_wid,
    input  wire                     req_valid,
    input  wire                     req_setup,
    input  wire [3:0]               req_step_m,
`ifdef VX_CFG_TCU_H100_ENABLE
    input  wire [4:0]               req_step_n,
`else
    input  wire [3:0]               req_step_n,
`endif
    input  wire [3:0]               req_step_k,
    input  wire [`VX_CFG_XLEN-1:0]  req_desc_a,
    input  wire                     req_a_is_smem,
    input  wire                     req_is_sparse,
    input  wire [UUID_WIDTH-1:0]    req_uuid,
`ifdef VX_CFG_TCU_H100_ENABLE
    input  wire                     req_is_first_uop,
    input  wire [`VX_CFG_LMEM_LOG_SIZE-1:0] req_slot_base,
    input  wire                     req_esize16,
`endif

    // LMEM bank-parallel read port
    VX_mem_bus_if.master            tcu_lmem_if,

    // Outputs
    output wire                     abuf_ready,
    output wire [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] abuf_rs1_data
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (req_wid)

`ifdef VX_CFG_TCU_H100_ENABLE
    // -----------------------------------------------------------------------
    // Operands read through wgmma shared-memory descriptors.
    //
    // The buffer holds one k-stripe of this block's 16 rows of the m64 tile:
    // rows 16 * BLOCK_IDX + r, r in [0, 16), one 16-byte chunk of K each,
    // stored as storage[r * FEDP_K + word]. A fill is 16 single-outstanding
    // reads of one 16-byte chunk:
    //   K-major:  read i is row i at the stripe's K chunk;
    //   MN-major: read i is k-row (i >> 1) of the stripe and MN chunk (i & 1),
    //             whose eight 16-bit elements go to eight rows, one half-word
    //             each.
    // -----------------------------------------------------------------------

    localparam LMEM_LOG       = `VX_CFG_LMEM_LOG_SIZE;
    localparam BANK_SEL_BITS  = $clog2(NUM_BANKS);
    localparam WORD_SIZE_LOG2 = $clog2(`VX_CFG_XLEN / 8);
    localparam ROW_BYTES_LOG  = BANK_SEL_BITS + WORD_SIZE_LOG2;
    localparam ROW_HWORDS     = NUM_BANKS * (`VX_CFG_XLEN / 16);
    localparam LANE_W         = ROW_BYTES_LOG - 2;
    localparam A_ROWS         = TCU_WG_M_STEPS * TCU_TC_M;
    localparam A_WORDS        = A_ROWS * TCU_WG_FEDP_K;
    localparam NUM_READS      = A_ROWS;
    localparam CTR_W          = `CLOG2(NUM_READS + 1);
    localparam K_STEPS_W      = `CLOG2(TCU_WG_K_STEPS);

    `STATIC_ASSERT (TCU_WG_FEDP_K == 4, ("wgmma descriptors need one 16-byte chunk per row and step"))
    `STATIC_ASSERT (TCU_TC_M == 8 && A_ROWS == 16, ("wgmma descriptors need 16 rows of 8 per block"))
    `STATIC_ASSERT (ROW_BYTES_LOG >= 4, ("a local-memory row must hold a 16-byte chunk"))
    `STATIC_ASSERT (LMEM_LOG == BANK_ADDR_WIDTH + ROW_BYTES_LOG, ("local-memory address width mismatch"))

    `UNUSED_VAR (req_setup)
    `UNUSED_VAR (req_step_n)
    `UNUSED_VAR (req_is_sparse)

    wire [`UP(K_STEPS_W)-1:0] req_step_k_trunc = `UP(K_STEPS_W)'(req_step_k);
    if (4 > K_STEPS_W) begin : g_step_k_upper_unused
        `UNUSED_VAR (req_step_k[3:`UP(K_STEPS_W)])
    end

    logic                      slot_valid_r;
    logic [`UP(K_STEPS_W)-1:0] slot_step_k_r;
    logic                      fetching_r;
    logic                      refetched_r;
    logic [63:0]               f_desc_r;
    logic [LMEM_LOG-1:0]       f_base_r;
    logic                      f_esize16_r;
    logic [CTR_W-1:0]          req_ctr_r;
    logic [CTR_W-1:0]          rsp_ctr_r;
    logic                      req_inflight_r;
    logic [LANE_W-1:0]         rsp_lane_r;

    // A WGMMA's first micro-op always refills: a K-tile loop rewrites the
    // slab in place and reissues the same descriptors.
    wire stripe_resident = slot_valid_r
                        && (slot_step_k_r == req_step_k_trunc)
                        && (!req_is_first_uop || refetched_r);

    wire need_smem = req_valid && req_a_is_smem;
    wire alloc_en  = need_smem && !stripe_resident && !fetching_r;

    assign abuf_ready = !req_valid || !req_a_is_smem || stripe_resident;
    wire fire = req_valid && abuf_ready;

    wire f_mn_major = f_desc_r[52];
    wire all_requested = (req_ctr_r == CTR_W'(NUM_READS));
    wire can_issue = fetching_r && !all_requested
                  && (!req_inflight_r || tcu_lmem_if.rsp_valid);
    wire last_rsp  = fetching_r && tcu_lmem_if.rsp_valid
                  && (rsp_ctr_r == CTR_W'(NUM_READS - 1));

    wire [3:0] rd_idx = req_ctr_r[3:0];
    wire [7:0] rd_mn  = 8'(BLOCK_IDX * A_ROWS)
                      + (f_mn_major ? {4'b0, rd_idx[0], 3'b0} : {4'b0, rd_idx});
    wire [5:0] rd_k   = f_mn_major ? 6'({slot_step_k_r, rd_idx[3:1]})
                                   : (f_esize16_r ? 6'({slot_step_k_r, 3'b0})
                                                  : 6'({slot_step_k_r, 4'b0}));

    wire [LMEM_LOG-1:0] rd_addr;
    VX_tcu_smem_addr #(
        .LMEM_LOG (LMEM_LOG)
    ) smem_addr (
        .clk       (clk),
        .reset     (reset),
        .check     (can_issue),
        .desc      (f_desc_r),
        .slot_base (f_base_r),
        .esize16   (f_esize16_r),
        .mn        (rd_mn),
        .k         (rd_k),
        .addr      (rd_addr)
    );

    assign tcu_lmem_if.req_valid       = can_issue;
    assign tcu_lmem_if.req_data.rw     = 1'b0;
    assign tcu_lmem_if.req_data.addr   = rd_addr[LMEM_LOG-1:ROW_BYTES_LOG];
    assign tcu_lmem_if.req_data.data   = '0;
    assign tcu_lmem_if.req_data.byteen = '0;
    assign tcu_lmem_if.req_data.attr   = '0;
    assign tcu_lmem_if.req_data.tag.uuid  = req_uuid;
    assign tcu_lmem_if.req_data.tag.value = '0;
    assign tcu_lmem_if.rsp_ready       = 1'b1;
    `UNUSED_VAR (tcu_lmem_if.rsp_data.tag)
    `UNUSED_VAR (rd_addr[1:0])

    always_ff @(posedge clk) begin
        if (reset) begin
            slot_valid_r   <= 1'b0;
            slot_step_k_r  <= '0;
            fetching_r     <= 1'b0;
            refetched_r    <= 1'b0;
            f_desc_r       <= '0;
            f_base_r       <= '0;
            f_esize16_r    <= 1'b0;
            req_ctr_r      <= '0;
            rsp_ctr_r      <= '0;
            req_inflight_r <= 1'b0;
            rsp_lane_r     <= '0;
        end else begin
            if (tcu_lmem_if.rsp_valid)
                req_inflight_r <= 1'b0;
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready) begin
                req_inflight_r <= 1'b1;
                req_ctr_r      <= req_ctr_r + CTR_W'(1);
                rsp_lane_r     <= rd_addr[ROW_BYTES_LOG-1:2];
            end
            if (fetching_r && tcu_lmem_if.rsp_valid)
                rsp_ctr_r <= rsp_ctr_r + CTR_W'(1);

            if (last_rsp && req_valid && req_is_first_uop)
                refetched_r <= 1'b1;
            else if (fire && !req_is_first_uop)
                refetched_r <= 1'b0;

            if (alloc_en) begin
                fetching_r     <= 1'b1;
                slot_valid_r   <= 1'b0;
                slot_step_k_r  <= req_step_k_trunc;
                f_desc_r       <= 64'(req_desc_a);
                f_base_r       <= req_slot_base;
                f_esize16_r    <= req_esize16;
                req_ctr_r      <= '0;
                rsp_ctr_r      <= '0;
                req_inflight_r <= 1'b0;
            end else if (last_rsp) begin
                fetching_r   <= 1'b0;
                slot_valid_r <= 1'b1;
            end
        end
    end

    // Storage in half-words, so a MN-major chunk can scatter its elements.
    logic [A_WORDS*32-1:0] storage_wdata;
    logic [A_WORDS*2-1:0]  storage_wren;

    always_comb begin
        storage_wdata = '0;
        storage_wren  = '0;
        if (fetching_r && tcu_lmem_if.rsp_valid) begin
            for (int h = 0; h < 8; ++h) begin
                automatic int src = int'(rsp_lane_r) * 2 + h;
                automatic int dst;
                if (f_mn_major) begin
                    // k-row kk of MN chunk c: element h is row 8c + h, half kk.
                    automatic int kk = int'(rsp_ctr_r[3:1]);
                    automatic int c  = int'(rsp_ctr_r[0]);
                    dst = ((8 * c + h) * TCU_WG_FEDP_K + kk / 2) * 2 + (kk % 2);
                end else begin
                    dst = int'(rsp_ctr_r[3:0]) * TCU_WG_FEDP_K * 2 + h;
                end
                if (src < ROW_HWORDS && dst < A_WORDS * 2) begin
                    storage_wren[dst] = 1'b1;
                    storage_wdata[dst * 16 +: 16] = tcu_lmem_if.rsp_data.data[src * 16 +: 16];
                end
            end
        end
    end

    wire [A_WORDS-1:0][31:0] storage_rdata;

    VX_dp_ram #(
        .DATAW   (A_WORDS * 32),
        .SIZE    (1),
        .WRENW   (A_WORDS * 2),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) storage_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (fetching_r && tcu_lmem_if.rsp_valid),
        .wren  (storage_wren),
        .waddr (1'b0),
        .wdata (storage_wdata),
        .raddr (1'b0),
        .rdata (storage_rdata)
    );

    localparam A_BLOCK_WORDS = TCU_TC_M * TCU_WG_FEDP_K;

    logic [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] rs1_mux;
    always_comb begin
        rs1_mux = '0;
        for (int lane = 0; lane < TCU_WG_A_DATA_SIZE; ++lane) begin
            automatic int src_idx = int'(req_step_m) * A_BLOCK_WORDS + lane;
            if (lane < A_BLOCK_WORDS && src_idx < A_WORDS)
                rs1_mux[lane] = `VX_CFG_XLEN'(storage_rdata[src_idx]);
        end
    end
    assign abuf_rs1_data = rs1_mux;

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] stall_ctr_r;
    reg [PERF_CTR_BITS-1:0] reads_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            stall_ctr_r <= '0;
            reads_ctr_r <= '0;
        end else begin
            if (req_valid && !abuf_ready)
                stall_ctr_r <= stall_ctr_r + PERF_CTR_BITS'(1);
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                reads_ctr_r <= reads_ctr_r + PERF_CTR_BITS'(1);
        end
    end
    assign abuf_stalls = stall_ctr_r;
    assign lmem_reads  = reads_ctr_r;
`endif

`ifdef DBG_TRACE_TCU
    // One line per event, fields as the SimX tile-buffer trace, for trace diffs.
    always @(posedge clk) begin
        if (!reset) begin
            if (alloc_en)
                `TRACE(1, ("%t: tcutr a_alloc b=%0d k=%0d desc=0x%0h base=0x%0h\n",
                    $time, BLOCK_IDX, req_step_k, req_desc_a, req_slot_base))
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                `TRACE(1, ("%t: tcutr a_req b=%0d i=%0d addr=0x%0h\n",
                    $time, BLOCK_IDX, req_ctr_r, rd_addr))
            if (fetching_r && tcu_lmem_if.rsp_valid)
                `TRACE(1, ("%t: tcutr a_rsp b=%0d i=%0d\n", $time, BLOCK_IDX, rsp_ctr_r))
        end
    end
`endif

`else

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    localparam BANK_SEL_BITS      = $clog2(NUM_BANKS);
    localparam WORD_SIZE_LOG2     = $clog2(`VX_CFG_XLEN / 8);
    localparam A_BLOCK_WORDS      = TCU_TC_M * TCU_WG_FEDP_K;
    localparam A_BLOCK_WORDS_SP   = TCU_TC_M * TCU_TC_K;
    localparam A_BLOCK_BANK_ROWS  = (A_BLOCK_WORDS + NUM_BANKS - 1) / NUM_BANKS;
    localparam A_BLOCK_BANK_ROWS_SP = (A_BLOCK_WORDS_SP + NUM_BANKS - 1) / NUM_BANKS;
    localparam BLOCK_WORDS_PADDED = A_BLOCK_BANK_ROWS * NUM_BANKS;
    localparam BLOCK_WORDS_PADDED_SP = A_BLOCK_BANK_ROWS_SP * NUM_BANKS;
    localparam A_STRIPE_BANK_ROWS = TCU_WG_M_STEPS * A_BLOCK_BANK_ROWS;
    localparam A_STRIPE_BANK_ROWS_SP = TCU_WG_M_STEPS * A_BLOCK_BANK_ROWS_SP;
    localparam A_STRIPE_WORDS     = A_STRIPE_BANK_ROWS * NUM_BANKS;
    // XLEN ratio: each physical LMEM bank-row carries XLEN_RATIO logical
    // 32-bit bank-rows side-by-side. Smem layout is XLEN-independent, so the
    // LMEM fetch count and stride must scale down by XLEN_RATIO.
    localparam XLEN_RATIO         = `VX_CFG_XLEN / 32;
    localparam BANK_ROW_WORDS     = NUM_BANKS * XLEN_RATIO; // 32-bit words per LMEM bank-row
    localparam BANK_ROW_WORDS_LOG2 = $clog2(BANK_ROW_WORDS);
    localparam A_STRIPE_LMEM_ROWS = (A_STRIPE_WORDS + BANK_ROW_WORDS - 1) / BANK_ROW_WORDS;
    localparam A_STRIPE_LMEM_ROWS_SP = (A_STRIPE_BANK_ROWS_SP * NUM_BANKS + BANK_ROW_WORDS - 1) / BANK_ROW_WORDS;
    // Row-major path issues one LMEM read per logical row (M_STEPS*TC_M rows).
    localparam A_TOTAL_ROWS       = TCU_WG_M_STEPS * TCU_TC_M;
    localparam FETCH_CTR_W_BM     = `CLOG2(A_STRIPE_LMEM_ROWS + 1);
    localparam FETCH_CTR_W_RM     = `CLOG2(A_TOTAL_ROWS + 1);
    localparam FETCH_CTR_W        = (FETCH_CTR_W_BM > FETCH_CTR_W_RM) ? FETCH_CTR_W_BM : FETCH_CTR_W_RM;
    localparam K_STEPS_W          = `CLOG2(TCU_WG_K_STEPS);
    // ldm field width: desc[31:16] is a 16-bit byte stride; >>2 gives the
    // per-row 32-bit-word stride. Cap at 14 bits (= 64KB smem / 4B).
    localparam LDM_W              = 14;

    // Canonical configs fit an A block in one bank-row normally and two
    // bank-rows with dense FEDP2K.

    // -----------------------------------------------------------------------
    // Resident slot state
    // -----------------------------------------------------------------------

    logic                       slot_valid_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_desc_a_row_base_r;
    logic [`UP(K_STEPS_W)-1:0]  slot_step_k_r;
    logic                       slot_fetching_r;
    // Row-major mode: latched per-row stride. ldm_words=0 means block-major;
    // the fetch FSM falls through to the existing path. Latched at alloc_en
    // alongside fetch_base_r and slot_step_k_r.
    logic                       slot_row_major_r;
    logic [LDM_W-1:0]           slot_ldm_words_r;
    logic                       slot_is_sparse_r;

    wire [`UP(K_STEPS_W)-1:0]   req_step_k_trunc = `UP(K_STEPS_W)'(req_step_k);
    if (4 > K_STEPS_W) begin : g_step_k_upper_unused
        `UNUSED_VAR (req_step_k[3:`UP(K_STEPS_W)])
    end

    // The WGMMA wrapper supplies the setup-latched descriptor on compute uops.
    wire is_first_compute_uop = (req_step_m == '0) && (req_step_n == '0) && (req_step_k == '0);
    `UNUSED_VAR (req_step_n)  // only used in is_first_uop computation

    // Force a refetch on every WGMMA's first compute uop. A cooperative-load
    // pattern (K-tile loop rewrites A_warp_smem in place, issues a fresh
    // WGMMA with an unchanged descriptor) would incorrectly hit the cached
    // stripe and serve stale A data if residency were checked by desc_a alone.
    // refetched_for_first_compute_r gates stripe_resident: cleared until the
    // current WGMMA's first compute fetch completes, then set; cleared again
    // on later compute uops so the next WGMMA always triggers a fresh fetch.
    reg refetched_for_first_compute_r;
    wire stripe_resident = slot_valid_r
                        && (slot_step_k_r == req_step_k_trunc)
                        && (!is_first_compute_uop || refetched_for_first_compute_r);

    // RS mode (a_from_smem=0): A from registers, abuf bypassed → always ready.
    wire need_smem  = req_valid && req_a_is_smem;
    wire need_fetch = need_smem && !stripe_resident;
    wire alloc_en   = need_fetch && !slot_fetching_r;

    assign abuf_ready = !req_valid || !req_a_is_smem || stripe_resident;

    wire fire = req_valid && abuf_ready;

    // -----------------------------------------------------------------------
    // Address compute
    // -----------------------------------------------------------------------
    // base_a is the bank-row address of block(k=0, m=0). One k-stripe is
    // M_STEPS adjacent bank-rows starting at base + k * M_STEPS.

    localparam DESC_ADDR_W = BANK_ADDR_WIDTH + BANK_SEL_BITS;
    wire [DESC_ADDR_W-1:0]      desc_a_word_base = DESC_ADDR_W'(req_desc_a[15:0] >> WORD_SIZE_LOG2);
    wire [BANK_ADDR_WIDTH-1:0]  desc_a_row_base  = desc_a_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    if (BANK_SEL_BITS > 0) begin : g_addr_lsb_unused
        `UNUSED_VAR (desc_a_word_base[BANK_SEL_BITS-1:0])
    end
    // desc_a's lower 16 bits encode the smem offset; upper 16 bits carry the
    // per-row byte stride (matches WGMMA SS descriptor). Stride=0 keeps
    // the canonical block-major layout; non-zero selects the row-major path.
    wire [LDM_W-1:0] desc_a_ldm_words = LDM_W'(req_desc_a[31:16] >> 2);
    if (`VX_CFG_XLEN > 32) begin : g_desc_a_upper_unused
        `UNUSED_VAR (req_desc_a[`VX_CFG_XLEN-1:32])
    end

    localparam STRIPE_STRIDE_BANK_ROWS = A_STRIPE_LMEM_ROWS;

    // Compute uops use the setup-latched descriptor. Without this,
    // k-stripe-transition refills mid-WGMMA would compute the wrong fetch_base.
    wire [BANK_ADDR_WIDTH-1:0]  effective_desc_a_row_base =
        is_first_compute_uop ? desc_a_row_base : slot_desc_a_row_base_r;
    wire [LDM_W-1:0]            effective_ldm_words =
        is_first_compute_uop ? desc_a_ldm_words : slot_ldm_words_r;
    wire                        effective_row_major = (effective_ldm_words != '0);

    // Block-major stripe base (unchanged): one fetch covers M_STEPS A-blocks.
    wire [BANK_ADDR_WIDTH-1:0]  stripe_base =
        effective_desc_a_row_base
      + BANK_ADDR_WIDTH'(req_step_k_trunc) * BANK_ADDR_WIDTH'(STRIPE_STRIDE_BANK_ROWS);
    // M_STEPS, A_BLOCK_BANK_ROWS are compile-time; STRIPE_STRIDE_BANK_ROWS
    // is a power of 2 in canonical configs, so synth folds the multiply
    // into a shift.

    // -----------------------------------------------------------------------
    // Fetch FSM
    // -----------------------------------------------------------------------

    typedef enum logic {
        S_IDLE  = 1'b0,
        S_FETCH = 1'b1
    } state_e;
    state_e fsm_state_r;

    wire in_fetch = (fsm_state_r == S_FETCH);

    logic [FETCH_CTR_W-1:0]     req_ctr_r;
    logic [FETCH_CTR_W-1:0]     rsp_ctr_r;
    logic                       req_inflight_r;
    logic [BANK_ADDR_WIDTH-1:0] fetch_base_r;

    // Total fetches differs by mode:
    //   block-major: A_STRIPE_LMEM_ROWS adjacent bank-row reads
    //   row-major:   A_TOTAL_ROWS per-row reads (one per logical row)
    wire [FETCH_CTR_W-1:0] target_fetches = slot_row_major_r
        ? FETCH_CTR_W'(A_TOTAL_ROWS)
        : (slot_is_sparse_r ? FETCH_CTR_W'(A_STRIPE_LMEM_ROWS_SP)
                            : FETCH_CTR_W'(A_STRIPE_LMEM_ROWS));

    wire all_requested = (req_ctr_r >= target_fetches);
    wire can_issue     = in_fetch && !all_requested
                      && (!req_inflight_r || tcu_lmem_if.rsp_valid);
    wire last_rsp      = in_fetch && tcu_lmem_if.rsp_valid
                      && (rsp_ctr_r == target_fetches - FETCH_CTR_W'(1));

    // Per-row 32-bit-word offset (relative to fetch_base_r * BANK_ROW_WORDS).
    // Width = LDM (row stride) + counter + a bit for step_k*TC_K headroom.
    // The offset is linear in the fetch counter and the first issue is at
    // least one cycle after alloc: the step_k product is registered at alloc
    // and each request/response accumulates the row stride, keeping the
    // multiplier off the LMEM request-address and response-lane paths.
    localparam ROW_OFF_W = LDM_W + FETCH_CTR_W + 4;
    wire [3:0] row_k_words = slot_is_sparse_r ? 4'(TCU_TC_K) : 4'(TCU_WG_FEDP_K);
    wire [3:0] row_k_words_alloc = req_is_sparse ? 4'(TCU_TC_K) : 4'(TCU_WG_FEDP_K);

    wire [ROW_OFF_W-1:0] row_word_off_init =
        ROW_OFF_W'(req_step_k_trunc) * ROW_OFF_W'(row_k_words_alloc);

    logic [ROW_OFF_W-1:0] row_word_off_req_r;
    logic [ROW_OFF_W-1:0] row_word_off_rsp_r;

    wire [BANK_ADDR_WIDTH-1:0] row_lmem_addr =
        fetch_base_r + BANK_ADDR_WIDTH'(row_word_off_req_r >> BANK_ROW_WORDS_LOG2);

    // LMEM master driver
    assign tcu_lmem_if.req_valid       = can_issue;
    assign tcu_lmem_if.req_data.rw     = 1'b0;
    assign tcu_lmem_if.req_data.addr   = slot_row_major_r
        ? row_lmem_addr
        : (fetch_base_r + BANK_ADDR_WIDTH'(req_ctr_r));
    assign tcu_lmem_if.req_data.data   = '0;
    assign tcu_lmem_if.req_data.byteen = '0;
    assign tcu_lmem_if.req_data.attr   = '0;
    assign tcu_lmem_if.req_data.tag.uuid  = req_uuid;
    assign tcu_lmem_if.req_data.tag.value = '0;
    assign tcu_lmem_if.rsp_ready       = 1'b1;
    `UNUSED_VAR (tcu_lmem_if.rsp_data.tag)

    always_ff @(posedge clk) begin
        if (reset) begin
            fsm_state_r            <= S_IDLE;
            req_ctr_r              <= '0;
            rsp_ctr_r              <= '0;
            row_word_off_req_r     <= '0;
            row_word_off_rsp_r     <= '0;
            req_inflight_r         <= 1'b0;
            slot_valid_r           <= 1'b0;
            slot_fetching_r        <= 1'b0;
            slot_desc_a_row_base_r <= '0;
            slot_step_k_r          <= '0;
            fetch_base_r           <= '0;
            slot_row_major_r       <= 1'b0;
            slot_ldm_words_r       <= '0;
            slot_is_sparse_r       <= 1'b0;
            refetched_for_first_compute_r <= 1'b0;
        end else begin
            if (req_setup) begin
                fsm_state_r      <= S_IDLE;
                req_ctr_r        <= '0;
                rsp_ctr_r        <= '0;
                req_inflight_r   <= 1'b0;
                slot_valid_r     <= 1'b0;
                slot_fetching_r  <= 1'b0;
                refetched_for_first_compute_r <= 1'b0;
            end else begin
                // Inflight tracker (single outstanding request at a time)
                if (tcu_lmem_if.rsp_valid)
                    req_inflight_r <= 1'b0;
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_inflight_r <= 1'b1;

                // Latch descriptor fields on the first compute uop.
                if (req_valid && is_first_compute_uop) begin
                    slot_desc_a_row_base_r <= desc_a_row_base;
                    slot_ldm_words_r       <= desc_a_ldm_words;
                    slot_row_major_r       <= (desc_a_ldm_words != '0);
                end

                if (last_rsp && is_first_compute_uop)
                    refetched_for_first_compute_r <= 1'b1;
                else if (fire && !is_first_compute_uop)
                    refetched_for_first_compute_r <= 1'b0;

                case (fsm_state_r)
                    S_IDLE: begin
                        if (alloc_en) begin
                            fsm_state_r      <= S_FETCH;
                            slot_fetching_r  <= 1'b1;
                            slot_valid_r     <= 1'b0;
                            slot_step_k_r    <= req_step_k_trunc;
                            slot_is_sparse_r  <= req_is_sparse;
                            // Row-major base is the A_warp start bank-row (no
                            // step_k offset; per-row arithmetic derives the
                            // exact LMEM addr). Block-major base is the stripe
                            // origin (step_k already factored in).
                            fetch_base_r     <= effective_row_major
                                ? effective_desc_a_row_base
                                : stripe_base;
                            req_ctr_r        <= '0;
                            rsp_ctr_r        <= '0;
                            row_word_off_req_r <= row_word_off_init;
                            row_word_off_rsp_r <= row_word_off_init;
                            req_inflight_r   <= 1'b0;
                        end
                    end
                    S_FETCH: begin
                        if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready) begin
                            req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                            row_word_off_req_r <= row_word_off_req_r + ROW_OFF_W'(slot_ldm_words_r);
                        end
                        if (last_rsp) begin
                            fsm_state_r     <= S_IDLE;
                            slot_fetching_r <= 1'b0;
                            slot_valid_r    <= 1'b1;
                            req_inflight_r  <= 1'b0;
                        end else if (tcu_lmem_if.rsp_valid) begin
                            rsp_ctr_r <= rsp_ctr_r + FETCH_CTR_W'(1);
                            row_word_off_rsp_r <= row_word_off_rsp_r + ROW_OFF_W'(slot_ldm_words_r);
                        end
                    end
                    default: fsm_state_r <= S_IDLE;
                endcase
            end
        end
    end

    // -----------------------------------------------------------------------
    // Storage (LUTRAM): A_STRIPE_BANK_ROWS × NUM_BANKS 32-bit words.
    // Each LMEM response writes one bank-row at offset rsp_ctr_r.
    // -----------------------------------------------------------------------

    logic [A_STRIPE_WORDS*32-1:0] storage_wdata;
    logic [A_STRIPE_WORDS-1:0]    storage_wren;

    // Row-major per-response decoding: the row being written is rsp_ctr_r,
    // and its tcK source words start at lane = row_word_off & (BANK_ROW_WORDS-1)
    // within the LMEM response (registered accumulator, see above).
    wire [BANK_ROW_WORDS_LOG2:0] row_lane_rsp = (BANK_ROW_WORDS_LOG2+1)'(
        row_word_off_rsp_r & ROW_OFF_W'(BANK_ROW_WORDS - 1));

    always_comb begin
        storage_wdata = '0;
        storage_wren  = '0;
        if (in_fetch && tcu_lmem_if.rsp_valid) begin
            if (slot_row_major_r) begin
                // Row-major: write k_words of row rsp_ctr_r into
                // storage[rsp_ctr_r * k_words .. + k_words). Source words start
                // at row_lane_rsp inside the LMEM response.
                for (int k = 0; k < TCU_WG_FEDP_K; ++k) begin
                    automatic int dst = int'(rsp_ctr_r) * int'(row_k_words) + k;
                    automatic int src = int'(row_lane_rsp) + k;
                    if (k < int'(row_k_words) && dst < A_STRIPE_WORDS && src < BANK_ROW_WORDS) begin
                        storage_wren[dst] = 1'b1;
                        storage_wdata[dst * 32 +: 32] =
                            tcu_lmem_if.rsp_data.data[src * 32 +: 32];
                    end
                end
            end else begin
                // Block-major: one LMEM response carries BANK_ROW_WORDS
                // 32-bit words (= NUM_BANKS * XLEN_RATIO). Store all of them
                // starting at the current LMEM-row offset within the stripe.
                for (int b = 0; b < BANK_ROW_WORDS; ++b) begin
                    if (int'(rsp_ctr_r) * BANK_ROW_WORDS + b < A_STRIPE_WORDS) begin
                        storage_wren[int'(rsp_ctr_r) * BANK_ROW_WORDS + b] = 1'b1;
                        storage_wdata[(int'(rsp_ctr_r) * BANK_ROW_WORDS + b) * 32 +: 32] =
                            tcu_lmem_if.rsp_data.data[b * 32 +: 32];
                    end
                end
            end
        end
    end

    wire [A_STRIPE_WORDS-1:0][31:0] storage_rdata;

    VX_dp_ram #(
        .DATAW   (A_STRIPE_WORDS * 32),
        .SIZE    (1),
        .WRENW   (A_STRIPE_WORDS),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) storage_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (in_fetch && tcu_lmem_if.rsp_valid),
        .wren  (storage_wren),
        .waddr (1'b0),
        .wdata (storage_wdata),
        .raddr (1'b0),
        .rdata (storage_rdata)
    );

    // -----------------------------------------------------------------------
    // Output: select A-block based on step_m. Sparse data is stored compactly,
    // then expanded into FEDP_K row stride with the unavailable tail lanes zero.
    // -----------------------------------------------------------------------

    logic [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] rs1_mux;
    always_comb begin
        rs1_mux = '0;
        for (int lane = 0; lane < TCU_WG_A_DATA_SIZE; ++lane) begin
            int src_idx;
            if (req_is_sparse) begin
                automatic int row = lane / int'(TCU_WG_FEDP_K);
                automatic int k   = lane % int'(TCU_WG_FEDP_K);
                if (row < int'(TCU_TC_M) && k < int'(TCU_TC_K)) begin
                    src_idx = int'(req_step_m) * int'(BLOCK_WORDS_PADDED_SP)
                            + row * int'(TCU_TC_K)
                            + k;
                    if (src_idx < int'(A_STRIPE_WORDS))
                        rs1_mux[lane] = `VX_CFG_XLEN'(storage_rdata[src_idx]);
                end
            end else if (lane < int'(A_BLOCK_WORDS)) begin
                src_idx = int'(req_step_m) * int'(BLOCK_WORDS_PADDED) + lane;
                if (src_idx < int'(A_STRIPE_WORDS))
                    rs1_mux[lane] = `VX_CFG_XLEN'(storage_rdata[src_idx]);
            end
        end
    end
    assign abuf_rs1_data = rs1_mux;

    // -----------------------------------------------------------------------
    // Performance counters
    // -----------------------------------------------------------------------

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] stall_ctr_r;
    reg [PERF_CTR_BITS-1:0] reads_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            stall_ctr_r <= '0;
            reads_ctr_r <= '0;
        end else begin
            if (req_valid && !abuf_ready)
                stall_ctr_r <= stall_ctr_r + PERF_CTR_BITS'(1);
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                reads_ctr_r <= reads_ctr_r + PERF_CTR_BITS'(1);
        end
    end
    assign abuf_stalls = stall_ctr_r;
    assign lmem_reads  = reads_ctr_r;
`endif

    // -----------------------------------------------------------------------
    // Debug trace
    // -----------------------------------------------------------------------

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (!reset) begin
            if (alloc_en)
                `TRACE(3, ("%t: %s abuf: alloc desc_a=0x%0h step_k=%0d base=0x%0h\n",
                    $time, INSTANCE_ID, req_desc_a, req_step_k, stripe_base))
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                `TRACE(3, ("%t: %s abuf: rd_req addr=0x%0h req_ctr=%0d\n",
                    $time, INSTANCE_ID, tcu_lmem_if.req_data.addr, req_ctr_r))
            if (tcu_lmem_if.rsp_valid)
                `TRACE(3, ("%t: %s abuf: rd_rsp rsp_ctr=%0d\n",
                    $time, INSTANCE_ID, rsp_ctr_r))
            if (last_rsp)
                `TRACE(3, ("%t: %s abuf: stripe READY\n", $time, INSTANCE_ID))
        end
    end
`endif

`endif // VX_CFG_TCU_H100_ENABLE

endmodule

`endif // VX_CFG_TCU_WGMMA_ENABLE
