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

// DXA Address Generator: CL-aware address + narrow metadata tokens.
// Supports 1D-5D tiles via nested odometer counters for outer dimensions.
// Inner loop: CL within a row (dim 0). Outer dims 1..4: odometer advance.
//
// Two-loop structure. The inner loop (CL within a row) advances every beat
// and is pure counters/flags — no 32-bit compares, no DSP on the path:
//   cl_addr_r   : GMEM CL address, ++1 / beat
//   lines_left_r: CLs remaining in the row, --1 / beat; is_last_line == (==1)
//   is_first_r  : current CL is the row's first
//   smem_byte_addr_r += a registered step
// The outer odometer (gmem_cursor, dim counters, num_lines, deltas) changes
// only at a row wrap, so its adds/compares sit on once-per-row paths.
//
// Single rolling-cursor scheme: gmem_cursor_r is updated by per-wrap
// deltas precomputed in setup (delta[0]=stride[0]; delta[d>0]=stride[d]
// minus the (tile-1)*stride span of every inner outer dim, as all of them wrap).

`include "VX_define.vh"

module VX_dxa_addr_gen import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter GMEM_LINE_SIZE  = `VX_CFG_L1_LINE_SIZE,
    parameter GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(GMEM_LINE_SIZE),
    localparam CL_OFF_BITS    = `CLOG2(GMEM_LINE_SIZE)
) (
    input  wire                        clk,
    input  wire                        reset,

    // Start + setup params (latched on start).
    input  wire                        start,
    input  dxa_setup_params_t          setup_params,

    // CL request output (valid/ready).
    output wire                        out_valid,
    input  wire                        out_ready,
    output wire [GMEM_ADDR_WIDTH-1:0]  out_cl_addr,
    output wire [DXA_SMEM_ADDR_W-1:0]  out_smem_byte_addr,
    output wire [CL_OFF_BITS-1:0]      out_byte_offset,
    output wire [CL_OFF_BITS:0]        out_valid_length,
    output wire                        out_oob,
    output wire                        out_last,

    // Pass-through params for downstream (stable during transfer).
    output wire [31:0]                 out_cfill,
    output wire                        out_dest_kmajor,
    output wire [15:0]                 out_per_lane_stride_bytes,
    output wire [3:0]                  out_elem_bytes,
    // Tiled (Flat/BlockMajor) scatter: stable geometry + per-CL (k_row, n_base).
    output wire [1:0]                  out_dest_mode,
    output wire [3:0]                  out_lg_ratio,
    output wire [3:0]                  out_lg_tcN,
    output wire [3:0]                  out_lg_nsteps,
    output wire [3:0]                  out_lg_bkK,
    output wire [15:0]                 out_k_row,    // dim1 index (K-row) of this CL
    output wire [15:0]                 out_n_base,   // N index of this CL's element 0
    output wire [DXA_SMEM_ADDR_W-1:0]  out_smem_base // tile SMEM byte base (for tiled dest)
`ifdef DXA_MBAR_ENABLE
    // A prefetch walk ended after tokens it dropped (a pulse).
  , output wire                        out_pf_end
`endif
);
    localparam VLEN_W      = CL_OFF_BITS + 1;

    `STATIC_ASSERT(`IS_POW2(GMEM_LINE_SIZE), ("GMEM_LINE_SIZE must be power of 2"))

    // ---- Registered state ----
    reg                        active_r;
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0] gmem_cursor_r;  // Current row's GMEM byte base
    reg [31:0]                 row_len_r;            // row_len_bytes (constant)

    // Inner-loop counters/flags (advance every beat).
    reg [GMEM_ADDR_WIDTH-1:0]  cl_addr_r;            // Current CL address (GMEM)
    reg [31:0]                 lines_left_r;         // CLs remaining in current row
    reg                        is_first_r;           // current CL is the row's first
    reg [VLEN_W-1:0]           last_vend_r;          // valid_end for the row's last CL

    // Outer-dim odometer.
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] dim_count_r;   // Current index per dim
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] dim_last_r;    // Last index per dim (tile-1)
`ifdef DXA_MBAR_ENABLE
    reg [DXA_MAX_OUTER_DIMS-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] delta_r;  // Wrap delta per dim, signed
`else
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] delta_r;       // Wrap delta per dim
`endif
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0] oob_limit_r;   // OOB limit per dim

    // SMEM byte address tracking.
    reg [DXA_SMEM_ADDR_W-1:0]  smem_byte_addr_r;
    // K-major scatter-mode state.
    //   km_row_base_r: SMEM base for the current outer-dim row (= initial +
    //                  dim_count[0] * elem_bytes). Updates only at row wrap.
    //   km_dest_kmajor_r / km_per_lane_stride / km_elem_bytes: stable params.
    reg                        km_dest_kmajor_r;
    reg [15:0]                 km_per_lane_stride_r;
    reg [3:0]                  km_elem_bytes_r;
    // per_lane_stride / elem_bytes, held stable per transfer. elem_bytes is a
    // power of 2 (per_lane_stride = tile1 * elem_bytes), so the divide is an
    // exact right shift by log2(elem_bytes) — no runtime divider on the path.
    reg [31:0]                 km_tile1_r;
    wire [1:0] start_km_esize = (setup_params.elem_bytes == 4'd8) ? 2'd3
                              : (setup_params.elem_bytes == 4'd4) ? 2'd2
                              : (setup_params.elem_bytes == 4'd2) ? 2'd1 : 2'd0;
    reg [DXA_SMEM_ADDR_W-1:0]  km_row_base_r;

    // Tiled (Flat/BlockMajor) scatter: stable geometry + per-CL token state.
    reg [1:0]                  tiled_mode_r;
    reg [3:0]                  tiled_lg_ratio_r, tiled_lg_tcN_r, tiled_lg_nsteps_r;
    reg [3:0]                  tiled_lg_bkK_r;
    reg [3:0]                  km_esize_r;       // log2(elem_bytes)
    reg [15:0]                 n_base_r;         // N index of current CL's element 0
    reg [DXA_SMEM_ADDR_W-1:0]  tiled_base_r;     // tile SMEM byte base (stable)

    // Pass-through latched params.
    reg [31:0]                 cfill_r;

    assign out_cfill = cfill_r;
    assign out_dest_kmajor = km_dest_kmajor_r;
    assign out_per_lane_stride_bytes = km_per_lane_stride_r;
    assign out_elem_bytes = km_elem_bytes_r;
    assign out_dest_mode  = tiled_mode_r;
    assign out_lg_ratio   = tiled_lg_ratio_r;
    assign out_lg_tcN     = tiled_lg_tcN_r;
    assign out_lg_nsteps  = tiled_lg_nsteps_r;
    assign out_lg_bkK     = tiled_lg_bkK_r;
    assign out_k_row      = dim_count_r[0][15:0];
    assign out_n_base     = n_base_r;
    assign out_smem_base  = tiled_base_r;

`ifdef DXA_MBAR_ENABLE
    // ════════════════════════════════════════════════════════════════════
    // Row-major box walk (PTX tile and im2col modes). A row (one outer
    // index, an im2col pixel) is up to three segments: the dim-0 elements
    // before lo0 and from hi0 on are filled (LEAD, TRAIL), the rest read
    // (IB), or filled too when an outer coordinate is out of bounds. LEAD and
    // TRAIL split into tokens at line boundaries of their LMEM address, IB at
    // those of its global address; with a dim-0 element stride an IB token is
    // one line's elements, gathered by smem_wr.
    // setup_params stay stable until transfer_done, after the last token.
    // ════════════════════════════════════════════════════════════════════
    localparam SEG_LEAD = 2'd0, SEG_IB = 2'd1, SEG_TRAIL = 2'd2;
    localparam ND = DXA_MAX_OUTER_DIMS;
    dxa_tma_params_t tp;
    assign tp = setup_params.tma;

    reg                    tma_r;
    reg [1:0]              seg_r;
    reg                    row_oob_r;
    reg [CL_OFF_BITS-1:0]  first_off_r;
    reg [ND-1:0][DXA_X_W-1:0] xcnt_r;   // outer coordinates of the row
    reg [ND-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] acmp_r;  // im2col address components, from the first pixel's
    reg                    pf_end_r;
    reg [10:0]             pix_r;
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0] gth_addr_r;   // next gathered element
    reg [DXA_N0_W-1:0]     gth_rem_r;

    wire [1:0]  esz      = start_km_esize;
    wire [31:0] lead_b   = 32'(tp.lo0) << esz;
    wire [DXA_N0_W-1:0] ib_e = tp.hi0 - tp.lo0;
    wire [31:0] ib_b     = 32'(ib_e) << esz;
    wire [31:0] trail_b  = 32'(tp.n0 - tp.hi0) << esz;
    wire        gather   = tma_r && (tp.es0 != 4'd1);

    // A prefetch reads only: it walks no LEAD or TRAIL, and skips an
    // out-of-bounds row in one cycle; a box with no in-bounds dim-0 element ends at start.
    wire pf_walk  = tp.prefetch;
    wire pf_skip  = active_r && tma_r && pf_walk && row_oob_r;
    wire pf_empty = tp.en && pf_walk && (ib_e == 0);
    wire lead_on  = ~pf_walk && (lead_b != 0);
    wire trail_on = ~pf_walk && (trail_b != 0);

    wire [1:0] row_seg0 = lead_on ? SEG_LEAD : ((ib_e != 0) ? SEG_IB : SEG_TRAIL);

    // The segment after seg_r in its row.
    wire [1:0] seg_next = (seg_r == SEG_LEAD && ib_e != 0) ? SEG_IB : SEG_TRAIL;
    wire seg_has_next = ~pf_skip && (((seg_r == SEG_LEAD) && ((ib_e != 0) || trail_on))
                                  || ((seg_r == SEG_IB) && trail_on));

    // Gathered IB token: the line's elements from the next one, every es0.
    localparam CNT_W = CL_OFF_BITS + 1;
    function automatic [CNT_W+3:0] small_divmod(input [CNT_W-1:0] n, input [3:0] d);
        logic [4:0] rem;
        logic [CNT_W-1:0] q;
        rem = '0;
        q   = '0;
        for (int i = CNT_W-1; i >= 0; --i) begin
            rem = {rem[3:0], n[i]};
            if (rem >= 5'(d)) begin
                rem  = rem - 5'(d);
                q[i] = 1'b1;
            end
        end
        small_divmod = {q, rem[3:0]};
    endfunction
    wire [CNT_W-1:0] g_ph    = CNT_W'(gth_addr_r[CL_OFF_BITS-1:0] >> esz);
    wire [CNT_W-1:0] g_avail = CNT_W'(GMEM_LINE_SIZE >> esz) - g_ph;
    wire [CNT_W+3:0] g_qr    = small_divmod(g_avail - CNT_W'(1), tp.es0);
    wire [CNT_W-1:0] g_cnt_line = g_qr[CNT_W+3:4] + CNT_W'(1);
    wire [3:0]       g_res   = tp.es0 - 4'd1 - g_qr[3:0];
    wire             g_last  = (gth_rem_r <= DXA_N0_W'(g_cnt_line));
    wire [CNT_W-1:0] g_cnt   = g_last ? CNT_W'(gth_rem_r) : g_cnt_line;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] g_next =
        {gth_addr_r[`VX_CFG_MEM_ADDR_WIDTH-1:CL_OFF_BITS] + 1'b1, CL_OFF_BITS'(0)}
      + (`VX_CFG_MEM_ADDR_WIDTH'(g_res) << esz);
    wire g_tok = gather && (seg_r == SEG_IB);

    wire [CL_OFF_BITS-1:0] first_off = tma_r ? first_off_r : gmem_cursor_r[CL_OFF_BITS-1:0];
`else
    wire [CL_OFF_BITS-1:0] first_off = gmem_cursor_r[CL_OFF_BITS-1:0];
`endif

    // ---- Narrow token: byte_offset + valid_length ----
    // is_last_line is a decremented trip-count reaching 1 — a register-fed
    // equality, so only a small mux/subtract precedes the SMEM-step add/DSP.
    wire is_last_line = (lines_left_r == 32'd1);

    wire [CL_OFF_BITS-1:0] cur_byte_offset = is_first_r ? first_off : '0;

    wire [VLEN_W-1:0] valid_end   = is_last_line ? last_vend_r : VLEN_W'(GMEM_LINE_SIZE);
    wire [VLEN_W-1:0] valid_start = is_first_r ? {1'b0, first_off} : '0;
    wire [VLEN_W-1:0] cur_valid_length = valid_end - valid_start;

    // ---- OOB detection: any outer dim exceeds its limit ----
    wire is_oob = (dim_count_r[0] >= oob_limit_r[0])
               || (dim_count_r[1] >= oob_limit_r[1])
               || (dim_count_r[2] >= oob_limit_r[2])
               || (dim_count_r[3] >= oob_limit_r[3]);

    // ---- Last-row detection: all outer dims at their last index ----
    // An odometer index reaches its bound by equality, not magnitude: with
    // dim_count in [0, tile-1], (dim_count+1 >= tile) ⟺ (dim_count == tile-1).
    // Equality is ~2 LUT levels vs a CARRY8 chain, which keeps the once-per-row
    // step_delta / num_lines wrap path short.
    wire dim0_last = (dim_count_r[0] == dim_last_r[0]);
    wire dim1_last = (dim_count_r[1] == dim_last_r[1]);
    wire dim2_last = (dim_count_r[2] == dim_last_r[2]);
    wire dim3_last = (dim_count_r[3] == dim_last_r[3]);
    wire is_last_outer = dim0_last && dim1_last && dim2_last && dim3_last;

    // ---- Output ----
`ifdef DXA_MBAR_ENABLE
    wire tok_seg_last = pf_skip || (g_tok ? g_last : is_last_line);
    wire is_last_row  = tp.im2col ? (pix_r == (tp.ppc - 11'd1)) : is_last_outer;
    wire [VLEN_W-1:0] tok_len = g_tok ? VLEN_W'(g_cnt << esz) : cur_valid_length;
    assign out_valid          = active_r && ~pf_skip;
    assign out_pf_end         = pf_end_r;
    assign out_cl_addr        = g_tok ? gth_addr_r[`VX_CFG_MEM_ADDR_WIDTH-1:CL_OFF_BITS] : cl_addr_r;
    assign out_smem_byte_addr = smem_byte_addr_r;
    assign out_byte_offset    = g_tok ? gth_addr_r[CL_OFF_BITS-1:0] : cur_byte_offset;
    assign out_valid_length   = tok_len;
    assign out_oob            = tma_r ? ((seg_r != SEG_IB) || row_oob_r) : is_oob;
    assign out_last           = tma_r ? (tok_seg_last && ~seg_has_next && is_last_row)
                                      : (is_last_line && is_last_outer);
`else
    assign out_valid          = active_r;
    assign out_cl_addr        = cl_addr_r;
    assign out_smem_byte_addr = smem_byte_addr_r;
    assign out_byte_offset    = cur_byte_offset;
    assign out_valid_length   = cur_valid_length;
    assign out_oob            = is_oob;
    assign out_last           = is_last_line && is_last_outer;
`endif

    // ---- Advance logic ----
    wire advance = out_valid && out_ready;

    // Which delta to apply on a row-boundary step. Selects delta[0..3]
    // based on which outer dim wraps (the first not yet at its last index).
    wire dim0_steps = ~dim0_last;
    wire dim1_steps = ~dim1_last;
    wire dim2_steps = ~dim2_last;
    // dim3 must step otherwise (we already checked is_last_outer).

`ifdef DXA_MBAR_ENABLE
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] step_delta =
        dim0_steps ? delta_r[0] :
        dim1_steps ? delta_r[1] :
        dim2_steps ? delta_r[2] :
                     delta_r[3];
`else
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] step_delta =
        dim0_steps ? `VX_CFG_MEM_ADDR_WIDTH'(delta_r[0]) :
        dim1_steps ? `VX_CFG_MEM_ADDR_WIDTH'(delta_r[1]) :
        dim2_steps ? `VX_CFG_MEM_ADDR_WIDTH'(delta_r[2]) :
                     `VX_CFG_MEM_ADDR_WIDTH'(delta_r[3]);
`endif

    // K-major SMEM step = (elements in this CL) * per_lane_stride/esize.
    //   Interior CL: valid_length == LINE, so step == tile1 << log2(LINE) — a
    //                shift, no multiply.
    //   First CL:    (LINE - first_off) * tile1 — one multiply, fed directly by
    //                registers (first_off, km_tile1_r), both row-stable.
    //   Last CL:     don't-care; the row wrap overrides smem_byte_addr_r below.
    // The per-beat path is therefore a 2:1 mux of these, never the long
    // valid_length cone in front of the multiply.
    wire [VLEN_W-1:0] cl0_len      = VLEN_W'(GMEM_LINE_SIZE) - {1'b0, first_off};
    wire [31:0] km_step_full  = km_tile1_r << CL_OFF_BITS;
    wire [31:0] km_step_first = 32'(cl0_len) * km_tile1_r;
    wire [31:0] km_step       = is_first_r ? km_step_first : km_step_full;

    // Span (first_off + row_len) for the row about to be entered, and its CL
    // count. At start it comes from setup params; at a row wrap from the
    // post-step cursor (reusing the gmem_cursor_r adder).
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] next_cursor = gmem_cursor_r + step_delta;
    wire [31:0] start_span = 32'(setup_params.initial_gmem_base[CL_OFF_BITS-1:0])
                           + setup_params.row_len_bytes;
    wire [31:0] wrap_span  = 32'(next_cursor[CL_OFF_BITS-1:0]) + row_len_r;

    function automatic logic [31:0] calc_num_lines(input logic [31:0] span);
        calc_num_lines = (span + GMEM_LINE_SIZE - 1) >> CL_OFF_BITS;
    endfunction
    function automatic logic [VLEN_W-1:0] calc_last_vend(input logic [CL_OFF_BITS-1:0] off);
        calc_last_vend = (off == '0) ? VLEN_W'(GMEM_LINE_SIZE) : {1'b0, off};
    endfunction

`ifdef DXA_MBAR_ENABLE
    // ---- Row wrap: the next row's outer coordinates and cursor ----
    // Tile: the odometer's dims wrap at their last index, back to the first
    // coordinate. im2col: a pixel coordinate past xhi wraps to xlo; a carry
    // out of the spatial dims steps the batch (whose xhi never wraps).
    wire [ND-1:0] dim_wrap_t = {dim3_last, dim2_last, dim1_last, dim0_last};
    wire [ND-1:0] dim_wrap;
    wire [ND:0]   dim_carry;
    wire [ND-1:0][DXA_X_W-1:0] x_next;
    wire [ND-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] a_next;
    assign dim_carry[0] = 1'b1;
    for (genvar d = 0; d < ND; ++d) begin : g_odo
        wire [DXA_X_W-1:0] x_step = xcnt_r[d] + DXA_X_W'(tp.es[d]);
        assign dim_wrap[d] = tp.im2col ? ($signed(x_step) > $signed(tp.xhi[d])) : dim_wrap_t[d];
        assign dim_carry[d+1] = dim_carry[d] && dim_wrap[d];
        assign x_next[d] = ~dim_carry[d] ? xcnt_r[d]
                         : (dim_wrap[d] ? (tp.im2col ? tp.xlo[d] : tp.xinit[d]) : x_step);
        assign a_next[d] = ~dim_carry[d] ? acmp_r[d]
                         : (dim_wrap[d] ? tp.alo[d] : (acmp_r[d] + delta_r[d]));
    end
    `UNUSED_VAR (dim_carry[ND])
    function automatic logic oob_of(input logic [ND-1:0][DXA_X_W-1:0] x, input logic [ND-1:0][31:0] sz);
        oob_of = 1'b0;
        for (int d = 0; d < ND; ++d) begin
            if (x[d][DXA_X_W-1] || (x[d][DXA_X_W-2:0] >= (DXA_X_W-1)'(sz[d]))) oob_of = 1'b1;
        end
    endfunction
    wire oob_next = oob_of(x_next, tp.size);
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] i2c_cursor = setup_params.initial_gmem_base
        + a_next[0] + a_next[1] + a_next[2] + a_next[3];

    // ---- Segment entry: at start, at a segment switch, at a row wrap ----
    wire start_oob = oob_of(tp.xinit, tp.size);
    wire wrap_now  = tok_seg_last && ~seg_has_next;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] row_cursor = tp.im2col ? i2c_cursor : next_cursor;
    wire [DXA_SMEM_ADDR_W-1:0] smem_after = smem_byte_addr_r + DXA_SMEM_ADDR_W'(tok_len);
    wire [1:0] ent_seg = (start || wrap_now) ? row_seg0 : seg_next;
    wire [CL_OFF_BITS-1:0] ent_smem = start ? setup_params.initial_smem_base[CL_OFF_BITS-1:0]
                                            : smem_after[CL_OFF_BITS-1:0];
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] ent_cursor = start ? setup_params.initial_gmem_base
                                                 : (wrap_now ? row_cursor : gmem_cursor_r);
    wire [CL_OFF_BITS-1:0] ent_off = (ent_seg == SEG_IB) ? ent_cursor[CL_OFF_BITS-1:0] : ent_smem;
    always @(posedge clk) begin
        if (reset) begin
            pf_end_r <= 1'b0;
        end else begin
            pf_end_r <= start ? pf_empty : (pf_skip && wrap_now && is_last_row);
        end
    end
    // The writer's constants.
    `UNUSED_VAR (tp.swizzle)
    `UNUSED_VAR (tp.blk_base)
    wire [31:0] ent_len = (ent_seg == SEG_LEAD) ? lead_b : ((ent_seg == SEG_IB) ? ib_b : trail_b);
    wire [31:0] ent_span = 32'(ent_off) + ent_len;
`endif

    always @(posedge clk) begin
        if (reset) begin
            active_r         <= 1'b0;
            gmem_cursor_r    <= '0;
            cl_addr_r        <= '0;
            lines_left_r     <= '0;
            is_first_r       <= 1'b0;
            last_vend_r      <= '0;
            smem_byte_addr_r <= '0;
            // Reset K-major params so out_dest_kmajor (a mux selector
            // downstream in smem_wr) doesn't carry X before the first
            // pipeline_start. Without this, fb_word_data / fb_word_byteen
            // muxes propagate X into the bus even though transfer_active=0.
            km_dest_kmajor_r <= 1'b0;
            km_per_lane_stride_r <= '0;
            km_elem_bytes_r  <= '0;
            km_tile1_r       <= '0;
            km_row_base_r    <= '0;
            tiled_mode_r     <= DXA_DEST_ROWMAJOR;
            tiled_lg_ratio_r <= '0;
            tiled_lg_tcN_r   <= '0;
            tiled_lg_nsteps_r<= '0;
            tiled_lg_bkK_r   <= '0;
            km_esize_r       <= '0;
            n_base_r         <= '0;
            tiled_base_r     <= '0;
            for (int d = 0; d < DXA_MAX_OUTER_DIMS; d++) begin
                dim_count_r[d] <= '0;
            end
        `ifdef DXA_MBAR_ENABLE
            tma_r            <= 1'b0;
            seg_r            <= SEG_IB;
            row_oob_r        <= 1'b0;
        `endif
        end else if (start) begin
            active_r         <= 1'b1;
            gmem_cursor_r    <= setup_params.initial_gmem_base;
            row_len_r        <= setup_params.row_len_bytes;
            cl_addr_r        <= setup_params.initial_gmem_base[`VX_CFG_MEM_ADDR_WIDTH-1:CL_OFF_BITS];
            lines_left_r     <= calc_num_lines(start_span);
            is_first_r       <= 1'b1;
            last_vend_r      <= calc_last_vend(start_span[CL_OFF_BITS-1:0]);
            cfill_r          <= setup_params.cfill;
            smem_byte_addr_r <= setup_params.initial_smem_base;
            km_dest_kmajor_r <= setup_params.dest_kmajor;
            km_per_lane_stride_r <= setup_params.per_lane_stride_bytes;
            km_elem_bytes_r  <= setup_params.elem_bytes;
            km_tile1_r       <= (setup_params.elem_bytes == 4'd0) ? 32'd0
                              : (32'(setup_params.per_lane_stride_bytes) >> start_km_esize);
            km_row_base_r    <= setup_params.initial_smem_base;
            tiled_mode_r     <= setup_params.dest_mode;
            tiled_lg_ratio_r <= setup_params.lg_ratio;
            tiled_lg_tcN_r   <= setup_params.lg_tcN;
            tiled_lg_nsteps_r<= setup_params.lg_nsteps;
            tiled_lg_bkK_r   <= setup_params.lg_bkK;
            km_esize_r       <= 4'(start_km_esize);
            n_base_r         <= '0;
            tiled_base_r     <= setup_params.initial_smem_base;
            for (int d = 0; d < DXA_MAX_OUTER_DIMS; d++) begin
                dim_count_r[d] <= '0;
                dim_last_r[d]  <= setup_params.dim_tiles[d] - 32'd1;
                delta_r[d]     <= setup_params.delta[d];
                oob_limit_r[d] <= setup_params.oob_limit[d];
            end
        `ifdef DXA_MBAR_ENABLE
            tma_r    <= tp.en;
            if (pf_empty) begin
                active_r <= 1'b0;
            end
            if (tp.en) begin
                seg_r        <= ent_seg;
                row_oob_r    <= start_oob;
                first_off_r  <= ent_off;
                lines_left_r <= calc_num_lines(ent_span);
                last_vend_r  <= calc_last_vend(ent_span[CL_OFF_BITS-1:0]);
                gth_addr_r   <= ent_cursor;
                gth_rem_r    <= ib_e;
                xcnt_r       <= tp.xinit;
                acmp_r       <= '0;
                pix_r        <= '0;
            end
        `endif
`ifdef DXA_MBAR_ENABLE
        end else if (advance || pf_skip) begin
          if (tma_r) begin
            smem_byte_addr_r <= smem_after;
            if (wrap_now && is_last_row) begin
                active_r <= 1'b0;
            end else if (tok_seg_last) begin
                // Next segment of the row, or the next row's first.
                seg_r        <= ent_seg;
                first_off_r  <= ent_off;
                cl_addr_r    <= ent_cursor[`VX_CFG_MEM_ADDR_WIDTH-1:CL_OFF_BITS];
                lines_left_r <= calc_num_lines(ent_span);
                is_first_r   <= 1'b1;
                last_vend_r  <= calc_last_vend(ent_span[CL_OFF_BITS-1:0]);
                gth_addr_r   <= ent_cursor;
                gth_rem_r    <= ib_e;
                if (wrap_now) begin
                    row_oob_r     <= oob_next;
                    gmem_cursor_r <= row_cursor;
                    xcnt_r        <= x_next;
                    acmp_r        <= a_next;
                    pix_r         <= pix_r + 11'd1;
                    if (dim0_steps) begin
                        dim_count_r[0] <= dim_count_r[0] + 1;
                    end else begin
                        dim_count_r[0] <= '0;
                        if (dim1_steps) begin
                            dim_count_r[1] <= dim_count_r[1] + 1;
                        end else begin
                            dim_count_r[1] <= '0;
                            if (dim2_steps) begin
                                dim_count_r[2] <= dim_count_r[2] + 1;
                            end else begin
                                dim_count_r[2] <= '0;
                                dim_count_r[3] <= dim_count_r[3] + 1;
                            end
                        end
                    end
                end
            end else if (g_tok) begin
                gth_addr_r <= g_next;
                gth_rem_r  <= gth_rem_r - DXA_N0_W'(g_cnt_line);
            end else begin
                cl_addr_r    <= cl_addr_r + 1;
                lines_left_r <= lines_left_r - 1;
                is_first_r   <= 1'b0;
            end
          end else begin
`else
        end else if (advance) begin
`endif
            // SMEM byte address advance:
            //   Row-major: linear += valid_length.
            //   K-major:   per-CL += km_step (scatter each element to its column
            //              position). On a row wrap the address is overridden
            //              below to the next column's row base.
            if (km_dest_kmajor_r) begin
                smem_byte_addr_r <= smem_byte_addr_r + DXA_SMEM_ADDR_W'(km_step);
            end else begin
                smem_byte_addr_r <= smem_byte_addr_r + DXA_SMEM_ADDR_W'(cur_valid_length);
            end

            if (is_last_line && is_last_outer) begin
                // All done.
                active_r <= 1'b0;
            end else if (is_last_line) begin
                // Last CL of current row → advance odometer + cursor. The new
                // row's geometry is recomputed from the post-step cursor; the
                // next beat is the new row's CL0.
                gmem_cursor_r <= next_cursor;
                cl_addr_r     <= next_cursor[`VX_CFG_MEM_ADDR_WIDTH-1:CL_OFF_BITS];
                lines_left_r  <= calc_num_lines(wrap_span);
                is_first_r    <= 1'b1;
                last_vend_r   <= calc_last_vend(wrap_span[CL_OFF_BITS-1:0]);
                // K-major: SMEM jumps back to the next row base (initial +
                // (i1+1) * esize). Maintained via km_row_base_r which we bump
                // by esize at each row wrap. Row-major path keeps its linear
                // cursor — the override above is unconditional, so it would
                // step PAST the row base; we restore it here for K-major.
                if (km_dest_kmajor_r) begin
                    smem_byte_addr_r <= km_row_base_r + DXA_SMEM_ADDR_W'(km_elem_bytes_r);
                    km_row_base_r    <= km_row_base_r + DXA_SMEM_ADDR_W'(km_elem_bytes_r);
                end
                // Tiled scatter: each new row restarts at N=0 (tile-relative).
                n_base_r <= '0;
                if (dim0_steps) begin
                    dim_count_r[0] <= dim_count_r[0] + 1;
                end else begin
                    dim_count_r[0] <= '0;
                    if (dim1_steps) begin
                        dim_count_r[1] <= dim_count_r[1] + 1;
                    end else begin
                        dim_count_r[1] <= '0;
                        if (dim2_steps) begin
                            dim_count_r[2] <= dim_count_r[2] + 1;
                        end else begin
                            dim_count_r[2] <= '0;
                            dim_count_r[3] <= dim_count_r[3] + 1;
                        end
                    end
                end
            end else begin
                // Next CL of the same row.
                cl_addr_r    <= cl_addr_r + 1;
                lines_left_r <= lines_left_r - 1;
                is_first_r   <= 1'b0;
                // Tiled scatter: advance N by the elements drained from this CL.
                n_base_r     <= n_base_r + 16'(cur_valid_length >> km_esize_r);
            end
        `ifdef DXA_MBAR_ENABLE
          end
        `endif
        end
    end

    // km_step is LMEM-bounded; only the low DXA_SMEM_ADDR_W bits feed smem_byte_addr_r.
    `UNUSED_VAR (km_step[31:DXA_SMEM_ADDR_W])

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset && advance) begin
            $write("DXA_PIPE,%0d,AG_OUT,addr=0x%0h,off=%0d,len=%0d,oob=%0d,last=%0d,dim={%0d,%0d,%0d,%0d},left=%0d,smem=0x%0h\n",
            `ifdef DXA_MBAR_ENABLE
                $time, out_cl_addr, out_byte_offset, out_valid_length, out_oob, out_last,
            `else
                $time, cl_addr_r, cur_byte_offset, cur_valid_length, is_oob, out_last,
            `endif
                dim_count_r[0], dim_count_r[1], dim_count_r[2], dim_count_r[3],
                lines_left_r, smem_byte_addr_r);
        end
    end
`endif

endmodule
