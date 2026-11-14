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

// DXA Setup: owns transfer lifecycle, issue decode, metadata latching,
// and DSP-based parameter precomputation. The setup engine runs in
// parallel with an active transfer's drain: a new request is accepted
// the cycle the DSPs are idle, the result lands in `staged_*`, and is
// promoted to the live `r_*` registers the cycle the previous transfer
// completes. Setup latency overlaps with drain, eliminating the
// pipeline bubble for streamed transfers.
//
// Rank-dependent setup latency (cycles from launch_accept to staged):
//   rank 1-2: 4, rank 3: 6, rank 4: 8, rank 5: 10.
// DSP operands are registered for timing closure; this overlaps drain
// and is throughput-neutral.

`include "VX_define.vh"

module VX_dxa_setup import VX_gpu_pkg::*, VX_dxa_pkg::*; (
    input  wire                        clk,
    input  wire                        reset,

    // Launch interface (from dispatch).
    input  wire                        req_valid,
    output wire                        req_ready,
    input  wire dxa_req_data_t         req_data,
    input  wire dxa_desc_t             desc_data,

    // Transfer done (from smem_wr).
    input  wire                        transfer_done,

    // State outputs.
    output wire                        transfer_active,
    output wire                        pipeline_start,

    // Precomputed parameters (stable during ACTIVE).
    output wire dxa_setup_params_t     setup_params,

    // Latched metadata (stable during ACTIVE).
    output wire [NC_WIDTH-1:0]         active_core_id,
    output wire [UUID_WIDTH-1:0]       active_uuid,
    output wire [NW_WIDTH-1:0]         active_wid,
    output wire [BAR_ADDR_W-1:0]       active_bar_addr,
    output wire                        active_notify_smem_done,
`ifdef DXA_MBAR_ENABLE
    output wire dxa_compl_t            active_compl,
    // The transfer is a bulk copy: its lines come from the issuer's local memory.
    output wire                        active_bulk,
    // A store-family transfer, and its bulk group.
    output wire dxa_st_t               active_st,
    output wire                        active_bg,
    output wire [DXA_BG_SEQ_W-1:0]     active_bg_seq,
`endif

    // Multicast (always available; active when cta_mask has >1 bit set).
    output wire                        active_is_multicast,
    output wire [`VX_CFG_NUM_WARPS-1:0]       active_cta_mask,
    output wire [31:0]                 active_smem_stride
);
    localparam MUL_LATENCY = 2;

    // ════════════════════════════════════════════════════════════════════
    // Transfer state: TS_IDLE / TS_ACTIVE.
    // (TS_SETUP is gone — setup is tracked separately by setup_state_r.)
    // ════════════════════════════════════════════════════════════════════
    localparam TS_IDLE   = 1'b0;
    localparam TS_ACTIVE = 1'b1;

    reg state_r;
    assign transfer_active = (state_r == TS_ACTIVE);

    // ════════════════════════════════════════════════════════════════════
    // Setup engine state: SS_IDLE / SS_RUNNING / SS_STAGED.
    // ════════════════════════════════════════════════════════════════════
    localparam SS_IDLE    = 2'd0;
    localparam SS_RUNNING = 2'd1;
    localparam SS_STAGED  = 2'd2;

    reg [1:0] setup_state_r;

    // Accept a new request whenever the setup engine has capacity (i.e.,
    // not currently computing nor holding a staged result).
    assign req_ready = (setup_state_r == SS_IDLE);

    wire launch_accept = req_valid && req_ready;

    // ════════════════════════════════════════════════════════════════════
    // Issue decode (combinatorial)
    // ════════════════════════════════════════════════════════════════════

    `STATIC_ASSERT(DXA_DESC_META_TOTAL_BITS <= $bits(desc_data.meta),
        ("descriptor-meta fields overflow the 32-bit meta ABI word"))

    wire [DXA_DESC_META_ELEMSZ_BITS-1:0] esize_enc =
        desc_data.meta[DXA_DESC_META_ELEMSZ_LSB +: DXA_DESC_META_ELEMSZ_BITS];
    wire [DXA_DESC_META_DIM_BITS-1:0] rank_raw =
        desc_data.meta[DXA_DESC_META_DIM_LSB +: DXA_DESC_META_DIM_BITS];
    // LAYOUT field (meta[6:5], 2 bits): 0=row-major, 1=K-major,
    // 2=Flat (sparse), 3=BlockMajor (dense). Restricted to rank ≤ 2.
    wire [1:0] dec_dest_mode =
        desc_data.meta[DXA_DESC_META_LAYOUT_LSB +: DXA_DESC_META_LAYOUT_BITS];
    wire dec_dest_kmajor = (dec_dest_mode == DXA_DEST_KMAJOR);

    // Tiled (Flat/BlockMajor) geometry — log shift amounts (computed below,
    // after dec_tile0). tcN rides ESTRIDE2[15:0] and the dense B block K
    // extent ESTRIDE2[31:16]; ratio = 4 >> esize.
    function automatic [3:0] find_lg2(input [15:0] v);
        find_lg2 = '0;
        for (int i = 1; i < 16; i++) if (v[i]) find_lg2 = 4'(i);
    endfunction

    // Assume correct input: rank in [1,5], tiles/sizes nonzero for active dims.
    wire [31:0] dec_rank       = 32'(rank_raw);
    wire [31:0] dec_elem_bytes = 32'(1) << esize_enc;

    // Shared rank predicates — evaluated once, reused by every rank-gated mux
    // below (the 32-bit compare is otherwise re-synthesized ~13 times).
    wire rank_ge2 = (dec_rank >= 32'd2);
    wire rank_ge3 = (dec_rank >= 32'd3);
    wire rank_ge4 = (dec_rank >= 32'd4);
    wire rank_ge5 = (dec_rank >= 32'd5);

    wire [31:0] dec_tile0 = 32'(desc_data.tile01[15:0]);
    wire [31:0] dec_tile1 = rank_ge2 ? 32'(desc_data.tile01[31:16]) : 32'd1;
    wire [31:0] dec_tile2 = rank_ge3 ? 32'(desc_data.tile23[15:0])  : 32'd1;
    wire [31:0] dec_tile3 = rank_ge4 ? 32'(desc_data.tile23[31:16]) : 32'd1;
    wire [31:0] dec_tile4 = rank_ge5 ? desc_data.tile4              : 32'd1;

    wire [31:0] dec_stride0 = rank_ge2 ? desc_data.stride0 : 32'd0;
    wire [31:0] dec_stride1 = rank_ge3 ? desc_data.stride1 : 32'd0;
    wire [31:0] dec_stride2 = rank_ge4 ? desc_data.stride2 : 32'd0;
    wire [31:0] dec_stride3 = rank_ge5 ? desc_data.stride3 : 32'd0;

    wire [31:0] dec_size1 = rank_ge2 ? desc_data.size1 : 32'd1;
    wire [31:0] dec_size2 = rank_ge3 ? desc_data.size2 : 32'd1;
    wire [31:0] dec_size3 = rank_ge4 ? desc_data.size3 : 32'd1;
    wire [31:0] dec_size4 = rank_ge5 ? desc_data.size4 : 32'd1;

    // row_len_bytes = tile0 × elem_bytes and dim0 offset = coord0 × elem_bytes
    // are multiplies by a power of two (elem_bytes = 1<<esize_enc), so they are
    // variable shifts — no DSP, resolved combinationally at launch. Only the
    // coordN×strideN offsets and the wrap deltas remain genuine multiplies.
    wire [31:0] dec_row_len_bytes = dec_tile0 << esize_enc;
    wire [31:0] dec_dim0_offset   = req_data.coords[0][31:0] << esize_enc;

    // dec_elem_bytes is now only consumed as a 4-bit value (s_elem_bytes); the
    // former 32-bit multiplier use is gone (it is a shift amount now).
    `UNUSED_VAR (dec_elem_bytes[31:4])

    `RUNTIME_ASSERT(!launch_accept || (dec_rank >= 1 && dec_rank <= 5), ("invalid DXA rank: %0d", dec_rank))
    `RUNTIME_ASSERT(!launch_accept || !dec_dest_kmajor || (dec_rank <= 2),
        ("DXA K-major (LAYOUT=1) requires rank ≤ 2, got rank=%0d", dec_rank))

`ifdef DXA_MBAR_ENABLE
    // ── The row-major box walk: signed coordinates, element strides, dim-0
    // bounds, im2col, swizzle (PTX tile and im2col modes) ──
    wire dec_mbar_mode = req_data.meta[30];
    wire dec_pf     = req_data.prefetch;
    wire dec_tma    = (dec_dest_mode == DXA_DEST_ROWMAJOR);
    wire dec_im2col = dec_tma && desc_data.meta[DXA_DESC_META_IM2COL_LSB];
    wire [1:0] dec_swz = desc_data.meta[DXA_DESC_META_SWIZZLE_LSB +: DXA_DESC_META_SWIZZLE_BITS];
    wire [1:0] dec_il  = desc_data.meta[DXA_DESC_META_INTERLEAVE_LSB +: DXA_DESC_META_INTERLEAVE_BITS];

    // An element stride field of 0 reads as 1.
    function automatic [3:0] es_of(input [3:0] w);
        es_of = (w == 4'd0) ? 4'd1 : w;
    endfunction
    wire [3:0] raw_es0 = es_of(desc_data.estride0[3:0]);
    wire [3:0] raw_es1 = rank_ge2 ? es_of(desc_data.estride1[3:0]) : 4'd1;
    wire [3:0] raw_es2 = rank_ge3 ? es_of(desc_data.estride2[3:0]) : 4'd1;
    wire [3:0] raw_es3 = rank_ge4 ? es_of(desc_data.estride3[3:0]) : 4'd1;
    wire [3:0] raw_es4 = rank_ge5 ? es_of(desc_data.estride4[3:0]) : 4'd1;
    `UNUSED_VAR (desc_data.estride0[31:4])
    `UNUSED_VAR (desc_data.estride1[31:4])
    `UNUSED_VAR (desc_data.estride3[31:4])
    `UNUSED_VAR (desc_data.estride4[31:4])
    `UNUSED_VAR (req_data.tm)
    // Other layouts step by one; im2col steps the batch (the last dim) by one.
    wire [3:0] dec_es0 = (dec_tma && ~dec_im2col) ? raw_es0 : 4'd1;
    wire [3:0] dec_es1 = dec_tma ? raw_es1 : 4'd1;
    wire [3:0] dec_es2 = (dec_tma && ~(dec_im2col && (dec_rank == 32'd3))) ? raw_es2 : 4'd1;
    wire [3:0] dec_es3 = (dec_tma && ~(dec_im2col && (dec_rank == 32'd4))) ? raw_es3 : 4'd1;
    wire [3:0] dec_es4 = (dec_tma && ~dec_im2col) ? raw_es4 : 4'd1;

    // A step of es (1..8) elements of a dimension of the given byte stride.
    function automatic [35:0] es_mul(input [31:0] v, input [3:0] es);
        logic [35:0] w;
        w = 36'(v);
        es_mul = ({36{es[0]}} & w) + ({36{es[1]}} & (w << 1)) + ({36{es[2]}} & (w << 2)) + ({36{es[3]}} & (w << 3));
    endfunction

    wire [10:0] dec_ppc = desc_data.tile01[26:16];

`ifdef SIMULATION
    // Issue-time checks. Box sizes in elements: ceil(tile / es) per dim, or the im2col pixels and channels.
    function automatic [31:0] cdiv(input [31:0] v, input [3:0] es);
        cdiv = (v + 32'(es) - 32'd1) / 32'(es);
    endfunction
    wire [127:0] dec_box_elems = dec_im2col ? (128'(dec_ppc) * 128'(dec_tile0))
        : (128'(cdiv(dec_tile0, dec_es0)) * 128'(cdiv(dec_tile1, dec_es1)) * 128'(cdiv(dec_tile2, dec_es2))
         * 128'(cdiv(dec_tile3, dec_es3)) * 128'(cdiv(dec_tile4, dec_es4)));
    // The swizzle applies to the destination's block-relative offset.
    wire [DXA_SMEM_ADDR_W-1:0] dec_blk_off = req_data.smem_addr - req_data.blk_base;
    wire [31:0] dec_inner = dec_im2col ? (dec_tile0 << esize_enc)
                          : (dec_il == 2'd1) ? 32'd16 : (dec_il == 2'd2) ? 32'd32
                          : (cdiv(dec_tile0, dec_es0) << esize_enc);
    wire [31:0] dec_span = 32'd16 << dec_swz;
    // Flat/BlockMajor read ESTRIDE2 as their geometry and step by one.
    wire dec_tiled_l = (dec_dest_mode == DXA_DEST_FLAT) || (dec_dest_mode == DXA_DEST_BLOCKMAJOR);
    wire dec_es_off  = ~dec_tiled_l && ((raw_es0 != 4'd1) || (raw_es1 != 4'd1) || (raw_es2 != 4'd1)
                                     || (raw_es3 != 4'd1) || (raw_es4 != 4'd1));
    wire raw_im2col  = desc_data.meta[DXA_DESC_META_IM2COL_LSB];
    // A tensor store or reduction: its reduction type has the descriptor's
    // element size, its layout is row-major (tile or im2col), then the load's
    // rules, then a tile box starts inside the tensor.
    wire st_tensor = req_data.st.en && ((req_data.st.kind == DXA_ST_TENSOR) || (req_data.st.kind == DXA_ST_RED_T));
    wire a_st_size = st_tensor && (req_data.st.kind == DXA_ST_RED_T) && (req_data.st.tsize != esize_enc);
    wire a_st_layout = st_tensor && ~a_st_size && ~dec_tma;
    // The first rule a request breaks, in the golden model's order (dxa_ref.h check_tensor).
    wire a_i2c    = ~a_st_layout && raw_im2col && ~rank_ge3;
    wire a_layout = ~a_st_layout && ~a_i2c && ~dec_tma && ((dec_swz != 2'd0) || dec_es_off);
    wire a_ok1    = ~a_st_layout && ~a_i2c && ~a_layout;
    wire a_ilinv  = a_ok1 && (dec_il == 2'd3);
    wire a_ilrank = a_ok1 && ~a_ilinv && (dec_il != 2'd0) && ~rank_ge3;
    wire a_il32   = a_ok1 && ~a_ilinv && ~a_ilrank && (dec_il == 2'd2) && (dec_swz != 2'd1);
    wire a_ok2    = a_ok1 && ~a_ilinv && ~a_ilrank && ~a_il32;
    wire a_es0    = a_ok2 && ~dec_tiled_l && (raw_es0 != 4'd1) && ((dec_il == 2'd0) || raw_im2col);
    wire a_span   = a_ok2 && ~a_es0 && (dec_swz != 2'd0) && (dec_inner > dec_span);
    wire a_align  = a_ok2 && ~a_es0 && ~a_span && (dec_swz != 2'd0) && ~dec_pf && (dec_blk_off[6:0] != 7'd0);

    // The bytes from the destination to the end of the box's footprint: the
    // swizzle moves chunks of the last, partial 128-byte row up to its end.
    function automatic [127:0] swz_extent(input [127:0] rel, input [127:0] nb, input [1:0] swz);
        logic [127:0] fin, row, cs, ce, hi;
        logic [2:0]   m;
        fin = rel + nb;
        hi  = fin;
        if (swz != 2'd0 && nb != '0) begin
            row = (fin - 128'd1) & ~128'd127;
            m   = 3'(row >> 7) & ((swz == 2'd1) ? 3'd1 : (swz == 2'd2) ? 3'd3 : 3'd7);
            for (int c = 0; c < 8; ++c) begin
                cs = row + 128'(c * 16);
                ce = cs + 128'd16;
                if (ce > rel && cs < fin) begin
                    hi = `MAX(hi, row + 128'(32'(3'(c) ^ m) * 16) + (((ce < fin) ? ce : fin) - cs));
                end
            end
        end
        swz_extent = hi - rel;
    endfunction
    wire [127:0] dec_fp_bytes = swz_extent(128'(dec_blk_off), dec_box_elems << esize_enc, dec_swz);
    // A tensor box lands in one block: the issuer's, a window's, or each multicast receiver's.
    wire dec_blk_chk = req_data.win || req_data.mc_cluster || (~req_data.bulk && ~req_data.bload && ~dec_pf);

    // The tx count is the full box, out-of-bounds fill included.
    `RUNTIME_ASSERT(!launch_accept || !dec_mbar_mode
                 || (dec_im2col ? (dec_box_elems != 0)
                     : ((128'(dec_tile0) * 128'(dec_tile1) * 128'(dec_tile2) * 128'(dec_tile3) * 128'(dec_tile4)) != 0)),
        ("invalid DXA rank: empty box, tiles=%0d,%0d,%0d,%0d,%0d", dec_tile0, dec_tile1, dec_tile2, dec_tile3, dec_tile4))
    `RUNTIME_ASSERT(!launch_accept || !dec_mbar_mode || dec_pf
                 || (dec_fp_bytes <= 128'(1 << `VX_CFG_LMEM_LOG_SIZE)),
        ("DXA box exceeds local memory: tiles=%0d,%0d,%0d,%0d,%0d elem_bytes=%0d", dec_tile0, dec_tile1, dec_tile2,
         dec_tile3, dec_tile4, dec_elem_bytes))
    `RUNTIME_ASSERT(!launch_accept || !dec_blk_chk || (dec_fp_bytes <= 128'(req_data.win_room)),
        ("DXA box exceeds the destination block: tiles=%0d,%0d,%0d,%0d,%0d elem_bytes=%0d bytes=%0d room=%0d",
         dec_tile0, dec_tile1, dec_tile2, dec_tile3, dec_tile4, dec_elem_bytes, dec_fp_bytes, req_data.win_room))
    `RUNTIME_ASSERT(!launch_accept || !a_st_layout,
        ("%t: *** DXA store needs a tile or im2col layout: layout=%0d", $time, dec_dest_mode))
    `RUNTIME_ASSERT(!launch_accept || !a_i2c,
        ("DXA im2col needs rank 3 to 5: rank=%0d", dec_rank))
    `RUNTIME_ASSERT(!launch_accept || !a_layout,
        ("DXA swizzle and element strides need the row-major layout: layout=%0d swizzle=%0d", dec_dest_mode, dec_swz))
    `RUNTIME_ASSERT(!launch_accept || !a_ilinv,
        ("DXA interleave mode 3 is invalid: meta=0x%0h", desc_data.meta))
    `RUNTIME_ASSERT(!launch_accept || !a_ilrank,
        ("DXA interleave needs rank 3 to 5: rank=%0d", dec_rank))
    `RUNTIME_ASSERT(!launch_accept || !a_il32,
        ("DXA 32B interleave needs 32B swizzle: swizzle=%0d", dec_swz))
    `RUNTIME_ASSERT(!launch_accept || !a_es0,
        ("DXA element stride on dimension 0 needs interleave: estride0=%0d", raw_es0))
    `RUNTIME_ASSERT(!launch_accept || !a_span,
        ("DXA swizzle span smaller than the inner box: inner=%0d span=%0d", dec_inner, dec_span))
    `RUNTIME_ASSERT(!launch_accept || !a_align,
        ("DXA swizzle needs a 128-byte aligned destination: offset=0x%0h", dec_blk_off))
`ifndef VX_CFG_TCU_H100_ENABLE
    `RUNTIME_ASSERT(!launch_accept || (dec_swz == 2'd0) || (DXA_LMEM_WORD_SIZE <= 128),
        ("DXA swizzle needs a local-memory word of at most 128 bytes: word=%0d", DXA_LMEM_WORD_SIZE))
`endif
    wire [4:0] c_neg = {req_data.coords[4][31], req_data.coords[3][31], req_data.coords[2][31],
                        req_data.coords[1][31], req_data.coords[0][31]};
    wire [4:0] c_in_rank = {rank_ge5, rank_ge4, rank_ge3, rank_ge2, 1'b1};
    wire a_st_neg  = st_tensor && ~a_st_size && ~a_st_layout && ~dec_im2col && ((c_neg & c_in_rank) != '0);
    `RUNTIME_ASSERT(!launch_accept || !a_st_neg,
        ("DXA tensor store starts below the tensor: c=%0d,%0d,%0d,%0d,%0d", $signed(req_data.coords[0]),
         $signed(req_data.coords[1]), $signed(req_data.coords[2]), $signed(req_data.coords[3]), $signed(req_data.coords[4])))
    `RUNTIME_ASSERT(!launch_accept || !a_st_size,
        ("DXA reduction type does not match the element size: type bytes=%0d elem_bytes=%0d",
         (1 << req_data.st.tsize), dec_elem_bytes))
`else
    `UNUSED_VAR (dec_il)
`endif

    // Signed outer coordinates (im2col: plus their offsets), and the box corners.
    wire [DXA_MAX_OUTER_DIMS-1:0][31:0] dec_c, dec_o, dec_size, dec_lower, dec_upper;
    wire [DXA_MAX_OUTER_DIMS-1:0][DXA_X_W-1:0] dec_x;
    assign dec_c[0] = rank_ge2 ? req_data.coords[1] : 32'd0;
    assign dec_c[1] = rank_ge3 ? req_data.coords[2] : 32'd0;
    assign dec_c[2] = rank_ge4 ? req_data.coords[3] : 32'd0;
    assign dec_c[3] = rank_ge5 ? req_data.coords[4] : 32'd0;
    assign dec_size[0] = dec_size1;
    assign dec_size[1] = dec_size2;
    assign dec_size[2] = dec_size3;
    assign dec_size[3] = dec_size4;
    // Spatial dims are 1 .. rank-2.
    for (genvar d = 0; d < DXA_MAX_OUTER_DIMS; ++d) begin : g_i2c_dims
        if (d < 3) begin : g_off
            assign dec_o[d] = (dec_im2col && (32'(d + 2) < dec_rank)) ? 32'(req_data.offs[d]) : 32'd0;
        end else begin : g_off0
            assign dec_o[d] = 32'd0;
        end
        assign dec_x[d] = DXA_X_W'($signed(dec_c[d])) + DXA_X_W'(dec_o[d]);
    end
    wire [31:0] t23 = desc_data.tile23;
    assign dec_lower[0] = (dec_rank == 32'd3) ? {{16{t23[15]}}, t23[15:0]}
                        : (dec_rank == 32'd4) ? {{24{t23[7]}}, t23[7:0]} : {{27{t23[4]}}, t23[4:0]};
    assign dec_upper[0] = (dec_rank == 32'd3) ? {{16{t23[31]}}, t23[31:16]}
                        : (dec_rank == 32'd4) ? {{24{t23[23]}}, t23[23:16]} : {{27{t23[19]}}, t23[19:15]};
    assign dec_lower[1] = (dec_rank == 32'd4) ? {{24{t23[15]}}, t23[15:8]} : {{27{t23[9]}}, t23[9:5]};
    assign dec_upper[1] = (dec_rank == 32'd4) ? {{24{t23[31]}}, t23[31:24]} : {{27{t23[24]}}, t23[24:20]};
    assign dec_lower[2] = {{27{t23[14]}}, t23[14:10]};
    assign dec_upper[2] = {{27{t23[29]}}, t23[29:25]};
    assign dec_lower[3] = 32'd0;
    assign dec_upper[3] = 32'd0;

    // Dim-0 elements [vlo, vhi) of tile0 are in bounds at element stride 1;
    // the dividers below scale them to a stride.
    wire [32:0] c0_s   = {req_data.coords[0][31], req_data.coords[0]};
    wire        c0_neg = req_data.coords[0][31];
    wire [32:0] t0_w   = 33'(dec_tile0[DXA_N0_W-1:0]);
    wire [32:0] neg_c0 = 33'd0 - c0_s;
    // size0 - c0 spans (-2^31, 2^32 + 2^31).
    wire [33:0] hi_raw = {2'b00, desc_data.size0} - {c0_s[32], c0_s};
    wire [DXA_N0_W-1:0] dec_vlo = ~c0_neg ? '0 : ((neg_c0 >= t0_w) ? dec_tile0[DXA_N0_W-1:0] : DXA_N0_W'(neg_c0));
    wire [DXA_N0_W-1:0] dec_vhi = hi_raw[33] ? '0 : ((hi_raw >= {1'b0, t0_w}) ? dec_tile0[DXA_N0_W-1:0] : DXA_N0_W'(hi_raw));
    // The first in-bounds element (stride 1); with a stride, phase 2 adds lo0 x es0.
    wire [31:0] dec_c0_ib = ((dec_es0 == 4'd1) && c0_neg) ? 32'd0 : req_data.coords[0];
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] dec_dim0_soff =
        `VX_CFG_MEM_ADDR_WIDTH'($signed(dec_c0_ib)) << esize_enc;
`endif

    // K-major per-lane SMEM stride = tile1 * elem_bytes. Since elem_bytes is
    // always a power of 2 (encoded as esize_enc), this collapses to a shift.
    // tile1 ≤ 32 and elem_bytes ≤ 8, so 16 bits is sufficient.
    wire [31:0] dec_per_lane_stride_bytes_full = dec_tile1 << esize_enc;
    wire [15:0] dec_per_lane_stride_bytes = dec_per_lane_stride_bytes_full[15:0];
    `UNUSED_VAR (dec_per_lane_stride_bytes_full[31:16])

    // Tiled-scatter log shift amounts: lg_tcN = log2(tcN), lg_nsteps =
    // log2(tile0/tcN) = log2(tile0) - log2(tcN), lg_ratio = 2 - esize.
    wire [3:0] dec_lg_tcN    = find_lg2(desc_data.estride2[15:0]);
    wire [3:0] dec_lg_bkK    = find_lg2(desc_data.estride2[31:16]);
    wire [3:0] dec_lg_tile0  = find_lg2(dec_tile0[15:0]);
    wire [3:0] dec_lg_nsteps = dec_lg_tile0 - dec_lg_tcN;
    wire [3:0] dec_lg_ratio  = (esize_enc <= 2'd2) ? (4'd2 - 4'(esize_enc)) : 4'd0;

    // ════════════════════════════════════════════════════════════════════
    // Bar address decode (from req_data.meta)
    // ════════════════════════════════════════════════════════════════════

    wire [BAR_ADDR_W-1:0] launch_bar_addr;
    if (`VX_CFG_NUM_WARPS > 1) begin : g_bar_w
        assign launch_bar_addr = {req_data.meta[4 +: NW_BITS], req_data.meta[(4 + BAR_ID_SHIFT) +: NB_BITS]};
    end else begin : g_bar_wo
        assign launch_bar_addr = req_data.meta[(4 + BAR_ID_SHIFT) +: NB_BITS];
    end

    // ════════════════════════════════════════════════════════════════════
    // Active result registers (used live by downstream).
    // ════════════════════════════════════════════════════════════════════

    reg [NC_WIDTH-1:0]                     r_core_id;
    reg [UUID_WIDTH-1:0]                   r_uuid;
    reg [NW_WIDTH-1:0]                     r_wid;
    reg [BAR_ADDR_W-1:0]                   r_bar_addr;
    reg                                    r_notify_smem_done;
    reg                                    r_is_multicast;
    reg [`VX_CFG_NUM_WARPS-1:0]            r_cta_mask;
    reg [31:0]                             r_smem_stride;
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0]       r_initial_gmem_base;
    reg [DXA_SMEM_ADDR_W-1:0]              r_initial_smem_base;
    reg [31:0]                             r_row_len_bytes;
`ifdef DXA_MBAR_ENABLE
    reg [DXA_MAX_OUTER_DIMS-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] r_delta;
`else
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     r_delta;
`endif
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     r_dim_tiles;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     r_oob_limit;
    reg [31:0]                             r_cfill;
    reg                                    r_dest_kmajor;
    reg [15:0]                             r_per_lane_stride_bytes;
    reg [3:0]                              r_elem_bytes;
    reg [1:0]                              r_dest_mode;
    reg [3:0]                              r_lg_ratio;
    reg [3:0]                              r_lg_tcN;
    reg [3:0]                              r_lg_nsteps;
    reg [3:0]                              r_lg_bkK;
`ifdef DXA_MBAR_ENABLE
    dxa_compl_t                            r_compl;
    reg                                    r_bulk;
    dxa_tma_params_t                       r_tma;
    dxa_st_t                               r_st;
    reg                                    r_bg;
    reg [DXA_BG_SEQ_W-1:0]                 r_bg_seq;
`endif

    // ════════════════════════════════════════════════════════════════════
    // Staged result registers (filled by the setup engine for the next
    // transfer while the current one is still draining).
    // ════════════════════════════════════════════════════════════════════

    reg [NC_WIDTH-1:0]                     s_core_id;
    reg [UUID_WIDTH-1:0]                   s_uuid;
    reg [NW_WIDTH-1:0]                     s_wid;
    reg [BAR_ADDR_W-1:0]                   s_bar_addr;
    reg                                    s_notify_smem_done;
    reg                                    s_is_multicast;
    reg [`VX_CFG_NUM_WARPS-1:0]            s_cta_mask;
    reg [31:0]                             s_smem_stride;
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0]       s_initial_gmem_base;
    reg [DXA_SMEM_ADDR_W-1:0]              s_initial_smem_base;
    reg [31:0]                             s_row_len_bytes;
`ifdef DXA_MBAR_ENABLE
    reg [DXA_MAX_OUTER_DIMS-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] s_delta;
`else
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     s_delta;
`endif
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     s_dim_tiles;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     s_oob_limit;
    reg [31:0]                             s_cfill;
    reg                                    s_dest_kmajor;
    reg [15:0]                             s_per_lane_stride_bytes;
    reg [3:0]                              s_elem_bytes;
    reg [1:0]                              s_dest_mode;
    reg [3:0]                              s_lg_ratio;
    reg [3:0]                              s_lg_tcN;
    reg [3:0]                              s_lg_nsteps;
    reg [3:0]                              s_lg_bkK;
`ifdef DXA_MBAR_ENABLE
    dxa_compl_t                            s_compl;
    reg                                    s_bulk;
    dxa_tma_params_t                       s_tma;
    dxa_st_t                               s_st;
    reg                                    s_bg;
    reg [DXA_BG_SEQ_W-1:0]                 s_bg_seq;
`endif

    assign active_core_id          = r_core_id;
    assign active_uuid             = r_uuid;
    assign active_wid              = r_wid;
    assign active_bar_addr         = r_bar_addr;
    assign active_notify_smem_done = r_notify_smem_done;
    assign active_is_multicast     = r_is_multicast;
    assign active_cta_mask         = r_cta_mask;
    assign active_smem_stride      = r_smem_stride;
`ifdef DXA_MBAR_ENABLE
    assign active_compl            = r_compl;
    assign active_bulk             = r_bulk;
    assign active_st               = r_st;
    assign active_bg               = r_bg;
    assign active_bg_seq           = r_bg_seq;
`endif

    // ════════════════════════════════════════════════════════════════════
    // Setup pipeline: 2 parallel DSP multipliers, multi-phase.
    // ════════════════════════════════════════════════════════════════════
    //
    // row_len_bytes (tile0×elem) and dim0 offset (coord0×elem) are power-of-2
    // shifts resolved at launch — no DSP. The remaining genuine products:
    //
    // Phase 0 (ctr_r=0, always):  mul_off = coord1×stride0  (→ dim1 offset)
    // Phase 1 (ctr_r=2, rank≥3):  mul_off = coord2×stride1  (→ dim2 offset)
    //                             mul_dlt = (tile1-1)×stride0 (→ delta[1])
    // Phase 2 (ctr_r=4, rank≥4):  mul_off = coord3×stride2  (→ dim3 offset)
    //                             mul_dlt = (tile2-1)×stride1 (→ delta[2])
    // Phase 3 (ctr_r=6, rank≥5):  mul_off = coord4×stride3  (→ dim4 offset)
    //                             mul_dlt = (tile3-1)×stride2 (→ delta[3])
    //
    // Captures (posedge): ctr_r ∈ {3, 5, 7, 9} (operand reg adds +1 cycle).

    reg [3:0] ctr_r;

    // Operand latches — set at launch_accept, consumed across phases.
    // 'lat_rank' is used to gate which phases run.
    reg [31:0] lat_rank;
`ifdef DXA_MBAR_ENABLE
    reg [31:0] lat_stride0;
`else
    reg [31:0] lat_coord1, lat_stride0;
    reg [31:0] lat_coord2, lat_coord3, lat_coord4;
`endif
    reg [31:0] lat_stride1, lat_stride2, lat_stride3;

`ifdef DXA_MBAR_ENABLE
    // Signed coordinate x unsigned stride products at the address width: the
    // coordinates are DXA_X_W-bit signed, a step es x stride is 35 bits.
    localparam AW = `VX_CFG_MEM_ADDR_WIDTH;
    wire [AW-1:0]      mul_off_result, mul_dlt_result;
    reg  [DXA_X_W-1:0] mul_off_a, mul_dlt_a, mul_off_a_r, mul_dlt_a_r;
    reg  [32:0]        mul_off_b, mul_off_b_r;
    reg  [35:0]        mul_dlt_b, mul_dlt_b_r;
    always @(posedge clk) begin
        mul_off_a_r <= mul_off_a; mul_off_b_r <= mul_off_b;
        mul_dlt_a_r <= mul_dlt_a; mul_dlt_b_r <= mul_dlt_b;
    end

    VX_multiplier #(
        .A_WIDTH (DXA_X_W),
        .B_WIDTH (33),
        .R_WIDTH (AW),
        .SIGNED  (1),
        .LATENCY (MUL_LATENCY)
    ) mul_off (
        .clk    (clk),
        .enable (1'b1),
        .dataa  (mul_off_a_r),
        .datab  (mul_off_b_r),
        .result (mul_off_result)
    );

    VX_multiplier #(
        .A_WIDTH (DXA_X_W),
        .B_WIDTH (36),
        .R_WIDTH (AW),
        .SIGNED  (1),
        .LATENCY (MUL_LATENCY)
    ) mul_dlt (
        .clk    (clk),
        .enable (1'b1),
        .dataa  (mul_dlt_a_r),
        .datab  (mul_dlt_b_r),
        .result (mul_dlt_result)
    );
`else
    // Two genuine 32×32 multipliers (DSP). mul_off computes the dim offsets
    // coordN×stride(N-1); mul_dlt computes the wrap deltas (tileN-1)×stride(N-1).
    // The phase-0 ×elem_bytes products are launch-time shifts (see
    // dec_row_len_bytes / dec_dim0_offset), so no third multiplier is needed.
    wire [31:0] mul_off_result, mul_dlt_result;
    reg  [31:0] mul_off_a, mul_off_b, mul_dlt_a, mul_dlt_b;
    // Registered operands: gives each DSP an input-register stage so the
    // operand-mux + (tile-1) subtract is not in the same combinational cone
    // as the multiply. Adds +1 cycle of setup latency (phase captures shift
    // by one below). Setup overlaps drain, so the extra cycle is immaterial.
    reg  [31:0] mul_off_a_r, mul_off_b_r, mul_dlt_a_r, mul_dlt_b_r;
    always @(posedge clk) begin
        mul_off_a_r <= mul_off_a; mul_off_b_r <= mul_off_b;
        mul_dlt_a_r <= mul_dlt_a; mul_dlt_b_r <= mul_dlt_b;
    end

    VX_multiplier #(
        .A_WIDTH (32),
        .B_WIDTH (32),
        .R_WIDTH (32),
        .LATENCY (MUL_LATENCY)
    ) mul_off (
        .clk    (clk),
        .enable (1'b1),
        .dataa  (mul_off_a_r),
        .datab  (mul_off_b_r),
        .result (mul_off_result)
    );

    VX_multiplier #(
        .A_WIDTH (32),
        .B_WIDTH (32),
        .R_WIDTH (32),
        .LATENCY (MUL_LATENCY)
    ) mul_dlt (
        .clk    (clk),
        .enable (1'b1),
        .dataa  (mul_dlt_a_r),
        .datab  (mul_dlt_b_r),
        .result (mul_dlt_result)
    );

`endif

    // Latched dim tiles (for wrap-delta computation).
    reg [31:0] lat_tile1, lat_tile2, lat_tile3;
`ifdef DXA_MBAR_ENABLE
    // Box bytes = (tile0 x elem_bytes) x tile1 x .. x tile4, one product per
    // cycle in phase-0 cycles ctr 0..3, which every rank runs; the box fits the
    // local memory, so DXA_TX_BYTES_W bits keep the exact count.
    reg  [DXA_TX_BYTES_W-1:0] lat_tile4;
    // The box's element counts come from one small divider, one per cycle:
    //   ctr 0: n1, 1: n2, 2: n0, 3: lo0, 4: hi0, 5: n3, 6: n4,
    // each in time for its use, so every rank keeps its latency. The tx
    // product is elem_bytes x n1 (ctr 1) x n2 (2) x n0 (3) x n3 (6) x n4 (7),
    // or x pixels (1) x channels (3) in im2col.
    reg  [3:0]  lat_es0, lat_es1, lat_es2, lat_es3, lat_es4;
    reg  [DXA_N0_W-1:0] lat_tile0, lat_vlo, lat_vhi;
    reg  [31:0] lat_n1, lat_n2, lat_n3;
    reg  [35:0] lat_S1, lat_S2, lat_S3, lat_S4;     // byte steps es x stride
    reg  [DXA_X_W-1:0] lat_x1, lat_x2, lat_x3, lat_x4;  // outer coordinates
    reg  [DXA_X_W-1:0] lat_xlo1, lat_xlo2, lat_xlo3;    // im2col wrap coordinates
    reg         lat_im2col;
    wire [19:0] div_v = (ctr_r == 4'd0) ? 20'(lat_tile1)
                      : (ctr_r == 4'd1) ? 20'(lat_tile2)
                      : (ctr_r == 4'd2) ? 20'(lat_tile0)
                      : (ctr_r == 4'd3) ? 20'(lat_vlo)
                      : (ctr_r == 4'd4) ? 20'(lat_vhi)
                      : (ctr_r == 4'd5) ? 20'(lat_tile3)
                      :                   20'(lat_tile4);
    wire [3:0]  div_es = (ctr_r == 4'd0) ? lat_es1
                       : (ctr_r == 4'd1) ? lat_es2
                       : (ctr_r <= 4'd4) ? lat_es0
                       : (ctr_r == 4'd5) ? lat_es3
                       :                   lat_es4;
    // ceil(v / es) for es in 1..8: restoring division, compares and subtracts only.
    function automatic [19:0] ceil_div8(input [19:0] v, input [3:0] es);
        logic [19:0] num;
        logic [4:0]  rem;
        logic [19:0] q;
        num = v + 20'(es) - 20'd1;
        rem = '0;
        q   = '0;
        for (int i = 19; i >= 0; --i) begin
            rem = {rem[3:0], num[i]};
            if (rem >= 5'(es)) begin
                rem  = rem - 5'(es);
                q[i] = 1'b1;
            end
        end
        ceil_div8 = q;
    endfunction
    `UNUSED_VAR (lat_tile1[31:20])
    `UNUSED_VAR (lat_tile2[31:20])
    `UNUSED_VAR (lat_tile3[31:20])
    wire [19:0] div_q = ceil_div8(div_v, div_es);
    wire [DXA_TX_BYTES_W-1:0] tx_mul = lat_im2col
        ? ((ctr_r == 4'd1) ? DXA_TX_BYTES_W'(s_tma.ppc) : (ctr_r == 4'd3) ? DXA_TX_BYTES_W'(s_tma.n0) : DXA_TX_BYTES_W'(1))
        : ((ctr_r == 4'd1) ? DXA_TX_BYTES_W'(lat_n1)
         : (ctr_r == 4'd2) ? DXA_TX_BYTES_W'(lat_n2)
         : (ctr_r == 4'd3) ? DXA_TX_BYTES_W'(s_tma.n0)
         : (ctr_r == 4'd6) ? DXA_TX_BYTES_W'(lat_n3)
         : (ctr_r == 4'd7) ? DXA_TX_BYTES_W'(s_dim_tiles[3])
         :                   DXA_TX_BYTES_W'(1));
    wire [DXA_TX_BYTES_W-1:0] tx_prod = s_compl.tx_bytes * tx_mul;
    // The first in-bounds dim-0 element of a strided row is lo0 x es0 past c0.
    wire [35:0] lo0_es = es_mul(32'(s_tma.lo0), lat_es0);
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] lo0_off = `VX_CFG_MEM_ADDR_WIDTH'(lo0_es) << lat_esz;
    reg  [1:0]  lat_esz;
`endif

    // Multiplier input mux (driven by ctr_r and lat_rank).
    //   mul_off: dim offset coordN×stride(N-1), one per phase.
    //   mul_dlt: wrap delta (tileN-1)×stride(N-1), phases 1-3 only.
`ifdef DXA_MBAR_ENABLE
    // With element strides the wrap delta is (nN-1) x esN x stride(N-1); in
    // im2col it is the pixel coordinate's wrap, (xloN - xN) x stride(N-1).
    always @(*) begin
        mul_off_a = '0; mul_off_b = '0;
        mul_dlt_a = '0; mul_dlt_b = '0;

        case (ctr_r)
        4'd0: begin
            mul_off_a = lat_x1; mul_off_b = 33'(lat_stride0);
        end
        4'd2: begin
            if (lat_rank >= 3) begin
                mul_off_a = lat_x2; mul_off_b = 33'(lat_stride1);
                mul_dlt_a = lat_im2col ? (lat_xlo1 - lat_x1) : DXA_X_W'(lat_n1 - 32'd1);
                mul_dlt_b = lat_im2col ? 36'(lat_stride0) : lat_S1;
            end
        end
        4'd4: begin
            if (lat_rank >= 4) begin
                mul_off_a = lat_x3; mul_off_b = 33'(lat_stride2);
                mul_dlt_a = lat_im2col ? (lat_xlo2 - lat_x2) : DXA_X_W'(lat_n2 - 32'd1);
                mul_dlt_b = lat_im2col ? 36'(lat_stride1) : lat_S2;
            end
        end
        4'd6: begin
            if (lat_rank >= 5) begin
                mul_off_a = lat_x4; mul_off_b = 33'(lat_stride3);
                mul_dlt_a = lat_im2col ? (lat_xlo3 - lat_x3) : DXA_X_W'(lat_n3 - 32'd1);
                mul_dlt_b = lat_im2col ? 36'(lat_stride2) : lat_S3;
            end
        end
        default: ;
        endcase
    end
`else
    always @(*) begin
        mul_off_a = '0; mul_off_b = '0;
        mul_dlt_a = '0; mul_dlt_b = '0;

        case (ctr_r)
        4'd0: begin
            // Phase 0: dim1 offset only (row_len + dim0 offset are shifts).
            mul_off_a = lat_coord1; mul_off_b = lat_stride0;
        end
        4'd2: begin
            // Phase 1: rank ≥ 3.
            if (lat_rank >= 3) begin
                mul_off_a = lat_coord2;        mul_off_b = lat_stride1;
                mul_dlt_a = lat_tile1 - 32'd1; mul_dlt_b = lat_stride0;  // (tile1-1)*stride0
            end
        end
        4'd4: begin
            // Phase 2: rank ≥ 4.
            if (lat_rank >= 4) begin
                mul_off_a = lat_coord3;        mul_off_b = lat_stride2;
                mul_dlt_a = lat_tile2 - 32'd1; mul_dlt_b = lat_stride1;  // (tile2-1)*stride1
            end
        end
        4'd6: begin
            // Phase 3: rank ≥ 5.
            if (lat_rank >= 5) begin
                mul_off_a = lat_coord4;        mul_off_b = lat_stride3;
                mul_dlt_a = lat_tile3 - 32'd1; mul_dlt_b = lat_stride2;  // (tile3-1)*stride2
            end
        end
        default: ;
        endcase
    end
`endif

    // Cycle at which the LAST capture fires (= setup_state goes STAGED).
    //   rank 1-2: cycle 2 (Phase 0 capture only).
    //   rank 3:   cycle 4.  rank 4: cycle 6.  rank 5: cycle 8.
    wire [3:0] done_at_ctr =
        (lat_rank <= 2) ? 4'd3 :
        (lat_rank == 3) ? 4'd5 :
        (lat_rank == 4) ? 4'd7 :
                          4'd9;

    // ════════════════════════════════════════════════════════════════════
    // Promote: copy staged → active and pulse pipeline_start.
    // Fires when a staged result exists AND the active slot is free
    // (either idle, or completing this cycle via transfer_done).
    // ════════════════════════════════════════════════════════════════════

    wire promote_now = (setup_state_r == SS_STAGED)
                    && (state_r == TS_IDLE || transfer_done);

    reg pipeline_start_r;
    assign pipeline_start = pipeline_start_r;

    // ── OOB limit helpers (subtraction, no DSP needed) ──
    function automatic [31:0] oob_sub(input [31:0] sz, input [31:0] coord);
        oob_sub = (sz > coord) ? (sz - coord) : 32'd0;
    endfunction

    // ════════════════════════════════════════════════════════════════════
    // Sequential update
    // ════════════════════════════════════════════════════════════════════
    always @(posedge clk) begin
        if (reset) begin
            state_r            <= TS_IDLE;
            setup_state_r      <= SS_IDLE;
            ctr_r              <= '0;
            pipeline_start_r   <= 1'b0;
            r_core_id          <= '0;
            r_uuid             <= '0;
            r_wid              <= '0;
            r_bar_addr         <= '0;
            r_notify_smem_done <= 1'b0;
            // dest_kmajor / elem_bytes / per_lane_stride feed mux selectors
            // in addr_gen / smem_wr. Reset them so X doesn't propagate via
            // ag_dest_kmajor before the first transfer is staged.
            r_dest_kmajor      <= 1'b0;
            r_per_lane_stride_bytes <= '0;
            r_elem_bytes       <= '0;
            r_dest_mode        <= DXA_DEST_ROWMAJOR;
            r_lg_ratio         <= '0;
            r_lg_tcN           <= '0;
            r_lg_nsteps        <= '0;
            r_lg_bkK           <= '0;
            s_dest_kmajor      <= 1'b0;
            s_per_lane_stride_bytes <= '0;
            s_elem_bytes       <= '0;
            s_dest_mode        <= DXA_DEST_ROWMAJOR;
            s_lg_ratio         <= '0;
            s_lg_tcN           <= '0;
            s_lg_nsteps        <= '0;
            s_lg_bkK           <= '0;
        `ifdef DXA_MBAR_ENABLE
            r_bulk             <= 1'b0;
            r_tma              <= '0;
            r_st               <= '0;
            r_bg               <= 1'b0;
        `endif
        end else begin
            // Default: pipeline_start is a 1-cycle pulse.
            pipeline_start_r <= 1'b0;

            // ── Transfer state ───────────────────────────────────────────
            if (promote_now) begin
                // Move staged → active and fire pipeline_start.
                r_core_id           <= s_core_id;
                r_uuid              <= s_uuid;
                r_wid               <= s_wid;
                r_bar_addr          <= s_bar_addr;
                r_notify_smem_done  <= s_notify_smem_done;
                r_is_multicast      <= s_is_multicast;
                r_cta_mask          <= s_cta_mask;
                r_smem_stride       <= s_smem_stride;
                r_initial_gmem_base <= s_initial_gmem_base;
                r_initial_smem_base <= s_initial_smem_base;
                r_row_len_bytes     <= s_row_len_bytes;
                r_delta             <= s_delta;
                r_dim_tiles         <= s_dim_tiles;
                r_oob_limit         <= s_oob_limit;
                r_cfill             <= s_cfill;
                r_dest_kmajor       <= s_dest_kmajor;
                r_per_lane_stride_bytes <= s_per_lane_stride_bytes;
                r_elem_bytes        <= s_elem_bytes;
                r_dest_mode         <= s_dest_mode;
                r_lg_ratio          <= s_lg_ratio;
                r_lg_tcN            <= s_lg_tcN;
                r_lg_nsteps         <= s_lg_nsteps;
                r_lg_bkK            <= s_lg_bkK;
            `ifdef DXA_MBAR_ENABLE
                r_compl             <= s_compl;
                r_bulk              <= s_bulk;
                r_tma               <= s_tma;
                r_st                <= s_st;
                r_bg                <= s_bg;
                r_bg_seq            <= s_bg_seq;
            `endif
                state_r             <= TS_ACTIVE;
                pipeline_start_r    <= 1'b1;
                setup_state_r       <= SS_IDLE;
            end else if (state_r == TS_ACTIVE && transfer_done) begin
                // Active drain complete, no staged result waiting.
                state_r <= TS_IDLE;
            end

            // ── Setup engine ─────────────────────────────────────────────
            case (setup_state_r)
            SS_IDLE: begin
                if (launch_accept) begin
                    // Latch metadata directly into staged_*.
                    s_core_id           <= req_data.core_id;
                    s_uuid              <= req_data.uuid;
                    s_wid               <= req_data.wid;
                    s_bar_addr          <= launch_bar_addr;
                `ifdef VX_CFG_EXT_DXA_ENABLE
                    s_notify_smem_done  <= 1'b1;
                `else
                    s_notify_smem_done  <= 1'b0;
                `endif
                `ifdef DXA_MBAR_ENABLE
                    s_is_multicast      <= req_data.mc_cluster || ($countones(req_data.cta_mask) > 1);
                `else
                    s_is_multicast      <= ($countones(req_data.cta_mask) > 1);
                `endif
                    s_cta_mask          <= req_data.cta_mask;
                    s_smem_stride       <= desc_data.smem_stride;
                    s_initial_smem_base <= req_data.smem_addr;
                    s_cfill             <= desc_data.cfill;
                    s_dest_kmajor       <= dec_dest_kmajor;
                    s_per_lane_stride_bytes <= dec_per_lane_stride_bytes;
                    s_elem_bytes        <= 4'(dec_elem_bytes);
                    s_dest_mode         <= dec_dest_mode;
                    s_lg_ratio          <= dec_lg_ratio;
                    s_lg_tcN            <= dec_lg_tcN;
                    s_lg_nsteps         <= dec_lg_nsteps;
                    s_lg_bkK            <= dec_lg_bkK;
                `ifdef DXA_MBAR_ENABLE
                    s_compl.mode        <= dec_mbar_mode;
                    s_compl.mbar_off    <= req_data.mbar_off;
                    s_compl.tx_bytes    <= DXA_TX_BYTES_W'(dec_row_len_bytes);
                    s_compl.slot        <= req_data.remote ? req_data.dst_slot : req_data.slot;
                    s_compl.gen         <= req_data.gen;
                    s_compl.remote      <= req_data.remote;
                    s_compl.dst_core    <= req_data.dst_core;
                    s_compl.cluster     <= req_data.mc_cluster;
                    s_compl.mask        <= req_data.mc_mask;
                    s_compl.peers       <= req_data.mc_peers;
                    lat_tile4           <= DXA_TX_BYTES_W'(dec_tile4);
                    s_bulk              <= req_data.bulk;
                    s_st                <= req_data.st;
                    s_bg                <= req_data.bg;
                    s_bg_seq            <= req_data.bg_seq;
                `endif
                    // Rolling-cursor deltas: delta[0] is the inner-dim step.
                    // Higher deltas are precomputed below (Phase 1-3 captures).
                `ifdef DXA_MBAR_ENABLE
                    s_delta[0]          <= AW'(dec_stride0);
                `else
                    s_delta[0]          <= dec_stride0;
                `endif
                    s_delta[1]          <= '0;
                    s_delta[2]          <= '0;
                    s_delta[3]          <= '0;
                    s_dim_tiles[0]      <= dec_tile1;
                    s_dim_tiles[1]      <= dec_tile2;
                    s_dim_tiles[2]      <= dec_tile3;
                    s_dim_tiles[3]      <= dec_tile4;
                    // OOB limits.
                    s_oob_limit[0]      <= oob_sub(dec_size1, rank_ge2 ? req_data.coords[1][31:0] : 32'd0);
                    s_oob_limit[1]      <= oob_sub(dec_size2, rank_ge3 ? req_data.coords[2][31:0] : 32'd0);
                    s_oob_limit[2]      <= oob_sub(dec_size3, rank_ge4 ? req_data.coords[3][31:0] : 32'd0);
                    s_oob_limit[3]      <= oob_sub(dec_size4, rank_ge5 ? req_data.coords[4][31:0] : 32'd0);
                    // Operand latches (DSP inputs across phases).
                    lat_rank            <= dec_rank;
                    lat_tile1           <= dec_tile1;
                    lat_tile2           <= dec_tile2;
                    lat_tile3           <= dec_tile3;
                `ifndef DXA_MBAR_ENABLE
                    lat_coord1          <= rank_ge2 ? req_data.coords[1][31:0] : 32'd0;
                    lat_coord2          <= rank_ge3 ? req_data.coords[2][31:0] : 32'd0;
                    lat_coord3          <= rank_ge4 ? req_data.coords[3][31:0] : 32'd0;
                    lat_coord4          <= rank_ge5 ? req_data.coords[4][31:0] : 32'd0;
                `endif
                    lat_stride0         <= dec_stride0;
                    lat_stride1         <= dec_stride1;
                    lat_stride2         <= dec_stride2;
                    lat_stride3         <= dec_stride3;
                    // row_len = tile0×elem and dim0 offset = coord0×elem are
                    // power-of-2 shifts, resolved here (no DSP, no phase).
                    s_row_len_bytes     <= dec_row_len_bytes;
                    // Seed staged base = desc.base + dim0 offset; phases add the
                    // remaining coordN×stride(N-1) products.
                    s_initial_gmem_base <= desc_data.base_addr
                                         + `VX_CFG_MEM_ADDR_WIDTH'(dec_dim0_offset);
                `ifdef DXA_MBAR_ENABLE
                    // The row-major walk's constants; these override the
                    // tile-mode latches above.
                    if (dec_tma) begin
                        s_initial_gmem_base <= desc_data.base_addr + dec_dim0_soff;
                    end
                    s_compl.tx_bytes    <= DXA_TX_BYTES_W'(dec_elem_bytes);
                    lat_x1              <= dec_x[0];
                    lat_x2              <= dec_x[1];
                    lat_x3              <= dec_x[2];
                    lat_x4              <= dec_x[3];
                    lat_tile0           <= dec_tile0[DXA_N0_W-1:0];
                    lat_vlo             <= dec_vlo;
                    lat_vhi             <= dec_vhi;
                    lat_es0             <= dec_es0;
                    lat_es1             <= dec_es1;
                    lat_es2             <= dec_es2;
                    lat_es3             <= dec_es3;
                    lat_es4             <= dec_es4;
                    lat_esz             <= esize_enc;
                    lat_im2col          <= dec_im2col;
                    lat_S1              <= es_mul(dec_stride0, dec_es1);
                    lat_S2              <= es_mul(dec_stride1, dec_es2);
                    lat_S3              <= es_mul(dec_stride2, dec_es3);
                    lat_S4              <= es_mul(dec_stride3, dec_es4);
                    s_delta[0]          <= AW'(es_mul(dec_stride0, dec_es1));
                    lat_xlo1            <= DXA_X_W'($signed(dec_lower[0])) + DXA_X_W'(dec_o[0]);
                    lat_xlo2            <= DXA_X_W'($signed(dec_lower[1])) + DXA_X_W'(dec_o[1]);
                    lat_xlo3            <= DXA_X_W'($signed(dec_lower[2])) + DXA_X_W'(dec_o[2]);
                    s_tma.en            <= dec_tma;
                    s_tma.im2col        <= dec_im2col;
                    s_tma.prefetch      <= dec_pf;
                    s_tma.es0           <= dec_es0;
                    s_tma.n0            <= dec_tile0[DXA_N0_W-1:0];
                    s_tma.lo0           <= dec_vlo;
                    s_tma.hi0           <= dec_vhi;
                    s_tma.ppc           <= dec_ppc;
                    s_tma.swizzle       <= (dec_tma && ~dec_pf) ? dec_swz : 2'd0;
                    s_tma.blk_base      <= req_data.blk_base;
                    for (integer d = 0; d < DXA_MAX_OUTER_DIMS; ++d) begin
                        s_tma.xinit[d]  <= dec_x[d];
                        s_tma.size[d]   <= dec_size[d];
                        s_tma.alo[d]    <= '0;
                        if (dec_im2col && (32'(d + 2) < dec_rank)) begin
                            s_tma.xlo[d] <= DXA_X_W'($signed(dec_lower[d])) + DXA_X_W'(dec_o[d]);
                            s_tma.xhi[d] <= DXA_X_W'(dec_size[d]) - DXA_X_W'(1) + DXA_X_W'($signed(dec_upper[d]))
                                          + DXA_X_W'(dec_o[d]);
                        end else begin
                            s_tma.xlo[d] <= '0;
                            s_tma.xhi[d] <= {1'b0, {(DXA_X_W-1){1'b1}}};
                        end
                    end
                    s_tma.es[0]         <= dec_es1;
                    s_tma.es[1]         <= dec_es2;
                    s_tma.es[2]         <= dec_es3;
                    s_tma.es[3]         <= dec_es4;
                    // im2col steps each pixel dim by delta[d].
                    if (dec_im2col) begin
                        s_delta[1]      <= AW'(es_mul(dec_stride1, dec_es2));
                        s_delta[2]      <= AW'(es_mul(dec_stride2, dec_es3));
                        s_delta[3]      <= AW'(es_mul(dec_stride3, dec_es4));
                    end
                `endif
                    ctr_r               <= '0;
                    setup_state_r       <= SS_RUNNING;
                end
            end
            SS_RUNNING: begin
                ctr_r <= ctr_r + 4'd1;
            `ifdef DXA_MBAR_ENABLE
                if (ctr_r <= 4'd7) begin
                    s_compl.tx_bytes <= tx_prod;
                end
            `endif

            `ifndef DXA_MBAR_ENABLE
                // ── Phase 0 capture at ctr=3 (operand reg adds +1 cycle) ──
                //   dim1 offset (coord1×stride0). row_len + dim0 offset were
                //   resolved as shifts at launch.
                if (ctr_r == 4'd3) begin
                    s_initial_gmem_base <= s_initial_gmem_base
                                         + `VX_CFG_MEM_ADDR_WIDTH'(mul_off_result);
                end

                // ── Phase 1 capture at ctr=5 (rank≥3) ──
                if (ctr_r == 4'd5 && lat_rank >= 3) begin
                    s_initial_gmem_base <= s_initial_gmem_base
                                         + `VX_CFG_MEM_ADDR_WIDTH'(mul_off_result);
                    // delta[1] = stride1 - (tile1-1)*stride0
                    s_delta[1] <= lat_stride1 - mul_dlt_result;
                end

                // ── Phase 2 capture at ctr=7 (rank≥4) ──
                if (ctr_r == 4'd7 && lat_rank >= 4) begin
                    s_initial_gmem_base <= s_initial_gmem_base
                                         + `VX_CFG_MEM_ADDR_WIDTH'(mul_off_result);
                    // delta[2] = stride2 - (tile2-1)*stride1 - (tile1-1)*stride0:
                    // dims 1 and 2 both wrap; (tile1-1)*stride0 = stride1 - delta[1].
                    s_delta[2] <= lat_stride2 - mul_dlt_result - lat_stride1 + s_delta[1];
                end

                // ── Phase 3 capture at ctr=9 (rank=5) ──
                if (ctr_r == 4'd9 && lat_rank >= 5) begin
                    s_initial_gmem_base <= s_initial_gmem_base
                                         + `VX_CFG_MEM_ADDR_WIDTH'(mul_off_result);
                    // delta[3] = stride3 - (tile3-1)*stride2 - (wraps of dims 1-2),
                    // the latter being stride2 - delta[2].
                    s_delta[3] <= lat_stride3 - mul_dlt_result - lat_stride2 + s_delta[2];
                end
            `endif

            `ifdef DXA_MBAR_ENABLE
                // The box's element counts (schedule at tx_mul above).
                case (ctr_r)
                4'd0: begin
                    lat_n1         <= 32'(div_q);
                    s_dim_tiles[0] <= 32'(div_q);
                end
                4'd1: begin
                    lat_n2         <= 32'(div_q);
                    s_dim_tiles[1] <= 32'(div_q);
                end
                4'd2: s_tma.n0  <= DXA_N0_W'(div_q);
                4'd3: s_tma.lo0 <= DXA_N0_W'(div_q);
                4'd4: begin
                    s_tma.hi0 <= DXA_N0_W'(div_q);
                    if (lat_es0 != 4'd1) begin
                        s_initial_gmem_base <= s_initial_gmem_base + lo0_off;
                    end
                end
                4'd5: begin
                    lat_n3         <= 32'(div_q);
                    s_dim_tiles[2] <= 32'(div_q);
                end
                4'd6: s_dim_tiles[3] <= 32'(div_q);
                default: ;
                endcase
                // Signed coordinate offsets and element-strided deltas, over
                // the captures above; im2col keeps its steps and takes the wraps.
                if (ctr_r == 4'd3) begin
                    s_initial_gmem_base <= s_initial_gmem_base + mul_off_result;
                end
                if (ctr_r == 4'd5 && lat_rank >= 3) begin
                    s_initial_gmem_base <= s_initial_gmem_base + mul_off_result;
                    if (lat_im2col) begin
                        s_tma.alo[0] <= mul_dlt_result;
                        s_delta[1]   <= AW'(lat_S2);
                    end else begin
                        s_delta[1]   <= AW'(lat_S2) - mul_dlt_result;
                    end
                end
                if (ctr_r == 4'd7 && lat_rank >= 4) begin
                    s_initial_gmem_base <= s_initial_gmem_base + mul_off_result;
                    if (lat_im2col) begin
                        s_tma.alo[1] <= mul_dlt_result;
                        s_delta[2]   <= AW'(lat_S3);
                    end else begin
                        s_delta[2]   <= AW'(lat_S3) - mul_dlt_result - AW'(lat_S2) + s_delta[1];
                    end
                end
                if (ctr_r == 4'd9 && lat_rank >= 5) begin
                    s_initial_gmem_base <= s_initial_gmem_base + mul_off_result;
                    if (lat_im2col) begin
                        s_tma.alo[2] <= mul_dlt_result;
                        s_delta[3]   <= AW'(lat_S4);
                    end else begin
                        s_delta[3]   <= AW'(lat_S4) - mul_dlt_result - AW'(lat_S3) + s_delta[2];
                    end
                end
            `endif

                // Transition to STAGED at the last capture cycle.
                if (ctr_r == done_at_ctr) begin
                    setup_state_r <= SS_STAGED;
                    ctr_r         <= '0;
                end
            end
            SS_STAGED: begin
                // Wait for promote_now to consume the staged result.
                // Handled in the promote block above.
            end
            default: setup_state_r <= SS_IDLE;
            endcase
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // Output assignments
    // ════════════════════════════════════════════════════════════════════
    assign setup_params.initial_gmem_base  = r_initial_gmem_base;
    assign setup_params.initial_smem_base  = r_initial_smem_base;
    assign setup_params.row_len_bytes      = r_row_len_bytes;
    assign setup_params.delta              = r_delta;
    assign setup_params.dim_tiles          = r_dim_tiles;
    assign setup_params.oob_limit          = r_oob_limit;
    assign setup_params.cfill              = r_cfill;
    assign setup_params.dest_kmajor        = r_dest_kmajor;
    assign setup_params.per_lane_stride_bytes = r_per_lane_stride_bytes;
    assign setup_params.elem_bytes         = r_elem_bytes;
    assign setup_params.dest_mode          = r_dest_mode;
    assign setup_params.lg_ratio           = r_lg_ratio;
    assign setup_params.lg_tcN             = r_lg_tcN;
    assign setup_params.lg_nsteps          = r_lg_nsteps;
    assign setup_params.lg_bkK             = r_lg_bkK;
`ifdef DXA_MBAR_ENABLE
    assign setup_params.tma                = r_tma;
`endif

`ifndef DXA_MBAR_ENABLE
    `UNUSED_VAR (desc_data.size0)
`endif
    `UNUSED_VAR (req_data.meta)
`ifdef DXA_MBAR_ENABLE
    // A bulk copy's source and size arrive in its descriptor.
    `UNUSED_VAR (req_data.src_addr)
    `UNUSED_VAR (req_data.size)
    `UNUSED_VAR (req_data.tload)
    `UNUSED_VAR (req_data.bload)
    `UNUSED_VAR (req_data.gsrc)
`endif

endmodule
