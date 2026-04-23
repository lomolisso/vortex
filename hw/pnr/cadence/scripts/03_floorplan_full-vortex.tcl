#=======================================================================
# Stage 3 — Floorplan / Macro placement / I-O pins  (config: full-vortex)
#
# The full-vortex PnR target elaborates the Vortex GPGPU with four
# VX_socket_top blackbox macros (one per core) and a shared L2 cache.
# Two macro classes need hand placement here:
#
#   · 4 × VX_socket_top    — the core blackboxes, geometry from
#                            VX_socket_top.lef (sourced via
#                            01_init_design.tcl).  Placed in the four
#                            die corners with pins facing the die centre.
#
#   · 32 L2 SRAM macros    — 16 tags (sram_256x24_1r1w) + 16 data arrays
#                            (sram_256x512_1rw).  L2_NUM_BANKS × L2_NUM_WAYS
#                            = 4 × 4 of each variant.  Placed as two
#                            horizontal rows centred in the die.
#
# LAYOUT (X grows right, Y grows up).  All L2 macros are laid out as
# 6 horizontal rows, centred on the die.  Data macros are rotated 90°
# so their long (333 µm) axis is horizontal — the user-requested
# "horizontal" orientation.  Cores sit in the four corners, inset
# CORE_EDGE_OFFSET µm from the die edge.
#
#     yhi ┌──────────────────────────────────────────────────────────┐
#         │  [CORE_TL (MX)]                   [CORE_TR (R180)]        │
#         │   ─── inset CORE_EDGE_OFFSET from every die edge ───     │
#         │                                                          │
#         │          ╔════════════════════════════════╗   row 1      │
#         │          ║ 16 × tag   (R270, pins  ↑)     ║              │
#         │          ╠════════════════════════════════╣ ← MACRO_GAP  │
#         │          ║ 8  × data  (R90,  pins  ↓)     ║   row 2      │
#         │          ╠════════════════════════════════╣ ← FACING_GAP │
#         │          ║ 8  × data  (R270, pins  ↑)     ║   row 3      │
#         │          ╠════════════════════════════════╣ ← MACRO_GAP  │
#         │          ║ 8  × data  (R90,  pins  ↓)     ║   row 4      │
#         │          ╠════════════════════════════════╣ ← FACING_GAP │
#         │          ║ 8  × data  (R270, pins  ↑)     ║   row 5      │
#         │          ╠════════════════════════════════╣ ← MACRO_GAP  │
#         │          ║ 16 × tag   (R90,  pins  ↓)     ║   row 6      │
#         │          ╚════════════════════════════════╝              │
#         │                                                          │
#         │  [CORE_BL (R0)]                   [CORE_BR (MY)]          │
#     ylo └──────────────────────────────────────────────────────────┘
#          xlo                                                    xhi
#
# Pin-direction rationale (bsg_fakeram pins are on the WEST edge in R0;
# see sram_64x*_*.lef: PORT RECT 0.000 … 0.070 …):
#   R90  — 90° CCW : west edge rotates onto the SOUTH edge → pins ↓
#   R270 — 90° CW  : west edge rotates onto the NORTH edge → pins ↑
#   Both R90/R270 swap W×H, making data macros 333.2 × 165.87 µm
#   (horizontal) instead of 165.87 × 333.2 µm (vertical) in R0.
#
# Orientation rationale for the cores (VX_socket_top pins on west edge):
#   BL (R0)    — pins face west (outward);  fine for die-edge ports.
#   BR (MY)    — mirror-Y: pins face east (outward).
#   TL (MX)    — mirror-X: pins still west; row flipped onto north.
#   TR (R180)  — 180°: pins face east; row flipped onto north.
#
# INSTANCE PATH SKELETON (verified against the synthesized full-vortex
# netlist via live runtime probes — see debug-00b12c.log, runId=post-fix):
#   sfx  = g_sync.g_bram.g_macro.macro_inst
#   l2   = g_clusters[0].cluster/l2cache/g_cache.cache
#
# Note: L2 is a *direct* VX_cache_wrap instance (hw/rtl/VX_cluster.sv:86–108),
# not wrapped in a VX_cache_cluster, so the path has NO `g_cache_wrap[0]`
# generate-array level. L2 geometry (4 banks × 8 ways) is also verified
# at runtime; see debug-00b12c.log lines 7-9 of the most recent run.
#
#   L2 tag   : $l2/g_banks[$b].bank/cache_tags/g_tag_store[$w].tag_store/$sfx
#   L2 data  : $l2/g_banks[$b].bank/cache_data/g_data_store[$w].data_store/$sfx
#   Core     : g_clusters[0].cluster/g_sockets[$s].socket
#=======================================================================

#-----------------------------------------------------------------------
# 3a. Floorplan — load the pre-computed area budget (or fall back to util).
#-----------------------------------------------------------------------
set budget_file "$RUN_DIR/floorplan_budget.tcl"
if {[file exists $budget_file]} {
    puts "INFO: Sourcing computed area budget from $budget_file"
    source $budget_file
    puts "INFO: Floorplan — W=$FLOORPLAN_W  H=$FLOORPLAN_H  margin=$FLOORPLAN_MARGIN µm"
    floorPlan \
        -site FreePDK45_38x28_10R_NP_162NW_34O \
        -s $FLOORPLAN_W $FLOORPLAN_H \
        $FLOORPLAN_MARGIN $FLOORPLAN_MARGIN $FLOORPLAN_MARGIN $FLOORPLAN_MARGIN
} else {
    puts "WARNING: area-budget file missing; falling back to 65% util square floorplan."
    floorPlan \
        -site FreePDK45_38x28_10R_NP_162NW_34O \
        -r 1.0 0.65 10 10 10 10
}

# Query the snapped core bounding box — all placements below are derived
# from these snapped values so they stay on-grid regardless of mode.
lassign [join [get_db current_design .core_bbox]] xlo ylo xhi yhi
set die_w [expr {$xhi - $xlo}]
set die_h [expr {$yhi - $ylo}]
puts "INFO: Core bbox: xlo=$xlo ylo=$ylo xhi=$xhi yhi=$yhi  (W=[format %.1f $die_w]  H=[format %.1f $die_h])"

#-----------------------------------------------------------------------
# 3b. Constants — macro dimensions, halos and gaps.
#-----------------------------------------------------------------------

# Halo around every bsg_fakeram macro.
set MACRO_HALO   5.0

# Larger halo around the four VX_socket_top core macros: the blackbox
# boundary carries many pins on every side, so we leave extra breathing
# room for the top-level router to reach them.
set CORE_HALO   10.0

# Minimum macro-to-macro edge gap. Must exceed 2×HALO so halo keepouts
# don't merge into a single blocked region.
set MACRO_GAP    [expr {2 * $MACRO_HALO + 0.5}]  ;# 10.5 µm

# Enlarged vertical gap used only between L2 rows whose pins face each
# other (rows 2↔3 and rows 4↔5). Gives the router a wide routing
# channel to fan out the pins of both facing macros.
set L2_FACING_GAP [expr {8 * $MACRO_GAP}]  ;# ~168 µm

# Extra horizontal breathing gap between cores and the central L2 block.
# Driven by CORE_HALO so the channel between a corner core and the L2
# row is at least ~2 tracks wide after halos are applied.
set CORE_L2_GAP  [expr {2 * $CORE_HALO + 5.0}]   ;# ~25 µm

# Inset applied to every core macro from the nearest die edge so the
# top-level router has a clear peripheral channel to reach the core
# blackbox pins *and* the I/O pin ring.
set CORE_EDGE_OFFSET  40.0   ;# µm

# L2 SRAM LEF-native (R0) footprints.
set W_l2tag   76.950 ;  set H_l2tag   63.000  ;# placeholder — overwritten below
set W_l2data 165.870 ;  set H_l2data 333.200  ;# placeholder — overwritten below

# Read the actual L2 SRAM sizes from the loaded LEFs via the physical
# database, so this script stays correct if the bsg_fakeram sizes change.
# H9 FIX (runtime-confirmed): the L2 tag/data macros are actually
# sram_64x24_1r1w and sram_64x512_1rw, not the 256x* variants. This was
# verified by querying a live inst's base_cell in the synthesized
# netlist (debug-00b12c.log:5-6, runId=post-fix).
set _l2tag_cell  [get_db base_cells sram_64x24_1r1w]
set _l2data_cell [get_db base_cells sram_64x512_1rw]
if {$_l2tag_cell eq "" || $_l2data_cell eq ""} {
    error "L2 SRAM base cells (sram_64x24_1r1w / sram_64x512_1rw) not found.\nCheck init_lef_file in 01_init_design.tcl."
}
# #region agent log — debug-00b12c: JSON NDJSON logger (retained)
# H1/H2 (rect.dx/.dy) verified; probe loop retired. Helper procs kept
# because later instrumentation (L2 path discovery) reuses them.
proc _dbg_esc {s} {
    set s [string map [list \\ \\\\ \" \\\" "\n" \\n "\r" \\r "\t" \\t] $s]
    return $s
}
proc _dbg_log {loc msg data} {
    set _f [open "/home/rid2/cs533.work/vortex/.cursor/debug-00b12c.log" a]
    set _ts [clock milliseconds]
    set _loc [_dbg_esc $loc]
    set _msg [_dbg_esc $msg]
    set _dat [_dbg_esc $data]
    puts $_f "{\"sessionId\":\"00b12c\",\"runId\":\"pre-fix\",\"hypothesisId\":\"H6-H8\",\"location\":\"$_loc\",\"message\":\"$_msg\",\"data\":\"$_dat\",\"timestamp\":$_ts}"
    close $_f
}
# #endregion agent log

# NOTE: Innovus 21.19's `rect` object has NO `.height` attribute, and
# `.width` exists only as an asymmetric alias. The canonical accessors
# are `.dx` (width) and `.dy` (height), confirmed at runtime; see the
# debug log referenced in the region above. Using .dx/.dy keeps the
# pair symmetric and avoids relying on the `.width` alias.
set W_l2tag  [get_db $_l2tag_cell  .bbox.dx]
set H_l2tag  [get_db $_l2tag_cell  .bbox.dy]
set W_l2data [get_db $_l2data_cell .bbox.dx]
set H_l2data [get_db $_l2data_cell .bbox.dy]
puts "INFO: L2 SRAM sizes — tag ${W_l2tag}×${H_l2tag} µm, data ${W_l2data}×${H_l2data} µm"

# #region agent log — debug-00b12c: post-fix verification
_dbg_log "03_floorplan_full-vortex.tcl:148" "post-fix L2 sizes" "W_l2tag=$W_l2tag H_l2tag=$H_l2tag W_l2data=$W_l2data H_l2data=$H_l2data"
# Re-tag subsequent runs as post-fix by redefining the logger's runId.
# (We overwrite the proc in place so downstream log entries show up
# under runId=post-fix instead of pre-fix.)
proc _dbg_log {loc msg data} {
    set _f [open "/home/rid2/cs533.work/vortex/.cursor/debug-00b12c.log" a]
    set _ts [clock milliseconds]
    set _loc [_dbg_esc $loc]
    set _msg [_dbg_esc $msg]
    set _dat [_dbg_esc $data]
    puts $_f "{\"sessionId\":\"00b12c\",\"runId\":\"post-fix\",\"hypothesisId\":\"H1-H2\",\"location\":\"$_loc\",\"message\":\"$_msg\",\"data\":\"$_dat\",\"timestamp\":$_ts}"
    close $_f
}
# #endregion agent log

# Core blackbox dimensions come from VX_socket_top.lef loaded in
# 01_init_design.tcl.  Pull them out of the DB so this script picks up
# the actual size produced by the single-core PnR run.
set _core_cell [get_db base_cells VX_socket_top]
if {$_core_cell eq ""} {
    error "VX_socket_top base cell not found.\nVerify that VX_socket_top.lef was loaded in 01_init_design.tcl."
}
set W_core [get_db $_core_cell .bbox.dx]
set H_core [get_db $_core_cell .bbox.dy]
puts "INFO: Core macro VX_socket_top — ${W_core} × ${H_core} µm"
# #region agent log — debug-00b12c
_dbg_log "03_floorplan_full-vortex.tcl:W_core" "post-fix core size" "W_core=$W_core H_core=$H_core"
# #endregion agent log

#-----------------------------------------------------------------------
# 3c. Build instance-name lists for the full-vortex netlist.
#-----------------------------------------------------------------------
set sfx     "g_sync.g_bram.g_macro.macro_inst"
set cluster "g_clusters\[0\].cluster"
# H6 FIX (runtime-confirmed): L2 is a *direct* VX_cache_wrap instance in
# VX_cluster (see hw/rtl/VX_cluster.sv:86-108), NOT wrapped in a
# VX_cache_cluster's `g_cache_wrap` generate array. So the L2 hierarchy
# goes straight from `l2cache` into the inner `g_cache.cache` generate,
# with no intermediate `g_cache_wrap[0].cache_wrap/` segment. Confirmed
# by runtime probe — `*g_cache_wrap*` has count=0 in the netlist and the
# actual first L2 tag store path is
# `g_clusters[0].cluster/l2cache/g_cache.cache/g_banks[0].bank/cache_tags/g_tag_store[0].tag_store/...`
# (debug-00b12c.log line 6, runId=post-fix).
set l2      "$cluster/l2cache/g_cache.cache"

# H7 FIX (runtime-confirmed): the full-vortex L2 uses 4 banks × 8 ways,
# not the 4×4 previously assumed. Confirmed at runtime by counting unique
# g_banks[*] / g_tag_store[*] indices in the synthesized netlist
# (debug-00b12c.log:7-9, runId=post-fix). 4×8×2 (tag+data) = 64 macros,
# which also matches the total macro_inst count observed earlier.
set L2_NUM_BANKS 4
set L2_NUM_WAYS  8

set l2_tags {} ; set l2_data {}
for {set b 0} {$b < $L2_NUM_BANKS} {incr b} {
    for {set w 0} {$w < $L2_NUM_WAYS} {incr w} {
        lappend l2_tags "$l2/g_banks\[$b\].bank/cache_tags/g_tag_store\[$w\].tag_store/$sfx"
        lappend l2_data "$l2/g_banks\[$b\].bank/cache_data/g_data_store\[$w\].data_store/$sfx"
    }
}

set cores {}
for {set s 0} {$s < 4} {incr s} {
    lappend cores "$cluster/g_sockets\[$s\].socket"
}

# #region agent log — debug-00b12c: post-fix verification (H6/H7/H9)
# Runtime check that every constructed L2 / core inst path actually
# resolves BEFORE we call placeInstance. Any missing path would cause
# IMPTCM-162 later, so failing early here makes post-fix verification
# unambiguous in the debug log.
set _unresolved 0
foreach _inst [concat $l2_tags $l2_data $cores] {
    set _obj [get_db insts $_inst]
    if {$_obj eq ""} { incr _unresolved ; _dbg_log "03_floorplan_full-vortex.tcl:verify" "unresolved inst" $_inst }
}
_dbg_log "03_floorplan_full-vortex.tcl:verify" "post-fix L2+cores path resolution" \
    "l2_tags=[llength $l2_tags] l2_data=[llength $l2_data] cores=[llength $cores] unresolved=$_unresolved"
if {$_unresolved > 0} {
    error "DEBUG-00b12c: $_unresolved constructed inst paths did not resolve. See debug-00b12c.log."
}
# #endregion agent log

#-----------------------------------------------------------------------
# 3d. L2 block — 6 horizontal rows, centred on the die.
#
#   Row 1 (top)    : 16 × tag   @ R270 (pins ↑, outward)
#   Row 2          :  8 × data  @ R90  (pins ↓)   ┐ pair faces each other
#   Row 3          :  8 × data  @ R270 (pins ↑)   ┘
#   Row 4          :  8 × data  @ R90  (pins ↓)   ┐ pair faces each other
#   Row 5          :  8 × data  @ R270 (pins ↑)   ┘
#   Row 6 (bottom) : 16 × tag   @ R90  (pins ↓, outward)
#
# In R90/R270 the macro bounding box is the R0 box transposed:
#   data : 333.20 (W) × 165.87 (H)   ← horizontal (was vertical in R0)
#   tag  :  63.00 (W) ×  76.95 (H)
#
# The two tag rows share a row-width target; both are centred on cx.
# The four data rows are stacked on a common x-origin (also centred on cx).
#-----------------------------------------------------------------------

# ---- Split the instance lists into per-row slices. --------------------
# Tags: 32 total → 16 for the top row, 16 for the bottom row.
set N_TAG_PER_ROW   16
set l2_tags_top [lrange $l2_tags 0                    [expr {$N_TAG_PER_ROW - 1}]]
set l2_tags_bot [lrange $l2_tags $N_TAG_PER_ROW       end]

# Data: 32 total → 8 per row × 4 rows (row 2 = top data row, row 5 = bottom).
set N_DATA_PER_ROW   8
set l2_data_r2 [lrange $l2_data 0                         [expr {    $N_DATA_PER_ROW - 1}]]
set l2_data_r3 [lrange $l2_data     $N_DATA_PER_ROW       [expr {2 * $N_DATA_PER_ROW - 1}]]
set l2_data_r4 [lrange $l2_data [expr {2 * $N_DATA_PER_ROW}] [expr {3 * $N_DATA_PER_ROW - 1}]]
set l2_data_r5 [lrange $l2_data [expr {3 * $N_DATA_PER_ROW}] end]

# ---- Rotated (R90/R270) macro footprints. -----------------------------
# R90 and R270 both transpose the R0 bounding box — W'←H, H'←W.
set W_tag_rot   $H_l2tag          ;# = 76.95 µm  (originally H, now W)
set H_tag_rot   $W_l2tag          ;# = 63.00 µm
set W_data_rot  $H_l2data         ;# = 333.20 µm (data is now horizontal)
set H_data_rot  $W_l2data         ;# = 165.87 µm

# ---- Row widths / block height. ---------------------------------------
set tag_row_w  [expr {$N_TAG_PER_ROW  * $W_tag_rot  + ($N_TAG_PER_ROW  - 1) * $MACRO_GAP}]
set data_row_w [expr {$N_DATA_PER_ROW * $W_data_rot + ($N_DATA_PER_ROW - 1) * $MACRO_GAP}]

# Six rows stacked vertically. Of the 5 inter-row gaps, 3 use the
# minimum MACRO_GAP (rows whose facing edges have no pins: 1↔2, 3↔4,
# 5↔6) and 2 use the enlarged L2_FACING_GAP (rows 2↔3 and 4↔5, where
# pins of both sides point into the gap).
set l2_block_h [expr {2 * $H_tag_rot + 4 * $H_data_rot \
                      + 3 * $MACRO_GAP + 2 * $L2_FACING_GAP}]

# Widest row governs the block's X span.
set l2_block_w [expr {$data_row_w > $tag_row_w ? $data_row_w : $tag_row_w}]

# ---- Centre the block on the die. -------------------------------------
set cx [expr {$xlo + $die_w / 2.0}]
set cy [expr {$ylo + $die_h / 2.0}]

set block_y0 [expr {$cy - $l2_block_h / 2.0}]    ;# bottom of row 6
set x0_tag   [expr {$cx - $tag_row_w  / 2.0}]
set x0_data  [expr {$cx - $data_row_w / 2.0}]

# ---- Per-row y-origin (lower-left of each oriented bbox). -------------
# Gap convention per inter-row boundary (y grows up, so row N is ABOVE
# row N+1):
#   Row 5 ↔ 6 : no pins on facing edges            → MACRO_GAP
#   Row 4 ↔ 5 : Row 4 pins ↓, Row 5 pins ↑  (FACE) → L2_FACING_GAP
#   Row 3 ↔ 4 : no pins on facing edges            → MACRO_GAP
#   Row 2 ↔ 3 : Row 2 pins ↓, Row 3 pins ↑  (FACE) → L2_FACING_GAP
#   Row 1 ↔ 2 : no pins on facing edges            → MACRO_GAP
set y_row6 $block_y0
set y_row5 [expr {$y_row6 + $H_tag_rot  + $MACRO_GAP}]
set y_row4 [expr {$y_row5 + $H_data_rot + $L2_FACING_GAP}]
set y_row3 [expr {$y_row4 + $H_data_rot + $MACRO_GAP}]
set y_row2 [expr {$y_row3 + $H_data_rot + $L2_FACING_GAP}]
set y_row1 [expr {$y_row2 + $H_data_rot + $MACRO_GAP}]

# ---- Helper: place a horizontal row of identical-sized macros. --------
proc _place_l2_row {insts x0 y w_each gap orient} {
    set x $x0
    foreach inst $insts {
        placeInstance $inst $x $y $orient -fixed
        set x [expr {$x + $w_each + $gap}]
    }
}

# ---- Row 1 — top tags, R270 (pins ↑, outward). ------------------------
_place_l2_row $l2_tags_top $x0_tag  $y_row1 $W_tag_rot  $MACRO_GAP R270
# ---- Row 2 — data, R90  (pins ↓).  Row 2/3 face each other. -----------
_place_l2_row $l2_data_r2  $x0_data $y_row2 $W_data_rot $MACRO_GAP R90
# ---- Row 3 — data, R270 (pins ↑). -------------------------------------
_place_l2_row $l2_data_r3  $x0_data $y_row3 $W_data_rot $MACRO_GAP R270
# ---- Row 4 — data, R90  (pins ↓).  Row 4/5 face each other. -----------
_place_l2_row $l2_data_r4  $x0_data $y_row4 $W_data_rot $MACRO_GAP R90
# ---- Row 5 — data, R270 (pins ↑). -------------------------------------
_place_l2_row $l2_data_r5  $x0_data $y_row5 $W_data_rot $MACRO_GAP R270
# ---- Row 6 — bottom tags, R90 (pins ↓, outward). ----------------------
_place_l2_row $l2_tags_bot $x0_tag  $y_row6 $W_tag_rot  $MACRO_GAP R90

# ---- Aggregate L2 bounding box (for diagnostics below). ---------------
set l2_xlo [expr {$cx - $l2_block_w / 2.0}]
set l2_xhi [expr {$cx + $l2_block_w / 2.0}]
set l2_ylo $block_y0
set l2_yhi [expr {$block_y0 + $l2_block_h}]

#-----------------------------------------------------------------------
# 3e. Cores — one in each die corner, inset by CORE_EDGE_OFFSET from
#     every adjacent die edge so the top-level router has a routing
#     channel between the core boundary and the die ring.
#
# Orientations are the same as before (logic faces the die centre).
#-----------------------------------------------------------------------
set core_bl_x  [expr {$xlo + $CORE_EDGE_OFFSET}]
set core_br_x  [expr {$xhi - $W_core - $CORE_EDGE_OFFSET}]
set core_tl_x  [expr {$xlo + $CORE_EDGE_OFFSET}]
set core_tr_x  [expr {$xhi - $W_core - $CORE_EDGE_OFFSET}]

set core_bl_y  [expr {$ylo + $CORE_EDGE_OFFSET}]
set core_br_y  [expr {$ylo + $CORE_EDGE_OFFSET}]
set core_tl_y  [expr {$yhi - $H_core - $CORE_EDGE_OFFSET}]
set core_tr_y  [expr {$yhi - $H_core - $CORE_EDGE_OFFSET}]

placeInstance [lindex $cores 0] $core_bl_x $core_bl_y R0   -fixed
placeInstance [lindex $cores 1] $core_br_x $core_br_y MY   -fixed
placeInstance [lindex $cores 2] $core_tl_x $core_tl_y MX   -fixed
placeInstance [lindex $cores 3] $core_tr_x $core_tr_y R180 -fixed

#-----------------------------------------------------------------------
# 3f. Diagnostics — verify the L2 block and corner cores have adequate
#     clearance on all four sides.
#-----------------------------------------------------------------------
set core_inner_top_y  [expr {$yhi - $H_core - $CORE_EDGE_OFFSET}]
set core_inner_bot_y  [expr {$ylo + $H_core + $CORE_EDGE_OFFSET}]
set core_inner_lft_x  [expr {$xlo + $W_core + $CORE_EDGE_OFFSET}]
set core_inner_rgt_x  [expr {$xhi - $W_core - $CORE_EDGE_OFFSET}]

set l2_to_top_clr [expr {$core_inner_top_y - $l2_yhi}]
set l2_to_bot_clr [expr {$l2_ylo - $core_inner_bot_y}]
set l2_to_lft_clr [expr {$l2_xlo - $core_inner_lft_x}]
set l2_to_rgt_clr [expr {$core_inner_rgt_x - $l2_xhi}]

puts "INFO: full-vortex floorplan (6-row horizontal L2 layout)"
puts "      Die W=[format %.1f $die_w]  H=[format %.1f $die_h]"
puts "      Core macro : ${W_core} × ${H_core} µm  (4 instances, inset $CORE_EDGE_OFFSET µm from each edge)"
puts "      L2 macro   : tag(rot)=${W_tag_rot}×${H_tag_rot} µm  data(rot)=${W_data_rot}×${H_data_rot} µm"
puts "      L2 rows    : tag row W=[format %.1f $tag_row_w] µm,  data row W=[format %.1f $data_row_w] µm,  block H=[format %.1f $l2_block_h] µm"
puts "      L2 gaps    : normal=$MACRO_GAP µm,  facing-pin rows (2↔3, 4↔5)=$L2_FACING_GAP µm"
puts "      L2 bbox    : xlo=[format %.1f $l2_xlo]  xhi=[format %.1f $l2_xhi]  ylo=[format %.1f $l2_ylo]  yhi=[format %.1f $l2_yhi]"
puts "      L2 vert  clearance: top=[format %.1f $l2_to_top_clr] µm  bottom=[format %.1f $l2_to_bot_clr] µm"
puts "      L2 horiz clearance: left=[format %.1f $l2_to_lft_clr] µm  right=[format %.1f $l2_to_rgt_clr] µm"

if {$l2_to_top_clr < $CORE_HALO || $l2_to_bot_clr < $CORE_HALO} {
    puts "WARNING: L2 block vertical clearance to cores (< $CORE_HALO µm) is tight."
    puts "         Consider lowering --utilization or raising --margin_pct."
}
if {$l2_to_lft_clr < $CORE_L2_GAP || $l2_to_rgt_clr < $CORE_L2_GAP} {
    puts "WARNING: L2 block horizontal clearance to corner cores (< $CORE_L2_GAP µm) is tight."
    puts "         Consider raising --aspect (wider die) or --margin_pct."
}

#-----------------------------------------------------------------------
# 3g. Halos — keepout around every macro; cores get a larger halo.
#-----------------------------------------------------------------------
# All bsg_fakeram macros (L2 tags + data).
addHaloToBlock $MACRO_HALO $MACRO_HALO $MACRO_HALO $MACRO_HALO -allMacro

# Extra halo specifically around each VX_socket_top instance. Uses the
# explicit instance-path list rather than -allMacro / -allBlackBox so
# we don't double-apply halos to the L2 SRAMs (addHaloToBlock
# overwrites per-instance). The per-instance form is POSITIONAL in
# Innovus 21.19 — there is no `-inst` flag; the instance path is the
# last token. Confirmed by Innovus's own usage printout.
foreach inst $cores {
    addHaloToBlock $CORE_HALO $CORE_HALO $CORE_HALO $CORE_HALO $inst
}
puts "INFO: Halos — SRAM=$MACRO_HALO µm, core=$CORE_HALO µm; macro-to-macro gap=$MACRO_GAP µm"

#=======================================================================
# 3h. I-O pin placement — distribute top-level ports across all 4 sides.
#     M4 on top/bottom (horizontal), M5 on left/right (vertical).
#=======================================================================
set all_ports [get_db ports .name]
set n_ports   [llength $all_ports]
if {$n_ports == 0} {
    puts "WARNING: no top-level ports; skipping editPin."
} else {
    puts "INFO: distributing $n_ports ports across 4 die edges."
    set per_side [expr {$n_ports / 4}]
    set s_left   [lrange $all_ports 0                      [expr {1 * $per_side - 1}]]
    set s_bot    [lrange $all_ports $per_side              [expr {2 * $per_side - 1}]]
    set s_right  [lrange $all_ports [expr {2 * $per_side}] [expr {3 * $per_side - 1}]]
    set s_top    [lrange $all_ports [expr {3 * $per_side}] end]

    editPin -pin $s_left  -side Left   -layer M5 -spreadType SIDE
    editPin -pin $s_bot   -side Bottom -layer M4 -spreadType SIDE
    editPin -pin $s_right -side Right  -layer M5 -spreadType SIDE
    editPin -pin $s_top   -side Top    -layer M4 -spreadType SIDE

    puts "INFO: pins — L:[llength $s_left]  B:[llength $s_bot]  R:[llength $s_right]  T:[llength $s_top]"
}
