#=======================================================================
# Stage 3 — Floorplan / Macro placement / I-O pins  (config: 1c1n4w4t)
#
# Hand-tuned, fully explicit placement for the baseline 1c1n4w4t config.
# No loops over clusters/cores, no parametric helpers: every macro is
# placed at a known (x, y) with a known orientation.
#
# MACRO COUNT (20 macros total):
#   4  icache tag     sram_64x24_1r1w      76.950 × 63.000
#   4  icache data    sram_64x512_1rw     165.870 × 333.200
#   4  dcache tag     sram_64x24_1r1w      76.950 × 63.000
#   4  dcache data    sram_64x512_1rw     165.870 × 333.200
#   4  GPR RAMs       sram_64x128_1r1w    121.220 × 183.400
#   4  LMem RAMs      sram_1024x32_1rw    104.500 × 317.800
#
# LAYOUT (X grows right, Y grows up).  bsg_fakeram pins are natively on
# the WEST edge; orientation choices below target each block's pins to
# face toward the stdcell region they talk to.
#
#     yhi ┌──────────────────────────────────────────────────────────┐
#         │              LMem row (4 × R90, pins → south)            │
#         │                                                          │
#         │ [IDAT][ IT ]             [ GPR ][ GPR ]           [ DT ][ DDAT ] │
#         │  col    tag-band (W_data)  col1 │ col2            tag-band  col  │
#         │  MY     MY (centred)        R0  │  MY              R0 (centred) R0 │
#         │                           pins← │ pins→                           │
#         │                                                          │
#         │         (empty — stdcells fill this band)                │
#     ylo └──────────────────────────────────────────────────────────┘
#          xlo                                                    xhi
#
# Orientation recap:
#   LMem row        R90 → pins face south (pin edge rotated onto south)
#   I-cache IDAT+IT MY  → pins on east (face stdcells, toward die centre)
#   D-cache DDAT+DT R0  → pins on west (face stdcells, toward die centre)
#   GPR col-1       R0  → pins on west (face −X / left)
#   GPR col-2       MY  → pins on east (face +X / right)   ← mirror-Y
#
# (MX alone keeps pins on west and only flips Y; that is why an earlier
#  revision left both GPR columns pointing left.  MY flips X, i.e. pins.)
#
# INSTANCE PATH SKELETON:
#   sfx  = g_sync.g_bram.g_macro.macro_inst
#   sock = <caller-set> — the socket instance path from the PnR top cell
#          · legacy 1c1n4w4t top  (Vortex)       : g_clusters[0].cluster/g_sockets[0].socket
#          · single-core     top  (VX_socket_top): socket
#          If $sock is not pre-set by the caller, we default to the legacy
#          path below (preserves historical behaviour).
#
#   Cache tag   : $sock/<ic|dc>/g_cache_wrap[0].cache_wrap/g_cache.cache/
#                   g_banks[0].bank/cache_tags/g_tag_store[$i].tag_store/$sfx
#   Cache data  : $sock/<ic|dc>/g_cache_wrap[0].cache_wrap/g_cache.cache/
#                   g_banks[0].bank/cache_data/g_data_store[$i].data_store/$sfx
#   GPR         : $sock/g_cores[0].core/issue/g_slices[0].issue_slice/
#                   operands/g_collectors[0].opc_unit/
#                   g_gpr_rams[$b].gpr_ram/$sfx
#   LMem        : $sock/g_cores[0].core/mem_unit/local_mem/
#                   g_data_store[$b].lmem_store/$sfx
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
puts "INFO: Core bbox: xlo=$xlo ylo=$ylo xhi=$xhi yhi=$yhi"

#-----------------------------------------------------------------------
# 3b. Constants — macro dimensions, halos and gaps.
#-----------------------------------------------------------------------

# Halo applied to every macro (see addHaloToBlock at the end of the script).
set MACRO_HALO   5.0

# Minimum macro-to-macro edge gap.  Must exceed 2×HALO so halo keepouts
# don't merge into a blocked seam.  Kept small so spacing is driven by
# halos, not by wide artificial channels.
set MACRO_GAP    [expr {2 * $MACRO_HALO + 0.5}]  ;# 10.5 µm

# Clear space between each cache data column and its matching tag column.
# Larger than MACRO_GAP so the tag macro does not sit on the data pin
# access corridor.  Bump if pin access looks blocked in the GUI.
set TAG_DATA_GAP 25.0

# LEF-native (R0) footprints (µm).
set W_tag   76.950 ;  set H_tag   63.000   ;# sram_64x24_1r1w
set W_data 165.870 ;  set H_data 333.200   ;# sram_64x512_1rw
set W_gpr  121.220 ;  set H_gpr  183.400   ;# sram_64x128_1r1w  (W=4)
set W_lmem 104.500 ;  set H_lmem 317.800   ;# sram_1024x32_1rw

# LMem macros sit in the top band rotated R90; in die coords width ←→ height.
set Wr_lmem $H_lmem ; set Hr_lmem $W_lmem

#-----------------------------------------------------------------------
# 3c. Build instance-name lists for the netlist.
#
# The $sock variable may be pre-set by the caller (e.g. by
# 03_floorplan_single-core.tcl before sourcing this file). Fall back to
# the legacy Vortex-top path when it is not provided so that directly
# invoking this stage for a Vortex-top configuration still works.
#-----------------------------------------------------------------------
set sfx  "g_sync.g_bram.g_macro.macro_inst"
if {![info exists sock]} {
    set sock "g_clusters\[0\].cluster/g_sockets\[0\].socket"
}

set itags {} ; set idata {}
set dtags {} ; set ddata {}
for {set i 0} {$i < 4} {incr i} {
    lappend itags "$sock/icache/g_cache_wrap\[0\].cache_wrap/g_cache.cache/g_banks\[0\].bank/cache_tags/g_tag_store\[$i\].tag_store/$sfx"
    lappend idata "$sock/icache/g_cache_wrap\[0\].cache_wrap/g_cache.cache/g_banks\[0\].bank/cache_data/g_data_store\[$i\].data_store/$sfx"
    lappend dtags "$sock/dcache/g_cache_wrap\[0\].cache_wrap/g_cache.cache/g_banks\[0\].bank/cache_tags/g_tag_store\[$i\].tag_store/$sfx"
    lappend ddata "$sock/dcache/g_cache_wrap\[0\].cache_wrap/g_cache.cache/g_banks\[0\].bank/cache_data/g_data_store\[$i\].data_store/$sfx"
}

set lmems {}
for {set b 0} {$b < 4} {incr b} {
    lappend lmems "$sock/g_cores\[0\].core/mem_unit/local_mem/g_data_store\[$b\].lmem_store/$sfx"
}

set gprs {}
for {set b 0} {$b < 4} {incr b} {
    lappend gprs "$sock/g_cores\[0\].core/issue/g_slices\[0\].issue_slice/operands/g_collectors\[0\].opc_unit/g_gpr_rams\[$b\].gpr_ram/$sfx"
}

#-----------------------------------------------------------------------
# 3d. LMem row at the top (R90, pins → south / toward stdcells below).
#     Horizontally centred across the die.
#-----------------------------------------------------------------------
set n_lmem   [llength $lmems]
set lmem_row_w [expr {$n_lmem * $Wr_lmem + ($n_lmem - 1) * $MACRO_GAP}]
set die_w      [expr {$xhi - $xlo}]
set lmem_x0    [expr {$xlo + ($die_w - $lmem_row_w) / 2.0}]
set y_lmem     [expr {$yhi - $Hr_lmem}]

set x $lmem_x0
foreach inst $lmems {
    placeInstance $inst $x $y_lmem R90 -fixed
    set x [expr {$x + $Wr_lmem + $MACRO_GAP}]
}

# Cache stacks top-anchor immediately below the LMem row.
set cache_yhi [expr {$y_lmem - $MACRO_GAP}]

#-----------------------------------------------------------------------
# 3e. I-cache (left side).
#
#   IDAT column [MY]  at xlo  (data at die edge, widest macros outermost).
#   IT   column [MY]  with TAG_DATA_GAP clearance from IDAT; each tag is
#                     centred inside a W_data-wide band so its X centroid
#                     matches its data way (tag "in front of" its data).
#   Per-way Y row:    way i sits at y_idata + i·(H_data + MACRO_GAP).
#-----------------------------------------------------------------------
set data_stack_h [expr {4 * $H_data + 3 * $MACRO_GAP}]
set y_idata      [expr {$cache_yhi - $data_stack_h}]
if {$y_idata < $ylo} { set y_idata $ylo }  ;# clamp: Innovus will warn on overlap

set col_idata  $xlo
set x_itag0    [expr {$col_idata + $W_data + $TAG_DATA_GAP + ($W_data - $W_tag) / 2.0}]
# Tag's Y is centred on the data's vertical span so the tag sits in the
# middle of its corresponding data macro (not flush with the bottom).
for {set i 0} {$i < 4} {incr i} {
    set y_data [expr {$y_idata + $i * ($H_data + $MACRO_GAP)}]
    set y_tag  [expr {$y_data + ($H_data - $H_tag) / 2.0}]
    placeInstance [lindex $idata $i] $col_idata $y_data MY -fixed
    placeInstance [lindex $itags $i] $x_itag0   $y_tag  MY -fixed
}
set icache_east [expr {$x_itag0 + $W_tag}]

#-----------------------------------------------------------------------
# 3f. D-cache (right side, mirror layout of I-cache).
#
#   DDAT column [R0] at xhi  (data at die edge).
#   DT   column [R0] with TAG_DATA_GAP clearance from DDAT; each tag
#                     centred in a W_data-wide band (same X centroid as data).
#-----------------------------------------------------------------------
set col_ddata [expr {$xhi - $W_data}]
set x_dtag0   [expr {$col_ddata - $TAG_DATA_GAP - $W_data + ($W_data - $W_tag) / 2.0}]
for {set i 0} {$i < 4} {incr i} {
    set y_data [expr {$y_idata + $i * ($H_data + $MACRO_GAP)}]
    set y_tag  [expr {$y_data + ($H_data - $H_tag) / 2.0}]
    placeInstance [lindex $ddata $i] $col_ddata $y_data R0 -fixed
    placeInstance [lindex $dtags $i] $x_dtag0   $y_tag  R0 -fixed
}
set dcache_west $x_dtag0

#-----------------------------------------------------------------------
# 3g. GPR — two columns centred horizontally and vertically on the die.
#
#   Banks 0, 1  → left  column, orientation R0 → pins on WEST (face left).
#   Banks 2, 3  → right column, orientation MY → pins on EAST (face right).
#
#   MY = mirror across Y axis, which flips the pin edge from west to east.
#   (MX mirrors across X: pins remain on west → wrong, both columns face
#   the same direction.)
#-----------------------------------------------------------------------
set gpr_block_w [expr {2 * $W_gpr + $MACRO_GAP}]
set gpr_block_h [expr {2 * $H_gpr + $MACRO_GAP}]
set die_h       [expr {$yhi - $ylo}]
set x_gpr_l     [expr {$xlo + ($die_w - $gpr_block_w) / 2.0}]
set x_gpr_r     [expr {$x_gpr_l + $W_gpr + $MACRO_GAP}]
set y_gpr_b     [expr {$ylo + ($die_h - $gpr_block_h) / 2.0}]
set y_gpr_t     [expr {$y_gpr_b + $H_gpr + $MACRO_GAP}]

placeInstance [lindex $gprs 0] $x_gpr_l $y_gpr_b R0 -fixed
placeInstance [lindex $gprs 1] $x_gpr_l $y_gpr_t R0 -fixed
placeInstance [lindex $gprs 2] $x_gpr_r $y_gpr_b MY -fixed
placeInstance [lindex $gprs 3] $x_gpr_r $y_gpr_t MY -fixed

#-----------------------------------------------------------------------
# 3h. Diagnostics — central channel, fit checks.
#-----------------------------------------------------------------------
set channel_w   [expr {$dcache_west - $icache_east}]
set channel_pct [expr {100.0 * $channel_w / $die_w}]
set cache_over  [expr {$data_stack_h - ($cache_yhi - $ylo)}]
set gpr_top     [expr {$y_gpr_t + $H_gpr}]
set lmem_gpr    [expr {$y_lmem - $gpr_top}]

puts "INFO: 1c1n4w4t floorplan"
puts "      Die W=[format %.1f $die_w]  H=[format %.1f $die_h]"
puts "      LMem row y=[format %.1f $y_lmem]   cache_yhi=[format %.1f $cache_yhi]"
puts "      I-cache : col_idata=[format %.1f $col_idata]  col_itag=[format %.1f $x_itag0]  east=[format %.1f $icache_east]"
puts "      D-cache : col_ddata=[format %.1f $col_ddata]  col_dtag=[format %.1f $x_dtag0]  west=[format %.1f $dcache_west]"
puts "      GPR     : left x=[format %.1f $x_gpr_l]  right x=[format %.1f $x_gpr_r]  y_b=[format %.1f $y_gpr_b]  y_t=[format %.1f $y_gpr_t]"
puts "      Channel : [format %.1f $channel_w] µm ([format %.0f $channel_pct]% of die)"
puts "      LMem↔GPR clearance: [format %.1f $lmem_gpr] µm"

if {$cache_over > 0.01} {
    puts "WARNING: cache data stack overflows vertical budget by [format %.1f $cache_over] µm."
    puts "         Re-run with lower util, e.g.  make 1c1n4w4t STAGE=3 UTIL=0.60"
}
if {$channel_w < [expr {$gpr_block_w + 2 * $MACRO_GAP}]} {
    puts "WARNING: central channel ([format %.1f $channel_w] µm) too narrow for GPR 2×2 block."
}

#-----------------------------------------------------------------------
# 3i. Macro halos — routing/stdcell keepout around every placed SRAM.
#-----------------------------------------------------------------------
addHaloToBlock $MACRO_HALO $MACRO_HALO $MACRO_HALO $MACRO_HALO -allMacro
puts "INFO: Macro halo = $MACRO_HALO µm; macro-to-macro gap = $MACRO_GAP µm"
puts "      Clear tracks between halos: [expr {$MACRO_GAP - 2 * $MACRO_HALO}] µm"

#=======================================================================
# 3j. I-O pin placement — distribute top-level ports across all 4 sides.
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
