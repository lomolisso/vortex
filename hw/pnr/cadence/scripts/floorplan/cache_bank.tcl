# Banked SRAM placement: data macros in a COLS×ROWS grid (ROWS from
# floorplan.params.grid_rows, COLS = num_ways/ROWS) and, when a tag macro is
# configured, the tag macros in an identically-shaped grid placed in its own
# cluster immediately to the right of the data cluster.
# The die is sized to the cluster geometry (plus room for the stdcells), not to
# an area/utilization estimate, because the short tag cluster beside the tall
# data cluster leaves whitespace the area model does not capture.
# Inst path: ${store_path}/${store_iter}[$w].${store_inst}/g_sync.g_bram.g_macro.macro_inst

set NUM_WAYS   $cfg(floorplan.params.num_ways)
set LEAF_MACRO $cfg(floorplan.params.leaf_macro)
set ROWS [expr {[info exists cfg(floorplan.params.grid_rows)] ? $cfg(floorplan.params.grid_rows) : 1}]
set COLS [expr {$NUM_WAYS / $ROWS}]
if {[expr {$COLS * $ROWS}] != $NUM_WAYS} {
    error "num_ways $NUM_WAYS not divisible by grid_rows $ROWS"
}

set store_path [expr {[info exists cfg(floorplan.params.store_path)] ? $cfg(floorplan.params.store_path) : "bank/cache_data"}]
set store_iter [expr {[info exists cfg(floorplan.params.store_iter)] ? $cfg(floorplan.params.store_iter) : "g_data_store"}]
set store_inst [expr {[info exists cfg(floorplan.params.store_inst)] ? $cfg(floorplan.params.store_inst) : "data_store"}]
set _prefix [expr {$store_path eq "" ? "" : "$store_path/"}]

set TAG_MACRO [expr {[info exists cfg(floorplan.params.tag_macro)] ? $cfg(floorplan.params.tag_macro) : ""}]
set tstore_path [expr {[info exists cfg(floorplan.params.tag_store_path)] ? $cfg(floorplan.params.tag_store_path) : "bank/cache_tags"}]
set tstore_iter [expr {[info exists cfg(floorplan.params.tag_store_iter)] ? $cfg(floorplan.params.tag_store_iter) : "g_tag_store"}]
set tstore_inst [expr {[info exists cfg(floorplan.params.tag_store_inst)] ? $cfg(floorplan.params.tag_store_inst) : "tag_store"}]
set _tprefix [expr {$tstore_path eq "" ? "" : "$tstore_path/"}]

# bsg_fakeram LEFs are flipPins=true → signal pins on the low-x edge; R90 rotates
# that edge to the bottom, matching side_distribution.bottom = 1.0. Both clusters
# are R90, so data and tag pins escape the same side.
set ROT R90
require_base_cells [list $LEAF_MACRO]
base_cell_bbox $LEAF_MACRO W_lef H_lef
set W_leaf $H_lef
set H_leaf $W_lef

set Wt 0.0
set Ht 0.0
if {$TAG_MACRO ne ""} {
    require_base_cells [list $TAG_MACRO]
    base_cell_bbox $TAG_MACRO Wt_lef Ht_lef
    set Wt $Ht_lef
    set Ht $Wt_lef
}

set ROW_GAP     30.0
set COL_GAP     30.0
set CLUSTER_GAP 40.0
set BOT_STRIP   30.0  ;# bottom channel for M3 pin escape

set _dw  [expr {$COLS * $W_leaf + ($COLS - 1) * $COL_GAP}]
set _tw  [expr {$TAG_MACRO ne "" ? $COLS * $Wt + ($COLS - 1) * $COL_GAP : 0.0}]
set _gap [expr {$TAG_MACRO ne "" ? $CLUSTER_GAP : 0.0}]
set _macros_w [expr {$_dw + $_gap + $_tw}]
set _core_h   [expr {$ROWS * $H_leaf + ($ROWS - 1) * $ROW_GAP + $BOT_STRIP}]

# Grow the core width if the macro bounding box can't host the stdcells at the
# target stdcell utilization.
set _raw     [expr {$COLS * $ROWS * ($W_leaf * $H_leaf + ($TAG_MACRO ne "" ? $Wt * $Ht : 0.0))}]
set _u       [expr {[info exists cfg(floorplan.target_utilization)] ? $cfg(floorplan.target_utilization) : 0.85}]
set _stdcell [expr {[info exists cfg(budget.stdcell_area_um2)] ? $cfg(budget.stdcell_area_um2) : 0.0}]
set _need    [expr {$_raw + $_stdcell / $_u}]
set _core_w  $_macros_w
if {[expr {$_core_w * $_core_h}] < $_need} {
    set _core_w [expr {$_need / $_core_h}]
}

set SITE_W 0.19
set SITE_H 1.4
set _core_w [expr {ceil($_core_w / $SITE_W) * $SITE_W}]
set _core_h [expr {ceil($_core_h / $SITE_H) * $SITE_H}]

set _mlr $cfg(budget.floorplan_margin_lr_um)
set _mtb $cfg(budget.floorplan_margin_tb_um)
floorPlan -site FreePDK45_38x28_10R_NP_162NW_34O -s \
    [expr {$_core_w + 2 * $_mlr}] [expr {$_core_h + 2 * $_mtb}] \
    $_mlr $_mtb $_mlr $_mtb
lassign [join [get_db current_design .core_bbox]] xlo ylo xhi yhi
puts [format "INFO: cache_bank — data ${COLS}×${ROWS} (post-$ROT W=%.1f H=%.1f), tag '${TAG_MACRO}' ${COLS}×${ROWS} (post-$ROT W=%.1f H=%.1f); core %.1f×%.1f" \
          $W_leaf $H_leaf $Wt $Ht [expr {$xhi - $xlo}] [expr {$yhi - $ylo}]]

proc place_cluster {prefix iter inst x0 macro_w macro_h cols rows col_gap row_gap yhi rot} {
    set _insts {}
    for {set w 0} {$w < [expr {$cols * $rows}]} {incr w} {
        lappend _insts "${prefix}${iter}\[$w\].${inst}/g_sync.g_bram.g_macro.macro_inst"
    }
    require_insts $_insts "banked sram macro"
    set _i 0
    foreach _inst $_insts {
        set _r [expr {$_i / $cols}]
        set _c [expr {$_i % $cols}]
        set _x [expr {$x0 + $_c * ($macro_w + $col_gap)}]
        set _y [expr {$yhi - ($_r + 1) * $macro_h - $_r * $row_gap}]
        placeInstance $_inst $_x $_y $rot -fixed
        incr _i
    }
    return $_insts
}

set _macros [place_cluster $_prefix $store_iter $store_inst $xlo $W_leaf $H_leaf $COLS $ROWS $COL_GAP $ROW_GAP $yhi $ROT]

set _tmacros {}
if {$TAG_MACRO ne ""} {
    set _tx0 [expr {$xlo + $_dw + $_gap}]
    set _tmacros [place_cluster $_tprefix $tstore_iter $tstore_inst $_tx0 $Wt $Ht $COLS $ROWS $COL_GAP $ROW_GAP $yhi $ROT]
}

load_halo_map _halo
foreach _inst $_macros  { apply_halo $_inst $_halo($LEAF_MACRO) }
foreach _inst $_tmacros { apply_halo $_inst $_halo($TAG_MACRO) }
place_pins_by_distribution
