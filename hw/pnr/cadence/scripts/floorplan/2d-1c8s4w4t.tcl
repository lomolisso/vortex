# 8-socket cluster: two horizontal core rows with L2 bank grid between.

set cluster_path "cluster"

init_die_from_budget xlo ylo xhi yhi "cluster core"
set die_w [expr {$xhi - $xlo}]
set die_h [expr {$yhi - $ylo}]

set MACRO_GAP        $cfg(floorplan.params.macro_gap_um)
set CORE_L2_GAP      $cfg(floorplan.params.core_l2_gap_um)
set CORE_EDGE_OFFSET $cfg(floorplan.params.core_edge_offset_um)
set L2_BANK_GAP      $cfg(floorplan.params.l2_bank_gap_um)
set NUM_CORES        $cfg(floorplan.params.num_cores)
set NUM_BANKS        $cfg(floorplan.params.l2_num_banks)
set GRID_COLS        $cfg(floorplan.params.l2_bank_grid_cols)
set GRID_ROWS        $cfg(floorplan.params.l2_bank_grid_rows)
if {[expr {$GRID_COLS * $GRID_ROWS}] != $NUM_BANKS} {
    error "l2_bank_grid $GRID_ROWS×$GRID_COLS != l2_num_banks=$NUM_BANKS"
}

require_base_cells {VX_socket_top VX_l2_bank_top}
base_cell_bbox VX_socket_top   W_core H_core
base_cell_bbox VX_l2_bank_top  W_l2b  H_l2b

set cores {}
for {set s 0} {$s < $NUM_CORES} {incr s} {
    lappend cores "$cluster_path/g_sockets\[$s\].socket"
}
set l2_banks {}
for {set b 0} {$b < $NUM_BANKS} {incr b} {
    lappend l2_banks "$cluster_path/l2cache/g_cache.cache/g_banks\[$b\].g_bank.bank"
}
require_insts [concat $cores $l2_banks] "cluster sockets + L2 banks" \
    "verify ASIC_SYNTHESIS + CACHE_KIND='l2' set in VX_cluster.sv"

set grid_w [expr {$GRID_COLS * $W_l2b + ($GRID_COLS - 1) * $L2_BANK_GAP}]
set grid_h [expr {$GRID_ROWS * $H_l2b + ($GRID_ROWS - 1) * $L2_BANK_GAP}]
set grid_x0 [expr {$xlo + ($die_w - $grid_w) / 2.0}]
set grid_y0 [expr {$ylo + ($die_h - $grid_h) / 2.0}]
for {set b 0} {$b < $NUM_BANKS} {incr b} {
    set _row [expr {$b / $GRID_COLS}]
    set _col [expr {$b % $GRID_COLS}]
    set _x [expr {$grid_x0 + $_col * ($W_l2b + $L2_BANK_GAP)}]
    set _y [expr {$grid_y0 + $_row * ($H_l2b + $L2_BANK_GAP)}]
    placeInstance [lindex $l2_banks $b] $_x $_y R0 -fixed
}
set l2_ylo $grid_y0
set l2_yhi [expr {$grid_y0 + $grid_h}]

# NUM_CORES/2 per row; top row R90, bottom R270.
set _per_row [expr {$NUM_CORES / 2}]
set W_core_rot $H_core
set H_core_rot $W_core
set _row_span    [expr {$die_w - 2 * $CORE_EDGE_OFFSET}]
set CORE_ROW_GAP [expr {($_row_span - $_per_row * $W_core_rot) / double($_per_row - 1)}]
if {$CORE_ROW_GAP < $MACRO_GAP} {
    error "Inter-core gap [format %.1f $CORE_ROW_GAP] µm < MACRO_GAP ($MACRO_GAP)."
}
set core_row_x0    [expr {$xlo + $CORE_EDGE_OFFSET}]
set core_row_top_y [expr {$yhi - $H_core_rot - $CORE_EDGE_OFFSET}]
set core_row_bot_y [expr {$ylo + $CORE_EDGE_OFFSET}]
for {set i 0} {$i < $_per_row} {incr i} {
    set _x [expr {$core_row_x0 + $i * ($W_core_rot + $CORE_ROW_GAP)}]
    placeInstance [lindex $cores $i]                      $_x $core_row_top_y R90  -fixed
    placeInstance [lindex $cores [expr {$i + $_per_row}]] $_x $core_row_bot_y R270 -fixed
}

set l2_to_top_clr [expr {$core_row_top_y - $l2_yhi}]
set l2_to_bot_clr [expr {$l2_ylo - ($ylo + $H_core_rot + $CORE_EDGE_OFFSET)}]
puts [format "INFO: cluster 2-row, $NUM_CORES sockets, $NUM_BANKS L2 banks ($GRID_ROWS×$GRID_COLS) — grid W=%.1f H=%.1f  clr top=%.1f bot=%.1f" \
        $grid_w $grid_h $l2_to_top_clr $l2_to_bot_clr]
if {$l2_to_top_clr < $CORE_L2_GAP || $l2_to_bot_clr < $CORE_L2_GAP} {
    error "L2 bank grid doesn't fit between core rows — clr [format %.1f $l2_to_top_clr]/[format %.1f $l2_to_bot_clr] < CORE_L2_GAP $CORE_L2_GAP."
}

load_halo_map _halo
foreach inst $l2_banks { apply_halo $inst $_halo(VX_l2_bank_top) }
foreach inst $cores    { apply_halo $inst $_halo(VX_socket_top)  }

place_pins_by_distribution
