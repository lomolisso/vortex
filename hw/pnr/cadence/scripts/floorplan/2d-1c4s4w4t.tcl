# 4-socket cluster: one socket per corner (BL R0, BR MY, TL MX, TR R180),
# L2 banks in a grid at die centre.
# L2 inst paths: cluster/l2cache/g_cache.cache/g_banks[$b].g_bank.bank

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
set l2_yhi [expr {$grid_y0 + $grid_h}]
set l2_ylo $grid_y0

set core_bl_x [expr {$xlo + $CORE_EDGE_OFFSET}]
set core_br_x [expr {$xhi - $W_core - $CORE_EDGE_OFFSET}]
set core_bl_y [expr {$ylo + $CORE_EDGE_OFFSET}]
set core_tl_y [expr {$yhi - $H_core - $CORE_EDGE_OFFSET}]
placeInstance [lindex $cores 0] $core_bl_x $core_bl_y R0   -fixed
placeInstance [lindex $cores 1] $core_br_x $core_bl_y MY   -fixed
placeInstance [lindex $cores 2] $core_bl_x $core_tl_y MX   -fixed
placeInstance [lindex $cores 3] $core_br_x $core_tl_y R180 -fixed

set l2_to_top_clr [expr {$yhi - $H_core - $CORE_EDGE_OFFSET - $l2_yhi}]
set l2_to_bot_clr [expr {$l2_ylo - ($ylo + $H_core + $CORE_EDGE_OFFSET)}]
puts [format "INFO: cluster corner, $NUM_CORES sockets, $NUM_BANKS L2 banks ($GRID_ROWS×$GRID_COLS) — grid W=%.1f H=%.1f  clr top=%.1f bot=%.1f" \
        $grid_w $grid_h $l2_to_top_clr $l2_to_bot_clr]
if {$l2_to_top_clr < $CORE_L2_GAP || $l2_to_bot_clr < $CORE_L2_GAP} {
    error "L2 bank grid doesn't fit between corner cores — clr [format %.1f $l2_to_top_clr]/[format %.1f $l2_to_bot_clr] < CORE_L2_GAP $CORE_L2_GAP."
}

load_halo_map _halo
foreach inst $l2_banks { apply_halo $inst $_halo(VX_l2_bank_top) }
foreach inst $cores    { apply_halo $inst $_halo(VX_socket_top)  }

place_pins_by_distribution
