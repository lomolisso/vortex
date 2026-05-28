require_base_cells {VX_local_mem_top VX_gpr_top \
                    VX_l1_icache_bank_top VX_l1_dcache_bank_top \
                    VX_fpu_top VX_tcu_top} \
    "check BLACKBOX_CORES + lef_path entries in 2d-single-core.toml"
base_cell_bbox VX_local_mem_top       W_lmem       H_lmem
base_cell_bbox VX_gpr_top             W_gpr        H_gpr
base_cell_bbox VX_l1_icache_bank_top  W_icache_bnk H_icache_bnk
base_cell_bbox VX_l1_dcache_bank_top  W_dcache_bnk H_dcache_bnk
base_cell_bbox VX_fpu_top             W_fpu        H_fpu
base_cell_bbox VX_tcu_top             W_tcu        H_tcu

set ROW_GAP        $cfg(floorplan.params.row_gap_um)
set COL_GAP        $cfg(floorplan.params.col_gap_um)
set PIN_ESCAPE_GAP $cfg(floorplan.params.pin_escape_gap_um)
set BOT_PIN_STRIP  $cfg(floorplan.params.bot_pin_strip_um)
set TOP_MARGIN     $cfg(floorplan.params.top_margin_um)
set ASPECT_HW      [expr {1.0 / $cfg(floorplan.aspect_ratio)}]

# OPC and D-cache placed R90 → original H becomes placed W, original W becomes placed H.
set W_opc_r90    $H_gpr
set H_opc_r90    $W_gpr
set W_dcache_r90 $H_dcache_bnk
set H_dcache_r90 $W_dcache_bnk

set _w_left  $W_dcache_r90
foreach _w [list $W_icache_bnk $W_opc_r90] { if {$_w > $_w_left} { set _w_left $_w } }
set _w_right $W_lmem
foreach _w [list $W_tcu $W_fpu] { if {$_w > $_w_right} { set _w_right $_w } }
set _h_left  [expr {$H_icache_bnk + $PIN_ESCAPE_GAP + $H_opc_r90 + $ROW_GAP + $H_dcache_r90}]
set _h_right [expr {$H_lmem + $PIN_ESCAPE_GAP + $H_tcu + $ROW_GAP + $H_fpu}]
set _h_max   [expr {$_h_left > $_h_right ? $_h_left : $_h_right}]
set _w_min   [expr {$_w_left + $COL_GAP + $_w_right}]
set _h_min   [expr {$TOP_MARGIN + $_h_max + $BOT_PIN_STRIP}]

set core_w $_w_min
set core_h [expr {$ASPECT_HW * $core_w}]
if {$core_h < $_h_min} {
    set core_h $_h_min
    set core_w [expr {$core_h / $ASPECT_HW}]
    if {$core_w < $_w_min} { set core_w $_w_min; set core_h [expr {$ASPECT_HW * $core_w}] }
}

set SITE_W 0.19
set SITE_H 1.4
set core_w [expr {ceil($core_w / $SITE_W) * $SITE_W}]
set core_h [expr {ceil($core_h / $SITE_H) * $SITE_H}]

set _mlr $cfg(budget.floorplan_margin_lr_um)
set _mtb $cfg(budget.floorplan_margin_tb_um)
floorPlan -site FreePDK45_38x28_10R_NP_162NW_34O -s \
    [expr {$core_w + 2 * $_mlr}] [expr {$core_h + 2 * $_mtb}] \
    $_mlr $_mtb $_mlr $_mtb
lassign [join [get_db current_design .core_bbox]] xlo ylo xhi yhi
puts [format "INFO: socket core bbox W=%.1f H=%.1f (aspect H/W=%.3f)" \
        [expr {$xhi - $xlo}] [expr {$yhi - $ylo}] \
        [expr {($yhi - $ylo) / ($xhi - $xlo)}]]

set sock "socket"
set _lmem  "$sock/g_cores\[0\].core/mem_unit/local_mem"
set _gpr   "$sock/g_cores\[0\].core/issue/g_slices\[0\].issue_slice/operands/g_collectors\[0\].opc_unit"
set _ibank "$sock/icache/g_cache_wrap\[0\].cache_wrap/g_cache.cache/g_banks\[0\].g_bank.bank"
set _dbank "$sock/dcache/g_cache_wrap\[0\].cache_wrap/g_cache.cache/g_banks\[0\].g_bank.bank"
set _fpu   "$sock/g_cores\[0\].core/execute/fpu_top"
set _tcu   "$sock/g_cores\[0\].core/execute/tcu_top"
require_insts [list $_lmem $_gpr $_ibank $_dbank $_fpu $_tcu] \
    "socket hard-macro children" ""

set _y_icache_bot  [expr {$yhi - $TOP_MARGIN - $H_icache_bnk}]
set _y_opc_bot     [expr {$_y_icache_bot - $PIN_ESCAPE_GAP - $H_opc_r90}]
set _y_dcache_bot  [expr {$_y_opc_bot - $ROW_GAP - $H_dcache_r90}]
set _y_lmem_bot    [expr {$yhi - $TOP_MARGIN - $H_lmem}]
set _y_tcu_bot     [expr {$_y_lmem_bot - $PIN_ESCAPE_GAP - $H_tcu}]
set _y_fpu_bot     [expr {$_y_tcu_bot - $ROW_GAP - $H_fpu}]

placeInstance $_ibank $xlo                       $_y_icache_bot R0  -fixed
placeInstance $_gpr   $xlo                       $_y_opc_bot    R90 -fixed
placeInstance $_dbank $xlo                       $_y_dcache_bot R90 -fixed
# MY (mirror about Y-axis) keeps LMEM pins on the macro's bottom edge so they face the die bottom.
placeInstance $_lmem  [expr {$xhi - $W_lmem}]    $_y_lmem_bot   MY  -fixed
placeInstance $_tcu   [expr {$xhi - $W_tcu}]     $_y_tcu_bot    R0  -fixed
placeInstance $_fpu   [expr {$xhi - $W_fpu}]     $_y_fpu_bot    R0  -fixed

set _macros [list $_lmem $_gpr $_ibank $_dbank $_fpu $_tcu]
set _bb {}
foreach _i $_macros {
    lassign [join [get_db [get_db insts $_i] .bbox]] _x1 _y1 _x2 _y2
    lappend _bb [list $_i $_x1 $_y1 $_x2 $_y2]
}
for {set i 0} {$i < [llength $_bb]} {incr i} {
    for {set j [expr {$i + 1}]} {$j < [llength $_bb]} {incr j} {
        lassign [lindex $_bb $i] _n1 _a1 _b1 _a2 _b2
        lassign [lindex $_bb $j] _n2 _c1 _d1 _c2 _d2
        if {$_a1 < $_c2 && $_a2 > $_c1 && $_b1 < $_d2 && $_b2 > $_d1} {
            error "macro overlap: $_n1 vs $_n2"
        }
    }
}

load_halo_map _halo
foreach _inst $_macros {
    set _ref [get_db [get_db insts $_inst] .base_cell.name]
    if {[info exists _halo($_ref)]} { apply_halo $_inst $_halo($_ref) }
}

place_pins_by_distribution
