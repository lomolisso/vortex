# Socket layout: LMem (top centre), I$/D$ bank macros (left/right),
# GPR (centre), FPU+TCU (below GPR). All children are hard macros.

init_die_from_budget xlo ylo xhi yhi "socket core"
set die_w [expr {$xhi - $xlo}]
set die_h [expr {$yhi - $ylo}]

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

set MACRO_GAP      $cfg(floorplan.params.macro_gap_um)
set LMEM_CACHE_GAP $cfg(floorplan.params.lmem_cache_gap_um)
set EXEC_GAP       [expr {[info exists cfg(floorplan.params.exec_unit_gap_um)] ? $cfg(floorplan.params.exec_unit_gap_um) : 20.0}]

set sock "socket"

# Hierarchical inst paths (NUM_BANKS=1, NUM_OPCS=1, SOCKET_SIZE=1).
set _lmem  "$sock/g_cores\[0\].core/mem_unit/local_mem"
set _gpr   "$sock/g_cores\[0\].core/issue/g_slices\[0\].issue_slice/operands/g_collectors\[0\].opc_unit"
set _ibank "$sock/icache/g_cache_wrap\[0\].cache_wrap/g_cache.cache/g_banks\[0\].g_bank.bank"
set _dbank "$sock/dcache/g_cache_wrap\[0\].cache_wrap/g_cache.cache/g_banks\[0\].g_bank.bank"
set _fpu   "$sock/g_cores\[0\].core/execute/fpu_top"
set _tcu   "$sock/g_cores\[0\].core/execute/tcu_top"

require_insts [list $_lmem $_gpr $_ibank $_dbank $_fpu $_tcu] \
    "socket hard-macro children" \
    "verify ASIC_SYNTHESIS + CACHE_KIND propagation to VX_cache_cluster"

# LMem: top of die, centred, R0.
set y_lmem  [expr {$yhi - $H_lmem}]
set x_lmem  [expr {$xlo + ($die_w - $W_lmem) / 2.0}]
placeInstance $_lmem $x_lmem $y_lmem R0 -fixed
set cache_yhi [expr {$y_lmem - $LMEM_CACHE_GAP}]

# I$/D$ banks side-by-side below LMem.
set H_caches [expr {$H_icache_bnk > $H_dcache_bnk ? $H_icache_bnk : $H_dcache_bnk}]
set y_caches [expr {$cache_yhi - $H_caches}]
if {$y_caches < $ylo} { set y_caches $ylo }
placeInstance $_ibank $xlo                          $y_caches R0 -fixed
placeInstance $_dbank [expr {$xhi - $W_dcache_bnk}] $y_caches R0 -fixed
puts [format "INFO: channel width = %.1f µm" \
        [expr {($xhi - $W_dcache_bnk) - ($xlo + $W_icache_bnk)}]]

# GPR: centred between caches.
set x_gpr [expr {$xlo + ($die_w - $W_gpr) / 2.0}]
set y_gpr [expr {$ylo + ($die_h - $H_gpr) / 2.0}]
placeInstance $_gpr $x_gpr $y_gpr R0 -fixed

# FPU + TCU side-by-side, just below the GPR block.
set exec_row_w [expr {$W_fpu + $EXEC_GAP + $W_tcu}]
set exec_row_h [expr {$H_fpu > $H_tcu ? $H_fpu : $H_tcu}]
set exec_x0    [expr {$xlo + ($die_w - $exec_row_w) / 2.0}]
set exec_y0    [expr {$y_gpr - $MACRO_GAP - $exec_row_h}]
if {$exec_y0 < $ylo} { set exec_y0 $ylo }
placeInstance $_fpu $exec_x0                               $exec_y0 R0 -fixed
placeInstance $_tcu [expr {$exec_x0 + $W_fpu + $EXEC_GAP}] $exec_y0 R0 -fixed

# Halos: per-macro from TOML map.
load_halo_map _halo
foreach _inst [list $_lmem $_gpr $_ibank $_dbank $_fpu $_tcu] {
    set _ref [get_db [get_db insts $_inst] .base_cell.name]
    if {[info exists _halo($_ref)]} { apply_halo $_inst $_halo($_ref) }
}

place_pins_by_distribution
