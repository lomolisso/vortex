# Helpers for scripts/floorplan/*.tcl. Sourced by 03_floorplan.tcl;
# all read the global `cfg` array.

proc init_die_from_budget {xlo_var ylo_var xhi_var yhi_var label} {
    upvar 1 $xlo_var xlo $ylo_var ylo $xhi_var xhi $yhi_var yhi
    global cfg
    floorPlan -site FreePDK45_38x28_10R_NP_162NW_34O \
        -s $cfg(budget.floorplan_w_um) $cfg(budget.floorplan_h_um) \
           $cfg(budget.floorplan_margin_lr_um) $cfg(budget.floorplan_margin_tb_um) \
           $cfg(budget.floorplan_margin_lr_um) $cfg(budget.floorplan_margin_tb_um)
    lassign [join [get_db current_design .core_bbox]] xlo ylo xhi yhi
    puts [format "INFO: %s bbox  W=%.1f H=%.1f" $label \
              [expr {$xhi - $xlo}] [expr {$yhi - $ylo}]]
}

proc require_base_cells {names {hint ""}} {
    set missing {}
    foreach n $names {
        if {[get_db base_cells $n] eq ""} { lappend missing $n }
    }
    if {[llength $missing] > 0} {
        set msg "base cell(s) not loaded: $missing"
        if {$hint ne ""} { append msg "  ($hint)" }
        error $msg
    }
}

proc base_cell_bbox {name w_var h_var} {
    upvar 1 $w_var w $h_var h
    require_base_cells [list $name]
    set bc [get_db base_cells $name]
    set w  [get_db $bc .bbox.dx]
    set h  [get_db $bc .bbox.dy]
}

proc require_insts {paths label {hint ""}} {
    set missing {}
    foreach p $paths {
        if {[get_db insts $p] eq ""} { lappend missing $p }
    }
    if {[llength $missing] > 0} {
        set msg "[llength $missing] $label inst path(s) failed to resolve. First few: [lrange $missing 0 4]"
        if {$hint ne ""} { append msg "  ($hint)" }
        error $msg
    }
}

# Skips entries with no halo set (link-only macros legitimately omit it).
proc load_halo_map {map_var} {
    upvar 1 $map_var halo
    global cfg
    array unset halo
    if {![info exists cfg(macros.count)]} return
    for {set i 0} {$i < $cfg(macros.count)} {incr i} {
        if {[info exists cfg(macros.$i.halo_um)]} {
            set halo($cfg(macros.$i.name)) $cfg(macros.$i.halo_um)
        }
    }
}

proc apply_halo {inst h} {
    addHaloToBlock $h $h $h $h $inst
}

proc apply_uniform_halo {h} {
    addHaloToBlock $h $h $h $h -allMacro
}

proc place_pins_by_distribution {} {
    global cfg
    set ports [get_db ports .name]
    set np    [llength $ports]
    if {$np == 0} {
        puts "WARNING: no top-level ports; skipping editPin."
        return
    }
    set n_left  [expr {int($np * $cfg(pins.side_distribution.left))}]
    set n_bot   [expr {int($np * $cfg(pins.side_distribution.bottom))}]
    set n_right [expr {int($np * $cfg(pins.side_distribution.right))}]
    set s_left  [lrange $ports 0                                    [expr {$n_left - 1}]]
    set s_bot   [lrange $ports $n_left                              [expr {$n_left + $n_bot - 1}]]
    set s_right [lrange $ports [expr {$n_left + $n_bot}]            [expr {$n_left + $n_bot + $n_right - 1}]]
    set s_top   [lrange $ports [expr {$n_left + $n_bot + $n_right}] end]

    setPinAssignMode -pinEditInBatch true
    if {[llength $s_left]  > 0} { editPin -pin $s_left  -side Left   -layer $cfg(pins.layer_lr) -spreadType SIDE }
    if {[llength $s_bot]   > 0} { editPin -pin $s_bot   -side Bottom -layer $cfg(pins.layer_tb) -spreadType SIDE }
    if {[llength $s_right] > 0} { editPin -pin $s_right -side Right  -layer $cfg(pins.layer_lr) -spreadType SIDE }
    if {[llength $s_top]   > 0} { editPin -pin $s_top   -side Top    -layer $cfg(pins.layer_tb) -spreadType SIDE }
    setPinAssignMode -pinEditInBatch false
    puts "INFO: pins — L:[llength $s_left] B:[llength $s_bot] R:[llength $s_right] T:[llength $s_top]"
}
