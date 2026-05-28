set init_mmmc_file "$SCRIPT_DIR/scripts/utils/mmmc.tcl"
set init_verilog   $NETLIST
set init_top_cell  $TOP
set init_pwr_net   VDD
set init_gnd_net   VSS

# IMPCCOPT-4255: CTS needs unique netlist tree per instance.
set init_design_uniquify 1

set init_lef_file [list $LIBS_DIR/rtk-tech.lef $LIBS_DIR/stdcells.lef]
if {[info exists cfg(macros.count)]} {
    for {set _i 0} {$_i < $cfg(macros.count)} {incr _i} {
        set _name $cfg(macros.$_i.name)
        if {![info exists cfg(macros.$_i.lef_path)]} {
            error "macros\[$_i\] '$_name' has no lef_path"
        }
        set _lef $cfg(macros.$_i.lef_path)
        if {![string match "/*" $_lef]} { set _lef "$LIBS_DIR/$_lef" }
        if {![file exists $_lef]} {
            error "Macro LEF not found for $_name: $_lef"
        }
        lappend init_lef_file $_lef
    }
}

init_design

_apply_design_modes
puts [expr {$TOP_RLAYER > 0 \
    ? "INFO: $CONFIG — routing window M1..M$TOP_RLAYER." \
    : "INFO: $CONFIG — routing window not capped (using tech default)."}]

# fixFanoutLoad is the only knob that *fixes* max_fanout.
setOptMode -fixFanoutLoad true
setOptMode -fixCap        true
setOptMode -fixTran       true
setOptMode -drcMargin     0.02

# IMPTCM-70: setOptMode -effort is obsolete; use setDesignMode -flowEffort.
switch -- $OPT_EFFORT {
    low     { set _fe express  }
    medium  { set _fe standard }
    high    { set _fe extreme  }
    default { set _fe standard }
}
setDesignMode -flowEffort $_fe
puts "INFO: flowEffort=$_fe  (EFFORT=$OPT_EFFORT)."

# TCLCMD-1048: derate via set_interactive_constraint_modes context.
set_interactive_constraint_modes [list func]
if {[info exists cfg(clock.max_fanout)]} {
    set_max_fanout $cfg(clock.max_fanout) [current_design]
    puts "INFO: set_max_fanout $cfg(clock.max_fanout) (from TOML)."
}
set_timing_derate -delay_corner typDC -early 0.92 -late 1.08 -cell_delay
set_timing_derate -delay_corner typDC -early 0.92 -late 1.08 -net_delay
set_interactive_constraint_modes {}
