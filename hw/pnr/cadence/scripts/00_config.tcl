foreach var {CONFIG LIBS_DIR SYN_DIR} {
    if {[info exists ::env($var)]} {
        set $var $::env($var)
    } elseif {![info exists $var]} {
        error "$var not set (use the Makefile or export it)."
    }
}
if {[info exists ::env(EFFORT)]} {
    set OPT_EFFORT $::env(EFFORT)
} elseif {![info exists OPT_EFFORT]} {
    set OPT_EFFORT low
}

set CONFIG_TOML "$SCRIPT_DIR/config/$CONFIG.toml"
if {![file exists $CONFIG_TOML]} { error "Config TOML not found: $CONFIG_TOML" }
source "$SCRIPT_DIR/scripts/toml_load.tcl"
toml::load $CONFIG_TOML cfg

set TOP $cfg(design.top)
# 0 = use tech-LEF top layer.
set TOP_RLAYER [expr {[info exists cfg(design.top_routing_layer)] ? int($cfg(design.top_routing_layer)) : 0}]
set FP_RECIPE  [expr {[info exists cfg(floorplan.recipe)] ? $cfg(floorplan.recipe) : ""}]

# DESIGN_KIND drives PG mesh + LEF abstract layers; must match routing window.
if {![info exists cfg(design.kind)]} { error "TOML missing design.kind (leaf|socket|cluster)." }
set DESIGN_KIND $cfg(design.kind)
if {[lsearch -exact {leaf socket cluster} $DESIGN_KIND] < 0} {
    error "design.kind='$DESIGN_KIND' invalid (expected leaf|socket|cluster)."
}

set RUN_DIR     [pwd]
set SYN_RESULTS "$SYN_DIR/runs/$CONFIG/results"
set REPORT_DIR  "$RUN_DIR/reports"
set RESULTS_DIR "$RUN_DIR/results"
set NETLIST     "$SYN_RESULTS/${TOP}_netlist.v"
set SDC_FILE    "$SYN_RESULTS/${TOP}.sdc"

if {![file exists $NETLIST]} { error "Netlist not found: $NETLIST (run 'make $CONFIG' under hw/syn/synopsys first)." }
if {![file exists $SDC_FILE]} { error "SDC not found: $SDC_FILE" }
file mkdir $REPORT_DIR $RESULTS_DIR

# SDC filter: TCLCMD-1461 (drop set_units), TCLNL-330 (clk-port input_delay →
# clock_latency), TCLNL-312 (collapse per-bit struct-field port refs).
set filtered_sdc "$RUN_DIR/${TOP}_filtered.sdc"
set raw_fh  [open $SDC_FILE r]
set filt_fh [open $filtered_sdc w]
set skipped 0 ; set replaced 0 ; set collapsed 0 ; set dropped 0
array set _bus_seen {}
while {[gets $raw_fh line] >= 0} {
    if {[regexp {^\s*set_units\s} $line]} { incr skipped; continue }
    if {[regexp {set_input_delay\s+-clock\s+\S+\s+-max\s+(\S+)\s+\[get_ports\s+clk\s*\]} $line _ _delay]} {
        puts $filt_fh "set_clock_latency -source $_delay \[get_clocks clk\]"
        incr replaced; continue
    }
    if {[regexp {^(.*)\[get_ports\s+\{([a-zA-Z_][a-zA-Z0-9_]*)(?:\[[^\]]+\]){2,}\}\](.*)$} $line _ pre name post]} {
        set key "[string trim $pre]|$name"
        if {[info exists _bus_seen($key)]} { incr dropped; continue }
        set _bus_seen($key) 1
        puts $filt_fh "${pre}\[get_ports $name\]${post}"
        incr collapsed
        continue
    }
    puts $filt_fh $line
}
close $raw_fh ; close $filt_fh
puts "INFO: SDC filter — dropped $skipped set_units; rewrote $replaced clk-port input_delay; collapsed $collapsed bus(es), dropped $dropped per-bit duplicate(s)."
set SDC_FILE $filtered_sdc

puts "INFO: $CONFIG  top=$TOP  kind=$DESIGN_KIND  recipe=$FP_RECIPE  rlayer=M1..M$TOP_RLAYER  effort=$OPT_EFFORT"

# Re-invoked on every checkpoint resume — these don't survive saveDesign+restore.
proc _apply_design_modes {} {
    global TOP_RLAYER
    setDesignMode -process 45
    set ::delaycal_use_default_delay_limit 1000
    setAnalysisMode -analysisType onChipVariation -cppr both
    setSIMode -enable_glitch_report true
    if {$TOP_RLAYER > 0} {
        setDesignMode -bottomRoutingLayer 1 -topRoutingLayer $TOP_RLAYER
    }
}
