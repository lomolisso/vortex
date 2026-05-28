# Vortex DC synthesis. Driver vars from Makefile via dc_shell -x:
#   CONFIG, TOP, TARGET_FREQ_MHZ, MAX_FANOUT
#   SRAM_DBS      space-separated .db paths
#   BLACKBOX_DBS  space-separated <module>:<db> pairs

foreach _v {CONFIG TOP TARGET_FREQ_MHZ MAX_FANOUT} {
    if {![info exists $_v]} { error "$_v not set." }
}
foreach _v {SRAM_DBS BLACKBOX_DBS} {
    if {![info exists $_v]} { set $_v "" }
}

set SCRIPT_DIR [file normalize [file dirname [info script]]]
set RUN_DIR    [pwd]
set LIBS_DIR   "$SCRIPT_DIR/libs"

file mkdir "$RUN_DIR/work"
define_design_lib WORK -path "$RUN_DIR/work"

set target_library "$LIBS_DIR/stdcells.db"
set link_library   [list * "$LIBS_DIR/stdcells.db"]
set symbol_library {}

foreach _db $SRAM_DBS {
    if {![file exists $_db]} { error "SRAM .db not found: $_db (run 'make libs')." }
    lappend link_library $_db
}
puts "INFO: linked [llength $SRAM_DBS] SRAM .db file(s)."

set blackbox_cells [list]
foreach pair $BLACKBOX_DBS {
    if {$pair eq ""} continue
    set _i [string first ":" $pair]
    if {$_i < 0} { error "BLACKBOX_DBS entry '$pair' missing ':' (need <module>:<db>)." }
    set _cell [string range $pair 0 [expr {$_i - 1}]]
    set _db   [string range $pair [expr {$_i + 1}] end]
    if {![file exists $_db]} { error "Blackbox .db not found for $_cell: $_db" }
    lappend link_library  $_db
    lappend blackbox_cells $_cell
    puts "INFO: $_cell linked from $_db"
}

source "$SCRIPT_DIR/../../scripts/parse_vcs_list.tcl"
set flist_path "$SCRIPT_DIR/flists/dc_flist_${CONFIG}.f"
if {![file exists $flist_path]} { error "Filelist not found: $flist_path (run 'make flist_${CONFIG}')." }
lassign [parse_vcs_list $flist_path] src_files inc_dirs defines
set search_path [concat $search_path $inc_dirs]

# defs_div_sqrt_mvp.sv is a package; *_pkg.sv glob misses it.
set _i [lsearch -glob $src_files "*defs_div_sqrt_mvp.sv"]
if {$_i >= 0} {
    set _f [lindex $src_files $_i]
    set src_files [linsert [lreplace $src_files $_i $_i] 0 $_f]
}

foreach _cell $blackbox_cells {
    set src_files [lsearch -inline -all -not -glob $src_files "*/${_cell}.sv"]
}

foreach f $src_files {
    analyze -format sverilog -define $defines $f
}
elaborate $TOP
current_design $TOP
link
check_design

# PnR placeInstance paths bake in the wrapper's `macro_inst` hierarchy.
foreach _w {sram_32x512_1rw sram_64x512_1rw sram_256x512_1rw sram_1024x32_1rw sram_64x128_1r1w} {
    if {[get_designs -quiet $_w] ne ""} { set_dont_touch [get_designs $_w] }
}

foreach _cell $blackbox_cells {
    set _insts [get_cells -hier -filter "ref_name == $_cell"]
    set _n [sizeof_collection $_insts]
    if {$_n == 0} { error "BLACKBOX_DBS listed $_cell but no $_cell instances in linked design." }
    set_dont_touch $_insts
    puts "INFO: dont_touch on $_n $_cell instance(s)."
}

set CLK_PERIOD [expr {1000.0 / double($TARGET_FREQ_MHZ)}]
puts "INFO: $CONFIG SDC — ${TARGET_FREQ_MHZ} MHz (period ${CLK_PERIOD} ns), max_fanout ${MAX_FANOUT}."

create_clock -name clk -period $CLK_PERIOD [get_ports clk]
set_ideal_network                          [get_ports clk]
set_max_fanout $MAX_FANOUT                 [get_ports clk]
set_false_path -from                       [get_ports reset]
set_max_fanout $MAX_FANOUT                 [get_ports reset]
set_clock_uncertainty [expr {$CLK_PERIOD * 0.05}] [get_clocks clk]
set_input_delay  -clock clk -max [expr {$CLK_PERIOD * 0.2}] [all_inputs]
set_output_delay -clock clk -max [expr {$CLK_PERIOD * 0.2}] [all_outputs]
set_max_fanout     $MAX_FANOUT [current_design]
set_max_transition 0.5         [current_design]

compile_ultra -no_autoungroup

set REPORT_DIR "$RUN_DIR/reports"
file mkdir $REPORT_DIR
report_area                      > "$REPORT_DIR/area.rpt"
report_timing -nworst 10         > "$REPORT_DIR/timing.rpt"
report_power                     > "$REPORT_DIR/power.rpt"
report_qor                       > "$REPORT_DIR/qor.rpt"
report_hierarchy                 > "$REPORT_DIR/hierarchy.rpt"
report_constraint -all_violators > "$REPORT_DIR/constraints.rpt"

set RESULTS_DIR "$RUN_DIR/results"
file mkdir $RESULTS_DIR
write_file -hierarchy -format verilog -output "$RESULTS_DIR/${TOP}_netlist.v"
write_file -format ddc                -output "$RESULTS_DIR/${TOP}.ddc"
write_sdc                                     "$RESULTS_DIR/${TOP}.sdc"

# TCLNL-312: strip 2-D SV port index from SDC to match flattened netlist.
set _sdc "$RESULTS_DIR/${TOP}.sdc"
set _fh [open $_sdc r] ; set _txt [read $_fh] ; close $_fh
regsub -all {\{([a-zA-Z_][a-zA-Z0-9_]*)\[[0-9]+\]\[([0-9]+)\]\}} $_txt {{\1[\2]}} _txt
set _fh [open $_sdc w] ; puts -nonewline $_fh $_txt ; close $_fh

exit
