# Elaboration-only check. Same link setup as dc_syn.tcl; stops after check_design.

foreach _v {CONFIG TOP} {
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
    if {![file exists $_db]} { error "SRAM .db not found: $_db" }
    lappend link_library $_db
}

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

foreach _cell $blackbox_cells {
    set _n [sizeof_collection [get_cells -hier -filter "ref_name == $_cell"]]
    if {$_n == 0} { error "BLACKBOX_DBS listed $_cell but no $_cell instances in linked design." }
    puts "INFO: $_n $_cell instance(s)."
}

report_hierarchy
exit
