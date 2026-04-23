# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#=======================================================================
# Vortex GPGPU — Synopsys DC Elaboration-Only Check
#
# Analyzes and elaborates without running compile_ultra.
# Use this to catch unresolved modules, DPI imports, string-type
# errors, and other DC parse issues before committing to a full run.
#
# Usage:
#   dc_shell -x "set CONFIG single-core; set TOP VX_socket_top" -f dc_elab.tcl
#   dc_shell -x "set CONFIG full-vortex; set TOP Vortex"        -f dc_elab.tcl
#   grep -E "Error|Warning|unresolved" logs/<config>_elab.log
#=======================================================================

if {![info exists CONFIG]} {
    error "CONFIG is not set. Invoke as: dc_shell -x {set CONFIG <config_name>; set TOP <top_module>} -f dc_elab.tcl"
}
if {![info exists TOP]} {
    set TOP Vortex
}

# BLACKBOX_SOCKET — mirror dc_syn.tcl so elaborate-only checks exercise
# the same library/analyze paths the real run would.
if {![info exists BLACKBOX_SOCKET]} {
    set BLACKBOX_SOCKET [expr {$CONFIG eq "full-vortex"}]
}

set SCRIPT_DIR [file normalize [file dirname [info script]]]
set RUN_DIR    [pwd]

#-----------------------------------------------------------------------
# Work library — redirect DC's intermediate .pvl/.syn/.mr files into
# work/ under the per-config run directory.
#-----------------------------------------------------------------------
file mkdir "$RUN_DIR/work"
define_design_lib WORK -path "$RUN_DIR/work"

#-----------------------------------------------------------------------
# Library setup (same as dc_syn.tcl — needed for link)
#-----------------------------------------------------------------------
set LIBS_DIR "$SCRIPT_DIR/libs"

set target_library "$LIBS_DIR/stdcells.db"

set link_library [concat \
    * \
    $LIBS_DIR/stdcells.db \
    $LIBS_DIR/sram_64x512_1rw.db \
    $LIBS_DIR/sram_256x512_1rw.db \
    $LIBS_DIR/sram_1024x32_1rw.db \
    $LIBS_DIR/sram_64x24_1r1w.db \
    $LIBS_DIR/sram_256x24_1r1w.db \
    $LIBS_DIR/sram_64x128_1r1w.db \
    $LIBS_DIR/sram_128x128_1r1w.db \
]

# Bottom-up: link against VX_socket_top.db when blackboxing (see
# dc_syn.tcl for the rationale and the two-tier path lookup).
if {$BLACKBOX_SOCKET} {
    set _socket_db_pref [file normalize "$SCRIPT_DIR/../../pnr/cadence/export/single-core/VX_socket_top.db"]
    set _socket_db_fall "$LIBS_DIR/VX_socket_top.db"
    if {[file exists $_socket_db_pref]} {
        set _socket_db $_socket_db_pref
    } elseif {[file exists $_socket_db_fall]} {
        set _socket_db $_socket_db_fall
    } else {
        error "BLACKBOX_SOCKET=1 but VX_socket_top.db was not found in either:\n  $_socket_db_pref\n  $_socket_db_fall\nRun 'make socket-db' (or 'make full-vortex' via the Makefile) to build it from the .lib, and if the .lib itself is missing, run 'make extract-macro' under hw/pnr/cadence first."
    }
    lappend link_library $_socket_db
    puts "INFO: BLACKBOX_SOCKET=1 — VX_socket_top linked from: $_socket_db"
} else {
    puts "INFO: BLACKBOX_SOCKET=0 — VX_socket_top will be elaborated from RTL (flat)."
}

set symbol_library {}

#-----------------------------------------------------------------------
# Parse filelist
#-----------------------------------------------------------------------
source "$SCRIPT_DIR/../../scripts/parse_vcs_list.tcl"

set flist_path "$SCRIPT_DIR/flists/dc_flist_${CONFIG}.f"
if {![file exists $flist_path]} {
    error "Filelist not found: $flist_path\nRun: make flist_${CONFIG}"
}

lassign [parse_vcs_list $flist_path] src_files inc_dirs defines

set search_path [concat $search_path $inc_dirs]

# defs_div_sqrt_mvp.sv defines a package but does not follow the *_pkg.sv
# naming convention, so gen_sources.sh cannot hoist it automatically.
# Move it to the front of the file list so it is analyzed before any file
# that imports the defs_div_sqrt_mvp package.
set _defs_idx [lsearch -glob $src_files "*defs_div_sqrt_mvp.sv"]
if {$_defs_idx >= 0} {
    set _defs_file [lindex $src_files $_defs_idx]
    set src_files [linsert [lreplace $src_files $_defs_idx $_defs_idx] 0 $_defs_file]
    unset _defs_idx _defs_file
}

# Bottom-up: drop VX_socket_top.sv so link resolves the reference to
# the blackbox cell in link_library (see dc_syn.tcl for the rationale).
if {$BLACKBOX_SOCKET} {
    set _pre [llength $src_files]
    set src_files [lsearch -inline -all -not -glob $src_files "*/VX_socket_top.sv"]
    set _post [llength $src_files]
    if {$_pre == $_post} {
        puts "WARNING: BLACKBOX_SOCKET=1 but VX_socket_top.sv was not in the filelist; nothing to skip."
    } else {
        puts "INFO: BLACKBOX_SOCKET=1 — skipped VX_socket_top.sv from analyze ([expr {$_pre - $_post}] file(s))."
    }
    unset _pre _post
}

#-----------------------------------------------------------------------
# Analyze
#-----------------------------------------------------------------------
foreach f $src_files {
    analyze -format sverilog -define $defines $f
}

#-----------------------------------------------------------------------
# Elaborate, link, and check — stop here (no compile)
#-----------------------------------------------------------------------
elaborate $TOP
current_design $TOP
link
check_design

# Bottom-up: assert the socket linked cleanly. Don't bother with
# set_dont_touch here — elab is read-only and never runs compile_ultra.
if {$BLACKBOX_SOCKET} {
    set _sockets [get_cells -hier -filter "ref_name == VX_socket_top"]
    set _n_sockets [sizeof_collection $_sockets]
    if {$_n_sockets == 0} {
        error "BLACKBOX_SOCKET=1 but no VX_socket_top cells appear in the linked design. Check link_library and ASIC_SYNTHESIS defines."
    }
    puts "INFO: found $_n_sockets VX_socket_top blackbox instance(s) in linked design."
    unset _sockets _n_sockets
}

report_hierarchy

exit
