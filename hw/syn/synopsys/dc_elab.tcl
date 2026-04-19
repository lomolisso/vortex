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
#   dc_shell -x "set CONFIG 1c1n4w4t" -f dc_elab.tcl 2>&1 | tee logs/1c1n4w4t_elab.log
#   grep -E "Error|Warning|unresolved" logs/1c1n4w4t_elab.log
#=======================================================================

if {![info exists CONFIG]} {
    error "CONFIG is not set. Invoke as: dc_shell -x {set CONFIG <config_name>} -f dc_elab.tcl"
}

set SCRIPT_DIR [file normalize [file dirname [info script]]]

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

set symbol_library {}

#-----------------------------------------------------------------------
# Parse filelist
#-----------------------------------------------------------------------
source "$SCRIPT_DIR/../../scripts/parse_vcs_list.tcl"

set flist_path "$SCRIPT_DIR/dc_flist_${CONFIG}.f"
if {![file exists $flist_path]} {
    error "Filelist not found: $flist_path\nRun: make dc_flist_${CONFIG}.f"
}

lassign [parse_vcs_list $flist_path] src_files inc_dirs defines

set search_path [concat $search_path $inc_dirs]

#-----------------------------------------------------------------------
# Analyze
#-----------------------------------------------------------------------
foreach f $src_files {
    analyze -format sverilog -define $defines $f
}

#-----------------------------------------------------------------------
# Elaborate, link, and check — stop here (no compile)
#-----------------------------------------------------------------------
elaborate Vortex
current_design Vortex
link
check_design
report_hierarchy

exit
