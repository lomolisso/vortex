#=======================================================================
# Vortex GPGPU — Cadence Innovus MMMC (Multi-Mode Multi-Corner) Setup
#
# This file is sourced by Innovus during init_design (via set init_mmmc_file).
# The following variables must be set in the enclosing pnr.tcl scope
# BEFORE init_design is called, as they are resolved here:
#
#   CONFIG    — "single-core" or "full-vortex" (from 00_config.tcl)
#   LIBS_DIR  — absolute path to hw/syn/synopsys/libs/
#   SDC_FILE  — absolute path to the per-config <TOP>_filtered.sdc
#
# All 7 SRAM library variants are listed unconditionally; Innovus ignores
# libraries whose cells are absent from the loaded netlist, so this file
# remains config-agnostic across all Vortex configurations.
#
# For the full-vortex config, the four VX_core blocks are instantiated
# as blackbox macros whose timing model lives in VX_socket_top.lib
# (produced by the earlier single-core PnR + LEF/LIB extraction step).
# It is appended to the library set below only when $CONFIG == full-vortex.
#=======================================================================

set _lib_list [list \
    $LIBS_DIR/stdcells.lib \
    $LIBS_DIR/sram_64x512_1rw.lib \
    $LIBS_DIR/sram_256x512_1rw.lib \
    $LIBS_DIR/sram_1024x32_1rw.lib \
    $LIBS_DIR/sram_64x24_1r1w.lib \
    $LIBS_DIR/sram_256x24_1r1w.lib \
    $LIBS_DIR/sram_64x128_1r1w.lib \
    $LIBS_DIR/sram_128x128_1r1w.lib \
]

# Lookup order mirrors 01_init_design.tcl's handling of the matching
# VX_socket_top.lef:
#   1. pnr/cadence/export/single-core/VX_socket_top.lib  (preferred;
#      canonical output of `make extract-macro`)
#   2. $LIBS_DIR/VX_socket_top.lib                       (fallback for
#      hand-copied installs)
if {$CONFIG eq "full-vortex"} {
    set _core_lib "$SCRIPT_DIR/export/single-core/VX_socket_top.lib"
    if {![file exists $_core_lib]} {
        set _core_lib "$LIBS_DIR/VX_socket_top.lib"
    }
    if {![file exists $_core_lib]} {
        error "VX_socket_top.lib not found in either\n  $SCRIPT_DIR/export/single-core/ (preferred)\n  $LIBS_DIR/ (fallback)\nRun 'make extract-macro' after a completed single-core PnR run."
    }
    lappend _lib_list $_core_lib
    puts "INFO: full-vortex — loaded core macro LIB: $_core_lib"
}

create_library_set -name typical -timing $_lib_list

create_constraint_mode -name func \
    -sdc_files [list $SDC_FILE]

create_delay_corner -name typDC \
    -library_set typical

create_analysis_view -name typView \
    -constraint_mode func \
    -delay_corner typDC

set_analysis_view -setup {typView} -hold {typView}
