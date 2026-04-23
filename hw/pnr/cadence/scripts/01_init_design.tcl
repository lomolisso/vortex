#=======================================================================
# Stage 1 — Design Initialization
#
# PURPOSE
#   Load the synthesized gate-level netlist, all physical libraries (LEF),
#   the timing libraries (via MMMC), and establish the top-level power
#   net names.  After this stage Innovus has a complete, un-placed design
#   in memory — all cells exist but none have physical coordinates yet.
#
# WHAT IS MMMC?
#   Multi-Mode Multi-Corner (MMMC) is Innovus's framework for analyzing
#   timing under different operating conditions simultaneously.  Even with
#   a single "typical" corner (as used here), you must define at least one
#   analysis view so Innovus knows which timing library and SDC constraint
#   set to use throughout the rest of the flow.
#   The MMMC setup lives in mmmc.tcl (one directory above scripts/).
#
# WHAT IS A LEF FILE?
#   Library Exchange Format.  Contains the physical footprint of each cell:
#   - The cell boundary (height × width in µm)
#   - Pin locations and metal layers
#   - Routing blockage layers inside the cell
#   Innovus needs LEFs for every cell in the netlist so it can legally
#   place and route them on the chip.  The timing (.lib) and physical (.lef)
#   information are deliberately separate so that the same cell can appear
#   in multiple process corners.
#
# INPUTS  (Tcl globals from 00_config.tcl)
#   SCRIPT_DIR, LIBS_DIR, NETLIST, SDC_FILE
#
# KNOWN WARNINGS FROM THIS STAGE
#   TECHLIB-9153  (×7)  — duplicate 'default_input_pin_cap' in each SRAM .lib.
#                         Harmless; the last definition wins (Innovus default).
#                         Cannot be eliminated without regenerating the bsg_fakeram
#                         .lib files via the memory compiler.
#   TECHLIB-302   (×7)  — no Boolean function defined for SRAM cells.
#                         Expected: SRAM cells are black-boxes used for timing
#                         analysis only, not logic simulation.
#   IMPVL-159     (×14) — VDD/VSS pins exist in LEF but not in the timing lib.
#                         Harmless; Innovus connects them via globalNetConnect
#                         in the next stage rather than through the liberty model.
#   IMPTS-17      (×6)  — capacitance unit inconsistency across .lib files.
#                         Root cause is in the auto-generated SRAM .lib files.
#                         setDesignMode -process 45 (added below) eliminates the
#                         most harmful downstream impact by setting process-node
#                         defaults.
#   TCLCMD-1461   (×1)  — RESOLVED by SDC filter in 00_config.tcl.
#                         The 'set_units' line is removed before Innovus reads
#                         the SDC.
#   TCLNL-330     (×1)  — RESOLVED by SDC filter in 00_config.tcl.
#                         'set_input_delay' on the clock root port is replaced
#                         with 'set_clock_latency -source 2 [get_clocks clk]'.
#   IMPEXT-3530   (×11) — RESOLVED by setDesignMode -process 45 below.
#   IMPDC-1629    (×2)  — RESOLVED by set delaycal_use_default_delay_limit 1000.
#   IMPEXT-2766   (×10) — sheet resistance missing from cap table; LEF fallback
#                         values are used.  Cannot be fixed without a PDK-specific
#                         cap table file.  Parasitic accuracy is acceptable for
#                         this academic project.
#   IMPEXT-2773   (×10) — via resistance missing in LEF for M0–M10; defaulting
#                         to 4 Ω per via.  Same root cause as IMPEXT-2766.
#=======================================================================

# --- Point Innovus to the MMMC configuration file ---
# Innovus reads this path from the init_mmmc_file global variable before
# init_design is called.  The file itself is mmmc.tcl, which creates the
# library set, constraint mode, delay corner, and analysis view objects.
set init_mmmc_file "$SCRIPT_DIR/mmmc.tcl"

# --- Synthesized gate-level netlist (Verilog) ---
# This is the output of Design Compiler (DC).  It contains only standard
# cells and macro instances by name — no geometry yet.
set init_verilog $NETLIST

# --- Physical libraries (LEF) ---
# Order matters: the technology LEF must come first, then the stdcell LEF,
# then any macro LEFs.  Innovus processes them in order and the tech LEF
# defines the routing layers that all subsequent LEFs reference.
#
# All 7 SRAM variants are listed unconditionally.  Innovus silently ignores
# LEFs for cells that do not appear in the loaded netlist, so this list is
# config-agnostic across all Vortex configurations (single-core, full-vortex).
set init_lef_file [list \
    $LIBS_DIR/rtk-tech.lef \
    $LIBS_DIR/stdcells.lef \
    $LIBS_DIR/sram_64x512_1rw.lef \
    $LIBS_DIR/sram_256x512_1rw.lef \
    $LIBS_DIR/sram_1024x32_1rw.lef \
    $LIBS_DIR/sram_64x24_1r1w.lef \
    $LIBS_DIR/sram_256x24_1r1w.lef \
    $LIBS_DIR/sram_64x128_1r1w.lef \
    $LIBS_DIR/sram_128x128_1r1w.lef \
]

# For the full-vortex config, the cores are instantiated as blackbox
# macros whose physical footprint lives in VX_socket_top.lef (produced
# by the single-core PnR + LEF extraction step: `make extract-macro`).
# Append it here so Innovus has geometry for the four VX_socket_top
# instances in the Vortex netlist.
#
# Lookup order:
#   1. pnr/cadence/export/single-core/VX_socket_top.lef
#        — canonical output of `make extract-macro`. Preferred because
#          it lives in the PnR flow's own export tree and is regenerated
#          automatically every time the single-core run is re-exported.
#   2. $LIBS_DIR/VX_socket_top.lef
#        — fallback for the historical layout where the core macro was
#          hand-copied into hw/syn/synopsys/libs/ alongside the SRAM LEFs.
#          Kept to avoid breaking existing workflows.
if {$CONFIG eq "full-vortex"} {
    set _core_lef "$SCRIPT_DIR/export/single-core/VX_socket_top.lef"
    if {![file exists $_core_lef]} {
        set _core_lef "$LIBS_DIR/VX_socket_top.lef"
    }
    if {![file exists $_core_lef]} {
        error "VX_socket_top.lef not found in either\n  $SCRIPT_DIR/export/single-core/ (preferred)\n  $LIBS_DIR/ (fallback)\nRun 'make extract-macro' after a completed single-core PnR run."
    }
    lappend init_lef_file $_core_lef
    puts "INFO: full-vortex — loaded core macro LEF: $_core_lef"
}

# --- Top-level cell name and power/ground net names ---
# init_top_cell must match the module name in the synthesized Verilog.
# TOP is set by 00_config.tcl (VX_socket_top for single-core, Vortex for
# full-vortex). init_pwr_net / init_gnd_net tell Innovus which nets are
# power/ground so it can build the internal power domain model.
set init_top_cell $TOP
set init_pwr_net  VDD
set init_gnd_net  VSS

# --- Perform initialization ---
# This is the "big bang" command: Innovus reads the netlist, parses all
# LEFs, loads the MMMC views, and constructs an internal design database.
# Expect several seconds to a few minutes depending on design size.
init_design

# --- Post-init settings that require the database to be loaded ---

# Set the process technology node.
# Innovus uses this to select optimal threshold voltages and capacitance
# models for extraction.  Without it, Innovus falls back to generic
# internal defaults, producing IMPEXT-3530 ("process node not set")
# and IMPEXT-6197 ("cap table not specified") warnings on every
# extraction run.  Setting it to 45 (nm) matches the FreePDK45 PDK.
setDesignMode -process 45

# Restore the delay-calculation path limit to the Innovus default of 1000.
# Innovus sometimes starts with an internal limit of 101 (below the 1000
# default), which triggers IMPDC-1629 and can cause inaccurate delay
# computation for high-fanout nets like the clock network.  Explicitly
# resetting it to 1000 ensures correct behaviour throughout all later
# optimization passes.
set delaycal_use_default_delay_limit 1000
