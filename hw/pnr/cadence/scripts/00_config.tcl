#=======================================================================
# Stage 0 — Configuration & Path Setup
#
# PURPOSE
#   Resolve all environment variables and derive every file path that
#   later stages will need.  This is the only stage that reads from the
#   outside world (env vars set by the Makefile); every other stage just
#   uses the Tcl variables established here.
#
# INPUTS  (must be set in the shell / Makefile before Innovus starts)
#   CONFIG    — short design identifier, e.g. "single-core" or "full-vortex".
#               Used to locate the matching synthesis results directory.
#   TOP       — (optional) name of the top-level cell elaborated by DC.
#               Defaults to "Vortex" for backward compatibility; the
#               Makefile sets it to VX_socket_top for the single-core
#               config. Drives the <TOP>_netlist.v / <TOP>.sdc filenames.
#   LIBS_DIR  — absolute path to hw/syn/synopsys/libs/
#               Contains the NanGate45 stdcell library and all bsg_fakeram
#               SRAM liberty (.lib) and LEF files.
#   SYN_DIR   — absolute path to hw/syn/synopsys/
#               The script derives SYN_RESULTS as $SYN_DIR/runs/$CONFIG/results
#   EFFORT    — (optional) optimization effort level: low | medium | high
#               Default: medium.  Controls all optDesign, ccopt_design, and
#               routeDesign calls in later stages.  Set via the Makefile
#               EFFORT variable; override on the command line with
#               'make <config> EFFORT=low'.
#
# OUTPUTS (Tcl globals used by every subsequent stage)
#   CONFIG, LIBS_DIR, SYN_DIR   — validated, kept as-is
#   TOP                         — top-level cell name (Vortex or VX_socket_top)
#   SCRIPT_DIR                  — directory containing pnr.tcl and scripts/
#   RUN_DIR                     — Innovus working directory (where pnr.tcl was
#                                  invoked from; each config gets its own subdir)
#   SYN_RESULTS                 — path to DC synthesis output directory
#   REPORT_DIR                  — $RUN_DIR/reports  (created here)
#   RESULTS_DIR                 — $RUN_DIR/results  (created here)
#   NETLIST                     — $SYN_RESULTS/${TOP}_netlist.v
#   SDC_FILE                    — Path to the filtered SDC (see below)
#   OPT_EFFORT                  — effort level string forwarded to all
#                                  optimization commands in later stages
#
# SDC FILTERING
#   The SDC produced by Design Compiler contains two constructs that
#   Innovus does not support:
#
#   1. set_units (TCLCMD-1461) — Innovus skips this command entirely.
#      It does not affect results, but the warning is confusing.
#
#   2. set_input_delay -clock clk -max 2 [get_ports clk] (TCLNL-330) —
#      Applying an input delay to the clock root port itself is not
#      meaningful in Innovus.  The correct representation is:
#        set_clock_latency -source 2 [get_clocks clk]
#      which models the insertion delay from the board/package to the
#      chip's clock input pin.
#
#   This stage reads the DC-generated SDC, rewrites it with both
#   constructs corrected, and saves the result as ${TOP}_filtered.sdc in
#   $RUN_DIR.  All subsequent stages use the filtered file.  The filter
#   runs fresh every time so it survives synthesis re-runs automatically.
#
# KNOWN WARNINGS FROM THIS STAGE
#   None — this stage does not call any Innovus commands; it is pure Tcl.
#=======================================================================

# --- Read and validate required environment variables ---
# "info exists ::env(VAR)" checks if the env var is set in the shell.
# If it is, copy it into a Tcl variable.  If not, check whether the
# variable was already defined (e.g. during an interactive session).
# If neither, abort with a helpful message.
foreach {var} {CONFIG LIBS_DIR SYN_DIR} {
    if {[info exists ::env($var)]} {
        set $var $::env($var)
    } elseif {![info exists $var]} {
        error "$var is not set. Use the Makefile or set the environment variable before invoking Innovus."
    }
}

# TOP is optional — default to "Vortex" so existing invocations that
# don't set it still work (they predate the single-core / full-vortex
# split and all used the Vortex top cell).
if {[info exists ::env(TOP)]} {
    set TOP $::env(TOP)
} elseif {![info exists TOP]} {
    set TOP Vortex
}

# --- Read the optional EFFORT variable ---
# Default to "medium" so the script works when Innovus is invoked manually
# without the Makefile (which sets EFFORT=low by default).
# Valid values: low | medium | high
if {[info exists ::env(EFFORT)]} {
    set OPT_EFFORT $::env(EFFORT)
} elseif {![info exists OPT_EFFORT]} {
    set OPT_EFFORT medium
}
puts "INFO: OPT_EFFORT = $OPT_EFFORT  (low = fast/academic, medium = balanced, high = near-tapeout)"

# --- Derive all paths from the three root variables ---
# SCRIPT_DIR is the directory where pnr.tcl lives (set in pnr.tcl before
# any stage is sourced, but re-confirmed here for documentation clarity).
# RUN_DIR is the current working directory; the Makefile cd's into
# runs/<config>/ before launching Innovus, so this points there.
set RUN_DIR     [pwd]
set SYN_RESULTS "$SYN_DIR/runs/$CONFIG/results"
set REPORT_DIR  "$RUN_DIR/reports"
set RESULTS_DIR "$RUN_DIR/results"
set NETLIST     "$SYN_RESULTS/${TOP}_netlist.v"
set SDC_FILE    "$SYN_RESULTS/${TOP}.sdc"

# --- Guard: make sure synthesis was actually run first ---
# Innovus will crash later with a confusing message if the netlist is
# missing; catching it here gives a clear action item.
if {![file exists $NETLIST]} {
    error "Synthesized netlist not found: $NETLIST\nRun 'make $CONFIG' in hw/syn/synopsys first (top: $TOP)."
}
if {![file exists $SDC_FILE]} {
    error "SDC not found: $SDC_FILE\nRe-run synthesis to regenerate it."
}

# --- Create output directories upfront ---
# "file mkdir" is idempotent (no error if the directory already exists).
file mkdir $REPORT_DIR $RESULTS_DIR

# --- Filter the DC-generated SDC ---
# Creates $RUN_DIR/Vortex_filtered.sdc with two corrections applied:
#
#   1. Remove  set_units ...
#      Innovus does not support this SDC command (TCLCMD-1461).  The
#      command is skipped silently by DC-aware tools but produces a
#      confusing warning in Innovus.  Removing it has no functional
#      effect because Innovus derives its internal units from the
#      technology LEF, not from the SDC.
#
#   2. Rewrite  set_input_delay -clock clk -max 2 [get_ports clk]
#              → set_clock_latency -source 2 [get_clocks clk]
#      Innovus does not allow set_input_delay on the clock root port
#      (TCLNL-330).  The semantically equivalent construct is
#      set_clock_latency -source, which models the board/package
#      insertion delay from the off-chip clock source to the die.
#      The 2 ns value from the DC SDC is preserved.
#
set filtered_sdc "$RUN_DIR/${TOP}_filtered.sdc"
set raw_fh   [open $SDC_FILE r]
set filt_fh  [open $filtered_sdc w]
set skipped  0
set replaced 0
while {[gets $raw_fh line] >= 0} {
    # Drop the set_units line completely.
    if {[regexp {^\s*set_units\s} $line]} {
        incr skipped
        continue
    }
    # Replace set_input_delay on the clock root port with set_clock_latency.
    if {[regexp {set_input_delay\s.*\[get_ports clk\]} $line]} {
        puts $filt_fh "set_clock_latency -source 2 \[get_clocks clk\]  ;# rewritten from set_input_delay on clk port"
        incr replaced
        continue
    }
    puts $filt_fh $line
}
close $raw_fh
close $filt_fh
puts "INFO: SDC filter — removed $skipped set_units line(s), rewrote $replaced set_input_delay-on-clk line(s)."
puts "      Filtered SDC: $filtered_sdc"

# Point SDC_FILE at the filtered copy.  All later stages source this variable.
set SDC_FILE $filtered_sdc

# --- Print a run banner so the log is easy to grep ---
puts "============================================================"
puts " Vortex PnR  |  config : $CONFIG   top : $TOP"
puts " Netlist     : $NETLIST"
puts " SDC         : $SDC_FILE"
puts " Libs        : $LIBS_DIR"
puts " Run dir     : $RUN_DIR"
puts " Effort      : $OPT_EFFORT"
puts "============================================================"
