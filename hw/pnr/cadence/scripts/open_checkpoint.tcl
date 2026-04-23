#=======================================================================
# open_checkpoint.tcl — Restore a saved Innovus checkpoint in GUI mode
#
# This script is sourced by open_checkpoint.sh via:
#   innovus -files scripts/open_checkpoint.tcl
#
# open_checkpoint.sh cd's into runs/<config>/ before launching Innovus,
# so [pwd] resolves to that directory and the checkpoint paths are correct.
#
# Required environment variable (set by open_checkpoint.sh):
#   CHECKPOINT  — stage name without the "after_" prefix, e.g. 03_floorplan
#
# After this script finishes, Innovus drops into its interactive GUI with
# the full design database loaded.  Use the standard Innovus GUI controls
# to inspect placement, routing, floorplan, timing, etc.
#=======================================================================

set _run_dir [pwd]

# --- Read checkpoint stage from the environment ---
if {![info exists ::env(CHECKPOINT)]} {
    error "CHECKPOINT environment variable is not set.\nRun this script via open_checkpoint.sh."
}
set _ckpt_stage $::env(CHECKPOINT)
set _ckpt       "$_run_dir/checkpoints/after_${_ckpt_stage}"

if {![file exists $_ckpt]} {
    # Build a helpful list of what is available
    set _available [list]
    foreach f [lsort [glob -nocomplain -type f "$_run_dir/checkpoints/after_*"]] {
        lappend _available "  [regsub {^after_} [file tail $f] {}]"
    }
    error "Checkpoint not found: $_ckpt\n\nAvailable checkpoints:\n[join $_available \n]"
}

puts "\n[string repeat = 60]"
puts " Restoring checkpoint: after_${_ckpt_stage}"
puts " Run dir : $_run_dir"
puts "[string repeat = 60]"

# Sourcing the checkpoint header file is the correct, version-agnostic way
# to call restoreDesign.  The header contains a self-referential Tcl snippet
# that passes the matching .dat directory to restoreDesign automatically.
source $_ckpt

puts "--- Checkpoint loaded ---"

# --- Re-apply Tcl-only settings not stored in the checkpoint database ---
# These are safe to set at any stage; they affect extraction and delay-calc
# if you continue the flow interactively from the GUI.
setDesignMode -process 45
set delaycal_use_default_delay_limit 1000

puts "\n[string repeat = 60]"
puts " Ready.  Checkpoint: after_${_ckpt_stage}"
puts " Innovus GUI is now active — explore the design freely."
puts "[string repeat = 60]\n"
