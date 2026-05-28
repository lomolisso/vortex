# Restore a saved checkpoint in GUI mode. Sourced by open_checkpoint.sh,
# which sets ::env(CHECKPOINT) to the stage name (without "after_" prefix).

if {![info exists ::env(CHECKPOINT)]} { error "CHECKPOINT not set (use open_checkpoint.sh)." }

set _run_dir    [pwd]
set _ckpt_stage $::env(CHECKPOINT)
set _ckpt       "$_run_dir/checkpoints/after_${_ckpt_stage}"

if {![file exists $_ckpt]} {
    set _avail [list]
    foreach f [lsort [glob -nocomplain -type f "$_run_dir/checkpoints/after_*"]] {
        lappend _avail "  [regsub {^after_} [file tail $f] {}]"
    }
    error "Checkpoint not found: $_ckpt\n\nAvailable:\n[join $_avail \n]"
}

# 00_config defines _apply_design_modes and resolves TOML globals (TOP_RLAYER,
# DESIGN_KIND, ...) that the restored settings depend on.
set SCRIPT_DIR [file normalize "[file dirname [info script]]/../.."]
source "$SCRIPT_DIR/scripts/00_config.tcl"

puts "\n=== Restoring checkpoint: after_${_ckpt_stage} ==="
source $_ckpt
_apply_design_modes

puts "=== Ready. Innovus GUI is active. ==="
gui_show
