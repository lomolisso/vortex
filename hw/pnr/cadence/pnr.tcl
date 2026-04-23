#=======================================================================
# Vortex GPGPU — Cadence Innovus PnR Stage Runner
# Target: NanGate45 stdcells + bsg_fakeram SRAM macros
#
# Runs exactly one stage of the PnR flow.  Invoked by the Makefile:
#   make <config> STAGE=<stage_name>
#
# What this script does:
#   1. Sources 00_config.tcl to set up all path variables.
#   2. Loads the checkpoint saved by the immediately preceding stage.
#      Fails with a clear error if that checkpoint does not exist.
#   3. Sources the requested stage script (scripts/<stage>.tcl).
#   4. Saves a new checkpoint: checkpoints/after_<stage>.
#   5. Returns — Innovus GUI stays open for inspection.
#
# Stage order:
#   01_init_design     02_power_connect   03_floorplan
#   04_power_plan      05_placement       06_pre_cts_opt
#   07_cts             08_post_cts_opt    09_routing
#   10_post_route_opt  11_reports         12_outputs
#
# Required environment variables (set by the Makefile):
#   CONFIG   — design config, e.g. 1c1n4w4t
#   LIBS_DIR — absolute path to hw/syn/synopsys/libs/
#   SYN_DIR  — absolute path to hw/syn/synopsys/
#   STAGE    — stage to run, e.g. 03_floorplan
#=======================================================================

set SCRIPT_DIR [file normalize [file dirname [info script]]]

# Ordered stage list — index position drives checkpoint lookup.
set stages {
    01_init_design
    02_power_connect
    03_floorplan
    04_power_plan
    05_placement
    06_pre_cts_opt
    07_cts
    08_post_cts_opt
    09_routing
    10_post_route_opt
    11_reports
    12_outputs
}

# --- Validate STAGE ---
if {![info exists ::env(STAGE)]} {
    error "STAGE is not set.\nUse:  make <config> STAGE=<number>  e.g. STAGE=3"
}
set STAGE $::env(STAGE)

# Accept a bare number (3 or 03) in addition to the full name (03_floorplan).
if {[string first "_" $STAGE] < 0} {
    set _num [format "%02d" [expr {int($STAGE)}]]
    set _match ""
    foreach s $stages { if {[string match "${_num}_*" $s]} { set _match $s; break } }
    if {$_match eq ""} {
        error "No stage found for number '$STAGE'.\nValid stages: $stages"
    }
    set STAGE $_match
}

set _stage_idx [lsearch -exact $stages $STAGE]
if {$_stage_idx < 0} {
    error "STAGE='$STAGE' is not valid.\nValid stages: $stages"
}

# --- Stage 0: config / path setup (always runs first) ---
puts "\n[string repeat = 60]"
puts " Setup: 00_config"
puts "[string repeat = 60]"
source "$SCRIPT_DIR/scripts/00_config.tcl"

# --- Load preceding checkpoint (all stages except 01_init_design) ---
# 01_init_design (index 0) is the first real stage — it loads from the
# synthesis netlist directly, so no checkpoint exists yet.
if {$_stage_idx > 0} {
    set _prev_stage [lindex $stages [expr {$_stage_idx - 1}]]
    set _ckpt       "$RUN_DIR/checkpoints/after_${_prev_stage}"

    if {![file exists $_ckpt]} {
        error "Checkpoint not found: $_ckpt\nRun stage '${_prev_stage}' first."
    }

    puts "\n[string repeat = 60]"
    puts " Loading checkpoint: after_${_prev_stage}"
    puts "[string repeat = 60]"
    source $_ckpt
    puts "--- Checkpoint loaded ---"

    # Re-apply Tcl settings that are not stored inside the checkpoint database.
    # These affect extraction accuracy and delay-calc for all later stages.
    setDesignMode -process 45
    set delaycal_use_default_delay_limit 1000

    # Re-source 02_power_connect to restore the connectGlobalNets procedure.
    # It is called again by 05_placement and 09_routing after inserting new
    # cells; without it those cells would have floating power pins.
    if {$_stage_idx > [lsearch -exact $stages 02_power_connect]} {
        source "$SCRIPT_DIR/scripts/02_power_connect.tcl"
    }
}

# --- Run the stage ---
# Prefer a config-specific override (scripts/<STAGE>_<CONFIG>.tcl) when it
# exists, so stages that need hand-tuned geometry (floorplan) can live as a
# separate file per config while the generic stages are shared.
set _stage_file "$SCRIPT_DIR/scripts/${STAGE}.tcl"
if {[info exists CONFIG]} {
    set _cfg_stage_file "$SCRIPT_DIR/scripts/${STAGE}_${CONFIG}.tcl"
    if {[file exists $_cfg_stage_file]} {
        set _stage_file $_cfg_stage_file
    }
}
puts "\n[string repeat = 60]"
puts " Stage: $STAGE (file: [file tail $_stage_file])"
puts "[string repeat = 60]"
source $_stage_file

# --- Save checkpoint ---
set _ckpt_dir  "$RUN_DIR/checkpoints"
set _ckpt_file "$_ckpt_dir/after_${STAGE}"
file mkdir $_ckpt_dir
puts "\n--- Saving checkpoint: $_ckpt_file ---"
saveDesign $_ckpt_file
puts "--- Checkpoint saved ---"

puts "\n[string repeat = 60]"
puts " Stage '$STAGE' complete."
puts " Innovus GUI is active — inspect the design, then close when done."
puts "[string repeat = 60]\n"
