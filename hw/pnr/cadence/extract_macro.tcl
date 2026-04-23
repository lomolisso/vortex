#=======================================================================
# Vortex GPGPU — Hard-Macro Extraction Entry Point
#
# Packages the fully-routed single-core database into the physical
# (.lef) and timing (.lib) abstract files that the full-vortex top
# level instantiates as a hard macro.
#
# Unlike pnr.tcl (which is staged and expected to be re-entered at any
# STAGE), this script runs the three export sub-stages in one shot:
#
#     00_config   → set up paths, validate env
#     load ckpt   → runs/single-core/checkpoints/after_12_outputs
#     export_prep → pin snap + on-grid + M9/M10 reservation check
#     export_lef  → write_lef_macromodel   (→ export/<config>/<TOP>.lef)
#     export_etm  → do_extract_model       (→ export/<config>/<TOP>.lib)
#
# Required environment variables (set by the Makefile's extract-macro
# target, can also be set by hand in an interactive session):
#   CONFIG   — must be "single-core" (the only config that produces
#              a hard macro today).
#   TOP      — top-level cell of the macro (VX_socket_top).
#   LIBS_DIR — absolute path to hw/syn/synopsys/libs/.
#   SYN_DIR  — absolute path to hw/syn/synopsys/.
#
# Optional environment variables (see scripts/export_etm.tcl for
# details):
#   VIEW_NAME   — MMMC analysis view(s) to extract; default "typView".
#   MACRO_CELL  — .lib cell name; defaults to $TOP.
#=======================================================================

set SCRIPT_DIR [file normalize [file dirname [info script]]]

# --- Step 1: config / path setup (no Innovus commands run here). ---
puts "\n[string repeat = 60]"
puts " Extract: 00_config"
puts "[string repeat = 60]"
source "$SCRIPT_DIR/scripts/00_config.tcl"

# --- Extraction is only defined for the single-core config today. ---
# The full-vortex top is the *consumer* of the exported macro, not a
# source of further abstraction.  Fail early if someone invokes this
# script with the wrong CONFIG.
if {$CONFIG ne "single-core"} {
    error "extract_macro.tcl only supports CONFIG=single-core (got '$CONFIG').\nThis target packages the single-core hard macro; the full-vortex config consumes it."
}

# --- Step 2: load the final routed-and-finalised checkpoint. ---
# after_12_outputs is used instead of after_10_post_route_opt because
# stage 12 carries the canonical post-extraction setup (final netlist,
# DEF, full session) that the rest of the flow treats as "done".  It
# is the only checkpoint a user can open with open_checkpoint.sh 12.
set _ckpt "$RUN_DIR/checkpoints/after_12_outputs"
if {![file exists $_ckpt]} {
    error "Final single-core checkpoint not found: $_ckpt\nRun 'make single-core STAGE=12' first (after re-running STAGE=9 with the updated M8 routing cap)."
}

puts "\n[string repeat = 60]"
puts " Loading checkpoint: after_12_outputs"
puts "[string repeat = 60]"
source $_ckpt
puts "--- Checkpoint loaded ---"

# --- Step 3: export sub-stages, in order. ---
# Each sub-stage is a separate Tcl file so the user can iterate on a
# single pass (e.g. tweak write_lef_macromodel flags) without re-running
# the others. The wrapper just sources them in the right order.
foreach sub {export_prep export_lef export_etm} {
    set path "$SCRIPT_DIR/scripts/${sub}.tcl"
    puts "\n[string repeat = 60]"
    puts " Export: $sub"
    puts "[string repeat = 60]"
    source $path
}

# --- Step 4: save a checkpoint of the extracted state. ---
# Gives the user a re-entry point for the post-extraction database
# (useful for debugging: open the .lef side-by-side with the routed DB
# to diff obstruction boundaries).
set _export_ckpt "$RUN_DIR/checkpoints/after_13_export_macro"
puts "\n--- Saving checkpoint: $_export_ckpt ---"
saveDesign $_export_ckpt
puts "--- Checkpoint saved ---"

puts "\n[string repeat = 60]"
puts " Hard-macro extraction complete."
puts " Artifacts:"
puts "   LEF : $SCRIPT_DIR/export/$CONFIG/${TOP}.lef"
puts "   LIB : $SCRIPT_DIR/export/$CONFIG/${TOP}.lib"
puts " The full-vortex config (01_init_design.tcl, mmmc.tcl) will"
puts " pick these up automatically on its next run."
puts "[string repeat = 60]\n"
