#=======================================================================
# Export ETM — extracted timing model (.lib) for the routed single-core
#
# PURPOSE
#   Package the routed single-core database into a Liberty timing model
#   the full-vortex top-level can include as a regular timing library
#   (alongside NanGate45 stdcells and the bsg_fakeram SRAMs).  The model
#   condenses every internal timing arc of VX_socket_top down to pin-
#   to-pin delay/transition tables, so Innovus never needs to re-open
#   the macro to compute timing at the top level.
#
# COMMAND DISCOVERY
#   Innovus 21.19 ships the ETM extractor as `do_extract_model`
#   (documented in $INNOVUS211/doc/innovusTCR/do_extract_model.html).
#   Older Innovus releases and some Tempus-only bundles expose it
#   instead as `extract_model`, so we probe for both at runtime and use
#   whichever is available. This keeps the flow portable across course
#   machines that may or may not have the Tempus signoff bundle
#   enabled.
#
# COMMAND SYNTAX (21.19 TCR, verbatim):
#   do_extract_model <model_filename>
#     [-cell_name <cell>] [-lib_name <lib>] [-view <view>] [-pg]
#
#   i.e. the OUTPUT FILE is a POSITIONAL argument (not -output_directory)
#   and the view flag is singular `-view` (not -view_name).  The whole
#   -output_directory / -view_name pattern from earlier tutorials is
#   from a later release and is rejected by 21.19.
#
# PARAMETERS (env overrides, with sensible defaults)
#   VIEW_NAME   — MMMC analysis view to extract from. Defaults to the
#                 single view created in mmmc.tcl ("typView"). When the
#                 MMMC file grows setup/hold corners, pass e.g.
#                 VIEW_NAME="setupView holdView" on the command line to
#                 emit one .lib per corner.
#   MACRO_CELL  — cell name recorded in the .lib. Defaults to $TOP so
#                 the LEF and LIB agree out of the box. Override when
#                 renaming the macro for versioning.
#
# INPUTS  (Tcl globals from 00_config.tcl)
#   SCRIPT_DIR, CONFIG, TOP, RUN_DIR
#=======================================================================

# --- Resolve parameters from the environment with fallback defaults ---
set VIEW_NAME  [expr {[info exists ::env(VIEW_NAME)]  ? $::env(VIEW_NAME)  : "typView"}]
set MACRO_CELL [expr {[info exists ::env(MACRO_CELL)] ? $::env(MACRO_CELL) : $TOP}]

puts "\n--- ETM parameters ---"
puts "    VIEW_NAME  = $VIEW_NAME"
puts "    MACRO_CELL = $MACRO_CELL"

# --- Guard: the requested view must exist in the current MMMC setup ---
set known_views [get_db analysis_views .name]
foreach v $VIEW_NAME {
    if {[lsearch -exact $known_views $v] < 0} {
        error "Analysis view '$v' not defined in MMMC.\nKnown views: $known_views\nDefine the view in mmmc.tcl or override VIEW_NAME."
    }
}

# --- Probe for the ETM extraction command ---
# Order matters: do_extract_model is the Tempus signoff flow (higher
# accuracy, standard for taped-out ETMs); extract_model is the Innovus
# built-in fallback that covers licenses without the Tempus bundle.
set etm_candidates {do_extract_model extract_model}
set etm_cmd ""
foreach c $etm_candidates {
    if {[llength [info commands $c]] > 0} {
        set etm_cmd $c
        break
    }
}
if {$etm_cmd eq ""} {
    error "No ETM extraction command is available in this Innovus build.\n  Checked: $etm_candidates\nPossible fixes:\n  - Verify the Tempus signoff bundle is licensed on this host.\n  - Or run `tempus -files extract_etm.tcl` with the Tempus binary directly\n    (Tcl identical to what this script does)."
}
puts "INFO: ETM command: $etm_cmd"

# --- Destination: hw/pnr/cadence/export/<config>/ ---
set EXPORT_ROOT "$SCRIPT_DIR/export"
set EXPORT_DIR  "$EXPORT_ROOT/$CONFIG"
file mkdir $EXPORT_DIR

# --- Extract one .lib per requested view ---
# Innovus 21.19's do_extract_model emits a single Liberty file per call
# for the currently-active view. We loop over $VIEW_NAME so the
# single-view default ("typView") yields exactly $MACRO_CELL.lib, and
# multi-view setups yield $MACRO_CELL_<view>.lib files plus a default
# $MACRO_CELL.lib alias pointing at the first view.
set primary_lib ""
foreach _view $VIEW_NAME {
    set_analysis_view -setup $_view -hold $_view

    if {[llength $VIEW_NAME] == 1} {
        set _out "$EXPORT_DIR/${MACRO_CELL}.lib"
    } else {
        set _out "$EXPORT_DIR/${MACRO_CELL}_${_view}.lib"
    }

    puts "\n--- Running $etm_cmd for view '$_view' → $_out ---"
    if {[catch {
        $etm_cmd \
            -cell_name $MACRO_CELL \
            -lib_name  $MACRO_CELL \
            -view      $_view \
            -pg \
            $_out
    } _etm_err]} {
        error "$etm_cmd failed for view '$_view': $_etm_err"
    }

    if {![file exists $_out]} {
        error "$etm_cmd finished without errors but did not create $_out"
    }
    puts "INFO: ETM written: $_out"
    puts [format "INFO: LIB size:    %.1f KiB" [expr {[file size $_out] / 1024.0}]]

    if {$primary_lib eq ""} { set primary_lib $_out }
}

# --- Multi-view case: also publish $MACRO_CELL.lib as an alias of the
# first view so 01_init_design.tcl / mmmc.tcl keep working with the
# default file path.
if {[llength $VIEW_NAME] > 1} {
    set _alias "$EXPORT_DIR/${MACRO_CELL}.lib"
    file copy -force $primary_lib $_alias
    puts "INFO: default handle: $_alias → [file tail $primary_lib]"
}
