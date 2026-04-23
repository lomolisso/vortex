#=======================================================================
# Stage 12 — Output Files
#
# PURPOSE
#   Write all deliverable files from the completed PnR run.  These files
#   are used for sign-off verification, fabrication, and re-entry into
#   Innovus for later inspection or engineering change orders (ECOs).
#   All files land in $RESULTS_DIR (runs/<config>/results/).
#
# Filenames are based on the $TOP variable set in 00_config.tcl so that
# the single-core run produces VX_socket_top.* and the full-vortex run
# produces Vortex.* without stepping on each other.
#
# OUTPUT FILE TYPES
# ─────────────────────────────────────────────────────────────────────
# ${TOP}.gds   — GDSII layout stream.
#   The industry-standard format for IC mask data.  Contains every
#   geometric shape (wires, vias, cell boundaries) on every layer.
#   Opened in Virtuoso (Cadence) or KLayout (open-source) for visual
#   inspection and DRC/LVS verification.
#   Requires a layer-map file (innovus.map) that translates Innovus
#   internal layer numbers to the GDSII layer numbers expected by the
#   PDK.  If the map is not found, GDS export is skipped gracefully.
#
# ${TOP}_pnr.v — Final gate-level Verilog netlist.
#   Same logical connectivity as the DC synthesis netlist but updated
#   to reflect any cells inserted, removed, or resized during PnR
#   (buffers, clock tree cells, hold-fix delays, filler cells).
#   Used for formal equivalence checking against the RTL.
#
# ${TOP}.def   — Design Exchange Format snapshot.
#   A text representation of the placed-and-routed design: all cell
#   placements, all wire routes, and all via locations.  Can be loaded
#   back into Innovus to resume work without re-running the flow.
#   Also consumed by OpenROAD and other open-source EDA tools.
#
# ${TOP}       — Full Innovus database (binary).
#   The complete Innovus session state.  Re-open with:
#     innovus -init results/${TOP}
#   This is the most complete re-entry point: all constraints, analysis
#   views, and optimization state are preserved.
#
# dieshot.png  — PNG screenshot of the routed layout (optional).
#   Requires an active X display (DISPLAY env var set).  In headless
#   batch mode (HPC cluster without X forwarding), this step is skipped
#   automatically with instructions for manual capture.
#   Alternatively, open the GDS in KLayout:
#     klayout results/${TOP}.gds
#
# KNOWN WARNINGS FROM THIS STAGE
#   None — output commands do not produce warnings in normal operation.
#=======================================================================

# --- GDS export ---
# streamOut writes the GDSII file.
# -mapFile  : translates Innovus layer names → GDSII layer numbers.
# -units 2000 : sets the database unit (0.5 nm resolution, standard for 45 nm).
# The map file path is installation-specific; wrapped in a guard so the
# run does not abort if the file is absent.
set GDS_MAP "/class/ece425/innovus.map"
if {[file exists $GDS_MAP]} {
    streamOut "$RESULTS_DIR/${TOP}.gds" -mapFile $GDS_MAP -units 2000
    puts "GDS written : $RESULTS_DIR/${TOP}.gds"
} else {
    puts "WARNING: GDS map not found at $GDS_MAP — skipping GDS export."
    puts "         Locate the map file and run:"
    puts "           streamOut $RESULTS_DIR/${TOP}.gds -mapFile <map> -units 2000"
}

# --- Final post-route gate-level netlist ---
saveNetlist "$RESULTS_DIR/${TOP}_pnr.v"

# --- DEF snapshot ---
defOut "$RESULTS_DIR/${TOP}.def"

# --- Full Innovus session database ---
saveDesign "$RESULTS_DIR/${TOP}"

# --- Dieshot (requires an X display) ---
# "info exists ::env(DISPLAY)" checks for the X11 display variable.
# On a login node with X forwarding (ssh -X) this is typically ":0" or
# "localhost:10.0"; on a batch compute node it is usually unset.
if {[info exists ::env(DISPLAY)] && $::env(DISPLAY) ne ""} {
    win                  ;# open the layout GUI window
    fit                  ;# zoom to fit the full die in the window
    uiSetTool select     ;# switch to the selection tool (clears any active mode)

    # #region agent log - helper for post-fix verification
    proc __dbg_log {hyp msg data} {
        set f [open "/home/rid2/cs533.work/vortex/.cursor/debug-13ecaf.log" a]
        set ts [clock milliseconds]
        set safe_msg  [string map [list \\ \\\\ \" \\\"] $msg]
        set safe_data [string map [list \\ \\\\ \" \\\"] $data]
        puts $f "{\"sessionId\":\"13ecaf\",\"hypothesisId\":\"$hyp\",\"runId\":\"post-fix\",\"location\":\"12_outputs.tcl\",\"message\":\"$safe_msg\",\"data\":\"$safe_data\",\"timestamp\":$ts}"
        close $f
    }
    # #endregion

    # Innovus 21.19-s058_1 does not expose `displaySnapshot` (or any camelCase
    # or snake_case variant) — this was verified at runtime via `info commands`.
    # There is no direct PNG dieshot command in this release, so we attempt the
    # historical call and fall back to a KLayout hint if it is unavailable.
    if {[catch {displaySnapshot -type PNG -resolution 2048 -file "$REPORT_DIR/dieshot.png"} __ds_err]} {
        puts "NOTE: displaySnapshot is not available in this Innovus build — dieshot skipped."
        puts "      ($__ds_err)"
        puts "      Open the GDS in KLayout for a layout image instead:"
        puts "        klayout $RESULTS_DIR/${TOP}.gds"
        __dbg_log "H5" "displaySnapshot gracefully skipped" $__ds_err
    } else {
        puts "Dieshot  : $REPORT_DIR/dieshot.png"
        __dbg_log "H5" "displaySnapshot succeeded" "ok"
    }
} else {
    puts "NOTE: No DISPLAY set — dieshot skipped."
    puts "      Re-open the saved database in the Innovus GUI to capture it:"
    puts "        innovus -init $RESULTS_DIR/${TOP}"
    puts "      Then run:  fit ; displaySnapshot -type PNG -resolution 2048 -file dieshot.png"
    puts "      Or open the GDS in KLayout:  klayout $RESULTS_DIR/${TOP}.gds"
}

puts "============================================================"
puts " PnR complete  |  config: $CONFIG"
puts " Results dir   : $RESULTS_DIR"
puts "============================================================"
