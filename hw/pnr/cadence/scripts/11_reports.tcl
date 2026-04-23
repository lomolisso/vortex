#=======================================================================
# Stage 11 — Reports
#
# PURPOSE
#   Generate human-readable sign-off quality reports for timing, area,
#   power, DRC, congestion, and interconnect.  All files land in
#   $REPORT_DIR (runs/<config>/reports/).
#
# REPORT TYPES
# ─────────────────────────────────────────────────────────────────────
# timing_setup.rpt  — The 20 worst setup (max-delay) paths.
#   Each path entry shows:
#     • Startpoint and endpoint (flip-flop names)
#     • Per-cell delay breakdown (cell delay + wire delay)
#     • Required and arrival time at the endpoint
#     • Slack = required_time − arrival_time
#       Positive slack  → timing met.  Negative slack → violation.
#
# timing_hold.rpt   — The 20 worst hold (min-delay) paths.
#   Same format as setup but uses -late flag for hold analysis.
#   Hold violations mean data changes too close to the capturing edge.
#
# area.rpt          — Post-placement cell area breakdown.
#   Reports total cell area, macro area, and standard-cell area.
#   Useful for comparing against the DC synthesis area estimate and for
#   verifying that utilization matches the floorplan target.
#
# power_total.rpt   — Chip-level dynamic + leakage power.
#   Dynamic power = switching activity × wire capacitance × VDD²
#   These numbers use actual routed wire capacitances, making them far
#   more accurate than the DC wire-load-model estimates.
#
# power_hierarchy.rpt — Same power data broken down by module hierarchy.
#   Identifies which subsystem (cache, FPU, LSU, etc.) consumes the most
#   power — valuable for microarchitectural power optimization.
#
# drc.rpt           — Design Rule Check violations.
#   Lists any remaining physical violations after routing (spacing errors,
#   width violations, unconnected nets, etc.).  A clean DRC is required
#   before tapeout.  For a course project, a small number of DRC errors
#   in the SRAM keepout regions is acceptable.
#
# congestion.rpt    — Routing congestion summary.
#   Reports the percentage of routing tracks used per layer per GCell.
#   Values >90% indicate congested areas where routeDesign had to detour
#   wires, which degrades timing.
#
# net_detail.rpt    — Per-net wire length and via count.
#   The primary source of interconnect PPA data beyond DC estimates:
#     • Total wire length in µm (per net and chip total)
#     • Via count (each via adds resistance and area)
#   Use this to identify the most expensive nets and understand routing
#   topology.
#
# KNOWN WARNINGS FROM THIS STAGE
#   None beyond those already reported by the optimization stages.
#=======================================================================

# --- Timing: 20 worst setup paths ---
# -nworst 20    → report up to 20 unique violating endpoints
# -max_paths 20 → report up to 20 paths total
# -path_type full → include full per-stage delay breakdown
#
# Note: Innovus's report_timing has no -out_file flag (that's a Tempus/Genus
# option).  Use the `redirect` command to capture output to a file instead.
redirect "$REPORT_DIR/timing_setup.rpt" {
    report_timing -nworst 20 -max_paths 20 -path_type full
}

# --- Timing: 20 worst hold paths ---
# -late is the flag for hold (min-delay) analysis in Innovus.
redirect "$REPORT_DIR/timing_hold.rpt" {
    report_timing -nworst 20 -max_paths 20 -path_type full -late
}

# --- Area: post-placement cell footprints ---
# Innovus 21 exposes the Tempus-compatible `report_area` (snake_case); the
# camelCase `reportArea` does NOT exist in this version.  `report_area` writes
# to stdout, so wrap it in `redirect` to capture to a file.
redirect "$REPORT_DIR/area.rpt" { report_area }

# --- Power: chip-level totals ---
redirect "$REPORT_DIR/power_total.rpt" { report_power }

# --- Power: per-module hierarchy breakdown ---
# Note: in Innovus 21 `report_power -hierarchy` takes a *depth* argument (int),
# not a boolean flag.  Use a depth large enough to reach the interesting
# sub-blocks (cache, FPU, LSU, ...) — 5 levels comfortably covers the Vortex
# module tree without flooding the report.
redirect "$REPORT_DIR/power_hierarchy.rpt" { report_power -hierarchy 5 }

# --- DRC: up to 1000 violations listed ---
# There is no `reportDRC` or camelCase `verifyDRC` in Innovus 21.19.  The
# canonical command is the snake_case `verify_drc`, which re-runs the checker
# and writes the report directly via `-report`.
verify_drc -limit 1000 -report "$REPORT_DIR/drc.rpt"

# --- Congestion: routing track utilisation per layer ---
# In Innovus 21, bare `reportCongestion` emits IMPSP-9110 and writes nothing
# useful: one of -hotspot (congestion hotspot score) or -overflow (per-layer
# GCell overflow distribution) is mandatory.  Emit both so the report captures
# the full picture.
redirect "$REPORT_DIR/congestion.rpt" {
    puts "== Hotspot score =="
    reportCongestion -hotspot
    puts "\n== Overflow distribution =="
    reportCongestion -overflow
}

# #region agent log - verify congestion fix produced a real report
proc __dbg_log {hyp msg data} {
    set f [open "/home/rid2/cs533.work/vortex/.cursor/debug-13ecaf.log" a]
    set ts [clock milliseconds]
    set safe_msg  [string map [list \\ \\\\ \" \\\"] $msg]
    set safe_data [string map [list \\ \\\\ \" \\\"] $data]
    puts $f "{\"sessionId\":\"13ecaf\",\"hypothesisId\":\"CONG\",\"runId\":\"post-fix\",\"location\":\"11_reports.tcl\",\"message\":\"$safe_msg\",\"data\":\"$safe_data\",\"timestamp\":$ts}"
    close $f
}
set __dbg_head ""
set __dbg_fh [open "$REPORT_DIR/congestion.rpt" r]
for {set __i 0} {$__i < 6} {incr __i} {
    if {[gets $__dbg_fh __line] < 0} break
    append __dbg_head "$__line || "
}
close $__dbg_fh
__dbg_log "CONG" "congestion.rpt size (bytes)"   [file size "$REPORT_DIR/congestion.rpt"]
__dbg_log "CONG" "congestion.rpt first 6 lines"  $__dbg_head
# #endregion

# --- Interconnect: per-net wire length and via count ---
# `reportNetDetail` does not exist in Innovus 21; use `reportNetStat` which
# produces per-net wire-length and via statistics.
redirect "$REPORT_DIR/net_detail.rpt" { reportNetStat }

puts "Reports written to $REPORT_DIR"
