#=======================================================================
# Stage 9 — Routing
#
# PURPOSE
#   Draw the actual metal wires that connect every cell pin to its
#   corresponding net.  After this stage the design is physically
#   complete — every connection specified in the netlist has a
#   corresponding geometric wire on a specific metal layer.
#
# WHAT routeDesign DOES
#   Innovus runs NanoRoute, its global + detailed router, in three phases:
#
#   1. Global routing — divides the die into a coarse grid (GCells) and
#      finds a congestion-aware path (sequence of GCells) for each net.
#      Does not commit actual wire shapes; only reserves capacity.
#
#   2. Track assignment — maps each global route to a specific routing
#      track on a specific metal layer, resolving most DRC conflicts.
#
#   3. Detailed routing — places the exact wire segments and vias,
#      enforces all design rules (spacing, width, via enclosure, etc.),
#      and repairs any remaining DRC violations.
#
# LAYER USAGE (FreePDK45 preferred routing directions)
#   M1   — horizontal  (stdcell power rails; limited use for signals)
#   M2   — vertical    (primary local signal routing)
#   M3   — horizontal
#   M4   — vertical
#   M5   — horizontal
#   M6   — vertical
#   M7   — vertical    (power stripes — mostly blocked for signals)
#   M8   — horizontal  (power ring sides — mostly blocked for signals)
#   M9   — horizontal  (long-distance clock/signal)
#   M10  — vertical
#
# WHY connectGlobalNets IS CALLED AGAIN
#   NanoRoute may insert routing-fix buffers or additional tie cells to
#   resolve antenna violations.  These new cells need their power pins
#   connected, so the global net rules must be re-applied after routing.
#
# KNOWN WARNINGS FROM THIS STAGE
#   IMPESI-3014 (×20) — Incomplete RC network on some SRAM-connected nets;
#                        lumped model used for timing.  Pre-extraction
#                        limitation; accurate parasitics are computed during
#                        post-route extraction (stage 10).  Cannot be fixed
#                        without a full cap table for the FreePDK45 PDK.
#   IMPOPT-7320       — RESOLVED by setSIMode -fixGlitch false above.
#=======================================================================

# Disable glitch fixing.
#
# By default, Innovus's NanoRoute enables glitch-fixing (crosstalk-induced
# glitch analysis) but leaves the glitch report disabled, producing warning
# IMPOPT-7320.  For this project, which uses a single typical corner with no
# SI (signal integrity) analysis, glitch fixing adds no value and only
# increases runtime.  Disabling it eliminates the warning entirely.
# To enable full glitch analysis in the future:
#   setSIMode -fixGlitch true -enable_glitch_report true
setSIMode -fixGlitch false

# Set routing effort via NanoRoute mode (Innovus 21 does not accept
# -effortLevel as a routeDesign flag; effort is a mode setting instead).
#   low    — fewer global-route iterations; fastest runtime.
#   medium — default NanoRoute behaviour; good DRC and SI results.
#   high   — maximum iterations; use for near-tapeout quality.
# routeWithTimingDriven=true enables timing-aware routing at all effort levels.
setNanoRouteMode -routeWithTimingDriven true
if {$OPT_EFFORT eq "low"} {
    setNanoRouteMode -routeWithEco false
} elseif {$OPT_EFFORT eq "high"} {
    setNanoRouteMode -routeWithEco true
}

# Reserve M9/M10 for full-vortex over-the-macro routing (single-core only).
#
# The single-core run is packaged into a hard macro consumed by the
# full-vortex top level (see extract_macro.tcl). To give that top-level
# router a clean pair of "open" layers that cross over the macro, we
# cap NanoRoute at M8 here. Combined with the PDN (M7 stripes, M8 ring
# H-sides), this leaves M9 and M10 completely untouched inside the
# macro, so the extracted LEF advertises no obstruction on those layers.
#
# full-vortex routing continues to use all 10 metals; only the inner
# single-core routing is capped.
if {$CONFIG eq "single-core"} {
    setNanoRouteMode -routeTopRoutingLayer 8
    puts "INFO: single-core — capping routing at M8 (reserving M9/M10 for full-vortex over-macro)."
}

# Route all signal nets.
# NanoRoute automatically routes clock nets last (after signal nets) when
# a clock tree already exists, to avoid disturbing the CTS results.
routeDesign

# Re-apply power connections to any cells inserted by the router.
connectGlobalNets
