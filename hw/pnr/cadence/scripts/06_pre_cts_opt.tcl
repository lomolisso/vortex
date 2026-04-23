#=======================================================================
# Stage 6 — Pre-CTS Optimization (Setup Timing)
#
# PURPOSE
#   Fix setup-time (max-delay) violations in the placed but un-routed
#   design before the clock tree is built.  This is the first of three
#   optimization passes; each targets a different point in the flow.
#
# WHY OPTIMIZE BEFORE ROUTING?
#   After placement, Innovus estimates wire delays using a statistical
#   model (virtual/ideal wires with estimated parasitics from placement
#   proximity).  These estimates are less accurate than post-route
#   extraction, but optimizing here is still valuable because:
#     - The placer may have put timing-critical cells too far apart.
#     - DC (synthesis) optimized for wire-load models, not real placement.
#   Early setup fixes reduce the number of violations the post-route
#   optimizer must handle, which shortens runtime and improves QoR.
#
# WHAT optDesign -preCTS DOES
#   - Identifies paths whose estimated delay exceeds the clock period.
#   - Applies logic transformations to reduce delay:
#       • Cell upsizing  — replace a slow cell with a higher-drive-strength
#                          variant (e.g. BUF_X1 → BUF_X4).
#       • Cell resizing  — same cell function, different drive strength.
#       • Buffer insertion/deletion — reshape high-fanout nets.
#       • Gate cloning   — duplicate a cell to split its fanout.
#       • Logic restructuring — remap to equivalent but faster topology.
#   - Does NOT fix hold violations (those are post-CTS concerns because
#     hold is dominated by clock skew, which is unknown before CTS).
#
# IDEAL CLOCK ASSUMPTION
#   At this stage the clock has no physical tree — Innovus treats it as
#   an ideal zero-skew, zero-latency wire.  Setup slack estimates here
#   are therefore optimistic (actual slack post-CTS will be lower due to
#   clock insertion delay and skew).  A typical guard-band is to target
#   ~200–500 ps of positive setup slack pre-CTS.
#
# KNOWN WARNINGS FROM THIS STAGE
#   IMPOPT-3195 (×2)  — Analysis mode changed during optimization.
#                       Innovus switches between setup and hold views
#                       internally; harmless informational message.
#   IMPEXT-6197 (×11) — Cap table not specified; lower parasitic accuracy.
#                       Partially improved by setDesignMode -process 45 in
#                       stage 1.  Cannot be fully eliminated without a PDK-
#                       specific cap table file; acceptable for this project.
#   IMPEXT-3530       — RESOLVED in stage 1 by setDesignMode -process 45.
#=======================================================================

# Set optimization effort before calling optDesign.
# In Innovus 21, effort is a mode setting (setOptMode), not an inline flag.
# $OPT_EFFORT is set in 00_config.tcl from the Makefile EFFORT variable
# (default: low).  Use 'make ... EFFORT=medium' or 'EFFORT=high' for higher
# quality at the cost of longer runtime.
setOptMode -effort $OPT_EFFORT

# Run pre-CTS setup optimization.
# -preCTS : use ideal (zero-delay) clock; focus on setup (max-delay) only.
optDesign -preCTS
