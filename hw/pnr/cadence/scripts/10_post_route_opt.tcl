#=======================================================================
# Stage 10 — Post-Route Optimization (Setup and Hold)
#
# PURPOSE
#   This is the final timing-closure pass.  Unlike the pre- and post-CTS
#   optimization stages, wire delays here are computed from ACTUAL routed
#   wire geometries extracted by Innovus's parasitic extractor.
#   Slack values reported after this stage are the authoritative PPA
#   numbers for the design.
#
# WHY ANOTHER OPTIMIZATION PASS?
#   The pre-route wire model (used in stages 6 and 8) estimates parasitics
#   from cell proximity.  Once real wires exist, actual delays can differ
#   significantly from the estimates — particularly for:
#     - Long wires on global metals (M7–M10) that cross the die.
#     - Dense areas where routing detours add extra wire length.
#     - High-fanout nets whose wires fan out to many distant sinks.
#   Some paths that passed pre-route timing may violate post-route, and
#   vice versa.  This pass reconciles those discrepancies.
#
# -postRoute vs -postRoute -hold
#   Same setup/hold split as stage 8, for the same reasons:
#     -postRoute        → fix setup violations (speed up slow paths)
#     -postRoute -hold  → fix hold violations (slow down fast paths)
#   At this stage, hold buffer insertion uses real wire delays, so the
#   hold fixes are more accurate (and typically fewer are needed if the
#   post-CTS hold pass was effective).
#
# KNOWN WARNINGS FROM THIS STAGE
#   IMPOPT-6080       — RESOLVED by switching to OCV analysis mode below.
#                        This was the fatal error that caused the previous
#                        run to abort.
#   IMPESI-3014 (×20) — Incomplete RC networks on SRAM-connected nets.
#                        Same root cause as stage 9; acceptable for this
#                        project.
#   IMPOPT-665  (×100)— Nets with unplaced terms (SRAM macro ports).
#                        Same as stage 8; intrinsic to fixed-macro placement.
#=======================================================================

# Switch timing analysis to On-Chip Variation (OCV) mode.
#
# By default, Innovus enables AAE-SI (Advanced Analysis Engine with Signal
# Integrity) optimization during postRoute optDesign.  AAE-SI models crosstalk-
# induced delay variations and REQUIRES the timing analysis mode to be OCV.
# With the default single-mode analysis, optDesign aborts with:
#   ERROR IMPOPT-6080: AAE-SI Optimization can only be turned on when the
#   timing analysis mode is set to OCV.
#
# (Note: there is no `setOptMode -enableAAE` flag in Innovus 21 — AAE-SI is
# controlled implicitly by the analysis mode.)
#
# For this project, which uses a single "typical" corner and a relaxed 100 MHz
# target clock, OCV adds no meaningful pessimism: with no derating tables
# defined in the delay corner, Innovus applies unity (1.0) deratings, so early
# and late path delays remain equal.  This switch simply satisfies the AAE-SI
# prerequisite and lets postRoute optimization run without aborting.
setAnalysisMode -analysisType onChipVariation

# Set optimization effort (Innovus 21: mode command, not an inline flag).
setOptMode -effort $OPT_EFFORT

# Fix post-route setup violations using real extracted wire delays.
optDesign -postRoute

# Fix post-route hold violations.
optDesign -postRoute -hold
