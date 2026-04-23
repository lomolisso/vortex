#=======================================================================
# Stage 8 — Post-CTS Optimization (Setup and Hold)
#
# PURPOSE
#   After CTS, the clock tree is real (non-ideal) and every flip-flop has
#   a known clock arrival time.  This stage fixes any remaining timing
#   violations using that accurate clock model — both setup (max-delay)
#   and hold (min-delay) violations.
#
# WHY TWO SEPARATE optDesign CALLS?
#   Setup and hold violations require opposite remedies:
#
#   Setup (max-delay) violation — data arrives AFTER the clock edge.
#     Fix: speed up the data path (upsize cells, remove buffers, shorten
#          routes) or accept slightly more skew to borrow time.
#
#   Hold (min-delay) violation — data arrives TOO EARLY relative to the
#     capturing clock edge (the data changes before the latch window opens
#     for the NEXT cycle).  Fix: slow down the data path (insert delay
#     buffers, usually on short paths between flip-flops with large clock
#     skew).  Fixing hold by adding delays cannot be done before CTS
#     because hold is dominated by clock skew, which is unknown until CTS.
#
#   Mixing both in a single pass can cause the optimizer to "chase its
#   tail" — fixing a setup violation creates a hold violation and vice
#   versa.  Running setup first, then hold, gives the tool a stable
#   starting point for the hold pass.
#
# WIRE DELAY MODEL
#   At this stage wires are still modelled using virtual (estimated)
#   parasitics from placement proximity.  Actual wire delays only become
#   available after routeDesign + extraction in stage 9/10.  Post-CTS
#   optimization therefore targets an aggressive setup margin to absorb
#   the uncertainty in the pre-route wire model.
#
# KNOWN WARNINGS FROM THIS STAGE
#   IMPOPT-665  (×100) — Nets with unplaced terms on SRAM macro I/O ports.
#                         The optimizer cannot see inside fixed macros, so
#                         it treats their terminal pins as "unplaced".
#                         Timing for those nets uses a lumped model until
#                         post-route extraction.  Cannot be eliminated
#                         without changing how Innovus models macro pin
#                         access points; harmless for this project.
#   EMS-27      (×1)   — Message display limit reached for IMPOPT-665
#                         (capped at 20 shown; 100 total occurrences).
#=======================================================================

# Set optimization effort (Innovus 21: mode command, not an inline flag).
setOptMode -effort $OPT_EFFORT

# Fix remaining setup (max-delay) violations.
# Uses real clock arrival times from the CTS stage.
optDesign -postCTS

# Fix hold (min-delay) violations.
# -hold instructs Innovus to insert delay buffers on paths that complete
# too quickly; it does NOT touch paths that already have setup margin.
optDesign -postCTS -hold
