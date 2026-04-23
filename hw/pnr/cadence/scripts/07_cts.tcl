#=======================================================================
# Stage 7 — Clock Tree Synthesis (CTS)
#
# PURPOSE
#   Build the physical clock distribution network that delivers the clock
#   signal from its source (the clk port) to every flip-flop in the design
#   with minimal skew and bounded insertion delay.
#
# WHY DOES THE CLOCK NEED A TREE?
#   A single wire driving 54,463 flip-flops (this design's clock fanout,
#   see warning IMPCCOPT-1157) would have enormous capacitance.  The
#   driving cell would be unable to switch that load fast enough, and
#   different parts of the wire would see different delays (skew).
#   A balanced binary tree of buffers/inverters distributes the load so
#   that each buffer drives only a small number of downstream cells, and
#   all clock paths from source to sink have nearly equal total delay.
#
# WHAT ccopt_design DOES
#   ccopt_design is Innovus's Concurrent Clock and Data Optimization engine
#   (available from Innovus 18+).  It simultaneously:
#     1. Synthesizes the clock tree (inserts and sizes clock buffers/
#        inverters, routes clock wires on dedicated clock layers).
#     2. Optimizes data paths for setup and hold, taking into account
#        the real (non-ideal) clock delays computed during tree synthesis.
#   This co-optimization is more effective than running CTS and data
#   optimization in separate sequential steps (the traditional approach).
#
# CELL SELECTION
#   set_ccopt_property buffer_cells   — list of cells Innovus may use as
#                                       non-inverting clock buffers.
#   set_ccopt_property inverter_cells — list of cells Innovus may use as
#                                       inverting clock buffers (two in
#                                       series give a non-inverting path
#                                       with better balanced rise/fall).
#   The BUF_X* / INV_X* cells are NanGate45 drive-strength variants:
#     X1 = ×1 drive (weakest), X32 = ×32 drive (strongest).
#   A wider range gives the CTS engine more freedom to balance the tree.
#
# KNOWN WARNINGS FROM THIS STAGE
#   IMPCCOPT-1157 (×1) — Root driver has 54,463 fanout; max_fanout
#                         constraint not met at the root.  Expected: the
#                         root pin always has high fanout before the tree
#                         is built.  CTS buffers the tree below the root.
#   IMPCCOPT-2314 (×1) — 1 clock net is marked ideal or dont_touch and
#                         will not be buffered.  This is the clock source
#                         pin itself (set_ideal_network in the SDC); correct
#                         and harmless.
#   IMPTCM-77     (×2) — Obsolete setNanoRouteMode options used internally
#                         by ccopt_design.  These are called from inside
#                         Innovus, not from our scripts, so they cannot be
#                         suppressed via user Tcl.  Still functional in
#                         this release; update when upgrading Innovus.
#=======================================================================

# --- Specify which standard cells the CTS engine may use ---
# Using all available drive strengths lets the engine pick the smallest
# cell that meets the target skew/transition constraints, which saves area
# and power.
set_ccopt_property buffer_cells   { BUF_X1 BUF_X2 BUF_X4 BUF_X8 BUF_X16 BUF_X32 }
set_ccopt_property inverter_cells { INV_X1 INV_X2 INV_X4 INV_X8 INV_X16 INV_X32 }

# Set optimization effort before CTS.
# ccopt_design uses setOptMode for effort control in Innovus 21.
setOptMode -effort $OPT_EFFORT

# --- Run concurrent clock-and-data optimization ---
# This builds the clock tree AND simultaneously adjusts data-path cells
# to correct any new setup/hold violations introduced by the real clock
# delays.  This is typically the longest-running stage.
#
# $OPT_EFFORT (low/medium/high) is set in 00_config.tcl from the Makefile
# EFFORT variable.  'low' reduces CTS optimization iterations, which is
# usually sufficient for this design's relaxed 100 MHz target clock.
ccopt_design
