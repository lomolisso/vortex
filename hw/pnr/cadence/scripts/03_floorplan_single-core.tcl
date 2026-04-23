#=======================================================================
# Stage 3 — Floorplan  (config: single-core)
#
# The single-core PnR target elaborates VX_socket_top, which wraps
# exactly ONE VX_core plus its private L1 I$/D$, local memory, and
# register file. The macro set and their relative positions are
# therefore identical to the legacy 1c1n4w4t configuration — we simply
# reuse 03_floorplan_1c1n4w4t.tcl as the floorplan body.
#
# Only two things differ between this config and the legacy Vortex-top
# 1c1n4w4t run:
#
#   1. Floorplan dimensions
#      FLOORPLAN_W / FLOORPLAN_H / FLOORPLAN_MARGIN are emitted per
#      config by the Python area-budget script (in our case
#      hw/scripts/compute_single_core_area_budget.py) into
#      runs/single-core/floorplan_budget.tcl, which the sourced file
#      picks up automatically.
#
#   2. Socket instance path
#      For single-core the PnR top is VX_socket_top itself, so the
#      socket instance is reached via just "socket" (no g_clusters /
#      g_sockets wrappers). We override $sock here before sourcing so
#      that the hand-tuned macro placements resolve correctly.
#=======================================================================

set sock "socket"

source "$SCRIPT_DIR/scripts/03_floorplan_1c1n4w4t.tcl"
