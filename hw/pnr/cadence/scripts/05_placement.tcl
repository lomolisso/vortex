#=======================================================================
# Stage 5 — Placement
#
# PURPOSE
#   Assign a legal (x, y) grid location to every standard cell in the
#   design.  The SRAM macros were already fixed in stage 3 and are not
#   moved here.  After this stage every cell has a physical coordinate
#   but no signal wires exist yet — routing happens in stage 9.
#
# WHAT place_design DOES
#   Innovus runs a multi-phase placement engine:
#     1. Global placement  — finds an approximate position for each cell
#                            that minimises estimated wirelength while
#                            respecting density and macro keepout halos.
#     2. Legalization      — snaps every cell to the nearest legal site on
#                            the placement grid and resolves any overlaps.
#     3. Detailed placement — locally improves the legal solution by
#                            swapping adjacent cells to reduce wirelength.
#
# WHY connectGlobalNets IS CALLED AGAIN
#   place_design may insert new cells:
#     - Filler cells     — dummy cells that fill gaps in the stdcell rows
#                          to satisfy density and well-continuity rules.
#     - Tie cells        — TIEHI/TIELO cells that feed constant 1/0 to nets
#                          that Innovus decides to implement that way.
#   These newly added cells need their VDD/VSS pins connected, so the
#   global net rules must be re-applied after placement completes.
#
# KNOWN WARNINGS FROM THIS STAGE
#   IMPOPT-576  (×5)  — 1181 nets have unplaced terms at the time
#                        place_design inspects the design.  These are
#                        connected to the SRAM macros whose I/O pins land
#                        at the macro boundary; they become placed once
#                        the macros are confirmed as fixed.  Timing
#                        estimation for those nets uses a lumped model
#                        until proper extraction runs post-route.
#   IMPSP-9025  (×3)  — No scan chain found/traced.  Scan chains are
#                        used for manufacturing test (DFT).  Not needed
#                        for this academic project.
#   IMPDC-1629  (×2)  — Default delay limit is 101 (below the default 1000).
#                        Affects accuracy for very high-fanout nets only.
#   IMPPSP-2001 (×10) — High pin density inside some GCells (routing grid
#                        squares).  Caused by the dense macro pin arrays.
#                        May slow early global routing but is not an error.
#=======================================================================

# Place all standard cells.  SRAM macros are already fixed (stage 3)
# and are treated as immovable obstacles by the placement engine.
place_design

# Re-apply power connections to cover any cells added during placement.
connectGlobalNets
