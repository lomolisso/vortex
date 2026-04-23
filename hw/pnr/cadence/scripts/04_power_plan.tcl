#=======================================================================
# Stage 4 — Power Planning
#
# PURPOSE
#   Build the physical power distribution network (PDN) — the metal
#   structures that carry VDD and VSS from the chip boundary to every
#   standard cell and macro.  A well-designed PDN minimises IR-drop
#   (voltage sag due to wire resistance) and electromigration risk.
#
# THREE-TIER PDN STRATEGY
# ─────────────────────────────────────────────────────────────────────
#   Tier 1 — Standard-cell rails (M1, horizontal)
#     Every standard-cell row has a VDD rail along its top edge and a
#     VSS rail along its bottom edge, both on metal 1.  These are the
#     finest-grained wires, running the full width of the row.
#     Command: sroute -connect corePin
#
#   Tier 2 — Power ring (M7 top/bottom, M8 left/right)
#     A closed rectangular ring encircles the entire core.  It acts as a
#     low-resistance "bus" connecting the I/O pads (outside) to the
#     internal stripe grid (inside).  Using M7/M8 (thick global metals)
#     minimises resistance and allows the ring to carry the full chip
#     current.  VSS and VDD each get their own ring wire side by side.
#     Command: addRing
#
#   Tier 3 — Vertical power stripes (M7)
#     Vertical wires spaced every 40 µm across the core connect the ring
#     (tier 2) down to the M1 rails (tier 1) via vias.  Without stripes,
#     cells far from the ring would see high IR-drop because current would
#     have to travel horizontally along the thin M1 rails.
#     Command: addStripe
#
# LAYER USAGE NOTE (FreePDK45)
#   M1  — used by stdcell internal routing and power rails (horizontal)
#   M2  — signal routing (vertical, preferred)
#   M3  — signal routing (horizontal, preferred)
#   ...
#   M7  — power ring top/bottom + power stripes (vertical)
#   M8  — power ring left/right (horizontal)
#   M9, M10 — left for clock and long-distance signal nets if needed
#
# KNOWN WARNINGS FROM THIS STAGE
#   IMPPP-532  (×20) — ViaGen warning: metal4 and metal7 run in the same
#                      preferred direction, but the via between them must be
#                      orthogonal.  Innovus handles this automatically by
#                      inserting a jog; no manual intervention needed.
#   IMPEXT-2882 (×9) — Via resistance for specific via types not found in
#                      the cap table; defaulted to 4 Ω.  Affects IR-drop
#                      accuracy but acceptable for a course project.
#=======================================================================

# -----------------------------------------------------------------------
# Tier 1: Connect M1 power rails to the core power pins.
# sroute routes the power/ground wires to the pins of the standard-cell
# rows ("corePin").  After this command, every row has a VDD rail on top
# and VSS rail on bottom, both touching the cell power pins.
# -----------------------------------------------------------------------
sroute -connect { corePin } -nets { VDD VSS }

# -----------------------------------------------------------------------
# Tier 2: Power ring around the core.
#
#   -follow  core        → ring hugs the core boundary rectangle
#   -offset  {... 2 ...} → ring starts 2 µm outside each core edge
#   -spacing {... 2 ...} → 2 µm gap between the VDD and VSS ring wires
#   -width   {... 2 ...} → each ring wire is 2 µm wide
#   -layer   {top metal7 bottom metal7 left metal8 right metal8}
#             Horizontal segments use M7 (which runs horizontal here);
#             vertical segments use M8.
#   -nets    { VSS VDD } → create one ring wire for VSS, then one for VDD.
#                          Innovus places them in the order listed.
# -----------------------------------------------------------------------
addRing \
    -follow  core \
    -offset  {top 2 bottom 2 left 2 right 2} \
    -spacing {top 2 bottom 2 left 2 right 2} \
    -width   {top 2 bottom 2 left 2 right 2} \
    -layer   {top metal7 bottom metal7 left metal8 right metal8} \
    -nets    { VSS VDD }

# -----------------------------------------------------------------------
# Tier 3: Vertical power stripes across the core.
#
#   -nets              { VSS VDD }   → alternating VSS / VDD stripes
#   -layer             metal7        → thick global metal for low resistance
#   -direction         vertical      → stripes run from ring-bottom to ring-top
#   -width             1.6           → each stripe is 1.6 µm wide
#   -spacing           1.6           → gap between VSS and VDD pair members
#   -set_to_set_distance 40          → centre-to-centre distance between
#                                      adjacent VSS–VDD stripe pairs (µm).
#                                      Decrease to reduce IR-drop at the
#                                      cost of routing congestion.
# -----------------------------------------------------------------------
addStripe \
    -nets              { VSS VDD } \
    -layer             metal7 \
    -direction         vertical \
    -width             1.6 \
    -spacing           1.6 \
    -set_to_set_distance 40
