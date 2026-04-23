#=======================================================================
# Stage 2 — Global Net Connections
#
# PURPOSE
#   Tell Innovus how to connect VDD and VSS to every cell in the design.
#   Standard cells and SRAM macros get their power through physical rails
#   on M1, but Innovus first needs a logical model of which pins are power
#   and which are ground before it can build those rails.
#
# WHY IS THIS NEEDED?
#   In a synthesized Verilog netlist, power and ground pins are often not
#   explicitly wired — they are implicit.  Innovus requires you to declare
#   the global connection rules so it can:
#     1. Add power rails to the standard-cell rows during sroute.
#     2. Connect tie-high/tie-low cells (used to drive constant 0/1 logic)
#        to the correct supply.
#     3. Verify that every cell has a valid power connection before routing.
#
# WHY IS connectGlobalNets CALLED MULTIPLE TIMES?
#   The procedure is defined once here and then called again after
#   place_design (stage 5) and after routeDesign (stage 9).  Each call
#   re-applies the rules to any newly inserted cells (buffers, clock
#   tree cells, tie cells) that Innovus added since the last call.
#   Without the post-placement and post-routing calls, newly inserted
#   cells would have floating power pins, causing LVS failures.
#
# INPUTS  (Tcl globals from 00_config.tcl)
#   None beyond the design already loaded by init_design.
#
# KNOWN WARNINGS FROM THIS STAGE
#   None specific.  The EMS-27 message-limit warnings seen in the log
#   originate from init_design (stage 1) not from globalNetConnect.
#=======================================================================

# Define the connection rules as a procedure so the same logic can be
# re-invoked cleanly after placement and after routing (see above).
proc connectGlobalNets {} {
    # Connect every cell's VDD pgpin to the VDD net.
    # "pgpin" means power/ground pin — the dedicated supply pin defined
    # in the liberty file (.lib) with pin_direction = "internal".
    globalNetConnect VDD -type pgpin -pin VDD -all

    # Same for VSS (ground).
    globalNetConnect VSS -type pgpin -pin VSS -all

    # Tie-high cells (e.g. TIEHX1) output a logic 1.  Their output pin
    # is not really a supply but must be tied to VDD for correct voltage.
    globalNetConnect VDD -type tiehi -all

    # Tie-low cells (e.g. TIELOX1) output a logic 0; connected to VSS.
    globalNetConnect VSS -type tielo -all

    # Commit the rules to the database.  Must be called after all
    # globalNetConnect declarations to take effect.
    applyGlobalNets
}

# First call — applies to all cells present after init_design.
connectGlobalNets
