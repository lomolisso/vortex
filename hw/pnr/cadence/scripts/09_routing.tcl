# flowEffort forced to `standard`: `express` makes metal post-route opt can't recover.
setDesignMode    -flowEffort             standard
setNanoRouteMode -routeWithTimingDriven  true
setNanoRouteMode -drouteFixAntenna       true
setNanoRouteMode -routeInsertAntennaDiode true
setNanoRouteMode -routeAntennaCellName    ANTENNA_X1

if {$OPT_EFFORT eq "low"} {
    setNanoRouteMode -routeWithSiDriven false
    puts "INFO: NanoRoute — TD + antenna (SI off; EFFORT=low)."
} else {
    setNanoRouteMode -routeWithSiDriven true
    puts "INFO: NanoRoute — TD + SI + antenna."
}

routeDesign
connectGlobalNets

if {$DESIGN_KIND eq "cluster"} {
    array set _cnt {M1 0 M2 0 M3 0 M4 0 M5 0 M6 0 M7 0 M8 0 M9 0 M10 0}
    foreach _net [get_db nets -if {!.is_power && !.is_ground}] {
        foreach _wire [get_db $_net .wires] {
            set _layer [get_db $_wire .layer.name]
            if {[info exists _cnt($_layer)]} { incr _cnt($_layer) }
        }
    }
    puts "INFO: $CONFIG per-layer wire counts:"
    foreach _l {M1 M2 M3 M4 M5 M6 M7 M8 M9 M10} {
        puts [format "       %-4s %d" $_l $_cnt($_l)]
    }
}
