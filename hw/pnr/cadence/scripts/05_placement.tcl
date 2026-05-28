# NanGate45 lacks the CORE WELLTAP subclass — name the cell explicitly.
set_well_tap_mode -cell WELLTAP_X1 -rule 60
addWellTap -cellInterval 120 -prefix WELLTAP

# IMPCCOPT-2314: guard clk/reset across place_design's deleteBufferTree;
# release after place so CCOpt can buffer them.
set _guarded {}
foreach _name {clk reset} {
    set _n [get_db nets $_name]
    if {[llength $_n] > 0} {
        set_db $_n .dont_touch true
        lappend _guarded $_n
    }
}

place_design

foreach _n $_guarded { set_db $_n .dont_touch false }

connectGlobalNets
