# Snap I/O pins; fail if any wire sits above $TOP_RLAYER.
setDesignMode -process 45
set delaycal_use_default_delay_limit 1000

puts "\n--- Snapping top-level I/O pins to the routing grid ---"
if {[catch {snapFPlanIO} _err]} { error "snapFPlanIO failed: $_err" }
puts "INFO: snapFPlanIO OK ([llength [get_db ports]] ports)."

set reserved_layers {}
set _reserve_nums {}
for {set _n [expr {$TOP_RLAYER + 1}]} {$_n <= 10} {incr _n} { lappend _reserve_nums $_n }
foreach num $_reserve_nums {
    foreach candidate [list "metal$num" "M$num" "Metal$num"] {
        if {[get_db layers $candidate] ne ""} { lappend reserved_layers $candidate; break }
    }
}
puts "INFO: reserved layers: $reserved_layers"

set violators {}
foreach net [get_db nets] {
    set nname [get_db $net .name]
    foreach wire [get_db $net .wires] {
        set wlyr ""
        catch {set wlyr [get_db $wire .layer.name]}
        if {[lsearch -exact $reserved_layers $wlyr] >= 0} {
            lappend violators [list $nname $wlyr]
        }
    }
}

if {[llength $violators] > 0} {
    puts "ERROR: reserved layers contain routed geometry:"
    set shown 0
    foreach entry $violators {
        lassign $entry nname wlyr
        puts "  net=$nname  layer=$wlyr"
        if {[incr shown] >= 20} { break }
    }
    if {[llength $violators] > 20} {
        puts "  ... and [expr {[llength $violators] - 20}] more."
    }
    error "Re-route stage 9 with the correct top-layer cap, then retry."
}
puts "INFO: reserved layers are clean."
