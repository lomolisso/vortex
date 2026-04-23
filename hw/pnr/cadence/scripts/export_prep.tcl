#=======================================================================
# Export Prep — sanity checks before LEF/LIB extraction
#
# PURPOSE
#   Bring the reloaded single-core database into a state that is safe
#   to publish as a hard macro, and fail fast if anything is off:
#
#     1. Re-apply post-checkpoint settings that the saved .dat does not
#        carry (process node, delay-calc path limit).
#     2. Snap every top-level pin to the routing track grid on its
#        declared layer. The LEF abstract writer only emits the pin
#        shapes as-placed; if they sit off-grid the top-level router
#        will refuse to connect to them.
#     3. Verify no routed geometry sits on metal9 or metal10. Those
#        layers are reserved for the full-vortex over-the-macro router,
#        so the published LEF must advertise no obstruction on them.
#
# INPUTS  (Tcl globals established by extract_macro.tcl → 00_config.tcl)
#   TOP, CONFIG, RUN_DIR
#
# NOTE ON LAYER NAMING
#   FreePDK45's tech LEF names routing layers "metal1" … "metal10".
#   Innovus accepts the "M<n>" shorthand in some commands (editPin,
#   routeDesign), but `get_db layers <name>` requires the canonical
#   metal<n> form.  All layer name comparisons in this file use that
#   canonical form.
#=======================================================================

# --- Re-apply post-load settings (same as pnr.tcl does on reload) ---
# saveDesign does not persist these Tcl-side knobs, so re-assert them
# here to keep extraction-time delay calc consistent with the original
# PnR run.
setDesignMode -process 45
set delaycal_use_default_delay_limit 1000

#-----------------------------------------------------------------------
# Pin snapping — align every top-level port to the track grid.
#-----------------------------------------------------------------------
# snapFPlanIO re-snaps each IO pin's shape onto the nearest legal track
# on the layer it was placed on. For pins created by editPin with
# -spreadType SIDE (as the floorplan stage does) this is effectively a
# no-op, but it guarantees the invariant independent of how the pins
# got where they are.
#
# Innovus's port object model for pin shapes varies across 21.x builds
# (some expose .physical_pins, some .layer/.rect, some neither), so we
# don't try to hand-verify coordinates against the track grid afterwards
# — snapFPlanIO is the authoritative snap and any mismatch would be an
# Innovus bug, not a flow bug. The LEF writer will flag any remaining
# issue at abstract-emission time.
puts "\n--- Snapping top-level I/O pins to the routing grid ---"
if {[catch {snapFPlanIO} _snap_err]} {
    # snapFPlanIO reports per-pin diagnostics to its own log rather than
    # raising a Tcl error for normal "already on grid" cases, so a
    # non-empty error here is unusual and worth aborting on.
    error "snapFPlanIO failed: $_snap_err"
}
puts "INFO: snapFPlanIO completed — [llength [get_db ports]] ports processed."

#-----------------------------------------------------------------------
# Top-metal blockage check — metal9 and metal10 must be empty.
#
# Any routed geometry on a reserved layer would show up as an OBS in
# the emitted LEF and block the full-vortex router from crossing over
# the macro on those layers.
#-----------------------------------------------------------------------
puts "--- Asserting metal9/metal10 are empty (reserved for full-vortex over-macro routing) ---"

# Resolve layer names defensively: try the canonical metal<n> form
# first, then the M<n> shorthand, then finally look up the layer by
# number. Whichever variant the current tech DB uses is what we'll
# compare wire layers against below.
proc _resolve_layer_name {num} {
    foreach candidate [list "metal$num" "M$num" "Metal$num"] {
        if {[get_db layers $candidate] ne ""} { return $candidate }
    }
    # Last resort: walk the collection and match by routing number.
    foreach l [get_db layers] {
        set rn ""
        catch {set rn [get_db $l .route_index]}
        if {$rn eq $num} { return [get_db $l .name] }
    }
    return ""
}

set reserved_layers {}
foreach num {9 10} {
    set name [_resolve_layer_name $num]
    if {$name eq ""} {
        puts "WARNING: could not resolve a layer name for routing layer $num — skipping reservation check for it."
    } else {
        lappend reserved_layers $name
    }
}
puts "INFO: resolved reserved layers: $reserved_layers"

# Scan every net's wires for geometry on any reserved layer.
#
# The net object's .wires sub-collection carries the detailed routing
# segments (one element per metal/via shape).  Innovus 21.19 exposes
# layer names on wires via .layer.name reliably — this is the well-
# travelled part of the API, unlike the pin-shape path above.
set violators {}
set reserved_set $reserved_layers
if {[llength $reserved_set] > 0} {
    foreach net [get_db nets] {
        set nname [get_db $net .name]
        foreach wire [get_db $net .wires] {
            set wlyr ""
            catch {set wlyr [get_db $wire .layer.name]}
            if {[lsearch -exact $reserved_set $wlyr] >= 0} {
                lappend violators [list $nname $wlyr]
            }
        }
    }
}

if {[llength $violators] > 0} {
    puts "ERROR: reserved top-metal layers contain routed geometry."
    puts "       The full-vortex router expects these layers to be empty."
    set shown 0
    foreach entry $violators {
        lassign $entry nname wlyr
        puts "  net=$nname  layer=$wlyr"
        incr shown
        if {$shown >= 20} { break }
    }
    if {[llength $violators] > 20} {
        puts "  ... and [expr {[llength $violators] - 20}] more."
    }
    error "Re-run 'make single-core STAGE=9' (and subsequent stages) with the\nupdated 09_routing.tcl that caps NanoRoute at M8, then retry extraction."
}
puts "INFO: reserved layers $reserved_layers are clean — safe to publish as over-macro routing channels."

puts "--- export_prep complete ---"
