source "$SCRIPT_DIR/scripts/floorplan/shared/util.tcl"

if {$FP_RECIPE eq ""} {
    set _fp "$SCRIPT_DIR/scripts/floorplan/shared/leaf.tcl"
} else {
    set _fp "$SCRIPT_DIR/scripts/floorplan/$FP_RECIPE.tcl"
}
if {![file exists $_fp]} { error "Floorplan recipe not found: $_fp" }
puts "INFO: floorplan recipe → $_fp"
source $_fp
