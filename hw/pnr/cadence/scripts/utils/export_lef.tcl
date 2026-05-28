switch -- $DESIGN_KIND {
    leaf    { set TOP_LAYER $TOP_RLAYER ; set PG_LAYERS {metal3 metal4} }
    socket  { set TOP_LAYER $TOP_RLAYER ; set PG_LAYERS {metal5 metal6} }
    default { error "export_lef: unsupported DESIGN_KIND='$DESIGN_KIND' (expected leaf|socket)" }
}

set EXPORT_DIR [expr {[info exists ::env(EXPORT_DIR)] ? $::env(EXPORT_DIR) : "$LIBS_DIR/core/$CONFIG"}]
file mkdir $EXPORT_DIR

set LEF_OUT "$EXPORT_DIR/${TOP}.lef"
puts "\n--- LEF abstract → $LEF_OUT (top=M$TOP_LAYER, PG=$PG_LAYERS) ---"

write_lef_abstract \
    -stripePin \
    -PGpinLayers     $PG_LAYERS \
    -specifyTopLayer $TOP_LAYER \
    -extractBlockObs \
    -cutObsMinSpacing \
    $LEF_OUT

if {![file exists $LEF_OUT] || [file size $LEF_OUT] < 512} {
    error "LEF abstract write failed (missing or tiny: $LEF_OUT)."
}
puts [format "INFO: LEF written (%.1f KiB): $LEF_OUT" [expr {[file size $LEF_OUT] / 1024.0}]]
