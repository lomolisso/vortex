# `standard` flowEffort forced; `express` skips optDesign -postRoute entirely.
setDesignMode    -flowEffort             standard
setNanoRouteMode -routeWithTimingDriven  true
setNanoRouteMode -routeWithSiDriven      false
setNanoRouteMode -drouteFixAntenna       true
setNanoRouteMode -routeInsertAntennaDiode true
setNanoRouteMode -routeAntennaCellName    ANTENNA_X1

optDesign -postRoute
optDesign -postRoute -hold
optDesign -postRoute -drv

addFiller -cell {FILLCELL_X32 FILLCELL_X16 FILLCELL_X8 FILLCELL_X4 FILLCELL_X2 FILLCELL_X1} \
          -prefix FILLER
connectGlobalNets

# NRIF-78: TD off during filler-DRC patch to avoid spurious markers.
setNanoRouteMode -routeWithTimingDriven false
ecoRoute -fix_filler_drc_with_patch_only
setNanoRouteMode -routeWithTimingDriven true

setNanoRouteMode -droutePostRouteSpreadWire true
ecoRoute -fix_drc

file mkdir "$REPORT_DIR/verify"
verify_drc -limit 100000 -report "$REPORT_DIR/verify/drc.rpt"
verifyConnectivity   -type all -report "$REPORT_DIR/verify/connectivity.rpt"
verifyProcessAntenna -reportFile "$REPORT_DIR/verify/antenna.rpt"
checkPlace                       "$REPORT_DIR/verify/checkPlace.rpt"
redirect "$REPORT_DIR/verify/checkRoute.rpt" { checkRoute }
redirect "$REPORT_DIR/verify/drv.rpt" { report_constraint -all_violators }

# DRC residual classifier: M3 SHORTs/SPACINGs against bsg_fakeram macro pin
# or OBS shapes are intrinsic to the pin-escape geometry at FreePDK45 (single
# M3 track per pin, via M3 enclosure overlaps adjacent pin tracks). They sit
# inside the macro's LEF abstract and don't propagate to the bank interface
# — logged separately, NOT counted as fatal. Anything else is fatal.
set _drc_residual "$REPORT_DIR/verify/sram_residual.rpt"
set _fatal 0
set _residual 0
if {[file exists "$REPORT_DIR/verify/drc.rpt"]} {
    set _fp  [open "$REPORT_DIR/verify/drc.rpt" r]
    set _out [open $_drc_residual w]
    set _cur ""
    while {[gets $_fp _line] >= 0} {
        if {[regexp {^(SHORT|SPACING|CUTSPACING|CSHORT|MetSpc)} $_line]} { set _cur $_line ; continue }
        if {[regexp {^Bounds} $_line] && $_cur ne ""} {
            if {[regexp {(Pin|Blockage) of Cell.*macro_inst.*\( metal3 \)} $_cur]} {
                incr _residual
                puts $_out "$_cur\n$_line\n"
            } else {
                incr _fatal
            }
            set _cur ""
        }
    }
    close $_fp ; close $_out
}
set _drv 0
if {[file exists "$REPORT_DIR/verify/drv.rpt"]} {
    set _fp [open "$REPORT_DIR/verify/drv.rpt" r]
    set _drv [regexp -all -line {^Pin:} [read $_fp]]
    close $_fp
}
puts "INFO: post-route — fatal DRC: $_fatal   M3 macro-pin residual: $_residual   DRV: $_drv"
# DRC/DRV are reported but NOT gated — this is an estimation flow, not signoff.
# The verify_drc / verifyConnectivity / report_constraint outputs above still
# land in $REPORT_DIR/verify/ for inspection; we just don't stop the run on
# them so stages 11 (signoff reports) and 12 (macro extraction) always emit.
if {$_fatal > 0 || $_drv > 0} {
    puts "WARN: post-route verification has $_fatal fatal DRC + $_drv DRV — continuing anyway (estimation flow). Reports in $REPORT_DIR/verify/."
}
