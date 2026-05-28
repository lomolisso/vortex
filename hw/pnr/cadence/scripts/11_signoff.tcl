# Post-route signoff. Read-only. Env: SKIP_POWER=1, MIN_FFS=N (default 1).

set SKIP_POWER [expr {[info exists ::env(SKIP_POWER)] && $::env(SKIP_POWER) eq "1"}]
set MIN_FFS    [expr {[info exists ::env(MIN_FFS)] ? int($::env(MIN_FFS)) : 1}]

puts "\n--- 11a: chip-wide timing ---"
redirect "$REPORT_DIR/timing_setup.rpt" { report_timing -nworst 20 -max_paths 20 -path_type full }
redirect "$REPORT_DIR/timing_hold.rpt"  { report_timing -nworst 20 -max_paths 20 -path_type full -late }

puts "\n--- 11b: deep area + power ---"
if {[catch { redirect "$REPORT_DIR/area_deep.rpt" { report_area -depth 64 } } _err]} {
    puts "  \[warn\] report_area -depth 64 failed ($_err) — fallback."
    redirect "$REPORT_DIR/area_deep.rpt" { report_area }
}
if {!$SKIP_POWER} {
    redirect "$REPORT_DIR/power_deep.rpt" { report_power -hierarchy 20 }
}

puts "\n--- 11c: per-module intrinsic timing ---"
file mkdir "$REPORT_DIR/timing"

set _all_ffs [get_db insts -if {.base_cell.is_sequential}]
puts "    sequential leaves: [llength $_all_ffs]"

proc _sanitize {s} { return [string map {. _ / _ \[ _ \] _ \\ _ { } _} $s] }

set _ix [open "$REPORT_DIR/hier_index.txt" w]
puts $_ix "# Vortex PnR per-module timing index — config: $CONFIG  top: $TOP"
puts $_ix [format "%-8s  %s  ->  %s" "FFs" "hier_path" "report"]

set _emitted 0 ; set _skipped 0
foreach _h [get_db hinsts] {
    set _hname [get_db $_h .name]
    if {$_hname eq ""} continue
    set _prefix "${_hname}/"
    set _ffs [list]
    foreach _ff $_all_ffs {
        if {[string equal -length [string length $_prefix] $_prefix [get_db $_ff .name]]} {
            lappend _ffs $_ff
        }
    }
    set _nff [llength $_ffs]
    if {$_nff < $MIN_FFS} { incr _skipped; continue }

    set _rpt "timing/[_sanitize $_hname].rpt"
    if {[catch {
        redirect "$REPORT_DIR/$_rpt" {
            report_timing -from $_ffs -to $_ffs -nworst 1 -max_paths 1 -path_type full
        }
    } err]} {
        puts "  \[warn\] $_hname: report_timing failed — $err"
        continue
    }
    puts $_ix [format "%-8d  %s  ->  %s" $_nff $_hname $_rpt]
    incr _emitted
    if {$_emitted % 50 == 0} { puts "      ... $_emitted reports emitted" }
}
close $_ix
puts "    emitted $_emitted reports; skipped $_skipped (< $MIN_FFS FFs)"

puts "\n--- 11d: SPEF + SDF ---"
if {!$SKIP_POWER} {
    if {[catch {write_parasitics -spef_file "$REPORT_DIR/post_route.spef"} msg]} {
        puts "  \[info\] write_parasitics unavailable ($msg) — trying rcOut."
        catch {rcOut -spef "$REPORT_DIR/post_route.spef"}
    }
}
catch {write_sdf "$REPORT_DIR/post_route.sdf" -version 3.0}

puts "\n--- 11e: deliverables ---"

set _gds_map_path ""
foreach _cand [list "$LIBS_DIR/innovus.map" "/class/ece425/innovus.map"] {
    if {[file exists $_cand]} { set _gds_map_path $_cand; break }
}

# streamOut -merge needs stdcells.gds + per-macro .gds for a complete GDS.
set _merge_gds {}
foreach _cand [list "$LIBS_DIR/stdcells.gds"] {
    if {[file exists $_cand]} { lappend _merge_gds $_cand }
}
if {[info exists cfg(macros.count)]} {
    for {set _i 0} {$_i < $cfg(macros.count)} {incr _i} {
        if {![info exists cfg(macros.$_i.lef_path)]} continue
        set _gds [regsub {\.lef$} $cfg(macros.$_i.lef_path) ".gds"]
        if {![string match "/*" $_gds]} { set _gds "$LIBS_DIR/$_gds" }
        if {[file exists $_gds]} { lappend _merge_gds $_gds }
    }
}

set _streamOut_args [list "$RESULTS_DIR/${TOP}.gds" -units 2000]
if {$_gds_map_path ne ""} { lappend _streamOut_args -mapFile $_gds_map_path }
if {[llength $_merge_gds] > 0} { lappend _streamOut_args -merge $_merge_gds }
puts "INFO: streamOut merge list ([llength $_merge_gds]): $_merge_gds"
streamOut {*}$_streamOut_args

saveNetlist "$RESULTS_DIR/${TOP}_pnr.v"
defOut      "$RESULTS_DIR/${TOP}.def"
saveDesign  "$RESULTS_DIR/${TOP}"

if {[info exists ::env(DISPLAY)] && $::env(DISPLAY) ne ""} {
    win
    fit
    uiSetTool select
    catch {displaySnapshot -type PNG -resolution 2048 -file "$REPORT_DIR/dieshot.png"}
}

set ::STAGE_READONLY 1
