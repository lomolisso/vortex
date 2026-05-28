# Innovus 21.19+ uses do_extract_model; older uses extract_model.

set VIEW_NAME  [expr {[info exists ::env(VIEW_NAME)]  ? $::env(VIEW_NAME)  : "typView"}]
set MACRO_CELL [expr {[info exists ::env(MACRO_CELL)] ? $::env(MACRO_CELL) : $TOP}]
set EXPORT_DIR [expr {[info exists ::env(EXPORT_DIR)] ? $::env(EXPORT_DIR) : "$LIBS_DIR/core/$CONFIG"}]
file mkdir $EXPORT_DIR

set known_views [get_db analysis_views .name]
foreach v $VIEW_NAME {
    if {[lsearch -exact $known_views $v] < 0} {
        error "Analysis view '$v' not in MMMC. Known: $known_views"
    }
}

set etm_cmd ""
foreach c {do_extract_model extract_model} {
    if {[llength [info commands $c]] > 0} { set etm_cmd $c; break }
}
if {$etm_cmd eq ""} { error "No ETM command available (do_extract_model / extract_model)." }

# Single-view → $MACRO_CELL.lib; multi-view → per-view files + alias.
set primary_lib ""
foreach _view $VIEW_NAME {
    set_analysis_view -setup $_view -hold $_view
    set _out [expr {[llength $VIEW_NAME] == 1 ? "$EXPORT_DIR/${MACRO_CELL}.lib" : "$EXPORT_DIR/${MACRO_CELL}_${_view}.lib"}]
    puts "\n--- $etm_cmd view=$_view → $_out ---"
    if {[catch { $etm_cmd -cell_name $MACRO_CELL -lib_name $MACRO_CELL -view $_view -pg $_out } _err]} {
        error "$etm_cmd failed for view '$_view': $_err"
    }
    if {![file exists $_out]} { error "$etm_cmd returned but did not produce $_out" }
    puts [format "INFO: LIB written (%.1f KiB): $_out" [expr {[file size $_out] / 1024.0}]]
    if {$primary_lib eq ""} { set primary_lib $_out }
}

if {[llength $VIEW_NAME] > 1} {
    set _alias "$EXPORT_DIR/${MACRO_CELL}.lib"
    file copy -force $primary_lib $_alias
    puts "INFO: default alias: $_alias → [file tail $primary_lib]"
}
