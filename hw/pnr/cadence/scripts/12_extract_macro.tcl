# Hard-macro extraction → LEF + LIB + DB for the parent build.
# Env: EXPORT_DIR (default $LIBS_DIR/core/$CONFIG), MACRO_CELL (default $TOP).
# Opt-out via design.extract_macro = false.

set _extract [expr {[info exists cfg(design.extract_macro)] ? $cfg(design.extract_macro) : 1}]
if {!$_extract} {
    puts "INFO: $CONFIG has design.extract_macro=false; skipping macro extraction."
    set ::STAGE_READONLY 1
    return
}

if {![info exists ::env(EXPORT_DIR)] || $::env(EXPORT_DIR) eq ""} {
    set ::env(EXPORT_DIR) "$LIBS_DIR/core/$CONFIG"
}
if {![info exists ::env(MACRO_CELL)] || $::env(MACRO_CELL) eq ""} {
    set ::env(MACRO_CELL) $TOP
}
if {![info exists ::env(VIEW_NAME)] || $::env(VIEW_NAME) eq ""} {
    set ::env(VIEW_NAME) "typView"
}
set EXPORT_DIR $::env(EXPORT_DIR)
set MACRO_CELL $::env(MACRO_CELL)
file mkdir $EXPORT_DIR

foreach sub {export_prep export_lef export_etm} {
    puts "\n--- 12: $sub ---"
    source "$SCRIPT_DIR/scripts/utils/${sub}.tcl"
}

# LD_PRELOAD: RHEL libkrb5 ABI mismatch in lc_shell.
set _lib "$EXPORT_DIR/${MACRO_CELL}.lib"
set _db  "$EXPORT_DIR/${MACRO_CELL}.db"
set _ctl "$SCRIPT_DIR/../../syn/synopsys/compile_lib_to_db.tcl"
if {![file exists $_lib]} { error "Liberty not found: $_lib (export_etm must run before db compile)." }
if {![file exists $_ctl]} { error "lc_shell control script not found: $_ctl" }
puts "\n--- 12: lib → db ($_db) ---"
if {[catch {
    exec >@stdout 2>@stdout env \
        "LIB_SRC=$_lib" "LIB_DB_OUT=$_db" "LIB_NAME=$MACRO_CELL" \
        "LD_PRELOAD=/lib64/libcrypto.so.1.1 /lib64/libk5crypto.so.3 /lib64/libtinfo.so.5" \
        lc_shell -batch -f $_ctl
} _err]} {
    error "lc_shell compile_lib_to_db failed: $_err"
}
if {![file exists $_db] || [file size $_db] < 1024} {
    error "lc_shell finished but $_db is missing or tiny."
}

set _gds_src "$RESULTS_DIR/${TOP}.gds"
set _gds_dst "$EXPORT_DIR/${MACRO_CELL}.gds"
if {[file exists $_gds_src]} {
    file copy -force $_gds_src $_gds_dst
    puts "  GDS: $_gds_dst"
}

puts "\nMacro extraction complete."
puts "  LEF: $EXPORT_DIR/${MACRO_CELL}.lef"
puts "  LIB: $_lib"
puts "  DB : $_db"
