# Liberty .lib → Synopsys .db via lc_shell.
# Env: LIB_SRC, LIB_DB_OUT. Library name is discovered from the loaded library
# (different generators tag it differently — bare `<m>` for bsg_fakeram, etc.).

foreach _v {LIB_SRC LIB_DB_OUT} {
    if {![info exists env($_v)]} { error "$_v not set." }
}

set _src [string trim $env(LIB_SRC)]
set _out [string trim $env(LIB_DB_OUT)]
if {![file exists $_src]} { error "LIB_SRC does not exist: $_src" }

file mkdir [file dirname $_out]
puts "=== compile_lib_to_db: $_src → $_out ==="

proc _lib_names {} {
    set out [list]
    set _c [get_libs -quiet *]
    if {$_c eq ""} { return $out }
    foreach_in_collection _l $_c { lappend out [get_object_name $_l] }
    return $out
}

set _before [_lib_names]
read_lib $_src
set _after  [_lib_names]

# Diff vs _before so multi-compile sessions stay locked onto the right lib.
set _new [list]
foreach _l $_after {
    if {[lsearch -exact $_before $_l] < 0} { lappend _new $_l }
}
if {[llength $_new] == 0} {
    error "read_lib produced no new library from $_src"
}
if {[llength $_new] > 1} {
    puts "WARN: $_src loaded [llength $_new] libraries; using the first."
}
set _name [lindex $_new 0]
puts "INFO: library name in $_src is '$_name'"

write_lib -format db -output $_out $_name

# Detect silent failure: a real .db is well above 1 KiB.
if {![file exists $_out]}     { error "write_lib finished but $_out is missing" }
if {[file size $_out] < 1024} { error "write_lib produced suspiciously small $_out ([file size $_out] bytes)" }

puts [format "INFO: wrote %s (%.1f KiB)" $_out [expr {[file size $_out] / 1024.0}]]
exit
