#=======================================================================
# compile_lib_to_db.tcl — dc_shell helper that precompiles a Liberty
# timing model (.lib) into Synopsys's binary .db format.
#
# Why this exists
#   Design Compiler's link_library expects .db (its internal compiled
#   Liberty representation). Our bottom-up flow ships VX_socket_top as
#   a plain .lib (emitted by Innovus's do_extract_model) and we need a
#   .db for DC to treat the socket as a hard-macro blackbox during the
#   1c8n4w4t synthesis run.
#
# Why dc_shell (and not lc_shell)
#   dc_shell is the tool everyone on this course machine already has
#   working. lc_shell (Library Compiler) would do the same job but
#   needs its own license bundle and adds one more moving part to
#   document. `read_lib` + `write_lib -format db` works inside dc_shell
#   and produces a .db that the target_library/link_library flow loads
#   identically to the vendor-supplied stdcells.db / sram_*.db.
#
# Why this script is parameterized via env vars (not command-line args)
#   dc_shell's -x / -f idioms make passing bare file paths awkward
#   (path escaping, embedded quoting). Env vars keep the Makefile rule
#   one short line, and let the same helper be reused for any future
#   hard macro the project might generate without touching this Tcl.
#
# Required env vars
#   LIB_SRC     — absolute path to the input .lib file
#   LIB_DB_OUT  — absolute path of the .db file to produce
#
# Optional env var
#   LIB_NAME    — logical library name to pass to write_lib. Defaults
#                 to the basename of LIB_SRC (without the .lib suffix),
#                 which matches what Innovus's do_extract_model puts in
#                 the Liberty header (e.g. VX_socket_top).
#=======================================================================

if {![info exists env(LIB_SRC)] || ![info exists env(LIB_DB_OUT)]} {
    error "compile_lib_to_db.tcl requires env vars LIB_SRC and LIB_DB_OUT"
}

set _src [string trim $env(LIB_SRC)]
set _out [string trim $env(LIB_DB_OUT)]

if {![file exists $_src]} {
    error "LIB_SRC does not exist: $_src"
}

if {[info exists env(LIB_NAME)] && [string length [string trim $env(LIB_NAME)]] > 0} {
    set _name [string trim $env(LIB_NAME)]
} else {
    set _name [file rootname [file tail $_src]]
}

# Make sure the destination directory exists; failing here with a clear
# Tcl error is friendlier than letting write_lib emit a cryptic file-
# system error later in the run.
file mkdir [file dirname $_out]

puts "=================================================================="
puts " compile_lib_to_db: $_src"
puts "                  → $_out"
puts "                    (library name: $_name)"
puts "=================================================================="

# read_lib names the loaded library after the `library(<name>)` header
# inside the .lib. write_lib needs to be told which in-memory library
# to serialize: we pass $_name so LIB_NAME can override it if a future
# macro's Liberty header ever drifts from its filename.
read_lib $_src
write_lib -format db -output $_out $_name

# Best-effort post-check: write_lib can (in some DC releases) succeed
# with a zero- or tiny-byte file if the .lib had a malformed cell.
# 1 KiB is a generous floor — even a trivial stdcell .db is larger.
if {![file exists $_out]} {
    error "write_lib finished but $_out is missing"
}
if {[file size $_out] < 1024} {
    error "write_lib produced suspiciously small $_out ([file size $_out] bytes); inspect the log for read_lib warnings"
}

puts [format "INFO: wrote %s (%.1f KiB)" $_out [expr {[file size $_out] / 1024.0}]]

exit
