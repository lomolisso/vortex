# Minimal TOML loader sufficient for config/<cfg>.toml.
#
# Supported syntax:
#   [section] / [section.subsection]      section header (dotted keys OK)
#   [[array]]                              array-of-tables element
#   key = "string" | 1.234 | true|false    scalar / boolean → 1|0
#   key = { a = 1, b = 2 }                 inline table (single-line, no nesting)
#   key = ["a", "b", "c"]                  inline array (single- or multi-line)
#                                          → TCL list
#   # comments / blank lines               ignored
#
# Output (flat array, passed by name):
#   <section>.<key>            scalar or TCL list
#   <section>.<key>.<sub>      inline-table member
#   <array>.<idx>.<key>        array-of-tables element field
#   <array>.count              element count

namespace eval ::toml {

proc _unquote {s} {
    set s [string trim $s]
    if {[regexp {^"(.*)"$} $s _ inner]} { return $inner }
    if {[regexp {^'(.*)'$} $s _ inner]} { return $inner }
    return $s
}

proc _scalar {s} {
    set s [string trim $s]
    if {$s eq "true"}  { return 1 }
    if {$s eq "false"} { return 0 }
    return [_unquote $s]
}

proc _inline {body} {
    set out {}
    foreach pair [split $body ","] {
        set pair [string trim $pair]
        if {$pair eq ""} continue
        if {![regexp {^(\S+?)\s*=\s*(.*)$} $pair _ k v]} continue
        lappend out $k [_scalar $v]
    }
    return $out
}

proc load {path arr_name} {
    upvar 1 $arr_name out

    set fh [open $path r]
    set text [read $fh]
    close $fh

    set section ""
    set array_name ""
    set array_idx -1
    set pending ""             ;# multi-line inline-array accumulator
    set pending_depth 0

    foreach raw [split $text "\n"] {
        regsub {\s+#[^"]*$} $raw {} line
        set line [string trim $line]
        if {$pending eq "" && ($line eq "" || [string index $line 0] eq "#")} continue

        if {$pending ne ""} {
            append pending " " $line
            incr pending_depth [expr {[regexp -all {\[} $line] - [regexp -all {\]} $line]}]
            if {$pending_depth > 0} continue
            set line $pending
            set pending ""
            set pending_depth 0
        } else {
            # Open a multi-line array only on `key = [ ...` (not on [section] or [[array]]).
            set _open  [regexp -all {\[} $line]
            set _close [regexp -all {\]} $line]
            if {$_open > $_close && [regexp {^[A-Za-z0-9_\-]+\s*=\s*\[} $line]} {
                set pending $line
                set pending_depth [expr {$_open - $_close}]
                continue
            }
        }

        if {[regexp {^\[\[([A-Za-z0-9_.\-]+)\]\]$} $line _ name]} {
            if {$name ne $array_name} { set array_name $name; set array_idx 0 } \
                                       else { incr array_idx }
            set section "$name.$array_idx"
            set out($name.count) [expr {$array_idx + 1}]
            continue
        }

        if {[regexp {^\[([A-Za-z0-9_.\-]+)\]$} $line _ name]} {
            set section $name
            set array_name ""
            set array_idx -1
            continue
        }

        if {[regexp {^([A-Za-z0-9_\-]+)\s*=\s*(.*)$} $line _ key value]} {
            set value [string trim $value]
            if {[regexp {^\{(.*)\}$} $value _ body]} {
                foreach {k v} [_inline $body] { set out($section.$key.$k) $v }
                continue
            }
            if {[regexp {^\[(.*)\]$} $value _ body]} {
                set lst {}
                foreach _e [split $body ","] {
                    set _e [string trim $_e]
                    if {$_e ne ""} { lappend lst [_scalar $_e] }
                }
                set out($section.$key) $lst
                continue
            }
            set out($section.$key) [_scalar $value]
        }
    }
}

}
package provide toml_load 1.0
