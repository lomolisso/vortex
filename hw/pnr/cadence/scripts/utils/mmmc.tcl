# Single-corner MMMC: NanGate45 + bsg_fakeram only ship TT @ 1.0V/25C.
# OCV derate (stage 01) gives hold opt a budget.

set _lib_list [list "$LIBS_DIR/stdcells.lib"]

if {[info exists cfg(macros.count)]} {
    for {set _i 0} {$_i < $cfg(macros.count)} {incr _i} {
        set _name $cfg(macros.$_i.name)
        if {[info exists cfg(macros.$_i.lib_path)]} {
            set _lib $cfg(macros.$_i.lib_path)
        } elseif {[info exists cfg(macros.$_i.lef_path)]} {
            set _lib [regsub {\.lef$} $cfg(macros.$_i.lef_path) ".lib"]
        } else {
            error "macros\[$_i\] '$_name' has no lef_path or lib_path"
        }
        if {![string match "/*" $_lib]} { set _lib "$LIBS_DIR/$_lib" }
        if {![file exists $_lib]} {
            error "Macro Liberty not found for $_name: $_lib"
        }
        lappend _lib_list $_lib
    }
}

create_library_set     -name typical -timing $_lib_list
create_constraint_mode -name func    -sdc_files [list $SDC_FILE]

# FreePDK45 ships no captable; Innovus synthesises one internally.
set _captbl "$LIBS_DIR/freepdk45.captable"
if {[file exists $_captbl]} {
    create_rc_corner    -name typRC   -cap_table $_captbl
    create_delay_corner -name typDC   -library_set typical -rc_corner typRC
} else {
    create_delay_corner -name typDC   -library_set typical
}
create_analysis_view   -name typView -constraint_mode func -delay_corner typDC
set_analysis_view -setup {typView} -hold {typView}
