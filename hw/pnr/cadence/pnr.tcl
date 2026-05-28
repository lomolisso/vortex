# Vortex PnR gateway. Modes (env):
#   (no STAGE)              run all stages
#   STAGE=N                 run N only (loads ckpt N-1)
#   STAGE=N CONTINUE=1      run N..final in one session
# Env: CONFIG, LIBS_DIR, SYN_DIR, EFFORT, STAGE, CONTINUE, BATCH, SKIP_POWER.

set SCRIPT_DIR [file normalize [file dirname [info script]]]

set stages {
    01_init_design   02_power_connect  03_floorplan      04_power_plan
    05_placement     06_pre_cts_opt    07_cts            08_post_cts_opt
    09_routing       10_post_route_opt 11_signoff        12_extract_macro
}

set continue_flag [expr {[info exists ::env(CONTINUE)] && $::env(CONTINUE) eq "1"}]
if {![info exists ::env(STAGE)] || $::env(STAGE) eq ""} {
    set start_idx 0
    set end_idx   [expr {[llength $stages] - 1}]
} else {
    set req $::env(STAGE)
    # `scan %d` to bypass Tcl's octal parser on leading-zero forms like "03".
    if {[string first "_" $req] < 0} {
        if {[scan $req %d _n] != 1} { error "STAGE='$req' not a valid stage number or full name." }
        set _pad [format "%02d" $_n]
        set req ""
        foreach s $stages { if {[string match "${_pad}_*" $s]} { set req $s; break } }
        if {$req eq ""} { error "No stage matches STAGE=$::env(STAGE)" }
    }
    set start_idx [lsearch -exact $stages $req]
    if {$start_idx < 0} { error "STAGE='$req' invalid. Valid: $stages" }
    set end_idx [expr {$continue_flag ? [llength $stages] - 1 : $start_idx}]
}

puts "\n--- Setup: 00_config ---"
source "$SCRIPT_DIR/scripts/00_config.tcl"

if {$start_idx > 0} {
    set _loaded ""
    for {set i [expr {$start_idx - 1}]} {$i >= 0} {incr i -1} {
        set _ckpt "$RUN_DIR/checkpoints/after_[lindex $stages $i]"
        if {[file exists $_ckpt]} {
            set _loaded [lindex $stages $i]
            puts "\n--- Loading checkpoint: after_${_loaded} ---"
            source $_ckpt
            break
        }
    }
    if {$_loaded eq ""} {
        error "No checkpoint found before [lindex $stages $start_idx]. Run earlier stages first."
    }
    _apply_design_modes
    # connectGlobalNets must be in scope when resuming past stage 02.
    if {$start_idx > [lsearch -exact $stages 02_power_connect]} {
        source "$SCRIPT_DIR/scripts/02_power_connect.tcl"
    }
}

for {set i $start_idx} {$i <= $end_idx} {incr i} {
    set stage [lindex $stages $i]
    unset -nocomplain ::STAGE_READONLY

    puts "\n--- Stage: $stage ---"
    source "$SCRIPT_DIR/scripts/${stage}.tcl"

    if {[info exists ::STAGE_READONLY] && $::STAGE_READONLY} {
        puts "--- '$stage' read-only; no checkpoint save ---"
        continue
    }
    set _ckpt "$RUN_DIR/checkpoints/after_${stage}"
    file mkdir [file dirname $_ckpt]
    puts "--- Saving checkpoint: $_ckpt ---"
    saveDesign $_ckpt
}

if {$end_idx == $start_idx} {
    puts "\nStage '[lindex $stages $start_idx]' complete."
} else {
    puts "\nStages '[lindex $stages $start_idx]'..'[lindex $stages $end_idx]' complete."
}
if {[info exists ::env(BATCH)] && $::env(BATCH) eq "1"} {
    puts "BATCH=1 — exiting Innovus.\n"
    exit 0
}
puts "Innovus GUI active — inspect, then close when done.\n"
