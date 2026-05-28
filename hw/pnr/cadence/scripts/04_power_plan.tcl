# PG above signal per kind:
#   leaf    → ring M3/M4, M4 V straps              (4-metal stack)
#   socket  → ring M5/M6, M4 V + M5 H straps       (6-metal stack)
#   cluster → ring M9/M10, M7 V + M8 H straps      (10-metal stack)
# addRing/addStripe must run BEFORE sroute -connect corePin, else M1 rails
# are left open (IMPVFC-200 in verifyConnectivity).

switch -- $DESIGN_KIND {
    cluster {
        addRing -follow core \
            -offset  {top 2 bottom 2 left 2 right 2} \
            -spacing {top 2 bottom 2 left 2 right 2} \
            -width   {top 2 bottom 2 left 2 right 2} \
            -layer   {top metal9 bottom metal9 left metal10 right metal10} \
            -nets    { VSS VDD }
        addStripe -nets { VSS VDD } -layer metal7 -direction vertical \
                  -width 1.6 -spacing 1.6 -set_to_set_distance 50
        addStripe -nets { VSS VDD } -layer metal8 -direction horizontal \
                  -width 1.6 -spacing 1.6 -set_to_set_distance 50
        puts "INFO: cluster PDN — ring M9/M10, V-stripes M7 + H-stripes M8 @ 50 µm."
    }
    socket {
        addRing -follow core \
            -offset  {top 2 bottom 2 left 2 right 2} \
            -spacing {top 2 bottom 2 left 2 right 2} \
            -width   {top 2 bottom 2 left 2 right 2} \
            -layer   {top metal5 bottom metal5 left metal6 right metal6} \
            -nets    { VSS VDD }
        addStripe -nets { VSS VDD } -layer metal4 -direction vertical \
                  -width 1.6 -spacing 1.6 -set_to_set_distance 30
        addStripe -nets { VSS VDD } -layer metal5 -direction horizontal \
                  -width 1.6 -spacing 1.6 -set_to_set_distance 30
        puts "INFO: socket PDN — ring M5/M6, V-stripes M4 + H-stripes M5 @ 30 µm."
    }
    leaf {
        # Ring on M3/M4 keeps PG off M1/M2 (signal pin layers).
        addRing -follow core \
            -offset  {top 1 bottom 1 left 1 right 1} \
            -spacing {top 1 bottom 1 left 1 right 1} \
            -width   {top 1 bottom 1 left 1 right 1} \
            -layer   {top metal3 bottom metal3 left metal4 right metal4} \
            -nets    { VSS VDD }
        if {[llength [get_db insts -if {.base_cell.class == block}]] == 0} {
            addStripe -nets { VSS VDD } -layer metal4 -direction vertical \
                      -width 1 -spacing 1 -set_to_set_distance 30
            puts "INFO: leaf PDN — ring M3/M4 + M4 V-straps @ 30 µm."
        } else {
            puts "INFO: leaf PDN — ring M3/M4 only (sroute handles PG to macro pins)."
        }
    }
}

# IMPSR-1254: append blockPin only if macros exist.
setSrouteMode -extendNearestTarget true
set _sr_connect { corePin }
set _sr_targets { ring stripe }
if {[llength [get_db insts -if {.base_cell.class == block}]] > 0} {
    lappend _sr_connect blockPin
    lappend _sr_targets blockpin
}
sroute -connect $_sr_connect \
       -nets { VDD VSS } \
       -allowJogging 1 \
       -allowLayerChange 1 \
       -corePinTarget $_sr_targets
