#=======================================================================
# Export LEF — physical abstract (.lef) for the routed single-core
#
# PURPOSE
#   Package the routed single-core database into a LEF macro model that
#   the full-vortex top-level router can instantiate as a hard macro.
#   The emitted file contains:
#
#     · One MACRO block named $TOP (VX_socket_top by default).
#     · PIN sections for every top-level signal port, located on metal4
#       / metal5 (as placed by 03_floorplan_single-core.tcl → editPin).
#     · PIN sections for the VDD/VSS power/ground nets, sampled from
#       the metal7 stripe stubs and metal8 ring sides laid down by
#       04_power_plan.tcl.
#     · OBS sections describing routing obstructions on metal1..metal8,
#       so the top-level router avoids shorts. metal9 and metal10 are
#       intentionally absent — they are reserved for over-the-macro
#       routing (verified in export_prep.tcl).
#     · Cut-layer (via) obstructions, enforced with minSpacing rules so
#       the top-level router must keep vias outside the macro's via
#       zones.
#
# COMMAND CHOICE: write_lef_abstract
#   Innovus 21.19 (doc verified against
#   $INNOVUS211/doc/innovusTCR/write_lef_abstract.html) exposes the
#   LEF abstract writer as snake_case `write_lef_abstract`.  The flags
#   below are taken verbatim from that reference — `write_lef_macromodel`
#   and `writeLefAbstract` are not shipped in this build.
#
# INPUTS  (Tcl globals)
#   SCRIPT_DIR — directory containing this script (from 00_config.tcl)
#   TOP        — top-level cell to wrap (VX_socket_top for single-core)
#   CONFIG     — design config name ("single-core")
#=======================================================================

# --- Destination: hw/pnr/cadence/export/<config>/${TOP}.lef ---
# SCRIPT_DIR points at the pnr/cadence directory (set in extract_macro.tcl
# or pnr.tcl), so export lands in the canonical location that
# 01_init_design.tcl / mmmc.tcl already look in first.
set EXPORT_ROOT "$SCRIPT_DIR/export"
set EXPORT_DIR  "$EXPORT_ROOT/$CONFIG"
file mkdir $EXPORT_DIR

set LEF_OUT "$EXPORT_DIR/${TOP}.lef"
puts "\n--- Writing LEF abstract → $LEF_OUT ---"

# -----------------------------------------------------------------------
# write_lef_abstract flags (as documented in Innovus 21.19 TCR)
#
#   -stripePin
#       Publish top-level PG stripes/rings as LEF PIN shapes so the
#       full-vortex PDN can hook in on metal7/metal8 at any stripe
#       crossing. Required as a prerequisite for -PGpinLayers.
#
#   -PGpinLayers {metal7 metal8}
#       Restrict the published PG pins to the layers 04_power_plan.tcl
#       actually uses (metal7 for vertical stripes, metal8 for ring
#       H-sides). Keeps the LEF free of spurious low-metal PG shapes.
#       Note: in Innovus 21.19 the flag is lowercase 'p' in 'pin'
#       (-PGpinLayers), unlike the editPin layer-option camelCase.
#
#   -specifyTopLayer 8
#       Seal the advertised obstruction ceiling at metal8. Belt-and-
#       suspenders with the routing cap in 09_routing.tcl: even if a
#       future run forgets the cap, the LEF still promises metal9 /
#       metal10 are free routing channels for the top level.
#
#   -extractBlockObs
#       Also extract OBS from any CLASS BLOCK / RING / PAD macros
#       instantiated inside the design. This picks up obstructions that
#       belong to nested hard macros (e.g. bsg_fakeram SRAMs) so the
#       top-level router respects their internal blockages.
#
#   -cutObsMinSpacing
#       When creating OBS cut-outs around PIN shapes, use the tech's
#       min-width rule for the spacing. This is the single-token flag
#       in the 21.19 TCR; there is no separate "-cutObs <mode>" form in
#       this release.
#
# NOTE: the `-extractPowerPGPin` flag referred to in some Cadence
# tutorials does not exist as a separate option in 21.19.  Its function
# is covered by the -stripePin / -PGpinLayers pair above (for top-level
# stripes) and would be spelled -extractBlockPGPinLayers if we also
# needed to republish PG pins belonging to nested block macros.
# -----------------------------------------------------------------------

write_lef_abstract \
    -stripePin \
    -PGpinLayers      {metal7 metal8} \
    -specifyTopLayer  8 \
    -extractBlockObs \
    -cutObsMinSpacing \
    $LEF_OUT

puts "INFO: LEF abstract written: $LEF_OUT"

# --- Quick self-check: file exists and is non-empty. ---
if {![file exists $LEF_OUT] || [file size $LEF_OUT] < 512} {
    error "LEF abstract write appears to have failed (missing or tiny file: $LEF_OUT)."
}
puts [format "INFO: LEF size: %.1f KiB" [expr {[file size $LEF_OUT] / 1024.0}]]
