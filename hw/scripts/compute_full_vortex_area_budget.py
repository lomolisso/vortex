#!/usr/bin/env python3
"""
compute_full_vortex_area_budget.py
──────────────────────────────────
Floorplan area budget for the `full-vortex` PnR run.

At the full-GPGPU level the four VX_core blocks are already laid out and
packaged as a .lef/.lib pair (`VX_socket_top.lef`, `VX_socket_top.lib`),
produced by the earlier single-core PnR step. So the area we need to
allocate at the top is:

    4 × core_area  +  L2 SRAM macro area  +  top-level stdcells

Where:
  · core_area    — Liberty 'area' of cell VX_socket_top (read from
                   <libs_dir>/VX_socket_top.lib).
  · L2 macros    — 4 banks × 4 ways = 16 tag + 16 data bsg_fakeram macros:
                     sram_256x24_1r1w   (tag arrays)
                     sram_256x512_1rw   (data arrays)
                   Dimensions read from each LEF's SIZE statement.
  · stdcells     — 'Total cell area' from the full-vortex DC area report.
                   After core blackboxing this covers only the L2 cache
                   controller + Vortex top-level glue; pre-blackbox it
                   is larger, which simply yields a conservative budget.

The script emits the same `floorplan_budget.tcl` variables
(FLOORPLAN_W / FLOORPLAN_H / FLOORPLAN_MARGIN) that the per-config stage-3
floorplan scripts already consume — so it is a drop-in replacement for
the single-core variant at the `full-vortex` target.

Default paths (relative to this script's directory)
────────────────────────────────────────────────────
  --runs_dir  ../syn/synopsys/runs   (area.rpt at <runs_dir>/full-vortex/reports/area.rpt)
  --libs_dir  ../syn/synopsys/libs   (VX_socket_top.lib + SRAM .lef files)

Usage
─────
  python3 compute_full_vortex_area_budget.py \\
      --emit_tcl runs/full-vortex/floorplan_budget.tcl
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Reuse the low-level parsers/emitter from compute_area_budget.py.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from compute_area_budget import (  # noqa: E402
    parse_dc_area_report,
    parse_lef_macro_size,
    _emit_tcl,
    hr,
    row,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixed full-vortex parameters — matches hw/syn/synopsys/Makefile's
# DEFINES_full-vortex (NUM_CLUSTERS=1, NUM_CORES=4, NUM_WARPS=4, NUM_THREADS=4,
# SOCKET_SIZE=1, L2_ENABLE, L2_CACHE_SIZE=131072).
#
# With the default L2_NUM_BANKS=4 and L2_NUM_WAYS=4 in VX_config.vh, the L2
# cache contains 16 tag arrays and 16 data arrays.  Both variants live in
# libs/ as bsg_fakeram macros.
# ──────────────────────────────────────────────────────────────────────────────

CONFIG          = "full-vortex"
NUM_CORES       = 4
L2_NUM_BANKS    = 4
L2_NUM_WAYS     = 4
L2_TAG_LIB      = "sram_256x24_1r1w"
L2_DATA_LIB     = "sram_256x512_1rw"
CORE_LIB_NAME   = "VX_socket_top"
CORE_CELL_NAME  = "VX_socket_top"


# ──────────────────────────────────────────────────────────────────────────────
# Liberty parser — extract 'area' from a named cell block
# ──────────────────────────────────────────────────────────────────────────────

def parse_lib_cell_area(lib_path: str, cell_name: str) -> float:
    """
    Return the `area` attribute (µm², Liberty convention) of the named cell
    inside a Liberty (.lib) file.

    The .lib file is expected to contain at least one block of the form:
        cell (VX_socket_top) {
            ...
            area : 1234567.89;
            ...
        }
    """
    cell_re = re.compile(
        r'cell\s*\(\s*' + re.escape(cell_name) + r'\s*\)\s*\{',
        re.IGNORECASE,
    )
    area_re = re.compile(r'^\s*area\s*:\s*([\d.]+)\s*;', re.IGNORECASE)

    in_cell = False
    depth   = 0
    with open(lib_path, 'r', errors='replace') as fh:
        for line in fh:
            if not in_cell:
                if cell_re.search(line):
                    in_cell = True
                    depth   = line.count('{') - line.count('}')
                continue

            # Inside the cell block — look for its 'area'.
            m = area_re.match(line)
            if m:
                return float(m.group(1))

            # Track braces so we stop when the cell block ends.
            depth += line.count('{') - line.count('}')
            if depth <= 0:
                break

    raise ValueError(
        f"Could not find 'area' for cell {cell_name!r} inside {lib_path}.\n"
        "Check that the .lib was produced by LEF/LIB extraction after "
        "single-core PnR (the cell block should contain 'area : <float> ;')."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    script_dir   = Path(__file__).resolve().parent
    default_runs = str(script_dir / '..' / 'syn' / 'synopsys' / 'runs')
    default_libs = str(script_dir / '..' / 'syn' / 'synopsys' / 'libs')

    ap = argparse.ArgumentParser(
        description=(
            "Compute the Innovus floorplan area budget for the `full-vortex` "
            "PnR run (Vortex GPGPU = 4 × VX_socket_top blackboxes + shared L2)."
        )
    )
    ap.add_argument(
        '--runs_dir', default=default_runs,
        help=(
            'Root of synthesis run directories. area.rpt is expected at '
            '<runs_dir>/full-vortex/reports/area.rpt  '
            f'(default: {default_runs})'
        ),
    )
    ap.add_argument(
        '--libs_dir', default=default_libs,
        help=(
            'Directory containing VX_socket_top.lib and all SRAM .lef files  '
            f'(default: {default_libs})'
        ),
    )
    ap.add_argument(
        '--utilization', type=float, default=0.70,
        help='Stdcell utilization of remaining area after macros [0–1], default 0.70',
    )
    ap.add_argument(
        '--aspect', type=float, default=1.0,
        help='Die aspect ratio width/height, default 1.0 (square)',
    )
    ap.add_argument(
        '--margin_pct', type=float, default=0.10,
        help='Guard-band margin added on top of computed area, default 0.10 (10%%)',
    )
    ap.add_argument(
        '--core_margin', type=float, default=5.0,
        help='Core-to-die-boundary margin in µm (passed to floorPlan -s), default 5.0',
    )
    ap.add_argument(
        '--emit_tcl', metavar='FILE',
        help=(
            'If given, write a small Tcl snippet to FILE containing the '
            'computed floorplan dimensions as Tcl variables (FLOORPLAN_W, '
            'FLOORPLAN_H, FLOORPLAN_MARGIN).'
        ),
    )
    args = ap.parse_args()

    if not (0.1 <= args.utilization <= 0.95):
        sys.exit("ERROR: --utilization should be between 0.10 and 0.95")
    if args.aspect <= 0:
        sys.exit("ERROR: --aspect must be > 0")

    libs_dir = os.path.normpath(args.libs_dir)

    # ── 1. Locate the core .lib (produced by single-core LEF/LIB extract) ─
    core_lib = os.path.join(libs_dir, CORE_LIB_NAME + '.lib')
    if not os.path.isfile(core_lib):
        sys.exit(
            f"ERROR: core Liberty file not found: {core_lib}\n"
            "Run single-core PnR and the LEF/LIB extraction step first so "
            "that VX_socket_top.lib exists alongside the bsg_fakeram macros."
        )

    core_area_um2 = parse_lib_cell_area(core_lib, CORE_CELL_NAME)

    # ── 2. Locate the full-vortex DC area report for the stdcell total ───
    area_rpt = os.path.normpath(
        os.path.join(args.runs_dir, CONFIG, 'reports', 'area.rpt')
    )
    if not os.path.isfile(area_rpt):
        sys.exit(
            f"ERROR: area report not found: {area_rpt}\n"
            "Run 'make full-vortex' in hw/syn/synopsys first."
        )

    stdcell_area_um2  = parse_dc_area_report(area_rpt)
    stdcell_with_util = stdcell_area_um2 / args.utilization

    # ── 3. L2 macro counts & sizes (RTL-rule based) ──────────────────────
    l2_tag_lef  = os.path.join(libs_dir, L2_TAG_LIB  + '.lef')
    l2_data_lef = os.path.join(libs_dir, L2_DATA_LIB + '.lef')
    for p in (l2_tag_lef, l2_data_lef):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: LEF file not found: {p}")

    l2_tag_w,  l2_tag_h  = parse_lef_macro_size(l2_tag_lef)
    l2_data_w, l2_data_h = parse_lef_macro_size(l2_data_lef)

    l2_tag_qty   = L2_NUM_BANKS * L2_NUM_WAYS
    l2_data_qty  = L2_NUM_BANKS * L2_NUM_WAYS
    l2_tag_area  = l2_tag_qty  * l2_tag_w  * l2_tag_h
    l2_data_area = l2_data_qty * l2_data_w * l2_data_h
    l2_total_area = l2_tag_area + l2_data_area

    core_total_area = NUM_CORES * core_area_um2

    # ── 4. Floorplan sizing ──────────────────────────────────────────────
    raw_area      = core_total_area + l2_total_area + stdcell_with_util
    margined_area = raw_area * (1.0 + args.margin_pct)
    fp_height     = (margined_area / args.aspect) ** 0.5
    fp_width      = fp_height * args.aspect

    # ── 5. Print report ──────────────────────────────────────────────────
    print()
    hr()
    print("  AREA BUDGET REPORT — full-vortex (Vortex GPGPU)")
    hr()
    print(f"  Config            : {CONFIG}  "
          f"(cores={NUM_CORES}  shared-L2 banks={L2_NUM_BANKS}  ways={L2_NUM_WAYS})")
    print(f"  area.rpt          : {area_rpt}")
    print(f"  core_lib          : {core_lib}")
    print(f"  libs_dir          : {libs_dir}")
    print(f"  Utilization target: {args.utilization*100:.0f}%")
    print(f"  Margin            : {args.margin_pct*100:.0f}%")
    print(f"  Aspect ratio (W/H): {args.aspect:.2f}")
    hr()

    print("\n  CORE BLACKBOXES  (area from VX_socket_top.lib)")
    hr()
    print(f"  {'ROLE':<14} {'QTY':>4}  {'Single (µm²)':>14}  {'Total (µm²)':>14}  LIB")
    hr()
    lib_name = Path(core_lib).name
    print(f"  {'CORE':<14} {NUM_CORES:>4}  {core_area_um2:>14,.2f}  "
          f"{core_total_area:>14,.2f}  {lib_name}")

    print("\n  L2 MEMORY MACROS  (counts from L2_NUM_BANKS × L2_NUM_WAYS)")
    hr()
    print(f"  {'ROLE':<14} {'QTY':>4}  {'W (µm)':>8}  {'H (µm)':>8}  "
          f"{'Single (µm²)':>14}  {'Total (µm²)':>14}  LEF")
    hr()
    print(f"  {'L2_TAGS':<14} {l2_tag_qty:>4}  {l2_tag_w:>8.3f}  {l2_tag_h:>8.3f}  "
          f"{l2_tag_w*l2_tag_h:>14,.2f}  {l2_tag_area:>14,.2f}  {L2_TAG_LIB}.lef")
    print(f"  {'L2_DATA':<14} {l2_data_qty:>4}  {l2_data_w:>8.3f}  {l2_data_h:>8.3f}  "
          f"{l2_data_w*l2_data_h:>14,.2f}  {l2_data_area:>14,.2f}  {L2_DATA_LIB}.lef")
    hr()
    row("L2 total macro area", f"{l2_total_area:,.2f}", "µm²")

    print("\n  STDCELLS (from full-vortex Design Compiler area.rpt)")
    hr()
    row("Total stdcell area",   f"{stdcell_area_um2:,.2f}",  "µm²")
    row("Area @ target util",   f"{stdcell_with_util:,.2f}", "µm²")

    print("\n  FLOORPLAN SIZING")
    hr()
    row("4 × core area",        f"{core_total_area:,.2f}", "µm²")
    row("L2 macro area",        f"{l2_total_area:,.2f}",   "µm²")
    row("Stdcells @ util",      f"{stdcell_with_util:,.2f}", "µm²")
    row("Raw total area",       f"{raw_area:,.2f}",        "µm²")
    row("After margin",         f"{margined_area:,.2f}",   "µm²")
    row("Floorplan width",      f"{fp_width:,.3f}",        "µm")
    row("Floorplan height",     f"{fp_height:,.3f}",       "µm")
    hr()

    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  PASTE INTO YOUR INNOVUS TCL SCRIPT                          │
  │                                                              │
  │  floorPlan -s {fp_width:>10.3f} {fp_height:>10.3f} 5.0 5.0 5.0 5.0        │
  │             ↑ width(µm)  ↑ height(µm)  ↑ core margins       │
  └──────────────────────────────────────────────────────────────┘
""")

    total_macro_area = core_total_area + l2_total_area
    macro_frac = total_macro_area / margined_area if margined_area > 0 else 0
    if macro_frac > 0.80:
        print(f"  ⚠  WARNING: macros (cores + L2) occupy {macro_frac*100:.1f}% of the die area.")
        print("     Consider increasing --margin_pct or reviewing L2/core sizes.\n")
    else:
        print(f"  ℹ  Macros (cores + L2) occupy {macro_frac*100:.1f}% of the die area. Looks reasonable.\n")

    # ── 6. Emit Tcl variable file (optional) ─────────────────────────────
    if args.emit_tcl:
        _emit_tcl(
            path=args.emit_tcl,
            config=CONFIG,
            fp_width=fp_width,
            fp_height=fp_height,
            core_margin=args.core_margin,
            utilization=args.utilization,
            margin_pct=args.margin_pct,
            aspect=args.aspect,
            stdcell_area=stdcell_area_um2,
            macro_area=total_macro_area,
        )
        print(f"  Tcl budget file  : {args.emit_tcl}\n")


if __name__ == '__main__':
    main()
