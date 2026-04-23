#!/usr/bin/env python3
"""
compute_single_core_area_budget.py
──────────────────────────────────
Floorplan area budget for the `single-core` PnR run.

The `single-core` synthesis target elaborates `VX_socket_top`, which wraps
exactly ONE VX_core together with its private L1 I$/D$, local memory, and
register file.  It is the "core block" that we will later package as a
.lef/.lib macro for the full-vortex PnR.

This script is a specialisation of `compute_area_budget.py`:
  · no free-form <C>c<N>n<W>w<T>t parser — the config is always 1/1/4/4
  · area.rpt is always <runs_dir>/single-core/reports/area.rpt
  · macro counts come from the same RTL rules (4 I$ tag + 4 I$ data + ...)

It emits a `floorplan_budget.tcl` with exactly the same variables as the
original script so that `03_floorplan_single-core.tcl` (which just
sources `03_floorplan_1c1n4w4t.tcl`) can consume it unchanged.

Default paths (relative to this script's directory)
────────────────────────────────────────────────────
  --runs_dir  ../syn/synopsys/runs   (area.rpt at <runs_dir>/single-core/reports/area.rpt)
  --libs_dir  ../syn/synopsys/libs   (all .lef files live here)

Usage
─────
  # Minimal:
  python3 compute_single_core_area_budget.py

  # Emit the Tcl snippet consumed by 03_floorplan_1c1n4w4t.tcl:
  python3 compute_single_core_area_budget.py \\
      --emit_tcl runs/single-core/floorplan_budget.tcl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Reuse helpers verbatim from compute_area_budget.py so both scripts stay
# in lockstep whenever the SRAM naming convention or Vortex RTL rules
# change.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from compute_area_budget import (  # noqa: E402
    parse_dc_area_report,
    detect_sram_libs_from_area_rpt,
    derive_macro_rows,
    _emit_tcl,
    hr,
    row,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixed single-core config — matches hw/syn/synopsys/Makefile's
# DEFINES_single-core (NUM_CLUSTERS=1, NUM_CORES=1, NUM_WARPS=4, NUM_THREADS=4,
# SOCKET_SIZE=1, no L2).
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = "single-core"
C, N, W, T = 1, 1, 4, 4


def main():
    script_dir   = Path(__file__).resolve().parent
    default_runs = str(script_dir / '..' / 'syn' / 'synopsys' / 'runs')
    default_libs = str(script_dir / '..' / 'syn' / 'synopsys' / 'libs')

    ap = argparse.ArgumentParser(
        description=(
            "Compute the Innovus floorplan area budget for the `single-core` "
            "PnR run (VX_socket_top = 1 VX_core + its L1 I$/D$ + lmem + "
            "register file)."
        )
    )
    ap.add_argument(
        '--runs_dir', default=default_runs,
        help=(
            'Root of synthesis run directories. area.rpt is expected at '
            '<runs_dir>/single-core/reports/area.rpt  '
            f'(default: {default_runs})'
        ),
    )
    ap.add_argument(
        '--libs_dir', default=default_libs,
        help=(
            'Directory containing .lef files for all SRAM macros  '
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
            'FLOORPLAN_H, FLOORPLAN_MARGIN). Consumed by 03_floorplan_*.tcl.'
        ),
    )
    args = ap.parse_args()

    if not (0.1 <= args.utilization <= 0.95):
        sys.exit("ERROR: --utilization should be between 0.10 and 0.95")
    if args.aspect <= 0:
        sys.exit("ERROR: --aspect must be > 0")

    # ── 1. Locate area report ─────────────────────────────────────────────
    area_rpt = os.path.normpath(
        os.path.join(args.runs_dir, CONFIG, 'reports', 'area.rpt')
    )
    if not os.path.isfile(area_rpt):
        sys.exit(
            f"ERROR: area report not found: {area_rpt}\n"
            "Run 'make single-core' in hw/syn/synopsys first."
        )

    # ── 2. Stdcell area from DC ───────────────────────────────────────────
    stdcell_area_um2 = parse_dc_area_report(area_rpt)

    # ── 3. Detect SRAM types actually used by DC ─────────────────────────
    sram_names = detect_sram_libs_from_area_rpt(area_rpt)
    if not sram_names:
        sys.exit(
            f"ERROR: no SRAM libraries found in {area_rpt}.\n"
            "Make sure it is the output of 'report_area -hierarchy'."
        )

    # ── 4. Build macro table (counts + LEF dimensions) ───────────────────
    macro_rows = derive_macro_rows(sram_names, C, N, W, T, args.libs_dir)

    total_macro_area_um2 = sum(r['total_um2'] for r in macro_rows)
    total_macro_qty      = sum(r['qty']       for r in macro_rows)

    # ── 5. Floorplan sizing ──────────────────────────────────────────────
    stdcell_with_util = stdcell_area_um2 / args.utilization
    raw_area          = total_macro_area_um2 + stdcell_with_util
    margined_area     = raw_area * (1.0 + args.margin_pct)

    fp_height = (margined_area / args.aspect) ** 0.5
    fp_width  = fp_height * args.aspect

    # ── 6. Print report ──────────────────────────────────────────────────
    print()
    hr()
    print("  AREA BUDGET REPORT — single-core (VX_socket_top)")
    hr()
    print(f"  Config            : {CONFIG}  "
          f"(clusters={C}  cores={N}  warps={W}  threads={T})")
    print(f"  area.rpt          : {area_rpt}")
    print(f"  libs_dir          : {os.path.normpath(args.libs_dir)}")
    print(f"  Utilization target: {args.utilization*100:.0f}%")
    print(f"  Margin            : {args.margin_pct*100:.0f}%")
    print(f"  Aspect ratio (W/H): {args.aspect:.2f}")
    hr()

    print("\n  STDCELLS (from Design Compiler)")
    hr()
    row("Total stdcell area",   f"{stdcell_area_um2:,.2f}", "µm²")
    row("Area @ target util",   f"{stdcell_with_util:,.2f}", "µm²")

    print("\n  MEMORY MACROS  (counts derived from Vortex RTL rules)")
    hr()
    print(f"  {'ROLE':<14} {'QTY':>4}  {'W (µm)':>8}  {'H (µm)':>8}  "
          f"{'Single (µm²)':>14}  {'Total (µm²)':>14}  LEF")
    hr()
    for m in macro_rows:
        lef_name = Path(m['lef']).name
        print(f"  {m['role']:<14} {m['qty']:>4}  {m['w']:>8.3f}  {m['h']:>8.3f}  "
              f"{m['single_um2']:>14,.2f}  {m['total_um2']:>14,.2f}  {lef_name}")
    hr()
    row("Total macro count",    f"{total_macro_qty}",               "macros")
    row("Total macro area",     f"{total_macro_area_um2:,.2f}",     "µm²")

    print("\n  FLOORPLAN SIZING")
    hr()
    row("Raw total area",       f"{raw_area:,.2f}",      "µm²")
    row("After margin",         f"{margined_area:,.2f}", "µm²")
    row("Floorplan width",      f"{fp_width:,.3f}",      "µm")
    row("Floorplan height",     f"{fp_height:,.3f}",     "µm")
    hr()

    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  PASTE INTO YOUR INNOVUS TCL SCRIPT                          │
  │                                                              │
  │  floorPlan -s {fp_width:>10.3f} {fp_height:>10.3f} 5.0 5.0 5.0 5.0        │
  │             ↑ width(µm)  ↑ height(µm)  ↑ core margins       │
  └──────────────────────────────────────────────────────────────┘
""")

    macro_frac = total_macro_area_um2 / margined_area if margined_area > 0 else 0
    if macro_frac > 0.60:
        print(f"  ⚠  WARNING: macros occupy {macro_frac*100:.1f}% of the die area.")
        print("     Consider increasing --margin_pct or reviewing macro counts.\n")
    else:
        print(f"  ℹ  Macros occupy {macro_frac*100:.1f}% of the die area. Looks reasonable.\n")

    # ── 7. Emit Tcl variable file (optional) ─────────────────────────────
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
            macro_area=total_macro_area_um2,
        )
        print(f"  Tcl budget file  : {args.emit_tcl}\n")


if __name__ == '__main__':
    main()
