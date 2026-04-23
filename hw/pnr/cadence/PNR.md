# Vortex GPGPU — Cadence Innovus PnR

## Prerequisites

1. **Synthesized netlist** for the target config must exist:
   ```
   hw/syn/synopsys/runs/<config>/results/Vortex_netlist.v
   ```
   If it is missing, run synthesis first:
   ```bash
   cd hw/syn/synopsys
   make <config>
   ```

2. **Load the Cadence module** (EWS — do this in every new shell):
   ```bash
   module load cadence
   ```
   Verify: `which innovus` should resolve.

## Running PnR

```bash
cd hw/pnr/cadence
make <config>
```

Examples:
```bash
make 1c1n4w4t
make 1c2n4w4t
make 2c2n4w4t_l2
```

Available configs: `1c1n4w4t`, `1c2n4w4t`, `2c2n4w4t_l2`, `4c2n4w4t_l2l3`, `4c4n8w4t_l2l3`.

Logs stream to `runs/<config>/logs/pnr_<config>.log`.

## Outputs

All outputs land under `runs/<config>/`:

| Path | Contents |
|------|----------|
| `reports/timing_setup.rpt` | Post-route setup timing (real wire delays) |
| `reports/timing_hold.rpt` | Post-route hold timing |
| `reports/area.rpt` | Cell area by module hierarchy |
| `reports/power_total.rpt` | Total dynamic + leakage power |
| `reports/power_hierarchy.rpt` | Per-module power including interconnect switching |
| `reports/net_detail.rpt` | Per-net routed wire length and via count |
| `reports/drc.rpt` | Design rule violations |
| `reports/congestion.rpt` | Routing congestion summary |
| `reports/dieshot.png` | Layout screenshot (requires `DISPLAY`; see below) |
| `results/Vortex.gds` | Full GDSII layout |
| `results/Vortex.def` | Placed-and-routed DEF |
| `results/Vortex_pnr.v` | Final gate-level netlist |
| `results/Vortex.enc` | Saved Innovus database |

## Dieshot

**With an X display** (`ssh -X` or an EWS FastX session): `dieshot.png` is written automatically to `reports/`.

**Without a display** (headless batch): open the saved database in the Innovus GUI after the run:
```bash
innovus -init runs/<config>/results/Vortex
```
Then in the Innovus shell:
```tcl
fit
displaySnapshot -type PNG -resolution 2048 -file dieshot.png
```

Alternatively, open the GDS in KLayout for an interactive view:
```bash
klayout runs/<config>/results/Vortex.gds
```

## Interpreting Interconnect PPA

Design Compiler gives wire-load-model estimates only.
After Innovus PnR the following reports contain real interconnect numbers:

- **`power_hierarchy.rpt`** — dynamic power split into *internal* (cell) and *switching* (interconnect wire capacitance); reported per module.
- **`net_detail.rpt`** — routed wire length and via count per net; aggregate by module to get per-block interconnect area.
- **`timing_setup.rpt`** — critical-path delays include actual extracted RC parasitics.

## Bottom-up Hierarchical Flow — Hard-Macro Extraction

The full-vortex GPGPU is PnR-ed on top of four `VX_socket_top` hard macros (one per core socket) that come out of the single-core run. The hand-off is two files:

```
hw/pnr/cadence/export/single-core/VX_socket_top.lef   # physical abstract
hw/pnr/cadence/export/single-core/VX_socket_top.lib   # extracted timing model
```

Both files are produced by a single `make` target after single-core PnR completes.

### Layer reservation policy

The single-core PDN occupies **M7** (power stripes) and **M8** (ring H-sides). To give the full-vortex top-level router clean over-the-macro channels we reserve **M9 and M10**:

1. `09_routing.tcl` caps NanoRoute at M8 when `CONFIG == single-core` (via `setNanoRouteMode -routeTopRoutingLayer 8`), so the single-core DB never contains geometry above M8.
2. `export_prep.tcl` asserts that M9/M10 are empty before extraction; the extraction fails loudly if not.
3. `export_lef.tcl` passes `-specifyTopLayer 8` to `write_lef_abstract`, so the emitted LEF advertises M9/M10 as unobstructed.

### Running the extraction

```bash
cd hw/pnr/cadence

# 1. Re-run single-core routing with the M8 cap (first time only, once you
#    pull the updated 09_routing.tcl):
make single-core STAGE=9
make single-core STAGE=10
make single-core STAGE=11
make single-core STAGE=12

# 2. Produce VX_socket_top.{lef,lib}:
make extract-macro
```

`make export-blackbox` is an alias for the same target.

Variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MACRO_CELL` | `VX_socket_top` | cell name recorded inside the LEF MACRO and LIB cell headers |
| `VIEW_NAME`  | `typView`       | MMMC analysis view(s) to extract timing from |

Examples:

```bash
make extract-macro VIEW_NAME=setup_view           # pick a specific MMMC view
make extract-macro MACRO_CELL=VX_socket_top_v2    # version the emitted cell
```

### Full-vortex consumption

`scripts/01_init_design.tcl` and `mmmc.tcl` look for the macro artefacts in this order:

1. `hw/pnr/cadence/export/single-core/VX_socket_top.{lef,lib}` (preferred — canonical output of `make extract-macro`).
2. `hw/syn/synopsys/libs/VX_socket_top.{lef,lib}` (fallback, for the historical hand-installed layout).

Once `make extract-macro` has run, `make full-vortex STAGE=1..12` proceeds normally.

### Flow summary

```
make single-core STAGE=1..12     (routing capped at M8 for single-core only)
        │
        ▼
runs/single-core/checkpoints/after_12_outputs
        │
        ▼
make extract-macro               (extract_macro.tcl)
        ├─ export_prep.tcl   snap pins, assert M9/M10 empty
        ├─ export_lef.tcl    write_lef_abstract
        └─ export_etm.tcl    do_extract_model
        │
        ▼
export/single-core/VX_socket_top.{lef,lib}
        │
        ▼
make full-vortex STAGE=1..12     (consumes the macro as a blackbox)
```

## Cleaning Up

```bash
make clean
```
Removes `runs/` and `export/`. The shared `libs/` directory is never touched.
