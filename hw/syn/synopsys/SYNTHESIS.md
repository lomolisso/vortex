# Vortex GPGPU — ASIC Synthesis (Synopsys DC)

This flow uses **Synopsys Design Compiler** with **NanGate45** standard cells
and **bsg_fakeram** SRAM macros to synthesize two complementary designs:

| Config         | DC top           | What it contains                                                                                        |
| -------------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| `single-core`  | `VX_socket_top`  | One `VX_core` + its **private** I$, D$, local memory and register file. `SOCKET_SIZE=1`, no L2.         |
| `1c8n4w4t`     | `Vortex`         | Full GPGPU: 8 cores (8 `VX_socket_top` instances) inside one cluster, sharing a 1 MiB **L2** cache.    |

`VX_socket_top` is a flat-port wrapper around `VX_socket` added in
[`hw/rtl/VX_socket_top.sv`](../../rtl/VX_socket_top.sv). It exists so the
PnR’d single-core block has explicit flat pins that can later be packaged
as a `.lef`/`.lib` macro and substituted back into the `1c8n4w4t` run as a
blackbox. `VX_cluster` is gated on `ASIC_SYNTHESIS` to instantiate
`VX_socket_top` directly (see [`hw/rtl/VX_cluster.sv`](../../rtl/VX_cluster.sv)),
so both scenarios share the exact same socket boundary.

## Prereqs

- **Clone + submodules** (required for FPnew + HardFloat):

```bash
git clone <your-fork-or-upstream>
cd vortex
git submodule update --init --recursive
```

- **Tooling**:

```bash
export PATH=/software/Synopsys-2024_x86_64/syn/W-2024.09/bin:$PATH
export LM_LICENSE_FILE=27005@synopsys.webstore.illinois.edu
export LD_PRELOAD=/lib64/libk5crypto.so.3:/lib64/libtinfo.so.5
```

`dc_shell` must be available in your environment.

## Run synthesis

From `hw/syn/synopsys/`:

```bash
cd hw/syn/synopsys

# (Optional) quick elaboration-only check
make elab_single-core
make elab_1c8n4w4t

# Synthesize one configuration
make single-core
make 1c8n4w4t

# Synthesize both
make all
```

> `make 1c8n4w4t` runs in **bottom-up mode by default**: the socket
> is linked as a blackbox from `libs/VX_socket_top.db`, which is
> auto-built from the routed single-core's `.lib`. See the
> [Bottom-up hierarchical synthesis](#bottom-up-hierarchical-synthesis)
> section below for the prerequisite `make extract-macro` step and for
> how to force a flat re-synth.

## Outputs

Results are written per-config under `runs/<config>/`:

| Path (under `runs/<config>/`)        | `single-core`              | `1c8n4w4t`          |
| ------------------------------------ | -------------------------- | ------------------- |
| `results/<TOP>_netlist.v`            | `VX_socket_top_netlist.v`  | `Vortex_netlist.v`  |
| `results/<TOP>.ddc`                  | `VX_socket_top.ddc`        | `Vortex.ddc`        |
| `results/<TOP>.sdc`                  | `VX_socket_top.sdc`        | `Vortex.sdc`        |
| `reports/`                           | area / timing / power / qor / hierarchy / constraints |
| `logs/<config>_syn.log`              | full DC transcript         | full DC transcript  |

The PnR flow consumes these directly via `SYN_DIR` and `CONFIG`; see
`hw/pnr/cadence/`.

## Workflow

The two configs slot into this end-to-end flow:

1. `make single-core` — synthesize `VX_socket_top` (RTL → netlist).
2. Run PnR on the single-core netlist (see `hw/pnr/cadence/`).
3. `make extract-macro` in `hw/pnr/cadence/` — emits
   `VX_socket_top.lef` and `VX_socket_top.lib` into
   `hw/pnr/cadence/export/single-core/`.
4. `make 1c8n4w4t` — synthesize the full GPGPU with the socket
   **blackboxed** from the exported `.lib` (see below).
5. Run PnR on the GPGPU netlist; Innovus reads the same `.lef`/`.lib`
   from `export/single-core/` to place the socket as a hard macro.

## Bottom-up hierarchical synthesis

`make 1c8n4w4t` runs DC in bottom-up mode by default. Instead of
re-synthesizing `VX_socket_top` from RTL every run (which is wasteful
— the routed single-core already has characterized timing/area), DC:

1. Reads `VX_socket_top.db` (a precompiled Liberty model) into
   `link_library`.
2. Skips `hw/rtl/VX_socket_top.sv` during `analyze` so `elaborate
   Vortex` leaves the socket as an unresolved reference that `link`
   resolves against the `.db`.
3. Applies `set_dont_touch` to every `VX_socket_top` instance so
   `compile_ultra` never optimizes across the blackbox boundary.

The resulting `Vortex_netlist.v` has eight `VX_socket_top` instance
lines and zero gate-level expansion of the socket internals.

### Artifact chain

```
hw/pnr/cadence/export/single-core/VX_socket_top.lib   ← make extract-macro
                     │                                   (Innovus do_extract_model)
                     │                                 compiled by
                     ▼ compile_lib_to_db.tcl (dc_shell)
hw/syn/synopsys/libs/VX_socket_top.db                 ← make socket-db
                     │                                   (auto-built as a prereq
                     │                                    of make 1c8n4w4t)
                     ▼
hw/syn/synopsys/runs/1c8n4w4t/results/Vortex_netlist.v
```

The `.db` is treated as a **derived** artifact: `make clean` deletes
it and the next `make 1c8n4w4t` rebuilds it from whichever `.lib`
exists. Set `KEEP_SOCKET_DB=1` on the `make clean` command line to
preserve it across cleans.

### `.lib` lookup order

The Makefile searches for `VX_socket_top.lib` in two places and uses
the first hit:

1. `hw/pnr/cadence/export/single-core/VX_socket_top.lib` — canonical
   output of `make extract-macro`. **Prefer this.**
2. `hw/syn/synopsys/libs/VX_socket_top.lib` — hand-copied fallback
   (useful when sharing a `.lib` across machines without a full PnR
   tree).

If neither exists, the Makefile fails with a message pointing at
`make extract-macro`. Do **not** commit the auto-generated `.db` to
git; it's a build artifact.

### Consistency rule

The `.lib`, `.lef`, and `Vortex_netlist.v` consumed by top-level PnR
must all come from the **same single-core PnR run**. Re-run
`make extract-macro` every time the single-core RTL, constraints, or
floorplan change — otherwise DC will characterize `VX_socket_top`
pins that don't match the PnR netlist, and Innovus will fail with
connectivity errors on the 1c8n4w4t run.

### Forcing a flat re-synth

To bypass the bottom-up flow — for example, to compare QoR against a
flat reference or to debug a netlist discrepancy — override
`BLACKBOX_SOCKET_1c8n4w4t` on the command line:

```bash
# Flat 1c8n4w4t (re-synthesizes VX_socket_top from RTL):
make 1c8n4w4t BLACKBOX_SOCKET_1c8n4w4t=0

# Forcing blackbox on single-core is nonsensical (single-core IS the
# synthesis of VX_socket_top), and is not supported.
```

The log line `INFO: BLACKBOX_SOCKET=0 — VX_socket_top will be
synthesized from RTL (flat).` confirms the flat path is active.

## Cleanup

```bash
cd hw/syn/synopsys
make clean
```
