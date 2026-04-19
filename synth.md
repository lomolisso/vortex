# Vortex GPGPU — ASIC Synthesis Documentation

> **Flow**: Synopsys Design Compiler (DC) · NanGate45 standard cells (`stdcells.db`) · `bsg_fakeram` SRAM macros · FPnew (FPU) · Berkeley HardFloat (TCU)

---

## Overview

The Vortex RTL was originally written to support three targets — simulation (DPI-C), Quartus (Intel FPGA), and Vivado (Xilinx FPGA) — via a cascading `` `ifdef `` chain in `VX_platform.vh`. This document describes the changes made to enable a fourth, fully synthesizable ASIC target under a new `ASIC_SYNTHESIS` preprocessor define, along with the corresponding changes to the helper scripts and the Synopsys DC synthesis directory.

The define activates the following substitutions:

| Concern | FPGA / Simulation | ASIC (`ASIC_SYNTHESIS`) |
|---|---|---|
| `string` type | `string` (simulation) or empty (Vivado) | empty — DC rejects `string` |
| Synthesis attributes | `(* ram_style = ... *)` etc. | empty |
| FPU | `xil_*` / `acl_*` / DPI | FPnew (`fpnew_top`, `third_party/cvfpu`) |
| TCU dot-product | `xil_fmul` / DPI | Berkeley HardFloat (`tcu/bhf/`) |
| Large SRAM | FPGA BRAM inference | `bsg_fakeram` 1RW / 1R1W macros |
| Integer MUL/DIV DPI | enabled in simulation | disabled (behavioral `*` / `/`) |

---

## 1. RTL Changes

### 1.1 `hw/rtl/VX_platform.vh` — New `ASIC_SYNTHESIS` Branch

The platform header controls all synthesis-tool-specific macro expansions. The existing chain was:

```
`ifdef QUARTUS   → Intel FPGA attributes, STRING = string
`elsif VIVADO    → Xilinx FPGA attributes, STRING = empty
`else            → generic fallback, STRING = string  ← DC ERROR
`endif
```

A new `elsif ASIC_SYNTHESIS` branch was inserted **between** `VIVADO` and the generic `else`:

```verilog
`elsif ASIC_SYNTHESIS
`define MAX_FANOUT      8
`define LATENCY_IMUL    3
`define FORCE_BRAM(d,w) (((d) >= 64 || (w) >= 16 || ((d) * (w)) >= 512) && ((d) * (w)) >= 64)
`define USE_SRAM_MACRO(d,w) `FORCE_BRAM(d,w)
`define USE_BLOCK_BRAM              // empty — macro handled in VX_sp_ram / VX_dp_ram
`define USE_FAST_BRAM               // empty
`define NO_RW_RAM_CHECK             // empty
`define RW_RAM_CHECK                // empty
`define DISABLE_BRAM                // empty
`define PRESERVE_NET                // empty
`define BLACKBOX_CELL               // empty
`define STRING                      // empty — DC does not support the 'string' type
```

Key points:
- `STRING` is empty so `parameter \`STRING ARBITER = "R"` becomes an untyped parameter — the same pattern Vivado already uses, which DC accepts.
- `USE_BLOCK_BRAM` is empty because SRAM macro selection is handled explicitly inside `VX_sp_ram` / `VX_dp_ram` (see §1.3 and §1.4) rather than via synthesis attributes.
- `USE_SRAM_MACRO` is an alias for `FORCE_BRAM` that makes ASIC-specific code self-documenting.

---

### 1.2 `hw/rtl/VX_config.vh` — Three Mode-Selection Changes

#### 1.2a FPU Mode — Force `FPU_FPNEW`

The auto-selection logic now checks `ASIC_SYNTHESIS` first:

```verilog
`ifndef FPU_FPNEW
`ifndef FPU_DSP
`ifndef FPU_DPI
`ifdef ASIC_SYNTHESIS
`define FPU_FPNEW        // FPnew (fpnew_top) — synthesizable, no FPGA IPs or DPI
`elsif SYNTHESIS
`define FPU_DSP
`else
`ifndef DPI_DISABLE
`define FPU_DPI
`else
`define FPU_DSP
`endif
`endif
`endif
`endif
`endif
```

Under `ASIC_SYNTHESIS`, `FPU_FPNEW` is selected unconditionally, routing the FPU through `VX_fpu_fpnew.sv` → `fpnew_top` (FPnew, `third_party/cvfpu`). The `FPU_DSP` path (`VX_fpu_dsp.sv`) that references `xil_fmul` / `acl_fmadd` is never compiled.

#### 1.2b Integer MUL/DIV — Disable DPI

An additional `ifndef ASIC_SYNTHESIS` guard was added around the `IMUL_DPI` / `IDIV_DPI` defines:

```verilog
`ifndef SYNTHESIS
`ifndef ASIC_SYNTHESIS
`ifndef DPI_DISABLE
`define IMUL_DPI
`define IDIV_DPI
`endif
`endif
`endif
```

Under `ASIC_SYNTHESIS`, neither DPI define is set, so `VX_alu_muldiv.sv` uses the synthesizable `VX_serial_mul` / `VX_serial_div` behavioral path.

#### 1.2c TCU Mode — Force `TCU_BHF`

Analogous to the FPU change:

```verilog
`ifndef TCU_DRL
`ifndef TCU_BHF
`ifndef TCU_DSP
`ifndef TCU_DPI
`ifdef ASIC_SYNTHESIS
`define TCU_BHF          // Berkeley HardFloat — synthesizable, no xil_* or DPI
`elsif SYNTHESIS
`define TCU_DSP
`else
...
`endif
`endif
`endif
`endif
`endif
`endif
```

Under `ASIC_SYNTHESIS`, `TCU_BHF` is selected, routing the tensor core dot-product through `VX_tcu_fedp_bhf.sv` and the `tcu/bhf/` Berkeley HardFloat wrappers. The `TCU_DSP` path (`VX_tcu_fedp_dsp.sv`) that falls back to `xil_fmul` / `xil_fadd` under non-VIVADO synthesis is never compiled.

---

### 1.3 `hw/rtl/libs/VX_sp_ram.sv` — bsg_fakeram 1RW Macro Instantiation

Inside the existing `if (OUT_REG) begin : g_sync → if (FORCE_BRAM) begin : g_bram` block, an `` `ifdef ASIC_SYNTHESIS `` branch was added. Under this branch, the appropriate bsg_fakeram 1RW macro is instantiated based on `SIZE` and `DATAW`:

| `SIZE` | `DATAW` | Macro |
|---|---|---|
| 64 | 512 | `sram_64x512_1rw` — L1 I/D-cache data arrays |
| 256 | 512 | `sram_256x512_1rw` — L2/LLC cache data arrays |
| 1024 | 32 | `sram_1024x32_1rw` — local memory (LMEM) banks |
| any other | any | behavioral stdcell fallback (DC infers flip-flops) |

The 1RW macro port connections:

```verilog
sram_64x512_1rw macro_inst (
    .clk      (clk),
    .ce_in    (write | read),   // chip enable: assert whenever port is active
    .we_in    (write),          // write enable
    .addr_in  (addr[5:0]),
    .wd_in    (wdata),
    .w_mask_in(w_mask),         // per-bit write mask, assembled from wren[i] stripes
    .rd_out   (rdata_macro)     // registered output (1-cycle latency)
);
```

The existing FPGA BRAM inference code is preserved in the `` `else `` branch. The `g_auto` (small memories), `g_async` (OUT_REG=0), and simulation paths are entirely unchanged.

---

### 1.4 `hw/rtl/libs/VX_dp_ram.sv` — bsg_fakeram 1R1W Macro Instantiation

The same pattern applies to the dual-port RAM inside `g_sync → g_bram`. The bsg_fakeram 1R1W macros use a single shared clock and independent read/write address buses:

| `SIZE` | `DATAW` | Macro |
|---|---|---|
| 64 | ≤ 24 | `sram_64x24_1r1w` — L1 cache tag arrays (zero-padded to 24b) |
| 256 | ≤ 24 | `sram_256x24_1r1w` — L2/LLC cache tag arrays (zero-padded to 24b) |
| 64 | 128 | `sram_64x128_1r1w` — GPR file (4-warp configs) |
| 128 | 128 | `sram_128x128_1r1w` — GPR file (8-warp configs) |
| any other | any | behavioral stdcell fallback |

The 1R1W macro port connections (example):

```verilog
sram_64x128_1r1w macro_inst (
    .clk      (clk),
    .r_ce_in  (read),
    .r_addr_in(raddr[5:0]),
    .rd_out   (rdata),          // registered output (1-cycle latency)
    .w_ce_in  (write),
    .w_addr_in(waddr[5:0]),
    .wd_in    (wdata),
    .w_mask_in(w_mask)          // per-bit write mask, assembled from wren[i] stripes
);
```

For macros where `DATAW < macro_width` (the 24-bit tag macros), `wdata` and `w_mask` are zero-extended to the macro width and `rdata` is truncated on output.

---

### 1.5 Files with No Changes Required

| File | Reason |
|---|---|
| `VX_trace_pkg.sv` | Entire body inside `` `ifdef SIMULATION `` |
| `VX_alu_muldiv.sv` | `IMUL_DPI`/`IDIV_DPI` off under `ASIC_SYNTHESIS`; uses `VX_serial_mul`/`VX_serial_div` |
| `VX_multiplier.sv` | Uses `*` — DC synthesizes via DesignWare |
| `VX_divider.sv` | Non-QUARTUS branch uses `/` and `%` — DC synthesizes via DesignWare |
| `VX_fpu_fpnew.sv` | Already written for ASIC; wraps `fpnew_top` from `third_party/cvfpu` |
| `VX_fpu_dpi.sv` | Guarded by `FPU_DPI` — never compiled under `ASIC_SYNTHESIS` |
| `VX_fpu_dsp.sv` | Guarded by `FPU_DSP` — never compiled under `ASIC_SYNTHESIS` |
| `VX_tcu_fedp_bhf.sv` + `tcu/bhf/*.sv` | Already written for ASIC; uses Berkeley HardFloat primitives |
| `VX_tcu_fedp_dpi.sv` | Guarded by `TCU_DPI` — never compiled |
| `VX_tcu_fedp_dsp.sv` | Guarded by `TCU_DSP` — never compiled under `ASIC_SYNTHESIS` |
| All `interfaces/*.sv` | Pure SV interface declarations — no tool-specific constructs |

---

## 2. Helper Script Changes

### 2.1 `hw/scripts/gen_sources.sh` — Exclude `.bb.v` Blackbox Stubs

`bsg_fakeram` generates two Verilog files per macro: a behavioral model (`<name>.v`) and a blackbox stub (`<name>.bb.v`). When an entire macro directory is passed via the `-J` flag, the original `find` pattern (`*.v`) would match both, causing DC to error on duplicate module definitions.

A `! -name "*.bb.v"` exclusion was added to the external-path file enumeration:

```bash
# Before
find "$(realpath $dir)" -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" \) ! -name "*_pkg.sv" -print

# After
find "$(realpath $dir)" -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" \) ! -name "*_pkg.sv" ! -name "*.bb.v" -print
```

This applies only to the external (`-J`) path loop. Internal (`-I`) paths in the Vortex RTL tree contain no `.bb.v` files and are unchanged.

### 2.2 `hw/scripts/parse_vcs_list.tcl` — No Changes

This script correctly parses `+define+`, `+incdir+`, and file path entries from the VCS-style filelist and returns them as three separate Tcl lists. It is consumed by `dc_syn.tcl` and `dc_elab.tcl` as-is.

---

## 3. Synthesis Directory — `hw/syn/synopsys/`

The previous contents of this directory (TSMC 28nm SRAM models in `models/`, NanGate 15nm `NanGate_15nm_OCL.db`, hardcoded TCL scripts with Georgia Tech tool paths, and a broken `Makefile`) were removed entirely and replaced with the following.

### 3.1 Directory Layout

```
hw/syn/synopsys/
├── Makefile              ← orchestrates all 5 configurations
├── gen_dc_sources.sh     ← filelist generator (wraps gen_sources.sh)
├── dc_syn.tcl            ← main DC synthesis script
├── dc_elab.tcl           ← elaboration-only check script
└── libs/                 ← pre-copied library files (shipped with the repo)
    ├── stdcells.db           ← NanGate45 standard cell timing library
    ├── sram_64x512_1rw.db    ← bsg_fakeram timing library
    ├── sram_256x512_1rw.db
    ├── sram_1024x32_1rw.db
    ├── sram_64x24_1r1w.db
    ├── sram_256x24_1r1w.db
    ├── sram_64x128_1r1w.db
    ├── sram_128x128_1r1w.db
    ├── sram_64x512_1rw.v     ← bsg_fakeram behavioral model (for DC analyze)
    ├── sram_256x512_1rw.v
    ├── sram_1024x32_1rw.v
    ├── sram_64x24_1r1w.v
    ├── sram_256x24_1r1w.v
    ├── sram_64x128_1r1w.v
    └── sram_128x128_1r1w.v
```

`libs/` is populated with copies of all files DC needs for elaboration (`.v` behavioral models) and timing closure (`.db` libraries). It ships committed to the repository so the flow is immediately runnable without referencing external paths.

At runtime, DC creates:
```
hw/syn/synopsys/
├── dc_flist_<config>.f   ← VCS filelist generated by gen_dc_sources.sh
├── logs/<config>_syn.log
├── reports/<config>/     ← area.rpt, timing.rpt, power.rpt, qor.rpt, hierarchy.rpt, constraints.rpt
└── results/<config>/     ← Vortex_netlist.v, Vortex.ddc, Vortex.sdc
```

---

### 3.2 `gen_dc_sources.sh`

Wraps `hw/scripts/gen_sources.sh` to produce a per-configuration VCS-style filelist (`dc_flist_<config>.f`). It hardcodes:

- **Mandatory ASIC defines**: `ASIC_SYNTHESIS SYNTHESIS NDEBUG FPU_FPNEW EXT_F_ENABLE EXT_M_ENABLE ICACHE_ENABLE DCACHE_ENABLE LMEM_ENABLE`
- **External (`-J`) paths**: `third_party/cvfpu/src`, `third_party/hardfloat/source`, `third_party/hardfloat/source/RISCV`, `libs/` (all 7 SRAM `.v` behavioral models)
- **Internal (`-I`) paths**: all nine RTL subdirectories

Configuration-specific defines (`NUM_CLUSTERS`, `NUM_CORES`, etc.) are passed in from the Makefile via `-D` flags.

`gen_sources.sh` ensures correct analysis ordering in the output filelist:
1. `+define+` lines
2. `+incdir+` lines for all external paths
3. Package files (`*_pkg.sv`) from external paths first — `fpnew_pkg.sv` appears before all other FPnew files
4. All remaining `.v` / `.sv` files from external paths (HardFloat, SRAM behavioral models)
5. `+incdir+` lines for Vortex RTL directories
6. Vortex package files (`VX_gpu_pkg.sv`, `VX_trace_pkg.sv`, etc.)
7. Remaining Vortex RTL files, with the top module (`Vortex.sv`) last

---

### 3.3 `dc_syn.tcl`

Main synthesis script. Invoked with the `CONFIG` variable set by the Makefile:

```
dc_shell -x "set CONFIG 1c1n4w4t" -f dc_syn.tcl
```

Flow:
1. Sets `target_library` and `link_library` to all `.db` files in `libs/`
2. Sources `hw/scripts/parse_vcs_list.tcl` and parses `dc_flist_${CONFIG}.f`
3. Extends `search_path` with `+incdir+` entries from the filelist
4. Calls `analyze -format sverilog -define $defines` for each file in filelist order
5. `elaborate Vortex` → `link` → `check_design`
6. Applies timing constraints: 10 ns clock (100 MHz), false path on reset, 20% I/O delays
7. `compile_ultra -no_autoungroup`
8. Writes reports to `reports/$CONFIG/`: `area.rpt`, `timing.rpt`, `power.rpt`, `qor.rpt`, `hierarchy.rpt`, `constraints.rpt`
9. Writes outputs to `results/$CONFIG/`: `Vortex_netlist.v`, `Vortex.ddc`, `Vortex.sdc`

---

### 3.4 `dc_elab.tcl`

Elaboration-only check. Same library setup and filelist loading as `dc_syn.tcl`, but exits after `check_design` + `report_hierarchy` without running `compile_ultra`. Used to quickly catch unresolved modules, leftover DPI imports, `string`-type errors, or missing SRAM macro definitions.

```
dc_shell -x "set CONFIG 1c1n4w4t" -f dc_elab.tcl 2>&1 | tee logs/1c1n4w4t_elab.log
grep -E "Error|Warning|unresolved" logs/1c1n4w4t_elab.log
```

---

## 4. Synthesis Configurations

Five configurations are defined, covering a progression of cluster count, cache hierarchy depth, and register-file geometry:

| Config name | Clusters | Cores | Warps | Threads | L2 | L3 | GPR macro |
|---|---|---|---|---|---|---|---|
| `1c1n4w4t` | 1 | 1 | 4 | 4 | — | — | `sram_64x128_1r1w` |
| `1c2n4w4t` | 1 | 2 | 4 | 4 | — | — | `sram_64x128_1r1w` |
| `2c2n4w4t_l2` | 2 | 2 | 4 | 4 | 128 KB | — | `sram_64x128_1r1w` |
| `4c2n4w4t_l2l3` | 4 | 2 | 4 | 4 | 128 KB | 512 KB | `sram_64x128_1r1w` |
| `4c4n8w4t_l2l3` | 4 | 4 | 8 | 4 | 256 KB | 512 KB | `sram_128x128_1r1w` |

The `4c4n8w4t_l2l3` configuration doubles the warp count to 8, which causes the GPR bank size to exceed the `sram_64x128_1r1w` threshold and promotes to the `sram_128x128_1r1w` macro.

The full preprocessor define sets for each configuration are captured in the `Makefile` as `DEFINES_<config>` variables:

```makefile
DEFINES_1c1n4w4t      := -DNUM_CLUSTERS=1 -DNUM_CORES=1 -DNUM_WARPS=4 -DNUM_THREADS=4
DEFINES_1c2n4w4t      := -DNUM_CLUSTERS=1 -DNUM_CORES=2 -DNUM_WARPS=4 -DNUM_THREADS=4
DEFINES_2c2n4w4t_l2   := -DNUM_CLUSTERS=2 -DNUM_CORES=2 -DNUM_WARPS=4 -DNUM_THREADS=4 \
                          -DL2_ENABLE -DL2_CACHE_SIZE=131072
DEFINES_4c2n4w4t_l2l3 := -DNUM_CLUSTERS=4 -DNUM_CORES=2 -DNUM_WARPS=4 -DNUM_THREADS=4 \
                          -DL2_ENABLE -DL3_ENABLE \
                          -DL2_CACHE_SIZE=131072 -DL3_CACHE_SIZE=524288
DEFINES_4c4n8w4t_l2l3 := -DNUM_CLUSTERS=4 -DNUM_CORES=4 -DNUM_WARPS=8 -DNUM_THREADS=4 \
                          -DL2_ENABLE -DL3_ENABLE \
                          -DL2_CACHE_SIZE=262144 -DL3_CACHE_SIZE=524288
```

---

## 5. Running Synthesis

All commands are run from `hw/syn/synopsys/`.

### Synthesize a single configuration

```bash
cd hw/syn/synopsys

# Step 1 — generate the filelist
make flist_1c1n4w4t

# Step 2 — elaboration check (optional but recommended before a full run)
make elab_1c1n4w4t

# Step 3 — full synthesis
make 1c1n4w4t
```

### Synthesize all five configurations sequentially

```bash
make all
```

### Synthesize only a subset

```bash
make 1c1n4w4t 1c2n4w4t
```

### Remove generated artifacts (keeps `libs/`)

```bash
make clean
```

### Refresh `libs/` from source (if `.db` or `.v` files change)

```bash
make libs
```

### Inspect results

```
reports/1c1n4w4t/area.rpt
reports/1c1n4w4t/timing.rpt
reports/1c1n4w4t/power.rpt
reports/1c1n4w4t/qor.rpt
reports/1c1n4w4t/hierarchy.rpt
reports/1c1n4w4t/constraints.rpt

results/1c1n4w4t/Vortex_netlist.v
results/1c1n4w4t/Vortex.ddc
results/1c1n4w4t/Vortex.sdc
```

---

## 6. Change Summary

| File | Change |
|---|---|
| `hw/rtl/VX_platform.vh` | Added `elsif ASIC_SYNTHESIS` branch: `STRING` = empty, all synthesis attributes = empty, added `USE_SRAM_MACRO` alias |
| `hw/rtl/VX_config.vh` | (a) `ASIC_SYNTHESIS` → `FPU_FPNEW`; (b) `ASIC_SYNTHESIS` guard on `IMUL_DPI`/`IDIV_DPI`; (c) `ASIC_SYNTHESIS` → `TCU_BHF` |
| `hw/rtl/libs/VX_sp_ram.sv` | Inside `g_sync → g_bram`: added `` `ifdef ASIC_SYNTHESIS `` block instantiating bsg_fakeram 1RW macros |
| `hw/rtl/libs/VX_dp_ram.sv` | Inside `g_sync → g_bram`: added `` `ifdef ASIC_SYNTHESIS `` block instantiating bsg_fakeram 1R1W macros |
| `hw/scripts/gen_sources.sh` | Added `! -name "*.bb.v"` to external-path `find` to exclude bsg_fakeram blackbox stubs |
| `hw/syn/synopsys/` | Replaced entirely: removed old TSMC 28nm models and broken scripts; added `gen_dc_sources.sh`, `dc_syn.tcl`, `dc_elab.tcl`, `Makefile`, and `libs/` with pre-copied `.db` / `.v` library files |
