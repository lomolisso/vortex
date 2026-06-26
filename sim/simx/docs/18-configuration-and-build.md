# 18 — Configuration & Build

**Scope:** how SimX is configured and compiled — the `VX_CFG_*` parameter system, how
`VX_config.toml` becomes the `VX_config.h` the simulator includes, the build (`sim/simx/Makefile`),
the optional external-simulator builds, and how a `CONFIGS` override flows end-to-end. Sources:
[`../Makefile`](../Makefile), [`../../../VX_config.toml`](../../../VX_config.toml),
[`../../../ci/gen_config.py`](../../../ci/gen_config.py), [`../../../AGENTS.md`](../../../AGENTS.md)
§2/§4. Prereqs: [02](02-architecture-hierarchy.md).

---

## 1. Configuration is compile-time

Almost everything about the machine — core/warp/thread counts, cache geometry, which extensions
exist — is a **build-time constant**. The simulator sizes its channel arrays and registers its
modules at construction from `VX_CFG_*` macros (doc 02 §5), so **changing a parameter requires
recompiling the simx driver** (`AGENTS.md` §4: "`make tests` builds with default macros; use
`CONFIGS` + an explicit rebuild for non-default configs"). There is no runtime reconfiguration.

The baseline values live in two TOML files at the repo root:

- [`../../../VX_config.toml`](../../../VX_config.toml) — architecture parameters (`VX_CFG_*`).
- [`../../../VX_types.toml`](../../../VX_types.toml) — the ABI: memory map, CSR/DCR addresses,
  opcode/MPM enums.

## 2. TOML → header generation

`ci/gen_config.py` turns the TOML into the C/Verilog headers and the `-D` flags. The simx Makefile
invokes it to compute `XCONFIGS`:

```make
XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py \
              --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')
```

So the resolution order is **`VX_config.toml` baseline → overridden by `CONFIGS="-D…"` → emitted as
`XCONFIGS`** ([`Makefile`](../Makefile)). The generated `VX_config.h`/`VX_types.h` (under the build
tree, doc per `AGENTS.md` §2) are what `constants.h` and every source include.

## 3. The build ([`../Makefile`](../Makefile))

`sim/simx/Makefile` includes `../common.mk` and:

- compiles C++17 with `-Wall -Wextra -Werror -fPIC`, include paths for every sub-dir
  (`dxa/ tex/ om/ raster/ kmu/ mem/ dtm/`), softfloat, and ramulator;
- sets `-DVX_CFG_XLEN=$(XLEN)` and appends `$(CONFIGS)`;
- links **softfloat** (`softfloat.a`, for bit-accurate FP, doc 07) and **ramulator**
  (`-lramulator`, the external DRAM model);
- compiles the `SRCS` list — `processor/cluster/socket/core/scheduler/...` plus `mem/*` and `dtm/*`
  — into a shared library under `DESTDIR`.

### 3.1 Extensions are conditionally compiled

The Makefile filters `XCONFIGS` to add extension sources only when enabled, e.g.:

```make
ifneq ($(filter -DVX_CFG_EXT_TCU_ENABLE, $(XCONFIGS)),)   # add tcu/*.cpp
ifneq ($(filter -DVX_CFG_EXT_A_ENABLE,   $(XCONFIGS)),)   # add amo/amo_unit.cpp
ifneq ($(filter -DVX_CFG_EXT_V_ENABLE,   $(XCONFIGS)),)   # vpu/*  else  operands.cpp
```

So a disabled extension contributes **zero** code/cost — matching the `#ifdef VX_CFG_EXT_*` gates in
the headers (docs 02, 17). The `operands.cpp` vs `vpu/*` swap shows the same pattern for the vector
unit. A `CONFIG_FILE` stamp tracks the active config so the build re-runs when it changes.

### 3.2 External simulators

`USE_SST=1` and `USE_GEM5=1` (mutually exclusive, [`Makefile`](../Makefile)) build the SST/gem5
wrappers (doc [21](21-host-runtime-and-external-sim.md)). They change `LDFLAGS`/sources but not the
core model.

## 4. Running with a custom config

Per `AGENTS.md` §4, build the app *and* the driver with matching `CONFIGS`, then run from `build/`:

```bash
make -C tests/regression/<app> clean
CONFIGS="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_EXT_TCU_ENABLE" make -C tests/regression/<app>
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE" ./ci/blackbox.sh --driver=simx --app=<app> --threads=32
```

`blackbox.sh` exposes common knobs as flags (`--clusters --cores --warps --threads --l2cache
--l3cache`); anything else goes through `CONFIGS="-DVX_CFG_*"`. **The app and the driver must agree**
— `blackbox.sh` rebuilds only the driver, so a mismatched `-DVX_CFG_NUM_THREADS` in the app is a
silent bug (`AGENTS.md` §4). The XLEN variant uses a separate build dir (`build32/`/`build64/`).

**Next:** [19 — Profiling & Performance Counters](19-profiling-and-perf-counters.md).
