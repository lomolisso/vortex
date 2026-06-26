# 20 — Debugging & Tracing

**Scope:** the tools for seeing inside a run — the `DP`/`DT` trace macros and their levels, how to
turn tracing on, UUID-based correlation, and the DTM/JTAG remote-debug path for source-level
debugging. Sources: [`../debug.h`](../debug.h), [`../main.cpp`](../main.cpp), [`../dtm/`](../dtm),
[`../instr_trace.h`](../instr_trace.h), [`../../../AGENTS.md`](../../../AGENTS.md) §4. Prereqs:
[06](06-instruction-lifecycle.md), [01](01-simulation-framework.md).

---

## 1. Trace macros ([`../debug.h`](../debug.h))

All debug output goes through a small macro family, gated by a compile-time level
`VX_DBG_DEBUG_LEVEL` (default **3**, [`debug.h:16`](../debug.h#L16)):

| Macro | Use |
|---|---|
| `DP(lvl, x)` | "DEBUG" line — `if (lvl <= VX_DBG_DEBUG_LEVEL) cout << x << endl` ([`debug.h:31`](../debug.h#L31)) |
| `DPH`/`DPN` | header-only / no-header continuations |
| `DT(lvl, x)` | **cycle-stamped** "TRACE" line — prefixes `SimPlatform::cycles()` ([`debug.h:49`](../debug.h#L49)) |
| `DTH`/`DTN` | trace header-only / continuation |

`DT` is the workhorse for pipeline tracing — every stage emits `DT(level, trace)` so you get a
cycle-stamped, ordered log of an instruction's journey. **Under `NDEBUG` all macros compile to
`do {} while(0)`** ([`debug.h:68`](../debug.h#L68)) — zero cost in release builds, so a debug build
is required to see traces.

### 1.1 Levels

Lower number = higher importance. A statement with `lvl=1` prints whenever the level is ≥1; `lvl=4`
prints only at the most verbose setting. Build with `-DVX_DBG_DEBUG_LEVEL=N` (or via `CONFIGS`) to
set the ceiling; `blackbox.sh --debug=<level>` (`AGENTS.md` §4) wires this for you.

## 2. UUID correlation

Every `instr_trace_t` has an immutable `uuid` (doc 06) and its `operator<<`
([`instr_trace.h:135`](../instr_trace.h#L135)) ends each line with `(#<uuid>)`; `MemReq`/`MemRsp`
do the same (doc 15). So to follow one instruction across the whole machine you
`grep '#12345'` the trace and watch it move fetch → decode → execute → memory → commit, cycle by
cycle. `instr_trace_t::log_once` ([`instr_trace.h:129`](../instr_trace.h#L129)) suppresses
repeat-logging of a trace that's stalled in place. `has_pending_instrs(wid)`
([`core.h:90`](../core.h#L90)) lets you check whether a warp still has in-flight work when a run
appears hung.

## 3. Running with traces

```bash
./ci/blackbox.sh --driver=simx --app=demo --debug=3          # build debug + dump level-3 trace
./ci/blackbox.sh --driver=simx --app=demo --debug=3 --log=run.log
```

The standalone simx binary takes `-s` (stats), `-V` (verbose debug-module logging), and a positional
program ([`main.cpp:39-83`](../main.cpp#L39)).

## 4. Source-level debugging — DTM / JTAG / OpenOCD

SimX implements the RISC-V **Debug Module** so you can attach a real debugger
([`../dtm/`](../dtm): `debug_module`, `jtag_dtm`, `remote_bitbang`). With the standalone binary's
`-d` flag ([`main.cpp:154`](../main.cpp#L154)):

- the processor is reset and the KMU started, warp 0's PC primed to the startup address;
- a `DebugModule` halts the hart at startup;
- a `remote_bitbang_t` server listens on `-p <port>` (default **9823**,
  [`main.cpp:46`](../main.cpp#L46)) for **OpenOCD**;
- the tick loop only advances the sim while the hart is not DTM-halted
  ([`main.cpp:198`](../main.cpp#L198)), implementing halt/resume/step at cycle boundaries.

The `Core` DTM accessors back this: `dtm_get_pc`/`dtm_set_pc`/`dtm_get_ireg`/`dtm_set_ireg`
([`core.h:129-132`](../core.h#L129)) read/write warp PC and integer registers (lane 0) directly,
which the `DebugModule` exposes as abstract register access. `-V` enables verbose DM logging.

## 5. SimX-as-oracle for RTL bugs

Per `AGENTS.md` §4, the canonical RTL-debug workflow uses SimX as the reference: bring the SimX
model to PASS, add matching cycle-stamped trace dumps (same CSV format) to both SimX and RTL, and
diff — the first divergence localizes the bug. The deterministic, UUID-correlated `DT` traces here
are exactly what makes that diff meaningful (doc 10 §5).

**Next:** [21 — Host Runtime & External Simulators](21-host-runtime-and-external-sim.md).
