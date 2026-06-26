# 19 — Profiling & Performance Counters

**Scope:** how SimX measures itself — the per-module `PerfStats` structs, the MPM (machine
performance-monitoring) CSR classes a kernel can read, how to turn profiling on, and how to add a
counter. Sources: [`../core.h`](../core.h), [`../mem/cache.h`](../mem/cache.h),
[`../processor_impl.h`](../processor_impl.h), [`../../../VX_types.toml`](../../../VX_types.toml),
[`../../../ci/blackbox.sh`](../../../ci/blackbox.sh). Prereq: [02](02-architecture-hierarchy.md).

---

## 1. Two layers of counters

1. **Host-side `PerfStats`** — every timing module owns a `PerfStats` struct it increments in
   `on_tick`, summed up the hierarchy with `operator+=`. Examples:
   - `Core::PerfStats` ([`core.h:37`](../core.h#L37)) — the richest: `cycles`, `instrs`,
     `sched_idle`, active/stalled/issued warps & threads, per-stage stalls (`fetch/ibuf/scrb/opds/
     alu/fpu/lsu/sfu/tcu`), `branches`, `divergence`, per-FU instr counts, `ifetch/load_latency`.
   - `Cache::PerfStats` ([`cache.h:47`](../mem/cache.h#L47)) — reads/writes/misses/evictions/
     bank_stalls/mshr_stalls/mem_latency.
   - `LocalMem::PerfStats` (reads/writes/bank_stalls), `Memory::PerfStats` (bank_stalls),
     `DxaCore`/`TcuUnit` PerfStats (docs 17).
   - `ProcessorImpl::PerfStats` ([`processor_impl.h:26`](../processor_impl.h#L26)) aggregates L3 +
     DRAM + the global `mem_reads/writes/latency`.

2. **Kernel-visible MPM CSRs** — a subset of the above exposed to running kernels via CSR reads.

---

## 2. The MPM CSR model

Performance counters are grouped into **classes** ([`VX_types.toml:393-401`](../../../VX_types.toml#L393)):

```
VX_DCR_MPM_CLASS_BASE=0  CORE=1  MEM=2  TEX=3  RASTER=4  OM=5  DXA=6  TCU=7  VM=8
```

A kernel selects a class by writing the `VX_DCR_BASE_MPM_VALUE`/class DCR, then reads the counters
through the `VX_CSR_MPM_*` window starting at `VX_CSR_MPM_BASE = 0xB00`
([`VX_types.toml:5`](../../../VX_types.toml#L5)); user counters begin at `VX_CSR_MPM_USER = 0xB03`.
Counters are 64-bit, split into a low CSR (`0xB0x`) and a high half (`0xB8x`, the `_H` variants) —
e.g. the `[csr_mpm_core]` class ([`VX_types.toml:411`](../../../VX_types.toml#L411)) maps
`VX_CSR_MPM_SCHED_IDLE=0xB03`, `ACTIVE_WARPS=0xB04`, `STALLED_WARPS=0xB05`, `ISSUED_WARPS=0xB06`, …
The `CsrUnit` (doc 10 §3) services these reads, pulling from the core's live `PerfStats`.

---

## 3. Turning profiling on

From `build/` (`AGENTS.md` §4):

```bash
./ci/blackbox.sh --driver=simx --app=sgemm --perf=1   # 1 = pipeline class, 2 = memsys
./ci/blackbox.sh --driver=simx --app=demo --perf=2
```

`--perf=<class>` selects which counter class the runtime programs (pipeline vs memory-system); the
driver prints the aggregated stats at the end of the run. The standalone simx binary's `-s` flag
([`main.cpp:53`](../main.cpp#L53)) likewise enables a stats dump. Per-core breakdowns come from
`ProcessorImpl::get_first_core()->perf_stats()` (doc 02).

---

## 4. Adding a counter (the 5-step recipe)

1. **Add a field** to the relevant `PerfStats` struct (e.g. `Core::PerfStats`,
   [`core.h:37`](../core.h#L37)) and to its `operator+=`.
2. **Increment it** in the owning module's `on_tick`/handler where the event occurs.
3. **Expose it** through `CsrUnit` (doc 10 §3) by mapping a `VX_CSR_MPM_*` address in its class.
4. **Declare the CSR** in the matching `[csr_mpm_*]` block of
   [`../../../VX_types.toml`](../../../VX_types.toml) (low + `_H` high half) and, if it's a new
   class, add a `VX_DCR_MPM_CLASS_*` id.
5. **Rebuild** the driver (and re-`configure` if you touched the TOML — doc 18) and read it with
   `--perf`.

This is one of the recipes expanded in doc [22](22-howto-add-features.md) §4.

**Next:** [20 — Debugging & Tracing](20-debugging-and-tracing.md).
