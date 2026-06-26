# SimX Simulator — Implementation Documentation

This is the implementation-detail reference for **SimX**, the Vortex GPGPU's cycle-level C++
simulator (sources under [`sim/simx/`](..), shared base in [`sim/common/`](../../common)). It
explains two intertwined things:

1. **How the simulator works** — the discrete-event framework that every component is built on.
2. **How the Vortex architecture is implemented** — every major C++ class, how they connect via
   typed channels, and how an instruction/request flows through the machine.

…plus the practical workflows: **how to add an architectural feature, how to profile it, and how
to debug it.**

> **Scope & freshness.** These docs describe the simulator *as it is today*. They are reference
> material for contributors; the Toy-H100 feature proposals live separately under
> [`docs/proposals/`](../../../docs/proposals). Every concrete claim cites `file:line`; links are
> relative to this folder (`sim/simx/docs/`), so `../core.cpp` is [`sim/simx/core.cpp`](../core.cpp).

---

## How to read this

Read in numeric order — it follows the life of the machine: framework → hierarchy → how a kernel
launches → the pipeline → how memory works → the accelerators → config/profiling/debug → how to
extend it. If you only have 30 minutes: read **01** (framework) and **06** (the `instr_trace_t`
lifecycle); those two unlock everything else.

| # | Doc | What it covers |
|---|---|---|
| — | [README.md](README.md) | This index, reading order, quick-reference tables. |
| 01 | [01-simulation-framework.md](01-simulation-framework.md) | `SimObject`/`SimChannel`/`SimPlatform`, the timing wheel, delta cycles, backpressure, and the **functional-vs-timing split**. The bedrock. |
| 02 | [02-architecture-hierarchy.md](02-architecture-hierarchy.md) | Processor → Cluster → Socket → Core ownership and channel wiring; tick order; how configuration drives instantiation. |
| 03 | [03-kernel-launch-and-dispatch.md](03-kernel-launch-and-dispatch.md) | DCR kernel descriptor, the KMU grid walk, `CtaDispatcher` slot/LMEM management, warp admission. |
| 04 | [04-pipeline-frontend.md](04-pipeline-frontend.md) | Fetch → decode → issue; warp scheduling; IPDOM divergence; the decoder; operand collection & scoreboard. |
| 05 | [05-pipeline-dispatch-execution.md](05-pipeline-dispatch-execution.md) | Dispatch, `op_type`→FU routing, SIMD-split (`num_pkts`) commit, writeback. |
| 06 | [06-instruction-lifecycle.md](06-instruction-lifecycle.md) | **The central data structure:** `instr_trace_t` from decode to commit; µop expansion. |
| 07 | [07-functional-units.md](07-functional-units.md) | `FuncUnit` base + latency model; ALU, FPU (+DPI), SFU. |
| 08 | [08-lsu-and-memory-ops.md](08-lsu-and-memory-ops.md) | LSU address-gen, coalescing, AMO, fences, response formatting. |
| 09 | [09-barriers-and-sync.md](09-barriers-and-sync.md) | Barrier unit, local/group/global barriers, async (`expect_tx`/event) completion. |
| 10 | [10-functional-emulator.md](10-functional-emulator.md) | The functional ISA model: warp/thread state, regfile, per-warp CSRs, trap handling. |
| 11 | [11-memory-hierarchy-overview.md](11-memory-hierarchy-overview.md) | L1/L2/L3/DRAM organization, address classification, the request path. |
| 12 | [12-cache-architecture.md](12-cache-architecture.md) | `Cache` internals: banks, MSHR, replacement, write-back/through, flush FSM. |
| 13 | [13-local-memory-scratchpad.md](13-local-memory-scratchpad.md) | `LocalMem`, `LocalMemSwitch`, per-CTA allocation, `read_word`. |
| 14 | [14-virtual-memory-mmu.md](14-virtual-memory-mmu.md) | MMU/TLB, SV32/SV39, page-table walk, fault → exception. |
| 15 | [15-memory-protocol-and-types.md](15-memory-protocol-and-types.md) | `MemReq`/`MemRsp`/`MemFlags`, `LsuReq`/`LsuRsp`, `hart_id` encoding, `AddrType`. |
| 16 | [16-arbitration-and-crossbars.md](16-arbitration-and-crossbars.md) | Arbiter policies, `TxArbiter`/`TxRxArbiter`/`TxRxCrossBar`, tag mangling. |
| 17 | [17-extension-engines.md](17-extension-engines.md) | The extension model; DXA & TCU in depth; graphics (TEX/OM/RASTER). |
| 18 | [18-configuration-and-build.md](18-configuration-and-build.md) | `VX_CFG_*`, TOML→header generation, the simx build/DPI, `CONFIGS` flow. |
| 19 | [19-profiling-and-perf-counters.md](19-profiling-and-perf-counters.md) | `PerfStats`, MPM CSR classes, `--perf`, end-of-run dump, adding a counter. |
| 20 | [20-debugging-and-tracing.md](20-debugging-and-tracing.md) | `DP()`/`DT()` macros, `--debug`, UUID correlation, DTM/JTAG remote debug. |
| 21 | [21-host-runtime-and-external-sim.md](21-host-runtime-and-external-sim.md) | The Processor C API, the runtime backend, SST/gem5 `memsim()` coupling. |
| 22 | [22-howto-add-features.md](22-howto-add-features.md) | Recipes: add an instruction/FU, an extension engine, a cache level, a perf counter. |

---

## Quick reference

### The architecture hierarchy (SimX class ↔ source ↔ what it owns)

| Level | SimX class | Source | Owns |
|---|---|---|---|
| Platform | `SimPlatform` | [`../../common/simobject.h`](../../common/simobject.h) | the event wheel + every `SimObject` |
| Top | `Processor` / `ProcessorImpl` | [`../processor.cpp`](../processor.cpp) | clusters, L3, memory, KMU; the `cycle()` loop |
| Cluster | `Cluster` | [`../cluster.cpp`](../cluster.cpp) | L2 + cluster-shared extension engines (DXA/TEX/OM/RASTER) |
| Socket | `Socket` | [`../socket.cpp`](../socket.cpp) | per-socket I-cache + D-cache + L1 arbiter |
| Core | `Core` | [`../core.cpp`](../core.cpp) | scheduler, FUs (ALU/FPU/LSU/SFU + per-core TCU), regfile/CSR, LMEM, barriers |

### The canonical data flows

- **An instruction:** fetch → `Decoder` → `Scheduler` (warp select) → operand collect → `Dispatcher`
  → a `FuncUnit` → writeback/commit. The whole journey rides one `instr_trace_t` (doc **06**).
- **A memory request:** `LsuUnit` → `LocalMemSwitch` (shared vs global) → either `LocalMem`
  (scratchpad) or coalescer → D-cache → L1 arb → L2 → L3 → DRAM. Packets are `MemReq`/`MemRsp`
  (doc **15**), routed by `hart_id`.
- **Time:** modules talk only through `SimChannel<Pkt>`; a `send(pkt, delay)` schedules delivery
  `delay` cycles later on `SimPlatform`'s timing wheel (doc **01**).

### Two ideas that recur everywhere

- **Functional-vs-timing split** — a FU computes its result *immediately* (0-cycle, functional)
  but the *response* is delayed by `latency_of()` cycles through a channel. Defined in doc **01**,
  used everywhere.
- **Backpressure discipline** — channels are bounded FIFOs; `full()` propagates upstream, so a
  stalled consumer cascades stalls backward through the pipeline (doc **01**).
