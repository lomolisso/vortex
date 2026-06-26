# 02 — Architecture Hierarchy

**Scope:** the five nesting levels of the simulated machine — **Platform → Processor → Cluster →
Socket → Core** — what each level owns, how they are wired with `SimChannel`s, how the per-cycle
clock actually advances them, and how the `VX_CFG_*` counts drive instantiation. Sources:
[`../processor.cpp`](../processor.cpp), [`../cluster.h`](../cluster.h), [`../socket.h`](../socket.h),
[`../core.h`](../core.h), [`../constants.h`](../constants.h). Prereq: [01](01-simulation-framework.md).

---

## 1. The levels at a glance

```
SimPlatform                      the event wheel + clock (doc 01)
└─ Processor / ProcessorImpl     processor.cpp — owns: Kmu, clusters[], L3 cache, Memory(DRAM)
   ├─ Kmu                        kernel grid walker (doc 03)
   ├─ Cluster  × NUM_CLUSTERS    cluster.cpp — owns: L2 cache, extension engines (DXA/TEX/OM/RASTER)
   │  └─ Socket × NUM_SOCKETS    socket.cpp  — owns: I-cache, D-cache (CacheCluster), L1 arbiter
   │     └─ Core × SOCKET_SIZE   core.cpp    — owns: Scheduler, FUs, regfile/CSR, LocalMem, barriers, TCU
   ├─ L3 cache                   the LLC when enabled, else a transparent bypass arbiter
   └─ Memory ("dram")            the DRAM timing model
```

Every box is a `SimObject` (doc [01](01-simulation-framework.md)); every arrow between boxes is a
pair of `SimChannel<MemReq>` / `SimChannel<MemRsp>` (doc [15](15-memory-protocol-and-types.md)).

> **pImpl everywhere.** `Cluster`, `Socket`, and `Core` each expose a thin public header and hide
> the real child-instantiation + channel-binding inside a private `class Impl; Impl* impl_;`
> (e.g. [`cluster.h:127`](../cluster.h#L127), [`core.h:167`](../core.h#L167)). When you want to
> see *how* children are created and wired, read the `Impl` constructor in the corresponding
> `.cpp`, not the header.

---

## 2. `ProcessorImpl` — the top

[`ProcessorImpl`](../processor.cpp#L36) is the root container. Its constructor builds, in order:

1. **KMU** — `Kmu::Create("kmu")` ([`:44`](../processor.cpp#L44)), the kernel grid walker (doc 03).
2. **DRAM** — `Memory::Create("dram", …)` ([`:47`](../processor.cpp#L47)) with
   `PLATFORM_MEMORY_NUM_BANKS`, `L3_MEM_PORTS`, `MEM_BLOCK_SIZE`, `MEM_CLOCK_RATIO`.
3. **Clusters** — one per `VX_CFG_NUM_CLUSTERS` ([`:57`](../processor.cpp#L57)).
4. **L3 cache** — `Cache::Create("l3cache", …)` ([`:64`](../processor.cpp#L64)). When
   `VX_CFG_L3_ENABLED` it is the **LLC**; when disabled it is constructed in *bypass* mode (a
   transparent arbiter) and the L2 (or L1) becomes the LLC — note the `is_llc` arg is
   `VX_CFG_L3_ENABLED != 0` ([`:79`](../processor.cpp#L79)).

### 2.1 Memory wiring

Clusters bind upward to L3, L3 binds to DRAM:

- `cluster.mem_req_out[j] → l3cache.core_req_in[i*L2_MEM_PORTS + j]` and the symmetric `rsp`
  ([`:103-108`](../processor.cpp#L103)).
- `l3cache.mem_req_out[i] → memsim.mem_req_in[i]` and symmetric ([`:111-114`](../processor.cpp#L111)).

The DRAM ports also carry the **global memory perf counters** via `tx_callback`s
([`:117-128`](../processor.cpp#L117)): every `MemReq` increments `perf_mem_reads_`/`writes_` and
pending-read tracking; every `MemRsp` decrements pending reads. This is how
`perf_mem_latency_` (accumulated each cycle in the run loop) is derived without per-request
timestamps.

### 2.2 AMO write-through invariant (a real footgun)

When the atomic extension is enabled, **every cache above the LLC must be write-through**, enforced
by `static_assert`s ([`:83-100`](../processor.cpp#L83)): a write-back intermediate could absorb a
store the LLC never sees, so a later `SC` on that line would *spuriously succeed* (RVA permits
spurious failure, not success). See doc [12](12-cache-architecture.md) for the AMO passthrough path.

---

## 3. The per-cycle clock — how levels actually advance

A common misconception is that `ProcessorImpl` ticks clusters → sockets → cores in order. It does
**not**. `ProcessorImpl::cycle()` ([`:251`](../processor.cpp#L251)) is just:

```cpp
SimPlatform::instance().tick();   // ticks ALL active SimObjects this cycle (doc 01 §4.3)
perf_mem_latency_ += perf_mem_pending_reads_;
return this->any_running();
```

Ordering among modules is determined by **`SimPlatform`'s active-object registration order plus
the delta-cycle mechanism** (doc [01](01-simulation-framework.md#4-simplatform--the-clock-and-the-timing-wheel)),
not by hand-rolled per-level sequencing. A producer's 0-delay `send` is drained before the next
module ticks, so combinational chains resolve within a cycle regardless of who registered first.

### 3.1 The run loop and termination

`run()` ([`:215`](../processor.cpp#L215)) calls `kmu_->start()` then loops `SimPlatform::tick()`
until **`!any_running() && SimChannelBase::inflight_count() == 0`** ([`:235`](../processor.cpp#L235)).
The `inflight_count` half is essential: a core can be idle while a writethrough or cache-pipe
packet is still traveling, and that packet sits in a `SimChannel`, so the global inflight counter
(doc [01](01-simulation-framework.md#37-introspection)) is the authoritative "machine is drained"
signal. `cycle()` ([`:251`](../processor.cpp#L251)) is the same step exposed to an *external*
driver (one cycle at a time), with lazy first-call `reset()` + `kmu start`.

### 3.2 Cache flush at teardown

`flush_caches()` ([`:154`](../processor.cpp#L154)) drains dirty lines to DRAM **inside-out**:
issue all L1 `flush_begin()` (dcache + icache + graphics caches) and tick until every surface
reports `flush_done()` *and* `inflight_count()==0`; then the same for L2; then L3
([`:194-212`](../processor.cpp#L194)). Triggered by the host reading
`VX_DCR_BASE_CACHE_FLUSH` ([`:282`](../processor.cpp#L282)) before reading back results. The walk
is a no-op on write-through caches (`Cache::flush_begin` early-exits, doc 12).

### 3.3 Device Control Registers (DCR)

`dcr_write` ([`:266`](../processor.cpp#L266)) routes **KMU DCRs** (`VX_DCR_KMU_STATE_BEGIN..END`)
to the KMU and **broadcasts everything else to every cluster** (which fans to sockets→cores).
`dcr_read` ([`:281`](../processor.cpp#L281)) special-cases the cache-flush address. This is the
host→device configuration channel; see doc [03](03-kernel-launch-and-dispatch.md) for the kernel
descriptor DCRs and doc [21](21-host-runtime-and-external-sim.md) for who writes them.

---

## 4. Cluster, Socket, Core — ownership

### 4.1 `Cluster` ([`../cluster.h`](../cluster.h))
Owns the **L2 cache** and the **cluster-shared extension engines** — DXA, TEX, OM, RASTER — each
behind a `VX_CFG_EXT_*` gate ([`cluster.h:39-56`](../cluster.h#L39), reflected in `PerfStats`).
Exposes `mem_req_out`/`mem_rsp_in` (to L3), `running()`, `get_exitcode()`,
`global_barrier_arrive()`, the per-level flush hooks, and `dxa_core()`/`raster_core()` accessors so
per-core SFUs can dispatch to the shared engines. (TCU is **per-core**, not here — doc 17.)

### 4.2 `Socket` ([`../socket.h`](../socket.h))
Owns the per-socket **I-cache and D-cache** (each a `CacheCluster`, doc 12) and the **L1 arbiter**
merging them toward L2 (icache prioritized). Exposes `mem_req_out`/`mem_rsp_in` (to the cluster's
L2), `core(idx)`, and `global_barrier_arrive`/`global_barrier_resume` (the cross-core barrier
cascade, doc 09). `PerfStats` = `{icache, dcache}`.

### 4.3 `Core` ([`../core.h`](../core.h))
The big one. Owns the `Scheduler`, all functional units (`SfuUnit`, `TcuUnit`, …), the register
file/`CsrUnit`, `LocalMem` + `LocalMemSwitch` + `MemCoalescer`, and the barrier machinery. Exposes
`icache_req_out`/`dcache_req_out` channel arrays, the rich `PerfStats` struct
([`core.h:37-70`](../core.h#L37)) that powers doc [19](19-profiling-and-perf-counters.md), the
barrier API (`barrier_arrive`/`barrier_wait`/`barrier_event_attach`/`barrier_event_release`,
doc 09), `wspawn`/`setTmask`, and DTM debug accessors (doc 20). Its internals are documented across
docs [04](04-pipeline-frontend.md)–[10](10-functional-emulator.md).

---

## 5. How counts drive instantiation

All derived in [`../constants.h`](../constants.h):

| Quantity | Formula | Source |
|---|---|---|
| Sockets per cluster grouping | `NUM_SOCKETS = up(NUM_CORES / SOCKET_SIZE)` | [`:61`](../constants.h#L61) |
| L2 input ports | `L2_NUM_REQS = NUM_SOCKETS * L1_MEM_PORTS` | [`:63`](../constants.h#L63) |
| L3 input ports | `L3_NUM_REQS = NUM_CLUSTERS * L2_MEM_PORTS` | [`:64`](../constants.h#L64) |
| Warps per issue slot | `PER_ISSUE_WARPS = NUM_WARPS / ISSUE_WIDTH` | [`:66`](../constants.h#L66) |
| LSU request fan | `LSU_NUM_REQS = NUM_LSU_BLOCKS * NUM_LSU_LANES` | [`:54`](../constants.h#L54) |
| D-cache request fan | `DCACHE_NUM_REQS = NUM_LSU_BLOCKS * DCACHE_CHANNELS` | [`:59`](../constants.h#L59) |

These constants come from `VX_config.h`, generated from
[`../../../VX_config.toml`](../../../VX_config.toml) (doc [18](18-configuration-and-build.md)).
Changing any count is a **build-time** change — the channel arrays are sized at construction, so a
new core count requires recompiling the simulator (doc 18 explains why).

**Next:** [03 — Kernel Launch & Warp Dispatch](03-kernel-launch-and-dispatch.md).
