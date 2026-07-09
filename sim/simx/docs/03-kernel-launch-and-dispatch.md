# 03 — Kernel Launch & Warp Dispatch

**Scope:** how a kernel goes from host launch parameters to warps executing on cores — the DCR
kernel descriptor, the **KMU** grid walk, and the per-core **`CtaDispatcher`** that admits warps
into slots and manages local-memory allocation. Sources: [`../kmu/kmu.h`](../kmu/kmu.h),
[`../kmu/kmu.cpp`](../kmu/kmu.cpp), [`../cta_dispatcher.h`](../cta_dispatcher.h),
[`../processor.cpp`](../processor.cpp). Prereqs: [02](02-architecture-hierarchy.md).

---

## 1. Vocabulary

- **CTA** (Cooperative Thread Array) = a thread block = one `block_size` group of warps that share
  local memory and can barrier together. The unit the dispatcher places onto a core.
- **CTA cluster** = `cluster_dim` CTAs that co-reside (the Hopper thread-block-cluster analog);
  used for cross-CTA local-memory sharing (DSMEM/DXA multicast, doc 17 / proposals).
- **Grid** = `grid_dim` CTAs total. The KMU walks it.

---

## 2. Step 1 — the host programs the descriptor (DCRs)

Before launch, the host writes the kernel descriptor through Device Control Registers. These route
to the **KMU** (the `VX_DCR_KMU_*` block; `ProcessorImpl::dcr_write` sends them straight to the KMU,
[`processor.cpp:268`](../processor.cpp#L268)). `Kmu::dcr_write` ([`kmu.cpp:47`](../kmu/kmu.cpp#L47))
latches each field:

| DCR | Field |
|---|---|
| `VX_DCR_KMU_STARTUP_ADDR0/1` | `PC_` (kernel start PC, 64-bit split across two writes) |
| `VX_DCR_KMU_KERNEL_ENTRY0/1` | `entry_` |
| `VX_DCR_KMU_STARTUP_ARG0/1` | `param_` (the kernel-arg pointer) |
| `VX_DCR_KMU_BLOCK_DIM_{X,Y,Z}` | `block_dim_[3]` (threads/CTA per axis) |
| `VX_DCR_KMU_GRID_DIM_{X,Y,Z}` | `grid_dim_[3]` (CTAs per axis) |
| `VX_DCR_KMU_CLUSTER_DIM_{X,Y,Z}` | `cluster_dim_[3]` (CTAs/cluster) |
| `VX_DCR_KMU_LMEM_SIZE` | `lmem_size_` (per-CTA scratchpad bytes) |
| `VX_DCR_KMU_BLOCK_SIZE` | `block_size_` (warps/CTA) |
| `VX_DCR_KMU_WARP_STEP_{X,Y,Z}` | `warp_step_[3]` (thread-index stride per warp) |

DCR names/addresses are defined in [`../../../VX_types.toml`](../../../VX_types.toml). The runtime
side that issues these writes is in doc [21](21-host-runtime-and-external-sim.md).

---

## 3. Step 2 — KMU walks the grid (one CTA per `step()`)

The **Kmu** ([`kmu.h:39`](../kmu/kmu.h#L39)) is a `SimObject` that *holds the descriptor and
enumerates CTAs*. `ProcessorImpl::run()`/`cycle()` calls `kmu_->start()` ([`kmu.cpp:73`](../kmu/kmu.cpp#L73)),
which validates the dims and arms `running_`. Note `on_reset()` deliberately **does not** clear the
descriptor — only the progression counters — because the descriptor is written *before* the run
and must survive `SimPlatform::reset()` ([`kmu.cpp:36-45`](../kmu/kmu.cpp#L36)).

`step(req)` ([`kmu.cpp:88`](../kmu/kmu.cpp#L88)) fills a `kmu_req_t` ([`kmu.h:22`](../kmu/kmu.h#L22))
for the next CTA and advances. The walk is **cluster-aware**, using two nested counters:

- `intra_offset_[3]` walks *within* a CTA cluster (fills the cluster first),
- `group_origin_[3]` advances by `cluster_dim` once the cluster is full, in **X→Y→Z** order
  ([`kmu.cpp:123-163`](../kmu/kmu.cpp#L123)).
- Effective `block_idx = group_origin + intra_offset` ([`kmu.cpp:92`](../kmu/kmu.cpp#L92)).
- `is_first_of_cluster` is set when `intra_offset == 0` ([`kmu.cpp:119`](../kmu/kmu.cpp#L119)) —
  the single source of truth the dispatcher uses for per-cluster LMEM contiguity (§4.2).

When `group_origin` wraps past `grid_dim` on all axes, `running_` clears ([`kmu.cpp:151`](../kmu/kmu.cpp#L151))
and the grid is exhausted.

---

## 4. Step 3 — `CtaDispatcher` admits warps onto a core

Each core's scheduler owns a **`CtaDispatcher`** ([`cta_dispatcher.h:46`](../cta_dispatcher.h#L46)).
It pulls CTAs from the KMU and admits **one warp rank per `step()`** into a free warp slot, returning
a fully-populated `cta_warp_record_t` ([`cta_dispatcher.h:26`](../cta_dispatcher.h#L26)) — PC, entry,
`tmask`, `cta_id`/`cta_rank`/`cta_size`, `thread_idx`/`block_idx`/`block_dim`/`grid_dim`, `param`,
`lmem_addr`, and `cluster_size`. The scheduler turns that record into a live warp (doc
[04](04-pipeline-frontend.md)).

### 4.1 Slot & in-flight state

The dispatcher tracks (all in [`cta_dispatcher.h:69-104`](../cta_dispatcher.h#L69)):

- a **current in-flight CTA** (`has_cta_`, `cur_slot_`, `rank_`, `block_size_rem_`) being placed
  warp-rank by warp-rank;
- a **peeked CTA** (`pending_cta_`) — the next CTA read *non-destructively* from the KMU each cycle
  (`kmu_->peek()`) and re-peeked until admission passes; it is **popped** (`kmu_->advance()`) only
  once a FIFO slot **and** LMEM are available. This peek-then-pop mirrors the RTL handshake
  (`VX_cta_dispatch.sv`: `kmu_bus_if.ready = table_notfull && lmem_ok`, the KMU holding the CTA valid
  until consumed). A core therefore never removes a CTA from the shared KMU it cannot seat — the
  earlier speculative pull did, stranding it and deadlocking grid-wide barriers at 100% occupancy;
- per-slot bookkeeping: `slot_rem_warps_` (live warps in a slot), `slot_lmem_size_`, `wid_to_slot_`,
  and a FIFO ring (`head_slot_`/`tail_slot_`) for in-order slot reuse.

`running()` ([`cta_dispatcher.h:59`](../cta_dispatcher.h#L59)) is `has_cta_ || kmu_->running()` —
i.e. the core is busy while a CTA is in flight or the KMU still has CTAs to hand out (a not-yet-popped
CTA still shows in `kmu_->running()`, since nothing is pre-consumed).

### 4.2 Local-memory allocation (ring buffer + cluster contiguity)

LMEM is a per-core ring buffer: `lmem_base_`/`lmem_capacity_`, an allocation pointer `lmem_tail_`,
and `free_size_`. A CTA reserves `lmem_size` bytes; **the first CTA of a cluster** (the KMU's
`is_first_of_cluster` flag) reserves a `cluster_dim × lmem_size` contiguous span and pre-wraps
`lmem_tail_` to 0 if the span would straddle the LMEM boundary
([`cta_dispatcher.h:106-109`](../cta_dispatcher.h#L106)). Contiguity matters because cross-CTA
sharing (DSMEM/DXA multicast) addresses peers by `issuer_base + rank·stride` (doc 17). The address
classification of these LMEM bytes is in doc [15](15-memory-protocol-and-types.md).

### 4.2b DSMEM mode: gang placement + the residency table

Under `VX_CFG_CLUSTER_MEM_MODE_DSMEM` (the Toy-H100 DSMEM mode), real CTA clusters
(K > 1) are **gang-placed** (the Hopper cluster-work-distributor analog): the peek-phase gate
admits rank r of cluster c only on core `(c + r) % cores`; other cores leave the CTA on the KMU
bus for its owner. Each admitted rank is recorded in the cluster's **`CtaResidencyTable`**
([`../dsmem/cta_residency_table.h`](../dsmem/cta_residency_table.h)) as
`{core, local CTA slot, LMEM slice base}` and retired when the CTA drains. The table — not
stride arithmetic — is how DSMEM remote addressing, DXA multicast fan-out, and the cluster
barrier resolve ranks, so the §4.2 K×span contiguity reservation is skipped in this mode.
Admission additionally backpressures on `can_admit()` (the table_notfull half of the handshake).

### 4.3 Reclaim

When a warp exits, the scheduler calls `warp_done(wid)` ([`cta_dispatcher.h:56`](../cta_dispatcher.h#L56));
once **all** warps of a CTA have exited, its LMEM slot is freed back into the ring, admitting the
next peeked CTA that was waiting for space.

---

## 5. The end-to-end launch lifecycle

```
host: dcr_write(VX_DCR_KMU_*  …)         program the descriptor          (processor.cpp:268 → kmu.cpp:47)
host: run() / first cycle()              reset + kmu_->start()           (processor.cpp:217 / :258)
loop (each SimPlatform::tick):
   per core: scheduler asks CtaDispatcher.step()
        ├─ peek next CTA from Kmu.peek()  non-destructive, if no in-flight (kmu.cpp:95)
        ├─ admit FIFO slot + LMEM         re-peek next cycle if full     (cta_dispatcher §4.2)
        ├─ pop it via Kmu.advance()       only once admitted             (kmu.cpp:132)
        └─ emit cta_warp_record_t → scheduler activates the warp slot
   warps fetch/execute (docs 04–10) …
   warp exits → CtaDispatcher.warp_done() → frees LMEM when CTA drains   (cta_dispatcher §4.3)
until: grid exhausted (Kmu not running) AND all warps done AND inflight_count()==0  (processor.cpp:235)
host: dcr_read(VX_DCR_BASE_CACHE_FLUSH)  flush dirty lines to DRAM       (processor.cpp:282)
host: read back results
```

Two things to carry forward: the **CTA cluster** (`cluster_dim`) is the unit of cross-CTA sharing
that the extension engines and the Toy-H100 proposals build on (doc 17); and admission is
**back-pressured by LMEM**, so a core's occupancy is bounded by `lmem_capacity_ / lmem_size`, not
just warp-slot count.

**Next:** [04 — Pipeline Frontend](04-pipeline-frontend.md).
