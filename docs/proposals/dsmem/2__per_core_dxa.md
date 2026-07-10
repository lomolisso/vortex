# Task 2 — Per-Core (Per-SM) DXA Relocation — SimX Proposal

**Status:** proposal (pre-implementation). **Scope:** SimX only
([`sim/simx/`](../../../sim/simx/)). Part of the
[DSMEM & TMA Roadmap](README.md); this is **Task 2**, and it
**depends on [Task 1](1__socket_removal.md)** (core-as-SM consolidation, core-resident
SMEM). Closes the
**"TMA placement"** divergence from
[`../../designs/h100_vortex_mapping.md`](../../designs/h100_vortex_mapping.md) §5.5, and implements the
already-specified-but-unbuilt **"per-SM DXA relocation"** from
[`../../designs/dxa_async_copy_multicast.md`](../../designs/dxa_async_copy_multicast.md) §7.1.

**Goal:** move the DXA engine model from the cluster down to the **core**. One `DxaCore` per core
yields a true **per-SM TMA**, writing directly into the core's `LocalMem` (SMEM) scratchpad.

---

## 1. Current state (as-built)

`DxaCore` ([`sim/simx/dxa/dxa_core.{h,cpp}`](../../../sim/simx/dxa/dxa_core.h)) is **cluster-resident**,
created in [`Cluster::Impl`](../../../sim/simx/cluster.cpp#L128):

- `dxa_req_in[]` sized to **cores-per-cluster**, bound from each core's `SfuUnit` (the per-core
  `DxaUnit` front end).
- `gmem_req_out[]`/`gmem_rsp_in[]` → the cluster **L2 arbiter** at a dedicated DXA row
  ([`cluster.cpp:134`](../../../sim/simx/cluster.cpp#L134), `kDxaRow=1`, below sockets).
- `lmem_req_out[]` sized to cores-per-cluster, each bound to **that core's** `LocalMem`
  scratchpad input port ([`cluster.cpp:158`](../../../sim/simx/cluster.cpp#L158)); a write carrying
  `flags.dxa_notify_done` fires a `tx_callback` → `core->barrier_event_release(...)`.
- A worker pool of `VX_CFG_NUM_DXA_UNITS = max(1, ceil(NUM_CORES/8))` workers fans out from one
  queue.

So one cluster-wide engine serves all cores, reaching into each core's private LMEM.

## 2. Target state

| | Owns (after Task 2) |
|---|---|
| **Core (== SM)** | L1I + dcache (L1D) + `LocalMem` (SMEM) + **`DxaCore`** |
| **Cluster** | L2 (no DXA fabric) |

One `DxaCore` per core == per SM == per-SM TMA. DXA's **GMEM-read side** joins the core's L1 path
to L2; DXA's **SMEM-write side** targets the core's `LocalMem` (SMEM) scratchpad (core-resident
after Task 1's socket removal).

**Why per-SM is the *faithful* model.** Hopper's TMA units genuinely are per-SM, and multicast
genuinely traverses the SM-to-SM fabric. The current cluster-resident DXA performs multicast
only **by accident of being shared** — it holds direct LMEM ports into every core, with no
network. That arrangement is the *less* faithful one, so tearing it down is correct, not a
regression to apologize for. The faithful cross-SM data movement (DSMEM addressing and TMA
multicast) is rebuilt on the DSMEM remote path in [Task 3](3__dsmem_microarchitecture.md) and
the TMA multicast fan-out in [Task 4](4__tma_multicast.md).

---

## 3. Changes

**3.1 Relocate the engine.** Move `DxaCore::Create` from `Cluster::Impl`
([`cluster.cpp:128`](../../../sim/simx/cluster.cpp#L128)) into `Core::Impl`
([`core.cpp`](../../../sim/simx/core.cpp)). `dxa_req_in[]` is now a single per-core port (= 1),
driven by that core's `SfuUnit`. Delete the cluster-level DXA request/gmem/lmem fabric and its
L2-arb DXA row.

**3.2 GMEM-read path → core L1 arbiter.** Instead of a DXA row on the *cluster* L2 arbiter,
add a **DXA input on the core's L1 `MemArbiter`** (the arbiter relocated into the core by Task 1),
so DXA GMEM reads share the core→L2 port with the icache/dcache traffic. Priority
per the DXA design doc: **dcache > DXA** (the "dcache→DXA arbiter"). The cluster L2 then sees
only per-core aggregated traffic — no DXA-specific rows.

**3.3 SMEM-write path → core `LocalMem` scratchpad.** `DxaCore::lmem_req_out` (now a single
per-core port) binds to a `LocalMem` (SMEM) input port on the **same core** — DXA and SMEM are
now co-resident in the core (Task 1), so this is a local bind, no cross-level interface. Preserve
the completion contract: the `flags.dxa_notify_done` write must still drive
`core->barrier_event_release(...)`, notifying the resident core's barrier state.

**3.4 Worker count.** `VX_CFG_NUM_DXA_UNITS` becomes **per-core** (typically 1 for the toy).
Recompute `VX_CFG_DXA_MEM_PORTS` at core granularity (Task 1 already dropped the `SOCKET_SIZE`
term from that expression). Update [`VX_config.toml`](../../../VX_config.toml) (`NUM_DXA_UNITS` /
`DXA_MEM_PORTS` exprs) and re-`configure`.

**3.5 1:1 mapping.** SimX `DxaCore` now stands in for a core-level `VX_dxa_core` (future RTL
move into `VX_core.sv`, the SM module). Keep ownership explicit: `Core` owns `DxaCore`; `Cluster`
owns it no more.

---

## 4. C++ kernel verification

Reuse the existing DXA kernels — they should pass unchanged under per-SM (per-core) DXA, and now
scale by giving **each core its own DXA**:

```bash
# single-CTA async copy — per-SM TMA
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=dxa_copy --cores=1 --warps=64 --threads=32
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=dxa_kmajor_check
# fused with WGMMA (DXA feeds the tensor B-buffer via K-major)
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE" \
  ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_wg_dxa --threads=32
# multi-SM scaling: each core runs its own DXA independently
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=dxa_copy --cores=4 --warps=64 --threads=32
```

Confirm DXA perf CSRs (`--perf=2`, the `[csr_mpm_dxa]` class) attribute traffic per core. 120 s
cap applies.

## 5. Risks & sequencing

- **Cross-core multicast regresses until Task 3.** Today's `dxa_copy_mcast` /
  `sgemm_tcu_wg_dxa_mcast` rely on the *cluster-resident* DXA reaching **multiple cores'** LMEM
  to multicast into co-resident CTAs. A per-SM DXA can only write its **own** core's
  `LocalMem` (SMEM). Therefore:
  - **Single-CTA / single-core DXA: fully supported by Task 2.**
  - **Faithful TMA multicast — single global read, fan-out over the SM-to-SM fabric — is
    deferred to [Task 4](4__tma_multicast.md)** (which builds on the DSMEM endpoints of
    [Task 3](3__dsmem_microarchitecture.md)). In Task 2, gate/skip the multicast tests (or
    restrict to one core) and document the regression explicitly — do not fake a cross-core
    path.
  - **No Task-2 decision may preclude the Task-4 faithfulness requirements**: a single active
    issuer with passive receivers, exactly one global read per multicast, and per-destination
    completion. The per-SM `DxaCore`'s GMEM-read path and the per-receiver
    `flags.dxa_notify_done → barrier_event_release` completion contract are preserved precisely
    so Task 4 can re-route the fan-out over the fabric without redesign.
- **Arbiter fairness/deadlock.** The new core dcache→DXA arbiter shares the L2 port; preserve
  forward progress (DXA must not starve demand misses, and vice-versa).

## 6. Proposed but not yet implemented (Task 2 scope edges)

- Faithful cross-core TMA multicast — single global read, hardware fan-out over the SM-to-SM
  fabric (→ [Task 4](4__tma_multicast.md)).
- The serial-replay hoist and per-receiver completion mechanics are preserved for Task 4, per
  [`../../designs/dxa_async_copy_multicast.md`](../../designs/dxa_async_copy_multicast.md) §7.2.
