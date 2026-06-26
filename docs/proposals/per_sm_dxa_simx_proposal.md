# Task 2 — Per-Socket (Per-SM) DXA Relocation — SimX Proposal

**Status:** proposal (pre-implementation). **Scope:** SimX only
([`sim/simx/`](../../sim/simx/)). Part of the
[Toy-H100 SimX Roadmap](toy_h100_simx_roadmap_proposal.md); this is **Task 2**, and it
**depends on [Task 1](unified_l1_simx_proposal.md)** (socket-resident `UnifiedL1`). Closes the
**"TMA placement"** divergence from
[`../designs/h100_vortex_mapping.md`](../designs/h100_vortex_mapping.md) §5.5, and implements the
already-specified-but-unbuilt **"per-socket DXA relocation"** from
[`../designs/dxa_async_copy_multicast.md`](../designs/dxa_async_copy_multicast.md) §7.1.

**Goal:** move the DXA engine model from the cluster down to the socket. With `SOCKET_SIZE=1`
this yields a true **per-SM TMA**, writing directly into the socket's `UnifiedL1` scratchpad.

---

## 1. Current state (as-built)

`DxaCore` ([`sim/simx/dxa/dxa_core.{h,cpp}`](../../sim/simx/dxa/dxa_core.h)) is **cluster-resident**,
created in [`Cluster::Impl`](../../sim/simx/cluster.cpp#L128):

- `dxa_req_in[]` sized to **cores-per-cluster**, bound from each core's `SfuUnit` (the per-core
  `DxaUnit` front end).
- `gmem_req_out[]`/`gmem_rsp_in[]` → the cluster **L2 arbiter** at a dedicated DXA row
  ([`cluster.cpp:134`](../../sim/simx/cluster.cpp#L134), `kDxaRow=1`, below sockets).
- `lmem_req_out[]` sized to cores-per-cluster, each bound to **that core's** `LocalMem`
  scratchpad input port ([`cluster.cpp:158`](../../sim/simx/cluster.cpp#L158)); a write carrying
  `flags.dxa_notify_done` fires a `tx_callback` → `core->barrier_event_release(...)`.
- A worker pool of `VX_CFG_NUM_DXA_UNITS = max(1, ceil(NUM_CORES/8))` workers fans out from one
  queue.

So one cluster-wide engine serves all cores, reaching into each core's private LMEM.

## 2. Target state

| | Owns (after Task 2) |
|---|---|
| **Socket (== SM)** | L1I + `UnifiedL1` + **`DxaCore`** |
| **Cluster** | L2 (no DXA fabric) |

With `SOCKET_SIZE=1`: one `DxaCore` per socket == per SM == per-SM TMA. DXA's **GMEM-read side**
joins the socket's L1 path to L2; DXA's **SMEM-write side** targets the socket's `UnifiedL1`
scratchpad region (built in Task 1).

**Why per-SM is the *faithful* model.** Hopper's TMA units genuinely are per-SM, and multicast
genuinely traverses the SM-to-SM fabric. The current cluster-resident DXA performs multicast
only **by accident of being shared** — it holds direct LMEM ports into every core, with no
network. That arrangement is the *less* faithful one, so tearing it down is correct, not a
regression to apologize for. The faithful cross-SM data movement (DSMEM addressing and TMA
multicast) is rebuilt on an explicit SM-to-SM network in [Task 3](dsmem_smtosm_simx_proposal.md).

The socket-resident DXA established here is also the **async-proxy agent** of the two-proxy
model: [Task 4](proxy_model_simx_proposal.md) tags its `UnifiedL1` writes as the async proxy
(vs the generic LSU proxy) and defines the generic↔async crossing contract. No proxy logic is
added in Task 2 — this only notes which agent the DXA becomes.

---

## 3. Changes

**3.1 Relocate the engine.** Move `DxaCore::Create` from `Cluster::Impl`
([`cluster.cpp:128`](../../sim/simx/cluster.cpp#L128)) into `Socket::Impl`
([`socket.cpp`](../../sim/simx/socket.cpp)). `dxa_req_in[]` is now sized to
`cores_per_socket` (= 1). Delete the cluster-level DXA request/gmem/lmem fabric and its L2-arb
DXA row.

**3.2 GMEM-read path → socket L1 arbiter.** Instead of a DXA row on the *cluster* L2 arbiter,
add a **DXA input on the socket's L1 `MemArbiter`** ([`socket.cpp:78`](../../sim/simx/socket.cpp#L78)),
so DXA GMEM reads share the socket→L2 port with the icache/`UnifiedL1`-cache traffic. Priority
per the DXA design doc: **dcache > DXA** (the "dcache→DXA arbiter"). The cluster L2 then sees
only per-socket aggregated traffic — no DXA-specific rows.

**3.3 SMEM-write path → `UnifiedL1` scratchpad.** `DxaCore::lmem_req_out` (now a single port,
cores_per_socket=1) binds to a `UnifiedL1` scratchpad input port on the same socket. Preserve
the completion contract: the `flags.dxa_notify_done` write must still drive
`core->barrier_event_release(...)`; route that callback across the new socket↔core interface
introduced in Task 1 (the core exposed an SMEM interface; DXA completion notifies the resident
core's barrier state).

**3.4 Worker count.** `VX_CFG_NUM_DXA_UNITS` becomes **per-socket** (typically 1 for the toy).
Recompute `VX_CFG_DXA_MEM_PORTS` at socket granularity. Update
[`VX_config.toml`](../../VX_config.toml) (`NUM_DXA_UNITS` / `DXA_MEM_PORTS` exprs) and
re-`configure`.

**3.5 1:1 mapping.** SimX `DxaCore` now stands in for a socket-level `VX_dxa_core` (future RTL
move into `VX_socket.sv`, exactly the unbuilt RTL item). Keep ownership explicit: `Socket` owns
`DxaCore`; `Cluster` owns it no more.

---

## 4. C++ kernel verification

Reuse the existing DXA kernels — they should pass unchanged under per-SM DXA with
`SOCKET_SIZE=1`, and now scale by giving **each core its own DXA**:

```bash
# single-CTA async copy — per-SM TMA
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=dxa_copy --cores=1 --warps=64 --threads=32
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=dxa_kmajor_check
# fused with WGMMA (DXA feeds the tensor B-buffer via K-major)
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE" \
  ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_wg_dxa --threads=32
# multi-SM scaling: each socket runs its own DXA independently
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=dxa_copy --cores=4 --warps=64 --threads=32
```

Confirm DXA perf CSRs (`--perf=2`, the `[csr_mpm_dxa]` class) attribute traffic per socket. 120 s
cap applies.

## 5. Risks & sequencing

- **Cross-socket multicast regresses until Task 3.** Today's `dxa_copy_mcast` /
  `sgemm_tcu_wg_dxa_mcast` rely on the *cluster-resident* DXA reaching **multiple cores'** LMEM
  to multicast into co-resident CTAs. A per-SM DXA can only write its **own** socket's
  `UnifiedL1`. Therefore:
  - **Single-CTA / single-socket DXA: fully supported by Task 2.**
  - **Faithful TMA multicast — single global read, fan-out over the SM-to-SM network — is
    deferred to [Task 3c](dsmem_smtosm_simx_proposal.md).** In Task 2, gate/skip the multicast
    tests (or restrict to one socket) and document the regression explicitly — do not fake a
    cross-core path. This is the program-level sequencing risk called out in the roadmap §8.
  - **No Task-2 decision may preclude the Task-3c faithfulness requirements**: a single active
    issuer with passive receivers, exactly one global read per multicast, per-destination
    completion, and a swappable fan-out policy. The per-SM `DxaCore`'s GMEM-read path and the
    per-receiver `flags.dxa_notify_done → barrier_event_release` completion contract are
    preserved precisely so Task 3c can re-route the fan-out over the network without redesign.
- **Arbiter fairness/deadlock.** The new socket dcache→DXA arbiter shares the L2 port; preserve
  forward progress (DXA must not starve demand misses, and vice-versa).

## 6. Proposed but not yet implemented (Task 2 scope edges)

- Faithful cross-socket TMA multicast — single global read, hardware fan-out over the SM-to-SM
  network (→ [Task 3c](dsmem_smtosm_simx_proposal.md)).
- Multicast fan-out policy (source-side replication vs. in-network) and the serial-replay hoist
  — promoted from "optimization" to a primary NoC-study axis in Task 3c, per
  [`../designs/dxa_async_copy_multicast.md`](../designs/dxa_async_copy_multicast.md) §7.2.
