# Task 3 — DSMEM Microarchitecture (Ideal Fabric) — SimX Proposal

**Status:** largely implemented in SimX (see §7 for as-built notes). **Scope:** SimX only
([`sim/simx/`](../../../sim/simx/)). Part of the [DSMEM & TMA Roadmap](README.md); this is
**Task 3**, and it **depends on [Task 1](1__socket_removal.md)** (core-resident SMEM) and
**[Task 2](2__per_core_dxa.md)** (per-core `DxaCore`).

**Goal:** add **DSMEM** — a thread on one SM directly addressing another SM's scratchpad — as a
complete microarchitecture, **except** the real SM-to-SM network, which is **idealized** here
(a fixed-delay `IdealTransport`) and replaced by a real NoC model in
[Task 5](5__omelet_cosimulation.md). TMA multicast, which rides this same machinery, is
[Task 4](4__tma_multicast.md).

> **DSMEM is additive scratchpad, not a memory mode.** DSMEM is cross-SM **shared memory**
> (scratchpad) — non-coherent, barrier-synchronized. It **coexists with** the global memory
> hierarchy (L1/L2/L3); it does **not** replace the L2/global path. It is therefore gated by its
> own **additive enable** (`VX_CFG_DSMEM_ENABLE`), orthogonal to the cache configuration — *not*
> an `L2 | DSMEM | NONE` selector. No coherence protocol and no memory-consistency model are in
> scope (roadmap §2).

---

## 1. Current state

- Shared memory is **core-local**: after Task 1 the `LocalMem` scratchpad is reachable only by
  its resident core (LSU/TCU/DXA). There is **no cross-core shared-memory addressing**.
- CTA-cluster membership exists only in software (`get_cluster_size()` / `get_cluster_rank()`),
  and nothing records **which core hosts which cluster rank** — cross-CTA sharing worked before
  only in the degenerate case where all ranks happened to land on one core.

---

## 2. Sub-task 3a — the DSMEM substrate (endpoints + ideal fabric)

The connectivity and bookkeeping, with no addressing semantics yet.

- **CTA-cluster identity plumbing.** Thread `cluster_id` and `cluster_rank` from the KMU grid
  walk → the dispatcher → the per-warp CTA CSRs, exposed as `VX_CSR_CTA_CLUSTER_ID` /
  `VX_CSR_CTA_CLUSTER_RANK`. `get_cluster_rank()` reads the real CSR (the old `cta_id % K` is
  wrong once a cluster's CTAs are on different cores).
- **CTA residency table** (`sim/simx/dsmem/cta_residency_table.{h,cpp}`, future `VX_cta_table`
  in `VX_cluster.sv`). Cluster-resident; for each in-flight CTA cluster records, per rank,
  `{host core, local CTA slot, LMEM slice base}`. Written by the per-core `CtaDispatcher`s at
  admission, retired when a CTA drains. It — not stride arithmetic — is how DSMEM addressing,
  TMA multicast, and the cluster barrier resolve a rank to a physical location.
- **Gang placement.** In DSMEM mode the dispatcher admits rank *r* of cluster *c* only on core
  `(c + r) % cores` (deterministic spread; the in-order KMU bus prevents cross-cluster
  interleaving), and records it in the table. One rank per core; cluster size `K ≤ cores`.
- **Core SMEM remote port + egress.** The core `LocalMem` gains one **remote** input port that
  arbitrates against the local LSU/TCU/DXA clients in the scratchpad's bank crossbar; a per-core
  egress arbiter merges the LSU-DSMEM and DXA-DSMEM traffic onto a single `dsmem_req_out`.
- **Ideal fabric.** `IdealTransport` (`sim/simx/dsmem/ideal_transport.{h,cpp}`) connects the
  per-core egress ports to the per-core remote ports with a **fixed delay** — no topology, no
  contention model. Requests route by `MemFlags::dsmem_target` (the destination core, resolved
  at the source from the residency table); responses back-route by a source index appended to
  the request tag. This is a deliberate **modeling placeholder** (not a future SV module); the
  real router-level network is [Task 5](5__omelet_cosimulation.md), and `IdealTransport` remains
  the fallback whenever co-simulation is off.

---

## 3. Sub-task 3b — DSMEM remote addressing

Thread-issued cross-core load/store on the 3a substrate.

- **Relative-region aperture.** The core's shared window widens (DSMEM-enabled builds only) so a
  few high bits above the per-CTA offset encode a **relative rank distance**: region 0 is the
  plain local window (unchanged semantics/bytes), region *r > 0* means "the CTA at rank
  `(my_rank + r) mod K`". Relative encoding keeps local accesses free of translation. On XLEN32
  only 2^16 B sit above `LMEM_BASE`, so the aperture uses **2 region bits**
  (`VX_CFG_DSMEM_REGION_BITS`) ⇒ cross-core `K ≤ 4`.
- **Instruction surface.** Free **custom0 / `funct7 = 0x05`** family:
  - `mapa.cluster` (funct3 0, ALU) — map a local shared pointer + target rank → a cluster-shared
    address (`r = (rank − my_rank) mod K`; region-encode).
  - `ld.cluster` / `st.cluster` (funct3 1/2, LSU) — remote word load/store on such an address.
  - `atom.cluster` (funct3 3) — **reserved, trapped** (cross-SM atomics are a future item, §6;
    the Task-3 kernels synchronize with barriers, not scoped atomics).
  Plain `lw`/`sw` to a `mapa`'d address route identically (routing is purely address-based).
  Intrinsics in [`vx_intrinsics.h`](../../../sw/kernel/include/vx_intrinsics.h);
  `get_cluster_rank()` in [`vx_spawn2.h`](../../../sw/kernel/include/vx_spawn2.h).
- **Routing.** Per lane, the `LocalMemSwitch` forks three ways: global → dcache, local-shared
  (region 0) → own `LocalMem`, **remote-shared (region ≠ 0) → a per-block `DsmemLsuAdapter`**
  (future `VX_dsmem_lsu_adapter`) that resolves region → absolute rank → `{core, slice base}`
  via the residency table, rebases the address to a plain shared address on the peer, tags it
  with `dsmem_target`, and hands it to the egress. Remote loads return data through the fabric;
  remote stores request an ack (`strsp`) used by the barrier drain (§4). The LSU sees remote
  loads uniformly — values are correct by construction (the response payload carries the peer's
  scratchpad data).

---

## 4. Sub-task 3c — the cluster barrier (drain + fence)

DSMEM correctness rests on a cross-SM barrier, **not** on coherence or a consistency model. The
barrier is a *mechanism*: it makes cross-SM writes visible by draining them, then ordering.

- **Cluster-scoped barrier.** A new `vortex::cluster_barrier` (a distinct barrier flag bit; the
  low byte carries `cluster_id % NUM_WARPS` so concurrent clusters use distinct hardware slots).
  Local arrivals complete one rank's worth, then escalate to `Cluster::cluster_barrier_arrive`,
  which counts rank arrivals against the residency table's participant set and broadcast-resumes
  the participant cores.
- **Drain.** Every remote store is counted per warp at LSU issue (the earliest post-retire
  point) and carries `strsp`; at barrier arrival the outstanding count becomes that many barrier
  **events**, each released when its store-ack returns over the fabric. Escalation waits for
  `arrivals == expected && events == 0` — i.e. every pre-barrier remote store has physically
  committed in the peers' SMEM banks.
- **Fence.** Structural: a warp is suspended in program order at the barrier, so no post-barrier
  read can hoist above it. The barrier is the sole ordering point (there is no proxy fence — the
  LSU pull path is generic→generic).

This drain-and-fence is a **barrier-implementation property**, the concrete testable form of
"model ordering, not coherence." It underwrites both DSMEM addressing here and per-receiver
completion in [Task 4](4__tma_multicast.md).

---

## 5. C++ kernel verification

`tests/regression/dsmem_pingpong/` — CTAs of one CTA-cluster on **different cores**; a thread on
core 1 reads/writes a scratchpad buffer **owned by core 0** via a DSMEM pointer built from
`vx_mapa_cluster(&smem[..], peer)`, synchronized with `vortex::cluster_barrier`. Asserts
peer-written values are observed correctly **and** that the barrier drains (no stale reads) —
including a phase where each rank remote-**stores** into a peer and the second barrier's drain is
what makes the writes visible.

```bash
CONFIGS="-DVX_CFG_DSMEM_ENABLE" \
  ./ci/blackbox.sh --driver=simx --app=dsmem_pingpong --clusters=1 --cores=2 --warps=64 --threads=32
CONFIGS="-DVX_CFG_DSMEM_ENABLE" \
  ./ci/blackbox.sh --driver=simx --app=dsmem_pingpong --clusters=1 --cores=4 --warps=64 --threads=32
```

Set the cluster dims so CTAs co-reside across cores. 120 s cap applies.

---

## 6. Risks & scope edges

- **Local/remote scratchpad deadlock.** The core `LocalMem` now serves LSU + TCU + DXA + the
  **remote** client; the remote port has independent buffering in the bank crossbar, so remote
  traffic cannot block the core's own LSU ingress. Stress with bidirectional pingpong.
- **Tag discipline across the fabric.** LSU tag → adapter → egress arbiter → fabric → peer bank
  crossbar → back. Verify the first remote load with `--debug` traces.
- **No coherence by design.** DSMEM is non-coherent scratchpad; correctness relies on the
  cluster barrier. Model ordering (§4), not coherence.
- **Cross-SM atomics (`atom.cluster`) — reserved, out of scope.** How RISC-V atomics behave
  across the non-coherent core boundary (whether the peer's SMEM banks host the atomic unit) is
  an atomics-implementation question; the Task-3/4 kernels use barriers, not scoped atomics.
  Known edge; not solved now.
- **Multi-cluster DSMEM** (a CTA cluster spanning Vortex-cluster boundaries) — out of scope; a
  CTA cluster stays within one Vortex cluster.

---

## 7. Implementation notes (as-built)

The DSMEM endpoints + ideal fabric are implemented in SimX; TMA multicast (below) is
[Task 4](4__tma_multicast.md). Key points:

- **Placement.** The per-core `CtaDispatcher` gang-places rank *r* of cluster *c* on core
  `(c + r) % cores` and records `{core, local CTA slot, LMEM slice base}` in the
  `CtaResidencyTable`, skipping the old K×lmem contiguity reservation (slices resolve through the
  table, not stride arithmetic).
- **Aperture.** Shared-window bits `[LMEM_LOG_SIZE +: 2]` encode the relative rank distance
  (region 0 = untouched local window). `mapa.cluster` (ALU), `ld.cluster`/`st.cluster` (LSU) in
  custom0 funct7=0x05; `atom.cluster` reserved-trapped. The `LocalMemSwitch` forks remote-shared
  lanes to a per-block `DsmemLsuAdapter` that resolves rank via the table, rebases the address,
  and rides the fabric; remote stores are ack'ed (`strsp`) for drain accounting.
- **Cluster barrier.** `vortex::cluster_barrier` (a distinct flag bit; low byte =
  `cluster_id % NUM_WARPS`). Local arrivals escalate to `Cluster::cluster_barrier_arrive`, which
  counts ranks against the residency table and broadcast-resumes participants. Drain: remote
  stores counted per warp at LSU issue → barrier events → released one-per-ack; fence is
  structural.
- **Fabric.** `IdealTransport` (`sim/simx/dsmem/ideal_transport.{h,cpp}`) — fixed-delay,
  `dsmem_target`-routed, tag-back-routed; an explicit placeholder, not a future SV module.
- **Knob.** The additive `VX_CFG_DSMEM_ENABLE` boolean described above is the as-built gate
  (the old `CLUSTER_MEM_MODE = L2|DSMEM|NONE` enum was removed; its non-DSMEM arms were never
  consumed by any code). Follows the `VX_CFG_EXT_DXA_ENABLE` idiom: default-false, opt in with
  `CONFIGS="-DVX_CFG_DSMEM_ENABLE"`.
- **Verification.** `dsmem_pingpong` passes: remote `ld.cluster`/`st.cluster` + plain `lw`/`sw`
  through mapa'd pointers (aperture decode) + region-0 self-map, barrier drain with mixed store
  paths, at 2-core K=2 and 4-core K=4 (the region-bit limit), with single- and multi-warp CTAs
  (`-w`), and with the L2 cache enabled (`--l2cache`) — the additive-to-L2 property is tested,
  not assumed. A debug-build trace shows the store-acks and rank escalations.
