# Task 3 — DSMEM Endpoints + TMA Multicast — SimX Proposal

**Status:** implemented in SimX (see §10 for the as-built notes). **Scope:** SimX only
([`sim/simx/`](../../sim/simx/)). Part of the
[Toy-H100 SimX Roadmap](toy_h100_simx_roadmap_proposal.md); this is **Task 3**, and it
**depends on [Task 1](socket_removal_simx_proposal.md)** (core-resident SMEM) and
**[Task 2](per_sm_dxa_simx_proposal.md)** (per-core `DxaCore`). Closes the **DSMEM / SM-to-SM
network** gap — the largest divergence from
[`../designs/h100_vortex_mapping.md`](../designs/h100_vortex_mapping.md) §5.4 — and the
"Hopper-style cross-core DSMEM path" and multicast items left open in
[`../designs/dxa_async_copy_multicast.md`](../designs/dxa_async_copy_multicast.md) §7.2, §7.6.

> **Scope amendment (Omelet pivot).** This task owns the **endpoint microarchitecture and
> semantics** (remote port, egress, residency table, addressing, cluster barrier, multicast
> fan-out). **NoC modeling — topology, per-hop timing, link accounting — is NOT modeled in
> SimX**: it is delegated to **[Task 4](omelet_noc_cosim_proposal.md)** (Omelet co-simulation).
> The originally-shipped `SmToSmNetwork` topology model was removed in that pivot; cross-core
> connectivity rides an explicit fixed-delay placeholder
> (`sim/simx/dsmem/ideal_transport.{h,cpp}`) until Task 4 lands, and permanently when
> co-simulation is off. §2's fabric text and §4.2's fan-out-policy study are retained below as
> design rationale but are **superseded** where they describe a SimX-internal topology model.

**Framing — substrate + two first-class consumers.** The SM-to-SM network is a **substrate**;
**DSMEM remote addressing** and **TMA multicast** are **two distinct, co-equal consumers** that
merely ride it. They are *not* the same capability:

- **DSMEM** (the `cluster.map_shared_rank` analog) is **thread-issued, fine-grained remote
  load/store** — a thread on SM A directly addresses SM B's scratchpad.
- **TMA multicast** is a **bulk async copy with hardware fan-out** — one SM issues one op, the
  engine reads a tile from global memory **exactly once**, and the data is broadcast into the
  scratchpad of *every CTA in the cluster mask*.

Delivered as three sub-tasks: **3a** the network substrate, **3b** DSMEM addressing, **3c** TMA
multicast. 3b and 3c are **peer deliverables**, each with its own design, faithfulness criteria,
and verification kernel.

---

## 1. Current state (as-built)

- The cluster offers a single memory facility: **L2** (a `Cache`,
  [`cluster.cpp:69`](../../sim/simx/cluster.cpp#L69)) or a passthrough. There is **no
  cross-core shared-memory addressing**.
- After Task 1, shared memory is core-local (the `LocalMem`), reachable only by the resident
  core / its DXA. After Task 2, DXA is per-SM and can write only its **own** core's
  scratchpad — cross-core multicast is deferred here.
- CTA-cluster membership exists in software: `get_cluster_size()` / `get_cluster_rank()` (CSRs
  `VX_CSR_CTA_CLUSTER_SIZE` / rank), and the multicast remap `dest[r] = issuer_smem +
  r·smem_stride` ([`../designs/dxa_async_copy_multicast.md`](../designs/dxa_async_copy_multicast.md) §1).

---

## 2. Sub-task 3a — SM-to-SM network fabric + core SMEM remote port (substrate)

The substrate, with **no consumer semantics yet**.

> **Superseded in part (Omelet pivot):** the pluggable-topology fabric described in the next
> two bullets was built, verified, then removed — a routing-function + per-hop-delay model is
> not a real NoC, and the program now models every Vortex fabric with Omelet
> ([Task 4](omelet_noc_cosim_proposal.md)). What survives of 3a as-built: the per-core ports,
> the `LocalMem` remote port, the residency table, the mode selector, and the endpoint
> contract (requests carry `dsmem_target`; responses back-route by tag) — plus the
> `IdealTransport` fixed-delay placeholder standing where the fabric was.

- **New SimObject `SmToSmNetwork`** (`sim/simx/dsmem/smtosm_network.{h,cpp}`), cluster-resident,
  1:1 with a future `VX_smtosm_network` in `VX_cluster.sv`. It exposes one request/response port
  pair per core; it decodes a remote request's **target core** and delivers to that core's
  `LocalMem` (SMEM) remote input. **The topology is a pluggable backend, not a fixed crossbar** —
  see below. The substrate's *interface* (per-core ports, target-core decode) is
  topology-independent, so 3b (DSMEM addressing) and 3c (multicast) ride it unchanged; only the
  routing function and per-hop timing model swap.
- **Topology is a first-class, swappable design axis (a deliberate H100 divergence).** H100's
  GPC-internal SM-to-SM fabric is effectively all-to-all; this program wants to *study*
  alternatives. Add a selector `VX_CFG_SMTOSM_TOPOLOGY = CROSSBAR | RING | MESH | TORUS | …` to
  [`VX_config.toml`](../../VX_config.toml), with **`CROSSBAR` the baseline/default**, plus
  per-link latency/bandwidth and router radix/buffering parameters. `SmToSmNetwork` is therefore
  an **abstraction** (a routing function + a per-hop timing model) with one backend per topology
  — *not* a concrete crossbar. The 1:1 mapping holds: `VX_smtosm_network` becomes a
  **parameterized RTL module family** selected by the same knob, not a single crossbar module.
  Emit per-class **trace metrics** (per-link bytes, hop count, contention/stall cycles) so
  topologies are quantitatively comparable — this is the NoC-characterization study the program
  targets (and §4.2's fan-out policy is topology-aware on top of it).
- **Core SMEM remote port.** Add a remote request/response input pair to the Task-1 `LocalMem`
  (SMEM), arbitrated with local LSU/TCU/DXA scratchpad accesses. Remote accesses target the
  **SMEM (`LocalMem`) only** (DSMEM is scratchpad, never the dcache) — no tags.
- **Cluster memory-mode selector.** Add `VX_CFG_CLUSTER_MEM_MODE = L2 | DSMEM | NONE` to
  [`VX_config.toml`](../../VX_config.toml). In `Cluster::Impl`
  ([`cluster.cpp`](../../sim/simx/cluster.cpp)): `L2` = today's `Cache`
  ([`cluster.cpp:69`](../../sim/simx/cluster.cpp#L69)); `DSMEM` = instantiate `SmToSmNetwork`,
  wire each core's remote port to it, cores reach global memory directly (toy uses `L3` as
  the global L2, per gap analysis §5.6); `NONE` = passthrough. H100 has *both* a global L2 and
  DSMEM; the toy disables the per-cluster L2, so `DSMEM` mode replaces it with the network. If a
  future config wants both, the modes can be made additive.
- **1:1 mapping.** `Cluster` owns the network; `Core`'s `LocalMem` (SMEM) exposes the remote port.
  The split in this task is along **feature** lines, not class lines, so the 1:1 discipline is
  undisturbed.

---

## 3. Sub-task 3b — DSMEM remote addressing

Thread-issued cross-core load/store on the 3a substrate.

- **Remote vs local shared decode.** Extend the core's shared-address handling (which after Task 1
  routes shared accesses to the core-resident SMEM) to distinguish **local shared** (home
  core) from **remote shared** (peer core). Encode the DSMEM address space as **(peer rank,
  offset)**, reusing the multicast remap (`issuer_base + rank·smem_stride`) as the cross-core
  translation seed; `peer rank` maps via the CTA-cluster CSRs to a **target core id** within
  the cluster.
- **Routing.** Local-shared → home core `LocalMem` (SMEM) input (Task 1 path). Remote-shared →
  core `dsmem_req_out` → `SmToSmNetwork` → target core `LocalMem` remote input; response
  returns the same way.
- **Generic-proxy instruction surface (owned by 3b).** Add the remote-shared LSU ops:
  **`mapa.cluster`** (map a local shared offset + target CTA rank → a cluster-shared address —
  the rank→core seed) and **`ld.cluster` / `st.cluster` / `atom.cluster`** (remote
  load/store/atomic). Recommended encodings in the free **custom0 / `funct7=0x05`** family
  (`mapa`=`funct3` 0, `ld/st/atom`=`funct3` 1..3) — `funct7=0x05` is free, no collisions — to
  finalize against the decode path ([`VX_decode.sv`](../../hw/rtl/core/VX_decode.sv),
  [`decode.cpp`](../../sim/simx/decode.cpp)). These are the **generic-proxy** ops; the
  async-proxy bulk-copy / mbarrier / `fence.proxy.async` ops are owned by
  [Task 4](proxy_model_simx_proposal.md). `sw/` intrinsics go in
  [`vx_intrinsics.h`](../../sw/kernel/include/vx_intrinsics.h); reuse `get_cluster_rank()`
  ([`vx_spawn2.h`](../../sw/kernel/include/vx_spawn2.h)) and the multicast remap — **no parallel
  rank scheme**.
- **Acceptance criterion — the cluster barrier must drain *and* fence** (the one hard
  consistency obligation, roadmap §10). The DSMEM cluster barrier (`group_barrier` /
  `barrier.cluster` analog) must: (1) **drain** — guarantee every remote scratchpad write issued
  before it has actually **landed** in the peer's core SMEM (`LocalMem`) (network drained, remote write
  committed); and (2) **fence** — fence each participating thread so post-barrier reads do not
  hoist above it. This is a barrier-*implementation* property, not a memory scope; it is the
  concrete, testable form of "model ordering, not coherence." **In proxy terms**
  ([Task 4](proxy_model_simx_proposal.md)): the LSU pull path is **generic→generic** — no proxy
  crossing, hence **no `fence.proxy.async`** — so the cluster barrier's drain+fence is the *sole*
  ordering point.

---

## 4. Sub-task 3c — TMA multicast (first-class)

Bulk async copy with hardware fan-out on the 3a substrate. **The defining property is the
single global read** — the issuing core's DXA reads the tile from GMEM **exactly once** and
the result fans out to N scratchpads. This property already exists in today's cluster-resident
DXA; the risk in the per-SM rebuild is **silently losing it**, so it is made explicit and
testable.

### 4.1 Faithfulness requirements (acceptance criteria)

1. **One global read, hardware fan-out.** The issuing core's DXA reads the tile from GMEM
   **exactly once**; the result fans out to N scratchpads (1 local + N−1 remote over the
   network). It must **not** degenerate into N cores each independently reading GMEM — that
   discards the L2/HBM-bandwidth saving that is the entire reason the feature exists. **Correct
   bytes in N scratchpads with N× global reads is a *failed* model.**
2. **Single issuer, passive receivers.** Exactly one core's DXA is the active engine (matches
   Hopper single-issuer multicast). Peers are **receive-only** — scratchpad written, barrier
   signaled, but their DXA does **not** participate in the read. Not a cooperative N-way gather.
3. **Per-destination completion.** Each receiver's barrier fires when **its** portion lands, not
   on one global completion. Preserve the `flags.dxa_notify_done → core->barrier_event_release(...)`
   contract **per receiver** across the network boundary (the mbarrier-transaction analog). This
   is the multicast-specific form of the 3b drain+fence obligation. When
   [Task 4](proxy_model_simx_proposal.md) promotes this proto-mechanism into the byte-granular
   **mbarrier** (`complete_tx::bytes`), per-receiver completion must survive intact — each
   receiver's mbarrier counts only **its own slice** of bytes.
4. **Fan-out is a NoC design choice — surface it, don't hardcode it** (see §4.2).

### 4.2 The fan-out policy is a NoC knob — swappable and traced *(superseded: now a Task-4/Omelet axis)*

> **Superseded by the Omelet pivot:** the fan-out policy and its per-link accounting are NoC
> properties and belong to [Task 4](omelet_noc_cosim_proposal.md), where router-level multicast
> replication exists physically rather than as an accounting trick. The DXA-side contract that
> survives in this task: the issuer replays per-receiver writes (source fan-out) with
> table-resolved destinations, and each receiver completes independently.

Whether `SmToSmNetwork` does **source-side replication** (the issuer unicasts N−1 copies) or
**true in-network multicast** (replication at routers) changes interconnect traffic
substantially — and that traffic is exactly what the NoC-characterization work targets. So:

- Model the fan-out mechanism as an **explicit, swappable policy**, not a hidden default.
- **Fan-out is topology-aware.** The valid replication points are defined by the §2 topology
  backend — "replication at routers" only exists where the topology *has* routers (ring/mesh/
  torus); on a `CROSSBAR` the choice degenerates to source-side replication vs. a single
  broadcast port. The fan-out policy is therefore a property *of the selected topology*, swept
  together with `VX_CFG_SMTOSM_TOPOLOGY`, not an independent crossbar-only knob.
- The per-class **trace dumps must expose the replication point and bytes-per-link**, so it can
  be measured (shared with the §2 per-link metrics).

The legacy "serial replay" note
([`../designs/dxa_async_copy_multicast.md`](../designs/dxa_async_copy_multicast.md) §7.2) is
**not** a throwaway optimization here — it is a **primary axis of study**.

---

## 5. Integration with Task 2 DXA

A per-SM DXA performing a multicast now issues its peer-targeted scratchpad writes through the
`SmToSmNetwork` (the 3a remote path / 3c fan-out policy), instead of the old cluster-resident
DXA reaching many cores directly. Single-core DXA is unchanged. The per-receiver completion
contract (Task 2 §5) carries straight into §4.1(3).

## 6. Memory-consistency notes (single-proxy through Task 3)

Through Task 3, treat Vortex as **single-proxy** and **do not add a scoped memory model**:

- **Single proxy (this task).** The DXA write and the consuming read share one proxy — Task 3
  needs **no `fence.proxy.async`**. The cluster barrier (drain+fence) is the only ordering point.
- **WGMMA-reads-multicast-written-SMEM** is not a cross-proxy hazard — it is just "is the tile
  present?", gated by the lockstep owner (`VX_tcu_lockstep`).

**Reconciliation:** the single-proxy treatment is correct *for Task 3*, whose correctness rests
on barriers. [Task 4](proxy_model_simx_proposal.md) **deliberately** introduces the two-proxy
model and a real `fence.proxy.async` as an **additive fidelity layer** — it does not change Task
3's correctness, only adds the generic↔async ordering Hopper exposes. Proxies (datapath
crossings) remain distinct from scopes (visibility radius); a scoped-consistency model stays out
of scope (roadmap §10).

The scope *radius* a Hopper `.cluster` qualifier would carry is realized instead by the
**network's span (= the cluster) plus the barrier** — nothing is annotated `.cluster`.

## 7. C++ kernel verification

Two new C++ kernels under [`tests/regression/`](../../tests/regression/) — genuine cross-core
traffic, no fakery — one per consumer:

1. **`dsmem_pingpong/`** (3b) — CTAs of one CTA-cluster on **different cores**; a thread on
   core 1 reads/writes a scratchpad buffer **owned by core 0** via a DSMEM pointer built
   from `get_cluster_rank()`, synchronized with `vortex::group_barrier`. Asserts peer-written
   values are observed correctly **and** that the barrier drains (no stale reads).
2. **`dxa_multicast/`** (3c) — per-SM DXA multicasts a GMEM tile into peer CTAs' scratchpad
   across cores. Asserts: **(a)** all N receivers' buffers correct; **(b)** each receiver's
   barrier fired (per-destination completion); **(c)** — the load-bearing check — **GMEM read
   traffic for the N-way multicast is ~1× the tile, not N×**, read from the `[csr_mpm_dxa]`
   perf counters / trace. **Byte-correctness alone is not sufficient and must not pass.**

```bash
# 3b: cross-core DSMEM
CONFIGS="-DVX_CFG_CLUSTER_MEM_MODE=DSMEM" \
  ./ci/blackbox.sh --driver=simx --app=dsmem_pingpong --clusters=1 --cores=2 --warps=64 --threads=32
# 3c: faithful TMA multicast — verify the single global read
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_CLUSTER_MEM_MODE=DSMEM" \
  ./ci/blackbox.sh --driver=simx --app=dxa_multicast --clusters=1 --cores=4 --threads=32 --perf=2
# regression: the cross-core multicast Task 2 deferred now passes faithfully
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_CLUSTER_MEM_MODE=DSMEM" \
  ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_wg_dxa_mcast --cores=4 --threads=32
```

Set `VX_CSR_CTA_CLUSTER_SIZE`/cluster dims so CTAs co-reside across cores. Sweep the §4.2
fan-out policy and inspect bytes-per-link in the trace. 120 s cap applies.

## 8. Risks

- **Local/remote scratchpad deadlock.** The core `LocalMem` (SMEM) now serves LSU + TCU + DXA +
  **remote network** clients; guarantee forward progress (no cyclic wait between a core awaiting
  a remote read and serving an inbound remote request).
- **Losing the single global read.** The single highest-value failure mode for 3c — guard it
  with the §7(c) assertion in CI, not just byte-correctness.
- **No coherence by design.** DSMEM in Hopper is not coherent; correctness relies on explicit
  cluster barriers. Model ordering (the 3b drain+fence), not coherence.
- **Address-map churn.** The DSMEM (peer-rank, offset) encoding must coexist with Task 1's
  scratchpad base/size and `VX_CSR_LOCAL_MEM_BASE`.

## 9. Proposed but not yet implemented (Task 3 scope edges)

- **Cross-core atomics on DSMEM.** The proposed kernels synchronize with **barriers**, not
  scoped atomics, so this does not arise for them. If we later want atomics on distributed
  shared memory, correctness depends on whether the SMEM (`LocalMem`) banks carry atomic units and how
  RISC-V atomics behave across the **non-coherent core boundary** — an atomics-implementation
  question (still orthogonal to scopes), the first place the unscoped/weak base would bite past
  barrier-based sync. **Known edge; do not solve now.**
- DSMEM coherence / cache-side participation (kept out — scratchpad-only).
- True in-network multicast replication if 3c ships source-side replication first (§4.2).
- Cross-issuer multicast (`dxa_multicast_from(rank)`) and SMEM swizzling
  ([`../designs/dxa_async_copy_multicast.md`](../designs/dxa_async_copy_multicast.md) §7.5–7.6).
- Multi-cluster DSMEM (a CTA cluster spanning Vortex-cluster boundaries) — out of scope; the toy
  keeps a CTA cluster within one cluster.
- Faithful scoped-consistency *behavior* (scope tree, proxy semantics) — separate, additive
  project per roadmap §10; **not** part of this task.

---

## 10. Implementation notes (as-built)

All of 3a/3b/3c is implemented in SimX behind `VX_CFG_CLUSTER_MEM_MODE_DSMEM`; the default
`L2` mode is compiled out entirely (byte-identical PERF verified on the pre-Task-3 oracle).
C++ builds select the mode with the **sibling flag form** (`-DVX_CFG_CLUSTER_MEM_MODE_DSMEM`,
like `-DVX_CFG_FPU_TYPE_STD`); the value form only normalizes inside `gen_config.py`.

- **Placement (the Hopper CWD analog).** Nothing in the pre-Task-3 dispatcher placed a
  cluster's CTAs — the same-core contiguity was an unenforced assumption. In DSMEM mode the
  per-core `CtaDispatcher` admits rank *r* of cluster *c* only on core `(c + r) % cores` (gang
  spread; the in-order KMU bus prevents cross-cluster interleaving), records the rank in the
  **`CtaResidencyTable`** (`sim/simx/dsmem/cta_residency_table.{h,cpp}`, future `VX_cta_table`)
  as `{core, local CTA slot, LMEM slice base}`, and skips the K×lmem contiguity reservation —
  slices resolve through the table, not stride arithmetic.
- **DSMEM aperture (3b).** XLEN32 leaves only 2^16 B above `LMEM_BASE`, so the aperture uses
  **2 region bits** (`VX_CFG_DSMEM_REGION_BITS`, K ≤ 4 cross-core): shared-window bits
  `[LMEM_LOG_SIZE +: 2]` encode the **relative** rank distance (region 0 = the untouched local
  window, so plain local accesses never translate). `mapa.cluster` (ALU), `ld.cluster` /
  `st.cluster` (LSU) live in custom0 funct7=0x05; `atom.cluster` (funct3=3) is reserved-but-
  trapped per §9. The `LocalMemSwitch` forks remote-shared lanes to a per-block
  **`DsmemLsuAdapter`** (future `VX_dsmem_lsu_adapter`) that resolves region → absolute rank →
  peer core/slice via the table, rebases the address, and rides the network; remote stores are
  ack'ed (`strsp`) for drain accounting.
- **Cluster barrier (3b drain+fence).** `vortex::cluster_barrier` uses free flag **bit 30**
  (bit 31 stays the global barrier); the low byte carries `cluster_id % NUM_WARPS` so concurrent
  clusters get distinct hardware slots. Local arrivals complete one rank's worth, then escalate
  to `Cluster::cluster_barrier_arrive`, which counts ranks against the residency table and
  broadcast-resumes participants. **Drain:** outstanding remote stores (counted per warp at LSU
  issue — the earliest post-retire point) convert to barrier events at arrive and are released
  one-per-ack; **fence** is structural (in-order warps cannot hoist past the suspension).
- **TMA multicast (3c).** The issuing DXA's serial replay resolves each `cta_mask` rank through
  the residency table: same-core ranks keep the local LMEM port; remote ranks leave through the
  core's DSMEM egress with `dsmem_target`. The **single global read is preserved structurally**
  (one work_list; only writes replay) and asserted by `tests/regression/dxa_multicast` via
  `VX_CSR_MPM_DXA_GMEM_READS` (measured exactly 1× tile at 4-way fan-out). Per-receiver
  completion re-targets `dxa_notify_bar_id` to the **receiver's own CTA slot** from the table
  (the issuer-relative `bar_id + cta_idx` arithmetic only holds same-core) and fires on the
  receiver core's network-ingress callback.
- **Transport (3a, post-pivot).** The originally-shipped `SmToSmNetwork` (four topology
  backends, fan-out policy, per-link accounting, MPM class 9) was **removed in the Omelet
  pivot** — that modeling now belongs to [Task 4](omelet_noc_cosim_proposal.md). In its place,
  **`IdealTransport`** (`sim/simx/dsmem/ideal_transport.{h,cpp}`) provides fixed-delay
  point-to-point delivery over the same endpoint contract: requests route by
  `MemFlags::dsmem_target`, responses back-route by source bits appended to the tag. It is an
  explicit modeling placeholder (deliberately **not** a future SV module) and remains the
  fallback transport whenever Omelet co-simulation is off.
- **Verification.** `dsmem_pingpong` (remote ld/st + drain, K=2/2-core and K=4/4-core, plus a
  debug-build trace showing the store-acks and rank escalations) and `dxa_multicast`
  (bytes + per-receiver completion + the 1×-GMEM assertion, measured `gmem_reads` exactly 1×
  tile and `lmem_writes` exactly N× at 4-way fan-out) pass on the ideal transport;
  `dxa_copy_mcast` (same-core degenerate) and `sgemm_tcu_wg_dxa_mcast` pass under DSMEM mode.
  Default L2 mode is byte-identical to the pre-Task-3 baseline on the 9-app oracle, with one
  understood exception: the two multicast apps differ **only in kernel instruction count**
  because `get_cluster_rank()` deliberately changed from the same-core-only `cta_id % K`
  computation to the hardware `VX_CSR_CTA_CLUSTER_RANK` read (fewer instructions per CTA;
  results and all sim-side paths unchanged — the other 7 apps are byte-identical).
