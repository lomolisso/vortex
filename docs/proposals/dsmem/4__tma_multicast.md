# Task 4 — TMA Multicast & TMA Features — SimX Proposal

**Status:** multicast implemented in SimX (see §6); the other TMA features are proposed.
**Scope:** SimX only ([`sim/simx/`](../../../sim/simx/)) plus the `sw/` intrinsic surface. Part of
the [DSMEM & TMA Roadmap](README.md); this is **Task 4**, and it **depends on
[Task 2](2__per_core_dxa.md)** (per-core `DxaCore`) and
**[Task 3](3__dsmem_microarchitecture.md)** (the DSMEM endpoints + ideal fabric).

**Goal:** implement the interesting **TMA** (Tensor Memory Accelerator) capabilities on top of
the per-SM DXA and the DSMEM machinery — headlined by **multicast** — over the idealized
SM-to-SM fabric (the real NoC is [Task 5](5__omelet_cosimulation.md)).

> **TMA-exclusive scope.** This task adds **TMA** features only. Cross-SM DSMEM *atomics*
> (`atom.cluster`) are a DSMEM extension, not a TMA feature, and are **not** here (reserved,
> Task 3 §6). No memory-consistency or proxy machinery — no `fence.proxy.async`, no two-proxy
> model, no scopes (roadmap §2). TMA completion is expressed with the existing **barrier**
> mechanism, nothing more.

---

## 1. What TMA offers (the feature set)

Hopper's TMA is a per-SM bulk async-copy engine (Vortex's `DxaCore`, per-SM after Task 2). Beyond
a plain tile copy it provides:

1. **Multicast** — one SM issues one copy; the tile is read from global memory **exactly once**
   and fanned out into the scratchpads of **every CTA in the cluster mask**. *(§2 — the
   headline, implemented.)*
2. **Async-copy completion** — a "the bulk copy has landed" signal the consumer waits on before
   reading the tile. *(§3 — barrier-based.)*
3. **Tiled / multi-dimensional copies** — 2D/3D box copies with a tensor descriptor (base,
   extents, strides, element size). *(§4 — additive.)*

All three ride the Task-2 per-SM DXA and, for the cross-SM parts, the Task-3 DSMEM endpoints +
ideal fabric.

---

## 2. TMA multicast (the headline)

Bulk async copy with hardware fan-out. **The defining property is the single global read** — the
issuing SM's DXA reads the tile from GMEM once and the result fans out to N scratchpads. This
already exists in the pre-Task-2 cluster-resident DXA "by accident of being shared"; the per-SM
rebuild must **not silently lose it**, so it is made explicit and testable.

### 2.1 Faithfulness requirements (acceptance criteria)

1. **One global read, hardware fan-out.** The issuing SM's DXA reads the tile from GMEM
   **exactly once**; the result fans out to N scratchpads (1 local + N−1 remote over the ideal
   fabric). It must **not** degenerate into N SMs each reading GMEM — that discards the
   L2/HBM-bandwidth saving that is the entire point. **Correct bytes in N scratchpads with N×
   global reads is a *failed* model.**
2. **Single issuer, passive receivers.** Exactly one SM's DXA is the active engine (Hopper
   single-issuer multicast). Peers are **receive-only**: scratchpad written, barrier signaled,
   their DXA does not participate in the read.
3. **Per-destination completion.** Each receiver's barrier fires when **its** slice lands, not on
   one global completion — the per-receiver `flags.dxa_notify_done → barrier_event_release`
   contract, re-targeted across the fabric to the **receiver's own** CTA barrier slot (from the
   residency table).

### 2.2 Mechanism

The per-SM DXA builds its GMEM read **work list once** (fan-out-independent) and replays only the
**writes** per receiver rank. Each replayed write resolves its destination through the
`CtaResidencyTable` (Task 3): a same-core rank uses the local `LocalMem` port; a remote rank
leaves via the core's DSMEM egress with `dsmem_target` set, crosses the ideal fabric, and lands
in the receiver core's remote SMEM port, whose ingress fires the receiver's own barrier event.
Because the fan-out is source-side replay over the fabric, the single-read property is structural
(there is simply no code path that re-reads GMEM per receiver).

---

## 3. TMA async-copy completion (barrier-based)

A TMA copy is asynchronous: the issuing thread continues, and a consumer must know when the tile
is resident. Express this with the **existing barrier-event machinery** — no new consistency
construct:

- A copy's final write carries `flags.dxa_notify_done` + a target barrier id; delivery fires
  `barrier_event_release` on the destination SM (per receiver under multicast, §2.1(3)).
- The consumer waits on that barrier (`expect_tx` / arrive-and-wait) before reading the tile.
- For multicast, each receiver counts only **its own** completion — the per-destination property
  from §2.1(3).

This is the same drain-style barrier mechanism as Task 3 §4, applied to async copies. It is
explicitly **not** a byte-granular mbarrier transaction model, and requires **no**
`fence.proxy.async` (single-proxy: the DXA write and the consuming read are ordered by the
barrier alone).

---

## 4. TMA tiled / multi-dimensional copies (additive)

The DXA descriptor already carries tile geometry; this item rounds it out to the TMA copy modes
that are not yet covered:

- **2D/3D box copies** — a tensor descriptor (global base, per-dim extents + strides, element
  size) copying a rectangular sub-tile into a (swizzled) scratchpad layout, so a WGMMA can
  consume it directly.
- **Descriptor formats** aligned with the WGMMA operand layout (K-major, already partly present
  for the tensor-core feed).

These extend the existing engine and descriptor path; they are additive and can land after
multicast. (Full descriptor/geometry rationale lives in
[`../../designs/dxa_async_copy_multicast.md`](../../designs/dxa_async_copy_multicast.md).)

---

## 5. C++ kernel verification

`tests/regression/dxa_multicast/` — a per-SM DXA multicasts a GMEM tile into peer CTAs'
scratchpad across cores. Asserts: **(a)** all N receivers' buffers correct; **(b)** each
receiver's barrier fired (per-destination completion); **(c)** — the load-bearing check —
**GMEM read traffic for the N-way multicast is ≈ 1× the tile, not N×**, read from the
`[csr_mpm_dxa]` perf counters. **Byte-correctness alone is not sufficient and must not pass.**

```bash
# faithful TMA multicast — verify the single global read
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_DSMEM_ENABLE" \
  ./ci/blackbox.sh --driver=simx --app=dxa_multicast --clusters=1 --cores=4 --threads=32 --perf=2
# the cross-core multicast Task 2 deferred now passes over the fabric
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_DSMEM_ENABLE" \
  ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_wg_dxa_mcast --cores=4 --threads=32
```

120 s cap applies.

---

## 6. Implementation notes (as-built)

Multicast is implemented over the ideal fabric:

- **Single global read preserved structurally** — the DXA builds one work list and replays only
  writes; `tests/regression/dxa_multicast` measures `gmem_reads` exactly 1× tile and
  `lmem_writes` exactly N× at 4-way fan-out via `VX_CSR_MPM_DXA_GMEM_READS`.
- **Per-receiver completion** re-targets `dxa_notify_bar_id` to the **receiver's own** CTA slot
  (from the residency table — the issuer-relative `bar_id + cta_idx` arithmetic only holds
  same-core) and fires on the receiver core's fabric-ingress callback.
- **Passes:** `dxa_multicast` (bytes + per-receiver completion + the 1×-GMEM assertion),
  `dxa_copy_mcast` (same-core degenerate), `sgemm_tcu_wg_dxa_mcast`.
- **Not yet built:** the async-completion API surface (§3, beyond the internal barrier event
  already used) and the multi-dimensional/tensor copy modes (§4).
