# Task 5 — Proxy Model and Async-Proxy Instruction Support — SimX Proposal

**Status:** proposal (pre-implementation). **Scope:** SimX
([`sim/simx/`](../../sim/simx/)) + the instruction surface in [`sw/`](../../sw/) under the
`sw`/`sim` isolation boundary. Part of the
[Toy-H100 SimX Roadmap](toy_h100_simx_roadmap_proposal.md); this is **Task 5**, inserted after
[Task 3](dsmem_smtosm_simx_proposal.md) (DSMEM / SM-to-SM network) and before
[OpenCL (now Task 6)](opencl_feature_enablement_proposal.md).

**Goal:** model PTX's **two-proxy memory model** — the **generic proxy** (ordinary LSU
load/store/atomic) and the **async proxy** (the TMA / DXA bulk-copy path) — and the instruction
support that orders crossings between them: the producer-side `fence.proxy.async`, the
asynchronous transaction barrier (mbarrier) completion semantics, and the proxy tag on memory
requests. This reproduces Hopper's generic↔async ordering, its async completion model, and the
**failure mode of an omitted fence**.

---

## 1. Scope & lineage

This task is the **deliberate scheduling of the proxy half** of the work the program parked in
[roadmap §10](toy_h100_simx_roadmap_proposal.md) ("faithfully reproducing scoped-consistency
behavior is separate, additive work"). It is a **fidelity layer on top of Tasks 1–3, not a
correction of them.**

It is about **proxies (datapath crossings)**, *not* **scopes (visibility radius)**. The
`::cluster` qualifier on the DSMEM ops stays a **routing/addressing selector** handled by
[Task 3](dsmem_smtosm_simx_proposal.md)'s rank→core map; it does **not** trigger a general
scoped-consistency engine. A full scope-qualifier visibility model remains separate, additive
work (§10).

## 2. Reconciling the prior "single proxy" stance

The roadmap (§10) and [Task 3 §6](dsmem_smtosm_simx_proposal.md) state — correctly — that Vortex
is single-proxy, that `fence.proxy.async` is unnecessary, and that proxy machinery should not be
ported. **That guidance was right for the structural goal and holds through Task 3:** Tasks 1–3
are correct on the single-path model because their correctness rests on **barriers**, not proxy
fences. **Do not rewrite Tasks 1–3 to depend on proxies.**

This task introduces the two-proxy model as an **additive, separately-motivated fidelity layer**.
The roadmap §10 note and Task 3 §6 are edited to say exactly this rather than silently
contradicting the new task. The proxy/scope distinction stays sharp throughout.

## 3. Proxy classification in the model

Today the proxy distinction is **real but unmodeled**: `DxaCore` is functionally the async-proxy
agent and the LSU is the generic-proxy agent, but no request carries a proxy tag and no
crossing-ordering is enforced.

- **Tag home.** Add a `proxy` field (1–2 bits; `generic` / `async`, reserve a value for the
  tensor-map proxy) to `MemFlags` at the free bits **21+**
  ([`sim/simx/types.h:411-447`](../../sim/simx/types.h#L411); bits 0–20 already used by
  `strsp`,`io`,`local`,`amo_unsigned`,`dxa_notify_done@4`,`dxa_notify_bar_id@5-20`).
- **Who sets it.** The LSU path sets `generic` where it already classifies addresses —
  `LocalMemSwitch` ([`mem/local_mem_switch.cpp:75-94`](../../sim/simx/mem/local_mem_switch.cpp#L75)).
  The DXA sets `async` on its LMEM writes
  ([`dxa/dxa_core.cpp:543-555`](../../sim/simx/dxa/dxa_core.cpp#L543)). The decode path that
  builds each request stamps the tag at issue.

## 4. The async-proxy datapath and the port ordering contract

After [Task 2](per_sm_dxa_simx_proposal.md) the core-resident `DxaCore` writes the
[Task 1](socket_removal_simx_proposal.md) core `LocalMem` (SMEM) scratchpad; that establishes it
as the **async-proxy agent**, distinct from the generic LSU path at the **same core SMEM port**.

The port arbiter — already two-issuer after Task 1 (LSU-shared + TCU + DXA) and gaining the
remote client after Task 3 — now carries a **proxy-ordering contract** layered on top of its
existing bandwidth arbitration:

- A `fence.proxy.async` issued by a thread orders that thread's prior **generic** writes ahead
  of subsequent **async** reads of the same scratchpad (see §6).
- An mbarrier completion-wait orders **async** writes ahead of the **generic** reads that follow
  the wait (see §5).
- **Forward progress:** state the no-cyclic-wait argument among the generic, async, and remote
  clients (a core awaiting a remote/async completion must still service inbound requests).

## 5. mbarrier formalization (promotion, not rebuild)

The asynchronous transaction barrier **already half-exists** and is promoted, not rebuilt:
`warp_barrier_t{wait_mask,arrival_mask,count,events,phase}`
([`types.h:90-104`](../../sim/simx/types.h#L90)) already carries dual counters (`count` arrival +
`events` transaction) and a `phase`; `BarrierUnit::event_release` flips `phase` only when
`count==0 && events==0` ([`barrier_unit.cpp:135-167`](../../sim/simx/barrier_unit.cpp#L135));
`expect_tx` → `barrier_event_attach` ([`wctl_unit.cpp:113-142`](../../sim/simx/wctl_unit.cpp#L113));
and DXA multicast already releases **per destination CTA** (`bar_id + cta`,
[`dxa_core.cpp:572-583`](../../sim/simx/dxa/dxa_core.cpp#L572)).

**What this task adds:**

1. **A shared-memory-resident mbarrier object** (directive §4.3). Promote the
   transaction/phase state into an mbarrier modeled in the core SMEM (`LocalMem`, the Task-1
   core-resident scratchpad); keep
   warp-arrival counting in `BarrierUnit`. Document the migration from today's `BarrierUnit`
   `events`/`phase` proto-mechanism and the per-core `tx_callback`
   ([`cluster.cpp:147-170`](../../sim/simx/cluster.cpp#L147)) to the resident object.
2. **Byte-granular `complete_tx`.** Today `events` is an **event count**; Hopper's mbarrier
   tracks **expected transaction *bytes*** (`mbarrier::complete_tx::bytes`). Formalize byte
   accounting: `expect_tx` arms an expected byte count, each async completion increments by the
   bytes delivered, `phase` flips when the expected bytes (and arrivals) are met. **This is the
   primary fidelity gap.**
3. **Ops to model:** `init`, `arrive`, `expect_tx`, `arrive.expect_tx`, `try_wait` (poll → phase
   flip), and the async `complete_tx::bytes` increment — reusing the existing `BAR.*` instruction
   encodings (§7).
4. **Per-destination completion preserved** — each receiver's mbarrier fires when **its** slice
   lands, reconciling [Task 3c](dsmem_smtosm_simx_proposal.md)'s multicast completion with the
   byte-count model.

## 6. `fence.proxy.async`

The generic→async crossing fence, required on the **producer** side when threads fill a buffer
with ordinary stores (generic proxy) and then hand it to the DXA to read (async proxy) — including
the shared→shared push.

- **Enforcement at the core SMEM port:** an async read issued *after* the fence must
  observe the generic-proxy writes that preceded it (even from the same thread).
- **The negative case must be expressible:** without the fence, the model must be **allowed to
  expose stale/reordered data**, so the negative verification kernel (§9a) fails loudly. Do not
  make the model accidentally ordered.
- **Async→generic needs no fence.** The mbarrier completion-wait is the ordering point; model
  this as an **explicit** ordering (a consumer's generic reads after `try_wait` observe the async
  writes), not an accident.

## 7. Instruction surface

Illustrative encodings — finalize against the decode path
([`hw/rtl/core/VX_decode.sv`](../../hw/rtl/core/VX_decode.sv),
[`sim/simx/decode.cpp`](../../sim/simx/decode.cpp), `.insn r` conventions in
[`vx_intrinsics.h`](../../sw/kernel/include/vx_intrinsics.h)). The custom space has clean,
collision-free homes (custom0/funct7=0x03 has funct3 1-7 free; funct7=0x05 is free; custom2/3 are
free). Most ops **already exist** and only gain proxy/byte semantics; the one genuinely new op
owned here is `fence.proxy.async`.

| Op | New? | Recommended encoding | Owner |
|---|---|---|---|
| async bulk copy (`DXA.ISSUE`) | exists | `0x0B / funct7=0x03 / funct3=0` | proxy-tagged `async` here |
| mbarrier `arrive`/`wait`/`expect_tx` | exists | `0x0B / funct7=0 / funct3=6` (`rs2[31]`=expect_tx) | byte-count extended here |
| **`fence.proxy.async`** | **new** | `0x0B / funct7=0x03 / funct3=1` (DXA family) | **this task** |
| `mapa.cluster` | new | `0x0B / funct7=0x05 / funct3=0` | [Task 3b](dsmem_smtosm_simx_proposal.md) |
| `ld/st/atom.cluster` (remote shared) | new | `0x0B / funct7=0x05 / funct3=1..3` | [Task 3b](dsmem_smtosm_simx_proposal.md) |

`sw/` surface: a `vx_fence_proxy_async()` intrinsic in
[`vx_intrinsics.h`](../../sw/kernel/include/vx_intrinsics.h) and the mbarrier byte-count form in
[`vx_barrier.h`](../../sw/kernel/include/vx_barrier.h). Any host↔kernel ABI goes in
[`sw/common/`](../../sw/common/); no `sw/*` → `sim/*` references
([`ci/check_sw_sim_boundary.sh`](../../ci/check_sw_sim_boundary.sh)).

## 8. 1:1 class→module mapping (no façades)

| SimX class (this task) | Future SV module | Level |
|---|---|---|
| `Mbarrier` (shared-resident, with the core SMEM) | `VX_mbarrier` (with the SMEM in `VX_core.sv`) | Core |
| proxy-ordering logic at the core SMEM port | the proxy-order block at the SMEM port in `VX_core.sv` | Core |
| async-proxy datapath tag/handling | `VX_dxa_core` async-proxy port attrs | Core |
| `fence.proxy.async` handling | decode + the port fence hook | Core |

Ownership stays explicit per level; each class is a clean stand-in for one future module.

## 9. Verification kernel specs (specs, not code)

C++ kernels under [`tests/regression/`](../../tests/regression/); built/run in the Docker dev
container, from `build/`, ≤120 s, `CONFIGS` matched on app + `blackbox.sh`.

a. **`proxy_fence/` — generic→async producer.** A correct version that issues
   `fence.proxy.async` before the DXA reads the thread-filled buffer (correct bytes), **and** a
   **negative** version that omits the fence and is **expected to observe stale/reordered data** —
   proving the model enforces the crossing. *Both variants are pass criteria; the negative one
   must fail-without-fence.*
b. **`mbarrier_consume/` — async→generic consumer.** Ordered purely by the mbarrier `try_wait`,
   **no** proxy fence; correct delivery proven through the completion path.
c. **`dsmem_pull_barrier/` — DSMEM LSU pull.** Carried from [Task 3b](dsmem_smtosm_simx_proposal.md),
   expressed in proxy terms (generic→generic, no proxy crossing); asserts the consumer never reads
   stale peer bytes without the cluster-barrier **drain+fence**.
d. **`async_push_mbar/` — shared→shared async push** with **per-destination** mbarrier completion
   under the byte-count model, reconciled with [Task 3c](dsmem_smtosm_simx_proposal.md)'s
   **single-global-read** assertion (N-way multicast ≈ 1× HBM read; each receiver's mbarrier fires
   on its own slice).

```bash
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=proxy_fence --cores=1 --warps=64 --threads=32
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=mbarrier_consume --threads=32
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_CLUSTER_MEM_MODE=DSMEM" ./ci/blackbox.sh --driver=simx --app=dsmem_pull_barrier --cores=2 --threads=32
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_CLUSTER_MEM_MODE=DSMEM" ./ci/blackbox.sh --driver=simx --app=async_push_mbar --cores=4 --threads=32 --perf=2
```

## 10. Scope edges (out of scope)

- **Tensor-map descriptor proxy** — deferred unless the WGMMA descriptor path forces it (reserve
  the `MemFlags.proxy` value).
- **Full scope-qualifier consistency model** — a separate axis from proxies (both are
  consistency-model fidelity, but proxies are datapath crossings, scopes are visibility radius).
  Out of scope here; remains separate, additive work per roadmap §10.
- **Cross-SM atomics across the non-coherent boundary** — the parked watch-item
  ([Task 3 §9](dsmem_smtosm_simx_proposal.md)); the proxy-ordering arbiter design may touch it but
  need not fully resolve it.

## 11. ISA-confirmation flags (epistemic standard)

These rest on PTX 9.x ISA semantics that **cannot be verified from this repo** — carry them as
**still-inferred** until confirmed against the ISA copy, per the program's epistemic standard:

- **TMA push-only / source-never-remote-shared** (a TMA shared-to-shared transfer pushes into a
  peer; the source is never remote shared). Load-bearing for the network direction the async
  path supports.
- **No global→global TMA** (every SM-side TMA op has a shared endpoint; a global→shared transfer
  still reads HBM through L2 — it terminates in shared, it does not bypass L2/HBM).
- **The precise generic→async vs async→generic fence rules** (§6): producer-side
  `fence.proxy.async` required for generic→async; mbarrier wait suffices for async→generic.

Mark each **confirmed** or **still-inferred** in the doc as verification proceeds.
