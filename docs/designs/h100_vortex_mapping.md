# H100 ↔ Vortex Hierarchy Mapping — Design / Gap Analysis

**Scope:** a side-by-side mapping of the NVIDIA **H100 (Hopper)** compute and memory
hierarchies onto the **Vortex** GPGPU, marking every correspondence as **MATCH**,
**DIVERGE**, or **MISSING**, and recording what would have to change in Vortex to close
each gap. The end goal this document serves is a small-scale **"toy H100"** — a Vortex
configuration (and eventual RTL/SimX modifications) whose compute *and* memory hierarchies
are *organized* like a real H100 SM.

> **The 1-to-1 correspondence we pursue is structural and organizational, never
> resource-for-resource.** "Toy H100" means Vortex gains the *architectural and
> organizational properties* of Hopper that it currently lacks — **SM-level SMEM placement**
> (SMEM relocated to the socket, kept *separate* from L1D), per-SM TMA, the SM-to-SM network and
> DSMEM addressing, the two-proxy memory model — **not** the physical dimensions of a datacenter
> GPU (nor the H100's *fused* L1D/SMEM array — a non-goal, §5.2). Vortex is a **soft GPU**;
> matching an H100's 256 KB SMEM, 2048-thread occupancy, ~50 MB L2, or 132 SMs is an
> explicit **non-goal**. Every H100 capacity in this document is a *reference point that
> explains the architecture*, not a size the toy must reproduce. See **§1.1 Non-goals**.

This is a reference / planning document. It proposes no code changes by itself; it is the
spec that drives later architecture work. It pairs with the two v3.0 feature designs that
introduce the relevant compute/DMA blocks:
[`tensor_core_wgmma_engine.md`](tensor_core_wgmma_engine.md) and
[`dxa_async_copy_multicast.md`](dxa_async_copy_multicast.md).

---

## 1. Purpose & scope

The H100 SM and the Vortex core are *almost* the same machine, but the boundaries differ in
three structural ways that matter for a faithful toy. NVIDIA fuses three roles into the **SM**
— it owns the unified L1D/SMEM, owns the L1 instruction cache, and runs a warpgroup `wgmma`
across its four SMSPs. Vortex **splits** those roles across two hierarchy levels: the warpgroup
and scratchpad live in the **core**, while the L1 data/instruction caches live one level up in
the **socket**. Most of the rest lines up cleanly. This document pins down exactly which parts
match, which diverge, and which are simply absent.

The target fidelity is the **organization of a single SM** — its *shape*: four warp-scheduler
partitions (SMSPs), 32-wide SIMT warps, **SM-level L1D + SMEM placement** (both per-SM; the toy
co-locates them at the socket but keeps them *separate* rather than fused — §5.2), per-SM async
copy, tensor cores. The SXM5-full reference points (132 SMs, 8 GPCs, 256 KB SMEM/SM, ~50 MB global L2) appear
throughout this document **only to explain Hopper's architecture** — they are *not* sizes the
toy targets. The toy runs at soft-GPU scale (a handful of cores, modest SRAM); the value is in
the *structure*, not the magnitude.

### 1.1 Non-goals

To be unambiguous about what "toy H100" does and does **not** mean:

- **Not a resource clone.** We do **not** aim to match H100 capacities or counts — not
  256 KB SMEM/SM, not 2048-thread occupancy, not ~50 MB L2, not 132 SMs / 8 GPCs. Those are
  datacenter-GPU dimensions and out of scope for a soft GPU.
- **What we *do* pursue** are Hopper's **architectural and organizational properties** that
  Vortex lacks today: **SMEM placed at the SM** (relocated to the socket, alongside L1D/L1I — but
  kept as a *separate* structure, **not** fused into one SRAM), per-SM TMA placement, the SM-to-SM
  network with DSMEM remote addressing, faithful TMA multicast (single global read), and the
  two-proxy memory ordering model.
- **Capacities stay at soft-GPU scale.** SRAM sizes, warp/CTA counts, and core/cluster counts
  are chosen to keep the toy small and buildable; any H100 number quoted below is an
  *illustrative reference*, and the structural feature it accompanies is correct at **any**
  capacity. Where this doc later writes e.g. `LMEM_LOG_SIZE = 18 (256 KB)`, read it as "this is
  what H100 uses; pick whatever modest size suits the experiment."

The 32-thread warp width and the 4-SMSP partition count are kept because they are *structural*
properties of the SM's organization (SIMT width and scheduler-partition count), not capacity
knobs — but even these may be scaled down if a smaller toy is wanted; nothing in the tasks
below depends on their exact values.

---

## 2. The two hierarchies side-by-side

```
   H100 (Hopper)                         Vortex
   ───────────────                       ───────────────
   GPU                                   Processor   (ProcessorImpl / Vortex.sv)   + L3D, DRAM
    └─ GPC          (×8)                  └─ Cluster  (Cluster / VX_cluster.sv)     + L2D, DXA
        └─ TPC      (×N, 2 SM)                └─ Socket   (Socket / VX_socket.sv)   + L1D + L1I
            └─ SM   (132 total)                   └─ Core (Core / VX_core.sv)       + LMEM, regfile, TCU
                └─ SMSP (×4)                          └─ issue slot (ISSUE_WIDTH)
                    └─ CUDA/Tensor cores                  └─ threads (NUM_THREADS lanes)
```

**The SimX simulator classes map 1-to-1 to the SystemVerilog modules**, level for level:

| Level | SimX class | RTL module | Owns |
|---|---|---|---|
| Processor | `ProcessorImpl` ([`processor_impl.h`](../../sim/simx/processor_impl.h)) | [`Vortex.sv`](../../hw/rtl/Vortex.sv) | **L3D** (optional) + DRAM |
| Cluster | `Cluster` ([`cluster.cpp`](../../sim/simx/cluster.cpp)) | [`VX_cluster.sv`](../../hw/rtl/VX_cluster.sv) | **L2D** + **DXA engine** |
| Socket | `Socket` ([`socket.cpp`](../../sim/simx/socket.cpp)) | [`VX_socket.sv`](../../hw/rtl/VX_socket.sv) | **L1D + L1I** (shared by its cores) |
| Core | `Core` ([`core.cpp`](../../sim/simx/core.cpp)) | [`VX_core.sv`](../../hw/rtl/core/VX_core.sv) | **LMEM**, register file, **TCU/WGMMA** |

Counts: `NUM_SOCKETS = ceil(NUM_CORES / SOCKET_SIZE)`, `cores/socket = SOCKET_SIZE`,
`clusters = NUM_CLUSTERS`. The L1I and L1D are each a `VX_cache_cluster` banked as
`ceil(SOCKET_SIZE/4)` units serving all `SOCKET_SIZE` cores.

---

## 3. Compute mapping

| H100 | Vortex | Verdict | Notes |
|---|---|---|---|
| Warp = 32 threads | `NUM_THREADS` lanes | **MATCH** | set `NUM_THREADS = 32` |
| SMSP (4 per SM) — warp scheduler + dispatch | **issue slot** (`ISSUE_WIDTH`) | **MATCH** | `ISSUE_WIDTH = up(NUM_WARPS/16)` ([`VX_config.toml:71`](../../VX_config.toml#L71)) — derived, not directly set |
| 16 warps resident / SMSP | `NUM_WARPS / ISSUE_WIDTH` | **MATCH** | `NUM_WARPS=64` ⇒ 16 warps × 4 slots |
| 2048 threads / SM max occupancy | `NUM_WARPS × NUM_THREADS` | **MATCH (structure)** | the *mechanism* matches (occupancy = warps × lanes); the **2048 magnitude is H100 reference, not a toy target** (§1.1) — `64 × 32 = 2048` only if you choose those values |
| Tensor core (one per SMSP, 4 per SM) | per-block FEDP datapath, `NUM_TCU_BLOCKS = ISSUE_WIDTH` | **MATCH** | [`VX_config.toml:240`](../../VX_config.toml#L240); Q blocks = Q tensor datapaths |
| `wgmma` = warpgroup MMA spanning **the SM's 4 SMSPs** | TCU `wgmma` = `Q = ISSUE_WIDTH` lock-stepped blocks **within one core** | **MATCH (as SM↔core)** | TCU is instantiated **per core** ([`VX_execute.sv:121-122`](../../hw/rtl/core/VX_execute.sv#L121)); warpgroup = the core's issue slots, *not* multiple cores — see §5 |
| Single-CTA ownership of SM shared tiles during a warpgroup | `VX_tcu_lockstep` single-owner CTA gate | **MATCH** | [`tensor_core_wgmma_engine.md §4`](tensor_core_wgmma_engine.md) |
| GPC groups SMs | Cluster groups sockets/cores | **APPROX** | structural analog; cache placement differs (§4) |
| TPC (2 SM) | *(no direct tier)* | **MISSING** | not modeled; irrelevant at toy scale |

**Headline result for compute: `H100 SM ↔ Vortex core`, `H100 SMSP ↔ Vortex issue slot`.**
The warpgroup is contained entirely inside one core. This correspondence is about *organization*
(the SM = a core with 4 scheduler partitions running 32-wide warps), not capacity: with
`NUM_WARPS=64` / `NUM_THREADS=32` the per-SM occupancy *happens* to reproduce H100's 2048
threads / 16 warps-per-SMSP, but those magnitudes are an illustrative reference — a smaller
`NUM_WARPS` gives a smaller, equally-valid toy SM (§1.1). The structural invariant is the
4-SMSP shape, not the thread count.

---

## 4. Memory mapping

| H100 | Vortex | Verdict | Notes |
|---|---|---|---|
| Register file (private per SMSP) | per-issue-slot register file | **MATCH** | |
| **L1D / SMEM** — unified per-SM SRAM (256 KB on H100), runtime-split cache↔scratchpad, shared by 4 SMSPs | **two separate structures, split across levels:** LMEM scratchpad (per **core**) + L1D cache (per **socket**) | **DIVERGE** | the toy closes the *placement* gap — relocate SMEM up to the socket so both sit per-SM (§5.1) — but **keeps them as two separate structures**; the H100's single *fused* SRAM is a non-goal (§5.2) |
| **L1 instruction cache** (per SM) | **L1I** (per **socket**, shared across cores) | **DIVERGE** | `VX_cache_cluster`, `NUM_INPUTS = SOCKET_SIZE` ([`VX_socket.sv:128-157`](../../hw/rtl/VX_socket.sv#L128)); icache prioritized at the L1 arbiter ([`VX_socket.sv:225-238`](../../hw/rtl/VX_socket.sv#L225)) |
| **L0 instruction cache** (per SMSP, feeds the local warp scheduler) | *(none)* | **MISSING** | Vortex has no per-issue-slot or per-core I-cache at all |
| **DSMEM** — addressable shared memory across SMs in a Thread-Block Cluster | *(no cross-core SMEM addressing)* | **MISSING** | only *approximated* by DXA multicast into co-resident CTAs' LMEM ([`dxa doc §7.6`](dxa_async_copy_multicast.md)) |
| **L2** — single **global** cache across all GPCs (~50 MB) | **L2 per cluster** (+ optional L3 at processor) | **DIVERGE** | Vortex L2 is per-cluster, finer than H100's global L2 |
| HBM3 | DRAM (`memsim_`) | **MATCH** | |
| **TMA** — async tiled GMEM→SMEM copy, per **SM** | **DXA** — async tiled GMEM→LMEM copy, **cluster-shared** | **MATCH (function) / DIVERGE (placement)** | semantics identical incl. barrier-tx + K-major tensor feed; placement differs — see §5.5 |

**Headline result for memory: the data *path* matches (regfile → SMEM/LMEM → L2 → DRAM), but
the data *placement* diverges** — Vortex puts the L1 caches at the socket and the scratchpad at
the core, where H100 fuses both into one per-SM block; and Vortex has no L0I and no DSMEM.

---

## 5. Divergences and how to close each

Ordered easiest → hardest. Items 1 and 6 are pure config; 5 is a planned-but-unbuilt RTL move;
2, 3, 4 are genuine RTL/SimX additions.

### 5.1 L1D/L1I/SMEM placement (per-SM) — **config + SimX relocation**
Fix `SOCKET_SIZE = 1` **and remove it as a knob**. The socket then contains exactly one core, so
its L1D and L1I become effectively **per-SM**, matching H100 — no RTL change needed for the
caches; the socket tier collapses onto the core for cache purposes. To place **SMEM** at the SM
as well, relocate the per-core `LMEM` scratchpad **up to the socket** so it sits alongside L1D and
L1I (Task 1 — [`../proposals/socket_removal_simx_proposal.md`](../proposals/socket_removal_simx_proposal.md)).
The three remain **separate** structures; this is a placement fix, **not** the fused-SRAM change
in §5.2.

### 5.2 Unified L1D/SMEM SRAM — **explicit non-goal (accepted split)**
The H100 fuses L1D and SMEM into **one** SRAM with a **configurable cache↔scratchpad split**.
**This program does not reproduce that fusion.** Vortex keeps **LMEM** (the scratchpad,
[`core.cpp`](../../sim/simx/core.cpp), size `2^LMEM_LOG_SIZE`) and the **L1D** (socket cache) as
**physically distinct** structures — §5.1 merely co-locates them at the socket, it does not merge
them. This is a deliberate soft-GPU simplification.
- **Capacity is a free parameter, kept at soft-GPU scale.** `LMEM_LOG_SIZE` sets the
  scratchpad size ([`VX_config.toml:229`](../../VX_config.toml#L229), default 14 = 16 KB).
  H100 uses 256 KB (`LMEM_LOG_SIZE = 18`) — quoted only as the reference; the toy should pick a
  modest size. Matching 256 KB is **not** a goal (§1.1).
- **The split is the accepted design** (LMEM = SMEM, L1D = the separate cache). Task 1
  ([`../proposals/socket_removal_simx_proposal.md`](../proposals/socket_removal_simx_proposal.md))
  implements the *placement* (SMEM → socket) **without** fusing. **True unification** — a shared
  banked SRAM with a configurable partition — needs RTL/SimX surgery and is recorded as an
  out-of-scope **follow-on** (§7), not part of the toy-H100 program.

### 5.3 L0 instruction cache (per SMSP) — **new RTL/SimX, or accept absence**
Vortex has no per-issue-slot or per-core I-cache. Adding an L0I in front of each issue slot,
backed by the socket L1I, would mirror Hopper's split fetch path. Instruction caching is the
least semantically critical structure for a functional toy; **recommend deferring** unless the
study specifically targets the fetch pipeline.

### 5.4 DSMEM / SM-to-SM network — **largest RTL item**
H100 lets SMs in a Thread-Block Cluster directly address each other's shared memory over an
SM-to-SM network. Vortex has no cross-core LMEM addressing; DXA **multicast** only *distributes*
data into co-resident CTAs' LMEM, it does not let a thread load a peer core's LMEM. Closing this
means a cross-core LMEM interconnect + address remap (issuer base + `r·smem_stride` style
addressing already exists for multicast and is a useful starting point). This aligns with the
explicitly unbuilt **"cross-core DSMEM path"** in [`dxa doc §7.6`](dxa_async_copy_multicast.md).

### 5.5 TMA placement (cluster-shared → per-SM) — **planned RTL move**
DXA is one `VX_dxa_core` per **cluster** ([`VX_cluster.sv:239`](../../hw/rtl/VX_cluster.sv#L239))
fanning out to `NUM_DXA_UNITS = max(1, ceil(NUM_CORES/8))` workers
([`VX_config.toml:45`](../../VX_config.toml#L45)). H100's TMA is per-SM. The DXA design already
specifies **"per-socket DXA relocation"** as its single biggest unbuilt item
([`dxa doc §7.1`](dxa_async_copy_multicast.md)): move `VX_dxa_core` into `VX_socket.sv` with a
dcache>DXA arbiter, plus a Socket-owned `DxaCore` in SimX. With `SOCKET_SIZE=1` that yields a
**per-SM** TMA. Until then, a single-core cluster already gives one DXA engine per SM in practice.

### 5.6 Global L2 — **config**
H100's L2 is global; GPCs have no private cache. To match: **disable the per-cluster L2 and use
the processor-level L3 as the single global L2** — `L2_ENABLE=false`, `L3_ENABLE=true` — treating
the cluster as a cache-less GPC tier.

---

## 6. Recommended "toy H100 SM" configuration

A minimal SM that reproduces Hopper's **organization** at **soft-GPU scale**. The values below
are a starting point, not requirements: the `*_SIZE` / `NUM_*` capacities are deliberately
small and may be tuned freely (§1.1). Only the *structural* choices (`SOCKET_SIZE=1` so the
socket is the SM; L2 off / L3 as global cache; TCU/DXA enabled) are load-bearing for the
architecture.

| Param | Value | Models | Kind |
|---|---|---|---|
| `VX_CFG_NUM_THREADS` | 32 | warp = 32 threads | structural (SIMT width) |
| `VX_CFG_NUM_WARPS` | 64 | ⇒ `ISSUE_WIDTH = up(64/16) = 4` SMSPs; 16 warps/SMSP | structural part (4 SMSPs); the 2048-thread occupancy it implies is an H100 *reference*, not a target — fewer warps are fine |
| `VX_CFG_NUM_TCU_BLOCKS` | = `ISSUE_WIDTH` = 4 | one tensor core per SMSP (auto-derived) | structural |
| `VX_CFG_SOCKET_SIZE` | 1 (fixed by Task 1) | socket == SM ⇒ L1D + L1I per-SM; SMEM relocated alongside (§5.1) | structural |
| `VX_CFG_LMEM_LOG_SIZE` | modest (e.g. 14–16) | SMEM/scratchpad capacity; **H100 = 18 (256 KB), quoted as reference only** (§5.2) | capacity — pick small |
| `VX_CFG_NUM_CORES` | small (e.g. 2–8) | SMs; H100 SXM5 = 132 is **not** the target | capacity — pick small |
| `VX_CFG_NUM_CLUSTERS` | 1–2 | GPCs; H100 = 8 is **not** the target | capacity — pick small |
| `VX_CFG_L2_ENABLE` | false | no GPC-level cache in H100 (§5.6) | structural |
| `VX_CFG_L3_ENABLE` | true | acts as the single global L2 (§5.6) | structural |
| `VX_CFG_EXT_TCU_ENABLE` | true | tensor cores | structural |
| `VX_CFG_TCU_WGMMA_ENABLE` | true | warpgroup MMA ([`VX_config.toml:246`](../../VX_config.toml#L246)) | structural |
| `VX_CFG_EXT_DXA_ENABLE` | true | TMA-style async copy | structural |

`ISSUE_WIDTH` and `NUM_TCU_BLOCKS` are **derived** from `NUM_WARPS`; you cannot set them
directly — pick `NUM_WARPS = 64` to obtain 4 of each (or a smaller `NUM_WARPS` for fewer SMSPs).
Capacity knobs (`LMEM_LOG_SIZE`, `NUM_CORES`, `NUM_CLUSTERS`, cache sizes) should stay at
soft-GPU scale; H100 figures are reference points, never goals.

---

## 7. Proposed but not yet implemented (toward the true toy H100)

The gaps in §5 that require source changes, in rough effort order. Several already exist as
recorded intentions in the v3.0 design docs, which makes them the natural next steps:

1. **Per-socket (per-SM) DXA relocation** — §5.5; specified as the top unbuilt item in
   [`dxa doc §7.1`](dxa_async_copy_multicast.md).
2. **Cross-core DSMEM path + SM-to-SM network** — §5.4; the Hopper DSMEM analog, noted as an
   open multicast question in [`dxa doc §7.6`](dxa_async_copy_multicast.md).
3. **Unified L1D/SMEM SRAM with configurable split** — §5.2; no precedent in the codebase. **A
   deliberate non-goal of the toy-H100 program** (which accepts the split and only *relocates*
   SMEM to the socket, §5.1) — recorded here as a longer-term follow-on.
4. **Per-SMSP L0 instruction cache** — §5.3; lowest priority for a functional toy.

The cache/global-L2 placement (§5.1 `SOCKET_SIZE = 1`, §5.6 L2-off / L3-as-global) is achievable
**today** purely through `VX_config.toml`; the **SMEM relocation** added to §5.1 is Task 1's SimX
change (not config-only). The recommended config in §6 applies the config-level items.

---

## 8. Sources

- SimX hierarchy: [`processor_impl.h`](../../sim/simx/processor_impl.h),
  [`cluster.cpp`](../../sim/simx/cluster.cpp), [`socket.cpp`](../../sim/simx/socket.cpp),
  [`core.cpp`](../../sim/simx/core.cpp).
- RTL hierarchy: [`Vortex.sv`](../../hw/rtl/Vortex.sv), [`VX_cluster.sv`](../../hw/rtl/VX_cluster.sv),
  [`VX_socket.sv`](../../hw/rtl/VX_socket.sv), [`VX_core.sv`](../../hw/rtl/core/VX_core.sv),
  [`VX_execute.sv`](../../hw/rtl/core/VX_execute.sv).
- v3.0 features: [`tensor_core_wgmma_engine.md`](tensor_core_wgmma_engine.md),
  [`dxa_async_copy_multicast.md`](dxa_async_copy_multicast.md),
  [`hw/rtl/tcu/`](../../hw/rtl/tcu), [`hw/rtl/dxa/`](../../hw/rtl/dxa).
- Config: [`VX_config.toml`](../../VX_config.toml).
