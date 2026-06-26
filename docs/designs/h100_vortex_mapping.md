# H100 ↔ Vortex Hierarchy Mapping — Design / Gap Analysis

**Scope:** a side-by-side mapping of the NVIDIA **H100 (Hopper)** compute and memory
hierarchies onto the **Vortex** GPGPU, marking every correspondence as **MATCH**,
**DIVERGE**, or **MISSING**, and recording what would have to change in Vortex to close
each gap. The end goal this document serves is a small-scale **"toy H100"** — a Vortex
configuration (and eventual RTL/SimX modifications) whose compute *and* memory hierarchies
map 1-to-1 to a real H100 SM.

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

The target fidelity is a **single faithful SM** (4 SMSPs, 32-thread warps, 2048-thread
occupancy, tensor cores, async copy), scaled out by core/cluster counts. SXM5-full reference
points (132 SMs, 8 GPCs, 256 KB SMEM/SM, ~50 MB global L2) are noted as the scale-up target.

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
| 2048 threads / SM max occupancy | `NUM_WARPS × NUM_THREADS` | **MATCH** | 64 × 32 = 2048 |
| Tensor core (one per SMSP, 4 per SM) | per-block FEDP datapath, `NUM_TCU_BLOCKS = ISSUE_WIDTH` | **MATCH** | [`VX_config.toml:240`](../../VX_config.toml#L240); Q blocks = Q tensor datapaths |
| `wgmma` = warpgroup MMA spanning **the SM's 4 SMSPs** | TCU `wgmma` = `Q = ISSUE_WIDTH` lock-stepped blocks **within one core** | **MATCH (as SM↔core)** | TCU is instantiated **per core** ([`VX_execute.sv:121-122`](../../hw/rtl/core/VX_execute.sv#L121)); warpgroup = the core's issue slots, *not* multiple cores — see §5 |
| Single-CTA ownership of SM shared tiles during a warpgroup | `VX_tcu_lockstep` single-owner CTA gate | **MATCH** | [`tensor_core_wgmma_engine.md §4`](tensor_core_wgmma_engine.md) |
| GPC groups SMs | Cluster groups sockets/cores | **APPROX** | structural analog; cache placement differs (§4) |
| TPC (2 SM) | *(no direct tier)* | **MISSING** | not modeled; irrelevant at toy scale |

**Headline result for compute: `H100 SM ↔ Vortex core`, `H100 SMSP ↔ Vortex issue slot`.**
The warpgroup is contained entirely inside one core. With `NUM_WARPS=64` / `NUM_THREADS=32`
the per-SM occupancy numbers fall out *exactly* (2048 threads, 16 warps/SMSP, 4 SMSPs).

---

## 4. Memory mapping

| H100 | Vortex | Verdict | Notes |
|---|---|---|---|
| Register file (private per SMSP) | per-issue-slot register file | **MATCH** | |
| **L1D / SMEM** — unified 256 KB per SM, runtime-split cache↔scratchpad, shared by 4 SMSPs | **two separate structures:** LMEM scratchpad (per **core**) + L1D cache (per **socket**) | **DIVERGE** | capacity matchable (LMEM=256 KB) but unified-SRAM behavior is not — see §5.2 |
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

### 5.1 L1D/L1I placement (per-socket → per-SM) — **config**
Set `SOCKET_SIZE = 1`. The socket then contains exactly one core, so its L1D and L1I become
effectively **per-SM**, matching H100. No RTL change needed; the socket tier collapses onto the
core for cache purposes.

### 5.2 Unified L1D/SMEM SRAM — **RTL (or accept as toy simplification)**
H100's 256 KB block is one SRAM whose split between L1 cache and shared memory is chosen per
kernel. Vortex keeps **LMEM** (per-core scratchpad, [`core.cpp`](../../sim/simx/core.cpp),
size `2^LMEM_LOG_SIZE`) and the **L1D** (socket cache) as physically distinct structures.
- **Config gets capacity right:** `LMEM_LOG_SIZE = 18` → 256 KB LMEM
  ([`VX_config.toml:229`](../../VX_config.toml#L229), default 14 = 16 KB).
- **True unification needs RTL surgery:** a shared banked SRAM with a configurable
  cache/scratchpad partition. Recommend **accepting the split for the toy** (LMEM = SMEM,
  L1D = the cache half) and recording unification as a follow-on.

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

A minimal but faithful single SM, scaled out by `NUM_CORES` / `NUM_CLUSTERS`:

| Param | Value | Models |
|---|---|---|
| `VX_CFG_NUM_THREADS` | 32 | warp = 32 threads |
| `VX_CFG_NUM_WARPS` | 64 | ⇒ `ISSUE_WIDTH = up(64/16) = 4` SMSPs; 2048 threads/SM; 16 warps/SMSP |
| `VX_CFG_NUM_TCU_BLOCKS` | = `ISSUE_WIDTH` = 4 | one tensor core per SMSP (auto-derived) |
| `VX_CFG_SOCKET_SIZE` | 1 | socket == SM ⇒ L1D + L1I become per-SM (§5.1) |
| `VX_CFG_LMEM_LOG_SIZE` | 18 (256 KB) | H100 SMEM capacity (§5.2) |
| `VX_CFG_NUM_CORES` | 2–8 (toy) … 132 (SXM5) | SMs |
| `VX_CFG_NUM_CLUSTERS` | 1–2 (toy) … 8 (SXM5) | GPCs |
| `VX_CFG_L2_ENABLE` | false | no GPC-level cache in H100 (§5.6) |
| `VX_CFG_L3_ENABLE` | true | acts as the single global L2 (§5.6) |
| `VX_CFG_EXT_TCU_ENABLE` | true | tensor cores |
| `VX_CFG_TCU_WGMMA_ENABLE` | true | warpgroup MMA ([`VX_config.toml:246`](../../VX_config.toml#L246)) |
| `VX_CFG_EXT_DXA_ENABLE` | true | TMA-style async copy |

`ISSUE_WIDTH` and `NUM_TCU_BLOCKS` are **derived** from `NUM_WARPS`; you cannot set them
directly — pick `NUM_WARPS = 64` to obtain 4 of each.

---

## 7. Proposed but not yet implemented (toward the true toy H100)

The gaps in §5 that require source changes, in rough effort order. Several already exist as
recorded intentions in the v3.0 design docs, which makes them the natural next steps:

1. **Per-socket (per-SM) DXA relocation** — §5.5; specified as the top unbuilt item in
   [`dxa doc §7.1`](dxa_async_copy_multicast.md).
2. **Cross-core DSMEM path + SM-to-SM network** — §5.4; the Hopper DSMEM analog, noted as an
   open multicast question in [`dxa doc §7.6`](dxa_async_copy_multicast.md).
3. **Unified L1D/SMEM SRAM with configurable split** — §5.2; no precedent in the codebase.
4. **Per-SMSP L0 instruction cache** — §5.3; lowest priority for a functional toy.

Items 1 and 6 of §5 (cache placement and global-L2) are achievable **today** purely through
`VX_config.toml`; the recommended config in §6 already applies them.

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
