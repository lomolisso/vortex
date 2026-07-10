# DSMEM & TMA for Vortex SimX — Roadmap

**Status:** proposal / in progress. **Scope:** SimX cycle-level simulator only
([`sim/simx/`](../../../sim/simx/)) — **no RTL/SystemVerilog in this program**. This folder is
the umbrella roadmap; each task has its own file (`<n>__<task>.md`, linked in §4) and is
implemented and verified in sequence.

This program adds **DSMEM** (distributed shared memory — a thread on one SM directly addressing
another SM's scratchpad) and **TMA** features (a per-SM bulk async-copy engine, including
**multicast**) to Vortex. These are the Hopper/H100 shared-memory and async-copy capabilities;
we reproduce their **microarchitecture** in SimX at soft-GPU scale. The design reference is the
gap analysis in [`../../designs/h100_vortex_mapping.md`](../../designs/h100_vortex_mapping.md)
(H100 SM ↔ Vortex core), kept as background material — this roadmap is about *building the
features*, not about matching any H100 magnitude.

> **Architecture, not resources.** Everything here reproduces the *shape* of the machine — how
> its blocks are placed and how they talk — **never** H100 capacities or counts. Vortex is a
> soft GPU: capacity knobs (`LMEM_LOG_SIZE`, cache sizes, `NUM_CORES`, `NUM_CLUSTERS`,
> `NUM_WARPS`) are free parameters; pick small values. Any H100 figure quoted below is an
> illustrative reference, not a target.

---

## 1. What we are building

Three microarchitectural additions plus their consumers:

1. **Core is the SM.** Vortex currently splits an SM's SRAM arrays across a redundant `Socket`
   level (L1I/L1D at the socket, scratchpad at the core). Remove the socket level and push its
   caches into the core, so each `Core` is a self-contained SM owning L1I + L1D + SMEM as
   separate arrays. (→ **Task 1**)
2. **Per-SM TMA.** Vortex's DXA async-copy engine is cluster-shared; a real TMA is per-SM. Move
   `DxaCore` down to the core. (→ **Task 2**)
3. **DSMEM.** Cross-SM scratchpad addressing does not exist in Vortex. Add the full
   microarchitecture — remote addressing, the cluster-residency table, the cross-SM barrier —
   over an **idealized** SM-to-SM fabric. (→ **Task 3**)
4. **TMA features.** With per-SM TMA and DSMEM in place, add the interesting TMA capabilities,
   headlined by **multicast** (one issuer, one global read, hardware fan-out into N
   scratchpads). (→ **Task 4**)
5. **Real NoC.** Replace the idealized fabric with a real network model via co-simulation with
   the **Omelet** gem5/Garnet chiplet-interconnect simulator. (→ **Task 5**)

## 2. Non-negotiable principles (apply to every task)

- **SimX only.** All work lands under [`sim/simx/`](../../../sim/simx/) (Task 4 also touches the
  `sw/` instruction surface; Task 5 also touches the Omelet checkout + gem5 glue in
  [`sim/simx/gem5/`](../../../sim/simx/gem5/)). No RTL — where a SimX change desynchronizes the
  model from the current RTL (e.g. Task 1 removing the socket level and its knobs), **breaking
  the RTL build is acceptable**; the RTL is realigned later as an out-of-scope companion change.
- **Preserve the 1:1 class↔module mapping.** Each SimX C++ class is a stand-in for exactly one
  future SystemVerilog module, shaped so the eventual RTL port is mechanical. Per-level
  ownership stays explicit. See §6.
- **Genuine reworks, no façades.** When a task says *remove* a class, delete it and replace it —
  do not wrap the old structure. Do not fake behavior (unmodeled partitions, unarbitrated
  routing). Reworks over workarounds.
- **Soft-GPU scale — reproduce architecture, never resources** (see the callout above).
- **DSMEM is scratchpad — no coherence, no consistency model.** DSMEM and TMA operate on
  **non-coherent scratchpad**, synchronized by **barriers**. There is no coherence domain to
  flush and no scoped-visibility choice to make, so **no coherence protocol and no
  memory-consistency model are in scope** anywhere in this roadmap. The cross-SM barrier's
  drain-and-fence (Task 3) is a *barrier-implementation* property, not a consistency feature.
  DSMEM is **additive** to the global memory hierarchy — it coexists with the L2/global path and
  does **not** replace it.
- **Root-cause discipline** per [`../../../AGENTS.md`](../../../AGENTS.md) §3.

## 3. Target hierarchy (end state)

| Level | SimX class | Owns — **before** | Owns — **after** |
|---|---|---|---|
| Cluster | `Cluster` | L2, **DxaCore**; creates sockets | L2 (global path, unchanged) + the **SM-to-SM fabric** (idealized in Tasks 3–4, Omelet-modeled in Task 5); no DXA; **creates cores directly** |
| ~~Socket~~ | ~~`Socket`~~ | ~~L1I + dcache (`CacheCluster`); creates cores~~ | **removed (Task 1)** |
| Core (== SM) | `Core` | regfile + TCU + **`LocalMem`** | regfile + TCU + `LocalMem` (SMEM) + **L1I + dcache (L1D)** — three separate SRAM arrays + **L1 arbiter** + **`DxaCore`** + the **DSMEM remote port** |

The `Socket` level is **removed** and its knobs deleted (Task 1), so **the core is the SM**
throughout. L1D and SMEM stay separate structures — the H100's *fused* L1D/SMEM array is a
non-goal (a recorded follow-on, gap analysis §5.2).

## 4. Task sequence (each builds on the previous)

1. **[Socket removal](1__socket_removal.md)** — remove the `Socket` level entirely (delete
   `SOCKET_SIZE`/`NUM_SOCKETS` and every socket knob); push its L1I + dcache (L1D) + L1 arbiter
   down into the **core**, alongside the already core-resident `LocalMem` (SMEM), kept as
   separate structures. `Cluster` creates cores directly. *Foundation — the core is now the SM.*
2. **[Per-core DXA](2__per_core_dxa.md)** — move `DxaCore` from cluster to the **core**; add a
   dcache→DXA arbiter on the core's L1→L2 path and target the core `LocalMem` (SMEM) on the
   write side. One `DxaCore` per core is a true per-SM TMA. *Requires Task 1.*
3. **[DSMEM microarchitecture](3__dsmem_microarchitecture.md)** — every DSMEM microarch piece
   **except** the real SM-to-SM network, which is **idealized** (`IdealTransport`, a fixed-delay
   fabric): remote shared-memory addressing (`mapa/ld/st.cluster`), the **CTA residency table**
   with gang placement, `cluster_id`/`rank` plumbing, the `LocalMem` remote port + egress, and
   the **cross-SM barrier with remote-store drain**. Additive to the global path; DSMEM is
   gated by its own enable, it does not replace L2. *Uses Task 1's core SMEM.*
4. **[TMA multicast + TMA features](4__tma_multicast.md)** — the interesting TMA capabilities on
   the idealized fabric + Task 1–3 machinery: **multicast** (single issuer, *one* global read,
   hardware fan-out into N scratchpads, per-receiver completion), TMA **async-copy completion**,
   and TMA **tiled/multi-dimensional copy modes**. *Uses Task 2's per-core DXA and Task 3's
   DSMEM.*
5. **[Omelet co-simulation](5__omelet_cosimulation.md)** — replace the idealized SM-to-SM fabric
   with a real router-level NoC by co-simulating with **Omelet** (gem5/Garnet). SimX generates
   the traffic; Omelet models the network; topology becomes a `VX_CFG_NOC_*` knob. Single 2D die
   now; the same interface carries a future 2.5D/3D chiplet partition. *Uses Task 3's endpoints.*

## 5. Complexity assessment

Honest difficulty of each task (building it correctly from scratch), with current status.

| Task | Complexity | Why | Status |
|---|---|---|---|
| 1 Socket removal | **Medium** | delete a hierarchy level + rewire cluster→core + drop socket knobs; broad but a refactor, no new algorithms | implemented |
| 2 Per-core DXA | **Medium** | relocate one engine + add a dcache→DXA arbiter + retarget writes; relocation/rewiring | implemented |
| 3 DSMEM microarch | **High** | new aperture/region encoding, three new instructions through decode/ALU/LSU, residency table, gang placement, `cluster_id`/`rank` plumbing, remote port + egress, cross-SM barrier + drain — many new SimObjects and cross-cutting plumbing (the ideal fabric keeps it bounded) | implemented |
| 4 TMA multicast + features | **Medium-High** | fan-out with single-global-read + per-receiver completion on the Task-3 machinery; completion and tiled modes are additive | implemented |
| 5 Omelet co-sim | **Very High** | embed SimX in gem5 v22.1 (VortexGPGPU v25→v22.1 port), a C-ABI token protocol, a Garnet NetworkInterface bridge, timing-only tokens + backpressure, a knob→yaml generator; the least-bounded, cross-simulator work | not started |

## 6. The 1:1 mapping discipline (intended future SV modules)

Each new/changed SimX class names the single RTL module it will become:

| SimX class (this program) | Future SV module | Level |
|---|---|---|
| `Socket` **removed**; L1I + dcache + L1 arbiter pushed into `Core` | `VX_socket.sv` **retired**; `VX_cache_cluster` + L1 arbiter move into `VX_core.sv` | Core |
| `LocalMem` (SMEM) + `LocalMemSwitch`, core-resident | `VX_local_mem` + `VX_lmem_switch` in `VX_core.sv` | Core |
| `DxaCore` relocated to `Core` | `VX_dxa_core` moved into `VX_core.sv` | Core |
| `CtaResidencyTable` (`sim/simx/dsmem/`) | `VX_cta_table` (in `VX_cluster.sv`) | Cluster |
| `DsmemLsuAdapter` + `LocalMem` remote port | `VX_dsmem_lsu_adapter` + remote port in `VX_core.sv` | Core |

**Modeling-infrastructure exemptions (outside the 1:1 discipline):** `IdealTransport`
(`sim/simx/dsmem/`) is a fixed-delay connectivity placeholder, not a future SV module — the
network it stands in for is *modeled*, not synthesized, by Omelet (Task 5). Likewise the Task-5
co-sim glue (`OmeletNocPort`, the gem5-side bridge SimObjects) is simulation infrastructure with
no RTL counterpart. This is why façades are banned elsewhere: a wrapper has no clean SV analog.

## 7. Verification harness conventions (shared by all tasks)

Build + run a single test on SimX (from `build/`):

```bash
# C++ regression kernel
make -C tests/regression/<app> clean
CONFIGS="-DVX_CFG_NUM_THREADS=32 -D<feature flags>" make -C tests/regression/<app>
CONFIGS="-D<feature flags>" ./ci/blackbox.sh --driver=simx --app=<app> --cores=<N> --warps=64 --threads=32
```

- **Re-read [`AGENTS.md`](../../../AGENTS.md) at the start of each task** and follow it throughout.
- **Builds and passes inside the Docker dev container.** Use
  [`miscs/docker/vortex-dev.sh`](../../../miscs/docker/vortex-dev.sh) (`build` → `up` → `shell`),
  run from the generated `build/` tree. The same container hosts Omelet for Task 5 (mounted at
  `/omelet`; one-time gem5 build via `miscs/docker/omelet-setup.sh`).
- **A C++ verification kernel proves each feature.** A feature with no passing kernel is
  incomplete. Kernels live under [`tests/regression/`](../../../tests/regression/) as
  `{kernel.*, main.*, common.h, Makefile}`.
- **120-second timeout cap** on every test invocation.
- **`CONFIGS` must match on both sides** — rebuild the app with the same `-DVX_CFG_*` before
  running `blackbox.sh`.
- **sw/sim isolation** ([`ci/check_sw_sim_boundary.sh`](../../../ci/check_sw_sim_boundary.sh)):
  `sw/{kernel,runtime}` must not reference `sim/*` or `hw/*`, and vice-versa.
- **Re-`configure`** from `build/` after editing TOML/Makefiles or adding files.

Common knobs are exposed as `blackbox.sh` flags (`--clusters --cores --warps --threads
--l2cache --l3cache --debug --perf`); everything else goes through `CONFIGS="-DVX_CFG_*"`.
Baseline parameters live in [`VX_config.toml`](../../../VX_config.toml) /
[`VX_types.toml`](../../../VX_types.toml).

## 8. Per-task deliverable flow

For each task, in order: (1) re-read `AGENTS.md`; (2) update that task's file with the concrete
SimX plan; (3) implement the SimX classes; (4) write the C++ verification kernel(s); (5) build
and run in Docker; (6) only then start the next task. When a task is implemented and merged,
promote its content into [`../../designs/`](../../designs/) per
[`../../../AGENTS.md`](../../../AGENTS.md) §7.
