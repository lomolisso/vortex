# Task 1 — Socket-Level Removal (Core-as-SM Consolidation) — SimX Proposal

**Status:** implemented in SimX (the socket level is gone from SimX; core == SM). **Scope:** SimX only
([`sim/simx/`](../../../sim/simx/)). Part of the
[DSMEM & TMA Roadmap](README.md); this is **Task 1**, the foundation
for Tasks 2–5. Closes the **"L1D/SMEM placement"** divergence from
[`../../designs/h100_vortex_mapping.md`](../../designs/h100_vortex_mapping.md) §5.1–5.2 by making the
**core the SM** rather than the socket.

> **RTL may break — that is acceptable for this task.** The work is SimX-only. Removing the
> socket config knobs and the `Socket` SimObject will desynchronize the SimX model from the
> current `hw/rtl/VX_socket.sv`; the RTL build is **expected to break** and is fixed by a later,
> out-of-scope companion change (§3.6). Do not spend effort keeping the RTL compiling.

**Goal:** **remove the `Socket` level from the compute hierarchy entirely**, collapsing
`Processor → Cluster → Socket → Core` to **`Processor → Cluster → Core`**. Everything the socket
owned — the L1I cache, the L1D (dcache) `CacheCluster`, and the L1 icache/dcache memory arbiter
(the on-chip "NoC" merge toward L2) — moves **down into the `Core`**. The `Cluster` creates cores
directly. Every **socket config knob is deleted**. The end state: each `Core` is a **self-contained
streaming multiprocessor** owning its own SRAM arrays — L1I, L1D, and SMEM (`LocalMem`, already
core-resident) — plus its regfile and TCU, and (after [Task 2](2__per_core_dxa.md)) its
own `DxaCore`. **Core == SM.**

> **What this task does *not* do — the fused L1D/SMEM is out of scope.** The H100 fuses L1D and
> SMEM into *one* banked SRAM with a runtime cache↔scratchpad split. This program **deliberately
> does not reproduce that fusion**: L1D (the core dcache) and SMEM (the core `LocalMem`
> scratchpad) stay **physically distinct SimObjects** with distinct request paths. We adopt the
> split explicitly as a soft-GPU simplification, exactly the recommendation recorded in
> [`../../designs/h100_vortex_mapping.md`](../../designs/h100_vortex_mapping.md) §5.2 ("accept the split
> for the toy … record unification as a follow-on"). This task **relocates ownership** (socket →
> core) and **removes a hierarchy level**; it does not fuse arrays and does not introduce a
> memory-consistency model. "Reproduce architecture, never resources"
> ([roadmap §2](README.md)).

---

## 1. Current state (as-built)

- **The `Socket` level is a pure structural container.** `Socket`
  ([`sim/simx/socket.{h,cpp}`](../../../sim/simx/socket.h)) is a `SimObject<Socket>` (pImpl) whose
  `on_tick()` is **empty** — it does no per-cycle work. It exists only to own the L1 caches and
  **create the cores**. It sits between `Cluster` and `Core`, sized `cores_per_socket =
  VX_CFG_SOCKET_SIZE` ([`socket.cpp:26`](../../../sim/simx/socket.cpp#L26)).
- **L1I + L1D live in the socket.** `Socket::Impl` creates an icache `CacheCluster`
  ([`socket.cpp:34`](../../../sim/simx/socket.cpp#L34)) and a dcache `CacheCluster`
  ([`socket.cpp:54`](../../../sim/simx/socket.cpp#L54)), each with `num_inputs = cores_per_socket`. An
  L1 `MemArbiter` merges icache + dcache traffic onto the socket's upward `mem_req_out`/`mem_rsp_in`
  ports (sized `VX_CFG_L1_MEM_PORTS`, icache prioritized,
  [`socket.cpp:73-100`](../../../sim/simx/socket.cpp#L73)).
- **The socket creates the cores.** `Socket::Impl` computes `core_id = socket_id *
  cores_per_socket + i` and calls `Core::Create(...)`
  ([`socket.cpp:103-107`](../../../sim/simx/socket.cpp#L103)), binding each core's
  `icache_req_out`/`dcache_req_out` to its `CacheCluster` inputs
  ([`socket.cpp:110-118`](../../../sim/simx/socket.cpp#L110)). Each `Core` holds a back-pointer
  `Socket* socket_` ([`core.h:165`](../../../sim/simx/core.h#L165),
  [`core.cpp:893`](../../../sim/simx/core.cpp#L893)) exposed as `Core::socket()`
  ([`core.h:111`](../../../sim/simx/core.h#L111)).
- **SMEM is already per-core** and is *not* owned by the socket. `Core::Impl` creates the
  `LocalMem` scratchpad ([`core.cpp:125`](../../../sim/simx/core.cpp#L125)), the per-LSU-block
  `LocalMemSwitch` ([`core.cpp:136`](../../../sim/simx/core.cpp#L136)) that routes shared→`LocalMem` /
  global→dcache, and the `MemCoalescer`s ([`core.cpp:119`](../../../sim/simx/core.cpp#L119)). Socket
  removal leaves the SMEM datapath in place — it is already where we want it.
- **The cluster reaches cores *through* the socket.** `Cluster::Impl` holds `sockets_` (sized
  `NUM_SOCKETS`, [`cluster.cpp:47`](../../../sim/simx/cluster.cpp#L47)) and
  `cores_per_socket_ = VX_CFG_SOCKET_SIZE` ([`cluster.cpp:49`](../../../sim/simx/cluster.cpp#L49)),
  creates the sockets ([`cluster.cpp:59-63`](../../../sim/simx/cluster.cpp#L59)), and binds
  `sockets_[i]->mem_req_out[j] → l2cache_->core_req_in[i*L1_MEM_PORTS + j]`
  ([`cluster.cpp:108-122`](../../../sim/simx/cluster.cpp#L108), or via a priority `l2arb` when
  extension engines share L2). It re-derives cores from sockets in `get_core`
  ([`cluster.cpp:520-526`](../../../sim/simx/cluster.cpp#L520)), the barrier cascade
  `global_barrier_arrive`/`resume` ([`cluster.cpp:382-410`](../../../sim/simx/cluster.cpp#L382)), and
  the DXA/TEX/OM/RASTER per-core wiring loops `sockets_[s]->core(c)->sfu_unit()` / `->local_mem()`
  ([`cluster.cpp:139-169,222-229,287-293,346-353`](../../../sim/simx/cluster.cpp#L139)).
- **L1 perf counters come from the socket.** The icache/dcache MPM CSRs are read from
  `Socket::perf_stats()` ([`csr_unit.cpp:187`](../../../sim/simx/csr_unit.cpp#L187)).

So today the socket is a thin relay: it owns the L1 caches + their arbiter, forwards barriers /
DCR / running / exitcode / flush between cluster and core, and otherwise does nothing per cycle.

## 2. Target state

| Level | SimX class | Owns — **before** | Owns — **after** |
|---|---|---|---|
| Cluster | `Cluster` | L2 + extension engines; creates sockets | L2 + extension engines; **creates cores directly** |
| ~~Socket (== SM)~~ | ~~`Socket`~~ | ~~L1I + L1D + L1 arbiter; creates cores~~ | **removed** |
| **Core (== SM)** | `Core` | regfile + TCU + `LocalMem` (SMEM) + coalescers | regfile + TCU + `LocalMem` (SMEM) + **L1I + L1D (`CacheCluster`, single-input) + L1 `MemArbiter`** |

The `Socket` level is deleted. Its L1I, L1D, and L1 arbiter move into the `Core`, which already
owns the SMEM scratchpad. Each core now exposes a single post-L1 upward memory interface
(`mem_req_out`/`mem_rsp_in`, sized `VX_CFG_L1_MEM_PORTS`) that the cluster binds straight to L2.
**L1D and SMEM remain two distinct structures** inside the core — this is a hierarchy collapse and
ownership relocation, not a fusion (§4). With the socket gone there is no `SOCKET_SIZE`: `--cores=N`
yields N cores == N SMs directly.

---

## 3. Changes

**3.1 Remove all socket config knobs.** Delete `VX_CFG_SOCKET_SIZE`
([`VX_config.toml:4`](../../../VX_config.toml)) from the tunable surface. In
[`sim/simx/constants.h`](../../../sim/simx/constants.h): **drop `NUM_SOCKETS`**
([`constants.h:61`](../../../sim/simx/constants.h#L61)) and redefine the L2 fan-in
`VX_CFG_L2_NUM_REQS = NUM_CORES * VX_CFG_L1_MEM_PORTS`
([`constants.h:63`](../../../sim/simx/constants.h#L63), was `NUM_SOCKETS * L1_MEM_PORTS`). Rework the
cache-count expressions `VX_CFG_NUM_ICACHES` / `VX_CFG_NUM_DCACHES`
([`VX_config.toml:192-193`](../../../VX_config.toml), were `up(SOCKET_SIZE/4)`) to per-core values (1
each — one L1I and one L1D per core). Drop the `SOCKET_SIZE` term from the `VX_CFG_DXA_MEM_PORTS`
expression ([`VX_config.toml:46`](../../../VX_config.toml)); [Task 2](2__per_core_dxa.md)
re-scopes DXA ports to per-core. Remove `socket_size=` from the CONFIGS banner
([`processor.cpp:137`](../../../sim/simx/processor.cpp#L137)). Re-`configure` from `build/`. (The RTL
build will break here — acceptable, §3.6.)

**3.2 Move the L1 caches into the `Core`.** Relocate the icache and dcache `CacheCluster`
instantiations ([`socket.cpp:34,54`](../../../sim/simx/socket.cpp#L34)) and the L1 `MemArbiter`
([`socket.cpp:73-100`](../../../sim/simx/socket.cpp#L73)) from `Socket::Impl` into `Core::Impl`
([`core.cpp`](../../../sim/simx/core.cpp)), now **single-input** (`num_inputs = 1`, one core). The
core's existing internal fetch→icache and `dcache_req_out`→dcache wiring stays, but now terminates
at the **core-owned** caches. The core exposes a new upward `mem_req_out`/`mem_rsp_in` channel pair
(sized `VX_CFG_L1_MEM_PORTS`) *after* its L1 arbiter — the same shape the socket used to present to
the cluster. The old `Core::icache_req_out`/`dcache_req_out` ports become **internal** to the core
(bound to its own `CacheCluster`s), no longer crossing a core→socket boundary.

**3.3 Rewire `Cluster` → `Core` directly.** In `Cluster::Impl`: drop the `sockets_` vector, the
`cores_per_socket_` field, and the two-stage `core_id` math; **create the cores directly**
(`core_id = cluster_id * cores_per_cluster + i`). Bind each `cores_[i]->mem_req_out[j] →
l2cache_->core_req_in[i*L1_MEM_PORTS + j]` (and the priority `l2arb` path when extensions share L2)
— the same index arithmetic as today with the socket dimension removed
([`cluster.cpp:108-122`](../../../sim/simx/cluster.cpp#L108)). Replace **every** `sockets_[s]->core(c)`
traversal with direct `cores_[i]` indexing: the DXA/TEX/OM/RASTER wiring loops
([`cluster.cpp:139-169,222-229,287-293,346-353`](../../../sim/simx/cluster.cpp#L139)), `get_core`
([`cluster.cpp:520-526`](../../../sim/simx/cluster.cpp#L520)), and the barrier cascade
`global_barrier_arrive`/`resume` ([`cluster.cpp:382-410`](../../../sim/simx/cluster.cpp#L382)), which
now resumes `cores_[c]` directly instead of relaying through `sockets_[s]->global_barrier_resume`.

**3.4 Replace `Core::socket()` with `Core::cluster()`.** The core's back-pointer becomes
`Cluster* cluster_` (the socket was only ever a pass-through to the cluster). Update every caller
of `Core::socket()`: the barrier arrive/resume in
[`barrier_unit.cpp:64,147`](../../../sim/simx/barrier_unit.cpp#L64) (`core_->socket()->global_barrier_arrive`
→ `core_->cluster()->global_barrier_arrive`), the raster path in
[`sfu_unit.cpp:150`](../../../sim/simx/sfu_unit.cpp#L150), the KMU reach in
[`cta_dispatcher.cpp:28`](../../../sim/simx/cta_dispatcher.cpp#L28), the processor/cluster stat reads
in [`csr_unit.cpp:144,186-205,245-285`](../../../sim/simx/csr_unit.cpp#L144), and
`socket_->cluster()->processor()` in [`core.cpp:910`](../../../sim/simx/core.cpp#L910). **Re-source
the L1 perf CSRs**: add icache/dcache to `Core::PerfStats` and read the MPM counters from the
core-owned caches instead of `Socket::perf_stats()`
([`csr_unit.cpp:187`](../../../sim/simx/csr_unit.cpp#L187)).

**3.5 Delete `socket.{h,cpp}`.** Remove [`sim/simx/socket.h`](../../../sim/simx/socket.h) and
[`sim/simx/socket.cpp`](../../../sim/simx/socket.cpp) and drop them from the SimX build
([`sim/simx/Makefile`](../../../sim/simx/Makefile) / CMake sources). No façade, no empty wrapper — the
level is gone (no-façades rule, [roadmap §2](README.md)).

**3.6 1:1 mapping.** The core-owned L1I, L1D, L1 `MemArbiter`, and SMEM now correspond to a future
[`VX_core.sv`](../../../hw/rtl/core/VX_core.sv) as **the SM module** — `VX_cache_cluster` (icache +
dcache), the L1 arbiter, `VX_local_mem` + `VX_lmem_switch` all land inside `VX_core.sv`.
[`VX_socket.sv`](../../../hw/rtl/VX_socket.sv) is **retired**; `VX_cluster.sv` instantiates cores
directly. **RTL is out of scope for this task** (SimX-only) — the RTL build is expected to break
until that companion move is done separately.

---

## 4. Explicit non-goal — the fused L1D/SMEM array

The H100's L1D and SMEM are **one** SRAM with a runtime-configurable cache↔scratchpad split. This
program **does not** reproduce that fusion, by design:

- **We keep two structures.** The core dcache (`CacheCluster`, tags/MSHR, write-through LLC
  behavior) and the core `LocalMem` (flat banked scratchpad, always-hits, no tags) stay physically
  and logically distinct, each with its own request path. There is **no** shared banked array,
  **no** way-based carveout, and **no** per-launch dynamic partition.
- **Why this is acceptable at soft-GPU scale.** Fusion is a *physical/capacity-sharing* property
  (one array, dual-rail power, floorplan). The *architectural placement* that matters for the toy
  — L1D and SMEM sitting together at the SM (now the core), reachable per-SM — is delivered by the
  hierarchy collapse in §3 without fusing. This is exactly the trade the gap analysis recommends
  ([`../../designs/h100_vortex_mapping.md`](../../designs/h100_vortex_mapping.md) §5.2).
- **Downstream tasks do not depend on fusion.** Tasks 2–4 attach to the **SMEM scratchpad port**
  (DXA writes, the DSMEM remote port, the proxy-ordering / mbarrier logic) — a property of the
  core-resident scratchpad, *not* of a fused array. Making the core the SM is the only Task-1
  property they require.

The fused array remains a recorded, out-of-scope follow-on (§8).

---

## 5. Config & ABI changes

- [`VX_config.toml`](../../../VX_config.toml): **remove `SOCKET_SIZE` entirely** (not just pin it to
  1) — no socket knob survives. `NUM_ICACHES`/`NUM_DCACHES` become per-core (1 each);
  `DXA_MEM_PORTS` drops its `SOCKET_SIZE` term. **Keep `DCACHE_SIZE` and `LMEM_LOG_SIZE` as two
  independent capacity knobs** — there is no unified capacity/partition knob, because there is no
  unified array. `LMEM_NUM_BANKS` and `DCACHE_NUM_BANKS` stay separate. Capacities stay at
  soft-GPU scale (H100's 256 KB is a reference, not a target).
- [`VX_types.toml`](../../../VX_types.toml): the shared-memory address map (`VX_MEM_LMEM_BASE_ADDR`,
  `VX_CFG_LMEM_LOG_SIZE`, and `VX_CSR_LOCAL_MEM_BASE` read by the WGMMA descriptor helper
  `vx_make_smem_desc`, [`sw/kernel/include/vx_tensor.h`](../../../sw/kernel/include/vx_tensor.h)) is
  **unchanged** — SMEM does not move (it was already in the core); only the level above it is
  removed. No carveout CSR/DCR is introduced.
- [`sw/common/`](../../../sw/common/): no new host↔kernel ABI is required (the isolation rule still
  applies — never reference `sim/*` from `sw/*`).

## 6. C++ kernel verification

The change is a structural refactor (level removal + ownership relocation), so verification is
**regression of the existing kernels** with the socket level gone, asserting **identical output**
and that both the scratchpad and cache paths still move traffic (`--perf=2` counters):

```bash
# scratchpad path (SMEM core-resident, unchanged) — core-as-SM shape, one SM
CONFIGS="-DVX_CFG_NUM_THREADS=32" ./ci/blackbox.sh --driver=simx --app=<shared-mem app> --cores=1 --warps=64 --threads=32

# global / cache path — now hits the core-owned L1D directly (no socket)
./ci/blackbox.sh --driver=simx --app=vecadd --threads=32

# DXA writes + TCU tile-buffer still target the core SMEM (DXA stays cluster-level until Task 2)
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=dxa_copy
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_wg --threads=32

# multi-SM: each core owns its own L1I + L1D + SMEM; cluster L2 aggregates
CONFIGS="-DVX_CFG_NUM_THREADS=32" ./ci/blackbox.sh --driver=simx --app=<shared-mem app> --cores=4 --warps=64 --threads=32
```

Run inside the Docker dev container
([`miscs/docker/vortex-dev.sh`](../../../miscs/docker/vortex-dev.sh): `build` → `up` → `shell`, from
`build/`). Inspect `--perf=2` counters to confirm the L1I/L1D hits/misses and scratchpad
reads/writes both still move and are now attributed **per core**. Behavior is **identical** to the
pre-task baseline; only the hierarchy/topology changed. Keep each invocation under the 120 s cap.

## 7. Risks

- **L1 perf CSR re-sourcing.** The icache/dcache MPM counters move from `Socket::perf_stats()` to
  the core-owned caches ([`csr_unit.cpp:187`](../../../sim/simx/csr_unit.cpp#L187)); confirm the CSR
  values match the pre-task baseline per core.
- **L2 fan-in index math.** `VX_CFG_L2_NUM_REQS` becomes `NUM_CORES * L1_MEM_PORTS`; the
  `cores_[i]->mem_req_out[j] → l2cache_->core_req_in[i*L1_MEM_PORTS + j]` binding must line up with
  the new count, and the `l2arb` extension path must re-index without the socket dimension.
- **Barrier cascade.** `global_barrier_arrive`/`resume` now runs `Core ↔ Cluster` directly with no
  socket relay ([`cluster.cpp:382-410`](../../../sim/simx/cluster.cpp#L382),
  [`barrier_unit.cpp:64,147`](../../../sim/simx/barrier_unit.cpp#L64)); preserve the per-core release
  semantics (the DXA `flags.dxa_notify_done → core->barrier_event_release` contract that
  [Task 2](2__per_core_dxa.md) carries).
- **This core-owned SMEM port is where later tasks attach.** [Task 3](3__dsmem_microarchitecture.md)
  bolts the DSMEM remote client onto the core `LocalMem` — no such logic in Task 1, but keep the
  port shaped cleanly.
- **`is_llc` / write-back semantics** of the dcache are untouched (it is not merged with SMEM);
  confirm the LLC role still holds with no L2 (toy uses L3 as the global L2).
- **Trace naming changes.** SimX emits `clusterX-socketY-coreZ` object names
  ([`cluster.cpp:61`](../../../sim/simx/cluster.cpp#L61),
  [`socket.cpp:105`](../../../sim/simx/socket.cpp#L105)); with the socket gone these become
  `clusterX-coreZ`, which breaks the socket-parsing in
  [`ci/trace_csv.py`](../../../ci/trace_csv.py) and [`ci/perfetto.py`](../../../ci/perfetto.py). Those
  are host tools (not compiled into simx) — a **follow-on tooling fix** (§8), out of scope for the
  simulator change itself.

## 8. Proposed but not yet implemented (Task 1 scope edges)

- **RTL companion move** — retiring `VX_socket.sv` and moving the L1 caches / arbiter / SMEM into
  `VX_core.sv`, with `VX_cluster.sv` instantiating cores directly. Out of scope here (SimX-only);
  the RTL build stays broken until it lands.
- **Trace-tooling socket-parse cleanup** — update `ci/trace_csv.py` / `ci/perfetto.py` to the
  `clusterX-coreZ` naming (drop `socket_size=` / `socketN` parsing).
- **Fused L1D/SMEM array** — the single banked SRAM with a runtime cache↔scratchpad partition
  (way-based carveout / per-launch dynamic partition, the
  `cudaFuncAttributePreferredSharedMemoryCarveout` analog). Explicitly **out of scope** (§4);
  recorded as a follow-on that would build on the core-resident SMEM established here.
- Bank-conflict swizzling for the scratchpad region (tracked with the WGMMA SMEM swizzle item).
