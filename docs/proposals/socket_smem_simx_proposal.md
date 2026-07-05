# Task 1 — Socket-Level SMEM Relocation — SimX Proposal

**Status:** proposal (pre-implementation). **Scope:** SimX only
([`sim/simx/`](../../sim/simx/)). Part of the
[Toy-H100 SimX Roadmap](toy_h100_simx_roadmap_proposal.md); this is **Task 1**, the foundation
for Tasks 2–3. Closes the **"L1D/SMEM placement"** divergence from
[`../designs/h100_vortex_mapping.md`](../designs/h100_vortex_mapping.md) §5.1–5.2.

**Goal:** relocate shared-memory ownership from the **core** up to the **socket**, and fix
`SOCKET_SIZE = 1` (removing the knob), so the socket owns **SMEM alongside its L1D and L1I** —
three *separate, co-located* per-SM structures. With one core per socket, socket == SM and all
three become effectively per-SM, matching the H100's placement.

> **What this task does *not* do — the fused L1D/SMEM is out of scope.** The H100 fuses L1D and
> SMEM into *one* banked SRAM with a runtime cache↔scratchpad split. This program **deliberately
> does not reproduce that fusion**: L1D (the socket dcache) and SMEM (the relocated scratchpad)
> stay **physically distinct SimObjects** with distinct request paths. We adopt the split
> explicitly as a soft-GPU simplification — exactly the recommendation already recorded in
> [`../designs/h100_vortex_mapping.md`](../designs/h100_vortex_mapping.md) §5.2 ("accept the split
> for the toy … record unification as a follow-on"). The *placement* gap (SMEM per-core →
> per-SM) is what Task 1 closes; the *unification* gap is a recorded non-goal (see §4, §8).

---

## 1. Current state (as-built)

- **SMEM is per-core.** [`Core::Impl`](../../sim/simx/core.cpp#L125) creates a `LocalMem`
  ([`sim/simx/mem/local_mem.h:21`](../../sim/simx/mem/local_mem.h#L21)) sized
  `1 << VX_CFG_LMEM_LOG_SIZE`, with `lmem_num_reqs = LSU_NUM_REQS + TCU_ENABLED + DXA_ENABLED`
  input ports ([`core.cpp:124`](../../sim/simx/core.cpp#L124)). Internally it is a
  `MemCrossBar` (`num_reqs → num_banks`, bank = hashed line address). Port allocation:
  `[0:LSU_NUM_REQS)` = LSU, `[LSU_NUM_REQS]` = TCU tile-buffer
  ([`core.cpp:250-251`](../../sim/simx/core.cpp#L250)), `[LSU_NUM_REQS+1]` = DXA write port.
- **L1D (dcache) and L1I are per-socket.** [`Socket::Impl`](../../sim/simx/socket.cpp#L54) creates
  a `CacheCluster` dcache (and an icache at [`socket.cpp:34`](../../sim/simx/socket.cpp#L34)) with
  `num_inputs = cores_per_socket = VX_CFG_SOCKET_SIZE`
  ([`socket.cpp:26`](../../sim/simx/socket.cpp#L26)). An L1 `MemArbiter`
  ([`socket.cpp:78`](../../sim/simx/socket.cpp#L78), icache prioritized) merges toward L2.
- **The shared/global split is enforced by address decode.** `LocalMemSwitch`
  ([`sim/simx/mem/local_mem_switch.h`](../../sim/simx/mem/local_mem_switch.h), created per LSU
  block at [`core.cpp:136`](../../sim/simx/core.cpp#L136)) routes each LSU request by
  `get_addr_type(addr)`: **Shared → `LocalMem`**, **Global → dcache** (via coalescer/MMU →
  `core->dcache_req_out` → socket dcache). `get_addr_type` keys on `VX_MEM_LMEM_BASE_ADDR` and
  `VX_CFG_LMEM_LOG_SIZE` (`sim/simx/types.h`).

So today: scratchpad lives in the **core**, the caches live in the **socket**, and they are
physically distinct SimObjects with distinct request paths.

## 2. Target state

| | Owns — **before** | Owns — **after** |
|---|---|---|
| **Socket (== SM)** | L1I + L1D (dcache) | L1I + L1D (dcache) + **SMEM (`LocalMem`)** — three separate structures |
| **Core** | regfile + TCU + **`LocalMem`** | regfile + TCU + an **SMEM *interface*** (no storage) |

`SOCKET_SIZE` fixed to **1** ⇒ exactly one core per socket ⇒ the socket's L1I, L1D, and SMEM are
each effectively per-SM, matching H100's *placement*. The core exposes shared memory the same way
it already exposes L1D: through ports, owning no array. **L1D and SMEM remain two distinct
structures** — this is a pure *ownership relocation*, not a fusion (§4).

The change is a single restructuring; there is no second "fuse the arrays" stage.

---

## 3. Changes

**3.1 Fix `SOCKET_SIZE = 1` and remove the knob.** Set `VX_CFG_SOCKET_SIZE = 1` in
[`VX_config.toml`](../../VX_config.toml) and **remove it as a tunable knob** — hardwire one core
per socket and drop any `blackbox.sh`/CONFIGS override path that sets it. `NUM_SOCKETS =
up(NUM_CORES / SOCKET_SIZE)` then equals `NUM_CORES` (`sim/simx/constants.h`); `--cores=N` yields
N sockets, each one SM. Re-`configure`. (This is gap analysis §5.1, now made permanent rather
than merely recommended.)

**3.2 Strip SMEM storage from the core; expose an SMEM interface.** Remove the `LocalMem`
member and the per-core local interconnect from [`Core::Impl`](../../sim/simx/core.cpp#L125): the
`LocalMemSwitch`'s `Lmem` path, the `lsu_lmem_adapter`, the TCU tile-buffer LMEM bind
([`core.cpp:250-251`](../../sim/simx/core.cpp#L250)), and the DXA LMEM input port. The core
instead **exposes an SMEM interface** — a `smem_req_out[]` / `smem_rsp_in[]` channel pair, shaped
exactly like the existing `dcache_req_out`/`dcache_rsp_in`
([`core.h`](../../sim/simx/core.h)). The three internal SMEM clients (LSU shared path, TCU
tile-buffer, DXA write port) now drive *out of the core* onto these ports.

**3.3 Relocate `LocalMem` to the socket.** Move the `LocalMem` instantiation from
[`Core::Impl`](../../sim/simx/core.cpp#L125) into [`Socket::Impl`](../../sim/simx/socket.cpp#L54),
so the socket owns one `LocalMem` scratchpad next to its existing dcache and icache. **Keep the
dcache `CacheCluster` and the icache exactly as they are** — this task does not touch, merge, or
delete either cache. `LocalMem`'s `Config` (capacity `1 << LMEM_LOG_SIZE`, banks, word size,
`read_word` functional path) is unchanged; only its owner moves. This is a *relocation*, not a
new class and not a wrapper (no façade — [roadmap §2](toy_h100_simx_roadmap_proposal.md)).

**3.4 Wiring** (`SOCKET_SIZE = 1` ⇒ one core ↔ one socket):
- Core `smem_req_out` / `smem_rsp_in` (shared) → the socket-resident `LocalMem` input/output
  ports. The core's `dcache_req_out` (global) continues to the socket dcache **unchanged**.
- The socket dcache's `mem_req_out`/`mem_rsp_in` still feed the L1 `MemArbiter` (with the icache,
  icache prioritized) → socket `mem_req_out` → cluster L2, **unchanged**. `LocalMem` is a
  terminal scratchpad and has no path to L2 (it never misses).
- TCU tile-buffer and DXA write port (now exiting the core via `smem_req_out`) reach the
  socket `LocalMem`. Preserve the DXA completion `tx_callback` →
  `core->barrier_event_release(...)` semantics ([`cluster.cpp:158`](../../sim/simx/cluster.cpp#L158)
  today) across the new socket↔core interface boundary — the resident core's barrier state is
  notified when a `flags.dxa_notify_done` write lands.

**3.5 1:1 mapping.** The relocated `LocalMem` stands in for the RTL `VX_local_mem`
([`hw/rtl/mem/VX_local_mem.sv`](../../hw/rtl/mem/VX_local_mem.sv)) **moved from
[`VX_core.sv`](../../hw/rtl/core/VX_core.sv) into [`VX_socket.sv`](../../hw/rtl/VX_socket.sv)** —
together with its `VX_lmem_switch`. The dcache stays `VX_cache_cluster` in `VX_socket.sv`.
Ownership stays explicit per level; no class is fused.

**3.6 Acceptance.** Existing kernels still pass with `SOCKET_SIZE = 1` and socket-resident SMEM:
`vecadd` (global / cache path), a shared-memory kernel (scratchpad path), and — if enabled —
`dxa_copy` (`-DVX_CFG_EXT_DXA_ENABLE`) and `sgemm_tcu_wg` (`-DVX_CFG_EXT_TCU_ENABLE
-DVX_CFG_TCU_WGMMA_ENABLE`). Behavior is **identical** to the pre-task baseline; only ownership
moved. No numerical or perf-counter change is expected beyond the ownership/topology relabeling.

---

## 4. Explicit non-goal — the fused L1D/SMEM array

The H100's L1D and SMEM are **one** SRAM with a runtime-configurable cache↔scratchpad split.
This program **does not** reproduce that fusion, by design:

- **We keep two structures.** The socket dcache (`CacheCluster`, tags/MSHR, write-through LLC
  behavior) and the socket `LocalMem` (flat banked scratchpad, always-hits, no tags) stay
  physically and logically distinct, each with its own request path. There is **no** shared
  banked array, **no** way-based carveout, and **no** per-launch dynamic partition.
- **Why this is acceptable at soft-GPU scale.** The fusion is a *physical/capacity-sharing*
  property (one array, dual-rail power, floorplan). The *architectural placement* that matters
  for the toy — SMEM sitting at the SM level next to the L1 caches, reachable per-SM — is
  delivered by the relocation in §3 without fusing. This is precisely the trade the gap analysis
  already recommends ([`../designs/h100_vortex_mapping.md`](../designs/h100_vortex_mapping.md)
  §5.2: "accept the split for the toy … record unification as a follow-on").
- **Downstream tasks do not depend on fusion.** Tasks 2–4 attach to the **SMEM scratchpad port**
  (DXA writes, the DSMEM remote port, the proxy-ordering/mbarrier logic) — a property of the
  socket-resident scratchpad, *not* of a fused array. Relocating SMEM to the socket is the only
  Task-1 property they require.

The fused array remains a recorded, out-of-scope follow-on (§8).

---

## 5. Config & ABI changes

- [`VX_config.toml`](../../VX_config.toml): set `SOCKET_SIZE = 1` and remove it from the tunable
  surface. **Keep `DCACHE_SIZE` and `LMEM_LOG_SIZE` as the two independent capacity knobs** —
  there is no unified capacity/partition knob, because there is no unified array. `LMEM_NUM_BANKS`
  and `DCACHE_NUM_BANKS` stay separate. Keep both capacities at soft-GPU scale (H100's 256 KB is a
  reference, not a target; [roadmap §2](toy_h100_simx_roadmap_proposal.md)).
- [`VX_types.toml`](../../VX_types.toml): the shared-memory address map
  (`VX_MEM_LMEM_BASE_ADDR`, `VX_CFG_LMEM_LOG_SIZE`, and `VX_CSR_LOCAL_MEM_BASE` read by the WGMMA
  descriptor helper `vx_make_smem_desc`,
  [`sw/kernel/include/vx_tensor.h`](../../sw/kernel/include/vx_tensor.h)) is **unchanged** — the
  scratchpad keeps the same base and size, it just lives one level up. No carveout CSR/DCR is
  introduced (that belonged to the dropped dynamic-partition milestone).
- [`sw/common/`](../../sw/common/): no new host↔kernel ABI is required for this task (the
  isolation rule still applies — never reference `sim/*` from `sw/*`).

## 6. C++ kernel verification

The feature is a topology relocation, so verification is **regression of the existing split-path
kernels** under `SOCKET_SIZE = 1`, asserting identical output and that both the scratchpad and
cache paths still move traffic (via `--perf=2` counters):

```bash
# scratchpad path (SMEM now socket-resident) — toy-H100 SM shape
CONFIGS="-DVX_CFG_NUM_THREADS=32" ./ci/blackbox.sh --driver=simx --app=<shared-mem app> --cores=1 --warps=64 --threads=32

# cache path unchanged under socket-resident SMEM
./ci/blackbox.sh --driver=simx --app=vecadd --threads=32

# DXA writes now exit the core to the socket SMEM; TCU tile-buffer likewise
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=dxa_copy
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_wg --threads=32

# multi-SM: each socket owns its own SMEM
CONFIGS="-DVX_CFG_NUM_THREADS=32" ./ci/blackbox.sh --driver=simx --app=<shared-mem app> --cores=4 --warps=64 --threads=32
```

Inspect `--perf=2` counters to confirm scratchpad reads/writes and cache hits/misses both still
move, and that per-socket attribution is correct. Keep each invocation under the 120 s cap.

## 7. Risks

- **Socket↔core SMEM interface fidelity.** The SMEM clients (LSU-shared, TCU tile-buffer, DXA
  write) now cross a core→socket boundary. Preserve the TCU/DXA deadlock contract (the
  read-during-write interlock flagged in
  [`../designs/tensor_core_wgmma_engine.md`](../designs/tensor_core_wgmma_engine.md) §7.1) across
  the new port. This socket SMEM port is also where later tasks attach: the DSMEM remote client
  ([Task 3](dsmem_smtosm_simx_proposal.md)) and the generic↔async **proxy-ordering** contract
  ([Task 4](proxy_model_simx_proposal.md), the LSU path = generic proxy, the DXA path = async
  proxy) both bolt onto this port. No proxy/remote logic in Task 1; just be aware the port is
  where it will live.
- **DXA completion callback across the boundary.** The `flags.dxa_notify_done` write must still
  drive `core->barrier_event_release(...)` now that the scratchpad is a socket sibling rather than
  a core child. Route the completion callback across the socket↔core interface without dropping
  the per-CTA release semantics.
- **`is_llc` / write-back semantics** of the **dcache** are untouched by this task (it is not
  merged with SMEM), but confirm the LLC role still holds when there is no L2 (toy uses L3 as
  global L2).
- **Coalescer placement.** The dcache coalescer ([`core.cpp` MemCoalescer](../../sim/simx/core.cpp#L136))
  continues to feed the socket dcache via `dcache_req_out` unchanged — confirm word/line-size
  assumptions still hold after `SOCKET_SIZE = 1`.

## 8. Proposed but not yet implemented (Task 1 scope edges)

- **Fused L1D/SMEM array** — the single banked SRAM with a runtime cache↔scratchpad partition
  (way-based carveout and/or per-launch dynamic partition, the
  `cudaFuncAttributePreferredSharedMemoryCarveout` analog). Explicitly **out of scope** for this
  program (§4); recorded here and in
  [`../designs/h100_vortex_mapping.md`](../designs/h100_vortex_mapping.md) §5.2 / §7 as a
  follow-on that would build on the socket-resident SMEM established here.
- Bank-conflict swizzling for the scratchpad region (tracked with the WGMMA SMEM swizzle item).
