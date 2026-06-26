# Task 1 — Unified L1D/SMEM at the Socket (`UnifiedL1`) — SimX Proposal

**Status:** proposal (pre-implementation). **Scope:** SimX only
([`sim/simx/`](../../sim/simx/)). Part of the
[Toy-H100 SimX Roadmap](toy_h100_simx_roadmap_proposal.md); this is **Task 1**, the foundation
for Tasks 2–3. Closes the **"unified L1D/SMEM"** and **"L1D placement"** divergences from
[`../designs/h100_vortex_mapping.md`](../designs/h100_vortex_mapping.md) §5.1–5.2.

**Goal:** relocate shared-memory ownership from the core up to the socket, and model the H100's
per-SM **fused L1D + SMEM** as a single banked SRAM with a configurable cache↔scratchpad
partition — replacing the two separate structures Vortex uses today.

---

## 1. Current state (as-built)

- **SMEM is per-core.** [`Core::Impl`](../../sim/simx/core.cpp#L125) creates a `LocalMem`
  ([`sim/simx/mem/local_mem.h:21`](../../sim/simx/mem/local_mem.h#L21)) sized
  `1 << VX_CFG_LMEM_LOG_SIZE`, with `lmem_num_reqs = LSU_NUM_REQS + TCU_ENABLED + DXA_ENABLED`
  input ports ([`core.cpp:124`](../../sim/simx/core.cpp#L124)). Internally it is a
  `MemCrossBar` (`num_reqs → num_banks`, bank = hashed line address). Port allocation:
  `[0:LSU_NUM_REQS)` = LSU, `[LSU_NUM_REQS]` = TCU tile-buffer
  ([`core.cpp:250-251`](../../sim/simx/core.cpp#L250)), `[LSU_NUM_REQS+1]` = DXA write port.
- **L1D (dcache) is per-socket.** [`Socket::Impl`](../../sim/simx/socket.cpp#L54) creates a
  `CacheCluster` dcache (and an icache at [`socket.cpp:34`](../../sim/simx/socket.cpp#L34)) with
  `num_inputs = cores_per_socket = VX_CFG_SOCKET_SIZE`
  ([`socket.cpp:26`](../../sim/simx/socket.cpp#L26)). An L1 `MemArbiter`
  ([`socket.cpp:78`](../../sim/simx/socket.cpp#L78), icache prioritized) merges toward L2.
- **The split is enforced by address decode.** `LocalMemSwitch`
  ([`sim/simx/mem/local_mem_switch.h`](../../sim/simx/mem/local_mem_switch.h), created per LSU
  block at [`core.cpp:136`](../../sim/simx/core.cpp#L136)) routes each LSU request by
  `get_addr_type(addr)`: **Shared → `LocalMem`**, **Global → dcache** (via coalescer/MMU →
  `core->dcache_req_out` → socket dcache). `get_addr_type` keys on `VX_MEM_LMEM_BASE_ADDR` and
  `VX_CFG_LMEM_LOG_SIZE` (`sim/simx/types.h`).

So today: scratchpad lives in the **core**, cache lives in the **socket**, and they are
physically distinct SimObjects with distinct request paths.

## 2. Target state

| | Owns |
|---|---|
| **Socket** | L1I + **`UnifiedL1`** (fused L1D/SMEM) |
| **Core** | regfile + TCU + an **SMEM/dcache *interface*** (no storage) |

`SOCKET_SIZE` fixed to **1** ⇒ exactly one core per socket ⇒ `UnifiedL1` is effectively per-SM,
matching H100. The core exposes shared-memory the same way it already exposes L1I/L1D: through
ports, owning no array.

This is delivered in two stages.

---

## 3. Stage 1a — restructure hierarchy & relocate SMEM ownership

**3a.1 Fix `SOCKET_SIZE = 1`.** Set `VX_CFG_SOCKET_SIZE = 1` in
[`VX_config.toml`](../../VX_config.toml) and remove it as a tunable knob (drop any
`blackbox.sh`/CONFIGS override path). `NUM_SOCKETS = up(NUM_CORES / SOCKET_SIZE)` then equals
`NUM_CORES` (`sim/simx/constants.h`); `--cores=N` yields N sockets, each one SM. Re-`configure`.

**3a.2 Strip SMEM storage from the core.** Delete the `LocalMem` member and all per-core
local interconnect from [`Core::Impl`](../../sim/simx/core.cpp#L125): the `LocalMemSwitch`'s
`Lmem` path, the `lsu_lmem_adapter`, the TCU tile-buffer LMEM bind
([`core.cpp:250-251`](../../sim/simx/core.cpp#L250)), and the DXA LMEM input port. The core
instead **exposes an SMEM interface** — a `smem_req_out[]` / `smem_rsp_in[]` channel pair,
shaped exactly like the existing `dcache_req_out`/`dcache_rsp_in`
([`core.h`](../../sim/simx/core.h)). The three internal LMEM clients (LSU shared path, TCU
tile-buffer, DXA write port) now drive *out of the core* onto these ports.

**3a.3 Move SMEM ownership to the socket; introduce `UnifiedL1`.** Delete the separate dcache
`CacheCluster` ([`socket.cpp:54`](../../sim/simx/socket.cpp#L54)) **and** the `LocalMem` class;
create a new socket-resident class **`UnifiedL1`** (`sim/simx/mem/unified_l1.{h,cpp}`),
implemented as a real module (not a wrapper over the two deleted ones). In Stage 1a it may keep
a logically-separate cache region and scratchpad region internally, but as **one class with one
set of request ports**; Stage 1b fuses them into one physical array. The icache
([`socket.cpp:34`](../../sim/simx/socket.cpp#L34)) is untouched.

**3a.4 Wiring** (`SOCKET_SIZE=1` ⇒ one core ↔ one `UnifiedL1`):
- Core `dcache_req_out` (global) **and** `smem_req_out` (shared) → `UnifiedL1` inputs.
- `UnifiedL1` `mem_req_out`/`mem_rsp_in` → L1 `MemArbiter` (with icache, icache prioritized) →
  socket `mem_req_out` → cluster L2.
- TCU tile-buffer and DXA write port (now exiting the core via `smem_req_out`) reach the
  `UnifiedL1` scratchpad region. Preserve the DXA completion `tx_callback` →
  `core->barrier_event_release(...)` semantics ([`cluster.cpp:158`](../../sim/simx/cluster.cpp#L158)
  today) across the new interface boundary.

**3a.5 Acceptance (1a):** existing kernels still pass with `SOCKET_SIZE=1`: `vecadd` (global /
cache region), a shared-memory kernel (scratchpad region), and — if enabled — `dxa_copy`
(`-DVX_CFG_EXT_DXA_ENABLE`) and `sgemm_tcu_wg` (`-DVX_CFG_EXT_TCU_ENABLE
-DVX_CFG_TCU_WGMMA_ENABLE`). Behavior is identical to pre-1a; only ownership moved.

---

## 4. Stage 1b — fuse L1D and SMEM inside `UnifiedL1`

Model the H100 fusion as **behavior/timing** only (floorplan, dual-rail power, power-gating are
physical/RTL concerns, out of scope). The five simulator-level equivalents:

**4b.1 One array, not two.** A single banked SRAM of configurable total capacity (target
**256 KB**), replacing the two fixed arrays. `UnifiedL1::Config`:
`{ total_capacity, num_banks (log2 B), line_size, word_size, max_cache_ways, partition, ... }`.

**4b.2 Configurable cache↔scratchpad partition (staged).**
- **Milestone A — config-time, way-based partition.** A new `VX_CFG_*` knob (e.g.
  `VX_CFG_UNIFIED_L1_SMEM_WAYS` or a carveout fraction) splits the array into a cache region
  (the remaining ways) and a scratchpad region at build time. Derive `LMEM` size and
  `DCACHE` size from the one capacity + partition rather than two independent knobs.
- **Milestone B — per-launch dynamic partition (end state).** The host selects the
  cache/scratchpad split per kernel launch (the CUDA `cudaFuncAttributePreferredShared
  MemoryCarveout` analog), delivered via a CSR/DCR consumed by `UnifiedL1` at launch. Wire the
  request through `vx_launch_info_t.lmem_size` already plumbed by the runtime.

**4b.3 Unified request routing.** A single LSU→array path: one `MemCrossBar`-style router that
sends each request to the correct bank whether that bank currently backs cache or scratchpad.
The router consults the active partition to pick banks. No second datapath.

**4b.4 Tags only over the cache partition.** Size the tag/MSHR model for the **maximum** cache
configuration (largest legal cache ways). The scratchpad region is directly addressed and
consults no tags — a scratchpad access skips lookup entirely.

**4b.5 Generic address decoding.** Keep `get_addr_type(addr)` (Global vs Shared vs IO) but
re-interpret inside `UnifiedL1`: **Shared → scratchpad banks**, **Global → cache banks**,
selected by the partition config. The `LocalMemSwitch` in the core collapses into "tag the
request's address space and forward to the single `smem`/`dcache` interface"; the bank/region
decision lives in `UnifiedL1`.

**4b.6 Address map consistency.** Ensure `VX_MEM_LMEM_BASE_ADDR` + the scratchpad size and
`VX_CSR_LOCAL_MEM_BASE` (read by the WGMMA descriptor helper `vx_make_smem_desc`,
[`sw/kernel/include/vx_tensor.h`](../../sw/kernel/include/vx_tensor.h)) follow the active
partition. With dynamic partition (Milestone B), the scratchpad base/size become launch-time
values surfaced through the existing CSR.

---

## 5. `UnifiedL1` SimX class shape (1:1 with future `VX_unified_l1`)

```
class UnifiedL1 : public SimObject<UnifiedL1> {
  struct Config { uint32_t total_capacity, line_size, word_size;
                  uint8_t  B /*log2 banks*/, max_cache_ways;
                  partition_t partition;  uint16_t mshr_size; uint8_t latency; ... };
  std::vector<SimChannel<MemReq>> Inputs;    // LSU(global), LSU(shared), TCU, DXA
  std::vector<SimChannel<MemRsp>> Outputs;
  std::vector<SimChannel<MemReq>> mem_req_out;  // miss/refill → L1 arb → L2
  std::vector<SimChannel<MemRsp>> mem_rsp_in;
  uint32_t read_word(uint64_t local_addr);   // functional path for TCU_LD (mirror LocalMem:57)
  const PerfStats& perf_stats() const;        // cache: reads/writes/misses/evictions
};                                            // scratchpad: reads/writes/bank_stalls
```

Mirror `LocalMem`'s functional `read_word` ([`local_mem.h:57`](../../sim/simx/mem/local_mem.h#L57))
so the TCU `TCU_LD`/AGU zero-latency path survives. Reuse `MemCrossBar`, `MemArbiter`,
`Cache`'s tag/MSHR helpers where they compose cleanly — but `UnifiedL1` is one class, not a
holder of a `Cache` + a `LocalMem`.

## 6. Config & ABI changes

- [`VX_config.toml`](../../VX_config.toml): `SOCKET_SIZE=1`; collapse `DCACHE_SIZE` +
  `LMEM_LOG_SIZE` into a unified-L1 `total_capacity` + partition knob(s); target 256 KB; keep
  `LMEM_NUM_BANKS`/`DCACHE_NUM_BANKS` reconciled into `UnifiedL1` banks.
- [`VX_types.toml`](../../VX_types.toml): scratchpad base/size in the address map; a CSR/DCR for
  the dynamic carveout (Milestone B).
- [`sw/common/`](../../sw/common/): any host↔kernel shared ABI for the carveout request
  (isolation rule — never reference `sim/*` from `sw/*`).

## 7. C++ kernel verification

Primary new kernel: **`tests/regression/unified_l1/`** — a kernel that uses **both**
`__local_mem()` scratchpad **and** `__global` arrays in the same launch (e.g. tiled copy /
blocked reduction), run across several partition settings, asserting numerically correct output
and exercising both regions. Plus regression of the existing split-path kernels:

```bash
# scratchpad + cache exercise, toy-H100 SM shape
make -C tests/regression/unified_l1 clean
CONFIGS="-DVX_CFG_NUM_THREADS=32" make -C tests/regression/unified_l1
CONFIGS="-DVX_CFG_NUM_THREADS=32" ./ci/blackbox.sh --driver=simx --app=unified_l1 --cores=1 --warps=64 --threads=32

# cache-only and DXA/TCU still pass under SOCKET_SIZE=1
./ci/blackbox.sh --driver=simx --app=vecadd --threads=32
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=dxa_copy
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_wg --threads=32
```

Inspect `--perf=2` counters to confirm scratchpad reads/writes and cache hits/misses both move.
Keep each invocation under the 120 s cap.

## 8. Risks

- **Shared scratchpad port contention.** LSU(shared), TCU tile-buffer, and DXA all target the
  scratchpad region through `UnifiedL1`; the arbiter must preserve the TCU/DXA deadlock contract
  (the read-during-write interlock flagged in
  [`../designs/tensor_core_wgmma_engine.md`](../designs/tensor_core_wgmma_engine.md) §7.1). This
  unified L1/SMEM port is also the **seed of the proxy distinction**: in
  [Task 4](proxy_model_simx_proposal.md) its arbiter additionally carries the generic↔async
  **proxy-ordering** contract (the LSU path is the generic proxy, the DXA path the async proxy) —
  layered on top of the bandwidth arbitration designed here. No proxy logic in Task 1; just be
  aware the port is where it will live.
- **`is_llc` / write-back semantics** of the cache region must be preserved when there is no L2
  (toy uses L3 as global L2).
- **Coalescer placement.** The dcache coalescer ([`core.cpp` MemCoalescer](../../sim/simx/core.cpp#L136))
  now feeds `UnifiedL1`'s cache region — confirm word/line-size assumptions still hold.

## 9. Proposed but not yet implemented (Task 1 scope edges)

- Dynamic per-launch partition (Milestone B) if 1b ships at config-time (Milestone A) first.
- Cache replacement/eviction nuances under a shrinking cache partition.
- Bank-conflict swizzling for the scratchpad region (tracked with the WGMMA SMEM swizzle item).
