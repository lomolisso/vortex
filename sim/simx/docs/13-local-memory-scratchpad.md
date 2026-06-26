# 13 — Local Memory & Scratchpad

**Scope:** the per-core scratchpad — `LocalMem` (the banked SRAM), the `LocalMemSwitch` that routes
shared-address traffic to it, its multiple client ports (LSU / TCU / DXA), and how per-CTA space is
allocated. Sources: [`../mem/local_mem.h`](../mem/local_mem.h),
[`../mem/local_mem_switch.h`](../mem/local_mem_switch.h), [`../core.cpp`](../core.cpp),
[`../cta_dispatcher.h`](../cta_dispatcher.h). Prereqs: [03](03-kernel-launch-and-dispatch.md),
[11](11-memory-hierarchy-overview.md).

---

## 1. `LocalMem` — the banked scratchpad

`LocalMem` ([`local_mem.h:21`](../mem/local_mem.h#L21)) is the per-core shared memory, shared by all
warps on the core. `Config` ([`local_mem.h:23`](../mem/local_mem.h#L23)):

| Field | Value |
|---|---|
| `capacity` | bytes = `1 << VX_CFG_LMEM_LOG_SIZE` |
| `line_size` | `LSU_WORD_SIZE` (= `XLENB`) |
| `num_reqs` | total client ports (§2) |
| `B` | log2 number of banks |
| `write_reponse` | emit a response for writes |

It exposes `Inputs`/`Outputs` channel arrays of `MemReq`/`MemRsp`
([`local_mem.h:44-45`](../mem/local_mem.h#L44)), and a `PerfStats{reads, writes, bank_stalls}`
([`local_mem.h:31`](../mem/local_mem.h#L31)). Internally (in `local_mem.cpp`) it is a
**`MemCrossBar`** that fans `num_reqs` inputs onto `2^B` banks, selecting the bank by hashing the
line address — concurrent accesses to the same bank serialize and count `bank_stalls`.

### 1.1 The functional backdoor — `read_word`

`read_word(local_addr)` ([`local_mem.h:57`](../mem/local_mem.h#L57)) is a **zero-latency,
side-effect-free** 4-byte read of the scratchpad's model state, used by the TCU's warp-level AGU
(`TCU_LD`) to get a loaded value during `execute()` without going through the async request channel
(doc 17). The *timing* path still uses the normal `Inputs`/`Outputs`; this is the functional-state
accessor only — a clean example of the functional-vs-timing split (doc 01 §5).

---

## 2. The client ports

`LocalMem` is multi-master. `Core::Impl` sizes `num_reqs` as
`LSU_NUM_REQS + VX_CFG_EXT_TCU_ENABLED + VX_CFG_EXT_DXA_ENABLED`
([`core.cpp:124`](../core.cpp#L124)) and binds:

| Port range | Client |
|---|---|
| `[0 : LSU_NUM_REQS)` | the **LSU** shared path (via `LocalMemSwitch` + adapter) |
| `[LSU_NUM_REQS]` | the **TCU** tile-buffer (if `VX_CFG_EXT_TCU_ENABLE`, [`core.cpp:250`](../core.cpp#L250)) |
| `[LSU_NUM_REQS + TCU]` | the **DXA** write port (if `VX_CFG_EXT_DXA_ENABLE`) |

The TCU and DXA bind their LMEM channels here; the DXA write port additionally carries the
completion `tx_callback` that releases barriers (doc 09 §5). This shared port is exactly what the
Toy-H100 `UnifiedL1`/proxy proposals build on.

---

## 3. `LocalMemSwitch` — the shared/global fork

Per LSU block there is a `LocalMemSwitch` ([`local_mem_switch.h:20`](../mem/local_mem_switch.h#L20))
created in `Core::Impl` ([`core.cpp:136`](../core.cpp#L136)). It takes the LSU's `LsuReq` on `ReqIn`
and, per lane, routes by `get_addr_type` (doc 11 §3):

- **Shared** lanes → `ReqOutLmem` → `LocalMem` (via an `LsuMemAdapter`),
- **Global** lanes → `ReqOutDC` → coalescer/dcache.

Both sub-requests can fire the same cycle if both have active lanes and downstream space. Responses
return on `RspInLmem`/`RspInDC` and merge back to `RspOut`. The switch carries a `delay_`
([`local_mem_switch.h:43`](../mem/local_mem_switch.h#L43)).

---

## 4. Per-CTA allocation

`LocalMem` is the storage; the `CtaDispatcher` (doc [03](03-kernel-launch-and-dispatch.md) §4.2) is
the *allocator*. It hands each CTA a `lmem_size`-byte slice from a ring buffer
(`lmem_base_`/`lmem_tail_`/`free_size_`), with the **first CTA of a cluster** reserving a contiguous
`cluster_dim × lmem_size` span (so co-resident CTAs of a cluster are laid out contiguously for
DSMEM/DXA-multicast addressing). The per-CTA base lands in `cta_csrs.lmem_addr`, readable by the
kernel via a CSR (doc 10 §3); `warp_done` frees the slice when a CTA drains.

---

## 5. Summary

`LocalMem` is a banked SRAM with `MemCrossBar` arbitration and a functional `read_word` backdoor;
`LocalMemSwitch` forks LSU traffic into it (shared) vs the cache (global); the TCU and DXA attach as
extra masters on dedicated ports; and the `CtaDispatcher` carves it into per-CTA slices with
cluster-contiguous placement.

**Next:** [14 — Virtual Memory & the MMU](14-virtual-memory-mmu.md).
