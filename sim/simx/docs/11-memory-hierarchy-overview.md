# 11 — Memory Hierarchy Overview

**Scope:** the shape of the memory system — what levels exist, where they live, how a request
travels from a thread to DRAM and back, and how the address space is classified. This is the map;
docs [12](12-cache-architecture.md)–[15](15-memory-protocol-and-types.md) are the territory.
Sources: [`../mem/`](../mem), [`../processor.cpp`](../processor.cpp), [`../types.h`](../types.h).
Prereqs: [02](02-architecture-hierarchy.md), [08](08-lsu-and-memory-ops.md).

---

## 1. The levels

| Level | Class | Where instantiated | Notes |
|---|---|---|---|
| **LMEM** (scratchpad) | `LocalMem` | per **core** | banked SRAM, shared by a core's warps (doc 13) |
| **L1 D-cache / I-cache** | `CacheCluster` of `Cache` | per **socket** | private to the socket's cores (doc 12) |
| **L2** | `Cache` | per **cluster** | shared across the cluster's sockets |
| **L3 (LLC)** | `Cache` | per **processor** | LLC when enabled, else a bypass arbiter ([`processor.cpp:64`](../processor.cpp#L64)) |
| **DRAM** | `Memory` (+ backing `RAM`) | per **processor** | the timing model + functional bytes (doc 14 of this set is MMU; DRAM is §4 here) |

The cache levels are all the *same* `Cache` class (doc 12) parameterized differently; "L1/L2/L3"
is a wiring distinction, not a type distinction. Coalescers, switches, and (optionally) the MMU sit
on the per-core request path.

---

## 2. The global-memory request path

```
LSU (doc 08)
  └─ Core.lmem_switch(b)  ── Shared ─► LocalMem (scratchpad)                          (doc 13)
                          └─ Global ─► [MemCoalescer] ─► [Mmu] ─► dcache (CacheCluster) (docs 12,14)
                                                                     │
                              socket L1 arbiter (icache prioritized) ◄┘                 (doc 02 §4.2)
                                     │
                                     ▼
                              L2 (cluster) ─► L3 (processor, LLC) ─► Memory/DRAM
                                                                         │
                              MemRsp travels the reverse path back to the LSU, matched by tag
```

Instruction fetch takes the analogous path through the **I-cache** instead of the D-cache. Each
hop is a `SimChannel<MemReq>` / `SimChannel<MemRsp>` pair, so a full downstream queue back-pressures
the whole chain (doc 01 §3.3).

### 2.1 The per-core front-end helpers

- **`MemCoalescer`** ([`../mem/mem_coalescer.h`](../mem/mem_coalescer.h)) — merges the per-lane LSU
  accesses of one warp into fewer cache-line-sized requests. It has `ReqIn`/`ReqOut` `LsuReq`
  channels, an `output_ratio_`, and a `pending_rd_reqs_` table keyed by tag to scatter the
  coalesced response back to lanes (doc 12 of the cache uses its own MSHR — different mechanism).
- **`LocalMemSwitch`** ([`../mem/local_mem_switch.h`](../mem/local_mem_switch.h)) — the shared/global
  fork: one `ReqIn` splits to `ReqOutLmem` (shared) or `ReqOutDC` (global) by `get_addr_type` (§3,
  doc 13).
- **`Mmu`** ([`../mem/mmu.h`](../mem/mmu.h)) — optional address translation, only when
  `VX_CFG_VM_ENABLE` (doc 14).
- **`LsuMemAdapter`** ([`../mem/lsu_mem_adapter.h`](../mem/lsu_mem_adapter.h)) — packs the LSU's
  per-lane `LsuReq` into the cache's per-channel `MemReq` and back.

---

## 3. Address classification — `get_addr_type`

Every address is classified by `get_addr_type(addr)` ([`types.h:818`](../types.h#L818)) into one of
`AddrType{Global, Shared, IO}` ([`types.h:812`](../types.h#L812)):

- **Shared** if it falls in `[VX_MEM_LMEM_BASE_ADDR, +2^VX_CFG_LMEM_LOG_SIZE)` → LMEM scratchpad;
- **IO** if in the IO range → straight to `Memory` (uncacheable);
- **Global** otherwise → through the cache hierarchy.

`MemReq::addr_type()` ([`types.h:1212`](../types.h#L1212)) is the convenience wrapper; the LSU sets
`MemFlags.local` to mirror Shared (doc 15). This single function is what makes the `LocalMemSwitch`
fork deterministic and is the seam the Toy-H100 socket-SMEM proposal reuses to route the shared
path to the relocated socket SMEM (Task 1 keeps SMEM separate from the L1D — no fused array).

---

## 4. DRAM — the `Memory` model

`Memory` ([`../mem/memory.h`](../mem/memory.h)) is the bottom of the hierarchy: `mem_req_in` /
`mem_rsp_out` ports, a banked DRAM timing model (`Config{num_banks, num_ports, block_size,
clock_ratio}`), and the **functional backing store** attached via `attach_ram(RAM*)`
([`memory.h:49`](../mem/memory.h#L49)) — this is where global data physically lives. It also
exposes `read_word(byte_addr)` as a backdoor functional read ([`memory.h:54`](../mem/memory.h#L54))
and a `set_pre_send_hook` ([`memory.h:65`](../mem/memory.h#L65)) used to mirror traffic to SST/gem5
(doc [21](21-host-runtime-and-external-sim.md)) without changing the local data path.

---

## 5. Where to go next

- The cache itself (banks, MSHR, replacement, flush, AMO passthrough): doc [12](12-cache-architecture.md).
- LMEM scratchpad (banking, per-CTA allocation, the TCU/DXA ports): doc [13](13-local-memory-scratchpad.md).
- Virtual memory / the MMU page-table walk: doc [14](14-virtual-memory-mmu.md).
- The packet types on every wire (`MemReq`/`MemRsp`/`MemFlags`/`hart_id`): doc [15](15-memory-protocol-and-types.md).
- The arbiters/crossbars that connect all of it: doc [16](16-arbitration-and-crossbars.md).

**Next:** [12 — Cache Architecture](12-cache-architecture.md).
