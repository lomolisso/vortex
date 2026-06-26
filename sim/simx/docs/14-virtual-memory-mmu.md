# 14 — Virtual Memory & the MMU

**Scope:** the optional address-translation path — the per-core `Mmu`, its TLB, the page-table-walk
FSM, and how faults surface as traps. Gated by `VX_CFG_VM_ENABLE`. Sources:
[`../mem/mmu.h`](../mem/mmu.h), [`../mem/mmu_tlb.h`](../mem/mmu_tlb.h),
[`../mem/mmu.cpp`](../mem/mmu.cpp). Prereqs: [10](10-functional-emulator.md) (traps),
[11](11-memory-hierarchy-overview.md).

---

## 1. Where the MMU sits

When `VX_CFG_VM_ENABLE` is defined, each core inserts an `Mmu` ([`mmu.h:38`](../mem/mmu.h#L38)) on
its **dcache and icache request paths**, between the upstream (LSU/coalescer/adapter or fetch) and
the downstream cache port. Per port `p`:

```
ReqIn[p]  ─► [ bypass | TLB lookup | PTW miss queue ] ─► ReqOut[p]   (to cache)
RspOut[p] ◄─ [ filter PTW responses, forward the rest ] ◄─ RspIn[p]  (from cache)
```

So the LSU/fetch emit **virtual** addresses; the MMU translates and forwards **physical** addresses
to the cache. The MMU is a `SimObject` ([`mmu.h:38`](../mem/mmu.h#L38)) with `ReqIn`/`RspOut`
(upstream) and `ReqOut`/`RspIn` (downstream) channel arrays
([`mmu.h:48-53`](../mem/mmu.h#L48)). When VM is disabled the whole class compiles out (`#ifdef`,
[`mmu.h:13`](../mem/mmu.h#L13)) and the core wires LSU→cache directly (doc 02 / core.cpp).

---

## 2. SATP and translation gating

`set_satp(satp)` ([`mmu.h:63`](../mem/mmu.h#L63)) is invoked from `CsrUnit` on a `csrw satp` (doc
10 §3, via `Core::set_satp` which fans to both per-core MMUs). It updates the translation root and
**flushes the TLB** (sfence.vma semantics). `needs_translation(addr)`
([`mmu.h:78`](../mem/mmu.h#L78)) decides whether an address is translated (SATP BARE mode → pass
through; SV mode → translate).

---

## 3. The TLB

`Tlb tlb_` ([`mmu.h:91`](../mem/mmu.h#L91), defined in [`../mem/mmu_tlb.h`](../mem/mmu_tlb.h)) caches
virtual→physical mappings. A hit translates inline; a miss triggers the PTW (§4). Perf counters are
exposed directly: `tlb_reads/hits/misses/evictions` ([`mmu.h:66-69`](../mem/mmu.h#L66)).

---

## 4. The page-table walk (PTW) FSM

On a TLB miss the MMU runs a **single in-flight** page-table walk via a level-counter FSM
([`mmu.h:96-101`](../mem/mmu.h#L96)):

```
PTW_IDLE → PTW_REQ (emit this level's PTE fetch) → PTW_WAIT (await PTE) → PTW_FILL (install TLB + replay)
```

- The walk is `VX_VM_PT_LEVEL` deep — **Sv32 = 2 levels, Sv39 = 3** — driven by `ptw_level_`
  counting down rather than per-level states ([`mmu.h:93-95`](../mem/mmu.h#L93)).
- A PTE address is computed as `base_ppn * VX_VM_PT_SIZE + vpn_idx * VX_VM_PTE_SIZE`
  ([`mmu.h:81-83`](../mem/mmu.h#L81)).
- **PTE fetches go through the cache hierarchy** — they are ordinary `MemReq`s on the MMU's *own*
  downstream `ReqOut`, distinguished from upstream traffic by a marker bit on the tag,
  `PTW_TAG_MARKER = 1u<<31` ([`mmu.h:45`](../mem/mmu.h#L45)). `on_ptw_response` consumes any response
  carrying that marker so the upstream caller never sees PTW traffic ([`mmu.h:86`](../mem/mmu.h#L86)).
- Superpages are supported via `ptw_leaf_level_` (0 = 4 KB, 1 = mega, 2 = giga,
  [`mmu.h:111`](../mem/mmu.h#L111)).

On completion the mapping fills the TLB and the original request (`ptw_orig_req_`,
[`mmu.h:112`](../mem/mmu.h#L112)) replays with its physical address. PTW perf: `ptw_walks()` /
`ptw_latency()` ([`mmu.h:70-71`](../mem/mmu.h#L70)).

---

## 5. Faults

An invalid PTE or an access-permission violation is a **page fault**: the MMU sets `mepc`/`mcause`/
`mtval` and raises an exception through the `Scheduler` trap path (doc [10](10-functional-emulator.md)
§4) — the same `raise_trap` mechanism ECALL/EBREAK use. The faulting warp redirects to `mtvec`.

---

## 6. Summary

With VM on, each core gets a per-port `Mmu` doing TLB-cached SV32/SV39 translation; misses run a
single level-counted PTW whose PTE fetches reuse the cache hierarchy (tagged so they're invisible
upstream); faults funnel into the standard trap path. With VM off, the class compiles away and
addresses are physical throughout.

**Next:** [15 — Memory Protocol & Types](15-memory-protocol-and-types.md).
