# 09 — Barriers & Synchronization

**Scope:** how warps synchronize — CTA-local barriers, cross-core (global) barriers, cross-CTA
(group) barriers, and the **asynchronous transaction** mechanism (`expect_tx` + events) that lets
a DXA copy wake a waiting warp. Sources: [`../barrier_unit.h`](../barrier_unit.h),
[`../barrier_unit.cpp`](../barrier_unit.cpp), [`../wctl_unit.h`](../wctl_unit.h),
[`../types.h`](../types.h), [`../cluster.cpp`](../cluster.cpp). Prereqs:
[04](04-pipeline-frontend.md) (scheduler suspend/resume), [03](03-kernel-launch-and-dispatch.md)
(CTAs/clusters).

---

## 1. Where barriers live

`BarrierUnit` ([`barrier_unit.h:27`](../barrier_unit.h#L27)) is a **child `SimObject` of the
`Scheduler`** (doc 04), one per core. It owns the per-warp barrier state and resumes parked warps
through the parent scheduler; for cross-core barriers it cooperates with the `Socket` (doc 02).
Kernels reach it indirectly: a `BAR` instruction is a warp-control op executed by `WctlUnit`
([`wctl_unit.h:26`](../wctl_unit.h#L26)), which calls into `BarrierUnit`.

### 1.1 State — `warp_barrier_t` ([`types.h:90`](../types.h#L90))

```cpp
struct warp_barrier_t {
  WarpMask wait_mask;     // warps actually suspended on this barrier
  WarpMask arrival_mask;  // global-barrier per-core arrival tracker
  uint32_t count;         // arrivals still needed
  uint32_t events;        // pending async transaction events (expect_tx)
  uint32_t phase;         // generation counter (flips on completion)
};
```

There are `NUM_WARPS × NUM_BARRIERS` of these. A kernel barrier handle is **packed** —
`(cta_no | bar_no<<8)` with bit 31 = global — and decoded to a flat index by `bar_decode_id`
([`types.h:67`](../types.h#L67)), which preserves the global bit.

> **This is the simulator's proto-mbarrier:** `count` + `events` are *two* completion counters and
> `phase` is the flip bit. The Toy-H100 proxy proposal (in `docs/proposals/`) formalizes this into
> a byte-granular mbarrier; everything here is the as-built mechanism it builds on.

---

## 2. The barrier API ([`barrier_unit.h:32-37`](../barrier_unit.h#L32))

| Method | Effect |
|---|---|
| `arrive(bar_id, count, wid, is_sync_bar)` | warp `wid` arrives; first arrival suspends it; when arrivals reach `count`, completes |
| `wait(bar_id, phase, wid)` | block until the barrier's `phase` advances past the supplied phase |
| `event_attach(bar_id, count=1)` | pre-register `count` pending **transaction** events (the `expect_tx` path) |
| `event_release(bar_id)` | one transaction landed; decrement `events` |
| `global_resume(bar_id)` | cross-core cascade resume (§4) |
| `get_phase(bar_id)` | current generation |

### 2.1 The completion rule

`event_release` ([`barrier_unit.cpp:135`](../barrier_unit.cpp#L135)) decrements `events`
([`:141`](../barrier_unit.cpp#L141)); the barrier completes — resuming `wait_mask` warps and doing
`++barrier.phase` ([`:163`](../barrier_unit.cpp#L163)) — **only when `count==0` AND `events==0`**.
So a CTA barrier with an outstanding async copy (events>0) will *not* release until the copy lands,
even after all warps have arrived. `arrive` flips the phase the same way once the last arrival
makes `count==0` ([`barrier_unit.cpp:85`](../barrier_unit.cpp#L85)). Parked warps are woken via
`Scheduler::resume` (doc 04 §1.1).

---

## 3. The three barrier flavors (kernel side)

From [`../../../sw/kernel/include/vx_barrier.h`](../../../sw/kernel/include/vx_barrier.h):

- **`barrier`** — CTA-local: `bar_id = (id<<8) | local_group_id`. Warps of one CTA rendezvous.
- **`gbarrier`** — global/cross-core: `bar_id = (id<<8) | 0x80000000`. The global bit routes it
  through the cross-core cascade (§4).
- **`group_barrier`** — cross-CTA within a cluster: `bar_id = (id<<8)` (wid=0), the *same slot* seen
  from every CTA, so co-resident CTAs of a cluster can rendezvous (used by DSMEM/DXA-multicast
  rendezvous, doc 17).

All expose `arrive()/wait(phase)/arrive_and_wait()` and (for the async forms) `expect_tx(count)`.

---

## 4. Global (cross-core) barriers

A global barrier can't complete on one core alone. `arrive` on a global `bar_id` calls
`core_->socket()->global_barrier_arrive(bar_id, count, core_id)`
([`barrier_unit.cpp:64`](../barrier_unit.cpp#L64)); the `Socket`/`Cluster` accumulate per-core
arrivals (`arrival_mask`), and when all participating cores have arrived, the cascade calls
`global_resume(bar_id)` back on each core ([`barrier_unit.cpp:113`](../barrier_unit.cpp#L113),
`++phase` at [`:126`](../barrier_unit.cpp#L126)), waking everyone. `Socket::global_barrier_arrive`
/ `global_barrier_resume` (doc 02 §4.2) are the cross-core hops.

---

## 5. Asynchronous completion — how DXA wakes a barrier

This is the `expect_tx`/event path end-to-end and the key to async copies:

```
kernel:  bar.expect_tx(N)                 → WctlUnit → BarrierUnit.event_attach(bar, N)   (events += N)
kernel:  vx_dxa_issue_…(…, bar.id(), …)   → SFU/DXA performs the copy (doc 17)
kernel:  bar.arrive_and_wait()            → warp suspends (count path)
DXA done: last LMEM write carries flags.dxa_notify_done + dxa_notify_bar_id
cluster: tx_callback on the DXA→LMEM channel fires:                                        (cluster.cpp:160)
            if (req.is_write() && req.flags.dxa_notify_done)
              core->barrier_event_release(bar_decode_id(req.flags.dxa_notify_bar_id, …))   (cluster.cpp:165)
BarrierUnit.event_release → events-- → when count==0 && events==0 → phase++ → resume warps
```

The `tx_callback` is registered on the per-core DXA→LocalMem `SimChannel`
([`cluster.cpp:147-165`](../cluster.cpp#L147)) — a textbook use of the channel sideband hook (doc
01 §3.5): the barrier is released *the cycle the write lands in LMEM*, with **per-destination**
granularity (multicast offsets the `bar_id` per CTA so each receiver's barrier fires on its own
slice — doc 17).

**Next:** [10 — The Functional Model](10-functional-emulator.md).
