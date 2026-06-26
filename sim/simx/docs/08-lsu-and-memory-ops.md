# 08 — LSU & Memory Operations

**Scope:** the load-store unit — how a memory instruction becomes per-lane addresses, gets routed
to scratchpad or the cache hierarchy, tracks in-flight requests, serializes fences, and writes
responses back to registers. Source: [`../lsu_unit.h`](../lsu_unit.h) (logic in
[`../lsu_unit.cpp`](../lsu_unit.cpp)); it connects to the memory subsystem documented in docs
[11](11-memory-hierarchy-overview.md)–[15](15-memory-protocol-and-types.md). Prereqs:
[05](05-pipeline-dispatch-execution.md), [06](06-instruction-lifecycle.md).

---

## 1. Where the LSU sits

`LsuUnit : FuncUnit<VX_CFG_NUM_LSU_BLOCKS>` ([`lsu_unit.h:69`](../lsu_unit.h#L69)). Like every FU it
takes traces on `Inputs[b]` and returns them on `Outputs[b]`, but instead of computing a value
inline it **emits memory requests** and later **demuxes responses** back into `trace->dst_data`.
Its downstream is the per-core memory fabric: `Core::lmem_switch(b)` routes each access to
**LocalMem** (shared) or the **D-cache** (global) by address class, optionally through the
`MemCoalescer` (doc [11](11-memory-hierarchy-overview.md) §request path, doc
[13](13-local-memory-scratchpad.md)).

---

## 2. Per-block state ([`lsu_unit.h:108-129`](../lsu_unit.h#L108))

The LSU has `VX_CFG_NUM_LSU_BLOCKS` independent blocks, each a `lsu_state_t`:

| Member | Role |
|---|---|
| `req_queue` (`RingQueue`, `VX_CFG_LSUQ_IN_SIZE`) | inbound traces awaiting address-gen/issue |
| `pending_reqs` (`HashTable`, keyed by LSU-allocated tag) | in-flight requests: the originating `trace` + per-lane `(tid, addr, size)` needed to write the response back to the right registers. **Not** a miss table — the cache owns its MSHR (doc 12) |
| `fence` (`FenceController`) | fence serialization (§4) |
| `addr_list` / `remain_addrs` | the per-lane addresses being issued for the current trace |

A global `pending_loads_` ([`lsu_unit.h:130`](../lsu_unit.h#L130)) counts outstanding loads;
`drained()` ([`lsu_unit.h:75`](../lsu_unit.h#L75)) reports when all blocks are quiescent (used by
the run-loop/flush drain checks, doc 02 §3).

---

## 3. The per-cycle pipeline ([`lsu_unit.h:83-90`](../lsu_unit.h#L83))

`on_tick()` runs three handlers per block:

1. **`ingest_inputs(b)`** — drain `Inputs[b]` into `req_queue`.
2. **`process_request_step(b)`** — `compute_addrs(b, trace)` produces the per-lane address list;
   allocate a tag, record a `pending_req_t`, and issue `MemReq`s to the switch/cache. A trace with
   more active lanes than memory channels is issued over multiple beats (`remain_addrs`).
3. **`process_response_step(b)`** — match a returning `MemRsp` to its `pending_reqs` entry by tag,
   write the data into the right `(thread)` slot of `trace->dst_data` (with sign-extension /
   NaN-boxing / `dst_bytesel` formatting), and when the last lane returns, forward the trace to
   `Outputs[b]` (commit, doc 05 §4).

Because responses are matched by **tag**, out-of-order completion is fine — the LSU reassembles the
warp's result regardless of return order, and the scoreboard's commit-counting (doc 05 §3) ensures
the destination register isn't released early.

---

## 4. Fences — `FenceController` ([`lsu_unit.h:39-67`](../lsu_unit.h#L39))

A fence must not let later memory ops pass until earlier ones are visible. `FenceController`
models this as a lock: `engage(trace)` locks the block on the fence instruction; `try_release`
([`lsu_unit.h:51`](../lsu_unit.h#L51)) succeeds only when **the pending-requests table is empty**
*and* the output channel can accept the trace — i.e. all prior accesses in this block have drained.
While locked, `process_request_step` won't issue new memory ops. The released fence trace carries
`resume_warp` semantics (doc 05 §4.1) to re-wake the warp.

---

## 5. Packed loads — `LsuUopGen` ([`lsu_unit.h:25-36`](../lsu_unit.h#L25))

`PACKLB.F`/`PACKLH.F` (packed byte/half float loads) are **macro instructions** cracked into
micro-ops by `LsuUopGen` in the `Sequencer` (doc [04](04-pipeline-frontend.md) §4, doc
[06](06-instruction-lifecycle.md) §4): `uop_count(instr)` says how many, `get(macro_instr,
uop_index)` produces each `Instr`. Each micro-op is an ordinary single-lane load the LSU pipeline
above handles normally; they share the parent UUID.

---

## 6. Atomics (AMO)

Atomic ops (`LR`/`SC`, `AMOADD`/`AMOMIN`/…) route through the LSU as `LsuType` ops and are computed
against memory via the `amo_compute()` helper (in the shared AMO support code). The **write-through
invariant** from doc [02](02-architecture-hierarchy.md) §2.2 exists for them: when the atomic
extension is on, caches above the LLC must be write-through so an `SC` can't spuriously succeed, and
non-LLC AMOs take a probe-and-invalidate passthrough path tagged `AMO_PASSTHRU_TAG_FLAG`
([`processor.cpp:96-99`](../processor.cpp#L96)) so the response routes back without installing a
fill. See doc [12](12-cache-architecture.md) for the cache side.

---

## 7. Address routing recap

```
LsuUnit.process_request_step
   compute_addrs → per-lane (addr,size,tid)
   for each lane: get_addr_type(addr)         (doc 15)
        Shared → Core.lmem_switch(b) → LocalMem            (doc 13)
        Global → [MemCoalescer] → D-cache → L1 arb → L2…   (docs 11,12)
   issue MemReq (tag = LSU-allocated), record pending_req_t
... later ...
LsuUnit.process_response_step
   match MemRsp by tag → write trace->dst_data[thread] → forward to Outputs[b] when complete
```

The packet types on these wires (`MemReq`/`MemRsp`, `hart_id`, flags) are doc
[15](15-memory-protocol-and-types.md).

**Next:** [09 — Barriers & Synchronization](09-barriers-and-sync.md).
