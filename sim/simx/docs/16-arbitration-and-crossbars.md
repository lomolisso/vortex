# 16 — Arbitration & Crossbars

**Scope:** the reusable arbiter/crossbar machinery that connects many producers to fewer consumers
all over the simulator — cache ports, LMEM banks, commit fan-in, the L1/L2 merges. Source:
[`../types.h`](../types.h) (it's all there). Prereqs: [01](01-simulation-framework.md),
[15](15-memory-protocol-and-types.md).

---

## 1. Why arbiters are everywhere

Hardware constantly funnels N requesters onto M < N resources: two L1 caches onto one L2 port,
many LSU lanes onto a few cache banks, several FU outputs onto one commit slot. SimX models this
with a small family of arbiter/crossbar `SimObject`s parameterized by an **arbitration policy**.

## 2. Policies — `ArbiterType` ([`types.h:842`](../types.h#L842))

```cpp
enum class ArbiterType { Priority, RoundRobin, Matrix, GTO };
```

Each maps to a concrete `IArbiterImpl` ([`types.h:860`](../types.h#L860)) whose core method is
`grant(requests) → winner index` (plus an optional `grant(requests, suppress)` that lets a mask veto
candidates):

| Policy | Class | Behavior |
|---|---|---|
| `Priority` | `PriorityArbiter` ([`:873`](../types.h#L873)) | lowest index wins (fixed priority) |
| `RoundRobin` | `RoundRobinArbiter` ([`:896`](../types.h#L896)) | rotating fairness |
| `Matrix` | `MatrixArbiter` ([`:924`](../types.h#L924)) | least-recently-granted (matrix of pairwise order) |
| `GTO` | `GTOArbiter` ([`:973`](../types.h#L973)) | greedy-then-oldest |

The `Arbiter` wrapper ([`types.h:1034`](../types.h#L1034)) is the factory: construct with an
`ArbiterType` and it instantiates the right impl. The `suppress` mask
([`types.h:865`](../types.h#L865)) is how priority overrides are layered on (e.g. forcing one client
to lose this cycle).

## 3. The channel-level arbiters

Three `SimObject` templates wrap policies into actual channel routers:

- **`TxArbiter<T>`** ([`types.h:1392`](../types.h#L1392)) — **N→1, single direction**. Merges N
  input channels of `T` onto one output, choosing a winner per cycle by policy. The commit fan-in
  uses `TraceArbiter = TxArbiter<instr_trace_t*>` ([`instr_trace.h:167`](../instr_trace.h#L167)) to
  merge FU outputs back to a warp's writeback lane.
- **`TxRxArbiter<Req,Rsp>`** ([`types.h:1626`](../types.h#L1626)) — **bidirectional N→1**. Arbitrates
  requests N→1 *and* routes the matching responses 1→N. Used wherever a request needs its response
  steered back to the originating input.
- **`TxRxCrossBar<Req,Rsp>`** ([`types.h:1738`](../types.h#L1738)) — **M→N switching**. A full
  crossbar from M input ports to N output ports (e.g. cores↔banks), with response routing.

### 3.1 Tag mangling

To route a response back to the right input, the arbiters **mangle the tag**: they prepend the
input index onto the request's `tag` on the way down and strip it on the way back, so the response
demuxes to the originating channel. This is why `MemReq::tag` is "application-specific" while
`hart_id` is the hardware id (doc 15 §4) — the arbiters own the tag's high bits in transit.

## 4. The memory specializations — `MemArbiter` / `MemCrossBar`

The memory fabric uses `MemReq`/`MemRsp` specializations of the above:

- **`MemArbiter`** — e.g. the core L1 arbiter merging icache+dcache onto the L2 port
  (`MemArbiter::Create(name, ArbiterType::RoundRobin, …)` in [`../core.cpp`](../core.cpp), with
  the icache effectively prioritized), the cluster L2 input arbiter, and the DXA `gmem_arb_`
  ([`dxa/dxa_core.h:59`](../dxa/dxa_core.h#L59)).
- **`MemCrossBar`** — e.g. `LocalMem`'s internal `num_reqs`→`banks` switch with a **bank-select hash
  function** (`MemCrossBar::Create(name, ArbiterType::Priority, num_reqs, num_banks, hash_fn)` in
  [`../mem/local_mem.cpp`](../mem/local_mem.cpp)); the bank index is `(addr >> lg2_line) &
  (num_banks-1)`.

## 5. Reading wiring code

When you see `X::Create(name, ArbiterType::RoundRobin, N_in, N_out)` followed by `bind()`s into its
`ReqIn`/`RspOut`, you're looking at an N_in→N_out arbitrated junction with round-robin fairness and
automatic response routing. That single pattern — policy + tag-mangling + bidirectional routing —
recurs at every level of the hierarchy (docs 02, 11–13).

**Next:** [17 — Extension Engines](17-extension-engines.md).
