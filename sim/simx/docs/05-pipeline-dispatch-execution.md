# 05 — Pipeline Dispatch & Execution

**Scope:** the back half of the pipeline — how a ready trace is routed to the right functional
unit, how a warp instruction is split across execution lanes, and how results commit (writeback +
scoreboard release). Sources: [`../dispatcher.h`](../dispatcher.h), [`../func_unit.h`](../func_unit.h),
[`../scoreboard.h`](../scoreboard.h), [`../operands.h`](../operands.h), [`../types.h`](../types.h);
wiring in [`../core.cpp`](../core.cpp). Prereqs: [04](04-pipeline-frontend.md),
[06](06-instruction-lifecycle.md).

---

## 1. The `Dispatcher` — issue lanes → execution lanes

[`Dispatcher`](../dispatcher.h#L24) sits between operand collection and the functional units. Its
job is **width adaptation + SIMD splitting**:

- It has `Inputs`/`Outputs` arrays of `SimChannel<instr_trace_t*>` ([`dispatcher.h:26-27`](../dispatcher.h#L26)).
- It aggregates `VX_CFG_ISSUE_WIDTH` **issue** lanes onto `NUM_BLOCKS` **execution** lanes — the
  two widths differ on purpose ([`func_unit.h:34-37`](../func_unit.h#L34)).
- It manages packetization: `block_size_`, `num_lanes_`, `num_blocks_`, `num_packets_`,
  `batch_idx_`, and `block_pids_` ([`dispatcher.h:39-45`](../dispatcher.h#L39)).

### 1.1 SIMD splitting (`num_pkts`, `pid`, `sop`/`eop`)

When a warp has more threads than a functional unit has physical lanes, the instruction is split
into **packets**. The dispatcher stamps each emitted trace with `pid` (packet id), `sop`/`eop`
(start/end of packet stream), and sets `num_pkts` = the total packet count
([`instr_trace.h:57-64`](../instr_trace.h#L57)). This is the key to correct commit (§3): commit
counts *completions*, not dispatch order, because cache responses can return out of order.

---

## 2. `op_type` → `FUType` routing

Routing is driven by the trace's `op_type`, a **variant** set at decode. The dispatcher (and each
FU) uses `std::visit`/`std::get` over that variant to select and extract the right arguments:

- `fu_type` (an `FUType` enum: ALU, FPU, LSU, SFU, TCU) selects which functional unit the trace
  is dispatched to;
- inside the FU, `std::get<IntrAluArgs>(op_type)` (etc.) extracts the op-specific fields.

Because each `op_type` alternative is a distinct type, the compiler generates a separate code path
per instruction class — there is no giant switch. The enum/variant definitions are in
[`../types.h`](../types.h) and [`../instr.h`](../instr.h) (doc 15 / 06).

---

## 3. Functional units — the `FuncUnit` base

Every execution unit derives from [`FuncUnit<NUM_BLOCKS>`](../func_unit.h#L38), which provides
`NUM_BLOCKS` input/output channel pairs (`Inputs`/`Outputs`, [`func_unit.h:43`](../func_unit.h#L43))
and a pure-virtual `on_tick()` each unit implements. A type-erased `FuncUnitBase`
([`func_unit.h:26`](../func_unit.h#L26)) lets `Core` keep heterogeneous units (ALU, FPU, LSU, SFU,
TCU — all different `NUM_BLOCKS`) in one container and bind their channels uniformly.

The **functional-vs-timing split** (doc 01 §5) lives in every FU's `on_tick`: it computes the
result *immediately* (functional) into `trace->dst_data`, then `Outputs[b].send(trace,
latency)` schedules the response after the modeled latency. The individual units are documented in
docs [07](07-functional-units.md) (ALU/FPU/SFU) and [08](08-lsu-and-memory-ops.md) (LSU).

---

## 4. Commit — writeback + scoreboard release

A trace leaving an FU's `Outputs` fans back in to the per-issue-slot commit logic (by
`trace->wid`, since the dispatcher remapped issue→execution lanes upstream). Commit does two
things:

1. **Writeback** — `Operands::writeback(trace)` ([`operands.h:37`](../operands.h#L37)) →
   `OpcUnit::writeback` applies `trace->dst_data` to the destination register (functional).
2. **Scoreboard release (deferred)** — `Scoreboard::commit_packet(trace)`
   ([`scoreboard.h:51`](../scoreboard.h#L51)) increments a per-instruction commit counter and
   returns `true` only when **all `num_pkts` packets have committed**; the caller then calls
   `release(trace)` ([`scoreboard.h:43`](../scoreboard.h#L43)) to clear the destination
   reservation. For a non-split trace (`num_pkts==1`) this is true on the first call.

> **Why count commits, not `eop`?** `eop` marks the last *dispatched* packet, but cache responses
> can complete out of order ([`scoreboard.h:46-50`](../scoreboard.h#L46)). Counting commits means
> an early-returning cache hit can't release the destination before a still-pending miss writes
> its lanes — preventing a younger instruction from reading a half-written register.

### 4.1 Fetch-stall resolution (`fetch_stall` / `resume_warp`)

Some instructions stall the warp at the front (e.g. a fence that must drain, or a long-latency
control op). The FU sets `trace->resume_warp` when the stall resolves
([`instr_trace.h:68-70`](../instr_trace.h#L68)); the warp is released when that trace drains from
the FU output at commit fan-in. This is how the LSU fence (doc 08) and barriers (doc 09) re-wake a
parked warp.

---

## 5. The execution half in one breath

`Dispatcher` takes a ready trace, picks `fu_type`, splits it into `num_pkts` packets across
`NUM_BLOCKS` lanes, and `send`s each to the chosen `FuncUnit`. The FU computes `dst_data`
immediately and sends the response after `latency` cycles. At commit, `writeback` applies the
result and `commit_packet` counts down to a deferred `release`, so the scoreboard reservation
lifts only when the whole warp instruction is truly done — even under out-of-order memory
responses.

**Next:** [06 — Instruction Lifecycle](06-instruction-lifecycle.md).
