# 04 — Pipeline Frontend (Fetch → Decode → Issue → Operands)

**Scope:** the per-core front of the pipeline — how a warp is selected, fetched, decoded,
expanded into micro-ops, has its hazards checked, and has its operands read before it reaches a
functional unit. Sources: [`../scheduler.h`](../scheduler.h), [`../decode.h`](../decode.h),
[`../sequencer.h`](../sequencer.h), [`../scoreboard.h`](../scoreboard.h),
[`../operands.h`](../operands.h), [`../opc_unit.h`](../opc_unit.h); stage wiring lives in the
`Core::Impl` constructor in [`../core.cpp`](../core.cpp). Prereqs:
[01](01-simulation-framework.md), [03](03-kernel-launch-and-dispatch.md).

The whole frontend is a chain of `SimObject`s linked by `SimChannel<instr_trace_t*>` — a warp's
instruction becomes one `instr_trace_t` (doc [06](06-instruction-lifecycle.md)) that flows
stage→stage. Each stage is one `on_tick()`; backpressure (doc 01) stalls the chain naturally.

```
Scheduler.schedule() ─trace─► Decoder ─► Sequencer ─► Scoreboard(check) ─► Operands(fetch) ─► Dispatcher ─► FUs
        ▲ warp select                 (µop expand)     (hazard)            (regfile read)      (doc 05)
```

---

## 1. The `Scheduler` and warp state

[`Scheduler`](../scheduler.h#L104) is the per-core warp-lifecycle owner. It holds the `warps_`
array of `warp_t` ([`scheduler.h:72`](../scheduler.h#L72)), each carrying:

- `PC`, `tmask` (active-lane mask), `fcsr`, `mscratch` (kernel-arg pointer, set at dispatch);
- the **IPDOM stack** `ipdom_stack` (§3);
- per-warp **machine trap CSRs** `mstatus/mtvec/mepc/mcause/mtval` ([`scheduler.h:87`](../scheduler.h#L87));
- the per-CTA CSR snapshot `cta_csrs` ([`scheduler.h:45`](../scheduler.h#L45), §2 of doc 03's
  record) — block/grid/thread indices, `lmem_addr`, `cluster_size` — read by CSR instructions.

> **Where are the registers?** Not in `warp_t`. The integer/float register files live in `OpcUnit`
> ([`scheduler.h:73`](../scheduler.h#L73) comment, §5). `warp_t` owns *control* state; `OpcUnit`
> owns *data* state.

### 1.1 Warp selection — `schedule()`

`schedule(warp_mask)` ([`scheduler.h:110`](../scheduler.h#L110)) picks a ready warp and produces
its next `instr_trace_t` (the fetch). Readiness is filtered by two masks tracked with a
**current/next split** for cycle-coherent updates: `active_warps_` and `stalled_warps_` vs
`stalled_warps_next_` ([`scheduler.h:161-163`](../scheduler.h#L161)) — `suspend()`/`resume()`
write the *next* mask, `schedule()` reads the *registered* one, so a resume this cycle is visible
next cycle (no mid-cycle race). New warps are admitted from the `CtaDispatcher`
(`activate_warp`, [`scheduler.h:153`](../scheduler.h#L153); doc 03).

`advance_pc(wid, inc)` ([`scheduler.h:114`](../scheduler.h#L114)) bumps the PC by 2 (compressed) or
4 (full-width). `suspend`/`resume` park and wake a warp (used by barriers, fences, traps).

### 1.2 Traps

Synchronous traps redirect a warp: `raise_trap(wid, cause, trap_pc)` snapshots `mepc`/`mcause`,
sets PC to `mtvec`; `mret` restores from `mepc`; `trigger_ecall`/`trigger_ebreak` are the entry
points ([`scheduler.h:131-135`](../scheduler.h#L131)). The functional side of trap handling is in
doc [10](10-functional-emulator.md).

---

## 2. Decode

[`Decoder`](../decode.h#L23) is **stateless**: `decode(code, uuid)` ([`decode.h:32`](../decode.h#L32))
turns a 32-bit instruction word into an `Instr`, allocating from a shared
`PoolAllocator<Instr,64>`. It auto-detects **RVC**: if the low two bits ≠ `0b11` the word is a
16-bit compressed instruction, so it calls `rvc_decompress` (in
[`../decompressor.cpp`](../decompressor.cpp)) and stamps `is_rvc` so the scheduler advances the PC
by 2. The decoded `Instr` sets the trace's `fu_type` and `op_type` (doc 05's routing key) plus its
`dst_reg`/`src_regs`.

---

## 3. Control-flow divergence — the IPDOM stack

Vortex is SIMT: a warp's `tmask` marks active lanes. A divergent branch is handled with an
**immediate-post-dominator (IPDOM) stack** of `ipdom_entry_t{orig_tmask, else_PC, fallthrough}`
([`scheduler.h:31`](../scheduler.h#L31)), one per warp:

- A **`SPLIT`** (taken by some lanes, not others) pushes the pre-branch `tmask` and the else-path
  PC, then narrows `tmask` to the taken lanes.
- A **`JOIN`** pops the stack, restoring the other path's lanes (or the full pre-branch mask at
  reconvergence).
- Uniform branches (all lanes agree) need no split; the stack depth is bounded by
  `ipdom_size()` ([`scheduler.h:139`](../scheduler.h#L139)), and overflow/mismatch raises a trap.

`SPLIT`/`JOIN` are executed by the SFU's control unit (`WctlUnit`, doc
[09](09-barriers-and-sync.md) / [07](07-functional-units.md)); the *state* they manipulate is the
warp's IPDOM stack here.

---

## 4. Micro-op expansion — the `Sequencer`

Between fetch/decode and issue sits the [`Sequencer`](../sequencer.h#L35), a **per-warp micro-op
expander**. Most instructions **pass through unchanged** (`get()` returns the input trace). Two
families expand into multiple micro-ops sharing the parent's UUID:

- **packed loads** via `LsuUopGen` ([`sequencer.h:74`](../sequencer.h#L74)) — `PACKLB.F`/`PACKLH.F`
  split into up to one micro-op per lane (doc 08);
- **tensor ops** via `TcuUopGen` ([`sequencer.h:76`](../sequencer.h#L76), gated by
  `VX_CFG_EXT_TCU_ENABLE`) — WMMA/WGMMA expand into a stream of FEDP micro-ops (doc 17).

`get(trace)` is **idempotent** (returns the same current micro-op until `advance()`); `advance()`
returns `true` when all micro-ops have been issued ([`sequencer.h:41-44`](../sequencer.h#L41)).
This is distinct from SIMD-split packets (doc 06 §4) — µop expansion makes *several traces*; SIMD
split makes *one trace* into several packets.

---

## 5. Operand collection — `Operands` + `OpcUnit`

[`Operands`](../operands.h#L24) is the per-issue-lane operand-collection stage. `fetch_operands`
([`operands.h:34`](../operands.h#L34)) reads the trace's source registers into
`trace->src_data`; `writeback` ([`operands.h:37`](../operands.h#L37)) applies `trace->dst_data`
back to the regfile at commit. It holds a set of `OpcUnit`s and a `TraceArbiter rsp_arb_`
([`operands.h:55-56`](../operands.h#L55)).

The register files themselves live in [`OpcUnit`](../opc_unit.h#L29): each `(issue_lane, opc_idx)`
partition owns `ireg_file`/`freg_file`, both `[reg][thread]` ([`opc_unit.h:64`](../opc_unit.h#L64)).
The **partition routing math** ([`opc_unit.h:24-28`](../opc_unit.h#L24)) is worth memorizing:

```
lane = wid % VX_CFG_ISSUE_WIDTH      // which Operands instance
wis  = wid / VX_CFG_ISSUE_WIDTH      // warp-in-slice index
opc  = wis % VX_CFG_NUM_OPCS         // which OpcUnit within the lane
slot = wis / VX_CFG_NUM_OPCS         // local slot inside this OpcUnit
```

`read_src`/`writeback` ([`opc_unit.h:39-45`](../opc_unit.h#L39)) operate per `wid`. **Register-file
bank stalls** are modeled here via `cur_trace_`/`release_cycle_` and reported through
`total_stalls()` ([`opc_unit.h:47`](../opc_unit.h#L47)) — which feeds `opds_stalls` in the core's
`PerfStats` (doc 19).

---

## 6. Hazards — the `Scoreboard`

[`Scoreboard`](../scoreboard.h#L24) tracks per-warp register reservations *between issue and
writeback*. At issue, `reserve(trace)` ([`scoreboard.h:41`](../scoreboard.h#L41)) marks the
destination register in-use; `in_use(trace)` ([`scoreboard.h:37`](../scoreboard.h#L37)) reports
whether a candidate instruction reads or writes a still-pending register, so the scheduler won't
issue it (a RAW/WAW stall → `scrb_stalls` in `PerfStats`). The register key folds in `wid`
([`scoreboard.h:57`](../scoreboard.h#L57)). Release is deferred until all SIMD packets commit —
see `commit_packet` and doc [05](05-pipeline-dispatch-execution.md) §3.

---

## 7. Putting it together

One frontend pass: `schedule()` selects warp *w* and emits its trace → `Decoder` fills
`op_type`/regs → `Sequencer` yields the current µop → `Scoreboard.in_use` gates it → on pass,
`reserve()` + `Operands.fetch_operands` populate `src_data` → the trace flows to the `Dispatcher`
(doc 05). Every arrow is a `SimChannel`; a full downstream channel back-pressures the scheduler to
idle (`sched_idle` in `PerfStats`).

**Next:** [05 — Pipeline Dispatch & Execution](05-pipeline-dispatch-execution.md).
