# 06 — The Instruction Lifecycle (`instr_trace_t`)

**Scope:** the single most important data structure in SimX — `instr_trace_t`, the object that
*is* an in-flight instruction as it threads through every pipeline stage. Understand this and the
pipeline docs (04–05) and the unit docs (07–08) all click into place. Source:
[`../instr_trace.h`](../instr_trace.h). Prereqs: [01](01-simulation-framework.md),
[04](04-pipeline-frontend.md), [05](05-pipeline-dispatch-execution.md).

---

## 1. What it is

Every pipeline stage is a `SimObject` linked by `SimChannel<instr_trace_t*>` (note: a **pointer**).
One instruction = one heap-pooled `instr_trace_t` that is *created at decode* and *destroyed at
commit*, traveling stage→stage by pointer the whole way. It is an **immutable-ish snapshot**: there
is no instruction state machine that mutates as it flows — stages fill in fields they own and pass
it along. This is the concrete embodiment of the functional-vs-timing split (doc 01 §5).

It is **pool-allocated** from `Core::trace_pool()` — a `PoolAllocator<instr_trace_t, 64>`
([`core.h:152`](../core.h#L152)) — so a few thousand instructions/cycle cost no malloc.

---

## 2. The fields ([`instr_trace.h:26-98`](../instr_trace.h#L26))

| Field | Set by | Meaning |
|---|---|---|
| `uuid` (const) | decode | unique id; ties the trace through every channel for debug correlation (doc 20) |
| `cid`, `wid`, `cta_id` | schedule | core / warp / CTA identity |
| `tmask` | schedule | active-lane mask (SIMT) |
| `PC`, `code` | fetch/decode | program counter and raw instruction word |
| `wb` | decode | does it write back a register? |
| `dst_reg` / `src_regs` | decode | `RegOpd` destination + up to `NUM_SRC_REGS` sources |
| `fu_type` | decode | which functional unit (`FUType`: ALU/FPU/LSU/SFU/TCU) — doc 05 routing |
| `op_type` | decode | the op variant (`std::visit`ed in the FU) |
| `src_data[src][thread]` | operand collect | per-thread source values read from the regfile |
| `dst_data[thread]` | FU execute | per-thread result (computed functionally, 0-cycle) |
| `dst_bytesel` | FU/LSU | byte write-enable for the destination |
| `instr_ptr` | decode | the decoded `Instr` |
| `pid`, `sop`, `eop` | dispatch | SIMD packet id + start/end-of-stream flags |
| `num_pkts` | dispatch | total SIMD-split packets; commit defers release until all complete (doc 05 §3) |
| `fetch_stall` | decode/FU | warp parked at the front until this resolves |
| `resume_warp` | FU | set when a `fetch_stall` op resolves → warp released at commit fan-in |
| `issue_time` | ctor | `SimPlatform::cycles()` at creation — for latency accounting |

The arrays are sized to `VX_CFG_NUM_THREADS` at construction
([`instr_trace.h:87-88`](../instr_trace.h#L87)): `src_data` is `[NUM_SRC_REGS][NUM_THREADS]`,
`dst_data` is `[NUM_THREADS]`. There is also a copy constructor
([`instr_trace.h:100`](../instr_trace.h#L100)) used when a trace is duplicated (e.g. SIMD split or
µop expansion); note `log_once_` is reset on copy.

---

## 3. The canonical lifecycle

```
1. DECODE       trace = trace_pool.allocate(uuid);  Decoder fills code/PC/fu_type/op_type/dst_reg/src_regs
2. SCHEDULE     Scheduler.schedule() latches cid/wid/cta_id/tmask/PC; Scoreboard.reserve(dst_reg)
3. SEQUENCE     Sequencer.get() yields this (or a µop derived from it) — doc 04 §4
4. OPERANDS     Operands.fetch_operands() → OpcUnit.read_src() fills src_data[src][thread]
5. DISPATCH     Dispatcher routes by fu_type; sets pid/sop/eop/num_pkts (SIMD split) — doc 05
6. EXECUTE      FU.on_tick(): computes dst_data[thread] IMMEDIATELY (functional);
                Outputs[b].send(trace, latency_of(...))   ← the only place "time" is added
7. COMMIT       (after `latency` cycles) Operands.writeback() applies dst_data;
                Scoreboard.commit_packet() → release() when all num_pkts done; resume_warp handled
8. FREE         trace returns to the pool
```

The UUID (step 1) is invariant across all of this, which is what makes the `--debug` trace
correlatable (doc 20): you can `grep` one `#<uuid>` and watch it move stage to stage.

---

## 4. Two kinds of "splitting" — don't conflate them

- **Micro-op expansion (Sequencer, doc 04 §4):** one decoded instruction becomes **several traces**
  — e.g. a `PACKLB.F` packed load (`LsuUopGen`) or a `WGMMA` (`TcuUopGen`). They share the parent
  `uuid`; `Sequencer::advance()` walks them.
- **SIMD-split packets (Dispatcher, doc 05 §1):** **one trace** is issued as several **packets**
  across execution lanes because the warp has more threads than the FU has physical lanes.
  Tracked by `pid`/`sop`/`eop`/`num_pkts` on the *same* trace object; commit counts completions.

Both exist because real hardware does both (instruction cracking vs. SIMD lane batching); keeping
them separate is why the model stays correct under out-of-order memory responses.

---

## 5. Debugging hook

`operator<<` ([`instr_trace.h:135`](../instr_trace.h#L135)) prints
`cid/wid/cta_id/tmask/PC/wb/regs/fu/pid/sop/eop (#uuid)` — this is exactly the line you see in
`--debug` traces (doc 20). `log_once(enable)` ([`instr_trace.h:129`](../instr_trace.h#L129)) gates
a one-shot log so a stalled trace doesn't spam every cycle.

`TraceArbiter` ([`instr_trace.h:167`](../instr_trace.h#L167)) — a `TxArbiter<instr_trace_t*>` — is
the arbiter type used wherever multiple trace streams merge (e.g. commit fan-in, doc 16).

**Next:** [07 — Functional Units](07-functional-units.md).
