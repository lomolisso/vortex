# 07 — Functional Units (ALU, FPU, SFU)

**Scope:** the compute functional units that hang off the dispatcher — the ALU, the FPU, and the
multi-purpose SFU (which fans out to control-flow, CSR, and the accelerator sub-units). Sources:
[`../func_unit.h`](../func_unit.h), [`../alu_unit.h`](../alu_unit.h), [`../fpu_unit.h`](../fpu_unit.h),
[`../sfu_unit.h`](../sfu_unit.h), [`../wctl_unit.h`](../wctl_unit.h), [`../csr_unit.h`](../csr_unit.h).
The LSU has its own doc ([08](08-lsu-and-memory-ops.md)); the TCU's is [17](17-extension-engines.md).
Prereq: [05](05-pipeline-dispatch-execution.md).

---

## 1. The shape every FU shares

All units derive from `FuncUnit<NUM_BLOCKS>` (doc [05](05-pipeline-dispatch-execution.md) §3): an
array of `NUM_BLOCKS` `Inputs`/`Outputs` channel pairs and a `on_tick()`. The per-unit pattern is
always the same two methods:

- **`execute(trace)`** — **functional**: read `trace->src_data[*][thread]`, compute
  `trace->dst_data[thread]`, mutate side state if needed. Runs in zero simulated time.
- **`latency_of(trace)`** — **timing**: returns the cycle count to delay the response. `on_tick`
  pops an input, calls `execute`, then `Outputs[b].send(trace, latency_of(trace))`.

This is the functional-vs-timing split (doc [01](01-simulation-framework.md) §5) realized per
unit. The `NUM_*_BLOCKS` widths are independent per unit (`VX_CFG_NUM_ALU_BLOCKS`,
`_FPU_`, `_SFU_`, `_LSU_`) — the dispatcher adapts issue width to each (doc 05 §1).

---

## 2. `AluUnit` ([`../alu_unit.h`](../alu_unit.h))

`AluUnit : FuncUnit<VX_CFG_NUM_ALU_BLOCKS>`. Its `execute` ([`alu_unit.h:31`](../alu_unit.h#L31))
covers the integer datapath and is the simplest unit to read first when learning the pattern:

- integer arithmetic/logic (ADD/SUB/SLL/SRL/SRA/AND/OR/XOR, immediates);
- **branches** (BR/JAL/JALR) — note the comment: `execute` "**mutates `warp.PC` for branches**"
  ([`alu_unit.h:29-30`](../alu_unit.h#L29)); the per-lane condition + redirect feed the IPDOM
  divergence machinery in the scheduler (doc [04](04-pipeline-frontend.md) §3);
- multiply/divide (the `MdvType` ops, longer `latency_of`);
- warp-level shuffle/vote ops (`WGATHER`/`VOTE`/`SHUFFLE`) route here as ALU-class ops.

The implementation is in [`../alu_unit.cpp`](../alu_unit.cpp); `op_type` is `std::get`-extracted to
the specific `IntrAluArgs`/`IntrMdvArgs` (doc 05 §2).

---

## 3. `FpuUnit` ([`../fpu_unit.h`](../fpu_unit.h))

`FpuUnit : FuncUnit<VX_CFG_NUM_FPU_BLOCKS>`. Same `execute`/`latency_of` shape
([`fpu_unit.h:29-31`](../fpu_unit.h#L29)). It implements IEEE-754 arithmetic (FADD/FMUL/FMA/FDIV/
SQRT, conversions) using the **softfloat** library so results are bit-accurate
(implementation in [`../fpu_unit.cpp`](../fpu_unit.cpp); the same softfloat backend the TCU's DPI
path uses, doc 17). FP latencies are larger and op-dependent (`latency_of`).

**FCSR** (rounding mode + accrued exception flags) is **per-warp** state: it lives on `warp_t.fcsr`
([`scheduler.h:78`](../scheduler.h#L78)) and is reached through `CsrUnit` (§4.2). Note from
doc [04](04-pipeline-frontend.md): the integer/float **register files are shared per warp-slot in
`OpcUnit`**, not duplicated per FU — the FPU reads/writes the same `freg_file` the operand
collector manages.

---

## 4. `SfuUnit` — the special-function fan-out ([`../sfu_unit.h`](../sfu_unit.h))

`SfuUnit : FuncUnit<VX_CFG_NUM_SFU_BLOCKS>` is a **single dispatch port that fans out to per-op
sub-units by `op_type`, then gathers results back to one result port**
([`sfu_unit.h:40-43`](../sfu_unit.h#L40)). The sub-units are plain (non-`SimObject`) helpers it
owns:

| Sub-unit | Handles | Doc |
|---|---|---|
| `WctlUnit` | warp control: TMC, WSPAWN, SPLIT/JOIN, BAR (barrier), PRED, WSYNC, vote/shuffle | §4.1, [09](09-barriers-and-sync.md) |
| `CsrUnit` | CSR read/write, perf-counter (MPM) reads, SATP | §4.2, [10](10-functional-emulator.md), [19](19-profiling-and-perf-counters.md) |
| `DxaUnit` | issues async-copy requests to the cluster `DxaCore` | [17](17-extension-engines.md) |
| `TexUnit`/`OmUnit`/`RasterUnit` | graphics requests to the cluster engines | [17](17-extension-engines.md) |

### 4.1 `WctlUnit` — control flow and barriers

`WctlUnit` ([`../wctl_unit.h`](../wctl_unit.h)) executes the warp-control ops decoded from the
custom opcode space. It is where **SPLIT/JOIN** push/pop the IPDOM stack (doc 04 §3) and where
**BAR.ARRIVE/WAIT/EXPECT_TX** call into the `BarrierUnit` (doc [09](09-barriers-and-sync.md)). It
also handles the cooperative `VOTE`/`SHUFFLE` cross-lane ops.

### 4.2 `CsrUnit` — CSR access

`CsrUnit` ([`../csr_unit.h`](../csr_unit.h), accessible via `SfuUnit::csr_unit()`
[`sfu_unit.h:53`](../sfu_unit.h#L53)) services `csrr*`/`csrw*`. It reads the per-CTA snapshot
(`cta_csrs`, doc 03/04) for thread/block indices, the per-warp FCSR/trap CSRs, and the **MPM
performance counters** (doc [19](19-profiling-and-perf-counters.md)). Under VM it handles `satp`
writes (`Core::set_satp`, doc [14](14-virtual-memory-mmu.md)).

### 4.3 The "fire-and-wait" sub-unit path

DXA/TEX/etc. don't compute inline. The SFU posts a request on an outbound channel (`dxa_req_out`,
`tex_req_out`, …, [`sfu_unit.h:55-81`](../sfu_unit.h#L55)) that the `Cluster` binds to the shared
engine, and — per the header comment ([`sfu_unit.h:44-48`](../sfu_unit.h#L44)) — for TEX **does not**
push the trace onto its writeback output; the cluster engine owns the trace until it returns it via
`tex_rsp_in`, at which point `on_tick` forwards it to the original writeback lane. This is the same
async pattern barriers and the LSU fence use (`fetch_stall`/`resume_warp`, doc 05 §4.1).

---

## 5. Summary

ALU and FPU are the canonical "read `src_data`, write `dst_data`, send after `latency_of`" units —
ALU additionally redirects branch PCs. The SFU is a router: it demuxes special ops to control/CSR
sub-units (inline) or to the cluster accelerators (fire-and-wait), then re-gathers to one writeback
port. The LSU, structurally similar but memory-coupled, is doc [08](08-lsu-and-memory-ops.md).

**Next:** [08 — LSU & Memory Operations](08-lsu-and-memory-ops.md).
