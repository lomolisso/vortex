# 10 — The Functional Model

**Scope:** how SimX computes the *architectural result* of a program — registers, CSRs, memory
values, traps — and how that functional model relates to the timing model. Sources:
[`../opc_unit.h`](../opc_unit.h), [`../scheduler.h`](../scheduler.h), [`../csr_unit.h`](../csr_unit.h),
[`../decode.h`](../decode.h), the unit `execute()` methods (docs 07–08), and the memory classes
(docs 11–15). Prereqs: [01](01-simulation-framework.md) §5, [06](06-instruction-lifecycle.md).

---

## 1. There is no separate emulator

A key thing to internalize: **SimX has no standalone ISA emulator file.** Functional execution and
timing are *unified in the same `SimObject`s* (this is the SimX v3 design — see
[`../../../docs/designs/simx_simulator_architecture.md`](../../../docs/designs/simx_simulator_architecture.md)).
The architectural effect of each instruction is computed by the **functional unit that owns it, the
cycle it executes** — `AluUnit::execute`, `FpuUnit::execute`, `LsuUnit` response writeback,
`CsrUnit::process`, `WctlUnit::process`. "Functional" is a *concern*, distributed across the
machine, not a place.

So the question "where does the program actually run?" has a four-part answer:

| Architectural state | Lives in | Doc |
|---|---|---|
| Integer / float register files | `OpcUnit` (`ireg_file`/`freg_file`, `[reg][thread]`) | [04](04-pipeline-frontend.md) §5 |
| Warp control: PC, `tmask`, IPDOM stack, `fcsr`, `mscratch`, trap CSRs, `cta_csrs` | `warp_t` in `Scheduler` | [04](04-pipeline-frontend.md) §1 |
| CSR semantics | `CsrUnit` | §3 |
| Memory bytes (global / shared) | DRAM backing `RAM` (via `attach_ram`) / `LocalMem` | [11](11-memory-hierarchy-overview.md), [13](13-local-memory-scratchpad.md) |

---

## 2. Computing a result

```
Decoder.decode(code, uuid)            → Instr (sets fu_type, op_type, dst_reg, src_regs)   (decode.h:32)
Operands.fetch_operands(trace)        → OpcUnit.read_src() fills trace->src_data            (opc_unit.h:42)
<owning FU>.execute(trace)            → computes trace->dst_data[thread]  (0-cycle, exact)  (docs 07–08)
   • ALU also mutates warp.PC for branches                                                  (alu_unit.h:29)
   • CSR ops run via CsrUnit::process(trace)                                                (csr_unit.h:47)
   • loads/stores read/write the memory classes via the LSU                                 (doc 08)
Operands.writeback(trace)             → OpcUnit.writeback() applies dst_data to the regfile (opc_unit.h:39)
```

The result is computed **exactly and immediately**; timing is added only as the channel `send`
delay (`latency_of`, doc 01 §5). The `instr_trace_t` (doc 06) is the carrier — it holds the
operands read in and the result computed out.

---

## 3. CSRs — `CsrUnit` ([`../csr_unit.h`](../csr_unit.h))

`CsrUnit` is a plain helper owned by the SFU. `process(trace)` ([`csr_unit.h:47`](../csr_unit.h#L47))
executes `CSRRW`/`CSRRS`/`CSRRC` side effects. It also exposes the access surface the FPU needs —
`get_fpu_rm` (rounding mode) and `update_fcrs` (accrued exception flags,
[`csr_unit.h:52-53`](../csr_unit.h#L52)). Per-warp CSR state (`fcsr`, `mscratch`, `cta_csrs`, the
trap CSRs) physically lives on the `Scheduler`'s `warp_t` and is reached via
`core->scheduler().warp(wid)` ([`csr_unit.h:36-39`](../csr_unit.h#L36)). Under VM, a `satp` write is
mirrored and forwarded to `Core::set_satp` so the MMU updates its root
([`csr_unit.h:67-71`](../csr_unit.h#L67), doc 14).

The CTA index CSRs a kernel reads (`threadIdx`/`blockIdx`/`blockDim`/`gridDim`, cluster size, LMEM
base) come from the `cta_csrs_t` snapshot ([`scheduler.h:45`](../scheduler.h#L45)) populated at CTA
dispatch (doc 03). MPM performance-counter CSRs are also serviced here (doc
[19](19-profiling-and-perf-counters.md)).

---

## 4. Traps — ECALL / EBREAK / MRET

Synchronous traps are handled by the `Scheduler` ([`scheduler.h:127-135`](../scheduler.h#L127)):

- `raise_trap(wid, cause, trap_pc)` snapshots the faulting PC into `mepc`, sets `mcause`, and
  redirects the warp PC to `mtvec`.
- `trigger_ecall` / `trigger_ebreak` are the entry points (an `ECALL` is how a kernel signals exit
  / host calls; the program exit code is read back via `Operands::get_exit_code`,
  [`operands.h:40`](../operands.h#L40)).
- `mret` restores the PC from `mepc`.

Page faults from the MMU (doc 14) funnel through the same `raise_trap` path. The per-warp trap CSRs
(`mstatus/mtvec/mepc/mcause/mtval`) live on `warp_t` ([`scheduler.h:87-91`](../scheduler.h#L87)).

---

## 5. Program load & determinism

The program image is loaded into the DRAM backing `RAM` by the ELF loader
([`../../common/elf_loader.cpp`](../../common/elf_loader.cpp)) and attached via
`Processor::attach_ram` (doc 02 §2, doc 21). The KMU's start PC/entry/arg DCRs (doc 03) then point
the first warps at it.

SimX is **deterministic**: there is no wall-clock or randomness in the model, so a given
program+config always produces the same cycle count and the same `--debug` trace — which is what
makes the SimX-as-oracle RTL-debugging workflow (per `AGENTS.md` §4) possible.

---

## 6. Functional vs timing — the one-paragraph version

Timing lives in the channels and `latency_of`; *correctness* lives in `execute()`/`writeback`,
`CsrUnit`, the trap path, and the memory classes. They are deliberately separate so the simulator's
answers are right regardless of how the timing model is tuned, and so a timing refinement can never
silently corrupt a result (doc 01 §5).

**Next:** [11 — Memory Hierarchy Overview](11-memory-hierarchy-overview.md).
