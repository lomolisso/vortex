# 21 — Host Runtime & External Simulators

**Scope:** how the simulator is driven from outside — the `Processor` C++ facade, the standalone
`simx` binary, the SimX *runtime backend* that implements the Vortex host API, and the optional
SST/gem5 coupling. Sources: [`../processor.cpp`](../processor.cpp),
[`../processor.h`](../processor.h), [`../main.cpp`](../main.cpp),
[`../../../sw/runtime/simx/vortex.cpp`](../../../sw/runtime/simx/vortex.cpp), [`../mem/memory.h`](../mem/memory.h),
[`../sst/`](../sst), [`../gem5/`](../gem5). Prereqs: [02](02-architecture-hierarchy.md),
[03](03-kernel-launch-and-dispatch.md).

---

## 1. The `Processor` facade ([`../processor.h`](../processor.h))

`Processor` is the public boundary; `ProcessorImpl` (doc 02) is the guts. The driving surface
([`processor.cpp:322-381`](../processor.cpp#L322)):

| Method | Role |
|---|---|
| `attach_ram(RAM*)` | install the functional backing store (global memory) |
| `reset()` | reset the platform |
| `start_kmu()` | arm the kernel grid walk (doc 03) |
| `run()` | run to completion (the internal `SimPlatform::tick` loop, doc 02 §3.1) |
| `cycle()` | advance **one** cycle; returns `any_running()` — for external drivers |
| `dcr_write` / `dcr_read` | the host↔device control-register channel (doc 02 §3.3, doc 03 §2) |
| `memsim()` | the DRAM `Memory` object — the external-sim hook point (§4) |
| `get_first_core()` | per-core inspection (profiling/debug) |

There are two ways to drive it: `run()` (own the loop) or repeated `cycle()` (you own the loop, e.g.
an external simulator or the debug server).

---

## 2. The standalone `simx` binary ([`../main.cpp`](../main.cpp))

`main.cpp` is the self-contained CLI (doc 20 covers its flags). It creates a `RAM`, a `Processor`,
`attach_ram`s, programs the **base DCRs** for a trivial 1×1×1 launch
([`main.cpp:103-122`](../main.cpp#L103)), loads the program (ELF / `.vxbin` / `.bin` / `.hex`,
[`main.cpp:128-148`](../main.cpp#L128)), then runs in one of three modes
([`main.cpp:154-243`](../main.cpp#L154)): **debug** (DTM/RBB, doc 20), **HTIF monitor** (for
upstream `riscv-tests` — tick until the `tohost` word), or **standard** (`cycle()` loop +
`CoutDrainer` for `vx_printf`, then a cache flush via `VX_DCR_BASE_CACHE_FLUSH`, then read the exit
code from `VX_MEM_IO_EXIT_CODE`). This binary is what `make -C tests/... run-rtlsim`-style flows do
*not* use — those go through the runtime backend (§3); `main.cpp` is the bare-metal harness.

---

## 3. The runtime backend ([`../../../sw/runtime/simx/vortex.cpp`](../../../sw/runtime/simx/vortex.cpp))

When a host program uses the Vortex API (`vx_dev_open`, `vx_mem_alloc`, `vx_copy_to_dev`,
`vx_start`, …) with `VORTEX_DRIVER=simx`, it links this backend. Its `DeviceImpl`
owns a `Processor processor_` and a functional `CommandProcessor cp_`
([`vortex.cpp:165-167`](../../../sw/runtime/simx/vortex.cpp#L165)):

- **Unified memory.** SimX has a single address space, so host buffers are plain process
  allocations; the device's backing `ram_` is attached with `processor_.attach_ram(&ram_)`
  ([`vortex.cpp:52`](../../../sw/runtime/simx/vortex.cpp#L52)) — no host↔device copy engine, the
  "copy" is a memcpy into the shared RAM.
- **Command processor.** The `CommandProcessor` (shared in [`../../common/cmd_processor.h`](../../common/cmd_processor.h))
  models the device's command/DCR interface functionally. Its hooks route to the processor:
  `vortex_dcr_write/read → processor_.dcr_write/read`
  ([`vortex.cpp:144-151`](../../../sw/runtime/simx/vortex.cpp#L144)). Kernel launch = program the KMU
  DCRs (doc 03) through these hooks, then drive `cycle()` in bounded bursts so the host stays
  responsive.

So the kernel descriptor flow of doc 03 is exactly what this backend does on a `vx_start`: write
`VX_DCR_KMU_*`, tick to completion, flush caches, let the host read results back from `ram_`.

---

## 4. External simulators (SST / gem5)

SimX can hand its **memory timing** to an external simulator while keeping the functional data path
local. The hook is `Memory::set_pre_send_hook` ([`memory.h:65`](../mem/memory.h#L65)): when
installed, every accepted `MemReq` is mirrored (by const-ref) to the external link *just before* the
local DRAM model enqueues it — **timing-only telemetry, zero-copy** (the bytes never leave Vortex's
RAM). `Processor::memsim()` ([`processor.cpp:371`](../processor.cpp#L371)) exposes the `Memory`
object so the wrapper can attach.

- **SST** ([`../sst/`](../sst): `vortex_simulator.cpp`, `vortex_gpgpu.*`) — an SST component that
  drives `Processor::cycle()` and forwards requests to an `SST::Interfaces::StandardMem` link.
- **gem5** ([`../gem5/`](../gem5): `VortexGPGPU` SimObject, `dev_mem.*`) — the gem5 equivalent.

Built via `USE_SST=1` / `USE_GEM5=1` (mutually exclusive, doc 18 §3.2). The local model is
unchanged; only the memory-timing source swaps. The `std::function` hook keeps the `Memory` module
itself SST/gem5-agnostic.

---

## 5. Summary

Standalone: `main.cpp` → `Processor::run()`. Hosted: the runtime `DeviceImpl` → `CommandProcessor`
→ DCRs → `Processor::cycle()`. External: an SST/gem5 wrapper → `Processor::cycle()` + a
`memsim()` pre-send hook. All three reduce to the same `Processor` driving surface over the same
`SimPlatform` clock (doc 01).

**Next:** [22 — How to Add a Feature](22-howto-add-features.md).
