# 22 — How to Add a Feature

**Scope:** the practical recipes — adding an instruction / functional unit, a cluster-level
extension engine, a cache/memory level, and a performance counter. Each lists the concrete
registration points (which file to touch, in order). This is the synthesis of docs 01–21; follow
the cross-references for the *why*. Read [`../../../AGENTS.md`](../../../AGENTS.md) first and obey
its rules (out-of-tree build, `CONFIGS` matching, sw/sim isolation, comments-explain-why).

**Three rules that apply to every recipe:**
- **Implement both halves** — a functional `execute()`/handler *and* a timing `latency_of()` /
  channel `send(delay)` (doc 01 §5). Forgetting the latency makes everything 1-cycle.
- **Respect backpressure** — check `full()` / use `try_send` before sending (doc 01 §3.3); never
  drop a packet silently.
- **Keep the 1:1 RTL correspondence** — one SimX class ≈ one future SV module; no façades (this is
  why the codebase splits cleanly into per-block classes).

---

## 1. Add an instruction / functional unit

To add a new compute op or a whole new unit (model: `AluUnit`, [`../alu_unit.h`](../alu_unit.h)):

1. **Decode** — in [`../decode.cpp`](../decode.cpp), recognize the encoding and set
   `trace->fu_type` (an `FUType`) and `trace->op_type` (the op variant) + `dst_reg`/`src_regs`. New
   opcodes live in the custom space; the enums are in [`../instr.h`](../instr.h)/[`../types.h`](../types.h).
2. **The unit** — subclass `FuncUnit<NUM_BLOCKS>` (doc 05 §3, [`../func_unit.h`](../func_unit.h)) with
   `on_tick()` that pops `Inputs[b]`, calls a private `execute(trace)` (read `src_data` → write
   `dst_data`), and `Outputs[b].send(trace, latency_of(trace))`. (For a *new op on an existing
   unit*, just extend that unit's `execute`/`latency_of` and skip to step 5.)
3. **Instantiate + wire** — in `Core::Impl` ([`../core.cpp`](../core.cpp)) create the unit and bind
   its `Inputs`/`Outputs` to the dispatcher and the commit fan-in (the existing units show the
   pattern). Add the `FUType` enum value and the dispatcher routing (doc 05 §2).
4. **Build** — add the `.cpp` to `SRCS` in [`../Makefile`](../Makefile) (gate with a
   `$(filter -DVX_CFG_EXT_*, $(XCONFIGS))` block if optional, doc 18 §3.1).
5. **Counters + test** — bump the relevant `Core::PerfStats` fields (doc 19), then write a kernel
   under `tests/regression/<app>` and run it (`blackbox.sh --driver=simx`, doc 18 §4). If the op
   cracks into micro-ops, add a generator to the `Sequencer` (doc 04 §4 / 06 §4).

---

## 2. Add a cluster-level extension engine

Model: DXA (`DxaUnit` + `DxaCore`, doc 17 §2). Pattern = a per-core SFU sub-unit that posts to a
cluster-shared engine:

1. **Request/response types** — define `XReq`/`XRsp` structs (in the engine's header or
   [`../types.h`](../types.h)).
2. **Per-core sub-unit** — a plain (non-`SimObject`) helper owned by `SfuUnit`
   ([`../sfu_unit.h`](../sfu_unit.h)) that lane-decodes the trace and pushes an `XReq` on an
   outbound `SimChannel<XReq>` declared on `SfuUnit` (model `dxa_req_out`,
   [`sfu_unit.h:59`](../sfu_unit.h#L59)). Register it in `SfuUnit::on_tick`'s `op_type` demux.
3. **Cluster engine** — a `SimObject` `XCore` (model `DxaCore`, [`../dxa/dxa_core.h`](../dxa/dxa_core.h))
   with per-core `XReq` inputs, a memory port to L2 (arbitrated, doc 16), and its result path
   (a response channel, or a barrier-event completion via `flags`+`tx_callback`, doc 09 §5).
4. **Wire in `Cluster::Impl`** ([`../cluster.cpp`](../cluster.cpp)) — bind each core's SFU
   `XReq` channel to `XCore` inputs; add the engine to `Cluster::PerfStats` and the L2 arbiter rows
   (doc 02 §4.1).
5. **Gate** — `#ifdef VX_CFG_EXT_X_ENABLE` around the includes/members (doc 17 §1), add the macro to
   [`../../../VX_config.toml`](../../../VX_config.toml) and a MISA bit, and a conditional `SRCS`
   block in [`../Makefile`](../Makefile).
6. **Counters + kernel** — perf-stats + MPM class (§4 / doc 19), then a regression kernel.

> For a *per-core* compute accelerator (model: TCU, doc 17 §3) you instead subclass
> `FuncUnit<NUM_BLOCKS>` like §1 and, if it needs scratchpad operands, bind an LMEM port in
> `Core::Impl` (doc 13 §2) and stage tiles through a `TcuTbuf`-style line cache.

---

## 3. Add a cache / memory level

The `Cache` class (doc 12) is reused at every level, so adding (say) an L4 is mostly wiring:

1. **Instantiate** a `Cache::Create(name, Cache::Config{…})` in the owning container (model: L3 in
   `ProcessorImpl`, [`processor.cpp:64`](../processor.cpp#L64)) with the right `C/L/W/A/B`,
   `mshr_size`, `write_back`, and **`is_llc`** (true only for the new last level — and re-check the
   AMO write-through invariant, doc 02 §2.2 / doc 12 §4).
2. **Splice the channels** — bind the upstream level's `mem_req_out`/`mem_rsp_in` to the new cache's
   `core_req_in`/`core_rsp_out`, and the new cache's `mem_req_out`/`mem_rsp_in` downstream (doc 02
   §2.1). Mind the port counts (doc 02 §5).
3. **Flush** — add the new level to `ProcessorImpl::flush_caches()`
   ([`processor.cpp:154`](../processor.cpp#L154)) in the correct inside-out order (doc 02 §3.2,
   doc 12 §5).
4. **Config + perf** — add `VX_CFG_L4_*` knobs (doc 18), aggregate its `PerfStats` (doc 19), and add
   a bypass path so the level can be disabled (`Config.bypass`, doc 12 §6).

---

## 4. Add a performance counter (the short recipe)

(Expanded from doc 19 §4.) (a) add a field + `operator+=` to the module's `PerfStats`; (b)
increment it in the module's `on_tick`/handler at the event; (c) surface it through `CsrUnit`
(doc 10 §3) at a `VX_CSR_MPM_*` address; (d) declare the CSR (low + `_H`) in the right `[csr_mpm_*]`
block of [`../../../VX_types.toml`](../../../VX_types.toml) (new class → add a `VX_DCR_MPM_CLASS_*`
id); (e) rebuild (+ re-`configure` if you touched TOML) and read with `--perf` (doc 18 §4).

---

## 5. Verification checklist (per `AGENTS.md` §4)

- Build **and run inside the Docker dev container**, from the generated `build/` tree.
- A **kernel proves the feature** — author one under `tests/regression/<app>` and confirm PASS via
  `blackbox.sh --driver=simx`; keep every invocation under the **120-second** cap.
- Match `CONFIGS` on the app build *and* the `blackbox.sh` run (doc 18 §4).
- If you'll later port to RTL, add cycle-stamped `DT` traces now (doc 20 §5) so the SimX-as-oracle
  diff works.

---

That closes the loop: docs 01–21 explain how the machine is built; this doc is how you extend it.
Back to the [index](README.md).
