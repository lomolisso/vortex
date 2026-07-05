# Feasibility: a `SLICE_WIDTH` bit-slice knob for the Vortex GPU RTL

> **Scope of this note.** A feasibility explanation only — no RTL is changed. It assesses
> how hard it would be to refactor Vortex's SystemVerilog so the integer datapath (and,
> where meaningful, the controller and memory) is built from cascaded bit-slices selected
> by a new `SLICE_WIDTH` knob.

---

## 1. Context

**Why this question.** The goal is to judge how hard it would be to refactor Vortex's
SystemVerilog so the integer datapath (and, where meaningful, the controller and memory)
is *bit-sliced*: expressed as a cascade of `NUM_SLICES = XLEN / SLICE_WIDTH` identical
`SLICE_WIDTH`-bit tiles, chained by explicit inter-slice signals. `SLICE_WIDTH` becomes a
new build-time knob (alongside the existing `XLEN` knob) that sets the granularity of that
decomposition.

**Chosen semantics:**

- **Spatial / cascaded slices** (classic AMD Am2900 model), *not* temporal digit-serial.
  The full `XLEN`-bit operation still completes in the **same number of cycles**; it is
  merely *built* from `NUM_SLICES` combinational tiles with a carry/borrow/shift chain
  threaded between adjacent tiles. Pipeline-crossing buses stay `XLEN`-wide; the slicing
  is *inside* the functional units. This matches the intuition that `SLICE_WIDTH` is "the
  bit width of most data signals" — within each tile, data is `SLICE_WIDTH` wide, and the
  datapath is a `genvar` cascade of tiles.
- **Motivation:** a research / teaching artifact. Correctness, clarity and
  parameterizability matter more than any single PPA (power/perf/area) number.
- **Scope:** the whole GPU is examined — ALU, MUL/DIV, FPU, register file, LSU/caches,
  PC/branch, and the decode/control plane — with an explicit verdict on each.

**Intended outcome.** A clear, per-subsystem feasibility map plus a phased plan that a
follow-up implementation task can execute, including which subsystems should be sliced,
which should be left as full-width "black boxes," and why.

---

## 2. Executive verdict

Bit-slicing Vortex **spatially** is **feasible as a structural refactor**, and it is a
good fit for a research/teaching artifact — but "most signals" honestly means *most of
the integer arithmetic datapath*, not literally every bit in the design. The work divides
cleanly:

| Class | Subsystems | Verdict |
|---|---|---|
| **Naturally sliceable** (carry/borrow chain) | bitwise ALU ops, ADD/SUB, SLT/branch compares, PC+imm & LSU address adders, counters | ✅ Easy–Medium. Vortex even ships a *cascadable* adder (`VX_ks_adder` with `cin`/`cout`). |
| **Sliceable but involved** (data crosses slice boundaries) | shifters, integer multiplier, register-file SRAM, LMEM/cache data arrays | ⚠️ Medium–Hard. Needs inter-slice data buses or bit-plane RAM partitioning. |
| **Impractical / leave as black box** | FPU (fpnew/DSP/std), integer divider, decode & most of the control plane | ❌ Do **not** slice. FP arithmetic and division don't decompose cleanly; the decoder/control is not "data" at all. |

Two facts make this tractable and cheap to *start*:

1. **The knob is trivial to add.** Width already flows `VX_config.toml → ci/gen_config.py
   → VX_config.vh → VX_gpu_pkg.sv localparams → module params`. Adding `SLICE_WIDTH` and
   `NUM_SLICES` is a handful of lines; the real cost is the per-unit RTL rewrite.
2. **The arithmetic is concentrated.** The `XLEN`-wide behavioral operators
   (`+ - << >> *`) live in a *small, enumerable* set of modules
   ([`VX_alu_int.sv`](../../hw/rtl/core/VX_alu_int.sv), [`VX_alu_muldiv.sv`](../../hw/rtl/core/VX_alu_muldiv.sv),
   the LSU address add in [`VX_lsu_slice.sv:64`](../../hw/rtl/core/VX_lsu_slice.sv#L64), the PC
   increment in [`VX_alu_int.sv:347`](../../hw/rtl/core/VX_alu_int.sv#L347), FPU internals, and a
   few counters). Everything else is *width-parametric plumbing* (buses, muxes,
   crossbars, FIFOs, interface structs) that already scales with `XLEN` and needs little
   or no change.

The honest caveat: because it is spatial (same cycle count), this refactor does **not**
buy area or timing on its own — a hand-cascaded ripple typically *costs* a little versus
letting the synthesizer build a monolithic adder. Its value is *structural*: a uniform,
Am2900-style construction that then *enables* downstream experiments (per-slice
clock/power gating, arbitrary narrow-width variants, approximate "drop the low slices"
studies, slice-level fault injection).

---

## 3. The fundamental partition: what slices vs. what does not

Vortex already draws the exact line bit-slicing needs. The `DECL_EXECUTE_T` macro
([`VX_define.vh:513-539`](../../hw/rtl/VX_define.vh#L513-L539)) splits every execute packet into a
**control header** and **data operands**:

```
header:  uuid, wid, cta_id, tmask, pid, sop, eop, PC, wb, wr_xregs, rd, bytesel   ← CONTROL (stays full-width)
data:    rs1_data / rs2_data / rs3_data  [LANES][XLEN]                            ← DATA   (candidate to slice)
result:  data                            [LANES][XLEN]                            ← DATA   (candidate to slice)
```

- **DATA (slice these):** `rs1/rs2/rs3_data`, ALU/FPU results, writeback `data`,
  immediates *after* they are sign-extended to `XLEN` in the datapath, and the arithmetic
  *operands* of PC+imm and effective-address adds.
- **CONTROL (never slice):** opcodes, `funct3/7`, register indices (`rd`, `rs1..3`),
  `ex_type`/`op_type`, thread/warp masks, warp/CTA IDs, `uuid`, `valid`/`ready`
  handshakes, `sop`/`eop`, `bytesel`, trap/branch-taken flags. These are not
  positional-numeric data; there is no "bit i of the opcode" that a neighboring slice
  computes. **This is why the decoder is "trickier"** — it isn't trickier, it simply isn't
  data and should stay whole.

Ambiguous cases and how to treat them:

- **PC.** As an *instruction tag* threaded through the pipeline (`PC_BITS`), it is
  control — leave whole. As an *arithmetic operand* (branch target = `PC + imm`, `PC + 4`
  increment), it feeds an adder and slices like any adder. In the spatial model this is
  painless because it stays single-cycle
  ([`VX_alu_int.sv:83-90`, `:347-349`](../../hw/rtl/core/VX_alu_int.sv#L83-L90)).
- **Immediates.** Stored compactly as control (`imm20` inside `op_args`) and expanded to
  `XLEN` at execute ([`VX_alu_int.sv:78`](../../hw/rtl/core/VX_alu_int.sv#L78)). The 20-bit
  encoding stays whole in the front-end; only the expanded `XLEN` operand enters a slice.
- **Addresses.** The *effective-address computation* (`rs1 + offset`,
  [`VX_lsu_slice.sv:64`](../../hw/rtl/core/VX_lsu_slice.sv#L64)) is a sliceable adder. The
  *address bits themselves* (tag/index/offset presented to SRAM) are control-like and are
  consumed whole by the memory system.

---

## 4. Introducing the knob (the easy part)

Width threading today, using `XLEN` as the worked example:

1. **Declared** as an enum in [`VX_config.toml:324`](../../VX_config.toml#L324):
   `VX_CFG_XLEN = [32, 64]`.
2. **Generated** into `VX_config.vh` by [`ci/gen_config.py`](../../ci/gen_config.py) (resolves
   `expr:` fields; emits `` `define VX_CFG_XLEN 32/64 ``).
3. **Included** everywhere via [`VX_define.vh`](../../hw/rtl/VX_define.vh) (`` `include
   "VX_config.vh" ``).
4. **Derived** into localparams in [`VX_gpu_pkg.sv`](../../hw/rtl/VX_gpu_pkg.sv) (`XLENB`,
   `XLENB_W`, `LSU_WORD_SIZE`, …) that the whole design imports.
5. **Consumed** by modules either as `` `VX_CFG_XLEN `` directly or via package localparams
   / module parameters.

Adding `SLICE_WIDTH` mirrors this exactly:

- New enum `VX_CFG_SLICE_WIDTH` in `VX_config.toml` (e.g. `[8, 16, 32, 64]`) plus a
  build-time constraint `VX_CFG_XLEN % VX_CFG_SLICE_WIDTH == 0` (validated in
  `gen_config.py`, which already evaluates `expr:` constraints).
- Derived localparams in `VX_gpu_pkg.sv`:
  `SLICE_WIDTH = `VX_CFG_SLICE_WIDTH`, `NUM_SLICES = `VX_CFG_XLEN / SLICE_WIDTH`.
- **Identity default:** ship with `SLICE_WIDTH = XLEN` (⇒ `NUM_SLICES = 1`) so the cascade
  degenerates to today's single monolithic operator. This lets the plumbing land and pass
  regressions with **zero behavioral change** before any unit is truly sliced.

Interactions to respect: `SLICE_WIDTH` is orthogonal to `SIMD_WIDTH`/`NUM_THREADS` (lane
count) — slicing is *within* a lane's `XLEN` word, lanes stay independent. `FLEN` (FP
width) is deliberately decoupled (FPU is out of scope, §6.3). `XLENB`/`LSU_WORD_SIZE` byte
granularity is a natural sub-multiple to keep aligned with `SLICE_WIDTH`.

---

## 5. Core mechanism: the inter-slice chain

Spatial slicing replaces one behavioral operator with a `genvar` cascade of `NUM_SLICES`
tiles plus a chain signal whose *type* depends on the operation:

| Chain type | Operations | What crosses the boundary |
|---|---|---|
| **1-bit carry / borrow** | ADD, SUB, SLT/SLTU, branch compares, counters, PC±imm, address add | carry-out → carry-in of next tile (ripple, or carry-select for speed) |
| **none (bit-plane independent)** | AND/OR/XOR, CZERO/zicond mask, byte-enable masks, SRAM bit-planes | nothing — tiles are fully parallel, trivial |
| **`SLICE_WIDTH`-bit data bus** | SLL/SRL/SRA, funnel/rotate | bits shifted *out* of one tile become bits shifted *into* the neighbor; shift amount decoded globally |
| **partial-product / carry-save vector** | MUL | cross-tile digit products summed in a CSA/Wallace network |

Vortex already contains reusable primitives for the first and last rows:

- [`VX_ks_adder.sv`](../../hw/rtl/libs/VX_ks_adder.sv) is a Kogge-Stone adder **with explicit
  `cin`/`cout` ports** — i.e. already a cascadable adder slice. A sliced ADD is literally
  `NUM_SLICES` instances chained by `cout→cin`.
- [`VX_csa_tree.sv`](../../hw/rtl/libs/VX_csa_tree.sv), [`VX_wallace_mul.sv`](../../hw/rtl/libs/VX_wallace_mul.sv),
  [`VX_multiplier.sv`](../../hw/rtl/libs/VX_multiplier.sv), and the *already digit-serial*
  [`VX_serial_mul.sv`](../../hw/rtl/libs/VX_serial_mul.sv) / [`VX_serial_div.sv`](../../hw/rtl/libs/VX_serial_div.sv)
  give building blocks for the multiply/divide story.

So the refactor is largely: *swap behavioral `+ - << *` for cascades of these existing
slice primitives, parameterized by `SLICE_WIDTH`.*

---

## 6. Per-subsystem feasibility (whole GPU)

### 6.1 Integer ALU — [`VX_alu_int.sv`](../../hw/rtl/core/VX_alu_int.sv) — the primary target ✅

| Op family | Site | Chain | Difficulty |
|---|---|---|---|
| AND / OR / XOR / CZERO | around [`:106`, `:122-124`](../../hw/rtl/core/VX_alu_int.sv#L122) | none (bit-plane) | **Easy** |
| ADD / ADDW | [`:89-90`](../../hw/rtl/core/VX_alu_int.sv#L89) | carry | **Easy** (drop-in `VX_ks_adder` cascade) |
| SUB / SUBW | [`:96-97`](../../hw/rtl/core/VX_alu_int.sv#L96) | borrow | **Easy** |
| SLT/SLTU + BEQ/BNE/BLT… | [`:94-96`, `:221+`](../../hw/rtl/core/VX_alu_int.sv#L94) | borrow-out + MSB-tile sign | **Medium** (result at the top tile) |
| LUI / AUIPC | [`:73-81`](../../hw/rtl/core/VX_alu_int.sv#L73) | carry (reuses ADD) | **Easy** |
| SLL / SRL / SRA | [`:100-128`](../../hw/rtl/core/VX_alu_int.sv#L100) | `SLICE_WIDTH`-bit inter-tile bus | **Hard** — see below |
| PC+4 / PC+imm branch target | [`:83-90`, `:347-349`](../../hw/rtl/core/VX_alu_int.sv#L347) | carry | **Easy** (single-cycle in spatial model) |

The **shifter is the hard case within the ALU.** Today it's a behavioral barrel shift
(`<<`, `>>>`). A shift is inherently all-to-all across the word: bit `i` of the output can
come from any input bit, so a spatial slice needs a *funnel network* passing up to
`SLICE_WIDTH` bits between neighbors, and the shift amount (`log2(XLEN)` bits) is decoded
globally and broadcast. Feasible, but the most code and the biggest timing/area risk of
the ALU. Reduction/lane ops (VOTE, SHFL, WGATHER) operate *across lanes*, not across bits
within a word — they are orthogonal to `SLICE_WIDTH` and untouched.

### 6.2 Integer MUL / DIV — [`VX_alu_muldiv.sv`](../../hw/rtl/core/VX_alu_muldiv.sv) ⚠️ / ❌

- **Multiply:** a spatial digit-sliced multiplier is the schoolbook decomposition —
  `NUM_SLICES²` `SLICE_WIDTH×SLICE_WIDTH` partial products summed in a carry-save tree
  (reuse [`VX_csa_tree`](../../hw/rtl/libs/VX_csa_tree.sv) / [`VX_wallace_mul`](../../hw/rtl/libs/VX_wallace_mul.sv)).
  Doable but the accumulation network couples all tiles — **Medium–Hard**. The pipelined
  fast path today infers DSP/fabric multipliers ([`VX_multiplier.sv`](../../hw/rtl/libs/VX_multiplier.sv));
  those would be replaced by the sliced tree.
- **Divide:** already **bit-serial** ([`VX_serial_div.sv`](../../hw/rtl/libs/VX_serial_div.sv),
  MSB→LSB non-restoring). Division does not spatially decompose (each quotient bit depends
  on the running remainder). **Recommendation: leave the divider as-is** — it is already
  the narrowest form and is orthogonal to `SLICE_WIDTH`.

### 6.3 FPU — [`hw/rtl/fpu/`](../../hw/rtl/fpu/) ❌ leave as full-width black box

Backends are `fpnew` (third-party PULP library, [`VX_fpu_fpnew.sv`](../../hw/rtl/fpu/VX_fpu_fpnew.sv)),
vendor `DSP` ([`VX_fpu_dsp.sv`](../../hw/rtl/fpu/VX_fpu_dsp.sv)), and generic `std`
([`VX_fpu_std.sv`](../../hw/rtl/fpu/VX_fpu_std.sv)). FP arithmetic — mantissa alignment,
normalization shifts, rounding, exponent handling — spans the whole operand in ways that
resist bit-plane cascading, and the fpnew/DSP paths are external IP. **Keep the FPU
full-width and out of the `SLICE_WIDTH` contract.** (If a slice-aware FP study is ever
wanted, only the `std` backend's internal integer add/normalize could be revisited — a
separate project.) This is the main reason "most" ≠ "all."

### 6.4 Register file / operand collector — [`VX_opc_unit.sv`](../../hw/rtl/core/VX_opc_unit.sv) ⚠️

GPR banks are wide dual-port SRAM: `BANK_DATA_WIDTH = XLEN * SIMD_WIDTH`
([`:46`](../../hw/rtl/core/VX_opc_unit.sv#L46)) instantiated as [`VX_dp_ram`](../../hw/rtl/libs/VX_dp_ram.sv)
([`:284-286`](../../hw/rtl/core/VX_opc_unit.sv#L284)). Spatial slicing here = **bit-plane
partitioning**: split each word into `NUM_SLICES` sub-RAMs of `SLICE_WIDTH * SIMD_WIDTH`
bits. Byte-enables already give sub-word write granularity, so the write path adapts
naturally. **Medium.** Note this is optional: a sliced *ALU* works fine reading from a
still-wide register file (the read simply fans the `XLEN` word into the tile cascade).

### 6.5 LSU / caches / local memory — [`VX_lsu_slice.sv`](../../hw/rtl/core/VX_lsu_slice.sv), [`hw/rtl/mem/`](../../hw/rtl/mem/), [`hw/rtl/cache/`](../../hw/rtl/cache/) ⚠️/❌

- **Sliceable:** the effective-address adder ([`VX_lsu_slice.sv:64`](../../hw/rtl/core/VX_lsu_slice.sv#L64))
  and the per-lane load/store *data* words (`LSU_WORD_SIZE*8` wide) — same bit-plane logic
  as the register file.
- **Hard boundary:** cache and LMEM arrays are **byte-addressable, line-oriented SRAM**
  (`WORD_SIZE = XLEN/8`, `LINE_SIZE = 64 B`, defined in
  [`VX_cache_define.vh`](../../hw/rtl/cache/VX_cache_define.vh) and
  [`VX_local_mem.sv`](../../hw/rtl/mem/VX_local_mem.sv)). Their width is set by the memory
  word/line contract, not by the arithmetic datapath. **Recommendation: keep memory-array
  word width fixed; slice only the datapath feeding it.** A load returns an `XLEN` word
  which is then fanned into the tile cascade downstream — the SRAM itself need not be
  re-sliced.

### 6.6 PC / branch / scheduler — [`VX_scheduler.sv`](../../hw/rtl/core/VX_scheduler.sv), [`VX_fetch.sv`](../../hw/rtl/core/VX_fetch.sv) ✅ arithmetic only

The branch-target and PC-increment adders slice like §6.1 and stay single-cycle. The
warp-PC storage, thread masks, and branch-taken/handshake control
([`VX_branch_ctl_if.sv`](../../hw/rtl/interfaces/VX_branch_ctl_if.sv)) are control — untouched.

### 6.7 Decode & control plane — [`VX_decode.sv`](../../hw/rtl/core/VX_decode.sv) ❌ stays whole

The 32-bit instruction word is parsed atomically: opcode/`funct` fields select immediate
format and route to a unit; register indices are scattered and feed the scoreboard/GPR
addressing; masks/UUIDs/handshakes are per-cycle atomic. None of this is positional
numeric data, so `SLICE_WIDTH` does not apply. Only the *output* of decode that becomes a
datapath operand (the expanded immediate) participates downstream.

---

## 7. Blast radius & effort

- **Knob plumbing:** ~½ day. `VX_config.toml` + `ci/gen_config.py` constraint +
  `VX_gpu_pkg.sv` localparams; ships as an identity no-op at `SLICE_WIDTH = XLEN`.
- **Arithmetic to actually rewrite is concentrated**, not spread across the 51/321 files
  that merely *reference* `XLEN`. The behavioral operators live in ~6–8 modules:
  `VX_alu_int` (add/sub/compare/shift/PC), `VX_alu_muldiv` (mul), `VX_lsu_slice`
  (address), the PC increment, FPU-internal (excluded), and a few counters (`VX_bar_unit`,
  CSR). Most `XLEN` references are *width-parametric plumbing* (buses, muxes,
  `VX_stream_xbar`, FIFOs, interface structs) that already scale and need no logic change.
- **Reusable slice primitives already exist:** `VX_ks_adder` (cascadable adder),
  `VX_csa_tree`/`VX_wallace_mul` (multiply), `VX_serial_div` (leave divide serial). This
  meaningfully lowers effort versus a from-scratch bit-slice library.

Rough sizing for a research artifact: **ALU carry-chain family ≈ 1 week; shifter ≈ 0.5–1
week; sliced multiplier ≈ 1 week; register-file bit-planes ≈ 0.5–1 week.** FPU, caches,
divider, and decode are deliberately excluded.

---

## 8. Risks & caveats

- **Timing.** A hand-cascaded ripple across `NUM_SLICES` tiles has a longer critical path
  than a synthesized monolithic prefix adder. Small `SLICE_WIDTH` → many tiles → worse
  timing. Mitigations: carry-select between tiles, or *register the inter-slice carry*
  (which turns that unit temporal/multi-cycle — a deliberate departure from the pure
  spatial contract, acceptable as an opt-in per unit).
- **Synthesizer freedom.** Replacing `+`/`*` with explicit structure removes the tool's
  latitude to pick optimal arithmetic; expect a small area/timing tax. Fine for a teaching
  artifact whose point is the explicit structure.
- **"Most" is not "all."** FPU, cache/LMEM arrays, the divider, and the decode/control
  plane stay full-width by design. State this boundary plainly so the knob isn't oversold.
- **Verification surface.** Every `SLICE_WIDTH` is a new configuration to validate against
  the RISC-V tests and SimX reference. Start from the identity default and sweep only
  divisors of `XLEN`.
- **Knob interplay.** Must hold `XLEN % SLICE_WIDTH == 0`; keep `SLICE_WIDTH` orthogonal to
  `SIMD_WIDTH`/`FLEN`; watch `XLENB`/byte-enable alignment.

---

## 9. Recommended approach (phased)

- **Phase 0 — Plumbing (no behavior change).** Add `VX_CFG_SLICE_WIDTH` enum +
  divisibility constraint + `NUM_SLICES` localparam; default `SLICE_WIDTH = XLEN`. Prove
  regressions are bit-identical.
- **Phase 1 — ALU carry-chain family.** Slice AND/OR/XOR (trivial), then ADD/SUB and
  SLT/branch compares via a `VX_ks_adder` cascade. Highest value, lowest risk; also covers
  LUI/AUIPC, PC±imm, PC+4, and the LSU address add (same adder).
- **Phase 2 — Shifter & multiplier.** Funnel-network shifter; schoolbook digit-sliced
  multiplier on `VX_csa_tree`. Leave the divider serial.
- **Phase 3 — Register-file bit-planes** (optional). Partition GPR SRAM into `NUM_SLICES`
  sub-arrays.
- **Explicitly out of scope:** FPU, cache/LMEM SRAM arrays, integer divider, decode &
  control plane — documented as full-width black boxes.

---

## 10. Verification

- **Identity check first:** at `SLICE_WIDTH = XLEN`, output must be bit-for-bit identical
  to today (proves the refactor is behavior-preserving before any true slicing).
- **Sweep** `SLICE_WIDTH ∈ {divisors of XLEN}` (e.g. 8/16/32 for XLEN=32) and run the
  existing RISC-V unit tests ([`hw/unittest/`](../../hw/unittest/), [`tests/`](../../tests/)) plus the
  SimX functional reference; require no regression at the default and matching
  architectural results at each slice width.
- **Targeted unit tests** per functional unit (adder carry propagation across tile
  boundaries, shifter at every shift amount, multiplier full product) since tile-boundary
  bugs are the characteristic failure mode.

---

## 11. Files a follow-up implementation would touch (representative, not exhaustive)

- Knob: [`VX_config.toml`](../../VX_config.toml), [`ci/gen_config.py`](../../ci/gen_config.py),
  [`VX_gpu_pkg.sv`](../../hw/rtl/VX_gpu_pkg.sv).
- Sliceable arithmetic: [`VX_alu_int.sv`](../../hw/rtl/core/VX_alu_int.sv),
  [`VX_alu_muldiv.sv`](../../hw/rtl/core/VX_alu_muldiv.sv),
  [`VX_lsu_slice.sv`](../../hw/rtl/core/VX_lsu_slice.sv), [`VX_opc_unit.sv`](../../hw/rtl/core/VX_opc_unit.sv).
- Reused primitives: [`VX_ks_adder.sv`](../../hw/rtl/libs/VX_ks_adder.sv),
  [`VX_csa_tree.sv`](../../hw/rtl/libs/VX_csa_tree.sv), [`VX_wallace_mul.sv`](../../hw/rtl/libs/VX_wallace_mul.sv),
  [`VX_serial_div.sv`](../../hw/rtl/libs/VX_serial_div.sv), [`VX_dp_ram.sv`](../../hw/rtl/libs/VX_dp_ram.sv).
- Left whole (documented boundaries): [`hw/rtl/fpu/`](../../hw/rtl/fpu/), [`hw/rtl/cache/`](../../hw/rtl/cache/),
  [`hw/rtl/mem/`](../../hw/rtl/mem/), [`VX_decode.sv`](../../hw/rtl/core/VX_decode.sv).
