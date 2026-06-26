# Toy-H100 SimX Roadmap — Proposal (Program Overview)

**Status:** proposal (pre-implementation). **Scope:** SimX cycle-level simulator only
([`sim/simx/`](../../sim/simx/)) — **no RTL/SystemVerilog in this program**. This document is
the umbrella roadmap; each task has its own proposal (linked in §4) and is implemented and
verified in sequence.

This program turns Vortex into a small-scale **"toy H100"** whose **compute *and* memory
hierarchies map 1-to-1 to a real H100 SM**, in SimX. It is the implementation arm of the gap
analysis in [`../designs/h100_vortex_mapping.md`](../designs/h100_vortex_mapping.md); read that
first. The headline correspondence it relies on: **H100 SM ↔ Vortex core**, **H100 SMSP ↔
Vortex issue slot**, and the cache placement / DSMEM / TMA gaps enumerated there are exactly
what these tasks close.

---

## 1. Why (motivation)

The gap analysis found three structural divergences between the H100 SM and the Vortex
core/socket, plus a memory-model fidelity gap and a software gap:

1. **Cache placement.** H100 fuses a 256 KB **L1D + SMEM** in one per-SM array; Vortex splits
   them — `LocalMem` scratchpad per **core**, dcache per **socket**. (→ **Task 1**)
2. **TMA placement.** H100's TMA is per-SM; Vortex's DXA is cluster-shared. (→ **Task 2**)
3. **DSMEM / SM-to-SM network *and* TMA multicast.** Absent in Vortex; only approximated by
   DXA multicast. The SM-to-SM network is a **substrate** with **two first-class consumers** —
   **DSMEM remote addressing** (thread-issued, fine-grained) and **TMA multicast** (bulk async
   copy with hardware fan-out, *one* global read). (→ **Task 3**)
4. **Two-proxy memory model.** PTX distinguishes the **generic proxy** (LSU) from the **async
   proxy** (TMA/DXA); Vortex models neither the proxy tag nor the generic↔async crossing fence.
   Reproducing it (incl. `fence.proxy.async` and byte-granular mbarrier completion) is an
   additive **fidelity** layer — *not* a scoped-consistency model. (→ **Task 4**)
5. **Feature reachability from OpenCL.** DXA/WGMMA/DSMEM are reachable only from the C++ kernel
   model; OpenCL C cannot drive them — which also blocks the future CUDA→OpenCL effort. (→ **Task 5**)

## 2. Non-negotiable principles (apply to every task)

- **SimX only.** All work lands under [`sim/simx/`](../../sim/simx/) (Tasks 4–5 also touch
  [`sw/`](../../sw/) — the instruction surface and the OpenCL surface respectively). No RTL.
- **Preserve the 1:1 class↔module mapping.** Each SimX C++ class is a stand-in for exactly one
  future SystemVerilog module. Every class added or reworked must be shaped so the eventual RTL
  port is a mechanical translation. Per-level ownership stays explicit. See §6.
- **Genuine reworks, no façades.** When a task says *remove* a class, delete it and replace it —
  do not wrap the old structure. Do not fake behavior (an unmodeled partition, unarbitrated
  routing, etc.). Reworks over workarounds.
- **Root-cause discipline** per [`../../AGENTS.md`](../../AGENTS.md) §3.
- **Do not introduce a scoped memory-consistency *model*** (PTX/SASS-style scope qualifiers,
  scope-tree visibility) to deliver the **structural** tasks (1–3) or OpenCL (5). The feature set
  is scratchpad-centric and barrier/transaction-synchronized, which is exactly the regime where
  scopes stop being load-bearing — see §10. The **two-proxy model** (`fence.proxy.async`, async
  completion) is a *separate, deliberate* fidelity task (Task 4); **proxies ≠ scopes** — proxies
  order datapath crossings, scopes bound visibility radius. Task 4 adds the former, not the latter.

## 3. Target hierarchy (end state of the program)

| Level | SimX class | Owns — **before** | Owns — **after this program** |
|---|---|---|---|
| Cluster | `Cluster` | L2, **DxaCore** | L2 *or* **DSMEM SM-to-SM network** (mode-selected), no DXA |
| Socket (== SM) | `Socket` | L1I + dcache (`CacheCluster`) | L1I + **`UnifiedL1`** (fused L1D/SMEM, + shared-resident **`Mbarrier`** & proxy-ordering port, Task 4) + **`DxaCore`** |
| Core | `Core` | regfile + TCU + **`LocalMem`** | regfile + TCU + **SMEM *interface*** (no storage) |

`SOCKET_SIZE` is fixed to **1**, so socket == SM throughout.

## 4. Task sequence (each builds on the previous)

1. **[UnifiedL1](unified_l1_simx_proposal.md)** — fix `SOCKET_SIZE=1`; strip SMEM storage from
   the core; delete the separate dcache + `LocalMem` classes; introduce a socket-resident
   **`UnifiedL1`** (one banked SRAM, configurable cache↔scratchpad partition). *Foundation for
   everything else — the socket now owns the shared memory.*
2. **[Per-SM DXA](per_sm_dxa_simx_proposal.md)** — move `DxaCore` from cluster to socket; add a
   dcache→DXA arbiter onto the socket's `UnifiedL1` path; with `SOCKET_SIZE=1` this is a true
   per-SM TMA. *Requires Task 1's socket-resident SMEM.*
3. **[SM-to-SM network: DSMEM + TMA multicast](dsmem_smtosm_simx_proposal.md)** — a cluster-level
   SM-to-SM network **substrate** plus **two first-class consumers**: **(3a)** the network fabric
   (**pluggable topology — crossbar default, ring/mesh/torus as a study axis**) + `UnifiedL1`
   remote port + cluster memory mode **L2 / DSMEM / none**; **(3b)** **DSMEM remote
   addressing** (real cross-socket shared-memory load/store); **(3c)** **faithful TMA multicast**
   (single issuer, *one* global read, hardware fan-out into N scratchpads). *Uses Task 1's
   `UnifiedL1` and Task 2's per-socket `DxaCore`; 3c restores — and makes faithful — the
   cross-socket multicast that Task 2 defers.*
4. **[Proxy model & async-proxy instructions](proxy_model_simx_proposal.md)** — model PTX's
   two-proxy memory model (generic = LSU, async = DXA/TMA): a `proxy` tag on requests, the
   producer-side **`fence.proxy.async`**, and the byte-granular **mbarrier** completion (promoted
   from the existing `warp_barrier_t` proto-mechanism). An **additive fidelity layer on top of
   Tasks 1–3, not a correction of them**; proxies ≠ scopes. *Uses Task 1's `UnifiedL1` port
   (generic↔async crossing) and Task 2's socket DXA (the async agent).*
5. **[OpenCL feature enablement](opencl_feature_enablement_proposal.md)** — expose the new
   features (UnifiedL1 partition carveout, DXA, DSMEM, WGMMA) to **OpenCL C**, port the Task 1–4
   C++ verification kernels to OpenCL, and verify the OpenCL equivalents produce identical
   results. *The bridge to the future CUDA→OpenCL transcription effort.*

## 5. Cross-cutting acceptance criteria (a task is **not done** until all hold)

Per [`../../AGENTS.md`](../../AGENTS.md) §4 and §6:

- **Re-read `AGENTS.md` at the start of the task** and follow it throughout.
- **Builds and passes inside the Docker dev container.** Use
  [`miscs/docker/vortex-dev.sh`](../../miscs/docker/vortex-dev.sh) (`build` → `up` → `shell`),
  run from the generated `build/` tree. "Compiles on my machine" is not done.
- **C++ verification kernel(s) prove the feature** (Tasks 1–4); **OpenCL equivalents** prove it
  (Task 5). A feature with no passing kernel is incomplete. Kernels live under
  [`tests/regression/`](../../tests/regression/) (C++) and [`tests/opencl/`](../../tests/opencl/)
  (Task 5), each as `{kernel.*, main.*, common.h, Makefile}`.
- **120-second timeout cap** on every test invocation.
- **`CONFIGS` must match on both sides** — rebuild the app with the same `-DVX_CFG_*` before
  running `blackbox.sh` (see §7).
- **sw/sim isolation** ([`ci/check_sw_sim_boundary.sh`](../../ci/check_sw_sim_boundary.sh)):
  `sw/{kernel,runtime}` must not reference `sim/*` or `hw/*`, and vice-versa; shared host/ABI
  helpers go in [`sw/common/`](../../sw/common/).
- **Re-`configure`** from `build/` after editing TOML/Makefiles or adding files.

## 6. The 1:1 mapping discipline (intended future SV modules)

Each new/changed SimX class names the single RTL module it will become:

| SimX class (this program) | Future SV module | Level |
|---|---|---|
| `UnifiedL1` (`sim/simx/mem/unified_l1.{h,cpp}`) | `VX_unified_l1` (in `VX_socket.sv`) | Socket |
| `DxaCore` relocated to `Socket` | `VX_dxa_core` moved into `VX_socket.sv` | Socket |
| `SmToSmNetwork` (`sim/simx/dsmem/`, pluggable topology backend) | `VX_smtosm_network` (parameterized module family, in `VX_cluster.sv`) | Cluster |
| `Mbarrier` (shared-resident, Task 4) | `VX_mbarrier` (in `VX_unified_l1`) | Socket |
| proxy-ordering port logic (Task 4) | proxy-order block in `VX_unified_l1` | Socket |

This is why façades are banned: a wrapper has no clean SV counterpart.

## 7. Verification harness conventions (shared by all tasks)

Build + run a single test on SimX (from `build/`):

```bash
# C++ regression kernel (Tasks 1-4)
make -C tests/regression/<app> clean
CONFIGS="-DVX_CFG_NUM_THREADS=32 -D<feature flags>" make -C tests/regression/<app>
CONFIGS="-D<feature flags>" ./ci/blackbox.sh --driver=simx --app=<app> --cores=<N> --warps=64 --threads=32

# OpenCL kernel (Task 5)
make -C tests/opencl/<app> run-simx
```

Common knobs are exposed as `blackbox.sh` flags (`--clusters --cores --warps --threads
--l2cache --l3cache --debug --perf`); everything else goes through `CONFIGS="-DVX_CFG_*"`.
Baseline parameters live in [`VX_config.toml`](../../VX_config.toml) /
[`VX_types.toml`](../../VX_types.toml) — edit those only for program-wide defaults (e.g.
`SOCKET_SIZE=1`) and re-`configure`.

The **toy-H100 SM config** (from the gap analysis) is the program's reference point:
`NUM_THREADS=32`, `NUM_WARPS=64` (⇒ `ISSUE_WIDTH=4` SMSPs, 2048 threads/SM), `SOCKET_SIZE=1`,
`LMEM_LOG_SIZE=18` (256 KB), `L2_ENABLE=false`, `L3_ENABLE=true`, `EXT_TCU/DXA` enabled.

## 8. Risks & open questions (program-level)

- **Per-SM DXA breaks cross-socket multicast until Task 3.** Task 2 makes DXA per-SM; the
  existing cluster-wide multicast (one DXA reaching many cores' LMEM) no longer works across
  sockets. Single-CTA copies are fine; **faithful** cross-socket multicast — single issuer, one
  global read, hardware fan-out — is explicitly restored in **Task 3c** over the SM-to-SM
  network. Sequencing matters — see each task's risk section. The current cluster-resident DXA
  multicasts only *by accident of being shared*; the per-SM rebuild is the more faithful model,
  but it must not silently lose the **single-global-read** property (correct bytes in N
  scratchpads with N× global reads is a *failed* model).
- **Address-map churn.** Tasks 1 and 3 both touch the shared-memory address space
  (`VX_MEM_LMEM_BASE_ADDR`, `VX_CSR_LOCAL_MEM_BASE`, the WGMMA descriptor base). Keep these
  consistent across tasks.
- **SimX-as-oracle for the eventual RTL.** Per [`../../AGENTS.md`](../../AGENTS.md) §4, these
  SimX models will later anchor the RTL via trace diffing. Keep per-class trace dumps in mind
  while designing (cycle, SRAM addr+data, partition/bank, hazards).

## 9. Per-task deliverable flow

For each task, in order: (1) re-read `AGENTS.md`; (2) update that task's proposal with the
concrete SimX plan; (3) implement the SimX classes; (4) write the C++ verification kernel(s)
(OpenCL for Task 5); (5) build and run in Docker; (6) only then start the next task. When a
task is implemented and merged, promote its proposal content into
[`../designs/`](../designs/) per [`../../AGENTS.md`](../../AGENTS.md) §7.

## 10. Memory consistency — what's in and out of scope

Vortex is a weakly-ordered RISC-V GPU with **no PTX/SASS-style memory scopes**. This is **not a
blocker** for the **structural** tasks (1–3) or OpenCL (5), because the entire Hopper-SM feature
set being reproduced is **scratchpad-centric and barrier/transaction-synchronized** — the regime
where scopes stop being load-bearing for correctness. **Proxies are a separate axis from scopes,
and the proxy half is deliberately scheduled as [Task 4](proxy_model_simx_proposal.md)** (see the
reconciliation below).

- **Why it's orthogonal.** SMEM, DSMEM, the TMA/DXA destination, and the WGMMA operands are all
  **scratchpad** — non-coherent, directly addressed, no tags. There is no coherence domain to
  flush, so there is no scope to choose. Visibility is bounded **structurally** (scratchpad is
  not cached coherently) and ordered **temporally** by an explicit barrier. A scope's *radius*
  is instead carried by the **barrier + the network's span** (e.g. the cluster-scoped mbarrier
  becomes a barrier event routed over the SM-to-SM network whose reach *is* the cluster).
- **What is simpler for Tasks 1–3 (single-proxy model).** Through Task 3, Vortex is treated as
  single-proxy: the DXA write and the consuming read share one proxy, so Tasks 1–3 need **no
  `fence.proxy.async`** and the "WGMMA reads TMA-written SMEM" hazard is just a "is the tile
  present?" check gated by the lockstep owner (`VX_tcu_lockstep`). **This simplification is
  correct for the structural goal** — Tasks 1–3 rest on barriers, not proxy fences.
- **Reconciliation — the proxy model is Task 4, additive.** [Task 4](proxy_model_simx_proposal.md)
  **deliberately introduces** the two-proxy model (a `proxy` tag, a real `fence.proxy.async` for
  the generic→async crossing, byte-granular mbarrier completion) as a **fidelity layer on top of
  Tasks 1–3, not a correction of them**. So "do not port proxy machinery" applies *through Task
  3*; Task 4 is exactly where it is ported, on purpose. **Proxies ≠ scopes**: Task 4 orders
  datapath crossings; it does **not** build a scoped-visibility model.
- **The one hard obligation.** The **DSMEM cluster barrier must drain *and* fence** (Task 3b
  acceptance): every remote scratchpad write before it has landed in the peer's `UnifiedL1`, and
  post-barrier reads cannot hoist above it. This is a barrier-*implementation* property, not a
  scope feature, and it underwrites both DSMEM addressing and multicast per-destination completion.
- **Explicitly out of scope (separate, additive work).** Faithfully reproducing **scoped**-
  consistency *behavior* — scope qualifiers, scope-tree visibility rules, scope-bounded ordering,
  or mis-scope failure modes — is a **different project** (a consistency-protocol study), **not
  part of Tasks 1–5**. (The *proxy* half of the originally-parked item is now Task 4; the *scope*
  half remains parked.) It must not be folded into the program.
