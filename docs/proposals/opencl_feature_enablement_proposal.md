# Task 5 — OpenCL Feature Enablement & Verification — SimX Proposal

**Status:** proposal (pre-implementation). **Scope:** software stack
([`sw/`](../../sw/), [`tests/opencl/`](../../tests/opencl/)) + SimX validation. Part of the
[Toy-H100 SimX Roadmap](toy_h100_simx_roadmap_proposal.md); this is **Task 5**, the **last**
task. It depends on Tasks 1–4 being implemented and C++-verified.

**Goal:** expose the toy-H100 hardware features — **socket-level SMEM** (via `__local`),
**DXA** (TMA), **DSMEM**, and **WGMMA** — to **OpenCL C**, then port the Task 1–4 C++
verification kernels to OpenCL and confirm the OpenCL equivalents produce identical results in
SimX. This is also the deliberate bridge to the future **CUDA→OpenCL workload transcription**
effort.

---

## 1. Why this is last, and why it matters

Tasks 1–4 are verified with **C++ kernels** because that is the only model that can currently
reach DXA/WGMMA/DSMEM (the intrinsics live in
[`sw/kernel/include/vx_dxa.h`](../../sw/kernel/include/vx_dxa.h),
[`vx_tensor.h`](../../sw/kernel/include/vx_tensor.h),
[`vx_barrier.h`](../../sw/kernel/include/vx_barrier.h)). **OpenCL C kernels today get only
standard OpenCL 1.2** via POCL — no path to these features. There is **no CUDA front-end** in
the tree (only doc-comment references; `vortex2.h` lists CUDA/HIP/Vulkan as *future* upper-layer
translators). Doing this last means every feature already has a known-good C++ oracle to diff
the OpenCL result against.

## 2. Current software stack (as-built)

- **OpenCL path:** [`tests/opencl/<app>/`](../../tests/opencl/) = `{kernel.cl, main.cc,
  common.h, Makefile}`; POCL compiles `.cl` → RISC-V (LLVM `+xvortex`); run via
  `make -C tests/opencl/<app> run-simx` or `blackbox.sh --driver=simx --app=<app>`. Existing
  kernels (vecadd, sgemm, lbm, kmeans, …) use only standard OpenCL builtins.
- **C++ kernel path:** [`tests/regression/<app>/`](../../tests/regression/); reaches all
  features through the `vx_*` headers.
- **Isolation rule** ([`../../AGENTS.md`](../../AGENTS.md) §6): `sw/*` must not reference
  `sim/*`/`hw/*`; shared host↔kernel ABI goes in [`sw/common/`](../../sw/common/).

## 3. Exposure mechanism (per feature)

The principle ([`../../AGENTS.md`](../../AGENTS.md) §5): **align with mainstream OpenCL** — use
the extension mechanism (`cl_khr_*` where one exists, else a vendor `cl_vortex_*` extension),
not one-off abstractions. Each OpenCL builtin lowers to the **same device intrinsic** the C++
headers already emit, so the SimX hardware path is unchanged.

| Feature | OpenCL surface | Lowers to | Notes |
|---|---|---|---|
| **Socket-level SMEM** (Task 1) | `__local` already maps to the scratchpad — now socket-resident; no new surface needed | the existing shared-memory intrinsics / `__local` address space | `__local` targets the relocated SMEM; `__global` exercises the **separate** socket dcache (no fused carveout — Task 1 keeps L1D and SMEM distinct) |
| **DXA / TMA** (Task 2) | `cl_vortex_async_copy` builtins (`vx_dxa_issue_*` equivalents, **incl. the multicast variant**) + host descriptor programming | the `vx_dxa.h` issue intrinsics + `dxa.h` host API | mirror the existing C++ issue/descriptor surface; multicast covers Task 3c |
| **DSMEM remote addressing** (Task 3b) | a work-group-cluster extension (cluster query + remote-shared pointer builtins) | the DSMEM (peer-rank, offset) addressing + `group_barrier` | OpenCL analog of CUDA thread-block clusters / `cluster.map_shared_rank` |
| **TMA multicast** (Task 3c) | the multicast form of `cl_vortex_async_copy` + the cluster mask | the per-SM DXA multicast over the SM-to-SM network | must preserve the **single-global-read** property end-to-end |
| **Proxy fence + mbarrier** (Task 4) | a `fence.proxy.async` builtin + the async-copy completion (`mbarrier` query/wait) | `vx_fence_proxy_async` + byte-granular mbarrier | required for a correct generic→async handoff when OpenCL drives the async copy |
| **WGMMA** (compute) | `cl_vortex_wmma` builtins or a tensor extension | `vx_tensor.h` `wgmma_context` | optional here — not part of the structural hierarchy changes, but listed for completeness |

Implementation lands in the **POCL/Vortex builtin library** (the Vortex POCL fork's builtin
lowering) and thin OpenCL-visible headers; the underlying device intrinsics stay in
`sw/kernel/include`. Any host↔kernel ABI (e.g. the DXA descriptor layout, cluster dims) that both
the OpenCL host and kernel need goes in [`sw/common/`](../../sw/common/).

## 4. CUDA → OpenCL → Vortex mapping (forward-looking)

This sets up the future transcription effort; building the OpenCL surface above makes it
mechanical:

| CUDA / Hopper | OpenCL surface (this task) | Vortex (Tasks 1–4) |
|---|---|---|
| SM | work-group on one core | core (== socket == SM) |
| SMSP | sub-group | issue slot (`ISSUE_WIDTH`) |
| `__shared__` / SMEM | `__local` | socket `LocalMem` (SMEM) (Task 1) |
| Thread-Block Cluster | work-group-cluster extension | CTA cluster + `SmToSmNetwork` |
| DSMEM (`cluster.map_shared_rank`) | remote-shared pointer builtin | DSMEM remote addressing (Task 3) |
| TMA (`cp.async.bulk`) | `cl_vortex_async_copy` | per-SM DXA (Task 2) |
| `mbarrier` + `fence.proxy.async` | async-copy completion + a fence builtin | byte-granular mbarrier + proxy fence (Task 4) |
| `wgmma` | `cl_vortex_wmma` | per-core TCU (WGMMA) |

## 5. Verification — port the Task 1–4 kernels to OpenCL

For each C++ verification kernel, author the OpenCL equivalent under
[`tests/opencl/`](../../tests/opencl/) and assert **identical numerical output** and that the
feature is exercised (same perf-CSR movement as the C++ oracle):

| C++ oracle (Tasks 1–4) | OpenCL equivalent (Task 5) | Exercises |
|---|---|---|
| a shared-mem+global regression kernel | `opencl/socket_smem` | `__local` (socket SMEM) + `__global` (separate dcache) |
| `regression/dxa_copy` | `opencl/dxa_copy` | per-SM DXA via `cl_vortex_async_copy` |
| `regression/dsmem_pingpong` | `opencl/dsmem_pingpong` | cross-socket DSMEM (3b) via cluster extension |
| `regression/dxa_multicast` | `opencl/dxa_multicast` | faithful TMA multicast (3c) — assert single global read, not just bytes |
| `regression/proxy_fence` | `opencl/proxy_fence` | generic→async `fence.proxy.async` (Task 4) — incl. the negative no-fence variant |
| `regression/async_push_mbar` | `opencl/async_push_mbar` | byte-granular mbarrier completion (Task 4) over the async push |

```bash
# OpenCL equivalents on SimX (toy-H100 SM shape)
make -C tests/opencl/socket_smem run-simx
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" make -C tests/opencl/dxa_copy run-simx
CONFIGS="-DVX_CFG_CLUSTER_MEM_MODE=DSMEM" make -C tests/opencl/dsmem_pingpong run-simx
# or via blackbox
CONFIGS="-DVX_CFG_EXT_DXA_ENABLE" ./ci/blackbox.sh --driver=simx --app=dxa_copy --cores=1 --warps=64 --threads=32
```

Pass criteria: the OpenCL result matches the C++ oracle bit-for-bit (or within the kernel's
documented tolerance), within the 120 s cap, built inside the Docker dev container.

## 6. Risks

- **POCL fork maintenance.** Builtin lowering lives in the Vortex POCL fork; coordinate the
  extension headers and the lowering so they stay in lockstep. Treat any temporary lowering
  shim as a labeled patch with a follow-up ([`../../AGENTS.md`](../../AGENTS.md) §3).
- **Extension-naming discipline.** Follow OpenCL extension conventions (`cl_khr_*` if a KHR
  extension fits the semantics, else `cl_vortex_*`); don't invent a non-standard API surface.
- **Isolation.** Keep all of this in `sw/*`/`tests/*` — no references into `sim/*`/`hw/*`
  ([`ci/check_sw_sim_boundary.sh`](../../ci/check_sw_sim_boundary.sh)).
- **WGMMA scope.** The tensor OpenCL surface is optional for the toy-H100 hierarchy goal;
  prioritize socket-level SMEM / DXA / DSMEM (the Task 1–3 features). Defer WGMMA-in-OpenCL if time-boxed.

## 7. Proposed but not yet implemented (beyond this task)

- A full **CUDA front-end** (or HIP via chipStar, already in the toolchain) feeding the OpenCL
  surface — the actual workload transcription, the motivating future effort.
- Vulkan / other upper-layer translators targeting the same intrinsics
  (`vortex2.h` future-translator note).
