# 17 — Extension Engines (DXA, TCU, and Graphics)

**Scope:** the optional accelerators — the **DXA** async-copy engine, the **TCU** tensor unit, and
the graphics trio (**TEX/OM/RASTER**) — how they're gated, where they live, how the SFU dispatches
to them, and how they touch memory and barriers. Sources: [`../dxa/`](../dxa), [`../tcu/`](../tcu),
[`../tex/`](../tex), [`../om/`](../om), [`../raster/`](../raster), [`../sfu_unit.h`](../sfu_unit.h),
[`../cluster.h`](../cluster.h). Prereqs: [07](07-functional-units.md) (SFU dispatch),
[09](09-barriers-and-sync.md), [13](13-local-memory-scratchpad.md).

---

## 1. The extension model

Every extension is **gated by a `VX_CFG_EXT_*` macro** (a MISA bit) and follows one of two
attachment patterns:

- **Per-core** — the **TCU** is a functional unit *inside the core* (`FuncUnit<NUM_TCU_BLOCKS>`).
- **Cluster-shared** — **DXA/TEX/OM/RASTER** live at the **cluster** (doc 02 §4.1); a per-core
  helper sub-unit *owned by the `SfuUnit`* (doc 07 §4) posts requests over an outbound `SimChannel`
  that the `Cluster` binds to the shared engine, which returns results via a response channel or a
  barrier event.

The dispatch is uniform: the decoder routes the op to the SFU (or to the TCU's `fu_type`); the SFU
demuxes by `op_type` to the right sub-unit (doc 07 §4.3, the "fire-and-wait" path).

---

## 2. DXA — async global→shared copy

**Per-core front (`DxaUnit`, [`../dxa/dxa_unit.h`](../dxa/dxa_unit.h)).** A plain helper owned by
`SfuUnit`. `process(trace)` ([`dxa_unit.h:54`](../dxa/dxa_unit.h#L54)) lane-decodes a `DxaType`
trace into a `DxaReq` ([`dxa_unit.h:27`](../dxa/dxa_unit.h#L27)) — `{desc_slot, bar_id (raw),
cta_mask, smem_addr, coords[5]}` — and pushes it on the SFU's `dxa_req_out`
([`sfu_unit.h:59`](../sfu_unit.h#L59)); on channel backpressure it returns `nullptr` to retry next
cycle.

**Cluster engine (`DxaCore`, [`../dxa/dxa_core.h`](../dxa/dxa_core.h)).** Cluster-shared
([`dxa_core.h:30`](../dxa/dxa_core.h#L30)):

- `dxa_req_in[]` — one per core in the cluster ([`dxa_core.h:53`](../dxa/dxa_core.h#L53)); the
  `Cluster` binds each core's `dxa_req_out` here.
- It dispatches to `VX_CFG_NUM_DXA_UNITS` workers, reads GMEM through `gmem_req_out`/`gmem_rsp_in`
  arbitrated by `gmem_arb_` into L2 ([`dxa_core.h:57-59`](../dxa/dxa_core.h#L57)), and writes LMEM
  through **per-core direct** `lmem_req_out` channels ([`dxa_core.h:63`](../dxa/dxa_core.h#L63),
  write-only).
- **Completion** rides on the LMEM write: the final write sets `flags.dxa_notify_done` +
  `dxa_notify_bar_id`, and the per-core `tx_callback` (doc 09 §5, [`cluster.cpp:160`](../cluster.cpp#L160))
  releases the barrier — **per destination CTA** under multicast (`cta_mask>1`, `bar_id + cta`).
- Descriptors are programmed by the host over DCRs (`dcr_write`, [`dxa_core.h:68`](../dxa/dxa_core.h#L68)).
- `PerfStats`: `transfers, gmem_reads, gmem_dedup, lmem_writes, total_latency`.

The full design rationale (descriptor format, K-major transpose, multicast) is in
[`../../../docs/designs/dxa_async_copy_multicast.md`](../../../docs/designs/dxa_async_copy_multicast.md);
the Toy-H100 proposals relocate this engine to the socket.

---

## 3. TCU — the tensor unit (WMMA / WGMMA)

**Per-core functional unit (`TcuUnit`, [`../tcu/tcu_unit.h`](../tcu/tcu_unit.h)),**
`FuncUnit<VX_CFG_NUM_TCU_BLOCKS>` ([`tcu_unit.h:48`](../tcu/tcu_unit.h#L48)). It implements
`wmma` (warp-level MMA), `wgmma` (warpgroup MMA, with A/B shared-memory descriptors), and
`meta_store` (sparse metadata) ([`tcu_unit.h:72-103`](../tcu/tcu_unit.h#L72)).

- **Micro-op expansion:** a macro MMA is cracked into a stream of FEDP micro-ops by `TcuUopGen`
  ([`tcu_unit.h:32`](../tcu/tcu_unit.h#L32)) inside the per-warp `Sequencer` (doc 04 §4 / doc 06 §4).
- **Operand staging — `TcuTbuf`** ([`../tcu/tcu_tbuf.h`](../tcu/tcu_tbuf.h)): the tile-buffer
  subsystem owns the A/B operand **line caches** — `abuf × Q` (per-block A tiles) + `bbuf × 1`
  (TB-shared B tile) — and arbitrates them onto a **single LMEM port pair** (`lmem_req_out`/
  `lmem_rsp_in`, [`tcu_tbuf.h:42`](../tcu/tcu_tbuf.h#L42)) that `Core` binds to `LocalMem` port
  `[LSU_NUM_REQS]` (doc 13 §2). The consumer protocol is `plan_a`/`plan_b` → wait for
  `ready_a`/`ready_b` → `read_a`/`read_b` ([`tcu_tbuf.h:49-56`](../tcu/tcu_tbuf.h#L49)); line caches
  are address-keyed, giving cross-WGMMA reuse (`tbuf_cache_hits`).
- **Functional read backdoor:** sparse `TCU_LD` uses `LocalMem::read_word` (doc 13 §1.1) to fetch
  metadata during `execute()` without the async pipeline.
- `PerfStats`: `latency, tbuf_stalls, tbuf_cache_hits, lmem_reads`
  ([`tcu_unit.h:54-67`](../tcu/tcu_unit.h#L54)).

Design rationale (formats, tile geometry, sparsity, the lockstep gate) is in
[`../../../docs/designs/tensor_core_wgmma_engine.md`](../../../docs/designs/tensor_core_wgmma_engine.md).

---

## 4. Graphics engines (TEX / OM / RASTER)

All three are cluster-shared, gated, and reached through `SfuUnit` sub-units bound to cluster-level
bus arbiters; each cluster engine has its own read-only/RMW cache that arbitrates into L2 (doc 02
§4.1, the L2-arbiter rows):

| Engine | Cluster core | Cache | SFU channels | Notes |
|---|---|---|---|---|
| **TEX** (texture sampler) | `TexCore` | `tcache` | `tex_req_out` / `tex_rsp_in` ([`sfu_unit.h:65`](../sfu_unit.h#L65)) | fire-and-wait: TexCore holds the trace until the filtered texel returns |
| **OM** (output merger) | `OmCore` | `ocache` | `om_req_out` ([`sfu_unit.h:73`](../sfu_unit.h#L73)) | **no response** — async read-modify-write through `ocache` |
| **RASTER** | `RasterCore` | `rcache` | `raster_req_out` / `raster_rsp_in` ([`sfu_unit.h:79`](../sfu_unit.h#L79)) | a `RasterRsp` latches `RasterCsrs` into `CsrUnit` (doc 10 §3, [`csr_unit.h:29`](../csr_unit.h#L29)) for the kernel to read |

These exist for the graphics pipeline (`tests/graphics`); they share the same dispatch/arbitration
machinery as the compute extensions and are otherwise orthogonal to the GPGPU compute path.

---

## 5. The common shape

Front sub-unit (in SFU) → outbound `SimChannel` → cluster engine (shared) → memory (its own
cache→L2) → completion (barrier event for DXA, response channel for TEX/RASTER, fire-and-forget for
OM). The TCU is the exception: it's a per-core FU with its own LMEM tile-buffer. To **add** one of
these, see doc [22](22-howto-add-features.md) §2.

**Next:** [18 — Configuration & Build](18-configuration-and-build.md).
