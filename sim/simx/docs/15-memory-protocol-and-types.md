# 15 — Memory Protocol & Types

**Scope:** the packet types that travel every memory wire — `MemReq`/`MemRsp`/`MemFlags`,
the per-warp `LsuReq`/`LsuRsp`, the `hart_id` routing encoding, and the `AddrType` classification.
These are the lingua franca between the LSU, the switches, the caches, and DRAM. Source:
[`../types.h`](../types.h). Prereqs: [08](08-lsu-and-memory-ops.md),
[11](11-memory-hierarchy-overview.md).

---

## 1. `MemReq` — a single cache/memory request ([`types.h:1185`](../types.h#L1185))

```cpp
struct MemReq {
  MemOp    op;        // LD / ST / FLUSH / AMO_*  (memop_is_write / _amo helpers)
  uint64_t addr;
  std::shared_ptr<mem_block_t> data;   // TLM payload (line-sized; carries real bytes)
  uint64_t byteen = 0;                 // per-byte write enable
  uint32_t tag;                        // request id (usually the instruction uuid)
  uint32_t hart_id;                    // hardware routing id (§4)
  uint64_t uuid;                       // debug correlation id
  MemFlags flags;                      // §3
  bool     is_write() const;
  AddrType addr_type() const { return get_addr_type(addr); }   // (types.h:1212)
};
```

This is what flows on every `core_req_in`/`mem_req_out` channel (docs 11–12). The **`data`** is a
shared `mem_block_t` — a full cache-line block (pooled in
[`../mem/mem_block_pool.h`](../mem/mem_block_pool.h)); loads return it, stores carry it. The
`operator<<` ([`types.h:1214`](../types.h#L1214)) prints `op/addr/type/tag/hart_id/flags (#uuid)` —
that's the line you see in memory `--debug` traces (doc 20).

## 2. `MemRsp` — the response ([`types.h:1228`](../types.h#L1228))

`{tag, hart_id, uuid, data}` — minimal: it echoes the `tag`/`hart_id`/`uuid` so the requester can
match it, and carries `data` for loads/AMO returns. **Matching is by tag** (LSU) or `hart_id`
(cache demux) — see §4.

---

## 3. `MemFlags` — the sideband ([`types.h:411`](../types.h#L411))

A 32-bit union (`static_assert`ed to 4 bytes, [`types.h:448`](../types.h#L448)):

| Bit | Field | Meaning |
|---|---|---|
| 0 | `strsp` | store-response enabled (opt-in write ack) |
| 1 | `io` | uncacheable I/O — bypasses caches to `Memory` |
| 2 | `local` | targets the LMEM port (mirrors `AddrType::Shared`) |
| 3 | `amo_unsigned` | AMO MIN/MAX signedness |
| 4 | `dxa_notify_done` | (DXA builds) this LMEM-DMA write is a barrier-completion point |
| 5..20 | `dxa_notify_bar_id` | (DXA builds) the RAW kernel barrier handle to release |

The `dxa_notify_*` fields ([`types.h:428-429`](../types.h#L428)) are how an async copy's final write
releases a barrier — snooped by the per-core `tx_callback` (doc 09 §5). The `bar_id` is the *raw*
encoded handle (`cta_local_id | bar_no<<8`); multicast offsets the low byte per CTA so each
receiver's barrier fires independently.

---

## 4. `hart_id` — the routing key ([`types.h:1177`](../types.h#L1177))

```cpp
inline uint32_t make_hart_id(uint32_t cid, uint32_t wid, uint32_t tid) {
  return (cid << (LOG_WARPS + LOG_THREADS)) | (wid << LOG_THREADS) | tid;
}
```

A globally-unique `(core, warp, thread)` id with **`tid` in the low bits** — the comment
([`types.h:1174-1176`](../types.h#L1174)) explains why: lower bits carry the most entropy, so the
small per-bank reservation tables (cache MSHR, coalescer) see fewer hash collisions. Responses
carry `hart_id` back so the cache/coalescer can recover `(cid, wid, tid)` and route data to the
right lane. **Don't confuse it with `tag`:** `tag` is application/request-specific (typically the
instruction `uuid`); `hart_id` is the hardware routing id.

---

## 5. `LsuReq` / `LsuRsp` — the per-warp form ([`types.h:1073`](../types.h#L1073), [`:1123`](../types.h#L1123))

Upstream of the cache (between LSU, `LocalMemSwitch`, `MemCoalescer`), requests are **per-warp
vectors**, not single beats:

```cpp
struct LsuReq {
  MemOp op;  MemFlags flags;
  std::vector<uint64_t> addrs;                          // per-lane address
  std::vector<std::shared_ptr<mem_block_t>> data;       // per-lane store data
  std::vector<uint64_t> byteen;
  BitVector<> mask;                                     // active lanes
  std::vector<uint32_t> tids;
  uint32_t tag, cid, wid;  uint64_t uuid;
};
struct LsuRsp { BitVector<> mask; uint64_t tag; uint32_t cid; uint64_t uuid;
                std::vector<std::shared_ptr<mem_block_t>> data; };
```

The `LsuMemAdapter` (doc 11 §2.1) is what unpacks a `LsuReq`'s active lanes into individual
`MemReq`s for the cache (assigning `hart_id = make_hart_id(cid, wid, tid)`) and repacks responses
into an `LsuRsp`. AMO mapping helpers (`amo_to_memop`, `amo_is_unsigned`,
[`types.h:1152-1172`](../types.h#L1152)) collapse RVA atomics onto `MemOp::AMO_*` + `amo_unsigned`.

---

## 6. `AddrType` & `get_addr_type` ([`types.h:812`](../types.h#L812))

`enum class AddrType { Global, Shared, IO }`; `get_addr_type(addr)`
([`types.h:818`](../types.h#L818)) returns `IO` for the IO window, `Shared` for
`[VX_MEM_LMEM_BASE_ADDR, +2^VX_CFG_LMEM_LOG_SIZE)`, else `Global`. This is the single decision
point that the `LocalMemSwitch` (doc 13) and the LSU's `flags.local` derive from; the Toy-H100
DSMEM roadmap (`docs/proposals/dsmem/`) keeps this shared/global
classification unchanged (the SMEM target is core-resident; DSMEM widens the shared window) — it does
**not** fold scratchpad and cache into one array.

---

## 7. Why this is the "lingua franca"

Every memory `SimObject` — LSU, switch, coalescer, cache, MMU, DRAM — speaks `MemReq`/`MemRsp` (or
the per-warp `LsuReq`/`LsuRsp` upstream of the adapter). Learn these six types and you can read any
memory-path `on_tick()`: it pops a request, inspects `op`/`addr`/`flags`/`hart_id`, and either
services it or forwards a transformed copy downstream.

**Next:** [16 — Arbitration & Crossbars](16-arbitration-and-crossbars.md).
