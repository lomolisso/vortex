# 12 — Cache Architecture

**Scope:** the one `Cache` class that implements L1/L2/L3, and the `CacheCluster` that banks it —
configuration, the hit/miss/MSHR/replacement/write-back machinery, the flush walk, and the atomic
passthrough. Sources: [`../mem/cache.h`](../mem/cache.h), [`../mem/cache.cpp`](../mem/cache.cpp)
(the largest file in `mem/`, ~1500 lines), [`../mem/cache_cluster.h`](../mem/cache_cluster.h).
Prereq: [11](11-memory-hierarchy-overview.md).

---

## 1. One class, every level

`Cache` ([`cache.h:21`](../mem/cache.h#L21)) is parameterized by a `Config`
([`cache.h:29`](../mem/cache.h#L29)); the same class is instantiated as L1 (inside `CacheCluster`),
L2 (cluster), and L3 (processor). Its public surface is four channel arrays —
`core_req_in`/`core_rsp_out` (upstream) and `mem_req_out`/`mem_rsp_in` (downstream)
([`cache.h:70-73`](../mem/cache.h#L70)) — plus `perf_stats()` and the flush API. The behavior lives
in a pImpl (`class Impl;`, [`cache.h:91`](../mem/cache.h#L91)) in `cache.cpp`.

### 1.1 `Config` fields ([`cache.h:29-45`](../mem/cache.h#L29))

| Field | Meaning |
|---|---|
| `bypass` | when set, the cache is a transparent arbiter (used when a level is disabled, e.g. L3 off → bypass) |
| `C` / `L` / `W` | log2 of cache size / line size / word size |
| `A` / `B` | log2 of associativity (ways) / number of banks |
| `addr_width` | word-address bits |
| `num_inputs` / `mem_ports` | upstream request ports / downstream memory ports |
| `write_back` | write-back (true) vs write-through (false) |
| `write_reponse` | emit a response for writes |
| `mshr_size` | miss-status holding-register depth (outstanding-miss coalescing) |
| `latency` | hit pipeline latency (cycles) |
| `repl_policy` | `ReplPolicy{RANDOM, FIFO, PLRU}` ([`cache.h:23`](../mem/cache.h#L23)) |
| `is_llc` | this is the last-level cache (affects AMO handling, §4) |

These come from `VX_CFG_{ICACHE,DCACHE,L2,L3}_*` (doc 18); see the construction sites in
[`processor.cpp:64`](../processor.cpp#L64) (L3), `cluster.cpp` (L2), `socket.cpp` (L1).

---

## 2. Banking — `CacheCluster`

A socket's L1 isn't one `Cache` but a `CacheCluster` ([`cache_cluster.h:20`](../mem/cache_cluster.h#L20))
of `num_units` `Cache` instances fronting `num_inputs` cores. Its `core_req_in` is a **2-D** array
`[input][port]` ([`cache_cluster.h:22`](../mem/cache_cluster.h#L22)); it fans the cores' requests
across the bank units and aggregates `mem_req_out` toward L2. Flush is forwarded to every unit
([`cache_cluster.h:36`](../mem/cache_cluster.h#L36)).

Within a single `Cache`, the `B` config gives `2^B` internal banks; requests are distributed across
banks by address, and bank conflicts produce `bank_stalls` (§3).

---

## 3. The hit/miss path and PerfStats

A request entering `core_req_in` is processed each `on_tick`: index by set, probe the `A` ways for a
tag match.

- **Hit** → respond after `latency` cycles (read returns line data; write updates the line, marking
  it dirty in write-back mode).
- **Miss** → allocate an **MSHR** entry (coalescing further misses to the same line) and issue a
  fill on `mem_req_out`; when the fill returns, install the line (evicting a victim per
  `repl_policy`, writing it back first if dirty) and replay the waiting requests.
- **Bank/MSHR pressure** → `bank_stalls` / `mshr_stalls`.

All of this is counted in `PerfStats` ([`cache.h:47-68`](../mem/cache.h#L47)): `reads`, `writes`,
`read_misses`, `write_misses`, `evictions`, `bank_stalls`, `mshr_stalls`, `mem_latency` — surfaced
to the profiler (doc [19](19-profiling-and-perf-counters.md)) and summed with `+=` up the hierarchy.

---

## 4. Write-back vs write-through, and AMO

`write_back` selects the policy. The **atomic-extension invariant** (doc 02 §2.2) constrains it:
every cache *above* the LLC must be write-through, so a store can't hide from the LLC and let a
later `SC` spuriously succeed. Non-LLC atomics take a **probe-and-invalidate passthrough**: the
local line is invalidated and the op is forwarded downstream tagged `AMO_PASSTHRU_TAG_FLAG`
([`processor.cpp:96-99`](../processor.cpp#L96)), so the response routes back to `core_rsp_out`
without installing a fill. `is_llc` tells a cache it is the resolution point for atomics.

---

## 5. Flush (write-back drain)

`flush_begin()` arms a write-back eviction walk that emits every dirty line on `mem_req_out`;
`flush_done()` reports completion ([`cache.h:83-84`](../mem/cache.h#L83)). **Write-through caches
no-op** (nothing dirty to drain). `ProcessorImpl::flush_caches()` drives this inside-out (L1→L2→L3,
doc 02 §3.2) before the host reads back results, ticking the sim until every level reports done and
`inflight_count()==0`.

---

## 6. Bypass mode

When a level is configured off (e.g. `VX_CFG_L3_ENABLE=false`), `Config.bypass` is set
([`processor.cpp:65`](../processor.cpp#L65)) and the `Cache` degenerates to a transparent arbiter
that just forwards `core_req_in`→`mem_req_out` (and the LLC role shifts down a level). This is how
the same wiring supports L1-only, L1+L2, or L1+L2+L3 builds without code changes — relevant to the
Toy-H100 config which uses L3-as-global-L2 (proposals).

**Next:** [13 — Local Memory & Scratchpad](13-local-memory-scratchpad.md).
