# Current L1D & SMEM Architecture in SimX

> **Scope.** This document records the *baseline* architecture of the two on-chip SRAM
> structures as modeled in the SimX simulator: the **L1 Data Cache (L1D / dcache)** and the
> **Local / Shared Memory (LMEM / SMEM)**. All numbers are reported for the **as-shipped
> `VX_config.toml` defaults** (RV32, 4 threads / 4 warps, single core / socket / cluster).
> It is descriptive only — a reference for the toy-H100 *Task 1* work
> (`docs/proposals/socket_smem_simx_proposal.md`), which relocates the SMEM (LMEM) up to the
> socket to sit alongside the L1D and L1I, **keeping them as separate structures** (no fusion).
>
> Every value is cited as `file:line`. Background prose lives in
> `sim/simx/docs/12-cache-architecture.md` and `sim/simx/docs/13-local-memory-scratchpad.md`.

---

## 0. Baseline configuration assumptions

Every derived number below follows from these default knobs:

| Knob | Value | Source |
|---|---|---|
| `VX_CFG_XLEN` | **32** (default = first of `[32, 64]`); `XLENB = 4` | `VX_config.toml:324`; `constants.h:44` |
| `NUM_CLUSTERS` / `NUM_CORES` / `SOCKET_SIZE` | 1 / 1 / 1 | `VX_config.toml:2-4` |
| `NUM_THREADS` / `NUM_WARPS` | 4 / 4 | `VX_config.toml:66,65` |
| `SIMD_WIDTH` = `NUM_THREADS` | 4 | `VX_config.toml:72` |
| `NUM_LSU_LANES` = `SIMD_WIDTH`, `NUM_LSU_BLOCKS` | 4, 1 | `VX_config.toml:112-113` |
| `MEM_BLOCK_SIZE` / `L1_LINE_SIZE` | 64 B / 64 B | `VX_config.toml:79,82` |
| `L2_ENABLE` / `L3_ENABLE` | false / false → **L1D is the LLC** | `VX_config.toml:13-14` |

Derived constants (from `constants.h:52-59` and `VX_config.toml:115,196,198`):

```
LSU_WORD_SIZE     = XLEN/8                         = 4 B
LSU_NUM_REQS      = NUM_LSU_BLOCKS * NUM_LSU_LANES = 1 * 4 = 4
LSU_LINE_SIZE     = min(NUM_LSU_LANES*XLENB, 64)   = min(16, 64) = 16 B
DCACHE_WORD_SIZE  = LSU_LINE_SIZE                  = 16 B   (coalesced access granule)
DCACHE_CHANNELS   = ceil(NUM_LSU_LANES*XLENB / DCACHE_WORD_SIZE) = ceil(16/16) = 1
DCACHE_NUM_REQS   = NUM_LSU_BLOCKS * DCACHE_CHANNELS = 1
DCACHE_NUM_BANKS  = 2^clog2(min(DCACHE_NUM_REQS,16)) = 1
NUM_DCACHES       = ceil(SOCKET_SIZE/4)            = 1
L1_MEM_PORTS      = min(DCACHE_NUM_BANKS, PLATFORM_MEMORY_NUM_BANKS) = min(1,2) = 1
```

---

## 1. L1 Data Cache (L1D / dcache)

Modeled by the single parameterized `Cache` class (`sim/simx/mem/cache.cpp`), wrapped per
socket by `CacheCluster` and instantiated at `sim/simx/socket.cpp:54-70`.

`Cache::Config` field values (verified at `socket.cpp:54-69`):
`C = log2(16384) = 14`, `L = log2(64) = 6`, `W = log2(16) = 4`, `A = log2(4) = 2`,
`B = log2(1) = 0`, `addr_width = XLEN = 32`.

| Parameter | Value | Source |
|---|---|---|
| Total size | **16 KiB** (16384 B) | `VX_config.toml:182`; `socket.cpp:56` |
| Associativity | **4-way set-associative** | `VX_config.toml:183`; `socket.cpp:59` |
| Number of sets | **64** = 16384 / (4 ways · 64 B) | derived; `cache.cpp:50,56` |
| Banks | **1** | `VX_config.toml:196`; `socket.cpp:60` |
| Cache line size | **64 B** | `VX_config.toml:82`; `socket.cpp:57` |
| Cache "word" / access granule | **16 B** (= coalesced `LSU_LINE_SIZE`, *not* the 4 B CPU word) | `constants.h:57`; `socket.cpp:58` |
| Address width | **32 bits** | `socket.cpp:61` |
| Ports / requests per cycle | **1** (`DCACHE_NUM_REQS`) | `constants.h:59`; `socket.cpp:62` |
| Mem ports to next level | **1** (`L1_MEM_PORTS`) | `VX_config.toml:198`; `socket.cpp:63` |
| Pipeline latency | **1 cycle** | `socket.cpp:67` |
| Write policy | **Write-through** (`WRITEBACK = 0`) | `VX_config.toml:184`; `socket.cpp:64` |
| Replacement policy | **FIFO** | `VX_config.toml:186`; `socket.cpp:68` |
| MSHR size | **16** entries | `VX_config.toml:187`; `socket.cpp:66` |
| MREQ / MRSQ / CRSQ queues | **4 / 4 / 2** | `VX_config.toml:188-190` |
| Is last-level cache | **Yes** (L2 & L3 disabled) | `socket.cpp:69` |
| Instances | **1 per socket**, shared by ≤4 cores/cache; here 1 dcache ↔ 1 core | `VX_config.toml:193` |

### Address breakdown

From `params_t` (`cache.cpp:48-74`), with `C=14, L=6, W=4, A=2, B=0` and `addr_width=32`:

```
offset_bits = L - W           = 2   → 4 access-granules (16 B each) per 64 B line
index_bits  = C - (L + A + B) = 6   → 64 sets

bit:  31 ............... 12 | 11 ...... 6 | (none) | 5 .. 4 | 3 ........ 0
       Tag                  |  Set index  |  Bank  |  Word  |  byte-in-granule
```

- **Word-select:** bits **[5:4]** (`cache.cpp:61-62`)
- **Bank-select:** **none** — `B=0`, so `bank_end < bank_start` and `addr_bank_id()` returns 0 (`cache.cpp:65-66,77-82`)
- **Set/index:** bits **[11:6]** (`cache.cpp:69-70`)
- **Tag:** bits **[31:12]** (`cache.cpp:73-74`)

### Behavior notes
- **Interleaving** across banks is address-based (`addr_bank_id`); with a single bank there is no
  spread — all traffic lands in bank 0.
- **MSHR** coalesces multiple in-flight misses to the same line; depth = 16 (`cache.cpp` miss path).
- **Write-through invariant:** because all caches above the LLC are write-through, AMOs take a
  probe/passthrough path at non-LLC levels — moot here since the L1D *is* the LLC.
- On teardown the processor performs a **flush walk** (inside-out) to drain dirty lines; with a
  write-through L1D there is nothing dirty to write back.

---

## 2. Local Memory / Shared Memory (LMEM / SMEM)

Modeled by `LocalMem` (`sim/simx/mem/local_mem.cpp`), instantiated **per core** at
`sim/simx/core.cpp:122-131`. It is a flat, banked SRAM scratchpad — no tags, no cache logic, always
hits.

`LocalMem::Config` field values (verified at `core.cpp:125-131`):
`capacity = 1<<14 = 16384`, `line_size = LSU_WORD_SIZE = 4`,
`num_reqs = LSU_NUM_REQS (+TCU +DXA) = 4`, `B = log2(4) = 2`, `write_response = false`.

| Parameter | Value | Source |
|---|---|---|
| Total size | **16 KiB** (`1 << LMEM_LOG_SIZE`, LOG_SIZE = 14) | `VX_config.toml:229`; `core.cpp:126` |
| Banks | **4** (= `NUM_LSU_LANES`) | `VX_config.toml:230`; `core.cpp:129` |
| Bank size | **4 KiB** each | derived |
| Word size | **4 B** (`LSU_WORD_SIZE`) | `constants.h:52`; `core.cpp:127` |
| Interleave granule | **4 B** word, round-robin across 4 banks | `local_mem.cpp:45-51` |
| Bank-select | `bank = (addr >> 2) & 3` → bits **[3:2]** | `local_mem.cpp:48-51` |
| Ports / requests per cycle | **4** (one per LSU lane; +1 each if TCU / DXA enabled) | `core.cpp:124,128` |
| Associativity / tags / MSHR | **None** — flat direct SRAM, always hits | `local_mem.cpp` (no tag logic) |
| Replacement policy | **N/A** | — |
| Write policy | Immediate byte-enabled write; loads always respond, stores respond only if `write_response` or `MEM_FLAG_STRSP` | `local_mem.cpp:75-98` |
| Arbitration | `MemCrossBar`, **Priority** arbiter, 4 reqs → 4 banks; conflicts ⇒ `bank_stalls` | `local_mem.cpp:47,112` |
| RAM access granule | Reads/writes operate on **64 B** (`MEM_BLOCK_SIZE`) lines with byte-enables, despite `line_size = 4` | `local_mem.cpp:76-93` |
| Instances | **1 per core** (not shared); here 1 total | `core.cpp:125` |

### Address & banking
- Addresses are masked to the 14-bit local space (`to_local_addr`, `local_mem.cpp:32-34`,
  `addr_bits_ = log2ceil(capacity) = 14`).
- Bank index = `(addr >> log2(line_size)) & (num_banks-1) = (addr >> 2) & 3` (`local_mem.cpp:48-51`)
  — consecutive 4 B words map to consecutive banks (classic conflict-free strided access pattern).
- Each bank request, when serviced, reads/writes a full 64 B block from the backing `RAM` using the
  request's byte-enable mask (`local_mem.cpp:76-93`); a backdoor 4 B functional read is available via
  `read_word()` (`local_mem.cpp:116-121`).

---

## 3. How the two are wired & addressed

- **Routing.** `get_addr_type(addr)` / `LocalMemSwitch` (`sim/simx/mem/local_mem_switch.*`, wired in
  `core.cpp:133-164`) classifies each LSU access:
  - **Shared range** `[VX_MEM_LMEM_BASE_ADDR, +2^LMEM_LOG_SIZE)` → `LocalMem` (uncached, direct).
  - **Global** → `MemCoalescer` (`core.cpp:116-119`) → `LsuMemAdapter` (`core.cpp:139-144`) → dcache.
- **Coalescing.** The `MemCoalescer` merges per-lane 4 B accesses into one 16 B (`DCACHE_WORD_SIZE`)
  request — this is *why* the cache's access granule is 16 B rather than the 4 B CPU word
  (`core.cpp:119`).

### Side-by-side comparison

| Attribute | L1D cache | Local / Shared Memory |
|---|---|---|
| Total size | 16 KiB | 16 KiB |
| Banks | 1 | 4 (4 KiB each) |
| Line size | 64 B | 64 B (RAM block granule) |
| Access granule / word | 16 B (coalesced) | 4 B |
| Associativity | 4-way (64 sets) | flat, no tags |
| Ports / cycle | 1 | 4 |
| Latency | 1-cycle pipeline | functional, + crossbar arbitration |
| Write policy | write-through | immediate byte-enabled |
| Miss handling | 16-entry MSHR, FIFO repl | none (always hits) |
| Last-level cache | yes (L2/L3 off) | n/a |
| Instances | 1 per socket (≤4 cores/cache) | 1 per core |

---

## 4. Source map

Open these for the authoritative definitions:

- `VX_config.toml` — knobs and derived expressions: `[l1cache]` lines 172-198, `[lmem]` lines 228-230,
  `[memory]` lines 78-90, platform lines 2-14, pipeline/lsu lines 65-117, `VX_CFG_XLEN` line 324
- `sim/simx/constants.h:44-59` — `XLENB`, `LSU_*`, `DCACHE_*`, `NUM_SOCKETS`
- `sim/simx/socket.cpp:54-70` — dcache `Cache::Config` instantiation
- `sim/simx/core.cpp:116-164` — coalescer, `LocalMem::Config`, lmem switch / adapter wiring
- `sim/simx/mem/cache.cpp:48-74` — `params_t` address breakdown; rest of file: banking, MSHR, repl
- `sim/simx/mem/local_mem.cpp:23-122` — banked scratchpad, crossbar interleaving, write/respond rules
- Background prose: `sim/simx/docs/12-cache-architecture.md`, `sim/simx/docs/13-local-memory-scratchpad.md`
