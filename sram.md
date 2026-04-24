# Vortex GPGPU — Memory Macros, Sharing Hierarchy, and SRAM Counts

---

## Section 1 — RTL Deep Dive

### 1.1 Hierarchy at a glance

```
Vortex  ─┐
         │  (top, optional L2 / L3)
         ▼
    VX_cluster    ×NUM_CLUSTERS  ──► VX_cache_wrap (L2)  (optional, `L2_ENABLE`)
         ▼
    VX_socket     ×NUM_SOCKETS   ──► VX_cache_cluster ×NUM_ICACHES, ×NUM_DCACHES
         ▼
    VX_core       ×SOCKET_SIZE   ──► VX_mem_unit ──► VX_local_mem
                                 ──► VX_issue_slice ×ISSUE_WIDTH
                                         └─ VX_operands ──► VX_opc_unit ×NUM_OPCS
                                                                  └─ GPR bank ×NUM_GPR_BANKS
```

The relevant knobs (all from `hw/rtl/VX_config.vh`):

| Knob | Default | Meaning |
|---|---|---|
| `NUM_CLUSTERS` | 1 | clusters inside one Vortex top |
| `NUM_CORES`    | 1 | cores per cluster |
| `SOCKET_SIZE`  | `MIN(4, NUM_CORES)` | cores that share one socket (→ share I$/D$) |
| `NUM_SOCKETS`  | `NUM_CORES/SOCKET_SIZE` | sockets per cluster |
| `NUM_WARPS`, `NUM_THREADS` | 4, 4 | warps per core, lanes per warp |
| `SIMD_WIDTH`   | `NUM_THREADS` | operand-data-path width |
| `ISSUE_WIDTH`  | `UP(NUM_WARPS/16)` | independent issue slices per core |
| `NUM_OPCS`     | `UP(NUM_WARPS/(4·ISSUE_WIDTH))` | operand collectors per issue slice |
| `NUM_GPR_BANKS`| 4 | register-file banks inside each OPC |
| `LMEM_LOG_SIZE`| 14 | log₂(lmem bytes) — default 16 KiB |
| `LMEM_NUM_BANKS` | `NUM_LSU_LANES` (=`SIMD_WIDTH`) | bank count for lmem |
| `ICACHE_SIZE` / `DCACHE_SIZE` | 16384 | bytes |
| `ICACHE_NUM_WAYS`, `DCACHE_NUM_WAYS` | 4 | ways |
| `DCACHE_NUM_BANKS` | `MIN(DCACHE_NUM_REQS, 16)` | banks |
| `NUM_ICACHES`, `NUM_DCACHES` | `UP(SOCKET_SIZE/4)` | physical cache slices per socket |
| `L2_CACHE_SIZE` | 1048576 | bytes (set to 1048576 in 1c8n4w4t) |
| `L2_NUM_BANKS`  | `MIN(L2_NUM_REQS, 16)` (=`NUM_SOCKETS*L1_MEM_PORTS`) | banks |
| `L2_NUM_WAYS`   | 8 | ways |
| `L1_LINE_SIZE`, `L2_LINE_SIZE` | 64 | bytes |

### 1.2 Physical SRAM macros in `/libs`

All macros are `bsg_fakeram`-style single-clock blocks. The `VX_sp_ram` and `VX_dp_ram` wrappers (`hw/rtl/libs`) pick a macro purely by `(SIZE, DATAW)` when `ASIC_SYNTHESIS` is defined:

| macro file | depth × width | ports | bits (B) | VX wrapper | picked when |
|---|---|---|---|---|---|
| `sram_64x512_1rw`  | 64 × 512 | 1RW  | 4096 B | `VX_sp_ram` | SIZE=64,  DATAW=512 |
| `sram_256x512_1rw` | 256 × 512 | 1RW | 16384 B | `VX_sp_ram` | SIZE=256, DATAW=512 |
| `sram_1024x32_1rw` | 1024 × 32 | 1RW | 4096 B  | `VX_sp_ram` | SIZE=1024, DATAW=32 |
| `sram_64x24_1r1w`  | 64 × 24  | 1R1W | 192 B   | `VX_dp_ram` | SIZE=64,  DATAW≤24 |
| `sram_256x24_1r1w` | 256 × 24 | 1R1W | 768 B   | `VX_dp_ram` | SIZE=256, DATAW≤24 |
| `sram_64x128_1r1w` | 64 × 128 | 1R1W | 1024 B  | `VX_dp_ram` | SIZE=64,  DATAW=128 |
| `sram_128x128_1r1w`| 128 × 128| 1R1W | 2048 B  | `VX_dp_ram` | SIZE=128, DATAW=128 |

The `if (SIZE==x && DATAW==y)` dispatch is literally spelled out in `VX_sp_ram.sv` / `VX_dp_ram.sv` (see `g_macro`). Anything not matched falls back to an inferred stdcell array ("flop array"). This is the key: the Vortex SRAM mapping is _exact_ — off-size arrays are not blown out into multiple banks, they are just stdcell-synthesized.

A second gate, `FORCE_BRAM(depth, width) = ((d≥64 || w≥16 || d*w≥512) && d*w≥64)`, decides whether the wrapper even considers mapping to a macro. Small arrays (e.g. MSHR of 16 entries, FIFO replacement counters) end up as flops.

---

### 1.3 The Register File

**Code:** `hw/rtl/core/VX_opc_unit.sv`, instantiated from `VX_operands.sv`.

The RF is partitioned across **operand collectors** (OPCs) and **banks** inside each OPC:

```
per VX_core:                                    per OPC (VX_opc_unit):
┌─────────────────────┐                         ┌────────────────────────┐
│ ISSUE_WIDTH slices  │   each slice has        │ NUM_GPR_BANKS VX_dp_ram│
│   (VX_issue_slice)  │    NUM_OPCS OPCs        │   (one per bank)       │
└─────────────────────┘                         └────────────────────────┘
```

Per-bank VX_dp_ram parameters (from `VX_opc_unit.sv:279`):

```
DATAW = XLEN * SIMD_WIDTH                           (128 b for RV32 · SIMD4)
SIZE  = NUM_REGS * SIMD_COUNT * PER_OPC_WARPS
        / NUM_GPR_BANKS
WRENW = BANK_DATA_WIDTH / 8                          (byte-enable per SIMD byte)
OUT_REG = 1, RDW_MODE = "R"
```

With our common config (`NUM_WARPS=4`, `NUM_THREADS=4`, `SIMD_WIDTH=4`, `ISSUE_WIDTH=1`, `NUM_OPCS=1`, `NUM_GPR_BANKS=4`, `EXT_F_ENABLE → NUM_REGS=64`, `SIMD_COUNT=1`, `PER_OPC_WARPS=4`):

- `BANK_SIZE = 64·1·4 / 4 = 64` rows
- Each bank → one `sram_64x128_1r1w`

**Warp / register layout in a bank (NUM_GPR_BANKS=4):**

```
  bank = rd[1:0]        (low 2 bits of the register number)
  row  = { warp_id[1:0] , rd[NUM_REGS_BITS-1:2] }    // 2 warp bits + 4 reg bits = 6 bits
  word = simd_word[3:0] = full 128-bit line  (4×32-bit lanes stored in a single SRAM word)
```

Example for `NUM_WARPS=4`:

```
                        bank0                bank1                bank2                bank3
row 0  warp0, reg0       x0[0..3]           x1[0..3]             x2[0..3]             x3[0..3]
row 1  warp0, reg4..7    x4[0..3]           x5[0..3]             x6[0..3]             x7[0..3]
...
row15  warp0, reg60..63  (FP28..31)
row16  warp1, reg0..3    ...
...
row63  warp3, reg60..63  ...
```

So **register numbers are low-bit interleaved across banks**, and **warps are packed as row MSBs** in the same bank arrays. The 4 SIMD lanes of a register are stored in parallel as the _columns_ of a single 128-bit SRAM word — there is no "lane" dimension spread across banks.

**Sharing:** the register file is strictly **per-core** (inside `VX_core`). Warps inside a core share their banks (and, in bigger configs, share OPCs across sub-groups of warps). No sharing between cores or sockets.

---

### 1.4 I$ and D$ (`VX_cache_wrap` / `VX_cache_bank`)

Both L1s use the exact same building block (`VX_cache_bank`) with different parameters. Each bank contains a **tag store** and a **data store**, organized as NUM_WAYS parallel SRAM instances (one per way):

**Tag store** — `hw/rtl/cache/VX_cache_tags.sv:99`:
```
VX_dp_ram  DATAW = TAG_WIDTH = 1(valid) + WRITEBACK + CS_TAG_SEL_BITS
           SIZE  = CS_LINES_PER_BANK
           count = NUM_WAYS         (one dp_ram per way)
```

**Data store** — `hw/rtl/cache/VX_cache_data.sv:124`:
```
VX_sp_ram  DATAW = CS_LINE_WIDTH = 8·LINE_SIZE
           SIZE  = CS_LINES_PER_BANK
           WRENW = LINE_SIZE (one write-enable per byte)
           count = NUM_WAYS         (one sp_ram per way)
```

With these formulas the derived quantities (from `VX_cache_define.vh`) are:
```
CS_LINES_PER_BANK = CACHE_SIZE / (LINE_SIZE · NUM_WAYS · NUM_BANKS)
CS_TAG_SEL_BITS   = CS_WORD_ADDR_WIDTH − CS_LINE_SEL_BITS − CS_BANK_SEL_BITS − CS_WORD_SEL_BITS
                  = (MEM_ADDR_WIDTH − log2 WORD_SIZE) − log2(LINES_PER_BANK)
                    − log2 NUM_BANKS − log2(LINE_SIZE / WORD_SIZE)
```

**Cache address layout** (`VX_cache.sv:274–303`):

```
   MSB ┌───────────┬──────────┬──────────┬────────────┐ LSB
       │   TAG     │ LINE IDX │ BANK IDX │ WORD_IN_LINE│
       └───────────┴──────────┴──────────┴────────────┘
```

So for a multi-bank cache, **successive cache lines go to successive banks** (low-bit interleaving at the line granularity). Within a bank, the **line index** picks a row across all ways simultaneously (one row per way is read in parallel by the NUM_WAYS parallel sp_rams). The set-associative match is decided outside the SRAMs by comparing the tag field against the `NUM_WAYS` tag outputs.

**I$ concrete params** in `VX_socket.sv:89–119`:
```
NUM_UNITS = NUM_ICACHES    NUM_BANKS = 1     NUM_WAYS = ICACHE_NUM_WAYS
LINE_SIZE = 64             WORD_SIZE = 4     CACHE_SIZE = ICACHE_SIZE
WRITE_ENABLE = 0 (read-only), no writeback
```

**D$ concrete params** in `VX_socket.sv:135–167`:
```
NUM_UNITS = NUM_DCACHES    NUM_BANKS = DCACHE_NUM_BANKS
NUM_WAYS  = DCACHE_NUM_WAYS  LINE_SIZE = 64
WORD_SIZE = DCACHE_WORD_SIZE = LSU_LINE_SIZE = MIN(NUM_LSU_LANES · XLENB, L1_LINE_SIZE)
NC_ENABLE = 1 (bypass path for non-cacheable flag)
```

**Sharing:** the L1s are instantiated **inside `VX_socket`**, so they are _shared across all `SOCKET_SIZE` cores_ in the same socket (a `VX_mem_arb` in front of each cache mixes the `SOCKET_SIZE` core streams). No sharing across sockets.

**Example — D$ data placement**, 16 KiB, 4 ways, 1 bank, 64 B lines, 16 B words → 64 lines/way:

```
  set index  way0 (sp_ram[0])   way1 (sp_ram[1])   way2 (sp_ram[2])   way3 (sp_ram[3])
  ─────────  ────────────────   ────────────────   ────────────────   ────────────────
   0         [512-b line]       [512-b line]       [512-b line]       [512-b line]
   1         ...                ...                ...                ...
   63        ...                ...                ...                ...
```

An access for address `A` picks `set = A[11:6]`, reads **all 4 sp_rams in parallel**, and meanwhile `tag_store[0..3]` (four 64×24 dp_rams) yield four tags that are compared against `A[31:12]`. The ways are **in different SRAM instances**, not multiplexed into one.

---

### 1.5 Local Memory (lmem)

**Code:** `hw/rtl/mem/VX_local_mem.sv:158–176`.

Plain word-banked scratchpad, **one VX_sp_ram per bank**:
```
DATAW = WORD_WIDTH = 8·WORD_SIZE = 8·LSU_WORD_SIZE = 32  (RV32)
SIZE  = WORDS_PER_BANK = (SIZE_bytes / WORD_SIZE) / NUM_BANKS
WRENW = WORD_SIZE
```

With default `SIZE = 2^LMEM_LOG_SIZE = 16 KiB`, `WORD_SIZE = 4`, `NUM_BANKS = LMEM_NUM_BANKS = NUM_LSU_LANES = SIMD_WIDTH = 4`:
- WORDS_PER_BANK = 4096/4 = 1024
- Each bank → one `sram_1024x32_1rw`

**Bank selection** (`VX_local_mem.sv:66–72`):
```
bank = addr[log2(NUM_BANKS)-1 : 0]         // low bits of word address
row  = addr[BANK_SEL_BITS +: BANK_ADDR_WIDTH]
```

→ **Word-interleaved across the 4 banks**, so a single SIMD-4 LSU request with 4 contiguous 32-bit words hits all 4 banks in parallel (that is exactly the point — no bank conflicts on linear access). If two lanes target the same bank, a crossbar in `VX_stream_xbar` (request arbiter) serializes them.

**Sharing:** lmem lives inside `VX_mem_unit`, which is **inside `VX_core`** — one lmem instance per core, shared by all warps/threads _in that core_. No sharing across cores. (This matches OpenCL/CUDA "shared memory" semantics, per-CU/per-SM.)

**Example — lmem layout**, addresses written as word-index:

```
   word 0  → bank0 row0            word 1  → bank1 row0            word 2  → bank2 row0            word 3  → bank3 row0
   word 4  → bank0 row1            word 5  → bank1 row1            ...
   ...
   word 4092 → bank0 row1023       word 4093 → bank1 row1023       word 4094 → bank2 row1023       word 4095 → bank3 row1023
```

---

### 1.6 L2 cache (cluster-level)

**Code:** `VX_cluster.sv:86–116` instantiates **one** `VX_cache_wrap` with
```
CACHE_SIZE = L2_CACHE_SIZE
LINE_SIZE  = L2_LINE_SIZE  = 64
WORD_SIZE  = L2_WORD_SIZE  = L1_LINE_SIZE = 64    (one "word" == one L1 line)
NUM_BANKS  = L2_NUM_BANKS
NUM_WAYS   = L2_NUM_WAYS   = 8
NUM_REQS   = L2_NUM_REQS   = NUM_SOCKETS · L1_MEM_PORTS
```

Because `VX_cache_wrap` → `VX_cache` → `VX_cache_bank` is the **same generic cache engine**, the L2 decomposes into:
- data: `L2_NUM_BANKS · L2_NUM_WAYS` `VX_sp_ram`s (one per way per bank)
- tags: `L2_NUM_BANKS · L2_NUM_WAYS` `VX_dp_ram`s

with identical address-layout semantics as §1.4. So an L1 miss address is striped:
```
addr → [ TAG | SET (line_idx) | BANK | 0 ]
```
meaning **consecutive L1-cache-line misses alternate across the L2 banks**. Within a bank, the 8 ways of a given set are in 8 parallel SRAMs.

**Sharing:** The L2 lives in `VX_cluster`, one per cluster, **shared by all `NUM_SOCKETS` sockets** in that cluster. Only present when `L2_ENABLE` is set.

---

### 1.7 Quick sharing summary

| Resource | Scope | Where it lives |
|---|---|---|
| Register file (GPR banks) | per-core, shared between warps | `VX_core/…/VX_opc_unit` |
| I-cache (tags + data) | per-socket, shared by `SOCKET_SIZE` cores | `VX_socket/icache` |
| D-cache (tags + data) | per-socket, shared by `SOCKET_SIZE` cores | `VX_socket/dcache` |
| Local memory | per-core, shared between warps | `VX_core/…/VX_local_mem` |
| L2 cache | per-cluster, shared by `NUM_SOCKETS` sockets | `VX_cluster/l2cache` |
| L3 cache (optional, not used here) | per-Vortex top | `Vortex/l3cache` |

No "socket group" level in the RTL — the hierarchy is `Vortex → clusters → sockets → cores`, and **SOCKET_SIZE is always forced to 1 in the two synthesis configs** so each socket contains exactly one core.

---

## Section 2 — The Two Synthesis/PnR Configurations

Defines (from `hw/syn/synopsys/Makefile:93–104`):

```
single-core : NUM_CLUSTERS=1 NUM_CORES=1 NUM_WARPS=4 NUM_THREADS=4 SOCKET_SIZE=1
              (no L2)                TOP = VX_socket_top
1c8n4w4t    : NUM_CLUSTERS=1 NUM_CORES=8 NUM_WARPS=4 NUM_THREADS=4 SOCKET_SIZE=1
              L2_ENABLE  L2_CACHE_SIZE=1048576  TOP = Vortex
```

Derived constants common to both configs:
- `SIMD_WIDTH = NUM_THREADS = 4`, `SIMD_COUNT = 1`
- `ISSUE_WIDTH = UP(NUM_WARPS/16) = 1`, `NUM_OPCS = UP(NUM_WARPS/(4·ISSUE_WIDTH)) = 1`
- `NUM_GPR_BANKS = 4`, `NUM_REGS = 64` (I + F)
- `NUM_LSU_LANES = SIMD_WIDTH = 4`, `LSU_LINE_SIZE = 16`
- `DCACHE_WORD_SIZE = 16`, `DCACHE_CHANNELS = 1`, `DCACHE_NUM_REQS = 1`, `DCACHE_NUM_BANKS = 1`
- `ICACHE_WORD_SIZE = 4`, `ICACHE_NUM_WAYS = DCACHE_NUM_WAYS = 4`
- `LMEM_NUM_BANKS = 4`, `LMEM` size = `2^14 = 16 KiB`
- `NUM_ICACHES = NUM_DCACHES = UP(SOCKET_SIZE/4) = 1`
- `NUM_SOCKETS = NUM_CORES / SOCKET_SIZE`
- For 1c8n4w4t: `L2_NUM_REQS = NUM_SOCKETS · L1_MEM_PORTS = 8·1 = 8`, so `L2_NUM_BANKS = 8`; `L2_NUM_WAYS = 8`; `CS_LINES_PER_BANK = 1048576 / (8·64·8) = 256`.

### 2.1 Macro instance count — closed-form formulas

Let me define:
```
N_RF_per_core  = ISSUE_WIDTH · NUM_OPCS · NUM_GPR_BANKS
N_ICACHE_way_per_socket = NUM_ICACHES · 1 (banks) · ICACHE_NUM_WAYS
N_DCACHE_way_per_socket = NUM_DCACHES · DCACHE_NUM_BANKS · DCACHE_NUM_WAYS
N_LMEM_per_core = LMEM_NUM_BANKS
N_L2_way       = L2_NUM_BANKS · L2_NUM_WAYS        (only if L2_ENABLE)
```

For every L1/L2 cache, the **tag** and **data** arrays each instantiate `N_*_way` SRAMs (one of each per way-per-bank). L1 caches map to `sram_64x24_1r1w` (tag) and `sram_64x512_1rw` (data); the 1c8n4w4t L2 maps to `sram_256x24_1r1w` (tag) and `sram_256x512_1rw` (data) because `CS_LINES_PER_BANK=256`.

---

### 2.2 `single-core` — per-core totals

This config has exactly one `VX_socket_top` holding one `VX_core`, its private I$/D$, lmem and RF.

| Structure | Macro file (`libs/`) | Instances |
|---|---|---|
| Register file banks (GPR) | **`sram_64x128_1r1w.v`** | `ISSUE_WIDTH · NUM_OPCS · NUM_GPR_BANKS = 1·1·4 = 4` |
| I$ data store (per way)  | **`sram_64x512_1rw.v`**   | `NUM_ICACHES · 1 · ICACHE_NUM_WAYS = 1·1·4 = 4` |
| I$ tag store  (per way)  | **`sram_64x24_1r1w.v`**   | `NUM_ICACHES · 1 · ICACHE_NUM_WAYS = 4` |
| D$ data store (per way-bank) | **`sram_64x512_1rw.v`** | `NUM_DCACHES · DCACHE_NUM_BANKS · DCACHE_NUM_WAYS = 1·1·4 = 4` |
| D$ tag store  (per way-bank) | **`sram_64x24_1r1w.v`** | `NUM_DCACHES · DCACHE_NUM_BANKS · DCACHE_NUM_WAYS = 4` |
| Local memory banks       | **`sram_1024x32_1rw.v`**  | `LMEM_NUM_BANKS = 4` |

**Total per single-core socket = 24 SRAM macros**: 4 × `sram_1024x32_1rw` + 4 × `sram_64x128_1r1w` + 8 × `sram_64x24_1r1w` + 8 × `sram_64x512_1rw`.

This is exactly what the synthesized netlist contains (confirmed by grepping `runs/single-core/results/VX_socket_top_netlist.v`):
```
     4 sram_1024x32_1rw
     4 sram_64x128_1r1w
     8 sram_64x24_1r1w
     8 sram_64x512_1rw
```

### 2.3 `1c8n4w4t` — adding the L2, blackboxing sockets

In this run DC runs **bottom-up**: `VX_socket_top` is linked as a pre-characterized blackbox (`.lib/.db` from PnR). So the Vortex-level netlist only exposes the SRAMs that live **outside** the socket — i.e. the L2 — plus 8 opaque socket instances.

L2 derived quantities (with `L2_CACHE_SIZE=1048576`, `L2_NUM_BANKS=8`, `L2_NUM_WAYS=8`, `L2_LINE_SIZE=L2_WORD_SIZE=64`):
- `CS_LINES_PER_BANK = L2_CACHE_SIZE / (L2_LINE_SIZE · L2_NUM_WAYS · L2_NUM_BANKS) = 1048576/(64·8·8) = 256`
- `CS_LINE_WIDTH = 8·L2_LINE_SIZE = 512`
- `CS_TAG_SEL_BITS = (MEM_ADDR_WIDTH − log2 L2_WORD_SIZE) − log2(CS_LINES_PER_BANK) − log2(L2_NUM_BANKS) − log2(L2_LINE_SIZE/L2_WORD_SIZE)`
  `= 32 − 6 − 8 − 3 − 0 = 15` → `TAG_WIDTH = 1 + 0 + 15 = 16` bits (fits in the 24-bit 1R1W macro).

| Structure | Macro file (`libs/`) | Instances |
|---|---|---|
| L2 data store (per way-bank) | **`sram_256x512_1rw.v`** | `L2_NUM_BANKS · L2_NUM_WAYS = 8·8 = 64` |
| L2 tag store  (per way-bank) | **`sram_256x24_1r1w.v`** | `L2_NUM_BANKS · L2_NUM_WAYS = 8·8 = 64` |
| (8 × `VX_socket_top` blackbox — internal SRAMs not visible here) | — | — |

**L2-only total = 128 macros**, matching exactly what `runs/1c8n4w4t/results/Vortex_netlist.v` holds:
```
    64 sram_256x24_1r1w
    64 sram_256x512_1rw
```

If you flatten the blackbox (or PnR the 1c8n4w4t flat), the fully-expanded macro count is:
```
  N_total = 8 · 24  (eight sockets, each 24 macros)  +  128 (L2)
          = 192 + 128 = 320 macros
```

---

### 2.4 Where the large macro count comes from

The L2 is 1 MiB, carved as **8 banks × 8 ways = 64 parallel cache-line arrays**, plus an **equal number of tag arrays**. Each way-bank combination becomes:

- one `sram_256x512_1rw` (256 lines × 512-bit line) → 16 KiB of data
- one `sram_256x24_1r1w` (256 lines × ≤24-bit tag)  → 672 B of tag (most bits unused; the macro is 24 bits wide)

That is **128 macros** (each data macro is 16 KiB, 64 of them = 1 MiB, which is the full L2) because:
1. `L2_NUM_WAYS=8` is baked into `VX_config.vh` — every set is materially 8 parallel SRAMs, one per way.
2. `L2_NUM_BANKS` is pinned by `MIN(L2_NUM_REQS, 16)` with `L2_NUM_REQS = NUM_SOCKETS · L1_MEM_PORTS = 8`, giving 8 banks.
3. The wrapper has two L2-eligible macro sizes (`64x512_1rw` and `256x512_1rw`); the chosen `CS_LINES_PER_BANK=256` snaps to the 256-deep flavor, giving good area density (16 KiB per data macro vs. 4 KiB for the 64-deep flavor).

### 2.5 Knobs that directly change the L2 macro count

Using the formula `N_L2 = 2 · L2_NUM_BANKS · L2_NUM_WAYS`:

| change | effect |
|---|---|
| `L2_NUM_WAYS=4` instead of 8 | halves the macro count (2 · 8 · 4 = 64) — but you lose associativity |
| `L2_NUM_BANKS` forced to 4 (via `-DL2_NUM_BANKS=4`) | halves the bank count → 2 · 4 · 8 = 64 macros, but serializes L2 accesses |
| smaller `L2_CACHE_SIZE` that causes `CS_LINES_PER_BANK=64` | each way-bank maps to `sram_64x512_1rw` instead of `sram_256x512_1rw` — same count but 4× smaller macros, worse area density |
| `L2_DISABLE` / no `L2_ENABLE` | zero L2 macros — L1 misses go straight to DRAM bypass tags |

The bottom line: the macro count in PnR is _not_ an SRAM-mapper artifact. It is a direct, unavoidable consequence of the way `VX_cache` instantiates one `sp_ram`+`dp_ram` **per way per bank**. In the 1c8n4w4t config the L2 geometry (`CS_LINES_PER_BANK=256`) lands on the 256-deep macro flavor, giving good capacity density (16 KiB each) at the cost of a large physical macro count.

