# Vortex organized like an H100: architecture skeleton

Status: approved by the user (2026-11-14); revision 2 of the same evening:
- CPC wording;
- the L2 shared by every GPC of a partition;
- L2 with 16 ways;
- the hash chain generalised from the H100's ideas;
- the H100-policy rule;
- memory controllers and Ramulator.

Roadmap task E11. Not committed: it lands with the first lane-B code sub-task (E16 or E17).

> **Ground truth (user, 2026-11-15):** this document is the main session's interpretation of the user's H100 spec, reviewed for the hash by the agent that wrote doc 09, but still not the spec. Every implementing agent reads the cited `NN/spec.md` sections itself (`dlmem_roadmap/h100_ref/h100-design-docs/`), starts each sub-task with a written spec check (step 0), and runs an H100 job (GPU 3 through the user's queue) or asks the user when the spec is silent, unclear or disagrees. Earlier drafts were wrong several times: mod 63 with an idle slice, a hard-coded 8 laps, label = slot, the fabric pairing, "a write waits for every ack", 4-partition litmus tests.

## 0. Configurability (user, 2026-11-15)
- **Every H100 trait that Vortex builds or reuses is a `VX_config.toml` knob or a formula of knobs (`expr:`), never a literal.** That covers the new features and the existing code the presets use. The trait's rationale is abstracted so that it adapts to any configuration, as for the hash (§3).
- **Four classes:**

  | Class | What it is | Rule | Examples |
  |---|---|---|---|
  | A, structural | counts and structure that follow from other counts | a formula of the configuration knobs | laps = slices per MP; max cluster size = SMs per GPC (a cluster stays in one GPC); request lanes per CPC = MPs per partition |
  | B, capacity | widths, rates, queue depths in front of shared units | share × sharers, derived in the toml (§5) | slice input, reply connections, CPC stage, queue depths |
  | C, timing | latencies, issue intervals | a knob with the H100 value as default, the same at every size | L1 hit 32, wgmma 56 + N/2 |
  | D, policy and free constants | replacement, allocation and protocol rules; hash masks | a knob or a selectable policy, defaulting to the H100's (§4); masks from the committed table (§3.0) | true LRU-16, R10, closed page |

- **Each trait carries a one-line rationale,** marked [spec] (quoted from the spec) or [ours] (our inference, to be verified in step 0).
- **Each knob states its constraints** (power of two, ranges, which knobs must agree), and a configuration-time check asserts them.
- **Presets set only the count knobs;** everything else derives from them (E21).
- **E21's static checker fails** on any unclassified literal in the registry.
- **Every sub-task's check includes a configuration sweep:** SimX at the presets, at one non-preset configuration and at the degenerate shape, with the feature's structural invariants asserted.
- **Each task file has an "H100 traits and knobs" table, and the hard-coded values in existing code have owners.** Both are listed in `dlmem_roadmap/task_E/` (the per-task tables and `audit_hardcoded.md`).

## 1. Goal and decisions
- **Goal:**
  - Vortex runs the same workloads as an NVIDIA H100 NVL and correlates with it.
  - It is built from the H100's structures: the SM, the compute tree, the on-chip networks, the partitioned L2, the coherence protocol between partitions, and HBM3.
- **Facts:**
  - The user's H100 spec, mirrored in `dlmem_roadmap/h100_ref/h100-design-docs/`.
  - Cited as `NN/spec.md §x` plus the `params.yaml` key and evidence tag.
- **User decisions (2026-11-14):**
  1. **Counts are powers of two and configurable.** That covers SMs, GPCs, CPCs, slices, MPs, sets and ways. The architecture stays the H100's at every size: the same levels, networks, arbitration, queues and protocol.
  2. **Scaling rule: keep the H100's per-SM shares.**
     - Every shared link or unit gets capacity = H100 per-sharer share × the number of its sharers in the preset.
     - So bandwidth per SM and DSMEM bandwidth per SM equal the H100's at every preset.
     - Latencies are H100 values at every size.
     - Single-unit microbenchmarks (one slice, one CPC) may then differ from the H100's unit values. This is accepted.
  3. **Presets** with the same per-SM ratios at every size (§5).
  4. **Timing knobs are set from the spec as each structure is built,** in RTL and SimX together (lockstep). Task G only correlates and fine-tunes.
  5. **The user's reference code may enter the repo** (C/C++). The Python models stay out and serve as test oracles.
  6. **The H100's policies everywhere (§4).** Wherever the H100 has a measured policy, Vortex uses that same policy, not Vortex's current one: replacement, write allocation, invalidation, page policy, scheduling, directory rules. For example, the L2 uses true LRU, not pseudo-LRU.

## 2. Levels and knobs
### 2.1 Compute: partition → GPC → CPC → TPC → SM
| Level | H100 NVL (GPU 3) | `h100` preset | What lives there |
|---|---|---|---|
| Partition | 2 | 2 | half of the chip: 4 GPCs and the partition's L2 slices; the point of coherence for its lines |
| GPC | 8 (4 per partition), 12–18 SMs each | 8 (4 per partition), 16 SMs each | cluster scope and per-GPC cluster dispatch (E16); one SM-to-SM network (N5, E18); one reply connection to every MP of its partition, plus the copy stage for TMA multicast (N4, E19) |
| **CPC** | 3 per full GPC (24 on the full die; GPU 3 has 23 because G0's CPC1 is fused), 3 TPCs = 6 SMs each (only 20 of the 23 have 6 SMs; 3 have 4) | **2 per GPC (16 in all), 4 TPCs = 8 SMs each** | the request stage and one request lane to every MP of its partition (N3, E19); the ingress stage for L2 replies (E19); the SM-to-SM uplinks to the GPC switch and the 2 egress blocks by SM parity (N5, E18) |
| TPC | 3 per CPC, 2 SMs | 4 per CPC, 2 SMs | 2 SMs; nothing bandwidth-relevant is shared at this level (`01/spec.md §5.2`) |
| SM | 132 | 128 | a `VX_core`: its own L1, shared memory, 4 sub-partitions (E14), egress 32 B/clk and ingress 128 B/clk ports (E12) |

- **The CPC is a new level:** Vortex has nothing like it today. E16 builds the container, E19 the request/ingress side, E18 the SM-to-SM side.
- **Two CPCs per GPC in every preset that has room** (user, 2026-11-14). The CPC is 8 SMs in `h100` and 2 SMs in `pocket_h100`. `nano_h100` has 2 SMs per GPC, so its GPC has one CPC of one TPC (§5).
- **Why 2 CPCs × 4 TPCs:**
  - it keeps the H100's (CPC, MP) request lane at 32 B, the one CPC element measured blind;
  - it keeps "request lanes per CPC = SMs per CPC" (H100: 6 = 6; here 8 = 8).
- **Knobs (pow2):** `NUM_PARTITIONS`, `GPCS_PER_PARTITION`, `CPCS_PER_GPC`, `TPCS_PER_CPC`, `SMS_PER_TPC` (2). E16 fixes the exact names.

### 2.2 Memory: partition → MP → LTC → L2 slice, one HBM3 channel per slice
- **No GPC owns slices.**
  - A line has one home slice, in one partition, chosen by the address hash (§3).
  - Every GPC has a reply connection to every MP of its own partition. So all GPCs of a partition share all its slices equally, through the partition's crossbar (N3/N4/N6).
  - The only GPC-to-slice difference is latency from placement: the near L2 block is about 28 cycles faster, and each TPC slot toward the block saves 5.4 cycles (`01/spec.md §7`; `18/spec.md §1.8`).
  - A GPC has no direct path to the other partition's slices. A far line is read through the slice of the same label in the reader's own partition (the far copy, E20), and that slice reaches the home over the fabric (N8).
- **Today's Vortex is different:** each `VX_cluster` owns a private L2. That ownership goes away: the H100 presets have one L2 per partition, made of slices shared by all its GPCs (E19).
- **Count knobs:** `MPS_PER_PARTITION`, `LTCS_PER_MP` (2), `SLICES_PER_LTC` (4), so 8 slices per MP as on the H100.
- **Coherence and the fabric between partitions are the H100's, defined for P ≤ 2** (`10/spec.md`; `18/spec.md` N8):
  - **Directory:** the home slice keeps a far-copy directory, 512 sets × 7 entries on the H100 (8 in Vortex). One entry covers a quad, with an exact sharer bit per line [M-blind].
  - **Home write or atomic to a tracked line** (sharer bit or ghost) [M-blind, `10/spec.md §0`, `§4.6`]:
    - it sends one invalidation to the copy partition;
    - the store itself is not held;
    - the writer's completion point waits for the ack: its next fence or release, or the atomic's return;
    - atomics to that line wait in a held queue until the ack;
    - loads are never held.
  - **Fabric:** one link per pair of labels, S/2 links per partition pair. In the presets' layout a pair is an LTC column, as on the H100 (E20 has the details).
- **More than 2 partitions:**
  - The hash supports any power-of-two P (§3), and P > 2 may be used for balance studies.
  - Coherence and the fabric are defined for P ≤ 2 only. They move to the backlog (AGENT.md §6, "Multi-partition coherence"), to be done after Vortex is fully upgraded.
  - **Assertion rule:** a configuration with P > 2 fails an assertion at configuration time if coherence or the fabric is enabled. The check runs in the toml configuration check, as an RTL `STATIC_ASSERT` and in the SimX constructor. There is no silent fallback.
  - A simulator build with more than one partition needs both to reach far lines, so P > 2 is in practice limited to the hash tool and its tests until the backlog item is done.

### 2.3 Networks (`18/spec.md`)
| Network | What it is | Sub-task |
|---|---|---|
| N1 | SM ports | E12 |
| N3 | CPC request stage and (CPC, MP) lanes | E19 |
| N4 | (GPC, MP) reply connections and the GPC copy stage | E19 |
| N5 | SM-to-SM network, one per GPC | E18 |
| N6/N7 | chip crossbar per partition; slice, LRC, MP | E19 |
| N8 | LTCFABRIC between same-label slices of the two partitions | E20 |
| N9 | slice → memory controller → HBM3 | E17 |

- Every network attaches through one interface (`VX_noc_if` / SimX `NocModel`) with a topology knob. E18 defines it.

### 2.4 How today's levels map (E16 writes the exact mapping)
- `VX_cluster` (DLMEM scope, CTA gang, KMU fan-out) becomes the GPC.
- `VX_socket` becomes the TPC.
- The CPC is new.
- The per-cluster L2 is replaced by the partition's shared slices.
- The L3 stays only for today's shapes.
- Each SM gets its own L1 in the H100 presets. Today one D-cache serves up to 4 cores of a socket.
- `NUM_CLUSTERS`, `NUM_CORES` and `SOCKET_SIZE` stay as aliases, so today's shapes are unchanged.

### 2.5 Clocks
- **Everything runs in SM cycles at 1635 MHz.**
- **L2-clock rates** are converted by 1559/1635.
- **DRAM** runs at 2619 MHz through the DRAM clock ratio (`19/spec.md §6`).

## 3. L2 address hashing: the H100 chain as a function of the configuration
Revision 7 (user, 2026-11-15). Reviewed four times by the agent that wrote doc 09, which then built the **golden reference** of this section: `hw/hash_golden/` (README.txt; `vortex_hash.py`; frozen `mask_table.json`; `gen_masks.py`; `check_all.py` → `results.txt` ALL PASS; `vectors/<preset>.csv.gz`). The golden reference is the ground truth for this section; where this text and the code disagree, the code wins. Its checks are in §3.9. The hash supports any power-of-two P; coherence and the fabric are defined for P ≤ 2 only (§2.2).

**Goal (user):** understand why each H100 function is built the way it is, and write it so that it works for every configuration.

**No fused slices (user):** no Vortex configuration has dead slices. The ring always has exactly P·S live slots. The H100's mod 47 is a side effect of yield fusing (the full die most likely uses 48 slots [I]), not a design goal.

### 3.0 Inputs, derived values, the order of the chain, the rank rule
- **From the configuration (all powers of two):**
  - P = partitions;
  - M = MPs per partition;
  - L = slices per MP (`LTCS_PER_MP × SLICES_PER_LTC`), which is the number of ring laps (the H100 has 8 laps because each MP has 8 slices);
  - S = M·L slices per partition;
  - N_set sets per slice;
  - N_dir directory sets per home slice;
  - N_bank DRAM banks per slice;
  - R_l lines per DRAM row.
- **Derived:**
  - **chunk size** 2^c bytes, c = max(17, 9 + log2(P·S)). That is 128 KiB for any chip up to 2 × 64 slices, as on the H100. The formula is exactly the condition under which exact balance per chunk is possible (the rank rule below).
  - **local bit count** k = (c − 7) − log2(P·S): the chunk's line bits minus the home and slice bits. It is 3 for `h100`, as on the H100.
  - **rotation step** g: odd, chosen by the stride test (§3.2, §3.8).
- **The chain:**
  1. home partition;
  2. ring slot;
  3. MP, position and label;
  4. the slice-local line number Lc;
  5. from Lc only: the L2 set, the far-copy directory set, and the DRAM row, column and bank.

  Because step 5 reads only Lc, a far copy lands in the same set as its home line.
- **Rank rule** (it defines the masks):
  - the chunk's hashed bits are PA 8 and PA 10 … c−1, that is the chunk's line bits without the quad bits PA7 and PA9;
  - all home masks and all slice-term masks, restricted to those bits, must be linearly independent over GF(2);
  - this is what gives exact balance per chunk, and it is possible only when c ≥ 9 + log2(P·S);
  - the H100 masks satisfy it: home plus the six T terms have rank 7 on the 8 hashed bits of a 128 KiB chunk, so any subset satisfies it automatically;
  - generated masks must be checked against it, and the code asserts it at configuration time.
- **Triangular rule** (per field; the H100 masks all follow it):

  | Field | H100 form |
  |---|---|
  | set superfamily | the masks m do not touch R bits 0–1 |
  | inner set | h3 uses only R bits; k3 uses R and j, not b |
  | directory set | bit k = n_k XOR (n bits ≥ 9) |
  | DRAM bank | bank_k = s_k XOR parity of row bits |

  - These functions read Lc only. Every bit of a set, directory or bank field is its own bit of the local number Lc, XORed with Lc bits outside that field.
  - That makes each field one-to-one, which is why set and bank balance is exact.
  - Generated masks keep this form, and the code asserts it next to the rank rule.
- **Choosing generated masks (needed for P > 2, M > 8, L > 8, more sets, directory sets or banks):**
  - the rank and triangular rules are necessary but not sufficient. For P = 4, 24 rule-passing home-mask sets ranged from 2.00 to 4.00 (busiest ÷ average, 1K window), against 3.25 for random placement;
  - so, for each configuration, generate about 64 candidate sets from a fixed seed, all passing both rules;
  - score each by its worst ratio over the stride test (§3.8);
  - keep the best, and record the masks and the score;
  - the presets' masks are frozen in a committed table, keyed by (P, M, L, N_set, N_dir, N_bank);
  - a configuration not in the table is computed once by the same tool and added to it, so RTL and SimX read fixed masks and nothing searches at build time;
  - this covers the extra home masks, slice terms, and set, directory and bank masks; g is chosen the same way.

### 3.1 Home partition
- **H100** (`09/spec.md §2`): `h = 1 ^ par(PA & 0x1cbd6ab000)`. The lowest input bit is 12, so the home is fixed per 4 KiB and splits exactly 50/50.
- **Why it works:**
  - a parity of address bits is balanced, so each partition gets half of any large region;
  - the lowest bit sets the granularity;
  - the high bits make the home depend on the physical page.
- **Vortex:** log2(P) parities h_0 … h_{p−1}, giving h_line = the line's home partition.
  - h_0 is the H100 mask with the H100 polarity.
  - More partitions use generated masks: lowest bit 12, PA7/PA9 excluded, and the rank rule.
  - P = 1 has no home bit.
  - **Extra home masks** are the ones the multi-partition test found most sensitive. They are chosen by the mask selection of §3.0.

### 3.2 Ring slot (which slice inside the home partition)
- **H100** (`09/spec.md §3`):
  - `Zf = 8·(PA >> 17) + (T1 + 2·T2 + 4·T4) + 6·(T6 + 2·T12 + 4·T24)`, `r = Zf mod 47`;
  - each Ti is the parity of a fixed mask.
- **Why it works:**
  - the XOR terms scatter the lines of a chunk over the ring;
  - the integer chunk term turns the ring by a step that is coprime with the slot count (gcd(8, 47) = 1);
  - the weighted terms form one integer, a sum with carries.
- **Vortex:**
  - `Z = g·(PA >> c) + Σ_{i<log2 M} 2^i·A_i + M·Σ_{j<log2 L} 2^j·B_j`, `r = Z mod S`, which is a bit select;
  - the A_i (log2 M in-lap terms) and B_j (log2 L lap terms) are mask parities that take the H100's masks in order: T1, T2, T4, then T6, T12, T24. A configuration that needs fewer terms takes the first ones; one that needs more uses generated masks, which pass the rank rule;
  - because M is a power of two, the in-lap sum never carries into the lap field. The chunk term carries, and it is always kept: the ring is never reduced to pure XOR.
- **The step g:**
  - **It must be odd:** any odd step reaches every slot of a 2^k ring, while the H100's 8 would reach only S/8 of them.
  - **Its value comes from the stride test on short windows (§3.8):**
    - over long runs every odd g spreads every power-of-two stride perfectly, so long windows cannot tell steps apart;
    - in the doc 09 agent's check, busiest slice ÷ average, worst over all power-of-two strides:

      | Window | H100 (mod 47) | g = 11 | g = 3 or 5 | g = 1 or 63 |
      |---|---|---|---|---|
      | 1024 lines | 2.29 | 2.00 | 1.75 | 3.50 |
      | 8192 lines | 1.45 | 1.38 | 1.25 | 1.75 |

    - g = 1 and g = S − 1 are clearly bad;
    - **g is chosen with both stride tests, 7a and 7b** (§3.8), and frozen in `mask_table.json`:

      | Preset | g |
      |---|---|
      | `h100` | 35 |
      | `small` | 3 |
      | `mid` | 7 |
      | `p4` | 5 |
      | `p8` | 21 |

      Our own presets, generated in a copy of the golden with `gen_masks.py` (2026-11-15, job `hg-presets-1`, `hg-micro-1`, `hg-final-1`):

      | Preset | Configuration | g |
      |---|---|---|
      | `pocket_h100` | 2 × 2 × 8, 256 sets, 2^35 B | 11 |
      | `nano_h100` | 2 × 1 × 8, 256 sets, 2^34 B | 3 |
      | `micro` | 2 × 1 × 4, 64 sets, 2^32 B | 3 |

      They go to the doc 09 agent to fold into the golden. A first `micro` with 2 slices per partition (L = 2, 64 sets) failed 7b on banks (worst 4.0 against a limit of 2.5 at a 128 KiB stride), so `micro` uses L = 4.

      The g = 11 of the table above is superseded for `h100`: for `mid`, g = 1 passed 7a and failed 7b.
- **Limit:** with a power-of-two ring the rotation sees only the low log2(S) bits of the chunk number (PA 17–22 for S = 64). Very large strides are spread only by the XOR masks, and the stride test covers them.

### 3.3 MP, position inside the MP, label (physical slice), far copies
- **H100** (`09/spec.md §4`, `§5`):
  - MP = perm[r mod 6]; lap = r div 6;
  - position q = lap ^ h_line;
  - label = LAB_OF[h_line][r], the physical slice position, which depends on the line's home: slot 0 is label 41 for home-P0 lines and 36 for home-P1 lines;
  - a far copy goes to the same label in the reader's partition, so the home slice and its copy sit at the same physical spot in the two partitions, which is the pair each fabric link joins [M].
- **Why:** the ring slot is logical; MP, position and label give the physical slice. The permutation and the label table are names, but the rule "same physical spot in both partitions" is structure.
- **Vortex:**
  - MP = r mod M (identity permutation);
  - lap = r div M;
  - position = lap XOR (h_line mod L), where h_line is the line's own home partition (not the slice's or the reader's);
  - **label = MP + M·position**: the physical slice ID inside a partition;
  - LTC = position & (n_ltc − 1) [A] (the H100 also XORs a constant per MP [M]; we drop it), and the slice inside the LTC = the next bits;
  - the home slice is (h_line, label);
  - **a far copy goes to (reader's partition, label);**
  - fabric link = MP + M·(position without its top bit) [A]: one link per pair of labels, S/2 links per partition pair. For L = 8 this pairs positions that differ in q bit 2, the H100's LTC-column pairing [I] (§2.2, E20).
- **More than 2 partitions:**
  - the H100 has only 2, so `lap XOR (h_line mod L)` is a design choice [A];
  - the doc 09 agent tested it for P = 2–16: it clearly beats no XOR, and a variant that also XORs the extra home bits into the MP field was not consistently better;
  - when P > L, some homes share an XOR value: no balance is lost, but those homes are not decorrelated from each other;
  - kept as is.
- **What this keeps from the H100:** home-P0 and home-P1 lines of the same slot use different physical slices; a home slice and its far copy always sit at the same spot; every label holds exactly 1/S of the lines.
- **Gone with the fused slot:** the 7-slice MP, the extra label chains, the lone fabric link and the one-slice DRAM channel.

### 3.4 Slice-local line number Lc
- **H100** (`09/spec.md §6`):
  - `n = (8·Zf) // 47 − u`, `Lc = 8n + (PA7, PA9, PA16)`;
  - it is dense and unique per slice;
  - an error here once hid in 14 of the 94 slices (e13 → e18/e19).
- **Why:** the set, the directory and the DRAM row and bank all come from Lc, so each slice needs its lines numbered 0, 1, 2… with no gaps and no repeats. The //47 form interleaves the chunk because 47 does not divide a chunk's line count.
- **Vortex** (no dead slots, so each chunk gives each slice exactly 2^k lines):
  - the local bits C = PA7, PA9, plus the highest hashed bits of the chunk that make (home terms, slice terms, C) an invertible function of the chunk's line bits;
  - `Lc = (PA >> c)·2^k + C`.
  - **For `h100`** C = (PA7, PA9, PA16), the H100's own local bits (rank 7 → 8 when PA16 is added).
  - **Properties, by construction:**
    - unique among the lines homed in each (partition, label);
    - dense (each chunk adds exactly 2^k);
    - independent of the partition, so a far copy has the same Lc as its home line;
    - the quad bits are Lc bits 0–1.
  - In this power-of-two form, Lc is literally the address with the home and slice bits removed. On the H100 that holds only in spirit, because its n interleaves chunks.

### 3.5 L2 set index
- **H100** (`09/spec.md §7`):
  - R = n >> 3, j = n & 7, b = Lc & 7;
  - m, h3, k3 = parities with masks M, H, K;
  - `set = 64·((R ^ m) mod 5) + ((j ^ h3) << 3 | (b ^ k3))`: 320 sets.
- **Why:**
  - an XOR set hash of the local number spreads lines over 64 inner sets, and the superfamily multiplies that;
  - mod 5 exists only because 320 is not a power of two;
  - reading Lc (not the PA) puts a far copy in the same set as its home line.
- **Vortex:**
  - `set = 64·((R ^ m) mod F) + inner`, with F = N_set / 64 (a bit select), and R, j, b taken from Lc as above;
  - if N_set < 64, the low log2(N_set) inner bits are kept;
  - the H100 masks M, H, K, plus generated masks where F needs more than the H100's 2 superfamily bits.
- **For `h100`:** F = 4 with the H100 masks unchanged.

### 3.6 Far-copy directory set and key
- **H100** (`09/spec.md §10`):
  - key = `PA & ~0x2ff`, one entry per quad;
  - set = 9 bits, bit i = `par(n & DN[i]) ^ (DP[i] & PA16)`;
  - 512 sets × 7 entries in the home slice.
- **Why:** it records who has a copy of each quad. It is indexed like the L2 but coarser, from the same local number.
- **Vortex:**
  - the same key;
  - set = log2(N_dir) bits, bit i = `par(n & DN[i]) ^ (DP[i] & b2)`, with n = Lc >> 3 and b2 = Lc bit 2 (= PA16 on the H100 and in `h100`);
  - the H100's DN and DP for the first 9 bits, generated beyond;
  - 8 entries per set; the rules (a)–(c) of `10/spec.md §2.2`.
- **Domain:** the H100 index is measured for n bits 0–16 only. In this design n is the chunk number, so n bit 17 starts at PA ≈ 16 GiB; above that, the directory bits are generated masks [A]. This matters only for configurations that simulate more than 16 GiB of memory.

### 3.7 DRAM row, column, bank (inside the slice's channel)
- **H100** (`09/spec.md §8`):
  - `row = Lc >> 8`;
  - `col = (Lc0, Lc6, Lc7)`;
  - `s = (Lc1^Lc6^Lc7, Lc2, Lc3, Lc4, Lc5)`;
  - `bank_k = s_k ^ par(row & HB[k])`;
  - 32 banks, 1 KB rows, closed page.
- **Why:**
  - the row is the high part of the local number, so each slice fills its own banks densely;
  - XOR-ing row bits into the bank spreads strided rows over the banks.
- **Vortex:**
  - `row = Lc >> (log2 R_l + log2 N_bank)`;
  - the column and bank bits are picked from Lc as on the H100;
  - `bank_k ^= par(row & HB[k])`, with the H100's HB masks and generated ones beyond 5 bank bits.
- **For `h100`:** identical to the H100.

### 3.8 Tests every configuration must pass (E19; they define "the same idea")
1. **Bijection and density:** Lc is unique among the lines homed in each (partition, label) and dense. A far copy shares its home line's Lc and set with that slice's own home lines; that is H100 behaviour (copies and home lines compete in the same sets), not a collision.
2. **Exact balance:** over contiguous regions, per partition, label, MP, set, bank and directory set: exactly equal shares.
3. **Quads:** the 4 lines of a quad share home, slice and directory entry.
4. **Home:** fixed per 4 KiB, exactly 1/P each, on every stride.
5. **Far copies:** a far copy sits at the same label in the reader's partition, with the same Lc and set as its home line.
6. **Rank and triangular rules:** asserted for every configuration's masks.
7. **Stride tests** (they choose g and the generated masks; implemented in the golden `stride.py`):
   - **7a, slice level:**
     - metrics: the home slices (P·S bins) and the reader partition's labels (S bins);
     - windows of 1K and 8K lines;
     - every power-of-two stride from 128 B, plus 10 odd strides;
     - a stride s with window w is tested only if w·s ≤ the memory size.
   - **7b, inside one slice:**
     - take the lines of each strided stream that land in one slice (4 × N_set lines, two slices checked);
     - measure busiest ÷ average over that slice's sets and over its banks.
     - Sets and banks exist per slice, so a whole-chip set or bank metric is meaningless: it flags contiguous memory as 32× unbalanced.
   - **Pass rule (replaces "≤ 1.1 × random p95"):** the worst ratio over the n tested strides must be ≤ the p95 of the worst of n random placements with the same bins and lines. It is computed as the single-placement quantile 0.95^(1/n) (`stride.py`, cached in `random_baseline.json`).
     - The old rule fails even a perfect random hash when 24–40 strides are tested.
     - The real H100 sits exactly at the new limit on 8K windows: 1.446 against 1.446.
   - **`h100` must also be no worse than the real H100** on the same streams (`reference/h100_hash.py`): worst 2.295 (1K) and 1.446 (8K) over its 94 home slices.
   - **Weak spot:** the golden's `small` preset passes only at the limit, its bank metric 2.000 against 2.000 with 16 banks. Prefer N_bank ≥ 32.
8. **RTL and SimX reproduce the golden vectors bit for bit:** every column of `hw/hash_golden/vectors/<preset>.csv.gz` (home, ring, mp, lap, pos, label, ltc, link, lc, set, dir_set, dir_key, dram_row, dram_col, dram_bank).
   - The masks and g are read from `mask_table.json`, converted to a header or toml at build time. Nothing searches at build time.
   - The Vortex side does not re-implement tests 1–7: they live in the golden (`check_all.py`).
   - A configuration not yet in the table is added in a copy of the golden (`gen_masks.py --config …`, `check_all.py`, `make_vectors.py`), and its entry and results go to the doc 09 agent, who folds them into the golden. The repo copy is never edited by hand.

There is no bit-exact H100 path. It would only re-test `h100_hash.py` and would put dead-slot code in the repo; the structural tests above prove faithfulness instead (doc 09 agent, user 2026-11-15).

### 3.9 Checks the doc 09 agent ran (2026-11-15)
- **The golden reference** (`hw/hash_golden/results.txt`, OVERALL: ALL PASS):
  - presets h100, small, mid, p4, p8 pass tests 1–7;
  - bit-exact against the H100's own functions on 50,000 lines wherever the generalisation keeps them: home, DRAM row/column/bank, directory set and key, inner set, and the `h100` local bits (PA7, PA9, PA16);
  - by design not compared: the ring (mod 47 vs mod 64), label numbering, Lc numbering and the superfamily (mod 5 vs mod 4).
- **Earlier checks, before the golden:**
Setup: 2 × 8 MPs × 8 slices, 128 KiB chunks, 3 local bits, g = 11, the H100 masks.
- rank 7 → 8 with PA16;
- (partition, slice, Lc) unique and gap-free over 64 MiB at 0, 3, 40 and 91 GiB;
- exactly 4096 lines per slice in all four regions;
- quads together on 20,000 random quads;
- every slice uses all 256 sets, exactly 64 lines each, over 256 MiB at 0, 5 and 77 GiB;
- the stride table of §3.2.
- **Multi-partition** (P, M, L) = (2,8,8), (4,8,4), (4,8,8), (8,8,4), (8,4,4), (16,8,4), each with 3 generated home-mask sets:
  - the home split stayed exactly 1/P on every stride;
  - the label rule `lap XOR (h mod L)` beat no XOR (e.g. 1.62 / 1.12 vs 2.00 / 1.38 at P = 2);
  - the generated home masks decided the rest (2.00–4.00 at 1K for P = 4), hence the mask selection of §3.0.

## 4. The H100's policies everywhere (user rule, 2026-11-14)
Each sub-task uses the policy the spec measured. Where the spec only assumes one ([A]), it uses that assumption and marks it.

| Structure | H100 policy (spec) | Vortex today | Sub-task |
|---|---|---|---|
| L2 replacement, loads | true LRU over 16 ways (`06/spec.md §5.1` [M-blind]; tree-PLRU and FIFO rejected) | tree pseudo-LRU (`VX_cache_repl.sv`) | E19 |
| L2 replacement, writes | write marks, rule R10: scan depth 8, cap 7, delayed drop (`06/spec.md §5.2`) | none | E19 |
| L2 partial lines after a 4-B store miss | PFD/PFDx (`06/spec.md §6`) | none | E19 |
| L2 fills | per-sector state, 64-B half fills, write allocation per sector (`06/spec.md §4`) | 64-B sectors | E19 |
| L2 write-back | write-back, eviction-driven, sector-granular; dirty cap 7 of 16 ways (`06/spec.md §4`, `§13`) | write-back LLC | E19 |
| L1 writes | write-through; stores allocate their bytes with a byte-valid mask; a load hits only if all its bytes are valid (`05/spec.md §4.5`) | write-through, no allocation on a write miss | E12 |
| L1 replacement | behaves fully associative; random-like, protecting the newest ~25 % ([A], `05/spec.md §4.6`) | set-associative, pseudo-LRU | E12 |
| L1 invalidation | whole-L1 invalidate on fences of cluster scope or wider and on every `ld.acquire` (`05/spec.md §4.7`) | fence walk | E12, E20 |
| Far-copy directory | 512 sets in recency order, rules (a)–(c), back-invalidate on drop (`10/spec.md §2.2`) | none | E20 |
| DRAM | closed page, reads pass queued writes, turnaround, age switch, 64 + 64 controller queue (`07/spec.md §4`, `§10`; `18/spec.md §9`) | open page, FR-FCFS, watermarks | E17 |
| Warp scheduler | open in the spec (`02/spec.md §6`); set from E14's H100 probe | round-robin (RTL), GTO (SimX) | E14 |
| Cluster placement | one GPC, all-or-nothing, GPC round-robin, lowest free SM ids (`03/spec.md §2`, `§4`) | SPREAD per `VX_cluster` | E16 |

- **Conflict to settle:** the L2 hint rules H4/H7 and the evict_last caps (`06/spec.md §7`) belong to the cache-hint instructions, which your Q9 decision of 2026-11-14 keeps as no-ops (an accepted difference). They stay out unless you lift Q9.
- **Leave the backlog under this rule:** L1 store-allocate, the L2 replacement and write-mark rules, and the dirty cap. They move into E12 and E19.
- **Stay in the backlog:** the TLBs (Vortex runs without VM) and the constant cache (Vortex has none).

## 5. Scaling rule in numbers, and presets (details: `dlmem_roadmap/task_E/E11_scaling_table.md`, `e11_presets.py`)
- **Capacity = share × sharers.**
  - All named presets have 1 SM per slice; GPU 3 has 66/47 = 1.404.
  - So every slice-side unit runs at 47/66 = 0.712 of its H100 unit value (the GPC-side reply connection at 0.727, from its native per-unit value):

    | Unit | H100 → preset |
    |---|---|
    | slice input | 32 → 22.8 B/L2clk |
    | slice reply | 64 → 45.6 B/L2clk |
    | atomic unit | (no unit width) → 0.477 ops/L2clk |
    | fabric link | 32 → 22.8 B/L2clk |
    | (GPC, MP) reply connection | 128 → 93.1 B/L2clk (native per-unit value, review 2) |
    | HBM per slice | 40.6 → 28.9 GB/s |

  - Per-SM units (egress 32 B, ingress 128 B, SM-to-SM injection 21.3 B) keep their H100 values.
- **Checks.**
  - The per-SM bottleneck order is the H100's.
    - Writes: slice input 21.7 < CPC stage 27.7 < lane 30.5 < SM egress 32 B/SMclk.
    - Reads: DRAM 17.7 < slice reply 43.5 < reply connection 44.4 < CPC ingress 86.7 < SM ingress 128 (re-run 2026-11-15 with native lane and connection widths; the order is unchanged, the old tie between slice reply and connection is now 2 % apart, as on the H100: 3072 vs 3008 B/L2clk).
  - The model's per-SM bound falls within 0.86–1.00 of the measured per-SM rates in `18/spec.md §S.A`.
- **Queues.** A queue in front of a scaled unit scales with it, so it holds for the same time as on the H100 (e.g. the slice read queue 512 → 365). Per-SM queues keep the H100 depth.
- **L2 slice:** 256 sets × 16 ways × 128-B lines (32-B sectors) = 512 KiB.
  - That is +12.3 % L2 per SM against the H100 (58.75 MiB / 132 = 455.8 KiB).
  - The power-of-two way rule (user, 2026-11-14) makes this the closest choice: 128 sets would give −44 %.
  - Replacement is true LRU-16, so R10's "scan 8, cap 7" applies unchanged.
- **Far-copy directory:** 512 sets × 8 entries per home slice.
  - That is 4096 quad entries against the H100's 3584, so +14 % per slice.
  - Against the slice's own lines, the reach is 1.0 entries per line, where the H100 has 0.70.
- **Memory controllers and DRAM:** §6.

| Preset | Shape | SMs | CPCs | Slices / MPs | L2 | HBM read | Use |
|---|---|---|---|---|---|---|---|
| `h100` | 2 partitions × 4 GPCs × 2 CPCs × 4 TPCs × 2 SMs | 128 | 16 | 128 / 16 | 64 MiB | 3701 GB/s, 8 stacks × 16 ch | SimX: workloads, correlation |
| `pocket_h100` | 2 × 4 × 2 CPCs × 1 TPC × 2 SMs | 32 | 16 | 32 / 4 | 16 MiB | 925 GB/s | SimX: long workloads, sweeps |
| `nano_h100` | 2 × 4 × 1 CPC × 1 TPC × 2 SMs | 16 | 8 | 16 / 2 | 8 MiB | 463 GB/s | SimX CI, large rtlsim |
| `micro` | 2 partitions × 1 GPC × 1 CPC × 1 TPC × 2 SMs; 1 MP × 2 LTCs × 2 slices per partition, 64 sets × 16 ways (structure only, ratios not kept) | 4 | 2 | 8 / 2 | 1 MiB | — | rtlsim feature tests, SimX/RTL parity |
| degenerate | today's shapes | — | — | — | — | — | cycle-identical to today |

- **The smaller presets behave like `h100` per SM.** They differ only in whole-chip effects:
  - total L2;
  - the number of CTAs and clusters that fit;
  - the largest cluster: K ≤ 4 on pocket, K ≤ 2 on nano;
  - one L2 block per partition on nano.

### Scale probe (2026-11-14, iacoma234, SimX of today's Vortex at `NUM_CLUSTERS × NUM_CORES`, weak scaling)
| SMs | SimX, 1 thread: sgemm | SimX, 8 threads: sgemm | SimX, 1 thread: vecadd | SimX, 8 threads: vecadd |
|---|---|---|---|---|
| 1 | 0.8 s (243k cycles) | 1.2 s | 3.6 s (317k cycles) | 3.4 s |
| 4 | 1.6 s (196k) | 1.9 s | — | — |
| 16 | 6.1 s (219k) | 4.7 s | 19.9 s (806k) | 12.1 s |
| 64 | 40.8 s (221k) | 16.0 s | — | — |
| 128 | 95.1 s (223k) = 2.3k cycles/s | 28.6 s = 7.8k cycles/s (3.3×) | 962 s (2.68M) = 2.8k cycles/s | 143 s = 18.7k cycles/s (6.7×) |

- **Setup:** today's default core (4 warps × 4 threads), L2, and an L3 above 1 cluster. Each figure is the second run, with the simulator already built.
- **rtlsim, default core:**

  | Cores | Build | Speed |
  |---|---|---|
  | 4 | 71 s | 13k cycles/s |
  | 16 | 10.4 min | 0.5k cycles/s (SimX about 70× faster) |

- **What it means for the presets:**
  - At today's speed, `h100` with 8 threads simulates 8–19k cycles/s.
  - So a kernel that runs 1 ms on the H100 (1.6M cycles) takes about 1.5–3.5 min.
  - An H100-like SM and the new networks cost more per cycle. E16's per-GPC execution domains are what keep this tractable.
- **Recommendation:**
  - `h100` for kernels up to a few ms;
  - `pocket_h100` for longer workloads;
  - remeasure after E16, then fix the paper's preset.
- **Records:** job `rf-probe-2`, `/scratch/rid2/dlmem-tmp/hostrun-logs/rf-probe-2/`.

## 6. Memory controllers and DRAM (E17)
- **H100 facts:**
  - **Organisation:** 12 memory controllers × 512 bits, one per MP [I]; 24 FBPAs, one per LTC; 48 128-bit counter channels: 46 with 2 slices plus 2 with 1 slice (label 20 in each partition, whose mate is fused); 6 HBM3 stacks at 2619 MHz, 5.24 Gb/s per pin (`07/spec.md §1`).
  - **Per slice (`18/spec.md §9`):**
    - one independent 64-bit path per slice, with no crossbar and no limit shared above the slice, up to a whole MP;
    - 40.6 GB/s read and 40.8 GB/s write per slice;
    - one in-order queue of about 490 items;
    - a controller queue of 64 reads + 64 write-backs, where reads pass writes;
    - a 60 ns read/write turnaround and a 600 ns age switch;
    - closed page (`07/spec.md §4`);
    - the slice's address map of §3 row 7.
- **Vortex presets:**
  - **One memory controller per MP**, keeping the H100 grouping: 8 slices × 64 bits = 512 bits, as on the H100.

    | Preset | MCs | HBM stacks |
    |---|---|---|
    | `h100` | 16 | 8 × 16 channels |
    | `pocket_h100` | 4 | 2 |
    | `nano_h100` | 2 | 1 |

  - **One 64-bit HBM3 channel per slice** at the H100's pin rate and timing.
  - **The controller queues and policies are modelled per slice,** as the spec measures them. The MC and FBPA are groupings, because the spec finds no limit at those levels.
  - **Bandwidth follows the scaling rule:** each slice path runs at 47/66 of an H100 slice (28.9 GB/s), through a service-interval limiter. Latencies stay H100 values (534 local / 697 far miss).
  - **Queues scale with that unit:** 64 → 46 entries.
- **DRAM timings:**
  - The spec does not measure tRCD, tCL, tRP, tRAS, tWR, tWTR, tFAW, tRRD or refresh (`07/spec.md §9`).
  - E17 starts from the JEDEC HBM3 values (JESD238) and tunes them to the measured targets: the 534/697-cycle misses, the +70 bank conflict, the per-slice rates, the read/write mix and the loaded latency.
- **The Vortex-to-Ramulator wrapper** (`sim/common/dram_sim.cpp`, upstream Vortex code) is replaced:
  - today it has one queue for all channels and sends one 16-B piece per cycle (about 16 B/cycle in total);
  - the pieces of neighbouring lines overlap in address;
  - a read completes on its first piece and a write on acceptance;
  - there is no backpressure.
- **Ramulator version:**
  - **What Vortex has:** the unmodified upstream `CMU-SAFARI/ramulator2` (a git submodule, not a fork), pinned to Ramulator 2.0 of 2024-06-29 (PR #52). It is built from source by `third_party/Makefile`, so a version change is an ordinary Vortex commit, with no prebuilt-toolchain step.
  - **What upstream has now:** Ramulator **2.1** (2026, a major rework; 2.0 lives on the `legacy/v2.0a` branch). For our needs 2.1 already provides:
    - JEDEC HBM3 timing rules and presets (JESD238), replacing 2.0's placeholder that copied the HBM2 numbers;
    - an HBM3/HBM4 controller with per-bank refresh;
    - a closed-page row policy (ClosedCAP), which 2.0 could not use with HBM;
    - read and write buffer sizes and write watermarks as configuration values (fixed at 32 in 2.0);
    - registered extension points (controller plugins, schedulers, row policies) for adding behaviour without forking.
  - **What 2.1 still lacks for the H100:** the age-based read/write switch (2.1 switches on watermarks only), and a 5.24 Gb/s speed bin (only 6400 Mb/s is preset; timings are configurable).
  - **E17 plan:**
    - move the submodule to Ramulator 2.1;
    - port the wrapper to its External front end;
    - add the H100 controller behaviour as a 2.1 extension registered from Vortex's own tree, so Ramulator's sources are not modified;
    - give the HBM3 timings in the configuration.

## 7. Differences the pow2 presets cannot keep (in E0 Part E as accepted differences)
1. **Uneven GPCs** (12–18 SMs; SMs 124–131 only for K ≤ 2) and their cluster occupancy. With 1 CTA per SM, M1(K):

   | | K = 8 | K = 16 |
   |---|---|---|
   | `h100` | 16 | 8 |
   | GPU 3 | 15 | 7 |

2. **The hash:** a power-of-two ring with no dead slots (mod S, an odd rotation step chosen by the short-window stride test, Lc built by removing the home and slice bits), label = MP + M·position, and mod 4 instead of mod 5 superfamilies. Power-of-two strides rely on the masks and the odd step instead of a prime modulus (stride test §3.8). The partition parity, the XOR masks, the quad rule, the directory index and the DRAM map are the H100's.
3. **Floorplan:** no fused TPC slots; 2 CPCs × 4 TPCs per GPC instead of 3 × 3, with the per-slot latency steps on that grid.
4. **The partition imbalance:** P0 has 11 CPCs and P1 has 12.
5. **Single-unit slice-side rates** run at 71 % of GPU 3's (decision 2).
6. **Per-MP traffic share:** each MP carries 1/8 of an SM's traffic instead of 1/6.
7. **One clock** (no lock-dependent behaviours).
8. **L2 per SM** is +12.3 %, with 256 sets instead of 320. The directory has 8 entries per set instead of 7.
9. **The `h100` chip HBM rate per SM** is +3.1 % (read) / +4.1 % (write) above GPU 3's measured rate.

## 8. Timing policy
- **Each sub-task's "H100 timing knobs" section** lists, for every knob:
  - its H100 value, `NN/spec.md §x` and params key;
  - its scaling: latency unchanged, or capacity = share × sharers.
- **Sources:**
  - `17/spec.md §3` (canonical values);
  - the `18/spec.md §0.6` path element table (queue depths, credits);
  - `02/spec.md §4` (instruction latencies and intervals).
- **One calibration anchor:** one local L2 hit is calibrated, and the other latency classes are set as offsets from it (`17/spec.md §1.2`).
- **RTL and SimX use the same values.** model_parity holds on `micro` and the degenerate shapes.
- **E21 adds a static check:** every knob carries its citation, and every capacity knob equals share × sharers.

## 9. Reference code in the repo
- **The L2 hash golden reference** `hw/hash_golden/` (from the doc 09 agent, 2026-11-15, sha256 of the archive d5045e66…22ec8b; never edited in the repo):
  - the chain of §3 in Python;
  - the frozen masks and g per configuration;
  - the tests and the golden vectors.
- **The C/C++ header** `sw/common/vx_l2_hash.h` (E19) implements the chain for RTL testbenches, SimX and host tests, reading the masks generated from `mask_table.json` at build time, and must match the vectors bit for bit. The exact H100 functions stay in the golden's `reference/h100_hash.py` as an oracle only.
- **Users:** RTL testbenches, SimX and the host tests (E19).
- **The DRAM map and the directory index** (§3 rows 6–7) are part of the same header.
- **Doc 10 litmus programs LT1a–LT19** become test apps with their expected outcomes. The §11 invariants are asserted in SimX (E20).
- **Oracles that stay out of the repo:**
  - `02 model/issue_model.py` (known error: 86/96 within 5 % on its blind check);
  - `15 model/wgmma_timing.py`;
  - `14 model/tma_timing.py`;
  - `18 model/noc_bw_model.py` and the five network simulators;
  - `07 model/dram_ref.py`;
  - the Murphi/Rumur model.

## 10. Test-shape policy
- **Degenerate shapes stay cycle-identical,** with knob-off identity.
- **Every other catalog shape is re-expressed** on a named preset and re-baselined (E21).
- **Exception:** the DRAM wrapper fix (§6) changes DRAM timing in every configuration. Recommended: fix it everywhere and regenerate the baselines once (decided at E17 planning).

## 11. SimX at H100 scale
- **Execution domains:** E16 makes each GPC, and each partition's memory side, a SimX execution domain, so `SIMX_MT` scales.
- **Boundary delays are knobs the RTL carries too,** so lockstep holds.
- **Limits:** E16 removes the SimX limits that stop 128 SMs.

## 12. Choices
- **Settled by the user (2026-11-14):**
  - the CPC split is 2 per GPC;
  - the L2 has 16 ways;
  - the H100's policies apply everywhere;
  - the hash generalises the H100's ideas (§3), with every count-tied constant a formula of the configuration (§3.1);
  - the directory has 8 entries per set.
- **Recommended, to confirm:**
  1. Ramulator: move to 2.1 in E17;
  2. the DRAM wrapper: fixed everywhere;
  3. the paper's preset: after E16 (§5).
