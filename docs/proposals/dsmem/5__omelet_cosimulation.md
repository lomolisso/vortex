# Task 5 — Omelet NoC Co-Simulation: All Vortex Fabrics on a Real Network Model

Part of the [DSMEM & TMA Roadmap](README.md); this is **Task 5** (the final task), and
it **depends on [Task 3](3__dsmem_microarchitecture.md)** (DSMEM endpoints, residency table,
cluster barrier) and **[Task 4](4__tma_multicast.md)** (TMA multicast fan-out). It replaces
SimX-internal NoC modeling — including Task 3's interim `IdealTransport` placeholder — with
**co-simulation against Omelet**, the
packaging-aware gem5/Garnet chiplet-interconnect simulator (sibling checkout `../omelet`,
mounted at `/omelet` in the dev container).

## 1. Why (motivation)

SimX's fabrics are functional arbiters/crossbars with fixed delays — adequate for pipeline
studies, useless for NoC research. The program needs router-level network models (buffering,
VCs, per-hop contention, heterogeneous links) for a single 2D die **now**, and for **2.5D/3D
chiplet partitions of Vortex** (NoI interposer + NoL vertical networks) next. Omelet provides
exactly that: gem5 v22.1 Garnet with per-flit latency-breakdown instrumentation,
technology-grounded link parameters (`⟨W, t_cyc, E_bit⟩` from EM+SPICE tables), placement-aware
link construction, and a unified NoX graph across hierarchy levels. Rather than reimplementing
any of it in SimX, SimX becomes the traffic source/sink and Omelet models every network.

**Division of labor: any NoC in Vortex is modeled by Omelet; everything else by SimX.**

## 2. Scope — the four Vortex fabrics

The SimX interconnect inventory identifies exactly four NoC-class fabrics (everything else —
L1 egress muxes, cache bank crossbars, the SMEM bank crossbar, LSU coalescer/switch — is
intra-block wiring and stays SimX-modeled):

| # | Fabric | SimX site | Endpoints (config-general) |
|---|---|---|---|
| ① | SM-to-SM DSMEM transport | `cluster.cpp` (DSMEM-enabled, `IdealTransport` today) | `NUM_CORES` × `NUM_CORES` |
| ② | cores → L2 (cluster interconnect) | `cluster.cpp` L2 fan-in | `NUM_CORES·L1_MEM_PORTS` → `L2_NUM_REQS` |
| ③ | clusters → L3/LLC (processor interconnect) | `processor.cpp` L3 wiring | `NUM_CLUSTERS·L2_MEM_PORTS` → `L3_NUM_REQS` |
| ④ | LLC → DRAM channels (memory-controller crossbar) | `mem/memory.cpp` `mem_xbar_` | `L3_MEM_PORTS` → `PLATFORM_MEMORY_NUM_BANKS` |

Phasing inside the task: **① first** (its endpoint contract — per-core `MemReq/MemRsp` port
pairs with `dsmem_target` routing and tag-based response return — is ready today), then ②③④
(degenerate 1→1 at the default 1-core config; they become real networks at multi-core /
multi-cluster configs, which is where this task's studies live). Current physical scope is a
**single 2D die** (Omelet NoC hierarchy only); the NoI/NoL layers activate when Vortex is
partitioned across chiplets — the co-sim interface below is deliberately identical for all
three hierarchy levels, so that step changes only the Omelet design-point yaml.

## 3. Architecture — single-process embedding, timing-only tokens

One gem5 process (Omelet's patched v22.1 `Garnet_standalone/gem5.opt`) embeds SimX; there is
no socket/IPC layer and no synchronization quantum — one event loop owns both simulators.

```
gem5.opt (Omelet build)
├── Omelet NoX network — built unchanged by omelet.nox from the generated design point
├── VortexGPGPU SimObject (exists: sim/simx/gem5/) — dlopens libvortex-gem5.so,
│     ticks Processor::cycle() at the Vortex clock
└── NEW VortexNocBridge SimObject (one per fabric)
      ├─ inject: pops (src_ep, dst_ep, bytes, is_write, cookie) tokens from the
      │    C-ABI queue → enqueues a message at the src NetworkInterface's protocol
      │    MessageBuffer (the Garnet injection seam; NetDest = dst endpoint)
      └─ eject: on message arrival at the dst NI, returns the cookie via the C ABI;
           SimX delivers the held MemReq to the destination block
```

- **Timing-only tokens.** Functional data never leaves SimX (the `Memory::set_pre_send_hook`
  idiom from the SST/gem5 memory coupling): the SimX-side bridge holds each real `MemReq` in a
  pending table keyed by cookie and forwards only `(src, dst, size, type, cookie)`. Garnet
  messages carry no payload anyway (Garnet_standalone's directory is a sink).
- **One-way messages.** A remote load is a request token A→B; on ejection SimX performs the
  LMEM/cache access and the response is an independent token B→A. Store acks (`strsp`) and DXA
  completion notifies ride the same way. This matches both the Garnet_standalone protocol shape
  and Task 3's endpoint contract.
- **Backpressure.** Injection checks NI buffer slots; a refused token leaves the `MemReq` in
  the SimX channel, backpressuring the egress arbiter → adapters → LSU exactly as the
  SimX-internal fabrics do. Cross-fabric congestion coupling (Omelet's L1 fix) then reaches the
  GPU pipeline for real.
- **SimX side.** A generic `OmeletNocPort` class replaces each fabric instance when co-sim is
  on; fabric ① drops in behind the same 4-binds-per-core contract as `IdealTransport`. When
  co-sim is off, `IdealTransport` (①) and the existing arbiters/crossbars (②③④) remain — the
  **standalone `simx` binary keeps working without gem5**.
- **C ABI.** `sim/simx/gem5/vortex_gpgpu.h` grows per-fabric inject/eject queue calls
  (`vortex_gem5_noc_pop_inject`, `vortex_gem5_noc_push_eject`, …); the SimObject side stays
  free of SimX types.

## 4. Knobs — `VX_CFG_*` drives Omelet (hard requirement)

Switching any Vortex NoC's topology must be a `VX_CFG_*` edit. Two layers:

1. **`VX_CFG_NOC_MODEL = "IDEAL" | "OMELET"`** (enum, default `IDEAL`): selects SimX-internal
   placeholder fabrics vs Omelet co-simulation. `IDEAL` builds/runs with no gem5 dependency and
   must stay byte-identical to pre-Task-4 behavior.
2. **Per-fabric knobs** consumed by a configure-time generator (`gen_omelet_cfg.py`, invoked
   from `configure` like `gen_config.py`) that emits the Omelet design-point yaml + an
   **endpoint map** (SimX endpoint id ↔ Garnet NI id) consumed by both the bridge SimObjects
   and the topology builders:

| Vortex knob | Omelet yaml field | Notes |
|---|---|---|
| `VX_CFG_NOC_SMTOSM_TOPOLOGY` | `network.noc.topology` (mesh / ring / …) | fabric ① |
| `VX_CFG_NOC_L2_TOPOLOGY`, `VX_CFG_NOC_L3_TOPOLOGY`, `VX_CFG_NOC_MEM_TOPOLOGY` | per-fabric `network.*.topology` | fabrics ②③④ |
| `VX_CFG_NOC_ROUTER_LATENCY` | `router_latency.*` | per-hop pipeline depth |
| `VX_CFG_NOC_CLOCK_GHZ` | `clock_ghz` / `chiplet_clock` | plus the Vortex↔Garnet clock ratio for the `VortexGPGPU` tick |
| `VX_CFG_NOC_FLIT_BITS` | `flit_bits` (default 128) | token size → flit count |
| `VX_CFG_NUM_CORES`, `VX_CFG_NUM_CLUSTERS`, … | `cores_per_chiplet`, router counts | endpoint counts derived, not duplicated |
| (future 2.5D/3D) `VX_CFG_NOC_PARTITION`, bonding/interposer knobs | `layout`, `tech.*`, `simulation.bonding` | activates NoI/NoL |

The generator is the **single source of truth for the knob↔yaml linkage**; hand-edited Omelet
yamls remain possible for packaging-side exploration (Omelet's dual-interface philosophy), but
the Vortex flow always regenerates from `VX_CFG_*`.

## 5. Run flow

`blackbox.sh --driver=simx` with `VX_CFG_NOC_MODEL=OMELET` swaps the launch: instead of
executing the standalone driver, it invokes Omelet's `gem5.opt` with a generated config that
instantiates the NoX network + `VortexGPGPU` (kernel/args passed through). Results: the app's
stdout/exit code as today, plus Omelet's `stats.txt` (per-link utilization, five-way latency
breakdown) and `energy.txt` in the run directory — the NoC study surface replacing the removed
MPM `SMTOSM` class.

## 6. Risks

- **gem5 version skew**: `VortexGPGPU` was written against gem5 v25.0.0.1; Omelet pins
  v22.1.0.0. A compatibility pass is step one (APIs used — `DmaDevice`,
  `EventFunctionWrapper`, params codegen — are stable, but verify by compiling).
- **Message-size granularity**: Garnet_standalone's `MessageSizeType` buckets (Control ≈ 8 B,
  Data ≈ line-sized) vs word-granularity DSMEM ops; acceptable initially (1-flit vs multi-flit
  is the first-order effect), custom sizes if needed.
- **Simulation speed**: gem5 + SimX in one process; Garnet cost scales with router count and
  traffic. Keep `IDEAL` the default for functional work.
- **Determinism**: single event loop is deterministic, but Garnet's random-tiebreak arbiters
  must be seeded fixed for byte-reproducible runs.
- **Occupancy of the memory-side fabrics** (②③④): these carry cache-line traffic at far higher
  rates than DSMEM; token-queue depths and per-cycle injection limits need sizing.

## 7. Verification

1. **IDEAL-mode oracle**: with `VX_CFG_NOC_MODEL=IDEAL` (default), the 9-app L2-mode oracle
   stays byte-identical — co-sim must be a pure opt-in.
2. **Fabric ① correctness**: `dsmem_pingpong` (K=2/K=4) and `dxa_multicast` (gmem_reads = 1×
   tile, lmem_writes = N×) PASS under `OMELET` mode.
3. **Latency cross-check**: Omelet-reported average flit latency vs SimX-observed token
   round-trip on a controlled pingpong (they must agree to within the header/serialization
   overheads).
4. **The knob works**: switching `VX_CFG_NOC_SMTOSM_TOPOLOGY` mesh↔ring changes `stats.txt`
   hops/latency in the physically expected direction with no kernel/test change.
5. **Fabrics ②③④**: run the oracle apps at `--cores=4 --l2cache` under OMELET; results correct,
   and `stats.txt` shows the demand traffic.
6. Boundary checks (`ci/check_sw_sim_boundary.sh`), container-clean, 120 s timeout discipline.

## 8. Out of scope (later tasks / future work)

- Chiplet partitioning of Vortex (activates NoI/NoL; needs the placement/`layout` knobs).
- gem5 FS-style real-workload traffic into Omelet other than through Vortex.
- Energy modeling beyond Omelet's per-link EPB accounting.
