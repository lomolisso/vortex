# Distributed Local Memory (DLMEM) — Proposal

**Status:** in progress. Knob `VX_CFG_EXT_DLMEM_ENABLE`, off by default, RV64 only.

## Why

CUDA's thread-block clusters let the blocks of a cluster load, store and
atomically update each other's shared memory ("distributed shared memory").
Producer/consumer pipelines, cluster-wide reductions and halo exchanges then
stay on chip instead of round-tripping through global memory.

Vortex already has `cluster_dim`, but its clusters are same-core: the members
are co-resident on one core and share only DXA multicast. DLMEM lets the warps
of a thread-block cluster access the local memory of **every block of their
cluster**, across the cores of one hardware cluster. Following AGENTS.md §5,
the programming model is CUDA's: a generic address window,
`cluster.map_shared_rank()`, `cluster.block_rank()` and `cluster.sync()`.

## Address window

| | RV64 |
|---|---|
| local memory (and stack) base | `VX_MEM_LMEM_BASE_ADDR = 0x1_FFFC_0000` (was `0x1_FFFF_0000`) |
| window base | `VX_MEM_DLMEM_BASE_ADDR = 0x2_0000_0000` |
| rank `r` | `BASE + r * STRIDE + offset`, `STRIDE = 1 << VX_CFG_LMEM_LOG_SIZE` |
| window size | `next_pow2(DLMEM_MAX_CLUSTER) * STRIDE`, decoded by prefix |

- The local-memory base moved so that a local memory of up to 256 KB (the
  largest DLMEM allows) is aligned and ends right below the window. It is the
  only change a build without DLMEM sees.
- The window is uncached and lies past the end of device memory.
- Rank `r` always means block `r` of the **caller's** cluster. The software
  map is `vx_cluster_map_shared_rank(p, r) = BASE + r * STRIDE + (p - CTA_LMEM_ADDR)`.
- RV32 is untouched: there is no room for the window, so the feature is a
  build error there.

## Ranks and the programming surface

- `VX_CSR_CTA_CLUSTER_RANK` (`0xCE4`, read-only, DLMEM builds only) holds a
  block's linear index in its cluster, in the KMU's walk order (x fastest,
  then y, then z), and 0 without clusters. The KMU sends it with every
  launch beat and the dispatcher latches it per CTA, because the members of
  a distributed cluster sit on different cores and their slots say nothing
  about their ranks.
- Kernel API: [`vx_dlmem.h`](../../sw/kernel/include/vx_dlmem.h), with
  `vx_cluster_block_rank()`, `vx_cluster_num_blocks()`,
  `vx_cluster_map_shared_rank()` and `vortex::cluster_group`.
  `vx_spawn2.h`'s `get_cluster_rank()` reads the CSR in a DLMEM build.
- `cluster_group::sync()` arrives with the hardware cluster barrier.
- Host: `VX_ISA_EXT_DLMEM` (`VX_CFG_MISA_EXT` bit 12) and
  `VX_CAPS_DLMEM_MAX_CLUSTER` (device-caps bits [48:42], 0 without DLMEM).

## Configuration

| Knob | Default | Meaning |
|---|---|---|
| `VX_CFG_EXT_DLMEM_ENABLE` | off | the feature |
| `VX_CFG_DLMEM_MAX_CLUSTER` | `min(8, NUM_CORES)` | largest distributed cluster (CUDA's portable limit is 8) |
| `VX_CFG_DLMEM_STORE_CREDITS` | 8 | remote-store credits (arrives with the network) |
| `VX_CFG_DLMEM_PENDING_SIZE` | 8 | remote loads in flight per core (arrives with remote loads) |

Illegal combinations stop the build in the RTL (`` `error`` or an elaboration
`$fatal`), in SimX (`static_assert`) and, for XLEN, in `vx_dlmem.h`
(`#error`), always with the same message:

- `DLMEM requires XLEN 64`
- `DLMEM requires the local memory`
- `DLMEM does not support the raster extension`
- `DLMEM_MAX_CLUSTER must be between 1 and the cores of a cluster`
- `DLMEM requires SOCKET_SIZE <= NUM_CORES`
- `DLMEM requires a local memory of at most 256 KB`

With the knob off, RTL and SimX behaviour and cycles are unchanged.

## Placement

- **Opt-in per launch.** `vx_launch_info_t.cluster_policy` is
  `VX_CLUSTER_POLICY_DEFAULT` (today's same-core clusters) or
  `VX_CLUSTER_POLICY_SPREAD` (CUDA's spread policy), written to the KMU
  through `VX_DCR_KMU_CLUSTER_POLICY`. `struct_size` versioning keeps old
  clients working.
- **All or nothing.** A gang is admitted only when all K blocks fit; a
  partial gang could deadlock on its own cluster barrier.
- **Spread first.** Blocks go to distinct cores of one hardware cluster. Two
  blocks of a gang share a core only when that is the only room, and a core
  may host blocks of several gangs.
- **Limits.** K ≤ `DLMEM_MAX_CLUSTER` and K fits the hardware cluster,
  otherwise the launch fails with `VX_ERR_INVALID_VALUE`. A same-core
  cluster must fit one core (K × warps per block ≤ `NUM_WARPS`) and fails
  the same way.
- **Freeing.** Slots free per block. A gang id is recycled only after all K
  blocks are done. As in CUDA, a block must not be accessed after it exits,
  so a cluster syncs before its blocks exit.

## Memory-model contract

- **Path.** A per-core translation stage resolves window addresses to
  `{core, slot}` through the block's peer row. Local peers stay in the core;
  remote ones travel on separate request and response networks of one
  hardware cluster (a switch per socket, a crossbar across sockets).
- **Liveness.** Response space is reserved before a request is accepted, an
  ack is a counter decrement, and the request graph is acyclic, so
  responses always drain.
- **Order.** Accesses from one warp to one destination stay in order (fixed
  routes, FIFO buffers). A thread's store and later load to the same remote
  address are ordered.
- **Visibility.** Each warp counts its remote stores that are not yet
  acknowledged; a store is acknowledged once committed at the destination
  bank. FENCE, barriers, cluster arrive and warp exit wait for the count to
  reach zero, so `cluster.sync()` orders all earlier remote stores before the
  peers' later accesses.
- **Atomics.** Remote AMOs (and the Zacas CAS) execute at the destination
  bank. Remote LR/SC is unsupported: a runtime assertion in simulation; in
  hardware LR acts as a load and SC always fails.
- **Errors.** A rank outside the cluster is a runtime assertion in the RTL and
  aborts SimX with the same message.

## Exclusions

RV32; `EXT_RASTER`; DXA multicast into a spread gang (until DXA × DLMEM).

## Open items

- **Cluster-size check.** The runtime rejects a grid that is not a multiple
  of the cluster, but not a cluster too large. The RTL sizes the rank and
  cluster size for `max(NUM_WARPS, DLMEM_MAX_CLUSTER)` and wraps above it,
  while SimX does not, so such a launch reads different ranks on the two
  models. Step 3's launch validation (`queue.cpp`) adds the limits above,
  with a negative app case.
- **Parity on wider shapes.** SimX and the RTL already disagree with the
  knob off, at XLEN 64 before DLMEM: 2 cores with an L2 on clustered
  launches (`multikernel -c`: 5907 SimX vs 5371 RTL cycles, 10%) and 8 warps
  per core (`sgemm2 -n32`: 350767 vs 430551, 18.5%). `cluster_rank` shows the same
  gaps there (17.5% and 21.5%), so its parity case runs 1 core at 4 warps.
  They must be modelled before a gang-launch parity case covers several
  cores.

## Steps

Each step lands RTL and SimX together, test-first: a unit testbench, a SimX
unit test where the model has a unit of its own, a regression app, CI cases
in `ci/testcases/dlmem.yaml` and a `model_parity` case at XLEN 64.

1. Upstream fixes DLMEM depends on: the local-memory switch's duplicate push,
   and the dispatcher's slot pointer at each launch.
2. Configuration, ABI, the rank CSR, the RV64 base move, parity at XLEN 64.
3. Gang admission and the distributed launch.
4. The hardware cluster barrier (`cluster.sync()`).
5. Window translation for peers on the same core.
6. A fence the last-level cache answers without walking.
7. The network, remote stores, acknowledgements and the fence contract.
8. Remote loads.
9. Remote atomics.
