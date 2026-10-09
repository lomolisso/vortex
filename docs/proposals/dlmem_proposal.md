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
- `DLMEM requires NUM_CORES to be a multiple of SOCKET_SIZE`
- `DLMEM requires a local memory of at most 256 KB`

With the knob off, RTL and SimX behaviour and cycles are unchanged.

## Placement

- **Opt-in per launch.** `vx_launch_info_t.cluster_policy` is
  `VX_CLUSTER_POLICY_DEFAULT` (today's same-core clusters) or
  `VX_CLUSTER_POLICY_SPREAD` (CUDA's spread policy). The runtime writes it to
  `VX_DCR_KMU_CLUSTER_POLICY` (`0x024`; `VX_DCR_KMU_STATE_END` moves to
  `0x025`) on a DLMEM device only, also in the QMD. A `reserved0` word in
  front of the field grows the struct from 88 to 96 bytes, so `struct_size`
  tells the versions apart: an 88-byte caller (prebuilt POCL or Mesa) gets
  DEFAULT.
- **The gang unit.** `VX_cta_gang`, one per hardware cluster in front of its
  socket distributor (SimX: `CtaGang`), mirrors every core's free slots and
  reserved warps for SPREAD blocks. It places a SPREAD cluster in K cycles,
  then sends K beats pinned to their cores, each carrying the forced slot,
  the gang id and the cluster's peer row. Design:
  [cta_dispatch_architecture.md §4.6](../designs/cta_dispatch_architecture.md).
- **All or nothing.** A gang is admitted only when all K blocks fit; a
  partial gang could deadlock on its own cluster barrier.
- **Spread first.** Blocks go to distinct cores of one hardware cluster. Two
  blocks of a gang share a core only when that is the only room, and a core
  may host blocks of several gangs.
- **Limits.** SPREAD on a device without `VX_ISA_EXT_DLMEM` fails with
  `VX_ERR_NOT_SUPPORTED` (CUDA's `cudaErrorNotSupported`). These fail with
  `VX_ERR_INVALID_VALUE`: a policy above SPREAD; SPREAD with K above
  `DLMEM_MAX_CLUSTER` or more than the hardware cluster's cores can hold;
  and, on a DLMEM device, any cluster larger than
  `max(NUM_WARPS, DLMEM_MAX_CLUSTER)`, the range of the KMU's rank counter.
  An empty hardware cluster always holds a valid gang, since
  K ≤ `DLMEM_MAX_CLUSTER` ≤ its cores.
- **Same-core clusters** keep today's admission and get no gang id and no
  done events; their cluster barrier completes inside the core. A DLMEM
  launch with DEFAULT therefore stays as close as possible to a build
  without DLMEM.
- **Freeing.** Slots free per block: a block's done event travels through
  the global-barrier path to the gang unit. A gang id is recycled only after
  all K blocks are done, and its epoch bit flips on every allocation, so a
  late event cannot match a reused id. As in CUDA, a block must not be
  accessed after it exits, so a cluster syncs before its blocks exit.
- **One launch at a time.** The command processor drains a launch before the
  next starts, and the gang unit keeps the device busy until its mirror is
  empty, so every launch starts with an empty mirror, its round-robin
  pointer and epochs reset.
- **Head-of-line blocking.** A SPREAD cluster that does not fit holds the
  device's launch stream (the device distributor's message lock and skid sit
  behind it) until enough blocks finish. It is correct but not what CUDA
  does (a later cluster that fits would go first); accepted for Feature 1.

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
