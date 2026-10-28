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
- `cluster_group::sync()`, `barrier_arrive()` and `barrier_wait()`, and their
  `_relaxed` forms, use the hardware cluster barrier (below).
  `vx_fence_cta()` is CUDA's `__threadfence_block()` (memory-model contract
  below).
- Host: `VX_ISA_EXT_DLMEM` (`VX_CFG_MISA_EXT` bit 12) and
  `VX_CAPS_DLMEM_MAX_CLUSTER` (device-caps bits [48:42], 0 without DLMEM).

## Configuration

| Knob | Default | Meaning |
|---|---|---|
| `VX_CFG_EXT_DLMEM_ENABLE` | off | the feature |
| `VX_CFG_DLMEM_MAX_CLUSTER` | `min(8, NUM_CORES)` | largest distributed cluster (CUDA's portable limit is 8) |
| `VX_CFG_DLMEM_STORE_CREDITS` | 8 | remote stores in flight per core, unacknowledged |
| `VX_CFG_DLMEM_PENDING_SIZE` | 8 | remote loads and atomics in flight per core |

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

## Cluster barrier

CUDA's `cluster.sync()`, or PTX `barrier.cluster.arrive` then
`barrier.cluster.wait`: one hardware barrier per block, with split arrive and
wait.

- **Encoding.** The existing `BAR` instructions with `rs1[30] = 1`:
  `rs1 = (1 << 30) | CTA_ID` (the block's slot), `rs2` = the block's warp
  count for the sync form and for arrive, the phase token for wait. Bit 30 is
  decoded only in a DLMEM build; with bit 31 (global), or with the
  transaction-count form, it is a runtime assertion, and SimX aborts.
- **One barrier per block, no id** (as in CUDA). The id bits of `rs1` must be
  0, and a non-zero id is a runtime assertion. A block may still use
  `__syncthreads()` and its own block barriers between an arrive and its wait;
  they are separate state.
- **Kernel API** ([`vx_dlmem.h`](../../sw/kernel/include/vx_dlmem.h)):
  `vx_cluster_sync()`, `vx_cluster_arrive()` (returns the token) and
  `vx_cluster_wait(token)`, and the fence-free `vx_cluster_arrive_relaxed()`
  and `vx_cluster_wait_relaxed(token)`; in C++ `cluster_group::sync()`,
  `barrier_arrive()`, `barrier_wait(token)`, `barrier_arrive_relaxed()` and
  `barrier_wait_relaxed(token)`. Every warp of every block of the cluster
  makes one arrive and one wait, or one sync, per phase, as `__syncthreads()`
  requires.
- **Completion.** A block's arrival counts when its last warp arrives. For a
  SPREAD cluster with K > 1, the core then sends `CBAR{gang_id, epoch}` on the
  global-barrier queue, the same path and order as the block's done event. The
  gang unit counts K of them and sends one release `{gang_id, epoch}` back
  down the global-barrier response fork; each core releases its slots of that
  gang that are waiting. A same-core cluster, and any K = 1 block, completes
  inside the core in the cycle of the last arrival and never uses the bus.
- **Stale releases.** A release carries the gang's epoch and matches only
  slots still waiting on that gang; each admission resets its slot's barrier
  state, and a gang id is reused only after its release has left. A block
  that arrives and then exits is fine: the release finds nothing to wake.
- **Memory.** An arrival counts only once the warp's earlier instructions have
  committed and the core's LSU queues are empty, so the earlier local and
  window stores have left the core's LSU and the earlier loads and atomics
  have returned; a warp leaves its wait only after every warp of the cluster
  has arrived (arrive = release, wait = acquire). Like every barrier arrival
  it is also a CTA-scope release of global memory (memory-model contract
  below), which reaches only the core's own L1. Since the sockets' L1s are
  not coherent, the default forms follow CUDA's defaults for cluster scope:
  `sync()` puts `vx_fence()` before the barrier and after it,
  `barrier_arrive()` fences before arriving (PTX `barrier.cluster.arrive`,
  `.release`) and `barrier_wait()` after waiting (`barrier.cluster.wait`,
  `.acquire`). The `_relaxed` forms have no fence (PTX `.relaxed`): across
  cores they order only local and window memory.
- **Remote stores.** An arrival also waits until every remote store of the
  warp is acknowledged, i.e. committed in the destination bank (see the
  network below), so the barrier orders them with no change to its own logic.
- **Local accesses.** A `BAR`, a cluster arrive, an exit, a `FENCE`, a `.rl`
  AMO and an mbarrier arrive also wait until every earlier local-memory
  access of the warp (load, store, atomic, mbarrier op) has committed in its
  bank, so none still queued in the local-memory switch or a bank queue can
  be overtaken by a peer's later access to the same word.
- **Global memory and tensor-core loads.** Each of these releases is first
  a CTA fence (memory-model contract below), so it also orders the warp's
  earlier global accesses and `TCU_LD` metadata loads at CTA scope.
- **Not covered.** A wgmma reads its local-memory operands asynchronously
  after it issues; no release waits for it. Software waits for the wgmma's
  group (`vx_wgmma_wait_group`) before releasing a buffer the wgmma reads, as
  on the H100.
- **Exit hold and plain `BAR`.** In a DLMEM build a warp cannot exit, and a
  plain `BAR` cannot arrive, while the warp has instructions in flight or
  unacknowledged remote stores.
- **Co-residency of same-core clusters.** A DEFAULT cluster is resident all
  at once only if `K <= usable slots` and `K * W <= NUM_WARPS`. Otherwise the
  dispatcher clamps its window and admits blocks as slots free, so a cluster
  barrier in it never completes (a CUDA launch would fail). This is
  documented, not rejected: DEFAULT launches without a cluster barrier must
  keep working. SPREAD clusters are co-resident by construction.
- **Undefined behaviour.** A block that exits without arriving leaves its
  peers waiting forever, as in CUDA; if the gang ends with a phase only part
  arrived, the gang unit's always-on assertion fires. A second arrival in a
  phase that has not completed is a runtime assertion.

## Window translation in the core

Each core resolves window addresses in one registered stage per LSU block,
between the slice and its memory scheduler
([lsu_pipeline_design.md §4.6](../designs/lsu_pipeline_design.md)); SimX
models it as `DlmemXlate`, behind an `LsuUnit`-owned channel.

- **Peer rows.** The dispatcher writes each admitted block's row (K and every
  rank's `{core, slot}`) and the launch's footprint into the core's peer
  table. A same-core (DEFAULT) cluster's row is its consecutive slots; a
  SPREAD block's row comes from the gang unit.
- **Same-core peers** stay in the core: the lane is steered into the peer
  slot's local memory at the same offset.
- **Remote peers.** A lane whose peer is on another core is marked remote
  (`is_addr_remote`, never `is_addr_local`), its address rewritten to
  `{core, local-memory byte}` inside the window and its hart id to the
  issuer's; it leaves through the local-memory switch's remote leg to the
  core's egress (next section). On an idle machine every SPREAD peer with
  K > 1 is on another core (spread first).
- **Fence hold.** A fence is held at the stage output until its warp has no
  remote access left to perform: every remote store acknowledged, every
  remote load, atomic and SC answered, and its local accesses committed. The
  warp's counter rises when the access leaves the stage and its drained flag
  is registered, so a fence right behind an access of the same warp always
  waits for its completion.
- **Out of bounds.** A rank at or past K inside the window, or an offset past
  the block's footprint, is the runtime assertion `DLMEM access out of
  bounds`; the lane is steered into the issuer's own slot. A DEFAULT cluster
  with K above `DLMEM_MAX_CLUSTER` gets a row of `DLMEM_MAX_CLUSTER` ranks;
  its ranks from `next_pow2(DLMEM_MAX_CLUSTER)` up fall past the window and
  are ordinary global addresses, as is every address past the window.
- **Timing.** One cycle per LSU request, fences included, in DLMEM builds, in
  both models. The stage's empty is folded into the LSU empty that the barrier
  drain, the flush gate and core busy use.
- **Warp exit** waits for the warp's earlier instructions and its
  unacknowledged remote stores to drain, so a slot admitted again cannot lend
  its new row to a store of the warp that left it.
- **The tensor unit's** requests bypass the stage. With `TCU_META` the RTL
  arbitrates them against the stage round-robin in the scheduler, while SimX
  orders same-cycle arrivals at the local-memory switch canonically: correct,
  not cycle-matched, and not tested.
- **Tests.** [`tests/regression/dlmem_local`](../../tests/regression/dlmem_local)
  runs six tests, each its own launch, over every rank pair of each cluster:
  own writes read by every peer, stores into every peer, an `amoadd` into rank
  0, global, own and window lanes in one store and one load, sub-word stores
  at the slice and line edges of every peer, and the self window against the
  plain pointer. It sweeps 64 B, 1 KB and SRAM/K per block, has an exit-stress
  mode and the out-of-bounds negative mode, and syncs a same-core cluster on a
  local barrier of its base slot (`-c` uses `cluster_group::sync()`). With
  `-s` and K > 1 the same tests run between cores, on `cluster_group::sync()`.
- **Parity** (`parity-dlmem-local`, `parity-dlmem-local-4c`): fixed loops of
  window loads, stores and atomics, at one core with K = 1 and at four cores
  with same-core clusters of 4. Both models retire the same instructions;
  the cycle gaps are 2.9% (29270 SimX vs 28435 RTL) and 2.8% (30575 vs 29731).

## Network, remote stores, loads and atomics

Each core of a hardware cluster has one egress, one ingress and one store
tracker; two separate networks (requests and responses), each a switch per
socket and a crossbar across sockets, connect them
(`hw/rtl/dlmem/`, SimX `sim/simx/dlmem/`).

- **Path.** A remote lane leaves the translation stage, takes the
  local-memory switch's third (remote) leg, and reaches the egress, which
  sends one packet per lane on the request network. The destination's
  ingress reserves a response slot, then writes its own input of the local
  memory, which round-robins with the core's LSU lanes. A load's or atomic's
  data, or a store's acknowledgement, returns on the response network to the
  egress, which hands loads and atomics back to the LSU as partial responses.
- **The switch is all or nothing** across its local, global and remote sides,
  as with two; its response merge takes the three sides with no side starved.
- **Store acknowledgements.** The local memory answers a store from the
  ingress input the cycle after the bank write, so an acknowledgement means
  committed and visible to every core. The access tracker counts each warp's
  remote accesses from the translation stage on until a store's
  acknowledgement or a load's or atomic's data returns; FENCE, `BAR`,
  cluster arrive and warp exit wait for the count to be zero.
- **Credits.** The egress holds `VX_CFG_DLMEM_STORE_CREDITS` stores and
  `VX_CFG_DLMEM_PENDING_SIZE` loads and atomics in flight; its response sink
  never refuses.
- **Atomics** (and the Zacas CAS) execute at the destination bank, carrying
  the issuer's hart id, so a remote store or atomic breaks other harts'
  reservations there. Remote LR/SC is a runtime assertion,
  `remote LR/SC is not supported`, in both models; in hardware without the
  assertion LR acts as a load and SC always fails.
- **Deadlock.** Responses always drain: response space is reserved before a
  request is accepted, the ingress' local-memory responses are never refused,
  and the remote leg back-pressures only the LSU request path. A held fence
  waits only on its warp's earlier remote accesses, which already left the
  stage and need other cores' ingress and local memory, the networks, this
  core's always-ready response sink, and egress pending entries that free as
  the LSU drains its responses.
- **Tests.** [`dlmem_store`](../../tests/regression/dlmem_store) (all-to-all
  stores of every width, and a producer/consumer ring on `vx_fence()` and a
  flag), [`dlmem_load`](../../tests/regression/dlmem_load) (every slice at
  every width, ping-pong, mixed global/local/remote lanes, a thread's store
  then load), [`dlmem_atomic`](../../tests/regression/dlmem_atomic)
  (histogram and tickets, an `amoswap` lock, every AMO at both widths, CAS,
  and the `-x lrsc` negative) and
  [`dlmem_stress`](../../tests/regression/dlmem_stress) (a seeded random mix
  against a model, with optional DXA copies), all SPREAD; and
  `dlmem_local -s`. Each fails a launch in which no cluster spans two cores.
  Parity: `parity-dlmem-store`, `parity-dlmem-load`, `parity-dlmem-atomic`,
  and `parity-dlmem-atomic-fence` (a fence after each iteration's atomics).

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
- **Visibility.** Each warp counts its remote accesses not yet performed; a
  load or atomic is performed once its data returns, and a store once it is
  acknowledged, i.e. committed at the destination bank. FENCE, barriers,
  cluster arrive and warp exit wait for the count to reach zero, so a fence
  orders every earlier remote access before the warp's later ones, and
  `cluster.sync()` orders them before the peers' later accesses.
- **Atomics.** Remote AMOs (and the Zacas CAS) execute at the destination
  bank. Remote LR/SC is unsupported: a runtime assertion in simulation; in
  hardware LR acts as a load and SC always fails.
- **Errors.** A rank outside the cluster is a runtime assertion in the RTL and
  aborts SimX with the same message.
- **CTA-scope release.** In a DLMEM build every release is a CTA-scope
  release of global memory too, like PTX `mbarrier.arrive`, `atom.release`
  and `bar.sync`: an mbarrier arrive (and `arrive_tx`), an AMO, SC or LR with
  `.rl`, and every `BAR` (`__syncthreads()`, block barriers, the cluster
  barrier, the async arrive). The async-arrive encoding also carries
  `vx_barrier_expect_tx` (its count is a register value, so decode cannot
  tell them apart), so `expect_tx` is a release here too, unlike on the
  H100. The uop sequencer issues each one as a CTA fence micro-op followed
  by the op itself, at the cost of one issue slot, plus a drain when the
  warp has written global memory. The op then keeps its own holds (the
  local-access and remote counts above).
- **CTA fence.** `vx_fence_cta()` (custom-2, `funct3 = 7`; CUDA's
  `__threadfence_block()`) is the fence alone. In the warp's LSU slice it
  waits until the warp's global loads, LRs and AMOs have returned. If the
  warp has written global memory since its last fence (or since it was
  started, by a CTA dispatch or a `wspawn`), it then sends one fence token
  marked CTA-scope; the L1 data cache drains it (its older requests commit,
  as at the shared level) and answers it as a read of its word, without
  walking, invalidating or forwarding. Otherwise it sends nothing. The token
  rides every lane of the slice, whatever the warp's thread mask, so each
  data-cache channel carries a copy behind the warp's older requests on it:
  the channels reach the L1 independently, and a drain covers only the
  requests the L1 has already accepted. The fence completes when every
  copy has answered. In a build with tensor metadata a CTA fence, and every `FENCE`,
  also waits through the scoreboard for the warp's earlier `TCU_LD`
  writebacks (and, since they share its bits, for in-flight `fflags`
  writers). Without a data cache the token is a GPU fence.
- **What a CTA release orders.** Global memory, i.e. every lane address that
  is neither local memory, nor the DLMEM window, nor the stack window
  (`STACK_WINDOW_BOTTOM` to `STACK_WINDOW_TOP`). Stacks hold CUDA's local
  memory, which is private to its thread, so a release does not order a
  thread's stack stores before another thread's reads of them through a
  pointer, as in CUDA. A thread that shares stack data must fence it with
  `vx_fence()` itself.
- **Acquire at CTA scope.** The block shares one L1, so a load issued after
  the acquiring wait (an mbarrier `test_wait` or `try_wait` branch, a barrier) sees the
  released data with no invalidate. Cluster and device scope still need
  `vx_fence()` on both sides, as above.
- **Fence lock.** While the token is out, the slice holds its fence lock, so
  the other warps of that LSU slice cannot send memory requests. The H100
  does not stall other warps this way; it costs only when the releasing warp
  has written global memory.

## Suspending mbarrier waits

- **Op.** `vx_mbar_try_wait(p, token)` returns true once the phase `token`
  was taken in has completed, like PTX `mbarrier.try_wait`: while the phase
  is incomplete the warp may suspend, and false means a time limit passed
  first. The instruction (custom-2, `funct3 = 5`, always `.D`) answers with
  the phase as its bank read it, before the warp suspends, so its answer is
  false whenever the warp slept. The library follows a false answer with one
  `test_wait`, so the call is true when the phase completed before the warp
  resumed. `vx_mbar_try_wait_parity` passes the parity as the token's phase
  bit. `vortex::mbarrier::wait()` and `wait_parity()` loop on it, like
  `cuda::barrier::wait()` and CUTLASS's `ClusterBarrier::wait()`.
  `test_wait` is unchanged and never suspends.
- **The LSU is never held.** The op decodes as a warp stall, so the warp
  issues nothing after it until it is released. The bank answers it at once,
  read-only, like `test_wait`. Suspended warps therefore hold no LSU entry
  and cannot block the arrive that frees them.
- **Events.** Each local-memory bank reports, in its own op order, an
  *insert* `{wid, row}` when a `try_wait` lane reads an incomplete phase, and
  a *flip* `{row}` when an mbarrier op completes a phase. The LSU reports a
  *retire* `{wid}` when the op's last response leaves, so every lane has
  read the object by then.
- **Waiter table.** One per core (RTL `VX_mbar_waiters` in `VX_mem_unit`,
  SimX `MbarWaiterTable`), one entry per warp `{bank, row, woke, retired,
  timer}`.
  - An insert fills the warp's entry; the last lane's insert wins.
  - A flip marks every entry on its `{bank, row}` woken.
  - The retire releases the warp at once if no lane inserted or the entry
    is already woken; otherwise it starts the timer.
  - A woken retired entry, or an expired timer, releases the warp.
  - No wake-up is lost: an object lives in one bank, which orders its ops,
    so a flip either comes before a lane's read (the read returns true) or
    after that lane's insert, on the same event stream.
- **Timing.** The table decides in cycle T, sends a registered unlock mask to
  the scheduler at T+1, and the warp can be scheduled at T+2, in both models
  (checked on debug traces of both). The library's `test_wait` then reads
  the phase complete after a wake-up, and incomplete after a time-out unless
  the phase completed meanwhile.
- **Time limit.** `VX_CFG_MBAR_SUSPEND_CYCLES`, 1024 by default. A build check
  keeps it below a quarter of the scheduler's stall timeout, so a core whose
  warps all sleep keeps waking up (`mbar_wait -m sleep` sleeps through 100
  time limits). A real mbarrier hang therefore no longer trips the stall
  assertion: its warps wake every time limit and poll again until the host
  times out, as a `test_wait` spin did.
- **Lanes.** Every active lane reads the object. Lanes can see different
  answers when the phase completes between their reads; the warp is released
  once, and the library's `test_wait` returns true in the lanes that saw it
  incomplete. When the lanes of one warp wait on different objects, the warp
  is woken when the object read by its last waiting lane completes, or at
  the time limit; the other lanes may then return false early and poll
  again, which is correct, only slower.
- **Gaps.** No `suspendTimeHint`: the knob is the only limit. The build
  check does not cover the scheduler's `EXT_SCHED_STALL_TIMEOUT` override
  (`VX_scheduler.sv`), which can set a stall timeout below four time limits.
  No wait on a
  remote mbarrier; PTX allows waits only on the executing CTA's objects, and
  remote mbarrier ops still assert. The limit counts core cycles, not
  nanoseconds. Wake-ups by DXA completion and by remote arrives will come
  through the same flip events once those ops reach the banks.

## Exclusions

RV32; `EXT_RASTER`; DXA multicast into a spread gang (until DXA × DLMEM).

## Open items

- **Parity on wider shapes.** SimX and the RTL already disagree with the
  knob off, at XLEN 64 before DLMEM: 2 cores with an L2 on clustered
  launches (`multikernel -c`: 5907 SimX vs 5371 RTL cycles, 10%) and 8 warps
  per core (`sgemm2 -n32`: 350767 vs 430551, 18.5%). `cluster_rank` shows the same
  gaps there (17.5% and 21.5%), so its parity case runs 1 core at 4 warps.
  `cluster_sync` too: with 8 warps it differs by 15.7% even with K = 1
  (`-p -d -k 1 -w 2 -n 24`: 501865 SimX vs 595664 RTL cycles), so its
  same-core parity case also runs at 4 warps.
  They must be modelled before a gang-launch parity case covers several
  cores.
- **Fence cost on several sockets without an L2.** With no shared level, a
  fence walks every data cache. The two models then cost it differently: at
  4 cores with SOCKET_SIZE=2 a fence loop takes 93162 SimX vs 141507 RTL cycles,
  with the knob off (1 core agrees within 0.9%). `cluster_group::sync()` puts a
  fence on each side of the barrier, so its parity case runs the relaxed
  arrive/wait (3.4%) until the fence model covers this shape.

## Steps

Each step lands RTL and SimX together, test-first: a unit testbench, a SimX
unit test where the model has a unit of its own, a regression app, CI cases
in `ci/testcases/dlmem.yaml` and a `model_parity` case at XLEN 64. All nine
have landed.

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
