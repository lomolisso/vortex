# 01 — The Simulation Framework

**Scope:** the discrete-event engine that *every* SimX component is built on, all defined in one
header: [`../../common/simobject.h`](../../common/simobject.h). Read this first — `Cluster`,
`Core`, `Cache`, every functional unit, and every accelerator are all `SimObject`s that talk
exclusively through `SimChannel`s driven by one `SimPlatform` clock. Nothing in the rest of the
simulator makes sense without these three types.

---

## 1. The three primitives

| Type | Role |
|---|---|
| `SimObject<Impl>` | A clocked module. Gets an `on_tick()` called once per cycle. |
| `SimChannel<Pkt>` | A typed, bounded FIFO link between modules. The *only* way modules communicate. |
| `SimPlatform` | The singleton engine: owns all objects, advances the clock, delivers channel traffic via a timing wheel. |

Everything is event-driven and cycle-stepped: there is no global "netlist evaluation"; instead,
each module reacts in `on_tick()` to packets that have arrived on its input channels and pushes
packets onto its output channels with a delay.

---

## 2. `SimObject` — a clocked module

`SimObjectBase` ([`simobject.h:50`](../../common/simobject.h#L50)) is the abstract node: it holds
a `name()` and two private virtuals, `do_reset()` and `do_tick()`, callable only by `SimPlatform`
(friendship, [`:64`](../../common/simobject.h#L64)).

Concrete modules derive from the CRTP template `SimObject<Impl>`
([`:454`](../../common/simobject.h#L454)):

```cpp
class Core : public SimObject<Core> {
  // ...
protected:
  void on_tick();           // called once per cycle by SimPlatform
  void on_reset();          // called on reset
  friend class SimObject<Core>;   // REQUIRED for override-detection (see §2.2)
};
```

### 2.1 Construction — `Create()` and the `SimContext` capability

Modules are never `new`-ed directly. `SimObject<Impl>::Create(args...)`
([`:460`](../../common/simobject.h#L460)) forwards to `SimPlatform::create_object<Impl>()`
([`:507`](../../common/simobject.h#L507)), which `make_shared`s the object, registers it in
`objects_`, and returns a `shared_ptr<Impl>`. Constructors take a `const SimContext&` as their
first parameter ([`:465`](../../common/simobject.h#L465)); `SimContext` has a private constructor
and only befriends `SimPlatform` ([`:42`](../../common/simobject.h#L42)), so **the only way to
build a `SimObject` is through `Create()`** — a compile-time capability check.

### 2.2 The "active subset" optimization

`on_tick()`/`on_reset()` are *protected no-ops* in the base ([`:470`](../../common/simobject.h#L470)).
At `create_object` time, SFINAE (`has_own_tick`/`has_own_reset`,
[`:481`](../../common/simobject.h#L481)) detects whether `Impl` actually *overrides* them by
comparing member-function pointers against the base. Only overriders are pushed into
`active_tick_` / `active_reset_` ([`:517`](../../common/simobject.h#L517)). The per-cycle loop then
ticks *only* those modules — a pure-wiring object (one that just forwards channels) costs nothing
per cycle.

> **Gotcha:** override detection works only for **direct** `SimObject<Impl>` CRTP derivatives
> (`has_direct_simobject_base`, [`:499`](../../common/simobject.h#L499)). A multi-level chain
> (`Derived → Intermediate → SimObject<Intermediate>`) can't see the protected pointer, so it is
> *conservatively kept always-active* ([`:522`](../../common/simobject.h#L522)). Two more rules,
> enforced by `static_assert` in `create_object` ([`:509`](../../common/simobject.h#L509)):
> `on_tick`/`on_reset` **must stay protected** (only `SimPlatform` may call them), and the `Impl`
> **must** `friend class SimObject<Impl>` or override detection silently fails.

---

## 3. `SimChannel<Pkt>` — typed, bounded links

A `SimChannel<Pkt>` ([`:227`](../../common/simobject.h#L227)) is a one-way FIFO carrying packets
of type `Pkt` (must be copy-constructible). Default capacity is **2**
([`:233`](../../common/simobject.h#L233)). It is the sole inter-module medium.

### 3.1 Producer side: `send` / `try_send`

`send(pkt, delay=1)` ([`:291`](../../common/simobject.h#L291)) asserts the channel isn't full,
`reserve()`s a slot, then asks `SimPlatform` to schedule delivery `delay` cycles out. `try_send`
([`:303`](../../common/simobject.h#L303)) is the non-asserting form — returns `false` if full.
**The producer must check `full()` first** (or use `try_send`); sending into a full channel trips
an assert.

### 3.2 Consumer side: `peek` / `pop` / `try_pop`

`peek()` reads the front without removing; `pop()` removes; `try_pop(&out)`
([`:332`](../../common/simobject.h#L332)) does both and returns `false` if empty. `empty()` and
`size()` report occupancy.

### 3.3 Backpressure — `full()`, `reserve()`, and `pending_count_`

This is the heart of timing realism. Occupancy is `queue_size() + pending_count_`
([`:392`](../../common/simobject.h#L392)): packets *physically resident* plus packets *in flight*
(scheduled but not yet delivered). `send()` calls `reserve()` ([`:351`](../../common/simobject.h#L351))
which bumps `pending_count_` immediately, so a slot is consumed *the moment you send*, not when the
packet lands. `full()` ([`:286`](../../common/simobject.h#L286)) compares occupancy to capacity.

**Consequence — cascading stalls.** When a downstream consumer stops popping, its channel fills;
the upstream producer's `full()` returns true; the producer can't `send`, so it stalls in its
`on_tick()`; that backs up *its* inputs, and so on up the pipeline. A single LSU stall thus ripples
back to the scheduler going idle — exactly the behavior you want from a hardware model. This is the
**backpressure discipline** referenced throughout these docs.

### 3.4 Binding, forwarding, and type conversion

Channels can be wired into chains with `bind()`:

- `bind(sink)` ([`:261`](../../common/simobject.h#L261)) — forward this channel's traffic straight
  into `sink`. A forwarded channel is *not* a readable endpoint: reading it asserts
  ("cannot read from a forwarded channel", [`:388`](../../common/simobject.h#L388)).
- `bind(sink, converter)` ([`:266`](../../common/simobject.h#L266)) — forward **with a type
  transform**, e.g. wrapping a `LsuReq` into a `MemReq` as it crosses a boundary. The converter's
  signature is checked at compile time.
- `bind<U>(sink)` ([`:275`](../../common/simobject.h#L275)) — implicit conversion when `Pkt` is
  convertible to `U`.

`receive_packet()` ([`:360`](../../common/simobject.h#L360)) is the delivery hook: it fires the
optional `tx_callback`, then either forwards (converting if needed) or, at a true endpoint,
decrements `pending_count_` and pushes into local storage.

### 3.5 `tx_callback` — sideband observation

`tx_callback(fn)` ([`:283`](../../common/simobject.h#L283)) registers `fn(pkt, cycle)` to fire on
*every* packet that transits the channel, before delivery. This is how the framework hangs
behavior off a wire without a dedicated module — e.g. the DXA completion path snoops LMEM writes
this way to release barriers (doc [09](09-barriers-and-sync.md)).

### 3.6 Port arrays — `make_sim_channels`

`SimChannel` isn't default-constructible (it needs an owner), so a module that needs *N* ports
builds them with `make_sim_channels<Pkt, N>(this)` ([`:424`](../../common/simobject.h#L424)),
producing a `std::array<SimChannel<Pkt>, N>`. You will see `Inputs` / `Outputs` arrays everywhere
in the memory subsystem.

### 3.7 Introspection

`SimChannelBase` ([`:68`](../../common/simobject.h#L68)) exposes `module()`, `sink()`, `source()`,
and occupancy queries, so the topology can be walked. A process-global
`inflight_count()` ([`:83`](../../common/simobject.h#L83)) tracks every sent-but-not-consumed
packet — a cheap "is the machine drained?" signal used at kernel teardown.

---

## 4. `SimPlatform` — the clock and the timing wheel

`SimPlatform::instance()` ([`:170`](../../common/simobject.h#L170)) is the singleton engine. It
owns `objects_`, the active subsets, the current `cycles_`, and the event structures.

### 4.1 Events

Channel sends become events. `SimEventBase` ([`:107`](../../common/simobject.h#L107)) carries a
fire-cycle and a `fire()` virtual. Two concrete kinds, both **pool-allocated**
([`:138`](../../common/simobject.h#L138), [`:161`](../../common/simobject.h#L161)) to avoid malloc
churn:

- `SimChannelEvent<Pkt>` ([`:121`](../../common/simobject.h#L121)) — deliver `pkt` to a channel;
  `fire()` calls `channel_->receive_packet(pkt_)` ([`:662`](../../common/simobject.h#L662)).
- `SimCallEvent<Pkt>` ([`:143`](../../common/simobject.h#L143)) — fire an arbitrary callback after
  a delay (the `schedule(func, pkt, delay)` public API, [`:185`](../../common/simobject.h#L185)).

### 4.2 The timing wheel + delta cycles

`schedule()` ([`:529`](../../common/simobject.h#L529)) splits on `delay`:

- **`delay > 0`** → the event goes into a **circular buffer** `reg_events_` of `WHEEL_SIZE = 4096`
  buckets ([`:193`](../../common/simobject.h#L193)), indexed `(cycles_ + delay) & WHEEL_MASK`. O(1)
  insert; the wheel wraps, so an event scheduled ≥ 4096 cycles out lands in an occupied slot and is
  skipped until its real cycle (the `evt->cycles() <= cycles_` check, [`:611`](../../common/simobject.h#L611)).
- **`delay == 0`** → a **delta cycle**: the event goes into `imm_events_` stamped with a
  monotonically increasing `delta_` ([`:531`](../../common/simobject.h#L531)). This models
  *same-cycle* combinational chains with deterministic ordering — multiple zero-delay hops resolve
  within one `cycles_` tick, in send order.

### 4.3 `tick()` — what one cycle does

`tick()` ([`:596`](../../common/simobject.h#L596)) is the whole simulation step:

```
1. fire_immediate_events()            // drain any pending delta-cycle (delay==0) events
2. for each object in active_tick_:   // only modules that override on_tick()
       object->do_tick()              //   → reads inputs, computes, send()s outputs
       fire_immediate_events()        //   drain delta events this module produced
3. ++cycles_
4. fire the reg_events_ bucket for the new cycles_   // deliver delayed channel traffic
```

Two subtleties: (a) immediate events are drained **after every module's tick**, so a 0-delay hop
produced by module A is visible to module B later in the same tick loop; (b) `fire_immediate_events`
([`:646`](../../common/simobject.h#L646)) replays its queue in `delta_` order so a chain of N
zero-delay hops resolves in N ordered passes within the cycle.

### 4.4 `reset()` and `cleanup()`

`reset()` ([`:567`](../../common/simobject.h#L567)) clears all pending events, calls `do_reset()`
on the `active_reset_` subset, and zeroes `cycles_`/`delta_`. `cleanup()`/`finalize()`
([`:622`](../../common/simobject.h#L622)) additionally drops all objects — used between runs.

---

## 5. The functional-vs-timing split (read this twice)

SimX is **not** a pure timing model and **not** a pure functional model — it is both, deliberately
separated, and the framework above is what makes the separation clean:

- **Functional** — the architectural *result* of an instruction (the value written to a register,
  the bytes written to memory) is computed *immediately and exactly* when the instruction executes.
  This lives in the emulator/execute path (doc [10](10-functional-emulator.md)) and the unit
  `execute()` methods (doc [07](07-functional-units.md)).
- **Timing** — *how long* things take is modeled by **delaying packets through channels**. A unit
  computes its result in "zero cycles" functionally, then `send`s the response with
  `delay = latency_of(...)`, so consumers don't observe it until the right cycle.

Because the two are separate, the simulator's *answers* are always correct regardless of the timing
model, and you can refine timing without risking functional correctness. The `instr_trace_t` that
threads through the pipeline (doc [06](06-instruction-lifecycle.md)) is an **immutable snapshot** of
an already-computed instruction whose only job downstream is to model occupancy and latency — there
is no instruction "state machine" mutating as it flows.

---

## 6. Reading the rest of the docs with this lens

Everywhere you see `X.send(pkt, latency)` and `Y.try_pop(&pkt)`, you are looking at a hardware link
with a latency and a depth. Everywhere you see a module's `on_tick()`, you are looking at one
cycle of that block's behavior. The architecture docs (02–17) are, at bottom, just descriptions of
*which* `SimObject`s exist, *which* `SimChannel`s connect them, and *what each `on_tick()` does* —
all resting on this header.

**Next:** [02 — Architecture Hierarchy](02-architecture-hierarchy.md).
