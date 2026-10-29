#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dlmem.h>
#include <vx_mbar.h>
#include <mbar_ref.h>
#include "common.h"

// Clusters of k blocks. Thread 0 of each block sets its objects up and a
// cluster sync publishes them to the peers. In phase p the blocks use
// FULL[p % 2] (and EMPTY[p % 2]), in that object's phase p / 2: a block's ops
// of phase p + 2 on a peer's object follow its own phase p + 1, which needs
// the peer's ops of phase p + 1, which the peer issues only after its own
// wait of phase p. At the end every block meets in a cluster sync, so no op
// reaches a block that has exited, and thread 0 copies the objects out for
// the host, which predicts each final word with mbar_ref.

namespace {

#define INL inline __attribute__((always_inline))

namespace mr = vortex::mbar_ref;

constexpr uint64_t kLimit = VX_CFG_MBAR_SUSPEND_CYCLES;

struct Ctx {
  uint32_t k, rank, b, base, tid, lane, nt, warp, nthreads, R;
  uint64_t* lm;
  vortex::cluster_group cl;
  uint32_t errors, first, what, negs, stale, waits, slept, woke, timeouts;
  uint64_t woke_mask, cyc_rlx, cyc_rel, max_rlx, min_rel;

  INL void fail(uint32_t p, uint32_t code) {
    if (errors++ == 0) {
      first = p + 1;
      what  = code;
    }
  }
  INL void check(bool ok, uint32_t p, uint32_t code) {
    if (!ok)
      fail(p, code);
  }
  INL uint64_t* peer(uint64_t* p, uint32_t r) const {
    return cl.map_shared_rank(p, r);
  }
  INL volatile uint64_t* vpeer(uint64_t* p, uint32_t r) const {
    return peer(p, r);
  }
  // A word returned by an op of the object's phase n: its phase, pending in
  // [1, max_pending] and tx in [tx_lo, tx_hi].
  INL void check_word(uint64_t w, uint32_t n, uint32_t max_pending, int64_t tx_lo, int64_t tx_hi, uint32_t p) {
    const uint32_t pend = mr::pending(w);
    const int64_t t = mr::tx(w);
    check(mr::phase(w) == (n & 1) && pend >= 1 && pend <= max_pending && t >= tx_lo && t <= tx_hi, p, CHK_TOKEN);
    negs += (t < 0);
  }
  // The op that completes a phase of rank tgt's FULL object counts itself in
  // the block's own table, which costs a local atomic, not a global one that
  // would keep the warp from its wait.
  INL void record(uint32_t op, uint64_t old, uint64_t operand, uint32_t tgt, uint32_t p) {
    if (mr::apply(op, old, operand).flip)
      __atomic_fetch_add(reinterpret_cast<uint32_t*>(lm + LM_COMP) + tgt * MAX_ROUNDS + p, 1u, __ATOMIC_RELAXED);
  }
};

INL void idle(uint32_t n) {
  for (uint32_t i = 0; i < n; ++i)
    __asm__ volatile ("");
}

// Zero, computed from t: an op whose operand includes it issues only once t
// is known, so a cycle count read before the op cannot complete after it.
INL uint64_t zero_after(uint64_t t) {
  uint64_t z;
  __asm__ volatile ("and %0, %1, zero" : "=r"(z) : "r"(t));
  return z;
}

// A value no other phase, element or rank stores.
INL uint64_t tag(uint32_t p, uint32_t i, uint32_t rank) {
  return ((uint64_t)(p + 1) << 40) | ((uint64_t)rank << 32) | i;
}

// The bank's own answer to a try_wait: false when it read the phase
// incomplete, and then the warp has slept.
INL bool bank_try_wait(uint64_t* obj, uint64_t tok) {
  uint64_t r;
  __asm__ volatile (".insn r %1, 5, 0, %0, %2, %3"
                    : "=r"(r) : "i"(RISCV_CUSTOM2), "r"(obj), "r"(tok) : "memory");
  return r != 0;
}

INL void ctx_init(Ctx& x, const rem_arg_t* arg) {
  x.k        = arg->k;
  x.rank     = vx_cluster_block_rank();
  x.b        = blockIdx.x;
  x.base     = x.b - x.rank;
  x.tid      = threadIdx.x;
  x.nthreads = blockDim.x;
  x.nt       = vx_num_threads();
  x.lane     = x.tid % x.nt;
  x.warp     = get_sub_group_id();
  x.R        = arg->rounds;
  x.lm       = reinterpret_cast<uint64_t*>(__local_mem());
  x.errors = x.first = x.what = x.negs = x.stale = 0;
  x.waits = x.slept = x.woke = x.timeouts = 0;
  x.woke_mask = x.cyc_rlx = x.cyc_rel = x.max_rlx = 0;
  x.min_rel = ~uint64_t(0);
  if (x.tid == 0) {
    auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + x.b * REC_WORDS;
    rec[REC_BLOCK] = x.b;
    rec[REC_CORE]  = vx_core_id();
    rec[REC_SLOT]  = get_local_group_id();
    rec[REC_RANK]  = x.rank;
    rec[REC_SIZE]  = vx_cluster_num_blocks();
    __atomic_fetch_add(&rec[REC_RAN], 1u, __ATOMIC_RELAXED);
    for (uint32_t i = 0; i < NUM_OBJS; ++i)
      vx_mbar_inval(x.lm + i);
    auto comp = reinterpret_cast<volatile uint32_t*>(x.lm + LM_COMP);
    for (uint32_t i = 0; i < x.k * MAX_ROUNDS; ++i)
      comp[i] = 0;
    reinterpret_cast<volatile uint64_t*>(x.lm)[LM_FLAG] = 0;
    reinterpret_cast<volatile uint64_t*>(x.lm)[LM_DONE] = 0;
  }
}

INL void init_objs(const Ctx& x, uint32_t first, uint32_t n, uint32_t count) {
  if (x.tid == 0) {
    for (uint32_t i = first; i < first + n; ++i)
      vx_mbar_init(x.lm + i, count);
  }
}

// Plain loads of the block's own object until `done` holds for its word: how a
// late rank learns that the peers' earlier ops of the phase have landed.
template <typename F>
INL void poll_own(Ctx& x, uint64_t* obj, uint32_t p, F done) {
  auto w = reinterpret_cast<volatile uint64_t*>(obj);
  for (uint32_t i = 0; !done(*w); ++i) {
    if (i >= POLL_LIMIT) {
      x.fail(p, CHK_WAIT);
      return;
    }
  }
}

// vortex::mbarrier::wait() on the block's own object, with the bank's answer
// to each try_wait in sight: a wait is woken when a try_wait that suspended
// the warp returns true before the time limit. Lane 0 notes the phases whose
// wait was. A parity wait must be exactly one such try_wait.
INL void wait_own(Ctx& x, uint64_t* obj, uint32_t n, uint32_t p, bool exact = false) {
  const uint64_t tok = (uint64_t)(n & 1) << 63;
  ++x.waits;
  for (uint32_t c = 1;; ++c) {
    const uint64_t t0 = vx_rdcycle();
    const uint64_t t = tok | zero_after(t0);
    const bool raw = bank_try_wait(obj, t);
    const bool done = raw || vx_mbar_test_wait(obj, t);
    const uint64_t dt = vx_rdcycle() - t0;
    if (!raw) {
      ++x.slept;
      if (!done && dt >= kLimit)
        ++x.timeouts;
    }
    if (exact)
      x.check(c == 1 && !raw && done && dt < kLimit, p, CHK_CALLS);
    if (done) {
      if (!raw && dt < kLimit) {
        ++x.woke;
        if (x.lane == 0)
          x.woke_mask |= uint64_t(1) << p;
      }
      return;
    }
    if (c >= TRY_LIMIT) {
      x.fail(p, CHK_WAIT);
      return;
    }
  }
}

// The cluster barrier without fences orders local and window memory only,
// which is all a run without global data needs.
INL void meet(const Ctx& x, bool fenced) {
  if (fenced) {
    x.cl.sync();
  } else {
    x.cl.barrier_wait_relaxed(x.cl.barrier_arrive_relaxed());
  }
}

// Every block meets, then thread 0 copies the objects out.
INL void finish(Ctx& x, const rem_arg_t* arg, bool fenced = true) {
  meet(x, fenced);
  if (x.tid == 0) {
    auto words = reinterpret_cast<uint64_t*>(arg->word_addr) + x.b * NUM_OBJS;
    for (uint32_t i = 0; i < NUM_OBJS; ++i)
      words[i] = reinterpret_cast<volatile uint64_t*>(x.lm)[i];
    auto comp = reinterpret_cast<uint32_t*>(arg->comp_addr) + x.b * x.k * x.R;
    auto mine = reinterpret_cast<volatile uint32_t*>(x.lm + LM_COMP);
    for (uint32_t t = 0; t < x.k; ++t)
      for (uint32_t p = 0; p < x.R; ++p)
        comp[t * x.R + p] = mine[t * MAX_ROUNDS + p];
  }
  auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + (x.b * x.nthreads + x.tid) * ERR_WORDS;
  err[ERR_COUNT]    = x.errors;
  err[ERR_FIRST]    = x.first;
  err[ERR_WHAT]     = x.what;
  err[ERR_NEG]      = x.negs;
  err[ERR_STALE]    = x.stale;
  err[ERR_WAITS]    = x.waits;
  err[ERR_SLEPT]    = x.slept;
  err[ERR_WOKE]     = x.woke;
  err[ERR_TIMEOUTS] = x.timeouts;
  err[ERR_WOKE_LO]  = (uint32_t)x.woke_mask;
  err[ERR_WOKE_HI]  = (uint32_t)(x.woke_mask >> 32);
  err[ERR_CYC_RLX]  = (uint32_t)x.cyc_rlx;
  err[ERR_CYC_REL]  = (uint32_t)x.cyc_rel;
  err[ERR_MAX_RLX]  = (uint32_t)x.max_rlx;
  err[ERR_MIN_REL]  = (x.min_rel > 0xFFFFFFFFu) ? 0xFFFFFFFFu : (uint32_t)x.min_rel;
}

} // namespace

// Every thread arrives on every rank's FULL object through the window, its
// own included, then waits on its own object. In phase p rank L = p % k
// arrives last: every rank arrives on L's object after the others, and L
// waits until its own object has every other rank's arrivals, so the other
// ranks are about to wait, then `delay` idle iterations more, so they sleep.
// Its arrive then completes every other rank's phase and must wake them.
__kernel void all_k(rem_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t count = x.k * x.nthreads;
  init_objs(x, OBJ_FULL, 2, count);
  x.cl.sync();
  for (uint32_t p = 0; p < x.R; ++p) {
    const uint32_t s = p & 1, n = p >> 1;
    const uint32_t L = p % x.k;
    if (x.rank == L) {
      // Monotonic, as another warp of the rank may have arrived meanwhile.
      poll_own(x, x.lm + OBJ_FULL + s, p, [&](uint64_t w) {
        return mr::phase(w) == (n & 1) && mr::pending(w) <= x.nthreads;
      });
      idle(arg->delay);
    }
    for (uint32_t j = 0; j < x.k; ++j) {
      const uint32_t tgt = (L + 1 + j) % x.k;
      uint64_t* obj = x.peer(x.lm + OBJ_FULL + s, tgt);
      // The arrive on L stays a release, held until the earlier ones are
      // answered: L counts on it to know every other arrival has landed.
      const bool rlx = arg->relaxed && j + 1 < x.k;
      const uint64_t old = rlx ? vx_mbar_arrive_relaxed(obj, 1) : vx_mbar_arrive(obj, 1);
      x.check_word(old, n, count, 0, 0, p);
      x.record(mr::OP_ARRIVE, old, 1, tgt, p);
    }
    wait_own(x, x.lm + OBJ_FULL + s, n, p);
  }
  finish(x, arg);
}

// A ring: each rank fills a buffer in the next rank's local memory with
// window stores (`burst` per element, the last one the tag), and every thread
// then arrives on the next rank's FULL object; the arrive releases the
// stores. Each rank waits on its own FULL, reads its buffer, and every thread
// arrives on the previous rank's EMPTY object, which the producer waits on
// before it refills that buffer. Rank p % k arrives late in phase p, after
// its stores: once the next rank says it is about to wait, and `delay` idle
// iterations more, so that rank sleeps until the late arrive completes its
// phase.
__kernel void data_k(rem_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t count = x.nthreads;
  const uint32_t E = arg->elems;
  init_objs(x, OBJ_FULL, 4, count);
  x.cl.sync();
  const uint32_t next = (x.rank + 1) % x.k;
  const uint32_t prev = (x.rank + x.k - 1) % x.k;
  for (uint32_t p = 0; p < x.R; ++p) {
    const uint32_t s = p & 1, n = p >> 1;
    if (p >= 2)
      vortex::mbarrier(x.lm + OBJ_EMPTY + s).wait_parity((n - 1) & 1);
    auto buf = x.vpeer(x.lm + LM_DATA + s * E, next);
    for (uint32_t i = x.tid; i < E; i += x.nthreads) {
      for (uint32_t r = 1; r < arg->burst; ++r)
        buf[i] = 0;
      buf[i] = tag(p, i, x.rank);
    }
    if (x.rank == p % x.k) {
      poll_own(x, x.lm + LM_FLAG, p, [&](uint64_t w) { return w > p; });
      idle(arg->delay);
    }
    uint64_t old = vx_mbar_arrive(x.peer(x.lm + OBJ_FULL + s, next), 1);
    x.check_word(old, n, count, 0, 0, p);
    x.record(mr::OP_ARRIVE, old, 1, next, p);

    // Once every thread has arrived, warp 0 tells the previous rank, which
    // may be this phase's late producer, that the block is about to wait.
    __atomic_fetch_add(reinterpret_cast<uint32_t*>(x.lm + LM_DONE), 1u, __ATOMIC_RELAXED);
    poll_own(x, x.lm + LM_DONE, p, [&](uint64_t w) { return (uint32_t)w >= x.nthreads * (p + 1); });
    if (x.warp == 0)
      *x.vpeer(x.lm + LM_FLAG, prev) = p + 1;
    wait_own(x, x.lm + OBJ_FULL + s, n, p);
    auto own = reinterpret_cast<volatile uint64_t*>(x.lm + LM_DATA + s * E);
    bool ok = true;
    for (uint32_t i = x.tid; i < E; i += x.nthreads)
      ok &= (own[i] == tag(p, i, prev));
    x.check(ok, p, CHK_DATA);
    old = vx_mbar_arrive(x.peer(x.lm + OBJ_EMPTY + s, prev), 1);
    x.check_word(old, n, count, 0, 0, p);
  }
  finish(x, arg);
}

// data_k with a fixed schedule: per phase the blocks meet at the cluster
// barrier, then rank p % k waits `delay` idle iterations, fills every other
// rank's buffer and arrives on its FULL object, while those ranks wait. The delay
// puts each wait to sleep before the arrive lands, so a wait is exactly one
// try_wait, which the arrive wakes, and the instruction count is exact. With
// kRelaxed each producer thread also arrives relaxed right behind its stores:
// that arrive is not held, the release arrive after it waits for both, and
// the object counts two arrivals per thread.
template <bool kRelaxed>
INL void pdata(const rem_arg_t* arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t count = (kRelaxed ? 2 : 1) * x.nthreads;
  const uint32_t E = arg->elems;
  init_objs(x, OBJ_FULL, 1, count);
  meet(x, false);
  uint32_t consumed = 0;
  for (uint32_t p = 0; p < x.R; ++p) {
    const uint32_t P = p % x.k;
    if (x.rank == P) {
      idle(arg->delay);
      for (uint32_t j = 1; j < x.k; ++j) {
        const uint32_t c = (P + j) % x.k;
        // Phases before p in which rank c consumed.
        const uint32_t n = p - (p + x.k - 1 - c) / x.k;
        auto buf = x.vpeer(x.lm + LM_DATA, c);
        for (uint32_t i = x.tid; i < E; i += x.nthreads)
          buf[i] = tag(p, i, x.rank);
        uint64_t* obj = x.peer(x.lm + OBJ_FULL, c);
        // Checked after the release arrive, which must not wait for its answer.
        const uint64_t rlx = kRelaxed ? vx_mbar_arrive_relaxed(obj, 1) : 0;
        const uint64_t old = vx_mbar_arrive(obj, 1);
        if (kRelaxed)
          x.check_word(rlx, n, count, 0, 0, p);
        x.check_word(old, n, count, 0, 0, p);
      }
    } else {
      wait_own(x, x.lm + OBJ_FULL, consumed, p, true);
      auto own = reinterpret_cast<volatile uint64_t*>(x.lm + LM_DATA);
      bool ok = true;
      for (uint32_t i = x.tid; i < E; i += x.nthreads)
        ok &= (own[i] == tag(p, i, P));
      x.check(ok, p, CHK_DATA);
      ++consumed;
    }
    meet(x, false);
  }
  finish(x, arg, false);
}

__kernel void pdata_k(rem_arg_t* __UNIFORM__ arg) {
  pdata<false>(arg);
}

__kernel void pdata_rlx_k(rem_arg_t* __UNIFORM__ arg) {
  pdata<true>(arg);
}

// The ring of data_k over global buffers: the producer's stores go to its
// core's L1, and the consumer reads its own buffer between its arrive and its
// wait, so its L1 holds the old lines when the phase's late producer, delayed
// past that read, arrives. Only the cluster-scope release (the arriver's
// stores reach the shared cache) and acquire (the waiter's L1 drops its
// lines) make the new tags visible; without them (fenced == 0) the
// consumer can read stale data, which it counts.
__kernel void global_k(rem_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t count = x.nthreads;
  const uint32_t E = arg->elems;
  init_objs(x, OBJ_FULL, 4, count);
  x.cl.sync();
  const uint32_t next = (x.rank + 1) % x.k;
  const uint32_t prev = (x.rank + x.k - 1) % x.k;
  auto gbuf = reinterpret_cast<volatile uint64_t*>(arg->glob_addr);
  for (uint32_t p = 0; p < x.R; ++p) {
    const uint32_t s = p & 1, n = p >> 1;
    if (p >= 2)
      vortex::mbarrier(x.lm + OBJ_EMPTY + s).wait_parity((n - 1) & 1);
    if (x.rank == p % x.k)
      idle(arg->delay);
    volatile uint64_t* buf = gbuf + ((size_t)(x.base + next) * 2 + s) * arg->stride;
    for (uint32_t i = x.tid; i < E; i += x.nthreads) {
      for (uint32_t r = 1; r < arg->burst; ++r)
        buf[i] = 0;
      buf[i] = tag(p, i, x.rank);
    }
    vortex::mbarrier full(x.peer(x.lm + OBJ_FULL + s, next));
    uint64_t old = arg->fenced ? full.arrive_release_cluster(1) : full.arrive(1);
    x.check_word(old, n, count, 0, 0, p);
    x.record(mr::OP_ARRIVE, old, 1, next, p);

    // The release's fence emptied the L1: refill it, so that only the
    // acquire's fence keeps the reads after the wait off the old lines.
    volatile uint64_t* own = gbuf + ((size_t)x.b * 2 + s) * arg->stride;
    for (uint32_t i = x.tid; i < E; i += x.nthreads)
      (void)own[i];
    vortex::mbarrier mine(x.lm + OBJ_FULL + s);
    const uint64_t tok = (uint64_t)(n & 1) << 63;
    ++x.waits;
    if (arg->fenced) {
      mine.wait_acquire_cluster(tok);
    } else {
      mine.wait(tok);
    }
    for (uint32_t i = x.tid; i < E; i += x.nthreads)
      x.stale += (own[i] != tag(p, i, prev));
    old = vx_mbar_arrive(x.peer(x.lm + OBJ_EMPTY + s, prev), 1);
    x.check_word(old, n, count, 0, 0, p);
  }
  finish(x, arg);
}

INL uint64_t mbar_op(uint32_t code, uint64_t* obj, uint32_t v) {
  switch (code) {
  case mr::OP_ARRIVE:    return vx_mbar_arrive(obj, v);
  case mr::OP_ARRIVE_TX: return vx_mbar_arrive_tx(obj, v);
  case mr::OP_EXPECT_TX: return vx_mbar_expect_tx(obj, v);
  default:               return vx_mbar_complete_tx(obj, v);
  }
}

// tx_k's per-phase constants.
struct TxPhase {
  uint32_t p, n, s, count, all_bytes, nthreads, nt;
  // T(r): the bytes the peers of rank r complete, every thread but the owner's.
  INL uint32_t total(uint32_t r) const {
    return all_bytes - TX_CHUNK * (nthreads * (1 + r * nthreads) + nthreads * (nthreads - 1) / 2);
  }
};

// Lane l's part of `v` when the nt lanes of a warp split it.
INL uint32_t share(uint32_t v, uint32_t l, uint32_t nt) {
  return (uint32_t)(((uint64_t)v * (l + 1)) / nt - ((uint64_t)v * l) / nt);
}

INL void tx_post(Ctx& x, const TxPhase& t, uint32_t r, uint32_t code, uint64_t old, uint32_t v) {
  const int64_t span = (int64_t)t.total(r) + TX_EXTRA;
  x.check_word(old, t.n, t.count, -span, span, t.p);
  x.record(code, old, v, r, t.p);
}

INL void tx_op(Ctx& x, const TxPhase& t, uint32_t r, uint32_t code, uint32_t v) {
  tx_post(x, t, r, code, mbar_op(code, x.peer(x.lm + OBJ_FULL + t.s, r), v), v);
}

// The ops of this thread on rank r's object in a phase, but for the late ones.
// Out of line: inlined, tx_k passes -vortex-divergence-max-bbs, so no split/join.
static __attribute__((noinline)) void tx_ops(Ctx& x, const TxPhase& t, uint32_t r, uint32_t mine) {
  const bool odd = (t.p & 1) != 0;
  if (odd && x.rank == (r + 1) % x.k && x.warp == 0) {
    tx_op(x, t, r, mr::OP_COMPLETE_TX, mine);
    tx_op(x, t, r, mr::OP_EXPECT_TX, share(TX_EXTRA, x.lane, t.nt));
    tx_op(x, t, r, mr::OP_ARRIVE_TX, share(t.total(r), x.lane, t.nt));
  } else if (odd && x.rank == r && x.warp == 0) {
    tx_op(x, t, r, mr::OP_COMPLETE_TX, share(TX_EXTRA, x.lane, t.nt));
    tx_op(x, t, r, mr::OP_ARRIVE, 1);
  } else if (x.rank == r) {
    tx_op(x, t, r, mr::OP_ARRIVE, 1);
  } else {
    tx_op(x, t, r, mr::OP_COMPLETE_TX, mine);
    tx_op(x, t, r, mr::OP_ARRIVE, 1);
  }
}

// The late expecting ops of warp 0 on rank `last`'s object (see tx_k).
static __attribute__((noinline)) void tx_late(Ctx& x, const TxPhase& t, uint32_t last, uint32_t mine,
                                              bool even, uint32_t delay) {
  const uint32_t p = t.p;
  const uint32_t T = share(t.total(last), x.lane, t.nt);
  uint64_t* obj = x.peer(x.lm + OBJ_FULL + t.s, last);
  if (even) {
    poll_own(x, x.lm + OBJ_FULL + t.s, p, [&](uint64_t w) {
      return mr::phase(w) == (t.n & 1) && mr::pending(w) == t.nt;
    });
    idle(delay);
    tx_post(x, t, last, mr::OP_ARRIVE_TX, vx_mbar_arrive_tx(obj, T), T);
  } else {
    poll_own(x, x.lm + LM_FLAG, p, [&](uint64_t w) { return w > p; });
    idle(delay);
    const uint32_t X = share(TX_EXTRA, x.lane, t.nt);
    const uint64_t o1 = vx_mbar_complete_tx(obj, mine);
    const uint64_t o2 = vx_mbar_expect_tx(obj, X);
    const uint64_t o3 = vx_mbar_arrive_tx(obj, T);
    tx_post(x, t, last, mr::OP_COMPLETE_TX, o1, mine);
    tx_post(x, t, last, mr::OP_EXPECT_TX, o2, X);
    tx_post(x, t, last, mr::OP_ARRIVE_TX, o3, T);
  }
}

// Software tx accounting on every rank's FULL object, one arrival per thread
// and rank. Rank q's thread i completes B(q, i) = TX_CHUNK * (1 + q * N + i)
// bytes on each peer's object (N threads per block), and T(r) is the sum over
// the peers of rank r. The expecting warp's lanes split T(r) and TX_EXTRA.
// Even phases: the owner's warp 0 expects T(r) with arrive_tx, last: it first
// waits until every other op has landed on its object, so the peers'
// completes have driven tx negative. Odd phases: the next rank's warp 0
// expects T(r) remotely (complete_tx of its own bytes, expect_tx of
// TX_EXTRA, arrive_tx of T(r)), while the owner's warp 0 completes TX_EXTRA.
// In odd phase p the expecting ops on rank T = (p / 2) % k come last: once
// every warp of rank T is about to wait, rank T says so with a window store
// to rank T + 1, whose warp 0 waits for it, then `delay` idle iterations
// more, so they sleep, and then completes rank T's phase. The divergent
// branches stay out of line (see tx_ops()).
__kernel void tx_k(rem_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  TxPhase t;
  t.nthreads = x.nthreads;
  t.nt = x.nt;
  t.count = x.k * x.nthreads;
  t.all_bytes = TX_CHUNK * t.count * (t.count + 1) / 2;
  init_objs(x, OBJ_FULL, 2, t.count);
  x.cl.sync();
  const uint32_t mine = TX_CHUNK * (1 + x.rank * x.nthreads + x.tid);
  auto done = reinterpret_cast<uint32_t*>(x.lm + LM_DONE);
  for (uint32_t p = 0; p < x.R; ++p) {
    t.p = p;
    t.n = p >> 1;
    t.s = p & 1;
    const bool odd = (p & 1) != 0;
    const uint32_t late_tgt = t.n % x.k;
    // The rank's target order: its own object last on even phases; on odd
    // ones rank T ends with T + 1 and rank T + 1 with T.
    const uint32_t first = !odd ? x.rank + 1 : (x.rank == late_tgt ? x.rank + 2 : x.rank);
    const uint32_t last = (first + x.k - 1) % x.k;
    for (uint32_t j = 0; j + 1 < x.k; ++j)
      tx_ops(x, t, (first + j) % x.k, mine);
    // The last target's expecting ops come late, once every other op has
    // landed on the owner's object (even phases, the owner's warp 0), or once
    // rank T says its warps are about to wait (odd, rank T + 1's warp 0).
    // They go out back to back, their words checked after.
    const bool late_even = !odd && x.warp == 0;
    const bool late_odd = odd && x.warp == 0 && last == late_tgt && x.rank == (late_tgt + 1) % x.k;
    if (late_even || late_odd) {
      tx_late(x, t, last, mine, late_even, arg->delay);
    } else {
      tx_ops(x, t, last, mine);
    }
    // The warps start waiting together, once every thread of the block has
    // issued its ops, so no wait outlasts the time limit before the late op
    // lands; on odd phases rank T then tells rank T + 1 with a window store
    // that it is about to wait.
    __atomic_fetch_add(done, 1u, __ATOMIC_RELAXED);
    poll_own(x, x.lm + LM_DONE, p, [&](uint64_t w) { return (uint32_t)w >= x.nthreads * (p + 1); });
    if (odd && x.rank == late_tgt && x.warp == 0)
      *x.vpeer(x.lm + LM_FLAG, (x.rank + 1) % x.k) = p + 1;
    wait_own(x, x.lm + OBJ_FULL + t.s, t.n, p);
  }
  finish(x, arg);
}

// Rank 0's thread 0 issues `stores` window stores to rank 1, then an arrive
// on rank 1's COST object, and times both to the arrive's answer: a relaxed
// arrive goes right behind the stores, a release arrive waits until they are
// performed. Each measurement starts with nothing outstanding.
__kernel void cost_k(rem_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t count = 2 * arg->iters;
  init_objs(x, OBJ_COST, 1, count);
  x.cl.sync();
  if (x.rank == 0 && x.tid == 0) {
    uint64_t* obj = x.peer(x.lm + OBJ_COST, 1);
    auto dst = x.vpeer(x.lm + LM_COST, 1);
    for (uint32_t i = 0; i < arg->iters; ++i) {
      for (uint32_t release = 0; release < 2; ++release) {
        vx_fence();
        const uint64_t t0 = vx_rdcycle();
        const uint64_t z = zero_after(t0);
        for (uint32_t j = 0; j < arg->stores; ++j)
          dst[j] = z + tag(i, j, release);
        const uint64_t old = release ? vx_mbar_arrive(obj, 1) : vx_mbar_arrive_relaxed(obj, 1);
        const uint64_t z1 = zero_after(old);
        const uint64_t dt = vx_rdcycle() + z1 - t0;
        x.check_word(old, 0, count, 0, 0, i);
        if (release) {
          x.cyc_rel += dt;
          if (dt < x.min_rel)
            x.min_rel = dt;
        } else {
          x.cyc_rlx += dt;
          if (dt > x.max_rlx)
            x.max_rlx = dt;
        }
      }
    }
  }
  finish(x, arg);
}

// Must stop the simulation: a wait on a peer's object on another core.
__kernel void neg_k(rem_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  init_objs(x, OBJ_FULL, 1, 1);
  x.cl.sync();
  if (x.rank == 0 && x.tid == 0)
    x.check(!vx_mbar_try_wait(x.peer(x.lm + OBJ_FULL, 1), 0), 0, CHK_NEVER);
  finish(x, arg);
}
