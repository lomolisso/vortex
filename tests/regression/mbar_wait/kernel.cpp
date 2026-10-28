#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include <vx_mbar.h>
#include "common.h"

// One block. Warps [0, S) stream DXA tile copies into their own tiles to keep
// the banks busy; the other warps, the workers, run the mode. Worker warp 0
// arrives (lane 0, one arrival per phase) and the others wait. At the end
// every warp meets in __syncthreads() and thread 0 copies the objects out for
// the host, which predicts each final word with mbar_ref.

namespace {

#define INL inline __attribute__((always_inline))

constexpr uint32_t kWorkerBar = 1;
constexpr uint32_t kCopyBar0  = 2;
constexpr uint64_t kLimit     = VX_CFG_MBAR_SUSPEND_CYCLES;
// Longer than a try_wait that does not suspend takes, even behind every
// lane's poll at its bank (a bank op every 3 cycles).
constexpr uint64_t kSlept     = kLimit / 4;

struct Ctx {
  uint32_t tid, lane, nt, warp, S;
  bool     worker, spin, parity;
  uint32_t ww, nww, wt, nwt;   // worker warp and thread indexes, and their counts
  uint64_t* lm;
  volatile uint64_t* data;
  uint32_t errors, first, what, calls, timeouts, early, slept, mixed;
  uint64_t minfalse, elapsed, lagmin, lagmax, met;

  INL void fail(uint32_t round, uint32_t code) {
    if (errors++ == 0) {
      first = round + 1;
      what  = code;
    }
  }
  INL void check(bool ok, uint32_t round, uint32_t code) {
    if (!ok)
      fail(round, code);
  }
  INL void meet() {
    vortex::barrier(kWorkerBar, nww).arrive_and_wait();
    met = vx_rdcycle();
  }
};

INL void idle(uint32_t n) {
  for (uint32_t i = 0; i < n; ++i)
    __asm__ volatile ("");
}

// n < MIXED_NOPS nops, as a jump into the tail of a sled: skews finer than an idle
// iteration, whose branch costs as much as many nops.
INL void nops(uint32_t n) {
  __asm__ volatile (".option push\n"
                    ".option norvc\n"
                    "la t0, 1f\n"
                    "slli t1, %0, 2\n"
                    "sub t0, t0, t1\n"
                    "jr t0\n"
                    ".rept %1\n"
                    "nop\n"
                    ".endr\n"
                    "1:\n"
                    ".option pop\n"
                    :: "r"(n), "i"(MIXED_NOPS) : "t0", "t1");
}

// A value no other round or object stores.
INL uint64_t tag(uint32_t round, uint32_t idx) {
  return ((uint64_t)(round + 1) << 32) | idx;
}

INL void ctx_init(Ctx& x, const mbar_arg_t* arg) {
  x.tid    = threadIdx.x;
  x.nt     = vx_num_threads();
  x.lane   = x.tid % x.nt;
  x.warp   = get_sub_group_id();
  x.S      = arg->streamers;
  x.worker = (x.warp >= x.S);
  x.spin   = (arg->spin != 0);
  x.parity = (arg->parity != 0);
  x.ww     = x.warp - x.S;
  x.nww    = get_num_sub_groups() - x.S;
  x.wt     = x.tid - x.S * x.nt;
  x.nwt    = x.nww * x.nt;
  x.lm     = reinterpret_cast<uint64_t*>(__local_mem());
  x.data   = x.lm + LM_DATA;
  x.errors = x.first = x.what = x.calls = x.timeouts = x.early = x.slept = x.mixed = 0;
  x.minfalse = x.lagmin = ~uint64_t(0);
  x.elapsed  = x.lagmax = x.met = 0;
  if (x.tid == 0) {
    for (uint32_t i = 0; i < NUM_OBJS; ++i)
      vx_mbar_init(x.lm + i, 1);
    reinterpret_cast<volatile uint64_t*>(x.lm)[LM_DONE] = 0;
  }
}

INL void stream(const Ctx& x, const mbar_arg_t* arg) {
  vortex::barrier copy_bar(kCopyBar0 + x.warp, 1);
  uint32_t* tile = reinterpret_cast<uint32_t*>(x.lm + arg->tile_word) + x.warp * arg->tile_elems;
  volatile uint64_t* done = x.lm + LM_DONE;
  uint32_t copies = 0;
  while (*done == 0) {
    copy_bar.expect_tx(1);
    vx_dxa_issue_1d_wg(0, copy_bar.id(), tile, (copies % arg->num_src_tiles) * arg->tile_elems);
    copy_bar.arrive_and_wait();
    ++copies;
  }
}

// Zero, computed from t: a try_wait whose token includes it issues only once
// the cycle counter read t has completed. A CSR read can complete after a
// later instruction has issued, so without it the start time can be late
// and a time-out look shorter than the limit.
INL uint64_t zero_after(uint64_t t) {
  uint64_t z;
  __asm__ volatile ("and %0, %1, zero" : "=r"(z) : "r"(t));
  return z;
}

INL uint64_t parity_token(uint32_t ph) {
  return (uint64_t)(ph & 1) << 63;
}

// The bank's own answer to a try_wait, which vx_mbar_try_wait() follows with
// one test_wait when it is false: false in the lanes that read the phase
// incomplete, and then the warp has slept.
INL bool bank_try_wait(uint64_t* obj, uint64_t tok) {
  uint64_t r;
  __asm__ volatile (".insn r %1, 5, 0, %0, %2, %3"
                    : "=r"(r) : "i"(RISCV_CUSTOM2), "r"(obj), "r"(tok) : "memory");
  return r != 0;
}

// One try_wait, timed. A false is a time-out: when every lane of the warp
// waits on this object it must come after the time limit; lanes on several
// objects wake with the last waiting lane's object and may give up early.
// `lead` test_waits of the object, from every lane but lane 0, go to its bank
// right ahead of the try_wait. With `raw`, the poll does what
// vx_mbar_try_wait() does with the bank's answer in sight, which it returns.
INL bool poll(Ctx& x, uint64_t* obj, uint32_t ph, uint32_t round, bool one_obj,
              uint32_t lead = 0, bool* raw = nullptr) {
  const uint64_t t0 = vx_rdcycle();
  const uint64_t tok = parity_token(ph) | zero_after(t0);
  if (x.lane != 0) {
    for (uint32_t i = 0; i < lead; ++i)
      (void)vx_mbar_test_wait(obj, tok);
  }
  bool done;
  if (raw) {
    *raw = bank_try_wait(obj, tok);
    done = *raw || vx_mbar_test_wait(obj, tok);
  } else {
    done = vx_mbar_try_wait(obj, tok);
  }
  const uint64_t dt = vx_rdcycle() - t0;
  ++x.calls;
  if (!done) {
    ++x.slept;
    if (dt >= kLimit)
      ++x.timeouts;
    else
      ++x.early;
    if (one_obj)
      x.check(dt >= kLimit, round, CHK_SPURIOUS);
    if (dt < x.minfalse)
      x.minfalse = dt;
  } else if (raw ? !*raw : dt >= kSlept) {
    ++x.slept;
  }
  return done;
}

INL void wait_parity(Ctx& x, uint64_t* obj, uint32_t ph, uint32_t round, bool one_obj) {
  if (x.spin) {
    uint32_t n = 1;
    while (!vx_mbar_test_wait_parity(obj, ph)) {
      if (++n > SPIN_LIMIT) {
        x.fail(round, CHK_WAIT);
        break;
      }
    }
    x.calls += n;
    return;
  }
  for (uint32_t n = 1; !poll(x, obj, ph, round, one_obj); ++n) {
    if (n >= TRY_LIMIT) {
      x.fail(round, CHK_WAIT);
      break;
    }
  }
}

// The arriver's lane 0 stores the object's tag and arrives, and notes how
// long after the workers' meeting it issued the arrive (the phase cannot flip
// sooner) and got its answer, which leaves the bank as the phase flips
// (parity runs keep their fixed schedule).
INL void arrive(Ctx& x, uint32_t obj, uint32_t round, uint32_t delay) {
  if (x.lane == 0) {
    idle(delay);
    x.data[obj] = tag(round, obj);
    if (!x.parity) {
      const uint64_t lag = vx_rdcycle() - x.met;
      if (lag < x.lagmin)
        x.lagmin = lag;
    }
    const uint64_t old = vortex::mbarrier(x.lm + obj).arrive(1);
    if (!x.parity) {
      const uint64_t z = zero_after(old);
      const uint64_t lag = vx_rdcycle() + z - x.met;
      if (lag > x.lagmax)
        x.lagmax = lag;
    }
  }
}

// The workers meet, then release the streamers; everyone meets, then thread 0
// copies the objects out.
INL void finish(Ctx& x, const mbar_arg_t* arg) {
  if (x.worker) {
    x.meet();
    if (x.wt == 0)
      reinterpret_cast<volatile uint64_t*>(x.lm)[LM_DONE] = 1;
  }
  __syncthreads();
  if (x.tid == 0) {
    auto words = reinterpret_cast<uint64_t*>(arg->word_addr);
    for (uint32_t i = 0; i < NUM_OBJS; ++i)
      words[i] = reinterpret_cast<volatile uint64_t*>(x.lm)[i];
  }
  auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + x.tid * ERR_WORDS;
  err[ERR_COUNT]    = x.errors;
  err[ERR_FIRST]    = x.first;
  err[ERR_WHAT]     = x.what;
  err[ERR_CALLS]    = x.calls;
  err[ERR_TIMEOUTS] = x.timeouts;
  err[ERR_EARLY]    = x.early;
  err[ERR_MINFALSE] = (x.minfalse > 0xFFFFFFFFu) ? 0xFFFFFFFFu : (uint32_t)x.minfalse;
  err[ERR_MIXED]    = x.mixed;
  err[ERR_ELAPSED]  = (uint32_t)x.elapsed;
  err[ERR_LAGMIN]   = (x.lagmin > 0xFFFFFFFFu) ? 0xFFFFFFFFu : (uint32_t)x.lagmin;
  err[ERR_LAGMAX]   = (uint32_t)x.lagmax;
  err[ERR_SLEPT]    = x.slept;
}

} // namespace

// Per phase the workers meet, then the arriver arrives after `delay` idle
// iterations while the other worker warps wait on the phase. Parity runs
// have a fixed schedule: the delay is long enough for every waiter to be
// asleep when the arrive lands and short enough to beat the time limit, so a
// wait is exactly one try_wait, which the arrive wakes and which returns
// true; at the end every worker warp times out once on an object nobody
// arrives on.
__kernel void late_k(mbar_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  __syncthreads();
  if (!x.worker) {
    stream(x, arg);
  } else {
    vortex::mbarrier bar(x.lm + OBJ_BAR);
    const uint64_t t0 = vx_rdcycle();
    for (uint32_t r = 0; r < arg->rounds; ++r) {
      x.meet();
      if (x.ww == 0) {
        arrive(x, OBJ_BAR, r, arg->delay);
        bar.wait_parity(r & 1);
      } else if (x.parity) {
        uint32_t n = 0;
        bool done;
        do {
          done = bar.try_wait_parity(r & 1);
        } while (!done && ++n < 4);
        x.check(n == 0, r, CHK_CALLS);
        x.calls += n + 1;
      } else {
        wait_parity(x, x.lm + OBJ_BAR, r & 1, r, true);
      }
      x.check(x.data[OBJ_BAR] == tag(r, OBJ_BAR), r, CHK_DATA);
    }
    x.elapsed = vx_rdcycle() - t0;
    if (x.parity) {
      x.check(!vortex::mbarrier(x.lm + OBJ_NEVER).try_wait_parity(0), arg->rounds, CHK_NEVER);
      ++x.calls;
    }
  }
  finish(x, arg);
}

// Worker warp 0 is a producer held up for `rounds` time limits: it sleeps on
// an object nobody arrives on, so every warp of the core is asleep for longer
// than the scheduler's stall timeout, and only the time limit wakes anyone.
// Each of its polls must fail, and only after the limit. Then it arrives on
// the phase the other worker warps wait on.
__kernel void sleep_k(mbar_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  __syncthreads();
  vortex::mbarrier bar(x.lm + OBJ_BAR);
  const uint64_t t0 = vx_rdcycle();
  if (x.ww == 0) {
    for (uint32_t i = 0; i < arg->rounds; ++i) {
      const uint64_t s = vx_rdcycle();
      const bool done = vx_mbar_try_wait(x.lm + OBJ_NEVER, zero_after(s));
      const uint64_t dt = vx_rdcycle() - s;
      ++x.calls;
      x.check(!done && dt >= kLimit, i, CHK_NEVER);
      ++x.timeouts;
      if (dt < x.minfalse)
        x.minfalse = dt;
    }
    arrive(x, OBJ_BAR, 0, 0);
    bar.wait_parity(0);
  } else {
    wait_parity(x, x.lm + OBJ_BAR, 0, 0, true);
  }
  x.check(x.data[OBJ_BAR] == tag(0, OBJ_BAR), 0, CHK_DATA);
  x.elapsed = vx_rdcycle() - t0;
  finish(x, arg);
}

// Every lane of every worker warp waits, the arriver's included. One object:
// all of them on it. Several: by warp (warp w on object w % objs), or by lane
// (worker thread t on object t % objs), where a warp's lanes wait on different
// objects and the warp is woken by its last waiting lane's object or the time
// limit. The arriver
// arrives on the objects in turn, `delay` idle iterations apart.
__kernel void many_k(mbar_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  __syncthreads();
  if (!x.worker) {
    stream(x, arg);
  } else {
    const uint32_t n = arg->objs;
    const uint32_t i = OBJ_MANY + (arg->lanes ? x.wt : x.ww) % n;
    const bool one_obj = !arg->lanes || n == 1;
    const uint64_t t0 = vx_rdcycle();
    for (uint32_t r = 0; r < arg->rounds; ++r) {
      x.meet();
      if (x.ww == 0) {
        for (uint32_t j = 0; j < n; ++j)
          arrive(x, OBJ_MANY + j, r, arg->delay);
      }
      wait_parity(x, x.lm + i, r & 1, r, one_obj);
      x.check(x.data[i] == tag(r, i), r, CHK_DATA);
    }
    x.elapsed = vx_rdcycle() - t0;
  }
  finish(x, arg);
}

// The arrive lands while a waiter warp's lanes read the object one after
// another: the lanes before it read the phase incomplete, the lanes after it
// complete. A warp's lanes reach the bank one after another: in several
// packets when the warp has more threads than LSU lanes, and each lane's
// requests in their own port, which the bank serves round robin. Each waiter
// warp sends MIXED_LEAD test_waits from every lane but lane 0 right ahead of
// its first poll, so the other lanes' requests queue behind them while lane
// 0's goes ahead. That poll reads the bank's answer per lane (bank_try_wait)
// and is delayed by a number of nops that tracks the arrive: one more when
// the whole warp read the phase incomplete, one less when it read it
// complete, plus an offset that sweeps eight nops around that point. Lane 0
// counts the phases whose first poll split the warp. The arrive wakes the
// warp, so the poll returns true in every lane.
__kernel void mixed_k(mbar_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  __syncthreads();
  if (!x.worker) {
    stream(x, arg);
  } else {
    vortex::mbarrier bar(x.lm + OBJ_BAR);
    int32_t center = 0;
    const uint64_t t0 = vx_rdcycle();
    for (uint32_t r = 0; r < arg->rounds; ++r) {
      x.meet();
      if (x.ww == 0) {
        arrive(x, OBJ_BAR, r, arg->delay);
        bar.wait_parity(r & 1);
      } else {
        int32_t skew = center + (int32_t)(r % 8) - 4;
        skew = skew < 0 ? 0 : (skew >= MIXED_NOPS ? MIXED_NOPS - 1 : skew);
        nops((uint32_t)skew);
        bool raw;
        bool done = poll(x, x.lm + OBJ_BAR, r & 1, r, true, MIXED_LEAD, &raw);
        const bool all_raw = vx_vote_all(raw);
        const bool none_raw = vx_vote_all(!raw);
        if (!all_raw && !none_raw && x.lane == 0)
          ++x.mixed;
        center += none_raw ? 1 : (all_raw ? -1 : 0);
        for (uint32_t n = 1; !done; ++n) {
          if (n >= TRY_LIMIT) {
            x.fail(r, CHK_WAIT);
            break;
          }
          done = poll(x, x.lm + OBJ_BAR, r & 1, r, true);
        }
      }
      x.check(x.data[OBJ_BAR] == tag(r, OBJ_BAR), r, CHK_DATA);
    }
    x.elapsed = vx_rdcycle() - t0;
  }
  finish(x, arg);
}
