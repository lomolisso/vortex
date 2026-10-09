#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Retry bound for each LR/SC increment. High enough that genuine contention
// always resolves, low enough that a store-conditional which never succeeds
// fails the case, with the count short, instead of hanging it.
static const uint32_t LRSC_MAX_RETRIES = 4096;

// All harts contend on a single shared word. Each test exercises a
// different RVA primitive; the host validates the closed-form expected
// final value against the post-run shared word, plus per-hart "observed
// old value" / "iterations completed" counters where applicable.

typedef void (*PFN_Kernel)(kernel_arg_t* __UNIFORM__ arg);

// Globally-unique hart id (mhartid): (cid * NUM_WARPS + wid) * NUM_THREADS
// + tid, matching the LLC reservation table's make_hart_id.
static inline uint32_t hart_id() {
  return (uint32_t)vx_hart_id();
}

// 1) AMOADD hammer.
//   each hart does N amoadd.w +1 → final = num_harts * iters.
void kernel_amoadd(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    __atomic_fetch_add(shared, 1, __ATOMIC_SEQ_CST);
  }
}

// 2) AMOOR hammer.
//   each hart sets its own bit (mod 32). After all harts, every used
//   bit is set; verify final == ((1 << min(num_harts,32)) - 1) for ≤32
//   harts. With more harts, multiple harts share bits; final is just
//   any value with all those bits set (they're idempotent).
void kernel_amoor(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (uint32_t*)arg->shared_addr;
  uint32_t bit = 1u << (hart_id() & 31);
  for (uint32_t i = 0; i < arg->iters; ++i) {
    __atomic_fetch_or(shared, bit, __ATOMIC_SEQ_CST);
  }
}

// 3) AMOAND hammer.
//   start with all bits set; each hart clears its own bit (mod 32).
//   For num_harts ≥ 32, final = 0; for fewer, final = ~mask of cleared bits.
void kernel_amoand(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (uint32_t*)arg->shared_addr;
  uint32_t bit = ~(1u << (hart_id() & 31));
  for (uint32_t i = 0; i < arg->iters; ++i) {
    __atomic_fetch_and(shared, bit, __ATOMIC_SEQ_CST);
  }
}

// 4) AMOXOR hammer.
//   each hart XORs its bit twice (per iter). Net effect per iter = 0,
//   so final = initial value. Tests that XOR commit + return is sane
//   under contention even if final is invariant.
void kernel_amoxor(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (uint32_t*)arg->shared_addr;
  uint32_t bit = 1u << (hart_id() & 31);
  for (uint32_t i = 0; i < arg->iters; ++i) {
    __atomic_fetch_xor(shared, bit, __ATOMIC_SEQ_CST);
    __atomic_fetch_xor(shared, bit, __ATOMIC_SEQ_CST);
  }
}

// 5) AMOMAX hammer (signed).
//   each hart amomax.w with its hart_id. Final = num_harts - 1.
void kernel_amomax(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  int32_t hid = (int32_t)hart_id();
  for (uint32_t i = 0; i < arg->iters; ++i) {
    int32_t expected = __atomic_load_n(shared, __ATOMIC_SEQ_CST);
    int32_t prev;
    asm volatile ("amomax.w %0, %2, (%1)"
                  : "=r"(prev)
                  : "r"(shared), "r"(hid)
                  : "memory");
    (void)expected; (void)prev;
  }
}

// 6) AMOMINU hammer (unsigned).
//   start with shared = 0xFFFFFFFF; each hart amominu.w with its hart_id.
//   Final = 0 (lowest hart_id wins).
void kernel_amominu(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (uint32_t*)arg->shared_addr;
  uint32_t hid = hart_id();
  for (uint32_t i = 0; i < arg->iters; ++i) {
    uint32_t prev;
    asm volatile ("amominu.w %0, %2, (%1)"
                  : "=r"(prev)
                  : "r"(shared), "r"(hid)
                  : "memory");
    (void)prev;
  }
}

// 7) AMOSWAP exchange.
//   each hart swaps in its hart_id repeatedly. The host accepts any
//   final value that is a valid hart_id. Per-hart buffer captures each
//   hart's last observed-old value to verify it's a previously-written
//   hart_id (not garbage).
void kernel_amoswap(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (uint32_t*)arg->shared_addr;
  auto observed = (uint32_t*)arg->per_hart_addr;
  uint32_t hid = hart_id();
  uint32_t last_old = hid;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    last_old = __atomic_exchange_n(shared, hid, __ATOMIC_SEQ_CST);
  }
  observed[hid] = last_old;
}

// 8) LR/SC counter (lock-free CAS-style increment).
//   each hart increments shared by 1 via LR/SC retry. Final =
//   num_harts * iters. Per-hart buffer captures retry count.
void kernel_lrsc_counter(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  auto retries = (uint32_t*)arg->per_hart_addr;
  uint32_t hid = hart_id();
  uint32_t r = 0;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    int32_t old, sc_fail;
    uint32_t guard = 0;
    do {
      asm volatile ("lr.w %0, (%1)"
                    : "=r"(old)
                    : "r"(shared)
                    : "memory");
      int32_t newv = old + 1;
      asm volatile ("sc.w %0, %2, (%1)"
                    : "=r"(sc_fail)
                    : "r"(shared), "r"(newv)
                    : "memory");
      if (sc_fail) ++r;
      ++guard;
    } while (sc_fail && guard < LRSC_MAX_RETRIES);
  }
  retries[hid] = r;
}

// 8) AMOADD.W.AQRL hammer.
//   Same closed-form as kernel_amoadd, but the AMO carries the
//   acquire+release ordering bits. SimX is sequentially consistent,
//   so the result must be identical to the plain variant — this test
//   verifies the .aqrl encoding decodes correctly and that the bank
//   commits the RMW the same way it does for plain AMOADD.
//   Final = num_harts * iters.
void kernel_amoadd_aqrl(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    int32_t one = 1;
    int32_t prev;
    asm volatile ("amoadd.w.aqrl %0, %2, (%1)"
                  : "=r"(prev)
                  : "r"(shared), "r"(one)
                  : "memory");
    (void)prev;
  }
}

// 9) LR.W.AQ / SC.W.RL counter (canonical acquire/release idiom).
//   Same lock-free CAS retry as kernel_lrsc_counter, but with the
//   acquire bit on LR and the release bit on SC. Functionally
//   equivalent under SimX's SC ordering; verifies the .aq/.rl
//   encoding paths through decode and the bank.
//   Final = num_harts * iters.
void kernel_lrsc_counter_aqrl(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    int32_t old, sc_fail;
    uint32_t guard = 0;
    do {
      asm volatile ("lr.w.aq %0, (%1)"
                    : "=r"(old)
                    : "r"(shared)
                    : "memory");
      int32_t newv = old + 1;
      asm volatile ("sc.w.rl %0, %2, (%1)"
                    : "=r"(sc_fail)
                    : "r"(shared), "r"(newv)
                    : "memory");
      ++guard;
    } while (sc_fail && guard < LRSC_MAX_RETRIES);
  }
}

// 10) Atomic reduction.
//   Every hart reads one element from per_hart[] and atomicAdds it
//   into the single global accumulator at shared_addr. Final =
//   sum_{h=0..n-1} per_hart[h]. With per_hart[h] = h+1 (set by host),
//   final = n*(n+1)/2.
void kernel_atomic_reduction(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = (int32_t*)arg->shared_addr;
  auto in     = (int32_t*)arg->per_hart_addr;
  uint32_t hid = hart_id();
  __atomic_fetch_add(shared, in[hid], __ATOMIC_SEQ_CST);
}

// 11) Atomic critical section (spinlock).
//   Uses amoswap.w to atomically swap 1 in and read back the prior value;
//   success when prior was 0. Release via amoswap.w with 0.
//
//   IMPORTANT — SIMD-lockstep gate: this pattern deadlocks within a
//   warp on architectures with strict warp-lockstep execution (Vortex,
//   pre-Volta CUDA): one thread acquires the lock and diverges into
//   the critical section, but the IPDOM stack stalls it until the
//   other threads in the warp leave the else-branch spin loop — which
//   they can't, because that requires the holder to release.
//   The workaround is to serialize at warp granularity: only one thread per warp
//   participates in the lock dance. Inter-warp contention still
//   exercises the AMO bank fully.
//   Final counter = num_active_harts * iters, where num_active_harts =
//   num_cores * num_warps (one thread per warp).
void kernel_atomic_critical(kernel_arg_t* __UNIFORM__ arg) {
  if (vx_thread_id() != 0) return;
  auto counter = (int32_t*)arg->shared_addr;
  auto lock    = (int32_t*)arg->per_hart_addr;  // [0] is the lock
  for (uint32_t i = 0; i < arg->iters; ++i) {
    // Acquire: amoswap.w with 1; succeeds when prior == 0.
    int32_t prev;
    do {
      asm volatile ("amoswap.w %0, %2, (%1)"
                    : "=r"(prev)
                    : "r"(lock), "r"(1)
                    : "memory");
    } while (prev != 0);

    // Critical section: non-atomic increment.
    int32_t v = *counter;
    *counter = v + 1;

    // Release: amoswap.w with 0 (atomicExch in CUDA).
    int32_t zero = 0, unused;
    asm volatile ("amoswap.w %0, %2, (%1)"
                  : "=r"(unused)
                  : "r"(lock), "r"(zero)
                  : "memory");
    (void)unused;
  }
}

// (LR/SC mutex omitted: a spin-lock acquire deadlocks under Vortex's
// SIMD-lockstep semantics. Same issue as kernel_atomic_critical but
// using LR/SC instead of amoswap; the warp-gated workaround used in
// kernel_atomic_critical is also the right way to write it.)

// 12) Issuer self-consistency.
//   Each hart operates on its OWN private word (no cross-hart sharing):
//   a plain load (which caches the line), an atomic +1, then a plain
//   load that MUST observe the increment — RISC-V same-hart program
//   order requires a hart to see its own prior atomic. The observed
//   value is recorded for the host. Fails when a non-LLC cache leaves
//   the issuer's cached copy un-invalidated after the AMO (the post-AMO
//   load then hits the stale line). Final per_hart[hid] == 1.
void kernel_self_consistency(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t hid = hart_id();
  // Each hart owns a private 64B cache line (no false sharing): shared + hid*64.
  volatile uint32_t* p = (volatile uint32_t*)((uint8_t*)arg->shared_addr + (size_t)hid * 64);
  uint32_t* per_hart = (uint32_t*)arg->per_hart_addr;
  (void)*p;  // plain load: bring this hart's line into the L1
  __atomic_fetch_add((uint32_t*)p, 1u, __ATOMIC_SEQ_CST);
  per_hart[hid] = *p;  // plain load after the AMO: must observe the +1
}

// 13) LMEM AMOADD hammer.
//   The same contention as test 0, moved onto local memory so the
//   read-modify-write is resolved by the LMEM banks instead of the dcache.
//   Local memory is per-workgroup and the host cannot read it, so each group's
//   leader publishes its group's final count into its own slot of the per-hart
//   buffer. Every hart contributes `iters` increments to exactly one group, so
//   the published values sum to num_harts * iters however the groups are
//   shaped -- a closed form that does not depend on the launch geometry.
void kernel_amoadd_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem     = (int32_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();

  if (threadIdx.x == 0) {
    *lmem = 0;
  }
  __syncthreads();

  for (uint32_t i = 0; i < arg->iters; ++i) {
    __atomic_fetch_add(lmem, 1, __ATOMIC_SEQ_CST);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    per_hart[hid] = (uint32_t)*lmem;
  }
}

// 14) LMEM LR/SC counter.
//   The lock-free retry of test 7, moved onto local memory so the reservation
//   is resolved by the LMEM banks. Each group's leader publishes its group's
//   final count, so the published values sum to num_harts * iters whatever the
//   launch geometry -- the same closed form test 13 uses.
//
//   The retry is bounded. A store-conditional that never reports success would
//   otherwise spin forever, and a test that hangs reports nothing; with the
//   bound, the published total comes up short and the case fails.
void kernel_lrsc_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem     = (int32_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  // Slot 1 (byte offset 4) sits in the upper lane of a 64-bit hart's lmem
  // bank word, so the contended loop exercises the sub-word LR/SC datapath
  // and the SC-outcome lane; on a 32-bit hart it is an ordinary word.
  auto counter  = &lmem[1];

  if (threadIdx.x == 0) {
    *counter = 0;
  }
  __syncthreads();

  for (uint32_t i = 0; i < arg->iters; ++i) {
    int32_t old, sc_fail;
    uint32_t guard = 0;
    do {
      asm volatile ("lr.w %0, (%1)"
                    : "=r"(old)
                    : "r"(counter)
                    : "memory");
      int32_t newv = old + 1;
      asm volatile ("sc.w %0, %2, (%1)"
                    : "=r"(sc_fail)
                    : "r"(counter), "r"(newv)
                    : "memory");
      ++guard;
    } while (sc_fail && guard < LRSC_MAX_RETRIES);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    per_hart[hid] = (uint32_t)*counter;
  }
}

#if VX_CFG_EXT_ZACAS_ENABLED
// 15) Native compare-and-swap contention.
//   Every hart races to claim each slot of a ladder with its own id, so for
//   slot i exactly one hart sees the empty value and swaps, and the others see
//   that winner's id. Each hart records how many slots it won.
//
//   The point is the per-hart return value, not just the final memory: a
//   compare-and-swap that returned the wrong old word would still leave the
//   ladder looking plausible while every loser believed it had won. The host
//   checks that the wins across harts sum to exactly the number of slots.
void kernel_cas_ladder(kernel_arg_t* __UNIFORM__ arg) {
  auto shared   = (uint32_t*)arg->shared_addr;
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  const uint32_t claim = hid + 1;   // 0 is the unclaimed marker
  uint32_t wins = 0;

  const uint32_t slots = (arg->iters < CAS_LADDER_SLOTS)
                       ? arg->iters : CAS_LADDER_SLOTS;
  for (uint32_t i = 0; i < slots; ++i) {
    uint32_t expected = 0;
    // Returns true only for the hart whose swap actually landed; `expected`
    // is updated with the observed word either way.
    if (__atomic_compare_exchange_n(&shared[i], &expected, claim, 0,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
      ++wins;
    }
  }
  per_hart[hid] = wins;
}

// 16) LMEM compare-and-swap ladder.
//   The cas_ladder race moved onto local memory, so the swap and the old word
//   it returns come from the LMEM banks. Each group races for its own ladder
//   and exactly one of its harts wins each slot, so the win counts sum to
//   groups * slots. The host cannot read local memory or the launch shape, so
//   the group count rides with each leader's publish as a marker well above
//   any possible win count -- the same closed-form idea as tests 13/14.
void kernel_cas_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem     = (uint32_t*)__local_mem();
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  const uint32_t claim = hid + 1;   // 0 is the unclaimed marker

  const uint32_t slots = (arg->iters < CAS_LMEM_SLOTS)
                       ? arg->iters : CAS_LMEM_SLOTS;
  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < slots; ++i) {
      lmem[i] = 0;
    }
  }
  __syncthreads();

  uint32_t wins = 0;
  for (uint32_t i = 0; i < slots; ++i) {
    uint32_t expected = 0;
    if (__atomic_compare_exchange_n(&lmem[i], &expected, claim, 0,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
      ++wins;
    }
  }
  per_hart[hid] = wins + ((threadIdx.x == 0) ? CAS_LMEM_GROUP_MARK : 0);
}
#endif

static inline uint32_t sc_w(volatile uint32_t* addr, uint32_t value) {
  uint32_t sc_fail;
  asm volatile ("sc.w %0, %2, (%1)"
                : "=r"(sc_fail)
                : "r"(addr), "r"(value)
                : "memory");
  return sc_fail;
}

// Every thread seeds and reserves its own word. Threads below
// LRSC_FOREIGN_DIST then store-conditional into the word LRSC_FOREIGN_DIST
// above, whose owner holds the reservation: that SC must fail and leave the
// reservation in place, so the owner's own SC after it succeeds unless another
// thread's LR displaced it from a shared station. A lost reservation only
// fails an SC, which the host reports, so it cannot fake a pass. The seed
// store breaks any reservation an earlier test left on the word, and the fence
// keeps it ahead of the LR. The foreign SC takes its data from the LR, so it
// issues once every thread holds its reservation. Each SC's result store waits
// for the SC's response and the warp issues in order, so the owners' SCs, and
// any access that follows, reach memory after the foreign SCs.
static inline void lrsc_foreign(volatile uint32_t* words, uint32_t stride,
                                uint32_t* result) {
  const uint32_t tid = (uint32_t)vx_thread_id();
  volatile uint32_t* own = &words[tid * stride];
  *own = LRSC_FOREIGN_SEED + tid;
  vx_fence();
  uint32_t old;
  asm volatile ("lr.w %0, (%1)"
                : "=r"(old)
                : "r"(own)
                : "memory");
  if (tid < LRSC_FOREIGN_DIST) {
    result[tid] = sc_w(&words[(tid + LRSC_FOREIGN_DIST) * stride], ~old);
  }
  if (tid >= LRSC_FOREIGN_DIST) {
    result[tid] = sc_w(own, ~old);
  }
}

// 15) Store-conditional into another thread's reservation (17 with Zacas).
//   One warp, each thread's word on a line of its own. One LR and one SC per
//   thread with no retry, so the outcome does not depend on timing.
void kernel_lrsc_foreign(kernel_arg_t* __UNIFORM__ arg) {
  lrsc_foreign((volatile uint32_t*)arg->shared_addr, LLC_LINE_WORDS,
               (uint32_t*)arg->per_hart_addr);
}

// 16) The same on local memory (18 with Zacas). Words sit one register apart,
//   so each fills a bank word of its own and no reservation covers two
//   threads' words. Local memory is not host-visible, so each thread copies
//   its word out.
void kernel_lrsc_foreign_lmem(kernel_arg_t* __UNIFORM__ arg) {
  constexpr uint32_t stride = sizeof(size_t) / sizeof(uint32_t);
  auto lmem  = (volatile uint32_t*)__local_mem();
  auto words = (uint32_t*)arg->shared_addr;
  lrsc_foreign(lmem, stride, (uint32_t*)arg->per_hart_addr);
  // The SC result stores in lrsc_foreign order this read-back after every SC.
  const uint32_t tid = (uint32_t)vx_thread_id();
  words[tid * LLC_LINE_WORDS] = lmem[tid * stride];
}

// Flag polls, over all of a role's waits, before it gives up on the other
// role. A peer that never runs then fails the case instead of hanging it.
static const uint32_t LRSC_PEER_MAX_POLLS = 1024;

static inline uint32_t lr_w(volatile uint32_t* addr) {
  uint32_t value;
  asm volatile ("lr.w %0, (%1)" : "=r"(value) : "r"(addr) : "memory");
  return value;
}

static inline uint32_t amo_swap(volatile uint32_t* addr, uint32_t value) {
  uint32_t old;
  asm volatile ("amoswap.w %0, %2, (%1)" : "=r"(old) : "r"(addr), "r"(value) : "memory");
  return old;
}

// An atomic read at the LLC, past any stale copy in the L1.
static inline uint32_t amo_read(volatile uint32_t* addr) {
  uint32_t value;
  asm volatile ("amoor.w %0, zero, (%1)" : "=r"(value) : "r"(addr) : "memory");
  return value;
}

// `addr`, computed from `dep`, so an access to it issues only once the access
// that produced `dep` has returned from the LLC.
static inline volatile uint32_t* after(volatile uint32_t* addr, uint32_t dep) {
  uint32_t zero;
  asm volatile ("and %0, %1, zero" : "=r"(zero) : "r"(dep));
  return addr + zero;
}

// Returns the flag, or 0 if it was still clear when the role's budget ran out.
static inline uint32_t wait_flag(volatile uint32_t* flag, uint32_t* polls) {
  uint32_t value = amo_read(flag);
  while (value == 0 && *polls < LRSC_PEER_MAX_POLLS) {
    ++*polls;
    value = amo_read(flag);
  }
  return value;
}

// 17) Store-conditional by a peer hart (19 with Zacas).
//   The holder (core 0, warp 0, thread 0) and the peer (thread 0 of the host's
//   peer warp and core) each take every access's address from the previous
//   access's response, and hand over through flags, so the order at the LLC is
//   fixed: holder LR X; peer LR Y, SC X, load X; holder SC X, LR X; peer swap
//   X, LR Y, SC Y; holder SC X. The peer reads X back with a plain load, since
//   an atomic read stores and would break the holder's reservation; its SC
//   dropped X from its caches above the LLC, so the load reads X at the LLC.
//   The seeds are writes, so they also break any reservation left on X or Y.
void kernel_lrsc_peer(kernel_arg_t* __UNIFORM__ arg) {
  if (vx_thread_id() != 0)
    return;
  const uint32_t cid = (uint32_t)vx_core_id();
  const uint32_t wid = (uint32_t)vx_warp_id();
  const bool holder = (cid == 0 && wid == 0);
  const bool peer   = (cid == arg->peer_core && wid == arg->peer_warp);
  if (!holder && !peer)
    return;

  auto base   = (volatile uint32_t*)arg->shared_addr;
  auto x      = base + LRSC_PEER_X * LLC_LINE_WORDS;
  auto y      = base + LRSC_PEER_Y * LLC_LINE_WORDS;
  auto held   = base + LRSC_PEER_HELD * LLC_LINE_WORDS;
  auto tried  = base + LRSC_PEER_TRIED * LLC_LINE_WORDS;
  auto reheld = base + LRSC_PEER_REHELD * LLC_LINE_WORDS;
  auto wrote  = base + LRSC_PEER_WROTE * LLC_LINE_WORDS;
  uint32_t polls = 0;

  if (holder) {
    auto out = base + LRSC_PEER_A_OUT * LLC_LINE_WORDS;
    uint32_t r = amo_swap(y, LRSC_PEER_Y_SEED);
    r = amo_swap(after(x, r), LRSC_PEER_X_SEED);
    const uint32_t xv   = lr_w(after(x, r));
    amo_swap(after(held, xv), 1);
    const uint32_t t    = wait_flag(tried, &polls);
    const uint32_t live = sc_w(after(x, t), LRSC_PEER_A_LIVE_VALUE);
    const uint32_t xv2  = lr_w(after(x, live));
    amo_swap(after(reheld, xv2), 1);
    const uint32_t w    = wait_flag(wrote, &polls);
    const uint32_t sc   = sc_w(after(x, w), LRSC_PEER_A_SC_VALUE);
    out[LRSC_PEER_A_LR]    = xv;
    out[LRSC_PEER_A_LIVE]  = live;
    out[LRSC_PEER_A_LR2]   = xv2;
    out[LRSC_PEER_A_SC]    = sc;
    out[LRSC_PEER_POLLS]   = polls;
    out[LRSC_PEER_TIMEOUT] = (t == 0 || w == 0);
    out[LRSC_PEER_RAN]     = 1;
  } else {
    auto out = base + LRSC_PEER_B_OUT * LLC_LINE_WORDS;
    const uint32_t h    = wait_flag(held, &polls);
    const uint32_t yv   = lr_w(after(y, h));
    const uint32_t sc   = sc_w(after(x, yv), LRSC_PEER_B_SC_VALUE);
    const uint32_t seen = *after(x, sc);
    amo_swap(after(tried, seen), 1);
    const uint32_t rh   = wait_flag(reheld, &polls);
    const uint32_t prev = amo_swap(after(x, rh), LRSC_PEER_B_SWAP_VALUE);
    const uint32_t yv2  = lr_w(after(y, prev));
    const uint32_t ctrl = sc_w(after(y, yv2), yv2 + 1);
    amo_swap(after(wrote, ctrl), 1);
    out[LRSC_PEER_B_SC]    = sc;
    out[LRSC_PEER_B_SEEN]  = seen;
    out[LRSC_PEER_B_PREV]  = prev;
    out[LRSC_PEER_B_CTRL]  = ctrl;
    out[LRSC_PEER_POLLS]   = polls;
    out[LRSC_PEER_TIMEOUT] = (h == 0 || rh == 0);
    out[LRSC_PEER_RAN]     = 1;
  }
}

static inline void amo_add(volatile uint32_t* addr, uint32_t value) {
  uint32_t old;
  asm volatile ("amoadd.w %0, %2, (%1)" : "=r"(old) : "r"(addr), "r"(value) : "memory");
}

// Writes no register, so the warp's next access need not wait for its response.
static inline void amo_add_nr(volatile uint32_t* addr, uint32_t value) {
  asm volatile ("amoadd.w zero, %1, (%0)" : : "r"(addr), "r"(value) : "memory");
}

// `local` on odd lanes and `global` on even ones. The empty asm hides that the
// mask is 0 or all ones, so the choice cannot become a branch that splits the
// warp: the access stays one request with lanes on both sides.
static inline volatile uint32_t* mixed_lane_ptr(volatile uint32_t* global,
                                                volatile uint32_t* local) {
  uintptr_t mask = -(uintptr_t)(vx_thread_id() & 1);
  asm volatile ("" : "+r"(mask));
  return (volatile uint32_t*)((uintptr_t)global
                              ^ (((uintptr_t)global ^ (uintptr_t)local) & mask));
}

// 18) Mixed-lane AMO (20 with Zacas). Each iteration, every lane adds to the
//   hot local word, then one amoadd adds to the hart's global word on even
//   lanes and its local word on odd lanes, so the mixed access meets the local
//   side still busy with the hot adds. Odd lanes then copy their local word into
//   their unused global one: every global word must read iters, and a lane whose
//   add was executed twice reads more. Leaders publish the hot word.
void kernel_amo_mixed_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem     = (volatile uint32_t*)__local_mem();
  auto shared   = (volatile uint32_t*)arg->shared_addr;
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  auto hot   = &lmem[0];
  auto local = &lmem[MIXED_LMEM_OWN + threadIdx.x];

  *local = 0;
  if (threadIdx.x == 0) {
    *hot = 0;
  }
  __syncthreads();

  auto target = mixed_lane_ptr(&shared[hid], local);
  for (uint32_t i = 0; i < arg->iters; ++i) {
    amo_add_nr(hot, 1);
    amo_add(target, 1);
  }
  // The hot adds return nothing, so the warp can reach the barrier with some
  // still queued; an atomic read the lane waits on returns only behind its adds.
  after(hot, amo_read(hot));
  __syncthreads();

  if (vx_thread_id() & 1) {
    shared[hid] = *local;
  }
  if (threadIdx.x == 0) {
    per_hart[hid] = *hot;
  }
}

// 19) Mixed-lane load (21 with Zacas). The same split and hot-word traffic with
//   plain loads: each lane sums its own seeded word, so data from another lane
//   or side, or a response delivered twice, shows in the sum.
void kernel_ld_mixed_lmem(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem     = (volatile uint32_t*)__local_mem();
  auto shared   = (volatile uint32_t*)arg->shared_addr;
  auto per_hart = (uint32_t*)arg->per_hart_addr;
  const uint32_t hid = hart_id();
  auto hot   = &lmem[0];
  auto local = &lmem[MIXED_LMEM_OWN + threadIdx.x];

  *local = MIXED_LOCAL_SEED + hid;
  if (threadIdx.x == 0) {
    *hot = 0;
  }
  __syncthreads();

  auto source = mixed_lane_ptr(&shared[hid], local);
  uint32_t sum = 0;
  for (uint32_t i = 0; i < arg->iters; ++i) {
    amo_add_nr(hot, 1);
    sum += *source;
  }
  per_hart[hid] = sum;
}

static const PFN_Kernel sc_tests[] = {
  kernel_amoadd,             // 0
  kernel_amoor,              // 1
  kernel_amoand,             // 2
  kernel_amoxor,             // 3
  kernel_amomax,             // 4
  kernel_amominu,            // 5
  kernel_amoswap,            // 6
  kernel_lrsc_counter,       // 7
  kernel_amoadd_aqrl,        // 8
  kernel_lrsc_counter_aqrl,  // 9
  kernel_atomic_reduction,   // 10  CUDA-style reduction
  kernel_atomic_critical,    // 11  CUDA-style critical section
  kernel_self_consistency,   // 12  issuer self-consistency (load->AMO->load)
  kernel_amoadd_lmem,        // 13  AMOADD on local memory (LMEM banks)
  kernel_lrsc_lmem,          // 14  LR/SC on local memory (LMEM banks)
#if VX_CFG_EXT_ZACAS_ENABLED
  kernel_cas_ladder,         // 15  native compare-and-swap (Zacas amocas)
  kernel_cas_lmem,           // 16  compare-and-swap on local memory (LMEM banks)
#endif
  kernel_lrsc_foreign,       // 15 (17 with Zacas)  SC into another thread's reservation
  kernel_lrsc_foreign_lmem,  // 16 (18 with Zacas)  the same on local memory
  kernel_lrsc_peer,          // 17 (19 with Zacas)  SC by a peer hart
  kernel_amo_mixed_lmem,     // 18 (20 with Zacas)  one AMO with global and local lanes
  kernel_ld_mixed_lmem,      // 19 (21 with Zacas)  one load with global and local lanes
};

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  sc_tests[arg->testid](arg);
}
