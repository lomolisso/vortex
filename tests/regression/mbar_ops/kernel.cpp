#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dlmem.h>
#include <vx_mbar.h>
#include "common.h"

// Thread 0 of rank 0 runs each script on its own object, one object per
// script; in the remote modes thread 0 of rank 1 runs the peer steps on that
// object through the window, and both blocks meet in a cluster sync after
// every step. Every step records the word its op returned (the pending count
// is read from it on the host) or the wait's result.

namespace {

#define INL inline __attribute__((always_inline))

constexpr uint32_t kHints[NUM_HINTS] = HINT_NS;

INL uint64_t run_op(uint32_t kind, uint64_t* p, uint32_t n, bool relaxed) {
  switch (kind) {
  case K_ARR:     return relaxed ? vx_mbar_arrive_relaxed(p, n) : vx_mbar_arrive(p, n);
  case K_DROP:    return relaxed ? vx_mbar_arrive_drop_relaxed(p, n) : vx_mbar_arrive_drop(p, n);
  case K_NC:      return vx_mbar_arrive_nocomplete(p, n);
  case K_DNC:     return vx_mbar_arrive_drop_nocomplete(p, n);
  case K_DROPTX:  return relaxed ? vx_mbar_arrive_drop_tx_relaxed(p, n) : vx_mbar_arrive_drop_tx(p, n);
  case K_CTX:     return vx_mbar_complete_tx(p, n);
  case K_CLSDROP: vortex::mbarrier(p).arrive_and_drop(n); return 0;
  default:        return ~uint64_t(0);
  }
}

INL bool try_wait(uint64_t* p, uint32_t par, uint32_t mode) {
  if (mode == MODE_RELAXED)
    return vx_mbar_try_wait_parity_relaxed(p, par);
  if (mode == MODE_HINT)
    return vx_mbar_try_wait_parity(p, par, 1000u);
  return vx_mbar_try_wait_parity(p, par);
}

INL uint64_t run_wait(uint32_t kind, uint64_t* p, uint32_t par, const ops_arg_t* arg, uint32_t& spins) {
  if (kind == K_TW)
    return arg->mode == MODE_RELAXED ? vx_mbar_test_wait_parity_relaxed(p, par) : vx_mbar_test_wait_parity(p, par);
  // One exit, so the loop never ends in a lane-divergent branch.
  bool done = false;
  for (uint32_t i = 0; !done && i < arg->spin_limit; ++i) {
    done = try_wait(p, par, arg->mode);
    spins += !done;
  }
  return done ? 1 : 0;
}

// A copy of v the compiler cannot relate to earlier tests of v: a branch on
// it is not threaded back across a barrier, which would duplicate the
// barrier into both sides of an earlier lane-divergent branch.
INL uint32_t opaque(uint32_t v) {
  __asm__ volatile ("" : "+r"(v));
  return v;
}

INL uint64_t cycles() {
  return csr_read(VX_CSR_MCYCLE);
}

// try_wait on a phase that never completes, without a hint and with each
// hint, then a test_wait: the duration of each and its result.
INL void hint_probe(uint64_t* bar, uint64_t* out) {
  vx_mbar_init(bar, 2);
  for (uint32_t rep = 0; rep < HINT_REPS; ++rep) {
    uint64_t* o = out + rep * (NUM_HINTS + 2);
    uint64_t t0 = cycles();
    uint64_t ok = vx_mbar_try_wait_parity(bar, 0) ? 1 : 0;
    o[0] = (cycles() - t0) | (ok << 63);
    for (uint32_t h = 0; h < NUM_HINTS; ++h) {
      t0 = cycles();
      ok = vx_mbar_try_wait_parity(bar, 0, kHints[h]) ? 1 : 0;
      o[1 + h] = (cycles() - t0) | (ok << 63);
    }
    t0 = cycles();
    ok = vx_mbar_test_wait_parity(bar, 0) ? 1 : 0;
    o[NUM_HINTS + 1] = (cycles() - t0) | (ok << 63);
  }
}

INL void negative(const ops_arg_t* arg, uint64_t* lm) {
  vortex::cluster_group cl;
  uint64_t* bar = lm + LM_OBJ;
  const bool samecore = (arg->neg == NEG_SAMECORE);
  if (opaque(threadIdx.x) == 0)
    vx_mbar_init(bar, arg->neg == NEG_DROPZERO ? 2 : 1);
  if (samecore)
    cl.sync();
  else
    __syncthreads();
  if (opaque(threadIdx.x) == 0) {
    uint64_t r = 0;
    switch (arg->neg) {
    case NEG_NOCOMPLETE: r = vx_mbar_arrive_nocomplete(bar, 1); break;
    case NEG_DROPZERO:   r = vx_mbar_arrive_drop(bar, 2); break;
    case NEG_SAMECORE:
      if (cl.block_rank() == 0)
        r = vx_mbar_test_wait_parity(cl.map_shared_rank(bar, 1), 0);
      break;
    default: break;
    }
    reinterpret_cast<volatile uint64_t*>(arg->info_addr)[INFO_CORE0 + (samecore ? cl.block_rank() : 0)] = r | 1;
  }
  if (samecore)
    cl.sync();
}

}  // namespace

__kernel void kernel_main(ops_arg_t* __UNIFORM__ arg) {
  uint64_t* lm = reinterpret_cast<uint64_t*>(__local_mem());
  if (arg->neg) {
    negative(arg, lm);
    return;
  }
  const bool remote = (arg->mode == MODE_REMOTE || arg->mode == MODE_REMOTE2S);
  const bool relaxed = (arg->mode == MODE_RELAXED);
  vortex::cluster_group cl;
  const uint32_t rank = remote ? cl.block_rank() : 0;
  auto steps = reinterpret_cast<const uint32_t*>(arg->steps_addr);
  auto rec = reinterpret_cast<uint64_t*>(arg->rec_addr);
  auto info = reinterpret_cast<volatile uint64_t*>(arg->info_addr);
  uint32_t spins = 0;

  if (opaque(threadIdx.x) == 0 && rank == 0) {
    for (uint32_t i = 0; i < LM_WORDS; ++i)
      vx_mbar_inval(lm + i);
  }
  if (remote)
    cl.sync();

  for (uint32_t s = 0; s < arg->num_scripts; ++s) {
    uint64_t* obj = lm + LM_OBJ + s;
    uint64_t* target = (rank == 1) ? cl.map_shared_rank(obj, 0) : obj;
    for (uint32_t i = 0; i < MAX_STEPS; ++i) {
      const uint32_t st = steps[s * MAX_STEPS + i];
      if (st == ~0u)
        break;
      const uint32_t kind = STEP_KIND(st);
      const uint32_t n = STEP_N(st);
      const uint32_t who = (remote && STEP_SIDE(st) == SIDE_PEER) ? 1 : 0;
      if (opaque(threadIdx.x) == 0 && rank == who) {
        uint64_t v = 0;
        if (kind == K_INIT) {
          vx_mbar_init(obj, n);
          vx_fence_mbarrier_init_release_cluster();
        } else if (kind == K_TW || kind == K_WAITP) {
          v = run_wait(kind, obj, n, arg, spins);
        } else {
          v = run_op(kind, target, n, relaxed);
        }
        // The store waits for the op's returned word, so the op has been
        // performed at the object before the cluster sync.
        rec[s * MAX_STEPS + i] = v;
      }
      if (remote)
        cl.sync();
    }
  }

  // Single-lane code stays straight-line: the copy-out loop runs on every
  // lane, and the hint probe's loops only in the hint mode, which has no
  // barrier after them.
  if (rank == 0) {
    auto words = reinterpret_cast<uint64_t*>(arg->word_addr);
    for (uint32_t s = threadIdx.x; s < arg->num_scripts; s += blockDim.x)
      words[s] = reinterpret_cast<volatile uint64_t*>(lm)[LM_OBJ + s];
  }
  if (opaque(threadIdx.x) == 0) {
    info[INFO_CORE0 + rank] = vx_core_id();
    if (rank == 0) {
      info[INFO_SPINS] = spins;
      info[INFO_DONE] = 1;
    }
  }
  if (remote)
    cl.sync();
  if (arg->mode == MODE_HINT && opaque(threadIdx.x) == 0)
    hint_probe(lm + LM_HINT, reinterpret_cast<uint64_t*>(arg->hint_addr));
}
