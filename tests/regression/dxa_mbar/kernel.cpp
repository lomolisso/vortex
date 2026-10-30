#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include <vx_mbar.h>
#include "common.h"

// One block fills the core. Per round, warp 0 copies source tiles into local
// memory with the DXA, one or several tiles side by side; warps 0 and 1 wait
// for them, then load the last tile's last word, which must already hold the
// copied value, and check every tile. The other warps hammer one bank with
// loads, stores and atomics meanwhile: the last word's, which holds the
// copy's write to that bank back, or the mbarrier's, which holds the
// complete_tx requests back.

namespace {

#define INL inline __attribute__((always_inline))

constexpr uint32_t kDesc     = 0;
constexpr uint32_t kMeetBar  = 1;   // the waiter warps meet before each copy
constexpr uint32_t kCopyBar  = 2;   // -l: the copy's barrier
constexpr uint64_t kLimit    = VX_CFG_MBAR_SUSPEND_CYCLES;

// Zero, computed from t: a try_wait whose token includes it issues only once
// the cycle counter read t has completed.
INL uint64_t zero_after(uint64_t t) {
  uint64_t z;
  __asm__ volatile ("and %0, %1, zero" : "=r"(z) : "r"(t));
  return z;
}

// The bank's own answer to a try_wait: false when it read the phase
// incomplete, and then the warp has slept.
INL bool bank_try_wait(uint64_t* obj, uint64_t tok) {
  uint64_t r;
  __asm__ volatile (".insn r %1, 5, 0, %0, %2, %3"
                    : "=r"(r) : "i"(RISCV_CUSTOM2), "r"(obj), "r"(tok) : "memory");
  return r != 0;
}

INL void idle_cycles(uint64_t n) {
  const uint64_t t0 = vx_rdcycle();
  while (vx_rdcycle() - t0 < n) {}
}

struct Out {
  uint32_t errs, first, spins, wakes, timeouts, iters;

  INL void fail(uint32_t round, uint32_t code) {
    if (errs++ == 0)
      first = (code << 16) | round;
  }
};

// A thread's wait for phase `ph` with try_wait, which sleeps while the phase
// is incomplete. A sleep that ends before the time limit with the phase
// complete was ended by the completion. Whether the try_wait slept depends on
// timing, so both outcomes run the same instructions: the test_wait always
// runs, and the wake is counted without a branch.
INL void wait_phase(uint64_t* mbar, uint32_t ph, uint32_t round, Out& o) {
  for (uint32_t n = 1; ; ++n) {
    const uint64_t t0  = vx_rdcycle();
    const uint64_t tok = ((uint64_t)(ph & 1) << 63) | zero_after(t0);
    const bool raw    = bank_try_wait(mbar, tok);
    const bool tested = vx_mbar_test_wait(mbar, tok);
    const uint64_t dt = vx_rdcycle() - t0;
    if (raw | tested) {
      o.wakes += (uint32_t)(!raw) & (uint32_t)(dt < kLimit);
      return;
    }
    ++o.timeouts;
    if (n >= TRY_LIMIT) {
      o.fail(round, CHK_WAIT);
      return;
    }
  }
}

INL void write_out(const kernel_arg_t* arg, uint32_t tid, const Out& o) {
  auto out = reinterpret_cast<volatile uint32_t*>(arg->out_addr) + tid * OUT_WORDS;
  out[OUT_ERRS]     = o.errs;
  out[OUT_FIRST]    = o.first;
  out[OUT_SPINS]    = o.spins;
  out[OUT_WAKES]    = o.wakes;
  out[OUT_TIMEOUTS] = o.timeouts;
  out[OUT_ITERS]    = o.iters;
  out[OUT_DONE]     = 1;
}

// Contention thread k: a load and a store of its own word in the probe's
// bank, and an atomic add on a word all of them share in that bank.
INL void contend(const kernel_arg_t* arg, uint8_t* lm, uint32_t k, uint32_t num, Out& o) {
  constexpr uint32_t nb = VX_CFG_LMEM_NUM_BANKS;
  auto scr  = reinterpret_cast<uint64_t*>(lm + arg->scr_off);
  auto mine = reinterpret_cast<volatile uint64_t*>(scr + k * nb + arg->probe_bank);
  auto ctr  = reinterpret_cast<uint32_t*>(scr + num * nb + arg->probe_bank);
  auto done = reinterpret_cast<volatile uint64_t*>(lm + LM_DONE);
  while (arg->iters ? o.iters < arg->iters : *done == 0) {
    *mine = *mine + 1;
    __atomic_fetch_add(ctr, 1u, __ATOMIC_RELAXED);
    ++o.iters;
  }
}

// A waiter warp's half of the QUEUE_COPIES copies, issued back to back:
// their operands are gathered first, so the issues of both warps follow each
// other in a few cycles and the copies land faster than the contended bank
// takes their complete_tx requests.
INL void issue_burst(uint64_t* mbar, volatile uint32_t* tile, uint32_t E, uint32_t r, uint32_t num_src,
                     uint32_t warp) {
  constexpr uint32_t kHalf = QUEUE_COPIES / NUM_WAITERS;
  const uint32_t meta = vx_dxa_pack_meta_mbar(kDesc, mbar);
  uint32_t a[kHalf];
#pragma GCC unroll 64
  for (uint32_t k = 0; k < kHalf; ++k) {
    const uint32_t q = warp * kHalf + k;
    const uint32_t coord = ((r * QUEUE_COPIES + q) % num_src) * E;
    a[k] = (uint32_t)vx_wgather((size_t)(uintptr_t)(tile + q * E), (size_t)meta, (size_t)coord, (size_t)0u);
  }
#pragma GCC unroll 64
  for (uint32_t k = 0; k < kHalf; ++k) {
    __asm__ volatile (".insn r %0, 0, %1, x0, %2, x0"
                      :: "i"(VX_DXA_EXT_OPCODE), "i"(VX_DXA_FUNCT7), "r"(a[k]) : "memory");
  }
}

} // namespace

__kernel void probe_k(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t nt   = vx_num_threads();
  const uint32_t lane = tid % nt;
  const uint32_t warp = get_sub_group_id();
  const uint32_t nwt  = NUM_WAITERS * nt;
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);
  auto tile = reinterpret_cast<volatile uint32_t*>(lm + LM_TILE);
  const uint32_t E = arg->tile_elems;
  Out o = {};

  if (tid == 0) {
    vx_mbar_init(mbar, 1);
    *reinterpret_cast<volatile uint64_t*>(lm + LM_DONE) = 0;
  }
  __syncthreads();

  if (warp >= NUM_WAITERS) {
    contend(arg, lm, tid - nwt, blockDim.x - nwt, o);
  } else {
    vortex::barrier meet(kMeetBar, NUM_WAITERS);
    vortex::barrier copy(kCopyBar, NUM_WAITERS);
    const uint32_t Q = arg->copies;
    for (uint32_t r = 0; r < arg->rounds; ++r) {
      meet.arrive_and_wait();
      if (warp == 0) {
        if (arg->legacy) {
          copy.expect_tx(1);
          vx_dxa_issue_1d_wg(kDesc, copy.id(), (void*)tile, (r % arg->num_src_tiles) * E);
        } else {
          if (lane == 0)
            vx_mbar_arrive_tx(mbar, arg->tx_bytes);
          if (Q == 1)
            vx_dxa_issue_1d_mbar(kDesc, mbar, (void*)tile, (r % arg->num_src_tiles) * E);
        }
      }
      if (Q != 1)
        issue_burst(mbar, tile, E, r, arg->num_src_tiles, warp);
      if (arg->legacy) {
        copy.arrive_and_wait();
      } else {
        wait_phase(mbar, r, r, o);
      }

      // The last copy's last word must be visible as soon as the wait returns.
      const uint32_t last = ((r * Q + Q - 1) % arg->num_src_tiles) * E;
      const uint32_t want = src_value(last + E - 1);
      uint32_t spins = 0;
      while (tile[Q * E - 1] != want) {
        if (++spins >= SPIN_BOUND) {
          o.fail(r, CHK_PROBE);
          break;
        }
      }
      o.spins += spins;

      for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t first = ((r * Q + q) % arg->num_src_tiles) * E;
        for (uint32_t j = tid; j < E; j += nwt) {
          if (tile[q * E + j] != src_value(first + j))
            o.fail(r, CHK_DATA);
        }
      }
    }
    meet.arrive_and_wait();
    if (tid == 0)
      *reinterpret_cast<volatile uint64_t*>(lm + LM_DONE) = 1;
  }
  __syncthreads();
  if (tid == 0)
    *reinterpret_cast<uint64_t*>(arg->word_addr) = *reinterpret_cast<volatile uint64_t*>(mbar);
  write_out(arg, tid, o);
}

// Runs the simulator must stop: an mbarrier-mode copy whose block exits
// before it lands (a second block then holds the slot while it does), whose
// mbarrier lies past the block's local memory, or that has a multicast mask.
__kernel void neg_k(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t lane = tid % vx_num_threads();
  const uint32_t warp = get_sub_group_id();
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);
  auto tile = reinterpret_cast<volatile uint32_t*>(lm + LM_TILE);
  const uint32_t E = arg->tile_elems;

  if (arg->neg == NEG_RETIRED && blockIdx.x != 0) {
    idle_cycles(arg->linger);
    return;
  }
  if (tid == 0)
    vx_mbar_init(mbar, 1);
  __syncthreads();
  if (warp == 0) {
    if (lane == 0)
      vx_mbar_arrive_tx(mbar, arg->tx_bytes);
    if (arg->neg == NEG_BOUNDS) {
      auto outside = reinterpret_cast<uint64_t*>(lm + arg->lmem_stride);
      vx_dxa_issue_1d_mbar(kDesc, outside, (void*)tile, 0);
    } else if (arg->neg == NEG_MCAST) {
      vx_dxa_issue_1d_multicast_wg(kDesc, vx_dxa_pack_meta_mbar(kDesc, mbar) >> 4, (void*)tile, 0, 0x3);
    } else {
      vx_dxa_issue_1d_mbar(kDesc, mbar, (void*)tile, 0);
    }
  }
  if (arg->neg == NEG_RETIRED)
    return;
  const uint32_t want = src_value(E - 1);
  for (uint32_t n = 0; tile[E - 1] != want && n < SPIN_BOUND * 100; ++n) {}
  idle_cycles(arg->linger);
}
