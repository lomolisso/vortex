#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"

// One block fills the core. Warps [0, S) stream DXA tile copies into their own
// tiles; DXA beats have priority at every local-memory bank, so the producer's
// store can sit queued behind them. Warp S is the producer, warps (S, N) the
// consumers; together they form a named barrier group. Each round the producer
// stores the round into X, and every consumer lane then loads X: a value older
// than the round is a read that reached the bank before the store did. The
// ordering point is the group barrier (bar), vx_fence() ahead of a flag store
// on another bank that the consumers poll (fence), or vx_fence() and then the
// barrier (fencebar). A second barrier per round keeps the producer from
// overwriting X before every consumer has read it.
//
// A warp issues an instruction only every few cycles, so one warp's stores
// alone never back a bank up. Instead every group lane stores into its own
// word of X's bank at the start of each round, the producer `burst` times and
// each consumer twice that, and the producer's last store carries X in lane 0:
// the bank serializes them all, so X is still on its way when the ordering
// point right behind it arrives.

constexpr uint32_t kDescSrc   = 0;
constexpr uint32_t kGroupBar  = 2;
constexpr uint32_t kCopyBar0  = 3;

static inline __attribute__((always_inline)) void burst_stores(volatile uint64_t* sink, uint64_t v, uint32_t n) {
  uint32_t b = 0;
  for (; b + 4 <= n; b += 4) {
    *sink = v;
    *sink = v;
    *sink = v;
    *sink = v;
  }
  for (; b < n; ++b)
    *sink = v;
}

// The store and the fence in one asm block, so no branch separates them.
static inline __attribute__((always_inline)) void store_fence(volatile uint64_t* p, uint64_t v) {
  __asm__ volatile ("sd %0, 0(%1)\n\tfence iorw, iorw" :: "r"(v), "r"(p) : "memory");
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid   = threadIdx.x;
  const uint32_t warp  = get_sub_group_id();
  const uint32_t nwarp = get_num_sub_groups();
  const uint32_t lane  = tid % vx_num_threads();
  const uint32_t S     = arg->streamers;

  auto tile = reinterpret_cast<uint32_t*>(__local_mem());
  auto ctrl = reinterpret_cast<volatile uint64_t*>(tile + S * arg->tile_elems);
  volatile uint64_t* X    = ctrl + arg->x_word;
  volatile uint64_t* flag = X + 1;
  volatile uint64_t* done = X + 2;

  if (tid == 0) {
    *X = 0;
    *flag = 0;
    *done = 0;
  }
  __syncthreads();

  auto res = reinterpret_cast<uint32_t*>(arg->res_addr);
  if (warp < S) {
    uint32_t copies = 0;
    if (arg->dxa_on) {
      vortex::barrier copy_bar(kCopyBar0 + warp, 1);
      uint32_t* my_tile = tile + warp * arg->tile_elems;
      while (*done == 0) {
        copy_bar.expect_tx(1);
        vx_dxa_issue_1d_wg(kDescSrc, copy_bar.id(), my_tile, (copies % arg->num_src_tiles) * arg->tile_elems);
        copy_bar.arrive_and_wait();
        ++copies;
      }
    }
    if (lane == 0)
      res[blockDim.x * RES_WORDS + warp] = copies;
    return;
  }

  vortex::barrier grp(kGroupBar, nwarp - S);
  const bool producer = (warp == S);
  const bool poll  = (arg->mode == MODE_FENCE);
  const bool fence = (arg->mode == MODE_FENCEBAR);
  const uint32_t burst = arg->burst;
  volatile uint64_t* sink = X + arg->banks * (SINK_ROWS(arg->banks) + lane);
  // Lane 0's last producer store goes to X without a branch, so it stays one warp-wide instruction.
  const uintptr_t to_x = -(uintptr_t)(lane == 0);
  volatile uint64_t* last = reinterpret_cast<volatile uint64_t*>(
      reinterpret_cast<uintptr_t>(sink) ^ ((reinterpret_cast<uintptr_t>(sink) ^ reinterpret_cast<uintptr_t>(X)) & to_x));
  uint32_t stale = 0, first = 0, timeouts = 0;
  for (uint32_t r = 1; r <= arg->rounds; ++r) {
    bool check = !producer;
    if (producer) {
      for (uint32_t d = 0; d < arg->delay; ++d)
        __asm__ volatile ("");
      burst_stores(sink, r, burst - 1);
      // The mode is chosen before X is stored, so X and its ordering point issue back to back.
      if (poll) {
        store_fence(last, r);
        *flag = r;
      } else if (fence) {
        store_fence(last, r);
        grp.arrive_and_wait();
      } else {
        *last = r;
        grp.arrive_and_wait();
      }
    } else {
      // Twice the producer's count, so X lands while the consumers still fill the bank.
      burst_stores(sink, r, 2 * burst);
      if (poll) {
        uint32_t spins = 0;
        uint64_t f;
        while ((f = *flag) < r && ++spins < SPIN_LIMIT) {}
        if (f < r) {
          ++timeouts;
          check = false;
        }
      } else {
        grp.arrive_and_wait();
      }
    }
    if (check && *X != r) {
      ++stale;
      if (first == 0)
        first = r;
    }
    grp.arrive_and_wait();
  }
  // The streamers stop only after the last round, so the banks stay busy throughout.
  if (producer && lane == 0)
    *done = 1;
  if (!producer) {
    res[tid * RES_WORDS + RES_FIRST]   = first;
    res[tid * RES_WORDS + RES_TIMEOUT] = timeouts;
    res[tid * RES_WORDS + RES_STALE]   = stale;
  }
}
