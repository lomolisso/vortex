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

// The producer's X access as a store, an amoswap.d or a load, each followed
// in the same asm block by its ordering point when that is an instruction.
static inline __attribute__((always_inline)) void amo_x(volatile uint64_t* p, uint64_t v) {
  __asm__ volatile ("amoswap.d zero, %0, (%1)" :: "r"(v), "r"(p) : "memory");
}
static inline __attribute__((always_inline)) uint64_t load_x(volatile uint64_t* p) {
  uint64_t v;
  __asm__ volatile ("ld %0, 0(%1)" : "=r"(v) : "r"(p) : "memory");
  return v;
}
static inline __attribute__((always_inline)) void amo_fence(volatile uint64_t* p, uint64_t v) {
  __asm__ volatile ("amoswap.d zero, %0, (%1)\n\tfence iorw, iorw" :: "r"(v), "r"(p) : "memory");
}
static inline __attribute__((always_inline)) uint64_t load_fence(volatile uint64_t* p) {
  uint64_t v;
  __asm__ volatile ("ld %0, 0(%1)\n\tfence iorw, iorw" : "=&r"(v) : "r"(p) : "memory");
  return v;
}
static inline __attribute__((always_inline)) void store_rl(volatile uint64_t* p, volatile uint64_t* f, uint64_t v) {
  __asm__ volatile ("sd %0, 0(%1)\n\tamoswap.d.rl zero, %0, (%2)" :: "r"(v), "r"(p), "r"(f) : "memory");
}
static inline __attribute__((always_inline)) void amo_rl(volatile uint64_t* p, volatile uint64_t* f, uint64_t v) {
  __asm__ volatile ("amoswap.d zero, %0, (%1)\n\tamoswap.d.rl zero, %0, (%2)" :: "r"(v), "r"(p), "r"(f) : "memory");
}
static inline __attribute__((always_inline)) uint64_t load_rl(volatile uint64_t* p, volatile uint64_t* f, uint64_t r) {
  uint64_t v;
  __asm__ volatile ("ld %0, 0(%1)\n\tamoswap.d.rl zero, %3, (%2)" : "=&r"(v) : "r"(p), "r"(f), "r"(r) : "memory");
  return v;
}

// kernel_main with two more axes, in a kernel of its own so that kernel_main's
// code stays as it is: the producer's X access (ACC_*) and the rl release,
// where the producer swaps the flag with amoswap.d.rl and the consumers poll it.
// In ld mode the producer loads X before its ordering point and lane 0 checks
// it still holds the previous round, while every consumer warp's lane 0 stores
// the round into X once released: a load the release did not wait for could
// reach the bank after that store.
__kernel void order_k(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid   = threadIdx.x;
  const uint32_t warp  = get_sub_group_id();
  const uint32_t nwarp = get_num_sub_groups();
  const uint32_t lane  = tid % vx_num_threads();
  const uint32_t S     = arg->streamers;
  const uint32_t rel   = arg->mode & ((1u << ACC_SHIFT) - 1);
  const uint32_t acc   = arg->mode >> ACC_SHIFT;

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
  const bool poll = (rel == MODE_FENCE || rel == MODE_RL);
  const bool ld   = (acc == ACC_LD);
  const uint32_t burst = arg->burst;
  volatile uint64_t* sink = X + arg->banks * (SINK_ROWS(arg->banks) + lane);
  const uintptr_t to_x = -(uintptr_t)(lane == 0);
  volatile uint64_t* last = reinterpret_cast<volatile uint64_t*>(
      reinterpret_cast<uintptr_t>(sink) ^ ((reinterpret_cast<uintptr_t>(sink) ^ reinterpret_cast<uintptr_t>(X)) & to_x));
  uint32_t stale = 0, first = 0, timeouts = 0;
  for (uint32_t r = 1; r <= arg->rounds; ++r) {
    bool check = !producer && !ld;
    if (producer) {
      for (uint32_t d = 0; d < arg->delay; ++d)
        __asm__ volatile ("");
      burst_stores(sink, r, burst - 1);
      uint64_t seen = r - 1;
      if (rel == MODE_RL) {
        if (acc == ACC_ST)       store_rl(last, flag, r);
        else if (acc == ACC_AMO) amo_rl(last, flag, r);
        else                     seen = load_rl(last, flag, r);
      } else if (poll) {
        if (acc == ACC_ST)       store_fence(last, r);
        else if (acc == ACC_AMO) amo_fence(last, r);
        else                     seen = load_fence(last);
        *flag = r;
      } else if (rel == MODE_FENCEBAR) {
        if (acc == ACC_ST)       store_fence(last, r);
        else if (acc == ACC_AMO) amo_fence(last, r);
        else                     seen = load_fence(last);
        grp.arrive_and_wait();
      } else {
        if (acc == ACC_ST)       *last = r;
        else if (acc == ACC_AMO) amo_x(last, r);
        else                     seen = load_x(last);
        grp.arrive_and_wait();
      }
      if (lane == 0 && seen != r - 1) {
        ++stale;
        if (first == 0)
          first = r;
      }
    } else {
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
      if (ld)
        *last = r;
    }
    if (check && *X != r) {
      ++stale;
      if (first == 0)
        first = r;
    }
    grp.arrive_and_wait();
  }
  if (producer && lane == 0)
    *done = 1;
  res[tid * RES_WORDS + RES_FIRST]   = first;
  res[tid * RES_WORDS + RES_TIMEOUT] = timeouts;
  res[tid * RES_WORDS + RES_STALE]   = stale;
}

#if VX_CFG_EXT_MBAR_ENABLED
#include <vx_mbar.h>
#endif

// The producer's X store and its release in one asm block, so they issue back to back.
static inline __attribute__((always_inline)) void gstore_rl(volatile uint64_t* p, volatile uint64_t* f, uint64_t v) {
  __asm__ volatile ("sd %0, 0(%1)\n\tamoswap.d.rl zero, %0, (%2)" :: "r"(v), "r"(p), "r"(f) : "memory");
}
#if VX_CFG_EXT_MBAR_ENABLED
static inline __attribute__((always_inline)) void gstore_arrive(volatile uint64_t* p, uint64_t* obj, uint64_t v) {
  __asm__ volatile ("sd %0, 0(%1)\n\t.insn r %3, 0, 0, zero, %2, %4"
                    :: "r"(v), "r"(p), "r"(obj), "i"(RISCV_CUSTOM2), "r"(1ull) : "memory");
}
#endif

// order_k's layout with X in global memory: the producer stores the round into
// X and releases the consumers through the group barrier (bar), amoswap.d.rl
// of the local flag they poll (rl), or an mbarrier arrive of every producer
// lane on a local object they wait on (mbar). Every group lane first stores
// into lines of its own on X's L1 bank, a new line per store and new ones
// each round of four, so the bank's misses fill its MSHR and the producer's X
// store can still be queued when the consumers, possibly on another LSU
// block, load X through the same L1: each of these releases is CTA-scope and
// must wait until the store has reached the L1.
__kernel void global_k(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid   = threadIdx.x;
  const uint32_t warp  = get_sub_group_id();
  const uint32_t nwarp = get_num_sub_groups();
  const uint32_t nt    = vx_num_threads();
  const uint32_t lane  = tid % nt;
  const uint32_t S     = arg->streamers;
  const uint32_t rel   = arg->mode & ((1u << ACC_SHIFT) - 1);

  auto tile = reinterpret_cast<uint32_t*>(__local_mem());
  auto ctrl = reinterpret_cast<volatile uint64_t*>(tile + S * arg->tile_elems);
  volatile uint64_t* obj  = ctrl + arg->x_word;
  volatile uint64_t* flag = obj + 1;
  volatile uint64_t* done = obj + 2;
  volatile uint64_t* X    = reinterpret_cast<volatile uint64_t*>(arg->gx_addr);

  if (tid == 0) {
    *X = 0;
    *obj = 0;
    *flag = 0;
    *done = 0;
#if VX_CFG_EXT_MBAR_ENABLED
    if (rel == MODE_MBAR)
      vx_mbar_init(const_cast<uint64_t*>(obj), nt);
#endif
    vx_fence();
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
  const uint32_t burst = arg->burst;
  const uint32_t gt = (nwarp - S) * nt;
  const uint32_t me = (warp - S) * nt + lane;
  const uintptr_t to_x = -(uintptr_t)(lane == 0);
  // Store b of the lane in a round of the four: line 1 + me + gt * (b + 2 * burst * k).
  const uint32_t step = arg->gx_stride * gt;
  uint32_t stale = 0, first = 0, timeouts = 0;
  for (uint32_t r = 1; r <= arg->rounds; ++r) {
    bool check = !producer;
    volatile uint64_t* sink = X + arg->gx_stride * (1 + me) + step * 2 * burst * (r & 3);
    if (producer) {
      for (uint32_t d = 0; d < arg->delay; ++d)
        __asm__ volatile ("");
      for (uint32_t b = 0; b + 1 < burst; ++b)
        sink[step * b] = r;
      volatile uint64_t* mine = sink + step * (burst - 1);
      volatile uint64_t* last = reinterpret_cast<volatile uint64_t*>(
          reinterpret_cast<uintptr_t>(mine) ^ ((reinterpret_cast<uintptr_t>(mine) ^ reinterpret_cast<uintptr_t>(X)) & to_x));
      if (rel == MODE_RL) {
        gstore_rl(last, flag, r);
#if VX_CFG_EXT_MBAR_ENABLED
      } else if (rel == MODE_MBAR) {
        gstore_arrive(last, const_cast<uint64_t*>(obj), r);
#endif
      } else {
        *last = r;
        grp.arrive_and_wait();
      }
    } else {
      for (uint32_t b = 0; b < 2 * burst; ++b)
        sink[step * b] = r;
      if (rel == MODE_BAR) {
        grp.arrive_and_wait();
      } else {
        uint32_t spins = 0;
        bool ok;
#if VX_CFG_EXT_MBAR_ENABLED
        if (rel == MODE_MBAR) {
          while (!(ok = vx_mbar_test_wait_parity(const_cast<uint64_t*>(obj), r - 1)) && ++spins < SPIN_LIMIT) {}
        } else
#endif
        {
          while (!(ok = (*flag >= r)) && ++spins < SPIN_LIMIT) {}
        }
        if (!ok) {
          ++timeouts;
          check = false;
        }
      }
    }
    if (check && *X != r) {
      ++stale;
      if (first == 0)
        first = r;
    }
    grp.arrive_and_wait();
  }
  if (producer && lane == 0)
    *done = 1;
  res[tid * RES_WORDS + RES_FIRST]   = first;
  res[tid * RES_WORDS + RES_TIMEOUT] = timeouts;
  res[tid * RES_WORDS + RES_STALE]   = stale;
}

// The producer's X load and its release in one asm block, so they issue back to back.
static inline __attribute__((always_inline)) uint64_t gload_rl(volatile uint64_t* p, volatile uint64_t* f, uint64_t r) {
  uint64_t v;
  __asm__ volatile ("ld %0, 0(%1)\n\tamoswap.d.rl zero, %3, (%2)" : "=&r"(v) : "r"(p), "r"(f), "r"(r) : "memory");
  return v;
}
#if VX_CFG_EXT_MBAR_ENABLED
static inline __attribute__((always_inline)) uint64_t gload_arrive(volatile uint64_t* p, uint64_t* obj) {
  uint64_t v;
  __asm__ volatile ("ld %0, 0(%1)\n\t.insn r %3, 0, 0, zero, %2, %4"
                    : "=&r"(v) : "r"(p), "r"(obj), "i"(RISCV_CUSTOM2), "r"(1ull) : "memory");
  return v;
}
#endif
static inline __attribute__((always_inline)) void swap_rl(volatile uint64_t* f, uint64_t v) {
  __asm__ volatile ("amoswap.d.rl zero, %0, (%1)" :: "r"(v), "r"(f) : "memory");
}

// global_k's layout with two more release shapes, in a kernel of its own so
// that global_k's code stays as it is.
// gld (load buffering): the producer loads X, then releases; the consumers
// store the round into X once released, and the producer's lane 0 must still
// have read the previous round. The X load can still be waiting on X's L1 bank
// at the release, which must hold until it is answered.
// elect (gst): the producer's last lane stores X, on the warp's last dcache
// channel, and lane 0 alone releases (rl, or an mbarrier of count 1). The
// release must also order the store lanes on the channels its own lane does
// not use, which can still wait at the L1's input after the lanes on its own
// channel have gone in.
// fpu (gld with bar): a double divide is still in flight at the producer's
// release; in a build with tensor metadata the release's fence waits for it,
// as it shares its scoreboard bit with the fflags writers.
__kernel void release_k(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid   = threadIdx.x;
  const uint32_t warp  = get_sub_group_id();
  const uint32_t nwarp = get_num_sub_groups();
  const uint32_t nt    = vx_num_threads();
  const uint32_t lane  = tid % nt;
  const uint32_t S     = arg->streamers;
  const uint32_t rel   = arg->mode & ((1u << ACC_SHIFT) - 1);
  const bool     ld    = ((arg->mode >> ACC_SHIFT) == ACC_GLD);
  const bool     elect = (arg->elect != 0);

  auto tile = reinterpret_cast<uint32_t*>(__local_mem());
  auto ctrl = reinterpret_cast<volatile uint64_t*>(tile + S * arg->tile_elems);
  volatile uint64_t* obj  = ctrl + arg->x_word;
  volatile uint64_t* flag = obj + 1;
  volatile uint64_t* done = obj + 2;
  volatile uint64_t* X    = reinterpret_cast<volatile uint64_t*>(arg->gx_addr);

  if (tid == 0) {
    *X = 0;
    *obj = 0;
    *flag = 0;
    *done = 0;
#if VX_CFG_EXT_MBAR_ENABLED
    if (rel == MODE_MBAR)
      vx_mbar_init(const_cast<uint64_t*>(obj), elect ? 1 : nt);
#endif
    vx_fence();
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
  const uint32_t burst = arg->burst;
  const uint32_t gt = (nwarp - S) * nt;
  const uint32_t me = (warp - S) * nt + lane;
  const uintptr_t to_x = -(uintptr_t)(lane == ((producer && elect) ? nt - 1 : 0));
  const uint32_t step = arg->gx_stride * gt;
  uint32_t stale = 0, first = 0, timeouts = 0;
  for (uint32_t r = 1; r <= arg->rounds; ++r) {
    bool check = !producer && !ld;
    volatile uint64_t* sink = X + arg->gx_stride * (1 + me) + step * 2 * burst * (r & 3);
    // elect: the producer's low lanes store one line over, on the next L1 bank,
    // so its first dcache channel runs ahead of the one queued behind X's bank.
    if (producer && elect && 2 * lane < nt)
      sink += arg->gx_stride / 16;
    volatile uint64_t* mine = sink + step * (burst - 1);
    volatile uint64_t* last = reinterpret_cast<volatile uint64_t*>(
        reinterpret_cast<uintptr_t>(mine) ^ ((reinterpret_cast<uintptr_t>(mine) ^ reinterpret_cast<uintptr_t>(X)) & to_x));
    if (producer) {
      for (uint32_t d = 0; d < arg->delay; ++d)
        __asm__ volatile ("");
      for (uint32_t b = 0; b + 1 < burst; ++b)
        sink[step * b] = r;
      uint64_t seen = r - 1;
      if (ld) {
        if (rel == MODE_RL) {
          seen = gload_rl(last, flag, r);
#if VX_CFG_EXT_MBAR_ENABLED
        } else if (rel == MODE_MBAR) {
          seen = gload_arrive(last, const_cast<uint64_t*>(obj));
#endif
        } else {
          seen = *last;
          if (arg->fpu) {
            // An fflags writer still in flight at the release.
            double q, num = (double)r, den = 3.0;
            __asm__ volatile ("fdiv.d %0, %1, %2" : "=f"(q) : "f"(num), "f"(den));
            grp.arrive_and_wait();
            if (q != q)
              ++stale;
          } else {
            grp.arrive_and_wait();
          }
        }
      } else {
        *last = r;
        if (lane == 0) {
#if VX_CFG_EXT_MBAR_ENABLED
          if (rel == MODE_MBAR)
            vx_mbar_arrive(const_cast<uint64_t*>(obj), 1);
          else
#endif
            swap_rl(flag, r);
        }
      }
      if (lane == 0 && seen != r - 1) {
        ++stale;
        if (first == 0)
          first = r;
      }
    } else {
      for (uint32_t b = 0; b < 2 * burst; ++b)
        sink[step * b] = r;
      if (rel == MODE_BAR) {
        grp.arrive_and_wait();
      } else {
        uint32_t spins = 0;
        bool ok;
#if VX_CFG_EXT_MBAR_ENABLED
        if (rel == MODE_MBAR) {
          while (!(ok = vx_mbar_test_wait_parity(const_cast<uint64_t*>(obj), r - 1)) && ++spins < SPIN_LIMIT) {}
        } else
#endif
        {
          while (!(ok = (*flag >= r)) && ++spins < SPIN_LIMIT) {}
        }
        if (!ok) {
          ++timeouts;
          check = false;
        }
      }
      if (ld)
        *last = r;
    }
    if (check && *X != r) {
      ++stale;
      if (first == 0)
        first = r;
    }
    grp.arrive_and_wait();
  }
  if (producer && lane == 0)
    *done = 1;
  res[tid * RES_WORDS + RES_FIRST]   = first;
  res[tid * RES_WORDS + RES_TIMEOUT] = timeouts;
  res[tid * RES_WORDS + RES_STALE]   = stale;
}
