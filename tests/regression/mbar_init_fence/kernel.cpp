#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dlmem.h>
#include <vx_mbar.h>
#include "common.h"

// Each round rank 0 initialises a fresh mbarrier with a plain store, fences,
// and hands it to rank 1 with a relaxed window store of the round number;
// rank 1 then arrives on it through the window. Every object starts out
// initialised for 2 arrivals, so an arrive that overtakes its init is lost
// and rank 0's bounded wait gives the round up. Rank 0's other warps keep its
// banks busy with atomics meanwhile.

// A copy of v the compiler cannot relate to earlier tests of v: a branch on
// it is not threaded back across a barrier, which would duplicate the
// barrier into both sides of an earlier lane-divergent branch.
static inline __attribute__((always_inline)) uint32_t opaque(uint32_t v) {
  __asm__ volatile ("" : "+r"(v));
  return v;
}

__kernel void kernel_main(fence_arg_t* __UNIFORM__ arg) {
  vortex::cluster_group cl;
  const uint32_t rank = cl.block_rank();
  const uint32_t tid = threadIdx.x;
  const uint32_t nt = vx_num_threads();
  uint64_t* own = reinterpret_cast<uint64_t*>(__local_mem());
  volatile uint64_t* vown = own;
  uint64_t* peer = cl.map_shared_rank(own, rank ^ 1);
  volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(arg->out_addr) + rank * OUT_WORDS;
  const uint32_t R = arg->rounds;
  for (uint32_t i = tid; i < LM_WORDS; i += blockDim.x)
    own[i] = 0;
  for (uint32_t i = tid; i < R; i += blockDim.x)
    vx_mbar_init(own + LM_BAR + i, 2);
  cl.sync();

  uint32_t lost = 0, first = 0, ham = 0;
  const uint32_t t0 = opaque(tid);
  if (rank == 0 && t0 == 0) {
    for (uint32_t i = 0; i < R; ++i) {
      uint64_t* bar = own + LM_BAR + i;
      vx_mbar_init(bar, 1);
      switch (arg->fence) {
      case FENCE_CTA:     vx_fence_cta(); break;
      case FENCE_CLUSTER: vx_fence_cluster(); break;
      case FENCE_INIT:    vx_fence_mbarrier_init_release_cluster(); break;
      default: break;
      }
      *(volatile uint64_t*)(peer + LM_FLAG) = i + 1;
      bool ok = false;
      for (uint32_t t = 0; t < arg->tries && !ok; ++t)
        ok = vx_mbar_try_wait_parity_relaxed(bar, 0);
      if (!ok && lost++ == 0)
        first = i + 1;
    }
    vown[LM_DONE] = 1;
  } else if (rank == 0 && t0 >= nt && arg->hammer) {
    uint64_t* h = own + LM_HAM + (tid & 63);
    while (vown[LM_DONE] == 0) {
      __atomic_fetch_add(h, 1, __ATOMIC_RELAXED);
      ++ham;
    }
  } else if (rank == 1 && t0 == 0) {
    uint64_t* pbar = cl.map_shared_rank(own + LM_BAR, 0);
    for (uint32_t i = 0; i < R; ++i) {
      while (vown[LM_FLAG] < i + 1) {}
      vx_mbar_arrive_relaxed(pbar + i, 1);
    }
  }
  cl.sync();
  const uint32_t t = opaque(tid);
  if (t == 0) {
    out[OUT_RAN] = 1;
    out[OUT_RANK] = rank;
    out[OUT_LOST] = lost;
    out[OUT_FIRST] = first;
    out[OUT_CORE] = vx_core_id();
  }
  if (t == nt)
    out[OUT_HAM] = ham;
}
