#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

// Every lane of the poll block's warps reads one local-memory flag with an
// atomic, back to back, so those lanes keep the flag's bank busy. Lane 0 of
// the producer warp, issuing through another LSU block, sets the flag with a
// plain store after a short delay. The bank must still accept that store: a
// poller that never sees the flag gives up after `bound` polls and reports 0.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto flag = reinterpret_cast<uint32_t*>(__local_mem());
  const uint32_t wid  = vx_warp_id();
  const uint32_t lane = vx_thread_id();
  const uint32_t nw   = vx_num_warps();
  const uint32_t nt   = vx_num_threads();

  // A previous launch leaves the flag set; the fence lands the clear before
  // any poller can read it.
  if (threadIdx.x == 0) {
    *reinterpret_cast<volatile uint32_t*>(flag) = 0;
    __sync_synchronize();
  }
  __syncthreads();

  if (wid == arg->producer_wid) {
    if (lane == 0) {
      const uint32_t delay = arg->delay;
      uint32_t x = 0;
      for (uint32_t i = 0; i < delay; ++i) {
        asm volatile ("addi %0, %0, 1" : "+r"(x));
      }
      *reinterpret_cast<volatile uint32_t*>(flag) = 1;
    }
  } else if (wid % arg->num_blocks == arg->poll_block) {
    const uint32_t bound = arg->bound;
    uint32_t polls = 0;
    uint32_t seen;
    do {
      ++polls;
      seen = __atomic_fetch_or(flag, 0u, __ATOMIC_RELAXED);
    } while (seen == 0 && polls < bound);
    auto out = reinterpret_cast<uint32_t*>(arg->polls_addr);
    out[(blockIdx.x * nw + wid) * nt + lane] = seen ? polls : 0;
  }
}
