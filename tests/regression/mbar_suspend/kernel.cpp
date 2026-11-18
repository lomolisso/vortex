#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dlmem.h>
#include <vx_mbar.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

// The clock is read only once the wait's predicate is consumed (12/spec.md §4.4's
// measurement trap).
INL void consume(bool v) {
  __asm__ volatile ("" :: "r"(v));
}

INL bool try_wait(uint64_t* obj, uint32_t parity, uint32_t hint) {
  return (hint == NO_HINT) ? vx_mbar_try_wait_parity(obj, parity)
                           : vx_mbar_try_wait_parity_hint(obj, parity, hint);
}

INL void knobs(uint32_t* o) {
#ifdef VX_CFG_EXT_MBAR_ENABLE
  o[0] = VX_CFG_MBAR_SUSPEND_CYCLES;
#endif
  o[1] = VX_CFG_H100_FE_ENABLED;
}

} // namespace

__kernel void sus_k(sus_arg_t* __UNIFORM__ arg) {
  const uint32_t w    = get_sub_group_id();
  const uint32_t lane = threadIdx.x % (uint32_t)vx_num_threads();
  auto lm = reinterpret_cast<uint8_t*>(__local_mem());
  auto o  = reinterpret_cast<uint32_t*>(arg->out_addr);
  auto never = reinterpret_cast<uint64_t*>(lm + LM_NEVER);
  auto done  = reinterpret_cast<uint64_t*>(lm + LM_DONE);
  auto wake  = reinterpret_cast<uint64_t*>(lm + LM_WAKE);

  if (threadIdx.x == 0) {
    vx_mbar_init(never, 1);
    vx_mbar_init(done, 1);
    vx_mbar_arrive(done, 1);
    for (uint32_t r = 0; r < MAX_ROUNDS; ++r)
      vx_mbar_init(wake + r, 1);
    o[O_ERR] = 0;
    knobs(o + O_KNOBS);
  }
  __syncthreads();

  // The arriver stores its time a round late: a global store right after the arrive
  // would share the LSU path with the woken warp's next try_wait.
  uint32_t ta = 0;
  for (uint32_t r = 0; r < arg->rounds; ++r) {
    __syncthreads();
    if (r > 0 && w == 1 && lane == 0 && arg->mode >= M_WAKE)
      o[O_TA + r - 1] = ta;
    const uint64_t t0 = vx_rdcycle();
    if (arg->mode == M_LIMIT || arg->mode == M_DONE) {
      if (w == 0 && lane < arg->lanes) {
        const bool ok = (arg->mode == M_LIMIT) ? try_wait(never, 0, arg->hint) : try_wait(done, 0, NO_HINT);
        consume(ok);
        const uint64_t t1 = vx_rdcycle();
        if (lane == 0) {
          o[O_T + r] = (uint32_t)(t1 - t0);
          if (ok != (arg->mode == M_DONE))
            o[O_ERR] = 1;
        }
      }
    } else if (w == 0) {
      bool ok;
      // Separate loops: the timed unhinted loop must not carry the hinted path's code.
      if (arg->mode == M_WAKE && arg->hint != NO_HINT) {
        do {
          ok = vx_mbar_try_wait_parity_hint(wake + r, 0, arg->hint);
          consume(ok);
        } while (!ok);
      } else {
        do {
          ok = (arg->mode == M_WAKE) ? vx_mbar_try_wait_parity(wake + r, 0) : vx_mbar_test_wait_parity(wake + r, 0);
          consume(ok);
        } while (!ok);
      }
      const uint64_t t1 = vx_rdcycle();
      if (lane == 0)
        o[O_TW + r] = (uint32_t)t1;
    } else if (w == 1 && lane == 0) {
      while (vx_rdcycle() - t0 < arg->delay) {}
      ta = (uint32_t)vx_rdcycle();
      vx_mbar_arrive(wake + r, 1);
    }
  }
  __syncthreads();
  if (arg->rounds > 0 && w == 1 && lane == 0 && arg->mode >= M_WAKE)
    o[O_TA + arg->rounds - 1] = ta;
}
