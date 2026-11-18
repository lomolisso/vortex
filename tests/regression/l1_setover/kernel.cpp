#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t lane = vx_thread_id();
  if (lane >= arg->lanes)
    return;
  auto starts = reinterpret_cast<const uint64_t*>(arg->start_addr);
  uint64_t p = starts[lane];
  for (uint32_t i = 0; i < arg->hops; ++i) {
    uint64_t q;
    __asm__ volatile ("ld %0, 0(%1)\n" : "=r"(q) : "r"(p) : "memory");
    if (arg->kind == SO_MIX) {
      __asm__ volatile ("sd %0, 8(%1)\n" :: "r"((uint64_t)i), "r"(p) : "memory");
    }
    p = q;
  }
  if (lane == 0) {
    reinterpret_cast<uint64_t*>(arg->out_addr)[0] = p;
  }
}
