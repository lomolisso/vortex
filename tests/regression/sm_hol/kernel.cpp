#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dlmem.h>

#include "common.h"

#define INL static inline __attribute__((always_inline))

INL uint64_t opaque(uint64_t v) {
  __asm__ volatile ("" : "+r"(v));
  return v;
}

// One lane's dependent 8-B load: the time until its value is usable.
INL uint64_t timed_ld(uint64_t a, uint64_t* v) {
  uint64_t t0 = vx_rdcycle();
  uint64_t x = 0;
  if (vx_thread_id() == 0) {
    __asm__ volatile ("ld %0, 0(%1)\n add %0, %0, zero\n" : "=&r"(x) : "r"(a) : "memory");
  }
  uint64_t t1 = vx_rdcycle();
  *v += x;
  return t1 - t0;
}

// Every lane stores 4 B of each line: one full 128-B line per instruction.
INL void burst(uint64_t base, uint32_t lines) {
  const uint64_t v = opaque(1);
  uint64_t p = base + vx_thread_id() * 4;
  for (uint32_t i = 0; i < lines; i += 8, p += 1024) {
    __asm__ volatile ("sw %0, 0(%1)\n sw %0, 128(%1)\n sw %0, 256(%1)\n sw %0, 384(%1)\n"
                      "sw %0, 512(%1)\n sw %0, 640(%1)\n sw %0, 768(%1)\n sw %0, 896(%1)\n"
                      :: "r"(v), "r"(p) : "memory");
  }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  char* own = reinterpret_cast<char*>(__local_mem());
  const uint32_t rank = vx_cluster_block_rank();
  vx_cluster_sync();
  if (rank == 0 && vx_warp_id() == 0) {
    auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
    const uint64_t peer = reinterpret_cast<uint64_t>(vx_cluster_map_shared_rank(own, 1));
    const uint64_t mine = reinterpret_cast<uint64_t>(own);
    uint64_t v = 0;
    // Each access alone, warm (its line touched once before).
    timed_ld(peer, &v);
    timed_ld(mine, &v);
    out[0] = timed_ld(peer + 64, &v);
    out[1] = timed_ld(mine + 64, &v);
    vx_fence();
    // A store burst, then the access issued right behind it.
    uint64_t t0 = vx_rdcycle();
    burst(arg->gmem_addr, arg->lines);
    out[2] = vx_rdcycle() - t0;
    out[3] = timed_ld(peer + 128, &v);
    vx_fence();
    t0 = vx_rdcycle();
    burst(arg->gmem_addr, arg->lines);
    out[4] = vx_rdcycle() - t0;
    out[5] = timed_ld(mine + 128, &v);
    vx_fence();
    out[6] = vx_rdcycle() - t0;
    out[7] = v;
  }
  vx_cluster_sync();
}
