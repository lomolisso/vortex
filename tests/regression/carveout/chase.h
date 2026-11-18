#ifndef _CARVEOUT_CHASE_H_
#define _CARVEOUT_CHASE_H_

#include <vx_intrinsics.h>

#include "common.h"

#define INL static inline __attribute__((always_inline))

// Cycles from a load's issue until an instruction that uses it can issue; *p follows the ring.
INL uint64_t timed_ld(uint64_t* p) {
  uint64_t t0 = vx_rdcycle();
  uint64_t v;
  __asm__ volatile ("ld %0, 0(%1)\n add %0, %0, zero\n" : "=&r"(v) : "r"(*p) : "memory");
  uint64_t t1 = vx_rdcycle();
  *p = v;
  return t1 - t0;
}

// CV_CHASE on one thread; smem is touched only when mark_new != 0. An untimed pass brings the ring into
// the L2 (the host wrote it to DRAM) and the fence clears the L1, so every first-pass load is an L2 hit,
// timed together with an L1 hit of the same line. A second-pass load is a miss above the midpoint of the
// fastest of each, taken after the first hop (which also waits for the L1 clear and runs cold code).
INL void cv_chase(const kernel_arg_t* arg, volatile uint32_t* smem) {
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
  const uint32_t n = arg->n;
  uint64_t p = arg->start_addr;
  uint64_t bad = 0;
  for (uint32_t i = 0; arg->mark_new != 0 && i < CV_MARK_WORDS; ++i) {
    if (arg->mark_old != 0)
      bad += (smem[i] != (arg->mark_old ^ i));
    smem[i] = arg->mark_new ^ i;
  }
  for (uint32_t i = 0; i < n; ++i)
    p = *reinterpret_cast<volatile uint64_t*>(p);
  // Arguments are read before the fence: after it the passes touch only the ring's lines.
  __asm__ volatile ("fence iorw, iorw" ::: "memory");
  uint64_t first = 0, l2 = ~0ull, hit = ~0ull, thr = 0, total = 0;
  uint32_t misses = 0;
  // Both passes run one loop, so the counted pass runs warm code.
  for (uint32_t i = 0; i < 2 * n; ++i) {
    uint64_t q = p;
    const uint64_t c = timed_ld(&p);
    if (i < n) {
      const uint64_t h = timed_ld(&q);
      if (i == 0) {
        first = c;
      } else {
        l2 = c < l2 ? c : l2;
        hit = h < hit ? h : hit;
      }
      thr = (hit + l2) / 2;
    } else {
      total += c;
      misses += (c > thr);
    }
  }
  out[0] = misses;
  out[1] = total;
  out[2] = first;
  out[3] = bad;
  out[4] = thr;
}

#endif
