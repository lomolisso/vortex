#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Lane 0 of each block records where it landed. With a rendezvous, every block
// of a cluster arrives on one counter and then polls it with an atomic (a plain
// load could hit a stale line in a non-coherent L1) until the whole cluster has
// arrived, which only happens if all its blocks are resident at once.
__kernel void gang_k(gang_arg_t* __UNIFORM__ arg) {
  const uint32_t work = arg->work;
  uint32_t x = threadIdx.x;
  for (uint32_t i = 0; i < work; ++i) {
    asm volatile ("addi %0, %0, 1" : "+r"(x));
  }
  if (threadIdx.x != 0)
    return;

  const uint32_t b = blockIdx.x;
  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * REC_WORDS;
  const uint64_t lmem = (uint64_t)(uintptr_t)__local_mem();

  uint32_t lmem_ok = 1;
  if (arg->lmem_bytes != 0) {
    auto slice = reinterpret_cast<volatile uint32_t*>(__local_mem());
    const uint32_t last = arg->lmem_bytes / sizeof(uint32_t) - 1;
    slice[0]    = b ^ 0xA5A50000u;
    slice[last] = ~b;
    lmem_ok = (uint32_t)(slice[0] == (b ^ 0xA5A50000u)) & (uint32_t)(slice[last] == ~b);
  }

  uint32_t polls = 0;
  if (arg->arrive_addr != 0) {
    const uint32_t k = arg->k;
    const uint32_t bound = arg->bound;
    auto cnt = reinterpret_cast<uint32_t*>(arg->arrive_addr) + b / k;
    __atomic_fetch_add(cnt, 1u, __ATOMIC_SEQ_CST);
    uint32_t seen;
    do {
      ++polls;
      seen = __atomic_fetch_or(cnt, 0u, __ATOMIC_SEQ_CST);
    } while (seen < k && polls < bound);
    if (seen < k)
      polls = 0;
  }

  rec[REC_BLOCK]   = b;
  rec[REC_CORE]    = vx_core_id();
  rec[REC_CTA_ID]  = get_local_group_id();
  rec[REC_RANK]    = get_cluster_rank();
  rec[REC_SIZE]    = get_cluster_size();
  rec[REC_LMEM_LO] = (uint32_t)lmem;
  rec[REC_LMEM_HI] = (uint32_t)(lmem >> 32);
  rec[REC_LMEM_OK] = lmem_ok;
  rec[REC_POLLS]   = polls;
  // Counting runs needs an atomic: a plain read could hit a stale L1 line.
  if (arg->arrive_addr != 0) {
    __atomic_fetch_add(&rec[REC_RUNS], 1u, __ATOMIC_SEQ_CST);
  } else {
    rec[REC_RUNS] = 1;
  }
}
