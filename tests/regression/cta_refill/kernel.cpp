#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Thread 0 reads the clock first and last, as the H100 measurements read
// %globaltimer, so the gap between two blocks on one SM spans the first
// block's exit tail, the refill and the next block's start.
__kernel void cta_t(cta_arg_t* __UNIFORM__ arg) {
  const uint64_t t0 = vx_rdcycle();
  const uint32_t b = blockIdx.x;
  const uint32_t work = (b / arg->k == arg->long_cluster) ? arg->long_work : arg->work;
  uint32_t x = threadIdx.x;
  for (uint32_t i = 0; i < work; ++i) {
    asm volatile ("addi %0, %0, 1" : "+r"(x));
  }
  if (threadIdx.x != 0)
    return;
  const uint64_t t1 = vx_rdcycle();

  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * REC_WORDS;
  rec[REC_BLOCK] = b;
  rec[REC_SMID]  = vx_core_id();
  rec[REC_SLOT]  = get_local_group_id();
  rec[REC_RANK]  = get_cluster_rank();
  rec[REC_SIZE]  = get_cluster_size();
  rec[REC_T0_LO] = (uint32_t)t0;
  rec[REC_T0_HI] = (uint32_t)(t0 >> 32);
  rec[REC_T1_LO] = (uint32_t)t1;
  rec[REC_T1_HI] = (uint32_t)(t1 >> 32);
  rec[REC_RUNS]  = 1;
}
