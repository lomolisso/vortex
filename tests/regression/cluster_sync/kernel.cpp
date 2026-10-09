#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dlmem.h>
#include "common.h"

static inline void work(uint32_t n) {
  uint32_t x = 0;
  for (uint32_t i = 0; i < n; ++i) {
    asm volatile ("addi %0, %0, 1" : "+r"(x));
  }
}

// Each phase, one lane per warp adds 1 to its cluster's counter for that phase
// before arriving, and reads it back with an atomic after the release (a plain
// load could hit a stale L1 line): every warp of the cluster must have counted.
// The arrive tokens follow the phase, since a block's barrier starts at phase 0.
__kernel void sync_k(sync_arg_t* __UNIFORM__ arg) {
  const uint32_t k      = arg->k;
  const uint32_t phases = arg->phases;
  const uint32_t mode   = arg->mode;
  const uint32_t spin   = arg->spin;
  const uint32_t b      = blockIdx.x;
  const uint32_t warps  = get_num_sub_groups();
  const uint32_t expect = k * warps;
  const bool lead = (threadIdx.x % arg->num_threads) == 0;
  auto cluster = vortex::this_cluster();
  const uint32_t rank = cluster.block_rank();
  auto count = arg->count_addr
             ? reinterpret_cast<uint32_t*>(arg->count_addr) + (b / k) * phases
             : nullptr;
  vortex::barrier block_bar(1);

  uint32_t errors = 0;
  uint32_t i = 0;
  for (; i < phases; ++i) {
    const bool last = (i + 1 == phases);
    // Late counts: a barrier that let a warp through early reads a short counter.
    // Without counters every block works the same, so the timing is fixed.
    uint32_t late;
    if (mode == MODE_SKEW) {
      late = rank * spin;
    } else if (mode == MODE_SPLIT) {
      late = (rank == 0 && (i & 1) == 0) ? spin * k : 0;
    } else {
      late = (count && rank == i % k) ? 2 * spin : spin;
    }
    work(late);
    if (count && lead) {
      if (__atomic_fetch_add(&count[i], 1u, __ATOMIC_SEQ_CST) >= expect)
        ++errors;
    }
    uint32_t token = i & 1;
    switch (mode) {
    case MODE_SPLIT:
      // Rank 0 arrives last on even phases; the work after arriving puts the
      // waits of the higher ranks after the release.
      token = cluster.barrier_arrive();
      work(2 * rank * spin);
      cluster.barrier_wait(token);
      break;
    case MODE_SKEW:
      cluster.sync();
      break;
    case MODE_MIXED:
      token = cluster.barrier_arrive();
      __syncthreads();
      block_bar.arrive_and_wait();
      cluster.barrier_wait(token);
      break;
    case MODE_EXIT:
      if (last) {
        token = cluster.barrier_arrive();
      } else {
        cluster.sync();
      }
      break;
    default:
      if (arg->relaxed) {
        token = cluster.barrier_arrive();
        cluster.barrier_wait(token);
      } else {
        cluster.sync();
      }
      break;
    }
    if (token != (i & 1))
      ++errors;
    if (mode == MODE_EXIT && last)
      break;
    if (count && lead) {
      if (__atomic_fetch_or(&count[i], 0u, __ATOMIC_SEQ_CST) != expect)
        ++errors;
    }
  }
  const uint32_t done = (mode == MODE_EXIT) ? i + 1 : i;

  if (lead && errors != 0)
    reinterpret_cast<uint32_t*>(arg->err_addr)[b * warps + get_sub_group_id()] = errors;
  if (threadIdx.x != 0)
    return;
  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * REC_WORDS;
  rec[REC_BLOCK]  = b;
  rec[REC_CORE]   = vx_core_id();
  rec[REC_CTA_ID] = get_local_group_id();
  rec[REC_RANK]   = rank;
  rec[REC_SIZE]   = cluster.num_blocks();
  rec[REC_PHASES] = done;
  // Counting runs needs an atomic: a plain read could hit a stale L1 line.
  if (count) {
    __atomic_fetch_add(&rec[REC_RUNS], 1u, __ATOMIC_SEQ_CST);
  } else {
    rec[REC_RUNS] = 1;
  }
}
