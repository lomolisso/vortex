#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"
#include "chase.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto smem = reinterpret_cast<volatile uint32_t*>(__local_mem());
  const uint32_t tid = threadIdx.x;
  const uint32_t nthr = blockDim.x;
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);

  if (arg->kind == CV_STAMP) {
    const uint64_t t0 = vx_rdcycle();
    if (tid == 0) {
      // The last cycle lives in shared memory, which the next launch sees at the same index: a global
      // store would add its write ack, whose DRAM latency follows the write queue's history, to the gap.
      auto last = reinterpret_cast<volatile uint64_t*>(smem);
      const uint64_t gap = t0 - *last;
      uint64_t t1;
      do {
        t1 = vx_rdcycle();
      } while (t1 - t0 < arg->spin);
      out[arg->n * 2 + 0] = gap;
      out[arg->n * 2 + 1] = t1;
      vx_fence();
      *last = vx_rdcycle();
    }
    return;
  }

  switch (arg->kind) {
  case CV_CHASE:
    if (tid == 0)
      cv_chase(arg, smem);
    break;
  case CV_FILL:
    for (uint32_t i = tid; i < arg->lmem_words; i += nthr)
      smem[i] = arg->mark_new ^ i;
    break;
  case CV_TOUCH:
    // Every word, so a sectored L1 fills each line whole.
    for (uint64_t o = (uint64_t)tid * 4; o < (uint64_t)arg->n * arg->line; o += (uint64_t)nthr * 4) {
      uint64_t v;
      __asm__ volatile ("lw %0, 0(%1)\n add %0, %0, zero\n" : "=&r"(v) : "r"(arg->start_addr + o) : "memory");
    }
    break;
  case CV_DUMP: {
    auto o = reinterpret_cast<uint32_t*>(arg->out_addr);
    for (uint32_t i = tid; i < arg->lmem_words; i += nthr)
      o[i] = smem[i];
    break;
  }
  case CV_OCC:
    // The barriers keep every warp of the block alive for the whole window: the dispatcher frees warp slots per warp.
    __syncthreads();
    if (tid < (uint32_t)vx_num_threads()) {
      // Warp 0 waits on a load of one line per lane each turn: the SM's L1 returns one line per cycle, so the
      // waiting warps leave the front end to the others, which a lowest-ready-warp-first pick would starve.
      const uint64_t a = arg->start_addr + (uint64_t)tid * arg->line;
      // Every block that starts within one period ends at the same boundary, so
      // the window holds all of them however long their start-up takes.
      uint64_t t0 = vx_rdcycle(), t1;
      const uint64_t end = (t0 / arg->spin + 1) * arg->spin;
      do {
        uintptr_t v;
        __asm__ volatile ("lw %0, 0(%1)\n and %0, %0, zero\n" : "=&r"(v) : "r"(a) : "memory");
        t1 = vx_rdcycle() + v;
      } while (t1 < end);
      if (tid == 0) {
        const uint32_t b = blockIdx.x;
        out[b * 3 + 0] = (uint64_t)vx_core_id();
        out[b * 3 + 1] = t0;
        out[b * 3 + 2] = t1;
      }
    }
    __syncthreads();
    break;
  }
}
