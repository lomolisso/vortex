#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Each block fills an SM, so the rendezvous completes only when every block is
// resident at once, one per SM.
__kernel void sm_id(id_arg_t* __UNIFORM__ arg) {
  if (threadIdx.x != 0)
    return;
  const uint32_t b = blockIdx.x;
  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * REC_WORDS;
  auto cnt = reinterpret_cast<uint32_t*>(arg->arrive_addr);
  __atomic_fetch_add(cnt, 1u, __ATOMIC_SEQ_CST);
  uint32_t polls = 0, seen;
  do {
    ++polls;
    seen = __atomic_fetch_or(cnt, 0u, __ATOMIC_SEQ_CST);
  } while (seen < arg->blocks && polls < arg->bound);
  rec[REC_SMID]  = vx_core_id();
  rec[REC_NSMID] = vx_num_cores();
  rec[REC_POLLS] = (seen < arg->blocks) ? 0 : polls;
  rec[REC_RUNS]  = 1;
}
