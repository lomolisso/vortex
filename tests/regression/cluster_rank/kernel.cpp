#include <vx_spawn2.h>
#include <vx_dlmem.h>
#include "common.h"

static inline uint32_t linear_block() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

// One lane per CTA records what the CTA sees of its cluster; with -m it also
// maps one of its local-memory objects into every rank of the cluster.
__kernel void rank_k(rank_arg_t* __UNIFORM__ arg) {
  if (threadIdx.x != 0)
    return;
  const uint32_t b = linear_block();
  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * REC_WORDS;
  auto cluster = vortex::this_cluster();
  rec[REC_BLOCK_X]    = blockIdx.x;
  rec[REC_BLOCK_Y]    = blockIdx.y;
  rec[REC_BLOCK_Z]    = blockIdx.z;
  rec[REC_CTA_ID]     = get_local_group_id();
  rec[REC_RANK]       = vx_cluster_block_rank();
  rec[REC_GROUP_RANK] = cluster.block_rank();
  rec[REC_SPAWN_RANK] = get_cluster_rank();
  rec[REC_NUM_BLOCKS] = vx_cluster_num_blocks();
  if (arg->map_addr) {
    auto map = reinterpret_cast<uint64_t*>(arg->map_addr) + (uint64_t)b * arg->num_blocks;
    auto obj = reinterpret_cast<uint64_t*>(
        static_cast<char*>(__local_mem()) + map_offset(b, arg->lmem_bytes));
    for (uint32_t r = 0; r < arg->num_blocks; ++r)
      map[r] = (uint64_t)(uintptr_t)cluster.map_shared_rank(obj, r);
  }
}

// Every thread of the CTA writes its share of the whole local-memory slice,
// then, after the barrier, checks the share another warp wrote.
__kernel void lmem_k(lmem_arg_t* __UNIFORM__ arg) {
  const uint32_t tid = threadIdx.x;
  const uint32_t nthreads = blockDim.x;
  const uint32_t b = linear_block();
  const uint32_t n = arg->lmem_bytes / sizeof(uint64_t);
  auto slice = reinterpret_cast<volatile uint64_t*>(__local_mem());
  for (uint32_t i = tid; i < n; i += nthreads)
    slice[i] = lmem_pattern(b, i);
  __syncthreads();
  uint32_t errors = 0, first = 0xffffffffu;
  for (uint32_t i = tid; i < n; i += nthreads) {
    const uint32_t j = n - 1 - i;
    if (slice[j] != lmem_pattern(b, j)) {
      if (errors == 0)
        first = j;
      ++errors;
    }
  }
  auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + 2 * (b * nthreads + tid);
  err[0] = errors;
  err[1] = first;
  if (tid == 0)
    reinterpret_cast<uint64_t*>(arg->base_addr)[b] = (uint64_t)(uintptr_t)slice;
}
