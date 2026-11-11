// Device-side access to the block-local layout of common.h.
#pragma once

#include <vx_cuda.h>
#include "common.h"

namespace credit {

namespace cg = cooperative_groups;

// The kernel's former __shared__ variables, at offset 0 of the block's local memory.
template <class S>
__device__ __forceinline__ S& shared_vars() {
  return *reinterpret_cast<S*>(__local_mem());
}

// The staging tile (the original's extern __shared__ float tile[]).
template <class S>
__device__ __forceinline__ float* tile() {
  return reinterpret_cast<float*>((char*)__local_mem() + tile_offset<S>());
}

// The block's core, for the harness's placement check.
__device__ __forceinline__ void record_smid(uint64_t smid_buf) {
  if (threadIdx.x == 0)
    ((unsigned*)smid_buf)[blockIdx.x] = __smid();
}

} // namespace credit
