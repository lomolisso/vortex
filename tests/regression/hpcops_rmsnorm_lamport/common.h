#ifndef _LAMPORT_COMMON_H_
#define _LAMPORT_COMMON_H_

#include <stddef.h>
#include <stdint.h>

// The argument block of rmsNormLamport<T, T, 1>, in the original's parameter
// order, plus the harness's smid buffer (one word per block).
struct LamportArgs {
  uint64_t outputPreNorm, outputNorm, bufferInput, gamma;
  float epsilon;
  uint64_t residual;
  uint32_t numTokens, dim, worldSize;
  uint64_t bufferFlags;
  uint64_t smid;
};

// The kernel's static shared variables (blockReduceSumFull's smem[32] and
// sharedVal[8]), at offset 0 of the block's local memory in every block; the
// original's dynamic shared memory starts at kDynOffset.
struct LamportSmem {
  float reduce_smem[32];
  float sharedVal[8];
};
constexpr uint32_t kDynOffset = (uint32_t)((sizeof(LamportSmem) + 15) & ~(size_t)15);

#endif
