// The coarse-key map and its exact f32 boundaries.
// Re-implementation of hpc-ops 2a2e265 src/topk/topk_filtered_boundary.cuh
// (hpc-ops__pin).
// Departures: fp32_boundary_accepts is left out (the cluster-8 kernel does not
// call it).
#pragma once

#include <vx_cuda.h>
#include <stdint.h>

namespace hpcops_topk {

// Descending 11-bit key of the f16 rounding of x: larger scores get smaller
// keys, so bin 0 holds the largest values.
__device__ __forceinline__ uint16_t to_coarse_key(float x) {
  const uint16_t bits = __half_as_ushort(__float2half_rn(x));
  const uint16_t key = (bits & 0x8000) ? bits : static_cast<uint16_t>((bits ^ 0xFFFF) & 0x7FFF);
  return static_cast<uint16_t>(key >> 5);
}

// The least f32 whose coarse key is at most tau; `empty` when no score of
// the operator's domain (no NaN) has such a key.
struct CoarseBoundary {
  float value;
  bool empty;
};

// The form the scans test: an empty boundary is a NaN, which no comparison
// accepts; the bit pattern keeps +0 and -0 apart.
struct PackedCoarseBoundary {
  float value;
  uint32_t bits;
};

enum class CoarseClass : uint8_t {
  kSelected,
  kCandidate,
  kDropped,
};

// The f16 value that ends key bucket tau (its smallest member) and its
// neighbour on the rejected side meet halfway; the halfway point itself
// rounds to the even one, so an odd accepted significand moves the boundary
// one f32 ulp toward +inf.
__device__ __forceinline__ CoarseBoundary coarse_boundary(int tau) {
  if (tau < 31)
    return {CUDART_INF_F, true};
  if (tau >= 2016)
    return {-CUDART_INF_F, false};

  const uint16_t max_key = static_cast<uint16_t>((tau << 5) | 31);
  const uint16_t accepted_bits = max_key < 0x8000 ? static_cast<uint16_t>(0x7FFFu - max_key) : max_key;
  if (accepted_bits == 0)
    return {0.0f, false};
  // +65520 is the least f32 that rounds to +inf.
  if (accepted_bits == 0x7C00u)
    return {65520.0f, false};
  // -65520 rounds to -inf, so the bucket of the most negative finite f16
  // starts one ulp above it.
  if (accepted_bits == 0xFBFFu)
    return {__uint_as_float(__float_as_uint(-65520.0f) - 1u), false};

  const uint16_t rejected_bits =
      (accepted_bits & 0x8000u) ? static_cast<uint16_t>(accepted_bits + 1) : static_cast<uint16_t>(accepted_bits - 1);
  const float accepted = __half2float(__ushort_as_half(accepted_bits));
  const float rejected = __half2float(__ushort_as_half(rejected_bits));
  float boundary = (accepted + rejected) * 0.5f;
  if (accepted_bits & 1u) {
    uint32_t bits = __float_as_uint(boundary);
    bits += (bits & 0x80000000u) ? static_cast<uint32_t>(-1) : 1u;
    boundary = __uint_as_float(bits);
  }
  return {boundary, false};
}

__device__ __forceinline__ PackedCoarseBoundary pack_coarse_boundary(const CoarseBoundary& b) {
  const float value = b.empty ? CUDART_NAN_F : b.value;
  return {value, __float_as_uint(value)};
}

__device__ __forceinline__ bool coarse_accepts(float x, const PackedCoarseBoundary& b) {
  return x > b.value || __float_as_uint(x) == b.bits;
}

// Selected above the strict boundary, candidate inside the threshold bucket,
// dropped below; most scores fail the first test.
__device__ __forceinline__ CoarseClass coarse_classify(float x, const PackedCoarseBoundary& select_boundary,
                                                      const PackedCoarseBoundary& candidate_boundary) {
  if (!coarse_accepts(x, candidate_boundary))
    return CoarseClass::kDropped;
  return coarse_accepts(x, select_boundary) ? CoarseClass::kSelected : CoarseClass::kCandidate;
}

} // namespace hpcops_topk
