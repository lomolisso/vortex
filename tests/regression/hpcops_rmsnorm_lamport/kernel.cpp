// The RMSNorm half of hpc-ops' fused all-reduce + RMSNorm (low-latency
// Lamport mode): one cluster per token, the hidden dimension split into one
// contiguous chunk per block. Each block copies its chunk of the residual and
// gamma into local memory with cp.async, loads the all-reduced input, writes
// h = input + residual (bf16) as the pre-norm output, and sums h^2; the
// block sums are all-reduced over the cluster by a replicated push into every
// block's sharedVal[rank] and one split cluster barrier, and each block writes
// its chunk of h * rsqrt(sum / dim + eps) * gamma.
//
// Port of hpc-ops src/allreduce/fuse_allreduce_rmsnorm_low_latency.cu
// (commit 2a2e265), kernels::rmsNormLamport<__nv_bfloat16, __nv_bfloat16, 1>,
// with its helpers from fuse_allreduce_rmsnorm_low_latency.h (PackedVec,
// loadPacked, copyF4, toFloat/fromFloat, ceil_div/round_up, warpReduceSumFull,
// blockReduceSum and LamportFlags::ctaArrive), following the extraction
// h100-dsmem-collectives/kernels/rmsnorm_lamport__fused_allreduce_rmsnorm__hpcops.cu
// for what is multi-GPU. Not run on the H100.
// Departures:
// - the multi-GPU Lamport parts are stubbed: the input is bufferInput itself
//   (no LamportFlags buffer rotation), it is read once (no poll until no
//   element is -0.0), the global arrival counter add of flag.ctaArrive() is
//   stubbed (its cluster.sync() is kept, at the call site), and
//   flag.waitAndUpdate() is left out; bufferFlags, numTokens and worldSize
//   stay in the argument block, unused;
// - blockReduceSum keeps only its whole-warp path (blockReduceSumFull): the
//   block is 128 threads;
// - the static shared variables (blockReduceSumFull's smem and sharedVal) are
//   the members of LamportSmem at offset 0 of the block's local memory, the
//   original's dynamic shared memory follows at kDynOffset, and the launch
//   requests both as dynamic local memory;
// - the parameters are one argument block, the template kernel is exported
//   by name, and thread 0 of each block also stores its core id (smid) for
//   the harness's placement check;
// - the 16-byte float4 accesses compile to narrower loads and stores (Vortex
//   has no 128-bit access), the volatile input load included;
// - PackedVec values are set by assigning .packed, not by a designated
//   initializer;
// - built with -fconvergent-functions (Makefile): the barriers are inline asm,
//   which clang may otherwise copy into both arms of a divergent branch.

#include <vx_cuda.h>
#include <type_traits>
#include "common.h"

namespace lamport {

template <typename T1, typename T2>
__forceinline__ __device__ constexpr T1 ceil_div(const T1 x, const T2 y) noexcept {
  return (x + y - 1) / y;
}

template <typename T1, typename T2>
__forceinline__ __device__ constexpr T1 round_up(const T1 x, const T2 y) noexcept {
  return ceil_div(x, y) * y;
}

template <typename T>
inline __device__ float toFloat(T val) {
  return val;
}
template <>
inline __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
template <>
inline __device__ float toFloat<__nv_half>(__nv_half val) {
  return __half2float(val);
}

template <typename T>
inline __device__ T fromFloat(float val) {
  return val;
}
template <>
inline __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}
template <>
inline __device__ __nv_half fromFloat<__nv_half>(float val) {
  return __float2half(val);
}

// A 16-byte packet seen as its elements; + adds element-wise in T.
template <typename PackedType, typename T>
union PackedVec {
  PackedType packed;
  T elements[sizeof(PackedType) / sizeof(T)];

  __device__ PackedVec operator+(PackedVec& other) {
    PackedVec result;
#pragma unroll
    for (int i = 0; i < (int)(sizeof(PackedType) / sizeof(T)); i++)
      result.elements[i] = elements[i] + other.elements[i];
    return result;
  }
};

template <typename PackedType, typename T>
inline __device__ PackedType loadPacked(T* ptr) {
  return *reinterpret_cast<PackedType*>(ptr);
}

// ld.volatile.global.v4.f32
inline __device__ float4 loadPackedVolatile(void const* ptr) {
  const volatile float* p = reinterpret_cast<const volatile float*>(ptr);
  float4 v;
  v.x = p[0];
  v.y = p[1];
  v.z = p[2];
  v.w = p[3];
  return v;
}

template <typename T_IN>
inline __device__ void copyF4(T_IN* dst, T_IN const* src) {
  __pipeline_memcpy_async(reinterpret_cast<float4*>(dst), reinterpret_cast<float4 const*>(src), sizeof(float4));
}

constexpr uint32_t kWARP_SIZE = 32U;
constexpr uint32_t kLOG2_WARP_SIZE = 5U;
constexpr uint32_t kLANE_ID_MASK = 0x1f;
constexpr uint32_t kFINAL_MASK = 0xffffffff;

template <typename T>
inline __device__ T warpReduceSumFull(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(kFINAL_MASK, val, mask, kWARP_SIZE);
  return val;
}

// Every warp reduces the per-warp partials itself, so every thread returns
// the block's sum.
template <typename T>
inline __device__ T blockReduceSumFull(T val) {
  T* smem = reinterpret_cast<LamportSmem*>(__local_mem())->reduce_smem;
  int lane_id = threadIdx.x & kLANE_ID_MASK;
  int warp_id = threadIdx.x >> kLOG2_WARP_SIZE;
  int warp_num = blockDim.x >> kLOG2_WARP_SIZE;

  val = warpReduceSumFull(val);
  if (lane_id == 0)
    smem[warp_id] = val;
  __syncthreads();

  val = (lane_id < warp_num) ? smem[lane_id] : (T)0.f;
  val = warpReduceSumFull(val);
  return val;
}

template <typename T_IN, typename T_OUT, int LoadsPerThread = 1>
__device__ void rmsNormLamport(LamportArgs* args) {
  static_assert(std::is_same_v<T_IN, T_OUT>, "T_IN and T_OUT must be the same type");
  static int const kELTS_PER_LOAD = sizeof(float4) / sizeof(T_IN);
  T_IN* outputPreNorm = (T_IN*)args->outputPreNorm;
  T_OUT* outputNorm = (T_OUT*)args->outputNorm;
  T_IN* bufferInput = (T_IN*)args->bufferInput;
  T_IN const* gamma = (T_IN const*)args->gamma;
  const float epsilon = args->epsilon;
  T_IN const* residual = (T_IN const*)args->residual;
  const uint32_t dim = args->dim;
  if (threadIdx.x == 0 && threadIdx.y == 0)
    ((unsigned*)args->smid)[(uint32_t)blockIdx.x * (uint32_t)gridDim.y + (uint32_t)blockIdx.y] = __smid();

  uint32_t const token = blockIdx.x;
  uint32_t const blockSize = blockDim.x;
  uint32_t const threadOffset = threadIdx.x;

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  uint32_t numThreads = cluster.num_threads();
  uint32_t clusterSize = cluster.num_blocks();
  uint32_t blockOffset = cluster.block_rank();
  uint32_t const dimPadded = round_up(dim, kELTS_PER_LOAD * numThreads);
  uint32_t const elemsPerThread = dimPadded / numThreads;
  uint32_t const loadStride = blockSize;

  uint8_t* smem = (uint8_t*)__local_mem() + kDynOffset;
  float* sharedVal = reinterpret_cast<LamportSmem*>(__local_mem())->sharedVal;
  float rInput[LoadsPerThread * kELTS_PER_LOAD];
  uint32_t offsets[LoadsPerThread * kELTS_PER_LOAD];

  uint32_t const smemBufferSize = blockSize * elemsPerThread * sizeof(T_IN);
  T_IN* smemInput = (T_IN*)&smem[0];
  T_IN* smemResidual = (T_IN*)&smem[smemBufferSize];
  T_IN* smemGamma = (T_IN*)&smem[2 * smemBufferSize];

  // Multi-GPU, stubbed: the input is the all-reduced buffer itself.
  T_IN* input = bufferInput;

  cudaTriggerProgrammaticLaunchCompletion();
  // Block b of the cluster takes the b-th contiguous chunk of the token's row.
  uint32_t const blockChunkSize = ceil_div(dim, clusterSize * kELTS_PER_LOAD) * kELTS_PER_LOAD;
  uint32_t const blockLoadOffset = token * dim + blockOffset * blockChunkSize;

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    offsets[i] = blockLoadOffset + threadLoadOffset;
  }

  // Two cp.async groups: the residual chunk, then the gamma chunk.
#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim)
      copyF4(&smemResidual[threadLoadOffset], &residual[blockLoadOffset + threadLoadOffset]);
  }
  __pipeline_commit();
#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim)
      copyF4(&smemGamma[threadLoadOffset], &gamma[blockOffset * blockChunkSize + threadLoadOffset]);
  }
  __pipeline_commit();

  // flag.ctaArrive(): the cluster barrier; its global arrival counter add is
  // multi-GPU and stubbed. The barrier also orders every block's start before
  // the remote stores into sharedVal below.
  cluster.sync();

  // Multi-GPU, stubbed: one pass over the input instead of polling for -0.0.
#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      float4* dst4 = reinterpret_cast<float4*>(&smemInput[threadLoadOffset]);
      float4 const* src4 = reinterpret_cast<float4 const*>(&input[offsets[i]]);
      float4 value = loadPackedVolatile(src4);
      *dst4 = value;
    }
  }

  __pipeline_wait_prior(1);
  __syncthreads();

  float threadSum = 0.f;
#pragma unroll
  for (int i = 0; i < LoadsPerThread; i++) {
    int threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      PackedVec<float4, T_IN> inp, res;
      inp.packed = loadPacked<float4>(&smemInput[threadLoadOffset]);
      res.packed = loadPacked<float4>(&smemResidual[threadLoadOffset]);

      PackedVec<float4, T_IN> inp_plus_res = inp + res;
#pragma unroll
      for (int j = 0; j < kELTS_PER_LOAD; j++) {
        rInput[i * kELTS_PER_LOAD + j] = toFloat<T_IN>(inp_plus_res.elements[j]);
        threadSum += toFloat<T_IN>(inp_plus_res.elements[j] * inp_plus_res.elements[j]);
      }

      *reinterpret_cast<float4*>(&outputPreNorm[blockLoadOffset + threadLoadOffset]) = inp_plus_res.packed;
    }
  }

  __pipeline_wait_prior(0);

  float blockSum = blockReduceSumFull<float>(threadSum);

  // All-reduce of the block sums: thread t writes this block's sum into
  // sharedVal[rank] of block t, then the cluster barrier, then a local sum.
  float fullSum = blockSum;
  int const numBlocks = cluster.num_blocks();
  if (numBlocks > 1) {
    fullSum = 0.F;
    int const blockRank = cluster.block_rank();
    if ((int)threadIdx.x < numBlocks)
      cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
    cluster.barrier_wait(cluster.barrier_arrive());
    for (int i = 0; i < numBlocks; ++i)
      fullSum += sharedVal[i];
  }

  float rcpRms = rsqrtf(fullSum / dim + epsilon);

#pragma unroll
  for (int i = 0; i < LoadsPerThread; i++) {
    PackedVec<float4, T_OUT> r_out;
    uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      PackedVec<float4, T_IN> g;
      g.packed = loadPacked<float4>(&smemGamma[threadLoadOffset]);
#pragma unroll
      for (uint32_t j = 0; j < kELTS_PER_LOAD; j++)
        r_out.elements[j] = fromFloat<T_OUT>(toFloat<T_IN>(g.elements[j]) * rInput[i * kELTS_PER_LOAD + j] * rcpRms);
      *reinterpret_cast<float4*>(&outputNorm[blockLoadOffset + threadLoadOffset]) = r_out.packed;
    }
  }

  cudaGridDependencySynchronize();
  // Multi-GPU, stubbed: flag.waitAndUpdate() (Lamport buffer rotation).
}

} // namespace lamport

namespace lamport_kernels {
VX_CUDA_KERNEL_EXPORT(rmsNormLamport, lamport::rmsNormLamport<__nv_bfloat16, __nv_bfloat16, 1>)
} // namespace lamport_kernels
