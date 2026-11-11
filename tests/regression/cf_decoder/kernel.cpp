// ClusterFusion's fused decode-step kernels, re-implemented over the CUDA
// porting layer from ClusterFusion 8755446 (https://github.com/xinhao-luo/ClusterFusion,
// NeurIPS 2025) with the H100-run patch clusterfusion__residual_out_c4_fix
// applied (h100-workloads, workloads/1a_collectives/clusterfusion_llama2_7b_decoder/patches):
//
//   LlamaDecoderLayerKernel     include/H100/llama/kernel_sglang.cuh (+ config.h,
//                               llama_kernel_sglang_dispatch.cu), the fused_c4 variant
//   DeepSeekDecoderLayerKernel  include/H100/deepseek/kernel.cuh (+ config.h,
//                               deepseek_kernel_dispatch.cu)
//   cluster_reduce              include/dsm.cuh, in cf_dsm.h
//
// Both keep the H100 build's 128-thread blocks, clusters of 4, grid of
// heads x 4, TMA boxes, shared-memory layout, double buffering, rounds,
// barriers and fences, including the patch's additions: block.sync() before
// the TMA refills of the Llama Q/K/V GEMV and flash-decoding loops and of the
// MLA W_q, W_q_pe, W_kv, W_k_pe, latent-cache and W_uv loops, before the two
// MLA local_qkv rewrites, the residual_output tensor, and the cluster_reduce
// hardening. The loops the patch leaves alone (Llama W_o, MLA W_uk and W_o)
// have no refill barrier here either.
//
// Reduced problem (make variables, Makefile): Llama HEAD_NUM 4 (2), so
// HIDDEN_DIM 512 (256), sequence length a runtime argument as upstream; MLA
// HEAD_NUM 2 (1), HIDDEN_DIM 1024, SEQ_LEN 128. HEAD_DIM 128, the MLA ranks
// (nope 128, rope 64, KV LoRA 512), every tile size and FFN_DIM / HEAD_NUM
// (384, the Llama weight buffer) are the H100's.
//
// Upstream behaviour kept as the H100 ran it:
// - both: the flash-decoding merge restarts its running max at 0, and the
//   cluster sums are shift-by-i rounds that are exact only for clusters of 4;
// - Llama: the new token is folded in by block 0 only, through tile row 1 of
//   warp 0; rows past the sequence end come from the TMA's zero fill and are
//   left out of the softmax sum but not of the running max;
// - MLA: the running max starts at 0; the score omits the rope part (q_pe and
//   k_pe are computed, roped and dropped; the k_pe rope reads its partner
//   from c_kv[32..63], and the result is discarded); the new latent replaces cache
//   row SEQ_LEN - 1 in registers and the cache is not appended; the last KV
//   tile writes `reduction` without the block barrier its loop body has.
//
// Departures:
// - no __shared__: the static and dynamic shared variables sit at fixed
//   local-memory offsets (common.h), statics first in declaration order, and
//   the launch requests statics + dynamic bytes;
// - the kernel parameters are one argument block (common.h), tensor maps
//   first; each block also records its core id (placement check);
// - Llama: only the NEOX RoPE branch the H100 build selects; the KV cache
//   pointers the kernel never dereferences are not passed;
// - MLA: the W_o map's row count is HEAD_NUM x 128 (the H100 build's
//   HIDDEN_DIM, which equals it there);
// - ex2.approx.ftz is __vx_ex2_approx; std::move of an arrival token is a
//   cast to an rvalue reference; the TMA issue, the GEMV tile loops, the
//   score and softmax steps and the scalar cluster rounds are shared helpers
//   with the original's order of operations; the tile loops and the
//   flash-decoding merges are noinline calls (f16 arithmetic is integer code
//   here, and inlined it made each kernel several times the instruction cache);
// - built with -fno-strict-aliasing for the uint4 / uint64_t moves of fp16
//   data, and with -fconvergent-functions (the barrier, shuffle and vote
//   intrinsics are inline asm; CUDA compiles device code this way).

#include <vx_cuda.h>
#include "common.h"
#include "cf_dsm.h"

namespace {

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

// The tile loops are calls: every f16 operation is integer code, and inlined
// they would make each kernel several times the instruction cache.
#define CF_NOINLINE __attribute__((noinline))

__device__ __forceinline__ float ptx_exp2(float x) { return __vx_ex2_approx(x); }

__device__ __forceinline__ barrier::arrival_token&& moved(barrier::arrival_token& t) {
  return static_cast<barrier::arrival_token&&>(t);
}

template <typename T>
__device__ __forceinline__ T* lmem_at(uint32_t off) {
  return reinterpret_cast<T*>((char*)__local_mem() + off);
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// Thread 0 issues a 2D TMA load and arrives with its byte count; every other
// thread arrives.
__device__ __forceinline__ barrier::arrival_token tma_issue(void* dst, const CUtensorMap* map, uint32_t c0,
                                                            uint32_t c1, barrier& bar, uint32_t bytes) {
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(dst, map, c0, c1, bar);
    return cuda::device::barrier_arrive_tx(bar, 1, bytes);
  }
  return bar.arrive();
}

///////////////////////////////////////////////////////////////////////////////
// Llama-2 attention block

namespace llama {

using namespace cf_llama;

__device__ __forceinline__ void mac8(float& acc, const half* a, const half* b) {
#pragma unroll
  for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
    acc += __half2float(a[d]) * __half2float(b[d]);
  }
}

// One of the q / k / v projections: W_qkv rows row0..row0+127 (box rows)
// against this block's DIM_PER_BLOCK slice of the normalised input, starting
// at column k0. Tiles of 64 columns alternate between the two weight buffers.
__device__ CF_NOINLINE float qkv_gemv(const CUtensorMap* map, half* weight, const half* input_shmem,
                                          barrier* bar, uint32_t k0, uint32_t row0, uint32_t input_idx,
                                          uint32_t weight_idx) {
  cg::thread_block block = cg::this_thread_block();
  constexpr uint32_t NT = DIM_PER_BLOCK / TMA_LOAD_ONCE;
  half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
  barrier::arrival_token token[2];
  float tmp = 0.0f;

  token[0] = tma_issue(&weight[0], map, k0, row0, bar[0], TMA_LOAD_ONCE_SIZE);
  for (uint32_t id = 1; id < NT; id++) {
    // Buffer id % 2 last held tile id - 2: wait until no warp reads it.
    block.sync();
    token[id % 2] = tma_issue(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], map, k0 + id * TMA_LOAD_ONCE, row0,
                              bar[id % 2], TMA_LOAD_ONCE_SIZE);
    bar[(id - 1) % 2].wait(moved(token[(id - 1) % 2]));
    for (uint32_t i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) {
      *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
      *(uint4*)(&reg_weight[0]) =
          *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + i]);
      mac8(tmp, reg_input, reg_weight);
    }
  }
  bar[(NT - 1) % 2].wait(moved(token[(NT - 1) % 2]));
  for (uint32_t i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) {
    *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (NT - 1) * TMA_LOAD_ONCE + i]);
    *(uint4*)(&reg_weight[0]) =
        *(uint4*)(&weight[((NT - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + i]);
    mac8(tmp, reg_input, reg_weight);
  }
  return tmp;
}

// The row's partial sums meet in its first lane, which stores the projection.
__device__ __forceinline__ void qkv_store(float tmp, half* out) {
  const uint32_t lane_id = threadIdx.x % WARP_SIZE, warp_id = threadIdx.x / WARP_SIZE;
#pragma unroll
  for (uint32_t mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
    tmp += __shfl_down_sync(0xffffffff, tmp, mask);
  }
  if (lane_id % NUM_THREAD_PER_ROW == 0) {
    out[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
  }
}

// The scores of one 32-row K tile against q, scaled to log2 units, and the
// running maximum.
__device__ CF_NOINLINE void kv_scores(float* qk, float& local_max, const half* tile, const half* reg_input,
                                          uint32_t weight_idx_2, uint32_t input_idx_2, float softmax_scale) {
  half __align__(16) reg_weight[NUM_PER_THREAD];
#pragma unroll
  for (uint32_t j = 0; j < DEC_TILE; j++) {
    *(uint4*)(&reg_weight[0]) = *(uint4*)(&tile[(weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
    qk[j] = 0.0f;
    mac8(qk[j], reg_input, reg_weight);
#pragma unroll
    for (uint32_t mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
      qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
    }
    qk[j] = qk[j] * softmax_scale;
    local_max = max(local_max, qk[j]);
  }
}

// Rescale by the new maximum and exponentiate the rows inside the sequence.
__device__ CF_NOINLINE void kv_softmax(float* qk, float& local_sum, float* reg_reduce, float pre_max,
                                       float local_max, uint32_t row0, uint32_t seq_len) {
  const uint32_t lane_id = threadIdx.x % WARP_SIZE, warp_id = threadIdx.x / WARP_SIZE;
  const uint32_t weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
  const float scale = ptx_exp2(pre_max - local_max);
  local_sum *= scale;
  // Rows past the sequence were zero-filled by the TMA.
#pragma unroll
  for (uint32_t j = 0; j < DEC_TILE; j++) {
    if ((row0 + weight_idx_2 + j) < seq_len) {
      qk[j] = ptx_exp2(qk[j] - local_max);
      local_sum += qk[j];
    }
  }
#pragma unroll
  for (uint32_t j = 0; j < NUM_PER_THREAD; j++) {
    reg_reduce[j] = reg_reduce[j] * scale;
  }
}

__device__ CF_NOINLINE void kv_accumulate(float* reg_reduce, const float* qk, const half* tile,
                                              uint32_t weight_idx_2, uint32_t input_idx_2) {
  half __align__(16) reg_weight[NUM_PER_THREAD];
  for (uint32_t j = 0; j < DEC_TILE; j++) {
    *(uint4*)(&reg_weight[0]) = *(uint4*)(&tile[(weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
#pragma unroll
    for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
      reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
    }
  }
}

// One 64-row W_o tile against the head's attention output; each pair of lanes
// owns an output row and adds it into the global output over the heads.
__device__ CF_NOINLINE void wo_tile(half* output, const half* attn, const half* tile, uint32_t out0,
                                        uint32_t input_idx_3, uint32_t weight_idx_3) {
  const uint32_t lane_id = threadIdx.x % WARP_SIZE;
  half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
  float tmp = 0.0f;
  for (uint32_t j = 0; j < HEAD_DIM; j += NUM_PER_ROW_3) {
    *(uint4*)(&reg_input[0]) = *(uint4*)(&attn[input_idx_3 + j]);
    *(uint4*)(&reg_weight[0]) = *(uint4*)(&tile[weight_idx_3 * HEAD_DIM + input_idx_3 + j]);
    mac8(tmp, reg_input, reg_weight);
  }
#pragma unroll
  for (uint32_t mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
    tmp += __shfl_down_sync(0xffffffff, tmp, mask);
  }
  if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
    atomicAdd(&output[out0 + weight_idx_3], __float2half(tmp));
  }
}

// Merge the block's DIM_BLOCK_REDUCE / 2 tile rows into every thread's 8
// output dimensions; the running maximum restarts at 0.
__device__ CF_NOINLINE void merge_rows(float* reg_reduce, float& local_sum, float& local_max, const half* weight,
                                       const float* reduction, uint32_t tile_col) {
  half __align__(16) reg_input[NUM_PER_THREAD];
  float pre_max, scale;
  for (uint32_t i = 0; i < NUM_PER_THREAD; i++)
    reg_reduce[i] = 0.0f;
  local_sum = 0.0, local_max = 0.0;
#pragma unroll
  for (uint32_t j = 0; j < DIM_BLOCK_REDUCE / 2; j++) {
    *(uint4*)(&reg_input[0]) = *(const uint4*)(&weight[j * HEAD_DIM + tile_col * NUM_PER_THREAD]);
    float m = reduction[j * 2], s = reduction[j * 2 + 1];
    pre_max = local_max;
    local_max = max(m, local_max);
    scale = ptx_exp2(m - local_max);
    s *= scale;
    local_sum = local_sum * ptx_exp2(pre_max - local_max) + s;
#pragma unroll
    for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
      reg_reduce[d] = reg_reduce[d] * ptx_exp2(pre_max - local_max) + __half2float(reg_input[d]) * scale;
    }
  }
}

} // namespace llama

} // namespace

__global__ void __cluster_dims__(cf_llama::CLUSTER_SIZE, 1, 1) LlamaDecoderLayerKernel(llama_arg_t* __UNIFORM__ arg) {
  VX_CUDA_KERNEL_ENTRY(LlamaDecoderLayerKernel);
  using namespace cf_llama;
  using namespace llama;

  half* output = (half*)arg->output;
  half* k_output = (half*)arg->k_output;
  half* v_output = (half*)arg->v_output;
  const half* input = (const half*)arg->input;
  const half* residual = (const half*)arg->residual;
  half* residual_output = (half*)arg->residual_output;
  const half* w_rms_input = (const half*)arg->w_rms_input;
  const float eps = arg->eps;
  const float* cos = (const float*)arg->cos;
  const float* sin = (const float*)arg->sin;
  const CUtensorMap* tensor_map = &arg->tensor_map;
  const CUtensorMap* tensor_map_k_cache = &arg->tensor_map_k_cache;
  const CUtensorMap* tensor_map_v_cache = &arg->tensor_map_v_cache;
  const CUtensorMap* tensor_map_weight_o = &arg->tensor_map_weight_o;
  const uint32_t SEQ_LEN = arg->seq_len;
  const uint32_t KV_DIM_PER_BLOCK = arg->kv_dim_per_block;

  cg::grid_group grid = cg::this_grid();
  cg::cluster_group cluster = cg::this_cluster();
  cg::thread_block block = cg::this_thread_block();
  const uint32_t head_id = grid.cluster_rank();
  const uint32_t cluster_block_id = cluster.block_rank();
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid % WARP_SIZE;
  const uint32_t warp_id = tid / WARP_SIZE;
  const uint32_t tile_row = tid / NUM_THREAD_PER_ROW_2;
  const uint32_t tile_col = tid % NUM_THREAD_PER_ROW_2;

  if (tid == 0)
    ((uint32_t*)arg->smid)[blockIdx.x] = __smid();

  // Shared memory: the statics, then the dynamic buffer aligned to 128.
  float& cluster_local_sum = *lmem_at<float>(LM_SUM);
  float& cluster_local_max = *lmem_at<float>(LM_MAX);
  barrier* bar = lmem_at<barrier>(LM_BAR);
  uint64_t* reduce_bar = lmem_at<uint64_t>(LM_RBAR);
  half* weight = reinterpret_cast<half*>(((uintptr_t)lmem_at<uint8_t>(LM_DYN) + 127) & ~(uintptr_t)127);
  half* local_qkv = weight + 2 * TMA_LOAD_ONCE * MAX_SMEM_DIM;
  half* input_shmem = local_qkv + 3 * HEAD_DIM;
  float* reduction = reinterpret_cast<float*>(input_shmem + DIM_PER_BLOCK);

  float local_sum = 0.0, rms_rcp = 0.0, tmp = 0.0, local_max = -CUDART_INF_F, pre_max = -CUDART_INF_F, scale = 0.0,
        softmax_scale = __frsqrt_rn(HEAD_DIM) * 1.44269504088896340736f;
  half __align__(16) reg_input[NUM_PER_THREAD], reg_residual[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
  float reg_reduce[NUM_PER_THREAD];
  float* dst_shmem = nullptr;
  float q_rope, q_rope_1, k_rope, k_rope_1, cos_reg, sin_reg;
  uint32_t size;
  uint32_t src_addr, dst_addr, neighbor_dst_bar = 0;
  float __align__(16) qk[DEC_TILE];

  barrier::arrival_token token[4];
  const uint32_t bar_ptr = smem_addr(reduce_bar);
  if (tid == 0) {
    for (uint32_t b = 0; b < 4; b++) {
      init(&bar[b], blockDim.x);
      cde::fence_proxy_async_shared_cta();
    }
  }
  block.sync();

  const uint32_t input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
  const uint32_t weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
  const uint32_t input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
  const uint32_t weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
  const uint32_t input_idx_3 = (lane_id % NUM_THREAD_PER_ROW_3) * NUM_PER_THREAD;
  const uint32_t weight_idx_3 = warp_id * NUM_ROW_PER_WARP_3 + lane_id / NUM_THREAD_PER_ROW_3;
  const uint32_t cluster_block_st_id = cluster_block_id * DIM_PER_BLOCK;
  const uint32_t cluster_head_idx = head_id * HEAD_DIM;

  // Residual add and RMSNorm over this block's slice; x = half(input + residual)
  // also goes to residual_output.
  for (uint32_t d = tid * 8; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 8) {
    *(uint4*)(&reg_input[0]) = *(const uint4*)(&input[cluster_block_st_id + d]);
    *(uint4*)(&reg_residual[0]) = *(const uint4*)(&residual[cluster_block_st_id + d]);
    for (int di = 0; di < 8; di++) {
      reg_input[di] = __float2half(__half2float(reg_input[di]) + __half2float(reg_residual[di]));
      local_sum += __half2float(reg_input[di]) * __half2float(reg_input[di]);
    }
    *(uint4*)(&residual_output[cluster_block_st_id + d]) = *(uint4*)(&reg_input[0]);
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
  }
  if (lane_id == 0) {
    reduction[warp_id] = local_sum;
  }
  block.sync();
  if (tid < NUM_WARPS)
    local_sum = reduction[tid];
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
  }
  if (tid == 0)
    cluster_local_sum = local_sum;
  cluster.sync();
  cf::cluster_sum_rounds(cluster_local_sum, local_sum, dst_shmem, cluster_block_id);
  rms_rcp = __frsqrt_rn(cluster_local_sum / HIDDEN_DIM + eps);
  // reg_input still holds this thread's x (one slot per thread at most).
  for (uint32_t d = tid * 8; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 8) {
    *(uint4*)(&reg_weight[0]) = *(const uint4*)(&w_rms_input[cluster_block_st_id + d]);
    for (int i = 0; i < 8; i++) {
      reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
    }
    *(uint4*)(&input_shmem[d]) = *(uint4*)(&reg_input[0]);
  }
  block.sync();

  // q, k and v partial projections over the slice (rows of W_qkv: q, then k
  // from HEAD_NUM x HEAD_DIM, then v from twice that).
  tmp = qkv_gemv(tensor_map, weight, input_shmem, bar, cluster_block_st_id, cluster_head_idx, input_idx, weight_idx);
  qkv_store(tmp, local_qkv);
  block.sync();

  tmp = qkv_gemv(tensor_map, weight, input_shmem, bar, cluster_block_st_id, cluster_head_idx + HEAD_NUM * HEAD_DIM,
                 input_idx, weight_idx);
  qkv_store(tmp, local_qkv + HEAD_DIM);
  block.sync();

  tmp = qkv_gemv(tensor_map, weight, input_shmem, bar, cluster_block_st_id,
                 cluster_head_idx + 2 * HEAD_NUM * HEAD_DIM, input_idx, weight_idx);
  qkv_store(tmp, local_qkv + HEAD_DIM * 2);
  block.sync();

  // Sum the partial q, k, v over the cluster.
  size = (HEAD_DIM * 3) * sizeof(half);
  src_addr = smem_addr(local_qkv);
  dst_addr = smem_addr(weight);
  cf::cluster_reduce<CLUSTER_SIZE, cf::Stage::LINEAR>(size, HEAD_DIM, cluster_block_id, src_addr, dst_addr, bar_ptr,
                                                      neighbor_dst_bar, local_qkv, weight);

  // NEOX RoPE on q and k: dimension d pairs with d +- 64.
  q_rope = __half2float(local_qkv[tid]);
  k_rope = __half2float(local_qkv[HEAD_DIM + tid]);
  cos_reg = cos[tid % (HEAD_DIM / 2)];
  sin_reg = sin[tid % (HEAD_DIM / 2)];
  if (tid < HEAD_DIM / 2) {
    q_rope_1 = __half2float(local_qkv[HEAD_DIM / 2 + tid]);
    k_rope_1 = __half2float(local_qkv[HEAD_DIM + HEAD_DIM / 2 + tid]);
  } else {
    q_rope_1 = __half2float(local_qkv[tid - HEAD_DIM / 2]);
    k_rope_1 = __half2float(local_qkv[HEAD_DIM + tid - HEAD_DIM / 2]);
  }
  block.sync();
  if (tid < HEAD_DIM / 2) {
    local_qkv[tid] = __float2half(q_rope * cos_reg - q_rope_1 * sin_reg);
    local_qkv[HEAD_DIM + tid] = __float2half(k_rope * cos_reg - k_rope_1 * sin_reg);
  } else {
    local_qkv[tid] = __float2half(q_rope * cos_reg + q_rope_1 * sin_reg);
    local_qkv[HEAD_DIM + tid] = __float2half(k_rope * cos_reg + k_rope_1 * sin_reg);
  }

  // The new token's k and v go out once per head.
  cluster.sync();
  if (cluster_block_id == 0) {
    k_output[head_id * HEAD_DIM + tid] = local_qkv[HEAD_DIM + tid];
    v_output[head_id * HEAD_DIM + tid] = local_qkv[2 * HEAD_DIM + tid];
  }
  cluster.sync();

  // Flash decoding over this block's KV_DIM_PER_BLOCK cache rows, in tiles of
  // 32 rows: K and V of tile t in buffer t % 2 (K at the start, V after it).
  local_sum = 0.0f;
  for (uint32_t i = 0; i < NUM_PER_THREAD; i++)
    reg_reduce[i] = 0.0f;
  *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2]);
  block.sync();

  const uint32_t kv_row0 = cluster_block_id * KV_DIM_PER_BLOCK;
  const uint32_t kv_tiles = KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN;
  token[0] = tma_issue(&weight[0], tensor_map_k_cache, cluster_head_idx, kv_row0, bar[0], TMA_LOAD_ONCE_SIZE_ATTN);
  token[2] = tma_issue(&weight[TMA_LOAD_ONCE_NUM_ATTN], tensor_map_v_cache, cluster_head_idx, kv_row0, bar[2],
                       TMA_LOAD_ONCE_SIZE_ATTN);
  for (uint32_t id = 1; id < kv_tiles; id++) {
    // Buffers id % 2 last held tile id - 2: wait until no warp reads them.
    block.sync();
    token[id % 2] = tma_issue(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], tensor_map_k_cache, cluster_head_idx,
                              kv_row0 + id * TMA_LOAD_ONCE_ATTN, bar[id % 2], TMA_LOAD_ONCE_SIZE_ATTN);
    bar[(id - 1) % 2].wait(moved(token[(id - 1) % 2]));
    pre_max = local_max;
    kv_scores(qk, local_max, &weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM], reg_input, weight_idx_2, input_idx_2,
              softmax_scale);
    kv_softmax(qk, local_sum, reg_reduce, pre_max, local_max, kv_row0 + (id - 1) * TMA_LOAD_ONCE_ATTN, SEQ_LEN);
    token[2 + id % 2] = tma_issue(&weight[(id % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN], tensor_map_v_cache,
                                  cluster_head_idx, kv_row0 + id * TMA_LOAD_ONCE_ATTN, bar[2 + id % 2],
                                  TMA_LOAD_ONCE_SIZE_ATTN);
    bar[2 + (id - 1) % 2].wait(moved(token[2 + (id - 1) % 2]));
    kv_accumulate(reg_reduce, qk, &weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN],
                  weight_idx_2, input_idx_2);
  }
  if (KV_DIM_PER_BLOCK > TMA_LOAD_ONCE_ATTN) {
    bar[(kv_tiles - 1) % 2].wait(moved(token[(kv_tiles - 1) % 2]));
  } else {
    bar[0].wait(moved(token[0]));
  }
  pre_max = local_max;
  kv_scores(qk, local_max, &weight[((kv_tiles - 1) % 2) * TMA_LOAD_ONCE_NUM], reg_input, weight_idx_2, input_idx_2,
            softmax_scale);
  kv_softmax(qk, local_sum, reg_reduce, pre_max, local_max, kv_row0 + (kv_tiles - 1) * TMA_LOAD_ONCE_ATTN, SEQ_LEN);
  if (KV_DIM_PER_BLOCK > TMA_LOAD_ONCE_ATTN) {
    bar[(kv_tiles - 1) % 2 + 2].wait(moved(token[(kv_tiles - 1) % 2 + 2]));
  } else {
    bar[2].wait(moved(token[2]));
  }
  kv_accumulate(reg_reduce, qk, &weight[((kv_tiles - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN],
                weight_idx_2, input_idx_2);
  block.sync();

  // The new token, which is not in the cache: block 0, tile row 1 of warp 0.
  if (cluster_block_id == 0 && warp_id == 0) {
    if (lane_id / NUM_THREAD_PER_ROW_2 == 1) {
      pre_max = local_max;
      *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[HEAD_DIM + input_idx_2]);
      qk[0] = 0.0f;
      mac8(qk[0], reg_input, reg_weight);
    }
#pragma unroll
    for (uint32_t mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
      qk[0] += __shfl_xor_sync(0xffffffff, qk[0], mask);
    }
    if (lane_id / NUM_THREAD_PER_ROW_2 == 1) {
      qk[0] = qk[0] * softmax_scale;
      local_max = max(local_max, qk[0]);
      scale = ptx_exp2(pre_max - local_max);
      local_sum *= scale;
      qk[0] = ptx_exp2(qk[0] - local_max);
      local_sum += qk[0];
#pragma unroll
      for (uint32_t j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * scale;
      }
      *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_2]);
#pragma unroll
      for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
        reg_reduce[d] = reg_reduce[d] + qk[0] * __half2float(reg_weight[d]);
      }
    }
  }
  block.sync();

  // Merge the block's tile rows: each row's (max, sum) and its partial output.
#pragma unroll
  for (uint32_t i = 0; i < NUM_PER_THREAD; i++) {
    weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
  }
  if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
    reduction[tile_row * 2] = local_max;
    reduction[tile_row * 2 + 1] = local_sum;
  }
  block.sync();
  merge_rows(reg_reduce, local_sum, local_max, weight, reduction, tile_col);
  block.sync();

  // Merge over the cluster: the maximum, then the rescaled sums.
  pre_max = local_max;
  if (tid == 0) {
    cluster_local_max = local_max;
  }
  cluster.sync();
  cf::cluster_max_rounds(cluster_local_max, local_max, dst_shmem, cluster_block_id);
  scale = ptx_exp2(pre_max - cluster_local_max);
  local_sum *= scale;
#pragma unroll
  for (uint32_t j = 0; j < NUM_PER_THREAD; j++) {
    reg_reduce[j] = reg_reduce[j] * scale;
  }
  if (tid == 0) {
    cluster_local_sum = local_sum;
  }
  cluster.sync();
  cf::cluster_sum_rounds(cluster_local_sum, local_sum, dst_shmem, cluster_block_id);
#pragma unroll
  for (uint32_t j = 0; j < NUM_PER_THREAD; j++) {
    reg_reduce[j] = reg_reduce[j] * __frcp_rn(cluster_local_sum);
  }
  if (tid < NUM_THREAD_PER_ROW_2) {
#pragma unroll
    for (uint32_t i = 0; i < NUM_PER_THREAD; i++) {
      local_qkv[2 * HEAD_DIM + tid * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
    }
  }
  block.sync();

  // Sum the attention output over the cluster.
  size = HEAD_DIM * sizeof(half);
  src_addr = smem_addr(&local_qkv[2 * HEAD_DIM]);
  dst_addr = smem_addr(weight);
  cf::cluster_reduce<CLUSTER_SIZE, cf::Stage::ATTN>(size, HEAD_DIM, cluster_block_id, src_addr, dst_addr, bar_ptr,
                                                    neighbor_dst_bar, &local_qkv[2 * HEAD_DIM], weight);

  // Output projection of this block's DIM_PER_BLOCK rows of W_o, summed over
  // the heads with f16 atomics into the zeroed output.
  const uint32_t wo_tiles = DIM_PER_BLOCK / TMA_LOAD_ONCE;
  token[0] = tma_issue(&weight[0], tensor_map_weight_o, cluster_head_idx, cluster_block_st_id, bar[0],
                       TMA_LOAD_ONCE_SIZE);
  for (uint32_t id = 1; id < wo_tiles; id++) {
    token[id % 2] = tma_issue(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], tensor_map_weight_o, cluster_head_idx,
                              cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2], TMA_LOAD_ONCE_SIZE);
    bar[(id - 1) % 2].wait(moved(token[(id - 1) % 2]));
    wo_tile(output, &local_qkv[2 * HEAD_DIM], &weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM],
            cluster_block_st_id + (id - 1) * TMA_LOAD_ONCE, input_idx_3, weight_idx_3);
    block.sync();
  }
  bar[(wo_tiles - 1) % 2].wait(moved(token[(wo_tiles - 1) % 2]));
  wo_tile(output, &local_qkv[2 * HEAD_DIM], &weight[(wo_tiles - 1) % 2 * TMA_LOAD_ONCE_NUM],
          cluster_block_st_id + (wo_tiles - 1) * TMA_LOAD_ONCE, input_idx_3, weight_idx_3);
}

///////////////////////////////////////////////////////////////////////////////
// DeepSeek-V2-Lite MLA

namespace {
namespace mla {

using namespace cf_mla;

// A double-buffered GEMV over NT TMA tiles of STEP input rows each; tile t of
// the input vector starts at in[in_idx + t * STEP], tile row r of the weight
// at weight[buffer + r * ROWLEN]. The map's coordinates are (c0, c1 + t * STEP).
// SYNC: the block barrier before each refill. The last wait names bar[1], as
// the original's does (NT is even).
template <uint32_t NT, uint32_t STEP, uint32_t ROWLEN, uint32_t IN_STEP, bool SYNC>
__device__ CF_NOINLINE float gemv(const CUtensorMap* map, uint32_t c0, uint32_t c1, half* weight, barrier* bar,
                                      const half* in, uint32_t in_idx, uint32_t w_idx) {
  static_assert(NT % 2 == 0, "the last tile is in buffer 1");
  cg::thread_block block = cg::this_thread_block();
  half __align__(16) reg_input[NUM_PER_THREAD];
  barrier::arrival_token token[2];
  float tmp = 0.0f;

  token[0] = tma_issue(&weight[0], map, c0, c1, bar[0], TMA_LOAD_ONCE_SIZE);
  for (uint32_t id = 1; id < NT; id++) {
    if constexpr (SYNC) {
      // Every warp must be done with the buffer this refill overwrites.
      block.sync();
    }
    token[id % 2] = tma_issue(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], map, c0, c1 + id * STEP, bar[id % 2],
                              TMA_LOAD_ONCE_SIZE);
    bar[(id - 1) % 2].wait(moved(token[(id - 1) % 2]));
    for (uint32_t i = 0; i < STEP; i += IN_STEP) {
      *(uint4*)(&reg_input[0]) = *(const uint4*)(&in[in_idx + (id - 1) * STEP + i]);
#pragma unroll
      for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
        tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (in_idx + i + d) * ROWLEN + w_idx]));
      }
    }
  }
  bar[1].wait(moved(token[1]));
  for (uint32_t i = 0; i < STEP; i += IN_STEP) {
    *(uint4*)(&reg_input[0]) = *(const uint4*)(&in[in_idx + (NT - 1) * STEP + i]);
#pragma unroll
    for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
      tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (in_idx + i + d) * ROWLEN + w_idx]));
    }
  }
  return tmp;
}

// The scores of one latent-cache tile row block: each warp holds half of the
// 512 latent columns (warp % 2) for its 8 rows (tile_row); the partner warp's
// half arrives through `reduction`. SYNC_FIRST: the block barrier before the
// write, which the last tile's loop does not have.
template <bool SYNC_FIRST>
__device__ __forceinline__ float latent_score(const half* reg_input, const half* reg_weight, float* reduction,
                                              uint32_t tile_row, float softmax_scale) {
  cg::thread_block block = cg::this_thread_block();
  const uint32_t lane_id = threadIdx.x % WARP_SIZE, warp_id = threadIdx.x / WARP_SIZE;
  float qk = 0.0f;
#pragma unroll
  for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
    qk += __half2float(__hmul(reg_input[d], reg_weight[d]));
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    qk += __shfl_xor_sync(0xffffffff, qk, mask);
  }
  if constexpr (SYNC_FIRST)
    block.sync();
  if (lane_id == 0)
    reduction[warp_id] = qk;
  block.sync();
  if (tile_row == 1)
    qk += reduction[2 + (warp_id + 1) % 2];
  else
    qk += reduction[(warp_id + 1) % 2];
  return qk * softmax_scale;
}

// One 16-row latent-cache tile (both column halves in `tile`): scores, online
// softmax update, and the f16 accumulation of the weighted rows. LAST: the
// final tile, whose last row in the last block is the new latent, and whose
// score loop lacks the leading block barrier.
template <bool LAST>
__device__ CF_NOINLINE void latent_tile(const half* tile_base, const half* new_latent, const half* reg_input,
                                        half* reg_reduce, float* reduction, float& local_max, float& local_sum,
                                        uint32_t cluster_block_id, float softmax_scale) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid % WARP_SIZE, warp_id = tid / WARP_SIZE;
  const uint32_t tile_row = tid / NUM_THREAD_PER_ROW_3, tile_col = tid % NUM_THREAD_PER_ROW_3;
  const uint32_t input_idx_3 = tile_col * NUM_PER_THREAD, weight_idx_3 = tile_row * DEC_TILE;
  const half* tile = tile_base + (warp_id % 2) * TMA_LOAD_ONCE_NUM_ATTN / 2;
  half __align__(16) reg_weight[NUM_PER_THREAD];
  float __align__(16) qk[DEC_TILE];
  const float pre_max = local_max;
#pragma unroll
  for (uint32_t j = 0; j < DEC_TILE; j++) {
    if (LAST && cluster_block_id == CLUSTER_SIZE - 1 && tile_row == 1 && j == DEC_TILE - 1)
      *(uint4*)(&reg_weight[0]) = *(const uint4*)(&new_latent[input_idx_3]);
    else
      *(uint4*)(&reg_weight[0]) = *(const uint4*)(&tile[(weight_idx_3 + j) * KV_LORA_RANK / 2 + lane_id * NUM_PER_THREAD]);
    qk[j] = latent_score<!LAST>(reg_input, reg_weight, reduction, tile_row, softmax_scale);
    local_max = fmaxf(local_max, qk[j]);
  }
  const float scale = __expf(pre_max - local_max);
  local_sum *= scale;
#pragma unroll
  for (uint32_t j = 0; j < DEC_TILE; j++) {
    qk[j] = __expf(qk[j] - local_max);
    local_sum += qk[j];
  }
#pragma unroll
  for (uint32_t j = 0; j < NUM_PER_THREAD; j++) {
    reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
  }
  for (uint32_t j = 0; j < DEC_TILE; j++) {
    if (LAST && cluster_block_id == CLUSTER_SIZE - 1 && tile_row == 1 && j == DEC_TILE - 1)
      *(uint4*)(&reg_weight[0]) = *(const uint4*)(&new_latent[input_idx_3]);
    else
      *(uint4*)(&reg_weight[0]) = *(const uint4*)(&tile[(weight_idx_3 + j) * KV_LORA_RANK / 2 + lane_id * NUM_PER_THREAD]);
#pragma unroll
    for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
      reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
    }
  }
}

// Merge the block's two tile rows into every thread's 8 latent dimensions; the
// running maximum restarts at 0.
__device__ CF_NOINLINE void merge_rows(half* reg_reduce, float& local_sum, float& local_max, const half* weight,
                                       const float* reduction, uint32_t tile_col) {
  half __align__(16) reg_input[NUM_PER_THREAD];
  float pre_max, scale;
  for (uint32_t i = 0; i < NUM_PER_THREAD; i++)
    reg_reduce[i] = __float2half(0.0f);
  local_sum = 0.0, local_max = 0.0;
#pragma unroll
  for (uint32_t j = 0; j < NUM_ROW_PER_BLOCK; j++) {
    *(uint4*)(&reg_input[0]) = *(const uint4*)(&weight[j * MLA_HEAD_DIM + tile_col * NUM_PER_THREAD]);
    float m = reduction[j * 2], s = reduction[j * 2 + 1];
    pre_max = local_max;
    local_max = fmaxf(m, local_max);
    scale = __expf(m - local_max);
    s *= scale;
    local_sum = local_sum * __expf(pre_max - local_max) + s;
#pragma unroll
    for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
      reg_reduce[d] = __hadd(__hmul(reg_reduce[d], __float2half(__expf(pre_max - local_max))),
                             __hmul(reg_input[d], __float2half(scale)));
    }
  }
}

} // namespace mla
} // namespace

__global__ void __cluster_dims__(cf_mla::CLUSTER_SIZE, 1, 1) DeepSeekDecoderLayerKernel(mla_arg_t* __UNIFORM__ arg) {
  VX_CUDA_KERNEL_ENTRY(DeepSeekDecoderLayerKernel);
  using namespace cf_mla;
  using namespace mla;

  half* output = (half*)arg->output;
  const half* input = (const half*)arg->input;
  const half* w_rms_input = (const half*)arg->w_rms_input;
  const half* w_rms_ckv = (const half*)arg->w_rms_ckv;
  const float* cos = (const float*)arg->cos;
  const float* sin = (const float*)arg->sin;
  const CUtensorMap* tensor_map_weight_q = &arg->tensor_map_weight_q;
  const CUtensorMap* tensor_map_weight_q_pe = &arg->tensor_map_weight_q_pe;
  const CUtensorMap* tensor_map_weight_uk = &arg->tensor_map_weight_uk;
  const CUtensorMap* tensor_map_weight_kv = &arg->tensor_map_weight_kv;
  const CUtensorMap* tensor_map_weight_k_pe = &arg->tensor_map_weight_k_pe;
  const CUtensorMap* tensor_map_kv_cache = &arg->tensor_map_kv_cache;
  const CUtensorMap* tensor_map_kv_cache_ = &arg->tensor_map_kv_cache_;
  const CUtensorMap* tensor_map_weight_uv = &arg->tensor_map_weight_uv;
  const CUtensorMap* tensor_map_weight_o = &arg->tensor_map_weight_o;

  cg::grid_group grid = cg::this_grid();
  cg::cluster_group cluster = cg::this_cluster();
  cg::thread_block block = cg::this_thread_block();
  const uint32_t head_id = grid.cluster_rank();
  const uint32_t cluster_block_id = cluster.block_rank();
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid % WARP_SIZE;
  const uint32_t warp_id = tid / WARP_SIZE;
  const uint32_t tile_row = tid / NUM_THREAD_PER_ROW_3;
  const uint32_t tile_col = tid % NUM_THREAD_PER_ROW_3;

  if (tid == 0)
    ((uint32_t*)arg->smid)[blockIdx.x] = __smid();

  // Shared memory, in the original's declaration order.
  half* input_shmem = lmem_at<half>(LM_INPUT);
  float* reduction = lmem_at<float>(LM_REDUCTION);
  float& cluster_local_sum = *lmem_at<float>(LM_SUM);
  float& cluster_local_max = *lmem_at<float>(LM_MAX);
  half* weight = lmem_at<half>(LM_WEIGHT);
  half* local_qkv = lmem_at<half>(LM_QKV);
  barrier* bar = lmem_at<barrier>(LM_BAR);
  uint64_t* reduce_bar = lmem_at<uint64_t>(LM_RBAR);

  float local_sum = 0.0, eps = 1e-6, rms_rcp = 0.0, tmp = 0.0, local_max = 0.0, pre_max = 0.0, scale = 0.0,
        softmax_scale = __frsqrt_rn(HEAD_DIM);
  half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD], reg_reduce[NUM_PER_THREAD];
  float* dst_shmem = nullptr;
  half2 q_rope, q_rope_1, k_rope, k_rope_1;
  float2 cos_reg, sin_reg;
  uint32_t size;
  uint32_t src_addr, dst_addr, neighbor_dst_bar = 0;

  barrier::arrival_token token[2];
  const uint32_t bar_ptr = smem_addr(reduce_bar);
  if (tid == 0) {
    init(&bar[0], blockDim.x);
    cde::fence_proxy_async_shared_cta();
    init(&bar[1], blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  block.sync();

  const uint32_t input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
  const uint32_t weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
  const uint32_t input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
  const uint32_t weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2;
  const uint32_t input_idx_3 = tile_col * NUM_PER_THREAD;
  const uint32_t cluster_head_idx = head_id * NOPE_HEAD_DIM;
  const uint32_t cluster_head_pe_idx = head_id * ROPE_HEAD_DIM;
  const uint32_t cluster_head_uk_idx = head_id * KV_LORA_RANK + cluster_block_id * 128;
  const uint32_t cluster_block_st_idx = cluster_block_id * DIM_PER_BLOCK;
  const uint32_t cluster_block_st_uv_idx = cluster_block_id * (KV_LORA_RANK / CLUSTER_SIZE);

  // RMSNorm of the input over the cluster.
  for (uint32_t d = tid * 4; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 4) {
    *(uint64_t*)(&reg_input[0]) = *(const uint64_t*)(&input[cluster_block_st_idx + d]);
    for (int di = 0; di < 4; di++)
      local_sum += __half2float(reg_input[di]) * __half2float(reg_input[di]);
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
  }
  if (lane_id == 0) {
    reduction[warp_id] = local_sum;
  }
  block.sync();
  if (tid < NUM_WARPS)
    local_sum = reduction[tid];
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
  }
  block.sync();
  if (tid == 0)
    cluster_local_sum = local_sum;
  cluster.sync();
  cf::cluster_sum_rounds(cluster_local_sum, local_sum, dst_shmem, cluster_block_id);
  rms_rcp = __frsqrt_rn(cluster_local_sum / HIDDEN_DIM + eps);
  for (uint32_t d = tid * 4; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 4) {
    *(uint64_t*)(&reg_weight[0]) = *(const uint64_t*)(&w_rms_input[cluster_block_st_idx + d]);
    for (int i = 0; i < 4; i++) {
      reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
    }
    *(uint64_t*)(&input_shmem[d]) = *(uint64_t*)(&reg_input[0]);
  }
  block.sync();

  // Partial projections over this block's slice of x_hat: q_nope (128) at
  // local_qkv[0], q_pe (64) at [512], c_kv (512) at [576], k_pe (64) at [1088].
  tmp = gemv<DIM_PER_BLOCK / TMA_LOAD_ONCE, TMA_LOAD_ONCE, NOPE_HEAD_DIM, NUM_PER_ROW, true>(
      tensor_map_weight_q, cluster_head_idx, cluster_block_st_idx, weight, bar, input_shmem, input_idx, weight_idx);
  if (lane_id % NUM_THREAD_PER_ROW == 0) {
    local_qkv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
  }
  block.sync();

  tmp = gemv<DIM_PER_BLOCK / 128, 128, ROPE_HEAD_DIM, NUM_PER_ROW_2, true>(
      tensor_map_weight_q_pe, cluster_head_pe_idx, cluster_block_st_idx, weight, bar, input_shmem, input_idx_2,
      weight_idx_2);
#pragma unroll
  for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
    tmp += __shfl_down_sync(0xffffffff, tmp, mask);
  }
  if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
    local_qkv[KV_LORA_RANK + warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2] = __float2half(tmp);
  }
  block.sync();

  for (uint32_t s = 0; s < KV_LORA_RANK / 128; s++) {
    // Every warp must be done with the buffer the slice's first tile refills.
    block.sync();
    tmp = gemv<DIM_PER_BLOCK / TMA_LOAD_ONCE, TMA_LOAD_ONCE, 128, NUM_PER_ROW, true>(
        tensor_map_weight_kv, s * 128, cluster_block_st_idx, weight, bar, input_shmem, input_idx, weight_idx);
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
      local_qkv[MLA_HEAD_DIM + s * 128 + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] =
          __float2half(tmp);
    }
  }
  block.sync();

  tmp = gemv<DIM_PER_BLOCK / 128, 128, ROPE_HEAD_DIM, NUM_PER_ROW_2, true>(
      tensor_map_weight_k_pe, 0, cluster_block_st_idx, weight, bar, input_shmem, input_idx_2, weight_idx_2);
#pragma unroll
  for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
    tmp += __shfl_down_sync(0xffffffff, tmp, mask);
  }
  if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
    local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2] =
        __float2half(tmp);
  }
  block.sync();

  // Sum the 1152 partial values over the cluster (local_qkv[128..512) is not
  // written above and is summed as it is).
  size = MLA_HEAD_DIM * 2 * sizeof(half);
  src_addr = smem_addr(local_qkv);
  dst_addr = smem_addr(weight);
  cf::cluster_reduce<CLUSTER_SIZE, cf::Stage::LINEAR_DEEPSEEK>(size, BLOCK_SIZE * 8, cluster_block_id, src_addr,
                                                               dst_addr, bar_ptr, neighbor_dst_bar, local_qkv, weight);

  // RoPE of q_pe and k_pe, whose results the attention below never reads.
  if (tid < ROPE_HEAD_DIM / 2) {
    q_rope = *(half2*)(&local_qkv[KV_LORA_RANK + tid * 2]);
    k_rope = *(half2*)(&local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + tid * 2]);
    if (tid * 2 < ROPE_HEAD_DIM / 2) {
      q_rope_1 = *(half2*)(&local_qkv[KV_LORA_RANK + ROPE_HEAD_DIM / 2 + tid * 2]);
      k_rope_1 = *(half2*)(&local_qkv[MLA_HEAD_DIM + ROPE_HEAD_DIM / 2 + tid * 2]);
      cos_reg = make_float2(cos[tid * 2], cos[tid * 2 + 1]);
      sin_reg = make_float2(-sin[ROPE_HEAD_DIM / 2 + tid * 2], -sin[ROPE_HEAD_DIM / 2 + tid * 2 + 1]);
    } else {
      q_rope_1 = *(half2*)(&local_qkv[KV_LORA_RANK + tid * 2 - ROPE_HEAD_DIM / 2]);
      k_rope_1 = *(half2*)(&local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + tid * 2 - ROPE_HEAD_DIM / 2]);
      cos_reg = make_float2(cos[tid * 2], cos[tid * 2 + 1]);
      sin_reg = make_float2(sin[tid * 2 - ROPE_HEAD_DIM / 2], sin[tid * 2 + 1 - ROPE_HEAD_DIM / 2]);
    }
    *(half2*)(&local_qkv[KV_LORA_RANK + tid * 2]) =
        __hadd2(__hmul2(q_rope, __float22half2_rn(cos_reg)), __hmul2(q_rope_1, __float22half2_rn(sin_reg)));
    *(half2*)(&local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + tid * 2]) =
        __hadd2(__hmul2(k_rope, __float22half2_rn(cos_reg)), __hmul2(k_rope_1, __float22half2_rn(sin_reg)));
  }

  // RMSNorm of the new latent c_kv (the whole vector is in every block).
  local_sum = 0.0;
  for (uint32_t d = tid * 4; d < KV_LORA_RANK; d += BLOCK_SIZE * 4) {
    *(uint64_t*)(&reg_input[0]) = *(uint64_t*)(&local_qkv[MLA_HEAD_DIM + d]);
    for (int di = 0; di < 4; di++)
      local_sum += __half2float(reg_input[di]) * __half2float(reg_input[di]);
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
  }
  if (lane_id == 0) {
    reduction[warp_id] = local_sum;
  }
  block.sync();
  if (tid < NUM_WARPS)
    local_sum = reduction[tid];
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
  }
  if (tid == 0)
    cluster_local_sum = local_sum;
  block.sync();
  rms_rcp = __frsqrt_rn(cluster_local_sum / KV_LORA_RANK + eps);
  for (uint32_t d = tid * 4; d < KV_LORA_RANK; d += BLOCK_SIZE * 4) {
    *(uint64_t*)(&reg_weight[0]) = *(const uint64_t*)(&w_rms_ckv[d]);
    for (int i = 0; i < 4; i++) {
      reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
    }
    *(uint64_t*)(&local_qkv[MLA_HEAD_DIM + d]) = *(uint64_t*)(&reg_input[0]);
  }
  block.sync();

  // q_lat slice [cluster_block_id * 128, +128) = q_nope @ W_UK[head].
  tmp = gemv<NOPE_HEAD_DIM / TMA_LOAD_ONCE, TMA_LOAD_ONCE, 128, NUM_PER_ROW, false>(
      tensor_map_weight_uk, cluster_head_uk_idx, 0, weight, bar, local_qkv, input_idx, weight_idx);
  // Block 0 overwrites q_nope, which other warps may still be reading.
  block.sync();
  if (lane_id % NUM_THREAD_PER_ROW == 0) {
    local_qkv[cluster_block_id * 128 + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
  }
  block.sync();

  // Gather the four q_lat slices in every block.
  size = 128 * sizeof(half);
  src_addr = smem_addr(&local_qkv[cluster_block_id * 128]);
  dst_addr = smem_addr(&local_qkv[cluster_block_id * 128]);
  cf::cluster_reduce<CLUSTER_SIZE, cf::Stage::QUK_DEEPSEEK>(size, 0, cluster_block_id, src_addr, dst_addr, bar_ptr,
                                                            neighbor_dst_bar, local_qkv, local_qkv);

  // Flash decoding over this block's KV_DIM_PER_BLOCK latent-cache rows, in
  // tiles of 16 rows; columns 0-255 and 256-511 come through two maps on one
  // barrier.
  local_sum = 0.0f;
  for (uint32_t i = 0; i < NUM_PER_THREAD; i++)
    reg_reduce[i] = __float2half(0.0f);
  *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3]);
  block.sync();

  const uint32_t kv_row0 = cluster_block_id * KV_DIM_PER_BLOCK;
  if (tid == 0) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], tensor_map_kv_cache, 0, kv_row0, bar[0]);
    cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_ATTN / 2], tensor_map_kv_cache_,
                                                  KV_LORA_RANK / 2, kv_row0, bar[0]);
    token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_ATTN);
  } else {
    token[0] = bar[0].arrive();
  }

  for (uint32_t id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
    // Every warp must be done with the buffer this refill overwrites.
    block.sync();
    if (tid == 0) {
      cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_ATTN], tensor_map_kv_cache,
                                                    0, kv_row0 + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          &weight[(id % 2) * TMA_LOAD_ONCE_NUM_ATTN + TMA_LOAD_ONCE_NUM_ATTN / 2], tensor_map_kv_cache_,
          KV_LORA_RANK / 2, kv_row0 + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
      token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
    } else {
      token[id % 2] = bar[id % 2].arrive();
    }
    bar[(id - 1) % 2].wait(moved(token[(id - 1) % 2]));
    latent_tile<false>(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM_ATTN], &local_qkv[MLA_HEAD_DIM], reg_input,
                       reg_reduce, reduction, local_max, local_sum, cluster_block_id, softmax_scale);
  }
  bar[1].wait(moved(token[1]));
  latent_tile<true>(&weight[TMA_LOAD_ONCE_NUM_ATTN], &local_qkv[MLA_HEAD_DIM], reg_input, reg_reduce, reduction,
                    local_max, local_sum, cluster_block_id, softmax_scale);
  block.sync();

  // Merge the two tile rows of the block.
  *(uint4*)(&weight[tile_row * MLA_HEAD_DIM + tile_col * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
  if (tid % NUM_THREAD_PER_ROW_3 == 0) {
    reduction[tile_row * 2] = local_max;
    reduction[tile_row * 2 + 1] = local_sum;
  }
  block.sync();
  merge_rows(reg_reduce, local_sum, local_max, weight, reduction, tile_col);
  block.sync();

  // Merge over the cluster: the maximum, then the rescaled sums.
  pre_max = local_max;
  if (tid == 0) {
    cluster_local_max = local_max;
  }
  cluster.sync();
  cf::cluster_max_rounds(cluster_local_max, local_max, dst_shmem, cluster_block_id);
  scale = __expf(pre_max - cluster_local_max);
  local_sum *= scale;
#pragma unroll
  for (uint32_t j = 0; j < NUM_PER_THREAD; j++) {
    reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
  }
  if (tid == 0) {
    cluster_local_sum = local_sum;
  }
  cluster.sync();
  cf::cluster_sum_rounds(cluster_local_sum, local_sum, dst_shmem, cluster_block_id);
#pragma unroll
  for (uint32_t j = 0; j < NUM_PER_THREAD; j++) {
    reg_reduce[j] = __hmul(reg_reduce[j], __float2half(__frcp_rn(cluster_local_sum)));
  }
  if (tid < NUM_THREAD_PER_ROW_3) {
    *(uint4*)(&local_qkv[tid * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
  }
  block.sync();

  // Sum the latent attention output a_lat (512) over the cluster.
  size = KV_LORA_RANK * sizeof(half);
  src_addr = smem_addr(local_qkv);
  dst_addr = smem_addr(weight);
  cf::cluster_reduce<CLUSTER_SIZE, cf::Stage::ATTN_DEEPSEEK>(size, KV_LORA_RANK, cluster_block_id, src_addr, dst_addr,
                                                             bar_ptr, neighbor_dst_bar, local_qkv, weight);

  // a_h partial over this block's 128 latent dimensions: a_lat @ W_UV[head].
  tmp = gemv<128 / TMA_LOAD_ONCE, TMA_LOAD_ONCE, 128, NUM_PER_ROW, true>(
      tensor_map_weight_uv, cluster_head_idx, cluster_block_st_uv_idx, weight, bar, local_qkv + cluster_block_st_uv_idx,
      input_idx, weight_idx);
  // This store overwrites a_lat[0..128), which block 0's other warps may still read.
  block.sync();
  if (lane_id % NUM_THREAD_PER_ROW == 0) {
    local_qkv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
  }
  block.sync();

  size = NOPE_HEAD_DIM * sizeof(half);
  src_addr = smem_addr(local_qkv);
  dst_addr = smem_addr(weight);
  cf::cluster_reduce<CLUSTER_SIZE, cf::Stage::ATTN>(size, NOPE_HEAD_DIM, cluster_block_id, src_addr, dst_addr, bar_ptr,
                                                    neighbor_dst_bar, local_qkv, weight);

  // Output projection: this block's DIM_PER_BLOCK outputs of a_h @ W_o[head],
  // added over the heads with f16 atomics into the zeroed output.
  constexpr uint32_t wo_tiles = DIM_PER_BLOCK / TMA_LOAD_ONCE;
  static_assert(wo_tiles % 2 == 0, "the last tile is in buffer 1");
  token[0] = tma_issue(&weight[0], tensor_map_weight_o, cluster_block_st_idx, cluster_head_idx, bar[0],
                       TMA_LOAD_ONCE_SIZE);
  for (uint32_t id = 1; id < wo_tiles; id++) {
    token[id % 2] = tma_issue(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], tensor_map_weight_o,
                              cluster_block_st_idx + id * TMA_LOAD_ONCE, cluster_head_idx, bar[id % 2],
                              TMA_LOAD_ONCE_SIZE);
    bar[(id - 1) % 2].wait(moved(token[(id - 1) % 2]));
    tmp = 0.0;
    for (uint32_t j = 0; j < NOPE_HEAD_DIM; j += NUM_PER_ROW_2) {
      *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2 + j]);
#pragma unroll
      for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
        tmp += __half2float(__hmul(reg_input[d], weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM + (input_idx_2 + j + d) * TMA_LOAD_ONCE + weight_idx_2]));
      }
    }
#pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
      tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
      atomicAdd(&output[cluster_block_st_idx + weight_idx_2 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    }
    block.sync();
  }
  bar[1].wait(moved(token[1]));
  tmp = 0.0;
  for (uint32_t j = 0; j < NOPE_HEAD_DIM; j += NUM_PER_ROW_2) {
    *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2 + j]);
#pragma unroll
    for (uint32_t d = 0; d < NUM_PER_THREAD; d++) {
      tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx_2 + j + d) * TMA_LOAD_ONCE + weight_idx_2]));
    }
  }
#pragma unroll
  for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
    tmp += __shfl_down_sync(0xffffffff, tmp, mask);
  }
  if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
    atomicAdd(&output[cluster_block_st_idx + weight_idx_2 + (wo_tiles - 1) * TMA_LOAD_ONCE], __float2half(tmp));
  }
}
