#ifndef _CF_DECODER_COMMON_H_
#define _CF_DECODER_COMMON_H_

// Shapes, local-memory layouts and argument blocks of the two ClusterFusion
// decoder kernels, shared by the host and the kernels.

#include <stdint.h>
#include <dxa_tmap.h>

// The compile-time problem sizes of the build (make variables, see the
// Makefile). Every block, cluster and tile constant below is the H100 build's.
#ifndef CF_LLAMA_HEADS
#define CF_LLAMA_HEADS 4
#endif
#ifndef CF_MLA_HEADS
#define CF_MLA_HEADS 2
#endif
#ifndef CF_MLA_HIDDEN
#define CF_MLA_HIDDEN 1024
#endif
#ifndef CF_MLA_SEQ
#define CF_MLA_SEQ 128
#endif

// Llama-2 attention block (include/H100/llama/config.h). HIDDEN_DIM is
// HEAD_NUM x HEAD_DIM as in Llama-2-7B; FFN_DIM keeps config.h's ratio of
// 3 x HIDDEN_DIM, so MAX_SMEM_DIM, which sizes the weight buffers, stays 384.
namespace cf_llama {
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t HEAD_NUM = CF_LLAMA_HEADS;
constexpr uint32_t HIDDEN_DIM = HEAD_NUM * HEAD_DIM;
constexpr uint32_t FFN_DIM = 3 * HIDDEN_DIM;
constexpr uint32_t NUM_WARPS = 4;
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
constexpr uint32_t CLUSTER_SIZE = 4;
constexpr uint32_t NUM_PER_THREAD = 8;
constexpr uint32_t NUM_ROW_PER_WARP = HEAD_DIM / NUM_WARPS;
constexpr uint32_t NUM_THREAD_PER_ROW = WARP_SIZE / NUM_ROW_PER_WARP;
constexpr uint32_t NUM_PER_ROW = NUM_PER_THREAD * NUM_THREAD_PER_ROW;
constexpr uint32_t DIM_PER_BLOCK = HIDDEN_DIM / CLUSTER_SIZE;
constexpr uint32_t FFN_DIM_PER_CLUSTER = FFN_DIM / HEAD_NUM;
constexpr uint32_t MAX_SMEM_DIM = FFN_DIM_PER_CLUSTER;
constexpr uint32_t TMA_LOAD_ONCE = 64;
constexpr uint32_t TMA_LOAD_ONCE_NUM = TMA_LOAD_ONCE * HEAD_DIM;
constexpr uint32_t TMA_LOAD_ONCE_SIZE = TMA_LOAD_ONCE_NUM * 2;
constexpr uint32_t TMA_LOAD_ONCE_ATTN = TMA_LOAD_ONCE / 2;
constexpr uint32_t TMA_LOAD_ONCE_NUM_ATTN = (TMA_LOAD_ONCE * HEAD_DIM) / 2;
constexpr uint32_t TMA_LOAD_ONCE_SIZE_ATTN = TMA_LOAD_ONCE_NUM_ATTN * 2;
constexpr uint32_t NUM_THREAD_PER_ROW_2 = HEAD_DIM / NUM_PER_THREAD;
constexpr uint32_t NUM_ROW_PER_WARP_2 = TMA_LOAD_ONCE_ATTN / NUM_WARPS;
constexpr uint32_t DIM_BLOCK_REDUCE = 2 * BLOCK_SIZE / NUM_THREAD_PER_ROW_2;
constexpr uint32_t DEC_TILE = NUM_ROW_PER_WARP_2 / (WARP_SIZE / NUM_THREAD_PER_ROW_2);
constexpr uint32_t NUM_ROW_PER_WARP_3 = TMA_LOAD_ONCE / NUM_WARPS;
constexpr uint32_t NUM_THREAD_PER_ROW_3 = WARP_SIZE / NUM_ROW_PER_WARP_3;
constexpr uint32_t NUM_PER_ROW_3 = NUM_PER_THREAD * NUM_THREAD_PER_ROW_3;
static_assert(HEAD_NUM >= 2, "the GEMVs need a whole TMA tile per block: HIDDEN_DIM / 4 >= 64");
static_assert(DIM_PER_BLOCK % TMA_LOAD_ONCE == 0, "whole TMA tiles per block");

// Local memory. The statics come first, in declaration order (the two
// cluster scalars, the four TMA barriers, the cluster-reduce mbarrier); the
// dynamic part follows and is aligned up to 128 bytes inside, as the
// original aligns its extern buffer.
constexpr uint32_t LM_SUM = 0;
constexpr uint32_t LM_MAX = 4;
constexpr uint32_t LM_BAR = 8;
constexpr uint32_t LM_RBAR = 40;
constexpr uint32_t LM_DYN = 48;
constexpr uint32_t DYN_SMEM = 128 + (2 * TMA_LOAD_ONCE * MAX_SMEM_DIM + DIM_PER_BLOCK + 3 * HEAD_DIM) * 2 +
                              DIM_BLOCK_REDUCE * 4;
constexpr uint32_t LMEM_BYTES = LM_DYN + DYN_SMEM;
} // namespace cf_llama

// DeepSeek-V2-Lite MLA (include/H100/deepseek/config.h).
namespace cf_mla {
constexpr uint32_t NOPE_HEAD_DIM = 128;
constexpr uint32_t ROPE_HEAD_DIM = 64;
constexpr uint32_t HEAD_DIM = ROPE_HEAD_DIM + NOPE_HEAD_DIM;
constexpr uint32_t KV_LORA_RANK = 512;
constexpr uint32_t MLA_HEAD_DIM = KV_LORA_RANK + ROPE_HEAD_DIM;
constexpr uint32_t HEAD_NUM = CF_MLA_HEADS;
constexpr uint32_t HIDDEN_DIM = CF_MLA_HIDDEN;
constexpr uint32_t SEQ_LEN = CF_MLA_SEQ;
constexpr uint32_t NUM_WARPS = 4;
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
constexpr uint32_t CLUSTER_SIZE = 4;
constexpr uint32_t NUM_PER_THREAD = 8;
constexpr uint32_t NUM_ROW_PER_WARP = NOPE_HEAD_DIM / NUM_WARPS;
constexpr uint32_t NUM_THREAD_PER_ROW = WARP_SIZE / NUM_ROW_PER_WARP;
constexpr uint32_t NUM_PER_ROW = NUM_PER_THREAD * NUM_THREAD_PER_ROW;
constexpr uint32_t DIM_PER_BLOCK = HIDDEN_DIM / CLUSTER_SIZE;
constexpr uint32_t KV_DIM_PER_BLOCK = SEQ_LEN / CLUSTER_SIZE;
constexpr uint32_t TMA_LOAD_ONCE = 64;
constexpr uint32_t TMA_LOAD_ONCE_NUM = TMA_LOAD_ONCE * NOPE_HEAD_DIM;
constexpr uint32_t TMA_LOAD_ONCE_SIZE = TMA_LOAD_ONCE_NUM * 2;
constexpr uint32_t TMA_LOAD_ONCE_ATTN = TMA_LOAD_ONCE / 4;
constexpr uint32_t TMA_LOAD_ONCE_NUM_ATTN = TMA_LOAD_ONCE_ATTN * KV_LORA_RANK;
constexpr uint32_t TMA_LOAD_ONCE_SIZE_ATTN = TMA_LOAD_ONCE_NUM_ATTN * 2;
constexpr uint32_t NUM_ROW_PER_WARP_2 = TMA_LOAD_ONCE / NUM_WARPS;
constexpr uint32_t NUM_THREAD_PER_ROW_2 = WARP_SIZE / NUM_ROW_PER_WARP_2;
constexpr uint32_t NUM_PER_ROW_2 = NUM_PER_THREAD * NUM_THREAD_PER_ROW_2;
constexpr uint32_t NUM_THREAD_PER_ROW_3 = KV_LORA_RANK / NUM_PER_THREAD;
constexpr uint32_t NUM_ROW_PER_BLOCK = BLOCK_SIZE / NUM_THREAD_PER_ROW_3;
constexpr uint32_t DEC_TILE = TMA_LOAD_ONCE_ATTN / NUM_ROW_PER_BLOCK;
// The kernel waits on bar[1] after every double-buffered loop: each loop
// must have an even number of tiles.
static_assert((DIM_PER_BLOCK / TMA_LOAD_ONCE) % 2 == 0 && (DIM_PER_BLOCK / 128) % 2 == 0,
              "HIDDEN_DIM must be a multiple of 1024");
static_assert((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN) % 2 == 0, "SEQ_LEN must be a multiple of 128");

// Local memory: the original's static shared variables in declaration order.
constexpr uint32_t LM_INPUT = 0;
constexpr uint32_t LM_REDUCTION = LM_INPUT + DIM_PER_BLOCK * 2;
constexpr uint32_t LM_SUM = LM_REDUCTION + NUM_WARPS * 4;
constexpr uint32_t LM_MAX = LM_SUM + 4;
constexpr uint32_t LM_WEIGHT = (LM_MAX + 4 + 127) & ~127u;
constexpr uint32_t LM_QKV = LM_WEIGHT + 2 * TMA_LOAD_ONCE * NOPE_HEAD_DIM * 2;
constexpr uint32_t LM_BAR = (LM_QKV + MLA_HEAD_DIM * 2 * 2 + 7) & ~7u;
constexpr uint32_t LM_RBAR = LM_BAR + 2 * 8;
constexpr uint32_t LMEM_BYTES = LM_RBAR + 8;
} // namespace cf_mla

// The argument blocks: the tensor maps first (the argument buffer is 64-byte
// aligned, a map must be), then the device addresses.
typedef struct {
  vx_tensor_map_t tensor_map;          // W_qkv
  vx_tensor_map_t tensor_map_k_cache;
  vx_tensor_map_t tensor_map_v_cache;
  vx_tensor_map_t tensor_map_weight_o;
  uint64_t output;
  uint64_t k_output;
  uint64_t v_output;
  uint64_t input;
  uint64_t residual;
  uint64_t residual_output;
  uint64_t w_rms_input;
  uint64_t cos;
  uint64_t sin;
  uint64_t smid;                       // one core id per block
  float    eps;
  uint32_t seq_len;
  uint32_t kv_dim_per_block;
  uint32_t pad;
} __attribute__((aligned(64))) llama_arg_t;

typedef struct {
  vx_tensor_map_t tensor_map_weight_q;
  vx_tensor_map_t tensor_map_weight_q_pe;
  vx_tensor_map_t tensor_map_weight_uk;
  vx_tensor_map_t tensor_map_weight_kv;
  vx_tensor_map_t tensor_map_weight_k_pe;
  vx_tensor_map_t tensor_map_kv_cache;
  vx_tensor_map_t tensor_map_kv_cache_;
  vx_tensor_map_t tensor_map_weight_uv;
  vx_tensor_map_t tensor_map_weight_o;
  uint64_t output;
  uint64_t input;
  uint64_t w_rms_input;
  uint64_t w_rms_ckv;
  uint64_t cos;
  uint64_t sin;
  uint64_t smid;
} __attribute__((aligned(64))) mla_arg_t;

// Both kernels are __cluster_dims__(CLUSTER_SIZE, 1, 1).
VX_CUDA_CLUSTER_DIMS(LlamaDecoderLayerKernel, cf_llama::CLUSTER_SIZE, 1, 1);
VX_CUDA_CLUSTER_DIMS(DeepSeekDecoderLayerKernel, cf_mla::CLUSTER_SIZE, 1, 1);

#endif
