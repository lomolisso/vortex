#include <vx_cuda.h>
#include <vx_print.h>
#include "common.h"
#include "types_ops.h"

namespace cg = cooperative_groups;

namespace {

#define INL inline __attribute__((always_inline))

typedef unsigned long long ull;
typedef long long ll;

template <typename T>
INL T from_bits(uint64_t b) {
  T v;
  __builtin_memcpy(&v, &b, sizeof(T));
  return v;
}

template <typename T>
INL uint64_t to_bits(T v) {
  uint64_t b = 0;
  __builtin_memcpy(&b, &v, sizeof(T));
  return b;
}

// The calling thread's rank in the grid, and its result words.
INL uint32_t grid_thread() {
  const uint32_t bt = blockDim.x * blockDim.y * blockDim.z;
  const uint32_t br = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  const uint32_t tr = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  return br * bt + tr;
}

INL uint64_t* out_of(const cuda_arg_t* arg) {
  return reinterpret_cast<uint64_t*>(arg->out) + (uint64_t)grid_thread() * arg->stride;
}

// Copies the block's local memory to its dump slot.
INL void dump_lmem(const cuda_arg_t* arg, uint32_t bytes) {
  const uint32_t br = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  const uint32_t nt = blockDim.x * blockDim.y * blockDim.z;
  const uint32_t tr = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  auto src = reinterpret_cast<const volatile uint32_t*>(__local_mem());
  auto dst = reinterpret_cast<uint32_t*>(arg->dump + (uint64_t)br * arg->lmem);
  for (uint32_t w = tr; w < bytes / 4; w += nt)
    dst[w] = src[w];
}

} // namespace

// ── cg: the identity of every group ─────────────────────────────────────
__global__ void cg_k(cuda_arg_t* __UNIFORM__ arg) {
  uint64_t* o = out_of(arg);
  cg::grid_group grid = cg::this_grid();
  cg::thread_block tb = cg::this_thread_block();
  cg::cluster_group cl = cg::this_cluster();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(tb);
  uint32_t k = 0;
  o[k++] = grid.thread_rank();
  o[k++] = grid.num_threads();
  o[k++] = grid.block_rank();
  o[k++] = grid.num_blocks();
  o[k++] = grid.dim_blocks().x;
  o[k++] = grid.dim_blocks().y;
  o[k++] = grid.dim_blocks().z;
  o[k++] = grid.block_index().x;
  o[k++] = grid.block_index().y;
  o[k++] = grid.block_index().z;
  o[k++] = grid.cluster_rank();
  o[k++] = grid.num_clusters();
  o[k++] = grid.dim_clusters().x;
  o[k++] = grid.dim_clusters().y;
  o[k++] = grid.dim_clusters().z;
  o[k++] = grid.is_valid();
  o[k++] = tb.thread_rank();
  o[k++] = tb.num_threads();
  o[k++] = tb.size();
  o[k++] = tb.group_index().x;
  o[k++] = tb.group_index().y;
  o[k++] = tb.group_index().z;
  o[k++] = tb.thread_index().x;
  o[k++] = tb.thread_index().y;
  o[k++] = tb.thread_index().z;
  o[k++] = tb.dim_threads().x;
  o[k++] = tb.dim_threads().y;
  o[k++] = tb.dim_threads().z;
  o[k++] = tb.group_dim().x;
  o[k++] = tb.group_dim().y;
  o[k++] = tb.group_dim().z;
  o[k++] = cl.block_rank();
  o[k++] = cl.num_blocks();
  o[k++] = cl.dim_blocks().x;
  o[k++] = cl.dim_blocks().y;
  o[k++] = cl.dim_blocks().z;
  o[k++] = cl.block_index().x;
  o[k++] = cl.block_index().y;
  o[k++] = cl.block_index().z;
  o[k++] = cl.thread_rank();
  o[k++] = cl.num_threads();
  o[k++] = cl.dim_threads().x;
  o[k++] = cl.dim_threads().y;
  o[k++] = cl.dim_threads().z;
  o[k++] = cl.is_valid();
  // Each block publishes its grid rank; every thread reads its successor's
  // through the window after the cluster barrier.
  auto lm = reinterpret_cast<uint32_t*>(__local_mem());
  if (tb.thread_rank() == 0)
    lm[0] = 0x1000 + grid.block_rank();
  cl.sync();
  const uint32_t peer = (cl.block_rank() + 1) % cl.num_blocks();
  uint32_t* pw = cl.map_shared_rank(lm, peer);
  o[k++] = *pw;
  o[k++] = cl.query_shared_rank(pw);
  o[k++] = tile.thread_rank();
  o[k++] = tile.size();
  o[k++] = tile.meta_group_rank();
  o[k++] = tile.meta_group_size();
  o[k++] = grid.size();
  tile.sync();
  auto counts = reinterpret_cast<unsigned*>(arg->aux);
  const uint32_t br = grid.block_rank();
  cg::invoke_one(tb, [&]() { atomicAdd(&counts[br * 2], 1u); });
  cg::invoke_one(tile, [&]() { atomicAdd(&counts[br * 2 + 1], 1u); });
  o[k++] = 0xC0DE;
  // Nobody may leave before its peer has read this block's word.
  cl.sync();
}

// ── warp: shuffles, votes, elect, reductions, match, lane masks ─────────
namespace {

template <typename T>
INL T shfl_kind(uint32_t kind, T v, uint32_t d, uint32_t w) {
  switch (kind) {
  case 0:  return __shfl_sync(0xffffffffu, v, (int)d, (int)w);
  case 1:  return __shfl_up_sync(0xffffffffu, v, d, (int)w);
  case 2:  return __shfl_down_sync(0xffffffffu, v, d, (int)w);
  default: return __shfl_xor_sync(0xffffffffu, v, (int)d, (int)w);
  }
}

INL uint64_t shfl_type(uint32_t type, uint32_t kind, uint64_t bits, uint32_t d, uint32_t w) {
  switch (type) {
  case 0:  return to_bits(shfl_kind(kind, from_bits<unsigned>(bits), d, w));
  case 1:  return to_bits(shfl_kind(kind, from_bits<ull>(bits), d, w));
  case 2:  return to_bits(shfl_kind(kind, from_bits<float>(bits), d, w));
  case 3:  return to_bits(shfl_kind(kind, from_bits<double>(bits), d, w));
  default: return to_bits(shfl_kind(kind, from_bits<__half2>(bits), d, w));
  }
}

INL void vote_case(uint64_t* o, uint32_t c, uint32_t t) {
  const uint32_t m = w_vcase_mask(c);
  const int p = (int)w_pred(c, t);
  o[0] = __ballot_sync(m, p);
  o[1] = (uint32_t)__any_sync(m, p);
  o[2] = (uint32_t)__all_sync(m, p);
  o[3] = __activemask();
  o[4] = (uint32_t)__all_sync(m, 1);
  o[5] = (uint32_t)__any_sync(m, 0);
}

INL uint32_t reduce_op(uint32_t op, uint32_t m, uint32_t v) {
  switch (op) {
  case 0:  return __reduce_add_sync(m, v);
  case 1:  return (uint32_t)__reduce_add_sync(m, (int)v);
  case 2:  return __reduce_min_sync(m, v);
  case 3:  return (uint32_t)__reduce_min_sync(m, (int)v);
  case 4:  return __reduce_max_sync(m, v);
  case 5:  return (uint32_t)__reduce_max_sync(m, (int)v);
  case 6:  return __reduce_and_sync(m, v);
  case 7:  return __reduce_or_sync(m, v);
  default: return __reduce_xor_sync(m, v);
  }
}

INL void match_case(uint64_t* o, uint32_t wide, uint32_t c, uint32_t lane) {
  const uint64_t key = w_key(wide, c, lane);
  int pred = -1;
  if (wide) {
    o[0] = __match_any_sync(0xffffffffu, (ull)key);
    o[1] = __match_all_sync(0xffffffffu, (ull)key, &pred);
  } else {
    o[0] = __match_any_sync(0xffffffffu, (unsigned)key);
    o[1] = __match_all_sync(0xffffffffu, (unsigned)key, &pred);
  }
  o[2] = (uint32_t)pred;
}

} // namespace

__global__ void warp_k(cuda_arg_t* __UNIFORM__ arg) {
  const uint32_t t = threadIdx.x;
  const uint32_t lane = t % WARP;
  uint64_t* o = out_of(arg);

  for (uint32_t kind = 0; kind < W_NSHFL; ++kind) {
    for (uint32_t wi = 0; wi < W_NWIDTH; ++wi) {
      const uint32_t w = w_width(wi);
      for (uint32_t di = 0; di < W_NDELTA; ++di) {
        const uint32_t d = w_delta(di, w);
        for (uint32_t type = 0; type < W_NTYPE; ++type)
          o[W_SHFL + ((kind * W_NWIDTH + wi) * W_NDELTA + di) * W_NTYPE + type]
              = shfl_type(type, kind, w_value(type, t), d, w);
      }
    }
  }

  for (uint32_t c = 0; c < W_NVCASE; ++c)
    o[W_VOTE + c * W_VOTE_N] = o[W_VOTE + c * W_VOTE_N + 1] = o[W_VOTE + c * W_VOTE_N + 2]
        = o[W_VOTE + c * W_VOTE_N + 3] = o[W_VOTE + c * W_VOTE_N + 4] = o[W_VOTE + c * W_VOTE_N + 5] = ~0ull;
  if (lane & 1)
    vote_case(o + W_VOTE + 0 * W_VOTE_N, 0, t);
  if (lane < 5)
    vote_case(o + W_VOTE + 1 * W_VOTE_N, 1, t);
  if (lane == 7)
    vote_case(o + W_VOTE + 2 * W_VOTE_N, 2, t);
  vote_case(o + W_VOTE + 3 * W_VOTE_N, 3, t);

  o[W_ELECT + 0] = cuda::ptx::elect_sync(0xffffffffu);
  o[W_ELECT + 1] = 2;
  o[W_ELECT + 2] = 2;
  if (lane & 1)
    o[W_ELECT + 1] = cuda::ptx::elect_sync(0xAAAAAAAAu);
  if (lane >= 4 && lane < 8)
    o[W_ELECT + 2] = cuda::ptx::elect_sync(0xF0u);
  o[W_ELECT + 3] = cute::elect_one_sync();

  for (uint32_t op = 0; op < W_NROP; ++op) {
    o[W_RED + op * W_NRCASE + 0] = reduce_op(op, 0xffffffffu, w_rvalue(0, t));
    o[W_RED + op * W_NRCASE + 1] = ~0ull;
    if (lane % 3 != 0)
      o[W_RED + op * W_NRCASE + 1] = reduce_op(op, w_rmask(1), w_rvalue(1, t));
    o[W_RED + op * W_NRCASE + 2] = reduce_op(op, 0xffffffffu, w_rvalue(2, t));
  }

  for (uint32_t wide = 0; wide < 2; ++wide) {
    for (uint32_t c = 0; c < W_NMCASE; ++c)
      match_case(o + W_MATCH + (wide * W_NMCASE + c) * W_MATCH_N, wide, c, lane);
  }

  o[W_LANE + 0] = __lanemask_eq();
  o[W_LANE + 1] = __lanemask_lt();
  o[W_LANE + 2] = __lanemask_le();
  o[W_LANE + 3] = __lanemask_gt();
  o[W_LANE + 4] = __lanemask_ge();
  o[W_LANE + 5] = __laneid();

  cg::thread_block tb = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(tb);
  const unsigned v = (unsigned)w_value(0, t);
  const int p = (int)w_pred(3, t);
  o[W_TILE + 0] = tile.shfl(v, (int)((lane + 5) % WARP));
  o[W_TILE + 1] = tile.shfl_up(v, 2);
  o[W_TILE + 2] = tile.shfl_down(v, 3);
  o[W_TILE + 3] = tile.shfl_xor(v, 6);
  o[W_TILE + 4] = (uint32_t)tile.any(p);
  o[W_TILE + 5] = (uint32_t)tile.all(p);
  o[W_TILE + 6] = tile.ballot(p);
  tile.sync();
  o[W_TILE + 7] = tile.thread_rank();
  o[W_TILE + 8] = __shfl_sync(0xffffffffu, v, 5);
}

// ── atomic: every op and type on global, own and peer local memory ──────
namespace {

INL uint64_t atom(uint32_t op, void* p, uint64_t v, uint64_t init) {
  switch (op) {
  case A_ADD_S32:    return to_bits(atomicAdd((int*)p, from_bits<int>(v)));
  case A_ADD_U32:    return to_bits(atomicAdd((unsigned*)p, from_bits<unsigned>(v)));
  case A_ADD_U64:    return to_bits(atomicAdd((ull*)p, from_bits<ull>(v)));
  case A_ADD_F32:    return to_bits(atomicAdd((float*)p, from_bits<float>(v)));
  case A_ADD_F64:    return to_bits(atomicAdd((double*)p, from_bits<double>(v)));
  case A_ADD_F16:    return to_bits(atomicAdd((__half*)p, from_bits<__half>(v)));
  case A_ADD_F16X2:  return to_bits(atomicAdd((__half2*)p, from_bits<__half2>(v)));
  case A_ADD_BF16:   return to_bits(atomicAdd((__nv_bfloat16*)p, from_bits<__nv_bfloat16>(v)));
  case A_ADD_BF16X2: return to_bits(atomicAdd((__nv_bfloat162*)p, from_bits<__nv_bfloat162>(v)));
  case A_SUB_S32:    return to_bits(atomicSub((int*)p, from_bits<int>(v)));
  case A_SUB_U32:    return to_bits(atomicSub((unsigned*)p, from_bits<unsigned>(v)));
  case A_EXCH_S32:   return to_bits(atomicExch((int*)p, from_bits<int>(v)));
  case A_EXCH_U32:   return to_bits(atomicExch((unsigned*)p, from_bits<unsigned>(v)));
  case A_EXCH_U64:   return to_bits(atomicExch((ull*)p, from_bits<ull>(v)));
  case A_EXCH_F32:   return to_bits(atomicExch((float*)p, from_bits<float>(v)));
  case A_MIN_S32:    return to_bits(atomicMin((int*)p, from_bits<int>(v)));
  case A_MIN_U32:    return to_bits(atomicMin((unsigned*)p, from_bits<unsigned>(v)));
  case A_MIN_S64:    return to_bits(atomicMin((ll*)p, from_bits<ll>(v)));
  case A_MIN_U64:    return to_bits(atomicMin((ull*)p, from_bits<ull>(v)));
  case A_MAX_S32:    return to_bits(atomicMax((int*)p, from_bits<int>(v)));
  case A_MAX_U32:    return to_bits(atomicMax((unsigned*)p, from_bits<unsigned>(v)));
  case A_MAX_S64:    return to_bits(atomicMax((ll*)p, from_bits<ll>(v)));
  case A_MAX_U64:    return to_bits(atomicMax((ull*)p, from_bits<ull>(v)));
  case A_INC_U32:    return to_bits(atomicInc((unsigned*)p, from_bits<unsigned>(v)));
  case A_DEC_U32:    return to_bits(atomicDec((unsigned*)p, from_bits<unsigned>(v)));
  case A_AND_S32:    return to_bits(atomicAnd((int*)p, from_bits<int>(v)));
  case A_AND_U32:    return to_bits(atomicAnd((unsigned*)p, from_bits<unsigned>(v)));
  case A_AND_U64:    return to_bits(atomicAnd((ull*)p, from_bits<ull>(v)));
  case A_OR_S32:     return to_bits(atomicOr((int*)p, from_bits<int>(v)));
  case A_OR_U32:     return to_bits(atomicOr((unsigned*)p, from_bits<unsigned>(v)));
  case A_OR_U64:     return to_bits(atomicOr((ull*)p, from_bits<ull>(v)));
  case A_XOR_S32:    return to_bits(atomicXor((int*)p, from_bits<int>(v)));
  case A_XOR_U32:    return to_bits(atomicXor((unsigned*)p, from_bits<unsigned>(v)));
  case A_XOR_U64:    return to_bits(atomicXor((ull*)p, from_bits<ull>(v)));
  case A_CAS_S32:    return to_bits(atomicCAS((int*)p, from_bits<int>(init), from_bits<int>(v)));
  case A_CAS_U32:    return to_bits(atomicCAS((unsigned*)p, from_bits<unsigned>(init), from_bits<unsigned>(v)));
  case A_CAS_U64:    return to_bits(atomicCAS((ull*)p, from_bits<ull>(init), from_bits<ull>(v)));
  case A_CAS_U16:
    return to_bits(atomicCAS((unsigned short*)p, from_bits<unsigned short>(init), from_bits<unsigned short>(v)));
  default:
    return to_bits(atomicCAS((unsigned short*)p + 1, from_bits<unsigned short>(init), from_bits<unsigned short>(v)));
  }
}

INL uint64_t atom_block(uint32_t k, unsigned* p, unsigned v, unsigned init) {
  switch (k) {
  case 0:  return atomicAdd_block(p, v);
  case 1:  return atomicSub_block(p, v);
  case 2:  return atomicExch_block(p, v);
  case 3:  return atomicMin_block(p, v);
  case 4:  return atomicMax_block(p, v);
  case 5:  return atomicInc_block(p, v);
  case 6:  return atomicDec_block(p, v);
  case 7:  return atomicAnd_block(p, v);
  case 8:  return atomicOr_block(p, v);
  case 9:  return atomicXor_block(p, v);
  default: return atomicCAS_block(p, init, v);
  }
}

INL uint64_t atom_system(uint32_t k, unsigned* p, unsigned v, unsigned init) {
  switch (k) {
  case 0:  return atomicAdd_system(p, v);
  case 1:  return atomicSub_system(p, v);
  case 2:  return atomicExch_system(p, v);
  case 3:  return atomicMin_system(p, v);
  case 4:  return atomicMax_system(p, v);
  case 5:  return atomicInc_system(p, v);
  case 6:  return atomicDec_system(p, v);
  case 7:  return atomicAnd_system(p, v);
  case 8:  return atomicOr_system(p, v);
  case 9:  return atomicXor_system(p, v);
  default: return atomicCAS_system(p, init, v);
  }
}

} // namespace

__global__ void atomic_k(cuda_arg_t* __UNIFORM__ arg) {
  cg::cluster_group cl = cg::this_cluster();
  const uint32_t t = threadIdx.x;
  const uint32_t nt = blockDim.x;
  const uint32_t blk = blockIdx.x;
  uint64_t* o = out_of(arg);
  auto lm = reinterpret_cast<uint64_t*>(__local_mem());
  for (uint32_t i = t; i < 2 * A_NUM_OPS; i += nt) {
    const uint32_t op = i % A_NUM_OPS;
    lm[i] = a_slot_init(op, (i < A_NUM_OPS ? A_NUM_OPS : 2 * A_NUM_OPS) + op);
  }
  cl.sync();
  const uint32_t peer = cl.block_rank() ^ 1;
  auto g = reinterpret_cast<uint64_t*>(arg->aux);
  uint64_t* win = cl.map_shared_rank(lm + A_NUM_OPS, peer);
  for (uint32_t sp = 0; sp < 3; ++sp) {
    uint64_t* base = sp == 0 ? g : sp == 1 ? lm : win;
    for (uint32_t op = 0; op < A_NUM_OPS; ++op) {
      const uint32_t tg = sp * A_NUM_OPS + op;
      o[tg] = atom(op, base + op, a_value(op, tg, blk, t), a_init(op, tg));
    }
  }
  for (uint32_t k = 0; k < 2 * A_NSCOPE_OPS; ++k) {
    const uint32_t tg = A_SCOPED + k;
    const uint32_t op = a_op(tg);
    unsigned* p = reinterpret_cast<unsigned*>(g + tg);
    const unsigned v = (unsigned)a_value(op, tg, blk, t), init = (unsigned)a_init(op, tg);
    o[tg] = k < A_NSCOPE_OPS ? atom_block(k, p, v, init) : atom_system(k - A_NSCOPE_OPS, p, v, init);
  }
  cl.sync();
  dump_lmem(arg, A_LMEM);
}

// ── barrier: cuda::barrier phases, drop, tx copies, memcpy_async ────────
__global__ void barrier_k(cuda_arg_t* __UNIFORM__ arg) {
  typedef cuda::barrier<cuda::thread_scope_block> Bar;
  const uint32_t t = threadIdx.x;
  const uint32_t warp = t / WARP;
  uint64_t* o = out_of(arg);
  auto lm = reinterpret_cast<uint8_t*>(__local_mem());
  Bar* bars = reinterpret_cast<Bar*>(lm + B_LM_BAR);
  auto data = reinterpret_cast<volatile uint32_t*>(lm + B_LM_DATA);
  const uint8_t* gsrc = reinterpret_cast<const uint8_t*>(arg->in);
  if (t == 0) {
    init(&bars[0], B_THREADS);
    init(&bars[1], B_THREADS);
    init(&bars[2], B_THREADS);
    init(&bars[3], B_THREADS);
  }
  {
    auto g = reinterpret_cast<volatile uint32_t*>(lm + B_LM_G);
    for (uint32_t w = t; w < 1024; w += B_THREADS)
      g[w] = 0;
  }
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  __syncthreads();

  const uint32_t parts = arg->sub;
  for (uint32_t p = 0; p < B_PHASES; ++p)
    o[p] = ~0ull;
  for (uint32_t p = 0; p < B_PHASES && (parts & 1); ++p) {
    data[p * B_THREADS + t] = b_data(p, t);
    if (p == B_DROP && warp == B_WARPS - 1) {
      bars[0].arrive_and_drop();
      break;
    }
    if (p & 1) {
      bars[0].arrive_and_wait();
    } else {
      Bar::arrival_token tok = bars[0].arrive();
      bars[0].wait(static_cast<Bar::arrival_token&&>(tok));
    }
    const uint32_t live = p <= B_DROP ? B_THREADS : (B_WARPS - 1) * WARP;
    o[p] = data[p * B_THREADS + (t + 1) % live];
  }
  __syncthreads();

  o[B_PHASES] = cuda::device::barrier_native_handle(bars[1]) == reinterpret_cast<uint64_t*>(&bars[1]);
  for (uint32_t i = 0; i < B_NTX && (parts & 2); ++i) {
    const uint32_t n = b_tx_size(i);
    void* dst = lm + B_LM_TX + i * 4096;
    const void* src = gsrc + i * 4096;
    Bar::arrival_token tok;
    if (t == 0) {
      if (i == B_NTX - 1) {
        cuda::device::barrier_expect_tx(bars[1], n);
        cuda::device::memcpy_async_tx(dst, src, cuda::aligned_size_t<16>(n), bars[1]);
        tok = bars[1].arrive();
      } else {
        cuda::device::memcpy_async_tx(dst, src, cuda::aligned_size_t<16>(n), bars[1]);
        tok = cuda::device::barrier_arrive_tx(bars[1], 1, n);
      }
    } else {
      tok = bars[1].arrive();
    }
    bars[1].wait(static_cast<Bar::arrival_token&&>(tok));
  }

  if (t == 0 && (parts & 4)) {
    cuda::memcpy_async(lm + B_LM_MC, gsrc + 4 * 4096, B_MC_ALIGNED, bars[2]);
    cuda::memcpy_async(lm + B_MC_DST_UN, gsrc + B_MC_SRC_UN, B_MC_UNALIGNED, bars[2]);
    cuda::memcpy_async(lm + B_MC_DST_SZ, gsrc + B_MC_SRC_SZ, cuda::aligned_size_t<16>(B_MC_SIZED), bars[2]);
  }
  bars[2].arrive_and_wait();

  if (parts & 8) {
    cg::thread_block tb = cg::this_thread_block();
    uint8_t* g = lm + B_LM_G;
    cuda::memcpy_async(tb, g + B_G_AL, gsrc + B_G_AL_SRC, B_G_AL_N, bars[3]);
    cuda::memcpy_async(tb, g + B_G_UN, gsrc + B_G_UN_SRC, B_G_UN_N, bars[3]);
    if (t < 4)
      cuda::memcpy_async(g + B_G_PL + B_G_PL_N * t, gsrc + B_G_PL_SRC + B_G_PL_N * t, B_G_PL_N, bars[3]);
    if (t == 5)
      cuda::memcpy_async(g + B_G_ODD, gsrc + B_G_ODD_SRC, B_G_ODD_N, bars[3]);
    bars[3].arrive_and_wait();
  }
  o[B_PHASES + 1] = 0xC0DE;
  __syncthreads();
  dump_lmem(arg, B_LMEM);
}

// ── ptx: every cuda::ptx name with its effect ──────────────────────────
namespace {

namespace ptx = cuda::ptx;

// The cluster barrier in its three spellings, by every thread.
INL void csync(uint32_t k) {
  switch (k % 3) {
  case 0:  ptx::barrier_cluster_arrive(); ptx::barrier_cluster_wait(); break;
  case 1:  ptx::barrier_cluster_arrive(ptx::sem_release); ptx::barrier_cluster_wait(ptx::sem_acquire); break;
  default: ptx::barrier_cluster_arrive(ptx::sem_relaxed); ptx::barrier_cluster_wait(); break;
  }
}

INL void wait_parity(uint64_t* mb, uint32_t parity) {
  while (!ptx::mbarrier_try_wait_parity(mb, parity)) {}
}

// The field-by-field map of the 64 x 16 u32 tensor at `gaddr`.
INL void build_map(void* tm, uint64_t gaddr) {
  ptx::tensormap_replace_global_address(ptx::space_shared, tm, gaddr);
  ptx::tensormap_replace_rank(ptx::space_shared, tm, 1u);
  ptx::tensormap_replace_box_dim(ptx::space_shared, tm, ptx::n32_t<0>(), (uint32_t)X_BX);
  ptx::tensormap_replace_box_dim(ptx::space_shared, tm, ptx::n32_t<1>(), (uint32_t)X_BY);
  ptx::tensormap_replace_global_dim(ptx::space_shared, tm, ptx::n32_t<0>(), (uint32_t)X_TW);
  ptx::tensormap_replace_global_dim(ptx::space_shared, tm, ptx::n32_t<1>(), (uint32_t)X_TH);
  ptx::tensormap_replace_global_stride(ptx::space_shared, tm, ptx::n32_t<0>(), (uint64_t)(X_TW * 4));
  ptx::tensormap_replace_element_stride(ptx::space_shared, tm, ptx::n32_t<0>(), 1u);
  ptx::tensormap_replace_element_stride(ptx::space_shared, tm, ptx::n32_t<1>(), 1u);
  ptx::tensormap_replace_elemtype(ptx::space_shared, tm, ptx::n32_t<TMAP_ET_U32>());
  ptx::tensormap_replace_interleave_layout(ptx::space_shared, tm, ptx::n32_t<0>());
  ptx::tensormap_replace_swizzle_mode(ptx::space_shared, tm, ptx::n32_t<0>());
  ptx::tensormap_replace_fill_mode(ptx::space_shared, tm, ptx::n32_t<0>());
}

} // namespace

namespace {

struct PtxCtx {
  cuda_arg_t* arg;
  cg::cluster_group cl;
  uint32_t b, peer;
  uint64_t* o;
  uint8_t* lm;
  uint64_t* mb;
  const uint8_t* in;
  uint8_t* aux;
  uint8_t* remote(uint32_t off) const { return cl.map_shared_rank(lm + off, peer); }
  uint64_t* rmb(uint32_t i) const { return reinterpret_cast<uint64_t*>(remote(XL_MB + 8 * i)); }
};

INL void px_init(const PtxCtx& x) {
  const uint32_t t = threadIdx.x;
  auto lw = reinterpret_cast<volatile uint32_t*>(x.lm);
  for (uint32_t w = XL_G2S / 4 + t; w < X_LMEM / 4; w += X_THREADS)
    lw[w] = (w >= XL_SRC / 4 && w < XL_SRC / 4 + 256) ? x_src(x.b, w - XL_SRC / 4) : 0;
  __syncthreads();
  if (t == 0) {
    for (uint32_t w = 0; w < 64; ++w)
      lw[XL_RC / 4 + w] = x_rc_init(w);
    for (uint32_t k = 0; k < 11; ++k)
      lw[XL_RED / 4 + k] = (uint32_t)x_red_init(k);
    *reinterpret_cast<volatile uint64_t*>(x.lm + XL_RED + 48) = x_red_init(11);
    *reinterpret_cast<volatile uint64_t*>(x.lm + XL_RED + 56) = x_red_init(12);
    const uint32_t counts[X_NMB] = {X_THREADS, X_THREADS, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1};
    for (uint32_t i = 0; i < X_NMB; ++i)
      ptx::mbarrier_init(&x.mb[i], counts[i]);
    ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster);
  }
  ptx::fence_proxy_async(ptx::space_shared);
  ptx::fence_proxy_async();
}

// o[0..4]: local mbarrier phases in every arrive / wait spelling.
INL void px_local(const PtxCtx& x) {
  uint64_t* mb = x.mb;
  uint64_t* o = x.o;
  uint64_t s = ptx::mbarrier_arrive(&mb[0]);
  while (!ptx::mbarrier_try_wait(&mb[0], s)) {}
  o[0] = ptx::mbarrier_test_wait(&mb[0], s) && ptx::mbarrier_test_wait_parity(&mb[0], 0u);
  s = ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &mb[0]);
  while (!ptx::mbarrier_try_wait(ptx::sem_acquire, ptx::scope_cta, &mb[0], s)) {}
  o[1] = ptx::mbarrier_test_wait(ptx::sem_acquire, ptx::scope_cta, &mb[0], s)
      && ptx::mbarrier_test_wait_parity(ptx::sem_relaxed, ptx::scope_cta, &mb[0], 1u);
  s = ptx::mbarrier_arrive(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &mb[0], 1u);
  while (!ptx::mbarrier_try_wait_parity(ptx::sem_relaxed, ptx::scope_cta, &mb[0], 0u)) {}
  o[2] = ptx::mbarrier_test_wait(ptx::sem_relaxed, ptx::scope_cluster, &mb[0], s)
      && ptx::mbarrier_test_wait_parity(ptx::sem_acquire, ptx::scope_cluster, &mb[0], 0u);
  s = ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cluster, ptx::space_shared, &mb[0]);
  while (!ptx::mbarrier_try_wait(&mb[0], s, 1000u)) {}
  while (!ptx::mbarrier_try_wait(ptx::sem_relaxed, ptx::scope_cluster, &mb[0], s)) {}
  while (!ptx::mbarrier_try_wait_parity(&mb[0], 1u, 1000u)) {}
  while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cluster, &mb[0], 1u)) {}
  o[3] = !ptx::mbarrier_test_wait_parity(&mb[0], 0u);
  if (threadIdx.x % WARP == 0)
    (void)ptx::mbarrier_arrive(&mb[1], 32u);
  while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &mb[1], 0u, 1000u)) {}
  o[4] = ptx::mbarrier_test_wait_parity(&mb[1], 0u);
}

// Thread 0: bulk and tensor loads into the block, the expected bytes of the rest.
INL void px_loads(const PtxCtx& x) {
  uint64_t* mb = x.mb;
  (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &mb[2], 512u);
  ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, x.lm + XL_G2S, x.in + XG_G2S + x.b * 512, 256u, &mb[2]);
  ptx::cp_async_bulk(ptx::space_cluster, ptx::space_global, x.lm + XL_G2S + 256, x.in + XG_G2S + x.b * 512 + 256,
                     256u, &mb[2]);
  ptx::mbarrier_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &mb[3], (uint32_t)X_TILE);
  const int32_t c[2] = {(int32_t)(8 + 16 * x.b), (int32_t)(4 + x.b)};
  ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, x.lm + XL_TENS, &x.arg->map[0], c, &mb[3]);
  (void)ptx::mbarrier_arrive(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &mb[3]);
  (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &mb[4], 256u);
  (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cluster, ptx::space_shared, &mb[5],
                                       (uint32_t)X_TILE);
  (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &mb[6], 256u);
}

// Thread 0 of rank 0: the multicasts.
INL void px_mcast(const PtxCtx& x) {
  ptx::cp_async_bulk(ptx::space_cluster, ptx::space_global, x.lm + XL_MC, x.in + XG_MC, 256u, &x.mb[4],
                     (uint16_t)0x3);
  const int32_t c[2] = {0, 8};
  ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, x.lm + XL_TMC, &x.arg->map[0], c, &x.mb[5],
                            (uint16_t)0x3);
}

// Thread 0: the block's source words into the peer, the remote arrives.
INL void px_remote(const PtxCtx& x) {
  ptx::cp_async_bulk(ptx::space_cluster, ptx::space_shared, x.remote(XL_S2C), x.lm + XL_SRC, 256u, x.rmb(6));
  ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cluster, ptx::space_cluster, x.rmb(9), 256u);
  ptx::cp_reduce_async_bulk(ptx::space_cluster, ptx::space_shared, ptx::op_add,
                            reinterpret_cast<uint32_t*>(x.remote(XL_RC)),
                            reinterpret_cast<const uint32_t*>(x.lm + XL_SRC), 256u, x.rmb(9));
  ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cluster, ptx::space_cluster, x.rmb(8));
  ptx::mbarrier_arrive(ptx::sem_relaxed, ptx::scope_cluster, ptx::space_cluster, x.rmb(11), 1u);
  (void)ptx::mbarrier_arrive(ptx::sem_release, ptx::scope_cluster, ptx::space_shared, &x.mb[8]);
}

// Warp 0: st.async and red.async into the peer's words, on its mbarrier 7.
INL void px_async(const PtxCtx& x) {
  const uint32_t w = x.b, lane = threadIdx.x % WARP;
  uint64_t* rb = x.rmb(7);
  ptx::st_async(reinterpret_cast<uint32_t*>(x.remote(XL_ST + 4 * lane)), mix32(w * 100 + lane, 0x51), rb);
  if (lane < 8)
    ptx::st_async(reinterpret_cast<uint64_t*>(x.remote(XL_ST + 128 + 8 * lane)), mix64(w * 100 + lane, 0x52), rb);
  if (lane < 4) {
    const uint32_t v[2] = {mix32(w * 100 + lane, 0x53), mix32(w * 100 + lane, 0x54)};
    ptx::st_async(reinterpret_cast<uint32_t*>(x.remote(XL_ST + 192 + 8 * lane)), v, rb);
  }
  if (lane < 2) {
    const uint32_t v[4] = {mix32(w * 100 + lane, 0x55), mix32(w * 100 + lane, 0x56), mix32(w * 100 + lane, 0x57),
                           mix32(w * 100 + lane, 0x58)};
    ptx::st_async(reinterpret_cast<uint32_t*>(x.remote(XL_ST + 224 + 16 * lane)), v, rb);
  }
  auto ru = [&](uint32_t k) { return reinterpret_cast<uint32_t*>(x.remote(XL_RED + 4 * k)); };
  auto rs = [&](uint32_t k) { return reinterpret_cast<int32_t*>(x.remote(XL_RED + 4 * k)); };
  auto val = [&](uint32_t k) { return (uint32_t)x_red_value(k, w, lane); };
  ptx::red_async(ptx::op_add, ru(0), val(0), rb);
  ptx::red_async(ptx::op_min, ru(1), val(1), rb);
  ptx::red_async(ptx::op_max, ru(2), val(2), rb);
  ptx::red_async(ptx::op_inc, ru(3), val(3), rb);
  ptx::red_async(ptx::op_dec, ru(4), val(4), rb);
  ptx::red_async(ptx::op_min, rs(5), (int32_t)val(5), rb);
  ptx::red_async(ptx::op_max, rs(6), (int32_t)val(6), rb);
  ptx::red_async(ptx::op_add, rs(7), (int32_t)val(7), rb);
  ptx::red_async(ptx::op_and_op, ru(8), val(8), rb);
  ptx::red_async(ptx::op_or_op, ru(9), val(9), rb);
  ptx::red_async(ptx::op_xor_op, ru(10), val(10), rb);
  ptx::red_async(ptx::op_add, reinterpret_cast<uint64_t*>(x.remote(XL_RED + 48)), x_red_value(11, w, lane), rb);
  ptx::red_async(ptx::op_add, reinterpret_cast<int64_t*>(x.remote(XL_RED + 56)), (int64_t)x_red_value(12, w, lane),
                 reinterpret_cast<int64_t*>(rb));
}

// Thread 0: bulk and tensor stores and reductions into global memory, a read back, the prefetches.
INL void px_global(const PtxCtx& x) {
  uint8_t* aux = x.aux;
  const uint32_t b = x.b;
  ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, aux + XA_STORE + b * 512, x.lm + XL_SRC, 512u);
  ptx::cp_reduce_async_bulk(ptx::space_global, ptx::space_shared, ptx::op_add,
                            reinterpret_cast<uint32_t*>(aux + XA_RED), reinterpret_cast<const uint32_t*>(x.lm + XL_SRC),
                            256u);
  const int32_t cs[2] = {(int32_t)(16 * b), 0};
  ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared, &x.arg->map[1], cs, x.lm + XL_SRC);
  const int32_t cr[2] = {(int32_t)(32 + 16 * b), 4};
  ptx::cp_reduce_async_bulk_tensor(ptx::space_global, ptx::space_shared, ptx::op_add, &x.arg->map[1], cr,
                                   x.lm + XL_SRC + X_TILE);
  ptx::cp_async_bulk_commit_group();
  ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  ptx::cp_async_bulk_wait_group(ptx::n32_t<0>());
  ptx::fence_proxy_async(ptx::space_global);
  auto st = reinterpret_cast<volatile uint32_t*>(aux + XA_STORE + b * 512);
  uint32_t sum = 0;
  for (uint32_t w = 0; w < 128; ++w)
    sum = sum * 31 + st[w];
  reinterpret_cast<uint32_t*>(aux + XA_FENCE)[b] = sum;
  ptx::cp_async_bulk_prefetch(x.in + XG_PF, 1024u);
  const int32_t cp[2] = {0, 0};
  ptx::cp_async_bulk_prefetch_tensor(ptx::space_global, &x.arg->map[0], cp);
  ptx::prefetch_tensormap(&x.arg->map[0]);
}

// Thread 0 of rank 0: a map from a zero buffer in shared memory, copied to
// global memory with cp_fenceproxy, acquired, loaded through; then replaced
// in global memory, released, acquired, loaded through again.
INL void px_tmap(const PtxCtx& x) {
  void* tm = x.lm + XL_MAP;
  void* gm = x.aux + XA_MAP;
  uint64_t* mb = x.mb;
  build_map(tm, (uint64_t)(uintptr_t)(x.in + XG_T2));
  ptx::tensormap_cp_fenceproxy(ptx::sem_release, ptx::scope_cta, gm, tm, ptx::n32_t<128>());
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_cta, gm, ptx::n32_t<128>());
  const int32_t c[2] = {16, 2};
  (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &mb[10],
                                       (uint32_t)X_TILE);
  ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, x.lm + XL_TR, gm, c, &mb[10]);
  wait_parity(&mb[10], 0);
  ptx::tensormap_replace_global_address(ptx::space_global, gm, (uint64_t)(uintptr_t)(x.in + XG_T3));
  ptx::fence_proxy_tensormap_generic(ptx::sem_release, ptx::scope_gpu);
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_gpu, gm, ptx::n32_t<128>());
  (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &mb[10],
                                       (uint32_t)X_TILE);
  ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, x.lm + XL_TR2, gm, c, &mb[10]);
  wait_parity(&mb[10], 1);
}


INL PtxCtx ptx_ctx(cuda_arg_t* arg) {
  PtxCtx x;
  x.arg = arg;
  x.cl = cg::this_cluster();
  x.b = x.cl.block_rank();
  x.peer = x.b ^ 1;
  x.o = out_of(arg);
  x.lm = reinterpret_cast<uint8_t*>(__local_mem());
  x.mb = reinterpret_cast<uint64_t*>(x.lm + XL_MB);
  x.in = reinterpret_cast<const uint8_t*>(arg->in);
  x.aux = reinterpret_cast<uint8_t*>(arg->aux);
  return x;
}

INL void ptx_done(const PtxCtx& x) {
  x.o[6] = 0xC0DE;
  __syncthreads();
  csync(1);
  dump_lmem(x.arg, XL_DUMP);
  csync(2);
}

} // namespace

// The cuda::ptx names in four kernels of a cluster of 2: mbarriers, copies,
// st/red.async, global stores and tensor maps.
__global__ void __cluster_dims__(2, 1, 1) ptx_mbar_k(cuda_arg_t* __UNIFORM__ arg) {
  VX_CUDA_KERNEL_ENTRY(ptx_mbar_k);
  const PtxCtx x = ptx_ctx(arg);
  px_init(x);
  csync(1);
  px_local(x);
  x.o[5] = ptx::getctarank(ptx::space_cluster, x.remote(0)) == x.peer
        && ptx::getctarank(ptx::space_cluster, x.lm) == x.b;
  ptx_done(x);
}

__global__ void __cluster_dims__(2, 1, 1) ptx_copy_k(cuda_arg_t* __UNIFORM__ arg) {
  VX_CUDA_KERNEL_ENTRY(ptx_copy_k);
  const PtxCtx x = ptx_ctx(arg);
  const uint32_t t = threadIdx.x;
  px_init(x);
  csync(1);
  if (t == 0)
    px_loads(x);
  csync(2);
  if (t == 0 && x.b == 0)
    px_mcast(x);
  if (t == 0)
    px_remote(x);
  for (uint32_t i = 2; i <= 6; ++i)
    wait_parity(&x.mb[i], 0);
  wait_parity(&x.mb[8], 0);
  wait_parity(&x.mb[9], 0);
  while (!ptx::mbarrier_try_wait_parity(ptx::sem_relaxed, ptx::scope_cluster, &x.mb[11], 0u)) {}
  ptx_done(x);
}

__global__ void __cluster_dims__(2, 1, 1) ptx_async_k(cuda_arg_t* __UNIFORM__ arg) {
  VX_CUDA_KERNEL_ENTRY(ptx_async_k);
  const PtxCtx x = ptx_ctx(arg);
  const uint32_t t = threadIdx.x;
  px_init(x);
  csync(1);
  if (t == 0)
    ptx::mbarrier_expect_tx(ptx::sem_relaxed, ptx::scope_cluster, ptx::space_cluster, x.rmb(7), (uint32_t)X_TX_ST);
  csync(2);
  if (t / WARP == 0)
    px_async(x);
  csync(0);
  if (t == 0)
    (void)ptx::mbarrier_arrive(&x.mb[7]);
  wait_parity(&x.mb[7], 0);
  ptx_done(x);
}

__global__ void __cluster_dims__(2, 1, 1) ptx_global_k(cuda_arg_t* __UNIFORM__ arg) {
  VX_CUDA_KERNEL_ENTRY(ptx_global_k);
  const PtxCtx x = ptx_ctx(arg);
  const uint32_t t = threadIdx.x;
  px_init(x);
  csync(1);
  if (t == 0)
    px_global(x);
  if (t == 0 && x.b == 0)
    px_tmap(x);
  ptx_done(x);
}

// ── sync: __syncthreads_count/and/or, __syncwarp ordering ───────────────
static_assert(S_RESERVE == VX_CUDA_SYNC_RESERVE, "the counter reserve of __syncthreads_count");

__global__ void sync_k(cuda_arg_t* __UNIFORM__ arg) {
  const uint32_t t = threadIdx.x;
  uint64_t* o = out_of(arg);
  uint32_t k = 0;
  for (uint32_t r = 0; r < S_ROUNDS; ++r) {
    for (uint32_t q = 0; q < S_NPRED; ++q) {
      const int p = (int)s_pred(q, r, t);
      o[k++] = (uint32_t)__syncthreads_count(p);
      o[k++] = (uint32_t)__syncthreads_and(p);
      o[k++] = (uint32_t)__syncthreads_or(p);
    }
  }
  auto gw = reinterpret_cast<volatile uint32_t*>(arg->aux);
  auto lw = reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uint8_t*>(__local_mem()) + 1024);
  for (uint32_t r = 0; r < S_SWROUNDS; ++r) {
    gw[r * S_THREADS + t] = s_sw_value(r, t, 0);
    lw[r * S_THREADS + t] = s_sw_value(r, t, 1);
    __syncwarp();
    o[k++] = gw[r * S_THREADS + (t ^ 1)];
    o[k++] = lw[r * S_THREADS + (t ^ 1)];
  }
}

// The block reductions with 8 bytes of the kernel's own data: a
// cuda::barrier at offset 0, used after them.
__global__ void sync8_k(cuda_arg_t* __UNIFORM__ arg) {
  typedef cuda::barrier<cuda::thread_scope_block> Bar;
  const uint32_t t = threadIdx.x;
  uint64_t* o = out_of(arg);
  Bar* bar = reinterpret_cast<Bar*>(__local_mem());
  if (t == 0)
    init(bar, S_THREADS);
  __syncthreads();
  uint32_t k = 0;
  for (uint32_t r = 0; r < S_ROUNDS; ++r) {
    for (uint32_t q = 0; q < S_NPRED; ++q) {
      const int p = (int)s_pred(q, r, t);
      o[k++] = (uint32_t)__syncthreads_count(p);
      o[k++] = (uint32_t)__syncthreads_and(p);
      o[k++] = (uint32_t)__syncthreads_or(p);
    }
  }
  bar->arrive_and_wait();
  bar->arrive_and_wait();
  o[k++] = *reinterpret_cast<volatile uint64_t*>(bar);
  o[k++] = 0xC0DE;
}

// ── warp, exited lanes: threads 48-63 of a 64-thread block return at once ─
__global__ void tail_k(cuda_arg_t* __UNIFORM__ arg) {
  const uint32_t t = threadIdx.x;
  if (t >= WT_THREADS)
    return;
  const uint32_t lane = t % WARP;
  uint64_t* o = out_of(arg);
  for (uint32_t op = 0; op < W_NROP; ++op)
    o[WT_RED + op] = reduce_op(op, 0xffffffffu, w_rvalue(0, t));
  int pred = -1;
  o[WT_MATCH + 0] = __match_any_sync(0xffffffffu, (unsigned)w_key(0, 2, lane));
  o[WT_MATCH + 1] = __match_any_sync(0xffffffffu, (ull)w_key(1, 2, lane));
  o[WT_MATCH + 2] = __match_all_sync(0xffffffffu, (unsigned)w_key(0, 0, lane), &pred);
  o[WT_MATCH + 3] = (uint32_t)pred;
  pred = -1;
  o[WT_MATCH + 4] = __match_all_sync(0xffffffffu, (ull)w_key(1, 1, lane), &pred);
  o[WT_MATCH + 5] = (uint32_t)pred;
  const unsigned v = (unsigned)w_value(0, t);
  o[WT_VOTE + 0] = __ballot_sync(0xffffffffu, (int)w_pred(3, t));
  o[WT_VOTE + 1] = __activemask();
  o[WT_VOTE + 2] = __shfl_sync(0xffffffffu, v, (int)((lane + 5) % WARP));
  o[WT_VOTE + 3] = __shfl_xor_sync(0xffffffffu, v, 1);
  o[WT_VOTE + 4] = __shfl_down_sync(0xffffffffu, v, 8u);
}

// ── types: the number types on the device ───────────────────────────────
__global__ void types_k(cuda_arg_t* __UNIFORM__ arg) {
  const uint32_t nt = blockDim.x * gridDim.x;
  const uint32_t t = threadIdx.x + blockDim.x * blockIdx.x;
  auto o = reinterpret_cast<uint32_t*>(arg->aux);
  for (uint32_t i = t; i < T_INPUTS; i += nt)
    types_ops::eval_input(i, o + i * T_OPS);
}

// ── launch: a kernel with __cluster_dims__ ───────────────────────────────
__global__ void __cluster_dims__(2, 1, 1) launch_k(cuda_arg_t* __UNIFORM__ arg) {
  VX_CUDA_KERNEL_ENTRY(launch_k);
  cg::thread_block tb = cg::this_thread_block();
  const uint32_t br = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  auto w = reinterpret_cast<uint32_t*>(arg->aux) + br * L_WORDS;
  atomicAdd(&w[6], 1u);
  if (tb.thread_rank() == 0) {
    cg::cluster_group cl = cg::this_cluster();
    w[0] = cl.dim_blocks().x;
    w[1] = cl.dim_blocks().y;
    w[2] = cl.dim_blocks().z;
    w[3] = cl.block_rank();
    w[4] = vx_is_explicit_cluster();
    w[5] = vx_dynamic_smem_size();
    w[7] = 1;
  }
}

// ── launch: a templated kernel, exported twice by name ───────────────────
template <int K>
__device__ void scale_k(cuda_arg_t* arg) {
  const uint32_t t = threadIdx.x + blockDim.x * blockIdx.x;
  reinterpret_cast<uint64_t*>(arg->out)[t] = (uint64_t)K * 1000 + t;
}
VX_CUDA_KERNEL_EXPORT(scale2_k, scale_k<2>)
VX_CUDA_KERNEL_EXPORT(scale3_k, scale_k<3>)

// ── launch: the other CUDA names, smoke use ─────────────────────────────
struct __align__(16) aligned16_t {
  uint32_t w[3];
};
static_assert(alignof(aligned16_t) == 16, "__align__(16)");
static_assert(sizeof(uint4) == 16 && alignof(uint4) == 16 && alignof(float4) == 16, "CUDA vector types");

__global__ void names_k(cuda_arg_t* __UNIFORM__ arg) {
  const uint32_t t = threadIdx.x;
  uint64_t* o = out_of(arg);
  uint8_t* lm = reinterpret_cast<uint8_t*>(__local_mem());
  const uint32_t x = n_word(t);
  uint32_t k = 0;
  o[k++] = (uint32_t)__popc(x);
  o[k++] = (uint32_t)__popcll(((unsigned long long)x << 32) | x);
  o[k++] = (uint32_t)__ffs((int)x);
  o[k++] = (uint32_t)__clz((int)(x >> (t % 32)));
  o[k++] = __brev(x);
  o[k++] = (uint64_t)max((int)x, -5);
  o[k++] = (uint64_t)min(x, 1000u);
  o[k++] = (uint64_t)max((long long)t, 7ll);
  o[k++] = to_bits(min(1.5f, (float)t));
  o[k++] = to_bits(max(2.5, (double)t));
  const uint4 u = make_uint4(x, x + 1, x + 2, x + 3);
  o[k++] = u.x + u.y + u.z + u.w;
  o[k++] = __cvta_generic_to_shared(lm + 64) - __cvta_generic_to_shared(lm);
  o[k++] = __cvta_generic_to_global(reinterpret_cast<void*>(arg->in)) == (size_t)arg->in;
  const long long c0 = clock64();
  __threadfence_block();
  __threadfence();
  __threadfence_system();
  __nanosleep(20);
  const long long c1 = clock64();
  o[k++] = c1 > c0;
  // cuda_pipeline: one 16-byte copy per thread, then wait for it.
  __pipeline_memcpy_async(lm + 16 * (t % 16), reinterpret_cast<const uint8_t*>(arg->in) + 16 * (t % 16), 16);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();
  o[k++] = reinterpret_cast<volatile uint32_t*>(lm)[t % 64];
  aligned16_t a;
  a.w[0] = t;
  o[k++] = ((uintptr_t)&a & 15) == 0;
  // atomicAdd(float4*) from every thread into one global float4.
  float4 v = make_float4(1.0f, 2.0f, 0.5f, (float)(t % 4));
  atomicAdd(reinterpret_cast<float4*>(arg->aux), v);
  o[k++] = 0xC0DE;
}

// ── perf (-p): warp reductions, shuffles and integer atomics ────────────
__global__ void perf_k(cuda_arg_t* __UNIFORM__ arg) {
  const uint32_t t = threadIdx.x;
  uint64_t* o = out_of(arg);
  auto lm = reinterpret_cast<unsigned*>(__local_mem());
  auto g = reinterpret_cast<unsigned*>(arg->aux);
  if (t < 16)
    lm[t] = 0;
  __syncthreads();
  unsigned v = mix32(grid_thread(), 1), acc = 0, mx = 0;
  for (uint32_t r = 0; r < arg->rounds; ++r) {
    v = __shfl_xor_sync(0xffffffffu, v, (int)(1 + r % 31)) + r;
    v ^= __shfl_down_sync(0xffffffffu, v, 3);
    acc += __reduce_add_sync(0xffffffffu, v);
    mx = __reduce_max_sync(0xffffffffu, v ^ acc);
    atomicAdd(&g[r % 8], v & 0xff);
    atomicMax(&g[8 + r % 8], v);
    atomicAdd(&lm[r % 8], v & 0xff);
    atomicXor(&lm[8 + r % 8], mx ^ v);
  }
  o[0] = acc;
  o[1] = mx;
  o[2] = v;
  __syncthreads();
  if (t < 16)
    g[16 + blockIdx.x * 16 + t] = lm[t];
}
