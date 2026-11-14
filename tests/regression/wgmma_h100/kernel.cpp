// Blocks of 1 or 2 warpgroups, each block with its own operands and outputs.
// The operand tiles are placed in the block's local memory either by plain stores of a host-made image (each element at
// the address the descriptor golden gives it) or by DXA tensor loads with the
// swizzle on an mbarrier; then each warpgroup runs a chain of wgmmas on its
// descriptors, from initial accumulators the host gave per thread and
// register, and writes every whole 64-bit register back as it is. In ring
// mode a producer warp refills two stages through mbarriers while the
// warpgroup consumes them.

#include <vx_cuda.h>
#include "common.h"

namespace h = vortex::h100;
namespace vt = vortex::tensor;

namespace {

__attribute__((noinline)) void fill_sw(const kernel_arg_t* arg, char* lm) {
  const uint32_t* img = (const uint32_t*)(arg->img_addr + (uint64_t)blockIdx.x * arg->img_stride);
  uint32_t* dst = (uint32_t*)(lm + arg->fill_lo);
  const uint32_t words = (arg->fill_hi - arg->fill_lo) / 4;
  for (uint32_t i = threadIdx.x; i < words; i += blockDim.x)
    dst[i] = img[i];
  // The tensor unit reads through the async proxy, as on the H100.
  cuda::device::experimental::fence_proxy_async_shared_cta();
  __syncthreads();
}

__attribute__((noinline)) void fill_tma(kernel_arg_t* arg, char* lm) {
  namespace ptx = cuda::ptx;
  uint64_t* mbar = (uint64_t*)(lm + arg->mbar_off);
  if (threadIdx.x == 0)
    ptx::mbarrier_init(mbar, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, mbar, arg->tx_bytes);
    const box_t* boxes = (const box_t*)arg->box_addr;
    for (uint32_t i = 0; i < arg->nboxes; ++i) {
      const box_t b = boxes[i];
      const int32_t c[2] = {(int32_t)b.c0, (int32_t)(b.c1 + blockIdx.x * arg->blk_outer[b.map])};
      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, lm + b.dst, &arg->map[b.map], c, mbar);
    }
  }
  while (!ptx::mbarrier_try_wait_parity(mbar, 0))
    ;
}

// Ring mode, the block's warp 4: refills stage i % 2 for wgmma i once the
// consumers have released it (empty), with that wgmma's slice of k.
__attribute__((noinline)) void ring_producer(kernel_arg_t* arg, char* lm) {
  namespace ptx = cuda::ptx;
  uint64_t* full = (uint64_t*)(lm + kRingMbar);
  uint64_t* empty = full + 2;
  if (__laneid() != 0)
    return;
  const box_t* boxes = (const box_t*)arg->box_addr;
  for (uint32_t i = 0; i < arg->chain; ++i) {
    const uint32_t s = i & 1;
    if (i >= 2) {
      while (!ptx::mbarrier_try_wait_parity(empty + s, ((i >> 1) - 1) & 1))
        ;
    }
    (void)ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, full + s,
                                         arg->ring_tx[s]);
    for (uint32_t j = 0; j < arg->ring_box_n[s]; ++j) {
      const box_t b = boxes[arg->ring_box_lo[s] + j];
      int32_t c[2] = {(int32_t)b.c0, (int32_t)(b.c1 + blockIdx.x * arg->blk_outer[b.map])};
      c[arg->kdim[b.map]] += (int32_t)(i * arg->ki);
      ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, lm + b.dst, &arg->map[b.map], c, full + s);
    }
  }
}

inline __attribute__((always_inline)) void spin_cycles(uint32_t n) {
  const uint64_t t0 = csr_read(VX_CSR_MCYCLE);
  while (csr_read(VX_CSR_MCYCLE) - t0 < n)
    ;
}

// C > 0: a chain of exactly C wgmmas, unrolled, so the accumulators stay in
// f0.. from the first wgmma to the last (no stack copies between them).
// C == 0: the chain length from the argument block, in a loop (and ring mode).
template <int N, typename Dt, typename At, bool TA, bool TB, int C>
__attribute__((noinline)) void run(const kernel_arg_t* arg, char* lm) {
  using Acc = h::acc<N, Dt>;
  const uint32_t g = threadIdx.x / 128, t = threadIdx.x % 128;
  // Register-major [b][g][r][t]: the 32 lanes of a warp touch consecutive words.
  const size_t slot = (size_t)blockIdx.x * kSlotsPerCta + (size_t)g * kMaxRegs * 128 + t;
  const uint32_t ga = arg->distinct_a ? g : 0;
  const uint32_t* c = (const uint32_t*)arg->c_addr + slot;
  const uint32_t chain = C ? (uint32_t)C : arg->chain;
  const h::swz mode = (h::swz)arg->layout;

  auto desc_a = [&](uint32_t i) { return h::make_desc(lm + arg->a_start[ga][i], mode, arg->lbo_a, arg->sbo_a); };
  auto desc_b = [&](uint32_t i) { return h::make_desc(lm + arg->b_start[g][i], mode, arg->lbo_b, arg->sbo_b); };
  if (t == 0) {
    uint64_t* out = (uint64_t*)arg->desc_addr + (size_t)blockIdx.x * kDescsPerCta + (size_t)g * kMaxChain * 2;
    for (uint32_t i = 0; i < chain; ++i) {
      out[2 * i] = desc_a(i);
      out[2 * i + 1] = desc_b(i);
    }
  }

  // Constant register indices throughout, so the fragment never needs a home
  // in memory.
  Acc d;
  vt::detail::unroll_for<Acc::NRC>([&](auto r) __attribute__((always_inline)) {
    d.data[r] = __builtin_bit_cast(float, c[(size_t)r * 128]);
  });

  namespace ptx = cuda::ptx;
  uint64_t* full = (uint64_t*)(lm + kRingMbar);
  uint64_t* empty = full + 2;
  const uint32_t warp = threadIdx.x / 32;
  const bool ring = arg->ring != 0;
  // One wgmma of the chain. Ring mode: stage i % 2 holds wgmma i's operands;
  // warps 1-3 start their wgmma late, and only thread 0 releases the stage
  // after wait_group 0, so the wait must cover the whole warpgroup's wgmma
  // (as on the H100).
  auto step = [&](uint32_t i) __attribute__((always_inline)) {
    if (C == 0 && ring) {
      const uint32_t s = i & 1;
      while (!ptx::mbarrier_try_wait_parity(full + s, (i >> 1) & 1))
        ;
      if (warp != 0)
        spin_cycles(arg->skew);
      h::wgmma<N, Dt, At, At, TA, TB>(desc_a(i), desc_b(i), d, !(arg->zero_first && i == 0));
      wgmma_commit_group();
      wgmma_wait_group<0>();
      if (threadIdx.x == 0)
        (void)ptx::mbarrier_arrive(empty + s);
      // Keep the four warps in the same round, so a stage released too early
      // shows up as wrong values (warps one mbarrier phase apart), not as a
      // wait on a phase parity that has already come round again.
      bar_sync(1, 128);
    } else {
      h::wgmma<N, Dt, At, At, TA, TB>(desc_a(i), desc_b(i), d, !(arg->zero_first && i == 0));
      wgmma_commit_group();
      if (i != 0)
        wgmma_wait_group<1>();
    }
  };

  wgmma_fence();
  if constexpr (C > 0) {
    vt::detail::unroll_for<C>([&](auto i) __attribute__((always_inline)) { step((uint32_t)i); });
  } else {
    for (uint32_t i = 0; i < chain; ++i)
      step(i);
  }
  wgmma_wait_group<0>();

  uint64_t* out = (uint64_t*)arg->d_addr + slot;
  uint32_t* frag = (uint32_t*)arg->frag_addr + slot;
  vt::detail::unroll_for<Acc::NRC>([&](auto r) __attribute__((always_inline)) {
    out[(size_t)r * 128] = d.bits64(r);
    frag[(size_t)r * 128] = (Acc::row(t, r) << 16) | Acc::col(t, r, 0);
  });
}

// The chain lengths the catalog uses run unrolled; any other length, and the
// ring (whose timing is set by its skew), loop.
template <int N, typename Dt, typename At, bool TA, bool TB>
void by_chain(const kernel_arg_t* arg, char* lm) {
  switch (arg->ring ? 0 : arg->chain) {
  case 2: run<N, Dt, At, TA, TB, 2>(arg, lm); break;
  case 4: run<N, Dt, At, TA, TB, 4>(arg, lm); break;
  case 6: run<N, Dt, At, TA, TB, 6>(arg, lm); break;
  case 8: run<N, Dt, At, TA, TB, 8>(arg, lm); break;
  default: run<N, Dt, At, TA, TB, 0>(arg, lm); break;
  }
}

template <typename Dt, typename At, bool TA, bool TB>
__attribute__((noinline)) void by_n(const kernel_arg_t* arg, char* lm) {
  constexpr int u = std::is_same_v<Dt, vt::fp16> ? 2 : 1;   // N per 8 registers: 16 or 32
  switch (arg->n) {
  case 16 * u: by_chain<16 * u, Dt, At, TA, TB>(arg, lm); break;
  case 32 * u: by_chain<32 * u, Dt, At, TA, TB>(arg, lm); break;
  case 64 * u: by_chain<64 * u, Dt, At, TA, TB>(arg, lm); break;
  default: break;
  }
}

template <typename Dt, typename At>
void by_major(const kernel_arg_t* arg, char* lm) {
  if constexpr (At::bits == 16) {
    switch (arg->tnsp_a * 2 + arg->tnsp_b) {
    case 0: by_n<Dt, At, false, false>(arg, lm); break;
    case 1: by_n<Dt, At, false, true>(arg, lm); break;
    case 2: by_n<Dt, At, true, false>(arg, lm); break;
    default: by_n<Dt, At, true, true>(arg, lm); break;
    }
  } else {
    by_n<Dt, At, false, false>(arg, lm);
  }
}

__attribute__((noinline)) void body(kernel_arg_t* arg) {
  char* lm = (char*)__local_mem();
  if (threadIdx.x == 0) {
    place_t* pl = (place_t*)arg->place_addr + blockIdx.x;
    pl->core = (uint32_t)vx_core_id();
    pl->slot = (uint32_t)(csr_read(VX_CSR_CTA_LMEM_ADDR) - csr_read(VX_CSR_LOCAL_MEM_BASE));
    pl->warp = (uint32_t)vx_warp_id();
  }
  if (arg->ring) {
    if (threadIdx.x == 0) {
      uint64_t* full = (uint64_t*)(lm + kRingMbar);
      for (uint32_t s = 0; s < 4; ++s)
        cuda::ptx::mbarrier_init(full + s, 1);
    }
    __syncthreads();
    if (threadIdx.x / 32 == 4) {
      ring_producer(arg, lm);
      return;
    }
  } else if (arg->tma) {
    fill_tma(arg, lm);
  } else {
    fill_sw(arg, lm);
  }
  switch (arg->pair) {
  case PAIR_F16F32: by_major<vt::fp32, vt::fp16>(arg, lm); break;
  case PAIR_F16F16: by_major<vt::fp16, vt::fp16>(arg, lm); break;
  case PAIR_BF16F32: by_major<vt::fp32, vt::bf16>(arg, lm); break;
  case PAIR_E4M3F32: by_major<vt::fp32, vt::fp8>(arg, lm); break;
  default: by_major<vt::fp32, vt::bf8>(arg, lm); break;
  }
}

} // namespace

__global__ void wgmma_h100(kernel_arg_t* arg) {
  body(arg);
}

// The same test with the blocks in clusters of 2.
__global__ void __cluster_dims__(2, 1, 1) wgmma_h100_c2(kernel_arg_t* arg) {
  VX_CUDA_KERNEL_ENTRY(wgmma_h100_c2);
  body(arg);
}
