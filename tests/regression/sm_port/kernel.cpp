#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#if VX_CFG_EXT_DLMEM_ENABLED
#include <vx_dlmem.h>
#endif
#if VX_CFG_EXT_DXA_ENABLED && VX_CFG_EXT_MBAR_ENABLED
#include <vx_mbar.h>
#include <vx_dxa.h>
#define SP_HAS_TMA 1
#endif

#include "common.h"

#define INL static inline __attribute__((always_inline))

#define R4(x) x x x x

INL uint64_t opaque(uint64_t v) {
  __asm__ volatile ("" : "+r"(v));
  return v;
}

template <bool WIDE, uint32_t O>
INL void store_at(uint64_t v, uint64_t p, uint64_t q) {
  if constexpr (O < 2048) {
    if constexpr (WIDE) __asm__ volatile ("sd %0, %2(%1)" :: "r"(v), "r"(p), "i"(O) : "memory");
    else                __asm__ volatile ("sw %0, %2(%1)" :: "r"(v), "r"(p), "i"(O) : "memory");
  } else {
    if constexpr (WIDE) __asm__ volatile ("sd %0, %2(%1)" :: "r"(v), "r"(q), "i"(O - 2048) : "memory");
    else                __asm__ volatile ("sw %0, %2(%1)" :: "r"(v), "r"(q), "i"(O - 2048) : "memory");
  }
}

// N stores at p + O, p + O + STEP, ...; q = p + 2048 reaches the upper half of a 4-KB chunk.
template <bool WIDE, uint32_t O, uint32_t STEP, uint32_t N>
INL void store_run(uint64_t v, uint64_t p, uint64_t q) {
  if constexpr (N > 0) {
    store_at<WIDE, O>(v, p, q);
    store_run<WIDE, O + STEP, STEP, N - 1>(v, p, q);
  }
}

// Stores at base + off, the base stepping by STEP per instruction within span.
// Immediate offsets over 4-KB chunks keep the address arithmetic off the issue
// slots, so the stores issue faster than the port drains them; the fence makes
// the end time include the drain of every store.
template <bool WIDE, uint32_t STEP>
INL void store_loop(uint64_t region, uint32_t off, uint32_t iters, uint32_t span) {
  constexpr uint32_t CHUNK = 4096, G = CHUNK / STEP;
  const uint64_t v = opaque(1);
  const uint64_t start = region + off;
  uint64_t left = (uint64_t)(iters / G) * CHUNK;
  while (left) {
    const uint64_t n = (left < span) ? left : span;
    left -= n;
    for (uint64_t p = start, e = start + n; p != e; p += CHUNK) {
      store_run<WIDE, 0, STEP, G>(v, p, p + 2048);
    }
  }
}

// The test's steps: one line per instruction (128, or 64 with 64-B lines), 256 (8-B lanes),
// one sector per line of 8 (1024, or 512 with 64-B lines).
template <bool WIDE>
INL void store_loop(uint64_t region, uint32_t off, uint32_t iters, uint32_t step, uint32_t span) {
  if (step == 128)
    store_loop<WIDE, 128>(region, off, iters, span);
  else if (step == 256)
    store_loop<WIDE, 256>(region, off, iters, span);
  else if (step == 64)
    store_loop<WIDE, 64>(region, off, iters, span);
  else if (step == 512)
    store_loop<WIDE, 512>(region, off, iters, span);
  else
    store_loop<WIDE, 1024>(region, off, iters, span);
}

INL void record(const kernel_arg_t* arg, uint64_t t0, uint64_t t1) {
  if (vx_thread_id() == 0) {
    auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
    uint32_t w = blockIdx.x * vx_num_warps() + vx_warp_id();
    out[w * 2 + 0] = t0;
    out[w * 2 + 1] = t1;
  }
}

#ifdef SP_HAS_TMA
// TMA_CHUNKS bulk loads of TMA_BYTES per phase of the mbarrier; the first
// `warm` phases pull the source into the L2 and are not timed.
#define TMA_BYTES  4096
#define TMA_CHUNKS 4
INL void tma_run(const kernel_arg_t* arg, char* own, uint32_t rank, uint32_t* phase, uint64_t* out) {
  uint64_t* mbar = reinterpret_cast<uint64_t*>(own);
  char* dst = own + 128;
  const uint64_t src = arg->gmem_addr + (uint64_t)rank * TMA_CHUNKS * TMA_BYTES * 4;
  const uint32_t warm = 4, reps = arg->iters;
  uint64_t t0 = 0;
  for (uint32_t r = 0; r < warm + reps; ++r) {
    if (r == warm)
      t0 = vx_rdcycle();
    if (vx_thread_id() == 0)
      vx_mbar_arrive_tx(mbar, TMA_CHUNKS * TMA_BYTES);
    vx_fence_proxy_async();
    const uint64_t s = src + (uint64_t)(r % 4) * TMA_CHUNKS * TMA_BYTES;
    for (uint32_t c = 0; c < TMA_CHUNKS; ++c)
      vx_dxa_bulk_load(dst + c * TMA_BYTES, reinterpret_cast<const void*>(s + c * TMA_BYTES), TMA_BYTES, mbar);
    while (!vx_mbar_try_wait_parity(mbar, *phase & 1)) {
    }
    ++*phase;
  }
  uint64_t t1 = vx_rdcycle();
  if (vx_thread_id() == 0) {
    out[0] = t0;
    out[1] = t1;
  }
}
#endif

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t lane = vx_thread_id();
  const uint32_t wid  = vx_warp_id();
  const uint32_t off  = reinterpret_cast<const uint32_t*>(arg->pat_addr)[lane];
  const bool active   = (off != 0xFFFFFFFFu);
  const uint32_t iters = arg->iters, step = arg->step, span = arg->span;

  if (arg->kind == SP_GSTORE) {
    __syncthreads();
    uint64_t t0 = vx_rdcycle();
    if (active) {
      store_loop<false>(arg->gmem_addr, off, iters, step, span);
    }
    vx_fence();
    record(arg, t0, vx_rdcycle());
    return;
  }

  if (arg->kind == SP_RW) {
    // Every warp's full-line stores feed the one FIFO faster than it drains;
    // warp 0's load then sits behind all of them.
    auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
    const uint64_t cold0 = arg->gmem_addr + span;
    const uint64_t cold1 = cold0 + 4096;
    const uint32_t per = arg->n / vx_num_warps();
    uint64_t v = 0, t0 = 0, t1 = 0;
    if (wid == 0) {
      // The load alone first, on a line nothing has touched.
      t0 = vx_rdcycle();
      if (lane == 0) {
        __asm__ volatile ("lw %0, 0(%1)\n add %0, %0, zero\n" : "=&r"(v) : "r"(cold0) : "memory");
      }
      t1 = vx_rdcycle();
      vx_fence();
    }
    __syncthreads();
    uint64_t t2 = vx_rdcycle();
    const uint64_t s = opaque(1);
    for (uint64_t p = arg->gmem_addr + (uint64_t)wid * per * 128 + lane * 4, e = p + (uint64_t)per * 128; p != e; p += 1024) {
      store_run<false, 0, 128, 8>(s, p, p + 2048);
    }
    __syncthreads();
    if (wid != 0) {
      return;
    }
    uint64_t t3 = vx_rdcycle();
    if (lane == 0) {
      __asm__ volatile ("lw %0, 0(%1)\n add %0, %0, zero\n" : "=&r"(v) : "r"(cold1) : "memory");
    }
    uint64_t t4 = vx_rdcycle();
    vx_fence();
    if (lane == 0) {
      out[0] = t1 - t0;  // the load alone
      out[1] = t3 - t2;  // issuing every warp's stores
      out[2] = t4 - t2;  // stores issued until the load behind them is usable
      out[3] = v;
    }
    return;
  }

#if VX_CFG_EXT_DLMEM_ENABLED
  char* own = reinterpret_cast<char*>(__local_mem());
  const uint32_t rank = vx_cluster_block_rank();
  vx_cluster_sync();
  uint64_t t0 = vx_rdcycle();
  switch (arg->kind) {
  case SP_DSTORE:
    if (rank != arg->recv && active) {
      uint64_t win = reinterpret_cast<uint64_t>(vx_cluster_map_shared_rank(own, arg->recv));
      store_loop<true>(win, off, iters, step, span);
    }
    break;
  case SP_DLOAD:
    if (rank == 0 && active) {
      uint64_t win = reinterpret_cast<uint64_t>(vx_cluster_map_shared_rank(own, 1));
      uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, sink;
      uint32_t b = 0;
      for (uint32_t i = 0; i < iters; i += 4) {
        uint64_t a0 = win + ((b + 0 * step) & (span - 1)) + off;
        uint64_t a1 = win + ((b + 1 * step) & (span - 1)) + off;
        uint64_t a2 = win + ((b + 2 * step) & (span - 1)) + off;
        uint64_t a3 = win + ((b + 3 * step) & (span - 1)) + off;
        __asm__ volatile ("ld %0, 0(%4)\n ld %1, 0(%5)\n ld %2, 0(%6)\n ld %3, 0(%7)\n"
                          : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3)
                          : "r"(a0), "r"(a1), "r"(a2), "r"(a3) : "memory");
        b += 4 * step;
      }
      __asm__ volatile ("add %0, %1, %2\n add %0, %0, %3\n add %0, %0, %4\n"
                        : "=&r"(sink) : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
      __asm__ volatile ("" :: "r"(sink));
    }
    break;
  case SP_DATOM:
    if (rank == 0 && active) {
      uint64_t win = reinterpret_cast<uint64_t>(vx_cluster_map_shared_rank(own, 1));
      uint64_t one = opaque(1), r0, r1, r2, r3, sink;
      uint32_t b = 0;
      for (uint32_t i = 0; i < iters; i += 4) {
        uint64_t a0 = win + ((b + 0 * step) & (span - 1)) + off;
        uint64_t a1 = win + ((b + 1 * step) & (span - 1)) + off;
        uint64_t a2 = win + ((b + 2 * step) & (span - 1)) + off;
        uint64_t a3 = win + ((b + 3 * step) & (span - 1)) + off;
        __asm__ volatile ("amoadd.w %0, %4, (%5)\n amoadd.w %1, %4, (%6)\n amoadd.w %2, %4, (%7)\n amoadd.w %3, %4, (%8)\n"
                          : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3)
                          : "r"(one), "r"(a0), "r"(a1), "r"(a2), "r"(a3) : "memory");
        b += 4 * step;
      }
      __asm__ volatile ("add %0, %1, %2\n add %0, %0, %3\n add %0, %0, %4\n"
                        : "=&r"(sink) : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
      __asm__ volatile ("" :: "r"(sink));
    }
    break;
  case SP_MIX:
    if (rank == 0 && active) {
      if ((wid & 1) == 0) {
        uint64_t win = reinterpret_cast<uint64_t>(vx_cluster_map_shared_rank(own, 1));
        store_loop<true>(win, off, iters, step, span);
      } else {
        store_loop<true>(arg->gmem_addr, off, iters, step, span);
      }
    }
    break;
#ifdef SP_HAS_TMA
  case SP_TMA: {
    auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
    uint32_t phase = 0;
    if (wid == 0 && lane == 0)
      vx_mbar_init(reinterpret_cast<uint64_t*>(own), 1);
    __syncthreads();
    for (uint32_t ph = 0; ph < 2; ++ph) {
      if (wid == 0 && (ph == 1 || rank == 0))
        tma_run(arg, own, rank, &phase, out + (ph * 2 + rank) * 2);
      vx_cluster_sync();
    }
    return;
  }
#endif
  default:
    break;
  }
  vx_fence_cluster();
  record(arg, t0, vx_rdcycle());
  vx_cluster_sync();
#endif
}
