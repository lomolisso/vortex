#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

// Warp 0 streams DXA tile copies into region A of local memory while every
// other warp hammers atomics on region B. The regions never share a word but
// do share every bank: each DXA beat writes a full bank row, and each hammer
// instruction puts its lanes on distinct banks. An atomic's read-modify-write
// must survive a DXA beat landing inside it.

constexpr uint32_t kDescSrc = 0;

// One bank word per slot, so consecutive slots walk the banks. On RV64 odd
// slots sit in the upper 32-bit lane of their word.
static inline uint32_t* lmem_slot(uint8_t* base, uint32_t k) {
  constexpr uint32_t word = sizeof(size_t);
  return reinterpret_cast<uint32_t*>(base + k * word + (k % (word / 4)) * 4);
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t threads  = vx_num_threads();
  const uint32_t hammers  = blockDim.x - threads;
  const uint32_t tid      = threadIdx.x;

  auto tile = reinterpret_cast<uint32_t*>(__local_mem());
  auto ctrs = reinterpret_cast<uint8_t*>(tile) + arg->tile_elems * sizeof(uint32_t);

  const uint32_t swap_base = hammers;
  const uint32_t ladd_base = 2 * hammers;
  const uint32_t lxor_base = ladd_base + threads;
  const uint32_t done_slot = lxor_base + threads;
  volatile uint32_t* done = lmem_slot(ctrs, done_slot);

  for (uint32_t k = tid; k <= done_slot; k += blockDim.x) {
    *lmem_slot(ctrs, k) = 0;
  }
  __syncthreads();

  if (get_sub_group_id() == 0) {
    // Private to warp 0, so the hammer warps never wait on a copy.
    vortex::barrier copy_bar(1, 1);
    for (uint32_t r = 0; r < arg->num_copies; ++r) {
      const uint32_t src_tile = (r + blockIdx.x) % arg->num_src_tiles;
      copy_bar.expect_tx(1);
      vx_dxa_issue_1d_wg(kDescSrc, copy_bar.id(), tile, src_tile * arg->tile_elems);
      copy_bar.arrive_and_wait();
    }
    // Hammer progress by the time the last copy landed: zero means the
    // atomics never overlapped the DXA writes and the run proves nothing.
    if (tid == 0) {
      uint32_t overlap = 0;
      for (uint32_t h = 0; h < hammers; ++h) {
        overlap += *static_cast<volatile uint32_t*>(lmem_slot(ctrs, h));
      }
      reinterpret_cast<uint32_t*>(arg->overlap_out_addr)[blockIdx.x] = overlap;
    }
    *done = 1;
  } else {
    const uint32_t h    = tid - threads;
    const uint32_t lane = tid % threads;
    uint32_t* add  = lmem_slot(ctrs, h);
    uint32_t* swp  = lmem_slot(ctrs, swap_base + h);
    uint32_t* ladd = lmem_slot(ctrs, ladd_base + lane);
    uint32_t* lxor = lmem_slot(ctrs, lxor_base + lane);

    const uint32_t min_iters = arg->min_iters;

    // Only this hart writes `add` and `swp`, so each returned old value is
    // known exactly: a dropped write-back shows up on the very next access.
    // `add` is checked against its own previous return so one dropped
    // write-back counts once rather than skewing every later iteration.
    uint32_t bad = 0;
    uint32_t next_add = 0;
    uint32_t i = 0;
    do {
      const uint32_t old_add = __atomic_fetch_add(add, 1u, __ATOMIC_RELAXED);
      bad += (old_add != next_add);
      next_add = old_add + 1;
      bad += (__atomic_exchange_n(swp, i + 1, __ATOMIC_RELAXED) != i);
      __atomic_fetch_add(ladd, h + 1, __ATOMIC_RELAXED);
      __atomic_fetch_xor(lxor, xor_pattern(h, i), __ATOMIC_RELAXED);
      ++i;
    } while (i < min_iters || *done == 0);

    auto out = reinterpret_cast<hammer_out_t*>(arg->hammer_out_addr) + blockIdx.x * hammers + h;
    out->add     = *add;
    out->swap    = *swp;
    out->iters   = i;
    out->bad_old = bad;
  }
  __syncthreads();

  auto tile_out = reinterpret_cast<uint32_t*>(arg->tile_out_addr) + blockIdx.x * arg->tile_elems;
  for (uint32_t k = tid; k < arg->tile_elems; k += blockDim.x) {
    tile_out[k] = tile[k];
  }
  if (tid < threads) {
    auto out = reinterpret_cast<lane_out_t*>(arg->lane_out_addr) + blockIdx.x * threads + tid;
    out->add     = *lmem_slot(ctrs, ladd_base + tid);
    out->xor_acc = *lmem_slot(ctrs, lxor_base + tid);
  }
}
