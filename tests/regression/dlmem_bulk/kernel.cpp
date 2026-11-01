#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dlmem.h>
#include <vx_dxa.h>
#include <vx_mbar.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

INL void idle_cycles(uint64_t n) {
  const uint64_t t0 = vx_rdcycle();
  while (vx_rdcycle() - t0 < n) {}
}

INL void record(const bulk_arg_t* arg, uint32_t b, uint32_t rank) {
  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * REC_WORDS;
  rec[REC_BLOCK] = b;
  rec[REC_CORE]  = vx_core_id();
  rec[REC_SLOT]  = get_local_group_id();
  rec[REC_RANK]  = rank;
  rec[REC_SIZE]  = vx_cluster_num_blocks();
  __atomic_fetch_add(&rec[REC_RAN], 1u, __ATOMIC_RELAXED);
}

// Waits for phase `parity`; true when it did not complete within WAIT_LIMIT try_waits.
INL bool wait_phase(uint64_t* mbar, uint32_t parity, uint32_t* tries) {
  ++*tries;
  while (!vx_mbar_try_wait_parity(mbar, parity)) {
    if (++*tries > WAIT_LIMIT)
      return true;
  }
  return false;
}

INL void dump_lmem(const bulk_arg_t* arg, uint32_t b, uint32_t tid, uint32_t nthr) {
  auto lw = reinterpret_cast<volatile uint32_t*>(__local_mem());
  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr + (uint64_t)b * arg->lmem_bytes);
  for (uint32_t w = tid; w < arg->lmem_bytes / 4; w += nthr)
    dump[w] = lw[w];
}

INL void finish(const bulk_arg_t* arg, uint32_t b, uint32_t tid, uint32_t nthr, uint32_t err, uint32_t tries) {
  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + (b * nthr + tid) * ERR_WORDS;
  out[ERR_WAIT]  = err;
  out[ERR_TRIES] = tries;
  out[ERR_DONE]  = 1;
}

// The reference path of -S: the warp copies with its own loads and stores,
// fences them and applies the copy's complete_tx itself.
INL void soft_copy(void* dst, const void* src, uint32_t size, uint64_t* mbar, uint32_t nt) {
  auto d = reinterpret_cast<volatile uint32_t*>(dst);
  auto s = reinterpret_cast<const volatile uint32_t*>(src);
  for (uint32_t w = threadIdx.x % nt; w < size / 4; w += nt)
    d[w] = s[w];
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (threadIdx.x % nt == 0)
    vx_mbar_complete_tx(mbar, size);
}

INL void bulk_copy(const bulk_arg_t* arg, void* dst, const void* src, uint32_t size, uint64_t* mbar) {
  if (arg->soft)
    soft_copy(dst, src, size, mbar, arg->warp_threads);
  else
    vx_dxa_bulk_copy(dst, src, size, mbar);
}

// Rounds of: warp 0 arms the mbarrier, stores the round's words into the
// source, optionally fences, and copies the source into the round's
// destination slot at once, then waits for the phase; meanwhile the other
// warps load and store their own words in a loop, on every bank, until warp 0
// raises the flag. With a hog, rank 0 instead copies the hog's bytes of its
// unchanged source each round, once the racer of its cluster is ready, and
// when that copy holds the DMA port lets the round go; the racer then
// stores and copies at once. Both counters are global words of the cluster.
INL void race(const bulk_arg_t* arg, uint32_t b, uint32_t r, uint32_t tid, uint32_t nthr) {
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto lw   = reinterpret_cast<volatile uint32_t*>(lm);
  auto mbar = reinterpret_cast<uint64_t*>(lm + arg->lm_mbar);
  auto flag = reinterpret_cast<volatile uint32_t*>(lm + lm_flag(arg));
  const uint32_t nt = arg->warp_threads;
  const uint32_t so = src_off(arg, r, 0);
  volatile uint32_t* go    = reinterpret_cast<volatile uint32_t*>(arg->sync_addr) + 2 * (b / arg->k);
  volatile uint32_t* ready = go + 1;
  uint32_t err = 0, tries = 0;

  if (get_sub_group_id() == 0 && is_hog(arg, r)) {
    // Thread 0 alone arms, copies and lets the round go.
    for (uint32_t j = 0; j < arg->rounds && !err; ++j) {
      while (*ready <= j) {}
      if (tid == 0) {
        vx_mbar_arrive_tx(mbar, arg->hog);
        vx_dxa_bulk_copy(lm + dst_off(arg, r, 0), lm + so, arg->hog, mbar);
        idle_cycles(HOG_LEAD);
        *go = j + 1;
      }
      err = wait_phase(mbar, j & 1, &tries);
    }
    *flag = 1;
  } else if (get_sub_group_id() == 0 && arg->hog != 0) {
    // The round's word of each thread and the copy's operands are ready
    // before the round goes, so that the store follows the go at once.
    const uint32_t size = arg->size, rounds = arg->rounds, fence = arg->fence;
    for (uint32_t j = 0; j < rounds && !err; ++j) {
      const uint32_t v = race_word(b, j, so / 4 + tid);
      uint8_t* dst = lm + dst_off(arg, r, j);
      if (tid == 0) {
        vx_mbar_arrive_tx(mbar, size);
        *ready = j + 1;
      }
      while (*go <= j) {}
      lw[so / 4 + tid] = v;
      if (fence)
        vx_fence_proxy_async();
      vx_dxa_bulk_copy(dst, lm + so, size, mbar);
      err = wait_phase(mbar, j & 1, &tries);
    }
    *flag = 1;
  } else if (get_sub_group_id() == 0) {
    for (uint32_t j = 0; j < arg->rounds && !err; ++j) {
      if (tid == 0)
        vx_mbar_arrive_tx(mbar, arg->size);
      for (uint32_t w = so / 4 + tid; w < (so + arg->size) / 4; w += nt)
        lw[w] = race_word(b, j, w);
      if (arg->fence)
        vx_fence_proxy_async();
      bulk_copy(arg, lm + dst_off(arg, r, j), lm + so, arg->size, mbar);
      err = wait_phase(mbar, j & 1, &tries);
    }
    *flag = 1;
  } else {
    volatile uint32_t* ham = lw + lm_ham(arg) / 4 + (tid - nt);
    const uint32_t v = ham_word(b, tid - nt);
    while (*flag == 0) {
      *ham = v;
      (void)*ham;
    }
  }
  __syncthreads();
  dump_lmem(arg, b, tid, nthr);
  vortex::this_cluster().sync();
  finish(arg, b, tid, nthr, err, tries);
}

} // namespace

// Each block fills its local memory with canaries and its source ranges,
// initializes its mbarrier and arrives on it declaring the bytes of the
// copies it will receive, fences its stores for the async proxy, all before
// the cluster barrier. Then warp 0 issues its copies into the destination
// block's slots, completing on that block's mbarrier; every thread waits for
// its own block's phase, and the block dumps its whole local memory for the
// host's byte-exact check.
__kernel void copy_k(bulk_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t nthr = blockDim.x;
  const uint32_t b    = blockIdx.x;
  const uint32_t k    = arg->k;
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t r    = cl.block_rank();
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto lw   = reinterpret_cast<volatile uint32_t*>(lm);
  auto mbar = reinterpret_cast<uint64_t*>(lm + arg->lm_mbar);
  const uint32_t words = arg->lmem_bytes / sizeof(uint32_t);

  if (tid == 0)
    record(arg, b, r);
  // Thread w % nthr writes word w: its canary, then any source value over it.
  for (uint32_t w = tid; w < words; w += nthr) {
    if (w / 2 != arg->lm_mbar / 8)
      lw[w] = canary_word(b, w);
  }
  for (uint32_t i = 0; i < num_src_slots(arg) && arg->mode != MODE_RACE; ++i) {
    const uint32_t w0 = src_off(arg, r, i) / 4, w1 = w0 + copy_size(arg, r, i) / 4;
    for (uint32_t w = w0 + (tid + nthr - w0 % nthr) % nthr; w < w1; w += nthr)
      lw[w] = src_word(b, w);
  }
  if (tid == 0) {
    vx_mbar_init(mbar, 1);
    // A hog's round counter starts in the data cache, so that the racers
    // see each round go at once.
    auto sync = reinterpret_cast<volatile uint32_t*>(arg->sync_addr) + 2 * (b / k);
    if (arg->mode == MODE_RACE && arg->hog != 0 && (sync[0] | sync[1]) != 0)
      lw[lm_flag(arg) / 4] = 1;
    if (arg->mode == MODE_RACE)
      lw[lm_flag(arg) / 4] = 0;
    else
      vx_mbar_arrive_tx(mbar, incoming(arg, r));
  }
  if (arg->fence)
    vx_fence_proxy_async();
  cl.sync();

  if (arg->mode == MODE_RACE) {
    race(arg, b, r, tid, nthr);
    return;
  }

  if (get_sub_group_id() == 0) {
    const uint32_t q = dst_rank(arg->mode, r, k);
    for (uint32_t i = 0; i < arg->copies; ++i) {
      void* src = lm + src_off(arg, r, i);
      void* dst = lm + dst_off(arg, r, i);
      const uint32_t size = copy_size(arg, r, i);
      if (arg->mode == MODE_SELF)
        bulk_copy(arg, dst, src, size, mbar);
      else
        bulk_copy(arg, cl.map_shared_rank(dst, q), src, size, cl.map_shared_rank(mbar, q));
    }
  }

  uint32_t tries = 0;
  const uint32_t err = wait_phase(mbar, 0, &tries);
  dump_lmem(arg, b, tid, nthr);
  cl.sync();
  finish(arg, b, tid, nthr, err, tries);
}

// Runs the simulator must stop. Rank 0 issues one copy; rank 1 stays alive
// meanwhile, except in -x retired, where it exits at once and rank 0 issues
// into it only after lingering.
__kernel void neg_k(bulk_arg_t* __UNIFORM__ arg) {
  const uint32_t tid = threadIdx.x;
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t r = cl.block_rank();
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto mbar = reinterpret_cast<uint64_t*>(lm + arg->lm_mbar);
  auto src  = lm + LM_SRC;
  auto dst  = lm + lm_dst(arg);

  if (tid == 0)
    record(arg, blockIdx.x, r);
  if (arg->neg == NEG_RETIRED && r != 0)
    return;
  if (tid == 0)
    vx_mbar_init(mbar, 1);
  __syncthreads();
  if (arg->neg == NEG_RETIRED)
    idle_cycles(arg->linger);
  if (r == 0 && get_sub_group_id() == 0) {
    auto peer_mbar = cl.map_shared_rank(mbar, 1);
    auto peer_dst  = cl.map_shared_rank(dst, 1);
    const uint32_t size = arg->size;
    switch (arg->neg) {
    case NEG_SIZE:
      vx_dxa_bulk_copy(peer_dst, src, 24, peer_mbar);
      break;
    case NEG_ALIGN:
      vx_dxa_bulk_copy(peer_dst, src + 8, size, peer_mbar);
      break;
    case NEG_DALIGN:
      vx_dxa_bulk_copy(peer_dst + 8, src, size, peer_mbar);
      break;
    case NEG_SRC:
      vx_dxa_bulk_copy(peer_dst, lm + arg->neg_off, size, peer_mbar);
      break;
    case NEG_RANK: {
      auto bad = reinterpret_cast<uint8_t*>((uintptr_t)VX_MEM_DLMEM_BASE_ADDR + arg->neg_win);
      vx_dxa_bulk_copy(bad + lm_dst(arg), src, size, reinterpret_cast<uint64_t*>(bad + arg->lm_mbar));
      break;
    }
    case NEG_MBAR:
      vx_dxa_bulk_copy(peer_dst, src, size, cl.map_shared_rank(mbar, 0));
      break;
    case NEG_BOX:
      vx_dxa_bulk_copy(cl.map_shared_rank(lm, 1) + arg->neg_off, src, size, peer_mbar);
      break;
    default:
      vx_dxa_bulk_copy(peer_dst, src, size, peer_mbar);
      break;
    }
  }
  idle_cycles(arg->linger);
  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + (blockIdx.x * blockDim.x + tid) * ERR_WORDS;
  out[ERR_DONE] = 1;
}
