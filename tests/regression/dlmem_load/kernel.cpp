#include <vx_spawn2.h>
#include <vx_dlmem.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

struct Ctx {
  uint32_t k, rank, c, b, tid, nt, n, seed;
  volatile uint32_t* own;
  vortex::cluster_group cl;
  uint32_t errors, first_test, first_index;
  uint64_t first_got, first_want;

  template <typename T>
  INL volatile T* window(uint32_t q) {
    return reinterpret_cast<volatile T*>(cl.map_shared_rank(const_cast<uint32_t*>(own), q));
  }

  INL void check(uint32_t test, uint32_t index, uint64_t got, uint64_t want) {
    if (got != want && errors++ == 0) {
      first_test  = test;
      first_index = index;
      first_got   = got;
      first_want  = want;
    }
  }
};

INL void ctx_init(Ctx& x, const load_arg_t* arg) {
  x.k     = arg->k;
  x.rank  = vx_cluster_block_rank();
  x.b     = blockIdx.x;
  x.c     = x.b / x.k;
  x.tid   = threadIdx.x;
  x.nt    = blockDim.x;
  x.n     = arg->lmem_bytes / sizeof(uint32_t);
  x.seed  = arg->seed;
  x.own   = reinterpret_cast<volatile uint32_t*>(__local_mem());
  x.errors = x.first_test = x.first_index = 0;
  x.first_got = x.first_want = 0;
  if (x.tid == 0) {
    auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + x.b * REC_WORDS;
    rec[REC_RANK] = x.rank;
    rec[REC_SIZE] = vx_cluster_num_blocks();
    rec[REC_CORE] = vx_core_id();
    rec[REC_SLOT] = get_local_group_id();
    rec[REC_RAN]  = 1;
  }
  x.check(TEST_SETUP, 0, vx_cluster_num_blocks(), x.k);
}

INL void ctx_done(const Ctx& x, const load_arg_t* arg) {
  auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + (x.b * x.nt + x.tid) * ERR_WORDS;
  err[ERR_TEST]      = x.first_test;
  err[ERR_INDEX]     = x.first_index;
  err[ERR_GOT]       = (uint32_t)x.first_got;
  err[ERR_GOT + 1]   = (uint32_t)(x.first_got >> 32);
  err[ERR_WANT]      = (uint32_t)x.first_want;
  err[ERR_WANT + 1]  = (uint32_t)(x.first_want >> 32);
  err[ERR_COUNT]     = x.errors;
}

// The doubleword at p read as one load, two signed words, four unsigned halves
// or eight signed bytes, each piece checked with its extension.
INL void check_split(Ctx& x, uint32_t kind, const volatile uint8_t* p, uint64_t want, uint32_t index) {
  switch (kind) {
  case 0:
    x.check(TEST_ALL, index, *reinterpret_cast<const volatile uint64_t*>(p), want);
    break;
  case 1:
    for (uint32_t h = 0; h < 2; ++h) {
      const int64_t got = *reinterpret_cast<const volatile int32_t*>(p + 4 * h);
      x.check(TEST_ALL, index | (h << 20), (uint64_t)got, (uint64_t)(int64_t)(int32_t)(want >> (32 * h)));
    }
    break;
  case 2:
    for (uint32_t h = 0; h < 4; ++h) {
      const uint64_t got = *reinterpret_cast<const volatile uint16_t*>(p + 2 * h);
      x.check(TEST_ALL, index | (h << 20), got, (uint16_t)(want >> (16 * h)));
    }
    break;
  default:
    for (uint32_t h = 0; h < 8; ++h) {
      const int64_t got = *reinterpret_cast<const volatile int8_t*>(p + h);
      x.check(TEST_ALL, index | (h << 20), (uint64_t)got, (uint64_t)(int64_t)(int8_t)(want >> (8 * h)));
    }
    break;
  }
}

// Ping-pong word: the round in the low half, a check of writer and round above.
INL uint64_t pp_word(uint32_t seed, uint32_t c, uint32_t q, uint32_t r) {
  return ((uint64_t)pattern(seed, TEST_PP, c, q, r) << 32) | r;
}

}  // namespace

// Fill the own slice, sync, then read sampled doublewords of every slice (the
// own one last, through the window) at every width.
__kernel void all_k(load_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t n8 = x.n / 2;
  auto own8 = reinterpret_cast<volatile uint64_t*>(x.own);
  for (uint32_t i = x.tid; i < n8; i += x.nt)
    own8[i] = pattern64(x.seed, TEST_ALL, x.c, x.rank, i);
  x.cl.sync();
  const uint32_t M = check_count(n8);
  for (uint32_t s = 1; s <= x.k; ++s) {
    const uint32_t q = (x.rank + s) % x.k;
    auto w = x.window<uint8_t>(q);
    for (uint32_t j = 0;; ++j) {
      const uint32_t m = x.tid + x.nt * j;
      if (m >= M)
        break;
      const uint32_t i = check_index(m, n8);
      check_split(x, (j + q) % 4u, w + 8 * i, pattern64(x.seed, TEST_ALL, x.c, q, i), (q << 24) | i);
    }
  }
  // Peers may still be reading this slice.
  x.cl.sync();
  ctx_done(x, arg);
}

// Ranks 2i and 2i + 1, on two cores, pass a round number back and forth: each
// polls the other's doubleword through the window and answers in its own.
__kernel void pingpong_k(load_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  auto own8 = reinterpret_cast<volatile uint64_t*>(x.own);
  if (x.tid == 0)
    own8[0] = 0;
  x.cl.sync();
  const uint32_t partner = x.rank ^ 1u;
  if (partner < x.k && x.tid == 0) {
    auto wp = x.window<uint64_t>(partner);
    const bool ping = (x.rank & 1u) == 0;
    for (uint32_t r = 1; r <= arg->rounds; ++r) {
      if (ping)
        own8[0] = pp_word(x.seed, x.c, x.rank, r);
      uint32_t spins = 0;
      uint64_t v;
      while ((uint32_t)(v = wp[0]) < r && ++spins < SPIN_LIMIT) {}
      x.check(TEST_PP, r << 24, v, pp_word(x.seed, x.c, partner, r));
      if (!ping)
        own8[0] = pp_word(x.seed, x.c, x.rank, r);
    }
  }
  x.cl.sync();
  ctx_done(x, arg);
}

// One word load and one doubleword load whose lanes split three ways:
// global, own local memory, and the window on another rank.
__kernel void mixed_k(load_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  for (uint32_t e = x.tid; e < x.n; e += x.nt)
    x.own[e] = pattern(x.seed, TEST_MIXED, x.c, x.rank, e);
  x.cl.sync();
  const uint32_t n8 = x.n / 2;
  const uint32_t kind = mixed_kind(x.tid);
  const uint32_t q = mixed_peer(x.rank, x.tid, x.k);
  const uint32_t e8 = mixed_cell(x.rank, x.tid, n8);
  const uint32_t h = x.tid & 1u;
  const uint32_t g = x.b * x.nt + x.tid;
  auto glob = reinterpret_cast<uint32_t*>(arg->glob_addr);
  // Branch-free, so the lanes stay in one instruction.
  const uintptr_t m0 = (uintptr_t)0 - (uintptr_t)(kind == 0);
  const uintptr_t m1 = (uintptr_t)0 - (uintptr_t)(kind == 1);
  const uintptr_t m2 = (uintptr_t)0 - (uintptr_t)(kind == 2);
  const uintptr_t pw = ((uintptr_t)(glob + 2 * g + h) & m0)
                     | ((uintptr_t)(x.own + 2 * e8 + h) & m1)
                     | ((uintptr_t)(x.window<uint32_t>(q) + 2 * e8 + h) & m2);
  const uintptr_t pd = ((uintptr_t)(glob + 2 * g) & m0)
                     | ((uintptr_t)(x.own + 2 * e8) & m1)
                     | ((uintptr_t)(x.window<uint32_t>(q) + 2 * e8) & m2);
  uint32_t got32;
  uint64_t got64;
  asm volatile ("lwu %0, 0(%1)" : "=r"(got32) : "r"(pw) : "memory");
  asm volatile ("ld %0, 0(%1)" : "=r"(got64) : "r"(pd) : "memory");
  const uint32_t src = (kind == 1) ? x.rank : q;
  uint32_t lo, hi;
  if (kind == 0) {
    lo = glob_word(x.seed, 2 * g);
    hi = glob_word(x.seed, 2 * g + 1);
  } else {
    lo = pattern(x.seed, TEST_MIXED, x.c, src, 2 * e8);
    hi = pattern(x.seed, TEST_MIXED, x.c, src, 2 * e8 + 1);
  }
  x.check(TEST_MIXED, (kind << 24) | (src << 16) | e8, got32, h ? hi : lo);
  x.check(TEST_MIXED, (kind << 24) | (src << 16) | (1u << 15) | e8, got64, ((uint64_t)hi << 32) | lo);
  x.cl.sync();
  ctx_done(x, arg);
}

// Each thread stores into its own doubleword of the next rank's slice and reads
// it straight back, with no fence: whole, then over a half, a byte and a
// halfword it just wrote. The owner checks the last value after the sync.
__kernel void raw_k(load_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t n8 = x.n / 2;
  const uint32_t next = (x.rank + 1) % x.k;
  const uint32_t prev = (x.rank + x.k - 1) % x.k;
  if (x.tid < n8) {
    auto p = x.window<uint8_t>(next) + 8 * x.tid;
    auto p64 = reinterpret_cast<volatile uint64_t*>(p);
    for (uint32_t r = 0; r < arg->rounds; ++r) {
      const uint32_t idx = (r << 24) | x.tid;
      uint64_t v = pattern64(x.seed, TEST_RAW, x.c, x.rank, (r << 8) | x.tid);
      *p64 = v;
      x.check(TEST_RAW, idx, *p64, v);
      const uint32_t hw = pattern(x.seed, TEST_RAW, x.c, x.rank, (r << 8) | 0x80u | x.tid);
      *reinterpret_cast<volatile uint32_t*>(p + 4) = hw;
      v = (v & 0xFFFFFFFFull) | ((uint64_t)hw << 32);
      x.check(TEST_RAW, idx | (1u << 20), *p64, v);
      p[1] = (uint8_t)(hw >> 8);
      v = (v & ~0xFF00ull) | ((uint64_t)((hw >> 8) & 0xFFu) << 8);
      x.check(TEST_RAW, idx | (2u << 20), *reinterpret_cast<volatile uint32_t*>(p), (uint32_t)v);
      *reinterpret_cast<volatile uint16_t*>(p + 6) = (uint16_t)~hw;
      v = (v & 0x0000FFFFFFFFFFFFull) | ((uint64_t)(uint16_t)~hw << 48);
      x.check(TEST_RAW, idx | (3u << 20), *reinterpret_cast<volatile uint16_t*>(p + 6), (uint16_t)(v >> 48));
      x.check(TEST_RAW, idx | (4u << 20), *p64, v);
    }
  }
  x.cl.sync();
  if (x.tid < n8) {
    const uint32_t r = arg->rounds - 1;
    const uint32_t hw = pattern(x.seed, TEST_RAW, x.c, prev, (r << 8) | 0x80u | x.tid);
    uint64_t v = pattern64(x.seed, TEST_RAW, x.c, prev, (r << 8) | x.tid);
    v = (v & 0xFFFFFFFFull) | ((uint64_t)hw << 32);
    v = (v & ~0xFF00ull) | ((uint64_t)((hw >> 8) & 0xFFu) << 8);
    v = (v & 0x0000FFFFFFFFFFFFull) | ((uint64_t)(uint16_t)~hw << 48);
    x.check(TEST_RAW, (prev << 24) | (1u << 23) | x.tid,
            reinterpret_cast<volatile uint64_t*>(x.own)[x.tid], v);
  }
  ctx_done(x, arg);
}

// Fixed loops of window loads from the next rank's slice between two
// fence-free cluster barriers: no spin and no data-dependent trip count, so
// both models retire the same count.
__kernel void parity_k(load_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  for (uint32_t e = x.tid; e < x.n; e += x.nt)
    x.own[e] = parity_fill(x.seed, x.c, x.rank, e);
  x.cl.barrier_wait_relaxed(x.cl.barrier_arrive_relaxed());
  auto w = x.window<uint32_t>((x.rank + 1) % x.k);
  uint32_t sum = 0;
  for (uint32_t j = 0; j < arg->rounds; ++j)
    sum += w[(j * x.nt + x.tid) % x.n];
  x.cl.barrier_wait_relaxed(x.cl.barrier_arrive_relaxed());
  reinterpret_cast<uint32_t*>(arg->glob_addr)[x.b * x.nt + x.tid] = sum;
  ctx_done(x, arg);
}
