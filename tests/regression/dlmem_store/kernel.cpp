#include <vx_spawn2.h>
#include <vx_dlmem.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

struct Ctx {
  uint32_t k, rank, c, b, tid, nt, n, seed, warps;
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

  // A block barrier only between warps: one warp's lanes order themselves.
  INL void block_sync() {
    if (warps > 1)
      __syncthreads();
  }
};

INL void ctx_init(Ctx& x, const store_arg_t* arg) {
  x.k     = arg->k;
  x.rank  = vx_cluster_block_rank();
  x.b     = blockIdx.x;
  x.c     = x.b / x.k;
  x.tid   = threadIdx.x;
  x.nt    = blockDim.x;
  x.n     = arg->lmem_bytes / sizeof(uint32_t);
  x.seed  = arg->seed;
  x.warps = get_num_sub_groups();
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

INL void ctx_done(const Ctx& x, const store_arg_t* arg) {
  auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + (x.b * x.nt + x.tid) * ERR_WORDS;
  err[ERR_TEST]      = x.first_test;
  err[ERR_INDEX]     = x.first_index;
  err[ERR_GOT]       = (uint32_t)x.first_got;
  err[ERR_GOT + 1]   = (uint32_t)(x.first_got >> 32);
  err[ERR_WANT]      = (uint32_t)x.first_want;
  err[ERR_WANT + 1]  = (uint32_t)(x.first_want >> 32);
  err[ERR_COUNT]     = x.errors;
}

// One doubleword as one store, two words, four halves or eight bytes.
INL void store_split(uint32_t kind, volatile uint8_t* p, uint64_t v) {
  switch (kind) {
  case 0:
    *reinterpret_cast<volatile uint64_t*>(p) = v;
    break;
  case 1:
    for (uint32_t h = 0; h < 2; ++h)
      *reinterpret_cast<volatile uint32_t*>(p + 4 * h) = (uint32_t)(v >> (32 * h));
    break;
  case 2:
    for (uint32_t h = 0; h < 4; ++h)
      *reinterpret_cast<volatile uint16_t*>(p + 2 * h) = (uint16_t)(v >> (16 * h));
    break;
  default:
    for (uint32_t h = 0; h < 8; ++h)
      p[h] = (uint8_t)(v >> (8 * h));
    break;
  }
}

}  // namespace

// Fill the own slice, sync, store into every slice (the own one last, through
// the window), sync, then check the own slice: each sampled doubleword from its
// writer, every other one still the fill.
__kernel void a2a_k(store_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t n8 = x.n / 2;
  auto own8 = reinterpret_cast<volatile uint64_t*>(x.own);
  for (uint32_t i = x.tid; i < n8; i += x.nt)
    own8[i] = pattern64(x.seed, TEST_FILL, x.c, x.rank, i);
  x.cl.sync();
  const uint32_t M = check_count(n8);
  for (uint32_t s = 1; s <= x.k; ++s) {
    const uint32_t q = (x.rank + s) % x.k;
    auto w = x.window<uint8_t>(q);
    for (uint32_t j = 0;; ++j) {
      const uint32_t m = x.rank + x.k * (x.tid + x.nt * j);
      if (m >= M)
        break;
      const uint32_t i = check_index(m, n8);
      store_split(a2a_kind(j, q), w + 8 * i, pattern64(x.seed, TEST_A2A, x.c, x.rank * 8 + q, i));
    }
  }
  x.cl.sync();
  for (uint32_t i = x.tid; i < n8; i += x.nt) {
    const uint32_t m = check_slot(i, n8);
    const uint32_t writer = (m == ~0u) ? x.rank : m % x.k;
    const uint64_t want = (m == ~0u) ? pattern64(x.seed, TEST_FILL, x.c, x.rank, i)
                                     : pattern64(x.seed, TEST_A2A, x.c, writer * 8 + x.rank, i);
    x.check(TEST_A2A, (writer << 24) | i, own8[i], want);
  }
  ctx_done(x, arg);
}

// A ring of producers and consumers: each round, a block stores data into the
// next rank's slice, fences, and raises the flag there; it then waits for its
// own flag, checks the data from the previous rank, and acks it back. Only the
// fence orders the data before the flag when the block is one warp.
__kernel void pc_k(store_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t next = (x.rank + 1) % x.k;
  const uint32_t prev = (x.rank + x.k - 1) % x.k;
  const uint32_t words = (x.n - PC_DATA < PC_MAX_WORDS) ? (x.n - PC_DATA) : PC_MAX_WORDS;
  auto wn = x.window<uint32_t>(next);
  auto wp = x.window<uint32_t>(prev);
  if (x.tid == 0) {
    x.own[PC_FLAG] = 0;
    x.own[PC_ACK]  = 0;
  }
  x.cl.sync();
  for (uint32_t r = 1; r <= arg->rounds; ++r) {
    if (x.tid == 0) {
      // The next rank must have consumed the previous round.
      uint32_t spins = 0, ack;
      while ((ack = x.own[PC_ACK]) < r - 1 && ++spins < SPIN_LIMIT) {}
      x.check(TEST_PC, (r << 24) | 0xFFFFA, ack, r - 1);
    }
    x.block_sync();
    for (uint32_t d = x.tid; d < words; d += x.nt)
      wn[PC_DATA + d] = pattern(x.seed, TEST_PC, x.c, x.rank, (r << 16) | d);
    vx_fence();
    x.block_sync();
    if (x.tid == 0) {
      wn[PC_FLAG] = r;
      uint32_t spins = 0, flag;
      while ((flag = x.own[PC_FLAG]) < r && ++spins < SPIN_LIMIT) {}
      x.check(TEST_PC, (r << 24) | 0xFFFFF, flag, r);
    }
    x.block_sync();
    for (uint32_t d = x.tid; d < words; d += x.nt)
      x.check(TEST_PC, (r << 24) | d, x.own[PC_DATA + d], pattern(x.seed, TEST_PC, x.c, prev, (r << 16) | d));
    // The reads must complete before the producer may overwrite them.
    vx_fence();
    x.block_sync();
    if (x.tid == 0)
      wp[PC_ACK] = r;
  }
  // The last acks land in slices whose blocks must still be resident.
  x.cl.sync();
  ctx_done(x, arg);
}

// Fixed loops of window stores into the next rank's slice, one fence per warp,
// and the fence-free cluster barrier: no spin and no data-dependent trip count,
// so both models retire the same count. Each block then copies its slice out.
__kernel void parity_k(store_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  for (uint32_t e = x.tid; e < x.n; e += x.nt)
    x.own[e] = 0;
  x.cl.barrier_wait_relaxed(x.cl.barrier_arrive_relaxed());
  const uint32_t rounds = arg->rounds, m = x.n / x.nt;
  auto mine = x.window<uint32_t>((x.rank + 1) % x.k) + x.tid * m;
  const uint32_t v0 = parity_value(x.seed, x.c, x.rank, 0, x.tid);
  // Eight independent stores per step keep the loop bound by the network, not
  // by the core's dependent-ALU latency.
  for (uint32_t j = 0; j < rounds; j += PARITY_STEP) {
    volatile uint32_t* q = mine + (j & (m - 1));
    const uint32_t v = v0 + j;
    q[0] = v;     q[1] = v + 1; q[2] = v + 2; q[3] = v + 3;
    q[4] = v + 4; q[5] = v + 5; q[6] = v + 6; q[7] = v + 7;
  }
  vx_fence();
  x.cl.barrier_wait_relaxed(x.cl.barrier_arrive_relaxed());
  auto glob = reinterpret_cast<uint32_t*>(arg->glob_addr) + x.b * x.n;
  for (uint32_t e = x.tid; e < x.n; e += x.nt)
    glob[e] = x.own[e];
  ctx_done(x, arg);
}
