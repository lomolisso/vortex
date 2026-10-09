#include <vx_spawn2.h>
#include <vx_dlmem.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

struct Ctx {
  uint32_t k, rank, c, b, tid, nt, n, seed, flags, base_slot, warps;
  volatile uint32_t* own;
  vortex::cluster_group cl;
  uint32_t errors, first_test, first_index, first_got;

  INL volatile uint32_t* window(uint32_t q) {
    return static_cast<volatile uint32_t*>(cl.map_shared_rank(const_cast<uint32_t*>(own), q));
  }

  INL void fail(uint32_t test, uint32_t index, uint32_t got) {
    if (errors++ == 0) {
      first_test  = test;
      first_index = index;
      first_got   = got;
    }
  }

  INL void check(uint32_t test, uint32_t index, uint32_t got, uint32_t want) {
    if (got != want)
      fail(test, index, got);
  }

  // Same-core members hold consecutive slots base + rank, so barrier 1 of the
  // base slot, counting every warp of the cluster, is the cluster's own.
  INL void sync() {
    if (flags & FLAG_CBAR) {
      cl.sync();
    } else if (k == 1) {
      __syncthreads();
    } else {
      vx_barrier((1 << 8) | base_slot, k * warps);
    }
  }
};

INL void ctx_init(Ctx& x, const local_arg_t* arg) {
  x.k     = arg->k;
  x.rank  = vx_cluster_block_rank();
  x.b     = blockIdx.x;
  x.c     = x.b / x.k;
  x.tid   = threadIdx.x;
  x.nt    = blockDim.x;
  x.n     = arg->lmem_bytes / sizeof(uint32_t);
  x.seed  = arg->seed;
  x.flags = arg->flags;
  x.base_slot = get_local_group_id() - x.rank;
  x.warps = get_num_sub_groups();
  x.own   = reinterpret_cast<volatile uint32_t*>(__local_mem());
  x.errors = x.first_test = x.first_index = x.first_got = 0;
  if (x.tid == 0) {
    auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + x.b * REC_WORDS;
    rec[REC_RANK] = x.rank;
    rec[REC_SIZE] = vx_cluster_num_blocks();
    rec[REC_CORE] = vx_core_id();
    rec[REC_SLOT] = get_local_group_id();
    rec[REC_RAN]  = 1;
  }
  if (vx_cluster_num_blocks() != x.k)
    x.fail(TEST_SETUP, 0, vx_cluster_num_blocks());
}

INL void ctx_done(const Ctx& x, const local_arg_t* arg) {
  auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + (x.b * x.nt + x.tid) * ERR_WORDS;
  err[ERR_TEST]  = x.first_test;
  err[ERR_INDEX] = x.first_index;
  err[ERR_GOT]   = x.first_got;
  err[ERR_COUNT] = x.errors;
}

// Write own slice, then read every peer's.
INL void test_own(Ctx& x) {
  const uint32_t M = check_count(x.n);
  for (uint32_t m = x.tid; m < M; m += x.nt) {
    const uint32_t i = check_index(m, x.n);
    x.own[i] = pattern(x.seed, TEST_OWN, x.c, x.rank, i);
  }
  x.sync();
  for (uint32_t q = 0; q < x.k; ++q) {
    auto w = x.window(q);
    for (uint32_t m = x.tid; m < M; m += x.nt) {
      const uint32_t i = check_index(m, x.n);
      x.check(TEST_OWN, (q << 24) | i, w[i], pattern(x.seed, TEST_OWN, x.c, q, i));
    }
  }
  x.sync();
}

// Sampled word m of every peer's slice is written by rank m % k.
INL void test_store(Ctx& x) {
  const uint32_t M = check_count(x.n);
  for (uint32_t q = 0; q < x.k; ++q) {
    auto w = x.window(q);
    for (uint32_t m = x.rank + x.k * x.tid; m < M; m += x.k * x.nt) {
      const uint32_t i = check_index(m, x.n);
      w[i] = pattern(x.seed, TEST_STORE, x.c, x.rank * 8 + q, i);
    }
  }
  x.sync();
  for (uint32_t m = x.tid; m < M; m += x.nt) {
    const uint32_t i = check_index(m, x.n);
    const uint32_t writer = m % x.k;
    x.check(TEST_STORE, (writer << 24) | i, x.own[i],
            pattern(x.seed, TEST_STORE, x.c, writer * 8 + x.rank, i));
  }
  x.sync();
}

// Every thread of the cluster adds 1 to word 0 of rank 0.
INL void test_amo(Ctx& x) {
  if (x.rank == 0 && x.tid == 0)
    x.own[0] = 0;
  x.sync();
  auto ctr = const_cast<uint32_t*>(x.window(0));
  __atomic_fetch_add(ctr, 1u, __ATOMIC_RELAXED);
  x.sync();
  const uint32_t want = x.k * x.nt;
  x.check(TEST_AMO, 0, x.window(0)[0], want);
  if (x.rank == 0)
    x.check(TEST_AMO, 1, x.own[0], want);
  x.sync();
}

// One store and one load instruction whose lanes split three ways.
INL void test_mixed(Ctx& x, uint32_t* glob) {
  const uint32_t kind = mixed_kind(x.rank, x.tid, x.nt, x.n);
  const uint32_t cell = x.rank * x.nt + x.tid;
  const uint32_t q = (x.rank + x.tid) % x.k;
  const uintptr_t pg = (uintptr_t)(glob + x.b * x.nt + x.tid);
  const uintptr_t pl = (uintptr_t)(x.own + cell);
  const uintptr_t pw = (uintptr_t)(x.window(q) + cell);
  // Branch-free, so the lanes stay in one instruction.
  const uintptr_t m0 = (uintptr_t)0 - (uintptr_t)(kind == 0);
  const uintptr_t m1 = (uintptr_t)0 - (uintptr_t)(kind == 1);
  const uintptr_t m2 = (uintptr_t)0 - (uintptr_t)(kind == 2);
  const uintptr_t p = (pg & m0) | (pl & m1) | (pw & m2);
  const uint32_t v = pattern(x.seed, TEST_MIXED, x.c, x.rank, x.tid);
  asm volatile ("sw %0, 0(%1)" :: "r"(v), "r"(p) : "memory");
  x.sync();
  uint32_t got;
  asm volatile ("lw %0, 0(%1)" : "=r"(got) : "r"(p) : "memory");
  x.check(TEST_MIXED, (kind << 24) | x.tid, got, v);
  // The cells of this slice: own lanes of this rank and window lanes aimed at it.
  for (uint32_t e = x.tid; e < x.k * x.nt && e < x.n; e += x.nt) {
    const uint32_t r = e / x.nt, t = e % x.nt;
    const uint32_t kd = mixed_kind(r, t, x.nt, x.n);
    if ((kd == 1 && r == x.rank) || (kd == 2 && (r + t) % x.k == x.rank))
      x.check(TEST_MIXED, (r << 24) | (1u << 16) | t, x.own[e], pattern(x.seed, TEST_MIXED, x.c, r, t));
  }
  x.sync();
}

// Sub-word stores around each boundary, from one lane, in an order that exposes a
// wrong byte enable: each store lands next to bytes an earlier one wrote. Byte +1
// is never written, so it must stay 0.
struct EdgePoint { int8_t off; uint8_t size; };
const EdgePoint kEdgeLo[] = {{-8, 4}, {-4, 2}, {-2, 1}, {-1, 1}};
const EdgePoint kEdgeHi[] = {{0, 1}, {2, 2}, {4, 4}};

// The slice's two ends and, past one line, the first line boundary inside it.
INL uint32_t edge_bounds(uint32_t L, uint32_t* bd) {
  uint32_t nb = 0;
  bd[nb++] = 0;
  bd[nb++] = L;
  if (L > 64)
    bd[nb++] = 64;
  return nb;
}

// Byte-distinct and never zero: one hash per writer and round, then a step per offset.
INL uint32_t edge_value(uint32_t base, uint32_t off, uint32_t size) {
  const uint32_t v = (base + off * 0x9E3779B1u) | 0x01010101u;
  return (size == 4) ? v : (v & ((1u << (8 * size)) - 1));
}

// The points around boundary bd that lie inside a slice of L bytes.
template <typename F>
INL void edge_points(uint32_t L, uint32_t bd, F f) {
  if (bd >= 8) {
    for (const auto& p : kEdgeLo)
      f(bd + p.off, p.size);
  }
  if (bd + 8 <= L) {
    for (const auto& p : kEdgeHi)
      f(bd + p.off, p.size);
  }
}

INL void test_edge(Ctx& x) {
  const uint32_t L = x.n * sizeof(uint32_t);
  uint32_t bd[3];
  const uint32_t nb = edge_bounds(L, bd);
  for (uint32_t s = 0; s < x.k; ++s) {
    const uint32_t q = (x.rank + s) % x.k;
    const uint32_t writer = (x.rank + x.k - s) % x.k;
    // The words around each boundary: four per boundary, the ones inside the slice.
    for (uint32_t e = x.tid; e < nb * 4; e += x.nt) {
      const int32_t a = int32_t(bd[e / 4]) - 8 + 4 * int32_t(e % 4);
      if (a >= 0 && uint32_t(a) < L)
        x.own[a / 4] = 0;
    }
    x.sync();
    if (x.tid == 0) {
      auto w8 = reinterpret_cast<volatile uint8_t*>(x.window(q));
      const uint32_t base = pattern(x.seed, TEST_EDGE, x.c, x.rank * 8 + s, 0);
      for (uint32_t j = 0; j < nb; ++j) {
        edge_points(L, bd[j], [&](uint32_t off, uint32_t size) {
          const uint32_t v = edge_value(base, off, size);
          uint32_t got;
          if (size == 1) {
            *(w8 + off) = (uint8_t)v;
            got = *(w8 + off);
          } else if (size == 2) {
            *reinterpret_cast<volatile uint16_t*>(w8 + off) = (uint16_t)v;
            got = *reinterpret_cast<volatile uint16_t*>(w8 + off);
          } else {
            *reinterpret_cast<volatile uint32_t*>(w8 + off) = v;
            got = *reinterpret_cast<volatile uint32_t*>(w8 + off);
          }
          x.check(TEST_EDGE, (q << 24) | (size << 20) | off, got, v);
        });
      }
    }
    x.sync();
    // The writer's bytes and zeros between them.
    const uint32_t base = pattern(x.seed, TEST_EDGE, x.c, writer * 8 + s, 0);
    for (uint32_t e = x.tid; e < nb * 4; e += x.nt) {
      const uint32_t j = e / 4;
      const int32_t a = int32_t(bd[j]) - 8 + 4 * int32_t(e % 4);
      if (a < 0 || uint32_t(a) >= L)
        continue;
      uint32_t want = 0;
      edge_points(L, bd[j], [&](uint32_t off, uint32_t size) {
        if ((off & ~3u) == uint32_t(a))
          want |= edge_value(base, off, size) << (8 * (off & 3u));
      });
      x.check(TEST_EDGE, (writer << 24) | (1u << 20) | uint32_t(a), x.own[a / 4], want);
    }
  }
  x.sync();
}

// The self window and the plain pointer name the same words, both ways.
INL void test_self(Ctx& x) {
  auto w = x.window(x.rank);
  if (x.tid == 0) {
    const uintptr_t want = (uintptr_t)VX_MEM_DLMEM_BASE_ADDR
                         + ((uintptr_t)x.rank << VX_CFG_LMEM_LOG_SIZE);
    if ((uintptr_t)w != want)
      x.fail(TEST_SELF, 0xFFFFFFu, (uint32_t)(uintptr_t)w);
  }
  const uint32_t M = check_count(x.n);
  for (uint32_t m = x.tid; m < M; m += x.nt) {
    const uint32_t i = check_index(m, x.n);
    const uint32_t a = pattern(x.seed, TEST_SELF, x.c, x.rank, i);
    x.own[i] = a;
    x.check(TEST_SELF, i, w[i], a);
    w[i] = ~a;
    x.check(TEST_SELF, (1u << 24) | i, x.own[i], ~a);
  }
}

}  // namespace

// One kernel per test: with all six inlined into one function the compiler
// emits no split/join for its divergent branches.
#define LOCAL_KERNEL(name, call)                                           \
  __kernel void name(local_arg_t* __UNIFORM__ arg) {                       \
    Ctx x;                                                                 \
    ctx_init(x, arg);                                                      \
    call;                                                                  \
    ctx_done(x, arg);                                                      \
  }

LOCAL_KERNEL(local1_k, test_own(x))
LOCAL_KERNEL(local2_k, test_store(x))
LOCAL_KERNEL(local3_k, test_amo(x))
LOCAL_KERNEL(local4_k, test_mixed(x, reinterpret_cast<uint32_t*>(arg->glob_addr)))
LOCAL_KERNEL(local5_k, test_edge(x))
LOCAL_KERNEL(local6_k, test_self(x))

// Each block reads what the slot's previous block left, then ends with window
// stores to its own rank right before it exits, while later clusters keep
// taking the slots it frees.
__kernel void exit_k(local_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  auto glob = reinterpret_cast<uint32_t*>(arg->glob_addr);
  const bool mine = x.tid < x.n;
  const uint32_t mark = pattern(x.seed, 0, x.c, x.rank, x.tid) & 0x0FFFFFFFu;
  if (mine) {
    glob[x.b * x.nt + x.tid] = x.own[x.tid];
    x.own[x.tid] = mark;
  }
  auto w = x.window(x.rank);
  for (uint32_t j = 0; j < 4 && mine; ++j)
    x.check(TEST_EXIT, x.tid, w[x.tid], mark);
  ctx_done(x, arg);
  if (mine)
    w[x.tid] = exit_word(x.b, x.tid);
}

// Fixed loops of window loads, stores and atomics on the next rank's slice (the
// own one for K = 1), each thread on cells no other thread touches: no spin and
// no data-dependent trip count, so both models retire the same count.
__kernel void parity_k(local_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const bool atomics = (x.flags & FLAG_ATOMICS) != 0;
  const uint32_t cells = x.k * x.nt;
  for (uint32_t e = x.tid; e < cells; e += x.nt) {
    x.own[e] = 0;
    x.own[cells + e] = 0;
  }
  x.sync();
  const uint32_t mine = x.rank * x.nt + x.tid;
  auto w = x.window((x.rank + 1) % x.k);
  for (uint32_t j = 0; j < PARITY_ITERS; ++j) {
    w[mine] = w[mine] + 1;
    if (atomics)
      __atomic_fetch_add(const_cast<uint32_t*>(w + cells + mine), 1u, __ATOMIC_RELAXED);
  }
  x.sync();
  const uint32_t from = ((x.rank + x.k - 1) % x.k) * x.nt + x.tid;
  auto glob = reinterpret_cast<uint32_t*>(arg->glob_addr) + 2 * (x.b * x.nt + x.tid);
  glob[0] = x.own[from];
  glob[1] = x.own[cells + from];
  ctx_done(x, arg);
}

// One-thread blocks: every load is a one-lane read, a single packet however
// narrow the SIMD width, and eight of them leave before the first returns. The
// LSU's pool of outstanding reads must hold back a warp that issues more.
__kernel void burst_k(local_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  const uint32_t first = (x.b * x.nt + x.tid) * BURST_WORDS;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(arg->glob_addr) + first;
  uint32_t got = 0, want = 0;
  for (uint32_t j = 0; j < BURST_ROUNDS; ++j) {
    const uint32_t* p = src + j * BURST_LOADS * BURST_STRIDE;
    uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
    asm volatile ("lw %0, 0(%8)\n\tlw %1, 64(%8)\n\tlw %2, 128(%8)\n\tlw %3, 192(%8)\n\t"
                  "lw %4, 256(%8)\n\tlw %5, 320(%8)\n\tlw %6, 384(%8)\n\tlw %7, 448(%8)"
                  : "=&r"(v0), "=&r"(v1), "=&r"(v2), "=&r"(v3),
                    "=&r"(v4), "=&r"(v5), "=&r"(v6), "=&r"(v7)
                  : "r"(p) : "memory");
    got += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    for (uint32_t i = 0; i < BURST_LOADS; ++i)
      want += pattern(x.seed, TEST_BURST, 0, 0, first + (j * BURST_LOADS + i) * BURST_STRIDE);
  }
  x.check(TEST_BURST, x.b, got, want);
  ctx_done(x, arg);
}

// One window store that must stop the simulation.
__kernel void neg_k(local_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  if (x.rank == 0 && x.tid == 0) {
    const uint32_t target = (arg->tests == NEG_OOB) ? 2 : 1;
    x.window(target)[0] = 0x0BADu;
  }
  ctx_done(x, arg);
}
