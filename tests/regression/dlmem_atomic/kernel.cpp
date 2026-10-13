#include <vx_spawn2.h>
#include <vx_dlmem.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

struct Ctx {
  uint32_t k, rank, c, b, tid, nt, n, seed, lane;
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

INL void ctx_init(Ctx& x, const atomic_arg_t* arg) {
  x.k     = arg->k;
  x.rank  = vx_cluster_block_rank();
  x.b     = blockIdx.x;
  x.c     = x.b / x.k;
  x.tid   = threadIdx.x;
  x.nt    = blockDim.x;
  x.n     = arg->lmem_bytes / sizeof(uint32_t);
  x.seed  = arg->seed;
  x.lane  = x.tid % arg->num_threads;
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

INL void ctx_done(const Ctx& x, const atomic_arg_t* arg) {
  auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + (x.b * x.nt + x.tid) * ERR_WORDS;
  err[ERR_TEST]      = x.first_test;
  err[ERR_INDEX]     = x.first_index;
  err[ERR_GOT]       = (uint32_t)x.first_got;
  err[ERR_GOT + 1]   = (uint32_t)(x.first_got >> 32);
  err[ERR_WANT]      = (uint32_t)x.first_want;
  err[ERR_WANT + 1]  = (uint32_t)(x.first_want >> 32);
  err[ERR_COUNT]     = x.errors;
}

// Per-block words of the global buffer, after the per-thread ones.
INL uint32_t* block_words(const Ctx& x, const atomic_arg_t* arg, uint32_t per_thread, uint32_t per_block) {
  const uint32_t threads = gridDim.x * x.nt;
  return reinterpret_cast<uint32_t*>(arg->glob_addr) + threads * per_thread + x.b * per_block;
}

// The returned word, sign-extended as RV64 returns it.
INL int64_t amoadd_w(volatile void* p, uint32_t v) {
  int64_t old;
  asm volatile ("amoadd.w %0, %2, (%1)" : "=r"(old) : "r"(p), "r"(v) : "memory");
  return old;
}
INL uint64_t amoadd_d(volatile void* p, uint64_t v) {
  uint64_t old;
  asm volatile ("amoadd.d %0, %2, (%1)" : "=r"(old) : "r"(p), "r"(v) : "memory");
  return old;
}
INL uint64_t amomaxu_d(volatile void* p, uint64_t v) {
  uint64_t old;
  asm volatile ("amomaxu.d %0, %2, (%1)" : "=r"(old) : "r"(p), "r"(v) : "memory");
  return old;
}
INL uint32_t amoswap_w_aq(volatile void* p, uint32_t v) {
  uint32_t old;
  asm volatile ("amoswap.w.aq %0, %2, (%1)" : "=r"(old) : "r"(p), "r"(v) : "memory");
  return old;
}
INL void amoswap_w_rl(volatile void* p, uint32_t v) {
  asm volatile ("amoswap.w.rl zero, %1, (%0)" :: "r"(p), "r"(v) : "memory");
}

// Operation o with no destination register: the result is the memory's.
#define AMO_NORET(insn, p, v) asm volatile (insn " zero, %1, (%0)" :: "r"(p), "r"(v) : "memory")
INL void amo_op(uint32_t o, uint32_t d, volatile void* p, uint64_t v) {
  if (d) {
    switch (o) {
    case OP_ADD:  AMO_NORET("amoadd.d",  p, v); break;
    case OP_XOR:  AMO_NORET("amoxor.d",  p, v); break;
    case OP_AND:  AMO_NORET("amoand.d",  p, v); break;
    case OP_OR:   AMO_NORET("amoor.d",   p, v); break;
    case OP_MIN:  AMO_NORET("amomin.d",  p, v); break;
    case OP_MAX:  AMO_NORET("amomax.d",  p, v); break;
    case OP_MINU: AMO_NORET("amominu.d", p, v); break;
    default:      AMO_NORET("amomaxu.d", p, v); break;
    }
  } else {
    switch (o) {
    case OP_ADD:  AMO_NORET("amoadd.w",  p, v); break;
    case OP_XOR:  AMO_NORET("amoxor.w",  p, v); break;
    case OP_AND:  AMO_NORET("amoand.w",  p, v); break;
    case OP_OR:   AMO_NORET("amoor.w",   p, v); break;
    case OP_MIN:  AMO_NORET("amomin.w",  p, v); break;
    case OP_MAX:  AMO_NORET("amomax.w",  p, v); break;
    case OP_MINU: AMO_NORET("amominu.w", p, v); break;
    default:      AMO_NORET("amomaxu.w", p, v); break;
    }
  }
}

}  // namespace

// Every thread of the cluster adds into rank 0's word and doubleword bins,
// then draws one ticket of each width from rank k - 1. The host checks the
// bins against its own sums and that the tickets are each value exactly once.
__kernel void hist_k(atomic_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  auto own8 = reinterpret_cast<volatile uint64_t*>(x.own);
  const uint32_t total = x.k * x.nt;
  if (x.rank == 0) {
    for (uint32_t i = x.tid; i < HIST_BINS; i += x.nt) {
      x.own[HIST_W + i] = 0;
      own8[HIST_D + i] = 0;
    }
  }
  if (x.rank + 1 == x.k && x.tid == 0) {
    x.own[TICKET_W] = ticket_w0(total);
    own8[TICKET_D] = ticket_d0(total);
  }
  x.cl.sync();
  auto w0 = x.window<uint32_t>(0);
  auto d0 = reinterpret_cast<volatile uint64_t*>(w0);
  for (uint32_t j = 0; j < arg->iters; ++j) {
    const uint32_t h = hist_hash(x.seed, x.c, x.rank, x.tid, j);
    amoadd_w(w0 + HIST_W + hist_bin(h), hist_w(h));
    amoadd_d(d0 + HIST_D + hist_bin(h), hist_d(h));
  }
  auto wl = x.window<uint32_t>(x.k - 1);
  const int64_t tw = amoadd_w(wl + TICKET_W, 1);
  const uint64_t td = amoadd_d(reinterpret_cast<volatile uint64_t*>(wl) + TICKET_D, 1);
  auto mine = reinterpret_cast<uint32_t*>(arg->glob_addr) + (x.b * x.nt + x.tid) * HIST_TWORDS;
  mine[0] = (uint32_t)tw;
  mine[1] = (uint32_t)((uint64_t)tw >> 32);
  mine[2] = (uint32_t)td;
  mine[3] = (uint32_t)(td >> 32);
  x.cl.sync();
  auto out = block_words(x, arg, HIST_TWORDS, HIST_BWORDS);
  if (x.rank == 0) {
    for (uint32_t i = x.tid; i < HIST_BINS; i += x.nt) {
      out[i] = x.own[HIST_W + i];
      out[HIST_BINS + 2 * i]     = x.own[2 * (HIST_D + i)];
      out[HIST_BINS + 2 * i + 1] = x.own[2 * (HIST_D + i) + 1];
    }
  }
  if (x.rank + 1 == x.k && x.tid == 0) {
    out[3 * HIST_BINS]     = x.own[TICKET_W];
    out[3 * HIST_BINS + 2] = x.own[2 * TICKET_D];
    out[3 * HIST_BINS + 3] = x.own[2 * TICKET_D + 1];
  }
  ctx_done(x, arg);
}

// One lane per warp takes an amoswap spinlock in rank k - 1, names itself the
// holder in rank 1 % k, increments rank 0's counter with a plain window load
// and store, checks it is still the holder, fences and releases.
__kernel void lock_k(atomic_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  auto own8 = reinterpret_cast<volatile uint64_t*>(x.own);
  if (x.tid == 0) {
    if (x.rank + 1 == x.k)
      x.own[LOCK_WORD] = 0;
    if (x.rank == 1 % x.k)
      x.own[LOCK_HOLDER] = ~0u;
    if (x.rank == 0)
      own8[LOCK_COUNT] = 0;
  }
  x.cl.sync();
  if (x.lane == 0) {
    auto lock   = x.window<uint32_t>(x.k - 1) + LOCK_WORD;
    auto holder = x.window<uint32_t>(1 % x.k) + LOCK_HOLDER;
    auto count  = x.window<uint64_t>(0) + LOCK_COUNT;
    const uint32_t me = (x.rank << 16) | x.tid;
    for (uint32_t j = 0; j < arg->iters; ++j) {
      uint32_t spins = 0, got;
      while ((got = amoswap_w_aq(lock, 1)) != 0 && ++spins < SPIN_LIMIT) {}
      x.check(TEST_LOCK, (j << 24) | 0xFFFFFF, got, 0);
      *holder = me;
      *count = *count + 1;
      x.check(TEST_LOCK, j << 24, *holder, me);
      vx_fence();
      amoswap_w_rl(lock, 0);
    }
  }
  x.cl.sync();
  if (x.rank == 0 && x.tid == 0) {
    auto out = block_words(x, arg, 0, LOCK_BWORDS);
    out[0] = x.own[2 * LOCK_COUNT];
    out[1] = x.own[2 * LOCK_COUNT + 1];
  }
  ctx_done(x, arg);
}

// Every thread applies each of the eight operations, at both widths, to every
// slice of its cluster (its own last, through the window), with no destination
// register; the words alternate between the low and high half of a doubleword.
__kernel void ops_k(atomic_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  auto own8 = reinterpret_cast<volatile uint64_t*>(x.own);
  if (x.tid == 0) {
    for (uint32_t o = 0; o < 8; ++o) {
      x.own[OPS_W + o] = (uint32_t)ops_init(x.seed, x.c, x.rank, o);
      own8[OPS_D + o] = ops_init(x.seed, x.c, x.rank, 8 + o);
    }
  }
  x.cl.sync();
  for (uint32_t s = 1; s <= x.k; ++s) {
    const uint32_t q = (x.rank + s) % x.k;
    auto w = x.window<uint32_t>(q);
    auto d = reinterpret_cast<volatile uint64_t*>(w);
    for (uint32_t o = 0; o < 8; ++o) {
      amo_op(o, 0, w + OPS_W + o, ops_operand(x.seed, x.c, x.rank, x.tid, q, o, 0));
      amo_op(o, 1, d + OPS_D + o, ops_operand(x.seed, x.c, x.rank, x.tid, q, o, 1));
    }
  }
  x.cl.sync();
  auto out = block_words(x, arg, 0, OPS_BWORDS);
  for (uint32_t i = x.tid; i < OPS_BWORDS; i += x.nt)
    out[i] = x.own[OPS_W + i];
  ctx_done(x, arg);
}

#if VX_CFG_EXT_ZACAS_ENABLED
// Every thread increments rank 0's word and rank k - 1's doubleword iters times
// with a compare-and-swap retry loop; a failed swap returns the word it found.
__kernel void cas_k(atomic_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  auto own8 = reinterpret_cast<volatile uint64_t*>(x.own);
  if (x.tid == 0) {
    if (x.rank == 0)
      x.own[CAS_WORD] = 0x7FFFFFF0u;
    if (x.rank + 1 == x.k)
      own8[CAS_DWORD] = 0xFFFFFFF0ull;
  }
  x.cl.sync();
  auto cw = const_cast<uint32_t*>(x.window<uint32_t>(0) + CAS_WORD);
  auto cd = const_cast<uint64_t*>(x.window<uint64_t>(x.k - 1) + CAS_DWORD);
  for (uint32_t j = 0; j < arg->iters; ++j) {
    uint32_t ew = 0x7FFFFFF0u, tries = 0;
    while (!__atomic_compare_exchange_n(cw, &ew, ew + 1, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)
           && ++tries < SPIN_LIMIT) {}
    x.check(TEST_CAS, j << 24, tries < SPIN_LIMIT, 1);
    uint64_t ed = 0xFFFFFFF0ull;
    tries = 0;
    while (!__atomic_compare_exchange_n(cd, &ed, ed + 1, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)
           && ++tries < SPIN_LIMIT) {}
    x.check(TEST_CAS, (j << 24) | 1, tries < SPIN_LIMIT, 1);
  }
  x.cl.sync();
  if (x.tid == 0) {
    auto out = block_words(x, arg, 0, CAS_BWORDS);
    if (x.rank == 0)
      out[0] = x.own[CAS_WORD];
    if (x.rank + 1 == x.k) {
      out[2] = x.own[2 * CAS_DWORD];
      out[3] = x.own[2 * CAS_DWORD + 1];
    }
  }
  ctx_done(x, arg);
}
#endif

// Fixed loops of window atomics, each thread on its own word and doubleword of
// the next rank's slice, so every returned value is fixed too; between two
// fence-free cluster barriers, with no spin, so both models retire the same count.
// FENCE adds a fence per iteration, held until the atomics are answered.
template <bool FENCE>
static inline __attribute__((always_inline)) void parity_body(atomic_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  auto own8 = reinterpret_cast<volatile uint64_t*>(x.own);
  x.own[x.tid] = 0;
  own8[x.nt + x.tid] = 0;
  x.cl.barrier_wait(x.cl.barrier_arrive());
  auto w = x.window<uint32_t>((x.rank + 1) % x.k);
  auto d = reinterpret_cast<volatile uint64_t*>(w);
  uint64_t sum = 0;
  for (uint32_t j = 0; j < arg->iters; ++j) {
    sum += (uint64_t)amoadd_w(w + x.tid, j + 1);
    sum ^= amomaxu_d(d + x.nt + x.tid, parity_d(x.seed, x.c, x.rank, x.tid, j));
    if (FENCE)
      vx_fence();
  }
  x.cl.barrier_wait(x.cl.barrier_arrive());
  auto mine = reinterpret_cast<uint32_t*>(arg->glob_addr) + (x.b * x.nt + x.tid) * PARITY_TWORDS;
  mine[0] = (uint32_t)sum;
  mine[1] = (uint32_t)(sum >> 32);
  auto out = block_words(x, arg, PARITY_TWORDS, PARITY_BWORDS(x.nt));
  out[x.tid] = x.own[x.tid];
  out[x.nt + 2 * x.tid]     = x.own[2 * (x.nt + x.tid)];
  out[x.nt + 2 * x.tid + 1] = x.own[2 * (x.nt + x.tid) + 1];
  ctx_done(x, arg);
}

__kernel void parity_k(atomic_arg_t* __UNIFORM__ arg) {
  parity_body<false>(arg);
}

__kernel void parity_fence_k(atomic_arg_t* __UNIFORM__ arg) {
  parity_body<true>(arg);
}

// Rank 0 takes a reservation on a word of rank 1 through the window: the
// simulation must stop.
__kernel void lrsc_k(atomic_arg_t* __UNIFORM__ arg) {
  Ctx x;
  ctx_init(x, arg);
  if (x.rank == 0 && x.tid == 0) {
    auto p = x.window<uint32_t>(1);
    uint32_t v, fail;
    asm volatile ("lr.w %0, (%1)" : "=r"(v) : "r"(p) : "memory");
    asm volatile ("sc.w %0, %2, (%1)" : "=r"(fail) : "r"(p), "r"(v + 1) : "memory");
    x.check(TEST_SETUP, 1, fail, 1);
  }
  x.cl.sync();
  ctx_done(x, arg);
}
