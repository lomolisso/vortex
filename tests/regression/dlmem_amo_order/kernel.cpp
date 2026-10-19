#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dlmem.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"

// One cluster of four SPREAD blocks, rank r on core r. In mp mode rank 0 lane
// 0 stores DATA = r into rank 1 and then swaps FLAG = r into rank 2 with
// amoswap.d.rl, for r = 1..n; rank 2 lane 0 polls its own FLAG and, on each new
// value f, loads DATA from rank 1, which must be >= f. In mpaq mode DATA lives
// in rank 2 and FLAG in rank 1: rank 2 reads the remote FLAG with amoor.d.aq
// and its own DATA right behind it, so only the .aq hold keeps the fast local
// load from overtaking the AMO. Rank 3 floods rank 1 with window stores and
// rank 1 hammers its own banks with atomics, or streams DXA copies into them.
// The counters only grow, so the producer needs no handshake. In lat mode rank
// 0 lane 0 times fixed probe sequences against rank 1 (LAT_* in common.h). In
// wide mode every lane of two warps per block runs mixed .aq/.rl atomics.
//
// The kernel is flat: per-mode helper functions lost their divergence
// handling in the toolchain, so only straight-line asm is factored out.

#define INL static inline __attribute__((always_inline))

INL void amoswap_d_rl(volatile uint64_t* p, uint64_t v) {
  asm volatile ("amoswap.d.rl zero, %1, (%0)" :: "r"(p), "r"(v) : "memory");
}
INL void amoswap_d(volatile uint64_t* p, uint64_t v) {
  asm volatile ("amoswap.d zero, %1, (%0)" :: "r"(p), "r"(v) : "memory");
}

// FLAG with an acquire AMO, then DATA with a load issued before any branch
// on the flag: a branch would stall the warp and order them anyway.
INL void poll_aq(volatile uint64_t* flag, volatile uint64_t* data, uint64_t& f, uint64_t& d) {
  asm volatile ("amoor.d.aq %0, zero, (%2)\n\t"
                "ld %1, 0(%3)"
                : "=&r"(f), "=&r"(d) : "r"(flag), "r"(data) : "memory");
}

// Waits until the warp's earlier accesses are performed: the fence holds the
// load behind it, and the add holds every later instruction behind the load.
INL void settle(volatile uint64_t* own_word) {
  uint64_t t;
  asm volatile ("fence iorw, iorw\n\t"
                "ld %0, 0(%1)\n\t"
                "add %0, %0, %0"
                : "=&r"(t) : "r"(own_word) : "memory");
}

// Each probe reads mcycle, runs its sequence, and reads mcycle again. Issue is
// in order, so the second read waits for whatever stalls the sequence.
#define PROBE_RTT(name, op)                                                  \
  INL uint32_t name(volatile uint64_t* p, uint64_t v) {                      \
    uint64_t t0, t1, old, tmp;                                               \
    asm volatile ("csrr %0, %6\n\t"                                          \
                  op " %2, %5, (%4)\n\t"                                     \
                  "add %3, %2, %2\n\t"                                       \
                  "csrr %1, %6"                                              \
                  : "=&r"(t0), "=&r"(t1), "=&r"(old), "=&r"(tmp)             \
                  : "r"(p), "r"(v), "i"(VX_CSR_MCYCLE) : "memory");          \
    return (uint32_t)(t1 - t0);                                              \
  }
#define PROBE_NORET(name, op)                                                \
  INL uint32_t name(volatile uint64_t* p, uint64_t v) {                      \
    uint64_t t0, t1;                                                         \
    asm volatile ("csrr %0, %4\n\t"                                          \
                  op " zero, %3, (%2)\n\t"                                   \
                  "csrr %1, %4"                                              \
                  : "=&r"(t0), "=&r"(t1)                                     \
                  : "r"(p), "r"(v), "i"(VX_CSR_MCYCLE) : "memory");          \
    return (uint32_t)(t1 - t0);                                              \
  }
#define PROBE_STORE(name, mid, op)                                           \
  INL uint32_t name(volatile uint64_t* r, volatile uint64_t* l, uint64_t v) {\
    uint64_t t0, t1, old, tmp;                                               \
    asm volatile ("csrr %0, %7\n\t"                                          \
                  "sd %6, 0(%4)\n\t"                                         \
                  mid                                                        \
                  op " %2, %6, (%5)\n\t"                                     \
                  "add %3, %2, %2\n\t"                                       \
                  "csrr %1, %7"                                              \
                  : "=&r"(t0), "=&r"(t1), "=&r"(old), "=&r"(tmp)             \
                  : "r"(r), "r"(l), "r"(v), "i"(VX_CSR_MCYCLE) : "memory");  \
    return (uint32_t)(t1 - t0);                                              \
  }

PROBE_RTT(probe_r, "amoadd.d")
PROBE_NORET(probe_a, "amoadd.d.aq")
PROBE_NORET(probe_c, "amoadd.d")
PROBE_RTT(probe_b, "amoswap.d")
PROBE_STORE(probe_s, "fence iorw, iorw\n\t", "amoswap.d")
PROBE_STORE(probe_l, "", "amoswap.d.rl")
PROBE_STORE(probe_l0, "", "amoswap.d")

__kernel void kernel_main(amo_arg_t* __UNIFORM__ arg) {
  vortex::cluster_group cl;
  const uint32_t rank = vx_cluster_block_rank();
  const uint32_t tid  = threadIdx.x;
  volatile uint64_t* own = reinterpret_cast<volatile uint64_t*>(__local_mem());
  volatile uint64_t* own1 = reinterpret_cast<volatile uint64_t*>(
      cl.map_shared_rank(const_cast<uint64_t*>(own), 1));
  volatile uint64_t* own2 = reinterpret_cast<volatile uint64_t*>(
      cl.map_shared_rank(const_cast<uint64_t*>(own), 2));
  volatile uint64_t* own3 = reinterpret_cast<volatile uint64_t*>(
      cl.map_shared_rank(const_cast<uint64_t*>(own), 3));
  const uint32_t nb = blockDim.x;
  const uint32_t words = (arg->mode == MODE_WIDE) ? WIDE_WORDS(nb) : AO_WORDS;
  for (uint32_t i = tid; i < words; i += nb)
    own[i] = 0;
  cl.sync();

  if (arg->mode == MODE_WIDE) {
    const uint64_t n = arg->iters;
    volatile uint64_t* next = reinterpret_cast<volatile uint64_t*>(
        cl.map_shared_rank(const_cast<uint64_t*>(own), (rank + 1) % gridDim.x));
    volatile uint64_t* gctr = reinterpret_cast<volatile uint64_t*>(
        arg->out_addr + gridDim.x * OUT_WORDS * sizeof(uint32_t));
    // Odd lanes hit their own block and even lanes the next rank, in one AMO.
    volatile uint64_t* mix = ((tid & 1) ? own : next) + WIDE_MIX * nb + tid;
    uint32_t e = 0;
    for (uint64_t i = 0; i < n; ++i) {
      uint64_t old;
      asm volatile ("amoadd.d.aqrl zero, %1, (%0)" :: "r"(mix), "r"(1ull) : "memory");
      asm volatile ("amoadd.d.aq %0, %2, (%1)"
                    : "=&r"(old) : "r"(own + WIDE_AQ * nb + tid), "r"(1ull) : "memory");
      e += (old != i);
      asm volatile ("amoadd.d.rl zero, %1, (%0)" :: "r"(next + WIDE_RL * nb + tid), "r"(1ull) : "memory");
      own[WIDE_ST * nb + tid] = i + 1;
      next[WIDE_STR * nb + tid] = i + 1;
      asm volatile ("amoadd.d.aqrl zero, %1, (%0)" :: "r"(gctr), "r"(1ull) : "memory");
      volatile uint64_t* w = own + WIDE_LRSC * nb + tid;
      uint64_t v, fail;
      uint32_t guard = 0;
      do {
        asm volatile ("lr.d.aq %0, (%1)" : "=r"(v) : "r"(w) : "memory");
        asm volatile ("sc.d.rl %0, %2, (%1)" : "=r"(fail) : "r"(w), "r"(v + 1) : "memory");
      } while (fail && ++guard < WIDE_LRSC_TRIES);
      e += (fail != 0);
    }
    cl.sync();
    for (uint32_t k = 0; k < WIDE_KINDS; ++k)
      e += (own[k * nb + tid] != n);
    asm volatile ("amoadd.d zero, %1, (%0)"
                  :: "r"(own + WIDE_ERR * nb), "r"((uint64_t)e) : "memory");
    cl.sync();
    if (tid == 0) {
      uint32_t* out = reinterpret_cast<uint32_t*>(arg->out_addr) + blockIdx.x * OUT_WORDS;
      out[OUT_RANK] = rank;
      out[OUT_CORE] = vx_core_id();
      out[OUT_VIOL] = (uint32_t)own[WIDE_ERR * nb];
      out[OUT_RAN]  = 1;
    }
    return;
  }

  uint32_t viol = 0, reads = 0, work = 0, first_f = 0, first_d = 0;
  const uint64_t n = arg->iters;
  if (arg->mode == MODE_LAT) {
    if (rank == 0 && tid == 0) {
      volatile uint64_t* rw = own1 + AO_LAT;
      volatile uint64_t* lw = own + AO_LAT + 1;
      uint32_t* lat = reinterpret_cast<uint32_t*>(arg->out_addr) + gridDim.x * OUT_WORDS;
      for (uint32_t i = 0; i < LAT_ITERS; ++i) {
        const uint64_t v = i + 1;
        settle(lw);
        lat[LAT_R * LAT_ITERS + i] = probe_r(rw, v);
        settle(lw);
        lat[LAT_A * LAT_ITERS + i] = probe_a(rw, v);
        settle(lw);
        lat[LAT_C * LAT_ITERS + i] = probe_c(rw, v);
        settle(lw);
        lat[LAT_B * LAT_ITERS + i] = probe_b(lw, v);
        settle(lw);
        lat[LAT_S * LAT_ITERS + i] = probe_s(rw, lw, v);
        settle(lw);
        lat[LAT_L * LAT_ITERS + i] = probe_l(rw, lw, v);
        settle(lw);
        lat[LAT_L0 * LAT_ITERS + i] = probe_l0(rw, lw, v);
      }
      settle(lw);
    }
  } else if (rank == 0 && tid == 0) {
    const bool aq = (arg->mode == MODE_MPAQ);
    volatile uint64_t* data = (aq ? own2 : own1) + AO_DATA;
    volatile uint64_t* flag = (aq ? own1 : own2) + AO_FLAG;
    // Nothing between the store and the AMO: the AMO issues right behind it.
    if (arg->fence) {
      for (uint64_t r = 1; r <= n; ++r) { *data = r; vx_fence(); amoswap_d(flag, r); }
    } else {
      for (uint64_t r = 1; r <= n; ++r) { *data = r; amoswap_d_rl(flag, r); }
    }
  } else if (rank == 2 && tid == 0) {
    const bool aq = (arg->mode == MODE_MPAQ);
    uint64_t last = 0;
    while (last < n) {
      uint64_t f = 0, d = 0;
      if (aq) {
        poll_aq(own1 + AO_FLAG, own + AO_DATA, f, d);
        if (f == last)
          continue;
      } else {
        f = own[AO_FLAG];
        if (f == last)
          continue;
        d = own1[AO_DATA];
      }
      last = f;
      ++reads;
      if (d < f && viol++ == 0) {
        first_f = (uint32_t)f;
        first_d = (uint32_t)d;
      }
    }
    own3[AO_DONE] = 1;
    own1[AO_DONE] = 1;
  } else if (rank == 3 && arg->flood) {
    // The stop flag is read once per 64 stores.
    volatile uint64_t* flood1 = own1 + AO_FLOOD;
    do {
      for (uint32_t j = 0; j < 64; ++j)
        flood1[(tid * 8 + j) % 64] = work + j;
      work += 64;
    } while (own[AO_DONE] == 0);
  } else if (rank == 1 && arg->dxa) {
    // DXA beats win every bank, so ingress writes stall and the ingress slots fill.
    uint32_t* tile = reinterpret_cast<uint32_t*>(const_cast<uint64_t*>(own) + AO_TILE);
    vortex::barrier copy_bar(1, 1);
    do {
      copy_bar.expect_tx(1);
      vx_dxa_issue_1d_wg(0, copy_bar.id(), tile, (work % 3) * arg->tile_elems);
      copy_bar.arrive_and_wait();
      ++work;
    } while (own[AO_DONE] == 0);
  } else if (rank == 1 && arg->flood) {
    // Local AMOs keep rank 1's banks busy, so ingress writes queue behind them.
    volatile uint64_t* ham = own + AO_HAMMER;
    do {
      for (uint32_t j = 0; j < 64; ++j)
        __atomic_fetch_add(&ham[tid % 64], 1ull, __ATOMIC_RELAXED);
      work += 64;
    } while (own[AO_DONE] == 0);
  }
  cl.sync();
  if (tid == 0) {
    uint32_t* out = reinterpret_cast<uint32_t*>(arg->out_addr) + blockIdx.x * OUT_WORDS;
    out[OUT_RANK]    = rank;
    out[OUT_CORE]    = vx_core_id();
    out[OUT_VIOL]    = viol;
    out[OUT_READS]   = reads;
    out[OUT_WORK]    = work;
    out[OUT_FIRST_F] = first_f;
    out[OUT_FIRST_D] = first_d;
    out[OUT_RAN]     = 1;
  }
}
