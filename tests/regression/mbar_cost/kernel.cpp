#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_mbar.h>
#include <vx_dxa.h>

#include "common.h"

#define INL static inline __attribute__((always_inline))

#define R4(x)  x x x x
#define R16(x) R4(R4(x))
#define R64(x) R4(R16(x))

// One hop: the op on *p, its result cleared and added to the base, so the next op's address waits for it.
#define HOP ".insn r %4, %5, 0, %1, %0, %3\n and %1, %1, zero\n add %0, %1, %2\n"

INL uint64_t opaque(uint64_t v) {
  __asm__ volatile ("" : "+r"(v));
  return v;
}

template <uint32_t F3>
INL uint64_t chase(uint64_t* bar, uint64_t v, uint32_t reps) {
  const uint64_t b = reinterpret_cast<uint64_t>(bar);
  uint64_t p = b, r = 0;
  // Untimed group first: instruction fetch and the barrier word's first access stay out of the window.
  __asm__ volatile (R64(HOP) : "+r"(p), "=&r"(r) : "r"(b), "r"(v), "i"(RISCV_CUSTOM2), "i"(F3) : "memory");
  const uint64_t t0 = vx_rdcycle();
  for (uint32_t i = 0; i < reps; ++i)
    __asm__ volatile (R64(HOP) : "+r"(p), "=&r"(r) : "r"(b), "r"(v), "i"(RISCV_CUSTOM2), "i"(F3) : "memory");
  const uint64_t t1 = vx_rdcycle();
  return (t1 - t0) + (p - b);
}

// The chase's AND + ADD after a response: a load chase with the same two ops
// minus a bare load chase.
#define LHOP "ld %1, 0(%0)\n and %1, %1, zero\n add %0, %1, %2\n"
#define LBARE "ld %0, 0(%0)\n"

INL uint64_t alu_after_load(uint64_t* w, uint32_t reps) {
  const uint64_t b = reinterpret_cast<uint64_t>(w);
  uint64_t p = b, r = 0;
  __asm__ volatile (R64(LHOP) : "+r"(p), "=&r"(r) : "r"(b) : "memory");
  uint64_t t0 = vx_rdcycle();
  for (uint32_t i = 0; i < reps; ++i)
    __asm__ volatile (R64(LHOP) : "+r"(p), "=&r"(r) : "r"(b) : "memory");
  const uint64_t ta = vx_rdcycle() - t0;
  uint64_t q = b;
  __asm__ volatile (R64(LBARE) : "+r"(q) :: "memory");
  t0 = vx_rdcycle();
  for (uint32_t i = 0; i < reps; ++i)
    __asm__ volatile (R64(LBARE) : "+r"(q) :: "memory");
  const uint64_t tb = vx_rdcycle() - t0;
  return ta - tb + (p - b) + (q - b);
}

// rd = x0, as the H100's SASS writes RZ: a written result register would hold
// each op for the previous one's response.
#define TX0(f3, bar, v) __asm__ volatile (".insn r %0, %1, 0, zero, %2, %3" :: "i"(RISCV_CUSTOM2), "i"(f3), "r"(bar), "r"(v) : "memory");
#define ARRIVE_RELAXED0(bar, v) __asm__ volatile (".insn r %0, 0, 1, zero, %1, %2" :: "i"(RISCV_CUSTOM2), "r"(bar), "r"(v) : "memory");

#define S4(x)  x x x x
#define S16(x) S4(S4(x))
#define S64(x) S4(S16(x))

template <uint32_t K>
INL void issue_group(uint64_t* bar, uint64_t one) {
  if (K == MC_IS_EXPECT_TX) {
    S64(TX0(2, bar, one))
  } else {
    S64(TX0(3, bar, one))
  }
}

// Every warp's lane 0 streams, so instruction issue is not the bound.
template <uint32_t K>
INL uint64_t issue_stream(uint64_t* bar, uint32_t reps, bool lead) {
  const uint64_t one = opaque(1);
  if (lead) issue_group<K>(bar, one);
  __syncthreads();
  const uint64_t t0 = vx_rdcycle();
  if (lead)
    for (uint32_t i = 0; i < reps; ++i)
      issue_group<K>(bar, one);
  __syncthreads();
  return vx_rdcycle() - t0;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto bars = reinterpret_cast<uint64_t*>(__local_mem());
  const uint32_t tid = threadIdx.x;
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
  // Pending objects never complete within a run: the count covers every arrival and drop.
  constexpr uint32_t BIG = (1u << 20) - 1;

  switch (arg->kind) {
  case MC_LAT:
    if (tid == 0) {
      for (uint32_t i = 0; i < 8; ++i)
        vx_mbar_init(&bars[i], BIG);
      vx_mbar_init(&bars[8], 1);
      bars[9] = reinterpret_cast<uint64_t>(&bars[9]);
    }
    __syncthreads();
    if (tid == 0) {
      (void)vx_mbar_arrive(&bars[8], 1);  // phase 0 of bars[8] completes
      const uint32_t n = arg->reps;
      out[MC_OP_ARRIVE]       = chase<0>(&bars[0], 1, n);
      out[MC_OP_ARRIVE_TX]    = chase<1>(&bars[1], 1, n);
      out[MC_OP_ARRIVE_DROP]  = chase<0>(&bars[2], 1 | __VX_MBAR_DROP, n);
      out[MC_OP_TEST_WAIT]    = chase<4>(&bars[8], 0, n);
      out[MC_OP_TRY_WAIT]     = chase<5>(&bars[8], 0, n);
      out[MC_OP_TEST_PENDING] = chase<4>(&bars[3], 0, n);
      out[MC_OP_ALU]          = alu_after_load(&bars[9], n);
    }
    break;
  case MC_ISSUE: {
    if (tid == 0) {
      vx_mbar_init(&bars[1], BIG);
      vx_mbar_init(&bars[2], BIG);
    }
    __syncthreads();
    const bool lead = (tid % vx_num_threads()) == 0;
    const uint32_t n = arg->reps;
    const uint64_t et = issue_stream<MC_IS_EXPECT_TX>(&bars[1], n, lead);
    const uint64_t ct = issue_stream<MC_IS_COMPLETE_TX>(&bars[2], n, lead);
    if (tid == 0) {
      out[MC_IS_EXPECT_TX]   = et;
      out[MC_IS_COMPLETE_TX] = ct;
    }
    break;
  }
  case MC_BULK: {
    uint64_t* bar = &bars[0];
    volatile uint32_t* stop = reinterpret_cast<volatile uint32_t*>(&bars[2]);
    char* dst = reinterpret_cast<char*>(&bars[16]);
    const void* src = reinterpret_cast<const void*>(arg->src_addr);
    const uint32_t copies = arg->reps * MC_UNROLL;
    if (tid == 0) {
      vx_mbar_init(bar, 1);
      vx_mbar_init(&bars[1], BIG);
      *stop = 0;
    }
    __syncthreads();
    // Warp 0 issues converged: the copy's operand gather also writes the
    // inactive lanes, which would clobber a value a parked lane still holds.
    if (tid < (uint32_t)vx_num_threads()) {
      uint64_t t = 0;
      // Phase 0 warms the copy path; phase 1 is timed.
      for (uint32_t phase = 0; phase < 2; ++phase) {
        if (tid == 0)
          (void)vx_mbar_arrive_tx(bar, copies * 128);
        const uint64_t t0 = vx_rdcycle();
        for (uint32_t i = 0; i < copies; ++i)
          vx_dxa_bulk_load(dst, src, 128, bar);
        while (!vx_mbar_try_wait_parity(bar, phase)) {}
        t = vx_rdcycle() - t0;
      }
      if (tid == 0) {
        out[0] = t;
        *stop = 1;
      }
    } else if (arg->nbars && tid == (uint32_t)vx_num_threads()) {
      while (*stop == 0) {
        R16((void)vx_mbar_complete_tx(&bars[1], 1);)
      }
    }
    break;
  }
  default: {
    // Barrier j is the 8-B word j: banks 2j and 2j + 1, one bank pair each.
    if (tid < arg->nbars)
      vx_mbar_init(&bars[tid], BIG);
    __syncthreads();
    uint64_t* mine = &bars[tid % arg->nbars];
    const uint64_t one = opaque(1);
    R16(ARRIVE_RELAXED0(mine, one))
    __syncthreads();
    const uint64_t t0 = vx_rdcycle();
    for (uint32_t i = 0; i < arg->reps; ++i) {
      R16(ARRIVE_RELAXED0(mine, one))
    }
    // The last arrivals are counted only once performed.
    uint64_t w = vx_mbar_test_wait(mine, 0);
    __asm__ volatile ("add %0, %0, zero" : "+r"(w));
    __syncthreads();
    const uint64_t t1 = vx_rdcycle() + (w & 0);
    if (tid == 0)
      out[0] = t1 - t0;
    break;
  }
  }
}
