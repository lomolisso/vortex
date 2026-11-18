#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dlmem.h>
#include <vx_dxa.h>
#include <vx_mbar.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

INL uint32_t wait_phase(uint64_t* mbar, uint32_t parity) {
  for (uint32_t t = 0; !vx_mbar_try_wait_parity(mbar, parity); ) {
    if (++t > WAIT_LIMIT)
      return 1;
  }
  return 0;
}

INL void put64(uint32_t* o, uint64_t v) {
  o[0] = (uint32_t)v;
  o[1] = (uint32_t)(v >> 32);
}

template <unsigned K>
INL uint32_t timed_wait() {
  const uint64_t t0 = vx_rdcycle();
  vx_bulk_wait_group<K>();
  return (uint32_t)(vx_rdcycle() - t0);
}

__attribute__((noinline)) uint32_t wait_k(uint32_t k) {
  switch (k) {
  case 0:  return timed_wait<0>();
  case 1:  return timed_wait<1>();
  case 2:  return timed_wait<2>();
  case 3:  return timed_wait<3>();
  case 4:  return timed_wait<4>();
  case 8:  return timed_wait<8>();
  default: return timed_wait<31>();
  }
}

INL void knobs(uint32_t* o) {
#if VX_CFG_H100_FE_ENABLED
  o[0] = VX_CFG_TMA_QUEUE;
  o[1] = VX_CFG_TMA_STORE_QUEUE_OPS;
  o[2] = VX_CFG_BULK_GROUPS;
  o[3] = VX_CFG_SM_EGRESS_BYTES;
#else
  for (uint32_t i = 0; i < 4; ++i)
    o[i] = 0;
#endif
}

#if VX_CFG_H100_FE_ENABLED
// The H100's issue loop: the copy's operands stay in URs and only the source
// advances, so the loop costs four instructions per copy. The issuing warps
// meet at a barrier after their setup: the core issues one instruction per
// cycle with low warp ids first, so the setup of warps beyond ~11 would
// otherwise overlap the first warps' copies and skew their starts. They meet
// again right after their last copy, so the post-loop code of the warps done
// first does not delay the last issuers' timestamps (~540 cycles at 32 warps).
// The eight instructions from barrier to barrier fit in one aligned 32-B block,
// so the line fetched for the first barrier holds the whole loop: a cold
// I-cache line at the loop's back edge would stall a round of copies by a miss.
INL void bulk_loads(uint8_t* dst, const uint8_t* gsrc, uint32_t bytes, uint64_t* mb, uint32_t n, uint32_t stride,
                    uint32_t bar, uint32_t nwarps, uint64_t* t0, uint64_t* t1) {
  size_t a, b, c = n;
  __asm__ volatile (
      VX_UASM_R2UR(0, "%[d]") VX_UASM_R2UR(1, "%[m]") VX_UASM_R2UR(2, "%[s]") VX_UASM_R2UR(3, "%[z]")
      VX_UASM_R2UR(4, "x0") VX_UASM_R2UR(5, "x0") VX_UASM_R2UR(6, "x0") VX_UASM_R2UR(7, "x0")
      VX_UASM_R2UR(10, "%[st]")
      ".p2align 5\n\t"
      ".insn r %[custom0], 4, 0, x0, %[bar], %[nw]\n\t"
      "csrr %[a], mcycle\n\t"
      "1:\n\t"
      ".insn r 0x0b, 3, " VX_UASM_STR(VX_FE_F7_DXA_UR) ", x0, x0, x0\n\t"
      VX_UASM_UALU(0, 2, 2, 10)
      "addi %[c], %[c], -1\n\t"
      "bnez %[c], 1b\n\t"
      "csrr %[b], mcycle\n\t"
      ".insn r %[custom0], 4, 0, x0, %[bar], %[nw]\n\t"
      : [a] "=&r"(a), [b] "=&r"(b), [c] "+r"(c)
      : [d] "r"(dst), [m] "r"(mb), [s] "r"(gsrc), [z] "r"((size_t)bytes), [st] "r"((size_t)stride),
        [bar] "r"((size_t)bar), [nw] "r"((size_t)nwarps), [custom0] "i"(RISCV_CUSTOM0)
      : "memory");
  *t0 = a;
  *t1 = b;
}
#endif

// Run twice: a first pass of one copy per warp brings the timed pass's code into
// the I-cache, so no cold line between the last issue and the wait's end adds a
// fetch whose cost follows the memory state the copies left (~200-400 cycles).
__attribute__((noinline)) void run_intake(const eng_arg_t* arg, uint8_t* lm, uint32_t w, uint32_t copies,
                                          uint32_t parity, uint32_t* o) {
  auto mb = reinterpret_cast<uint64_t*>(lm + LM_MBAR + 8 * w);
  const uint32_t bytes = arg->bytes;
  if (parity)
    vx_mbar_arrive_tx(mb, copies * bytes);
  const uint8_t* gsrc = reinterpret_cast<const uint8_t*>(arg->gsrc_addr) + (uint64_t)w * copies * arg->src_stride;
  uint64_t t0, t1;
#if VX_CFG_H100_FE_ENABLED
  bulk_loads(lm + LM_DATA, gsrc, bytes, mb, copies, arg->src_stride, get_local_group_id() + 1, arg->warps, &t0, &t1);
#else
  t0 = vx_rdcycle();
  for (uint32_t p = 0; p < copies; ++p)
    vx_dxa_bulk_load(lm + LM_DATA, gsrc + (uint64_t)p * arg->src_stride, bytes, mb);
  t1 = vx_rdcycle();
#endif
  const uint32_t err = wait_phase(mb, parity);
  const uint64_t t2 = vx_rdcycle();
  put64(o + O_WARP + w * O_WARP_W, t0);
  put64(o + O_WARP + w * O_WARP_W + 2, t1);
  put64(o + O_WARP + w * O_WARP_W + 4, t2);
  o[O_ERR] |= err;
}

// The commit times go to local memory: a global store in the timed loop would
// wait for its line and pace the issue. A first wait with nothing pending
// fetches the timed wait's code and jump table, whose cold misses (~450 cycles
// each in the H100 organization) would otherwise land in the timed wait; a
// second one gives the timed window's own cost.
__attribute__((noinline)) void run_store(const eng_arg_t* arg, uint8_t* lm, uint32_t* o) {
  uint8_t* gdst = reinterpret_cast<uint8_t*>(arg->gdst_addr);
  const uint32_t groups = arg->groups, copies = arg->copies, bytes = arg->bytes, waitk = arg->waitk;
  auto ts = reinterpret_cast<volatile uint32_t*>(lm + LM_TS);
  wait_k(waitk);
  const uint32_t tb = wait_k(waitk);
  const uint64_t t0 = vx_rdcycle();
  for (uint32_t g = 0; g < groups; ++g) {
    for (uint32_t s = 0; s < copies; ++s)
      vx_dxa_bulk_store(gdst + (uint64_t)(g * copies + s) * bytes, lm + LM_DATA, bytes);
    vx_bulk_commit_group();
    ts[g] = (uint32_t)(vx_rdcycle() - t0);
  }
  const uint32_t tk = wait_k(waitk);
  const uint32_t t0w = timed_wait<0>();
  for (uint32_t g = 0; g < groups; ++g)
    o[O_COMMIT + g] = ts[g];
  o[O_WAITK] = tk;
  o[O_WAIT0] = t0w;
  o[O_WAITB] = tb;
}

__attribute__((noinline)) void run_reduce(const eng_arg_t* arg, uint8_t* lm, uint32_t* o) {
  void* gdst = reinterpret_cast<void*>(arg->gdst_addr);
  const uint32_t bytes = arg->bytes;
  const uint64_t t0 = vx_rdcycle();
  vx_dxa_reduce_global(gdst, lm + LM_DATA, bytes, DXA_RED_ADD, DXA_RT_U32);
  vx_bulk_commit_group();
  vx_bulk_wait_group<0>();
  o[O_T1] = (uint32_t)(vx_rdcycle() - t0);
}

__attribute__((noinline)) void read_pass(void* gdst, const uint8_t* src, uint32_t bytes, uint32_t* o) {
  const uint64_t t0 = vx_rdcycle();
  vx_dxa_bulk_store(gdst, src, bytes);
  vx_bulk_commit_group();
  vx_bulk_wait_group_read<0>();
  const uint64_t t1 = vx_rdcycle();
  vx_bulk_wait_group<0>();
  const uint64_t t2 = vx_rdcycle();
  o[O_T1] = (uint32_t)(t1 - t0);
  o[O_T2] = (uint32_t)(t2 - t1);
}

__attribute__((noinline)) void full_pass(void* gdst, const uint8_t* src, uint32_t bytes, uint32_t* o) {
  const uint64_t t0 = vx_rdcycle();
  vx_dxa_bulk_store(gdst, src, bytes);
  vx_bulk_commit_group();
  vx_bulk_wait_group<0>();
  o[O_T3] = (uint32_t)(vx_rdcycle() - t0);
}

// The first pass warms the I-cache: a cold line after a wait would add a
// DRAM fetch (~530 cycles) to the window, depending on the code layout.
__attribute__((noinline)) void run_read(const eng_arg_t* arg, uint8_t* lm, uint32_t* o) {
  auto gdst = reinterpret_cast<uint8_t*>(arg->gdst_addr);
  read_pass(gdst, lm + LM_DATA, arg->bytes, o);
  full_pass(gdst, lm + LM_DATA, arg->bytes, o);
  read_pass(gdst + (arg->cold ? COLD_OFFSET(arg->bytes) : 0), lm + LM_DATA, arg->bytes, o);
  if (!arg->cold)
    full_pass(gdst, lm + LM_DATA, arg->bytes, o);
}

} // namespace

__kernel void eng_k(eng_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t nthr = blockDim.x;
  const uint32_t w    = get_sub_group_id();
  const uint32_t lane = threadIdx.x % (uint32_t)vx_num_threads();
  auto lm  = reinterpret_cast<uint8_t*>(__local_mem());
  auto o   = reinterpret_cast<uint32_t*>(arg->out_addr);

  auto lw = reinterpret_cast<volatile uint32_t*>(lm + LM_DATA);
  for (uint32_t i = tid; i < arg->bytes / 4; i += nthr)
    lw[i] = 0x01000000u + i * 3u;
  if (tid < MAX_WARPS)
    vx_mbar_init(reinterpret_cast<uint64_t*>(lm + LM_MBAR + 8 * tid), 1);
  // Every warp's expected bytes are set before the barrier, so no warp's
  // first copy waits for the other warps' mbarrier operations.
  // The H100 measurement copies L2-warm lines (14 M7): one load per copy's
  // source line brings it into the L2 before the timed copies (the sum makes
  // each thread wait for its loads before the barrier).
  if (arg->mode == M_INTAKE) {
    auto src = reinterpret_cast<const volatile uint32_t*>(arg->gsrc_addr);
    uint32_t acc = 0;
    for (uint32_t i = tid; i < arg->warps * arg->copies; i += nthr)
      acc += src[(uint64_t)i * arg->src_stride / 4];
    __asm__ volatile ("" :: "r"(acc));
  }
  // The H100 stores went to L2-warm lines as well (14 M6): a store to a cold
  // line can cost a write-allocating L2 a fill, so the drain would follow the
  // DRAM round trip instead of the egress. One load per 32-B sector.
  if (arg->mode == M_STORE) {
    auto dst = reinterpret_cast<const volatile uint32_t*>(arg->gdst_addr);
    const uint32_t words = arg->groups * arg->copies * arg->bytes / 4;
    uint32_t acc = 0;
    for (uint32_t i = tid * 8; i < words; i += nthr * 8)
      acc += dst[i];
    __asm__ volatile ("" :: "r"(acc));
  }
  if (arg->mode == M_INTAKE && tid < arg->warps)
    vx_mbar_arrive_tx(reinterpret_cast<uint64_t*>(lm + LM_MBAR + 8 * tid), arg->bytes);
  if (tid == 0) {
    o[O_ERR] = 0;
    knobs(o + O_KNOBS);
  }
  vx_fence_proxy_async();
  __syncthreads();

  if (lane == 0) {
    if (arg->mode == M_INTAKE) {
      if (w < arg->warps) {
        run_intake(arg, lm, w, 1, 0, o);
        run_intake(arg, lm, w, arg->copies, 1, o);
      }
    } else if (w == 0) {
      if (arg->mode == M_STORE)
        run_store(arg, lm, o);
      else if (arg->mode == M_REDUCE)
        run_reduce(arg, lm, o);
      else
        run_read(arg, lm, o);
    }
  }
  __syncthreads();
}
