#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dlmem.h>
#include <vx_mbar.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

INL void idle_cycles(uint64_t n) {
  const uint64_t t0 = vx_rdcycle();
  while (vx_rdcycle() - t0 < n) {}
}

INL void record(const sa_arg_t* arg, uint32_t b, uint32_t rank) {
  auto rec = reinterpret_cast<uint32_t*>(arg->rec_addr) + b * REC_WORDS;
  rec[REC_BLOCK] = b;
  rec[REC_CORE]  = vx_core_id();
  rec[REC_SLOT]  = get_local_group_id();
  rec[REC_RANK]  = rank;
  __atomic_fetch_add(&rec[REC_RAN], 1u, __ATOMIC_RELAXED);
}

// The bank's own answer to a try_wait: false when it read the phase
// incomplete, and then the warp has slept.
INL bool bank_try_wait(uint64_t* obj, uint64_t tok) {
  uint64_t r;
  __asm__ volatile (".insn r %1, 5, 0, %0, %2, %3"
                    : "=r"(r) : "i"(RISCV_CUSTOM2), "r"(obj), "r"(tok) : "memory");
  return r != 0;
}

// A wait for the phase of parity `ph` that runs the same instructions
// whether or not the try_wait slept, so the instruction count does not
// depend on timing. Returns 1 if it gave up.
INL uint32_t wait_phase(uint64_t* mbar, uint32_t ph, uint32_t& tries) {
  const uint64_t tok = (uint64_t)(ph & 1) << 63;
  for (uint32_t n = 1; ; ++n) {
    ++tries;
    const bool raw    = bank_try_wait(mbar, tok);
    const bool tested = vx_mbar_test_wait(mbar, tok);
    if (raw | tested)
      return 0;
    if (n >= WAIT_LIMIT)
      return 1;
  }
}

} // namespace

// Each round, every block arms one mbarrier per op it receives with
// arrive_tx of the bytes coming in, and the cluster barrier publishes that.
// Then every thread issues its st.async (red.async) ops into the destination
// block, each completing 4 or 8 bytes on that block's mbarrier, with no fence;
// every thread waits for its own block's phases and checks right away that
// each stored word holds the round's value and each counter the total so far.
// At the end every block dumps its whole local memory for the host.
__kernel void st_k(sa_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t T    = blockDim.x;
  const uint32_t b    = blockIdx.x;
  const uint32_t k    = arg->k;
  const uint32_t n    = arg->elems;
  const uint32_t mode = arg->mode;
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t r    = cl.block_rank();
  const uint32_t base = b - r;
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto lw   = reinterpret_cast<volatile uint32_t*>(lm);
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);
  auto st32 = reinterpret_cast<volatile uint32_t*>(lm + LM_ST32);
  auto st64 = reinterpret_cast<volatile uint64_t*>(lm + lm_st64(T, n));
  auto ctr32 = reinterpret_cast<volatile uint32_t*>(lm + LM_CTR32);
  auto ctr64 = reinterpret_cast<volatile uint64_t*>(lm + LM_CTR64);
  const uint32_t words = arg->lmem_bytes / 4;

  if (tid == 0)
    record(arg, b, r);
  for (uint32_t w = LM_CANARY + tid; w < words; w += T)
    lw[w] = canary_word(b, w);
  if (tid == 0) {
    *reinterpret_cast<volatile uint64_t*>(lm + LM_CTR32) = 0;
    *ctr64 = 0;
    for (uint32_t o = 0; o < NUM_OPS; ++o)
      vx_mbar_init(mbar + o, 1);
  }

  // Ops per thread that land in this block, by op: stores from one sender,
  // adds from red_senders() of them.
  const uint32_t st_in  = n * T;
  const uint32_t red_in = red_senders(mode, r, k) * n * T;
  const uint32_t q_st   = st_dst(mode, r, k);
  const uint32_t q_red  = red_dst(mode, r);
  const uint32_t src    = base + (mode == MODE_RING ? (r + k - 1) % k : r);
  auto peer_lm  = cl.map_shared_rank(lm, q_st);
  auto peer_red = cl.map_shared_rank(lm, q_red);
  auto peer_mbar = reinterpret_cast<uint64_t*>(peer_lm + LM_MBAR);
  auto red_mbar  = reinterpret_cast<uint64_t*>(peer_red + LM_MBAR);

  uint32_t gave_up = 0, stale = 0, bad_red = 0, tries = 0;
  uint32_t want32 = 0;
  uint64_t want64 = 0;
  for (uint32_t p = 0; p < arg->rounds; ++p) {
    if (tid == 0) {
      for (uint32_t o = 0; o < NUM_OPS; ++o) {
        const uint32_t in = (o == OP_ST32 || o == OP_ST64) ? st_in : red_in;
        if (((arg->ops >> o) & 1u) && in)
          vx_mbar_arrive_tx(mbar + o, in * op_bytes(o));
      }
    }
    cl.sync();

    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t w = i * T + tid;
      if (arg->ops & (1u << OP_ST32))
        vx_st_async_u32(peer_lm + LM_ST32 + w * 4, st32_value(p, b, tid, i), peer_mbar + OP_ST32);
      if (arg->ops & (1u << OP_ST64))
        vx_st_async_u64(peer_lm + lm_st64(T, n) + w * 8, st64_value(p, b, tid, i), peer_mbar + OP_ST64);
      if (arg->ops & (1u << OP_RED32))
        vx_red_async_add_u32(peer_red + LM_CTR32, red32_value(p, b, tid, i), red_mbar + OP_RED32);
      if (arg->ops & (1u << OP_RED64))
        vx_red_async_add_u64(peer_red + LM_CTR64, red64_value(p, b, tid, i), red_mbar + OP_RED64);
    }

    for (uint32_t o = 0; o < NUM_OPS; ++o) {
      const uint32_t in = (o == OP_ST32 || o == OP_ST64) ? st_in : red_in;
      if (((arg->ops >> o) & 1u) && in)
        gave_up += wait_phase(mbar + o, p, tries);
    }

    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t w = i * T + tid;
      if ((arg->ops & (1u << OP_ST32)) && st32[w] != st32_value(p, src, tid, i))
        ++stale;
      if ((arg->ops & (1u << OP_ST64)) && st64[w] != st64_value(p, src, tid, i))
        ++stale;
    }
    // Thread 0 adds up every sender's contributions of the round.
    if (tid == 0 && red_in) {
      for (uint32_t s = 0; s < k; ++s) {
        if (red_dst(mode, s) != r)
          continue;
        for (uint32_t t = 0; t < T; ++t) {
          for (uint32_t i = 0; i < n; ++i) {
            want32 += red32_value(p, base + s, t, i);
            want64 += red64_value(p, base + s, t, i);
          }
        }
      }
      if ((arg->ops & (1u << OP_RED32)) && *ctr32 != want32)
        ++bad_red;
      if ((arg->ops & (1u << OP_RED64)) && *ctr64 != want64)
        ++bad_red;
    }
  }

  cl.sync();
  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr + (uint64_t)b * arg->lmem_bytes);
  for (uint32_t w = tid; w < words; w += T)
    dump[w] = lw[w];

  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + (b * T + tid) * ERR_WORDS;
  out[ERR_WAIT]  = gave_up;
  out[ERR_STALE] = stale;
  out[ERR_RED]   = bad_red;
  out[ERR_TRIES] = tries;
  out[ERR_DONE]  = 1;
}

// Runs the simulator must stop: rank 0 issues one st.async; rank 1 stays
// alive meanwhile.
__kernel void neg_k(sa_arg_t* __UNIFORM__ arg) {
  const uint32_t tid = threadIdx.x;
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t r = cl.block_rank();
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);

  if (tid == 0) {
    record(arg, blockIdx.x, r);
    vx_mbar_init(mbar, 1);
  }
  cl.sync();
  if (r == 0 && tid == 0) {
    auto peer      = cl.map_shared_rank(lm, 1);
    auto peer_mbar = reinterpret_cast<uint64_t*>(peer + LM_MBAR);
    switch (arg->neg) {
    case NEG_LOCAL:
      vx_st_async_u32(lm + LM_ST32, 1, peer_mbar);
      break;
    case NEG_MBAR:
      vx_st_async_u32(peer + LM_ST32, 1, cl.map_shared_rank(mbar, 0));
      break;
    case NEG_MBARLOCAL:
      vx_st_async_u32(peer + LM_ST32, 1, mbar);
      break;
    case NEG_RANK: {
      auto bad = reinterpret_cast<uint8_t*>((uintptr_t)VX_MEM_DLMEM_BASE_ADDR + arg->neg_win);
      vx_st_async_u32(bad + LM_ST32, 1, reinterpret_cast<uint64_t*>(bad + LM_MBAR));
      break;
    }
    case NEG_OFF:
      vx_st_async_u32(peer + arg->neg_win, 1, peer_mbar);
      break;
    default:
      vx_st_async_u32(peer + LM_ST32, 1, peer_mbar);
      break;
    }
  }
  idle_cycles(arg->linger);
  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + (blockIdx.x * blockDim.x + tid) * ERR_WORDS;
  out[ERR_DONE] = 1;
}
