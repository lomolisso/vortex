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

template <uint32_t V> struct op_c { static constexpr uint32_t value = V; };

// Calls f(op_c<o>()) for every op o in the uniform mask `ops`, so each op's
// code is specialized at compile time; D5's ops (bits 0-3) are tested first
// and the others behind one test.
template <typename F>
INL void for_ops(uint32_t ops, F&& f) {
  if (ops & (1u << 0)) f(op_c<0>());
  if (ops & (1u << 1)) f(op_c<1>());
  if (ops & (1u << 2)) f(op_c<2>());
  if (ops & (1u << 3)) f(op_c<3>());
  if (ops & ~0xFu) {
    if (ops & (1u << 4))  f(op_c<4>());
    if (ops & (1u << 5))  f(op_c<5>());
    if (ops & (1u << 6))  f(op_c<6>());
    if (ops & (1u << 7))  f(op_c<7>());
    if (ops & (1u << 8))  f(op_c<8>());
    if (ops & (1u << 9))  f(op_c<9>());
    if (ops & (1u << 10)) f(op_c<10>());
    if (ops & (1u << 11)) f(op_c<11>());
    if (ops & (1u << 12)) f(op_c<12>());
    if (ops & (1u << 13)) f(op_c<13>());
    if (ops & (1u << 14)) f(op_c<14>());
    if (ops & (1u << 15)) f(op_c<15>());
    if (ops & (1u << 16)) f(op_c<16>());
  }
}

// Op o of sender block b's thread t, element i of round p, into `dst` (the
// element's or the counter's window address).
INL void issue(uint32_t o, uint8_t* dst, uint64_t* pmbar, uint32_t p, uint32_t b, uint32_t t, uint32_t i) {
  switch (o) {
  case OP_ST32:
    vx_st_async_u32(dst, st_word(o, p, b, t, i, 0), pmbar);
    break;
  case OP_ST64:
    vx_st_async_u64(dst, ((uint64_t)st_word(o, p, b, t, i, 1) << 32) | st_word(o, p, b, t, i, 0), pmbar);
    break;
  case OP_RED64:
    vx_red_async_add_u64(dst, red64_value(p, b, t, i), pmbar);
    break;
  case OP_V2B32:
    vx_st_async_v2_u32(dst, st_word(o, p, b, t, i, 0), st_word(o, p, b, t, i, 1), pmbar);
    break;
  case OP_V4B32:
    vx_st_async_v4_u32(dst, st_word(o, p, b, t, i, 0), st_word(o, p, b, t, i, 1), st_word(o, p, b, t, i, 2),
                       st_word(o, p, b, t, i, 3), pmbar);
    break;
  case OP_V2B64:
    vx_st_async_v2_u64(dst, ((uint64_t)st_word(o, p, b, t, i, 1) << 32) | st_word(o, p, b, t, i, 0),
                       ((uint64_t)st_word(o, p, b, t, i, 3) << 32) | st_word(o, p, b, t, i, 2), pmbar);
    break;
  default: {
    const uint32_t v = red_value(o, p, b, t, i);
    switch (o) {
    case OP_RED32:  vx_red_async_add_u32(dst, v, pmbar); break;
    case OP_ADDS32: vx_red_async_add_s32(dst, (int32_t)v, pmbar); break;
    case OP_MINS32: vx_red_async_min_s32(dst, (int32_t)v, pmbar); break;
    case OP_MAXS32: vx_red_async_max_s32(dst, (int32_t)v, pmbar); break;
    case OP_MINU32: vx_red_async_min_u32(dst, v, pmbar); break;
    case OP_MAXU32: vx_red_async_max_u32(dst, v, pmbar); break;
    case OP_AND:    vx_red_async_and_b32(dst, v, pmbar); break;
    case OP_OR:     vx_red_async_or_b32(dst, v, pmbar); break;
    case OP_XOR:    vx_red_async_xor_b32(dst, v, pmbar); break;
#if VX_CFG_EXT_FPAMO_ENABLED
    case OP_INC:    vx_red_async_inc_u32(dst, v, pmbar); break;
    case OP_DEC:    vx_red_async_dec_u32(dst, v, pmbar); break;
#endif
    default: break;
    }
  } break;
  }
}

} // namespace

// Each round, every block arms one mbarrier per op it receives with
// arrive_tx of the bytes coming in, and the cluster barrier publishes that.
// Then every thread issues its st.async (red.async) ops into the destination
// block, each completing its bytes on that block's mbarrier (a 16-byte store
// as two 8-byte halves), with no fence; every thread waits for its own
// block's phases and checks right away that each stored word holds the
// round's value, and thread 0 that each counter holds the result the host
// computed for the round. At the end every block dumps its whole local
// memory for the host.
__kernel void st_k(sa_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t T    = blockDim.x;
  const uint32_t b    = blockIdx.x;
  const uint32_t k    = arg->k;
  const uint32_t n    = arg->elems;
  const uint32_t ops  = arg->ops;
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t r    = cl.block_rank();
  const uint32_t base = b - r;
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto lw   = reinterpret_cast<volatile uint32_t*>(lm);
  const uint32_t words = arg->lmem_bytes / 4;

  if (tid == 0)
    record(arg, b, r);
  for (uint32_t w = arg->data_off / 4 + tid; w < words; w += T)
    lw[w] = canary_word(b, w);
  if (tid == 0) {
    for_ops(ops, [&](auto oc) {
      constexpr uint32_t o = decltype(oc)::value;
      vx_mbar_init(reinterpret_cast<uint64_t*>(lm + arg->mbar[o]), 1);
      if (op_is_red(o))
        *reinterpret_cast<volatile uint64_t*>(lm + (arg->area[o] & ~7u)) = ctr_word_init(b, o, arg->area[o]);
    });
  }

  // Ops per thread that land in this block, by kind: stores from one
  // sender, reductions from red_senders() of them.
  const uint32_t st_in  = n * T;
  const uint32_t red_in = red_senders(r, k) * n * T;
  const uint32_t src    = base + st_src(r, k);
  auto peer_st  = cl.map_shared_rank(lm, st_dst(r, k));
  auto peer_red = cl.map_shared_rank(lm, red_dst(r));
  auto want = reinterpret_cast<const uint64_t*>(arg->want_addr) + (uint64_t)b * arg->rounds * NUM_OPS;

  uint32_t gave_up = 0, stale = 0, bad_red = 0, tries = 0;
  for (uint32_t p = 0; p < arg->rounds; ++p) {
    if (tid == 0) {
      for_ops(ops, [&](auto oc) {
        constexpr uint32_t o = decltype(oc)::value;
        const uint32_t in = op_is_red(o) ? red_in : st_in;
        if (in)
          vx_mbar_arrive_tx(reinterpret_cast<uint64_t*>(lm + arg->mbar[o]), in * op_bytes(o));
      });
    }
    cl.sync();

    for (uint32_t i = 0; i < n; ++i) {
      for_ops(ops, [&](auto oc) {
        constexpr uint32_t o = decltype(oc)::value;
        if (op_is_red(o))
          issue(o, peer_red + arg->area[o], reinterpret_cast<uint64_t*>(peer_red + arg->mbar[o]), p, b, tid, i);
        else
          issue(o, peer_st + arg->area[o] + (i * T + tid) * op_bytes(o),
                reinterpret_cast<uint64_t*>(peer_st + arg->mbar[o]), p, b, tid, i);
      });
    }

    for_ops(ops, [&](auto oc) {
      constexpr uint32_t o = decltype(oc)::value;
      if (op_is_red(o) ? red_in : st_in)
        gave_up += wait_phase(reinterpret_cast<uint64_t*>(lm + arg->mbar[o]), p, tries);
    });

    for_ops(ops & OPS_ST, [&](auto oc) {
      constexpr uint32_t o = decltype(oc)::value;
      for (uint32_t i = 0; i < n; ++i) {
        auto e = reinterpret_cast<volatile uint32_t*>(lm + arg->area[o] + (i * T + tid) * op_bytes(o));
        for (uint32_t j = 0; j < op_bytes(o) / 4; ++j) {
          if (e[j] != st_word(o, p, src, tid, i, j))
            ++stale;
        }
      }
    });
    if (tid == 0 && red_in) {
      for_ops(ops & OPS_RED, [&](auto oc) {
        constexpr uint32_t o = decltype(oc)::value;
        const uint64_t got = (o == OP_RED64) ? *reinterpret_cast<volatile uint64_t*>(lm + arg->area[o])
                                             : lw[arg->area[o] / 4];
        if (got != want[p * NUM_OPS + o])
          ++bad_red;
      });
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
  auto mbar = reinterpret_cast<uint64_t*>(lm + arg->mbar[0]);
  const uint32_t dst = arg->area[0];

  if (tid == 0) {
    record(arg, blockIdx.x, r);
    vx_mbar_init(mbar, 1);
  }
  cl.sync();
  if (r == 0 && tid == 0) {
    auto peer      = cl.map_shared_rank(lm, 1);
    auto peer_mbar = reinterpret_cast<uint64_t*>(peer + arg->mbar[0]);
    switch (arg->neg) {
    case NEG_LOCAL:
      vx_st_async_u32(lm + dst, 1, peer_mbar);
      break;
    case NEG_MBAR:
      vx_st_async_u32(peer + dst, 1, cl.map_shared_rank(mbar, 0));
      break;
    case NEG_MBARLOCAL:
      vx_st_async_u32(peer + dst, 1, mbar);
      break;
    case NEG_RANK: {
      auto bad = reinterpret_cast<uint8_t*>((uintptr_t)VX_MEM_DLMEM_BASE_ADDR + arg->neg_win);
      vx_st_async_u32(bad + dst, 1, reinterpret_cast<uint64_t*>(bad + arg->mbar[0]));
      break;
    }
    case NEG_OFF:
      vx_st_async_u32(peer + arg->neg_win, 1, peer_mbar);
      break;
    case NEG_SELF:
      vx_st_async_u32(cl.map_shared_rank(lm, 0) + dst, 1, cl.map_shared_rank(mbar, 0));
      break;
    case NEG_ALIGN16:
      vx_st_async_v4_u32(peer + dst + 8, 1, 2, 3, 4, peer_mbar);
      break;
    case NEG_ODDPAIR: {
      // The 16-byte form with its data in a5, the odd register of a pair.
      register uint64_t odd __asm__("a5") = 1;
      __asm__ volatile (".insn r %0, 4, %1, %2, %3, %4"
                        :: "i"(RISCV_CUSTOM2), "i"(1 << 2), "r"(peer_mbar), "r"(peer + dst), "r"(odd)
                        : "memory");
      break;
    }
    default:
      vx_st_async_u32(peer + dst, 1, peer_mbar);
      break;
    }
  }
  idle_cycles(arg->linger);
  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + (blockIdx.x * blockDim.x + tid) * ERR_WORDS;
  out[ERR_DONE] = 1;
}
