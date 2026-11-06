#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dlmem.h>
#include <vx_mbar.h>
#include <vx_cp_async.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

// A copy of v the compiler cannot relate to earlier tests of v: a branch on
// it is not threaded back across a barrier, which would duplicate the
// barrier into both sides of an earlier lane-divergent branch.
INL uint32_t opaque(uint32_t v) {
  __asm__ volatile ("" : "+r"(v));
  return v;
}

// One exit, so the loop never ends in a lane-divergent branch.
INL void wait_parity(uint64_t* bar, uint32_t par, const cpa_arg_t* arg, uint32_t& spins) {
  bool done = false;
  for (uint32_t i = 0; !done && i < arg->spin_limit; ++i) {
    done = vx_mbar_try_wait_parity(bar, par);
    spins += !done;
  }
}

INL void copy_case(uint32_t c, uint8_t* slot, const uint8_t* src, const void* bogus) {
  void* dst = slot + ZF_DOFF(c);
  const uint8_t* s = src + ZF_SOFF(c);
  const uint32_t ss = ZF_SS(c);
  // ignore-src true: the source is never read, so any aligned address will do.
  const bool ign = ZF_IGN(c) != 0;
  const void* sp = ign ? bogus : s;
  switch (ZF_SIZE(c)) {
  case 4:  vx_cp_async_ca_4(dst, sp, ss, ign); break;
  case 8:  vx_cp_async_ca_8(dst, sp, ss, ign); break;
  default:
    if (c & ZF_CG)
      vx_cp_async_cg_16(dst, sp, ss, ign);
    else
      vx_cp_async_ca_16(dst, sp, ss, ign);
    break;
  }
}

INL void zfill(const cpa_arg_t* arg, uint8_t* lm) {
  const uint32_t T = blockDim.x, tid = threadIdx.x;
  auto cases = reinterpret_cast<const uint32_t*>(arg->aux_addr);
  auto src = reinterpret_cast<const uint8_t*>(arg->src_addr);
  for (uint32_t c = tid; c < arg->count; c += T) {
    auto w = reinterpret_cast<volatile uint64_t*>(lm + c * ZF_SLOT);
    for (uint32_t i = 0; i < ZF_SLOT / 8; ++i)
      w[i] = 0x0101010101010101ull * ZF_FILL;
  }
  __syncthreads();
  for (uint32_t c = tid; c < arg->count; c += T)
    copy_case(cases[c], lm + c * ZF_SLOT, src, lm);
  vx_cp_async_wait_all();
  __syncthreads();
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
  for (uint32_t i = tid; i < arg->count * ZF_SLOT / 8; i += T)
    out[i] = reinterpret_cast<volatile uint64_t*>(lm)[i];
}

INL bool same16(const volatile uint64_t* a, const uint64_t* b) {
  return a[0] == b[0] && a[1] == b[1];
}

INL void groups(const cpa_arg_t* arg, uint8_t* lm) {
  const uint32_t T = blockDim.x, tid = threadIdx.x;
  auto src = reinterpret_cast<const uint8_t*>(arg->src_addr);
  auto buf = reinterpret_cast<volatile uint64_t*>(lm);
  for (uint32_t g = 0; g < GR_GROUPS; ++g) {
    buf[2 * (g * T + tid)] = 0;
    buf[2 * (g * T + tid) + 1] = 0;
  }
  __syncthreads();
  for (uint32_t g = 0; g < GR_GROUPS; ++g) {
    vx_cp_async_cg_16(lm + 16 * (g * T + tid), src + g * GR_STRIDE + 16 * tid);
    vx_cp_async_commit_group();
  }
  vx_cp_async_wait_group<2>();
  uint32_t seen = 0;
  for (uint32_t g = 0; g < GR_GROUPS; ++g) {
    seen |= same16(buf + 2 * (g * T + tid), reinterpret_cast<const uint64_t*>(src + g * GR_STRIDE + 16 * tid)) << g;
  }
  // N >= 8 never waits.
  vx_cp_async_wait_group<9>();
  vx_cp_async_wait_all();
  for (uint32_t g = 0; g < GR_GROUPS; ++g) {
    seen |= same16(buf + 2 * (g * T + tid), reinterpret_cast<const uint64_t*>(src + g * GR_STRIDE + 16 * tid))
            << (4 + g);
  }
  reinterpret_cast<uint32_t*>(arg->out_addr)[tid] = seen;
}

// The H100 probe's k_cpabar, all four modes in one launch, each on its own
// object and buffer. Every thread waits; the single-thread work of modes 2
// and 3 is straight-line code in tid 0, so no loop or barrier sits in a
// lane-divergent branch. Every thread checks its own element into the bad
// count (zeroed by the host).
INL void bar(const cpa_arg_t* arg, uint8_t* lm, uint32_t& spins) {
  const uint32_t T = blockDim.x, tid = threadIdx.x;
  auto src = reinterpret_cast<const uint64_t*>(arg->src_addr);
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
  uint64_t* bars = reinterpret_cast<uint64_t*>(lm);
  for (uint32_t m = 0; m < BAR_MODES; ++m) {
    uint64_t* b = bars + m;
    auto sm = reinterpret_cast<volatile uint64_t*>(lm + 64 + m * 16 * T);
    if (opaque(tid) == 0)
      vx_mbar_init(b, m == 2 ? 2 : (m == 3 ? 1 : T));
    sm[2 * tid] = 0;
    sm[2 * tid + 1] = 0;
    __syncthreads();
    uint64_t* rec = out + BAR_OUT(m) + 2;
    const bool all = (m <= 1);
    const uint64_t* s = all ? src + 2 * tid : src + 2 * BAR_SRC_SINGLE;
    if (all) {
      vx_cp_async_cg_16((void*)(sm + 2 * tid), s);
      if (m == 0) {
        vx_cp_async_mbar_arrive_noinc(b);
      } else {
        vx_cp_async_mbar_arrive(b);
        out[BAR_PC + tid] = vx_mbar_pending_count(vx_mbar_arrive(b, 1));
      }
    }
    if (!all && opaque(tid) == 0) {
      vx_cp_async_cg_16((void*)sm, s);
      if (m == 2)
        vx_cp_async_mbar_arrive_noinc(b);
      else
        vx_cp_async_mbar_arrive(b);
      const uint64_t w = vx_mbar_arrive(b, 1);
      rec[0] = vx_mbar_pending_count(w);
      rec[1] = vx_mbar_test_wait_parity(b, 0);
    }
    wait_parity(b, 0, arg, spins);
    // Every thread is past its wait before tid 0 arrives in the next phase:
    // in mode 3 that arrive completes phase 1, and a late waiter on parity 0
    // would then wait for phase 2.
    __syncthreads();
    if (!all && opaque(tid) == 0) {
      rec[2] = vx_mbar_test_wait_parity(b, 0);
      const uint64_t w2 = vx_mbar_arrive(b, 1);
      rec[3] = vx_mbar_pending_count(w2);
      rec[4] = vx_mbar_test_wait_parity(b, 1);
    }
    __syncthreads();
    const uint64_t* e = all ? src + 2 * tid : src + 2 * BAR_SRC_SINGLE;
    if (all || opaque(tid) == 0) {
      if (!same16(sm + 2 * tid, e))
        __atomic_fetch_add(out + BAR_OUT(m), (uint64_t)1, __ATOMIC_RELAXED);
    }
    if (opaque(tid) == 0)
      out[BAR_OUT(m) + 1] = reinterpret_cast<volatile uint64_t*>(b)[0];
    __syncthreads();
  }
}

// Row r of the block: 16 bytes per thread, into buffer r % 2, completing on
// FULL[r % 2]: the noinc arrive, or (rows with bit 1 set) the tx form and a
// plain arrive. The consumer sums another warp's chunk.
INL void issue_row(const cpa_arg_t* arg, uint8_t* lm, uint64_t* full, uint32_t r) {
  const uint32_t T = blockDim.x, tid = threadIdx.x;
  auto src = reinterpret_cast<const uint8_t*>(arg->src_addr) + (size_t)blockIdx.x * arg->count * 16 * T;
  uint8_t* buf = lm + 64 + (r & 1) * 16 * T;
  uint64_t* b = full + (r & 1);
  vx_cp_async_16(buf + 16 * tid, src + (size_t)r * 16 * T + 16 * tid);
  if ((r >> 1) & 1) {
    vx_cp_async_mbar_arrive(b);
    vx_mbar_arrive(b, 1);
  } else {
    vx_cp_async_mbar_arrive_noinc(b);
  }
}

INL void stream(const cpa_arg_t* arg, uint8_t* lm, uint32_t& spins) {
  const uint32_t T = blockDim.x, tid = threadIdx.x, nt = vx_num_threads();
  uint64_t* full = reinterpret_cast<uint64_t*>(lm);
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr) + (size_t)blockIdx.x * arg->count * T;
  auto info = reinterpret_cast<volatile uint64_t*>(arg->info_addr) + blockIdx.x * INFO_WORDS;
  const uint32_t R = arg->count;
  uint32_t open = 0;
  if (opaque(tid) == 0) {
    vx_mbar_init(full, T);
    vx_mbar_init(full + 1, T);
  }
  __syncthreads();
  if (!arg->parity)
    issue_row(arg, lm, full, 0);
  for (uint32_t r = 0; r < R; ++r) {
    uint64_t* b = full + (r & 1);
    const uint32_t ph = (r >> 1) & 1;
    if (arg->parity) {
      issue_row(arg, lm, full, r);
      // Every copy of the row has landed once each warp's wait_all and the
      // barrier pass; the try_wait op (custom-2 funct3 5) then sleeps until
      // the arrives that follow the data complete the phase, and a test_wait
      // reads it. Both run unconditionally, so the instruction count does not
      // depend on when the copies or the arrives land.
      vx_cp_async_wait_all();
      __syncthreads();
      uint64_t slept;
      __asm__ volatile (".insn r %1, 5, 0, %0, %2, %3"
                        : "=r"(slept) : "i"(RISCV_CUSTOM2), "r"(b), "r"((uint64_t)ph << 63) : "memory");
      (void)slept;
      open += !vx_mbar_test_wait_parity(b, ph);
    } else {
      if (r + 1 < R)
        issue_row(arg, lm, full, r + 1);
      wait_parity(b, ph, arg, spins);
    }
    auto buf = reinterpret_cast<volatile uint64_t*>(lm + 64 + (r & 1) * 16 * T);
    const uint32_t j = (tid + nt) % T;
    out[r * T + tid] = buf[2 * j] + buf[2 * j + 1];
    __syncthreads();
  }
  if (opaque(tid) == 0) {
    info[INFO_WORD0] = reinterpret_cast<volatile uint64_t*>(full)[0];
    info[INFO_WORD0 + 1] = reinterpret_cast<volatile uint64_t*>(full)[1];
  }
  __atomic_fetch_add(const_cast<uint64_t*>(info + INFO_OPEN), (uint64_t)open, __ATOMIC_RELAXED);
}

// RMSNorm's access shape (Lamport's one-shot kernel): rows of int32 staged
// with __pipeline_memcpy_async into a double buffer, the copy of row r + 1
// in flight while row r is reduced; the row's tail past row_elems zero-filled.
// out[r][i] = x[r][i] * w[i] + sum of x[r][k]^2.
INL void pipe(const cpa_arg_t* arg, uint8_t* lm) {
  const uint32_t T = blockDim.x, tid = threadIdx.x;
  const uint32_t D = 4 * T;
  const uint32_t R = arg->count;
  auto x = reinterpret_cast<const uint8_t*>(arg->src_addr);
  auto w = reinterpret_cast<const uint8_t*>(arg->aux_addr);
  auto out = reinterpret_cast<uint32_t*>(arg->out_addr);
  uint8_t* wbuf = lm;
  uint8_t* xbuf = lm + 4 * D;
  const uint32_t valid = arg->row_elems * 4;
  auto load_row = [&](uint32_t r) {
    const uint32_t off = 16 * tid;
    const uint32_t zf = (off + 16 <= valid) ? 0 : (off >= valid ? 16 : off + 16 - valid);
    vortex::pipeline_memcpy_async(xbuf + (r & 1) * 4 * D + off, x + (size_t)r * 4 * D + off, 16, zf);
  };
  vortex::pipeline_memcpy_async(wbuf + 16 * tid, w + 16 * tid, 16);
  load_row(0);
  vortex::pipeline_commit();
  for (uint32_t r = 0; r < R; ++r) {
    if (r + 1 < R)
      load_row(r + 1);
    vortex::pipeline_commit();
    vortex::pipeline_wait_prior<1>();
    __syncthreads();
    auto xs = reinterpret_cast<volatile uint32_t*>(xbuf + (r & 1) * 4 * D);
    auto ws = reinterpret_cast<volatile uint32_t*>(wbuf);
    uint32_t ss = 0;
    for (uint32_t k = 0; k < D; ++k)
      ss += xs[k] * xs[k];
    for (uint32_t i = 4 * tid; i < 4 * tid + 4; ++i)
      out[r * D + i] = xs[i] * ws[i] + ss;
    __syncthreads();
  }
}

// A block marks its 16 bytes per thread and its mbarrier, waits a while and
// checks them: the previous block on these warps left copies and an arrive in
// flight over the same bytes, which must have landed before it exited. Then
// it does the same to the next block, and exits without waiting.
INL void orphan(const cpa_arg_t* arg, uint8_t* lm) {
  const uint32_t T = blockDim.x, tid = threadIdx.x;
  auto buf = reinterpret_cast<volatile uint64_t*>(lm);
  uint64_t* b = reinterpret_cast<uint64_t*>(lm + 16 * T);
  const uint64_t mark = ((uint64_t)(blockIdx.x + 1) << 32) | tid;
  buf[2 * tid] = mark;
  buf[2 * tid + 1] = ~mark;
  if (opaque(tid) == 0)
    vx_mbar_init(b, T);
  __syncthreads();
  for (volatile uint32_t i = 0; i < arg->count; i = i + 1) {
  }
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr) + (size_t)blockIdx.x * OR_OUT;
  const uint64_t bad = (buf[2 * tid] != mark) + (buf[2 * tid + 1] != ~mark);
  __atomic_fetch_add(out, bad, __ATOMIC_RELAXED);
  if (opaque(tid) == 0)
    out[1] = reinterpret_cast<volatile uint64_t*>(b)[0];
  __syncthreads();
  auto junk = reinterpret_cast<volatile uint64_t*>(arg->aux_addr) + ((size_t)blockIdx.x * T + tid) * OR_STORES * 8;
  for (uint32_t k = 0; k < OR_STORES; ++k)
    junk[k * 8] = k;
  auto src = reinterpret_cast<const uint8_t*>(arg->src_addr) + (size_t)blockIdx.x * OR_COPIES * T * OR_STRIDE;
  for (uint32_t k = 0; k < OR_COPIES; ++k)
    vx_cp_async_16(lm + 16 * tid, src + (size_t)(k * T + tid) * OR_STRIDE);
  vx_cp_async_mbar_arrive_noinc(b);
  if (opaque(tid) == 0)
    reinterpret_cast<volatile uint64_t*>(arg->info_addr)[blockIdx.x * INFO_WORDS + INFO_DONE] = 1;
}

// Per row, a copy into buffer r % 2 and MIX_LOADS global loads per thread,
// all in flight together; wait_group 1 keeps one row's copies outstanding.
// out[r][t]: the loads' sum plus the copied row of another warp's thread.
INL void mix(const cpa_arg_t* arg, uint8_t* lm) {
  const uint32_t T = blockDim.x, tid = threadIdx.x, nt = vx_num_threads();
  const uint32_t R = arg->count;
  auto src = reinterpret_cast<const uint8_t*>(arg->src_addr);
  auto ld = reinterpret_cast<const volatile uint64_t*>(arg->aux_addr);
  auto out = reinterpret_cast<uint64_t*>(arg->out_addr);
  vx_cp_async_16(lm + 16 * tid, src + 16 * tid);
  vx_cp_async_commit_group();
  for (uint32_t r = 0; r < R; ++r) {
    if (r + 1 < R)
      vx_cp_async_16(lm + ((r + 1) & 1) * 16 * T + 16 * tid, src + (size_t)(r + 1) * 16 * T + 16 * tid);
    vx_cp_async_commit_group();
    uint64_t sum = 0;
    for (uint32_t k = 0; k < MIX_LOADS; ++k)
      sum += ld[((size_t)r * MIX_LOADS + k) * T + tid];
    vx_cp_async_wait_group<1>();
    __syncthreads();
    auto buf = reinterpret_cast<volatile uint64_t*>(lm + (r & 1) * 16 * T);
    const uint32_t j = (tid + nt) % T;
    out[r * T + tid] = sum + buf[2 * j] + buf[2 * j + 1];
    __syncthreads();
  }
}

INL void negative(const cpa_arg_t* arg, uint8_t* lm) {
  vortex::cluster_group cl;
  const bool cluster = (arg->neg == NEG_REMOTE_DST || arg->neg == NEG_REMOTE_MBAR);
  uint64_t* b = reinterpret_cast<uint64_t*>(lm);
  auto src = reinterpret_cast<const uint8_t*>(arg->src_addr);
  if (opaque(threadIdx.x) == 0)
    vx_mbar_init(b, 1);
  if (cluster)
    cl.sync();
  else
    __syncthreads();
  if (opaque(threadIdx.x) == 0 && (!cluster || cl.block_rank() == 0)) {
    switch (arg->neg) {
    case NEG_REMOTE_DST:  vx_cp_async_16(cl.map_shared_rank(lm + 64, 1), src); break;
    case NEG_GLOBAL_DST:  vx_cp_async_16(reinterpret_cast<void*>(arg->out_addr), src); break;
    case NEG_MISALIGNED:  vx_cp_async_8(lm + 68, src); break;
    case NEG_SRCSIZE:     vx_cp_async_16(lm + 64, src, 17); break;
    case NEG_REMOTE_MBAR: vx_cp_async_mbar_arrive_noinc(cl.map_shared_rank(b, 1)); break;
    case NEG_LOCAL_SRC:   vx_cp_async_16(lm + 64, lm + 128); break;
    case NEG_ENCODING:
      __asm__ volatile (".insn r %0, 7, 3, x0, x0, x0" :: "i"(RISCV_CUSTOM2) : "memory");
      break;
    default: break;
    }
    vx_cp_async_wait_all();
    reinterpret_cast<volatile uint64_t*>(arg->info_addr)[INFO_DONE] = 1;
  }
  if (cluster)
    cl.sync();
}

}  // namespace

__kernel void kernel_main(cpa_arg_t* __UNIFORM__ arg) {
  uint8_t* lm = reinterpret_cast<uint8_t*>(__local_mem());
  if (arg->neg) {
    negative(arg, lm);
    return;
  }
  // No barrier after it: its warps exit with their copies in flight.
  if (arg->mode == MODE_ORPHAN) {
    orphan(arg, lm);
    return;
  }
  uint32_t spins = 0;
  switch (arg->mode) {
  case MODE_ZFILL:  zfill(arg, lm); break;
  case MODE_GROUPS: groups(arg, lm); break;
  case MODE_BAR:    bar(arg, lm, spins); break;
  case MODE_STREAM: stream(arg, lm, spins); break;
  case MODE_MIX:    mix(arg, lm); break;
  default:          pipe(arg, lm); break;
  }
  auto info = reinterpret_cast<uint64_t*>(arg->info_addr) + blockIdx.x * INFO_WORDS;
  // Unconditional, so no lane-divergent branch precedes the barrier.
  __atomic_fetch_add(info + INFO_SPINS, (uint64_t)spins, __ATOMIC_RELAXED);
  __syncthreads();
  if (opaque(threadIdx.x) == 0)
    reinterpret_cast<volatile uint64_t*>(info)[INFO_DONE] = 1;
}
