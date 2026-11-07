#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>
#include <vx_dlmem.h>
#include <vx_dxa.h>
#include <vx_mbar.h>
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

INL void idle_cycles(uint64_t n) {
  const uint64_t t0 = vx_rdcycle();
  while (vx_rdcycle() - t0 < n) {}
}

INL void tstore(const job_t& j, const void* src) {
  const int32_t* c = j.coord;
  if (j.im2col) {
    switch (j.rank) {
    case 3:  vx_dxa_store_im2col_3d(j.slot, src, c[0], c[1], c[2]); break;
    case 4:  vx_dxa_store_im2col_4d(j.slot, src, c[0], c[1], c[2], c[3]); break;
    default: vx_dxa_store_im2col_5d(j.slot, src, c[0], c[1], c[2], c[3], c[4]); break;
    }
    return;
  }
  switch (j.rank) {
  case 1:  vx_dxa_store_1d(j.slot, src, c[0]); break;
  case 2:  vx_dxa_store_2d(j.slot, src, c[0], c[1]); break;
  case 3:  vx_dxa_store_3d(j.slot, src, c[0], c[1], c[2]); break;
  case 4:  vx_dxa_store_4d(j.slot, src, c[0], c[1], c[2], c[3]); break;
  default: vx_dxa_store_5d(j.slot, src, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

INL void tred(const job_t& j, const void* src) {
  const int32_t* c = j.coord;
  if (j.im2col) {
    switch (j.rank) {
    case 3:  vx_dxa_reduce_im2col_3d(j.slot, src, j.op, j.type, c[0], c[1], c[2]); break;
    case 4:  vx_dxa_reduce_im2col_4d(j.slot, src, j.op, j.type, c[0], c[1], c[2], c[3]); break;
    default: vx_dxa_reduce_im2col_5d(j.slot, src, j.op, j.type, c[0], c[1], c[2], c[3], c[4]); break;
    }
    return;
  }
  switch (j.rank) {
  case 1:  vx_dxa_reduce_1d(j.slot, src, j.op, j.type, c[0]); break;
  case 2:  vx_dxa_reduce_2d(j.slot, src, j.op, j.type, c[0], c[1]); break;
  case 3:  vx_dxa_reduce_3d(j.slot, src, j.op, j.type, c[0], c[1], c[2]); break;
  case 4:  vx_dxa_reduce_4d(j.slot, src, j.op, j.type, c[0], c[1], c[2], c[3]); break;
  default: vx_dxa_reduce_5d(j.slot, src, j.op, j.type, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

INL void wait_group(uint32_t n) {
  switch (n) {
  case 0:  vx_bulk_wait_group<0>(); break;
  case 1:  vx_bulk_wait_group<1>(); break;
  case 2:  vx_bulk_wait_group<2>(); break;
  case 3:  vx_bulk_wait_group<3>(); break;
  case 8:  vx_bulk_wait_group<8>(); break;
  default: vx_bulk_wait_group<31>(); break;
  }
}

INL void wait_group_read(uint32_t n) {
  switch (n) {
  case 0:  vx_bulk_wait_group_read<0>(); break;
  case 1:  vx_bulk_wait_group_read<1>(); break;
  case 2:  vx_bulk_wait_group_read<2>(); break;
  case 3:  vx_bulk_wait_group_read<3>(); break;
  case 8:  vx_bulk_wait_group_read<8>(); break;
  default: vx_bulk_wait_group_read<31>(); break;
  }
}

INL void raw(const job_t& j, uint8_t* lm) {
  if (j.n == 0) {
    __vx_dxa_store(vx_dxa_st_ctl(5, 0, 0, 0), lm + j.src, (size_t)j.gdst, (size_t)j.size, (size_t)0, 0);
  } else {
    __asm__ volatile (".insn r %0, 7, 1, x0, x3, x0" :: "i"(RISCV_CUSTOM2) : "memory");
  }
}

// One issue job; the others are handled by the caller.
INL void issue(const job_t& j, uint8_t* lm) {
  void* g = reinterpret_cast<void*>(j.gdst);
  switch (j.kind) {
  case J_BSTORE: vx_dxa_bulk_store(g, lm + j.src, j.size); break;
  case J_TSTORE: tstore(j, lm + j.src); break;
  case J_BRED:   vx_dxa_reduce_global(g, lm + j.src, j.size, j.op, j.type); break;
  case J_TRED:   tred(j, lm + j.src); break;
  case J_COMMIT: vx_bulk_commit_group(); break;
  case J_RAW:    raw(j, lm); break;
  default: break;
  }
}

// Words of the bulk stores of groups [lo, hi) that a generic load finds
// different from their source.
INL uint32_t check_groups(const job_t* jobs, uint32_t upto, uint32_t lo, uint32_t hi, const uint8_t* lm,
                          uint32_t lane, uint32_t nt) {
  uint32_t bad = 0;
  for (uint32_t k = 0; k < upto; ++k) {
    const job_t& s = jobs[k];
    if (s.kind != J_BSTORE || s.group < lo || s.group >= hi)
      continue;
    auto g = reinterpret_cast<const volatile uint32_t*>(s.gdst);
    auto l = reinterpret_cast<const volatile uint32_t*>(lm + s.src);
    for (uint32_t w = lane; w < s.size / 4; w += nt)
      bad += (g[w] != l[w]);
  }
  return bad;
}

INL void run_jobs(const store_arg_t* arg, uint8_t* lm, uint32_t* out, uint32_t lane, uint32_t nt) {
  auto jobs = reinterpret_cast<const job_t*>(arg->jobs_addr);
  auto lw = reinterpret_cast<volatile uint32_t*>(lm);
  uint32_t committed = 0, checked = 0, bad = 0;
  for (uint32_t i = 0; i < arg->num_jobs; ++i) {
    const job_t& j = jobs[i];
    switch (j.kind) {
    case J_COMMIT:
      vx_bulk_commit_group();
      ++committed;
      break;
    case J_WAIT:
      wait_group(j.n);
      if (j.check && committed > j.n) {
        vx_fence_proxy_async_global();
        bad += check_groups(jobs, i, checked, committed - j.n, lm, lane, nt);
        checked = committed - j.n;
      }
      break;
    case J_WAITR:
      wait_group_read(j.n);
      break;
    case J_REWRITE:
      for (uint32_t w = j.src / 4 + lane; w < (j.src + j.size) / 4; w += nt)
        lw[w] = lmem_word2(w);
      vx_fence_proxy_async();
      break;
    default:
      issue(j, lm);
      break;
    }
  }
  out[ERR_BAD] = bad;
}

// -p: warp 0 alone issues the launch's repeats, each closed by commit +
// wait_group 0 (jobs of the plan), and times them.
INL void perf_run(const store_arg_t* arg, uint8_t* lm, uint32_t* out) {
  auto jobs = reinterpret_cast<const job_t*>(arg->jobs_addr);
  const uint64_t t0 = vx_rdcycle();
  for (uint32_t i = 0; i < arg->num_jobs; ++i) {
    const job_t& j = jobs[i];
    if (j.kind == J_WAIT)
      wait_group(j.n);
    else
      issue(j, lm);
  }
  const uint64_t t1 = vx_rdcycle();
  if (threadIdx.x == 0)
    out[ERR_CYCLES] = (uint32_t)(t1 - t0);
}

INL void fill_lmem(const store_arg_t* arg, uint8_t* lm, uint32_t tid, uint32_t nthr) {
  auto lw = reinterpret_cast<volatile uint32_t*>(lm);
  for (uint32_t w = 2 + tid; w < arg->lmem_bytes / 4; w += nthr)
    lw[w] = lmem_word(w);
  auto fills = reinterpret_cast<const fill_t*>(arg->fills_addr);
  for (uint32_t f = 0; f < arg->num_fills; ++f) {
    __syncthreads();
    const fill_t& x = fills[f];
    const uint32_t E = type_size(x.type);
    for (uint32_t e = tid; e < x.size / E; e += nthr) {
      const uint64_t v = typed_val(x.type, x.seed, e);
      uint8_t* p = lm + x.src + e * E;
      if (E == 8)
        *reinterpret_cast<volatile uint64_t*>(p) = v;
      else if (E == 4)
        *reinterpret_cast<volatile uint32_t*>(p) = (uint32_t)v;
      else
        *reinterpret_cast<volatile uint16_t*>(p) = (uint16_t)v;
    }
  }
}

// i7a: a bulk load of the region warms the L2 with its old bytes; warp 0
// then stores new words over it, fences with fence.proxy.async.global and
// bulk-loads it again: the second copy must hold the new words.
INL void i7a(const store_arg_t* arg, uint8_t* lm, uint32_t* out, uint32_t lane, uint32_t nt) {
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);
  auto g = reinterpret_cast<volatile uint32_t*>(arg->gaddr);
  const uint32_t size = arg->size;
  uint8_t* a = lm + LM_DATA;
  uint8_t* b = lm + LM_DATA + size;
  uint32_t err = 0, bad = 0;
  if (lane == 0) {
    vx_mbar_init(mbar, 1);
    vx_mbar_arrive_tx(mbar, size);
  }
  vx_fence_proxy_async();
  vx_dxa_bulk_load(a, reinterpret_cast<const void*>(arg->gaddr), size, mbar);
  for (uint32_t t = 0; !vx_mbar_try_wait_parity(mbar, 0); ) {
    if (++t > WAIT_LIMIT) { err = 1; break; }
  }
  if (lane == 0)
    vx_mbar_arrive_tx(mbar, size);
  for (uint32_t w = lane; w < size / 4; w += nt)
    g[w] = new_word(w);
  vx_fence_proxy_async_global();
  vx_dxa_bulk_load(b, reinterpret_cast<const void*>(arg->gaddr), size, mbar);
  for (uint32_t t = 0; !vx_mbar_try_wait_parity(mbar, 1); ) {
    if (++t > WAIT_LIMIT) { err = 1; break; }
  }
  auto bw = reinterpret_cast<const volatile uint32_t*>(b);
  for (uint32_t w = lane; w < size / 4; w += nt)
    bad += (bw[w] != new_word(w));
  out[ERR_WAIT] = err;
  out[ERR_BAD] = bad;
}

// i7b: warp 0 loads the region into its data cache, bulk-stores local memory
// over it, waits for the group and fences with fence.proxy.async.global: a
// generic load must then see the stored bytes.
INL void i7b(const store_arg_t* arg, uint8_t* lm, uint32_t* out, uint32_t lane, uint32_t nt) {
  auto g = reinterpret_cast<volatile uint32_t*>(arg->gaddr);
  const uint32_t size = arg->size;
  const uint8_t* s = lm + LM_DATA;
  uint32_t bad = 0;
  for (uint32_t w = lane; w < size / 4; w += nt)
    (void)g[w];
  vx_dxa_bulk_store(reinterpret_cast<void*>(arg->gaddr), s, size);
  vx_bulk_commit_group();
  vx_bulk_wait_group<0>();
  vx_fence_proxy_async_global();
  auto sw = reinterpret_cast<const volatile uint32_t*>(s);
  for (uint32_t w = lane; w < size / 4; w += nt)
    bad += (g[w] != sw[w]);
  out[ERR_BAD] = bad;
}

// i7c: as i7b, but lanes 0-3 store the region's first 16 bytes generically
// and the bulk store writes the rest of the region: the generic stores leave
// the line dirty in a write-back data cache, and its write-back must not
// rewrite the bytes the bulk store wrote.
INL void i7c(const store_arg_t* arg, uint8_t* lm, uint32_t* out, uint32_t lane, uint32_t nt) {
  auto g = reinterpret_cast<volatile uint32_t*>(arg->gaddr);
  const uint32_t size = arg->size;
  const uint8_t* s = lm + LM_DATA;
  auto sw = reinterpret_cast<const volatile uint32_t*>(s);
  uint32_t bad = 0;
  for (uint32_t w = lane; w < size / 4; w += nt)
    (void)g[w];
  if (lane < 4)
    g[lane] = sw[lane];
  vx_dxa_bulk_store(reinterpret_cast<void*>(arg->gaddr + 16), s + 16, size - 16);
  vx_bulk_commit_group();
  vx_bulk_wait_group<0>();
  vx_fence_proxy_async_global();
  for (uint32_t w = lane; w < size / 4; w += nt)
    bad += (g[w] != sw[w]);
  out[ERR_BAD] = bad;
}

// exit: a bulk store of the region, waited for its reads only; the warp then
// exits with the writes in flight, which the kernel's end must still wait for.
INL void exit_run(const store_arg_t* arg, uint8_t* lm) {
  vx_dxa_bulk_store(reinterpret_cast<void*>(arg->gaddr), lm + LM_DATA, arg->size);
  vx_bulk_commit_group();
  vx_bulk_wait_group_read<0>();
}

} // namespace

// Every block fills its local memory with the pattern and the typed fills,
// fences the stores for the async proxy; warp 0 then runs the jobs (or i7a,
// i7b, i7c, exit, or the -p loop) and ends with commit + wait_group 0 (exit:
// with wait_group.read 0 only). The host checks
// the whole global buffer byte for byte against sw/common/dxa_ref.h.
__kernel void store_k(store_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t nthr = blockDim.x;
  const uint32_t nt   = (uint32_t)vx_num_threads();
  auto lm  = reinterpret_cast<uint8_t*>(__local_mem());
  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + tid * ERR_WORDS;

  if (arg->kernel == K_JOBS && !arg->repeats)
    fill_lmem(arg, lm, tid, nthr);
  else if (arg->kernel == K_I7B || arg->kernel == K_I7C || arg->kernel == K_EXIT) {
    auto lw = reinterpret_cast<volatile uint32_t*>(lm);
    for (uint32_t w = LM_DATA / 4 + tid; w < (LM_DATA + arg->size) / 4; w += nthr)
      lw[w] = lmem_word(w);
  }
  vx_fence_proxy_async();
  __syncthreads();

  if (get_sub_group_id() == 0) {
    const uint32_t lane = tid % nt;
    if (arg->kernel == K_I7A)
      i7a(arg, lm, out, lane, nt);
    else if (arg->kernel == K_I7B)
      i7b(arg, lm, out, lane, nt);
    else if (arg->kernel == K_I7C)
      i7c(arg, lm, out, lane, nt);
    else if (arg->kernel == K_EXIT)
      exit_run(arg, lm);
    else if (arg->repeats)
      perf_run(arg, lm, out);
    else
      run_jobs(arg, lm, out, lane, nt);
    if (arg->neg)
      idle_cycles(arg->linger);
    if (arg->kernel != K_I7A && arg->kernel != K_EXIT)
      vx_bulk_wait_all();
  }
  __syncthreads();
  out[ERR_DONE] = 1;
}
