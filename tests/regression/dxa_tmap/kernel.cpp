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

INL vx_tensor_map_t* map_of(tmap_arg_t* arg, uint8_t* lm, const job_t& j) {
  switch (j.sel) {
  case SEL_PARAM:  return &arg->map;
  case SEL_GLOBAL: return reinterpret_cast<vx_tensor_map_t*>(arg->maps_addr + j.map_off);
  default:         return reinterpret_cast<vx_tensor_map_t*>(lm + j.map_off);
  }
}

INL void tload_map(const vx_tensor_map_t* m, const job_t& j, void* dst, uint64_t* mb) {
  const int32_t* c = j.coord;
  const uint16_t k = (uint16_t)j.mask;
  if (j.im2col) {
    const uint16_t o1 = (uint16_t)j.offs[0], o2 = (uint16_t)j.offs[1], o3 = (uint16_t)j.offs[2];
    switch (j.rank) {
    case 3:  vx_dxa_load_im2col_3d(m, dst, mb, c[0], c[1], c[2], o1); break;
    case 4:  vx_dxa_load_im2col_4d_hint(m, dst, mb, c[0], c[1], c[2], c[3], o1, o2, 0x77ull); break;
    default: vx_dxa_load_im2col_5d(m, dst, mb, c[0], c[1], c[2], c[3], c[4], o1, o2, o3); break;
    }
    return;
  }
  switch (j.rank) {
  case 1:  vx_dxa_load_1d(m, dst, mb, c[0]); break;
  case 2:
    if (k) vx_dxa_load_2d_multicast(m, dst, mb, c[0], c[1], k);
    else   vx_dxa_load_2d_hint(m, dst, mb, c[0], c[1], 0x1234ull);
    break;
  case 3:  vx_dxa_load_3d(m, dst, mb, c[0], c[1], c[2]); break;
  case 4:  vx_dxa_load_4d(m, dst, mb, c[0], c[1], c[2], c[3]); break;
  default: vx_dxa_load_5d(m, dst, mb, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

INL void tload_slot(const job_t& j, void* dst, uint64_t* mb) {
  const int32_t* c = j.coord;
  switch (j.rank) {
  case 1:  vx_dxa_load_1d(j.slot, dst, mb, c[0]); break;
  case 2:  vx_dxa_load_2d(j.slot, dst, mb, c[0], c[1]); break;
  case 3:  vx_dxa_load_3d(j.slot, dst, mb, c[0], c[1], c[2]); break;
  case 4:  vx_dxa_load_4d(j.slot, dst, mb, c[0], c[1], c[2], c[3]); break;
  default: vx_dxa_load_5d(j.slot, dst, mb, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

INL void tstore(tmap_arg_t* arg, uint8_t* lm, const job_t& j) {
  const int32_t* c = j.coord;
  const void* src = lm + j.dst;
  if (j.sel == SEL_SLOT) {
    switch (j.rank) {
    case 1:  vx_dxa_store_1d(j.slot, src, c[0]); break;
    case 2:  vx_dxa_store_2d(j.slot, src, c[0], c[1]); break;
    default: vx_dxa_store_3d(j.slot, src, c[0], c[1], c[2]); break;
    }
    return;
  }
  const vx_tensor_map_t* m = map_of(arg, lm, j);
  if (j.im2col) {
    vx_dxa_store_im2col_4d(m, src, c[0], c[1], c[2], c[3]);
    return;
  }
  switch (j.rank) {
  case 1:  vx_dxa_store_1d(m, src, c[0]); break;
  case 2:  vx_dxa_store_2d(m, src, c[0], c[1]); break;
  case 3:  vx_dxa_store_3d(m, src, c[0], c[1], c[2]); break;
  case 4:  vx_dxa_store_4d(m, src, c[0], c[1], c[2], c[3]); break;
  default: vx_dxa_store_5d(m, src, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

INL void tred(tmap_arg_t* arg, uint8_t* lm, const job_t& j) {
  const int32_t* c = j.coord;
  const void* src = lm + j.dst;
  const vx_tensor_map_t* m = map_of(arg, lm, j);
  switch (j.rank) {
  case 1:  vx_dxa_reduce_1d(m, src, j.op, c[0]); break;
  case 2:  vx_dxa_reduce_2d(m, src, j.op, c[0], c[1]); break;
  case 3:  vx_dxa_reduce_3d(m, src, j.op, c[0], c[1], c[2]); break;
  case 4:  vx_dxa_reduce_4d(m, src, j.op, c[0], c[1], c[2], c[3]); break;
  default: vx_dxa_reduce_5d(m, src, j.op, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

// Every field from a zero buffer, in PTX's order of the cuda::ptx helpers.
INL void replace_all(vx_tensor_map_t* m, const rspec_t& s) {
  vx_tensormap_replace_global_address(m, s.gaddr);
  vx_tensormap_replace_rank(m, s.rank - 1);
  for (uint32_t i = 0; i < s.rank; ++i) {
    vx_tensormap_replace_box_dim(m, i, s.box[i]);
    vx_tensormap_replace_global_dim(m, i, s.gdim[i]);
  }
  for (uint32_t i = 0; i + 1 < s.rank; ++i)
    vx_tensormap_replace_global_stride(m, i, s.gstride[i]);
  for (uint32_t i = 0; i < s.rank; ++i)
    vx_tensormap_replace_element_stride(m, i, s.es[i]);
  vx_tensormap_replace_elemtype(m, s.elemtype);
  vx_tensormap_replace_interleave_layout(m, s.interleave);
  vx_tensormap_replace_swizzle_mode(m, s.swizzle);
  vx_tensormap_replace_fill_mode(m, s.fill);
}

// True when the phase did not complete within WAIT_LIMIT try_waits.
INL bool wait_phase(uint64_t* mbar, uint32_t parity) {
  for (uint32_t t = 0; !vx_mbar_try_wait_parity(mbar, parity); ) {
    if (++t > WAIT_LIMIT)
      return true;
  }
  return false;
}

INL void run_jobs(tmap_arg_t* arg, uint8_t* lm, uint64_t* mbar, uint32_t* out, uint32_t* phase) {
  const uint32_t w    = get_sub_group_id();
  const uint32_t lane = threadIdx.x % (uint32_t)vx_num_threads();
  auto jobs = reinterpret_cast<const job_t*>(arg->jobs_addr);
  uint32_t err = 0;
  for (uint32_t i = 0; i < arg->num_jobs; ++i) {
    const job_t& j = jobs[i];
    if (j.kind == J_SYNC) {
      __syncthreads();
      continue;
    }
    const uint32_t ph = *phase;
    if (j.kind == J_LOAD)
      ++*phase;
    if (j.warp != w)
      continue;
    switch (j.kind) {
    case J_LOAD:
      if (lane == 0)
        vx_mbar_arrive_tx(mbar, j.tx);
      vx_fence_proxy_async();
      if (j.sel == SEL_SLOT)
        tload_slot(j, lm + j.dst, mbar);
      else
        tload_map(map_of(arg, lm, j), j, lm + j.dst, mbar);
      err |= wait_phase(mbar, ph & 1);
      break;
    case J_STORE: tstore(arg, lm, j); break;
    case J_RED:   tred(arg, lm, j); break;
    case J_BWAIT: vx_bulk_wait_all(); break;
    case J_PF:    vx_prefetch_tensormap(map_of(arg, lm, j)); break;
    case J_ACQ:   vx_fence_proxy_tensormap_acquire(map_of(arg, lm, j)); break;
    case J_REL:   vx_fence_proxy_tensormap_release(); break;
    case J_COPY: {
      auto s = reinterpret_cast<const volatile uint32_t*>(arg->maps_addr + j.map_off);
      auto d = reinterpret_cast<volatile uint32_t*>(lm + j.dst);
      for (uint32_t k = lane; k < TMAP_WORDS; k += (uint32_t)vx_num_threads())
        d[k] = s[k];
      break;
    }
    case J_ZERO: {
      auto d = reinterpret_cast<volatile uint32_t*>(lm + j.dst);
      for (uint32_t k = lane; k < TMAP_WORDS; k += (uint32_t)vx_num_threads())
        d[k] = 0;
      break;
    }
    case J_RADDR:
      if (lane == 0)
        vx_tensormap_replace_global_address(map_of(arg, lm, j), j.val);
      break;
    case J_RSTRIDE:
      if (lane == 0)
        vx_tensormap_replace_global_stride(map_of(arg, lm, j), j.arg, j.val);
      break;
    case J_RALL:
      if (lane == 0)
        replace_all(map_of(arg, lm, j), arg->spec);
      break;
    case J_CP:
      if (!j.mask || ((j.mask >> lane) & 1))
        vx_tensormap_cp_fenceproxy(reinterpret_cast<vx_tensor_map_t*>(arg->maps_addr + j.map_off),
                                   reinterpret_cast<const vx_tensor_map_t*>(lm + j.dst));
      break;
    default:
      break;
    }
  }
  out[ERR_WAIT] |= err;
}

// Warp 0 issues n loads of job 0's box, one mbarrier phase each, through
// maps first, first + 1, ... (step 1) or map first n times (step 0) of the
// maps buffer, and returns their cycles.
INL uint32_t timed_loads(tmap_arg_t* arg, uint8_t* lm, uint64_t* mbar, uint32_t first, uint32_t step, uint32_t n,
                         uint32_t* phase, uint32_t* err) {
  const uint32_t lane = threadIdx.x % (uint32_t)vx_num_threads();
  const job_t& j = reinterpret_cast<const job_t*>(arg->jobs_addr)[0];
  const uint64_t t0 = vx_rdcycle();
  for (uint32_t r = 0; r < n; ++r) {
    auto m = reinterpret_cast<const vx_tensor_map_t*>(arg->maps_addr + (first + step * r) * 128u);
    if (lane == 0)
      vx_mbar_arrive_tx(mbar, j.tx);
    vx_fence_proxy_async();
    tload_map(m, j, lm + j.dst, mbar);
    *err |= wait_phase(mbar, (*phase)++ & 1);
  }
  return (uint32_t)(vx_rdcycle() - t0);
}

// A try_wait and a test_wait, retried only past the time limit: a try_wait
// that slept answers false and the test_wait then reads the completed phase,
// so the instructions do not depend on when the phase completes.
INL bool wait_fixed(uint64_t* mbar, uint32_t parity, uint32_t* tmo) {
  const uint64_t tok = (uint64_t)(parity & 1) << 63;
  for (uint32_t t = 0; ; ++t) {
    const bool raw    = __VX_MBAR_OP(5, 0, mbar, tok) != 0;
    const bool tested = vx_mbar_test_wait(mbar, tok);
    if (raw | tested)
      return false;
    ++*tmo;
    if (t >= WAIT_LIMIT)
      return true;
  }
}

// Warp 0 issues N prefetch.tensormap back to back, through maps first,
// first + 1, ... (MISS) or maps r % DXA_TMAP_ENTRIES of the maps buffer, then
// one load of job 0's box through the last of them, and waits for the load
// once. The DXA takes its requests in order, so the load completes after
// every prefetch: a missing prefetch holds the queue head for the map's
// fetch, a cached one pops at once. The issue is unrolled: a loop costs a
// single warp more cycles per prefetch than a fetch takes.
template <uint32_t N, bool MISS>
INL bool batch(tmap_arg_t* arg, uint8_t* lm, uint64_t* mbar, uint32_t first, uint32_t parity, uint32_t* tmo) {
  const uint32_t lane = threadIdx.x % (uint32_t)vx_num_threads();
  const job_t& j = reinterpret_cast<const job_t*>(arg->jobs_addr)[0];
  const uint8_t* maps = reinterpret_cast<const uint8_t*>(arg->maps_addr) + (MISS ? first * 128u : 0);
  if (lane == 0)
    vx_mbar_arrive_tx(mbar, j.tx);
  vx_fence_proxy_async();
#pragma unroll
  for (uint32_t r = 0; r < N; ++r)
    vx_prefetch_tensormap(reinterpret_cast<const vx_tensor_map_t*>(maps + (MISS ? r : r % DXA_TMAP_ENTRIES) * 128u));
  tload_map(reinterpret_cast<const vx_tensor_map_t*>(maps + (MISS ? N - 1 : (N - 1) % DXA_TMAP_ENTRIES) * 128u),
            j, lm + j.dst, mbar);
  return wait_fixed(mbar, parity, tmo);
}

// -p: a first batch through maps 0-7 fills the descriptor cache; the timed
// batch of PERF_PREFETCHES goes through maps 8, 9, ... (each a fetch) or
// through maps 0-7 again (each cached). Each thread reports a word of the box.
// Not a batch of loads: the worker's time per load hides the next request's
// fetch, so a run of missing loads is as fast as one of cached loads.
INL void perf_run(tmap_arg_t* arg, uint8_t* lm, uint64_t* mbar, uint32_t* out) {
  const uint32_t lane = threadIdx.x % (uint32_t)vx_num_threads();
  const job_t& j = reinterpret_cast<const job_t*>(arg->jobs_addr)[0];
  uint32_t tmo = 0;
  bool err = batch<DXA_TMAP_ENTRIES, true>(arg, lm, mbar, 0, 0, &tmo);
  const uint64_t t0 = vx_rdcycle();
  if (arg->perf_miss)
    err |= batch<PERF_PREFETCHES, true>(arg, lm, mbar, DXA_TMAP_ENTRIES, 1, &tmo);
  else
    err |= batch<PERF_PREFETCHES, false>(arg, lm, mbar, 0, 1, &tmo);
  const uint32_t c = (uint32_t)(vx_rdcycle() - t0);
  if (lane == 0)
    out[ERR_T0] = c;
  out[ERR_BOX] = reinterpret_cast<volatile uint32_t*>(lm + j.dst)[lane & 3];
  out[ERR_TMO] = tmo;
  out[ERR_WAIT] |= err;
}

// -m prefetch: 8 loads through 8 maps not yet cached; prefetch.tensormap of 8
// other maps, an idle wait, the 8 loads through them; the same 8 loads again
// (cached). The prefetched loads must be as fast as the cached ones.
INL void pf_run(tmap_arg_t* arg, uint8_t* lm, uint64_t* mbar, uint32_t* out) {
  const uint32_t lane = threadIdx.x % (uint32_t)vx_num_threads();
  const uint32_t n = DXA_TMAP_ENTRIES;
  uint32_t phase = 0, err = 0;
  const uint32_t cold = timed_loads(arg, lm, mbar, 0, 1, n, &phase, &err);
  for (uint32_t r = 0; r < n; ++r)
    vx_prefetch_tensormap(reinterpret_cast<const vx_tensor_map_t*>(arg->maps_addr + (n + r) * 128u));
  idle_cycles(arg->linger);
  const uint32_t pref = timed_loads(arg, lm, mbar, n, 1, n, &phase, &err);
  const uint32_t hit = timed_loads(arg, lm, mbar, n, 1, n, &phase, &err);
  if (lane == 0) {
    out[ERR_T0] = cold;
    out[ERR_T0 + 1] = pref;
    out[ERR_T0 + 2] = hit;
  }
  out[ERR_WAIT] |= err;
}

} // namespace

// Every block fills its local memory with canaries and initializes its
// mbarrier; the warps run the jobs (K_MCAST: first the cluster multicast of
// the param map's box from rank 0 into every block), and the block dumps its
// local memory for the host's byte-exact check against dxa_ref. The tensor
// stores and reductions are checked in the global buffer. K_PERF has no
// canaries and no dump, so the run's cycles are the loads'.
__kernel void tmap_k(tmap_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t nthr = blockDim.x;
  const uint32_t b    = blockIdx.x;
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto lw   = reinterpret_cast<volatile uint32_t*>(lm);
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);
  const uint32_t words = arg->lmem_bytes / 4;
  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + (b * nthr + tid) * ERR_WORDS;

  const bool perf = arg->kernel == K_PERF;
  if (!perf) {
    for (uint32_t w = tid; w < words; w += nthr) {
      if (w >= 2)
        lw[w] = canary_word(b, w);
    }
  }
  if (tid == 0) {
    vx_mbar_init(mbar, 1);
    out[ERR_ARG_LO] = (uint32_t)(uintptr_t)&arg->map;
    out[ERR_ARG_HI] = (uint32_t)((uint64_t)(uintptr_t)&arg->map >> 32);
  }
  vx_fence_proxy_async();
  __syncthreads();

  uint32_t phase = 0;
  if (arg->kernel == K_MCAST) {
    const vortex::cluster_group cl = vortex::this_cluster();
    if (tid == 0)
      vx_mbar_arrive_tx(mbar, arg->mc_tx);
    vx_fence_proxy_async();
    cl.sync();
    if (cl.block_rank() == 0 && get_sub_group_id() == 0)
      vx_dxa_load_2d_multicast(&arg->map, lm + arg->mc_dst, mbar, arg->mc_coord[0], arg->mc_coord[1], 0x3);
    out[ERR_WAIT] |= wait_phase(mbar, 0);
    cl.sync();
    phase = 1;
  }

  if (perf || arg->kernel == K_PF) {
    if (get_sub_group_id() == 0) {
      if (perf)
        perf_run(arg, lm, mbar, out);
      else
        pf_run(arg, lm, mbar, out);
    }
  } else {
    run_jobs(arg, lm, mbar, out, &phase);
  }
  if (arg->neg)
    idle_cycles(arg->linger);

  __syncthreads();
  if (!perf) {
    auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr + (uint64_t)b * arg->lmem_bytes);
    for (uint32_t w = tid; w < words; w += nthr)
      dump[w] = lw[w];
  }
  out[ERR_DONE] = 1;
}
