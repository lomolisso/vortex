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

INL void tload(const job_t& j, void* dst, uint64_t* mb) {
  const int32_t* c = j.coord;
  const uint16_t m = (uint16_t)j.mask;
  if (j.im2col) {
    const uint16_t o1 = (uint16_t)j.offs[0], o2 = (uint16_t)j.offs[1], o3 = (uint16_t)j.offs[2];
    switch (j.rank) {
    case 3:
      if (m) vx_dxa_load_im2col_3d_multicast(j.slot, dst, mb, c[0], c[1], c[2], o1, m);
      else   vx_dxa_load_im2col_3d(j.slot, dst, mb, c[0], c[1], c[2], o1);
      break;
    case 4:
      if (m) vx_dxa_load_im2col_4d_multicast(j.slot, dst, mb, c[0], c[1], c[2], c[3], o1, o2, m);
      else   vx_dxa_load_im2col_4d_hint(j.slot, dst, mb, c[0], c[1], c[2], c[3], o1, o2, 0x5A5Aull);
      break;
    default:
      if (m) vx_dxa_load_im2col_5d_multicast(j.slot, dst, mb, c[0], c[1], c[2], c[3], c[4], o1, o2, o3, m);
      else   vx_dxa_load_im2col_5d(j.slot, dst, mb, c[0], c[1], c[2], c[3], c[4], o1, o2, o3);
      break;
    }
    return;
  }
  switch (j.rank) {
  case 1:
    if (m) vx_dxa_load_1d_multicast(j.slot, dst, mb, c[0], m);
    else   vx_dxa_load_1d(j.slot, dst, mb, c[0]);
    break;
  case 2:
    if (m) vx_dxa_load_2d_multicast_hint(j.slot, dst, mb, c[0], c[1], m, 0x1234ull);
    else   vx_dxa_load_2d_hint(j.slot, dst, mb, c[0], c[1], 0x1234ull);
    break;
  case 3:
    if (m) vx_dxa_load_3d_multicast(j.slot, dst, mb, c[0], c[1], c[2], m);
    else   vx_dxa_load_3d(j.slot, dst, mb, c[0], c[1], c[2]);
    break;
  case 4:
    if (m) vx_dxa_load_4d_multicast(j.slot, dst, mb, c[0], c[1], c[2], c[3], m);
    else   vx_dxa_load_4d(j.slot, dst, mb, c[0], c[1], c[2], c[3]);
    break;
  default:
    if (m) vx_dxa_load_5d_multicast(j.slot, dst, mb, c[0], c[1], c[2], c[3], c[4], m);
    else   vx_dxa_load_5d(j.slot, dst, mb, c[0], c[1], c[2], c[3], c[4]);
    break;
  }
}

INL void tprefetch(const job_t& j) {
  const int32_t* c = j.coord;
  if (j.im2col) {
    const uint16_t o1 = (uint16_t)j.offs[0], o2 = (uint16_t)j.offs[1], o3 = (uint16_t)j.offs[2];
    switch (j.rank) {
    case 3:  vx_dxa_prefetch_im2col_3d(j.slot, c[0], c[1], c[2], o1); break;
    case 4:  vx_dxa_prefetch_im2col_4d(j.slot, c[0], c[1], c[2], c[3], o1, o2); break;
    default: vx_dxa_prefetch_im2col_5d(j.slot, c[0], c[1], c[2], c[3], c[4], o1, o2, o3); break;
    }
    return;
  }
  switch (j.rank) {
  case 1:  vx_dxa_prefetch_1d(j.slot, c[0]); break;
  case 2:  vx_dxa_prefetch_2d(j.slot, c[0], c[1]); break;
  case 3:  vx_dxa_prefetch_3d(j.slot, c[0], c[1], c[2]); break;
  case 4:  vx_dxa_prefetch_4d(j.slot, c[0], c[1], c[2], c[3]); break;
  default: vx_dxa_prefetch_5d(j.slot, c[0], c[1], c[2], c[3], c[4]); break;
  }
}

INL void issue(const job_t& j, uint8_t* lm, const vortex::cluster_group& cl) {
  void* dst = lm + j.dst_off;
  uint64_t* mb = reinterpret_cast<uint64_t*>(lm + LM_MBAR + j.mbar_skew);
  if (j.dst_rank != DST_LOCAL) {
    dst = cl.map_shared_rank(dst, j.dst_rank);
    mb  = cl.map_shared_rank(mb, j.dst_rank);
  }
  switch (j.kind) {
  case JOB_TLOAD:
    tload(j, dst, mb);
    break;
  case JOB_TPREFETCH:
    tprefetch(j);
    break;
  case JOB_BLOAD:
    if (j.mask)
      vx_dxa_bulk_load_multicast(dst, reinterpret_cast<const void*>(j.gsrc), j.size, mb, (uint16_t)j.mask);
    else
      vx_dxa_bulk_load(dst, reinterpret_cast<const void*>(j.gsrc), j.size, mb);
    break;
  default:
    vx_dxa_prefetch_l2(reinterpret_cast<const void*>(j.gsrc), j.size);
    break;
  }
}

// Waits for the phase of the given parity; true when it did not complete
// within WAIT_LIMIT try_waits.
INL bool wait_phase(uint64_t* mbar, uint32_t parity, uint32_t* tries) {
  ++*tries;
  while (!vx_mbar_try_wait_parity(mbar, parity)) {
    if (++*tries > WAIT_LIMIT)
      return true;
  }
  return false;
}

} // namespace

// -p: warp 0 alone issues the jobs `repeats` times, each a phase of the
// mbarrier armed with arrive_tx, and times the whole loop. Nothing else runs,
// so the run's cycles are the engine's; thread 0 then checks the host's spot
// bytes and reports the mbarrier word.
INL void perf_run(const load_arg_t* arg, uint8_t* lm, uint64_t* mbar, uint32_t* out,
                  const vortex::cluster_group& cl) {
  const uint32_t tid = threadIdx.x;
  const uint32_t tx  = arg->tx_in[0];
  auto jobs = reinterpret_cast<const job_t*>(arg->jobs_addr);
  if (tid == 0)
    vx_mbar_init(mbar, 1);
  uint32_t tries = 0, err = 0;
  const uint64_t t0 = vx_rdcycle();
  for (uint32_t rep = 0; rep < arg->repeats; ++rep) {
    if (tid == 0 && tx)
      vx_mbar_arrive_tx(mbar, tx);
    vx_fence_proxy_async();
    for (uint32_t i = 0; i < arg->num_jobs; ++i)
      issue(jobs[i], lm, cl);
    if (tx)
      err |= wait_phase(mbar, rep & 1, &tries);
  }
  const uint64_t t1 = vx_rdcycle();
  if (tid == 0) {
    auto spots = reinterpret_cast<const spot_t*>(arg->spots_addr);
    auto lb = reinterpret_cast<volatile uint8_t*>(lm);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < arg->num_spots; ++i)
      bad += (lb[spots[i].off] != spots[i].val);
    const uint64_t mw = *reinterpret_cast<volatile uint64_t*>(mbar);
    out[ERR_SPOTS]   = bad;
    out[ERR_CYCLES]  = (uint32_t)(t1 - t0);
    out[ERR_MBAR_LO] = (uint32_t)mw;
    out[ERR_MBAR_HI] = (uint32_t)(mw >> 32);
  }
  out[ERR_WAIT]  = err;
  out[ERR_TRIES] = tries;
}

// Every block fills its local memory with canaries, initializes its mbarrier
// and arms it with arrive_tx of the bytes it will receive, fences the stores
// for the async proxy, then the cluster barrier. Warp 0 of each issuing rank
// issues its jobs in order; every thread of a receiving block waits for phase
// 0, and the block dumps its whole local memory for the host's byte-exact
// check against dxa_ref.
__kernel void load_k(load_arg_t* __UNIFORM__ arg) {
  const uint32_t tid  = threadIdx.x;
  const uint32_t nthr = blockDim.x;
  const uint32_t b    = blockIdx.x;
  const vortex::cluster_group cl = vortex::this_cluster();
  const uint32_t r    = cl.block_rank();
  auto lm   = reinterpret_cast<uint8_t*>(__local_mem());
  auto lw   = reinterpret_cast<volatile uint32_t*>(lm);
  auto mbar = reinterpret_cast<uint64_t*>(lm + LM_MBAR);
  const uint32_t words = arg->lmem_bytes / sizeof(uint32_t);
  const uint32_t tx = arg->tx_in[r];
  auto out = reinterpret_cast<uint32_t*>(arg->err_addr) + (b * nthr + tid) * ERR_WORDS;

  if (arg->repeats) {
    if (get_sub_group_id() == 0)
      perf_run(arg, lm, mbar, out, cl);
    out[ERR_DONE] = 1;
    return;
  }

  for (uint32_t w = tid; w < words; w += nthr) {
    if (w >= 2)
      lw[w] = canary_word(b, w);
  }
  if (tid == 0) {
    vx_mbar_init(mbar, 1);
    if (tx)
      vx_mbar_arrive_tx(mbar, tx);
  }
  vx_fence_proxy_async();
  cl.sync();

  if (get_sub_group_id() == 0) {
    auto jobs = reinterpret_cast<const job_t*>(arg->jobs_addr);
    for (uint32_t i = 0; i < arg->num_jobs; ++i) {
      if (jobs[i].issuer == r)
        issue(jobs[i], lm, cl);
    }
  }

  if (arg->neg) {
    idle_cycles(arg->linger);
    out[ERR_DONE] = 1;
    return;
  }

  uint32_t tries = 0, err = 0;
  if (tx)
    err = wait_phase(mbar, 0, &tries);
  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr + (uint64_t)b * arg->lmem_bytes);
  for (uint32_t w = tid; w < words; w += nthr)
    dump[w] = lw[w];
  cl.sync();
  out[ERR_WAIT]  = err;
  out[ERR_TRIES] = tries;
  out[ERR_DONE]  = 1;
}
