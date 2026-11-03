#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_atomic.h>
#if VX_CFG_EXT_DLMEM_ENABLED
#include <vx_dlmem.h>
#endif
#include "common.h"

namespace {

#define INL inline __attribute__((always_inline))

// The FPAMO encoding, independent of vx_atomic.h: custom-3 R-type,
// funct7 = (op - 16) << 2, funct3 = width.
#define FPAMO(op, w, p, v) ({                                                  \
    uint64_t __r;                                                              \
    __asm__ volatile (".insn r 0x7B, %2, %3, %0, %1, %4"                       \
                      : "=r"(__r) : "r"(p), "i"(w), "i"(((op) - 16) << 2), "r"(v) \
                      : "memory");                                             \
    __r; })
#define FPAMO_RED(op, w, p, v)                                                 \
    __asm__ volatile (".insn r 0x7B, %1, %2, x0, %0, %3"                       \
                      :: "r"(p), "i"(w), "i"(((op) - 16) << 2), "r"(v) : "memory")

#define HW_CASES(X)                                                            \
  X(16, 2) X(17, 2) X(18, 2) X(18, 3) X(20, 2) X(20, 3)                        \
  X(21, 2) X(22, 2) X(23, 2) X(24, 2) X(25, 2)

INL uint64_t hw_op(uint32_t op, uint32_t w, void* p, uint64_t v, bool red) {
#define X(o, ww)                                                               \
  case ((o) << 2) | (ww):                                                      \
    if (red) { FPAMO_RED(o, ww, p, v); return 0; }                             \
    return FPAMO(o, ww, p, v);
  switch ((op << 2) | w) {
  HW_CASES(X)
  default: return 0xbadbadbadull;
  }
#undef X
}

INL uint32_t f2u(float x) { return __vx_f2u(x); }
INL uint64_t d2u(double x) { return __vx_d2u(x); }

// A call of the space variant sp of a vx_atomic.h function.
#define SPC(sp, pre, rest, args)                                               \
  ((sp) == SP_GLOBAL ? pre##global_##rest args                                 \
   : (sp) == SP_SHARED ? pre##shared_##rest args                               \
   : (sp) == SP_CLUSTER ? pre##shared_cluster_##rest args : pre##rest args)
// Vectors exist only as generic and global forms.
#define VSPC(sp, pre, rest, args) \
  ((sp) == SP_GLOBAL ? pre##global_##rest args : pre##rest args)

// A vx_atomic.h form on p with operand (v, v2) and comparand cmp; the old
// value(s) in ret.
INL void form_op(uint32_t form, uint32_t sp, void* p, uint64_t v, uint64_t v2, uint64_t cmp, bool red,
                 uint64_t* ret) {
  union U { uint64_t u[2]; uint32_t w[4]; uint16_t h[8]; float f[4]; };
  U vv, old;
  vv.u[0] = v;
  vv.u[1] = v2;
  old.u[0] = old.u[1] = 0;
  uint16_t* p16 = (uint16_t*)p;
  uint32_t* p32 = (uint32_t*)p;
  float* pf = (float*)p;
  switch (form) {
  case FORM_ADD_F16:
    if (red) SPC(sp, vx_red_, add_relaxed_gpu_f16, (p16, (uint16_t)v));
    else old.u[0] = SPC(sp, vx_atom_, add_acq_rel_cta_f16, (p16, (uint16_t)v));
    break;
  case FORM_ADD_BF16:
    if (red) SPC(sp, vx_red_, add_release_cluster_bf16, (p16, (uint16_t)v));
    else old.u[0] = SPC(sp, vx_atom_, add_relaxed_gpu_bf16, (p16, (uint16_t)v));
    break;
  case FORM_CAS_B16:
    old.u[0] = SPC(sp, vx_atom_, cas_acquire_gpu_b16, (p16, (uint16_t)cmp, (uint16_t)v));
    break;
  case FORM_ADD_F32:
    if (red) SPC(sp, vx_red_, add_relaxed_gpu_f32, (pf, vv.f[0]));
    else old.f[0] = SPC(sp, vx_atom_, add_relaxed_gpu_f32, (pf, vv.f[0]));
    break;
  case FORM_ADD_F64:
    if (red) SPC(sp, vx_red_, add_relaxed_gpu_f64, ((double*)p, __vx_u2d(v)));
    else old.u[0] = __vx_d2u(SPC(sp, vx_atom_, add_release_gpu_f64, ((double*)p, __vx_u2d(v))));
    break;
  case FORM_ADD_F16X2:
    if (red) SPC(sp, vx_red_, add_relaxed_gpu_f16x2, (p32, vv.w[0]));
    else old.u[0] = SPC(sp, vx_atom_, add_relaxed_gpu_f16x2, (p32, vv.w[0]));
    break;
  case FORM_ADD_BF16X2:
    if (red) SPC(sp, vx_red_, add_relaxed_gpu_bf16x2, (p32, vv.w[0]));
    else old.u[0] = SPC(sp, vx_atom_, add_relaxed_gpu_bf16x2, (p32, vv.w[0]));
    break;
  case FORM_INC_U32:
    if (red) SPC(sp, vx_red_, inc_relaxed_gpu_u32, (p32, vv.w[0]));
    else old.u[0] = SPC(sp, vx_atom_, inc_relaxed_gpu_u32, (p32, vv.w[0]));
    break;
  case FORM_DEC_U32:
    if (red) SPC(sp, vx_red_, dec_relaxed_gpu_u32, (p32, vv.w[0]));
    else old.u[0] = SPC(sp, vx_atom_, dec_acquire_cluster_u32, (p32, vv.w[0]));
    break;
  case FORM_ADD_V2_F32:
    if (red) VSPC(sp, vx_red_, add_relaxed_gpu_v2_f32, (pf, vv.f));
    else VSPC(sp, vx_atom_, add_relaxed_gpu_v2_f32, (pf, vv.f, old.f));
    break;
  case FORM_ADD_V4_F32:     VSPC(sp, vx_atom_, add_release_gpu_v4_f32, (pf, vv.f, old.f)); break;
  case FORM_RED_ADD_V4_F32: VSPC(sp, vx_red_, add_relaxed_gpu_v4_f32, (pf, vv.f)); break;
  case FORM_ADD_V8_F16:     VSPC(sp, vx_atom_, add_relaxed_gpu_v8_f16, (p16, vv.h, old.h)); break;
  case FORM_ADD_V8_BF16:    VSPC(sp, vx_atom_, add_relaxed_gpu_v8_bf16, (p16, vv.h, old.h)); break;
  case FORM_MIN_V8_F16:     VSPC(sp, vx_atom_, min_relaxed_gpu_v8_f16, (p16, vv.h, old.h)); break;
  case FORM_MAX_V8_BF16:    VSPC(sp, vx_atom_, max_acquire_cta_v8_bf16, (p16, vv.h, old.h)); break;
  case FORM_ADD_V4_F16X2:   VSPC(sp, vx_atom_, add_relaxed_gpu_v4_f16x2, (p32, vv.w, old.w)); break;
  case FORM_MAX_V4_BF16X2:  VSPC(sp, vx_atom_, max_relaxed_gpu_v4_bf16x2, (p32, vv.w, old.w)); break;
  case FORM_MIN_V2_F16:     VSPC(sp, vx_atom_, min_relaxed_gpu_v2_f16, (p16, vv.h, old.h)); break;
  case FORM_MAX_V2_BF16:    VSPC(sp, vx_atom_, max_relaxed_gpu_v2_bf16, (p16, vv.h, old.h)); break;
  default:
    old.u[0] = old.u[1] = 0xbadf0badbadf0badull;
    break;
  }
  ret[0] = old.u[0];
  ret[1] = old.u[1];
}

INL void record(const arg_t* a, uint32_t rank) {
  if (threadIdx.x == 0) {
    auto out = (uint32_t*)a->out_addr + blockIdx.x * OUT_WORDS(a->m);
    out[OUT_RANK] = rank;
    out[OUT_CORE] = vx_core_id();
    out[OUT_RAN]  = 1;
  }
}

INL uint64_t* out_data(const arg_t* a, uint32_t b) {
  return (uint64_t*)((uint32_t*)a->out_addr + b * OUT_WORDS(a->m) + OUT_DATA);
}

INL uint32_t my_rank(const arg_t* a) {
#if VX_CFG_EXT_DLMEM_ENABLED
  if (a->space == SPACE_REMOTE)
    return vx_cluster_block_rank();
#endif
  (void)a;
  return 0;
}

INL void space_sync(const arg_t* a) {
#if VX_CFG_EXT_DLMEM_ENABLED
  if (a->space == SPACE_REMOTE) {
    vx_cluster_sync();
    return;
  }
#endif
  (void)a;
  __syncthreads();
}

// Base of the targets: per block in global memory, the block's own local
// memory, or rank `rank` of the cluster through the window.
INL char* target(const arg_t* a, uint32_t per_block, uint32_t rank) {
  if (a->space == SPACE_GLOBAL)
    return (char*)a->glob_addr + blockIdx.x * per_block;
  char* own = (char*)__local_mem();
#if VX_CFG_EXT_DLMEM_ENABLED
  if (a->space == SPACE_REMOTE)
    return (char*)vx_cluster_map_shared_rank(own, rank);
#endif
  (void)rank;
  return own;
}

INL uint32_t to_f16(uint32_t v) { return __vx_f32_to_f16(f2u((float)v)); }
INL uint32_t to_bf16(uint32_t v) { return f2u((float)v) >> 16; }

// hist/contend operand of `form` from integer values a (lane 0) and b (lane 1).
INL void encode(uint32_t form, uint32_t a, uint32_t b, uint64_t* v, uint64_t* v2) {
  *v2 = 0;
  switch (form) {
  case FORM_ADD_F32:    *v = f2u((float)a); break;
  case FORM_ADD_F64:    *v = d2u((double)a); break;
  case FORM_ADD_F16:    *v = to_f16(a); break;
  case FORM_ADD_BF16:   *v = to_bf16(a); break;
  case FORM_ADD_F16X2:  *v = to_f16(a) | (to_f16(b) << 16); break;
  case FORM_ADD_BF16X2: *v = to_bf16(a) | (to_bf16(b) << 16); break;
  case FORM_ADD_V2_F32: *v = f2u((float)a) | ((uint64_t)f2u((float)b) << 32); break;
  default:              *v = 0xffffffffu; break;  // inc: never wraps
  }
}

}  // namespace

__kernel void edge_k(arg_t* __UNIFORM__ a) {
  const uint32_t t = threadIdx.x, nt = blockDim.x;
  const uint32_t rank = my_rank(a);
  record(a, rank);
  space_sync(a);
  volatile uint64_t* slot = (volatile uint64_t*)(target(a, nt * SLOT_BYTES, (rank + 1) % a->k) + t * SLOT_BYTES);
  const rec_t* recs = (const rec_t*)a->rec_addr;
  res_t* res = (res_t*)a->res_addr + blockIdx.x * a->num_recs;
  for (uint32_t i = t; i < a->num_recs; i += nt) {
    const rec_t& r = recs[i];
    slot[0] = r.old[0];
    slot[1] = r.old[1];
    vx_fence();
    void* p = (char*)slot + r.offset;
    uint64_t ret[2] = {0, 0};
    // fflags is per warp: a value every lane of it writes alike.
    const uint32_t fin = ((i / nt) * 7 + 3) & 0x1f;
    csr_write(VX_CSR_FFLAGS, fin);
    if (r.impl == IMPL_HDR)
      form_op(r.form, r.sp, p, r.rhs[0], r.rhs[1], r.cmp, false, ret);
    else
      ret[0] = hw_op(r.op, r.width, p, r.rhs[0], r.impl == IMPL_RED);
    const uint32_t fout = csr_read(VX_CSR_FFLAGS);
    vx_fence();
    res[i].mem[0] = slot[0];
    res[i].mem[1] = slot[1];
    res[i].ret[0] = ret[0];
    res[i].ret[1] = ret[1];
    res[i].fflags = (fin << 8) | fout;
  }
  space_sync(a);
}

__kernel void hist_k(arg_t* __UNIFORM__ a) {
  const uint32_t t = threadIdx.x, b = blockIdx.x;
  const uint32_t rank = my_rank(a);
  record(a, rank);
  if (a->space != SPACE_GLOBAL && rank == 0 && t == 0) {
    volatile uint64_t* own = (volatile uint64_t*)__local_mem();
    for (uint32_t i = 0; i < HIST_BINS; ++i)
      own[i] = 0;
  }
  space_sync(a);
  char* bins = (a->space == SPACE_GLOBAL) ? (char*)a->glob_addr : target(a, 0, 0);
  for (uint32_t j = 0; j < a->iters; ++j) {
    const uint32_t h = hist_hash(a->seed, b, t, j);
    uint64_t v, v2, ret[2];
    encode(a->form, hist_val(h, 0, a->m), hist_val(h, 1, a->m), &v, &v2);
    form_op(a->form, a->sp, bins + 8 * hist_bin(h), v, v2, 0, a->red != 0, ret);
  }
  space_sync(a);
  if (a->space != SPACE_GLOBAL && rank == 0 && t == 0) {
    volatile uint64_t* own = (volatile uint64_t*)__local_mem();
    uint64_t* out = out_data(a, b);
    for (uint32_t i = 0; i < HIST_BINS; ++i)
      out[i] = own[i];
  }
  space_sync(a);
}

__kernel void contend_k(arg_t* __UNIFORM__ a) {
  const uint32_t t = threadIdx.x, b = blockIdx.x;
  const uint32_t rank = my_rank(a);
  record(a, rank);
  if (a->space != SPACE_GLOBAL && rank == 0 && t == 0)
    *(volatile uint64_t*)__local_mem() = 0;
  space_sync(a);
  char* word = (a->space == SPACE_GLOBAL) ? (char*)a->glob_addr : target(a, 0, 0);
  const uint64_t c0 = csr_read(VX_CSR_MCYCLE);
  for (uint32_t j = 0; j < a->iters; ++j) {
    uint64_t ret[2];
    form_op(a->form, a->sp, word, a->operand, 0, 0, a->red != 0, ret);
  }
  __syncthreads();
  const uint64_t c1 = csr_read(VX_CSR_MCYCLE);
  if (t == 0) {
    auto out = (uint32_t*)a->out_addr + b * OUT_WORDS(a->m);
    out[OUT_CYCLES]     = (uint32_t)(c1 - c0);
    out[OUT_CYCLES + 1] = (uint32_t)((c1 - c0) >> 32);
  }
  space_sync(a);
  if (a->space != SPACE_GLOBAL && rank == 0 && t == 0)
    out_data(a, b)[0] = *(volatile uint64_t*)__local_mem();
  space_sync(a);
}

#if VX_CFG_EXT_DLMEM_ENABLED
// Every rank adds its vector into every peer's accumulator with
// red.shared::cluster.add.f32, as the atomic_all_push collective.
__kernel void allpush_k(arg_t* __UNIFORM__ a) {
  const uint32_t t = threadIdx.x, nt = blockDim.x, k = a->k, m = a->m;
  const uint32_t rank = vx_cluster_block_rank();
  record(a, rank);
  float* acc = (float*)__local_mem();
  for (uint32_t e = t; e < m; e += nt)
    acc[e] = (float)allpush_val(rank, e);
  vx_cluster_sync();
  for (uint32_t idx = t; idx < (k - 1) * m; idx += nt) {
    const uint32_t peer = (rank + 1 + idx / m) % k, e = idx % m;
    vx_red_shared_cluster_add_relaxed_cluster_f32((float*)vx_cluster_map_shared_rank(&acc[e], peer),
                                   (float)allpush_val(rank, e));
  }
  vx_cluster_sync();
  uint32_t* out = (uint32_t*)out_data(a, blockIdx.x);
  for (uint32_t e = t; e < m; e += nt)
    out[e] = f2u(acc[e]);
  vx_cluster_sync();
}
#endif

// The global 16-bit add loop, as ClusterFusion's atomicAdd(__half*): threads
// of every block add 1.0 to the elements of one 16-byte vector.
__kernel void f16loop_k(arg_t* __UNIFORM__ a) {
  const uint32_t t = threadIdx.x, b = blockIdx.x;
  record(a, 0);
  if (a->space == SPACE_LOCAL && t == 0) {
    volatile uint64_t* own = (volatile uint64_t*)__local_mem();
    own[0] = own[1] = 0;
  }
  __syncthreads();
  uint16_t* vec = (uint16_t*)((a->space == SPACE_GLOBAL) ? (char*)a->glob_addr : (char*)__local_mem());
  const uint16_t one = (a->form == FORM_ADD_BF16) ? 0x3f80 : 0x3c00;
  for (uint32_t j = 0; j < a->iters; ++j) {
    uint16_t* p = vec + (t + j + b) % F16LOOP_ELEMS;
    uint64_t ret[2];
    form_op(a->form, a->sp, p, one, 0, 0, a->red != 0, ret);
  }
  __syncthreads();
  if (a->space == SPACE_LOCAL && t == 0) {
    volatile uint64_t* own = (volatile uint64_t*)__local_mem();
    out_data(a, b)[0] = own[0];
    out_data(a, b)[1] = own[1];
  }
}
