// One block of 8 warps. Local memory starts as a host image; each warp runs
// every stmatrix and ldmatrix variant on its own regions with host-given row
// offsets and registers, one warpgroup stores DAK's accumulator tile with
// stmatrix.x4.trans and both warpgroups store DeepGEMM's with stmatrix.x2,
// at the addresses those kernels compute; then named barriers 1-15 hand a
// word from a delayed writer warp to a reader warp, the writer syncing or
// arriving. Local memory is copied back for the host to compare byte for byte.

#include <vx_cuda.h>
#include "common.h"

namespace {

template <uint32_t V>
void st_variant(char* lm, const mat_entry_t& e) {
  char* p = lm + e.row;
  if constexpr (V == 0) stmatrix_x1(p, e.r[0]);
  else if constexpr (V == 1) stmatrix_x2(p, e.r[0], e.r[1]);
  else if constexpr (V == 2) stmatrix_x4(p, e.r[0], e.r[1], e.r[2], e.r[3]);
  else if constexpr (V == 3) stmatrix_x1_trans(p, e.r[0]);
  else if constexpr (V == 4) stmatrix_x2_trans(p, e.r[0], e.r[1]);
  else stmatrix_x4_trans(p, e.r[0], e.r[1], e.r[2], e.r[3]);
}

template <uint32_t V>
void ld_variant(const char* lm, const mat_entry_t& e, uint32_t* out) {
  const char* p = lm + e.row;
  uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  if constexpr (V == 0) ldmatrix_x1(p, r0);
  else if constexpr (V == 1) ldmatrix_x2(p, r0, r1);
  else if constexpr (V == 2) ldmatrix_x4(p, r0, r1, r2, r3);
  else if constexpr (V == 3) ldmatrix_x1_trans(p, r0);
  else if constexpr (V == 4) ldmatrix_x2_trans(p, r0, r1);
  else ldmatrix_x4_trans(p, r0, r1, r2, r3);
  out[0] = r0;
  out[1] = r1;
  out[2] = r2;
  out[3] = r3;
}

template <uint32_t V>
__attribute__((noinline)) void matrix(const kernel_arg_t* arg, char* lm, uint32_t warp, uint32_t lane) {
  const mat_entry_t* tab = (const mat_entry_t*)arg->mat_addr;
  const mat_entry_t e = tab[(V * kWarps + warp) * 32 + lane];
  if constexpr (V < 6) {
    st_variant<V>(lm, e);
  } else {
    uint32_t* out = (uint32_t*)arg->ld_addr + (((V - 6) * kWarps + warp) * 32 + lane) * 4;
    ld_variant<V - 6>(lm, e, out);
  }
}

// DAK's epilogue (stmatrix.x4.trans of the packed m64n256 f16 fragment into
// the 128B-swizzled sC): call c stores registers 4c..4c+3; lane 8q + j of warp
// w gives the row of sC(m = 16w + 8((4c + q) % 2), n = 8((4c + q) / 2) + j).
__attribute__((noinline)) void dak_epilogue(const kernel_arg_t* arg, char* lm, uint32_t warp, uint32_t lane) {
  const uint32_t t = warp * 32 + lane;
  const uint32_t* regs = (const uint32_t*)arg->dak_regs + t * 64;
  uint32_t* rec = (uint32_t*)arg->dak_addr + t * kDakCalls;
  const uint32_t q = lane / 8, j = lane % 8;
  for (uint32_t c = 0; c < kDakCalls; ++c) {
    const uint32_t m = 16 * warp + 8 * ((4 * c + q) % 2), n = 8 * ((4 * c + q) / 2) + j;
    const uint32_t off = n * 128 + (((m / 8) ^ (n % 8)) * 16);
    rec[c] = off;
    stmatrix_x4_trans(lm + LM_DAK + off, regs[4 * c], regs[4 * c + 1], regs[4 * c + 2], regs[4 * c + 3]);
  }
}

// DeepGEMM's epilogue (stmatrix.x2 of bf16 pairs into three 64-column
// 128B-swizzled boxes): lane l < 16 of warp w gives
// w * 2048 + (i / 8) * 16384 + l * 128 + ((i % 8) ^ (l % 8)) * 16.
__attribute__((noinline)) void dg_epilogue(const kernel_arg_t* arg, char* lm, uint32_t warp, uint32_t lane) {
  const uint32_t t = warp * 32 + lane;
  const uint32_t* words = (const uint32_t*)arg->dg_words + t * kDgCalls * 2;
  const uint32_t l = lane % 16;
  for (uint32_t i = 0; i < kDgCalls; ++i) {
    const uint32_t off = warp * 2048 + (i / 8) * 16384 + l * 128 + (((i % 8) ^ (l % 8)) * 16);
    stmatrix_x2(lm + LM_DG + off, words[2 * i], words[2 * i + 1]);
  }
}

// Round r of the named-barrier table; lane 0 of the writer warp stores the
// round's word after a delay, lane 0 of the reader warp reads it after the barrier.
__attribute__((noinline)) void bar_rounds(const kernel_arg_t* arg, char* lm) {
  const uint32_t warp = threadIdx.x / 32, lane = __laneid();
  volatile uint32_t* box = (volatile uint32_t*)(lm + LM_BAR);
  for (uint32_t r = 0; r < kBarRounds; ++r) {
    const bar_round_t br = kBarRound[r];
    const uint32_t group_lo = (br.nthreads == 128) ? (br.writer / 4) * 4 : 0;
    const bool member = (br.nthreads == 64) ? (warp == br.writer || warp == br.reader)
                                            : (warp >= group_lo && warp < group_lo + br.nthreads / 32);
    if (member) {
      if (warp == br.writer && lane == 0) {
        for (uint32_t d = 0; d < kBarDelay; ++d)
          __asm__ volatile ("nop");
        box[r] = arg->bar_tokens + r;
      }
      // CUDA ports name their barriers with constants; DAK's and DeepGEMM's
      // rounds use the template form, the others the runtime form.
      if (br.arrive && warp == br.writer) {
        if (r == 21)
          bar_arrive<12, 128>();
        else
          bar_arrive(br.id, br.nthreads);
      } else if (r == 15)
        bar_sync<8, 256>();
      else if (r == 16)
        bar_sync<9, 256>();
      else if (r == 17)
        bar_sync<13, 128>();
      else if (r == 18)
        bar_sync<15, 128>();
      else
        bar_sync(br.id, br.nthreads);
      if (warp == br.reader && lane == 0)
        ((uint32_t*)arg->bar_addr)[r] = box[r];
    }
  }
}

} // namespace

__global__ void cuda_matrix(kernel_arg_t* arg) {
  char* lm = (char*)__local_mem();
  const uint32_t warp = threadIdx.x / 32, lane = __laneid();

  const uint32_t* init = (const uint32_t*)arg->init_addr;
  for (uint32_t i = threadIdx.x; i < LM_BYTES / 4; i += blockDim.x)
    ((uint32_t*)lm)[i] = init[i];
  __syncthreads();

  if (arg->phases & PHASE_MATRIX) {
    matrix<0>(arg, lm, warp, lane);
    matrix<1>(arg, lm, warp, lane);
    matrix<2>(arg, lm, warp, lane);
    matrix<3>(arg, lm, warp, lane);
    matrix<4>(arg, lm, warp, lane);
    matrix<5>(arg, lm, warp, lane);
    matrix<6>(arg, lm, warp, lane);
    matrix<7>(arg, lm, warp, lane);
    matrix<8>(arg, lm, warp, lane);
    matrix<9>(arg, lm, warp, lane);
    matrix<10>(arg, lm, warp, lane);
    matrix<11>(arg, lm, warp, lane);
  }
  if ((arg->phases & PHASE_DAK) && warp < 4)
    dak_epilogue(arg, lm, warp, lane);
  if (arg->phases & PHASE_DG)
    dg_epilogue(arg, lm, warp, lane);
  __syncthreads();

  if (arg->phases & PHASE_BAR)
    bar_rounds(arg, lm);
  __syncthreads();

  uint32_t* image = (uint32_t*)arg->image_addr;
  for (uint32_t i = threadIdx.x; i < LM_BYTES / 4; i += blockDim.x)
    image[i] = ((const uint32_t*)lm)[i];
}
