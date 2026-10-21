#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_mbar.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// Two warps per block. Warp 0 stages each k-tile's MX scale metadata into one
// local-memory buffer and arrives on FULL; warp 1 waits on FULL, loads the
// metadata into the tensor core with TCU_LD, arrives on EMPTY at once, and
// only then loads its A and B tiles and multiplies. Warp 0 has the next
// k-tile's metadata in registers and overwrites the buffer as soon as EMPTY
// completes: the EMPTY arrive must not complete before the TCU_LDs have read
// the buffer, or the multiply uses the next k-tile's scales. Every lane
// arrives with 1.
static inline __attribute__((always_inline)) bool wait_parity(uint64_t* obj, uint32_t parity) {
  uint32_t spins = 0;
  while (!vx_mbar_test_wait_parity(obj, parity)) {
    if (++spins >= SPIN_LIMIT)
      return false;
  }
  return true;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
  auto pMxA = reinterpret_cast<const uint32_t *>(arg->MX_A_addr);
  auto pMxB = reinterpret_cast<const uint32_t *>(arg->MX_B_addr);

  const uint32_t N = arg->N;
  const uint32_t K = arg->K;
  const uint32_t lane = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t warp = get_sub_group_id();
  const uint32_t tile_row = blockIdx.y * ctx::tileM;
  const uint32_t tile_col = blockIdx.x * ctx::tileN;
  const uint32_t num_k_tiles = K / ctx::tileK;

  auto lm = reinterpret_cast<uint64_t*>(__local_mem());
  uint64_t* full  = lm + LM_FULL;
  uint64_t* empty = lm + LM_EMPTY;
  auto meta = reinterpret_cast<volatile uint32_t*>(lm + LM_META);
  if (threadIdx.x == 0) {
    vx_mbar_init(full, VX_CFG_NUM_THREADS);
    vx_mbar_init(empty, VX_CFG_NUM_THREADS);
  }
  __syncthreads();

  uint32_t timeouts = 0;
  if (warp == 0) {
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
      const uint32_t a = pMxA[(blockIdx.y * num_k_tiles + kt) * VX_CFG_NUM_THREADS + lane];
      const uint32_t b = pMxB[(blockIdx.x * num_k_tiles + kt) * VX_CFG_NUM_THREADS + lane];
      if (kt != 0 && !wait_parity(empty, (kt - 1) & 1))
        ++timeouts;
      meta[lane] = a;
      meta[VX_CFG_NUM_THREADS + lane] = b;
      vx_mbar_arrive(full, 1);
    }
  } else {
    ctx::fragment_a   fragA;
    ctx::fragment_b   fragB;
    ctx::fragment_acc fragC;
    ctx::fill_fragment(fragC, 0);
    auto smeta = const_cast<const uint32_t*>(meta);
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
      const uint32_t i = kt * ctx::tileK;
      if (!wait_parity(full, kt & 1))
        ++timeouts;
      ctx::load_mx_metadata(fragA, smeta);
      ctx::load_mx_metadata(fragB, smeta + VX_CFG_NUM_THREADS);
      vx_mbar_arrive(empty, 1);
      ctx::load_matrix_sync(fragA, pA + tile_row * K + i, K);
      ctx::load_matrix_sync<vt::col_major>(fragB, pB + tile_col * K + i, K);
      ctx::mma_sync(fragC, fragA, fragB, fragC);
    }
    ctx::store_matrix_sync(pC + tile_row * N + tile_col, fragC, N);
  }
  if (lane == 0)
    reinterpret_cast<uint32_t*>(arg->err_addr)[2 * (blockIdx.y * gridDim.x + blockIdx.x) + warp] = timeouts;
}
