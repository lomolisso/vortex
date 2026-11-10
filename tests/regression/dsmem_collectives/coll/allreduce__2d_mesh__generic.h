// AllReduce on a logical R x C grid of the cluster's blocks: reduce-scatter
// along each row, all-reduce of the row partial along each column, then
// all-gather along each row, so every message stays within one row or column.
// Port of h100-dsmem-collectives/collectives/allreduce__2d_mesh__generic.cu
// (commit 2b9a97e); origin: the generic 2D mesh/torus all-reduce (TPU pods,
// hierarchical NCCL).
// Departures: none.
#pragma once

#include <stddef.h>
#include <stdint.h>
#ifdef __riscv
#include "../dsmem_primitives.h"
#endif

namespace allreduce__2d_mesh {

constexpr int kOp = 0;
constexpr bool kTypeAgnostic = false;

// 16 -> 4x4, 8 -> 2x4, 4 -> 2x2, 2 -> 1x2; rank = row * C + col.
template <int N>
struct Grid2D {
  static constexpr int R = (N >= 16) ? 4 : (N >= 4 ? 2 : 1);
  static constexpr int C = N / R;
};

struct CollectiveSmem {
  static constexpr int N = CLUSTER_N, M = MSG_ELEMS, C = Grid2D<N>::C, R = Grid2D<N>::R;
  alignas(16) ELEM_T buf[M];
  alignas(16) ELEM_T row_slots[C][M / C];
  alignas(16) ELEM_T col_slots[R][M / C];
  alignas(16) ELEM_T chunk[M / C];
  alignas(8) uint64_t bars[3];
};

template <int N, int M, class T>
constexpr const char* illegal() {
  constexpr int R = Grid2D<N>::R, C = Grid2D<N>::C;
  if (R * C != N)
    return "grid must cover the cluster";
  if (M % C != 0 || ((M / C) * sizeof(T)) % 16 != 0)
    return "chunk must be a multiple of 16 bytes";
  return nullptr;
}

#ifdef __riscv
template <int N, class T, int M, class Op>
__device__ void run(const T* in, T* out, unsigned long long* ts, unsigned* smid_out) {
  using namespace dsmem;
  constexpr int R = Grid2D<N>::R, C = Grid2D<N>::C;
  static_assert(N == CollectiveSmem::N && M == CollectiveSmem::M, "run() must match the build's CLUSTER_N and MSG_ELEMS");
  static_assert(R * C == N, "grid must cover the cluster");
  static_assert(M % C == 0 && ((M / C) * sizeof(T)) % 16 == 0, "chunk must be a multiple of 16 bytes");
  constexpr int K = M / C;
  constexpr uint32_t bytes = K * sizeof(T);

  CollectiveSmem& sm = dyn_smem<CollectiveSmem>();
  T* buf = sm.buf;
  T* chunk = sm.chunk;
  const Op op{};
  const uint32_t rank = cluster_rank();
  Stamps st{ts + rank * kStamps};

  load_block<T, M>(buf, in + rank * M);
  fence_proxy_async_smem();
  if (threadIdx.x == 0) {
    for (int i = 0; i < 3; ++i)
      mbar_init(smem_u32(&sm.bars[i]), 1);
    fence_mbar_init();
    smid_out[rank] = smid();
  }
  cluster_sync();
  st.mark(0);

  const uint32_t row = rank / C, col = rank % C;
  const uint32_t buf_a = smem_u32(buf);
  const uint32_t rs_a = smem_u32(&sm.row_slots[0][0]);
  const uint32_t cs_a = smem_u32(&sm.col_slots[0][0]);
  const uint32_t chunk_a = smem_u32(chunk);
  const uint32_t bar_row = smem_u32(&sm.bars[0]);
  const uint32_t bar_col = smem_u32(&sm.bars[1]);
  const uint32_t bar_ag = smem_u32(&sm.bars[2]);

  // Row reduce-scatter: chunk dc goes to row_slots[col] of block (row, dc).
  if (threadIdx.x == 0) {
    mbar_arrive_expect_tx(bar_row, (C - 1) * bytes);
    for (uint32_t j = 1; j < (uint32_t)C; ++j) {
      const uint32_t dc = (col + j) % C, dst = row * C + dc;
      bulk_push(dst, rs_a + col * bytes, buf_a + dc * bytes, bytes, bar_row);
    }
  }
  mbar_wait_parity(bar_row, 0);
  for (int i = threadIdx.x; i < K; i += (int)blockDim.x) {
    T acc = buf[col * K + i];
#pragma unroll
    for (int c = 0; c < C; ++c)
      if (c != (int)col)
        acc = op(acc, sm.row_slots[c][i]);
    buf[col * K + i] = acc;
  }
  fence_proxy_async_smem();
  __syncthreads();
  st.round(0);

  // Column all-reduce of chunk col: the row partial goes to col_slots[row] of
  // every (dr, col). The result goes to `chunk`, since this block's own pushes
  // may still be reading buf[col].
  if constexpr (R > 1) {
    if (threadIdx.x == 0) {
      mbar_arrive_expect_tx(bar_col, (R - 1) * bytes);
      for (uint32_t j = 1; j < (uint32_t)R; ++j) {
        const uint32_t dr = (row + j) % R, dst = dr * C + col;
        bulk_push(dst, cs_a + row * bytes, buf_a + col * bytes, bytes, bar_col);
      }
    }
    mbar_wait_parity(bar_col, 0);
    for (int i = threadIdx.x; i < K; i += (int)blockDim.x) {
      T acc = buf[col * K + i];
#pragma unroll
      for (int r = 0; r < R; ++r)
        if (r != (int)row)
          acc = op(acc, sm.col_slots[r][i]);
      chunk[i] = acc;
    }
  } else {
    (void)cs_a;
    (void)bar_col;
    for (int i = threadIdx.x; i < K; i += (int)blockDim.x)
      chunk[i] = buf[col * K + i];
  }
  fence_proxy_async_smem();
  __syncthreads();
  st.round(1);

  // Row all-gather: chunk col, reduced, goes to offset col of every block in the row.
  if (threadIdx.x == 0) {
    mbar_arrive_expect_tx(bar_ag, (C - 1) * bytes);
    for (uint32_t j = 1; j < (uint32_t)C; ++j) {
      const uint32_t dst = row * C + (col + j) % C;
      bulk_push(dst, buf_a + col * bytes, chunk_a, bytes, bar_ag);
    }
  }
  mbar_wait_parity(bar_ag, 0);
  st.round(2);

  st.mark(1);
  __syncthreads();
  // Chunk col is in `chunk`; buf[col] is stale.
  for (int i = threadIdx.x; i < M; i += (int)blockDim.x) {
    const int c = i / K;
    out[rank * M + i] = (c == (int)col) ? chunk[i % K] : buf[i];
  }
  cluster_sync();
}
#endif

} // namespace allreduce__2d_mesh
