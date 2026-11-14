// DAK's multicast GEMV kernel, gemv_multicast<64, 256, 64, 3, 3, 4096, 16384,
// 16384>, as the H100 ran it: DirectAccessKernel 7d1a278 with the patch
// dak_gemm_multicast__sync_fix, include/task/gemv_multicast.cuh
// (task_wgmma_local and gemv_multicast). Our own re-implementation.
//
// One cluster of 2 blocks; each block holds 3 A slots per consumer group, 3 B
// slots and 2 C tiles in shared memory. Producer P0 waits for a slot to drain,
// then rank 0 multicasts group 0's A tile (m64 x k64 of W) to both blocks and
// each block loads its own 256-token B tile; P1 does the same for group 1's A
// tile. Each consumer group runs m64 x n256 x k64 per k-tile on the tensor
// unit (fp16 in, fp16 accumulate), signals the slots drained (rank 0's A slot
// through a cluster mbarrier, since rank 0's multicast overwrites both blocks'
// copies), and after the last k-tile writes its tile to shared memory with
// stmatrix and stores it with a tensor store.
//
// Departures:
// - N split: each H100 consumer warpgroup (m64n256) is two Vortex warpgroups
//   of m64n128 (the tensor unit's packed-fp16 N limit), so a consumer group is
//   256 threads and the block 576 (H100: 128 and 320). NUM_CONSUMER_THREADS =
//   256 keeps the original's role tests as written.
// - Barrier counts and arrival predicates follow from the split: filled,
//   drained and filled_b[0..2] count 257, filled_b[3..5] 258 (a relay from each
//   of group 0's two warpgroups), drained_b 513, cluster_drained_a 4 (each
//   warpgroup's first thread in each block arrives after its own wait_group,
//   which covers only its own four warps); the prologue's and the k-loop's
//   "first thread of the group" arrivals become "first thread of each
//   warpgroup"; bar.sync 13 and 15 count 256 threads.
// - task_wgmma_local is noinline (forceinline on the H100): its pinned
//   accumulators next to the TMA operands in t5/t6 crash the register
//   allocator, and inlining it twice would push the kernel entry past the
//   size vxbin.py reads. It takes no thread index: the per-thread predicates
//   and indices are read from threadIdx.x inside it, since an argument is
//   taken as warp-uniform and a branch on it would lose its split/join.
// - Inside it, wt = threadIdx.x % 128 is the thread of the Vortex warpgroup and
//   nh = (threadIdx.x / 128) & 1 its N half: the B descriptor starts 128 rows
//   (16384 bytes) further for nh = 1, and stmatrix call c of the 8 a thread
//   makes is the H100's call 8 nh + c, its registers shifted down by 32 nh.
// - CuTe's tiled MMA, partitioning and tiled copy are written out: 4 wgmmas
//   of m64n128k16 per k-tile on hand-built descriptors (A MN-major, LBO 0;
//   B K-major, LBO 16; SBO 1024, swizzle 128B, CTA-relative addresses), 32
//   packed fp16 accumulator registers per thread (H100: 64), and 8
//   stmatrix.x4.trans per thread (H100: 16), the swizzle XOR on the
//   CTA-relative address as CuTe's on the shared address.
// - No __shared__: the static mbarriers sit at local-memory offset 0 in
//   declaration order, padded to the H100's 256-byte static block, and the
//   dynamic buffers follow at 0x100, the same 1024-byte phase as the H100's
//   0x500.
// - One argument block holds the three tensor maps (first, 64-byte aligned)
//   and the scalars; the kernel is a template exported by name, checks its
//   cluster shape at entry and is compiled with convergent calls.
// - ClusterBarrier's remote arrive is a plain mbarrier arrive on the address
//   the cluster window maps (the same CTA-scope release).
// - Instrumentation: thread 0 of each block records its core and hardware
//   warp, and a debug option dumps every descriptor the consumers build for
//   slots 0-2 on their first use.
// - Problem shrunk: M 28672 -> 256 (smoke) / 384 (nightly), K 7168 -> 448 /
//   1024, T 512 kept, grid 132 -> 2 (one cluster). Every tile, slot, box,
//   block and cluster constant is kept; K/64 = 7 and 16 are 1 mod 3, as the
//   H100's 112, so the slots wrap mid-tile as there.
// - Inherited tensor-unit gaps: tnspA travels in descriptor bit 52, the f16
//   accumulator is rounded once per k8 FEDP step (the host's bit-exact model
//   follows the tensor unit, not the H100), and the B buffer serialises
//   (performance only).
// - Host: no per-call torch.zeros of C (a NaN prefill instead), our own RNG,
//   a bit-exact check in place of the wrapper's informational "5 repetitions
//   bit-identical" line, and launch_kernel_ex (no <<<>>>) taking the cluster
//   only from __cluster_dims__, with no runtime cluster attribute.

#include <vx_cuda.h>
#include "common.h"
#include "dak_cluster.h"

namespace {

using half_t = __half;
namespace cde = cuda::device::experimental;
namespace h = vortex::h100;
namespace vt = vortex::tensor;
using dak::NUM_CONSUMER_THREADS;
using dak::NUM_PRODUCER_THREADS;
using dak::CLUSTER_SIZE;
using dak::WG_THREADS;
using dak::WGS_PER_GROUP;
using ClusterBarrier = dak::ClusterBarrier;
using block_barrier = cuda::barrier<cuda::thread_scope_block>;

} // namespace

// External linkage, as on the H100: an internal function with this many
// arguments gets the fast calling convention, which passes some of them in
// t5/t6, the registers the DXA operands reserve.
template <int TILE_M_T, int TILE_N_T, int TILE_K_T, int BUFFER_SLOTS_T, int BUFFER_SLOTS_B_T, int TILE_ELEMS_A_T,
          int TILE_ELEMS_B_T, int GROUP_ID>
__noinline__ void task_wgmma_local(half_t* __restrict__ a_buffer, half_t* __restrict__ b_buffer,
                                   half_t* __restrict__ sC, int& nbuf, int tile_ik_base, int tile_ik_stride,
                                   const int NUM_TILES_K, block_barrier filled[], block_barrier drained[],
                                   block_barrier filled_b[], block_barrier drained_b[],
                                   ClusterBarrier::ValueType cluster_drained[], uint64_t* desc_dump) {
  static_assert(TILE_M_T == 64 && TILE_K_T % 16 == 0, "one m64 wgmma row, k16 steps");
  constexpr int N_WG = TILE_N_T / WGS_PER_GROUP;           // 128 tokens per Vortex warpgroup
  constexpr int K_STEPS = TILE_K_T / 16;
  using Acc = h::acc<N_WG, vt::fp16>;

  // Per-thread values, read here (an argument would be taken as uniform).
  const uint32_t wt = threadIdx.x % WG_THREADS;
  const uint32_t nh = (threadIdx.x / WG_THREADS) % WGS_PER_GROUP;

  Acc frag_C;
  frag_C.zero();

  for (int tile_ik = tile_ik_base; tile_ik < NUM_TILES_K; tile_ik += tile_ik_stride) {
    const int next_slot = nbuf % BUFFER_SLOTS_T;
    const int next_slot_b = nbuf % BUFFER_SLOTS_B_T;
    if constexpr (GROUP_ID == 0) {
      filled[next_slot].arrive_and_wait();
      filled_b[next_slot_b].arrive_and_wait();
    } else {
      filled[next_slot].arrive_and_wait();
      filled_b[next_slot_b + BUFFER_SLOTS_B_T].arrive_and_wait();
    }

    half_t* sA = a_buffer + next_slot * TILE_ELEMS_A_T;
    half_t* sB = b_buffer + next_slot_b * TILE_ELEMS_B_T;

    // sA(m, k) = sw128(k * 128 + 2m): one 64-element MN atom (LBO 0), 8 k rows
    // per 1024-byte group (SBO), 16 k rows = 2048 bytes per k-step.
    // sB(n, k) = sw128(n * 128 + 2k): 8 tokens per 1024-byte group (SBO), 32
    // bytes per k-step; this warpgroup's tokens start N_WG rows in for nh = 1.
    uint64_t dA[K_STEPS], dB[K_STEPS];
    vt::detail::unroll_for<K_STEPS>([&](auto kk) __attribute__((always_inline)) {
      dA[kk] = h::make_desc((char*)sA + kk * 16 * 128, h::swz::b128, 0, 1024);
      dB[kk] = h::make_desc((char*)sB + nh * N_WG * 128 + kk * 32, h::swz::b128, 16, 1024);
    });
    if (desc_dump != nullptr && nbuf < BUFFER_SLOTS_T && wt == 0) {
      const uint32_t wg = (uint32_t)GROUP_ID * WGS_PER_GROUP + nh;
      for (int kk = 0; kk < K_STEPS; ++kk) {
        desc_dump[dak::desc_index(wg, (uint32_t)next_slot, (uint32_t)kk, 0)] = dA[kk];
        desc_dump[dak::desc_index(wg, (uint32_t)next_slot, (uint32_t)kk, 1)] = dB[kk];
      }
    }

    cute::warpgroup_arrive();
    vt::detail::unroll_for<K_STEPS>([&](auto kk) __attribute__((always_inline)) {
      h::wgmma<N_WG, vt::fp16, vt::fp16, vt::fp16, true, false>(dA[kk], dB[kk], frag_C);
    });
    cute::warpgroup_commit_batch();

    if (GROUP_ID == 0 && wt == 0) {
      (void)filled_b[next_slot_b + BUFFER_SLOTS_B_T].arrive();
    }

    cute::warpgroup_wait<0>();
    if (wt == 0) {
      ClusterBarrier::arrive(&cluster_drained[next_slot], 0, 1);
    }
    (void)drained[next_slot].arrive();
    (void)drained_b[next_slot_b].arrive();
    nbuf++;
  }

  // The accumulators to sC(m, n) = sw128(n * 128 + 2m) with stmatrix.x4.trans:
  // call c stores registers 4c..4c+3 as four transposed 8x8 matrices; lane L
  // gives the row address of row L % 8 of matrix L / 8, which holds 8
  // consecutive m of one token n.
  char* const lm = (char*)__local_mem();
  const uint32_t w = wt / 32, L = wt % 32, q = L / 8, jj = L % 8;
  const uint32_t c_base = h::cta_addr(sC);
  vt::detail::unroll_for<Acc::NRC / 4>([&](auto c) __attribute__((always_inline)) {
    const uint32_t r = 4 * (uint32_t)c + q;
    const uint32_t n = nh * N_WG + 8 * (r / 2) + jj;
    const uint32_t m8 = 2 * w + r % 2;
    const uint32_t x = c_base + n * 128 + 16 * m8;
    void* row = lm + (x ^ (((x >> 7) & 7) << 4));
    stmatrix_x4_trans(row, frag_C.bits(4 * c), frag_C.bits(4 * c + 1), frag_C.bits(4 * c + 2),
                      frag_C.bits(4 * c + 3));
  });
  cuda::ptx::fence_proxy_async();
}

template <int TILE_M_T, int TILE_N_T, int TILE_K_T, int BUFFER_SLOTS_T, int BUFFER_SLOTS_B_T, int CHUNK_SIZE_A_T,
          int CHUNK_SIZE_B_T, int CHUNK_SIZE_C_T>
__device__ void gemv_multicast(kernel_arg_t* __UNIFORM__ arg) {
  VX_CUDA_KERNEL_ENTRY(gemv_multicast);
  const CUtensorMap* a_tensor_map = &arg->a_map;
  const CUtensorMap* b_tensor_map = &arg->b_map;
  const CUtensorMap* c_tensor_map = &arg->c_map;
  const int M = arg->M;
  const int N = arg->N;
  const int K = arg->K;
  const int num_blocks = arg->num_blocks;
  const int sms_per_row = arg->sms_per_row;
  uint64_t* const desc = arg->desc ? (uint64_t*)arg->desc + (size_t)blockIdx.x * dak::DESC_PER_CTA : nullptr;

  int sm_id = blockIdx.x;
  if (sm_id >= num_blocks) {
    return;
  }

  if (threadIdx.x == 0) {
    ((uint32_t*)arg->smid)[blockIdx.x] = __smid();
    ((uint32_t*)arg->wid0)[blockIdx.x] = (uint32_t)vx_warp_id();
  }

  char* const lm = (char*)__local_mem();
  half_t* shared_mem = (half_t*)(lm + dak::LM_DYN);
  half_t* buffer_a = shared_mem;
  half_t* buffer_a_2 = buffer_a + BUFFER_SLOTS_T * CHUNK_SIZE_A_T;
  half_t* buffer_b = buffer_a_2 + BUFFER_SLOTS_T * CHUNK_SIZE_A_T;
  half_t* buffer_C = buffer_b + BUFFER_SLOTS_B_T * CHUNK_SIZE_B_T;
  half_t* buffer_C_2 = buffer_C + CHUNK_SIZE_C_T;

  block_barrier* filled = (block_barrier*)(lm + dak::LM_FILLED);
  block_barrier* drained = (block_barrier*)(lm + dak::LM_DRAINED);
  block_barrier* filled_b = (block_barrier*)(lm + dak::LM_FILLED_B);
  block_barrier* drained_b = (block_barrier*)(lm + dak::LM_DRAINED_B);
  ClusterBarrier::ValueType* cluster_drained_a = (ClusterBarrier::ValueType*)(lm + dak::LM_CLUSTER_DRAINED_A);

  // A barriers
  if (threadIdx.x < BUFFER_SLOTS_T) {
    init(&filled[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    init(&drained[threadIdx.x], NUM_CONSUMER_THREADS + 1);
  }

  if (threadIdx.x >= BUFFER_SLOTS_T && threadIdx.x < BUFFER_SLOTS_T * 2) {
    init(&filled[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    init(&drained[threadIdx.x], NUM_CONSUMER_THREADS + 1);
  }

  // B barriers; group 1's filled_b also takes one relay per group-0 warpgroup.
  if (threadIdx.x < BUFFER_SLOTS_B_T) {
    init(&filled_b[threadIdx.x], NUM_CONSUMER_THREADS + 1);
    init(&drained_b[threadIdx.x], NUM_CONSUMER_THREADS * 2 + 1);
  }

  if (threadIdx.x >= BUFFER_SLOTS_B_T && threadIdx.x < BUFFER_SLOTS_B_T * 2) {
    init(&filled_b[threadIdx.x], NUM_CONSUMER_THREADS + WGS_PER_GROUP);
  }

  // One arrival per warpgroup of the group, from each block of the cluster.
  if (threadIdx.x < BUFFER_SLOTS_T * 2) {
    ClusterBarrier::init(&cluster_drained_a[threadIdx.x], CLUSTER_SIZE * WGS_PER_GROUP);
  }

  __syncthreads();
  dak::cluster_sync();

  int num_tiles_k = (K + TILE_K_T - 1) / TILE_K_T;
  int num_tiles_m = (M + TILE_M_T - 1) / TILE_M_T / 2;

  int cluster_rank = dak::block_rank_in_cluster();
  constexpr uint16_t tma_mcast_mask = (uint16_t(1) << CLUSTER_SIZE) - 1;

  int blockId = blockIdx.x;
  int num_sms_per_m_tile_row = N / TILE_N_T;
  int tile_in = blockId % num_sms_per_m_tile_row;
  int num_copy_blocks = gridDim.x / (num_sms_per_m_tile_row * sms_per_row);
  blockId /= (num_sms_per_m_tile_row * sms_per_row);

  int tile_ik_start = blockIdx.x % sms_per_row;

  if (threadIdx.x == NUM_CONSUMER_THREADS * 2) { // producer 0: A0 + B
    uint32_t cluster_drain_phase[BUFFER_SLOTS_T] = {};
    int nbuff = 0;
    for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
      for (int tile_ik = tile_ik_start; tile_ik < num_tiles_k; tile_ik += sms_per_row) {
        int next_slot = nbuff % BUFFER_SLOTS_T;
        int next_slot_b = nbuff % BUFFER_SLOTS_B_T;
        drained[next_slot].arrive_and_wait();

        if (cluster_rank == 0) {
          ClusterBarrier::wait(&cluster_drained_a[next_slot], cluster_drain_phase[next_slot]);
          cluster_drain_phase[next_slot] ^= 1;
        }

        (void)cuda::device::barrier_arrive_tx(filled[next_slot], 1, (CHUNK_SIZE_A_T) * sizeof(half_t));

        if (cluster_rank == 0) {
          half_t* dst_ptr = buffer_a + next_slot * CHUNK_SIZE_A_T;
          const int32_t coords[4] = {0, 0, 2 * tile_im * (TILE_M_T / 64), tile_ik * (TILE_K_T / 8)};
          cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster, cuda::ptx::space_global, dst_ptr, a_tensor_map,
                                          coords, cuda::device::barrier_native_handle(filled[next_slot]),
                                          tma_mcast_mask);
        }

        drained_b[next_slot_b].arrive_and_wait();
        (void)cuda::device::barrier_arrive_tx(filled_b[next_slot_b], 1, (CHUNK_SIZE_B_T) * sizeof(half_t));

        half_t* dst_ptr_b = buffer_b + next_slot_b * CHUNK_SIZE_B_T;
        cde::cp_async_bulk_tensor_3d_global_to_shared(dst_ptr_b, b_tensor_map, 0, tile_in * TILE_N_T,
                                                      tile_ik * (TILE_K_T / 64), filled_b[next_slot_b]);

        nbuff++;
      }
    }
  } else if (threadIdx.x == NUM_CONSUMER_THREADS * 2 + NUM_PRODUCER_THREADS) { // producer 1: A1
    uint32_t cluster_drain_phase[BUFFER_SLOTS_T] = {};
    int nbuff = 0;
    for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
      for (int tile_ik = tile_ik_start; tile_ik < num_tiles_k; tile_ik += sms_per_row) {
        int next_slot = nbuff % BUFFER_SLOTS_T;

        drained[next_slot + BUFFER_SLOTS_T].arrive_and_wait();

        if (cluster_rank == 0) {
          ClusterBarrier::wait(&cluster_drained_a[next_slot + BUFFER_SLOTS_T], cluster_drain_phase[next_slot]);
          cluster_drain_phase[next_slot] ^= 1;
        }

        (void)cuda::device::barrier_arrive_tx(filled[next_slot + BUFFER_SLOTS_T], 1,
                                              (CHUNK_SIZE_A_T) * sizeof(half_t));

        if (cluster_rank == 0) {
          half_t* dst_ptr = buffer_a_2 + next_slot * CHUNK_SIZE_A_T;
          const int32_t coords[4] = {0, 0, (2 * tile_im + 1) * (TILE_M_T / 64), tile_ik * (TILE_K_T / 8)};
          cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster, cuda::ptx::space_global, dst_ptr, a_tensor_map,
                                          coords,
                                          cuda::device::barrier_native_handle(filled[next_slot + BUFFER_SLOTS_T]),
                                          tma_mcast_mask);
        }

        nbuff++;
      }
    }
  } else if (threadIdx.x < NUM_CONSUMER_THREADS) {
#pragma unroll
    for (int i = 0; i < BUFFER_SLOTS_T; i++) {
      (void)drained[i].arrive();
      if (threadIdx.x % WG_THREADS == 0) {
        ClusterBarrier::arrive(&cluster_drained_a[i], 0, 1);
      }
    }

#pragma unroll
    for (int i = 0; i < BUFFER_SLOTS_B_T; i++) {
      (void)drained_b[i].arrive();
    }

    int nbuf = 0;
    for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
      task_wgmma_local<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T,
                       0>(buffer_a, buffer_b, buffer_C, nbuf, tile_ik_start, sms_per_row, num_tiles_k, filled,
                          drained, filled_b, drained_b, cluster_drained_a, desc);
      dak::sync_barrier<13, NUM_CONSUMER_THREADS>();   // every sC write before the TMA store
      if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_4d_shared_to_global(c_tensor_map, 0, 0, (tile_im * 2) * TILE_M_T / 64,
                                                      tile_in * (TILE_N_T / 8), buffer_C);
        cde::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      }
      dak::sync_barrier<13, NUM_CONSUMER_THREADS>();   // sC rewritten only after the store read it
    }
  } else if (threadIdx.x < 2 * NUM_CONSUMER_THREADS) {
#pragma unroll
    for (int i = 0; i < BUFFER_SLOTS_T; i++) {
      (void)drained[i + BUFFER_SLOTS_T].arrive();
      if (threadIdx.x % WG_THREADS == 0) {
        ClusterBarrier::arrive(&cluster_drained_a[i + BUFFER_SLOTS_T], 0, 1);
      }
    }

#pragma unroll
    for (int i = 0; i < BUFFER_SLOTS_B_T; i++) {
      (void)drained_b[i].arrive();
    }

    int tid = threadIdx.x - NUM_CONSUMER_THREADS;

    int nbuf = 0;
    for (int tile_im = blockId; tile_im < num_tiles_m; tile_im += num_copy_blocks) {
      task_wgmma_local<TILE_M_T, TILE_N_T, TILE_K_T, BUFFER_SLOTS_T, BUFFER_SLOTS_B_T, CHUNK_SIZE_A_T, CHUNK_SIZE_B_T,
                       1>(buffer_a_2, buffer_b, buffer_C_2, nbuf, tile_ik_start, sms_per_row, num_tiles_k,
                          filled + BUFFER_SLOTS_T, drained + BUFFER_SLOTS_T, filled_b, drained_b,
                          cluster_drained_a + BUFFER_SLOTS_T, desc);
      dak::sync_barrier<15, NUM_CONSUMER_THREADS>();
      if (tid == 0) {
        cde::cp_async_bulk_tensor_4d_shared_to_global(c_tensor_map, 0, 0, (tile_im * 2 + 1) * TILE_M_T / 64,
                                                      tile_in * (TILE_N_T / 8), buffer_C_2);
        cde::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      }
      dak::sync_barrier<15, NUM_CONSUMER_THREADS>();
    }
  }
  // No block of the cluster exits while its peer may still arrive on its mbarriers.
  __syncwarp();
  dak::cluster_sync();
}

VX_CUDA_KERNEL_EXPORT(gemv_multicast, gemv_multicast<dak::TILE_M, dak::TILE_N, dak::TILE_K, dak::BUFFER_SLOTS,
                                                     dak::BUFFER_SLOTS_B, dak::CHUNK_SIZE_A, dak::CHUNK_SIZE_B,
                                                     dak::CHUNK_SIZE_C>)
