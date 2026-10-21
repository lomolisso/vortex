#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include <vx_mbar.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

constexpr uint32_t kDescA   = 0;
constexpr uint32_t kDescB   = 1;
constexpr uint32_t kCopyBar = 1;   // the producer's DXA completions
constexpr uint32_t kMeetBar = 2;   // parity runs: every warp of the block

// A CUTLASS-style warp-specialized mainloop. Warp 0 is the producer; warps
// 1..Q (one per issue slot, the WGMMA group) are the consumers, each owning
// xtileM rows of the block's tile. Local memory holds FULL[S] and EMPTY[S],
// then S stages of an {A, B} slab pair. Stage s of k-tile i = s + S*n:
//   producer: wait EMPTY[s] phase n-1 (all consumers done with k-tile i-S),
//             arrive_tx on FULL[s] with the stage's bytes, copy A and B into
//             the slabs with DXA, then complete_tx those bytes on FULL[s];
//   consumer: wait FULL[s] phase n, wgmma, commit_group, wait_group<1>, and
//             arrive on EMPTY of k-tile i-1, whose wgmma has now read its slabs.
// DXA completes on a register barrier, not on an mbarrier, so the producer
// waits for it and does the complete_tx itself. -m nowait arrives on the
// stage's own EMPTY right after the wgmma issues, with no groups: the producer
// can then overwrite slabs the wgmma has not read yet. -m groups is -m wait
// whose k-tile 0 also waits on a one-wgmma group at once, then on a large
// group, then on three groups sharing one boundary (wait_group<2>); the extra
// wgmmas read a zeroed slab, so the result is unchanged. -m refill waits for each k-tile's
// group with wait_group<0> and arrives on that stage's EMPTY, so one stage is
// refilled in place under the same descriptors; -e E then has each consumer
// run E wgmmas on the zero slab after its arrive, the later consumers
// starting each k-tile a little later, so the consumers' wgmmas overlap each
// other and the refill. Every lane does each
// mbarrier op, with its share of the bytes: a value computed under a lane-0
// branch is not valid in the other lanes, and the DXA issue gathers its
// operands across lanes.

namespace {

#define INL inline __attribute__((always_inline))

// The k-tile's wgmma, from the consumer's A slice (registers or memory) and B.
#if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
#define ISSUE_MMA() do {                                \
    ctx::fragment_a fragA;                              \
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);   \
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);       \
  } while (0)
#else
#define ISSUE_MMA() do {                                \
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t)); \
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);      \
  } while (0)
#endif

struct Ctx {
  uint32_t timeouts = 0, tokens = 0;
  bool parity;

  INL void wait(uint64_t* obj, uint32_t ph) {
    if (parity) {
      timeouts += !vx_mbar_test_wait_parity(obj, ph);
      return;
    }
    uint32_t spins = 0;
    while (!vx_mbar_test_wait_parity(obj, ph)) {
      if (++spins >= SPIN_LIMIT) {
        ++timeouts;
        break;
      }
    }
  }
  INL void token(uint64_t tok, uint32_t ph) {
    tokens += ((uint32_t)(tok >> 63) != (ph & 1));
  }
};

// Cycles to issue a read of each of the first eight accumulator registers,
// timed inside one asm block so nothing else is scheduled between the cycle
// reads; a register a wgmma has not written yet stalls its read. Each read
// rewrites its own register (fsgnj.s), and both timestamps go to registers
// written once beforehand, so no other outstanding write can stall the block.
template <typename Frag>
INL uint32_t read_acc(Frag& f) {
  uint64_t a, b;
  __asm__ volatile ("csrr %1, mcycle\n\tcsrr %0, mcycle\n\t"
                    "fsgnj.s %2, %2, %2\n\tfsgnj.s %3, %3, %3\n\tfsgnj.s %4, %4, %4\n\tfsgnj.s %5, %5, %5\n\t"
                    "fsgnj.s %6, %6, %6\n\tfsgnj.s %7, %7, %7\n\tfsgnj.s %8, %8, %8\n\tfsgnj.s %9, %9, %9\n\t"
                    "csrr %1, mcycle"
                    : "=&r"(a), "=&r"(b),
                      "+f"(f.data[0]), "+f"(f.data[1]), "+f"(f.data[2]), "+f"(f.data[3]),
                      "+f"(f.data[4]), "+f"(f.data[5]), "+f"(f.data[6]), "+f"(f.data[7]));
  return (uint32_t)(b - a);
}

} // namespace

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N;
  const uint32_t K = arg->K;
  const uint32_t S = arg->stages;
  const uint32_t mode = arg->mode;
  const uint32_t warp = get_sub_group_id();
  const uint32_t nwarps = get_num_sub_groups();
  const uint32_t lane = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t Q = nwarps - 1;
  const uint32_t cta_M = Q * ctx::xtileM;
  const uint32_t tile_row = blockIdx.y * cta_M;
  const uint32_t tile_col = blockIdx.x * ctx::xtileN;
  const uint32_t num_k_tiles = K / ctx::tileK;
  const uint32_t a_bytes = cta_M * ctx::tileK * sizeof(ctx::input_t);
  const uint32_t lane_bytes = (a_bytes + ctx::tileK * ctx::xtileN * sizeof(ctx::input_t)) / VX_CFG_NUM_THREADS;
  const uint32_t block = blockIdx.y * gridDim.x + blockIdx.x;

  auto lm = reinterpret_cast<uint8_t*>(__local_mem());
  auto obj = reinterpret_cast<uint64_t*>(lm);
  uint64_t* full  = obj;
  uint64_t* empty = obj + S;
  auto slab_a = [&](uint32_t s) { return reinterpret_cast<ctx::input_t*>(lm + arg->slab_base + s * arg->slab_bytes); };
  auto slab_b = [&](uint32_t s) { return reinterpret_cast<ctx::input_t*>(lm + arg->slab_base + s * arg->slab_bytes + a_bytes); };

  if (threadIdx.x == 0) {
    for (uint32_t s = 0; s < S; ++s) {
      vx_mbar_init(full + s, VX_CFG_NUM_THREADS);
      vx_mbar_init(empty + s, Q * VX_CFG_NUM_THREADS);
    }
  }
  auto zero = reinterpret_cast<uint32_t*>(lm + arg->zero_base);
  if (mode == MODE_GROUPS || mode == MODE_REFILL) {
    for (uint32_t w = threadIdx.x; w < arg->slab_bytes / 4; w += blockDim.x)
      zero[w] = 0;
  }
  __syncthreads();

  Ctx x;
  x.parity = (arg->parity != 0);
  vortex::barrier meet(kMeetBar, nwarps);

  if (warp == 0) {
    vortex::barrier copy_bar(kCopyBar, 1);
    for (uint32_t i = 0; i < num_k_tiles; ++i) {
      const uint32_t s = i % S, n = i / S;
      if (i >= S)
        x.wait(empty + s, n - 1);
      x.token(vx_mbar_arrive_tx(full + s, lane_bytes), n);
      copy_bar.expect_tx(2);
      vx_dxa_issue_2d_wg(kDescA, copy_bar.id(), slab_a(s), i * ctx::tileK, tile_row);
      vx_dxa_issue_2d_wg(kDescB, copy_bar.id(), slab_b(s), tile_col, i * ctx::tileK);
      copy_bar.arrive_and_wait();
      vx_mbar_complete_tx(full + s, lane_bytes);
      if (x.parity) {
        meet.arrive_and_wait();
        meet.arrive_and_wait();
      }
    }
  } else {
    const uint32_t cw = warp - 1;
    ctx::fragment_acc fragC;
    ctx::fill_fragment(fragC, 0);
    uint32_t t0 = 0;
    // A wgmma on the zero slab adds nothing to fragC.
    const auto zero_a = vt::vx_make_smem_desc(reinterpret_cast<ctx::input_t*>(zero) + cw * ctx::xtileM * ctx::tileK,
                                              ctx::tileK * sizeof(ctx::input_t));
    const auto zero_b = vt::vx_make_smem_desc(reinterpret_cast<ctx::input_t*>(zero) + cta_M * ctx::tileK, 0);
    // Its own loop, so no branch separates the wgmma from the wait, which
    // then finds it running and is released by its retire.
    for (uint32_t i = 0; mode == MODE_REFILL && i < num_k_tiles; ++i) {
      const uint32_t s = i % S, n = i / S;
      if (x.parity)
        meet.arrive_and_wait();
      x.wait(full + s, n);
      for (uint32_t d = 0; arg->extra != 0 && d < cw * 16; ++d)
        __asm__ volatile ("nop");
      auto A_warp = slab_a(s) + cw * ctx::xtileM * ctx::tileK;
      auto desc_b = vt::vx_make_smem_desc(slab_b(s), 0);
      ISSUE_MMA();
      ctx::commit_group();
      ctx::wait_group<0>();
      x.token(vx_mbar_arrive(empty + s, 1), n);
      for (uint32_t e = 0; e < arg->extra; ++e)
        ctx::wgmma_sync(fragC, zero_a, zero_b, fragC);
      if (x.parity)
        meet.arrive_and_wait();
    }
    for (uint32_t i = 0; mode != MODE_REFILL && i < num_k_tiles; ++i) {
      const uint32_t s = i % S, n = i / S;
      if (x.parity)
        meet.arrive_and_wait();
      x.wait(full + s, n);
      auto A_warp = slab_a(s) + cw * ctx::xtileM * ctx::tileK;
      auto desc_b = vt::vx_make_smem_desc(slab_b(s), 0);
      ISSUE_MMA();
      if (mode == MODE_LAT)
        t0 = csr_read(VX_CSR_MCYCLE);
      if (mode == MODE_NOWAIT) {
        x.token(vx_mbar_arrive(empty + s, 1), n);
      } else {
        ctx::commit_group();
        if (mode == MODE_GROUPS && i == 0) {
          // Straight-line code, so each wait after the first finds its
          // wgmmas still running (a branch would hold the warp until they
          // are done). The large group retires far past the boundaries
          // before it, which are complete.
          ctx::wait_group<0>();
          ctx::wgmma_sync(fragC, zero_a, zero_b, fragC);
          ctx::commit_group();
          ctx::wait_group<0>();
          #pragma unroll
          for (uint32_t j = 0; j < BIG_GROUP; ++j)
            ctx::wgmma_sync(fragC, zero_a, zero_b, fragC);
          ctx::commit_group();
          ctx::wait_group<0>();
          ctx::wgmma_sync(fragC, zero_a, zero_b, fragC);
          ctx::commit_group();
          ctx::commit_group();
          ctx::commit_group();
          ctx::wait_group<2>();
        }
        if (mode == MODE_LAT && i + 1 == num_k_tiles)
          ctx::wait_group<0>();
        else
          ctx::wait_group<1>();
        if (mode == MODE_LAT) {
          // The probe runs every k-tile, so its code is in the I-cache when the
          // last one, after wait_group<0>, is recorded.
          const uint32_t t1 = csr_read(VX_CSR_MCYCLE);
          const uint32_t after = read_acc(fragC);
          const uint32_t ref = read_acc(fragC);
          if (i + 1 == num_k_tiles && lane == 0) {
            auto lat = reinterpret_cast<uint32_t*>(arg->lat_addr) + (block * Q + cw) * LAT_WORDS;
            lat[LAT_WAIT]  = t1 - t0;
            lat[LAT_AFTER] = after;
            lat[LAT_REF]   = ref;
          }
        }
        if (i != 0)
          x.token(vx_mbar_arrive(empty + (i - 1) % S, 1), (i - 1) / S);
      }
      if (x.parity)
        meet.arrive_and_wait();
    }
    if (mode != MODE_NOWAIT && mode != MODE_REFILL) {
      ctx::wait_group<0>();
      const uint32_t i = num_k_tiles - 1;
      x.token(vx_mbar_arrive(empty + i % S, 1), i / S);
    }
    auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
    ctx::store_matrix_sync(pC + (tile_row + cw * ctx::xtileM) * N + tile_col, fragC, N);
  }
  if (lane == 0) {
    auto err = reinterpret_cast<uint32_t*>(arg->err_addr) + (block * nwarps + warp) * ERR_WORDS;
    err[ERR_TIMEOUT] = x.timeouts;
    err[ERR_TOKEN]   = x.tokens;
  }
}
