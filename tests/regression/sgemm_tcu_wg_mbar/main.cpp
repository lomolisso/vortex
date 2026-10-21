// sgemm_tcu_wg_mbar: a CUTLASS-style multi-stage WGMMA mainloop over
// mbarriers (see kernel.cpp): a DXA producer warp fills S stages of {A, B}
// slabs guarded by FULL and EMPTY mbarriers; the ISSUE_WIDTH consumer warps
// (one WGMMA group) wait on FULL, run the wgmma, commit it as a group, wait
// until at most one group is pending and arrive on the previous stage's EMPTY.
// The result is checked against the CPU. RV64, on a DLMEM device with the A
// extension, mbarriers, DXA and the WGMMA tensor core.
//
//   -m wait    wait_group<1> before each EMPTY arrive (default)
//   -m nowait  no groups: EMPTY is arrived on right after the wgmma issues,
//              so a slab can be refilled before the wgmma has read it
//   -m lat     wait, the last k-tile with wait_group<0>, plus cycle probes,
//              printed: from the last wgmma's issue to wait_group<0>
//              returning, then reading eight accumulators right after it and
//              again (the reads share the FPU with the other warps, so they
//              are reported, not checked)
//   -m groups  wait, k-tile 0 also waiting on its group at once, on a group
//              of BIG_GROUP wgmmas and with wait_group<2> on three groups that
//              share a boundary; the added wgmmas read a zeroed slab
//   -m refill  wait_group<0> on each k-tile's group, then an arrive on its
//              own stage's EMPTY, so -s 1 refills one stage in place
//   -e E       refill: each consumer then runs E wgmmas on the zeroed slab,
//              the later consumers starting each k-tile a little later, so
//              their wgmmas overlap each other and the refill (default 0)
//   -s S       stages, >= 2 (default 3); refill also takes 1
//   -p         parity: no spin, the warps meet in a register barrier where
//              they would wait, and every phase must then be complete
//   -M M, -n N, -k K  matrix sizes (default 64 each)

#include "common.h"
#include <algorithm>
#include <chrono>
#include <type_traits>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <string.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex2.h>
#include <dxa.h>

// The TCU chains K/tileK dot-product ops, each rounding its accumulator to fp32
// once, while the CPU reference accumulates in double (no intermediate rounding).
// A wider tile rounds fewer times but sums more elements per rounding, so the two
// effects cancel and the drift tracks K rather than the chain depth: it measures
// ~K/8 ULP across every gated NUM_THREADS (16 ULP at K=128 for both tileK=8 and
// tileK=16). float_ulp() therefore scales with K and carries 2x headroom, since a
// fixed bound silently depends on the tile shape. The RTL datapath itself is
// verified bit-exactly against its windowed reference by hw/unittest/tcu_fedp
// (`make -C hw/unittest run-tcu`), so this bounds a reference idealization, not a
// hardware tolerance.
static uint32_t g_float_ulp = 16;
static uint32_t g_max_ulp = 0;

static uint32_t float_ulp(uint32_t K) {
  return std::max<uint32_t>(8, K / 5);
}
#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

using namespace vortex;
namespace vt = tensor;

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<vt::fp32> {
public:
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    g_max_ulp = std::max(g_max_ulp, static_cast<uint32_t>(d));
    if (d > static_cast<int32_t>(g_float_ulp)) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, fb.f, fa.f);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::fp16> {
public:
  static uint16_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftoh_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
  }
  static bool compare(uint16_t a, uint16_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::int8> {
public:
  static int8_t generate() {
    return static_cast<int8_t>(rand());
  }
  static bool compare(int8_t a, int8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::int32> {
public:
  static int32_t generate() {
    return static_cast<int32_t>(rand());
  }
  static bool compare(int32_t a, int32_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::tf32> {
public:
  static uint32_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftotf32_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
  }
  static bool compare(uint32_t a, uint32_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename S, typename D>
struct muladd_t {
  using stype = typename S::dtype;
  using dtype = typename D::dtype;
  static dtype eval(stype a, stype b, dtype c) {
    return static_cast<dtype>(a) * static_cast<dtype>(b) + c;
  }
};

template <>
struct muladd_t<vt::fp16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::tf32, vt::fp32> {
  static float eval(uint32_t a, uint32_t b, float c) {
    auto fa = bit_cast<float>(rv_tf32tof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_tf32tof_s(b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::tf32, vt::tf32> {
  static uint32_t eval(uint32_t a, uint32_t b, uint32_t c) {
    auto fa = bit_cast<float>(rv_tf32tof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_tf32tof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_tf32tof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftotf32_s(bit_cast<uint32_t>(fd), 0, nullptr);
  }
};

///////////////////////////////////////////////////////////////////////////////

using cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;

using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

static void matmul_cpu(otype_t *C, const itype_t *A, const itype_t *B,
                       uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      if constexpr (std::is_same<vt::OTYPE, vt::fp32>::value) {
        // fp32 output: the tensor core accumulates the K products in a wide
        // accumulator and rounds to fp32 once; a per-step-rounded reference
        // drifts by several ULP over K. Each product is exact in fp32, so a
        // double accumulation reproduces the single-rounding dot product.
        double acc = 0.0;
        for (uint32_t k = 0; k < K; ++k) {
          acc += static_cast<double>(muladd_t<vt::ITYPE, vt::OTYPE>::eval(A[m * K + k], B[k * N + n], otype_t(0)));
        }
        C[m * N + n] = static_cast<otype_t>(acc);
      } else {
        // Narrow outputs re-round the accumulator to the output type every
        // step, matching the hardware chain exactly.
        otype_t sum(0);
        for (uint32_t k = 0; k < K; ++k) {
          sum = muladd_t<vt::ITYPE, vt::OTYPE>::eval(A[m * K + k], B[k * N + n], sum);
        }
        C[m * N + n] = sum;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

const char *kernel_file = "kernel.vxbin";

uint32_t xm = 64;
uint32_t xn = 64;
uint32_t xk = 64;
uint32_t mode = MODE_WAIT;
uint32_t stages = 3;
uint32_t extra = 0;
bool parity = false;
// The WGMMA group is ISSUE_WIDTH warps, one per TCU block; the block adds the
// producer warp.
uint32_t warps = 0;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h err_buffer = nullptr;
vx_buffer_h lat_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

static void show_usage() {
  std::cout << "Vortex SGEMM TCU WGMMA mainloop over mbarriers." << std::endl;
  std::cout << "Usage: [-m wait|nowait|lat|groups|refill] [-s stages] [-e extra] [-p] [-M M] [-n N] [-k K] [-h help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:M:n:k:s:e:ph")) != -1) {
    switch (c) {
    case 'm':
      if (!strcmp(optarg, "wait")) mode = MODE_WAIT;
      else if (!strcmp(optarg, "nowait")) mode = MODE_NOWAIT;
      else if (!strcmp(optarg, "lat")) mode = MODE_LAT;
      else if (!strcmp(optarg, "groups")) mode = MODE_GROUPS;
      else if (!strcmp(optarg, "refill")) mode = MODE_REFILL;
      else { show_usage(); exit(-1); }
      break;
    case 'M': xm = atoi(optarg); break;
    case 'n': xn = atoi(optarg); break;
    case 'k': xk = atoi(optarg); break;
    case 's': stages = atoi(optarg); break;
    case 'e': extra = atoi(optarg); break;
    case 'p': parity = true; break;
    case 'h': show_usage(); exit(0); break;
    default:  show_usage(); exit(-1);
    }
  }
  if (optind < argc || stages < (mode == MODE_REFILL ? 1u : 2u) || (extra != 0 && mode != MODE_REFILL)) {
    show_usage();
    exit(-1);
  }
}

void cleanup() {
  if (device) {
    if (A_buffer) vx_buffer_release(A_buffer);
    if (B_buffer) vx_buffer_release(B_buffer);
    if (C_buffer) vx_buffer_release(C_buffer);
    if (err_buffer) vx_buffer_release(err_buffer);
    if (lat_buffer) vx_buffer_release(lat_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Check TCU extension
  uint64_t isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (!(isa_flags & VX_ISA_EXT_TCU)) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  if ((isa_flags & VX_ISA_EXT_DXA) == 0 || (isa_flags & VX_ISA_EXT_DLMEM) == 0
      || (isa_flags & VX_ISA_STD_A) == 0) {
    std::cerr << "Error: the device needs DXA, DLMEM and the A extension." << std::endl;
    cleanup();
    return -1;
  }

  uint64_t NT;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != VX_CFG_NUM_THREADS) {
    std::cout << "Error: device thread count (" << NT
              << ") must match VX_CFG_NUM_THREADS=" << VX_CFG_NUM_THREADS << std::endl;
    return -1;
  }

  uint64_t num_warps;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));

  uint64_t issue_width;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISSUE_WIDTH, &issue_width));
  warps = (uint32_t)issue_width;
  if (warps + 1 > num_warps) {
    std::cout << "Error: the WGMMA group (" << warps
              << " = VX_CFG_ISSUE_WIDTH) and the producer need more warps than a core has ("
              << num_warps << ")" << std::endl;
    return -1;
  }
  uint64_t lmem_cap = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_LOCAL_MEM_SIZE, &lmem_cap));

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;

  g_float_ulp = float_ulp(K);

  uint32_t cta_M = warps * cfg::xtileM;

  if ((M % cta_M) != 0) {
    std::cout << "Error: M (" << M << ") must be a multiple of cta_M=" << cta_M << std::endl;
    return -1;
  }
  if ((N % cfg::xtileN) != 0) {
    std::cout << "Error: N (" << N << ") must be a multiple of tileN=" << cfg::xtileN << std::endl;
    return -1;
  }
  if ((K % cfg::tileK) != 0) {
    std::cout << "Error: K (" << K << ") must be a multiple of tileK=" << cfg::tileK << std::endl;
    return -1;
  }
  if (K / cfg::tileK < stages) {
    std::cout << "Error: K (" << K << ") gives fewer k-tiles than the " << stages << " stages" << std::endl;
    return -1;
  }

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;

  // Grid: one block per CTA output tile. Block: the producer and the group.
  uint32_t grid_dim[2]  = {N / cfg::xtileN, M / cta_M};
  uint32_t block_dim[2] = {(warps + 1) * (uint32_t)NT, 1};
  const uint32_t num_blocks = grid_dim[0] * grid_dim[1];

  // Local memory: FULL[S] and EMPTY[S], then S slab pairs, each on its own
  // lines, then the zeroed slab of the groups and refill modes.
  const uint32_t line = 64;
  auto round_up = [line](uint32_t v) { return (v + line - 1) / line * line; };
  const uint32_t slab_base  = round_up(2 * stages * sizeof(uint64_t));
  const uint32_t slab_bytes = round_up((cta_M * cfg::tileK + cfg::tileK * cfg::xtileN) * sizeof(itype_t));
  const uint32_t zero_base  = slab_base + stages * slab_bytes;
  const bool has_zero = (mode == MODE_GROUPS || mode == MODE_REFILL);
  const uint32_t smem_size  = zero_base + (has_zero ? slab_bytes : 0);
  if (((cta_M * cfg::tileK + cfg::tileK * cfg::xtileN) * sizeof(itype_t)) % NT != 0) {
    std::cout << "Error: a stage's bytes do not split evenly over the producer's lanes" << std::endl;
    return -1;
  }
  if (smem_size > lmem_cap || smem_size > 65536) {
    std::cout << "Error: " << stages << " stages need " << smem_size << " bytes of local memory, the core has "
              << lmem_cap << " (a WGMMA descriptor reaches 64 KB)" << std::endl;
    return -1;
  }

  std::cout << "input type: " << vt::ITYPE::name << ", output type: " << vt::OTYPE::name << std::endl;
  std::cout << "WGMMA tile: M=" << cfg::xtileM << ", N=" << cfg::xtileN << ", K=" << cfg::tileK << std::endl;
  std::cout << "CTA tile: M=" << cta_M << " (warps=" << warps << ")" << std::endl;
  std::cout << "grid: " << grid_dim[0] << "x" << grid_dim[1] << std::endl;
  std::cout << "block: " << block_dim[0] << "x" << block_dim[1] << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "smem: " << smem_size << " bytes, " << stages << " stages, mode "
            << (mode == MODE_NOWAIT ? "nowait" : mode == MODE_LAT ? "lat" : mode == MODE_GROUPS ? "groups"
                : mode == MODE_REFILL ? "refill" : "wait")
            << (parity ? " (parity)" : "") << ", " << extra << " extra wgmmas" << std::endl;
  std::cout << "lmem layout: FULL[" << stages << "], EMPTY[" << stages << "], slab base " << slab_base
            << ", slab bytes " << slab_bytes << std::endl;

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;
  kernel_arg.mode = mode;
  kernel_arg.stages = stages;
  kernel_arg.parity = parity;
  kernel_arg.slab_bytes = slab_bytes;
  kernel_arg.slab_base = slab_base;
  kernel_arg.zero_base = zero_base;
  kernel_arg.extra = extra;

  std::vector<uint32_t> h_err((size_t)num_blocks * (warps + 1) * ERR_WORDS, ~0u);
  std::vector<uint32_t> h_lat((size_t)num_blocks * warps * LAT_WORDS, 0);
  RT_CHECK(vx_buffer_create(device, h_err.size() * sizeof(uint32_t), VX_MEM_READ_WRITE, &err_buffer));
  RT_CHECK(vx_buffer_address(err_buffer, &kernel_arg.err_addr));
  RT_CHECK(vx_enqueue_write(queue, err_buffer, 0, h_err.data(), h_err.size() * sizeof(uint32_t), 0, nullptr, nullptr));
  RT_CHECK(vx_buffer_create(device, h_lat.size() * sizeof(uint32_t), VX_MEM_READ_WRITE, &lat_buffer));
  RT_CHECK(vx_buffer_address(lat_buffer, &kernel_arg.lat_addr));
  RT_CHECK(vx_enqueue_write(queue, lat_buffer, 0, h_lat.data(), h_lat.size() * sizeof(uint32_t), 0, nullptr, nullptr));

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, sizeA * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_buffer_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_buffer_create(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_buffer_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_buffer_create(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_buffer_address(C_buffer, &kernel_arg.C_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::dec << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::dec << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::dec << std::endl;

  // Generate source data
  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A[i] = Comparator<vt::ITYPE>::generate();
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = Comparator<vt::ITYPE>::generate();
  }

  std::cout << "upload source data" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, A_buffer, 0, h_A.data(), sizeA * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, B_buffer, 0, h_B.data(), sizeB * sizeof(itype_t), 0, nullptr, nullptr));

  // Program DXA descriptors.
  // Descriptor A: fetches tileK columns x cta_M rows from A[row, k].
  //   dim0 = K-axis (tile0 = tileK), dim1 = M-axis (tile1 = cta_M)
  //   stride0_bytes = row stride of A = K * sizeof(itype_t)
  RT_CHECK(vortex::dxa::program_2d(device, kDescA, kernel_arg.A_addr,
    /*size0=*/K, /*size1=*/M,
    /*stride0_bytes=*/K * sizeof(itype_t),
    /*tile0=*/cfg::tileK, /*tile1=*/cta_M,
    /*elem_bytes=*/sizeof(itype_t)));

  // Descriptor B: fetches tileN columns x tileK rows from B[k, col].
  //   dim0 = N-axis (tile0 = tileN), dim1 = K-axis (tile1 = tileK)
  //   stride0_bytes = row stride of B = N * sizeof(itype_t)
  //   layout = BLOCK_MAJOR → DXA reads B[K][N] row-major and scatters each
  //            element to the bbuf-native dense block-major destination
  //            (vx_tensor.h::b_blockmajor_idx); set_tile_geometry conveys tcN
  //            and the dense block K extent.
  RT_CHECK(vortex::dxa::program_2d(device, kDescB, kernel_arg.B_addr,
    /*size0=*/N, /*size1=*/K,
    /*stride0_bytes=*/N * sizeof(itype_t),
    /*tile0=*/cfg::xtileN, /*tile1=*/cfg::tileK,
    /*elem_bytes=*/sizeof(itype_t)));
  RT_CHECK(vortex::dxa::set_layout(device, kDescB,
    vortex::dxa::Layout::BlockMajor, /*rank=*/2, /*elem_bytes=*/sizeof(itype_t)));
  RT_CHECK(vortex::dxa::set_tile_geometry(device, kDescB, /*tcN=*/cfg::tcN, /*blk_k=*/cfg::b_blk_k));

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // Host result buffer — must outlive the async read enqueued below.
  std::vector<otype_t> h_C(sizeC);

  auto time_start = std::chrono::high_resolution_clock::now();

  std::cout << "start device" << std::endl;
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = grid_dim[0];
    li.grid_dim[1]  = grid_dim[1];
    li.block_dim[0] = block_dim[0];
    li.block_dim[1] = block_dim[1];
    li.lmem_size    = smem_size;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  // Download result — chained after the launch on the same queue
  std::cout << "download result" << std::endl;
  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t), 1, &launch_ev, &read_ev));
  vx_event_h err_ev = nullptr, lat_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_err.data(), err_buffer, 0, h_err.size() * sizeof(uint32_t), 1, &launch_ev, &err_ev));
  RT_CHECK(vx_enqueue_read(queue, h_lat.data(), lat_buffer, 0, h_lat.size() * sizeof(uint32_t), 1, &launch_ev, &lat_ev));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_event_wait_value(err_ev, 1, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_event_wait_value(lat_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(lat_ev);
  vx_event_release(err_ev);
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<otype_t> h_ref(sizeC);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);
    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<vt::OTYPE>::compare(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  // std::dec: an earlier dump leaves the stream in hex.
  std::cout << std::dec << "reference drift: max=" << g_max_ulp << " ULP, bound="
            << g_float_ulp << " ULP (K=" << K << ")" << std::endl;

  uint32_t timeouts = 0, bad_tokens = 0, unfinished = 0;
  for (size_t w = 0; w < h_err.size() / ERR_WORDS; ++w) {
    if (h_err[w * ERR_WORDS + ERR_TIMEOUT] == ~0u) {
      ++unfinished;
      continue;
    }
    timeouts += h_err[w * ERR_WORDS + ERR_TIMEOUT];
    bad_tokens += h_err[w * ERR_WORDS + ERR_TOKEN];
  }
  std::cout << "mbarriers: " << timeouts << " phases not complete in time, " << bad_tokens
            << " tokens of the wrong phase, " << unfinished << " warps unfinished" << std::endl;
  if (timeouts || bad_tokens || unfinished)
    ++errors;

  if (mode == MODE_LAT) {
    uint32_t wmin = ~0u, wmax = 0, amin = ~0u, amax = 0, rmin = ~0u, rmax = 0;
    for (size_t w = 0; w < h_lat.size() / LAT_WORDS; ++w) {
      const uint32_t* l = &h_lat[w * LAT_WORDS];
      wmin = std::min(wmin, l[LAT_WAIT]);
      wmax = std::max(wmax, l[LAT_WAIT]);
      amin = std::min(amin, l[LAT_AFTER]);
      amax = std::max(amax, l[LAT_AFTER]);
      rmin = std::min(rmin, l[LAT_REF]);
      rmax = std::max(rmax, l[LAT_REF]);
    }
    std::cout << "lat: last wgmma issue to wait_group<0> return " << wmin << ".." << wmax
              << " cycles; accumulator reads after it " << amin << ".." << amax
              << ", again " << rmin << ".." << rmax << std::endl;
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " / " << sizeC << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
