// Host driver for dxa_tma_modes — TMA copy-mode features (Task 4).
//
// Three sub-tests, one launch each, all against descriptor slot 0:
//   estride  — 2D copy with element strides (2,2): dense SMEM tile gathers
//              every-2nd element of the source tensor in both dims.
//   kmajor3d — rank-3 K-major: each 2D (dim0×dim1) slice is transposed into
//              SMEM; dim-2 stacks slices row-major.
//   swizzle  — 2D row-major copy with the 128B XOR bank swizzle: 16-byte
//              units within each 128-byte chunk land at
//              off ^ (((off >> 7) & 7) << 4), tile-relative.
// Each sub-test dumps the raw SMEM image and compares it against a host
// reference model. All three must pass.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>
#include <dxa.h>

#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret) break;                                       \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

static const char* kernel_file = "kernel.vxbin";

static vx_device_h device     = nullptr;
static vx_buffer_h src_buffer = nullptr;
static vx_buffer_h dst_buffer = nullptr;
static vx_queue_h  queue      = nullptr;
static vx_module_h module_    = nullptr;
static vx_kernel_h kernel     = nullptr;

static constexpr uint32_t kDescSlot = 0;
static constexpr uint32_t kMaxTileWords = 256; // 1 KB SMEM image (swizzle mode)

void cleanup() {
  if (device) {
    if (src_buffer) vx_buffer_release(src_buffer);
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

// Run one launch: descriptor already programmed; dump tile_words back.
static int run_and_dump(uint32_t ndim, uint32_t tile_words, uint32_t block_size,
                        uint64_t dst_addr, std::vector<uint32_t>* out) {
  kernel_arg_t ka = {};
  ka.ndim       = ndim;
  ka.tile_words = tile_words;
  ka.block_size = block_size;
  ka.dst_addr   = dst_addr;

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &ka;
  li.args_size    = sizeof(ka);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = block_size;
  li.lmem_size    = tile_words * sizeof(uint32_t);
  int err = vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev);
  if (err) return err;
  out->assign(tile_words, 0);
  err = vx_enqueue_read(queue, out->data(), dst_buffer, 0,
                        tile_words * sizeof(uint32_t), 1, &launch_ev, &read_ev);
  if (err) return err;
  err = vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE);
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  return err;
}

static int compare(const char* name, const std::vector<uint32_t>& exp,
                   const std::vector<uint32_t>& got) {
  int errors = 0;
  for (size_t i = 0; i < exp.size(); ++i) {
    if (got[i] != exp[i]) {
      if (errors < 10)
        printf("*** %s: word=%zu expected=0x%08x actual=0x%08x\n",
               name, i, exp[i], got[i]);
      ++errors;
    }
  }
  printf("%s: %s\n", name, errors ? "FAILED" : "ok");
  return errors;
}

int main(int argc, char* argv[]) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    if (c == 'k') kernel_file = optarg;
    else { std::cout << "Usage: [-k kernel]\n"; exit(c == 'h' ? 0 : -1); }
  }

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags = 0, num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  if ((isa_flags & VX_ISA_EXT_DXA) == 0) {
    std::cerr << "Error: DXA ISA extension is disabled.\n";
    cleanup();
    return -1;
  }
  const uint32_t block_size = (uint32_t)num_threads; // one warp

  RT_CHECK(vx_check_occupancy(device, block_size, kMaxTileWords * sizeof(uint32_t)));

  // Source tensor: 32×32 words covers every sub-test's reach (also viewed
  // as 16×16×4 for the 3D case; both are 1024 words).
  constexpr uint32_t kSrcWords = 32 * 32;
  RT_CHECK(vx_buffer_create(device, kSrcWords * 4, VX_MEM_READ, &src_buffer));
  uint64_t src_addr = 0;
  RT_CHECK(vx_buffer_address(src_buffer, &src_addr));
  std::vector<uint32_t> h_src(kSrcWords);
  for (uint32_t i = 0; i < kSrcWords; ++i)
    h_src[i] = 0xA0000000u + i;
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), kSrcWords * 4, 0, nullptr, nullptr));

  RT_CHECK(vx_buffer_create(device, kMaxTileWords * 4, VX_MEM_WRITE, &dst_buffer));
  uint64_t dst_addr = 0;
  RT_CHECK(vx_buffer_address(dst_buffer, &dst_addr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::vector<uint32_t> got;
  int errors = 0;
  std::cout << "start\n";

  // ── 1. estride: 2D 8×8 tile, element strides (2,2), sizes 32×32 ────────
  {
    RT_CHECK(vortex::dxa::program_2d(device, kDescSlot, src_addr,
      /*size0*/32, /*size1*/32, /*stride0*/32 * 4, /*tile0*/8, /*tile1*/8, 4));
    RT_CHECK(vortex::dxa::set_element_strides(device, kDescSlot, 2, 2));
    RT_CHECK(run_and_dump(2, 8 * 8, block_size, dst_addr, &got));
    std::vector<uint32_t> exp(8 * 8);
    for (uint32_t i1 = 0; i1 < 8; ++i1)
      for (uint32_t i0 = 0; i0 < 8; ++i0)
        exp[i1 * 8 + i0] = h_src[(i1 * 2) * 32 + (i0 * 2)];
    errors += compare("estride(2,2)", exp, got);
  }

  // ── 2. kmajor3d: rank-3 (4×8×2) K-major, sizes 16×16×4 ─────────────────
  {
    RT_CHECK(vortex::dxa::program_3d(device, kDescSlot, src_addr,
      /*sizes*/16, 16, 4, /*stride0*/16 * 4, /*stride1*/16 * 16 * 4,
      /*tiles*/4, 8, 2, 4));
    RT_CHECK(vortex::dxa::set_layout(device, kDescSlot,
      vortex::dxa::Layout::KMajor, 3, 4));
    RT_CHECK(run_and_dump(3, 4 * 8 * 2, block_size, dst_addr, &got));
    std::vector<uint32_t> exp(4 * 8 * 2);
    for (uint32_t i2 = 0; i2 < 2; ++i2)
      for (uint32_t i1 = 0; i1 < 8; ++i1)
        for (uint32_t e0 = 0; e0 < 4; ++e0)
          exp[i2 * (4 * 8) + e0 * 8 + i1] = h_src[e0 + i1 * 16 + i2 * 256];
    errors += compare("kmajor3d(4x8x2)", exp, got);
  }

  // ── 3. swizzle: 2D 32×8 tile (128B rows), XOR swizzle at 32/64/128B ─────
  for (uint32_t mode = 1; mode <= 3; ++mode) {
    RT_CHECK(vortex::dxa::program_2d(device, kDescSlot, src_addr,
      /*size0*/32, /*size1*/8, /*stride0*/32 * 4, /*tile0*/32, /*tile1*/8, 4));
    RT_CHECK(vortex::dxa::set_layout(device, kDescSlot,
      vortex::dxa::Layout::RowMajor, 2, 4,
      static_cast<vortex::dxa::Swizzle>(mode)));
    RT_CHECK(run_and_dump(2, 32 * 8, block_size, dst_addr, &got));
    std::vector<uint32_t> exp(32 * 8);
    const uint32_t bmask = (1u << mode) - 1u;             // Swizzle<B,4,3>
    for (uint32_t i1 = 0; i1 < 8; ++i1) {
      for (uint32_t i0 = 0; i0 < 32; ++i0) {
        uint32_t off = (i1 * 32 + i0) * 4;                // dense tile offset
        uint32_t swz = off ^ (((off >> 7) & bmask) << 4);
        exp[swz / 4] = h_src[i1 * 32 + i0];
      }
    }
    char name[32];
    snprintf(name, sizeof(name), "swizzle%uB(32x8)", 16u << mode);
    errors += compare(name, exp, got);
  }

  cleanup();
  if (errors) {
    std::cout << "FAILED (" << errors << " errors)\n";
    return errors;
  }
  std::cout << "PASSED\n";
  return 0;
}
