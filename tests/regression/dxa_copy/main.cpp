#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>

#ifdef VX_CFG_EXT_DXA_ENABLE
#include <dxa.h>
#endif

#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t ndim = 2;
uint32_t elect = 0;
bool mbar_mode = false;
bool oob = false;
bool canary = false;

// -o: the out-of-bounds fill; distinct bytes, so a misplaced byte shows.
constexpr uint32_t kCfill = 0x5AA5C33Cu;

// Default sizes/tiles per dimension (overridable via CLI).
uint32_t sizes[DXA_MAX_DIMS] = {32, 32, 1, 1, 1};
uint32_t tiles[DXA_MAX_DIMS] = {4, 4, 1, 1, 1};

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h err_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};
ext_arg_t ext_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-d1|-d2|-d3|-d4|-d5] [-s<d> size] [-t<d> tile] [-e] [-m] [-o] [-c] [-h]\n";
  std::cout << "  -d1..-d5: set number of dimensions (default: -d2)\n";
  std::cout << "  -s0 N:    set size of dimension 0 (innermost)\n";
  std::cout << "  -t0 N:    set tile size of dimension 0\n";
  std::cout << "  -e:       issue the copy from one elected thread instead of the whole warp\n";
  std::cout << "  -m:       complete the copy on an mbarrier (complete_tx of the box bytes)\n";
  std::cout << "  -o:       the descriptor ends half a tile early along an outer dimension, so the\n"
               "            last tiles are partly out of bounds and filled with the descriptor's cfill\n";
  std::cout << "  -c:       canary words follow the tile in local memory; the copy must leave them\n";
}

static void set_defaults() {
  // Set reasonable defaults based on ndim.
  switch (ndim) {
  case 1: sizes[0] = 128; tiles[0] = 16; break;
  case 2: sizes[0] = 32; sizes[1] = 32; tiles[0] = 4; tiles[1] = 4; break;
  case 3: sizes[0] = 8; sizes[1] = 8; sizes[2] = 8; tiles[0] = 2; tiles[1] = 2; tiles[2] = 2; break;
  case 4: sizes[0] = 4; sizes[1] = 4; sizes[2] = 4; sizes[3] = 4;
          tiles[0] = 2; tiles[1] = 2; tiles[2] = 2; tiles[3] = 2; break;
  case 5: sizes[0] = 4; sizes[1] = 4; sizes[2] = 4; sizes[3] = 2; sizes[4] = 2;
          tiles[0] = 2; tiles[1] = 2; tiles[2] = 2; tiles[3] = 2; tiles[4] = 1; break;
  }
  // Unused dims = 1.
  for (uint32_t d = ndim; d < DXA_MAX_DIMS; ++d) {
    sizes[d] = 1;
    tiles[d] = 1;
  }
}

static void parse_args(int argc, char** argv) {
  // First pass: find -d flag.
  for (int i = 1; i < argc; ++i) {
    if (strlen(argv[i]) == 3 && argv[i][0] == '-' && argv[i][1] == 'd'
        && argv[i][2] >= '1' && argv[i][2] <= '5') {
      ndim = argv[i][2] - '0';
    }
  }
  set_defaults();

  // Second pass: manual parse for all flags (getopt doesn't handle -d1, -s0 N, -t0 N well).
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') continue;
    if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) { kernel_file = argv[++i]; continue; }
    if (strcmp(argv[i], "-h") == 0) { show_usage(); exit(0); }
    if (strcmp(argv[i], "-e") == 0) { elect = 1; continue; }
    if (strcmp(argv[i], "-m") == 0) { mbar_mode = true; continue; }
    if (strcmp(argv[i], "-o") == 0) { oob = true; continue; }
    if (strcmp(argv[i], "-c") == 0) { canary = true; continue; }
    // -s<d> N and -t<d> N
    if (strlen(argv[i]) == 3 && i + 1 < argc) {
      int d = argv[i][2] - '0';
      if (d >= 0 && d < (int)DXA_MAX_DIMS) {
        if (argv[i][1] == 's') { sizes[d] = atoi(argv[++i]); continue; }
        if (argv[i][1] == 't') { tiles[d] = atoi(argv[++i]); continue; }
      }
    }
  }
}

void cleanup() {
  if (device) {
    if (src_buffer) vx_buffer_release(src_buffer);
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (err_buffer) vx_buffer_release(err_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  // Validate sizes are divisible by tiles.
  uint32_t total_elems = 1;
  uint32_t group_size = 1;
  uint32_t total_groups = 1;
  for (uint32_t d = 0; d < ndim; ++d) {
    if (tiles[d] == 0 || sizes[d] == 0) {
      std::cout << "Error: size and tile must be > 0 for dim " << d << "\n";
      return -1;
    }
    if ((sizes[d] % tiles[d]) != 0) {
      std::cout << "Error: size[" << d << "]=" << sizes[d]
                << " must be divisible by tile[" << d << "]=" << tiles[d] << "\n";
      return -1;
    }
    total_elems *= sizes[d];
    group_size *= tiles[d];
    total_groups *= sizes[d] / tiles[d];
  }

  // -o: the outermost dimension whose tile spans at least 2 elements. The
  // engines clip only the outer dimensions at element granularity.
  uint32_t oob_dim = 0;
  uint32_t desc_sizes[DXA_MAX_DIMS];
  for (uint32_t d = 0; d < DXA_MAX_DIMS; ++d) desc_sizes[d] = sizes[d];
  if (oob) {
    for (uint32_t d = 1; d < ndim; ++d) {
      if (tiles[d] >= 2) oob_dim = d;
    }
    if (oob_dim == 0) {
      std::cout << "Error: -o needs a tile of at least 2 along an outer dimension\n";
      return -1;
    }
    desc_sizes[oob_dim] = sizes[oob_dim] - tiles[oob_dim] / 2;
  }
  const bool ext = mbar_mode || canary;
#ifndef VX_CFG_EXT_MBAR_ENABLE
  if (mbar_mode) {
    std::cout << "Error: -m needs a device built with VX_CFG_EXT_MBAR_ENABLE\n";
    return -1;
  }
#endif

  std::srand(42);

#ifdef VX_CFG_EXT_DXA_ENABLE
  std::cout << "mode: DXA\n";
#else
  std::cout << "mode: LSU\n";
#endif
  if (elect)
    std::cout << "issue: elected thread\n";
  if (mbar_mode)
    std::cout << "completion: mbarrier\n";
  if (oob)
    std::cout << "oob: dim " << oob_dim << " ends at " << desc_sizes[oob_dim] << " of " << sizes[oob_dim] << "\n";
  if (canary)
    std::cout << "canaries: " << CANARY_WORDS << " words after the tile\n";
  std::cout << "ndim: " << ndim << "\n";
  std::cout << "sizes:";
  for (uint32_t d = 0; d < ndim; ++d) std::cout << " " << sizes[d];
  std::cout << "\ntiles:";
  for (uint32_t d = 0; d < ndim; ++d) std::cout << " " << tiles[d];
  std::cout << "\ntotal_elems: " << total_elems << ", groups: " << total_groups << "\n";

  const uint32_t buf_size = total_elems * sizeof(TYPE);
  const uint32_t local_mem = ext ? TILE_OFF + group_size * sizeof(TYPE) + (canary ? CANARY_WORDS * 4 : 0)
                                 : group_size * sizeof(TYPE);

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

#ifdef VX_CFG_EXT_DXA_ENABLE
  uint64_t isa_flags = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_DXA) == 0) {
    std::cerr << "Error: DXA ISA extension is disabled.\n";
    cleanup();
    return -1;
  }
#endif

  RT_CHECK(vx_check_occupancy(device, group_size, local_mem));

  // Set up kernel args.
  kernel_arg.ndim = ndim;
  kernel_arg.elect = elect;
  for (uint32_t d = 0; d < DXA_MAX_DIMS; ++d) {
    kernel_arg.sizes[d] = sizes[d];
    kernel_arg.tiles[d] = tiles[d];
    kernel_arg.grids[d] = (d < ndim) ? sizes[d] / tiles[d] : 1;
  }

  // Allocate and fill source array.
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));

  std::vector<TYPE> h_src(total_elems);
  for (uint32_t i = 0; i < total_elems; ++i)
    h_src[i] = static_cast<TYPE>(i + 1);
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), buf_size, 0, nullptr, nullptr));

  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));
  std::vector<TYPE> h_dst(total_elems, static_cast<TYPE>(0));
  RT_CHECK(vx_enqueue_write(queue, dst_buffer, 0, h_dst.data(), buf_size, 0, nullptr, nullptr));

#ifdef VX_CFG_EXT_DXA_ENABLE
  // Program DXA descriptor for N-D source tile.
  constexpr uint32_t kDescSrc = 0;

  // Compute byte strides for row-major layout.
  // stride[d] = product(sizes[0..d]) * sizeof(TYPE)
  uint32_t byte_strides[4] = {};
  uint32_t stride_acc = sizes[0] * sizeof(TYPE);
  for (uint32_t d = 0; d < 4 && d + 1 < ndim; ++d) {
    byte_strides[d] = stride_acc;
    stride_acc *= sizes[d + 1];
  }

  switch (ndim) {
  case 1:
    RT_CHECK(vortex::dxa::program_1d(device, kDescSrc, kernel_arg.src_addr,
      desc_sizes[0], tiles[0], sizeof(TYPE)));
    break;
  case 2:
    RT_CHECK(vortex::dxa::program_2d(device, kDescSrc, kernel_arg.src_addr,
      desc_sizes[0], desc_sizes[1], byte_strides[0], tiles[0], tiles[1], sizeof(TYPE)));
    break;
  case 3:
    RT_CHECK(vortex::dxa::program_3d(device, kDescSrc, kernel_arg.src_addr,
      desc_sizes[0], desc_sizes[1], desc_sizes[2],
      byte_strides[0], byte_strides[1],
      tiles[0], tiles[1], tiles[2], sizeof(TYPE)));
    break;
  case 4:
    RT_CHECK(vortex::dxa::program_4d(device, kDescSrc, kernel_arg.src_addr,
      desc_sizes[0], desc_sizes[1], desc_sizes[2], desc_sizes[3],
      byte_strides[0], byte_strides[1], byte_strides[2],
      tiles[0], tiles[1], tiles[2], tiles[3], sizeof(TYPE)));
    break;
  case 5:
    RT_CHECK(vortex::dxa::program_5d(device, kDescSrc, kernel_arg.src_addr,
      desc_sizes[0], desc_sizes[1], desc_sizes[2], desc_sizes[3], desc_sizes[4],
      byte_strides[0], byte_strides[1], byte_strides[2], byte_strides[3],
      tiles[0], tiles[1], tiles[2], tiles[3], tiles[4], sizeof(TYPE)));
    break;
  }
  if (oob) {
    RT_CHECK(vx_dcr_write(device, VX_DCR_DXA_DESC_BASE + kDescSrc * VX_DCR_DXA_DESC_STRIDE
                                  + VX_DCR_DXA_DESC_CFILL_OFF, kCfill));
  }
#endif

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, ext ? "copy_ext" : "main", &kernel));

  std::vector<uint32_t> h_err;
  if (ext) {
    const uint32_t threads = total_groups * group_size;
    h_err.assign(threads, ~0u);
    RT_CHECK(vx_buffer_create(device, threads * sizeof(uint32_t), VX_MEM_READ_WRITE, &err_buffer));
    RT_CHECK(vx_enqueue_write(queue, err_buffer, 0, h_err.data(), threads * sizeof(uint32_t), 0, nullptr, nullptr));
    ext_arg.base = kernel_arg;
    RT_CHECK(vx_buffer_address(err_buffer, &ext_arg.err_addr));
    ext_arg.mbar         = mbar_mode;
    ext_arg.canary_words = canary ? CANARY_WORDS : 0;
    ext_arg.tx_bytes     = group_size * sizeof(TYPE);
  }

  // Launch with flattened 1D grid; kernel decomposes flat group ID.
  uint32_t grid_dim[1] = {total_groups};
  uint32_t block_dim[1] = {group_size};

  std::cout << "start\n";
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = ext ? (const void*)&ext_arg : (const void*)&kernel_arg;
    li.args_size    = ext ? sizeof(ext_arg) : sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid_dim[0];
    li.block_dim[0] = block_dim[0];
    li.lmem_size    = local_mem;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev));
  vx_event_h err_ev = nullptr;
  if (ext) {
    RT_CHECK(vx_enqueue_read(queue, h_err.data(), err_buffer, 0, h_err.size() * sizeof(uint32_t), 1, &launch_ev, &err_ev));
    RT_CHECK(vx_event_wait_value(err_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(err_ev);
  }
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  uint32_t errors = 0;
  if (ext) {
    uint32_t waits = 0, canaries = 0, unfinished = 0;
    for (uint32_t v : h_err) {
      if (v == ~0u) { ++unfinished; continue; }
      waits    += (v & ERR_WAIT) != 0;
      canaries += (v & ERR_CANARY) != 0;
    }
    if (waits)
      std::cout << "*** error: mbarrier phase did not complete (" << waits << " threads gave up)\n";
    if (canaries)
      std::cout << "*** error: " << canaries << " threads found a damaged canary after the tile\n";
    if (unfinished)
      std::cout << "*** error: " << unfinished << " threads did not finish\n";
    errors += waits + canaries + unfinished;
  }

  // Every element belongs to exactly one tile, so the written-back tiles
  // must reproduce the whole source array; with -o the elements past the
  // descriptor's end must hold the fill, byte for byte.
  uint32_t outer_stride = 1;
  for (uint32_t d = 0; d < oob_dim; ++d) outer_stride *= sizes[d];
  uint8_t fill[sizeof(TYPE)];
  for (uint32_t b = 0; b < sizeof(TYPE); ++b)
    fill[b] = uint8_t(kCfill >> ((b & 3) * 8));
  uint32_t mismatches = 0, filled = 0;
  for (uint32_t i = 0; i < total_elems; ++i) {
    const bool in_fill = oob && (i / outer_stride) % sizes[oob_dim] >= desc_sizes[oob_dim];
    const void* want = in_fill ? (const void*)fill : (const void*)&h_src[i];
    filled += in_fill;
    if (std::memcmp(&h_dst[i], want, sizeof(TYPE)) != 0) {
      if (mismatches < 10) {
        std::cout << "*** error: [" << i << "] expected=";
        if (in_fill)
          std::cout << "fill 0x" << std::hex << kCfill << std::dec;
        else
          std::cout << h_src[i];
        std::cout << ", actual=" << h_dst[i] << "\n";
      }
      ++mismatches;
    }
  }
  if (oob)
    std::cout << filled << " elements out of bounds\n";
  errors += mismatches;
  if (errors != 0) {
    std::cout << "Found " << errors << " errors!\nFAILED!\n";
    cleanup();
    return errors;
  }

  std::cout << "PASSED\n";

  cleanup();
  return 0;
}
