// cluster_rank: the cluster-rank CSR, the distributed local memory address map
// and the local-memory window of a DLMEM device (RV64).
//
//   default  three launches in one process, in clusters of K = 1 (none),
//            2 (1x2x1) and 4 (2x1x2), each over a different number of CTAs;
//            every CTA's rank must be its intra-cluster index (x fastest, then
//            y, then z)
//   -k K     one launch in clusters of K x 1 x 1 instead
//   -m       also map an object into every rank and check each address
//            against DLMEM_BASE + rank * local-memory size + offset
//   -l B     also fill and check a B-byte local-memory slice per CTA
//
// A default launch the device cannot hold prints SKIPPED; a requested one fails.

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

namespace {

// Per-CTA local memory of the rank launches: room for the mapped object.
constexpr uint32_t kRankLmemBytes = 256;

const char* kernel_file = "kernel.vxbin";
uint32_t    req_k       = 0;
bool        map_mode    = false;
uint32_t    lmem_bytes  = 0;

struct Caps {
  uint64_t num_cores = 0, num_warps = 0, num_threads = 0;
  uint64_t lmem_size = 0, max_cluster = 0, isa = 0;
};

struct Shape {
  uint32_t ndim;
  uint32_t cluster[3];
  uint32_t grid[3];
  bool     requested;
};

void usage() {
  std::printf("Usage: [-k cluster size] [-m: check the rank address map]"
              " [-l local-memory bytes per CTA] [-h]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:ml:h")) != -1) {
    switch (c) {
      case 'k': req_k      = (uint32_t)std::atoi(optarg); break;
      case 'm': map_mode   = true;                        break;
      case 'l': lmem_bytes = (uint32_t)std::atoi(optarg); break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc) {
    usage();
    std::exit(-1);
  }
}

// rank_k over one launch shape: the mismatches, or -1 when the device cannot
// hold the shape.
int run_rank(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel,
             const Caps& caps, const Shape& s) {
  const uint32_t k = s.cluster[0] * s.cluster[1] * s.cluster[2];
  const uint32_t num_ctas = s.grid[0] * s.grid[1] * s.grid[2];
  std::printf("cluster_rank: K=%u, cluster %ux%ux%u, grid %ux%ux%u (%u CTAs)%s\n",
              k, s.cluster[0], s.cluster[1], s.cluster[2],
              s.grid[0], s.grid[1], s.grid[2], num_ctas,
              map_mode ? ", with the rank address map" : "");
  // A same-core cluster holds K single-warp CTAs on one core.
  if (k > caps.num_warps || (uint64_t)k * kRankLmemBytes > caps.lmem_size) {
    std::printf("%s: clusters of %u need %u warps and %u bytes of local memory"
                " per core; the device has %u and %u\n",
                s.requested ? "Error" : "SKIPPED", k, k, k * kRankLmemBytes,
                (uint32_t)caps.num_warps, (uint32_t)caps.lmem_size);
    return -1;
  }

  const uint64_t rec_bytes = (uint64_t)num_ctas * REC_WORDS * sizeof(uint32_t);
  const uint64_t map_bytes = (uint64_t)num_ctas * k * sizeof(uint64_t);
  vx_buffer_h rec_buf = nullptr, map_buf = nullptr;
  CHECK(vx_buffer_create(dev, rec_bytes, VX_MEM_READ_WRITE, &rec_buf));
  if (map_mode)
    CHECK(vx_buffer_create(dev, map_bytes, VX_MEM_READ_WRITE, &map_buf));

  // A CTA that never ran leaves the fill, which no field it writes can equal.
  std::vector<uint32_t> recs((size_t)num_ctas * REC_WORDS, 0xffffffffu);
  std::vector<uint64_t> maps(map_mode ? (size_t)num_ctas * k : 0, ~0ull);
  CHECK(vx_enqueue_write(q, rec_buf, 0, recs.data(), rec_bytes, 0, nullptr, nullptr));
  if (map_mode)
    CHECK(vx_enqueue_write(q, map_buf, 0, maps.data(), map_bytes, 0, nullptr, nullptr));

  rank_arg_t arg = {};
  CHECK(vx_buffer_address(rec_buf, &arg.rec_addr));
  if (map_mode)
    CHECK(vx_buffer_address(map_buf, &arg.map_addr));
  arg.num_blocks = k;
  arg.lmem_bytes = kRankLmemBytes;

  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = s.ndim;
  li.lmem_size    = kRankLmemBytes;
  li.block_dim[0] = (uint32_t)caps.num_threads;
  li.block_dim[1] = 1;
  li.block_dim[2] = 1;
  for (uint32_t d = 0; d < 3; ++d) {
    li.grid_dim[d] = s.grid[d];
    // The default K=1 launch has no clusters at all.
    li.cluster_dim[d] = (k == 1 && !s.requested) ? 0 : s.cluster[d];
  }

  vx_event_h launch_ev = nullptr, rec_ev = nullptr, map_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, recs.data(), rec_buf, 0, rec_bytes, 1, &launch_ev, &rec_ev));
  if (map_mode)
    CHECK(vx_enqueue_read(q, maps.data(), map_buf, 0, map_bytes, 1, &launch_ev, &map_ev));
  CHECK(vx_event_wait_value(rec_ev, 1, VX_TIMEOUT_INFINITE));
  if (map_mode)
    CHECK(vx_event_wait_value(map_ev, 1, VX_TIMEOUT_INFINITE));

  int errors = 0;
  auto report = [&](uint32_t b, const char* what, uint64_t got, uint64_t want) {
    if (errors < 16)
      std::printf("*** K=%u cta %u (slot %u): %s = %#llx, expected %#llx\n", k, b,
                  recs[(size_t)b * REC_WORDS + REC_CTA_ID], what,
                  (unsigned long long)got, (unsigned long long)want);
    ++errors;
  };
  for (uint32_t z = 0; z < s.grid[2]; ++z) {
    for (uint32_t y = 0; y < s.grid[1]; ++y) {
      for (uint32_t x = 0; x < s.grid[0]; ++x) {
        const uint32_t b = x + s.grid[0] * (y + s.grid[1] * z);
        const uint32_t* rec = &recs[(size_t)b * REC_WORDS];
        if (rec[REC_BLOCK_X] != x || rec[REC_BLOCK_Y] != y || rec[REC_BLOCK_Z] != z) {
          report(b, "blockIdx (never ran?) x", rec[REC_BLOCK_X], x);
          continue;
        }
        const uint32_t rank = (x % s.cluster[0])
            + s.cluster[0] * ((y % s.cluster[1]) + s.cluster[1] * (z % s.cluster[2]));
        if (rec[REC_RANK] != rank)
          report(b, "vx_cluster_block_rank()", rec[REC_RANK], rank);
        if (rec[REC_GROUP_RANK] != rank)
          report(b, "cluster_group::block_rank()", rec[REC_GROUP_RANK], rank);
        if (rec[REC_SPAWN_RANK] != rank)
          report(b, "get_cluster_rank()", rec[REC_SPAWN_RANK], rank);
        if (rec[REC_NUM_BLOCKS] != k)
          report(b, "vx_cluster_num_blocks()", rec[REC_NUM_BLOCKS], k);
        // A same-core cluster's slot window opens at a multiple of K.
        if (rec[REC_CTA_ID] % k != rank)
          report(b, "slot % K", rec[REC_CTA_ID] % k, rank);
        if (!map_mode)
          continue;
        for (uint32_t r = 0; r < k; ++r) {
          const uint64_t got  = maps[(size_t)b * k + r];
          const uint64_t want = (uint64_t)VX_MEM_DLMEM_BASE_ADDR + r * caps.lmem_size
                              + map_offset(b, kRankLmemBytes);
          if (got != want)
            report(b, "map_shared_rank(obj, r) for r in order", got, want);
        }
      }
    }
  }

  if (map_ev)
    vx_event_release(map_ev);
  vx_event_release(rec_ev);
  vx_event_release(launch_ev);
  if (map_buf)
    vx_buffer_release(map_buf);
  vx_buffer_release(rec_buf);
  return errors;
}

// lmem_k over 2 CTAs per core, each a whole core's warps and lmem_bytes of local
// memory: the mismatches, or -1 when the device cannot hold one such CTA.
int run_lmem(vx_device_h dev, vx_queue_h q, vx_kernel_h kernel, const Caps& caps) {
  const uint32_t block = (uint32_t)(caps.num_threads * caps.num_warps);
  const uint32_t num_ctas = 2 * (uint32_t)caps.num_cores;
  std::printf("cluster_rank: local memory, %u CTAs of %u threads, %u bytes each\n",
              num_ctas, block, lmem_bytes);
  if (lmem_bytes % sizeof(uint64_t) != 0 || lmem_bytes > caps.lmem_size) {
    std::printf("Error: -l %u must be a multiple of 8 and at most the device's"
                " local memory, %u bytes\n", lmem_bytes, (uint32_t)caps.lmem_size);
    return -1;
  }

  const uint64_t err_bytes  = (uint64_t)num_ctas * block * 2 * sizeof(uint32_t);
  const uint64_t base_bytes = (uint64_t)num_ctas * sizeof(uint64_t);
  vx_buffer_h err_buf = nullptr, base_buf = nullptr;
  CHECK(vx_buffer_create(dev, err_bytes, VX_MEM_READ_WRITE, &err_buf));
  CHECK(vx_buffer_create(dev, base_bytes, VX_MEM_READ_WRITE, &base_buf));
  std::vector<uint32_t> errs((size_t)num_ctas * block * 2, 0xffffffffu);
  std::vector<uint64_t> bases(num_ctas, 0);
  CHECK(vx_enqueue_write(q, err_buf, 0, errs.data(), err_bytes, 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, base_buf, 0, bases.data(), base_bytes, 0, nullptr, nullptr));

  lmem_arg_t arg = {};
  CHECK(vx_buffer_address(err_buf, &arg.err_addr));
  CHECK(vx_buffer_address(base_buf, &arg.base_addr));
  arg.lmem_bytes = lmem_bytes;

  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = num_ctas;
  li.block_dim[0] = block;
  li.lmem_size    = lmem_bytes;

  vx_event_h launch_ev = nullptr, err_ev = nullptr, base_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, errs.data(), err_buf, 0, err_bytes, 1, &launch_ev, &err_ev));
  CHECK(vx_enqueue_read(q, bases.data(), base_buf, 0, base_bytes, 1, &launch_ev, &base_ev));
  CHECK(vx_event_wait_value(err_ev, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(base_ev, 1, VX_TIMEOUT_INFINITE));

  // The dispatcher rounds a CTA's local memory up to whole memory blocks.
  const uint64_t stride = (lmem_bytes + VX_CFG_MEM_BLOCK_SIZE - 1)
                        / VX_CFG_MEM_BLOCK_SIZE * VX_CFG_MEM_BLOCK_SIZE;
  int errors = 0;
  for (uint32_t b = 0; b < num_ctas; ++b) {
    const uint64_t base = bases[b];
    const uint64_t off  = base - (uint64_t)VX_MEM_LMEM_BASE_ADDR;
    if (base < (uint64_t)VX_MEM_LMEM_BASE_ADDR || off % stride != 0
     || off + lmem_bytes > caps.lmem_size
     || base + lmem_bytes > (uint64_t)VX_MEM_DLMEM_BASE_ADDR) {
      if (errors < 16)
        std::printf("*** cta %u: local memory at %#llx, outside [%#llx, +%#llx)\n", b,
                    (unsigned long long)base, (unsigned long long)VX_MEM_LMEM_BASE_ADDR,
                    (unsigned long long)caps.lmem_size);
      ++errors;
    }
    for (uint32_t t = 0; t < block; ++t) {
      const uint32_t* e = &errs[((size_t)b * block + t) * 2];
      if (e[0] == 0)
        continue;
      if (errors < 16)
        std::printf("*** cta %u thread %u: %u bad words, the first at index %u\n",
                    b, t, e[0], e[1]);
      errors += (e[0] == 0xffffffffu) ? 1 : (int)e[0];
    }
  }
  std::printf("cluster_rank: local memory at %#llx\n", (unsigned long long)bases[0]);

  vx_event_release(base_ev);
  vx_event_release(err_ev);
  vx_event_release(launch_ev);
  vx_buffer_release(base_buf);
  vx_buffer_release(err_buf);
  return errors;
}

}  // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  Caps caps;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,         &caps.num_cores));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,         &caps.num_warps));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &caps.num_threads));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE,    &caps.lmem_size));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &caps.max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &caps.isa));

  int errors = 0;
  if (!(caps.isa & VX_ISA_EXT_DLMEM)
   || caps.max_cluster != (uint64_t)VX_CFG_DLMEM_MAX_CLUSTER) {
    std::printf("*** device caps: DLMEM flag %d, largest cluster %u; expected 1 and %u\n",
                (caps.isa & VX_ISA_EXT_DLMEM) ? 1 : 0, (uint32_t)caps.max_cluster,
                (uint32_t)VX_CFG_DLMEM_MAX_CLUSTER);
    ++errors;
  }

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h rank_kernel = nullptr, lmem_kernel = nullptr;
  CHECK(vx_module_get_kernel(mod, "rank_k", &rank_kernel));
  CHECK(vx_module_get_kernel(mod, "lmem_k", &lmem_kernel));

  const uint32_t nc = (uint32_t)caps.num_cores;
  std::vector<Shape> shapes;
  if (req_k != 0) {
    shapes.push_back({2, {req_k, 1, 1}, {2 * req_k, nc + 1, 1}, true});
  } else {
    // Different CTA counts per launch, each several rounds of the machine.
    const uint32_t slots = nc * (uint32_t)caps.num_warps;
    shapes.push_back({1, {1, 1, 1}, {2 * slots + 1, 1, 1}, false});
    shapes.push_back({2, {1, 2, 1}, {3, 2 * (nc + 1), 1}, false});
    shapes.push_back({3, {2, 1, 2}, {4, nc + 1, 2}, false});
  }
  for (const Shape& s : shapes) {
    const int e = run_rank(dev, q, rank_kernel, caps, s);
    if (e > 0 || (e < 0 && s.requested))
      errors += (e > 0) ? e : 1;
  }
  if (lmem_bytes != 0) {
    const int e = run_lmem(dev, q, lmem_kernel, caps);
    errors += (e < 0) ? 1 : e;
  }

  vx_kernel_release(lmem_kernel);
  vx_kernel_release(rank_kernel);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::printf("Found %d errors!\nFAILED!\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
