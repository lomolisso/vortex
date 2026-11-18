// TMA engine timing of the H100 front end, one block on one core:
//   -m0 SM-wide intake and queue: W warps (lane 0) issue P bulk loads of B bytes,
//       each then waiting for its own copies (14 M7: last done - last issue);
//   -m1 store queue and groups: G groups of S bulk stores of B bytes, wait_group K;
//   -m2 reduce drain: one bulk reduction of B bytes;
//   -m3 done after read: one bulk store, wait_group.read 0, then wait_group 0.
// Prints the timings; engine_check.py holds the pass conditions. Data are checked here.
#include <vortex2.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include "common.h"

#define CHECK(expr)                                                                     \
  do {                                                                                  \
    vx_result_t _r = (expr);                                                            \
    if (_r != VX_SUCCESS) {                                                             \
      std::printf("*** %s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,             \
                  vx_result_string(_r));                                                \
      std::exit(-1);                                                                    \
    }                                                                                   \
  } while (0)

int main(int argc, char** argv) {
  eng_arg_t arg = {};
  arg.mode = M_INTAKE;
  arg.warps = 1;
  arg.copies = 8;
  arg.bytes = 16;
  arg.groups = 1;
  arg.waitk = 0;
  int c;
  while ((c = getopt(argc, argv, "m:w:p:b:g:k:ch")) != -1) {
    switch (c) {
    case 'm': arg.mode = atoi(optarg); break;
    case 'w': arg.warps = atoi(optarg); break;
    case 'p': arg.copies = atoi(optarg); break;
    case 'b': arg.bytes = atoi(optarg); break;
    case 'g': arg.groups = atoi(optarg); break;
    case 'k': arg.waitk = atoi(optarg); break;
    case 'c': arg.cold = 1; break;
    default:
      std::printf("usage: %s [-m mode] [-w warps] [-p copies] [-b bytes] [-g groups] [-k waitK] [-c]\n", argv[0]);
      return -1;
    }
  }
  if (arg.warps < 1 || arg.warps > MAX_WARPS || arg.copies < 1 || arg.copies > MAX_COPIES ||
      arg.groups < 1 || arg.groups > MAX_GROUPS || arg.bytes < 16 || (arg.bytes % 16) != 0) {
    std::printf("*** bad arguments\n");
    return -1;
  }

  vx_device_h dev;
  vx_queue_h q;
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  CHECK(vx_device_open(0, &dev));
  CHECK(vx_queue_create(dev, &qi, &q));
  uint64_t nt = 0, nw = 0, lmem = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));
  CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem));
  const uint32_t warps = (arg.mode == M_INTAKE) ? arg.warps : 1;
  if (warps > nw || LM_DATA + arg.bytes > lmem) {
    std::printf("*** the device has %u warps and %u B of local memory: -w%u -b%u do not fit\n",
                (uint32_t)nw, (uint32_t)lmem, warps, arg.bytes);
    return -1;
  }

  // Consecutive lines, so every copy's warmed source line stays in the L2.
  arg.src_stride = (arg.bytes + 127) / 128 * 128;
  const uint64_t src_bytes = (uint64_t)MAX_WARPS * MAX_COPIES * SRC_STRIDE;
  const uint32_t cold_off = (arg.mode == M_READ && arg.cold) ? COLD_OFFSET(arg.bytes) : 0;
  const uint64_t dst_bytes = (arg.mode == M_STORE) ? (uint64_t)arg.groups * arg.copies * arg.bytes : cold_off + arg.bytes;
  vx_buffer_h gsrc, gdst, obuf;
  CHECK(vx_buffer_create(dev, src_bytes, VX_MEM_READ, &gsrc));
  CHECK(vx_buffer_create(dev, dst_bytes, VX_MEM_READ_WRITE, &gdst));
  CHECK(vx_buffer_create(dev, O_WORDS * 4, VX_MEM_READ_WRITE, &obuf));
  CHECK(vx_buffer_address(gsrc, &arg.gsrc_addr));
  CHECK(vx_buffer_address(gdst, &arg.gdst_addr));
  CHECK(vx_buffer_address(obuf, &arg.out_addr));

  std::vector<uint32_t> h_src(src_bytes / 4), h_dst(dst_bytes / 4), out(O_WORDS, 0);
  for (size_t i = 0; i < h_src.size(); ++i)
    h_src[i] = (uint32_t)(i * 2654435761u);
  for (size_t i = 0; i < h_dst.size(); ++i)
    h_dst[i] = (uint32_t)(i * 7u + 5u);
  CHECK(vx_enqueue_write(q, gsrc, 0, h_src.data(), src_bytes, 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, gdst, 0, h_dst.data(), dst_bytes, 0, nullptr, nullptr));

  vx_module_h mod;
  vx_kernel_h kernel;
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  CHECK(vx_module_get_kernel(mod, "eng_k", &kernel));

  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = kernel;
  li.args_host = &arg;
  li.args_size = sizeof(arg);
  li.ndim = 1;
  li.grid_dim[0] = li.grid_dim[1] = li.grid_dim[2] = 1;
  li.block_dim[0] = warps * (uint32_t)nt;
  li.block_dim[1] = li.block_dim[2] = 1;
  li.lmem_size = LM_DATA + arg.bytes;
  li.cluster_dim[0] = li.cluster_dim[1] = li.cluster_dim[2] = 1;
  li.cluster_policy = VX_CLUSTER_POLICY_DEFAULT;
  vx_event_h ev = nullptr, oev = nullptr, dev_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
  CHECK(vx_enqueue_read(q, out.data(), obuf, 0, O_WORDS * 4, 1, &ev, &oev));
  std::vector<uint32_t> res(dst_bytes / 4);
  CHECK(vx_enqueue_read(q, res.data(), gdst, 0, dst_bytes, 1, &ev, &dev_ev));
  CHECK(vx_event_wait_value(oev, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(dev_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(dev_ev);
  vx_event_release(oev);
  vx_event_release(ev);

  int errors = 0;
  if (out[O_ERR]) {
    std::printf("*** an mbarrier wait gave up\n");
    ++errors;
  }
  // The local-memory pattern the kernel stores or reduces: word i = 0x01000000 + 3 i.
  if (arg.mode != M_INTAKE) {
    const uint32_t words = arg.bytes / 4;
    const size_t cold = cold_off / 4;
    for (size_t i = 0; i < res.size() && errors < 8; ++i) {
      const bool untouched = cold && i >= words && i < cold;
      const uint32_t v = 0x01000000u + (uint32_t)((i >= cold ? i - cold : i) % words) * 3u;
      const uint32_t want = untouched ? h_dst[i] : (arg.mode == M_REDUCE) ? h_dst[i] + v : v;
      if (res[i] != want) {
        std::printf("*** gdst word %zu: 0x%08x, expected 0x%08x\n", i, res[i], want);
        ++errors;
      }
    }
  }

  std::printf("knobs queue=%u st_fences=%u groups=%u egress=%u\n",
              out[O_KNOBS + 0], out[O_KNOBS + 1], out[O_KNOBS + 2], out[O_KNOBS + 3]);
  if (arg.mode == M_INTAKE) {
    uint64_t first = ~0ull, last = 0, done = 0;
    for (uint32_t w = 0; w < warps; ++w) {
      const uint32_t* p = out.data() + O_WARP + w * O_WARP_W;
      const uint64_t t0 = p[0] | ((uint64_t)p[1] << 32), t1 = p[2] | ((uint64_t)p[3] << 32);
      first = std::min(first, t0);
      last = std::max(last, t1);
      done = std::max(done, p[4] | ((uint64_t)p[5] << 32));
      std::printf("intake warp=%u t0=%llu t1=%llu pergap_x100=%llu done=%llu\n", w, (unsigned long long)t0,
                  (unsigned long long)t1, (unsigned long long)((t1 - t0) * 100 / arg.copies),
                  (unsigned long long)(p[4] | ((uint64_t)p[5] << 32)));
    }
    std::printf("intake warps=%u copies=%u bytes=%u span=%llu percopy_x100=%llu lag=%llu\n", warps, arg.copies, arg.bytes,
                (unsigned long long)(last - first),
                (unsigned long long)((last - first) * 100 / (warps * arg.copies)), (unsigned long long)(done - last));
  } else if (arg.mode == M_STORE) {
    for (uint32_t g = 0; g < arg.groups; ++g)
      std::printf("store group=%u commit=%u\n", g, out[O_COMMIT + g]);
    std::printf("store groups=%u per_group=%u bytes=%u waitk=%u t_waitk=%u t_wait0=%u t_waitb=%u\n", arg.groups,
                arg.copies, arg.bytes, arg.waitk, out[O_WAITK], out[O_WAIT0], out[O_WAITB]);
  } else if (arg.mode == M_REDUCE) {
    std::printf("reduce bytes=%u t=%u\n", arg.bytes, out[O_T1]);
  } else {
    std::printf("read bytes=%u t_read=%u t_done=%u t_full=%u cold=%u\n", arg.bytes, out[O_T1], out[O_T2], out[O_T3],
                arg.cold);
  }

  vx_buffer_release(obuf);
  vx_buffer_release(gdst);
  vx_buffer_release(gsrc);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);
  if (errors) {
    std::printf("FAILED!\n");
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
