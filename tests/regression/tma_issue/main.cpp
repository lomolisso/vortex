// Every DXA form issued by one lane of a warp, the other lanes inactive
// (-k0), and the per-warp issue cost of each form (-k1).
#include <vortex2.h>
#include <dxa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include "common.h"

#define CHECK(expr)                                                                     \
  do {                                                                                  \
    vx_result_t _r = (expr);                                                            \
    if (_r != VX_SUCCESS) {                                                                      \
      std::printf("*** %s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,             \
                  vx_result_string(_r));                                                \
      std::exit(-1);                                                                    \
    }                                                                                   \
  } while (0)

namespace dh = vortex::dxa;

namespace {

const char* kernel_file = "kernel.vxbin";
uint32_t opt_kernel = K_FORMS;
int opt_lane = -1;
uint32_t opt_seed = 1;
uint32_t opt_warps = 1;
int opt_form = -1;

const char* const cost_names[C_COUNT] = {
  "bulk", "tensor", "bulk_store", "bulk_reduce", "tensor_store", "tensor_reduce",
  "prefetch_bulk", "prefetch_tensor", "tmap_inval", "tmap_prefetch",
};

const char* const form_names[F_COUNT] = {
  "bulk_g2s", "bulk_mc", "copy_s2s", "tload_1d", "tload_2d", "tload_3d", "tload_4d", "tload_5d",
  "bulk_store", "bulk_reduce", "cluster_reduce", "tensor_store", "tensor_reduce", "prefetch_bulk",
  "prefetch_tensor", "tmap_prefetch", "tmap_inval",
};

uint32_t gsrc_word(uint32_t i) { return (i * 2654435761u) ^ 0x5A5A0000u; }
uint32_t src_word(uint32_t i)  { return 0xC0000000u | (i * 7919u); }
uint32_t red_word(uint32_t i)  { return i * 3u + 1u; }
uint32_t gred_word(uint32_t i) { return 0x01000000u + i * 5u; }
constexpr uint32_t CANARY = 0xA5A5A5A5u;

const uint32_t kDim[5] = {TDIM0, TDIM1, TDIM2, TDIM3, TDIM4};
const uint32_t kBox[5] = {TBOX0, TBOX1, TBOX2, TBOX3, TBOX4};
const int32_t  kC[5]   = {TC0, TC1, TC2, TC3, TC4};

void encode(vx_tensor_map_t* m, uint32_t rank, uint64_t base, const uint32_t* dim, const uint32_t* box) {
  uint64_t gdim[5], gstr[4];
  uint32_t bx[5], es[5];
  uint64_t bytes = 4;
  for (uint32_t i = 0; i < rank; ++i) {
    gdim[i] = dim[i];
    bx[i] = box[i];
    es[i] = 1;
    bytes *= dim[i];
    if (i + 1 < rank)
      gstr[i] = bytes;
  }
  const int r = dh::encode_tiled(m, dh::TensorMapDataType::UInt32, rank, base, gdim, gstr, bx, es,
                                 dh::TensorMapInterleave::None, dh::TensorMapSwizzle::None,
                                 dh::TensorMapL2Promotion::None, dh::TensorMapFloatOOBfill::None);
  if (r != VX_SUCCESS) {
    std::printf("*** encode_tiled rank %u failed: %d\n", rank, r);
    std::exit(-1);
  }
}

// The box of a rank-r load from gsrc, dim 0 fastest.
std::vector<uint32_t> box_words(uint32_t rank) {
  std::vector<uint32_t> v;
  uint32_t b[5] = {0, 0, 0, 0, 0};
  uint32_t n = 1;
  for (uint32_t i = 0; i < rank; ++i)
    n *= kBox[i];
  for (uint32_t k = 0; k < n; ++k) {
    uint32_t idx = 0, mul = 1;
    for (uint32_t i = 0; i < rank; ++i) {
      idx += (kC[i] + b[i]) * mul;
      mul *= kDim[i];
    }
    v.push_back(gsrc_word(idx));
    for (uint32_t i = 0; i < rank; ++i) {
      if (++b[i] < kBox[i])
        break;
      b[i] = 0;
    }
  }
  return v;
}

void show_usage() {
  std::printf("Usage: [-k kernel: 0 forms, 1 cost, 2 visibility] [-l issuing lane] [-s seed] [-w cost warps] [-f cost form]\n");
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:l:s:w:f:h")) != -1) {
    switch (c) {
    case 'k': opt_kernel = std::atoi(optarg); break;
    case 'l': opt_lane = std::atoi(optarg); break;
    case 's': opt_seed = std::atoi(optarg); break;
    case 'w': opt_warps = std::atoi(optarg); break;
    case 'f': opt_form = std::atoi(optarg); break;
    default:  show_usage(); std::exit(-1);
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  parse_args(argc, argv);

  vx_device_h dev = nullptr;
  vx_queue_h q = nullptr;
  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority = VX_QUEUE_PRIORITY_NORMAL;
  CHECK(vx_device_open(0, &dev));
  CHECK(vx_queue_create(dev, &qi, &q));
  uint64_t nt = 0, nw = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &nw));

  const uint32_t lane = opt_lane >= 0 ? (uint32_t)opt_lane : 1 + opt_seed % (uint32_t)(nt - 1);
  if (lane >= nt || opt_warps == 0 || opt_warps > nw || opt_warps > MAX_COST_WARPS) {
    std::printf("*** bad lane %u or warps %u (threads %lu, warps %lu)\n", lane, opt_warps, nt, nw);
    return -1;
  }

  const uint32_t tbytes = TENSOR_WORDS * 4;
  vx_buffer_h mbuf, gsrc, gdst, gst, gred, gvis, obuf, dbuf;
  uint64_t a_maps, a_gsrc, a_gdst, a_gst, a_gred, a_vis, a_out, a_dump;
  const uint32_t threads_max = (uint32_t)(nt * MAX_COST_WARPS);
  CHECK(vx_buffer_create(dev, NUM_MAPS * 128, VX_MEM_READ, &mbuf));
  CHECK(vx_buffer_create(dev, tbytes, VX_MEM_READ, &gsrc));
  CHECK(vx_buffer_create(dev, 2 * BULK_BYTES, VX_MEM_READ_WRITE, &gdst));
  CHECK(vx_buffer_create(dev, tbytes, VX_MEM_READ_WRITE, &gst));
  CHECK(vx_buffer_create(dev, tbytes, VX_MEM_READ_WRITE, &gred));
  CHECK(vx_buffer_create(dev, 2 * VIS_BYTES, VX_MEM_READ_WRITE, &gvis));
  CHECK(vx_buffer_create(dev, threads_max * O_WORDS * 4, VX_MEM_READ_WRITE, &obuf));
  CHECK(vx_buffer_create(dev, 2 * LM_BYTES, VX_MEM_READ_WRITE, &dbuf));
  CHECK(vx_buffer_address(mbuf, &a_maps));
  CHECK(vx_buffer_address(gsrc, &a_gsrc));
  CHECK(vx_buffer_address(gdst, &a_gdst));
  CHECK(vx_buffer_address(gst, &a_gst));
  CHECK(vx_buffer_address(gred, &a_gred));
  CHECK(vx_buffer_address(gvis, &a_vis));
  CHECK(vx_buffer_address(obuf, &a_out));
  CHECK(vx_buffer_address(dbuf, &a_dump));

  std::vector<vx_tensor_map_t> maps(NUM_MAPS);
  for (uint32_t r = 1; r <= 5; ++r)
    encode(&maps[r - 1], r, a_gsrc, kDim, kBox);
  encode(&maps[MAP_STORE], 2, a_gst, kDim, kBox);
  encode(&maps[MAP_RED], 2, a_gred, kDim, kBox);
  CHECK(vx_enqueue_write(q, mbuf, 0, maps.data(), NUM_MAPS * 128, 0, nullptr, nullptr));

  std::vector<uint32_t> h_gsrc(TENSOR_WORDS), h_gred(TENSOR_WORDS), h_zero(TENSOR_WORDS, 0);
  for (uint32_t i = 0; i < TENSOR_WORDS; ++i) {
    h_gsrc[i] = gsrc_word(i);
    h_gred[i] = gred_word(i);
  }
  std::vector<uint32_t> h_gdst(2 * BULK_BYTES / 4);
  for (uint32_t i = 0; i < h_gdst.size(); ++i)
    h_gdst[i] = i < BULK_BYTES / 4 ? 0 : gred_word(i);
  std::vector<uint32_t> h_gvis(2 * VIS_BYTES / 4);
  for (uint32_t i = 0; i < VIS_BYTES / 4; ++i) {
    h_gvis[i] = 0;
    h_gvis[VIS_BYTES / 4 + i] = vis_red_init(i);
  }
  CHECK(vx_enqueue_write(q, gsrc, 0, h_gsrc.data(), tbytes, 0, nullptr, nullptr));

  vx_module_h mod;
  vx_kernel_h kernel;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "tma_k", &kernel));

  auto read_back = [&](void* dst, vx_buffer_h b, uint64_t size) {
    vx_event_h ev = nullptr;
    CHECK(vx_enqueue_read(q, dst, b, 0, size, 0, nullptr, &ev));
    CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  };

  auto launch = [&](tma_arg_t& arg, uint32_t blocks, uint32_t threads, std::vector<uint32_t>& out,
                    std::vector<uint32_t>& dump) {
    CHECK(vx_enqueue_write(q, gdst, 0, h_gdst.data(), 2 * BULK_BYTES, 0, nullptr, nullptr));
    CHECK(vx_enqueue_write(q, gst, 0, h_zero.data(), tbytes, 0, nullptr, nullptr));
    CHECK(vx_enqueue_write(q, gred, 0, h_gred.data(), tbytes, 0, nullptr, nullptr));
    if (arg.kernel == K_VISIBLE)
      CHECK(vx_enqueue_write(q, gvis, 0, h_gvis.data(), 2 * VIS_BYTES, 0, nullptr, nullptr));
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = kernel;
    li.args_host = &arg;
    li.args_size = sizeof(arg);
    li.ndim = 1;
    li.grid_dim[0] = blocks;
    li.grid_dim[1] = li.grid_dim[2] = 1;
    li.block_dim[0] = threads;
    li.block_dim[1] = li.block_dim[2] = 1;
    li.lmem_size = LM_BYTES;
    li.cluster_dim[0] = blocks;
    li.cluster_dim[1] = li.cluster_dim[2] = 1;
    li.cluster_policy = blocks > 1 ? VX_CLUSTER_POLICY_SPREAD : VX_CLUSTER_POLICY_DEFAULT;
    li.cluster_flags = 0;
    vx_event_h ev = nullptr, oev = nullptr, dev_ev = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
    CHECK(vx_enqueue_read(q, out.data(), obuf, 0, blocks * threads * O_WORDS * 4, 1, &ev, &oev));
    CHECK(vx_enqueue_read(q, dump.data(), dbuf, 0, blocks * LM_BYTES, 1, &ev, &dev_ev));
    CHECK(vx_event_wait_value(oev, 1, VX_TIMEOUT_INFINITE));
    CHECK(vx_event_wait_value(dev_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(dev_ev);
    vx_event_release(oev);
    vx_event_release(ev);
  };

  tma_arg_t arg = {};
  arg.maps_addr = a_maps;
  arg.gsrc_addr = a_gsrc;
  arg.gdst_addr = a_gdst;
  arg.gst_addr = a_gst;
  arg.gred_addr = a_gred;
  arg.vis_addr = a_vis;
  arg.out_addr = a_out;
  arg.dump_addr = a_dump;
  arg.kernel = opt_kernel;
  arg.lane = lane;
  arg.warps = opt_warps;

  std::vector<uint32_t> out(threads_max * O_WORDS), dump(2 * LM_BYTES / 4);
  int errors = 0;
  auto fail = [&](const std::string& s) {
    if (errors++ < 20)
      std::printf("*** %s\n", s.c_str());
  };
  auto check_threads = [&](uint32_t threads, uint32_t per_block) {
    for (uint32_t t = 0; t < threads; ++t) {
      const uint32_t* o = &out[t * O_WORDS];
      if (o[O_DONE] != 1)
        fail("thread " + std::to_string(t) + " did not finish");
      if (o[O_ALIVE] != (t % per_block) * 0x9E3779B1u + 7u)
        fail("thread " + std::to_string(t) + " lost a register value across the issue");
      if (o[O_WAIT])
        fail("thread " + std::to_string(t) + " gave up an mbarrier wait");
    }
  };

  if (opt_kernel == K_FORMS) {
    std::printf("forms: issuing lane %u of %lu\n", lane, nt);
    launch(arg, 2, (uint32_t)nt, out, dump);
    check_threads(2 * (uint32_t)nt, (uint32_t)nt);

    std::vector<uint32_t> want(LM_BYTES / 4, CANARY);
    for (uint32_t i = 0; i < SRC_BYTES / 4; ++i)
      want[LM_SRC / 4 + i] = src_word(i);
    auto area = [&](uint32_t f) { return (LM_AREA + f * AREA_BYTES) / 4; };
    for (uint32_t i = 0; i < AREA_BYTES / 4; ++i)
      want[area(F_CRED) + i] = red_word(i);
    std::vector<uint32_t> want1 = want;
    for (uint32_t i = 0; i < BULK_BYTES / 4; ++i) {
      want[area(F_BULK_G2S) + i] = gsrc_word(i);
      want[area(F_BULK_MC) + i] = gsrc_word(i);
      want1[area(F_CRED) + i] = red_word(i) + src_word(i);
      want1[area(F_COPY_S2S) + i] = src_word(i);
    }
    want.insert(want.end(), want1.begin(), want1.end());
    for (uint32_t r = 1; r <= 5; ++r) {
      const auto b = box_words(r);
      for (uint32_t i = 0; i < b.size(); ++i)
        want[area(F_TLOAD_1D + r - 1) + i] = b[i];
    }
    const auto b2 = box_words(2);
    for (uint32_t i = 0; i < b2.size(); ++i) {
      want[area(F_TMAP_PF) + i] = b2[i];
      want[area(F_TMAP_INV) + i] = b2[i];
    }
    for (uint32_t i = 0; i < 2 * LM_BYTES / 4; ++i) {
      if ((i % (LM_BYTES / 4)) < (LM_MBAR + 8 * NUM_MBARS) / 4)
        continue;
      if (dump[i] != want[i]) {
        const uint32_t byte = (i % (LM_BYTES / 4)) * 4;
        const char* what = byte < LM_AREA ? "source" : form_names[(byte - LM_AREA) / AREA_BYTES];
        char s[160];
        std::snprintf(s, sizeof(s), "block %u local word +%#x (%s): %#x, expected %#x", i / (LM_BYTES / 4), byte,
                      what, dump[i], want[i]);
        fail(s);
      }
    }

    std::vector<uint32_t> g(2 * BULK_BYTES / 4), t(TENSOR_WORDS);
    read_back(g.data(), gdst, 2 * BULK_BYTES);
    for (uint32_t i = 0; i < g.size(); ++i) {
      const uint32_t w = i < BULK_BYTES / 4 ? src_word(i) : gred_word(i) + src_word(i - BULK_BYTES / 4);
      if (g[i] != w)
        fail("bulk " + std::string(i < BULK_BYTES / 4 ? "store" : "reduce") + " word " + std::to_string(i)
             + ": " + std::to_string(g[i]) + ", expected " + std::to_string(w));
    }
    // The 2D box at (TC0, TC1) of the TDIM0-wide store/reduce tensors takes src words 0..15.
    auto in_box = [&](uint32_t i) {
      const uint32_t x = i % TDIM0, y = i / TDIM0;
      return y < TDIM1 && x - (uint32_t)TC0 < TBOX0 && y - (uint32_t)TC1 < TBOX1;
    };
    auto box_src = [&](uint32_t i) { return src_word((i / TDIM0 - TC1) * TBOX0 + (i % TDIM0 - TC0)); };
    read_back(t.data(), gst, tbytes);
    for (uint32_t i = 0; i < TENSOR_WORDS; ++i) {
      const uint32_t w = in_box(i) ? box_src(i) : 0;
      if (t[i] != w)
        fail("tensor store word " + std::to_string(i) + ": " + std::to_string(t[i]) + ", expected "
             + std::to_string(w));
    }
    read_back(t.data(), gred, tbytes);
    for (uint32_t i = 0; i < TENSOR_WORDS; ++i) {
      const uint32_t w = gred_word(i) + (in_box(i) ? box_src(i) : 0);
      if (t[i] != w)
        fail("tensor reduce word " + std::to_string(i) + ": " + std::to_string(t[i]) + ", expected "
             + std::to_string(w));
    }
  } else if (opt_kernel == K_VISIBLE) {
    std::printf("visibility: issuing lane %u of %lu, %u-byte bulk store and reduce\n", lane, nt, VIS_BYTES);
    launch(arg, 1, (uint32_t)nt, out, dump);
    check_threads((uint32_t)nt, (uint32_t)nt);
    for (uint32_t t = 0; t < nt; ++t) {
      const uint32_t s = out[t * O_WORDS + O_STALE];
      if (s & 0xffffu)
        fail("thread " + std::to_string(t) + " read " + std::to_string(s & 0xffffu)
             + " stale words after the bulk store's wait_group");
      if (s >> 16)
        fail("thread " + std::to_string(t) + " read " + std::to_string(s >> 16)
             + " stale words after the bulk reduce's wait_group");
    }
    std::vector<uint32_t> g(2 * VIS_BYTES / 4);
    read_back(g.data(), gvis, 2 * VIS_BYTES);
    for (uint32_t i = 0; i < VIS_BYTES / 4; ++i) {
      if (g[i] != vis_word(i))
        fail("bulk store word " + std::to_string(i));
      if (g[VIS_BYTES / 4 + i] != vis_red_init(i) + vis_word(i))
        fail("bulk reduce word " + std::to_string(i));
    }
  } else {
    const uint32_t threads = (uint32_t)nt * opt_warps;
    for (uint32_t c = 0; c < C_COUNT; ++c) {
      if (opt_form >= 0 && (uint32_t)opt_form != c)
        continue;
      arg.form = c;
      launch(arg, 1, threads, out, dump);
      check_threads(threads, threads);
      for (uint32_t w = 0; w < opt_warps; ++w) {
        const uint32_t* o = &out[(w * nt + lane) * O_WORDS];
        const uint32_t per_x100 = (o[O_T2] - o[O_T1]) * 100u / (COST_N2 - COST_N1);
        std::printf("cost form=%s warp=%u warps=%u t1=%u t2=%u percopy_x100=%u\n", cost_names[c], w,
                    opt_warps, o[O_T1], o[O_T2], per_x100);
        if (o[O_T2] <= o[O_T1])
          fail(std::string("cost ") + cost_names[c] + ": more copies took no longer");
      }
    }
  }

  vx_buffer_release(dbuf);
  vx_buffer_release(obuf);
  vx_buffer_release(gvis);
  vx_buffer_release(gred);
  vx_buffer_release(gst);
  vx_buffer_release(gdst);
  vx_buffer_release(gsrc);
  vx_buffer_release(mbuf);
  vx_queue_release(q);
  vx_device_dump_perf(dev, stdout);
  vx_device_release(dev);

  if (errors) {
    std::printf("FAILED! (%d errors)\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
