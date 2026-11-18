// mbarrier operation costs on the H100 SM path. Every op is one AMO pass on the array, queued
// in the mbarrier op queue; expectations are structural (smem_bank_ref.h); the H100's measured
// costs (12/spec.md §4.1) are printed as "H100REF metric=... h100=... vortex=..." lines.
//   -m lat      one thread, a dependent chase of each op (its result feeds the next op's address through
//               one AND and one ADD); the AND + ADD cost after a response is a load chase with the same ops
//               minus a bare load chase, so the op's response time is the hop minus that: a shared atomic's
//               chase plus the op queue's stages, +-3 cycles, as the one-warp front end lands the dependent
//               ops at a response-dependent phase; every kind costs the same (one pass), a pending
//               test_wait as a completed one. try_wait (completed) adds the warp's sleep and wake and is
//               reported only.
//   -m issue    lane 0 of every warp of a core, back-to-back expect_tx / complete_tx with rd = x0: the op
//               queue treats both kinds alike (cycles per op equal +-0.2); both reported against the H100.
//               init / inval are plain stores, not op-queue ops.
//   -m update   every warp of a core, relaxed arrives with rd = x0 on one barrier (one bank pair: one update
//               per cycle, +-5 %) and on four barriers in four bank pairs (four per cycle, +-10 %)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <vortex.h>
#include <smem_bank_ref.h>

#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

#ifndef VX_CFG_H100_SM_ENABLE
#error "mbar_cost tests the H100 SM's mbarrier op costs: build with -DVX_CFG_H100_SM_ENABLE"
#endif

using namespace vortex::smem_bank_ref;

const char* kernel_file = "kernel.vxbin";
std::vector<std::string> modes;

vx_device_h device = nullptr;
vx_queue_h  queue = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
uint64_t out_addr = 0;
uint32_t lanes = 0;
uint32_t warps = 0;
uint32_t errors = 0;

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-m lat|issue|update|bulk[,...]] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:m:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'm': {
      std::string a(optarg);
      size_t s = 0;
      while (s <= a.size()) {
        size_t e = a.find(',', s);
        if (e == std::string::npos) e = a.size();
        if (e > s) modes.push_back(a.substr(s, e - s));
        s = e + 1;
      }
      break;
    }
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (out_buffer) vx_buffer_release(out_buffer);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static void check(const std::string& what, bool ok, const std::string& detail) {
  printf("  %-44s %s %s\n", what.c_str(), detail.c_str(), ok ? "ok" : "MISMATCH");
  errors += !ok;
}

static void launch(kernel_arg_t& arg, uint32_t block, uint64_t* out, size_t n) {
  arg.out_addr = out_addr;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = block;
  li.lmem_size    = 1024;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_enqueue_read(queue, out, out_buffer, 0, n * 8, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
}

// A one-pass shared atomic's chase plus the op queue's stages.
static double op_response() {
  SmemCfg c = {};
  c.banks      = VX_CFG_LMEM_NUM_BANKS;
  c.bank_bytes = VX_CFG_LMEM_BANK_BYTES;
#ifdef VX_CFG_EXT_DLMEM_ENABLE
  c.pipe       = smem_pipe_stages(VX_CFG_NUM_OPCS, true);
#else
  c.pipe       = smem_pipe_stages(VX_CFG_NUM_OPCS, false);
#endif
  return smem_atom_latency(1, c) + MBAR_QUEUE_STAGES;
}

static void h100ref(const char* metric, double h100, double vortex) {
  printf("H100REF metric=%s h100=%.2f vortex=%.3f\n", metric, h100, vortex);
}

static std::string fmt(double v) {
  char b[32];
  snprintf(b, sizeof(b), "%.2f", v);
  return b;
}

static void mode_lat() {
  const uint32_t reps = 8;
  const double hops = reps * MC_UNROLL;
  kernel_arg_t arg = {};
  arg.kind = MC_LAT;
  arg.reps = reps;
  uint64_t o[MC_LAT_N];
  launch(arg, lanes, o, MC_LAT_N);
  const double alu = (double)(int64_t)o[MC_OP_ALU] / hops;
  printf("AND + ADD after a response %.2f cycles\n", alu);
  const struct { uint32_t k; const char* name; const char* metric; double h100; bool checked; } rows[] = {
    {MC_OP_ARRIVE,       "arrive",                "mbar_arrive",       39.81, true},
    {MC_OP_ARRIVE_TX,    "arrive.expect_tx",      "mbar_arrive_tx",    39.12, true},
    {MC_OP_ARRIVE_DROP,  "arrive_drop",           "mbar_arrive_drop",  35.12, true},
    {MC_OP_TEST_WAIT,    "test_wait (completed)", "mbar_test_wait",    41.97, true},
    {MC_OP_TRY_WAIT,     "try_wait (completed)",  "mbar_try_wait",     42.12, false},
    {MC_OP_TEST_PENDING, "test_wait (pending)",   "mbar_test_pending", 46.16, true},
  };
  const double want = op_response();
  const double first = o[rows[0].k] / hops - alu;
  for (auto& r : rows) {
    const double got = o[r.k] / hops - alu;
    const std::string what = std::string(r.name) + " response";
    if (r.checked) {
      check(what, std::fabs(got - want) <= 3.0, fmt(got) + " (want " + fmt(want) + " +-3)");
      check(what + " as arrive's", std::fabs(got - first) <= 1.0, fmt(got) + " (arrive " + fmt(first) + " +-1)");
    } else {
      printf("  %-44s %s (%s + the warp's sleep and wake) reported\n", what.c_str(), fmt(got).c_str(),
             fmt(want).c_str());
    }
    h100ref(r.metric, r.h100, got);
  }
}

static void mode_issue() {
  const uint32_t reps = 8;
  const double ops = double(reps) * MC_UNROLL * warps;
  kernel_arg_t arg = {};
  arg.kind = MC_ISSUE;
  arg.reps = reps;
  uint64_t o[MC_ISSUE_N];
  launch(arg, warps * lanes, o, MC_ISSUE_N);
  printf("  %u warps, lane 0 each\n", warps);
  const double ct = o[MC_IS_COMPLETE_TX] / ops;
  const double et = o[MC_IS_EXPECT_TX] / ops;
  check("complete_tx cycles per op as expect_tx's", std::fabs(ct - et) <= 0.2,
        fmt(ct) + " (expect_tx " + fmt(et) + " +-0.20)");
  h100ref("mbar_complete_tx_interval", 6.38, ct);
  h100ref("mbar_expect_tx_interval", 0.53, et);
  printf("  %-44s not modelled (plain 64-bit stores)\n", "init / inval (H100 7.28 / 31.03)");
}

static void mode_update() {
  for (uint32_t nbars : {1u, 4u}) {
    const uint32_t reps = 16;
    kernel_arg_t arg = {};
    arg.kind = MC_UPDATE;
    arg.reps = reps;
    arg.nbars = nbars;
    uint64_t o[1];
    launch(arg, warps * lanes, o, 1);
    const double rate = double(reps) * 16 * lanes * warps / o[0];
    const double tol = (nbars == 1) ? 0.05 : 0.10;
    char what[64];
    snprintf(what, sizeof(what), "%u barrier(s): updates per cycle", nbars);
    check(what, std::fabs(rate - nbars) <= tol * nbars, fmt(rate) + " (want " + fmt(nbars) + " +-" +
          std::to_string(int(tol * 100)) + " %)");
  }
}

// Completions of bulk copies reach their mbarrier beside the op queue, not
// through it: a copy stream lasts as long with another warp's op stream as without.
static void mode_bulk() {
#ifndef VX_CFG_EXT_DXA_ENABLE
  check("bulk copies need the DXA", false, "VX_CFG_EXT_DXA_ENABLE is off");
#else
  uint64_t t[2];
  for (uint32_t loaded : {0u, 1u}) {
    kernel_arg_t arg = {};
    arg.kind = MC_BULK;
    arg.reps = 8;
    arg.nbars = loaded;
    arg.src_addr = out_addr + 2048;
    launch(arg, 2 * lanes, &t[loaded], 1);
  }
  const double copies = 8.0 * MC_UNROLL;
  printf("  128-B copies: %.2f cycles per copy alone, %.2f beside a complete_tx stream\n",
         t[0] / copies, t[1] / copies);
  check("bulk copy stream beside op stream", t[1] <= t[0] * 1.05,
        fmt(t[1] / copies) + " (want <= " + fmt(t[0] * 1.05 / copies) + ")");
#endif
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  if (modes.empty()) modes = {"lat", "issue", "update"};
  for (auto& m : modes) {
    if (m != "lat" && m != "issue" && m != "update" && m != "bulk") {
      show_usage();
      return -1;
    }
  }
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  uint64_t v;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &v));
  lanes = (uint32_t)v;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &v));
  warps = (uint32_t)v;
  RT_CHECK(vx_buffer_create(device, 4096, VX_MEM_READ_WRITE, &out_buffer));
  RT_CHECK(vx_buffer_address(out_buffer, &out_addr));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  for (auto& m : modes) {
    std::cout << "mode " << m << std::endl;
    if (m == "lat") mode_lat();
    else if (m == "issue") mode_issue();
    else if (m == "bulk") mode_bulk();
    else mode_update();
  }

  cleanup();
  if (errors) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
