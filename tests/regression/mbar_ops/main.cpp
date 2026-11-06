// mbar_ops: the mbarrier completion ops on a DLMEM device with mbarriers
// (RV64): arrive_drop, noComplete, pending_count, the relaxed waits and
// try_wait's suspend-time hint, run as the scripts of the H100 semantics
// probe (scripts.h), one object per script.
//
//   -m local     thread 0 of one block runs every script (default)
//   -m remote    a SPREAD cluster of two blocks on two cores of one socket:
//                rank 1 issues the scripts' arrives and drops on rank 0's
//                object through the window, rank 0 its init and waits; the
//                blocks meet in a cluster sync after every step
//   -m remote2s  the same with the two blocks in different sockets
//   -m relaxed   local, with the relaxed arrives, drops and waits
//   -m hint      local, every try_wait with a hint; then try_wait on a phase
//                that never completes, without a hint and with each hint of
//                the H100 probe: every one must time out alike
//   -x neg       a run that must stop the simulator: nocomplete (a noComplete
//                arrive completes the phase), dropzero (arrive_drop of the
//                whole expected count), samecore-wait (a test_wait on a peer's
//                object on the same core)
//
// The host replays every script with mbar_ref: each returned word, each wait
// result and the final word must match, and each value the H100 recorded
// must equal the device's (the pending count of a returned word).

#include <vortex2.h>
#include <VX_types.h>
#include <mbar_ref.h>
#include "common.h"
#include "scripts.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace {

namespace mr = vortex::mbar_ref;

const char* const kModeNames[] = {"local", "remote", "remote2s", "relaxed", "hint"};
const char* const kNegNames[]  = {"", "nocomplete", "dropzero", "samecore-wait"};

uint32_t mode = MODE_LOCAL;
uint32_t neg  = NEG_NONE;

#define CHECK(expr) do {                                              \
    vx_result_t _r = (expr);                                          \
    if (_r != VX_SUCCESS) {                                           \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n",          \
                   __FILE__, __LINE__, #expr, vx_result_string(_r));  \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

void usage() {
  std::printf("Usage: [-m local|remote|remote2s|relaxed|hint] [-x nocomplete|dropzero|samecore-wait] [-h]\n");
}

uint32_t parse_name(const char* s, const char* const* names, uint32_t n, uint32_t first) {
  for (uint32_t i = first; i < n; ++i) {
    if (!std::strcmp(s, names[i]))
      return i;
  }
  usage();
  std::exit(-1);
}

void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "m:x:h")) != -1) {
    switch (c) {
      case 'm': mode = parse_name(optarg, kModeNames, 5, 0); break;
      case 'x': neg  = parse_name(optarg, kNegNames, 4, 1); break;
      case 'h': usage(); std::exit(0);
      default:  usage(); std::exit(-1);
    }
  }
  if (optind < argc) {
    usage();
    std::exit(-1);
  }
}

vx_buffer_h make_buffer(vx_device_h dev, vx_queue_h q, const void* data, uint64_t bytes, uint64_t* addr) {
  vx_buffer_h b = nullptr;
  CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &b));
  CHECK(vx_buffer_address(b, addr));
  CHECK(vx_enqueue_write(q, b, 0, data, bytes, 0, nullptr, nullptr));
  return b;
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

  uint64_t nt = 0, nw = 0, cores = 0, max_cluster = 0, socket_size = 0, isa = 0;
  CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,       &nt));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,         &nw));
  CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES,         &cores));
  CHECK(vx_device_query(dev, VX_CAPS_DLMEM_MAX_CLUSTER, &max_cluster));
  CHECK(vx_device_query(dev, VX_CAPS_SOCKET_SIZE,       &socket_size));
  CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS,         &isa));

  const bool remote = !neg && (mode == MODE_REMOTE || mode == MODE_REMOTE2S);
  const bool samecore = (neg == NEG_SAMECORE);
  if (neg)
    std::printf("mbar_ops: negative run '%s'\n", kNegNames[neg]);
  else
    std::printf("mbar_ops: %s mode, %u cores, %u per socket\n", kModeNames[mode], (uint32_t)cores,
                (uint32_t)socket_size);

  if (!(isa & VX_ISA_EXT_DLMEM) || !(isa & VX_ISA_STD_A)) {
    std::printf("Error: the device needs DLMEM and the A extension\nFAILED!\n");
    return 1;
  }
  if (remote && (cores < 2 || max_cluster < 2
                 || (mode == MODE_REMOTE ? socket_size < 2 : socket_size != 1))) {
    std::printf("Error: %s needs 2 cores %s (%u cores, %u per socket, largest cluster %u)\nFAILED!\n",
                kModeNames[mode], mode == MODE_REMOTE ? "in one socket" : "in two sockets", (uint32_t)cores,
                (uint32_t)socket_size, (uint32_t)max_cluster);
    return 1;
  }
  if (samecore && (nw < 2 || max_cluster < 2)) {
    std::printf("Error: samecore-wait needs two one-warp blocks on a core\nFAILED!\n");
    return 1;
  }

  const std::vector<mbar_ops::Script> scripts = mbar_ops::scripts();
  if (scripts.size() > MAX_SCRIPTS) {
    std::printf("Error: %zu scripts, at most %u fit\nFAILED!\n", scripts.size(), MAX_SCRIPTS);
    return 1;
  }
  std::vector<uint32_t> steps(MAX_SCRIPTS * MAX_STEPS, ~0u);
  for (size_t s = 0; s < scripts.size(); ++s) {
    if (scripts[s].steps.size() > MAX_STEPS) {
      std::printf("Error: script %s has %zu steps, at most %u fit\nFAILED!\n", scripts[s].name,
                  scripts[s].steps.size(), MAX_STEPS);
      return 1;
    }
    for (size_t i = 0; i < scripts[s].steps.size(); ++i) {
      const auto& st = scripts[s].steps[i];
      steps[s * MAX_STEPS + i] = STEP(st.kind, st.side, st.n);
    }
  }
  std::vector<uint64_t> rec(MAX_SCRIPTS * MAX_STEPS, 0xdeadbeefull);
  std::vector<uint64_t> words(MAX_SCRIPTS, 0xdeadbeefull);
  std::vector<uint64_t> info(INFO_WORDS, ~uint64_t(0));
  std::vector<uint64_t> hints(HINT_REPS * (NUM_HINTS + 2), 0);

  ops_arg_t arg = {};
  vx_buffer_h steps_buf = make_buffer(dev, q, steps.data(), steps.size() * 4, &arg.steps_addr);
  vx_buffer_h rec_buf   = make_buffer(dev, q, rec.data(), rec.size() * 8, &arg.rec_addr);
  vx_buffer_h word_buf  = make_buffer(dev, q, words.data(), words.size() * 8, &arg.word_addr);
  vx_buffer_h info_buf  = make_buffer(dev, q, info.data(), info.size() * 8, &arg.info_addr);
  vx_buffer_h hint_buf  = make_buffer(dev, q, hints.data(), hints.size() * 8, &arg.hint_addr);
  arg.mode        = mode;
  arg.neg         = neg;
  arg.num_scripts = (uint32_t)scripts.size();
  arg.spin_limit  = 64;

  vx_module_h mod = nullptr;
  vx_kernel_h kernel = nullptr;
  CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kernel));

  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &arg;
  li.args_size    = sizeof(arg);
  li.ndim         = 1;
  li.grid_dim[0]  = (remote || samecore) ? 2 : 1;
  li.grid_dim[1]  = 1;
  li.grid_dim[2]  = 1;
  li.block_dim[0] = (uint32_t)nt;
  li.block_dim[1] = 1;
  li.block_dim[2] = 1;
  li.lmem_size    = LM_WORDS * sizeof(uint64_t);
  if (remote || samecore) {
    li.cluster_dim[0] = 2;
    li.cluster_dim[1] = 1;
    li.cluster_dim[2] = 1;
    li.cluster_policy = samecore ? VX_CLUSTER_POLICY_SINGLE_CORE : VX_CLUSTER_POLICY_SPREAD;
  }

  vx_event_h launch_ev = nullptr;
  std::vector<vx_event_h> evs(4, nullptr);
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
  CHECK(vx_enqueue_read(q, rec.data(),   rec_buf,  0, rec.size() * 8,   1, &launch_ev, &evs[0]));
  CHECK(vx_enqueue_read(q, words.data(), word_buf, 0, words.size() * 8, 1, &launch_ev, &evs[1]));
  CHECK(vx_enqueue_read(q, info.data(),  info_buf, 0, info.size() * 8,  1, &launch_ev, &evs[2]));
  CHECK(vx_enqueue_read(q, hints.data(), hint_buf, 0, hints.size() * 8, 1, &launch_ev, &evs[3]));
  for (auto e : evs) {
    CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e);
  }
  vx_event_release(launch_ev);

  int errors = 0;
  if (neg) {
    std::printf("*** the negative run '%s' completed: the simulator must stop it\n", kNegNames[neg]);
    ++errors;
  } else {
    if (info[INFO_DONE] != 1) {
      std::printf("*** rank 0 did not finish\n");
      ++errors;
    }
    if (remote) {
      const uint64_t c0 = info[INFO_CORE0], c1 = info[INFO_CORE1];
      const bool same_socket = (c0 / socket_size) == (c1 / socket_size);
      std::printf("mbar_ops: rank 0 on core %lu, rank 1 on core %lu\n", (unsigned long)c0, (unsigned long)c1);
      if (c0 == c1 || same_socket != (mode == MODE_REMOTE)) {
        std::printf("*** the ranks must sit on two cores of %s\n", mode == MODE_REMOTE ? "one socket" : "two sockets");
        ++errors;
      }
    }
    if (info[INFO_SPINS] != 0) {
      std::printf("*** %lu try_waits on a completed phase returned false\n", (unsigned long)info[INFO_SPINS]);
      ++errors;
    }
    uint32_t h100_checked = 0, h100_bad = 0, model_bad = 0;
    for (size_t s = 0; s < scripts.size(); ++s) {
      const auto& sc = scripts[s];
      std::vector<uint64_t> want;
      uint64_t want_word = 0;
      if (!mbar_ops::replay(sc, want, want_word)) {
        std::printf("*** mbar_ref rejects script %s (app bug)\n", sc.name);
        ++errors;
      }
      for (size_t i = 0; i < sc.steps.size(); ++i) {
        const auto& st = sc.steps[i];
        const uint64_t got = rec[s * MAX_STEPS + i];
        if (got != want[i]) {
          if (model_bad++ < 8)
            std::printf("*** %s step %zu (kind %u, n %u): recorded %#lx, mbar_ref %#lx\n", sc.name, i, st.kind,
                        st.n, (unsigned long)got, (unsigned long)want[i]);
        }
        if (st.h100 == mbar_ops::NA)
          continue;
        const int64_t v = mbar_ops::h100_view(st, got);
        h100_checked += sc.h100;
        if (v != st.h100) {
          if (h100_bad++ < 8)
            std::printf("*** %s step %zu (kind %u, n %u): %ld, %s %ld\n", sc.name, i, st.kind, st.n, (long)v,
                        sc.h100 ? "the H100" : "PTX", (long)st.h100);
        }
      }
      if (words[s] != want_word) {
        std::printf("*** %s: final word %#lx, mbar_ref %#lx\n", sc.name, (unsigned long)words[s],
                    (unsigned long)want_word);
        ++errors;
      }
    }
    std::printf("mbar_ops: %zu scripts, %u H100 values checked, %u differ; %u steps differ from mbar_ref\n",
                scripts.size(), h100_checked, h100_bad, model_bad);
    errors += (h100_bad != 0) + (model_bad != 0);

    if (mode == MODE_HINT) {
      const uint64_t kMask = ~(uint64_t(1) << 63);
      uint32_t bad = 0;
      for (uint32_t r = 0; r < HINT_REPS; ++r) {
        const uint64_t* o = &hints[r * (NUM_HINTS + 2)];
        const uint64_t base = o[0] & kMask;
        std::printf("mbar_ops: hint rep %u: try_wait %lu cycles, hinted", r, (unsigned long)base);
        for (uint32_t h = 1; h <= NUM_HINTS; ++h)
          std::printf(" %lu", (unsigned long)(o[h] & kMask));
        std::printf(", test_wait %lu\n", (unsigned long)(o[NUM_HINTS + 1] & kMask));
        // Every try_wait times out: the hint changes nothing.
        bad += (base < 256);
        for (uint32_t h = 0; h <= NUM_HINTS; ++h) {
          const uint64_t d = o[h] & kMask;
          bad += (o[h] >> 63) != 0;
          bad += (4 * d < 3 * base) || (4 * d > 5 * base);
        }
        bad += (o[NUM_HINTS + 1] >> 63) != 0;
        bad += 4 * (o[NUM_HINTS + 1] & kMask) > base;
      }
      if (bad) {
        std::printf("*** %u hint probe checks failed: each try_wait must time out like the unhinted one\n", bad);
        ++errors;
      }
    }
  }

  vx_kernel_release(kernel);
  vx_module_release(mod);
  vx_buffer_release(hint_buf);
  vx_buffer_release(info_buf);
  vx_buffer_release(word_buf);
  vx_buffer_release(rec_buf);
  vx_buffer_release(steps_buf);
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
