// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Cluster-dispatch bench: VX_cta_admit under the scenario of
// tests/unittest/cluster_dispatch/scenario.h (round-robin over the GPCs with
// room from CLUSTER_GPC_START at each kernel start, a full GPC skipped, all or
// nothing, distinct SMs, the in-GPC policy, the per-SM cluster cap, K = 1
// clusters without a gang id, the in-order head, the cluster dispatch interval),
// replayed against dlmem_ref::place_chip; its CSV must equal the SimX twin's.
// Then the SM position functions of VX_gpu_pkg against vx_hier.h for every SM.
//
//   --csv=FILE   the placement and done log (default cluster_dispatch.csv)
//   --seed=N     CTA lifetimes (default 1)

#include "vl_simulator.h"
#include "VVX_cluster_dispatch_top.h"
#include "VX_config.h"
#include "scenario.h"
#include <cstring>
#include <string>

namespace cd = cluster_dispatch;

namespace {

using Sim = vl_simulator<VVX_cluster_dispatch_top>;

template <typename T>
void set_field(T& port, uint32_t idx, uint32_t width, uint64_t value) {
  const uint64_t mask = (width >= 64) ? ~0ull : ((1ull << width) - 1);
  uint64_t v = uint64_t(port);
  v &= ~(mask << (idx * width));
  v |= (value & mask) << (idx * width);
  port = T(v);
}

uint32_t clog2(uint32_t x) {
  uint32_t n = 0;
  while ((1u << n) < x)
    ++n;
  return n;
}

class RtlDut : public cd::Dut {
public:
  explicit RtlDut(Sim& sim) : sim_(sim) {}

  void reset() override {
    sim_->in_valid = 0;
    sim_->out_ready = 0;
    sim_->done_valid = 0;
    tick_ = sim_.reset(tick_);
  }

  cd::Out cycle(const cd::In& in) override {
    const uint32_t gpcs = VX_CFG_NUM_CLUSTERS;
    const uint32_t core_w = std::max<uint32_t>(1, clog2(VX_CFG_NUM_CORES));
    const uint32_t nw_w = std::max<uint32_t>(1, clog2(VX_CFG_NUM_WARPS));
    const uint32_t gang_w = clog2(VX_CFG_DLMEM_NUM_GANGS);
    sim_->in_valid = in.head != nullptr;
    if (in.head) {
      const cd::Beat& b = *in.head;
      sim_->in_ctx_id       = b.ctx;
      sim_->in_policy       = b.policy;
      sim_->in_explicit     = b.explicit_cluster;
      sim_->in_cluster_size = b.k;
      sim_->in_cluster_rank = b.rank;
      sim_->in_is_first     = (b.rank == 0);
      sim_->in_eop          = b.eop();
      sim_->in_block_size   = b.w * VX_CFG_NUM_THREADS;
      sim_->in_lmem_stride  = 0;
    }
    sim_->out_ready = in.out_ready;
    uint64_t valid = 0;
    for (uint32_t g = 0; g < gpcs; ++g) {
      const cd::Done& d = in.done[g];
      valid |= uint64_t(d.valid) << g;
      set_field(sim_->done_core, g, core_w, d.core);
      set_field(sim_->done_slot, g, nw_w, d.slot);
      if (gang_w)
        set_field(sim_->done_gang_id, g, gang_w, d.gang);
      set_field(sim_->done_epoch, g, 1, d.epoch);
    }
    sim_->done_valid = valid;
    sim_->eval();

    cd::Out o;
    o.busy      = sim_->busy;
    o.in_ready  = sim_->in_ready;
    o.out_valid = sim_->out_valid;
    o.out_pin   = sim_->out_pin;
    o.out_dest  = sim_->out_dest;
    o.out_slot  = sim_->out_forced_slot;
    o.out_gang  = sim_->out_gang_id;
    o.out_epoch = sim_->out_epoch;
    o.out_rank  = sim_->out_cluster_rank;
    o.out_ctx   = sim_->out_ctx_id;
    tick_ = sim_.step(tick_, 2);
    return o;
  }

  std::vector<uint32_t> pos(uint32_t sm) {
    sim_->pos_sm = sm;
    sim_->eval();
    return {uint32_t(sim_->pos_partition), uint32_t(sim_->pos_gpc), uint32_t(sim_->pos_cpc),
            uint32_t(sim_->pos_tpc_slot), uint32_t(sim_->pos_parity), uint32_t(sim_->pos_slot_dist)};
  }

private:
  Sim&     sim_;
  uint64_t tick_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  std::string csv = "cluster_dispatch.csv";
  uint32_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--csv=", 6) == 0) {
      csv = argv[i] + 6;
    } else if (std::strncmp(argv[i], "--seed=", 7) == 0) {
      seed = (uint32_t)std::strtoul(argv[i] + 7, nullptr, 0);
    } else if (argv[i][0] != '+') {
      std::fprintf(stderr, "usage: %s [--csv=FILE] [--seed=N]\n", argv[0]);
      return 2;
    }
  }
  FILE* f = std::fopen(csv.c_str(), "w");
  if (!f) {
    std::perror(csv.c_str());
    return 2;
  }
  Sim sim;
  RtlDut dut(sim);
  cd::Scenario sc(dut, seed, f);
  const bool ok = sc.run();
  std::fclose(f);
  const uint32_t pos_errors = cd::check_positions([&](uint32_t sm) { return dut.pos(sm); });
  std::printf("cluster_dispatch (RTL): %s (%u placement errors, %u position errors), log %s\n",
              (ok && pos_errors == 0) ? "PASSED" : "FAILED", sc.errors, pos_errors, csv.c_str());
  return (ok && pos_errors == 0) ? 0 : 1;
}
