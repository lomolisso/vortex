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

// SimX twin of the cluster-dispatch bench: CtaAdmit under the shared scenario,
// whose CSV must match the RTL bench's line for line.
//
//   --csv=FILE   the placement and done log (default cluster_dispatch.csv)
//   --seed=N     CTA lifetimes (default 1)

#include "kmu/cta_admit.h"
#include "scenario.h"
#include <cstring>
#include <memory>
#include <string>

using namespace vortex;
namespace cd = cluster_dispatch;

namespace {

class SimxDut : public cd::Dut {
public:
  SimxDut() : admit_(cd::build_cfg()) {}

  void reset() override {
    admit_.reset();
  }

  cd::Out cycle(const cd::In& in) override {
    const uint32_t gpcs = VX_CFG_NUM_CLUSTERS;
    CtaAdmit::Inputs ai;
    kmu_req_t req{};
    if (in.head) {
      const cd::Beat& b = *in.head;
      req.ctx_id              = b.ctx;
      req.block_size          = b.w * VX_CFG_NUM_THREADS;
      req.block_dim[0]        = req.block_size;
      req.block_dim[1]        = 1;
      req.block_dim[2]        = 1;
      req.lmem_size           = 0;
      req.cluster_dim[0]      = b.k;
      req.cluster_dim[1]      = 1;
      req.cluster_dim[2]      = 1;
      req.cluster_rank        = b.rank;
      req.is_first_of_cluster = (b.rank == 0);
      req.policy              = b.policy;
      req.explicit_cluster    = b.explicit_cluster;
      req.block_idx[0]        = b.msg * b.k + b.rank;
      ai.head = &req;
    }
    ai.out_ready = in.out_ready;
    std::vector<AdmitDone> dones(gpcs);
    ai.done.assign(gpcs, nullptr);
    for (uint32_t g = 0; g < gpcs; ++g) {
      if (in.done[g].valid) {
        dones[g] = {in.done[g].core, in.done[g].slot, in.done[g].gang, in.done[g].epoch};
        ai.done[g] = &dones[g];
      }
    }
    ai.rel_pend.assign(gpcs, std::vector<bool>(VX_CFG_DLMEM_NUM_GANGS, false));

    cd::Out o;
    o.busy = admit_.busy();
    const CtaAdmit::Outputs ao = admit_.tick(ai);
    o.in_ready  = ao.in_ready;
    o.out_valid = ao.out_valid;
    o.out_pin   = ao.out_pinned;
    o.out_dest  = ao.out_dest;
    o.out_slot  = ao.out.forced_slot;
    o.out_gang  = ao.out.gang_id;
    o.out_epoch = ao.out.epoch;
    o.out_rank  = ao.out.cluster_rank;
    o.out_ctx   = ao.out.ctx_id;
    return o;
  }

private:
  CtaAdmit admit_;
};

}  // namespace

int main(int argc, char** argv) {
  std::string csv = "cluster_dispatch.csv";
  uint32_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--csv=", 6) == 0) {
      csv = argv[i] + 6;
    } else if (std::strncmp(argv[i], "--seed=", 7) == 0) {
      seed = (uint32_t)std::strtoul(argv[i] + 7, nullptr, 0);
    } else {
      std::fprintf(stderr, "usage: %s [--csv=FILE] [--seed=N]\n", argv[0]);
      return 2;
    }
  }
  FILE* f = std::fopen(csv.c_str(), "w");
  if (!f) {
    std::perror(csv.c_str());
    return 2;
  }
  SimxDut dut;
  cd::Scenario sc(dut, seed, f);
  const bool ok = sc.run();
  std::fclose(f);
  std::printf("cluster_dispatch (SimX): %s (%u errors), log %s\n", ok ? "PASSED" : "FAILED", sc.errors, csv.c_str());
  return ok ? 0 : 1;
}
