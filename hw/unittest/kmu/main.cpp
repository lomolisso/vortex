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

#include "vl_simulator.h"
#include "VVX_kmu_top.h"
#include <iostream>
#include "VX_config.h"
#include "VX_types.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

static uint64_t timestamp = 0;
static bool trace_enabled = false;
static uint64_t trace_start_time = TRACE_START_TIME;
static uint64_t trace_stop_time  = TRACE_STOP_TIME;

double sc_time_stamp() {
  return timestamp;
}

bool sim_trace_enabled() {
  if (timestamp >= trace_start_time
   && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable(bool enable) {
  trace_enabled = enable;
}

using Sim = vl_simulator<VVX_kmu_top>;

// A same-core cluster fills at most one core's warps; a distributed one spans
// up to DLMEM_MAX_CLUSTER cores.
#if VX_CFG_EXT_DLMEM_ENABLED
constexpr uint32_t MAX_K = std::max<uint32_t>(VX_CFG_NUM_WARPS, VX_CFG_DLMEM_MAX_CLUSTER);
#else
constexpr uint32_t MAX_K = VX_CFG_NUM_WARPS;
#endif

struct Beat {
  uint32_t idx[3];
  uint32_t size;
  uint32_t rank;
  bool     first;
  bool     eop;
  uint32_t ctx_id;
  uint32_t policy = 0;   // DLMEM: the cluster policy DCR; pin and the gang fields stay 0
};

struct Launch {
  uint32_t grid[3];
  uint32_t cluster[3];
  int      policy = -1;   // DLMEM: the policy DCR to write, or -1 to keep the last one
};

static uint64_t write_dcr(Sim& sim, uint32_t addr, uint32_t value, uint64_t tick) {
  sim->dcr_req_valid = 1;
  sim->dcr_req_rw    = 1;
  sim->dcr_req_addr  = addr;
  sim->dcr_req_data  = value;
  tick = sim.step(tick, 2);
  sim->dcr_req_valid = 0;
  sim->dcr_req_rw    = 0;
  return tick;
}

// Clusters in grid order (x fastest), and inside a cluster its blocks with the
// intra offset x fastest; the rank is that intra offset, linearized.
static std::vector<Beat> reference_walk(const Launch& l) {
  std::vector<Beat> beats;
  uint32_t k = l.cluster[0] * l.cluster[1] * l.cluster[2];
  for (uint32_t gz = 0; gz < l.grid[2]; gz += l.cluster[2]) {
    for (uint32_t gy = 0; gy < l.grid[1]; gy += l.cluster[1]) {
      for (uint32_t gx = 0; gx < l.grid[0]; gx += l.cluster[0]) {
        for (uint32_t iz = 0; iz < l.cluster[2]; ++iz) {
          for (uint32_t iy = 0; iy < l.cluster[1]; ++iy) {
            for (uint32_t ix = 0; ix < l.cluster[0]; ++ix) {
              Beat b;
              b.idx[0] = gx + ix;
              b.idx[1] = gy + iy;
              b.idx[2] = gz + iz;
              b.size   = k;
              b.rank   = ix + l.cluster[0] * (iy + l.cluster[1] * iz);
              b.first  = (b.rank == 0);
              b.eop    = (b.rank == k - 1);
              b.ctx_id = 0;
              beats.push_back(b);
            }
          }
        }
      }
    }
  }
  return beats;
}

static Beat sample(Sim& sim) {
  Beat b;
  b.idx[0] = sim->bus_block_idx_x;
  b.idx[1] = sim->bus_block_idx_y;
  b.idx[2] = sim->bus_block_idx_z;
  b.size   = sim->bus_cluster_size;
#if VX_CFG_EXT_DLMEM_ENABLED
  b.rank   = sim->bus_cluster_rank;
#else
  b.rank   = 0;
#endif
  b.first  = sim->bus_is_first;
  b.eop    = sim->bus_eop;
  b.ctx_id = sim->bus_ctx_id;
#if VX_CFG_EXT_DLMEM_ENABLED
  // a pinned beat or a gang field would come from the KMU, not the gang unit
  b.policy = sim->bus_policy | (sim->bus_pin << 1) | (sim->bus_gang_fields << 2);
#endif
  return b;
}

static bool same_beat(const Beat& a, const Beat& b) {
  return a.idx[0] == b.idx[0] && a.idx[1] == b.idx[1] && a.idx[2] == b.idx[2]
      && a.size == b.size && a.rank == b.rank && a.first == b.first
      && a.eop == b.eop && a.ctx_id == b.ctx_id && a.policy == b.policy;
}

// The policy register resets to DEFAULT and holds its value across launches.
static uint32_t policy_r = 0;

// Programs one launch, pulses start, and takes its beats under random
// back-pressure; every beat must match the reference walk, and a stalled beat
// must hold until it is taken.
static bool run_launch(Sim& sim, uint64_t& tick, std::mt19937& rng, const Launch& l,
                       uint32_t ready_pct, uint32_t& ctx_id) {
  const uint32_t dcr_grid[3]    = {VX_DCR_KMU_GRID_DIM_X, VX_DCR_KMU_GRID_DIM_Y, VX_DCR_KMU_GRID_DIM_Z};
  const uint32_t dcr_cluster[3] = {VX_DCR_KMU_CLUSTER_DIM_X, VX_DCR_KMU_CLUSTER_DIM_Y, VX_DCR_KMU_CLUSTER_DIM_Z};
  for (int i = 0; i < 3; ++i) {
    tick = write_dcr(sim, dcr_grid[i], l.grid[i], tick);
    tick = write_dcr(sim, dcr_cluster[i], l.cluster[i], tick);
  }
  tick = write_dcr(sim, VX_DCR_KMU_BLOCK_DIM_X, VX_CFG_NUM_THREADS, tick);
  tick = write_dcr(sim, VX_DCR_KMU_BLOCK_DIM_Y, 1, tick);
  tick = write_dcr(sim, VX_DCR_KMU_BLOCK_DIM_Z, 1, tick);
  tick = write_dcr(sim, VX_DCR_KMU_BLOCK_SIZE, VX_CFG_NUM_THREADS, tick);
  tick = write_dcr(sim, VX_DCR_KMU_WARP_STEP_X, VX_CFG_NUM_THREADS, tick);
#if VX_CFG_EXT_DLMEM_ENABLED
  if (l.policy >= 0) {
    tick = write_dcr(sim, VX_DCR_KMU_CLUSTER_POLICY, uint32_t(l.policy), tick);
    policy_r = uint32_t(l.policy) & 1;
  }
#endif

  sim->bus_ready = 0;
  sim->start = 1;
  tick = sim.step(tick, 2);
  sim->start = 0;
  ctx_id = (ctx_id + 1) & 0xff;

  std::vector<Beat> want = reference_walk(l);
  for (auto& b : want) {
    b.ctx_id = ctx_id;
    b.policy = policy_r;
  #if !VX_CFG_EXT_DLMEM_ENABLED
    b.rank = 0; // the launch message carries no rank
  #endif
  }
  uint32_t k = l.cluster[0] * l.cluster[1] * l.cluster[2];
  char name[64];
  std::snprintf(name, sizeof(name), "grid %ux%ux%u cluster %ux%ux%u",
                l.grid[0], l.grid[1], l.grid[2], l.cluster[0], l.cluster[1], l.cluster[2]);

  size_t got = 0;
  uint32_t errors = 0;
  uint32_t idle = 0;
  bool held = false;
  Beat held_beat{};
  while (got < want.size()) {
    bool ready = (rng() % 100) < ready_pct;
    sim->bus_ready = ready;
    sim->eval();
    if (!sim->bus_valid) {
      if (held && ++errors <= 10) {
        std::printf("FAIL %s: a stalled beat was withdrawn at beat %zu\n", name, got);
      }
      held = false;
      if (++idle > 1000) {
        std::printf("FAIL %s: no beat for 1000 cycles after %zu of %zu\n", name, got, want.size());
        return false;
      }
      tick = sim.step(tick, 2);
      continue;
    }
    idle = 0;
    Beat b = sample(sim);
    if (held && !same_beat(b, held_beat) && ++errors <= 10) {
      std::printf("FAIL %s: beat %zu changed while stalled\n", name, got);
    }
    if (ready) {
      const Beat& w = want[got];
      if (!same_beat(b, w) && ++errors <= 10) {
        std::printf("FAIL %s (K=%u): beat %zu block (%u,%u,%u) size %u rank %u first %d eop %d ctx %u policy %u;"
                    " want block (%u,%u,%u) size %u rank %u first %d eop %d ctx %u policy %u\n",
                    name, k, got, b.idx[0], b.idx[1], b.idx[2], b.size, b.rank, b.first, b.eop, b.ctx_id, b.policy,
                    w.idx[0], w.idx[1], w.idx[2], w.size, w.rank, w.first, w.eop, w.ctx_id, w.policy);
      }
      ++got;
      held = false;
    } else {
      held = true;
      held_beat = b;
    }
    tick = sim.step(tick, 2);
  }

  // The walk must stop after its last cluster and drop busy.
  sim->bus_ready = 1;
  for (int i = 0; i < 8; ++i) {
    sim->eval();
    if (sim->bus_valid && ++errors <= 10) {
      std::printf("FAIL %s: extra beat after the last cluster\n", name);
    }
    tick = sim.step(tick, 2);
  }
  sim->eval();
  if (sim->busy && ++errors <= 10) {
    std::printf("FAIL %s: still busy after the last cluster\n", name);
  }
  sim->bus_ready = 0;

  std::printf("%s (K=%u, ready %u%%, policy %u): %zu beats %s\n", name, k, ready_pct, policy_r, want.size(),
              errors ? "FAILED" : "ok");
  return errors == 0;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  uint32_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '+') {
      continue;
    }
    if (std::strncmp(argv[i], "--seed=", 7) == 0) {
      seed = static_cast<uint32_t>(std::strtoul(argv[i] + 7, nullptr, 0));
    } else {
      std::printf("usage: %s [--seed=N]\n", argv[0]);
      return 2;
    }
  }

  std::printf("kmu: warps=%u max cluster=%u dlmem=%d seed=%u\n", VX_CFG_NUM_WARPS, MAX_K,
              VX_CFG_EXT_DLMEM_ENABLED, seed);

  Sim sim;
  uint64_t tick = 0;
  sim->start         = 0;
  sim->bus_ready     = 0;
  sim->dcr_req_valid = 0;
  sim->dcr_req_rw    = 0;
  tick = sim.reset(tick);

  std::vector<Launch> launches = {
    {{3, 2, 2}, {1, 1, 1}},
    {{4, 3, 1}, {2, 1, 1}},
    {{4, 2, 2}, {2, 2, 1}},
    {{2, 4, 2}, {1, 2, 2}},
    {{2 * MAX_K, 2, 1}, {MAX_K, 1, 1}},
    {{1, MAX_K, 3}, {1, MAX_K, 1}},
    {{2, 1, 2 * MAX_K}, {1, 1, MAX_K}},
  };
  if (MAX_K >= 4) {
    launches.push_back({{4, MAX_K, 2}, {2, MAX_K / 2, 1}});
  }
  if (MAX_K >= 8) {
    launches.push_back({{4, 2, 4}, {2, 2, 2}});
  }
  // Rank restarts with each launch: end on a single-block cluster, then reuse K = 2.
  launches.push_back({{3, 1, 1}, {1, 1, 1}});
  launches.push_back({{2, 2, 1}, {2, 1, 1}});
#if VX_CFG_EXT_DLMEM_ENABLED
  // The first launch keeps the reset policy; then SPREAD, kept, DEFAULT, kept, SPREAD.
  for (size_t i = 1; i < launches.size(); ++i) {
    const int seq[] = {1, -1, 0, -1, 1};
    launches[i].policy = seq[(i - 1) % 5];
  }
#endif

  std::mt19937 rng(seed);
  uint32_t ctx_id = 0;
  bool ok = true;
  for (const auto& l : launches) {
    ok = run_launch(sim, tick, rng, l, 100, ctx_id) && ok;
    ok = run_launch(sim, tick, rng, l, 40, ctx_id) && ok;
  }

  if (!ok) {
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
