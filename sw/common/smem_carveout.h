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

#pragma once

// Golden model of the per-launch split of the SM's unified L1 / shared-memory
// array: which configured shared-memory size a launch gets, the CTA footprint,
// the resident-block bound and the L1 left over. Shared by the runtime, SimX,
// the testbenches and the host tests; smem_carve_cfg() reads VX_config.h, every
// other function takes the shape from SmemCarveCfg.

#include <cstdint>

#define SMEM_CARVE_NUM_CONFIGS 9

struct SmemCarveCfg {
  uint32_t configs_kb[SMEM_CARVE_NUM_CONFIGS];
  uint32_t sram_bytes;
  uint32_t reserve_bytes;
  uint32_t slot_align;
  uint32_t per_sm_max;
  uint32_t max_ctas_per_sm;
  uint32_t max_ctas_per_sm_cluster;
  uint32_t warps_per_sm;
  uint32_t threads_per_warp;
};

#ifdef VX_CFG_H100_SM_ENABLE
// The carve-out word is VX_CFG_LMEM_LOG_SIZE + 1 bits wide and every slot sits in the shared-memory window.
static_assert(uint64_t(VX_CFG_SMEM_PER_SM_MAX) <= (uint64_t(1) << VX_CFG_LMEM_LOG_SIZE),
              "VX_CFG_SMEM_PER_SM_MAX must fit the shared-memory window of 1 << VX_CFG_LMEM_LOG_SIZE bytes");
inline SmemCarveCfg smem_carve_cfg() {
  SmemCarveCfg cfg = {
    {VX_CFG_SMEM_CONFIG_KB_0, VX_CFG_SMEM_CONFIG_KB_1, VX_CFG_SMEM_CONFIG_KB_2,
     VX_CFG_SMEM_CONFIG_KB_3, VX_CFG_SMEM_CONFIG_KB_4, VX_CFG_SMEM_CONFIG_KB_5,
     VX_CFG_SMEM_CONFIG_KB_6, VX_CFG_SMEM_CONFIG_KB_7, VX_CFG_SMEM_CONFIG_KB_8},
    VX_CFG_SM_SRAM_BYTES,
    VX_CFG_L1_RESERVE_BYTES,
    VX_CFG_LMEM_SLOT_ALIGN,
    VX_CFG_SMEM_PER_SM_MAX,
    VX_CFG_MAX_CTAS_PER_SM,
    VX_CFG_MAX_CTAS_PER_SM_CLUSTER,
    VX_CFG_NUM_WARPS,
    VX_CFG_NUM_THREADS,
  };
  return cfg;
}
#endif

// A slot keeps the pitch the block addresses (DSMEM ranks and DXA multicast offset by it).
inline uint32_t smem_cta_footprint(uint32_t lmem_bytes, const SmemCarveCfg& cfg) {
  uint64_t a = cfg.slot_align ? cfg.slot_align : 1;
  uint64_t need = (uint64_t(lmem_bytes) + a - 1) / a * a;
  return (need > 0xffffffffull) ? 0xffffffffu : uint32_t(need);
}

// Every per-SM block limit other than shared memory; Vortex has no
// register-file limit.
inline uint32_t smem_blocks_other(uint32_t block_threads, bool clustered, const SmemCarveCfg& cfg) {
  uint32_t ctas = clustered ? cfg.max_ctas_per_sm_cluster : cfg.max_ctas_per_sm;
  uint32_t tpw = cfg.threads_per_warp ? cfg.threads_per_warp : 1;
  uint32_t warps = (block_threads + tpw - 1) / tpw;
  if (warps == 0)
    warps = 1;
  uint32_t by_warps = cfg.warps_per_sm / warps;
  return (by_warps < ctas) ? by_warps : ctas;
}

inline uint32_t smem_blocks_per_sm(uint32_t s_bytes, uint32_t lmem_bytes, uint32_t blocks_other,
                                   const SmemCarveCfg& cfg) {
  uint32_t fp = smem_cta_footprint(lmem_bytes, cfg);
  if (fp == 0)
    return blocks_other;
  uint32_t by_smem = s_bytes / fp;
  return (by_smem < blocks_other) ? by_smem : blocks_other;
}

// Slots the dispatcher and the cluster admission open to a launch at S; at
// least one, so a block larger than S still runs (and overruns, reported).
inline uint32_t smem_usable_slots(uint32_t s_bytes, uint32_t lmem_bytes, uint32_t slots,
                                  const SmemCarveCfg& cfg) {
  uint32_t fp = smem_cta_footprint(lmem_bytes, cfg);
  uint32_t fit = (fp == 0) ? slots : s_bytes / fp;
  return (fit > slots) ? slots : (fit == 0) ? 1 : fit;
}

// hint_pct: -1 = no hint, else 0..100. Returns S in bytes, or -1 when one
// block does not fit the largest configuration.
inline int32_t smem_config_choose(uint32_t lmem_bytes, uint32_t block_threads, bool clustered,
                                  int32_t hint_pct, const SmemCarveCfg& cfg) {
  const uint32_t n = SMEM_CARVE_NUM_CONFIGS;
  uint64_t need = smem_cta_footprint(lmem_bytes, cfg);
  uint64_t largest = uint64_t(cfg.configs_kb[n - 1]) * 1024;
  if (need > largest)
    return -1;
  if (hint_pct >= 0) {
    for (uint32_t i = 0; i < n; ++i) {
      uint64_t c = cfg.configs_kb[i];
      if (c * 1024 >= need && c * 100 >= uint64_t(hint_pct) * cfg.configs_kb[n - 1])
        return int32_t(c * 1024);
    }
    return int32_t(largest);
  }
  // The smallest configuration that still allows the most resident blocks.
  uint32_t other = smem_blocks_other(block_threads, clustered, cfg);
  uint32_t most = smem_blocks_per_sm(uint32_t(largest), lmem_bytes, other, cfg);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t c = cfg.configs_kb[i] * 1024;
    if (smem_blocks_per_sm(c, lmem_bytes, other, cfg) == most)
      return int32_t(c);
  }
  return int32_t(largest);
}

inline uint32_t smem_usable_l1(uint32_t s_bytes, const SmemCarveCfg& cfg) {
  return cfg.sram_bytes - cfg.reserve_bytes - s_bytes;
}

// Resident blocks at the largest configuration (the occupancy API's answer).
inline uint32_t smem_occupancy(uint32_t lmem_bytes, uint32_t block_threads, bool clustered,
                               const SmemCarveCfg& cfg) {
  uint32_t largest = cfg.configs_kb[SMEM_CARVE_NUM_CONFIGS - 1] * 1024;
  return smem_blocks_per_sm(largest, lmem_bytes, smem_blocks_other(block_threads, clustered, cfg), cfg);
}
