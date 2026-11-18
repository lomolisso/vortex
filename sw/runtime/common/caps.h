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

#ifndef VORTEX_RUNTIME_VX_CAPS_H
#define VORTEX_RUNTIME_VX_CAPS_H

// ============================================================================
// caps.h — shared device/ISA capability decoder for all runtime backends.
//
// The three 64-bit capability words (GPU_DEV_CAPS / GPU_ISA_CAPS /
// GPU_ORG_CAPS) are read from the Command Processor regfile at the
// CP-internal offsets below.
//
// CACHE_LINE_SIZE / CLOCK_RATE / PEAK_MEM_BW are NOT
// encoded in the caps words (they are platform/runtime-specific); for
// those ids decode_caps() returns false and the backend resolves them.
// ============================================================================

#include <cstdint>
#include <vortex2.h>   // VX_CAPS_* ids

namespace vortex {

// CP regfile offsets for the static capability words.
// Pass to a backend's cp_mmio_read(), which applies any platform base offset.
static constexpr uint32_t CP_REG_GPU_DEV_CAPS_LO = 0x018;
static constexpr uint32_t CP_REG_GPU_DEV_CAPS_HI = 0x01C;
static constexpr uint32_t CP_REG_GPU_ISA_CAPS_LO = 0x020;
static constexpr uint32_t CP_REG_GPU_ISA_CAPS_HI = 0x024;
static constexpr uint32_t CP_REG_GPU_ORG_CAPS_LO = 0x040;
static constexpr uint32_t CP_REG_GPU_ORG_CAPS_HI = 0x044;

// Runtime-internal id, not in vortex2.h: 1 when the CTAs of a cluster are
// always placed on distinct cores (H100 placement), 0 for today's placement.
static constexpr uint32_t CAPS_INTERNAL_CLUSTER_DISTINCT_SMS = 0x10000;

// GPU_ORG_CAPS fields: [3:0] log2 partitions, [7:4] log2 CPCs per GPC,
// [14:8] MAX_CTAS_PER_SM, [21:15] MAX_CTAS_PER_SM_CLUSTER, [22] distinct SMs,
// [31:23] reserved (compute), [63:32] reserved (memory side).

// Decode a VX_CAPS_* id from the CP capability words. Returns true
// and writes *out when the id is encoded in the words; returns false
// for ids the caller must resolve itself (CACHE_LINE_SIZE, CLOCK_RATE,
// PEAK_MEM_BW).
inline bool decode_caps(uint64_t dev_caps, uint64_t isa_caps, uint64_t org_caps,
                        uint32_t caps_id, uint64_t* out) {
  const uint64_t num_clusters = 1ull << ((dev_caps >> 20) & 0x7);
  const uint64_t cluster_size = 1ull << ((dev_caps >> 17) & 0x7);
  const uint64_t partitions   = 1ull << ((org_caps >> 0) & 0xf);
  const uint64_t cpcs         = 1ull << ((org_caps >> 4) & 0xf);
  switch (caps_id) {
  case VX_CAPS_VERSION:
    *out = (dev_caps >> 0) & 0xff;
    return true;
  case VX_CAPS_NUM_THREADS:
    *out = 1ull << ((dev_caps >> 8) & 0x7);
    return true;
  case VX_CAPS_NUM_WARPS:
    *out = 1ull << ((dev_caps >> 11) & 0x7);
    return true;
  case VX_CAPS_NUM_CORES:
    *out = num_clusters * cluster_size * (1ull << ((dev_caps >> 14) & 0x7));
    return true;
  case VX_CAPS_SOCKET_SIZE:
    *out = 1ull << ((dev_caps >> 14) & 0x7);
    return true;
  case VX_CAPS_NUM_CLUSTERS:
    *out = num_clusters;
    return true;
  case VX_CAPS_ISSUE_WIDTH:
    *out = 1ull << ((dev_caps >> 23) & 0x7);
    return true;
  case VX_CAPS_LOCAL_MEM_SIZE:
    *out = 1ull << ((dev_caps >> 26) & 0xff);
    return true;
  case VX_CAPS_ISA_FLAGS:
    *out = isa_caps;
    return true;
  case VX_CAPS_NUM_MEM_BANKS:
    *out = 1ull << ((dev_caps >> 34) & 0x7);
    return true;
  case VX_CAPS_MEM_BANK_SIZE:
    *out = 1ull << (20 + ((dev_caps >> 37) & 0x1f));
    return true;
  case VX_CAPS_GLOBAL_MEM_SIZE:
    *out = 1ull << (((dev_caps >> 34) & 0x7) + 20 + ((dev_caps >> 37) & 0x1f));
    return true;
  case VX_CAPS_DLMEM_MAX_CLUSTER:
    // A plain count, not a log2: the knob need not be a power of two.
    *out = (dev_caps >> 42) & 0x7f;
    return true;
  case VX_CAPS_NUM_PARTITIONS:
    *out = partitions;
    return true;
  case VX_CAPS_GPCS_PER_PARTITION:
    *out = num_clusters / partitions;
    return true;
  case VX_CAPS_CPCS_PER_GPC:
    *out = cpcs;
    return true;
  case VX_CAPS_TPCS_PER_CPC:
    *out = cluster_size / cpcs;
    return true;
  case VX_CAPS_SMS_PER_TPC:
    *out = 1ull << ((dev_caps >> 14) & 0x7);
    return true;
  case VX_CAPS_MAX_CTAS_PER_SM:
    *out = (org_caps >> 8) & 0x7f;
    return true;
  case VX_CAPS_MAX_CTAS_PER_SM_CLUSTER:
    *out = (org_caps >> 15) & 0x7f;
    return true;
  case CAPS_INTERNAL_CLUSTER_DISTINCT_SMS:
    *out = (org_caps >> 22) & 0x1;
    return true;
#ifdef VX_CFG_H100_SM_ENABLE
  // Build constants of the carve-out, as the runtime's carve-out choice reads them.
  case VX_CAPS_SMEM_PER_SM:
    *out = VX_CFG_SMEM_PER_SM_MAX;
    return true;
  case VX_CAPS_SMEM_PER_BLOCK_OPTIN:
    *out = VX_CFG_SMEM_PER_CTA_MAX;
    return true;
  case VX_CAPS_SMEM_RESERVED_PER_BLOCK:
    *out = 0;
    return true;
#else
  case VX_CAPS_SMEM_PER_SM:
  case VX_CAPS_SMEM_PER_BLOCK_OPTIN:
    *out = 1ull << ((dev_caps >> 26) & 0xff);
    return true;
  case VX_CAPS_SMEM_RESERVED_PER_BLOCK:
    *out = 0;
    return true;
#endif
  default:
    return false;
  }
}

// Read GPU_DEV_CAPS / GPU_ISA_CAPS / GPU_ORG_CAPS from the CP regfile through a
// backend's cp_mmio_read primitive. `Reader` is any callable matching
// int(uint32_t off, uint32_t* out) — returns non-zero on failure.
// Returns 0 on success.
template <typename Reader>
inline int load_caps(Reader&& cp_mmio_read,
                     uint64_t* dev_caps, uint64_t* isa_caps, uint64_t* org_caps) {
  uint32_t lo = 0, hi = 0;
  if (cp_mmio_read(CP_REG_GPU_DEV_CAPS_LO, &lo)) return -1;
  if (cp_mmio_read(CP_REG_GPU_DEV_CAPS_HI, &hi)) return -1;
  *dev_caps = uint64_t(lo) | (uint64_t(hi) << 32);
  if (cp_mmio_read(CP_REG_GPU_ISA_CAPS_LO, &lo)) return -1;
  if (cp_mmio_read(CP_REG_GPU_ISA_CAPS_HI, &hi)) return -1;
  *isa_caps = uint64_t(lo) | (uint64_t(hi) << 32);
  if (cp_mmio_read(CP_REG_GPU_ORG_CAPS_LO, &lo)) return -1;
  if (cp_mmio_read(CP_REG_GPU_ORG_CAPS_HI, &hi)) return -1;
  *org_caps = uint64_t(lo) | (uint64_t(hi) << 32);
  return 0;
}

} // namespace vortex

#endif // VORTEX_RUNTIME_VX_CAPS_H
