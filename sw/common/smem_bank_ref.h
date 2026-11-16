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

// Golden model of the H100-shaped shared memory: banked wavefronts, latency
// and atomic passes of one warp request. Shared by SimX, the testbenches and
// the host tests; shapes come from SmemCfg, never VX_config.h.

#include <cstdint>

namespace vortex {
namespace smem_bank_ref {

struct SmemCfg {
  uint32_t banks;        // power of two
  uint32_t bank_bytes;   // power of two
  uint32_t lat32;        // one-thread chase, accesses up to 4 B
  uint32_t lat64;        // one-thread chase, 8-B accesses
  uint32_t penalty;      // per extra wavefront or atomic pass
  uint32_t wide_off64;   // 8-B accesses touching more than 2 distinct words
  uint32_t atom_extra;
  uint32_t cas_extra;
  uint32_t cas_passes;
};

namespace detail {

constexpr uint32_t MAX_LANES = 32;
// Bank words one lane can touch: an 8-B lane on 1-B banks.
constexpr uint32_t MAX_WORDS_PER_LANE = 8;

inline uint32_t lane_words(uint64_t addr, uint32_t access_bytes, uint32_t bank_bytes, uint64_t* out) {
  uint64_t first = addr / bank_bytes;
  uint64_t last  = (addr + access_bytes - 1) / bank_bytes;
  uint32_t n = 0;
  for (uint64_t w = first; w <= last && n < MAX_WORDS_PER_LANE; ++w) {
    out[n++] = w;
  }
  return n;
}

} // namespace detail

// D: the largest number of distinct bank words one bank serves, over every
// byte of every active lane; lanes on the same word count once (broadcast).
inline uint32_t smem_wavefronts(const uint64_t* addr, uint32_t lane_mask, uint32_t nlanes,
                                uint32_t access_bytes, const SmemCfg& cfg) {
  uint64_t words[detail::MAX_LANES * detail::MAX_WORDS_PER_LANE];
  uint32_t nwords = 0;
  for (uint32_t i = 0; i < nlanes && i < detail::MAX_LANES; ++i) {
    if (!((lane_mask >> i) & 1))
      continue;
    uint64_t lw[detail::MAX_WORDS_PER_LANE];
    uint32_t n = detail::lane_words(addr[i], access_bytes, cfg.bank_bytes, lw);
    for (uint32_t k = 0; k < n; ++k) {
      bool seen = false;
      for (uint32_t j = 0; j < nwords; ++j) {
        if (words[j] == lw[k]) { seen = true; break; }
      }
      if (!seen)
        words[nwords++] = lw[k];
    }
  }
  uint32_t d = 0;
  for (uint32_t b = 0; b < cfg.banks; ++b) {
    uint32_t cnt = 0;
    for (uint32_t j = 0; j < nwords; ++j) {
      if ((words[j] & (cfg.banks - 1)) == b)
        ++cnt;
    }
    if (cnt > d)
      d = cnt;
  }
  return d;
}

// The distinct access_bytes-sized words the active lanes touch.
inline uint32_t smem_distinct_words(const uint64_t* addr, uint32_t lane_mask, uint32_t nlanes,
                                    uint32_t access_bytes) {
  uint64_t words[detail::MAX_LANES];
  uint32_t n = 0;
  for (uint32_t i = 0; i < nlanes && i < detail::MAX_LANES; ++i) {
    if (!((lane_mask >> i) & 1))
      continue;
    uint64_t w = addr[i] / access_bytes;
    bool seen = false;
    for (uint32_t j = 0; j < n; ++j) {
      if (words[j] == w) { seen = true; break; }
    }
    if (!seen)
      words[n++] = w;
  }
  return n;
}

// Dependent-chase time of a load served in `wavefronts` wavefronts.
inline uint32_t smem_load_latency(uint32_t wavefronts, uint32_t access_bytes,
                                  uint32_t distinct_words, const SmemCfg& cfg) {
  uint32_t d = (wavefronts == 0) ? 1 : wavefronts;
  bool wide = (access_bytes == 8);
  return (wide ? cfg.lat64 : cfg.lat32)
       + cfg.penalty * (d - 1)
       + ((wide && distinct_words > 2) ? cfg.wide_off64 : 0);
}

// Passes of a warp atomic: the most lane-updates one bank performs, every
// active lane counted (no broadcast), an 8-B lane in both of its banks, CAS
// lanes cas_passes times.
inline uint32_t smem_atom_passes(const uint64_t* addr, uint32_t lane_mask, uint32_t nlanes,
                                 uint32_t access_bytes, bool cas, const SmemCfg& cfg) {
  uint32_t per_bank[256] = {0};
  uint32_t banks = (cfg.banks > 256) ? 256 : cfg.banks;
  for (uint32_t i = 0; i < nlanes && i < detail::MAX_LANES; ++i) {
    if (!((lane_mask >> i) & 1))
      continue;
    uint64_t lw[detail::MAX_WORDS_PER_LANE];
    uint32_t n = detail::lane_words(addr[i], access_bytes, cfg.bank_bytes, lw);
    for (uint32_t k = 0; k < n; ++k) {
      per_bank[lw[k] & (banks - 1)] += 1;
    }
  }
  uint32_t p = 0;
  for (uint32_t b = 0; b < banks; ++b) {
    if (per_bank[b] > p)
      p = per_bank[b];
  }
  return cas ? p * cfg.cas_passes : p;
}

// Dependent-chase time of a warp atomic of `passes` passes, relative to a
// one-lane op of the same kind.
inline uint32_t smem_atom_latency(uint32_t passes, bool cas, const SmemCfg& cfg) {
  uint32_t p1 = cas ? cfg.cas_passes : 1;
  uint32_t p = (passes < p1) ? p1 : passes;
  return cfg.lat32 + (cas ? cfg.cas_extra : cfg.atom_extra) + cfg.penalty * (p - p1);
}

} // namespace smem_bank_ref
} // namespace vortex
