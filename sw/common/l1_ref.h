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

// Golden model of the H100-shaped L1 data cache (sectored lines, per-byte
// valid, store allocate, XOR-folded set index, newest-protected random
// replacement) and of its warp-request timing, for SimX, the testbenches and
// the host tests. Shapes come from arguments, never VX_config.h.

#include <cstdint>
#include <vector>

namespace vortex {
namespace l1_ref {

// Every line-address bit above the set field folded (XOR) into it.
inline uint32_t l1_set_index(uint64_t line_addr, uint32_t sets_log2) {
  if (sets_log2 == 0)
    return 0;
  uint64_t mask = (uint64_t(1) << sets_log2) - 1;
  uint64_t idx = 0;
  while (line_addr != 0) {
    idx ^= line_addr & mask;
    line_addr >>= sets_log2;
  }
  return uint32_t(idx);
}

// P: the sum over the address passes of the distinct lines each pass touches.
inline uint32_t l1_line_passes(const uint64_t* addr, uint32_t lane_mask, uint32_t nlanes,
                               uint32_t passes, uint32_t line_bytes) {
  uint32_t per_pass = nlanes / passes;
  uint32_t total = 0;
  for (uint32_t p = 0; p < passes; ++p) {
    std::vector<uint64_t> seen;
    for (uint32_t i = p * per_pass; i < (p + 1) * per_pass; ++i) {
      if (!((lane_mask >> i) & 1))
        continue;
      uint64_t line = addr[i] / line_bytes;
      bool found = false;
      for (auto l : seen)
        found |= (l == line);
      if (!found)
        seen.push_back(line);
    }
    total += uint32_t(seen.size());
  }
  return total;
}

// Distinct lines of the whole warp request: cycles the data port is busy.
inline uint32_t l1_distinct_lines(const uint64_t* addr, uint32_t lane_mask, uint32_t nlanes,
                                  uint32_t passes, uint32_t line_bytes) {
  (void)passes;
  std::vector<uint64_t> seen;
  for (uint32_t i = 0; i < nlanes; ++i) {
    if (!((lane_mask >> i) & 1))
      continue;
    uint64_t line = addr[i] / line_bytes;
    bool found = false;
    for (auto l : seen)
      found |= (l == line);
    if (!found)
      seen.push_back(line);
  }
  return uint32_t(seen.size());
}

inline uint32_t l1_hit_latency(uint32_t P, uint32_t base_centi, uint32_t step_centi) {
  return (base_centi + step_centi * P + 50) / 100;
}

// 16-bit Galois LFSR, x^16 + x^14 + x^13 + x^11 + 1, one per bank.
constexpr uint32_t L1_LFSR_SEED = 0xACE1;
constexpr uint32_t L1_LFSR_TAPS = 0xB400;

struct L1Lfsr {
  uint32_t s = L1_LFSR_SEED;
  uint32_t next() {
    uint32_t lsb = s & 1;
    s >>= 1;
    if (lsb)
      s ^= L1_LFSR_TAPS;
    return s;
  }
};

// insert_rank: 0 = newest insertion of the set. Candidates are the ways of
// rank >= protect; the pick is candidate number rnd % count, in way order.
inline uint32_t l1_victim(const uint8_t* insert_rank, uint32_t ways, uint32_t protect, uint32_t rnd) {
  uint32_t count = 0;
  for (uint32_t w = 0; w < ways; ++w)
    count += (insert_rank[w] >= protect);
  uint32_t pick = rnd % count;
  for (uint32_t w = 0; w < ways; ++w) {
    if (insert_rank[w] >= protect) {
      if (pick == 0)
        return w;
      --pick;
    }
  }
  return 0;
}

// The ranks stay a permutation of 0..ways-1 (reset: rank = way).
inline void l1_insert(uint8_t* insert_rank, uint32_t ways, uint32_t way) {
  uint8_t old = insert_rank[way];
  for (uint32_t w = 0; w < ways; ++w) {
    if (insert_rank[w] < old)
      ++insert_rank[w];
  }
  insert_rank[way] = 0;
}

inline uint32_t l1_sector_mask(uint32_t line_off, uint32_t bytes, uint32_t sector_bytes) {
  if (bytes == 0)
    return 0;
  uint32_t first = line_off / sector_bytes;
  uint32_t last = (line_off + bytes - 1) / sector_bytes;
  uint32_t mask = 0;
  for (uint32_t s = first; s <= last; ++s)
    mask |= 1u << s;
  return mask;
}

// Functional L1: a miss fills at once (the testbenches drive one access at a
// time). An insertion is any line allocation (a fill of an absent line or a
// store to one); each steps its bank's LFSR once, whether or not the victim is
// an invalid way. Invalid ways are taken first, lowest way first.
class L1Ref {
public:
  struct Result {
    bool     hit;
    bool     inserted;
    uint32_t bank;
    uint32_t set;
    uint32_t way;
    uint32_t sector_mask;  // sectors a load miss requests (0 on a hit or a store)
  };

  L1Ref(uint32_t size, uint32_t ways, uint32_t line_bytes, uint32_t sector_bytes,
        uint32_t banks, uint32_t protect)
    : ways_(ways), line_(line_bytes), sector_(sector_bytes), banks_(banks), protect_(protect) {
    sets_ = size / (ways * line_bytes * banks);
    sets_log2_ = 0;
    while ((1u << sets_log2_) < sets_)
      ++sets_log2_;
    lfsr_.resize(banks);
    lines_.resize(uint64_t(banks) * sets_ * ways);
    rank_.resize(uint64_t(banks) * sets_ * ways);
    for (uint32_t b = 0; b < banks; ++b) {
      for (uint32_t s = 0; s < sets_; ++s) {
        for (uint32_t w = 0; w < ways; ++w) {
          auto& ln = this->line(b, s, w);
          ln.valid = false;
          ln.bytes.assign(line_bytes, 0);
          rank_[idx(b, s, w)] = uint8_t(w);
        }
      }
    }
  }

  uint32_t sets() const { return sets_; }
  uint32_t lfsr_state(uint32_t bank) const { return lfsr_[bank].s; }
  void set_lfsr_state(uint32_t bank, uint32_t s) { lfsr_[bank].s = s; }

  // byteen: one flag per byte of the line, from line offset 0.
  Result load(uint64_t line_addr, const std::vector<uint8_t>& byteen) {
    Result r = locate(line_addr);
    int w = find(r.bank, r.set, line_addr);
    if (w >= 0) {
      r.way = uint32_t(w);
      auto& ln = line(r.bank, r.set, r.way);
      r.sector_mask = missing_sectors(ln, byteen);
      r.hit = (r.sector_mask == 0);
    } else {
      r.sector_mask = missing_sectors_all(byteen);
      r.way = insert(r.bank, r.set, line_addr);
      r.inserted = true;
    }
    if (!r.hit) {
      auto& ln = line(r.bank, r.set, r.way);
      for (uint32_t s = 0; s < line_ / sector_; ++s) {
        if ((r.sector_mask >> s) & 1) {
          for (uint32_t i = s * sector_; i < (s + 1) * sector_; ++i)
            ln.bytes[i] = 1;
        }
      }
    }
    return r;
  }

  Result store(uint64_t line_addr, const std::vector<uint8_t>& byteen) {
    Result r = locate(line_addr);
    int w = find(r.bank, r.set, line_addr);
    if (w >= 0) {
      r.way = uint32_t(w);
      r.hit = true;
    } else {
      r.way = insert(r.bank, r.set, line_addr);
      r.inserted = true;
    }
    auto& ln = line(r.bank, r.set, r.way);
    for (uint32_t i = 0; i < line_; ++i)
      ln.bytes[i] |= byteen[i];
    return r;
  }

  void invalidate_all() {
    for (auto& ln : lines_) {
      ln.valid = false;
      ln.bytes.assign(line_, 0);
    }
  }

private:
  struct Line {
    bool                 valid;
    uint64_t             line_addr;
    std::vector<uint8_t> bytes;
  };

  uint64_t idx(uint32_t b, uint32_t s, uint32_t w) const {
    return (uint64_t(b) * sets_ + s) * ways_ + w;
  }
  Line& line(uint32_t b, uint32_t s, uint32_t w) { return lines_[idx(b, s, w)]; }

  Result locate(uint64_t line_addr) const {
    Result r{};
    r.bank = uint32_t(line_addr % banks_);
    r.set = l1_set_index(line_addr / banks_, sets_log2_);
    return r;
  }

  int find(uint32_t b, uint32_t s, uint64_t line_addr) {
    for (uint32_t w = 0; w < ways_; ++w) {
      auto& ln = line(b, s, w);
      if (ln.valid && ln.line_addr == line_addr)
        return int(w);
    }
    return -1;
  }

  uint32_t insert(uint32_t b, uint32_t s, uint64_t line_addr) {
    uint32_t rnd = lfsr_[b].next();
    uint32_t way = ways_;
    for (uint32_t w = 0; w < ways_; ++w) {
      if (!line(b, s, w).valid) {
        way = w;
        break;
      }
    }
    if (way == ways_)
      way = l1_victim(&rank_[idx(b, s, 0)], ways_, protect_, rnd);
    l1_insert(&rank_[idx(b, s, 0)], ways_, way);
    auto& ln = line(b, s, way);
    ln.valid = true;
    ln.line_addr = line_addr;
    ln.bytes.assign(line_, 0);
    return way;
  }

  uint32_t missing_sectors(const Line& ln, const std::vector<uint8_t>& byteen) const {
    uint32_t mask = 0;
    for (uint32_t i = 0; i < line_; ++i) {
      if (byteen[i] && !ln.bytes[i])
        mask |= 1u << (i / sector_);
    }
    return mask;
  }

  uint32_t missing_sectors_all(const std::vector<uint8_t>& byteen) const {
    uint32_t mask = 0;
    for (uint32_t i = 0; i < line_; ++i) {
      if (byteen[i])
        mask |= 1u << (i / sector_);
    }
    return mask;
  }

  uint32_t ways_, line_, sector_, banks_, protect_;
  uint32_t sets_, sets_log2_;
  std::vector<L1Lfsr>  lfsr_;
  std::vector<Line>    lines_;
  std::vector<uint8_t> rank_;
};

} // namespace l1_ref
} // namespace vortex
