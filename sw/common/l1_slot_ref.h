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

// Golden model of the L1 that lives in the SM's unified SRAM array: tags
// decoupled from data slots, the slot pool (bump pointer, free FIFO, age list
// of a young FIFO and an old table), per-word valid bits, the tag-bank ports
// and the warp-request timing. Shared by SimX, the testbenches and the host
// tests; shapes come from L1SlotCfg, never VX_config.h.

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <unordered_map>
#include <vector>
#include "l1_ref.h"

namespace vortex {
namespace l1_slot_ref {

using L1Lfsr = l1_ref::L1Lfsr;

inline uint32_t log2_floor(uint32_t x) {
  uint32_t n = 0;
  while ((x >> (n + 1)) != 0)
    ++n;
  return n;
}

inline uint32_t log2_ceil(uint32_t x) {
  uint32_t n = 0;
  while ((uint64_t(1) << n) < x)
    ++n;
  return n;
}

// hash 0: line mod sets; hash 1: every line-address bit XOR-folded into the set field.
inline uint32_t l1_set(uint64_t line, uint32_t sets, uint32_t hash) {
  if (hash == 0)
    return uint32_t(line & (sets - 1));
  return l1_ref::l1_set_index(line, log2_ceil(sets));
}

// XOR of the set's bank-width groups: for a fixed bank, set -> set / banks is a bijection.
inline uint32_t l1_tag_bank(uint32_t set, uint32_t banks) {
  if (banks <= 1)
    return 0;
  uint32_t w = log2_ceil(banks);
  uint32_t b = 0;
  while (set != 0) {
    b ^= set & (banks - 1);
    set >>= w;
  }
  return b;
}

inline uint32_t l1_tag_row(uint32_t set, uint32_t banks) {
  return set / banks;
}

// Register stages an L1 hit passes outside the tag batches and the wavefront
// stage, from its issue to the dependent instruction's issue (VX_gpu_pkg
// L1_PIPE_OVERHEAD): issue, operand collection (+ routing between collectors)
// and the LSU 10, the local-memory switch 2, the tag stage's input 1 and its
// registered read and compare 2, the commit FIFO 1, the lane data to the LSU 1.
constexpr uint32_t l1_pipe_stages(uint32_t num_opcs) {
  return 10 + ((num_opcs > 1) ? 1 : 0) + 2 + 1 + 2 + 1 + 1;
}

// The array's response register after the wavefront stage's last cycle.
constexpr uint32_t L1_ARRAY_RSP_STAGES = 1;

// Register stages a miss passes outside the lower levels, from its issue to the
// dependent instruction's issue (VX_gpu_pkg L1_MISS_PIPE_OVERHEAD): a hit's
// stages less the commit FIFO, plus the MSHR allocate and egress register 2,
// the fill register 1, the slot write and the slot read 2, the response register.
constexpr uint32_t l1_miss_pipe_stages(uint32_t num_opcs) {
  return l1_pipe_stages(num_opcs) - 1 + 2 + 1 + 2 + L1_ARRAY_RSP_STAGES;
}

// Dependent-chase time of a warp load hitting the L1: the stages, one tag
// cycle per batch (a tag bank looks up one line a cycle), one wavefront-stage
// cycle per wavefront over the lines' slot rows, and the response register.
inline uint32_t l1_hit_latency(uint32_t wavefronts, uint32_t tag_cycles, uint32_t pipe) {
  return pipe + (tag_cycles - 1) + ((wavefronts == 0) ? 1 : wavefronts) + L1_ARRAY_RSP_STAGES;
}

struct L1SlotCfg {
  uint32_t s_bytes;        // shared-memory carve-out of the launch
  uint32_t sram_bytes;     // the whole array
  uint32_t reserve_bytes;  // rows between the shared region and the L1
  uint32_t row_bytes;      // one array row = one L1 line
  uint32_t valid_bytes;    // bytes per valid bit
  uint32_t sector_bytes;
  uint32_t tag_sets;
  uint32_t tag_ways;
  uint32_t tag_banks;
  uint32_t index_hash;
  uint32_t upd_depth;      // per-tag-bank update queue
  uint32_t protect_pct;
  uint32_t mshr_lines;     // lines in flight at most (SM_MAX_READ_LINES)
  uint32_t max_lines;      // the largest L1 over the configured carve-outs (sizes the young FIFO and the old table)
};

inline uint32_t l1_lines(const L1SlotCfg& c) {
  return (c.sram_bytes - c.reserve_bytes - c.s_bytes) / c.row_bytes;
}

inline uint32_t l1_protect_of(uint32_t lines, const L1SlotCfg& c) {
  uint32_t p = c.protect_pct * lines / 100;
  return (p > c.mshr_lines) ? p : c.mshr_lines;
}

inline uint32_t l1_protect(const L1SlotCfg& c) {
  return l1_protect_of(l1_lines(c), c);
}

// The young FIFO holds L1_PROTECT_MAX + 1 entries.
inline uint32_t l1_protect_max(const L1SlotCfg& c) {
  return l1_protect_of(c.max_lines, c);
}

inline uint32_t l1_old_depth(const L1SlotCfg& c) {
  return 1u << log2_ceil(c.max_lines);
}

inline uint32_t l1_valid_bits(const L1SlotCfg& c) {
  return c.row_bytes / c.valid_bytes;
}

// Valid-bit mask of the words a sector mask covers.
inline uint64_t l1_sector_words(uint32_t sector_mask, const L1SlotCfg& c) {
  uint32_t per = c.sector_bytes / c.valid_bytes;
  uint64_t m = 0;
  for (uint32_t s = 0; s < c.row_bytes / c.sector_bytes; ++s) {
    if ((sector_mask >> s) & 1)
      m |= ((per >= 64) ? ~uint64_t(0) : ((uint64_t(1) << per) - 1)) << (s * per);
  }
  return m;
}

// Valid bits a byte range sets when written: only the words it covers completely.
inline uint64_t l1_full_words(uint32_t off, uint32_t bytes, const L1SlotCfg& c) {
  uint64_t m = 0;
  for (uint32_t w = 0; w < l1_valid_bits(c); ++w) {
    uint32_t lo = w * c.valid_bytes, hi = lo + c.valid_bytes;
    if (lo >= off && hi <= off + bytes)
      m |= uint64_t(1) << w;
  }
  return m;
}

// Valid bits a byte range needs to hit: every word it touches.
inline uint64_t l1_touched_words(uint32_t off, uint32_t bytes, const L1SlotCfg& c) {
  uint64_t m = 0;
  if (bytes == 0)
    return 0;
  for (uint32_t w = off / c.valid_bytes; w <= (off + bytes - 1) / c.valid_bytes; ++w)
    m |= uint64_t(1) << w;
  return m;
}

// Cycles the tag stage spends on one warp request: the most distinct lines in one tag bank, at least 1.
inline uint32_t l1_tag_cycles(const uint64_t* lines, uint32_t n, uint32_t banks, const L1SlotCfg& c) {
  std::vector<uint64_t> seen;
  std::vector<uint32_t> per_bank(banks, 0);
  uint32_t worst = 1;
  for (uint32_t i = 0; i < n; ++i) {
    bool dup = false;
    for (auto l : seen)
      dup |= (l == lines[i]);
    if (dup)
      continue;
    seen.push_back(lines[i]);
    uint32_t b = l1_tag_bank(l1_set(lines[i], c.tag_sets, c.index_hash), banks);
    if (++per_bank[b] > worst)
      worst = per_bank[b];
  }
  return worst;
}

// The slot directory: tags, slot pool, age list and the open (MSHR) entries.
// Call order within a cycle: a pending age-list draw's allocate() first, then
// the cycle's other calls in tag-stage order, then replay_done() (a slot
// returned in a cycle is taken from the next), then tick(). Lookups see the tag
// state as of the cycle start plus updates queued in earlier cycles; an
// allocation's way choice sees every earlier allocation at once.
class L1SlotDir {
public:
  static constexpr uint32_t NONE = ~0u;

  enum Kind { K_RESIDENT, K_FREE, K_BUMP, K_INSET, K_AGE, K_BORROW };
  enum Wait { W_NONE, W_YOUNG_FULL, W_OLD_EMPTY, W_VICTIM_IN_FLIGHT, W_DRAW_PENDING,
              W_SET_IN_FLIGHT, W_ENTRY_OPEN, W_N };

  struct Ref {
    uint32_t slot = NONE, set = NONE, way = NONE;
  };

  struct Lookup {
    bool tag = false;      // the line has a way
    bool hit = false;      // and every requested word is valid
    bool pending = false;  // an open entry holds the line
    uint32_t slot = NONE;
    uint32_t set = 0, way = NONE;
  };

  struct Alloc {
    bool ok = false;
    Wait wait = W_NONE;
    Kind kind = K_RESIDENT;
    uint32_t slot = NONE, set = 0, way = NONE;
    Ref victim;            // age-list or in-set victim, slot == NONE when none
    uint64_t victim_line = 0;
  };

  void configure(const L1SlotCfg& c) {
    cfg_ = c;
    lines_ = l1_lines(c);
    protect_ = l1_protect(c);
    young_cap_ = l1_protect_max(c) + 1;
    old_.assign(l1_old_depth(c), Ref{});
    ways_.assign(size_t(c.tag_sets) * c.tag_ways, Way{});
    slots_.assign(c.max_lines, Slot{});
    if (lines_ > c.max_lines || lines_ <= c.mshr_lines)
      std::abort();
    invalidate_all();
  }

  // Whole-L1 clear: the caller has drained every open entry.
  void invalidate_all() {
    if (!entries_.empty())
      std::abort();
    for (auto& w : ways_)
      w = Way{};
    for (auto& s : slots_)
      s = Slot{};
    bump_ = 0;
    free_.clear();
    young_.clear();
    n_old_ = 0;
    borrowed_ = 0;
    draw_ = Draw{};
    overlay_.clear();
    swapped_ = false;
    want_migrate_ = false;
  }

  const L1SlotCfg& cfg() const { return cfg_; }
  uint64_t now() const { return now_; }
  uint32_t lfsr_state() const { return lfsr_.s; }
  void set_lfsr_state(uint32_t s) { lfsr_.s = s; }

  uint32_t set_of(uint64_t line) const { return l1_set(line, cfg_.tag_sets, cfg_.index_hash); }
  uint32_t bank_of(uint64_t line) const { return l1_tag_bank(set_of(line), cfg_.tag_banks); }

  // need: valid bits of the words the lanes read. A pending line may still
  // hit on its valid words; the caller joins it to the entry's replays when
  // the entry holds a queued store.
  Lookup lookup(uint64_t line, uint64_t need) const {
    Lookup r;
    r.set = set_of(line);
    for (auto& o : overlay_) {
      if (o.set == r.set && o.old.valid && o.old.line == line) {
        r.tag = true;
        r.way = o.way;
        r.slot = o.old.slot;
        r.hit = (o.old.vmask & need) == need;
      }
    }
    if (!r.tag) {
      int w = find(r.set, line);
      if (w >= 0 && !fresh(r.set, uint32_t(w))) {
        auto& x = way(r.set, uint32_t(w));
        r.tag = true;
        r.way = uint32_t(w);
        r.slot = x.slot;
        r.hit = (x.vmask & need) == need;
      }
    }
    auto e = entries_.find(line);
    if (e != entries_.end()) {
      r.pending = true;
      if (!r.tag)
        r.slot = e->second.slot;
    }
    return r;
  }

  // A miss: opens the line's entry (sector_mask != 0) and, if the line has no
  // way, allocates a way and a slot. sector_mask == 0 is a store-allocate: a
  // way and a slot, no entry. seq: the fill interlock number recorded.
  Alloc allocate(uint64_t line, bool cacheable, uint32_t seq, uint32_t sector_mask) {
    Alloc a;
    a.set = set_of(line);
    if (entries_.count(line))
      return wait(a, W_ENTRY_OPEN);
    if (draw_.active) {
      if (draw_.line != line || draw_.cycle == now_)
        return wait(a, W_DRAW_PENDING);
      return complete_draw(a, line, seq, sector_mask);
    }
    int w = find(a.set, line);
    if (w >= 0) {
      auto& x = way(a.set, uint32_t(w));
      a.ok = true;
      a.kind = K_RESIDENT;
      a.way = uint32_t(w);
      a.slot = x.slot;
      open(line, a.slot, a.set, a.way, sector_mask, x.vmask, seq, true);
      return a;
    }
    if (!cacheable && (!free_.empty() || bump_ < lines_)) {
      a.ok = true;
      a.kind = K_BORROW;
      a.slot = take_free(a.kind);
      a.kind = K_BORROW;
      slots_[a.slot].where = S_BORROWED;
      ++borrowed_;
      open(line, a.slot, a.set, NONE, sector_mask, 0, seq, false);
      return a;
    }
    // the way: an invalid one, else an in-set victim not in flight
    int inv = -1;
    for (uint32_t i = 0; i < cfg_.tag_ways && inv < 0; ++i) {
      if (!way(a.set, i).valid)
        inv = int(i);
    }
    if (inv < 0) {
      std::vector<uint32_t> cand;
      for (uint32_t i = 0; i < cfg_.tag_ways; ++i) {
        if (!in_flight(way(a.set, i).slot))
          cand.push_back(i);
      }
      if (cand.empty())
        return wait(a, W_SET_IN_FLIGHT);
      uint32_t v = cand[lfsr_.next() % cand.size()];
      auto& x = way(a.set, v);
      a.ok = true;
      a.kind = K_INSET;
      a.way = v;
      a.slot = x.slot;
      a.victim = Ref{x.slot, a.set, v};
      a.victim_line = x.line;
      overlay_.push_back(Overlay{a.set, v, x});
      x = Way{true, line, x.slot, 0, now_};
      slots_[a.slot].line = line;
      open(line, a.slot, a.set, v, sector_mask, 0, seq, true);
      return a;
    }
    a.way = uint32_t(inv);
    if (!free_.empty() || bump_ < lines_) {
      if (young_.size() >= young_cap_)
        return wait(a, W_YOUNG_FULL);
      a.slot = take_free(a.kind);
      young_.push_back(Ref{a.slot, a.set, a.way});
      slots_[a.slot].where = S_YOUNG;
      a.ok = true;
      install(line, a, sector_mask, seq);
      return a;
    }
    return start_draw(a, line);
  }

  // Requested sectors' words minus those valid when the entry opened.
  uint64_t fill_byteen(uint32_t slot) const {
    auto& e = entry_of(slot);
    return l1_sector_words(e.sectors, cfg_) & ~e.valid_at_miss;
  }

  void fill(uint32_t slot, uint32_t sector_mask) {
    entry_of(slot).filled |= sector_mask;
  }

  // The fill's valid bits are written; the slot stays in flight.
  void fill_done(uint32_t slot) {
    auto& e = entry_of(slot);
    if (e.cacheable) {
      auto& x = way(e.set, e.way);
      if (x.valid && x.slot == slot)
        x.vmask |= l1_sector_words(e.filled, cfg_);
    }
  }

  // The entry's last replay left the wavefront stage: it retires.
  void replay_done(uint32_t slot) {
    auto& s = slots_[slot];
    auto it = entries_.find(s.entry_line);
    if (!s.open || it == entries_.end())
      std::abort();
    bool cacheable = it->second.cacheable;
    entries_.erase(it);
    s.open = false;
    if (!cacheable) {
      s.where = S_FREE;
      free_.push_back(slot);
      --borrowed_;
    }
  }

  // At a store's lookup: the fully written words become valid.
  void store(uint64_t line, uint64_t words) {
    int w = find(set_of(line), line);
    if (w >= 0)
      way(set_of(line), uint32_t(w)).vmask |= words;
  }

  // A global atomic at its lookup: valid bits cleared, tag and slot kept.
  void clear(uint64_t line) {
    int w = find(set_of(line), line);
    if (w >= 0)
      way(set_of(line), uint32_t(w)).vmask = 0;
  }

  void tick() {
    // An in-flight head migrates too: a slot reused in-set while queued
    // stays in flight wherever its entry sits, so old must tolerate in-flight
    // entries anyway (a drawn one cancels the draw).
    if (!swapped_ && young_.size() > 0 && (young_.size() > protect_ || want_migrate_)) {
      auto h = young_.front();
      young_.pop_front();
      old_[n_old_] = h;
      slots_[h.slot].where = S_OLD;
      slots_[h.slot].old_idx = n_old_++;
    }
    want_migrate_ = false;
    swapped_ = false;
    overlay_.clear();
    ++now_;
  }

  uint32_t lines() const { return lines_; }
  uint32_t protect() const { return protect_; }
  uint32_t resident() const { return uint32_t(young_.size()) + n_old_; }
  uint32_t free_slots() const { return uint32_t(free_.size()) + (lines_ - bump_); }
  uint32_t free_fifo() const { return uint32_t(free_.size()); }
  uint32_t bump() const { return bump_; }
  uint32_t borrowed() const { return borrowed_; }
  uint32_t young() const { return uint32_t(young_.size()); }
  uint32_t old() const { return n_old_; }
  uint32_t open_entries() const { return uint32_t(entries_.size()); }
  uint32_t row_of(uint32_t slot) const { return cfg_.sram_bytes / cfg_.row_bytes - 1 - slot; }
  bool in_flight(uint32_t slot) const { return slot != NONE && slots_[slot].open; }
  // Position of a slot in allocation order among young entries (0 = newest), NONE when old.
  uint32_t young_rank(uint32_t slot) const {
    for (size_t i = 0; i < young_.size(); ++i) {
      if (young_[young_.size() - 1 - i].slot == slot)
        return uint32_t(i);
    }
    return NONE;
  }
  uint64_t line_of(uint32_t slot) const { return slots_[slot].line; }
  uint32_t last_draw_index() const { return last_r_; }
  uint32_t last_draw_old() const { return last_n_; }

  // Pool invariants; returns 0 when they hold, else the number of the one broken.
  int check() const {
    if (uint32_t(young_.size()) + n_old_ + borrowed_ + uint32_t(free_.size()) + (lines_ - bump_) != lines_)
      return 1;
    if (young_.size() > young_cap_)
      return 2;
    std::vector<uint8_t> seen(slots_.size(), 0);
    for (auto& r : young_) {
      if (seen[r.slot]++ || slots_[r.slot].where != S_YOUNG)
        return 3;
    }
    for (uint32_t i = 0; i < n_old_; ++i) {
      if (seen[old_[i].slot]++ || slots_[old_[i].slot].where != S_OLD || slots_[old_[i].slot].old_idx != i)
        return 4;
    }
    std::vector<uint8_t> held(slots_.size(), 0);
    uint32_t valid = 0;
    for (uint32_t s = 0; s < cfg_.tag_sets; ++s) {
      for (uint32_t w = 0; w < cfg_.tag_ways; ++w) {
        auto& x = way(s, w);
        if (!x.valid)
          continue;
        ++valid;
        if (held[x.slot]++ || !seen[x.slot])
          return 5;
      }
    }
    for (auto& r : young_) {
      if (!way(r.set, r.way).valid || way(r.set, r.way).slot != r.slot)
        return 6;
    }
    for (uint32_t i = 0; i < n_old_; ++i) {
      auto& r = old_[i];
      if (!way(r.set, r.way).valid || way(r.set, r.way).slot != r.slot)
        return 7;
    }
    if (valid != young_.size() + n_old_)
      return 8;
    return 0;
  }

private:
  enum Where { S_UNUSED, S_FREE, S_YOUNG, S_OLD, S_BORROWED };

  struct Way {
    bool valid = false;
    uint64_t line = 0;
    uint32_t slot = NONE;
    uint64_t vmask = 0;
    uint64_t born = ~uint64_t(0);  // cycle of the allocation write
  };

  struct Slot {
    Where where = S_UNUSED;
    bool open = false;
    uint64_t entry_line = 0;
    uint64_t line = 0;
    uint32_t old_idx = 0;
  };

  struct Entry {
    uint32_t slot, set, way;
    uint32_t sectors;
    uint32_t filled = 0;
    uint64_t valid_at_miss;
    uint32_t seq;
    bool cacheable;
  };

  struct Overlay {
    uint32_t set, way;
    Way old;
  };

  struct Draw {
    bool active = false;
    uint64_t line = 0;
    uint64_t cycle = 0;
    uint32_t r = 0;
    uint32_t set = 0, way = 0;
  };

  Way& way(uint32_t set, uint32_t w) { return ways_[size_t(set) * cfg_.tag_ways + w]; }
  const Way& way(uint32_t set, uint32_t w) const { return ways_[size_t(set) * cfg_.tag_ways + w]; }

  int find(uint32_t set, uint64_t line) const {
    for (uint32_t i = 0; i < cfg_.tag_ways; ++i) {
      auto& x = way(set, i);
      if (x.valid && x.line == line)
        return int(i);
    }
    return -1;
  }

  // Written this cycle: not yet visible to lookups.
  bool fresh(uint32_t set, uint32_t w) const { return way(set, w).born == now_; }

  const Entry& entry_of(uint32_t slot) const {
    auto it = entries_.find(slots_[slot].entry_line);
    if (!slots_[slot].open || it == entries_.end())
      std::abort();
    return it->second;
  }
  Entry& entry_of(uint32_t slot) {
    return const_cast<Entry&>(static_cast<const L1SlotDir*>(this)->entry_of(slot));
  }

  Alloc& wait(Alloc& a, Wait w) {
    a.ok = false;
    a.wait = w;
    return a;
  }

  uint32_t take_free(Kind& kind) {
    if (!free_.empty()) {
      uint32_t s = free_.front();
      free_.pop_front();
      kind = K_FREE;
      return s;
    }
    kind = K_BUMP;
    return bump_++;
  }

  void open(uint64_t line, uint32_t slot, uint32_t set, uint32_t w, uint32_t sectors,
            uint64_t valid_at_miss, uint32_t seq, bool cacheable) {
    if (sectors == 0)
      return;
    entries_[line] = Entry{slot, set, w, sectors, 0, valid_at_miss, seq, cacheable};
    slots_[slot].open = true;
    slots_[slot].entry_line = line;
  }

  void install(uint64_t line, const Alloc& a, uint32_t sectors, uint32_t seq) {
    way(a.set, a.way) = Way{true, line, a.slot, 0, now_};
    slots_[a.slot].line = line;
    open(line, a.slot, a.set, a.way, sectors, 0, seq, true);
  }

  Alloc start_draw(Alloc& a, uint64_t line) {
    if (young_.size() >= young_cap_)
      return wait(a, W_YOUNG_FULL);
    if (n_old_ == 0) {
      want_migrate_ = true;
      return wait(a, W_OLD_EMPTY);
    }
    draw_.active = true;
    draw_.line = line;
    draw_.cycle = now_;
    draw_.r = uint32_t((uint64_t(lfsr_.next()) * n_old_) >> 16);
    last_r_ = draw_.r;
    last_n_ = n_old_;
    draw_.set = a.set;
    draw_.way = a.way;
    return wait(a, W_DRAW_PENDING);
  }

  Alloc complete_draw(Alloc& a, uint64_t line, uint32_t seq, uint32_t sectors) {
    Ref v = old_[draw_.r];
    a.way = draw_.way;
    draw_.active = false;
    if (in_flight(v.slot)) {
      Alloc r = start_draw(a, line);
      if (r.wait == W_DRAW_PENDING)
        r.wait = W_VICTIM_IN_FLIGHT;
      return r;
    }
    auto& vx = way(v.set, v.way);
    a.ok = true;
    a.kind = K_AGE;
    a.slot = v.slot;
    a.victim = v;
    a.victim_line = vx.line;
    overlay_.push_back(Overlay{v.set, v.way, vx});
    vx = Way{};
    auto h = young_.front();
    young_.pop_front();
    old_[draw_.r] = h;
    slots_[h.slot].where = S_OLD;
    slots_[h.slot].old_idx = draw_.r;
    young_.push_back(Ref{v.slot, a.set, a.way});
    slots_[v.slot].where = S_YOUNG;
    swapped_ = true;
    install(line, a, sectors, seq);
    return a;
  }

  L1SlotCfg cfg_{};
  uint32_t lines_ = 0, protect_ = 0, young_cap_ = 0;
  std::vector<Way> ways_;
  std::vector<Slot> slots_;
  std::vector<Ref> old_;
  uint32_t n_old_ = 0;
  std::deque<Ref> young_;
  std::deque<uint32_t> free_;
  uint32_t bump_ = 0, borrowed_ = 0;
  std::unordered_map<uint64_t, Entry> entries_;
  std::vector<Overlay> overlay_;
  Draw draw_;
  bool swapped_ = false, want_migrate_ = false;
  L1Lfsr lfsr_;
  uint64_t now_ = 0;
  uint32_t last_r_ = 0, last_n_ = 0;
};

// Per tag bank and cycle: one read (the lookup) and one write. Write priority:
// invalidate-all, the bank's update queue (arrival order), an allocation.
// Updates enter all-or-nothing in the order lookup, victim, fill.
class L1TagPortRef {
public:
  enum Kind { SET, CLEAR };
  enum Src { LOOKUP, VICTIM, FILL };
  enum Grant { G_NONE, G_INVAL, G_UPDATE, G_ALLOC };

  struct Update {
    uint32_t row, way;
    uint64_t mask;
    Kind kind;
    Src src;
  };

  L1TagPortRef(uint32_t banks, uint32_t depth) : banks_(banks), q_(banks), alloc_(banks), depth_(depth) {}

  bool can_push(uint32_t bank, uint32_t n) const { return q_[bank].size() + n <= depth_; }

  bool push_update(uint32_t bank, uint32_t row, uint32_t way, uint64_t mask, Kind kind, Src src) {
    if (!can_push(bank, 1))
      return false;
    q_[bank].push_back(Update{row, way, mask, kind, src});
    return true;
  }

  void request_alloc(uint32_t bank, uint32_t row, uint32_t way) {
    alloc_[bank] = Alloc{true, row, way};
  }

  void request_invalidate() { inval_ = true; }

  // Masks queued for {row, way} applied over the RAM's valid bits.
  uint64_t pending(uint32_t bank, uint32_t row, uint32_t way, uint64_t ram) const {
    for (auto& u : q_[bank]) {
      if (u.row == row && u.way == way)
        ram = (u.kind == SET) ? (ram | u.mask) : (ram & ~u.mask);
    }
    return ram;
  }

  // One write per bank; an allocation not granted holds (the tag stage waits a cycle).
  std::vector<Grant> tick(std::vector<Update>* written = nullptr) {
    std::vector<Grant> g(banks_, G_NONE);
    for (uint32_t b = 0; b < banks_; ++b) {
      if (inval_) {
        g[b] = G_INVAL;
      } else if (!q_[b].empty()) {
        if (written)
          written->push_back(q_[b].front());
        q_[b].pop_front();
        g[b] = G_UPDATE;
      } else if (alloc_[b].valid) {
        g[b] = G_ALLOC;
        alloc_[b].valid = false;
      }
    }
    inval_ = false;
    return g;
  }

  bool alloc_held(uint32_t bank) const { return alloc_[bank].valid; }
  uint32_t queued(uint32_t bank) const { return uint32_t(q_[bank].size()); }

private:
  struct Alloc {
    bool valid = false;
    uint32_t row = 0, way = 0;
  };
  uint32_t banks_;
  std::vector<std::deque<Update>> q_;
  std::vector<Alloc> alloc_;
  uint32_t depth_;
  bool inval_ = false;
};

} // namespace l1_slot_ref
} // namespace vortex
